"""
City-tier teacher pretraining via TorchDistributor + native PyTorch DDP.

This is the *centralized big-data training* counterpart to the federated path
in `spark_fl_runner.py`. It uses TorchDistributor in its conventional mode -
spawning N torch processes that cooperate via `torch.distributed.all_reduce`
under DistributedDataParallel - to train a single CityLSTM teacher model on
the dedicated Spark-FL Parquet store.

Why both this and the FL runner?
    The FL runner uses TorchDistributor in rank-isolated mode (no all_reduce)
    because that matches federated-learning semantics. This module uses
    TorchDistributor in conventional DDP mode for centralized teacher
    pretraining. The resulting teacher can warm-start federated rounds through
    `--pretrained_teacher` on the CLI.

The DDP backend is **gloo** (CPU). NCCL would require GPU; we keep this CPU-
only so a laptop demo (`local[*]`) works end-to-end.

Design constraints:
  * DDP all_reduce is the textbook distributed-SGD primitive. We use it
    correctly: per-rank gradient computation on a sharded Parquet slice,
    synchronous all_reduce-mean, single optimizer step.
  * Sharding is by household (`Household_ID=*` Parquet partitions). Each rank
    owns a disjoint set; no replication, no contention.
  * Output bundle is identical in shape to the curriculum-trained teacher, so
    downstream consumers (`evaluator.py`, FL warm-start) treat both paths the
    same.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn

from src.data.publication import resolve_current_dataset
from src.data.schema import FEATURE_COLUMNS, INPUT_DIM
from src.federated.spark_local_train import _make_windows  # window helper reused
from src.models.city_lstm import CityLSTM, tier_size


@dataclass
class DDPTrainConfig:
    parquet_root: str = "data/processed_parquet_spark_fl"
    output_dir: str = "src/models/stored/spark_ddp_teacher"
    tier: str = "city"
    pred_len: int = 4
    seq_len: int = 24
    epochs: int = 5
    batch_size: int = 32
    lr: float = 5e-4
    world_size: int = 2
    seed: int = 0
    # Spark MLlib KMeans cohort manifest. When set, teacher trains on the
    # cluster-stratified household slice so it sees representative
    # consumption profiles (good generalization). When None, falls back to
    # all households in the parquet root.
    manifest_path: str | None = None


def _shard_household_dirs(
    parquet_root: str,
    rank: int,
    world_size: int,
    manifest_path: str | None = None,
) -> list[str]:
    """Partition the household directories deterministically across ranks.

    When `manifest_path` is set, restricts the cohort to the Spark MLlib
    KMeans-stratified slice - the teacher then trains on representative
    consumption profiles instead of whatever happened to land in the
    parquet root. This is what makes the trained model generalize across
    heterogeneous households at FL warm-start time.
    """
    root = resolve_current_dataset(parquet_root)
    dirs_all = sorted(
        {p.resolve() for p in root.rglob("Household_ID=*") if p.is_dir()},
        key=lambda p: p.name,
    )
    if not dirs_all:
        raise FileNotFoundError(
            f"No household partitions found under {parquet_root}. "
            "Run `python -m src.cli etl-spark-fl` first."
        )
    if manifest_path:
        from src.data.representative_sampling import load_manifest
        selected = set(load_manifest(manifest_path))
        dirs = [d for d in dirs_all if d.name.split("=", 1)[1] in selected]
        if not dirs:
            raise ValueError(
                f"Manifest at {manifest_path} intersected with parquet root "
                f"{parquet_root} produced 0 households."
            )
    else:
        dirs = dirs_all
    return [str(d) for i, d in enumerate(dirs) if i % world_size == rank]


def _load_shard_windows(
    shard_paths: list[str], seq_len: int, pred_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate windowed (X, y) tensors across this rank's household slice."""
    Xs, ys = [], []
    for p in shard_paths:
        df = pd.read_parquet(p)
        for ts_col in ("Timestamp", "timestamp"):
            if ts_col in df.columns:
                df = df.sort_values(ts_col)
                break
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            logging.warning("[ddp] %s missing %s; skipping.", p, missing)
            continue
        feats = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        X, y = _make_windows(feats, seq_len, pred_len)
        if X.size:
            Xs.append(X)
            ys.append(y)
    if not Xs:
        return torch.empty(0, seq_len, INPUT_DIM), torch.empty(0, pred_len)
    return torch.from_numpy(np.concatenate(Xs)), torch.from_numpy(np.concatenate(ys))


def _ddp_train_entry(cfg_json: str) -> None:
    """TorchDistributor entrypoint. Sets up DDP, trains, persists on rank 0."""
    cfg = DDPTrainConfig(**json.loads(cfg_json))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", str(cfg.world_size)))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    # Windows: PyTorch wheels are built without libuv. Force the legacy
    # TCPStore backend so init_process_group succeeds. Harmless elsewhere.
    os.environ.setdefault("USE_LIBUV", "0")

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(cfg.seed + rank)

        sz = tier_size(cfg.tier)
        model = CityLSTM(pred_len=cfg.pred_len, input_dim=INPUT_DIM, **sz)
        ddp_model = nn.parallel.DistributedDataParallel(model)
        opt = torch.optim.Adam(ddp_model.parameters(), lr=cfg.lr)
        crit = nn.SmoothL1Loss()

        shard = _shard_household_dirs(
            cfg.parquet_root, rank, world_size, manifest_path=cfg.manifest_path
        )
        X, y = _load_shard_windows(shard, cfg.seq_len, cfg.pred_len)
        n = X.shape[0]
        logging.info(
            "[ddp rank=%d/%d] shard=%d households windows=%d",
            rank, world_size, len(shard), n,
        )

        if n == 0:
            logging.warning("[ddp rank=%d] empty shard; idle through DDP barrier.", rank)
            dist.barrier()
            return

        rng = np.random.default_rng(cfg.seed + rank)
        steps_per_epoch = max(1, n // cfg.batch_size)
        for epoch in range(cfg.epochs):
            ddp_model.train()
            ep_loss = 0.0
            for _ in range(steps_per_epoch):
                idx = rng.integers(0, n, size=min(cfg.batch_size, n))
                xb, yb = X[idx], y[idx]
                opt.zero_grad()
                pred = ddp_model(xb)
                loss = crit(pred, yb)
                loss.backward()
                # all_reduce of gradients happens inside DDP backward.
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                opt.step()
                ep_loss += float(loss.item())
            ep_loss_mean = ep_loss / max(1, steps_per_epoch)

            # Synchronously aggregate the per-rank loss for honest reporting.
            t = torch.tensor([ep_loss_mean, 1.0])
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            global_loss = float((t[0] / t[1]).item())
            if rank == 0:
                logging.info(
                    "[ddp epoch=%d] global_train_loss=%.4f (ranks contributing=%d)",
                    epoch, global_loss, int(t[1].item()),
                )

        # rank-0 persists the final model bundle. Other ranks just exit.
        if rank == 0:
            out = Path(cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            # Unwrap DDP - save the underlying model's state_dict so the
            # FL warm-start path can load it without DDP key mangling.
            torch.save(ddp_model.module.state_dict(), str(out / "model.pth"))
            (out / "ddp_config.json").write_text(
                json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8"
            )
            logging.info("[ddp rank=0] wrote teacher bundle to %s", out)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def train_city_teacher_ddp(cfg: DDPTrainConfig) -> Path:
    """Driver entrypoint. Spawns world_size torch ranks via TorchDistributor."""
    from pyspark.ml.torch.distributor import TorchDistributor
    from pyspark.sql import SparkSession

    if not resolve_current_dataset(cfg.parquet_root).exists():
        raise FileNotFoundError(
            f"Spark-FL parquet root missing: {cfg.parquet_root}\n"
            "Run `python -m src.cli etl-spark-fl` first."
        )

    spark = (
        SparkSession.builder.appName("NeuroGrid_DDPTeacher")
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        t0 = time.monotonic()
        # Windows-specific guard (see spark_fl_runner.py for rationale).
        os.environ.setdefault("USE_LIBUV", "0")
        distributor = TorchDistributor(
            num_processes=int(cfg.world_size),
            local_mode=True,
            use_gpu=False,
        )
        distributor.run(_ddp_train_entry, json.dumps(asdict(cfg)))
        wall = time.monotonic() - t0
        logging.info("[ddp] training complete in %.1fs.", wall)
        out_dir = Path(cfg.output_dir)
        (out_dir / "ddp_run_summary.json").write_text(
            json.dumps({
                "wallclock_s": wall,
                "world_size": cfg.world_size,
                "epochs": cfg.epochs,
                "parquet_root": cfg.parquet_root,
            }, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return out_dir
    finally:
        spark.stop()
