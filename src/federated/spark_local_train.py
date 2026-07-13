"""
Rank-isolated FL local SGD kernel for the Spark + TorchDistributor path.

Design intent:
    TorchDistributor is conventionally used to run a single PyTorch model under
    DistributedDataParallel (DDP) with `dist.all_reduce` synchronisation. Here
    we use it as a Spark-aware torch process spawner *without* DDP - each rank
    owns a distinct agent, trains an isolated copy of the global model on its
    own Parquet shard, and emits a parameter delta. Aggregation happens on the
    driver via Multi-Krum and trimmed mean, not via gradient
    averaging. That is the FL signature; Spark only provides the parallelism
    substrate.

This module deliberately contains *no* SparkSession references. It must be
serialisable by TorchDistributor and importable inside a torch worker process
where the parent SparkSession is not available. SparkSession-aware code lives
in `src/simulation/spark_fl_runner.py`.

Per-tier `local_steps` semantics:
    - building : SGD steps performed locally before computing Δθ (FL local).
    - district : optional post-aggregation distillation steps on G_d (0 disables).
    - city     : optional post-aggregation distillation steps on W*  (0 disables).

The rank kernel here only handles the *building* path; district and city
post-aggregation steps run on the driver after Krum (see spark_fl_runner.py).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.schema import INPUT_DIM
from src.federated.dp import clip_and_noise
from src.federated.payload import deserialize_update, serialize_update
from src.federated.sparsification import topk_sparsify_with_masks
from src.models.city_lstm import CityLSTM, tier_size


# --------------------------------------------------------------------------- #
# Manifest dataclass: the per-rank work item the driver assigns.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentShard:
    """One row of the Spark FL manifest: rank ↔ agent assignment."""
    agent_id: str
    district_id: str
    parquet_path: str       # Parquet partition holding this household's rows
    n_samples: int          # advisory; actual count read from Parquet


# --------------------------------------------------------------------------- #
# Hparams: deliberately a plain dict so it survives the TorchDistributor
# pickle/unpickle round trip without import-time torch deps.
# --------------------------------------------------------------------------- #
@dataclass
class LocalTrainHparams:
    pred_len: int = 4
    seq_len: int = 24
    tier: str = "building"
    local_steps_building: int = 5
    lr: float = 1e-3
    batch_size: int = 8
    dp_sigma: float = 0.0
    dp_clip_C: float = 1.0
    topk_ratio: float = 0.1
    byzantine_fraction: float = 0.0     # injects attackers; defender story
    byzantine_attack: str = "scale"     # "scale" | "flip" | "noise"
    seed: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


# --------------------------------------------------------------------------- #
# Helpers - pure torch / pandas, no Spark.
# --------------------------------------------------------------------------- #
def _load_parquet_features(parquet_path: str) -> np.ndarray:
    """Read the household's Parquet partition into an (N, INPUT_DIM) array.

    Falls back gracefully if optional extended columns are absent. We use
    pandas + pyarrow rather than spark.read because this runs *inside* a
    torch rank that does not own a SparkSession.
    """
    from src.data.schema import FEATURE_COLUMNS
    df = pd.read_parquet(parquet_path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Parquet at {parquet_path} missing required columns: {missing}"
        )
    # Spark ETL writes the timestamp column as `Timestamp` (capital T).
    for ts_col in ("Timestamp", "timestamp"):
        if ts_col in df.columns:
            df = df.sort_values(ts_col)
            break
    feats = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    if feats.shape[1] != INPUT_DIM:
        raise ValueError(
            f"Feature dim {feats.shape[1]} != schema INPUT_DIM {INPUT_DIM}."
        )
    return feats


def _make_windows(feats: np.ndarray, seq_len: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide a (seq_len, pred_len) window over the row-major feature array.

    Returns (X, y) with shapes (B, seq_len, INPUT_DIM) and (B, pred_len).
    Target is the kWh column (index 0) of the next pred_len rows.
    """
    n = feats.shape[0]
    if n < seq_len + pred_len:
        return np.empty((0, seq_len, feats.shape[1]), dtype=np.float32), \
               np.empty((0, pred_len), dtype=np.float32)
    last_start = n - seq_len - pred_len
    X = np.stack([feats[i : i + seq_len] for i in range(last_start + 1)])
    y = np.stack([feats[i + seq_len : i + seq_len + pred_len, 0]
                  for i in range(last_start + 1)])
    return X.astype(np.float32), y.astype(np.float32)


def _apply_byzantine_attack(
    delta: dict[str, torch.Tensor],
    attack: str,
    rng: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Return a corrupted Δθ simulating an attacker.

    Honest attack semantics so the Krum defender story is real:
        scale : multiply by 10 (large-magnitude push, classic gradient attack)
        flip  : negate every parameter (pushes away from honest direction)
        noise : add N(0, 1.0) noise (random direction, large magnitude)

    The defender (Multi-Krum) should reject these consistently; that's the
    plot you'll show in the report.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in delta.items():
        if attack == "flip":
            out[k] = -v
        elif attack == "scale":
            out[k] = v * 10.0
        elif attack == "noise":
            noise = torch.empty_like(v).normal_(0.0, 1.0, generator=rng)
            out[k] = v + noise
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# The rank entrypoint. Called once per round per agent.
# --------------------------------------------------------------------------- #
def local_train_one_agent(
    shard: AgentShard,
    w_global_path: str,
    round_id: int,
    output_root: str,
    hparams: LocalTrainHparams,
    is_byzantine: bool = False,
) -> dict[str, Any]:
    """Run K SGD steps on `shard`'s Parquet partition starting from W*, then
    compute Δθ = θ_local − W*, apply DP + TopK, persist to
    `output_root/round=<R>/agent=<id>/delta.pt`, and return a metrics row.

    This function is the contract between TorchDistributor and the driver.
    It must be:
      * Pure (no SparkSession, no global state)
      * Idempotent (rerunning produces the same files)
      * Self-contained (all imports inside / at module top)

    Returns a metrics dict suitable for `spark.createDataFrame([row, ...])`
    on the driver. No torch tensors leave this function - only paths +
    floats + ints + strings.
    """
    torch.manual_seed(hparams.seed + hash(shard.agent_id) % (2**31))
    rng = torch.Generator().manual_seed(hparams.seed + round_id)

    # 1. Materialise the local model and load W*.
    sz = tier_size(hparams.tier)
    model = CityLSTM(pred_len=hparams.pred_len, input_dim=INPUT_DIM, **sz)
    w_global = torch.load(w_global_path, map_location="cpu", weights_only=True)
    model.load_state_dict(w_global, strict=False)

    # 2. Snapshot baseline so we can compute Δθ cleanly later.
    baseline = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 3. Local data → windowed dataset.
    feats = _load_parquet_features(shard.parquet_path)
    X, y = _make_windows(feats, hparams.seq_len, hparams.pred_len)
    n_windows = X.shape[0]

    metrics: dict[str, Any] = {
        "agent_id": shard.agent_id,
        "district_id": shard.district_id,
        "round_id": int(round_id),
        "tier": hparams.tier,
        "n_samples": int(n_windows),
        "is_byzantine": bool(is_byzantine),
        "byzantine_attack": hparams.byzantine_attack if is_byzantine else "",
        "local_steps_planned": int(hparams.local_steps_building),
        "local_steps_executed": 0,
        "loss_first": float("nan"),
        "loss_last": float("nan"),
        "delta_l2_pre_dp": 0.0,
        "delta_l2_post_dp": 0.0,
        "topk_kept_ratio": 0.0,
        "delta_path": "",
        "status": "ok",
    }

    if n_windows == 0:
        metrics["status"] = "no_data"
        return metrics

    # 4. Local SGD for K steps. Mini-batches sampled with replacement so
    #    K is honoured even when n_windows < batch_size.
    optim = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    crit = torch.nn.SmoothL1Loss()
    model.train()
    losses: list[float] = []
    rng_np = np.random.default_rng(hparams.seed + round_id + (hash(shard.agent_id) & 0xFFFF))
    for step in range(hparams.local_steps_building):
        idx = rng_np.integers(0, n_windows, size=min(hparams.batch_size, n_windows))
        xb = torch.from_numpy(X[idx])
        yb = torch.from_numpy(y[idx])
        optim.zero_grad()
        pred = model(xb)
        loss = crit(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        losses.append(float(loss.item()))
    model.eval()

    metrics["local_steps_executed"] = len(losses)
    if losses:
        metrics["loss_first"] = losses[0]
        metrics["loss_last"] = losses[-1]

    # 5. Δθ = θ_local − W*  (CPU tensors).
    local_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    delta = {k: (local_state[k] - baseline[k]).cpu() for k in local_state.keys()}

    pre_dp_norm = float(
        torch.sqrt(sum((v.float() ** 2).sum() for v in delta.values())).item()
    )
    metrics["delta_l2_pre_dp"] = pre_dp_norm

    # 6. Optional Byzantine attack injection (after honest training, before DP).
    if is_byzantine and hparams.byzantine_fraction > 0.0:
        delta = _apply_byzantine_attack(delta, hparams.byzantine_attack, rng)

    # 7. DP clip + Gaussian noise (when sigma > 0).
    if hparams.dp_sigma > 0.0:
        delta, _ = clip_and_noise(
            delta, sensitivity=hparams.dp_clip_C, sigma=hparams.dp_sigma
        )

    # 8. TopK sparsification (uplink budget) with per-agent error feedback.
    residual_path = Path(output_root) / "_compression_residuals" / f"{shard.agent_id}.pt"
    if hparams.dp_sigma == 0.0 and residual_path.exists():
        residual, _residual_masks, _metadata = deserialize_update(
            residual_path.read_bytes()
        )
        delta = {
            key: value + residual.get(key, torch.zeros_like(value))
            for key, value in delta.items()
        }
    sparse, sparse_masks = topk_sparsify_with_masks(
        delta, keep_ratio=hparams.topk_ratio
    )
    post_dp_norm = float(
        torch.sqrt(sum((v.float() ** 2).sum() for v in sparse.values())).item()
    )
    metrics["delta_l2_post_dp"] = post_dp_norm

    total_params = sum(v.numel() for v in delta.values()) or 1
    transmitted_params = sum(int(mask.sum().item()) for mask in sparse_masks.values())
    metrics["topk_kept_ratio"] = transmitted_params / total_params

    # 9. Persist Δθ to the per-round Parquet-shaped tree.
    out_dir = Path(output_root) / f"round={round_id:04d}" / f"agent={shard.agent_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    delta_path = out_dir / "delta.pt"
    delta_path.write_bytes(
        serialize_update(
            sparse,
            masks=sparse_masks,
            model_version=f"spark-city-round-{round_id}",
        )
    )
    if hparams.dp_sigma == 0.0:
        next_residual = {key: delta[key] - sparse[key] for key in delta}
        residual_path.parent.mkdir(parents=True, exist_ok=True)
        residual_tmp = residual_path.with_suffix(".tmp")
        residual_tmp.write_bytes(
            serialize_update(
                next_residual,
                model_version=f"spark-compression-after-round-{round_id}",
            )
        )
        os.replace(residual_tmp, residual_path)
    metrics["delta_path"] = str(delta_path)

    logging.info(
        "[rank-agent=%s] round=%d steps=%d loss %.4f → %.4f ΔθL2 %.3f → %.3f%s",
        shard.agent_id,
        round_id,
        len(losses),
        metrics["loss_first"],
        metrics["loss_last"],
        pre_dp_norm,
        post_dp_norm,
        " [BYZANTINE]" if is_byzantine else "",
    )
    return metrics


# --------------------------------------------------------------------------- #
# TorchDistributor entrypoint. Receives (rank, world_size) implicitly via
# env vars; reads the manifest from disk; dispatches to local_train_one_agent.
# --------------------------------------------------------------------------- #
def torch_distributor_entry(
    manifest_path: str,
    w_global_path: str,
    round_id: int,
    output_root: str,
    hparams_json: str,
    byzantine_ranks_csv: str = "",
) -> None:
    """Top-level function passed to TorchDistributor.run().

    TorchDistributor sets RANK / WORLD_SIZE env vars before invoking. We do not
    initialise a torch.distributed process group - ranks are isolated by
    design (FL semantics, not DDP).

    Idempotent: rerunning round R will overwrite the same delta.pt files.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    manifest = pd.read_parquet(manifest_path)
    if rank >= len(manifest):
        logging.warning(
            "rank %d ≥ manifest size %d - skipping (idle rank).",
            rank, len(manifest),
        )
        return

    row = manifest.iloc[rank]
    shard = AgentShard(
        agent_id=str(row["agent_id"]),
        district_id=str(row["district_id"]),
        parquet_path=str(row["parquet_path"]),
        n_samples=int(row.get("n_samples", 0)),
    )

    hparams = LocalTrainHparams(**json.loads(hparams_json))
    byzantine_ranks = (
        set(int(x) for x in byzantine_ranks_csv.split(",") if x.strip())
        if byzantine_ranks_csv else set()
    )
    is_byzantine = rank in byzantine_ranks

    metrics = local_train_one_agent(
        shard=shard,
        w_global_path=w_global_path,
        round_id=round_id,
        output_root=output_root,
        hparams=hparams,
        is_byzantine=is_byzantine,
    )

    # Persist per-rank metrics row so the driver can collect with a single
    # spark.read.parquet glob. Each rank writes its own file - no rank-to-rank
    # contention, no driver-side serialisation hazard.
    metrics_dir = Path(output_root) / f"round={round_id:04d}" / "_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_row_path = metrics_dir / f"rank_{rank:04d}.json"
    metrics_row_path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")

    logging.info(
        "[rank=%d/%d agent=%s] round=%d done; metrics=%s",
        rank, world_size, shard.agent_id, round_id, str(metrics_row_path),
    )
