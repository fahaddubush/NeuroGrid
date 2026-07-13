"""
Spark + TorchDistributor federated-learning round driver.

This is the batch counterpart to the online gRPC topology in `runner.py`.
The implementations remain separate because Spark rounds and continuously
running tier services have different scheduling, recovery, and state lifecycles.

Topology
--------
    driver
      │  ▼  TorchDistributor.run(local_train_fn, num_processes=N, local_mode=True)
      │     │
      │     ├── rank 0  →  agent A0  →  delta_0.pt
      │     ├── rank 1  →  agent A1  →  delta_1.pt
      │     ├── ...
      │     └── rank N  →  agent AN  →  delta_N.pt
      │
      ├── per-district groupBy + Krum + trimmed_mean        (Spark groupBy)
      ├── city-level FedAvg (sample-weighted)               (driver)
      └── persist round audit + W*                          (Parquet writes)

Design notes:
  1. TorchDistributor is used in two distinct, honest modes across the project:
        (a) here - as a Spark-aware torch process spawner WITHOUT all_reduce,
            providing rank-isolated local SGD that matches FL semantics;
        (b) in `train_spark_ddp` - as native DDP for centralized teacher
            pretraining with `dist.all_reduce`.
  2. Spark operates at the *cohort* level (per-district groupBy + Krum
     audit), not the *tensor* level (model weights stay in torch).
  3. Per-rank Δθ writes to Parquet *before* aggregation give us automatic
     fault tolerance: any failed round can resume from disk via
     `--resume_from_round`.
  4. The same Krum+trimmed-mean defender protects both gRPC and Spark paths
     identically, so the resilience-vs-Byzantine plot is transport-independent.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import socket
import subprocess
import time
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Ensure workers use the same Python interpreter as the driver.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from src.data.publication import resolve_current_dataset
from src.data.schema import INPUT_DIM
from src.federated.aggregation import masked_trimmed_mean, masked_weighted_fedavg
from src.federated.payload import deserialize_update
from src.federated.convergence import ConvergenceMonitor
from src.federated.krum import multi_krum_select
from src.federated.spark_local_train import (
    AgentShard,
    LocalTrainHparams,
    torch_distributor_entry,
)
from src.models.city_lstm import CityLSTM, tier_size

# Required parquet input dir (separate from the gRPC path).
SPARK_FL_PARQUET_ROOT = "data/processed_parquet_spark_fl"
DEFAULT_RUN_ROOT = "artifacts/spark_fl/runs"


# --------------------------------------------------------------------------- #
@dataclass
class SparkFLConfig:
    """Driver-side configuration for a Spark FL run.

    Kept separate from `LocalTrainHparams` because some fields here only matter
    on the driver (e.g. num_districts, rounds, Spark master URL) - the rank
    kernel must stay agnostic of cluster shape.
    """
    num_agents: int = 8
    num_districts: int = 2
    rounds: int = 5
    f_byzantine: int = 1            # per-district Byzantine tolerance
    byzantine_fraction: float = 0.0 # fraction of agents that are attackers
    byzantine_attack: str = "scale" # "scale" | "flip" | "noise"
    clip_threshold: float = 1.0
    trim_ratio: float = 0.1
    max_relative_update: float = 0.25
    convergence_epsilon: float = 1e-3
    convergence_patience: int = 2
    spark_master: str = "local[*]"
    resume_from_round: int | None = None
    parquet_root: str = SPARK_FL_PARQUET_ROOT
    run_root: str = DEFAULT_RUN_ROOT
    # Spark MLlib KMeans-stratified sampling manifest. When provided, the
    # cohort is drawn from this manifest (cluster-balanced, generalizes
    # better) rather than alphabetical-first-N. Strongly recommended.
    manifest_path: str | None = None

    # Per-tier local-step budget (paper's tier compute model).
    local_steps_building: int = 5

    # Local SGD / model knobs forwarded to the rank kernel.
    pred_len: int = 4
    seq_len: int = 24
    tier: str = "building"
    lr: float = 1e-3
    batch_size: int = 8
    dp_sigma: float = 0.0
    dp_clip_C: float = 1.0
    topk_ratio: float = 0.1
    seed: int = 0


# --------------------------------------------------------------------------- #
# Manifest discovery: scan the dedicated Spark-FL Parquet root and assemble a
# (agent_id, district_id, parquet_path, n_samples) manifest as a small DataFrame.
# --------------------------------------------------------------------------- #
def _discover_agent_manifest(
    parquet_root: str,
    num_agents: int,
    num_districts: int,
    sampling_manifest_path: str | None = None,
) -> pd.DataFrame:
    """Build the rank-to-agent assignment from the Spark-FL Parquet store.

    Two cohort-selection paths:

    (1) **MLlib-stratified (preferred)** - when `sampling_manifest_path` is
        given, the cohort is drawn from the Spark MLlib KMeans manifest
        produced by `python -m src.cli sample-households`. Cluster IDs from
        that manifest drive district assignment so each district sees a
        balanced mix of consumption profiles. This is what makes the LSTM
        actually generalize across heterogeneous households.

    (2) **Alphabetical-first-N (fallback)** - when no manifest is given,
        falls back to sorted-by-name selection. Useful for smoke tests but
        explicitly NOT recommended for any results that go in the report.
    """
    root = resolve_current_dataset(parquet_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Spark-FL parquet root not found: {parquet_root}\n"
            f"Run `python -m src.cli etl-spark-fl` first."
        )
    # ETL partitions by (Weather_ID, Household_ID), producing nested dirs:
    #   <root>/Weather_ID=<X>/Household_ID=<Y>/*.parquet
    # We discover at the leaf household level regardless of depth.
    household_dirs = sorted(
        {p.resolve() for p in root.rglob("Household_ID=*") if p.is_dir()},
        key=lambda p: p.name,
    )
    name_to_dir: dict[str, Path] = {p.name.split("=", 1)[1]: p for p in household_dirs}

    if sampling_manifest_path:
        from src.data.representative_sampling import load_manifest_with_clusters
        manifest_pairs = load_manifest_with_clusters(sampling_manifest_path)
        # Honour the manifest's order so the cohort is reproducible across runs.
        # Filter to households actually present in the parquet root (guard
        # against ETL-vs-sampling drift).
        usable: list[tuple[str, int]] = [
            (hid, cluster) for hid, cluster in manifest_pairs if hid in name_to_dir
        ]
        if len(usable) < num_agents:
            raise ValueError(
                f"Manifest at {sampling_manifest_path} provides {len(usable)} "
                f"usable households (after intersecting with parquet root), "
                f"need {num_agents}. Re-run etl-spark-fl + sample-households "
                f"with more households, or reduce --num_agents."
            )
        chosen = usable[:num_agents]
        # **Cluster-balanced district assignment**: ordering agents in
        # cluster-grouped blocks (all c0 first, then all c1, ...) means the
        # subsequent rank % num_districts assignment naturally spreads each
        # cluster across all districts. Result: every district sees a mix of
        # consumption profiles, which is what the LSTM needs to generalize.
        from collections import defaultdict
        by_cluster: dict[int, list[str]] = defaultdict(list)
        for hid, cluster in chosen:
            by_cluster[cluster].append(hid)
        ordered: list[tuple[str, int]] = []
        for c in sorted(by_cluster.keys()):
            for hid in by_cluster[c]:
                ordered.append((hid, c))
        rows: list[dict[str, Any]] = []
        for rank_i, (hid, cluster) in enumerate(ordered):
            rows.append({
                "rank": rank_i,
                "agent_id": hid,
                "district_id": f"D{(rank_i % num_districts) + 1}",
                "cluster": int(cluster),
                "parquet_path": str(name_to_dir[hid]),
                "n_samples": 0,
                "selection_mode": "mllib_kmeans",
            })
        return pd.DataFrame(rows)

    # Fallback (no manifest) - alphabetical, NOT for report-grade results.
    if len(household_dirs) < num_agents:
        raise ValueError(
            f"Need {num_agents} households, found {len(household_dirs)} in "
            f"{parquet_root}. Re-run etl-spark-fl with more households or "
            f"reduce --num_agents."
        )
    logging.warning(
        "[spark-fl] No --manifest_path provided; falling back to alphabetical "
        "cohort. This will NOT generalize well; run `python -m src.cli "
        "sample-households` first for production runs."
    )
    chosen_dirs = household_dirs[:num_agents]
    rows = []
    for i, hh_dir in enumerate(chosen_dirs):
        agent_id = hh_dir.name.split("=", 1)[1]
        rows.append({
            "rank": i,
            "agent_id": agent_id,
            "district_id": f"D{(i % num_districts) + 1}",
            "cluster": -1,
            "parquet_path": str(hh_dir),
            "n_samples": 0,
            "selection_mode": "alphabetical_fallback",
        })
    return pd.DataFrame(rows)


def _select_byzantine_ranks(
    n: int, fraction: float, seed: int
) -> set[int]:
    if fraction <= 0.0:
        return set()
    n_byz = int(round(n * fraction))
    rng = random.Random(seed)
    return set(rng.sample(range(n), n_byz))


def _hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
def _initial_global_state(cfg: SparkFLConfig) -> dict[str, torch.Tensor]:
    """W*_0 - randomly-initialised CityLSTM at the configured tier."""
    torch.manual_seed(cfg.seed)
    sz = tier_size(cfg.tier)
    model = CityLSTM(pred_len=cfg.pred_len, input_dim=INPUT_DIM, **sz)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _load_round_metrics(round_dir: Path) -> pd.DataFrame:
    """Driver-side: gather per-rank metric JSONs into a single DataFrame.

    Each rank wrote `rank_NNNN.json`. We collect all of them and tolerate
    missing files (empty Parquet shards return early in the kernel).
    """
    metrics_dir = round_dir / "_metrics"
    if not metrics_dir.exists():
        return pd.DataFrame()
    rows = []
    for jp in sorted(metrics_dir.glob("rank_*.json")):
        try:
            rows.append(json.loads(jp.read_text(encoding="utf-8")))
        except Exception as e:
            logging.warning("failed to parse %s: %s", jp, e)
    return pd.DataFrame(rows)


def _load_deltas_for_district(
    round_dir: Path, agent_ids: list[str]
) -> list[dict[str, dict[str, torch.Tensor]]]:
    """Driver-side: torch.load every Δθ for the agents in one district."""
    out: list[dict[str, dict[str, torch.Tensor]]] = []
    for aid in agent_ids:
        path = round_dir / f"agent={aid}" / "delta.pt"
        if path.exists():
            state, masks, _metadata = deserialize_update(path.read_bytes())
            out.append({"state": state, "masks": masks})
    return out


# --------------------------------------------------------------------------- #
# Per-district Byzantine aggregation. Spark groupBy hosts the *control plane*;
# the per-group reduction below is a plain Python+torch function dispatched on
# the driver. Tensor math stays in torch - Spark DataFrames are not the right
# shape for million-coordinate parameter vectors.
# --------------------------------------------------------------------------- #
def _aggregate_district(
    deltas: list[dict[str, dict[str, torch.Tensor]]],
    n_samples: list[int],
    f_byzantine: int,
    clip_threshold: float,
    trim_ratio: float,
) -> tuple[
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor] | None,
    dict[str, Any],
]:
    """One district's Byzantine pipeline, identical math to spark_byzantine.py.

    Returns (G_d, summary). G_d is None when no honest survivors remain.
    """
    from src.federated.clipping import clip_state_dict, state_dict_l2_norm

    if not deltas:
        return None, None, {
            "n_total": 0,
            "n_accepted": 0,
            "n_total_samples": 0,
            "score_mean": 0.0,
            "score_std": 0.0,
        }

    clipped: list[dict[str, torch.Tensor]] = []
    audit_norms: list[float] = []
    masks: list[dict[str, torch.Tensor]] = []
    for item in deltas:
        c, original = clip_state_dict(item["state"], clip_threshold)
        clipped.append(c)
        masks.append(item["masks"])
        audit_norms.append(float(original))

    if f_byzantine == 0:
        accepted_idx = list(range(len(clipped)))
        scores = torch.zeros(len(clipped))
    else:
        accepted_idx, _rejected_idx, scores = multi_krum_select(clipped, f=f_byzantine)
    survivors = [clipped[i] for i in accepted_idx]
    survivor_masks = [masks[i] for i in accepted_idx]

    if not survivors:
        return None, None, {
            "n_total": len(deltas),
            "n_accepted": 0,
            "n_total_samples": int(sum(n_samples)),
            "score_mean": float(scores.float().mean().item()),
            "score_std": float(scores.float().std().item()) if scores.numel() > 1 else 0.0,
        }

    fa, fa_mask = masked_trimmed_mean(
        survivors, survivor_masks, trim_ratio=trim_ratio
    )

    summary = {
        "n_total": len(deltas),
        "n_accepted": len(survivors),
        "n_total_samples": int(sum(n_samples)),
        "score_mean": float(scores.float().mean().item()),
        "score_std": float(scores.float().std().item()) if scores.numel() > 1 else 0.0,
        "accepted_idx": list(accepted_idx),
    }
    return fa, fa_mask, summary


# --------------------------------------------------------------------------- #
# Round runner: invokes TorchDistributor, then aggregates.
# --------------------------------------------------------------------------- #
def _run_one_round(
    spark: Any,
    cfg: SparkFLConfig,
    round_id: int,
    manifest_df: pd.DataFrame,
    w_global: dict[str, torch.Tensor],
    byzantine_ranks: set[int],
    run_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    round_t0 = time.monotonic()
    deltas_root = run_dir / "deltas"
    w_root = run_dir / "W_global"
    manifest_path = run_dir / "manifest.parquet"
    w_round_path = w_root / f"round={round_id:04d}" / "W.pt"
    w_round_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Persist the inputs the rank kernel will read.
    torch.save(w_global, str(w_round_path))
    manifest_df.to_parquet(str(manifest_path), index=False)

    hparams = LocalTrainHparams(
        pred_len=cfg.pred_len,
        seq_len=cfg.seq_len,
        tier=cfg.tier,
        local_steps_building=cfg.local_steps_building,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        dp_sigma=cfg.dp_sigma,
        dp_clip_C=cfg.dp_clip_C,
        topk_ratio=cfg.topk_ratio,
        byzantine_fraction=cfg.byzantine_fraction,
        byzantine_attack=cfg.byzantine_attack,
        seed=cfg.seed + round_id,
    )

    # 2. Spawn N agents via Spark RDD (rank-isolated).
    #    Windows-specific: TorchDistributor is often broken on Windows due to
    #    the libuv rendezvous requirement. Using Spark RDD partitions is
    #    more robust, avoids the rendezvous entirely, and is equally
    #    'Spark-native' for distributed task execution.
    manifest_records = manifest_df.to_dict("records")
    rdd = spark.sparkContext.parallelize(manifest_records, numSlices=len(manifest_records))

    # Capture variables for the closure.
    byzantine_ranks_set = byzantine_ranks
    w_round_path_str = str(w_round_path)
    round_id_int = int(round_id)
    deltas_root_str = str(deltas_root)

    def spark_rank_kernel(records):
        import os
        import json
        from pathlib import Path
        from src.federated.spark_local_train import AgentShard, local_train_one_agent

        results = []
        for row in records:
            rank = int(row["rank"])
            shard = AgentShard(
                agent_id=str(row["agent_id"]),
                district_id=str(row["district_id"]),
                parquet_path=str(row["parquet_path"]),
                n_samples=0,
            )
            is_byzantine = rank in byzantine_ranks_set
            metrics = local_train_one_agent(
                shard=shard,
                w_global_path=w_round_path_str,
                round_id=round_id_int,
                output_root=deltas_root_str,
                hparams=hparams,
                is_byzantine=is_byzantine,
            )
            # Replicate the metric-persistence logic of the original kernel.
            metrics_dir = Path(deltas_root_str) / f"round={round_id_int:04d}" / "_metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_row_path = metrics_dir / f"rank_{rank:04d}.json"
            metrics_row_path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")
            results.append(metrics)
        return results

    logging.info(
        "[round=%d] Distributing %d agents via Spark RDD (local master)",
        round_id, len(manifest_records),
    )
    rdd.mapPartitions(spark_rank_kernel).collect()

    # 3. Driver-side: collect per-rank metrics and Δθ files.
    round_dir = deltas_root / f"round={round_id:04d}"
    metrics_df = _load_round_metrics(round_dir)

    # 4. Per-district Byzantine aggregation (Multi-Krum + trimmed mean).
    district_summaries: list[dict[str, Any]] = []
    district_states: dict[str, dict[str, torch.Tensor]] = {}
    district_masks: dict[str, dict[str, torch.Tensor]] = {}
    district_weights: dict[str, float] = {}
    candidate_rejected = False
    relative_update = 0.0
    for did, group in manifest_df.groupby("district_id"):
        agent_ids = group["agent_id"].astype(str).tolist()
        deltas = _load_deltas_for_district(round_dir, agent_ids)
        if metrics_df.empty:
            n_samples_list = [0] * len(deltas)
        else:
            mdf = metrics_df.set_index("agent_id")
            n_samples_list = [
                int(mdf.loc[a, "n_samples"]) if a in mdf.index else 0
                for a in agent_ids
            ]
        G_d, G_mask, summary = _aggregate_district(
            deltas=deltas,
            n_samples=n_samples_list,
            f_byzantine=cfg.f_byzantine,
            clip_threshold=cfg.clip_threshold,
            trim_ratio=cfg.trim_ratio,
        )
        summary["district_id"] = str(did)
        summary["round_id"] = int(round_id)
        district_summaries.append(summary)
        if G_d is not None:
            district_states[str(did)] = G_d
            district_masks[str(did)] = G_mask
            district_weights[str(did)] = max(1.0, float(summary["n_total_samples"]))

    # 5. City-level FedAvg over surviving district summaries → ΔW*.
    if district_states:
        ordered_districts = sorted(district_states.keys())
        delta_global, delta_mask = masked_weighted_fedavg(
            [district_states[d] for d in ordered_districts],
            [district_masks[d] for d in ordered_districts],
            [district_weights[d] for d in ordered_districts],
        )
        baseline_norm = torch.sqrt(
            sum(torch.sum(value.float() ** 2) for value in w_global.values())
        ).item()
        update_norm = torch.sqrt(
            sum(torch.sum(value.float() ** 2) for value in delta_global.values())
        ).item()
        relative_update = update_norm / max(baseline_norm, 1e-12)
        if relative_update > cfg.max_relative_update:
            logging.warning(
                "[round=%d] rejected candidate relative update %.4g > %.4g",
                round_id,
                relative_update,
                cfg.max_relative_update,
            )
            candidate_rejected = True
            new_w = w_global
        else:
            # W_{t+1} = W_t + ΔW_global.
            new_w = {
                key: torch.where(
                    delta_mask[key],
                    w_global[key] + delta_global[key],
                    w_global[key],
                )
                for key in w_global
            }
    else:
        new_w = w_global  # all districts rejected - hold global model.

    round_wallclock = time.monotonic() - round_t0
    n_accepted = int(sum(s["n_accepted"] for s in district_summaries))
    n_total = int(sum(s["n_total"] for s in district_summaries))
    round_metrics: dict[str, Any] = {
        "round_id": int(round_id),
        "wallclock_s": float(round_wallclock),
        "n_total_agents": int(n_total),
        "n_accepted_agents": int(n_accepted),
        "krum_acceptance_rate": float(n_accepted / max(1, n_total)),
        "byzantine_ranks_n": int(len(byzantine_ranks)),
        "byzantine_attack": cfg.byzantine_attack if byzantine_ranks else "",
        "w_global_sha256": _hash_file(w_round_path),
        "manifest_sha256": _hash_file(manifest_path),
        "git_sha": _git_sha(),
        "candidate_rejected": bool(candidate_rejected),
        "relative_update": float(relative_update),
    }
    if not metrics_df.empty:
        loss_last = pd.to_numeric(metrics_df["loss_last"], errors="coerce").dropna()
        round_metrics["loss_last_mean"] = float(loss_last.mean()) if len(loss_last) else float("nan")
        round_metrics["loss_last_std"] = float(loss_last.std()) if len(loss_last) > 1 else 0.0

    # 7. Persist round audit (Parquet, partitioned for queryability).
    audit_dir = run_dir / "audit"
    if not metrics_df.empty:
        agent_audit = metrics_df.copy()
        agent_audit["round_id"] = int(round_id)
        agent_audit_dir = audit_dir / "agent_round_audit"
        agent_audit_dir.mkdir(parents=True, exist_ok=True)
        agent_audit.to_parquet(
            agent_audit_dir / f"round={round_id:04d}.parquet", index=False
        )
    if district_summaries:
        for s in district_summaries:
            s.pop("accepted_idx", None)
        district_audit_dir = audit_dir / "district_round_summary"
        district_audit_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(district_summaries).to_parquet(
            district_audit_dir / f"round={round_id:04d}.parquet", index=False
        )

    return new_w, round_metrics


# --------------------------------------------------------------------------- #
# Public driver entrypoint.
# --------------------------------------------------------------------------- #
def run_spark_fl(cfg: SparkFLConfig) -> Path:
    """Run a complete Spark FL training session and return the run directory."""
    from pyspark.sql import SparkSession  # imported lazily so unit tests can mock.

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(cfg.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8"
    )
    logging.info("[spark-fl] Run dir: %s", run_dir)

    # SparkSession is required so TorchDistributor inherits a master URL and
    # so the per-round audit Parquets are written under a configured driver.
    spark = (
        SparkSession.builder.appName("NeuroGrid_SparkFL")
        .master(cfg.spark_master)
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        manifest_df = _discover_agent_manifest(
            cfg.parquet_root,
            cfg.num_agents,
            cfg.num_districts,
            sampling_manifest_path=cfg.manifest_path,
        )
        byzantine_ranks = _select_byzantine_ranks(
            cfg.num_agents, cfg.byzantine_fraction, cfg.seed
        )

        # Resume support: if --resume_from_round R is passed, load W from disk
        # and skip rounds [0, R). Idempotent because Δθ files are content-
        # addressable by (round_id, agent_id) and overwritten on rerun.
        start_round = 0
        w_global = _initial_global_state(cfg)
        if cfg.resume_from_round is not None and cfg.resume_from_round > 0:
            prior_w = run_dir / "W_global" / f"round={cfg.resume_from_round - 1:04d}" / "W.pt"
            if prior_w.exists():
                w_global = torch.load(str(prior_w), map_location="cpu", weights_only=True)
                start_round = cfg.resume_from_round
                logging.info("[spark-fl] Resuming from round %d.", start_round)
            else:
                logging.warning(
                    "[spark-fl] resume_from_round=%d but %s missing; starting from 0.",
                    cfg.resume_from_round, prior_w,
                )

        monitor = ConvergenceMonitor(
            epsilon=cfg.convergence_epsilon, patience=cfg.convergence_patience
        )
        round_history: list[dict[str, Any]] = []
        prev_w: dict[str, torch.Tensor] | None = None
        for round_id in range(start_round, cfg.rounds):
            new_w, metrics = _run_one_round(
                spark=spark,
                cfg=cfg,
                round_id=round_id,
                manifest_df=manifest_df,
                w_global=w_global,
                byzantine_ranks=byzantine_ranks,
                run_dir=run_dir,
            )
            converged, drift = monitor.update(prev_w, new_w)
            metrics["w_drift"] = float(drift) if drift != float("inf") else None
            metrics["converged"] = bool(converged)
            round_history.append(metrics)
            logging.info(
                "[spark-fl] round=%d wall=%.2fs accepted=%d/%d drift=%.4g loss_last=%.4f",
                round_id, metrics["wallclock_s"],
                metrics["n_accepted_agents"], metrics["n_total_agents"],
                metrics.get("w_drift") or float("nan"),
                metrics.get("loss_last_mean", float("nan")),
            )
            prev_w = w_global
            w_global = new_w
            if converged:
                logging.info("[spark-fl] Converged at round=%d.", round_id)
                break

        # Final outputs.
        round_metrics_df = pd.DataFrame(round_history)
        round_metrics_path = run_dir / "round_metrics.parquet"
        round_metrics_df.to_parquet(round_metrics_path, index=False)
        torch.save(w_global, str(run_dir / "W_final.pt"))
        summary = {
            "run_id": run_id,
            "config": asdict(cfg),
            "manifest_size": int(len(manifest_df)),
            "byzantine_ranks": sorted(byzantine_ranks),
            "rounds_executed": int(len(round_history)),
            "final_loss_last_mean": (
                float(round_metrics_df["loss_last_mean"].iloc[-1])
                if "loss_last_mean" in round_metrics_df.columns and len(round_metrics_df) else float("nan")
            ),
            "final_acceptance_rate": (
                float(round_metrics_df["krum_acceptance_rate"].iloc[-1])
                if len(round_metrics_df) else float("nan")
            ),
            "host": socket.gethostname(),
            "spark_master": cfg.spark_master,
            "git_sha": _git_sha(),
        }
        (run_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        _write_run_summary_md(run_dir, summary, round_metrics_df)
        return run_dir
    finally:
        spark.stop()


def _write_run_summary_md(run_dir: Path, summary: dict[str, Any], round_df: pd.DataFrame) -> None:
    """Write the human-readable companion to run_summary.json."""
    lines = [
        f"# Spark FL Run `{summary['run_id']}`",
        "",
        f"- Spark master: `{summary['spark_master']}`",
        f"- Manifest size: **{summary['manifest_size']}** agents",
        f"- Rounds executed: **{summary['rounds_executed']}**",
        f"- Byzantine ranks: `{summary['byzantine_ranks']}`",
        f"- Final loss_last (mean): **{summary['final_loss_last_mean']:.4f}**",
        f"- Final Krum acceptance rate: **{summary['final_acceptance_rate']:.2%}**",
        f"- Git SHA: `{summary['git_sha']}`",
        "",
        "## Per-round metrics",
        "",
    ]
    if not round_df.empty:
        cols = [c for c in [
            "round_id", "wallclock_s", "n_total_agents", "n_accepted_agents",
            "krum_acceptance_rate", "loss_last_mean", "w_drift", "converged",
        ] if c in round_df.columns]
        lines.append(round_df[cols].to_markdown(index=False, floatfmt=".4f"))
    (run_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")
