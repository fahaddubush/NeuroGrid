"""
Spark-backed Byzantine pipeline used by the District tier (Algorithm 3).

Pipeline (Diagram 5, left column):
    1. Receive Δθ_k for k in 1..K
    2. Clip:    Δθ'_k = Δθ_k / max(1, ||Δθ_k|| / C)
    3. Score:   s_k = sum over n-f-2 nearest neighbours of ||Δθ'_k - Δθ'_j||^2
    4. Trimmed mean over Krum-accepted survivors → W_d
    5. W_d → City aggregator

The Spark role is intentionally real, not decorative: the tensor-heavy clip /
distance / Krum / trimmed-mean math stays in Torch, but the *row-shaped*
district audit lives in Spark so per-agent metrics and one-row district round
summaries can be persisted directly to partitioned Parquet. That audit trail
is what feeds the convergence dashboards downstream.

We use Spark for:
  * Row-shaped audit (`createDataFrame` on the per-agent metric rows)
  * Cohort statistics (mean / stddev / min / max + n_accepted / n_total
    over the Krum-score vector)
  * Partitioned Parquet writes for:
      - `agent_round_audit/`
      - `district_round_summary/`
    keyed by `district_id, round_id`

Acceptance is decided by **Multi-Krum**, not a MAD/z-score threshold: the
top `n - f` agents by Krum score are kept; the rest are rejected.
(`src.federated.krum.multi_krum_select`).

The tensor math (clip + flatten + pairwise distances) stays in Torch - Spark
DataFrames are not the right shape for parameter vectors with millions of
coordinates. Spark adds value at the *cohort* level, not the *tensor* level.
"""
from __future__ import annotations

from typing import Any

import torch
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.federated.audit import build_district_round_summary
from src.federated.clipping import clip_state_dict, state_dict_l2_norm
from src.federated.krum import multi_krum_select
from src.federated.aggregation import masked_trimmed_mean


def run_spark_byzantine_round(
    spark: SparkSession,
    incoming: list[dict[str, Any]],
    f_byzantine: int,
    clip_threshold: float,
    trim_ratio: float = 0.1,
    audit_path: str | None = None,
    district_id: str = "D0",
    round_id: int = 0,
) -> tuple[
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor] | None,
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Run one full Byzantine FL round at the District tier.

    Args:
        incoming: list of dicts {"agent_id": str, "state_dict": dict, "n_samples": int}
        f_byzantine: expected number of Byzantine agents in this cohort
        clip_threshold: per-update L2 clip (paper's `C`)
        trim_ratio: coordinate-wise trim fraction over Krum survivors
        audit_path: optional Parquet path; appended with district_id + round_id
                    partition columns
        district_id, round_id: audit labels

    Returns:
        (aggregated_state_dict, audit_rows, district_summary).
        Returns (None, [], summary) if no input.
    """
    if not incoming:
        return None, None, [], build_district_round_summary(
            [],
            district_id=district_id,
            round_id=round_id,
            clip_threshold=clip_threshold,
            trim_ratio=trim_ratio,
            f_byzantine=f_byzantine,
            n_total_samples=0,
            aggregated_available=False,
        )

    # Step 1+2: clip every incoming update and record original L2 norm.
    clipped_updates: list[dict[str, torch.Tensor]] = []
    update_masks: list[dict[str, torch.Tensor]] = []
    audit: list[dict[str, Any]] = []
    for item in incoming:
        clipped, original_norm = clip_state_dict(item["state_dict"], clip_threshold)
        clipped_updates.append(clipped)
        update_masks.append(
            item.get("masks")
            or {key: torch.ones_like(value, dtype=torch.bool) for key, value in clipped.items()}
        )
        audit.append(
            {
                "agent_id": str(item["agent_id"]),
                "n_samples": int(item.get("n_samples", 1)),
                "original_l2": float(original_norm),
                "clipped_l2": float(state_dict_l2_norm(clipped)),
            }
        )

    # Step 3: Krum scoring. Multi-Krum keeps n - f survivors.
    if f_byzantine == 0:
        accepted_idx = list(range(len(clipped_updates)))
        rejected_idx: list[int] = []
        scores = torch.zeros(len(clipped_updates))
    else:
        accepted_idx, rejected_idx, scores = multi_krum_select(
            clipped_updates, f=f_byzantine
        )
    score_list = scores.tolist()

    accepted_set = set(accepted_idx)
    for idx, row in enumerate(audit):
        row["krum_score"] = float(score_list[idx])
        row["accepted"] = idx in accepted_set
        row["selection_mode"] = "krum_accepted" if row["accepted"] else "krum_rejected"

    # Step 4: trimmed mean over survivors → district summary G_d.
    survivors = [clipped_updates[i] for i in accepted_idx]
    survivor_masks = [update_masks[i] for i in accepted_idx]
    n_total_samples = sum(int(row.get("n_samples", 1)) for row in audit)
    if not survivors:
        summary = build_district_round_summary(
            audit,
            district_id=district_id,
            round_id=round_id,
            clip_threshold=clip_threshold,
            trim_ratio=trim_ratio,
            f_byzantine=f_byzantine,
            n_total_samples=n_total_samples,
            aggregated_available=False,
        )
        return None, None, audit, summary
    aggregated, aggregated_masks = masked_trimmed_mean(
        survivors, survivor_masks, trim_ratio=trim_ratio
    )

    # Audit cohort + persistence via Spark.
    score_df = spark.createDataFrame(
        [
            {
                k: v
                for k, v in row.items()
                if k in {"agent_id", "n_samples", "original_l2", "clipped_l2", "krum_score", "accepted"}
            }
            for row in audit
        ]
    )
    cohort = (
        score_df.agg(
            F.avg("krum_score").alias("score_mean"),
            F.stddev_pop("krum_score").alias("score_std"),
            F.min("krum_score").alias("score_min"),
            F.max("krum_score").alias("score_max"),
            F.sum(F.when(F.col("accepted"), 1).otherwise(0)).alias("n_accepted"),
            F.count(F.lit(1)).alias("n_total"),
        )
        .collect()[0]
        .asDict()
    )
    for row in audit:
        row["cohort_score_mean"] = float(cohort.get("score_mean") or 0.0)
        row["cohort_score_std"] = float(cohort.get("score_std") or 0.0)
        row["n_accepted"] = int(cohort.get("n_accepted") or 0)
        row["n_total"] = int(cohort.get("n_total") or 0)
        row["district_id"] = district_id
        row["round_id"] = int(round_id)

    summary = build_district_round_summary(
        audit,
        district_id=district_id,
        round_id=round_id,
        clip_threshold=clip_threshold,
        trim_ratio=trim_ratio,
        f_byzantine=f_byzantine,
        n_total_samples=n_total_samples,
        aggregated_available=True,
    )

    if audit_path:
        audit_df = (
            spark.createDataFrame(audit)
            .withColumn("district_id", F.lit(district_id))
            .withColumn("round_id", F.lit(int(round_id)))
        )
        summary_df = spark.createDataFrame([summary])
        summary_df.write.mode("append").partitionBy("district_id", "round_id").parquet(
            f"{audit_path}/district_round_summary"
        )
        audit_df.write.mode("append").partitionBy("district_id", "round_id").parquet(
            f"{audit_path}/agent_round_audit"
        )

    return aggregated, aggregated_masks, audit, summary
