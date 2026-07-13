"""
Helpers for federated-round audit summaries and Spark-readable persistence.

District rounds already execute inside Spark, so their per-agent audit rows and
their one-row district summary are written by the Spark job itself. The City
tier does not own a SparkSession, but it still needs to emit a Spark-readable
round summary artifact for downstream SQL / presentation analysis. This module
keeps those helpers pure Python so they stay easy to unit test.
"""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


def build_district_round_summary(
    audit_rows: list[dict[str, Any]],
    *,
    district_id: str,
    round_id: int,
    clip_threshold: float,
    trim_ratio: float,
    f_byzantine: int,
    n_total_samples: int | None = None,
    aggregated_available: bool = True,
) -> dict[str, Any]:
    """Aggregate one district round's audit rows into a single summary row."""
    if not audit_rows:
        return {
            "district_id": str(district_id),
            "round_id": int(round_id),
            "clip_threshold": float(clip_threshold),
            "trim_ratio": float(trim_ratio),
            "f_byzantine": int(f_byzantine),
            "aggregated_available": bool(aggregated_available),
            "n_agents_total": 0,
            "n_agents_accepted": 0,
            "n_agents_rejected": 0,
            "n_samples_total": int(n_total_samples or 0),
            "acceptance_rate": 0.0,
            "krum_score_mean": 0.0,
            "krum_score_std": 0.0,
            "krum_score_min": 0.0,
            "krum_score_max": 0.0,
            "original_l2_mean": 0.0,
            "clipped_l2_mean": 0.0,
        }

    n_agents_total = len(audit_rows)
    n_agents_accepted = sum(1 for row in audit_rows if row.get("accepted"))
    n_samples_total = int(
        n_total_samples
        if n_total_samples is not None
        else sum(int(row.get("n_samples", 1)) for row in audit_rows)
    )

    def _mean(key: str) -> float:
        return float(sum(float(row.get(key, 0.0)) for row in audit_rows) / n_agents_total)

    score_min = min(float(row.get("krum_score", 0.0)) for row in audit_rows)
    score_max = max(float(row.get("krum_score", 0.0)) for row in audit_rows)
    score_mean = _mean("krum_score")
    score_var = sum(
        (float(row.get("krum_score", 0.0)) - score_mean) ** 2 for row in audit_rows
    ) / n_agents_total

    return {
        "district_id": str(district_id),
        "round_id": int(round_id),
        "clip_threshold": float(clip_threshold),
        "trim_ratio": float(trim_ratio),
        "f_byzantine": int(f_byzantine),
        "aggregated_available": bool(aggregated_available),
        "n_agents_total": int(n_agents_total),
        "n_agents_accepted": int(n_agents_accepted),
        "n_agents_rejected": int(n_agents_total - n_agents_accepted),
        "n_samples_total": int(n_samples_total),
        "acceptance_rate": float(n_agents_accepted / n_agents_total),
        "krum_score_mean": float(score_mean),
        "krum_score_std": float(score_var ** 0.5),
        "krum_score_min": float(score_min),
        "krum_score_max": float(score_max),
        "original_l2_mean": _mean("original_l2"),
        "clipped_l2_mean": _mean("clipped_l2"),
    }


def build_city_round_summary(
    *,
    round_id: int,
    expected_districts: int,
    district_sample_counts: dict[str, int],
    drift: float | None,
    converged: bool,
    had_previous_global: bool,
) -> dict[str, Any]:
    """Build one structured city-round summary row."""
    district_ids = sorted(str(did) for did in district_sample_counts.keys())
    n_samples_total = sum(int(v) for v in district_sample_counts.values())
    finite_drift = (
        float(drift)
        if drift is not None and math.isfinite(float(drift))
        else None
    )
    return {
        "round_id": int(round_id),
        "expected_districts": int(expected_districts),
        "participating_districts": int(len(district_ids)),
        "district_ids": district_ids,
        "district_sample_counts": {str(k): int(v) for k, v in district_sample_counts.items()},
        "n_samples_total": int(n_samples_total),
        "drift": finite_drift,
        "converged": bool(converged),
        "had_previous_global": bool(had_previous_global),
    }


def append_partitioned_parquet(
    base_dir: str | Path,
    rows: dict[str, Any] | Iterable[dict[str, Any]],
    *,
    partition_cols: Iterable[str] = (),
) -> Path:
    """Append one or more rows as a Spark-readable Hive-partitioned dataset."""
    if isinstance(rows, dict):
        records = [rows]
    else:
        records = list(rows)
    if not records:
        raise ValueError("rows must contain at least one record.")

    partition_cols = tuple(str(c) for c in partition_cols)
    dataset_dir = Path(base_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for row in records:
        normalized = {
            str(k): _normalize_scalar(v)
            for k, v in row.items()
        }
        part_dir = dataset_dir
        payload = dict(normalized)
        for col in partition_cols:
            value = payload.pop(col, None)
            safe = "null" if value is None else str(value)
            part_dir = part_dir / f"{col}={safe}"
        part_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist([payload])
        pq.write_table(table, part_dir / f"part-{uuid.uuid4().hex}.parquet")
    return dataset_dir


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)
