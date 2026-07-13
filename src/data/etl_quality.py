"""
ETL data-quality helpers for NeuroGrid Spark ETL.

Pure-Python helpers (no Spark dependency) that:
  * validate the presence of required schema columns and raise loudly if
    any are missing,
  * derive ratios + summary fields from raw row counts,
  * write a structured run report (JSON + Markdown) under
    `artifacts/etl_runs/<run_id>/`.

The Spark side of the ETL collects raw counts (rows in / rows out / null
rates / weather join coverage) and hands them to `build_dq_report` here.
Keeping the math pure-Python makes it testable without starting a
SparkSession and reusable by sampling and validation tools.
"""
from __future__ import annotations

import json
import os
import platform
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.schema import DAILY_FEATURE_COLUMNS

# --------------------------------------------------------------------------- #
# Required schema contracts
# --------------------------------------------------------------------------- #

# Raw meter CSV (HEAPO 15-min export). Semicolon-separated.
REQUIRED_METER_COLUMNS: tuple[str, ...] = (
    "Household_ID",
    "Timestamp",
    "kWh_received_Total",
)

# Raw hourly weather CSV.
REQUIRED_WEATHER_COLUMNS: tuple[str, ...] = (
    "Weather_ID",
    "Timestamp",
    "Temperature_avg_hourly",
    "Humidity_avg_hourly",
    "Sunshine_duration_hourly",
)

# Households metadata join.
REQUIRED_META_COLUMNS: tuple[str, ...] = (
    "Household_ID",
    "Weather_ID",
)

# Output Parquet must always carry the canonical 11-dim base feature set
# plus the partition keys. Lag/rolling columns are optional (extended mode).
REQUIRED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "Household_ID",
    "Weather_ID",
    "Timestamp",
    *DAILY_FEATURE_COLUMNS,
)


class SchemaValidationError(ValueError):
    """Raised when a required column is missing or unparseable."""


def validate_columns(
    present: Iterable[str],
    required: Iterable[str],
    label: str,
) -> None:
    """Raise `SchemaValidationError` if any required column is absent.

    `label` is included in the error to disambiguate (e.g. "raw meter").
    """
    present_set = set(present)
    missing = [c for c in required if c not in present_set]
    if missing:
        raise SchemaValidationError(
            f"[{label}] missing required column(s): {missing}. "
            f"Found: {sorted(present_set)}"
        )


# --------------------------------------------------------------------------- #
# DQ report dataclass
# --------------------------------------------------------------------------- #


@dataclass
class ETLRunReport:
    """Structured ETL run summary persisted to JSON / Markdown."""

    run_id: str
    started_at: str
    finished_at: str
    runtime_seconds: float

    # Inputs
    data_dir: str
    weather_dir: str | None
    meta_path: str
    output_dir: str

    # Spark configuration snapshot
    spark_conf: dict[str, str]

    # Row counts
    rows_in_meter_raw: int
    rows_after_meta_join: int
    rows_after_household_sample: int
    rows_after_resample_15min: int
    rows_after_weather_join: int
    rows_written: int

    # Quality / coverage
    distinct_households_in: int
    distinct_households_out: int
    null_kwh_pre_impute: int
    null_kwh_post_impute: int
    weather_missing_rows: int
    load_missing_rows: int

    # Layout
    partition_columns: list[str]
    extended_features: bool
    output_files: int
    output_size_bytes: int

    # Derived ratios (filled by post_init)
    null_kwh_pre_impute_ratio: float = 0.0
    null_kwh_post_impute_ratio: float = 0.0
    weather_join_coverage: float = 0.0
    sampled: bool = False
    max_households_requested: int | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Derived ratios - guard against divide-by-zero so a smoke-test run
        # with N=0 still produces a valid report.
        n_pre = max(self.rows_after_resample_15min, 1)
        n_post = max(self.rows_after_weather_join, 1)
        self.null_kwh_pre_impute_ratio = self.null_kwh_pre_impute / n_pre
        self.null_kwh_post_impute_ratio = self.null_kwh_post_impute / n_post
        self.weather_join_coverage = (
            (n_post - self.weather_missing_rows) / n_post if n_post else 0.0
        )


# --------------------------------------------------------------------------- #
# Report assembly + write helpers
# --------------------------------------------------------------------------- #


def make_run_id(prefix: str = "etl") -> str:
    """UTC timestamp run id, sortable lexicographically."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def build_dq_report(
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    data_dir: str,
    weather_dir: str | None,
    meta_path: str,
    output_dir: str,
    spark_conf: Mapping[str, str],
    counts: Mapping[str, int],
    partition_columns: Iterable[str],
    extended_features: bool,
    output_files: int,
    output_size_bytes: int,
    max_households_requested: int | None,
    notes: Iterable[str] = (),
) -> ETLRunReport:
    """Assemble an ETLRunReport from raw count fields. All counts default to 0."""
    runtime = (finished_at - started_at).total_seconds()
    sampled = bool(max_households_requested and max_households_requested > 0)

    def c(k: str) -> int:
        return int(counts.get(k, 0))

    return ETLRunReport(
        run_id=run_id,
        started_at=started_at.astimezone(timezone.utc).isoformat(),
        finished_at=finished_at.astimezone(timezone.utc).isoformat(),
        runtime_seconds=runtime,
        data_dir=data_dir,
        weather_dir=weather_dir,
        meta_path=meta_path,
        output_dir=output_dir,
        spark_conf=dict(spark_conf),
        rows_in_meter_raw=c("rows_in_meter_raw"),
        rows_after_meta_join=c("rows_after_meta_join"),
        rows_after_household_sample=c("rows_after_household_sample"),
        rows_after_resample_15min=c("rows_after_resample_15min"),
        rows_after_weather_join=c("rows_after_weather_join"),
        rows_written=c("rows_written"),
        distinct_households_in=c("distinct_households_in"),
        distinct_households_out=c("distinct_households_out"),
        null_kwh_pre_impute=c("null_kwh_pre_impute"),
        null_kwh_post_impute=c("null_kwh_post_impute"),
        weather_missing_rows=c("weather_missing_rows"),
        load_missing_rows=c("load_missing_rows"),
        partition_columns=list(partition_columns),
        extended_features=bool(extended_features),
        output_files=int(output_files),
        output_size_bytes=int(output_size_bytes),
        sampled=sampled,
        max_households_requested=max_households_requested,
        notes=list(notes),
    )


def write_dq_report(
    report: ETLRunReport,
    runs_dir: str | os.PathLike,
) -> Path:
    """Write `summary.json` + `summary.md` under `runs_dir/<run_id>/`.

    Returns the run directory path.
    """
    run_dir = Path(runs_dir) / report.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = asdict(report)
    payload["host"] = socket.gethostname()
    payload["platform"] = platform.platform()

    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(_to_markdown(payload), encoding="utf-8")
    return run_dir


def _to_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# ETL Run Report - `{payload['run_id']}`")
    lines.append("")
    lines.append(f"- **Started:** {payload['started_at']}")
    lines.append(f"- **Finished:** {payload['finished_at']}")
    lines.append(f"- **Runtime (s):** {payload['runtime_seconds']:.2f}")
    lines.append(f"- **Host / Platform:** {payload.get('host')} / {payload.get('platform')}")
    lines.append(f"- **Sampled:** {payload['sampled']} "
                 f"(max_households_requested={payload['max_households_requested']})")
    lines.append(f"- **Extended features:** {payload['extended_features']}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- data_dir: `{payload['data_dir']}`")
    lines.append(f"- weather_dir: `{payload['weather_dir']}`")
    lines.append(f"- meta_path: `{payload['meta_path']}`")
    lines.append(f"- output_dir: `{payload['output_dir']}`")
    lines.append("")
    lines.append("## Row counts")
    lines.append("| Stage | Rows |")
    lines.append("|---|---:|")
    lines.append(f"| meter raw          | {payload['rows_in_meter_raw']:,} |")
    lines.append(f"| after meta join    | {payload['rows_after_meta_join']:,} |")
    lines.append(f"| after sampling     | {payload['rows_after_household_sample']:,} |")
    lines.append(f"| after 15-min resample | {payload['rows_after_resample_15min']:,} |")
    lines.append(f"| after weather join | {payload['rows_after_weather_join']:,} |")
    lines.append(f"| written            | {payload['rows_written']:,} |")
    lines.append("")
    lines.append("## Quality")
    lines.append(f"- distinct households in : {payload['distinct_households_in']:,}")
    lines.append(f"- distinct households out: {payload['distinct_households_out']:,}")
    lines.append(f"- null kWh pre-impute    : {payload['null_kwh_pre_impute']:,} "
                 f"({payload['null_kwh_pre_impute_ratio']:.4%})")
    lines.append(f"- null kWh post-impute   : {payload['null_kwh_post_impute']:,} "
                 f"({payload['null_kwh_post_impute_ratio']:.4%})")
    lines.append(f"- weather join coverage  : {payload['weather_join_coverage']:.4%} "
                 f"(missing rows: {payload['weather_missing_rows']:,})")
    lines.append(f"- load_missing rows      : {payload['load_missing_rows']:,}")
    lines.append("")
    lines.append("## Output layout")
    lines.append(f"- partition columns: `{payload['partition_columns']}`")
    lines.append(f"- output files     : {payload['output_files']:,}")
    lines.append(f"- output size (bytes): {payload['output_size_bytes']:,}")
    lines.append("")
    lines.append("## Spark configuration")
    for k, v in sorted(payload["spark_conf"].items()):
        lines.append(f"- `{k}` = `{v}`")
    if payload.get("notes"):
        lines.append("")
        lines.append("## Notes")
        for n in payload["notes"]:
            lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Output dir inspection (called after the Spark write completes)
# --------------------------------------------------------------------------- #


def inspect_output_dir(output_dir: str | os.PathLike) -> tuple[int, int]:
    """Return (file_count, total_bytes) for all .parquet files under output_dir."""
    p = Path(output_dir)
    if not p.exists():
        return 0, 0
    files = list(p.rglob("*.parquet"))
    total = sum(f.stat().st_size for f in files if f.is_file())
    return len(files), int(total)
