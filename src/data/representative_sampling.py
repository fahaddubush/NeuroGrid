"""
Representative household sampling with Spark MLlib.

Replaces the alphabetical `LIMIT N` "sampling" path in `spark_etl.py` with a
distributed Spark pipeline:

    raw HEAPO CSVs                       (Spark CSV reader)
        │
        ▼
    per-household profile                (Spark groupBy aggregations)
      mean_load, std_load, peak_load,
      missingness_rate, zero_load_fraction, n_records
        │
        ▼
    StandardScaler + VectorAssembler     (Spark MLlib feature pipeline)
        │
        ▼
    Spark MLlib KMeans (k clusters)      (Spark MLlib clustering)
        │
        ▼
    balanced representative selection    (pure Python - testable)
        │
        ▼
    artifacts/sampling/<run_id>/
        profiles.parquet      (every household + cluster id + features)
        manifest.json         (selected household IDs + cluster + reason)
        summary.md            (counts per cluster, stats, runtime)

The selection logic is pure-Python (`stratified_select`) so it can be
unit-tested without spinning a SparkSession. Spark MLlib does the heavy
lifting (clustering K households across the cluster), while selection
intentionally favors cluster balance over raw population proportionality so
minority consumption profiles remain represented in the sampled cohort.
"""
from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.etl_quality import make_run_id
from src.utils.paths import ensure_dir, resolve_path
import sys

# Ensure workers use the same Python interpreter as the driver.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Default profile feature names used for clustering. Kept as a tuple so
# tests / docs can import without pulling pyspark.
PROFILE_FEATURE_COLUMNS: tuple[str, ...] = (
    "mean_load",
    "std_load",
    "peak_load",
    "missingness_rate",
    "zero_load_fraction",
)


# --------------------------------------------------------------------------- #
# Pure-Python selection (testable without Spark)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SelectedHousehold:
    household_id: str
    cluster: int
    rank_in_cluster: int


def stratified_select(
    assignments: Iterable[tuple[str, int]],
    target_n: int,
    seed: int = 0,
) -> list[SelectedHousehold]:
    """Pick `target_n` households across clusters with balanced coverage.

    Within each cluster, selection is deterministic-given-`seed`: a stable
    shuffle by `(seed, cluster, household_id)`.

    Allocation policy:
      1. Try to give each non-empty cluster an equal share (`target_n / k`),
         capped by cluster size.
      2. Redistribute any leftover slots to clusters that still have
         headroom, in deterministic order, until `target_n` is reached.

    This intentionally avoids the old purely proportional behaviour, which
    could collapse a 30-household cohort into ~29/1/0 selections when the
    learned cluster populations were heavily skewed.

    Returns a list of `SelectedHousehold` entries, sorted by household_id.
    """
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for hh, c in assignments:
        by_cluster[int(c)].append(str(hh))

    if not by_cluster or target_n <= 0:
        return []

    total = sum(len(v) for v in by_cluster.values())
    target_n = min(target_n, total)
    ordered_clusters = sorted(by_cluster)

    # Deterministic shuffle inside each cluster.
    for c in ordered_clusters:
        ordered = sorted(by_cluster[c])
        rng_local = random.Random(hash((seed, c)) & 0xFFFFFFFF)
        rng_local.shuffle(ordered)
        by_cluster[c] = ordered

    quotas: dict[int, int] = {}
    allocated = 0
    n_clusters = len(ordered_clusters)
    base_share = target_n // n_clusters
    remainder = target_n % n_clusters

    # Equal-share allocation first, capped by cluster size.
    for idx, c in enumerate(ordered_clusters):
        desired = base_share + (1 if idx < remainder else 0)
        quotas[c] = min(desired, len(by_cluster[c]))
        allocated += quotas[c]

    # Redistribute any unallocated remainder to clusters with headroom.
    while allocated < target_n:
        progressed = False
        for c in sorted(ordered_clusters, key=lambda cid: (-len(by_cluster[cid]), cid)):
            if quotas[c] < len(by_cluster[c]):
                quotas[c] += 1
                allocated += 1
                progressed = True
                if allocated >= target_n:
                    break
        if not progressed:
            break

    # Materialise selection.
    selected: list[SelectedHousehold] = []
    for c in ordered_clusters:
        members = by_cluster[c]
        for rank, hh in enumerate(members[: quotas[c]]):
            selected.append(SelectedHousehold(household_id=hh, cluster=c, rank_in_cluster=rank))

    selected.sort(key=lambda s: s.household_id)
    return selected


# --------------------------------------------------------------------------- #
# Manifest / report dataclasses + writers
# --------------------------------------------------------------------------- #


@dataclass
class SamplingReport:
    run_id: str
    started_at: str
    finished_at: str
    runtime_seconds: float
    data_dir: str
    target_n: int
    k_clusters: int
    seed: int
    total_households: int
    selected_count: int
    cluster_sizes: dict[int, int]
    cluster_selected: dict[int, int]
    feature_columns: list[str]
    output_dir: str
    notes: list[str] = field(default_factory=list)


def write_sampling_manifest(
    selected: list[SelectedHousehold],
    report: SamplingReport,
    output_root: str | os.PathLike,
) -> Path:
    """Write `manifest.json` + `summary.md` under `output_root/<run_id>/`.

    The manifest is the canonical output the ETL reads; `summary.md` is
    presentation-ready.
    """
    run_dir = Path(output_root) / report.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": report.run_id,
        "selection_method": "spark_mllib_kmeans_stratified",
        "k_clusters": report.k_clusters,
        "seed": report.seed,
        "target_n": report.target_n,
        "total_households": report.total_households,
        "feature_columns": report.feature_columns,
        "selected": [asdict(s) for s in selected],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(_to_markdown(report, selected), encoding="utf-8")
    (run_dir / "report.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    return run_dir


def _to_markdown(report: SamplingReport, selected: list[SelectedHousehold]) -> str:
    lines: list[str] = []
    lines.append(f"# Representative Sampling Report - `{report.run_id}`")
    lines.append("")
    lines.append(f"- **Started:** {report.started_at}")
    lines.append(f"- **Finished:** {report.finished_at}")
    lines.append(f"- **Runtime (s):** {report.runtime_seconds:.2f}")
    lines.append(f"- **Method:** Spark MLlib KMeans + balanced per-cluster selection")
    lines.append(f"- **k clusters:** {report.k_clusters}, **seed:** {report.seed}")
    lines.append(f"- **Total households:** {report.total_households:,}")
    lines.append(f"- **Selected:** {report.selected_count:,} (target {report.target_n:,})")
    lines.append("")
    lines.append("## Per-cluster selection")
    lines.append("| Cluster | Cluster size | Selected |")
    lines.append("|---:|---:|---:|")
    for c in sorted(report.cluster_sizes):
        size = report.cluster_sizes[c]
        sel = report.cluster_selected.get(c, 0)
        lines.append(f"| {c} | {size:,} | {sel:,} |")
    lines.append("")
    lines.append("## Feature columns clustered on")
    for col in report.feature_columns:
        lines.append(f"- `{col}`")
    lines.append("")
    lines.append("## Selected household IDs (first 50)")
    for s in selected[:50]:
        lines.append(f"- `{s.household_id}` - cluster {s.cluster}")
    if len(selected) > 50:
        lines.append(f"- … and {len(selected) - 50} more (see `manifest.json`)")
    if report.notes:
        lines.append("")
        lines.append("## Notes")
        for n in report.notes:
            lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines)


def load_manifest(manifest_path: str | os.PathLike) -> list[str]:
    """Return the list of selected household IDs from a sampling manifest."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [s["household_id"] for s in payload.get("selected", [])]


def load_manifest_with_clusters(
    manifest_path: str | os.PathLike,
) -> list[tuple[str, int]]:
    """Return [(household_id, cluster), ...] from a sampling manifest."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [(s["household_id"], int(s["cluster"])) for s in payload.get("selected", [])]


def cluster_stratified_household_split(
    assignments: Iterable[tuple[str, int]],
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Household-disjoint, cluster-stratified train/val/test split.

    Allocation per cluster uses largest-remainder rounding:
    each cluster contributes households to (val, test, train) in that
    priority order so val and test receive at least one member of each
    cluster when feasible. Within a cluster, household ordering is the
    same deterministic shuffle used by `stratified_select` (seeded by
    `(seed, cluster)`), so a fixed seed reproduces the same splits.

    Returns a dict with keys 'train', 'val', 'test' → sorted list of
    household IDs. Households are disjoint across the three lists.

    The default configuration is target_n=30, k=3, and a 24/3/3 split, but
    the algorithm works for any (target_n, k, train_n, val_n, test_n) with
    `train_n + val_n + test_n <= total`. Unused households (e.g. when
    sums don't reach the cluster total) are assigned to train.
    """
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for hh, c in assignments:
        by_cluster[int(c)].append(str(hh))
    if not by_cluster:
        return {"train": [], "val": [], "test": []}

    total = sum(len(v) for v in by_cluster.values())
    if train_n + val_n + test_n > total:
        raise ValueError(
            f"train_n+val_n+test_n={train_n + val_n + test_n} exceeds "
            f"available households={total}"
        )

    # Deterministic per-cluster ordering - match `stratified_select`.
    for c in by_cluster:
        ordered = sorted(by_cluster[c])
        rng_local = random.Random(hash((seed, c)) & 0xFFFFFFFF)
        rng_local.shuffle(ordered)
        by_cluster[c] = ordered

    # Largest-remainder quota for each split.
    def _quotas(n_target: int) -> dict[int, int]:
        quotas: dict[int, int] = {}
        fractional: list[tuple[float, int]] = []
        allocated = 0
        for c, members in by_cluster.items():
            share = len(members) / total * n_target
            base = int(share)
            quotas[c] = min(base, len(members))
            fractional.append((share - base, c))
            allocated += quotas[c]
        # Distribute remainder in cluster order (so each cluster has a
        # chance of getting the +1) - preserves "every cluster represented"
        # for small splits like val=3, test=3 against k=3.
        fractional.sort(key=lambda x: (-x[0], -len(by_cluster[x[1]]), x[1]))
        i = 0
        while allocated < n_target:
            _, c = fractional[i % len(fractional)]
            if quotas[c] < len(by_cluster[c]):
                quotas[c] += 1
                allocated += 1
            i += 1
            if i > 4 * len(fractional):
                break  # all clusters maxed
        return quotas

    # Carve val first (highest priority for cluster coverage), then test,
    # then train fills the remainder.
    val_quota = _quotas(val_n)
    val_ids: list[str] = []
    for c, q in val_quota.items():
        val_ids.extend(by_cluster[c][:q])
        by_cluster[c] = by_cluster[c][q:]

    # Recompute totals for test allocation (val already removed).
    total_after_val = sum(len(v) for v in by_cluster.values())

    def _quotas_remaining(n_target: int) -> dict[int, int]:
        quotas: dict[int, int] = {}
        fractional: list[tuple[float, int]] = []
        allocated = 0
        for c, members in by_cluster.items():
            share = (len(members) / total_after_val) * n_target if total_after_val else 0.0
            base = int(share)
            quotas[c] = min(base, len(members))
            fractional.append((share - base, c))
            allocated += quotas[c]
        fractional.sort(key=lambda x: (-x[0], -len(by_cluster[x[1]]), x[1]))
        i = 0
        while allocated < n_target:
            _, c = fractional[i % len(fractional)]
            if quotas[c] < len(by_cluster[c]):
                quotas[c] += 1
                allocated += 1
            i += 1
            if i > 4 * len(fractional):
                break
        return quotas

    test_quota = _quotas_remaining(test_n)
    test_ids: list[str] = []
    for c, q in test_quota.items():
        test_ids.extend(by_cluster[c][:q])
        by_cluster[c] = by_cluster[c][q:]

    # Train takes up to train_n from the remainder, in cluster order.
    train_ids: list[str] = []
    remaining_pool = [(c, hh) for c, members in by_cluster.items() for hh in members]
    remaining_pool.sort(key=lambda x: (x[0], x[1]))
    for _, hh in remaining_pool[:train_n]:
        train_ids.append(hh)

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }


# --------------------------------------------------------------------------- #
# Spark orchestration (imported lazily so unit tests don't require pyspark)
# --------------------------------------------------------------------------- #


def _build_household_profiles(spark, data_dir: str):
    """Spark groupBy aggregation → one row per household, with profile features."""
    from pyspark.sql import functions as F
    from src.data.etl_quality import REQUIRED_METER_COLUMNS, validate_columns

    raw = spark.read.option("header", "true").option("sep", ";").csv(data_dir)
    validate_columns(raw.columns, REQUIRED_METER_COLUMNS, "raw meter (sampling)")

    df = (
        raw.withColumn("Timestamp", F.col("Timestamp").cast("timestamp"))
        .withColumn("kWh_received_Total", F.col("kWh_received_Total").cast("float"))
        .withColumn("Household_ID", F.col("Household_ID").cast("string"))
    ).coalesce(50)
    profiles = (
        df.groupBy("Household_ID")
        .agg(
            F.avg("kWh_received_Total").alias("mean_load"),
            F.coalesce(F.stddev_pop("kWh_received_Total"), F.lit(0.0)).alias("std_load"),
            F.max("kWh_received_Total").alias("peak_load"),
            F.count(F.lit(1)).alias("n_records"),
            F.sum(F.col("kWh_received_Total").isNull().cast("int")).alias("n_nulls"),
            F.sum(
                F.when(
                    (F.col("kWh_received_Total").isNotNull()) & (F.col("kWh_received_Total") == 0.0),
                    1,
                ).otherwise(0)
            ).alias("n_zero"),
        )
        .withColumn("missingness_rate", F.col("n_nulls") / F.col("n_records"))
        .withColumn("zero_load_fraction", F.col("n_zero") / F.col("n_records"))
        .fillna(0.0)
    )
    return profiles


def _cluster_profiles(profiles_df, k: int, seed: int):
    """Run Spark MLlib KMeans on standardized profile features."""
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.clustering import KMeans

    assembler = VectorAssembler(
        inputCols=list(PROFILE_FEATURE_COLUMNS),
        outputCol="_features_raw",
        handleInvalid="skip",
    )
    df = assembler.transform(profiles_df)
    scaler = StandardScaler(
        inputCol="_features_raw",
        outputCol="features",
        withStd=True,
        withMean=True,
    )
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    kmeans = KMeans(k=k, seed=seed, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(df)
    clustered = model.transform(df).drop("_features_raw", "features")
    return clustered, model


def run_sampling(
    *,
    target_n: int,
    k_clusters: int = 5,
    seed: int = 0,
    data_dir: str | None = None,
    output_root: str | os.PathLike | None = None,
    notes: Iterable[str] = (),
) -> tuple[Path, list[str]]:
    """Run the full Spark MLlib sampling pipeline. Returns (run_dir, selected_ids)."""
    from src.data.spark_etl import create_spark_session  # reuse same session config

    if data_dir is None:
        data_dir = os.getenv("HEAPO_DATA_DIR")
    if not data_dir:
        raise ValueError("HEAPO_DATA_DIR is required.")
    data_dir_resolved = str(resolve_path(data_dir))

    if output_root is None:
        output_root = ensure_dir("artifacts/sampling")
    else:
        output_root = ensure_dir(output_root)

    run_id = make_run_id(prefix="sample")
    started_at = datetime.now(timezone.utc)

    logging.info(
        "[Sampling] data_dir=%s target_n=%d k=%d seed=%d",
        data_dir_resolved, target_n, k_clusters, seed,
    )
    spark = create_spark_session(app_name="NeuroGrid_Sampling")

    profiles = _build_household_profiles(spark, data_dir_resolved).cache()
    total_households = profiles.count()
    if total_households == 0:
        spark.stop()
        raise RuntimeError(f"No households found under {data_dir_resolved}")

    # Cap k to the number of households so KMeans does not error on tiny inputs.
    effective_k = max(1, min(int(k_clusters), int(total_households)))
    adaptive_notes = list(notes)
    min_cluster_size = 2  # enough to support one val + one test household per cluster
    while True:
        clustered, _model = _cluster_profiles(profiles, k=effective_k, seed=seed)
        rows = clustered.select("Household_ID", "cluster").collect()
        assignments = [(r["Household_ID"], int(r["cluster"])) for r in rows]
        cluster_sizes: dict[int, int] = defaultdict(int)
        for _, c in assignments:
            cluster_sizes[c] += 1
        smallest = min(cluster_sizes.values()) if cluster_sizes else 0
        if effective_k <= 1 or smallest >= min_cluster_size:
            break
        adaptive_notes.append(
            f"Adaptive reclustering: requested k={k_clusters} produced min cluster size "
            f"{smallest}; reran with k={effective_k - 1}."
        )
        effective_k -= 1

    # Persist per-household profile + cluster id.
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (
        clustered.select(
            "Household_ID",
            *PROFILE_FEATURE_COLUMNS,
            "n_records",
            "n_nulls",
            "n_zero",
            "cluster",
        )
        .write.mode("overwrite")
        .parquet(str(run_dir / "profiles.parquet"))
    )

    selected = stratified_select(assignments, target_n=target_n, seed=seed)

    # Aggregate cluster sizes for the report.
    cluster_selected: dict[int, int] = defaultdict(int)
    for s in selected:
        cluster_selected[s.cluster] += 1

    finished_at = datetime.now(timezone.utc)
    report = SamplingReport(
        run_id=run_id,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        runtime_seconds=(finished_at - started_at).total_seconds(),
        data_dir=data_dir_resolved,
        target_n=int(target_n),
        k_clusters=int(effective_k),
        seed=int(seed),
        total_households=int(total_households),
        selected_count=len(selected),
        cluster_sizes={int(c): int(n) for c, n in cluster_sizes.items()},
        cluster_selected={int(c): int(n) for c, n in cluster_selected.items()},
        feature_columns=list(PROFILE_FEATURE_COLUMNS),
        output_dir=str(run_dir),
        notes=adaptive_notes,
    )
    write_sampling_manifest(selected, report, output_root)

    spark.stop()
    selected_ids = [s.household_id for s in selected]
    logging.info("[Sampling] Wrote %d selected households to %s", len(selected_ids), run_dir)
    return run_dir, selected_ids
