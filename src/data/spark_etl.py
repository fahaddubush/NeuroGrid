"""
Spark/Hadoop ETL: HEAPO smart-meter CSVs + weather → partitioned Parquet.

This is the entry point that turns ~1,400 raw 15-minute CSVs into a single
columnar store the loader can read efficiently. Spark's role here is genuine:
parallel CSV ingestion, distributed timestamp resampling, broadcast join with
hourly weather, and partitioned Parquet write keyed by Weather_ID (the
"district" proxy in the paper) and Household_ID.

Run once before training. Output dir lives next to the raw CSV directory:
    <heapo_root>/processed_parquet_15min/

Recent additive changes:
  * Household sampling (`max_households`) so a Spark cluster can produce a
    medium-light slice for laptop-scale training without re-reading the raw CSVs.
  * Causal expanding-past imputation replaces zero-fill for missing kWh.
    No future or whole-timeline statistic is allowed to influence time t;
    missingness remains explicit in the `load_missing` flag.
  * Temporal + lag / rolling-stat columns are now always materialised in
    Parquet for the official daily training path. Spark is the sole owner of
    preprocessing and feature engineering for 96->96 forecasting.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from src.utils.paths import resolve_path, ensure_dir
from src.data.etl_quality import (
    REQUIRED_META_COLUMNS,
    REQUIRED_METER_COLUMNS,
    REQUIRED_OUTPUT_COLUMNS,
    REQUIRED_WEATHER_COLUMNS,
    build_dq_report,
    inspect_output_dir,
    make_run_id,
    validate_columns,
    write_dq_report,
)

import sys

# Hadoop env must be set BEFORE pyspark imports on Windows.
_HADOOP_RAW = os.getenv("HADOOP_HOME")
if _HADOOP_RAW:
    _hadoop_path = resolve_path(_HADOOP_RAW)
    if _hadoop_path is not None:
        _HADOOP_STR = str(_hadoop_path).replace("/", "\\")
        os.environ["HADOOP_HOME"] = _HADOOP_STR
        os.environ["PATH"] = (
            os.path.join(_HADOOP_STR, "bin") + os.pathsep + os.environ.get("PATH", "")
        )

# Ensure workers use the same Python interpreter as the driver.
# This is critical on Windows where multiple Python versions might exist.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F


def create_spark_session(app_name: str = "NeuroGrid_ETL") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g"))
        .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", "4g"))
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def _add_lag_and_rolling_features(df):
    """Append lag and rolling-window features per household via Spark windows.

    Lags: 1 step (15 min), 4 steps (1 h), 96 steps (24 h).
    Rolling: 4-step mean / std, 96-step max (daily peak so far in window).
    All columns are float; null-safe (rolling skips nulls; lags coalesced to
    the current value at series start).
    """
    w = Window.partitionBy("Household_ID").orderBy("Timestamp")
    df = (
        df.withColumn("lag_1", F.coalesce(F.lag("kWh_received_Total", 1).over(w), F.col("kWh_received_Total")))
        .withColumn("lag_4", F.coalesce(F.lag("kWh_received_Total", 4).over(w), F.col("kWh_received_Total")))
        .withColumn("lag_96", F.coalesce(F.lag("kWh_received_Total", 96).over(w), F.col("kWh_received_Total")))
    )
    w_roll4 = w.rowsBetween(-3, 0)
    w_roll96 = w.rowsBetween(-95, 0)
    df = (
        df.withColumn("roll_mean_4", F.avg("kWh_received_Total").over(w_roll4))
        .withColumn("roll_std_4", F.coalesce(F.stddev_pop("kWh_received_Total").over(w_roll4), F.lit(0.0)))
        .withColumn("roll_max_96", F.max("kWh_received_Total").over(w_roll96))
    )
    return df


def _add_temporal_features(df):
    hour_float = F.hour("Timestamp") + (F.minute("Timestamp") / F.lit(60.0))
    dow = F.dayofweek("Timestamp")  # 1=Sun ... 7=Sat
    dow_zero = ((dow + F.lit(5)) % F.lit(7)).cast("double")  # Mon=0 ... Sun=6
    two_pi = F.lit(2.0 * 3.141592653589793)
    return (
        df.withColumn("hour_sin", F.sin(two_pi * hour_float / F.lit(24.0)))
        .withColumn("hour_cos", F.cos(two_pi * hour_float / F.lit(24.0)))
        .withColumn("dow_sin", F.sin(two_pi * dow_zero / F.lit(7.0)))
        .withColumn("dow_cos", F.cos(two_pi * dow_zero / F.lit(7.0)))
        .withColumn("is_weekend", F.when(dow_zero >= F.lit(5.0), F.lit(1.0)).otherwise(F.lit(0.0)))
    )


def _winsorize_per_household(df):
    quantiles = (
        df.groupBy("Household_ID")
        .agg(
            F.expr("percentile_approx(kWh_received_Total, 0.001, 10000)").alias("_kwh_lo"),
            F.expr("percentile_approx(kWh_received_Total, 0.999, 10000)").alias("_kwh_hi"),
        )
    )
    clipped = df.join(F.broadcast(quantiles), on="Household_ID", how="left")
    clipped = clipped.withColumn("_kwh_lo", F.greatest(F.col("_kwh_lo"), F.lit(0.0)))
    clipped = clipped.withColumn(
        "kWh_received_Total",
        F.when(
            F.col("_kwh_hi").isNotNull() & (F.col("_kwh_hi") > F.col("_kwh_lo")),
            F.least(F.greatest(F.col("kWh_received_Total"), F.col("_kwh_lo")), F.col("_kwh_hi")),
        ).otherwise(F.col("kWh_received_Total")),
    )
    return clipped.drop("_kwh_lo", "_kwh_hi")


def run_etl(
    data_dir: str | None = None,
    output_dir: str | None = None,
    max_households: int | None = None,
    household_manifest: str | None = None,
    source_timezone: str | None = None,
) -> str:
    """Read raw CSVs, resample to 15-min, join hourly weather, write Parquet.

    Args:
        data_dir: HEAPO 15-min CSV directory (defaults to env HEAPO_DATA_DIR).
        output_dir: Parquet output (defaults to <heapo_root>/processed_parquet_15min).
        max_households: alphabetical-limit sampling fallback.
            Ignored when `household_manifest` is given.
        household_manifest: optional path to a sampling manifest JSON
            (produced by `python -m src.cli sample-households`). When given,
            ETL filters to exactly that cohort.

    Returns the Parquet output directory.
    """
    if data_dir is None:
        data_dir = os.getenv("HEAPO_DATA_DIR")
    if not data_dir:
        raise ValueError("HEAPO_DATA_DIR is required.")
    data_dir = str(resolve_path(data_dir))
    source_timezone = source_timezone or os.getenv("HEAPO_TIMEZONE", "UTC")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(data_dir), "processed_parquet_15min")
    else:
        output_dir = str(resolve_path(output_dir))

    heapo_root = os.path.dirname(os.path.dirname(data_dir))
    weather_dir = os.path.join(heapo_root, "weather_data", "hourly")
    meta_path = os.path.join(heapo_root, "meta_data", "households.csv")

    run_id = make_run_id()
    started_at = datetime.now(timezone.utc)
    counts: dict[str, int] = {}
    notes: list[str] = []

    spark = create_spark_session()
    print(f"[ETL] Reading meter CSVs from {data_dir}")
    meter_raw = (
        spark.read.option("header", "true").option("sep", ";").csv(data_dir)
    )
    # Schema gate: fail loudly if HEAPO export changed shape, rather than
    # silently producing NULL columns downstream.
    validate_columns(meter_raw.columns, REQUIRED_METER_COLUMNS, "raw meter")
    meter = (
        meter_raw
        .withColumn(
            "Timestamp",
            F.to_utc_timestamp(F.to_timestamp("Timestamp"), source_timezone),
        )
        .withColumn("kWh_received_Total", F.col("kWh_received_Total").cast("float"))
        .withColumn("Household_ID", F.col("Household_ID").cast("string"))
    ).coalesce(50)
    meter_stats = meter.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("Household_ID").alias("households"),
    ).first()
    counts["rows_in_meter_raw"] = int(meter_stats["rows"])
    counts["distinct_households_in"] = int(meter_stats["households"])

    print(f"[ETL] Reading metadata from {meta_path}")
    meta_raw = (
        spark.read.option("header", "true").option("sep", ";").csv(meta_path)
    )
    validate_columns(meta_raw.columns, REQUIRED_META_COLUMNS, "households metadata")
    meta = meta_raw.select("Household_ID", "Weather_ID")
    meter = meter.join(meta, on="Household_ID", how="left")
    counts["rows_after_meta_join"] = meter.count()

    # Medium-light sampling: keep only the first N household IDs (alphabetical
    # - deterministic). This is done BEFORE the heavy resample/join so the
    # downstream stages process less data.
    if household_manifest:
        from src.data.representative_sampling import load_manifest
        manifest_ids = load_manifest(household_manifest)
        if not manifest_ids:
            raise ValueError(f"Manifest at {household_manifest} contains no households.")
        rows = [(hid,) for hid in manifest_ids]
        manifest_df = spark.createDataFrame(rows, schema=["Household_ID"])
        meter = meter.join(F.broadcast(manifest_df), on="Household_ID", how="inner")
        print(
            f"[ETL] Filtering to {len(manifest_ids)} households from manifest "
            f"{household_manifest} (KMeans-stratified cohort)."
        )
        notes.append(
            f"Cohort-restricted ETL via manifest {household_manifest} "
            f"({len(manifest_ids)} households)."
        )
    elif max_households is not None and max_households > 0:
        sampled = (
            meter.select("Household_ID").distinct().orderBy("Household_ID").limit(max_households)
        )
        meter = meter.join(F.broadcast(sampled), on="Household_ID", how="inner")
        print(f"[ETL] Sampling to first {max_households} households (alphabetical).")
        notes.append(
            f"Alphabetical-LIMIT sampling to {max_households} households. "
            "Use --household_manifest <path> for the principled KMeans-stratified path."
        )
    counts["rows_after_household_sample"] = meter.count()

    # Resample existing observations, then explicitly generate the complete
    # per-household 15-minute grid. A groupBy(window) alone does not create
    # absent intervals, making row-based lags cease to represent elapsed time.
    meter_aggregated = (
        meter.groupBy(
            F.col("Household_ID"),
            F.col("Weather_ID"),
            F.window(F.col("Timestamp"), "15 minutes"),
        )
        .agg(F.sum("kWh_received_Total").alias("kWh_received_Total"))
        .withColumn("Timestamp", F.col("window.start"))
        .drop("window")
    )
    bounds = meter_aggregated.groupBy("Household_ID", "Weather_ID").agg(
        F.min("Timestamp").alias("_start"), F.max("Timestamp").alias("_end")
    )
    grid = bounds.select(
        "Household_ID",
        "Weather_ID",
        F.explode(
            F.sequence("_start", "_end", F.expr("INTERVAL 15 MINUTES"))
        ).alias("Timestamp"),
    )
    meter_15min = grid.join(
        meter_aggregated,
        on=["Household_ID", "Weather_ID", "Timestamp"],
        how="left",
    ).cache()
    counts["rows_after_resample_15min"] = meter_15min.count()
    counts["null_kwh_pre_impute"] = (
        meter_15min.filter(F.col("kWh_received_Total").isNull()).count()
    )

    if os.path.isdir(weather_dir):
        print(f"[ETL] Joining hourly weather from {weather_dir}")
        weather_raw = (
            spark.read.option("header", "true").option("sep", ";").csv(weather_dir)
        )
        validate_columns(weather_raw.columns, REQUIRED_WEATHER_COLUMNS, "raw weather")
        weather = (
            weather_raw
            .withColumn(
                "Timestamp",
                F.to_utc_timestamp(F.to_timestamp("Timestamp"), source_timezone),
            )
            .withColumn("Temperature_avg_hourly", F.col("Temperature_avg_hourly").cast("float"))
            .withColumn("Humidity_avg_hourly", F.col("Humidity_avg_hourly").cast("float"))
            .withColumn("Sunshine_duration_hourly", F.col("Sunshine_duration_hourly").cast("float"))
            .withColumn("Weather_ID", F.col("Weather_ID").cast("string"))
            .select(
                "Weather_ID",
                "Timestamp",
                "Temperature_avg_hourly",
                "Humidity_avg_hourly",
                "Sunshine_duration_hourly",
            )
        )
        duplicate_weather = (
            weather.groupBy("Weather_ID", "Timestamp").count().filter(F.col("count") > 1)
        )
        counts["duplicate_weather_keys"] = duplicate_weather.count()
        weather = weather.groupBy("Weather_ID", "Timestamp").agg(
            F.avg("Temperature_avg_hourly").alias("Temperature_avg_hourly"),
            F.avg("Humidity_avg_hourly").alias("Humidity_avg_hourly"),
            F.avg("Sunshine_duration_hourly").alias("Sunshine_duration_hourly"),
        )
        # Forward-fill hourly weather to the 15-min grid via floor() join key.
        meter_15min = meter_15min.withColumn(
            "WeatherHour", F.date_trunc("hour", F.col("Timestamp"))
        )
        weather = weather.withColumnRenamed("Timestamp", "WeatherHour")
        joined = meter_15min.join(weather, on=["Weather_ID", "WeatherHour"], how="left").drop(
            "WeatherHour"
        )
    else:
        joined = (
            meter_15min.withColumn("Temperature_avg_hourly", F.lit(0.0))
            .withColumn("Humidity_avg_hourly", F.lit(0.0))
            .withColumn("Sunshine_duration_hourly", F.lit(0.0))
        )
        notes.append(
            f"Weather directory not found at {weather_dir}; weather columns zero-filled."
        )

    # Track the missingness flags BEFORE imputation so the model can see where
    # we filled. Then impute kWh with a per-household rolling MEAN over a
    # 5-slot centred window (rowsBetween(-2, 2)); fall back to per-household
    # mean, then to zero. Weather is 0-filled (hourly join handles most cases).
    # Convert IEEE non-finite values into NULL so the normal imputation and
    # missingness paths handle them consistently.
    numeric_cols = [
        "kWh_received_Total",
        "Temperature_avg_hourly",
        "Humidity_avg_hourly",
        "Sunshine_duration_hourly",
    ]
    for column in numeric_cols:
        joined = joined.withColumn(
            column,
            F.when(
                F.col(column).isNull()
                | F.isnan(F.col(column))
                | (F.abs(F.col(column)) == F.lit(float("inf"))),
                F.lit(None).cast("float"),
            ).otherwise(F.col(column)),
        )

    joined = joined.withColumn(
        "load_missing", F.col("kWh_received_Total").isNull().cast("float")
    ).withColumn(
        "weather_missing",
        (
            F.col("Temperature_avg_hourly").isNull()
            | F.col("Humidity_avg_hourly").isNull()
            | F.col("Sunshine_duration_hourly").isNull()
        ).cast("float"),
    ).cache()
    missing_stats = joined.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.col("weather_missing")).alias("weather_missing"),
        F.sum(F.col("load_missing")).alias("load_missing"),
    ).first()
    counts["rows_after_weather_join"] = int(missing_stats["rows"])
    counts["weather_missing_rows"] = int(missing_stats["weather_missing"] or 0)
    counts["load_missing_rows"] = int(missing_stats["load_missing"] or 0)

    # Causal imputation: only observations strictly before t may fill t.
    # Centered windows and whole-household means leak future information.
    w_past = (
        Window.partitionBy("Household_ID")
        .orderBy("Timestamp")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    joined = joined.withColumn(
        "_kwh_past", F.avg("kWh_received_Total").over(w_past)
    ).withColumn(
        "kWh_received_Total",
        F.when(
            F.col("kWh_received_Total").isNull(),
            F.coalesce(F.col("_kwh_past"), F.lit(0.0)),
        ).otherwise(F.col("kWh_received_Total")),
    ).drop("_kwh_past").fillna(
        {
            "Temperature_avg_hourly": 0.0,
            "Humidity_avg_hourly": 0.0,
            "Sunshine_duration_hourly": 0.0,
        }
    )

    joined = _winsorize_per_household(joined)
    joined = _add_temporal_features(joined)
    print("[ETL] Materialising Spark-owned temporal and lag/rolling features.")
    joined = _add_lag_and_rolling_features(joined)

    # Output schema gate (post-impute, pre-write).
    validate_columns(joined.columns, REQUIRED_OUTPUT_COLUMNS, "ETL output")

    output_stats = joined.agg(
        F.sum(F.col("kWh_received_Total").isNull().cast("int")).alias("null_kwh"),
        F.countDistinct("Household_ID").alias("households"),
    ).first()
    counts["null_kwh_post_impute"] = int(output_stats["null_kwh"] or 0)
    counts["distinct_households_out"] = int(output_stats["households"])

    publish_root = Path(output_dir)
    publish_dir = publish_root / "runs" / run_id
    print(f"[ETL] Writing versioned Parquet run to {publish_dir}")
    (
        joined.write.partitionBy("Weather_ID", "Household_ID")
        .mode("errorifexists")
        .parquet(str(publish_dir))
    )
    counts["rows_written"] = counts["rows_after_weather_join"]

    # Snapshot Spark configuration BEFORE stopping the session.
    sc = spark.sparkContext
    spark_conf_keys = (
        "spark.app.name",
        "spark.master",
        "spark.driver.memory",
        "spark.executor.memory",
        "spark.sql.shuffle.partitions",
        "spark.sql.execution.arrow.pyspark.enabled",
    )
    spark_conf_snapshot = {k: sc.getConf().get(k, "") for k in spark_conf_keys}

    spark.stop()

    # Inspect the written output dir + emit DQ artifacts.
    output_files, output_bytes = inspect_output_dir(str(publish_dir))
    publish_root.mkdir(parents=True, exist_ok=True)
    pointer_tmp = publish_root / "CURRENT.tmp"
    pointer_tmp.write_text(run_id + "\n", encoding="utf-8")
    os.replace(pointer_tmp, publish_root / "CURRENT")
    finished_at = datetime.now(timezone.utc)
    runs_dir = ensure_dir("artifacts/etl_runs")
    report = build_dq_report(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        data_dir=str(data_dir),
        weather_dir=str(weather_dir) if os.path.isdir(weather_dir) else None,
        meta_path=str(meta_path),
        output_dir=str(publish_dir),
        spark_conf=spark_conf_snapshot,
        counts=counts,
        partition_columns=["Weather_ID", "Household_ID"],
        extended_features=True,
        output_files=output_files,
        output_size_bytes=output_bytes,
        max_households_requested=max_households,
        notes=notes,
    )
    run_dir = write_dq_report(report, runs_dir)
    print(f"[ETL] DQ report written to {run_dir}")
    print("[ETL] Done.")
    return str(publish_dir)


def main():
    p = argparse.ArgumentParser("neurogrid-etl")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_households", type=int, default=None,
                   help="Sample only the first N households (alphabetical) for medium-light training.")
    p.add_argument("--household_manifest", default=None,
                   help="Optional path to a sampling manifest JSON.")
    p.add_argument("--source_timezone", default=None,
                   help="IANA timezone of naive source timestamps (default HEAPO_TIMEZONE or UTC).")
    args = p.parse_args()
    run_etl(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_households=args.max_households,
        household_manifest=args.household_manifest,
        source_timezone=args.source_timezone,
    )


if __name__ == "__main__":
    main()
