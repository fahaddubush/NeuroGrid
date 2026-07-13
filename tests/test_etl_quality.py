"""Pure-Python tests for src.data.etl_quality (no Spark dependency)."""
from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.etl_quality import (
    REQUIRED_METER_COLUMNS,
    REQUIRED_OUTPUT_COLUMNS,
    REQUIRED_WEATHER_COLUMNS,
    SchemaValidationError,
    build_dq_report,
    inspect_output_dir,
    make_run_id,
    validate_columns,
    write_dq_report,
)


class ValidateColumnsTests(unittest.TestCase):
    def test_passes_when_all_required_present(self) -> None:
        validate_columns(
            present=["a", "b", "c", "extra"],
            required=["a", "b", "c"],
            label="t",
        )

    def test_raises_listing_missing_columns(self) -> None:
        with self.assertRaises(SchemaValidationError) as ctx:
            validate_columns(present=["a"], required=["a", "b", "c"], label="raw meter")
        msg = str(ctx.exception)
        self.assertIn("raw meter", msg)
        self.assertIn("'b'", msg)
        self.assertIn("'c'", msg)

    def test_required_constants_are_disjoint_to_each_other_where_expected(self) -> None:
        # Spot-check a couple of contract invariants.
        self.assertIn("Household_ID", REQUIRED_METER_COLUMNS)
        self.assertIn("Weather_ID", REQUIRED_WEATHER_COLUMNS)
        self.assertIn("load_missing", REQUIRED_OUTPUT_COLUMNS)
        self.assertIn("weather_missing", REQUIRED_OUTPUT_COLUMNS)


class BuildReportTests(unittest.TestCase):
    def _counts(self, **overrides):
        base = dict(
            rows_in_meter_raw=1000,
            rows_after_meta_join=1000,
            rows_after_household_sample=500,
            rows_after_resample_15min=480,
            rows_after_weather_join=480,
            rows_written=480,
            distinct_households_in=20,
            distinct_households_out=20,
            null_kwh_pre_impute=12,
            null_kwh_post_impute=0,
            weather_missing_rows=4,
            load_missing_rows=12,
        )
        base.update(overrides)
        return base

    def test_derives_ratios(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(seconds=42)
        report = build_dq_report(
            run_id="t1",
            started_at=start,
            finished_at=end,
            data_dir="/d",
            weather_dir="/w",
            meta_path="/m",
            output_dir="/o",
            spark_conf={"spark.master": "local[1]"},
            counts=self._counts(),
            partition_columns=["Weather_ID", "Household_ID"],
            extended_features=False,
            output_files=10,
            output_size_bytes=1234,
            max_households_requested=20,
        )
        self.assertEqual(report.runtime_seconds, 42.0)
        self.assertAlmostEqual(report.null_kwh_pre_impute_ratio, 12 / 480, places=6)
        self.assertEqual(report.null_kwh_post_impute_ratio, 0.0)
        self.assertAlmostEqual(report.weather_join_coverage, (480 - 4) / 480, places=6)
        self.assertTrue(report.sampled)
        self.assertEqual(report.partition_columns, ["Weather_ID", "Household_ID"])

    def test_zero_division_safe_when_empty(self) -> None:
        start = datetime.now(timezone.utc)
        report = build_dq_report(
            run_id="empty",
            started_at=start,
            finished_at=start,
            data_dir="/d",
            weather_dir=None,
            meta_path="/m",
            output_dir="/o",
            spark_conf={},
            counts=self._counts(
                rows_after_resample_15min=0,
                rows_after_weather_join=0,
                weather_missing_rows=0,
            ),
            partition_columns=["Weather_ID", "Household_ID"],
            extended_features=True,
            output_files=0,
            output_size_bytes=0,
            max_households_requested=None,
        )
        self.assertEqual(report.null_kwh_pre_impute_ratio, 12 / 1)  # safe denom
        self.assertEqual(report.weather_join_coverage, 1.0)
        self.assertFalse(report.sampled)
        self.assertTrue(report.extended_features)


class WriteReportTests(unittest.TestCase):
    def test_writes_summary_json_and_md(self) -> None:
        start = datetime.now(timezone.utc)
        end = start + timedelta(seconds=1)
        report = build_dq_report(
            run_id=make_run_id(),
            started_at=start,
            finished_at=end,
            data_dir="/d",
            weather_dir="/w",
            meta_path="/m",
            output_dir="/o",
            spark_conf={"spark.master": "local[1]"},
            counts={
                "rows_in_meter_raw": 1,
                "rows_after_meta_join": 1,
                "rows_after_household_sample": 1,
                "rows_after_resample_15min": 1,
                "rows_after_weather_join": 1,
                "rows_written": 1,
                "distinct_households_in": 1,
                "distinct_households_out": 1,
                "null_kwh_pre_impute": 0,
                "null_kwh_post_impute": 0,
                "weather_missing_rows": 0,
                "load_missing_rows": 0,
            },
            partition_columns=["Weather_ID", "Household_ID"],
            extended_features=False,
            output_files=1,
            output_size_bytes=500,
            max_households_requested=1,
        )
        with TemporaryDirectory() as tmp:
            run_dir = write_dq_report(report, tmp)
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "summary.md").exists())
            payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], report.run_id)
            self.assertIn("host", payload)
            md = (run_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("ETL Run Report", md)
            self.assertIn("Spark configuration", md)


class InspectOutputDirTests(unittest.TestCase):
    def test_counts_parquet_files_and_bytes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Weather_ID=A" / "Household_ID=h1").mkdir(parents=True)
            (root / "Weather_ID=A" / "Household_ID=h1" / "part-001.parquet").write_bytes(b"x" * 100)
            (root / "Weather_ID=A" / "Household_ID=h2").mkdir(parents=True)
            (root / "Weather_ID=A" / "Household_ID=h2" / "part-002.parquet").write_bytes(b"y" * 250)
            (root / "Weather_ID=A" / "Household_ID=h2" / "_SUCCESS").write_bytes(b"")
            n, size = inspect_output_dir(root)
            self.assertEqual(n, 2)
            self.assertEqual(size, 350)

    def test_returns_zero_for_missing_dir(self) -> None:
        n, size = inspect_output_dir("/definitely/not/a/path/xyzzy")
        self.assertEqual((n, size), (0, 0))


if __name__ == "__main__":
    unittest.main()
