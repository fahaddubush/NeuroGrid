"""Tests for the feature pipeline + schema."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data.feature_pipeline import (
    ForecastConfig,
    add_temporal_features,
    build_feature_frame,
    feature_frame_to_matrix,
    save_forecast_config,
    load_forecast_config,
)
from src.data.schema import DAILY_FEATURE_COLUMNS, FEATURE_COLUMNS, INPUT_DIM


def _make_meter_frame(n: int = 200) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "Timestamp": ts,
            "kWh_received_Total": 0.5 + 0.1 * np.sin(np.arange(n) / 4),
            "Household_ID": "H1",
        }
    )


class TestSchema(unittest.TestCase):
    def test_kwh_is_index_zero(self):
        self.assertEqual(FEATURE_COLUMNS[0], "kWh_received_Total")
        self.assertEqual(INPUT_DIM, len(FEATURE_COLUMNS))


class TestFeaturePipeline(unittest.TestCase):
    def test_daily_spark_config_uses_extended_schema(self):
        cfg = ForecastConfig.for_daily_spark(seq_len=96, pred_len=96)
        self.assertEqual(tuple(cfg.feature_columns), DAILY_FEATURE_COLUMNS)
        self.assertEqual(cfg.feature_version, "spark_daily_v1")

    def test_build_feature_frame_columns(self):
        df = _make_meter_frame()
        cfg = ForecastConfig(seq_len=24, pred_len=4, use_weather=False)
        out = build_feature_frame(df, cfg, household_id="H1")
        for c in FEATURE_COLUMNS:
            self.assertIn(c, out.columns)

    def test_feature_frame_to_matrix_shape(self):
        df = _make_meter_frame()
        cfg = ForecastConfig(seq_len=24, pred_len=4, use_weather=False)
        out = build_feature_frame(df, cfg, household_id="H1")
        m = feature_frame_to_matrix(out, FEATURE_COLUMNS)
        self.assertEqual(m.shape[1], INPUT_DIM)

    def test_temporal_features_in_unit_circle(self):
        df = _make_meter_frame(50)
        out = add_temporal_features(df.copy())
        for c in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
            self.assertTrue((out[c].abs() <= 1.0 + 1e-6).all())

    def test_config_round_trip(self):
        import tempfile
        cfg = ForecastConfig(seq_len=64, pred_len=8, use_weather=True)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            save_forecast_config(cfg, f.name)
            path = f.name
        loaded = load_forecast_config(path)
        self.assertEqual(loaded.seq_len, 64)
        self.assertEqual(loaded.pred_len, 8)
        self.assertEqual(tuple(loaded.feature_columns), FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
