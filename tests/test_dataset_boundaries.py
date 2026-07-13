from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import NeuroGridDataset, SparkFeatureSchemaError
from src.data.feature_pipeline import ForecastConfig
from src.data.schema import DAILY_FEATURE_COLUMNS


_TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "test_tmp"
_TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _fresh_tmp_dir() -> Path:
    path = _TEST_TMP_ROOT / f"dataset_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_household_boundaries_reset_window_generation() -> None:
    tmp = _fresh_tmp_dir()
    try:
        raw_dir = tmp / "raw"
        parquet_dir = tmp / "processed_parquet_15min"
        raw_dir.mkdir(parents=True, exist_ok=True)
        parquet_dir.mkdir(parents=True, exist_ok=True)

        def _household_frame(hid: str) -> pd.DataFrame:
            n = 200  # 200 - 96 - 96 + 1 = 9 windows per household
            ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
            hour = ts.hour + ts.minute / 60.0
            dow = ts.dayofweek
            df = pd.DataFrame(
                {
                    "Timestamp": ts,
                    "Household_ID": hid,
                    "kWh_received_Total": np.linspace(0.1, 1.0, n, dtype=np.float32),
                    "Temperature_avg_hourly": np.zeros(n, dtype=np.float32),
                    "Humidity_avg_hourly": np.zeros(n, dtype=np.float32),
                    "Sunshine_duration_hourly": np.zeros(n, dtype=np.float32),
                    "load_missing": np.zeros(n, dtype=np.float32),
                    "weather_missing": np.zeros(n, dtype=np.float32),
                    "hour_sin": np.sin(2 * np.pi * hour / 24.0).astype(np.float32),
                    "hour_cos": np.cos(2 * np.pi * hour / 24.0).astype(np.float32),
                    "dow_sin": np.sin(2 * np.pi * dow / 7.0).astype(np.float32),
                    "dow_cos": np.cos(2 * np.pi * dow / 7.0).astype(np.float32),
                    "is_weekend": (dow >= 5).astype(np.float32),
                    "lag_1": np.linspace(0.1, 1.0, n, dtype=np.float32),
                    "lag_4": np.linspace(0.1, 1.0, n, dtype=np.float32),
                    "lag_96": np.linspace(0.1, 1.0, n, dtype=np.float32),
                    "roll_mean_4": np.linspace(0.1, 1.0, n, dtype=np.float32),
                    "roll_std_4": np.zeros(n, dtype=np.float32),
                    "roll_max_96": np.linspace(0.1, 1.0, n, dtype=np.float32),
                }
            )
            return df

        df = pd.concat([_household_frame("H1"), _household_frame("H2")], ignore_index=True)
        df.to_parquet(parquet_dir / "data.parquet", index=False)

        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "selected": [
                        {"household_id": "H1", "cluster": 0},
                        {"household_id": "H2", "cluster": 0},
                    ]
                }
            ),
            encoding="utf-8",
        )

        ds = NeuroGridDataset(
            ForecastConfig.for_daily_spark(seq_len=96, pred_len=96, use_weather=False),
            data_dir=str(raw_dir),
            split="train",
            manifest_path=str(manifest_path),
            train_n=2,
            val_n=0,
            test_n=0,
        )

        # If households were concatenated before windowing, we'd see
        # 400 - 96 - 96 + 1 = 209 windows. Correct boundary-protected
        # behaviour resets per household, giving 9 + 9 = 18.
        assert len(ds) == 18
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_daily_dataset_fails_loudly_when_spark_columns_missing() -> None:
    tmp = _fresh_tmp_dir()
    try:
        raw_dir = tmp / "raw"
        parquet_dir = tmp / "processed_parquet_15min"
        raw_dir.mkdir(parents=True, exist_ok=True)
        parquet_dir.mkdir(parents=True, exist_ok=True)
        ts = pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC")
        pd.DataFrame(
            {
                "Timestamp": ts,
                "Household_ID": "H1",
                "kWh_received_Total": np.linspace(0.1, 1.0, 200, dtype=np.float32),
                "Temperature_avg_hourly": np.zeros(200, dtype=np.float32),
                "Humidity_avg_hourly": np.zeros(200, dtype=np.float32),
                "Sunshine_duration_hourly": np.zeros(200, dtype=np.float32),
                "load_missing": np.zeros(200, dtype=np.float32),
                "weather_missing": np.zeros(200, dtype=np.float32),
            }
        ).to_parquet(parquet_dir / "data.parquet", index=False)

        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(
            json.dumps({"selected": [{"household_id": "H1", "cluster": 0}]}),
            encoding="utf-8",
        )

        try:
            NeuroGridDataset(
                ForecastConfig.for_daily_spark(seq_len=96, pred_len=96, use_weather=False),
                data_dir=str(raw_dir),
                split="train",
                manifest_path=str(manifest_path),
                train_n=1,
                val_n=0,
                test_n=0,
            )
            raise AssertionError("Expected SparkFeatureSchemaError")
        except SparkFeatureSchemaError as exc:
            for col in ("hour_sin", "lag_96", "roll_max_96"):
                assert col in str(exc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
