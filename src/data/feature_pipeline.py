"""
Feature pipeline utilities shared by training, evaluation, and tests.

The official daily training path is Spark-first: Spark ETL owns preprocessing
and feature engineering, and the dataset loader reads Spark-materialized
Parquet directly. The pandas helpers in this module remain for config
serialization and legacy/unit-test utilities, not for the daily training path.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.schema import (
    DAILY_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class ForecastConfig:
    seq_len: int = 96  # 24 hours of 15-min steps
    pred_len: int = 1
    use_weather: bool = True
    feature_columns: tuple = FEATURE_COLUMNS
    feature_version: str = "ismc_v1"
    # Per-household z-score normalization. Each household's feature matrix is
    # standardized by its own mean/std before the global scaler sees it.
    # Removes inter-household amplitude/baseline shift - the dominant cause
    # of Tloss << Vloss when training on a small representative cohort with
    # household-disjoint val/test splits.
    # Disabled by default: fitting normalization on a household's complete
    # timeline leaks future statistics and cannot be reproduced online.
    per_household_norm: bool = False

    @property
    def input_dim(self) -> int:
        return len(self.feature_columns)

    @classmethod
    def with_extended_features(cls, **kwargs) -> "ForecastConfig":
        """Build a config wired to the Spark-managed 17-dim schema."""
        kwargs.setdefault("feature_columns", DAILY_FEATURE_COLUMNS)
        kwargs.setdefault("feature_version", "ismc_v1_ext")
        return cls(**kwargs)

    @classmethod
    def for_daily_spark(cls, **kwargs) -> "ForecastConfig":
        kwargs.setdefault("feature_columns", DAILY_FEATURE_COLUMNS)
        kwargs.setdefault("feature_version", "spark_daily_v1")
        return cls(**kwargs)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    hour = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60.0
    dow = df["Timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["is_weekend"] = (dow >= 5).astype(np.float32)
    return df


def attach_weather(df: pd.DataFrame, household_id: str | None, use_weather: bool) -> pd.DataFrame:
    weather_cols = ["Temperature_avg_hourly", "Humidity_avg_hourly", "Sunshine_duration_hourly"]
    if use_weather and household_id is not None:
        try:
            from src.data.weather import join_weather_to_meter
            df = join_weather_to_meter(df, household_id)
        except Exception as exc:
            logging.warning("Weather join failed for household %s: %s", household_id, exc)
    for c in weather_cols:
        if c not in df.columns:
            df[c] = np.nan
    weather_missing = df[weather_cols].isna().any(axis=1).astype(np.float32)
    df["weather_missing"] = weather_missing
    for c in weather_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag_1/4/96 + roll_mean_4 / roll_std_4 / roll_max_96 if missing.

    Pandas fallback used by the dataset when the Parquet
    store was produced WITHOUT --extended_features. When the columns already
    exist (Spark precomputed them) this is a no-op.
    """
    if "kWh_received_Total" not in df.columns:
        return df
    src = pd.to_numeric(df["kWh_received_Total"], errors="coerce")
    if "lag_1" not in df.columns:
        df["lag_1"] = src.shift(1).fillna(src.iloc[0] if len(src) else 0.0).astype(np.float32)
    if "lag_4" not in df.columns:
        df["lag_4"] = src.shift(4).bfill().fillna(0.0).astype(np.float32)
    if "lag_96" not in df.columns:
        df["lag_96"] = src.shift(96).bfill().fillna(0.0).astype(np.float32)
    if "roll_mean_4" not in df.columns:
        df["roll_mean_4"] = src.rolling(window=4, min_periods=1).mean().fillna(0.0).astype(np.float32)
    if "roll_std_4" not in df.columns:
        df["roll_std_4"] = src.rolling(window=4, min_periods=1).std().fillna(0.0).astype(np.float32)
    if "roll_max_96" not in df.columns:
        df["roll_max_96"] = src.rolling(window=96, min_periods=1).max().fillna(0.0).astype(np.float32)
    return df


def winsorize_kwh(df: pd.DataFrame, lower_q: float = 0.001, upper_q: float = 0.999) -> pd.DataFrame:
    if df.empty or "kWh_received_Total" not in df.columns:
        return df
    series = pd.to_numeric(df["kWh_received_Total"], errors="coerce")
    finite = series[series.notna()]
    if len(finite) < 10:
        return df
    lo = max(0.0, float(finite.quantile(lower_q)))
    hi = float(finite.quantile(upper_q))
    if hi <= lo:
        return df
    df["kWh_received_Total"] = series.clip(lower=lo, upper=hi)
    return df


def build_feature_frame(
    df: pd.DataFrame,
    config: ForecastConfig,
    household_id: str | None = None,
) -> pd.DataFrame:
    """Produce a feature DataFrame with exactly `config.feature_columns` populated."""
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df["kWh_received_Total"] = pd.to_numeric(df["kWh_received_Total"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("Input meter data is empty after timestamp cleaning.")

    if household_id is None and "Household_ID" in df.columns:
        household_id = str(df["Household_ID"].iloc[0])

    df["load_missing"] = df["kWh_received_Total"].isna().astype(np.float32)
    df["kWh_received_Total"] = df["kWh_received_Total"].ffill().fillna(0.0).astype(np.float32)

    df = winsorize_kwh(df)
    df = attach_weather(df, household_id, config.use_weather)
    df = add_temporal_features(df)
    df = df.dropna(subset=list(config.feature_columns)).reset_index(drop=True)
    return df


def feature_frame_to_matrix(df: pd.DataFrame, feature_columns) -> np.ndarray:
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Feature frame missing required columns: {missing}")
    matrix = df.loc[:, list(feature_columns)].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature frame contains NaN or infinity.")
    return matrix


def save_forecast_config(config: ForecastConfig, path: str) -> None:
    payload = asdict(config)
    payload["feature_columns"] = list(config.feature_columns)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_forecast_config(path: str) -> ForecastConfig:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["feature_columns"] = tuple(payload["feature_columns"])
    return ForecastConfig(**payload)


def save_scaler(scaler, path: str) -> None:
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Only sklearn StandardScaler artifacts are supported.")
    with open(path, "wb") as f:
        np.savez_compressed(
            f,
            format_version=np.array([1], dtype=np.int64),
            mean=np.asarray(scaler.mean_, dtype=np.float64),
            var=np.asarray(scaler.var_, dtype=np.float64),
            scale=np.asarray(scaler.scale_, dtype=np.float64),
            n_samples_seen=np.asarray(scaler.n_samples_seen_),
            n_features_in=np.array([scaler.n_features_in_], dtype=np.int64),
        )


def load_scaler(path: str):
    try:
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["format_version"][0]) != 1:
                raise ValueError("Unsupported scaler artifact version.")
            scaler = StandardScaler()
            scaler.mean_ = payload["mean"]
            scaler.var_ = payload["var"]
            scaler.scale_ = payload["scale"]
            scaler.n_samples_seen_ = payload["n_samples_seen"]
            scaler.n_features_in_ = int(payload["n_features_in"][0])
            return scaler
    except (ValueError, KeyError, OSError):
        if os.getenv("NEUROGRID_ALLOW_LEGACY_PICKLE", "0") != "1":
            raise ValueError(
                "Unsafe or legacy scaler artifact rejected. Retrain/export as scaler.npz, "
                "or explicitly set NEUROGRID_ALLOW_LEGACY_PICKLE=1 for trusted local files."
            )
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)


__all__ = [
    "ForecastConfig",
    "add_temporal_features",
    "attach_weather",
    "winsorize_kwh",
    "build_feature_frame",
    "feature_frame_to_matrix",
    "save_forecast_config",
    "load_forecast_config",
    "save_scaler",
    "load_scaler",
]
