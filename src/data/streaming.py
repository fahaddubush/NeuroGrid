"""
Real-time HEAPO smart-meter stream used by the building tier S module.

Reads one row at a time from a per-household CSV and joins weather columns at
read time so the on-line stream and the offline Parquet store expose the same
schema.
"""
from __future__ import annotations

import math
import os
from typing import Iterator

import pandas as pd

from src.data.weather import join_weather_to_meter


def _safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


class HouseholdStream:
    """Iterator that yields one normalised reading per call.

    The reading dict matches the canonical FEATURE_COLUMNS schema (kWh +
    weather), plus timestamp / household_id metadata. The S module turns this
    into the observation o_t.
    """

    def __init__(self, csv_path: str, household_id: str | None = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Stream CSV not found: {csv_path}")
        self.csv_path = csv_path
        self.household_id = household_id
        self._iter: Iterator[pd.DataFrame] = pd.read_csv(
            csv_path, sep=";", parse_dates=["Timestamp"], chunksize=1
        )

    def next_reading(self) -> dict | None:
        try:
            chunk = next(self._iter)
        except StopIteration:
            return None

        row = chunk.iloc[0]
        hid = self.household_id or str(row.get("Household_ID", "unknown"))

        chunk_with_weather = join_weather_to_meter(chunk, hid)
        wrow = chunk_with_weather.iloc[0]

        ts = row["Timestamp"]
        ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        kwh_raw = row.get("kWh_received_Total")
        load_missing = 1.0 if pd.isna(kwh_raw) else 0.0
        kwh = _safe_float(kwh_raw, default=0.0)

        return {
            "timestamp": ts_iso,
            "household_id": hid,
            "kwh": kwh,
            "load_missing": load_missing,
            "Temperature_avg_hourly": _safe_float(wrow.get("Temperature_avg_hourly")),
            "Humidity_avg_hourly": _safe_float(wrow.get("Humidity_avg_hourly")),
            "Sunshine_duration_hourly": _safe_float(wrow.get("Sunshine_duration_hourly")),
        }

    def __iter__(self):
        return self

    def __next__(self):
        r = self.next_reading()
        if r is None:
            raise StopIteration
        return r
