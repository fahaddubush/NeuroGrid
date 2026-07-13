"""
S - Sensing module (ISMCC Algorithm 2, Diagram 2 left column).

Reads raw smart-meter readings (kWh, temperature, humidity, sunshine), runs
preprocessing, and emits the canonical observation o_t. Drift detection lives
here as well: an exponentially-weighted moving average + 2σ test that flags
sensor anomalies upstream of the X module so the agent can react.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.data.schema import FEATURE_COLUMNS


@dataclass
class DriftState:
    ema_kwh: float | None = None
    ema_var: float | None = None
    alpha: float = 0.05


def _temporal(ts) -> tuple[float, float, float, float, float]:
    """Return (hour_sin, hour_cos, dow_sin, dow_cos, is_weekend) for a timestamp."""
    if not hasattr(ts, "hour"):
        import pandas as pd
        ts = pd.Timestamp(ts)
    hour = ts.hour + ts.minute / 60.0
    dow = ts.dayofweek
    return (
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
        math.sin(2 * math.pi * dow / 7.0),
        math.cos(2 * math.pi * dow / 7.0),
        1.0 if dow >= 5 else 0.0,
    )


class SensingModule:
    """Turn a raw stream reading into the canonical observation o_t.

    `preprocess(reading)` returns a dict that contains:
      * `kwh`, `timestamp`, `household_id` (forwarded from the reading)
      * `feature_vector`: np.ndarray of length INPUT_DIM, ordered per
        FEATURE_COLUMNS - feed-ready for the LSTM after scaling
    """

    def __init__(self):
        self.drift = DriftState()

    def preprocess(self, reading: dict) -> dict:
        kwh = float(reading.get("kwh", 0.0))
        ts = reading["timestamp"]
        hour_sin, hour_cos, dow_sin, dow_cos, is_weekend = _temporal(ts)
        temp = float(reading.get("Temperature_avg_hourly", 0.0))
        hum = float(reading.get("Humidity_avg_hourly", 0.0))
        sun = float(reading.get("Sunshine_duration_hourly", 0.0))
        weather_missing = 1.0 if (temp == 0.0 and hum == 0.0 and sun == 0.0) else 0.0
        load_missing = float(reading.get("load_missing", 0.0))

        values = {
            "kWh_received_Total": kwh,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "is_weekend": is_weekend,
            "Temperature_avg_hourly": temp,
            "Humidity_avg_hourly": hum,
            "Sunshine_duration_hourly": sun,
            "weather_missing": weather_missing,
            "load_missing": load_missing,
        }
        feature_vector = np.array([values[c] for c in FEATURE_COLUMNS], dtype=np.float32)
        return {
            "timestamp": str(ts),
            "household_id": reading.get("household_id"),
            "kwh": kwh,
            "feature_vector": feature_vector,
            "raw": values,
        }

    def detect_drift(self, observation: dict) -> bool:
        kwh = observation["kwh"]
        if self.drift.ema_kwh is None:
            self.drift.ema_kwh = kwh
            self.drift.ema_var = 0.01
            return False
        deviation = kwh - self.drift.ema_kwh
        std = max(self.drift.ema_var ** 0.5, 0.01)
        a = self.drift.alpha
        self.drift.ema_kwh = (1 - a) * self.drift.ema_kwh + a * kwh
        self.drift.ema_var = (1 - a) * self.drift.ema_var + a * deviation ** 2
        return abs(deviation) > 2.0 * std
