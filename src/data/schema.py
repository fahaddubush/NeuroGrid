"""
Canonical feature schema for the active city LSTM.

The legacy code carried two parallel schemas (9-feature 15-min "agent loop" and
37-feature hourly "weekly batch"). Per the ISMCC-MAS paper there is exactly one
LSTM family - the City teacher and the Building/District students share an
architecture and a feature space, only the model size differs.

This module is the single source of truth for that schema. Everything else
(streaming, dataset loader, training, distillation) reads from it.

Added EXTENDED_FEATURE_COLUMNS (17-dim with lag/rolling
columns from Spark) for richer features. The daily 96->96 forecasting path now
standardises on the Spark-managed extended schema.
"""
from __future__ import annotations

from dataclasses import dataclass

# Order matters: kWh must be index 0 because inverse-transform / forecast
# extraction read column 0.
FEATURE_COLUMNS = (
    "kWh_received_Total",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "Temperature_avg_hourly",
    "Humidity_avg_hourly",
    "Sunshine_duration_hourly",
    "weather_missing",
    "load_missing",
)
INPUT_DIM = len(FEATURE_COLUMNS)

# Extended feature set (opt-in). When the Parquet store was produced by
# `spark_etl --extended_features`, these six columns are present and a model
# can be trained against the richer 17-dim vector. Lag/rolling features are
# expensive to compute in pandas, cheap in Spark - that's the entire point.
LAG_FEATURE_COLUMNS = (
    "lag_1",
    "lag_4",
    "lag_96",
    "roll_mean_4",
    "roll_std_4",
    "roll_max_96",
)
EXTENDED_FEATURE_COLUMNS = FEATURE_COLUMNS + LAG_FEATURE_COLUMNS
EXTENDED_INPUT_DIM = len(EXTENDED_FEATURE_COLUMNS)
DAILY_FEATURE_COLUMNS = EXTENDED_FEATURE_COLUMNS
DAILY_INPUT_DIM = EXTENDED_INPUT_DIM

RESOLUTION_MINUTES = 15
STEPS_PER_HOUR = 60 // RESOLUTION_MINUTES
STEPS_PER_DAY = STEPS_PER_HOUR * 24


@dataclass(frozen=True)
class HorizonStage:
    """One stage of the gradual / curriculum forecasting pipeline."""

    name: str
    minutes: int

    @property
    def steps(self) -> int:
        return max(1, self.minutes // RESOLUTION_MINUTES)


CURRICULUM_STAGES: tuple[HorizonStage, ...] = (
    HorizonStage("h15min", 15),
    HorizonStage("h30min", 30),
    HorizonStage("h45min", 45),
    HorizonStage("h1h", 60),
    HorizonStage("h2h", 120),
    HorizonStage("h3h", 180),
)

# Daily / next-day forecasting stage (96 steps × 15 min = 24 h). Kept
# separate from the curriculum so the existing 6-stage pipeline is
# unchanged. Use via `python -m src.cli forecast-daily`.
DAILY_HORIZON = HorizonStage("h24h", 1440)
