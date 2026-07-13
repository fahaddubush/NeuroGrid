"""
Weather data integration module.
Joins hourly weather data to 15-min smart meter data via forward-fill.
Uses meta_data/households.csv to map Household_ID → Weather_ID.

Loading is lazy: only the requested Weather_ID's rows are read into memory,
keyed by an index over the available weather files. The previous
all-corpus-into-RAM eager load has been removed.
"""
import os
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

# Module-level caches (small)
_household_map_cache: dict[str, dict] = {}
_weather_index_cache: dict[str, dict[str, str]] = {}  # heapo_root -> {weather_id: file_path}


def _get_heapo_root(heapo_root: str = None) -> str:
    if heapo_root is None:
        heapo_root = os.getenv("HEAPO_DATA_DIR", "")
        heapo_root = str(Path(heapo_root).parent.parent)
    return heapo_root


def load_household_weather_map(heapo_root: str = None) -> dict:
    """Load the Household_ID → Weather_ID mapping from metadata. Cached."""
    heapo_root = _get_heapo_root(heapo_root)

    if heapo_root in _household_map_cache:
        return _household_map_cache[heapo_root]

    meta_path = os.path.join(heapo_root, "meta_data", "households.csv")
    if not os.path.exists(meta_path):
        _household_map_cache[heapo_root] = {}
        return {}

    df = pd.read_csv(meta_path, sep=";")
    mapping = dict(zip(df["Household_ID"].astype(str), df["Weather_ID"]))
    _household_map_cache[heapo_root] = mapping
    return mapping


def _build_weather_index(heapo_root: str) -> dict[str, str]:
    """Map each Weather_ID to the CSV file that contains its rows. Cheap to build."""
    if heapo_root in _weather_index_cache:
        return _weather_index_cache[heapo_root]

    weather_dir = os.path.join(heapo_root, "weather_data", "hourly")
    index: dict[str, str] = {}
    if not os.path.isdir(weather_dir):
        _weather_index_cache[heapo_root] = index
        return index

    for fname in os.listdir(weather_dir):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(weather_dir, fname)
        # Read only Weather_ID column to discover which IDs live in this file.
        try:
            ids = pd.read_csv(path, sep=";", usecols=["Weather_ID"])["Weather_ID"].dropna().unique()
        except (OSError, ValueError, KeyError) as exc:
            logging.warning("Skipping unreadable weather file %s: %s", path, exc)
            continue
        for wid in ids:
            index.setdefault(str(wid), path)

    _weather_index_cache[heapo_root] = index
    return index


@lru_cache(maxsize=128)
def load_weather_data(weather_id: str, heapo_root: str = None) -> pd.DataFrame:
    """Load hourly weather data for a specific Weather_ID. Lazy + cached per ID."""
    heapo_root = _get_heapo_root(heapo_root)
    weather_id = str(weather_id)

    index = _build_weather_index(heapo_root)
    path = index.get(weather_id)
    if path is None:
        return pd.DataFrame()

    weather_cols = [
        "Temperature_avg_hourly",
        "Humidity_avg_hourly",
        "Sunshine_duration_hourly",
    ]

    df = pd.read_csv(path, sep=";")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["Timestamp"])
    df = df[df["Weather_ID"].astype(str) == weather_id]

    available = [c for c in weather_cols if c in df.columns]
    df = df.sort_values("Timestamp").set_index("Timestamp")[available]
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.resample("15min").ffill()
    return df


def join_weather_to_meter(
    meter_df: pd.DataFrame, household_id: str, heapo_root: str = None
) -> pd.DataFrame:
    """
    Join weather data to a smart meter DataFrame.
    meter_df must have a 'Timestamp' column (datetime).
    Returns the meter_df with weather columns appended.
    """
    heapo_root = _get_heapo_root(heapo_root)
    mapping = load_household_weather_map(heapo_root)
    weather_id = mapping.get(str(household_id))

    default_cols = [
        "Temperature_avg_hourly",
        "Humidity_avg_hourly",
        "Sunshine_duration_hourly",
    ]

    if not weather_id:
        for col in default_cols:
            meter_df[col] = pd.NA
        return meter_df

    weather_df = load_weather_data(weather_id, heapo_root)
    if weather_df.empty:
        for col in default_cols:
            meter_df[col] = pd.NA
        return meter_df

    meter_df = meter_df.set_index("Timestamp")
    meter_df = meter_df.join(weather_df, how="left")

    for col in default_cols:
        if col not in meter_df.columns:
            meter_df[col] = pd.NA

    meter_df = meter_df.reset_index()
    return meter_df
