"""
NeuroGridDataset - Parquet-backed windowed dataset for the city LSTM.

The dataset always reads from the Spark-produced Parquet store
(`<heapo_root>/processed_parquet_15min`). If the Parquet store is missing the
dataset raises immediately so silent fallbacks cannot mask a stale ETL.

Two split modes are supported:

1. **Cluster-stratified, household-disjoint (preferred).**
   When a sampling manifest is supplied (`manifest_path=`), the dataset:
     * restricts the household population to the manifest cohort
       (default target: target_n=30, k=3 KMeans clusters);
     * carves train / val / test into household-disjoint groups using
       `cluster_stratified_household_split` (default 24/3/3);
     * uses the **full timeline** of each household within its assigned
       split (no temporal cut needed - splits are already disjoint by
       household, so no leakage).
   With the default configuration, each KMeans cluster is represented in
   validation and test (one household per cluster when k=3 and
   val_n=test_n=3).

2. **Legacy temporal / hash split (fallback when no manifest).**
   When `manifest_path is None`, the original behaviour applies - train/val
   use a household-hash split and test uses the temporal tail of every
   household.
"""
from __future__ import annotations

import os
import hashlib
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.data.feature_pipeline import (
    ForecastConfig,
    feature_frame_to_matrix,
)
from src.data.representative_sampling import (
    cluster_stratified_household_split,
    load_manifest_with_clusters,
)
from src.data.schema import DAILY_FEATURE_COLUMNS, FEATURE_COLUMNS
from src.utils.paths import resolve_path
from src.data.publication import resolve_current_dataset

# Cohort defaults (target_n=30, k=3, 24/3/3) are centralized here so trainer /
# evaluator / CLI can rely on the same numbers without re-declaring them.
DEFAULT_TRAIN_N = 24
DEFAULT_VAL_N = 3
DEFAULT_TEST_N = 3


def _processed_parquet_dir(heapo_data_dir: str) -> str:
    """Resolve the Parquet output directory produced by `src.data.spark_etl`."""
    root = os.path.join(os.path.dirname(heapo_data_dir), "processed_parquet_15min")
    return str(resolve_current_dataset(root))


class SparkFeatureSchemaError(ValueError):
    """Raised when the Parquet store lacks the Spark-materialized feature schema."""


class NeuroGridDataset(Dataset):
    def __init__(
        self,
        config: ForecastConfig,
        data_dir: str | None = None,
        split: str = "train",
        train_ratio: float = 0.8,
        scaler: StandardScaler | None = None,
        max_households: int | None = None,
        val_household_fraction: float = 0.15,
        global_mode: bool = True,
        manifest_path: str | os.PathLike | None = None,
        split_seed: int = 0,
        train_n: int = DEFAULT_TRAIN_N,
        val_n: int = DEFAULT_VAL_N,
        test_n: int = DEFAULT_TEST_N,
    ):
        # 'test' is a held-out tail (last 10% of each
        # household's timeline). 'train' uses the first 80%, 'val' the next
        # 10%, 'test' the final 10% - ensures no temporal leakage and lets
        # us report a model-quality number on data the model never saw.
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'.")
        self.config = config
        self.split = split
        self.train_ratio = train_ratio
        self.max_households = max_households
        self.val_household_fraction = val_household_fraction
        self.global_mode = global_mode
        self.split_seed = int(split_seed)

        raw_data_dir = data_dir or os.getenv("HEAPO_DATA_DIR")
        if not raw_data_dir:
            raise ValueError("HEAPO_DATA_DIR not set and no data_dir provided.")
        self.data_dir = str(resolve_path(raw_data_dir))

        self.parquet_dir = _processed_parquet_dir(self.data_dir)
        if not os.path.isdir(self.parquet_dir):
            raise FileNotFoundError(
                f"Parquet store not found at {self.parquet_dir}. "
                "Run `python -m src.data.spark_etl` first."
            )

        self.scaler = scaler
        self._fit_scaler = scaler is None and split == "train"
        self._daily_spark_mode = tuple(config.feature_columns) == tuple(DAILY_FEATURE_COLUMNS)

        # Use a cluster-stratified household-disjoint split when a
        # sampling manifest is provided. The manifest carries the KMeans
        # cluster id for each selected household; we use that to assign
        # whole households (no temporal leakage) to train/val/test.
        self.manifest_path = str(resolve_path(manifest_path)) if manifest_path else None
        self._allowed_household_ids: set[str] | None = None
        self._stratified_split: dict[str, list[str]] | None = None
        if self.manifest_path:
            assignments = load_manifest_with_clusters(self.manifest_path)
            if not assignments:
                raise ValueError(f"Manifest at {self.manifest_path} is empty.")
            self._stratified_split = cluster_stratified_household_split(
                assignments,
                train_n=train_n,
                val_n=val_n,
                test_n=test_n,
                seed=split_seed,
            )
            self._allowed_household_ids = set(self._stratified_split[split])

        self._chunks: list[np.ndarray] = []
        self._index_map: list[tuple[int, int]] = []
        self._selected_household_count = 0
        self._load()

    @staticmethod
    def _household_in_val(hid, val_fraction: float, seed: int = 0) -> bool:
        digest = hashlib.sha256(
            f"ismc_split_v2:{int(seed)}:{hid}".encode("utf-8")
        ).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        return value < val_fraction

    def _iter_household_frames(self) -> Iterable[tuple[str, pd.DataFrame]]:
        dataset = pa_dataset.dataset(self.parquet_dir, format="parquet", partitioning="hive")
        self._validate_required_columns(dataset.schema.names)
        ids = (
            dataset.to_table(columns=["Household_ID"])
            .column("Household_ID")
            .to_pandas()
            .dropna()
            .unique()
            .tolist()
        )
        ids.sort()
        if self._allowed_household_ids is not None:
            ids = [hid for hid in ids if str(hid) in self._allowed_household_ids]
        elif self.max_households is not None:
            ids = ids[: self.max_households]
        if not self.global_mode and ids:
            ids = ids[:1]
        self._selected_household_count = len(ids)

        wanted_cols = ["Timestamp", "Household_ID", *self.config.feature_columns]

        for hid in ids:
            tbl = dataset.to_table(
                filter=(pa_dataset.field("Household_ID") == hid),
                columns=wanted_cols,
            ).to_pandas()
            if tbl.empty:
                continue
            tbl["Timestamp"] = pd.to_datetime(tbl["Timestamp"], utc=True, errors="coerce")
            tbl = tbl.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
            yield hid, tbl

    def _validate_required_columns(self, present_columns: Iterable[str]) -> None:
        present = set(present_columns)
        required = {"Timestamp", "Household_ID", *self.config.feature_columns}
        missing = sorted(required - present)
        if not missing:
            return
        if self._daily_spark_mode:
            raise SparkFeatureSchemaError(
                "Spark-complete daily schema missing required column(s): "
                f"{missing}. Rerun `python -m src.cli etl --household_manifest <manifest>` "
                "to regenerate the canonical Parquet store."
            )
        raise SparkFeatureSchemaError(
            f"Parquet store missing required column(s) for feature_version={self.config.feature_version}: {missing}"
        )

    def _load(self) -> None:
        raw_chunks: list[np.ndarray] = []

        # Single-pass load: count households incrementally to decide the split
        # strategy, avoiding a redundant full Parquet scan.
        n_hh = 0
        for hid, hdf in self._iter_household_frames():
            n_hh += 1
            use_household_split = self.global_mode and self._selected_household_count >= 2

            hdf = hdf.dropna(subset=list(self.config.feature_columns)).reset_index(drop=True)
            if hdf.empty:
                continue

            matrix = feature_frame_to_matrix(hdf, self.config.feature_columns)

            # Per-household z-score: standardize each household by its own
            # mean/std before the global scaler. Removes inter-household
            # amplitude/baseline shift so the model only learns normalized
            # shape - primary fix for Vloss >> Tloss on a small cohort with
            # household-disjoint val/test splits.
            if getattr(self.config, 'per_household_norm', False):
                raise ValueError(
                    "per_household_norm fits on a complete timeline and is disabled because "
                    "it leaks future statistics and cannot be reproduced online."
                )

            # Split selection:
            #   * manifest path: household is in self._allowed_household_ids
            #     for the requested split, so we already filtered the population.
            #     Use the FULL timeline - no temporal cut needed because splits
            #     are household-disjoint, so there's no leakage.
            #   * Legacy fallback (no manifest): retains the prior temporal /
            #     household-hash behaviour for backward compatibility.
            if self._allowed_household_ids is None:
                n = len(matrix)
                train_cut = int(n * 0.8)
                val_cut = int(n * 0.9)
                if use_household_split and self.split in {"train", "val"}:
                    in_val = self._household_in_val(
                        hid, self.val_household_fraction, self.split_seed
                    )
                    if (in_val and self.split != "val") or (not in_val and self.split != "train"):
                        continue
                else:
                    if self.split == "train":
                        matrix = matrix[:train_cut]
                    elif self.split == "val":
                        matrix = matrix[max(0, train_cut - self.config.seq_len):val_cut]
                    else:  # test
                        matrix = matrix[max(0, val_cut - self.config.seq_len):]

            # Boundary protection: each household stays in its own chunk.
            if len(matrix) < self.config.seq_len + self.config.pred_len:
                continue
            raw_chunks.append(matrix)

        if not raw_chunks:
            raise ValueError(
                f"No usable windows for split='{self.split}'. Check Parquet content "
                f"and seq_len/pred_len ({self.config.seq_len}/{self.config.pred_len})."
            )

        if self._fit_scaler:
            self.scaler = StandardScaler()
            for chunk in raw_chunks:
                self.scaler.partial_fit(chunk)

        for chunk in raw_chunks:
            if self.scaler is not None:
                chunk = self.scaler.transform(chunk).astype(np.float32)
            if not np.isfinite(chunk).all():
                raise ValueError("Scaler produced NaN or infinity.")
            chunk_idx = len(self._chunks)
            self._chunks.append(chunk)
            n_windows = len(chunk) - self.config.seq_len - self.config.pred_len + 1
            for offset in range(n_windows):
                self._index_map.append((chunk_idx, offset))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int):
        chunk_idx, off = self._index_map[idx]
        m = self._chunks[chunk_idx]
        x = m[off : off + self.config.seq_len]
        y = m[off + self.config.seq_len : off + self.config.seq_len + self.config.pred_len, 0]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def target_quantile(self, quantile: float) -> float:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1).")
        return float(np.quantile(np.concatenate([chunk[:, 0] for chunk in self._chunks]), quantile))


def get_dataloader(
    config: ForecastConfig,
    batch_size: int = 32,
    split: str = "train",
    scaler=None,
    max_households: int | None = None,
    data_dir: str | None = None,
    manifest_path: str | os.PathLike | None = None,
) -> DataLoader:
    ds = NeuroGridDataset(
        config=config,
        split=split,
        scaler=scaler,
        max_households=max_households,
        data_dir=data_dir,
        manifest_path=manifest_path,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=split == "train", num_workers=0)
