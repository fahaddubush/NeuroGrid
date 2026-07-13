"""
Tests for the Spark FL runner's driver-side helpers.

Spinning a real Spark+TorchDistributor session inside pytest is flaky on
Windows (port allocation, JVM lifetime). We test the driver-side logic that
does NOT require Spark - manifest discovery, district aggregation, byzantine
rank selection - and let the integration tests run via the CLI on real data.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.simulation.spark_fl_runner import (
    SparkFLConfig,
    _aggregate_district,
    _discover_agent_manifest,
    _initial_global_state,
    _select_byzantine_ranks,
)


_TMP_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "test_tmp"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _fresh_tmp() -> Path:
    p = _TMP_ROOT / f"sparkfl_runner_{uuid.uuid4().hex}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _make_fake_parquet_root(root: Path, n_households: int = 4) -> None:
    """Build a (Weather_ID, Household_ID)-partitioned tree with empty parquets."""
    for i in range(n_households):
        d = root / f"Weather_ID=W{i % 2}" / f"Household_ID=H{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        # Empty placeholder parquet file - discovery only checks dir presence.
        pd.DataFrame({"x": [0.0]}).to_parquet(d / "part-00000.parquet", index=False)


def test_discover_agent_manifest_assigns_ranks_and_districts():
    tmp = _fresh_tmp()
    try:
        root = tmp / "parquet"
        _make_fake_parquet_root(root, n_households=6)
        df = _discover_agent_manifest(
            parquet_root=str(root), num_agents=4, num_districts=2
        )
        assert len(df) == 4
        required = {"rank", "agent_id", "district_id", "parquet_path", "selection_mode"}
        assert required.issubset(set(df.columns))
        # Round-robin district assignment: ranks 0,2 → D1; 1,3 → D2.
        assert set(df["district_id"]) == {"D1", "D2"}
        # Ranks are 0..N-1 contiguously.
        assert sorted(df["rank"].tolist()) == [0, 1, 2, 3]
        # Without a manifest, runner falls back to alphabetical (with warning).
        assert (df["selection_mode"] == "alphabetical_fallback").all()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_discover_agent_manifest_uses_mllib_manifest_for_cohort():
    """When a sampling manifest is provided, cohort + cluster come from it."""
    import json as _json
    tmp = _fresh_tmp()
    try:
        root = tmp / "parquet"
        _make_fake_parquet_root(root, n_households=6)  # H000..H005

        # Synthesize a sampling manifest selecting 4 households across 2 clusters.
        manifest = {
            "selected": [
                {"household_id": "H001", "cluster": 0, "rank_in_cluster": 0},
                {"household_id": "H003", "cluster": 0, "rank_in_cluster": 1},
                {"household_id": "H002", "cluster": 1, "rank_in_cluster": 0},
                {"household_id": "H004", "cluster": 1, "rank_in_cluster": 1},
            ]
        }
        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(_json.dumps(manifest), encoding="utf-8")

        df = _discover_agent_manifest(
            parquet_root=str(root),
            num_agents=4,
            num_districts=2,
            sampling_manifest_path=str(manifest_path),
        )
        assert len(df) == 4
        assert (df["selection_mode"] == "mllib_kmeans").all()
        # Cohort matches the manifest, not the alphabetical first-N.
        assert set(df["agent_id"]) == {"H001", "H002", "H003", "H004"}
        # Cluster IDs round-robined across districts: each district sees both clusters.
        per_district_clusters = df.groupby("district_id")["cluster"].nunique()
        assert (per_district_clusters > 1).all(), \
            "Each district should see >1 cluster (cluster-balanced assignment)."
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_discover_agent_manifest_raises_when_short():
    tmp = _fresh_tmp()
    try:
        root = tmp / "parquet"
        _make_fake_parquet_root(root, n_households=2)
        with pytest.raises(ValueError, match="Need 8 households"):
            _discover_agent_manifest(str(root), num_agents=8, num_districts=2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_select_byzantine_ranks_deterministic():
    a = _select_byzantine_ranks(n=10, fraction=0.3, seed=0)
    b = _select_byzantine_ranks(n=10, fraction=0.3, seed=0)
    assert a == b
    assert len(a) == 3
    c = _select_byzantine_ranks(n=10, fraction=0.0, seed=0)
    assert c == set()


def test_aggregate_district_keeps_honest_majority():
    """Krum should reject a single 10x-scaled outlier among honest agents."""
    rng = torch.Generator().manual_seed(0)

    def _honest_delta() -> dict[str, torch.Tensor]:
        return {"w": torch.empty(8).normal_(0.0, 0.1, generator=rng)}

    honest = [_honest_delta() for _ in range(4)]
    attacker = {"w": _honest_delta()["w"] * 50.0}
    deltas = [
        {
            "state": delta,
            "masks": {key: torch.ones_like(value, dtype=torch.bool) for key, value in delta.items()},
        }
        for delta in honest + [attacker]
    ]
    n_samples = [10, 10, 10, 10, 10]

    agg, agg_mask, summary = _aggregate_district(
        deltas=deltas,
        n_samples=n_samples,
        f_byzantine=1,
        clip_threshold=10.0,  # high enough to leave the attack visible to Krum
        trim_ratio=0.0,
    )
    assert agg is not None
    assert agg_mask is not None
    assert summary["n_total"] == 5
    # With f=1, Multi-Krum keeps n - f = 4 survivors, ideally rejecting the attacker.
    assert summary["n_accepted"] == 4
    accepted = set(summary["accepted_idx"])
    assert 4 not in accepted, "Multi-Krum should have rejected the 50x outlier (idx 4)."


def test_aggregate_district_empty_returns_none():
    agg, agg_mask, summary = _aggregate_district(
        deltas=[], n_samples=[],
        f_byzantine=1, clip_threshold=1.0, trim_ratio=0.0,
    )
    assert agg is None
    assert agg_mask is None
    assert summary["n_total"] == 0


def test_initial_global_state_shapes_match_tier():
    cfg = SparkFLConfig(tier="building", pred_len=4)
    state = _initial_global_state(cfg)
    assert isinstance(state, dict) and len(state) > 0
    for v in state.values():
        assert isinstance(v, torch.Tensor)
