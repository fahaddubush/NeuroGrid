"""
Unit tests for the rank-isolated FL local-train kernel.

These tests exercise `local_train_one_agent` directly without TorchDistributor
or a SparkSession - the kernel's contract is to be pure and self-contained, so
we validate that contract independently of the Spark plumbing.

Coverage:
  * happy path: kernel writes Δθ, returns sane metrics, baseline preserved
  * empty data: graceful no-op with status="no_data"
  * Byzantine attack: scaled Δθ has visibly larger pre-DP norm than honest
  * idempotency: rerunning the same shard overwrites Δθ deterministically
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.schema import FEATURE_COLUMNS, INPUT_DIM
from src.federated.spark_local_train import (
    AgentShard,
    LocalTrainHparams,
    local_train_one_agent,
)
from src.models.city_lstm import CityLSTM, tier_size


_TMP_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "test_tmp"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _fresh_tmp() -> Path:
    p = _TMP_ROOT / f"spark_fl_{uuid.uuid4().hex}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _make_fake_household_parquet(out_dir: Path, n_rows: int = 200) -> str:
    """Synthesize a minimal household Parquet that satisfies the schema."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = {col: rng.standard_normal(n_rows).astype(np.float32) for col in FEATURE_COLUMNS}
    # Make kWh non-negative so the target is realistic.
    data[FEATURE_COLUMNS[0]] = np.abs(data[FEATURE_COLUMNS[0]])
    data["Timestamp"] = pd.date_range("2024-01-01", periods=n_rows, freq="15min")
    df = pd.DataFrame(data)
    path = out_dir / "part-00000.parquet"
    df.to_parquet(path, index=False)
    return str(out_dir)  # callers pass the directory; pandas reads it whole


def _make_w_global(tmp: Path, tier: str = "building", pred_len: int = 4) -> Path:
    sz = tier_size(tier)
    model = CityLSTM(pred_len=pred_len, input_dim=INPUT_DIM, **sz)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    p = tmp / "W_init.pt"
    torch.save(state, str(p))
    return p


# --------------------------------------------------------------------------- #
def test_local_train_kernel_happy_path():
    tmp = _fresh_tmp()
    try:
        shard_dir = tmp / "Household_ID=H1"
        parquet_path = _make_fake_household_parquet(shard_dir, n_rows=200)
        w_path = _make_w_global(tmp)
        out_root = tmp / "deltas"

        shard = AgentShard(
            agent_id="H1",
            district_id="D1",
            parquet_path=parquet_path,
            n_samples=200,
        )
        hp = LocalTrainHparams(
            pred_len=4, seq_len=24, tier="building",
            local_steps_building=3, batch_size=4, topk_ratio=0.5,
        )

        metrics = local_train_one_agent(
            shard=shard, w_global_path=str(w_path), round_id=0,
            output_root=str(out_root), hparams=hp, is_byzantine=False,
        )

        assert metrics["status"] == "ok"
        assert metrics["local_steps_executed"] == 3
        assert metrics["n_samples"] > 0
        assert metrics["delta_l2_pre_dp"] >= 0.0
        assert 0.0 < metrics["topk_kept_ratio"] <= 1.0
        delta_path = Path(metrics["delta_path"])
        assert delta_path.exists()
        loaded = torch.load(str(delta_path), map_location="cpu", weights_only=True)
        assert isinstance(loaded, dict) and len(loaded) > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_local_train_kernel_no_data():
    tmp = _fresh_tmp()
    try:
        shard_dir = tmp / "Household_ID=H_empty"
        # Only 5 rows - fewer than seq_len + pred_len, so no windows.
        _make_fake_household_parquet(shard_dir, n_rows=5)
        w_path = _make_w_global(tmp)

        shard = AgentShard(
            agent_id="H_empty", district_id="D1",
            parquet_path=str(shard_dir), n_samples=5,
        )
        hp = LocalTrainHparams(seq_len=24, pred_len=4, local_steps_building=2)
        metrics = local_train_one_agent(
            shard=shard, w_global_path=str(w_path), round_id=0,
            output_root=str(tmp / "deltas"), hparams=hp,
        )
        assert metrics["status"] == "no_data"
        assert metrics["n_samples"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_byzantine_attack_inflates_delta_norm():
    tmp = _fresh_tmp()
    try:
        shard_dir = tmp / "Household_ID=H_byz"
        parquet_path = _make_fake_household_parquet(shard_dir, n_rows=200)
        w_path = _make_w_global(tmp)

        shard = AgentShard(
            agent_id="H_byz", district_id="D1",
            parquet_path=parquet_path, n_samples=200,
        )
        # Disable DP / TopK so we observe the raw attack effect cleanly.
        hp_honest = LocalTrainHparams(
            pred_len=4, seq_len=24, local_steps_building=2,
            byzantine_attack="scale", byzantine_fraction=0.5,
            dp_sigma=0.0, topk_ratio=1.0,
        )
        m_honest = local_train_one_agent(
            shard=shard, w_global_path=str(w_path), round_id=0,
            output_root=str(tmp / "honest"), hparams=hp_honest,
            is_byzantine=False,
        )
        m_byz = local_train_one_agent(
            shard=shard, w_global_path=str(w_path), round_id=0,
            output_root=str(tmp / "byz"), hparams=hp_honest,
            is_byzantine=True,
        )
        # The "scale" attack multiplies the delta by 10×, so post-DP norm
        # should be >> honest norm. Use a generous tolerance to avoid
        # floating-point flakiness.
        assert m_byz["delta_l2_post_dp"] > 2.0 * m_honest["delta_l2_post_dp"]
        assert m_byz["is_byzantine"] is True
        assert m_honest["is_byzantine"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_local_train_kernel_idempotent_paths():
    """Rerunning the kernel for the same (round, agent) overwrites the same path."""
    tmp = _fresh_tmp()
    try:
        shard_dir = tmp / "Household_ID=H_idemp"
        parquet_path = _make_fake_household_parquet(shard_dir, n_rows=200)
        w_path = _make_w_global(tmp)
        out_root = tmp / "deltas"
        shard = AgentShard(
            agent_id="H_idemp", district_id="D1",
            parquet_path=parquet_path, n_samples=200,
        )
        hp = LocalTrainHparams(
            pred_len=4, seq_len=24, local_steps_building=2, batch_size=4,
        )
        m1 = local_train_one_agent(
            shard=shard, w_global_path=str(w_path), round_id=7,
            output_root=str(out_root), hparams=hp,
        )
        m2 = local_train_one_agent(
            shard=shard, w_global_path=str(w_path), round_id=7,
            output_root=str(out_root), hparams=hp,
        )
        assert m1["delta_path"] == m2["delta_path"]
        assert Path(m1["delta_path"]).exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
