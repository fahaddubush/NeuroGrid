"""Tests for the REFACTOR_2026_04_27 additions.

Covers:
- schema.EXTENDED_FEATURE_COLUMNS / DAILY_HORIZON
- feature_pipeline.add_lag_features (pandas fallback)
- ForecastConfig.with_extended_features
- computation.optimise_schedule (heuristic + OR-Tools paths)
- llm.recommender template fallback
- agent_core wiring of recommender (dummy, no API)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------- schema ----------
def test_schema_default_dim_unchanged():
    from src.data.schema import FEATURE_COLUMNS, INPUT_DIM
    assert INPUT_DIM == 11
    assert FEATURE_COLUMNS[0] == "kWh_received_Total"


def test_schema_extended_features_present():
    from src.data.schema import (
        EXTENDED_FEATURE_COLUMNS, EXTENDED_INPUT_DIM, LAG_FEATURE_COLUMNS,
    )
    assert EXTENDED_INPUT_DIM == 17
    assert set(LAG_FEATURE_COLUMNS).issubset(set(EXTENDED_FEATURE_COLUMNS))
    assert EXTENDED_FEATURE_COLUMNS[0] == "kWh_received_Total"


def test_schema_daily_horizon():
    from src.data.schema import DAILY_HORIZON
    assert DAILY_HORIZON.steps == 96
    assert DAILY_HORIZON.minutes == 1440


def test_cli_forecast_daily_threads_manifest(monkeypatch):
    import src.cli as cli

    captured: dict = {}

    def _fake_train_stage(**kwargs):
        captured.update(kwargs)

    import types
    monkeypatch.setitem(__import__("sys").modules, "src.training.trainer", types.SimpleNamespace(train_stage=_fake_train_stage))

    args = types.SimpleNamespace(
        output_dir="src/models/stored/forecast_daily",
        epochs=3,
        batch_size=16,
        seq_len=96,
        max_households=None,
        pretrained=None,
        tier="city",
        no_weather=False,
        manifest_path="artifacts/sampling/run_x/manifest.json",
    )
    cli._cmd_forecast_daily(args)
    assert captured["manifest_path"] == "artifacts/sampling/run_x/manifest.json"
    assert captured["seq_len"] == 96
    assert captured["stage"].steps == 96


# ---------- feature_pipeline ----------
def test_add_lag_features_creates_all_columns():
    from src.data.feature_pipeline import add_lag_features
    n = 200
    df = pd.DataFrame({"kWh_received_Total": np.linspace(0.1, 2.0, n)})
    out = add_lag_features(df)
    for col in ("lag_1", "lag_4", "lag_96", "roll_mean_4", "roll_std_4", "roll_max_96"):
        assert col in out.columns
        assert out[col].notna().all()


def test_add_lag_features_idempotent():
    from src.data.feature_pipeline import add_lag_features
    df = pd.DataFrame({"kWh_received_Total": [1.0] * 200, "lag_1": [9.9] * 200})
    out = add_lag_features(df)
    assert (out["lag_1"] == 9.9).all()


def test_forecast_config_with_extended_features():
    from src.data.feature_pipeline import ForecastConfig
    cfg = ForecastConfig.with_extended_features()
    assert cfg.input_dim == 17
    assert cfg.feature_version == "ismc_v1_ext"


# ---------- optimise_schedule ----------
def test_optimise_schedule_returns_savings():
    from src.ismcc.computation import ComputationModule
    cm = ComputationModule(agent_id="t1", pred_len=96, tier="building")
    forecast = np.abs(np.sin(np.linspace(0, 4 * np.pi, 96))) + 0.5
    out = cm.optimise_schedule(forecast)
    assert out["status"] in ("optimal", "solver_unavailable")
    assert "battery_charge_kwh" in out
    assert "battery_discharge_kwh" in out
    assert "savings" in out
    assert out["baseline_cost"] > 0


def test_optimise_schedule_empty_forecast():
    from src.ismcc.computation import ComputationModule
    cm = ComputationModule(agent_id="t1", pred_len=4, tier="building")
    out = cm.optimise_schedule(np.array([]))
    assert out["status"] == "empty_forecast"


def test_optimise_schedule_respects_tariff_shape():
    from src.ismcc.computation import ComputationModule
    cm = ComputationModule(agent_id="t1", pred_len=12, tier="building")
    forecast = np.ones(12) * 0.8
    tariff = np.linspace(0.1, 0.4, 12)
    out = cm.optimise_schedule(forecast, tariff=tariff)
    assert len(out["tariff"]) == 12


def test_optimise_schedule_tariff_length_mismatch_raises():
    from src.ismcc.computation import ComputationModule
    cm = ComputationModule(agent_id="t1", pred_len=4, tier="building")
    with pytest.raises(ValueError):
        cm.optimise_schedule(np.ones(8), tariff=np.ones(4))


# ---------- LLM recommender ----------
def test_recommender_template_fallback_no_api_key(monkeypatch):
    from src.llm.recommender import EnergyRecommender
    rec = EnergyRecommender(api_key=None)
    assert rec.has_llm is True
    forecast = np.linspace(0.2, 1.5, 96)
    schedule = {
        "peak_slot": 70,
        "precool_start_slot": 66,
        "battery_charge_kwh": [0.0] * 96,
        "battery_discharge_kwh": [0.0] * 96,
        "savings": 0.42,
        "status": "heuristic",
    }
    import urllib.request
    import urllib.error
    def mock_urlopen(*args, **kwargs):
        raise urllib.error.URLError("Target machine actively refused it")
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    out = rec.recommend(forecast=forecast, schedule=schedule, household_id="abc")
    assert out.source == "fallback_error"
    assert "peak" in out.text.lower()
    assert out.peak_slot == 70
    assert out.precool_slot == 66


def test_recommender_handles_short_forecast(monkeypatch):
    from src.llm.recommender import EnergyRecommender
    rec = EnergyRecommender(api_key=None)
    import urllib.request
    import urllib.error
    def mock_urlopen(*args, **kwargs):
        raise urllib.error.URLError("Target machine actively refused it")
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    out = rec.recommend(forecast=np.array([0.1, 0.2, 0.3, 0.4]), schedule={})
    assert out.source == "fallback_error"


# ---------- agent_core wiring ----------
def test_agent_core_recommender_optional():
    """Without a recommender the tick result still has 'recommendation' key=None."""
    from src.ismcc.agent_core import AgentCore

    class _StubMem:
        def write(self, *_args, **_kw): pass
        def window(self): return []

    class _StubSensing:
        def preprocess(self, raw): return {"raw": raw, "kwh": float(raw.get("kwh", 0.0)),
                                             "feature_vector": np.zeros(11, dtype=np.float32)}
        def detect_drift(self, _): return False

    class _StubCompute:
        def train_step(self, *_a, **_kw): return None
        def forecast(self, *_a, **_kw): return None
        def update_capacity(self, *_a, **_kw): pass
        def optimise_action(self, *_a, **_kw): return {"actions": ["MAINTAIN"], "reasoning": "x"}

    core = AgentCore(
        agent_id="t",
        sensing=_StubSensing(),
        memory={"stm": _StubMem()},
        compute=_StubCompute(),
    )
    out = core.tick({"kwh": 0.5})
    assert "recommendation" in out
    assert out["recommendation"] is None
