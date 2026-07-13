import math
import threading
import json

import numpy as np
import pytest
import torch

from src.data.feature_pipeline import feature_frame_to_matrix
from src.federated.aggregation import trimmed_mean, weighted_fedavg
from src.federated.clipping import clip_state_dict
from src.federated.dp import DPAccountant
from src.federated.krum import pairwise_sq_distances
from src.models.city_lstm import CityLSTM
from src.tiers.pending_store import PendingStore


def test_aggregation_rejects_nonfinite_and_negative_weights():
    update = {"w": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="non-negative"):
        weighted_fedavg([update, update], [2.0, -1.0])
    with pytest.raises(ValueError, match="NaN or infinity"):
        weighted_fedavg([{"w": torch.tensor([float("nan")])}])


def test_aggregation_rejects_shape_mismatch_and_invalid_trim():
    with pytest.raises(ValueError, match="inconsistent shapes"):
        weighted_fedavg([{"w": torch.ones(1)}, {"w": torch.ones(2)}])
    with pytest.raises(ValueError, match="trim_ratio"):
        trimmed_mean([{"w": torch.ones(1)}], trim_ratio=0.5)


def test_clipping_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="NaN or infinity"):
        clip_state_dict({"w": torch.tensor([float("inf")])}, threshold=1.0)


def test_krum_distance_matches_direct_calculation():
    updates = [
        {"w": torch.tensor([1.0, 2.0])},
        {"w": torch.tensor([4.0, 6.0])},
    ]
    distances = pairwise_sq_distances(updates)
    assert torch.allclose(distances, torch.tensor([[0.0, 25.0], [25.0, 0.0]]))


def test_pending_store_finalize_is_atomic_and_single_winner(tmp_path):
    store = PendingStore(tmp_path / "pending.db", "D1")
    store.append_upload(0, "A", 2, {"w": torch.ones(1)})
    winners = []
    errors = []

    def finalize():
        try:
            winners.append(
                store.finalize_round(
                    0, 1, uplink_state={"w": torch.tensor([2.0])}, uplink_samples=2
                )
            )
        except RuntimeError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=finalize) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert winners == [1]
    assert len(errors) == 1
    assert store.current_round == 1
    assert store.round_size(0) == 0
    assert store.recover()["completed_rounds"] == 1
    uplink = store.pending_uplink()
    assert uplink["round_id"] == 0
    assert uplink["n_samples"] == 2
    assert torch.equal(uplink["state_dict"]["w"], torch.tensor([2.0]))
    store.complete_uplink(0)
    assert store.pending_uplink() is None


def test_mc_forward_single_sample_has_finite_zero_uncertainty():
    model = CityLSTM(pred_len=2, input_dim=3, hidden_dim=4, num_layers=1)
    _, std = model.mc_forward(torch.zeros(1, 3, 3), n_samples=1)
    assert torch.isfinite(std).all()
    assert torch.count_nonzero(std) == 0


def test_feature_matrix_rejects_nan():
    import pandas as pd

    with pytest.raises(ValueError, match="NaN or infinity"):
        feature_frame_to_matrix(pd.DataFrame({"x": [np.nan]}), ["x"])


def test_fallback_accountant_validates_inputs_and_composes():
    accountant = DPAccountant()
    with pytest.raises(ValueError, match="sample_rate"):
        accountant.record(1.0, sample_rate=0.0)
    with pytest.raises(ValueError, match="delta"):
        accountant.epsilon(delta=1.0)
    accountant.record(2.0)
    accountant.record(2.0)
    assert math.isfinite(accountant.epsilon())


def test_scheduler_fallback_is_energy_conserving_noop(monkeypatch):
    import src.ismcc.computation as computation

    monkeypatch.setattr(computation, "pywraplp", None)
    module = computation.ComputationModule("A", pred_len=4)
    result = module.optimise_schedule(np.ones(4), tariff=np.arange(1, 5, dtype=float))
    assert result["status"] == "solver_unavailable"
    assert result["savings"] == 0.0
    assert not any(result["battery_charge_kwh"])
    assert not any(result["battery_discharge_kwh"])


def test_recommender_rejects_ungrounded_hallucination(monkeypatch):
    from src.llm.recommender import EnergyRecommender
    import urllib.request

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"message": {"content": "Use everything at slot 99 for huge savings."}}
            ).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response())
    result = EnergyRecommender(timeout_seconds=0.1).recommend(
        np.ones(96), {"peak_slot": 4, "precool_start_slot": 0}
    )
    assert result.source == "fallback_error"
    assert "slot 99" not in result.text
