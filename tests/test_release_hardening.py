from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.data.dataset import NeuroGridDataset
from src.data.feature_pipeline import load_scaler, save_scaler
from src.federated.aggregation import masked_weighted_fedavg
from src.federated.krum import krum_scores
from src.federated.payload import deserialize_update, serialize_update
from src.tiers.city_store import CityStateStore
from src.tiers.building import BuildingAgent


def test_masked_fedavg_does_not_treat_omission_as_zero():
    updates = [{"w": torch.tensor([8.0])}, {"w": torch.tensor([0.0])}]
    masks = [{"w": torch.tensor([True])}, {"w": torch.tensor([False])}]
    result, present = masked_weighted_fedavg(updates, masks, [1.0, 100.0])
    assert result["w"].item() == 8.0
    assert present["w"].item() is True


def test_versioned_payload_roundtrip_and_size_limit():
    state = {"w": torch.tensor([1.0, 0.0])}
    masks = {"w": torch.tensor([True, False])}
    encoded = serialize_update(state, masks=masks, model_version="v7")
    decoded, decoded_masks, metadata = deserialize_update(encoded)
    assert torch.equal(decoded["w"], state["w"])
    assert torch.equal(decoded_masks["w"], masks["w"])
    assert metadata["model_version"] == "v7"
    with pytest.raises(ValueError, match="byte-size"):
        deserialize_update(encoded, max_payload_bytes=8)


def test_krum_fails_closed_for_invalid_byzantine_cohort():
    updates = [{"w": torch.tensor([float(i)])} for i in range(4)]
    with pytest.raises(ValueError, match=r"2f \+ 3"):
        krum_scores(updates, f=1)


def test_household_split_hash_is_seeded_and_stable():
    first = NeuroGridDataset._household_in_val("house-42", 0.5, seed=7)
    assert first == NeuroGridDataset._household_in_val("house-42", 0.5, seed=7)
    # Exercise the stable digest across many IDs rather than depending on one
    # particular seed pair being on opposite sides of the threshold.
    assignments = [
        NeuroGridDataset._household_in_val(f"house-{i}", 0.5, seed=7)
        for i in range(50)
    ]
    assert any(assignments) and not all(assignments)


def test_scaler_npz_roundtrip_without_pickle(tmp_path):
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = StandardScaler().fit(values)
    path = tmp_path / "scaler.npz"
    save_scaler(scaler, str(path))
    restored = load_scaler(str(path))
    assert np.allclose(restored.transform(values), scaler.transform(values))


def test_city_store_recovers_pending_and_committed_state(tmp_path):
    store = CityStateStore(tmp_path / "city.db")
    baseline = {"w": torch.tensor([10.0])}
    update = {"w": torch.tensor([2.0])}
    masks = {"w": torch.tensor([True])}
    store.save_baseline(baseline)
    store.append_pending(0, "D1", 5, update, masks, model_version="r0")
    recovered = CityStateStore(tmp_path / "city.db").recover()
    assert torch.equal(recovered["global_state"]["w"], baseline["w"])
    assert recovered["pending"][0]["D1"]["n_samples"] == 5
    store.commit_round(
        0,
        {"w": torch.tensor([12.0])},
        converged=False,
        stable_rounds=0,
        last_drift=2.0,
    )
    committed = store.recover()
    assert committed["current_round"] == 1
    assert committed["pending"] == {}
    assert torch.equal(committed["global_state"]["w"], torch.tensor([12.0]))


def test_building_uploads_at_most_once_per_global_round():
    class Compute:
        def __init__(self):
            self.delta_calls = 0

        def parameter_delta(self, baseline):
            self.delta_calls += 1
            return {"w": torch.tensor([1.0])}

    class Comms:
        current_round = 3

        def __init__(self):
            self.uploads = 0

        def pull_global(self):
            return None

        def upload_delta(self, delta, n_samples):
            self.uploads += 1
            return True

    agent = BuildingAgent.__new__(BuildingAgent)
    agent.agent_id = "B1"
    agent.federation_interval = 1
    agent.compute = Compute()
    agent.comms = Comms()
    agent.core = type("Core", (), {"memory": {"stm": [1, 2]}})()
    agent._metrics = {"federation_ticks": 0, "upload_successes": 0, "upload_failures": 0}
    agent._baseline_snapshot = {"w": torch.tensor([0.0])}
    agent._has_global_baseline = True
    agent._pending_delta = None
    agent._pending_delta_round = None
    agent._uploaded_round = None

    agent._maybe_federate(1)
    agent._maybe_federate(2)
    assert agent.comms.uploads == 1
    assert agent.compute.delta_calls == 1
