from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pyarrow.dataset as pa_dataset
import torch

from src.federated.audit import (
    append_partitioned_parquet,
    build_city_round_summary,
    build_district_round_summary,
)
from src.proto import neurogrid_pb2
from src.tiers.city import CityAggregator
from src.tiers.district import DistrictOrchestrator

_TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "test_tmp"
_TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _fresh_tmp_dir() -> Path:
    path = _TEST_TMP_ROOT / f"audit_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _weight_payload(agent_id: str, round_id: int, scale: float, n_samples: int = 1):
    from src.federated.payload import serialize_update

    state = {"w": torch.tensor([scale], dtype=torch.float32)}
    return neurogrid_pb2.WeightPayload(
        agent_id=agent_id,
        round=round_id,
        n_samples=n_samples,
        state_dict_bytes=serialize_update(
            state, model_version=f"city-round-{round_id}"
        ),
        tier="district",
    )


def test_build_district_round_summary_counts_and_rates():
    summary = build_district_round_summary(
        [
            {"agent_id": "A", "accepted": True, "n_samples": 5, "krum_score": 1.0, "original_l2": 2.0, "clipped_l2": 1.0},
            {"agent_id": "B", "accepted": False, "n_samples": 3, "krum_score": 4.0, "original_l2": 5.0, "clipped_l2": 1.0},
        ],
        district_id="D1",
        round_id=7,
        clip_threshold=1.0,
        trim_ratio=0.1,
        f_byzantine=1,
        aggregated_available=True,
    )
    assert summary["district_id"] == "D1"
    assert summary["round_id"] == 7
    assert summary["n_agents_total"] == 2
    assert summary["n_agents_accepted"] == 1
    assert summary["n_agents_rejected"] == 1
    assert summary["n_samples_total"] == 8
    assert summary["acceptance_rate"] == 0.5
    assert summary["krum_score_mean"] == 2.5


def test_append_partitioned_parquet_writes_hive_layout():
    row = build_city_round_summary(
        round_id=3,
        expected_districts=3,
        district_sample_counts={"D1": 10, "D2": 10, "D3": 10},
        drift=0.25,
        converged=False,
        had_previous_global=True,
    )
    tmp = _fresh_tmp_dir()
    try:
        append_partitioned_parquet(tmp, row, partition_cols=("round_id",))
        dataset = pa_dataset.dataset(tmp, format="parquet", partitioning="hive")
        tbl = dataset.to_table().to_pydict()
        assert tbl["expected_districts"] == [3]
        assert tbl["participating_districts"] == [3]
        assert tbl["round_id"] == [3]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_city_aggregator_persists_round_summary():
    tmp = _fresh_tmp_dir()
    try:
        city = CityAggregator(expected_districts=2, audit_dir=tmp, state_dir=tmp / "state")
        city.global_state = {"w": torch.tensor([10.0], dtype=torch.float32)}
        ack1 = city.SendGradient(_weight_payload("D1", 0, 1.0, n_samples=4), None)
        ack2 = city.SendGradient(_weight_payload("D2", 0, 3.0, n_samples=6), None)
        assert ack1.success is True
        assert ack2.success is True
        assert torch.allclose(city.global_state["w"], torch.tensor([12.2]))
        dataset = pa_dataset.dataset(
            tmp / "city_round_summary",
            format="parquet",
            partitioning="hive",
        )
        tbl = dataset.to_table().to_pydict()
        assert tbl["round_id"] == [0]
        assert tbl["n_samples_total"] == [10]
        assert tbl["participating_districts"] == [2]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_district_poll_distillation_forwards_city_converged_flag():
    class _FakeUplink:
        def __init__(self):
            self.last_poll_converged = True

        def pull_global(self):
            return {"w": torch.tensor([2.0], dtype=torch.float32)}

    tmp = _fresh_tmp_dir()
    try:
        orch = DistrictOrchestrator(
            district_id="D1",
            expected_agents=1,
            spark=object(),
            audit_dir=tmp / "audit",
            state_dir=tmp / "state",
        )
        orch.uplink = _FakeUplink()
        res = orch.PollDistillation(None, None)
        assert res.available is True
        assert res.converged is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
