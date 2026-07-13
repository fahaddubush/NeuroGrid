from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from src.federated.audit import append_partitioned_parquet
from src.simulation.reporting import (
    load_building_summaries,
    summarise_buildings,
    summarise_federated_audit,
    write_simulation_report,
)


_TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "test_tmp"
_TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _fresh_tmp_dir() -> Path:
    path = _TEST_TMP_ROOT / f"sim_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_summarise_buildings_aggregates_actions_and_counters():
    rows = [
        {
            "ticks_completed": 10,
            "train_steps": 8,
            "drift_events": 2,
            "bootstrap_events": 1,
            "distillation_events": 1,
            "upload_successes": 2,
            "upload_failures": 1,
            "recommendations_emitted": 0,
            "action_counts": {"MAINTAIN": 7, "SHED_EV": 1},
            "communication": {"poll_successes": 1, "poll_errors": 0},
        },
        {
            "ticks_completed": 12,
            "train_steps": 9,
            "drift_events": 1,
            "bootstrap_events": 1,
            "distillation_events": 2,
            "upload_successes": 3,
            "upload_failures": 0,
            "recommendations_emitted": 1,
            "action_counts": {"MAINTAIN": 10, "CURTAIL_HVAC": 2},
            "communication": {"poll_successes": 2, "poll_errors": 1},
        },
    ]
    summary = summarise_buildings(rows)
    assert summary["n_buildings"] == 2
    assert summary["ticks_completed_total"] == 22
    assert summary["upload_successes_total"] == 5
    assert summary["poll_errors_total"] == 1
    assert summary["action_counts"]["MAINTAIN"] == 17
    assert summary["action_counts"]["CURTAIL_HVAC"] == 2


def test_federated_summary_reads_partitioned_parquet():
    tmp = _fresh_tmp_dir()
    try:
        audit_root = tmp / "federated_audit"
        append_partitioned_parquet(
            audit_root / "district_round_summary",
            {
                "district_id": "D1",
                "round_id": 0,
                "n_agents_accepted": 2,
                "n_agents_rejected": 1,
                "n_samples_total": 10,
            },
            partition_cols=("district_id", "round_id"),
        )
        append_partitioned_parquet(
            audit_root / "city" / "city_round_summary",
            {
                "round_id": 0,
                "participating_districts": 1,
                "n_samples_total": 10,
                "drift": 0.5,
                "converged": False,
            },
            partition_cols=("round_id",),
        )
        summary = summarise_federated_audit(audit_root)
        assert summary["district_rounds"] == 1
        assert summary["accepted_updates_total"] == 2
        assert summary["rejected_updates_total"] == 1
        assert summary["city_rounds"] == 1
        assert summary["city_last_converged"] is False
        assert summary["city_last_drift"] == 0.5
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_simulation_report_persists_json_and_markdown():
    tmp = _fresh_tmp_dir()
    try:
        (tmp / "building_agents").mkdir(parents=True, exist_ok=True)
        (tmp / "building_agents" / "B1.json").write_text(
            json.dumps({"ticks_completed": 5, "action_counts": {}, "communication": {}}),
            encoding="utf-8",
        )
        rows = load_building_summaries(tmp)
        summary = summarise_buildings(rows)
        out_dir = write_simulation_report(
            tmp,
            metadata={"run_id": "sim_x", "scenario": "baseline", "num_agents": 1, "num_districts": 1, "ticks": 5},
            building_summary=summary,
            federated_summary={"district_rounds": 0, "city_rounds": 0, "accepted_updates_total": 0, "rejected_updates_total": 0, "city_last_converged": False, "city_last_drift": None},
        )
        assert (out_dir / "simulation_summary.json").exists()
        assert (out_dir / "simulation_summary.md").exists()
        payload = json.loads((out_dir / "simulation_summary.json").read_text(encoding="utf-8"))
        assert payload["metadata"]["run_id"] == "sim_x"
        md = (out_dir / "simulation_summary.md").read_text(encoding="utf-8")
        assert "Simulation Run Report" in md
        assert "Topology" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
