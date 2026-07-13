from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from src.reporting.benchmark_summary import (
    build_benchmark_summary,
    write_benchmark_summary,
)


_TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "test_tmp"
_TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _fresh_tmp_dir() -> Path:
    path = _TEST_TMP_ROOT / f"bench_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_build_benchmark_summary_rolls_up_all_sections():
    tmp = _fresh_tmp_dir()
    try:
        etl = tmp / "etl_run"
        samp = tmp / "sampling_run"
        sim = tmp / "sim_run"
        train = tmp / "train"
        etl.mkdir(parents=True, exist_ok=True)
        samp.mkdir(parents=True, exist_ok=True)
        sim.mkdir(parents=True, exist_ok=True)
        train.mkdir(parents=True, exist_ok=True)

        (etl / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": "etl_x",
                    "runtime_seconds": 12.5,
                    "rows_in_meter_raw": 1000,
                    "rows_written": 800,
                    "distinct_households_in": 100,
                    "distinct_households_out": 30,
                    "weather_join_coverage": 0.97,
                    "output_files": 4,
                    "output_size_bytes": 4096,
                    "partition_columns": ["Weather_ID", "Household_ID"],
                    "extended_features": True,
                    "notes": [],
                }
            ),
            encoding="utf-8",
        )
        (samp / "report.json").write_text(
            json.dumps(
                {
                    "run_id": "sample_x",
                    "runtime_seconds": 5.0,
                    "total_households": 1400,
                    "selected_count": 30,
                    "k_clusters": 3,
                    "feature_columns": ["mean_load", "std_load"],
                    "cluster_sizes": {"0": 500, "1": 450, "2": 450},
                    "cluster_selected": {"0": 10, "1": 10, "2": 10},
                }
            ),
            encoding="utf-8",
        )
        (samp / "manifest.json").write_text(
            json.dumps({"selection_method": "spark_mllib_kmeans_stratified", "selected": []}),
            encoding="utf-8",
        )
        (train / "progression_report.json").write_text(
            json.dumps(
                {
                    "curriculum": [
                        {"stage": "h15m", "best_val_loss": 0.3, "train_size": 100, "val_size": 10},
                        {"stage": "h1h", "best_val_loss": 0.2, "train_size": 120, "val_size": 12},
                    ]
                }
            ),
            encoding="utf-8",
        )
        (train / "evaluation_report.json").write_text(
            json.dumps(
                {
                    "metadata": {"stage": "h24h"},
                    "config": {
                        "seq_len": 96,
                        "pred_len": 96,
                        "feature_version": "ismc_v1",
                        "use_weather": True,
                    },
                    "forecasting_mode": "direct_next_day_96_to_96",
                    "horizon_minutes": 1440,
                    "split": "test",
                    "split_mode": "cluster_stratified_household_disjoint",
                    "manifest_path": "artifacts/sampling/sample_x/manifest.json",
                    "aggregate": {"mae": 1.0, "rmse": 1.5, "wape": 10.0, "r2": 0.8, "mase": 0.7, "rmsse": 0.9},
                    "peak_regression": {
                        "peak_timing_mae_minutes": 30.0,
                        "peak_magnitude_mae": 0.25,
                    },
                    "peak_event_classification": {
                        "accuracy": 0.91,
                        "precision": 0.8,
                        "recall": 0.75,
                        "f1": 0.774,
                        "balanced_accuracy": 0.84,
                        "mcc": 0.62,
                        "cohen_kappa": 0.59,
                        "roc_auc": 0.89,
                        "average_precision": 0.81,
                        "threshold_kwh": 2.7,
                    },
                    "peak_threshold_sweep": {
                        "q80": {"recall": 0.81, "f1": 0.79, "mcc": 0.64},
                        "q90": {"recall": 0.75, "f1": 0.774, "mcc": 0.62},
                        "q95": {"recall": 0.55, "f1": 0.58, "mcc": 0.49},
                    },
                    "baselines": {
                        "persistence": {"mae": 1.4},
                        "seasonal_naive_7d": {"mae": 1.2},
                    },
                    "coverage_90": {"percent": 88.0},
                    "n_val_windows": 42,
                }
            ),
            encoding="utf-8",
        )
        (sim / "simulation_summary.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "run_id": "sim_x",
                        "scenario": "baseline",
                        "num_agents": 9,
                        "num_districts": 3,
                        "ticks": 200,
                    },
                    "building_summary": {
                        "ticks_completed_total": 1800,
                        "train_steps_total": 120,
                        "distillation_events_total": 9,
                        "upload_successes_total": 60,
                        "upload_failures_total": 1,
                    },
                    "federated_summary": {
                        "district_rounds": 12,
                        "city_rounds": 4,
                        "accepted_updates_total": 57,
                        "rejected_updates_total": 3,
                        "city_last_converged": False,
                        "city_last_drift": 0.2,
                    },
                }
            ),
            encoding="utf-8",
        )

        summary = build_benchmark_summary(
            etl_run_dir=etl,
            sampling_run_dir=samp,
            progression_report=train / "progression_report.json",
            evaluation_report=train / "evaluation_report.json",
            simulation_run_dir=sim,
        )
        assert summary["etl"]["rows_in_meter_raw"] == 1000
        assert summary["sampling"]["selected_count"] == 30
        assert summary["training"]["final_stage"] == "h1h"
        assert summary["training"]["forecasting_mode"] == "direct_next_day_96_to_96"
        assert summary["training"]["seq_len"] == 96
        assert summary["training"]["pred_len"] == 96
        assert summary["training"]["mae_vs_persistence_delta"] == 0.4
        assert summary["training"]["evaluation_mase"] == 0.7
        assert summary["training"]["peak_mcc"] == 0.62
        assert summary["training"]["peak_q95_mcc"] == 0.49
        assert summary["training"]["peak_f1"] == 0.774
        assert summary["training"]["peak_timing_mae_minutes"] == 30.0
        assert summary["simulation"]["district_rounds"] == 12
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_benchmark_summary_persists_json_and_markdown():
    tmp = _fresh_tmp_dir()
    try:
        out_dir = write_benchmark_summary(
            {
                "generated_at": "2026-01-01T00:00:00Z",
                "sources": {"etl_run_dir": None},
                "etl": {"runtime_seconds": 1.0, "rows_in_meter_raw": 10, "rows_written": 9,
                        "distinct_households_in": 2, "distinct_households_out": 2,
                        "weather_join_coverage": 1.0, "output_files": 1, "output_size_bytes": 123,
                        "partition_columns": ["Weather_ID"], "extended_features": False},
                "sampling": None,
                "training": None,
                "simulation": None,
            },
            output_root=tmp / "benchmarks",
        )
        assert (out_dir / "benchmark_summary.json").exists()
        assert (out_dir / "benchmark_summary.md").exists()
        payload = json.loads((out_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
        assert payload["etl"]["rows_written"] == 9
        md = (out_dir / "benchmark_summary.md").read_text(encoding="utf-8")
        assert "NeuroGrid Benchmark Summary" in md
        assert "Spark ETL" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
