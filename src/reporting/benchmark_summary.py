"""
Benchmark / evidence roll-up for NeuroGrid Phases 6 and 7.

This module does not run ETL, training, or simulation itself. It reads the
artifacts already emitted by those stages and assembles one presentation-ready
summary under `artifacts/benchmarks/<run_id>/`.

Inputs are intentionally loose and optional:
  * latest ETL run summary (`artifacts/etl_runs/*/summary.json`)
  * latest representative-sampling report (`artifacts/sampling/*/report.json`)
  * daily or curriculum training artifacts (`src/models/stored/...`)
  * evaluation report (`.../evaluation_report.json`)
  * latest simulation summary (`artifacts/simulation_runs/*/simulation_summary.json`)

That keeps the benchmark layer honest: it reports what the project actually
ran, and degrades gracefully when some stages have not been executed yet.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.paths import ensure_dir, resolve_path


def _latest_run_dir(root: str | Path | None) -> Path | None:
    root_path = resolve_path(root) if root is not None else None
    if root_path is None or not root_path.exists() or not root_path.is_dir():
        return None
    candidates = [p for p in root_path.iterdir() if p.is_dir()]
    if not candidates:
        return None

    def get_mtime(p: Path) -> float:
        # Prefer the summary file's mtime as it truly reflects completion.
        for fname in ("run_summary.json", "simulation_summary.json", "summary.json", "report.json"):
            fpath = p / fname
            if fpath.exists():
                return fpath.stat().st_mtime
        return p.stat().st_mtime

    return max(candidates, key=get_mtime)


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def _resolve_existing(path: str | Path | None) -> Path | None:
    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        return None
    return resolved


def _load_etl_summary(etl_run_dir: str | Path | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    run_dir = _resolve_existing(etl_run_dir) if etl_run_dir else _latest_run_dir("artifacts/etl_runs")
    if run_dir is None:
        return None, None
    return _read_json(run_dir / "summary.json"), run_dir


def _load_sampling_summary(
    sampling_run_dir: str | Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Path | None]:
    run_dir = _resolve_existing(sampling_run_dir) if sampling_run_dir else _latest_run_dir("artifacts/sampling")
    if run_dir is None:
        return None, None, None
    return _read_json(run_dir / "report.json"), _read_json(run_dir / "manifest.json"), run_dir


def _load_simulation_summary(
    simulation_run_dir: str | Path | None = None,
) -> tuple[dict[str, Any] | None, Path | None]:
    if simulation_run_dir:
        run_dir = _resolve_existing(simulation_run_dir)
    else:
        # Check both the gRPC path and the Spark FL path for the latest run.
        grpc_latest = _latest_run_dir("artifacts/simulation_runs")
        spark_latest = _latest_run_dir("artifacts/spark_fl/runs")

        if grpc_latest and spark_latest:
            # Helper to get comparable mtime.
            def get_mtime(p: Path) -> float:
                for fname in ("run_summary.json", "simulation_summary.json"):
                    fpath = p / fname
                    if fpath.exists():
                        return fpath.stat().st_mtime
                return p.stat().st_mtime

            # Pick the truly latest based on summary file mtime.
            run_dir = max(grpc_latest, spark_latest, key=get_mtime)
        else:
            run_dir = grpc_latest or spark_latest

    if run_dir is None:
        return None, None

    # gRPC path uses simulation_summary.json; Spark FL path uses run_summary.json.
    for fname in ("run_summary.json", "simulation_summary.json"):
        report = _read_json(run_dir / fname)
        if report:
            return report, run_dir

    return None, run_dir


def _load_training_summary(
    progression_report: str | Path | None = None,
    evaluation_report: str | Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Path | None]:
    prog_path = _resolve_existing(progression_report)
    if prog_path is None:
        default_prog = resolve_path("src/models/stored/progression_report.json")
        prog_path = default_prog if default_prog and default_prog.exists() else None
    eval_path = _resolve_existing(evaluation_report)
    if eval_path is None:
        candidate_paths = (
            resolve_path("src/models/stored/forecast_daily/evaluation_report.json"),
            resolve_path("src/models/stored/curriculum_h24h/evaluation_report.json"),
            resolve_path("src/models/stored/curriculum_h1h/evaluation_report.json"),
        )
        for candidate in candidate_paths:
            if candidate and candidate.exists():
                eval_path = candidate
                break
    bundle_root = prog_path.parent if prog_path is not None else (eval_path.parent if eval_path is not None else None)
    return _read_json(prog_path), _read_json(eval_path), bundle_root


def summarise_etl(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    rows_in = int(report.get("rows_in_meter_raw", 0))
    rows_written = int(report.get("rows_written", 0))
    hh_in = int(report.get("distinct_households_in", 0))
    hh_out = int(report.get("distinct_households_out", 0))
    return {
        "run_id": report.get("run_id"),
        "runtime_seconds": float(report.get("runtime_seconds", 0.0)),
        "rows_in_meter_raw": rows_in,
        "rows_written": rows_written,
        "rows_retained_ratio": (rows_written / rows_in) if rows_in else None,
        "distinct_households_in": hh_in,
        "distinct_households_out": hh_out,
        "household_retained_ratio": (hh_out / hh_in) if hh_in else None,
        "weather_join_coverage": report.get("weather_join_coverage"),
        "output_files": int(report.get("output_files", 0)),
        "output_size_bytes": int(report.get("output_size_bytes", 0)),
        "partition_columns": report.get("partition_columns", []),
        "extended_features": bool(report.get("extended_features", False)),
        "notes": list(report.get("notes", [])),
    }


def summarise_sampling(
    report: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not report:
        return None
    total = int(report.get("total_households", 0))
    selected = int(report.get("selected_count", 0))
    cluster_selected = {
        int(k): int(v) for k, v in (report.get("cluster_selected") or {}).items()
    }
    return {
        "run_id": report.get("run_id"),
        "runtime_seconds": float(report.get("runtime_seconds", 0.0)),
        "selection_method": (manifest or {}).get("selection_method", "unknown"),
        "total_households": total,
        "selected_count": selected,
        "household_reduction_ratio": (1.0 - (selected / total)) if total else None,
        "k_clusters": int(report.get("k_clusters", 0)),
        "feature_columns": list(report.get("feature_columns", [])),
        "cluster_sizes": report.get("cluster_sizes", {}),
        "cluster_selected": cluster_selected,
    }


def summarise_training(
    progression_report: dict[str, Any] | None,
    evaluation_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not progression_report and not evaluation_report:
        return None
    summary: dict[str, Any] = {}
    if progression_report:
        stages = list(progression_report.get("curriculum", []))
        summary.update(
            {
                "n_stages": len(stages),
                "stages": [s.get("stage") for s in stages],
                "final_stage": stages[-1].get("stage") if stages else None,
                "best_val_loss_final": stages[-1].get("best_val_loss") if stages else None,
                "train_windows_final": stages[-1].get("train_size") if stages else None,
                "val_windows_final": stages[-1].get("val_size") if stages else None,
            }
        )
    if evaluation_report:
        agg = evaluation_report.get("aggregate", {})
        baselines = evaluation_report.get("baselines", {})
        peak_cls = evaluation_report.get("peak_event_classification", {})
        peak_reg = evaluation_report.get("peak_regression", {})
        peak_sweep = evaluation_report.get("peak_threshold_sweep", {})
        persistence = baselines.get("persistence", {})
        seasonal = baselines.get("seasonal_naive_7d", {})
        cfg = evaluation_report.get("config", {})
        metadata = evaluation_report.get("metadata", {})
        if "n_stages" not in summary:
            summary["n_stages"] = 1 if metadata else None
        if "final_stage" not in summary:
            summary["final_stage"] = metadata.get("stage")
        summary.update(
            {
                "forecasting_mode": evaluation_report.get("forecasting_mode"),
                "split": evaluation_report.get("split"),
                "split_mode": evaluation_report.get("split_mode"),
                "manifest_path": evaluation_report.get("manifest_path"),
                "seq_len": cfg.get("seq_len"),
                "pred_len": cfg.get("pred_len"),
                "horizon_minutes": evaluation_report.get("horizon_minutes"),
                "evaluation_mae": agg.get("mae"),
                "evaluation_rmse": agg.get("rmse"),
                "evaluation_wape": agg.get("wape"),
                "evaluation_r2": agg.get("r2"),
                "evaluation_mase": agg.get("mase"),
                "evaluation_rmsse": agg.get("rmsse"),
                "evaluation_nrmse": agg.get("nrmse"),
                "evaluation_cvrmse": agg.get("cvrmse"),
                "evaluation_mbe": agg.get("mbe"),
                "evaluation_medae": agg.get("medae"),
                "evaluation_pearson_r": agg.get("pearson_r"),
                "persistence_mae": persistence.get("mae"),
                "seasonal_naive_mae": seasonal.get("mae"),
                "mae_vs_persistence_delta": _rounded_delta(persistence.get("mae"), agg.get("mae")),
                "mae_vs_seasonal_delta": _rounded_delta(seasonal.get("mae"), agg.get("mae")),
                "coverage_90_percent": (evaluation_report.get("coverage_90") or {}).get("percent"),
                "n_eval_windows": evaluation_report.get("n_val_windows"),
                "peak_accuracy": peak_cls.get("accuracy"),
                "peak_precision": peak_cls.get("precision"),
                "peak_recall": peak_cls.get("recall"),
                "peak_f1": peak_cls.get("f1"),
                "peak_balanced_accuracy": peak_cls.get("balanced_accuracy"),
                "peak_mcc": peak_cls.get("mcc"),
                "peak_cohen_kappa": peak_cls.get("cohen_kappa"),
                "peak_roc_auc": peak_cls.get("roc_auc"),
                "peak_average_precision": peak_cls.get("average_precision"),
                "peak_threshold_kwh": peak_cls.get("threshold_kwh"),
                "peak_timing_mae_minutes": peak_reg.get("peak_timing_mae_minutes"),
                "peak_magnitude_mae": peak_reg.get("peak_magnitude_mae"),
            }
        )
        for bucket in ("q80", "q90", "q95"):
            stats = peak_sweep.get(bucket, {})
            summary.update(
                {
                    f"peak_{bucket}_recall": stats.get("recall"),
                    f"peak_{bucket}_f1": stats.get("f1"),
                    f"peak_{bucket}_mcc": stats.get("mcc"),
                }
            )
    return summary


def _rounded_delta(a: Any, b: Any) -> float | None:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    return round(float(a) - float(b), 6)


def summarise_simulation(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None

    # Handle Spark FL report (run_summary.json)
    if "config" in report and "rounds_executed" in report:
        cfg = report.get("config", {})
        return {
            "run_id": report.get("run_id"),
            "scenario": cfg.get("tier", "unknown") + "_spark_fl",
            "num_agents": cfg.get("num_agents"),
            "num_districts": cfg.get("num_districts"),
            "ticks_requested": cfg.get("rounds"),
            "ticks_completed_total": report.get("rounds_executed"),
            "train_steps_total": report.get("rounds_executed", 0) * cfg.get("local_steps_building", 0) * cfg.get("num_agents", 0),
            "distillation_events_total": 0,
            "upload_successes_total": report.get("rounds_executed", 0) * cfg.get("num_agents", 0),
            "upload_failures_total": 0,
            "district_rounds": report.get("rounds_executed"),
            "city_rounds": report.get("rounds_executed"),
            "accepted_updates_total": int(report.get("final_acceptance_rate", 1.0) * cfg.get("num_agents", 1) * report.get("rounds_executed", 1)),
            "rejected_updates_total": int((1.0 - report.get("final_acceptance_rate", 1.0)) * cfg.get("num_agents", 1) * report.get("rounds_executed", 1)),
            "city_last_converged": report.get("final_acceptance_rate", 0.0) > 0.5,
            "city_last_drift": report.get("final_loss_last_mean"),
            "is_spark_native": True,
        }

    # Handle gRPC report (simulation_summary.json)
    meta = report.get("metadata", {})
    building = report.get("building_summary", {})
    federated = report.get("federated_summary", {})
    return {
        "run_id": meta.get("run_id"),
        "scenario": meta.get("scenario"),
        "num_agents": meta.get("num_agents"),
        "num_districts": meta.get("num_districts"),
        "ticks_requested": meta.get("ticks"),
        "ticks_completed_total": building.get("ticks_completed_total"),
        "train_steps_total": building.get("train_steps_total"),
        "distillation_events_total": building.get("distillation_events_total"),
        "upload_successes_total": building.get("upload_successes_total"),
        "upload_failures_total": building.get("upload_failures_total"),
        "district_rounds": federated.get("district_rounds"),
        "city_rounds": federated.get("city_rounds"),
        "accepted_updates_total": federated.get("accepted_updates_total"),
        "rejected_updates_total": federated.get("rejected_updates_total"),
        "city_last_converged": federated.get("city_last_converged"),
        "city_last_drift": federated.get("city_last_drift"),
        "is_spark_native": False,
    }


def build_benchmark_summary(
    *,
    etl_run_dir: str | Path | None = None,
    sampling_run_dir: str | Path | None = None,
    progression_report: str | Path | None = None,
    evaluation_report: str | Path | None = None,
    simulation_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    etl_report, etl_dir = _load_etl_summary(etl_run_dir)
    sampling_report, sampling_manifest, sampling_dir = _load_sampling_summary(sampling_run_dir)
    curriculum_report, eval_report, training_root = _load_training_summary(progression_report, evaluation_report)
    simulation_report, simulation_dir = _load_simulation_summary(simulation_run_dir)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "etl_run_dir": str(etl_dir) if etl_dir else None,
            "sampling_run_dir": str(sampling_dir) if sampling_dir else None,
            "training_root": str(training_root) if training_root else None,
            "simulation_run_dir": str(simulation_dir) if simulation_dir else None,
        },
        "etl": summarise_etl(etl_report),
        "sampling": summarise_sampling(sampling_report, sampling_manifest),
        "training": summarise_training(curriculum_report, eval_report),
        "simulation": summarise_simulation(simulation_report),
    }


def write_benchmark_summary(
    summary: dict[str, Any],
    output_root: str | Path = "artifacts/benchmarks",
) -> Path:
    run_id = datetime.now(timezone.utc).strftime("benchmark_%Y%m%dT%H%M%SZ")
    out_dir = ensure_dir(Path(output_root) / run_id)
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "benchmark_summary.md").write_text(
        _to_markdown(summary),
        encoding="utf-8",
    )
    return out_dir


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# NeuroGrid Benchmark Summary",
        "",
        f"- Generated at: `{summary.get('generated_at')}`",
        "",
        "## Sources",
        "",
    ]
    for k, v in (summary.get("sources") or {}).items():
        lines.append(f"- `{k}`: `{v}`")

    etl = summary.get("etl")
    if etl:
        lines += [
            "",
            "## Spark ETL",
            "",
            f"- Runtime (s): `{etl.get('runtime_seconds')}`",
            f"- Rows in / written: `{etl.get('rows_in_meter_raw')}` / `{etl.get('rows_written')}`",
            f"- Households in / out: `{etl.get('distinct_households_in')}` / `{etl.get('distinct_households_out')}`",
            f"- Weather join coverage: `{etl.get('weather_join_coverage')}`",
            f"- Output files / size bytes: `{etl.get('output_files')}` / `{etl.get('output_size_bytes')}`",
            f"- Partition columns: `{etl.get('partition_columns')}`",
        ]

    sampling = summary.get("sampling")
    if sampling:
        lines += [
            "",
            "## Representative Sampling",
            "",
            f"- Method: `{sampling.get('selection_method')}`",
            f"- Total households / selected: `{sampling.get('total_households')}` / `{sampling.get('selected_count')}`",
            f"- Household reduction ratio: `{sampling.get('household_reduction_ratio')}`",
            f"- k clusters: `{sampling.get('k_clusters')}`",
            f"- Feature columns: `{sampling.get('feature_columns')}`",
            f"- Cluster selected: `{sampling.get('cluster_selected')}`",
        ]

    training = summary.get("training")
    if training:
        lines += [
            "",
            "## Forecast Training",
            "",
            f"- Forecasting mode: `{training.get('forecasting_mode')}`",
            f"- Seq len / pred len: `{training.get('seq_len')}` / `{training.get('pred_len')}`",
            f"- Horizon minutes: `{training.get('horizon_minutes')}`",
            f"- Split / split mode: `{training.get('split')}` / `{training.get('split_mode')}`",
            f"- Curriculum stages: `{training.get('n_stages')}`",
            f"- Final stage: `{training.get('final_stage')}`",
            f"- Final best val loss: `{training.get('best_val_loss_final')}`",
            f"- Final train / val windows: `{training.get('train_windows_final')}` / `{training.get('val_windows_final')}`",
            f"- Evaluation MAE / RMSE / WAPE: `{training.get('evaluation_mae')}` / `{training.get('evaluation_rmse')}` / `{training.get('evaluation_wape')}`",
            f"- Evaluation MASE / RMSSE: `{training.get('evaluation_mase')}` / `{training.get('evaluation_rmsse')}`",
            f"- Evaluation NRMSE / CVRMSE / Pearson r: `{training.get('evaluation_nrmse')}` / `{training.get('evaluation_cvrmse')}` / `{training.get('evaluation_pearson_r')}`",
            f"- MAE delta vs persistence / seasonal: `{training.get('mae_vs_persistence_delta')}` / `{training.get('mae_vs_seasonal_delta')}`",
            f"- 90% interval coverage: `{training.get('coverage_90_percent')}`",
            f"- Peak classification Acc / F1 / Recall / ROC-AUC: `{training.get('peak_accuracy')}` / `{training.get('peak_f1')}` / `{training.get('peak_recall')}` / `{training.get('peak_roc_auc')}`",
            f"- Peak MCC / Cohen kappa: `{training.get('peak_mcc')}` / `{training.get('peak_cohen_kappa')}`",
            f"- Peak threshold sweep q80/q90/q95 (Recall / F1 / MCC): `({training.get('peak_q80_recall')}, {training.get('peak_q80_f1')}, {training.get('peak_q80_mcc')})` / `({training.get('peak_q90_recall')}, {training.get('peak_q90_f1')}, {training.get('peak_q90_mcc')})` / `({training.get('peak_q95_recall')}, {training.get('peak_q95_f1')}, {training.get('peak_q95_mcc')})`",
            f"- Peak timing MAE (min) / magnitude MAE: `{training.get('peak_timing_mae_minutes')}` / `{training.get('peak_magnitude_mae')}`",
        ]

    sim = summary.get("simulation")
    if sim:
        lines += [
            "",
            "## Distributed Simulation",
            "",
            f"- Implementation: `{'Spark Native (RDD-Distributor)' if sim.get('is_spark_native') else 'gRPC (Hierarchical-MAS)'}`",
            f"- Scenario: `{sim.get('scenario')}`",
            f"- Buildings / districts: `{sim.get('num_agents')}` / `{sim.get('num_districts')}`",
            f"- Ticks/Rounds completed: `{sim.get('ticks_completed_total')}` (requested `{sim.get('ticks_requested')}`)",
            f"- Total local steps: `{sim.get('train_steps_total')}`",
            f"- Updates accepted / rejected: `{sim.get('accepted_updates_total')}` / `{sim.get('rejected_updates_total')}`",
            f"- Final converged / drift: `{sim.get('city_last_converged')}` / `{sim.get('city_last_drift')}`",
        ]

    lines += [
        "",
        "## Interpretation",
        "",
        "- Spark is being used for ETL, representative sampling, and federated audit analytics." if not (sim and sim.get('is_spark_native')) else "- The entire pipeline (ETL -> Sampling -> FL Training -> Audit) is end-to-end Spark-native.",
        "- PyTorch + gRPC remain the learning/control plane for the hierarchical federated simulation." if not (sim and sim.get('is_spark_native')) else "- PyTorch is used for local SGD within Spark partitions; Spark RDD handles agent orchestration.",
        "- This summary reflects only artifacts that exist on disk; missing sections indicate runs that have not been executed yet.",
        "",
    ]
    return "\n".join(lines)
