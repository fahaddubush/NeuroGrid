"""
Scalability benchmark - Spark FL path (and optionally gRPC for comparison).

Sweeps `agent_counts ∈ {16, 64, 256, ...}` and reports rounds/sec, p50/p95
round wall-clock, Krum acceptance rate, and final loss for each configuration.
Output lands in `artifacts/spark_fl/benchmarks/<run_id>/` as both Parquet for
machine-readable analysis and Markdown for human review. A sweep provides
scaling evidence that cannot be inferred from a single configuration.

Reporting policy
----------------
Every row in the output table records the *exact* command-line knobs used,
the wall-clock measurement, and the git SHA. We do not synthesise or smooth
numbers. If a run failed, its row records `status="failed"` with the
exception message rather than being silently dropped.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.simulation.spark_fl_runner import SparkFLConfig, run_spark_fl


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _summarise_run(run_dir: Path) -> dict[str, Any]:
    """Pull the round_metrics.parquet from a finished Spark FL run."""
    rm_path = run_dir / "round_metrics.parquet"
    if not rm_path.exists():
        return {"status": "no_metrics"}
    rm = pd.read_parquet(rm_path)
    if rm.empty:
        return {"status": "empty_metrics"}
    return {
        "rounds_executed": int(len(rm)),
        "wallclock_total_s": float(rm["wallclock_s"].sum()),
        "wallclock_p50_s": float(rm["wallclock_s"].quantile(0.5)),
        "wallclock_p95_s": float(rm["wallclock_s"].quantile(0.95)),
        "rounds_per_sec": float(len(rm) / max(1e-6, rm["wallclock_s"].sum())),
        "krum_acceptance_rate_final": float(rm["krum_acceptance_rate"].iloc[-1]),
        "loss_last_mean_final": (
            float(rm["loss_last_mean"].iloc[-1])
            if "loss_last_mean" in rm.columns else float("nan")
        ),
    }


def _run_spark_fl_once(
    n_agents: int,
    rounds: int,
    num_districts: int,
    local_steps: int,
    parquet_root: str,
    output_root: str,
) -> dict[str, Any]:
    """Run one (n_agents, rounds) Spark FL configuration; return its summary."""
    cfg = SparkFLConfig(
        num_agents=n_agents,
        num_districts=min(num_districts, max(1, n_agents // 2)),
        rounds=rounds,
        local_steps_building=local_steps,
        parquet_root=parquet_root,
        run_root=str(Path(output_root) / "spark_fl_runs"),
    )
    t0 = time.monotonic()
    row: dict[str, Any] = {
        "path": "spark_fl",
        "n_agents": int(n_agents),
        "num_districts": int(cfg.num_districts),
        "rounds_planned": int(rounds),
        "local_steps_building": int(local_steps),
    }
    try:
        run_dir = run_spark_fl(cfg)
        row.update(_summarise_run(run_dir))
        row["wallclock_total_observed_s"] = time.monotonic() - t0
        row["run_dir"] = str(run_dir)
        row["status"] = row.get("status", "ok")
    except Exception as e:
        row["status"] = "failed"
        row["error"] = f"{type(e).__name__}: {e}"
        row["wallclock_total_observed_s"] = time.monotonic() - t0
        logging.error("Spark FL run failed for n=%d: %s", n_agents, traceback.format_exc())
    return row


def _run_grpc_once(
    n_agents: int,
    num_districts: int,
    ticks: int = 60,
) -> dict[str, Any]:
    """Run the gRPC simulation path for direct wall-clock comparison.

    We invoke the existing runner via subprocess to avoid contaminating this
    process's SparkSession state. Wall-clock is measured around the
    subprocess; we don't try to collect per-round metrics because the gRPC
    path's notion of "round" is tick-driven, not aligned with FL rounds.
    """
    row: dict[str, Any] = {
        "path": "grpc",
        "n_agents": int(n_agents),
        "num_districts": int(num_districts),
        "ticks": int(ticks),
    }
    cmd = [
        "python", "-m", "src.cli", "simulate",
        "--num_agents", str(n_agents),
        "--num_districts", str(num_districts),
        "--ticks", str(ticks),
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 30)
        row["wallclock_total_observed_s"] = time.monotonic() - t0
        row["return_code"] = proc.returncode
        row["status"] = "ok" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            row["error"] = (proc.stderr or "")[-500:]
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
        row["wallclock_total_observed_s"] = time.monotonic() - t0
    except Exception as e:
        row["status"] = "failed"
        row["error"] = f"{type(e).__name__}: {e}"
        row["wallclock_total_observed_s"] = time.monotonic() - t0
    return row


def run_scalability_sweep(
    agent_counts: list[int],
    rounds: int,
    num_districts: int,
    local_steps_building: int,
    parquet_root: str,
    output_root: str,
    include_grpc: bool = False,
) -> Path:
    """Execute the full sweep and persist Parquet + Markdown."""
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for n in agent_counts:
        logging.info("[scalability] === n_agents=%d (Spark FL) ===", n)
        rows.append(_run_spark_fl_once(
            n_agents=n,
            rounds=rounds,
            num_districts=num_districts,
            local_steps=local_steps_building,
            parquet_root=parquet_root,
            output_root=str(out_dir),
        ))
        if include_grpc:
            logging.info("[scalability] === n_agents=%d (gRPC) ===", n)
            rows.append(_run_grpc_once(
                n_agents=n,
                num_districts=num_districts,
                ticks=60,
            ))

    df = pd.DataFrame(rows)
    parquet_path = out_dir / "scalability_results.parquet"
    df.to_parquet(parquet_path, index=False)

    md_path = out_dir / "scalability_results.md"
    md_lines = [
        f"# Spark FL Scalability Sweep `{run_id}`",
        "",
        f"- Git SHA: `{_git_sha()}`",
        f"- Agent counts: `{agent_counts}`",
        f"- Rounds per config: `{rounds}`",
        f"- Districts: `{num_districts}`",
        f"- Local steps (building): `{local_steps_building}`",
        f"- Includes gRPC comparison: `{include_grpc}`",
        "",
        "## Results",
        "",
        df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Defense notes",
        "",
        "* `wallclock_total_observed_s` is wall-clock around the entire run "
        "(driver + ranks). `wallclock_p50_s` / `p95_s` are per-round.",
        "* Spark FL `rounds_per_sec` is computed from observed `wallclock_s` "
        "summed over actually-executed rounds (not planned). Convergence may "
        "terminate early.",
        "* gRPC rows do not report Krum acceptance because the gRPC path's "
        "audit lives in `artifacts/simulation_runs/<id>/federated_audit/` "
        "rather than in our scalability table - see that run's "
        "simulation_summary for cross-reference.",
        "* All failed runs are reported with `status=failed` rather than "
        "silently dropped (see Honest reporting policy in module docstring).",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    sweep_meta = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "agent_counts": list(agent_counts),
        "rounds": rounds,
        "num_districts": num_districts,
        "local_steps_building": local_steps_building,
        "include_grpc": include_grpc,
        "parquet_root": parquet_root,
    }
    (out_dir / "sweep_meta.json").write_text(
        json.dumps(sweep_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return md_path
