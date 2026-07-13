"""
Pure-Python reporting helpers for simulation runs.

The simulation itself is multi-process and gRPC-based; these helpers keep the
reporting side deterministic and easy to test. They aggregate the per-building
JSON summaries and the Parquet round summaries emitted by the district/city
tiers into one presentation-ready run report.
"""
from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.dataset as pa_dataset


def make_run_id(prefix: str = "sim") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def load_building_summaries(run_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(run_dir) / "building_agents"
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def summarise_buildings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_buildings": 0,
            "ticks_completed_total": 0,
            "train_steps_total": 0,
            "drift_events_total": 0,
            "bootstrap_events_total": 0,
            "distillation_events_total": 0,
            "upload_successes_total": 0,
            "upload_failures_total": 0,
            "poll_successes_total": 0,
            "poll_errors_total": 0,
            "recommendations_total": 0,
            "action_counts": {},
        }

    out = {
        "n_buildings": len(rows),
        "ticks_completed_total": sum(int(r.get("ticks_completed", 0)) for r in rows),
        "train_steps_total": sum(int(r.get("train_steps", 0)) for r in rows),
        "drift_events_total": sum(int(r.get("drift_events", 0)) for r in rows),
        "bootstrap_events_total": sum(int(r.get("bootstrap_events", 0)) for r in rows),
        "distillation_events_total": sum(int(r.get("distillation_events", 0)) for r in rows),
        "upload_successes_total": sum(int(r.get("upload_successes", 0)) for r in rows),
        "upload_failures_total": sum(int(r.get("upload_failures", 0)) for r in rows),
        "poll_successes_total": sum(int(r.get("communication", {}).get("poll_successes", 0)) for r in rows),
        "poll_errors_total": sum(int(r.get("communication", {}).get("poll_errors", 0)) for r in rows),
        "recommendations_total": sum(int(r.get("recommendations_emitted", 0)) for r in rows),
        "action_counts": {},
    }
    action_counts: dict[str, int] = {}
    for row in rows:
        for action, n in row.get("action_counts", {}).items():
            action_counts[str(action)] = action_counts.get(str(action), 0) + int(n)
    out["action_counts"] = dict(sorted(action_counts.items()))
    return out


def load_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    if not root.exists():
        return []
    dataset = pa_dataset.dataset(root, format="parquet", partitioning="hive")
    return dataset.to_table().to_pylist()


def summarise_federated_audit(audit_root: str | Path) -> dict[str, Any]:
    audit_root = Path(audit_root)
    district_rows = load_parquet_rows(audit_root / "district_round_summary")
    city_rows = load_parquet_rows(audit_root / "city" / "city_round_summary")
    district_ids = sorted({str(r.get("district_id")) for r in district_rows if r.get("district_id") is not None})
    last_city = max(city_rows, key=lambda r: int(r.get("round_id", -1)), default=None)
    return {
        "district_rounds": len(district_rows),
        "district_ids": district_ids,
        "accepted_updates_total": sum(int(r.get("n_agents_accepted", 0)) for r in district_rows),
        "rejected_updates_total": sum(int(r.get("n_agents_rejected", 0)) for r in district_rows),
        "district_samples_total": sum(int(r.get("n_samples_total", 0)) for r in district_rows),
        "city_rounds": len(city_rows),
        "city_last_converged": bool(last_city.get("converged")) if last_city else False,
        "city_last_drift": (last_city.get("drift") if last_city else None),
    }


def write_simulation_report(
    run_dir: str | Path,
    metadata: dict[str, Any],
    building_summary: dict[str, Any],
    federated_summary: dict[str, Any],
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "building_summary": building_summary,
        "federated_summary": federated_summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
    }
    (run_dir / "simulation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "simulation_summary.md").write_text(
        _to_markdown(payload),
        encoding="utf-8",
    )
    return run_dir


def _to_markdown(payload: dict[str, Any]) -> str:
    meta = payload["metadata"]
    b = payload["building_summary"]
    f = payload["federated_summary"]
    lines = [
        "# Simulation Run Report",
        "",
        "## Topology",
        "",
        f"- Run id: `{meta.get('run_id')}`",
        f"- Scenario: `{meta.get('scenario', 'baseline')}`",
        f"- Buildings: `{meta.get('num_agents')}`",
        f"- Districts: `{meta.get('num_districts')}`",
        f"- Ticks requested: `{meta.get('ticks')}`",
        "",
        "## Building Metrics",
        "",
        f"- Ticks completed: `{b.get('ticks_completed_total', 0)}`",
        f"- Train steps: `{b.get('train_steps_total', 0)}`",
        f"- Drift events: `{b.get('drift_events_total', 0)}`",
        f"- Bootstrap events: `{b.get('bootstrap_events_total', 0)}`",
        f"- Distillation events: `{b.get('distillation_events_total', 0)}`",
        f"- Upload successes / failures: `{b.get('upload_successes_total', 0)}` / `{b.get('upload_failures_total', 0)}`",
        "",
        "## Federated Metrics",
        "",
        f"- District rounds: `{f.get('district_rounds', 0)}`",
        f"- Accepted / rejected updates: `{f.get('accepted_updates_total', 0)}` / `{f.get('rejected_updates_total', 0)}`",
        f"- City rounds: `{f.get('city_rounds', 0)}`",
        f"- Last city converged: `{f.get('city_last_converged', False)}`",
        f"- Last city drift: `{f.get('city_last_drift')}`",
    ]
    if b.get("action_counts"):
        lines += ["", "## Action Counts", ""]
        for action, n in b["action_counts"].items():
            lines.append(f"- `{action}`: `{n}`")
    return "\n".join(lines) + "\n"
