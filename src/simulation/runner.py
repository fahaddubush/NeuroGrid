"""
Multi-process orchestrator for the 3-tier ISMCC-MAS topology.

Spawns:
    1 City aggregator     (gRPC, port = base_port - 1)
    K District orchestrators (gRPC, port = base_port + i)  with Spark
    N Building agents       (one per process, assigned a HEAPO household CSV)

Buildings → connect to Districts (parent_port).
Districts → connect upstream to City (city_port).
Districts → write Byzantine-round audit Parquet under the run-local
            `artifacts/simulation_runs/<run_id>/federated_audit/`.
Buildings → write per-agent JSON metrics under
            `artifacts/simulation_runs/<run_id>/building_agents/`.
Runner    → aggregates those into `simulation_summary.{json,md}`.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import socket
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _select_household_csvs(num_agents: int) -> list[str]:
    data_dir = os.getenv("HEAPO_DATA_DIR")
    if not data_dir:
        raise ValueError("HEAPO_DATA_DIR not set.")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if len(files) < num_agents:
        raise ValueError(f"Need {num_agents} CSVs, found {len(files)} in {data_dir}.")
    step = max(1, len(files) // num_agents)
    return [os.path.join(data_dir, files[i * step]) for i in range(num_agents)]


def _wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((host, port))
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _validate_topology(num_agents: int, num_districts: int) -> list[int]:
    if num_agents <= 0 or num_districts <= 0:
        raise ValueError("num_agents and num_districts must be positive.")
    if num_districts > num_agents:
        raise ValueError("num_districts cannot exceed num_agents.")
    base = num_agents // num_districts
    rem = num_agents % num_districts
    return [base + (1 if i < rem else 0) for i in range(num_districts)]


# ---------------------------------------------------------------------- #
# Entry-point shims for multiprocessing.
# ---------------------------------------------------------------------- #
def _city_proc(
    num_districts: int,
    port: str,
    teacher_run_dir: str | None,
    audit_dir: str | None,
):
    from src.tiers.city import serve_city
    serve_city(
        expected_districts=num_districts,
        port=port,
        teacher_run_dir=teacher_run_dir,
        audit_dir=audit_dir,
    )


def _district_proc(
    district_id: str,
    expected: int,
    port: str,
    city_port: str,
    audit_dir: str | None,
):
    from src.tiers.district import serve_district
    serve_district(
        district_id=district_id,
        expected_agents=expected,
        port=port,
        city_host="127.0.0.1",
        city_port=city_port,
        audit_dir=audit_dir,
    )


def _building_proc(
    agent_id: str,
    csv_path: str,
    pred_len: int,
    scaler_path: str | None,
    parent_port: str,
    total_ticks: int,
    metrics_dir: str | None,
    enable_recommendations: bool = False,
):
    from src.tiers.building import BuildingAgent
    agent = BuildingAgent(
        agent_id=agent_id,
        csv_path=csv_path,
        pred_len=pred_len,
        scaler_path=scaler_path,
        district_port=parent_port,
        enable_recommendations=enable_recommendations,
    )
    try:
        for tick in range(total_ticks):
            if agent.step(tick) is None:
                return
            time.sleep(0.005)
    finally:
        if metrics_dir:
            agent.write_summary(metrics_dir)


# ---------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="ISMCC-MAS distributed simulation")
    parser.add_argument("--num_agents", type=int, default=9)
    parser.add_argument("--num_districts", type=int, default=3)
    parser.add_argument("--ticks", type=int, default=150)
    parser.add_argument("--base_port", type=int, default=50051)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--pred_len", type=int, default=4, help="Forecast horizon in steps (15-min each)"
    )
    parser.add_argument(
        "--teacher_run", default=None, help="Directory of a trained city LSTM bundle"
    )
    parser.add_argument(
        "--scaler_path", default=None, help="Optional safe scaler.npz for buildings"
    )
    parser.add_argument(
        "--scenario", default="baseline",
        help="Scenario label persisted in run artifacts (baseline by default)."
    )
    parser.add_argument(
        "--enable_recommendations", action="store_true",
        help="Enable LLM energy recommendations."
    )
    args = parser.parse_args()

    from src.simulation.reporting import (
        load_building_summaries,
        make_run_id,
        summarise_buildings,
        summarise_federated_audit,
        write_simulation_report,
    )

    counts = _validate_topology(args.num_agents, args.num_districts)
    csvs = _select_household_csvs(args.num_agents)
    city_port = str(args.base_port - 1)
    run_id = make_run_id()
    run_dir = Path("artifacts") / "simulation_runs" / run_id
    audit_dir = run_dir / "federated_audit"
    building_metrics_dir = run_dir / "building_agents"
    run_dir.mkdir(parents=True, exist_ok=True)
    building_metrics_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "scenario": args.scenario,
        "num_agents": args.num_agents,
        "num_districts": args.num_districts,
        "ticks": args.ticks,
        "pred_len": args.pred_len,
        "teacher_run": args.teacher_run,
        "scaler_path": args.scaler_path,
        "base_port": args.base_port,
        "district_agent_counts": counts,
        "household_csvs": csvs,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    procs: list[mp.Process] = []

    city = mp.Process(
        target=_city_proc,
        args=(args.num_districts, city_port, args.teacher_run, str(audit_dir / "city")),
        name="City_Aggregator",
    )
    city.start()
    procs.append(city)
    if not _wait_for_port(args.host, int(city_port)):
        logging.error("City did not start.")
        city.terminate()
        return 2

    district_procs: list[tuple[mp.Process, str, str]] = []
    for i in range(args.num_districts):
        port = str(args.base_port + i)
        did = f"D{i + 1}"
        p = mp.Process(
            target=_district_proc,
            args=(did, counts[i], port, city_port, str(audit_dir)),
            name=f"District_{did}",
        )
        p.start()
        procs.append(p)
        district_procs.append((p, port, did))

    for _, port, did in district_procs:
        if not _wait_for_port(args.host, int(port)):
            logging.error("District %s on :%s did not start.", did, port)
            for p in procs:
                p.terminate()
            return 2

    # Map building → parent district port.
    agent_to_port: list[str] = []
    for d_idx, n_in_district in enumerate(counts):
        for _ in range(n_in_district):
            agent_to_port.append(str(args.base_port + d_idx))

    building_procs: list[mp.Process] = []
    for i in range(args.num_agents):
        ap = mp.Process(
            target=_building_proc,
            args=(
                f"Bldg_{i + 1}",
                csvs[i],
                args.pred_len,
                args.scaler_path,
                agent_to_port[i],
                args.ticks,
                str(building_metrics_dir),
            ),
            name=f"Bldg_{i + 1}",
        )
        ap.start()
        building_procs.append(ap)
        procs.append(ap)

    for ap in building_procs:
        ap.join()
    time.sleep(1.0)
    logging.info("Buildings done. Shutting down districts and city.")
    for p, _, _ in district_procs:
        p.terminate()
        p.join(timeout=5)
    city.terminate()
    city.join(timeout=5)
    metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_simulation_report(
        run_dir,
        metadata=metadata,
        building_summary=summarise_buildings(load_building_summaries(run_dir)),
        federated_summary=summarise_federated_audit(audit_dir),
    )
    logging.info("Simulation artifacts written under %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
