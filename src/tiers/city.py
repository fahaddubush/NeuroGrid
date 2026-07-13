"""
Tier 3 - City Aggregator.

Diagram 4 (right column):
    S: city-wide grid + weather signals
    M: full STM + LTM, city-wide shared M
    C: broadcast W* · policy π* top-down
    X: Large DNN (>~100M params) · global FedAvg + KD
    Goal: city-wide peak shaving
    Privacy: trusted execution
    Tier role: global policy authority

This module:
  * Receives validated parameter summaries from each District.
  * Runs weighted FedAvg over them: W* = Σ_d (N_d / N) · W_d.
  * Optionally blends with the existing teacher (continuous online operation).
  * Runs the convergence monitor ‖W_{t+1} − W_t‖ < ε.
  * Persists one-row city round summaries as Spark-readable Parquet.
  * Serves the freshest W* back down (Districts forward it to Buildings).
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent import futures
from pathlib import Path
from typing import Callable

import torch

from src.federated.audit import append_partitioned_parquet, build_city_round_summary
from src.federated.aggregation import masked_weighted_fedavg
from src.federated.payload import deserialize_update, serialize_update
from src.federated.convergence import ConvergenceMonitor
from src.models.artifacts import load_artifact_bundle
from src.tiers.city_store import CityStateStore
from src.ismcc.grpc_security import allow_insecure, authorize_agent, server_credentials

try:
    import grpc
    from src.proto import neurogrid_pb2, neurogrid_pb2_grpc
except ImportError:
    grpc = None
    neurogrid_pb2 = None
    neurogrid_pb2_grpc = None


_BASE_SERVICER = (
    neurogrid_pb2_grpc.FederatedAggregatorServicer if neurogrid_pb2_grpc else object
)


class CityAggregator(_BASE_SERVICER):
    def __init__(
        self,
        expected_districts: int,
        teacher_run_dir: str | None = None,
        epsilon: float = 1e-3,
        audit_dir: str | None = None,
        state_dir: str | None = None,
        candidate_validator: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], tuple[bool, str]] | None = None,
        max_relative_update: float = 0.25,
    ):
        if neurogrid_pb2_grpc is None:
            raise RuntimeError("grpcio is required to run the City tier.")
        self.expected_districts = int(expected_districts)
        self.monitor = ConvergenceMonitor(epsilon=epsilon, patience=2)
        self.audit_dir = Path(
            audit_dir or os.path.join("artifacts", "federated_audit", "city")
        )
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_validator = candidate_validator
        self.max_relative_update = float(max_relative_update)

        self._lock = threading.Lock()
        state_root = Path(
            state_dir
            or os.getenv("CITY_STATE_DIR")
            or os.path.join("artifacts", "federated_state")
        )
        self.store = CityStateStore(state_root / "city.db")
        recovered = self.store.recover()
        self._pending: dict[int, dict[str, dict]] = {
            round_id: {did: item["state_dict"] for did, item in cohort.items()}
            for round_id, cohort in recovered["pending"].items()
        }
        self._n_samples: dict[int, dict[str, int]] = {
            round_id: {did: item["n_samples"] for did, item in cohort.items()}
            for round_id, cohort in recovered["pending"].items()
        }
        self._masks: dict[int, dict[str, dict[str, torch.Tensor]]] = {
            round_id: {did: item["masks"] for did, item in cohort.items()}
            for round_id, cohort in recovered["pending"].items()
        }
        self._aggregating_rounds: set[int] = set()
        self.current_round = int(recovered["current_round"])
        self.global_state = recovered["global_state"]
        self.converged = bool(recovered["converged"])
        self.monitor.restore(
            stable_rounds=recovered["stable_rounds"], last_drift=recovered["last_drift"]
        )

        if self.global_state is None and teacher_run_dir and Path(teacher_run_dir).exists():
            bundle = load_artifact_bundle(teacher_run_dir, tier="city")
            self.global_state = {k: v.detach().clone() for k, v in bundle["model"].state_dict().items()}
            self.store.save_baseline(self.global_state)
            logging.info("[City] Loaded initial global teacher from %s.", teacher_run_dir)

    # ------------------------------------------------------------------ #
    def SendGradient(self, request, context):
        district_id = request.agent_id
        round_id = int(request.round)
        n_samples = int(request.n_samples) if request.n_samples > 0 else 1
        if not authorize_agent(district_id, context):
            return neurogrid_pb2.AckResponse(
                success=False, message="unauthorized agent identity", round=self.current_round
            )
        with self._lock:
            current = self.current_round
        if round_id < current:
            return neurogrid_pb2.AckResponse(
                success=False, message="stale round", round=current
            )
        if round_id > current:
            return neurogrid_pb2.AckResponse(
                success=False, message="future round", round=current
            )
        if not district_id or not request.state_dict_bytes:
            return neurogrid_pb2.AckResponse(
                success=False, message="invalid empty upload", round=current
            )
        try:
            state_dict, masks, metadata = deserialize_update(request.state_dict_bytes)
        except (RuntimeError, ValueError, TypeError) as exc:
            return neurogrid_pb2.AckResponse(
                success=False, message=f"invalid state payload: {exc}", round=current
            )
        expected_version = f"city-round-{current}"
        if (
            metadata.get("format_version") != 2
            or metadata.get("model_version") != expected_version
            or request.tier != "district"
        ):
            return neurogrid_pb2.AckResponse(
                success=False,
                message=f"expected district update for {expected_version}",
                round=current,
            )

        with self._lock:
            if round_id < self.current_round:
                return neurogrid_pb2.AckResponse(
                    success=False, message="stale round", round=self.current_round
                )
            if round_id > self.current_round + 1:
                return neurogrid_pb2.AckResponse(
                    success=False, message="future round", round=self.current_round
                )
            self._pending.setdefault(round_id, {})[district_id] = state_dict
            self._masks.setdefault(round_id, {})[district_id] = masks
            self._n_samples.setdefault(round_id, {})[district_id] = n_samples
            self.store.append_pending(
                round_id,
                district_id,
                n_samples,
                state_dict,
                masks,
                model_version=metadata.get("model_version", ""),
            )
            ready = (
                round_id == self.current_round
                and len(self._pending[round_id]) >= self.expected_districts
                and round_id not in self._aggregating_rounds
            )
            snapshot: list[tuple[dict, int]] = []
            snapshot_masks: list[dict[str, torch.Tensor]] = []
            district_sample_counts: dict[str, int] = {}
            if ready:
                self._aggregating_rounds.add(round_id)
                snapshot = [
                    (sd, self._n_samples[round_id].get(did, 1))
                    for did, sd in self._pending[round_id].items()
                ]
                snapshot_masks = [
                    self._masks[round_id][did] for did in self._pending[round_id]
                ]
                district_sample_counts = {
                    str(did): int(self._n_samples[round_id].get(did, 1))
                    for did in self._pending[round_id].keys()
                }

        if not ready:
            return neurogrid_pb2.AckResponse(
                success=True, message="received", round=self.current_round
            )

        logging.info(
            "[City] Round %d full (%d districts). Running global FedAvg.",
            round_id,
            len(snapshot),
        )
        updates = [s[0] for s in snapshot]
        weights = [float(s[1]) for s in snapshot]
        try:
            aggregated_delta, aggregate_mask = masked_weighted_fedavg(
                updates, snapshot_masks, weights
            )
        except (ValueError, RuntimeError) as exc:
            with self._lock:
                self._aggregating_rounds.discard(round_id)
            logging.warning("[City] Rejected round %d: %s", round_id, exc)
            return neurogrid_pb2.AckResponse(
                success=False, message=f"invalid update cohort: {exc}", round=self.current_round
            )

        with self._lock:
            if self.current_round != round_id:
                self._aggregating_rounds.discard(round_id)
                return neurogrid_pb2.AckResponse(
                    success=True, message="superseded", round=self.current_round
                )
            prev = self.global_state
            if prev is None:
                # A delta has no meaning without a shared baseline. Do not
                # silently reinterpret it as a complete model checkpoint.
                self._aggregating_rounds.discard(round_id)
                return neurogrid_pb2.AckResponse(
                    success=False,
                    message="global baseline unavailable",
                    round=self.current_round,
                )
            if set(prev) != set(aggregated_delta):
                self._aggregating_rounds.discard(round_id)
                return neurogrid_pb2.AckResponse(
                    success=False, message="model schema mismatch", round=self.current_round
                )
            candidate_state = {
                key: torch.where(
                    aggregate_mask[key],
                    prev[key].detach().cpu().float() + aggregated_delta[key],
                    prev[key].detach().cpu().float(),
                )
                for key in prev
            }
            baseline_norm = torch.sqrt(
                sum(torch.sum(value.detach().float() ** 2) for value in prev.values())
            ).item()
            update_norm = torch.sqrt(
                sum(torch.sum(value.detach().float() ** 2) for value in aggregated_delta.values())
            ).item()
            relative_update = update_norm / max(baseline_norm, 1e-12)
            if relative_update > self.max_relative_update:
                self._aggregating_rounds.discard(round_id)
                return neurogrid_pb2.AckResponse(
                    success=False,
                    message=f"candidate rejected: relative update {relative_update:.4g} exceeds limit",
                    round=self.current_round,
                )
            if self.candidate_validator is not None:
                accepted, reason = self.candidate_validator(prev, candidate_state)
                if not accepted:
                    self._aggregating_rounds.discard(round_id)
                    return neurogrid_pb2.AckResponse(
                        success=False,
                        message=f"candidate rejected: {reason}",
                        round=self.current_round,
                    )
            self.global_state = candidate_state
            converged, drift = self.monitor.update(prev, self.global_state)
            self.converged = converged
            monitor_state = self.monitor.snapshot()
            self.store.commit_round(
                round_id,
                self.global_state,
                converged=converged,
                stable_rounds=int(monitor_state["stable_rounds"]),
                last_drift=monitor_state["last_drift"],
            )
            self._pending.pop(round_id, None)
            self._n_samples.pop(round_id, None)
            self._masks.pop(round_id, None)
            self._aggregating_rounds.discard(round_id)
            self.current_round = round_id + 1

        summary = build_city_round_summary(
            round_id=round_id,
            expected_districts=self.expected_districts,
            district_sample_counts=district_sample_counts,
            drift=drift,
            converged=converged,
            had_previous_global=prev is not None,
        )
        append_partitioned_parquet(
            self.audit_dir / "city_round_summary",
            summary,
            partition_cols=("round_id",),
        )

        logging.info(
            "[City] Round %d done. drift=%.4g  converged=%s", round_id, drift, converged
        )
        return neurogrid_pb2.AckResponse(
            success=True, message="aggregated", round=self.current_round
        )

    def PollDistillation(self, request, context):
        if request is not None and not authorize_agent(request.agent_id, context):
            return neurogrid_pb2.DistilledModel(available=False)
        with self._lock:
            if self.global_state is None:
                return neurogrid_pb2.DistilledModel(available=False)
            encoded = serialize_update(
                self.global_state,
                model_version=f"city-round-{self.current_round}",
            )
            return neurogrid_pb2.DistilledModel(
                available=True,
                round=self.current_round,
                state_dict_bytes=encoded,
                converged=self.converged,
            )


def serve_city(
    expected_districts: int,
    port: str,
    teacher_run_dir: str | None = None,
    epsilon: float = 1e-3,
    audit_dir: str | None = None,
    state_dir: str | None = None,
) -> None:
    if grpc is None:
        raise RuntimeError("grpcio is required to run the City tier.")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    city = CityAggregator(
        expected_districts=expected_districts,
        teacher_run_dir=teacher_run_dir,
        epsilon=epsilon,
        audit_dir=audit_dir,
        state_dir=state_dir,
    )
    neurogrid_pb2_grpc.add_FederatedAggregatorServicer_to_server(city, server)
    bind_host = os.getenv("NEUROGRID_BIND_HOST", "127.0.0.1")
    credentials = server_credentials(grpc)
    address = f"{bind_host}:{port}"
    if credentials is not None:
        server.add_secure_port(address, credentials)
    elif allow_insecure(bind_host):
        server.add_insecure_port(address)
    else:
        raise RuntimeError("Refusing an insecure City listener on a non-loopback host.")
    server.start()
    logging.info(
        "[City] gRPC up on :%s | expected districts=%d", port, expected_districts
    )
    server.wait_for_termination()
