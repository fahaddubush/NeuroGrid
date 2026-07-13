"""
Tier 2 - District Orchestrator.

Diagram 4 (right column):
    S: aggregated district readings
    M: STM + LTM + Shared M (district scope)
    C: up + down + peer · Byzantine filter on received messages
    X: Mid DNN (<10M) · FedAvg aggregator
    Goal: district balance
    Privacy: authenticated transport + optional round-level perturbation
    Tier role: aggregator + KD relay

This module is a gRPC servicer. Each round it:
  1. Receives clipped Δθ uplinks from buildings (persisted to a SQLite
     PendingStore for crash durability).
  2. Runs the Spark-backed Byzantine pipeline (clip → Krum → trimmed mean),
     persisting both per-agent audit rows and one-row district summaries.
  3. Forwards the regional summary G_d to the City tier.
  4. Polls the City for the latest global model W* and convergence flag, then
     serves that freshest state back down to children polling.
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent import futures
from pathlib import Path

import torch
from pyspark.sql import SparkSession

from src.federated.spark_byzantine import run_spark_byzantine_round
from src.federated.payload import deserialize_update, serialize_update
from src.ismcc.communication import CommunicationModule
from src.ismcc.grpc_security import allow_insecure, authorize_agent, server_credentials
from src.tiers.pending_store import PendingStore

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


class DistrictOrchestrator(_BASE_SERVICER):
    def __init__(
        self,
        district_id: str,
        expected_agents: int,
        spark: SparkSession,
        f_byzantine: int = 1,
        clip_threshold: float = 1.0,
        trim_ratio: float = 0.1,
        city_host: str | None = None,
        city_port: str | None = None,
        audit_dir: str | None = None,
        state_dir: str | None = None,
    ):
        if neurogrid_pb2_grpc is None:
            raise RuntimeError("grpcio is required to run the District tier.")
        self.district_id = district_id
        self.expected_agents = int(expected_agents)
        self.spark = spark
        self.f_byzantine = int(f_byzantine)
        self.clip_threshold = float(clip_threshold)
        self.trim_ratio = float(trim_ratio)
        self.audit_dir = audit_dir or os.path.join(
            "artifacts", "federated_audit"
        )
        Path(self.audit_dir).mkdir(parents=True, exist_ok=True)

        # Durable pending store - replaces the in-memory _pending / _n_samples
        # dicts so a District crash mid-round does not drop building uploads.
        state_root = Path(
            state_dir
            or os.getenv("DISTRICT_STATE_DIR")
            or os.path.join("artifacts", "federated_state")
        )
        state_root.mkdir(parents=True, exist_ok=True)
        self.store = PendingStore(
            db_path=state_root / f"{district_id}.db",
            district_id=district_id,
        )
        recovery = self.store.recover()
        logging.info(
            "[District %s] Pending store recovery: current_round=%d "
            "in_flight=%s completed_rounds=%d",
            district_id,
            recovery["current_round"],
            recovery["in_flight_rounds"],
            recovery["completed_rounds"],
        )

        # Lock guards aggregation snapshot + uplink, not store reads/writes
        # (the store is internally thread-safe).
        self._aggr_lock = threading.Lock()
        self.global_state: dict[str, torch.Tensor] | None = None
        self.global_converged = False
        self.uplink: CommunicationModule | None = None
        if city_host and city_port:
            self.uplink = CommunicationModule(
                agent_id=district_id,
                district_host=city_host,
                district_port=city_port,
                topk_ratio=1.0,  # district→city uplinks are dense
            )

    @property
    def current_round(self) -> int:
        return self.store.current_round

    def _flush_uplink(self) -> bool:
        if self.uplink is None:
            return False
        item = self.store.pending_uplink()
        if item is None:
            return True
        self.uplink.current_round = item["round_id"]
        self.uplink.model_version = f"city-round-{item['round_id']}"
        if self.uplink.upload_delta(
            item["state_dict"], n_samples=item["n_samples"], masks=item["masks"]
        ):
            self.store.complete_uplink(item["round_id"])
            return True
        return False

    # ------------------------------------------------------------------ #
    def SendGradient(self, request, context):
        agent_id = request.agent_id
        round_id = int(request.round)
        n_samples = int(request.n_samples) if request.n_samples > 0 else 1
        if not authorize_agent(agent_id, context):
            return neurogrid_pb2.AckResponse(
                success=False, message="unauthorized agent identity", round=self.current_round
            )

        cur = self.current_round
        if round_id < cur:
            return neurogrid_pb2.AckResponse(
                success=False, message="stale round", round=cur
            )
        if round_id > cur:
            return neurogrid_pb2.AckResponse(
                success=False, message="future round", round=cur
            )
        if not agent_id or not request.state_dict_bytes:
            return neurogrid_pb2.AckResponse(
                success=False, message="invalid empty upload", round=cur
            )
        try:
            state_dict, masks, metadata = deserialize_update(request.state_dict_bytes)
        except (RuntimeError, ValueError, TypeError) as exc:
            return neurogrid_pb2.AckResponse(
                success=False, message=f"invalid state payload: {exc}", round=cur
            )
        if metadata.get("format_version") != 2 or request.tier != "building":
            return neurogrid_pb2.AckResponse(
                success=False, message="unsupported client payload version or tier", round=cur
            )

        # Persist the upload - duplicates overwrite, matching prior semantics.
        n_in_round = self.store.append_upload(
            round_id=round_id,
            agent_id=agent_id,
            n_samples=n_samples,
            state_dict=state_dict,
            masks=masks,
        )

        # Only the round currently being collected can fire aggregation.
        if round_id != self.current_round or n_in_round < self.expected_agents:
            return neurogrid_pb2.AckResponse(
                success=True, message="received", round=self.current_round
            )

        # Single-flight aggregation: only one thread may snapshot the round.
        with self._aggr_lock:
            cur = self.current_round
            if cur != round_id:
                # Another thread already advanced.
                return neurogrid_pb2.AckResponse(
                    success=True, message="superseded", round=cur
                )
            snapshot_items = self.store.snapshot_round(round_id)
            if len(snapshot_items) < self.expected_agents:
                # Last-minute uploads slipped between the count check and the
                # snapshot - defer; the late upload's own SendGradient call
                # will trigger aggregation.
                return neurogrid_pb2.AckResponse(
                    success=True, message="received", round=cur
                )

            logging.info(
                "[District %s] Round %d full (%d agents). Running Spark Byzantine round.",
                self.district_id,
                round_id,
                len(snapshot_items),
            )
            aggregated, aggregated_masks, _audit, _summary = run_spark_byzantine_round(
                spark=self.spark,
                incoming=snapshot_items,
                f_byzantine=self.f_byzantine,
                clip_threshold=self.clip_threshold,
                trim_ratio=self.trim_ratio,
                audit_path=self.audit_dir,
                district_id=self.district_id,
                round_id=round_id,
            )
            n_total = sum(int(item["n_samples"]) for item in snapshot_items)
            self.store.finalize_round(
                round_id,
                n_uploads=len(snapshot_items),
                uplink_state=aggregated if self.uplink is not None else None,
                uplink_masks=aggregated_masks if self.uplink is not None else None,
                uplink_samples=n_total,
            )
            # ``aggregated`` is a regional delta, not a complete model. It is
            # never exposed to buildings as a downlink checkpoint; only a City
            # response may populate ``global_state`` below.

        # Forward summary to City tier outside the lock (gRPC I/O can block).
        self._flush_uplink()

        return neurogrid_pb2.AckResponse(
            success=True, message="aggregated", round=self.current_round
        )

    def PollDistillation(self, request, context):
        if request is not None and not authorize_agent(request.agent_id, context):
            return neurogrid_pb2.DistilledModel(available=False)
        if self.uplink is not None:
            self._flush_uplink()
            latest_global = self.uplink.pull_global()
            if latest_global is not None:
                self.global_state = latest_global
            self.global_converged = bool(self.uplink.last_poll_converged)
        if self.global_state is None:
            return neurogrid_pb2.DistilledModel(available=False)
        encoded = serialize_update(
            self.global_state, model_version=f"district-city-round-{self.current_round}"
        )
        return neurogrid_pb2.DistilledModel(
            available=True,
            round=self.current_round,
            state_dict_bytes=encoded,
            converged=self.global_converged,
        )


def serve_district(
    district_id: str,
    expected_agents: int,
    port: str,
    city_host: str | None = None,
    city_port: str | None = None,
    f_byzantine: int = 1,
    clip_threshold: float = 1.0,
    audit_dir: str | None = None,
) -> None:
    if grpc is None:
        raise RuntimeError("grpcio is required to run the District tier.")
    spark = (
        SparkSession.builder.appName(f"NeuroGrid_District_{district_id}")
        .master(os.getenv("SPARK_MASTER", "local[1]"))
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "1g"))
        .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", "1g"))
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "4"))
        .getOrCreate()
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    orch = DistrictOrchestrator(
        district_id=district_id,
        expected_agents=expected_agents,
        spark=spark,
        city_host=city_host,
        city_port=city_port,
        f_byzantine=f_byzantine,
        clip_threshold=clip_threshold,
        audit_dir=audit_dir,
    )
    neurogrid_pb2_grpc.add_FederatedAggregatorServicer_to_server(orch, server)
    bind_host = os.getenv("NEUROGRID_BIND_HOST", "127.0.0.1")
    credentials = server_credentials(grpc)
    address = f"{bind_host}:{port}"
    if credentials is not None:
        server.add_secure_port(address, credentials)
    elif allow_insecure(bind_host):
        server.add_insecure_port(address)
    else:
        raise RuntimeError("Refusing an insecure District listener on a non-loopback host.")
    server.start()
    logging.info(
        "[District %s] gRPC up on :%s | expected agents=%d", district_id, port, expected_agents
    )
    try:
        server.wait_for_termination()
    finally:
        spark.stop()
