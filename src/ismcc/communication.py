"""
C - Communication module (ISMCC Algorithm 2, Diagram 2 lower-right).

Building tier responsibilities (Diagram 4):
    Uplink:   send Δθ gradient with TopK sparsification
    Downlink: receive W* / π* from parent
    Peer:     shared-memory sync; Byzantine filter on received messages
    Scheduler: round-robin or priority cadence

The transport is gRPC. Imports are guarded so unit tests can exercise the
sparsification + buffering logic without a running gRPC server.
"""
from __future__ import annotations

import logging
import os


import torch

from src.federated.dp import DPAccountant, clip_and_noise
from src.federated.payload import deserialize_update, serialize_update
from src.federated.sparsification import topk_sparsify_with_masks
from src.ismcc.grpc_security import allow_insecure, client_credentials

try:
    import grpc
    from src.proto import neurogrid_pb2, neurogrid_pb2_grpc
except ImportError:
    grpc = None
    neurogrid_pb2 = None
    neurogrid_pb2_grpc = None


class CommunicationModule:
    """gRPC client wrapper used by the building agent.

    upload_delta() - Δθ TopK + serialize + SendGradient
    pull_global()  - PollDistillation; returns the latest W* state dict
    """

    def __init__(
        self,
        agent_id: str,
        district_host: str | None = None,
        district_port: str | None = None,
        topk_ratio: float = 0.1,
        dp_sigma: float = 0.0,
        dp_clip_C: float = 1.0,
        dp_sample_rate: float = 1.0,
        model_version: str = "",
        dp_accountant_path: str | None = None,
        dp_max_epsilon: float | None = None,
        dp_delta: float = 1e-5,
    ):
        self.agent_id = agent_id
        self.host = district_host or os.getenv("AGGREGATOR_HOST", "127.0.0.1")
        self.port = str(district_port or os.getenv("AGGREGATOR_PORT", "50051"))
        self.topk_ratio = float(topk_ratio)
        self.dp_sigma = float(dp_sigma)
        self.dp_clip_C = float(dp_clip_C)
        self.dp_sample_rate = float(dp_sample_rate)
        self.dp_accountant_path = dp_accountant_path or os.path.join(
            "artifacts", "privacy", f"{agent_id}.json"
        )
        self.dp_accountant = DPAccountant.load(self.dp_accountant_path)
        self.dp_max_epsilon = (
            float(dp_max_epsilon)
            if dp_max_epsilon is not None
            else float(os.getenv("DP_MAX_EPSILON", "inf"))
        )
        self.dp_delta = float(dp_delta)
        self.model_version = str(model_version)
        self._compression_residual: dict[str, torch.Tensor] = {}
        self.current_round = 0
        self._last_applied_round = -1
        self.last_poll_converged = False
        self.upload_attempts = 0
        self.upload_successes = 0
        self.upload_failures = 0
        self.poll_attempts = 0
        self.poll_successes = 0
        self.poll_unavailable = 0
        self.poll_errors = 0

        if grpc is None:
            self.channel = None
            self.stub = None
        else:
            credentials = client_credentials(grpc)
            target = f"{self.host}:{self.port}"
            if credentials is not None:
                self.channel = grpc.secure_channel(target, credentials)
            elif allow_insecure(self.host):
                self.channel = grpc.insecure_channel(target)
            else:
                raise RuntimeError(
                    "Refusing insecure gRPC to a non-loopback host. Configure mTLS "
                    "or explicitly set NEUROGRID_ALLOW_INSECURE=1."
                )
            self.stub = neurogrid_pb2_grpc.FederatedAggregatorStub(self.channel)

    @property
    def connected(self) -> bool:
        return self.stub is not None

    def upload_delta(
        self,
        delta_state: dict[str, torch.Tensor],
        n_samples: int = 1,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> bool:
        """DP-clip → noise → TopK sparsify → serialise → send. Returns True on ACK."""
        self.upload_attempts += 1
        if self.stub is None or neurogrid_pb2 is None:
            logging.debug("[%s] gRPC unavailable; skipping upload.", self.agent_id)
            self.upload_failures += 1
            return False
        if self.dp_sigma > 0:
            # Round-level output perturbation: clip Δθ to sensitivity C, add
            # Gaussian noise, record the privacy event in the accountant.
            payload, _orig_norm = clip_and_noise(
                delta_state,
                sensitivity=self.dp_clip_C,
                sigma=self.dp_sigma,
            )
            self.dp_accountant.record(
                sigma=self.dp_sigma, sample_rate=self.dp_sample_rate
            )
            if self.dp_accountant.epsilon(self.dp_delta) > self.dp_max_epsilon:
                self.dp_accountant.events.pop()
                logging.error("[%s] Privacy budget exhausted; upload blocked.", self.agent_id)
                self.upload_failures += 1
                return False
            # Persist before attempting the RPC: an ambiguous timeout may mean
            # the privacy-relevant payload reached the server.
            self.dp_accountant.save(self.dp_accountant_path)
        else:
            payload = delta_state
        next_residual: dict[str, torch.Tensor] | None = None
        if masks is None:
            if self.dp_sigma == 0:
                payload = {
                    key: value.detach().float()
                    + self._compression_residual.get(key, torch.zeros_like(value)).to(value.device)
                    for key, value in payload.items()
                }
            sparse, masks = topk_sparsify_with_masks(payload, keep_ratio=self.topk_ratio)
            if self.dp_sigma == 0:
                next_residual = {
                    key: (payload[key].detach().cpu().float() - sparse[key].detach().cpu().float())
                    for key in payload
                }
        else:
            sparse = payload
        encoded = serialize_update(
            sparse, masks=masks, model_version=self.model_version
        )
        payload = neurogrid_pb2.WeightPayload(
            agent_id=self.agent_id,
            round=self.current_round,
            n_samples=int(n_samples),
            state_dict_bytes=encoded,
            tier="building" if self.topk_ratio < 1.0 else "district",
        )
        try:
            ack = self.stub.SendGradient(payload, timeout=10.0)
            if ack.success:
                if next_residual is not None:
                    self._compression_residual = next_residual
                logging.debug("[%s] District ACK round=%d", self.agent_id, self.current_round)
                self.upload_successes += 1
                return True
            if ack.message == "stale round":
                # The server already committed this idempotent upload; this is
                # success from a durable-outbox perspective.
                self.current_round = max(self.current_round, int(ack.round))
                if next_residual is not None:
                    self._compression_residual = next_residual
                self.upload_successes += 1
                return True
            logging.warning("[%s] District NACK: %s", self.agent_id, ack.message)
            self.upload_failures += 1
            return False
        except grpc.RpcError as e:
            logging.warning("[%s] gRPC upload failed: %s", self.agent_id, e.details())
            self.upload_failures += 1
            return False

    def pull_global(self) -> dict[str, torch.Tensor] | None:
        self.poll_attempts += 1
        if self.stub is None or neurogrid_pb2 is None:
            self.poll_errors += 1
            return None
        req = neurogrid_pb2.PollRequest(
            agent_id=self.agent_id, current_round=self.current_round
        )
        try:
            res = self.stub.PollDistillation(req, timeout=10.0)
        except grpc.RpcError as e:
            logging.warning("[%s] gRPC poll failed: %s", self.agent_id, e.details())
            self.poll_errors += 1
            return None
        self.last_poll_converged = bool(getattr(res, "converged", False))
        if not res.available or res.round <= self._last_applied_round:
            self.poll_unavailable += 1
            return None
        self.current_round = max(self.current_round, res.round)
        self._last_applied_round = res.round
        state, _masks, metadata = deserialize_update(res.state_dict_bytes)
        self.model_version = metadata.get("model_version", self.model_version)
        self.poll_successes += 1
        return state

    def metrics(self) -> dict[str, int | float | bool]:
        return {
            "upload_attempts": int(self.upload_attempts),
            "upload_successes": int(self.upload_successes),
            "upload_failures": int(self.upload_failures),
            "poll_attempts": int(self.poll_attempts),
            "poll_successes": int(self.poll_successes),
            "poll_unavailable": int(self.poll_unavailable),
            "poll_errors": int(self.poll_errors),
            "last_poll_converged": bool(self.last_poll_converged),
            "current_round": int(self.current_round),
            "epsilon": float(self.dp_accountant.epsilon(self.dp_delta)),
        }

    def close(self) -> None:
        if self.channel is not None:
            self.channel.close()
