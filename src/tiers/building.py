"""
Tier 1 - Building Agent.

Diagram 4 (right column): Building tier characteristics
    S: IoT + smart meters
    M: STM (τ ≈ 24h) · Episodic (~10k samples)
    C: Uplink Δθ only · TopK sparsified
    X: Small DNN (<1M params) · clipped/noised round updates
    Goal: local energy optimisation
    Privacy: optional round-level Gaussian output perturbation
    Tier role: gradient producer only

This module wires together S/M/C/X (via AgentCore), drives the per-tick
lifecycle, and uploads Δθ to its parent District after every federation cadence.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from src.data.streaming import HouseholdStream
from src.data.feature_pipeline import load_scaler
from src.ismcc import (
    AgentCore,
    SensingModule,
    ComputationModule,
    CommunicationModule,
    ShortTermMemory,
    LongTermMemory,
    EpisodicMemory,
)
from src.data.schema import STEPS_PER_DAY
from src.utils.paths import workspace_root as _default_workspace_root


class BuildingAgent:
    """Tier-1 ISMCC agent. One per household."""

    DEFAULT_FEDERATION_INTERVAL = 15  # ticks

    def __init__(
        self,
        agent_id: str,
        csv_path: str,
        pred_len: int,
        scaler_path: Optional[str] = None,
        district_host: Optional[str] = None,
        district_port: Optional[str] = None,
        federation_interval: int = DEFAULT_FEDERATION_INTERVAL,
        topk_ratio: float = 0.1,
        workspace_root: Optional[str] = None,
        dp_sigma: float = 0.0,
        dp_clip_C: float = 1.0,
        enable_recommendations: bool = False,
    ):
        self.agent_id = agent_id
        self.federation_interval = int(federation_interval)
        self.pred_len = int(pred_len)

        scaler = load_scaler(scaler_path) if scaler_path and os.path.exists(scaler_path) else None

        # Module wiring (S, M, X, C)
        self.sensing = SensingModule()
        self.compute = ComputationModule(
            agent_id=agent_id, pred_len=pred_len, tier="building", scaler=scaler
        )
        self.comms = CommunicationModule(
            agent_id=agent_id,
            district_host=district_host,
            district_port=district_port,
            topk_ratio=topk_ratio,
            dp_sigma=dp_sigma,
            dp_clip_C=dp_clip_C,
        )

        from src.llm.recommender import EnergyRecommender
        recommender = EnergyRecommender() if enable_recommendations else None

        root = Path(workspace_root).expanduser().resolve() if workspace_root else _default_workspace_root()
        def _under(rel: str) -> Path:
            p = Path(rel).expanduser()
            return p if p.is_absolute() else (root / p).resolve()
        ltm_dir = _under(os.getenv("LTM_DB_DIR", "data/district_logs"))
        ep_dir = _under(os.getenv("EPISODIC_MEMORY_DIR", "docs/system_memory/episodic")) / agent_id
        memory = {
            "stm": ShortTermMemory(capacity=STEPS_PER_DAY),
            "ltm": LongTermMemory(
                agent_id=agent_id,
                db_path=ltm_dir / f"{agent_id}_ltm.db",
                feature_dim=self.compute.model.encoder.input_size,
            ),
            "episodic": EpisodicMemory(
                agent_id=agent_id,
                capacity=10_000,
                episode_dir=ep_dir,
            ),
        }

        self.core = AgentCore(
            agent_id=agent_id,
            sensing=self.sensing,
            memory=memory,
            compute=self.compute,
            comms=self.comms,
            recommender=recommender,
            recommend_every=STEPS_PER_DAY,
        )

        self.stream = HouseholdStream(csv_path, household_id=None)
        self._baseline_snapshot = self.compute.snapshot()
        self._has_global_baseline = False
        self._pending_delta = None
        self._pending_delta_round: int | None = None
        self._uploaded_round: int | None = None
        self._metrics = {
            "agent_id": self.agent_id,
            "csv_path": str(csv_path),
            "pred_len": int(self.pred_len),
            "federation_interval": int(self.federation_interval),
            "ticks_completed": 0,
            "federation_ticks": 0,
            "bootstrap_events": 0,
            "distillation_events": 0,
            "upload_successes": 0,
            "upload_failures": 0,
            "drift_events": 0,
            "train_steps": 0,
            "recommendations_emitted": 0,
            "stream_exhausted": False,
        }
        self._action_counts: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def _maybe_federate(self, tick_idx: int) -> None:
        # Every `federation_interval` ticks (and on tick 0 to bootstrap).
        is_fed_tick = tick_idx == 0 or (tick_idx > 0 and tick_idx % self.federation_interval == 0)
        if not is_fed_tick:
            return
        self._metrics["federation_ticks"] += 1

        # Downlink first: pull global model W*
        global_state = self.comms.pull_global()
        if global_state is not None:
            if tick_idx == 0:
                logging.info("[%s] Bootstrap: adopting global teacher.", self.agent_id)
                self.compute.adopt_global(global_state)
                self._has_global_baseline = True
                self._metrics["bootstrap_events"] += 1
            else:
                logging.info("[%s] Distill receive: KD adapt from global.", self.agent_id)
                # KD adapt against the most recent STM context
                stm = self.core.memory["stm"].window()
                # Build a temporary teacher with the broadcast state
                from src.models.city_lstm import CityLSTM, tier_size
                cfg = tier_size("city")
                teacher = CityLSTM(pred_len=self.pred_len, **cfg)
                # Tolerate dim mismatch by adopting only same-shape tensors.
                t_state = teacher.state_dict()
                for k, v in global_state.items():
                    if k in t_state and t_state[k].shape == v.shape:
                        t_state[k] = v
                teacher.load_state_dict(t_state)
                self.compute.distill_from(teacher, stm)
                self._metrics["distillation_events"] += 1
            self._baseline_snapshot = self.compute.snapshot()
            self._pending_delta = None
            self._pending_delta_round = None
            self._uploaded_round = None
            return

        # If we never got a teacher, we're still in baseline mode - nothing to upload.
        if tick_idx == 0:
            return
        if not self._has_global_baseline:
            logging.warning(
                "[%s] No global baseline received; refusing to upload an incomparable delta.",
                self.agent_id,
            )
            return

        round_id = int(self.comms.current_round)
        if self._uploaded_round == round_id:
            return

        # Uplink Δθ relative to the last accepted baseline.
        if self._pending_delta is None or self._pending_delta_round != round_id:
            self._pending_delta = self.compute.parameter_delta(self._baseline_snapshot)
            self._pending_delta_round = round_id
        sent = self.comms.upload_delta(
            self._pending_delta, n_samples=len(self.core.memory["stm"])
        )
        if sent:
            # Receipt means the District durably owns this round's update. The
            # baseline changes only after a newer global checkpoint arrives.
            self._uploaded_round = round_id
            self._pending_delta = None
            self._pending_delta_round = None
            self._metrics["upload_successes"] += 1
        else:
            self._metrics["upload_failures"] += 1

    # ------------------------------------------------------------------ #
    def step(self, tick_idx: int) -> Optional[dict]:
        """Run one tick. Returns the AgentCore tick dict, or None if stream exhausted."""
        try:
            reading = next(self.stream)
        except StopIteration:
            self._metrics["stream_exhausted"] = True
            return None
        self._maybe_federate(tick_idx)
        out = self.core.tick(reading)
        self._metrics["ticks_completed"] += 1
        if out.get("drifting"):
            self._metrics["drift_events"] += 1
        if out.get("train_loss") is not None:
            self._metrics["train_steps"] += 1
        if out.get("recommendation") is not None:
            self._metrics["recommendations_emitted"] += 1
        for action in out.get("decision", {}).get("actions", []):
            self._action_counts[action] = self._action_counts.get(action, 0) + 1
        return out

    def run(self, max_ticks: int = 10_000) -> None:
        try:
            for tick in range(max_ticks):
                result = self.step(tick)
                if result is None:
                    logging.info("[%s] Stream exhausted after %d ticks.", self.agent_id, tick)
                    return
        finally:
            self.core.close()
            self.comms.close()

    def summary(self) -> dict:
        payload = dict(self._metrics)
        payload["action_counts"] = dict(sorted(self._action_counts.items()))
        payload["communication"] = self.comms.metrics()
        return payload

    def write_summary(self, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.agent_id}.json"
        path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True), encoding="utf-8")
        return path
