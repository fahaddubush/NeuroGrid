"""
ISMCC Agent Core (Algorithm 2, Diagram 2 centre block).

Glues the four ISMCC modules into a single per-tick state machine. The same
core is used by the building agent (where it owns a Sensing pipe, a small
LSTM, an episodic buffer, and a Communication uplink) and - at lower
frequency - by the district / city tiers (where the Sensing pipe is replaced
by an inbound gRPC queue and the Computation module is sized up).

State is the paper's (s_t, r_t, a_t) triple. The agent core's `tick()` method
returns a dict containing the chosen action so the orchestrator can log,
score, and persist it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import Future, ThreadPoolExecutor
import logging
from typing import Optional

import numpy as np

from src.ismcc.memory.retrieval import attention_retrieval


@dataclass
class AgentState:
    s_t: dict | None = None
    r_t: float = 0.0
    a_t: dict = field(default_factory=lambda: {"actions": ["MONITOR"], "reasoning": "init"})
    drifting: bool = False
    last_forecast: Optional[np.ndarray] = None


class AgentCore:
    """Coordinator that wires S, M, X, C together.

    The four modules are passed in by the tier so one Core class serves all
    three tiers. Memory is a dict of the four sub-stores (stm/ltm/episodic/
    shared) - only the building tier carries all four; the district omits
    `shared` entries it doesn't own; the city tier may keep only stm + ltm.
    """

    def __init__(
        self,
        agent_id: str,
        sensing,
        memory: dict,
        compute,
        comms=None,
        recommender=None,
        recommend_every: int = 96,
    ):
        self.agent_id = agent_id
        self.sensing = sensing
        self.memory = memory
        self.compute = compute
        self.comms = comms
        self.state = AgentState()
        # Optional LLM advisor. None ⇒ behaviour unchanged.
        # identical to before. When present, fires a recommendation every
        # `recommend_every` ticks (default 96 = once per 24 h at 15-min).
        self.recommender = recommender
        self.recommend_every = max(1, int(recommend_every))
        self._tick_count = 0
        self._recommendation_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"recommender-{agent_id}")
            if recommender is not None
            else None
        )
        self._recommendation_future: Future | None = None

    # ------------------------------------------------------------------ #
    # Memory helpers (Algorithm 4)
    # ------------------------------------------------------------------ #
    def _retrieve_context(self, query_vec: np.ndarray) -> Optional[np.ndarray]:
        ltm = self.memory.get("ltm")
        if ltm is None or ltm.pool_size() == 0:
            return None
        ctx, _ = attention_retrieval(query_vec, ltm.keys, ltm.values, top_k=8)
        return ctx

    # ------------------------------------------------------------------ #
    # Per-tick lifecycle.
    # ------------------------------------------------------------------ #
    def tick(self, raw_reading: dict, train: bool = True) -> dict:
        # 1) S: preprocess o_t and check drift
        o_t = self.sensing.preprocess(raw_reading)
        drifting = self.sensing.detect_drift(o_t)

        # 2) M: write into STM + persist + (optionally) update pattern pool
        stm = self.memory["stm"]
        stm.write(o_t)
        ltm = self.memory.get("ltm")
        if ltm is not None:
            ltm.persist(o_t, feature_vec=o_t["feature_vector"])

        # 3) Algorithm 4 retrieval: pull historical context similar to o_t
        context = self._retrieve_context(o_t["feature_vector"])

        # 4) X: train one mini-batch SGD step (paper g_t = ∇L(W_t; x_t, y_t))
        train_loss = None
        if train:
            train_loss = self.compute.train_step(stm.window())

        # 5) X: forecast + optimise - produces action a_t
        forecast = self.compute.forecast(stm.window())
        if forecast is not None:
            self.compute.update_capacity(o_t["kwh"])
            predicted_next = float(forecast[0])
            decision = self.compute.optimise_action(predicted_next)
        else:
            decision = {"actions": ["MONITOR"], "reasoning": "STM warming up."}

        # 6) M: write episodic memory if anything significant happened.
        episodic = self.memory.get("episodic")
        reward = -abs(o_t["kwh"] - (predicted_next if forecast is not None else o_t["kwh"]))
        if episodic is not None:
            actions = decision["actions"]
            significant = drifting or (
                "MAINTAIN" not in actions and "MONITOR" not in actions
            )
            if significant:
                episodic.write(
                    observation=o_t["raw"],
                    action=decision,
                    reward=reward,
                )

        # 7) Update agent state (s_t, r_t, a_t)
        self.state.s_t = o_t
        self.state.r_t = reward
        self.state.a_t = decision
        self.state.drifting = drifting
        self.state.last_forecast = forecast

        # 8) Optional LLM recommendation.
        recommendation = None
        self._tick_count += 1

        if self._recommendation_future is not None and self._recommendation_future.done():
            try:
                recommendation = self._recommendation_future.result()
            except Exception as e:
                logging.warning("[%s] Recommendation failed: %s", self.agent_id, e)
            finally:
                self._recommendation_future = None

        if (
            self.recommender is not None
            and forecast is not None
            and len(forecast) >= 4
            and self._tick_count % self.recommend_every == 0
            and self._recommendation_future is None
        ):
            try:
                schedule = self.compute.optimise_schedule(np.asarray(forecast))
                # Copy mutable inputs before handing them to the worker. The
                # sensing/training tick must never wait on a local LLM server.
                self._recommendation_future = self._recommendation_executor.submit(
                    self.recommender.recommend,
                    forecast=np.asarray(forecast).copy(),
                    schedule=dict(schedule),
                    household_id=self.agent_id,
                )
            except Exception as e:
                logging.warning("[%s] Recommendation failed: %s", self.agent_id, e)
                recommendation = None

        return {
            "observation": o_t,
            "decision": decision,
            "drifting": drifting,
            "train_loss": train_loss,
            "context": context,
            "forecast": forecast,
            "reward": reward,
            "recommendation": recommendation,
        }

    def close(self) -> None:
        if self._recommendation_executor is not None:
            self._recommendation_executor.shutdown(wait=False, cancel_futures=True)
