"""
Episodic memory - experience replay buffer.

Diagram 4: "Episodic · Experience replay buffer B = {(o, a, r)} · Priority
sampling · Feeds X training · FIFO eviction".

We store (observation, action, reward) triples. Sampling weights are
proportional to |reward|^priority_alpha + epsilon so episodes with strong
positive *or* negative outcomes get revisited more often than neutral ones.
"""
from __future__ import annotations

import json
import os
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np


class EpisodicMemory:
    def __init__(
        self,
        agent_id: str,
        capacity: int = 10_000,
        priority_alpha: float = 0.6,
        episode_dir: str | os.PathLike | None = None,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.agent_id = agent_id
        self.capacity = int(capacity)
        self.priority_alpha = float(priority_alpha)
        self._buffer: deque[dict] = deque(maxlen=self.capacity)
        self.episode_dir: Path | None = Path(episode_dir) if episode_dir else None
        if self.episode_dir is not None:
            self.episode_dir.mkdir(parents=True, exist_ok=True)

    def write(self, observation: dict, action: str | dict, reward: float) -> None:
        episode = {
            "id": uuid.uuid4().hex,
            "timestamp": observation.get("timestamp") or datetime.utcnow().isoformat(),
            "agent_id": self.agent_id,
            "observation": observation,
            "action": action,
            "reward": float(reward),
        }
        self._buffer.append(episode)
        self._maybe_persist(episode)

    def _maybe_persist(self, episode: dict) -> None:
        if self.episode_dir is None:
            return
        # Significant-event journal: only persist absolute |reward| > 0.5
        # episodes so the disk doesn't fill with neutral monitoring ticks.
        if abs(episode["reward"]) <= 0.5:
            return
        ts = episode["timestamp"].replace(":", "-").replace(".", "-")
        fname = f"{ts}_{episode['id'][:8]}.json"
        with open(self.episode_dir / fname, "w", encoding="utf-8") as f:
            json.dump(episode, f, indent=2, default=str)

    def __len__(self) -> int:
        return len(self._buffer)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> list[dict]:
        n = len(self._buffer)
        if n == 0:
            return []
        rng = rng or np.random.default_rng()
        rewards = np.array([abs(e["reward"]) for e in self._buffer], dtype=np.float64)
        weights = rewards ** self.priority_alpha + 1e-3
        probs = weights / weights.sum()
        idx = rng.choice(n, size=min(batch_size, n), replace=False, p=probs)
        return [self._buffer[int(i)] for i in idx]

    def all(self) -> list[dict]:
        return list(self._buffer)
