"""
LTM - Long-Term Memory.

Diagram 4: "LTM - Long-term memory · Compressed patterns · Attention pooling ·
Slow update cycle · Persistent (disk)".

Two responsibilities:

  1. Persist every observation to a SQLite store (durable history).
  2. Maintain a pool of compressed *pattern keys* - one mean-pooled feature
     vector per persisted block. Algorithm 4's attention retrieval reads from
     this pool to find historical context most similar to the current query.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np


class LongTermMemory:
    def __init__(
        self,
        agent_id: str,
        db_path: str | os.PathLike,
        feature_dim: int,
        pool_capacity: int = 4096,
        pool_block_size: int = 32,
    ):
        if pool_capacity <= 0 or pool_block_size <= 0:
            raise ValueError("pool sizes must be positive.")
        self.agent_id = agent_id
        self.db_path = Path(db_path)
        self.feature_dim = int(feature_dim)
        self.pool_capacity = int(pool_capacity)
        self.pool_block_size = int(pool_block_size)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

        # In-memory pattern pool: aligned float32 arrays.
        self.keys = np.zeros((0, self.feature_dim), dtype=np.float32)
        self.values = np.zeros((0, self.feature_dim), dtype=np.float32)
        self._block: list[np.ndarray] = []

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id          TEXT PRIMARY KEY,
                    timestamp   TEXT NOT NULL,
                    agent_id    TEXT NOT NULL,
                    kwh         REAL,
                    payload     TEXT
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(timestamp)"
            )
            conn.commit()

    def persist(self, observation: dict, feature_vec: np.ndarray | None = None) -> None:
        """Write the observation to SQLite and (optionally) update the pool."""
        ts = observation.get("timestamp") or datetime.utcnow().isoformat()
        kwh = float(observation.get("kwh", 0.0))
        payload = ";".join(f"{k}={v}" for k, v in observation.items() if k != "timestamp")
        row_id = uuid.uuid4().hex
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO observations(id, timestamp, agent_id, kwh, payload) "
                "VALUES (?,?,?,?,?)",
                (row_id, str(ts), self.agent_id, kwh, payload),
            )
            conn.commit()

        if feature_vec is not None and feature_vec.size == self.feature_dim:
            self._block.append(feature_vec.astype(np.float32, copy=False))
            if len(self._block) >= self.pool_block_size:
                self._flush_block()

    def _flush_block(self) -> None:
        block = np.stack(self._block)
        pooled = block.mean(axis=0, keepdims=True)
        # key == value (mean pooled feature vector); attention treats them as
        # both keys and values in the simplest LTM realisation.
        self.keys = np.concatenate([self.keys, pooled], axis=0)
        self.values = np.concatenate([self.values, pooled], axis=0)
        if self.keys.shape[0] > self.pool_capacity:
            overflow = self.keys.shape[0] - self.pool_capacity
            self.keys = self.keys[overflow:]
            self.values = self.values[overflow:]
        self._block.clear()

    def recent(self, limit: int = 96) -> list[dict]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT timestamp, kwh FROM observations "
                "WHERE agent_id = ? ORDER BY timestamp DESC LIMIT ?",
                (self.agent_id, int(limit)),
            ).fetchall()
        return [{"timestamp": r[0], "kwh": r[1]} for r in reversed(rows)]

    def pool_size(self) -> int:
        return int(self.keys.shape[0])
