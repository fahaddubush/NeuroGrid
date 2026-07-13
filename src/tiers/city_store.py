"""Transactional SQLite persistence for the City federated authority."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import torch

from src.federated.payload import deserialize_update, serialize_update


class CityStateStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS city_state (
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    current_round INTEGER NOT NULL,
                    global_blob BLOB,
                    converged INTEGER NOT NULL,
                    stable_rounds INTEGER NOT NULL,
                    last_drift REAL
                );
                INSERT OR IGNORE INTO city_state
                    (id, current_round, global_blob, converged, stable_rounds, last_drift)
                    VALUES (1, 0, NULL, 0, 0, NULL);
                CREATE TABLE IF NOT EXISTS city_pending (
                    round_id INTEGER NOT NULL,
                    district_id TEXT NOT NULL,
                    n_samples INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    PRIMARY KEY(round_id, district_id)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30.0)

    def recover(self) -> dict:
        with self._connect() as conn:
            state_row = conn.execute(
                "SELECT current_round, global_blob, converged, stable_rounds, last_drift "
                "FROM city_state WHERE id=1"
            ).fetchone()
            pending_rows = conn.execute(
                "SELECT round_id, district_id, n_samples, payload FROM city_pending "
                "ORDER BY round_id, district_id"
            ).fetchall()
        global_state = None
        if state_row[1] is not None:
            global_state, _masks, _meta = deserialize_update(state_row[1])
        pending: dict[int, dict[str, dict]] = {}
        for round_id, district_id, n_samples, payload in pending_rows:
            update, masks, metadata = deserialize_update(payload)
            pending.setdefault(int(round_id), {})[str(district_id)] = {
                "state_dict": update,
                "masks": masks,
                "metadata": metadata,
                "n_samples": int(n_samples),
            }
        return {
            "current_round": int(state_row[0]),
            "global_state": global_state,
            "converged": bool(state_row[2]),
            "stable_rounds": int(state_row[3]),
            "last_drift": state_row[4],
            "pending": pending,
        }

    def save_baseline(self, state: dict[str, torch.Tensor]) -> None:
        blob = serialize_update(state, model_version="city-baseline")
        with self._connect() as conn:
            conn.execute("UPDATE city_state SET global_blob=? WHERE id=1", (blob,))
            conn.commit()
    def append_pending(
        self,
        round_id: int,
        district_id: str,
        n_samples: int,
        state: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        model_version: str = "",
    ) -> None:
        blob = serialize_update(state, masks=masks, model_version=model_version)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO city_pending"
                "(round_id, district_id, n_samples, payload) VALUES (?, ?, ?, ?)",
                (int(round_id), str(district_id), int(n_samples), blob),
            )
            conn.commit()

    def commit_round(
        self,
        expected_round: int,
        global_state: dict[str, torch.Tensor],
        *,
        converged: bool,
        stable_rounds: int,
        last_drift: float | None,
    ) -> None:
        blob = serialize_update(
            global_state, model_version=f"city-round-{expected_round + 1}"
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            actual = int(
                conn.execute("SELECT current_round FROM city_state WHERE id=1").fetchone()[0]
            )
            if actual != int(expected_round):
                conn.rollback()
                raise RuntimeError(
                    f"City commit lost a race: expected {expected_round}, found {actual}."
                )
            conn.execute(
                "UPDATE city_state SET current_round=?, global_blob=?, converged=?, "
                "stable_rounds=?, last_drift=? WHERE id=1",
                (
                    int(expected_round) + 1,
                    blob,
                    int(bool(converged)),
                    int(stable_rounds),
                    last_drift,
                ),
            )
            conn.execute("DELETE FROM city_pending WHERE round_id=?", (int(expected_round),))
            conn.commit()
