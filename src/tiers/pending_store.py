"""
SQLite-backed pending-upload store for the District tier.

Replaces the in-memory `_pending` dict with a durable journal so a District
crash mid-round does not silently drop the buildings' Δθ uploads. On restart,
`PendingStore.recover()` returns the round id that was in flight (if any) so
the gRPC servicer can resume from the same point.

Schema:

    pending_uploads:    keyed by (round_id, agent_id) - one row per upload
    round_state:        single-row table holding current_round
    completed_rounds:   audit row inserted when a round is aggregated + popped
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import torch
from src.federated.payload import deserialize_update, serialize_update


class PendingStore:
    def __init__(self, db_path: str | Path, district_id: str):
        self.db_path = Path(db_path)
        self.district_id = str(district_id)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # RLock so methods that hold the lock can call other helpers (e.g.
        # advance_round → current_round) without deadlocking on Windows.
        self._lock = threading.RLock()
        self._init_schema()

    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pending_uploads (
                    round_id        INTEGER NOT NULL,
                    agent_id        TEXT    NOT NULL,
                    n_samples       INTEGER NOT NULL,
                    state_blob      BLOB    NOT NULL,
                    received_at     TEXT    NOT NULL,
                    PRIMARY KEY (round_id, agent_id)
                );
                CREATE INDEX IF NOT EXISTS idx_pending_round
                    ON pending_uploads(round_id);

                CREATE TABLE IF NOT EXISTS round_state (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    current_round   INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO round_state(id, current_round) VALUES (1, 0);

                CREATE TABLE IF NOT EXISTS completed_rounds (
                    round_id        INTEGER PRIMARY KEY,
                    n_uploads       INTEGER NOT NULL,
                    aggregated_at   TEXT    NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pending_uplinks (
                    round_id        INTEGER PRIMARY KEY,
                    n_samples       INTEGER NOT NULL,
                    state_blob      BLOB    NOT NULL,
                    queued_at       TEXT    NOT NULL
                );
                """
            )

    # ------------------------------------------------------------------ #
    @property
    def current_round(self) -> int:
        with self._lock, sqlite3.connect(self.db_path, timeout=30.0) as conn:
            row = conn.execute("SELECT current_round FROM round_state WHERE id=1").fetchone()
            return int(row[0]) if row else 0

    def advance_round(self, expected_round: int) -> int:
        """Atomically advance current_round if it equals `expected_round`."""
        with self._lock, sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cur = conn.execute(
                "UPDATE round_state SET current_round = current_round + 1 "
                "WHERE id = 1 AND current_round = ?",
                (int(expected_round),),
            )
            if cur.rowcount != 1:
                # Read the current value in the same transaction without
                # re-entering self.current_round (avoid lock contention).
                row = conn.execute(
                    "SELECT current_round FROM round_state WHERE id=1"
                ).fetchone()
                actual = int(row[0]) if row else -1
                raise RuntimeError(
                    f"advance_round({expected_round}) lost a race: "
                    f"current_round is now {actual}."
                )
            conn.commit()
            row = conn.execute(
                "SELECT current_round FROM round_state WHERE id=1"
            ).fetchone()
            return int(row[0]) if row else int(expected_round) + 1

    # ------------------------------------------------------------------ #
    def append_upload(
        self,
        round_id: int,
        agent_id: str,
        n_samples: int,
        state_dict: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> int:
        """Persist an upload. Returns the count of distinct agents in this round.

        Duplicate (round_id, agent_id) overwrites the previous row - last write
        wins, matching the in-memory behaviour of the previous orchestrator.
        """
        blob = serialize_update(state_dict, masks=masks)
        ts = datetime.now(timezone.utc).isoformat()

        with self._lock, sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute(
                "INSERT INTO pending_uploads(round_id, agent_id, n_samples, state_blob, received_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(round_id, agent_id) DO UPDATE SET "
                "  n_samples = excluded.n_samples, "
                "  state_blob = excluded.state_blob, "
                "  received_at = excluded.received_at",
                (int(round_id), str(agent_id), int(n_samples), blob, ts),
            )
            row = conn.execute(
                "SELECT COUNT(*) FROM pending_uploads WHERE round_id = ?",
                (int(round_id),),
            ).fetchone()
            conn.commit()
            return int(row[0])

    def round_size(self, round_id: int) -> int:
        with self._lock, sqlite3.connect(self.db_path, timeout=30.0) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM pending_uploads WHERE round_id = ?",
                (int(round_id),),
            ).fetchone()
            return int(row[0]) if row else 0

    def snapshot_round(self, round_id: int) -> list[dict]:
        with self._lock, sqlite3.connect(self.db_path, timeout=30.0) as conn:
            rows = conn.execute(
                "SELECT agent_id, n_samples, state_blob FROM pending_uploads "
                "WHERE round_id = ? ORDER BY received_at ASC",
                (int(round_id),),
            ).fetchall()
        items: list[dict] = []
        for agent_id, n_samples, blob in rows:
            sd, masks, metadata = deserialize_update(blob)
            items.append(
                {
                    "agent_id": str(agent_id),
                    "n_samples": int(n_samples),
                    "state_dict": sd,
                    "masks": masks,
                    "metadata": metadata,
                }
            )
        return items

    def pop_round(self, round_id: int, n_uploads: int) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute("DELETE FROM pending_uploads WHERE round_id = ?", (int(round_id),))
            conn.execute(
                "INSERT OR REPLACE INTO completed_rounds(round_id, n_uploads, aggregated_at) "
                "VALUES (?, ?, ?)",
                (int(round_id), int(n_uploads), ts),
            )
            conn.commit()

    def finalize_round(
        self,
        round_id: int,
        n_uploads: int,
        uplink_state: dict[str, torch.Tensor] | None = None,
        uplink_masks: dict[str, torch.Tensor] | None = None,
        uplink_samples: int = 0,
    ) -> int:
        """Atomically archive uploads and advance the active round.

        Keeping these writes in one SQLite transaction prevents a crash from
        deleting a completed cohort while leaving ``current_round`` behind.
        """
        ts = datetime.now(timezone.utc).isoformat()
        uplink_blob = None
        if uplink_state is not None:
            uplink_blob = serialize_update(uplink_state, masks=uplink_masks)
        with self._lock, sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT current_round FROM round_state WHERE id=1"
            ).fetchone()
            actual = int(row[0]) if row else -1
            if actual != int(round_id):
                conn.rollback()
                raise RuntimeError(
                    f"finalize_round({round_id}) lost a race: current_round is {actual}."
                )
            conn.execute(
                "INSERT OR REPLACE INTO completed_rounds(round_id, n_uploads, aggregated_at) "
                "VALUES (?, ?, ?)",
                (int(round_id), int(n_uploads), ts),
            )
            conn.execute(
                "DELETE FROM pending_uploads WHERE round_id = ?", (int(round_id),)
            )
            conn.execute(
                "UPDATE round_state SET current_round = current_round + 1 WHERE id=1"
            )
            if uplink_blob is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO pending_uplinks"
                    "(round_id, n_samples, state_blob, queued_at) VALUES (?, ?, ?, ?)",
                    (int(round_id), int(uplink_samples), uplink_blob, ts),
                )
            conn.commit()
            return actual + 1

    def pending_uplink(self) -> dict | None:
        with self._lock, sqlite3.connect(self.db_path, timeout=30.0) as conn:
            row = conn.execute(
                "SELECT round_id, n_samples, state_blob FROM pending_uplinks "
                "ORDER BY round_id LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        state, masks, metadata = deserialize_update(row[2])
        return {
            "round_id": int(row[0]),
            "n_samples": int(row[1]),
            "state_dict": state,
            "masks": masks,
            "metadata": metadata,
        }

    def complete_uplink(self, round_id: int) -> None:
        with self._lock, sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute("DELETE FROM pending_uplinks WHERE round_id = ?", (int(round_id),))
            conn.commit()

    # ------------------------------------------------------------------ #
    def recover(self) -> dict:
        """Inspect the store after a crash. Returns a status report.

        Does NOT mutate state - the caller's startup logic decides whether to
        resume the in-flight round (typical) or to abandon it.
        """
        with self._lock, sqlite3.connect(self.db_path, timeout=30.0) as conn:
            current = int(
                conn.execute("SELECT current_round FROM round_state WHERE id=1").fetchone()[0]
            )
            in_flight = conn.execute(
                "SELECT round_id, COUNT(*) FROM pending_uploads "
                "GROUP BY round_id ORDER BY round_id",
            ).fetchall()
            completed = int(
                conn.execute("SELECT COUNT(*) FROM completed_rounds").fetchone()[0]
            )
        return {
            "current_round": current,
            "in_flight_rounds": [
                {"round_id": int(r), "n_uploads": int(n)} for r, n in in_flight
            ],
            "completed_rounds": completed,
        }

    def reset(self) -> None:
        """Drop every row. Test-only helper."""
        with self._lock, sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.executescript(
                """
                DELETE FROM pending_uploads;
                DELETE FROM completed_rounds;
                DELETE FROM pending_uplinks;
                UPDATE round_state SET current_round = 0 WHERE id = 1;
                """
            )
            conn.commit()
