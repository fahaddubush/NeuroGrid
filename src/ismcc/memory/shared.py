"""
Shared memory - cross-agent gossip layer with differential-privacy noise.

Diagram 4: "Shared M · Cross-agent sync · Gossip protocol · District-scope ·
Consensus write · Privacy: DP noise".

`SharedMemory` is the front-end the agents talk to. The underlying storage is
swappable via a `Backend` protocol so the same API works in three modes:

  * `InMemoryBackend` - single-process default. Used by unit tests and for
    runs where every agent lives in one Python interpreter.
  * `FileBackend`     - durable JSON-on-disk, safe for multi-process use via
    OS file locks. Used by the multi-process simulation runner so District
    workers and Building workers can share keys without a network broker.
  * `SparkBroadcastBackend` - broadcasts the consensus tensor through the
    District's existing Spark session. Hooked but not enabled by default;
    flip on with `NEUROGRID_SHARED_BACKEND=spark`.

Backend selection is driven by the `NEUROGRID_SHARED_BACKEND` env var
(`memory` / `file` / `spark`). The factory `SharedMemory.from_env()` returns
a default instance ready to use.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Protocol

import numpy as np


# --------------------------------------------------------------------- #
class SharedBackend(Protocol):
    def put(self, agent_id: str, key: str, value: np.ndarray) -> None: ...
    def get(self, agent_id: str, key: str) -> np.ndarray | None: ...
    def collect(self, key: str) -> dict[str, np.ndarray]: ...
    def n_agents(self) -> int: ...


# --------------------------------------------------------------------- #
class InMemoryBackend:
    """Process-local store. Same behaviour as the previous SharedMemory."""

    def __init__(self):
        self._lock = threading.RLock()
        self._store: dict[str, dict[str, np.ndarray]] = {}

    def put(self, agent_id: str, key: str, value: np.ndarray) -> None:
        with self._lock:
            self._store.setdefault(agent_id, {})[key] = value.copy()

    def get(self, agent_id: str, key: str) -> np.ndarray | None:
        with self._lock:
            v = self._store.get(agent_id, {}).get(key)
            return v.copy() if v is not None else None

    def collect(self, key: str) -> dict[str, np.ndarray]:
        with self._lock:
            return {
                aid: bag[key].copy() for aid, bag in self._store.items() if key in bag
            }

    def n_agents(self) -> int:
        with self._lock:
            return len(self._store)


# --------------------------------------------------------------------- #
class FileBackend:
    """JSON-on-disk store keyed by `<root>/<agent_id>/<key>.npy`.

    Multi-process safe via per-key file locks. Numpy arrays serialise via
    np.save / np.load so the disk format is stable across Python versions.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _agent_dir(self, agent_id: str) -> Path:
        d = self.root / agent_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _path(self, agent_id: str, key: str) -> Path:
        return self._agent_dir(agent_id) / f"{key}.npy"

    def put(self, agent_id: str, key: str, value: np.ndarray) -> None:
        path = self._path(agent_id, key)
        # Atomic write: dump to .tmp via an open file (np.save would otherwise
        # append .npy to a string path that doesn't end in .npy), then replace.
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, np.asarray(value, dtype=np.float32), allow_pickle=False)
        os.replace(tmp, path)

    def get(self, agent_id: str, key: str) -> np.ndarray | None:
        path = self._path(agent_id, key)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError):
            return None  # mid-write race - the next read will succeed

    def collect(self, key: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if not self.root.exists():
            return out
        for agent_dir in self.root.iterdir():
            if not agent_dir.is_dir():
                continue
            v = self.get(agent_dir.name, key)
            if v is not None:
                out[agent_dir.name] = v
        return out

    def n_agents(self) -> int:
        if not self.root.exists():
            return 0
        return sum(1 for d in self.root.iterdir() if d.is_dir())


# --------------------------------------------------------------------- #
class SparkBroadcastBackend:
    """Optional Spark-broadcast backend for the District tier.

    Each `put` triggers a re-broadcast; reads consume the broadcast value's
    `.value` field. We keep a process-local mirror so reads don't round-trip
    to the driver every time. A no-op if no SparkSession is bound.
    """

    def __init__(self, spark=None):
        self.spark = spark
        self._mirror = InMemoryBackend()
        self._broadcasts: dict[tuple[str, str], object] = {}

    def put(self, agent_id: str, key: str, value: np.ndarray) -> None:
        self._mirror.put(agent_id, key, value)
        if self.spark is not None:
            self._broadcasts[(agent_id, key)] = self.spark.sparkContext.broadcast(
                np.asarray(value, dtype=np.float32)
            )

    def get(self, agent_id: str, key: str) -> np.ndarray | None:
        bc = self._broadcasts.get((agent_id, key))
        if bc is not None:
            return np.asarray(bc.value)
        return self._mirror.get(agent_id, key)

    def collect(self, key: str) -> dict[str, np.ndarray]:
        return self._mirror.collect(key)

    def n_agents(self) -> int:
        return self._mirror.n_agents()


# --------------------------------------------------------------------- #
class SharedMemory:
    """Front-end with DP noise. The backend is swappable."""

    def __init__(
        self,
        dp_sigma: float = 0.0,
        backend: SharedBackend | None = None,
    ):
        if dp_sigma < 0:
            raise ValueError("dp_sigma must be >= 0.")
        self.dp_sigma = float(dp_sigma)
        self.backend: SharedBackend = backend or InMemoryBackend()

    @classmethod
    def from_env(
        cls,
        dp_sigma: float = 0.0,
        spark=None,
        default_root: str | None = None,
    ) -> "SharedMemory":
        """Pick a backend based on `NEUROGRID_SHARED_BACKEND` (memory / file / spark)."""
        choice = os.getenv("NEUROGRID_SHARED_BACKEND", "memory").lower()
        if choice == "file":
            root = default_root or os.getenv(
                "NEUROGRID_SHARED_DIR",
                os.path.join("artifacts", "shared_memory"),
            )
            return cls(dp_sigma=dp_sigma, backend=FileBackend(root))
        if choice == "spark":
            return cls(dp_sigma=dp_sigma, backend=SparkBroadcastBackend(spark=spark))
        return cls(dp_sigma=dp_sigma, backend=InMemoryBackend())

    # ------------------------------------------------------------------ #
    def write(self, agent_id: str, key: str, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=np.float32)
        if self.dp_sigma > 0:
            value = value + np.random.normal(
                0.0, self.dp_sigma, size=value.shape
            ).astype(value.dtype)
        self.backend.put(agent_id, key, value)

    def read(self, agent_id: str, key: str) -> np.ndarray | None:
        return self.backend.get(agent_id, key)

    def gather(self, key: str) -> dict[str, np.ndarray]:
        return self.backend.collect(key)

    def consensus(self, key: str) -> np.ndarray | None:
        contributions = list(self.gather(key).values())
        if not contributions:
            return None
        return np.stack(contributions).mean(axis=0)

    def __len__(self) -> int:
        return self.backend.n_agents()
