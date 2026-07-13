"""
STM - Short-Term Memory.

Diagram 4: "STM - Short-term memory · Sliding window τ · Recent o_t states ·
Fast read/write · Volatile (RAM)".

Implementation: a fixed-size deque of observations. The Building tier sets
τ = 24h (= 96 steps at 15-min resolution).
"""
from __future__ import annotations

from collections import deque
from typing import Iterable


class ShortTermMemory:
    def __init__(self, capacity: int = 96):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self._buffer: deque[dict] = deque(maxlen=self.capacity)

    def write(self, observation: dict) -> None:
        self._buffer.append(observation)

    def window(self) -> list[dict]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def is_warm(self, min_size: int) -> bool:
        return len(self._buffer) >= min_size

    def extend(self, observations: Iterable[dict]) -> None:
        for o in observations:
            self.write(o)

    def clear(self) -> None:
        self._buffer.clear()
