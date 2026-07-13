"""
Algorithm 3 - Step 4: Convergence check.

Paper: stop the FL round loop once

    || W_{t+1} - W_t || < epsilon

The convergence monitor tracks consecutive rounds within tolerance so a single
flat round does not falsely terminate a run that is still moving. This mirrors
the "patience" pattern used elsewhere in the project for early stopping.
"""
from __future__ import annotations

import torch


class ConvergenceMonitor:
    def __init__(self, epsilon: float = 1e-3, patience: int = 1):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if patience < 1:
            raise ValueError("patience must be >= 1.")
        self.epsilon = float(epsilon)
        self.patience = int(patience)
        self._stable_rounds = 0
        self._last_drift: float | None = None

    @staticmethod
    def state_drift(
        prev: dict[str, torch.Tensor],
        curr: dict[str, torch.Tensor],
    ) -> float:
        sq = 0.0
        for k in curr:
            if k not in prev:
                continue
            diff = curr[k].detach().float() - prev[k].detach().float()
            sq += float(torch.sum(diff * diff).item())
        return float(sq ** 0.5)

    def update(
        self,
        prev: dict[str, torch.Tensor] | None,
        curr: dict[str, torch.Tensor],
    ) -> tuple[bool, float]:
        if prev is None:
            self._last_drift = float("inf")
            self._stable_rounds = 0
            return False, float("inf")
        drift = self.state_drift(prev, curr)
        self._last_drift = drift
        if drift < self.epsilon:
            self._stable_rounds += 1
        else:
            self._stable_rounds = 0
        converged = self._stable_rounds >= self.patience
        return converged, drift

    @property
    def last_drift(self) -> float | None:
        return self._last_drift

    def snapshot(self) -> dict[str, float | int | None]:
        return {"stable_rounds": self._stable_rounds, "last_drift": self._last_drift}

    def restore(self, *, stable_rounds: int = 0, last_drift: float | None = None) -> None:
        self._stable_rounds = max(0, int(stable_rounds))
        self._last_drift = None if last_drift is None else float(last_drift)
