"""
Differential-privacy primitives for the building uplink.

Diagram 5 (right column): "DP-SGD: Gaussian noise σ added · Moments accountant
→ (ε, δ)-DP". Diagram 4 (Building tier) lists "Privacy: DP-SGD".

The online building path perturbs one clipped round-level model delta. This is
an output-perturbation mechanism, not per-example DP-SGD; any published privacy
claim must therefore justify the round-delta sensitivity bound C:

  1. Δθ_k for round k is the sum of all per-tick parameter updates between
     uplinks (not a single sample's gradient). The `clip_state_dict` already
     in `src.federated.clipping` is the L2 clip step.
  2. Apply Gaussian noise N(0, (σ · C)² I) to the clipped Δθ before TopK
     sparsification and gRPC upload.

For the privacy accountant we lean on `opacus.accountants.RDPAccountant` when
available (tight RDP composition); otherwise we use conservative basic
composition, splitting δ across events and summing their Gaussian bounds.
"""
from __future__ import annotations

import math
import json
import os
from pathlib import Path
from dataclasses import dataclass, field

import torch

from src.federated.clipping import clip_state_dict, state_dict_l2_norm

try:  # Optional, much tighter accounting if installed.
    from opacus.accountants import RDPAccountant  # type: ignore
    _HAS_OPACUS = True
except Exception:  # pragma: no cover - exercised only when opacus is absent.
    RDPAccountant = None  # type: ignore
    _HAS_OPACUS = False


def gaussian_noise_state_dict(
    state_dict: dict[str, torch.Tensor],
    sigma: float,
    sensitivity: float,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Add iid N(0, (σ · C)²) noise to every coordinate.

    `sensitivity` is the L2 clip threshold C used immediately upstream. The
    standard deviation of the noise added to each coordinate is σ·C.
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    if sensitivity <= 0:
        raise ValueError("sensitivity (clip C) must be > 0.")
    if sigma == 0:
        return {k: v.detach().float().clone() for k, v in state_dict.items()}
    std = float(sigma) * float(sensitivity)
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        v = v.detach().float()
        noise = torch.empty_like(v).normal_(mean=0.0, std=std, generator=generator)
        out[k] = v + noise
    return out


def clip_and_noise(
    state_dict: dict[str, torch.Tensor],
    sensitivity: float,
    sigma: float,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], float]:
    """Convenience: clip to sensitivity then add Gaussian noise.

    Returns (noised_state_dict, original_l2_norm).
    """
    clipped, original = clip_state_dict(state_dict, threshold=sensitivity)
    noised = gaussian_noise_state_dict(
        clipped, sigma=sigma, sensitivity=sensitivity, generator=generator
    )
    return noised, original


# ---------------------------------------------------------------------- #
# Privacy accountant.
# ---------------------------------------------------------------------- #
@dataclass
class DPEvent:
    sigma: float
    sample_rate: float = 1.0  # subsampling rate (1.0 = full participation)


@dataclass
class DPAccountant:
    """Composition accountant.

    Records each Gaussian application and returns an ε bound for a target δ.
    Uses Opacus's RDPAccountant when available; otherwise a loose closed-form
    Gaussian-mechanism composition.
    """

    events: list[DPEvent] = field(default_factory=list)

    def record(self, sigma: float, sample_rate: float = 1.0) -> None:
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError("sample_rate must be in (0, 1].")
        if sigma <= 0:
            return  # no privacy spent
        self.events.append(DPEvent(sigma=float(sigma), sample_rate=float(sample_rate)))

    def epsilon(self, delta: float = 1e-5) -> float:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1).")
        if not self.events:
            return 0.0
        if _HAS_OPACUS:
            acc = RDPAccountant()
            for e in self.events:
                acc.step(noise_multiplier=e.sigma, sample_rate=e.sample_rate)
            return float(acc.get_epsilon(delta=delta))
        # Conservative basic composition. Allocate delta equally across events
        # and ignore subsampling amplification; summing each Gaussian bound is
        # looser than RDP but does not omit positive composition terms.
        T = len(self.events)
        per_event_delta = delta / T
        per_step = [
            math.sqrt(2.0 * math.log(1.25 / per_event_delta)) / e.sigma
            for e in self.events
        ]
        return float(sum(per_step))

    @property
    def using_opacus(self) -> bool:
        return _HAS_OPACUS

    def reset(self) -> None:
        self.events.clear()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps([{"sigma": event.sigma, "sample_rate": event.sample_rate} for event in self.events]),
            encoding="utf-8",
        )
        os.replace(tmp, target)

    @classmethod
    def load(cls, path: str | Path) -> "DPAccountant":
        target = Path(path)
        accountant = cls()
        if not target.exists():
            return accountant
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Invalid privacy-accountant journal.")
        for item in payload:
            accountant.record(float(item["sigma"]), float(item["sample_rate"]))
        return accountant
