"""
Aggregation primitives used by the District and City tiers.

Implements the two reductions the paper calls out explicitly:

  * `weighted_fedavg(updates, weights)` - weighted hierarchical reduction:
        W = sum_k (n_k / N) * Delta theta_k
    The per-agent weight is the data-quality score q_k (number of valid
    samples that contributed to the local round, scaled to a probability).

  * `trimmed_mean(updates)` - Step 4 of the Byzantine pipeline: average the
    surviving (clipped + Krum-accepted) updates element-wise. We deliberately
    keep this as a standalone helper so tests can validate it independently
    of the gRPC servicer.
"""
from __future__ import annotations

import torch


def _check_aligned(updates: list[dict[str, torch.Tensor]]) -> list[str]:
    if not updates:
        raise ValueError("No updates to aggregate.")
    keys = list(updates[0].keys())
    keyset = set(keys)
    for u in updates[1:]:
        if set(u.keys()) != keyset:
            raise ValueError("All updates must share the same parameter keys.")
    for key in keys:
        expected_shape = updates[0][key].shape
        for update in updates:
            value = update[key]
            if value.shape != expected_shape:
                raise ValueError(f"Parameter '{key}' has inconsistent shapes.")
            if not torch.isfinite(value).all():
                raise ValueError(f"Parameter '{key}' contains NaN or infinity.")
    return keys


def weighted_fedavg(
    updates: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    keys = _check_aligned(updates)
    n = len(updates)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError("weights length must match updates length.")
    if any(not torch.isfinite(torch.tensor(w)).item() or w < 0 for w in weights):
        raise ValueError("weights must be finite and non-negative.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Sum of weights must be positive.")
    norm = [w / total for w in weights]

    out: dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = torch.stack([u[k].detach().float() for u in updates], dim=0)
        w = torch.tensor(norm, dtype=stacked.dtype).reshape(
            (-1,) + (1,) * (stacked.dim() - 1)
        )
        out[k] = (stacked * w).sum(dim=0)
    return out


def trimmed_mean(
    updates: list[dict[str, torch.Tensor]],
    trim_ratio: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Coordinate-wise trimmed mean. With trim_ratio=0 this reduces to mean."""
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5).")
    keys = _check_aligned(updates)
    n = len(updates)
    cut = int(trim_ratio * n)

    out: dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = torch.stack([u[k].detach().float() for u in updates], dim=0)
        if cut > 0 and n - 2 * cut >= 1:
            sorted_, _ = torch.sort(stacked, dim=0)
            out[k] = sorted_[cut : n - cut].mean(dim=0)
        else:
            out[k] = stacked.mean(dim=0)
    return out


def masked_trimmed_mean(
    updates: list[dict[str, torch.Tensor]],
    masks: list[dict[str, torch.Tensor]],
    trim_ratio: float = 0.0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Coordinate-wise trimmed mean over clients that transmitted a coordinate."""
    keys = _check_aligned(updates)
    if len(masks) != len(updates):
        raise ValueError("masks length must match updates length")
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5).")
    out: dict[str, torch.Tensor] = {}
    contributed: dict[str, torch.Tensor] = {}
    for key in keys:
        values = torch.stack([update[key].detach().float() for update in updates])
        present = torch.stack([mask[key].detach().bool() for mask in masks])
        if present.shape != values.shape:
            raise ValueError(f"mask for '{key}' has an invalid shape")
        # Sort absent values to the end, then trim only within the actual
        # contributor count for each coordinate.
        sortable = torch.where(present, values, torch.full_like(values, float("inf")))
        sorted_values, _ = torch.sort(sortable, dim=0)
        counts = present.sum(dim=0)
        result = torch.zeros_like(values[0])
        for count in torch.unique(counts).tolist():
            count = int(count)
            if count == 0:
                continue
            cut = int(trim_ratio * count)
            if count - 2 * cut < 1:
                cut = 0
            coordinate_mask = counts == count
            mean = sorted_values[cut : count - cut].mean(dim=0)
            result = torch.where(coordinate_mask, mean, result)
        out[key] = result
        contributed[key] = counts > 0
    return out, contributed


def masked_weighted_fedavg(
    updates: list[dict[str, torch.Tensor]],
    masks: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """FedAvg normalized independently over contributors per coordinate."""
    keys = _check_aligned(updates)
    n = len(updates)
    if len(masks) != n:
        raise ValueError("masks length must match updates length")
    weights = [1.0] * n if weights is None else weights
    if len(weights) != n or any(not torch.isfinite(torch.tensor(w)) or w < 0 for w in weights):
        raise ValueError("weights must be finite, non-negative, and aligned")
    out: dict[str, torch.Tensor] = {}
    contributed: dict[str, torch.Tensor] = {}
    for key in keys:
        values = torch.stack([update[key].detach().float() for update in updates])
        present = torch.stack([mask[key].detach().bool() for mask in masks])
        if present.shape != values.shape:
            raise ValueError(f"mask for '{key}' has an invalid shape")
        shape = (-1,) + (1,) * (values.dim() - 1)
        weight_tensor = torch.tensor(weights, dtype=values.dtype).reshape(shape)
        effective = weight_tensor * present.to(values.dtype)
        denominator = effective.sum(dim=0)
        numerator = (values * effective).sum(dim=0)
        has_value = denominator > 0
        out[key] = torch.where(has_value, numerator / denominator.clamp_min(1e-12), 0.0)
        contributed[key] = has_value
    return out, contributed
