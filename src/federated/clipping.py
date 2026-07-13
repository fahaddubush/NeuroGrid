"""
Algorithm 3 - Step 1: Gradient Clipping.

Per-update L2-norm clipping. From the paper:

    g'_i = g_i * min(1, C / ||g_i||_2)

This prevents any single agent (malicious or numerically pathological) from
dominating the aggregation. Clipping is applied to the *parameter delta*
Delta theta_k, treated as a flattened vector across the whole state dict.
"""
from __future__ import annotations

import torch


def state_dict_l2_norm(state_dict: dict[str, torch.Tensor]) -> float:
    if not state_dict:
        raise ValueError("state_dict must not be empty.")
    sq = 0.0
    for tensor in state_dict.values():
        if not torch.isfinite(tensor).all():
            raise ValueError("state_dict contains NaN or infinity.")
        sq += float(torch.sum(tensor.detach().float() ** 2).item())
    return float(sq ** 0.5)


def clip_state_dict(
    state_dict: dict[str, torch.Tensor],
    threshold: float,
) -> tuple[dict[str, torch.Tensor], float]:
    """Scale every tensor in `state_dict` by min(1, C / ||·||_2).

    Returns the clipped copy and the original L2 norm (for audit).
    """
    if threshold <= 0:
        raise ValueError("clip threshold must be positive.")

    norm = state_dict_l2_norm(state_dict)
    factor = 1.0 if norm <= threshold else (threshold / norm)
    clipped = {k: (v.detach().float() * factor) for k, v in state_dict.items()}
    return clipped, norm


def clip_updates(
    updates: list[dict[str, torch.Tensor]],
    threshold: float,
) -> tuple[list[dict[str, torch.Tensor]], list[float]]:
    """Clip a batch of updates. Returns (clipped_updates, original_norms)."""
    clipped: list[dict[str, torch.Tensor]] = []
    norms: list[float] = []
    for u in updates:
        c, n = clip_state_dict(u, threshold)
        clipped.append(c)
        norms.append(n)
    return clipped, norms
