"""
Algorithm 3 - Step 2: Krum scoring (Blanchard et al., 2017).

For each candidate update g_i, compute the sum of squared L2 distances to its
n - f - 2 nearest neighbours (where n is the number of received updates and f
is the expected number of Byzantine clients). The update with the lowest score
is the most "central" honest update.

We expose two operating modes used by the District tier:

  * `krum_select(updates, f)` - return the index of the single most trustworthy
    update (vanilla Krum). Useful when only one survivor is forwarded upward.
  * `multi_krum_select(updates, f, k)` - return the indices of the k lowest-
    scoring updates (Multi-Krum). The District tier feeds these into the
    trimmed-mean step before pushing the regional summary G_d to the City.

Both functions also return the per-agent score vector and the indices of the
rejected outliers so the audit log captures the decision.
"""
from __future__ import annotations

import torch


def _flatten(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [state_dict[k].detach().float().reshape(-1) for k in sorted(state_dict)]
    )


def pairwise_sq_distances(updates: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    """n x n matrix of squared Euclidean distances between flattened updates."""
    flats = torch.stack([_flatten(u) for u in updates])
    # ||x-y||² = ||x||² + ||y||² - 2x·y avoids materialising an
    # n×n×parameter_count broadcast tensor.
    norms = torch.sum(flats * flats, dim=1, keepdim=True)
    return (norms + norms.T - 2.0 * (flats @ flats.T)).clamp_min_(0.0)


def krum_scores(
    updates: list[dict[str, torch.Tensor]],
    f: int,
) -> torch.Tensor:
    """Score(g_i) = sum over n-f-2 nearest neighbours of ||g_i - g_j||^2."""
    n = len(updates)
    if n == 0:
        return torch.zeros(0)
    if f < 0:
        raise ValueError("Byzantine fraction f must be >= 0.")
    if f > 0 and n < 2 * f + 3:
        raise ValueError(f"Krum requires n >= 2f + 3; received n={n}, f={f}.")
    if f == 0 and n < 3:
        return torch.zeros(n)
    m = n - f - 2

    dists = pairwise_sq_distances(updates)
    dists.fill_diagonal_(float("inf"))
    nearest, _ = torch.topk(dists, k=m, dim=-1, largest=False)
    return nearest.sum(dim=-1)


def krum_select(
    updates: list[dict[str, torch.Tensor]],
    f: int,
) -> tuple[int, torch.Tensor]:
    """Return the index of the single Krum-best update."""
    scores = krum_scores(updates, f)
    if scores.numel() == 0:
        raise ValueError("No updates to score.")
    best = int(torch.argmin(scores).item())
    return best, scores


def multi_krum_select(
    updates: list[dict[str, torch.Tensor]],
    f: int,
    k: int = None,
) -> tuple[list[int], list[int], torch.Tensor]:
    """Multi-Krum: return indices of the k lowest-scoring updates as accepted,
    the remainder as rejected, plus the score vector."""
    n = len(updates)
    if n == 0:
        return [], [], torch.zeros(0)
    if k is None:
        k = max(1, n - f)
    k = max(1, min(k, n))

    scores = krum_scores(updates, f)
    order = torch.argsort(scores).tolist()
    accepted = sorted(order[:k])
    rejected = sorted(order[k:])
    return accepted, rejected, scores
