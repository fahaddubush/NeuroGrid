"""
Algorithm 4 - Memory Module Retrieval & Update.

Implements the two formulas from the paper:

    Score        = softmax( Q · K^T / sqrt(d_k) )
    Context      = sum_j  Score_j · V_j
    M_{t+1}      = (1 - lambda) · M_t  +  lambda · M_new
"""
from __future__ import annotations

import numpy as np


def attention_retrieval(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention over the LTM pool.

    Args:
        query: (d_k,)
        keys:  (N, d_k)
        values:(N, d_v)
        top_k: if set, restrict to the top_k highest-scoring keys before
               normalizing scores.

    Returns:
        (context_vector (d_v,), score_vector (N,))
    """
    if keys.ndim != 2 or values.ndim != 2:
        raise ValueError("keys and values must be 2D arrays.")
    if keys.shape[0] == 0:
        return np.zeros(values.shape[1] if values.ndim == 2 else query.shape[0]), np.zeros(0)
    if keys.shape[0] != values.shape[0]:
        raise ValueError("keys and values must share the leading dimension.")
    if query.ndim != 1 or query.shape[0] != keys.shape[1]:
        raise ValueError("query dimension must match keys' last dim.")

    d_k = keys.shape[1]
    logits = (keys @ query) / np.sqrt(d_k)

    if top_k is not None and top_k < keys.shape[0]:
        idx = np.argpartition(-logits, kth=top_k - 1)[:top_k]
        masked = np.full_like(logits, fill_value=-np.inf)
        masked[idx] = logits[idx]
        logits = masked

    logits = logits - np.max(logits[np.isfinite(logits)], initial=0.0)
    exps = np.exp(logits)
    exps[~np.isfinite(exps)] = 0.0
    denom = exps.sum()
    if denom <= 0:
        return np.zeros(values.shape[1], dtype=values.dtype), np.zeros_like(exps)
    scores = exps / denom
    context = scores @ values
    return context, scores


def gated_update(prior: np.ndarray, new: np.ndarray, lambda_: float) -> np.ndarray:
    """M_{t+1} = (1 - λ) · M_t + λ · M_new   - paper's anti-forgetting rule."""
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda must be in [0, 1].")
    return (1.0 - lambda_) * prior + lambda_ * new
