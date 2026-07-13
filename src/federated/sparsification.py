"""
TopK sparsification of building uplinks.

Diagram 4 (right column): "C: Uplink Δθ only - TopK sparsification" for the
Building tier. We keep only the magnitudes-largest fraction of coordinates per
parameter tensor and zero the rest. The receiving District treats zeros as
"no update" so the global model remains well-defined.
"""
from __future__ import annotations

import torch


def topk_sparsify(
    delta: dict[str, torch.Tensor],
    keep_ratio: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Keep top `keep_ratio` of coordinates by absolute magnitude per tensor."""
    if not 0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1].")

    out: dict[str, torch.Tensor] = {}
    for key, tensor in delta.items():
        flat = tensor.detach().float().reshape(-1)
        n = flat.numel()
        k = max(1, int(n * keep_ratio))
        if k >= n:
            out[key] = tensor.detach().float().clone()
            continue
        _, idx = torch.topk(flat.abs(), k=k, largest=True, sorted=False)
        sparse = torch.zeros_like(flat)
        sparse[idx] = flat[idx]
        out[key] = sparse.reshape(tensor.shape)
    return out


def topk_sparsify_with_masks(
    delta: dict[str, torch.Tensor], keep_ratio: float = 0.1
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return Top-K values plus explicit contribution masks."""
    sparse = topk_sparsify(delta, keep_ratio=keep_ratio)
    masks: dict[str, torch.Tensor] = {}
    for key, original in delta.items():
        flat = original.detach().float().reshape(-1)
        n = flat.numel()
        k = max(1, int(n * keep_ratio))
        mask = torch.ones(n, dtype=torch.bool, device=flat.device)
        if k < n:
            mask.zero_()
            _, idx = torch.topk(flat.abs(), k=k, largest=True, sorted=False)
            mask[idx] = True
        masks[key] = mask.reshape(original.shape).cpu()
    return sparse, masks


def sparsity(state_dict: dict[str, torch.Tensor]) -> float:
    """Fraction of zero coordinates across the whole state dict."""
    total = 0
    zero = 0
    for t in state_dict.values():
        flat = t.detach().reshape(-1)
        total += int(flat.numel())
        zero += int((flat == 0).sum().item())
    return zero / total if total else 0.0
