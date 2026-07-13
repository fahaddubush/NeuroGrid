"""Versioned, validated serialization for federated tensor updates."""
from __future__ import annotations

import hashlib
import io
from typing import Any

import torch

FORMAT_VERSION = 2
DEFAULT_MAX_PAYLOAD_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_PARAMETERS = 20_000_000


def state_schema_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
    return digest.hexdigest()


def validate_state(
    state: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor] | None = None,
    *,
    max_parameters: int = DEFAULT_MAX_PARAMETERS,
) -> None:
    if not isinstance(state, dict) or not state:
        raise ValueError("state must be a non-empty tensor dictionary")
    total = 0
    for key, tensor in state.items():
        if not isinstance(key, str) or not key:
            raise ValueError("state keys must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"state entry '{key}' is not a tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"state entry '{key}' must be a dense strided tensor")
        if not tensor.is_floating_point():
            raise ValueError(f"state entry '{key}' must be floating point")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"state entry '{key}' contains NaN or infinity")
        total += tensor.numel()
        if total > max_parameters:
            raise ValueError("state exceeds the parameter-count limit")
    if masks is None:
        return
    if set(masks) != set(state):
        raise ValueError("sparsity masks must have exactly the state keys")
    for key, mask in masks.items():
        if not isinstance(mask, torch.Tensor) or mask.shape != state[key].shape:
            raise ValueError(f"mask for '{key}' has an invalid shape")
        if mask.dtype != torch.bool:
            raise ValueError(f"mask for '{key}' must have boolean dtype")


def serialize_update(
    state: dict[str, torch.Tensor],
    *,
    masks: dict[str, torch.Tensor] | None = None,
    model_version: str = "",
) -> bytes:
    validate_state(state, masks)
    if masks is None:
        masks = {key: torch.ones_like(value, dtype=torch.bool) for key, value in state.items()}
    envelope: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_version": str(model_version),
        "schema_hash": state_schema_hash(state),
        "state": {key: value.detach().cpu().float() for key, value in state.items()},
        "masks": {key: value.detach().cpu().bool() for key, value in masks.items()},
    }
    buf = io.BytesIO()
    torch.save(envelope, buf)
    return buf.getvalue()


def deserialize_update(
    blob: bytes,
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    max_parameters: int = DEFAULT_MAX_PARAMETERS,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    if not blob:
        raise ValueError("empty update payload")
    if len(blob) > max_payload_bytes:
        raise ValueError("update payload exceeds the byte-size limit")
    loaded = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and loaded.get("format_version") == FORMAT_VERSION:
        state = loaded.get("state")
        masks = loaded.get("masks")
        metadata = {
            "format_version": FORMAT_VERSION,
            "model_version": str(loaded.get("model_version", "")),
            "schema_hash": str(loaded.get("schema_hash", "")),
        }
    elif isinstance(loaded, dict) and all(isinstance(v, torch.Tensor) for v in loaded.values()):
        # Read-only compatibility for pre-v2 local journals. New network writes
        # always use the explicit envelope.
        state = loaded
        masks = {key: torch.ones_like(value, dtype=torch.bool) for key, value in state.items()}
        metadata = {
            "format_version": 1,
            "model_version": "legacy",
            "schema_hash": state_schema_hash(state),
        }
    else:
        raise ValueError("unsupported federated payload format")
    validate_state(state, masks, max_parameters=max_parameters)
    if metadata["schema_hash"] != state_schema_hash(state):
        raise ValueError("payload schema hash does not match its tensors")
    return state, masks, metadata
