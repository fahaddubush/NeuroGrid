"""Helpers for resolving atomically published, versioned datasets."""
from __future__ import annotations

from pathlib import Path


def resolve_current_dataset(root: str | Path) -> Path:
    root = Path(root)
    pointer = root / "CURRENT"
    if not pointer.is_file():
        return root
    run_id = pointer.read_text(encoding="utf-8").strip()
    normalized = run_id.replace("\\", "/")
    if not run_id or "/" in normalized or run_id in {".", ".."}:
        raise ValueError(f"Invalid dataset CURRENT pointer at {pointer}.")
    resolved = root / "runs" / run_id
    if not resolved.is_dir():
        raise FileNotFoundError(f"Published dataset run does not exist: {resolved}")
    return resolved
