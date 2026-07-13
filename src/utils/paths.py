"""
Workspace-aware path resolution.

`workspace_root()` returns the project root directory. Resolution order:
    1. `WORKSPACE_ROOT` env var (absolute path, useful for containers).
    2. Auto-derive: `Path(__file__).resolve().parents[2]` - three up from
       `src/utils/paths.py` is the repo root.

`resolve_path(p)` makes any string / Path absolute. If `p` is already absolute
it is returned unchanged. If relative, it is joined under the workspace root.
This lets `.env` declare e.g.
    HEAPO_DATA_DIR=data/raw/heapo_data/heapo_data/smart_meter_data/15min
and have it work on every machine without per-user editing.
"""
from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    env = os.getenv("WORKSPACE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # repo_root/src/utils/paths.py → repo_root
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | os.PathLike | None) -> Path | None:
    """Normalise a path to absolute. Relative paths resolve under workspace_root()."""
    if p is None:
        return None
    p = Path(p).expanduser()
    if p.is_absolute():
        return p
    return (workspace_root() / p).resolve()


def ensure_dir(p: str | os.PathLike) -> Path:
    """Resolve `p` and create the directory (parents included). Returns the absolute path."""
    resolved = resolve_path(p)
    assert resolved is not None
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
