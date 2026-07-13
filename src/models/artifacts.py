"""
Artifact save/load for trained city LSTMs.

Each trained run produces a directory:

    <stored>/<run_name>/
        model.pth         - torch state dict
        scaler.npz        - fitted StandardScaler numeric state (no pickle)
        config.json       - ForecastConfig (feature schema + horizons)
        metadata.json     - tier, num_params, training summary

That bundle is everything inference / distillation / federation needs.
"""
from __future__ import annotations

import json
import os
import hashlib
import platform
import sys
import subprocess
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import torch

from src.data.feature_pipeline import (
    ForecastConfig,
    load_forecast_config,
    save_forecast_config,
    save_scaler,
    load_scaler,
)
from src.models.city_lstm import CityLSTM, tier_size

MODEL_FILENAME = "model.pth"
SCALER_FILENAME = "scaler.npz"
CONFIG_FILENAME = "config.json"
METADATA_FILENAME = "metadata.json"


def _git_sha() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save_artifact_bundle(
    run_dir: str,
    model: CityLSTM,
    scaler,
    config: ForecastConfig,
    tier: str,
    extra: dict | None = None,
) -> Path:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / MODEL_FILENAME)
    save_scaler(scaler, str(out / SCALER_FILENAME))
    save_forecast_config(config, str(out / CONFIG_FILENAME))
    metadata = {
        "tier": tier,
        "num_parameters": model.num_parameters(),
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "pred_len": model.pred_len,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "git_sha": _git_sha(),
        **(extra or {}),
    }
    metadata["checksums"] = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in (MODEL_FILENAME, SCALER_FILENAME, CONFIG_FILENAME)
    }
    with open(out / METADATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return out


def load_artifact_bundle(run_dir: str, tier: str | None = None) -> dict:
    p = Path(run_dir)
    config = load_forecast_config(str(p / CONFIG_FILENAME))
    scaler_path = p / SCALER_FILENAME
    if not scaler_path.exists():
        scaler_path = p / "scaler.pkl"
    scaler = load_scaler(str(scaler_path))
    metadata = {}
    meta_path = p / METADATA_FILENAME
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    resolved_tier = tier or metadata.get("tier", "city")
    sz = tier_size(resolved_tier)
    model = CityLSTM(pred_len=config.pred_len, input_dim=config.input_dim, **sz)
    state = torch.load(p / MODEL_FILENAME, map_location="cpu", weights_only=True)
    checksums = metadata.get("checksums", {})
    for name, expected in checksums.items():
        actual = hashlib.sha256((p / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return {
        "model": model,
        "scaler": scaler,
        "config": config,
        "metadata": metadata,
    }


__all__ = [
    "save_artifact_bundle",
    "load_artifact_bundle",
    "MODEL_FILENAME",
    "SCALER_FILENAME",
    "CONFIG_FILENAME",
    "METADATA_FILENAME",
]
