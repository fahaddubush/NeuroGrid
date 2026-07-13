"""
Cleaned up trainer for the City LSTM.
Executes curriculum stages with a fixed Z-Score peak threshold (1.5) for robust forecasting.
"""
from __future__ import annotations

import json
import logging
import os
import random
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.dataset import NeuroGridDataset
from src.data.feature_pipeline import ForecastConfig
from src.data.schema import DAILY_HORIZON, HorizonStage
from src.models.artifacts import save_artifact_bundle
from src.models.city_lstm import CityLSTM, tier_size


@dataclass
class StageResult:
    stage_name: str
    best_val_loss: float
    epochs_run: int
    output_dir: str
    pred_len: int
    train_size: int
    val_size: int


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    lr: float
    is_best: bool
    best_val_loss_so_far: float


class PeakWeightedSmoothL1Loss(nn.Module):
    """Applies a higher penalty above a training-only target threshold."""
    def __init__(self, peak_weight: float = 3.0, peak_threshold: float = 1.5):
        super().__init__()
        self.peak_weight = float(peak_weight)
        self.peak_threshold = float(peak_threshold)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.smooth_l1_loss(pred, target, reduction="none")
        weights = torch.where(
            target >= self.peak_threshold,
            torch.full_like(target, self.peak_weight),
            torch.ones_like(target),
        )
        return (loss * weights).sum() / torch.clamp(weights.sum(), min=1.0)


class BlendedPeakSmoothL1Loss(nn.Module):
    """Blends standard SmoothL1 and the PeakWeightedSmoothL1."""
    def __init__(self, peak_weight: float = 3.0, peak_lambda: float = 0.25, peak_threshold: float = 1.5):
        super().__init__()
        if not 0.0 <= peak_lambda <= 1.0:
            raise ValueError("peak_lambda must be in [0, 1].")
        self.base = nn.SmoothL1Loss()
        self.peak = PeakWeightedSmoothL1Loss(
            peak_weight=peak_weight, peak_threshold=peak_threshold
        )
        self.peak_lambda = float(peak_lambda)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            (1.0 - self.peak_lambda) * self.base(pred, target)
            + self.peak_lambda * self.peak(pred, target)
        )


def _build_loss(
    loss_name: str,
    peak_weight: float = 3.0,
    peak_lambda: float = 0.25,
    peak_threshold: float = 1.5,
) -> nn.Module:
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    if loss_name == "peak_weighted_smooth_l1":
        return PeakWeightedSmoothL1Loss(
            peak_weight=peak_weight, peak_threshold=peak_threshold
        )
    if loss_name == "blended_peak_smooth_l1":
        return BlendedPeakSmoothL1Loss(
            peak_weight=peak_weight,
            peak_lambda=peak_lambda,
            peak_threshold=peak_threshold,
        )
    raise ValueError(f"Unsupported loss '{loss_name}'")


def _build_loaders(
    config: ForecastConfig,
    batch_size: int,
    max_households: Optional[int],
    manifest_path: Optional[str] = None,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, object, NeuroGridDataset]:
    train_ds = NeuroGridDataset(config=config, split="train", max_households=max_households, manifest_path=manifest_path)
    if train_ds.scaler is None:
        raise RuntimeError("Training scaler did not fit.")
    val_ds = NeuroGridDataset(config=config, split="val", scaler=train_ds.scaler, max_households=max_households, manifest_path=manifest_path)
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(int(seed)),
        ),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        train_ds.scaler,
        train_ds,
    )


def train_stage(
    stage: HorizonStage,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 64,
    seq_len: int = 96,
    max_households: Optional[int] = None,
    patience: int = 3,
    pretrained_path: Optional[str] = None,
    tier: str = "city",
    lr: float = 5e-4,
    dropout: Optional[float] = None,
    use_weather: bool = True,
    manifest_path: Optional[str] = None,
    loss: str = "smooth_l1",
    peak_weight: float = 3.0,
    peak_quantile: float = 0.95,
    peak_lambda: float = 0.25,
    seed: int = 0,
) -> StageResult:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    config = ForecastConfig.for_daily_spark(seq_len=int(seq_len), pred_len=int(stage.steps), use_weather=use_weather) if stage.name == DAILY_HORIZON.name else ForecastConfig(seq_len=int(seq_len), pred_len=int(stage.steps), use_weather=use_weather)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[Train] stage={stage.name} pred_len={config.pred_len} seq_len={config.seq_len} device={device}")

    train_loader, val_loader, scaler, train_dataset = _build_loaders(
        config, batch_size, max_households, manifest_path=manifest_path, seed=seed
    )
    sz = tier_size(tier)
    if dropout is not None: sz["dropout"] = dropout
    model = CityLSTM(pred_len=config.pred_len, input_dim=config.input_dim, **sz).to(device)

    if pretrained_path and os.path.exists(pretrained_path):
        prev = torch.load(pretrained_path, map_location=device, weights_only=True)
        own = model.state_dict()
        adopted = 0
        for key, value in prev.items():
            if key in own and own[key].shape == value.shape:
                own[key] = value
                adopted += 1
        model.load_state_dict(own)
        logging.info(f"[Train] Curriculum warm-start: adopted {adopted}/{len(own)} tensors from {pretrained_path}")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    peak_threshold = train_dataset.target_quantile(peak_quantile)
    criterion = _build_loss(
        loss,
        peak_weight=peak_weight,
        peak_lambda=peak_lambda,
        peak_threshold=peak_threshold,
    )

    best_val = float("inf")
    no_improve, epochs_run = 0, 0
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history: list[EpochRecord] = []
    
    for epoch in range(epochs):
        epochs_run += 1
        model.train()
        train_sum, n_batches = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x.to(device))
            loss_value = criterion(pred, y.to(device))
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_sum += float(loss_value.item())
            n_batches += 1
            
        train_loss = train_sum / max(n_batches, 1)

        model.eval()
        val_sum, n_val_batches = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                val_sum += float(criterion(model(x.to(device)), y.to(device)).item())
                n_val_batches += 1
                
        val_loss = val_sum / max(n_val_batches, 1)
        scheduler.step(val_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])
        improved = val_loss < best_val

        if improved:
            best_val = val_loss
            no_improve = 0
            manifest_sha256 = None
            if manifest_path and Path(manifest_path).exists():
                manifest_sha256 = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
            save_artifact_bundle(
                str(out),
                model,
                scaler,
                config,
                tier,
                {
                    "stage": stage.name,
                    "best_val_loss": best_val,
                    "seed": int(seed),
                    "manifest_path": manifest_path,
                    "manifest_sha256": manifest_sha256,
                    "peak_quantile": float(peak_quantile),
                    "scaled_peak_threshold": float(peak_threshold),
                },
            )
        else:
            no_improve += 1

        history.append(EpochRecord(epoch + 1, train_loss, val_loss, lr_now, bool(improved), best_val))
        print(f"[Train {stage.name}] epoch {epoch+1}/{epochs} Tloss={train_loss:.6f} Vloss={val_loss:.6f} Best={best_val:.6f} lr={lr_now:.2e}", flush=True)

        if not improved and no_improve >= patience:
            print(f"[Train {stage.name}] Early stopping at epoch {epoch + 1}.", flush=True)
            break

    summary_data = {
        "stage": stage.name,
        "best_val_loss": best_val,
        "epochs_run": epochs_run,
        "history": [asdict(record) for record in history]
    }
    with open(out / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return StageResult(stage.name, best_val, epochs_run, str(out), config.pred_len, len(train_loader.dataset), len(val_loader.dataset))
