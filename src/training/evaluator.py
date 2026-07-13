"""
Cleaned up evaluator for trained city LSTM bundles.
Evaluates using a fixed Z-Score threshold (1.5) for true peak event classification.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import NeuroGridDataset
from src.models.artifacts import load_artifact_bundle


def _inverse_kwh(arr_2d: np.ndarray, scaler, input_dim: int) -> np.ndarray:
    flat = arr_2d.reshape(-1)
    dummy = np.zeros((len(flat), input_dim), dtype=np.float32)
    dummy[:, 0] = flat
    return scaler.inverse_transform(dummy)[:, 0].reshape(arr_2d.shape)


def _err_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    p = pred.reshape(-1).astype(np.float64)
    t = target.reshape(-1).astype(np.float64)
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _peak_event_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    flat_target = target.reshape(-1).astype(np.float64)
    flat_pred = pred.reshape(-1).astype(np.float64)
    if len(flat_target) == 0:
        return {}
        
    threshold = 1.5 # 1.5 standard deviations (Z-score) is our absolute peak definition
    y_true = flat_target >= threshold
    y_pred = flat_pred >= threshold

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum((~y_true) & (~y_pred)))
    fp = int(np.sum((~y_true) & y_pred))
    fn = int(np.sum(y_true & (~y_pred)))
    
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "threshold_zscore": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def evaluate_run(
    run_dir: str,
    batch_size: int = 64,
    max_households: int | None = None,
    n_mc_samples: int = 10,
    manifest_path: str | None = None,
    split: str = "test",
) -> dict:
    bundle = load_artifact_bundle(run_dir)
    model = bundle["model"]
    scaler = bundle["scaler"]
    config = bundle["config"]

    val_ds = NeuroGridDataset(
        config=config,
        split=split,
        scaler=scaler,
        max_households=max_households,
        manifest_path=manifest_path,
    )
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds_chunks, target_chunks = [], []

    with torch.no_grad():
        for x, y in loader:
            mean_pred, _ = model.mc_forward(x.to(device), n_samples=n_mc_samples)
            pred_kwh = _inverse_kwh(mean_pred.cpu().numpy(), scaler, config.input_dim)
            y_kwh = _inverse_kwh(y.cpu().numpy(), scaler, config.input_dim)
            preds_chunks.append(pred_kwh)
            target_chunks.append(y_kwh)

    pred = np.concatenate(preds_chunks, axis=0)
    target = np.concatenate(target_chunks, axis=0)

    aggregate = _err_metrics(pred, target)
    peak_classification = _peak_event_metrics(pred, target)

    report = {
        "run_dir": run_dir,
        "aggregate": aggregate,
        "peak_event_classification": peak_classification,
    }
    
    out_path = Path(run_dir) / "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logging.info(f"[Eval] Wrote {out_path}")
    return report
