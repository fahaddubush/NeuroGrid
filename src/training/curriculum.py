"""
6-stage gradual / curriculum forecasting pipeline.

Stages (per the user's brief):
    1. 15 min   →  1 step
    2. 30 min   →  2 steps
    3. 45 min   →  3 steps
    4.  1 h     →  4 steps
    5.  2 h     →  8 steps
    6.  3 h     → 12 steps

Each stage warm-starts from the previous stage's best checkpoint. The result
is six trained model bundles under `src/models/stored/curriculum_*` plus a
`progression_report.json` documenting trend evolution across horizons. This
gives downstream code (week / month trend estimation) a sequence of
horizon-specialised models to reason about *trend progression* without
forcing a single multi-day training burden.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from src.data.schema import CURRICULUM_STAGES
from src.training.trainer import train_stage


CURRICULUM_RUN_PREFIX = "curriculum"


def run_curriculum(
    output_root: str,
    epochs: int = 10,
    batch_size: int = 64,
    seq_len: int = 96,
    max_households: int | None = None,
    tier: str = "city",
    use_weather: bool = True,
    manifest_path: str | None = None,
    seed: int = 0,
) -> dict:
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)

    progression: list[dict] = []
    pretrained: str | None = None

    for stage in CURRICULUM_STAGES:
        stage_dir = out / f"{CURRICULUM_RUN_PREFIX}_{stage.name}"
        logging.info("=" * 70)
        logging.info(
            "Curriculum stage: %s (%d minutes, %d steps)",
            stage.name,
            stage.minutes,
            stage.steps,
        )
        logging.info("=" * 70)
        result = train_stage(
            stage=stage,
            output_dir=str(stage_dir),
            epochs=epochs,
            batch_size=batch_size,
            seq_len=seq_len,
            max_households=max_households,
            pretrained_path=pretrained,
            tier=tier,
            use_weather=use_weather,
            manifest_path=manifest_path,
            seed=seed,
        )
        pretrained = str(stage_dir / "model.pth")
        progression.append(
            {
                "stage": stage.name,
                "minutes": stage.minutes,
                "steps": stage.steps,
                "best_val_loss": result.best_val_loss,
                "epochs_run": result.epochs_run,
                "train_size": result.train_size,
                "val_size": result.val_size,
                "model_path": pretrained,
                "scaler_path": str(stage_dir / "scaler.npz"),
                "config_path": str(stage_dir / "config.json"),
            }
        )

    report = {
        "curriculum": progression,
        "tier": tier,
        "stages": [
            {"name": s.name, "minutes": s.minutes, "steps": s.steps}
            for s in CURRICULUM_STAGES
        ],
    }
    report_path = out / "progression_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logging.info("[Curriculum] Report saved to %s", report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Run the 6-stage gradual forecasting curriculum.")
    parser.add_argument("--output_root", default="src/models/stored")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--max_households", type=int, default=None)
    parser.add_argument("--tier", default="city", choices=["city", "district", "building"])
    parser.add_argument("--no_weather", action="store_true")
    parser.add_argument(
        "--manifest_path",
        default=None,
        help="Sampling manifest JSON for cluster-stratified train/val split.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_curriculum(
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_households=args.max_households,
        tier=args.tier,
        use_weather=not args.no_weather,
        manifest_path=args.manifest_path,
    )


if __name__ == "__main__":
    main()
