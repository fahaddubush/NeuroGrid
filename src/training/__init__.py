"""Training and evaluation pipeline (single stage + 6-stage curriculum)."""

from src.training.trainer import train_stage, StageResult
from src.training.curriculum import run_curriculum, CURRICULUM_RUN_PREFIX
from src.training.evaluator import evaluate_run

__all__ = [
    "train_stage",
    "StageResult",
    "run_curriculum",
    "CURRICULUM_RUN_PREFIX",
    "evaluate_run",
]
