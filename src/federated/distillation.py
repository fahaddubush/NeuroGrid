"""
Algorithm 5 - City Tier: Knowledge Distillation.

Paper:
    L_total = (1 - alpha) * L_task + alpha * L_distill(W_city, W_building)

with L_distill = KL divergence between the City teacher's predictions and the
Building student's predictions. Forecasting outputs are continuous, not class
probabilities, so we cast the KD signal as a softened distribution over the
prediction horizon: each step's predicted load is converted to a softmax over
horizon steps with temperature T. This preserves the *shape* of the temporal
forecast that the City teacher learned and lets the Building student inherit
it without forcing point-wise weight blending.

This is "true KD" (loss on soft targets), not the previous weight-blending
shortcut.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_distribution(prediction: torch.Tensor, temperature: float) -> torch.Tensor:
    """Convert (B, H) horizon predictions into a softmax distribution over H."""
    if prediction.dim() != 2:
        raise ValueError("prediction must be a (batch, horizon) tensor.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    return F.log_softmax(prediction / temperature, dim=-1)


def kl_distillation_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL(teacher || student) on horizon-softmax distributions, scaled by T^2.

    The T^2 scale follows Hinton et al. (2015) - it keeps the magnitude of the
    KD gradient comparable to L_task as T increases.
    """
    student_log_q = soft_distribution(student_pred, temperature)
    teacher_log_q = soft_distribution(teacher_pred.detach(), temperature)
    teacher_q = teacher_log_q.exp()
    kl = F.kl_div(student_log_q, teacher_q, reduction="batchmean")
    return kl * (temperature ** 2)


class DistillationLoss(nn.Module):
    """L_total = (1 - alpha) * L_task + alpha * L_KL(student, teacher)."""

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.task_loss = nn.SmoothL1Loss()

    def forward(
        self,
        student_pred: torch.Tensor,
        target: torch.Tensor,
        teacher_pred: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_task = self.task_loss(student_pred, target)
        l_distill = kl_distillation_loss(student_pred, teacher_pred, self.temperature)
        l_total = (1.0 - self.alpha) * l_task + self.alpha * l_distill
        return {"total": l_total, "task": l_task, "distill": l_distill}
