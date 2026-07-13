"""Paper-faithful federated primitives (Algorithms 3, 5)."""

from src.federated.clipping import clip_state_dict, clip_updates, state_dict_l2_norm
from src.federated.krum import krum_scores, krum_select, multi_krum_select
from src.federated.aggregation import weighted_fedavg, trimmed_mean
from src.federated.sparsification import topk_sparsify, sparsity
from src.federated.distillation import DistillationLoss, kl_distillation_loss
from src.federated.convergence import ConvergenceMonitor
from src.federated.dp import (
    gaussian_noise_state_dict,
    clip_and_noise,
    DPAccountant,
    DPEvent,
)

__all__ = [
    "clip_state_dict",
    "clip_updates",
    "state_dict_l2_norm",
    "krum_scores",
    "krum_select",
    "multi_krum_select",
    "weighted_fedavg",
    "trimmed_mean",
    "topk_sparsify",
    "sparsity",
    "DistillationLoss",
    "kl_distillation_loss",
    "ConvergenceMonitor",
    "gaussian_noise_state_dict",
    "clip_and_noise",
    "DPAccountant",
    "DPEvent",
]
