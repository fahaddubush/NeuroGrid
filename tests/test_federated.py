"""Tests for the paper-faithful federated primitives (Algorithms 3, 5)."""
from __future__ import annotations

import math
import unittest

import torch

from src.federated.clipping import clip_state_dict, state_dict_l2_norm
from src.federated.krum import krum_scores, krum_select, multi_krum_select
from src.federated.aggregation import weighted_fedavg, trimmed_mean
from src.federated.sparsification import topk_sparsify, sparsity
from src.federated.distillation import DistillationLoss, kl_distillation_loss
from src.federated.convergence import ConvergenceMonitor


def make_state(scale: float = 1.0) -> dict[str, torch.Tensor]:
    return {
        "fc.weight": torch.ones(4, 4) * scale,
        "fc.bias": torch.zeros(4),
    }


class TestClipping(unittest.TestCase):
    def test_clip_no_op_below_threshold(self):
        s = make_state(0.1)
        clipped, n = clip_state_dict(s, threshold=10.0)
        self.assertAlmostEqual(state_dict_l2_norm(clipped), n, places=4)

    def test_clip_scales_above_threshold(self):
        s = make_state(5.0)
        clipped, original = clip_state_dict(s, threshold=1.0)
        clipped_norm = state_dict_l2_norm(clipped)
        self.assertGreater(original, 1.0)
        self.assertAlmostEqual(clipped_norm, 1.0, places=4)


class TestKrum(unittest.TestCase):
    def test_krum_picks_central_update(self):
        # 4 honest near-zero updates, 1 outlier far away.
        updates = [make_state(0.0) for _ in range(4)] + [make_state(1000.0)]
        idx, scores = krum_select(updates, f=1)
        self.assertIn(idx, range(4))
        self.assertGreater(scores[4].item(), scores[idx].item())

    def test_multi_krum_rejects_outliers(self):
        updates = [make_state(0.0) for _ in range(5)] + [make_state(500.0)]
        accepted, rejected, _ = multi_krum_select(updates, f=1, k=5)
        self.assertEqual(len(accepted), 5)
        self.assertIn(5, rejected)

    def test_score_is_nonnegative(self):
        updates = [make_state(i * 0.1) for i in range(6)]
        scores = krum_scores(updates, f=1)
        self.assertTrue(bool((scores >= 0).all()))


class TestAggregation(unittest.TestCase):
    def test_weighted_fedavg_uniform(self):
        updates = [make_state(i) for i in range(1, 4)]  # 1, 2, 3
        avg = weighted_fedavg(updates)
        self.assertTrue(torch.allclose(avg["fc.weight"], torch.ones(4, 4) * 2.0))

    def test_weighted_fedavg_weighted(self):
        updates = [make_state(0.0), make_state(10.0)]
        avg = weighted_fedavg(updates, weights=[3.0, 1.0])
        # (0 * 3 + 10 * 1) / 4 = 2.5
        self.assertTrue(torch.allclose(avg["fc.weight"], torch.ones(4, 4) * 2.5))

    def test_trimmed_mean_drops_extremes(self):
        updates = [make_state(0.0), make_state(1.0), make_state(2.0), make_state(100.0)]
        out = trimmed_mean(updates, trim_ratio=0.25)
        # After trimming 1 from each end coordinate-wise: mean of {1, 2} = 1.5
        self.assertTrue(torch.allclose(out["fc.weight"], torch.ones(4, 4) * 1.5))


class TestSparsification(unittest.TestCase):
    def test_topk_keep_ratio(self):
        s = {"w": torch.arange(100, dtype=torch.float32)}
        out = topk_sparsify(s, keep_ratio=0.1)
        nonzero = int((out["w"] != 0).sum().item())
        self.assertEqual(nonzero, 10)
        # The 10 largest values are 90..99 - sum should be 945.
        self.assertEqual(float(out["w"].sum().item()), float(sum(range(90, 100))))

    def test_sparsity_metric(self):
        s = {"w": torch.tensor([0.0, 0.0, 1.0, 2.0])}
        self.assertAlmostEqual(sparsity(s), 0.5)


class TestDistillation(unittest.TestCase):
    def test_loss_converges_when_pred_matches_teacher(self):
        student = torch.zeros(4, 8, requires_grad=True)
        teacher = torch.zeros(4, 8)
        kl = kl_distillation_loss(student, teacher, temperature=2.0)
        self.assertAlmostEqual(float(kl.item()), 0.0, places=4)

    def test_full_loss_combines_task_and_kd(self):
        loss_fn = DistillationLoss(alpha=0.5, temperature=2.0)
        student = torch.randn(2, 4, requires_grad=True)
        teacher = student.detach() + 0.1
        target = torch.zeros(2, 4)
        out = loss_fn(student, target, teacher)
        for k in ("total", "task", "distill"):
            self.assertIn(k, out)
            self.assertTrue(torch.isfinite(out[k]))


class TestConvergence(unittest.TestCase):
    def test_drift_detection(self):
        prev = make_state(0.0)
        curr = make_state(0.0)
        m = ConvergenceMonitor(epsilon=1e-3)
        converged, drift = m.update(prev, curr)
        self.assertEqual(drift, 0.0)
        # patience=1 default → first stable round triggers convergence
        self.assertTrue(converged)

    def test_drift_above_eps_does_not_converge(self):
        m = ConvergenceMonitor(epsilon=1e-3, patience=1)
        prev = make_state(0.0)
        curr = make_state(1.0)
        converged, drift = m.update(prev, curr)
        self.assertGreater(drift, 1e-3)
        self.assertFalse(converged)


if __name__ == "__main__":
    unittest.main()
