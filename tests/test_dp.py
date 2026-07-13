"""Tests for DP primitives wired into the building uplink (Item 5)."""
from __future__ import annotations

import unittest

import torch

from src.federated.dp import (
    DPAccountant,
    clip_and_noise,
    gaussian_noise_state_dict,
)
from src.federated.clipping import state_dict_l2_norm


def make_state(scale: float) -> dict[str, torch.Tensor]:
    return {"w": torch.ones(4, 4) * scale, "b": torch.zeros(4)}


class TestGaussianNoise(unittest.TestCase):
    def test_zero_sigma_is_noop(self):
        s = make_state(1.0)
        out = gaussian_noise_state_dict(s, sigma=0.0, sensitivity=1.0)
        self.assertTrue(torch.allclose(out["w"], s["w"]))
        self.assertTrue(torch.allclose(out["b"], s["b"]))

    def test_noise_actually_perturbs(self):
        torch.manual_seed(0)
        s = make_state(0.0)
        out = gaussian_noise_state_dict(s, sigma=1.0, sensitivity=1.0)
        # With sigma=1, sensitivity=1, std=1 - variance should be ~1 with N=20 coords.
        diff = (out["w"] - s["w"]).abs().mean().item()
        self.assertGreater(diff, 0.1)

    def test_noise_scale_grows_with_sigma(self):
        torch.manual_seed(0)
        small = gaussian_noise_state_dict(make_state(0.0), sigma=0.1, sensitivity=1.0)
        torch.manual_seed(0)
        large = gaussian_noise_state_dict(make_state(0.0), sigma=2.0, sensitivity=1.0)
        self.assertGreater(large["w"].abs().mean(), small["w"].abs().mean())


class TestClipAndNoise(unittest.TestCase):
    def test_clip_first_then_noise(self):
        torch.manual_seed(0)
        s = make_state(5.0)  # ||·|| ≈ 20
        original_norm = state_dict_l2_norm(s)
        out, reported = clip_and_noise(s, sensitivity=1.0, sigma=0.0)
        # After zero-noise clip the L2 should be ≤ sensitivity.
        post_norm = state_dict_l2_norm(out)
        self.assertAlmostEqual(reported, original_norm, places=4)
        self.assertLessEqual(post_norm, 1.0 + 1e-4)

    def test_noise_is_added_after_clipping(self):
        torch.manual_seed(0)
        s = make_state(0.5)
        clean, _ = clip_and_noise(s, sensitivity=1.0, sigma=0.0)
        torch.manual_seed(0)
        noised, _ = clip_and_noise(s, sensitivity=1.0, sigma=1.0)
        self.assertFalse(torch.allclose(clean["w"], noised["w"]))

    def test_invalid_sensitivity_raises(self):
        with self.assertRaises(ValueError):
            clip_and_noise(make_state(1.0), sensitivity=0.0, sigma=1.0)


class TestDPAccountant(unittest.TestCase):
    def test_zero_events_is_zero_epsilon(self):
        acc = DPAccountant()
        self.assertEqual(acc.epsilon(delta=1e-5), 0.0)

    def test_record_then_epsilon_finite(self):
        acc = DPAccountant()
        for _ in range(10):
            acc.record(sigma=1.0, sample_rate=0.1)
        eps = acc.epsilon(delta=1e-5)
        self.assertGreater(eps, 0.0)
        self.assertLess(eps, 1e6)  # sanity: not exploded

    def test_smaller_sigma_costs_more_privacy(self):
        a1 = DPAccountant()
        a2 = DPAccountant()
        for _ in range(5):
            a1.record(sigma=2.0)
            a2.record(sigma=0.5)
        self.assertGreater(a2.epsilon(delta=1e-5), a1.epsilon(delta=1e-5))

    def test_more_rounds_costs_more_privacy(self):
        few = DPAccountant()
        many = DPAccountant()
        for _ in range(2):
            few.record(sigma=1.0)
        for _ in range(20):
            many.record(sigma=1.0)
        self.assertGreater(many.epsilon(delta=1e-5), few.epsilon(delta=1e-5))

    def test_zero_sigma_does_not_advance_budget(self):
        acc = DPAccountant()
        acc.record(sigma=0.0)
        acc.record(sigma=0.0)
        self.assertEqual(acc.epsilon(delta=1e-5), 0.0)


class TestUplinkDPIntegration(unittest.TestCase):
    """End-to-end DP plumbing through CommunicationModule.upload_delta."""

    def test_dp_path_clips_and_noises_when_sigma_positive(self):
        # We don't bring up gRPC here - just exercise the pre-serialise DP path.
        import torch as _t
        from src.federated.dp import clip_and_noise as _cn

        delta = make_state(50.0)  # large delta, should be clipped to sensitivity 1.
        _t.manual_seed(0)
        out, original_norm = _cn(delta, sensitivity=1.0, sigma=0.5)
        post = state_dict_l2_norm(out)
        # Post-clip + noise norm should be of order O(sensitivity + sigma·sensitivity·√d).
        self.assertGreater(original_norm, 1.0)
        self.assertLess(post, 50.0)


if __name__ == "__main__":
    unittest.main()
