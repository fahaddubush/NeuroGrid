"""Tests for the city LSTM + artifact bundle."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from src.data.feature_pipeline import ForecastConfig
from src.data.schema import INPUT_DIM, CURRICULUM_STAGES
from src.models.artifacts import load_artifact_bundle, save_artifact_bundle
from src.models.city_lstm import CityLSTM, tier_size


class TestCityLSTM(unittest.TestCase):
    def test_forward_shape(self):
        m = CityLSTM(pred_len=4, input_dim=INPUT_DIM, hidden_dim=32, num_layers=1)
        x = torch.randn(2, 24, INPUT_DIM)
        y = m(x)
        self.assertEqual(tuple(y.shape), (2, 4))

    def test_tier_sizes_distinct(self):
        sizes = {t: tier_size(t)["hidden_dim"] for t in ("city", "district", "building")}
        for t, dim in sizes.items():
            self.assertGreater(dim, 0)

    def test_for_tier_helper(self):
        for t in ("city", "district", "building"):
            m = CityLSTM.for_tier(t, pred_len=4)
            self.assertEqual(m.pred_len, 4)
            self.assertGreater(m.num_parameters(), 0)

    def test_building_and_city_federated_schemas_match(self):
        building = CityLSTM.for_tier("building", pred_len=4).state_dict()
        city = CityLSTM.for_tier("city", pred_len=4).state_dict()
        self.assertEqual(set(building), set(city))
        for key in city:
            self.assertEqual(building[key].shape, city[key].shape)

    def test_mc_forward_shapes(self):
        m = CityLSTM(pred_len=8, input_dim=INPUT_DIM, hidden_dim=16, num_layers=1, dropout=0.5)
        x = torch.randn(3, 12, INPUT_DIM)
        mean, std = m.mc_forward(x, n_samples=4)
        self.assertEqual(tuple(mean.shape), (3, 8))
        self.assertEqual(tuple(std.shape), (3, 8))

    def test_curriculum_stages_constants(self):
        names = [s.name for s in CURRICULUM_STAGES]
        self.assertEqual(
            names, ["h15min", "h30min", "h45min", "h1h", "h2h", "h3h"]
        )
        self.assertEqual(CURRICULUM_STAGES[0].steps, 1)
        self.assertEqual(CURRICULUM_STAGES[-1].steps, 12)


class TestArtifactRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_load_round_trip(self):
        cfg = ForecastConfig(seq_len=24, pred_len=4)
        model = CityLSTM.for_tier("building", pred_len=cfg.pred_len)
        # Use a tiny pickleable scaler stand-in.
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        scaler = StandardScaler()
        scaler.fit(np.random.rand(8, INPUT_DIM))
        save_artifact_bundle(
            run_dir=str(self.tmp / "run"),
            model=model,
            scaler=scaler,
            config=cfg,
            tier="building",
            extra={"test": True},
        )
        bundle = load_artifact_bundle(str(self.tmp / "run"), tier="building")
        self.assertEqual(bundle["config"].pred_len, cfg.pred_len)
        self.assertEqual(bundle["metadata"]["tier"], "building")
        self.assertTrue(bundle["metadata"].get("test"))


if __name__ == "__main__":
    unittest.main()
