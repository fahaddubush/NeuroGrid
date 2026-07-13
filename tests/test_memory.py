"""Tests for the ISMCC Memory module (Algorithm 4)."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np

from src.ismcc.memory import (
    ShortTermMemory,
    LongTermMemory,
    EpisodicMemory,
    SharedMemory,
    attention_retrieval,
    gated_update,
)


class TestSTM(unittest.TestCase):
    def test_capacity_and_window(self):
        stm = ShortTermMemory(capacity=3)
        for i in range(5):
            stm.write({"i": i})
        w = stm.window()
        self.assertEqual(len(w), 3)
        self.assertEqual([x["i"] for x in w], [2, 3, 4])

    def test_warm(self):
        stm = ShortTermMemory(capacity=10)
        for i in range(5):
            stm.write({"i": i})
        self.assertFalse(stm.is_warm(6))
        self.assertTrue(stm.is_warm(5))


class TestLTM(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pool_grows(self):
        ltm = LongTermMemory(
            agent_id="A1", db_path=self.tmp / "ltm.db",
            feature_dim=4, pool_block_size=2, pool_capacity=10,
        )
        for i in range(6):
            ltm.persist({"timestamp": f"t{i}", "kwh": i}, feature_vec=np.ones(4) * i)
        # 6 vectors / block_size 2 = 3 pooled keys
        self.assertEqual(ltm.pool_size(), 3)
        self.assertEqual(len(ltm.recent(10)), 6)


class TestEpisodic(unittest.TestCase):
    def test_priority_sampling(self):
        ep = EpisodicMemory(agent_id="A1", capacity=100)
        for i in range(20):
            reward = -10.0 if i == 0 else 0.0  # one strongly negative episode
            ep.write({"i": i}, action="MAINTAIN", reward=reward)
        rng = np.random.default_rng(0)
        # Priority ∝ |reward|^alpha + ε  → episode 0 dominates samples.
        seen = []
        for _ in range(50):
            batch = ep.sample(1, rng=rng)
            seen.append(batch[0]["observation"]["i"])
        self.assertGreater(seen.count(0), 25)


class TestShared(unittest.TestCase):
    def test_consensus(self):
        sm = SharedMemory(dp_sigma=0.0)
        sm.write("a1", "k", np.ones(4))
        sm.write("a2", "k", np.zeros(4))
        c = sm.consensus("k")
        self.assertTrue(np.allclose(c, np.array([0.5, 0.5, 0.5, 0.5])))


class TestRetrieval(unittest.TestCase):
    def test_attention_returns_self_for_identical_key(self):
        keys = np.eye(4, dtype=np.float32)
        values = np.eye(4, dtype=np.float32) * 10
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        ctx, scores = attention_retrieval(query, keys, values)
        # Highest score should be on row 0; context should be biased toward [10, 0, 0, 0].
        self.assertEqual(int(np.argmax(scores)), 0)
        self.assertGreater(ctx[0], ctx[1])

    def test_gated_update_lerps(self):
        prior = np.zeros(4)
        new = np.ones(4) * 2.0
        out = gated_update(prior, new, lambda_=0.25)
        self.assertTrue(np.allclose(out, np.array([0.5, 0.5, 0.5, 0.5])))


if __name__ == "__main__":
    unittest.main()
