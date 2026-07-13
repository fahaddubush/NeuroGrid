"""Tests for the cross-process SharedMemory backends (Item 4)."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ismcc.memory.shared import (
    FileBackend,
    InMemoryBackend,
    SharedMemory,
    SparkBroadcastBackend,
)


class TestInMemoryBackend(unittest.TestCase):
    def test_put_get_collect(self):
        b = InMemoryBackend()
        b.put("a1", "k", np.array([1.0, 2.0]))
        b.put("a2", "k", np.array([3.0, 4.0]))
        self.assertTrue(np.allclose(b.get("a1", "k"), [1, 2]))
        self.assertEqual(b.n_agents(), 2)
        coll = b.collect("k")
        self.assertEqual(set(coll.keys()), {"a1", "a2"})

    def test_get_missing_returns_none(self):
        self.assertIsNone(InMemoryBackend().get("missing", "k"))


class TestFileBackend(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip(self):
        b = FileBackend(self.tmp)
        b.put("agent_x", "weights", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        out = b.get("agent_x", "weights")
        self.assertTrue(np.allclose(out, [1, 2, 3]))

    def test_durable_across_instances(self):
        b1 = FileBackend(self.tmp)
        b1.put("a1", "k", np.ones(3))
        b1.put("a2", "k", np.ones(3) * 2)
        del b1
        b2 = FileBackend(self.tmp)
        self.assertEqual(b2.n_agents(), 2)
        coll = b2.collect("k")
        self.assertTrue(np.allclose(coll["a1"], [1, 1, 1]))
        self.assertTrue(np.allclose(coll["a2"], [2, 2, 2]))

    def test_atomic_write(self):
        b = FileBackend(self.tmp)
        b.put("a1", "k", np.array([1.0, 2.0]))
        b.put("a1", "k", np.array([99.0, 100.0]))  # overwrites atomically
        out = b.get("a1", "k")
        self.assertTrue(np.allclose(out, [99, 100]))

    def test_missing_returns_none(self):
        b = FileBackend(self.tmp)
        self.assertIsNone(b.get("nope", "k"))


class TestSparkBroadcastBackend(unittest.TestCase):
    def test_falls_back_to_mirror_without_spark(self):
        b = SparkBroadcastBackend(spark=None)
        b.put("a1", "k", np.array([1.0, 2.0]))
        out = b.get("a1", "k")
        self.assertTrue(np.allclose(out, [1, 2]))


class TestSharedMemoryFrontend(unittest.TestCase):
    def test_dp_noise_perturbs(self):
        np.random.seed(0)
        sm_quiet = SharedMemory(dp_sigma=0.0, backend=InMemoryBackend())
        sm_noisy = SharedMemory(dp_sigma=1.0, backend=InMemoryBackend())
        sm_quiet.write("a1", "k", np.zeros(8))
        sm_noisy.write("a1", "k", np.zeros(8))
        v_quiet = sm_quiet.read("a1", "k")
        v_noisy = sm_noisy.read("a1", "k")
        self.assertTrue(np.allclose(v_quiet, 0.0))
        self.assertGreater(np.abs(v_noisy).sum(), 0.0)

    def test_consensus_is_mean(self):
        sm = SharedMemory(backend=InMemoryBackend())
        sm.write("a1", "k", np.array([1.0, 1.0]))
        sm.write("a2", "k", np.array([3.0, 3.0]))
        c = sm.consensus("k")
        self.assertTrue(np.allclose(c, [2, 2]))

    def test_from_env_picks_memory_by_default(self):
        os.environ.pop("NEUROGRID_SHARED_BACKEND", None)
        sm = SharedMemory.from_env()
        self.assertIsInstance(sm.backend, InMemoryBackend)

    def test_from_env_picks_file_when_requested(self):
        tmp = tempfile.mkdtemp()
        try:
            os.environ["NEUROGRID_SHARED_BACKEND"] = "file"
            os.environ["NEUROGRID_SHARED_DIR"] = tmp
            sm = SharedMemory.from_env()
            self.assertIsInstance(sm.backend, FileBackend)
        finally:
            os.environ.pop("NEUROGRID_SHARED_BACKEND", None)
            os.environ.pop("NEUROGRID_SHARED_DIR", None)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
