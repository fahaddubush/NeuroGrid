"""Tests for the durable District pending store (Item 3 of the limitations list)."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from src.tiers.pending_store import PendingStore


def make_state(scale: float = 1.0) -> dict[str, torch.Tensor]:
    return {"fc.weight": torch.ones(2, 2) * scale, "fc.bias": torch.zeros(2)}


class TestPendingStore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _store(self) -> PendingStore:
        return PendingStore(self.tmp / "d1.db", district_id="D1")

    def test_initial_round_is_zero(self):
        s = self._store()
        self.assertEqual(s.current_round, 0)
        self.assertEqual(s.recover()["completed_rounds"], 0)

    def test_append_and_size(self):
        s = self._store()
        n = s.append_upload(0, "B1", n_samples=5, state_dict=make_state(1.0))
        self.assertEqual(n, 1)
        n = s.append_upload(0, "B2", n_samples=3, state_dict=make_state(2.0))
        self.assertEqual(n, 2)
        self.assertEqual(s.round_size(0), 2)

    def test_duplicate_overwrites(self):
        s = self._store()
        s.append_upload(0, "B1", n_samples=1, state_dict=make_state(0.0))
        s.append_upload(0, "B1", n_samples=10, state_dict=make_state(99.0))
        items = s.snapshot_round(0)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["n_samples"], 10)
        self.assertTrue(torch.allclose(items[0]["state_dict"]["fc.weight"], torch.ones(2, 2) * 99))

    def test_snapshot_round_trips(self):
        s = self._store()
        s.append_upload(0, "B1", 7, make_state(1.5))
        s.append_upload(0, "B2", 11, make_state(2.5))
        items = s.snapshot_round(0)
        self.assertEqual(len(items), 2)
        agents = sorted(i["agent_id"] for i in items)
        self.assertEqual(agents, ["B1", "B2"])

    def test_pop_and_advance(self):
        s = self._store()
        s.append_upload(0, "B1", 1, make_state(0.0))
        s.pop_round(0, n_uploads=1)
        s.advance_round(expected_round=0)
        self.assertEqual(s.current_round, 1)
        self.assertEqual(s.round_size(0), 0)
        self.assertEqual(s.recover()["completed_rounds"], 1)

    def test_advance_round_race(self):
        s = self._store()
        s.advance_round(0)
        with self.assertRaises(RuntimeError):
            s.advance_round(0)  # already at round 1

    def test_durable_across_reopens(self):
        """Crash-recovery analogue: write, drop, reopen, snapshot is intact."""
        s1 = self._store()
        s1.append_upload(0, "B1", 4, make_state(7.0))
        s1.append_upload(0, "B2", 6, make_state(8.0))
        del s1

        s2 = self._store()
        report = s2.recover()
        self.assertEqual(report["current_round"], 0)
        self.assertEqual(report["in_flight_rounds"], [{"round_id": 0, "n_uploads": 2}])

        items = s2.snapshot_round(0)
        self.assertEqual(len(items), 2)
        weights = sorted(float(i["state_dict"]["fc.weight"][0, 0].item()) for i in items)
        self.assertEqual(weights, [7.0, 8.0])

    def test_recover_after_completion(self):
        s = self._store()
        s.append_upload(0, "B1", 1, make_state(0.0))
        s.pop_round(0, n_uploads=1)
        s.advance_round(0)
        del s
        s2 = self._store()
        rec = s2.recover()
        self.assertEqual(rec["current_round"], 1)
        self.assertEqual(rec["completed_rounds"], 1)
        self.assertEqual(rec["in_flight_rounds"], [])


if __name__ == "__main__":
    unittest.main()
