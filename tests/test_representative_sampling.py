"""Pure-Python tests for src.data.representative_sampling.

The Spark / MLlib orchestration (`run_sampling`) is exercised in integration
runs; here we cover the testable surface - stratified selection logic and
manifest round-trip - without spinning a SparkSession.
"""
from __future__ import annotations

import json
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid

from src.data.representative_sampling import (
    PROFILE_FEATURE_COLUMNS,
    SamplingReport,
    SelectedHousehold,
    cluster_stratified_household_split,
    load_manifest,
    load_manifest_with_clusters,
    stratified_select,
    write_sampling_manifest,
)

_TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "test_tmp"
_TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _fresh_tmp_dir() -> Path:
    path = _TEST_TMP_ROOT / f"sampling_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class StratifiedSelectTests(unittest.TestCase):
    def test_balanced_allocation_prefers_cluster_coverage(self) -> None:
        # Cluster sizes 60/30/10 with target_n=10 should now aim for
        # balanced cohort coverage rather than raw population proportionality.
        assignments = (
            [(f"h_a_{i}", 0) for i in range(60)]
            + [(f"h_b_{i}", 1) for i in range(30)]
            + [(f"h_c_{i}", 2) for i in range(10)]
        )
        selected = stratified_select(assignments, target_n=10, seed=42)
        self.assertEqual(len(selected), 10)
        cnt = Counter(s.cluster for s in selected)
        self.assertEqual(cnt[0], 4)
        self.assertEqual(cnt[1], 3)
        self.assertEqual(cnt[2], 3)

    def test_target_caps_at_total(self) -> None:
        assignments = [(f"h{i}", i % 2) for i in range(8)]
        selected = stratified_select(assignments, target_n=100, seed=1)
        self.assertEqual(len(selected), 8)

    def test_target_zero_returns_empty(self) -> None:
        self.assertEqual(stratified_select([("h1", 0)], target_n=0), [])

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(stratified_select([], target_n=10), [])

    def test_deterministic_for_fixed_seed(self) -> None:
        assignments = [(f"h{i}", i % 3) for i in range(30)]
        a = stratified_select(assignments, target_n=9, seed=7)
        b = stratified_select(assignments, target_n=9, seed=7)
        self.assertEqual([s.household_id for s in a], [s.household_id for s in b])

    def test_different_seed_changes_membership_within_cluster(self) -> None:
        # 30 households across 3 equal clusters → 9 selected. Picking a
        # different seed should change at least one chosen id.
        assignments = [(f"h{i:03d}", i % 3) for i in range(30)]
        a = {s.household_id for s in stratified_select(assignments, target_n=9, seed=1)}
        b = {s.household_id for s in stratified_select(assignments, target_n=9, seed=999)}
        self.assertNotEqual(a, b)

    def test_quota_capped_at_cluster_size(self) -> None:
        # Tiny cluster: cluster 1 has only 1 member; even if its share
        # would be larger, we cannot select more than its members.
        assignments = [(f"h_a_{i}", 0) for i in range(99)] + [("solo", 1)]
        selected = stratified_select(assignments, target_n=50, seed=0)
        clusters = Counter(s.cluster for s in selected)
        self.assertEqual(clusters[1], 1)
        self.assertEqual(sum(clusters.values()), 50)

    def test_balanced_selection_avoids_29_1_0_style_collapse(self) -> None:
        assignments = (
            [(f"h_a_{i}", 0) for i in range(1379)]
            + [(f"h_b_{i}", 1) for i in range(27)]
            + [("h_c_0", 2)]
        )
        selected = stratified_select(assignments, target_n=30, seed=0)
        cnt = Counter(s.cluster for s in selected)
        self.assertEqual(sum(cnt.values()), 30)
        self.assertGreaterEqual(cnt[1], 10)
        self.assertEqual(cnt[2], 1)

    def test_results_sorted_by_household_id(self) -> None:
        assignments = [(f"h{i:02d}", i % 4) for i in range(40)]
        selected = stratified_select(assignments, target_n=12, seed=5)
        ids = [s.household_id for s in selected]
        self.assertEqual(ids, sorted(ids))


class ManifestRoundTripTests(unittest.TestCase):
    def _make_report(self) -> SamplingReport:
        now = datetime.now(timezone.utc)
        return SamplingReport(
            run_id="sample_test",
            started_at=now.isoformat(),
            finished_at=now.isoformat(),
            runtime_seconds=1.5,
            data_dir="/d",
            target_n=4,
            k_clusters=2,
            seed=0,
            total_households=10,
            selected_count=4,
            cluster_sizes={0: 6, 1: 4},
            cluster_selected={0: 3, 1: 1},
            feature_columns=list(PROFILE_FEATURE_COLUMNS),
            output_dir="/o",
        )

    def test_writes_manifest_summary_and_report(self) -> None:
        tmp = _fresh_tmp_dir()
        try:
            report = self._make_report()
            selected = [
                SelectedHousehold("h001", 0, 0),
                SelectedHousehold("h002", 0, 1),
                SelectedHousehold("h003", 0, 2),
                SelectedHousehold("h010", 1, 0),
            ]
            run_dir = write_sampling_manifest(selected, report, tmp)
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "summary.md").exists())
            self.assertTrue((run_dir / "report.json").exists())

            # Round-trip selected ids via load_manifest.
            ids = load_manifest(run_dir / "manifest.json")
            self.assertEqual(ids, ["h001", "h002", "h003", "h010"])

            # Manifest carries feature columns + method metadata.
            payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["selection_method"], "spark_mllib_kmeans_stratified")
            self.assertEqual(payload["feature_columns"], list(PROFILE_FEATURE_COLUMNS))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_summary_md_contains_per_cluster_rows(self) -> None:
        tmp = _fresh_tmp_dir()
        try:
            report = self._make_report()
            run_dir = write_sampling_manifest(
                [SelectedHousehold("h001", 0, 0)], report, tmp
            )
            md = (run_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Per-cluster selection", md)
            self.assertIn("`mean_load`", md)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ClusterStratifiedSplitTests(unittest.TestCase):
    """Cluster-stratified, household-disjoint train/val/test splitting."""

    def _three_cluster_population(self, n_per_cluster: int = 10):
        # Generate the default target_n=30, k=3 cohort.
        return [(f"hh_{c}_{i:02d}", c) for c in range(3) for i in range(n_per_cluster)]

    def test_default_24_3_3_split_with_k3_each_cluster_in_val_and_test(self) -> None:
        assignments = self._three_cluster_population(10)
        split = cluster_stratified_household_split(
            assignments, train_n=24, val_n=3, test_n=3, seed=0
        )
        self.assertEqual(len(split["train"]), 24)
        self.assertEqual(len(split["val"]), 3)
        self.assertEqual(len(split["test"]), 3)

        # Splits must be disjoint.
        all_ids = split["train"] + split["val"] + split["test"]
        self.assertEqual(len(all_ids), len(set(all_ids)))

        # Each of the 3 clusters represented in val and in test (project requirement).
        cluster_of = dict(assignments)
        val_clusters = {cluster_of[h] for h in split["val"]}
        test_clusters = {cluster_of[h] for h in split["test"]}
        self.assertEqual(val_clusters, {0, 1, 2})
        self.assertEqual(test_clusters, {0, 1, 2})

    def test_split_is_deterministic_for_fixed_seed(self) -> None:
        assignments = self._three_cluster_population(10)
        a = cluster_stratified_household_split(assignments, 24, 3, 3, seed=11)
        b = cluster_stratified_household_split(assignments, 24, 3, 3, seed=11)
        self.assertEqual(a, b)

    def test_split_changes_with_seed(self) -> None:
        assignments = self._three_cluster_population(10)
        a = cluster_stratified_household_split(assignments, 24, 3, 3, seed=1)
        b = cluster_stratified_household_split(assignments, 24, 3, 3, seed=42)
        # At least the val sets should differ for different seeds.
        self.assertNotEqual(a["val"], b["val"])

    def test_oversize_split_raises(self) -> None:
        assignments = self._three_cluster_population(5)  # 15 households
        with self.assertRaises(ValueError):
            cluster_stratified_household_split(assignments, 12, 3, 3, seed=0)

    def test_unbalanced_clusters_still_disjoint(self) -> None:
        # cluster 0 dominates; val/test should still get a representative
        # from clusters 1 and 2 where feasible.
        assignments = (
            [(f"hh_a_{i:02d}", 0) for i in range(20)]
            + [(f"hh_b_{i:02d}", 1) for i in range(5)]
            + [(f"hh_c_{i:02d}", 2) for i in range(5)]
        )
        split = cluster_stratified_household_split(assignments, 24, 3, 3, seed=0)
        all_ids = split["train"] + split["val"] + split["test"]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        # Val + test together should cover all clusters.
        cluster_of = dict(assignments)
        eval_clusters = {cluster_of[h] for h in split["val"] + split["test"]}
        self.assertEqual(eval_clusters, {0, 1, 2})

    def test_empty_returns_empty(self) -> None:
        out = cluster_stratified_household_split([], 0, 0, 0, seed=0)
        self.assertEqual(out, {"train": [], "val": [], "test": []})

    def test_load_manifest_with_clusters_round_trip(self) -> None:
        tmp = _fresh_tmp_dir()
        try:
            now = datetime.now(timezone.utc)
            report = SamplingReport(
                run_id="x", started_at=now.isoformat(), finished_at=now.isoformat(),
                runtime_seconds=0.1, data_dir="/d", target_n=2, k_clusters=2,
                seed=0, total_households=2, selected_count=2,
                cluster_sizes={0: 1, 1: 1}, cluster_selected={0: 1, 1: 1},
                feature_columns=list(PROFILE_FEATURE_COLUMNS), output_dir="/o",
            )
            run_dir = write_sampling_manifest(
                [
                    SelectedHousehold("h1", 0, 0),
                    SelectedHousehold("h2", 1, 0),
                ],
                report,
                tmp,
            )
            pairs = load_manifest_with_clusters(run_dir / "manifest.json")
            self.assertEqual(pairs, [("h1", 0), ("h2", 1)])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
