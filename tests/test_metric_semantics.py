from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step7_train_baseline_models as step7  # noqa: E402


class MetricSemanticsV2Tests(unittest.TestCase):
    def test_average_precision_is_invariant_to_order_within_ties(self) -> None:
        labels_a = np.asarray([1, 0, 1, 0, 1, 0], dtype=float)
        labels_b = np.asarray([0, 1, 0, 1, 1, 0], dtype=float)
        scores = np.asarray([0.9, 0.9, 0.5, 0.5, 0.1, 0.1], dtype=float)
        self.assertAlmostEqual(
            step7.average_precision_score(labels_a, scores),
            step7.average_precision_score(labels_b, scores),
            places=12,
        )

    def test_pr_auc_is_not_an_alias_for_average_precision(self) -> None:
        labels = np.asarray([1, 0, 1, 0], dtype=float)
        scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=float)
        ap = step7.average_precision_score(labels, scores)
        pr_auc = step7.precision_recall_auc_score(labels, scores)
        self.assertIsNotNone(ap)
        self.assertIsNotNone(pr_auc)
        self.assertNotAlmostEqual(float(ap), float(pr_auc), places=10)

    def test_pr_auc_matches_known_trapezoidal_curve(self) -> None:
        labels = np.asarray([1, 0, 1, 0], dtype=float)
        scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=float)
        self.assertAlmostEqual(
            float(step7.precision_recall_auc_score(labels, scores)),
            19.0 / 24.0,
            places=12,
        )

    def test_average_precision_matches_standard_no_tie_definition(self) -> None:
        labels = np.asarray([1, 0, 1, 0], dtype=float)
        scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=float)
        self.assertAlmostEqual(
            float(step7.average_precision_score(labels, scores)),
            (1.0 + 2.0 / 3.0) / 2.0,
            places=12,
        )

    def test_global_pair_evaluation_does_not_claim_map_or_mrr(self) -> None:
        metrics = step7.evaluate_probabilities(
            np.asarray([1, 0, 1, 0], dtype=float),
            np.asarray([0.9, 0.8, 0.7, 0.1], dtype=float),
            0.5,
        )
        self.assertEqual(metrics["metric_semantics_version"], "2026-07-v2-tie-aware")
        self.assertEqual(
            metrics["pr_auc_definition"],
            "trapezoidal_area_under_tie_grouped_precision_recall_curve",
        )
        self.assertIsNone(metrics["map"])
        self.assertIsNone(metrics["mrr"])
        self.assertEqual(metrics["map_mrr_status"], "not_applicable_without_preregistered_query_groups")

    def test_frozen_v5r_seed_mean_ap_is_preserved(self) -> None:
        paths = sorted(
            (ROOT / "reports").glob(
                "step15_v5r_identity_only_curriculum_domain_balanced_public_noise_weighted_strong_"
                "weighted_mixup_phase4_add_positive_pair_mixup_seed_*_predictions.zh_test.csv"
            )
        )
        self.assertEqual(len(paths), 3)
        score_maps = []
        labels = {}
        for path in paths:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            score_maps.append({row["pair_uid"]: float(row["prob_positive"]) for row in rows})
            labels.update({row["pair_uid"]: float(row["y_true"]) for row in rows})
        order = sorted(labels)
        y_true = np.asarray([labels[pair_uid] for pair_uid in order])
        scores = np.asarray(
            [[score_map[pair_uid] for pair_uid in order] for score_map in score_maps]
        ).mean(axis=0)
        self.assertAlmostEqual(float(step7.average_precision_score(y_true, scores)), 0.7439464343, places=9)


if __name__ == "__main__":
    unittest.main()
