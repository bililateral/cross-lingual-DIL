"""Hand-checkable data, metric and fail-before-truth contracts; no project labels."""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn import metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_continual_data as data
import step28_continual_evaluate as evaluate
import step28_continual_inputs as inputs
import step28_continual_run as runner


def csv_fixture(root: Path) -> tuple[Path, Path]:
    rows, labels = [], []
    for world in range(2):
        for left, right in combinations([f"w{world}a{i}" for i in range(4)], 2):
            pair = left + "||" + right
            rows.append(("train", world, f"world{world}", pair, left, right))
            positive = (left[-1], right[-1]) in (("0", "1"), ("2", "3"))
            labels.append((pair, f"world{world}", int(positive)))
    for name, fields, values in (("rows.csv", data.ROW_FIELDS, rows), ("labels.csv", data.LABEL_FIELDS, labels)):
        with (root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(values)
    return root / "rows.csv", root / "labels.csv"


class ContinualRunContracts(unittest.TestCase):
    def test_stage_loader_releases_mapping_and_exposes_current_or_arrived_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.arange(30 * 57).reshape(30, 57).astype(float)
            y = (np.arange(30) % 2).astype(np.uint8)
            for name, values in (("base.npy", x[:, :24]), ("identity.npy", x[:, 24:]), ("labels.npy", y)):
                np.save(root / name, values, allow_pickle=False)
            source = data.StageSource(root / "base.npy", root / "identity.npy", root / "labels.npy",
                                      [[3], [0], [4], [1], [2]], 6, 5)
            current, labels = source.load(2)
            np.testing.assert_array_equal(current, x[:6])
            np.testing.assert_array_equal(labels, y[:6])
            prefix, labels = source.load(2, cumulative=True)
            np.testing.assert_array_equal(prefix, np.concatenate((x[18:24], x[:6])))
            self.assertFalse(any(isinstance(v, np.ndarray) for v in vars(source).values()))
            current[0, 0] = -99
            self.assertEqual(source.load(2)[0][0, 0], x[0, 0])
            for stage in (0, 6):
                with self.assertRaises(ValueError):
                    source.load(stage)

    def test_arrival_and_world_selection_reject_duplicates_future_and_wrong_order(self):
        arrivals = inputs.random_arrivals(15, 5)
        data.validate_arrival(arrivals, 15, 5, [11, 23, 37])
        for mode in ("duplicate", "swap"):
            bad = copy.deepcopy(arrivals)
            stage = bad["orders"][0]["train_world_ordinals_by_stage"][0]
            if mode == "duplicate":
                stage[0] = stage[1]
            else:
                stage[0], stage[1] = stage[1], stage[0]
            with self.assertRaises(ValueError):
                data.validate_arrival(bad, 15, 5, [11, 23, 37])
        for worlds in ([], [1, 1], [-1], [5], [True]):
            with self.assertRaises(ValueError):
                data.selected_rows(worlds, 5, 6)

    def test_labels_are_aligned_before_conversion_and_bad_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row_path, label_path = csv_fixture(root)
            y = data.aligned_labels(label_path, row_path, "train", 2, 4, 2)
            np.testing.assert_array_equal(y, [1, 0, 0, 0, 0, 1] * 2)
            original = label_path.read_text()
            for bad in (original.replace("world0", "wrong", 1), original.replace(",1\n", ",2\n", 1),
                        "\n".join(original.splitlines()[:-1]) + "\n", original + original.splitlines()[-1] + "\n"):
                label_path.write_text(bad)
                with self.assertRaises(ValueError):
                    data.aligned_labels(label_path, row_path, "train", 2, 4, 2)
            label_path.write_text(original)
            with self.assertRaises(ValueError):
                data.aligned_labels(label_path, row_path, "development", 2, 4, 2)

    def test_pin_rejects_changed_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "fixture.json"
            path.write_text("{}")
            spec = data.record(path, root)
            data.verify(spec, root)
            path.write_text("[]")
            with self.assertRaises(ValueError):
                data.verify(spec, root)

    def test_relative_cli_artifact_path_matches_absolute_record(self):
        relative = Path(os.path.relpath(data.POLICY, Path.cwd()))
        self.assertEqual(data.record(relative), data.record(data.POLICY.resolve()))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "artifact.json"
            path.write_text("{}")
            relative_path = Path(os.path.relpath(path, Path.cwd()))
            self.assertEqual(data.record(relative_path, root), data.record(path, root.resolve()))

    def test_stage_budgets_count_tails_and_additional_replay(self):
        self.assertEqual(runner.expected_log(17, 2, 8, 4), {
            "updates": 6, "current_presentations": 34, "replay_presentations": 24})
        current = runner.expected_log(37800, 5, 256, 2284)
        self.assertEqual(current, {"updates": 740, "current_presentations": 189000, "replay_presentations": 189440})
        cumulative = [runner.expected_log(37800 * stage, 5, 256)["updates"] for stage in range(1, 6)]
        self.assertEqual(cumulative, [740, 1480, 2215, 2955, 3695])
        self.assertEqual(3 * (740 + 2 * 4 * 740 + sum(cumulative[1:])), 51015)

    def test_tied_AP_PR_and_ROC_have_distinct_hand_values(self):
        y, scores = np.array([1, 0, 1, 0]), np.array([.9, .5, .5, .1])
        r = evaluate.curve_metrics(y, scores)
        self.assertAlmostEqual(r["average_precision"], 5 / 6)
        self.assertAlmostEqual(r["trapezoidal_pr_auc"], 11 / 12)
        self.assertAlmostEqual(r["roc_auc"], .875)
        self.assertEqual(r["recall_at_fpr_1pct"], .5)
        constant = evaluate.curve_metrics(y, np.zeros(4))
        self.assertEqual(constant["average_precision"], .5)
        self.assertEqual(constant["trapezoidal_pr_auc"], .75)
        self.assertEqual(constant["roc_auc"], .5)
        self.assertEqual(constant["recall_at_fpr_1pct"], 0.)

    def test_metric_equations_match_independent_sklearn_with_ties(self):
        rng = np.random.default_rng(12)
        for _ in range(8):
            y = np.r_[np.zeros(19, dtype=int), np.ones(5, dtype=int)]
            s = rng.integers(-3, 4, len(y)).astype(float)
            result = evaluate.classification(y, s)
            precision, recall, _ = metrics.precision_recall_curve(y, s)
            self.assertAlmostEqual(result["average_precision"], metrics.average_precision_score(y, s))
            self.assertAlmostEqual(result["trapezoidal_pr_auc"], metrics.auc(recall, precision))
            self.assertAlmostEqual(result["roc_auc"], metrics.roc_auc_score(y, s))
            self.assertAlmostEqual(result["mcc"], metrics.matthews_corrcoef(y, s >= 0))
            self.assertAlmostEqual(result["f1"], metrics.f1_score(y, s >= 0))
            self.assertAlmostEqual(result["balanced_accuracy"], metrics.balanced_accuracy_score(y, s >= 0))
            p = 1 / (1 + np.exp(-s))
            self.assertAlmostEqual(result["brier"], metrics.brier_score_loss(y, p))
            self.assertAlmostEqual(result["log_loss"], metrics.log_loss(y, p))

    def test_extreme_logits_and_curve_errors_do_not_hide_invalid_values(self):
        result = evaluate.classification(np.array([0, 1]), np.array([-1000., 1000.]))
        self.assertEqual(result["brier"], 0.)
        self.assertTrue(math.isfinite(result["log_loss"]))
        self.assertEqual(result["mcc"], 1.)
        for y, s in (([0, 0], [0, 1]), ([0, 1], [np.nan, 1]), ([0, 2], [0, 1]), ([0, 1], [0])):
            with self.assertRaises(ValueError):
                evaluate.curve_metrics(np.array(y), np.array(s))

    def test_retrieval_uses_symmetric_edges_and_public_tie_order(self):
        y = np.array([[1, 0, 0, 0, 0, 1]], dtype=np.uint8)
        got = dict(zip(evaluate.RETRIEVAL_KEYS, evaluate.retrieval(y, np.zeros_like(y, dtype=float), 4)[0]))
        self.assertAlmostEqual(got["map"], 2 / 3)
        self.assertAlmostEqual(got["mrr"], 2 / 3)
        self.assertEqual(got["recall_at_1"], .5)
        self.assertEqual(got["recall_at_3"], 1.)
        self.assertEqual(got["ndcg_at_1"], .5)
        self.assertEqual(got["ndcg_at_3"], .75)
        with self.assertRaises(ValueError):
            evaluate.retrieval(np.zeros_like(y), np.zeros_like(y, dtype=float), 4)

    def test_complete_metric_columns_and_rule_subset_remain_separate(self):
        y = np.array([[1, 0, 0, 0, 0, 1]] * 2, dtype=np.uint8)
        mask = np.array([[True, True, True, False, False, True]] * 2)
        report, values = evaluate.point_metrics(y, np.zeros_like(y, dtype=float), mask, 4)
        self.assertEqual(values.shape, (2, 34))
        self.assertAlmostEqual(report["world_equal"]["average_precision"], 1 / 3)
        self.assertEqual(report["world_equal"]["no_strong_average_precision"], .5)
        self.assertEqual(report["pooled"]["confusion"], {"tp": 4, "fp": 8, "fn": 0, "tn": 0})
        self.assertAlmostEqual(report["all_world_equal_confusion"]["metrics"]["f1"], .5)
        with self.assertRaises(ValueError):
            evaluate.point_metrics(y, np.zeros_like(y, dtype=float), mask.astype(np.uint8), 4)

    def test_bootstrap_pairs_worlds_across_fixed_order_mean(self):
        r = evaluate.paired_interval(np.array([[1., 2.], [3., 4.]]), np.array([[0, 0], [1, 1]]))
        self.assertEqual(r["mean"], 2.5)
        self.assertEqual(r["per_order"], [1.5, 3.5])
        np.testing.assert_allclose(r["conditional_95pct_interval"], [2.025, 2.975])
        with self.assertRaises(ValueError):
            evaluate.paired_interval(np.ones((3, 4)), np.array([[0, 1]]))
        with self.assertRaises(ValueError):
            evaluate.paired_interval(np.ones((3, 4)), np.array([[0., 1., 2., 3.]]))

    def test_partial_run_fails_before_development_labels_are_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"status": "PARTIAL"}))
            with mock.patch.object(data, "load_policy", return_value={}), mock.patch.object(
                    data, "verify_public"), mock.patch.object(data, "aligned_labels") as labels:
                with self.assertRaises(ValueError):
                    evaluate.evaluate(root, root / "evaluation")
                labels.assert_not_called()
                self.assertFalse((root / "evaluation").exists())

    def test_policy_refuses_audit_path_before_opening_any_supervision(self):
        policy = data.load_policy()
        bad = copy.deepcopy(policy)
        bad["train_labels"]["path"] = bad["train_labels"]["path"].replace("/train/", "/audit_a/")
        with mock.patch.object(data, "read_json", return_value=bad):
            with self.assertRaises(ValueError):
                data.load_policy()


if __name__ == "__main__":
    unittest.main()
