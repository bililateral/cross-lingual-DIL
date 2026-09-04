from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_v6_style_transfer_common_v2 as common
import step28_v13_v1_13_v9_4_1_v6_style_transfer_source_linux_v2 as source
import step28_v13_v1_13_v9_4_1_v6_style_transfer_target_linux_v2 as target


class StyleTransferPrimitiveTests(unittest.TestCase):
    def test_style_windows_are_lossless_and_budgeted(self):
        stream = "prefix-" + " ".join(
            f"{'W' if index % 2 else 'N'}{index % 99 + 1}!"
            for index in range(237)
        )
        windows = common.split_style_windows(stream, 100)
        self.assertEqual("".join(windows), stream)
        self.assertEqual(
            [common.count_placeholders(value) for value in windows],
            [100, 100, 37],
        )

    def test_order_format_mass_ablation_uses_numeric_sort(self):
        value = common.ablate_order_format_mass("W10, N9! W2 W1 N11")
        self.assertEqual(value, "N9 N11 W1 W2 W10")

    def test_derangement_is_input_order_invariant_bijection(self):
        values = ["account-c", "account-a", "account-b", "account-d"]
        first = common.deterministic_stream_derangement(values, 20260911)
        second = common.deterministic_stream_derangement(
            list(reversed(values)), 20260911
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(values))
        self.assertEqual(set(first.values()), set(values))
        self.assertFalse(any(key == value for key, value in first.items()))

    def test_projection_is_language_neutral_for_letter_runs(self):
        self.assertEqual(common.transferable_style_projection("苹果 ABC 123!"), "W2 W3 N3!")
        self.assertEqual(common.transferable_style_projection("apple ABC 123!"), "W5 W3 N3!")

    def test_ranking_probability_and_retrieval_metrics_are_well_formed(self):
        ranking = source.ranking_metrics(
            [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]
        )
        self.assertGreater(ranking["average_precision"], 0.5)
        self.assertNotEqual(
            ranking["average_precision"], ranking["trapezoidal_pr_auc"]
        )
        probability = source.probability_metrics(
            [0.9, 0.8, 0.7, 0.1], [1, 0, 1, 0], None, 0.5
        )
        self.assertAlmostEqual(probability["precision"], 2.0 / 3.0)
        retrieval = source.aggregate_retrieval(
            {
                "q": {"r": 0.8, "n": 0.9},
            },
            {"q": {"r"}},
        )
        self.assertEqual(retrieval["mrr"], 0.5)
        self.assertEqual(retrieval["recall_at_1"], 0.0)
        self.assertEqual(retrieval["recall_at_3"], 1.0)

    def test_component_bootstrap_preserves_zero_paired_delta(self):
        labels = [1, 0, 1, 0]
        scores = [0.9, 0.2, 0.8, 0.1]
        weights = [1.0, 1.0, 1.0, 1.0]
        left = [0, 0, 1, 1]
        right = [0, 1, 1, 0]
        draws = __import__("numpy").asarray(
            [[0, 1], [0, 0], [1, 1]], dtype="int32"
        )
        first = source.bootstrap_global_ap(
            labels, scores, weights, left, right, draws
        )
        second = source.bootstrap_global_ap(
            labels, scores, weights, left, right, draws
        )
        self.assertTrue(__import__("numpy").array_equal(first, second))

    def test_component_bootstrap_keeps_labels_aligned_after_score_sort(self):
        np = __import__("numpy")
        labels = [0, 1, 0, 1]
        scores = [0.9, 0.8, 0.7, 0.6]
        weights = [1.0, 2.0, 3.0, 4.0]
        left = [0, 0, 1, 1]
        right = [1, 0, 0, 1]
        draws = np.asarray([[0, 0], [1, 1], [0, 1]], dtype="int32")
        values = source.bootstrap_global_ap(
            labels, scores, weights, left, right, draws
        )
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))
        self.assertTrue(np.allclose(values, [0.8, 2.0 / 3.0, 28.0 / 45.0]))

    def test_component_bootstrap_two_row_counterexample(self):
        np = __import__("numpy")
        value = source.bootstrap_global_ap(
            [1, 0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0, 0],
            [0, 1],
            np.asarray([[0, 0]], dtype="int32"),
        )[0]
        self.assertAlmostEqual(value, 2.0 / 3.0)

    def test_component_bootstrap_matches_explicit_weighted_ap(self):
        np = __import__("numpy")
        from sklearn.metrics import average_precision_score

        labels = np.asarray([1, 0, 1, 0, 0, 1], dtype="int8")
        scores = np.asarray([0.8, 0.8, 0.5, 0.4, 0.4, 0.1], dtype="float64")
        base = np.asarray([0.5, 1.0, 1.5, 0.75, 0.25, 2.0])
        left = np.asarray([0, 0, 1, 1, 2, 2], dtype="int32")
        right = np.asarray([0, 1, 1, 2, 0, 2], dtype="int32")
        draws = np.asarray(
            [[0, 1, 2], [0, 0, 2], [2, 2, 1], [1, 1, 1]],
            dtype="int32",
        )
        observed = source.bootstrap_global_ap(
            labels, scores, base, left, right, draws
        )
        expected = []
        for draw in draws:
            multiplicity = np.bincount(draw, minlength=3)
            weights = base * (multiplicity[left] + multiplicity[right]) / 2.0
            expected.append(average_precision_score(labels, scores, sample_weight=weights))
        self.assertTrue(np.allclose(observed, expected, atol=1e-15, rtol=0.0))

    def test_recall_at_one_percent_keeps_empirical_tied_threshold(self):
        labels = []
        scores = []
        for score in (3.0, 2.0, 1.0):
            labels.extend([0] + [1] * 10)
            scores.extend([score] * 11)
        labels.extend([0] * 197 + [1] * 70)
        scores.extend([0.0] * 267)
        metrics = source.ranking_metrics(labels, scores)
        self.assertAlmostEqual(metrics["recall_at_fpr_1pct"], 0.2)

    def test_target_world_selection_is_order_invariant(self):
        values = ["world-c", "world-a", "world-b", "world-d"]
        self.assertEqual(
            target._selected_worlds(values, 2, 20260921),
            target._selected_worlds(list(reversed(values)), 2, 20260921),
        )
        self.assertEqual(
            target._selected_worlds(values, 2, 20260921),
            target._selected_worlds(values, 3, 20260921)[:2],
        )

    def test_target_paired_interval_is_zero_for_identical_arms(self):
        np = __import__("numpy")
        policy = common.load_policy()
        pairs = [
            {
                "world_uid": "w0",
                "label": 1,
            },
            {
                "world_uid": "w0",
                "label": 0,
            },
            {
                "world_uid": "w1",
                "label": 1,
            },
            {
                "world_uid": "w1",
                "label": 0,
            },
        ]
        values = np.asarray([0.9, 0.1, 0.8, 0.2], dtype="<f8")
        interval = target.paired_world_intervals(
            policy,
            pairs,
            [values, values, values],
            [values, values, values],
            [values, values, values],
        )
        self.assertEqual(
            interval["correct_minus_generic"], {"q025": 0.0, "q975": 0.0}
        )
        self.assertEqual(
            interval["correct_minus_permuted"], {"q025": 0.0, "q975": 0.0}
        )

    def test_target_ranking_view_keeps_raw_cosine_when_probability_ties(self):
        labels = [1, 0, 1, 0]
        cosines = [0.9, 0.8, 0.7, 0.1]
        saturated_probabilities = [1.0, 1.0, 1.0, 1.0]
        ranking, probability = target.target_metric_views(
            labels, cosines, saturated_probabilities, 0.5
        )
        self.assertEqual(
            ranking["average_precision"],
            source.ranking_metrics(labels, cosines)["average_precision"],
        )
        self.assertNotEqual(
            ranking["average_precision"],
            source.ranking_metrics(labels, saturated_probabilities)[
                "average_precision"
            ],
        )
        self.assertEqual(probability["recall"], 1.0)

    def test_source_replay_requires_exact_synthetic_audit_scores(self):
        np = __import__("numpy")
        reference = np.asarray([0.25, 0.5, 0.75], dtype="<f8")
        exact = (reference.copy(), {}, object(), ("a", "b"))
        with mock.patch.object(
            target.source, "_evaluate_v6_split", return_value=exact
        ):
            result = target.verify_source_replay_scores(
                object(), object(), object(), {}, {}, reference
            )
        self.assertTrue(result["exact_array_match"])
        self.assertEqual(result["max_abs_error"], 0.0)
        drifted = (reference + np.asarray([0.0, 0.0, 1e-12]), {}, object(), ())
        with mock.patch.object(
            target.source, "_evaluate_v6_split", return_value=drifted
        ):
            with self.assertRaises(target.TargetTransferRuntimeError):
                target.verify_source_replay_scores(
                    object(), object(), object(), {}, {}, reference
                )


class StyleTransferFrozenInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = common.load_policy()
        cls.v6 = common.load_v6(cls.policy)

    def test_policy_hash_and_truth_boundary(self):
        self.assertEqual(
            common.canonical_hash(self.policy), self.policy["canonical_self_hash"]
        )
        self.assertEqual(
            self.policy["evaluation"]["chinese_primary_aggregation"],
            "world_equal",
        )
        self.assertEqual(
            self.policy["evaluation"]["ranking_score"],
            "raw_account_cosine_float64",
        )
        self.assertEqual(
            self.policy["source_replay"]["comparison"],
            "exact_float64_raw_cosine_array",
        )
        truth = self.policy["truth_boundary"]
        self.assertFalse(truth["audit_a_labels_qrels_controllers_allowed"])
        self.assertFalse(truth["audit_b_labels_qrels_controllers_allowed"])
        self.assertEqual(
            self.policy["source_optimization"]["expected_updates_per_epoch"], 39
        )
        for budget in self.policy["target_optimization"]["budgets"].values():
            worlds = int(budget["world_count"])
            width = int(budget["worlds_per_gradient_step"])
            epochs = int(budget["epochs"])
            self.assertEqual(worlds % width, 0)
            self.assertEqual(
                int(budget["optimizer_updates"]), worlds // width * epochs
            )
        evaluation = self.policy["evaluation"]
        self.assertEqual(evaluation["primary_budget"], "worlds_025")
        self.assertEqual(evaluation["saturation_budget"], "worlds_500")
        self.assertEqual(
            evaluation["learning_curve_budgets"],
            list(self.policy["target_optimization"]["budgets"]),
        )
        for name in evaluation["learning_curve_budgets"][:-1]:
            self.assertEqual(
                self.policy["target_optimization"]["budgets"][name]
                ["optimizer_updates"],
                evaluation["low_resource_optimizer_updates_per_arm_seed"],
            )
        self.assertAlmostEqual(
            self.policy["target_optimization"]["bias_initial"],
            math.log(20.0 / 358.0),
        )

    def test_metrics_include_distinct_ap_pr_auc_and_retrieval(self):
        evaluation = self.policy["evaluation"]
        self.assertIn("average_precision", evaluation["ranking_metrics"])
        self.assertIn("trapezoidal_pr_auc", evaluation["ranking_metrics"])
        self.assertNotEqual("average_precision", "trapezoidal_pr_auc")
        for metric in (
            "mrr",
            "map",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "ndcg_at_1",
            "ndcg_at_3",
            "ndcg_at_5",
            "ndcg_at_10",
        ):
            self.assertIn(metric, evaluation["retrieval_metrics"])

    def test_v6_component_objective_is_exactly_class_balanced(self):
        accounts = {
            uid: row
            for uid, row in self.v6["accounts"].items()
            if row["split"] == "train"
        }
        pairs = [row for row in self.v6["pairs"] if row["split"] == "train"]
        account_to_component, components = common.positive_components(accounts, pairs)
        contributions = common.component_loss_contributions(
            pairs, account_to_component
        )
        mass = common.audit_component_class_mass(pairs, contributions)
        self.assertEqual(len(components), 310)
        self.assertEqual(len(mass), 310)
        for value in mass.values():
            self.assertTrue(math.isclose(value["positive"], 1.0, abs_tol=1e-12))
            self.assertTrue(math.isclose(value["negative"], 1.0, abs_tol=1e-12))

    def test_permuted_control_breaks_correspondence_without_moving_rows(self):
        accounts = {
            uid: row
            for uid, row in self.v6["accounts"].items()
            if row["split"] == "train"
        }
        pairs = [row for row in self.v6["pairs"] if row["split"] == "train"]
        component, _ = common.positive_components(accounts, pairs)
        expected = {
            20260911: (3, 0),
            20260912: (0, 3),
            20260913: (0, 0),
        }
        for seed, expected_counts in expected.items():
            mapping = common.deterministic_stream_derangement(tuple(accounts), seed)
            positive_same = 0
            negative_same = 0
            for row in pairs:
                same = (
                    component[mapping[row["account_left_uid"]]]
                    == component[mapping[row["account_right_uid"]]]
                )
                if same and row["label"] == 1:
                    positive_same += 1
                if same and row["label"] == 0:
                    negative_same += 1
            self.assertEqual((positive_same, negative_same), expected_counts)

    def test_static_train_development_inputs(self):
        result = common.validate_static_inputs()
        self.assertEqual(
            result["status"], "PASSED_V6_STYLE_TRANSFER_STATIC_INPUTS"
        )
        self.assertEqual(result["v5_accounts"], 82)
        self.assertEqual(result["chinese"]["train"], {"worlds": 500, "sellers": 14000})
        self.assertEqual(
            result["chinese"]["development"], {"worlds": 500, "sellers": 14000}
        )
        self.assertEqual(result["audit_a_truth_reads"], 0)
        self.assertEqual(result["audit_b_truth_reads"], 0)

    def test_audit_splits_are_rejected_by_chinese_loaders(self):
        with self.assertRaises(common.StyleTransferContractError):
            common.load_chinese_pairs(
                self.policy, "audit_a", include_labels=False
            )


if __name__ == "__main__":
    unittest.main()
