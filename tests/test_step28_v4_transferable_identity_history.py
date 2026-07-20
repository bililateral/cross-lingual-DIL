from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step28_common as base  # noqa: E402
import step28_generate_transferable_identity_histories as generator  # noqa: E402
import step28_history_common as history  # noqa: E402
import step28_train_transferable_identity_model as trainer  # noqa: E402


class Step28V4TransferableIdentityHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policies = [
            history.load_policy(history.POLICY_PATH),
            history.load_policy(ROOT / "schema" / "step28_transferable_identity_history_v4_1_policy.json"),
            history.load_policy(ROOT / "schema" / "step28_transferable_identity_history_v4_2_policy.json"),
            history.load_policy(ROOT / "schema" / "step28_transferable_identity_history_v4_3_policy.json"),
            history.load_policy(ROOT / "schema" / "step28_transferable_identity_history_v5_policy.json"),
            history.load_policy(ROOT / "schema" / "step28_transferable_identity_history_v6_policy.json"),
            history.load_policy(ROOT / "schema" / "step28_transferable_identity_history_v7_policy.json"),
            history.load_policy(ROOT / "schema" / "step28_transferable_identity_history_v8_policy.json"),
            history.load_policy(ROOT / "schema" / "step28_transferable_identity_history_v9_policy.json"),
            history.load_policy(ROOT / "schema" / "step28_transferable_identity_history_v10_policy.json"),
            history.load_policy(ROOT / "schema" / "step28_transferable_identity_history_v11_policy.json"),
            history.load_policy(ROOT / "schema" / "step28_transferable_identity_history_v12_policy.json"),
        ]
        cls.policy = cls.policies[-1]
        cls.v5_policy = cls.policies[4]
        cls.carrier = {
            "pair_uid": "train-only-carrier",
            "review_label": "positive",
            "identifier_redacted_e5_cosine": "0.93",
        }

    def test_every_recipe_passes_the_production_parser(self) -> None:
        rng = random.Random(11)
        for policy in self.policies:
            for split, cfg in policy["generation"]["splits"].items():
                for index, recipe in enumerate(cfg["recipes"]):
                    truth, items, signals, model_row = generator.build_world(
                        split, recipe, index, cfg["template_bank"], self.carrier,
                        policy, rng,
                    )
                    self.assertEqual(truth["parser_recovery"], 1.0, recipe)
                    self.assertTrue(items, recipe)
                    self.assertEqual(truth["review_label"], model_row["review_label"])
                    if "source_only" not in recipe:
                        self.assertTrue(signals, recipe)

    def test_model_feature_boundary_has_no_oracle_names(self) -> None:
        forbidden = self.policy["feature_boundary"]["forbidden_feature_name_fragments"]
        leaked = [
            name for name in self.policy["model"]["feature_names"]
            if any(fragment in name.lower() for fragment in forbidden)
        ]
        self.assertEqual(leaked, [])

    def test_context_roles_reach_distinct_model_states(self) -> None:
        cfg = self.policy["generation"]["splits"]["synthetic_train"]
        expected = {
            "positive_stable_reuse": "verified_direct_token_count_log1p",
            "negative_product_leakage": "risky_token_count_log1p",
            "negative_support_leakage": "support_token_count_log1p",
            "negative_public_hub": "high_frequency_token_count_log1p",
        }
        for index, (recipe, feature) in enumerate(expected.items()):
            _truth, _items, _signals, model_row = generator.build_world(
                "synthetic_train", recipe, 500 + index, cfg["template_bank"],
                self.carrier, self.policy, random.Random(200 + index),
            )
            self.assertGreater(float(model_row[feature]), 0.0, f"{recipe}:{feature}")

    def test_zero_history_keeps_frozen_source_exactly(self) -> None:
        count = len(self.policy["model"]["feature_names"])
        artifact = {
            "feature_scales": [1.0] * count,
            "coefficients": [0.25] * count,
        }
        source = np.asarray([0.17, 0.83])
        matrix = np.zeros((2, count), dtype=float)
        observed = history.predict_with_artifact(source, matrix, artifact)
        np.testing.assert_allclose(observed, source, rtol=0.0, atol=1e-15)

    def test_average_precision_groups_tied_scores(self) -> None:
        scores = np.asarray([0.9, 0.9, 0.2, 0.2])
        first = base.average_precision(np.asarray([1, 0, 1, 0]), scores)
        reordered = base.average_precision(np.asarray([0, 1, 0, 1]), scores)
        self.assertEqual(first, 0.5)
        self.assertEqual(reordered, 0.5)

    def test_weighted_metrics_are_tie_invariant_and_state_equal(self) -> None:
        labels = np.asarray([1, 0, 1, 0, 1], dtype=float)
        scores = np.asarray([0.9, 0.9, 0.3, 0.3, 0.3], dtype=float)
        weights = np.asarray([0.5, 0.5, 1 / 3, 1 / 3, 1 / 3], dtype=float)
        order = np.asarray([1, 0, 4, 2, 3])
        first_ap = base.weighted_average_precision(labels, scores, weights)
        second_ap = base.weighted_average_precision(
            labels[order], scores[order], weights[order]
        )
        first_auc = base.weighted_roc_auc(labels, scores, weights)
        second_auc = base.weighted_roc_auc(
            labels[order], scores[order], weights[order]
        )
        self.assertAlmostEqual(first_ap, second_ap, places=15)
        self.assertAlmostEqual(first_auc, second_auc, places=15)
        self.assertAlmostEqual(float(np.sum(weights)), 2.0, places=15)

    def test_v12_policy_is_fully_expanded_and_has_no_dead_feature_contracts(self) -> None:
        path = ROOT / "schema" / "step28_transferable_identity_history_v12_policy.json"
        raw = base.load_json(path)
        self.assertNotIn("_extends", raw)
        self.assertEqual(
            raw["model"]["audit_evaluation_unit"],
            "all_rows_equal_observable_state_weight",
        )
        self.assertEqual(len(raw["model"]["feature_names"]), 33)
        for removed in (
            "mixed_context_token_count_log1p",
            "same_token_path_count_log1p",
            "rotation_external_url_edge_token_count_log1p",
        ):
            self.assertNotIn(removed, raw["model"]["feature_names"])
        rules = raw["audit_gates"]["audit_recipe_rules"]
        for rule in rules.values():
            self.assertEqual(
                set(rule),
                {"layer", "interpretation", "metric", "comparison", "threshold"},
            )

    def test_v5_uses_only_frozen_english_source_carriers(self) -> None:
        policy = self.v5_policy
        self.assertEqual(
            policy["generation"]["source_carrier_domain"],
            "frozen_english_source_train_401",
        )
        self.assertIn("en_content_train_pool", policy["inputs"]["real_train_source_carriers"])
        pools = generator.carrier_pools(policy)
        unique = {
            row["pair_uid"]
            for labels in pools.values()
            for rows in labels.values()
            for row in rows
        }
        self.assertEqual(len(unique), 401)

    def test_v6_carriers_have_no_label_input_or_label_column(self) -> None:
        policy = self.policy
        self.assertNotIn("source_carrier_labels", policy["inputs"])
        pools = generator.carrier_pools(policy)
        unique = {
            row["pair_uid"]
            for split_pools in pools.values()
            for rows in split_pools.values()
            for row in rows
        }
        self.assertEqual(len(unique), 401)
        self.assertTrue(all(
            "review_label" not in row
            for split_pools in pools.values()
            for rows in split_pools.values()
            for row in rows
        ))

    def test_v6_refuses_a_source_label_file(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["inputs"]["source_carrier_labels"] = "forbidden.csv"
        with self.assertRaisesRegex(ValueError, "forbids a source-carrier label file"):
            generator.carrier_pools(policy)

    def test_v6_exactly_pairs_source_carriers_across_labels(self) -> None:
        policy = copy.deepcopy(self.policy)
        for cfg in policy["generation"]["splits"].values():
            cfg["rows_per_recipe"] = 2
            cfg["recipes"] = [
                "positive_stable_reuse",
                "negative_private_collision",
            ]
        _truth, _items, _signals, _model_rows, summary = generator.generate(policy)
        for diagnostics in summary["source_label_balance_by_split"].values():
            self.assertTrue(diagnostics["source_carrier_uid_multiset_exactly_equal"])
            self.assertTrue(diagnostics["source_probability_multiset_exactly_equal"])
            self.assertEqual(diagnostics["source_only_roc_auc"], 0.5)

    def test_current_namespace_does_not_reuse_previous_worlds_or_identifiers(self) -> None:
        previous = self.policies[-2]
        current = self.policies[-1]
        previous_cfg = previous["generation"]["splits"]["synthetic_train"]
        current_cfg = current["generation"]["splits"]["synthetic_train"]
        previous_truth, *_ = generator.build_world(
            "synthetic_train", "positive_stable_reuse", 0,
            previous_cfg["template_bank"], self.carrier, previous, random.Random(1),
        )
        current_truth, *_ = generator.build_world(
            "synthetic_train", "positive_stable_reuse", 0,
            current_cfg["template_bank"], self.carrier, current, random.Random(1),
        )
        self.assertNotEqual(previous_truth["world_uid"], current_truth["world_uid"])
        self.assertTrue(
            set(previous_truth["identifier_values"]).isdisjoint(
                current_truth["identifier_values"]
            )
        )

    def test_v6_preflight_stops_a_correlated_source_distribution(self) -> None:
        labels = np.asarray([1.0, 1.0, 0.0, 0.0])
        matrix = np.zeros((4, len(self.policy["model"]["feature_names"])))
        rows = [
            {"source_carrier_pair_uid": uid}
            for uid in ("a", "b", "a", "b")
        ]
        good_source = np.asarray([0.2, 0.8, 0.2, 0.8])
        recorded = {
            "source_carrier_uid_multiset_exactly_equal": True,
            "source_probability_multiset_exactly_equal": True,
            "source_only_roc_auc": 0.5,
        }
        summary = {
            "source_carrier_label_file_open_count": 0,
            "source_carrier_label_column_count": 0,
            "source_label_balance_by_split": {
                split: dict(recorded)
                for split in (
                    "synthetic_train", "synthetic_development", "synthetic_audit"
                )
            },
        }
        splits = {
            split: (rows, matrix, labels, good_source)
            for split in summary["source_label_balance_by_split"]
        }
        self.assertTrue(
            trainer.source_carrier_independence_preflight(
                self.policy, splits, summary
            )["passed"]
        )
        bad_splits = dict(splits)
        bad_splits["synthetic_audit"] = (
            rows, matrix, labels, np.asarray([0.8, 0.9, 0.1, 0.2])
        )
        with self.assertRaisesRegex(RuntimeError, "failed before fitting"):
            trainer.source_carrier_independence_preflight(
                self.policy, bad_splits, summary
            )

    def test_split_names_do_not_include_old_evaluation_inputs(self) -> None:
        for key, value in self.policy["inputs"].items():
            if key.endswith("_sha256"):
                continue
            lowered = str(value).lower()
            self.assertNotIn("valid", lowered)
            self.assertNotIn("internal_test", lowered)


if __name__ == "__main__":
    unittest.main()
