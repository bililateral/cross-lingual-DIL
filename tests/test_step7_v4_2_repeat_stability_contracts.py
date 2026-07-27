from __future__ import annotations

import copy
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step7_v4_2_repeat_stability_audit as repeat


class Step7V42RepeatStabilityContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy, cls.base_policy = repeat.load_policy(
            require_frozen=False
        )

    def test_repeat_seeds_are_new_and_frozen(self) -> None:
        seeds = self.policy["repeat_design"]["outer_seeds"]
        self.assertEqual(
            seeds,
            [
                2026072701,
                2026072702,
                2026072703,
                2026072704,
                2026072705,
            ],
        )
        self.assertFalse(
            set(seeds) & set(self.base_policy["training"]["outer_seeds"])
        )

    def test_runtime_policy_changes_only_frozen_repeat_seeds(self) -> None:
        observed = repeat.runtime_policy(
            self.policy, self.base_policy
        )
        expected = copy.deepcopy(observed)
        expected["training"]["outer_seeds"] = list(
            self.base_policy["training"]["outer_seeds"]
        )
        expected["evaluation"]["bootstrap"]["seed"] = int(
            self.base_policy["evaluation"]["bootstrap"]["seed"]
        )
        self.assertEqual(expected, self.base_policy)

    def test_no_valid_or_test_label_split_is_loaded(self) -> None:
        source = inspect.getsource(repeat.run_repeat)
        self.assertIn(
            'load_label_split(\n        parent_policy, pair_rows, "train"',
            source,
        )
        self.assertNotIn(
            'load_label_split(\n        parent_policy, pair_rows, "valid"',
            source,
        )
        self.assertNotIn(
            'load_label_split(\n        parent_policy, pair_rows, "test"',
            source,
        )

    def test_retrieval_metrics_include_map_and_effective_queries(self) -> None:
        rows = [
            {
                "review_label": "positive",
                "seller_uid_left": "q",
                "seller_uid_right": "p1",
            },
            {
                "review_label": "negative",
                "seller_uid_left": "q",
                "seller_uid_right": "n1",
            },
            {
                "review_label": "positive",
                "seller_uid_left": "q",
                "seller_uid_right": "p2",
            },
        ]
        metrics = repeat.strict_retrieval_metrics(
            rows, np.asarray([0.9, 0.8, 0.7])
        )
        self.assertEqual(metrics["eligible_query_count"], 1)
        self.assertAlmostEqual(metrics["mrr"], 1.0)
        self.assertAlmostEqual(
            metrics["map"], (1.0 + 2.0 / 3.0) / 2.0
        )
        self.assertAlmostEqual(metrics["precision_at_1"], 1.0)
        self.assertAlmostEqual(metrics["recall_at_3"], 1.0)

    def test_parent_and_formal_result_pins_match(self) -> None:
        for role, record in self.policy["frozen_parent"].items():
            with self.subTest(role=role):
                path = repeat.verify_file_record(record, role)
                self.assertTrue(path.is_file())

    def test_output_root_is_new_and_isolated(self) -> None:
        root = self.policy["outputs"]["root"]
        self.assertEqual(
            root,
            "reports/step7_v4_2_repeat_stability/v1_20260727",
        )
        self.assertNotIn("v1_20260724", root)

    def test_claim_boundary_is_diagnostic_only(self) -> None:
        boundary = self.policy["claim_boundary"]
        self.assertFalse(boundary["new_real_english_data"])
        self.assertFalse(boundary["old_valid_label_values_may_be_read"])
        self.assertFalse(
            boundary["historical_test_label_values_may_be_read"]
        )
        self.assertIn(
            "cannot independently confirm",
            boundary["maximum_claim"],
        )


if __name__ == "__main__":
    unittest.main()
