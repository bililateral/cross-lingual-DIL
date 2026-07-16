from __future__ import annotations

import json
import csv
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step22_build_same_seller_split_augmentation as step22  # noqa: E402
import step22_evaluate_same_seller_split as step22_eval  # noqa: E402
import step22_grouped_bootstrap_audit as step22_bootstrap  # noqa: E402


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def item(index: int, category: str = "cat", title: str | None = None) -> dict:
    return {
        "item_uid": f"item-{index}",
        "seller_uid": "seller-a",
        "source_row_number": index,
        "source_market_raw": "market",
        "source_seller_raw": "seller-a",
        "title": title or f"title-{index}",
        "description": f"description-{index}",
        "price": str(index),
        "category": category,
        "ship_from": "",
        "content_signature": f"signature-{index}",
    }


class Step22SameSellerSplitContractTests(unittest.TestCase):
    def test_grouped_bootstrap_metric_helpers(self):
        self.assertAlmostEqual(
            step22_bootstrap.average_precision([1, 0, 1], [0.9, 0.8, 0.7]),
            (1.0 + (2.0 / 3.0)) / 2.0,
        )
        self.assertEqual(step22_bootstrap.percentile([0.0, 1.0], 0.5), 0.5)
        with self.assertRaises(ValueError):
            step22_bootstrap.average_precision([0, 0], [0.2, 0.1])

    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(
            (ROOT / "schema" / "step22_same_seller_split_policy.json").read_text(
                encoding="utf-8"
            )
        )

    def test_policy_is_train_only_and_requires_matched_duplication(self) -> None:
        constraints = " ".join(self.policy["scientific_constraints"])
        self.assertIn("training-only", constraints)
        self.assertIn("not a newly observed cross-account sockpuppet relation", constraints)
        self.assertTrue(self.policy["evaluation"]["forbid_valid_or_test_selection"])
        self.assertTrue(
            self.policy["weighting"]["equal_effective_weight_duplication_controls_required"]
        )
        self.assertFalse(self.policy["generation"]["fabricate_identifiers"])
        self.assertFalse(self.policy["generation"]["fabricate_market_provenance"])
        self.assertEqual(
            self.policy["outputs_root"],
            "reports/step22_same_seller_split/v1_20260716",
        )

    def test_positive_partition_is_item_and_exact_content_disjoint(self) -> None:
        rows = [item(index, "cat-a" if index % 2 else "cat-b") for index in range(8)]
        output = step22.partition_positive_items(
            rows, "seller-a", 20260716, self.policy["eligibility"]
        )
        self.assertIsNotNone(output)
        left, right = output or ([], [])
        self.assertGreaterEqual(len(left), 3)
        self.assertGreaterEqual(len(right), 3)
        self.assertFalse({row["item_uid"] for row in left} & {row["item_uid"] for row in right})
        self.assertFalse({row["title"] for row in left} & {row["title"] for row in right})
        self.assertFalse(
            {row["description"] for row in left} & {row["description"] for row in right}
        )

    def test_duplicate_content_group_never_crosses_positive_views(self) -> None:
        rows = [item(index) for index in range(8)]
        duplicate = dict(rows[0])
        duplicate["item_uid"] = "duplicate-item"
        duplicate["description"] = "a different description with the same exact title"
        duplicate["content_signature"] = "different-combined-signature"
        rows.append(duplicate)
        output = step22.partition_positive_items(
            rows, "seller-a", 20260716, self.policy["eligibility"]
        )
        self.assertIsNotNone(output)
        left, right = output or ([], [])
        left_signatures = {row["content_signature"] for row in left}
        right_signatures = {row["content_signature"] for row in right}
        self.assertFalse(left_signatures & right_signatures)
        self.assertFalse({row["title"] for row in left} & {row["title"] for row in right})

    def test_pseudo_profile_contains_no_real_seller_or_contact_fields(self) -> None:
        profile = step22.build_profile(
            [item(index) for index in range(3)],
            "synthetic://step22/v1/positive/p00000/left",
            "SYNTHETIC_TRAIN_ONLY",
            12,
        )
        self.assertEqual(profile["source_seller_raw"], "")
        self.assertEqual(profile["alias_normalized"], "")
        self.assertEqual(profile["contact_token_count_total"], 0)
        self.assertEqual(profile["contact_concat_top"], "")
        self.assertNotIn("seller-a", profile["profile_text"])
        self.assertNotIn("[CATEGORIES]", profile["profile_text"])
        self.assertNotIn("SYNTHETIC_TRAIN_ONLY", profile["profile_text"])
        self.assertTrue(profile["synthetic_train_only"])
        self.assertFalse(profile["benchmark_eligible"])

    def test_heldout_entities_exclude_uid_and_alias_and_retain_one_train_component(self) -> None:
        assignments = [
            {
                "dataset": "zh_target_strict",
                "split_name": "valid",
                "seller_uid_left": "held-a",
                "seller_uid_right": "held-b",
                "recomputed_component_id": "valid-component",
            },
            {
                "dataset": "zh_target_strict",
                "split_name": "train",
                "seller_uid_left": "train-a",
                "seller_uid_right": "train-b",
                "recomputed_component_id": "train-component",
            },
        ]
        exclusions = [
            {
                "dataset": "zh_target_strict",
                "entity_type": "seller_alias",
                "entity_id": "HeldAlias",
                "source_splits": "test",
            }
        ]
        sellers, aliases, components = step22.heldout_entities(
            assignments, exclusions, {"valid", "test"}
        )
        self.assertEqual(sellers, {"held-a", "held-b"})
        self.assertEqual(aliases, {"heldalias"})
        self.assertEqual(components["train-a"], "train-component")
        self.assertEqual(components["train-b"], "train-component")

    def test_synthetic_and_duplication_budgets_are_exactly_matched(self) -> None:
        synthetic = step22_eval.scaled_weights(np.ones(7), 3.5)
        duplicate_x, duplicate_y, duplicate_w = step22_eval.duplicate_class(
            np.arange(20, dtype=float).reshape(10, 2),
            np.asarray([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float),
            np.asarray([1, 1, 2, 1, 3, 1, 4, 1, 5, 1], dtype=float),
            1.0,
            3.5,
        )
        self.assertEqual(len(duplicate_x), 5)
        self.assertTrue(np.all(duplicate_y == 1.0))
        self.assertAlmostEqual(float(np.sum(synthetic)), 3.5, places=12)
        self.assertAlmostEqual(float(np.sum(duplicate_w)), 3.5, places=12)
        self.assertGreater(duplicate_w[-1], duplicate_w[0])

    def test_actual_chinese_train_components_produce_five_two_class_folds(self) -> None:
        labels = [
            row
            for row in load_csv(ROOT / self.policy["inputs"]["frozen_labels"])
            if row["split_name"] == "train"
            and row["review_label"] in {"positive", "negative"}
        ]
        assignments = {
            row["pair_uid"]: row
            for row in load_csv(ROOT / self.policy["inputs"]["component_assignments"])
            if row["dataset"] == "zh_target_strict"
        }
        rows = [
            {
                "pair_uid": row["pair_uid"],
                "review_label": row["review_label"],
                "v7_component_id": assignments[row["pair_uid"]]["recomputed_component_id"],
            }
            for row in labels
        ]
        folds = step22_eval.grouped_folds(rows, 5, 20260716)
        counts = {fold: {"positive": 0, "negative": 0} for fold in range(5)}
        for row in rows:
            counts[folds[row["v7_component_id"]]][row["review_label"]] += 1
        self.assertEqual(len(rows), 573)
        self.assertTrue(
            all(values["positive"] > 0 and values["negative"] > 0 for values in counts.values())
        )


if __name__ == "__main__":
    unittest.main()
