from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step23_build_item_text_cache as item_builder  # noqa: E402
import step23_build_multi_instance_features as feature_builder  # noqa: E402
import step23_evaluate_item_multi_instance as evaluator  # noqa: E402


POLICY_PATH = ROOT / "schema" / "step23_item_multi_instance_policy.json"


def fake_item(index: int, seller: str, category: str, signature: str | None = None) -> dict:
    return {
        "item_uid": f"{seller}-item-{index}",
        "pool": "zh_target_strict",
        "domain": "zh",
        "seller_uid": seller,
        "source_dataset": "market_item.xlsx",
        "source_row_number": index,
        "category": category,
        "category_key": category,
        "title_hash": f"title-{index}",
        "description_hash": f"description-{index}",
        "content_signature": signature or f"signature-{index}",
        "clean_text": f"content {index}",
        "length_log": float(index + 1),
        "digit_ratio": 0.01 * index,
        "punct_ratio": 0.02 * index,
        "cjk_ratio": 0.5,
        "identifier_redacted": True,
        "synthetic": False,
        "split_name": "train",
    }


class Step23ItemMultiInstanceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    def test_policy_is_train_only_and_has_no_synthetic_labels(self):
        selection = self.policy["item_selection"]
        evaluation = self.policy["evaluation"]
        self.assertEqual(selection["allowed_split"], "train")
        self.assertFalse(selection["selection_uses_labels"])
        self.assertFalse(selection["selection_uses_cross_seller_statistics"])
        self.assertTrue(selection["identifier_redaction_required"])
        self.assertTrue(evaluation["valid_or_test_selection_forbidden"])
        self.assertEqual(evaluation["publication_holdout"], "Step20 genuinely prospective real data only")
        self.assertEqual(
            evaluation["source_only_controls"],
            [f"source_only_{name}" for name in evaluation["models"]],
        )
        self.assertIn("Step23 creates no synthetic labels, sellers, items or identity relations.", self.policy["scientific_constraints"])

    def test_declared_step23_inputs_exist_in_current_workspace(self):
        for value in self.policy["inputs"].values():
            self.assertTrue((ROOT / value).is_file(), value)
        for pool_cfg in self.policy["pools"].values():
            for key in ("frozen_labels", "evidence_labels", "seller_profiles", "item_identity_signals"):
                self.assertTrue((ROOT / pool_cfg[key]).is_file(), pool_cfg[key])

    def test_category_round_robin_deduplicates_and_caps(self):
        items = [
            fake_item(1, "seller", "a", "duplicate"),
            fake_item(2, "seller", "a", "duplicate"),
            fake_item(3, "seller", "a"),
            fake_item(4, "seller", "b"),
            fake_item(5, "seller", "b"),
        ]
        selected = item_builder.category_round_robin(items, 3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({row["content_signature"] for row in selected}), 3)
        self.assertEqual({row["category_key"] for row in selected[:2]}, {"a", "b"})

    def test_identifier_redaction_removes_literal_without_marker(self):
        meta = {
            "item_uid": "item",
            "pool": "zh_target_strict",
            "domain": "zh",
            "seller_uid": "seller",
            "source_dataset": "market_item.xlsx",
            "source_row_number": 2,
        }
        cfg = self.policy["item_selection"]
        row, diagnostics = item_builder.build_item(
            meta,
            "普通标题",
            "联系 uniquehandle 获取服务",
            "服务",
            ["uniquehandle"],
            cfg,
        )
        self.assertNotIn("uniquehandle", row["clean_text"].casefold())
        self.assertNotIn("identifier", row["clean_text"].casefold())
        self.assertGreaterEqual(diagnostics["signal_literal_match_count"], 1)

    def test_cross_field_identifier_redaction_disables_exact_overlap_hashes(self):
        meta = {
            "item_uid": "item-cross-field",
            "pool": "zh_target_strict",
            "domain": "zh",
            "seller_uid": "seller",
            "source_dataset": "market_item.xlsx",
            "source_row_number": 3,
        }
        row, diagnostics = item_builder.build_item(
            meta,
            "联系 telegram",
            "crossfieldhandle",
            "服务",
            [],
            self.policy["item_selection"],
        )
        self.assertNotIn("crossfieldhandle", row["clean_text"].casefold())
        self.assertEqual(row["title_hash"], "")
        self.assertEqual(row["description_hash"], "")
        self.assertEqual(diagnostics["cross_field_redaction_item_count"], 1)

    def test_pair_features_are_endpoint_symmetric(self):
        left = [fake_item(1, "left", "a"), fake_item(2, "left", "b")]
        right = [fake_item(3, "right", "a"), fake_item(4, "right", "c")]
        all_items = left + right
        embedding_index = {row["item_uid"]: index for index, row in enumerate(all_items)}
        embeddings = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 0.6, 0.8],
        ])
        cfg = self.policy["multi_instance_features"]
        forward = feature_builder.pair_features(left, right, embedding_index, embeddings, cfg)
        reverse = feature_builder.pair_features(right, left, embedding_index, embeddings, cfg)
        self.assertEqual(set(forward), set(reverse))
        for key in forward:
            self.assertAlmostEqual(forward[key], reverse[key], places=12, msg=key)
            self.assertNotIn("contact", key)
            self.assertNotIn("identifier", key)

    def test_grouped_folds_keep_components_intact_and_binary(self):
        rows = []
        for component in range(10):
            for label in ("positive", "negative"):
                rows.append({
                    "step23_component_id": f"component-{component}",
                    "review_label": label,
                })
        assignment = evaluator.grouped_folds(rows, 5, 20260716)
        self.assertEqual(len(assignment), 10)
        for fold in range(5):
            held = [row for row in rows if assignment[row["step23_component_id"]] == fold]
            self.assertEqual({row["review_label"] for row in held}, {"positive", "negative"})

    def test_promotion_gates_are_nontrivial(self):
        evaluation = self.policy["evaluation"]
        self.assertGreaterEqual(float(evaluation["minimum_ap_gain_over_aggregate"]), 0.02)
        self.assertLessEqual(float(evaluation["minimum_bootstrap_lower_bound"]), 0.0)
        self.assertGreaterEqual(float(evaluation["minimum_non_silver_ap_delta"]), -0.02)
        self.assertLessEqual(float(evaluation["maximum_template_or_topic_mean_score_increase"]), 0.02)
        self.assertEqual(int(evaluation["grouped_bootstrap_resamples"]), 5000)

    def test_current_chinese_train_components_produce_five_binary_folds(self):
        labels = item_builder.load_csv(
            ROOT / self.policy["pools"]["zh_target_strict"]["frozen_labels"]
        )
        assignments = {
            row["pair_uid"]: row
            for row in item_builder.load_csv(ROOT / self.policy["inputs"]["component_assignments"])
        }
        rows = []
        for row in labels:
            if (
                row.get("split_name") != "train"
                or row.get("review_label") not in {"positive", "negative"}
                or not item_builder.bool_value(row.get("usable_for_supervision"))
            ):
                continue
            assignment = assignments[row["pair_uid"]]
            rows.append({
                "step23_component_id": assignment["recomputed_component_id"],
                "review_label": row["review_label"],
            })
        self.assertEqual(len(rows), 573)
        self.assertEqual(len({row["step23_component_id"] for row in rows}), 222)
        fold_assignment = evaluator.grouped_folds(rows, 5, 20260716)
        for fold in range(5):
            held = [row for row in rows if fold_assignment[row["step23_component_id"]] == fold]
            self.assertGreater(sum(row["review_label"] == "positive" for row in held), 0)
            self.assertGreater(sum(row["review_label"] == "negative" for row in held), 0)

    def test_current_pool_train_sellers_are_disjoint_from_heldout(self):
        for pool_name, pool_cfg in self.policy["pools"].items():
            sellers, diagnostics = item_builder.train_scope(
                pool_name, pool_cfg, self.policy["item_selection"]
            )
            self.assertTrue(sellers)
            self.assertEqual(diagnostics["train_heldout_seller_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
