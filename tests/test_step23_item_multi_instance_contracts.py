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
        self.assertTrue(evaluation["primary_model_preregistered"])
        self.assertTrue(evaluation["candidate_selection_forbidden"])
        self.assertTrue(evaluation["frozen_scorer_requires_promotion"])
        self.assertEqual(evaluation["primary_model"], "aggregate_plus_distribution_primary")
        self.assertEqual(evaluation["matched_baseline_model"], "same_item_aggregate_control")
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

    def test_cross_field_redaction_disables_overlap_but_preserves_content_dedup(self):
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
        self.assertFalse(row["exact_overlap_eligible"])
        self.assertEqual(diagnostics["cross_field_redaction_item_count"], 1)
        other_meta = dict(meta)
        other_meta["item_uid"] = "item-cross-field-other"
        other_meta["source_row_number"] = 4
        other, _ = item_builder.build_item(
            other_meta,
            "联系 telegram",
            "differenthandle 完全不同的商品内容",
            "服务",
            [],
            self.policy["item_selection"],
        )
        self.assertNotEqual(row["clean_text"], other["clean_text"])
        self.assertNotEqual(row["content_signature"], other["content_signature"])

    def test_dedup_signature_uses_full_redacted_text_before_encoder_truncation(self):
        meta = {
            "item_uid": "long-a",
            "pool": "zh_target_strict",
            "domain": "zh",
            "seller_uid": "seller",
            "source_dataset": "market_item.xlsx",
            "source_row_number": 5,
        }
        prefix = "相同商品正文" * 400
        first, _ = item_builder.build_item(
            meta, "标题", prefix + "甲结尾", "服务", [], self.policy["item_selection"]
        )
        other_meta = dict(meta)
        other_meta["item_uid"] = "long-b"
        other_meta["source_row_number"] = 6
        second, _ = item_builder.build_item(
            other_meta, "标题", prefix + "乙结尾", "服务", [], self.policy["item_selection"]
        )
        self.assertEqual(first["clean_text"], second["clean_text"])
        self.assertNotEqual(first["content_signature"], second["content_signature"])

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
        self.assertIn("mi_mean_pool_cosine", forward)

    def test_matched_baseline_and_primary_use_the_same_item_feature_universe(self):
        evaluation = self.policy["evaluation"]
        feature_sets = evaluation["model_feature_sets"]
        baseline = set(feature_sets[evaluation["matched_baseline_model"]])
        primary = set(feature_sets[evaluation["primary_model"]])
        self.assertTrue(baseline)
        self.assertTrue(baseline < primary)
        self.assertIn("mi_mean_pool_cosine", baseline)
        self.assertNotIn("mi_cosine_q95", baseline)
        self.assertIn("mi_cosine_q95", primary)
        self.assertEqual(
            set(feature_sets["item_structure_only"]),
            baseline - {"mi_mean_pool_cosine"},
        )

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
        self.assertGreaterEqual(
            float(evaluation["minimum_ap_gain_over_matched_aggregate"]), 0.02
        )
        self.assertGreaterEqual(float(evaluation["minimum_bootstrap_lower_bound"]), 0.0)
        self.assertGreaterEqual(float(evaluation["minimum_non_silver_ap_delta"]), 0.0)
        self.assertGreaterEqual(float(evaluation["minimum_strong_evidence_ap_delta"]), 0.0)
        self.assertLessEqual(float(evaluation["maximum_template_or_topic_mean_score_increase"]), 0.02)
        self.assertLessEqual(float(evaluation["maximum_template_or_topic_q95_score_increase"]), 0.02)
        self.assertLessEqual(
            float(evaluation["maximum_template_or_topic_top_decile_mean_score_increase"]),
            0.02,
        )
        self.assertEqual(int(evaluation["grouped_bootstrap_resamples"]), 5000)

    def test_negative_tail_metrics_expose_high_score_tail(self):
        result = evaluator.negative_tail_metrics(np.asarray([0.1, 0.2, 0.3, 0.95]))
        self.assertGreater(result["q95"], result["mean"])
        self.assertEqual(result["top_decile_mean"], 0.95)
        self.assertEqual(result["maximum"], 0.95)

    def test_fitted_artifact_is_json_serializable_and_replayable(self):
        v7_policy = json.loads(
            (ROOT / self.policy["inputs"]["v7_policy"]).read_text(encoding="utf-8")
        )
        rows = []
        for index in range(8):
            positive = index % 2 == 0
            rows.append({
                "pair_uid": f"pair-{index}",
                "review_label": "positive" if positive else "negative",
                "domain": "zh",
                "evidence_type": (
                    "same_controller_direct_identifier" if positive else "ordinary_negative"
                ),
                "training_sample_weight": "1.0",
                "v7_component_id": f"component-{index}",
            })
        matrix = np.asarray(
            [[float(index), float(index % 3)] for index in range(len(rows))], dtype=float
        )
        scores, _weights, artifact = evaluator.fit_and_score(
            rows,
            matrix,
            matrix,
            dict(v7_policy["step9_latent_mixup"]["logistic"]),
            v7_policy["factorized_evidence_weighting"],
            ["feature_a", "feature_b"],
        )
        json.dumps(artifact)
        replay_matrix = evaluator.common.apply_imputation(matrix, artifact["imputation"])
        replay_scores = evaluator.step9.apply_logistic_artifact_to_matrix(
            replay_matrix, artifact["logistic_artifact"]
        )
        np.testing.assert_allclose(scores, replay_scores, rtol=0.0, atol=1e-12)

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
        pool_sellers = {}
        for pool_name, pool_cfg in self.policy["pools"].items():
            sellers, diagnostics = item_builder.train_scope(
                pool_name, pool_cfg, self.policy["item_selection"]
            )
            self.assertTrue(sellers)
            self.assertEqual(diagnostics["train_heldout_seller_overlap"], 0)
            pool_sellers[pool_name] = sellers
        item_builder.assert_disjoint_pool_sellers(pool_sellers)
        self.assertFalse(
            pool_sellers["en_content_train_pool"] & pool_sellers["zh_target_strict"]
        )

    def test_current_source_target_components_are_disjoint(self):
        assignments = {
            row["pair_uid"]: row
            for row in item_builder.load_csv(ROOT / self.policy["inputs"]["component_assignments"])
        }
        rows_by_pool = {}
        for pool_name, pool_cfg in self.policy["pools"].items():
            labels = item_builder.load_csv(ROOT / pool_cfg["frozen_labels"])
            rows = []
            for row in labels:
                if (
                    row.get("split_name") != "train"
                    or row.get("review_label") not in {"positive", "negative"}
                    or not item_builder.bool_value(row.get("usable_for_supervision"))
                ):
                    continue
                rows.append({
                    **row,
                    "step23_component_id": assignments[row["pair_uid"]]["recomputed_component_id"],
                })
            rows_by_pool[pool_name] = rows
        result = evaluator.assert_domain_component_isolation(
            rows_by_pool["en_content_train_pool"], rows_by_pool["zh_target_strict"]
        )
        self.assertEqual(result["source_target_component_overlap"], 0)
        self.assertEqual(result["source_target_seller_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
