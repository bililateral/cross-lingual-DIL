from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step25_build_sync_manifest as sync_manifest  # noqa: E402
import step25_common as common  # noqa: E402
import step25_evaluate_template_decontaminated_authorship as evaluation  # noqa: E402


class Step25TemplateDecontaminatedAuthorshipContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy_path, cls.policy, cls.step24_policy = common.load_policy()

    def test_step24_is_frozen_and_output_root_is_isolated(self) -> None:
        self.assertEqual(
            self.policy["inputs"]["step24_outputs_root"],
            "reports/step24_content_independent_authorship/v1_20260717",
        )
        self.assertEqual(
            self.policy["outputs_root"],
            "reports/step25_template_decontaminated_authorship/v1_20260717",
        )
        self.assertNotEqual(
            self.policy["outputs_root"], self.policy["inputs"]["step24_outputs_root"]
        )

    def test_windows_config_validation_does_not_require_model_directories(self) -> None:
        for cfg in self.policy["frozen_style_encoders"].values():
            self.assertTrue(cfg["local_path"].startswith("models/step24/authorship/"))
            self.assertTrue(cfg["local_finetuning_forbidden"])
        common.validate_policy(self.policy, self.step24_policy)

    def test_d0_d1_f1_isolation_is_explicit(self) -> None:
        boundaries = self.policy["development_boundaries"]
        self.assertFalse(
            boundaries["d0_current_canonical_train"]["publication_promotion_allowed"]
        )
        self.assertTrue(
            boundaries["d1_future_independent_development"][
                "seller_component_overlap_with_d0_forbidden"
            ]
        )
        self.assertTrue(
            boundaries["f1_future_prospective_holdout"][
                "collection_after_model_freeze_required"
            ]
        )
        self.assertTrue(
            boundaries["f1_future_prospective_holdout"][
                "seller_component_overlap_with_d0_or_d1_forbidden"
            ]
        )

    def test_template_detector_masks_only_external_component_support(self) -> None:
        cfg = dict(self.policy["template_decontamination"])
        shared = "这是一个跨卖家重复出现的广告模板固定片段用于测试去污染机制"
        sellers = ["seller-a", "seller-b", "seller-c", "seller-d"]
        components = {seller: f"component-{index}" for index, seller in enumerate(sellers)}
        texts = [f"独立开头{index}{shared}独立结尾{index}" for index in range(len(sellers))]
        records, catalog, summary = common.decontaminate_corpus(
            sellers, texts, components, cfg
        )
        self.assertTrue(catalog)
        self.assertTrue(all(record["masked_character_count"] >= 24 for record in records))
        self.assertTrue(summary["component_cross_fitted"])
        self.assertFalse(summary["label_evidence_type_or_model_score_read"])
        self.assertNotIn(shared, records[0]["decontaminated_text"])

    def test_same_component_repetition_does_not_self_justify_masking(self) -> None:
        cfg = dict(self.policy["template_decontamination"])
        shared = "仅在同一个卖家组件内部重复的身份相关固定文本不应被自己支持删除"
        sellers = ["seller-a", "seller-b", "seller-c", "seller-d"]
        components = {
            "seller-a": "component-one",
            "seller-b": "component-one",
            "seller-c": "component-two",
            "seller-d": "component-three",
        }
        texts = [shared + "甲", shared + "乙", "完全不同的商品说明丙", "另一个不同描述丁"]
        records, _catalog, _summary = common.decontaminate_corpus(
            sellers, texts, components, cfg
        )
        by_seller = {record["seller_uid"]: record for record in records}
        self.assertEqual(by_seller["seller-a"]["masked_character_count"], 0)
        self.assertEqual(by_seller["seller-b"]["masked_character_count"], 0)

    def test_catalog_persists_hashes_not_raw_ngrams(self) -> None:
        cfg = dict(self.policy["template_decontamination"])
        shared = "这是足够长的重复模板文本片段用于验证目录中绝不保存原始内容"
        sellers = ["a", "b", "c", "d"]
        components = {seller: seller for seller in sellers}
        _records, catalog, _summary = common.decontaminate_corpus(
            sellers, [shared] * len(sellers), components, cfg
        )
        self.assertTrue(catalog)
        self.assertEqual(
            set(catalog[0]),
            {
                "shingle_sha256",
                "seller_document_frequency",
                "component_document_frequency",
                "character_length",
            },
        )
        serialized = json.dumps(catalog, ensure_ascii=False)
        self.assertNotIn(shared[:12], serialized)

    def test_zero_template_catalog_is_a_valid_header_only_artifact(self) -> None:
        fields = [
            "shingle_sha256",
            "seller_document_frequency",
            "component_document_frequency",
            "character_length",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            common.write_csv_immutable_allow_empty(path, [], fields)
            self.assertEqual(common.step24.load_csv(path), [])
            self.assertEqual(
                path.read_text(encoding="utf-8-sig").strip(),
                ",".join(fields),
            )

    def test_unreliable_decontaminated_pair_receives_no_style_support(self) -> None:
        row = {
            "pair_uid": "pair",
            "seller_uid_left": "left",
            "seller_uid_right": "right",
        }
        index = {"left": 0, "right": 1}
        matrix = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=float)
        score = common.pair_cosine(
            row,
            index,
            matrix,
            {"left": True, "right": False},
            insufficient_value=0.0,
        )
        self.assertEqual(score, 0.0)

    def test_direction_constrained_reliability_cannot_reverse_actions(self) -> None:
        clean = np.asarray([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        evidence = [
            {"evidence_state": "verified_direct_both_sides"},
            {"evidence_state": "risky_only_shared"},
            {"evidence_state": "verified_direct_both_sides"},
            {"evidence_state": "risky_only_shared"},
            {"evidence_state": "direct_with_mixed_context"},
            {"evidence_state": "no_shared_identifier"},
        ]
        corrections = np.asarray([-2.0, 2.0, 2.0, -2.0, 3.0, -3.0])
        fused, decisions = common.apply_direction_constrained_reliability(
            clean, evidence, corrections
        )
        np.testing.assert_allclose(fused[[0, 1, 4, 5]], clean[[0, 1, 4, 5]])
        self.assertGreater(fused[2], clean[2])
        self.assertLess(fused[3], clean[3])
        self.assertEqual(decisions[0]["applied_logit_correction"], 0.0)
        self.assertEqual(decisions[1]["applied_logit_correction"], 0.0)
        self.assertGreater(decisions[2]["applied_logit_correction"], 0.0)
        self.assertLess(decisions[3]["applied_logit_correction"], 0.0)
        self.assertEqual(decisions[4]["applied_logit_correction"], 0.0)
        self.assertEqual(decisions[5]["applied_logit_correction"], 0.0)

    def test_secondary_models_cannot_select_the_primary(self) -> None:
        evaluation = self.policy["evaluation"]
        self.assertEqual(
            evaluation["primary_model"], "decontaminated_style_lr_l2_primary"
        )
        self.assertEqual(evaluation["matched_baseline_model"], "raw_style_lr_l2_control")
        self.assertTrue(
            evaluation["secondary_and_exploratory_models_forbidden_for_selection"]
        )

    def test_sync_manifest_has_a_closed_expected_payload_set(self) -> None:
        paths = sync_manifest.expected_paths(self.policy, self.step24_policy)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertGreaterEqual(len(paths), 20)
        self.assertTrue(
            all(
                str(path).startswith(str(common.resolve(self.policy["outputs_root"])))
                for path in paths
            )
        )
        self.assertTrue(
            {
                "scripts/step9_run_few_shot_adaptation.py",
                "scripts/step15_v7_common.py",
                "scripts/step24_common.py",
                "scripts/step24_build_style_embedding_cache.py",
                "scripts/step24_evaluate_content_independent_authorship.py",
                "schema/step15_v7_two_stage_policy.json",
                "schema/step24_content_independent_authorship_policy.json",
            }.issubset(set(sync_manifest.PRODUCERS))
        )

    def test_rank_tail_metrics_are_invariant_to_monotone_score_scaling(self) -> None:
        original = np.asarray([0.1, 0.2, 0.2, 0.9])
        transformed = original * 7.0 - 4.0
        np.testing.assert_allclose(
            evaluation.rank_percentiles(original),
            evaluation.rank_percentiles(transformed),
        )

    def test_pairwise_violation_rate_uses_half_credit_for_ties(self) -> None:
        negatives = np.asarray([0.2, 0.5])
        positives = np.asarray([0.5, 0.8])
        self.assertAlmostEqual(
            evaluation.pairwise_violation_rate(negatives, positives), 0.125
        )


if __name__ == "__main__":
    unittest.main()
