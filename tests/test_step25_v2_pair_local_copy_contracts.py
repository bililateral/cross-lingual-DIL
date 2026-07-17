import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step25_v2_build_sync_manifest as sync_manifest  # noqa: E402
import step25_v2_common as common  # noqa: E402


class Step25V2PairLocalCopyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.policy_path,
            cls.policy,
            cls.step24_policy,
            cls.step25_v1_policy,
        ) = common.load_policy()

    def test_parent_results_are_frozen_and_output_root_is_isolated(self) -> None:
        self.assertEqual(
            self.policy["inputs"]["step25_v1_outputs_root"],
            "reports/step25_template_decontaminated_authorship/v1_20260717",
        )
        self.assertEqual(
            self.policy["outputs_root"],
            "reports/step25_template_decontaminated_authorship/v2_pair_local_diagnostic_20260717",
        )
        self.assertEqual(len({
            self.policy["outputs_root"],
            self.policy["inputs"]["step24_outputs_root"],
            self.policy["inputs"]["step25_v1_outputs_root"],
        }), 3)

    def test_retrospective_boundary_can_never_promote(self) -> None:
        boundary = self.policy["boundary"]
        self.assertTrue(boundary["hypothesis_informed_retrospective"])
        self.assertTrue(boundary["d1_candidate_eligibility_hard_false"])
        self.assertTrue(boundary["publication_promotion_hard_false"])
        self.assertTrue(boundary["step11_or_step17_entry_forbidden"])
        self.assertTrue(boundary["valid_or_test_read_forbidden"])

    def test_detector_recovers_copy_supported_by_only_the_pair(self) -> None:
        cfg = self.policy["pair_local_copy_detector"]
        shared = "仅在这两个卖家之间出现的长复制广告固定内容用于配对局部检测"
        result = common.detect_pair_local_copy(
            f"甲方独立风格开头{shared}甲方独立风格结尾",
            f"乙方完全不同开头{shared}乙方完全不同结尾",
            cfg,
        )
        self.assertGreater(result["shared_shingle_count"], 0)
        self.assertGreaterEqual(result["left_masked_character_count"], 24)
        self.assertGreaterEqual(result["right_masked_character_count"], 24)
        self.assertNotIn(shared, result["left_clean_text"])
        self.assertNotIn(shared, result["right_clean_text"])
        self.assertTrue(result["shared_shingle_hashes"])
        self.assertTrue(all(len(value) == 64 for value in result["shared_shingle_hashes"]))
        self.assertNotIn("shared_span_text", result)

    def test_short_incidental_overlap_is_not_masked(self) -> None:
        cfg = self.policy["pair_local_copy_detector"]
        result = common.detect_pair_local_copy(
            "卖家甲提供独立商品说明和交付方式共同词",
            "卖家乙提供不同服务介绍和售后规则共同词",
            cfg,
        )
        self.assertEqual(result["left_masked_character_count"], 0)
        self.assertEqual(result["right_masked_character_count"], 0)

    def test_oversized_copy_is_bounded_instead_of_erasing_the_pair(self) -> None:
        cfg = self.policy["pair_local_copy_detector"]
        shared = "重复模板正文" * 80
        result = common.detect_pair_local_copy(shared + "甲方尾部", shared + "乙方尾部", cfg)
        self.assertLessEqual(result["left_mask_fraction"], 0.95)
        self.assertLessEqual(result["right_mask_fraction"], 0.95)
        self.assertTrue(result["left_clean_text"])
        self.assertTrue(result["right_clean_text"])

    def test_fold_local_median_imputation_never_uses_zero_missingness(self) -> None:
        train_style = np.asarray([[0.2, 0.4], [np.nan, np.nan], [0.8, 0.6]])
        score_style = np.asarray([[np.nan, np.nan], [0.7, 0.9]])
        train_design, score_design, artifact = common.matched_missingness_design(
            train_style,
            np.asarray([True, False, True]),
            score_style,
            np.asarray([False, True]),
            "fold_train_reliable_median_plus_indicator",
        )
        np.testing.assert_allclose(artifact["fold_train_style_medians"], [0.5, 0.5])
        np.testing.assert_allclose(train_design[1], [0.5, 0.5, 0.0])
        np.testing.assert_allclose(score_design[0], [0.5, 0.5, 0.0])
        np.testing.assert_allclose(score_design[1], [0.7, 0.9, 1.0])
        self.assertFalse(artifact["missing_encoded_as_fixed_zero"])

    def test_raw_fallback_is_explicit_and_finite(self) -> None:
        style = np.asarray([[0.1, 0.2], [0.8, 0.9]])
        train, score, artifact = common.matched_missingness_design(
            style,
            np.asarray([False, True]),
            style[::-1],
            np.asarray([True, False]),
            "raw_style_fallback_plus_indicator",
        )
        self.assertTrue(np.all(np.isfinite(train)))
        self.assertTrue(np.all(np.isfinite(score)))
        self.assertIsNone(artifact["fold_train_style_medians"])
        self.assertFalse(artifact["missing_encoded_as_fixed_zero"])

    def test_p0_and_p2_share_the_exact_same_reliability_mask(self) -> None:
        specs = self.policy["evaluation"]["model_specs"]
        self.assertEqual(
            specs["P0_raw_style_matched_missingness"]["reliability_feature"],
            specs["P2_pair_local_clean_style_matched_missingness"]["reliability_feature"],
        )
        self.assertEqual(
            specs["P0_raw_style_matched_missingness"]["missingness_mode"],
            specs["P2_pair_local_clean_style_matched_missingness"]["missingness_mode"],
        )
        self.assertEqual(
            specs["P1_global_clean_style_matched_missingness"]["reliability_feature"],
            "global_and_pair_local_style_reliable",
        )

    def test_sensitivity_controls_cannot_select_or_promote(self) -> None:
        evaluation = self.policy["evaluation"]
        self.assertEqual(
            evaluation["model_specs"]["P3_pair_local_clean_raw_fallback"]["selection_role"],
            "sensitivity_only",
        )
        self.assertTrue(evaluation["P4_reliable_pair_only_sensitivity"]["selection_forbidden"])
        self.assertTrue(evaluation["candidate_selection_forbidden"])
        self.assertTrue(evaluation["valid_or_test_selection_forbidden"])

    def test_sync_manifest_has_a_closed_payload_set(self) -> None:
        paths = sync_manifest.expected_paths(self.policy)
        self.assertEqual(len(paths), 19)
        self.assertEqual(len(paths), len(set(paths)))
        output_root = common.resolve(self.policy["outputs_root"])
        self.assertTrue(all(str(path).startswith(str(output_root)) for path in paths))
        self.assertIn(
            "scripts/step25_v2_evaluate_pair_local_copy.py",
            sync_manifest.PRODUCERS,
        )

    def test_policy_has_no_valid_or_test_artifact_input(self) -> None:
        serialized = json.dumps(self.policy["inputs"], sort_keys=True)
        self.assertNotIn("valid", serialized.lower())
        self.assertNotIn("test", serialized.lower())


if __name__ == "__main__":
    unittest.main()
