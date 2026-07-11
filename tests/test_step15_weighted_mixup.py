from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step15_train_incremental_hard_negative as step15  # noqa: E402


class Step15WeightedMixupTests(unittest.TestCase):
    def test_v5r_policy_runtime_contract(self) -> None:
        with (ROOT / "schema" / "step15_evidence_type_policy.json").open("r", encoding="utf-8") as handle:
            policy = json.load(handle)
        experiment_names = {
            "step15_v5r_identity_only_curriculum_public_noise_weighted_strong_weighted_mixup",
            "step15_v5r_identity_only_curriculum_domain_balanced_public_noise_weighted_strong_weighted_mixup",
        }
        self.assertTrue(experiment_names.issubset(policy["experiments"]))
        self.assertEqual(policy["outputs"]["summary_json"], "reports/step15_v5r_weighted_mixup_summary.json")
        for experiment_name in experiment_names:
            cfg = policy["experiments"][experiment_name]
            mixup_cfg = dict(policy["training"]["positive_mixup"])
            mixup_cfg.update(cfg["positive_mixup"])
            self.assertEqual(mixup_cfg["scope"], "same_domain_same_evidence_type")
            self.assertEqual(mixup_cfg["minimum_source_training_sample_weight"], 0.55)
            self.assertEqual(mixup_cfg["nearest_neighbor_k"], 5)
            self.assertEqual(mixup_cfg["synthetic_weight_mode"], "minimum_parent_weight")

    def test_trusted_mixup_preserves_domain_evidence_weight_and_discrete_features(self) -> None:
        feature_names = ["continuous", "binary", "count"]
        x = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 3.0],
                [10.0, 0.0, 2.0],
                [11.0, 1.0, 4.0],
            ],
            dtype=float,
        )
        rows = [
            {
                "pair_uid": "en_a",
                "step15_pool": "en_content_train_pool",
                "evidence_type": "same_controller_direct_identifier",
                "evidence_type_confident": "1",
                "usable_for_core_transfer": "1",
                "core_transfer_eligible": "1",
                "training_sample_weight": "1.0",
            },
            {
                "pair_uid": "en_b",
                "step15_pool": "en_content_train_pool",
                "evidence_type": "same_controller_direct_identifier",
                "evidence_type_confident": "1",
                "usable_for_core_transfer": "1",
                "core_transfer_eligible": "1",
                "training_sample_weight": "0.55",
            },
            {
                "pair_uid": "zh_a",
                "step15_pool": "zh_target_strict",
                "evidence_type": "same_controller_style_structural_soft",
                "evidence_type_confident": "1",
                "usable_for_core_transfer": "1",
                "core_transfer_eligible": "1",
                "training_sample_weight": "1.0",
            },
            {
                "pair_uid": "zh_b",
                "step15_pool": "zh_target_strict",
                "evidence_type": "same_controller_style_structural_soft",
                "evidence_type_confident": "1",
                "usable_for_core_transfer": "1",
                "core_transfer_eligible": "1",
                "training_sample_weight": "0.55",
            },
        ]
        y = np.ones(4, dtype=float)
        y_evidence = np.asarray([0, 0, 1, 1], dtype=int)
        cfg = {
            "scope": "same_domain_same_evidence_type",
            "multiplier": 1.0,
            "beta_alpha": 0.4,
            "minimum_source_training_sample_weight": 0.55,
            "require_usable_for_core_transfer": True,
            "require_core_transfer_eligible": True,
            "require_evidence_type_confident": True,
            "nearest_neighbor_k": 1,
            "synthetic_weight_mode": "minimum_parent_weight",
            "copy_anchor_feature_names": ["binary", "count"],
        }

        x_aug, _, _, augmented_rows, count, diagnostics = step15.add_positive_mixup(
            x,
            y,
            y_evidence,
            rows,
            cfg,
            np.random.default_rng(20260711),
            feature_names,
        )

        self.assertEqual(count, 4)
        self.assertEqual(diagnostics["cross_domain_parent_count"], 0)
        self.assertEqual(diagnostics["cross_evidence_type_parent_count"], 0)
        row_index = {row["pair_uid"]: idx for idx, row in enumerate(rows)}
        for offset, synthetic_row in enumerate(augmented_rows[len(rows):]):
            left_idx = row_index[synthetic_row["mixup_parent_left_pair_uid"]]
            right_idx = row_index[synthetic_row["mixup_parent_right_pair_uid"]]
            self.assertEqual(rows[left_idx]["step15_pool"], rows[right_idx]["step15_pool"])
            self.assertEqual(rows[left_idx]["evidence_type"], rows[right_idx]["evidence_type"])
            self.assertEqual(float(synthetic_row["training_sample_weight"]), 0.55)
            np.testing.assert_array_equal(x_aug[len(rows) + offset, [1, 2]], x[left_idx, [1, 2]])

    def test_effective_domain_balance_equalizes_weight_mass(self) -> None:
        weights = np.asarray([2.0, 1.0, 0.25, 0.25], dtype=float)
        rows = [
            {"step15_pool": "en_content_train_pool"},
            {"step15_pool": "en_content_train_pool"},
            {"step15_pool": "zh_target_strict"},
            {"step15_pool": "zh_target_strict"},
        ]
        adjusted, diagnostics = step15.apply_effective_domain_balance(
            weights,
            rows,
            ["en_content_train_pool", "zh_target_strict"],
        )
        self.assertAlmostEqual(float(np.sum(adjusted[:2])), float(np.sum(adjusted[2:])), places=12)
        self.assertAlmostEqual(float(np.mean(adjusted)), float(np.mean(weights)), places=12)
        self.assertEqual(diagnostics["method"], "post_quality_effective_weight_mass")

    def test_effective_domain_balance_rejects_synthetic_pseudo_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown domains"):
            step15.apply_effective_domain_balance(
                np.asarray([1.0, 1.0, 1.0], dtype=float),
                [
                    {"step15_pool": "en_content_train_pool"},
                    {"step15_pool": "zh_target_strict"},
                    {"step15_pool": "cross_domain_mixup"},
                ],
                ["en_content_train_pool", "zh_target_strict"],
            )

    def test_legacy_domain_balance_remains_reproducible(self) -> None:
        adjusted, diagnostics = step15.legacy_domain_balanced_binary_weights(
            np.asarray([1.0, 0.0, 1.0], dtype=float),
            [
                {"step15_pool": "en_content_train_pool"},
                {"step15_pool": "zh_target_strict"},
                {"step15_pool": "cross_domain_mixup"},
            ],
        )
        self.assertEqual(len(adjusted), 3)
        self.assertEqual(diagnostics["method"], "legacy_raw_row_count_before_quality_weights")


if __name__ == "__main__":
    unittest.main()
