from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from scripts import (
    step28_v13_v1_13_v9_4_1_english_initialized_labse_finetune_linux_v1 as transfer,
)


class EnglishInitializedLabseFinetuneV1Tests(unittest.TestCase):
    def test_contract_and_exact_english_training_boundary(self) -> None:
        result = transfer.validate_contract()
        self.assertEqual(result["english_train_pairs"], 401)
        self.assertEqual(result["english_positive"], 116)
        self.assertEqual(result["english_negative"], 285)
        self.assertEqual(result["english_sellers"], 582)
        self.assertEqual(result["english_components"], 229)
        self.assertEqual(result["english_global_unique_texts"], 23557)
        self.assertEqual(result["english_validation_or_test_label_reads"], 0)
        self.assertEqual(result["audit_a_truth_reads"], 0)
        self.assertEqual(result["audit_b_truth_reads"], 0)

    def test_component_class_weights_balance_both_classes(self) -> None:
        labels = np.asarray([1, 1, 1, 0, 0, 0, 0], dtype=np.int64)
        components = ("a", "a", "b", "c", "c", "d", "e")
        weights = transfer._component_class_weights(labels, components)
        self.assertAlmostEqual(float(weights[labels == 1].sum()), 0.5)
        self.assertAlmostEqual(float(weights[labels == 0].sum()), 0.5)
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]))

    def test_only_two_capacity_matched_text_models_are_compared(self) -> None:
        policy = transfer.load_policy()
        self.assertEqual(set(policy["models"]), set(transfer.MODEL_IDS))
        generic = policy["models"]["generic_init_base"]
        english = policy["models"]["english_init_base"]
        self.assertEqual(generic["target_features"], english["target_features"])
        self.assertFalse(generic["identity33"])
        self.assertFalse(english["identity33"])

    def test_english_feature_order_matches_frozen_m0(self) -> None:
        policy = transfer.load_policy()
        artifact = json.loads(
            Path(
                "reports/step7_v4_1_style_free_classifier_selection/"
                "v1_20260724/final_train_model_artifacts.json"
            ).read_text(encoding="utf-8")
        )
        expected = (
            policy["english_source"]["legacy18_feature_names"]
            + policy["english_source"]["semantic6_feature_names"]
        )
        self.assertEqual(
            artifact["candidates"]["lightgbm__legacy18_labse"]["feature_names"],
            expected,
        )

    def test_run_has_no_audit_truth_loader_or_audit_split(self) -> None:
        policy = transfer.load_policy()
        self.assertEqual(
            policy["chinese_target"]["allowed_splits"], ["train", "development"]
        )
        self.assertEqual(
            policy["chinese_target"]["forbidden_splits"], ["audit_a", "audit_b"]
        )
        self.assertFalse(policy["audit_a_truth_authorized"])
        self.assertFalse(policy["audit_b_truth_authorized"])

    def test_smoke_does_not_load_formal_supervision(self) -> None:
        source = inspect.getsource(transfer.smoke_runtime)
        self.assertNotIn("load_english_source", source)
        self.assertNotIn("_load_inputs", source)


if __name__ == "__main__":
    unittest.main()
