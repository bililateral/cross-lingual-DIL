from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step26_common as common  # noqa: E402


POLICY_PATH = ROOT / "schema" / "step26_frozen_authorship_bridge_policy.json"


class Step26FrozenAuthorshipBridgeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy_path, cls.policy, cls.step24_policy = common.load_policy(POLICY_PATH)

    def test_question_boundary_and_primary_are_preregistered(self) -> None:
        self.assertEqual(
            self.policy["frozen_models"]["primary_bridge_model"],
            "source_only_semantic_style_lr_l2_primary",
        )
        self.assertEqual(self.policy["evaluation"]["primary_metric"], "average_precision")
        self.assertTrue(self.policy["evaluation"]["internal_test_satisfies_no_promotion_gate"])
        self.assertTrue(self.policy["evaluation"]["publication_claim_requires_step20"])

    def test_exact_corrected_pair_boundaries_are_present(self) -> None:
        rows = common.load_evaluation_rows(self.policy)
        observed = {
            name: (
                len(values),
                sum(row["review_label"] == "positive" for row in values),
                sum(row["review_label"] == "negative" for row in values),
            )
            for name, values in rows.items()
        }
        self.assertEqual(observed["representative_valid"], (120, 30, 90))
        self.assertEqual(
            observed["internal_development_test_diagnostic_only"], (200, 50, 150)
        )
        self.assertFalse(
            set(row["pair_uid"] for row in rows["representative_valid"])
            & set(
                row["pair_uid"]
                for row in rows["internal_development_test_diagnostic_only"]
            )
        )

    def test_blind_encoding_allowlist_needs_no_label_file(self) -> None:
        allowlists = common.load_blind_pair_allowlists(self.policy)
        self.assertEqual(len(allowlists["representative_valid"]), 120)
        self.assertEqual(len(allowlists["internal_development_test_diagnostic_only"]), 200)
        for pair_uid in allowlists["representative_valid"][:10]:
            left, right = common.pair_uid_sellers(pair_uid)
            self.assertTrue(left)
            self.assertTrue(right)

    def test_source_models_are_frozen_english_artifacts_with_matching_dimensions(self) -> None:
        frozen = common.validate_frozen_sources(self.policy)
        artifacts = frozen["step24_artifacts"]["artifacts"]["source_only"]
        expected = {
            "e5_lr_l2_control": 1,
            "style_only_lr_l2_control": 2,
            "semantic_style_lr_l2_primary": 3,
        }
        for key, dimension in expected.items():
            record = artifacts[key]
            artifact = record["logistic_artifact"]
            self.assertEqual(record["train_row_count"], 401)
            self.assertEqual(len(record["feature_names"]), dimension)
            self.assertEqual(len(artifact["parameter_coefficients"]), dimension)
            self.assertEqual(len(artifact["standardization"]["mean"]), dimension)
            self.assertEqual(len(artifact["standardization"]["scale"]), dimension)

    def test_clean_text_and_encoder_contracts_are_immutable(self) -> None:
        self.assertFalse(self.step24_policy["clean_text_contract"]["encode_valid_or_test"])
        for key in self.policy["frozen_models"]["encoder_keys"]:
            cfg = self.step24_policy["frozen_style_encoders"][key]
            self.assertTrue(cfg["local_finetuning_forbidden"])
            self.assertRegex(cfg["revision"], r"^[0-9a-f]{40}$")
        self.assertFalse(self.policy["frozen_models"]["encoder_parameters_updated"])
        self.assertTrue(self.policy["frozen_models"]["source_artifact_refit_forbidden"])

    def test_component_grouped_bootstrap_is_deterministic(self) -> None:
        rows = [
            {
                "review_label": "positive" if index % 2 == 0 else "negative",
                "v7_component_id": f"component-{index:02d}",
            }
            for index in range(20)
        ]
        baseline = np.asarray(
            [0.6 if row["review_label"] == "positive" else 0.4 for row in rows]
        )
        candidate = np.asarray(
            [0.8 if row["review_label"] == "positive" else 0.2 for row in rows]
        )

        def separation(labels: np.ndarray, scores: np.ndarray) -> float:
            return float(np.mean(scores[labels == 1]) - np.mean(scores[labels == 0]))

        first = common.grouped_bootstrap_delta(
            rows, baseline, candidate, separation, 1000, 20260718
        )
        second = common.grouped_bootstrap_delta(
            rows, baseline, candidate, separation, 1000, 20260718
        )
        self.assertEqual(first, second)
        self.assertGreater(first["point_delta"], 0.0)

    def test_runtime_scripts_contain_no_model_fit_or_threshold_selection(self) -> None:
        paths = [
            ROOT / "scripts" / "step26_build_frozen_style_cache.py",
            ROOT / "scripts" / "step26_evaluate_frozen_authorship_bridge.py",
        ]
        source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for forbidden in (
            "fit_regularized_logistic",
            "LogisticRegression(",
            ".fit(",
            "best_threshold",
            "select_threshold",
        ):
            self.assertNotIn(forbidden, source)

    def test_promotion_gates_cannot_use_internal_test(self) -> None:
        gates = self.policy["promotion_gates"]
        self.assertEqual(gates["gate_scope"], "internal_mechanism_promotion_to_step26b_only_not_publication")
        self.assertEqual(gates["primary_valid_ap_gain_over_v8_clean_minimum"], 0.03)
        self.assertTrue(gates["all_gates_required"])


if __name__ == "__main__":
    unittest.main()
