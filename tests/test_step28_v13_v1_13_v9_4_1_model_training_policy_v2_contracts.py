from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_model_training_common_v2 as common


class ModelTrainingPolicyV2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = common.load_policy()

    def test_exact_policy_bytes_and_closed_authorization(self) -> None:
        raw = common.DEFAULT_POLICY.read_bytes()
        self.assertEqual(len(raw), common.EXPECTED_POLICY_SIZE_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), common.EXPECTED_POLICY_SHA256)
        policy = common.parse_exact_policy_bytes(raw)
        unsigned = dict(policy)
        claimed = unsigned.pop("canonical_self_hash")
        self.assertEqual(claimed, common.EXPECTED_POLICY_CANONICAL_SELF_HASH)
        self.assertEqual(common.canonical_sha256(unsigned), claimed)
        self.assertFalse(policy["m0_m1_m2_m3_training_authorized"])
        self.assertFalse(policy["audit_a_truth_authorized"])
        self.assertFalse(policy["audit_b_truth_authorized"])
        self.assertFalse(any(
            value
            for key, value in policy["authorization_state"].items()
            if key.endswith("_authorized")
        ))

    def test_predecessor_bytes_are_inherited_but_invalid_fixture_has_no_authority(self) -> None:
        predecessor = common._validate_predecessor(self.policy)
        self.assertIn("compatibility_fixture", predecessor["labse_encoding"])
        supersession = self.policy["supersession"]
        self.assertEqual(supersession["success_prerequisites_logical_operator"], "AND")
        self.assertEqual(
            supersession["success_prerequisites"],
            [
                "english_151_m0_c0_exact_replay",
                "full_english_labse_v2_exact_replay",
            ],
        )
        self.assertFalse(supersession["full_v2_replaces_english_151_replay"])
        self.assertFalse(supersession["v1_fixture_can_authorize_any_later_stage"])
        self.assertIn(
            "predecessor.labse_encoding.compatibility_fixture",
            supersession["historical_invalidated_fields"],
        )
        self.assertIn(
            "predecessor.m2.active_offset_clip",
            supersession["historical_invalidated_fields"],
        )
        self.assertIn("active_offset_clip", predecessor["m2"])
        effective_m2 = dict(predecessor["m2"])
        effective_m2.pop("active_offset_clip")
        self.assertEqual(
            common.canonical_sha256(effective_m2),
            self.policy["inherited_section_canonical_sha256"][
                "m2_effective_remainder"
            ],
        )
        effective_projection = dict(predecessor["public_projection"])
        for field in (
            "allowed_observed_files",
            "m0_forbidden_inputs",
            "identity_stage_forbidden_inputs",
        ):
            effective_projection.pop(field)
        self.assertEqual(
            common.canonical_sha256(effective_projection),
            self.policy["inherited_section_canonical_sha256"][
                "public_projection_effective_remainder"
            ],
        )

    def test_full_v2_manifest_is_exact_label_free_success(self) -> None:
        spec = self.policy["authority_registry"][
            "full_english_compatibility_v2_success_manifest"
        ]
        path = common._verify_pin(spec, label="full v2 success manifest")
        manifest = common._verify_json_self_hash(
            path,
            spec["canonical_self_hash"],
            label="full v2 success manifest",
        )
        self.assertEqual(manifest["status"], spec["required_status"])
        self.assertTrue(manifest["embedding_matrix_exact_byte_match"])
        self.assertTrue(manifest["complete_733_pair_score_file_exact_byte_match"])
        self.assertEqual(manifest["supervised_labels_or_identity_evidence_read"], 0)
        self.assertEqual(manifest["audit_truth_read"], 0)
        self.assertFalse(manifest["model_parameters_updated"])
        self.assertEqual(
            manifest["exact_runtime"]["cudnn_runtime"],
            self.policy["formal_chinese_labse_runtime"]["cudnn_runtime"],
        )

    def test_stage_boundaries_are_positive_allowlists(self) -> None:
        allowlists = self.policy["stage_input_allowlists"]
        self.assertEqual(
            tuple(allowlists["base_public_projection_observed_basenames"]),
            common.EXPECTED_BASE_ALLOWLIST,
        )
        self.assertEqual(
            tuple(allowlists["identity_public_projection_observed_basenames"]),
            common.EXPECTED_IDENTITY_ALLOWLIST,
        )
        self.assertNotIn(
            "identity33_all_pairs.csv",
            allowlists["base_public_projection_observed_basenames"],
        )
        for forbidden in (
            "sellers.jsonl",
            "redacted_items.jsonl",
            "model_seller_profiles.jsonl",
        ):
            self.assertNotIn(
                forbidden,
                allowlists["identity_public_projection_observed_basenames"],
            )

    def test_resigned_allowlist_or_gate_drift_is_rejected(self) -> None:
        altered = copy.deepcopy(self.policy)
        altered["stage_input_allowlists"][
            "identity_public_projection_observed_basenames"
        ].append("sellers.jsonl")
        with self.assertRaisesRegex(common.ModelTrainingContractError, "allowlist"):
            common._validate_static_contract(altered)

        altered = copy.deepcopy(self.policy)
        altered["confirmatory_gates"]["audit_a"][
            "primary_simultaneous_lower_bound_gt"
        ] = 0.0
        with self.assertRaisesRegex(common.ModelTrainingContractError, "Audit A"):
            common._validate_static_contract(altered)

        altered = copy.deepcopy(self.policy)
        altered["confirmatory_gates"]["audit_b"][
            "cannot_rescue_audit_a_failure"
        ] = False
        with self.assertRaisesRegex(common.ModelTrainingContractError, "rescue"):
            common._validate_static_contract(altered)

    def test_raw_p0_contract_rejects_endpoints_without_training_clip(self) -> None:
        with self.assertRaisesRegex(common.ModelTrainingContractError, "strictly"):
            common.validate_p0(np.asarray([0.0, 0.5], dtype=np.float64))
        with self.assertRaisesRegex(common.ModelTrainingContractError, "strictly"):
            common.validate_p0(np.asarray([0.5, 1.0], dtype=np.float64))
        with self.assertRaisesRegex(common.ModelTrainingContractError, "non-finite"):
            common.validate_p0(np.asarray([0.5, np.nan], dtype=np.float64))

        p0 = np.asarray([1e-12, 0.25, 0.75, 1.0 - 1e-12], dtype=np.float64)
        observed = common.raw_logit(p0)
        expected = np.log(p0) - np.log1p(-p0)
        np.testing.assert_array_equal(observed, expected)
        self.assertFalse(self.policy["p0_offset_contract"][
            "training_offset_probability_clipping_allowed"
        ])

    def test_zero_residual_exactly_nests_m0_for_all_rows(self) -> None:
        p0 = np.asarray(
            [
                np.nextafter(0.0, 1.0),
                1e-12,
                0.1234567890123456,
                0.5,
                np.nextafter(1.0, 0.0),
            ],
            dtype="<f8",
        )
        phi = np.arange(15, dtype=np.float64).reshape(5, 3)
        beta = np.zeros(3, dtype=np.float64)
        active = np.asarray([True, False, True, True, False])
        observed = common.residual_probabilities(p0, phi, beta, active)
        self.assertEqual(observed.tobytes(order="C"), p0.tobytes(order="C"))

    def test_inactive_and_per_row_zero_residual_preserve_original_bytes(self) -> None:
        p0 = np.asarray([0.2, 0.3, 0.4, 0.5], dtype="<f8")
        phi = np.asarray(
            [[1.0, 0.0], [0.0, 0.0], [4.0, 0.0], [1.0, 0.0]],
            dtype=np.float64,
        )
        beta = np.asarray([0.25, 0.0], dtype=np.float64)
        active = np.asarray([True, True, False, False])
        observed = common.residual_probabilities(p0, phi, beta, active)
        self.assertNotEqual(observed[0].tobytes(), p0[0].tobytes())
        self.assertEqual(observed[1].tobytes(), p0[1].tobytes())
        self.assertEqual(observed[2].tobytes(), p0[2].tobytes())
        self.assertEqual(observed[3].tobytes(), p0[3].tobytes())

    def test_current_common_has_no_formal_confirmatory_gate(self) -> None:
        self.assertFalse(hasattr(common, "simultaneous_lower_bounds"))
        self.assertFalse(hasattr(common, "audit_gate_passes"))
        with self.assertRaisesRegex(
            common.ModelTrainingContractError,
            "separate frozen evaluator",
        ):
            common.require_no_current_confirmatory_evaluator()

    def test_future_confirmatory_evaluator_must_build_ap_from_real_indices(self) -> None:
        gates = self.policy["confirmatory_gates"]
        current = gates["current_common_evaluator"]
        self.assertEqual(
            current["status"], "NOT_IMPLEMENTED_CURRENT_COMMON_FAIL_CLOSED"
        )
        self.assertFalse(current["audit_a_evaluation_allowed"])
        self.assertFalse(current["audit_b_evaluation_allowed"])
        self.assertFalse(
            current["precomputed_bootstrap_ap_series_may_enter_formal_gate"]
        )
        self.assertTrue(current["future_evaluator_must_be_a_separate_frozen_module"])
        simultaneous = gates["simultaneous_lower_bound"]
        self.assertTrue(
            simultaneous["formal_evaluator_must_generate_bootstrap_ap_from_that_matrix"]
        )
        self.assertFalse(
            simultaneous[
                "precomputed_bootstrap_ap_series_may_be_passed_to_formal_gate"
            ]
        )
        self.assertEqual(simultaneous["index_matrix_shape"], [9999, 500])
        self.assertEqual(simultaneous["index_matrix_dtype"], "<i8")

    def test_confirmatory_formulas_and_strict_thresholds_remain_frozen(self) -> None:
        gates = self.policy["confirmatory_gates"]
        simultaneous = gates["simultaneous_lower_bound"]
        self.assertEqual(
            simultaneous["error_formula"], "e_bj=delta_hat_j-delta_star_bj"
        )
        self.assertEqual(simultaneous["critical_value"], "q=Q_0.95(max_j(e_bj))")
        self.assertEqual(simultaneous["lower_bound"], "L_j=delta_hat_j-q")
        self.assertEqual(gates["audit_a"]["primary_simultaneous_lower_bound_gt"], 0.03)
        self.assertEqual(gates["audit_b"]["primary_simultaneous_lower_bound_gt"], 0.015)
        self.assertTrue(gates["audit_b"]["cannot_rescue_audit_a_failure"])
        self.assertFalse(
            gates["audit_b"][
                "current_common_may_evaluate_before_exact_a_conclusion_pin"
            ]
        )

    def test_metric_registry_cannot_omit_c0_ap_pr_auc_mrr_or_map(self) -> None:
        registry = self.policy["required_report_registry"]
        self.assertEqual(tuple(registry["models"]), common.EXPECTED_MODELS)
        self.assertIn("c0", registry["models"])
        self.assertIn("average_precision", registry["pooled_classification_metrics"])
        self.assertIn("trapezoidal_pr_auc", registry["pooled_classification_metrics"])
        self.assertIn("mrr", registry["retrieval_metrics"])
        self.assertIn("map", registry["retrieval_metrics"])
        self.assertEqual(
            set(registry["required_confusion_matrix_objects"]),
            {"raw_confusion_matrix", "world_equal_confusion_matrix"},
        )
        self.assertEqual(
            registry["threshold_metric_aggregations"],
            ["raw_rows", "world_equal_confusion"],
        )
        self.assertFalse(registry["selective_omission_allowed"])

    def test_invalidated_prepublication_attempt_cannot_be_reused(self) -> None:
        attempts = self.policy["invalidated_prepublication_attempts"]
        self.assertEqual(len(attempts), 1)
        attempt = attempts[0]
        self.assertFalse(attempt["payload_retained"])
        self.assertFalse(attempt["path_reuse_allowed"])
        self.assertEqual(
            self.policy["outputs"]["english_151_replay"],
            "english_151_replay_attempt2",
        )

    def test_successor_policy_never_embeds_private_truth_values(self) -> None:
        text = common.DEFAULT_POLICY.read_text(encoding="utf-8")
        forbidden = (
            "28b45881522b1b21f2c7434fe137b04c03c3f846205ce4b504f7d3d717de5d69",
            "79c2ded145af89ff509d8d2f43de1d9e839a14082cb599f1e9fafb00bb34d303",
            "037a63ab4ee6325e98711730020b97def0ea7c7f254dfd2cd5138873abe5213f",
            "d3d993483b53cf46dfd85251ffe8cc2e7af4a39efbb1b0db10310cf371a1cc80",
        )
        for digest in forbidden:
            self.assertNotIn(digest, text)


if __name__ == "__main__":
    unittest.main()
