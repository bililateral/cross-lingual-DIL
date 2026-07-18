import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step25_v3_build_sync_manifest as sync_manifest  # noqa: E402
import step25_v3_common as common  # noqa: E402


class Step25V3CopyAwareDualChannelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.policy_path,
            cls.policy,
            cls.v7_policy,
            cls.step24_policy,
            cls.step25_v1_policy,
            cls.step25_v2_policy,
        ) = common.load_policy()

    def test_parent_results_are_immutable_and_output_root_isolated(self) -> None:
        roots = {
            self.policy["outputs_root"],
            self.policy["inputs"]["step24_outputs_root"],
            self.policy["inputs"]["step25_v1_outputs_root"],
            self.policy["inputs"]["step25_v2_outputs_root"],
        }
        self.assertEqual(len(roots), 4)
        self.assertIn("v3_copy_aware_dual_channel_20260718", self.policy["outputs_root"])
        self.assertTrue(
            self.policy["immutable_parent_contract"]["parent_outputs_overwrite_forbidden"]
        )

    def test_frozen_parent_references_are_present_and_non_promoted(self) -> None:
        references = common.frozen_parent_references(self.policy)
        self.assertEqual(set(references), {"step25_v1", "step25_v2"})
        for reference in references.values():
            self.assertFalse(reference["publication_promotion_eligible"])
            self.assertTrue(reference["metrics"])

    def test_d0_is_retrospective_and_cannot_promote_or_enter_graph_steps(self) -> None:
        boundary = self.policy["boundary"]
        self.assertTrue(boundary["hypothesis_informed_by_step25_v1_v2"])
        self.assertTrue(boundary["valid_or_test_read_forbidden"])
        self.assertTrue(boundary["parameter_or_threshold_search_on_d0_forbidden"])
        self.assertTrue(boundary["publication_promotion_hard_false"])
        self.assertTrue(boundary["step11_or_step17_entry_forbidden"])

    def test_fixed_model_matrix_has_one_primary_and_no_candidate_search(self) -> None:
        evaluation = self.policy["evaluation"]
        self.assertEqual(
            set(evaluation["model_specs"]),
            {
                "C0_matched_raw_style",
                "C1_raw_plus_clean_no_copy_penalty",
                "C2_copy_aware_dual_channel_primary",
                "C3_semantic_plus_copy_aware_sensitivity",
            },
        )
        self.assertEqual(evaluation["primary_model"], "C2_copy_aware_dual_channel_primary")
        self.assertEqual(evaluation["matched_baseline_model"], "C0_matched_raw_style")
        self.assertTrue(evaluation["candidate_selection_forbidden"])
        self.assertTrue(evaluation["sensitivity_model_selection_forbidden"])
        self.assertTrue(evaluation["valid_or_test_selection_forbidden"])

    def test_clean_feature_contract_excludes_identity_and_candidate_rules(self) -> None:
        contract = self.policy["feature_contract"]
        names = {
            name
            for values in common.feature_groups(self.policy).values()
            for name in values
        }
        self.assertNotIn("candidate_rule_count", names)
        self.assertNotIn("shared_contact_exact", names)
        self.assertTrue(contract["clean_scorer_identifier_features_forbidden"])
        self.assertTrue(contract["candidate_rule_features_forbidden"])
        self.assertTrue(contract["review_label_or_evidence_type_as_feature_forbidden"])

    def test_unreliable_clean_channel_uses_raw_fallback_not_zero_or_median(self) -> None:
        contract = self.policy["feature_contract"]
        self.assertEqual(contract["unreliable_pair_local_clean_value"], "raw_style_fallback")
        self.assertEqual(contract["unreliable_pair_local_delta_value"], 0.0)
        self.assertTrue(contract["missing_clean_style_encoded_as_zero_forbidden"])

    def test_primary_copy_residual_and_copy_risk_cannot_be_positive(self) -> None:
        contract = self.policy["feature_contract"]
        directions = self.policy["evaluation"]["model_specs"][
            "C2_copy_aware_dual_channel_primary"
        ]["coefficient_directions"]
        for name in contract["copy_residual_channel"]:
            self.assertEqual(directions[name], "nonpositive")
        for name in contract["copy_risk_channel"]:
            if name.endswith("reliable"):
                self.assertEqual(directions[name], "unconstrained")
            else:
                self.assertEqual(directions[name], "nonpositive")

    def test_primary_similarity_coefficients_cannot_be_negative(self) -> None:
        contract = self.policy["feature_contract"]
        directions = self.policy["evaluation"]["model_specs"][
            "C2_copy_aware_dual_channel_primary"
        ]["coefficient_directions"]
        for name in contract["raw_channel"] + contract["clean_channel"]:
            self.assertEqual(directions[name], "nonnegative")

    def test_projected_logistic_enforces_coefficient_directions(self) -> None:
        matrix = np.asarray(
            [
                [-2.0, 2.0],
                [-1.5, 1.8],
                [-1.0, 1.4],
                [1.0, -1.2],
                [1.5, -1.7],
                [2.0, -2.1],
            ],
            dtype=float,
        )
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
        cfg = dict(self.policy["evaluation"]["logistic"])
        cfg["l2_penalty"] = 1.0
        cfg["max_iter"] = 5000
        cfg["tolerance"] = 1e-7
        artifact = common.fit_direction_constrained_logistic(
            matrix,
            labels,
            np.ones(len(labels)),
            ["identity_similarity", "copy_risk"],
            {"identity_similarity": "nonnegative", "copy_risk": "nonpositive"},
            cfg,
        )
        self.assertTrue(artifact["solver_converged"])
        coefficients = artifact["parameter_coefficients_raw"]
        self.assertGreaterEqual(coefficients[0], -1e-10)
        self.assertLessEqual(coefficients[1], 1e-10)
        scores = common.apply_direction_constrained_logistic(matrix, artifact)
        self.assertGreater(float(np.mean(scores[3:])), float(np.mean(scores[:3])))

    def test_missingness_closure_uses_only_global_reliability(self) -> None:
        closure = self.policy["missingness_closure_control"]
        self.assertEqual(closure["reliability_feature"], "global_style_reliable")
        self.assertTrue(closure["pair_local_reliability_intersection_forbidden"])
        self.assertTrue(closure["selection_or_promotion_use_forbidden"])

    def test_operational_identifier_control_is_english_only_and_separate(self) -> None:
        operational = self.policy["operational_identifier_control"]
        self.assertEqual(operational["training_domain"], "en_only")
        self.assertTrue(operational["chinese_labels_for_expert_training_forbidden"])
        self.assertTrue(operational["clean_model_selection_use_forbidden"])
        self.assertEqual(operational["mixed_ambiguous_or_no_identifier_action"], "no_score_change")

    def test_all_preregistered_gates_are_required(self) -> None:
        gates = self.policy["d0_to_d1_replication_gates"]
        self.assertTrue(gates["all_gates_required"])
        self.assertEqual(gates["minimum_target_oof_ap_delta_over_C0"], 0.02)
        self.assertEqual(gates["minimum_target_oof_grouped_bootstrap_lower_bound"], 0.0)
        self.assertEqual(gates["maximum_target_template_clone_violation_rate_delta"], -0.05)

    def test_d1_and_f1_require_new_component_disjoint_boundaries(self) -> None:
        d1 = self.policy["future_boundaries"]["d1_independent_replication"]
        f1 = self.policy["future_boundaries"]["f1_prospective_final"]
        self.assertTrue(d1["seller_component_overlap_with_d0_forbidden"])
        self.assertGreaterEqual(d1["minimum_non_silver_direct_or_component_positive"], 30)
        self.assertTrue(d1["method_or_threshold_refit_on_d1_forbidden"])
        self.assertTrue(f1["collection_after_model_and_threshold_freeze_required"])
        self.assertTrue(f1["single_evaluation_only"])

    def test_output_map_is_collision_free_and_manifest_is_closed(self) -> None:
        outputs = self.policy["outputs"]
        self.assertEqual(len(outputs), len(set(outputs.values())))
        paths = sync_manifest.expected_paths(self.policy)
        self.assertEqual(len(paths), 9)
        self.assertEqual(len(paths), len(set(paths)))
        root = common.resolve(self.policy["outputs_root"])
        self.assertTrue(all(str(path).startswith(str(root)) for path in paths))

    def test_policy_inputs_do_not_reference_valid_or_test_artifacts(self) -> None:
        serialized = json.dumps(self.policy["inputs"], sort_keys=True).lower()
        self.assertNotIn("valid", serialized)
        self.assertNotIn("test", serialized)


if __name__ == "__main__":
    unittest.main()
