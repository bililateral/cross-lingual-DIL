import copy
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step25_v3_1_build_sync_manifest as sync_manifest  # noqa: E402
import step25_v3_1_common as common  # noqa: E402
import step25_v3_common as frozen_v3  # noqa: E402


class Step25V31SolverConvergenceContracts(unittest.TestCase):
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
        _, cls.base_policy, *_ = frozen_v3.load_policy(
            ROOT / "schema" / "step25_v3_copy_aware_dual_channel_policy.json"
        )

    def test_only_solver_termination_and_output_namespace_changed(self) -> None:
        for key in (
            "feature_contract",
            "boundary",
            "immutable_parent_contract",
            "missingness_closure_control",
            "operational_identifier_control",
            "d0_to_d1_replication_gates",
        ):
            self.assertEqual(self.policy[key], self.base_policy[key])
        for key in (
            "canonical_split",
            "fold_count",
            "fold_seed",
            "model_specs",
            "primary_model",
            "matched_baseline_model",
            "grouped_bootstrap_resamples",
            "grouped_bootstrap_seed",
        ):
            self.assertEqual(
                self.policy["evaluation"][key], self.base_policy["evaluation"][key]
            )
        self.assertNotEqual(
            self.policy["outputs_root"], self.base_policy["outputs_root"]
        )

    def test_optimizer_hyperparameters_are_frozen(self) -> None:
        repaired = self.policy["evaluation"]["logistic"]
        base = self.base_policy["evaluation"]["logistic"]
        for key in (
            "l2_penalty",
            "max_iter",
            "tolerance",
            "minimum_step_size",
            "backtracking_factor",
            "armijo_constant",
            "class_weight",
            "standardize_features",
            "sample_weight_total_normalization",
        ):
            self.assertEqual(repaired[key], base[key])
        self.assertFalse(repaired["relative_loss_is_a_convergence_criterion"])

    def test_relative_loss_alone_cannot_mark_convergence(self) -> None:
        matrix = np.asarray(
            [[-2.0, 0.0], [-1.0, 0.5], [1.0, 0.5], [2.0, 0.0]],
            dtype=float,
        )
        labels = np.asarray([0.0, 0.0, 1.0, 1.0])
        cfg = copy.deepcopy(self.policy["evaluation"]["logistic"])
        cfg["max_iter"] = 1
        cfg["projected_gradient_tolerance"] = 1e-20
        cfg["relative_loss_diagnostic_tolerance"] = 1.0
        artifact = common.fit_direction_constrained_logistic(
            matrix,
            labels,
            np.ones(len(labels)),
            ["positive_feature", "negative_feature"],
            {
                "positive_feature": "nonnegative",
                "negative_feature": "nonpositive",
            },
            cfg,
        )
        self.assertFalse(artifact["solver_converged"])
        self.assertEqual(
            artifact["solver_termination_reason"],
            "maximum_iterations_with_unmet_kkt",
        )
        self.assertGreater(artifact["solver_final_projected_gradient"], 1e-20)
        self.assertFalse(artifact["relative_loss_used_for_convergence"])

    def test_solver_reaches_kkt_and_respects_directions(self) -> None:
        matrix = np.asarray(
            [
                [-2.0, 2.0],
                [-1.5, 1.5],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.5, -1.5],
                [2.0, -2.0],
            ],
            dtype=float,
        )
        labels = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        cfg = copy.deepcopy(self.policy["evaluation"]["logistic"])
        artifact = common.fit_direction_constrained_logistic(
            matrix,
            labels,
            np.ones(len(labels)),
            ["positive_feature", "negative_feature"],
            {
                "positive_feature": "nonnegative",
                "negative_feature": "nonpositive",
            },
            cfg,
        )
        self.assertTrue(artifact["solver_converged"])
        self.assertEqual(
            artifact["solver_termination_reason"],
            "projected_gradient_kkt_tolerance",
        )
        self.assertLessEqual(
            artifact["solver_final_projected_gradient"],
            cfg["projected_gradient_tolerance"],
        )
        self.assertGreaterEqual(artifact["parameter_coefficients_raw"][0], 0.0)
        self.assertLessEqual(artifact["parameter_coefficients_raw"][1], 0.0)

    def test_solver_accepts_a_kkt_active_boundary(self) -> None:
        matrix = np.asarray(
            [[-2.0], [-1.0], [1.0], [2.0]],
            dtype=float,
        )
        labels = np.asarray([0.0, 0.0, 1.0, 1.0])
        artifact = common.fit_direction_constrained_logistic(
            matrix,
            labels,
            np.ones(len(labels)),
            ["wrong_direction_feature"],
            {"wrong_direction_feature": "nonpositive"},
            copy.deepcopy(self.policy["evaluation"]["logistic"]),
        )
        self.assertTrue(artifact["solver_converged"])
        self.assertLessEqual(
            artifact["solver_final_projected_gradient"],
            self.policy["evaluation"]["logistic"][
                "projected_gradient_tolerance"
            ],
        )
        self.assertAlmostEqual(artifact["parameter_coefficients_raw"][0], 0.0)

    def test_manifest_rejects_non_kkt_artifact(self) -> None:
        payload = {
            "artifact": {
                "model_family": "step25_v3_1_direction_constrained_logistic_l2",
                "solver_converged": True,
                "solver_termination_reason": "relative_loss_stagnation",
                "solver_final_projected_gradient": 0.1,
                "relative_loss_used_for_convergence": True,
            }
        }
        with self.assertRaises(ValueError):
            sync_manifest.validate_solver_artifacts(self.policy, payload)

    def test_manifest_accepts_only_kkt_artifacts(self) -> None:
        tolerance = self.policy["evaluation"]["logistic"][
            "projected_gradient_tolerance"
        ]
        payload = {
            "artifact": {
                "model_family": "step25_v3_1_direction_constrained_logistic_l2",
                "solver_converged": True,
                "solver_termination_reason": "projected_gradient_kkt_tolerance",
                "solver_final_projected_gradient": tolerance / 2.0,
                "relative_loss_used_for_convergence": False,
            }
        }
        audit = sync_manifest.validate_solver_artifacts(self.policy, payload)
        self.assertEqual(audit["artifact_count"], 1)
        self.assertLessEqual(audit["maximum_final_projected_gradient"], tolerance)

    def test_old_v3_results_are_not_overwritten(self) -> None:
        self.assertIn("v3_1_solverfix_20260718", self.policy["outputs_root"])
        self.assertNotEqual(
            self.policy["outputs_root"],
            "reports/step25_template_decontaminated_authorship/v3_copy_aware_dual_channel_20260718",
        )

    def test_valid_test_and_publication_boundaries_remain_closed(self) -> None:
        boundary = self.policy["boundary"]
        self.assertTrue(boundary["valid_or_test_read_forbidden"])
        self.assertTrue(boundary["parameter_or_threshold_search_on_d0_forbidden"])
        self.assertTrue(boundary["publication_promotion_hard_false"])
        self.assertTrue(boundary["step11_or_step17_entry_forbidden"])

    def test_closed_manifest_hashes_every_repair_producer(self) -> None:
        required = {
            "schema/step25_v3_1_solver_convergence_policy.json",
            "scripts/step25_v3_1_common.py",
            "scripts/step25_v3_1_build_dual_channel_features.py",
            "scripts/step25_v3_1_evaluate_copy_aware_fusion.py",
            "scripts/step25_v3_1_train_operational_identifier_control.py",
            "scripts/step25_v3_1_build_sync_manifest.py",
            "scripts/run_step25_v3_1_solverfix_linux_20260718.sh",
            "tests/test_step25_v3_1_solver_convergence_contracts.py",
            "docs/STEP25_V3_1_SOLVER_CONVERGENCE_REPAIR_20260718.zh.md",
        }
        self.assertTrue(required.issubset(set(sync_manifest.PRODUCERS)))


if __name__ == "__main__":
    unittest.main()
