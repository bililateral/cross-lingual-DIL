from __future__ import annotations

import copy
import hashlib
import inspect
import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_probe_policy_v9_4 as policy_v94


class QualityProbePolicyV94Contracts(unittest.TestCase):
    def test_pinned_policy_loads_read_only_and_authorizes_nothing(self) -> None:
        policy = policy_v94.load_formal_policy()
        self.assertEqual(
            policy["status"],
            "FROZEN_IMPLEMENTATION_POLICY_NO_RUN_NO_DATA_NO_TRAINING",
        )
        self.assertFalse(any(policy["authorization"].values()))
        self.assertIsNone(
            policy["upstream_contract"]["time_key_commitment_sha256"]
        )
        with self.assertRaises(TypeError):
            policy["gates"]["maximum_family_symmetric_auc"] = "1.000000000000"

    def test_every_registered_policy_value_is_exact_type_checked(self) -> None:
        raw = __import__("json").loads(policy_v94.POLICY_PATH.read_text(encoding="utf-8"))
        for label, mutation in (
            ("authorization", lambda value: value["authorization"].update(
                {"prebuild_shortcut_gate": 0}
            )),
            ("models", lambda value: value["probe_models"]["logistic_l2"].update(
                {"C": 1}
            )),
            ("upstream", lambda value: value["upstream_contract"].update(
                {"direct_r2_plan_read": True}
            )),
            ("bootstrap", lambda value: value["bootstrap"].update(
                {"replicates": 9998}
            )),
            ("gates", lambda value: value["gates"].update(
                {"maximum_family_symmetric_auc": "0.540000000000"}
            )),
        ):
            with self.subTest(label=label):
                forged = copy.deepcopy(raw)
                mutation(forged)
                with self.assertRaisesRegex(
                    policy_v94.QualityProbePolicyV94Error, label
                ):
                    policy_v94.validate_policy_payload(forged)

    def test_formal_split_requires_500_by_378_and_20_positives(self) -> None:
        row_keys: list[tuple[str, str]] = []
        labels: list[int] = []
        for world_index in range(500):
            world = f"world_{world_index:03d}"
            for pair_index in range(378):
                row_keys.append((world, f"pair_{pair_index:03d}"))
                labels.append(int(pair_index < 20))
        label_array = np.asarray(labels, dtype=np.int8)
        policy_v94.validate_formal_split(
            row_keys=tuple(row_keys), labels=label_array, split="train"
        )
        forged = label_array.copy()
        forged[20] = 1
        with self.assertRaisesRegex(
            policy_v94.QualityProbePolicyV94Error, "per-world closure"
        ):
            policy_v94.validate_formal_split(
                row_keys=tuple(row_keys), labels=forged, split="train"
            )

    def test_five_gates_are_all_required_and_boundary_is_inclusive(self) -> None:
        policy = policy_v94.load_formal_policy()
        result = {
            "single_feature_maximum_symmetric_roc_auc_by_view": {
                "model_visible_14": 0.52
            },
            "model_results": {
                "model_visible_14::logistic_l2": {
                    "symmetric_roc_auc": 0.53,
                    "average_precision": policy_v94.FORMAL_AP_BASELINE + 0.01,
                    "score_vector_sha256": "1" * 64,
                },
                "model_visible_14::hist_gradient_boosting_depth2": {
                    "symmetric_roc_auc": 0.52,
                    "average_precision": policy_v94.FORMAL_AP_BASELINE + 0.005,
                    "score_vector_sha256": "2" * 64,
                },
            },
            "maximum_symmetric_roc_auc": 0.53,
            "maximum_average_precision_uplift": (
                policy_v94.FORMAL_AP_BASELINE
                + 0.01
                - policy_v94.FORMAL_AP_BASELINE
            ),
            "bootstrap": {
                "replicates": 9999,
                "world_count": 500,
                "score_family_size": 2,
                "draws_raw_i8_c_sha256": (
                    "111b1338cc607c6bd78bad88efe47606ffa2230e9cc764eec940e84f86e56661"
                ),
                "family_max_symmetric_auc_vector_sha256": "3" * 64,
                "family_max_average_precision_uplift_vector_sha256": "4" * 64,
                "symmetric_auc_95_upper": 0.53,
                "average_precision_uplift_95_upper": 0.015,
            },
        }
        comparison = policy_v94._compare_formal_gates(result, policy)
        self.assertTrue(comparison["all_gates_passed"])
        self.assertNotIn("status", comparison)
        self.assertNotIn("claim", comparison)
        self.assertEqual(len(comparison["comparisons"]), 5)
        failed = copy.deepcopy(result)
        failed["bootstrap"]["average_precision_uplift_95_upper"] = 0.0150001
        self.assertFalse(
            policy_v94._compare_formal_gates(failed, policy)["all_gates_passed"]
        )

    def test_formal_runner_is_narrow_and_currently_unauthorized(self) -> None:
        policy = policy_v94.load_formal_policy()
        with self.assertRaisesRegex(
            policy_v94.QualityProbePolicyV94Error, "unauthorized"
        ):
            policy_v94.run_authorized_formal_gate(
                train_schedule=None,
                development_schedule=None,
                noise_signature_set=None,
                time_key_hex="00" * 32,
                policy=policy,
            )

    def test_material_commitment_drift_fails_before_schedule_access(self) -> None:
        time_key_hex = "01" * 32
        time_key_commitment = hashlib.sha256(
            bytes.fromhex(time_key_hex)
        ).hexdigest()
        with self.assertRaisesRegex(
            policy_v94.QualityProbePolicyV94Error,
            "upstream capability drift",
        ):
            policy_v94._assemble_formal_inputs_after_authorization(
                train_schedule=None,
                development_schedule=None,
                noise_signature_set=None,
                time_key_hex=time_key_hex,
                expected_noise_signature_rows_sha256=(
                    policy_v94.signatures_v94.EXPECTED_SIGNATURE_ROWS_SHA256
                ),
                expected_noise_signature_set_commitment_sha256=(
                    policy_v94.signatures_v94
                    .EXPECTED_SIGNATURE_SET_COMMITMENT_SHA256
                ),
                expected_time_key_commitment_sha256="f" * 64,
            )

        signatures = policy_v94.signatures_v94.build_noise_signatures()
        policy_v94._validate_upstream_material_commitments(
            noise_signature_set=signatures,
            time_key_hex=time_key_hex,
            expected_noise_signature_rows_sha256=(
                policy_v94.signatures_v94.EXPECTED_SIGNATURE_ROWS_SHA256
            ),
            expected_noise_signature_set_commitment_sha256=(
                policy_v94.signatures_v94
                .EXPECTED_SIGNATURE_SET_COMMITMENT_SHA256
            ),
            expected_time_key_commitment_sha256=time_key_commitment,
        )
        with self.assertRaisesRegex(
            policy_v94.QualityProbePolicyV94Error,
            "upstream capability drift",
        ):
            policy_v94._assemble_formal_inputs_after_authorization(
                train_schedule=None,
                development_schedule=None,
                noise_signature_set=signatures,
                time_key_hex=time_key_hex,
                expected_noise_signature_rows_sha256="f" * 64,
                expected_noise_signature_set_commitment_sha256=(
                    policy_v94.signatures_v94
                    .EXPECTED_SIGNATURE_SET_COMMITMENT_SHA256
                ),
                expected_time_key_commitment_sha256=time_key_commitment,
            )

    def test_no_public_decision_bypass_can_emit_passed_or_claim(self) -> None:
        self.assertFalse(hasattr(policy_v94, "decide_formal_gates"))
        self.assertFalse(
            hasattr(policy_v94.preparer_v94, "prepare_formal_split")
        )
        self.assertFalse(
            hasattr(
                policy_v94.labels_v94,
                "open_controller_truth_after_preparation",
            )
        )
        parameters = tuple(inspect.signature(
            policy_v94.run_authorized_formal_gate
        ).parameters)
        self.assertEqual(parameters, (
            "train_schedule",
            "development_schedule",
            "noise_signature_set",
            "time_key_hex",
            "policy",
        ))
        runner_source = inspect.getsource(
            policy_v94.run_authorized_formal_gate
        )
        self.assertLess(
            runner_source.index('prebuild_shortcut_gate"] is not True'),
            runner_source.index("_assemble_formal_inputs_after_authorization"),
        )
        source = policy_v94.POLICY_PATH.read_text(encoding="utf-8")
        self.assertIn('"prebuild_shortcut_gate": false', source)


if __name__ == "__main__":
    unittest.main()
