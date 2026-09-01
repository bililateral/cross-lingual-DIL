from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_audit_a_blind_predict_v1 as runner
import step28_v13_v1_13_v9_4_1_blind_stage_protocol_v3 as blind
import step28_v13_v1_13_v9_4_1_model_training_core_v3 as core


class _FixedBooster:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if len(matrix) != len(self.values):
            raise AssertionError("fixture row-count drift")
        return self.values.copy()


class AuditABlindPredictionV1Contracts(unittest.TestCase):
    def test_live_contract_loads_only_frozen_public_models(self) -> None:
        result = runner.validate_contract()
        self.assertEqual(
            result["status"],
            "PASSED_AUDIT_A_BLIND_CONTRACT_NO_TRUTH_READ_NO_PREDICTION",
        )
        self.assertEqual(result["threshold_model_count"], 10)
        self.assertEqual(result["loaded_trained_model_count"], 8)
        self.assertEqual(result["audit_a_truth_reads"], 0)
        self.assertEqual(result["audit_b_truth_reads"], 0)
        self.assertFalse(result["prediction_written"])

    def test_policy_authorizes_audit_a_prediction_but_no_truth_or_audit_b(self) -> None:
        policy = runner.load_policy()
        self.assertEqual(policy["split"], "audit_a")
        self.assertEqual(policy["private_inputs"], [])
        self.assertEqual(
            policy["truth_read_budget"],
            {"audit_a_labels_or_qrels": 0, "audit_b_labels_or_qrels": 0},
        )
        self.assertEqual(
            policy["authorization"],
            {
                "audit_a_blind_prediction_authorized": True,
                "audit_a_truth_authorized": False,
                "audit_b_blind_prediction_authorized": False,
                "audit_b_truth_authorized": False,
            },
        )
        serialized = json.dumps(policy, ensure_ascii=False)
        self.assertNotIn("private_custody", serialized)

    def test_prediction_registry_preserves_zero_residual_nesting(self) -> None:
        row_count = 4
        p0 = np.asarray([0.1, 0.2, 0.7, 0.8], dtype="<f8")
        c0 = np.asarray([0.2, 0.3, 0.6, 0.7], dtype="<f8")
        rows = {
            "pair_uids": [f"pair_{index}" for index in range(row_count)],
            "m0_probability": p0,
            "c0_probability": c0,
            "base24": np.zeros((row_count, 24), dtype="<f8"),
            "identity33": np.ones((row_count, 33), dtype="<f8"),
        }
        residual = {
            model_id: {
                "beta": np.zeros(33, dtype="<f8"),
                "scale": np.ones(33, dtype="<f8"),
                "mu": np.zeros(33, dtype="<f8"),
            }
            for model_id in core.M1_IDS + ("m2",)
        }
        models = {
            "residual": residual,
            "m3": {
                "m3_base": {
                    "model": _FixedBooster(np.full(row_count, 0.4, dtype="<f8")),
                    "medians": np.zeros(24, dtype="<f8"),
                },
                "m3_joint": {
                    "model": _FixedBooster(np.full(row_count, 0.6, dtype="<f8")),
                    "medians": np.zeros(57, dtype="<f8"),
                },
            },
        }
        predictions = runner._predict_models(rows, models)
        self.assertEqual(set(predictions), set(core.MODEL_IDS))
        np.testing.assert_array_equal(predictions["c0"], c0)
        np.testing.assert_array_equal(predictions["m0"], p0)
        for model_id in core.M1_IDS + ("m2",):
            np.testing.assert_array_equal(predictions[model_id], p0)

    def test_blind_numerical_payload_accepts_no_truth_argument(self) -> None:
        probabilities = {
            model_id: np.asarray([0.1, 0.9], dtype="<f8")
            for model_id in core.MODEL_IDS
        }
        thresholds = {model_id: 0.5 for model_id in core.MODEL_IDS}
        payload = blind.build_blind_prediction_payload(
            split="audit_a",
            predictions=probabilities,
            thresholds=thresholds,
            row_key_sha256="a" * 64,
            training_parent_sha256="b" * 64,
        )
        self.assertEqual(payload["row_count"], 2)
        self.assertFalse(payload["audit_truth_read"])
        self.assertFalse(payload["labels_qrels_membership_or_controllers_read"])


if __name__ == "__main__":
    unittest.main()
