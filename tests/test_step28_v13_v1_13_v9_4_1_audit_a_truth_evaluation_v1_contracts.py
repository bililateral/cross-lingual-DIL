from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_audit_a_truth_evaluate_v1 as runner
import step28_v13_v1_13_v9_4_1_model_training_core_v3 as core


class AuditATruthEvaluationV1Contracts(unittest.TestCase):
    def test_live_contract_validates_without_reading_truth(self) -> None:
        with mock.patch.object(
            runner.training,
            "_read_labels",
            side_effect=AssertionError("validate-contract must not read labels"),
        ), mock.patch.object(
            runner.training,
            "_read_qrels_relevance",
            side_effect=AssertionError("validate-contract must not read qrels"),
        ):
            result = runner.validate_contract()
        self.assertEqual(
            result["status"],
            "PASSED_AUDIT_A_TRUTH_EVALUATION_CONTRACT_NO_TRUTH_READ",
        )
        self.assertEqual(result["row_count"], 189000)
        self.assertEqual(result["model_count"], 10)
        self.assertEqual(result["threshold_count"], 10)
        self.assertEqual(result["audit_a_truth_reads"], 0)
        self.assertEqual(result["audit_b_truth_reads"], 0)
        self.assertFalse(result["formal_evaluation_performed"])

    def test_policy_authorizes_only_audit_a_labels_and_qrels(self) -> None:
        policy = runner.load_policy()
        self.assertEqual(
            policy["canonical_self_hash"],
            "29a46f14a9ea767dd46e3f7779910912eb1f5aa41dd1765b978c2482854ad17f",
        )
        self.assertEqual(policy["split"], "audit_a")
        self.assertEqual(
            set(policy["authorized_private_inputs"]),
            {"audit_a_labels", "audit_a_qrels"},
        )
        self.assertEqual(
            policy["truth_read_budget"],
            {
                "audit_a_labels_semantic_reads": 1,
                "audit_a_qrels_semantic_reads": 1,
                "audit_b_labels_or_qrels_semantic_reads": 0,
            },
        )
        self.assertEqual(
            policy["authorization"],
            {
                "audit_a_truth_evaluation_authorized": True,
                "model_or_threshold_update_authorized": False,
                "audit_b_blind_prediction_authorized": False,
                "audit_b_truth_authorized": False,
            },
        )
        paths = [
            item["path"] for item in policy["authorized_private_inputs"].values()
        ]
        self.assertTrue(all(path.startswith("audit_a/") for path in paths))
        self.assertFalse(any(path.startswith("audit_b/") for path in paths))

    def test_frozen_predictions_and_row_keys_are_bound_before_truth(self) -> None:
        policy = runner.load_policy()
        _, blind_manifest, rows, predictions, thresholds = (
            runner._load_frozen_public_inputs(policy)
        )
        self.assertEqual(
            blind_manifest["canonical_self_hash"],
            "2f4b39899f27eb1dec6ff57ab2b9ce3f7c7dffb3f50d3424c51c3f322367cc4f",
        )
        self.assertEqual(len(rows["pair_uids"]), 189000)
        self.assertEqual(set(predictions), set(core.MODEL_IDS))
        self.assertEqual(set(thresholds), set(core.MODEL_IDS))

    def test_run_reads_each_audit_a_truth_once_and_keeps_audit_b_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            output_root = temp_root / "formal_output"
            labels_path = temp_root / "pair_labels.csv"
            qrels_path = temp_root / "qrels.jsonl"
            labels_path.write_bytes(b"fixture-labels")
            qrels_path.write_bytes(b"fixture-qrels")
            labels = np.asarray([1, 0], dtype=np.int8)
            indices = np.zeros((2, 1), dtype="<i8")
            index_hash = hashlib.sha256(indices.tobytes(order="C")).hexdigest()
            policy = {
                "formal_output_root": "formal_output",
                "private_supervision_root": ".",
                "authorized_private_inputs": {
                    "audit_a_labels": {"path": "pair_labels.csv"},
                    "audit_a_qrels": {"path": "qrels.jsonl"},
                },
                "truth_read_budget": {
                    "audit_a_labels_semantic_reads": 1,
                    "audit_a_qrels_semantic_reads": 1,
                    "audit_b_labels_or_qrels_semantic_reads": 0,
                },
                "expected_layout": {
                    "rows_per_world": 2,
                    "positive_rows_per_world": 1,
                },
                "bootstrap": {
                    "replicates": 2,
                    "world_count": 1,
                    "index_bytes_sha256": index_hash,
                },
                "canonical_self_hash": "policy-hash",
            }
            rows = {
                "pair_uids": ["a||b", "a||c"],
                "world_ordinals": np.asarray([0, 0], dtype="<i8"),
                "seller_uid_left": ["a", "a"],
                "seller_uid_right": ["b", "c"],
            }
            predictions = {
                model_id: np.asarray([0.8, 0.2], dtype="<f8")
                for model_id in core.MODEL_IDS
            }
            thresholds = {model_id: 0.5 for model_id in core.MODEL_IDS}
            blind_manifest = {"canonical_self_hash": "blind-hash"}
            v3_policy = {"canonical_self_hash": "v3-hash"}
            evaluation = {
                "status": "NUMERICAL_EVALUATION_COMPLETE_NO_FORMAL_TRUTH_AUTHORITY",
                "gate": {"numerical_gate_passed": True},
            }

            def private_path(_base, spec, _label):
                return temp_root / spec["path"]

            with mock.patch.object(runner, "ROOT", temp_root), mock.patch.object(
                runner, "load_policy", return_value=policy
            ), mock.patch.object(
                runner,
                "_load_frozen_public_inputs",
                return_value=(v3_policy, blind_manifest, rows, predictions, thresholds),
            ), mock.patch.object(
                runner.training, "_verify_file_record", side_effect=private_path
            ), mock.patch.object(
                runner.training, "_read_labels", return_value=labels
            ) as read_labels, mock.patch.object(
                runner.training, "_read_qrels_relevance", return_value=labels.copy()
            ) as read_qrels, mock.patch.object(
                runner.core, "build_bootstrap_indices", return_value=indices
            ), mock.patch.object(
                runner.evaluator,
                "evaluate_split_from_raw_inputs",
                return_value=evaluation,
            ) as evaluate:
                result = runner.run_evaluation()

            self.assertEqual(read_labels.call_count, 1)
            self.assertEqual(read_qrels.call_count, 1)
            self.assertEqual(evaluate.call_count, 1)
            self.assertTrue(result["numerical_gate_passed"])
            self.assertEqual(result["audit_b_truth_reads"], 0)
            summary = json.loads(
                (output_root / "audit_a_evaluation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(summary["audit_b_predictions_created"])
            self.assertFalse(summary["audit_b_truth_authorized_by_this_result"])
            self.assertTrue(summary["future_audit_b_blind_prediction_may_be_requested"])


if __name__ == "__main__":
    unittest.main()
