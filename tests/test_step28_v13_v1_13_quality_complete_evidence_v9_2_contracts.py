#!/usr/bin/env python3
"""Contracts for canonical V9.2 gate order and durable complete evidence."""

from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_complete_evidence_v9_2 as evidence
import step28_v13_v1_13_quality_gate_registry_v9_2 as registry
import step28_v13_v1_13_quality_audit_runner_v9_2 as runner
import step28_v13_v1_13_quality_result_assembler_v9_2 as assembler


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _passing_or_failing_value(spec: dict, fail: bool) -> float:
    threshold = float(spec["threshold"])
    comparison = str(spec["comparison"])
    if not fail:
        return threshold
    if comparison == "LESS_OR_EQUAL" or comparison == "EQUAL":
        return threshold + 1.0
    if comparison == "GREATER_OR_EQUAL":
        return threshold - 1.0
    raise AssertionError(comparison)


def _observations(*, failures: set[str] | None = None) -> dict[str, dict]:
    failures = set() if failures is None else failures
    output: dict[str, dict] = {}
    for spec in registry.GATE_REGISTRY:
        gate_id = str(spec["gate_id"])
        if spec["qualification_role"] == registry.DESCRIPTIVE:
            row = {
                "gate_id": gate_id,
                "observed": {
                    "symmetric_auc": 0.999,
                    "average_precision": 0.999,
                    "channel_differences": {"registered": True},
                },
                "gate_status": registry.NOT_APPLICABLE,
            }
        else:
            observed = _passing_or_failing_value(spec, gate_id in failures)
            passed = gate_id not in failures
            row = {
                "gate_id": gate_id,
                "observed": observed,
                "gate_status": "PASS" if passed else "FAIL",
                "passed": passed,
            }
        requirements = {
            "input_matrix_sha256": "input_matrix",
            "prediction_vector_sha256": "prediction_vector",
            "bootstrap_family_maxima_vector_sha256": (
                "bootstrap_family_maxima_vector"
            ),
        }
        for field, requirement in requirements.items():
            reason_field = field.removesuffix("_sha256") + "_not_applicable_reason"
            if spec[requirement] == registry.REQUIRED:
                row[field] = _hash(gate_id + ":" + field)
                row[reason_field] = None
            else:
                row[field] = None
                row[reason_field] = evidence._not_applicable_reason(spec, field)
        output[gate_id] = row
    return output


def _bindings() -> dict[str, str]:
    return {field: _hash(field) for field in evidence.BINDING_FIELDS}


class CompleteEvidenceV92Contracts(unittest.TestCase):
    def test_registry_has_complete_structure_42_5_5_family_partition(self) -> None:
        roles = [row["qualification_role"] for row in registry.GATE_REGISTRY]
        families = [row["family"] for row in registry.GATE_REGISTRY]
        expected_structure_count = 2 + len(
            registry.V9_2_STRUCTURE_METRICS
        ) + len(registry.ZERO_TOLERANCE_STRUCTURE_METRICS)
        self.assertEqual(
            len(registry.GATE_REGISTRY), expected_structure_count + 42 + 5 + 5
        )
        self.assertEqual(roles.count(registry.DESCRIPTIVE), 42)
        self.assertEqual(families.count("structure"), expected_structure_count)
        self.assertEqual(families.count("counterfactual_text"), 5)
        self.assertEqual(families.count("public_code_private_slot"), 5)
        self.assertRegex(registry.GATE_REGISTRY_SHA256, r"^[0-9a-f]{64}$")
        self.assertIn(
            "hard.structure.registered_visible_occurrence_multiset_difference_count",
            registry.GATE_IDS,
        )
        descriptive_ids = [
            value for value in registry.GATE_IDS if value.startswith("descriptive.")
        ]
        self.assertTrue(all("logistic_l2" in value or "hist_gradient_boosting_depth2" in value for value in descriptive_ids))

    def test_two_through_five_text_failures_keep_registry_order_and_all_entries(
        self,
    ) -> None:
        text_ids = [
            gate_id
            for gate_id in registry.GATE_IDS
            if gate_id.startswith("hard.text_deranged.")
        ]
        for count in range(2, 6):
            with self.subTest(count=count):
                failures = set(text_ids[:count])
                receipt = evidence.assemble_complete_quality_evidence(
                    observations=_observations(failures=failures),
                    bindings=_bindings(),
                )
                self.assertEqual(receipt["status"], "DATASET_INVALIDATED")
                self.assertEqual(receipt["failed_gate_ids"], text_ids[:count])
                self.assertEqual(
                    len(receipt["entries"]), len(registry.GATE_REGISTRY)
                )
                self.assertTrue(receipt["complete_quality_calculation"])

    def test_descriptive_extreme_values_never_change_qualification(self) -> None:
        receipt = evidence.assemble_complete_quality_evidence(
            observations=_observations(),
            bindings=_bindings(),
        )
        self.assertEqual(receipt["status"], "PASS")
        descriptive = [
            row
            for row in receipt["entries"]
            if row["qualification_role"] == registry.DESCRIPTIVE
        ]
        self.assertEqual(len(descriptive), 42)
        self.assertTrue(all(row["gate_status"] == "NOT_APPLICABLE" for row in descriptive))
        self.assertTrue(all("passed" not in row for row in descriptive))

    def test_name_sorting_drift_is_rejected_even_with_recomputed_self_hash(self) -> None:
        text_ids = [
            gate_id
            for gate_id in registry.GATE_IDS
            if gate_id.startswith("hard.text_deranged.")
        ]
        receipt = evidence.assemble_complete_quality_evidence(
            observations=_observations(failures=set(text_ids)),
            bindings=_bindings(),
        )
        drifted = copy.deepcopy(receipt)
        drifted["failed_gate_ids"] = sorted(receipt["failed_gate_ids"])
        self.assertNotEqual(drifted["failed_gate_ids"], receipt["failed_gate_ids"])
        drifted["canonical_self_hash"] = evidence._self_hash(drifted)
        with self.assertRaisesRegex(
            evidence.CompleteEvidenceV92Error, "result drift"
        ):
            evidence.validate_complete_quality_evidence(drifted)

    def test_wrapper_failure_preserves_complete_dataset_invalidation(self) -> None:
        first_hard = next(
            gate_id
            for gate_id in registry.GATE_IDS
            if gate_id.startswith("hard.structure.")
        )
        receipt = evidence.assemble_complete_quality_evidence(
            observations=_observations(failures={first_hard}),
            bindings=_bindings(),
        )
        terminal = evidence.wrapper_terminal_after_complete_evidence(
            evidence=receipt,
            wrapper_error=ValueError("fixture name-sort drift"),
        )
        self.assertEqual(
            terminal["status"], "DATASET_INVALIDATED_WITH_OUTER_WRAPPER_FAILURE"
        )
        self.assertTrue(terminal["dataset_invalidation_preserved"])
        self.assertFalse(terminal["pass_certified"])

    def test_primary_publication_failure_preserves_computed_invalidation(self) -> None:
        first_hard = next(
            gate_id
            for gate_id in registry.GATE_IDS
            if gate_id.startswith("hard.structure.")
        )
        receipt = evidence.assemble_complete_quality_evidence(
            observations=_observations(failures={first_hard}),
            bindings=_bindings(),
        )
        result = runner._complete_run_result(
            evidence=receipt,
            publication=None,
            publication_error=OSError("fixture exclusive publication failure"),
            consumed=(
                ROOT
                / "private_custody"
                / "fixture_quality_authorization.consumed.json"
            ),
        )
        self.assertEqual(
            result["status"],
            "DATASET_INVALIDATED_WITH_OUTER_WRAPPER_FAILURE",
        )
        self.assertTrue(result["cleanup_required"])
        self.assertTrue(
            result["terminal_wrapper"]["dataset_invalidation_preserved"]
        )

    def test_complete_evidence_is_exclusive_and_survives_later_wrapper_error(
        self,
    ) -> None:
        receipt = evidence.assemble_complete_quality_evidence(
            observations=_observations(),
            bindings=_bindings(),
        )
        with tempfile.TemporaryDirectory(prefix="step28-v9-2-evidence-") as temp:
            path = Path(temp) / "complete_quality_evidence.json"
            published = evidence.publish_complete_evidence_exclusive(path, receipt)
            self.assertTrue(path.is_file())
            self.assertEqual(published["status"], "PASS")
            with self.assertRaises(FileExistsError):
                evidence.publish_complete_evidence_exclusive(path, receipt)
            terminal = evidence.wrapper_terminal_after_complete_evidence(
                evidence=receipt,
                wrapper_error=RuntimeError("fixture outer wrapper failure"),
            )
            self.assertTrue(path.is_file())
            self.assertEqual(
                terminal["status"], "AUDITOR_EXECUTION_FAILED_PASS_NOT_CERTIFIED"
            )

    def test_structure_and_numerical_partitions_merge_only_in_registry_order(
        self,
    ) -> None:
        all_observations = _observations()
        structure_values = {
            str(spec["metric"]): all_observations[str(spec["gate_id"])][
                "observed"
            ]
            for spec in registry.GATE_REGISTRY
            if spec["family"] == "structure"
        }
        numerical = {
            gate_id: value
            for gate_id, value in all_observations.items()
            if not gate_id.startswith("hard.structure.")
        }
        merged = assembler.merge_complete_observations(
            structure_metric_values=structure_values,
            numerical_observations=numerical,
        )
        self.assertEqual(tuple(merged), registry.GATE_IDS)
        receipt = evidence.assemble_complete_quality_evidence(
            observations=merged,
            bindings=_bindings(),
        )
        self.assertEqual(receipt["status"], "PASS")

    def test_rehashed_top_level_field_or_entry_extension_is_rejected(self) -> None:
        receipt = evidence.assemble_complete_quality_evidence(
            observations=_observations(), bindings=_bindings()
        )
        drifted = copy.deepcopy(receipt)
        drifted["audit_a_b_truth_open_count"] = 1
        drifted["canonical_self_hash"] = evidence._self_hash(drifted)
        with self.assertRaisesRegex(evidence.CompleteEvidenceV92Error, "result drift"):
            evidence.validate_complete_quality_evidence(drifted)
        extended = copy.deepcopy(receipt)
        extended["entries"][0]["unexpected"] = "not allowed"
        extended["canonical_self_hash"] = evidence._self_hash(extended)
        with self.assertRaisesRegex(
            evidence.CompleteEvidenceV92Error, "entry exact schema drift"
        ):
            evidence.validate_complete_quality_evidence(extended)


if __name__ == "__main__":
    unittest.main()
