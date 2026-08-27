#!/usr/bin/env python3
"""Contracts for the reusable V9.3 label-free residual checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_build_residual_checkpoint_v9_3 as checkpoint
import step28_v13_v1_13_construct_registered_negative_plan_v9_3 as constructor


PREFLIGHT = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_balanced_schedule_v9_3"
    / "design_preflight_v2_20260825"
)
JOINT_SIGNATURE = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_balanced_schedule_v9_3"
    / "joint_noise_signature_preflight_v2_20260826.json"
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return value


class ResidualCheckpointContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = read_json(PREFLIGHT / "train_balanced_schedule.json")
        cls.signatures = read_json(JOINT_SIGNATURE)

    def test_formal_path_and_label_free_inputs_are_exactly_pinned(self) -> None:
        output = ROOT / checkpoint.FORMAL_OUTPUT_RELATIVE
        audit = checkpoint.validate_formal_invocation(
            output_directory=output,
            train_schedule_path=PREFLIGHT / "train_balanced_schedule.json",
            joint_signature_path=JOINT_SIGNATURE,
        )
        self.assertEqual(
            audit["status"],
            "PASS_FORMAL_INVOCATION_ONLY_NO_CHECKPOINT_RUN",
        )
        self.assertEqual(
            set(audit["inputs"]), {"train_schedule", "joint_signatures"}
        )
        with self.assertRaisesRegex(
            checkpoint.ResidualCheckpointError, "checkpoint path"
        ):
            checkpoint.validate_formal_invocation(
                output_directory=output.with_name("wrong_checkpoint_path"),
                train_schedule_path=PREFLIGHT / "train_balanced_schedule.json",
                joint_signature_path=JOINT_SIGNATURE,
            )

    def test_publisher_source_files_are_exact_and_tamper_evident(self) -> None:
        records = checkpoint.expected_source_files(ROOT)
        checkpoint.validate_source_files(records, repository_root=ROOT)
        tampered = json.loads(json.dumps(records))
        tampered["constructor"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            checkpoint.ResidualCheckpointError, "source-file drift"
        ):
            checkpoint.validate_source_files(tampered, repository_root=ROOT)

    def test_state_audit_independently_rebuilds_all_cells(self) -> None:
        search = constructor.JointSearch(
            self.train, self.signatures, split="train"
        )
        cells = search._constraint_cells()
        l1 = sum(
            search._cell_violation(
                int(search.arrays[family][index]), lower, upper
            )
            for family, index, lower, upper in cells
        )
        audit = checkpoint.audit_search_state(
            search,
            expected_l1=l1,
            expected_objective=search.objective,
        )
        self.assertEqual(audit["constraint_cell_count"], 5_324)
        self.assertEqual(audit["valid_world_count"], 500)
        self.assertEqual(audit["l1_bound_violation"], l1)
        self.assertEqual(audit["squared_objective"], search.objective)

    def test_state_audit_rejects_invalid_world_before_rebuild(self) -> None:
        search = constructor.JointSearch(
            self.train, self.signatures, split="train"
        )
        original = search.assignments[0].copy()
        search.assignments[0, 1] = search.assignments[0, 0]
        with self.assertRaisesRegex(
            checkpoint.ResidualCheckpointError, "invalid world"
        ):
            checkpoint.audit_search_state(
                search,
                expected_l1=0,
                expected_objective=0,
            )
        self.assertFalse((search.assignments[0] == original).all())

    def test_state_audit_rejects_violated_cell_cardinality_drift(self) -> None:
        search = constructor.JointSearch(
            self.train, self.signatures, split="train"
        )
        cells = search._constraint_cells()
        l1 = sum(
            search._cell_violation(
                int(search.arrays[family][index]), lower, upper
            )
            for family, index, lower, upper in cells
        )
        violated_count = sum(
            not lower <= int(search.arrays[family][index]) <= upper
            for family, index, lower, upper in cells
        )
        with self.assertRaisesRegex(
            checkpoint.ResidualCheckpointError,
            "violated-cell cardinality drift",
        ):
            checkpoint.audit_search_state(
                search,
                expected_l1=l1,
                expected_objective=search.objective,
                expected_violated_cell_count=violated_count + 1,
            )


if __name__ == "__main__":
    unittest.main()
