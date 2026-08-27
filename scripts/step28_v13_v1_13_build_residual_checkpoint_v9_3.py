#!/usr/bin/env python3
"""Build the one reusable label-free V9.3 train residual checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

import step28_v13_v1_13_balanced_schedule_v9_3 as balanced
import step28_v13_v1_13_build_joint_noise_signatures_v9_3 as signatures
import step28_v13_v1_13_construct_registered_negative_plan_v9_3 as constructor
import step28_v13_v1_13_registered_negative_plan_v9_3 as plan_contract


VERSION = "2026-08-27-step28-v13-v1-13-train-residual-checkpoint-v1"
FORMAL_OUTPUT_RELATIVE = Path(
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/"
    "train_residual_checkpoint_v1_20260827"
)
FORMAL_INPUT_PINS = {
    "train_schedule": (
        Path(
            "reports/step28_v13_v1_13_balanced_schedule_v9_3/"
            "design_preflight_v2_20260825/train_balanced_schedule.json"
        ),
        "c4a058fb97e351b033d2f4c595985251cf790b1693123bdec01fc64bcc18a4c2",
    ),
    "joint_signatures": (
        Path(
            "reports/step28_v13_v1_13_balanced_schedule_v9_3/"
            "joint_noise_signature_preflight_v2_20260826.json"
        ),
        "05caa6a2b19591fdbf042384b1d31dccfde4a3f1096275ec44671f9f4b3c4d37",
    ),
}
EXPECTED_SPLIT = "train"
EXPECTED_OBJECTIVE = 20
EXPECTED_CELL_COUNT = 5_324
EXPECTED_VIOLATED_CELL_COUNT = 20


class ResidualCheckpointError(RuntimeError):
    """Raised when the reusable abstract checkpoint cannot be published."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path, repository_root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(repository_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def expected_source_files(repository_root: Path) -> dict[str, dict[str, Any]]:
    """Return the exact source bytes that are allowed to publish a checkpoint."""

    return {
        "builder": _file_record(Path(__file__), repository_root),
        "constructor": _file_record(Path(constructor.__file__), repository_root),
    }


def validate_source_files(
    supplied: Mapping[str, Mapping[str, Any]], *, repository_root: Path
) -> None:
    if dict(supplied) != expected_source_files(repository_root):
        raise ResidualCheckpointError("Residual checkpoint source-file drift")


def validate_formal_invocation(
    *,
    output_directory: Path,
    train_schedule_path: Path,
    joint_signature_path: Path,
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    if output_directory.resolve() != (
        repository_root / FORMAL_OUTPUT_RELATIVE
    ).resolve():
        raise ResidualCheckpointError(
            "Only the fresh V9.3 train residual checkpoint path is authorized"
        )
    supplied = {
        "train_schedule": train_schedule_path,
        "joint_signatures": joint_signature_path,
    }
    observed: dict[str, dict[str, Any]] = {}
    for name, path in supplied.items():
        relative, expected_sha256 = FORMAL_INPUT_PINS[name]
        expected_path = (repository_root / relative).resolve()
        if path.resolve() != expected_path:
            raise ResidualCheckpointError(f"Checkpoint input path drift: {name}")
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise ResidualCheckpointError(f"Checkpoint input payload drift: {name}")
        observed[name] = _file_record(path, repository_root)
    return {
        "status": "PASS_FORMAL_INVOCATION_ONLY_NO_CHECKPOINT_RUN",
        "output_path": FORMAL_OUTPUT_RELATIVE.as_posix(),
        "inputs": observed,
    }


def audit_search_state(
    search: constructor.JointSearch,
    *,
    expected_l1: int,
    expected_objective: int,
    expected_violated_cell_count: int | None = None,
) -> dict[str, Any]:
    assignments = np.asarray(search.assignments)
    if assignments.shape != (balanced.WORLD_COUNT, 12):
        raise ResidualCheckpointError("Residual assignment shape drift")
    if assignments.dtype.kind not in {"i", "u"}:
        raise ResidualCheckpointError("Residual assignments are not integral")
    if np.any(assignments < 0) or np.any(assignments >= 28):
        raise ResidualCheckpointError("Residual assignments left the seller domain")
    invalid_worlds = [
        world
        for world, row in enumerate(assignments)
        if not search._valid_row(world, row)
    ]
    if invalid_worlds:
        raise ResidualCheckpointError("Residual checkpoint contains an invalid world")
    search._rebuild_all_counts()
    cells = search._constraint_cells()
    if len(cells) != EXPECTED_CELL_COUNT:
        raise ResidualCheckpointError("Residual checkpoint cell-count drift")
    constraint_cells = []
    violated_cells = []
    l1_bound_violation = 0
    family_l1_bound_violation: dict[str, int] = {
        family: 0 for family in search.arrays
    }
    for ordinal, (family, index, lower, upper) in enumerate(cells):
        current = int(search.arrays[family][index])
        violation = search._cell_violation(current, lower, upper)
        signed_deviation = (
            current - lower
            if current < lower
            else current - upper
            if current > upper
            else 0
        )
        if abs(signed_deviation) != violation:
            raise ResidualCheckpointError(
                "Residual checkpoint signed-deviation accounting drift"
            )
        l1_bound_violation += violation
        family_l1_bound_violation[family] += violation
        cell_record = {
            "cell_ordinal": ordinal,
            "family": family,
            "index": list(index),
            "current": current,
            "lower": lower,
            "upper": upper,
            "signed_deviation": signed_deviation,
            "absolute_violation": violation,
        }
        constraint_cells.append(cell_record)
        if violation:
            violated_cells.append(cell_record)
    full_objective = search._full_objective()
    if (
        search.objective != full_objective
        or full_objective != expected_objective
        or l1_bound_violation != expected_l1
    ):
        raise ResidualCheckpointError(
            "Residual checkpoint objective did not close to its frozen value"
        )
    if (
        expected_violated_cell_count is not None
        and len(violated_cells) != expected_violated_cell_count
    ):
        raise ResidualCheckpointError(
            "Residual checkpoint violated-cell cardinality drift"
        )
    return {
        "assignment_shape": list(assignments.shape),
        "assignment_minimum": int(assignments.min()),
        "assignment_maximum": int(assignments.max()),
        "valid_world_count": balanced.WORLD_COUNT,
        "constraint_cell_count": len(cells),
        "violated_cell_count": len(violated_cells),
        "l1_bound_violation": l1_bound_violation,
        "squared_objective": full_objective,
        "objective_breakdown": search._objective_breakdown(),
        "family_l1_bound_violation": family_l1_bound_violation,
        "constraint_cells_sha256": plan_contract.canonical_sha256(
            constraint_cells
        ),
        "constraint_cells": constraint_cells,
        "violated_cells": violated_cells,
    }


def validate_checkpoint(
    payload: Mapping[str, Any],
    *,
    train_schedule: Mapping[str, Any],
    joint_signatures: Mapping[str, Any],
    expected_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required = {
        "version",
        "status",
        "split",
        "inputs",
        "source_files",
        "search_receipt",
        "assignments",
        "state_audit",
        "canonical_self_sha256",
    }
    if set(payload) != required:
        raise ResidualCheckpointError("Residual checkpoint schema drift")
    if (
        payload["version"] != VERSION
        or payload["status"]
        != "PASS_LABEL_FREE_ABSTRACT_RESIDUAL_CHECKPOINT_NOT_A_PLAN_OR_DATASET"
        or payload["split"] != EXPECTED_SPLIT
        or payload["inputs"] != expected_inputs
    ):
        raise ResidualCheckpointError("Residual checkpoint provenance drift")
    repository_root = Path(__file__).resolve().parents[1]
    validate_source_files(
        payload["source_files"], repository_root=repository_root
    )
    if payload["canonical_self_sha256"] != plan_contract.canonical_self_sha256(
        dict(payload)
    ):
        raise ResidualCheckpointError("Residual checkpoint self-hash drift")
    assignments = np.asarray(payload["assignments"], dtype=np.int16)
    replay = constructor.JointSearch(
        train_schedule, joint_signatures, split=EXPECTED_SPLIT
    )
    if assignments.shape != replay.assignments.shape:
        raise ResidualCheckpointError("Residual checkpoint replay shape drift")
    replay.assignments = assignments.astype(np.int8, copy=True)
    observed_audit = audit_search_state(
        replay,
        expected_l1=EXPECTED_OBJECTIVE,
        expected_objective=EXPECTED_OBJECTIVE,
        expected_violated_cell_count=EXPECTED_VIOLATED_CELL_COUNT,
    )
    if payload["state_audit"] != observed_audit:
        raise ResidualCheckpointError("Residual checkpoint state audit drift")
    return {
        "status": "PASS_INDEPENDENT_FULL_RESIDUAL_CHECKPOINT_REPLAY",
        "assignment_sha256": hashlib.sha256(assignments.tobytes()).hexdigest(),
        "constraint_cell_count": observed_audit["constraint_cell_count"],
        "l1_bound_violation": observed_audit["l1_bound_violation"],
        "squared_objective": observed_audit["squared_objective"],
    }


def publish(
    *,
    output_directory: Path,
    train_schedule_path: Path,
    joint_signature_path: Path,
) -> dict[str, Any]:
    invocation = validate_formal_invocation(
        output_directory=output_directory,
        train_schedule_path=train_schedule_path,
        joint_signature_path=joint_signature_path,
    )
    if output_directory.exists():
        raise ResidualCheckpointError("Residual checkpoint output already exists")
    building = output_directory.with_name(output_directory.name + ".building")
    if building.exists():
        raise ResidualCheckpointError("Stale residual checkpoint building path exists")
    train_schedule = plan_contract.load_json(train_schedule_path)
    joint_signatures = plan_contract.load_json(joint_signature_path)
    balanced.validate_schedule(train_schedule)
    signatures.validate_payload(joint_signatures)
    search = constructor.JointSearch(
        train_schedule, joint_signatures, split=EXPECTED_SPLIT
    )
    search_receipt = search.run(stop_at_residual_checkpoint=True)
    state_audit = audit_search_state(
        search,
        expected_l1=EXPECTED_OBJECTIVE,
        expected_objective=EXPECTED_OBJECTIVE,
        expected_violated_cell_count=EXPECTED_VIOLATED_CELL_COUNT,
    )
    repository_root = Path(__file__).resolve().parents[1]
    source_files = expected_source_files(repository_root)
    payload: dict[str, Any] = {
        "version": VERSION,
        "status": (
            "PASS_LABEL_FREE_ABSTRACT_RESIDUAL_CHECKPOINT_NOT_A_PLAN_OR_DATASET"
        ),
        "split": EXPECTED_SPLIT,
        "inputs": invocation["inputs"],
        "source_files": source_files,
        "search_receipt": search_receipt,
        "assignments": search.assignments.astype(int).tolist(),
        "state_audit": state_audit,
    }
    payload["canonical_self_sha256"] = plan_contract.canonical_self_sha256(
        payload
    )
    replay_audit = validate_checkpoint(
        payload,
        train_schedule=train_schedule,
        joint_signatures=joint_signatures,
        expected_inputs=invocation["inputs"],
    )
    try:
        building.mkdir(parents=True, exist_ok=False)
        checkpoint_path = building / "train_residual_checkpoint.json"
        with checkpoint_path.open("xb") as stream:
            stream.write(plan_contract.canonical_json_bytes(payload))
            stream.write(b"\n")
        reopened_payload = plan_contract.load_json(checkpoint_path)
        if reopened_payload != payload:
            raise ResidualCheckpointError(
                "Residual checkpoint changed during persistence"
            )
        replay_audit = validate_checkpoint(
            reopened_payload,
            train_schedule=train_schedule,
            joint_signatures=joint_signatures,
            expected_inputs=invocation["inputs"],
        )
        receipt: dict[str, Any] = {
            "version": VERSION,
            "status": "PASS_RESIDUAL_CHECKPOINT_PUBLICATION_ONLY",
            "checkpoint_file": {
                "path": "train_residual_checkpoint.json",
                "size_bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
            },
            "replay_audit": replay_audit,
        }
        receipt["canonical_self_sha256"] = plan_contract.canonical_self_sha256(
            receipt
        )
        with (building / "checkpoint_receipt.json").open("xb") as stream:
            stream.write(plan_contract.canonical_json_bytes(receipt))
            stream.write(b"\n")
        building.replace(output_directory)
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--train-schedule", required=True, type=Path)
    parser.add_argument("--joint-signatures", required=True, type=Path)
    args = parser.parse_args()
    receipt = publish(
        output_directory=args.output_directory,
        train_schedule_path=args.train_schedule,
        joint_signature_path=args.joint_signatures,
    )
    print(plan_contract.canonical_json_bytes(receipt).decode("utf-8"))


if __name__ == "__main__":
    main()
