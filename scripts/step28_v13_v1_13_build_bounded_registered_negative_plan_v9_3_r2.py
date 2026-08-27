#!/usr/bin/env python3
"""Publish the one V9.3-R2 user-accepted-residual plan pair."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

import step28_v13_v1_13_balanced_schedule_v9_3 as balanced
import step28_v13_v1_13_build_joint_noise_signatures_v9_3 as signatures
import step28_v13_v1_13_build_residual_checkpoint_v9_3 as checkpoint
import step28_v13_v1_13_construct_registered_negative_plan_v9_3 as constructor
import step28_v13_v1_13_registered_negative_plan_v9_3 as plan_contract


VERSION = (
    "2026-08-27-step28-v13-v1-13-build-registered-negative-plan-"
    "v9-3-r2-user-accepted-residual-22"
)
PLAN_VERSION = plan_contract.BOUNDED_RESIDUAL_VERSION
PLAN_STATUS = plan_contract.BOUNDED_RESIDUAL_STATUS
PAIR_STATUS = (
    "PASS_INDEPENDENT_BOUNDED_RESIDUAL_SPLIT_PLAN_PAIR_PENDING_STRUCTURE_GATE"
)
FORMAL_OUTPUT_RELATIVE = Path(
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/"
    "registered_negative_bounded_preflight_r2_20260827"
)
FORMAL_INPUT_PINS = constructor.FORMAL_INPUT_PINS
EXPECTED_CELL_COUNT = 5_324
TRAIN_EXPECTED_L1 = 20
TRAIN_EXPECTED_OBJECTIVE = 20
TRAIN_EXPECTED_VIOLATED_CELL_COUNT = 20
DEVELOPMENT_EXPECTED_L1 = 22
DEVELOPMENT_EXPECTED_OBJECTIVE = 22
DEVELOPMENT_EXPECTED_VIOLATED_CELL_COUNT = 22
MAXIMUM_L1_BOUND_VIOLATION = constructor.LOCAL_REPAIR_COARSE_MILP_THRESHOLD
MAXIMUM_SQUARED_OBJECTIVE = constructor.LOCAL_REPAIR_COARSE_MILP_THRESHOLD


class BoundedPlanBuildError(RuntimeError):
    """Raised when the one V9.3-R2 plan publication cannot close."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path, repository_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(repository_root.resolve()).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def expected_source_files(repository_root: Path) -> dict[str, dict[str, Any]]:
    return {
        "builder": _file_record(Path(__file__), repository_root),
        "checkpoint_auditor": _file_record(
            Path(checkpoint.__file__), repository_root
        ),
        "constructor": _file_record(Path(constructor.__file__), repository_root),
        "plan_validator": _file_record(
            Path(plan_contract.__file__), repository_root
        ),
    }


def validate_source_files(
    supplied: Mapping[str, Mapping[str, Any]], *, repository_root: Path
) -> None:
    if dict(supplied) != expected_source_files(repository_root):
        raise BoundedPlanBuildError("V9.3-R2 plan source-file drift")


def validate_formal_invocation(
    *,
    output_directory: Path,
    train_schedule_path: Path,
    development_schedule_path: Path,
    joint_signature_path: Path,
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    if output_directory.resolve() != (
        repository_root / FORMAL_OUTPUT_RELATIVE
    ).resolve():
        raise BoundedPlanBuildError("V9.3-R2 formal output path drift")
    supplied_paths = {
        "train_schedule": train_schedule_path,
        "development_schedule": development_schedule_path,
        "joint_signatures": joint_signature_path,
    }
    observed: dict[str, dict[str, Any]] = {}
    for name, path in supplied_paths.items():
        relative, expected_sha256 = FORMAL_INPUT_PINS[name]
        expected_path = (repository_root / relative).resolve()
        if path.resolve() != expected_path:
            raise BoundedPlanBuildError(f"V9.3-R2 input path drift: {name}")
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise BoundedPlanBuildError(f"V9.3-R2 input payload drift: {name}")
        observed[name] = _file_record(path, repository_root)
    return {
        "status": "PASS_V9_3_R2_FORMAL_INVOCATION_ONLY_NO_PLAN_RUN",
        "output_path": FORMAL_OUTPUT_RELATIVE.as_posix(),
        "inputs": observed,
    }


def assignments_from_plan(plan: Mapping[str, Any]) -> np.ndarray:
    worlds = plan.get("worlds")
    if not isinstance(worlds, list) or len(worlds) != balanced.WORLD_COUNT:
        raise BoundedPlanBuildError("V9.3-R2 plan world cardinality drift")
    assignments = np.empty((balanced.WORLD_COUNT, 12), dtype=np.int8)
    for expected_world, world in enumerate(worlds):
        if not isinstance(world, Mapping) or world.get("world_ordinal") != expected_world:
            raise BoundedPlanBuildError("V9.3-R2 plan world order drift")
        raw_assignments = world.get("assignments")
        if not isinstance(raw_assignments, list) or len(raw_assignments) != 6:
            raise BoundedPlanBuildError("V9.3-R2 plan assignment cardinality drift")
        row: list[int] = []
        for assignment in raw_assignments:
            endpoints = assignment.get("endpoints") if isinstance(assignment, Mapping) else None
            if not isinstance(endpoints, list) or len(endpoints) != 2:
                raise BoundedPlanBuildError("V9.3-R2 endpoint cardinality drift")
            for endpoint in endpoints:
                seller_slot = endpoint.get("seller_slot") if isinstance(endpoint, Mapping) else None
                if type(seller_slot) is not int or not 0 <= seller_slot < 28:
                    raise BoundedPlanBuildError("V9.3-R2 endpoint seller-slot drift")
                row.append(seller_slot)
        if len(row) != 12:
            raise BoundedPlanBuildError("V9.3-R2 endpoint row width drift")
        assignments[expected_world] = np.asarray(row, dtype=np.int8)
    return assignments


def _raw_residual_totals(search: constructor.JointSearch) -> tuple[int, int, int]:
    search._rebuild_all_counts()
    cells = search._constraint_cells()
    if len(cells) != EXPECTED_CELL_COUNT:
        raise BoundedPlanBuildError("V9.3-R2 constraint-cell cardinality drift")
    violations = [
        search._cell_violation(
            int(search.arrays[family][index]), lower, upper
        )
        for family, index, lower, upper in cells
    ]
    return sum(violations), search.objective, sum(value > 0 for value in violations)


def audit_bounded_state(
    search: constructor.JointSearch,
    *,
    split: str,
    expected_l1: int | None = None,
    expected_objective: int | None = None,
    expected_violated_cell_count: int | None = None,
) -> dict[str, Any]:
    l1, objective, violated_count = _raw_residual_totals(search)
    if expected_l1 is not None and l1 != expected_l1:
        raise BoundedPlanBuildError(f"{split} frozen L1 residual drift")
    if expected_objective is not None and objective != expected_objective:
        raise BoundedPlanBuildError(f"{split} frozen squared residual drift")
    if (
        expected_violated_cell_count is not None
        and violated_count != expected_violated_cell_count
    ):
        raise BoundedPlanBuildError(f"{split} frozen nonzero-cell count drift")
    if (
        l1 > MAXIMUM_L1_BOUND_VIOLATION
        or objective > MAXIMUM_SQUARED_OBJECTIVE
    ):
        raise BoundedPlanBuildError(
            f"{split} did not stop inside the frozen coarse-repair boundary"
        )
    audit = checkpoint.audit_search_state(
        search,
        expected_l1=l1,
        expected_objective=objective,
        expected_violated_cell_count=violated_count,
    )
    if (
        audit["constraint_cell_count"] != EXPECTED_CELL_COUNT
        or audit["l1_bound_violation"] != l1
        or audit["squared_objective"] != objective
        or audit["violated_cell_count"] != violated_count
    ):
        raise BoundedPlanBuildError(f"{split} residual disclosure drift")
    return audit


def validate_split_bundle(
    *,
    plan: Mapping[str, Any],
    disclosure: Mapping[str, Any],
    schedule: Mapping[str, Any],
    joint_signatures: Mapping[str, Any],
    expected_inputs: Mapping[str, Mapping[str, Any]],
    expected_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    split = str(plan.get("split"))
    if split not in ("train", "development"):
        raise BoundedPlanBuildError("V9.3-R2 split bundle drift")
    plan_audit = plan_contract.validate_plan(
        plan,
        schedule,
        joint_signatures,
        expected_version=PLAN_VERSION,
        require_exact_balance=False,
        success_status=PLAN_STATUS,
    )
    required_disclosure = {
        "version",
        "status",
        "split",
        "inputs",
        "source_files",
        "public_design_seed",
        "plan_canonical_self_sha256",
        "search_receipt",
        "role_eligibility_authority",
        "state_audit",
        "canonical_self_sha256",
    }
    if set(disclosure) != required_disclosure:
        raise BoundedPlanBuildError(f"{split} disclosure schema drift")
    if (
        disclosure["version"] != VERSION
        or disclosure["status"] != PLAN_STATUS
        or disclosure["split"] != split
        or disclosure["inputs"] != expected_inputs
        or disclosure["source_files"] != expected_sources
        or disclosure["public_design_seed"]
        != constructor.PUBLIC_DESIGN_SEEDS[split]
        or disclosure["plan_canonical_self_sha256"]
        != plan["canonical_self_sha256"]
        or disclosure["canonical_self_sha256"]
        != plan_contract.canonical_self_sha256(dict(disclosure))
    ):
        raise BoundedPlanBuildError(f"{split} disclosure provenance drift")
    repository_root = Path(__file__).resolve().parents[1]
    validate_source_files(disclosure["source_files"], repository_root=repository_root)
    assignments = assignments_from_plan(plan)
    replay = constructor.JointSearch(schedule, joint_signatures, split=split)
    replay.assignments = assignments.copy()
    observed_audit = audit_bounded_state(
        replay,
        split=split,
        expected_l1=disclosure["state_audit"]["l1_bound_violation"],
        expected_objective=disclosure["state_audit"]["squared_objective"],
        expected_violated_cell_count=disclosure["state_audit"][
            "violated_cell_count"
        ],
    )
    if disclosure["state_audit"] != observed_audit:
        raise BoundedPlanBuildError(f"{split} full 5,324-cell replay drift")
    role_authority = disclosure["role_eligibility_authority"]
    expected_role_authority = {
        "shape": list(replay.role_eligible.shape),
        "predicate_names": list(
            constructor.ROLE_ELIGIBILITY_PREDICATE_NAMES
        ),
        "true_count": int(replay.role_eligible.sum()),
        "sha256": hashlib.sha256(replay.role_eligible.tobytes()).hexdigest(),
    }
    if role_authority != expected_role_authority:
        raise BoundedPlanBuildError(f"{split} role-eligibility authority drift")
    return {
        "split": split,
        "plan_audit_sha256": plan_contract.canonical_sha256(plan_audit),
        "assignment_sha256": hashlib.sha256(assignments.tobytes()).hexdigest(),
        "constraint_cells_sha256": observed_audit["constraint_cells_sha256"],
        "l1_bound_violation": observed_audit["l1_bound_violation"],
        "squared_objective": observed_audit["squared_objective"],
        "violated_cell_count": observed_audit["violated_cell_count"],
        "status": "PASS_INDEPENDENT_FULL_BOUNDED_RESIDUAL_REPLAY",
    }


def build_split(
    *,
    split: str,
    schedule: Mapping[str, Any],
    joint_signatures: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
    source_files: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print(
        plan_contract.canonical_json_bytes(
            {"event": "v9_3_r2_split_search_started", "split": split}
        ).decode("utf-8"),
        flush=True,
    )
    search = constructor.JointSearch(schedule, joint_signatures, split=split)
    search_receipt = search.run(stop_at_residual_checkpoint=True)
    state_audit = audit_bounded_state(
        search,
        split=split,
        expected_l1=(
            TRAIN_EXPECTED_L1
            if split == "train"
            else DEVELOPMENT_EXPECTED_L1
        ),
        expected_objective=(
            TRAIN_EXPECTED_OBJECTIVE
            if split == "train"
            else DEVELOPMENT_EXPECTED_OBJECTIVE
        ),
        expected_violated_cell_count=(
            TRAIN_EXPECTED_VIOLATED_CELL_COUNT
            if split == "train"
            else DEVELOPMENT_EXPECTED_VIOLATED_CELL_COUNT
        ),
    )
    plan = constructor.materialize_plan(
        split=split,
        assignments=search.assignments,
        schedule=schedule,
        joint_signatures=joint_signatures,
        plan_version=PLAN_VERSION,
    )
    plan_contract.validate_plan(
        plan,
        schedule,
        joint_signatures,
        expected_version=PLAN_VERSION,
        require_exact_balance=False,
        success_status=PLAN_STATUS,
    )
    disclosure: dict[str, Any] = {
        "version": VERSION,
        "status": PLAN_STATUS,
        "split": split,
        "inputs": dict(inputs),
        "source_files": dict(source_files),
        "public_design_seed": constructor.PUBLIC_DESIGN_SEEDS[split],
        "plan_canonical_self_sha256": plan["canonical_self_sha256"],
        "search_receipt": search_receipt,
        "role_eligibility_authority": {
            "shape": list(search.role_eligible.shape),
            "predicate_names": list(
                constructor.ROLE_ELIGIBILITY_PREDICATE_NAMES
            ),
            "true_count": int(search.role_eligible.sum()),
            "sha256": hashlib.sha256(search.role_eligible.tobytes()).hexdigest(),
        },
        "state_audit": state_audit,
    }
    disclosure["canonical_self_sha256"] = plan_contract.canonical_self_sha256(
        disclosure
    )
    replay = validate_split_bundle(
        plan=plan,
        disclosure=disclosure,
        schedule=schedule,
        joint_signatures=joint_signatures,
        expected_inputs=inputs,
        expected_sources=source_files,
    )
    print(
        plan_contract.canonical_json_bytes(
            {
                "event": "v9_3_r2_split_search_complete",
                "split": split,
                "l1_bound_violation": replay["l1_bound_violation"],
                "squared_objective": replay["squared_objective"],
                "violated_cell_count": replay["violated_cell_count"],
            }
        ).decode("utf-8"),
        flush=True,
    )
    return plan, disclosure, replay


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(plan_contract.canonical_json_bytes(payload))
        stream.write(b"\n")


def publish(
    *,
    output_directory: Path,
    train_schedule_path: Path,
    development_schedule_path: Path,
    joint_signature_path: Path,
) -> dict[str, Any]:
    invocation = validate_formal_invocation(
        output_directory=output_directory,
        train_schedule_path=train_schedule_path,
        development_schedule_path=development_schedule_path,
        joint_signature_path=joint_signature_path,
    )
    if output_directory.exists():
        raise BoundedPlanBuildError("V9.3-R2 output already exists")
    building = output_directory.with_name(output_directory.name + ".building")
    if building.exists():
        raise BoundedPlanBuildError("Stale V9.3-R2 building directory exists")
    train_schedule = plan_contract.load_json(train_schedule_path)
    development_schedule = plan_contract.load_json(development_schedule_path)
    joint_signatures = plan_contract.load_json(joint_signature_path)
    balanced.validate_train_development_pair(train_schedule, development_schedule)
    signatures.validate_payload(joint_signatures)
    repository_root = Path(__file__).resolve().parents[1]
    source_files = expected_source_files(repository_root)
    train_plan, train_disclosure, train_replay = build_split(
        split="train",
        schedule=train_schedule,
        joint_signatures=joint_signatures,
        inputs=invocation["inputs"],
        source_files=source_files,
    )
    development_plan, development_disclosure, development_replay = build_split(
        split="development",
        schedule=development_schedule,
        joint_signatures=joint_signatures,
        inputs=invocation["inputs"],
        source_files=source_files,
    )
    pair_audit = plan_contract.validate_train_development_plan_pair(
        train_plan,
        development_plan,
        train_schedule,
        development_schedule,
        joint_signatures,
        expected_version=PLAN_VERSION,
        require_exact_balance=False,
        plan_success_status=PLAN_STATUS,
        pair_success_status=PAIR_STATUS,
    )
    closing_invocation = validate_formal_invocation(
        output_directory=output_directory,
        train_schedule_path=train_schedule_path,
        development_schedule_path=development_schedule_path,
        joint_signature_path=joint_signature_path,
    )
    if closing_invocation != invocation:
        raise BoundedPlanBuildError("V9.3-R2 input provenance changed during search")
    validate_source_files(source_files, repository_root=repository_root)
    payloads: dict[str, Mapping[str, Any]] = {
        "train_registered_negative_plan.json": train_plan,
        "development_registered_negative_plan.json": development_plan,
        "train_residual_disclosure.json": train_disclosure,
        "development_residual_disclosure.json": development_disclosure,
    }
    try:
        building.mkdir(parents=True, exist_ok=False)
        for name, payload in payloads.items():
            _write_new_json(building / name, payload)
        reopened = {
            name: plan_contract.load_json(building / name)
            for name in sorted(payloads)
        }
        if any(reopened[name] != payloads[name] for name in payloads):
            raise BoundedPlanBuildError("V9.3-R2 payload changed during persistence")
        persisted_train_replay = validate_split_bundle(
            plan=reopened["train_registered_negative_plan.json"],
            disclosure=reopened["train_residual_disclosure.json"],
            schedule=train_schedule,
            joint_signatures=joint_signatures,
            expected_inputs=invocation["inputs"],
            expected_sources=source_files,
        )
        persisted_development_replay = validate_split_bundle(
            plan=reopened["development_registered_negative_plan.json"],
            disclosure=reopened["development_residual_disclosure.json"],
            schedule=development_schedule,
            joint_signatures=joint_signatures,
            expected_inputs=invocation["inputs"],
            expected_sources=source_files,
        )
        if (
            persisted_train_replay != train_replay
            or persisted_development_replay != development_replay
        ):
            raise BoundedPlanBuildError("V9.3-R2 persisted replay changed")
        persisted_pair_audit = plan_contract.validate_train_development_plan_pair(
            reopened["train_registered_negative_plan.json"],
            reopened["development_registered_negative_plan.json"],
            train_schedule,
            development_schedule,
            joint_signatures,
            expected_version=PLAN_VERSION,
            require_exact_balance=False,
            plan_success_status=PLAN_STATUS,
            pair_success_status=PAIR_STATUS,
        )
        if persisted_pair_audit != pair_audit:
            raise BoundedPlanBuildError("V9.3-R2 split-pair replay changed")
        receipt: dict[str, Any] = {
            "version": VERSION,
            "status": PLAN_STATUS,
            "inputs": invocation["inputs"],
            "source_files": source_files,
            "public_design_seeds": constructor.PUBLIC_DESIGN_SEEDS,
            "train_replay": train_replay,
            "development_replay": development_replay,
            "plan_pair_audit": pair_audit,
            "published_files": {
                name: {
                    "size_bytes": (building / name).stat().st_size,
                    "sha256": _sha256(building / name),
                }
                for name in sorted(payloads)
            },
        }
        receipt["canonical_self_sha256"] = plan_contract.canonical_self_sha256(
            receipt
        )
        _write_new_json(building / "construction_receipt.json", receipt)
        reopened_receipt = plan_contract.load_json(
            building / "construction_receipt.json"
        )
        if (
            reopened_receipt != receipt
            or reopened_receipt["canonical_self_sha256"]
            != plan_contract.canonical_self_sha256(reopened_receipt)
        ):
            raise BoundedPlanBuildError("V9.3-R2 receipt persistence drift")
        building.replace(output_directory)
        expected_names = {*payloads, "construction_receipt.json"}
        observed_names = {
            path.name for path in output_directory.iterdir() if path.is_file()
        }
        if observed_names != expected_names or any(
            path.is_dir() for path in output_directory.iterdir()
        ):
            raise BoundedPlanBuildError("V9.3-R2 final file-set drift")
        final_reopened = {
            name: plan_contract.load_json(output_directory / name)
            for name in sorted(expected_names)
        }
        if final_reopened["construction_receipt.json"] != receipt:
            raise BoundedPlanBuildError("V9.3-R2 final receipt drift")
        validate_split_bundle(
            plan=final_reopened["train_registered_negative_plan.json"],
            disclosure=final_reopened["train_residual_disclosure.json"],
            schedule=train_schedule,
            joint_signatures=joint_signatures,
            expected_inputs=invocation["inputs"],
            expected_sources=source_files,
        )
        validate_split_bundle(
            plan=final_reopened["development_registered_negative_plan.json"],
            disclosure=final_reopened["development_residual_disclosure.json"],
            schedule=development_schedule,
            joint_signatures=joint_signatures,
            expected_inputs=invocation["inputs"],
            expected_sources=source_files,
        )
        final_invocation = validate_formal_invocation(
            output_directory=output_directory,
            train_schedule_path=train_schedule_path,
            development_schedule_path=development_schedule_path,
            joint_signature_path=joint_signature_path,
        )
        if final_invocation != invocation:
            raise BoundedPlanBuildError("V9.3-R2 final input provenance drift")
        validate_source_files(source_files, repository_root=repository_root)
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        if output_directory.exists():
            shutil.rmtree(output_directory)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--train-schedule", required=True, type=Path)
    parser.add_argument("--development-schedule", required=True, type=Path)
    parser.add_argument("--joint-signatures", required=True, type=Path)
    args = parser.parse_args()
    receipt = publish(
        output_directory=args.output_directory,
        train_schedule_path=args.train_schedule,
        development_schedule_path=args.development_schedule,
        joint_signature_path=args.joint_signatures,
    )
    print(plan_contract.canonical_json_bytes(receipt).decode("utf-8"))


if __name__ == "__main__":
    main()
