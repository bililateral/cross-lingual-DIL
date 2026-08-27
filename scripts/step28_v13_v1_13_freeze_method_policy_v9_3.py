#!/usr/bin/env python3
"""Freeze exact V9.3 input and implementation file pins into the method policy."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import step28_v13_common as common
import step28_v13_v1_13_method_policy_v9_3 as method_policy


PIN_PATHS = (
    "schema/step28_v13_synthetic_chinese_dataset_policy.json",
    "schema/step28_v13_synthetic_text_templates.json",
    "scripts/step28_v13_common.py",
    "scripts/step28_v13_history_features.py",
    "scripts/step28_v13_identity_plan.py",
    "scripts/step28_v13_nonidentity.py",
    "scripts/step28_v13_production_chain.py",
    "scripts/step28_v13_profiles.py",
    "scripts/step28_v13_structure.py",
    "scripts/step28_v13_text_renderer.py",
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/design_preflight_v2_20260825/train_balanced_schedule.json",
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/design_preflight_v2_20260825/development_balanced_schedule.json",
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/joint_noise_signature_preflight_v2_20260826.json",
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/blind_audit_design_preflight_v1_20260826.json",
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/registered_negative_bounded_preflight_r2_20260827/construction_receipt.json",
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/registered_negative_bounded_preflight_r2_20260827/development_registered_negative_plan.json",
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/registered_negative_bounded_preflight_r2_20260827/development_residual_disclosure.json",
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/registered_negative_bounded_preflight_r2_20260827/train_registered_negative_plan.json",
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/registered_negative_bounded_preflight_r2_20260827/train_residual_disclosure.json",
    "scripts/step28_v13_v1_13_audit_design_v9_3.py",
    "scripts/step28_v13_v1_13_balanced_schedule_v9_3.py",
    "scripts/step28_v13_v1_13_build_joint_noise_signatures_v9_3.py",
    "scripts/step28_v13_v1_13_build_bounded_registered_negative_plan_v9_3_r2.py",
    "scripts/step28_v13_v1_13_build_residual_checkpoint_v9_3.py",
    "scripts/step28_v13_v1_13_construct_registered_negative_plan_v9_3.py",
    "scripts/step28_v13_v1_13_counterfactual_text_v9_3.py",
    "scripts/step28_v13_v1_13_create_method_random_authority_v9_3.py",
    "scripts/step28_v13_v1_13_freeze_method_policy_v9_3.py",
    "scripts/step28_v13_v1_13_method_dataset_builder_v9_3.py",
    "scripts/step28_v13_v1_13_method_policy_v9_3.py",
    "scripts/step28_v13_v1_13_method_world_v9_3.py",
    "scripts/step28_v13_v1_13_prebuild_structure_gate_v9_3_r2.py",
    "scripts/step28_v13_v1_13_quality_auditor_v9_3.py",
    "scripts/step28_v13_v1_13_quality_probe_core_v9_3.py",
    "scripts/step28_v13_v1_13_quality_probe_validator_v9.py",
    "scripts/step28_v13_v1_13_quality_text_probe_views_v9.py",
    "scripts/step28_v13_v1_13_registered_negative_plan_v9_3.py",
    "scripts/step28_v13_v1_13_scientific_common_v9.py",
    "scripts/step28_v13_v1_13_scientific_dataset_builder_v9.py",
    "scripts/step28_v13_v1_13_structure_matrix_v9_3.py",
    "scripts/step28_v13_v1_13_world_builder_v9_3.py",
    "tests/test_step28_v13_v1_13_method_dataset_builder_v9_3_contracts.py",
    "tests/test_step28_v13_v1_13_quality_auditor_v9_3_contracts.py",
    "tests/test_step28_v13_v1_13_quality_probe_core_v9_3_contracts.py",
    "tests/test_step28_v13_v1_13_quality_text_probe_views_v9_contracts.py",
    "tests/test_step28_v13_v1_13_structure_matrix_v9_3_contracts.py",
    "tests/test_step28_v13_v1_13_world_builder_v9_3_contracts.py",
    "tests/test_step28_v13_v1_13_v9_3_r2_bounded_plan.py",
    "tests/test_step28_v13_v1_13_v9_3_residual_checkpoint.py",
)


def build_pins() -> list[dict[str, Any]]:
    if len(PIN_PATHS) != len(set(PIN_PATHS)):
        raise RuntimeError("V9.3 policy pin path collision")
    rows: list[dict[str, Any]] = []
    for relative in PIN_PATHS:
        path = common.repo_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"V9.3 policy pin input is absent: {relative}")
        row: dict[str, Any] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": common.sha256_file(path),
            "canonical_self_sha256": None,
        }
        if path.suffix == ".json":
            value = common.load_json(path)
            canonical = value.get("canonical_self_sha256")
            if canonical is not None:
                if not isinstance(canonical, str) or len(canonical) != 64:
                    raise RuntimeError(f"Malformed JSON self hash: {relative}")
                row["canonical_self_sha256"] = canonical
        rows.append(row)
    return rows


def refresh(path: Path) -> dict[str, Any]:
    policy = common.load_json(path)
    if policy.get("status") != "PREEXECUTION_IMPLEMENTATION_NOT_TRAINING_QUALIFIED":
        raise RuntimeError("V9.3 method policy is not in the refreshable preexecution state")
    policy["status"] = "FROZEN_METHOD_QUALIFICATION_INPUTS_NOT_TRAINING_QUALIFIED"
    policy["frozen_file_pins"] = build_pins()
    policy["canonical_self_sha256"] = None
    canonical = deepcopy(policy)
    canonical["canonical_self_sha256"] = None
    policy["canonical_self_sha256"] = common.canonical_sha256(canonical)
    payload = (
        json.dumps(policy, ensure_ascii=False, sort_keys=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + ".refreshing")
    if temporary.exists():
        raise RuntimeError("Stale V9.3 policy refreshing file exists")
    try:
        temporary.write_bytes(payload)
        method_policy.validate_policy(common.load_json(temporary))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "status": "V9_3_METHOD_POLICY_FILE_PINS_FROZEN",
        "path": path.relative_to(common.ROOT).as_posix(),
        "canonical_self_sha256": policy["canonical_self_sha256"],
        "file_sha256": common.sha256_file(path),
        "pin_count": len(policy["frozen_file_pins"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", type=Path, default=common.repo_path(method_policy.POLICY_PATH)
    )
    args = parser.parse_args()
    print(json.dumps(refresh(args.policy.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
