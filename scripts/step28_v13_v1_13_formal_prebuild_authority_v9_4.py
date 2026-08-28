#!/usr/bin/env python3
"""Freeze and consume the single V9.4 prebuild-gate authorization."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Callable

import numpy as np

import step28_v13_v1_13_balanced_world_schedule_v9_4 as schedule_v94
import step28_v13_v1_13_joint_noise_signatures_v9_4 as signatures_v94
import step28_v13_v1_13_model_visible_matrix_v9_4 as matrix_v94
import step28_v13_v1_13_model_visible_source_guard_v9_4 as source_guard_v94
import step28_v13_v1_13_quality_probe_core_v9_4 as core_v94
import step28_v13_v1_13_quality_probe_policy_v9_4 as implementation_v94


VERSION = (
    "2026-08-28-step28-v13-v1-13-formal-prebuild-authority-v9-4-attempt-3"
)
POLICY_VERSION = (
    "2026-08-28-step28-v13-v1-13-v9-4-single-prebuild-authorization-attempt-3"
)
ATTEMPT_ID = "step28-v13-v1-13-v9-4-prebuild-gate-attempt-3-20260828"
EXPECTED_BRANCH = "method/step27-english-pretrained-synthetic-adaptation"
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_SCRIPT_RELATIVE = (
    "scripts/step28_v13_v1_13_formal_prebuild_authority_v9_4.py"
)
CONTRACT_RELATIVE = (
    "docs/STEP28_V13_V1_13_V9_4_FORMAL_PREBUILD_AUTHORIZATION_"
    "CONTRACT_20260828.zh.md"
)
IMPLEMENTATION_POLICY_RELATIVE = (
    "schema/step28_v13_v1_13_v9_4_model_visible_shortcut_policy.json"
)
AUTHORIZATION_POLICY_RELATIVE = (
    "schema/step28_v13_v1_13_v9_4_prebuild_gate_authorization_attempt3.json"
)
PRIVATE_ROOT_RELATIVE = (
    "private_custody/step28_v13_v1_13_v9_4_prebuild_gate_"
    "attempt3_20260828"
)
OUTPUT_ROOT_RELATIVE = (
    "reports/step28_synthetic_chinese_dataset/"
    "v9_4_prebuild_gate_attempt3_20260828"
)
UNCONSUMED_KEY_NAME = "time_key.unconsumed.bin"
CONSUMED_KEY_NAME = "time_key.consumed.bin"
ISSUANCE_CLAIM_NAME = "key.issuance.claim.json"
ISSUANCE_RECEIPT_NAME = "key.issuance.json"
ISSUANCE_FAILURE_NAME = "key.issuance.failure.json"
CLAIM_NAME = "attempt.claim.json"
CONSUMPTION_RECEIPT_NAME = "key.consumption.json"
MECHANICAL_FAILURE_RECEIPT_NAME = "mechanical.failure.receipt.json"
PASS_TERMINAL_VALIDATION_PENDING_NAME = (
    "pass.terminal.validation.pending.json"
)
PASS_TERMINAL_VALIDATION_COMPLETION_NAME = (
    "pass.terminal.validation.completed.json"
)
PASS_TERMINAL_VALIDATION_REVALIDATING_NAME = (
    "pass.terminal.validation.completed.revalidating.json"
)
LAUNCH_NAME = "launch_manifest.json"
RESULT_NAME = "prebuild_gate_result.json"
TERMINAL_NAME = "terminal.json"
RESULT_BUILDING_NAME = "prebuild_gate_result.json.building"
UNIQUE_COMMAND = (
    "python -B scripts/step28_v13_v1_13_formal_prebuild_authority_v9_4.py "
    "--run-once"
)
UNIQUE_COMMAND_SHA256 = hashlib.sha256(UNIQUE_COMMAND.encode("utf-8")).hexdigest()

REGISTERED_SCRIPT_RELATIVES = tuple(
    f"scripts/{name}" for name in source_guard_v94.REGISTERED_IMPORTS
)
PINNED_SOURCE_RELATIVES = tuple(sorted(
    {
        AUTHORITY_SCRIPT_RELATIVE,
        CONTRACT_RELATIVE,
        IMPLEMENTATION_POLICY_RELATIVE,
        *REGISTERED_SCRIPT_RELATIVES,
    },
    key=lambda value: value.encode("utf-8"),
))

AUTHORIZATION = {
    "prebuild_shortcut_gate": True,
    "method_root_build": False,
    "audit_truth_unsealing": False,
    "m0_m1_m2_m3": False,
    "train_truth_reads_after_matrix_freeze": 1,
    "development_truth_reads_after_matrix_freeze": 1,
    "audit_a_truth_reads": 0,
    "audit_b_truth_reads": 0,
}


class FormalPrebuildAuthorityV94Error(ValueError):
    """Raised when the V9.4 one-shot authorization contract drifts."""


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    authorization_policy: Path
    private_root: Path
    unconsumed_key: Path
    consumed_key: Path
    issuance_claim: Path
    issuance_receipt: Path
    issuance_failure: Path
    claim: Path
    consumption_receipt: Path
    mechanical_failure_receipt: Path
    pass_terminal_validation_pending: Path
    pass_terminal_validation_completion: Path
    pass_terminal_validation_revalidating: Path
    output_root: Path
    launch: Path
    result: Path
    terminal: Path
    result_building: Path


def runtime_paths(root: Path = ROOT) -> RuntimePaths:
    private_root = root / PRIVATE_ROOT_RELATIVE
    output_root = root / OUTPUT_ROOT_RELATIVE
    return RuntimePaths(
        root=root,
        authorization_policy=root / AUTHORIZATION_POLICY_RELATIVE,
        private_root=private_root,
        unconsumed_key=private_root / UNCONSUMED_KEY_NAME,
        consumed_key=private_root / CONSUMED_KEY_NAME,
        issuance_claim=private_root / ISSUANCE_CLAIM_NAME,
        issuance_receipt=private_root / ISSUANCE_RECEIPT_NAME,
        issuance_failure=private_root / ISSUANCE_FAILURE_NAME,
        claim=private_root / CLAIM_NAME,
        consumption_receipt=private_root / CONSUMPTION_RECEIPT_NAME,
        mechanical_failure_receipt=(
            private_root / MECHANICAL_FAILURE_RECEIPT_NAME
        ),
        pass_terminal_validation_pending=(
            private_root / PASS_TERMINAL_VALIDATION_PENDING_NAME
        ),
        pass_terminal_validation_completion=(
            private_root / PASS_TERMINAL_VALIDATION_COMPLETION_NAME
        ),
        pass_terminal_validation_revalidating=(
            private_root / PASS_TERMINAL_VALIDATION_REVALIDATING_NAME
        ),
        output_root=output_root,
        launch=output_root / LAUNCH_NAME,
        result=output_root / RESULT_NAME,
        terminal=output_root / TERMINAL_NAME,
        result_building=output_root / RESULT_BUILDING_NAME,
    )


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise FormalPrebuildAuthorityV94Error(
        f"Non-JSON public value type: {type(value).__name__}"
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _with_self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "canonical_self_sha256" in payload:
        raise FormalPrebuildAuthorityV94Error("Self-hash field already exists")
    result = _json_value(payload)
    result["canonical_self_sha256"] = _canonical_sha256(result)
    return result


def _verify_self_hash(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, dict) or not _is_sha256(
        payload.get("canonical_self_sha256")
    ):
        raise FormalPrebuildAuthorityV94Error("Canonical self-hash missing")
    without_self = dict(payload)
    observed = without_self.pop("canonical_self_sha256")
    if observed != _canonical_sha256(without_self):
        raise FormalPrebuildAuthorityV94Error("Canonical self-hash drift")


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (
            tuple(left) == tuple(right)
            and all(_exact_json_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    return left == right


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _require_clean_repository(
    root: Path, *, allow_attempt_output: bool = False
) -> None:
    lines = tuple(
        line
        for line in _git(
            root, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line
    )
    if allow_attempt_output:
        prefix = f"?? {OUTPUT_ROOT_RELATIVE}/"
        lines = tuple(line for line in lines if not line.startswith(prefix))
    if lines:
        raise FormalPrebuildAuthorityV94Error("Repository is not clean")


def _collect_source_records(root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for relative in PINNED_SOURCE_RELATIVES:
        path = (root / relative).resolve()
        if path.parent == path or not path.is_file():
            raise FormalPrebuildAuthorityV94Error(
                f"Pinned source is missing: {relative}"
            )
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise FormalPrebuildAuthorityV94Error(
                "Pinned source escaped repository"
            ) from error
        records.append({"path": relative, "sha256": _sha256_file(path)})
    return records


def _runtime_contract() -> dict[str, Any]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": implementation_v94.scipy.__version__,
        "scikit_learn_version": implementation_v94.sklearn.__version__,
        "threadpoolctl_version": implementation_v94.threadpoolctl.__version__,
        "thread_limit": 1,
        "byteorder": sys.byteorder,
    }


def _collect_frozen_input_binding() -> dict[str, Any]:
    noise_signatures = signatures_v94.build_noise_signatures()
    train_schedule = schedule_v94.build_split_schedule("train")
    development_schedule = schedule_v94.build_split_schedule("development")
    pair_receipt = schedule_v94.validate_train_development_pair(
        train_schedule,
        development_schedule,
    )
    return {
        "balanced_schedule_version": schedule_v94.VERSION,
        "train_public_design_seed": schedule_v94.PUBLIC_DESIGN_SEEDS["train"],
        "development_public_design_seed": schedule_v94.PUBLIC_DESIGN_SEEDS[
            "development"
        ],
        "balanced_schedule_maximum_iterations": schedule_v94.MAX_ITERATIONS,
        "direct_r2_plan_read": False,
        "train_schedule_commitment_sha256": train_schedule.commitment[
            "split_schedule_commitment_sha256"
        ],
        "development_schedule_commitment_sha256": development_schedule.commitment[
            "split_schedule_commitment_sha256"
        ],
        "train_latent_schedule_sha256": train_schedule.commitment[
            "latent_schedule_sha256"
        ],
        "development_latent_schedule_sha256": development_schedule.commitment[
            "latent_schedule_sha256"
        ],
        "schedule_pair_audit_commitment_sha256": pair_receipt[
            "pair_audit_commitment_sha256"
        ],
        "noise_signature_version": signatures_v94.VERSION,
        "noise_signature_source_pins": _json_value(
            noise_signatures.commitment["source_pins"]
        ),
        "noise_signature_rows_sha256": noise_signatures.commitment[
            "signature_rows_sha256"
        ],
        "noise_signature_set_commitment_sha256": noise_signatures.commitment[
            "signature_set_commitment_sha256"
        ],
    }


def _build_issuance_claim(
    *, implementation_commit: str, implementation_tree: str
) -> dict[str, Any]:
    return _with_self_hash({
        "version": VERSION,
        "status": "KEY_ISSUANCE_CLAIMED_NO_KEY_YET",
        "attempt_id": ATTEMPT_ID,
        "authorization_policy_target_path": AUTHORIZATION_POLICY_RELATIVE,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "issuance_sequence": 1,
        "additional_key_candidates_authorized": 0,
        "same_attempt_issuance_retry_authorized": False,
    })


def _build_issuance_receipt(
    *,
    implementation_commit: str,
    issuance_claim_canonical_sha256: str,
    time_key_commitment_sha256: str,
) -> dict[str, Any]:
    return _with_self_hash({
        "version": VERSION,
        "status": "ISSUED_ONCE_NOT_CONSUMED",
        "attempt_id": ATTEMPT_ID,
        "authorization_policy_target_path": AUTHORIZATION_POLICY_RELATIVE,
        "implementation_commit": implementation_commit,
        "issuance_claim_canonical_sha256": issuance_claim_canonical_sha256,
        "time_key_commitment_sha256": time_key_commitment_sha256,
        "key_expected_length_bytes": 32,
        "issuance_sequence": 1,
        "additional_key_candidates_authorized": 0,
    })


def build_authorization_payload(
    *,
    root: Path,
    implementation_commit: str,
    implementation_tree: str,
    time_key_commitment_sha256: str,
    key_issuance_claim_canonical_sha256: str,
    key_issuance_receipt_canonical_sha256: str,
    frozen_input_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not _is_sha256(time_key_commitment_sha256):
        raise FormalPrebuildAuthorityV94Error("Time-key commitment drift")
    if any(
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
        for value in (implementation_commit, implementation_tree)
    ):
        raise FormalPrebuildAuthorityV94Error("Implementation commit drift")
    if not _is_sha256(key_issuance_claim_canonical_sha256):
        raise FormalPrebuildAuthorityV94Error("Key issuance claim drift")
    if not _is_sha256(key_issuance_receipt_canonical_sha256):
        raise FormalPrebuildAuthorityV94Error("Key issuance receipt drift")
    implementation_policy = implementation_v94.load_formal_policy()
    source_records = _collect_source_records(root)
    payload = {
        "version": POLICY_VERSION,
        "status": "FROZEN_SINGLE_ATTEMPT_PREBUILD_GATE_AUTHORIZED",
        "authorization": dict(AUTHORIZATION),
        "attempt": {
            "attempt_id": ATTEMPT_ID,
            "single_attempt": True,
            "rerun_after_claim_forbidden": True,
            "authorization_policy_path": AUTHORIZATION_POLICY_RELATIVE,
            "private_unconsumed_key_path": (
                f"{PRIVATE_ROOT_RELATIVE}/{UNCONSUMED_KEY_NAME}"
            ),
            "private_consumed_key_path": (
                f"{PRIVATE_ROOT_RELATIVE}/{CONSUMED_KEY_NAME}"
            ),
            "private_key_issuance_receipt_path": (
                f"{PRIVATE_ROOT_RELATIVE}/{ISSUANCE_RECEIPT_NAME}"
            ),
            "private_key_issuance_claim_path": (
                f"{PRIVATE_ROOT_RELATIVE}/{ISSUANCE_CLAIM_NAME}"
            ),
            "private_key_issuance_failure_path": (
                f"{PRIVATE_ROOT_RELATIVE}/{ISSUANCE_FAILURE_NAME}"
            ),
            "private_claim_path": f"{PRIVATE_ROOT_RELATIVE}/{CLAIM_NAME}",
            "private_consumption_receipt_path": (
                f"{PRIVATE_ROOT_RELATIVE}/{CONSUMPTION_RECEIPT_NAME}"
            ),
            "private_mechanical_failure_receipt_path": (
                f"{PRIVATE_ROOT_RELATIVE}/{MECHANICAL_FAILURE_RECEIPT_NAME}"
            ),
            "private_pass_terminal_validation_pending_path": (
                f"{PRIVATE_ROOT_RELATIVE}/"
                f"{PASS_TERMINAL_VALIDATION_PENDING_NAME}"
            ),
            "private_pass_terminal_validation_completion_path": (
                f"{PRIVATE_ROOT_RELATIVE}/"
                f"{PASS_TERMINAL_VALIDATION_COMPLETION_NAME}"
            ),
            "private_pass_terminal_validation_revalidating_path": (
                f"{PRIVATE_ROOT_RELATIVE}/"
                f"{PASS_TERMINAL_VALIDATION_REVALIDATING_NAME}"
            ),
            "output_root": OUTPUT_ROOT_RELATIVE,
            "launch_manifest_path": f"{OUTPUT_ROOT_RELATIVE}/{LAUNCH_NAME}",
            "result_path": f"{OUTPUT_ROOT_RELATIVE}/{RESULT_NAME}",
            "result_building_path": (
                f"{OUTPUT_ROOT_RELATIVE}/{RESULT_BUILDING_NAME}"
            ),
            "terminal_path": f"{OUTPUT_ROOT_RELATIVE}/{TERMINAL_NAME}",
            "unique_command": UNIQUE_COMMAND,
            "unique_command_sha256": UNIQUE_COMMAND_SHA256,
            "expected_output_allowlist": [
                LAUNCH_NAME,
                RESULT_NAME,
                TERMINAL_NAME,
            ],
        },
        "implementation_binding": {
            "branch": EXPECTED_BRANCH,
            "implementation_commit": implementation_commit,
            "implementation_tree": implementation_tree,
            "policy_commit_parent_must_equal_implementation_commit": True,
            "policy_commit_only_path": AUTHORIZATION_POLICY_RELATIVE,
            "implementation_policy_path": IMPLEMENTATION_POLICY_RELATIVE,
            "implementation_policy_sha256": implementation_v94.POLICY_SHA256,
            "registered_source_closure_required": True,
            "source_files": source_records,
            "source_files_commitment_sha256": _canonical_sha256(source_records),
        },
        "scientific_contract": {
            "delegated_to_frozen_implementation_policy": True,
            "view": implementation_policy["matrix_contract"]["view"],
            "train_world_count": implementation_policy["dataset_contract"][
                "train_world_count"
            ],
            "development_world_count": implementation_policy[
                "dataset_contract"
            ]["development_world_count"],
            "pair_count_per_world": implementation_policy["dataset_contract"][
                "pair_count_per_world"
            ],
            "positive_pair_count_per_world": implementation_policy[
                "dataset_contract"
            ]["positive_pair_count_per_world"],
            "probe_model_names": list(implementation_policy["probe_models"]),
            "bootstrap_draws_sha256": implementation_policy["bootstrap"][
                "draws_raw_i8_c_sha256"
            ],
            "gates": _json_value(implementation_policy["gates"]),
            "pass_claim": implementation_policy["pass_claim"],
        },
        "input_binding": {
            **_json_value(frozen_input_binding or _collect_frozen_input_binding()),
            "time_key_commitment_sha256": time_key_commitment_sha256,
            "key_issuance_claim_canonical_sha256": (
                key_issuance_claim_canonical_sha256
            ),
            "key_issuance_receipt_canonical_sha256": (
                key_issuance_receipt_canonical_sha256
            ),
        },
        "runtime_contract": _runtime_contract(),
        "publication_contract": {
            "publish_raw_time_key": False,
            "publish_controller_groups": False,
            "publish_row_labels": False,
            "publish_matrices": False,
            "publish_score_vectors": False,
            "publish_small_receipts_and_hashes_only": True,
            "passed_makes_method_root_eligible": True,
            "passed_authorizes_method_root_build": False,
            "passed_authorizes_training": False,
            "passed_retains_private_time_index_continuation": True,
            "failed_retires_time_index_continuation": True,
            "failed_closes_attempt": True,
            "mechanical_failure_closes_attempt": True,
            "mechanical_failure_facts_frozen_before_cleanup": True,
            "pass_terminal_validation_is_monotonic": True,
        },
    }
    return _with_self_hash(payload)


def validate_authorization_payload(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    verify_source_bytes: bool = True,
) -> None:
    if type(payload) is not dict or tuple(payload) != (
        "version",
        "status",
        "authorization",
        "attempt",
        "implementation_binding",
        "scientific_contract",
        "input_binding",
        "runtime_contract",
        "publication_contract",
        "canonical_self_sha256",
    ):
        raise FormalPrebuildAuthorityV94Error("Authorization policy schema drift")
    _verify_self_hash(payload)
    if (
        payload["version"] != POLICY_VERSION
        or payload["status"] != "FROZEN_SINGLE_ATTEMPT_PREBUILD_GATE_AUTHORIZED"
        or not _exact_json_equal(payload["authorization"], AUTHORIZATION)
    ):
        raise FormalPrebuildAuthorityV94Error("Authorization capability drift")
    expected_attempt = {
        "attempt_id": ATTEMPT_ID,
        "single_attempt": True,
        "rerun_after_claim_forbidden": True,
        "authorization_policy_path": AUTHORIZATION_POLICY_RELATIVE,
        "private_unconsumed_key_path": f"{PRIVATE_ROOT_RELATIVE}/{UNCONSUMED_KEY_NAME}",
        "private_consumed_key_path": f"{PRIVATE_ROOT_RELATIVE}/{CONSUMED_KEY_NAME}",
        "private_key_issuance_receipt_path": (
            f"{PRIVATE_ROOT_RELATIVE}/{ISSUANCE_RECEIPT_NAME}"
        ),
        "private_key_issuance_claim_path": (
            f"{PRIVATE_ROOT_RELATIVE}/{ISSUANCE_CLAIM_NAME}"
        ),
        "private_key_issuance_failure_path": (
            f"{PRIVATE_ROOT_RELATIVE}/{ISSUANCE_FAILURE_NAME}"
        ),
        "private_claim_path": f"{PRIVATE_ROOT_RELATIVE}/{CLAIM_NAME}",
        "private_consumption_receipt_path": (
            f"{PRIVATE_ROOT_RELATIVE}/{CONSUMPTION_RECEIPT_NAME}"
        ),
        "private_mechanical_failure_receipt_path": (
            f"{PRIVATE_ROOT_RELATIVE}/{MECHANICAL_FAILURE_RECEIPT_NAME}"
        ),
        "private_pass_terminal_validation_pending_path": (
            f"{PRIVATE_ROOT_RELATIVE}/"
            f"{PASS_TERMINAL_VALIDATION_PENDING_NAME}"
        ),
        "private_pass_terminal_validation_completion_path": (
            f"{PRIVATE_ROOT_RELATIVE}/"
            f"{PASS_TERMINAL_VALIDATION_COMPLETION_NAME}"
        ),
        "private_pass_terminal_validation_revalidating_path": (
            f"{PRIVATE_ROOT_RELATIVE}/"
            f"{PASS_TERMINAL_VALIDATION_REVALIDATING_NAME}"
        ),
        "output_root": OUTPUT_ROOT_RELATIVE,
        "launch_manifest_path": f"{OUTPUT_ROOT_RELATIVE}/{LAUNCH_NAME}",
        "result_path": f"{OUTPUT_ROOT_RELATIVE}/{RESULT_NAME}",
        "result_building_path": f"{OUTPUT_ROOT_RELATIVE}/{RESULT_BUILDING_NAME}",
        "terminal_path": f"{OUTPUT_ROOT_RELATIVE}/{TERMINAL_NAME}",
        "unique_command": UNIQUE_COMMAND,
        "unique_command_sha256": UNIQUE_COMMAND_SHA256,
        "expected_output_allowlist": [LAUNCH_NAME, RESULT_NAME, TERMINAL_NAME],
    }
    if not _exact_json_equal(payload["attempt"], expected_attempt):
        raise FormalPrebuildAuthorityV94Error("Authorization attempt drift")
    binding = payload["implementation_binding"]
    if (
        type(binding) is not dict
        or tuple(binding) != (
            "branch",
            "implementation_commit",
            "implementation_tree",
            "policy_commit_parent_must_equal_implementation_commit",
            "policy_commit_only_path",
            "implementation_policy_path",
            "implementation_policy_sha256",
            "registered_source_closure_required",
            "source_files",
            "source_files_commitment_sha256",
        )
        or binding["branch"] != EXPECTED_BRANCH
        or binding["policy_commit_parent_must_equal_implementation_commit"] is not True
        or binding["policy_commit_only_path"] != AUTHORIZATION_POLICY_RELATIVE
        or binding["implementation_policy_path"] != IMPLEMENTATION_POLICY_RELATIVE
        or binding["implementation_policy_sha256"] != implementation_v94.POLICY_SHA256
        or binding["registered_source_closure_required"] is not True
        or type(binding["source_files"]) is not list
        or [record.get("path") for record in binding["source_files"]]
        != list(PINNED_SOURCE_RELATIVES)
        or not all(
            type(record) is dict
            and tuple(record) == ("path", "sha256")
            and _is_sha256(record["sha256"])
            for record in binding["source_files"]
        )
        or binding["source_files_commitment_sha256"]
        != _canonical_sha256(binding["source_files"])
    ):
        raise FormalPrebuildAuthorityV94Error("Implementation binding drift")
    if any(
        type(binding[field]) is not str
        or len(binding[field]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in binding[field]
        )
        for field in ("implementation_commit", "implementation_tree")
    ):
        raise FormalPrebuildAuthorityV94Error("Implementation commit drift")
    if verify_source_bytes and binding["source_files"] != _collect_source_records(root):
        raise FormalPrebuildAuthorityV94Error("Pinned source byte drift")
    implementation_policy = implementation_v94.load_formal_policy()
    expected_science = {
        "delegated_to_frozen_implementation_policy": True,
        "view": implementation_v94.FORMAL_VIEW,
        "train_world_count": implementation_v94.FORMAL_SPLIT_WORLDS,
        "development_world_count": implementation_v94.FORMAL_SPLIT_WORLDS,
        "pair_count_per_world": implementation_v94.FORMAL_PAIRS_PER_WORLD,
        "positive_pair_count_per_world": implementation_v94.FORMAL_POSITIVES_PER_WORLD,
        "probe_model_names": list(implementation_policy["probe_models"]),
        "bootstrap_draws_sha256": implementation_v94.FORMAL_BOOTSTRAP[
            "draws_raw_i8_c_sha256"
        ],
        "gates": dict(implementation_v94.FORMAL_GATES),
        "pass_claim": implementation_v94.PASS_CLAIM,
    }
    expected_source_pins = [
        ["market_item.xlsx", signatures_v94.WORKBOOK_SHA256],
        [
            "reports/step2_content_item_manifest.csv",
            signatures_v94.MANIFEST_SHA256,
        ],
        [
            "reports/step28_synthetic_chinese_dataset/"
            "v13_dev_smoke_v1_20260727/reference/"
            "style_source_train_sellers.csv",
            signatures_v94.ALLOWLIST_SHA256,
        ],
    ]
    expected_input_fields = (
        "balanced_schedule_version",
        "train_public_design_seed",
        "development_public_design_seed",
        "balanced_schedule_maximum_iterations",
        "direct_r2_plan_read",
        "train_schedule_commitment_sha256",
        "development_schedule_commitment_sha256",
        "train_latent_schedule_sha256",
        "development_latent_schedule_sha256",
        "schedule_pair_audit_commitment_sha256",
        "noise_signature_version",
        "noise_signature_source_pins",
        "noise_signature_rows_sha256",
        "noise_signature_set_commitment_sha256",
        "time_key_commitment_sha256",
        "key_issuance_claim_canonical_sha256",
        "key_issuance_receipt_canonical_sha256",
    )
    input_binding = payload["input_binding"]
    if (
        not _exact_json_equal(payload["scientific_contract"], expected_science)
        or type(input_binding) is not dict
        or tuple(input_binding) != expected_input_fields
        or input_binding["balanced_schedule_version"] != schedule_v94.VERSION
        or type(input_binding["train_public_design_seed"]) is not int
        or input_binding["train_public_design_seed"]
        != schedule_v94.PUBLIC_DESIGN_SEEDS["train"]
        or type(input_binding["development_public_design_seed"]) is not int
        or input_binding["development_public_design_seed"]
        != schedule_v94.PUBLIC_DESIGN_SEEDS["development"]
        or type(input_binding["balanced_schedule_maximum_iterations"])
        is not int
        or input_binding["balanced_schedule_maximum_iterations"]
        != schedule_v94.MAX_ITERATIONS
        or input_binding["direct_r2_plan_read"] is not False
        or input_binding["noise_signature_version"] != signatures_v94.VERSION
        or input_binding["noise_signature_source_pins"] != expected_source_pins
        or input_binding["noise_signature_rows_sha256"]
        != signatures_v94.EXPECTED_SIGNATURE_ROWS_SHA256
        or input_binding["noise_signature_set_commitment_sha256"]
        != signatures_v94.EXPECTED_SIGNATURE_SET_COMMITMENT_SHA256
        or not all(
            _is_sha256(input_binding[field])
            for field in (
                "train_schedule_commitment_sha256",
                "development_schedule_commitment_sha256",
                "train_latent_schedule_sha256",
                "development_latent_schedule_sha256",
                "schedule_pair_audit_commitment_sha256",
            )
        )
        or not _is_sha256(input_binding["time_key_commitment_sha256"])
        or not _is_sha256(
            input_binding["key_issuance_claim_canonical_sha256"]
        )
        or not _is_sha256(
            input_binding["key_issuance_receipt_canonical_sha256"]
        )
        or not _exact_json_equal(
            payload["runtime_contract"], _runtime_contract()
        )
        or not _exact_json_equal(payload["publication_contract"], {
            "publish_raw_time_key": False,
            "publish_controller_groups": False,
            "publish_row_labels": False,
            "publish_matrices": False,
            "publish_score_vectors": False,
            "publish_small_receipts_and_hashes_only": True,
            "passed_makes_method_root_eligible": True,
            "passed_authorizes_method_root_build": False,
            "passed_authorizes_training": False,
            "passed_retains_private_time_index_continuation": True,
            "failed_retires_time_index_continuation": True,
            "failed_closes_attempt": True,
            "mechanical_failure_closes_attempt": True,
            "mechanical_failure_facts_frozen_before_cleanup": True,
            "pass_terminal_validation_is_monotonic": True,
        })
    ):
        raise FormalPrebuildAuthorityV94Error("Scientific authorization drift")


def _write_new_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.building")
    if path.exists() or temporary.exists():
        raise FormalPrebuildAuthorityV94Error(f"Refusing to overwrite {path}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(
            _json_value(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_new_bytes(path, raw)


def _replace_existing_building_with_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Finish a publication without ever removing its durable marker first."""
    raw = (
        json.dumps(
            _json_value(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    building = path.with_name(f"{path.name}.building")
    if path.exists() or not building.is_file() or building.is_symlink():
        raise FormalPrebuildAuthorityV94Error(
            f"Interrupted publication marker drift for {path}"
        )
    # Truncating or partially rewriting this file never removes the marker.
    # Every interruption therefore remains recoverable only as a mechanical
    # failure until the final atomic replace succeeds.
    with building.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(building, path)


def _write_durable_failure_marker(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Claim failure publication; partial bytes still remain a valid fence."""
    raw = (
        json.dumps(
            _json_value(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FormalPrebuildAuthorityV94Error(
            f"Refusing to overwrite failure marker {path}"
        )
    # Do not clean this file on any exception: existence, not valid JSON, is
    # the monotonic recovery fence.
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def validate_issuance_claim(
    policy: Mapping[str, Any], *, paths: RuntimePaths
) -> dict[str, Any]:
    try:
        claim = json.loads(paths.issuance_claim.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPrebuildAuthorityV94Error(
            "Key issuance claim drift"
        ) from error
    _verify_self_hash(claim)
    expected = {
        "version": VERSION,
        "status": "KEY_ISSUANCE_CLAIMED_NO_KEY_YET",
        "attempt_id": ATTEMPT_ID,
        "authorization_policy_target_path": AUTHORIZATION_POLICY_RELATIVE,
        "implementation_commit": policy["implementation_binding"][
            "implementation_commit"
        ],
        "implementation_tree": policy["implementation_binding"][
            "implementation_tree"
        ],
        "issuance_sequence": 1,
        "additional_key_candidates_authorized": 0,
        "same_attempt_issuance_retry_authorized": False,
    }
    without_self = dict(claim)
    without_self.pop("canonical_self_sha256", None)
    if (
        not _exact_json_equal(without_self, expected)
        or claim["canonical_self_sha256"]
        != policy["input_binding"][
            "key_issuance_claim_canonical_sha256"
        ]
    ):
        raise FormalPrebuildAuthorityV94Error("Key issuance claim drift")
    return claim


def validate_issuance_receipt(
    policy: Mapping[str, Any],
    *,
    paths: RuntimePaths,
    verify_unconsumed_key: bool,
) -> dict[str, Any]:
    issuance_claim = validate_issuance_claim(policy, paths=paths)
    try:
        receipt = json.loads(paths.issuance_receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPrebuildAuthorityV94Error(
            "Key issuance receipt drift"
        ) from error
    _verify_self_hash(receipt)
    expected = {
        "version": VERSION,
        "status": "ISSUED_ONCE_NOT_CONSUMED",
        "attempt_id": ATTEMPT_ID,
        "authorization_policy_target_path": AUTHORIZATION_POLICY_RELATIVE,
        "implementation_commit": policy["implementation_binding"][
            "implementation_commit"
        ],
        "issuance_claim_canonical_sha256": issuance_claim[
            "canonical_self_sha256"
        ],
        "time_key_commitment_sha256": policy["input_binding"][
            "time_key_commitment_sha256"
        ],
        "key_expected_length_bytes": 32,
        "issuance_sequence": 1,
        "additional_key_candidates_authorized": 0,
    }
    without_self = dict(receipt)
    without_self.pop("canonical_self_sha256", None)
    if (
        not _exact_json_equal(without_self, expected)
        or receipt["canonical_self_sha256"]
        != policy["input_binding"][
            "key_issuance_receipt_canonical_sha256"
        ]
    ):
        raise FormalPrebuildAuthorityV94Error("Key issuance receipt drift")
    if verify_unconsumed_key:
        if (
            not paths.unconsumed_key.is_file()
            or paths.unconsumed_key.stat().st_size != 32
            or _sha256_file(paths.unconsumed_key)
            != policy["input_binding"]["time_key_commitment_sha256"]
        ):
            raise FormalPrebuildAuthorityV94Error("Issued private key drift")
    return receipt


def freeze_authorization(
    *,
    root: Path = ROOT,
    random_bytes: Callable[[int], bytes] = os.urandom,
) -> dict[str, Any]:
    paths = runtime_paths(root)
    issuance_claim_building = paths.issuance_claim.with_name(
        f"{paths.issuance_claim.name}.building"
    )
    if not paths.issuance_claim.exists() and issuance_claim_building.exists():
        _remove_path(issuance_claim_building)
    implementation_v94.load_formal_policy()
    _require_clean_repository(root)
    if _git(root, "branch", "--show-current") != EXPECTED_BRANCH:
        raise FormalPrebuildAuthorityV94Error("Formal branch drift")
    forbidden = (
        paths.authorization_policy,
        paths.unconsumed_key,
        paths.consumed_key,
        paths.issuance_claim,
        paths.issuance_receipt,
        paths.issuance_failure,
        paths.claim,
        paths.consumption_receipt,
        paths.mechanical_failure_receipt,
        paths.pass_terminal_validation_pending,
        paths.pass_terminal_validation_completion,
        paths.pass_terminal_validation_revalidating,
        paths.output_root,
    )
    if any(path.exists() for path in forbidden):
        raise FormalPrebuildAuthorityV94Error(
            "Formal authorization or attempt state already exists"
        )
    implementation_commit = _git(root, "rev-parse", "HEAD")
    implementation_tree = _git(root, "rev-parse", "HEAD^{tree}")
    frozen_input_binding = _collect_frozen_input_binding()
    issuance_claim = _build_issuance_claim(
        implementation_commit=implementation_commit,
        implementation_tree=implementation_tree,
    )
    _write_new_json(paths.issuance_claim, issuance_claim)
    stage = "random_key_generation"
    try:
        key = random_bytes(32)
        if type(key) is not bytes or len(key) != 32:
            raise FormalPrebuildAuthorityV94Error("Random time key drift")
        stage = "authorization_payload_construction"
        time_commitment = _sha256_bytes(key)
        issuance_receipt = _build_issuance_receipt(
            implementation_commit=implementation_commit,
            issuance_claim_canonical_sha256=issuance_claim[
                "canonical_self_sha256"
            ],
            time_key_commitment_sha256=time_commitment,
        )
        policy = build_authorization_payload(
            root=root,
            implementation_commit=implementation_commit,
            implementation_tree=implementation_tree,
            time_key_commitment_sha256=time_commitment,
            key_issuance_claim_canonical_sha256=issuance_claim[
                "canonical_self_sha256"
            ],
            key_issuance_receipt_canonical_sha256=issuance_receipt[
                "canonical_self_sha256"
            ],
            frozen_input_binding=frozen_input_binding,
        )
        stage = "authorization_material_publication"
        _write_new_bytes(paths.unconsumed_key, key)
        _write_new_json(paths.issuance_receipt, issuance_receipt)
        _write_new_json(paths.authorization_policy, policy)
        stage = "authorization_material_validation"
        loaded = json.loads(paths.authorization_policy.read_text(encoding="utf-8"))
        validate_authorization_payload(loaded, root=root)
        if _sha256_file(paths.unconsumed_key) != time_commitment:
            raise FormalPrebuildAuthorityV94Error("Written time key drift")
        validate_issuance_receipt(
            policy,
            paths=paths,
            verify_unconsumed_key=True,
        )
    except Exception as error:
        if paths.authorization_policy.exists():
            paths.authorization_policy.unlink()
        if paths.unconsumed_key.exists():
            paths.unconsumed_key.unlink()
        if paths.issuance_receipt.exists():
            paths.issuance_receipt.unlink()
        failure = _with_self_hash({
            "version": VERSION,
            "status": "KEY_ISSUANCE_FAILED_ATTEMPT_CLOSED",
            "attempt_id": ATTEMPT_ID,
            "implementation_commit": implementation_commit,
            "issuance_claim_canonical_sha256": issuance_claim[
                "canonical_self_sha256"
            ],
            "failure_stage": stage,
            "error_type": type(error).__name__,
            "error_message_sha256": _sha256_bytes(
                str(error).encode("utf-8")
            ),
            "same_attempt_issuance_retry_authorized": False,
            "raw_key_retained": False,
        })
        try:
            if not paths.issuance_failure.exists():
                _write_new_json(paths.issuance_failure, failure)
        except Exception:
            # The retained issuance claim alone permanently closes the attempt.
            pass
        raise
    return {
        "version": VERSION,
        "status": "AUTHORIZATION_FROZEN_NOT_CONSUMED",
        "attempt_id": ATTEMPT_ID,
        "implementation_commit": implementation_commit,
        "authorization_policy_path": AUTHORIZATION_POLICY_RELATIVE,
        "authorization_policy_sha256": _sha256_file(paths.authorization_policy),
        "time_key_commitment_sha256": time_commitment,
        "key_issuance_claim_canonical_sha256": issuance_claim[
            "canonical_self_sha256"
        ],
        "key_issuance_receipt_canonical_sha256": issuance_receipt[
            "canonical_self_sha256"
        ],
    }


def load_authorization_policy(
    *, root: Path = ROOT, path: Path | None = None
) -> dict[str, Any]:
    expected = (root / AUTHORIZATION_POLICY_RELATIVE).resolve()
    resolved = (path or expected).resolve()
    if resolved != expected:
        raise FormalPrebuildAuthorityV94Error("Authorization policy path drift")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPrebuildAuthorityV94Error(
            "Authorization policy bytes drift"
        ) from error
    validate_authorization_payload(payload, root=root)
    return payload


def _validate_policy_commit(
    policy: Mapping[str, Any],
    *,
    root: Path,
    allow_attempt_output: bool = False,
) -> None:
    _require_clean_repository(root, allow_attempt_output=allow_attempt_output)
    if _git(root, "branch", "--show-current") != EXPECTED_BRANCH:
        raise FormalPrebuildAuthorityV94Error("Formal branch drift")
    parent_line = _git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    implementation_commit = policy["implementation_binding"][
        "implementation_commit"
    ]
    if (
        len(parent_line) != 2
        or parent_line[1] != implementation_commit
        or _git(root, "rev-parse", f"{implementation_commit}^{{tree}}")
        != policy["implementation_binding"]["implementation_tree"]
    ):
        raise FormalPrebuildAuthorityV94Error("Authorization commit parent drift")
    changed = tuple(
        line.replace("\\", "/")
        for line in _git(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "HEAD",
        ).splitlines()
        if line
    )
    if changed != (AUTHORIZATION_POLICY_RELATIVE,):
        raise FormalPrebuildAuthorityV94Error("Authorization commit scope drift")


def _preflight_materials(policy: Mapping[str, Any]) -> dict[str, Any]:
    implementation_policy = implementation_v94.load_formal_policy()
    source_closure = source_guard_v94.audit_registered_sources()
    noise_signatures = signatures_v94.build_noise_signatures()
    train_schedule = schedule_v94.build_split_schedule("train")
    development_schedule = schedule_v94.build_split_schedule("development")
    pair_receipt = schedule_v94.validate_train_development_pair(
        train_schedule,
        development_schedule,
    )
    input_binding = policy["input_binding"]
    if (
        _json_value(noise_signatures.commitment["source_pins"])
        != input_binding["noise_signature_source_pins"]
        or noise_signatures.commitment["signature_rows_sha256"]
        != input_binding["noise_signature_rows_sha256"]
        or noise_signatures.commitment["signature_set_commitment_sha256"]
        != input_binding["noise_signature_set_commitment_sha256"]
        or train_schedule.commitment["split_schedule_commitment_sha256"]
        != input_binding["train_schedule_commitment_sha256"]
        or development_schedule.commitment["split_schedule_commitment_sha256"]
        != input_binding["development_schedule_commitment_sha256"]
        or train_schedule.commitment["latent_schedule_sha256"]
        != input_binding["train_latent_schedule_sha256"]
        or development_schedule.commitment["latent_schedule_sha256"]
        != input_binding["development_latent_schedule_sha256"]
        or pair_receipt["pair_audit_commitment_sha256"]
        != input_binding["schedule_pair_audit_commitment_sha256"]
    ):
        raise FormalPrebuildAuthorityV94Error("Formal input commitment drift")
    return {
        "implementation_policy": implementation_policy,
        "source_closure": source_closure,
        "noise_signatures": noise_signatures,
        "train_schedule": train_schedule,
        "development_schedule": development_schedule,
        "pair_receipt": pair_receipt,
    }


def _preflight_public_commitment(preflight: Mapping[str, Any]) -> str:
    public = {
        "source_closure": _json_value(preflight["source_closure"]),
        "noise_signature_commitment": _json_value(
            preflight["noise_signatures"].commitment
        ),
        "train_schedule_commitment": _json_value(
            preflight["train_schedule"].commitment
        ),
        "development_schedule_commitment": _json_value(
            preflight["development_schedule"].commitment
        ),
        "pair_receipt": _json_value(preflight["pair_receipt"]),
    }
    return _canonical_sha256(public)


def _validate_fresh_issued_state(
    policy: Mapping[str, Any], *, paths: RuntimePaths
) -> None:
    if (
        not paths.unconsumed_key.is_file()
        or paths.unconsumed_key.stat().st_size != 32
        or paths.consumed_key.exists()
        or paths.claim.exists()
        or paths.consumption_receipt.exists()
        or paths.mechanical_failure_receipt.exists()
        or paths.pass_terminal_validation_pending.exists()
        or paths.pass_terminal_validation_completion.exists()
        or paths.pass_terminal_validation_revalidating.exists()
        or paths.output_root.exists()
    ):
        raise FormalPrebuildAuthorityV94Error("Formal attempt state is not fresh")
    if (
        not paths.issuance_claim.is_file()
        or not paths.issuance_receipt.is_file()
        or paths.issuance_failure.exists()
    ):
        raise FormalPrebuildAuthorityV94Error("Formal issuance state is missing")


def _validate_fresh_attempt_paths(*, paths: RuntimePaths) -> None:
    if (
        paths.consumed_key.exists()
        or paths.issuance_failure.exists()
        or paths.claim.exists()
        or paths.consumption_receipt.exists()
        or paths.mechanical_failure_receipt.exists()
        or paths.pass_terminal_validation_pending.exists()
        or paths.pass_terminal_validation_completion.exists()
        or paths.pass_terminal_validation_revalidating.exists()
        or paths.output_root.exists()
    ):
        raise FormalPrebuildAuthorityV94Error("Formal attempt state is not fresh")


def _claim_formal_launch(
    policy: Mapping[str, Any],
    *,
    paths: RuntimePaths,
) -> dict[str, Any]:
    policy_file_sha256 = _sha256_file(paths.authorization_policy)
    claim = _with_self_hash({
        "version": VERSION,
        "status": "FORMAL_LAUNCH_CLAIMED_KEY_NOT_YET_CONSUMED",
        "attempt_id": ATTEMPT_ID,
        "authorization_policy_file_sha256": policy_file_sha256,
        "authorization_policy_canonical_sha256": policy[
            "canonical_self_sha256"
        ],
        "implementation_commit": policy["implementation_binding"][
            "implementation_commit"
        ],
        "implementation_tree": policy["implementation_binding"][
            "implementation_tree"
        ],
        "authority_script_sha256": next(
            record["sha256"]
            for record in policy["implementation_binding"]["source_files"]
            if record["path"] == AUTHORITY_SCRIPT_RELATIVE
        ),
        "unique_command": UNIQUE_COMMAND,
        "unique_command_sha256": UNIQUE_COMMAND_SHA256,
        "output_root": OUTPUT_ROOT_RELATIVE,
        "time_key_commitment_sha256": policy["input_binding"][
            "time_key_commitment_sha256"
        ],
        "key_issuance_claim_canonical_sha256": policy["input_binding"][
            "key_issuance_claim_canonical_sha256"
        ],
        "key_issuance_receipt_canonical_sha256": policy["input_binding"][
            "key_issuance_receipt_canonical_sha256"
        ],
        "key_consumed": False,
    })
    _write_new_json(paths.claim, claim)
    paths.output_root.mkdir(parents=True, exist_ok=False)
    _write_new_json(paths.launch, claim)
    return _load_and_validate_launch_claim(policy, paths=paths)


def _load_and_validate_launch_claim(
    policy: Mapping[str, Any],
    *,
    paths: RuntimePaths,
) -> dict[str, Any]:
    try:
        claim = json.loads(paths.claim.read_text(encoding="utf-8"))
        launch = json.loads(paths.launch.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPrebuildAuthorityV94Error(
            "Formal launch claim drift"
        ) from error
    _verify_self_hash(claim)
    _verify_self_hash(launch)
    expected = {
        "version": VERSION,
        "status": "FORMAL_LAUNCH_CLAIMED_KEY_NOT_YET_CONSUMED",
        "attempt_id": ATTEMPT_ID,
        "authorization_policy_file_sha256": _sha256_file(
            paths.authorization_policy
        ),
        "authorization_policy_canonical_sha256": policy[
            "canonical_self_sha256"
        ],
        "implementation_commit": policy["implementation_binding"][
            "implementation_commit"
        ],
        "implementation_tree": policy["implementation_binding"][
            "implementation_tree"
        ],
        "authority_script_sha256": next(
            record["sha256"]
            for record in policy["implementation_binding"]["source_files"]
            if record["path"] == AUTHORITY_SCRIPT_RELATIVE
        ),
        "unique_command": UNIQUE_COMMAND,
        "unique_command_sha256": UNIQUE_COMMAND_SHA256,
        "output_root": OUTPUT_ROOT_RELATIVE,
        "time_key_commitment_sha256": policy["input_binding"][
            "time_key_commitment_sha256"
        ],
        "key_issuance_claim_canonical_sha256": policy["input_binding"][
            "key_issuance_claim_canonical_sha256"
        ],
        "key_issuance_receipt_canonical_sha256": policy["input_binding"][
            "key_issuance_receipt_canonical_sha256"
        ],
        "key_consumed": False,
    }
    without_self = dict(claim)
    without_self.pop("canonical_self_sha256", None)
    if (
        not _exact_json_equal(without_self, expected)
        or not _exact_json_equal(launch, claim)
    ):
        raise FormalPrebuildAuthorityV94Error("Formal launch claim drift")
    return claim


def _consume_claimed_key(
    policy: Mapping[str, Any],
    *,
    paths: RuntimePaths,
    claim: Mapping[str, Any],
    preflight_commitment_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    try:
        os.replace(paths.unconsumed_key, paths.consumed_key)
    except Exception:
        # The exclusive launch claim permanently consumes this attempt even if rename fails.
        if paths.unconsumed_key.exists():
            paths.unconsumed_key.unlink()
        raise
    if not paths.consumed_key.is_file():
        raise FormalPrebuildAuthorityV94Error("Consumed time-key missing")
    key = paths.consumed_key.read_bytes()
    observed_commitment = _sha256_bytes(key)
    key_valid = (
        len(key) == 32
        and observed_commitment
        == policy["input_binding"]["time_key_commitment_sha256"]
    )
    receipt = _with_self_hash({
        "version": VERSION,
        "status": (
            "CONSUMED_FOR_ATTEMPT_VALIDATED"
            if key_valid
            else "CONSUMED_FOR_ATTEMPT_COMMITMENT_MISMATCH"
        ),
        "attempt_id": ATTEMPT_ID,
        "claim_canonical_sha256": claim["canonical_self_sha256"],
        "authorization_policy_file_sha256": claim[
            "authorization_policy_file_sha256"
        ],
        "authorization_policy_canonical_sha256": policy[
            "canonical_self_sha256"
        ],
        "time_key_commitment_sha256": policy["input_binding"][
            "time_key_commitment_sha256"
        ],
        "observed_consumed_key_sha256": observed_commitment,
        "observed_consumed_key_length_bytes": len(key),
        "key_issuance_claim_canonical_sha256": policy["input_binding"][
            "key_issuance_claim_canonical_sha256"
        ],
        "key_issuance_receipt_canonical_sha256": policy["input_binding"][
            "key_issuance_receipt_canonical_sha256"
        ],
        "preflight_commitment_sha256": preflight_commitment_sha256,
        "consumption_sequence": 1,
        "same_attempt_gate_rerun_authorized": False,
        "method_root_continuation_eligible_only_on_gate_pass": True,
    })
    _write_new_json(paths.consumption_receipt, receipt)
    if not key_valid:
        raise FormalPrebuildAuthorityV94Error("Consumed time-key drift")
    return key, receipt


def _load_and_validate_consumption_receipt(
    policy: Mapping[str, Any],
    *,
    paths: RuntimePaths,
    claim: Mapping[str, Any],
    require_valid_key: bool,
) -> dict[str, Any]:
    try:
        receipt = json.loads(
            paths.consumption_receipt.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPrebuildAuthorityV94Error(
            "Key consumption receipt drift"
        ) from error
    _verify_self_hash(receipt)
    expected_fields = (
        "version",
        "status",
        "attempt_id",
        "claim_canonical_sha256",
        "authorization_policy_file_sha256",
        "authorization_policy_canonical_sha256",
        "time_key_commitment_sha256",
        "observed_consumed_key_sha256",
        "observed_consumed_key_length_bytes",
        "key_issuance_claim_canonical_sha256",
        "key_issuance_receipt_canonical_sha256",
        "preflight_commitment_sha256",
        "consumption_sequence",
        "same_attempt_gate_rerun_authorized",
        "method_root_continuation_eligible_only_on_gate_pass",
        "canonical_self_sha256",
    )
    observed_valid = (
        type(receipt.get("observed_consumed_key_length_bytes")) is int
        and receipt.get("observed_consumed_key_length_bytes") == 32
        and receipt.get("observed_consumed_key_sha256")
        == policy["input_binding"]["time_key_commitment_sha256"]
    )
    expected_status = (
        "CONSUMED_FOR_ATTEMPT_VALIDATED"
        if observed_valid
        else "CONSUMED_FOR_ATTEMPT_COMMITMENT_MISMATCH"
    )
    if (
        tuple(receipt) != expected_fields
        or receipt["version"] != VERSION
        or receipt["status"] != expected_status
        or receipt["attempt_id"] != ATTEMPT_ID
        or receipt["claim_canonical_sha256"]
        != claim["canonical_self_sha256"]
        or receipt["authorization_policy_file_sha256"]
        != _sha256_file(paths.authorization_policy)
        or receipt["authorization_policy_canonical_sha256"]
        != policy["canonical_self_sha256"]
        or receipt["time_key_commitment_sha256"]
        != policy["input_binding"]["time_key_commitment_sha256"]
        or not _is_sha256(receipt["observed_consumed_key_sha256"])
        or receipt["key_issuance_claim_canonical_sha256"]
        != policy["input_binding"][
            "key_issuance_claim_canonical_sha256"
        ]
        or receipt["key_issuance_receipt_canonical_sha256"]
        != policy["input_binding"][
            "key_issuance_receipt_canonical_sha256"
        ]
        or not _is_sha256(receipt["preflight_commitment_sha256"])
        or type(receipt["consumption_sequence"]) is not int
        or receipt["consumption_sequence"] != 1
        or receipt["same_attempt_gate_rerun_authorized"] is not False
        or receipt[
            "method_root_continuation_eligible_only_on_gate_pass"
        ] is not True
        or (require_valid_key and not observed_valid)
    ):
        raise FormalPrebuildAuthorityV94Error("Key consumption receipt drift")
    return receipt


def _validate_consumption_lineage(
    policy: Mapping[str, Any],
    *,
    paths: RuntimePaths,
    claim: Mapping[str, Any],
    consumption_receipt: Mapping[str, Any],
    expected_preflight_commitment_sha256: str,
) -> dict[str, Any]:
    if paths.issuance_failure.exists():
        raise FormalPrebuildAuthorityV94Error(
            "Formal consumption lineage drift"
        )
    validate_issuance_receipt(
        policy,
        paths=paths,
        verify_unconsumed_key=False,
    )
    observed_claim = _load_and_validate_launch_claim(policy, paths=paths)
    observed_receipt = _load_and_validate_consumption_receipt(
        policy,
        paths=paths,
        claim=claim,
        require_valid_key=True,
    )
    if (
        not _is_sha256(expected_preflight_commitment_sha256)
        or not _exact_json_equal(observed_claim, claim)
        or not _exact_json_equal(observed_receipt, consumption_receipt)
        or observed_receipt["preflight_commitment_sha256"]
        != expected_preflight_commitment_sha256
    ):
        raise FormalPrebuildAuthorityV94Error(
            "Formal consumption lineage drift"
        )
    return observed_receipt


def _validate_method_root_continuation(
    policy: Mapping[str, Any],
    *,
    paths: RuntimePaths,
    claim: Mapping[str, Any],
    consumption_receipt: Mapping[str, Any],
    expected_preflight_commitment_sha256: str,
) -> None:
    _validate_consumption_lineage(
        policy,
        paths=paths,
        claim=claim,
        consumption_receipt=consumption_receipt,
        expected_preflight_commitment_sha256=(
            expected_preflight_commitment_sha256
        ),
    )
    if (
        paths.unconsumed_key.exists()
        or not paths.consumed_key.is_file()
        or paths.consumed_key.stat().st_size != 32
        or _sha256_file(paths.consumed_key)
        != policy["input_binding"]["time_key_commitment_sha256"]
    ):
        raise FormalPrebuildAuthorityV94Error(
            "Method-root time-index continuation drift"
        )


def _build_pass_terminal_validation_pending(
    *,
    policy: Mapping[str, Any],
    paths: RuntimePaths,
    claim: Mapping[str, Any],
    consumption_receipt: Mapping[str, Any],
    preflight_commitment_sha256: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return _with_self_hash({
        "version": VERSION,
        "status": "PASS_TERMINAL_VALIDATION_PENDING",
        "attempt_id": ATTEMPT_ID,
        "authorization_policy_file_sha256": _sha256_file(
            paths.authorization_policy
        ),
        "authorization_policy_canonical_sha256": policy[
            "canonical_self_sha256"
        ],
        "claim_canonical_sha256": claim["canonical_self_sha256"],
        "key_consumption_receipt_canonical_sha256": consumption_receipt[
            "canonical_self_sha256"
        ],
        "preflight_commitment_sha256": preflight_commitment_sha256,
        "result_canonical_sha256": result["canonical_self_sha256"],
        "continuation_validations_completed": 2,
        "same_attempt_pass_recovery_authorized": False,
    })


def _load_and_validate_pass_terminal_validation_pending(
    policy: Mapping[str, Any],
    *,
    paths: RuntimePaths,
    claim: Mapping[str, Any],
    consumption_receipt: Mapping[str, Any],
    expected_preflight_commitment_sha256: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        receipt = json.loads(
            paths.pass_terminal_validation_pending.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPrebuildAuthorityV94Error(
            "Pass terminal validation pending receipt drift"
        ) from error
    if type(receipt) is not dict or tuple(receipt) != (
        "version",
        "status",
        "attempt_id",
        "authorization_policy_file_sha256",
        "authorization_policy_canonical_sha256",
        "claim_canonical_sha256",
        "key_consumption_receipt_canonical_sha256",
        "preflight_commitment_sha256",
        "result_canonical_sha256",
        "continuation_validations_completed",
        "same_attempt_pass_recovery_authorized",
        "canonical_self_sha256",
    ):
        raise FormalPrebuildAuthorityV94Error(
            "Pass terminal validation pending receipt schema drift"
        )
    _verify_self_hash(receipt)
    if (
        receipt["version"] != VERSION
        or receipt["status"] != "PASS_TERMINAL_VALIDATION_PENDING"
        or receipt["attempt_id"] != ATTEMPT_ID
        or receipt["authorization_policy_file_sha256"]
        != _sha256_file(paths.authorization_policy)
        or receipt["authorization_policy_canonical_sha256"]
        != policy["canonical_self_sha256"]
        or receipt["claim_canonical_sha256"]
        != claim["canonical_self_sha256"]
        or receipt["key_consumption_receipt_canonical_sha256"]
        != consumption_receipt["canonical_self_sha256"]
        or receipt["preflight_commitment_sha256"]
        != expected_preflight_commitment_sha256
        or receipt["result_canonical_sha256"]
        != result["canonical_self_sha256"]
        or type(receipt["continuation_validations_completed"]) is not int
        or receipt["continuation_validations_completed"] != 2
        or receipt["same_attempt_pass_recovery_authorized"] is not False
    ):
        raise FormalPrebuildAuthorityV94Error(
            "Pass terminal validation pending receipt context drift"
        )
    return receipt


def _build_pass_terminal_validation_completion(
    *,
    policy: Mapping[str, Any],
    paths: RuntimePaths,
    claim: Mapping[str, Any],
    consumption_receipt: Mapping[str, Any],
    preflight_commitment_sha256: str,
    result: Mapping[str, Any],
    pending_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return _with_self_hash({
        "version": VERSION,
        "status": "PASS_TERMINAL_VALIDATION_COMPLETED",
        "attempt_id": ATTEMPT_ID,
        "authorization_policy_file_sha256": _sha256_file(
            paths.authorization_policy
        ),
        "authorization_policy_canonical_sha256": policy[
            "canonical_self_sha256"
        ],
        "claim_canonical_sha256": claim["canonical_self_sha256"],
        "key_consumption_receipt_canonical_sha256": consumption_receipt[
            "canonical_self_sha256"
        ],
        "preflight_commitment_sha256": preflight_commitment_sha256,
        "result_canonical_sha256": result["canonical_self_sha256"],
        "result_file_sha256": _sha256_file(paths.result),
        "pending_receipt_canonical_sha256": pending_receipt[
            "canonical_self_sha256"
        ],
        "continuation_validations_completed": 3,
        "same_attempt_pass_recovery_authorized": True,
    })


def _load_and_validate_pass_terminal_validation_completion(
    policy: Mapping[str, Any],
    *,
    paths: RuntimePaths,
    claim: Mapping[str, Any],
    consumption_receipt: Mapping[str, Any],
    expected_preflight_commitment_sha256: str,
    result: Mapping[str, Any],
    pending_receipt: Mapping[str, Any],
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    observed_path = (
        receipt_path
        if receipt_path is not None
        else paths.pass_terminal_validation_completion
    )
    try:
        receipt = json.loads(
            observed_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPrebuildAuthorityV94Error(
            "Pass terminal validation completion receipt drift"
        ) from error
    if type(receipt) is not dict or tuple(receipt) != (
        "version",
        "status",
        "attempt_id",
        "authorization_policy_file_sha256",
        "authorization_policy_canonical_sha256",
        "claim_canonical_sha256",
        "key_consumption_receipt_canonical_sha256",
        "preflight_commitment_sha256",
        "result_canonical_sha256",
        "result_file_sha256",
        "pending_receipt_canonical_sha256",
        "continuation_validations_completed",
        "same_attempt_pass_recovery_authorized",
        "canonical_self_sha256",
    ):
        raise FormalPrebuildAuthorityV94Error(
            "Pass terminal validation completion receipt schema drift"
        )
    _verify_self_hash(receipt)
    if (
        receipt["version"] != VERSION
        or receipt["status"] != "PASS_TERMINAL_VALIDATION_COMPLETED"
        or receipt["attempt_id"] != ATTEMPT_ID
        or receipt["authorization_policy_file_sha256"]
        != _sha256_file(paths.authorization_policy)
        or receipt["authorization_policy_canonical_sha256"]
        != policy["canonical_self_sha256"]
        or receipt["claim_canonical_sha256"]
        != claim["canonical_self_sha256"]
        or receipt["key_consumption_receipt_canonical_sha256"]
        != consumption_receipt["canonical_self_sha256"]
        or receipt["preflight_commitment_sha256"]
        != expected_preflight_commitment_sha256
        or receipt["result_canonical_sha256"]
        != result["canonical_self_sha256"]
        or receipt["result_file_sha256"] != _sha256_file(paths.result)
        or receipt["pending_receipt_canonical_sha256"]
        != pending_receipt["canonical_self_sha256"]
        or type(receipt["continuation_validations_completed"]) is not int
        or receipt["continuation_validations_completed"] != 3
        or receipt["same_attempt_pass_recovery_authorized"] is not True
    ):
        raise FormalPrebuildAuthorityV94Error(
            "Pass terminal validation completion receipt context drift"
        )
    return receipt


def _input_commitments(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "train_prepared": _json_value(inputs["train_prepared"].commitment),
        "development_prepared": _json_value(
            inputs["development_prepared"].commitment
        ),
        "train_labels": _json_value(inputs["train_labels"].commitment),
        "development_labels": _json_value(
            inputs["development_labels"].commitment
        ),
    }


def _build_result(
    *,
    policy: Mapping[str, Any],
    claim: Mapping[str, Any],
    consumption_receipt: Mapping[str, Any],
    preflight: Mapping[str, Any],
    inputs: Mapping[str, Any],
    metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    passed = comparison["all_gates_passed"] is True
    return _with_self_hash({
        "version": VERSION,
        "status": (
            "PASSED_PREBUILD_SHORTCUT_GATE"
            if passed
            else "FAILED_PREBUILD_SHORTCUT_GATE"
        ),
        "attempt_id": ATTEMPT_ID,
        "authorization_policy_canonical_sha256": policy[
            "canonical_self_sha256"
        ],
        "implementation_commit": policy["implementation_binding"][
            "implementation_commit"
        ],
        "claim_canonical_sha256": claim["canonical_self_sha256"],
        "key_issuance_claim_canonical_sha256": policy["input_binding"][
            "key_issuance_claim_canonical_sha256"
        ],
        "key_issuance_receipt_canonical_sha256": policy["input_binding"][
            "key_issuance_receipt_canonical_sha256"
        ],
        "key_consumption_receipt_canonical_sha256": consumption_receipt[
            "canonical_self_sha256"
        ],
        "source_closure": _json_value(preflight["source_closure"]),
        "schedule_pair_receipt": _json_value(preflight["pair_receipt"]),
        "input_commitments": _input_commitments(inputs),
        "truth_access": {
            "train_reads": 1,
            "development_reads": 1,
            "audit_a_reads": 0,
            "audit_b_reads": 0,
            "join_after_matrix_freeze": True,
        },
        "metrics": _json_value(metrics),
        "decision": {
            "comparisons": _json_value(comparison["comparisons"]),
            "claim": implementation_v94.PASS_CLAIM if passed else None,
            "method_root_build_eligible": passed,
            "authorizes_method_root_build": False,
            "authorizes_training": False,
        },
        "time_index_continuation": {
            "eligible": passed,
            "type": (
                "private_consumed_time_key_for_method_root_only"
                if passed
                else None
            ),
            "private_path": (
                f"{PRIVATE_ROOT_RELATIVE}/{CONSUMED_KEY_NAME}"
                if passed
                else None
            ),
            "commitment_sha256": (
                policy["input_binding"]["time_key_commitment_sha256"]
                if passed
                else None
            ),
            "gate_rerun_authorized": False,
            "method_root_build_authorized": False,
        },
    })


def validate_public_result(result: Mapping[str, Any]) -> None:
    if type(result) is not dict or tuple(result) != (
        "version",
        "status",
        "attempt_id",
        "authorization_policy_canonical_sha256",
        "implementation_commit",
        "claim_canonical_sha256",
        "key_issuance_claim_canonical_sha256",
        "key_issuance_receipt_canonical_sha256",
        "key_consumption_receipt_canonical_sha256",
        "source_closure",
        "schedule_pair_receipt",
        "input_commitments",
        "truth_access",
        "metrics",
        "decision",
        "time_index_continuation",
        "canonical_self_sha256",
    ):
        raise FormalPrebuildAuthorityV94Error("Public result schema drift")
    _verify_self_hash(result)
    passed = result["status"] == "PASSED_PREBUILD_SHORTCUT_GATE"
    comparison = implementation_v94._compare_formal_gates(
        result["metrics"],
        implementation_v94.load_formal_policy(),
    )
    if (
        result["version"] != VERSION
        or result["status"] not in {
            "PASSED_PREBUILD_SHORTCUT_GATE",
            "FAILED_PREBUILD_SHORTCUT_GATE",
        }
        or result["attempt_id"] != ATTEMPT_ID
        or type(result["implementation_commit"]) is not str
        or len(result["implementation_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in result["implementation_commit"]
        )
        or not all(
            _is_sha256(result[field])
            for field in (
                "authorization_policy_canonical_sha256",
                "claim_canonical_sha256",
                "key_issuance_claim_canonical_sha256",
                "key_issuance_receipt_canonical_sha256",
                "key_consumption_receipt_canonical_sha256",
            )
        )
        or not _exact_json_equal(result["truth_access"], {
            "train_reads": 1,
            "development_reads": 1,
            "audit_a_reads": 0,
            "audit_b_reads": 0,
            "join_after_matrix_freeze": True,
        })
        or result["decision"]["method_root_build_eligible"] is not passed
        or result["decision"]["authorizes_method_root_build"] is not False
        or result["decision"]["authorizes_training"] is not False
        or result["decision"]["comparisons"] != comparison["comparisons"]
        or comparison["all_gates_passed"] is not passed
        or (result["decision"]["claim"] == implementation_v94.PASS_CLAIM)
        is not passed
        or not _exact_json_equal(result["time_index_continuation"], {
            "eligible": passed,
            "type": (
                "private_consumed_time_key_for_method_root_only"
                if passed
                else None
            ),
            "private_path": (
                f"{PRIVATE_ROOT_RELATIVE}/{CONSUMED_KEY_NAME}"
                if passed
                else None
            ),
            "commitment_sha256": (
                result["input_commitments"]["train_prepared"][
                    "time_key_commitment_sha256"
                ]
                if passed
                else None
            ),
            "gate_rerun_authorized": False,
            "method_root_build_authorized": False,
        })
    ):
        raise FormalPrebuildAuthorityV94Error("Public result value drift")
    forbidden_keys = {
        "time_key_hex",
        "controller_groups_by_world",
        "row_labels",
        "matrix_values",
        "score_vector",
    }
    stack: list[Any] = [result]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            if forbidden_keys & set(value):
                raise FormalPrebuildAuthorityV94Error(
                    "Private or large field leaked into public result"
                )
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)


def _validate_result_context(
    result: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    claim: Mapping[str, Any],
    consumption_receipt: Mapping[str, Any],
) -> None:
    validate_public_result(result)
    input_commitments = result.get("input_commitments")
    if (
        result["authorization_policy_canonical_sha256"]
        != policy["canonical_self_sha256"]
        or result["implementation_commit"]
        != policy["implementation_binding"]["implementation_commit"]
        or result["claim_canonical_sha256"]
        != claim["canonical_self_sha256"]
        or result["key_issuance_claim_canonical_sha256"]
        != policy["input_binding"][
            "key_issuance_claim_canonical_sha256"
        ]
        or result["key_issuance_receipt_canonical_sha256"]
        != policy["input_binding"][
            "key_issuance_receipt_canonical_sha256"
        ]
        or result["key_consumption_receipt_canonical_sha256"]
        != consumption_receipt["canonical_self_sha256"]
        or type(input_commitments) is not dict
        or tuple(input_commitments) != (
            "train_prepared",
            "development_prepared",
            "train_labels",
            "development_labels",
        )
        or any(
            type(input_commitments[name]) is not dict
            for name in input_commitments
        )
        or input_commitments["train_prepared"].get(
            "time_key_commitment_sha256"
        )
        != policy["input_binding"]["time_key_commitment_sha256"]
        or input_commitments["development_prepared"].get(
            "time_key_commitment_sha256"
        )
        != policy["input_binding"]["time_key_commitment_sha256"]
    ):
        raise FormalPrebuildAuthorityV94Error("Public result context drift")


def _terminal_payload(
    *,
    policy: Mapping[str, Any],
    paths: RuntimePaths,
    claim_sha256: str,
    status: str,
    result_sha256: str | None,
    failure_stage: str | None = None,
    error: BaseException | None = None,
    key_consumed_observed: bool | None = None,
    claim_reference_kind: str = "canonical_self_sha256",
    invalidated_result_sha256: str | None = None,
    error_type: str | None = None,
    error_message_sha256: str | None = None,
    cleanup_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completed = status in {
        "PASSED_PREBUILD_SHORTCUT_GATE",
        "FAILED_PREBUILD_SHORTCUT_GATE",
    }
    consumption_sha256 = (
        _sha256_file(paths.consumption_receipt)
        if paths.consumption_receipt.is_file()
        else None
    )
    key_consumed = (
        paths.consumption_receipt.is_file()
        if key_consumed_observed is None
        else key_consumed_observed
    )
    if completed:
        truth_access: dict[str, Any] = {
            "exact": True,
            "train_reads": 1,
            "development_reads": 1,
            "audit_a_reads": 0,
            "audit_b_reads": 0,
        }
    elif failure_stage in {
        "formal_launch_publication",
        "issuance_receipt_validation",
        "truth_free_preflight",
        "consume_time_key",
    } or (
        failure_stage == "interrupted_after_claim_without_result"
        and not key_consumed
    ):
        truth_access = {
            "exact": True,
            "train_reads": 0,
            "development_reads": 0,
            "audit_a_reads": 0,
            "audit_b_reads": 0,
        }
    elif failure_stage in {
        "fit_fixed_probe_and_bootstrap",
        "compare_frozen_gates",
        "validate_pass_continuation_before_result",
        "public_result_construction",
        "public_result_context_validation",
        "validate_pass_continuation_after_result_context",
        "pass_terminal_validation_pending_publication",
        "result_publication",
        "validate_pass_continuation_before_terminal",
        "pass_terminal_validation_completion_publication",
        "output_allowlist_validation",
        "recovery_output_allowlist_validation",
        "terminal_publication",
        "post_result_continuation_validation",
    }:
        truth_access = {
            "exact": True,
            "train_reads": 1,
            "development_reads": 1,
            "audit_a_reads": 0,
            "audit_b_reads": 0,
        }
    elif failure_stage in {
        "recovery_state_validation",
        "interrupted_after_claim_without_result",
    } and key_consumed:
        truth_access = {
            "exact": False,
            "train_reads": None,
            "development_reads": None,
            "audit_a_reads": 0,
            "audit_b_reads": 0,
        }
    else:
        truth_access = {
            "exact": False,
            "train_reads": None,
            "development_reads": None,
            "audit_a_reads": 0,
            "audit_b_reads": 0,
        }
    if completed or failure_stage in {
        "compare_frozen_gates",
        "validate_pass_continuation_before_result",
        "public_result_construction",
        "public_result_context_validation",
        "validate_pass_continuation_after_result_context",
        "pass_terminal_validation_pending_publication",
        "result_publication",
        "validate_pass_continuation_before_terminal",
        "pass_terminal_validation_completion_publication",
        "output_allowlist_validation",
        "recovery_output_allowlist_validation",
        "terminal_publication",
        "post_result_continuation_validation",
    }:
        execution_reach = {
            "matrices_constructed": True,
            "truth_join_completed": True,
            "fixed_models_fit_completed": True,
            "bootstrap_completed": True,
        }
    elif failure_stage == "fit_fixed_probe_and_bootstrap":
        execution_reach = {
            "matrices_constructed": True,
            "truth_join_completed": True,
            "fixed_models_fit_completed": None,
            "bootstrap_completed": None,
        }
    elif failure_stage == "assemble_formal_inputs":
        execution_reach = {
            "matrices_constructed": None,
            "truth_join_completed": None,
            "fixed_models_fit_completed": False,
            "bootstrap_completed": False,
        }
    elif failure_stage in {
        "recovery_state_validation",
        "interrupted_after_claim_without_result",
    } and key_consumed:
        execution_reach = {
            "matrices_constructed": None,
            "truth_join_completed": None,
            "fixed_models_fit_completed": None,
            "bootstrap_completed": None,
        }
    else:
        execution_reach = {
            "matrices_constructed": False,
            "truth_join_completed": False,
            "fixed_models_fit_completed": False,
            "bootstrap_completed": False,
        }
    payload: dict[str, Any] = {
        "version": VERSION,
        "status": status,
        "attempt_id": ATTEMPT_ID,
        "authorization_policy_file_sha256": _sha256_file(
            paths.authorization_policy
        ),
        "authorization_policy_canonical_sha256": policy[
            "canonical_self_sha256"
        ],
        "implementation_commit": policy["implementation_binding"][
            "implementation_commit"
        ],
        "implementation_tree": policy["implementation_binding"][
            "implementation_tree"
        ],
        "implementation_policy_sha256": implementation_v94.POLICY_SHA256,
        "authority_script_sha256": next(
            record["sha256"]
            for record in policy["implementation_binding"]["source_files"]
            if record["path"] == AUTHORITY_SCRIPT_RELATIVE
        ),
        "unique_command_sha256": UNIQUE_COMMAND_SHA256,
        "output_root": OUTPUT_ROOT_RELATIVE,
        "claim_reference": {
            "kind": claim_reference_kind,
            "sha256": claim_sha256,
        },
        "time_key_commitment_sha256": policy["input_binding"][
            "time_key_commitment_sha256"
        ],
        "key_issuance_claim_canonical_sha256": policy["input_binding"][
            "key_issuance_claim_canonical_sha256"
        ],
        "key_issuance_receipt_canonical_sha256": policy["input_binding"][
            "key_issuance_receipt_canonical_sha256"
        ],
        "key_consumed": key_consumed,
        "key_consumption_receipt_file_sha256": consumption_sha256,
        "mechanical_failure_receipt_canonical_sha256": (
            json.loads(
                paths.mechanical_failure_receipt.read_text(encoding="utf-8")
            )["canonical_self_sha256"]
            if paths.mechanical_failure_receipt.is_file()
            else None
        ),
        "pass_terminal_validation_pending_file_sha256": (
            _sha256_file(paths.pass_terminal_validation_pending)
            if paths.pass_terminal_validation_pending.is_file()
            else None
        ),
        "pass_terminal_validation_completion_file_sha256": (
            _sha256_file(paths.pass_terminal_validation_completion)
            if paths.pass_terminal_validation_completion.is_file()
            else None
        ),
        "result_file_sha256": result_sha256,
        "scientific_result_valid": completed,
        "invalidated_result_sha256": invalidated_result_sha256,
        "method_root_build_eligible": (
            status == "PASSED_PREBUILD_SHORTCUT_GATE"
        ),
        "continuation_validated_at_terminal_publication": (
            status == "PASSED_PREBUILD_SHORTCUT_GATE"
        ),
        "failure_or_completion_stage": failure_stage or "terminal_publication",
        "truth_access": truth_access,
        "execution_reach": execution_reach,
        "error_type": (
            error_type
            if error_type is not None
            else type(error).__name__ if error is not None else None
        ),
        "error_message_sha256": (
            error_message_sha256
            if error_message_sha256 is not None
            else (
                _sha256_bytes(str(error).encode("utf-8"))
                if error is not None
                else None
            )
        ),
        "same_attempt_reusable": False,
        "large_failed_payloads_retained": (
            False
            if cleanup_summary is None
            else cleanup_summary.get(
                "unexpected_output_artifacts_retained"
            ) is True
        ),
        "cleanup_summary": (
            _json_value(cleanup_summary)
            if cleanup_summary is not None
            else {
                "result_building_exists": paths.result_building.exists(),
                "raw_matrix_or_score_payloads_written": False,
            }
        ),
    }
    return _with_self_hash(payload)


def _delete_private_keys(paths: RuntimePaths) -> None:
    for path in (paths.unconsumed_key, paths.consumed_key):
        if path.exists():
            path.unlink()


def _output_root_entry_names(paths: RuntimePaths) -> tuple[str, ...]:
    if not paths.output_root.exists():
        return ()
    return tuple(sorted(
        (entry.name for entry in paths.output_root.iterdir()),
        key=lambda value: value.encode("utf-8"),
    ))


def _validate_output_root(
    paths: RuntimePaths, *, allowed_names: set[str]
) -> None:
    observed = _output_root_entry_names(paths)
    if set(observed) != allowed_names or len(observed) != len(allowed_names):
        raise FormalPrebuildAuthorityV94Error(
            "Formal output allowlist drift"
        )
    for name in observed:
        path = paths.output_root / name
        if not path.is_file() or path.is_symlink():
            raise FormalPrebuildAuthorityV94Error(
                "Formal output allowlist drift"
            )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _cleanup_failed_attempt_artifacts(paths: RuntimePaths) -> None:
    private_building_paths = (
        paths.claim.with_name(f"{paths.claim.name}.building"),
        paths.consumption_receipt.with_name(
            f"{paths.consumption_receipt.name}.building"
        ),
        paths.mechanical_failure_receipt.with_name(
            f"{paths.mechanical_failure_receipt.name}.building"
        ),
        paths.pass_terminal_validation_pending.with_name(
            f"{paths.pass_terminal_validation_pending.name}.building"
        ),
        paths.pass_terminal_validation_completion.with_name(
            f"{paths.pass_terminal_validation_completion.name}.building"
        ),
        paths.pass_terminal_validation_revalidating.with_name(
            f"{paths.pass_terminal_validation_revalidating.name}.building"
        ),
    )
    for path in private_building_paths:
        if path.exists():
            _remove_path(path)
    if paths.pass_terminal_validation_revalidating.exists():
        _remove_path(paths.pass_terminal_validation_revalidating)
    if paths.output_root.exists():
        for path in tuple(paths.output_root.iterdir()):
            if path.name in {LAUNCH_NAME, TERMINAL_NAME} and path.is_file():
                continue
            _remove_path(path)
    retained = set(_output_root_entry_names(paths))
    if not retained.issubset({LAUNCH_NAME, TERMINAL_NAME}):
        raise FormalPrebuildAuthorityV94Error(
            "Failed-output cleanup drift"
        )


def _build_mechanical_failure_receipt(
    *,
    policy: Mapping[str, Any],
    paths: RuntimePaths,
    claim_sha256: str,
    claim_reference_kind: str,
    stage: str,
    error: BaseException,
    key_consumed_observed: bool,
    invalidated_result_sha256: str | None,
) -> dict[str, Any]:
    output_entry_names = _output_root_entry_names(paths)
    return _with_self_hash({
        "version": VERSION,
        "status": "MECHANICAL_FAILURE_FACTS_FROZEN_ATTEMPT_CLOSED",
        "attempt_id": ATTEMPT_ID,
        "authorization_policy_file_sha256": _sha256_file(
            paths.authorization_policy
        ),
        "authorization_policy_canonical_sha256": policy[
            "canonical_self_sha256"
        ],
        "claim_reference": {
            "kind": claim_reference_kind,
            "sha256": claim_sha256,
        },
        "failure_stage": stage,
        "error_type": type(error).__name__,
        "error_message_sha256": _sha256_bytes(
            str(error).encode("utf-8")
        ),
        "key_consumed_observed": key_consumed_observed,
        "invalidated_result_sha256": invalidated_result_sha256,
        "output_root_entry_count_before_cleanup": len(output_entry_names),
        "output_root_entry_names_before_cleanup_sha256": (
            _canonical_sha256(output_entry_names)
        ),
        "same_attempt_reusable": False,
    })


def _load_and_validate_mechanical_failure_receipt(
    policy: Mapping[str, Any], *, paths: RuntimePaths
) -> dict[str, Any]:
    try:
        receipt = json.loads(
            paths.mechanical_failure_receipt.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPrebuildAuthorityV94Error(
            "Mechanical failure receipt drift"
        ) from error
    if type(receipt) is not dict or tuple(receipt) != (
        "version",
        "status",
        "attempt_id",
        "authorization_policy_file_sha256",
        "authorization_policy_canonical_sha256",
        "claim_reference",
        "failure_stage",
        "error_type",
        "error_message_sha256",
        "key_consumed_observed",
        "invalidated_result_sha256",
        "output_root_entry_count_before_cleanup",
        "output_root_entry_names_before_cleanup_sha256",
        "same_attempt_reusable",
        "canonical_self_sha256",
    ):
        raise FormalPrebuildAuthorityV94Error(
            "Mechanical failure receipt schema drift"
        )
    _verify_self_hash(receipt)
    claim_reference = receipt["claim_reference"]
    invalidated_result_sha256 = receipt["invalidated_result_sha256"]
    if (
        receipt["version"] != VERSION
        or receipt["status"]
        != "MECHANICAL_FAILURE_FACTS_FROZEN_ATTEMPT_CLOSED"
        or receipt["attempt_id"] != ATTEMPT_ID
        or receipt["authorization_policy_file_sha256"]
        != _sha256_file(paths.authorization_policy)
        or receipt["authorization_policy_canonical_sha256"]
        != policy["canonical_self_sha256"]
        or type(claim_reference) is not dict
        or tuple(claim_reference) != ("kind", "sha256")
        or claim_reference["kind"]
        not in {"canonical_self_sha256", "claim_file_sha256"}
        or not _is_sha256(claim_reference["sha256"])
        or type(receipt["failure_stage"]) is not str
        or not receipt["failure_stage"]
        or type(receipt["error_type"]) is not str
        or not receipt["error_type"]
        or not _is_sha256(receipt["error_message_sha256"])
        or type(receipt["key_consumed_observed"]) is not bool
        or (
            invalidated_result_sha256 is not None
            and not _is_sha256(invalidated_result_sha256)
        )
        or type(receipt["output_root_entry_count_before_cleanup"]) is not int
        or receipt["output_root_entry_count_before_cleanup"] < 0
        or not _is_sha256(
            receipt["output_root_entry_names_before_cleanup_sha256"]
        )
        or receipt["same_attempt_reusable"] is not False
    ):
        raise FormalPrebuildAuthorityV94Error(
            "Mechanical failure receipt context drift"
        )
    return receipt


def _finalize_mechanical_failure(
    *,
    policy: Mapping[str, Any],
    paths: RuntimePaths,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _delete_private_keys(paths)
    _cleanup_failed_attempt_artifacts(paths)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    status = (
        "POST_CONSUMPTION_MECHANICAL_FAILURE_NO_DATASET_CONCLUSION"
        if receipt["key_consumed_observed"]
        else "PRE_CONSUMPTION_MECHANICAL_FAILURE_ATTEMPT_CLOSED"
    )
    terminal = _terminal_payload(
        policy=policy,
        paths=paths,
        claim_sha256=receipt["claim_reference"]["sha256"],
        status=status,
        result_sha256=None,
        failure_stage=receipt["failure_stage"],
        key_consumed_observed=receipt["key_consumed_observed"],
        claim_reference_kind=receipt["claim_reference"]["kind"],
        invalidated_result_sha256=receipt["invalidated_result_sha256"],
        error_type=receipt["error_type"],
        error_message_sha256=receipt["error_message_sha256"],
        cleanup_summary={
            "output_root_entry_count_before_cleanup": receipt[
                "output_root_entry_count_before_cleanup"
            ],
            "output_root_entry_names_before_cleanup_sha256": receipt[
                "output_root_entry_names_before_cleanup_sha256"
            ],
            "retained_output_artifacts_before_terminal": list(
                _output_root_entry_names(paths)
            ),
            "unexpected_output_artifacts_retained": False,
            "raw_matrix_or_score_payloads_written": None,
        },
    )
    if not paths.terminal.exists():
        _write_new_json(paths.terminal, terminal)
    return terminal


def _publish_mechanical_failure(
    *,
    policy: Mapping[str, Any],
    claim: Mapping[str, Any],
    paths: RuntimePaths,
    stage: str,
    error: BaseException,
    claim_reference_kind: str = "canonical_self_sha256",
) -> None:
    receipt_building = paths.mechanical_failure_receipt.with_name(
        f"{paths.mechanical_failure_receipt.name}.building"
    )
    if paths.mechanical_failure_receipt.exists():
        receipt = _load_and_validate_mechanical_failure_receipt(
            policy,
            paths=paths,
        )
    else:
        receipt = _build_mechanical_failure_receipt(
            policy=policy,
            paths=paths,
            claim_sha256=claim["canonical_self_sha256"],
            claim_reference_kind=claim_reference_kind,
            stage=stage,
            error=error,
            key_consumed_observed=(
                paths.consumption_receipt.is_file()
                or paths.consumed_key.exists()
            ),
            invalidated_result_sha256=(
                _sha256_file(paths.result) if paths.result.is_file() else None
            ),
        )
        if not receipt_building.exists():
            _write_durable_failure_marker(
                receipt_building,
                _with_self_hash({
                    "version": VERSION,
                    "status": "MECHANICAL_FAILURE_PUBLICATION_CLAIMED",
                    "attempt_id": ATTEMPT_ID,
                }),
            )
        _replace_existing_building_with_json(
            paths.mechanical_failure_receipt,
            receipt,
        )
    _finalize_mechanical_failure(
        policy=policy,
        paths=paths,
        receipt=receipt,
    )


def _recover_claimed_attempt(
    *, policy: Mapping[str, Any], paths: RuntimePaths
) -> dict[str, Any]:
    if paths.terminal.exists():
        raise FormalPrebuildAuthorityV94Error("Formal attempt is already terminal")
    terminal_building = paths.terminal.with_name(
        f"{paths.terminal.name}.building"
    )
    if terminal_building.exists():
        _remove_path(terminal_building)
    mechanical_failure_building = paths.mechanical_failure_receipt.with_name(
        f"{paths.mechanical_failure_receipt.name}.building"
    )
    if paths.mechanical_failure_receipt.exists():
        receipt = _load_and_validate_mechanical_failure_receipt(
            policy,
            paths=paths,
        )
        return _finalize_mechanical_failure(
            policy=policy,
            paths=paths,
            receipt=receipt,
        )
    pending_building = paths.pass_terminal_validation_pending.with_name(
        f"{paths.pass_terminal_validation_pending.name}.building"
    )
    completion_building = paths.pass_terminal_validation_completion.with_name(
        f"{paths.pass_terminal_validation_completion.name}.building"
    )
    revalidating_building = (
        paths.pass_terminal_validation_revalidating.with_name(
            f"{paths.pass_terminal_validation_revalidating.name}.building"
        )
    )
    prior_pass_revalidation_interrupted = (
        paths.pass_terminal_validation_revalidating.exists()
        or revalidating_building.exists()
    )
    try:
        claim_file_sha256 = (
            _sha256_file(paths.claim)
            if paths.claim.is_file()
            else "0" * 64
        )
    except OSError:
        claim_file_sha256 = "0" * 64
    claim_file_reference = {
        "canonical_self_sha256": claim_file_sha256,
    }
    if mechanical_failure_building.exists():
        interrupted = RuntimeError(
            "Mechanical failure receipt publication was interrupted"
        )
        _publish_mechanical_failure(
            policy=policy,
            claim=claim_file_reference,
            paths=paths,
            stage="recovery_incomplete_mechanical_failure_receipt",
            error=interrupted,
            claim_reference_kind="claim_file_sha256",
        )
        return json.loads(paths.terminal.read_text(encoding="utf-8"))
    if prior_pass_revalidation_interrupted:
        interrupted = RuntimeError(
            "Pass recovery validation was previously interrupted"
        )
        _publish_mechanical_failure(
            policy=policy,
            claim=claim_file_reference,
            paths=paths,
            stage="recovery_incomplete_pass_revalidation",
            error=interrupted,
            claim_reference_kind="claim_file_sha256",
        )
        return json.loads(paths.terminal.read_text(encoding="utf-8"))
    pass_revalidation_claimed = False
    pass_revalidation_claim_error: BaseException | None = None
    if (
        not prior_pass_revalidation_interrupted
        and paths.result.exists()
        and paths.pass_terminal_validation_completion.exists()
    ):
        try:
            _write_new_json(
                paths.pass_terminal_validation_revalidating,
                _with_self_hash({
                    "version": VERSION,
                    "status": "PASS_RECOVERY_REVALIDATION_CLAIMED",
                    "attempt_id": ATTEMPT_ID,
                }),
            )
            os.replace(
                paths.pass_terminal_validation_completion,
                paths.pass_terminal_validation_revalidating,
            )
            if (
                paths.pass_terminal_validation_completion.exists()
                or not paths.pass_terminal_validation_revalidating.is_file()
            ):
                raise FormalPrebuildAuthorityV94Error(
                    "Pass recovery validation claim drift"
                )
            pass_revalidation_claimed = True
        except Exception as error:
            pass_revalidation_claim_error = error
    if pass_revalidation_claim_error is not None:
        _publish_mechanical_failure(
            policy=policy,
            claim=claim_file_reference,
            paths=paths,
            stage="claim_pass_recovery_validation",
            error=pass_revalidation_claim_error,
            claim_reference_kind="claim_file_sha256",
        )
        return json.loads(paths.terminal.read_text(encoding="utf-8"))
    try:
        claim = _load_and_validate_launch_claim(policy, paths=paths)
    except Exception as error:
        _publish_mechanical_failure(
            policy=policy,
            claim=claim_file_reference,
            paths=paths,
            stage="recovery_state_validation",
            error=error,
            claim_reference_kind="claim_file_sha256",
        )
        return json.loads(paths.terminal.read_text(encoding="utf-8"))
    if paths.result.exists():
        try:
            result = json.loads(paths.result.read_text(encoding="utf-8"))
            recovered_preflight_commitment = _preflight_public_commitment(
                _preflight_materials(policy)
            )
            consumption_receipt = _load_and_validate_consumption_receipt(
                policy,
                paths=paths,
                claim=claim,
                require_valid_key=True,
            )
            _validate_consumption_lineage(
                policy,
                paths=paths,
                claim=claim,
                consumption_receipt=consumption_receipt,
                expected_preflight_commitment_sha256=(
                    recovered_preflight_commitment
                ),
            )
            _validate_result_context(
                result,
                policy=policy,
                claim=claim,
                consumption_receipt=consumption_receipt,
            )
            if result["status"] == "PASSED_PREBUILD_SHORTCUT_GATE":
                if pending_building.exists() or completion_building.exists():
                    raise FormalPrebuildAuthorityV94Error(
                        "Pass terminal validation receipt publication interrupted"
                    )
                pending_receipt = (
                    _load_and_validate_pass_terminal_validation_pending(
                        policy,
                        paths=paths,
                        claim=claim,
                        consumption_receipt=consumption_receipt,
                        expected_preflight_commitment_sha256=(
                            recovered_preflight_commitment
                        ),
                        result=result,
                    )
                )
                _load_and_validate_pass_terminal_validation_completion(
                    policy,
                    paths=paths,
                    claim=claim,
                    consumption_receipt=consumption_receipt,
                    expected_preflight_commitment_sha256=(
                        recovered_preflight_commitment
                    ),
                    result=result,
                    pending_receipt=pending_receipt,
                    receipt_path=(
                        paths.pass_terminal_validation_revalidating
                        if pass_revalidation_claimed
                        else None
                    ),
                )
                _validate_method_root_continuation(
                    policy,
                    paths=paths,
                    claim=claim,
                    consumption_receipt=consumption_receipt,
                    expected_preflight_commitment_sha256=(
                        recovered_preflight_commitment
                    ),
                )
            elif (
                paths.pass_terminal_validation_pending.exists()
                or paths.pass_terminal_validation_completion.exists()
                or paths.pass_terminal_validation_revalidating.exists()
                or pending_building.exists()
                or completion_building.exists()
                or revalidating_building.exists()
            ):
                raise FormalPrebuildAuthorityV94Error(
                    "Failed result has pass terminal validation state"
                )
        except Exception as error:
            _publish_mechanical_failure(
                policy=policy,
                claim=claim,
                paths=paths,
                stage="recovery_state_validation",
                error=error,
            )
            return json.loads(paths.terminal.read_text(encoding="utf-8"))
        if result["status"] != "PASSED_PREBUILD_SHORTCUT_GATE":
            _delete_private_keys(paths)
        try:
            _validate_output_root(
                paths,
                allowed_names={LAUNCH_NAME, RESULT_NAME},
            )
        except Exception as error:
            _publish_mechanical_failure(
                policy=policy,
                claim=claim,
                paths=paths,
                stage="recovery_output_allowlist_validation",
                error=error,
            )
            return json.loads(paths.terminal.read_text(encoding="utf-8"))
        if pass_revalidation_claimed:
            try:
                os.replace(
                    paths.pass_terminal_validation_revalidating,
                    paths.pass_terminal_validation_completion,
                )
                if (
                    paths.pass_terminal_validation_revalidating.exists()
                    or not paths.pass_terminal_validation_completion.is_file()
                ):
                    raise FormalPrebuildAuthorityV94Error(
                        "Pass recovery validation completion drift"
                    )
            except Exception as error:
                _publish_mechanical_failure(
                    policy=policy,
                    claim=claim,
                    paths=paths,
                    stage="complete_pass_recovery_validation",
                    error=error,
                )
                return json.loads(paths.terminal.read_text(encoding="utf-8"))
        terminal = _terminal_payload(
            policy=policy,
            paths=paths,
            claim_sha256=claim["canonical_self_sha256"],
            status=result["status"],
            result_sha256=_sha256_file(paths.result),
        )
        _write_new_json(paths.terminal, terminal)
        return terminal
    if (
        paths.pass_terminal_validation_pending.exists()
        or paths.pass_terminal_validation_completion.exists()
        or pending_building.exists()
        or completion_building.exists()
        or paths.pass_terminal_validation_revalidating.exists()
        or revalidating_building.exists()
    ):
        interrupted = RuntimeError(
            "Pass terminal validation state exists without a result"
        )
        _publish_mechanical_failure(
            policy=policy,
            claim=claim,
            paths=paths,
            stage="recovery_terminal_validation_state",
            error=interrupted,
        )
        return json.loads(paths.terminal.read_text(encoding="utf-8"))
    interrupted = RuntimeError("Attempt interrupted after exclusive claim")
    _publish_mechanical_failure(
        policy=policy,
        claim=claim,
        paths=paths,
        stage="interrupted_after_claim_without_result",
        error=interrupted,
    )
    return json.loads(paths.terminal.read_text(encoding="utf-8"))


def run_once(*, root: Path = ROOT) -> dict[str, Any]:
    paths = runtime_paths(root)
    claim_building = paths.claim.with_name(f"{paths.claim.name}.building")
    if not paths.claim.exists() and claim_building.exists():
        _remove_path(claim_building)
    policy = load_authorization_policy(root=root)
    _validate_policy_commit(
        policy,
        root=root,
        allow_attempt_output=paths.claim.exists(),
    )
    if paths.claim.exists():
        return _recover_claimed_attempt(policy=policy, paths=paths)
    _validate_fresh_attempt_paths(paths=paths)
    try:
        claim = _claim_formal_launch(
            policy,
            paths=paths,
        )
    except Exception as launch_error:
        if paths.claim.exists():
            claim = json.loads(paths.claim.read_text(encoding="utf-8"))
            _verify_self_hash(claim)
            _publish_mechanical_failure(
                policy=policy,
                claim=claim,
                paths=paths,
                stage="formal_launch_publication",
                error=launch_error,
            )
        raise
    stage = "issuance_receipt_validation"
    key = b""
    try:
        validate_issuance_receipt(
            policy,
            paths=paths,
            verify_unconsumed_key=False,
        )
        if (
            not paths.unconsumed_key.is_file()
            or paths.unconsumed_key.stat().st_size != 32
        ):
            raise FormalPrebuildAuthorityV94Error("Issued private key drift")
        stage = "truth_free_preflight"
        print(json.dumps({"event": "v9_4_prebuild_preflight_start"}, ensure_ascii=False))
        preflight = _preflight_materials(policy)
        preflight_commitment = _preflight_public_commitment(preflight)
        stage = "consume_time_key"
        key, consumption_receipt = _consume_claimed_key(
            policy,
            paths=paths,
            claim=claim,
            preflight_commitment_sha256=preflight_commitment,
        )
        stage = "assemble_formal_inputs"
        print(json.dumps({"event": stage}, ensure_ascii=False))
        time_key_hex = key.hex()
        inputs = implementation_v94._assemble_formal_inputs_after_authorization(
            train_schedule=preflight["train_schedule"],
            development_schedule=preflight["development_schedule"],
            noise_signature_set=preflight["noise_signatures"],
            time_key_hex=time_key_hex,
            expected_noise_signature_rows_sha256=policy["input_binding"][
                "noise_signature_rows_sha256"
            ],
            expected_noise_signature_set_commitment_sha256=policy[
                "input_binding"
            ]["noise_signature_set_commitment_sha256"],
            expected_time_key_commitment_sha256=policy["input_binding"][
                "time_key_commitment_sha256"
            ],
        )
        stage = "fit_fixed_probe_and_bootstrap"
        print(json.dumps({"event": stage}, ensure_ascii=False))
        metrics = core_v94._evaluate_family(
            train={
                implementation_v94.FORMAL_VIEW: inputs["train_prepared"].matrix
            },
            development={
                implementation_v94.FORMAL_VIEW: inputs[
                    "development_prepared"
                ].matrix
            },
            train_labels=inputs["train_labels"].values,
            development_labels=inputs["development_labels"].values,
            train_label_row_keys=inputs["train_labels"].row_keys,
            development_label_row_keys=inputs["development_labels"].row_keys,
            policy=preflight["implementation_policy"],
            average_precision_baseline=implementation_v94.FORMAL_AP_BASELINE,
            bootstrap=True,
        )
        stage = "compare_frozen_gates"
        comparison = implementation_v94._compare_formal_gates(
            metrics,
            preflight["implementation_policy"],
        )
        passed = comparison["all_gates_passed"] is True
        if passed:
            stage = "validate_pass_continuation_before_result"
            _validate_method_root_continuation(
                policy,
                paths=paths,
                claim=claim,
                consumption_receipt=consumption_receipt,
                expected_preflight_commitment_sha256=preflight_commitment,
            )
        stage = "public_result_construction"
        result = _build_result(
            policy=policy,
            claim=claim,
            consumption_receipt=consumption_receipt,
            preflight=preflight,
            inputs=inputs,
            metrics=metrics,
            comparison=comparison,
        )
        stage = "public_result_context_validation"
        _validate_result_context(
            result,
            policy=policy,
            claim=claim,
            consumption_receipt=consumption_receipt,
        )
        if passed:
            stage = "validate_pass_continuation_after_result_context"
            _validate_method_root_continuation(
                policy,
                paths=paths,
                claim=claim,
                consumption_receipt=consumption_receipt,
                expected_preflight_commitment_sha256=preflight_commitment,
            )
            stage = "pass_terminal_validation_pending_publication"
            pending_receipt = _build_pass_terminal_validation_pending(
                policy=policy,
                paths=paths,
                claim=claim,
                consumption_receipt=consumption_receipt,
                preflight_commitment_sha256=preflight_commitment,
                result=result,
            )
            _write_new_json(
                paths.pass_terminal_validation_pending,
                pending_receipt,
            )
        stage = "result_publication"
        _write_new_json(paths.result, result)
        if result["status"] == "FAILED_PREBUILD_SHORTCUT_GATE":
            _delete_private_keys(paths)
        else:
            stage = "validate_pass_continuation_before_terminal"
            _validate_method_root_continuation(
                policy,
                paths=paths,
                claim=claim,
                consumption_receipt=consumption_receipt,
                expected_preflight_commitment_sha256=preflight_commitment,
            )
            stage = "pass_terminal_validation_completion_publication"
            completion_receipt = (
                _build_pass_terminal_validation_completion(
                    policy=policy,
                    paths=paths,
                    claim=claim,
                    consumption_receipt=consumption_receipt,
                    preflight_commitment_sha256=preflight_commitment,
                    result=result,
                    pending_receipt=pending_receipt,
                )
            )
            _write_new_json(
                paths.pass_terminal_validation_completion,
                completion_receipt,
            )
        stage = "output_allowlist_validation"
        _validate_output_root(
            paths,
            allowed_names={LAUNCH_NAME, RESULT_NAME},
        )
        stage = "terminal_publication"
        terminal = _terminal_payload(
            policy=policy,
            paths=paths,
            claim_sha256=claim["canonical_self_sha256"],
            status=result["status"],
            result_sha256=_sha256_file(paths.result),
        )
        _write_new_json(paths.terminal, terminal)
        return result
    except Exception as error:
        mandatory_continuation_stages = {
            "validate_pass_continuation_before_result",
            "validate_pass_continuation_after_result_context",
            "validate_pass_continuation_before_terminal",
        }
        if stage in mandatory_continuation_stages:
            _publish_mechanical_failure(
                policy=policy,
                claim=claim,
                paths=paths,
                stage=stage,
                error=error,
            )
        elif paths.result.exists() and not paths.terminal.exists():
            _recover_claimed_attempt(policy=policy, paths=paths)
        else:
            _publish_mechanical_failure(
                policy=policy,
                claim=claim,
                paths=paths,
                stage=stage,
                error=error,
            )
        raise
    finally:
        key = b""
        if "time_key_hex" in locals():
            time_key_hex = ""


def _print_public_summary(payload: Mapping[str, Any]) -> None:
    allowed = {
        key: payload[key]
        for key in (
            "version",
            "status",
            "attempt_id",
            "implementation_commit",
            "authorization_policy_path",
            "authorization_policy_sha256",
            "time_key_commitment_sha256",
            "key_issuance_claim_canonical_sha256",
            "key_issuance_receipt_canonical_sha256",
            "canonical_self_sha256",
        )
        if key in payload
    }
    print(json.dumps(allowed, ensure_ascii=False, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or consume the single V9.4 prebuild authorization."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze-authorization", action="store_true")
    modes.add_argument("--validate-authorization", action="store_true")
    modes.add_argument("--run-once", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze_authorization:
        _print_public_summary(freeze_authorization())
    elif args.validate_authorization:
        policy = load_authorization_policy()
        _validate_policy_commit(policy, root=ROOT)
        paths = runtime_paths()
        _validate_fresh_issued_state(policy, paths=paths)
        validate_issuance_receipt(
            policy,
            paths=paths,
            verify_unconsumed_key=True,
        )
        _preflight_materials(policy)
        _print_public_summary({
            "version": VERSION,
            "status": "AUTHORIZATION_VALID_NOT_CONSUMED",
            "attempt_id": ATTEMPT_ID,
            "canonical_self_sha256": policy["canonical_self_sha256"],
        })
    else:
        result = run_once()
        _print_public_summary(result)


if __name__ == "__main__":
    main()
