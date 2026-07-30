#!/usr/bin/env python3
"""Build or verify the immutable Step 28-v13 shortcut-audit lock."""

from __future__ import annotations

import argparse
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import step28_v13_common as dataset_common
import step28_v13_metadata_shortcut_common as shortcut_common


PARENT_POLICY_PATH = (
    "schema/step28_v13_synthetic_chinese_dataset_policy.json"
)
IMPLEMENTATION_DOCUMENT_PATH = (
    "docs/STEP28_V13_METADATA_SHORTCUT_AUDIT_LOCK_20260729.zh.md"
)
SMOKE_MANIFEST_PATH = (
    "reports/step28_synthetic_chinese_dataset/"
    "v13_dev_smoke_v2_20260728/dataset_smoke_v3/"
    "release_manifest.json"
)
DEPENDENCY_PATHS = ("scripts/step28_v13_common.py",)


def build_payload() -> dict[str, Any]:
    policy_path = dataset_common.repo_path(PARENT_POLICY_PATH)
    policy, policy_snapshot = shortcut_common.load_json_snapshot(
        policy_path
    )
    if not isinstance(policy, dict):
        raise shortcut_common.ShortcutAuditError(
            "Parent policy is not an object"
        )
    parent_metadata = policy["metadata_shortcut_audit"]
    contract_path_text = str(policy["contract"]["path"])
    contract_path = dataset_common.repo_path(contract_path_text)
    document_path = dataset_common.repo_path(
        IMPLEMENTATION_DOCUMENT_PATH
    )
    smoke_manifest_path = dataset_common.repo_path(
        SMOKE_MANIFEST_PATH
    )
    contract_snapshot = shortcut_common.snapshot_regular_file(
        contract_path
    )
    document_snapshot = shortcut_common.snapshot_regular_file(
        document_path
    )
    smoke_manifest, smoke_snapshot = (
        shortcut_common.load_json_snapshot(smoke_manifest_path)
    )
    if not isinstance(smoke_manifest, dict):
        raise shortcut_common.ShortcutAuditError(
            "Engineering-smoke release is not an object"
        )
    if (
        policy.get("status") != "DRAFT_SMOKE_ONLY"
        or policy.get("formal_generation_enabled") is not False
        or smoke_manifest.get("status")
        != "DEVELOPMENT_SMOKE_COMPLETE_NOT_SCIENTIFIC_EVIDENCE"
    ):
        raise shortcut_common.ShortcutAuditError(
            "Current Step28 parent/smoke status no longer matches "
            "the blocked implementation lock"
        )
    source_files = shortcut_common._source_closure_hashes()
    payload = {
        "version": shortcut_common.LOCK_VERSION,
        "status": (
            "CORE_IMPLEMENTATION_LOCKED_"
            "FORMAL_EXECUTION_BLOCKED"
        ),
        "objective": (
            "Freeze an exact, label-isolated implementation of the "
            "pre-registered 14-feature metadata-shortcut audit "
            "without executing it on development smoke data."
        ),
        "parent_contract": {
            "path": contract_path_text,
            "sha256": contract_snapshot.sha256,
        },
        "parent_policy": {
            "path": PARENT_POLICY_PATH,
            "sha256": policy_snapshot.sha256,
        },
        "implementation_document": {
            "path": IMPLEMENTATION_DOCUMENT_PATH,
            "sha256": document_snapshot.sha256,
        },
        "parent_metadata_contract_sha256": (
            dataset_common.canonical_sha256(parent_metadata)
        ),
        "engineering_smoke_release": {
            "path": SMOKE_MANIFEST_PATH,
            "sha256": smoke_snapshot.sha256,
            "content_sha256": smoke_manifest[
                "canonical_self_hash"
            ],
            "status": smoke_manifest["status"],
            "numeric_shortcut_execution_forbidden": True,
            "reason": (
                "The parent contract forbids feature-by-label and "
                "AUC outputs for smoke; development has only three "
                "worlds and cannot satisfy fixed five-fold OOF."
            ),
        },
        "splits": list(shortcut_common.SPLITS),
        "pretrain_audit_splits": list(
            shortcut_common.PRETRAIN_AUDIT_SPLITS
        ),
        "formal_run_id": policy["modes"]["formal"]["run_id"],
        "formal_world_counts": policy["modes"]["formal"][
            "world_counts"
        ],
        "seller_feature_order": list(
            shortcut_common.SELLER_FEATURES
        ),
        "pair_feature_order": list(shortcut_common.PAIR_FEATURES),
        "projection_schema": list(
            shortcut_common.PROJECTION_FIELDS
        ),
        "label_schema": list(shortcut_common.LABEL_FIELDS),
        "private_oof_schema": list(shortcut_common.OOF_FIELDS),
        "float_serialization": ".12f",
        "bootstrap_draw_hash_dtype": ">u8",
        "projector": {
            "input_basenames": {
                "candidate_pairs": "candidate_pairs.csv",
                "redacted_items": "redacted_items.jsonl",
                "history_item_index": "history_item_index.csv",
            },
            "all_other_successful_opens_forbidden": True,
            "label_access_forbidden": True,
        },
        "label_sealer": {
            "input_basenames": {
                "candidate_pairs": "candidate_pairs.csv",
                "controller_membership": (
                    "controller_membership.csv"
                ),
            },
            "formula": (
                "int(controller(left)==controller(right))"
            ),
            "all_other_successful_opens_forbidden": True,
            "nuisance_projection_access_forbidden": True,
        },
        "label_formula_validator": {
            "input_basenames": {
                "candidate_pairs": "candidate_pairs.csv",
                "controller_membership": (
                    "controller_membership.csv"
                ),
                "labels": shortcut_common.LABEL_FILENAME,
                "label_manifest": (
                    shortcut_common.LABEL_MANIFEST_FILENAME
                ),
            },
            "alternative_derivation_required": True,
            "class_counts_withheld": True,
        },
        "audit_runner": {
            "input_basenames": {
                "projection": shortcut_common.PROJECTION_FILENAME,
                "labels": shortcut_common.LABEL_FILENAME,
                "label_formula_receipt": (
                    shortcut_common.LABEL_VALIDATION_FILENAME
                ),
            },
            "dataset_parent_mount_forbidden": True,
            "oracle_text_identity_m0_adapter_mount_forbidden": True,
            "train_development_combination_forbidden": True,
            "audit_a_b_only_at_sequential_unseal": True,
        },
        "statistics": {
            "models": parent_metadata["models"],
            "world_grouped_folds": parent_metadata[
                "world_grouped_folds"
            ],
            "fold_random_seed": parent_metadata["random_seed"],
            "bootstrap": parent_metadata["bootstrap"],
            "point_maximum": parent_metadata[
                "auc_symmetric_point_maximum"
            ],
            "bootstrap_95_upper_maximum": parent_metadata[
                "auc_symmetric_world_bootstrap_95_upper_maximum"
            ],
            "numpy_version": parent_metadata["numpy_version"],
            "scikit_learn_version": parent_metadata[
                "sklearn_version"
            ],
        },
        "formal_execution": {
            "enabled": False,
            "formal_release_content_sha256": None,
            "custody_access_manifest_content_sha256": None,
            "execution_environment_content_sha256": None,
            "exact_input_bindings": None,
            "missing_prerequisites": [
                "parent policy status FROZEN and formal_generation_enabled true",
                "frozen power artifact and selected audit world count",
                "one-shot formal structure-key ceremonies and formal dataset release",
                "frozen Identity experiment child policy and custody deployment",
                "separate read-only projection, supervision, and validation capabilities",
                "exact formal root/child/world-set and input-file SHA bindings",
                "externally generated custody deployment and access-log receipts",
                "split-specific train/development/A/B authorization receipts",
                "exact NumPy 2.2.6 and scikit-learn 1.7.1 execution environment",
            ],
            "enablement_rule": (
                "Create a new versioned execution lock after every "
                "prerequisite is independently verified; never edit "
                "this blocked lock in place."
            ),
            "split_authorizations": {
                split: {
                    "authorized": False,
                    "allowed_operations": [
                        "label_sealing",
                        "label_formula_validation",
                        "metadata_shortcut_audit",
                    ],
                    "authorization_receipts": {
                        "label_sealing": None,
                        "label_formula_validation": None,
                        "metadata_shortcut_audit": None,
                    },
                    "required_prior_receipts": [],
                    "required_prior_statuses": (
                        shortcut_common.prior_requirements(split)
                    ),
                }
                for split in shortcut_common.SPLITS
            },
        },
        "claim_boundary": {
            "pass_scope": "PASS_METADATA_SHORTCUT_ONLY",
            "pass_dataset_only_granted": False,
            "mechanism_coverage_validator_included": False,
            "absence_of_all_unknown_shortcuts_proven": False,
        },
        "source_files_sha256": source_files,
        "source_closure_sha256": (
            dataset_common.canonical_sha256(source_files)
        ),
        "dependency_closure_sha256": {
            relative: dataset_common.sha256_file(
                dataset_common.repo_path(relative)
            )
            for relative in DEPENDENCY_PATHS
        },
    }
    return shortcut_common.add_self_hash(payload)


def _write_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    shortcut_common.reject_reparse_components(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    shortcut_common.reject_reparse_components(path.parent)
    if os.path.lexists(dataset_common.filesystem_path(path)):
        raise FileExistsError(
            f"Refusing to overwrite shortcut-audit lock: {path}"
        )
    stage = path.parent / f".staging-{path.name}-{uuid.uuid4().hex}"
    try:
        dataset_common.write_json(stage, payload)
        with open(
            dataset_common.filesystem_path(stage),
            "r+b",
        ) as handle:
            os.fsync(handle.fileno())
        dataset_common.atomic_rename_no_replace(stage, path)
        shortcut_common._fsync_directory(path.parent)
    finally:
        if os.path.exists(dataset_common.filesystem_path(stage)):
            os.unlink(dataset_common.filesystem_path(stage))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=shortcut_common.DEFAULT_LOCK_PATH,
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = build_payload()
    if args.check:
        observed, _snapshot = shortcut_common.load_json_snapshot(
            args.output
        )
        if observed != expected:
            raise shortcut_common.ShortcutAuditError(
                "Shortcut-audit lock payload drift"
            )
        shortcut_common.load_lock(args.output)
        print("Step28-v13 metadata-shortcut lock check passed")
        return
    _write_no_replace(args.output, expected)
    shortcut_common.load_lock(args.output)
    print(f"Wrote immutable shortcut-audit lock: {args.output}")


if __name__ == "__main__":
    main()
