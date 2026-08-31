#!/usr/bin/env python3
"""Freeze the source-bound identity33 projection under the V1 public policy.

The command line remains validation-only.  A future exact-commit one-time
wrapper must own formal execution and publication.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import step28_v13_v1_13_v9_4_1_freeze_identity_projection_v1 as predecessor
import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as experiment_common
import step28_v13_v1_13_v9_4_1_public_projection_common_v1 as common


def render_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def validate_identity_output(
    policy: Mapping[str, Any], identity_root: Path
) -> dict[str, Any]:
    manifest_path = identity_root / "identity_projection_manifest.json"
    manifest = common.load_json(manifest_path)
    claimed = manifest.get("canonical_self_hash")
    body = dict(manifest)
    body.pop("canonical_self_hash", None)
    if not isinstance(claimed, str) or common.canonical_sha256(body) != claimed:
        raise common.PublicProjectionContractError("Identity V2 manifest self-hash drift")
    old_policy = experiment_common.load_policy()
    if (
        manifest.get("status")
        != "FROZEN_LABEL_FREE_IDENTITY33_AND_TRAIN_M1_MAPS_NO_MODEL_TRAINING"
        or manifest.get("policy_canonical_self_hash")
        != policy["canonical_self_hash"]
        or manifest.get("predecessor_policy_canonical_self_hash")
        != old_policy["canonical_self_hash"]
        or manifest.get("total_world_count") != 2000
        or manifest.get("total_row_count") != 756000
        or manifest.get("labels_qrels_membership_or_controller_read") is not False
        or manifest.get("text_or_seller_profile_files_read") is not False
        or manifest.get("audit_truth_read") is not False
        or manifest.get("model_training_or_scoring_performed") is not False
        or manifest.get("training_authorized") is not False
    ):
        raise common.PublicProjectionContractError("Identity V2 lineage drift")
    splits = manifest.get("splits")
    if not isinstance(splits, list) or [row.get("split") for row in splits] != list(
        common.SPLITS
    ):
        raise common.PublicProjectionContractError("Identity V2 split order drift")
    for record in splits:
        predecessor._validate_split_payload(old_policy, identity_root, record)
    expected_paths = {"identity_projection_manifest.json"}
    for split in common.SPLITS:
        expected_paths.update((f"{split}/row_keys.csv", f"{split}/identity33.npy"))
    expected_paths.update(
        f"train/m1_source_row_index_{repeat_id}.npy"
        for repeat_id in old_policy["m1"]["repeat_ids"]
    )
    actual_paths = {
        path.relative_to(identity_root).as_posix()
        for path in identity_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise common.PublicProjectionContractError("Identity V2 file universe drift")
    return manifest


def build_to_temporary(
    policy: Mapping[str, Any], identity_root: Path
) -> dict[str, Any]:
    """Build an unpublished identity projection for an authorized wrapper."""

    if identity_root.exists():
        raise common.PublicProjectionContractError("Identity output root already exists")
    old_policy = experiment_common.load_policy()
    identity_root.mkdir(parents=True)
    try:
        splits = [
            predecessor.build_split(old_policy, split, identity_root)
            for split in common.SPLITS
        ]
        manifest = {
            "step": "step28_v13_v1_13_v9_4_1_identity_projection_v2",
            "status": "FROZEN_LABEL_FREE_IDENTITY33_AND_TRAIN_M1_MAPS_NO_MODEL_TRAINING",
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "predecessor_policy_canonical_self_hash": old_policy[
                "canonical_self_hash"
            ],
            "splits": splits,
            "total_world_count": 2000,
            "total_row_count": 756000,
            "labels_qrels_membership_or_controller_read": False,
            "text_or_seller_profile_files_read": False,
            "audit_truth_read": False,
            "model_training_or_scoring_performed": False,
            "training_authorized": False,
        }
        manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
        render_json(identity_root / "identity_projection_manifest.json", manifest)
        validate_identity_output(policy, identity_root)
    except BaseException:
        shutil.rmtree(identity_root, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-contract",))
    parser.parse_args()
    policy = common.load_policy()
    print(
        json.dumps(
            {
                "status": "PASSED_IDENTITY_V2_CONTRACT_NO_FORMAL_EXECUTION",
                "policy_canonical_self_hash": policy["canonical_self_hash"],
                "formal_projection_executed": False,
                "supervision_or_audit_truth_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
