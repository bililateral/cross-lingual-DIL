#!/usr/bin/env python3
"""Cross-bind the base and identity public projections without supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import step28_v13_v1_13_v9_4_1_finalize_base_projection_v1 as base_projection
import step28_v13_v1_13_v9_4_1_freeze_identity_projection_v2 as identity_projection
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


def _verify_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("canonical_self_hash")
    body = dict(value)
    body.pop("canonical_self_hash", None)
    if not isinstance(claimed, str) or common.canonical_sha256(body) != claimed:
        raise common.PublicProjectionContractError(f"{label} self-hash drift")


def cross_bind_row_keys(
    base_root: Path,
    base_manifest: Mapping[str, Any],
    identity_root: Path,
    identity_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    base_by_split = {str(row["split"]): row for row in base_manifest["splits"]}
    identity_by_split = {
        str(row["split"]): row for row in identity_manifest["splits"]
    }
    records = []
    for split in common.SPLITS:
        base_split_path = base_root / str(base_by_split[split]["manifest_file"]["path"])
        base_split = common.load_json(base_split_path)
        identity_split = identity_by_split[split]
        base_row_spec = base_split["row_keys_file"]
        identity_row_spec = identity_split["row_keys_file"]
        base_row_path = base_root / str(base_row_spec["path"])
        identity_row_path = identity_root / str(identity_row_spec["path"])
        if (
            base_row_spec["size_bytes"] != identity_row_spec["size_bytes"]
            or base_row_spec["sha256"] != identity_row_spec["sha256"]
            or base_row_path.read_bytes() != identity_row_path.read_bytes()
        ):
            raise common.PublicProjectionContractError(
                f"Base/identity row keys differ for {split}"
            )
        records.append(
            {
                "split": split,
                "row_count": 189000,
                "row_keys_size_bytes": int(base_row_spec["size_bytes"]),
                "row_keys_sha256": str(base_row_spec["sha256"]),
                "identity_row_key_stream_sha256": identity_split[
                    "row_key_stream_sha256"
                ],
            }
        )
    return records


def freeze_combined_manifest(
    policy: Mapping[str, Any], publication_root: Path
) -> dict[str, Any]:
    base_root = publication_root / policy["formal_outputs"]["base_subdirectory"]
    identity_root = publication_root / policy["formal_outputs"][
        "identity_subdirectory"
    ]
    root_manifest_path = publication_root / "public_projection_manifest.json"
    if root_manifest_path.exists():
        raise common.PublicProjectionContractError("Combined projection manifest exists")
    base_manifest = base_projection.validate_base_output(policy, base_root)
    identity_manifest = identity_projection.validate_identity_output(
        policy, identity_root
    )
    row_keys = cross_bind_row_keys(
        base_root, base_manifest, identity_root, identity_manifest
    )
    manifest = {
        "step": "step28_v13_v1_13_v9_4_1_public_projection_v1",
        "status": "FROZEN_LABEL_FREE_FOUR_SPLIT_PUBLIC_PROJECTION_TRAINING_INPUT_READY",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "base_manifest_file": file_record(
            base_root / "base_projection_manifest.json", publication_root
        ),
        "base_manifest_canonical_self_hash": base_manifest["canonical_self_hash"],
        "identity_manifest_file": file_record(
            identity_root / "identity_projection_manifest.json", publication_root
        ),
        "identity_manifest_canonical_self_hash": identity_manifest[
            "canonical_self_hash"
        ],
        "split_row_key_bindings": row_keys,
        "total_world_count": 2000,
        "total_pair_count": 756000,
        "cpu_stage_retained": False,
        "opaque_transfer_retained": False,
        "gpu_return_retained": False,
        "temporary_chunks_or_embeddings_retained": False,
        "supervision_or_audit_truth_read": False,
        "model_parameters_updated": False,
        "threshold_selected": False,
        "public_projection_complete": True,
        "training_authorized": False,
    }
    manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
    render_json(root_manifest_path, manifest)
    return validate_publication(policy, publication_root)


def validate_publication(
    policy: Mapping[str, Any], publication_root: Path
) -> dict[str, Any]:
    manifest_path = publication_root / "public_projection_manifest.json"
    manifest = common.load_json(manifest_path)
    _verify_self_hash(manifest, label="combined public projection manifest")
    if (
        manifest.get("status")
        != "FROZEN_LABEL_FREE_FOUR_SPLIT_PUBLIC_PROJECTION_TRAINING_INPUT_READY"
        or manifest.get("policy_canonical_self_hash") != policy["canonical_self_hash"]
        or manifest.get("total_world_count") != 2000
        or manifest.get("total_pair_count") != 756000
        or manifest.get("cpu_stage_retained") is not False
        or manifest.get("opaque_transfer_retained") is not False
        or manifest.get("gpu_return_retained") is not False
        or manifest.get("temporary_chunks_or_embeddings_retained") is not False
        or manifest.get("supervision_or_audit_truth_read") is not False
        or manifest.get("model_parameters_updated") is not False
        or manifest.get("threshold_selected") is not False
        or manifest.get("public_projection_complete") is not True
        or manifest.get("training_authorized") is not False
    ):
        raise common.PublicProjectionContractError("Combined projection lineage drift")
    base_root = publication_root / policy["formal_outputs"]["base_subdirectory"]
    identity_root = publication_root / policy["formal_outputs"][
        "identity_subdirectory"
    ]
    base_path = base_root / "base_projection_manifest.json"
    identity_path = identity_root / "identity_projection_manifest.json"
    base_spec = manifest["base_manifest_file"]
    identity_spec = manifest["identity_manifest_file"]
    if (
        base_spec != file_record(base_path, publication_root)
        or identity_spec != file_record(identity_path, publication_root)
    ):
        raise common.PublicProjectionContractError("Combined child-manifest pin drift")
    base_manifest = base_projection.validate_base_output(policy, base_root)
    identity_manifest = identity_projection.validate_identity_output(
        policy, identity_root
    )
    if (
        manifest.get("base_manifest_canonical_self_hash")
        != base_manifest["canonical_self_hash"]
        or manifest.get("identity_manifest_canonical_self_hash")
        != identity_manifest["canonical_self_hash"]
        or manifest.get("split_row_key_bindings")
        != cross_bind_row_keys(
            base_root, base_manifest, identity_root, identity_manifest
        )
    ):
        raise common.PublicProjectionContractError("Combined row/child binding drift")
    actual_top_level = sorted(path.name for path in publication_root.iterdir())
    if actual_top_level != sorted(
        [
            policy["formal_outputs"]["base_subdirectory"],
            policy["formal_outputs"]["identity_subdirectory"],
            "public_projection_manifest.json",
        ]
    ):
        raise common.PublicProjectionContractError("Publication top-level universe drift")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-contract",))
    parser.parse_args()
    policy = common.load_policy()
    print(
        json.dumps(
            {
                "status": "PASSED_PUBLIC_PROJECTION_PROTOCOL_NO_FORMAL_EXECUTION",
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
