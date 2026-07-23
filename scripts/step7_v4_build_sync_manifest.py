#!/usr/bin/env python3
"""Build the minimal label-free Step 7-v4 Linux GPU transfer contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step7_v4_common as common


GPU_IMPLEMENTATION_ROLES = [
    "common",
    "sync_builder",
    "encoder",
]
GPU_PUBLIC_OUTPUT_ROLES = [
    "gpu_pair_manifest",
    "unique_text_corpus",
    "gpu_seller_text_index",
]
FORBIDDEN_WORKSPACE_PATHS = [
    "market_item.xlsx",
    "2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx",
    "reports/step2_content_item_manifest.csv",
    "reports/step7_v4_raw_item_authorship_selection/v2_20260723/pair_manifest.no_labels.csv",
    "reports/step7_v4_raw_item_authorship_selection/v2_20260723/raw_item_lineage.no_labels.csv",
    "reports/step7_v4_raw_item_authorship_selection/v2_20260723/seller_unique_text_index.no_labels.jsonl",
    "reports/step7_v4_raw_item_authorship_selection/v2_20260723/source_preparation_manifest.json",
    "reports/step7_v4_raw_item_authorship_selection/v2_20260723/private_labels.train.csv",
    "reports/step7_v4_raw_item_authorship_selection/v2_20260723/private_labels.valid.csv",
    "reports/step7_v4_raw_item_authorship_selection/v2_20260723/private_evidence.train.csv",
    "reports/step7_v4_raw_item_authorship_selection/v2_20260723/private_evidence.valid.csv",
    "reports/step7_v4_raw_item_authorship_selection/v2_20260723/development_labels_manifest.json",
    "reports/step7_v3_1_full_text_chunked_selection/v1_20260722/private_labels.train.csv",
    "reports/step7_v3_1_full_text_chunked_selection/v1_20260722/private_labels.valid.csv",
]


def preparation_manifest_hash_contract(
    preparation_manifest: dict, preparation_manifest_path: Path
) -> dict[str, str]:
    common.verify_canonical_self_hash(
        preparation_manifest,
        "manifest_content_sha256",
        "public preparation manifest",
    )
    return {
        "source_preparation_manifest_file_sha256": common.sha256_file(
            preparation_manifest_path
        ),
        "source_preparation_manifest_content_sha256": preparation_manifest[
            "manifest_content_sha256"
        ],
    }


def payload_paths(policy: dict, policy_path: Path) -> list[Path]:
    paths = [policy_path.resolve()]
    paths.extend(
        common.resolve(policy["implementation"][role]["path"])
        for role in GPU_IMPLEMENTATION_ROLES
    )
    paths.extend(
        common.resolve(policy["outputs"][role])
        for role in GPU_PUBLIC_OUTPUT_ROLES
    )
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Step7-v4 GPU payload contains a duplicate path")
    return paths


def expected_gpu_output_paths(policy: dict) -> list[str]:
    outputs = policy["outputs"]
    paths = [outputs["shared_chunks"], outputs["shared_chunks_manifest"]]
    for model_key in common.MODEL_KEYS:
        paths.append(outputs["model_runtime_manifest_template"].format(model_key=model_key))
        paths.append(outputs["pair_scores_template"].format(model_key=model_key))
    paths.append(outputs["gpu_output_manifest"])
    return paths


def build_payload(policy: dict, policy_path: Path) -> dict:
    common.validate_policy(policy)
    common.verify_implementation_files(policy)
    preparation_manifest = common.load_json(
        common.resolve(policy["outputs"]["preparation_manifest"])
    )
    common.verify_canonical_self_hash(
        preparation_manifest,
        "manifest_content_sha256",
        "public preparation manifest",
    )
    if (
        preparation_manifest.get("step") != "step7_v4_prepare_source_data_public"
        or preparation_manifest.get("labels_read") is not False
        or preparation_manifest.get("evidence_types_read") is not False
        or preparation_manifest.get(
            "pair_label_or_evidence_bearing_input_files_opened"
        )
        is not False
        or preparation_manifest.get("labelled_component_assignment_file_opened")
        is not False
        or preparation_manifest.get(
            "frozen_label_free_parent_pair_projection_used"
        )
        is not True
        or preparation_manifest.get("historical_test_labels_read") is not False
        or preparation_manifest.get("policy_sha256")
        != common.sha256_file(common.DEFAULT_POLICY)
    ):
        raise ValueError("Step7-v4 GPU sync refuses an unverified public preparation")
    for role in ("gpu_pair_manifest", "unique_text_corpus", "gpu_seller_text_index"):
        record = preparation_manifest.get("outputs", {}).get(role)
        if record != common.file_record(common.resolve(policy["outputs"][role])):
            raise ValueError(f"Step7-v4 preparation output drift before sync: {role}")

    parent_label_paths = {
        policy["inputs"][role]["path"]
        for role in ("parent_train_labels", "parent_valid_labels")
    }
    if not parent_label_paths.issubset(FORBIDDEN_WORKSPACE_PATHS):
        raise ValueError(
            "Step7-v4 parent label inputs are not forbidden from the GPU workspace"
        )

    records = [common.file_record(path) for path in payload_paths(policy, policy_path)]
    forbidden = {str(value).replace("\\", "/") for value in FORBIDDEN_WORKSPACE_PATHS}
    observed_paths = {record["path"] for record in records}
    if observed_paths & forbidden:
        raise ValueError("Step7-v4 GPU payload includes a forbidden path")
    if policy["outputs"]["gpu_sync_manifest"] in observed_paths:
        raise ValueError("Step7-v4 GPU sync manifest cannot list itself")

    model_directories = {
        model_key: {
            "path": cfg["local_path"],
            "file_count": int(cfg["expected_file_count"]),
            "total_size_bytes": int(cfg["expected_total_size_bytes"]),
            "content_sha256": cfg["expected_content_sha256"],
        }
        for model_key, cfg in policy["embedding_models"].items()
    }
    preparation_hashes = preparation_manifest_hash_contract(
        preparation_manifest,
        common.resolve(policy["outputs"]["preparation_manifest"]),
    )
    result = {
        "step": "step7_v4_label_free_gpu_sync",
        "version": policy["version"],
        "policy_sha256": common.sha256_file(policy_path),
        "policy_contract_sha256": common.canonical_hash(policy),
        **preparation_hashes,
        "pair_level_label_or_evidence_value_files_in_payload": False,
        "raw_source_workbook_or_item_manifest_file_bytes_in_payload": False,
        "aggregate_supervision_counts_and_evidence_vocabulary_in_policy_only": True,
        "files": records,
        "file_count": len(records),
        "total_file_bytes": sum(record["size_bytes"] for record in records),
        "model_directories": model_directories,
        "forbidden_workspace_paths": FORBIDDEN_WORKSPACE_PATHS,
        "expected_gpu_outputs_to_sync_back": expected_gpu_output_paths(policy),
    }
    result["manifest_content_sha256"] = common.canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the manifest without writing it.",
    )
    args = parser.parse_args()
    policy = common.load_policy()
    payload = build_payload(policy, common.DEFAULT_POLICY)
    path = common.resolve(policy["outputs"]["gpu_sync_manifest"])
    if args.validate_only:
        if path.is_file() and common.load_json(path) != payload:
            raise ValueError("Step7-v4 existing GPU sync manifest is stale")
    else:
        common.write_json_immutable(path, payload)
    print(
        json.dumps(
            {
                "status": "pass",
                "written": not args.validate_only,
                "file_count": payload["file_count"],
                "expected_gpu_output_count": len(
                    payload["expected_gpu_outputs_to_sync_back"]
                ),
                "manifest": common.relative(path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
