#!/usr/bin/env python3
"""Build the strictly label/raw-source-free Step7-v3.1 GPU payload manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step7_v3_1_source_data as source
import step7_v3_1_common as common
import step7_v3_1_encode_chunked_models as encoder


SYNC_SCRIPT = Path(__file__).resolve()
GPU_CODE_PATHS = [
    "schema/step7_v3_1_source_data_policy.json",
    "scripts/step3_build_seller_profiles.py",
    "scripts/step7_v3_1_source_data.py",
    "scripts/step7_v3_1_common.py",
    "scripts/step7_v3_1_build_sync_manifest.py",
    "scripts/step7_v3_1_materialize_gpu_workspace.py",
    "scripts/step7_v3_1_encode_chunked_models.py",
    "scripts/run_step7_v3_1_full_text_chunked_linux_20260722.sh",
]


def file_record(path_value: str) -> dict:
    path = common.resolve(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Step7-v3.1 GPU sync input is missing: {path}")
    return {
        "path": common.relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def gpu_payload_paths(policy: dict, policy_path: Path) -> list[str]:
    return [
        common.relative(policy_path),
        *GPU_CODE_PATHS,
        policy["outputs"]["pair_manifest"],
        policy["outputs"]["field_corpus"],
        policy["outputs"]["preparation_manifest"],
    ]


def build_payload(policy: dict, policy_path: Path) -> dict:
    source_manifest, field_rows = encoder.verify_source_preparation(policy)
    files = [file_record(path) for path in gpu_payload_paths(policy, policy_path)]
    paths = [record["path"] for record in files]
    if len(paths) != len(set(paths)):
        raise ValueError("Step7-v3.1 GPU sync payload contains duplicate paths")
    source_policy = common.source_policy(policy)
    forbidden_paths = sorted(
        {spec["path"] for spec in source_policy["inputs"].values()}
        | {
            source_policy["outputs"]["train_labels"],
            source_policy["outputs"]["valid_labels"],
            source_policy["outputs"]["development_labels_manifest"],
            source_policy["outputs"]["train_feature_reference"],
            source_policy["outputs"]["safe_pair_features"],
        }
    )
    overlap = sorted(set(paths) & set(forbidden_paths))
    if overlap:
        raise ValueError(f"Step7-v3.1 GPU payload includes forbidden input: {overlap[0]}")
    model_directories = {}
    for model_key, cfg in policy["embedding_models"].items():
        source.validate_sentence_transformer_layout(model_key, cfg)
        model_directories[model_key] = {
            "path": cfg["local_path"],
            **source.validate_model_content_pin(model_key, cfg),
        }
    expected_outputs = encoder.expected_payload_paths(policy) + [
        policy["outputs"]["gpu_output_manifest"]
    ]
    return {
        "step": "step7_v3_1_label_free_gpu_sync",
        "version": policy["version"],
        "generator_script_path": common.relative(SYNC_SCRIPT),
        "generator_script_sha256": common.sha256_file(SYNC_SCRIPT),
        "policy_sha256": common.sha256_file(policy_path),
        "policy_contract_sha256": common.canonical_hash(policy),
        "source_preparation_manifest_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["preparation_manifest"])
        ),
        "field_corpus_sha256": common.sha256_file(
            common.resolve(policy["outputs"]["field_corpus"])
        ),
        "field_seller_count": len(field_rows),
        "source_preparation_reads_labels": source_manifest[
            "feature_generation_uses_review_label_values"
        ],
        "file_count": len(files),
        "total_file_bytes": sum(record["size_bytes"] for record in files),
        "files": files,
        "model_directories": model_directories,
        "label_files_included": False,
        "raw_source_files_included": False,
        "forbidden_workspace_paths": forbidden_paths,
        "gpu_workspace_requires_forbidden_paths_absent": True,
        "expected_gpu_outputs_to_sync_back": expected_outputs,
        "formal_command": "bash scripts/run_step7_v3_1_full_text_chunked_linux_20260722.sh",
        "post_gpu_windows_command": "python scripts/step7_v3_1_select_source_model.py --stage select",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    policy_path = common.resolve(args.policy)
    policy = common.load_json(policy_path)
    common.validate_policy(policy)
    payload = build_payload(policy, policy_path)
    output_path = common.resolve(policy["outputs"]["gpu_sync_manifest"])
    if args.validate_only:
        if not output_path.is_file() or common.load_json(output_path) != payload:
            raise ValueError("Step7-v3.1 GPU sync manifest is missing or stale")
        print(json.dumps({"status": "pass", "manifest_current": True}, indent=2))
        return
    common.write_json_atomic(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
