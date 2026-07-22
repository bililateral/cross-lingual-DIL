#!/usr/bin/env python3
"""Build a strictly label-free Windows-to-Linux Step7-v3 GPU payload manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step7_v3_common as common


SYNC_SCRIPT = Path(__file__).resolve()


GPU_CODE_PATHS = [
    "scripts/step3_build_seller_profiles.py",
    "scripts/step7_v3_common.py",
    "scripts/step7_v3_build_sync_manifest.py",
    "scripts/step7_v3_encode_clean_models.py",
    "scripts/run_step7_v3_clean_source_linux_20260722.sh",
]


def relative(path) -> str:
    return str(path.relative_to(common.ROOT)).replace("\\", "/")


def file_record(path_value: str) -> dict:
    path = common.resolve(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Step7-v3 GPU sync input is missing: {path}")
    return {
        "path": relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def build_payload(policy: dict, policy_path) -> dict:
    outputs = policy["outputs"]
    public_manifest_path = common.resolve(outputs["preparation_manifest"])
    if not public_manifest_path.is_file():
        raise FileNotFoundError("Run Step7-v3 public preparation before building GPU sync")
    public_manifest = common.load_json(public_manifest_path)
    if public_manifest.get("step") != "step7_v3_prepare_public_label_free_data":
        raise ValueError("Step7-v3 GPU sync refuses a non-public preparation manifest")
    if public_manifest.get("feature_generation_uses_review_label_values") is not False:
        raise ValueError("Step7-v3 public preparation is not label isolated")
    if public_manifest.get("pair_feature_roles") != policy["pair_feature_roles"] or public_manifest.get(
        "shortcut_features_eligible_for_model_training_or_selection"
    ) is not False:
        raise ValueError("Step7-v3 public shortcut feature roles are stale or unsafe")
    required_public_hashes = {
        "generator_script_sha256": common.sha256_file(
            common.resolve("scripts/step7_v3_prepare_clean_data.py")
        ),
        "common_script_sha256": common.sha256_file(
            common.resolve("scripts/step7_v3_common.py")
        ),
        "redaction_dependency_script_sha256": common.sha256_file(
            common.resolve("scripts/step3_build_seller_profiles.py")
        ),
        "policy_sha256": common.sha256_file(policy_path),
    }
    for field, expected in required_public_hashes.items():
        if public_manifest.get(field) != expected:
            raise ValueError(f"Step7-v3 public preparation is stale: {field}")
    residue_scan = public_manifest.get("identity_residue_scan", {})
    if residue_scan.get("status") != "pass" or residue_scan.get(
        "total_residue_count"
    ) != 0:
        raise ValueError("Step7-v3 public identity-residue audit did not pass")
    common.validate_content_fidelity_manifest(policy, public_manifest)
    common.validate_global_identity_audit_manifest(policy, public_manifest)
    for key in ("pair_manifest", "clean_corpus"):
        record = public_manifest.get("output_files", {}).get(key)
        if record is None:
            raise ValueError(f"Step7-v3 public preparation omits {key}")
        path = common.resolve(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or common.sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"Step7-v3 public preparation artifact drift: {key}")

    data_paths = [
        outputs["pair_manifest"],
        outputs["clean_corpus"],
        outputs["preparation_manifest"],
    ]
    policy_relative = relative(policy_path)
    code_paths = [policy_relative, *GPU_CODE_PATHS]
    files = [file_record(path) for path in [*code_paths, *data_paths]]
    file_paths = {record["path"] for record in files}
    forbidden_paths = sorted(
        {
            spec["path"] for spec in policy["inputs"].values()
        }
        | {
            outputs["train_labels"],
            outputs["valid_labels"],
            outputs["historical_test_labels"],
            outputs["development_labels_manifest"],
            outputs["historical_test_labels_manifest"],
        }
    )
    overlap = sorted(file_paths & set(forbidden_paths))
    if overlap:
        raise ValueError(f"Step7-v3 GPU payload includes forbidden source/label file: {overlap[0]}")

    model_directories = {}
    for key, cfg in policy["embedding_models"].items():
        common.validate_sentence_transformer_layout(key, cfg)
        model_directories[key] = {
            "path": cfg["local_path"],
            **common.validate_model_content_pin(key, cfg),
        }
    reranker_cfg = policy["shared_reranker"]
    common.validate_reranker_layout(reranker_cfg["model_key"], reranker_cfg)
    model_directories[reranker_cfg["model_key"]] = {
        "path": reranker_cfg["local_path"],
        **common.validate_model_content_pin(reranker_cfg["model_key"], reranker_cfg),
    }

    expected_gpu_outputs = []
    for model_key in policy["embedding_models"]:
        expected_gpu_outputs.extend(
            [
                outputs["embedding_matrix_template"].format(model_key=model_key),
                outputs["embedding_manifest_template"].format(model_key=model_key),
                outputs["embedding_pair_scores_template"].format(model_key=model_key),
            ]
        )
    expected_gpu_outputs.extend(
        [
            outputs["reranker_pair_scores"],
            outputs["reranker_manifest"],
            outputs["gpu_output_manifest"],
        ]
    )
    return {
        "step": "step7_v3_label_free_windows_to_linux_gpu_sync",
        "version": policy["version"],
        "generator_script_path": relative(SYNC_SCRIPT),
        "generator_script_sha256": common.sha256_file(SYNC_SCRIPT),
        "policy_sha256": common.sha256_file(policy_path),
        "policy_contract_sha256": common.canonical_hash(policy),
        "public_preparation_manifest_sha256": common.sha256_file(public_manifest_path),
        "file_count": len(files),
        "total_file_bytes": sum(record["size_bytes"] for record in files),
        "files": files,
        "model_directories": model_directories,
        "label_files_included": False,
        "raw_source_files_included": False,
        "forbidden_workspace_paths": forbidden_paths,
        "gpu_workspace_requires_forbidden_paths_absent": True,
        "model_payloads_must_already_exist_on_linux_or_be_synced_separately": True,
        "expected_gpu_outputs_to_sync_back": expected_gpu_outputs,
        "formal_command": "bash scripts/run_step7_v3_clean_source_linux_20260722.sh",
        "post_gpu_windows_command": (
            "python scripts/step7_v3_select_source_model.py --stage select"
        ),
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
        if not output_path.is_file():
            raise FileNotFoundError("Step7-v3 GPU sync manifest has not been built")
        if common.load_json(output_path) != payload:
            raise ValueError("Step7-v3 GPU sync manifest is stale; rebuild it before transfer")
        print(
            json.dumps(
                {"status": "pass", "existing_manifest_matches_current_payload": True},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    # This is an operational transfer inventory, not an experimental result.
    # It must safely refresh whenever code changes before the formal GPU run.
    common.write_json_atomic(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
