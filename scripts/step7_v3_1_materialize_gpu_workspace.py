#!/usr/bin/env python3
"""Stage and collect a physically label-free Step7-v3.1 GPU workspace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path, PurePosixPath

import step7_v3_1_build_sync_manifest as sync_builder
import step7_v3_1_common as common


def safe_relative_path(root: Path, path_value: str) -> Path:
    resolved_root = root.resolve()
    normalized = str(path_value).replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or ".." in relative.parts
        or any(":" in part for part in relative.parts)
    ):
        raise ValueError(f"Unsafe Step7-v3.1 workspace path: {path_value}")
    candidate = resolved_root.joinpath(*relative.parts)
    try:
        candidate.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Step7-v3.1 workspace path escapes through a symlink: {path_value}"
        ) from exc
    return candidate


def verify_record(root: Path, record: dict, role: str) -> Path:
    path = safe_relative_path(root, record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Step7-v3.1 {role} is missing: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"Step7-v3.1 {role} size drift: {record['path']}")
    if common.sha256_file(path) != record["sha256"]:
        raise ValueError(f"Step7-v3.1 {role} hash drift: {record['path']}")
    return path


def verify_workspace_is_isolated(workspace: Path, manifest: dict) -> None:
    present = [
        value
        for value in manifest.get("forbidden_workspace_paths", [])
        if safe_relative_path(workspace, value).exists()
    ]
    if present:
        raise ValueError(
            "Step7-v3.1 staged workspace contains a forbidden label/raw-source path: "
            f"{present[0]}"
        )


def hardlink_model_tree(source: Path, destination: Path) -> int:
    if not source.is_dir() or destination.exists():
        raise ValueError(
            f"Step7-v3.1 model staging path is invalid: {source} -> {destination}"
        )
    destination.mkdir(parents=True)
    count = 0
    for source_path in sorted(source.rglob("*")):
        if source_path.is_symlink():
            raise ValueError(
                f"Step7-v3.1 model staging refuses symlinks: {source_path}"
            )
        target = destination / source_path.relative_to(source)
        if source_path.is_dir():
            target.mkdir(exist_ok=True)
        elif source_path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_path, target)
            except OSError as exc:
                raise OSError(
                    "Step7-v3.1 model hardlink failed; keep the workspace on the same filesystem"
                ) from exc
            count += 1
    return count


def materialize_workspace(
    source_root: Path, destination: Path, policy: dict, manifest: dict
) -> dict:
    source_root = source_root.resolve()
    destination = destination.resolve()
    try:
        destination.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("Step7-v3.1 isolated workspace must be outside the source repository")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("Step7-v3.1 isolated workspace must start empty")
    copied = []
    for record in manifest["files"]:
        source = verify_record(source_root, record, "source payload")
        target = safe_relative_path(destination, record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        verify_record(destination, record, "staged payload")
        copied.append(record["path"])

    sync_relative = policy["outputs"]["gpu_sync_manifest"]
    if sync_relative in copied:
        raise ValueError("Step7-v3.1 sync manifest cannot recursively list itself")
    source_sync = safe_relative_path(source_root, sync_relative)
    target_sync = safe_relative_path(destination, sync_relative)
    if not source_sync.is_file() or common.load_json(source_sync) != manifest:
        raise ValueError("Step7-v3.1 source sync manifest drifted during staging")
    target_sync.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_sync, target_sync)
    if target_sync.read_bytes() != source_sync.read_bytes():
        raise ValueError("Step7-v3.1 staged sync manifest byte drift")

    model_file_count = 0
    for model_key, record in manifest["model_directories"].items():
        count = hardlink_model_tree(
            safe_relative_path(source_root, record["path"]),
            safe_relative_path(destination, record["path"]),
        )
        if count <= 0:
            raise ValueError(f"Step7-v3.1 staged model is empty: {model_key}")
        model_file_count += count
    verify_workspace_is_isolated(destination, manifest)
    return {
        "status": "pass",
        "operation": "materialize_label_free_gpu_workspace",
        "source_root": str(source_root),
        "workspace": str(destination),
        "copied_payload_file_count": len(copied),
        "staged_sync_manifest": sync_relative,
        "staged_sync_manifest_sha256": common.sha256_file(target_sync),
        "hardlinked_model_file_count": model_file_count,
        "forbidden_paths_present": False,
    }


def collect_outputs(
    source_root: Path, workspace: Path, policy: dict, manifest: dict
) -> dict:
    source_root = source_root.resolve()
    workspace = workspace.resolve()
    if source_root == workspace:
        raise ValueError("Step7-v3.1 output collection requires a separate workspace")
    verify_workspace_is_isolated(workspace, manifest)
    for record in manifest["files"]:
        verify_record(workspace, record, "staged input replay")
    output_manifest_relative = policy["outputs"]["gpu_output_manifest"]
    output_manifest_path = safe_relative_path(workspace, output_manifest_relative)
    if not output_manifest_path.is_file():
        raise FileNotFoundError("Step7-v3.1 GPU output manifest is missing")
    bundle = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    if (
        bundle.get("step") != "step7_v3_1_label_free_gpu_output_bundle"
        or bundle.get("version") != policy["version"]
        or bundle.get("label_or_raw_source_files_present_in_gpu_workspace") is not False
    ):
        raise ValueError("Step7-v3.1 GPU output bundle role/isolation drift")
    expected = set(manifest["expected_gpu_outputs_to_sync_back"])
    if output_manifest_relative not in expected:
        raise ValueError("Step7-v3.1 sync contract omits the GPU output manifest")
    expected_payload = expected - {output_manifest_relative}
    records = bundle.get("files", [])
    observed = {record.get("path") for record in records}
    if (
        observed != expected_payload
        or len(records) != len(expected_payload)
        or int(bundle.get("file_count", -1)) != len(records)
        or int(bundle.get("total_file_bytes", -1))
        != sum(int(record["size_bytes"]) for record in records)
    ):
        raise ValueError("Step7-v3.1 GPU output file universe is incomplete")
    staged_sync = safe_relative_path(workspace, policy["outputs"]["gpu_sync_manifest"])
    if bundle.get("gpu_sync_manifest_sha256") != common.sha256_file(staged_sync):
        raise ValueError("Step7-v3.1 GPU output does not replay the staged sync manifest")

    verified = []
    for record in records:
        source = verify_record(workspace, record, "GPU output")
        target = safe_relative_path(source_root, record["path"])
        if target.exists() and (
            not target.is_file()
            or target.stat().st_size != source.stat().st_size
            or common.sha256_file(target) != record["sha256"]
        ):
            raise FileExistsError(
                f"Step7-v3.1 refuses to overwrite a different GPU output: {target}"
            )
        verified.append((record, source, target))
    manifest_target = safe_relative_path(source_root, output_manifest_relative)
    if manifest_target.exists() and (
        not manifest_target.is_file()
        or manifest_target.read_bytes() != output_manifest_path.read_bytes()
    ):
        raise FileExistsError("Step7-v3.1 refuses to overwrite a different output manifest")
    copied = []
    for record, source, target in verified:
        common.write_bytes_immutable(target, source.read_bytes())
        copied.append(record["path"])
    common.write_bytes_immutable(manifest_target, output_manifest_path.read_bytes())
    copied.append(output_manifest_relative)
    return {
        "status": "pass",
        "operation": "collect_verified_gpu_outputs",
        "workspace": str(workspace),
        "destination_root": str(source_root),
        "copied_output_file_count": len(copied),
        "copied_outputs": copied,
    }


def load_current_contract() -> tuple[dict, dict]:
    policy = common.load_json(common.DEFAULT_POLICY)
    common.validate_policy(policy)
    manifest_path = common.resolve(policy["outputs"]["gpu_sync_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("Build the Step7-v3.1 GPU sync manifest before staging")
    manifest = common.load_json(manifest_path)
    expected = sync_builder.build_payload(policy, common.DEFAULT_POLICY)
    if manifest != expected:
        raise ValueError("Step7-v3.1 GPU sync manifest is stale; rebuild before staging")
    return policy, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--destination", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--workspace", required=True)
    args = parser.parse_args()
    policy, manifest = load_current_contract()
    if args.operation == "stage":
        result = materialize_workspace(common.ROOT, Path(args.destination), policy, manifest)
    else:
        result = collect_outputs(common.ROOT, Path(args.workspace), policy, manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
