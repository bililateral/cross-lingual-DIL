#!/usr/bin/env python3
"""Materialize and collect a physically label-free Step7-v3 GPU workspace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path, PurePosixPath

import step7_v3_build_sync_manifest as sync_builder
import step7_v3_common as common


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
        raise ValueError(f"Unsafe Step7-v3 workspace path: {path_value}")
    candidate = resolved_root.joinpath(*relative.parts)
    try:
        candidate.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Step7-v3 workspace path escapes through a symlink: {path_value}"
        ) from exc
    return candidate


def verify_record(root: Path, record: dict, role: str) -> Path:
    path = safe_relative_path(root, record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Step7-v3 {role} is missing: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"Step7-v3 {role} size drift: {record['path']}")
    if common.sha256_file(path) != record["sha256"]:
        raise ValueError(f"Step7-v3 {role} hash drift: {record['path']}")
    return path


def verify_workspace_is_label_free(workspace: Path, manifest: dict) -> None:
    present = [
        path_value
        for path_value in manifest.get("forbidden_workspace_paths", [])
        if safe_relative_path(workspace, path_value).exists()
    ]
    if present:
        raise ValueError(
            "Step7-v3 staged GPU workspace contains a forbidden label/raw-source "
            f"path: {present[0]}"
        )


def hardlink_model_tree(source: Path, destination: Path) -> int:
    if not source.is_dir() or destination.exists():
        raise ValueError(
            f"Step7-v3 model staging path is invalid: {source} -> {destination}"
        )
    destination.mkdir(parents=True)
    file_count = 0
    for source_path in sorted(source.rglob("*")):
        if source_path.is_symlink():
            raise ValueError(
                f"Step7-v3 model staging refuses source symlinks: {source_path}"
            )
        target_path = destination / source_path.relative_to(source)
        if source_path.is_dir():
            target_path.mkdir(exist_ok=True)
        elif source_path.is_file():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_path, target_path)
            except OSError as exc:
                raise OSError(
                    "Step7-v3 model hardlink staging failed. Put the isolated "
                    "workspace on the same filesystem as the source repository. "
                    f"source={source_path} destination={target_path}"
                ) from exc
            file_count += 1
        else:
            raise ValueError(f"Unsupported Step7-v3 model payload entry: {source_path}")
    return file_count


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
        raise ValueError("Step7-v3 isolated workspace must be outside the source repository")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(
            f"Step7-v3 isolated workspace must start empty: {destination}"
        )

    copied_files = []
    for record in manifest.get("files", []):
        source_path = verify_record(source_root, record, "source payload file")
        target_path = safe_relative_path(destination, record["path"])
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        verify_record(destination, record, "staged payload file")
        copied_files.append(record["path"])

    # The sync manifest cannot list itself in ``manifest["files"]`` because
    # doing so would make its hash recursively self-referential.  It is still
    # a required runtime input: the isolated encoder replays this exact
    # contract before touching any model.  Stage it explicitly and verify
    # byte-for-byte identity with the already validated source contract.
    sync_manifest_relative = policy["outputs"]["gpu_sync_manifest"]
    if sync_manifest_relative in copied_files:
        raise ValueError(
            "Step7-v3 GPU sync manifest must be staged explicitly, not listed "
            "inside its own payload"
        )
    source_sync_manifest = safe_relative_path(source_root, sync_manifest_relative)
    if not source_sync_manifest.is_file():
        raise FileNotFoundError(
            f"Step7-v3 source GPU sync manifest is missing: {source_sync_manifest}"
        )
    if common.load_json(source_sync_manifest) != manifest:
        raise ValueError("Step7-v3 source GPU sync manifest changed during staging")
    staged_sync_manifest = safe_relative_path(destination, sync_manifest_relative)
    staged_sync_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_sync_manifest, staged_sync_manifest)
    if staged_sync_manifest.read_bytes() != source_sync_manifest.read_bytes():
        raise ValueError("Step7-v3 staged GPU sync manifest byte drift")

    model_file_count = 0
    for model_key, record in manifest.get("model_directories", {}).items():
        source_path = safe_relative_path(source_root, record["path"])
        target_path = safe_relative_path(destination, record["path"])
        staged_model_file_count = hardlink_model_tree(source_path, target_path)
        if staged_model_file_count <= 0:
            raise ValueError(f"Step7-v3 staged model is empty: {model_key}")
        model_file_count += staged_model_file_count

    verify_workspace_is_label_free(destination, manifest)
    return {
        "status": "pass",
        "operation": "materialize_label_free_gpu_workspace",
        "source_root": str(source_root),
        "workspace": str(destination),
        "copied_payload_file_count": len(copied_files),
        "staged_sync_manifest": sync_manifest_relative,
        "staged_sync_manifest_sha256": common.sha256_file(staged_sync_manifest),
        "hardlinked_model_file_count": model_file_count,
        "forbidden_paths_present": False,
    }


def collect_outputs(source_root: Path, workspace: Path, policy: dict, manifest: dict) -> dict:
    source_root = source_root.resolve()
    workspace = workspace.resolve()
    if source_root == workspace:
        raise ValueError("Step7-v3 output collection requires a separate workspace")
    verify_workspace_is_label_free(workspace, manifest)
    for record in manifest.get("files", []):
        verify_record(workspace, record, "staged input replay")

    output_manifest_relative = policy["outputs"]["gpu_output_manifest"]
    output_manifest_path = safe_relative_path(workspace, output_manifest_relative)
    if not output_manifest_path.is_file():
        raise FileNotFoundError(
            f"Step7-v3 GPU output manifest is missing: {output_manifest_path}"
        )
    bundle = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    if (
        bundle.get("step") != "step7_v3_label_free_gpu_output_bundle"
        or bundle.get("version") != policy["version"]
        or bundle.get("label_or_raw_source_files_present_in_gpu_workspace") is not False
    ):
        raise ValueError("Step7-v3 GPU output bundle role or isolation claim is invalid")

    expected = set(manifest.get("expected_gpu_outputs_to_sync_back", []))
    if output_manifest_relative not in expected:
        raise ValueError("Step7-v3 sync contract omits the GPU output manifest")
    expected_payload = expected - {output_manifest_relative}
    records = bundle.get("files", [])
    observed_payload = {record.get("path") for record in records}
    if (
        observed_payload != expected_payload
        or len(records) != len(expected_payload)
        or int(bundle.get("file_count", -1)) != len(records)
        or int(bundle.get("total_file_bytes", -1))
        != sum(int(record["size_bytes"]) for record in records)
    ):
        raise ValueError("Step7-v3 GPU output bundle file universe is incomplete")
    sync_manifest_path = safe_relative_path(
        workspace, policy["outputs"]["gpu_sync_manifest"]
    )
    if bundle.get("gpu_sync_manifest_sha256") != common.sha256_file(
        sync_manifest_path
    ):
        raise ValueError("Step7-v3 GPU output bundle does not replay the staged sync manifest")

    verified = []
    for record in records:
        source_path = verify_record(workspace, record, "GPU output")
        target_path = safe_relative_path(source_root, record["path"])
        if target_path.exists() and (
            not target_path.is_file()
            or target_path.stat().st_size != source_path.stat().st_size
            or common.sha256_file(target_path) != record["sha256"]
        ):
            raise FileExistsError(
                f"Step7-v3 refuses to overwrite a different GPU output: {target_path}"
            )
        verified.append((record, source_path, target_path))
    output_manifest_target = safe_relative_path(
        source_root, output_manifest_relative
    )
    if output_manifest_target.exists() and (
        not output_manifest_target.is_file()
        or output_manifest_target.read_bytes() != output_manifest_path.read_bytes()
    ):
        raise FileExistsError(
            "Step7-v3 refuses to overwrite a different GPU output manifest: "
            f"{output_manifest_target}"
        )

    copied = []
    for record, source_path, target_path in verified:
        common.write_bytes_immutable(target_path, source_path.read_bytes())
        copied.append(record["path"])
    common.write_bytes_immutable(
        output_manifest_target,
        output_manifest_path.read_bytes(),
    )
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
    policy_path = common.DEFAULT_POLICY
    policy = common.load_json(policy_path)
    common.validate_policy(policy)
    manifest_path = common.resolve(policy["outputs"]["gpu_sync_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("Build the Step7-v3 GPU sync manifest before staging")
    manifest = common.load_json(manifest_path)
    expected = sync_builder.build_payload(policy, policy_path)
    if manifest != expected:
        raise ValueError("Step7-v3 GPU sync manifest is stale; rebuild before staging")
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
        result = materialize_workspace(
            common.ROOT, Path(args.destination), policy, manifest
        )
    else:
        result = collect_outputs(
            common.ROOT, Path(args.workspace), policy, manifest
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
