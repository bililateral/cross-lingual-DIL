#!/usr/bin/env python3
"""Stage and collect a physically label/raw-source isolated Step 7-v4 GPU run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path, PurePosixPath

import step7_v4_build_sync_manifest as sync_builder
import step7_v4_common as common


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
        raise ValueError(f"Unsafe Step7-v4 workspace path: {path_value}")
    candidate = resolved_root.joinpath(*relative.parts)
    try:
        candidate.resolve().relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"Step7-v4 workspace path escapes through a symlink: {path_value}"
        ) from error
    return candidate


def verify_record(root: Path, record: dict, role: str) -> Path:
    path = safe_relative_path(root, record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Step7-v4 {role} is missing: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"Step7-v4 {role} size drift: {record['path']}")
    if common.sha256_file(path) != record["sha256"]:
        raise ValueError(f"Step7-v4 {role} hash drift: {record['path']}")
    return path


def included_model_files(source: Path) -> list[Path]:
    files = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError(f"Step7-v4 model staging refuses symlinks: {path}")
        if ".cache" in path.parts or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(source).as_posix())


def hardlink_model_tree(source: Path, destination: Path) -> int:
    if not source.is_dir() or destination.exists():
        raise ValueError(f"Step7-v4 model staging path is invalid: {source} -> {destination}")
    destination.mkdir(parents=True)
    files = included_model_files(source)
    if not files:
        raise ValueError(f"Step7-v4 model payload is empty: {source}")
    for source_path in files:
        target = destination / source_path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_path, target)
        except OSError as error:
            raise OSError(
                "Step7-v4 model hardlink failed; stage on the same filesystem"
            ) from error
    return len(files)


def expected_workspace_files(workspace: Path, policy: dict, manifest: dict) -> set[Path]:
    expected = {
        safe_relative_path(workspace, record["path"]).resolve()
        for record in manifest["files"]
    }
    expected.add(
        safe_relative_path(workspace, policy["outputs"]["gpu_sync_manifest"]).resolve()
    )
    for model_key, record in manifest["model_directories"].items():
        model_root = safe_relative_path(workspace, record["path"])
        fingerprint = common.model_content_fingerprint(model_root)
        if {
            key: fingerprint[key]
            for key in ("file_count", "total_size_bytes", "content_sha256")
        } != {
            key: record[key]
            for key in ("file_count", "total_size_bytes", "content_sha256")
        }:
            raise ValueError(f"Step7-v4 staged model fingerprint drift: {model_key}")
        expected.update(
            (model_root / item["path"]).resolve() for item in fingerprint["files"]
        )
    return expected


def verify_workspace_is_isolated(
    workspace: Path, policy: dict, manifest: dict, *, allow_outputs: bool
) -> None:
    present_forbidden = [
        value
        for value in manifest["forbidden_workspace_paths"]
        if safe_relative_path(workspace, value).exists()
    ]
    if present_forbidden:
        raise ValueError(
            "Step7-v4 workspace contains a forbidden label/raw-source path: "
            + present_forbidden[0]
        )
    expected = expected_workspace_files(workspace, policy, manifest)
    if allow_outputs:
        for path_value in manifest["expected_gpu_outputs_to_sync_back"]:
            path = safe_relative_path(workspace, path_value)
            if path.is_file():
                expected.add(path.resolve())
    observed = {
        path.resolve()
        for path in workspace.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    unexpected = sorted(str(path) for path in observed - expected)
    missing_inputs = sorted(str(path) for path in expected - observed)
    if unexpected:
        raise ValueError(f"Step7-v4 isolated workspace contains an undeclared file: {unexpected[0]}")
    if missing_inputs:
        raise FileNotFoundError(f"Step7-v4 isolated workspace is incomplete: {missing_inputs[0]}")


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
        raise ValueError("Step7-v4 isolated workspace must be outside the repository")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("Step7-v4 isolated workspace must start empty")

    for record in manifest["files"]:
        source = verify_record(source_root, record, "source payload")
        target = safe_relative_path(destination, record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        verify_record(destination, record, "staged payload")

    sync_relative = policy["outputs"]["gpu_sync_manifest"]
    source_sync = safe_relative_path(source_root, sync_relative)
    target_sync = safe_relative_path(destination, sync_relative)
    if common.load_json(source_sync) != manifest:
        raise ValueError("Step7-v4 source sync manifest drifted during staging")
    target_sync.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_sync, target_sync)

    model_file_count = 0
    for model_key, record in manifest["model_directories"].items():
        cfg = policy["embedding_models"][model_key]
        source_model = safe_relative_path(source_root, record["path"])
        observed = common.validate_model_payload(model_key, cfg)
        if {
            key: observed[key]
            for key in ("file_count", "total_size_bytes", "content_sha256")
        } != {
            key: record[key]
            for key in ("file_count", "total_size_bytes", "content_sha256")
        }:
            raise ValueError(f"Step7-v4 source model/sync drift: {model_key}")
        model_file_count += hardlink_model_tree(
            source_model, safe_relative_path(destination, record["path"])
        )
    verify_workspace_is_isolated(destination, policy, manifest, allow_outputs=False)
    return {
        "status": "pass",
        "operation": "materialize_label_and_raw_source_isolated_gpu_workspace",
        "workspace": str(destination),
        "copied_payload_file_count": len(manifest["files"]),
        "hardlinked_model_file_count": model_file_count,
        "forbidden_paths_present": False,
    }


def collect_outputs(
    source_root: Path, workspace: Path, policy: dict, manifest: dict
) -> dict:
    source_root = source_root.resolve()
    workspace = workspace.resolve()
    if source_root == workspace:
        raise ValueError("Step7-v4 output collection requires a separate workspace")
    verify_workspace_is_isolated(workspace, policy, manifest, allow_outputs=True)
    for record in manifest["files"]:
        verify_record(workspace, record, "staged input replay")

    bundle_relative = policy["outputs"]["gpu_output_manifest"]
    bundle_path = safe_relative_path(workspace, bundle_relative)
    bundle = common.load_json(bundle_path)
    common.verify_canonical_self_hash(
        bundle, "bundle_content_sha256", "GPU output bundle"
    )
    if (
        bundle.get("policy_sha256") != common.sha256_file(common.DEFAULT_POLICY)
        or bundle.get("generator_script_sha256")
        != common.sha256_file(
            common.resolve(policy["implementation"]["encoder"]["path"])
        )
        or bundle.get("step") != "step7_v4_label_free_gpu_output_bundle"
        or bundle.get("version") != policy["version"]
        or bundle.get("label_or_raw_source_files_present_in_gpu_workspace") is not False
    ):
        raise ValueError("Step7-v4 GPU output bundle role/isolation drift")
    expected = set(manifest["expected_gpu_outputs_to_sync_back"])
    payload_expected = expected - {bundle_relative}
    records = bundle.get("files", [])
    if (
        {record.get("path") for record in records} != payload_expected
        or len(records) != len(payload_expected)
        or int(bundle.get("file_count", -1)) != len(records)
        or int(bundle.get("total_file_bytes", -1))
        != sum(int(record["size_bytes"]) for record in records)
    ):
        raise ValueError("Step7-v4 GPU output file universe is incomplete")
    staged_sync = safe_relative_path(workspace, policy["outputs"]["gpu_sync_manifest"])
    if bundle.get("gpu_sync_manifest_sha256") != common.sha256_file(staged_sync):
        raise ValueError("Step7-v4 GPU output does not replay its sync contract")

    verified = []
    for record in records:
        source = verify_record(workspace, record, "GPU output")
        target = safe_relative_path(source_root, record["path"])
        if target.exists() and (
            not target.is_file()
            or target.stat().st_size != source.stat().st_size
            or common.sha256_file(target) != record["sha256"]
        ):
            raise FileExistsError(f"Step7-v4 refuses to overwrite a different GPU output: {target}")
        verified.append((record, source, target))
    target_bundle = safe_relative_path(source_root, bundle_relative)
    if target_bundle.exists() and target_bundle.read_bytes() != bundle_path.read_bytes():
        raise FileExistsError("Step7-v4 refuses to overwrite a different GPU output manifest")
    for _record, source, target in verified:
        common.write_bytes_immutable(target, source.read_bytes())
    common.write_bytes_immutable(target_bundle, bundle_path.read_bytes())
    return {
        "status": "pass",
        "operation": "collect_verified_compact_gpu_outputs",
        "copied_output_file_count": len(verified) + 1,
        "embedding_matrix_files_copied": 0,
    }


def load_current_contract() -> tuple[dict, dict]:
    policy = common.load_policy()
    path = common.resolve(policy["outputs"]["gpu_sync_manifest"])
    if not path.is_file():
        raise FileNotFoundError("Build and transfer the Step7-v4 GPU sync manifest first")
    manifest = common.load_json(path)
    expected = sync_builder.build_payload(policy, common.DEFAULT_POLICY)
    if manifest != expected:
        raise ValueError("Step7-v4 GPU sync manifest is stale; rebuild before staging")
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
