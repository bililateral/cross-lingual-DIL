#!/usr/bin/env python3
"""Materialize and collect the split-blind Linux GPU projection workspace."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import step28_v13_v1_13_v9_4_1_encode_base_projection_linux_v1 as encoder
import step28_v13_v1_13_v9_4_1_public_projection_gpu_common_v1 as common
import step7_v4_common as step7_common


STATIC_FILES = (
    "schema/step28_v13_v1_13_v9_4_1_public_projection_gpu_policy_v1.json",
    "schema/step7_v4_raw_item_authorship_selection_policy.json",
    "scripts/step28_v13_v1_13_v9_4_1_public_projection_gpu_common_v1.py",
    "scripts/step28_v13_v1_13_v9_4_1_encode_base_projection_linux_v1.py",
    "scripts/step7_v4_common.py",
    "scripts/step7_v4_build_sync_manifest.py",
    "scripts/step7_v4_encode_item_models.py",
)


def safe_path(root: Path, relative_value: str) -> Path:
    normalized = str(relative_value).replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or ".." in relative.parts
        or any(":" in part for part in relative.parts)
    ):
        raise common.GPUProjectionContractError("Unsafe GPU workspace path")
    target = root.resolve().joinpath(*relative.parts)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise common.GPUProjectionContractError("GPU path escapes through a link") from exc
    return target


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def verify_runner_record(source_root: Path, spec: Mapping[str, Any]) -> Path:
    path = safe_path(source_root, str(spec["path"]))
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["size_bytes"])
        or common.sha256_file(path) != spec["sha256"]
    ):
        raise common.GPUProjectionContractError("Authorized GPU runner pin drift")
    return path


def included_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"GPU payload directory is missing: {root}")
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise common.GPUProjectionContractError("GPU staging refuses symlink files")
        if ".cache" in path.parts or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append(path)
    return sorted(files, key=lambda value: value.relative_to(root).as_posix())


def hardlink_tree(source: Path, destination: Path) -> list[Path]:
    if destination.exists():
        raise FileExistsError(f"GPU hardlink destination exists: {destination}")
    files = included_files(source)
    if not files:
        raise common.GPUProjectionContractError("GPU hardlink source is empty")
    linked = []
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError as exc:
            raise OSError(
                "GPU staging hardlinks require the repository and workspace on one filesystem"
            ) from exc
        linked.append(target)
    return linked


def copy_static_file(source_root: Path, workspace: Path, relative: str) -> Path:
    source = safe_path(source_root, relative)
    if not source.is_file():
        raise FileNotFoundError(f"GPU static implementation is missing: {source}")
    target = safe_path(workspace, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if file_record(source, source_root) != file_record(target, workspace):
        raise common.GPUProjectionContractError("GPU static file changed while staging")
    return target


def verify_materialized_workspace(
    policy: Mapping[str, Any],
    transfer_root: Path,
    workspace: Path,
    authorized_runner: Mapping[str, Any],
    *,
    require_gpu_return: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Reopen every declared workspace byte and reject every undeclared byte."""

    source_root = common.ROOT.resolve()
    workspace = workspace.resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"GPU workspace is missing: {workspace}")
    try:
        workspace.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise common.GPUProjectionContractError(
            "GPU workspace must remain outside the repository"
        )
    if any(path.is_symlink() for path in workspace.rglob("*")):
        raise common.GPUProjectionContractError("GPU workspace contains a symlink")

    runner_path = verify_runner_record(source_root, authorized_runner)
    runner_relative = str(authorized_runner["path"])
    if runner_relative in STATIC_FILES:
        raise common.GPUProjectionContractError("Authorized runner duplicates static files")
    expected_paths = set(STATIC_FILES) | {runner_relative}
    for relative in expected_paths:
        source = runner_path if relative == runner_relative else safe_path(
            source_root, relative
        )
        target = safe_path(workspace, relative)
        if (
            not target.is_file()
            or target.stat().st_size != source.stat().st_size
            or common.sha256_file(target) != common.sha256_file(source)
        ):
            raise common.GPUProjectionContractError("GPU static implementation drift")

    source_transfer, _ = encoder.validate_transfer(policy, transfer_root)
    staged_transfer, staged_parts = encoder.validate_transfer(
        policy, workspace / "transfer"
    )
    if staged_transfer != source_transfer:
        raise common.GPUProjectionContractError("Workspace transfer drift")
    expected_paths.update(
        f"transfer/{path.relative_to(workspace / 'transfer').as_posix()}"
        for path in included_files(workspace / "transfer")
    )

    for model_key, pin in policy["model_payloads"].items():
        model_root = safe_path(workspace, str(pin["path"]))
        observed = step7_common.model_content_fingerprint(model_root)
        if {
            key: observed[key]
            for key in ("file_count", "total_size_bytes", "content_sha256")
        } != {
            "file_count": pin["file_count"],
            "total_size_bytes": pin["total_size_bytes"],
            "content_sha256": pin["content_sha256"],
        }:
            raise common.GPUProjectionContractError(
                f"Workspace model payload drift for {model_key}"
            )
        expected_paths.update(
            f"{pin['path']}/{record['path']}" for record in observed["files"]
        )

    gpu_return_manifest = None
    if require_gpu_return:
        gpu_return_manifest, _ = encoder.validate_gpu_return(
            policy, staged_transfer, staged_parts, workspace / "gpu_return"
        )
        expected_paths.update(
            f"gpu_return/{path.relative_to(workspace / 'gpu_return').as_posix()}"
            for path in included_files(workspace / "gpu_return")
        )
    elif (workspace / "gpu_return").exists():
        raise common.GPUProjectionContractError(
            "GPU return exists before the authorized encoding run"
        )

    actual_paths = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise common.GPUProjectionContractError(
            "Isolated GPU workspace contains an undeclared file"
        )
    return staged_transfer, gpu_return_manifest


def materialize_workspace(
    policy: Mapping[str, Any],
    transfer_root: Path,
    workspace: Path,
    authorized_runner: Mapping[str, Any],
) -> dict[str, Any]:
    source_root = common.ROOT.resolve()
    workspace = workspace.resolve()
    try:
        workspace.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise common.GPUProjectionContractError("GPU workspace must be outside the repository")
    if workspace.exists():
        raise FileExistsError("GPU workspace must not exist before materialization")
    transfer_manifest, _parts = encoder.validate_transfer(policy, transfer_root)
    runner_path = verify_runner_record(source_root, authorized_runner)
    workspace.mkdir(parents=True)
    try:
        staged_files = [
            copy_static_file(source_root, workspace, relative)
            for relative in STATIC_FILES
        ]
        runner_relative = str(authorized_runner["path"])
        if runner_relative in STATIC_FILES:
            raise common.GPUProjectionContractError("Authorized runner duplicates static files")
        staged_files.append(copy_static_file(source_root, workspace, runner_relative))
        transfer_destination = workspace / "transfer"
        transfer_files = hardlink_tree(transfer_root, transfer_destination)
        model_file_count = 0
        model_files: list[Path] = []
        step7_policy = step7_common.load_policy()
        for model_key, pin in policy["model_payloads"].items():
            source_model = safe_path(source_root, pin["path"])
            observed = step7_common.validate_model_payload(
                model_key, step7_policy["embedding_models"][model_key]
            )
            if {
                key: observed[key]
                for key in ("file_count", "total_size_bytes", "content_sha256")
            } != {
                "file_count": pin["file_count"],
                "total_size_bytes": pin["total_size_bytes"],
                "content_sha256": pin["content_sha256"],
            }:
                raise common.GPUProjectionContractError("Source model payload drift")
            linked = hardlink_tree(source_model, safe_path(workspace, pin["path"]))
            model_file_count += len(linked)
            model_files.extend(linked)
        staged_transfer, _ = verify_materialized_workspace(
            policy,
            transfer_root,
            workspace,
            authorized_runner,
            require_gpu_return=False,
        )
        if staged_transfer != transfer_manifest:
            raise common.GPUProjectionContractError("Staged transfer drift")
    except BaseException:
        shutil.rmtree(workspace, ignore_errors=True)
        raise
    return {
        "status": "MATERIALIZED_OPAQUE_GPU_WORKSPACE_NO_EXECUTION",
        "workspace": str(workspace),
        "static_file_count": len(staged_files),
        "hardlinked_model_file_count": model_file_count,
        "transfer_manifest_canonical_self_hash": transfer_manifest[
            "canonical_self_hash"
        ],
        "formal_encoding_executed": False,
    }


def collect_gpu_return(
    policy: Mapping[str, Any],
    transfer_root: Path,
    workspace: Path,
    destination: Path,
    authorized_runner: Mapping[str, Any],
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError("GPU return destination already exists")
    transfer_manifest, parts = encoder.validate_transfer(policy, transfer_root)
    staged_transfer, verified_root_manifest = verify_materialized_workspace(
        policy,
        transfer_root,
        workspace,
        authorized_runner,
        require_gpu_return=True,
    )
    if staged_transfer != transfer_manifest:
        raise common.GPUProjectionContractError("Workspace transfer changed after staging")
    workspace_return = workspace / "gpu_return"
    root_manifest, _values = encoder.validate_gpu_return(
        policy, staged_transfer, parts, workspace_return
    )
    if root_manifest != verified_root_manifest:
        raise common.GPUProjectionContractError("Verified GPU return changed before collect")
    shutil.copytree(workspace_return, destination)
    try:
        copied_manifest, _copied_values = encoder.validate_gpu_return(
            policy, transfer_manifest, parts, destination
        )
        if copied_manifest != root_manifest:
            raise common.GPUProjectionContractError("Collected GPU return drift")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {
        "status": "COLLECTED_VERIFIED_COMPACT_GPU_RETURN",
        "gpu_return_manifest_canonical_self_hash": root_manifest[
            "canonical_self_hash"
        ],
        "temporary_chunk_or_embedding_files_collected": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-contract",))
    parser.parse_args()
    policy = common.load_policy()
    print(
        __import__("json").dumps(
            {
                "status": "PASSED_GPU_MATERIALIZER_CONTRACT_NO_FORMAL_EXECUTION",
                "gpu_policy_canonical_self_hash": policy["canonical_self_hash"],
                "formal_projection_executed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
