#!/usr/bin/env python3
"""Prepare, encode, and finalize the public projection by file copy only."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import step28_v13_v1_13_v9_4_1_encode_base_projection_linux_v1 as gpu_encoder
import step28_v13_v1_13_v9_4_1_finalize_base_projection_v1 as base_finalizer
import step28_v13_v1_13_v9_4_1_freeze_identity_projection_v2 as identity_builder
import step28_v13_v1_13_v9_4_1_prepare_base_projection_v1 as base_preparer
import step28_v13_v1_13_v9_4_1_public_projection_common_v1 as common
import step28_v13_v1_13_v9_4_1_public_projection_gpu_common_v1 as gpu_common
import step28_v13_v1_13_v9_4_1_public_projection_protocol_v1 as protocol


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT / "reports/step28_model_experiment/v9_4_1_public_projection_portable_v2_work_20260831"
BUNDLE_ROOT = ROOT / "reports/step28_model_experiment/v9_4_1_public_projection_portable_v2_linux_bundle_20260831"
CPU_ROOT = WORK_ROOT / "cpu_stage"
IDENTITY_ROOT = WORK_ROOT / "identity_v1"
TRANSFER_ROOT = BUNDLE_ROOT / "transfer"
GPU_RETURN_ROOT = BUNDLE_ROOT / "gpu_return"
PUBLICATION_ROOT = ROOT / "reports/step28_model_experiment/v9_4_1_public_projection_v1_20260831"
PUBLICATION_BUILDING = PUBLICATION_ROOT.with_name(PUBLICATION_ROOT.name + ".building")
GPU_RECEIPT = ROOT / "reports/step28_model_experiment/v9_4_1_public_projection_portable_v2_gpu_receipt_20260831.json"


class PortableProjectionError(ValueError):
    pass


def prepare_windows() -> dict[str, Any]:
    if not sys.platform.startswith("win"):
        raise PortableProjectionError("prepare-windows runs only on Windows")
    if any(
        path.exists()
        for path in (
            WORK_ROOT,
            BUNDLE_ROOT,
            PUBLICATION_ROOT,
            PUBLICATION_BUILDING,
            GPU_RECEIPT,
        )
    ):
        raise PortableProjectionError("V2 work or output path already exists")
    policy = common.load_policy()
    WORK_ROOT.mkdir(parents=True)
    BUNDLE_ROOT.mkdir(parents=True)
    try:
        cpu, transfer = base_preparer.prepare_to_temporary(
            policy, CPU_ROOT, TRANSFER_ROOT
        )
        identity = identity_builder.build_to_temporary(policy, IDENTITY_ROOT)
        return {
            "status": "READY_TO_COPY_TRANSFER_TO_LINUX",
            "transfer_manifest_canonical_self_hash": transfer[
                "canonical_self_hash"
            ],
            "cpu_manifest_canonical_self_hash": cpu["canonical_self_hash"],
            "identity_manifest_canonical_self_hash": identity[
                "canonical_self_hash"
            ],
            "linux_git_required": False,
            "supervision_or_audit_truth_read": False,
        }
    except BaseException:
        shutil.rmtree(WORK_ROOT, ignore_errors=True)
        shutil.rmtree(BUNDLE_ROOT, ignore_errors=True)
        raise


def encode_linux() -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        raise PortableProjectionError("encode-linux runs only on Linux")
    if GPU_RETURN_ROOT.exists():
        raise PortableProjectionError("GPU return already exists")
    gpu_policy = gpu_common.load_policy()
    transfer, _parts = gpu_encoder.validate_transfer(gpu_policy, TRANSFER_ROOT)
    if transfer.get(
        "labels_controllers_membership_qrels_or_audit_truth_present"
    ) is not False:
        raise PortableProjectionError("Linux transfer is not label-free")
    try:
        result = gpu_encoder.encode_transfer_to_temporary(
            gpu_policy, TRANSFER_ROOT, GPU_RETURN_ROOT
        )
        return {
            "status": "COMPLETED_LINUX_LABSE6_ENCODING",
            "gpu_return_manifest_canonical_self_hash": result[
                "canonical_self_hash"
            ],
            "exact_runtime": result["exact_runtime"],
            "linux_git_read": False,
            "supervision_or_audit_truth_read": False,
            "model_parameters_updated": False,
        }
    except BaseException:
        shutil.rmtree(GPU_RETURN_ROOT, ignore_errors=True)
        raise


def finalize_windows() -> dict[str, Any]:
    if not sys.platform.startswith("win"):
        raise PortableProjectionError("finalize-windows runs only on Windows")
    if any(path.exists() for path in (PUBLICATION_ROOT, PUBLICATION_BUILDING, GPU_RECEIPT)):
        raise PortableProjectionError("V2 publication or receipt already exists")
    policy = common.load_policy()
    gpu_policy = gpu_common.load_policy()
    transfer, parts = gpu_encoder.validate_transfer(gpu_policy, TRANSFER_ROOT)
    gpu_manifest, _ = gpu_encoder.validate_gpu_return(
        gpu_policy, transfer, parts, GPU_RETURN_ROOT
    )
    PUBLICATION_BUILDING.mkdir(parents=True)
    try:
        base_finalizer.finalize_to_temporary(
            policy,
            CPU_ROOT,
            TRANSFER_ROOT,
            GPU_RETURN_ROOT,
            PUBLICATION_BUILDING / policy["formal_outputs"]["base_subdirectory"],
        )
        shutil.copytree(
            IDENTITY_ROOT,
            PUBLICATION_BUILDING / policy["formal_outputs"]["identity_subdirectory"],
        )
        publication = protocol.freeze_combined_manifest(policy, PUBLICATION_BUILDING)
        shutil.copyfile(GPU_RETURN_ROOT / "gpu_return_manifest.json", GPU_RECEIPT)
        PUBLICATION_BUILDING.replace(PUBLICATION_ROOT)
        shutil.rmtree(WORK_ROOT, ignore_errors=True)
        shutil.rmtree(BUNDLE_ROOT, ignore_errors=True)
        return {
            "status": "PUBLISHED_LABEL_FREE_PUBLIC_PROJECTION",
            "publication_manifest_canonical_self_hash": publication[
                "canonical_self_hash"
            ],
            "gpu_return_manifest_canonical_self_hash": gpu_manifest[
                "canonical_self_hash"
            ],
            "supervision_or_audit_truth_read": False,
            "model_parameters_updated": False,
            "training_authorized": False,
        }
    except BaseException:
        shutil.rmtree(PUBLICATION_BUILDING, ignore_errors=True)
        if not PUBLICATION_ROOT.exists():
            GPU_RECEIPT.unlink(missing_ok=True)
        raise


def validate_output() -> dict[str, Any]:
    return protocol.validate_publication(common.load_policy(), PUBLICATION_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare-windows", "encode-linux", "finalize-windows", "validate-output"),
    )
    command = parser.parse_args().command
    functions = {
        "prepare-windows": prepare_windows,
        "encode-linux": encode_linux,
        "finalize-windows": finalize_windows,
        "validate-output": validate_output,
    }
    print(json.dumps(functions[command](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
