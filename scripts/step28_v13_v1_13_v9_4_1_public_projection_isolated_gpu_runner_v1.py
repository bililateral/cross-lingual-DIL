#!/usr/bin/env python3
"""Run exactly one opaque LaBSE projection inside the isolated workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_encode_base_projection_linux_v1 as encoder
import step28_v13_v1_13_v9_4_1_public_projection_gpu_common_v1 as gpu_common


def run_once() -> dict[str, object]:
    transfer_root = ROOT / "transfer"
    return_root = ROOT / "gpu_return"
    if return_root.exists():
        raise gpu_common.GPUProjectionContractError(
            "Isolated GPU return already exists"
        )
    policy = gpu_common.load_policy()
    manifest = encoder.encode_transfer_to_temporary(
        policy, transfer_root, return_root
    )
    return {
        "status": manifest["status"],
        "gpu_return_manifest_canonical_self_hash": manifest[
            "canonical_self_hash"
        ],
        "canonical_identifiers_or_split_names_read": False,
        "supervision_or_audit_truth_read": False,
        "model_parameters_updated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-once", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(run_once(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
