#!/usr/bin/env python3
"""Prepare only the label-free base inputs in an isolated Windows process."""

from __future__ import annotations

import argparse
import json
from typing import Any

import step28_v13_v1_13_v9_4_1_prepare_base_projection_v1 as base_preparer
import step28_v13_v1_13_v9_4_1_public_projection_authority_common_v1 as authority
import step28_v13_v1_13_v9_4_1_public_projection_common_v1 as projection_common


def run_once() -> dict[str, Any]:
    policy = authority.load_policy()
    auth = authority.validate_authorization(policy, require_raw_key=False)
    authority.validate_consumption(policy, auth)
    execution = policy["execution_paths"]
    building = authority.resolve(execution["building_root"])
    cpu_root = building / execution["cpu_stage_subdirectory"]
    transfer_root = building / execution["transfer_subdirectory"]
    cpu_manifest, transfer_manifest = base_preparer.prepare_to_temporary(
        projection_common.load_policy(), cpu_root, transfer_root
    )
    return {
        "status": "PREPARED_LABEL_FREE_BASE_INPUTS_IN_ISOLATED_PROCESS",
        "cpu_manifest_canonical_self_hash": cpu_manifest["canonical_self_hash"],
        "transfer_manifest_canonical_self_hash": transfer_manifest[
            "canonical_self_hash"
        ],
        "supervision_or_audit_truth_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-once", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(run_once(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
