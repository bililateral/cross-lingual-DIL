#!/usr/bin/env python3
"""Prepare only the label-free identity inputs in an isolated Windows process."""

from __future__ import annotations

import argparse
import json
from typing import Any

import step28_v13_v1_13_v9_4_1_freeze_identity_projection_v2 as identity_builder
import step28_v13_v1_13_v9_4_1_public_projection_authority_common_v1 as authority
import step28_v13_v1_13_v9_4_1_public_projection_common_v1 as projection_common


def run_once() -> dict[str, Any]:
    policy = authority.load_policy()
    auth = authority.validate_authorization(policy, require_raw_key=False)
    authority.validate_consumption(policy, auth)
    execution = policy["execution_paths"]
    identity_root = (
        authority.resolve(execution["building_root"])
        / execution["identity_subdirectory"]
    )
    identity_manifest = identity_builder.build_to_temporary(
        projection_common.load_policy(), identity_root
    )
    return {
        "status": "PREPARED_LABEL_FREE_IDENTITY_INPUTS_IN_ISOLATED_PROCESS",
        "identity_manifest_canonical_self_hash": identity_manifest[
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
