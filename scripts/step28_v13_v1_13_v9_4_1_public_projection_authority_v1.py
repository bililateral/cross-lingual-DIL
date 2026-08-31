#!/usr/bin/env python3
"""Issue one non-supervised authority for the frozen public projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from typing import Any

import step28_v13_v1_13_v9_4_1_public_projection_authority_common_v1 as common
import step28_v13_v1_13_v9_4_1_public_projection_common_v1 as projection_common


CLAIM_VERSION = (
    "2026-08-31-step28-v13-v1-13-v9-4-1-public-projection-issuance-claim-v1"
)
FAILURE_VERSION = (
    "2026-08-31-step28-v13-v1-13-v9-4-1-public-projection-issuance-failure-v1"
)


def _all_issued_paths_absent(policy: dict[str, Any]) -> None:
    for label, path in common.issued_paths(policy).items():
        if path.exists():
            raise common.PublicProjectionAuthorityError(
                f"Authority issuance path already exists: {label}"
            )


def issue() -> dict[str, Any]:
    policy = common.load_policy()
    if common.git_status_lines():
        raise common.PublicProjectionAuthorityError(
            "Authority issuance requires an exactly clean implementation commit"
        )
    _all_issued_paths_absent(policy)
    paths = common.issued_paths(policy)
    issued = policy["issued_authorization_contract"]
    implementation_commit = common.git_head()
    implementation_tree = common.git_tree()
    paths["authority_root"].mkdir(parents=True)
    claim: dict[str, Any] = {
        "version": CLAIM_VERSION,
        "status": "PUBLIC_PROJECTION_AUTHORITY_ISSUANCE_CLAIMED",
        "issuance_ordinal": 1,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "projection_implementation_commit": common.PROJECTION_IMPLEMENTATION_COMMIT,
        "projection_implementation_tree": common.PROJECTION_IMPLEMENTATION_TREE,
        "policy_path": common.POLICY_PATH.relative_to(common.ROOT).as_posix(),
        "policy_sha256": common.sha256_file(common.POLICY_PATH),
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "authorization_path": issued["authorization_path"],
        "authority_root": issued["authority_root"],
        "key_path": issued["key_path"],
        "key_size_bytes": issued["key_size_bytes"],
        "candidate_draws_at_claim": 0,
        "output_root": policy["execution_paths"]["formal_output_root"],
        "building_root": policy["execution_paths"]["building_root"],
        "state_root": policy["execution_paths"]["state_root"],
        "rerun_authorized": False,
    }
    claim["canonical_self_hash"] = common.canonical_sha256(claim)
    claim_written = False
    key_written = False
    stage = "write_issuance_claim"
    commitment = ""
    try:
        common.write_json_exclusive(paths["issuance_claim"], claim)
        claim_written = True
        stage = "draw_projection_authority_once"
        raw_key = secrets.token_bytes(32)
        commitment = hashlib.sha256(raw_key).hexdigest()
        stage = "write_projection_authority"
        with paths["key"].open("xb") as stream:
            stream.write(raw_key)
        key_written = True
        public_policy = projection_common.load_policy()
        authorization: dict[str, Any] = {
            "version": issued["version"],
            "status": issued["status"],
            "issuance_ordinal": 1,
            "implementation_commit": implementation_commit,
            "implementation_tree": implementation_tree,
            "projection_implementation_commit": common.PROJECTION_IMPLEMENTATION_COMMIT,
            "projection_implementation_tree": common.PROJECTION_IMPLEMENTATION_TREE,
            "policy_sha256": common.sha256_file(common.POLICY_PATH),
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "public_policy_sha256": common.sha256_file(projection_common.POLICY_PATH),
            "public_policy_canonical_self_hash": public_policy["canonical_self_hash"],
            "issuance_claim_sha256": common.sha256_file(paths["issuance_claim"]),
            "implementation_files": common.implementation_file_records(policy),
            "authorization_state": {
                "formal_public_projection_authorized": True,
                "train_development_truth_authorized": False,
                "model_training_authorized": False,
                "audit_a_prediction_authorized": False,
                "audit_a_truth_authorized": False,
                "audit_b_prediction_authorized": False,
                "audit_b_truth_authorized": False,
            },
            "key_file": {
                "path": issued["key_path"],
                "commitment_sha256": commitment,
            },
            "output_root": policy["execution_paths"]["formal_output_root"],
            "building_root": policy["execution_paths"]["building_root"],
            "state_root": policy["execution_paths"]["state_root"],
        }
        authorization["canonical_self_hash"] = common.canonical_sha256(authorization)
        stage = "write_final_authorization"
        common.write_json_exclusive(paths["authorization"], authorization)
        return {
            "status": authorization["status"],
            "authorization_path": issued["authorization_path"],
            "authorization_canonical_self_hash": authorization[
                "canonical_self_hash"
            ],
            "key_commitment_sha256": commitment,
            "raw_key_returned": False,
            "formal_projection_executed": False,
            "supervision_or_audit_truth_read": False,
            "model_training_authorized": False,
        }
    except BaseException as exc:
        paths["authorization"].unlink(missing_ok=True)
        if key_written or paths["key"].exists():
            paths["key"].unlink(missing_ok=True)
        if claim_written:
            failure: dict[str, Any] = {
                "version": FAILURE_VERSION,
                "status": "PUBLIC_PROJECTION_AUTHORITY_ISSUANCE_FAILED_NO_RERUN",
                "issuance_claim_sha256": common.sha256_file(paths["issuance_claim"]),
                "failure_stage": stage,
                "exception_type": type(exc).__name__,
                "exception_message_sha256": hashlib.sha256(
                    str(exc).encode("utf-8")
                ).hexdigest(),
                "generated_commitment_sha256": commitment or None,
                "raw_key_material_retained": False,
                "formal_projection_executed": False,
                "rerun_authorized": False,
            }
            failure["canonical_self_hash"] = common.canonical_sha256(failure)
            try:
                common.write_json_exclusive(paths["issuance_failure"], failure)
            except BaseException:
                pass
        elif paths["authority_root"].exists():
            paths["authority_root"].rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(issue(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
