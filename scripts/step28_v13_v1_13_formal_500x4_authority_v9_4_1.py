#!/usr/bin/env python3
"""Issue the one-time private authority for the V9.4.1 formal 500x4 build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_formal_500x4_builder_v9_4_1 as builder


VERSION = builder.AUTHORIZATION_VERSION


class Formal500x4AuthorityError(ValueError):
    """Raised when the one-time authority cannot be issued safely."""


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def git_tree() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def require_clean_worktree() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    if status:
        raise Formal500x4AuthorityError(
            "Formal authority may be issued only from an exactly clean worktree"
        )


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def issue() -> dict[str, Any]:
    require_clean_worktree()
    policy = builder.validate_policy(formal=True)
    auth_path = ROOT / str(policy["formal_authorization_path"])
    authority_root = ROOT / str(policy["formal_authority_root"])
    claim_path = ROOT / str(policy["formal_issuance_claim_path"])
    output_root = ROOT / str(policy["formal_output_root"])
    private_root = ROOT / str(policy["formal_private_root"])
    consumption_path = ROOT / str(policy["formal_consumption_path"])
    failure_path = ROOT / str(policy["formal_failure_path"])
    key_cleanup_path = ROOT / str(policy["formal_key_cleanup_path"])
    completion_path = ROOT / str(policy["formal_completion_path"])
    if any(path.exists() for path in (
        auth_path, authority_root, claim_path, output_root, private_root,
        consumption_path, failure_path, key_cleanup_path, completion_path,
    )):
        raise Formal500x4AuthorityError(
            "Authorization, authority, output, private or consumption path exists"
        )
    quality = policy["method_qualification"]
    forbidden_commitments = builder.forbidden_authority_commitments(policy)
    authority_root.mkdir(parents=True)
    implementation_commit = git_head()
    implementation_tree = git_tree()
    policy_sha256 = builder.sha256_file(builder.POLICY_PATH)
    claim: dict[str, Any] = {
        "version": builder.ISSUANCE_CLAIM_VERSION,
        "status": "FORMAL_500X4_AUTHORITY_ISSUANCE_CLAIMED",
        "issuance_ordinal": 1,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "policy_path": builder.POLICY_PATH.relative_to(ROOT).as_posix(),
        "policy_sha256": policy_sha256,
        "authorization_path": policy["formal_authorization_path"],
        "authority_root": policy["formal_authority_root"],
        "output_root": policy["formal_output_root"],
        "private_root": policy["formal_private_root"],
        "key_names": list(policy["private_authority"]["key_names"]),
        "key_size_bytes": policy["private_authority"]["key_size_bytes"],
        "candidate_draws_at_claim": 0,
        "rerun_authorized": False,
    }
    claim["canonical_self_hash"] = builder.canonical_sha256(claim)
    key_files: dict[str, dict[str, str]] = {}
    commitments: set[str] = set()
    claim_written = False
    stage = "writing_issuance_claim"
    try:
        write_json_exclusive(claim_path, claim)
        claim_written = True
        for name in policy["private_authority"]["key_names"]:
            stage = f"drawing_{name}_authority_once"
            key = secrets.token_bytes(32)
            commitment = hashlib.sha256(key).hexdigest()
            if (
                commitment in commitments
                or commitment in forbidden_commitments
            ):
                raise Formal500x4AuthorityError(
                    "One-time authority draw is duplicate or forbidden"
                )
            commitments.add(commitment)
            path = authority_root / f"{name}_key.bin"
            stage = f"writing_{name}_authority"
            with path.open("xb") as stream:
                stream.write(key)
            key_files[name] = {
                "path": path.relative_to(ROOT).as_posix(),
                "commitment_sha256": commitment,
            }
        payload: dict[str, Any] = {
            "version": VERSION,
            "status": builder.AUTHORIZATION_STATUS,
            "implementation_commit": implementation_commit,
            "implementation_tree": implementation_tree,
            "policy_sha256": policy_sha256,
            "quality_result_sha256": quality["quality_result_sha256"],
            "root_manifest_sha256": quality["root_manifest_sha256"],
            "issuance_claim_sha256": builder.sha256_file(claim_path),
            "output_root": policy["formal_output_root"],
            "private_root": policy["formal_private_root"],
            "key_files": key_files,
        }
        payload["canonical_self_hash"] = builder.canonical_sha256(payload)
        stage = "writing_final_authorization"
        write_json_exclusive(auth_path, payload)
        return {
            "status": payload["status"],
            "authorization_path": auth_path.relative_to(ROOT).as_posix(),
            "canonical_self_hash": payload["canonical_self_hash"],
            "issuance_claim_sha256": payload["issuance_claim_sha256"],
            "key_commitments": {
                name: key_files[name]["commitment_sha256"]
                for name in policy["private_authority"]["key_names"]
            },
            "key_material_returned": False,
        }
    except BaseException as exc:
        if auth_path.exists():
            auth_path.unlink()
        if authority_root.exists():
            for path in authority_root.iterdir():
                if path.is_file() and path != claim_path:
                    path.unlink()
        if claim_written:
            failure: dict[str, Any] = {
                "version": (
                    "2026-08-29-step28-v13-v1-13-v9-4-1-formal-500x4-"
                    "authority-issuance-failure-v1"
                ),
                "status": "FORMAL_500X4_AUTHORITY_ISSUANCE_FAILED_NO_RERUN",
                "issuance_claim_sha256": builder.sha256_file(claim_path),
                "failure_stage": stage,
                "exception_type": type(exc).__name__,
                "exception_message_sha256": hashlib.sha256(
                    str(exc).encode("utf-8")
                ).hexdigest(),
                "generated_commitments": sorted(commitments),
                "raw_key_material_retained": False,
                "rerun_authorized": False,
            }
            failure["canonical_self_hash"] = builder.canonical_sha256(failure)
            try:
                write_json_exclusive(
                    authority_root / "formal_500x4_issuance.failed.json",
                    failure,
                )
            except BaseException:
                # The immutable claim still occupies this attempt and prevents retry.
                pass
        elif authority_root.exists():
            for path in authority_root.iterdir():
                if path.is_file():
                    path.unlink()
            authority_root.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(issue(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
