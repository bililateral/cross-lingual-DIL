#!/usr/bin/env python3
"""Issue one private V9.4 method-root build authorization after code freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "schema" / "step28_v13_v1_13_v9_4_method_root_policy.json"
PREBUILD_RESULT_PATH = (
    ROOT / "reports" / "step28_synthetic_chinese_dataset"
    / "v9_4_prebuild_gate_attempt3_20260828" / "prebuild_gate_result.json"
)
TIME_KEY_PATH = (
    ROOT / "private_custody"
    / "step28_v13_v1_13_v9_4_prebuild_gate_attempt3_20260828"
    / "time_key.consumed.bin"
)
AUTHORITY_ROOT = (
    ROOT / "private_custody"
    / "step28_v13_v1_13_v9_4_method_root_authority_attempt1_20260828"
)
KEY_NAMES = ("text", "identity", "style", "audit_schedule", "uid")


class MethodRootAuthorityError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise MethodRootAuthorityError(f"JSON object required: {path}")
    return value


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def require_clean_worktree() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    if status:
        raise MethodRootAuthorityError(
            "Random authority may be issued only from an exactly clean worktree"
        )


def issue() -> dict[str, Any]:
    require_clean_worktree()
    policy = read_json(POLICY_PATH)
    result = read_json(PREBUILD_RESULT_PATH)
    auth_path = ROOT / str(policy["formal_authorization_path"])
    output_root = ROOT / str(policy["formal_output_root"])
    private_root = ROOT / str(policy["formal_private_root"])
    if any(path.exists() for path in (auth_path, AUTHORITY_ROOT, output_root, private_root)):
        raise MethodRootAuthorityError("Authorization, key, or output path already exists")
    if (
        result.get("status") != "PASSED_PREBUILD_SHORTCUT_GATE"
        or result.get("decision", {}).get("method_root_build_eligible") is not True
        or result.get("time_index_continuation", {}).get("eligible") is not True
    ):
        raise MethodRootAuthorityError("V9.4 prebuild result does not permit an application")
    time_key = TIME_KEY_PATH.read_bytes() if TIME_KEY_PATH.is_file() else b""
    expected_time = result["time_index_continuation"]["commitment_sha256"]
    if len(time_key) != 32 or hashlib.sha256(time_key).hexdigest() != expected_time:
        raise MethodRootAuthorityError("Retained time authority drift")
    AUTHORITY_ROOT.mkdir(parents=True)
    key_files: dict[str, dict[str, str]] = {}
    commitments = {expected_time}
    try:
        for name in KEY_NAMES:
            while True:
                key = secrets.token_bytes(32)
                commitment = hashlib.sha256(key).hexdigest()
                if commitment not in commitments:
                    break
            commitments.add(commitment)
            path = AUTHORITY_ROOT / f"{name}_key.bin"
            path.write_bytes(key)
            key_files[name] = {
                "path": path.relative_to(ROOT).as_posix(),
                "commitment_sha256": commitment,
            }
        payload: dict[str, Any] = {
            "version": "2026-08-28-step28-v13-v1-13-v9-4-method-root-authority-attempt1",
            "status": "AUTHORIZED_ONCE_NOT_CONSUMED",
            "implementation_commit": git_head(),
            "policy_sha256": sha256_file(POLICY_PATH),
            "prebuild_result_sha256": sha256_file(PREBUILD_RESULT_PATH),
            "output_root": str(policy["formal_output_root"]),
            "private_root": str(policy["formal_private_root"]),
            "key_files": key_files,
            "time_key": {
                "path": TIME_KEY_PATH.relative_to(ROOT).as_posix(),
                "commitment_sha256": expected_time,
                "prebuild_gate_rerun_authorized": False,
                "method_root_continuation_only": True,
            },
        }
        payload["canonical_self_hash"] = canonical_sha256(payload)
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        with auth_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        return {
            "status": payload["status"],
            "authorization_path": auth_path.relative_to(ROOT).as_posix(),
            "canonical_self_hash": payload["canonical_self_hash"],
            "key_commitments": {
                name: key_files[name]["commitment_sha256"] for name in KEY_NAMES
            },
            "time_key_commitment": expected_time,
        }
    except BaseException:
        if AUTHORITY_ROOT.exists():
            for path in AUTHORITY_ROOT.iterdir():
                if path.is_file():
                    path.unlink()
            AUTHORITY_ROOT.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(issue(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
