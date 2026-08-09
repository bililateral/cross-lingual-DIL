#!/usr/bin/env python3
"""Record full tests and freeze the v1.12 seed-only formal prelock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_prelock_evidence as authorization
import step28_v13_v1_12_preceremony as preceremony


ROOT = Path(__file__).resolve().parents[1]
TEST_RECEIPT_PATH = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "design_preflights"
    / "v1_12_cleanroom_20260803"
    / "full_repository_test_receipt.json"
)
TWO_WORLD_RECEIPT_PATH = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "design_preflights"
    / "v1_12_cleanroom_20260803"
    / "two_world_persisted_stage_receipt.postclosure_20260809.json"
)
SHORTCUT_RECEIPT_PATH = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "design_preflights"
    / "v1_12_cleanroom_20260803"
    / "exact_shortcut_preflight_receipt.postclosure_20260809.json"
)
NEW_SOURCE_PATHS = (
    ".gitignore",
    "docs/STEP28_V13_V1_12_FORMAL_AUTHORIZATION_OVERLAY_20260809.zh.md",
    "docs/STEP28_V13_V1_12_FORMAL_DATASET_BUILD_PLAN_20260803.zh.md",
    "docs/STEP28_V13_V1_12_TEXT_SHORTCUT_AUDIT_CONTRACT_20260808.zh.md",
    "reports/step28_synthetic_chinese_dataset/design_preflights/v1_12_cleanroom_20260803/exact_shortcut_preflight_receipt.postclosure_20260809.json",
    "reports/step28_synthetic_chinese_dataset/design_preflights/v1_12_cleanroom_20260803/external_interruption_receipt.json",
    "reports/step28_synthetic_chinese_dataset/design_preflights/v1_12_cleanroom_20260803/historical_manifest_waiver_receipt.json",
    "reports/step28_synthetic_chinese_dataset/design_preflights/v1_12_cleanroom_20260803/text_shortcut_preflight_receipt.json",
    "reports/step28_synthetic_chinese_dataset/design_preflights/v1_12_cleanroom_20260803/two_world_persisted_stage_receipt.postclosure_20260809.json",
    "schema/step28_v13_v1_12_cleanroom_preceremony_policy.json",
    "schema/step28_v13_v1_12_formal_build_draft.json",
    "schema/step28_v13_v1_12_text_shortcut_audit_policy.json",
    "scripts/step28_v13_v1_12_assignment_null.py",
    "scripts/step28_v13_v1_12_counterfactual_text.py",
    "scripts/step28_v13_v1_12_exact_shortcut_preflight.py",
    "scripts/step28_v13_v1_12_close_failed_run.py",
    "scripts/step28_v13_v1_12_finalize_release.py",
    "scripts/step28_v13_v1_12_formal_common.py",
    "scripts/step28_v13_v1_12_formal_executor.py",
    "scripts/step28_v13_v1_12_formal_quality_audit.py",
    "scripts/step28_v13_v1_12_freeze_prelock.py",
    "scripts/step28_v13_v1_12_generate_split.py",
    "scripts/step28_v13_v1_12_historical_identity_coverage.py",
    "scripts/step28_v13_v1_12_prelock_evidence.py",
    "scripts/step28_v13_v1_12_preceremony.py",
    "scripts/step28_v13_v1_12_seed_ceremony.py",
    "scripts/step28_v13_v1_12_style_derangement.py",
    "scripts/step28_v13_v1_12_text_shortcut_preflight.py",
    "scripts/step28_v13_v1_12_text_shortcut_runner.py",
    "scripts/step28_v13_v1_12_unittest_json_runner.py",
    "tests/test_step28_v13_v1_12_authorization_overlay_contracts.py",
    "tests/test_step28_v13_v1_12_formal_build_contracts.py",
    "tests/test_step28_v13_v1_12_formal_execution_contracts.py",
    "tests/test_step28_v13_v1_12_preceremony_contracts.py",
    "tests/test_step28_v13_v1_12_text_shortcut_contracts.py",
)


class FreezeError(ValueError):
    """Raised when a reproducible prelock cannot be frozen."""


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _pin(path: Path, *, self_hash: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _repo_relative(path),
        "size_bytes": preceremony.stat_long_path(path).st_size,
        "sha256": preceremony.sha256_file(path),
    }
    if self_hash:
        document = preceremony.load_json_strict(path)
        preceremony.validate_canonical_self_hash(
            document, label=f"prelock input {path.name}"
        )
        result["canonical_self_hash"] = document["canonical_self_hash"]
    return result


def source_closure() -> dict[str, Any]:
    baseline = formal.load_and_validate_draft()["baseline"]["policy"]
    historical = baseline["reusable_historical_source_closure"]["members"]
    paths = {str(record["path"]) for record in historical}
    paths.update(NEW_SOURCE_PATHS)
    text_receipt = authorization.load_current_authorization_receipts_exact(
        dereference_waiver_state=False
    )["text_receipt"]
    paths.update(
        str(record["path"])
        for record in text_receipt["source_files"].values()
    )
    ordered = sorted(paths, key=lambda value: value.encode("utf-8"))
    if (
        any("c40" in value.casefold() for value in ordered)
        or any(re.search(r"v1_(?:[3-9]|10|11)(?:\D|$)", value) for value in ordered)
    ):
        raise FreezeError("Forbidden C40/failed-version source entered closure")
    members = [_pin(ROOT / relative) for relative in ordered]
    return {
        "member_count": len(members),
        "baseline_reusable_member_count": len(historical),
        "c40_member_count": 0,
        "failed_version_member_count": 0,
        "canonical_sha256": preceremony.canonical_sha256(members),
        "members": members,
    }


def _closure_git_status(closure: dict[str, Any]) -> list[str]:
    command = [
        "git",
        "status",
        "--porcelain=v1",
        "--",
        *[str(record["path"]) for record in closure["members"]],
    ]
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def require_source_closure_members_tracked(
    closure: dict[str, Any]
) -> None:
    expected = {str(record["path"]) for record in closure["members"]}
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", "ls-files", "--", *sorted(expected)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    tracked = {
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.strip()
    }
    if tracked != expected:
        raise FreezeError(
            "Every formal source-closure member must be Git-tracked"
        )


def record_full_repository_tests(output: Path = TEST_RECEIPT_PATH) -> dict[str, Any]:
    formal.require_canonical_path(
        output, TEST_RECEIPT_PATH, label="v1.12 full-test receipt"
    )
    if preceremony.exists_long_path(output):
        raise FreezeError(f"Refusing existing full-test receipt: {output}")
    authorization_receipts = (
        authorization.load_current_authorization_receipts_exact(
            dereference_waiver_state=True
        )
    )
    waiver = authorization_receipts["waiver_receipt"]
    before = source_closure()
    dirty = _closure_git_status(before)
    if dirty:
        raise FreezeError(
            "Commit the exact formal source closure before recording full tests"
        )
    require_source_closure_members_tracked(before)
    authorization.require_tracked_worktree_matches_head()
    test_suite_before = authorization.test_suite_source_closure()
    authorization.require_committed_clean_test_tree(test_suite_before)
    git_head_before = authorization.current_git_head()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
    start = time.perf_counter()
    runner_path = ROOT / "scripts/step28_v13_v1_12_unittest_json_runner.py"
    command = [
        sys.executable,
        str(runner_path),
        "--start-directory",
        "tests",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    elapsed = time.perf_counter() - start
    combined = result.stdout + result.stderr
    try:
        structured = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise FreezeError(
            "Structured full-test result is missing or ambiguous"
        ) from exc
    if not isinstance(structured, dict):
        raise FreezeError("Structured full-test result must be one object")
    failures = list(structured.get("failure_ids", []))
    errors = list(structured.get("error_ids", []))
    skipped_rows = list(structured.get("skipped", []))
    fixture_skipped_rows = list(structured.get("fixture_skipped", []))
    expected_failures = list(structured.get("expected_failure_ids", []))
    unexpected_successes = list(structured.get("unexpected_success_ids", []))
    failed_subtests = list(structured.get("failed_subtest_ids", []))
    skipped_subtests = list(structured.get("skipped_subtest_ids", []))
    started_ids = list(structured.get("started_test_ids", []))
    success_ids = list(structured.get("success_ids", []))
    test_count = int(structured.get("tests_run", -1))
    skipped_ids = [str(row.get("id", "")) for row in skipped_rows]
    fixture_skipped_ids = [
        str(row.get("id", "")) for row in fixture_skipped_rows
    ]
    if (
        result.returncode != 1
        or structured.get("was_successful") is not False
        or failures != [authorization.WAIVED_TEST_ID]
        or errors
        or expected_failures
        or unexpected_successes
        or failed_subtests
        or skipped_subtests
        or skipped_rows != authorization.EXPECTED_STARTED_SKIPPED
        or fixture_skipped_rows != authorization.EXPECTED_FIXTURE_SKIPPED
        or test_count < 1
        or len(started_ids) != test_count
        or len(set(started_ids)) != test_count
        or len(success_ids) + len(skipped_ids) + 1 != test_count
    ):
        raise FreezeError(
            "Full repository tests do not match the exact historical boundary: "
            + json.dumps(
                {
                    "return_code": result.returncode,
                    "tests_run": test_count,
                    "failure_ids": failures,
                    "error_ids": errors,
                    "skipped_ids": skipped_ids,
                    "fixture_skipped_ids": fixture_skipped_ids,
                    "expected_failure_ids": expected_failures,
                    "unexpected_success_ids": unexpected_successes,
                    "failed_subtest_ids": failed_subtests,
                    "skipped_subtest_ids": skipped_subtests,
                    "started_count": len(started_ids),
                    "success_count": len(success_ids),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    targeted_ids = sorted(
        test_id
        for test_id in started_ids
        if "test_step28_v13_v1_12_" in test_id
    )
    targeted_successes = sorted(
        test_id
        for test_id in success_ids
        if "test_step28_v13_v1_12_" in test_id
    )
    targeted_failures = sorted(
        test_id for test_id in failures if "test_step28_v13_v1_12_" in test_id
    )
    targeted_errors = sorted(
        test_id for test_id in errors if "test_step28_v13_v1_12_" in test_id
    )
    targeted_skips = sorted(
        test_id for test_id in skipped_ids if "test_step28_v13_v1_12_" in test_id
    )
    if (
        not targeted_ids
        or targeted_successes != targeted_ids
        or targeted_failures
        or targeted_errors
        or targeted_skips
    ):
        raise FreezeError("Current Step28-v13 v1.12 targeted tests did not all pass")
    after = source_closure()
    test_suite_after = authorization.test_suite_source_closure()
    git_head_after = authorization.current_git_head()
    if (
        before != after
        or _closure_git_status(after)
        or test_suite_before != test_suite_after
        or git_head_before != git_head_after
    ):
        raise FreezeError("Source/test closure or Git HEAD changed during full tests")
    require_source_closure_members_tracked(after)
    authorization.require_tracked_worktree_matches_head()
    authorization.require_committed_clean_test_tree(test_suite_after)
    authorization.load_current_authorization_receipts_exact(
        dereference_waiver_state=True
    )
    raw_skipped = len(skipped_ids)
    raw_fixture_skipped = len(fixture_skipped_ids)
    raw_passed = len(success_ids)
    receipt = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-09-step28-v13-v1-12-full-tests-v3",
            "status": "PASS_FULL_REPOSITORY_TESTS",
            "status_semantics": (
                "PASS_WITH_ONE_EXACT_HISTORICAL_MANIFEST_FAILURE_WAIVER"
            ),
            "count_semantics": (
                "raw_tests_use_unittest_testsRun; raw_skips_cover_started_tests; "
                "raw_fixture_skips_are_nonstarted_events; compatibility_skipped_"
                "count_includes_one_accepted_waiver"
            ),
            "command": " ".join(command),
            "git_head": git_head_before,
            "source_closure_canonical_sha256": before["canonical_sha256"],
            "source_closure_member_count": before["member_count"],
            "test_suite_source_closure": test_suite_before,
            "test_count": test_count,
            "passed_count": raw_passed,
            "skipped_count": raw_skipped + 1,
            "failed_count": 0,
            "error_count": 0,
            "raw_test_count": test_count,
            "raw_passed_count": raw_passed,
            "raw_skipped_count": raw_skipped,
            "raw_fixture_skipped_count": raw_fixture_skipped,
            "raw_failed_count": 1,
            "raw_error_count": 0,
            "raw_failure_ids": failures,
            "raw_error_ids": errors,
            "raw_skipped_ids": skipped_ids,
            "raw_fixture_skipped_ids": fixture_skipped_ids,
            "raw_expected_failure_ids": expected_failures,
            "raw_unexpected_success_ids": unexpected_successes,
            "raw_failed_subtest_ids": failed_subtests,
            "raw_skipped_subtest_ids": skipped_subtests,
            "accepted_waived_failure_count": 1,
            "accepted_waived_failure_ids": failures,
            "historical_manifest_waiver": dict(
                authorization.WAIVER_RECEIPT_PIN
            ),
            "current_v1_12_targeted_tests": {
                "tests_run": len(targeted_ids),
                "passed": len(targeted_successes),
                "skipped_ids": targeted_skips,
                "failure_ids": targeted_failures,
                "error_ids": targeted_errors,
            },
            "subprocess_return_code": result.returncode,
            "raw_subprocess_return_code": result.returncode,
            "test_runner_reported_seconds": float(
                structured.get("wall_seconds", -1.0)
            ),
            "wall_seconds": elapsed,
            "raw_structured_result": structured,
            "structured_result_sha256": preceremony.canonical_sha256(
                structured
            ),
            "captured_output_sha256": hashlib.sha256(
                combined.encode("utf-8")
            ).hexdigest(),
            "runtime_versions": formal.runtime_versions(),
            "warning_policy": "non_optimizer_environment_warnings_do_not_mask_failures",
            "formal_seed_or_key_access": False,
            "formal_rows_produced": 0,
        }
    )
    preceremony.validate_canonical_self_hash(
        receipt, label="full repository test receipt"
    )
    authorization.validate_full_test_receipt(receipt, waiver)
    preceremony.write_bytes_no_replace_long_path(
        output,
        (
            json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        ),
    )
    return receipt


def validate_evidence_source_pins(
    *,
    two_world: dict[str, Any],
    shortcut: dict[str, Any],
    tests: dict[str, Any],
    closure: dict[str, Any],
) -> None:
    """Reject design evidence produced from any earlier source/draft byte."""

    draft_sha256 = preceremony.sha256_file(formal.DEFAULT_DRAFT_PATH)
    common_sha256 = preceremony.sha256_file(
        ROOT / "scripts/step28_v13_v1_12_formal_common.py"
    )
    versions = formal.runtime_versions()
    if (
        two_world.get("producer_path")
        != "scripts/step28_v13_v1_12_generate_split.py"
        or two_world.get("producer_sha256")
        != preceremony.sha256_file(
            ROOT / "scripts/step28_v13_v1_12_generate_split.py"
        )
        or two_world.get("formal_common_sha256") != common_sha256
        or two_world.get("formal_build_draft_sha256") != draft_sha256
        or two_world.get("runtime_versions") != versions
        or shortcut.get("producer_path")
        != "scripts/step28_v13_v1_12_exact_shortcut_preflight.py"
        or shortcut.get("producer_sha256")
        != preceremony.sha256_file(
            ROOT / "scripts/step28_v13_v1_12_exact_shortcut_preflight.py"
        )
        or shortcut.get("formal_common_sha256") != common_sha256
        or shortcut.get("formal_build_draft_sha256") != draft_sha256
        or shortcut.get("runtime_versions") != versions
        or tests.get("source_closure_canonical_sha256")
        != closure["canonical_sha256"]
        or tests.get("runtime_versions") != versions
    ):
        raise FreezeError("Prelock evidence/source closure pin drift")


def build_prelock(output: Path = formal.DEFAULT_PRELOCK_PATH) -> dict[str, Any]:
    formal.require_canonical_path(
        output, formal.DEFAULT_PRELOCK_PATH, label="v1.12 formal prelock"
    )
    if preceremony.exists_long_path(output):
        legacy = formal.load_and_validate_prelock(output)
        start_path = ROOT / str(
            legacy["prelock"]["custody"]["seed_ceremony_start_receipt_path"]
        )
        validated = authorization.validate_authorization_prelock_document(
            legacy["prelock"],
            legacy_validation=legacy,
            dereference_waiver_state=not preceremony.exists_long_path(start_path),
        )
        return dict(validated["prelock"])
    validated_draft = formal.load_and_validate_draft()
    draft = validated_draft["draft"]
    if (
        draft.get("status")
        != "DRAFT_IMPLEMENTATION_NO_SEED_OR_DATA_AUTHORIZATION"
        or set(draft["authorizations"].values()) != {False}
        or set(draft["implementation_freeze"].values()) != {False}
    ):
        raise FreezeError("Formal build draft authorization boundary drift")
    closure = source_closure()
    if _closure_git_status(closure):
        raise FreezeError("Formal source closure must be committed and clean")
    require_source_closure_members_tracked(closure)
    authorization.require_tracked_worktree_matches_head()
    two_world = preceremony.load_json_strict(TWO_WORLD_RECEIPT_PATH)
    shortcut = preceremony.load_json_strict(SHORTCUT_RECEIPT_PATH)
    tests = preceremony.load_json_strict(TEST_RECEIPT_PATH)
    for document, label in (
        (two_world, "two-world receipt"),
        (shortcut, "shortcut receipt"),
        (tests, "full-test receipt"),
    ):
        preceremony.validate_canonical_self_hash(document, label=label)
    validate_evidence_source_pins(
        two_world=two_world,
        shortcut=shortcut,
        tests=tests,
        closure=closure,
    )
    authorization_receipts = (
        authorization.load_current_authorization_receipts_exact(
            dereference_waiver_state=True
        )
    )
    waiver = authorization_receipts["waiver_receipt"]
    authorization.validate_full_test_receipt(tests, waiver)
    release_inputs = (
        "reports/step28_synthetic_chinese_dataset/release_inputs/"
        "v1_12_cleanroom_20260803"
    )
    private_root = str(draft["release"]["private_root"])
    prelock = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-formal-prelock-v1",
            "status": "READY_FOR_SEED_CEREMONY_ONLY",
            "run_id": draft["run_id"],
            "authorizations": {
                "formal_seed_ceremony": True,
                "formal_train_generation": False,
                "formal_development_generation": False,
                "formal_audit_a_generation": False,
                "formal_audit_b_generation": False,
                "model_training": False,
                "audit_truth_unsealing": False,
            },
            "formal_build_draft": _pin(
                formal.DEFAULT_DRAFT_PATH, self_hash=True
            ),
            "design_evidence_role": authorization.DESIGN_EVIDENCE_ROLE,
            "design_evidence": {
                "two_world_persisted_stage": _pin(
                    TWO_WORLD_RECEIPT_PATH, self_hash=True
                ),
                "exact_shortcut_preflight": _pin(
                    SHORTCUT_RECEIPT_PATH, self_hash=True
                ),
                "full_repository_tests": _pin(
                    TEST_RECEIPT_PATH, self_hash=True
                ),
            },
            "authorization_evidence_role": (
                authorization.AUTHORIZATION_EVIDENCE_ROLE
            ),
            "authorization_overlay_contract": _pin(
                authorization.OVERLAY_PATH
            ),
            "authorization_evidence": {
                "final_text_shortcut_preflight": _pin(
                    authorization.TEXT_RECEIPT_PATH, self_hash=True
                ),
                "external_interruption": _pin(
                    authorization.INTERRUPTION_RECEIPT_PATH, self_hash=True
                ),
                "historical_manifest_waiver": _pin(
                    authorization.WAIVER_RECEIPT_PATH, self_hash=True
                ),
            },
            "source_closure": closure,
            "dependency_versions": formal.runtime_versions(),
            "custody": {
                "public_root": draft["release"]["public_root"],
                "private_root": private_root,
                "private_seed_bundle_root": f"{private_root}/seed_custody",
                "private_seed_stage_root": f"{private_root}/_seed_ceremony_stage",
                "seed_ceremony_start_receipt_path": f"{release_inputs}/seed_ceremony_start_receipt.json",
                "public_ceremony_receipt_path": f"{release_inputs}/seed_ceremony_receipt.json",
                "train_development_execution_lock_path": f"{release_inputs}/train_development_execution_lock.json",
                "train_development_quality_receipt_path": f"{release_inputs}/train_development_quality_gate.json",
                "audit_a_generation_lock_path": f"{release_inputs}/audit_a_generation_lock.json",
                "audit_b_generation_lock_path": f"{release_inputs}/audit_b_generation_lock.json",
                "permanent_failure_receipt_path": f"{release_inputs}/permanent_failure_receipt.json",
                "master_mounted_to_generator": False,
                "master_mounted_to_model": False,
                "one_draw_per_split_no_retry": True,
                "private_root_git_ignored": True,
            },
            "git_head": tests["git_head"],
            "formal_master_or_capability_created": False,
        }
    )
    authorization.validate_authorization_prelock_document(
        prelock,
        legacy_validation={
            "test_receipt": tests,
            "source_members": closure["members"],
        },
        dereference_waiver_state=True,
    )
    preceremony.write_bytes_no_replace_long_path(
        output,
        (
            json.dumps(
                prelock,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        ),
    )
    legacy = formal.load_and_validate_prelock(output)
    authorization.validate_authorization_prelock_document(
        legacy["prelock"],
        legacy_validation=legacy,
        dereference_waiver_state=True,
    )
    return prelock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--record-full-tests", action="store_true")
    action.add_argument("--build-prelock", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.record_full_tests:
        receipt = record_full_repository_tests(
            args.output.resolve() if args.output else TEST_RECEIPT_PATH
        )
        print(
            receipt["status"],
            receipt["test_count"],
            receipt["skipped_count"],
        )
    else:
        prelock = build_prelock(
            args.output.resolve() if args.output else formal.DEFAULT_PRELOCK_PATH
        )
        print(
            prelock["status"],
            prelock["source_closure"]["member_count"],
            prelock["canonical_self_hash"],
        )


if __name__ == "__main__":
    main()
