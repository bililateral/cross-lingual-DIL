#!/usr/bin/env python3
"""Run current and frozen-V9.1 regressions as two explicit partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from typing import Any, Iterable, Mapping
import uuid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT / "schema" / "step28_v13_v1_13_regression_partition_v1.json"
)
RESULT_VERSION = "2026-08-30-step28-v13-v1-13-regression-result-v1"


class RegressionPartitionError(RuntimeError):
    """Raised when the frozen regression partition cannot be reproduced."""


class StructuredResult(unittest.TextTestResult):
    """Collect exact unittest outcomes without converting exclusions to skips."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.started_ids: list[str] = []
        self.success_ids: list[str] = []

    def startTest(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        self.started_ids.append(test.id())
        super().startTest(test)

    def addSuccess(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        self.success_ids.append(test.id())
        super().addSuccess(test)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: Iterable[str]) -> str:
    ordered = sorted(ids)
    return sha256_bytes(("\n".join(ordered) + "\n").encode("utf-8"))


def _check_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    if set(value) != expected:
        raise RegressionPartitionError(f"{context} schema drift")


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    path = path.resolve()
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegressionPartitionError("Cannot load regression partition contract") from exc
    if not isinstance(contract, dict):
        raise RegressionPartitionError("Regression partition contract must be an object")
    _check_exact_keys(
        contract,
        {
            "version",
            "status",
            "canonical_self_hash",
            "current_policy_pin",
            "partition_runner_pin",
            "partition_tests_pin",
            "historical_snapshot",
            "historical_test_selectors",
            "historical_expanded_test_count",
            "historical_expanded_test_ids_sha256",
            "current_expected_skip_boundary",
            "current_suite_rule",
            "reporting_rule",
            "training_authorized",
            "audit_truth_authorized",
        },
        "Regression partition contract",
    )
    if contract["version"] != "2026-08-30-step28-v13-v1-13-regression-partition-v1":
        raise RegressionPartitionError("Regression partition version drift")
    if contract["status"] != "FROZEN_REGRESSION_PARTITION_NO_SCIENTIFIC_AUTHORIZATION":
        raise RegressionPartitionError("Regression partition status drift")
    if contract["training_authorized"] is not False:
        raise RegressionPartitionError("Regression partition cannot authorize training")
    if contract["audit_truth_authorized"] is not False:
        raise RegressionPartitionError("Regression partition cannot open audit truth")
    unsigned = dict(contract)
    observed_self_hash = unsigned.pop("canonical_self_hash")
    if not isinstance(observed_self_hash, str) or len(observed_self_hash) != 64:
        raise RegressionPartitionError("Regression partition self hash is malformed")
    if sha256_bytes(canonical_json_bytes(unsigned)) != observed_self_hash:
        raise RegressionPartitionError("Regression partition self hash drift")
    selectors = contract["historical_test_selectors"]
    if (
        not isinstance(selectors, list)
        or not selectors
        or any(not isinstance(value, str) or not value for value in selectors)
        or len(selectors) != len(set(selectors))
    ):
        raise RegressionPartitionError("Historical selectors are not unique strings")
    skip_boundary = contract["current_expected_skip_boundary"]
    if not isinstance(skip_boundary, dict):
        raise RegressionPartitionError("Current expected skip boundary must be an object")
    _check_exact_keys(
        skip_boundary,
        {
            "fixture_selector",
            "fixture_unstarted_test_count",
            "fixture_unstarted_test_ids_sha256",
            "skip_event_count",
            "skip_events_sha256",
        },
        "Current expected skip boundary",
    )
    if (
        not isinstance(skip_boundary["fixture_selector"], str)
        or not skip_boundary["fixture_selector"]
        or type(skip_boundary["fixture_unstarted_test_count"]) is not int
        or skip_boundary["fixture_unstarted_test_count"] <= 0
        or type(skip_boundary["skip_event_count"]) is not int
        or skip_boundary["skip_event_count"] <= 0
        or any(
            not isinstance(skip_boundary[key], str)
            or len(skip_boundary[key]) != 64
            for key in (
                "fixture_unstarted_test_ids_sha256",
                "skip_events_sha256",
            )
        )
    ):
        raise RegressionPartitionError("Current expected skip boundary is malformed")
    return contract


def validate_file_pin(root: Path, pin: Mapping[str, Any], prefix: str = "") -> None:
    path_key = f"{prefix}path"
    size_key = f"{prefix}size_bytes"
    sha_key = f"{prefix}sha256"
    path = (root / str(pin[path_key])).resolve()
    if root.resolve() not in path.parents:
        raise RegressionPartitionError("Pinned path escapes repository root")
    if not path.is_file():
        raise RegressionPartitionError(f"Pinned file is missing: {pin[path_key]}")
    if path.stat().st_size != pin[size_key] or sha256_file(path) != pin[sha_key]:
        raise RegressionPartitionError(f"Pinned file drift: {pin[path_key]}")


def git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _flatten(suite: unittest.TestSuite) -> Iterable[unittest.case.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


def _prepare_import_root(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    os.chdir(repo_root)
    current_root = ROOT.resolve()
    retained: list[str] = []
    for entry in sys.path:
        candidate = Path(entry or os.getcwd()).resolve()
        if candidate in {current_root, current_root / "scripts"}:
            continue
        retained.append(entry)
    sys.path[:] = [str(repo_root / "tests"), str(repo_root), *retained]


def load_historical_suite(
    contract: Mapping[str, Any], repo_root: Path
) -> tuple[unittest.TestSuite, list[str]]:
    _prepare_import_root(repo_root)
    combined = unittest.TestSuite()
    expanded_ids: list[str] = []
    for selector in contract["historical_test_selectors"]:
        selected = unittest.defaultTestLoader.loadTestsFromName(selector)
        selected_tests = list(_flatten(selected))
        if not selected_tests or any(test.__class__.__name__ == "_FailedTest" for test in selected_tests):
            raise RegressionPartitionError(f"Historical selector cannot load: {selector}")
        combined.addTest(selected)
        expanded_ids.extend(test.id() for test in selected_tests)
    if len(expanded_ids) != len(set(expanded_ids)):
        raise RegressionPartitionError("Historical selectors overlap")
    if len(expanded_ids) != contract["historical_expanded_test_count"]:
        raise RegressionPartitionError("Historical expanded test count drift")
    if ids_sha256(expanded_ids) != contract["historical_expanded_test_ids_sha256"]:
        raise RegressionPartitionError("Historical expanded test IDs drift")
    return combined, sorted(expanded_ids)


def load_current_suite(
    contract: Mapping[str, Any], repo_root: Path
) -> tuple[unittest.TestSuite, list[str], list[str]]:
    _prepare_import_root(repo_root)
    discovered = unittest.defaultTestLoader.discover(
        start_dir="tests", pattern="test*.py"
    )
    discovered_tests = list(_flatten(discovered))
    discovered_ids = [test.id() for test in discovered_tests]
    if len(discovered_ids) != len(set(discovered_ids)):
        raise RegressionPartitionError("Current discovery contains duplicate test IDs")
    _, historical_ids = load_historical_suite(contract, repo_root)
    historical_set = set(historical_ids)
    missing = historical_set.difference(discovered_ids)
    if missing:
        raise RegressionPartitionError("Historical partition is absent from current discovery")
    active_tests = [test for test in discovered_tests if test.id() not in historical_set]
    if len(active_tests) + len(historical_ids) != len(discovered_tests):
        raise RegressionPartitionError("Current/historical partition does not close")
    return unittest.TestSuite(active_tests), sorted(discovered_ids), historical_ids


def load_expected_fixture_unstarted_ids(
    contract: Mapping[str, Any], repo_root: Path
) -> list[str]:
    _prepare_import_root(repo_root)
    boundary = contract["current_expected_skip_boundary"]
    selected = unittest.defaultTestLoader.loadTestsFromName(
        boundary["fixture_selector"]
    )
    selected_tests = list(_flatten(selected))
    if not selected_tests or any(
        test.__class__.__name__ == "_FailedTest" for test in selected_tests
    ):
        raise RegressionPartitionError("Current fixture-skip selector cannot load")
    ids = sorted(test.id() for test in selected_tests)
    if len(ids) != len(set(ids)):
        raise RegressionPartitionError("Current fixture-skip selector overlaps itself")
    if len(ids) != boundary["fixture_unstarted_test_count"]:
        raise RegressionPartitionError("Current fixture-skip test count drift")
    if ids_sha256(ids) != boundary["fixture_unstarted_test_ids_sha256"]:
        raise RegressionPartitionError("Current fixture-skip test IDs drift")
    return ids


def _record_ids(records: list[tuple[Any, str]]) -> list[str]:
    return sorted({test.id() for test, _detail in records})


def run_suite(
    suite: unittest.TestSuite,
    *,
    mode: str,
    repo_root: Path,
    contract: Mapping[str, Any],
    discovered_ids: list[str],
    historical_ids: list[str],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    runner = unittest.TextTestRunner(
        stream=sys.stderr,
        verbosity=2,
        resultclass=StructuredResult,
        buffer=True,
    )
    result = runner.run(suite)
    elapsed = time.perf_counter() - started_at
    historical_set = set(historical_ids)
    if mode == "current_active":
        scheduled_ids = [
            test_id
            for test_id in discovered_ids
            if test_id not in historical_set
        ]
    else:
        scheduled_ids = list(discovered_ids)
    if len(result.started_ids) != len(set(result.started_ids)):
        raise RegressionPartitionError("A test started more than once")
    started_set = set(result.started_ids)
    if not started_set.issubset(scheduled_ids):
        raise RegressionPartitionError("A test outside the scheduled partition started")
    unstarted_ids = sorted(set(scheduled_ids).difference(started_set))
    skipped = sorted(
        ({"id": test.id(), "reason": reason} for test, reason in result.skipped),
        key=lambda row: (row["id"], row["reason"]),
    )
    payload: dict[str, Any] = {
        "version": RESULT_VERSION,
        "mode": mode,
        "contract_self_hash": contract["canonical_self_hash"],
        "repository_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "repository_tree": git_value(repo_root, "rev-parse", "HEAD^{tree}"),
        "discovered_test_count": len(discovered_ids),
        "discovered_test_ids_sha256": ids_sha256(discovered_ids),
        "historical_partition_test_count": len(historical_ids),
        "historical_partition_test_ids_sha256": ids_sha256(historical_ids),
        "scheduled_test_count": len(scheduled_ids),
        "scheduled_test_ids_sha256": ids_sha256(scheduled_ids),
        "tests_run": int(result.testsRun),
        "started_test_ids_sha256": ids_sha256(result.started_ids),
        "unstarted_test_count": len(unstarted_ids),
        "unstarted_test_ids": unstarted_ids,
        "unstarted_test_ids_sha256": ids_sha256(unstarted_ids),
        "success_count": len(result.success_ids),
        "failure_ids": _record_ids(result.failures),
        "error_ids": _record_ids(result.errors),
        "skipped": skipped,
        "skip_events_sha256": sha256_bytes(canonical_json_bytes(skipped)),
        "expected_failure_ids": _record_ids(result.expectedFailures),
        "unexpected_success_ids": sorted(test.id() for test in result.unexpectedSuccesses),
        "wall_seconds": elapsed,
        "was_successful": result.wasSuccessful(),
        "training_authorized": False,
        "audit_truth_authorized": False,
    }
    unsigned = dict(payload)
    payload["canonical_self_hash"] = sha256_bytes(canonical_json_bytes(unsigned))
    return payload


def write_new_result(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    reports_root = (ROOT / "reports").resolve()
    if reports_root not in path.parents:
        raise RegressionPartitionError("Regression result must be below reports/")
    if path.exists():
        raise RegressionPartitionError("Regression result already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".building")
    if temporary.exists():
        raise RegressionPartitionError("Regression result staging path already exists")
    try:
        temporary.write_bytes(canonical_json_bytes(dict(payload)) + b"\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_current(contract_path: Path, output: Path | None) -> int:
    contract = load_contract(contract_path)
    validate_file_pin(ROOT, contract["current_policy_pin"])
    validate_file_pin(ROOT, contract["partition_runner_pin"])
    validate_file_pin(ROOT, contract["partition_tests_pin"])
    suite, discovered_ids, historical_ids = load_current_suite(contract, ROOT)
    expected_unstarted_ids = load_expected_fixture_unstarted_ids(contract, ROOT)
    active_count = len(discovered_ids) - len(historical_ids)
    payload = run_suite(
        suite,
        mode="current_active",
        repo_root=ROOT,
        contract=contract,
        discovered_ids=discovered_ids,
        historical_ids=historical_ids,
    )
    skip_boundary = contract["current_expected_skip_boundary"]
    if payload["unstarted_test_ids"] != expected_unstarted_ids:
        raise RegressionPartitionError("Current fixture-skip boundary drift")
    if payload["tests_run"] != active_count - len(expected_unstarted_ids):
        raise RegressionPartitionError("Current active test count did not close")
    if (
        len(payload["skipped"]) != skip_boundary["skip_event_count"]
        or payload["skip_events_sha256"] != skip_boundary["skip_events_sha256"]
    ):
        raise RegressionPartitionError("Current documented skip events drift")
    payload["active_scheduled_test_count"] = active_count
    unsigned = dict(payload)
    unsigned.pop("canonical_self_hash")
    payload["canonical_self_hash"] = sha256_bytes(canonical_json_bytes(unsigned))
    if output is not None:
        write_new_result(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["was_successful"] else 1


def validate_historical_snapshot(
    contract: Mapping[str, Any], worktree: Path
) -> None:
    snapshot = contract["historical_snapshot"]
    if git_value(worktree, "rev-parse", "HEAD") != snapshot["commit"]:
        raise RegressionPartitionError("Historical worktree commit drift")
    if git_value(worktree, "rev-parse", "HEAD^{tree}") != snapshot["tree"]:
        raise RegressionPartitionError("Historical worktree tree drift")
    validate_file_pin(
        worktree,
        {
            "path": snapshot["policy_path"],
            "size_bytes": snapshot["policy_size_bytes"],
            "sha256": snapshot["policy_sha256"],
        },
    )
    validate_file_pin(
        worktree,
        {
            "path": snapshot["shared_text_source_path"],
            "size_bytes": snapshot["shared_text_source_size_bytes"],
            "sha256": snapshot["shared_text_source_sha256"],
        },
    )


def _safe_worktree_path() -> Path:
    temp_root = Path(tempfile.gettempdir()).resolve()
    candidate = (
        temp_root / f"step28-v9_1-history-{os.getpid()}-{uuid.uuid4().hex}"
    ).resolve()
    if temp_root not in candidate.parents or not candidate.name.startswith(
        "step28-v9_1-history-"
    ):
        raise RegressionPartitionError("Unsafe historical worktree path")
    if candidate.exists():
        raise RegressionPartitionError("Historical worktree path already exists")
    return candidate


def _remove_worktree(path: Path) -> None:
    temp_root = Path(tempfile.gettempdir()).resolve()
    resolved = path.resolve()
    if temp_root not in resolved.parents or not resolved.name.startswith(
        "step28-v9_1-history-"
    ):
        raise RegressionPartitionError("Refusing unsafe historical worktree cleanup")
    subprocess.run(
        ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(resolved)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if resolved.exists():
        shutil.rmtree(resolved)
    if resolved.exists():
        raise RegressionPartitionError("Historical worktree cleanup failed")
    subprocess.run(
        ["git", "-C", str(ROOT), "worktree", "prune"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def run_selected(contract_path: Path, repo_root: Path) -> int:
    contract = load_contract(contract_path)
    validate_historical_snapshot(contract, repo_root)
    suite, historical_ids = load_historical_suite(contract, repo_root)
    payload = run_suite(
        suite,
        mode="historical_v9_1_selected",
        repo_root=repo_root,
        contract=contract,
        discovered_ids=historical_ids,
        historical_ids=historical_ids,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["was_successful"] else 1


def run_historical(contract_path: Path, output: Path | None) -> int:
    contract = load_contract(contract_path)
    validate_file_pin(ROOT, contract["current_policy_pin"])
    validate_file_pin(ROOT, contract["partition_runner_pin"])
    validate_file_pin(ROOT, contract["partition_tests_pin"])
    worktree = _safe_worktree_path()
    environment = dict(os.environ)
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    payload: dict[str, Any] | None = None
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "worktree",
                "add",
                "--detach",
                str(worktree),
                contract["historical_snapshot"]["commit"],
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
        )
        validate_historical_snapshot(contract, worktree)
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "_selected",
                "--contract",
                str(contract_path.resolve()),
                "--repo-root",
                str(worktree),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stderr)
            raise RegressionPartitionError("Historical V9.1 regression failed")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise RegressionPartitionError("Historical child result framing drift")
        payload = json.loads(lines[0])
        if (
            payload.get("mode") != "historical_v9_1_selected"
            or payload.get("tests_run")
            != contract["historical_expanded_test_count"]
            or payload.get("failure_ids")
            or payload.get("error_ids")
            or payload.get("unstarted_test_ids")
            or payload.get("skipped")
            or payload.get("was_successful") is not True
        ):
            raise RegressionPartitionError("Historical V9.1 result did not close")
    finally:
        if worktree.exists():
            _remove_worktree(worktree)
    if payload is None:
        raise RegressionPartitionError("Historical V9.1 result is missing")
    payload["mode"] = "historical_v9_1_isolated_worktree"
    payload["isolated_worktree_removed"] = True
    unsigned = dict(payload)
    unsigned.pop("canonical_self_hash")
    payload["canonical_self_hash"] = sha256_bytes(canonical_json_bytes(unsigned))
    if output is not None:
        write_new_result(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("current", "historical", "_selected"))
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "current":
            if args.repo_root is not None:
                raise RegressionPartitionError("current mode does not accept --repo-root")
            status = run_current(args.contract, args.output)
        elif args.mode == "historical":
            if args.repo_root is not None:
                raise RegressionPartitionError("historical mode does not accept --repo-root")
            status = run_historical(args.contract, args.output)
        else:
            if args.repo_root is None or args.output is not None:
                raise RegressionPartitionError(
                    "_selected mode requires --repo-root and forbids --output"
                )
            status = run_selected(args.contract, args.repo_root)
    except (RegressionPartitionError, OSError, subprocess.SubprocessError) as exc:
        print(f"Regression partition failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
