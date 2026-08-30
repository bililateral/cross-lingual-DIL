#!/usr/bin/env python3
"""Run the current regression after closed V1.12 workspace cleanup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_regression_partition_v1 as parent


DEFAULT_CONTRACT = (
    ROOT / "schema" / "step28_v13_v1_13_regression_partition_v2.json"
)
RESULT_VERSION = "2026-08-30-step28-v13-v1-13-regression-result-v2"
PARENT_SCHEDULING_MODE = "current_active"
RESULT_MODE = "current_active_post_cleanup"


class RegressionPartitionV2Error(RuntimeError):
    """Raised when the post-cleanup current regression cannot be reproduced."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise RegressionPartitionV2Error(f"{context} schema drift")


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    try:
        contract = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegressionPartitionV2Error("Cannot load V2 regression contract") from exc
    if not isinstance(contract, dict):
        raise RegressionPartitionV2Error("V2 regression contract must be an object")
    _exact_keys(
        contract,
        {
            "version",
            "status",
            "canonical_self_hash",
            "parent_contract_pin",
            "parent_runner_pin",
            "historical_result_pin",
            "partition_runner_pin",
            "partition_tests_pin",
            "current_discovery_boundary",
            "current_expected_skip_boundary",
            "removed_v1_12_boundary",
            "reporting_rule",
            "training_authorized",
            "audit_truth_authorized",
        },
        "V2 regression contract",
    )
    if contract["version"] != "2026-08-30-step28-v13-v1-13-regression-partition-v2":
        raise RegressionPartitionV2Error("V2 regression version drift")
    if contract["status"] != "FROZEN_POST_CLEANUP_CURRENT_REGRESSION":
        raise RegressionPartitionV2Error("V2 regression status drift")
    if contract["training_authorized"] is not False:
        raise RegressionPartitionV2Error("V2 regression cannot authorize training")
    if contract["audit_truth_authorized"] is not False:
        raise RegressionPartitionV2Error("V2 regression cannot open audit truth")
    unsigned = dict(contract)
    observed = unsigned.pop("canonical_self_hash")
    if parent.sha256_bytes(parent.canonical_json_bytes(unsigned)) != observed:
        raise RegressionPartitionV2Error("V2 regression self hash drift")
    for key in (
        "parent_contract_pin",
        "parent_runner_pin",
        "historical_result_pin",
        "partition_runner_pin",
        "partition_tests_pin",
    ):
        pin = contract[key]
        if not isinstance(pin, dict):
            raise RegressionPartitionV2Error(f"{key} must be an object")
        _exact_keys(pin, {"path", "size_bytes", "sha256"}, key)
    discovery = contract["current_discovery_boundary"]
    _exact_keys(
        discovery,
        {
            "discovered_test_count",
            "discovered_test_ids_sha256",
            "historical_partition_test_count",
            "historical_partition_test_ids_sha256",
            "active_scheduled_test_count",
            "active_scheduled_test_ids_sha256",
            "fixture_unstarted_test_count",
            "fixture_unstarted_test_ids_sha256",
        },
        "current_discovery_boundary",
    )
    skip = contract["current_expected_skip_boundary"]
    _exact_keys(skip, {"skip_event_count", "skip_events_sha256"}, "skip boundary")
    removed = contract["removed_v1_12_boundary"]
    _exact_keys(
        removed,
        {
            "removed_file_count",
            "removed_test_case_count",
            "removed_size_bytes",
            "predelete_manifest_sha256",
        },
        "removed_v1_12_boundary",
    )
    return contract


def validate_pins(contract: Mapping[str, Any]) -> None:
    for key in (
        "parent_contract_pin",
        "parent_runner_pin",
        "historical_result_pin",
        "partition_runner_pin",
        "partition_tests_pin",
    ):
        parent.validate_file_pin(ROOT, contract[key])


def load_current_suite(
    contract: Mapping[str, Any],
) -> tuple[Any, list[str], list[str], list[str]]:
    parent_contract = parent.load_contract(ROOT / contract["parent_contract_pin"]["path"])
    suite, discovered_ids, historical_ids = parent.load_current_suite(
        parent_contract, ROOT
    )
    active_ids = [test.id() for test in parent._flatten(suite)]
    boundary = contract["current_discovery_boundary"]
    checks = (
        (len(discovered_ids), boundary["discovered_test_count"]),
        (parent.ids_sha256(discovered_ids), boundary["discovered_test_ids_sha256"]),
        (len(historical_ids), boundary["historical_partition_test_count"]),
        (
            parent.ids_sha256(historical_ids),
            boundary["historical_partition_test_ids_sha256"],
        ),
        (len(active_ids), boundary["active_scheduled_test_count"]),
        (
            parent.ids_sha256(active_ids),
            boundary["active_scheduled_test_ids_sha256"],
        ),
    )
    if any(observed != expected for observed, expected in checks):
        raise RegressionPartitionV2Error("Post-cleanup discovery boundary drift")
    return suite, discovered_ids, historical_ids, active_ids


def validate_cleanup_absence() -> None:
    active_old = []
    for base in (ROOT / "schema", ROOT / "scripts", ROOT / "tests"):
        active_old.extend(base.glob("*v1_12*"))
    if active_old:
        raise RegressionPartitionV2Error("Closed V1.12 active files reappeared")


def run_parent_current_suite(
    suite: Any,
    contract: Mapping[str, Any],
    discovered_ids: list[str],
    historical_ids: list[str],
) -> dict[str, Any]:
    """Use the frozen parent's active-mode semantics, then name the V2 result."""
    payload = parent.run_suite(
        suite,
        mode=PARENT_SCHEDULING_MODE,
        repo_root=ROOT,
        contract=contract,
        discovered_ids=discovered_ids,
        historical_ids=historical_ids,
    )
    if payload.get("mode") != PARENT_SCHEDULING_MODE:
        raise RegressionPartitionV2Error("Parent active scheduling mode drift")
    payload["mode"] = RESULT_MODE
    return payload


def run_current(contract_path: Path, output: Path | None) -> int:
    contract = load_contract(contract_path)
    validate_pins(contract)
    validate_cleanup_absence()
    suite, discovered_ids, historical_ids, active_ids = load_current_suite(contract)
    parent_contract = parent.load_contract(ROOT / contract["parent_contract_pin"]["path"])
    expected_unstarted = parent.load_expected_fixture_unstarted_ids(
        parent_contract, ROOT
    )
    boundary = contract["current_discovery_boundary"]
    if (
        len(expected_unstarted) != boundary["fixture_unstarted_test_count"]
        or parent.ids_sha256(expected_unstarted)
        != boundary["fixture_unstarted_test_ids_sha256"]
    ):
        raise RegressionPartitionV2Error("Fixture unstarted boundary drift")
    parent.RESULT_VERSION = RESULT_VERSION
    payload = run_parent_current_suite(
        suite,
        contract,
        discovered_ids,
        historical_ids,
    )
    if payload["unstarted_test_ids"] != expected_unstarted:
        raise RegressionPartitionV2Error("Current fixture-skip boundary drift")
    if payload["tests_run"] != len(active_ids) - len(expected_unstarted):
        raise RegressionPartitionV2Error("Current active test count did not close")
    skip = contract["current_expected_skip_boundary"]
    if (
        len(payload["skipped"]) != skip["skip_event_count"]
        or payload["skip_events_sha256"] != skip["skip_events_sha256"]
    ):
        raise RegressionPartitionV2Error("Current documented skip events drift")
    payload["active_scheduled_test_count"] = len(active_ids)
    payload["historical_result_sha256"] = contract["historical_result_pin"]["sha256"]
    unsigned = dict(payload)
    unsigned.pop("canonical_self_hash")
    payload["canonical_self_hash"] = parent.sha256_bytes(
        parent.canonical_json_bytes(unsigned)
    )
    if output is not None:
        parent.write_new_result(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["was_successful"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        status = run_current(args.contract, args.output)
    except (RegressionPartitionV2Error, parent.RegressionPartitionError, OSError) as exc:
        print(f"Post-cleanup regression failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
