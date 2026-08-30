#!/usr/bin/env python3
"""Run the current regression after invalid compatibility-v1 cleanup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_regression_partition_v2 as previous


DEFAULT_CONTRACT = ROOT / "schema" / "step28_v13_v1_13_regression_partition_v3.json"
RESULT_VERSION = "2026-08-30-step28-v13-v1-13-regression-result-v3"
RESULT_MODE = "current_active_post_compatibility_v1_cleanup"


class RegressionPartitionV3Error(RuntimeError):
    """Raised when the compatibility-cleanup regression cannot be reproduced."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise RegressionPartitionV3Error(f"{context} schema drift")


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    try:
        contract = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegressionPartitionV3Error("Cannot load V3 regression contract") from exc
    if not isinstance(contract, dict):
        raise RegressionPartitionV3Error("V3 regression contract must be an object")
    _exact_keys(
        contract,
        {
            "version",
            "status",
            "canonical_self_hash",
            "previous_contract_pin",
            "previous_runner_pin",
            "previous_tests_pin",
            "historical_result_pin",
            "predecessor_result_pin",
            "partition_runner_pin",
            "partition_tests_pin",
            "current_discovery_boundary",
            "current_expected_skip_boundary",
            "compatibility_cleanup_boundary",
            "reporting_rule",
            "training_authorized",
            "audit_truth_authorized",
        },
        "V3 regression contract",
    )
    if contract["version"] != "2026-08-30-step28-v13-v1-13-regression-partition-v3":
        raise RegressionPartitionV3Error("V3 regression version drift")
    if contract["status"] != "FROZEN_POST_COMPATIBILITY_V1_CLEANUP_REGRESSION":
        raise RegressionPartitionV3Error("V3 regression status drift")
    if contract["training_authorized"] is not False:
        raise RegressionPartitionV3Error("V3 regression cannot authorize training")
    if contract["audit_truth_authorized"] is not False:
        raise RegressionPartitionV3Error("V3 regression cannot open audit truth")
    unsigned = dict(contract)
    observed = unsigned.pop("canonical_self_hash")
    if previous.parent.sha256_bytes(previous.parent.canonical_json_bytes(unsigned)) != observed:
        raise RegressionPartitionV3Error("V3 regression self hash drift")
    for key in (
        "previous_contract_pin",
        "previous_runner_pin",
        "previous_tests_pin",
        "historical_result_pin",
        "predecessor_result_pin",
        "partition_runner_pin",
        "partition_tests_pin",
    ):
        pin = contract[key]
        if not isinstance(pin, dict):
            raise RegressionPartitionV3Error(f"{key} must be an object")
        _exact_keys(pin, {"path", "size_bytes", "sha256"}, key)
    _exact_keys(
        contract["current_discovery_boundary"],
        {
            "discovered_test_count",
            "discovered_test_ids_sha256",
            "historical_partition_test_count",
            "historical_partition_test_ids_sha256",
            "v9_1_historical_test_count",
            "v9_1_historical_test_ids_sha256",
            "predecessor_partition_test_count",
            "predecessor_partition_test_id",
            "active_scheduled_test_count",
            "active_scheduled_test_ids_sha256",
            "fixture_unstarted_test_count",
            "fixture_unstarted_test_ids_sha256",
        },
        "current discovery boundary",
    )
    _exact_keys(
        contract["current_expected_skip_boundary"],
        {"skip_event_count", "skip_events_sha256"},
        "current skip boundary",
    )
    _exact_keys(
        contract["compatibility_cleanup_boundary"],
        {
            "removed_script_count",
            "removed_test_file_count",
            "removed_test_case_count",
            "removed_payload_file_count",
            "replacement_test_case_count",
        },
        "compatibility cleanup boundary",
    )
    return contract


def validate_pins(contract: Mapping[str, Any]) -> None:
    for key in (
        "previous_contract_pin",
        "previous_runner_pin",
        "previous_tests_pin",
        "historical_result_pin",
        "predecessor_result_pin",
        "partition_runner_pin",
        "partition_tests_pin",
    ):
        previous.parent.validate_file_pin(ROOT, contract[key])
    old = previous.load_contract(ROOT / contract["previous_contract_pin"]["path"])
    previous.validate_pins(old)


def _v1_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    old = previous.load_contract(ROOT / contract["previous_contract_pin"]["path"])
    return previous.parent.load_contract(ROOT / old["parent_contract_pin"]["path"])


def validate_predecessor_outcome(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the single partitioned V2 meta-test started and passed."""
    boundary = contract["current_discovery_boundary"]
    predecessor_id = boundary["predecessor_partition_test_id"]
    previous_contract = previous.load_contract(
        ROOT / contract["previous_contract_pin"]["path"]
    )
    result = json.loads(
        (ROOT / contract["predecessor_result_pin"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    previous_boundary = previous_contract["current_discovery_boundary"]
    if (
        result.get("scheduled_test_count")
        != previous_boundary["active_scheduled_test_count"]
        or result.get("scheduled_test_ids_sha256")
        != previous_boundary["active_scheduled_test_ids_sha256"]
        or result.get("was_successful") is not True
        or result.get("failure_ids")
        or result.get("error_ids")
        or result.get("expected_failure_ids")
        or result.get("unexpected_success_ids")
    ):
        raise RegressionPartitionV3Error("V2 predecessor result is not successful")
    excluded_ids = {
        *result.get("unstarted_test_ids", []),
        *(row.get("id") for row in result.get("skipped", [])),
    }
    if predecessor_id in excluded_ids:
        raise RegressionPartitionV3Error("V2 predecessor did not start and pass")
    module_name, class_name, method_name = predecessor_id.split(".", 2)
    source = (ROOT / contract["previous_tests_pin"]["path"]).read_text(
        encoding="utf-8"
    )
    if (
        module_name != Path(contract["previous_tests_pin"]["path"]).stem
        or source.count(f"class {class_name}(") != 1
        or source.count(f"def {method_name}(") != 1
    ):
        raise RegressionPartitionV3Error(
            "V2 predecessor scheduled-membership source drift"
        )
    return {
        "test_id": predecessor_id,
        "scheduled_in_v2": True,
        "started_in_v2": True,
        "outcome_in_v2": "success",
        "v2_result_sha256": contract["predecessor_result_pin"]["sha256"],
    }


def load_current_suite(
    contract: Mapping[str, Any],
) -> tuple[Any, list[str], list[str], list[str]]:
    v1_contract = _v1_contract(contract)
    suite, discovered_ids, v9_1_historical_ids = previous.parent.load_current_suite(
        v1_contract, ROOT
    )
    boundary = contract["current_discovery_boundary"]
    predecessor_id = boundary["predecessor_partition_test_id"]
    inherited_active_tests = list(previous.parent._flatten(suite))
    predecessor_tests = [
        test for test in inherited_active_tests if test.id() == predecessor_id
    ]
    if len(predecessor_tests) != 1:
        raise RegressionPartitionV3Error(
            "V2 predecessor regression test is absent or duplicated"
        )
    active_tests = [
        test for test in inherited_active_tests if test.id() != predecessor_id
    ]
    suite = unittest.TestSuite(active_tests)
    active_ids = [test.id() for test in active_tests]
    historical_ids = sorted([*v9_1_historical_ids, predecessor_id])
    checks = (
        (len(discovered_ids), boundary["discovered_test_count"]),
        (
            previous.parent.ids_sha256(discovered_ids),
            boundary["discovered_test_ids_sha256"],
        ),
        (len(historical_ids), boundary["historical_partition_test_count"]),
        (
            previous.parent.ids_sha256(historical_ids),
            boundary["historical_partition_test_ids_sha256"],
        ),
        (len(v9_1_historical_ids), boundary["v9_1_historical_test_count"]),
        (
            previous.parent.ids_sha256(v9_1_historical_ids),
            boundary["v9_1_historical_test_ids_sha256"],
        ),
        (len(predecessor_tests), boundary["predecessor_partition_test_count"]),
        (len(active_ids), boundary["active_scheduled_test_count"]),
        (
            previous.parent.ids_sha256(active_ids),
            boundary["active_scheduled_test_ids_sha256"],
        ),
    )
    if any(observed != expected for observed, expected in checks):
        raise RegressionPartitionV3Error("Compatibility-cleanup discovery drift")
    return suite, discovered_ids, historical_ids, active_ids


def validate_compatibility_cleanup() -> None:
    removed = (
        "scripts/run_step28_v13_v1_13_v9_4_1_compatibility_linux_20260830.sh",
        "scripts/step28_v13_v1_13_v9_4_1_prepare_compatibility_fixture_v1.py",
        "scripts/step28_v13_v1_13_v9_4_1_replay_compatibility_fixture_linux_v1.py",
        "tests/test_step28_v13_v1_13_v9_4_1_compatibility_fixture_v1_contracts.py",
        "tests/test_step28_v13_v1_13_v9_4_1_linux_fixture_replay_v1_contracts.py",
        "reports/step28_model_experiment/v9_4_1_implementation_v1_20260830/compatibility_fixture",
    )
    if any((ROOT / relative).exists() for relative in removed):
        raise RegressionPartitionV3Error("Invalid compatibility-v1 artifact reappeared")
    required = (
        "schema/step28_v13_v1_13_v9_4_1_full_english_compatibility_policy_v2.json",
        "scripts/step28_v13_v1_13_v9_4_1_replay_full_english_compatibility_linux_v2.py",
        "scripts/run_step28_v13_v1_13_v9_4_1_full_english_compatibility_v2_linux_20260830.sh",
        "tests/test_step28_v13_v1_13_v9_4_1_full_english_compatibility_v2_contracts.py",
    )
    if any(not (ROOT / relative).is_file() for relative in required):
        raise RegressionPartitionV3Error("Compatibility-v2 replacement is incomplete")


def run_current(contract_path: Path, output: Path | None) -> int:
    contract = load_contract(contract_path)
    validate_pins(contract)
    validate_compatibility_cleanup()
    predecessor_outcome = validate_predecessor_outcome(contract)
    suite, discovered_ids, historical_ids, active_ids = load_current_suite(contract)
    v1_contract = _v1_contract(contract)
    expected_unstarted = previous.parent.load_expected_fixture_unstarted_ids(
        v1_contract, ROOT
    )
    boundary = contract["current_discovery_boundary"]
    if (
        len(expected_unstarted) != boundary["fixture_unstarted_test_count"]
        or previous.parent.ids_sha256(expected_unstarted)
        != boundary["fixture_unstarted_test_ids_sha256"]
    ):
        raise RegressionPartitionV3Error("Fixture-unstarted boundary drift")
    previous.parent.RESULT_VERSION = RESULT_VERSION
    payload = previous.parent.run_suite(
        suite,
        mode="current_active",
        repo_root=ROOT,
        contract=contract,
        discovered_ids=discovered_ids,
        historical_ids=historical_ids,
    )
    payload["mode"] = RESULT_MODE
    if payload["unstarted_test_ids"] != expected_unstarted:
        raise RegressionPartitionV3Error("Current fixture-skip boundary drift")
    if payload["tests_run"] != len(active_ids) - len(expected_unstarted):
        raise RegressionPartitionV3Error("Current active test count did not close")
    skip = contract["current_expected_skip_boundary"]
    if (
        len(payload["skipped"]) != skip["skip_event_count"]
        or payload["skip_events_sha256"] != skip["skip_events_sha256"]
    ):
        raise RegressionPartitionV3Error("Current documented skip events drift")
    payload["active_scheduled_test_count"] = len(active_ids)
    payload["historical_result_sha256"] = contract["historical_result_pin"]["sha256"]
    payload["predecessor_result_sha256"] = contract["predecessor_result_pin"][
        "sha256"
    ]
    payload["predecessor_outcome_evidence"] = predecessor_outcome
    unsigned = dict(payload)
    unsigned.pop("canonical_self_hash")
    payload["canonical_self_hash"] = previous.parent.sha256_bytes(
        previous.parent.canonical_json_bytes(unsigned)
    )
    if output is not None:
        previous.parent.write_new_result(output, payload)
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
    except (
        RegressionPartitionV3Error,
        previous.RegressionPartitionV2Error,
        previous.parent.RegressionPartitionError,
        OSError,
    ) as exc:
        print(f"Compatibility-cleanup regression failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
