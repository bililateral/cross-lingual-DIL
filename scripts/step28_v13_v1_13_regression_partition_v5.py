#!/usr/bin/env python3
"""Run the current regression after freezing the V9.4.1 training-v3 core."""

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

import step28_v13_v1_13_regression_partition_v1 as base
import step28_v13_v1_13_regression_partition_v4 as previous


DEFAULT_CONTRACT = ROOT / "schema" / "step28_v13_v1_13_regression_partition_v5.json"
RESULT_VERSION = "2026-08-31-step28-v13-v1-13-regression-result-v5"
RESULT_MODE = "current_active_post_model_training_v3_implementation"


class RegressionPartitionV5Error(RuntimeError):
    """Raised when the training-v3 successor regression cannot be reproduced."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise RegressionPartitionV5Error(f"{context} schema drift")


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    try:
        contract = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegressionPartitionV5Error("Cannot load V5 regression contract") from exc
    if not isinstance(contract, dict):
        raise RegressionPartitionV5Error("V5 regression contract must be an object")
    _exact_keys(
        contract,
        {
            "version",
            "status",
            "canonical_self_hash",
            "base_contract_pin",
            "base_runner_pin",
            "previous_contract_pin",
            "previous_runner_pin",
            "previous_tests_pin",
            "historical_result_pin",
            "previous_result_pin",
            "partition_runner_pin",
            "partition_tests_pin",
            "superseded_discovery_test_ids",
            "current_discovery_boundary",
            "current_expected_skip_boundary",
            "reporting_rule",
            "training_authorized",
            "audit_truth_authorized",
        },
        "V5 regression contract",
    )
    if contract["version"] != "2026-08-31-step28-v13-v1-13-regression-partition-v5":
        raise RegressionPartitionV5Error("V5 regression version drift")
    if contract["status"] != "FROZEN_POST_MODEL_TRAINING_V3_IMPLEMENTATION_REGRESSION":
        raise RegressionPartitionV5Error("V5 regression status drift")
    if contract["training_authorized"] is not False or contract["audit_truth_authorized"] is not False:
        raise RegressionPartitionV5Error("V5 regression cannot authorize training or audit truth")
    unsigned = dict(contract)
    observed = unsigned.pop("canonical_self_hash")
    if base.sha256_bytes(base.canonical_json_bytes(unsigned)) != observed:
        raise RegressionPartitionV5Error("V5 regression self hash drift")
    pin_keys = (
        "base_contract_pin",
        "base_runner_pin",
        "previous_contract_pin",
        "previous_runner_pin",
        "previous_tests_pin",
        "historical_result_pin",
        "previous_result_pin",
        "partition_runner_pin",
        "partition_tests_pin",
    )
    for key in pin_keys:
        _exact_keys(contract[key], {"path", "size_bytes", "sha256"}, key)
    superseded = contract["superseded_discovery_test_ids"]
    if not isinstance(superseded, list) or len(superseded) != 3 or len(set(superseded)) != 3:
        raise RegressionPartitionV5Error("Exactly three predecessor discovery tests must be partitioned")
    _exact_keys(
        contract["current_discovery_boundary"],
        {
            "discovered_test_count",
            "discovered_test_ids_sha256",
            "v9_1_historical_test_count",
            "v9_1_historical_test_ids_sha256",
            "superseded_discovery_test_count",
            "superseded_discovery_test_ids_sha256",
            "historical_partition_test_count",
            "historical_partition_test_ids_sha256",
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
    return contract


def validate_pins(contract: Mapping[str, Any]) -> None:
    for key in (
        "base_contract_pin",
        "base_runner_pin",
        "previous_contract_pin",
        "previous_runner_pin",
        "previous_tests_pin",
        "historical_result_pin",
        "previous_result_pin",
        "partition_runner_pin",
        "partition_tests_pin",
    ):
        base.validate_file_pin(ROOT, contract[key])
    base_contract = base.load_contract(ROOT / contract["base_contract_pin"]["path"])
    base.validate_file_pin(ROOT, base_contract["partition_runner_pin"])
    previous_contract = previous.load_contract(
        ROOT / contract["previous_contract_pin"]["path"]
    )
    previous.validate_pins(previous_contract)


def _load_json_pin(pin: Mapping[str, Any]) -> dict[str, Any]:
    base.validate_file_pin(ROOT, pin)
    value = json.loads((ROOT / str(pin["path"])).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RegressionPartitionV5Error("Pinned result must be an object")
    return value


def validate_predecessor_outcomes(contract: Mapping[str, Any]) -> dict[str, Any]:
    previous_contract = previous.load_contract(
        ROOT / contract["previous_contract_pin"]["path"]
    )
    inherited = previous.validate_predecessor_outcomes(previous_contract)
    ids = contract["superseded_discovery_test_ids"]
    if ids[:2] != previous_contract["superseded_discovery_test_ids"]:
        raise RegressionPartitionV5Error("Inherited V2/V3 discovery IDs drift")
    result = _load_json_pin(contract["previous_result_pin"])
    v4_id = ids[2]
    disqualifying = {
        *result.get("failure_ids", []),
        *result.get("error_ids", []),
        *result.get("expected_failure_ids", []),
        *result.get("unexpected_success_ids", []),
        *result.get("unstarted_test_ids", []),
        *(row.get("id") for row in result.get("skipped", [])),
    }
    source = (ROOT / contract["previous_tests_pin"]["path"]).read_text(encoding="utf-8")
    module_name, class_name, method_name = v4_id.split(".", 2)
    if (
        result.get("was_successful") is not True
        or result.get("failure_ids")
        or result.get("error_ids")
        or v4_id in disqualifying
        or module_name != Path(contract["previous_tests_pin"]["path"]).stem
        or source.count(f"class {class_name}(") != 1
        or source.count(f"def {method_name}(") != 1
    ):
        raise RegressionPartitionV5Error("V4 discovery predecessor did not pass")
    return {
        **inherited,
        "v4_test_id": v4_id,
        "v4_outcome": "success_in_v4_result",
        "v4_result_sha256": contract["previous_result_pin"]["sha256"],
    }


def load_current_suite(
    contract: Mapping[str, Any],
) -> tuple[Any, list[str], list[str], list[str], list[str]]:
    base_contract = base.load_contract(ROOT / contract["base_contract_pin"]["path"])
    inherited_suite, discovered_ids, v9_1_ids = base.load_current_suite(base_contract, ROOT)
    superseded_ids = sorted(contract["superseded_discovery_test_ids"])
    inherited_tests = list(base._flatten(inherited_suite))
    observed_superseded = sorted(
        test.id() for test in inherited_tests if test.id() in set(superseded_ids)
    )
    if observed_superseded != superseded_ids:
        raise RegressionPartitionV5Error("Superseded discovery tests are absent or duplicated")
    active_tests = [test for test in inherited_tests if test.id() not in set(superseded_ids)]
    active_ids = [test.id() for test in active_tests]
    historical_ids = sorted([*v9_1_ids, *superseded_ids])
    boundary = contract["current_discovery_boundary"]
    checks = (
        (len(discovered_ids), boundary["discovered_test_count"]),
        (base.ids_sha256(discovered_ids), boundary["discovered_test_ids_sha256"]),
        (len(v9_1_ids), boundary["v9_1_historical_test_count"]),
        (base.ids_sha256(v9_1_ids), boundary["v9_1_historical_test_ids_sha256"]),
        (len(superseded_ids), boundary["superseded_discovery_test_count"]),
        (base.ids_sha256(superseded_ids), boundary["superseded_discovery_test_ids_sha256"]),
        (len(historical_ids), boundary["historical_partition_test_count"]),
        (base.ids_sha256(historical_ids), boundary["historical_partition_test_ids_sha256"]),
        (len(active_ids), boundary["active_scheduled_test_count"]),
        (base.ids_sha256(active_ids), boundary["active_scheduled_test_ids_sha256"]),
    )
    if any(observed != expected for observed, expected in checks):
        raise RegressionPartitionV5Error("Training-v3 discovery boundary drift")
    return unittest.TestSuite(active_tests), discovered_ids, historical_ids, active_ids, v9_1_ids


def run_current(contract_path: Path, output: Path | None) -> int:
    contract = load_contract(contract_path)
    validate_pins(contract)
    predecessor_evidence = validate_predecessor_outcomes(contract)
    suite, discovered_ids, historical_ids, active_ids, _ = load_current_suite(contract)
    base_contract = base.load_contract(ROOT / contract["base_contract_pin"]["path"])
    expected_unstarted = base.load_expected_fixture_unstarted_ids(base_contract, ROOT)
    boundary = contract["current_discovery_boundary"]
    if (
        len(expected_unstarted) != boundary["fixture_unstarted_test_count"]
        or base.ids_sha256(expected_unstarted) != boundary["fixture_unstarted_test_ids_sha256"]
    ):
        raise RegressionPartitionV5Error("Fixture-unstarted boundary drift")
    base.RESULT_VERSION = RESULT_VERSION
    payload = base.run_suite(
        suite,
        mode="current_active",
        repo_root=ROOT,
        contract=contract,
        discovered_ids=discovered_ids,
        historical_ids=historical_ids,
    )
    payload["mode"] = RESULT_MODE
    if payload["unstarted_test_ids"] != expected_unstarted:
        raise RegressionPartitionV5Error("Current fixture-skip boundary drift")
    if payload["tests_run"] != len(active_ids) - len(expected_unstarted):
        raise RegressionPartitionV5Error("Current active test count did not close")
    skip = contract["current_expected_skip_boundary"]
    if len(payload["skipped"]) != skip["skip_event_count"] or payload["skip_events_sha256"] != skip["skip_events_sha256"]:
        raise RegressionPartitionV5Error("Current documented skip events drift")
    payload["active_scheduled_test_count"] = len(active_ids)
    payload["historical_result_sha256"] = contract["historical_result_pin"]["sha256"]
    payload["previous_result_sha256"] = contract["previous_result_pin"]["sha256"]
    payload["predecessor_outcome_evidence"] = predecessor_evidence
    unsigned = dict(payload)
    unsigned.pop("canonical_self_hash")
    payload["canonical_self_hash"] = base.sha256_bytes(base.canonical_json_bytes(unsigned))
    if output is not None:
        base.write_new_result(output, payload)
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
        RegressionPartitionV5Error,
        previous.RegressionPartitionV4Error,
        previous.previous.RegressionPartitionV3Error,
        base.RegressionPartitionError,
        OSError,
    ) as exc:
        print(f"Training-v3 successor regression failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
