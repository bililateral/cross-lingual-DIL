from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_regression_partition_v7 as partition


class RegressionPartitionV7Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = partition.load_contract()

    def test_contract_is_self_hashed_pinned_and_authorizes_nothing(self) -> None:
        self.assertFalse(self.contract["training_authorized"])
        self.assertFalse(self.contract["audit_truth_authorized"])
        partition.validate_pins(self.contract)

    def test_current_discovery_boundary_is_exact(self) -> None:
        _, discovered, historical, active, v9_1 = partition.load_current_suite(
            self.contract
        )
        boundary = self.contract["current_discovery_boundary"]
        self.assertEqual(len(discovered), boundary["discovered_test_count"])
        self.assertEqual(len(v9_1), 34)
        self.assertEqual(len(historical), 39)
        self.assertEqual(len(active), boundary["active_scheduled_test_count"])

    def test_only_five_passed_predecessor_discovery_assertions_are_partitioned(self) -> None:
        evidence = partition.validate_predecessor_outcomes(self.contract)
        self.assertEqual(evidence["v2_outcome"], "success_in_v2_result")
        self.assertEqual(evidence["v3_outcome"], "success_in_v3_result")
        self.assertEqual(evidence["v4_outcome"], "success_in_v4_result")
        self.assertEqual(evidence["v5_outcome"], "success_in_v5_result")
        self.assertEqual(evidence["v6_outcome"], "success_in_v6_result")
        self.assertEqual(
            evidence["v6_test_id"],
            self.contract["superseded_discovery_test_ids"][4],
        )

    def test_historical_v9_1_result_remains_34_of_34(self) -> None:
        pin = partition.historical_result_pin(self.contract)
        result = partition._load_json_pin(pin)
        self.assertEqual(result["tests_run"], 34)
        self.assertEqual(result["success_count"], 34)
        self.assertFalse(result["failure_ids"])
        self.assertFalse(result["error_ids"])
        self.assertFalse(result["skipped"])
        self.assertTrue(result["was_successful"])

    def test_result_aggregation_uses_inherited_historical_pin(self) -> None:
        base_contract = partition.base.load_contract(
            ROOT / self.contract["base_contract_pin"]["path"]
        )
        unstarted = partition.base.load_expected_fixture_unstarted_ids(
            base_contract, ROOT
        )
        skip = self.contract["current_expected_skip_boundary"]
        mocked_payload = {
            "canonical_self_hash": "0" * 64,
            "unstarted_test_ids": unstarted,
            "tests_run": 0,
            "skipped": [{} for _ in range(skip["skip_event_count"])],
            "skip_events_sha256": skip["skip_events_sha256"],
            "was_successful": True,
        }
        active_ids = [f"fixture-boundary-{index}" for index in range(len(unstarted))]
        with (
            mock.patch.object(
                partition,
                "load_current_suite",
                return_value=(unittest.TestSuite(), [], [], active_ids, []),
            ),
            mock.patch.object(
                partition.base, "run_suite", return_value=mocked_payload
            ),
            mock.patch.object(partition.base, "write_new_result") as write_result,
            mock.patch("builtins.print"),
        ):
            status = partition.run_current(
                partition.DEFAULT_CONTRACT, ROOT / "unused-v7-result.json"
            )
        self.assertEqual(status, 0)
        written = write_result.call_args.args[1]
        self.assertEqual(
            written["historical_result_sha256"],
            partition.historical_result_pin(self.contract)["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
