from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_regression_partition_v6 as partition


class RegressionPartitionV6Contracts(unittest.TestCase):
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
        self.assertEqual(len(historical), 38)
        self.assertEqual(len(active), boundary["active_scheduled_test_count"])

    def test_only_four_passed_predecessor_discovery_assertions_are_partitioned(self) -> None:
        evidence = partition.validate_predecessor_outcomes(self.contract)
        self.assertEqual(evidence["v2_outcome"], "success_in_v2_result")
        self.assertEqual(evidence["v3_outcome"], "success_in_v3_result")
        self.assertEqual(evidence["v4_outcome"], "success_in_v4_result")
        self.assertEqual(evidence["v5_outcome"], "success_in_v5_result")
        self.assertEqual(
            evidence["v5_test_id"],
            self.contract["superseded_discovery_test_ids"][3],
        )

    def test_historical_v9_1_result_remains_34_of_34(self) -> None:
        previous_contract = partition.previous.load_contract(
            ROOT / self.contract["previous_contract_pin"]["path"]
        )
        result = partition._load_json_pin(previous_contract["historical_result_pin"])
        self.assertEqual(result["tests_run"], 34)
        self.assertEqual(result["success_count"], 34)
        self.assertFalse(result["failure_ids"])
        self.assertFalse(result["error_ids"])
        self.assertFalse(result["skipped"])
        self.assertTrue(result["was_successful"])


if __name__ == "__main__":
    unittest.main()
