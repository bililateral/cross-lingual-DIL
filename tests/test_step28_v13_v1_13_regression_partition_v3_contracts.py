from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_regression_partition_v3 as partition


class RegressionPartitionV3Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = partition.load_contract()

    def test_contract_is_self_hashed_and_authorizes_no_scientific_action(self) -> None:
        self.assertFalse(self.contract["training_authorized"])
        self.assertFalse(self.contract["audit_truth_authorized"])
        partition.validate_pins(self.contract)

    def test_current_discovery_boundary_is_exact(self) -> None:
        _, discovered, historical, active = partition.load_current_suite(self.contract)
        boundary = self.contract["current_discovery_boundary"]
        self.assertEqual(len(discovered), boundary["discovered_test_count"])
        self.assertEqual(len(historical), 35)
        self.assertEqual(len(active), boundary["active_scheduled_test_count"])

    def test_invalid_fixture_is_absent_and_full_replay_is_present(self) -> None:
        partition.validate_compatibility_cleanup()
        cleanup = self.contract["compatibility_cleanup_boundary"]
        self.assertEqual(cleanup["removed_test_case_count"], 17)
        self.assertEqual(cleanup["replacement_test_case_count"], 13)

    def test_historical_result_and_fixture_skip_boundary_are_unchanged(self) -> None:
        result = json.loads(
            (ROOT / self.contract["historical_result_pin"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(result["tests_run"], 34)
        self.assertTrue(result["was_successful"])
        predecessor = json.loads(
            (ROOT / self.contract["predecessor_result_pin"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(predecessor["was_successful"])
        self.assertFalse(predecessor["failure_ids"])
        self.assertFalse(predecessor["error_ids"])
        evidence = partition.validate_predecessor_outcome(self.contract)
        self.assertTrue(evidence["scheduled_in_v2"])
        self.assertTrue(evidence["started_in_v2"])
        self.assertEqual(evidence["outcome_in_v2"], "success")
        v1_contract = partition._v1_contract(self.contract)
        ids = partition.previous.parent.load_expected_fixture_unstarted_ids(
            v1_contract, ROOT
        )
        boundary = self.contract["current_discovery_boundary"]
        self.assertEqual(len(ids), boundary["fixture_unstarted_test_count"])
        self.assertEqual(
            partition.previous.parent.ids_sha256(ids),
            boundary["fixture_unstarted_test_ids_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
