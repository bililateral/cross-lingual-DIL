from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_regression_partition_v1 as parent
import step28_v13_v1_13_regression_partition_v2 as partition


class RegressionPartitionV2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = partition.load_contract()

    def test_contract_is_self_hashed_and_authorizes_no_scientific_action(self) -> None:
        self.assertFalse(self.contract["training_authorized"])
        self.assertFalse(self.contract["audit_truth_authorized"])
        partition.validate_pins(self.contract)
        with mock.patch.object(
            partition.parent,
            "run_suite",
            return_value={"mode": partition.PARENT_SCHEDULING_MODE},
        ) as run_suite:
            payload = partition.run_parent_current_suite(
                unittest.TestSuite(), self.contract, [], []
            )
        self.assertEqual(payload["mode"], partition.RESULT_MODE)
        self.assertEqual(
            run_suite.call_args.kwargs["mode"], partition.PARENT_SCHEDULING_MODE
        )

    def test_current_discovery_boundary_is_exact(self) -> None:
        _, discovered, historical, active = partition.load_current_suite(self.contract)
        boundary = self.contract["current_discovery_boundary"]
        self.assertEqual(len(discovered), boundary["discovered_test_count"])
        self.assertEqual(len(historical), 34)
        self.assertEqual(len(active), boundary["active_scheduled_test_count"])

    def test_closed_v1_12_code_is_absent(self) -> None:
        partition.validate_cleanup_absence()
        removed = self.contract["removed_v1_12_boundary"]
        self.assertEqual(removed["removed_file_count"], 27)
        self.assertEqual(removed["removed_test_case_count"], 107)

    def test_live_preflight_matrices_remain_present(self) -> None:
        matrices = list(
            (
                ROOT
                / "reports/step28_synthetic_chinese_dataset/design_preflights"
            ).glob("**/*.checkpoint.shortcut_inputs.npz")
        )
        self.assertEqual(len(matrices), 21)
        self.assertEqual(sum(path.stat().st_size for path in matrices), 73979098)

    def test_historical_v9_1_result_remains_closed(self) -> None:
        pin = self.contract["historical_result_pin"]
        result = json.loads((ROOT / pin["path"]).read_text(encoding="utf-8"))
        self.assertEqual(result["tests_run"], 34)
        self.assertEqual(result["success_count"], 34)
        self.assertFalse(result["failure_ids"])
        self.assertFalse(result["error_ids"])
        self.assertFalse(result["skipped"])
        self.assertTrue(result["was_successful"])

    def test_fixture_unstarted_boundary_is_inherited_without_drift(self) -> None:
        parent_contract = parent.load_contract(
            ROOT / self.contract["parent_contract_pin"]["path"]
        )
        ids = parent.load_expected_fixture_unstarted_ids(parent_contract, ROOT)
        boundary = self.contract["current_discovery_boundary"]
        self.assertEqual(len(ids), boundary["fixture_unstarted_test_count"])
        self.assertEqual(
            parent.ids_sha256(ids), boundary["fixture_unstarted_test_ids_sha256"]
        )

    def test_skip_boundary_removes_only_the_deleted_v1_12_event(self) -> None:
        old = json.loads(
            (
                ROOT
                / "reports/step28_regression_partition/v1_20260830/"
                "current_active_regression.json"
            ).read_text(encoding="utf-8")
        )
        retained = [row for row in old["skipped"] if "v1_12" not in row["id"]]
        boundary = self.contract["current_expected_skip_boundary"]
        self.assertEqual(len(retained), boundary["skip_event_count"])
        self.assertEqual(
            parent.sha256_bytes(parent.canonical_json_bytes(retained)),
            boundary["skip_events_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
