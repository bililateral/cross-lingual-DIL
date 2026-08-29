from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_regression_partition_v1 as partition


class RegressionPartitionV1Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = partition.load_contract()

    def test_contract_is_self_hashed_and_authorizes_no_scientific_action(self) -> None:
        self.assertFalse(self.contract["training_authorized"])
        self.assertFalse(self.contract["audit_truth_authorized"])
        partition.validate_file_pin(ROOT, self.contract["current_policy_pin"])
        partition.validate_file_pin(ROOT, self.contract["partition_runner_pin"])
        partition.validate_file_pin(ROOT, self.contract["partition_tests_pin"])

    def test_historical_commit_tree_and_frozen_blobs_exist_exactly(self) -> None:
        snapshot = self.contract["historical_snapshot"]
        commit = partition.git_value(ROOT, "rev-parse", snapshot["commit"])
        tree = partition.git_value(ROOT, "rev-parse", f"{snapshot['commit']}^{{tree}}")
        self.assertEqual(commit, snapshot["commit"])
        self.assertEqual(tree, snapshot["tree"])
        for path_key in ("policy_path", "shared_text_source_path"):
            completed = subprocess.run(
                ["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}:{snapshot[path_key]}"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0)

    def test_partition_is_exact_and_not_implemented_as_skips(self) -> None:
        suite, discovered_ids, historical_ids = partition.load_current_suite(
            self.contract, ROOT
        )
        active_ids = [test.id() for test in partition._flatten(suite)]
        self.assertEqual(len(historical_ids), 34)
        self.assertEqual(len(active_ids) + len(historical_ids), len(discovered_ids))
        self.assertFalse(set(active_ids).intersection(historical_ids))
        source = Path(partition.__file__).read_text(encoding="utf-8")
        self.assertNotIn("SkipTest", source)
        self.assertNotIn("addSkip", source)

    def test_selector_expansion_is_fail_closed(self) -> None:
        changed = json.loads(json.dumps(self.contract))
        changed["historical_expanded_test_count"] = 33
        with self.assertRaisesRegex(
            partition.RegressionPartitionError,
            "Historical expanded test count drift",
        ):
            partition.load_historical_suite(changed, ROOT)

    def test_current_fixture_skip_is_explicit_and_exact(self) -> None:
        ids = partition.load_expected_fixture_unstarted_ids(self.contract, ROOT)
        boundary = self.contract["current_expected_skip_boundary"]
        self.assertEqual(len(ids), 20)
        self.assertEqual(
            partition.ids_sha256(ids),
            boundary["fixture_unstarted_test_ids_sha256"],
        )
        self.assertEqual(boundary["skip_event_count"], 9)

    def test_current_partition_keeps_discovery_import_failures_active(self) -> None:
        fake_failure = unittest.defaultTestLoader.loadTestsFromName(
            "tests.module_that_does_not_exist"
        )
        with patch.object(
            unittest.defaultTestLoader,
            "discover",
            return_value=fake_failure,
        ):
            with self.assertRaisesRegex(
                partition.RegressionPartitionError,
                "Historical partition is absent",
            ):
                partition.load_current_suite(self.contract, ROOT)

    def test_worktree_cleanup_guard_rejects_repository_path(self) -> None:
        with self.assertRaisesRegex(
            partition.RegressionPartitionError,
            "unsafe historical worktree cleanup",
        ):
            partition._remove_worktree(ROOT)


if __name__ == "__main__":
    unittest.main()
