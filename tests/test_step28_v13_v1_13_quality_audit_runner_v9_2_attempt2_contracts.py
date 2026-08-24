#!/usr/bin/env python3
"""Regression contracts for the V9.2 attempt-2 missing-argument repair."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_audit_runner_v9_2_attempt2 as attempt2


class QualityAuditRunnerV92Attempt2Contracts(unittest.TestCase):
    def test_missing_pair_count_is_supplied_as_frozen_378(self) -> None:
        sentinel = ((('world', 'pair'),), ('world',), {'world': {'seller'}})
        with mock.patch.object(
            attempt2,
            "_ORIGINAL_ENDPOINT_VALIDATOR",
            return_value=sentinel,
        ) as validator:
            observed = attempt2._validate_endpoints_with_frozen_pair_count(
                (), ordered_world_uids=("world",)
            )
        self.assertEqual(observed, sentinel)
        validator.assert_called_once_with(
            (),
            ordered_world_uids=("world",),
            expected_pairs_per_world=378,
        )

    def test_pair_count_cannot_be_widened(self) -> None:
        with self.assertRaisesRegex(
            attempt2.QualityAuditAttempt2Error,
            "pair count drift",
        ):
            attempt2._validate_endpoints_with_frozen_pair_count(
                (), ordered_world_uids=("world",), expected_pairs_per_world=377
            )

    def test_real_public_closure_call_reaches_validator_with_378(self) -> None:
        class ReachedEndpointValidator(RuntimeError):
            pass

        def validator(
            endpoints: object,
            *,
            ordered_world_uids: object,
            expected_pairs_per_world: int,
        ) -> None:
            self.assertEqual(endpoints, ())
            self.assertEqual(ordered_world_uids, ("world",))
            self.assertEqual(expected_pairs_per_world, 378)
            raise ReachedEndpointValidator

        loaded = {
            split: {
                "worlds": ({"world_uid": "world", "split_ordinal": 0},),
                "endpoints": (),
            }
            for split in attempt2.base.SPLITS
        }
        with attempt2._installed_direct_repair():
            with mock.patch.object(
                attempt2,
                "_ORIGINAL_ENDPOINT_VALIDATOR",
                side_effect=validator,
            ):
                with self.assertRaises(ReachedEndpointValidator):
                    attempt2.base._validate_public_closure(
                        root_manifest={}, manifests={}, loaded=loaded
                    )

    def test_repair_changes_only_runtime_bindings_and_restores_them(self) -> None:
        base = attempt2.base
        preparer = attempt2.preparer
        truth = attempt2.truth_capability
        with attempt2._installed_direct_repair():
            self.assertIs(
                preparer._validate_endpoints,
                attempt2._validate_endpoints_with_frozen_pair_count,
            )
            self.assertEqual(base.AUTHORIZATION_PATH, attempt2.AUTHORIZATION_PATH)
            self.assertIs(
                base.load_run_authorization,
                attempt2._load_attempt2_authorization,
            )
            self.assertEqual(base.VERSION, attempt2.VERSION)
            self.assertEqual(
                truth.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH,
                attempt2.CONSUMED_AUTHORIZATION_PATH,
            )
        self.assertIs(
            preparer._validate_endpoints,
            attempt2._ORIGINAL_ENDPOINT_VALIDATOR,
        )
        self.assertEqual(
            base.AUTHORIZATION_PATH,
            attempt2._ORIGINAL_BASE_AUTHORIZATION_PATH,
        )
        self.assertIs(
            base.load_run_authorization,
            attempt2._ORIGINAL_LOAD_AUTHORIZATION,
        )
        self.assertEqual(base.VERSION, attempt2._ORIGINAL_BASE_VERSION)
        self.assertEqual(
            truth.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH,
            attempt2._ORIGINAL_TRUTH_AUTHORIZATION_PATH,
        )

    def test_attempt2_uses_fresh_receipt_paths(self) -> None:
        self.assertNotEqual(
            attempt2.AUTHORIZATION_PATH,
            attempt2._ORIGINAL_BASE_AUTHORIZATION_PATH,
        )
        self.assertNotEqual(
            attempt2.CONSUMED_AUTHORIZATION_PATH,
            attempt2._ORIGINAL_TRUTH_AUTHORIZATION_PATH,
        )
        self.assertTrue(
            attempt2.AUTHORIZATION_PATH.name.endswith(
                "v9_2_quality_run_attempt2_authorization.json"
            )
        )


if __name__ == "__main__":
    unittest.main()
