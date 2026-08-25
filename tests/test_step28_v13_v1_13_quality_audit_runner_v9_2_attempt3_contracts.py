#!/usr/bin/env python3
"""Regression contracts for the two V9.2 attempt-3 wiring repairs."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_audit_runner_v9_2_attempt3 as attempt3


class _Capability:
    def __init__(self, id_key_hex: str) -> None:
        self.id_key_hex = id_key_hex

    def private_key_hex(self, name: str) -> str:
        if name != "id_key_hex":
            raise AssertionError(name)
        return self.id_key_hex


class QualityAuditRunnerV92Attempt3Contracts(unittest.TestCase):
    ID_KEY_HEX = "31" * 32

    @staticmethod
    def _policy() -> dict[str, object]:
        return {
            "design_scale": {
                "world_counts": dict(attempt3.EXPECTED_WORLD_COUNTS),
            }
        }

    @classmethod
    def _reversed_split_assignment(cls) -> dict[str, dict[str, object]]:
        total = sum(attempt3.EXPECTED_WORLD_COUNTS.values())
        uid_by_ordinal = [
            attempt3.structure.base_uid(
                key_hex=cls.ID_KEY_HEX,
                entity_kind="world",
                parent_uid_or_mode=attempt3.WORLD_UID_PARENT_MODE,
                ordinal=ordinal,
            )
            for ordinal in range(total)
        ]
        assigned = list(reversed(uid_by_ordinal))
        loaded: dict[str, dict[str, object]] = {}
        start = 0
        for split in attempt3.base.SPLITS:
            count = attempt3.EXPECTED_WORLD_COUNTS[split]
            split_uids = assigned[start : start + count]
            loaded[split] = {
                "worlds": tuple(
                    {"world_uid": uid, "split_ordinal": ordinal}
                    for ordinal, uid in enumerate(split_uids)
                )
            }
            start += count
        return loaded

    def test_missing_pair_count_is_supplied_as_frozen_378(self) -> None:
        sentinel = ((("world", "pair"),), ("world",), {"world": {"seller"}})
        with mock.patch.object(
            attempt3,
            "_ORIGINAL_ENDPOINT_VALIDATOR",
            return_value=sentinel,
        ) as validator:
            observed = attempt3._validate_endpoints_with_frozen_pair_count(
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
            attempt3.QualityAuditAttempt3Error,
            "pair count drift",
        ):
            attempt3._validate_endpoints_with_frozen_pair_count(
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
            for split in attempt3.base.SPLITS
        }
        with attempt3._installed_direct_repairs():
            with mock.patch.object(
                attempt3,
                "_ORIGINAL_ENDPOINT_VALIDATOR",
                side_effect=validator,
            ):
                with self.assertRaises(ReachedEndpointValidator):
                    attempt3.base._validate_public_closure(
                        root_manifest={}, manifests={}, loaded=loaded
                    )

    def test_reconstruction_closes_shuffled_1004_world_bijection(self) -> None:
        loaded = self._reversed_split_assignment()
        observed = attempt3._reconstruct_world_ordinals(
            policy=self._policy(),
            loaded=loaded,
            id_key_hex=self.ID_KEY_HEX,
        )
        self.assertEqual(len(observed), 1004)
        mismatches = 0
        offset = 0
        for split in attempt3.base.SPLITS:
            for row in loaded[split]["worlds"]:
                uid = row["world_uid"]
                legacy_guess = offset + row["split_ordinal"]
                mismatches += observed[uid] != legacy_guess
            offset += attempt3.EXPECTED_WORLD_COUNTS[split]
        self.assertEqual(mismatches, 1004)

    def test_reconstruction_rejects_world_universe_drift(self) -> None:
        loaded = self._reversed_split_assignment()
        train = list(loaded["train"]["worlds"])
        train[0] = dict(train[1], split_ordinal=0)
        loaded["train"]["worlds"] = tuple(train)
        with self.assertRaisesRegex(
            attempt3.QualityAuditAttempt3Error,
            "world UID universe drift",
        ):
            attempt3._reconstruct_world_ordinals(
                policy=self._policy(),
                loaded=loaded,
                id_key_hex=self.ID_KEY_HEX,
            )

    def test_freeze_wrapper_uses_true_ordinals_and_clears_authority(self) -> None:
        loaded = self._reversed_split_assignment()
        sentinel = object()

        def freeze(**kwargs: object) -> object:
            rows = loaded["train"]["worlds"]
            observed = attempt3._global_ordinals_from_frozen_identity(
                self._policy(), "train", rows
            )
            first_uid = rows[0]["world_uid"]
            self.assertEqual(observed[first_uid], 1003)
            return sentinel

        with mock.patch.object(
            attempt3,
            "_ORIGINAL_FREEZE_TRAIN_DEVELOPMENT",
            side_effect=freeze,
        ):
            result = (
                attempt3._freeze_train_development_with_reconstructed_world_ordinals(
                    loaded=loaded,
                    policy=self._policy(),
                    run_capability=_Capability(self.ID_KEY_HEX),
                )
            )
        self.assertIs(result, sentinel)
        self.assertIsNone(attempt3._ACTIVE_WORLD_ORDINAL_BY_UID)

    def test_runtime_bindings_restore_after_keyboard_interrupt(self) -> None:
        base = attempt3.base
        preparer = attempt3.preparer
        truth = attempt3.truth_capability
        with self.assertRaises(KeyboardInterrupt):
            with attempt3._installed_direct_repairs():
                self.assertIs(
                    preparer._validate_endpoints,
                    attempt3._validate_endpoints_with_frozen_pair_count,
                )
                self.assertIs(
                    base._global_ordinals,
                    attempt3._global_ordinals_from_frozen_identity,
                )
                self.assertIs(
                    base._freeze_train_development,
                    attempt3._freeze_train_development_with_reconstructed_world_ordinals,
                )
                self.assertEqual(base.AUTHORIZATION_PATH, attempt3.AUTHORIZATION_PATH)
                self.assertEqual(
                    truth.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH,
                    attempt3.CONSUMED_AUTHORIZATION_PATH,
                )
                raise KeyboardInterrupt
        self.assertIs(preparer._validate_endpoints, attempt3._ORIGINAL_ENDPOINT_VALIDATOR)
        self.assertIs(base._global_ordinals, attempt3._ORIGINAL_GLOBAL_ORDINALS)
        self.assertIs(
            base._freeze_train_development,
            attempt3._ORIGINAL_FREEZE_TRAIN_DEVELOPMENT,
        )
        self.assertEqual(
            truth.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH,
            attempt3._ORIGINAL_TRUTH_AUTHORIZATION_PATH,
        )

    def test_attempt3_uses_fresh_receipt_paths(self) -> None:
        self.assertNotEqual(
            attempt3.AUTHORIZATION_PATH,
            attempt3._ORIGINAL_BASE_AUTHORIZATION_PATH,
        )
        self.assertNotEqual(
            attempt3.CONSUMED_AUTHORIZATION_PATH,
            attempt3._ORIGINAL_TRUTH_AUTHORIZATION_PATH,
        )
        self.assertTrue(
            attempt3.AUTHORIZATION_PATH.name.endswith(
                "v9_2_quality_run_attempt3_authorization.json"
            )
        )
        self.assertNotIn("attempt2", attempt3.AUTHORIZATION_PATH.name)


if __name__ == "__main__":
    unittest.main()
