from __future__ import annotations

import sys
from pathlib import Path
from types import MappingProxyType
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_balanced_world_schedule_v9_4 as schedule_v94


class BalancedWorldScheduleV94Contracts(unittest.TestCase):
    def test_noise_latin_construction_has_exact_17_18_margins(self) -> None:
        rng = np.random.Generator(np.random.PCG64(281320260825))
        noise = schedule_v94._balanced_noise_assignments(rng)
        self.assertEqual(noise.shape, (500, 28))
        counts = np.zeros((28, 28), dtype=np.int16)
        for row in noise:
            self.assertEqual(set(row.tolist()), set(range(28)))
            counts[np.arange(28), row] += 1
        for values in (*counts, *counts.T):
            self.assertEqual(
                dict(sorted(__import__("collections").Counter(values).items())),
                {17: 4, 18: 24},
            )

    def test_contract_keeps_old_public_trajectory_but_reads_no_r2_plan(self) -> None:
        contract = schedule_v94.contract_payload()
        self.assertEqual(
            contract["public_design_seeds"],
            {"train": 281320260825, "development": 281320260827},
        )
        self.assertFalse(contract["direct_r2_plan_read"])
        self.assertTrue(contract["issued_schedule_required"])
        self.assertTrue(contract["latent_schedule_commitment"])
        self.assertTrue(contract["train_development_pair_audit_required"])
        source = Path(schedule_v94.__file__).read_text(encoding="utf-8")
        self.assertNotIn("registered_negative_bounded_preflight_r2", source)
        self.assertNotIn("git show", source)

    def test_group_shape_implies_exactly_20_positive_pairs(self) -> None:
        self.assertEqual(schedule_v94.POSITIVE_COUNT, 20)
        self.assertEqual(schedule_v94.PAIR_COUNT, 378)
        self.assertEqual(sum(schedule_v94.GROUP_SIZES), 28)

    def test_unissued_self_described_schedule_is_rejected(self) -> None:
        forged = schedule_v94.SplitSchedule(
            split="train",
            public_worlds=(),
            controller_groups_by_world=(),
            commitment=MappingProxyType({}),
            _issuer=object(),
        )
        with self.assertRaisesRegex(
            schedule_v94.BalancedWorldScheduleV94Error,
            "capability drift",
        ):
            schedule_v94.verify_split_schedule(forged)

    def test_latent_hash_is_invariant_to_controller_group_order(self) -> None:
        first = np.arange(28, dtype=np.int16).reshape(1, 28)
        second = first.copy()
        second[0, 0:3], second[0, 3:6] = (
            first[0, 3:6],
            first[0, 0:3],
        )
        noise = np.arange(28, dtype=np.int16).reshape(1, 28)
        self.assertEqual(
            schedule_v94._latent_schedule_sha256(first, noise),
            schedule_v94._latent_schedule_sha256(second, noise),
        )

    def test_pair_audit_rejects_identical_latent_schedules_before_scoring(self) -> None:
        commitment = {"latent_schedule_sha256": "0" * 64}
        train = SimpleNamespace(split="train", commitment=commitment)
        development = SimpleNamespace(
            split="development",
            commitment=commitment,
        )
        arrays = (
            np.zeros((500, 28), dtype=np.int16),
            np.zeros((500, 28), dtype=np.int16),
        )
        with patch.object(
            schedule_v94,
            "_schedule_arrays",
            return_value=arrays,
        ):
            with self.assertRaisesRegex(
                schedule_v94.BalancedWorldScheduleV94Error,
                "latent schedule identity drift",
            ):
                schedule_v94.validate_train_development_pair(
                    train,
                    development,
                )


if __name__ == "__main__":
    unittest.main()
