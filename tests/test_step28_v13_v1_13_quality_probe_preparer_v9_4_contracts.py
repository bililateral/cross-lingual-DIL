from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_probe_preparer_v9_4 as preparer_v94


def fixtures(world_count: int = 2):
    signatures = [
        {
            "noise_slot": index,
            "item_count": 2 + index % 7,
            "title_present_mask": "1" * (2 + index % 7),
            "description_present_mask": "1" * (2 + index % 7),
            "joint_empty_mask": "0" * (2 + index % 7),
        }
        for index in range(28)
    ]
    worlds = [
        {
            "split": "train",
            "world_ordinal": world_index,
            "world_uid": f"world_{world_index:03d}",
            "seller_uids": [
                f"world_{world_index:03d}_seller_{seller_index:02d}"
                for seller_index in range(28)
            ],
            "noise_slot_by_seller_slot": list(range(28)),
        }
        for world_index in range(world_count)
    ]
    return worlds, signatures


class QualityProbePreparerV94Contracts(unittest.TestCase):
    def test_label_free_preparer_freezes_only_registered_view(self) -> None:
        worlds, signatures = fixtures()
        prepared = preparer_v94._prepare_split(
            worlds=worlds,
            noise_signatures=signatures,
            time_key_hex="01" * 32,
            expected_world_count=2,
            split_schedule_commitment_sha256="02" * 32,
            schedule_pair_audit_commitment_sha256="04" * 32,
            noise_signature_set_commitment_sha256="03" * 32,
        )
        preparer_v94.verify_prepared_split(prepared, expected_world_count=2)
        self.assertEqual(prepared.matrix.values.shape, (756, 14))
        self.assertFalse(prepared.matrix.values.flags.writeable)
        self.assertEqual(len(prepared.world_commitments), 2)
        with self.assertRaises(TypeError):
            prepared.commitment["row_count"] = 0

    def test_world_order_split_collision_and_private_fields_are_rejected(self) -> None:
        worlds, signatures = fixtures()
        reversed_worlds = list(reversed(worlds))
        with self.assertRaisesRegex(
            preparer_v94.QualityProbePreparerV94Error,
            "ordinal drift",
        ):
            preparer_v94._prepare_split(
                worlds=reversed_worlds,
                noise_signatures=signatures,
                time_key_hex="01" * 32,
                expected_world_count=2,
                split_schedule_commitment_sha256="02" * 32,
                schedule_pair_audit_commitment_sha256="04" * 32,
                noise_signature_set_commitment_sha256="03" * 32,
            )
        collision = [dict(row) for row in worlds]
        collision[1]["seller_uids"] = list(collision[0]["seller_uids"])
        with self.assertRaisesRegex(
            preparer_v94.QualityProbePreparerV94Error,
            "seller UID split collision",
        ):
            preparer_v94._prepare_split(
                worlds=collision,
                noise_signatures=signatures,
                time_key_hex="01" * 32,
                expected_world_count=2,
                split_schedule_commitment_sha256="02" * 32,
                schedule_pair_audit_commitment_sha256="04" * 32,
                noise_signature_set_commitment_sha256="03" * 32,
            )
        private = [dict(row) for row in worlds]
        private[0] = {**private[0], "controller_groups": []}
        with self.assertRaisesRegex(ValueError, "schema/order"):
            preparer_v94._prepare_split(
                worlds=private,
                noise_signatures=signatures,
                time_key_hex="01" * 32,
                expected_world_count=2,
                split_schedule_commitment_sha256="02" * 32,
                schedule_pair_audit_commitment_sha256="04" * 32,
                noise_signature_set_commitment_sha256="03" * 32,
            )


if __name__ == "__main__":
    unittest.main()
