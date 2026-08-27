from __future__ import annotations

from types import MappingProxyType
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_probe_labels_v9_4 as labels_v94
import step28_v13_v1_13_quality_probe_preparer_v9_4 as preparer_v94


def fixture():
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
            "world_ordinal": index,
            "world_uid": f"world_{index:03d}",
            "seller_uids": [
                f"world_{index:03d}_seller_{seller:02d}"
                for seller in range(28)
            ],
            "noise_slot_by_seller_slot": list(range(28)),
        }
        for index in range(2)
    ]
    prepared = preparer_v94._prepare_split(
        worlds=worlds,
        noise_signatures=signatures,
        time_key_hex="01" * 32,
        expected_world_count=2,
        split_schedule_commitment_sha256="02" * 32,
        schedule_pair_audit_commitment_sha256="04" * 32,
        noise_signature_set_commitment_sha256="03" * 32,
    )
    labels = tuple(
        MappingProxyType({
            "world_uid": world_uid,
            "canonical_pair_uid": pair_uid,
            "y_true": int(pair_index < 20),
        })
        for row_index, (world_uid, pair_uid) in enumerate(prepared.matrix.row_keys)
        for pair_index in (row_index % 378,)
    )
    return prepared, labels


class QualityProbeLabelsV94Contracts(unittest.TestCase):
    def test_labels_are_immutable_and_bound_to_the_prepared_matrix(self) -> None:
        prepared, rows = fixture()
        frozen = labels_v94._freeze_labels_after_preparation(
            prepared=prepared,
            label_rows=rows,
            split_schedule_commitment_sha256="02" * 32,
            private_controller_truth_sha256="04" * 32,
            truth_source_version=labels_v94.schedule_v94.VERSION,
            truth_formula=labels_v94.TRUTH_FORMULA,
            truth_read_count=1,
            audit_truth_read_count=0,
            expected_world_count=2,
        )
        labels_v94.verify_frozen_labels(
            frozen,
            prepared=prepared,
            expected_world_count=2,
        )
        self.assertFalse(frozen.values.flags.writeable)
        with self.assertRaises(ValueError):
            frozen.values.setflags(write=True)
        with self.assertRaises(TypeError):
            frozen.commitment["row_count"] = 0

    def test_plain_dict_reordering_and_wrong_positive_count_are_rejected(self) -> None:
        prepared, rows = fixture()
        plain = tuple(dict(row) for row in rows)
        with self.assertRaisesRegex(
            labels_v94.QualityProbeLabelsV94Error,
            "schema/type",
        ):
            labels_v94._freeze_labels_after_preparation(
                prepared=prepared,
                label_rows=plain,
                split_schedule_commitment_sha256="02" * 32,
                private_controller_truth_sha256="04" * 32,
                truth_source_version=labels_v94.schedule_v94.VERSION,
                truth_formula=labels_v94.TRUTH_FORMULA,
                truth_read_count=1,
                audit_truth_read_count=0,
                expected_world_count=2,
            )
        forged = list(rows)
        forged[20] = MappingProxyType({
            **dict(forged[20]),
            "y_true": 1,
        })
        with self.assertRaisesRegex(
            labels_v94.QualityProbeLabelsV94Error,
            "per-world closure",
        ):
            labels_v94._freeze_labels_after_preparation(
                prepared=prepared,
                label_rows=tuple(forged),
                split_schedule_commitment_sha256="02" * 32,
                private_controller_truth_sha256="04" * 32,
                truth_source_version=labels_v94.schedule_v94.VERSION,
                truth_formula=labels_v94.TRUTH_FORMULA,
                truth_read_count=1,
                audit_truth_read_count=0,
                expected_world_count=2,
            )

    def test_truth_provenance_must_match_prepared_schedule(self) -> None:
        prepared, rows = fixture()
        with self.assertRaisesRegex(
            labels_v94.QualityProbeLabelsV94Error,
            "truth provenance drift",
        ):
            labels_v94._freeze_labels_after_preparation(
                prepared=prepared,
                label_rows=rows,
                split_schedule_commitment_sha256="05" * 32,
                private_controller_truth_sha256="04" * 32,
                truth_source_version=labels_v94.schedule_v94.VERSION,
                truth_formula=labels_v94.TRUTH_FORMULA,
                truth_read_count=1,
                audit_truth_read_count=0,
                expected_world_count=2,
            )

    def test_audit_truth_is_absent_from_connector_contract(self) -> None:
        contract = labels_v94.contract_payload()
        self.assertEqual(contract["audit_truth_access_count"], 0)
        self.assertEqual(contract["allowed_splits"], ["train", "development"])
        self.assertEqual(contract["formal_truth_read_count"], 1)
        self.assertFalse(contract["public_truth_connector"])
        self.assertFalse(
            hasattr(labels_v94, "open_controller_truth_after_preparation")
        )


if __name__ == "__main__":
    unittest.main()
