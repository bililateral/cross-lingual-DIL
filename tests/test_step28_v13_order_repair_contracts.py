from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import step28_v13_build_training_ready_dataset as builder  # noqa: E402
import step28_v13_common as common  # noqa: E402
import step28_v13_finalize_training_ready_dataset as finalizer  # noqa: E402
import step28_v13_validate_order_repair_equivalence as equivalence  # noqa: E402


class Step28V13OrderRepairContracts(unittest.TestCase):
    key_hex = "31" * 32

    def _policy(self):
        return {
            "randomness": {
                builder.MODE: {"candidate_key_hex": self.key_hex}
            }
        }

    def _rows(self, world_uid: str):
        pair_uids = [f"pair_{index:02d}" for index in range(40)]
        ordered = sorted(
            pair_uids,
            key=lambda pair_uid: (
                common.hmac_digest(
                    self.key_hex,
                    world_uid,
                    "selected_global_rank",
                    pair_uid,
                ),
                pair_uid.encode("utf-8"),
            ),
        )
        return [
            {
                "canonical_pair_uid": pair_uid,
                "world_uid": world_uid,
                "seller_uid_left": "seller_left",
                "seller_uid_right": "seller_right",
            }
            for pair_uid in ordered
        ]

    def test_builder_and_finalizer_independently_accept_exact_order(
        self,
    ) -> None:
        rows = self._rows("world_1")
        expected = {
            "world_count": 1,
            "candidate_pair_count": 40,
            "world_blocks_contiguous_and_exact": True,
            "independent_selected_global_rank_exact": True,
            "labels_or_controller_membership_read": False,
        }
        self.assertEqual(
            builder._validate_candidate_output_order(
                self._policy(),
                candidate_rows=rows,
                expected_world_uids=["world_1"],
            ),
            expected,
        )
        self.assertEqual(
            finalizer._replay_candidate_output_order_independently(
                policy=self._policy(),
                candidates=rows,
                expected_world_uids=["world_1"],
            ),
            expected,
        )

    def test_old_or_interleaved_order_fails_closed(self) -> None:
        rows = self._rows("world_1")
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaises(common.ContractError):
            builder._validate_candidate_output_order(
                self._policy(),
                candidate_rows=rows,
                expected_world_uids=["world_1"],
            )
        with self.assertRaises(common.ContractError):
            finalizer._replay_candidate_output_order_independently(
                policy=self._policy(),
                candidates=rows,
                expected_world_uids=["world_1"],
            )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_equivalence_comparison_is_order_insensitive_but_not_value_blind(
        self,
    ) -> None:
        parent_rows = [
            {
                "canonical_pair_uid": "pair_a",
                "world_uid": "world_1",
                "label": "1",
                "selected_rank": "1",
            },
            {
                "canonical_pair_uid": "pair_b",
                "world_uid": "world_1",
                "label": "0",
                "selected_rank": "2",
            },
        ]
        repaired_rows = [
            {**parent_rows[1], "selected_rank": "1"},
            {**parent_rows[0], "selected_rank": "2"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent.csv"
            repaired = Path(directory) / "repaired.csv"
            self._write_csv(parent, parent_rows)
            self._write_csv(repaired, repaired_rows)
            receipt = equivalence._sorted_csv_equivalent(
                parent,
                repaired,
                ignore_fields=("selected_rank",),
            )
            self.assertTrue(receipt["pair_keyset_exact"])
            repaired_rows[0]["label"] = "1"
            self._write_csv(repaired, repaired_rows)
            with self.assertRaises(common.ContractError):
                equivalence._sorted_csv_equivalent(
                    parent,
                    repaired,
                    ignore_fields=("selected_rank",),
                )

    def test_equivalence_paths_follow_public_and_sealed_split_layouts(
        self,
    ) -> None:
        train_paths = equivalence._split_semantic_paths("train")
        development_paths = equivalence._split_semantic_paths(
            "development"
        )
        audit_a_paths = equivalence._split_semantic_paths("audit_a")
        audit_b_paths = equivalence._split_semantic_paths("audit_b")
        self.assertEqual(train_paths, development_paths)
        self.assertEqual(audit_a_paths, audit_b_paths)
        self.assertEqual(
            train_paths["metadata_shortcut_report"],
            "audit/metadata_shortcut_audit.json",
        )
        self.assertEqual(
            train_paths["metadata_shortcut_oof"],
            "private_audit/metadata_shortcut_oof.csv",
        )
        self.assertEqual(
            audit_a_paths["metadata_shortcut_report"],
            (
                "sealed_supervision/"
                "metadata_shortcut_audit.private.json"
            ),
        )
        self.assertEqual(
            audit_a_paths["metadata_shortcut_oof"],
            (
                "sealed_supervision/"
                "metadata_shortcut_oof.private.csv"
            ),
        )
        self.assertEqual(
            equivalence.ORDER_ONLY_DIFFERING_PATHS,
            {
                "observed/candidate_pairs.csv",
                "private_audit/candidate_sampling_audit.csv",
                "private_audit/world_generation_audit.jsonl",
            },
        )

    def test_equivalence_missing_path_fails_as_contract_error(
        self,
    ) -> None:
        required = {"present.csv", "missing.csv"}
        with self.assertRaisesRegex(
            common.ContractError,
            r"required path missing for audit_a:.*missing\.csv",
        ):
            equivalence._require_manifest_paths(
                split="audit_a",
                parent_files={"present.csv": {}},
                repaired_files={"present.csv": {}, "missing.csv": {}},
                required=required,
            )


if __name__ == "__main__":
    unittest.main()
