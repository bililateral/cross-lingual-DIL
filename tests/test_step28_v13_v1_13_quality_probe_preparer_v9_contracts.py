from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_channel_views_v9 as channel
import step28_v13_v1_13_quality_probe_preparer_v9 as preparer


def encode_plain_coordinate(value: int) -> str:
    return "Q" + "".join(
        chr(ord("A") + int(symbol, 16)) for symbol in f"{value:010x}"
    )


def decode_plain_coordinate(_world_uid: str, code: str) -> int:
    return int("".join(f"{ord(symbol) - ord('A'):x}" for symbol in code[1:]), 16)


def expected_seller_slots(*world_uids: str) -> dict[tuple[str, str], int]:
    return {
        (world_uid, f"{world_uid}_seller_{slot:02d}"): slot
        for world_uid in world_uids
        for slot in range(28)
    }


def build_world(
    world_uid: str, mode_ordinal: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    public_rows: list[dict[str, object]] = []
    seller_uids = [f"{world_uid}_seller_{slot:02d}" for slot in range(28)]
    for seller_slot, seller_uid in enumerate(seller_uids):
        codes = tuple(
            encode_plain_coordinate(
                mode_ordinal * 256 + seller_slot * 8 + item_slot
            )
            for item_slot in range(1 + seller_slot % 3)
        )
        item_occurrences = [
            {
                "field": "title" if item_slot % 2 == 0 else "description",
                "code": code,
                "is_own": True,
            }
            for item_slot, code in enumerate(codes)
        ]
        profile_occurrences = [
            {"field": field, "code": code, "is_own": True}
            for field in channel.PROFILE_FIELDS
            for code in codes
        ]
        public_rows.append(
            {
                "world_uid": world_uid,
                "seller_uid": seller_uid,
                "owned_codes": list(codes),
                "item_occurrences": item_occurrences,
                "profile_occurrences": profile_occurrences,
                "numeric_profile_deltas": {
                    name: 0.0 for name in channel.NUMERIC_DELTA_FIELDS
                },
            }
        )
    endpoints = [
        {
            "canonical_pair_uid": f"{world_uid}_pair_{left:02d}_{right:02d}",
            "world_uid": world_uid,
            "seller_uid_left": seller_uids[left],
            "seller_uid_right": seller_uids[right],
        }
        for left in range(28)
        for right in range(left + 1, 28)
    ]
    endpoints.sort(key=lambda row: str(row["canonical_pair_uid"]).encode("utf-8"))
    return public_rows, endpoints


class QualityProbePreparerV9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.public_rows, cls.endpoints = build_world("world_000", 17)
        cls.public_rows = json.loads(
            json.dumps(cls.public_rows, ensure_ascii=False, sort_keys=True)
        )
        cls.sources = (
            preparer.SourceCommitment(
                path="train/private/public_code_probe_input.jsonl",
                size_bytes=123,
                sha256="1" * 64,
            ),
        )

    def test_preparer_has_no_label_or_controller_input(self) -> None:
        for function in (
            preparer.prepare_public_code_matrix,
            preparer.prepare_decoded_slot_matrix,
            preparer.freeze_feature_matrix,
        ):
            parameters = set(inspect.signature(function).parameters)
            self.assertFalse(
                parameters & {"label", "labels", "truth", "controller", "qrels"}
            )
        with self.assertRaises(preparer.QualityProbePreparationError):
            preparer.reject_truth_input(labels=[0, 1])

    def test_public_and_decoded_matrices_freeze_exactly(self) -> None:
        public = preparer.prepare_public_code_matrix(
            public_rows=self.public_rows,
            endpoints=self.endpoints,
            ordered_world_uids=("world_000",),
            sources=self.sources,
        )
        decoded = preparer.prepare_decoded_slot_matrix(
            public_rows=self.public_rows,
            endpoints=self.endpoints,
            ordered_world_uids=("world_000",),
            expected_mode_global_ordinal_by_world={"world_000": 17},
            expected_seller_slot_by_world_and_seller=expected_seller_slots(
                "world_000"
            ),
            decode_coordinate=decode_plain_coordinate,
            sources=self.sources,
        )
        combined = preparer.combine_frozen_matrices(
            view="public_plus_decoded_3380", matrices=(public, decoded)
        )
        self.assertEqual(public.values.shape, (378, 2992))
        self.assertEqual(decoded.values.shape, (378, 388))
        self.assertEqual(combined.values.shape, (378, 3380))
        self.assertFalse(public.values.flags.writeable)
        self.assertEqual(
            preparer.verify_frozen_feature_matrix(public)["label_count_read"], 0
        )
        self.assertEqual(
            preparer.verify_frozen_feature_matrix(decoded)[
                "decoded_values_persisted"
            ],
            False,
        )

    def test_matrix_mutation_after_freeze_is_detected(self) -> None:
        frozen = preparer.prepare_decoded_slot_matrix(
            public_rows=self.public_rows,
            endpoints=self.endpoints,
            ordered_world_uids=("world_000",),
            expected_mode_global_ordinal_by_world={"world_000": 17},
            expected_seller_slot_by_world_and_seller=expected_seller_slots(
                "world_000"
            ),
            decode_coordinate=decode_plain_coordinate,
            sources=self.sources,
        )
        frozen.values.setflags(write=True)
        frozen.values[0, 0] += 1.0
        frozen.values.setflags(write=False)
        with self.assertRaises(preparer.QualityProbePreparationError):
            preparer.verify_frozen_feature_matrix(frozen)

    def test_empty_source_commitment_list_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            preparer.QualityProbePreparationError, "list is empty"
        ):
            preparer.freeze_feature_matrix(
                family="fixture",
                view="fixture",
                values=np.zeros((1, 1), dtype=np.float64),
                row_keys=(("world", "pair"),),
                column_names=("feature",),
                sources=(),
            )

    def test_matrix_values_are_not_silently_cast_to_float64(self) -> None:
        for values in (
            np.asarray([[1]], dtype=np.int64),
            np.asarray([["1.0"]], dtype=object),
            [[1.0]],
        ):
            with self.subTest(value_type=type(values).__name__):
                with self.assertRaisesRegex(
                    preparer.QualityProbePreparationError,
                    "must already be.*float64",
                ):
                    preparer.freeze_feature_matrix(
                        family="fixture",
                        view="fixture",
                        values=values,  # type: ignore[arg-type]
                        row_keys=(("world", "pair"),),
                        column_names=("feature",),
                        sources=self.sources,
                    )

    def test_chunked_matrix_hash_matches_canonical_raw_bytes(self) -> None:
        values = np.arange(24, dtype=np.float64).reshape(4, 6)
        self.assertEqual(
            preparer._matrix_sha256(values),
            hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        )

    def test_streamed_bitmap_crosses_eight_megabyte_boundary(self) -> None:
        for remainder in range(1, 8):
            values = np.zeros((1_048_576 + remainder, 1), dtype=np.float64)
            values[-1, 0] = np.nan
            nonfinite, observed = preparer._stream_nonfinite_and_missing_bitmap(values)
            expected = hashlib.sha256(
                np.packbits(
                    np.isnan(values).ravel(order="C"), bitorder="little"
                ).tobytes()
            ).hexdigest()
            with self.subTest(remainder=remainder):
                self.assertEqual(nonfinite, 1)
                self.assertEqual(observed, expected)

    def test_freeze_finite_checks_never_exceed_fixed_chunk(self) -> None:
        values = np.zeros((25, 1), dtype=np.float64)
        row_keys = tuple(("world", f"pair_{index}") for index in range(25))
        observed_sizes: list[int] = []
        original_isfinite = np.isfinite

        def monitored_isfinite(value: np.ndarray) -> np.ndarray:
            observed_sizes.append(np.asarray(value).size)
            return original_isfinite(value)

        with (
            patch.object(preparer, "MATRIX_CHUNK_BYTES", 64),
            patch.object(preparer.np, "isfinite", monitored_isfinite),
        ):
            preparer.freeze_feature_matrix(
                family="fixture",
                view="fixture",
                values=values,
                row_keys=row_keys,
                column_names=("feature",),
                sources=self.sources,
            )
        self.assertTrue(observed_sizes)
        self.assertLessEqual(max(observed_sizes), 8)

    def test_public_freeze_has_no_caller_array_alias(self) -> None:
        values = np.asarray([[1.0], [2.0]], dtype=np.float64)
        frozen = preparer.freeze_feature_matrix(
            family="fixture",
            view="fixture",
            values=values,
            row_keys=(("world", "pair_a"), ("world", "pair_b")),
            column_names=("feature",),
            sources=self.sources,
        )
        values[:, 0] = 99.0
        np.testing.assert_array_equal(frozen.values[:, 0], np.asarray([1.0, 2.0]))
        preparer.verify_frozen_feature_matrix(frozen)

    def test_frozen_matrix_dtype_replacement_is_detected(self) -> None:
        frozen = preparer.freeze_feature_matrix(
            family="fixture",
            view="fixture",
            values=np.asarray([[1.0]], dtype=np.float64),
            row_keys=(("world", "pair"),),
            column_names=("feature",),
            sources=self.sources,
        )
        replacement = np.asarray([[1]], dtype=np.int64)
        replacement.setflags(write=False)
        object.__setattr__(frozen, "values", replacement)
        with self.assertRaises(preparer.QualityProbePreparationError):
            preparer.verify_frozen_feature_matrix(frozen)

    def test_uid_change_changes_only_row_commitment_not_values(self) -> None:
        first = preparer.freeze_feature_matrix(
            family="fixture",
            view="fixture",
            values=np.asarray([[1.0], [2.0]]),
            row_keys=(("world", "pair_a"), ("world", "pair_b")),
            column_names=("safe_feature",),
            sources=self.sources,
        )
        second = preparer.freeze_feature_matrix(
            family="fixture",
            view="fixture",
            values=np.asarray([[1.0], [2.0]]),
            row_keys=(("world", "renamed_a"), ("world", "renamed_b")),
            column_names=("safe_feature",),
            sources=self.sources,
        )
        first_receipt = preparer.verify_frozen_feature_matrix(first)
        second_receipt = preparer.verify_frozen_feature_matrix(second)
        self.assertEqual(
            first_receipt["matrix_sha256"], second_receipt["matrix_sha256"]
        )
        self.assertNotEqual(
            first_receipt["row_keys_canonical_json_sha256"],
            second_receipt["row_keys_canonical_json_sha256"],
        )

    def test_extra_label_field_and_endpoint_reorder_fail_closed(self) -> None:
        contaminated = [dict(row) for row in self.public_rows]
        contaminated[0]["label"] = 1
        with self.assertRaises(preparer.QualityProbePreparationError):
            preparer.prepare_public_code_matrix(
                public_rows=contaminated,
                endpoints=self.endpoints,
                ordered_world_uids=("world_000",),
                sources=self.sources,
            )
        reordered = [dict(row) for row in self.endpoints]
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaises(preparer.QualityProbePreparationError):
            preparer.prepare_public_code_matrix(
                public_rows=self.public_rows,
                endpoints=reordered,
                ordered_world_uids=("world_000",),
                sources=self.sources,
            )

    def test_identifier_and_code_values_are_not_string_coerced(self) -> None:
        endpoints = [dict(row) for row in self.endpoints]
        endpoints[0]["world_uid"] = 123
        with self.assertRaises(preparer.QualityProbePreparationError):
            preparer.prepare_public_code_matrix(
                public_rows=self.public_rows,
                endpoints=endpoints,
                ordered_world_uids=("world_000",),
                sources=self.sources,
            )

    def test_public_occurrences_close_to_unique_world_code_owners(self) -> None:
        legitimate = copy.deepcopy(self.public_rows)
        foreign_code = legitimate[1]["owned_codes"][0]
        legitimate[0]["item_occurrences"].append(
            {"field": "title", "code": foreign_code, "is_own": False}
        )
        accepted = preparer.prepare_public_code_matrix(
            public_rows=legitimate,
            endpoints=self.endpoints,
            ordered_world_uids=("world_000",),
            sources=self.sources,
        )
        self.assertEqual(accepted.values.shape, (378, 2992))

        unregistered = copy.deepcopy(self.public_rows)
        unregistered[0]["item_occurrences"].append(
            {"field": "title", "code": "QPPPPPPPPPP", "is_own": False}
        )
        with self.assertRaisesRegex(
            preparer.QualityProbePreparationError, "world owner universe"
        ):
            preparer.prepare_public_code_matrix(
                public_rows=unregistered,
                endpoints=self.endpoints,
                ordered_world_uids=("world_000",),
                sources=self.sources,
            )

        duplicate_owner = copy.deepcopy(self.public_rows)
        old_code = duplicate_owner[1]["owned_codes"][0]
        shared_code = duplicate_owner[0]["owned_codes"][0]
        duplicate_owner[1]["owned_codes"][0] = shared_code
        for field in ("item_occurrences", "profile_occurrences"):
            for occurrence in duplicate_owner[1][field]:
                if occurrence["code"] == old_code:
                    occurrence["code"] = shared_code
        with self.assertRaisesRegex(
            preparer.QualityProbePreparationError, "multiple owners"
        ):
            preparer.prepare_public_code_matrix(
                public_rows=duplicate_owner,
                endpoints=self.endpoints,
                ordered_world_uids=("world_000",),
                sources=self.sources,
            )

    def test_378_rows_cannot_hide_duplicate_unordered_seller_pair(self) -> None:
        endpoints = [dict(row) for row in self.endpoints]
        endpoints[0]["seller_uid_left"] = endpoints[1]["seller_uid_left"]
        endpoints[0]["seller_uid_right"] = endpoints[1]["seller_uid_right"]
        with self.assertRaises(preparer.QualityProbePreparationError):
            preparer.prepare_public_code_matrix(
                public_rows=self.public_rows,
                endpoints=endpoints,
                ordered_world_uids=("world_000",),
                sources=self.sources,
            )
        public_rows = [dict(row) for row in self.public_rows]
        public_rows[0]["owned_codes"] = list(public_rows[0]["owned_codes"])
        public_rows[0]["owned_codes"][0] = 123
        with self.assertRaises(preparer.QualityProbePreparationError):
            preparer.prepare_public_code_matrix(
                public_rows=public_rows,
                endpoints=self.endpoints,
                ordered_world_uids=("world_000",),
                sources=self.sources,
            )

    def test_frozen_text_eligibility_detects_bitmap_mutation(self) -> None:
        rows = [
            {
                "world_uid": row["world_uid"],
                "canonical_pair_uid": row["canonical_pair_uid"],
                "text_probe_eligible": index >= 6,
            }
            for index, row in enumerate(self.endpoints)
        ]
        rows = json.loads(json.dumps(rows, ensure_ascii=False, sort_keys=True))
        frozen = preparer.freeze_text_eligibility(
            eligibility_rows=rows,
            endpoints=self.endpoints,
            ordered_world_uids=("world_000",),
            sources=self.sources,
        )
        receipt = preparer.verify_frozen_text_eligibility(frozen)
        self.assertEqual(receipt["complete_row_count"], 378)
        self.assertEqual(receipt["eligible_row_count"], 372)
        frozen.values.setflags(write=True)
        frozen.values[0] = True
        frozen.values.setflags(write=False)
        with self.assertRaises(preparer.QualityProbePreparationError):
            preparer.verify_frozen_text_eligibility(frozen)

    def test_wrong_decoded_world_and_seller_slots_fail_closed(self) -> None:
        def wrong_world(world_uid: str, code: str) -> int:
            value = decode_plain_coordinate(world_uid, code)
            return value + 256 if code == self.public_rows[0]["owned_codes"][0] else value

        with self.assertRaises(preparer.QualityProbePreparationError):
            preparer.prepare_decoded_slot_matrix(
                public_rows=self.public_rows,
                endpoints=self.endpoints,
                ordered_world_uids=("world_000",),
                expected_mode_global_ordinal_by_world={"world_000": 17},
                expected_seller_slot_by_world_and_seller=expected_seller_slots(
                    "world_000"
                ),
                decode_coordinate=wrong_world,
                sources=self.sources,
            )
        with self.assertRaises(preparer.QualityProbePreparationError):
            preparer.prepare_decoded_slot_matrix(
                public_rows=self.public_rows,
                endpoints=self.endpoints,
                ordered_world_uids=("world_000",),
                expected_mode_global_ordinal_by_world={"world_000": 18},
                expected_seller_slot_by_world_and_seller=expected_seller_slots(
                    "world_000"
                ),
                decode_coordinate=decode_plain_coordinate,
                sources=self.sources,
            )
        def reverse_seller_slot(world_uid: str, code: str) -> int:
            value = decode_plain_coordinate(world_uid, code)
            return (
                (value // preparer.WORLD_STRIDE) * preparer.WORLD_STRIDE
                + (27 - ((value % preparer.WORLD_STRIDE) // preparer.SELLER_STRIDE))
                * preparer.SELLER_STRIDE
                + value % preparer.SELLER_STRIDE
            )

        with self.assertRaisesRegex(
            preparer.QualityProbePreparationError, "seller slot disagrees"
        ):
            preparer.prepare_decoded_slot_matrix(
                public_rows=self.public_rows,
                endpoints=self.endpoints,
                ordered_world_uids=("world_000",),
                expected_mode_global_ordinal_by_world={"world_000": 17},
                expected_seller_slot_by_world_and_seller=expected_seller_slots(
                    "world_000"
                ),
                decode_coordinate=reverse_seller_slot,
                sources=self.sources,
            )

    def test_world_blocks_and_decoded_ordinals_cannot_overlap(self) -> None:
        rows_a, endpoints_a = build_world("world_000", 17)
        rows_b, endpoints_b = build_world("world_001", 18)
        frozen = preparer.prepare_decoded_slot_matrix(
            public_rows=rows_a + rows_b,
            endpoints=endpoints_a + endpoints_b,
            ordered_world_uids=("world_000", "world_001"),
            expected_mode_global_ordinal_by_world={"world_000": 17, "world_001": 18},
            expected_seller_slot_by_world_and_seller=expected_seller_slots(
                "world_000", "world_001"
            ),
            decode_coordinate=decode_plain_coordinate,
            sources=self.sources,
        )
        self.assertEqual(frozen.values.shape, (756, 388))
        with self.assertRaises(preparer.QualityProbePreparationError):
            preparer.prepare_decoded_slot_matrix(
                public_rows=rows_a + rows_b,
                endpoints=endpoints_a + endpoints_b,
                ordered_world_uids=("world_000", "world_001"),
                expected_mode_global_ordinal_by_world={"world_000": 17, "world_001": 18},
                expected_seller_slot_by_world_and_seller=expected_seller_slots(
                    "world_000", "world_001"
                ),
                decode_coordinate=lambda world_uid, code: (
                    decode_plain_coordinate(world_uid, code) % 256
                ),
                sources=self.sources,
            )


if __name__ == "__main__":
    unittest.main()
