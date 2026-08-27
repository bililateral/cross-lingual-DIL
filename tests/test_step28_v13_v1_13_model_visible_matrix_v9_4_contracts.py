from __future__ import annotations

import ast
import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_model_visible_matrix_v9_4 as matrix_v94


def seller(
    item_count: float,
    title_missing: float,
    description_missing: float,
    buckets: tuple[float, float, float, float],
) -> dict[str, float]:
    return {
        "item_count": item_count,
        "title_missing_rate": title_missing,
        "description_missing_rate": description_missing,
        "time_bucket_probability_00": buckets[0],
        "time_bucket_probability_01": buckets[1],
        "time_bucket_probability_02": buckets[2],
        "time_bucket_probability_03": buckets[3],
    }


class ModelVisibleMatrixV94Contracts(unittest.TestCase):
    def setUp(self) -> None:
        self.left = seller(2.0, 0.5, 0.0, (0.5, 0.5, 0.0, 0.0))
        self.right = seller(4.0, 0.0, 0.5, (0.0, 0.25, 0.25, 0.5))
        self.row_a = matrix_v94.pair_feature_row(
            world_uid="world_a",
            canonical_pair_uid="pair_a",
            left=self.left,
            right=self.right,
        )
        self.row_b = matrix_v94.pair_feature_row(
            world_uid="world_b",
            canonical_pair_uid="pair_b",
            left=self.right,
            right=self.left,
        )

    def test_contract_is_exact_fourteen_columns_and_label_free(self) -> None:
        payload = matrix_v94.contract_payload()
        self.assertEqual(payload["pair_feature_count"], 14)
        self.assertEqual(tuple(payload["pair_features"]), matrix_v94.PAIR_FEATURES)
        self.assertFalse(payload["labels_read"])
        self.assertFalse(
            any(
                fragment in name
                for fragment in matrix_v94.FORBIDDEN_FIELD_FRAGMENTS
                for name in matrix_v94.PAIR_FEATURES
            )
        )

    def test_matrix_module_has_no_project_data_dependency(self) -> None:
        source = Path(matrix_v94.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        self.assertFalse(any(name.startswith("step28") for name in imported))

    def test_freeze_matrix_is_read_only_and_deterministic(self) -> None:
        first = matrix_v94.freeze_matrix([self.row_a, self.row_b], expected_row_count=2)
        second = matrix_v94.freeze_matrix([self.row_a, self.row_b], expected_row_count=2)
        self.assertFalse(first.values.flags.writeable)
        with self.assertRaises(ValueError):
            first.values.setflags(write=True)
        self.assertEqual(first.raw_f8_c_sha256, second.raw_f8_c_sha256)
        self.assertEqual(
            first.joint_commitment_sha256, second.joint_commitment_sha256
        )
        np.testing.assert_array_equal(first.values, second.values)
        self.assertEqual(first.column_names, matrix_v94.PAIR_FEATURES)

    def test_private_or_label_derived_extra_columns_are_rejected(self) -> None:
        for field in (
            "controller_size_min",
            "registered_treatment",
            "role_clone_source_count",
            "label",
            "candidate_index",
            "noise_slot_min",
        ):
            with self.subTest(field=field):
                forged = {**self.row_a, field: "0.000000000000"}
                with self.assertRaisesRegex(
                    matrix_v94.ModelVisibleMatrixV94Error, "schema/order"
                ):
                    matrix_v94.freeze_matrix([forged], expected_row_count=1)

    def test_duplicate_or_noncanonical_row_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "unique canonical order"
        ):
            matrix_v94.freeze_matrix([self.row_b, self.row_a], expected_row_count=2)
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "unique canonical order"
        ):
            matrix_v94.freeze_matrix([self.row_a, self.row_a], expected_row_count=2)

    def test_float_serialization_and_pair_semantics_are_enforced(self) -> None:
        malformed = dict(self.row_a)
        malformed["absdiff__item_count"] = "2.0"
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "serialization"
        ):
            matrix_v94.freeze_matrix([malformed], expected_row_count=1)
        negative_zero = dict(self.row_a)
        negative_zero["absdiff__title_missing_rate"] = "-0.000000000000"
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "noncanonical zero"
        ):
            matrix_v94.freeze_matrix([negative_zero], expected_row_count=1)
        impossible = dict(self.row_a)
        impossible["sum__item_count"] = "5.000000000000"
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "Item-count"
        ):
            matrix_v94.freeze_matrix([impossible], expected_row_count=1)
        broken_time = dict(self.row_a)
        broken_time["absdiff__time_bucket_probability_03"] = "0.400000000000"
        broken_time["sum__time_bucket_probability_03"] = "0.400000000000"
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "sum to two"
        ):
            matrix_v94.freeze_matrix([broken_time], expected_row_count=1)

    def test_reopened_pair_requires_integer_item_endpoint_witness(self) -> None:
        impossible = dict(self.row_a)
        impossible["absdiff__item_count"] = "0.000000000000"
        impossible["sum__item_count"] = "4.000000000000"
        impossible["absdiff__title_missing_rate"] = "0.100000000000"
        impossible["sum__title_missing_rate"] = "0.500000000000"
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "integer-item endpoint witness"
        ):
            matrix_v94.freeze_matrix([impossible], expected_row_count=1)
        negative_difference = dict(self.row_a)
        negative_difference["absdiff__title_missing_rate"] = "-0.000000000001"
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "Bounded pair semantics"
        ):
            matrix_v94.freeze_matrix(
                [negative_difference], expected_row_count=1
            )

    def test_joint_time_witness_rejects_individually_plausible_bucket_pairs(self) -> None:
        impossible = dict(self.row_a)
        impossible["absdiff__item_count"] = "1.000000000000"
        impossible["sum__item_count"] = "5.000000000000"
        for name in ("title_missing_rate", "description_missing_rate"):
            impossible[f"absdiff__{name}"] = "0.000000000000"
            impossible[f"sum__{name}"] = "0.000000000000"
        for index in range(4):
            impossible[f"absdiff__time_bucket_probability_{index:02d}"] = (
                "0.500000000000"
            )
            impossible[f"sum__time_bucket_probability_{index:02d}"] = (
                "0.500000000000"
            )
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error,
            "integer-item endpoint witness",
        ):
            matrix_v94.freeze_matrix([impossible], expected_row_count=1)

    def test_one_third_rounding_and_all_item_count_pairs_have_witnesses(self) -> None:
        left = seller(2.0, 0.0, 0.0, (0.5, 0.5, 0.0, 0.0))
        right = seller(
            3.0,
            0.0,
            0.0,
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        )
        rounded = matrix_v94.pair_feature_row(
            world_uid="rounding",
            canonical_pair_uid="seller_00||seller_01",
            left=left,
            right=right,
        )
        self.assertEqual(
            rounded["absdiff__time_bucket_probability_00"],
            "0.166666666667",
        )
        self.assertEqual(
            rounded["sum__time_bucket_probability_00"],
            "0.833333333333",
        )
        rows = []
        for left_count in range(2, 9):
            for right_count in range(2, 9):
                left_buckets = [0.0] * 4
                right_buckets = [0.0] * 4
                left_buckets[left_count % 4] = 1.0
                right_buckets[(right_count + 1) % 4] = 1.0
                rows.append(matrix_v94.pair_feature_row(
                    world_uid="all_counts",
                    canonical_pair_uid=(
                        f"left_{left_count:02d}||right_{right_count:02d}"
                    ),
                    left=seller(
                        float(left_count),
                        (left_count // 2) / left_count,
                        (left_count // 3) / left_count,
                        tuple(left_buckets),
                    ),
                    right=seller(
                        float(right_count),
                        (right_count // 2) / right_count,
                        (right_count // 3) / right_count,
                        tuple(right_buckets),
                    ),
                ))
        rows.sort(key=lambda row: row["canonical_pair_uid"].encode("utf-8"))
        frozen = matrix_v94.freeze_matrix(rows, expected_row_count=49)
        self.assertEqual(frozen.values.shape, (49, 14))

    def test_fixed_twelve_decimal_noninteger_fraction_is_rejected(self) -> None:
        impossible = dict(self.row_a)
        impossible["absdiff__item_count"] = "0.000000000000"
        impossible["sum__item_count"] = "6.000000000000"
        impossible["absdiff__title_missing_rate"] = "0.000000000000"
        impossible["sum__title_missing_rate"] = "0.666666667100"
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error,
            "integer-item endpoint witness",
        ):
            matrix_v94.freeze_matrix([impossible], expected_row_count=1)

    def test_seller_schema_order_is_exact(self) -> None:
        reordered = dict(reversed(tuple(self.left.items())))
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "schema/order"
        ):
            matrix_v94.pair_feature_row(
                world_uid="world_a",
                canonical_pair_uid="pair_a",
                left=reordered,
                right=self.right,
            )

    def test_nonmapping_and_nonnumeric_inputs_are_contract_rejected(self) -> None:
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "not a mapping"
        ):
            matrix_v94.freeze_matrix(
                [list(matrix_v94.ROW_FIELDS)], expected_row_count=1
            )
        bad = dict(self.left)
        bad["item_count"] = "2"
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "nonnumeric"
        ):
            matrix_v94.pair_feature_row(
                world_uid="world_a",
                canonical_pair_uid="pair_a",
                left=bad,
                right=self.right,
            )

    def test_seller_values_must_be_real_item_fractions(self) -> None:
        bad_sum = dict(self.left)
        bad_sum["time_bucket_probability_00"] = 0.25
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "sum to one"
        ):
            matrix_v94.pair_feature_row(
                world_uid="world_a",
                canonical_pair_uid="pair_a",
                left=bad_sum,
                right=self.right,
            )
        bad_fraction = dict(self.left)
        bad_fraction["title_missing_rate"] = 0.25
        with self.assertRaisesRegex(
            matrix_v94.ModelVisibleMatrixV94Error, "item-count fraction"
        ):
            matrix_v94.pair_feature_row(
                world_uid="world_a",
                canonical_pair_uid="pair_a",
                left=bad_fraction,
                right=self.right,
            )


if __name__ == "__main__":
    unittest.main()
