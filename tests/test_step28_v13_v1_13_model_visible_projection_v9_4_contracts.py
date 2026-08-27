from __future__ import annotations

import ast
import itertools
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_model_visible_matrix_v9_4 as matrix_v94
import step28_v13_v1_13_model_visible_projection_v9_4 as projection_v94


def fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    world_uid = "world_a"
    sellers = [f"seller_{index:02d}" for index in range(28)]
    items: list[dict[str, object]] = []
    for seller_index, seller_uid in enumerate(sellers):
        count = 2 + seller_index % 7
        for ordinal in range(count):
            items.append(
                {
                    "world_uid": world_uid,
                    "seller_uid": seller_uid,
                    "logical_item_ordinal": ordinal,
                    "title_nonempty": (seller_index + ordinal) % 3 != 0,
                    "description_nonempty": (seller_index + ordinal) % 4 != 0,
                    "time_bucket": ordinal % 4,
                }
            )
    endpoints = [
        {
            "world_uid": world_uid,
            "canonical_pair_uid": "||".join((left, right)),
            "seller_uid_left": left,
            "seller_uid_right": right,
        }
        for left, right in itertools.combinations(sellers, 2)
    ]
    return endpoints, items


class ModelVisibleProjectionV94Contracts(unittest.TestCase):
    def test_projection_builds_exact_read_only_compatible_rows(self) -> None:
        endpoints, items = fixture()
        rows = projection_v94.build_world_projection(
            endpoint_rows=endpoints, item_rows=items
        )
        self.assertEqual(len(rows), 378)
        self.assertEqual(tuple(rows[0]), matrix_v94.ROW_FIELDS)
        frozen = matrix_v94.freeze_matrix(rows, expected_row_count=378)
        self.assertEqual(frozen.values.shape, (378, 14))
        self.assertFalse(frozen.values.flags.writeable)
        with self.assertRaises(TypeError):
            rows[0]["absdiff__item_count"] = "9.000000000000"
        first = rows[0]
        self.assertEqual(first["canonical_pair_uid"], "seller_00||seller_01")
        self.assertEqual(first["absdiff__item_count"], "1.000000000000")
        self.assertEqual(first["sum__item_count"], "5.000000000000")
        self.assertEqual(first["absdiff__title_missing_rate"], "0.166666666667")
        self.assertEqual(first["sum__title_missing_rate"], "0.833333333333")
        self.assertEqual(
            first["absdiff__description_missing_rate"], "0.500000000000"
        )
        self.assertEqual(
            first["sum__description_missing_rate"], "0.500000000000"
        )
        reversed_rows = projection_v94.build_world_projection(
            endpoint_rows=list(reversed(endpoints)), item_rows=list(reversed(items))
        )
        self.assertEqual([dict(row) for row in rows], [dict(row) for row in reversed_rows])

    def test_projection_module_imports_no_truth_or_plan_module(self) -> None:
        source = Path(projection_v94.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        project_imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                project_imports.update(
                    alias.name for alias in node.names if alias.name.startswith("step28")
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("step28")
            ):
                project_imports.add(node.module)
        self.assertEqual(
            project_imports,
            {"step28_v13_v1_13_model_visible_matrix_v9_4"},
        )

    def test_full_private_ast_or_extra_endpoint_field_is_rejected(self) -> None:
        endpoints, items = fixture()
        private_item = {**items[0], "controller_uid": "private"}
        with self.assertRaisesRegex(
            projection_v94.ModelVisibleProjectionV94Error, "schema/order"
        ):
            projection_v94.build_world_projection(
                endpoint_rows=endpoints, item_rows=[private_item, *items[1:]]
            )
        private_endpoint = {**endpoints[0], "registered_treatment": 1}
        with self.assertRaisesRegex(
            projection_v94.ModelVisibleProjectionV94Error, "schema/order"
        ):
            projection_v94.build_world_projection(
                endpoint_rows=[private_endpoint, *endpoints[1:]], item_rows=items
            )

    def test_missing_or_duplicate_item_ordinal_is_rejected(self) -> None:
        endpoints, items = fixture()
        missing = items[1:]
        with self.assertRaisesRegex(
            projection_v94.ModelVisibleProjectionV94Error, "ordinal closure"
        ):
            projection_v94.build_world_projection(
                endpoint_rows=endpoints, item_rows=missing
            )
        duplicate = [dict(row) for row in items]
        duplicate[1]["logical_item_ordinal"] = duplicate[0]["logical_item_ordinal"]
        with self.assertRaisesRegex(
            projection_v94.ModelVisibleProjectionV94Error, "value drift"
        ):
            projection_v94.build_world_projection(
                endpoint_rows=endpoints, item_rows=duplicate
            )

    def test_incomplete_or_duplicate_pair_graph_is_rejected(self) -> None:
        endpoints, items = fixture()
        with self.assertRaisesRegex(
            projection_v94.ModelVisibleProjectionV94Error, "row count"
        ):
            projection_v94.build_world_projection(
                endpoint_rows=endpoints[:-1], item_rows=items
            )
        duplicate = [dict(row) for row in endpoints]
        duplicate[-1]["seller_uid_left"] = duplicate[0]["seller_uid_left"]
        duplicate[-1]["seller_uid_right"] = duplicate[0]["seller_uid_right"]
        duplicate[-1]["canonical_pair_uid"] = duplicate[0]["canonical_pair_uid"]
        with self.assertRaisesRegex(
            projection_v94.ModelVisibleProjectionV94Error,
            "key/value",
        ):
            projection_v94.build_world_projection(
                endpoint_rows=duplicate, item_rows=items
            )

    def test_pair_uid_must_match_endpoints(self) -> None:
        endpoints, items = fixture()
        forged = [dict(row) for row in endpoints]
        forged[0]["canonical_pair_uid"] = "forged"
        with self.assertRaisesRegex(
            projection_v94.ModelVisibleProjectionV94Error, "key/value"
        ):
            projection_v94.build_world_projection(
                endpoint_rows=forged, item_rows=items
            )

    def test_nonmapping_rows_are_contract_rejected(self) -> None:
        endpoints, items = fixture()
        with self.assertRaisesRegex(
            projection_v94.ModelVisibleProjectionV94Error, "not a mapping"
        ):
            projection_v94.build_world_projection(
                endpoint_rows=endpoints,
                item_rows=[list(projection_v94.ITEM_FIELDS), *items[1:]],
            )
        with self.assertRaisesRegex(
            projection_v94.ModelVisibleProjectionV94Error, "not a mapping"
        ):
            projection_v94.build_world_projection(
                endpoint_rows=[list(projection_v94.ENDPOINT_FIELDS), *endpoints[1:]],
                item_rows=items,
            )


if __name__ == "__main__":
    unittest.main()
