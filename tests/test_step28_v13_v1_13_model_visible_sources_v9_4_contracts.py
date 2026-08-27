from __future__ import annotations

import ast
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_model_visible_prebuild_source_v9_4 as prebuild_v94
import step28_v13_v1_13_model_visible_public_replay_v9_4 as replay_v94


def fixture():
    sellers = [f"seller_{index:02d}" for index in range(28)]
    world = {
        "split": "train",
        "world_ordinal": 0,
        "world_uid": "world_000",
        "seller_uids": sellers,
        "noise_slot_by_seller_slot": list(range(28)),
    }
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
    return world, signatures


class ModelVisibleSourcesV94Contracts(unittest.TestCase):
    def test_prebuild_and_actual_public_rows_replay_exactly(self) -> None:
        world, signatures = fixture()
        endpoints, items = prebuild_v94.build_truth_free_world_source(
            world=world,
            noise_signatures=signatures,
            time_key_hex="01" * 32,
        )
        registered = prebuild_v94.build_truth_free_world_projection(
            world=world,
            noise_signatures=signatures,
            time_key_hex="01" * 32,
        )
        public_items = [
            {
                "world_uid": row["world_uid"],
                "seller_uid": row["seller_uid"],
                "item_uid": (
                    f"item_{row['seller_uid']}_{row['logical_item_ordinal']:02d}"
                ),
                "logical_item_ordinal": row["logical_item_ordinal"],
                "title": "标题" if row["title_nonempty"] else "",
                "description": "描述" if row["description_nonempty"] else "",
                "time_bucket": row["time_bucket"],
            }
            for row in items
        ]
        replayed = replay_v94.require_exact_replay(
            registered_rows=registered,
            public_endpoint_rows=endpoints,
            public_item_rows=public_items,
        )
        self.assertEqual(replayed.values.shape, (378, 14))
        self.assertEqual(len(endpoints), 378)

    def test_prebuild_rejects_private_or_registered_fields(self) -> None:
        world, signatures = fixture()
        for extra in ("controller_groups", "registered_treatment", "label"):
            with self.subTest(extra=extra):
                forged = {**world, extra: []}
                with self.assertRaisesRegex(
                    prebuild_v94.ModelVisiblePrebuildSourceV94Error,
                    "schema/order",
                ):
                    prebuild_v94.build_truth_free_world_projection(
                        world=forged,
                        noise_signatures=signatures,
                        time_key_hex="01" * 32,
                    )

    def test_signature_masks_and_time_key_are_fail_closed(self) -> None:
        world, signatures = fixture()
        forged = [dict(row) for row in signatures]
        forged[0]["joint_empty_mask"] = "10"
        with self.assertRaisesRegex(
            prebuild_v94.ModelVisiblePrebuildSourceV94Error,
            "value drift",
        ):
            prebuild_v94.build_truth_free_world_projection(
                world=world,
                noise_signatures=forged,
                time_key_hex="01" * 32,
            )
        with self.assertRaisesRegex(
            prebuild_v94.ModelVisiblePrebuildSourceV94Error,
            "Time key",
        ):
            prebuild_v94.build_truth_free_world_projection(
                world=world,
                noise_signatures=signatures,
                time_key_hex="not-a-key",
            )

    def test_signatures_enforce_training_ready_title_and_description_support(self) -> None:
        world, signatures = fixture()
        for field, value in (
            ("title_present_mask", "00"),
            ("description_present_mask", "10"),
        ):
            with self.subTest(field=field):
                forged = [dict(row) for row in signatures]
                forged[0][field] = value
                forged[0]["joint_empty_mask"] = "".join(
                    "1"
                    if title == "0" and description == "0"
                    else "0"
                    for title, description in zip(
                        forged[0]["title_present_mask"],
                        forged[0]["description_present_mask"],
                        strict=True,
                    )
                )
                with self.assertRaisesRegex(
                    prebuild_v94.ModelVisiblePrebuildSourceV94Error,
                    "value drift",
                ):
                    prebuild_v94.build_truth_free_world_projection(
                        world=world,
                        noise_signatures=forged,
                        time_key_hex="01" * 32,
                    )

    def test_public_replay_detects_visible_nuisance_drift(self) -> None:
        world, signatures = fixture()
        endpoints, items = prebuild_v94.build_truth_free_world_source(
            world=world,
            noise_signatures=signatures,
            time_key_hex="01" * 32,
        )
        registered = prebuild_v94.build_truth_free_world_projection(
            world=world,
            noise_signatures=signatures,
            time_key_hex="01" * 32,
        )
        public_items = [
            {
                "world_uid": row["world_uid"],
                "seller_uid": row["seller_uid"],
                "item_uid": (
                    f"item_{row['seller_uid']}_{row['logical_item_ordinal']:02d}"
                ),
                "logical_item_ordinal": row["logical_item_ordinal"],
                "title": "标题",
                "description": "描述",
                "time_bucket": row["time_bucket"],
            }
            for row in items
        ]
        public_items[0]["time_bucket"] = (public_items[0]["time_bucket"] + 1) % 4
        with self.assertRaisesRegex(
            replay_v94.ModelVisiblePublicReplayV94Error,
            "projection drift",
        ):
            replay_v94.require_exact_replay(
                registered_rows=registered,
                public_endpoint_rows=endpoints,
                public_item_rows=public_items,
            )

    def test_public_replay_rejects_missing_or_rewritten_persisted_endpoint(self) -> None:
        world, signatures = fixture()
        endpoints, items = prebuild_v94.build_truth_free_world_source(
            world=world,
            noise_signatures=signatures,
            time_key_hex="01" * 32,
        )
        registered = prebuild_v94.build_truth_free_world_projection(
            world=world,
            noise_signatures=signatures,
            time_key_hex="01" * 32,
        )
        public_items = [
            {
                "world_uid": row["world_uid"],
                "seller_uid": row["seller_uid"],
                "item_uid": f"item_{index:03d}",
                "logical_item_ordinal": row["logical_item_ordinal"],
                "title": "标题" if row["title_nonempty"] else "",
                "description": "描述" if row["description_nonempty"] else "",
                "time_bucket": row["time_bucket"],
            }
            for index, row in enumerate(items)
        ]
        with self.assertRaisesRegex(ValueError, "Endpoint row count"):
            replay_v94.require_exact_replay(
                registered_rows=registered,
                public_endpoint_rows=endpoints[:-1],
                public_item_rows=public_items,
            )
        rewritten = [dict(row) for row in endpoints]
        rewritten[0]["canonical_pair_uid"] = "wrong||pair"
        with self.assertRaisesRegex(ValueError, "Endpoint key/value"):
            replay_v94.require_exact_replay(
                registered_rows=registered,
                public_endpoint_rows=rewritten,
                public_item_rows=public_items,
            )

    def test_project_import_closure_is_minimal(self) -> None:
        expected = {
            prebuild_v94.__file__: {
                "step28_v13_v1_13_model_visible_projection_v9_4"
            },
            replay_v94.__file__: {
                "step28_v13_v1_13_model_visible_matrix_v9_4",
                "step28_v13_v1_13_model_visible_projection_v9_4",
            },
        }
        for source_path, expected_imports in expected.items():
            source = Path(source_path).read_text(encoding="utf-8")
            tree = ast.parse(source)
            observed: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    observed.update(
                        alias.name for alias in node.names
                        if alias.name.startswith("step28")
                    )
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.startswith("step28")
                ):
                    observed.add(node.module)
            self.assertEqual(observed, expected_imports)


if __name__ == "__main__":
    unittest.main()
