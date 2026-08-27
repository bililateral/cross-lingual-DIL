from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common
import step28_v13_structure as structure
import step28_v13_text_renderer as renderer
import step28_v13_v1_13_balanced_schedule_v9_3 as balanced
import step28_v13_v1_13_method_world_v9_3 as method_world
import step28_v13_v1_13_quality_auditor_v9_3 as quality_auditor
import step28_v13_v1_13_world_builder_v9_3 as builder


SCHEDULE_PATH = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_balanced_schedule_v9_3"
    / "design_preflight_v2_20260825"
    / "train_balanced_schedule.json"
)
SIGNATURE_PATH = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_balanced_schedule_v9_3"
    / "joint_noise_signature_preflight_v2_20260826.json"
)
POLICY_PATH = ROOT / "schema" / "step28_v13_synthetic_chinese_dataset_policy.json"
BLIND_AUDIT_PATH = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_balanced_schedule_v9_3"
    / "blind_audit_design_preflight_v1_20260826.json"
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return value


class CodeFreeWorldBuilderContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule = read_json(SCHEDULE_PATH)
        cls.signatures = read_json(SIGNATURE_PATH)
        cls.policy = read_json(POLICY_PATH)
        cls.template = read_json(
            ROOT / str(cls.policy["template_library"]["path"])
        )
        cls.fixture = read_json(
            ROOT
            / str(
                cls.policy["identity_design"]
                ["role_template_parser_flag_fixture"]["path"]
            )
        )
        cls.style_profile = read_json(
            ROOT
            / str(
                cls.policy["style_reference_boundary"]
                ["generator_release_inputs"]["profile"]["path"]
            )
        )
        cls.blind_audit = read_json(BLIND_AUDIT_PATH)
        cls.mode = "development_smoke"
        balanced.validate_schedule(cls.schedule)

    @staticmethod
    def base_membership(prefix: str) -> dict:
        return {
            "seller_uids": [f"{prefix}_seller_{index:02d}" for index in range(28)],
            "controller_uids": [
                f"{prefix}_controller_{index:02d}" for index in range(12)
            ],
        }

    @staticmethod
    def styles(membership: dict) -> dict[str, dict]:
        style = {
            "effective_style_uid": "estyle_test",
            "separator": "，",
            "ending": "。",
            "line_mode": "single",
            "english_tag": "PRO",
            "traditional_variant": False,
            "repeat_punctuation": False,
        }
        return {
            seller_uid: dict(style) for seller_uid in membership["seller_uids"]
        }

    def planned_items(self, prefix: str) -> tuple[dict, dict[str, list[dict]]]:
        membership, noise = builder._planned_membership(
            base=self.base_membership(prefix),
            schedule_world=self.schedule["worlds"][0],
            world_ordinal=0,
        )
        items = builder._planned_items(
            policy=self.policy,
            template=self.template,
            mode=self.mode,
            split="train",
            world_uid="world_private_coordinate_test",
            world_ordinal=0,
            candidate_index=0,
            membership=membership,
            noise_by_seller=noise,
            joint_signatures=self.signatures,
            effective_styles=self.styles(membership),
        )
        return membership, items

    def test_schedule_materializes_exact_planned_partition(self) -> None:
        membership, noise = builder._planned_membership(
            base=self.base_membership("a"),
            schedule_world=self.schedule["worlds"][0],
            world_ordinal=0,
        )
        groups, expected_noise = balanced._validate_world(
            self.schedule["worlds"][0], expected_ordinal=0
        )
        sellers = common.utf8_sort(membership["seller_uids"])
        observed_groups = [
            tuple(sellers.index(value) for value in membership["controller_members"][uid])
            for uid in common.utf8_sort(membership["controller_uids"])
        ]
        self.assertEqual(tuple(observed_groups), groups)
        self.assertEqual(noise, expected_noise)

    def test_render_selector_uses_only_logical_noise_coordinates(self) -> None:
        expected = builder._logical_render_selector(
            split="train",
            world_ordinal=7,
            noise_slot=13,
            logical_item_ordinal=2,
        )
        self.assertEqual(
            expected,
            builder._logical_render_selector(
                split="train",
                world_ordinal=7,
                noise_slot=13,
                logical_item_ordinal=2,
            ),
        )
        self.assertNotIn("seller", expected)
        with self.assertRaises(common.ContractError):
            builder._logical_render_selector(
                split="train",
                world_ordinal=7,
                noise_slot=28,
                logical_item_ordinal=2,
            )

    def test_identity_clause_order_ignores_private_slot_uids(self) -> None:
        rows = [
            {
                "global_asset_index": 9,
                "identity_type": "qq",
                "role": "direct_or_private",
                "identity_value": "value-b",
                "slot_uid": "private-a",
            },
            {
                "global_asset_index": 3,
                "identity_type": "email",
                "role": "public_support",
                "identity_value": "value-a",
                "slot_uid": "private-z",
            },
        ]
        relabelled = [
            {**rows[1], "slot_uid": "private-0"},
            {**rows[0], "slot_uid": "private-9"},
        ]
        logical = lambda values: [
            (
                row["global_asset_index"],
                row["identity_type"],
                row["role"],
                row["identity_value"],
            )
            for row in builder._logical_identity_slot_order(values)
        ]
        self.assertEqual(logical(rows), logical(relabelled))
        with self.assertRaises(common.ContractError):
            builder._logical_identity_slot_order([rows[0], dict(rows[0])])

    def test_joint_noise_signatures_define_all_item_counts_and_masks(self) -> None:
        _membership, items = self.planned_items("a")
        rows = [row for seller_rows in items.values() for row in seller_rows]
        self.assertEqual(len(rows), 99)
        self.assertTrue(all("code" not in row for row in rows))
        self.assertFalse(
            any(
                builder.ARTIFICIAL_CODE_PATTERN.search(
                    str(row["title"]) + str(row["base_description"])
                )
                for row in rows
            )
        )
        signature_by_slot = {
            int(row["noise_slot"]): row["signature"]
            for row in self.signatures["noise_slot_multiset"]
        }
        for seller_rows in items.values():
            signature = signature_by_slot[int(seller_rows[0]["noise_slot"])]
            self.assertEqual(len(seller_rows), int(signature["item_count"]))
            self.assertEqual(
                "".join("1" if row["title_nonempty"] else "0" for row in seller_rows),
                signature["title_present_mask"],
            )
            self.assertEqual(
                "".join(
                    "1" if row["description_nonempty"] else "0"
                    for row in seller_rows
                ),
                signature["description_present_mask"],
            )

    def test_private_uid_relabel_does_not_change_feature_bearing_fields(self) -> None:
        _membership_a, items_a = self.planned_items("a")
        _membership_b, items_b = self.planned_items("b")

        def projection(items: dict[str, list[dict]]) -> list[dict]:
            rows = [row for seller_rows in items.values() for row in seller_rows]
            return [
                {
                    key: row[key]
                    for key in (
                        "seller_slot",
                        "noise_slot",
                        "logical_item_ordinal",
                        "time_bucket",
                        "category",
                        "product",
                        "attribute",
                        "delivery",
                        "service",
                        "title_skeleton_index",
                        "description_skeleton_index",
                        "natural_variation_ordinal",
                        "natural_variation_phrase",
                        "title_nonempty",
                        "description_nonempty",
                        "title",
                        "base_description",
                    )
                }
                for row in sorted(
                    rows,
                    key=lambda value: (
                        int(value["seller_slot"]),
                        int(value["logical_item_ordinal"]),
                    ),
                )
            ]

        self.assertEqual(projection(items_a), projection(items_b))
        item_uids_a = {
            row["item_uid"] for seller_rows in items_a.values() for row in seller_rows
        }
        item_uids_b = {
            row["item_uid"] for seller_rows in items_b.values() for row in seller_rows
        }
        self.assertTrue(item_uids_a.isdisjoint(item_uids_b))

    def test_natural_variation_capacity_is_injective_for_frozen_domain(self) -> None:
        required = 4 * 500 * 28 * 8 * 32
        self.assertEqual(required, 14_336_000)
        self.assertGreaterEqual(builder.NATURAL_VARIATION_CAPACITY, required)
        self.assertTrue(
            all(
                len(values) == len(set(values)) == 64
                for values in builder.NATURAL_VARIATION_SEGMENTS
            )
        )
        coordinates = (
            ("train", 0, 0, 0, 0),
            ("train", 499, 27, 7, 31),
            ("development", 0, 0, 0, 0),
            ("audit_a", 317, 13, 4, 17),
            ("audit_b", 499, 27, 7, 31),
        )
        observed = {
            builder._natural_variation(
                split=split,
                world_ordinal=world,
                noise_slot=noise,
                logical_item_ordinal=item,
                candidate_index=candidate,
            )
            for split, world, noise, item, candidate in coordinates
        }
        self.assertEqual(len(observed), len(coordinates))
        maximum_ordinal, _phrase = builder._natural_variation(
            split="audit_b",
            world_ordinal=499,
            noise_slot=27,
            logical_item_ordinal=7,
            candidate_index=31,
        )
        self.assertEqual(maximum_ordinal, required - 1)

    def test_natural_segments_remain_distinct_under_every_reachable_style(self) -> None:
        styles = renderer.reachable_effective_styles(self.template)
        self.assertEqual(len(styles), 176)
        for dimension, values in enumerate(builder.NATURAL_VARIATION_SEGMENTS):
            owners: dict[str, set[int]] = {}
            for index, value in enumerate(values):
                for style in styles:
                    surface = renderer._transform_base(
                        value,
                        style=style,
                        template=self.template,
                        description=False,
                    )
                    owners.setdefault(surface, set()).add(index)
            with self.subTest(dimension=dimension):
                self.assertFalse(
                    any(len(source_indexes) != 1 for source_indexes in owners.values())
                )

    def test_private_uid_relabel_does_not_change_logical_nuisance_assignments(
        self,
    ) -> None:
        membership_a, _noise_a = builder._planned_membership(
            base=self.base_membership("a"),
            schedule_world=self.schedule["worlds"][0],
            world_ordinal=0,
        )
        membership_b, _noise_b = builder._planned_membership(
            base=self.base_membership("b"),
            schedule_world=self.schedule["worlds"][0],
            world_ordinal=0,
        )
        structure_key = common.structure_key_for_split(
            self.policy, mode=self.mode, split="train"
        )
        graph_name = str(
            self.policy["identity_design"]["mechanism_by_split"]["train"]
        )

        def assignments(membership: dict) -> tuple:
            return builder._logical_nonidentity_assignments(
                policy=self.policy,
                template=self.template,
                mode=self.mode,
                split="train",
                world_ordinal=0,
                graph_name=graph_name,
                structure_key_hex=structure_key,
                membership=membership,
            )

        observed_a = assignments(membership_a)
        observed_b = assignments(membership_b)
        self.assertEqual(observed_a[1], observed_b[1])

        def seller_projection(values: dict, membership: dict) -> list:
            return [values[uid] for uid in common.utf8_sort(membership["seller_uids"])]

        def controller_projection(values: dict, membership: dict) -> list:
            return [
                values[uid]
                for uid in common.utf8_sort(membership["controller_uids"])
            ]

        self.assertEqual(
            seller_projection(observed_a[0], membership_a),
            seller_projection(observed_b[0], membership_b),
        )
        self.assertEqual(
            controller_projection(observed_a[2], membership_a),
            controller_projection(observed_b[2], membership_b),
        )
        self.assertEqual(
            controller_projection(observed_a[3], membership_a),
            controller_projection(observed_b[3], membership_b),
        )
        self.assertEqual(
            seller_projection(observed_a[4], membership_a),
            seller_projection(observed_b[4], membership_b),
        )

    def test_identity_solver_private_links_are_logically_nonintervening(self) -> None:
        graph_name = str(
            self.policy["identity_design"]["mechanism_by_split"]["train"]
        )
        structure_key = common.structure_key_for_split(
            self.policy, mode=self.mode, split="train"
        )

        def solve(prefix: str) -> tuple[dict, dict, dict]:
            membership, items = self.planned_items(prefix)
            nuisance = builder._logical_nonidentity_assignments(
                policy=self.policy,
                template=self.template,
                mode=self.mode,
                split="train",
                world_ordinal=0,
                graph_name=graph_name,
                structure_key_hex=structure_key,
                membership=membership,
            )
            expected_noise_count = sum(
                any(row["description_nonempty"] for row in seller_rows)
                for seller_rows in items.values()
            )

            def callback(_flags: object) -> dict:
                return {
                    "override_audit_count": 6,
                    "high_semantic_count": 4,
                    "exact_title_clone_count": 2,
                    "unique_override_seller_count": 12,
                    "unique_override_item_count": 12,
                    "noise_record_count": expected_noise_count,
                    "override_audit_sha256": "0" * 64,
                    "noise_record_sha256": "1" * 64,
                }

            solved = builder._solve_identity_plan_logically(
                policy=self.policy,
                mode=self.mode,
                split="train",
                world_uid=f"{prefix}_private_world",
                world_ordinal=0,
                world_mode_global_ordinal=0,
                structure_key_hex=structure_key,
                membership=membership,
                markets=nuisance[0],
                mechanisms=nuisance[2],
                items_by_seller=items,
                pre_slot_callback=callback,
            )
            return membership, items, solved

        membership_a, items_a, solved_a = solve("a")
        membership_b, items_b, solved_b = solve("b")

        def slot_projection(membership: dict, items: dict) -> list:
            output = []
            for seller_slot, seller_uid in enumerate(
                common.utf8_sort(membership["seller_uids"])
            ):
                for item in sorted(
                    items[seller_uid],
                    key=lambda row: int(row["logical_item_ordinal"]),
                ):
                    output.append(
                        {
                            "seller_slot": seller_slot,
                            "logical_item_ordinal": item["logical_item_ordinal"],
                            "slots": [
                                {
                                    key: slot[key]
                                    for key in (
                                        "identity_type",
                                        "identity_value",
                                        "role",
                                        "global_asset_index",
                                    )
                                }
                                for slot in sorted(
                                    item["identity_slots"],
                                    key=lambda row: row["slot_uid"].encode("utf-8"),
                                )
                            ],
                        }
                    )
            return output

        self.assertEqual(
            slot_projection(membership_a, items_a),
            slot_projection(membership_b, items_b),
        )
        for prefix, solved in (("a", solved_a), ("b", solved_b)):
            self.assertFalse(
                any(
                    str(row.get("seller_uid", "")).startswith("logical_")
                    or str(row.get("item_uid", "")).startswith("logical_")
                    for row in solved["slots"]
                )
            )
            self.assertTrue(
                all(
                    str(row["seller_uid_left"]).startswith(prefix + "_")
                    and str(row["seller_uid_right"]).startswith(prefix + "_")
                    for row in solved["positive_targets"]
                )
            )

    def test_blind_audit_world_closes_end_to_end_without_truth(self) -> None:
        world_record = next(
            row
            for row in structure.build_mode_world_pool(
                self.policy, mode=self.mode
            )
            if row["split"] == "audit_a" and row["split_ordinal"] == 0
        )
        built = method_world.build_method_world(
            policy=self.policy,
            template=self.template,
            fixture=self.fixture,
            style_profile=self.style_profile,
            mode=self.mode,
            world_record=world_record,
            structure_key_hex=common.structure_key_for_split(
                self.policy, mode=self.mode, split="audit_a"
            ),
            balanced_schedule=None,
            registered_negative_plan=None,
            joint_signatures=self.signatures,
            blind_audit_design=self.blind_audit,
            candidate_index=0,
        )
        self.assertEqual(len(built["public"]["sellers"]), 28)
        self.assertEqual(len(built["public"]["complete_pair_endpoints"]), 378)
        self.assertEqual(len(built["public"]["identity33"]), 378)
        self.assertEqual(
            len(built["public"]["seller_slot_structure_rows"]), 378
        )
        self.assertEqual(
            len(built["public"]["noise_visible_structure_rows"]), 378
        )
        self.assertFalse(built["audit"]["truth_materialized"])
        self.assertEqual(built["audit"]["artificial_code_occurrence_count"], 0)
        intervention = built["audit"]["counterfactual_intervention"]
        self.assertEqual(intervention["style_changed_seller_count"], 28)
        self.assertTrue(
            all(
                count > 0
                for count in intervention[
                    "style_factor_changed_seller_counts"
                ].values()
            )
        )
        self.assertTrue(
            built["audit"]["noise_time_counterfactual_identity33_unchanged"]
        )
        label_free = quality_auditor.SplitData(
            split="audit_a",
            worlds=[
                {
                    "world_uid": built["world_uid"],
                    "split": "audit_a",
                    "split_ordinal": 0,
                    "candidate_index": 0,
                }
            ],
            sellers=built["public"]["sellers"],
            endpoints=built["public"]["complete_pair_endpoints"],
            original_items=built["public"]["original_redacted_items"],
            original_profiles=built["public"]["original_model_seller_profiles"],
            deranged_items=built["public"]["deranged_redacted_items"],
            deranged_profiles=built["public"]["deranged_model_seller_profiles"],
            identity33=built["public"]["identity33"],
            overrides=built["private_without_truth"]["override_audit"],
            seller_structure=built["public"]["seller_slot_structure_rows"],
            noise_structure=built["public"]["noise_visible_structure_rows"],
            world_audits=[{"world_uid": built["world_uid"], **built["audit"]}],
        )
        self.assertTrue(
            all(
                quality_auditor._validate_label_free_surfaces(
                    {"audit_a": label_free}
                ).values()
            )
        )
        structure_views = quality_auditor._freeze_structure(label_free)
        self.assertEqual(structure_views["seller_slot"].values.shape, (378, 382))
        self.assertEqual(structure_views["noise_visible"].values.shape[0], 378)
        for surface in ("style_deranged", "original_author"):
            text_matrices = quality_auditor._freeze_text(
                label_free, surface=surface
            )
            self.assertEqual(tuple(text_matrices), tuple(quality_auditor.text_views.VIEW_ORDER))
            self.assertTrue(
                all(matrix.values.shape[0] == 372 for matrix in text_matrices.values())
            )


if __name__ == "__main__":
    unittest.main()
