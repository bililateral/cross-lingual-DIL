from __future__ import annotations

import copy
import itertools
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as legacy_common
import step28_v13_text_renderer as renderer
import step28_v13_v1_12_preceremony as subject


def _uid(prefix: str, value: int) -> str:
    return f"{prefix}_{value:064x}"


def _pair_rows(world_uid: str) -> list[dict[str, str]]:
    sellers = [_uid("sel", value) for value in range(1, 29)]
    return [
        {
            "canonical_pair_uid": legacy_common.canonical_pair_uid(left, right),
            "world_uid": world_uid,
            "seller_uid_left": left,
            "seller_uid_right": right,
        }
        for left, right in itertools.combinations(sellers, 2)
    ]


class Step28V13V112PreceremonyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validated = subject.validate_policy()

    def test_policy_remains_non_authorizing_and_archive_is_complete(self) -> None:
        policy = self.validated["policy"]
        self.assertEqual(set(policy["authorizations"].values()), {False})
        self.assertEqual(len(self.validated["failed_identity_hashes"]), 915996)
        self.assertEqual(
            len(self.validated["forbidden_master_commitments"]), 90
        )
        self.assertFalse(
            policy["next_lock_requirements"][
                "formal_seed_generation_may_be_authorized_by_this_policy"
            ]
        )

    def test_world_scoped_mechanism_slot_allows_cross_world_reuse(self) -> None:
        rows = []
        for world_number in (1, 2):
            for slot in ("slot_a", "slot_b"):
                rows.append(
                    {
                        "world_uid": _uid("w", world_number),
                        "controller_uid": f"ctl_{world_number}_{slot}",
                        "mechanism": "shared" if slot == "slot_a" else "rotated",
                        "mechanism_slot_uid": slot,
                    }
                )
        audit = subject.validate_world_scoped_mechanism_slots(
            rows, expected_world_count=2, expected_rows_per_world=2
        )
        self.assertEqual(audit["world_scoped_unique_key_count"], 4)
        self.assertEqual(audit["global_template_slot_count"], 2)
        self.assertEqual(
            audit["expected_cross_world_template_reuse_row_count"], 2
        )

    def test_world_scoped_mechanism_slot_rejects_within_world_duplicate(self) -> None:
        row = {
            "world_uid": _uid("w", 1),
            "controller_uid": "ctl_a",
            "mechanism": "shared",
            "mechanism_slot_uid": "slot_a",
        }
        with self.assertRaises(subject.PreceremonyError):
            subject.validate_world_scoped_mechanism_slots(
                [row, {**row, "controller_uid": "ctl_b"}],
                expected_world_count=1,
                expected_rows_per_world=2,
            )

    def test_long_path_and_manifest_share_one_version_source(self) -> None:
        audit = subject.run_windows_long_path_replay()
        self.assertEqual(
            audit["status"], "PASS_LONG_PATH_SINGLE_IMPLEMENTATION_REPLAY"
        )
        receipt = subject.build_split_manifest_receipt(
            split="train", file_records=[]
        )
        subject.validate_split_manifest_receipt(receipt)
        stale = subject.with_canonical_self_hash(
            {
                **{key: value for key, value in receipt.items() if key != "canonical_self_hash"},
                "version": "stale-producer-version",
            }
        )
        with self.assertRaises(subject.PreceremonyError):
            subject.validate_split_manifest_receipt(stale)

    def test_per_asset_collision_resolution_does_not_require_global_salt(self) -> None:
        a = "0" * 63 + "1"
        b = "0" * 63 + "2"
        c = "0" * 63 + "3"
        d = "0" * 63 + "4"
        allocated = {b}
        counter, selected = subject.select_first_admissible_per_asset_candidate(
            [a, b, c],
            historical_forbidden=frozenset({a}),
            allocated_in_current_run=allocated,
        )
        self.assertEqual((counter, selected), (2, c))
        counter2, selected2 = subject.select_first_admissible_per_asset_candidate(
            [c, d],
            historical_forbidden=frozenset({a}),
            allocated_in_current_run=allocated,
        )
        self.assertEqual((counter2, selected2), (1, d))

    def test_identity_candidate_rejects_visible_text_collision(self) -> None:
        value = "tg0123456789abcd"
        self.assertTrue(
            subject.identity_value_collides_with_visible_text(
                value,
                visible_texts=["普通标题"],
                visible_compacts=["联系tg0123456789abcd获取详情"],
            )
        )
        self.assertFalse(
            subject.identity_value_collides_with_visible_text(
                value,
                visible_texts=["普通标题"],
                visible_compacts=["完全无关内容"],
            )
        )

    def test_self_hash_is_invalid_after_any_field_append(self) -> None:
        document = subject.with_canonical_self_hash({"complete": True})
        subject.validate_canonical_self_hash(document, label="unit receipt")
        mutated = dict(document)
        mutated["post_validation_append"] = True
        with self.assertRaises(subject.PreceremonyError):
            subject.validate_canonical_self_hash(mutated, label="unit receipt")

    def test_optimizer_must_converge_before_ceremony(self) -> None:
        audit = {
            "solver_success": True,
            "convergence_warning_count": 0,
            "iteration_count": 120,
            "maximum_iterations": 1000,
            "normalized_gradient": 1e-9,
            "gradient_tolerance": 1e-8,
            "objective_finite": True,
            "preceremony_exact_configuration": True,
        }
        subject.validate_optimizer_audit(audit)
        warning = {**audit, "convergence_warning_count": 1}
        with self.assertRaises(subject.PreceremonyError):
            subject.validate_optimizer_audit(warning)
        boundary = {**audit, "iteration_count": 1000}
        with self.assertRaises(subject.PreceremonyError):
            subject.validate_optimizer_audit(boundary)

    def test_visible_projection_preserves_identity_free_title(self) -> None:
        template = legacy_common.load_json(
            ROOT / "schema" / "step28_v13_synthetic_text_templates.json"
        )
        guard = renderer.context_guard_pool(template)[0]
        sellers = [
            {"world_uid": _uid("w", 1), "seller_uid": _uid("sel", value)}
            for value in range(1, 29)
        ]
        identity_surface = "tg0123456789abcd"
        title = "普通商品 tg看似身份但不是登记槽"
        item = {
            "world_uid": _uid("w", 1),
            "seller_uid": _uid("sel", 1),
            "item_uid": _uid("itm", 1),
            "time_bucket": 0,
            "category": "其他",
            "title": title,
            "description": f"公开描述{guard}联系 {identity_surface}",
        }
        parsed = [
            {
                "item_uid": _uid("itm", 1),
                "raw_value": identity_surface,
                "normalized_value": identity_surface,
            }
        ]
        projected = subject.project_registered_visible_text(
            policy={},
            template=template,
            sellers=sellers,
            items=[item],
            parsed_rows=parsed,
        )
        self.assertEqual(projected["redacted_items"][0]["title"], title)
        self.assertEqual(
            projected["redacted_items"][0]["description"], "公开描述"
        )

    def test_join_only_uid_substring_is_not_scanned_as_natural_language(self) -> None:
        value = "deadbeef" + "0" * 56
        rows = [
            {
                "world_uid": _uid("w", 1),
                "seller_uid": f"sel_{value}",
                "item_uid": _uid("itm", 1),
            }
        ]
        subject.validate_join_only_uid_lineage(
            rows, fields=("world_uid", "seller_uid", "item_uid")
        )
        with self.assertRaises(subject.PreceremonyError):
            subject.validate_join_only_uid_lineage(
                [{**rows[0], "seller_uid": "sel_not_hex"}],
                fields=("world_uid", "seller_uid", "item_uid"),
            )

    def test_stale_premodel_member_name_is_rejected(self) -> None:
        current = ["near_link_calibration_v2.json", "shortcut_preflight_v2.json"]
        subject.validate_exact_member_contract(current, list(current))
        with self.assertRaises(subject.PreceremonyError):
            subject.validate_exact_member_contract(
                current,
                ["near_link_calibration_v1.json", "shortcut_preflight_v1.json"],
            )

    def test_endpoint_disjoint_m1_is_full_bijection(self) -> None:
        world_uid = _uid("w", 1)
        pairs = _pair_rows(world_uid)
        mapping = subject.build_endpoint_disjoint_derangement(
            pairs, world_uid=world_uid, key_hex="1" * 64
        )
        self.assertEqual(len(mapping), 378)
        self.assertEqual(len({row["source_pair_uid"] for row in mapping}), 378)
        self.assertEqual(
            len({row["destination_pair_uid"] for row in mapping}), 378
        )
        identity_rows = [
            {
                "canonical_pair_uid": row["canonical_pair_uid"],
                "world_uid": world_uid,
                "feature_a": str(index),
                "feature_b": str(index % 7),
            }
            for index, row in enumerate(pairs)
        ]
        rewired = subject.rewire_identity33_rows(identity_rows, mapping)
        self.assertEqual(len(rewired), 378)

    def test_full_pair_label_formula_is_20_positive_358_negative(self) -> None:
        world_uid = _uid("w", 1)
        pairs = _pair_rows(world_uid)
        sellers = [_uid("sel", value) for value in range(1, 29)]
        memberships = []
        cursor = 0
        for controller_number, size in enumerate([2] * 8 + [3] * 4):
            for seller_uid in sellers[cursor : cursor + size]:
                memberships.append(
                    {
                        "world_uid": world_uid,
                        "controller_uid": f"ctl_{controller_number}",
                        "seller_uid": seller_uid,
                    }
                )
            cursor += size
        labels = subject.validate_full_pair_labels(
            pair_rows=pairs,
            controller_membership=memberships,
            expected_world_uid=world_uid,
        )
        self.assertEqual(len(labels), 378)
        self.assertEqual(sum(row["label"] for row in labels), 20)

    def test_two_world_design_preflight_is_not_formal_result(self) -> None:
        receipt = subject.run_design_preflight()
        self.assertEqual(
            receipt["status"], "PASS_DESIGN_ONLY_NO_FORMAL_AUTHORIZATION"
        )
        self.assertEqual(receipt["world_count"], 2)
        self.assertEqual(receipt["pair_count"], 756)
        self.assertEqual(receipt["identity33_row_count"], 756)
        self.assertEqual(receipt["identity_asset_count"], 168)
        self.assertEqual(receipt["forced_identity_collision_count"], 1)
        self.assertEqual(receipt["m1_mapping_count"], 10)
        self.assertFalse(receipt["scientific_metrics_produced"])
        self.assertEqual(
            set(receipt["formal_authorizations_after_preflight"].values()),
            {False},
        )

    def test_policy_mutation_after_hash_fails(self) -> None:
        policy = copy.deepcopy(self.validated["policy"])
        policy["authorizations"]["formal_seed_generation"] = True
        with self.assertRaises(subject.PreceremonyError):
            subject.validate_canonical_self_hash(policy, label="mutated policy")


if __name__ == "__main__":
    unittest.main()
