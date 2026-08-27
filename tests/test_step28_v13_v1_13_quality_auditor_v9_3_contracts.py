from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_method_policy_v9_3 as method_policy
import step28_v13_v1_13_quality_auditor_v9_3 as auditor


class QualityAuditorV93Contracts(unittest.TestCase):
    @staticmethod
    def _minimal_split() -> auditor.SplitData:
        item = {
            "world_uid": "world-row-link",
            "seller_uid": "seller-row-link",
            "item_uid": "item-row-link",
            "title": "普通标题",
            "description": "普通描述",
        }
        profile = {
            "seller_uid": "seller-row-link",
            "category_concat_top": "普通类别",
            "signature_title_concat": "普通标题",
            "title_concat_top": "普通标题",
            "signature_description_concat": "普通描述",
            "description_concat_top": "普通描述",
            "item_count": 1,
            "title_length_stats": {"median": 4},
            "description_length_stats": {"median": 4},
            "style_stats": {
                "digit_ratio_mean": 0.0,
                "punct_ratio_mean": 0.0,
                "repeated_title_share": 0.0,
                "repeated_description_share": 0.0,
                "max_category_share": 1.0,
            },
        }
        identity33 = {
            "canonical_pair_uid": "pair-row-link",
            "world_uid": "world-row-link",
            **{f"feature_{index:02d}": "0" for index in range(33)},
        }
        return auditor.SplitData(
            split="audit_a",
            worlds=[
                {
                    "world_uid": "world-row-link",
                    "split": "audit_a",
                    "split_ordinal": 0,
                    "candidate_index": 0,
                }
            ],
            sellers=[
                {
                    "world_uid": "world-row-link",
                    "seller_uid": "seller-row-link",
                    "market": "market-test",
                }
            ],
            endpoints=[],
            original_items=[deepcopy(item)],
            original_profiles=[deepcopy(profile)],
            deranged_items=[deepcopy(item)],
            deranged_profiles=[deepcopy(profile)],
            identity33=[identity33],
            overrides=[],
            seller_structure=[],
            noise_structure=[],
            world_audits=[],
        )

    def test_private_coordinate_nonintervention_is_computed_from_rows(self) -> None:
        forbidden = ("seller_slot", "noise_slot", "logical_item_ordinal")
        clean = self._minimal_split()
        self.assertTrue(
            auditor._validate_private_coordinate_nonintervention(
                {"audit_a": clean}, forbidden_fields=forbidden
            )
        )
        persisted_order = self._minimal_split()
        for name in (
            "original_items",
            "deranged_items",
            "original_profiles",
            "deranged_profiles",
        ):
            setattr(
                persisted_order,
                name,
                [dict(sorted(row.items())) for row in getattr(persisted_order, name)],
            )
        self.assertTrue(
            auditor._validate_private_coordinate_nonintervention(
                {"audit_a": persisted_order}, forbidden_fields=forbidden
            )
        )

        leaked_field = self._minimal_split()
        leaked_field.original_items[0]["seller_slot"] = 3
        with self.assertRaisesRegex(
            auditor.QualityAuditorV93Error, "Private.coordinate"
        ):
            auditor._validate_private_coordinate_nonintervention(
                {"audit_a": leaked_field}, forbidden_fields=forbidden
            )

        leaked_uid = self._minimal_split()
        leaked_uid.deranged_profiles[0]["title_concat_top"] = (
            "sel_" + "a" * 64
        )
        with self.assertRaisesRegex(auditor.QualityAuditorV93Error, "Private UID"):
            auditor._validate_private_coordinate_nonintervention(
                {"audit_a": leaked_uid}, forbidden_fields=forbidden
            )

        for wrapped_uid in (
            "prefix_sel_" + "b" * 64,
            "sel_" + "c" * 64 + "_suffix",
        ):
            leaked_wrapped_uid = self._minimal_split()
            leaked_wrapped_uid.original_profiles[0]["title_concat_top"] = wrapped_uid
            with self.assertRaisesRegex(
                auditor.QualityAuditorV93Error, "Private UID"
            ):
                auditor._validate_private_coordinate_nonintervention(
                    {"audit_a": leaked_wrapped_uid}, forbidden_fields=forbidden
                )

        profile_schema_drift = self._minimal_split()
        profile_schema_drift.original_profiles[0]["world_uid"] = "world-row-link"
        profile_schema_drift.deranged_profiles[0]["world_uid"] = "world-row-link"
        with self.assertRaisesRegex(
            auditor.QualityAuditorV93Error, "model schema.*alignment"
        ):
            auditor._validate_private_coordinate_nonintervention(
                {"audit_a": profile_schema_drift}, forbidden_fields=forbidden
            )

        leaked_identity = self._minimal_split()
        leaked_identity.identity33[0]["noise_slot"] = "4"
        with self.assertRaisesRegex(
            auditor.QualityAuditorV93Error, "Private.coordinate"
        ):
            auditor._validate_private_coordinate_nonintervention(
                {"audit_a": leaked_identity}, forbidden_fields=forbidden
            )

    def test_prebuild_gate_uses_remove_field_self_hash_convention(self) -> None:
        payload = {"version": "gate", "scientific_pass": True}
        expected = auditor.common.canonical_sha256(payload)
        persisted = {**payload, "canonical_self_sha256": expected}
        self.assertEqual(auditor._prebuild_gate_self_hash(persisted), expected)
        self.assertNotEqual(auditor._self_hash(persisted), expected)

    def test_audit_truth_loader_is_structurally_unreachable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(auditor.QualityAuditorV93Error):
                auditor._verify_and_load_controller_membership(
                    root=Path(temporary), records={}, split="audit_a"
                )
            with self.assertRaises(auditor.QualityAuditorV93Error):
                auditor._verify_and_load_controller_membership(
                    root=Path(temporary), records={}, split="audit_b"
                )

    def test_registered_observation_composition_is_exact(self) -> None:
        policy = method_policy.load_policy()
        registry = {
            row["observation_id"]: row for row in policy["observation_registry"]
        }
        model_results = {
            f"{view}::{model}": {
                "symmetric_roc_auc": 0.5,
                "average_precision": 20 / 378,
            }
            for view in ("seller_slot", "noise_visible")
            for model in method_policy.PROBE_MODELS
        }
        result = {
            "single_feature_maximum_symmetric_roc_auc_by_view": {
                "seller_slot": 0.5,
                "noise_visible": 0.5,
            },
            "model_results": model_results,
            "maximum_symmetric_roc_auc": 0.5,
            "maximum_average_precision_uplift": 0.0,
            "bootstrap": {
                "symmetric_auc_95_upper": 0.5,
                "average_precision_uplift_95_upper": 0.0,
            },
        }
        observations = auditor._family_observations(
            registry=registry,
            prefix="structure",
            result=result,
            baseline=20 / 378,
            gates=policy["quality_gates"],
            hard_gate=True,
        )
        self.assertEqual(len(observations), 14)
        self.assertTrue(all(row["passed"] is True for row in observations))
        self.assertEqual(
            len({row["observation_id"] for row in observations}), 14
        )

    def test_quality_output_path_is_single_use_and_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "evidence.json"
            with self.assertRaises(auditor.QualityAuditorV93Error):
                auditor.run(root=Path(temporary), output=wrong)


if __name__ == "__main__":
    unittest.main()
