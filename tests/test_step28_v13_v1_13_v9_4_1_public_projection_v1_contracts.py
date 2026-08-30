from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as common
import step28_v13_v1_13_v9_4_1_prepare_public_projection_v1 as projection
import step7_v3_1_source_data as source


class PublicProjectionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = common.load_policy()

    def test_public_role_allowlist_excludes_all_supervision(self) -> None:
        combined = set(projection.BASE_PUBLIC_ROLES) | set(
            projection.IDENTITY_PUBLIC_ROLES
        )
        self.assertEqual(
            combined, set(self.policy["public_projection"]["allowed_observed_files"])
        )
        self.assertNotIn("identity33_all_pairs.csv", projection.BASE_PUBLIC_ROLES)
        self.assertNotIn("redacted_items.jsonl", projection.IDENTITY_PUBLIC_ROLES)
        self.assertNotIn("model_seller_profiles.jsonl", projection.IDENTITY_PUBLIC_ROLES)
        joined = " ".join(combined)
        for forbidden in ("label", "qrel", "controller", "membership", "audit"):
            self.assertNotIn(forbidden, joined.casefold())
        with self.assertRaisesRegex(common.ModelExperimentContractError, "view"):
            projection.verify_split_public_inputs(
                self.policy,
                "train",
                ("redacted_items.jsonl", "identity33_all_pairs.csv"),
            )

    def test_model_profile_projection_uses_frozen_fields_without_redaction(self) -> None:
        row = {
            "seller_uid": "seller_01",
            "category_concat_top": "类别甲 || 类别乙",
            "signature_title_concat": "标题甲 || 标题乙",
            "title_concat_top": "标题乙 || 标题丙",
            "signature_description_concat": "描述甲",
            "description_concat_top": "描述乙 || 描述甲",
            "item_count": 3,
            "title_length_stats": {"median": 10},
            "description_length_stats": {"median": 20},
            "style_stats": {
                "digit_ratio_mean": 0.1,
                "punct_ratio_mean": 0.2,
                "repeated_title_share": 0.3,
                "repeated_description_share": 0.4,
                "max_category_share": 0.5,
            },
        }
        projected = projection.project_model_profile(row)
        self.assertEqual(projected["clean_categories"], ["类别乙", "类别甲"])
        self.assertEqual(set(projected["clean_titles"]), {"标题甲", "标题乙", "标题丙"})
        self.assertEqual(set(projected["clean_descriptions"]), {"描述甲", "描述乙"})
        self.assertEqual(projected["numeric_profile"]["item_count"], 3.0)
        self.assertNotIn("model_text", projected)

    def test_legacy18_formula_matches_frozen_step7_implementation(self) -> None:
        sellers = {
            "left": {
                "seller_uid": "left",
                "clean_categories": ["a", "b"],
                "clean_titles": ["same", "left"],
                "clean_descriptions": ["desc"],
                "source_dataset": "",
                "source_market_raw": "",
                "numeric_profile": {
                    name: float(index + 1)
                    for index, name in enumerate(source.NUMERIC_PROFILE_FIELDS)
                },
            },
            "right": {
                "seller_uid": "right",
                "clean_categories": ["b", "c"],
                "clean_titles": ["same", "right"],
                "clean_descriptions": ["other"],
                "source_dataset": "",
                "source_market_raw": "",
                "numeric_profile": {
                    name: float(index + 2)
                    for index, name in enumerate(source.NUMERIC_PROFILE_FIELDS)
                },
            },
        }
        reference = {
            "train_seller_count": 4,
            "title_df": {"same": 2},
            "description_df": {},
            "numeric_references": {
                name: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
                for name in source.NUMERIC_PROFILE_FIELDS
            },
        }
        pair = {
            "pair_uid": "p",
            "seller_uid_left": "left",
            "seller_uid_right": "right",
        }
        expected_row = source.build_safe_pair_rows([pair], sellers, reference)[0]
        expected = np.asarray(
            [expected_row[name] for name in source.MODEL_ELIGIBLE_TRANSFER_FEATURE_NAMES],
            dtype=np.float64,
        )
        observed = projection.legacy18_row(
            pair,
            sellers,
            reference,
            self.policy["feature_contract"]["legacy18"],
        )
        np.testing.assert_array_equal(observed, expected)

    def test_identity33_header_hash_is_pinned_not_inferred_from_labels(self) -> None:
        paths = projection.verify_split_public_inputs(
            self.policy, "train", projection.IDENTITY_PUBLIC_ROLES
        )
        with paths["identity33_all_pairs.csv"].open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            import csv

            header = next(csv.reader(handle))[2:]
        self.assertEqual(len(header), 33)
        self.assertEqual(
            common.canonical_sha256(header),
            self.policy["feature_contract"]["column_name_hashes"]["identity33"],
        )


if __name__ == "__main__":
    unittest.main()
