from __future__ import annotations

import copy
from pathlib import Path
import hashlib
import json
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_structure_aggregator_v9 as aggregator


def public_row(split: str, index: int) -> dict[str, object]:
    split_symbol = chr(ord("A") + aggregator.SPLITS.index(split))
    code = "Q" + split_symbol + ("A" * 8) + chr(ord("A") + index)
    return {
        "world_uid": f"{split}_world",
        "seller_uid": f"{split}_seller_{index}",
        "owned_codes": [code],
    }


def audit_row(split: str) -> dict[str, object]:
    item_uids = [f"{split}_item_0", f"{split}_item_1"]
    neutral_codes = ["QAAAAAAAABA", "QAAAAAAABBA"]
    neutral_code_sha256 = hashlib.sha256(
        json.dumps(
            neutral_codes,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    neutral_receipt = {
        "version": "fixture-v9",
        "neutral_render_code_ordinal_zero": "QAAAAAAAABA",
        "neutral_code_family_rule": "Q_plus_item_ordinal_as_eight_base16_A_to_P_digits_plus_BA",
        "neutral_code_family_count": 2,
        "neutral_code_family_sha256": neutral_code_sha256,
        "original_code_value_argument_count": 0,
        "original_code_value_read_count": 0,
        "neutral_metadata_source_value_read_count": 10,
        "neutral_metadata_source_value_read_counts": {
            field: 2 for field in aggregator.NEUTRAL_ITEM_METADATA_FIELDS
        },
        "neutralizer_input_capability": "NeutralItemProjection[NeutralItemMetadata]",
        "neutralizer_input_fields": list(aggregator.NEUTRAL_ITEM_METADATA_FIELDS),
        "neutral_profiles_recomputed_after_code_collapse": True,
        "neutral_profile_safe_item_sha256": "2" * 64,
        "clone_count": 0,
        "title_template_mapping": [{"index": index} for index in range(8)],
        "description_template_mapping": [{"index": index} for index in range(8)],
        "per_item_template_mapping": [
            {"item_uid": item_uid} for item_uid in item_uids
        ],
        "non_code_projection_commitment": {
            "verified": True,
            "source_sha256": "3" * 64,
            "neutral_sha256": "3" * 64,
            "ast_row_count": 2,
            "identity_slot_count": 1,
            "noise_slot_count": 1,
            "absolute_offsets_compared": False,
            "relative_ast_boundaries_compared": True,
            "allowed_removed_nodes": list(aggregator.ALLOWED_REMOVED_NODES),
        },
        "non_code_projection_nodes": [
            {"item_uid": item_uid} for item_uid in item_uids
        ],
        "neutral_item_sha256": "c" * 64,
        "neutral_profile_sha256": "f" * 64,
    }
    values: dict[str, object] = {
        "version": "fixture-v9",
        "world_uid": f"{split}_world",
        "item_count": 2,
        "seller_count": 2,
        "registered_code_count": 2,
        "registered_item_occurrence_count": 2,
        "registered_visible_occurrence_expected_count": 4,
        "registered_visible_occurrence_actual_count": 4,
        "clone_directions": [],
        "neutral_receipt": neutral_receipt,
        "full_item_sha256": "a" * 64,
        "masked_item_sha256": "b" * 64,
        "neutral_item_sha256": "c" * 64,
        "full_profile_sha256": "d" * 64,
        "masked_profile_sha256": "e" * 64,
        "neutral_profile_sha256": "f" * 64,
        "forbidden_capability_mounted": {
            name: False for name in aggregator.FORBIDDEN_CAPABILITY_FIELDS
        },
    }
    values.update({field: 0 for field in aggregator.ZERO_TOLERANCE_FIELDS})
    return {field: values[field] for field in aggregator.STRUCTURE_AUDIT_FIELDS}


class QualityStructureAggregatorV9Contracts(unittest.TestCase):
    def setUp(self) -> None:
        self.public = {split: [public_row(split, 0), public_row(split, 1)] for split in aggregator.SPLITS}
        self.audit = {split: [audit_row(split)] for split in aggregator.SPLITS}
        self.counts = {split: 1 for split in aggregator.SPLITS}

    def test_fixture_passes_without_labels_or_formal_claim(self) -> None:
        receipt = aggregator.aggregate_fixture_structure(
            public_rows_by_split=self.public,
            structure_rows_by_split=self.audit,
            expected_world_counts=self.counts,
            expected_sellers_per_world=2,
        )
        self.assertEqual(receipt["gate_failures"], [])
        self.assertIn("NO_DATASET_CONCLUSION", receipt["status"])
        self.assertEqual(receipt["truth_label_row_count_read"], 0)
        self.assertEqual(receipt["zero_tolerance_counts"]["prior_world_code_hits"], 0)
        self.assertEqual(
            receipt["forbidden_read_counts"],
            {
                "audit_truth": {
                    "open_count": 0,
                    "read_count": 0,
                    "materialized_row_count": 0,
                },
                "generator_quality_result": 0,
                "candidate_quality_result": 0,
                "view_builder_quality_result": 0,
            },
        )

    def test_each_zero_tolerance_counter_invalidates_fixture_mechanism(self) -> None:
        for field in aggregator.ZERO_TOLERANCE_FIELDS:
            audit = copy.deepcopy(self.audit)
            audit["train"][0][field] = 1
            with self.subTest(field=field):
                receipt = aggregator.aggregate_fixture_structure(
                    public_rows_by_split=self.public,
                    structure_rows_by_split=audit,
                    expected_world_counts=self.counts,
                    expected_sellers_per_world=2,
                )
                self.assertIn(field, receipt["gate_failures"])
                self.assertEqual(
                    receipt["zero_tolerance_counts"][field], 1
                )

    def test_structure_schema_rejects_unknown_fields_and_accepts_json_key_order(self) -> None:
        extra = copy.deepcopy(self.audit)
        extra["train"][0]["unknown_field"] = 0
        with self.assertRaisesRegex(
            aggregator.QualityStructureAggregationError, "exact schema"
        ):
            aggregator.aggregate_fixture_structure(
                public_rows_by_split=self.public,
                structure_rows_by_split=extra,
                expected_world_counts=self.counts,
                expected_sellers_per_world=2,
            )
        canonical_round_trip = json.loads(
            json.dumps(self.audit, ensure_ascii=False, sort_keys=True)
        )
        receipt = aggregator.aggregate_fixture_structure(
            public_rows_by_split=self.public,
            structure_rows_by_split=canonical_round_trip,
            expected_world_counts=self.counts,
            expected_sellers_per_world=2,
        )
        self.assertEqual(receipt["gate_failures"], [])

    def test_critical_hash_count_and_neutral_receipt_mutations_fail_closed(self) -> None:
        mutations = (
            ("hash", lambda row: row.__setitem__("full_item_sha256", "x" * 64)),
            ("item_count", lambda row: row.__setitem__("item_count", 3)),
            (
                "neutral_hash",
                lambda row: row["neutral_receipt"].__setitem__(
                    "neutral_item_sha256", "9" * 64
                ),
            ),
            (
                "neutral_read_count",
                lambda row: row["neutral_receipt"].__setitem__(
                    "neutral_metadata_source_value_read_count", 9
                ),
            ),
            (
                "non_code_verified",
                lambda row: row["neutral_receipt"][
                    "non_code_projection_commitment"
                ].__setitem__("verified", False),
            ),
            (
                "non_code_hash_mismatch",
                lambda row: row["neutral_receipt"][
                    "non_code_projection_commitment"
                ].__setitem__("neutral_sha256", "4" * 64),
            ),
        )
        for name, mutate in mutations:
            audit = copy.deepcopy(self.audit)
            mutate(audit["audit_a"][0])
            with self.subTest(name=name), self.assertRaises(
                aggregator.QualityStructureAggregationError
            ):
                aggregator.aggregate_fixture_structure(
                    public_rows_by_split=self.public,
                    structure_rows_by_split=audit,
                    expected_world_counts=self.counts,
                    expected_sellers_per_world=2,
                )

    def test_item_count_is_bounded_before_neutral_family_reconstruction(self) -> None:
        for item_count in (225, 10**12):
            audit = copy.deepcopy(self.audit)
            audit["train"][0]["item_count"] = item_count
            audit["train"][0]["registered_code_count"] = item_count
            audit["train"][0]["neutral_receipt"][
                "neutral_code_family_count"
            ] = item_count
            with self.subTest(item_count=item_count), self.assertRaisesRegex(
                aggregator.QualityStructureAggregationError,
                "Item-count capacity",
            ):
                aggregator.aggregate_fixture_structure(
                    public_rows_by_split=self.public,
                    structure_rows_by_split=audit,
                    expected_world_counts=self.counts,
                    expected_sellers_per_world=28,
                )

    def test_cross_world_code_reuse_is_counted(self) -> None:
        public = copy.deepcopy(self.public)
        public["development"][0]["owned_codes"] = list(
            public["train"][0]["owned_codes"]
        )
        receipt = aggregator.aggregate_fixture_structure(
            public_rows_by_split=public,
            structure_rows_by_split=self.audit,
            expected_world_counts=self.counts,
            expected_sellers_per_world=2,
        )
        self.assertGreater(receipt["zero_tolerance_counts"]["prior_world_code_hits"], 0)
        self.assertIn("prior_world_code_hits", receipt["gate_failures"])

    def test_duplicate_owner_and_missing_world_fail_closed(self) -> None:
        duplicate = copy.deepcopy(self.public)
        duplicate["train"][1]["owned_codes"] = list(duplicate["train"][0]["owned_codes"])
        with self.assertRaises(aggregator.QualityStructureAggregationError):
            aggregator.aggregate_fixture_structure(
                public_rows_by_split=duplicate,
                structure_rows_by_split=self.audit,
                expected_world_counts=self.counts,
                expected_sellers_per_world=2,
            )
        missing = copy.deepcopy(self.audit)
        missing["audit_b"] = []
        with self.assertRaises(aggregator.QualityStructureAggregationError):
            aggregator.aggregate_fixture_structure(
                public_rows_by_split=self.public,
                structure_rows_by_split=missing,
                expected_world_counts=self.counts,
                expected_sellers_per_world=2,
            )

    def test_formal_entry_refuses_current_unauthorized_policy(self) -> None:
        import step28_v13_v1_13_quality_channel_policy_v9 as policy_module

        policy = policy_module.load_policy()
        with self.assertRaisesRegex(
            aggregator.QualityStructureAggregationError, "unauthorized"
        ):
            aggregator.aggregate_formal_structure(
                public_rows_by_split=self.public,
                structure_rows_by_split=self.audit,
                policy=policy,
            )


if __name__ == "__main__":
    unittest.main()
