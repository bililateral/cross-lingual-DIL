#!/usr/bin/env python3
"""Contracts for the V9.2 label-free author-style counterfactual replay."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common
import step28_v13_v1_13_counterfactual_text_v9_2 as counterfactual
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_scientific_common_v9 as scientific
import step28_v13_v1_13_scientific_world_v9 as world_module
import step28_v13_v1_13_scientific_dataset_builder_v9_2 as builder_v9_2
import step28_v13_v1_13_scientific_world_v9_2 as world_v9_2
import step28_v13_v1_13_quality_probe_preparer_v9_2 as preparer_v9_2
import step28_v13_v1_13_quality_structure_aggregator_v9_2 as structure_v9_2


class CounterfactualTextV92Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        policy = scientific.load_policy()
        cls.context = scientific.build_execution_context(
            policy, execution_mode="small_smoke"
        )
        cls.template, cls.fixture, cls.style_profile = scientific.load_release_inputs(
            cls.context
        )
        historical = collision.load_historical_exclusion_registries()
        cls.record = next(
            row for row in cls.context.world_records if row["split"] == "train"
        )
        cls.accepted = world_module.build_scientific_world(
            policy=cls.context.effective_policy,
            template=cls.template,
            fixture=cls.fixture,
            style_profile=cls.style_profile,
            mode=cls.context.base_mode,
            world_record=cls.record,
            structure_key_hex=common.structure_key_for_split(
                cls.context.effective_policy,
                mode=cls.context.base_mode,
                split="train",
            ),
            document_variation_key=cls.context.document_variation_key,
            anonymous_handle_key=cls.context.anonymous_handle_key,
            historical_item_hashes=historical.item_document_hashes,
            historical_seller_hashes=historical.seller_document_hashes,
            historical_identity_hashes=historical.identity_value_hashes,
            current_item_hashes=set(),
            current_seller_hashes=set(),
            current_identity_hashes=set(),
            current_item_codes=set(),
        )

    def build_counterfactual(self) -> counterfactual.CounterfactualFullSurface:
        accepted = self.accepted
        split = accepted.split
        policy = self.context.effective_policy
        safe_library = world_module._candidate_safe_library(
            policy=policy,
            template=self.template,
            fixture=self.fixture,
            split=split,
        )
        effective_style_rows = world_module.stage_parent._effective_style_rows(
            policy=policy,
            template=self.template,
            mode=self.context.base_mode,
            world=accepted.world,
        )
        effective_styles = {
            str(row["seller_uid"]): dict(row["style_factors"])
            for row in effective_style_rows
        }
        candidate_key = collision.derive_candidate_key(
            self.context.document_variation_key,
            split=split,
            world_uid=accepted.world_uid,
            candidate_index=accepted.candidate_index,
        )
        private = accepted.world["private"]
        return counterfactual.materialize_style_deranged_full_surface(
            profile_policy=policy,
            mode=self.context.base_mode,
            split=split,
            base_template=self.template,
            safe_library=safe_library,
            fixture=self.fixture,
            world_uid=accepted.world_uid,
            candidate_key=candidate_key,
            public_world=accepted.world["public"]["world"],
            public_sellers=accepted.world["public"]["sellers"],
            public_items=accepted.world["public"]["items"],
            original_redacted_items=accepted.redacted_items,
            original_seller_profiles=accepted.seller_profiles,
            complete_model_pair_endpoints=accepted.world["public"][
                "complete_model_pair_endpoints"
            ],
            render_asts=private["render_asts"],
            identity_slots_audit=private["identity_slots_audit"],
            noise_slots_audit=private["noise_slots_audit"],
            override_audit=private["override_audit"],
            effective_styles=effective_styles,
            baseline_identity33=accepted.identity33,
        )

    def test_public_signature_exposes_no_truth_or_controller_capability(self) -> None:
        parameters = set(
            inspect.signature(
                counterfactual.materialize_style_deranged_full_surface
            ).parameters
        )
        forbidden = {
            "controller_membership",
            "pair_labels",
            "qrels",
            "audit_a_truth",
            "audit_b_truth",
            "quality_results",
        }
        self.assertFalse(parameters & forbidden)

    def test_double_production_replay_and_style_only_ast_change_close(self) -> None:
        result = self.build_counterfactual()
        audit = result.audit
        self.assertTrue(audit["double_replay"]["byte_identical"])
        self.assertEqual(
            audit["double_replay"]["independent_production_replay_count"], 2
        )
        self.assertEqual(audit["mapping"]["fixed_point_count"], 0)
        self.assertEqual(
            audit["forbidden_capability_mounted"],
            {
                name: False for name in counterfactual.FORBIDDEN_CAPABILITIES
            },
        )
        self.assertEqual(len(result.redacted_items), len(self.accepted.redacted_items))
        self.assertEqual(len(result.seller_profiles), 28)
        self.assertEqual(
            audit["model_inputs"]["counterfactual_full_items_sha256"],
            common.canonical_sha256(result.redacted_items),
        )
        self.assertEqual(
            audit["model_inputs"]["counterfactual_full_profiles_sha256"],
            world_module.channel_materializer._persisted_profile_sha256(
                result.seller_profiles
            ),
        )

    def test_mapping_is_not_redrawn_for_low_or_duplicate_style_dose(self) -> None:
        first = self.build_counterfactual()
        second = self.build_counterfactual()
        self.assertEqual(
            first.audit["mapping"]["mapping_sha256"],
            second.audit["mapping"]["mapping_sha256"],
        )
        self.assertEqual(
            first.audit["mapping"]["target_source_pairs"],
            second.audit["mapping"]["target_source_pairs"],
        )

    def test_five_m1_commitments_are_public_id_only_and_distinct(self) -> None:
        signature = set(
            inspect.signature(world_v9_2.build_m1_mapping_commitments).parameters
        )
        self.assertEqual(signature, {"endpoints", "world_uid"})
        endpoints = self.accepted.world["public"][
            "complete_model_pair_endpoints"
        ]
        first = world_v9_2.build_m1_mapping_commitments(
            endpoints, world_uid=self.accepted.world_uid
        )
        second = world_v9_2.build_m1_mapping_commitments(
            endpoints, world_uid=self.accepted.world_uid
        )
        self.assertEqual(first, second)
        self.assertEqual(
            [row["repeat_id"] for row in first],
            list(world_v9_2.M1_REPEAT_IDS),
        )
        self.assertEqual(len({row["mapping_sha256"] for row in first}), 5)

    def test_world_layer_materializes_counterfactual_before_private_truth(self) -> None:
        historical = collision.load_historical_exclusion_registries()
        call_order: list[str] = []
        real_counterfactual = (
            world_v9_2.counterfactual.materialize_style_deranged_full_surface
        )
        real_private_truth = world_v9_2.v9._build_private_truth

        def observe_counterfactual(**kwargs):
            call_order.append("counterfactual")
            return real_counterfactual(**kwargs)

        def observe_private_truth(world):
            call_order.append("private_truth")
            return real_private_truth(world)

        with mock.patch.object(
            world_v9_2.counterfactual,
            "materialize_style_deranged_full_surface",
            side_effect=observe_counterfactual,
        ), mock.patch.object(
            world_v9_2.v9,
            "_build_private_truth",
            side_effect=observe_private_truth,
        ):
            accepted = world_v9_2.build_scientific_world(
                policy=self.context.effective_policy,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style_profile,
                mode=self.context.base_mode,
                world_record=self.record,
                structure_key_hex=common.structure_key_for_split(
                    self.context.effective_policy,
                    mode=self.context.base_mode,
                    split="train",
                ),
                document_variation_key=self.context.document_variation_key,
                anonymous_handle_key=self.context.anonymous_handle_key,
                historical_item_hashes=historical.item_document_hashes,
                historical_seller_hashes=historical.seller_document_hashes,
                historical_identity_hashes=historical.identity_value_hashes,
                current_item_hashes=set(),
                current_seller_hashes=set(),
                current_identity_hashes=set(),
                current_item_codes=set(),
            )
        self.assertEqual(call_order, ["counterfactual", "private_truth"])
        self.assertEqual(accepted.channel_structure_audit["model_input_file_count"], 8)
        self.assertEqual(
            accepted.channel_structure_audit[
                "labels_or_retrieval_truth_materialized_before_audit"
            ],
            False,
        )
        self.assertEqual(
            len(accepted.counterfactual_redacted_items),
            len(accepted.redacted_items),
        )
        self.assertEqual(len(accepted.counterfactual_seller_profiles), 28)
        public_by_split = {
            split: (
                accepted.public_code_probe_input if split == "train" else ()
            )
            for split in ("train", "development", "audit_a", "audit_b")
        }
        structure_by_split = {
            split: (
                (accepted.channel_structure_audit,) if split == "train" else ()
            )
            for split in ("train", "development", "audit_a", "audit_b")
        }
        eligibility_by_split = {
            split: (
                accepted.text_probe_eligibility_input if split == "train" else ()
            )
            for split in ("train", "development", "audit_a", "audit_b")
        }
        train_surfaces = {
            "surface_full": (
                tuple(
                    builder_v9_2.v9_builder._project_model_redacted_item(row)
                    for row in accepted.redacted_items
                ),
                tuple(
                    builder_v9_2.v9_builder._project_model_seller_profile(row)
                    for row in accepted.seller_profiles
                ),
            ),
            "surface_code_masked": (
                tuple(
                    builder_v9_2.v9_builder._project_model_redacted_item(row)
                    for row in accepted.masked_redacted_items
                ),
                tuple(
                    builder_v9_2.v9_builder._project_model_seller_profile(row)
                    for row in accepted.masked_seller_profiles
                ),
            ),
            "surface_code_neutralized": (
                tuple(
                    builder_v9_2.v9_builder._project_model_redacted_item(row)
                    for row in accepted.neutral_redacted_items
                ),
                tuple(
                    builder_v9_2.v9_builder._project_model_seller_profile(row)
                    for row in accepted.neutral_seller_profiles
                ),
            ),
            preparer_v9_2.COUNTERFACTUAL_HARD_SURFACE: (
                tuple(
                    builder_v9_2.v9_builder._project_model_redacted_item(row)
                    for row in accepted.counterfactual_redacted_items
                ),
                tuple(
                    builder_v9_2.v9_builder._project_model_seller_profile(row)
                    for row in accepted.counterfactual_seller_profiles
                ),
            ),
        }
        model_surfaces_by_split = {
            split: (
                train_surfaces
                if split == "train"
                else {
                    surface: ((), ())
                    for surface in structure_v9_2.MODEL_SURFACES
                }
            )
            for split in ("train", "development", "audit_a", "audit_b")
        }
        structure_receipt = structure_v9_2.aggregate_fixture_structure(
            public_rows_by_split=public_by_split,
            structure_rows_by_split=structure_by_split,
            eligibility_rows_by_split=eligibility_by_split,
            model_surface_rows_by_split=model_surfaces_by_split,
            expected_world_counts={
                "train": 1,
                "development": 0,
                "audit_a": 0,
                "audit_b": 0,
            },
            expected_sellers_per_world=28,
        )
        self.assertEqual(structure_receipt["world_receipt_count"], 1)
        self.assertEqual(
            structure_receipt["metric_values"][
                "cross_branch_invariant_mismatch_count"
            ],
            0,
        )
        self.assertIn(
            "pretruth_counterfactual_text_matrix_count_mismatch_split_count",
            structure_receipt["pending_structure_metrics"],
        )
        endpoint = accepted.world["public"]["complete_model_pair_endpoints"][0]
        with tempfile.TemporaryDirectory(prefix="step28-v9-2-writer-") as temp:
            root = Path(temp)
            writers = builder_v9_2.SplitWritersV92.open(
                root,
                endpoint_fields=tuple(endpoint),
                identity_fields=tuple(accepted.identity33[0]),
            )
            try:
                builder_v9_2.write_world(writers, accepted)
                counts = builder_v9_2.validate_one_world_model_input_counts(
                    writers,
                    expected_item_count=len(accepted.counterfactual_redacted_items),
                )
            finally:
                writers.close()
            self.assertEqual(tuple(counts), builder_v9_2.MODEL_INPUT_PATHS)
            for relative in builder_v9_2.MODEL_INPUT_PATHS:
                self.assertTrue((root / relative).is_file(), relative)
            persisted_profiles = [
                json.loads(line)
                for line in (root / builder_v9_2.COUNTERFACTUAL_PROFILE_PATH)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                accepted.channel_structure_audit[
                    "counterfactual_full_profile_sha256"
                ],
                common.canonical_sha256(persisted_profiles),
            )
            path_pairs = {
                "surface_full": (
                    "observed/redacted_items.jsonl",
                    "observed/model_seller_profiles.jsonl",
                ),
                "surface_code_masked": (
                    "observed/redacted_items.code_masked.jsonl",
                    "observed/model_seller_profiles.code_masked.jsonl",
                ),
                "surface_code_neutralized": (
                    "observed/redacted_items.code_neutralized.jsonl",
                    "observed/model_seller_profiles.code_neutralized.jsonl",
                ),
                preparer_v9_2.COUNTERFACTUAL_HARD_SURFACE: (
                    builder_v9_2.COUNTERFACTUAL_ITEM_PATH,
                    builder_v9_2.COUNTERFACTUAL_PROFILE_PATH,
                ),
            }
            surfaces = {}
            sources = {}
            for surface, relatives in path_pairs.items():
                item_path, profile_path = (root / value for value in relatives)
                surfaces[surface] = (
                    [
                        json.loads(line)
                        for line in item_path.read_text(encoding="utf-8").splitlines()
                    ],
                    [
                        json.loads(line)
                        for line in profile_path.read_text(encoding="utf-8").splitlines()
                    ],
                )
                sources[surface] = tuple(
                    preparer_v9_2.SourceCommitment(
                        path=relative,
                        size_bytes=(root / relative).stat().st_size,
                        sha256=common.sha256_file(root / relative),
                    )
                    for relative in sorted(
                        relatives, key=lambda value: value.encode("utf-8")
                    )
                )
            endpoint_rows = [
                {
                    field: row[field]
                    for field in preparer_v9_2.v9.ENDPOINT_FIELDS
                }
                for row in accepted.world["public"][
                    "complete_model_pair_endpoints"
                ]
            ]
            eligibility_relative = "private/text_probe_eligibility_input.jsonl"
            eligibility_path = root / eligibility_relative
            eligibility = preparer_v9_2.v9.freeze_text_eligibility(
                eligibility_rows=[
                    json.loads(line)
                    for line in eligibility_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ],
                endpoints=endpoint_rows,
                ordered_world_uids=(accepted.world_uid,),
                sources=(
                    preparer_v9_2.SourceCommitment(
                        path=eligibility_relative,
                        size_bytes=eligibility_path.stat().st_size,
                        sha256=common.sha256_file(eligibility_path),
                    ),
                ),
            )
            bundle = preparer_v9_2.freeze_all_text_surfaces_before_truth(
                surface_rows=surfaces,
                endpoints=endpoint_rows,
                ordered_world_uids=(accepted.world_uid,),
                sources_by_surface=sources,
                text_eligibility=eligibility,
            )
            descriptive, hard = preparer_v9_2.split_text_matrix_roles(bundle)
            self.assertEqual(len(bundle.matrices), 28)
            self.assertEqual(len(descriptive), 21)
            self.assertEqual(len(hard), 7)
            self.assertEqual(len(bundle.actual_consumption_receipts), 12)
            f_p_u = preparer_v9_2.validate_counterfactual_f_p_u_consumption(
                bundle
            )
            self.assertEqual(
                f_p_u["f_p_u_actual_consumption_mismatch_count"], 0
            )


if __name__ == "__main__":
    unittest.main()
