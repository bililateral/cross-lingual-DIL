#!/usr/bin/env python3
"""Contracts for the Step28-v13 v1.13 v9 document-capacity repair."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common
import step28_v13_structure as structure
import step28_v13_v1_13_document_capacity_v9 as capacity
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_pure_natural_renderer_v9 as pure_renderer
import step28_v13_v1_13_quality_channel_materializer_v9 as channel_materializer
import step28_v13_v1_13_scientific_common_v9 as scientific
import step28_v13_v1_13_scientific_dataset_builder_v9 as dataset_builder
import step28_v13_v1_13_scientific_world_v9 as world_module
import step28_v13_v1_13_v9_causal_replay_0_283 as causal_replay


class DocumentCapacityPrimitiveContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.document_key = bytes(range(32))
        self.code_key = capacity.derive_code_key(self.document_key)

    def test_feistel_round_trip_boundaries_and_used_coordinate_domain(self) -> None:
        probes = [0, 1, 7, 8, 223, 255, 256, (1 << 40) - 1]
        for value in probes:
            with self.subTest(value=value):
                permuted = capacity.permute_40(code_key=self.code_key, value=value)
                self.assertEqual(
                    capacity.invert_40(code_key=self.code_key, value=permuted),
                    value,
                )
                code = capacity.encode_code(code_key=self.code_key, value=value)
                self.assertRegex(code, r"^Q[A-P]{10}$")
                self.assertEqual(
                    capacity.decode_code(code_key=self.code_key, code=code), value
                )

        values = [
            capacity.coordinate(
                mode_global_ordinal=world_ordinal,
                seller_slot_ordinal=seller_slot,
                item_slot_ordinal=item_slot,
            )
            for world_ordinal in range(64)
            for seller_slot in range(28)
            for item_slot in range(8)
        ]
        self.assertEqual(len(values), len(set(values)))
        codes = {
            capacity.encode_code(code_key=self.code_key, value=value)
            for value in values
        }
        self.assertEqual(len(codes), len(values))

    def test_coordinate_rejects_boolean_and_out_of_range_slots(self) -> None:
        invalid = (
            {"mode_global_ordinal": True, "seller_slot_ordinal": 0, "item_slot_ordinal": 0},
            {"mode_global_ordinal": 1 << 32, "seller_slot_ordinal": 0, "item_slot_ordinal": 0},
            {"mode_global_ordinal": 0, "seller_slot_ordinal": 28, "item_slot_ordinal": 0},
            {"mode_global_ordinal": 0, "seller_slot_ordinal": 0, "item_slot_ordinal": 8},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(
                capacity.DocumentCapacityError
            ):
                capacity.coordinate(**kwargs)


class TemplateCapacityContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = scientific.load_policy()
        cls.context = scientific.build_execution_context(
            cls.policy, execution_mode="small_smoke"
        )
        cls.template, cls.fixture, _style = scientific.load_release_inputs(cls.context)

    def test_candidate_template_is_only_the_registered_extension(self) -> None:
        for split in scientific.SPLITS:
            library = world_module._candidate_safe_library(
                policy=self.context.effective_policy,
                template=self.template,
                fixture=self.fixture,
                split=split,
            )
            pure_renderer.validate_safe_library(library)
            for field in ("title_skeletons", "description_skeletons"):
                mapping = pure_renderer._capacity_index_map(
                    library[field], description=field.startswith("description")
                )
                self.assertEqual(set(mapping), set(range(8)))
                self.assertEqual(len(set(mapping.values())), 8)
                for source, target in mapping.items():
                    self.assertIn("{code}", library[field][target])
                    if "{code}" in library[field][source]:
                        self.assertEqual(source, target)

    def test_injective_twins_preserve_high_semantic_title_inequality(self) -> None:
        for split in scientific.SPLITS:
            library = world_module._candidate_safe_library(
                policy=self.context.effective_policy,
                template=self.template,
                fixture=self.fixture,
                split=split,
            )
            mapping = pure_renderer._capacity_index_map(
                library["title_skeletons"], description=False
            )
            for left in range(8):
                for right in range(8):
                    self.assertEqual(left == right, mapping[left] == mapping[right])


class ScientificWorldCapacityContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = scientific.load_policy()
        cls.context = scientific.build_execution_context(
            cls.policy, execution_mode="small_smoke"
        )
        cls.template, cls.fixture, cls.style_profile = scientific.load_release_inputs(
            cls.context
        )
        cls.historical = collision.load_historical_exclusion_registries()
        cls.record = next(
            row for row in cls.context.world_records if row["split"] == "train"
        )

    def build_one(
        self,
        *,
        current_item_codes: set[str] | None = None,
    ) -> world_module.AcceptedScientificWorld:
        return world_module.build_scientific_world(
            policy=self.context.effective_policy,
            template=self.template,
            fixture=self.fixture,
            style_profile=self.style_profile,
            mode=self.context.base_mode,
            world_record=self.record,
            structure_key_hex=common.structure_key_for_split(
                self.context.effective_policy,
                mode=self.context.base_mode,
                split=str(self.record["split"]),
            ),
            document_variation_key=self.context.document_variation_key,
            anonymous_handle_key=self.context.anonymous_handle_key,
            historical_item_hashes=self.historical.item_document_hashes,
            historical_seller_hashes=self.historical.seller_document_hashes,
            historical_identity_hashes=self.historical.identity_value_hashes,
            current_item_hashes=set(),
            current_seller_hashes=set(),
            current_identity_hashes=set(),
            current_item_codes=(
                set() if current_item_codes is None else current_item_codes
            ),
        )

    def test_four_split_small_smoke_closes_with_shared_registries(self) -> None:
        current_items: set[str] = set()
        current_sellers: set[str] = set()
        current_identities: set[str] = set()
        current_codes: set[str] = set()
        accepted_splits: list[str] = []
        for record in sorted(
            self.context.world_records,
            key=lambda row: scientific.SPLITS.index(str(row["split"])),
        ):
            accepted = world_module.build_scientific_world(
                policy=self.context.effective_policy,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style_profile,
                mode=self.context.base_mode,
                world_record=record,
                structure_key_hex=common.structure_key_for_split(
                    self.context.effective_policy,
                    mode=self.context.base_mode,
                    split=str(record["split"]),
                ),
                document_variation_key=self.context.document_variation_key,
                anonymous_handle_key=self.context.anonymous_handle_key,
                historical_item_hashes=self.historical.item_document_hashes,
                historical_seller_hashes=self.historical.seller_document_hashes,
                historical_identity_hashes=self.historical.identity_value_hashes,
                current_item_hashes=current_items,
                current_seller_hashes=current_sellers,
                current_identity_hashes=current_identities,
                current_item_codes=current_codes,
            )
            accepted_splits.append(accepted.split)
            self.assertEqual(
                sum(accepted.document_capacity_receipt["projection_counts"].values()),
                accepted.document_capacity_receipt["item_count"],
            )
        self.assertEqual(accepted_splits, list(scientific.SPLITS))
        self.assertEqual(len(current_codes), len(current_items))

    def test_joint_empty_projection_is_a_representation_only_counterfactual(self) -> None:
        accepted = self.build_one()
        base = copy.deepcopy(accepted.world)
        identity_items = {
            str(row["item_uid"])
            for row in base["private"]["identity_slots_audit"]
        }
        noise_items = {
            str(row["item_uid"])
            for row in base["private"]["noise_slots_audit"]
        }
        override_items = {
            str(row[field])
            for row in base["private"]["override_audit"]
            for field in ("item_uid_left", "item_uid_right")
        }
        candidate_uid = next(
            str(row["item_uid"])
            for row in base["public"]["items"]
            if str(row["item_uid"])
            not in identity_items | noise_items | override_items
        )
        public = next(
            row for row in base["public"]["items"] if row["item_uid"] == candidate_uid
        )
        ast = next(
            row
            for row in base["private"]["render_asts"]
            if row["item_uid"] == candidate_uid
        )
        public["title"] = ""
        public["description"] = ""
        ast["title_nonempty"] = False
        ast["description_nonempty"] = False
        ast["identity_slot_uids"] = []
        ast["noise_slot_uid"] = ""
        before = copy.deepcopy(base)
        projected, receipt = capacity.apply_capacity_parent(
            policy=self.context.effective_policy,
            mode=self.context.base_mode,
            world_record=self.record,
            document_variation_key=self.context.document_variation_key,
            world=base,
        )
        self.assertEqual(base, before)
        self.assertEqual(
            receipt["projection_counts"]["title_only"]
            + receipt["projection_counts"]["description_only"],
            1,
        )
        projected_ast = next(
            row
            for row in projected["private"]["render_asts"]
            if row["item_uid"] == candidate_uid
        )
        self.assertEqual(
            (projected_ast["title_nonempty"], projected_ast["description_nonempty"])
            in {(True, False), (False, True)},
            True,
        )

    def test_capacity_parent_rejects_identity_edit_projection_drift(self) -> None:
        accepted = self.build_one()
        tampered = copy.deepcopy(accepted.world)
        self.assertTrue(tampered["private"]["identity_slots_edit"])
        tampered["private"]["identity_slots_edit"][0]["raw_surface"] += "篡改"
        with self.assertRaisesRegex(
            capacity.DocumentCapacityError, "audit/edit projections disagree"
        ):
            capacity.apply_capacity_parent(
                policy=self.context.effective_policy,
                mode=self.context.base_mode,
                world_record=self.record,
                document_variation_key=self.context.document_variation_key,
                world=tampered,
            )

    def test_joint_empty_projection_rejects_hidden_public_text(self) -> None:
        accepted = self.build_one()
        tampered = copy.deepcopy(accepted.world)
        identity_items = {
            str(row["item_uid"])
            for row in tampered["private"]["identity_slots_audit"]
        }
        noise_items = {
            str(row["item_uid"])
            for row in tampered["private"]["noise_slots_audit"]
        }
        override_items = {
            str(row[field])
            for row in tampered["private"]["override_audit"]
            for field in ("item_uid_left", "item_uid_right")
        }
        candidate_uid = next(
            str(row["item_uid"])
            for row in tampered["public"]["items"]
            if str(row["item_uid"])
            not in identity_items | noise_items | override_items
        )
        public = next(
            row
            for row in tampered["public"]["items"]
            if row["item_uid"] == candidate_uid
        )
        ast = next(
            row
            for row in tampered["private"]["render_asts"]
            if row["item_uid"] == candidate_uid
        )
        public["title"] = "隐藏但非空的原始文本"
        public["description"] = ""
        ast["title_nonempty"] = False
        ast["description_nonempty"] = False
        ast["identity_slot_uids"] = []
        ast["noise_slot_uid"] = ""
        with self.assertRaisesRegex(
            capacity.DocumentCapacityError, "already carries text"
        ):
            capacity.apply_capacity_parent(
                policy=self.context.effective_policy,
                mode=self.context.base_mode,
                world_record=self.record,
                document_variation_key=self.context.document_variation_key,
                world=tampered,
            )

    def test_one_world_closes_capacity_identity_and_document_invariants(self) -> None:
        accepted = self.build_one()
        receipt = accepted.document_capacity_receipt
        audit = accepted.document_capacity_audit
        self.assertEqual(receipt["version"], capacity.VERSION)
        self.assertEqual(receipt["item_count"], receipt["unique_code_count"])
        self.assertEqual(audit["item_count"], audit["unique_code_count"])
        self.assertEqual(audit["seller_count"], 28)
        self.assertTrue(audit["all_items_retain_own_code"])
        self.assertTrue(
            audit["all_seller_descriptions_retain_exclusive_owned_code"]
        )
        self.assertEqual(len(accepted.identity33), 378)
        self.assertEqual(len(accepted.code_registry_delta), len(accepted.redacted_items))
        self.assertEqual(len(set(accepted.code_registry_delta)), len(accepted.code_registry_delta))
        self.assertEqual(
            len(accepted.masked_redacted_items), len(accepted.redacted_items)
        )
        self.assertEqual(
            len(accepted.neutral_redacted_items),
            len(accepted.redacted_items),
        )
        self.assertEqual(len(accepted.masked_seller_profiles), 28)
        self.assertEqual(len(accepted.neutral_seller_profiles), 28)
        self.assertEqual(len(accepted.public_code_probe_input), 28)
        self.assertEqual(len(accepted.text_probe_eligibility_input), 378)
        self.assertEqual(
            sum(
                bool(row["text_probe_eligible"])
                for row in accepted.text_probe_eligibility_input
            ),
            372,
        )
        self.assertEqual(
            accepted.channel_structure_audit["audit_truth_open_count"],
            0,
        )
        self.assertEqual(
            accepted.channel_structure_audit["audit_truth_read_count"],
            0,
        )
        self.assertEqual(
            accepted.channel_structure_audit[
                "audit_truth_materialized_row_count"
            ],
            0,
        )
        self.assertRegex(
            accepted.candidate_zero_lineage_reference_sha256, r"^[0-9a-f]{64}$"
        )

    def test_creation_slots_reconstruct_every_actual_uid(self) -> None:
        accepted = self.build_one()
        world = accepted.world
        id_key = self.context.effective_policy["randomness"][
            self.context.base_mode
        ]["id_key_hex"]
        world_uid = accepted.world_uid
        expected_sellers = {
            structure.base_uid(
                key_hex=id_key,
                entity_kind="seller",
                parent_uid_or_mode=world_uid,
                ordinal=ordinal,
            )
            for ordinal in range(28)
        }
        self.assertEqual(
            expected_sellers,
            {str(row["seller_uid"]) for row in world["public"]["sellers"]},
        )
        code_key = capacity.derive_code_key(self.context.document_variation_key)
        decoded = {
            capacity.decode_code(code_key=code_key, code=str(row["code"]))
            for row in world["private"]["render_asts"]
        }
        self.assertEqual(len(decoded), len(world["private"]["render_asts"]))

    def test_clone_and_high_semantic_boundaries_survive(self) -> None:
        accepted = self.build_one()
        redacted = {str(row["item_uid"]): row for row in accepted.redacted_items}
        ast = {
            str(row["item_uid"]): row
            for row in accepted.world["private"]["render_asts"]
        }
        for row in accepted.world["private"]["override_audit"]:
            left = str(row["item_uid_left"])
            right = str(row["item_uid_right"])
            if row["override_kind"] == "exact_title_clone":
                self.assertEqual(redacted[left]["title"], redacted[right]["title"])
                self.assertIn(str(ast[right]["code"]), redacted[right]["description"])
                self.assertNotEqual(
                    collision.item_document_hash(
                        title=str(redacted[left]["title"]),
                        description=str(redacted[left]["description"]),
                    ),
                    collision.item_document_hash(
                        title=str(redacted[right]["title"]),
                        description=str(redacted[right]["description"]),
                    ),
                )
            elif row["override_kind"] == "high_semantic_similarity":
                self.assertNotEqual(
                    ast[left]["title_skeleton_index"],
                    ast[right]["title_skeleton_index"],
                )

    def test_code_registry_reuse_fails_after_lineage_before_collision(self) -> None:
        first = self.build_one()
        reused = set(first.code_registry_delta)
        with self.assertRaisesRegex(
            world_module.ScientificWorldError, "item-code registry collision"
        ):
            self.build_one(current_item_codes=reused)

    def test_candidate_zero_is_assembled_before_first_collision_read(self) -> None:
        assembled = 0
        real_assemble = world_module._assemble_candidate
        real_collision = world_module._collision_categories

        def assemble_probe(*args, **kwargs):
            nonlocal assembled
            result = real_assemble(*args, **kwargs)
            assembled += 1
            return result

        def collision_probe(*args, **kwargs):
            self.assertGreaterEqual(assembled, 1)
            return real_collision(*args, **kwargs)

        with mock.patch.object(
            world_module, "_assemble_candidate", side_effect=assemble_probe
        ), mock.patch.object(
            world_module, "_collision_categories", side_effect=collision_probe
        ):
            accepted = self.build_one()
        self.assertGreaterEqual(accepted.candidates_examined, 1)

    def test_channel_views_materialize_before_private_pair_truth(self) -> None:
        events: list[str] = []
        real_materialize = (
            channel_materializer.materialize_label_free_channel_views
        )
        real_truth = world_module._build_private_truth

        def materialize_probe(*args, **kwargs):
            self.assertNotIn("world", kwargs)
            self.assertNotIn("pair_labels", kwargs)
            self.assertNotIn("qrels", kwargs)
            events.append("views")
            return real_materialize(*args, **kwargs)

        def truth_probe(*args, **kwargs):
            events.append("truth")
            return real_truth(*args, **kwargs)

        with mock.patch.object(
            channel_materializer,
            "materialize_label_free_channel_views",
            side_effect=materialize_probe,
        ), mock.patch.object(
            world_module,
            "_build_private_truth",
            side_effect=truth_probe,
        ):
            self.build_one()
        self.assertEqual(events, ["views", "truth"])

    def test_builder_split_tree_names_all_channel_inputs(self) -> None:
        required = {
            "observed/redacted_items.code_masked.jsonl",
            "observed/redacted_items.code_neutralized.jsonl",
            "observed/model_seller_profiles.code_masked.jsonl",
            "observed/model_seller_profiles.code_neutralized.jsonl",
            "private/public_code_probe_input.jsonl",
            "private/text_probe_eligibility_input.jsonl",
            "private/channel_structure_audit.jsonl",
        }
        self.assertTrue(
            required.issubset(set(dataset_builder.EXPECTED_SPLIT_DATA_PATHS))
        )

    def test_one_world_writer_materializes_and_recounts_all_channel_inputs(
        self,
    ) -> None:
        accepted = self.build_one()
        endpoint = accepted.world["public"]["complete_model_pair_endpoints"][0]
        with tempfile.TemporaryDirectory(prefix="step28-v9-writer-") as temp:
            root = Path(temp)
            writers = dataset_builder._SplitWriters.open(
                root,
                endpoint_fields=tuple(endpoint),
                identity_fields=tuple(accepted.identity33[0]),
            )
            try:
                dataset_builder._write_world(writers, accepted)
                dataset_builder._validate_split_counts(
                    split=accepted.split,
                    world_count=1,
                    writers=writers,
                    positive_count=sum(
                        int(row["label"]) for row in accepted.pair_labels
                    ),
                    expected_item_count=len(accepted.item_registry_delta),
                    item_document_hashes=set(accepted.item_registry_delta),
                    seller_document_hashes=set(accepted.seller_registry_delta),
                    identity_value_hashes=set(accepted.identity_registry_delta),
                )
            finally:
                writers.close()
            for relative in dataset_builder.EXPECTED_SPLIT_DATA_PATHS:
                self.assertTrue((root / relative).is_file(), relative)
            eligibility_rows = [
                json.loads(line)
                for line in (
                    root / "private" / "text_probe_eligibility_input.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(eligibility_rows), 378)
            self.assertEqual(
                sum(bool(row["text_probe_eligible"]) for row in eligibility_rows),
                372,
            )

    def test_document_registries_are_unread_until_candidate_zero_lineage_exists(
        self,
    ) -> None:
        lineage_exists = False
        real_digest = world_module._profile_provenance_source_multiset_sha256

        class GuardedSet(set):
            def __iter__(self):
                if not lineage_exists:
                    raise AssertionError("document registry read before lineage")
                return super().__iter__()

            def __and__(self, other):
                if not lineage_exists:
                    raise AssertionError("document registry read before lineage")
                return super().__and__(other)

        def digest_probe(value):
            nonlocal lineage_exists
            result = real_digest(value)
            lineage_exists = True
            return result

        with mock.patch.object(
            world_module,
            "_profile_provenance_source_multiset_sha256",
            side_effect=digest_probe,
        ):
            accepted = world_module.build_scientific_world(
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
                historical_item_hashes=GuardedSet(
                    self.historical.item_document_hashes
                ),
                historical_seller_hashes=GuardedSet(
                    self.historical.seller_document_hashes
                ),
                historical_identity_hashes=self.historical.identity_value_hashes,
                current_item_hashes=GuardedSet(),
                current_seller_hashes=GuardedSet(),
                current_identity_hashes=set(),
                current_item_codes=GuardedSet(),
            )
        self.assertTrue(lineage_exists)
        self.assertGreaterEqual(accepted.candidates_examined, 1)


class AuthorizationContracts(unittest.TestCase):
    def test_causal_replay_boundary_is_exact_and_has_no_output_argument(self) -> None:
        self.assertEqual(causal_replay.AUTHORIZED_SPLIT, "train")
        self.assertEqual(causal_replay.AUTHORIZED_FINAL_ORDINAL, 283)
        self.assertEqual(causal_replay.AUTHORIZED_WORLD_COUNT, 284)
        self.assertEqual(
            tuple(causal_replay.run_replay.__code__.co_varnames[: causal_replay.run_replay.__code__.co_argcount]),
            (),
        )

    def test_parameterless_design_entry_remains_closed_before_any_build_io(self) -> None:
        self.assertEqual(
            tuple(
                dataset_builder.run_design_preflight_once.__code__.co_varnames[
                    : dataset_builder.run_design_preflight_once.__code__.co_argcount
                ]
            ),
            (),
        )
        self.assertFalse(hasattr(dataset_builder, "run_build"))
        self.assertFalse(
            hasattr(dataset_builder, "_run_build_unreachable_until_fresh_review")
        )
        transaction = dataset_builder._run_design_preflight_transaction
        self.assertEqual(transaction.__code__.co_argcount, 0)
        self.assertEqual(transaction.__code__.co_kwonlyargcount, 1)
        self.assertEqual(transaction.__code__.co_varnames[0], "context")
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            missing_receipt = Path(temp) / "receipt-that-does-not-exist.json"
            watched = (
                mock.patch.object(
                    dataset_builder.scientific, "build_execution_context"
                ),
                mock.patch.object(
                    dataset_builder.collision,
                    "load_historical_exclusion_registries",
                ),
                mock.patch.object(dataset_builder.world_module, "build_scientific_world"),
                mock.patch.object(dataset_builder.Path, "mkdir"),
                mock.patch.object(dataset_builder.Path, "rename"),
            )
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        dataset_builder,
                        "_authorization_receipt_path",
                        return_value=missing_receipt,
                    )
                )
                entered = [stack.enter_context(patcher) for patcher in watched]
                with self.assertRaisesRegex(
                    dataset_builder.DatasetBuildError, "receipt is absent"
                ):
                    dataset_builder.run_design_preflight_once()
                for probe in entered:
                    probe.assert_not_called()
        policy = scientific.load_policy()
        with self.assertRaises(scientific.ScientificBuilderError):
            scientific.build_execution_context(policy, execution_mode="formal")

    def test_design_receipt_is_consumed_once_before_transaction(self) -> None:
        payload = b'{"receipt":"fixture-only"}\n'
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            receipt_path = Path(temp) / "authorization.json"
            receipt_path.write_bytes(payload)
            sha256 = hashlib.sha256(payload).hexdigest()
            receipt = dataset_builder._VerifiedDesignBuildReceipt(
                path=receipt_path,
                size_bytes=len(payload),
                sha256=sha256,
                receipt_id="1" * 64,
                review_response_sha256="2" * 64,
                git_commit="3" * 40,
                git_tree="4" * 40,
                random_authority_commitment_sha256="5" * 64,
                builder_policy_binding={},
                quality_policy_binding={},
                builder_source_file={},
            )
            consumed = dataset_builder._consume_design_build_receipt(receipt)
            self.assertFalse(receipt_path.exists())
            consumed_path = ROOT / consumed["path"]
            self.assertTrue(consumed_path.is_file())
            self.assertEqual(consumed_path.read_bytes(), payload)
            self.assertEqual(consumed["sha256"], sha256)
            receipt_path.write_bytes(payload)
            with self.assertRaisesRegex(
                dataset_builder.DatasetBuildError, "already consumed"
            ):
                dataset_builder._consume_design_build_receipt(receipt)

    def test_receipt_path_state_distinguishes_pending_from_consumed(self) -> None:
        policy = scientific.load_policy()
        sha256 = "5" * 64
        pending = dataset_builder._expected_receipt_relative_path(
            policy, receipt_sha256=sha256, consumed=False
        )
        consumed = dataset_builder._expected_receipt_relative_path(
            policy, receipt_sha256=sha256, consumed=True
        )
        self.assertEqual(
            pending,
            "private_custody/"
            "step28_v13_v1_13_v9_design_build_authorization.json",
        )
        self.assertEqual(
            consumed,
            "private_custody/"
            "step28_v13_v1_13_v9_design_build_authorization."
            f"consumed.{sha256}.json",
        )
        self.assertNotEqual(pending, consumed)

    def test_public_entry_consumes_receipt_before_private_transaction(self) -> None:
        policy = scientific.load_policy()
        quality_policy = dataset_builder.quality_policy_module.load_policy()
        receipt_path = ROOT / policy["design_build_authorization_overlay"][
            "receipt_path"
        ]
        receipt_sha256 = "5" * 64
        receipt = dataset_builder._VerifiedDesignBuildReceipt(
            path=receipt_path,
            size_bytes=123,
            sha256=receipt_sha256,
            receipt_id="6" * 64,
            review_response_sha256="7" * 64,
            git_commit="8" * 40,
            git_tree="9" * 40,
            random_authority_commitment_sha256="a" * 64,
            builder_policy_binding={},
            quality_policy_binding={"canonical_self_hash": "b" * 64},
            builder_source_file={},
        )
        events: list[str] = []
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            execution = mock.Mock()
            execution.output_root = Path(temp) / "design-root"

            def validate_pending(_context: object) -> None:
                events.append("pending_validated")

            def consume(_receipt: object) -> dict[str, object]:
                events.append("receipt_consumed")
                return {
                    "path": dataset_builder._expected_receipt_relative_path(
                        policy,
                        receipt_sha256=receipt_sha256,
                        consumed=True,
                    ),
                    "size_bytes": 123,
                    "sha256": receipt_sha256,
                }

            def run_transaction(*, context: object) -> dict[str, str]:
                self.assertEqual(
                    events, ["pending_validated", "receipt_consumed"]
                )
                events.append("transaction_called")
                self.assertIn(".consumed.", context.receipt_file["path"])
                return {"status": "FIXTURE_ONLY"}

            with (
                mock.patch.object(
                    dataset_builder.scientific,
                    "load_policy",
                    return_value=policy,
                ),
                mock.patch.object(dataset_builder, "_validate_model_mount_contract"),
                mock.patch.object(
                    dataset_builder,
                    "_load_and_validate_design_build_receipt",
                    return_value=(receipt, quality_policy),
                ),
                mock.patch.object(
                    dataset_builder.scientific,
                    "build_execution_context",
                    return_value=execution,
                ),
                mock.patch.object(
                    dataset_builder,
                    "_validate_pending_design_preflight_context",
                    side_effect=validate_pending,
                ),
                mock.patch.object(
                    dataset_builder,
                    "_consume_design_build_receipt",
                    side_effect=consume,
                ),
                mock.patch.object(
                    dataset_builder,
                    "_run_design_preflight_transaction",
                    side_effect=run_transaction,
                ),
            ):
                result = dataset_builder.run_design_preflight_once()
        self.assertEqual(result, {"status": "FIXTURE_ONLY"})
        self.assertEqual(
            events,
            ["pending_validated", "receipt_consumed", "transaction_called"],
        )

    def test_private_transaction_validates_authorization_before_mkdir(self) -> None:
        with (
            mock.patch.object(
                dataset_builder,
                "_validate_authorized_design_preflight_context",
                side_effect=dataset_builder.DatasetBuildError(
                    "fixture authorization rejected"
                ),
            ) as validate,
            mock.patch.object(dataset_builder.Path, "mkdir") as mkdir,
            self.assertRaisesRegex(
                dataset_builder.DatasetBuildError,
                "fixture authorization rejected",
            ),
        ):
            dataset_builder._run_design_preflight_transaction(
                context=mock.Mock()
            )
        validate.assert_called_once()
        mkdir.assert_not_called()

    def test_exact_design_receipt_binding_accepts_current_files_and_rejects_drift(
        self,
    ) -> None:
        policy = scientific.load_policy()
        quality_policy = dataset_builder.quality_policy_module.load_policy()
        overlay = policy["design_build_authorization_overlay"]
        fake_commit = "a" * 40
        fake_tree = "b" * 40
        receipt = {
            "version": overlay["receipt_version"],
            "status": dataset_builder.AUTHORIZATION_STATUS,
            "claim_boundary": dataset_builder.AUTHORIZATION_CLAIM_BOUNDARY,
            "review_final_line": overlay["required_review_final_line"],
            "review_conversation_url": "https://chatgpt.com/c/fixture-review",
            "review_response_sha256": "c" * 64,
            "reviewed_at_utc": "2026-08-15T00:00:00Z",
            "execution_mode": dataset_builder.DESIGN_EXECUTION_MODE,
            "attempt_index": policy["single_attempt_random_authority"][
                "attempt_index"
            ],
            "world_counts": policy["execution_modes"]["design_preflight"][
                "world_counts"
            ],
            "output_root": policy["execution_modes"]["design_preflight"][
                "output_root"
            ],
            "random_authority_commitment_sha256": common.canonical_sha256(
                policy["public_preflight_keys"]["design_preflight"]
            ),
            "builder_policy": dataset_builder._policy_binding(
                scientific.DEFAULT_POLICY_PATH, policy
            ),
            "quality_policy": dataset_builder._policy_binding(
                dataset_builder.quality_policy_module.DEFAULT_POLICY,
                quality_policy,
            ),
            "builder_source": dataset_builder._repo_file_binding(
                Path(dataset_builder.__file__)
            ),
            "git_commit": fake_commit,
            "git_tree": fake_tree,
        }
        receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp) / "authorization.json"
            path.write_text(
                json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    dataset_builder,
                    "_authorization_receipt_path",
                    return_value=path,
                ),
                mock.patch.object(
                    dataset_builder,
                    "_git_identity",
                    return_value=(fake_commit, fake_tree),
                ),
            ):
                verified, observed_quality = (
                    dataset_builder._load_and_validate_design_build_receipt(policy)
                )
                self.assertEqual(verified.receipt_id, receipt["canonical_self_hash"])
                self.assertEqual(observed_quality, quality_policy)

                mutated = copy.deepcopy(receipt)
                mutated["output_root"] = "reports/alternate-root-forbidden"
                mutated["canonical_self_hash"] = common.canonical_sha256(
                    {
                        key: value
                        for key, value in mutated.items()
                        if key != "canonical_self_hash"
                    }
                )
                path.write_text(
                    json.dumps(mutated, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    dataset_builder.DatasetBuildError, "binding drift"
                ):
                    dataset_builder._load_and_validate_design_build_receipt(policy)

    def test_cli_rejects_legacy_write_mode_selector(self) -> None:
        with (
            mock.patch.object(
                sys,
                "argv",
                ["builder", "--mode", "small_smoke"],
            ),
            mock.patch.object(
                dataset_builder, "run_design_preflight_once"
            ) as run_design,
            self.assertRaises(SystemExit) as raised,
        ):
            dataset_builder.main()
        self.assertEqual(raised.exception.code, 2)
        run_design.assert_not_called()


if __name__ == "__main__":
    unittest.main()
