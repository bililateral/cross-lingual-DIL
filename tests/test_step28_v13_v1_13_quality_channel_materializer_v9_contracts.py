from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common
import step28_v13_v1_13_candidate_parent as stage_parent
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_quality_channel_materializer_v9 as materializer
import step28_v13_v1_13_quality_channel_policy_v9 as quality_policy
import step28_v13_v1_13_quality_channel_views_v9 as channel
import step28_v13_v1_13_quality_structure_aggregator_v9 as structure_aggregator
import step28_v13_v1_13_scientific_common_v9 as scientific
import step28_v13_v1_13_scientific_world_v9 as world_module


class TextReadTripwire(dict):
    def __getitem__(self, key: object) -> object:
        if key in {"title", "description"}:
            raise AssertionError("neutral metadata projection read original text")
        return super().__getitem__(key)


class QualityChannelMaterializerV9Contracts(unittest.TestCase):
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
            historical_item_hashes=cls.historical.item_document_hashes,
            historical_seller_hashes=cls.historical.seller_document_hashes,
            historical_identity_hashes=cls.historical.identity_value_hashes,
            current_item_hashes=set(),
            current_seller_hashes=set(),
            current_identity_hashes=set(),
            current_item_codes=set(),
        )
        cls.processing_template = common.load_json(
            common.repo_path(world_module.CANDIDATE_TEMPLATE_RELATIVE_PATH)
        )
        cls.processing_policy = json.loads(
            common.canonical_json_bytes(cls.context.effective_policy).decode("utf-8")
        )
        cls.processing_policy["template_library"]["path"] = (
            world_module.CANDIDATE_TEMPLATE_RELATIVE_PATH
        )
        cls.processing_policy["template_library"]["sha256"] = (
            world_module.CANDIDATE_TEMPLATE_SHA256
        )
        cls.safe_library = world_module._candidate_safe_library(
            policy=cls.context.effective_policy,
            template=cls.template,
            fixture=cls.fixture,
            split="train",
        )
        style_rows = stage_parent._effective_style_rows(
            policy=cls.context.effective_policy,
            template=cls.template,
            mode=cls.context.base_mode,
            world=cls.accepted.world,
        )
        cls.effective_styles = {
            str(row["seller_uid"]): dict(row["style_factors"])
            for row in style_rows
        }
        private = cls.accepted.world["private"]
        cls.materialized = materializer.materialize_label_free_channel_views(
            processing_policy=cls.processing_policy,
            profile_policy=cls.context.effective_policy,
            mode=cls.context.base_mode,
            split="train",
            processing_template=cls.processing_template,
            safe_library=cls.safe_library,
            fixture=cls.fixture,
            world_uid=cls.accepted.world_uid,
            public_sellers=cls.accepted.world["public"]["sellers"],
            public_items=cls.accepted.world["public"]["items"],
            complete_model_pair_endpoints=cls.accepted.world["public"][
                "complete_model_pair_endpoints"
            ],
            render_asts=private["render_asts"],
            identity_slots_audit=private["identity_slots_audit"],
            noise_slots_audit=private["noise_slots_audit"],
            override_audit=private["override_audit"],
            effective_styles=cls.effective_styles,
            full_redacted_items=cls.accepted.redacted_items,
            full_seller_profiles=cls.accepted.seller_profiles,
        )

    def test_all_authority_sections_are_semantically_pinned(self) -> None:
        baseline = json.loads(
            quality_policy.DEFAULT_POLICY.read_text(encoding="utf-8")
        )
        for authority_name in sorted(baseline["authority_fields"]):
            with self.subTest(authority_name=authority_name):
                mutated = copy.deepcopy(baseline)
                mutated["authority_fields"][authority_name][
                    "authority_section"
                ] = "篡改章节"
                payload = dict(mutated)
                payload.pop("canonical_self_hash")
                mutated_self_hash = hashlib.sha256(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                mutated["canonical_self_hash"] = mutated_self_hash
                with mock.patch.object(
                    quality_policy,
                    "EXPECTED_POLICY_SELF_HASH",
                    mutated_self_hash,
                ):
                    with self.assertRaisesRegex(
                        quality_policy.QualityChannelPolicyError,
                        "Field authority source does not match",
                    ):
                        quality_policy.validate_policy(mutated, check_pins=False)

    def test_public_entry_has_no_truth_or_complete_world_parameter(self) -> None:
        parameters = set(
            inspect.signature(
                materializer.materialize_label_free_channel_views
            ).parameters
        )
        self.assertFalse(
            parameters
            & {
                "world",
                "pair_labels",
                "labels",
                "controller_membership",
                "qrels",
                "positive_targets",
                "negative_flags",
                "quality_results",
            }
        )
        neutral_parameters = set(
            inspect.signature(
                materializer._neutralize_without_original_code_values
            ).parameters
        )
        self.assertNotIn("code_by_item", neutral_parameters)
        self.assertNotIn("render_asts", neutral_parameters)
        self.assertNotIn("public_items", neutral_parameters)
        self.assertIn("render_asts_without_codes", neutral_parameters)
        self.assertIn("public_item_projection", neutral_parameters)
        observation = materializer._forbidden_access_observation(
            entrypoint_parameter_names=tuple(
                inspect.signature(
                    materializer.materialize_label_free_channel_views
                ).parameters
            ),
            accessed_capability_names=(),
        )
        self.assertEqual(
            dict(observation.capability_mounted),
            {
                name: False
                for name in materializer.FORBIDDEN_ACCESS_CAPABILITY_FIELDS
            },
        )
        self.assertEqual(
            dict(observation.read_counts),
            {
                name: 0
                for name in materializer.FORBIDDEN_ACCESS_CAPABILITY_FIELDS
            },
        )
        with self.assertRaisesRegex(
            materializer.QualityChannelMaterializationError,
            "forbidden quality/truth capability",
        ):
            materializer._forbidden_access_observation(
                entrypoint_parameter_names=(
                    *inspect.signature(
                        materializer.materialize_label_free_channel_views
                    ).parameters,
                    "candidate_quality_result",
                ),
                accessed_capability_names=(),
            )

    def test_neutral_input_capability_never_reads_original_text_values(self) -> None:
        source = TextReadTripwire(
            {
                "world_uid": "world",
                "seller_uid": "seller",
                "item_uid": "item",
                "time_bucket": 0,
                "category": "category",
                "title": "must-not-be-read",
                "description": "must-not-be-read",
            }
        )
        projected = materializer._project_neutral_item_metadata((source,))
        self.assertIsInstance(projected, materializer.NeutralItemProjection)
        self.assertEqual(len(projected.rows), 1)
        self.assertIsInstance(projected.rows[0], materializer.NeutralItemMetadata)
        self.assertEqual(projected.source_value_read_count, 5)
        self.assertEqual(projected.forbidden_value_read_count, 0)
        self.assertEqual(
            dict(projected.source_value_read_counts),
            {field: 1 for field in materializer.NEUTRAL_ITEM_METADATA_FIELDS},
        )
        self.assertFalse(
            {"title", "description"}
            & set(materializer.NeutralItemMetadata.__dataclass_fields__)
        )

    def test_registered_item_spans_are_derived_from_ast_rendering(self) -> None:
        source = inspect.getsource(materializer._registered_item_spans)
        self.assertIn("_render_registered_carrier", source)
        self.assertNotIn("RAW_CODE_RE.finditer", source)

    def test_neutral_code_family_is_unique_and_freezes_both_derived_symbols(
        self,
    ) -> None:
        codes = tuple(materializer._neutral_render_code(index) for index in range(224))
        self.assertEqual(codes[0], materializer.NEUTRAL_RENDER_CODE)
        self.assertEqual(len(codes), len(set(codes)))
        self.assertTrue(all(code.endswith("BA") for code in codes))
        self.assertTrue(all(channel.CODE_RE.fullmatch(code) for code in codes))
        self.assertEqual(
            {
                materializer.pure_renderer._title_modifier(
                    code, self.safe_library
                )
                for code in codes
            },
            {"常规款"},
        )

    def test_three_item_and_profile_views_close_on_one_real_generated_world(self) -> None:
        full_items = self.accepted.redacted_items
        views = self.materialized
        self.assertEqual(len(views.masked_redacted_items), len(full_items))
        self.assertEqual(len(views.neutral_redacted_items), len(full_items))
        self.assertEqual(len(views.masked_seller_profiles), 28)
        self.assertEqual(len(views.neutral_seller_profiles), 28)
        self.assertEqual(len(views.text_probe_eligibility_input), 378)
        self.assertEqual(
            sum(
                bool(row["text_probe_eligible"])
                for row in views.text_probe_eligibility_input
            ),
            372,
        )
        self.assertTrue(
            all(
                set(row)
                == {"world_uid", "canonical_pair_uid", "text_probe_eligible"}
                for row in views.text_probe_eligibility_input
            )
        )
        excluded = {
            str(row["canonical_pair_uid"])
            for row in views.text_probe_eligibility_input
            if not row["text_probe_eligible"]
        }
        self.assertEqual(
            excluded,
            {
                str(row["canonical_pair_uid"])
                for row in self.accepted.world["private"]["override_audit"]
            },
        )
        full_keys = {str(row["item_uid"]) for row in full_items}
        self.assertEqual(
            full_keys, {str(row["item_uid"]) for row in views.masked_redacted_items}
        )
        self.assertEqual(
            full_keys, {str(row["item_uid"]) for row in views.neutral_redacted_items}
        )
        for rows in (
            views.masked_redacted_items,
            views.neutral_redacted_items,
        ):
            for row in rows:
                self.assertIsNone(
                    channel.RAW_CODE_RE.search(
                        str(row["title"]) + str(row["description"])
                    )
                )
        for profiles in (
            views.masked_seller_profiles,
            views.neutral_seller_profiles,
        ):
            for row in profiles:
                for field in channel.PROFILE_FIELDS:
                    self.assertIsNone(channel.RAW_CODE_RE.search(str(row[field])))

    def test_clone_titles_and_nonempty_patterns_survive_both_counterfactuals(self) -> None:
        masked = {
            str(row["item_uid"]): row
            for row in self.materialized.masked_redacted_items
        }
        neutral = {
            str(row["item_uid"]): row
            for row in self.materialized.neutral_redacted_items
        }
        full = {str(row["item_uid"]): row for row in self.accepted.redacted_items}
        for item_uid in full:
            for field in ("title", "description"):
                self.assertEqual(bool(full[item_uid][field]), bool(masked[item_uid][field]))
                self.assertEqual(bool(full[item_uid][field]), bool(neutral[item_uid][field]))
        for row in self.accepted.world["private"]["override_audit"]:
            if row["override_kind"] != "exact_title_clone":
                continue
            left = str(row["item_uid_left"])
            right = str(row["item_uid_right"])
            self.assertEqual(masked[left]["title"], masked[right]["title"])
            self.assertEqual(neutral[left]["title"], neutral[right]["title"])

    def test_neutral_projection_registers_derived_nodes_and_base_templates(self) -> None:
        receipt = self.materialized.channel_structure_audit["neutral_receipt"]
        nodes = receipt["non_code_projection_nodes"]
        mappings = receipt["per_item_template_mapping"]
        self.assertEqual(len(nodes), len(self.accepted.redacted_items))
        self.assertEqual(len(mappings), len(self.accepted.redacted_items))
        self.assertEqual(
            receipt["neutral_code_family_count"], len(self.accepted.redacted_items)
        )
        self.assertTrue(
            all(row["derived_title_modifier_value"] == "常规款" for row in nodes)
        )
        self.assertEqual(
            receipt["neutralizer_input_capability"],
            "NeutralItemProjection[NeutralItemMetadata]",
        )
        self.assertEqual(
            tuple(receipt["neutralizer_input_fields"]),
            materializer.NEUTRAL_ITEM_METADATA_FIELDS,
        )
        commitment = receipt["non_code_projection_commitment"]
        self.assertIs(commitment["verified"], True)
        self.assertEqual(
            commitment["source_sha256"], commitment["neutral_sha256"]
        )
        self.assertIs(commitment["absolute_offsets_compared"], False)
        self.assertIs(commitment["relative_ast_boundaries_compared"], True)
        self.assertEqual(receipt["original_code_value_read_count"], 0)
        self.assertEqual(
            receipt["neutral_metadata_source_value_read_count"],
            len(self.accepted.redacted_items)
            * len(materializer.NEUTRAL_ITEM_METADATA_FIELDS),
        )
        self.assertIs(
            receipt["neutral_profiles_recomputed_after_code_collapse"], True
        )
        self.assertTrue(
            all(
                row["derived_title_modifier_node_id"]
                and row["conditional_english_tag_visibility_node_id"]
                and row["english_tag_value_node_id"]
                for row in nodes
            )
        )
        clone_targets = {
            str(row["item_uid_right"]): str(row["item_uid_left"])
            for row in self.accepted.world["private"]["override_audit"]
            if row["override_kind"] == "exact_title_clone"
        }
        nodes_by_item = {str(row["item_uid"]): row for row in nodes}
        for target, source in clone_targets.items():
            self.assertEqual(
                nodes_by_item[target]["visible_title_source_item_uid"], source
            )
        for row in mappings:
            neutral_index = int(row["neutral_description_template_id"])
            is_carrier = "{code}" in str(
                self.safe_library["description_skeletons"][neutral_index]
            )
            self.assertEqual(row["neutral_description_is_code_carrier"], is_carrier)

    def test_neutral_profiles_are_recomputed_from_collapsed_item_bytes(self) -> None:
        profile_safe = materializer.production.build_profile_safe_items(
            self.processing_policy,
            items=self.accepted.world["public"]["items"],
            redacted_items=self.materialized.neutral_redacted_items,
        )
        expected, audit = materializer.profiles_module.build_world_profiles(
            self.context.effective_policy,
            mode=self.context.base_mode,
            split="train",
            sellers=self.accepted.world["public"]["sellers"],
            items=profile_safe,
        )
        self.assertIs(audit["labels_or_private_structure_read"], False)
        self.assertEqual(
            common.canonical_json_bytes(expected),
            common.canonical_json_bytes(self.materialized.neutral_seller_profiles),
        )

    def test_all_256_derived_states_collapse_in_actual_neutral_mount(self) -> None:
        private = self.accepted.world["private"]
        source_asts = tuple(private["render_asts"])
        target_item_uid = str(source_asts[0]["item_uid"])

        def build_model_mount(original_code: str) -> bytes:
            varied_asts = copy.deepcopy(source_asts)
            target_count = 0
            for row in varied_asts:
                if str(row["item_uid"]) == target_item_uid:
                    row["code"] = original_code
                    target_count += 1
            self.assertEqual(target_count, 1)
            items, profiles, receipt = (
                materializer._materialize_neutral_from_render_asts(
                    processing_policy=self.processing_policy,
                    profile_policy=self.context.effective_policy,
                    mode=self.context.base_mode,
                    split="train",
                    processing_template=self.processing_template,
                    safe_library=self.safe_library,
                    fixture=self.fixture,
                    world_uid=self.accepted.world_uid,
                    public_sellers=self.accepted.world["public"]["sellers"],
                    public_items=self.accepted.world["public"]["items"],
                    render_asts=varied_asts,
                    identity_slots_audit=private["identity_slots_audit"],
                    noise_slots_audit=private["noise_slots_audit"],
                    override_audit=private["override_audit"],
                    effective_styles=self.effective_styles,
                )
            )
            self.assertEqual(receipt["original_code_value_read_count"], 0)
            return common.canonical_json_bytes(
                {"items": items, "profiles": profiles}
            )

        channel.assert_all_derived_symbol_states_collapse(
            build_model_mount=build_model_mount
        )
        mount_bytes = build_model_mount("Q" + ("A" * 10))
        for second_last in channel.CODE_ALPHABET:
            for last in channel.CODE_ALPHABET:
                original_code = "Q" + ("A" * 8) + second_last + last
                self.assertNotIn(original_code.encode("ascii"), mount_bytes)

    def test_derived_state_audit_detects_a_leaking_mount(self) -> None:
        with self.assertRaisesRegex(
            channel.QualityChannelViewError, "depends on a derived-symbol state"
        ):
            channel.assert_all_derived_symbol_states_collapse(
                build_model_mount=lambda original_code: original_code[-2:].encode(
                    "ascii"
                )
            )

    def test_public_probe_rows_build_exact_2992_feature_vectors(self) -> None:
        rows = self.materialized.public_code_probe_input
        self.assertEqual(len(rows), 28)
        converted: list[channel.SellerCodeView] = []
        for row in rows:
            self.assertNotIn("override_kind", row)
            self.assertNotIn("clone_direction", row)
            self.assertEqual(
                set(row),
                {
                    "world_uid",
                    "seller_uid",
                    "owned_codes",
                    "item_occurrences",
                    "profile_occurrences",
                    "numeric_profile_deltas",
                },
            )
            self.assertTrue(
                all(
                    set(value) == {"field", "code", "is_own"}
                    for value in row["item_occurrences"]
                )
            )
            self.assertTrue(
                all(
                    set(value) == {"field", "code", "is_own"}
                    for value in row["profile_occurrences"]
                )
            )
            item_occurrences = tuple(
                channel.CodeOccurrence(
                    code=str(value["code"]),
                    field=str(value["field"]),
                    is_own=bool(value["is_own"]),
                )
                for value in row["item_occurrences"]
            )
            profile_occurrences = {
                field: tuple(
                    channel.CodeOccurrence(
                        code=str(value["code"]),
                        field=field,
                        is_own=bool(value["is_own"]),
                    )
                    for value in row["profile_occurrences"]
                    if value["field"] == field
                )
                for field in channel.PROFILE_FIELDS
            }
            converted.append(
                channel.SellerCodeView(
                    owned_codes=tuple(row["owned_codes"]),
                    visible_occurrences=item_occurrences,
                    profile_occurrences=profile_occurrences,
                    numeric_profile_deltas=dict(row["numeric_profile_deltas"]),
                )
            )
        features = channel.build_public_code_pair_features(converted[0], converted[1])
        reverse = channel.build_public_code_pair_features(converted[1], converted[0])
        self.assertEqual(len(features), 2992)
        self.assertEqual(features, reverse)

    def test_structure_receipt_uses_computed_zero_tolerance_counts(self) -> None:
        receipt = self.materialized.channel_structure_audit
        for field in (
            "registered_visible_occurrence_multiset_difference_count",
            "literal_code_hits_in_masked",
            "literal_code_hits_in_neutralized",
            "unregistered_code_hits",
            "unregistered_clone_foreign_code_hits",
            "view_keyset_difference_count",
            "neutralized_legal_code_permutation_byte_difference_count",
            "audit_truth_open_count",
            "audit_truth_read_count",
            "audit_truth_materialized_row_count",
            "generator_quality_result_read_count",
            "candidate_quality_result_read_count",
            "view_builder_quality_result_read_count",
        ):
            self.assertEqual(receipt[field], 0, field)
        self.assertEqual(
            receipt["forbidden_capability_mounted"],
            {
                name: False
                for name in materializer.FORBIDDEN_ACCESS_CAPABILITY_FIELDS
            },
        )
        self.assertEqual(
            receipt["registered_visible_occurrence_expected_count"],
            receipt["registered_visible_occurrence_actual_count"],
        )
        contaminated = [dict(row) for row in self.materialized.masked_redacted_items]
        contaminated[0]["title"] += " QAAAAAAAAAA"
        hits = sum(
            len(channel.RAW_CODE_RE.findall(str(row[field])))
            for row in contaminated
            for field in ("title", "description")
        )
        self.assertGreater(hits, 0)

    def test_structure_profile_hashes_bind_exact_persisted_projection(self) -> None:
        receipt = self.materialized.channel_structure_audit
        cases = (
            (
                self.accepted.seller_profiles,
                "full_profile_sha256",
            ),
            (
                self.materialized.masked_seller_profiles,
                "masked_profile_sha256",
            ),
            (
                self.materialized.neutral_seller_profiles,
                "neutral_profile_sha256",
            ),
        )
        for profiles, field in cases:
            with self.subTest(field=field):
                projected = scientific.project_model_seller_profiles(profiles)
                self.assertEqual(
                    receipt[field],
                    common.canonical_sha256(projected),
                )
                self.assertNotEqual(
                    receipt[field],
                    common.canonical_sha256(profiles),
                    "regression fixture must retain wider internal Step3 fields",
                )

        augmented = copy.deepcopy(self.accepted.seller_profiles[0])
        augmented["private_audit_marker"] = "must-not-enter-model-view"
        self.assertEqual(
            scientific.project_model_seller_profile(
                self.accepted.seller_profiles[0]
            ),
            scientific.project_model_seller_profile(augmented),
        )
        self.assertNotEqual(
            common.canonical_sha256(self.accepted.seller_profiles[0]),
            common.canonical_sha256(augmented),
        )

    def test_persisted_real_structure_receipt_closes_in_aggregator(self) -> None:
        def code_for_ordinal(value: int) -> str:
            return "Q" + "".join(
                chr(ord("A") + int(symbol, 16))
                for symbol in f"{value:010x}"
            )

        public_rows_by_split: dict[str, list[dict[str, object]]] = {}
        structure_rows_by_split: dict[str, list[dict[str, object]]] = {}
        item_count = int(self.materialized.channel_structure_audit["item_count"])
        for split_index, split in enumerate(structure_aggregator.SPLITS):
            world_uid = f"persisted_{split}_world"
            public_rows = json.loads(
                json.dumps(
                    self.materialized.public_code_probe_input,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            ordinal = split_index * 1000
            for row in public_rows:
                row["world_uid"] = world_uid
                owned_count = len(row["owned_codes"])
                row["owned_codes"] = [
                    code_for_ordinal(value)
                    for value in range(ordinal, ordinal + owned_count)
                ]
                ordinal += owned_count
            self.assertEqual(ordinal - split_index * 1000, item_count)
            structure_row = json.loads(
                json.dumps(
                    self.materialized.channel_structure_audit,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            structure_row["world_uid"] = world_uid
            public_rows_by_split[split] = public_rows
            structure_rows_by_split[split] = [structure_row]
        receipt = structure_aggregator.aggregate_fixture_structure(
            public_rows_by_split=public_rows_by_split,
            structure_rows_by_split=structure_rows_by_split,
            expected_world_counts={split: 1 for split in structure_aggregator.SPLITS},
            expected_sellers_per_world=28,
        )
        self.assertEqual(receipt["gate_failures"], [])

    def test_real_world_covers_required_carrier_and_clone_cases(self) -> None:
        clone_rows = [
            row
            for row in self.accepted.world["private"]["override_audit"]
            if row["override_kind"] == "exact_title_clone"
        ]
        self.assertTrue(clone_rows)
        source_ast = dict(self.accepted.world["private"]["render_asts"][0])
        title_index = next(
            index
            for index, value in enumerate(self.safe_library["title_skeletons"])
            if "{code}" in str(value)
        )
        description_index = next(
            index
            for index, value in enumerate(
                self.safe_library["description_skeletons"]
            )
            if "{code}" in str(value)
        )
        all_codes = {str(source_ast["code"])}
        for title_nonempty, description_nonempty in (
            (True, False),
            (False, True),
            (True, True),
        ):
            ast = dict(source_ast)
            ast["title_nonempty"] = title_nonempty
            ast["description_nonempty"] = description_nonempty
            ast["title_skeleton_index"] = title_index
            ast["description_skeleton_index"] = description_index
            title, _code, _index = materializer._render_registered_carrier(
                field="title",
                ast=ast,
                safe_library=self.safe_library,
                effective_styles=self.effective_styles,
            )
            description, _code, _index = materializer._render_registered_carrier(
                field="description",
                ast=ast,
                safe_library=self.safe_library,
                effective_styles=self.effective_styles,
            )
            masked, occurrences = materializer._registered_item_spans(
                row={
                    "item_uid": str(ast["item_uid"]),
                    "title": title,
                    "description": description,
                },
                ast=ast,
                ast_by_item={str(ast["item_uid"]): ast},
                clone_source_by_target={},
                safe_library=self.safe_library,
                all_codes=all_codes,
                effective_styles=self.effective_styles,
            )
            self.assertEqual(bool(masked["title"]), title_nonempty)
            self.assertEqual(bool(masked["description"]), description_nonempty)
            expected_occurrences = int(title_nonempty) + int(description_nonempty)
            self.assertEqual(len(occurrences), expected_occurrences)


if __name__ == "__main__":
    unittest.main()
