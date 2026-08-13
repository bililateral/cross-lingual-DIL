from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common
import step28_v13_v1_13_scientific_common_v8 as scientific
import step28_v13_v1_13_scientific_dataset_builder_v8 as dataset_builder
import step28_v13_v1_13_scientific_world_v8 as world_module
import step28_v13_v1_13_pure_natural_renderer_v8 as pure_renderer_v8
import step7_v3_1_source_data as step7_source


class ScientificBuilderPolicyTests(unittest.TestCase):
    def test_policy_and_four_split_small_smoke_close(self) -> None:
        policy = scientific.load_policy()
        context = scientific.build_execution_context(
            policy, execution_mode="small_smoke"
        )
        self.assertEqual(len(context.world_records), 4)
        self.assertEqual(
            {split: sum(row["split"] == split for row in context.world_records)
             for split in scientific.SPLITS},
            {split: 1 for split in scientific.SPLITS},
        )
        self.assertTrue(context.scientific_use_forbidden)
        dataset_builder._validate_model_mount_contract(policy)

    def test_formal_generation_fails_before_seed_ceremony(self) -> None:
        policy = scientific.load_policy()
        with self.assertRaisesRegex(
            scientific.ScientificBuilderError, "one-shot seed ceremony"
        ):
            scientific.build_execution_context(policy, execution_mode="formal")

    def test_v8_design_scale_and_single_attempt_are_exact(self) -> None:
        policy = scientific.load_policy()
        self.assertEqual(
            policy["execution_modes"]["design_preflight"]["world_counts"],
            {"train": 500, "development": 500, "audit_a": 2, "audit_b": 2},
        )
        context = scientific.build_execution_context(
            policy, execution_mode="design_preflight"
        )
        self.assertEqual(len(context.world_records), 1004)
        self.assertEqual(
            policy["single_attempt_random_authority"],
            {
                "attempt_index": 1,
                "total_world_count": 1004,
                "alternate_authority_forbidden": True,
                "alternate_output_root_forbidden": True,
                "build_execution_failure_reuses_same_authority": True,
                "audit_execution_failure_reuses_same_dataset_root": True,
                "data_quality_failure_closes_v8_permanently": True,
            },
        )

    def test_v8_policy_rejects_scale_fallback_and_second_attempt(self) -> None:
        for mutation in ("scale", "attempt"):
            policy = scientific.load_policy()
            if mutation == "scale":
                policy["execution_modes"]["design_preflight"]["world_counts"][
                    "development"
                ] = 200
            else:
                policy["single_attempt_random_authority"]["attempt_index"] = 2
            policy.pop("canonical_self_hash")
            policy["canonical_self_hash"] = common.canonical_sha256(policy)
            with self.assertRaises(scientific.ScientificBuilderError):
                scientific.validate_policy(policy)

    def test_claim_boundary_and_v8_renderer_binding_are_exact(self) -> None:
        policy = scientific.load_policy()
        self.assertEqual(
            policy["claim_boundary"], scientific.EXPECTED_CLAIM_BOUNDARY
        )
        self.assertIs(world_module.pure_renderer, pure_renderer_v8)
        renderer_pin = policy["implementation"]["pure_natural_renderer"]
        self.assertEqual(
            common.repo_path(renderer_pin["path"]).resolve(),
            Path(pure_renderer_v8.__file__).resolve(),
        )

        mutated = copy.deepcopy(policy)
        mutated["claim_boundary"] += " Drift."
        mutated.pop("canonical_self_hash")
        mutated["canonical_self_hash"] = common.canonical_sha256(mutated)
        with self.assertRaisesRegex(
            scientific.ScientificBuilderError, "claim boundary"
        ):
            scientific.validate_policy(mutated)

    def test_v8_authorities_do_not_reuse_v7_authorities(self) -> None:
        current = scientific.load_policy()
        retired = common.load_json(
            ROOT
            / "schema"
            / "step28_v13_v1_13_scientific_dataset_builder_policy.json"
        )

        def values(policy: dict) -> set[str]:
            output: set[str] = set()
            for block in policy["public_preflight_keys"].values():
                output.update(
                    value
                    for name, value in block.items()
                    if name != "rewire_key_hexes"
                )
                output.update(block["rewire_key_hexes"])
            return output

        self.assertFalse(values(current) & values(retired))
        mutated = copy.deepcopy(current)
        mutated["public_preflight_keys"]["small_smoke"]["id_key_hex"] = next(
            iter(values(retired))
        )
        mutated.pop("canonical_self_hash")
        mutated["canonical_self_hash"] = common.canonical_sha256(mutated)
        with self.assertRaisesRegex(
            scientific.ScientificBuilderError, "Retired preflight authority"
        ):
            scientific.validate_policy(mutated)

    def test_preflight_random_authorities_are_disjoint(self) -> None:
        policy = scientific.load_policy()
        observed: set[str] = set()
        for mode in scientific.DESIGN_MODES:
            block = policy["public_preflight_keys"][mode]
            values = {
                value
                for key, value in block.items()
                if key != "rewire_key_hexes"
            } | set(block["rewire_key_hexes"])
            self.assertFalse(values & observed)
            observed.update(values)
        base = common.load_json(common.repo_path(policy["base_dataset_policy"]["path"]))
        self.assertFalse(
            observed & scientific._collect_random_authorities(base["randomness"])
        )

    def test_preflight_authority_reuse_with_base_fails_closed(self) -> None:
        policy = scientific.load_policy()
        base = common.load_json(common.repo_path(policy["base_dataset_policy"]["path"]))
        reused = sorted(scientific._collect_random_authorities(base["randomness"]))[0]
        policy["public_preflight_keys"]["small_smoke"][
            "document_variation_key_hex"
        ] = reused
        policy.pop("canonical_self_hash")
        policy["canonical_self_hash"] = common.canonical_sha256(policy)
        with self.assertRaisesRegex(
            scientific.ScientificBuilderError, "reuses a pinned base authority"
        ):
            scientific.validate_policy(policy)

    def test_v8_renderer_maps_are_complete_bijections_and_response_closed(self) -> None:
        policy = scientific.load_policy()
        context = scientific.build_execution_context(
            policy, execution_mode="design_preflight"
        )
        template, fixture, _style_profile = scientific.load_release_inputs(context)
        translation = str.maketrans(
            template["renderer_contract"]["traditional_substitutions"]
        )

        def response(value: object) -> bool:
            text = str(value)
            return text.translate(translation) != text

        for split in scientific.SPLITS:
            with self.subTest(split=split):
                library = world_module.stage_variation._safe_library(
                    base_policy=context.effective_policy,
                    template=template,
                    fixture=fixture,
                    split=split,
                )
                maps = pure_renderer_v8._build_v8_permutation_maps(
                    candidate_key=b"\x19" * 32,
                    library=library,
                )

                domains = {
                    "category": list(library["categories"]),
                    "attribute": list(library["attributes"]),
                    "delivery": list(library["delivery"]),
                    "service": list(library["service"]),
                    "title_skeleton": list(range(len(library["title_skeletons"]))),
                    "description_skeleton": list(
                        range(len(library["description_skeletons"]))
                    ),
                    "noise_template": list(
                        range(len(library["must_ignore_templates"]))
                    ),
                    "noise_value": list(library["must_ignore_values"]),
                }
                for name, domain in domains.items():
                    self.assertEqual(set(maps[name]), set(domain))
                    self.assertEqual(set(maps[name].values()), set(domain))

                for source, target in maps["category"].items():
                    self.assertEqual(response(source), response(target))
                    self.assertEqual(
                        tuple(
                            response(value)
                            for value in library["category_products"][source]
                        ),
                        tuple(
                            response(value)
                            for value in library["category_products"][target]
                        ),
                    )
                for name in ("attribute", "delivery", "service", "noise_value"):
                    for source, target in maps[name].items():
                        self.assertEqual(response(source), response(target))
                self.assertEqual(
                    tuple(
                        tuple(orbit)
                        for orbit in policy["candidate_selection"][
                            "attribute_variation_repair"
                        ]["semantic_orbits"]
                    ),
                    pure_renderer_v8.ATTRIBUTE_SEMANTIC_ORBITS,
                )
                pure_renderer_v8._validate_attribute_map(
                    maps["attribute"], library=library
                )
                self.assertIn(
                    maps["attribute"]["标准版"],
                    {"标准版", "组合版", "多规格"},
                )
                self.assertNotEqual(
                    maps["attribute"]["标准版"],
                    maps["attribute"]["组合版"],
                )
                for name, values_field in (
                    ("title_skeleton", "title_skeletons"),
                    ("description_skeleton", "description_skeletons"),
                    ("noise_template", "must_ignore_templates"),
                ):
                    for source, target in maps[name].items():
                        source_text = library[values_field][source]
                        target_text = library[values_field][target]
                        self.assertEqual(response(source_text), response(target_text))
                        self.assertEqual(
                            pure_renderer_v8._placeholder_signature(source_text),
                            pure_renderer_v8._placeholder_signature(target_text),
                        )
                for size, mapping in maps["product_index"].items():
                    self.assertEqual(set(mapping), set(range(size)))
                    self.assertEqual(set(mapping.values()), set(range(size)))
                    categories = sorted(
                        (
                            category
                            for category, products in library[
                                "category_products"
                            ].items()
                            if len(products) == size
                        ),
                        key=lambda value: value.encode("utf-8"),
                    )
                    for source, target in mapping.items():
                        self.assertEqual(
                            tuple(
                                response(
                                    library["category_products"][category][source]
                                )
                                for category in categories
                            ),
                            tuple(
                                response(
                                    library["category_products"][category][target]
                                )
                                for category in categories
                            ),
                        )

    def test_attribute_repair_does_not_perturb_existing_map_domains(self) -> None:
        expected = {
            "train": "d8f40430a275265429a5ba111b7146846b6f234deedb55b7b355d7c40bf75869",
            "development": "ae4bf4e6078cf8fd3a099f87742fbe57f625133c9035d649bfbbaecc49a9d378",
            "audit_a": "1422d84397aaa51f8051ce8f8a4d200086f174e2684d7408afab57b01b75ab47",
            "audit_b": "caf20426797d42d42bfb1e0781ed4c490528103afdc24ab68138f9364c8051ad",
        }
        policy = scientific.load_policy()
        context = scientific.build_execution_context(
            policy, execution_mode="design_preflight"
        )
        template, fixture, _style_profile = scientific.load_release_inputs(context)
        for split in scientific.SPLITS:
            library = world_module.stage_variation._safe_library(
                base_policy=context.effective_policy,
                template=template,
                fixture=fixture,
                split=split,
            )
            maps = pure_renderer_v8._build_v8_permutation_maps(
                candidate_key=b"\x19" * 32,
                library=library,
            )
            maps.pop("attribute")
            serializable = {
                name: (
                    {
                        str(size): {
                            str(source): target
                            for source, target in mapping.items()
                        }
                        for size, mapping in value.items()
                    }
                    if name == "product_index"
                    else {str(source): target for source, target in value.items()}
                )
                for name, value in maps.items()
            }
            self.assertEqual(common.canonical_sha256(serializable), expected[split])

    def test_three_state_repair_does_not_perturb_old_domains_for_target_keys(
        self,
    ) -> None:
        policy = scientific.load_policy()
        context = scientific.build_execution_context(
            policy, execution_mode="design_preflight"
        )
        template, fixture, _style_profile = scientific.load_release_inputs(context)
        library = world_module.stage_variation._safe_library(
            base_policy=context.effective_policy,
            template=template,
            fixture=fixture,
            split="train",
        )
        record = next(
            row
            for row in context.world_records
            if row["split"] == "train" and row["split_ordinal"] == 159
        )
        old_orbits = (
            ("标准版", "组合版"),
            ("轻量版", "更新版"),
            ("多规格",),
            *pure_renderer_v8.ATTRIBUTE_SEMANTIC_ORBITS[2:],
        )
        for candidate_index in range(32):
            key = dataset_builder.collision.derive_candidate_key(
                context.document_variation_key,
                split="train",
                world_uid=record["world_uid"],
                candidate_index=candidate_index,
            )
            current = pure_renderer_v8._build_v8_permutation_maps(
                candidate_key=key,
                library=library,
            )
            with mock.patch.object(
                pure_renderer_v8, "ATTRIBUTE_SEMANTIC_ORBITS", old_orbits
            ):
                old = pure_renderer_v8._build_v8_permutation_maps(
                    candidate_key=key,
                    library=library,
                )
            self.assertEqual(
                {name: value for name, value in current.items() if name != "attribute"},
                {name: value for name, value in old.items() if name != "attribute"},
            )
            for attribute in (
                "轻量版",
                "更新版",
                "可选配色",
                "分批交付",
                "附使用说明",
                "支持自选参数",
                "含基础售后",
            ):
                self.assertEqual(
                    current["attribute"][attribute], old["attribute"][attribute]
                )

    def test_attribute_map_rejects_cross_semantic_orbit(self) -> None:
        policy = scientific.load_policy()
        context = scientific.build_execution_context(
            policy, execution_mode="design_preflight"
        )
        template, fixture, _style_profile = scientific.load_release_inputs(context)
        library = world_module.stage_variation._safe_library(
            base_policy=context.effective_policy,
            template=template,
            fixture=fixture,
            split="train",
        )
        mapping = {value: value for value in library["attributes"]}
        mapping["标准版"] = "轻量版"
        mapping["轻量版"] = "标准版"
        with self.assertRaisesRegex(
            pure_renderer_v8.PureNaturalVariationError,
            "semantic orbit",
        ):
            pure_renderer_v8._validate_attribute_map(mapping, library=library)

    def test_attribute_rotation_is_deterministic_and_registry_blind(self) -> None:
        policy = scientific.load_policy()
        spec = policy["candidate_selection"]["attribute_variation_repair"]
        self.assertEqual(spec["shared_sequential_rng_reads"], 0)
        self.assertFalse(spec["historical_or_current_registry_reads"])
        self.assertFalse(spec["labels_or_model_scores_read"])
        context = scientific.build_execution_context(
            policy, execution_mode="design_preflight"
        )
        template, fixture, _style_profile = scientific.load_release_inputs(context)
        library = world_module.stage_variation._safe_library(
            base_policy=context.effective_policy,
            template=template,
            fixture=fixture,
            split="train",
        )
        key = hashlib.sha256(b"fixed-attribute-test-key").digest()
        first = pure_renderer_v8._attribute_rotation_map(
            candidate_key=key, library=library
        )
        second = pure_renderer_v8._attribute_rotation_map(
            candidate_key=key, library=copy.deepcopy(library)
        )
        self.assertEqual(first, second)
        mappings = [
            pure_renderer_v8._attribute_rotation_map(
                candidate_key=hashlib.sha256(
                    f"fixed-attribute-test-key-{index}".encode("ascii")
                ).digest(),
                library=library,
            )
            for index in range(32)
        ]
        self.assertEqual(
            {mapping["标准版"] for mapping in mappings},
            {"标准版", "组合版", "多规格"},
        )
        self.assertEqual(
            {mapping["轻量版"] for mapping in mappings},
            {"轻量版", "更新版"},
        )

    def test_three_state_content_slot_all_directions_and_contexts_close(self) -> None:
        policy = scientific.load_policy()
        spec = policy["candidate_selection"]["attribute_variation_repair"]
        self.assertEqual(
            spec["semantic_slot_id"],
            "product_version_or_specification_attribute",
        )
        self.assertFalse(spec["strict_synonymy_required"])
        self.assertEqual(
            spec["candidate_semantics"],
            "admissible_alternative_realizations_within_one_content_slot",
        )
        self.assertEqual(
            pure_renderer_v8.ATTRIBUTE_SEMANTIC_ORBITS[0],
            ("标准版", "组合版", "多规格"),
        )
        context = scientific.build_execution_context(
            policy, execution_mode="design_preflight"
        )
        template, fixture, _style_profile = scientific.load_release_inputs(context)
        translation = str.maketrans(
            template["renderer_contract"]["traditional_substitutions"]
        )
        for split in scientific.SPLITS:
            library = world_module.stage_variation._safe_library(
                base_policy=context.effective_policy,
                template=template,
                fixture=fixture,
                split=split,
            )
            values = tuple(library["attributes"])
            for shift in range(3):
                mapping = {value: value for value in values}
                orbit = pure_renderer_v8.ATTRIBUTE_SEMANTIC_ORBITS[0]
                mapping.update(
                    {
                        value: orbit[(index + shift) % len(orbit)]
                        for index, value in enumerate(orbit)
                    }
                )
                pure_renderer_v8._validate_attribute_map(mapping, library=library)

            products = [
                product
                for category in library["categories"]
                for product in library["category_products"][category]
            ]
            for traditional in (False, True):
                style = {
                    "separator": "，",
                    "ending": "。",
                    "line_mode": "single",
                    "english_tag": "",
                    "traditional_variant": traditional,
                    "repeat_punctuation": False,
                }
                reachable_attributes = []
                rendered_by_attribute: dict[
                    str, tuple[list[str], list[str]]
                ] = {}
                for attribute in pure_renderer_v8.ATTRIBUTE_SEMANTIC_ORBITS[0]:
                    reachable = (
                        attribute.translate(translation)
                        if traditional
                        else attribute
                    )
                    reachable_attributes.append(reachable)
                    title_outputs: list[str] = []
                    description_outputs: list[str] = []
                    for product in products:
                        for skeleton in library["title_skeletons"]:
                            rendered = pure_renderer_v8._render_base_title(
                                skeleton=skeleton,
                                product=product,
                                attribute=attribute,
                                code="QABCDEFGHIJ",
                                style=style,
                                library=library,
                            )
                            title_outputs.append(rendered)
                            if "{attribute}" in skeleton:
                                self.assertIn(reachable, rendered)
                        for skeleton in library["description_skeletons"]:
                            rendered = pure_renderer_v8._render_base_description(
                                skeleton=skeleton,
                                product=product,
                                attribute=attribute,
                                code="QABCDEFGHIJ",
                                delivery=library["delivery"][0],
                                service=library["service"][0],
                                style=style,
                                library=library,
                            )
                            description_outputs.append(rendered)
                            if "{attribute}" in skeleton:
                                self.assertIn(reachable, rendered)
                    rendered_by_attribute[attribute] = (
                        title_outputs,
                        description_outputs,
                    )
                self.assertEqual(len(set(reachable_attributes)), 3)
                baseline_titles, baseline_descriptions = rendered_by_attribute[
                    pure_renderer_v8.ATTRIBUTE_SEMANTIC_ORBITS[0][0]
                ]
                for title_outputs, description_outputs in rendered_by_attribute.values():
                    offset = 0
                    for _product in products:
                        for skeleton in library["title_skeletons"]:
                            if "{attribute}" not in skeleton:
                                self.assertEqual(
                                    title_outputs[offset], baseline_titles[offset]
                                )
                            offset += 1
                    offset = 0
                    for _product in products:
                        for skeleton in library["description_skeletons"]:
                            if "{attribute}" not in skeleton:
                                self.assertEqual(
                                    description_outputs[offset],
                                    baseline_descriptions[offset],
                                )
                            offset += 1

    def test_attribute_repair_source_has_no_exposed_failure_special_case(self) -> None:
        source = inspect.getsource(pure_renderer_v8)
        for forbidden in (
            "d926e73bee174bc26ddbe2f563b2c679c2908fdf60b685916d1b71e59dd74a01",
            "ec9ba637bdb97d31083a89df00ea1f8004b97ae299a10ac75b51deffd5869943",
            "文件整理工具",
            "ordinal == 159",
            "candidate_index == 5",
        ):
            self.assertNotIn(forbidden, source)
        self.assertEqual(
            tuple(inspect.signature(pure_renderer_v8._attribute_rotation_map).parameters),
            ("candidate_key", "library"),
        )


class ScientificWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = scientific.load_policy()
        cls.context = scientific.build_execution_context(
            cls.policy, execution_mode="small_smoke"
        )
        cls.template, cls.fixture, cls.style_profile = scientific.load_release_inputs(
            cls.context
        )
        cls.record = cls.context.world_records[0]
        cls.structure_key = cls.context.effective_policy["randomness"][
            cls.context.base_mode
        ]["structure_key_hex"]
        cls.accepted = cls._build()

    @classmethod
    def _build(
        cls,
        *,
        historical_items: frozenset[str] = frozenset(),
        historical_sellers: frozenset[str] = frozenset(),
    ) -> world_module.AcceptedScientificWorld:
        return world_module.build_scientific_world(
            policy=cls.context.effective_policy,
            template=cls.template,
            fixture=cls.fixture,
            style_profile=cls.style_profile,
            mode=cls.context.base_mode,
            world_record=cls.record,
            structure_key_hex=cls.structure_key,
            document_variation_key=cls.context.document_variation_key,
            anonymous_handle_key=cls.context.anonymous_handle_key,
            historical_item_hashes=historical_items,
            historical_seller_hashes=historical_sellers,
            historical_identity_hashes=frozenset(),
            current_item_hashes=set(),
            current_seller_hashes=set(),
            current_identity_hashes=set(),
        )

    def test_world_truth_and_retrieval_cardinalities(self) -> None:
        accepted = self.accepted
        self.assertEqual(len(accepted.seller_profiles), 28)
        self.assertEqual(len(accepted.identity33), 378)
        self.assertEqual(len(accepted.pair_labels), 378)
        self.assertEqual(sum(row["label"] for row in accepted.pair_labels), 20)
        self.assertEqual(len(accepted.qrels), 28)
        self.assertTrue(
            all(len(row["relevant_seller_uids"]) in {1, 2} for row in accepted.qrels)
        )

    def test_identity33_and_pair_endpoint_keysets_match(self) -> None:
        endpoints = {
            (row["world_uid"], row["canonical_pair_uid"])
            for row in self.accepted.world["public"]["complete_model_pair_endpoints"]
        }
        identity = {
            (row["world_uid"], row["canonical_pair_uid"])
            for row in self.accepted.identity33
        }
        labels = {
            (row["world_uid"], row["canonical_pair_uid"])
            for row in self.accepted.pair_labels
        }
        self.assertEqual(endpoints, identity)
        self.assertEqual(endpoints, labels)

    def test_provenance_invariant_ignores_only_valid_rank_permutations(self) -> None:
        _profiles, provenance, _identity33, _redacted = (
            world_module._build_profiles_and_identity33(
                policy=self.context.effective_policy,
                mode=self.context.base_mode,
                split=str(self.record["split"]),
                template=self.template,
                world=self.accepted.world,
            )
        )
        baseline_exact = common.canonical_sha256(provenance)
        baseline_source = (
            world_module._profile_provenance_source_multiset_sha256(provenance)
        )

        swapped = copy.deepcopy(provenance)
        groups = {}
        for index, row in enumerate(swapped["rows"]):
            groups.setdefault((row["seller_uid"], row["output_field"]), []).append(
                index
            )
        first_group = next(indices for indices in groups.values() if len(indices) >= 2)
        left, right = first_group[:2]
        swapped["rows"][left]["output_rank"], swapped["rows"][right][
            "output_rank"
        ] = (
            swapped["rows"][right]["output_rank"],
            swapped["rows"][left]["output_rank"],
        )
        swapped["rows"].sort(
            key=lambda row: (
                row["seller_uid"].encode("utf-8"),
                row["output_field"].encode("utf-8"),
                int(row["output_rank"]),
            )
        )
        swapped["rows_sha256"] = common.canonical_sha256(swapped["rows"])
        self.assertNotEqual(common.canonical_sha256(swapped), baseline_exact)
        self.assertEqual(
            world_module._profile_provenance_source_multiset_sha256(swapped),
            baseline_source,
        )

        mutations = {
            "seller_uid": lambda value: f"{value}_mutated",
            "output_field": lambda value: f"{value}_mutated",
            "aggregation_role": lambda value: f"{value}_mutated",
            "source_item_uids": lambda value: [*value, "i_mutated"],
            "source_item_uids_sha256": lambda _value: "0" * 64,
            "source_item_count": lambda value: value + 1,
            "first_seen_position": lambda value: value + 1,
            "item_uid": lambda value: f"{value}_mutated",
            "extracted_segment_ordinal": lambda value: value + 1,
            "seller_df": lambda value: value + 1,
            "seller_df_seller_count": lambda value: value + 1,
            "seller_df_seller_uids_sha256": lambda _value: "f" * 64,
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(provenance)
                changed["rows"][0][field] = mutate(changed["rows"][0][field])
                changed["rows_sha256"] = common.canonical_sha256(changed["rows"])
                try:
                    changed_source = (
                        world_module._profile_provenance_source_multiset_sha256(
                            changed
                        )
                    )
                except world_module.ScientificWorldError:
                    continue
                self.assertNotEqual(changed_source, baseline_source)

        invalid_rank = copy.deepcopy(provenance)
        invalid_rank["rows"][left]["output_rank"] = 0
        invalid_rank["rows_sha256"] = common.canonical_sha256(invalid_rank["rows"])
        with self.assertRaisesRegex(
            world_module.ScientificWorldError, "output rank"
        ):
            world_module._profile_provenance_source_multiset_sha256(invalid_rank)

        for label, mutate_rank in (
            (
                "duplicate",
                lambda changed: changed["rows"][left].__setitem__(
                    "output_rank", changed["rows"][right]["output_rank"]
                ),
            ),
            (
                "gap",
                lambda changed: changed["rows"][right].__setitem__(
                    "output_rank", len(first_group) + 1
                ),
            ),
            (
                "bool",
                lambda changed: changed["rows"][left].__setitem__(
                    "output_rank", True
                ),
            ),
        ):
            with self.subTest(rank_failure=label):
                changed = copy.deepcopy(provenance)
                mutate_rank(changed)
                changed["rows_sha256"] = common.canonical_sha256(changed["rows"])
                with self.assertRaisesRegex(
                    world_module.ScientificWorldError, "output rank"
                ):
                    world_module._profile_provenance_source_multiset_sha256(
                        changed
                    )

    def test_lineage_failure_precedes_collision_and_registry_commit(self) -> None:
        items: set[str] = set()
        sellers: set[str] = set()
        identities: set[str] = set()
        real_digest = world_module._profile_provenance_source_multiset_sha256
        calls = 0

        def force_candidate_drift(provenance: dict) -> str:
            nonlocal calls
            calls += 1
            digest = real_digest(provenance)
            return digest if calls == 1 else ("0" * 64)

        with (
            mock.patch.object(
                world_module,
                "_profile_provenance_source_multiset_sha256",
                side_effect=force_candidate_drift,
            ),
            mock.patch.object(
                world_module,
                "_collision_categories",
                wraps=world_module._collision_categories,
            ) as collision_probe,
        ):
            with self.assertRaisesRegex(
                world_module.ScientificWorldError,
                "changed frozen profile contribution lineage",
            ):
                world_module.build_scientific_world(
                    policy=self.context.effective_policy,
                    template=self.template,
                    fixture=self.fixture,
                    style_profile=self.style_profile,
                    mode=self.context.base_mode,
                    world_record=self.record,
                    structure_key_hex=self.structure_key,
                    document_variation_key=self.context.document_variation_key,
                    anonymous_handle_key=self.context.anonymous_handle_key,
                    historical_item_hashes=frozenset(),
                    historical_seller_hashes=frozenset(),
                    historical_identity_hashes=frozenset(),
                    current_item_hashes=items,
                    current_seller_hashes=sellers,
                    current_identity_hashes=identities,
                )
        collision_probe.assert_not_called()
        self.assertEqual(items, set())
        self.assertEqual(sellers, set())
        self.assertEqual(identities, set())

    def test_full_candidate_provenance_hash_enters_private_world_audit(self) -> None:
        row = dataset_builder._private_world_audit_row(self.accepted)
        self.assertEqual(
            row["profile_provenance_sha256"],
            self.accepted.profile_provenance_sha256,
        )
        self.assertRegex(row["profile_provenance_sha256"], r"^[0-9a-f]{64}$")

    def test_model_text_fields_do_not_contain_internal_uids(self) -> None:
        world = self.accepted.world
        forbidden = {
            world["public"]["world"]["world_uid"],
            *(row["seller_uid"] for row in world["public"]["sellers"]),
            *(row["controller_uid"] for row in world["private"]["controller_membership"]),
        }
        visible_fields = (
            "category_concat_top",
            "signature_title_concat",
            "title_concat_top",
            "signature_description_concat",
            "description_concat_top",
        )
        for profile in self.accepted.seller_profiles:
            text = "\n".join(str(profile[field]) for field in visible_fields)
            self.assertFalse(any(value in text for value in forbidden))

    def test_model_profile_projection_is_an_exact_allowlist(self) -> None:
        profile = self.accepted.seller_profiles[0]
        projected = dataset_builder._project_model_seller_profile(profile)
        self.assertEqual(tuple(projected), dataset_builder.MODEL_PROFILE_FIELDS)
        self.assertEqual(
            tuple(projected["style_stats"]),
            dataset_builder.MODEL_PROFILE_STYLE_FIELDS,
        )
        self.assertEqual(set(projected["title_length_stats"]), {"median"})
        self.assertEqual(set(projected["description_length_stats"]), {"median"})
        for forbidden in (
            "data_bucket",
            "source_market_raw",
            "source_dataset",
            "profile_text",
            "split",
            "candidate_index",
            "controller_uid",
            "label",
        ):
            self.assertNotIn(forbidden, projected)
        for field in dataset_builder.MODEL_PROFILE_TEXT_FIELDS:
            self.assertEqual(projected[field], profile[field])

    def test_projection_replays_frozen_legacy18_sources_exactly(self) -> None:
        clean_policy = common.load_json(
            ROOT / "schema" / "step7_v3_1_source_data_policy.json"
        )
        clean_cfg = clean_policy["clean_text_contract"]
        full_records = {}
        projected_records = {}
        for profile in self.accepted.seller_profiles:
            seller_uid = str(profile["seller_uid"])
            projected = dataset_builder._project_model_seller_profile(profile)
            full_record, _ = step7_source.build_clean_seller_record(
                dict(profile), clean_cfg
            )
            projected_record, _ = step7_source.build_clean_seller_record(
                projected, clean_cfg
            )
            for field in (
                "model_text",
                "clean_categories",
                "clean_titles",
                "clean_descriptions",
                "numeric_profile",
            ):
                self.assertEqual(full_record[field], projected_record[field])
            full_records[seller_uid] = full_record
            projected_records[seller_uid] = projected_record
        full_reference = step7_source.train_reference(
            full_records, set(full_records)
        )
        projected_reference = step7_source.train_reference(
            projected_records, set(projected_records)
        )
        self.assertEqual(full_reference, projected_reference)
        pair_rows = [
            {
                "pair_uid": row["canonical_pair_uid"],
                "seller_uid_left": row["seller_uid_left"],
                "seller_uid_right": row["seller_uid_right"],
            }
            for row in self.accepted.world["public"][
                "complete_model_pair_endpoints"
            ]
        ]
        full_features = step7_source.build_safe_pair_rows(
            pair_rows, full_records, full_reference
        )
        projected_features = step7_source.build_safe_pair_rows(
            pair_rows, projected_records, projected_reference
        )
        self.assertEqual(len(full_features), 378)
        self.assertEqual(
            len(step7_source.MODEL_ELIGIBLE_TRANSFER_FEATURE_NAMES), 18
        )
        for full_row, projected_row in zip(full_features, projected_features):
            self.assertEqual(full_row["pair_uid"], projected_row["pair_uid"])
            for name in step7_source.MODEL_ELIGIBLE_TRANSFER_FEATURE_NAMES:
                self.assertEqual(full_row[name], projected_row[name])

    def test_global_uid_registry_rejects_reuse_without_partial_commit(self) -> None:
        values = dataset_builder._world_uid_sets(self.accepted)
        for collision_kind in dataset_builder.GLOBAL_UID_KINDS:
            registries = {
                kind: set() for kind in dataset_builder.GLOBAL_UID_KINDS
            }
            registries[collision_kind].add(
                sorted(values[collision_kind], key=lambda value: value.encode("utf-8"))[0]
            )
            before = {kind: set(items) for kind, items in registries.items()}
            with self.assertRaisesRegex(
                dataset_builder.DatasetBuildError,
                rf"{collision_kind} UID reuse",
            ):
                dataset_builder._commit_uid_values(values, seen=registries)
            self.assertEqual(registries, before)

    def test_exact_document_collision_alone_advances_candidate(self) -> None:
        observations: list[dict] = []

        def expose_first_two_candidates(**kwargs: object) -> tuple[str, ...]:
            observations.append(copy.deepcopy(kwargs))
            return (
                ("historical_item_document",)
                if len(observations) == 1
                else ()
            )

        with mock.patch.object(
            world_module,
            "_collision_categories",
            side_effect=expose_first_two_candidates,
        ):
            exposed = self._build()
        self.assertEqual(exposed.candidate_index, 1)
        self.assertEqual(len(observations), 2)
        first_items = set(observations[0]["item_hashes"])
        second_items = set(observations[1]["item_hashes"])
        candidate_zero_only = first_items - second_items
        self.assertTrue(candidate_zero_only)
        historical_collision = min(
            candidate_zero_only, key=lambda value: value.encode("utf-8")
        )

        self.assertEqual(
            world_module._collision_categories(
                **{
                    **observations[0],
                    "historical_item_hashes": frozenset({historical_collision}),
                }
            ),
            ("historical_item_document",),
        )
        self.assertEqual(
            world_module._collision_categories(
                **{
                    **observations[1],
                    "historical_item_hashes": frozenset({historical_collision}),
                }
            ),
            (),
        )
        replay = self._build(
            historical_items=frozenset({historical_collision}),
        )
        self.assertEqual(replay.candidate_index, 1)
        self.assertEqual(replay.candidates_examined, replay.candidate_index + 1)
        self.assertEqual(replay.rejection_counts["historical_item_document"], 1)
        self.assertEqual(replay.rejection_counts["historical_seller_document"], 0)

    def test_failed_candidate_exhaustion_does_not_mutate_registries(self) -> None:
        items: set[str] = set()
        sellers: set[str] = set()
        identities: set[str] = set()
        with mock.patch.object(
            world_module,
            "_collision_categories",
            return_value=("same_world_item_document",),
        ):
            with self.assertRaisesRegex(
                world_module.ScientificWorldError, "All 32"
            ):
                world_module.build_scientific_world(
                    policy=self.context.effective_policy,
                    template=self.template,
                    fixture=self.fixture,
                    style_profile=self.style_profile,
                    mode=self.context.base_mode,
                    world_record=self.record,
                    structure_key_hex=self.structure_key,
                    document_variation_key=self.context.document_variation_key,
                    anonymous_handle_key=self.context.anonymous_handle_key,
                    historical_item_hashes=frozenset(),
                    historical_seller_hashes=frozenset(),
                    historical_identity_hashes=frozenset(),
                    current_item_hashes=items,
                    current_seller_hashes=sellers,
                    current_identity_hashes=identities,
                )
        self.assertEqual(items, set())
        self.assertEqual(sellers, set())
        self.assertEqual(identities, set())

    def test_deterministic_replay_is_byte_identical(self) -> None:
        replay = self._build()
        self.assertEqual(
            common.canonical_json_bytes(replay.world),
            common.canonical_json_bytes(self.accepted.world),
        )
        self.assertEqual(replay.item_registry_delta, self.accepted.item_registry_delta)
        self.assertEqual(
            replay.identity_registry_delta, self.accepted.identity_registry_delta
        )


class ScientificExactTitleEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = scientific.load_policy()
        cls.context = scientific.build_execution_context(
            cls.policy, execution_mode="design_preflight"
        )
        cls.template, cls.fixture, cls.style_profile = scientific.load_release_inputs(
            cls.context
        )

    @classmethod
    def _baseline(cls, record: dict) -> tuple[dict, str]:
        split = str(record["split"])
        structure_key = common.structure_key_for_split(
            cls.context.effective_policy,
            mode=cls.context.base_mode,
            split=split,
        )
        world = world_module.world_builder.build_world(
            policy=world_module._canonical_clone(cls.context.effective_policy),
            template=world_module._canonical_clone(cls.template),
            fixture=world_module._canonical_clone(cls.fixture),
            style_profile=world_module._canonical_clone(cls.style_profile),
            mode=cls.context.base_mode,
            world_record=world_module._canonical_clone(record),
            structure_key_hex=structure_key,
        )
        return world, structure_key

    @staticmethod
    def _clone_rows(world: dict) -> list[dict]:
        return [
            row
            for row in world["private"]["override_audit"]
            if row["override_kind"] == "exact_title_clone"
        ]

    def test_development_world_429_closes_the_observed_description_gap(self) -> None:
        record = next(
            row
            for row in self.context.world_records
            if row["split"] == "development" and row["split_ordinal"] == 429
        )
        world, structure_key = self._baseline(record)
        render_asts = {
            str(row["item_uid"]): row for row in world["private"]["render_asts"]
        }
        original = self._clone_rows(world)
        self.assertEqual(
            [
                bool(render_asts[str(row["item_uid_right"])]["description_nonempty"])
                for row in original
            ],
            [True, False],
        )
        original_pairs = [
            (row["seller_uid_left"], row["seller_uid_right"]) for row in original
        ]
        receipt = world_module._qualify_exact_title_clone_endpoints(
            policy=self.context.effective_policy,
            template=self.template,
            mode=self.context.base_mode,
            split="development",
            structure_key_hex=structure_key,
            world=world,
        )
        qualified = self._clone_rows(world)
        self.assertEqual(
            original_pairs,
            [(row["seller_uid_left"], row["seller_uid_right"]) for row in qualified],
        )
        self.assertTrue(
            all(
                render_asts[str(row["item_uid_left"])]["title_nonempty"] is True
                and render_asts[str(row["item_uid_right"])]["title_nonempty"]
                is True
                and render_asts[str(row["item_uid_right"])][
                    "description_nonempty"
                ]
                is True
                for row in qualified
            )
        )
        self.assertTrue(
            any(
                row["original_target_item_uid"]
                != row["qualified_target_item_uid"]
                for row in receipt["rows"]
            )
        )
        self.assertFalse(receipt["seller_pairs_or_direction_changed"])
        self.assertFalse(receipt["labels_or_model_scores_read"])

    def test_train_world_69_preserves_frozen_profile_lineage_at_candidate_zero(
        self,
    ) -> None:
        """Regression for the first v8 execution failure, not a result preview."""

        record = next(
            row
            for row in self.context.world_records
            if row["split"] == "train" and row["split_ordinal"] == 69
        )
        historical = dataset_builder.collision.load_historical_exclusion_registries()
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
        )
        self.assertEqual(accepted.split_ordinal, 69)
        self.assertEqual(accepted.candidate_index, 0)
        self.assertEqual(accepted.candidates_examined, 1)
        self.assertEqual(sum(accepted.rejection_counts.values()), 0)
        self.assertEqual(len(accepted.seller_profiles), 28)
        self.assertEqual(len(accepted.identity33), 378)
        self.assertEqual(len(accepted.pair_labels), 378)
        self.assertEqual(sum(row["label"] for row in accepted.pair_labels), 20)

    def test_all_1004_preflight_worlds_have_qualified_clone_endpoints(self) -> None:
        relocated = 0
        row_count = 0
        for record in self.context.world_records:
            world, structure_key = self._baseline(record)
            before_pairs = [
                (
                    row["canonical_pair_uid"],
                    row["seller_uid_left"],
                    row["seller_uid_right"],
                )
                for row in self._clone_rows(world)
            ]
            before_truth = common.canonical_sha256(
                {
                    "controller_membership": world["private"][
                        "controller_membership"
                    ],
                    "negative_flags": world["private"]["negative_flags"],
                }
            )
            receipt = world_module._qualify_exact_title_clone_endpoints(
                policy=self.context.effective_policy,
                template=self.template,
                mode=self.context.base_mode,
                split=str(record["split"]),
                structure_key_hex=structure_key,
                world=world,
            )
            clones = self._clone_rows(world)
            render_asts = {
                str(row["item_uid"]): row
                for row in world["private"]["render_asts"]
            }
            self.assertEqual(
                before_pairs,
                [
                    (
                        row["canonical_pair_uid"],
                        row["seller_uid_left"],
                        row["seller_uid_right"],
                    )
                    for row in clones
                ],
            )
            self.assertEqual(
                before_truth,
                common.canonical_sha256(
                    {
                        "controller_membership": world["private"][
                            "controller_membership"
                        ],
                        "negative_flags": world["private"]["negative_flags"],
                    }
                ),
            )
            self.assertEqual(len(clones), 2)
            self.assertEqual(
                len(
                    {
                        str(row[field])
                        for row in world["private"]["override_audit"]
                        for field in ("item_uid_left", "item_uid_right")
                    }
                ),
                12,
            )
            self.assertTrue(
                all(
                    render_asts[str(row["item_uid_left"])]["title_nonempty"]
                    is True
                    and render_asts[str(row["item_uid_right"])]["title_nonempty"]
                    is True
                    and render_asts[str(row["item_uid_right"])][
                        "description_nonempty"
                    ]
                    is True
                    for row in clones
                )
            )
            self.assertFalse(receipt["labels_or_model_scores_read"])
            self.assertFalse(receipt["shortcut_probe_results_read"])
            row_count += receipt["row_count"]
            relocated += sum(
                row["original_source_item_uid"]
                != row["qualified_source_item_uid"]
                or row["original_target_item_uid"]
                != row["qualified_target_item_uid"]
                for row in receipt["rows"]
            )
        self.assertEqual(row_count, 2008)
        self.assertGreater(relocated, 0)

    def test_qualification_fails_when_a_target_seller_has_no_described_item(self) -> None:
        record = next(
            row
            for row in self.context.world_records
            if row["split"] == "train" and row["split_ordinal"] == 3
        )
        world, structure_key = self._baseline(record)
        target_seller = str(self._clone_rows(world)[0]["seller_uid_right"])
        for ast in world["private"]["render_asts"]:
            if str(ast["seller_uid"]) == target_seller:
                ast["description_nonempty"] = False
        with self.assertRaisesRegex(
            world_module.ScientificWorldError,
            "No structurally qualified exact-title clone endpoint",
        ):
            world_module._qualify_exact_title_clone_endpoints(
                policy=self.context.effective_policy,
                template=self.template,
                mode=self.context.base_mode,
                split="train",
                structure_key_hex=structure_key,
                world=world,
            )


class ScientificSequentialCollisionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = scientific.load_policy()
        cls.context = scientific.build_execution_context(
            cls.policy, execution_mode="design_preflight"
        )
        cls.template, cls.fixture, cls.style_profile = scientific.load_release_inputs(
            cls.context
        )
        cls.historical = (
            dataset_builder.collision.load_historical_exclusion_registries()
        )

    def _build_world(
        self,
        record: dict,
        *,
        current_items: set[str],
        current_sellers: set[str],
        current_identities: set[str],
    ) -> world_module.AcceptedScientificWorld:
        return world_module.build_scientific_world(
            policy=self.context.effective_policy,
            template=self.template,
            fixture=self.fixture,
            style_profile=self.style_profile,
            mode=self.context.base_mode,
            world_record=record,
            structure_key_hex=common.structure_key_for_split(
                self.context.effective_policy,
                mode=self.context.base_mode,
                split="train",
            ),
            document_variation_key=self.context.document_variation_key,
            anonymous_handle_key=self.context.anonymous_handle_key,
            historical_item_hashes=self.historical.item_document_hashes,
            historical_seller_hashes=self.historical.seller_document_hashes,
            historical_identity_hashes=self.historical.identity_value_hashes,
            current_item_hashes=current_items,
            current_seller_hashes=current_sellers,
            current_identity_hashes=current_identities,
        )

    def _replay_old_two_state_prefix_and_target(
        self, records: list[dict]
    ) -> None:
        old_orbits = (
            ("标准版", "组合版"),
            ("轻量版", "更新版"),
            ("多规格",),
            ("可选配色",),
            ("分批交付",),
            ("附使用说明",),
            ("支持自选参数",),
            ("含基础售后",),
        )
        current_items: set[str] = set()
        current_sellers: set[str] = set()
        current_identities: set[str] = set()
        accepted_indices: list[int] = []
        with mock.patch.object(
            pure_renderer_v8, "ATTRIBUTE_SEMANTIC_ORBITS", old_orbits
        ):
            for record in records[:-1]:
                accepted = self._build_world(
                    record,
                    current_items=current_items,
                    current_sellers=current_sellers,
                    current_identities=current_identities,
                )
                accepted_indices.append(accepted.candidate_index)

            self.assertEqual(
                {
                    index: accepted_indices.count(index)
                    for index in set(accepted_indices)
                },
                {0: 144, 1: 6, 2: 5, 3: 2, 4: 1, 7: 1},
            )
            self.assertEqual(
                (len(current_items), len(current_sellers), len(current_identities)),
                (16034, 4452, 13356),
            )
            self.assertEqual(
                common.canonical_sha256(sorted(current_items)),
                "c3a3ffe3187837c3984205d8c2a4d2e67820d43b598efb9a2db16de80c2f2162",
            )
            self.assertEqual(
                common.canonical_sha256(sorted(current_sellers)),
                "4d7c6daac239a6b28813f58ac142cae4ca974033af98db0c9fc717c6d72f549c",
            )
            self.assertEqual(
                common.canonical_sha256(sorted(current_identities)),
                "ddfecefdf1157678a5b4136ec04944ba8faba14c8a840c730d08c4b3983484af",
            )

            collision_categories: list[tuple[str, ...]] = []
            historical_hits: list[frozenset[str]] = []
            target_standard_mappings: list[str] = []
            natural_outputs: list[str] = []
            original_categories = world_module._collision_categories
            original_attribute_map = pure_renderer_v8._attribute_rotation_map
            original_assemble = world_module._assemble_candidate

            def capture_categories(**kwargs: object) -> tuple[str, ...]:
                categories = original_categories(**kwargs)
                collision_categories.append(categories)
                historical_hits.append(
                    frozenset(
                        set(kwargs["item_hashes"])
                        & set(kwargs["historical_item_hashes"])
                    )
                )
                return categories

            def capture_attribute_map(**kwargs: object) -> dict[str, str]:
                mapping = original_attribute_map(**kwargs)
                target_standard_mappings.append(mapping["标准版"])
                return mapping

            def capture_assemble(**kwargs: object) -> world_module.CandidateObservation:
                observation = original_assemble(**kwargs)
                natural_outputs.append(observation.natural_output_sha256)
                return observation

            with (
                mock.patch.object(
                    world_module,
                    "_collision_categories",
                    side_effect=capture_categories,
                ),
                mock.patch.object(
                    pure_renderer_v8,
                    "_attribute_rotation_map",
                    side_effect=capture_attribute_map,
                ),
                mock.patch.object(
                    world_module,
                    "_assemble_candidate",
                    side_effect=capture_assemble,
                ),
            ):
                with self.assertRaisesRegex(
                    world_module.ScientificWorldError,
                    "All 32 exact-document candidates collided",
                ):
                    self._build_world(
                        records[-1],
                        current_items=current_items,
                        current_sellers=current_sellers,
                        current_identities=current_identities,
                    )

        self.assertEqual(
            collision_categories,
            [("historical_item_document",)] * 32,
        )
        historical_union = set().union(*historical_hits)
        historical_intersection = set.intersection(
            *(set(values) for values in historical_hits)
        )
        self.assertEqual(len(historical_union), 2)
        self.assertEqual(historical_intersection, set())
        self.assertEqual(
            common.canonical_sha256(sorted(historical_union)),
            "ec9ba637bdb97d31083a89df00ea1f8004b97ae299a10ac75b51deffd5869943",
        )
        self.assertEqual(set(target_standard_mappings), {"标准版", "组合版"})
        self.assertEqual(len(set(natural_outputs)), 29)
        self.assertEqual(
            (len(current_items), len(current_sellers), len(current_identities)),
            (16034, 4452, 13356),
        )

    def test_train_worlds_zero_through_159_close_with_real_cumulative_registries(
        self,
    ) -> None:
        current_items: set[str] = set()
        current_sellers: set[str] = set()
        current_identities: set[str] = set()
        records = sorted(
            (
                row
                for row in self.context.world_records
                if row["split"] == "train" and row["split_ordinal"] <= 159
            ),
            key=lambda row: row["split_ordinal"],
        )
        self.assertEqual([row["split_ordinal"] for row in records], list(range(160)))
        self._replay_old_two_state_prefix_and_target(records)
        accepted = [
            self._build_world(
                record,
                current_items=current_items,
                current_sellers=current_sellers,
                current_identities=current_identities,
            )
            for record in records[:-1]
        ]
        self.assertEqual(len(current_items), 16034)
        self.assertEqual(len(current_sellers), 4452)
        self.assertEqual(len(current_identities), 13356)
        self.assertEqual(
            common.canonical_sha256(sorted(current_items)),
            "c4d7fd0ec524824144f3e1e99cb5120bbb10bab50673e1d322379787f6e704ab",
        )
        self.assertEqual(
            common.canonical_sha256(sorted(current_sellers)),
            "ebe213a24c5fbf4b18261eb3e99d1278dda4b82a9afa611d571e938a12f3ebc5",
        )
        self.assertEqual(
            common.canonical_sha256(sorted(current_identities)),
            "ddfecefdf1157678a5b4136ec04944ba8faba14c8a840c730d08c4b3983484af",
        )

        candidate_categories: list[tuple[str, ...]] = []
        target_standard_mappings: list[str] = []
        original_categories = world_module._collision_categories
        original_attribute_map = pure_renderer_v8._attribute_rotation_map

        def capture_categories(**kwargs: object) -> tuple[str, ...]:
            categories = original_categories(**kwargs)
            candidate_categories.append(categories)
            return categories

        def capture_attribute_map(**kwargs: object) -> dict[str, str]:
            mapping = original_attribute_map(**kwargs)
            target_standard_mappings.append(mapping["标准版"])
            return mapping

        with (
            mock.patch.object(
                world_module,
                "_collision_categories",
                side_effect=capture_categories,
            ),
            mock.patch.object(
                pure_renderer_v8,
                "_attribute_rotation_map",
                side_effect=capture_attribute_map,
            ),
        ):
            target = self._build_world(
                records[-1],
                current_items=current_items,
                current_sellers=current_sellers,
                current_identities=current_identities,
            )
        accepted.append(target)

        self.assertEqual(target.split_ordinal, 159)
        self.assertEqual(target.candidate_index, 5)
        self.assertEqual(target.candidates_examined, 6)
        self.assertEqual(
            target.rejection_counts,
            {
                "same_world_item_document": 0,
                "same_world_seller_document": 0,
                "historical_item_document": 5,
                "historical_seller_document": 0,
                "current_dataset_item_document": 2,
                "current_dataset_seller_document": 0,
            },
        )
        self.assertEqual(
            candidate_categories,
            [
                ("historical_item_document",),
                ("historical_item_document", "current_dataset_item_document"),
                ("historical_item_document", "current_dataset_item_document"),
                ("historical_item_document",),
                ("historical_item_document",),
                (),
            ],
        )
        self.assertEqual(
            target_standard_mappings,
            ["标准版", "组合版", "组合版", "组合版", "多规格", "多规格"],
        )
        self.assertEqual(
            {index: sum(row.candidate_index == index for row in accepted)
             for index in {row.candidate_index for row in accepted}},
            {0: 147, 1: 6, 2: 4, 3: 1, 5: 1, 7: 1},
        )
        self.assertEqual(len(current_items), 16138)
        self.assertEqual(len(current_sellers), 4480)
        self.assertEqual(len(current_identities), 13440)
        self.assertEqual(
            common.canonical_sha256(sorted(current_items)),
            "ba90909a83566e1b31d5d49fcd8df28e881d2b2686f1a8b9861cb7486fdb3af2",
        )
        self.assertEqual(
            common.canonical_sha256(sorted(current_sellers)),
            "e4a6a0818162c9fa6f83343a34a07043098fdba9af24e90894b9d669744c285d",
        )
        self.assertEqual(
            common.canonical_sha256(sorted(current_identities)),
            "e00b11e464e4638401353cc2a9e8bd240f1ebca75d21df38fe5d05f47424428a",
        )

    def test_world_29_old_singleton_attribute_domain_reproduces_known_collision(
        self,
    ) -> None:
        target = next(
            row
            for row in self.context.world_records
            if row["split"] == "train" and row["split_ordinal"] == 29
        )
        observed_item_hashes: list[tuple[str, ...]] = []

        def identity_attributes(*, candidate_key: bytes, library: dict) -> dict:
            del candidate_key
            return {value: value for value in library["attributes"]}

        def capture_first_candidate(**kwargs: object) -> tuple[str, ...]:
            observed_item_hashes.append(tuple(kwargs["item_hashes"]))
            return ()

        with (
            mock.patch.object(
                pure_renderer_v8,
                "_attribute_rotation_map",
                side_effect=identity_attributes,
            ),
            mock.patch.object(
                world_module,
                "_collision_categories",
                side_effect=capture_first_candidate,
            ),
        ):
            accepted = self._build_world(
                target,
                current_items=set(),
                current_sellers=set(),
                current_identities=set(),
            )
        self.assertEqual(accepted.candidate_index, 0)
        self.assertEqual(len(observed_item_hashes), 1)
        self.assertIn(
            "1b27758b380e57e90baf967db68a319540ea7909581d96aab7b4c4953ac03082",
            observed_item_hashes[0],
        )
        self.assertIn(
            "1b27758b380e57e90baf967db68a319540ea7909581d96aab7b4c4953ac03082",
            self.historical.item_document_hashes,
        )


class ScientificDatasetEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_directory.name)
        policy = scientific.load_policy()
        context = scientific.build_execution_context(
            policy, execution_mode="small_smoke"
        )
        historical = dataset_builder.collision.load_historical_exclusion_registries()
        cls.outputs: list[Path] = []
        cls.manifests: list[dict] = []
        for ordinal in range(2):
            output = cls.root / f"small_smoke_replay_{ordinal}"
            patched_context = replace(context, output_root=output)
            with (
                mock.patch.object(
                    dataset_builder.scientific,
                    "build_execution_context",
                    return_value=patched_context,
                ),
                mock.patch.object(
                    dataset_builder.collision,
                    "load_historical_exclusion_registries",
                    return_value=historical,
                ),
            ):
                manifest = dataset_builder.run_build(execution_mode="small_smoke")
            cls.outputs.append(output)
            cls.manifests.append(manifest)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_directory.cleanup()

    def test_repeated_four_split_build_is_content_identical(self) -> None:
        self.assertEqual(
            common.canonical_json_bytes(self.manifests[0]),
            common.canonical_json_bytes(self.manifests[1]),
        )
        for split in scientific.SPLITS:
            first = common.load_json(self.outputs[0] / split / "split_manifest.json")
            second = common.load_json(self.outputs[1] / split / "split_manifest.json")
            self.assertEqual(
                common.canonical_json_bytes(first),
                common.canonical_json_bytes(second),
            )

    def test_four_split_file_routing_schemas_and_counts_close(self) -> None:
        root_manifest = common.load_json(self.outputs[0] / "root_manifest.json")
        claimed = root_manifest.pop("canonical_self_hash")
        self.assertEqual(claimed, common.canonical_sha256(root_manifest))
        self.assertEqual(root_manifest["status"], "PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED")
        self.assertTrue(root_manifest["scientific_use_forbidden"])
        self.assertFalse(root_manifest["formal_seed_created"])
        self.assertEqual(root_manifest["formal_rows_created"], 0)
        self.assertEqual(root_manifest["world_count"], 4)
        self.assertEqual(root_manifest["seller_count"], 112)
        self.assertEqual(root_manifest["pair_count"], 1512)
        self.assertEqual(root_manifest["positive_pair_count"], 80)
        expected_uid_counts = {
            "world": 4,
            "seller": 112,
            "pair": 1512,
            "query": 112,
            "controller": 48,
        }
        for kind, count in expected_uid_counts.items():
            self.assertEqual(root_manifest["uid_registries"][kind]["count"], count)
        self.assertEqual(
            root_manifest["uid_registries"]["item"]["count"],
            root_manifest["item_document_registry_count"],
        )

        expected_observed = {
            "worlds.jsonl",
            "redacted_items.jsonl",
            "model_seller_profiles.jsonl",
            "complete_model_pair_endpoints.csv",
            "identity33_all_pairs.csv",
        }
        expected_private = {
            "controller_membership.jsonl",
            "pair_labels.csv",
            "qrels.jsonl",
            "world_generation_audit.jsonl",
            "document_collision_attempts.jsonl",
            "identity_allocation_receipts.jsonl",
        }
        for split in scientific.SPLITS:
            split_root = self.outputs[0] / split
            self.assertEqual(
                {path.name for path in (split_root / "observed").iterdir()},
                expected_observed,
            )
            self.assertEqual(
                {path.name for path in (split_root / "private").iterdir()},
                expected_private,
            )
            manifest = common.load_json(split_root / "split_manifest.json")
            manifest_claimed = manifest.pop("canonical_self_hash")
            self.assertEqual(manifest_claimed, common.canonical_sha256(manifest))
            self.assertEqual(manifest["world_count"], 1)
            self.assertEqual(manifest["world_ordinal_count"], 1)
            self.assertEqual(
                manifest["world_ordinals_sha256"], common.canonical_sha256([0])
            )
            self.assertEqual(manifest["seller_count"], 28)
            self.assertEqual(manifest["pair_count"], 378)
            self.assertEqual(manifest["positive_pair_count"], 20)
            self.assertEqual(manifest["uid_registries"]["world"]["count"], 1)
            self.assertEqual(manifest["uid_registries"]["seller"]["count"], 28)
            self.assertEqual(manifest["uid_registries"]["pair"]["count"], 378)
            self.assertEqual(manifest["uid_registries"]["query"]["count"], 28)
            self.assertEqual(manifest["uid_registries"]["controller"]["count"], 12)
            self.assertEqual(
                manifest["item_count"],
                manifest["item_document_registry_count"],
            )

            with (split_root / "observed" / "model_seller_profiles.jsonl").open(
                "r", encoding="utf-8"
            ) as handle:
                profile = json.loads(next(handle))
            self.assertEqual(set(profile), set(dataset_builder.MODEL_PROFILE_FIELDS))
            self.assertNotIn("data_bucket", profile)
            self.assertNotIn("source_market_raw", profile)
            self.assertNotIn("profile_text", profile)

            with (split_root / "observed" / "redacted_items.jsonl").open(
                "r", encoding="utf-8"
            ) as handle:
                item_rows = [json.loads(line) for line in handle]
            self.assertTrue(item_rows)
            self.assertTrue(
                all(
                    set(row) == set(dataset_builder.MODEL_REDACTED_ITEM_FIELDS)
                    for row in item_rows
                )
            )
            self.assertEqual(len(item_rows), manifest["item_count"])
            self.assertEqual(
                tuple(dataset_builder.MODEL_REDACTED_ITEM_JOIN_ONLY_FIELDS),
                ("item_uid", "seller_uid", "world_uid"),
            )
            self.assertEqual(
                tuple(dataset_builder.MODEL_REDACTED_ITEM_TEXT_FIELDS),
                ("title", "description"),
            )

            with (split_root / "observed" / "worlds.jsonl").open(
                "r", encoding="utf-8"
            ) as handle:
                worlds = [json.loads(line) for line in handle]
            self.assertEqual(len(worlds), 1)
            self.assertEqual(set(worlds[0]), {"world_uid", "split_ordinal"})
            self.assertEqual(worlds[0]["split_ordinal"], 0)

            with (split_root / "private" / "pair_labels.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                labels = list(csv.DictReader(handle))
            self.assertEqual(len(labels), 378)
            self.assertEqual(sum(int(row["label"]) for row in labels), 20)

    def test_output_tree_replay_rejects_one_byte_tamper(self) -> None:
        output = self.outputs[1]
        target = output / "train" / "observed" / "worlds.jsonl"
        original = target.read_bytes()
        manifest = common.load_json(output / "root_manifest.json")
        try:
            target.write_bytes(original + b"\n")
            with self.assertRaisesRegex(
                dataset_builder.DatasetBuildError, "Split file replay drift"
            ):
                dataset_builder._verify_output_tree(output, manifest)
        finally:
            target.write_bytes(original)
        dataset_builder._verify_output_tree(output, manifest)


if __name__ == "__main__":
    unittest.main()
