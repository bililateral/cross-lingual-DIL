from __future__ import annotations

import copy
import csv
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
        replay = self._build(
            historical_items=frozenset(self.accepted.item_registry_delta),
            historical_sellers=frozenset(self.accepted.seller_registry_delta),
        )
        self.assertGreater(replay.candidate_index, 0)
        self.assertEqual(replay.candidates_examined, replay.candidate_index + 1)
        self.assertGreaterEqual(
            replay.rejection_counts["historical_item_document"]
            + replay.rejection_counts["historical_seller_document"],
            replay.candidate_index,
        )

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
