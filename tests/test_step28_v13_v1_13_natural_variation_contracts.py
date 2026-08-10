from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common  # noqa: E402
import step28_v13_v1_13_document_collision as collision  # noqa: E402
import step28_v13_v1_13_natural_variation as subject  # noqa: E402


class Step28V13V113NaturalVariationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = subject.load_policy()
        cls.base_policy = common.load_policy(
            ROOT / "schema" / "step28_v13_synthetic_chinese_dataset_policy.json",
            mode="development_smoke",
        )
        cls.text_key_before = cls.base_policy["randomness"]["development_smoke"][
            "text_key_hex"
        ]
        cls.session = subject.DevelopmentSmokeVariationSession()
        cls.view_object, cls.binding_object = subject._build_restricted_view_and_binding(
            frozen=cls.session._frozen,
            expected_parent=cls.session._parent,
        )
        cls.view = cls.view_object.thaw()
        cls.binding = cls.binding_object.thaw()
        if cls.session._view != cls.view_object:
            raise AssertionError("Session view differs from independently rebuilt view")
        cls.frozen_world = cls.session._frozen.thaw_world()
        cls.results = [cls.session.render(index) for index in range(32)]
        cls.worlds = [result.thaw_world() for result in cls.results]
        document_variation_key = bytes.fromhex(
            cls.policy["development_smoke_keys"]["document_variation_key_hex"]
        )
        cls.candidate0_key = collision.derive_candidate_key(
            document_variation_key=document_variation_key,
            split=cls.session._frozen.split,
            world_uid=cls.session._frozen.world_uid,
            candidate_index=0,
        )
        cls.natural0 = subject.render_candidate_natural_expressions(
            restricted_view=cls.session._view,
            candidate_key=cls.candidate0_key,
        )

    def test_policy_is_design_only_and_has_exact_frozen_input_closure(self) -> None:
        self.assertEqual(self.policy["allowed_mode"], "development_smoke")
        self.assertEqual(self.policy["status"], subject.POLICY_STATUS)
        self.assertEqual(self.policy["claim_boundary"], subject.CLAIM_BOUNDARY)
        self.assertEqual(set(self.policy["frozen_inputs"]), subject.FROZEN_INPUT_KEYS)
        self.assertEqual(
            set(self.policy["formal_authorizations"]),
            subject.FORMAL_AUTHORIZATION_KEYS,
        )
        self.assertTrue(
            all(value is False for value in self.policy["formal_authorizations"].values())
        )
        self.assertIn("No formal seed", self.policy["claim_boundary"])

    def test_every_frozen_input_pin_matches_current_bytes(self) -> None:
        for name, spec in self.policy["frozen_inputs"].items():
            path = ROOT / spec["path"]
            with self.subTest(name=name):
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, spec["size_bytes"])
                self.assertEqual(common.sha256_file(path), spec["sha256"])

    def test_direct_local_imports_are_pinned_and_ancestor_policies_are_pinned(self) -> None:
        source_path = SCRIPTS / "step28_v13_v1_13_natural_variation.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        local_imports: set[Path] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                path = SCRIPTS / f"{name}.py"
                if path.is_file():
                    local_imports.add(path.resolve())
        pinned = {
            (ROOT / spec["path"]).resolve()
            for spec in self.policy["frozen_inputs"].values()
            if str(spec["path"]).endswith(".py")
        }
        self.assertTrue(local_imports <= pinned)
        self.assertIn(
            (ROOT / "schema/step28_v13_v1_13_candidate_parent_policy.json").resolve(),
            {
                (ROOT / spec["path"]).resolve()
                for spec in self.policy["frozen_inputs"].values()
            },
        )
        self.assertIn(
            (ROOT / "schema/step28_v13_v1_13_document_collision_policy.json").resolve(),
            {
                (ROOT / spec["path"]).resolve()
                for spec in self.policy["frozen_inputs"].values()
            },
        )

    def test_re_self_hashed_semantic_policy_tampering_fails(self) -> None:
        for mutate in (
            lambda value: value["formal_authorizations"].__setitem__(
                "formal_candidate_generation", True
            ),
            lambda value: value["frozen_inputs"].pop("candidate_parent"),
            lambda value: value["restricted_candidate_view"]["style_fields"].append(
                "effective_style_uid"
            ),
            lambda value: value["development_smoke_keys"].__setitem__(
                "formal_reuse_forbidden", False
            ),
            lambda value: value.__setitem__("status", "DESIGN_ONLY_TAMPERED"),
            lambda value: value.__setitem__("claim_boundary", "tampered"),
            lambda value: value["formal_authorizations"].pop("audit_truth_access"),
            lambda value: value["restricted_candidate_view"].__setitem__(
                "contains_registered_override_mechanisms", True
            ),
            lambda value: value["restricted_candidate_view"].__setitem__(
                "pure_renderer_loads_policy", True
            ),
            lambda value: value["candidate_output"].__setitem__(
                "natural_output_hash_covers_trusted_registered_overrides", False
            ),
        ):
            policy = copy.deepcopy(self.policy)
            mutate(policy)
            policy.pop("canonical_self_hash")
            policy["canonical_self_hash"] = common.canonical_sha256(policy)
            with self.subTest(mutation=mutate):
                with self.assertRaises(subject.NaturalVariationError):
                    subject._validate_policy(policy)

    def test_public_test_keys_are_distinct_from_base_randomness_and_nonformal(self) -> None:
        keys = self.policy["development_smoke_keys"]
        test_authorities = {
            keys["document_variation_key_hex"],
            keys["anonymous_handle_key_hex"],
            keys["document_variation_key_sha256"],
            keys["anonymous_handle_key_sha256"],
        }
        base_values = set(
            subject._iter_lower_hex_32_values(self.base_policy["randomness"])
        )
        self.assertEqual(len(test_authorities), 4)
        self.assertFalse(test_authorities & base_values)
        self.assertTrue(keys["keys_are_public_test_vectors"])
        self.assertTrue(keys["formal_reuse_forbidden"])

    def test_base_text_key_is_not_replaced_or_mutated(self) -> None:
        reloaded = common.load_policy(
            ROOT / "schema" / "step28_v13_synthetic_chinese_dataset_policy.json",
            mode="development_smoke",
        )
        self.assertEqual(
            reloaded["randomness"]["development_smoke"]["text_key_hex"],
            self.text_key_before,
        )
        self.assertNotEqual(self.candidate0_key.hex(), self.text_key_before)

    def test_pure_variation_signature_has_only_safe_view_and_candidate_key(self) -> None:
        parameters = inspect.signature(
            subject.render_candidate_natural_expressions
        ).parameters
        self.assertEqual(list(parameters), ["restricted_view", "candidate_key"])
        forbidden = {
            "binding",
            "candidate_index",
            "world_uid",
            "seller_uid",
            "identity",
            "label",
            "controller",
            "text_key_hex",
        }
        self.assertFalse(forbidden & set(parameters))
        self.assertEqual(
            subject.render_candidate_natural_expressions.__module__,
            "step28_v13_v1_13_pure_natural_renderer",
        )

    def test_pure_renderer_has_no_project_or_filesystem_authority(self) -> None:
        source_path = SCRIPTS / "step28_v13_v1_13_pure_natural_renderer.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertEqual(
            imported_roots,
            {
                "__future__",
                "hashlib",
                "hmac",
                "json",
                "unicodedata",
                "collections",
                "dataclasses",
                "typing",
            },
        )
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertFalse(
            {
                "load_policy",
                "Path",
                "open",
                "candidate_parent",
                "collision",
                "history_features",
                "production_chain",
                "world_builder",
            }
            & (names | attributes)
        )

    def test_restricted_view_exact_schema_and_cardinality(self) -> None:
        self.assertEqual(set(self.view), set(subject.SAFE_VIEW_FIELDS))
        self.assertEqual(self.view["item_count"], 105)
        self.assertEqual(len(self.view["items"]), 105)
        self.assertEqual(len(self.view["noise_targets"]), 28)
        self.assertNotIn("shared_lexical_groups", self.view)
        self.assertNotIn("title_clone_groups", self.view)
        self.assertEqual(
            self.session.restricted_view_sha256,
            self.policy["expected_smoke_world"]["restricted_view_sha256"],
        )

    def test_restricted_view_contains_no_forbidden_key_atoms(self) -> None:
        subject._scan_for_forbidden_view_content(self.view)
        serialized = json.dumps(self.view, ensure_ascii=False, sort_keys=True)
        for atom in (
            "world_uid",
            "seller_uid",
            "item_uid",
            "canonical_pair_uid",
            "controller_uid",
            "identity_value",
            "candidate_index",
            "retry",
            "override_kind",
            "clone",
            "semantic",
        ):
            self.assertNotIn(atom, serialized)

    def test_restricted_view_contains_no_raw_private_literal(self) -> None:
        serialized = json.dumps(self.view, ensure_ascii=False, sort_keys=True)
        private_literals = {
            self.frozen_world["public"]["world"]["world_uid"],
            *(row["seller_uid"] for row in self.frozen_world["public"]["sellers"]),
            *(row["item_uid"] for row in self.frozen_world["public"]["items"]),
            *(
                row["identity_value"]
                for row in self.frozen_world["private"]["identity_assets"]
            ),
            *(
                row["controller_uid"]
                for row in self.frozen_world["private"]["controller_membership"]
            ),
        }
        self.assertTrue(private_literals)
        self.assertFalse(
            any(len(value) >= 5 and value in serialized for value in private_literals)
        )

    def test_private_binding_is_separate_and_complete(self) -> None:
        self.assertNotIn("item_handle_to_item_uid", self.view)
        self.assertEqual(self.binding["view_sha256"], self.session.restricted_view_sha256)
        self.assertEqual(len(self.binding["item_handle_to_item_uid"]), 105)
        self.assertEqual(len(self.binding["noise_handle_to_noise_slot_uid"]), 28)
        self.assertEqual(len(self.binding["registered_overrides"]), 6)
        self.assertFalse(hasattr(self.session, "_binding"))
        self.assertNotIn(
            "binding", inspect.signature(subject.render_candidate_natural_expressions).parameters
        )

    def test_safe_library_is_exactly_pinned_and_classes_partition_domains(self) -> None:
        library = self.view["safe_library"]
        self.assertEqual(set(library), set(subject.SAFE_LIBRARY_FIELDS))
        self.assertNotIn("category_weights", library)
        self.assertEqual(
            common.canonical_sha256(library),
            self.policy["restricted_candidate_view"]["safe_library_sha256"],
        )
        subject._validate_safe_library(library)
        for field in (
            "attribute_permutation_classes",
            "delivery_permutation_classes",
            "service_permutation_classes",
            "description_skeleton_permutation_classes",
            "noise_value_permutation_classes",
        ):
            self.assertTrue(all(len(group) == 1 for group in library[field]))

    def test_candidate0_golden_hashes_match_policy(self) -> None:
        expected = self.policy["expected_smoke_world"]
        result = self.results[0]
        self.assertEqual(result.natural_output_sha256, expected["candidate0_natural_output_sha256"])
        self.assertEqual(result.world_sha256, expected["candidate0_world_sha256"])
        self.assertEqual(result.candidate_invariant_sha256, expected["candidate_invariant_sha256"])
        self.assertEqual(
            subject._candidate_parent_full_state_sha256(self.session._parent),
            expected["candidate_parent_full_state_sha256"],
        )
        self.assertEqual(result.identity_parent_sha256, expected["identity_parent_sha256"])
        self.assertEqual(
            subject._frozen_identity_full_state_sha256(
                self.session._frozen, parent=self.session._parent
            ),
            expected["frozen_trial_identity_full_state_sha256"],
        )
        self.assertEqual(result.identity33_sha256, expected["identity33_sha256"])
        self.assertEqual(result.profile_provenance_sha256, expected["profile_provenance_sha256"])

    def test_all_32_candidates_are_unique_and_match_combined_golden_hashes(self) -> None:
        expected = self.policy["expected_smoke_world"]
        natural_hashes = [result.natural_output_sha256 for result in self.results]
        world_hashes = [result.world_sha256 for result in self.results]
        self.assertEqual(len(set(natural_hashes)), 32)
        self.assertEqual(len(set(world_hashes)), 32)
        self.assertEqual(common.canonical_sha256(natural_hashes), expected["natural_output_hashes_sha256"])
        self.assertEqual(common.canonical_sha256(world_hashes), expected["world_hashes_sha256"])

    def test_all_32_candidates_share_one_invariant_identity_and_provenance_parent(self) -> None:
        expected = self.policy["expected_smoke_world"]
        fields = (
            ("candidate_invariant_sha256", "candidate_invariant_sha256"),
            ("identity_parent_sha256", "identity_parent_sha256"),
            ("identity33_sha256", "identity33_sha256"),
            ("profile_provenance_sha256", "profile_provenance_sha256"),
        )
        for attribute, policy_field in fields:
            values = {getattr(result, attribute) for result in self.results}
            self.assertEqual(values, {expected[policy_field]})

    def test_candidates_materially_change_visible_items(self) -> None:
        baseline = self.worlds[0]["public"]["items"]
        changed = []
        for world in self.worlds[1:]:
            changed.append(
                sum(
                    (left["category"], left["title"], left["description"])
                    != (right["category"], right["title"], right["description"])
                    for left, right in zip(baseline, world["public"]["items"], strict=True)
                )
            )
        expected = self.policy["expected_smoke_world"]
        self.assertEqual(min(changed), expected["changed_item_count_vs_candidate0_min"])
        self.assertEqual(max(changed), expected["changed_item_count_vs_candidate0_max"])

    def test_candidate_output_contains_no_identity_values_or_real_uids(self) -> None:
        serialized = self.natural0.output_bytes.decode("utf-8")
        forbidden = {
            *(row["item_uid"] for row in self.frozen_world["public"]["items"]),
            *(row["seller_uid"] for row in self.frozen_world["public"]["sellers"]),
            *(
                row["identity_value"]
                for row in self.frozen_world["private"]["identity_assets"]
            ),
        }
        self.assertFalse(any(len(value) >= 5 and value in serialized for value in forbidden))
        self.assertNotIn(self.candidate0_key.hex(), serialized)

    def test_candidate_output_does_not_render_anonymous_handles(self) -> None:
        value = self.natural0.thaw()
        handles = set(self.binding["item_handle_to_item_uid"])
        visible = "\n".join(
            str(row[field])
            for row in value["items"]
            for field in (
                "category",
                "product",
                "attribute",
                "delivery",
                "service",
                "title",
                "base_description",
                "noise_clause",
            )
        )
        self.assertFalse(any(handle in visible for handle in handles))

    def test_pure_text_rendering_matches_production_renderer_on_smoke_domain(self) -> None:
        _base_policy, template, _fixture, _style_profile = (
            subject.candidate_parent._load_validated_base_inputs()
        )
        library = self.view["safe_library"]
        view_items = {row["item_handle"]: row for row in self.view["items"]}
        for row in self.natural0.thaw()["items"]:
            item = view_items[row["item_handle"]]
            style = dict(item["effective_style"])
            if item["title_nonempty"]:
                pure_title = subject.pure_renderer._render_base_title(
                    skeleton=library["title_skeletons"][row["title_skeleton_index"]],
                    product=row["product"],
                    attribute=row["attribute"],
                    code=item["code"],
                    style=style,
                    library=library,
                )
                production_title = subject.renderer.render_base_title(
                    skeleton=library["title_skeletons"][row["title_skeleton_index"]],
                    product=row["product"],
                    attribute=row["attribute"],
                    code=item["code"],
                    style=style,
                    template=template,
                )
                self.assertEqual(pure_title, production_title)
            if item["description_nonempty"]:
                pure_description = subject.pure_renderer._render_base_description(
                    skeleton=library["description_skeletons"][
                        row["description_skeleton_index"]
                    ],
                    product=row["product"],
                    attribute=row["attribute"],
                    code=item["code"],
                    delivery=row["delivery"],
                    service=row["service"],
                    style=style,
                    library=library,
                )
                production_description = subject.renderer.render_base_description(
                    skeleton=library["description_skeletons"][
                        row["description_skeleton_index"]
                    ],
                    product=row["product"],
                    attribute=row["attribute"],
                    code=item["code"],
                    delivery=row["delivery"],
                    service=row["service"],
                    style=style,
                    template=template,
                )
                self.assertEqual(pure_description, production_description)
        for template_index, skeleton in enumerate(library["must_ignore_templates"]):
            for value in library["must_ignore_values"]:
                self.assertEqual(
                    skeleton.format(value=value),
                    subject.renderer.must_ignore_clause(
                        template_index=template_index,
                        value=value,
                        template=template,
                    ),
                )

    def test_private_overrides_are_hidden_from_pure_view_and_survive_trusted_assembly(self) -> None:
        serialized_view = json.dumps(self.view, ensure_ascii=False, sort_keys=True)
        self.assertNotIn("high_semantic_similarity", serialized_view)
        self.assertNotIn("exact_title_clone", serialized_view)
        assembled_items = {
            row["item_uid"]: row for row in self.worlds[0]["public"]["items"]
        }
        pure_items = {
            row["item_handle"]: row for row in self.natural0.thaw()["items"]
        }
        uid_to_handle = {
            uid: handle
            for handle, uid in self.binding["item_handle_to_item_uid"].items()
        }
        for override in self.binding["registered_overrides"]:
            if override["override_kind"] == "high_semantic_similarity":
                left = pure_items[uid_to_handle[override["item_uid_left"]]]
                right = pure_items[uid_to_handle[override["item_uid_right"]]]
                self.assertEqual(
                    (left["category"], left["product"], left["attribute"]),
                    (right["category"], right["product"], right["attribute"]),
                )
                self.assertNotEqual(
                    left["title_skeleton_index"], right["title_skeleton_index"]
                )
            else:
                left = assembled_items[override["item_uid_left"]]
                right = assembled_items[override["item_uid_right"]]
                self.assertEqual(left["title"], right["title"])
        trusted_natural = self.natural0.thaw()
        trusted_rows = {
            row["item_handle"]: row for row in trusted_natural["items"]
        }
        for override in self.binding["registered_overrides"]:
            if override["override_kind"] == "exact_title_clone":
                source = uid_to_handle[override["item_uid_left"]]
                destination = uid_to_handle[override["item_uid_right"]]
                trusted_rows[destination]["title"] = trusted_rows[source]["title"]
        trusted_natural["items"] = [
            trusted_rows[handle]
            for handle in sorted(trusted_rows, key=lambda value: value.encode("utf-8"))
        ]
        trusted_sha256 = hashlib.sha256(
            common.canonical_json_bytes(trusted_natural)
        ).hexdigest()
        self.assertEqual(trusted_sha256, self.results[0].natural_output_sha256)
        self.assertNotEqual(self.natural0.output_sha256, trusted_sha256)

    def test_every_candidate_rebuilds_378_rows_with_33_identity_features(self) -> None:
        for result in self.results:
            rows = json.loads(result.identity33_bytes.decode("utf-8"))
            self.assertEqual(len(rows), 378)
            self.assertEqual(result.identity33_sha256, hashlib.sha256(result.identity33_bytes).hexdigest())

    def test_noncanonical_and_semantically_tampered_safe_views_fail(self) -> None:
        with self.assertRaises(subject.NaturalVariationError):
            subject.render_candidate_natural_expressions(
                restricted_view=subject.RestrictedCandidateView(
                    view_bytes=b'{"b":1, "a":2}',
                    view_sha256=hashlib.sha256(b'{"b":1, "a":2}').hexdigest(),
                ),
                candidate_key=self.candidate0_key,
            )
        tampered = copy.deepcopy(self.view)
        tampered["safe_library"]["categories"][0] = "未登记类别"
        tampered["safe_library_sha256"] = common.canonical_sha256(tampered["safe_library"])
        payload = common.canonical_json_bytes(tampered)
        with self.assertRaises(subject.NaturalVariationError):
            subject.render_candidate_natural_expressions(
                restricted_view=subject.RestrictedCandidateView(
                    view_bytes=payload,
                    view_sha256=hashlib.sha256(payload).hexdigest(),
                ),
                candidate_key=self.candidate0_key,
            )

    def test_wrong_candidate_key_lengths_fail_closed(self) -> None:
        for key in (b"", b"x" * 31, b"x" * 33, "x" * 64):
            with self.subTest(key=key):
                with self.assertRaises(subject.NaturalVariationError):
                    subject.render_candidate_natural_expressions(
                        restricted_view=self.session._view,
                        candidate_key=key,  # type: ignore[arg-type]
                    )

    def test_trusted_assembler_rejects_arbitrary_or_wrong_index_candidate_key(self) -> None:
        arbitrary_key = b"\x00" * 32
        self.assertNotEqual(arbitrary_key, self.candidate0_key)
        arbitrary = subject.render_candidate_natural_expressions(
            restricted_view=self.view_object,
            candidate_key=arbitrary_key,
        )
        with self.assertRaises(subject.NaturalVariationError):
            subject._assemble_and_validate(
                candidate_index=0,
                expected_parent=self.session._parent,
                frozen=self.session._frozen,
                natural=arbitrary,
            )
        document_key = bytes.fromhex(
            self.policy["development_smoke_keys"]["document_variation_key_hex"]
        )
        candidate1_key = collision.derive_candidate_key(
            document_variation_key=document_key,
            split=self.session._frozen.split,
            world_uid=self.session._frozen.world_uid,
            candidate_index=1,
        )
        candidate1 = subject.render_candidate_natural_expressions(
            restricted_view=self.view_object,
            candidate_key=candidate1_key,
        )
        with self.assertRaises(subject.NaturalVariationError):
            subject._assemble_and_validate(
                candidate_index=0,
                expected_parent=self.session._parent,
                frozen=self.session._frozen,
                natural=candidate1,
            )

    def test_trusted_assembler_rejects_forged_natural_bytes_and_alternate_view(self) -> None:
        tampered_value = self.natural0.thaw()
        tampered_value["items"][0]["category"] += "_tamper"
        tampered_bytes = common.canonical_json_bytes(tampered_value)
        tampered = subject.NaturalExpressionCandidate(
            output_bytes=tampered_bytes,
            output_sha256=hashlib.sha256(tampered_bytes).hexdigest(),
            view_sha256=self.natural0.view_sha256,
            candidate_key_sha256=self.natural0.candidate_key_sha256,
        )
        with self.assertRaises(subject.NaturalVariationError):
            subject._assemble_and_validate(
                candidate_index=0,
                expected_parent=self.session._parent,
                frozen=self.session._frozen,
                natural=tampered,
            )

        alternate_view = copy.deepcopy(self.view)
        alternate_view["items"] = list(reversed(alternate_view["items"]))
        alternate_bytes = common.canonical_json_bytes(alternate_view)
        alternate = subject.RestrictedCandidateView(
            view_bytes=alternate_bytes,
            view_sha256=hashlib.sha256(alternate_bytes).hexdigest(),
        )
        alternate_natural = subject.render_candidate_natural_expressions(
            restricted_view=alternate,
            candidate_key=self.candidate0_key,
        )
        with self.assertRaises(subject.NaturalVariationError):
            subject._assemble_and_validate(
                candidate_index=0,
                expected_parent=self.session._parent,
                frozen=self.session._frozen,
                natural=alternate_natural,
            )

    def test_trusted_assembler_exposes_no_view_binding_or_candidate_key_parameter(self) -> None:
        parameters = inspect.signature(subject._assemble_and_validate).parameters
        self.assertEqual(
            list(parameters),
            ["candidate_index", "expected_parent", "frozen", "natural"],
        )

    def test_trusted_assembler_rejects_forged_parent_and_frozen_full_state_first(self) -> None:
        forged_cases = (
            (
                self.session._parent,
                replace(self.session._frozen, identity_parent_sha256="0" * 64),
            ),
            (
                self.session._parent,
                replace(self.session._frozen, world_bytes=self.results[1].world_bytes),
            ),
            (
                replace(self.session._parent, invariant_sha256="1" * 64),
                replace(self.session._frozen, candidate_invariant_sha256="1" * 64),
            ),
        )
        for parent, frozen in forged_cases:
            with self.subTest(
                parent_invariant=parent.invariant_sha256,
                frozen_identity_parent=frozen.identity_parent_sha256,
            ):
                with mock.patch.object(
                    subject,
                    "_build_restricted_view_and_binding",
                    side_effect=AssertionError("forged state reached redraw"),
                ):
                    with self.assertRaises(subject.NaturalVariationError):
                        subject._assemble_and_validate(
                            candidate_index=0,
                            expected_parent=parent,
                            frozen=frozen,
                            natural=self.natural0,
                        )

    def test_binding_commitment_tampering_fails(self) -> None:
        binding = self.binding_object
        tampered = subject._PrivateHandleBinding(
            binding_bytes=binding.binding_bytes,
            binding_sha256=binding.binding_sha256,
            view_sha256="0" * 64,
        )
        with self.assertRaises(subject.NaturalVariationError):
            tampered.thaw()

        forged_value = copy.deepcopy(self.binding)
        handles = list(forged_value["item_handle_to_item_uid"])
        left, right = handles[:2]
        forged_value["item_handle_to_item_uid"][left], forged_value[
            "item_handle_to_item_uid"
        ][right] = (
            forged_value["item_handle_to_item_uid"][right],
            forged_value["item_handle_to_item_uid"][left],
        )
        forged_bytes = common.canonical_json_bytes(forged_value)
        forged = subject._PrivateHandleBinding(
            binding_bytes=forged_bytes,
            binding_sha256=hashlib.sha256(forged_bytes).hexdigest(),
            view_sha256=self.view_object.view_sha256,
        )
        self.assertIsInstance(forged.thaw(), dict)
        with self.assertRaises(TypeError):
            subject._assemble_and_validate(
                candidate_index=0,
                expected_parent=self.session._parent,
                frozen=self.session._frozen,
                natural=self.natural0,
                binding=forged,  # type: ignore[call-arg]
            )

    def test_session_requires_order_and_is_poisoned_after_failure(self) -> None:
        session = subject.DevelopmentSmokeVariationSession.__new__(
            subject.DevelopmentSmokeVariationSession
        )
        session._failed = False
        session._rendered_indices = set()
        with self.assertRaises(subject.NaturalVariationError):
            session.render(1)
        self.assertTrue(session._failed)
        with self.assertRaises(subject.NaturalVariationError):
            session.render(0)

    def test_session_is_poisoned_by_keyboard_interrupt_and_system_exit(self) -> None:
        for failure in (KeyboardInterrupt(), SystemExit(9)):
            session = subject.DevelopmentSmokeVariationSession.__new__(
                subject.DevelopmentSmokeVariationSession
            )
            session._policy = self.policy
            session._parent = self.session._parent
            session._frozen = self.session._frozen
            session._view = self.view_object
            session._rendered_indices = set()
            session._failed = False
            with self.subTest(failure=type(failure).__name__):
                with mock.patch.object(
                    subject,
                    "render_candidate_natural_expressions",
                    side_effect=failure,
                ):
                    with self.assertRaises(type(failure)):
                        session.render(0)
                self.assertTrue(session._failed)
                with self.assertRaises(subject.NaturalVariationError):
                    session.render(1)

    def test_completed_session_rejects_duplicate_or_out_of_order_candidate(self) -> None:
        with self.assertRaises(subject.NaturalVariationError):
            self.session.render(0)
        self.assertTrue(self.session._failed)

    def test_source_has_no_filesystem_mutation_calls(self) -> None:
        forbidden_attributes = {
            "write_text",
            "write_bytes",
            "mkdir",
            "unlink",
            "replace",
            "rename",
            "rmdir",
        }
        for name in (
            "step28_v13_v1_13_natural_variation.py",
            "step28_v13_v1_13_pure_natural_renderer.py",
        ):
            source_path = SCRIPTS / name
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"), filename=str(source_path)
            )
            calls = {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            with self.subTest(name=name):
                self.assertFalse(calls & forbidden_attributes)

    def test_main_prints_only_design_smoke_receipt_and_zero_formal_counts(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            subject.main()
        receipt = json.loads(stream.getvalue())
        self.assertEqual(receipt["status"], "PASS_DEVELOPMENT_SMOKE_NATURAL_VARIATION_ONLY")
        self.assertEqual(receipt["formal_seeds_generated"], 0)
        self.assertEqual(receipt["formal_rows_generated"], 0)
        self.assertEqual(receipt["formal_models_trained"], 0)
        self.assertEqual(receipt["in_memory_candidate_count"], 1)
        self.assertNotIn("identity_value", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
