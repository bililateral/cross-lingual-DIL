from __future__ import annotations

import copy
import ast
import hashlib
import inspect
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common  # noqa: E402
import step28_v13_v1_13_candidate_parent as subject  # noqa: E402


class Step28V13V113CandidateParentContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage_policy = subject.load_policy()
        cls.base_policy = common.load_policy(
            ROOT / "schema" / "step28_v13_synthetic_chinese_dataset_policy.json",
            mode="development_smoke",
        )
        cls.template = common.load_json(
            ROOT / "schema" / "step28_v13_synthetic_text_templates.json"
        )
        cls.parent = subject.build_candidate_independent_parent()
        cls.split = cls.parent.split
        cls.registries = subject.collision.load_historical_exclusion_registries()
        cls.allocator = subject.OneTimeTrialIdentityAllocator()
        with (
            mock.patch.object(
                subject.collision,
                "load_historical_exclusion_registries",
                return_value=cls.registries,
            ) as authority,
            mock.patch.object(
                subject.identity_remap,
                "remap_world_identity_values",
                wraps=subject.identity_remap.remap_world_identity_values,
            ) as remap,
        ):
            cls.identity_parent = cls.allocator.allocate(
                parent=cls.parent,
            )
            cls.authority_call_count = authority.call_count
            cls.remap_call_count = remap.call_count
            cls.repeat_identity_parent = (
                subject.OneTimeTrialIdentityAllocator().allocate(
                    parent=cls.parent,
                )
            )

    def _projection(self, world: dict) -> dict:
        return subject.candidate_invariant_projection(
            policy=self.base_policy,
            template=self.template,
            mode="development_smoke",
            split=self.split,
            world=world,
            profile_provenance=self.parent.thaw_profile_provenance(),
        )

    def _validate_receipt(self, receipt: dict) -> None:
        world = self.identity_parent.thaw_world()
        trial_contract = self.stage_policy["trial_identity_allocation"]
        identity_key = self.base_policy["randomness"]["development_smoke"][
            "identity_value_key_hex"
        ]
        subject._validate_allocation_receipt(
            receipt,
            world_uid=self.parent.world_uid,
            identity_asset_count=len(world["private"]["identity_assets"]),
            identity_slot_count=len(world["private"]["identity_slots_audit"]),
            changed_item_count=len(
                {row["item_uid"] for row in world["private"]["identity_slots_audit"]}
            ),
            maximum_counter=128,
            allocation_delta=self.identity_parent.allocation_delta,
            allocated_asset_hashes={
                row["identity_asset_uid"]: subject.identity_values.value_hash(
                    row["identity_value"]
                )
                for row in world["private"]["identity_assets"]
            },
            expected_allocation_audit_rows=subject._replay_identity_allocation_audit(
                parent_world=self.parent.thaw_bootstrap_world(),
                allocated_world=world,
                template=self.template,
                key_hex=identity_key,
                historical_forbidden=self.registries.identity_value_hashes,
                maximum_counter=trial_contract["maximum_counter"],
            ),
        )

    def test_policy_is_design_only_and_exact_dependency_closure_is_pinned(self) -> None:
        policy = subject.load_policy()
        self.assertTrue(
            all(value is False for value in policy["formal_authorizations"].values())
        )
        self.assertEqual(policy["allowed_mode"], "development_smoke")
        self.assertEqual(set(policy["frozen_inputs"]), subject.FROZEN_INPUT_KEYS)
        self.assertFalse(policy["trial_identity_allocation"]["formal_custody_released"])
        self.assertNotIn("v1_12", json.dumps(policy["frozen_inputs"]))

    def test_local_python_import_closure_matches_exact_policy_pins(self) -> None:
        entry = SCRIPTS / "step28_v13_v1_13_candidate_parent.py"
        pending = [entry]
        discovered: set[Path] = set()
        while pending:
            path = pending.pop()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module_names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    module_names.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module_names.add(node.module.split(".")[0])
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    module_names.add(node.args[0].value.split(".")[0])
            for module_name in module_names:
                candidate = SCRIPTS / f"{module_name}.py"
                if candidate.is_file() and candidate not in discovered:
                    discovered.add(candidate)
                    pending.append(candidate)
        pinned_python = {
            common.repo_path(spec["path"]).resolve()
            for spec in self.stage_policy["frozen_inputs"].values()
            if str(spec["path"]).endswith(".py")
        }
        self.assertEqual({path.resolve() for path in discovered}, pinned_python)

    def test_re_self_hashed_policy_semantic_tamper_fails(self) -> None:
        for mutate in (
            lambda value: value["candidate_independent_parent"][
                "candidate_invariant_render_ast_projection"
            ].remove("code"),
            lambda value: value["trial_identity_allocation"].__setitem__(
                "historical_identity_hash_count", 0
            ),
            lambda value: value["frozen_inputs"].pop("source_data"),
            lambda value: value["profile_contribution_provenance"].__setitem__(
                "private_audit_only", False
            ),
        ):
            policy = copy.deepcopy(self.stage_policy)
            mutate(policy)
            policy.pop("canonical_self_hash")
            policy["canonical_self_hash"] = common.canonical_sha256(policy)
            with self.assertRaises(subject.CandidateParentError):
                subject._validate_policy_document(policy)

    def test_parent_builder_has_no_caller_selected_inputs(self) -> None:
        self.assertEqual(list(inspect.signature(subject.build_candidate_independent_parent).parameters), [])
        parent = subject.build_candidate_independent_parent()
        smoke = self.stage_policy["development_smoke_parent"]
        self.assertEqual(parent.world_uid, smoke["world_uid"])
        self.assertEqual(parent.split, smoke["split"])
        self.assertEqual(parent, self.parent)

    def test_parent_builder_revalidates_release_documents_and_style_pin(self) -> None:
        with (
            mock.patch.object(
                subject.common,
                "validate_policy_release_documents",
                wraps=subject.common.validate_policy_release_documents,
            ) as release_validator,
            mock.patch.object(
                subject.common,
                "verify_file_pin",
                wraps=subject.common.verify_file_pin,
            ) as pin_validator,
        ):
            self.assertEqual(subject.build_candidate_independent_parent(), self.parent)
        self.assertEqual(release_validator.call_count, 1)
        self.assertTrue(
            any(
                call.kwargs.get("label") == "candidate-parent style profile"
                for call in pin_validator.call_args_list
            )
        )

    def test_parent_is_canonical_immutable_bytes_with_expected_shape(self) -> None:
        parent = self.parent
        self.assertEqual(
            hashlib.sha256(parent.invariant_projection_bytes).hexdigest(),
            parent.invariant_sha256,
        )
        world_a = parent.thaw_bootstrap_world()
        world_b = parent.thaw_bootstrap_world()
        self.assertIsNot(world_a, world_b)
        world_a["public"]["items"][0]["title"] = "tampered"
        self.assertNotEqual(world_a, world_b)
        self.assertEqual(len(world_b["public"]["sellers"]), 28)
        self.assertEqual(len(world_b["public"]["complete_model_pair_endpoints"]), 378)

    def test_visible_text_and_seven_candidate_ast_fields_are_not_invariant(self) -> None:
        baseline_world = self.parent.thaw_bootstrap_world()
        baseline = self._projection(baseline_world)
        visible = copy.deepcopy(baseline_world)
        visible["public"]["items"][0]["title"] += " 允许的候选表述"
        visible["public"]["items"][0]["description"] += " 允许的候选表述"
        self.assertEqual(baseline, self._projection(visible))
        for field in (
            "category",
            "product",
            "attribute",
            "delivery",
            "service",
            "title_skeleton_index",
            "description_skeleton_index",
        ):
            changed = copy.deepcopy(baseline_world)
            row = changed["private"]["render_asts"][0]
            row[field] = int(row[field]) + 1 if isinstance(row[field], int) else str(row[field]) + "_candidate"
            self.assertEqual(baseline, self._projection(changed), field)

    def test_candidate_invariant_ast_fields_each_change_fingerprint(self) -> None:
        baseline_world = self.parent.thaw_bootstrap_world()
        baseline = self._projection(baseline_world)
        for field in subject.CANDIDATE_INVARIANT_RENDER_AST_FIELDS:
            changed = copy.deepcopy(baseline_world)
            row = changed["private"]["render_asts"][0]
            value = row[field]
            if isinstance(value, bool):
                row[field] = not value
            elif isinstance(value, int):
                row[field] = value + 1
            elif isinstance(value, list):
                row[field] = [*value, "slot_tamper"]
            else:
                row[field] = str(value) + "_tamper"
            try:
                changed_projection = self._projection(changed)
            except subject.CandidateParentError:
                continue
            self.assertNotEqual(baseline, changed_projection, field)

    def test_override_lineage_change_alters_fingerprint(self) -> None:
        world = self.parent.thaw_bootstrap_world()
        baseline = self._projection(world)
        changed = copy.deepcopy(world)
        changed["private"]["override_audit"][0]["override_kind"] += "_tamper"
        self.assertNotEqual(baseline, self._projection(changed))

    def test_full_render_ast_schema_still_fails_closed(self) -> None:
        world = self.parent.thaw_bootstrap_world()
        world["private"]["render_asts"][0]["unexpected"] = 1
        with self.assertRaises(subject.CandidateParentError):
            self._projection(world)

    def test_duplicate_effective_style_keys_fail_before_dictionary_build(self) -> None:
        for table, duplicate_index in (
            ("controller_membership", 0),
            ("controller_style_groups", 0),
        ):
            world = self.parent.thaw_bootstrap_world()
            world["private"][table].append(copy.deepcopy(world["private"][table][duplicate_index]))
            with self.assertRaises(subject.CandidateParentError):
                self._projection(world)
        template = copy.deepcopy(self.template)
        template["style_prototypes"].append(copy.deepcopy(template["style_prototypes"][0]))
        with self.assertRaises(subject.CandidateParentError):
            subject.candidate_invariant_projection(
                policy=self.base_policy,
                template=template,
                mode="development_smoke",
                split=self.split,
                world=self.parent.thaw_bootstrap_world(),
                profile_provenance=self.parent.thaw_profile_provenance(),
            )

    def test_parent_projection_contains_all_six_effective_style_factors(self) -> None:
        projection = json.loads(self.parent.invariant_projection_bytes)
        factors = self.template["renderer_contract"]["style_factor_order"]
        self.assertEqual(len(factors), 6)
        self.assertEqual(len(projection["effective_styles"]), 28)
        for row in projection["effective_styles"]:
            self.assertEqual(list(row["style_factors"]), sorted(factors))

    def test_provenance_covers_exact_five_profile_fields_without_values(self) -> None:
        provenance = self.parent.thaw_profile_provenance()
        rows = provenance["rows"]
        self.assertGreater(len(rows), 0)
        self.assertEqual(provenance["seller_count"], 28)
        self.assertEqual(
            {row["output_field"] for row in rows},
            {role[0] for role in subject.PROFILE_ROLES},
        )
        self.assertTrue(provenance["private_audit_only"])
        self.assertFalse(provenance["raw_contribution_values_persisted"])
        self.assertTrue(all("value" not in row for row in rows))
        self.assertEqual(provenance["rows_sha256"], common.canonical_sha256(rows))

    def test_every_provenance_support_set_and_digest_is_exact(self) -> None:
        world_item_uids = {
            row["item_uid"]
            for row in self.parent.thaw_bootstrap_world()["public"]["items"]
        }
        for row in self.parent.thaw_profile_provenance()["rows"]:
            support = row["source_item_uids"]
            self.assertEqual(support, common.utf8_sort(set(support)))
            self.assertTrue(set(support).issubset(world_item_uids))
            self.assertEqual(row["source_item_count"], len(support))
            self.assertEqual(row["source_item_uids_sha256"], common.canonical_sha256(support))

    def test_description_signature_provenance_has_segment_and_df_lineage(self) -> None:
        rows = [
            row
            for row in self.parent.thaw_profile_provenance()["rows"]
            if row["output_field"] == "signature_description_concat"
        ]
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn(row["item_uid"], row["source_item_uids"])
            self.assertGreater(row["extracted_segment_ordinal"], 0)
            self.assertGreater(row["seller_df"], 0)
            self.assertEqual(row["seller_df"], row["seller_df_seller_count"])
            self.assertRegex(row["seller_df_seller_uids_sha256"], r"^[0-9a-f]{64}$")

    def test_provenance_tamper_fails_closed(self) -> None:
        profiles = self.parent.thaw_profiles()
        world = self.parent.thaw_bootstrap_world()
        processed = subject.production.process_world(
            self.base_policy,
            mode="development_smoke",
            split=self.split,
            template=self.template,
            world=world,
        )
        profiles[0]["top_titles"][0]["value"] = "不存在的伪造贡献值"
        profiles[0]["title_concat_top"] = subject.step3.concat_top(profiles[0]["top_titles"])
        with self.assertRaises(subject.CandidateParentError):
            subject.build_profile_contribution_provenance(
                world_uid=self.parent.world_uid,
                profiles=profiles,
                profile_safe_items=processed["public"]["profile_safe_items"],
            )

    def test_allocator_api_exposes_no_history_key_counter_or_fault_hook(self) -> None:
        parameters = set(inspect.signature(subject.OneTimeTrialIdentityAllocator.allocate).parameters)
        self.assertEqual(parameters, {"self", "parent"})
        self.assertNotIn("v1_12", inspect.getsource(subject.identity_remap))
        self.assertEqual(
            subject.identity_remap.IDENTITY_VALUE_DOMAIN,
            b"step28-v13-v1.13-identity-value",
        )

    def test_allocator_uses_authoritative_full_history_and_calls_remapper_once(self) -> None:
        self.assertEqual(self.authority_call_count, 1)
        self.assertEqual(self.remap_call_count, 1)
        self.assertEqual(
            len(self.registries.identity_value_hashes),
            subject.HISTORICAL_IDENTITY_HASH_COUNT,
        )
        self.assertEqual(
            common.canonical_sha256(common.utf8_sort(self.registries.identity_value_hashes)),
            subject.HISTORICAL_IDENTITY_HASHES_SHA256,
        )

    def test_two_fresh_allocators_have_identical_frozen_result(self) -> None:
        self.assertEqual(self.identity_parent, self.repeat_identity_parent)

    def test_allocator_entry_is_consumed_on_mid_remap_error(self) -> None:
        allocator = subject.OneTimeTrialIdentityAllocator()

        def fail_midway(*args, **kwargs):
            kwargs["allocated_in_trial"].add("03" * 32)
            raise RuntimeError("injected mid-remap failure")

        with (
            mock.patch.object(subject, "build_candidate_independent_parent", return_value=self.parent),
            mock.patch.object(
                subject.collision,
                "load_historical_exclusion_registries",
                return_value=self.registries,
            ),
            mock.patch.object(subject.identity_remap, "remap_world_identity_values", side_effect=fail_midway),
        ):
            with self.assertRaises(RuntimeError):
                allocator.allocate(parent=self.parent)
        with self.assertRaises(subject.CandidateParentError):
            allocator.allocate(parent=self.parent)

    def test_caller_cannot_supply_nonempty_committed_screening_set(self) -> None:
        allocator = subject.OneTimeTrialIdentityAllocator()
        with self.assertRaises(TypeError):
            allocator.allocate(  # type: ignore[call-arg]
                parent=self.parent,
                committed_identity_hashes=frozenset({"03" * 32}),
            )

    def test_trial_allocation_uses_no_caller_committed_state(self) -> None:
        delta = self.identity_parent.allocation_delta
        self.assertGreater(len(delta), 0)
        self.assertEqual(len(delta), len(set(delta)))
        self.assertFalse(set(delta) & set(self.registries.identity_value_hashes))
        world = self.identity_parent.thaw_world()
        observed = {
            subject.identity_values.value_hash(row["identity_value"])
            for row in world["private"]["identity_assets"]
        }
        self.assertEqual(set(delta), observed)

    def test_trial_allocation_receipt_is_independently_recomputed(self) -> None:
        receipt = json.loads(self.identity_parent.allocation_receipt_bytes)
        self._validate_receipt(receipt)
        for field, value in (
            ("same_run_intersection_count", 1),
            ("changed_item_count", receipt["changed_item_count"] + 1),
            ("forced_design_collision_count", 1),
            ("visible_text_candidate_rejection_count", -1),
            ("identity_asset_count", -1),
            ("maximum_selected_counter", receipt["maximum_selected_counter"] + 1),
        ):
            forged = dict(receipt)
            forged[field] = value
            with self.assertRaises(subject.CandidateParentError, msg=field):
                self._validate_receipt(forged)
        forged = copy.deepcopy(receipt)
        forged["allocation_audit_rows"][0]["visible_text_candidate_rejection_count"] += 1
        forged["allocation_audit_rows_sha256"] = common.canonical_sha256(
            forged["allocation_audit_rows"]
        )
        forged["visible_text_candidate_rejection_count"] += 1
        with self.assertRaises(subject.CandidateParentError):
            self._validate_receipt(forged)

    def test_identity_parent_freezes_identity33_and_profile_lineage(self) -> None:
        state = self.identity_parent
        rows = state.thaw_identity33()
        self.assertEqual(len(rows), 378)
        self.assertEqual(hashlib.sha256(state.identity33_bytes).hexdigest(), state.identity33_sha256)
        self.assertEqual(state.candidate_invariant_sha256, self.parent.invariant_sha256)
        self.assertEqual(state.profile_provenance_sha256, self.parent.profile_provenance_sha256)
        self.assertEqual(state.profile_sha256, self.parent.profile_sha256)

    def test_trial_state_exposes_no_key_or_live_identity_set(self) -> None:
        fields = set(vars(self.identity_parent))
        self.assertFalse(any("key" in name for name in fields))
        self.assertFalse(any("set" in name for name in fields))
        self.assertIsInstance(self.identity_parent.allocation_delta, tuple)


if __name__ == "__main__":
    unittest.main()
