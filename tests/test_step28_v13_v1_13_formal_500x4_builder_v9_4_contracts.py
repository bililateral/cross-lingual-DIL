from __future__ import annotations

import importlib
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

formal = importlib.import_module(
    "step28_v13_v1_13_formal_500x4_builder_v9_4"
)
authority = importlib.import_module(
    "step28_v13_v1_13_formal_500x4_authority_v9_4"
)


class Formal500x4BuilderV94Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(formal.POLICY_PATH.read_text(encoding="utf-8"))

    def test_policy_keeps_generation_training_and_audit_truth_closed(self) -> None:
        self.assertEqual(
            self.policy["status"],
            "READY_FOR_ONE_TIME_FORMAL_AUTHORIZATION",
        )
        self.assertEqual(
            self.policy["authorization"],
            {
                "formal_build": False,
                "training_qualification": False,
                "audit_truth_unsealing": False,
                "model_training": False,
            },
        )
        self.assertEqual(
            self.policy["world_counts"],
            {split: 500 for split in formal.SPLITS},
        )
        self.assertEqual(sum(self.policy["world_counts"].values()), 2000)

    def test_policy_binds_passed_method_root_without_training_claim(self) -> None:
        observed = formal.validate_policy(formal=False)
        quality_path = ROOT / observed["method_qualification"][
            "quality_result_path"
        ]
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        self.assertEqual(quality["status"], formal.QUALITY_STATUS)
        self.assertTrue(quality["eligible_for_formal_500x4_generation_application"])
        self.assertFalse(quality["formal_500x4_generated"])
        self.assertFalse(quality["training_qualified"])
        self.assertFalse(quality["m0_m1_m2_m3_training_authorized"])
        self.assertEqual(quality["truth_access"]["audit_a_semantic_reads"], 0)
        self.assertEqual(quality["truth_access"]["audit_b_semantic_reads"], 0)

    def test_formal_mode_accepts_frozen_self_hashed_policy(self) -> None:
        observed = formal.validate_policy(formal=True)
        self.assertEqual(observed["status"], formal.POLICY_READY_STATUS)
        formal.require_self_hash(observed, label="formal policy")

    @staticmethod
    def fake_basis(split: str) -> SimpleNamespace:
        public_worlds = []
        groups_by_world = []
        sizes = (3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2)
        for world_ordinal in range(500):
            world_uid = f"basis_{split}_{world_ordinal:03d}"
            sellers = tuple(
                f"{world_uid}_seller_{slot:02d}" for slot in range(28)
            )
            public_worlds.append({
                "seller_uids": sellers,
                "noise_slot_by_seller_slot": tuple(range(28)),
            })
            groups = []
            cursor = 0
            for size in sizes:
                groups.append(tuple(sellers[cursor:cursor + size]))
                cursor += size
            groups_by_world.append(tuple(groups))
        return SimpleNamespace(
            public_worlds=tuple(public_worlds),
            controller_groups_by_world=tuple(groups_by_world),
        )

    def test_formal_schedule_is_new_namespaced_and_split_separated(self) -> None:
        fake = {
            "train": self.fake_basis("train"),
            "development": self.fake_basis("development"),
        }
        key = bytes(range(32))
        commitments = []
        all_world_uids: set[str] = set()
        all_seller_uids: set[str] = set()
        with mock.patch.object(
            formal,
            "_load_basis_schedule",
            side_effect=lambda name: fake[name],
        ):
            for split in formal.SPLITS:
                worlds, commitment = formal._transform_schedule(
                    split=split, schedule_key=key, policy=self.policy,
                )
                self.assertEqual(len(worlds), 500)
                self.assertTrue(all(
                    world.world_uid.startswith(
                        f"v9_4_formal_500x4_{split}_world_"
                    )
                    for world in worlds
                ))
                self.assertFalse(all_world_uids.intersection(
                    world.world_uid for world in worlds
                ))
                all_world_uids.update(world.world_uid for world in worlds)
                split_sellers = {
                    seller for world in worlds for seller in world.seller_uids
                }
                self.assertEqual(len(split_sellers), 14000)
                self.assertFalse(all_seller_uids.intersection(split_sellers))
                all_seller_uids.update(split_sellers)
                for world in worlds:
                    self.assertEqual(len(world.seller_uids), 28)
                    self.assertEqual(set(world.noise_slots), set(range(28)))
                    self.assertEqual(
                        sorted(len(group) for group in world.controller_groups),
                        [2] * 8 + [3] * 4,
                    )
                commitments.append(commitment)
        for name in (
            "world_order_sha256",
            "seller_slot_permutation_sha256",
            "noise_slot_permutation_sha256",
            "transformed_schedule_sha256",
        ):
            self.assertEqual(
                len({commitment[name] for commitment in commitments}), 4,
            )
        self.assertEqual(len(all_world_uids), 2000)
        self.assertEqual(len(all_seller_uids), 56000)

    def test_formal_schedule_rejects_same_transform_across_splits(self) -> None:
        fake = {
            "train": self.fake_basis("train"),
            "development": self.fake_basis("development"),
        }
        with (
            mock.patch.object(
                formal,
                "_load_basis_schedule",
                side_effect=lambda name: fake[name],
            ),
            mock.patch.object(
                formal,
                "_ranked_ints",
                side_effect=lambda values, key, *parts: list(values),
            ),
        ):
            with self.assertRaisesRegex(
                formal.Formal500x4BuildError,
                "transform commitments collide",
            ):
                formal.build_world_schedules(
                    formal=True,
                    authorities=formal.smoke_authorities(),
                    policy=self.policy,
                )

    def test_noise_permutation_preserves_market_equivalence(self) -> None:
        for split in formal.SPLITS:
            permutation = formal._market_preserving_noise_permutation(
                bytes(range(32)), split,
            )
            self.assertEqual(set(permutation), set(range(28)))
            self.assertTrue(all(
                source % len(formal.engine.MARKETS)
                == target % len(formal.engine.MARKETS)
                for source, target in enumerate(permutation)
            ))

    def test_schedule_balance_gate_rejects_repeated_unbalanced_world(self) -> None:
        schedules = {}
        for split in formal.SPLITS:
            source = formal.engine._smoke_world(split)
            schedules[split] = tuple(source for _ in range(500))
        with self.assertRaisesRegex(
            formal.Formal500x4BuildError,
            "balanced schedule invariant drift",
        ):
            formal.audit_formal_schedule_balance(schedules)

    def test_collision_registration_is_label_free_and_fail_closed(self) -> None:
        state = formal.CollisionState(
            blocked_item_documents=set(),
            blocked_seller_documents=set(),
            blocked_world_uids=set(),
            blocked_seller_uids=set(),
            blocked_item_uids=set(),
            blocked_pair_uids=set(),
            blocked_historical_uid_hashes={
                "canonical_pair_uid": set(),
                "controller_uid": set(),
                "item_uid": set(),
                "query_uid": set(),
                "seller_uid": set(),
                "world_uid": set(),
            },
            blocked_identity_values=set(),
        )
        seller_uids = [f"seller_{index:02d}" for index in range(28)]
        value = {
            "world": {"world_uid": "formal_world"},
            "sellers": [
                {"seller_uid": seller_uid} for seller_uid in seller_uids
            ],
            "redacted_items": [
                {
                    "item_uid": f"item_{index}",
                    "title": f"标题{index}",
                    "description": f"描述{index}",
                }
                for index in range(28)
            ],
            "model_profiles": [
                {
                    "seller_uid": seller_uids[index],
                    "category_concat_top": f"类别{index}",
                    "signature_title_concat": f"标题签名{index}",
                    "title_concat_top": f"标题全文{index}",
                    "signature_description_concat": f"描述签名{index}",
                    "description_concat_top": f"描述全文{index}",
                }
                for index in range(28)
            ],
            "endpoints": [
                {
                    "canonical_pair_uid": f"{left}||{right}",
                    "world_uid": "formal_world",
                    "seller_uid_left": left,
                    "seller_uid_right": right,
                }
                for left, right in combinations(seller_uids, 2)
            ],
            "identity_plan": [
                {
                    "asset_uid": f"asset_{index}",
                    "value_sha256": f"{index + 1:064x}",
                }
                for index in range(8)
            ],
        }
        self.assertNotIn("labels", value)
        sealed = mock.Mock()
        formal.register_world(
            split="train",
            world=value["world"],
            sellers=value["sellers"],
            redacted_items=value["redacted_items"],
            model_profiles=value["model_profiles"],
            endpoints=value["endpoints"],
            identity_projection=value["identity_plan"],
            state=state,
            sealed_method_identities=sealed,
        )
        sealed.require_disjoint.assert_called_once()
        with self.assertRaises(formal.Formal500x4BuildError):
            formal.register_world(
                split="train",
                world=value["world"],
                sellers=value["sellers"],
                redacted_items=value["redacted_items"],
                model_profiles=value["model_profiles"],
                endpoints=value["endpoints"],
                identity_projection=value["identity_plan"],
                state=state,
                sealed_method_identities=sealed,
            )

    def test_historical_hashed_item_and_pair_uids_are_rejected(self) -> None:
        seller_uids = [f"seller_{index:02d}" for index in range(28)]
        endpoints = [
            {
                "canonical_pair_uid": f"{left}||{right}",
                "world_uid": "formal_world",
                "seller_uid_left": left,
                "seller_uid_right": right,
            }
            for left, right in combinations(seller_uids, 2)
        ]
        items = [
            {
                "item_uid": f"item_{index:02d}",
                "title": f"标题{index}",
                "description": f"描述{index}",
            }
            for index in range(28)
        ]
        profiles = [
            {
                "seller_uid": seller_uid,
                "category_concat_top": f"类别{index}",
                "signature_title_concat": f"标题签名{index}",
                "title_concat_top": f"标题全文{index}",
                "signature_description_concat": f"描述签名{index}",
                "description_concat_top": f"描述全文{index}",
            }
            for index, seller_uid in enumerate(seller_uids)
        ]
        identities = [
            {"asset_uid": "asset_00", "value_sha256": "1" * 64},
        ]

        def state_with(*, item_hashes=(), pair_hashes=(), method_pair_uids=()):
            return formal.CollisionState(
                blocked_item_documents=set(),
                blocked_seller_documents=set(),
                blocked_world_uids=set(),
                blocked_seller_uids=set(),
                blocked_item_uids=set(),
                blocked_pair_uids=set(method_pair_uids),
                blocked_historical_uid_hashes={
                    "canonical_pair_uid": set(pair_hashes),
                    "controller_uid": set(),
                    "item_uid": set(item_hashes),
                    "query_uid": set(),
                    "seller_uid": set(),
                    "world_uid": set(),
                },
                blocked_identity_values=set(),
            )

        kwargs = {
            "split": "train",
            "world": {"world_uid": "formal_world"},
            "sellers": [{"seller_uid": value} for value in seller_uids],
            "redacted_items": items,
            "model_profiles": profiles,
            "endpoints": endpoints,
            "identity_projection": identities,
            "sealed_method_identities": mock.Mock(),
        }
        item_hash = hashlib.sha256(items[0]["item_uid"].encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(
            formal.Formal500x4BuildError,
            "Historical hashed item UID collision",
        ):
            formal.register_world(
                **kwargs, state=state_with(item_hashes=(item_hash,)),
            )
        pair_hash = hashlib.sha256(
            endpoints[0]["canonical_pair_uid"].encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(
            formal.Formal500x4BuildError,
            "Historical hashed canonical pair UID collision",
        ):
            formal.register_world(
                **kwargs, state=state_with(pair_hashes=(pair_hash,)),
            )
        with self.assertRaisesRegex(
            formal.Formal500x4BuildError,
            "Historical or formal canonical pair UID collision",
        ):
            formal.register_world(
                **kwargs,
                state=state_with(method_pair_uids=(
                    endpoints[0]["canonical_pair_uid"],
                )),
            )
        corrupted = state_with(pair_hashes=(pair_hash,))
        corrupted.formal_pair_uids.add(endpoints[0]["canonical_pair_uid"])
        with self.assertRaisesRegex(
            formal.Formal500x4BuildError,
            "collision registry contains an exclusion hit",
        ):
            formal.public_collision_registry(
                corrupted,
                hash_list={
                    "path": "documents.jsonl",
                    "row_count": 0,
                    "size_bytes": 0,
                    "sha256": "0" * 64,
                },
            )

    def test_sealed_method_identity_validator_returns_no_private_rows(self) -> None:
        manifest_path = ROOT / self.policy["method_qualification"][
            "root_manifest_path"
        ]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validator = formal.identity_exclusion.open_validator(
            method_private_root=(
                ROOT / self.policy["method_qualification"]["private_root"]
            ),
            root_manifest=manifest,
        )
        audit = validator.public_audit()
        self.assertEqual(audit["private_values_returned"], 0)
        self.assertEqual(audit["private_rows_returned"], 0)
        self.assertEqual(
            set(audit["projected_fields"]),
            {"asset_uid", "value_sha256"},
        )
        self.assertNotIn("identity_value_hashes", audit)
        self.assertNotIn("identity_asset_uids", audit)

    def test_manifest_payload_validation_rejects_byte_drift_and_extra_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            payload = root / "payload.txt"
            payload.write_text("frozen\n", encoding="utf-8", newline="\n")
            expected = formal.engine.file_manifest(root)
            formal.validate_manifest_payloads(
                root=root, expected_rows=expected, label="fixture",
            )
            payload.write_text("drifted\n", encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(
                formal.Formal500x4BuildError, "committed payload bytes drift",
            ):
                formal.validate_manifest_payloads(
                    root=root, expected_rows=expected, label="fixture",
                )
            payload.write_text("frozen\n", encoding="utf-8", newline="\n")
            (root / "extra.txt").write_text(
                "extra\n", encoding="utf-8", newline="\n",
            )
            with self.assertRaisesRegex(
                formal.Formal500x4BuildError, "committed payload bytes drift",
            ):
                formal.validate_manifest_payloads(
                    root=root, expected_rows=expected, label="fixture",
                )

    def test_authority_issues_six_distinct_commitments_without_key_material(self) -> None:
        policy = json.loads(json.dumps(self.policy))
        suffix = "_test_v9_4_formal_500x4_authority"
        policy["formal_authorization_path"] = f"schema/{suffix}.json"
        policy["formal_authority_root"] = f"private_custody/{suffix}"
        policy["formal_output_root"] = (
            f"reports/step28_synthetic_chinese_dataset/{suffix}_output"
        )
        policy["formal_private_root"] = f"private_custody/{suffix}_private"
        policy["formal_consumption_path"] = (
            f"private_custody/{suffix}/formal_500x4_build.consumed.json"
        )
        policy["formal_issuance_claim_path"] = (
            f"private_custody/{suffix}/formal_500x4_issuance.claimed.json"
        )
        policy["formal_failure_path"] = (
            f"private_custody/{suffix}/formal_500x4_build.failed.json"
        )
        policy["formal_completion_path"] = (
            f"private_custody/{suffix}/formal_500x4_build.completed.json"
        )
        paths = [
            ROOT / policy["formal_authorization_path"],
            ROOT / policy["formal_authority_root"],
            ROOT / policy["formal_output_root"],
            ROOT / policy["formal_private_root"],
        ]
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        generated = [bytes([index]) * 32 for index in range(1, 7)]
        try:
            with (
                mock.patch.object(authority, "require_clean_worktree"),
                mock.patch.object(
                    authority.builder,
                    "validate_policy",
                    return_value=policy,
                ),
                mock.patch.object(
                    authority.builder,
                    "forbidden_authority_commitments",
                    return_value={"f" * 64},
                ),
                mock.patch.object(authority, "git_head", return_value="a" * 40),
                mock.patch.object(authority, "git_tree", return_value="b" * 40),
                mock.patch.object(
                    authority.secrets,
                    "token_bytes",
                    side_effect=generated,
                ),
            ):
                result = authority.issue()
            self.assertFalse(result["key_material_returned"])
            self.assertEqual(len(result["key_commitments"]), 6)
            self.assertEqual(len(set(result["key_commitments"].values())), 6)
            serialized = json.dumps(result, sort_keys=True)
            for raw in generated:
                self.assertNotIn(raw.hex(), serialized)
            authorization = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(
                set(authorization["key_files"]),
                {"text", "identity", "style", "schedule", "uid", "time"},
            )
            formal.require_self_hash(
                authorization, label="test formal authorization"
            )
            claim_path = ROOT / policy["formal_issuance_claim_path"]
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            formal.require_self_hash(claim, label="test issuance claim")
            self.assertEqual(claim["candidate_draws_at_claim"], 0)
            self.assertFalse(claim["rerun_authorized"])
            self.assertEqual(
                authorization["issuance_claim_sha256"],
                formal.sha256_file(claim_path),
            )
            with (
                mock.patch.object(formal, "git_head", return_value="a" * 40),
                mock.patch.object(formal, "git_tree", return_value="b" * 40),
                mock.patch.object(
                    formal,
                    "forbidden_authority_commitments",
                    return_value={"f" * 64},
                ),
                mock.patch.object(
                    formal.subprocess,
                    "run",
                    return_value=SimpleNamespace(
                        stdout=(
                            "?? "
                            f"{paths[0].relative_to(ROOT).as_posix()}\n"
                        )
                    ),
                ),
            ):
                loaded, commitments = formal.load_formal_authorities(policy)
            self.assertEqual(
                commitments,
                {
                    name: hashlib.sha256(raw).hexdigest()
                    for name, raw in zip(
                        policy["private_authority"]["key_names"],
                        generated,
                        strict=True,
                    )
                },
            )
            self.assertEqual(loaded.text, generated[0])
            self.assertEqual(loaded.audit_schedule, generated[3])
            receipt = formal.consume_authorization(policy)
            self.assertTrue(receipt["sha256"])
            with self.assertRaises(FileExistsError):
                formal.consume_authorization(policy)
        finally:
            for path in reversed(paths):
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()

    def test_authority_failure_after_random_draw_is_permanently_occupied(
        self,
    ) -> None:
        policy = json.loads(json.dumps(self.policy))
        suffix = "_test_v9_4_formal_500x4_failed_issuance"
        policy["formal_authorization_path"] = f"schema/{suffix}.json"
        policy["formal_authority_root"] = f"private_custody/{suffix}"
        policy["formal_output_root"] = f"reports/{suffix}_output"
        policy["formal_private_root"] = f"private_custody/{suffix}_private"
        for name, filename in (
            ("formal_issuance_claim_path", "formal_500x4_issuance.claimed.json"),
            ("formal_consumption_path", "formal_500x4_build.consumed.json"),
            ("formal_failure_path", "formal_500x4_build.failed.json"),
            ("formal_completion_path", "formal_500x4_build.completed.json"),
        ):
            policy[name] = f"private_custody/{suffix}/{filename}"
        auth_path = ROOT / policy["formal_authorization_path"]
        authority_root = ROOT / policy["formal_authority_root"]
        for path in (auth_path, authority_root):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        original_write = authority.write_json_exclusive

        def fail_final_authorization(path: Path, value) -> None:
            if path == auth_path:
                raise OSError("injected authorization publication failure")
            original_write(path, value)

        draws = [bytes([index]) * 32 for index in range(1, 7)]
        try:
            with (
                mock.patch.object(authority, "require_clean_worktree"),
                mock.patch.object(
                    authority.builder, "validate_policy", return_value=policy,
                ),
                mock.patch.object(
                    authority.builder,
                    "forbidden_authority_commitments",
                    return_value={"f" * 64},
                ),
                mock.patch.object(authority, "git_head", return_value="a" * 40),
                mock.patch.object(authority, "git_tree", return_value="b" * 40),
                mock.patch.object(
                    authority.secrets, "token_bytes", side_effect=draws,
                ) as token_bytes,
                mock.patch.object(
                    authority, "write_json_exclusive",
                    side_effect=fail_final_authorization,
                ),
            ):
                with self.assertRaises(OSError):
                    authority.issue()
                self.assertEqual(token_bytes.call_count, 6)
                self.assertTrue(
                    (authority_root / "formal_500x4_issuance.claimed.json").is_file()
                )
                self.assertTrue(
                    (authority_root / "formal_500x4_issuance.failed.json").is_file()
                )
                self.assertFalse(any(authority_root.glob("*_key.bin")))
                self.assertFalse(auth_path.exists())
                with self.assertRaises(authority.Formal500x4AuthorityError):
                    authority.issue()
                self.assertEqual(token_bytes.call_count, 6)
        finally:
            if auth_path.exists():
                auth_path.unlink()
            if authority_root.exists():
                shutil.rmtree(authority_root)

    def test_forbidden_authority_draw_fails_without_redraw(self) -> None:
        policy = json.loads(json.dumps(self.policy))
        suffix = "_test_v9_4_formal_500x4_forbidden_draw"
        policy["formal_authorization_path"] = f"schema/{suffix}.json"
        policy["formal_authority_root"] = f"private_custody/{suffix}"
        policy["formal_output_root"] = f"reports/{suffix}_output"
        policy["formal_private_root"] = f"private_custody/{suffix}_private"
        for name, filename in (
            ("formal_issuance_claim_path", "formal_500x4_issuance.claimed.json"),
            ("formal_consumption_path", "formal_500x4_build.consumed.json"),
            ("formal_failure_path", "formal_500x4_build.failed.json"),
            ("formal_completion_path", "formal_500x4_build.completed.json"),
        ):
            policy[name] = f"private_custody/{suffix}/{filename}"
        auth_path = ROOT / policy["formal_authorization_path"]
        authority_root = ROOT / policy["formal_authority_root"]
        forbidden = b"f" * 32
        forbidden_commitment = hashlib.sha256(forbidden).hexdigest()
        for path in (auth_path, authority_root):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        try:
            with (
                mock.patch.object(authority, "require_clean_worktree"),
                mock.patch.object(
                    authority.builder, "validate_policy", return_value=policy,
                ),
                mock.patch.object(
                    authority.builder,
                    "forbidden_authority_commitments",
                    return_value={forbidden_commitment},
                ),
                mock.patch.object(authority, "git_head", return_value="a" * 40),
                mock.patch.object(authority, "git_tree", return_value="b" * 40),
                mock.patch.object(
                    authority.secrets,
                    "token_bytes",
                    side_effect=[forbidden, b"g" * 32],
                ) as token_bytes,
            ):
                with self.assertRaisesRegex(
                    authority.Formal500x4AuthorityError,
                    "duplicate or forbidden",
                ):
                    authority.issue()
                self.assertEqual(token_bytes.call_count, 1)
                self.assertFalse(auth_path.exists())
                self.assertFalse(any(authority_root.glob("*_key.bin")))
                self.assertTrue(
                    (authority_root / "formal_500x4_issuance.claimed.json").is_file()
                )
                self.assertTrue(
                    (authority_root / "formal_500x4_issuance.failed.json").is_file()
                )
                with self.assertRaises(authority.Formal500x4AuthorityError):
                    authority.issue()
                self.assertEqual(token_bytes.call_count, 1)
        finally:
            if auth_path.exists():
                auth_path.unlink()
            if authority_root.exists():
                shutil.rmtree(authority_root)

    def test_private_root_publishes_first_and_rolls_back_if_public_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            temporary = parent / "public.building"
            root = parent / "public"
            private_temporary = parent / "private.building"
            private = parent / "private"
            temporary.mkdir()
            private_temporary.mkdir()
            original_rename = Path.rename
            rename_calls: list[tuple[Path, Path]] = []

            def fail_public(source: Path, target: Path):
                rename_calls.append((source, target))
                if source == temporary and target == root:
                    raise OSError("injected public publication failure")
                return original_rename(source, target)

            with mock.patch.object(Path, "rename", autospec=True, side_effect=fail_public):
                with self.assertRaises(OSError):
                    formal.publish_dual_roots(
                        temporary=temporary,
                        root=root,
                        private_temporary=private_temporary,
                        private=private,
                    )
            self.assertTrue(temporary.is_dir())
            self.assertTrue(private_temporary.is_dir())
            self.assertFalse(root.exists())
            self.assertFalse(private.exists())
            self.assertEqual(
                rename_calls,
                [
                    (private_temporary, private),
                    (temporary, root),
                    (private, private_temporary),
                ],
            )

    def test_post_consumption_failure_receipt_is_terminal_and_self_hashed(
        self,
    ) -> None:
        policy = json.loads(json.dumps(self.policy))
        suffix = "_test_v9_4_formal_500x4_failure_receipt"
        auth_path = ROOT / "schema" / f"{suffix}.json"
        authority_root = ROOT / "private_custody" / suffix
        failure_path = authority_root / "formal_500x4_build.failed.json"
        policy["formal_authorization_path"] = auth_path.relative_to(ROOT).as_posix()
        policy["formal_failure_path"] = failure_path.relative_to(ROOT).as_posix()
        for path in (auth_path, authority_root):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        try:
            authority_root.mkdir(parents=True)
            auth_path.write_text("{}\n", encoding="utf-8", newline="\n")
            with (
                mock.patch.object(formal, "git_head", return_value="a" * 40),
                mock.patch.object(formal, "git_tree", return_value="b" * 40),
            ):
                formal.write_post_consumption_failure(
                    policy,
                    consumption={
                        "path": "private/consumed.json",
                        "sha256": "c" * 64,
                        "canonical_self_hash": "d" * 64,
                    },
                    stage="injected_stage",
                    exc=RuntimeError("injected private detail"),
                    worlds_completed=17,
                    path_presence_before_cleanup={
                        "public_temporary": True,
                        "private_temporary": True,
                        "public_final": False,
                        "private_final": False,
                    },
                )
            receipt = json.loads(failure_path.read_text(encoding="utf-8"))
            formal.require_self_hash(receipt, label="test formal failure")
            self.assertEqual(
                receipt["claim_boundary"],
                "MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
            )
            self.assertEqual(receipt["worlds_completed"], 17)
            self.assertFalse(receipt["rerun_authorized"])
            self.assertNotIn("injected private detail", json.dumps(receipt))
            with self.assertRaises(FileExistsError):
                formal.write_post_consumption_failure(
                    policy,
                    consumption=receipt["consumption"],
                    stage="second_attempt",
                    exc=RuntimeError("second"),
                    worlds_completed=0,
                    path_presence_before_cleanup={},
                )
        finally:
            if auth_path.exists():
                auth_path.unlink()
            if authority_root.exists():
                shutil.rmtree(authority_root)

    def test_completion_receipt_accepts_canonical_json_integer_key_roundtrip(
        self,
    ) -> None:
        policy = json.loads(json.dumps(self.policy))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            output = parent / "public"
            private = parent / "private"
            completion = parent / "authority" / "completed.json"
            output.mkdir()
            private.mkdir()
            policy["formal_output_root"] = output.relative_to(ROOT).as_posix()
            policy["formal_private_root"] = private.relative_to(ROOT).as_posix()
            policy["formal_completion_path"] = completion.relative_to(ROOT).as_posix()
            manifest = {
                "status": "BUILT_NOT_TRAINING_QUALIFIED",
                "schedule_balance_audit": {
                    "train": {"seller_pair_histogram": {26: 206, 27: 172}},
                },
                "private_file_commitments": [],
            }
            manifest["canonical_self_hash"] = formal.canonical_sha256(manifest)
            formal.engine.write_json(output / "root_manifest.json", manifest)
            receipt = formal.write_completion_receipt(
                policy,
                consumption={
                    "path": "private/consumed.json",
                    "sha256": "c" * 64,
                    "canonical_self_hash": "d" * 64,
                },
                manifest=manifest,
            )
            self.assertTrue(receipt["sha256"])
            value = json.loads(completion.read_text(encoding="utf-8"))
            formal.require_self_hash(value, label="test completion receipt")
            self.assertFalse(value["training_qualified"])

    def test_four_world_smoke_build_closes_without_persistent_fixture(self) -> None:
        output = (
            ROOT / "reports" / "step28_synthetic_chinese_dataset"
            / "_test_v9_4_formal_500x4_smoke"
        )
        private = output.parent / f".{output.name}.private"
        temporary = output.parent / f".{output.name}.building"
        private_temporary = private.parent / f".{private.name}.building"
        for path in (output, private, temporary, private_temporary):
            if path.exists():
                shutil.rmtree(path)
        try:
            result = formal.build_dataset(formal=False, output_root=output)
            self.assertEqual(
                result["world_counts"],
                {split: 1 for split in formal.SPLITS},
            )
            self.assertEqual(result["world_count"], 4)
            self.assertEqual(result["seller_count"], 112)
            self.assertEqual(result["pair_count"], 1512)
            self.assertEqual(result["positive_pair_count"], 80)
            self.assertEqual(result["negative_pair_count"], 1432)
            self.assertEqual(
                result["audit_truth_read_counts"],
                {"audit_a": 0, "audit_b": 0},
            )
            self.assertFalse(result["training_qualified"])
            self.assertFalse(result["m0_m1_m2_m3_training_authorized"])
            self.assertTrue((output / "root_manifest.json").is_file())
            self.assertTrue(
                (output / "document_collision_registry.json").is_file()
            )
            self.assertTrue(
                (output / "document_collision_hashes.jsonl").is_file()
            )
            public_registry = json.loads(
                (output / "document_collision_registry.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("identity_value_hashes", public_registry)
            self.assertNotIn("item_document_hashes", public_registry)
            self.assertNotIn("seller_document_hashes", public_registry)
            self.assertEqual(
                public_registry["hash_list"]["row_count"],
                public_registry["counts"]["item_documents"]
                + public_registry["counts"]["seller_documents"],
            )
            self.assertEqual(
                public_registry["counts"]["canonical_pair_uids"], 1512,
            )
            self.assertEqual(
                public_registry["formal_uid_values_checked"][
                    "canonical_pair_uid"
                ],
                1512,
            )
            self.assertEqual(
                public_registry[
                    "historical_and_method_root_intersection_counts"
                ]["canonical_pair_uids"],
                0,
            )
            self.assertTrue(
                (private / "identity_collision_registry.json").is_file()
            )
        finally:
            for path in (output, private, temporary, private_temporary):
                if path.exists():
                    shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()
