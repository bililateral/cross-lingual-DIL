from __future__ import annotations

import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import step28_v13_build_training_ready_dataset as builder  # noqa: E402
import step28_v13_common as common  # noqa: E402
import step28_v13_finalize_training_ready_dataset as finalizer  # noqa: E402
import step28_v13_initialize_training_ready_keys as key_init  # noqa: E402
import step28_v13_integrity_receipts as integrity_receipts  # noqa: E402
import step28_v13_structure as structure  # noqa: E402
import step28_v13_validate_model_inputs as mount_validator  # noqa: E402


class TrainingReadyBuilderContracts(unittest.TestCase):
    def test_builder_recursive_source_closure_is_exact(self) -> None:
        overlay = common.load_json(builder.DEFAULT_OVERLAY)
        closure = overlay["dataset_builder"]["implementation_closure"]
        members = list(
            integrity_receipts._source_closure_members(
                builder.BUILDER_SOURCE_CLOSURE_ROLE
            )
        )
        self.assertEqual(closure["members"], members)
        self.assertEqual(closure["member_count"], len(members))
        self.assertEqual(
            closure["canonical_sha256"],
            integrity_receipts._source_closure_sha256(
                builder.BUILDER_SOURCE_CLOSURE_ROLE
            ),
        )
        self.assertIn(
            "scripts/step28_v13_independent_private_dgp_replay.py",
            members,
        )
        self.assertIn("scripts/step3_build_seller_profiles.py", members)
        corrupted = copy.deepcopy(overlay)
        corrupted["dataset_builder"]["implementation_closure"][
            "members"
        ] = members[:-1]
        with self.assertRaises(common.ContractError):
            builder._validate_dataset_builder_closure(corrupted)

    def test_implementation_contract_ignores_only_release_phase_fields(
        self,
    ) -> None:
        overlay = common.load_json(builder.DEFAULT_OVERLAY)
        baseline = builder.implementation_contract_sha256(overlay)
        post_preflight = copy.deepcopy(overlay)
        post_preflight["status"] = "READY_FOR_KEY_CEREMONY"
        post_preflight["exact_implementation_preflights"] = {
            split: {"path": f"unused/{split}.json", "sha256": "0" * 64}
            for split in builder.SPLITS
        }
        post_preflight["release_contract"]["sha256"] = "1" * 64
        post_preflight["private_structure_key_custody"]["commitments"] = {
            split: str(index) * 64
            for index, split in enumerate(builder.SPLITS, start=2)
        }
        post_preflight["private_structure_key_custody"][
            "ceremony_receipt"
        ] = {"path": "unused/receipt.json", "sha256": "6" * 64}
        self.assertEqual(
            builder.implementation_contract_sha256(post_preflight),
            baseline,
        )
        changed_science = copy.deepcopy(overlay)
        changed_science["world_counts"]["train"] += 1
        self.assertNotEqual(
            builder.implementation_contract_sha256(changed_science),
            baseline,
        )
        changed_machine_contract = copy.deepcopy(overlay)
        changed_machine_contract["scientific_contract"]["sha256"] = (
            "7" * 64
        )
        self.assertNotEqual(
            builder.implementation_contract_sha256(
                changed_machine_contract
            ),
            baseline,
        )
        changed_release_tool = copy.deepcopy(overlay)
        changed_release_tool["release_tools"]["finalizer"]["sha256"] = (
            "8" * 64
        )
        self.assertNotEqual(
            builder.implementation_contract_sha256(
                changed_release_tool
            ),
            baseline,
        )

    @classmethod
    def _mini_result(cls, split: str):
        overlay = builder.load_overlay(
            builder.DEFAULT_OVERLAY,
            require_generation_frozen=False,
        )
        mini = copy.deepcopy(overlay)
        mini["world_counts"][split] = 5
        mini["shortcut_gate"]["bootstrap_replicates"] = 19
        mini["shortcut_gate"]["maximum_symmetric_auc"] = 1.0
        mini["shortcut_gate"][
            "maximum_world_bootstrap_95_upper"
        ] = 1.0
        base = builder._load_pinned_base(overlay)
        design_key = builder.DESIGN_ONLY_STRUCTURE_KEY_HEX
        policy = builder._execution_policy(
            base,
            mini,
            structure_key_hex=design_key,
        )
        result = builder.build_split_in_memory(
            policy,
            mini,
            split=split,
            structure_key_hex=design_key,
            progress_every=5,
            mini_exercise_allow_incomplete_train_support=(
                split == "train"
            ),
        )
        return policy, mini, result

    @classmethod
    def setUpClass(cls) -> None:
        cls.train_policy, cls.train_overlay, cls.train_result = (
            cls._mini_result("train")
        )
        cls.audit_policy, cls.audit_overlay, cls.audit_result = (
            cls._mini_result("audit_a")
        )
        cls.audit_b_policy, cls.audit_b_overlay, cls.audit_b_result = (
            cls._mini_result("audit_b")
        )

    def test_aggregate_lineage_and_formula_close(self) -> None:
        audit = self.train_result["aggregate_audit"]
        self.assertTrue(audit["all_keysets_and_foreign_keys_exact"])
        self.assertTrue(audit["identity_values_replayed_exactly"])
        self.assertEqual(audit["world_uid_count"], 5)
        self.assertEqual(audit["seller_uid_count"], 140)
        self.assertEqual(audit["pair_uid_count"], 1890)
        self.assertEqual(
            self.train_result["formula_audit"]["positive_count"],
            5 * 16,
        )
        self.assertIs(
            self.train_result["identity33_audit"][
                "no_all_zero_columns_required"
            ],
            False,
        )
        self.assertIs(
            self.train_result["identity33_audit"][
                "no_all_zero_columns_gate_pass"
            ],
            True,
        )
        self.assertEqual(
            self.train_result["identity33_audit"]["all_zero_columns"],
            ["verified_x_risky", "verified_x_support"],
        )
        self.assertIs(
            self.audit_result["identity33_audit"][
                "no_all_zero_columns_required"
            ],
            False,
        )

    def test_train_support_gate_is_default_and_mini_bypass_is_scoped(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            common.ContractError,
            "^Identity33 has all-zero columns:",
        ):
            builder.build_split_in_memory(
                self.train_policy,
                self.train_overlay,
                split="train",
                structure_key_hex=builder.DESIGN_ONLY_STRUCTURE_KEY_HEX,
                progress_every=5,
            )
        invalid = copy.deepcopy(self.train_overlay)
        invalid["generation_enabled"] = True
        with self.assertRaisesRegex(
            common.ContractError,
            "^Incomplete train identity support is allowed only",
        ):
            builder.build_split_in_memory(
                self.train_policy,
                invalid,
                split="train",
                structure_key_hex=builder.DESIGN_ONLY_STRUCTURE_KEY_HEX,
                progress_every=5,
                mini_exercise_allow_incomplete_train_support=True,
            )

    def test_single_pass_solver_preserves_reference_payload_bytes(self) -> None:
        expected = {
            "train": {
                "payload": (
                    "38a55e903d9de32590888672973da59d9428ad221e613792d6"
                    "fc91cb96bc0465"
                ),
                "world_generation_audit": (
                    "2f9a4887e278eb7f2d24c98d798bd2e30e80367e66426c43"
                    "be4dea3bde66204c"
                ),
            },
            "audit_a": {
                "payload": (
                    "9aaf4f15c2f8f0c48edda02daadbd2fba0ad1fa0c018bfb1e"
                    "13a90d966c50e30"
                ),
                "world_generation_audit": (
                    "3b36c22e661f58a10f2981c85bac385d415e691c54c362a26"
                    "06a5b7472b75ddb"
                ),
            },
        }
        for split, result in (
            ("train", self.train_result),
            ("audit_a", self.audit_result),
        ):
            self.assertEqual(
                common.canonical_sha256(result["payload"]),
                expected[split]["payload"],
            )
            self.assertEqual(
                common.canonical_sha256(
                    result["payload"]["world_generation_audit"]
                ),
                expected[split]["world_generation_audit"],
            )

    def test_audit_b_direct_hub_capacity_fallback_closes(self) -> None:
        overlay = builder.load_overlay(
            builder.DEFAULT_OVERLAY,
            require_generation_frozen=False,
        )
        base = builder._load_pinned_base(overlay)
        design_key = builder.DESIGN_ONLY_STRUCTURE_KEY_HEX
        policy = builder._execution_policy(
            base,
            overlay,
            structure_key_hex=design_key,
        )
        self.assertEqual(
            builder._expected_graph_counts(policy, split="audit_a"),
            {"identity_assets": 84, "negative_flags": 42},
        )
        self.assertEqual(
            builder._expected_graph_counts(policy, split="audit_b"),
            {"identity_assets": 89, "negative_flags": 100},
        )
        template, fixture, style_profile = (
            builder.legacy_generator._load_release_inputs(
                policy,
                mode=builder.MODE,
            )
        )
        records = [
            row
            for row in structure.build_mode_world_pool(
                policy,
                mode=builder.MODE,
            )
            if row["split"] == "audit_b"
        ]
        record = records[1]
        self.assertEqual(record["split_ordinal"], 456)
        self.assertEqual(
            record["world_uid"],
            "w_00965230433284f17ac83082f8f9d50f3b58d6c33e25780143"
            "a34051362ed3a9",
        )
        result = builder._one_world(
            policy=policy,
            split="audit_b",
            record=record,
            structure_key_hex=design_key,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
        )
        solver = result["world"]["private"]["solver_audit"]
        self.assertEqual(
            solver["selected_membership_complete_assignment_ordinal"],
            0,
        )
        self.assertEqual(solver["membership_solver_node_count"], 14)
        self.assertEqual(solver["type_solver_node_count"], 33)
        hubs = [
            row
            for row in result["world"]["private"]["identity_assets"]
            if row["descriptor_kind"] == "high_frequency_direct_hub"
        ]
        self.assertEqual(len(hubs), 1)
        self.assertEqual(hubs[0]["identity_type"], "crypto_wallet")
        self.assertEqual(len(hubs[0]["sellers"]), 8)
        world_audit = result["world_audit"]
        self.assertTrue(
            world_audit["independent_typed_dgp_replay_audit"][
                "full_typed_projection_exact"
            ]
        )
        self.assertTrue(
            world_audit["producer_regeneration_audit"][
                "producer_regeneration_match_pass"
            ]
        )

    def test_audit_b_graph_specific_aggregate_counts_close(self) -> None:
        payload = self.audit_b_result["payload"]
        self.assertEqual(len(payload["identity_assets"]), 5 * 89)
        self.assertEqual(len(payload["negative_flags"]), 5 * 100)
        self.assertTrue(
            self.audit_b_result["aggregate_audit"][
                "all_keysets_and_foreign_keys_exact"
            ]
        )
        self.assertEqual(len(payload["retrieval_queries"]), 5 * 4)
        self.assertEqual(len(payload["retrieval_relations"]), 5 * 4 * 27)

    def test_train_writer_builds_five_m1_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            files = builder._write_payload(
                self.train_policy,
                self.train_overlay,
                split="train",
                stage=stage,
                result=self.train_result,
            )
            receipt_path = (
                stage / "audit" / "m1_derangement_receipts.json"
            )
            receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(len(receipts), 5)
            self.assertEqual(
                len({row["rewire_seed_id"] for row in receipts}), 5
            )
            self.assertTrue(
                all(
                    row[
                        "joint_vector_multiset_exact_by_world_and_universe"
                    ]
                    and row["endpoint_disjoint_bijection_exact"]
                    and not row["labels_or_controller_inputs_read"]
                    and row["matrix_row_count"] == 5 * 378
                    and row["persisted_matrix_reread_exact"]
                    and row["persisted_mapping_reread_exact"]
                    and row["persisted_whole_vector_replay_exact"]
                    and row[
                        "persisted_endpoint_disjoint_bijection_exact"
                    ]
                    for row in receipts
                )
            )
            self.assertEqual(
                {row["matrix_path"] for row in receipts},
                {
                    f"m1/r{index:02d}/identity33.csv"
                    for index in range(1, 6)
                },
            )
            mounted = {
                row["path"]
                for row in files
                if row["model_mount_allowed"]
            }
            self.assertEqual(
                mounted,
                {
                    "observed/complete_model_pair_endpoints.csv",
                    "observed/redacted_items.jsonl",
                    "observed/seller_profiles.jsonl",
                },
            )
            files_by_path = {row["path"]: row for row in files}
            manifest = {
                "run_id": self.train_overlay["run_id"],
                "split": "train",
                "files": files,
            }
            manifest["canonical_self_hash"] = common.canonical_sha256(
                manifest
            )
            common.write_json(stage / "split_manifest.json", manifest)
            receipt = mount_validator.validate_mount(
                split_directory=stage,
                role="m0",
                mounted_relative_paths=[
                    "observed/complete_model_pair_endpoints.csv",
                    "observed/redacted_items.jsonl",
                    "observed/seller_profiles.jsonl",
                ],
                replicate=None,
            )
            self.assertEqual(
                receipt["status"],
                "PASS_EXACT_MODEL_INPUT_ALLOWLIST",
            )
            self.assertIn(
                "observed/redacted_items.jsonl",
                files_by_path,
            )
            with self.assertRaises(common.ContractError):
                mount_validator.validate_mount(
                    split_directory=stage,
                    role="m0",
                    mounted_relative_paths=[
                        "observed/complete_model_pair_endpoints.csv",
                        "observed/redacted_items.jsonl",
                        "observed/seller_profiles.jsonl",
                        "observed/candidate_pairs.csv",
                    ],
                    replicate=None,
                )

            first = receipts[0]
            matrix_path = stage / first["matrix_path"]
            with matrix_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                matrix_rows = [dict(row) for row in reader]
            feature = self.train_policy["history_features"][
                "feature_names"
            ][0]
            matrix_rows[0][feature] = str(
                float(matrix_rows[0][feature]) + 1.0
            )
            with matrix_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(matrix_rows)
            with self.assertRaises(common.ContractError):
                builder._validate_persisted_m1(
                    self.train_policy,
                    seed_id=first["rewire_seed_id"],
                    matrix_path=matrix_path,
                    mapping_path=stage / first["mapping_path"],
                    m2_rows=self.train_result["payload"][
                        "identity33_all_pairs"
                    ],
                    candidate_rows=self.train_result["payload"][
                        "candidate_pairs"
                    ],
                    endpoint_rows=self.train_result["payload"][
                        "complete_model_pair_endpoints"
                    ],
                )

    def test_audit_writer_withholds_metrics_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            builder._write_payload(
                self.audit_policy,
                self.audit_overlay,
                split="audit_a",
                stage=stage,
                result=self.audit_result,
            )
            self.assertTrue(
                (
                    stage
                    / "audit"
                    / "metadata_shortcut_gate.json"
                ).is_file()
            )
            self.assertFalse(
                (
                    stage
                    / "audit"
                    / "metadata_shortcut_audit.json"
                ).exists()
            )
            self.assertTrue(
                (
                    stage
                    / "sealed_supervision"
                    / "classification_labels.csv"
                ).is_file()
            )
            self.assertTrue(
                (
                    stage
                    / "sealed_supervision"
                    / "retrieval_qrels.csv"
                ).is_file()
            )
            self.assertEqual(
                len(self.audit_result["payload"]["retrieval_queries"]),
                5 * 4,
            )
            self.assertEqual(
                len(self.audit_result["payload"]["retrieval_relations"]),
                5 * 4 * 27,
            )

    def test_invalid_progress_interval_fails_before_generation(self) -> None:
        with self.assertRaises(common.ContractError):
            builder.build_split_in_memory(
                self.train_policy,
                self.train_overlay,
                split="train",
                structure_key_hex=builder.DESIGN_ONLY_STRUCTURE_KEY_HEX,
                progress_every=0,
            )

    def test_key_documents_are_unique_and_do_not_leak_via_receipt(self) -> None:
        draws = iter(bytes([value]) * 32 for value in range(1, 5))
        documents, commitments = key_init._generate_documents(
            run_id="test_run",
            forbidden_commitments=set(),
            random_bytes=lambda _count: next(draws),
        )
        self.assertEqual(set(documents), set(builder.SPLITS))
        self.assertEqual(set(commitments), set(builder.SPLITS))
        self.assertEqual(len(set(commitments.values())), 4)
        self.assertTrue(
            all(
                documents[split]["sha256_commitment"]
                == commitments[split]
                and len(documents[split]["key_hex"]) == 64
                for split in builder.SPLITS
            )
        )

    @staticmethod
    def _write_private_key_bundle(
        directory: Path,
        *,
        overlay: dict,
        overlay_path: Path,
        raw_by_split: dict[str, bytes],
        extra_receipt_field: bool = False,
    ) -> None:
        commitments = {}
        for split in builder.SPLITS:
            raw = raw_by_split[split]
            commitment = common.sha256_bytes(raw)
            commitments[split] = commitment
            filename = overlay["private_structure_key_custody"][
                "key_filename_pattern"
            ].format(split=split)
            common.write_json(
                directory / filename,
                {
                    "version": key_init.KEY_DOCUMENT_VERSION,
                    "run_id": overlay["run_id"],
                    "split": split,
                    "key_hex": raw.hex(),
                    "sha256_commitment": commitment,
                },
            )
        initializer_path = (
            ROOT
            / "scripts"
            / "step28_v13_initialize_training_ready_keys.py"
        )
        receipt = {
            "version": key_init.RECEIPT_VERSION,
            "status": "PASS_SPLIT_PRIVATE_KEY_CEREMONY",
            "run_id": overlay["run_id"],
            "pre_ceremony_overlay_path": overlay_path.relative_to(
                ROOT
            ).as_posix(),
            "pre_ceremony_overlay_sha256": common.sha256_file(
                overlay_path
            ),
            "initializer_path": (
                "scripts/step28_v13_initialize_training_ready_keys.py"
            ),
            "initializer_sha256": common.sha256_file(
                initializer_path
            ),
            "commitments": commitments,
            "commitments_unique": (
                len(set(commitments.values())) == len(builder.SPLITS)
            ),
            "forbidden_commitment_intersection_count": 0,
            "one_split_key_per_file": True,
            "raw_structure_keys_serialized": False,
            "os_custody_attested": False,
        }
        if extra_receipt_field:
            receipt["unexpected"] = True
        receipt["canonical_self_hash"] = common.canonical_sha256(
            receipt
        )
        common.write_json(
            directory / key_init.PUBLIC_RECEIPT_FILENAME,
            receipt,
        )

    def test_private_key_recovery_rejects_forbidden_commitment(self) -> None:
        overlay = common.load_json(builder.DEFAULT_OVERLAY)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            bundle = Path(directory)
            raw_by_split = {
                split: bytes([index]) * 32
                for index, split in enumerate(
                    builder.SPLITS,
                    start=1,
                )
            }
            raw_by_split["train"] = bytes.fromhex(
                builder.DESIGN_ONLY_STRUCTURE_KEY_HEX
            )
            self._write_private_key_bundle(
                bundle,
                overlay=overlay,
                overlay_path=builder.DEFAULT_OVERLAY,
                raw_by_split=raw_by_split,
            )
            with self.assertRaisesRegex(
                common.ContractError,
                "commitment is forbidden",
            ):
                key_init._validate_private_bundle(
                    directory=bundle,
                    overlay=overlay,
                    overlay_path=builder.DEFAULT_OVERLAY,
                )

    def test_private_key_recovery_requires_exact_bundle_and_receipt(self) -> None:
        overlay = common.load_json(builder.DEFAULT_OVERLAY)
        raw_by_split = {
            split: bytes([index + 20]) * 32
            for index, split in enumerate(builder.SPLITS)
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            bundle = Path(directory)
            self._write_private_key_bundle(
                bundle,
                overlay=overlay,
                overlay_path=builder.DEFAULT_OVERLAY,
                raw_by_split=raw_by_split,
            )
            (bundle / "unexpected.txt").write_text(
                "forbidden",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                common.ContractError,
                "member set drift",
            ):
                key_init._validate_private_bundle(
                    directory=bundle,
                    overlay=overlay,
                    overlay_path=builder.DEFAULT_OVERLAY,
                )
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            bundle = Path(directory)
            self._write_private_key_bundle(
                bundle,
                overlay=overlay,
                overlay_path=builder.DEFAULT_OVERLAY,
                raw_by_split=raw_by_split,
                extra_receipt_field=True,
            )
            with self.assertRaisesRegex(
                common.ContractError,
                "receipt copy drift",
            ):
                key_init._validate_private_bundle(
                    directory=bundle,
                    overlay=overlay,
                    overlay_path=builder.DEFAULT_OVERLAY,
                )

    def test_private_key_recovery_rejects_reparse_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "plain.json"
            path.write_text("{}", encoding="utf-8")
            with mock.patch.object(
                key_init,
                "_has_reparse_attribute",
                return_value=True,
            ):
                with self.assertRaisesRegex(
                    common.ContractError,
                    "non-reparse regular file",
                ):
                    key_init._require_plain_file(
                        path,
                        label="test key",
                    )

    def test_split_writer_rejects_reparse_members_before_publish(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            member = root / "member.json"
            member.write_text("{}", encoding="utf-8")
            with mock.patch.object(
                builder,
                "_has_reparse_attribute",
                side_effect=lambda path: Path(path).name == member.name,
            ):
                with self.assertRaisesRegex(
                    common.ContractError,
                    "reparse file",
                ):
                    builder._file_records(root)


class TrainingReadyFinalizerContracts(unittest.TestCase):
    def test_jsonl_reader_rejects_duplicate_object_keys(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "duplicate.jsonl"
            path.write_text(
                '{"item_uid":"a","item_uid":"b"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                common.ContractError,
                "Duplicate JSON object key",
            ):
                finalizer._read_jsonl(path)

    def test_finalizer_rejects_reparse_release_members(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "member.json"
            path.write_text("{}", encoding="utf-8")
            with mock.patch.object(
                finalizer,
                "_has_reparse_attribute",
                return_value=True,
            ):
                with self.assertRaisesRegex(
                    common.ContractError,
                    "non-reparse regular file",
                ):
                    finalizer._require_plain_file(
                        path,
                        label="test release member",
                    )

    def test_split_manifest_must_explicitly_bind_scientific_contract(
        self,
    ) -> None:
        overlay = common.load_json(builder.DEFAULT_OVERLAY)
        manifest = {
            "version": builder.MANIFEST_VERSION,
            "status": "PASS_SPLIT_DATASET_READY",
            "run_id": overlay["run_id"],
            "split": "train",
            "claim_level": overlay["target_release_claim_level"],
            "overlay_canonical_sha256": common.canonical_sha256(overlay),
            "implementation_contract": overlay["implementation_contract"],
            "implementation_contract_sha256": (
                builder.implementation_contract_sha256(overlay)
            ),
            "scientific_contract": {
                "path": overlay["scientific_contract"]["path"],
                "sha256": "0" * 64,
            },
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            common.write_json(root / "split_manifest.json", manifest)
            with mock.patch.object(
                finalizer.builder,
                "_validate_split_tree",
            ):
                with self.assertRaisesRegex(
                    common.ContractError,
                    "Split manifest semantic drift",
                ):
                    finalizer._validate_one_split(
                        policy={},
                        overlay=overlay,
                        split="train",
                        directory=root,
                    )


if __name__ == "__main__":
    unittest.main()
