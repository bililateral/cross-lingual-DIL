#!/usr/bin/env python3
"""Contracts for the V9.2 immutable policy and one-shot audit call layer."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common
import step28_v13_v1_13_quality_audit_runner_v9_2 as runner
import step28_v13_v1_13_quality_policy_v9_2 as policy_module
import step28_v13_v1_13_quality_probe_validator_v9_2 as validator
import step28_v13_v1_13_quality_structure_aggregator_v9_2 as aggregator
import step28_v13_v1_13_quality_truth_capability_v9_2 as truth_v9_2
import step28_v13_v1_13_scientific_dataset_builder_v9_2 as builder


class QualityAuditRunnerV92Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = policy_module.load_policy()

    def authorization(self) -> dict:
        value = {
            "version": runner.AUTHORIZATION_VERSION,
            "status": "ONE_SHOT_V9_2_METHOD_ROOT_QUALITY_AUDIT_AUTHORIZED",
            "single_use": True,
            "receipt_generation_by_repository_code_forbidden": True,
            "quality_policy": {
                **runner._sha256_pin(policy_module.DEFAULT_POLICY_PATH),
                "canonical_self_hash": self.policy["canonical_self_hash"],
            },
            "design_root_manifest": {
                "path": "reports/step28_v13_v9_2_fixture/root_manifest.json",
                "size_bytes": 123,
                "sha256": "1" * 64,
                "canonical_self_hash": "2" * 64,
            },
            "complete_evidence_output_path": (
                "reports/step28_v13_v9_2_fixture_quality/"
                "complete_quality_evidence.json"
            ),
            "capabilities": dict(runner.CAPABILITIES),
            "private_key_material": {
                "id_key_hex": "3" * 64,
                "document_variation_key_hex": "4" * 64,
            },
            "git_commit": "5" * 40,
            "git_tree": "6" * 40,
            "review_response_sha256": "7" * 64,
            "review_final_line": runner.REQUIRED_REVIEW_FINAL_LINE,
        }
        value["canonical_self_hash"] = runner._canonical_self_hash(value)
        return value

    def random_authority(self) -> dict:
        def key(label: str) -> str:
            return hashlib.sha256(f"v9.2-fixture::{label}".encode("utf-8")).hexdigest()

        keys = {
            name: key(name)
            for name in builder.KEY_FIELDS - {"rewire_key_hexes"}
        }
        keys["rewire_key_hexes"] = [key(f"rewire::{index}") for index in range(5)]
        values = builder._key_values(keys)
        value = {
            "version": builder.RANDOM_AUTHORITY_VERSION,
            "status": "FROZEN_FRESH_V9_2_RANDOM_AUTHORITY",
            "created_by_repository_code": False,
            "single_use_for_method_qualification_root": True,
            "keys": keys,
            "authority_bundle_sha256": common.canonical_sha256(sorted(values)),
            "git_commit": "8" * 40,
            "git_tree": "9" * 40,
            "review_response_sha256": "a" * 64,
            "review_final_line": builder.RANDOM_REVIEW_FINAL_LINE,
        }
        value["canonical_self_hash"] = builder._self_hash(value)
        return value

    def root_manifest(self, authorization: dict) -> dict:
        receipt = {
            "path_sha256": "a" * 64,
            "size_bytes": 100,
            "sha256": "b" * 64,
            "canonical_self_hash": "c" * 64,
        }
        key_material = authorization["private_key_material"]
        value = {
            "version": builder.VERSION,
            "status": "PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED",
            "execution_mode": builder.EXECUTION_MODE,
            "scientific_use_forbidden": True,
            "formal_seed_created": False,
            "formal_rows_created": 0,
            "training_started": False,
            "quality_policy_canonical_self_hash": self.policy["canonical_self_hash"],
            "quality_policy_file": runner._sha256_pin(policy_module.DEFAULT_POLICY_PATH),
            "builder_source_file": runner._sha256_pin(Path(builder.__file__)),
            "design_build_authorization": {
                "status": "CONSUMED_ONE_SHOT_BUILD_AUTHORIZATION",
                "receipt": dict(receipt),
                "review_response_sha256": "d" * 64,
                "git_commit": "e" * 40,
                "git_tree": "f" * 40,
            },
            "random_authority": {
                "status": "CONSUMED_FRESH_V9_2_RANDOM_AUTHORITY",
                "receipt": dict(receipt),
                "authority_bundle_sha256": "1" * 64,
            },
            "quality_required_key_commitments": {
                name.removesuffix("_hex") + "_sha256": hashlib.sha256(
                    bytes.fromhex(value)
                ).hexdigest()
                for name, value in key_material.items()
            },
            "model_input_file_count": 8,
            "split_order": list(runner.SPLITS),
            "world_count": 1004,
            "seller_count": 1004 * 28,
            "pair_count": 1004 * 378,
            "positive_pair_count": 1004 * 20,
            "negative_pair_count": 1004 * 358,
            "uid_registries": {},
            "item_document_registry_count": 0,
            "item_document_registry_sha256": "2" * 64,
            "seller_document_registry_count": 0,
            "seller_document_registry_sha256": "3" * 64,
            "item_code_registry_count": 0,
            "item_code_registry_sha256": "4" * 64,
            "identity_value_registry_count": 0,
            "identity_value_registry_sha256": "5" * 64,
            "historical_exclusion_counts": {},
            "split_manifest_self_hashes": {split: "6" * 64 for split in runner.SPLITS},
            "canonical_self_hash": "7" * 64,
        }
        self.assertEqual(set(value), runner.ROOT_MANIFEST_FIELDS)
        return value

    def test_base_policy_cannot_contain_runtime_root_or_execution_authority(self) -> None:
        self.assertEqual(
            self.policy["authorization"], policy_module.EXPECTED_AUTHORIZATION
        )
        self.assertFalse(self.policy["authorization"]["quality_audit_run"])
        self.assertFalse(self.policy["authorization"]["metric_generation"])
        self.assertNotIn("pins", self.policy)
        self.assertNotIn("design_root_manifest", self.policy)

    def test_separate_authorization_is_narrow_and_binds_future_root(self) -> None:
        value = self.authorization()
        normalized = runner.validate_run_authorization(
            value, policy=self.policy, verify_bound_files=False
        )
        self.assertEqual(normalized, value)
        self.assertEqual(normalized["capabilities"], runner.CAPABILITIES)
        self.assertNotIn("audit_a_truth", normalized["private_key_material"])
        self.assertNotIn("audit_b_truth", normalized["private_key_material"])

    def test_quality_runner_closes_the_saved_build_and_random_lineage(self) -> None:
        authorization = self.authorization()
        root_manifest = self.root_manifest(authorization)
        with tempfile.TemporaryDirectory(prefix="step28-v9-2-lineage-") as temp:
            consumed = Path(temp) / "quality.consumed.json"
            consumed.write_bytes(common.canonical_json_bytes(authorization) + b"\n")
            with mock.patch.object(
                truth_v9_2,
                "EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH",
                consumed.resolve(),
            ):
                capability = (
                    truth_v9_2.ConsumedQualityRunCapabilityV92._from_consumed_authorization(
                        authorization=authorization,
                        consumed_path=consumed,
                    )
                )
            runner._validate_root_claim_and_bindings(
                root_manifest=root_manifest,
                policy=self.policy,
                run_capability=capability,
            )
            root_manifest["design_build_authorization"]["receipt"]["sha256"] = "z" * 64
            with self.assertRaisesRegex(
                runner.QualityAuditRunnerV92Error,
                "root claim/policy/key binding drift",
            ):
                runner._validate_root_claim_and_bindings(
                    root_manifest=root_manifest,
                    policy=self.policy,
                    run_capability=capability,
                )

    def test_any_capability_widening_is_rejected(self) -> None:
        for field in (
            "audit_a_b_truth_open",
            "formal_500_by_4",
            "model_training",
            "model_metric_generation",
        ):
            with self.subTest(field=field):
                value = self.authorization()
                value["capabilities"][field] = True
                value["canonical_self_hash"] = runner._canonical_self_hash(value)
                with self.assertRaisesRegex(
                    runner.QualityAuditRunnerV92Error,
                    "authorization identity drift",
                ):
                    runner.validate_run_authorization(
                        value, policy=self.policy, verify_bound_files=False
                    )

    def test_formal_consumers_require_the_separate_authorization(self) -> None:
        self.assertIn(
            "run_capability",
            inspect.signature(validator.evaluate_formal_probe_families).parameters,
        )
        self.assertIn(
            "run_capability",
            inspect.signature(aggregator.aggregate_formal_structure).parameters,
        )
        self.assertIs(validator.truth_capability, truth_v9_2)
        with self.assertRaisesRegex(
            aggregator.QualityStructureAggregationV92Error,
            "consumed formal structure capability",
        ):
            aggregator.aggregate_formal_structure(
                public_rows_by_split={},
                structure_rows_by_split={},
                eligibility_rows_by_split={},
                model_surface_rows_by_split={},
                policy=self.policy,
                run_capability={"capabilities": runner.CAPABILITIES},
            )
        with self.assertRaisesRegex(
            validator.QualityProbeValidationV92Error,
            "consumed formal quality capability",
        ):
            validator.evaluate_formal_probe_families(
                text_train_matrices=(),
                text_development_matrices=(),
                code_train_matrices=(),
                code_development_matrices=(),
                train_text_eligibility=None,
                development_text_eligibility=None,
                dataset_root=ROOT,
                root_manifest_pin=truth_v9_2.RootManifestPin(
                    path="root_manifest.json",
                    size_bytes=0,
                    sha256="0" * 64,
                    canonical_self_hash="0" * 64,
                ),
                policy=self.policy,
                run_capability={"capabilities": runner.CAPABILITIES},
                verify_label_free_bytes=lambda: None,
            )
        with self.assertRaisesRegex(
            truth_v9_2.QualityTruthCapabilityError,
            "_from_consumed_authorization",
        ):
            truth_v9_2.ConsumedQualityRunCapabilityV92()

    def test_manifest_contract_has_exactly_eight_model_inputs_and_twenty_files(self) -> None:
        self.assertEqual(len(runner.SURFACE_FILES), 4)
        self.assertEqual(
            [path for pair in runner.SURFACE_FILES.values() for path in pair],
            list(runner.builder_v9_2.MODEL_INPUT_PATHS),
        )
        self.assertEqual(len(runner.EXPECTED_SPLIT_DATA_PATHS), 20)
        self.assertEqual(len(set(runner.EXPECTED_SPLIT_DATA_PATHS)), 20)

    def test_formal_capability_uses_only_canonical_consumed_authorization(self) -> None:
        self.assertEqual(
            truth_v9_2.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH.resolve(),
            runner.AUTHORIZATION_PATH.with_name(
                runner.AUTHORIZATION_PATH.stem + ".consumed.json"
            ).resolve(),
        )

    def test_every_manifest_payload_is_physically_rehashed_and_row_counted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step28-v9-2-all-payloads-") as temp:
            root = Path(temp)
            manifests = {}
            for split in runner.SPLITS:
                files = []
                for relative in runner.EXPECTED_SPLIT_DATA_PATHS:
                    path = root / split / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = b"field\n" if path.suffix == ".csv" else b""
                    path.write_bytes(payload)
                    files.append(
                        {
                            "path": relative,
                            "size_bytes": len(payload),
                            "sha256": common.sha256_file(path),
                            "row_count": 0,
                        }
                    )
                manifests[split] = {"files": files}
            verified = runner._verify_all_manifest_payloads(
                dataset_root=root,
                manifests=manifests,
            )
            self.assertTrue(
                all(
                    len(verified[split]["sources"]) == 20
                    for split in runner.SPLITS
                )
            )
            self.assertIn(
                "private/pair_labels.csv", verified["audit_a"]["sources"]
            )
            identity_path = root / "train" / "observed/identity33_all_pairs.csv"
            identity_path.write_bytes(b"other\n")
            with self.assertRaisesRegex(
                runner.QualityAuditRunnerV92Error,
                "bytes drift",
            ):
                runner._verify_all_manifest_payloads(
                    dataset_root=root,
                    manifests=manifests,
                )

    def test_authorization_consumption_is_atomic_and_one_shot(self) -> None:
        value = self.authorization()
        with tempfile.TemporaryDirectory(prefix="step28-v9-2-auth-") as temp:
            path = Path(temp) / "authorization.json"
            path.write_bytes(common.canonical_json_bytes(value) + b"\n")
            consumed = runner._consume_authorization(path, value)
            self.assertFalse(path.exists())
            self.assertTrue(consumed.is_file())
            with mock.patch.object(
                truth_v9_2,
                "EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH",
                consumed.resolve(),
            ):
                capability = (
                    truth_v9_2.ConsumedQualityRunCapabilityV92._from_consumed_authorization(
                        authorization=value,
                        consumed_path=consumed,
                    )
                )
            self.assertEqual(
                capability.design_root_binding(), value["design_root_manifest"]
            )
            with self.assertRaisesRegex(
                runner.QualityAuditRunnerV92Error, "already consumed"
            ):
                path.write_bytes(common.canonical_json_bytes(value) + b"\n")
                runner._consume_authorization(path, value)

    def test_arbitrary_consumed_receipt_cannot_issue_formal_capability(self) -> None:
        value = self.authorization()
        with tempfile.TemporaryDirectory(prefix="step28-v9-2-unconsumed-") as temp:
            path = Path(temp) / "forged.consumed.json"
            path.write_bytes(common.canonical_json_bytes(value) + b"\n")
            with self.assertRaisesRegex(
                truth_v9_2.QualityTruthCapabilityError,
                "identity drift",
            ):
                truth_v9_2.ConsumedQualityRunCapabilityV92._from_consumed_authorization(
                    authorization=value,
                    consumed_path=path,
                )

    def test_default_entry_is_closed_without_external_receipt(self) -> None:
        if runner.AUTHORIZATION_PATH.exists():
            self.skipTest("A separately approved external receipt exists")
        with self.assertRaisesRegex(
            runner.QualityAuditRunnerV92Error, "remains unauthorized"
        ):
            runner.load_run_authorization()

    def test_builder_entry_is_closed_without_two_external_receipts(self) -> None:
        if builder.RANDOM_AUTHORITY_PATH.exists() or builder.BUILD_AUTHORIZATION_PATH.exists():
            self.skipTest("Separately approved external build receipts exist")
        with self.assertRaisesRegex(
            builder.DatasetBuilderV92Error, "remain unauthorized"
        ):
            builder.run_design_preflight_once()

    def test_builder_reverifies_the_final_path_and_rolls_back_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step28-v9-2-publish-") as temp:
            parent = Path(temp)
            temporary = parent / ".root.building"
            output = parent / "root"
            temporary.mkdir()
            calls: list[Path] = []

            def verify(path: Path, _manifest: dict) -> None:
                calls.append(path)
                if path == output:
                    raise builder.DatasetBuilderV92Error(
                        "fixture final-path replay drift"
                    )

            with mock.patch.object(builder, "_verify_output_tree", side_effect=verify):
                with self.assertRaisesRegex(
                    builder.DatasetBuilderV92Error,
                    "final-path replay drift",
                ):
                    builder._publish_verified_output(
                        temp_root=temporary,
                        output_root=output,
                        root_manifest={},
                    )
            self.assertEqual(calls, [temporary, output])
            self.assertTrue(temporary.is_dir())
            self.assertFalse(output.exists())

    def test_fresh_fixture_authority_builds_only_an_in_memory_1004_context(self) -> None:
        authority = builder.validate_random_authority(
            self.random_authority(), quality_policy=self.policy
        )
        context, _parent = builder._build_execution_context(
            quality_policy=self.policy,
            random_authority=authority,
            build_authorization={
                "output_root": (
                    "reports/step28_v13_v1_13_scientific_builder/"
                    "v9_2_fixture_never_written/method_qualification_1004"
                )
            },
        )
        self.assertEqual(len(context.world_records), 1004)
        self.assertEqual(context.execution_mode, builder.EXECUTION_MODE)
        self.assertTrue(context.scientific_use_forbidden)
        self.assertFalse(context.output_root.exists())

    def test_reuse_of_any_parent_authority_is_rejected(self) -> None:
        parent = builder._load_parent_builder_policy(self.policy)
        reused_values = (
            parent["public_preflight_keys"]["small_smoke"][
                "id_namespace_key_hex"
            ],
            next(iter(builder.RETIRED_PREFLIGHT_AUTHORITIES)),
        )
        for reused in reused_values:
            with self.subTest(reused=reused):
                value = self.random_authority()
                value["keys"]["id_namespace_key_hex"] = reused
                values = builder._key_values(value["keys"])
                value["authority_bundle_sha256"] = common.canonical_sha256(
                    sorted(values)
                )
                value["canonical_self_hash"] = builder._self_hash(value)
                with self.assertRaisesRegex(
                    builder.DatasetBuilderV92Error, "reuses a prior authority"
                ):
                    builder.validate_random_authority(
                        value, quality_policy=self.policy
                    )

    def test_truth_adapter_rejects_the_old_v9_execution_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step28-v9-2-truth-root-") as temp:
            root = Path(temp)
            manifest = {
                "version": truth_v9_2.BUILDER_VERSION,
                "status": "PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED",
                "execution_mode": "design_preflight",
                "split_order": list(truth_v9_2.SPLITS),
                "world_count": 1004,
                "model_input_file_count": 8,
                "scientific_use_forbidden": True,
                "formal_seed_created": False,
                "formal_rows_created": 0,
                "training_started": False,
                "split_manifest_self_hashes": {split: "b" * 64 for split in truth_v9_2.SPLITS},
            }
            manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
            path = root / "root_manifest.json"
            common.write_json(path, manifest)
            pin = truth_v9_2.RootManifestPin(
                path="root_manifest.json",
                size_bytes=path.stat().st_size,
                sha256=common.sha256_file(path),
                canonical_self_hash=manifest["canonical_self_hash"],
            )
            with self.assertRaisesRegex(
                truth_v9_2.QualityTruthDatasetGateError,
                "root boundary drift",
            ):
                truth_v9_2.FormalTrainDevelopmentTruthCapability.from_pinned_design_root(
                    dataset_root=root,
                    root_manifest_pin=pin,
                )

    def test_post_truth_label_free_byte_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step28-v9-2-reverify-") as temp:
            root = Path(temp)
            path = root / "observed.jsonl"
            path.write_bytes(b'{"value":1}\n')
            snapshots = {
                "observed.jsonl": (path.stat().st_size, common.sha256_file(path))
            }
            runner._reverify_label_free_bytes(
                dataset_root=root, snapshots=snapshots
            )
            path.write_bytes(b'{"value":2}\n')
            with self.assertRaisesRegex(
                runner.QualityAuditRunnerV92Error, "changed after truth"
            ):
                runner._reverify_label_free_bytes(
                    dataset_root=root, snapshots=snapshots
                )

    def test_machine_terminal_result_is_exclusive_and_replayable(self) -> None:
        result = runner._classified_failure(
            status="AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
            stage="fixture",
            exc=ValueError("fixture"),
        )
        with tempfile.TemporaryDirectory(prefix="step28-v9-2-terminal-") as temp:
            path = Path(temp) / "quality_audit_terminal.json"
            persisted = runner._publish_terminal_exclusive(path, result)
            self.assertEqual(persisted, result)
            self.assertTrue(path.is_file())
            with self.assertRaises(runner.AuditorExecutionV92Error):
                runner._publish_terminal_exclusive(path, result)


if __name__ == "__main__":
    unittest.main()
