#!/usr/bin/env python3
"""Contracts for the V9.2 immutable policy and one-shot audit call layer."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys
import tempfile
import unittest


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
            "run_authorization",
            inspect.signature(validator.evaluate_formal_probe_families).parameters,
        )
        self.assertIn(
            "run_authorization",
            inspect.signature(aggregator.aggregate_formal_structure).parameters,
        )
        self.assertIs(validator.truth_capability, truth_v9_2)

    def test_manifest_contract_has_exactly_eight_model_inputs_and_twenty_files(self) -> None:
        self.assertEqual(len(runner.SURFACE_FILES), 4)
        self.assertEqual(
            [path for pair in runner.SURFACE_FILES.values() for path in pair],
            list(runner.builder_v9_2.MODEL_INPUT_PATHS),
        )
        self.assertEqual(len(runner.EXPECTED_SPLIT_DATA_PATHS), 20)
        self.assertEqual(len(set(runner.EXPECTED_SPLIT_DATA_PATHS)), 20)

    def test_authorization_consumption_is_atomic_and_one_shot(self) -> None:
        value = self.authorization()
        with tempfile.TemporaryDirectory(prefix="step28-v9-2-auth-") as temp:
            path = Path(temp) / "authorization.json"
            path.write_bytes(common.canonical_json_bytes(value) + b"\n")
            consumed = runner._consume_authorization(path, value)
            self.assertFalse(path.exists())
            self.assertTrue(consumed.is_file())
            with self.assertRaisesRegex(
                runner.QualityAuditRunnerV92Error, "already consumed"
            ):
                path.write_bytes(common.canonical_json_bytes(value) + b"\n")
                runner._consume_authorization(path, value)

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


if __name__ == "__main__":
    unittest.main()
