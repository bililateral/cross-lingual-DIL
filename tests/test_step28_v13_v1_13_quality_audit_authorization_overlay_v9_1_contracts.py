#!/usr/bin/env python3
"""Contracts for the V9.1 one-shot design-quality audit authorization overlay."""

from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_audit_runner_v9 as frozen_runner
import step28_v13_v1_13_quality_audit_execution_adapter_v9_1 as execution_adapter
import step28_v13_v1_13_quality_probe_validator_v9 as frozen_probe_validator
import step28_v13_v1_13_run_quality_audit_once_v9_1 as overlay


class QualityAuditAuthorizationOverlayV9Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = overlay._load_overlay_policy()
        cls.static = overlay._validate_static_inputs(cls.policy)
        cls.git_commit = "a" * 40
        cls.git_tree = "b" * 40
        private_root = ROOT / "private_custody"
        private_root.mkdir(parents=True, exist_ok=True)
        cls.execution_pending_path = (
            ROOT / execution_adapter.PENDING_RECEIPT_RELATIVE_PATH
        ).resolve()
        if cls.execution_pending_path.exists():
            raise RuntimeError(
                "Refusing to overwrite a pending formal quality-audit receipt"
            )
        cls.execution_fixture_consumed_paths: set[Path] = set()
        cls.execution_git_identity_patch = mock.patch.object(
            execution_adapter,
            "_git_identity",
            return_value=(cls.git_commit, cls.git_tree),
        )
        cls.execution_git_identity_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.execution_git_identity_patch.stop()
        for path in cls.execution_fixture_consumed_paths:
            path.unlink(missing_ok=True)
        cls.execution_pending_path.unlink(missing_ok=True)

    def _receipt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            **overlay._expected_receipt_bindings(
                policy=self.policy,
                static=self.static,
                git_commit=self.git_commit,
                git_tree=self.git_tree,
            ),
            "review_conversation_url": (
                "https://chatgpt.com/c/00000000-0000-0000-0000-000000000001"
            ),
            "review_response_sha256": "c" * 64,
            "reviewed_at_utc": "2026-08-20T12:34:56Z",
        }
        payload["canonical_self_hash"] = hashlib.sha256(
            overlay._canonical_json_bytes(payload)
        ).hexdigest()
        return payload

    def _execution_receipt(
        self,
    ) -> tuple[dict[str, object], Path, dict[str, object]]:
        payload = self._receipt_payload()
        raw = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        receipt_sha256 = hashlib.sha256(raw).hexdigest()
        pending = self.execution_pending_path
        consumed = pending.with_name(
            f"{pending.stem}.consumed.{receipt_sha256}.json"
        )
        if consumed.exists():
            self.assertEqual(consumed.read_bytes(), raw)
        else:
            pending.write_bytes(raw)
            pending.replace(consumed)
        self.execution_fixture_consumed_paths.add(consumed)
        return payload, pending, {
            "path": consumed.relative_to(ROOT).as_posix(),
            "size_bytes": len(raw),
            "sha256": receipt_sha256,
        }

    def _execution(self) -> execution_adapter.ConsumedQualityAuditExecution:
        payload, pending, consumed = self._execution_receipt()
        return execution_adapter.build_consumed_execution(
            receipt_id=str(payload["canonical_self_hash"]),
            overlay_policy_canonical_self_hash=self.policy[
                "canonical_self_hash"
            ],
            base_policy=self.static["quality_policy"],
            capabilities=self.policy["external_receipt"][
                "required_capabilities"
            ],
            pending_receipt_path=pending,
            consumed_receipt_binding=consumed,
            result_path=ROOT / self.policy["execution"]["result_path"],
            dataset_root=self.static["design_root"],
            root_manifest_binding=self.static["root_manifest"],
        )

    @staticmethod
    def _write_receipt(path: Path, payload: dict[str, object]) -> bytes:
        raw = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        path.write_bytes(raw)
        return raw

    def _early_invalidation_result(self) -> dict[str, object]:
        split_receipts = {
            split: {
                "world_count": self.static["quality_policy"]["design_scale"][
                    "world_counts"
                ][split],
                "seller_row_count": 0,
                "registered_code_count": 0,
                "code_character_position_maximum_absolute_deviation": 0.0,
            }
            for split in frozen_runner.SPLITS
        }
        zeros = {
            key: 0
            for key in frozen_runner.structure_aggregator.ZERO_TOLERANCE_FIELDS
        }
        zeros["prior_world_code_hits"] = 1
        structure: dict[str, object] = {
            "version": frozen_runner.structure_aggregator.VERSION,
            "status": "DATASET_INVALIDATED",
            "claim_boundary": (
                "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
            ),
            "split_receipts": split_receipts,
            "zero_tolerance_counts": zeros,
            "gate_failures": ["prior_world_code_hits"],
            "truth_label_row_count_read": 0,
            "audit_truth_open_count": 0,
            "audit_truth_read_count": 0,
            "audit_truth_materialized_row_count": 0,
            "forbidden_read_counts": {
                "audit_truth": {
                    "open_count": 0,
                    "read_count": 0,
                    "materialized_row_count": 0,
                },
                "generator_quality_result": 0,
                "candidate_quality_result": 0,
                "view_builder_quality_result": 0,
            },
        }
        structure["canonical_self_hash"] = overlay._canonical_self_hash(structure)
        declared = overlay._declared_data_bindings(self.static)
        required = {
            frozen_runner.WORLDS_PATH,
            frozen_runner.ENDPOINT_PATH,
            frozen_runner.PUBLIC_CODE_PATH,
            frozen_runner.ELIGIBILITY_PATH,
            frozen_runner.STRUCTURE_AUDIT_PATH,
            *(
                path
                for pair in frozen_runner.SURFACE_FILES.values()
                for path in pair
            ),
        }
        actual = sorted(
            {
                f"{split}/{relative}"
                for split in frozen_runner.SPLITS
                for relative in required
            },
            key=str.encode,
        )
        manifest_only = sorted(set(declared) - set(actual), key=str.encode)

        def commitment(paths: list[str]) -> str:
            return hashlib.sha256(
                overlay._canonical_json_bytes([declared[path] for path in paths])
            ).hexdigest()

        scope = {
            "declared_data_file_count": 72,
            "label_free_actual_byte_verified_count": 44,
            "supervised_truth_actual_byte_verified_count": 0,
            "actual_byte_verified_count": 44,
            "actual_byte_verified_paths": actual,
            "actual_byte_verified_binding_sha256": commitment(actual),
            "manifest_pin_only_count": 28,
            "manifest_pin_only_paths": manifest_only,
            "manifest_pin_only_binding_sha256": commitment(manifest_only),
            "audit_a_b_truth_manifest_pin_only_paths": [
                "audit_a/private/pair_labels.csv",
                "audit_b/private/pair_labels.csv",
            ],
            "audit_a_b_truth_actual_byte_read_count": 0,
            "declared_unclassified_count": 0,
            "scope_claim": (
                "ACTUAL_BYTES_VERIFIED_ONLY_FOR_LISTED_PATHS_"
                "OTHER_PATHS_MANIFEST_PIN_ONLY"
            ),
        }
        result: dict[str, object] = {
            "version": frozen_runner.VERSION,
            "status": "DATASET_INVALIDATED",
            "claim_boundary": (
                "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
            ),
            "structure": structure,
            "input_file_verification_scope": scope,
            "supervised_truth_opened": False,
            "audit_a_b_truth_open_count": 0,
            "formal_500_by_4_generated": False,
            "training_started": False,
        }
        result["canonical_self_hash"] = overlay._canonical_self_hash(result)
        return result

    def test_overlay_and_current_static_inputs_close_exactly(self) -> None:
        self.assertEqual(
            self.policy["canonical_self_hash"],
            overlay.OVERLAY_POLICY_CANONICAL_SELF_HASH,
        )

    def test_v9_1_equivalence_summary_is_exact_and_mutations_fail(self) -> None:
        root_path = self.static["design_root"] / "root_manifest.json"
        observed = json.loads(root_path.read_text(encoding="utf-8"))[
            "v9_1_equivalence_replay"
        ]
        expected = self.policy["design_root"]["v9_1_equivalence_replay"]
        overlay._validate_v9_1_equivalence_binding(expected, observed)
        mutations: list[tuple[dict[str, object], dict[str, object]]] = []
        changed_expected = copy.deepcopy(expected)
        changed_expected["allowed_changed_json_paths"] = [
            *changed_expected["allowed_changed_json_paths"],
            "/unexpected_path",
        ]
        mutations.append((changed_expected, copy.deepcopy(observed)))
        changed_observed = copy.deepcopy(observed)
        changed_observed["same_random_authority"] = False
        mutations.append((copy.deepcopy(expected), changed_observed))
        changed_observed = copy.deepcopy(observed)
        changed_observed["unchanged_file_count"] = 67
        mutations.append((copy.deepcopy(expected), changed_observed))
        changed_observed = copy.deepcopy(observed)
        changed_observed["canonical_self_hash"] = "0" * 64
        mutations.append((copy.deepcopy(expected), changed_observed))
        for expected_value, observed_value in mutations:
            with self.subTest(
                expected=expected_value, observed=observed_value
            ):
                with self.assertRaisesRegex(
                    overlay.QualityAuditAuthorizationError,
                    "equivalence binding drift",
                ):
                    overlay._validate_v9_1_equivalence_binding(
                        expected_value, observed_value
                    )

    def test_quality_attempt_one_is_distinct_from_design_attempt_two(self) -> None:
        root_manifest = json.loads(
            (self.static["design_root"] / "root_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            root_manifest["design_build_authorization"]["attempt_index"], 2
        )
        self.assertEqual(self.policy["external_receipt"]["attempt_index"], 1)
        self.assertNotEqual(
            root_manifest["design_build_authorization"]["attempt_index"],
            self.policy["external_receipt"]["attempt_index"],
        )
        self.assertEqual(
            overlay._canonical_self_hash(self.policy),
            overlay.OVERLAY_POLICY_CANONICAL_SELF_HASH,
        )
        self.assertEqual(
            self.static["root_manifest"]["canonical_self_hash"],
            "f10086faa5f68b08a4d25a6e49943fb18ede0858ca50bad711d7bb2f4d94200f",
        )
        self.assertEqual(
            {
                split: binding["canonical_self_hash"]
                for split, binding in self.static["split_manifests"].items()
            },
            {
                "train": "414e314842d52d0d367d5a72a6f17498c3091cdf01c20bf12ba3fa310d40ba8a",
                "development": "844ea6f73872b91c0e5a93e5b45c7f919d0dd049c46f1634f5c06f1c49c05b02",
                "audit_a": "75999b5609df6d1626182f5e5aa019738012cb697a664c526b7377367e226a0c",
                "audit_b": "ad5210685f6a4656f84eac21884a5288e0355d772bedf69ea830153ad7f91558",
            },
        )

    def test_frozen_policy_bytes_and_closed_authorization_remain_unchanged(self) -> None:
        spec = self.policy["frozen_quality_contract"]["policy"]
        path = ROOT / spec["path"]
        before = path.read_bytes()
        with mock.patch.object(
            execution_adapter,
            "PENDING_RECEIPT_RELATIVE_PATH",
            self.policy["external_receipt"]["pending_path"],
        ):
            overlay._validate_static_inputs(self.policy)
        after = path.read_bytes()
        self.assertEqual(before, after)
        self.assertEqual(len(after), spec["size_bytes"])
        self.assertEqual(hashlib.sha256(after).hexdigest(), spec["sha256"])
        authorization = self.static["quality_policy"]["authorization"]
        self.assertTrue(authorization["implementation_and_fixture_tests"])
        self.assertTrue(
            all(
                value is False
                for key, value in authorization.items()
                if key != "implementation_and_fixture_tests"
            )
        )

    def test_original_public_runner_remains_unauthorized(self) -> None:
        with self.assertRaisesRegex(
            frozen_runner.QualityAuditRunnerError, "unauthorized"
        ):
            frozen_runner.run_formal_quality_audit()

    def test_v9_1_attempt_one_paths_cannot_reuse_v9_artifacts(self) -> None:
        self.assertEqual(self.policy["external_receipt"]["attempt_index"], 1)
        self.assertIn(
            "v9_1",
            self.policy["external_receipt"]["pending_path"],
        )
        self.assertIn(
            "attempt1",
            self.policy["execution"]["result_path"],
        )
        self.assertEqual(execution_adapter.EXPECTED_ATTEMPT_INDEX, 1)

    def test_execution_context_keeps_base_policy_closed_and_binds_exact_root(
        self,
    ) -> None:
        execution = self._execution()
        execution_adapter.validate_consumed_execution(
            execution, base_policy=self.static["quality_policy"]
        )
        self.assertEqual(
            execution.root_binding(), self.static["root_manifest"]
        )
        self.assertEqual(
            execution.root_pin().path,
            execution_adapter.ROOT_MANIFEST_PIN_PATH,
        )
        self.assertEqual(execution.root_pin().path, "root_manifest.json")
        authorization = self.static["quality_policy"]["authorization"]
        self.assertFalse(authorization["quality_audit_run"])
        self.assertFalse(authorization["metric_generation"])
        widened = dict(
            self.policy["external_receipt"]["required_capabilities"]
        )
        widened["model_training"] = True
        payload, pending, consumed = self._execution_receipt()
        with self.assertRaisesRegex(
            execution_adapter.QualityAuditExecutionAdapterError, "widened"
        ):
            execution_adapter.build_consumed_execution(
                receipt_id=str(payload["canonical_self_hash"]),
                overlay_policy_canonical_self_hash=self.policy[
                    "canonical_self_hash"
                ],
                base_policy=self.static["quality_policy"],
                capabilities=widened,
                pending_receipt_path=pending,
                consumed_receipt_binding=consumed,
                result_path=ROOT / self.policy["execution"]["result_path"],
                dataset_root=self.static["design_root"],
                root_manifest_binding=self.static["root_manifest"],
            )
        changed = replace(execution, root_manifest_sha256="e" * 64)
        with self.assertRaisesRegex(
            execution_adapter.QualityAuditExecutionAdapterError, "root pin"
        ):
            execution_adapter.validate_consumed_execution(
                changed, base_policy=self.static["quality_policy"]
            )

    def test_real_truth_capability_composition_accepts_root_local_pin(self) -> None:
        execution = self._execution()
        truth = execution_adapter._build_bound_truth_capability(
            execution=execution,
            policy=self.static["quality_policy"],
        )
        self.assertEqual(truth.root_binding(), execution.root_binding())
        self.assertEqual(execution.root_pin().path, "root_manifest.json")

    def test_direct_adapter_without_consumed_receipt_fails_before_root_read(
        self,
    ) -> None:
        execution = self._execution()
        consumed = execution.consumed_receipt_path
        pending = execution.pending_receipt_path
        consumed.replace(pending)
        try:
            with mock.patch.object(
                frozen_runner, "_load_root_manifests"
            ) as load_root:
                with self.assertRaisesRegex(
                    execution_adapter.QualityAuditExecutionAdapterError,
                    "receipt file binding",
                ):
                    execution_adapter.run_authorized_formal_quality_audit(
                        policy=self.static["quality_policy"],
                        execution=execution,
                        state={},
                    )
            load_root.assert_not_called()
        finally:
            pending.replace(consumed)

    def test_direct_adapter_rechecks_complete_receipt_binding(self) -> None:
        payload, pending, _valid_binding = self._execution_receipt()
        changed = copy.deepcopy(payload)
        changed["frozen_quality_runner"]["sha256"] = "0" * 64
        changed.pop("canonical_self_hash")
        changed["canonical_self_hash"] = hashlib.sha256(
            overlay._canonical_json_bytes(changed)
        ).hexdigest()
        raw = (
            json.dumps(changed, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        receipt_sha256 = hashlib.sha256(raw).hexdigest()
        consumed = pending.with_name(
            f"{pending.stem}.consumed.{receipt_sha256}.json"
        )
        consumed.write_bytes(raw)
        try:
            with self.assertRaisesRegex(
                execution_adapter.QualityAuditExecutionAdapterError,
                "complete binding drift",
            ) as raised:
                execution_adapter.build_consumed_execution(
                    receipt_id=str(changed["canonical_self_hash"]),
                    overlay_policy_canonical_self_hash=self.policy[
                        "canonical_self_hash"
                    ],
                    base_policy=self.static["quality_policy"],
                    capabilities=self.policy["external_receipt"][
                        "required_capabilities"
                    ],
                    pending_receipt_path=pending,
                    consumed_receipt_binding={
                        "path": consumed.relative_to(ROOT).as_posix(),
                        "size_bytes": len(raw),
                        "sha256": receipt_sha256,
                    },
                    result_path=ROOT / self.policy["execution"]["result_path"],
                    dataset_root=self.static["design_root"],
                    root_manifest_binding=self.static["root_manifest"],
                )
            self.assertRegex(str(raised.exception.__cause__), "receipt binding drift")
        finally:
            consumed.unlink(missing_ok=True)

    def test_file_verification_scope_is_explicit_for_all_72_records(self) -> None:
        root = self.static["design_root"]
        manifests = {
            split: json.loads(
                (root / split / "split_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            for split in frozen_runner.SPLITS
        }
        loaded: dict[str, dict[str, object]] = {}
        truth_access: dict[str, dict[str, object]] = {}
        for split in frozen_runner.SPLITS:
            records = frozen_runner._manifest_records(manifests[split])
            required = {
                frozen_runner.WORLDS_PATH,
                frozen_runner.ENDPOINT_PATH,
                frozen_runner.PUBLIC_CODE_PATH,
                frozen_runner.ELIGIBILITY_PATH,
                frozen_runner.STRUCTURE_AUDIT_PATH,
                *(
                    path
                    for pair in frozen_runner.SURFACE_FILES.values()
                    for path in pair
                ),
            }
            loaded[split] = {
                "sources": {
                    relative: execution_adapter.preparer.SourceCommitment(
                        path=f"{split}/{relative}",
                        size_bytes=int(records[relative]["size_bytes"]),
                        sha256=str(records[relative]["sha256"]),
                    )
                    for relative in required
                }
            }
            if split in execution_adapter.truth_capability.SUPERVISED_SPLITS:
                truth_record = records[
                    execution_adapter.truth_capability.TRUTH_RELATIVE_PATH
                ]
                truth_access[split] = {
                    "file_open_count": 1,
                    "byte_read_count": int(truth_record["size_bytes"]),
                    "materialized_row_count": int(truth_record["row_count"]),
                    "sha256": str(truth_record["sha256"]),
                }
            else:
                truth_access[split] = {
                    "file_open_count": 0,
                    "byte_read_count": 0,
                    "materialized_row_count": 0,
                }
        scope = execution_adapter._input_file_verification_scope(
            manifests=manifests,
            loaded=loaded,
            supervised_receipt={"truth_file_access": truth_access},
        )
        self.assertEqual(scope["declared_data_file_count"], 72)
        self.assertEqual(scope["label_free_actual_byte_verified_count"], 44)
        self.assertEqual(
            scope["supervised_truth_actual_byte_verified_count"], 2
        )
        self.assertEqual(scope["actual_byte_verified_count"], 46)
        self.assertEqual(scope["manifest_pin_only_count"], 26)
        self.assertEqual(scope["declared_unclassified_count"], 0)
        self.assertEqual(
            scope["audit_a_b_truth_manifest_pin_only_paths"],
            [
                "audit_a/private/pair_labels.csv",
                "audit_b/private/pair_labels.csv",
            ],
        )

    def test_real_adapter_reaches_root_manifest_without_mocking_body(self) -> None:
        class ReachedBuilderBinding(RuntimeError):
            pass

        class StageStop(dict[str, str]):
            def __setitem__(self, key: str, value: str) -> None:
                super().__setitem__(key, value)
                if value == "builder_policy_binding":
                    raise ReachedBuilderBinding(value)

        policy_before = overlay._canonical_json_bytes(
            self.static["quality_policy"]
        )
        state: dict[str, str] = StageStop()
        with self.assertRaisesRegex(
            ReachedBuilderBinding, "builder_policy_binding"
        ):
            execution_adapter.run_authorized_formal_quality_audit(
                policy=self.static["quality_policy"],
                execution=self._execution(),
                state=state,
            )
        self.assertEqual(state["stage"], "builder_policy_binding")
        self.assertEqual(
            overlay._canonical_json_bytes(self.static["quality_policy"]),
            policy_before,
        )

    def test_actual_v9_1_label_free_views_reach_feature_freeze(self) -> None:
        class ReachedFeatureFreeze(RuntimeError):
            pass

        class StageStop(dict[str, str]):
            def __setitem__(self, key: str, value: str) -> None:
                super().__setitem__(key, value)
                if value == "label_free_feature_freeze":
                    raise ReachedFeatureFreeze(value)

        state: dict[str, str] = StageStop()
        with self.assertRaisesRegex(
            ReachedFeatureFreeze, "label_free_feature_freeze"
        ):
            execution_adapter.run_authorized_formal_quality_audit(
                policy=self.static["quality_policy"],
                execution=self._execution(),
                state=state,
            )
        self.assertEqual(state["stage"], "label_free_feature_freeze")
        self.assertNotEqual(state["stage"], "train_development_truth_and_supervised_gates")

    def test_consumed_capability_crosses_structure_authorization_gate(self) -> None:
        empty = {split: () for split in frozen_runner.SPLITS}
        with self.assertRaisesRegex(
            frozen_runner.structure_aggregator.QualityStructureAggregationError,
            "Structural world universe drift",
        ):
            execution_adapter.aggregate_authorized_formal_structure(
                public_rows_by_split=empty,
                structure_rows_by_split=empty,
                policy=self.static["quality_policy"],
                execution=self._execution(),
            )

    def test_production_adapter_uses_consumed_structure_bridge(self) -> None:
        source = inspect.getsource(
            execution_adapter.run_authorized_formal_quality_audit
        )
        self.assertIn("aggregate_authorized_formal_structure(", source)
        self.assertNotIn("aggregate_formal_structure(", source)

    def test_structure_bridge_revalidates_consumed_receipt_after_core(self) -> None:
        execution = self._execution()
        original = execution.consumed_receipt_path.read_bytes()

        def drift_receipt(**_kwargs: object) -> dict[str, object]:
            execution.consumed_receipt_path.write_bytes(original + b" ")
            return {"status": "PASS"}

        try:
            with mock.patch.object(
                frozen_runner.structure_aggregator,
                "_aggregate",
                side_effect=drift_receipt,
            ), self.assertRaisesRegex(
                execution_adapter.QualityAuditExecutionAdapterError,
                "receipt file binding drift",
            ):
                execution_adapter.aggregate_authorized_formal_structure(
                    public_rows_by_split={
                        split: () for split in frozen_runner.SPLITS
                    },
                    structure_rows_by_split={
                        split: () for split in frozen_runner.SPLITS
                    },
                    policy=self.static["quality_policy"],
                    execution=execution,
                )
        finally:
            execution.consumed_receipt_path.write_bytes(original)

    def test_structure_bridge_forwards_exact_frozen_formal_parameters(self) -> None:
        empty = {split: () for split in frozen_runner.SPLITS}
        expected = {"status": "PASS"}
        with mock.patch.object(
            frozen_runner.structure_aggregator,
            "_aggregate",
            return_value=expected,
        ) as core:
            observed = execution_adapter.aggregate_authorized_formal_structure(
                public_rows_by_split=empty,
                structure_rows_by_split=empty,
                policy=self.static["quality_policy"],
                execution=self._execution(),
            )
        self.assertIs(observed, expected)
        self.assertEqual(
            core.call_args.kwargs,
            {
                "public_rows_by_split": empty,
                "structure_rows_by_split": empty,
                "expected_world_counts": self.static["quality_policy"][
                    "design_scale"
                ]["world_counts"],
                "expected_sellers_per_world": self.static["quality_policy"][
                    "design_scale"
                ]["seller_count_per_world"],
                "maximum_position_deviation": self.static["quality_policy"][
                    "quality_gates"
                ][
                    "code_character_position_maximum_absolute_deviation_from_one_sixteenth"
                ],
                "enforce_position_margin": True,
                "claim_boundary": (
                    "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
                ),
            },
        )

    def test_adapter_preserves_frozen_runner_stage_order(self) -> None:
        def stages(function: object) -> list[str]:
            tree = ast.parse(inspect.getsource(function))
            return [
                node.value.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "state"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ]

        frozen_stages = stages(
            frozen_runner._run_authorized_formal_quality_audit
        )
        adapted_stages = stages(
            execution_adapter.run_authorized_formal_quality_audit
        )
        self.assertEqual(
            adapted_stages,
            ["consumed_execution_and_root_binding", *frozen_stages],
        )
        source = inspect.getsource(
            execution_adapter.run_authorized_formal_quality_audit
        )
        self.assertNotIn("_root_pin_from_policy", source)
        self.assertNotIn("mock.patch", source)
        self.assertNotIn("monkeypatch", source)
        self.assertIn("execution.root_pin()", source)
        self.assertIn("evaluate_authorized_formal_probe_families", source)

    def test_real_supervised_adapter_crosses_old_auth_and_root_deadlocks(self) -> None:
        with self.assertRaisesRegex(
            frozen_probe_validator.QualityProbeValidationError,
            "matrix view cardinality",
        ):
            execution_adapter.evaluate_authorized_formal_probe_families(
                text_train_matrices=(),
                text_development_matrices=(),
                code_train_matrices=(),
                code_development_matrices=(),
                policy=self.static["quality_policy"],
                train_text_eligibility=None,
                development_text_eligibility=None,
                execution=self._execution(),
            )

    def test_authorized_numeric_core_matches_frozen_core_on_bounded_fixture(
        self,
    ) -> None:
        preparer = frozen_probe_validator.preparer
        np = frozen_probe_validator.np
        source = (
            preparer.SourceCommitment(
                path="fixture/label_free.jsonl",
                size_bytes=10,
                sha256="2" * 64,
            ),
        )

        def matrix_pair(split: str) -> tuple[object, object]:
            keys = tuple(
                (
                    f"{split}_world_{world}",
                    f"{split}_world_{world}_pair_{pair}",
                )
                for world in range(3)
                for pair in range(6)
            )
            first = np.asarray(
                [
                    [((index * 7) % 13) / 13.0, ((index * 5) % 17) / 17.0]
                    for index in range(len(keys))
                ],
                dtype=np.float64,
            )
            second = np.asarray(
                [[((index * 11) % 19) / 19.0] for index in range(len(keys))],
                dtype=np.float64,
            )
            return (
                preparer.freeze_feature_matrix(
                    family="fixture",
                    view="view_a",
                    values=first,
                    row_keys=keys,
                    column_names=("a", "b"),
                    sources=source,
                ),
                preparer.freeze_feature_matrix(
                    family="fixture",
                    view="view_b",
                    values=second,
                    row_keys=keys,
                    column_names=("c",),
                    sources=source,
                ),
            )

        train = matrix_pair("train")
        development = matrix_pair("development")
        truth = {
            split: [
                {
                    "canonical_pair_uid": pair_uid,
                    "world_uid": world_uid,
                    "label": int(pair_uid.endswith("_0") or pair_uid.endswith("_1")),
                }
                for world_uid, pair_uid in values[0].row_keys
            ]
            for split, values in (
                ("train", train),
                ("development", development),
            )
        }
        formal_design = frozen_probe_validator.ProbeFamilyDesign(
            family="fixture",
            view_widths=(("view_a", 2), ("view_b", 1)),
            expected_views=2,
            expected_total_features=3,
            expected_column_name_hashes=None,
            expected_worlds=3,
            pairs_per_world=6,
            positives_per_world=2,
            excluded_pairs_per_world=0,
            average_precision_baseline=2 / 6,
            bootstrap_replicates=31,
            bootstrap_seed=12345,
            require_formal_bootstrap_binding=False,
            claim_boundary=(
                "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
            ),
        )
        fixture_design = replace(
            formal_design,
            claim_boundary="FIXTURE_ONLY_NO_DATASET_CONCLUSION",
        )
        frozen = frozen_probe_validator.evaluate_fixture_probe_family(
            train_matrices=train,
            development_matrices=development,
            truth_loader=lambda split: truth[split],
            design=fixture_design,
            policy=self.static["quality_policy"],
        )
        adapted = execution_adapter._evaluate_authorized_formal_family(
            train_matrices=train,
            development_matrices=development,
            preloaded_truth=truth,
            design=replace(formal_design),
            policy=self.static["quality_policy"],
            train_eligibility=None,
            development_eligibility=None,
            execution=self._execution(),
        )
        comparable_keys = {
            "family",
            "train_world_count",
            "development_world_count",
            "full_pair_count_per_world",
            "eligible_pair_count_per_world",
            "positive_pair_count_per_world",
            "average_precision_baseline",
            "quality_policy_canonical_self_hash",
            "input_commitments",
            "single_feature",
            "model_family",
            "bootstrap",
            "gate_checks",
            "gate_failures",
            "truth_loader_call_counts",
            "row_level_labels_returned",
            "row_level_predictions_returned",
        }
        self.assertEqual(
            {key: frozen[key] for key in comparable_keys},
            {key: adapted[key] for key in comparable_keys},
        )

    def test_public_entry_is_parameterless_and_cli_rejects_arguments(self) -> None:
        self.assertEqual(len(inspect.signature(overlay.run_quality_audit_once).parameters), 0)
        with mock.patch.object(sys, "argv", ["entry.py", "--mode", "other"]), mock.patch.object(
            overlay, "run_quality_audit_once"
        ) as run:
            with self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "accepts no arguments"
            ):
                overlay.main()
        run.assert_not_called()

    def test_strict_json_rejects_duplicate_keys_and_non_finite_values(self) -> None:
        with self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "duplicate"
        ):
            overlay._strict_json_object(b'{"a":1,"a":2}', label="fixture")
        with self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "non-finite"
        ):
            overlay._strict_json_object(b'{"a":NaN}', label="fixture")

    def test_exact_pending_receipt_accepts_and_binding_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            pending = Path(temp) / "quality_authorization.json"
            payload = self._receipt_payload()
            self._write_receipt(pending, payload)
            with mock.patch.object(
                overlay,
                "_receipt_paths",
                return_value=(pending, pending.relative_to(ROOT).as_posix()),
            ), mock.patch.object(
                overlay,
                "_git_identity",
                return_value=(self.git_commit, self.git_tree),
            ):
                verified = overlay._load_and_validate_pending_receipt(
                    policy=self.policy, static=self.static
                )
                self.assertEqual(
                    verified.receipt_id, payload["canonical_self_hash"]
                )
                self.assertEqual(verified.raw, pending.read_bytes())

                mutated = copy.deepcopy(payload)
                mutated["capabilities"]["audit_a_b_truth_open"] = True
                unsigned = dict(mutated)
                unsigned.pop("canonical_self_hash")
                mutated["canonical_self_hash"] = hashlib.sha256(
                    overlay._canonical_json_bytes(unsigned)
                ).hexdigest()
                self._write_receipt(pending, mutated)
                with self.assertRaisesRegex(
                    overlay.QualityAuditAuthorizationError, "binding drift"
                ):
                    overlay._load_and_validate_pending_receipt(
                        policy=self.policy, static=self.static
                    )

    def test_review_metadata_and_self_hash_are_fail_closed(self) -> None:
        for url in (
            "https://example.invalid/review",
            "https://chatgpt.com/c/",
            "https://chatgpt.com/c/00000000-0000-0000-0000-000000000001?x=1",
            "https://chatgpt.com/c/00000000-0000-0000-0000-000000000001/extra",
        ):
            payload = self._receipt_payload()
            payload["review_conversation_url"] = url
            unsigned = dict(payload)
            unsigned.pop("canonical_self_hash")
            payload["canonical_self_hash"] = hashlib.sha256(
                overlay._canonical_json_bytes(unsigned)
            ).hexdigest()
            with self.subTest(url=url), self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "review metadata"
            ):
                overlay._validate_receipt_payload(
                    payload=payload,
                    policy=self.policy,
                    static=self.static,
                    git_commit=self.git_commit,
                    git_tree=self.git_tree,
                )

    def test_receipt_boolean_and_attempt_types_are_strict(self) -> None:
        mutations = (
            ("attempt_index", True),
            (("capabilities", "quality_audit_run"), 1),
            (("capabilities", "formal_seed"), 0),
            (("capabilities", "model_training"), "false"),
            (("capabilities", "audit_a_b_truth_open"), None),
        )
        for path, value in mutations:
            payload = self._receipt_payload()
            if isinstance(path, tuple):
                payload[path[0]][path[1]] = value
            else:
                payload[path] = value
            unsigned = dict(payload)
            unsigned.pop("canonical_self_hash")
            payload["canonical_self_hash"] = hashlib.sha256(
                overlay._canonical_json_bytes(unsigned)
            ).hexdigest()
            with self.subTest(path=path, value=value), self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "JSON type drift"
            ):
                overlay._validate_receipt_payload(
                    payload=payload,
                    policy=self.policy,
                    static=self.static,
                    git_commit=self.git_commit,
                    git_tree=self.git_tree,
                )

    def test_static_input_recomputes_aggregate_file_counts(self) -> None:
        changed = copy.deepcopy(self.policy)
        changed["design_root"]["data_file_count"] += 1
        with self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError,
            "aggregate size drift",
        ):
            overlay._validate_static_inputs(changed)

    def test_receipt_binds_git_wrapper_policy_runner_validator_and_manifests(self) -> None:
        mutation_paths = (
            ("authorization_entry_source", "sha256"),
            ("frozen_quality_policy", "sha256"),
            ("frozen_quality_runner", "sha256"),
            ("frozen_quality_validator", "sha256"),
            ("root_manifest", "sha256"),
            ("split_manifests", "train", "sha256"),
        )
        for path in mutation_paths:
            payload = self._receipt_payload()
            target = payload
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = "0" * 64
            unsigned = dict(payload)
            unsigned.pop("canonical_self_hash")
            payload["canonical_self_hash"] = hashlib.sha256(
                overlay._canonical_json_bytes(unsigned)
            ).hexdigest()
            with self.subTest(path=path), self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "binding drift"
            ):
                overlay._validate_receipt_payload(
                    payload=payload,
                    policy=self.policy,
                    static=self.static,
                    git_commit=self.git_commit,
                    git_tree=self.git_tree,
                )

        payload = self._receipt_payload()
        payload["git_commit"] = "1" * 40
        unsigned = dict(payload)
        unsigned.pop("canonical_self_hash")
        payload["canonical_self_hash"] = hashlib.sha256(
            overlay._canonical_json_bytes(unsigned)
        ).hexdigest()
        with self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "binding drift"
        ):
            overlay._validate_receipt_payload(
                payload=payload,
                policy=self.policy,
                static=self.static,
                git_commit=self.git_commit,
                git_tree=self.git_tree,
            )

    def test_git_identity_requires_a_fully_clean_worktree(self) -> None:
        dirty = mock.Mock(
            stdout="?? browser-temp\n", stderr="", returncode=0
        )
        with mock.patch.object(overlay.subprocess, "run", return_value=dirty) as run:
            with self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "not clean"
            ):
                overlay._git_identity()
        self.assertEqual(
            run.call_args.args[0],
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        )
        payload = self._receipt_payload()
        payload["canonical_self_hash"] = "d" * 64
        with self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "self-hash"
        ):
            overlay._validate_receipt_payload(
                payload=payload,
                policy=self.policy,
                static=self.static,
                git_commit=self.git_commit,
                git_tree=self.git_tree,
            )

    def test_consumption_is_atomic_and_receipt_cannot_be_reused(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            pending = Path(temp) / "quality_authorization.json"
            payload = self._receipt_payload()
            raw = self._write_receipt(pending, payload)
            receipt = overlay.VerifiedQualityAuditReceipt(
                path=pending,
                raw=raw,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                receipt_id=str(payload["canonical_self_hash"]),
                payload=payload,
            )
            consumed = overlay._consume_receipt(receipt)
            consumed_path = ROOT / consumed["path"]
            self.assertFalse(pending.exists())
            self.assertEqual(consumed_path.read_bytes(), raw)
            pending.write_bytes(raw)
            with self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "already consumed"
            ):
                overlay._consume_receipt(receipt)

    def test_consumed_path_is_revalidated_before_audit(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            pending = Path(temp) / "quality_authorization.json"
            payload = self._receipt_payload()
            raw = self._write_receipt(pending, payload)
            receipt = overlay.VerifiedQualityAuditReceipt(
                path=pending,
                raw=raw,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                receipt_id=str(payload["canonical_self_hash"]),
                payload=payload,
            )
            consumed = overlay._consume_receipt(receipt)
            with mock.patch.object(
                overlay,
                "_git_identity",
                return_value=(self.git_commit, self.git_tree),
            ):
                overlay._validate_consumed_receipt(
                    receipt=receipt,
                    consumed_file=consumed,
                    policy=self.policy,
                    static=self.static,
                )
                bad = dict(consumed)
                bad["path"] = pending.relative_to(ROOT).as_posix()
                with self.assertRaisesRegex(
                    overlay.QualityAuditAuthorizationError, "binding drift"
                ):
                    overlay._validate_consumed_receipt(
                        receipt=receipt,
                        consumed_file=bad,
                        policy=self.policy,
                        static=self.static,
                    )

    def test_missing_receipt_fails_before_body_or_result_creation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            missing = base / "missing.json"
            result_directory = base / "result"
            result_path = result_directory / "quality_audit_receipt.json"
            temporary = base / ".result.tmp"
            with mock.patch.object(
                overlay, "_load_overlay_policy", return_value=self.policy
            ), mock.patch.object(
                overlay, "_validate_static_inputs", return_value=self.static
            ), mock.patch.object(
                overlay,
                "_result_paths",
                return_value=(result_directory, result_path, temporary),
            ), mock.patch.object(
                overlay,
                "_receipt_paths",
                return_value=(missing, missing.relative_to(ROOT).as_posix()),
            ), mock.patch.object(overlay, "_execute_frozen_body") as body:
                with self.assertRaisesRegex(
                    overlay.QualityAuditAuthorizationError, "remains unauthorized"
                ):
                    overlay.run_quality_audit_once()
            body.assert_not_called()
            self.assertFalse(result_directory.exists())

    def test_old_v9_receipt_cannot_authorize_v9_1(self) -> None:
        private_root = ROOT / "private_custody"
        with tempfile.TemporaryDirectory(
            prefix="v9-1-old-receipt-rejection-", dir=private_root
        ) as directory:
            root = Path(directory)
            pending = root / "step28_v13_v1_13_v9_1_quality_audit_authorization.json"
            old_v9 = root / "step28_v13_v1_13_v9_quality_audit_authorization.json"
            old_v9.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                overlay,
                "_receipt_paths",
                return_value=(pending, pending.relative_to(ROOT).as_posix()),
            ):
                with self.assertRaisesRegex(
                    overlay.QualityAuditAuthorizationError,
                    "exact one-time receipt is absent",
                ):
                    overlay._load_and_validate_pending_receipt(
                        policy=self.policy, static=self.static
                    )
            self.assertTrue(old_v9.is_file())
            self.assertFalse(pending.exists())

    def test_existing_result_blocks_before_receipt_lookup(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            result_directory = base / "result"
            result_directory.mkdir()
            with mock.patch.object(
                overlay, "_load_overlay_policy", return_value=self.policy
            ), mock.patch.object(
                overlay, "_validate_static_inputs", return_value=self.static
            ), mock.patch.object(
                overlay,
                "_result_paths",
                return_value=(
                    result_directory,
                    result_directory / "quality_audit_receipt.json",
                    base / ".result.tmp",
                ),
            ), mock.patch.object(
                overlay, "_load_and_validate_pending_receipt"
            ) as load_receipt:
                with self.assertRaisesRegex(
                    overlay.QualityAuditAuthorizationError, "resume is forbidden"
                ):
                    overlay.run_quality_audit_once()
            load_receipt.assert_not_called()

    def test_post_consumption_wrapper_failure_publishes_terminal_receipt(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            pending = base / "quality_authorization.json"
            payload = self._receipt_payload()
            raw = self._write_receipt(pending, payload)
            receipt = overlay.VerifiedQualityAuditReceipt(
                path=pending,
                raw=raw,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                receipt_id=str(payload["canonical_self_hash"]),
                payload=payload,
            )
            result_directory = base / "result"
            result_path = result_directory / "quality_audit_receipt.json"
            temporary = base / ".result.tmp"
            secret = "consumed-wrapper-secret"
            with mock.patch.object(
                overlay, "_load_overlay_policy", return_value=self.policy
            ), mock.patch.object(
                overlay, "_validate_static_inputs", return_value=self.static
            ), mock.patch.object(
                overlay,
                "_result_paths",
                return_value=(result_directory, result_path, temporary),
            ), mock.patch.object(
                overlay,
                "_load_and_validate_pending_receipt",
                return_value=receipt,
            ), mock.patch.object(
                overlay,
                "_validate_consumed_receipt",
                side_effect=RuntimeError(secret),
            ):
                terminal = overlay.run_quality_audit_once()
            self.assertEqual(
                terminal["status"],
                "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
            )
            self.assertEqual(
                terminal["failure_stage"], "consumed_receipt_revalidation"
            )
            self.assertEqual(
                terminal["exception_message_sha256"],
                hashlib.sha256(secret.encode("utf-8")).hexdigest(),
            )
            self.assertNotIn(secret, json.dumps(terminal, ensure_ascii=False))
            self.assertTrue(result_path.is_file())
            self.assertFalse(pending.exists())
            self.assertEqual(
                len(list(base.glob("quality_authorization.consumed.*.json"))),
                1,
            )

    def test_concurrent_non_owner_cannot_publish_terminal_receipt(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            pending = base / "quality_authorization.json"
            payload = self._receipt_payload()
            raw = self._write_receipt(pending, payload)
            receipt = overlay.VerifiedQualityAuditReceipt(
                path=pending,
                raw=raw,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                receipt_id=str(payload["canonical_self_hash"]),
                payload=payload,
            )
            consumed = pending.with_name(
                f"{pending.stem}.consumed.{receipt.sha256}.json"
            )
            result_directory = base / "result"
            result_path = result_directory / "quality_audit_receipt.json"
            temporary = base / ".result.tmp"

            def lose_atomic_rename_ownership(
                _receipt: overlay.VerifiedQualityAuditReceipt,
            ) -> dict[str, object]:
                pending.replace(consumed)
                raise overlay.QualityAuditAuthorizationError(
                    "Quality-audit receipt could not be consumed"
                )

            with mock.patch.object(
                overlay, "_load_overlay_policy", return_value=self.policy
            ), mock.patch.object(
                overlay, "_validate_static_inputs", return_value=self.static
            ), mock.patch.object(
                overlay,
                "_result_paths",
                return_value=(result_directory, result_path, temporary),
            ), mock.patch.object(
                overlay,
                "_load_and_validate_pending_receipt",
                return_value=receipt,
            ), mock.patch.object(
                overlay,
                "_consume_receipt",
                side_effect=lose_atomic_rename_ownership,
            ), mock.patch.object(
                overlay, "_publish_wrapper_terminal_failure"
            ) as publish_terminal:
                with self.assertRaisesRegex(
                    overlay.QualityAuditAuthorizationError,
                    "could not be consumed",
                ):
                    overlay.run_quality_audit_once()

            publish_terminal.assert_not_called()
            self.assertTrue(consumed.is_file())
            self.assertFalse(pending.exists())
            self.assertFalse(result_directory.exists())
            self.assertFalse(result_path.exists())
            self.assertFalse(temporary.exists())

    def test_terminal_publisher_never_accepts_a_different_existing_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            result_directory = base / "result"
            result_directory.mkdir()
            result_path = result_directory / "quality_audit_receipt.json"
            temporary = base / ".result.tmp"
            artifact = {
                "status": "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
                "canonical_self_hash": "e" * 64,
            }
            result_path.write_bytes(b"different\n")
            with self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError,
                "differs from terminal failure",
            ):
                overlay._publish_wrapper_terminal_failure(
                    artifact=artifact,
                    result_directory=result_directory,
                    result_path=result_path,
                    temporary_path=temporary,
                )
            result_path.write_bytes(overlay._result_payload(artifact))
            overlay._publish_wrapper_terminal_failure(
                artifact=artifact,
                result_directory=result_directory,
                result_path=result_path,
                temporary_path=temporary,
            )
            self.assertEqual(
                result_path.read_bytes(), overlay._result_payload(artifact)
            )

    def test_public_sequence_consumes_before_frozen_body(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            pending = base / "quality_authorization.json"
            payload = self._receipt_payload()
            raw = self._write_receipt(pending, payload)
            receipt = overlay.VerifiedQualityAuditReceipt(
                path=pending,
                raw=raw,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                receipt_id=str(payload["canonical_self_hash"]),
                payload=payload,
            )
            result_directory = base / "result"
            result_path = result_directory / "quality_audit_receipt.json"
            temporary = base / ".result.tmp"
            events: list[str] = []

            def validate_consumed(**_kwargs: object) -> None:
                events.append("consumed_validated")
                self.assertFalse(pending.exists())
                self.assertFalse(result_directory.exists())

            def execute(_policy: object, _execution: object) -> dict[str, object]:
                events.append("body_called")
                self.assertFalse(pending.exists())
                self.assertTrue(any(base.glob("quality_authorization.consumed.*.json")))
                return {
                    "status": "PASS",
                    "formal_500_by_4_generated": False,
                    "training_started": False,
                }

            artifact = {
                "status": "PASS",
                "canonical_self_hash": "e" * 64,
            }
            with mock.patch.object(
                overlay, "_load_overlay_policy", return_value=self.policy
            ), mock.patch.object(
                overlay, "_validate_static_inputs", return_value=self.static
            ), mock.patch.object(
                overlay,
                "_result_paths",
                return_value=(result_directory, result_path, temporary),
            ), mock.patch.object(
                overlay,
                "_load_and_validate_pending_receipt",
                return_value=receipt,
            ), mock.patch.object(
                overlay,
                "_validate_consumed_receipt",
                side_effect=validate_consumed,
            ), mock.patch.object(
                execution_adapter,
                "build_consumed_execution",
                return_value=self._execution(),
            ), mock.patch.object(
                overlay, "_execute_frozen_body", side_effect=execute
            ), mock.patch.object(
                overlay, "_build_result_artifact", return_value=artifact
            ):
                observed = overlay.run_quality_audit_once()
            self.assertEqual(observed, artifact)
            self.assertEqual(events, ["consumed_validated", "body_called"])
            self.assertTrue(result_path.is_file())
            self.assertFalse(temporary.exists())

    def test_frozen_body_is_called_with_unchanged_base_policy(self) -> None:
        quality_policy = self.static["quality_policy"]

        execution = self._execution()

        def body(
            *,
            policy: object,
            execution: object,
            state: dict[str, str],
        ) -> dict[str, object]:
            self.assertIs(policy, quality_policy)
            self.assertIs(execution, execution_context)
            self.assertEqual(state["stage"], "authorized_overlay_entry")
            return {
                "status": "PASS",
                "formal_500_by_4_generated": False,
                "training_started": False,
            }

        execution_context = execution
        with mock.patch.object(
            execution_adapter,
            "run_authorized_formal_quality_audit",
            side_effect=body,
        ):
            result = overlay._execute_frozen_body(
                quality_policy, execution_context
            )
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(quality_policy["authorization"]["quality_audit_run"])
        self.assertFalse(quality_policy["authorization"]["metric_generation"])

    def test_frozen_body_exception_is_hash_only_and_classified(self) -> None:
        secret = "row-label-secret"
        execution = self._execution()
        with mock.patch.object(
            execution_adapter,
            "run_authorized_formal_quality_audit",
            side_effect=frozen_runner.DatasetGateFailure(secret),
        ):
            result = overlay._execute_frozen_body(
                self.static["quality_policy"], execution
            )
        self.assertEqual(result["status"], "DATASET_INVALIDATED")
        self.assertEqual(result["row_level_labels_returned"], 0)
        self.assertNotIn(secret, json.dumps(result, ensure_ascii=False))
        self.assertEqual(
            result["exception_message_sha256"],
            hashlib.sha256(secret.encode("utf-8")).hexdigest(),
        )

        with mock.patch.object(
            execution_adapter,
            "run_authorized_formal_quality_audit",
            side_effect=frozen_runner.AuditorExecutionFailure(secret),
        ):
            result = overlay._execute_frozen_body(
                self.static["quality_policy"], execution
            )
        self.assertEqual(
            result["status"], "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION"
        )
        self.assertFalse(result["cleanup_required"])
        self.assertNotIn(secret, json.dumps(result, ensure_ascii=False))

        authorization_error = (
            frozen_runner.structure_aggregator.QualityStructureAggregationError(
                "Formal quality audit remains unauthorized"
            )
        )
        with mock.patch.object(
            execution_adapter,
            "run_authorized_formal_quality_audit",
            side_effect=authorization_error,
        ):
            result = overlay._execute_frozen_body(
                self.static["quality_policy"], execution
            )
        self.assertEqual(
            result["status"],
            "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
        )
        self.assertFalse(result["cleanup_required"])

        dataset_error = (
            frozen_runner.structure_aggregator.QualityStructureAggregationError(
                "Structural world universe drift"
            )
        )
        with mock.patch.object(
            execution_adapter,
            "run_authorized_formal_quality_audit",
            side_effect=dataset_error,
        ):
            result = overlay._execute_frozen_body(
                self.static["quality_policy"], execution
            )
        self.assertEqual(result["status"], "DATASET_INVALIDATED")
        self.assertTrue(result["cleanup_required"])

    def test_result_artifact_rejects_formal_or_training_claim_drift(self) -> None:
        payload = self._receipt_payload()
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        receipt = overlay.VerifiedQualityAuditReceipt(
            path=ROOT / "private_custody" / "fixture.json",
            raw=raw,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            receipt_id=str(payload["canonical_self_hash"]),
            payload=payload,
        )
        consumed = {
            "path": "private_custody/fixture.consumed." + receipt.sha256 + ".json",
            "size_bytes": len(raw),
            "sha256": receipt.sha256,
        }
        for key in ("formal_500_by_4_generated", "training_started"):
            result = self._early_invalidation_result()
            result[key] = True
            result["canonical_self_hash"] = overlay._canonical_self_hash(result)
            with self.subTest(key=key), mock.patch.object(
                overlay,
                "_git_identity",
                return_value=(self.git_commit, self.git_tree),
            ), self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "boundary drift"
            ):
                overlay._build_result_artifact(
                    policy=self.policy,
                    static=self.static,
                    receipt=receipt,
                    consumed_file=consumed,
                    audit_result=result,
                )

    def test_result_artifact_rejects_minimal_pass_and_unregistered_payload(self) -> None:
        payload = self._receipt_payload()
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        receipt = overlay.VerifiedQualityAuditReceipt(
            path=ROOT / "private_custody" / "fixture.json",
            raw=raw,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            receipt_id=str(payload["canonical_self_hash"]),
            payload=payload,
        )
        consumed = {
            "path": "private_custody/fixture.consumed." + receipt.sha256 + ".json",
            "size_bytes": len(raw),
            "sha256": receipt.sha256,
        }
        base = {
            "status": "PASS",
            "formal_500_by_4_generated": False,
            "training_started": False,
            "audit_a_b_truth_remained_sealed": True,
        }
        with mock.patch.object(
            overlay,
            "_git_identity",
            return_value=(self.git_commit, self.git_tree),
        ), self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "schema drift"
        ):
            overlay._build_result_artifact(
                policy=self.policy,
                static=self.static,
                receipt=receipt,
                consumed_file=consumed,
                audit_result=base,
            )
        leaked = self._early_invalidation_result()
        leaked["predictions_by_pair"] = [0.1, 0.9]
        with mock.patch.object(
            overlay,
            "_git_identity",
            return_value=(self.git_commit, self.git_tree),
        ), self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "schema drift"
        ):
            overlay._build_result_artifact(
                policy=self.policy,
                static=self.static,
                receipt=receipt,
                consumed_file=consumed,
                audit_result=leaked,
            )

    def test_result_artifact_accepts_exact_early_invalidation_and_rehashes_inner_receipts(
        self,
    ) -> None:
        payload = self._receipt_payload()
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        receipt = overlay.VerifiedQualityAuditReceipt(
            path=ROOT / "private_custody" / "fixture.json",
            raw=raw,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            receipt_id=str(payload["canonical_self_hash"]),
            payload=payload,
        )
        consumed = {
            "path": "private_custody/fixture.consumed." + receipt.sha256 + ".json",
            "size_bytes": len(raw),
            "sha256": receipt.sha256,
        }
        result = self._early_invalidation_result()
        with mock.patch.object(
            overlay,
            "_git_identity",
            return_value=(self.git_commit, self.git_tree),
        ):
            artifact = overlay._build_result_artifact(
                policy=self.policy,
                static=self.static,
                receipt=receipt,
                consumed_file=consumed,
                audit_result=result,
            )
        self.assertEqual(artifact["status"], "DATASET_INVALIDATED")
        changed = copy.deepcopy(result)
        changed["structure"]["canonical_self_hash"] = "0" * 64
        changed["canonical_self_hash"] = overlay._canonical_self_hash(changed)
        with self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError,
            "Structure receipt self-hash drift",
        ):
            overlay._validate_audit_result(
                audit_result=changed,
                policy=self.policy,
                static=self.static,
                receipt=receipt,
            )

    def test_atomic_result_publication_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            result = base / "result.json"
            temporary = base / ".result.tmp"
            artifact = {
                "status": "PASS",
                "canonical_self_hash": "f" * 64,
            }
            overlay._publish_result(
                artifact=artifact,
                result_path=result,
                temporary_path=temporary,
            )
            self.assertEqual(
                json.loads(result.read_text(encoding="utf-8")), artifact
            )
            self.assertFalse(temporary.exists())
            with self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "already exists"
            ):
                overlay._publish_result(
                    artifact=artifact,
                    result_path=result,
                    temporary_path=temporary,
                )

    def test_result_publication_never_overwrites_a_concurrent_target(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            result = base / "result.json"
            temporary = base / ".result.tmp"
            artifact = {
                "status": "PASS",
                "canonical_self_hash": "f" * 64,
            }
            with mock.patch.object(
                overlay.os, "link", side_effect=FileExistsError("fixture race")
            ), self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "could not be published"
            ):
                overlay._publish_result(
                    artifact=artifact,
                    result_path=result,
                    temporary_path=temporary,
                )
            self.assertFalse(result.exists())
            self.assertFalse(temporary.exists())

    def test_wrapper_does_not_read_row_truth_or_generate_authorization(self) -> None:
        source = Path(overlay.__file__).read_text(encoding="utf-8")
        self.assertNotIn("qrels", source)
        declared_source = inspect.getsource(overlay._declared_data_bindings)
        self.assertIn("split_manifest.json", declared_source)
        self.assertNotIn("_read_pinned_truth_csv", declared_source)
        self.assertNotIn("csv.DictReader", declared_source)
        tree = ast.parse(source)
        function_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        forbidden = {
            name
            for name in function_names
            if name.startswith("generate_")
            or name.startswith("create_authorization")
            or name.startswith("issue_receipt")
        }
        self.assertEqual(forbidden, set())


if __name__ == "__main__":
    unittest.main()
