from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_formal_500x4_quality_attempt1_v9_4_1 as quality
import step28_v13_v1_13_formal_500x4_sealed_literal_scan_attempt1_v9_4_1 as sealed_scan
import step28_v13_v1_13_quality_text_probe_views_v9 as text_views


class Formal500x4QualityAttempt1V941Contracts(unittest.TestCase):
    def test_policy_targets_only_the_completed_formal_root(self) -> None:
        policy = quality.read_json(quality.POLICY_PATH)
        self.assertEqual(
            policy["splits"]["world_counts"],
            {"train": 500, "development": 500, "audit_a": 500, "audit_b": 500},
        )
        self.assertEqual(
            policy["dataset_root"],
            "reports/step28_synthetic_chinese_dataset/"
            "v9_4_1_formal_500x4_attempt1_20260829",
        )
        self.assertEqual(
            policy["claim_boundary"],
            "FORMAL_500X4_ROOT_QUALITY_ONLY_NO_AUDIT_TRUTH_NO_MODEL_TRAINING",
        )
        self.assertEqual(
            policy["pins"]["root_manifest"]["canonical_self_hash"],
            "3cc7505488c9f47558e9756254ef4c77cccb2a9ccc0c2ba70c9c3531383e1fe6",
        )
        self.assertEqual(
            policy["splits"]["non_supervision_private_sealed_scan"],
            ["audit_a", "audit_b"],
        )
        self.assertEqual(policy["truth_access"]["audit_a_truth_reads"], 0)
        self.assertEqual(policy["truth_access"]["audit_b_truth_reads"], 0)

    def test_success_qualifies_data_but_does_not_authorize_model_training(self) -> None:
        source = Path(quality.__file__).read_text(encoding="utf-8")
        result_block = source.split('"formal_root_quality_passed": passed', 1)[1]
        self.assertIn('"training_qualified": passed', result_block)
        self.assertIn('"m0_m1_m2_m3_training_authorized": False', result_block)
        self.assertIn(
            "PASSED_FORMAL_500X4_ROOT_QUALITY_TRAINING_QUALIFIED",
            source,
        )

    def test_dataset_gate_and_auditor_failure_have_distinct_terminals(self) -> None:
        gate = quality.FormalRootDatasetGateError(
            "train_truth_exact_reconciliation", ValueError("private drift")
        )
        dataset_terminal = quality.classified_failure_terminal(
            exc=gate,
            stage="quality_audit",
            temporary_matrices_deleted=True,
        )
        self.assertEqual(
            dataset_terminal["status"],
            "DATASET_INVALIDATED_PENDING_DOCUMENTATION_AND_CLEANUP",
        )
        self.assertEqual(
            dataset_terminal["failure_gate"], "train_truth_exact_reconciliation"
        )
        self.assertTrue(dataset_terminal["dataset_cleanup_required"])

        mechanical_terminal = quality.classified_failure_terminal(
            exc=RuntimeError("out of memory"),
            stage="quality_audit",
            temporary_matrices_deleted=False,
        )
        self.assertEqual(
            mechanical_terminal["status"],
            "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
        )
        self.assertIsNone(mechanical_terminal["failure_gate"])
        self.assertFalse(mechanical_terminal["dataset_cleanup_required"])
        self.assertFalse(mechanical_terminal["temporary_matrices_deleted"])

    def test_dataset_gate_wrapper_does_not_relabel_generic_code_errors(self) -> None:
        def contract_failure() -> None:
            raise quality.FormalRootDatasetEvidenceError("schema drift")

        with self.assertRaises(quality.FormalRootDatasetGateError):
            quality.evaluate_dataset_gate("public_schema", contract_failure)

        def ambiguous_auditor_failure() -> None:
            raise quality.FormalRootQualityAttempt1Error("ambiguous drift")

        with self.assertRaisesRegex(
            quality.FormalRootQualityAttempt1Error, "ambiguous drift"
        ):
            quality.evaluate_dataset_gate("public_schema", ambiguous_auditor_failure)

        def generic_failure() -> None:
            raise ValueError("implementation bug")

        with self.assertRaisesRegex(ValueError, "implementation bug"):
            quality.evaluate_dataset_gate("public_schema", generic_failure)

    def test_completed_early_false_gate_is_immediately_monotonic(self) -> None:
        receipt = {"passed": False, "row_count": 17}
        with self.assertRaises(quality.FormalRootDatasetGateError) as captured:
            quality.require_completed_dataset_gate(
                gate="visible_text_readability",
                passed=receipt["passed"],
                receipt=receipt,
            )
        terminal = quality.classified_failure_terminal(
            exc=captured.exception,
            stage="quality_audit",
            temporary_matrices_deleted=True,
        )
        self.assertEqual(
            terminal["status"],
            "DATASET_INVALIDATED_PENDING_DOCUMENTATION_AND_CLEANUP",
        )
        self.assertEqual(terminal["failure_gate"], "visible_text_readability")
        self.assertEqual(
            terminal["failure_evidence"]["receipt_canonical_sha256"],
            quality.canonical_sha256(receipt),
        )

    def test_formal_run_cleans_temporary_matrices_before_success_publication(self) -> None:
        source = Path(quality.__file__).read_text(encoding="utf-8")
        function = source.split("def formal_run(", 1)[1].split("\ndef main(", 1)[0]
        cleanup_offset = function.index("cleanup_formal_temporary_root(temp_root)")
        publication_offset = function.index('stage = "result_publication"')
        self.assertLess(cleanup_offset, publication_offset)

    def test_formal_run_publishes_only_after_cleanup_and_classifies_failure(self) -> None:
        policy = {
            "status": "FROZEN_IMPLEMENTATION_NO_FORMAL_RUN_YET",
            "dataset_root": "public-root",
            "private_root": "private-root",
            "output_root": (
                "reports/step28_synthetic_chinese_dataset/"
                "v9_4_1_formal_500x4_quality_attempt1_20260829"
            ),
            "temporary_root": (
                "reports/step28_synthetic_chinese_dataset/"
                ".v9_4_1_formal_quality_attempt1_20260829.building"
            ),
            "consumption_path": "private/consumed.json",
        }
        consumption = {"canonical_self_hash": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with (
                patch.object(quality, "ROOT", root),
                patch.object(quality, "verify_formal_authorization", return_value={}),
                patch.object(
                    quality,
                    "consume_authorization",
                    return_value=(consumption, "b" * 64),
                ),
                patch.object(quality, "run_audit", return_value={"status": "PASS"}),
                patch("builtins.print"),
            ):
                quality.formal_run(policy)
            output = root / policy["output_root"] / "quality_result.json"
            temporary = root / policy["temporary_root"]
            self.assertTrue(output.is_file())
            self.assertFalse(temporary.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            failure = quality.FormalRootDatasetGateError(
                "public_schema", quality.FormalRootQualityAttempt1Error("drift")
            )
            with (
                patch.object(quality, "ROOT", root),
                patch.object(quality, "verify_formal_authorization", return_value={}),
                patch.object(
                    quality,
                    "consume_authorization",
                    return_value=(consumption, "b" * 64),
                ),
                patch.object(quality, "run_audit", side_effect=failure),
                patch.object(
                    quality,
                    "formal_failure_lineage",
                    return_value={"canonical_self_hash": "c" * 64},
                ),
            ):
                with self.assertRaises(quality.FormalRootDatasetGateError):
                    quality.formal_run(policy)
            terminal_path = root / policy["output_root"] / "terminal.json"
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            self.assertEqual(
                terminal["status"],
                "DATASET_INVALIDATED_PENDING_DOCUMENTATION_AND_CLEANUP",
            )
            self.assertEqual(terminal["failure_gate"], "public_schema")
            self.assertTrue(terminal["temporary_matrices_deleted"])
            self.assertFalse((root / policy["temporary_root"]).exists())

    def test_publication_failure_cannot_erase_computed_dataset_invalidation(
        self,
    ) -> None:
        policy = {
            "status": "FROZEN_IMPLEMENTATION_NO_FORMAL_RUN_YET",
            "dataset_root": "public-root",
            "private_root": "private-root",
            "output_root": (
                "reports/step28_synthetic_chinese_dataset/"
                "v9_4_1_formal_500x4_quality_attempt1_20260829"
            ),
            "temporary_root": (
                "reports/step28_synthetic_chinese_dataset/"
                ".v9_4_1_formal_quality_attempt1_20260829.building"
            ),
            "consumption_path": "private/consumed.json",
        }
        consumption = {"canonical_self_hash": "a" * 64}
        failed_result = {
            "status": "DATASET_INVALIDATED_PENDING_DOCUMENTATION_AND_CLEANUP",
            "formal_root_quality_passed": False,
            "hard_gates": {"failed_gate": False, "passed_gate": True},
        }
        failed_result["canonical_self_hash"] = quality.canonical_sha256(
            failed_result
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            real_write = quality.write_json_exclusive

            def fail_only_quality_result(path: Path, value: object) -> None:
                if path.name == "quality_result.json":
                    raise OSError("simulated publication failure")
                real_write(path, value)

            with (
                patch.object(quality, "ROOT", root),
                patch.object(quality, "verify_formal_authorization", return_value={}),
                patch.object(
                    quality,
                    "consume_authorization",
                    return_value=(consumption, "b" * 64),
                ),
                patch.object(quality, "run_audit", return_value=failed_result),
                patch.object(
                    quality,
                    "write_json_exclusive",
                    side_effect=fail_only_quality_result,
                ),
                patch.object(
                    quality,
                    "formal_failure_lineage",
                    return_value={"canonical_self_hash": "c" * 64},
                ),
            ):
                with self.assertRaisesRegex(OSError, "publication failure"):
                    quality.formal_run(policy)
            terminal = json.loads(
                (
                    root
                    / policy["output_root"]
                    / "terminal.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                terminal["status"],
                "DATASET_INVALIDATED_PENDING_DOCUMENTATION_AND_CLEANUP",
            )
            self.assertTrue(terminal["complete_quality_result_computed"])
            self.assertEqual(terminal["failed_hard_gates"], ["failed_gate"])
            self.assertEqual(
                terminal["failure_gate"], "complete_quality_result_hard_gates"
            )

    def test_cleanup_failure_cannot_publish_a_scientific_terminal(self) -> None:
        policy = {
            "status": "FROZEN_IMPLEMENTATION_NO_FORMAL_RUN_YET",
            "dataset_root": "public-root",
            "private_root": "private-root",
            "output_root": (
                "reports/step28_synthetic_chinese_dataset/"
                "v9_4_1_formal_500x4_quality_attempt1_20260829"
            ),
            "temporary_root": (
                "reports/step28_synthetic_chinese_dataset/"
                ".v9_4_1_formal_quality_attempt1_20260829.building"
            ),
            "consumption_path": "private/consumed.json",
        }
        consumption = {"canonical_self_hash": "a" * 64}
        failed_result = {
            "status": "DATASET_INVALIDATED_PENDING_DOCUMENTATION_AND_CLEANUP",
            "formal_root_quality_passed": False,
            "hard_gates": {"early_gate": False, "passed_gate": True},
        }
        failed_result["canonical_self_hash"] = quality.canonical_sha256(
            failed_result
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with (
                patch.object(quality, "ROOT", root),
                patch.object(quality, "verify_formal_authorization", return_value={}),
                patch.object(
                    quality,
                    "consume_authorization",
                    return_value=(consumption, "b" * 64),
                ),
                patch.object(quality, "run_audit", return_value=failed_result),
                patch.object(
                    quality,
                    "cleanup_formal_temporary_root",
                    side_effect=OSError("simulated cleanup failure"),
                ),
                patch.object(
                    quality,
                    "formal_failure_lineage",
                    return_value={"canonical_self_hash": "c" * 64},
                ),
            ):
                with self.assertRaisesRegex(OSError, "cleanup failure"):
                    quality.formal_run(policy)
            output = root / policy["output_root"]
            pending = json.loads(
                (output / "cleanup_pending.json").read_text(encoding="utf-8")
            )
            self.assertEqual(pending["cleanup_scope"], "temporary_matrix_root")
            self.assertEqual(pending["cleanup_exception_type"], "OSError")
            self.assertTrue(pending["complete_failed_quality_result_computed"])
            self.assertEqual(pending["failed_hard_gates"], ["early_gate"])
            self.assertFalse((output / "terminal.json").exists())
            self.assertFalse((output / "quality_result.json").exists())

    def test_cleanup_pending_preserves_early_dataset_gate_evidence(self) -> None:
        policy = {
            "status": "FROZEN_IMPLEMENTATION_NO_FORMAL_RUN_YET",
            "dataset_root": "public-root",
            "private_root": "private-root",
            "output_root": (
                "reports/step28_synthetic_chinese_dataset/"
                "v9_4_1_formal_500x4_quality_attempt1_20260829"
            ),
            "temporary_root": (
                "reports/step28_synthetic_chinese_dataset/"
                ".v9_4_1_formal_quality_attempt1_20260829.building"
            ),
            "consumption_path": "private/consumed.json",
        }
        consumption = {"canonical_self_hash": "a" * 64}
        evidence = {
            "completed_gate": "visible_text_readability",
            "receipt_canonical_sha256": "d" * 64,
        }
        evidence["canonical_self_hash"] = quality.canonical_sha256(evidence)
        failure = quality.FormalRootDatasetGateError(
            "visible_text_readability",
            quality.FormalRootDatasetEvidenceError("completed gate false"),
            evidence=evidence,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with (
                patch.object(quality, "ROOT", root),
                patch.object(quality, "verify_formal_authorization", return_value={}),
                patch.object(
                    quality,
                    "consume_authorization",
                    return_value=(consumption, "b" * 64),
                ),
                patch.object(quality, "run_audit", side_effect=failure),
                patch.object(
                    quality,
                    "cleanup_formal_temporary_root",
                    side_effect=OSError("simulated cleanup failure"),
                ),
                patch.object(
                    quality,
                    "formal_failure_lineage",
                    return_value={"canonical_self_hash": "c" * 64},
                ),
            ):
                with self.assertRaises(quality.FormalRootDatasetGateError):
                    quality.formal_run(policy)
            pending = json.loads(
                (
                    root
                    / policy["output_root"]
                    / "cleanup_pending.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                pending["primary_dataset_failure_gate"],
                "visible_text_readability",
            )
            self.assertEqual(
                pending["primary_dataset_failure_cause_type"],
                "FormalRootDatasetEvidenceError",
            )
            self.assertEqual(
                pending["primary_dataset_failure_evidence"], evidence
            )
            self.assertFalse((root / policy["output_root"] / "terminal.json").exists())

    def test_failed_output_cleanup_failure_emits_recoverable_nonterminal(self) -> None:
        policy = {
            "status": "FROZEN_IMPLEMENTATION_NO_FORMAL_RUN_YET",
            "dataset_root": "public-root",
            "private_root": "private-root",
            "output_root": (
                "reports/step28_synthetic_chinese_dataset/"
                "v9_4_1_formal_500x4_quality_attempt1_20260829"
            ),
            "temporary_root": (
                "reports/step28_synthetic_chinese_dataset/"
                ".v9_4_1_formal_quality_attempt1_20260829.building"
            ),
            "consumption_path": "private/consumed.json",
        }
        consumption = {"canonical_self_hash": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with (
                patch.object(quality, "ROOT", root),
                patch.object(quality, "verify_formal_authorization", return_value={}),
                patch.object(
                    quality,
                    "consume_authorization",
                    return_value=(consumption, "b" * 64),
                ),
                patch.object(
                    quality,
                    "run_audit",
                    side_effect=RuntimeError("simulated auditor failure"),
                ),
                patch.object(
                    quality,
                    "cleanup_failed_output_root",
                    side_effect=OSError("simulated output cleanup failure"),
                ),
                patch.object(
                    quality,
                    "formal_failure_lineage",
                    return_value={"canonical_self_hash": "c" * 64},
                ),
            ):
                with self.assertRaisesRegex(OSError, "output cleanup failure"):
                    quality.formal_run(policy)
            output = root / policy["output_root"]
            pending = json.loads(
                (output / "cleanup_pending.json").read_text(encoding="utf-8")
            )
            self.assertEqual(pending["cleanup_scope"], "failed_output_root")
            self.assertTrue(pending["temporary_matrices_deleted"])
            self.assertEqual(pending["primary_exception_type"], "RuntimeError")
            self.assertEqual(pending["cleanup_exception_type"], "OSError")
            self.assertFalse((output / "terminal.json").exists())

    def test_consumption_has_no_fallible_post_write_hash_read(self) -> None:
        source = Path(quality.__file__).read_text(encoding="utf-8")
        function = source.split("def consume_authorization(", 1)[1].split(
            "\ndef grouped(", 1
        )[0]
        self.assertIn("marker_sha256 = hashlib.sha256(marker_bytes)", function)
        self.assertNotIn("sha256_file(path)", function)

    def test_policy_freezes_complete_c_amendment_family(self) -> None:
        policy = quality.verify_policy()
        family = policy["text_probe_family"]
        self.assertEqual(tuple(family["view_names"]), text_views.VIEW_ORDER)
        self.assertEqual(tuple(family["feature_widths"]), text_views.EXPECTED_WIDTHS)
        self.assertEqual(sum(family["feature_widths"]), 346)
        self.assertEqual(family["total_model_count"], 14)
        self.assertEqual(policy["bootstrap"]["replicates"], 9999)
        self.assertEqual(policy["splits"]["eligible_pairs_per_world"], 372)
        self.assertEqual(policy["splits"]["positive_pairs_per_world"], 20)

    def test_rehashed_policy_cannot_change_models_gates_or_bootstrap(self) -> None:
        original = quality.read_json(quality.POLICY_PATH)
        mutations = (
            ("probe_models", "logistic_l2", "C", 2.0),
            (
                "gates",
                "maximum_model_family_symmetric_roc_auc",
                None,
                0.99,
            ),
            ("bootstrap", "replicates", None, 100),
        )
        for section, name, nested_name, value in mutations:
            with self.subTest(section=section, name=name):
                changed = copy.deepcopy(original)
                if nested_name is None:
                    changed[section][name] = value
                else:
                    changed[section][name][nested_name] = value
                changed.pop("canonical_self_hash")
                changed["canonical_self_hash"] = quality.canonical_sha256(changed)
                with patch.object(quality, "read_json", return_value=changed):
                    with self.assertRaisesRegex(
                        quality.FormalRootQualityAttempt1Error,
                        "Frozen model, gate, bootstrap",
                    ):
                        quality.verify_policy()

    def test_policy_freezes_all_four_split_public_structure_checks(self) -> None:
        policy = quality.verify_policy()
        self.assertEqual(
            policy["four_split_public_structure"],
            {
                "exact_schema_order_for_all_eight_split_payloads": True,
                "world_schedule_and_seller_universe_exact": True,
                "item_replay_redacted_keysets_exact": True,
                "profile_seller_and_item_count_joins_exact": True,
                "complete_pair_graph_and_identity33_keysets_exact": True,
                "identity33_values_finite": True,
                "model_profile_visible_strings_in_all_visible_text_scans": True,
            },
        )
        source = Path(quality.__file__).read_text(encoding="utf-8")
        self.assertIn("def four_split_public_structure_audit(", source)
        self.assertIn('"four_split_public_structure"', source)
        self.assertIn('"four_split_public_row_schemas_and_joins"', source)

    def test_policy_self_hash_is_canonical(self) -> None:
        path = quality.POLICY_PATH
        policy = json.loads(path.read_text(encoding="utf-8"))
        claimed = policy.pop("canonical_self_hash")
        observed = hashlib.sha256(quality.canonical_bytes(policy)).hexdigest()
        self.assertEqual(claimed, observed)

    def test_identity_positive_control_cannot_open_private_truth(self) -> None:
        parameters = quality.identity_positive_control.__code__.co_varnames[
            : quality.identity_positive_control.__code__.co_argcount
            + quality.identity_positive_control.__code__.co_kwonlyargcount
        ]
        self.assertNotIn("private_root", parameters)
        source = Path(quality.__file__).read_text(encoding="utf-8")
        function = source.split("def identity_positive_control(", 1)[1].split(
            "\ndef exact_v941_formal_public_replay(", 1
        )[0]
        self.assertNotIn("pair_labels.csv", function)
        self.assertIn("truth_indexes", function)

    def test_counterfactual_second_replay_mutation_is_rejected(self) -> None:
        first = {
            "counterfactual_redacted": [{"x": 1}],
            "counterfactual_profiles": [{"x": 2}],
            "original_profile_provenance": {"x": 2},
            "counterfactual_profile_provenance": {"x": 2},
            "profile_lineage_difference_receipt": {"x": 2},
            "world_lineage_alignment_receipt": {"x": 2},
            "original_parsed": [{"x": 3}],
            "original_history": [{"x": 4}],
            "original_identity33": [{"x": 5}],
            "original_parser_audit": {"x": 6},
            "counterfactual_parsed": [{"x": 3}],
            "counterfactual_history": [{"x": 4}],
            "counterfactual_identity33": [{"x": 5}],
            "counterfactual_parser_audit": {"x": 6},
            "derangement_mapping_sha256": "a" * 64,
            "excluded_pair_uids": ("p",),
            "control_type_by_pair_uid": {"p": "exact_title_clone_negative"},
            "original_path_alignment": {"x": 8},
            "counterfactual_path_alignment": {"x": 8},
        }
        quality.assert_counterfactual_replay(first, copy.deepcopy(first))
        changed = copy.deepcopy(first)
        changed["counterfactual_profiles"][0]["x"] = 7
        with self.assertRaisesRegex(
            quality.FormalRootQualityAttempt1Error,
            "Counterfactual independent replay drift",
        ):
            quality.assert_counterfactual_replay(first, changed)

    def test_exact_public_replay_projects_endpoint_order_and_limits_all_four_splits(
        self,
    ) -> None:
        calls: list[tuple[str, ...]] = []

        def model_worlds(split: str) -> list[quality.ModelSurfaceWorld]:
            return [
                quality.ModelSurfaceWorld(
                    split=split,
                    ordinal=0,
                    world_uid=f"{split}-world-0",
                    seller_uids=("seller-left", "seller-right"),
                    noise_slots=(0, 1),
                ),
                quality.ModelSurfaceWorld(
                    split=split,
                    ordinal=1,
                    world_uid=f"{split}-world-1",
                    seller_uids=("seller-left-1", "seller-right-1"),
                    noise_slots=(1, 0),
                ),
            ]

        def split_from_path(path: Path) -> str:
            return next(
                split
                for split in ("train", "development", "audit_a", "audit_b")
                if split in path.parts
            )

        def read_endpoints(path: Path) -> list[dict[str, str]]:
            split = split_from_path(path)
            # Deliberately preserve the published CSV order that caused attempt 2
            # to fail when the row dictionaries were forwarded without projection.
            return [
                {
                    "canonical_pair_uid": f"{split}-pair-0",
                    "world_uid": f"{split}-world-0",
                    "seller_uid_left": "seller-left",
                    "seller_uid_right": "seller-right",
                }
            ]

        def read_items(path: Path) -> list[dict[str, str]]:
            split = split_from_path(path)
            return [{"world_uid": f"{split}-world-0", "item_uid": "item-0"}]

        def require_exact_replay(**kwargs: object) -> None:
            endpoint_rows = kwargs["public_endpoint_rows"]
            self.assertIsInstance(endpoint_rows, list)
            row = endpoint_rows[0]  # type: ignore[index]
            keys = tuple(row)
            self.assertEqual(
                keys,
                (
                    "world_uid",
                    "canonical_pair_uid",
                    "seller_uid_left",
                    "seller_uid_right",
                ),
            )
            calls.append(keys)

        with (
            patch.object(
                quality.noise_v94,
                "build_noise_signatures",
                return_value=SimpleNamespace(rows=[]),
            ),
            patch.object(
                quality,
                "formal_schedule_bundle",
                return_value=(
                    {},
                    {
                        "train": {"transformed_schedule_sha256": "a" * 64},
                        "development": {
                            "transformed_schedule_sha256": "b" * 64
                        },
                        "audit_a": {"transformed_schedule_sha256": "c" * 64},
                        "audit_b": {"transformed_schedule_sha256": "d" * 64},
                    },
                    SimpleNamespace(),
                    {"canonical_self_hash": "e" * 64},
                ),
            ),
            patch.object(
                quality,
                "scheduled_model_surface_worlds",
                side_effect=model_worlds,
            ),
            patch.object(quality, "read_csv", side_effect=read_endpoints),
            patch.object(quality, "read_jsonl", side_effect=read_items),
            patch.object(
                quality,
                "build_formal_truth_free_world_projection",
                return_value=[],
            ),
            patch.object(
                quality.replay_v94,
                "require_exact_replay",
                side_effect=require_exact_replay,
            ),
        ):
            result = quality.exact_v941_formal_public_replay(
                public_root=ROOT / "unused-public-root",
                time_key=b"t" * 32,
                world_limit=1,
            )

        self.assertEqual(result["world_count"], 4)
        self.assertTrue(result["exact_public_14d_replay_passed"])
        self.assertEqual(len(calls), 4)

    def test_four_split_projection_matches_frozen_train_helper_and_accepts_audit(self) -> None:
        signatures = [dict(row) for row in quality.noise_v94.build_noise_signatures().rows]
        sellers = tuple(f"seller-{slot:02d}" for slot in range(28))
        time_key = b"t" * 32
        train_world = quality.ModelSurfaceWorld(
            split="train",
            ordinal=7,
            world_uid="formal-train-world-007",
            seller_uids=sellers,
            noise_slots=tuple(range(28)),
        )
        expected = quality.prebuild_v94.build_truth_free_world_projection(
            world={
                "split": train_world.split,
                "world_ordinal": train_world.ordinal,
                "world_uid": train_world.world_uid,
                "seller_uids": list(train_world.seller_uids),
                "noise_slot_by_seller_slot": list(train_world.noise_slots),
            },
            noise_signatures=signatures,
            time_key_hex=time_key.hex(),
        )
        observed = quality.build_formal_truth_free_world_projection(
            world=train_world,
            noise_signatures=signatures,
            time_key=time_key,
        )
        self.assertEqual([dict(row) for row in observed], [dict(row) for row in expected])

        audit_world = quality.ModelSurfaceWorld(
            split="audit_a",
            ordinal=7,
            world_uid="formal-audit-a-world-007",
            seller_uids=sellers,
            noise_slots=tuple(range(28)),
        )
        audit_projection = quality.build_formal_truth_free_world_projection(
            world=audit_world,
            noise_signatures=signatures,
            time_key=time_key,
        )
        self.assertEqual(len(audit_projection), 378)

    def test_sealed_literal_helpers_cover_index_keys_and_short_identifiers(self) -> None:
        all_variants, indexed_variants, leaves, keys = sealed_scan.collect_private_strings(
            {
                "occurrence_counts": {"seller_private_01": 2},
                "intended_style": "预定现货",
            }
        )
        self.assertGreaterEqual(leaves, 1)
        self.assertEqual(keys, 1)
        self.assertIn("seller_private_01", indexed_variants)
        self.assertIn("预定现货", all_variants)
        self.assertNotIn("预定现货", indexed_variants)
        self.assertIn("qq", sealed_scan.literal_variants("QQ"))
        self.assertIn("qq", sealed_scan.visible_literal_variants("仅用 QQ 联系"))
        self.assertNotIn("bat", sealed_scan.visible_literal_variants("combat"))

    def test_pretruth_sealed_scanner_never_opens_supervision_files(self) -> None:
        source = Path(sealed_scan.__file__).read_text(encoding="utf-8")
        self.assertNotIn("controller_membership.jsonl", source)
        self.assertNotIn("pair_labels.csv", source)
        self.assertNotIn("qrels.jsonl", source)
        self.assertIn('"controller_membership_parsed": 0', source)
        self.assertIn('"qrels_parsed": 0', source)
        self.assertIn('registers["item"].update(public["item_owner"])', source)
        self.assertIn('registers["pair"].update(', source)

    def test_sealed_public_indexes_recompute_model_visible_identity_activity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed = root / "train" / "observed"
            observed.mkdir(parents=True)

            def write_jsonl(name: str, rows: list[dict[str, object]]) -> None:
                with (observed / name).open(
                    "w", encoding="utf-8", newline="\n"
                ) as stream:
                    for row in rows:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")

            write_jsonl("worlds.jsonl", [{"world_uid": "world-a"}])
            write_jsonl(
                "sellers.jsonl",
                [
                    {"world_uid": "world-a", "seller_uid": "seller-a"},
                    {"world_uid": "world-a", "seller_uid": "seller-b"},
                ],
            )
            write_jsonl(
                "items.jsonl",
                [
                    {
                        "world_uid": "world-a",
                        "seller_uid": "seller-a",
                        "item_uid": "item-a",
                    }
                ],
            )
            with (observed / "complete_model_pair_endpoints.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "canonical_pair_uid",
                        "world_uid",
                        "seller_uid_left",
                        "seller_uid_right",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "canonical_pair_uid": "pair-a",
                        "world_uid": "world-a",
                        "seller_uid_left": "seller-a",
                        "seller_uid_right": "seller-b",
                    }
                )
            feature_names = tuple(f"identity_{index:02d}" for index in range(33))
            with (observed / "identity33_all_pairs.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("canonical_pair_uid", "world_uid", *feature_names),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "canonical_pair_uid": "pair-a",
                        "world_uid": "world-a",
                        **{
                            name: (1.0 if index == 0 else 0.0)
                            for index, name in enumerate(feature_names)
                        },
                    }
                )
            with patch.object(sealed_scan, "PUBLIC_ROOT", root):
                indexes = sealed_scan.public_split_indexes("train")
            self.assertEqual(indexes["identity33_row_counts"]["world-a"], 1)
            self.assertEqual(
                indexes["identity33_active_pair_counts"]["world-a"], 1
            )
            self.assertEqual(set(indexes["item_owner"]), {"item-a"})
            self.assertEqual(
                set(indexes["pair_sellers"]), {("world-a", "pair-a")}
            )

    def test_generation_audit_is_projected_before_feature_construction(self) -> None:
        style = {
            "seller_uid": "seller-a",
            "separator": "，",
            "ending": "。",
            "line_mode": "single",
            "english_tag": "",
            "traditional_variant": False,
            "repeat_punctuation": False,
            "base_style_id": "style-a",
            "perturbed_fields": ["ending", "line_mode"],
            "controller_group_index": 0,
        }
        control = {
            "canonical_pair_uid": "pair-a",
            "control_type": "exact_title_clone_negative",
            "source_item_uid": "item-a",
            "target_item_uid": "item-b",
        }
        raw = {
            "world_uid": "world-a",
            "style_assignments": [style],
            "registered_negative_controls": [control],
            "mechanism_assignments": [
                {"members": ["seller-a", "seller-b"], "mechanism": "private"}
            ],
            "identity33_active_pair_count": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation_audit.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(raw, ensure_ascii=False) + "\n")
            projected = quality.load_model_surface_generation_audit(path)
        self.assertEqual(
            set(projected["world-a"]),
            {"world_uid", "style_assignments", "registered_negative_controls"},
        )
        self.assertNotIn("mechanism_assignments", projected["world-a"])
        self.assertNotIn(
            "controller_group_index",
            projected["world-a"]["style_assignments"][0],
        )
        raw["style_assignments"][0]["controller_group_index"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation_audit.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(raw, ensure_ascii=False) + "\n")
            with self.assertRaisesRegex(
                quality.FormalRootQualityAttempt1Error,
                "private controller index drift",
            ):
                quality.load_model_surface_generation_audit(path)

    def test_model_surface_world_excludes_controller_membership(self) -> None:
        public = SimpleNamespace(
            ordinal=0,
            world_uid="world-a",
            seller_uids=tuple(f"seller-{index:02d}" for index in range(28)),
            noise_slots=tuple(range(28)),
            controller_groups=(tuple(f"seller-{index:02d}" for index in range(28)),),
        )
        with patch.object(quality, "scheduled_worlds", return_value=[public]):
            worlds = quality.scheduled_model_surface_worlds("train")
        self.assertEqual(len(worlds), 1)
        self.assertFalse(hasattr(worlds[0], "controller_groups"))
        self.assertEqual(
            set(worlds[0].__dataclass_fields__),
            {"split", "ordinal", "world_uid", "seller_uids", "noise_slots"},
        )

    def test_actual_constructor_row_keys_cannot_be_replaced_by_input_keys(self) -> None:
        sellers = [f"seller-{index:02d}" for index in range(28)]
        endpoints = [
            {
                "canonical_pair_uid": f"pair-{left:02d}-{right:02d}",
                "world_uid": "world-a",
                "seller_uid_left": sellers[left],
                "seller_uid_right": sellers[right],
            }
            for left in range(28)
            for right in range(left + 1, 28)
        ]
        row_keys = tuple(
            (row["world_uid"], row["canonical_pair_uid"]) for row in endpoints
        )
        fixed = {
            name: np.zeros((378, width), dtype="<f8")
            for name, width in zip(
                text_views.VIEW_ORDER[:3], text_views.EXPECTED_WIDTHS[:3], strict=True
            )
        }
        production = {
            name: np.zeros((378, width), dtype="<f8")
            for name, width in zip(
                text_views.VIEW_ORDER[3:6],
                text_views.EXPECTED_WIDTHS[3:6],
                strict=True,
            )
        }
        with (
            patch.object(
                text_views,
                "_build_fixed_support_views",
                return_value=(fixed, {}, row_keys),
            ),
            patch.object(
                text_views,
                "_build_production_views",
                return_value=(
                    production,
                    {},
                    np.zeros((378, 16), dtype="<f8"),
                    (),
                    tuple(reversed(row_keys)),
                ),
            ),
        ):
            with self.assertRaisesRegex(
                quality.FormalRootQualityAttempt1Error,
                "actual constructor row-key order drift",
            ):
                quality.build_views_with_path_alignment(
                    items=[
                        {
                            "world_uid": "world-a",
                            "seller_uid": seller,
                            "item_uid": f"item-{index:02d}",
                            "title": "",
                            "description": "",
                        }
                        for index, seller in enumerate(sellers)
                    ],
                    profiles=[{"seller_uid": seller} for seller in sellers],
                    endpoints=endpoints,
                )

    def test_style_dose_rejects_counterfactual_presence_drift(self) -> None:
        world = quality.ModelSurfaceWorld(
            split="train",
            ordinal=0,
            world_uid="world-a",
            seller_uids=("seller-a",),
            noise_slots=(0,),
        )
        style = {
            "separator": "，",
            "ending": "。",
            "line_mode": "single",
            "english_tag": "",
            "traditional_variant": False,
            "repeat_punctuation": False,
        }
        with self.assertRaisesRegex(
            quality.FormalRootQualityAttempt1Error,
            "field-presence pattern drift",
        ):
            quality.style_intervention_dose_receipt(
                world=world,
                styles={"seller-a": style},
                source_style={"seller-a": "seller-a"},
                replay_items=[{"seller_uid": "seller-a", "item_uid": "item-a"}],
                original_render={"item-a": ("标题", "描述")},
                counterfactual_render={"item-a": ("", "描述")},
            )

    def test_style_assignment_requires_two_declared_actual_domain_steps(self) -> None:
        template = quality.read_json(quality.TEXT_TEMPLATE_PATH)
        base = template["style_prototypes"][0]
        domains = template["renderer_contract"]["style_factor_domains"]
        style = {
            name: base[name] for name in quality.builder.STYLE_FIELDS
        }
        for name in ("separator", "ending"):
            domain = list(domains[name])
            style[name] = domain[(domain.index(base[name]) + 1) % len(domain)]
        style.update(
            {
                "base_style_id": base["style_id"],
                "perturbed_fields": ["separator", "ending"],
            }
        )
        self.assertEqual(
            quality.validate_style_assignment_contract(
                styles={"seller-a": style},
                seller_uids=("seller-a",),
                template=template,
            ),
            1,
        )
        changed = copy.deepcopy(style)
        changed["perturbed_fields"] = ["separator", "line_mode"]
        with self.assertRaisesRegex(
            quality.FormalRootQualityAttempt1Error,
            "declaration/value mismatch",
        ):
            quality.validate_style_assignment_contract(
                styles={"seller-a": changed},
                seller_uids=("seller-a",),
                template=template,
            )

    def test_read_only_matrix_freeze_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrices = quality.allocate_matrix_files(
                Path(directory), "train", "counterfactual", 2
            )
            for matrix in matrices.values():
                matrix[:] = 1.0
            commitments = {
                name: quality.matrix_sha256(matrix) for name, matrix in matrices.items()
            }
            frozen = quality.reopen_matrices_read_only(matrices, commitments)
            try:
                self.assertTrue(all(not matrix.flags.writeable for matrix in frozen.values()))
                with self.assertRaises(ValueError):
                    frozen["fs_full"][0, 0] = 2.0
            finally:
                for matrix in frozen.values():
                    matrix._mmap.close()  # type: ignore[union-attr]

    def test_identity33_is_frozen_read_only_before_truth_join(self) -> None:
        feature_names = tuple(f"identity_{index:02d}" for index in range(33))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_root = root / "public"
            temp_root = root / "temp"
            temp_root.mkdir()
            for split in ("train", "development"):
                path = public_root / split / "observed" / "identity33_all_pairs.csv"
                path.parent.mkdir(parents=True)
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=(
                            "canonical_pair_uid",
                            "world_uid",
                            *feature_names,
                        ),
                    )
                    writer.writeheader()
                    for index in range(378):
                        writer.writerow(
                            {
                                "canonical_pair_uid": f"pair-{index:03d}",
                                "world_uid": f"{split}-world",
                                **{
                                    name: float(index + column)
                                    for column, name in enumerate(feature_names)
                                },
                            }
                        )
            frozen = quality.freeze_identity_positive_control_matrices(
                public_root=public_root,
                temp_root=temp_root,
                expected_worlds=1,
            )
            try:
                self.assertEqual(tuple(frozen["feature_names"]), feature_names)
                self.assertTrue(
                    all(
                        not matrix.flags.writeable
                        for matrix in frozen["matrices"].values()
                    )
                )
            finally:
                for matrix in frozen["matrices"].values():
                    matrix._mmap.close()  # type: ignore[union-attr]

    def test_truth_requires_six_registered_controls_to_be_negative(self) -> None:
        world_uid = "world-0"
        pair_uids = [f"pair-{index:03d}" for index in range(378)]
        excluded = tuple(pair_uids[-6:])
        labels = [1 if index < 20 else 0 for index in range(378)]
        row_keys = tuple((world_uid, pair_uid) for pair_uid in pair_uids[:-6])
        with tempfile.TemporaryDirectory() as directory:
            private_root = Path(directory)
            split_root = private_root / "train"
            split_root.mkdir()
            path = split_root / "pair_labels.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("canonical_pair_uid", "world_uid", "label"),
                )
                writer.writeheader()
                writer.writerows(
                    {
                        "world_uid": world_uid,
                        "canonical_pair_uid": pair_uid,
                        "label": label,
                    }
                    for pair_uid, label in zip(pair_uids, labels, strict=True)
                )
            observed, _receipt, _index = quality.load_truth_once(
                private_root=private_root,
                split="train",
                row_keys=row_keys,
                excluded_pair_uids_by_world={world_uid: excluded},
                expected_truth={
                    (world_uid, pair_uid): label
                    for pair_uid, label in zip(pair_uids, labels, strict=True)
                },
                expected_worlds=1,
            )
            self.assertEqual(int(observed.sum()), 20)
            path.unlink()
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("canonical_pair_uid", "world_uid", "label"),
                )
                writer.writeheader()
                writer.writerows(
                    {
                        "world_uid": world_uid,
                        "canonical_pair_uid": pair_uid,
                        "label": label,
                    }
                    for pair_uid, label in reversed(
                        list(zip(pair_uids, labels, strict=True))
                    )
                )
            with self.assertRaisesRegex(
                quality.FormalRootQualityAttempt1Error,
                "label rows/order disagree",
            ):
                quality.load_truth_once(
                    private_root=private_root,
                    split="train",
                    row_keys=row_keys,
                    excluded_pair_uids_by_world={world_uid: excluded},
                    expected_truth={
                        (world_uid, pair_uid): label
                        for pair_uid, label in zip(pair_uids, labels, strict=True)
                    },
                    expected_worlds=1,
                )
            labels[0] = 0
            labels[-1] = 1
            path.unlink()
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("canonical_pair_uid", "world_uid", "label"),
                )
                writer.writeheader()
                writer.writerows(
                    {
                        "world_uid": world_uid,
                        "canonical_pair_uid": pair_uid,
                        "label": label,
                    }
                    for pair_uid, label in zip(pair_uids, labels, strict=True)
                )
            with self.assertRaisesRegex(
                quality.FormalRootQualityAttempt1Error,
                "Persistent label rows/order disagree with frozen-schedule truth",
            ):
                quality.load_truth_once(
                    private_root=private_root,
                    split="train",
                    row_keys=row_keys,
                    excluded_pair_uids_by_world={world_uid: excluded},
                    expected_truth={
                        (world_uid, pair_uid): int(index < 20)
                        for index, pair_uid in enumerate(pair_uids)
                    },
                    expected_worlds=1,
                )
            with self.assertRaisesRegex(
                quality.FormalRootQualityAttempt1Error,
                "Registered negative-control truth drift",
            ):
                quality.load_truth_once(
                    private_root=private_root,
                    split="train",
                    row_keys=row_keys,
                    excluded_pair_uids_by_world={world_uid: excluded},
                    expected_truth={
                        (world_uid, pair_uid): label
                        for pair_uid, label in zip(pair_uids, labels, strict=True)
                    },
                    expected_worlds=1,
                )

    def test_membership_and_qrels_require_exact_independent_replay(self) -> None:
        seller_uids = tuple(f"seller-{index:02d}" for index in range(28))
        groups = tuple(
            (seller_uids[index], seller_uids[index + 1])
            for index in range(0, 24, 2)
        ) + ((seller_uids[24], seller_uids[25], seller_uids[26], seller_uids[27]),)
        # The synthetic one-world fixture has 18 positive pairs rather than the
        # formal schedule's 20; this test exercises row-exact replay only.
        world = SimpleNamespace(
            world_uid="world-0",
            seller_uids=seller_uids,
            controller_groups=groups,
        )
        with patch.object(quality, "scheduled_worlds", return_value=[world]):
            with self.assertRaisesRegex(
                quality.FormalRootQualityAttempt1Error,
                "membership/qrels cardinality drift",
            ):
                quality.build_expected_membership_and_qrels_from_frozen_schedule(
                    split="train", expected_worlds=1
                )

        groups = tuple(
            (seller_uids[index], seller_uids[index + 1])
            for index in range(0, 16, 2)
        ) + tuple(
            (seller_uids[index], seller_uids[index + 1], seller_uids[index + 2])
            for index in range(16, 28, 3)
        )
        world.controller_groups = groups
        with patch.object(quality, "scheduled_worlds", return_value=[world]):
            membership, qrels, receipt = (
                quality.build_expected_membership_and_qrels_from_frozen_schedule(
                    split="train", expected_worlds=1
                )
            )
        self.assertEqual(receipt["directed_relevant_relation_count"], 40)
        with tempfile.TemporaryDirectory() as directory:
            private_root = Path(directory)
            split_root = private_root / "train"
            split_root.mkdir()
            for filename, rows in (
                ("controller_membership.jsonl", membership),
                ("qrels.jsonl", qrels),
            ):
                with (split_root / filename).open(
                    "w", encoding="utf-8", newline="\n"
                ) as stream:
                    for row in rows:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            observed = quality.verify_membership_and_qrels_once(
                private_root=private_root,
                split="train",
                expected_membership=membership,
                expected_qrels=qrels,
            )
            self.assertTrue(observed["exact_frozen_schedule_qrels_match"])
            changed_qrels = copy.deepcopy(qrels)
            changed_qrels[20]["relevant_seller_uids"].reverse()
            with self.assertRaisesRegex(
                quality.FormalRootQualityAttempt1Error,
                "Persisted qrels disagree",
            ):
                quality.verify_membership_and_qrels_once(
                    private_root=private_root,
                    split="train",
                    expected_membership=membership,
                    expected_qrels=changed_qrels,
                )

    def test_profile_lineage_receipt_detects_support_changes(self) -> None:
        original_row = {
            "seller_uid": "seller-a",
            "output_field": "title_concat_top",
            "output_rank": 1,
            "source_item_uids": ["item-a"],
            "source_item_uids_sha256": "a" * 64,
            "source_item_count": 1,
            "first_seen_position": 1,
            "item_uid": "",
            "extracted_segment_ordinal": -1,
            "seller_df": 0,
            "seller_df_seller_count": 0,
            "seller_df_seller_uids_sha256": "b" * 64,
        }
        changed_row = dict(original_row)
        changed_row["source_item_uids"] = ["item-b"]
        changed_row["source_item_uids_sha256"] = "c" * 64
        original = {
            "world_uid": "world",
            "rows": [original_row],
            "rows_sha256": "d" * 64,
        }
        counterfactual = {
            "world_uid": "world",
            "rows": [changed_row],
            "rows_sha256": "e" * 64,
        }
        receipt = quality.profile_lineage_difference_receipt(
            original, counterfactual
        )
        field = receipt["output_fields"][0]
        self.assertEqual(field["aligned_slot_count"], 1)
        self.assertEqual(field["changed_aligned_slot_count"], 1)
        self.assertEqual(field["source_support_change_count"], 1)
        self.assertEqual(field["support_jaccard_minimum"], 0.0)

    def test_model_and_single_feature_families_cannot_silently_shrink(self) -> None:
        policy = quality.read_json(quality.POLICY_PATH)
        train = {
            view: np.zeros((4, width), dtype="<f8")
            for view, width in zip(
                text_views.VIEW_ORDER, text_views.EXPECTED_WIDTHS, strict=True
            )
        }
        development = {view: matrix.copy() for view, matrix in train.items()}
        labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
        with patch.object(
            quality.probe_validator,
            "_fit_probe_models",
            return_value={"logistic_l2": np.full(4, 0.5, dtype="<f8")},
        ):
            with self.assertRaisesRegex(
                quality.FormalRootQualityAttempt1Error,
                "Per-view probe model family drift",
            ):
                quality.fit_model_family(
                    train_matrices=train,
                    development_matrices=development,
                    train_labels=labels,
                    development_labels=labels,
                    policy=policy,
                    role="TEST",
                )
        short_names = {view: ("only-one",) for view in text_views.VIEW_ORDER}
        with self.assertRaisesRegex(
            quality.FormalRootQualityAttempt1Error,
            "Single-feature family count drift",
        ):
            quality.maximum_single_feature(development, labels, short_names)

    def test_f_p_u_alignment_rejects_one_reordered_path(self) -> None:
        row_keys = tuple(("world", f"pair-{index:03d}") for index in range(378))
        alignment = {
            "row_keys": {
                "F": row_keys,
                "P": tuple(reversed(row_keys)),
                "U": row_keys,
            },
            "source_commitments": {"F": "a" * 64, "P": "b" * 64, "U": "c" * 64},
            "fixed_slot_key_set_sha256": "d" * 64,
            "fixed_slot_presence_sha256": "0" * 64,
            "production_profile_key_set_sha256": "e" * 64,
            "numeric_projection_sha256": "f" * 64,
        }
        views = {
            view: np.zeros((378, width), dtype="<f8")
            for view, width in zip(
                text_views.VIEW_ORDER, text_views.EXPECTED_WIDTHS, strict=True
            )
        }
        excluded = {pair_uid for _world, pair_uid in row_keys[-6:]}
        with self.assertRaisesRegex(
            quality.FormalRootQualityAttempt1Error,
            "F/P/U eligible row alignment drift",
        ):
            quality.finalize_path_alignment(
                alignment=alignment,
                views=views,
                excluded_pair_uids=excluded,
            )


if __name__ == "__main__":
    unittest.main()
