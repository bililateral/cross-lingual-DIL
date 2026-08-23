#!/usr/bin/env python3
"""Bounded contracts for the V9.2 42/14/4 numerical topology."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common
import step28_v13_v1_13_quality_gate_registry_v9_2 as gate_registry
import step28_v13_v1_13_quality_probe_preparer_v9 as preparer
import step28_v13_v1_13_quality_probe_preparer_v9_2 as preparer_v9_2
import step28_v13_v1_13_quality_probe_validator_v9 as validator_v9
import step28_v13_v1_13_quality_probe_validator_v9_2 as validator
import step28_v13_v1_13_quality_truth_capability_v9_2 as truth_v9_2


def _source(label: str) -> preparer.SourceCommitment:
    payload = label.encode("utf-8")
    return preparer.SourceCommitment(
        path=f"fixture/{label}.jsonl",
        size_bytes=len(payload),
        sha256=common.sha256_bytes(payload),
    )


def _row_keys(split: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"{split}_world_{world}", f"{split}_world_{world}_pair_{pair}")
        for world in range(2)
        for pair in range(6)
    )


def _truth(split: str) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "canonical_pair_uid": pair_uid,
            "world_uid": world_uid,
            "label": int(int(pair_uid.rsplit("_", 1)[1]) < 1),
        }
        for world_uid, pair_uid in _row_keys(split)
    )


def _eligibility(split: str) -> preparer.FrozenTextEligibility:
    endpoints = []
    rows = []
    seller_pairs = tuple(
        (left, right) for left in range(4) for right in range(left + 1, 4)
    )
    for world_uid, pair_uid in _row_keys(split):
        pair_index = int(pair_uid.rsplit("_", 1)[1])
        left_index, right_index = seller_pairs[pair_index]
        endpoints.append(
            {
                "canonical_pair_uid": pair_uid,
                "world_uid": world_uid,
                "seller_uid_left": f"{world_uid}_seller_{left_index}",
                "seller_uid_right": f"{world_uid}_seller_{right_index}",
            }
        )
        rows.append(
            {
                "world_uid": world_uid,
                "canonical_pair_uid": pair_uid,
                "text_probe_eligible": pair_index < 4,
            }
        )
    return preparer.freeze_text_eligibility(
        eligibility_rows=rows,
        endpoints=endpoints,
        ordered_world_uids=(f"{split}_world_0", f"{split}_world_1"),
        sources=(_source(f"{split}_eligibility"),),
        expected_pairs_per_world=6,
        expected_excluded_pairs_per_world=2,
    )


def _matrix_values(view_index: int, *, development: bool) -> np.ndarray:
    values = np.empty((12, 1), dtype=np.float64)
    for index in range(12):
        pair_index = index % 6
        world_index = index // 6
        values[index, 0] = (
            ((pair_index * (view_index + 3) + world_index * 5) % 11) / 10.0
            + (0.013 if development else 0.0)
        )
    return values


def _text_matrices(split: str) -> tuple[preparer.FrozenFeatureMatrix, ...]:
    output = []
    view_index = 0
    for surface in preparer_v9_2.TEXT_SURFACES:
        for view in preparer.text_views.VIEW_ORDER:
            output.append(
                preparer.freeze_feature_matrix(
                    family="text",
                    view=f"{surface}::{view}",
                    values=_matrix_values(
                        view_index, development=split == "development"
                    ),
                    row_keys=_row_keys(split),
                    column_names=(f"fixture_feature_{view_index}",),
                    sources=(_source(f"{split}_{surface}_{view}"),),
                )
            )
            view_index += 1
    return tuple(output)


def _code_matrices(split: str) -> tuple[preparer.FrozenFeatureMatrix, ...]:
    return tuple(
        preparer.freeze_feature_matrix(
            family="code_and_slot",
            view=view,
            values=_matrix_values(index + 41, development=split == "development"),
            row_keys=_row_keys(split),
            column_names=(f"fixture_{view}",),
            sources=(_source(f"{split}_{view}"),),
        )
        for index, view in enumerate(("public_code_fixture", "decoded_slot_fixture"))
    )


def _designs() -> validator.ProbeDesignsV92:
    shared = {
        "expected_worlds": 2,
        "pairs_per_world": 6,
        "positives_per_world": 1,
        "excluded_pairs_per_world": 2,
        "average_precision_baseline": 1.0 / 4.0,
        "bootstrap_replicates": 31,
        "bootstrap_seed": 281320260810,
        "require_formal_bootstrap_binding": False,
        "claim_boundary": "FIXTURE_ONLY_NO_DATASET_CONCLUSION",
    }
    descriptive_widths = tuple(
        (f"{surface}::{view}", 1)
        for surface in preparer_v9_2.ORIGINAL_AUTHOR_SURFACES
        for view in preparer.text_views.VIEW_ORDER
    )
    hard_widths = tuple(
        (f"{preparer_v9_2.COUNTERFACTUAL_HARD_SURFACE}::{view}", 1)
        for view in preparer.text_views.VIEW_ORDER
    )
    return validator.ProbeDesignsV92(
        descriptive=validator_v9.ProbeFamilyDesign(
            family="text",
            view_widths=descriptive_widths,
            expected_views=21,
            expected_total_features=21,
            expected_column_name_hashes=None,
            **shared,
        ),
        counterfactual_text=validator_v9.ProbeFamilyDesign(
            family="text",
            view_widths=hard_widths,
            expected_views=7,
            expected_total_features=7,
            expected_column_name_hashes=None,
            **shared,
        ),
        code_and_slot=validator_v9.ProbeFamilyDesign(
            family="code_and_slot",
            view_widths=(("public_code_fixture", 1), ("decoded_slot_fixture", 1)),
            expected_views=2,
            expected_total_features=2,
            expected_column_name_hashes=None,
            expected_worlds=2,
            pairs_per_world=6,
            positives_per_world=1,
            excluded_pairs_per_world=0,
            average_precision_baseline=1.0 / 6.0,
            bootstrap_replicates=31,
            bootstrap_seed=281320260810,
            require_formal_bootstrap_binding=False,
            claim_boundary="FIXTURE_ONLY_NO_DATASET_CONCLUSION",
        ),
    )


def _policy() -> dict:
    return json.loads(
        (ROOT / "schema" / "step28_v13_v1_13_quality_channel_sensitivity_policy_v9.json").read_text(
            encoding="utf-8"
        )
    )


class QualityProbeValidatorV92Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_text = _text_matrices("train")
        cls.development_text = _text_matrices("development")
        cls.train_code = _code_matrices("train")
        cls.development_code = _code_matrices("development")
        cls.train_eligibility = _eligibility("train")
        cls.development_eligibility = _eligibility("development")

    def evaluate(self, truth_loader=None) -> dict:
        if truth_loader is None:
            truth_loader = lambda split: _truth(split)
        return validator.evaluate_fixture_probe_families(
            text_train_matrices=self.train_text,
            text_development_matrices=self.development_text,
            code_train_matrices=self.train_code,
            code_development_matrices=self.development_code,
            train_text_eligibility=self.train_eligibility,
            development_text_eligibility=self.development_eligibility,
            truth_loader=truth_loader,
            policy=_policy(),
            designs=_designs(),
        )

    def test_shared_truth_42_14_4_roles_and_all_metrics_complete(self) -> None:
        receipt = self.evaluate()
        families = receipt["family_receipts"]
        descriptive = families["original_author_descriptive"]
        hard_text = families["counterfactual_text"]
        code = families["public_code_private_slot"]
        self.assertEqual(descriptive["model_count"], 42)
        self.assertEqual(hard_text["model_family"]["model_count"], 14)
        self.assertEqual(code["model_family"]["model_count"], 4)
        self.assertEqual(
            receipt["truth_loader_call_counts"],
            {"train": 1, "development": 1, "audit_a": 0, "audit_b": 0},
        )
        self.assertEqual(
            len(receipt["numerical_gate_observations"]), 42 + 5 + 5
        )
        self.assertNotIn("passed", next(iter(descriptive["models"].values())))
        self.assertEqual(descriptive["gate_failures"], [])
        self.assertTrue(descriptive["passed_field_forbidden"])

    def test_descriptive_channel_differences_have_fixed_directions_and_hashes(self) -> None:
        receipt = self.evaluate()
        models = receipt["family_receipts"]["original_author_descriptive"][
            "models"
        ]
        full = models["surface_full::fs_full::logistic_l2"]
        masked = models["surface_code_masked::fs_full::logistic_l2"]
        neutral = models["surface_code_neutralized::fs_full::logistic_l2"]
        self.assertEqual(
            tuple(full["channel_differences"]),
            ("full_minus_code_masked", "full_minus_code_neutralized"),
        )
        self.assertEqual(
            tuple(masked["channel_differences"]),
            ("code_masked_minus_code_neutralized",),
        )
        self.assertEqual(neutral["channel_differences"], {})
        for value in full["channel_differences"].values():
            self.assertRegex(
                value["bootstrap_symmetric_auc_difference_vector_sha256"],
                r"^[0-9a-f]{64}$",
            )

    def test_truth_callback_mutation_is_detected_before_any_fit(self) -> None:
        target = self.train_text[0].values
        original = float(target[0, 0])

        def mutating_loader(split: str):
            if split == "train":
                target.setflags(write=True)
                target[0, 0] = original + 1.0
                target.setflags(write=False)
            return _truth(split)

        try:
            with self.assertRaisesRegex(
                validator_v9.QualityProbeValidationError,
                "changed after truth open",
            ):
                self.evaluate(mutating_loader)
        finally:
            target.setflags(write=True)
            target[0, 0] = original
            target.setflags(write=False)

    def test_descriptive_extreme_results_never_enter_failed_hard_gate_ids(self) -> None:
        receipt = self.evaluate()
        observations = receipt["numerical_gate_observations"]
        descriptive_ids = {
            gate_id
            for gate_id in observations
            if gate_id.startswith("descriptive.original.")
        }
        hard_failures = {
            gate_id
            for gate_id, value in observations.items()
            if value.get("passed") is False
        }
        self.assertTrue(descriptive_ids)
        self.assertFalse(descriptive_ids & hard_failures)
        self.assertTrue(
            all(
                observations[gate_id]["gate_status"]
                == gate_registry.NOT_APPLICABLE
                and "passed" not in observations[gate_id]
                for gate_id in descriptive_ids
            )
        )

    def test_formal_call_layer_uses_one_root_bound_truth_read_per_split(self) -> None:
        root_binding = {
            "path": "reports/fixture_v9_2/root_manifest.json",
            "size_bytes": 321,
            "sha256": "a" * 64,
            "canonical_self_hash": "b" * 64,
        }
        with tempfile.TemporaryDirectory(prefix="step28-v9-2-formal-truth-") as temp:
            root = Path(temp)
            pins = {}
            for split in truth_v9_2.SUPERVISED_SPLITS:
                path = root / f"{split}.csv"
                lines = ["canonical_pair_uid,world_uid,label"]
                lines.extend(
                    f"{row['canonical_pair_uid']},{row['world_uid']},{row['label']}"
                    for row in _truth(split)
                )
                path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
                pins[split] = truth_v9_2.TruthFilePin(
                    split=split,
                    path=path,
                    size_bytes=path.stat().st_size,
                    sha256=common.sha256_file(path),
                    row_count=12,
                    split_manifest_self_hash="c" * 64,
                )
            capability = (
                truth_v9_2.FormalTrainDevelopmentTruthCapability._from_bounded_composition_fixture(
                    root_binding=root_binding,
                    pins=pins,
                )
            )
            policy = json.loads(
                (ROOT / "schema" / "step28_v13_v1_13_quality_channel_sensitivity_policy_v9.json").read_text(
                    encoding="utf-8"
                )
            )
            run_authorization = {
                "capabilities": {
                    "quality_audit_run": True,
                    "metric_generation": True,
                    "audit_a_b_truth_open": False,
                    "formal_500_by_4": False,
                    "model_training": False,
                },
                "design_root_manifest": root_binding,
            }
            reverification_calls = 0

            def reverify() -> None:
                nonlocal reverification_calls
                reverification_calls += 1

            with mock.patch.object(
                validator,
                "formal_designs",
                return_value=_designs(),
            ), mock.patch.object(
                validator.truth_capability.FormalTrainDevelopmentTruthCapability,
                "from_pinned_design_root",
                return_value=capability,
            ):
                receipt = validator.evaluate_formal_probe_families(
                    text_train_matrices=self.train_text,
                    text_development_matrices=self.development_text,
                    code_train_matrices=self.train_code,
                    code_development_matrices=self.development_code,
                    train_text_eligibility=self.train_eligibility,
                    development_text_eligibility=self.development_eligibility,
                    dataset_root=root,
                    root_manifest_pin=truth_v9_2.RootManifestPin(
                        path="root_manifest.json",
                        size_bytes=321,
                        sha256="a" * 64,
                        canonical_self_hash="b" * 64,
                    ),
                    policy=policy,
                    run_authorization=run_authorization,
                    verify_label_free_bytes=reverify,
                )
        self.assertEqual(reverification_calls, 1)
        self.assertEqual(receipt["label_free_byte_reverification_call_count"], 1)
        self.assertEqual(receipt["truth_file_access"]["train"]["file_open_count"], 1)
        self.assertEqual(
            receipt["truth_file_access"]["development"]["file_open_count"], 1
        )
        self.assertEqual(receipt["truth_file_access"]["audit_a"]["file_open_count"], 0)
        self.assertEqual(receipt["truth_file_access"]["audit_b"]["file_open_count"], 0)


if __name__ == "__main__":
    unittest.main()
