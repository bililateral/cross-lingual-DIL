from __future__ import annotations

import copy
import csv
from dataclasses import replace
import hashlib
import inspect
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_channel_policy_v9 as channel_policy
import step28_v13_v1_13_quality_probe_preparer_v9 as preparer
import step28_v13_v1_13_quality_probe_validator_v9 as validator
import step28_v13_v1_13_quality_truth_capability_v9 as truth_capability


SOURCE = (
    preparer.SourceCommitment(
        path="fixture/label_free.jsonl", size_bytes=10, sha256="2" * 64
    ),
)


def row_keys(prefix: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"{prefix}_world_{world}", f"{prefix}_world_{world}_pair_{pair}")
        for world in range(3)
        for pair in range(6)
    )


def labels_for(keys: tuple[tuple[str, str], ...]) -> list[dict[str, object]]:
    return [
        {
            "canonical_pair_uid": pair_uid,
            "world_uid": world_uid,
            "label": int(pair_uid.endswith("_0") or pair_uid.endswith("_1")),
        }
        for world_uid, pair_uid in keys
    ]


def eligibility_for(keys: tuple[tuple[str, str], ...]) -> list[dict[str, object]]:
    return [
        {
            "world_uid": world_uid,
            "canonical_pair_uid": pair_uid,
            "text_probe_eligible": not pair_uid.endswith("_5"),
        }
        for world_uid, pair_uid in keys
    ]


def endpoints_for(keys: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    endpoints: list[dict[str, str]] = []
    offset_by_world: dict[str, int] = {}
    sellers_by_world: dict[str, list[str]] = {}
    for world_uid, pair_uid in keys:
        sellers = sellers_by_world.setdefault(
            world_uid, [f"{world_uid}_seller_{index}" for index in range(4)]
        )
        pairs = [
            (sellers[left], sellers[right])
            for left in range(4)
            for right in range(left + 1, 4)
        ]
        offset = offset_by_world.get(world_uid, 0)
        left, right = pairs[offset]
        offset_by_world[world_uid] = offset + 1
        endpoints.append(
            {
                "canonical_pair_uid": pair_uid,
                "world_uid": world_uid,
                "seller_uid_left": left,
                "seller_uid_right": right,
            }
        )
    return endpoints


def frozen_eligibility(
    keys: tuple[tuple[str, str], ...],
    rows: list[dict[str, object]] | None = None,
) -> preparer.FrozenTextEligibility:
    worlds = tuple(dict.fromkeys(world_uid for world_uid, _pair_uid in keys))
    return preparer.freeze_text_eligibility(
        eligibility_rows=eligibility_for(keys) if rows is None else rows,
        endpoints=endpoints_for(keys),
        ordered_world_uids=worlds,
        sources=SOURCE,
        expected_pairs_per_world=6,
        expected_excluded_pairs_per_world=1,
    )


def matrices(
    split: str,
) -> tuple[preparer.FrozenFeatureMatrix, preparer.FrozenFeatureMatrix]:
    keys = row_keys(split)
    offset = 0.05 if split == "development" else 0.0
    first = np.asarray(
        [
            [
                ((index * 7) % 13) / 13.0 + offset,
                ((index * 5 + 3) % 17) / 17.0,
            ]
            for index in range(len(keys))
        ],
        dtype=np.float64,
    )
    second = np.asarray(
        [[((index * 11 + 1) % 19) / 19.0] for index in range(len(keys))],
        dtype=np.float64,
    )
    return (
        preparer.freeze_feature_matrix(
            family="fixture",
            view="view_a",
            values=first,
            row_keys=keys,
            column_names=("a", "b"),
            sources=SOURCE,
        ),
        preparer.freeze_feature_matrix(
            family="fixture",
            view="view_b",
            values=second,
            row_keys=keys,
            column_names=("c",),
            sources=SOURCE,
        ),
    )


def fixture_design(*, text: bool = False) -> validator.ProbeFamilyDesign:
    return validator.ProbeFamilyDesign(
        family="fixture",
        view_widths=(("view_a", 2), ("view_b", 1)),
        expected_views=2,
        expected_total_features=3,
        expected_column_name_hashes=None,
        expected_worlds=3,
        pairs_per_world=6,
        positives_per_world=2,
        excluded_pairs_per_world=1 if text else 0,
        average_precision_baseline=2 / (5 if text else 6),
        bootstrap_replicates=31,
        bootstrap_seed=12345,
        require_formal_bootstrap_binding=False,
        claim_boundary="FIXTURE_ONLY_NO_DATASET_CONCLUSION",
    )


class QualityProbeValidatorV9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = channel_policy.load_policy()
        cls.train = matrices("train")
        cls.development = matrices("development")
        cls.truth = {
            "train": labels_for(cls.train[0].row_keys),
            "development": labels_for(cls.development[0].row_keys),
        }

    def test_fixture_opens_only_train_and_development_after_freeze(self) -> None:
        calls: list[str] = []

        def load(split: str) -> list[dict[str, object]]:
            calls.append(split)
            return self.truth[split]

        receipt = validator.evaluate_fixture_probe_family(
            train_matrices=self.train,
            development_matrices=self.development,
            truth_loader=load,
            design=fixture_design(),
            policy=self.policy,
        )
        self.assertEqual(calls, ["train", "development"])
        self.assertEqual(
            receipt["truth_loader_call_counts"],
            {"train": 1, "development": 1, "audit_a": 0, "audit_b": 0},
        )
        self.assertEqual(receipt["single_feature"]["evaluated_feature_count"], 3)
        self.assertEqual(receipt["model_family"]["model_count"], 4)
        self.assertEqual(receipt["row_level_labels_returned"], 0)
        self.assertEqual(receipt["row_level_predictions_returned"], 0)
        self.assertIn(
            receipt["status"],
            {
                "FIXTURE_MECHANISM_PASS_NO_DATASET_CONCLUSION",
                "FIXTURE_MECHANISM_GATE_TRIGGERED_NO_DATASET_CONCLUSION",
            },
        )
        self.assertNotIn(receipt["status"], {"PASS", "DATASET_INVALIDATED"})
        self.assertEqual(len(receipt["canonical_self_hash"]), 64)
        self.assertEqual(len(receipt["gate_checks"]), 5)
        self.assertEqual(len(receipt["input_commitments"]["train"]), 2)
        for model in receipt["model_family"]["models"].values():
            self.assertEqual(len(model["prediction_vector_sha256"]), 64)

    def test_tampered_matrix_fails_before_truth_loader(self) -> None:
        tampered = matrices("development")
        tampered[0].values.setflags(write=True)
        tampered[0].values[0, 0] += 1.0
        tampered[0].values.setflags(write=False)
        calls: list[str] = []
        with self.assertRaises(preparer.QualityProbePreparationError):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=tampered,
                truth_loader=lambda split: calls.append(split) or self.truth[split],
                design=fixture_design(),
                policy=self.policy,
            )
        self.assertEqual(calls, [])

    def test_truth_loader_cannot_mutate_frozen_features_after_open(self) -> None:
        development = matrices("development")

        def mutating_loader(split: str) -> list[dict[str, object]]:
            if split == "development":
                development[0].values.setflags(write=True)
                development[0].values[0, 0] += 1.0
                development[0].values.setflags(write=False)
            return self.truth[split]

        with self.assertRaises(preparer.QualityProbePreparationError):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=development,
                truth_loader=mutating_loader,
                design=fixture_design(),
                policy=self.policy,
            )

    def test_truth_loader_cannot_replace_matrix_and_its_stored_commitment(self) -> None:
        development = matrices("development")

        def mutating_loader(split: str) -> list[dict[str, object]]:
            if split == "development":
                frozen = development[0]
                replacement = np.array(frozen.values, copy=True)
                replacement[:, 0] = np.asarray(
                    [row["label"] for row in self.truth["development"]],
                    dtype=np.float64,
                )
                replacement.setflags(write=False)
                object.__setattr__(frozen, "values", replacement)
                forged = preparer.current_feature_matrix_commitment_json(frozen)
                object.__setattr__(frozen, "commitment_json", forged)
                object.__setattr__(
                    frozen,
                    "commitment_sha256",
                    preparer._sha256(forged),
                )
            return self.truth[split]

        with self.assertRaisesRegex(
            validator.QualityProbeValidationError,
            "Feature matrix changed after truth open",
        ):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=development,
                truth_loader=mutating_loader,
                design=fixture_design(),
                policy=self.policy,
            )

    def test_truth_loader_cannot_mutate_policy_after_open(self) -> None:
        mutable_policy = copy.deepcopy(self.policy)

        def mutating_loader(split: str) -> list[dict[str, object]]:
            if split == "development":
                mutable_policy["quality_gates"][
                    "maximum_single_feature_symmetric_auc"
                ] = 1.0
            return self.truth[split]

        with self.assertRaises(channel_policy.QualityChannelPolicyError):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=self.development,
                truth_loader=mutating_loader,
                design=fixture_design(),
                policy=mutable_policy,
            )

    def test_truth_loader_cannot_mutate_private_design_after_open(self) -> None:
        caller_design = fixture_design()

        def mutating_loader(split: str) -> list[dict[str, object]]:
            if split == "development":
                object.__setattr__(caller_design, "average_precision_baseline", 1.0)
            return self.truth[split]

        with self.assertRaisesRegex(
            validator.QualityProbeValidationError,
            "Probe design changed after truth open",
        ):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=self.development,
                truth_loader=mutating_loader,
                design=caller_design,
                policy=self.policy,
            )

    def test_truth_loader_cannot_mutate_frozen_eligibility_after_open(self) -> None:
        train_eligibility = frozen_eligibility(self.train[0].row_keys)
        development_eligibility = frozen_eligibility(
            self.development[0].row_keys
        )

        def mutating_loader(split: str) -> list[dict[str, object]]:
            if split == "development":
                development_eligibility.values.setflags(write=True)
                development_eligibility.values[4] = False
                development_eligibility.values[5] = True
                development_eligibility.values.setflags(write=False)
            return self.truth[split]

        with self.assertRaises(preparer.QualityProbePreparationError):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=self.development,
                truth_loader=mutating_loader,
                design=fixture_design(text=True),
                policy=self.policy,
                train_eligibility=train_eligibility,
                development_eligibility=development_eligibility,
            )

    def test_validator_uses_private_mask_when_callback_detaches_old_alias(self) -> None:
        train_eligibility = frozen_eligibility(self.train[0].row_keys)
        development_eligibility = frozen_eligibility(
            self.development[0].row_keys
        )
        old_values = development_eligibility.values

        baseline = validator.evaluate_fixture_probe_family(
            train_matrices=self.train,
            development_matrices=self.development,
            truth_loader=lambda split: self.truth[split],
            design=fixture_design(text=True),
            policy=self.policy,
            train_eligibility=frozen_eligibility(self.train[0].row_keys),
            development_eligibility=frozen_eligibility(
                self.development[0].row_keys
            ),
        )

        def mutating_loader(split: str) -> list[dict[str, object]]:
            if split == "development":
                clean = np.array(old_values, copy=True)
                clean.setflags(write=False)
                object.__setattr__(development_eligibility, "values", clean)
                old_values.setflags(write=True)
                old_values[0] = False
                old_values[5] = True
                old_values.setflags(write=False)
            return self.truth[split]

        receipt = validator.evaluate_fixture_probe_family(
            train_matrices=self.train,
            development_matrices=self.development,
            truth_loader=mutating_loader,
            design=fixture_design(text=True),
            policy=self.policy,
            train_eligibility=train_eligibility,
            development_eligibility=development_eligibility,
        )
        self.assertEqual(receipt["eligible_pair_count_per_world"], 5)
        self.assertEqual(receipt["canonical_self_hash"], baseline["canonical_self_hash"])

    def test_frozen_formal_column_names_are_not_width_only(self) -> None:
        wrong_hashes = tuple(
            (value.view, "0" * 64) for value in self.train
        )
        calls: list[str] = []
        with self.assertRaisesRegex(
            validator.QualityProbeValidationError, "column-name commitment"
        ):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=self.development,
                truth_loader=lambda split: calls.append(split) or self.truth[split],
                design=replace(
                    fixture_design(),
                    expected_column_name_hashes=wrong_hashes,
                ),
                policy=self.policy,
            )
        self.assertEqual(calls, [])

    def test_pair_uids_cannot_cross_train_development(self) -> None:
        train_pair_uids = [pair_uid for _world_uid, pair_uid in self.train[0].row_keys]
        development_keys = tuple(
            (world_uid, train_pair_uids[index])
            for index, (world_uid, _pair_uid) in enumerate(
                self.development[0].row_keys
            )
        )
        development = tuple(
            preparer.freeze_feature_matrix(
                family=value.family,
                view=value.view,
                values=value.values,
                row_keys=development_keys,
                column_names=value.column_names,
                sources=value.sources,
            )
            for value in self.development
        )
        calls: list[str] = []
        with self.assertRaisesRegex(
            validator.QualityProbeValidationError, "world boundary drift"
        ):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=development,
                truth_loader=lambda split: calls.append(split) or self.truth[split],
                design=fixture_design(),
                policy=self.policy,
            )
        self.assertEqual(calls, [])

    def test_code_family_rejects_even_one_text_eligibility_capability(self) -> None:
        calls: list[str] = []
        with self.assertRaises(validator.QualityProbeValidationError):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=self.development,
                truth_loader=lambda split: calls.append(split) or self.truth[split],
                design=fixture_design(),
                policy=self.policy,
                train_eligibility=frozen_eligibility(self.train[0].row_keys),
            )
        self.assertEqual(calls, [])

    def test_formal_evaluator_is_closed_before_quality_authorization(self) -> None:
        self.assertNotIn(
            "truth_loader",
            inspect.signature(validator.evaluate_formal_probe_family).parameters,
        )
        self.assertFalse(hasattr(validator, "_FORMAL_EVALUATION_TOKEN"))
        self.assertFalse(hasattr(truth_capability, "_VALIDATOR_CONSUME_TOKEN"))
        with self.assertRaisesRegex(
            validator.QualityProbeValidationError, "remain unauthorized"
        ):
            validator.evaluate_formal_probe_family(
                family="code_and_slot",
                train_matrices=(),
                development_matrices=(),
                policy=self.policy,
            )

    def test_formal_combined_evaluator_rejects_alternate_root_before_matrices(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["authorization"]["quality_audit_run"] = True
        policy["authorization"]["metric_generation"] = True
        policy["pins"]["design_root_manifest"] = {
            "path": "expected/root_manifest.json",
            "size_bytes": 2,
            "sha256": "c" * 64,
            "canonical_self_hash": "d" * 64,
        }
        with patch.object(channel_policy, "validate_policy", return_value=None):
            with self.assertRaisesRegex(
                validator.QualityProbeValidationError, "does not match"
            ):
                validator.evaluate_formal_probe_families(
                    text_train_matrices=(),
                    text_development_matrices=(),
                    code_train_matrices=(),
                    code_development_matrices=(),
                    dataset_root=Path("alternate"),
                    root_manifest_pin=truth_capability.RootManifestPin(
                        path="root_manifest.json",
                        size_bytes=1,
                        sha256="a" * 64,
                        canonical_self_hash="b" * 64,
                    ),
                    policy=policy,
                    train_text_eligibility=None,
                    development_text_eligibility=None,
                )

    def test_between_family_bundle_check_detects_mutation(self) -> None:
        expected = tuple(
            preparer.current_feature_matrix_commitment_json(value)
            for value in self.train
        )
        values = self.train[1].values
        original = float(values[0, 0])
        values.setflags(write=True)
        values[0, 0] += 1.0
        try:
            with self.assertRaisesRegex(
                validator.QualityProbeValidationError,
                "changed between probe families",
            ):
                validator._verify_feature_bundle_unchanged(
                    self.train,
                    expected,
                    error_message=(
                        "Formal matrix bundle changed between probe families"
                    ),
                )
        finally:
            values[0, 0] = original
            values.setflags(write=False)
        validator._verify_feature_bundle_unchanged(
            self.train,
            expected,
            error_message="Formal matrix bundle changed between probe families",
        )

    def test_root_bound_composition_fixture_physically_reads_both_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pins: dict[str, truth_capability.TruthFilePin] = {}
            expected_bytes: dict[str, int] = {}
            for split in truth_capability.SUPERVISED_SPLITS:
                path = Path(temp) / f"{split}.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=truth_capability.TRUTH_FIELDS
                    )
                    writer.writeheader()
                    writer.writerows(self.truth[split])
                raw = path.read_bytes()
                expected_bytes[split] = len(raw)
                pins[split] = truth_capability.TruthFilePin(
                    split=split,
                    path=path,
                    size_bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    row_count=len(self.truth[split]),
                    split_manifest_self_hash=("a" if split == "train" else "b")
                    * 64,
                )
            root_binding = {
                "path": "fixture/root_manifest.json",
                "size_bytes": 123,
                "sha256": "c" * 64,
                "canonical_self_hash": "d" * 64,
            }
            capability = truth_capability.FormalTrainDevelopmentTruthCapability._from_bounded_composition_fixture(
                root_binding=root_binding,
                pins=pins,
            )
            boundary = (
                "ROOT_BOUND_COMPOSITION_FIXTURE_ONLY_NO_DATASET_CONCLUSION"
            )
            receipt = validator.evaluate_root_bound_composition_fixture(
                text_train_matrices=self.train,
                text_development_matrices=self.development,
                code_train_matrices=self.train,
                code_development_matrices=self.development,
                truth=capability,
                expected_root_binding=root_binding,
                policy=self.policy,
                text_design=replace(
                    fixture_design(text=True), claim_boundary=boundary
                ),
                code_design=replace(fixture_design(), claim_boundary=boundary),
                train_text_eligibility=frozen_eligibility(
                    self.train[0].row_keys
                ),
                development_text_eligibility=frozen_eligibility(
                    self.development[0].row_keys
                ),
            )
            self.assertIn(
                receipt["status"],
                {
                    "ROOT_BOUND_COMPOSITION_FIXTURE_PASS_NO_DATASET_CONCLUSION",
                    "ROOT_BOUND_COMPOSITION_FIXTURE_GATE_TRIGGERED_NO_DATASET_CONCLUSION",
                },
            )
            for split in truth_capability.SUPERVISED_SPLITS:
                split_receipt = receipt["truth_file_access"][split]
                self.assertEqual(split_receipt["file_open_count"], 1)
                self.assertEqual(
                    split_receipt["byte_read_count"], expected_bytes[split]
                )
                self.assertEqual(
                    split_receipt["materialized_row_count"],
                    len(self.truth[split]),
                )
            self.assertEqual(receipt["row_level_labels_returned"], 0)
            self.assertEqual(receipt["row_level_predictions_returned"], 0)

    def test_internal_evaluator_rejects_nonfixture_callback_claim(self) -> None:
        calls: list[str] = []
        with self.assertRaisesRegex(
            validator.QualityProbeValidationError,
            "preloaded rows from the root-bound transaction",
        ):
            validator._evaluate(
                train_matrices=self.train,
                development_matrices=self.development,
                truth_loader=lambda split: calls.append(split) or self.truth[split],
                design=replace(
                    fixture_design(),
                    claim_boundary="FORMAL_CALLBACK_FORBIDDEN",
                ),
                policy=self.policy,
                train_eligibility=None,
                development_eligibility=None,
            )
        self.assertEqual(calls, [])

    def test_text_mask_is_frozen_before_truth_and_may_exclude_only_negatives(self) -> None:
        receipt = validator.evaluate_fixture_probe_family(
            train_matrices=self.train,
            development_matrices=self.development,
            truth_loader=lambda split: self.truth[split],
            design=fixture_design(text=True),
            policy=self.policy,
            train_eligibility=frozen_eligibility(self.train[0].row_keys),
            development_eligibility=frozen_eligibility(
                self.development[0].row_keys
            ),
        )
        self.assertEqual(receipt["eligible_pair_count_per_world"], 5)
        invalid = eligibility_for(self.train[0].row_keys)
        invalid[0]["text_probe_eligible"] = False
        invalid[5]["text_probe_eligible"] = True
        with self.assertRaises(validator.QualityProbeValidationError):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=self.development,
                truth_loader=lambda split: self.truth[split],
                design=fixture_design(text=True),
                policy=self.policy,
                train_eligibility=frozen_eligibility(
                    self.train[0].row_keys, invalid
                ),
                development_eligibility=frozen_eligibility(
                    self.development[0].row_keys
                ),
            )

    def test_truth_extra_field_and_audit_open_fail_closed(self) -> None:
        contaminated = {
            split: [dict(row) for row in rows] for split, rows in self.truth.items()
        }
        contaminated["train"][0]["controller_uid"] = "forbidden"
        with self.assertRaises(validator.QualityProbeValidationError):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=self.development,
                truth_loader=lambda split: contaminated[split],
                design=fixture_design(),
                policy=self.policy,
            )
        with self.assertRaises(validator.QualityProbeValidationError):
            validator.reject_audit_truth_open("audit_b")
        wrong_type = {
            split: [dict(row) for row in rows] for split, rows in self.truth.items()
        }
        wrong_type["train"][0]["canonical_pair_uid"] = 123
        with self.assertRaises(validator.QualityProbeValidationError):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=self.development,
                truth_loader=lambda split: wrong_type[split],
                design=fixture_design(),
                policy=self.policy,
            )

    def test_no_model_fit_uses_sample_weight(self) -> None:
        source = inspect.getsource(validator._fit_probe_models)
        self.assertNotIn("sample_weight", source)

    def test_runtime_fit_and_metrics_receive_no_sample_weight(self) -> None:
        events: list[str] = []
        original_scaler = validator.StandardScaler
        original_logistic = validator.LogisticRegression
        original_tree = validator.HistGradientBoostingClassifier
        original_auc = validator.roc_auc_score
        original_ap = validator.average_precision_score

        class SpyScaler(original_scaler):
            def fit(self, x: np.ndarray, y: object = None, **kwargs: object):
                self.assert_no_weight(kwargs)
                events.append("scaler")
                return super().fit(x, y, **kwargs)

            @staticmethod
            def assert_no_weight(kwargs: dict[str, object]) -> None:
                if "sample_weight" in kwargs:
                    raise AssertionError("sample_weight reached StandardScaler")

        class SpyLogistic(original_logistic):
            def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: object):
                if "sample_weight" in kwargs:
                    raise AssertionError("sample_weight reached LogisticRegression")
                events.append("logistic")
                return super().fit(x, y, **kwargs)

        class SpyTree(original_tree):
            def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: object):
                if "sample_weight" in kwargs:
                    raise AssertionError("sample_weight reached tree")
                events.append("tree")
                return super().fit(x, y, **kwargs)

        def spy_auc(y: np.ndarray, score: np.ndarray, *args: object, **kwargs: object):
            if args or "sample_weight" in kwargs:
                raise AssertionError("sample_weight reached AUC")
            events.append("auc")
            return original_auc(y, score, **kwargs)

        def spy_ap(y: np.ndarray, score: np.ndarray, *args: object, **kwargs: object):
            if args or "sample_weight" in kwargs:
                raise AssertionError("sample_weight reached AP")
            events.append("ap")
            return original_ap(y, score, **kwargs)

        with (
            patch.object(validator, "StandardScaler", SpyScaler),
            patch.object(validator, "LogisticRegression", SpyLogistic),
            patch.object(validator, "HistGradientBoostingClassifier", SpyTree),
            patch.object(validator, "roc_auc_score", spy_auc),
            patch.object(validator, "average_precision_score", spy_ap),
        ):
            validator.evaluate_fixture_probe_family(
                train_matrices=self.train,
                development_matrices=self.development,
                truth_loader=lambda split: self.truth[split],
                design=fixture_design(),
                policy=self.policy,
            )
        self.assertTrue({"scaler", "logistic", "tree", "auc", "ap"} <= set(events))

    def test_receipt_contains_only_aggregate_json_values(self) -> None:
        receipt = validator.evaluate_fixture_probe_family(
            train_matrices=self.train,
            development_matrices=self.development,
            truth_loader=lambda split: self.truth[split],
            design=fixture_design(),
            policy=self.policy,
        )

        def assert_json_aggregate(value: object) -> None:
            self.assertNotIsInstance(value, np.ndarray)
            if isinstance(value, dict):
                self.assertTrue(all(type(key) is str for key in value))
                for nested in value.values():
                    assert_json_aggregate(nested)
            elif isinstance(value, list):
                for nested in value:
                    assert_json_aggregate(nested)
            else:
                self.assertIn(type(value), {str, int, float, bool, type(None)})

        assert_json_aggregate(receipt)

    def test_missing_preregistered_model_fails_closed(self) -> None:
        def incomplete_fit(**kwargs: object) -> dict[str, np.ndarray]:
            development_x = np.asarray(kwargs["development_x"])
            return {"logistic_l2": np.full(development_x.shape[0], 0.5)}

        with patch.object(validator, "_fit_probe_models", incomplete_fit):
            with self.assertRaisesRegex(
                validator.QualityProbeValidationError, "cardinality drift"
            ):
                validator.evaluate_fixture_probe_family(
                    train_matrices=self.train,
                    development_matrices=self.development,
                    truth_loader=lambda split: self.truth[split],
                    design=fixture_design(),
                    policy=self.policy,
                )

    def test_formal_bootstrap_index_matrix_is_exact(self) -> None:
        draws = validator.generate_bootstrap_draws(
            replicates=9999, world_count=500, seed=281320260810
        )
        self.assertEqual(draws.shape, (9999, 500))
        self.assertEqual(draws.nbytes, 39_996_000)
        self.assertEqual(
            validator._sha256_bytes(draws.tobytes(order="C")),
            validator.FORMAL_BOOTSTRAP_SHA256,
        )

    def test_vectorized_bootstrap_matches_literal_world_repetition(self) -> None:
        labels = np.asarray([1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int64)
        scores = np.asarray([0.8, 0.2, 0.3, 0.4, 0.5, 0.1, 0.9, 0.7, 0.6])
        tied_scores = np.asarray([0.8, 0.8, 0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
        score_family = {"unique": scores, "tied": tied_scores}
        worlds = ("w0", "w1", "w2")
        row_worlds = tuple(world for world in worlds for _ in range(3))
        draws = validator.generate_bootstrap_draws(
            replicates=31, world_count=3, seed=999
        )
        captured_vectors: list[np.ndarray] = []
        original_vector_sha256 = validator._vector_sha256

        def capture_vector(values: np.ndarray) -> str:
            captured_vectors.append(np.asarray(values).copy())
            return original_vector_sha256(values)

        with patch.object(validator, "_vector_sha256", capture_vector):
            observed = validator._bootstrap_family_upper(
                labels=labels,
                row_world_uids=row_worlds,
                ordered_world_uids=worlds,
                score_family=score_family,
                baseline=1 / 3,
                draws=draws,
            )
        auc_values: list[float] = []
        ap_values: list[float] = []
        indices_by_world = [np.flatnonzero(np.asarray(row_worlds) == world) for world in worlds]
        for draw in draws:
            repeated = np.concatenate([indices_by_world[int(index)] for index in draw])
            replicate_aucs: list[float] = []
            replicate_aps: list[float] = []
            for values in score_family.values():
                auc = float(roc_auc_score(labels[repeated], values[repeated]))
                replicate_aucs.append(max(auc, 1.0 - auc))
                replicate_aps.append(
                    float(
                        average_precision_score(labels[repeated], values[repeated])
                    )
                    - 1 / 3
                )
            auc_values.append(max(replicate_aucs))
            ap_values.append(max(replicate_aps))
        self.assertAlmostEqual(
            observed["symmetric_auc_95_upper"],
            float(np.quantile(auc_values, 0.95, method="linear")),
            places=14,
        )
        self.assertAlmostEqual(
            observed["average_precision_uplift_95_upper"],
            float(np.quantile(ap_values, 0.95, method="linear")),
            places=14,
        )
        np.testing.assert_allclose(
            captured_vectors[0], np.asarray(auc_values), rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            captured_vectors[1], np.asarray(ap_values), rtol=0.0, atol=1e-15
        )

    def test_formal_design_is_exactly_500_worlds_and_two_separate_families(self) -> None:
        text = validator.formal_design_for_family("text", self.policy)
        code = validator.formal_design_for_family("code_and_slot", self.policy)
        self.assertEqual((text.expected_views, text.expected_total_features), (21, 1038))
        self.assertEqual((code.expected_views, code.expected_total_features), (2, 3380))
        self.assertEqual(len(text.expected_column_name_hashes or ()), 21)
        self.assertEqual(
            code.expected_column_name_hashes,
            (
                (
                    "public_code_2992",
                    self.policy["public_code_probe"][
                        "feature_names_canonical_json_sha256"
                    ],
                ),
                (
                    "decoded_slot_388",
                    self.policy["decoded_slot_probe"][
                        "feature_names_canonical_json_sha256"
                    ],
                ),
            ),
        )
        self.assertEqual((text.expected_worlds, code.expected_worlds), (500, 500))
        self.assertEqual((text.pairs_per_world, code.pairs_per_world), (378, 378))
        self.assertEqual((text.excluded_pairs_per_world, code.excluded_pairs_per_world), (6, 0))


if __name__ == "__main__":
    unittest.main()
