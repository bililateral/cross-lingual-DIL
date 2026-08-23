#!/usr/bin/env python3
"""Shared-truth V9.2 validator for 42 descriptive, 14 text and 4 code models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

import step28_v13_common as common
import step28_v13_v1_13_quality_complete_evidence_v9_2 as complete_evidence
import step28_v13_v1_13_quality_gate_registry_v9_2 as gate_registry
import step28_v13_v1_13_quality_probe_preparer_v9 as preparer_v9
import step28_v13_v1_13_quality_probe_preparer_v9_2 as preparer_v9_2
import step28_v13_v1_13_quality_probe_validator_v9 as validator_v9
import step28_v13_v1_13_quality_truth_capability_v9_2 as truth_capability


VERSION = "2026-08-23-step28-v13-v1-13-quality-probe-validator-v9-2"
MODEL_KEYS = ("logistic_l2", "hist_gradient_boosting_depth2")
CHANNEL_DIFFERENCE_DIRECTIONS = (
    ("full_minus_code_masked", "surface_full", "surface_code_masked"),
    ("full_minus_code_neutralized", "surface_full", "surface_code_neutralized"),
    (
        "code_masked_minus_code_neutralized",
        "surface_code_masked",
        "surface_code_neutralized",
    ),
)


class QualityProbeValidationV92Error(validator_v9.QualityProbeValidationError):
    """Raised on V9.2 matrix, truth, model, or numerical contract drift."""


@dataclass(frozen=True)
class ProbeDesignsV92:
    descriptive: validator_v9.ProbeFamilyDesign
    counterfactual_text: validator_v9.ProbeFamilyDesign
    code_and_slot: validator_v9.ProbeFamilyDesign


def formal_designs(policy: Mapping[str, Any]) -> ProbeDesignsV92:
    text = policy["text_probe_family_v9_2"]
    code = policy["public_code_probe"]
    slot = policy["decoded_slot_probe"]
    worlds = int(policy["design_scale"]["world_counts"]["development"])
    pairs = int(policy["design_scale"]["pair_count_per_world"])
    positives = int(policy["design_scale"]["positive_pair_count_per_world"])
    bootstrap = policy["bootstrap"]
    original_widths = tuple(
        (f"{surface}::{view}", int(width))
        for surface in preparer_v9_2.ORIGINAL_AUTHOR_SURFACES
        for view, width in zip(
            text["view_names"], text["feature_widths"], strict=True
        )
    )
    hard_widths = tuple(
        (f"{preparer_v9_2.COUNTERFACTUAL_HARD_SURFACE}::{view}", int(width))
        for view, width in zip(
            text["view_names"], text["feature_widths"], strict=True
        )
    )
    name_hashes_original = tuple(
        (f"{surface}::{view}", preparer_v9.text_views.EXPECTED_NAME_HASHES[view])
        for surface in preparer_v9_2.ORIGINAL_AUTHOR_SURFACES
        for view in text["view_names"]
    )
    name_hashes_hard = tuple(
        (
            f"{preparer_v9_2.COUNTERFACTUAL_HARD_SURFACE}::{view}",
            preparer_v9.text_views.EXPECTED_NAME_HASHES[view],
        )
        for view in text["view_names"]
    )
    shared = {
        "expected_worlds": worlds,
        "pairs_per_world": pairs,
        "positives_per_world": positives,
        "excluded_pairs_per_world": int(text["excluded_negative_pairs_per_world"]),
        "average_precision_baseline": float(text["average_precision_baseline"]),
        "bootstrap_replicates": int(bootstrap["replicates"]),
        "bootstrap_seed": int(bootstrap["text_design_seed"]),
        "require_formal_bootstrap_binding": True,
        "claim_boundary": "V9_2_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
    }
    return ProbeDesignsV92(
        descriptive=validator_v9.ProbeFamilyDesign(
            family="text",
            view_widths=original_widths,
            expected_views=21,
            expected_total_features=int(text["descriptive_single_feature_count"]),
            expected_column_name_hashes=name_hashes_original,
            **shared,
        ),
        counterfactual_text=validator_v9.ProbeFamilyDesign(
            family="text",
            view_widths=hard_widths,
            expected_views=7,
            expected_total_features=int(text["hard_single_feature_count"]),
            expected_column_name_hashes=name_hashes_hard,
            **shared,
        ),
        code_and_slot=validator_v9.ProbeFamilyDesign(
            family="code_and_slot",
            view_widths=(
                ("public_code_2992", int(code["feature_width"])),
                ("decoded_slot_388", int(slot["feature_width"])),
            ),
            expected_views=2,
            expected_total_features=int(code["feature_width"])
            + int(slot["feature_width"]),
            expected_column_name_hashes=(
                (
                    "public_code_2992",
                    str(code["feature_names_canonical_json_sha256"]),
                ),
                (
                    "decoded_slot_388",
                    str(slot["feature_names_canonical_json_sha256"]),
                ),
            ),
            expected_worlds=worlds,
            pairs_per_world=pairs,
            positives_per_world=positives,
            excluded_pairs_per_world=0,
            average_precision_baseline=float(slot["average_precision_baseline"]),
            bootstrap_replicates=int(bootstrap["replicates"]),
            bootstrap_seed=int(bootstrap["metadata_design_seed"]),
            require_formal_bootstrap_binding=True,
            claim_boundary="V9_2_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
        ),
    )


def _verify_all_state(
    matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    matrix_snapshots: Sequence[bytes],
    eligibilities: Sequence[preparer_v9.FrozenTextEligibility],
    eligibility_snapshots: Sequence[bytes],
    *,
    message: str,
) -> None:
    validator_v9._verify_feature_bundle_unchanged(
        matrices, matrix_snapshots, error_message=message
    )
    validator_v9._verify_eligibility_bundle_unchanged(
        eligibilities, eligibility_snapshots, error_message=message
    )


def _fit_matrix_models(
    *,
    train: Sequence[preparer_v9.FrozenFeatureMatrix],
    development: Sequence[preparer_v9.FrozenFeatureMatrix],
    train_labels: np.ndarray,
    development_labels: np.ndarray,
    train_mask: np.ndarray | None,
    development_mask: np.ndarray | None,
    policy: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    scores_by_name: dict[str, np.ndarray] = {}
    metrics_by_name: dict[str, dict[str, Any]] = {}
    for train_value, development_value in zip(train, development, strict=True):
        train_x = train_value.values if train_mask is None else train_value.values[train_mask]
        development_x = (
            development_value.values
            if development_mask is None
            else development_value.values[development_mask]
        )
        scores = validator_v9._fit_probe_models(
            train_x=train_x,
            train_y=train_labels,
            development_x=development_x,
            policy=policy,
        )
        if tuple(scores) != MODEL_KEYS:
            raise QualityProbeValidationV92Error("Probe model key order drift")
        for model_key, score in scores.items():
            name = f"{development_value.view}::{model_key}"
            score = np.ascontiguousarray(score, dtype=np.dtype("<f8"))
            score.setflags(write=False)
            scores_by_name[name] = score
            metrics_by_name[name] = {
                "symmetric_auc": validator_v9.symmetric_auc(
                    development_labels, score
                ),
                "average_precision": float(
                    average_precision_score(development_labels, score)
                ),
                "prediction_vector_sha256": validator_v9._vector_sha256(score),
            }
    return scores_by_name, metrics_by_name


def _bootstrap_one_model_vectors(
    *,
    labels: np.ndarray,
    row_world_uids: Sequence[str],
    ordered_world_uids: Sequence[str],
    scores: np.ndarray,
    baseline: float,
    draws: np.ndarray,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    worlds = tuple(ordered_world_uids)
    if (
        not worlds
        or len(worlds) != len(set(worlds))
        or set(worlds) != set(row_world_uids)
        or scores.shape != labels.shape
        or not np.isfinite(scores).all()
        or draws.ndim != 2
        or draws.shape[1] != len(worlds)
        or draws.dtype != np.dtype("<i8")
        or batch_size != 16
    ):
        raise QualityProbeValidationV92Error("Descriptive bootstrap input drift")
    world_index = {world_uid: index for index, world_uid in enumerate(worlds)}
    row_world_index = np.fromiter(
        (world_index[value] for value in row_world_uids),
        dtype=np.int64,
        count=len(row_world_uids),
    )
    multiplicities = np.asarray(
        [np.bincount(draw, minlength=len(worlds)) for draw in draws],
        dtype=np.float64,
    )
    unique_scores, group_index = np.unique(scores, return_inverse=True)
    use_dense = len(unique_scores) * len(worlds) <= len(labels)
    if use_dense:
        group_positive = np.zeros((len(unique_scores), len(worlds)), dtype=np.float64)
        group_negative = np.zeros_like(group_positive)
        np.add.at(group_positive, (group_index, row_world_index), labels)
        np.add.at(group_negative, (group_index, row_world_index), 1 - labels)
    else:
        descending = np.argsort(scores, kind="stable")[::-1]
        descending_scores = scores[descending]
        descending_labels = labels[descending].astype(np.float64, copy=False)
        descending_world_index = row_world_index[descending]
        group_starts = np.flatnonzero(
            np.concatenate(
                (np.asarray([True]), descending_scores[1:] != descending_scores[:-1])
            )
        )
    auc_vector = np.empty(len(draws), dtype=np.float64)
    ap_vector = np.empty(len(draws), dtype=np.float64)
    for start in range(0, len(draws), batch_size):
        stop = min(start + batch_size, len(draws))
        selected = multiplicities[start:stop]
        if use_dense:
            positive = selected @ group_positive.T
            negative = selected @ group_negative.T
            negative_before = np.cumsum(negative, axis=1) - negative
            positive_desc = positive[:, ::-1]
            total_desc = (positive + negative)[:, ::-1]
        else:
            row_weights = selected[:, descending_world_index]
            positive_rows = row_weights * descending_labels
            negative_rows = row_weights * (1.0 - descending_labels)
            positive = np.add.reduceat(positive_rows, group_starts, axis=1)
            negative = np.add.reduceat(negative_rows, group_starts, axis=1)
            negative_before = negative.sum(axis=1, keepdims=True) - np.cumsum(
                negative, axis=1
            )
            positive_desc = positive
            total_desc = positive + negative
        total_positive = positive.sum(axis=1)
        total_negative = negative.sum(axis=1)
        if np.any(total_positive <= 0) or np.any(total_negative <= 0):
            raise QualityProbeValidationV92Error("Bootstrap replicate lost a class")
        auc = np.sum(
            positive * (negative_before + 0.5 * negative), axis=1
        ) / (total_positive * total_negative)
        auc_vector[start:stop] = np.maximum(auc, 1.0 - auc)
        cumulative_positive = np.cumsum(positive_desc, axis=1)
        cumulative_total = np.cumsum(total_desc, axis=1)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        ap_vector[start:stop] = (
            np.sum(positive_desc * precision, axis=1) / total_positive
        ) - baseline
    if not np.isfinite(auc_vector).all() or not np.isfinite(ap_vector).all():
        raise QualityProbeValidationV92Error("Descriptive bootstrap output is nonfinite")
    return (
        np.ascontiguousarray(auc_vector, dtype=np.dtype("<f8")),
        np.ascontiguousarray(ap_vector, dtype=np.dtype("<f8")),
    )


def _descriptive_receipt(
    *,
    train: Sequence[preparer_v9.FrozenFeatureMatrix],
    development: Sequence[preparer_v9.FrozenFeatureMatrix],
    train_labels: np.ndarray,
    development_labels: np.ndarray,
    train_mask: np.ndarray,
    development_mask: np.ndarray,
    ordered_development_worlds: Sequence[str],
    policy: Mapping[str, Any],
    design: validator_v9.ProbeFamilyDesign,
) -> dict[str, Any]:
    scores, metrics = _fit_matrix_models(
        train=train,
        development=development,
        train_labels=train_labels,
        development_labels=development_labels,
        train_mask=train_mask,
        development_mask=development_mask,
        policy=policy,
    )
    draws = validator_v9.generate_bootstrap_draws(
        replicates=design.bootstrap_replicates,
        world_count=design.expected_worlds,
        seed=design.bootstrap_seed,
    )
    draws_sha256 = common.sha256_bytes(draws.tobytes(order="C"))
    if design.require_formal_bootstrap_binding and draws_sha256 != (
        validator_v9.FORMAL_BOOTSTRAP_SHA256
    ):
        raise QualityProbeValidationV92Error("Formal bootstrap draw hash drift")
    development_world_uids_full = tuple(
        world_uid for world_uid, _pair_uid in development[0].row_keys
    )
    development_world_uids = tuple(
        value
        for value, keep in zip(
            development_world_uids_full, development_mask, strict=True
        )
        if keep
    )
    bootstrap_vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with threadpool_limits(limits=1):
        for name in sorted(scores, key=lambda value: value.encode("utf-8")):
            auc_vector, ap_uplift_vector = _bootstrap_one_model_vectors(
                labels=development_labels,
                row_world_uids=development_world_uids,
                ordered_world_uids=ordered_development_worlds,
                scores=scores[name],
                baseline=design.average_precision_baseline,
                draws=draws,
            )
            bootstrap_vectors[name] = (auc_vector, ap_uplift_vector)
            metrics[name]["bootstrap_symmetric_auc_vector_sha256"] = (
                validator_v9._vector_sha256(auc_vector)
            )
            metrics[name]["bootstrap_average_precision_uplift_vector_sha256"] = (
                validator_v9._vector_sha256(ap_uplift_vector)
            )
            metrics[name]["bootstrap_symmetric_auc_interval_95"] = [
                float(np.quantile(auc_vector, 0.025, method="linear")),
                float(np.quantile(auc_vector, 0.975, method="linear")),
            ]
            ap_vector = ap_uplift_vector + design.average_precision_baseline
            metrics[name]["bootstrap_average_precision_interval_95"] = [
                float(np.quantile(ap_vector, 0.025, method="linear")),
                float(np.quantile(ap_vector, 0.975, method="linear")),
            ]
            metrics[name]["channel_differences"] = {}
    for view in preparer_v9.text_views.VIEW_ORDER:
        for model_key in MODEL_KEYS:
            for direction, left_surface, right_surface in CHANNEL_DIFFERENCE_DIRECTIONS:
                left_name = f"{left_surface}::{view}::{model_key}"
                right_name = f"{right_surface}::{view}::{model_key}"
                left_auc, left_ap = bootstrap_vectors[left_name]
                right_auc, right_ap = bootstrap_vectors[right_name]
                auc_difference = left_auc - right_auc
                ap_difference = left_ap - right_ap
                metrics[left_name]["channel_differences"][direction] = {
                    "right_model_name": right_name,
                    "symmetric_auc_point_difference": float(
                        metrics[left_name]["symmetric_auc"]
                        - metrics[right_name]["symmetric_auc"]
                    ),
                    "average_precision_point_difference": float(
                        metrics[left_name]["average_precision"]
                        - metrics[right_name]["average_precision"]
                    ),
                    "bootstrap_symmetric_auc_difference_vector_sha256": (
                        validator_v9._vector_sha256(auc_difference)
                    ),
                    "bootstrap_average_precision_difference_vector_sha256": (
                        validator_v9._vector_sha256(ap_difference)
                    ),
                    "bootstrap_symmetric_auc_difference_interval_95": [
                        float(np.quantile(auc_difference, 0.025, method="linear")),
                        float(np.quantile(auc_difference, 0.975, method="linear")),
                    ],
                    "bootstrap_average_precision_difference_interval_95": [
                        float(np.quantile(ap_difference, 0.025, method="linear")),
                        float(np.quantile(ap_difference, 0.975, method="linear")),
                    ],
                }
    return {
        "status": "DESCRIPTIVE_NOT_A_QUALITY_GATE",
        "qualification_role": gate_registry.DESCRIPTIVE,
        "model_count": len(metrics),
        "models": metrics,
        "draws_raw_i8_c_sha256": draws_sha256,
        "gate_failures": [],
        "passed_field_forbidden": True,
    }


def _hard_family_receipt(
    *,
    family: str,
    train: Sequence[preparer_v9.FrozenFeatureMatrix],
    development: Sequence[preparer_v9.FrozenFeatureMatrix],
    train_labels: np.ndarray,
    development_labels: np.ndarray,
    train_mask: np.ndarray | None,
    development_mask: np.ndarray | None,
    ordered_development_worlds: Sequence[str],
    policy: Mapping[str, Any],
    design: validator_v9.ProbeFamilyDesign,
) -> dict[str, Any]:
    single = validator_v9._single_feature_maximum(
        development, development_labels, development_mask
    )
    scores, metrics = _fit_matrix_models(
        train=train,
        development=development,
        train_labels=train_labels,
        development_labels=development_labels,
        train_mask=train_mask,
        development_mask=development_mask,
        policy=policy,
    )
    model_auc = max(value["symmetric_auc"] for value in metrics.values())
    model_ap_uplift = max(
        value["average_precision"] - design.average_precision_baseline
        for value in metrics.values()
    )
    development_world_uids_full = tuple(
        world_uid for world_uid, _pair_uid in development[0].row_keys
    )
    development_world_uids = (
        development_world_uids_full
        if development_mask is None
        else tuple(
            value
            for value, keep in zip(
                development_world_uids_full, development_mask, strict=True
            )
            if keep
        )
    )
    draws = validator_v9.generate_bootstrap_draws(
        replicates=design.bootstrap_replicates,
        world_count=design.expected_worlds,
        seed=design.bootstrap_seed,
    )
    draws_sha256 = common.sha256_bytes(draws.tobytes(order="C"))
    if design.require_formal_bootstrap_binding and draws_sha256 != (
        validator_v9.FORMAL_BOOTSTRAP_SHA256
    ):
        raise QualityProbeValidationV92Error("Formal bootstrap draw hash drift")
    with threadpool_limits(limits=1):
        bootstrap = validator_v9._bootstrap_family_upper(
            labels=development_labels,
            row_world_uids=development_world_uids,
            ordered_world_uids=ordered_development_worlds,
            score_family=scores,
            baseline=design.average_precision_baseline,
            draws=draws,
        )
    values = {
        "maximum_single_feature_symmetric_auc": float(
            single["maximum_symmetric_auc"]
        ),
        "maximum_family_symmetric_auc": float(model_auc),
        "maximum_family_average_precision_uplift": float(model_ap_uplift),
        "bootstrap_95_upper_symmetric_auc": float(
            bootstrap["symmetric_auc_95_upper"]
        ),
        "bootstrap_95_upper_average_precision_uplift": float(
            bootstrap["average_precision_uplift_95_upper"]
        ),
    }
    thresholds = policy["quality_gates"]
    checks = {
        name: {
            "observed": observed,
            "threshold": float(thresholds[name]),
            "passed": observed <= float(thresholds[name]),
        }
        for name, observed in values.items()
    }
    return {
        "family": family,
        "single_feature": single,
        "model_family": {
            "model_count": len(metrics),
            "models": metrics,
            "prediction_vector_bundle_sha256": common.canonical_sha256(
                [
                    [name, metrics[name]["prediction_vector_sha256"]]
                    for name in sorted(metrics, key=lambda value: value.encode("utf-8"))
                ]
            ),
        },
        "input_matrix_bundle_sha256": common.canonical_sha256(
            {
                "train": [value.commitment_sha256 for value in train],
                "development": [
                    value.commitment_sha256 for value in development
                ],
            }
        ),
        "bootstrap": bootstrap,
        "gate_checks": checks,
        "gate_failures": [name for name, value in checks.items() if not value["passed"]],
        "draws_raw_i8_c_sha256": draws_sha256,
    }


def _n_a_fields(spec: Mapping[str, Any], row: dict[str, Any]) -> None:
    requirements = {
        "input_matrix_sha256": "input_matrix",
        "prediction_vector_sha256": "prediction_vector",
        "bootstrap_family_maxima_vector_sha256": "bootstrap_family_maxima_vector",
    }
    for field, requirement in requirements.items():
        reason_field = field.removesuffix("_sha256") + "_not_applicable_reason"
        if spec[requirement] == gate_registry.REQUIRED:
            row[reason_field] = None
        else:
            row[field] = None
            row[reason_field] = complete_evidence._not_applicable_reason(spec, field)


def build_numerical_gate_observations(
    *,
    descriptive: Mapping[str, Any],
    counterfactual_text: Mapping[str, Any],
    code_and_slot: Mapping[str, Any],
    train_descriptive_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    development_descriptive_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    train_matrix_by_view = {value.view: value for value in train_descriptive_matrices}
    development_matrix_by_view = {
        value.view: value for value in development_descriptive_matrices
    }
    for spec in gate_registry.GATE_REGISTRY:
        gate_id = str(spec["gate_id"])
        if spec["qualification_role"] != gate_registry.DESCRIPTIVE:
            continue
        parts = gate_id.split(".")
        surface, view, model_key = parts[2], parts[3], parts[4]
        model_name = f"{surface}::{view}::{model_key}"
        metric = descriptive["models"][model_name]
        row = {
            "gate_id": gate_id,
            "observed": {
                "symmetric_auc": metric["symmetric_auc"],
                "average_precision": metric["average_precision"],
                "bootstrap_symmetric_auc_interval_95": metric[
                    "bootstrap_symmetric_auc_interval_95"
                ],
                "bootstrap_average_precision_interval_95": metric[
                    "bootstrap_average_precision_interval_95"
                ],
                "bootstrap_symmetric_auc_vector_sha256": metric[
                    "bootstrap_symmetric_auc_vector_sha256"
                ],
                "bootstrap_average_precision_uplift_vector_sha256": metric[
                    "bootstrap_average_precision_uplift_vector_sha256"
                ],
                "channel_differences": metric["channel_differences"],
            },
            "gate_status": gate_registry.NOT_APPLICABLE,
            "input_matrix_sha256": common.canonical_sha256(
                {
                    "train": train_matrix_by_view[
                        f"{surface}::{view}"
                    ].commitment_sha256,
                    "development": development_matrix_by_view[
                        f"{surface}::{view}"
                    ].commitment_sha256,
                }
            ),
            "prediction_vector_sha256": metric["prediction_vector_sha256"],
        }
        _n_a_fields(spec, row)
        observations[gate_id] = row
    for prefix, receipt in (
        ("text_deranged", counterfactual_text),
        ("code_slot", code_and_slot),
    ):
        for metric_name, check in receipt["gate_checks"].items():
            gate_id = f"hard.{prefix}.{metric_name}"
            spec = next(
                value
                for value in gate_registry.GATE_REGISTRY
                if value["gate_id"] == gate_id
            )
            is_bootstrap_auc = metric_name == "bootstrap_95_upper_symmetric_auc"
            is_bootstrap_ap = (
                metric_name == "bootstrap_95_upper_average_precision_uplift"
            )
            row = {
                "gate_id": gate_id,
                "observed": float(check["observed"]),
                "gate_status": "PASS" if check["passed"] else "FAIL",
                "passed": bool(check["passed"]),
                "input_matrix_sha256": receipt["input_matrix_bundle_sha256"],
            }
            if spec["prediction_vector"] == gate_registry.REQUIRED:
                row["prediction_vector_sha256"] = receipt["model_family"][
                    "prediction_vector_bundle_sha256"
                ]
            if is_bootstrap_auc:
                row["bootstrap_family_maxima_vector_sha256"] = receipt[
                    "bootstrap"
                ]["family_max_symmetric_auc_vector_sha256"]
            elif is_bootstrap_ap:
                row["bootstrap_family_maxima_vector_sha256"] = receipt[
                    "bootstrap"
                ]["family_max_average_precision_uplift_vector_sha256"]
            _n_a_fields(spec, row)
            observations[gate_id] = row
    return observations


def _calculate_all_families(
    *,
    descriptive_train: Sequence[preparer_v9.FrozenFeatureMatrix],
    descriptive_dev: Sequence[preparer_v9.FrozenFeatureMatrix],
    hard_train: Sequence[preparer_v9.FrozenFeatureMatrix],
    hard_dev: Sequence[preparer_v9.FrozenFeatureMatrix],
    code_train: Sequence[preparer_v9.FrozenFeatureMatrix],
    code_dev: Sequence[preparer_v9.FrozenFeatureMatrix],
    train_mask: np.ndarray,
    development_mask: np.ndarray,
    train_labels_full: np.ndarray,
    development_labels_full: np.ndarray,
    ordered_development_worlds: Sequence[str],
    policy: Mapping[str, Any],
    designs: ProbeDesignsV92,
    verify_between_families: Callable[[str], None],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    train_labels = train_labels_full[train_mask]
    development_labels = development_labels_full[development_mask]
    descriptive = _descriptive_receipt(
        train=descriptive_train,
        development=descriptive_dev,
        train_labels=train_labels,
        development_labels=development_labels,
        train_mask=train_mask,
        development_mask=development_mask,
        ordered_development_worlds=ordered_development_worlds,
        policy=policy,
        design=designs.descriptive,
    )
    verify_between_families(
        "V9.2 state changed between descriptive and hard text families"
    )
    hard_text = _hard_family_receipt(
        family="counterfactual_text",
        train=hard_train,
        development=hard_dev,
        train_labels=train_labels,
        development_labels=development_labels,
        train_mask=train_mask,
        development_mask=development_mask,
        ordered_development_worlds=ordered_development_worlds,
        policy=policy,
        design=designs.counterfactual_text,
    )
    verify_between_families("V9.2 state changed before the code/slot family")
    code_and_slot = _hard_family_receipt(
        family="public_code_private_slot",
        train=code_train,
        development=code_dev,
        train_labels=train_labels_full,
        development_labels=development_labels_full,
        train_mask=None,
        development_mask=None,
        ordered_development_worlds=ordered_development_worlds,
        policy=policy,
        design=designs.code_and_slot,
    )
    family_receipts = {
        "original_author_descriptive": descriptive,
        "counterfactual_text": hard_text,
        "public_code_private_slot": code_and_slot,
    }
    observations = build_numerical_gate_observations(
        descriptive=descriptive,
        counterfactual_text=hard_text,
        code_and_slot=code_and_slot,
        train_descriptive_matrices=descriptive_train,
        development_descriptive_matrices=descriptive_dev,
    )
    return family_receipts, observations


def evaluate_fixture_probe_families(
    *,
    text_train_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    text_development_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    code_train_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    code_development_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    train_text_eligibility: preparer_v9.FrozenTextEligibility,
    development_text_eligibility: preparer_v9.FrozenTextEligibility,
    truth_loader: Callable[[str], Sequence[Mapping[str, Any]]],
    policy: Mapping[str, Any],
    designs: ProbeDesignsV92,
) -> dict[str, Any]:
    """Exercise the complete numerical topology at no more than three worlds."""

    validator_v9._validate_runtime()
    if (
        not callable(truth_loader)
        or any(
            design.expected_worlds > 3 or design.bootstrap_replicates > 31
            for design in (
                designs.descriptive,
                designs.counterfactual_text,
                designs.code_and_slot,
            )
        )
    ):
        raise QualityProbeValidationV92Error("V9.2 fixture boundary widened")
    descriptive_train_source, hard_train_source = preparer_v9_2.split_text_matrix_roles(
        text_train_matrices
    )
    descriptive_dev_source, hard_dev_source = preparer_v9_2.split_text_matrix_roles(
        text_development_matrices
    )
    descriptive_train, descriptive_dev = validator_v9._validate_matrix_sets(
        descriptive_train_source, descriptive_dev_source, designs.descriptive
    )
    hard_train, hard_dev = validator_v9._validate_matrix_sets(
        hard_train_source, hard_dev_source, designs.counterfactual_text
    )
    code_train, code_dev = validator_v9._validate_matrix_sets(
        code_train_matrices, code_development_matrices, designs.code_and_slot
    )
    if not (
        descriptive_train[0].row_keys
        == hard_train[0].row_keys
        == code_train[0].row_keys
        and descriptive_dev[0].row_keys
        == hard_dev[0].row_keys
        == code_dev[0].row_keys
    ):
        raise QualityProbeValidationV92Error("Cross-family pair order drift")
    train_mask_source = validator_v9._validate_eligibility(
        train_text_eligibility,
        row_keys=hard_train[0].row_keys,
        excluded_pairs_per_world=designs.counterfactual_text.excluded_pairs_per_world,
    )
    development_mask_source = validator_v9._validate_eligibility(
        development_text_eligibility,
        row_keys=hard_dev[0].row_keys,
        excluded_pairs_per_world=designs.counterfactual_text.excluded_pairs_per_world,
    )
    train_mask = np.array(train_mask_source, dtype=bool, order="C", copy=True)
    development_mask = np.array(
        development_mask_source, dtype=bool, order="C", copy=True
    )
    train_mask.setflags(write=False)
    development_mask.setflags(write=False)
    all_matrices = (
        *descriptive_train,
        *hard_train,
        *code_train,
        *descriptive_dev,
        *hard_dev,
        *code_dev,
    )
    matrix_snapshots = tuple(
        preparer_v9.current_feature_matrix_commitment_json(value)
        for value in all_matrices
    )
    eligibilities = (train_text_eligibility, development_text_eligibility)
    eligibility_snapshots = tuple(
        preparer_v9.current_text_eligibility_commitment_json(value)
        for value in eligibilities
    )
    policy_snapshot = gate_registry.canonical_json_bytes(policy)
    loader_calls: Counter[str] = Counter()

    def load_once(split: str) -> tuple[Mapping[str, Any], ...]:
        if split not in {"train", "development"}:
            raise QualityProbeValidationV92Error("Audit truth loader call attempted")
        loader_calls[split] += 1
        if loader_calls[split] != 1:
            raise QualityProbeValidationV92Error("Truth loader called more than once")
        return tuple(truth_loader(split))

    train_truth = load_once("train")
    development_truth = load_once("development")
    _verify_all_state(
        all_matrices,
        matrix_snapshots,
        eligibilities,
        eligibility_snapshots,
        message="V9.2 feature/mask state changed after truth open",
    )
    if gate_registry.canonical_json_bytes(policy) != policy_snapshot:
        raise QualityProbeValidationV92Error("V9.2 policy changed after truth open")
    train_labels_full = validator_v9._load_and_validate_truth(
        split="train",
        truth_loader=lambda _split: train_truth,
        row_keys=hard_train[0].row_keys,
        design=designs.counterfactual_text,
        eligibility=train_mask,
    )
    development_labels_full = validator_v9._load_and_validate_truth(
        split="development",
        truth_loader=lambda _split: development_truth,
        row_keys=hard_dev[0].row_keys,
        design=designs.counterfactual_text,
        eligibility=development_mask,
    )
    ordered_development_worlds = validator_v9._ordered_worlds(hard_dev[0].row_keys)
    def verify_between(message: str) -> None:
        _verify_all_state(
            all_matrices,
            matrix_snapshots,
            eligibilities,
            eligibility_snapshots,
            message=message,
        )

    family_receipts, observations = _calculate_all_families(
        descriptive_train=descriptive_train,
        descriptive_dev=descriptive_dev,
        hard_train=hard_train,
        hard_dev=hard_dev,
        code_train=code_train,
        code_dev=code_dev,
        train_mask=train_mask,
        development_mask=development_mask,
        train_labels_full=train_labels_full,
        development_labels_full=development_labels_full,
        ordered_development_worlds=ordered_development_worlds,
        policy=policy,
        designs=designs,
        verify_between_families=verify_between,
    )
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": "FIXTURE_COMPLETE_NO_DATASET_CONCLUSION",
        "claim_boundary": "FIXTURE_ONLY_NO_DATASET_CONCLUSION",
        "gate_registry_sha256": gate_registry.GATE_REGISTRY_SHA256,
        "family_receipts": family_receipts,
        "numerical_gate_observations": observations,
        "truth_loader_call_counts": {
            "train": loader_calls["train"],
            "development": loader_calls["development"],
            "audit_a": loader_calls["audit_a"],
            "audit_b": loader_calls["audit_b"],
        },
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
    }
    receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
    return receipt


def evaluate_formal_probe_families(
    *,
    text_train_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    text_development_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    code_train_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    code_development_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    train_text_eligibility: preparer_v9.FrozenTextEligibility,
    development_text_eligibility: preparer_v9.FrozenTextEligibility,
    dataset_root: Path,
    root_manifest_pin: truth_capability.RootManifestPin,
    policy: Mapping[str, Any],
    run_authorization: Mapping[str, Any],
    verify_label_free_bytes: Callable[[], None],
) -> dict[str, Any]:
    """Open train/development truth once after all 30 matrices are frozen."""

    validator_v9._validate_runtime()
    # The immutable scientific/quality policy deliberately authorizes no
    # execution.  A future, separately reviewed one-shot run receipt owns the
    # design-root pin and the narrowly scoped truth-read capability.  Keeping
    # those bytes out of ``policy`` prevents a post-freeze policy rewrite.
    authorization = run_authorization.get("capabilities", {})
    if (
        authorization.get("quality_audit_run") is not True
        or authorization.get("metric_generation") is not True
        or authorization.get("audit_a_b_truth_open") is not False
        or authorization.get("model_training") is not False
        or authorization.get("formal_500_by_4") is not False
    ):
        raise QualityProbeValidationV92Error(
            "V9.2 formal quality capability is absent or over-broad"
        )
    designs = formal_designs(policy)
    descriptive_train_source, hard_train_source = preparer_v9_2.split_text_matrix_roles(
        text_train_matrices
    )
    descriptive_dev_source, hard_dev_source = preparer_v9_2.split_text_matrix_roles(
        text_development_matrices
    )
    descriptive_train, descriptive_dev = validator_v9._validate_matrix_sets(
        descriptive_train_source, descriptive_dev_source, designs.descriptive
    )
    hard_train, hard_dev = validator_v9._validate_matrix_sets(
        hard_train_source, hard_dev_source, designs.counterfactual_text
    )
    code_train, code_dev = validator_v9._validate_matrix_sets(
        code_train_matrices, code_development_matrices, designs.code_and_slot
    )
    if not (
        descriptive_train[0].row_keys
        == hard_train[0].row_keys
        == code_train[0].row_keys
        and descriptive_dev[0].row_keys
        == hard_dev[0].row_keys
        == code_dev[0].row_keys
    ):
        raise QualityProbeValidationV92Error("Formal cross-family pair order drift")
    train_mask_source = validator_v9._validate_eligibility(
        train_text_eligibility,
        row_keys=hard_train[0].row_keys,
        excluded_pairs_per_world=designs.counterfactual_text.excluded_pairs_per_world,
    )
    development_mask_source = validator_v9._validate_eligibility(
        development_text_eligibility,
        row_keys=hard_dev[0].row_keys,
        excluded_pairs_per_world=designs.counterfactual_text.excluded_pairs_per_world,
    )
    train_mask = np.array(train_mask_source, dtype=bool, order="C", copy=True)
    development_mask = np.array(
        development_mask_source, dtype=bool, order="C", copy=True
    )
    train_mask.setflags(write=False)
    development_mask.setflags(write=False)
    all_matrices = (
        *descriptive_train,
        *hard_train,
        *code_train,
        *descriptive_dev,
        *hard_dev,
        *code_dev,
    )
    matrix_snapshots = tuple(
        preparer_v9.current_feature_matrix_commitment_json(value)
        for value in all_matrices
    )
    eligibilities = (train_text_eligibility, development_text_eligibility)
    eligibility_snapshots = tuple(
        preparer_v9.current_text_eligibility_commitment_json(value)
        for value in eligibilities
    )
    policy_snapshot = gate_registry.canonical_json_bytes(policy)
    expected_root_binding = run_authorization.get("design_root_manifest")
    if not isinstance(expected_root_binding, Mapping) or set(
        expected_root_binding
    ) != {"path", "size_bytes", "sha256", "canonical_self_hash"}:
        raise QualityProbeValidationV92Error("Formal root binding is absent")
    truth = truth_capability.FormalTrainDevelopmentTruthCapability.from_pinned_design_root(
        dataset_root=dataset_root,
        root_manifest_pin=root_manifest_pin,
    )
    if truth.root_binding() != dict(expected_root_binding):
        raise QualityProbeValidationV92Error("Formal truth root binding drift")
    truth_pins = truth._begin_bound_transaction(
        expected_root_binding=expected_root_binding
    )
    truth_rows: dict[str, Sequence[Mapping[str, Any]]] = {}
    for split in truth_capability.SUPERVISED_SPLITS:
        rows, split_receipt = truth_capability._read_pinned_truth_csv(
            truth_pins[split]
        )
        truth._record_split_receipt(split=split, receipt=split_receipt)
        truth_rows[split] = rows
    if not callable(verify_label_free_bytes):
        raise QualityProbeValidationV92Error(
            "Formal label-free byte re-verifier is absent"
        )
    verify_label_free_bytes()
    _verify_all_state(
        all_matrices,
        matrix_snapshots,
        eligibilities,
        eligibility_snapshots,
        message="Formal V9.2 feature/mask state changed after truth open",
    )
    if gate_registry.canonical_json_bytes(policy) != policy_snapshot:
        raise QualityProbeValidationV92Error("Formal V9.2 policy changed after truth open")
    train_labels_full = validator_v9._load_and_validate_truth(
        split="train",
        truth_loader=lambda _split: truth_rows["train"],
        row_keys=hard_train[0].row_keys,
        design=designs.counterfactual_text,
        eligibility=train_mask,
    )
    development_labels_full = validator_v9._load_and_validate_truth(
        split="development",
        truth_loader=lambda _split: truth_rows["development"],
        row_keys=hard_dev[0].row_keys,
        design=designs.counterfactual_text,
        eligibility=development_mask,
    )
    ordered_development_worlds = validator_v9._ordered_worlds(hard_dev[0].row_keys)

    def verify_between(message: str) -> None:
        _verify_all_state(
            all_matrices,
            matrix_snapshots,
            eligibilities,
            eligibility_snapshots,
            message=message,
        )
        if gate_registry.canonical_json_bytes(policy) != policy_snapshot:
            raise QualityProbeValidationV92Error(
                "Formal policy changed between probe families"
            )

    family_receipts, observations = _calculate_all_families(
        descriptive_train=descriptive_train,
        descriptive_dev=descriptive_dev,
        hard_train=hard_train,
        hard_dev=hard_dev,
        code_train=code_train,
        code_dev=code_dev,
        train_mask=train_mask,
        development_mask=development_mask,
        train_labels_full=train_labels_full,
        development_labels_full=development_labels_full,
        ordered_development_worlds=ordered_development_worlds,
        policy=policy,
        designs=designs,
        verify_between_families=verify_between,
    )
    truth_receipt = truth.aggregate_receipt()
    truth_rows.clear()
    if any(
        truth_receipt[split][field] != 0
        for split in ("audit_a", "audit_b")
        for field in ("file_open_count", "byte_read_count", "materialized_row_count")
    ):
        raise QualityProbeValidationV92Error("Audit A/B truth access count drift")
    hard_failures = [
        gate_id
        for gate_id, value in observations.items()
        if gate_id.startswith("hard.") and value.get("passed") is False
    ]
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": "PASS" if not hard_failures else "DATASET_INVALIDATED",
        "claim_boundary": "V9_2_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
        "gate_registry_sha256": gate_registry.GATE_REGISTRY_SHA256,
        "family_receipts": family_receipts,
        "numerical_gate_observations": observations,
        "truth_file_access": truth_receipt,
        "label_free_byte_reverification_call_count": 1,
        "truth_and_order_bindings": {
            "train_label_vector_sha256": common.sha256_bytes(
                np.ascontiguousarray(train_labels_full, dtype=np.dtype("<i8")).tobytes(
                    order="C"
                )
            ),
            "development_label_vector_sha256": common.sha256_bytes(
                np.ascontiguousarray(
                    development_labels_full, dtype=np.dtype("<i8")
                ).tobytes(order="C")
            ),
            "train_full_row_order_sha256": common.canonical_sha256(
                [list(value) for value in hard_train[0].row_keys]
            ),
            "development_full_row_order_sha256": common.canonical_sha256(
                [list(value) for value in hard_dev[0].row_keys]
            ),
            "train_eligible_row_order_sha256": common.canonical_sha256(
                [
                    list(value)
                    for value, keep in zip(
                        hard_train[0].row_keys, train_mask, strict=True
                    )
                    if keep
                ]
            ),
            "development_eligible_row_order_sha256": common.canonical_sha256(
                [
                    list(value)
                    for value, keep in zip(
                        hard_dev[0].row_keys, development_mask, strict=True
                    )
                    if keep
                ]
            ),
            "train_text_eligibility_mask_sha256": common.sha256_bytes(
                np.ascontiguousarray(train_mask, dtype=np.uint8).tobytes(order="C")
            ),
            "development_text_eligibility_mask_sha256": common.sha256_bytes(
                np.ascontiguousarray(development_mask, dtype=np.uint8).tobytes(
                    order="C"
                )
            ),
        },
        "structure_metric_values": {
            "full_pair_count_mismatch_world_count": 0,
            "eligible_pair_count_mismatch_world_count": 0,
            "positive_pair_count_mismatch_world_count": 0,
            "excluded_positive_pair_count": 0,
            "cross_branch_label_vector_mismatch_count": 0,
            "train_truth_open_count": int(
                truth_receipt["train"]["file_open_count"]
            ),
            "development_truth_open_count": int(
                truth_receipt["development"]["file_open_count"]
            ),
        },
        "hard_gate_failures_in_registry_order": [
            gate_id for gate_id in gate_registry.GATE_IDS if gate_id in hard_failures
        ],
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
    }
    receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
    return receipt
