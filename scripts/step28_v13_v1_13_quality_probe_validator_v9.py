#!/usr/bin/env python3
"""Privileged train/development validator for frozen v9 shortcut probes.

Feature matrices must already be frozen by the label-free preparer.  Fixture
tests alone may supply a bounded truth callback.  Formal evaluation accepts
only the exact policy-pinned design root and manifest pin, constructs its
one-shot truth capability internally after every label-free input is frozen,
opens train and development together, and returns aggregate metrics and hashes
rather than row-level labels or predictions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import step28_v13_v1_13_quality_channel_policy_v9 as channel_policy
import step28_v13_v1_13_quality_probe_preparer_v9 as preparer
import step28_v13_v1_13_quality_text_probe_views_v9 as text_views
import step28_v13_v1_13_quality_truth_capability_v9 as truth_capability


VERSION = "2026-08-14-step28-v13-v1-13-quality-probe-validator-v9"
TRUTH_FIELDS = ("canonical_pair_uid", "world_uid", "label")
FORMAL_BOOTSTRAP_SHA256 = (
    "111b1338cc607c6bd78bad88efe47606ffa2230e9cc764eec940e84f86e56661"
)


class QualityProbeValidationError(ValueError):
    """Raised on execution, isolation, schema, or numerical drift."""


class QualityProbeDatasetGateError(QualityProbeValidationError):
    """Raised only when persisted preregistered data violates a hard gate."""


@dataclass(frozen=True)
class ProbeFamilyDesign:
    family: str
    view_widths: tuple[tuple[str, int], ...]
    expected_views: int
    expected_total_features: int
    expected_column_name_hashes: tuple[tuple[str, str], ...] | None
    expected_worlds: int
    pairs_per_world: int
    positives_per_world: int
    excluded_pairs_per_world: int
    average_precision_baseline: float
    bootstrap_replicates: int
    bootstrap_seed: int
    require_formal_bootstrap_binding: bool
    claim_boundary: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _vector_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    if array.ndim != 1 or not np.isfinite(array).all():
        raise QualityProbeValidationError("Prediction/metric vector is invalid")
    return _sha256_bytes(array.tobytes(order="C"))


def _column_names_sha256(values: Sequence[str]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(values)))


def _validate_runtime() -> None:
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    expected = {
        "python": "3.10.11",
        "numpy": "2.2.6",
        "scikit_learn": "1.7.2",
    }
    if observed != expected:
        raise QualityProbeValidationError(
            f"Frozen quality runtime drift: expected={expected} observed={observed}"
        )


def symmetric_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if (
        labels.ndim != 1
        or scores.shape != labels.shape
        or set(np.unique(labels).tolist()) != {0, 1}
        or not np.isfinite(scores).all()
    ):
        raise QualityProbeValidationError("Metric vector shape/class drift")
    auc = float(roc_auc_score(labels, scores))
    return max(auc, 1.0 - auc)


def _ordered_worlds(row_keys: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    worlds: list[str] = []
    active: str | None = None
    closed: set[str] = set()
    for world_uid, _pair_uid in row_keys:
        if world_uid != active:
            if active is not None:
                closed.add(active)
            if world_uid in closed:
                raise QualityProbeValidationError("Frozen world rows are not contiguous")
            worlds.append(world_uid)
            active = world_uid
    return tuple(worlds)


def _validate_matrix_sets(
    train_matrices: Sequence[preparer.FrozenFeatureMatrix],
    development_matrices: Sequence[preparer.FrozenFeatureMatrix],
    design: ProbeFamilyDesign,
) -> tuple[
    tuple[preparer.FrozenFeatureMatrix, ...],
    tuple[preparer.FrozenFeatureMatrix, ...],
]:
    train = tuple(train_matrices)
    development = tuple(development_matrices)
    if (
        len(train) != design.expected_views
        or len(development) != design.expected_views
        or not train
    ):
        raise QualityProbeValidationError("Probe matrix view cardinality drift")
    for value in (*train, *development):
        preparer.verify_frozen_feature_matrix(value)
        if value.family != design.family:
            raise QualityProbeValidationError("Probe matrix family drift")
    if (
        len({value.view for value in train}) != len(train)
        or len({value.view for value in development}) != len(development)
        or tuple(value.view for value in train)
        != tuple(value.view for value in development)
    ):
        raise QualityProbeValidationError("Train/development view order drift")
    observed_view_widths = tuple(
        (value.view, len(value.column_names)) for value in train
    )
    if observed_view_widths != design.view_widths:
        raise QualityProbeValidationError("Probe view name/width contract drift")
    if design.expected_column_name_hashes is not None:
        observed_hashes = tuple(
            (value.view, _column_names_sha256(value.column_names))
            for value in train
        )
        if observed_hashes != design.expected_column_name_hashes:
            raise QualityProbeValidationError(
                "Probe formal column-name commitment drift"
            )
    for train_value, development_value in zip(train, development):
        if train_value.column_names != development_value.column_names:
            raise QualityProbeValidationError("Train/development column schema drift")
    if sum(len(value.column_names) for value in train) != design.expected_total_features:
        raise QualityProbeValidationError("Probe family feature width drift")
    if any(value.row_keys != train[0].row_keys for value in train[1:]) or any(
        value.row_keys != development[0].row_keys for value in development[1:]
    ):
        raise QualityProbeValidationError("Within-split row alignment drift")
    train_worlds = _ordered_worlds(train[0].row_keys)
    development_worlds = _ordered_worlds(development[0].row_keys)
    train_pair_uids = {pair_uid for _world_uid, pair_uid in train[0].row_keys}
    development_pair_uids = {
        pair_uid for _world_uid, pair_uid in development[0].row_keys
    }
    if (
        len(train_worlds) != design.expected_worlds
        or len(development_worlds) != design.expected_worlds
        or set(train_worlds) & set(development_worlds)
        or train_pair_uids & development_pair_uids
    ):
        raise QualityProbeValidationError("Train/development world boundary drift")
    for value, worlds in ((train[0], train_worlds), (development[0], development_worlds)):
        counts = Counter(world_uid for world_uid, _pair_uid in value.row_keys)
        if counts != Counter({world_uid: design.pairs_per_world for world_uid in worlds}):
            raise QualityProbeValidationError("Frozen pair count per world drift")
    return train, development


def _validate_eligibility(
    frozen: preparer.FrozenTextEligibility,
    *,
    row_keys: Sequence[tuple[str, str]],
    excluded_pairs_per_world: int,
) -> np.ndarray:
    preparer.verify_frozen_text_eligibility(frozen)
    if frozen.row_keys != tuple(row_keys):
        raise QualityProbeValidationError("Text eligibility row count drift")
    mask = frozen.values
    excluded: Counter[str] = Counter()
    for keep, (world_uid, _pair_uid) in zip(mask, row_keys):
        if not keep:
            excluded[world_uid] += 1
    worlds = _ordered_worlds(row_keys)
    if excluded != Counter(
        {world_uid: excluded_pairs_per_world for world_uid in worlds}
    ):
        raise QualityProbeValidationError("Text exclusion count per world drift")
    return mask


def _load_and_validate_truth(
    *,
    split: str,
    truth_loader: Callable[[str], Sequence[Mapping[str, Any]]],
    row_keys: Sequence[tuple[str, str]],
    design: ProbeFamilyDesign,
    eligibility: np.ndarray | None,
) -> np.ndarray:
    if split not in {"train", "development"}:
        raise QualityProbeValidationError("Audit truth must remain sealed")
    rows = tuple(truth_loader(split))
    if len(rows) != len(row_keys):
        raise QualityProbeDatasetGateError("Truth row count drift")
    labels = np.empty(len(rows), dtype=np.int64)
    positives: Counter[str] = Counter()
    for index, (row, (world_uid, pair_uid)) in enumerate(zip(rows, row_keys)):
        if (
            not isinstance(row, Mapping)
            or tuple(row) != TRUTH_FIELDS
            or type(row["world_uid"]) is not str
            or row["world_uid"] != world_uid
            or type(row["canonical_pair_uid"]) is not str
            or row["canonical_pair_uid"] != pair_uid
            or type(row["label"]) is not int
            or row["label"] not in {0, 1}
        ):
            raise QualityProbeDatasetGateError("Truth schema/order/label drift")
        labels[index] = row["label"]
        positives[world_uid] += row["label"]
    worlds = _ordered_worlds(row_keys)
    if positives != Counter(
        {world_uid: design.positives_per_world for world_uid in worlds}
    ):
        raise QualityProbeDatasetGateError("Positive count per world drift")
    if eligibility is not None:
        if eligibility.shape != labels.shape or np.any(labels[~eligibility] != 0):
            raise QualityProbeDatasetGateError(
                "Text mask excluded a positive or changed row alignment"
            )
        eligible_counts = Counter(
            world_uid
            for (world_uid, _pair_uid), keep in zip(row_keys, eligibility)
            if keep
        )
        expected_eligible = design.pairs_per_world - design.excluded_pairs_per_world
        if eligible_counts != Counter(
            {world_uid: expected_eligible for world_uid in worlds}
        ):
            raise QualityProbeDatasetGateError("Eligible pair count per world drift")
    labels.setflags(write=False)
    return labels


def _single_feature_maximum(
    matrices: Sequence[preparer.FrozenFeatureMatrix],
    labels: np.ndarray,
    mask: np.ndarray | None,
) -> dict[str, Any]:
    best = -math.inf
    winners: list[str] = []
    evaluated = 0
    for frozen in matrices:
        values = frozen.values if mask is None else frozen.values[mask]
        for index, name in enumerate(frozen.column_names):
            metric = symmetric_auc(labels, values[:, index])
            evaluated += 1
            qualified_name = f"{frozen.view}::{name}"
            if metric > best:
                best = metric
                winners = [qualified_name]
            elif metric == best:
                winners.append(qualified_name)
    return {
        "evaluated_feature_count": evaluated,
        "maximum_symmetric_auc": float(best),
        "winner_name": min(winners, key=lambda value: value.encode("utf-8")),
        "tie_count": len(winners),
    }


def _fit_probe_models(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    development_x: np.ndarray,
    policy: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if (
        train_x.ndim != 2
        or development_x.ndim != 2
        or train_x.dtype != np.dtype("<f8")
        or development_x.dtype != np.dtype("<f8")
        or not train_x.flags.c_contiguous
        or not development_x.flags.c_contiguous
        or train_x.shape[1] != development_x.shape[1]
        or train_x.shape[0] != train_y.shape[0]
    ):
        raise QualityProbeValidationError("Probe fit matrix closure drift")
    if (
        preparer._stream_nonfinite_and_missing_bitmap(train_x)[0] != 0
        or preparer._stream_nonfinite_and_missing_bitmap(development_x)[0] != 0
    ):
        raise QualityProbeValidationError("Probe fit matrix contains nonfinite values")
    config = policy["probe_models"]
    logistic_config = dict(config["logistic_l2"])
    tree_config = dict(config["hist_gradient_boosting_depth2"])
    tree_class = tree_config.pop("class")
    if tree_class != "sklearn.ensemble.HistGradientBoostingClassifier":
        raise QualityProbeValidationError("Tree class commitment drift")
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(train_x)
        scaled_train = scaler.transform(train_x)
        logistic = LogisticRegression(**logistic_config).fit(
            scaled_train, train_y
        )
        if int(logistic.n_iter_[0]) >= int(logistic_config["max_iter"]):
            raise QualityProbeValidationError("Quality logistic reached max_iter")
        del scaled_train
        scaled_development = scaler.transform(development_x)
        logistic_score = logistic.predict_proba(scaled_development)[:, 1]
        del scaled_development
        tree = HistGradientBoostingClassifier(**tree_config).fit(train_x, train_y)
        scores = {
            "logistic_l2": logistic_score,
            "hist_gradient_boosting_depth2": tree.predict_proba(development_x)[:, 1],
        }
    finite_objects = (
        scaler.mean_,
        scaler.var_,
        scaler.scale_,
        logistic.coef_,
        logistic.intercept_,
        *scores.values(),
    )
    if any(not np.isfinite(value).all() for value in finite_objects):
        raise QualityProbeValidationError("Probe fit produced nonfinite state")
    return {name: np.asarray(value, dtype=np.float64) for name, value in scores.items()}


def generate_bootstrap_draws(
    *, replicates: int, world_count: int, seed: int
) -> np.ndarray:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (replicates, world_count, seed)
    ):
        raise QualityProbeValidationError("Bootstrap constants are invalid")
    draws = np.random.Generator(np.random.PCG64(seed)).integers(
        0,
        world_count,
        size=(replicates, world_count),
        dtype=np.int64,
        endpoint=False,
    )
    return np.ascontiguousarray(draws, dtype=np.dtype("<i8"))


def _bootstrap_family_upper(
    *,
    labels: np.ndarray,
    row_world_uids: Sequence[str],
    ordered_world_uids: Sequence[str],
    score_family: Mapping[str, np.ndarray],
    baseline: float,
    draws: np.ndarray,
    batch_size: int = 16,
) -> dict[str, Any]:
    worlds = tuple(ordered_world_uids)
    if (
        not worlds
        or len(worlds) != len(set(worlds))
        or set(worlds) != set(row_world_uids)
        or not score_family
        or draws.ndim != 2
        or draws.shape[1] != len(worlds)
        or draws.dtype != np.dtype("<i8")
        or np.any(draws < 0)
        or np.any(draws >= len(worlds))
        or batch_size != 16
    ):
        raise QualityProbeValidationError("Bootstrap input contract drift")
    world_index = {world_uid: index for index, world_uid in enumerate(worlds)}
    row_world_index = np.fromiter(
        (world_index[value] for value in row_world_uids),
        dtype=np.int64,
        count=len(row_world_uids),
    )
    replicate_count = draws.shape[0]
    draw_multiplicities = np.asarray(
        [np.bincount(draw, minlength=len(worlds)) for draw in draws],
        dtype=np.float64,
    )
    family_auc_max = np.full(replicate_count, -np.inf, dtype=np.float64)
    family_ap_uplift_max = np.full(replicate_count, -np.inf, dtype=np.float64)
    for model_name in sorted(score_family, key=lambda value: value.encode("utf-8")):
        scores = np.asarray(score_family[model_name], dtype=np.float64)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise QualityProbeValidationError("Bootstrap score vector drift")
        unique_scores, group_index = np.unique(scores, return_inverse=True)
        use_dense_groups = len(unique_scores) * len(worlds) <= len(labels)
        if use_dense_groups:
            group_positive = np.zeros(
                (len(unique_scores), len(worlds)), dtype=np.float64
            )
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
                    (
                        np.asarray([True]),
                        descending_scores[1:] != descending_scores[:-1],
                    )
                )
            )
        for start in range(0, replicate_count, batch_size):
            stop = min(start + batch_size, replicate_count)
            multiplicities = draw_multiplicities[start:stop]
            if use_dense_groups:
                # np.unique orders scores ascending in this branch.
                positive = multiplicities @ group_positive.T
                negative = multiplicities @ group_negative.T
                negative_before = np.cumsum(negative, axis=1) - negative
                positive_desc = positive[:, ::-1]
                total_desc = (positive + negative)[:, ::-1]
            else:
                row_weights = multiplicities[:, descending_world_index]
                positive_rows = row_weights * descending_labels
                negative_rows = row_weights * (1.0 - descending_labels)
                positive = np.add.reduceat(positive_rows, group_starts, axis=1)
                negative = np.add.reduceat(negative_rows, group_starts, axis=1)
                # Groups are already descending.  A positive outranks all
                # negatives in later groups; tied negatives contribute 0.5.
                negative_before = (
                    negative.sum(axis=1, keepdims=True)
                    - np.cumsum(negative, axis=1)
                )
                positive_desc = positive
                total_desc = positive + negative
            total_positive = positive.sum(axis=1)
            total_negative = negative.sum(axis=1)
            if np.any(total_positive <= 0) or np.any(total_negative <= 0):
                raise QualityProbeValidationError("Bootstrap replicate lost a class")
            auc = np.sum(
                positive * (negative_before + 0.5 * negative), axis=1
            ) / (total_positive * total_negative)
            symmetric = np.maximum(auc, 1.0 - auc)
            cumulative_positive = np.cumsum(positive_desc, axis=1)
            cumulative_total = np.cumsum(total_desc, axis=1)
            precision = np.divide(
                cumulative_positive,
                cumulative_total,
                out=np.zeros_like(cumulative_positive),
                where=cumulative_total > 0,
            )
            ap_uplift = (
                np.sum(positive_desc * precision, axis=1) / total_positive
            ) - baseline
            family_auc_max[start:stop] = np.maximum(
                family_auc_max[start:stop], symmetric
            )
            family_ap_uplift_max[start:stop] = np.maximum(
                family_ap_uplift_max[start:stop], ap_uplift
            )
    if not np.isfinite(family_auc_max).all() or not np.isfinite(
        family_ap_uplift_max
    ).all():
        raise QualityProbeValidationError("Bootstrap family maxima are nonfinite")
    return {
        "replicates": replicate_count,
        "world_count": len(worlds),
        "score_family_size": len(score_family),
        "draws_raw_i8_c_sha256": _sha256_bytes(draws.tobytes(order="C")),
        "family_max_symmetric_auc_vector_sha256": _vector_sha256(family_auc_max),
        "family_max_average_precision_uplift_vector_sha256": _vector_sha256(
            family_ap_uplift_max
        ),
        "symmetric_auc_95_upper": float(
            np.quantile(family_auc_max, 0.95, method="linear")
        ),
        "average_precision_uplift_95_upper": float(
            np.quantile(family_ap_uplift_max, 0.95, method="linear")
        ),
    }


def _evaluate(
    *,
    train_matrices: Sequence[preparer.FrozenFeatureMatrix],
    development_matrices: Sequence[preparer.FrozenFeatureMatrix],
    truth_loader: Callable[[str], Sequence[Mapping[str, Any]]] | None,
    preloaded_truth: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    design: ProbeFamilyDesign,
    policy: Mapping[str, Any],
    train_eligibility: preparer.FrozenTextEligibility | None,
    development_eligibility: preparer.FrozenTextEligibility | None,
) -> dict[str, Any]:
    _validate_runtime()
    channel_policy.validate_policy(policy)
    # The caller retains the original frozen dataclass reference.  Rebind to a
    # validator-private copy before any fixture callback can run.
    caller_design = design
    caller_design_snapshot = replace(caller_design)
    design = replace(caller_design_snapshot)
    policy_snapshot = _canonical_json_bytes(policy)
    private_policy = json.loads(policy_snapshot.decode("utf-8"))
    channel_policy.validate_policy(private_policy)
    authorization = private_policy["authorization"]
    is_fixture = design.claim_boundary == "FIXTURE_ONLY_NO_DATASET_CONCLUSION"
    if is_fixture:
        if (
            not callable(truth_loader)
            or preloaded_truth is not None
            or authorization["implementation_and_fixture_tests"] is not True
            or design.expected_worlds > 3
            or design.bootstrap_replicates > 31
            or design.expected_views > 21
            or design.expected_total_features > 3380
            or design.pairs_per_world > 378
        ):
            raise QualityProbeValidationError("Fixture execution boundary widened")
    elif (
        truth_loader is None
        and isinstance(preloaded_truth, Mapping)
        and set(preloaded_truth) == {"train", "development"}
        and design.claim_boundary
        == "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
        and authorization["quality_audit_run"] is True
        and authorization["metric_generation"] is True
    ):
        # This core may calculate aggregate probe statistics from rows that
        # the single root-bound formal transaction has already opened.  It
        # never emits a standalone formal claim; only the outer transaction
        # may wrap its INTERNAL_* receipt after checking the policy root pin.
        pass
    elif (
        truth_loader is None
        and isinstance(preloaded_truth, Mapping)
        and set(preloaded_truth) == {"train", "development"}
        and design.claim_boundary
        == "ROOT_BOUND_COMPOSITION_FIXTURE_ONLY_NO_DATASET_CONCLUSION"
        and authorization["implementation_and_fixture_tests"] is True
        and design.expected_worlds <= 3
        and design.bootstrap_replicates <= 31
        and design.expected_views <= 21
        and design.expected_total_features <= 3380
        and design.pairs_per_world <= 378
    ):
        pass
    else:
        raise QualityProbeValidationError(
            "Formal calculation requires preloaded rows from the root-bound transaction"
        )
    train, development = _validate_matrix_sets(
        train_matrices, development_matrices, design
    )
    private_matrix_commitments = tuple(
        preparer.current_feature_matrix_commitment_json(frozen)
        for frozen in (*train, *development)
    )
    is_text = design.excluded_pairs_per_world > 0
    if (
        is_text
        and (train_eligibility is None or development_eligibility is None)
    ) or (
        not is_text
        and (train_eligibility is not None or development_eligibility is not None)
    ):
        raise QualityProbeValidationError("Eligibility capability/family drift")
    train_mask_source = (
        _validate_eligibility(
            train_eligibility,
            row_keys=train[0].row_keys,
            excluded_pairs_per_world=design.excluded_pairs_per_world,
        )
        if is_text
        else None
    )
    development_mask_source = (
        _validate_eligibility(
            development_eligibility,
            row_keys=development[0].row_keys,
            excluded_pairs_per_world=design.excluded_pairs_per_world,
        )
        if is_text
        else None
    )
    train_mask = (
        None
        if train_mask_source is None
        else np.array(train_mask_source, dtype=bool, order="C", copy=True)
    )
    development_mask = (
        None
        if development_mask_source is None
        else np.array(development_mask_source, dtype=bool, order="C", copy=True)
    )
    for mask in (train_mask, development_mask):
        if mask is not None:
            mask.setflags(write=False)
    private_eligibility_commitments = tuple(
        preparer.current_text_eligibility_commitment_json(frozen)
        for frozen in (train_eligibility, development_eligibility)
        if frozen is not None
    )

    loader_calls: Counter[str] = Counter()

    def counted_truth_loader(split: str) -> Sequence[Mapping[str, Any]]:
        if split not in {"train", "development"}:
            raise QualityProbeValidationError("Audit truth loader call attempted")
        loader_calls[split] += 1
        if loader_calls[split] != 1:
            raise QualityProbeValidationError("Truth loader called more than once")
        if is_fixture:
            assert truth_loader is not None
            return truth_loader(split)
        assert preloaded_truth is not None
        return preloaded_truth[split]

    # The only truth-loader calls occur after all matrix/mask commitments pass.
    train_labels_full = _load_and_validate_truth(
        split="train",
        truth_loader=counted_truth_loader,
        row_keys=train[0].row_keys,
        design=design,
        eligibility=train_mask,
    )
    development_labels_full = _load_and_validate_truth(
        split="development",
        truth_loader=counted_truth_loader,
        row_keys=development[0].row_keys,
        design=design,
        eligibility=development_mask,
    )
    # A loader is an external privileged callback in fixture mode.  Rehash all
    # label-free state after it returns so it cannot mutate features or masks
    # as a function of the truth it just opened.
    for index, frozen in enumerate((*train, *development)):
        preparer.verify_frozen_feature_matrix(frozen)
        if (
            preparer.current_feature_matrix_commitment_json(frozen)
            != private_matrix_commitments[index]
        ):
            raise QualityProbeValidationError(
                "Feature matrix changed after truth open"
            )
    if train_eligibility is not None:
        preparer.verify_frozen_text_eligibility(train_eligibility)
    if development_eligibility is not None:
        preparer.verify_frozen_text_eligibility(development_eligibility)
    for index, frozen in enumerate(
        value
        for value in (train_eligibility, development_eligibility)
        if value is not None
    ):
        if (
            preparer.current_text_eligibility_commitment_json(frozen)
            != private_eligibility_commitments[index]
        ):
            raise QualityProbeValidationError(
                "Text eligibility changed after truth open"
            )
    channel_policy.validate_policy(policy)
    if _canonical_json_bytes(policy) != policy_snapshot:
        raise QualityProbeValidationError("Quality policy changed after truth open")
    if caller_design != caller_design_snapshot:
        raise QualityProbeValidationError("Probe design changed after truth open")
    train_labels = (
        train_labels_full if train_mask is None else train_labels_full[train_mask]
    )
    development_labels = (
        development_labels_full
        if development_mask is None
        else development_labels_full[development_mask]
    )
    single = _single_feature_maximum(
        development, development_labels, development_mask
    )
    model_scores: dict[str, np.ndarray] = {}
    model_metrics: dict[str, dict[str, Any]] = {}
    for train_value, development_value in zip(train, development):
        train_x = (
            train_value.values
            if train_mask is None
            else train_value.values[train_mask]
        )
        development_x = (
            development_value.values
            if development_mask is None
            else development_value.values[development_mask]
        )
        scores = _fit_probe_models(
            train_x=train_x,
            train_y=train_labels,
            development_x=development_x,
            policy=private_policy,
        )
        if tuple(scores) != (
            "logistic_l2",
            "hist_gradient_boosting_depth2",
        ):
            raise QualityProbeValidationError("Probe model family cardinality drift")
        for model_kind, score in scores.items():
            name = f"{development_value.view}::{model_kind}"
            model_scores[name] = score
            model_metrics[name] = {
                "symmetric_auc": symmetric_auc(development_labels, score),
                "average_precision": float(
                    average_precision_score(development_labels, score)
                ),
                "prediction_vector_sha256": _vector_sha256(score),
            }
    if len(model_metrics) != design.expected_views * 2:
        raise QualityProbeValidationError("Probe model family cardinality drift")
    model_auc_maximum = max(
        value["symmetric_auc"] for value in model_metrics.values()
    )
    model_ap_uplift_maximum = max(
        value["average_precision"] - design.average_precision_baseline
        for value in model_metrics.values()
    )
    auc_winners = sorted(
        (
            name
            for name, value in model_metrics.items()
            if value["symmetric_auc"] == model_auc_maximum
        ),
        key=lambda value: value.encode("utf-8"),
    )
    ap_winners = sorted(
        (
            name
            for name, value in model_metrics.items()
            if value["average_precision"] - design.average_precision_baseline
            == model_ap_uplift_maximum
        ),
        key=lambda value: value.encode("utf-8"),
    )
    development_world_uids_full = tuple(
        world_uid for world_uid, _pair_uid in development[0].row_keys
    )
    development_world_uids = (
        development_world_uids_full
        if development_mask is None
        else tuple(
            value
            for value, keep in zip(development_world_uids_full, development_mask)
            if keep
        )
    )
    ordered_development_worlds = _ordered_worlds(development[0].row_keys)
    draws = generate_bootstrap_draws(
        replicates=design.bootstrap_replicates,
        world_count=design.expected_worlds,
        seed=design.bootstrap_seed,
    )
    draws_hash = _sha256_bytes(draws.tobytes(order="C"))
    if design.require_formal_bootstrap_binding and draws_hash != FORMAL_BOOTSTRAP_SHA256:
        raise QualityProbeValidationError("Formal bootstrap matrix hash drift")
    with threadpool_limits(limits=1):
        bootstrap = _bootstrap_family_upper(
            labels=development_labels,
            row_world_uids=development_world_uids,
            ordered_world_uids=ordered_development_worlds,
            score_family=model_scores,
            baseline=design.average_precision_baseline,
            draws=draws,
        )
    gates = private_policy["quality_gates"]
    failures: list[str] = []
    comparisons = (
        (
            "maximum_single_feature_symmetric_auc",
            single["maximum_symmetric_auc"],
            gates["maximum_single_feature_symmetric_auc"],
        ),
        (
            "maximum_family_symmetric_auc",
            model_auc_maximum,
            gates["maximum_family_symmetric_auc"],
        ),
        (
            "maximum_family_average_precision_uplift",
            model_ap_uplift_maximum,
            gates["maximum_family_average_precision_uplift"],
        ),
        (
            "bootstrap_95_upper_symmetric_auc",
            bootstrap["symmetric_auc_95_upper"],
            gates["bootstrap_95_upper_symmetric_auc"],
        ),
        (
            "bootstrap_95_upper_average_precision_uplift",
            bootstrap["average_precision_uplift_95_upper"],
            gates["bootstrap_95_upper_average_precision_uplift"],
        ),
    )
    for name, observed, threshold in comparisons:
        if not math.isfinite(float(observed)) or float(observed) > float(threshold):
            failures.append(name)
    gate_checks = {
        name: {
            "observed": float(observed),
            "maximum_allowed": float(threshold),
            "passed": math.isfinite(float(observed))
            and float(observed) <= float(threshold),
        }
        for name, observed, threshold in comparisons
    }
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": (
            (
                "FIXTURE_MECHANISM_PASS_NO_DATASET_CONCLUSION"
                if not failures
                else "FIXTURE_MECHANISM_GATE_TRIGGERED_NO_DATASET_CONCLUSION"
            )
            if is_fixture
            else (
                "INTERNAL_PROBE_PASS_NO_STANDALONE_CLAIM"
                if not failures
                else "INTERNAL_PROBE_GATE_TRIGGERED_NO_STANDALONE_CLAIM"
            )
        ),
        "claim_boundary": (
            design.claim_boundary
            if is_fixture
            else "INTERNAL_FORMAL_PROBE_CALCULATION_NO_STANDALONE_CLAIM"
        ),
        "design_claim_boundary": design.claim_boundary,
        "family": design.family,
        "train_world_count": design.expected_worlds,
        "development_world_count": design.expected_worlds,
        "full_pair_count_per_world": design.pairs_per_world,
        "eligible_pair_count_per_world": (
            design.pairs_per_world - design.excluded_pairs_per_world
        ),
        "positive_pair_count_per_world": design.positives_per_world,
        "average_precision_baseline": design.average_precision_baseline,
        "quality_policy_canonical_self_hash": private_policy["canonical_self_hash"],
        "input_commitments": {
            "train": [
                {"view": value.view, "sha256": value.commitment_sha256}
                for value in train
            ],
            "development": [
                {"view": value.view, "sha256": value.commitment_sha256}
                for value in development
            ],
            "train_text_eligibility_sha256": (
                None
                if train_eligibility is None
                else train_eligibility.commitment_sha256
            ),
            "development_text_eligibility_sha256": (
                None
                if development_eligibility is None
                else development_eligibility.commitment_sha256
            ),
        },
        "single_feature": single,
        "model_family": {
            "model_count": len(model_metrics),
            "maximum_symmetric_auc": model_auc_maximum,
            "maximum_symmetric_auc_winner": auc_winners[0],
            "maximum_symmetric_auc_tie_count": len(auc_winners),
            "maximum_average_precision_uplift": model_ap_uplift_maximum,
            "maximum_average_precision_uplift_winner": ap_winners[0],
            "maximum_average_precision_uplift_tie_count": len(ap_winners),
            "models": model_metrics,
        },
        "bootstrap": bootstrap,
        "gate_checks": gate_checks,
        "gate_failures": failures,
        "truth_loader_call_counts": {
            "train": loader_calls["train"],
            "development": loader_calls["development"],
            "audit_a": loader_calls["audit_a"],
            "audit_b": loader_calls["audit_b"],
        },
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
    }
    receipt["canonical_self_hash"] = _sha256_bytes(_canonical_json_bytes(receipt))
    return receipt


def formal_design_for_family(
    family: str, policy: Mapping[str, Any]
) -> ProbeFamilyDesign:
    """Construct the only training-preflight design allowed for a v9 run."""

    if family == "text":
        text = policy["text_probe_family"]
        return ProbeFamilyDesign(
            family="text",
            view_widths=tuple(
                (f"{surface}::{view}", width)
                for surface in policy["model_views"]["order"]
                for view, width in zip(text["view_names"], text["feature_widths"])
            ),
            expected_views=text["surface_view_count"] * len(text["view_names"]),
            expected_total_features=text["single_feature_count"],
            expected_column_name_hashes=tuple(
                (
                    f"{surface}::{view}",
                    text_views.EXPECTED_NAME_HASHES[view],
                )
                for surface in policy["model_views"]["order"]
                for view in text["view_names"]
            ),
            expected_worlds=policy["design_scale"]["world_counts"]["development"],
            pairs_per_world=text["model_pair_keyspace_per_world"],
            positives_per_world=text["positive_pairs_per_world"],
            excluded_pairs_per_world=text["excluded_negative_pairs_per_world"],
            average_precision_baseline=text["average_precision_baseline"],
            bootstrap_replicates=policy["bootstrap"]["replicates"],
            bootstrap_seed=policy["bootstrap"]["text_design_seed"],
            require_formal_bootstrap_binding=True,
            claim_boundary="V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
        )
    if family == "code_and_slot":
        public = policy["public_code_probe"]
        decoded = policy["decoded_slot_probe"]
        return ProbeFamilyDesign(
            family="code_and_slot",
            view_widths=(
                ("public_code_2992", public["feature_width"]),
                ("decoded_slot_388", decoded["feature_width"]),
            ),
            expected_views=2,
            expected_total_features=public["feature_width"] + decoded["feature_width"],
            expected_column_name_hashes=(
                (
                    "public_code_2992",
                    public["feature_names_canonical_json_sha256"],
                ),
                (
                    "decoded_slot_388",
                    decoded["feature_names_canonical_json_sha256"],
                ),
            ),
            expected_worlds=policy["design_scale"]["world_counts"]["development"],
            pairs_per_world=policy["design_scale"]["pair_count_per_world"],
            positives_per_world=policy["design_scale"]["positive_pair_count_per_world"],
            excluded_pairs_per_world=0,
            average_precision_baseline=decoded["average_precision_baseline"],
            bootstrap_replicates=policy["bootstrap"]["replicates"],
            bootstrap_seed=policy["bootstrap"]["metadata_design_seed"],
            require_formal_bootstrap_binding=True,
            claim_boundary="V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
        )
    raise QualityProbeValidationError("Unknown quality-probe family")


def evaluate_formal_probe_family(
    *,
    family: str,
    train_matrices: Sequence[preparer.FrozenFeatureMatrix],
    development_matrices: Sequence[preparer.FrozenFeatureMatrix],
    policy: Mapping[str, Any],
    train_eligibility: preparer.FrozenTextEligibility | None = None,
    development_eligibility: preparer.FrozenTextEligibility | None = None,
) -> dict[str, Any]:
    if (
        policy.get("authorization", {}).get("quality_audit_run") is not True
        or policy.get("authorization", {}).get("metric_generation") is not True
    ):
        raise QualityProbeValidationError(
            "Formal quality audit and metric generation remain unauthorized"
        )
    raise QualityProbeValidationError(
        "Pinned formal truth capability runner requires both preregistered "
        "families in one transaction"
    )


def _verify_feature_bundle_unchanged(
    matrices: Sequence[preparer.FrozenFeatureMatrix],
    expected_commitment_json: Sequence[bytes],
    *,
    error_message: str,
) -> None:
    if len(matrices) != len(expected_commitment_json):
        raise QualityProbeValidationError(error_message)
    for index, value in enumerate(matrices):
        try:
            preparer.verify_frozen_feature_matrix(value)
        except preparer.QualityProbePreparationError as exc:
            raise QualityProbeValidationError(error_message) from exc
        if (
            preparer.current_feature_matrix_commitment_json(value)
            != expected_commitment_json[index]
        ):
            raise QualityProbeValidationError(error_message)


def _verify_eligibility_bundle_unchanged(
    values: Sequence[preparer.FrozenTextEligibility],
    expected_commitment_json: Sequence[bytes],
    *,
    error_message: str,
) -> None:
    if len(values) != len(expected_commitment_json):
        raise QualityProbeValidationError(error_message)
    for index, value in enumerate(values):
        try:
            preparer.verify_frozen_text_eligibility(value)
        except preparer.QualityProbePreparationError as exc:
            raise QualityProbeValidationError(error_message) from exc
        if (
            preparer.current_text_eligibility_commitment_json(value)
            != expected_commitment_json[index]
        ):
            raise QualityProbeValidationError(error_message)


def evaluate_root_bound_composition_fixture(
    *,
    text_train_matrices: Sequence[preparer.FrozenFeatureMatrix],
    text_development_matrices: Sequence[preparer.FrozenFeatureMatrix],
    code_train_matrices: Sequence[preparer.FrozenFeatureMatrix],
    code_development_matrices: Sequence[preparer.FrozenFeatureMatrix],
    truth: truth_capability.FormalTrainDevelopmentTruthCapability,
    expected_root_binding: Mapping[str, Any],
    policy: Mapping[str, Any],
    text_design: ProbeFamilyDesign,
    code_design: ProbeFamilyDesign,
    train_text_eligibility: preparer.FrozenTextEligibility,
    development_text_eligibility: preparer.FrozenTextEligibility,
) -> dict[str, Any]:
    """Exercise the formal two-family call topology at a bounded fixture scale."""

    channel_policy.validate_policy(policy)
    fixture_boundary = (
        "ROOT_BOUND_COMPOSITION_FIXTURE_ONLY_NO_DATASET_CONCLUSION"
    )
    if (
        type(truth) is not truth_capability.FormalTrainDevelopmentTruthCapability
        or set(expected_root_binding)
        != {"path", "size_bytes", "sha256", "canonical_self_hash"}
        or truth.root_binding() != dict(expected_root_binding)
        or text_design.claim_boundary != fixture_boundary
        or code_design.claim_boundary != fixture_boundary
        or text_design.expected_worlds > 3
        or code_design.expected_worlds > 3
        or text_design.bootstrap_replicates > 31
        or code_design.bootstrap_replicates > 31
    ):
        raise QualityProbeValidationError(
            "Root-bound composition fixture boundary widened"
        )
    text_train, text_development = _validate_matrix_sets(
        text_train_matrices, text_development_matrices, text_design
    )
    code_train, code_development = _validate_matrix_sets(
        code_train_matrices, code_development_matrices, code_design
    )
    _validate_eligibility(
        train_text_eligibility,
        row_keys=text_train[0].row_keys,
        excluded_pairs_per_world=text_design.excluded_pairs_per_world,
    )
    _validate_eligibility(
        development_text_eligibility,
        row_keys=text_development[0].row_keys,
        excluded_pairs_per_world=text_design.excluded_pairs_per_world,
    )
    all_matrices = (
        *text_train,
        *text_development,
        *code_train,
        *code_development,
    )
    matrix_bytes = tuple(
        preparer.current_feature_matrix_commitment_json(value)
        for value in all_matrices
    )
    eligibility_values = (
        train_text_eligibility,
        development_text_eligibility,
    )
    eligibility_bytes = tuple(
        preparer.current_text_eligibility_commitment_json(value)
        for value in eligibility_values
    )
    truth_pins = truth._begin_bound_transaction(
        expected_root_binding=expected_root_binding
    )
    preloaded_truth: dict[str, Sequence[Mapping[str, Any]]] = {}
    for split in truth_capability.SUPERVISED_SPLITS:
        rows, split_receipt = truth_capability._read_pinned_truth_csv(
            truth_pins[split]
        )
        truth._record_split_receipt(split=split, receipt=split_receipt)
        preloaded_truth[split] = rows
    _verify_feature_bundle_unchanged(
        all_matrices,
        matrix_bytes,
        error_message="Composition fixture matrix changed after truth open",
    )
    _verify_eligibility_bundle_unchanged(
        eligibility_values,
        eligibility_bytes,
        error_message="Composition fixture eligibility changed after truth open",
    )
    text_receipt = _evaluate(
        train_matrices=text_train,
        development_matrices=text_development,
        truth_loader=None,
        preloaded_truth=preloaded_truth,
        design=text_design,
        policy=policy,
        train_eligibility=train_text_eligibility,
        development_eligibility=development_text_eligibility,
    )
    _verify_feature_bundle_unchanged(
        all_matrices,
        matrix_bytes,
        error_message="Composition fixture matrix changed between probe families",
    )
    code_receipt = _evaluate(
        train_matrices=code_train,
        development_matrices=code_development,
        truth_loader=None,
        preloaded_truth=preloaded_truth,
        design=code_design,
        policy=policy,
        train_eligibility=None,
        development_eligibility=None,
    )
    truth_receipt = truth.aggregate_receipt()
    preloaded_truth.clear()
    gate_failures = tuple(
        (*text_receipt["gate_failures"], *code_receipt["gate_failures"])
    )
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": (
            "ROOT_BOUND_COMPOSITION_FIXTURE_PASS_NO_DATASET_CONCLUSION"
            if not gate_failures
            else "ROOT_BOUND_COMPOSITION_FIXTURE_GATE_TRIGGERED_NO_DATASET_CONCLUSION"
        ),
        "claim_boundary": fixture_boundary,
        "family_receipts": {
            "text": text_receipt,
            "code_and_slot": code_receipt,
        },
        "truth_file_access": truth_receipt,
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
    }
    receipt["canonical_self_hash"] = _sha256_bytes(
        _canonical_json_bytes(receipt)
    )
    return receipt


def evaluate_formal_probe_families(
    *,
    text_train_matrices: Sequence[preparer.FrozenFeatureMatrix],
    text_development_matrices: Sequence[preparer.FrozenFeatureMatrix],
    code_train_matrices: Sequence[preparer.FrozenFeatureMatrix],
    code_development_matrices: Sequence[preparer.FrozenFeatureMatrix],
    dataset_root: Path,
    root_manifest_pin: truth_capability.RootManifestPin,
    policy: Mapping[str, Any],
    train_text_eligibility: preparer.FrozenTextEligibility,
    development_text_eligibility: preparer.FrozenTextEligibility,
) -> dict[str, Any]:
    """Evaluate both formal shortcut families after one shared truth open."""

    channel_policy.validate_policy(policy)
    authorization = policy["authorization"]
    if (
        authorization["quality_audit_run"] is not True
        or authorization["metric_generation"] is not True
    ):
        raise QualityProbeValidationError(
            "Formal quality audit and metric generation remain unauthorized"
        )
    expected_root_binding = policy.get("pins", {}).get("design_root_manifest")
    manifest_path = (dataset_root.resolve() / "root_manifest.json").resolve()
    try:
        root_path = manifest_path.relative_to(truth_capability.ROOT).as_posix()
    except ValueError:
        root_path = manifest_path.as_posix()
    supplied_root_binding = {
        "path": root_path,
        "size_bytes": root_manifest_pin.size_bytes,
        "sha256": root_manifest_pin.sha256,
        "canonical_self_hash": root_manifest_pin.canonical_self_hash,
    }
    if (
        not isinstance(expected_root_binding, Mapping)
        or set(expected_root_binding)
        != {"path", "size_bytes", "sha256", "canonical_self_hash"}
        or supplied_root_binding != dict(expected_root_binding)
    ):
        raise QualityProbeValidationError(
            "Formal dataset root does not match the policy root pin"
        )
    text_design = formal_design_for_family("text", policy)
    code_design = formal_design_for_family("code_and_slot", policy)
    # Both families and both masks are verified before the first truth byte is
    # opened.  These local bytes are independent of mutable dataclass fields.
    text_train, text_development = _validate_matrix_sets(
        text_train_matrices, text_development_matrices, text_design
    )
    code_train, code_development = _validate_matrix_sets(
        code_train_matrices, code_development_matrices, code_design
    )
    _validate_eligibility(
        train_text_eligibility,
        row_keys=text_train[0].row_keys,
        excluded_pairs_per_world=text_design.excluded_pairs_per_world,
    )
    _validate_eligibility(
        development_text_eligibility,
        row_keys=text_development[0].row_keys,
        excluded_pairs_per_world=text_design.excluded_pairs_per_world,
    )
    all_matrices = (
        *text_train,
        *text_development,
        *code_train,
        *code_development,
    )
    pretruth_matrix_bytes = tuple(
        preparer.current_feature_matrix_commitment_json(value)
        for value in all_matrices
    )
    pretruth_eligibility_bytes = (
        preparer.current_text_eligibility_commitment_json(train_text_eligibility),
        preparer.current_text_eligibility_commitment_json(
            development_text_eligibility
        ),
    )
    policy_bytes = _canonical_json_bytes(policy)
    truth = truth_capability.FormalTrainDevelopmentTruthCapability.from_pinned_design_root(
        dataset_root=dataset_root,
        root_manifest_pin=root_manifest_pin,
    )
    if truth.root_binding() != dict(expected_root_binding):
        raise QualityProbeValidationError(
            "Formal truth capability does not match the policy root pin"
        )
    truth_pins = truth._begin_bound_transaction(
        expected_root_binding=expected_root_binding
    )
    preloaded_truth: dict[str, Sequence[Mapping[str, Any]]] = {}
    for split in truth_capability.SUPERVISED_SPLITS:
        rows, split_receipt = truth_capability._read_pinned_truth_csv(
            truth_pins[split]
        )
        truth._record_split_receipt(split=split, receipt=split_receipt)
        preloaded_truth[split] = rows
    # Reclose every label-free object immediately after the only physical
    # truth read and before any single-feature metric or model fit.
    _verify_feature_bundle_unchanged(
        all_matrices,
        pretruth_matrix_bytes,
        error_message="Formal matrix bundle changed after truth open",
    )
    _verify_eligibility_bundle_unchanged(
        (train_text_eligibility, development_text_eligibility),
        pretruth_eligibility_bytes,
        error_message="Formal eligibility bundle changed after truth open",
    )
    channel_policy.validate_policy(policy)
    if _canonical_json_bytes(policy) != policy_bytes:
        raise QualityProbeValidationError("Quality policy changed after truth open")
    text_receipt = _evaluate(
        train_matrices=text_train,
        development_matrices=text_development,
        truth_loader=None,
        preloaded_truth=preloaded_truth,
        design=text_design,
        policy=policy,
        train_eligibility=train_text_eligibility,
        development_eligibility=development_text_eligibility,
    )
    # Reclose the not-yet-evaluated code/slot matrices again after the text
    # calculations and before the code/slot family can be fitted.
    _verify_feature_bundle_unchanged(
        all_matrices,
        pretruth_matrix_bytes,
        error_message="Formal matrix bundle changed between probe families",
    )
    code_receipt = _evaluate(
        train_matrices=code_train,
        development_matrices=code_development,
        truth_loader=None,
        preloaded_truth=preloaded_truth,
        design=code_design,
        policy=policy,
        train_eligibility=None,
        development_eligibility=None,
    )
    truth_receipt = truth.aggregate_receipt()
    preloaded_truth.clear()
    if any(
        truth_receipt[split][field] != 0
        for split in ("audit_a", "audit_b")
        for field in (
            "file_open_count",
            "byte_read_count",
            "materialized_row_count",
        )
    ):
        raise QualityProbeValidationError("Audit truth access count drift")
    status = (
        "PASS"
        if not text_receipt["gate_failures"]
        and not code_receipt["gate_failures"]
        else "DATASET_INVALIDATED"
    )
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": status,
        "claim_boundary": "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
        "family_receipts": {
            "text": text_receipt,
            "code_and_slot": code_receipt,
        },
        "truth_file_access": truth_receipt,
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
    }
    receipt["canonical_self_hash"] = _sha256_bytes(_canonical_json_bytes(receipt))
    return receipt


def evaluate_fixture_probe_family(
    *,
    train_matrices: Sequence[preparer.FrozenFeatureMatrix],
    development_matrices: Sequence[preparer.FrozenFeatureMatrix],
    truth_loader: Callable[[str], Sequence[Mapping[str, Any]]],
    design: ProbeFamilyDesign,
    policy: Mapping[str, Any],
    train_eligibility: preparer.FrozenTextEligibility | None = None,
    development_eligibility: preparer.FrozenTextEligibility | None = None,
) -> dict[str, Any]:
    if design.claim_boundary != "FIXTURE_ONLY_NO_DATASET_CONCLUSION":
        raise QualityProbeValidationError("Fixture design claim boundary widened")
    return _evaluate(
        train_matrices=train_matrices,
        development_matrices=development_matrices,
        truth_loader=truth_loader,
        design=design,
        policy=policy,
        train_eligibility=train_eligibility,
        development_eligibility=development_eligibility,
    )


def reject_audit_truth_open(_split: str = "audit_a") -> None:
    raise QualityProbeValidationError("Audit A/B truth must remain sealed")
