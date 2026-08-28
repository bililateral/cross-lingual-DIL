#!/usr/bin/env python3
"""Frozen metric core for the V9.4 shortcut-quality audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import step28_v13_common as common


VERSION = "2026-08-27-step28-v13-v1-13-quality-probe-core-v9-4"
LOGISTIC_PREPROCESSING = (
    "StandardScaler_fit_on_train_only_then_transform_train_and_development"
)
STANDARD_SCALER_CONFIG = {"copy": True, "with_mean": True, "with_std": True}
LOGISTIC_CONFIG = {
    "C": 1.0,
    "class_weight": None,
    "dual": False,
    "fit_intercept": True,
    "intercept_scaling": 1,
    "l1_ratio": None,
    "max_iter": 10000,
    "multi_class": "deprecated",
    "n_jobs": None,
    "penalty": "l2",
    "random_state": 793820367,
    "solver": "lbfgs",
    "tol": 1e-10,
    "verbose": 0,
    "warm_start": False,
}
TREE_PREPROCESSING = "raw_unstandardized_float64"
TREE_CLASS = "sklearn.ensemble.HistGradientBoostingClassifier"
TREE_CONFIG = {
    "categorical_features": "from_dtype",
    "class_weight": None,
    "early_stopping": False,
    "interaction_cst": None,
    "l2_regularization": 1.0,
    "learning_rate": 0.03,
    "loss": "log_loss",
    "max_bins": 255,
    "max_depth": 2,
    "max_features": 1.0,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "monotonic_cst": None,
    "n_iter_no_change": 10,
    "random_state": 793820367,
    "scoring": "loss",
    "tol": 1e-7,
    "validation_fraction": 0.1,
    "verbose": 0,
    "warm_start": False,
}


class QualityProbeCoreV94Error(common.ContractError):
    """Raised when a frozen matrix or metric computation drifts."""


def _exact_mapping(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if type(observed) is not dict or tuple(observed) != tuple(expected):
        return False
    return all(
        type(observed[key]) is type(expected[key]) and observed[key] == expected[key]
        for key in expected
    )


@dataclass(frozen=True)
class FrozenMatrix:
    view: str
    values: np.ndarray
    row_keys: tuple[tuple[str, str], ...]
    column_names: tuple[str, ...]
    commitment: Mapping[str, Any]


def _matrix_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    raw = memoryview(values).cast("B")
    chunk = 8 * 1024 * 1024
    for start in range(0, len(raw), chunk):
        digest.update(raw[start : start + chunk])
    return digest.hexdigest()


def verify_frozen_matrix(matrix: FrozenMatrix) -> None:
    expected = {
        "version": VERSION,
        "view": matrix.view,
        "shape": tuple(matrix.values.shape),
        "dtype": "little-endian float64",
        "row_keys_sha256": common.canonical_sha256(
            [list(key) for key in matrix.row_keys]
        ),
        "column_names_sha256": common.canonical_sha256(
            list(matrix.column_names)
        ),
        "matrix_raw_f8_c_sha256": _matrix_sha256(matrix.values),
    }
    if (
        type(matrix) is not FrozenMatrix
        or type(matrix.commitment) is not MappingProxyType
        or tuple(matrix.commitment) != tuple(expected)
        or any(
            type(matrix.commitment[key]) is not type(expected[key])
            or matrix.commitment[key] != expected[key]
            for key in expected
        )
        or matrix.values.dtype != np.dtype("<f8")
        or not matrix.values.flags.c_contiguous
        or matrix.values.flags.writeable
        or not np.isfinite(matrix.values).all()
    ):
        raise QualityProbeCoreV94Error(
            f"Frozen matrix commitment drift: {matrix.view}"
        )


def freeze_matrix(
    *,
    view: str,
    values: np.ndarray,
    row_keys: Sequence[tuple[str, str]],
    column_names: Sequence[str],
    take_ownership: bool = False,
) -> FrozenMatrix:
    if not isinstance(view, str) or not view:
        raise QualityProbeCoreV94Error("Matrix view drift")
    if any(
        not isinstance(key, tuple)
        or len(key) != 2
        or any(not isinstance(value, str) or not value for value in key)
        for key in row_keys
    ):
        raise QualityProbeCoreV94Error("Matrix row-key type drift")
    if any(not isinstance(name, str) or not name for name in column_names):
        raise QualityProbeCoreV94Error("Matrix column-name type drift")
    if type(take_ownership) is not bool:
        raise QualityProbeCoreV94Error("Matrix ownership flag drift")
    if take_ownership:
        matrix = np.asarray(values)
        if (
            matrix.dtype != np.dtype("<f8")
            or not matrix.flags.c_contiguous
            or not matrix.flags.owndata
            or not matrix.flags.writeable
        ):
            raise QualityProbeCoreV94Error(
                "Owned frozen matrix does not have exclusive canonical storage"
            )
    else:
        matrix = np.array(values, dtype=np.dtype("<f8"), order="C", copy=True)
    keys = tuple(row_keys)
    names = tuple(column_names)
    if (
        matrix.ndim != 2
        or matrix.shape != (len(keys), len(names))
        or not keys
        or len(keys) != len(set(keys))
        or not names
        or len(names) != len(set(names))
        or not np.isfinite(matrix).all()
    ):
        raise QualityProbeCoreV94Error(f"Frozen matrix closure drift: {view}")
    commitment = MappingProxyType({
        "version": VERSION,
        "view": view,
        "shape": tuple(matrix.shape),
        "dtype": "little-endian float64",
        "row_keys_sha256": common.canonical_sha256([list(key) for key in keys]),
        "column_names_sha256": common.canonical_sha256(list(names)),
        "matrix_raw_f8_c_sha256": _matrix_sha256(matrix),
    })
    matrix = np.frombuffer(matrix.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(
        matrix.shape
    )
    frozen = FrozenMatrix(
        view=view,
        values=matrix,
        row_keys=keys,
        column_names=names,
        commitment=commitment,
    )
    verify_frozen_matrix(frozen)
    return frozen


def _validate_family(
    train: Mapping[str, FrozenMatrix], development: Mapping[str, FrozenMatrix]
) -> tuple[str, ...]:
    views = tuple(train)
    if not views or tuple(development) != views:
        raise QualityProbeCoreV94Error("Probe family view order drift")
    train_keys = next(iter(train.values())).row_keys
    development_keys = next(iter(development.values())).row_keys
    train_worlds = {key[0] for key in train_keys}
    development_worlds = {key[0] for key in development_keys}
    if train_worlds & development_worlds:
        raise QualityProbeCoreV94Error("Train/development world overlap")
    for view in views:
        verify_frozen_matrix(train[view])
        verify_frozen_matrix(development[view])
        if (
            train[view].view != view
            or development[view].view != view
            or train[view].row_keys != train_keys
            or development[view].row_keys != development_keys
            or train[view].column_names != development[view].column_names
        ):
            raise QualityProbeCoreV94Error(f"Probe family matrix drift: {view}")
    return views


def _validate_labels(labels: np.ndarray, expected_rows: int, *, label: str) -> np.ndarray:
    if not isinstance(labels, np.ndarray) or labels.dtype != np.dtype("int8"):
        raise QualityProbeCoreV94Error(f"{label} label dtype drift")
    values = np.frombuffer(labels.tobytes(order="C"), dtype=np.dtype("int8"))
    if (
        values.shape != (expected_rows,)
        or not np.isin(values, (0, 1)).all()
        or len(np.unique(values)) != 2
    ):
        raise QualityProbeCoreV94Error(f"{label} label closure drift")
    return values


def symmetric_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise QualityProbeCoreV94Error("Symmetric AUC input drift")
    auc = float(roc_auc_score(labels, scores))
    return max(auc, 1.0 - auc)


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
        or not np.isfinite(train_x).all()
        or not np.isfinite(development_x).all()
    ):
        raise QualityProbeCoreV94Error("Probe fit matrix closure drift")
    config = policy["probe_models"]
    logistic_config = dict(config["logistic_l2"])
    tree_config = dict(config["hist_gradient_boosting_depth2"])
    logistic_preprocessing = logistic_config.pop("preprocessing")
    scaler_config = dict(logistic_config.pop("standard_scaler"))
    tree_preprocessing = tree_config.pop("preprocessing")
    tree_class = tree_config.pop("class")
    if (
        logistic_preprocessing != LOGISTIC_PREPROCESSING
        or not _exact_mapping(scaler_config, STANDARD_SCALER_CONFIG)
        or tree_preprocessing != TREE_PREPROCESSING
        or tree_class != TREE_CLASS
        or not _exact_mapping(logistic_config, LOGISTIC_CONFIG)
        or not _exact_mapping(tree_config, TREE_CONFIG)
    ):
        raise QualityProbeCoreV94Error("Probe model commitment drift")
    with threadpool_limits(limits=1):
        scaler = StandardScaler(**scaler_config).fit(train_x)
        scaled_train = scaler.transform(train_x)
        logistic = LogisticRegression(**logistic_config).fit(scaled_train, train_y)
        if int(logistic.n_iter_[0]) >= int(logistic_config["max_iter"]):
            raise QualityProbeCoreV94Error("Quality logistic reached max_iter")
        logistic_score = logistic.predict_proba(
            scaler.transform(development_x)
        )[:, 1]
        tree = HistGradientBoostingClassifier(**tree_config).fit(train_x, train_y)
        if int(tree.n_iter_) != int(tree_config["max_iter"]):
            raise QualityProbeCoreV94Error("Quality tree iteration count drift")
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
        raise QualityProbeCoreV94Error("Probe fit produced nonfinite state")
    return {
        name: np.ascontiguousarray(value, dtype=np.float64)
        for name, value in scores.items()
    }


def generate_bootstrap_draws(
    *, replicates: int, world_count: int, seed: int
) -> np.ndarray:
    if any(type(value) is not int or value <= 0 for value in (replicates, world_count, seed)):
        raise QualityProbeCoreV94Error("Bootstrap constants are invalid")
    values = np.random.Generator(np.random.PCG64(seed)).integers(
        0,
        world_count,
        size=(replicates, world_count),
        dtype=np.int64,
        endpoint=False,
    )
    return np.ascontiguousarray(values, dtype=np.dtype("<i8"))


def _bootstrap_family_upper(
    *,
    labels: np.ndarray,
    row_world_uids: Sequence[str],
    ordered_world_uids: Sequence[str],
    score_family: Mapping[str, np.ndarray],
    baseline: float,
    draws: np.ndarray,
    batch_size: int,
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
        raise QualityProbeCoreV94Error("Bootstrap input contract drift")
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
    family_auc = np.full(len(draws), -np.inf, dtype=np.float64)
    family_ap = np.full(len(draws), -np.inf, dtype=np.float64)
    for model_name in sorted(score_family, key=lambda value: value.encode("utf-8")):
        scores = np.asarray(score_family[model_name], dtype=np.float64)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise QualityProbeCoreV94Error("Bootstrap score vector drift")
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
            descending_worlds = row_world_index[descending]
            group_starts = np.flatnonzero(
                np.concatenate(
                    (np.asarray([True]), descending_scores[1:] != descending_scores[:-1])
                )
            )
        for start in range(0, len(draws), batch_size):
            stop = min(start + batch_size, len(draws))
            weights = multiplicities[start:stop]
            if use_dense:
                positive = weights @ group_positive.T
                negative = weights @ group_negative.T
                negative_before = np.cumsum(negative, axis=1) - negative
                positive_desc = positive[:, ::-1]
                total_desc = (positive + negative)[:, ::-1]
            else:
                row_weights = weights[:, descending_worlds]
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
                raise QualityProbeCoreV94Error("Bootstrap replicate lost a class")
            auc = np.sum(
                positive * (negative_before + 0.5 * negative), axis=1
            ) / (total_positive * total_negative)
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
            family_auc[start:stop] = np.maximum(
                family_auc[start:stop], np.maximum(auc, 1.0 - auc)
            )
            family_ap[start:stop] = np.maximum(family_ap[start:stop], ap_uplift)
    if not np.isfinite(family_auc).all() or not np.isfinite(family_ap).all():
        raise QualityProbeCoreV94Error("Bootstrap family maxima are nonfinite")
    return {
        "replicates": len(draws),
        "world_count": len(worlds),
        "score_family_size": len(score_family),
        "draws_raw_i8_c_sha256": hashlib.sha256(draws.tobytes(order="C")).hexdigest(),
        "family_max_symmetric_auc_vector_sha256": hashlib.sha256(
            family_auc.astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "family_max_average_precision_uplift_vector_sha256": hashlib.sha256(
            family_ap.astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "symmetric_auc_95_upper": float(
            np.quantile(family_auc, 0.95, method="linear")
        ),
        "average_precision_uplift_95_upper": float(
            np.quantile(family_ap, 0.95, method="linear")
        ),
    }


def _single_feature_maximum(matrix: FrozenMatrix, labels: np.ndarray) -> float:
    best = -math.inf
    for column in range(matrix.values.shape[1]):
        best = max(
            best,
            symmetric_auc(labels, matrix.values[:, column]),
        )
    if not math.isfinite(best):
        raise QualityProbeCoreV94Error("Single-feature scan is nonfinite")
    return float(best)


def _evaluate_family(
    *,
    train: Mapping[str, FrozenMatrix],
    development: Mapping[str, FrozenMatrix],
    train_labels: np.ndarray,
    development_labels: np.ndarray,
    train_label_row_keys: Sequence[tuple[str, str]],
    development_label_row_keys: Sequence[tuple[str, str]],
    policy: Mapping[str, Any],
    average_precision_baseline: float,
    bootstrap: bool,
) -> dict[str, Any]:
    views = _validate_family(train, development)
    train_y = _validate_labels(
        train_labels, len(next(iter(train.values())).row_keys), label="train"
    )
    development_y = _validate_labels(
        development_labels,
        len(next(iter(development.values())).row_keys), label="development"
    )
    if (
        tuple(train_label_row_keys) != next(iter(train.values())).row_keys
        or tuple(development_label_row_keys)
        != next(iter(development.values())).row_keys
    ):
        raise QualityProbeCoreV94Error("Label/matrix row-key join drift")
    for matrix in (*train.values(), *development.values()):
        verify_frozen_matrix(matrix)
    if not 0.0 < average_precision_baseline < 1.0:
        raise QualityProbeCoreV94Error("Average-precision baseline drift")
    if not math.isclose(
        average_precision_baseline,
        float(development_y.mean()),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise QualityProbeCoreV94Error(
            "Average-precision baseline does not equal development prevalence"
        )
    single_feature: dict[str, float] = {}
    model_results: dict[str, dict[str, float]] = {}
    score_family: dict[str, np.ndarray] = {}
    for view in views:
        single_feature[view] = _single_feature_maximum(
            development[view], development_y
        )
        scores = _fit_probe_models(
            train_x=np.ascontiguousarray(train[view].values),
            train_y=train_y,
            development_x=np.ascontiguousarray(development[view].values),
            policy=policy,
        )
        for model, score in scores.items():
            qualified = f"{view}::{model}"
            score = np.ascontiguousarray(score, dtype=np.float64)
            if score.shape != development_y.shape or not np.isfinite(score).all():
                raise QualityProbeCoreV94Error("Probe score vector drift")
            score_family[qualified] = score
            model_results[qualified] = {
                "symmetric_roc_auc": symmetric_auc(
                    development_y, score
                ),
                "average_precision": float(
                    average_precision_score(development_y, score)
                ),
                "score_vector_sha256": hashlib.sha256(
                    score.astype("<f8", copy=False).tobytes(order="C")
                ).hexdigest(),
            }
    family_auc = max(value["symmetric_roc_auc"] for value in model_results.values())
    family_ap_uplift = max(
        value["average_precision"] - average_precision_baseline
        for value in model_results.values()
    )
    output: dict[str, Any] = {
        "single_feature_maximum_symmetric_roc_auc_by_view": single_feature,
        "model_results": model_results,
        "maximum_symmetric_roc_auc": float(family_auc),
        "maximum_average_precision_uplift": float(family_ap_uplift),
        "bootstrap": None,
    }
    if bootstrap:
        bootstrap_policy = policy["bootstrap"]
        draws = generate_bootstrap_draws(
            replicates=bootstrap_policy["replicates"],
            world_count=bootstrap_policy["development_world_count"],
            seed=bootstrap_policy["seed"],
        )
        observed_draw_hash = hashlib.sha256(draws.tobytes(order="C")).hexdigest()
        if observed_draw_hash != bootstrap_policy["draws_raw_i8_c_sha256"]:
            raise QualityProbeCoreV94Error("Bootstrap draw commitment drift")
        world_uids = tuple(key[0] for key in next(iter(development.values())).row_keys)
        ordered_worlds = tuple(
            sorted(set(world_uids), key=lambda value: value.encode("utf-8"))
        )
        output["bootstrap"] = _bootstrap_family_upper(
            labels=development_y,
            row_world_uids=world_uids,
            ordered_world_uids=ordered_worlds,
            score_family=score_family,
            baseline=average_precision_baseline,
            draws=draws,
            batch_size=bootstrap_policy["streaming_batch_size"],
        )
    for matrix in (*train.values(), *development.values()):
        verify_frozen_matrix(matrix)
    return output
