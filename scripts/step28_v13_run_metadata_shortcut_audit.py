#!/usr/bin/env python3
"""Run the frozen Step 28-v13 null-nuisance shortcut audit."""

from __future__ import annotations

import argparse
import hashlib
import math
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import step28_v13_common as dataset_common
import step28_v13_metadata_shortcut_common as shortcut_common


REPORT_VERSION = shortcut_common.AUDIT_REPORT_VERSION
MANIFEST_VERSION = shortcut_common.AUDIT_MANIFEST_VERSION
MODEL_ORDER = ("logistic_l2", "gradient_tree", "rbf_svm")


def _utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def _exact_input_record_map(
    rows: Any,
    *,
    expected_roles: set[str],
) -> dict[str, dict[str, Any]]:
    return shortcut_common.exact_input_record_map(
        rows,
        expected_roles=expected_roles,
    )


def require_exact_environment(lock: Mapping[str, Any]) -> None:
    statistics = lock["statistics"]
    expected_numpy = str(statistics["numpy_version"])
    expected_sklearn = str(
        statistics["scikit_learn_version"]
    )
    if np.__version__ != expected_numpy:
        raise shortcut_common.ShortcutAuditError(
            "Formal shortcut audit NumPy version mismatch: "
            f"expected {expected_numpy}, observed {np.__version__}"
        )
    if sklearn.__version__ != expected_sklearn:
        raise shortcut_common.ShortcutAuditError(
            "Formal shortcut audit scikit-learn version mismatch: "
            f"expected {expected_sklearn}, "
            f"observed {sklearn.__version__}"
        )
    if np.random.PCG64DXSM.__name__ != "PCG64DXSM":
        raise shortcut_common.ShortcutAuditError(
            "Formal shortcut audit bit generator is unavailable"
        )


def assign_world_folds(
    world_uids: Sequence[str],
    *,
    seed: int,
    fold_count: int,
) -> dict[str, int]:
    if fold_count != 5 or seed != 2026072707:
        raise shortcut_common.ShortcutAuditError(
            "World-fold assignment contract drift"
        )
    unique = set(world_uids)
    if len(unique) < fold_count or any(not value for value in unique):
        raise shortcut_common.ShortcutAuditError(
            "World-fold assignment requires at least five valid worlds"
        )

    def key(world_uid: str) -> tuple[int, bytes]:
        digest = hashlib.sha256(
            str(seed).encode("ascii")
            + b"\x1f"
            + world_uid.encode("utf-8")
        ).digest()
        return (
            int.from_bytes(digest, byteorder="big", signed=False),
            world_uid.encode("utf-8"),
        )

    ordered = sorted(unique, key=key)
    output = {
        world_uid: ordinal % fold_count
        for ordinal, world_uid in enumerate(ordered)
    }
    if set(output.values()) != set(range(fold_count)):
        raise shortcut_common.ShortcutAuditError(
            "World-fold assignment produced an empty fold"
        )
    return output


def _validate_projection_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_world_count: int,
    expected_pairs_per_world: int = 40,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    pair_keys: set[str] = set()
    world_counts: Counter[str] = Counter()
    for source in rows:
        if tuple(source) != shortcut_common.PROJECTION_FIELDS:
            raise shortcut_common.ShortcutAuditError(
                "Audit projection schema/order drift"
            )
        row = {key: str(source[key]) for key in source}
        pair_uid = row["canonical_pair_uid"]
        world_uid = row["world_uid"]
        if not pair_uid or not world_uid or pair_uid in pair_keys:
            raise shortcut_common.ShortcutAuditError(
                "Audit projection key contract failed"
            )
        pair_keys.add(pair_uid)
        world_counts[world_uid] += 1
        for name in shortcut_common.PAIR_FEATURES:
            text = row[name]
            try:
                value = float(text)
            except ValueError as error:
                raise shortcut_common.ShortcutAuditError(
                    "Audit projection feature is not float64"
                ) from error
            if (
                not math.isfinite(value)
                or format(value, ".12f") != text
            ):
                raise shortcut_common.ShortcutAuditError(
                    "Audit projection float serialization drift"
                )
        count_difference = float(row["absdiff__item_count"])
        count_sum = float(row["sum__item_count"])
        if (
            not count_difference.is_integer()
            or not count_sum.is_integer()
            or not 0.0 <= count_difference <= 6.0
            or not 4.0 <= count_sum <= 16.0
            or int(count_sum - count_difference) % 2 != 0
            or not 2.0
            <= (count_sum - count_difference) / 2.0
            <= 8.0
            or not 2.0
            <= (count_sum + count_difference) / 2.0
            <= 8.0
        ):
            raise shortcut_common.ShortcutAuditError(
                "Audit projection item-count pair semantics failed"
            )
        bounded_features = (
            "title_missing_rate",
            "description_missing_rate",
            "time_bucket_probability_00",
            "time_bucket_probability_01",
            "time_bucket_probability_02",
            "time_bucket_probability_03",
        )
        for seller_feature in bounded_features:
            difference = float(
                row[f"absdiff__{seller_feature}"]
            )
            total = float(row[f"sum__{seller_feature}"])
            tolerance = 5e-12
            if (
                difference < -tolerance
                or difference > 1.0 + tolerance
                or total < -tolerance
                or total > 2.0 + tolerance
                or difference > total + tolerance
                or difference > 2.0 - total + tolerance
            ):
                raise shortcut_common.ShortcutAuditError(
                    "Audit projection bounded pair semantics failed"
                )
        time_sum = sum(
            float(
                row[
                    "sum__time_bucket_probability_"
                    f"{bucket:02d}"
                ]
            )
            for bucket in range(4)
        )
        if not math.isclose(
            time_sum,
            2.0,
            rel_tol=0.0,
            abs_tol=2e-11,
        ):
            raise shortcut_common.ShortcutAuditError(
                "Audit projection time probabilities do not sum to two"
            )
        output.append(row)
    if (
        len(world_counts) != expected_world_count
        or set(world_counts.values()) != {expected_pairs_per_world}
    ):
        raise shortcut_common.ShortcutAuditError(
            "Audit projection is not exact C40 per formal world"
        )
    return sorted(
        output,
        key=lambda row: (
            _utf8_key(row["world_uid"]),
            _utf8_key(row["canonical_pair_uid"]),
        ),
    )


def _validate_label_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_pair_keys: set[str],
) -> dict[str, int]:
    output: dict[str, int] = {}
    for source in rows:
        if tuple(source) != shortcut_common.LABEL_FIELDS:
            raise shortcut_common.ShortcutAuditError(
                "Audit label schema/order drift"
            )
        pair_uid = str(source["canonical_pair_uid"])
        label = str(source["label"])
        if (
            pair_uid in output
            or pair_uid not in expected_pair_keys
            or label not in {"0", "1"}
        ):
            raise shortcut_common.ShortcutAuditError(
                "Audit label key/value contract failed"
            )
        output[pair_uid] = int(label)
    if set(output) != expected_pair_keys:
        raise shortcut_common.ShortcutAuditError(
            "Audit projection and label keysets differ"
        )
    if set(output.values()) != {0, 1}:
        raise shortcut_common.ShortcutAuditError(
            "Audit split labels are single-class"
        )
    return output


def _fit_and_score_fold(
    model_key: str,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    if model_key == "logistic_l2":
        scaler = StandardScaler(
            with_mean=True,
            with_std=True,
        )
        transformed_train = scaler.fit_transform(x_train)
        transformed_test = scaler.transform(x_test)
        model: Any = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            fit_intercept=True,
            class_weight=None,
            max_iter=10000,
            tol=1e-10,
        )
    elif model_key == "gradient_tree":
        transformed_train = x_train
        transformed_test = x_test
        model = HistGradientBoostingClassifier(
            max_depth=2,
            max_iter=200,
            learning_rate=0.03,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=2026072707,
            class_weight=None,
        )
    elif model_key == "rbf_svm":
        scaler = StandardScaler(
            with_mean=True,
            with_std=True,
        )
        transformed_train = scaler.fit_transform(x_train)
        transformed_test = scaler.transform(x_test)
        model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=False,
            class_weight=None,
        )
    else:
        raise shortcut_common.ShortcutAuditError(
            f"Unknown frozen shortcut model: {model_key}"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(transformed_train, y_train)
        except ConvergenceWarning as error:
            raise shortcut_common.ShortcutAuditError(
                f"{model_key} failed to converge"
            ) from error
    scores = np.asarray(
        model.decision_function(transformed_test),
        dtype=np.float64,
    )
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise shortcut_common.ShortcutAuditError(
            f"{model_key} produced invalid decision scores"
        )
    return scores


def compute_oof_scores(
    *,
    x: np.ndarray,
    y: np.ndarray,
    world_uids: Sequence[str],
    fold_by_world: Mapping[str, int],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    row_count = len(y)
    if (
        x.shape != (row_count, len(shortcut_common.PAIR_FEATURES))
        or len(world_uids) != row_count
        or not np.all(np.isfinite(x))
        or not set(np.unique(y)).issubset({0, 1})
    ):
        raise shortcut_common.ShortcutAuditError(
            "OOF input matrix contract failed"
        )
    folds = np.asarray(
        [fold_by_world[world_uid] for world_uid in world_uids],
        dtype=np.int64,
    )
    score_by_model = {
        key: np.full(row_count, np.nan, dtype=np.float64)
        for key in MODEL_ORDER
    }
    fold_audit: list[dict[str, Any]] = []
    for fold in range(5):
        test_mask = folds == fold
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise shortcut_common.ShortcutAuditError(
                "OOF train or test fold is empty"
            )
        train_classes = set(np.unique(y[train_mask]).tolist())
        test_classes = set(np.unique(y[test_mask]).tolist())
        if train_classes != {0, 1} or test_classes != {0, 1}:
            raise shortcut_common.ShortcutAuditError(
                "OOF train or test fold is single-class"
            )
        train_worlds = {
            world_uids[index]
            for index in np.flatnonzero(train_mask)
        }
        test_worlds = {
            world_uids[index]
            for index in np.flatnonzero(test_mask)
        }
        if train_worlds & test_worlds:
            raise shortcut_common.ShortcutAuditError(
                "World leakage detected across an OOF fold"
            )
        for model_key in MODEL_ORDER:
            fold_scores = _fit_and_score_fold(
                model_key,
                x_train=x[train_mask],
                y_train=y[train_mask],
                x_test=x[test_mask],
            )
            if len(fold_scores) != int(np.sum(test_mask)):
                raise shortcut_common.ShortcutAuditError(
                    "OOF score cardinality drift"
                )
            score_by_model[model_key][test_mask] = fold_scores
        fold_audit.append(
            {
                "fold": fold,
                "train_world_count": len(train_worlds),
                "test_world_count": len(test_worlds),
                "train_row_count": int(np.sum(train_mask)),
                "test_row_count": int(np.sum(test_mask)),
                "train_class_count": 2,
                "test_class_count": 2,
            }
        )
    if any(
        not np.all(np.isfinite(values))
        for values in score_by_model.values()
    ):
        raise shortcut_common.ShortcutAuditError(
            "OOF score coverage is incomplete"
        )
    return score_by_model, folds, fold_audit


def symmetric_auc(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if (
        y.ndim != 1
        or scores.ndim != 1
        or len(y) != len(scores)
        or set(np.unique(y).tolist()) != {0, 1}
        or not np.all(np.isfinite(scores))
    ):
        raise shortcut_common.ShortcutAuditError(
            "AUC input validity failure"
        )
    auc = float(roc_auc_score(y, scores))
    symmetric = max(auc, 1.0 - auc)
    if not math.isfinite(auc) or not 0.5 <= symmetric <= 1.0:
        raise shortcut_common.ShortcutAuditError(
            "AUC output validity failure"
        )
    return auc, symmetric


def _split_bootstrap_seed(base_seed: int, split: str) -> int:
    digest = hashlib.sha256(
        str(base_seed).encode("ascii")
        + b"\x1f"
        + split.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def world_bootstrap_upper(
    *,
    y: np.ndarray,
    world_uids: Sequence[str],
    score_by_model: Mapping[str, np.ndarray],
    split: str,
    replicates: int,
    base_seed: int,
    statistics_sink: Callable[[np.ndarray], None] | None = None,
) -> tuple[float, str]:
    if split not in shortcut_common.SPLITS or replicates < 1:
        raise shortcut_common.ShortcutAuditError(
            "Bootstrap split or replicate contract failed"
        )
    ordered_worlds = sorted(
        set(world_uids),
        key=_utf8_key,
    )
    if len(ordered_worlds) < 2:
        raise shortcut_common.ShortcutAuditError(
            "World bootstrap requires multiple worlds"
        )
    world_ordinal = {
        world_uid: index
        for index, world_uid in enumerate(ordered_worlds)
    }
    row_world_ordinals = np.asarray(
        [world_ordinal[value] for value in world_uids],
        dtype=np.int64,
    )
    split_seed = _split_bootstrap_seed(base_seed, split)
    generator = np.random.Generator(np.random.PCG64DXSM(split_seed))
    draws = generator.integers(
        0,
        len(ordered_worlds),
        size=(replicates, len(ordered_worlds)),
        dtype=np.int64,
    )
    draw_sha256 = hashlib.sha256(
        np.ascontiguousarray(
            draws.astype(">u8", copy=False)
        ).tobytes(order="C")
    ).hexdigest()
    statistics = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        world_multiplicity = np.bincount(
            draws[replicate],
            minlength=len(ordered_worlds),
        ).astype(np.float64, copy=False)
        sample_weight = world_multiplicity[row_world_ordinals]
        active = sample_weight > 0
        if (
            not np.any(active & (y == 0))
            or not np.any(active & (y == 1))
        ):
            raise shortcut_common.ShortcutAuditError(
                "World bootstrap produced a single-class replicate"
            )
        replicate_max = 0.5
        for model_key in MODEL_ORDER:
            auc = float(
                roc_auc_score(
                    y[active],
                    score_by_model[model_key][active],
                    sample_weight=sample_weight[active],
                )
            )
            replicate_max = max(
                replicate_max,
                auc,
                1.0 - auc,
            )
        statistics[replicate] = replicate_max
    if not np.all(np.isfinite(statistics)):
        raise shortcut_common.ShortcutAuditError(
            "World bootstrap statistic is nonfinite"
        )
    if statistics_sink is not None:
        statistics_sink(statistics.copy())
    upper = float(
        np.quantile(statistics, 0.95, method="higher")
    )
    return upper, draw_sha256


def run_audit(
    *,
    projection_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    split: str,
    expected_world_count: int,
    bootstrap_replicates: int,
    fold_seed: int = 2026072707,
    bootstrap_base_seed: int = 2026072711,
    point_maximum: float = 0.52,
    upper_maximum: float = 0.53,
    evidence_sink: Callable[[Mapping[str, np.ndarray]], None]
    | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if split not in shortcut_common.SPLITS:
        raise shortcut_common.ShortcutAuditError(
            "Unknown shortcut-audit split"
        )
    projections = _validate_projection_rows(
        projection_rows,
        expected_world_count=expected_world_count,
    )
    labels = _validate_label_rows(
        label_rows,
        expected_pair_keys={
            row["canonical_pair_uid"] for row in projections
        },
    )
    x = np.asarray(
        [
            [float(row[name]) for name in shortcut_common.PAIR_FEATURES]
            for row in projections
        ],
        dtype=np.float64,
    )
    y = np.asarray(
        [labels[row["canonical_pair_uid"]] for row in projections],
        dtype=np.int64,
    )
    world_uids = [row["world_uid"] for row in projections]
    fold_by_world = assign_world_folds(
        world_uids,
        seed=fold_seed,
        fold_count=5,
    )
    score_by_model, folds, fold_audit = compute_oof_scores(
        x=x,
        y=y,
        world_uids=world_uids,
        fold_by_world=fold_by_world,
    )
    model_metrics: dict[str, dict[str, float]] = {}
    point_statistic = 0.5
    for model_key in MODEL_ORDER:
        auc, auc_sym = symmetric_auc(y, score_by_model[model_key])
        model_metrics[model_key] = {
            "roc_auc": auc,
            "roc_auc_symmetric": auc_sym,
        }
        point_statistic = max(point_statistic, auc_sym)
    bootstrap_statistics: np.ndarray | None = None

    def capture_bootstrap(values: np.ndarray) -> None:
        nonlocal bootstrap_statistics
        bootstrap_statistics = values

    upper, draw_sha256 = world_bootstrap_upper(
        y=y,
        world_uids=world_uids,
        score_by_model=score_by_model,
        split=split,
        replicates=bootstrap_replicates,
        base_seed=bootstrap_base_seed,
        statistics_sink=(
            capture_bootstrap if evidence_sink is not None else None
        ),
    )
    if evidence_sink is not None:
        if bootstrap_statistics is None:
            raise shortcut_common.ShortcutAuditError(
                "Shortcut evidence capture is incomplete"
            )
        evidence_sink(
            {
                "folds": folds.copy(),
                "score_logistic_l2": score_by_model[
                    "logistic_l2"
                ].copy(),
                "score_gradient_tree": score_by_model[
                    "gradient_tree"
                ].copy(),
                "score_rbf_svm": score_by_model[
                    "rbf_svm"
                ].copy(),
                "bootstrap_statistics": bootstrap_statistics.copy(),
            }
        )
    point_gate = point_statistic <= point_maximum
    upper_gate = upper <= upper_maximum
    report = {
        "version": REPORT_VERSION,
        "status": (
            "PASS_METADATA_SHORTCUT_ONLY"
            if point_gate and upper_gate
            else "FAIL_METADATA_SHORTCUT_ONLY"
        ),
        "mode": "formal",
        "split": split,
        "row_count": len(projections),
        "world_count": expected_world_count,
        "rows_per_world": 40,
        "feature_count": len(shortcut_common.PAIR_FEATURES),
        "feature_order": list(shortcut_common.PAIR_FEATURES),
        "model_order": list(MODEL_ORDER),
        "fold_seed": fold_seed,
        "fold_count": 5,
        "fold_audit": fold_audit,
        "model_metrics": model_metrics,
        "point_statistic_max_auc_symmetric": point_statistic,
        "point_maximum": point_maximum,
        "point_gate_pass": point_gate,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_base_seed": bootstrap_base_seed,
        "bootstrap_draw_matrix_sha256": draw_sha256,
        "bootstrap_95_upper": upper,
        "bootstrap_95_upper_maximum": upper_maximum,
        "bootstrap_upper_gate_pass": upper_gate,
        "threshold_metrics_not_applicable": True,
        "permutation_test_preregistered": False,
        "pass_dataset_only_granted": False,
    }
    oof_rows: list[dict[str, str]] = []
    for index, row in enumerate(projections):
        oof_rows.append(
            {
                "canonical_pair_uid": row["canonical_pair_uid"],
                "world_uid": row["world_uid"],
                "label": str(y[index]),
                "fold": str(folds[index]),
                "score_logistic_l2": format(
                    score_by_model["logistic_l2"][index],
                    ".17g",
                ),
                "score_gradient_tree": format(
                    score_by_model["gradient_tree"][index],
                    ".17g",
                ),
                "score_rbf_svm": format(
                    score_by_model["rbf_svm"][index],
                    ".17g",
                ),
            }
        )
    return report, oof_rows


def _load_and_validate_projection_release(
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    split: str,
    projection_path: Path,
) -> tuple[
    list[dict[str, str]],
    dict[str, Any],
    dict[str, shortcut_common.FileSnapshot],
]:
    if projection_path.name != shortcut_common.PROJECTION_FILENAME:
        raise shortcut_common.ShortcutAuditError(
            "Audit projection basename drift"
        )
    manifest_path = (
        projection_path.parent
        / shortcut_common.PROJECTION_MANIFEST_FILENAME
    )
    manifest, manifest_snapshot = shortcut_common.load_json_snapshot(
        manifest_path
    )
    shortcut_common.validate_self_hash(
        manifest,
        label="projection manifest",
    )
    expected_manifest_keys = {
        "version",
        "status",
        "mode",
        "split",
        "row_count",
        "world_count",
        "rows_per_world",
        "projection_schema",
        "projection_content_sha256",
        "step",
        "stage",
        "run_id",
        "policy_sha256",
        "policy_contract_sha256",
        "producer_sha256",
        "lock_file_sha256",
        "lock_content_sha256",
        "source_closure_sha256",
        "parent_manifests",
        "upstream_custody_parent_seal_required",
        "input_allowlist",
        "access_isolation_status",
        "forbidden_open_count_not_self_asserted",
        "files",
        "canonical_self_hash",
    }
    if set(manifest) != expected_manifest_keys:
        raise shortcut_common.ShortcutAuditError(
            "Projection manifest schema drift"
        )
    shortcut_common.validate_manifest_identity(
        manifest,
        lock,
        lock_path=lock_path,
        stage="project_null_nuisance",
        producer_relative_path=(
            "scripts/step28_v13_project_null_nuisance.py"
        ),
    )
    if (
        manifest.get("version")
        != shortcut_common.PROJECTION_MANIFEST_VERSION
        or manifest.get("status") != "SEALED_LABEL_FREE_PROJECTION"
        or manifest.get("mode") != "formal"
        or manifest.get("split") != split
        or manifest.get("row_count")
        != int(lock["formal_world_counts"][split]) * 40
        or manifest.get("world_count")
        != int(lock["formal_world_counts"][split])
        or manifest.get("rows_per_world") != 40
        or manifest.get("projection_schema")
        != list(shortcut_common.PROJECTION_FIELDS)
        or manifest.get("lock_file_sha256")
        != dataset_common.sha256_file(lock_path)
        or manifest.get("lock_content_sha256")
        != lock["canonical_self_hash"]
        or manifest.get("source_closure_sha256")
        != lock["source_closure_sha256"]
        or manifest.get("access_isolation_status")
        != shortcut_common.BLOCKED_ACCESS_STATUS
        or manifest.get("forbidden_open_count_not_self_asserted")
        is not True
    ):
        raise shortcut_common.ShortcutAuditError(
            "Projection release manifest contract failed"
        )
    projection_inputs = _exact_input_record_map(
        manifest["input_allowlist"],
        expected_roles={
            "candidate_pairs",
            "history_item_index",
            "redacted_items",
        },
    )
    expected_input_basenames = lock["projector"][
        "input_basenames"
    ]
    if {
        role: row["basename"]
        for role, row in projection_inputs.items()
    } != expected_input_basenames:
        raise shortcut_common.ShortcutAuditError(
            "Projection manifest parent basename drift"
        )
    if (
        not isinstance(manifest.get("files"), list)
        or len(manifest["files"]) != 1
    ):
        raise shortcut_common.ShortcutAuditError(
            "Projection manifest file list drift"
        )
    rows, projection_snapshot = (
        shortcut_common.read_csv_exact_snapshot(
            projection_path,
            fieldnames=shortcut_common.PROJECTION_FIELDS,
        )
    )
    shortcut_common.validate_snapshot_file_record(
        manifest["files"][0],
        snapshot=projection_snapshot,
        path=shortcut_common.PROJECTION_FILENAME,
        role="null_nuisance_projection",
    )
    if shortcut_common.exact_file_set(projection_path.parent) != {
        shortcut_common.PROJECTION_FILENAME,
        shortcut_common.PROJECTION_MANIFEST_FILENAME,
    }:
        raise shortcut_common.ShortcutAuditError(
            "Projection release physical file set drift"
        )
    if (
        dataset_common.canonical_sha256(rows)
        != manifest["projection_content_sha256"]
    ):
        raise shortcut_common.ShortcutAuditError(
            "Projection release content hash drift"
        )
    return rows, manifest, {
        "sealed_projection": projection_snapshot,
        "projection_manifest": manifest_snapshot,
    }


def _load_and_validate_label_release(
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    split: str,
    labels_path: Path,
) -> tuple[
    list[dict[str, str]],
    dict[str, Any],
    dict[str, shortcut_common.FileSnapshot],
]:
    if labels_path.name != shortcut_common.LABEL_FILENAME:
        raise shortcut_common.ShortcutAuditError(
            "Audit label basename drift"
        )
    manifest_path = (
        labels_path.parent
        / shortcut_common.LABEL_MANIFEST_FILENAME
    )
    manifest, manifest_snapshot = shortcut_common.load_json_snapshot(
        manifest_path
    )
    shortcut_common.validate_self_hash(
        manifest,
        label="label manifest",
    )
    expected_manifest_keys = {
        "version",
        "status",
        "mode",
        "split",
        "row_count",
        "world_count",
        "rows_per_world",
        "label_schema",
        "formula",
        "formula_equality_required",
        "class_counts_withheld",
        "label_content_sha256",
        "step",
        "stage",
        "run_id",
        "policy_sha256",
        "policy_contract_sha256",
        "producer_sha256",
        "lock_file_sha256",
        "lock_content_sha256",
        "source_closure_sha256",
        "parent_manifests",
        "upstream_custody_parent_seal_required",
        "input_allowlist",
        "access_isolation_status",
        "forbidden_open_count_not_self_asserted",
        "files",
        "canonical_self_hash",
    }
    if set(manifest) != expected_manifest_keys:
        raise shortcut_common.ShortcutAuditError(
            "Label manifest schema drift"
        )
    shortcut_common.validate_manifest_identity(
        manifest,
        lock,
        lock_path=lock_path,
        stage="seal_classification_labels",
        producer_relative_path=(
            "scripts/step28_v13_seal_classification_labels.py"
        ),
    )
    if (
        manifest.get("version")
        != shortcut_common.LABEL_MANIFEST_VERSION
        or manifest.get("status")
        != "SEALED_PRIVATE_CLASSIFICATION_LABELS"
        or manifest.get("mode") != "formal"
        or manifest.get("split") != split
        or manifest.get("row_count")
        != int(lock["formal_world_counts"][split]) * 40
        or manifest.get("world_count")
        != int(lock["formal_world_counts"][split])
        or manifest.get("rows_per_world") != 40
        or manifest.get("label_schema")
        != list(shortcut_common.LABEL_FIELDS)
        or manifest.get("formula")
        != "int(controller(left)==controller(right))"
        or manifest.get("formula_equality_required") is not True
        or manifest.get("class_counts_withheld") is not True
        or manifest.get("lock_file_sha256")
        != dataset_common.sha256_file(lock_path)
        or manifest.get("lock_content_sha256")
        != lock["canonical_self_hash"]
        or manifest.get("source_closure_sha256")
        != lock["source_closure_sha256"]
        or manifest.get("access_isolation_status")
        != shortcut_common.BLOCKED_ACCESS_STATUS
        or manifest.get("forbidden_open_count_not_self_asserted")
        is not True
    ):
        raise shortcut_common.ShortcutAuditError(
            "Label release manifest contract failed"
        )
    label_inputs = _exact_input_record_map(
        manifest["input_allowlist"],
        expected_roles={
            "candidate_pairs",
            "controller_membership",
        },
    )
    if {
        role: row["basename"]
        for role, row in label_inputs.items()
    } != lock["label_sealer"]["input_basenames"]:
        raise shortcut_common.ShortcutAuditError(
            "Label manifest parent basename drift"
        )
    if (
        not isinstance(manifest.get("files"), list)
        or len(manifest["files"]) != 1
    ):
        raise shortcut_common.ShortcutAuditError(
            "Label manifest file list drift"
        )
    rows, label_snapshot = shortcut_common.read_csv_exact_snapshot(
        labels_path,
        fieldnames=shortcut_common.LABEL_FIELDS,
    )
    shortcut_common.validate_label_manifest_release(
        lock=lock,
        lock_path=lock_path,
        split=split,
        labels_path=labels_path,
        label_rows=rows,
        label_snapshot=label_snapshot,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_snapshot=manifest_snapshot,
    )
    shortcut_common.validate_snapshot_file_record(
        manifest["files"][0],
        snapshot=label_snapshot,
        path=shortcut_common.LABEL_FILENAME,
        role="private_classification_labels",
    )
    if shortcut_common.exact_file_set(labels_path.parent) != {
        shortcut_common.LABEL_FILENAME,
        shortcut_common.LABEL_MANIFEST_FILENAME,
    }:
        raise shortcut_common.ShortcutAuditError(
            "Label release physical file set drift"
        )
    if (
        dataset_common.canonical_sha256(rows)
        != manifest["label_content_sha256"]
    ):
        raise shortcut_common.ShortcutAuditError(
            "Label release content hash drift"
        )
    return rows, manifest, {
        "sealed_labels": label_snapshot,
        "label_manifest": manifest_snapshot,
    }


def _load_and_validate_formula_receipt(
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    split: str,
    receipt_path: Path,
    labels_path: Path,
    label_snapshot: shortcut_common.FileSnapshot,
    label_manifest_snapshot: shortcut_common.FileSnapshot,
    label_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], shortcut_common.FileSnapshot]:
    if (
        receipt_path.name
        != shortcut_common.LABEL_VALIDATION_FILENAME
    ):
        raise shortcut_common.ShortcutAuditError(
            "Label-formula receipt basename drift"
        )
    receipt, receipt_snapshot = shortcut_common.load_json_snapshot(
        receipt_path
    )
    if shortcut_common.exact_file_set(receipt_path.parent) != {
        shortcut_common.LABEL_VALIDATION_FILENAME
    }:
        raise shortcut_common.ShortcutAuditError(
            "Label-formula release physical file set drift"
        )
    shortcut_common.validate_self_hash(
        receipt,
        label="label-formula validation receipt",
    )
    expected_keys = {
        "version",
        "status",
        "mode",
        "split",
        "validated",
        "row_count",
        "world_count",
        "rows_per_world",
        "class_counts_withheld",
        "alternative_derivation",
        "step",
        "stage",
        "run_id",
        "policy_sha256",
        "policy_contract_sha256",
        "producer_sha256",
        "lock_file_sha256",
        "lock_content_sha256",
        "source_closure_sha256",
        "parent_manifests",
        "upstream_custody_parent_seal_required",
        "input_allowlist",
        "access_isolation_status",
        "forbidden_open_count_not_self_asserted",
        "canonical_self_hash",
    }
    if set(receipt) != expected_keys:
        raise shortcut_common.ShortcutAuditError(
            "Label-formula receipt schema drift"
        )
    shortcut_common.validate_manifest_identity(
        receipt,
        lock,
        lock_path=lock_path,
        stage="validate_label_formula",
        producer_relative_path=(
            "scripts/step28_v13_validate_label_formula.py"
        ),
        additional_parent_manifests=[
            {
                "role": "classification_label_manifest",
                "file_sha256": label_manifest_snapshot.sha256,
                "content_sha256": label_manifest[
                    "canonical_self_hash"
                ],
            }
        ],
    )
    if (
        receipt.get("version")
        != shortcut_common.LABEL_FORMULA_RECEIPT_VERSION
        or receipt.get("status") != "PASS_LABEL_FORMULA_ONLY"
        or receipt.get("mode") != "formal"
        or receipt.get("split") != split
        or receipt.get("validated") is not True
        or receipt.get("class_counts_withheld") is not True
        or receipt.get("row_count")
        != int(lock["formal_world_counts"][split]) * 40
        or receipt.get("world_count")
        != int(lock["formal_world_counts"][split])
        or receipt.get("rows_per_world") != 40
        or receipt.get("alternative_derivation")
        != shortcut_common.LABEL_FORMULA_ALTERNATIVE_DERIVATION
        or receipt.get("lock_file_sha256")
        != dataset_common.sha256_file(lock_path)
        or receipt.get("lock_content_sha256")
        != lock["canonical_self_hash"]
        or receipt.get("source_closure_sha256")
        != lock["source_closure_sha256"]
        or receipt.get("access_isolation_status")
        != shortcut_common.BLOCKED_ACCESS_STATUS
        or receipt.get("forbidden_open_count_not_self_asserted")
        is not True
    ):
        raise shortcut_common.ShortcutAuditError(
            "Label-formula receipt contract failed"
        )
    formula_inputs = _exact_input_record_map(
        receipt["input_allowlist"],
        expected_roles={
            "candidate_pairs",
            "controller_membership",
            "sealed_labels",
            "label_manifest",
        },
    )
    label_inputs = _exact_input_record_map(
        label_manifest["input_allowlist"],
        expected_roles={
            "candidate_pairs",
            "controller_membership",
        },
    )
    for role in ("candidate_pairs", "controller_membership"):
        if (
            formula_inputs[role].get("sha256")
            != label_inputs[role].get("sha256")
            or formula_inputs[role].get("size_bytes")
            != label_inputs[role].get("size_bytes")
            or formula_inputs[role].get("basename")
            != label_inputs[role].get("basename")
        ):
            raise shortcut_common.ShortcutAuditError(
                "Formula and label parent inputs differ"
            )
    sealed_label = formula_inputs["sealed_labels"]
    if (
        sealed_label.get("basename") != labels_path.name
        or sealed_label.get("size_bytes")
        != label_snapshot.size_bytes
        or sealed_label.get("sha256") != label_snapshot.sha256
    ):
        raise shortcut_common.ShortcutAuditError(
            "Formula receipt does not bind the sealed labels"
        )
    sealed_manifest = formula_inputs["label_manifest"]
    if (
        sealed_manifest.get("basename")
        != shortcut_common.LABEL_MANIFEST_FILENAME
        or sealed_manifest.get("size_bytes")
        != label_manifest_snapshot.size_bytes
        or sealed_manifest.get("sha256")
        != label_manifest_snapshot.sha256
    ):
        raise shortcut_common.ShortcutAuditError(
            "Formula receipt does not bind the label manifest"
        )
    return receipt, receipt_snapshot


def write_audit_release(
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    split: str,
    projection_path: Path,
    labels_path: Path,
    label_formula_receipt_path: Path,
    output_dir: Path,
) -> Path:
    shortcut_common.require_formal_execution_envelope(lock)
    shortcut_common.require_split_supervision_authorization(
        lock,
        split=split,
        operation="metadata_shortcut_audit",
    )
    require_exact_environment(lock)
    expected_basenames = lock["audit_runner"]["input_basenames"]
    if {
        "projection": projection_path.name,
        "labels": labels_path.name,
        "label_formula_receipt": label_formula_receipt_path.name,
    } != expected_basenames:
        raise shortcut_common.ShortcutAuditError(
            "Audit-runner input basename allow-list drift"
        )
    (
        projection_rows,
        _projection_manifest,
        projection_snapshots,
    ) = _load_and_validate_projection_release(
        lock=lock,
        lock_path=lock_path,
        split=split,
        projection_path=projection_path,
    )
    (
        label_rows,
        label_manifest,
        label_snapshots,
    ) = _load_and_validate_label_release(
        lock=lock,
        lock_path=lock_path,
        split=split,
        labels_path=labels_path,
    )
    _formula_receipt, formula_receipt_snapshot = (
        _load_and_validate_formula_receipt(
            lock=lock,
            lock_path=lock_path,
            split=split,
            receipt_path=label_formula_receipt_path,
            labels_path=labels_path,
            label_snapshot=label_snapshots["sealed_labels"],
            label_manifest_snapshot=(
                label_snapshots["label_manifest"]
            ),
            label_manifest=label_manifest,
        )
    )
    audit_parent_manifests = [
        {
            "role": "null_nuisance_projection_manifest",
            "file_sha256": projection_snapshots[
                "projection_manifest"
            ].sha256,
            "content_sha256": _projection_manifest[
                "canonical_self_hash"
            ],
        },
        {
            "role": "classification_label_manifest",
            "file_sha256": label_snapshots[
                "label_manifest"
            ].sha256,
            "content_sha256": label_manifest[
                "canonical_self_hash"
            ],
        },
        {
            "role": "label_formula_validation_receipt",
            "file_sha256": formula_receipt_snapshot.sha256,
            "content_sha256": _formula_receipt[
                "canonical_self_hash"
            ],
        },
    ]
    statistics = lock["statistics"]
    report, oof_rows = run_audit(
        projection_rows=projection_rows,
        label_rows=label_rows,
        split=split,
        expected_world_count=int(
            lock["formal_world_counts"][split]
        ),
        bootstrap_replicates=int(
            statistics["bootstrap"]["replicates"]
        ),
        fold_seed=int(statistics["fold_random_seed"]),
        bootstrap_base_seed=int(
            statistics["bootstrap"]["base_seed"]
        ),
        point_maximum=float(statistics["point_maximum"]),
        upper_maximum=float(
            statistics["bootstrap_95_upper_maximum"]
        ),
    )

    def writer(stage: Path) -> None:
        report_path = (
            stage / shortcut_common.AUDIT_REPORT_FILENAME
        )
        oof_path = stage / shortcut_common.OOF_FILENAME
        dataset_common.write_json(report_path, report)
        dataset_common.write_csv(
            oof_path,
            oof_rows,
            shortcut_common.OOF_FIELDS,
        )
        input_snapshots = {
            **projection_snapshots,
            **label_snapshots,
            "label_formula_validation": (
                formula_receipt_snapshot
            ),
        }
        input_rows = [
            input_snapshots[role].record(role=role)
            for role in (
                "sealed_projection",
                "projection_manifest",
                "sealed_labels",
                "label_manifest",
                "label_formula_validation",
            )
        ]
        manifest = shortcut_common.add_self_hash(
            {
                "version": MANIFEST_VERSION,
                "status": report["status"],
                "mode": "formal",
                "split": split,
                **shortcut_common.manifest_identity(
                    lock,
                    lock_path=lock_path,
                    stage="run_metadata_shortcut_audit",
                    producer_relative_path=(
                        "scripts/"
                        "step28_v13_run_metadata_shortcut_audit.py"
                    ),
                    additional_parent_manifests=(
                        audit_parent_manifests
                    ),
                ),
                "input_allowlist": input_rows,
                "access_isolation_status": (
                    shortcut_common.BLOCKED_ACCESS_STATUS
                ),
                "formal_scientific_claim_authorized": False,
                "report_content_sha256": (
                    dataset_common.canonical_sha256(report)
                ),
                "oof_content_sha256": (
                    dataset_common.canonical_sha256(oof_rows)
                ),
                "files": [
                    shortcut_common.file_record(
                        report_path,
                        role="metadata_shortcut_report",
                        root=stage,
                    ),
                    shortcut_common.file_record(
                        oof_path,
                        role="private_oof_scores",
                        root=stage,
                    ),
                ],
            }
        )
        dataset_common.write_json(
            stage / shortcut_common.AUDIT_MANIFEST_FILENAME,
            manifest,
        )

    return shortcut_common.publish_directory(
        output_dir,
        writer=writer,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=shortcut_common.DEFAULT_LOCK_PATH,
    )
    parser.add_argument(
        "--split",
        choices=shortcut_common.SPLITS,
        required=True,
    )
    parser.add_argument("--projection", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--label-formula-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = shortcut_common.load_lock(args.lock)
    if args.validate_config_only:
        print(
            "Step28-v13 metadata-shortcut runner is locked; "
            "formal execution remains blocked"
        )
        return
    shortcut_common.require_formal_execution_envelope(lock)
    shortcut_common.require_split_supervision_authorization(
        lock,
        split=args.split,
        operation="metadata_shortcut_audit",
    )
    required = (
        args.projection,
        args.labels,
        args.label_formula_receipt,
        args.output_dir,
    )
    if any(value is None for value in required):
        raise shortcut_common.ShortcutAuditError(
            "Shortcut audit requires every exact input and output"
        )
    try:
        require_exact_environment(lock)
        write_audit_release(
            lock=lock,
            lock_path=args.lock,
            split=args.split,
            projection_path=args.projection,
            labels_path=args.labels,
            label_formula_receipt_path=(
                args.label_formula_receipt
            ),
            output_dir=args.output_dir,
        )
    except Exception as error:
        shortcut_common.publish_stage_failure(
            lock=lock,
            lock_path=args.lock,
            split=args.split,
            stage="run_metadata_shortcut_audit_failure",
            producer_relative_path=(
                "scripts/"
                "step28_v13_run_metadata_shortcut_audit.py"
            ),
            output_dir=args.output_dir,
            error=error,
        )
        raise


if __name__ == "__main__":
    main()
