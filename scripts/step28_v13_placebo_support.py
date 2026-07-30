#!/usr/bin/env python3
"""Label-free M1/M2 support-comparability preflight for Step 28-v13."""

from __future__ import annotations

import hashlib
import math
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import step28_v13_common as common


PREFLIGHT_VERSION = (
    "2026-07-28-step28-v13-placebo-support-preflight-v1-draft"
)
EVIDENCE_LEVEL = (
    "DEVELOPMENT_LABEL_FREE_SUPPORT_PREFLIGHT_NOT_FORMAL_CUSTODY"
)
PLACEBO_HASH_FIELDS = (
    "rewire_seed_id",
    "identity33_all_pairs",
    "feature_derangement_mapping",
    "joint_vector_multiset_exact_by_world_and_universe",
    "endpoint_disjoint_bijection_exact",
    "labels_or_controller_inputs_read",
    "candidate_trigger_or_audit_inputs_read",
)


def _registered_seed_ids(
    policy: Mapping[str, Any], *, mode: str
) -> list[str]:
    seeds = list(policy["randomness"][mode]["rewire_key_hexes"])
    ids = [
        "rws_" + hashlib.sha256(bytes.fromhex(seed_hex)).hexdigest()
        for seed_hex in seeds
    ]
    if (
        len(ids) != int(policy["placebo"]["replicates"])
        or len(set(ids)) != 5
    ):
        raise common.ContractError(
            "Support preflight does not have five registered seed IDs"
        )
    return ids


def _validate_placebo_result(
    policy: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    expected_seed_id: str,
    m2_rows_by_key: Mapping[
        tuple[str, str], Mapping[str, Any]
    ],
    endpoint_index: Mapping[
        tuple[str, str], tuple[str, str]
    ],
    candidate_keys: set[tuple[str, str]],
) -> None:
    required_keys = {*PLACEBO_HASH_FIELDS, "canonical_self_hash"}
    if (
        not isinstance(output, Mapping)
        or set(output) != required_keys
        or str(output.get("rewire_seed_id")) != expected_seed_id
        or output.get("labels_or_controller_inputs_read") is not False
        or output.get("candidate_trigger_or_audit_inputs_read") is not False
        or output.get(
            "joint_vector_multiset_exact_by_world_and_universe"
        )
        is not True
        or output.get("endpoint_disjoint_bijection_exact") is not True
    ):
        raise common.ContractError(
            "Support preflight placebo boundary drift"
        )
    expected_hash = common.canonical_sha256(
        {name: output[name] for name in PLACEBO_HASH_FIELDS}
    )
    if output.get("canonical_self_hash") != expected_hash:
        raise common.ContractError(
            "Support preflight placebo self-hash mismatch"
        )
    mapping = output["feature_derangement_mapping"]
    matrix = output["identity33_all_pairs"]
    mapping_schema = policy["placebo"][
        "feature_derangement_mapping_schema"
    ]
    matrix_schema = [
        "canonical_pair_uid",
        "world_uid",
        *policy["history_features"]["feature_names"],
    ]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3780
        or any(list(row) != matrix_schema for row in matrix)
        or len(
            {
                (
                    str(row["world_uid"]),
                    str(row["canonical_pair_uid"]),
                )
                for row in matrix
            }
        )
        != 3780
        or not isinstance(mapping, list)
        or len(mapping) != 3780
        or any(list(row) != mapping_schema for row in mapping)
        or any(row["endpoint_disjoint_bool"] is not True for row in mapping)
        or len(
            {
                (
                    str(row["world_uid"]),
                    str(row["destination_pair_uid"]),
                )
                for row in mapping
            }
        )
        != 3780
        or len(
            {
                (
                    str(row["world_uid"]),
                    str(row["universe"]),
                    str(row["source_pair_uid"]),
                )
                for row in mapping
            }
        )
        != 3780
        or any(
            str(row["rewire_seed_id"]) != expected_seed_id
            for row in mapping
        )
    ):
        raise common.ContractError(
            "Support preflight derangement mapping drift"
        )
    m1_rows_by_key = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"])): row
        for row in matrix
    }
    if (
        len(m1_rows_by_key) != 3780
        or set(m1_rows_by_key) != set(m2_rows_by_key)
        or set(m2_rows_by_key) != set(endpoint_index)
    ):
        raise common.ContractError(
            "Support preflight derangement matrix keyset drift"
        )
    feature_names = list(policy["history_features"]["feature_names"])
    source_keys_by_stratum: defaultdict[
        tuple[str, str], set[tuple[str, str]]
    ] = defaultdict(set)
    destination_keys_by_stratum: defaultdict[
        tuple[str, str], set[tuple[str, str]]
    ] = defaultdict(set)
    for row in mapping:
        world_uid = str(row["world_uid"])
        destination_key = (
            world_uid,
            str(row["destination_pair_uid"]),
        )
        source_key = (world_uid, str(row["source_pair_uid"]))
        expected_universe = (
            "primary_c40"
            if destination_key in candidate_keys
            else "secondary_complement"
        )
        if (
            str(row["universe"]) != expected_universe
            or destination_key not in endpoint_index
            or source_key not in endpoint_index
            or (
                (source_key in candidate_keys)
                != (destination_key in candidate_keys)
            )
            or destination_key == source_key
            or set(endpoint_index[destination_key]).intersection(
                endpoint_index[source_key]
            )
        ):
            raise common.ContractError(
                "Support preflight derangement endpoint lineage drift"
            )
        source_vector = [
            str(m2_rows_by_key[source_key][name])
            for name in feature_names
        ]
        destination_vector = [
            str(m1_rows_by_key[destination_key][name])
            for name in feature_names
        ]
        if (
            source_vector != destination_vector
            or str(row["feature_vector_sha256"])
            != common.canonical_sha256(source_vector)
        ):
            raise common.ContractError(
                "Support preflight derangement vector lineage drift"
            )
        stratum = (world_uid, expected_universe)
        source_keys_by_stratum[stratum].add(source_key)
        destination_keys_by_stratum[stratum].add(destination_key)
    if (
        set(source_keys_by_stratum) != set(destination_keys_by_stratum)
        or any(
            source_keys_by_stratum[stratum]
            != destination_keys_by_stratum[stratum]
            for stratum in destination_keys_by_stratum
        )
    ):
        raise common.ContractError(
            "Support preflight derangement source/destination bijection drift"
        )


def _matrix_index(
    policy: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[tuple[str, str], np.ndarray]:
    feature_names = [
        str(value) for value in policy["history_features"]["feature_names"]
    ]
    schema = ["canonical_pair_uid", "world_uid", *feature_names]
    output: dict[tuple[str, str], np.ndarray] = {}
    for row in rows:
        if list(row) != schema:
            raise common.ContractError(
                f"{label} identity33 schema/order drift"
            )
        key = (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        if key in output or not key[0] or not key[1]:
            raise common.ContractError(f"{label} identity33 key drift")
        try:
            values = np.asarray(
                [float(row[name]) for name in feature_names],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise common.ContractError(
                f"{label} identity33 contains a nonnumeric value"
            ) from exc
        if values.shape != (33,) or not np.isfinite(values).all():
            raise common.ContractError(
                f"{label} identity33 numeric-domain drift"
            )
        output[key] = values
    if not output:
        raise common.ContractError(f"{label} identity33 matrix is empty")
    return output


def _pair_universes(
    policy: Mapping[str, Any],
    *,
    candidate_pairs: Sequence[Mapping[str, Any]],
    complete_pair_endpoints: Sequence[Mapping[str, Any]],
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    dict[tuple[str, str], tuple[str, str]],
]:
    candidate_schema = policy["candidate_design"][
        "public_safe_projection_columns"
    ]
    complete_schema = policy["relational_integrity"][
        "pair_projection_contract"
    ]["complete_model_pair_endpoints_schema"]
    if any(list(row) != candidate_schema for row in candidate_pairs):
        raise common.ContractError(
            "Support preflight C40 endpoint schema drift"
        )
    if any(list(row) != complete_schema for row in complete_pair_endpoints):
        raise common.ContractError(
            "Support preflight complete-pair schema drift"
        )
    primary = [
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in candidate_pairs
    ]
    secondary = [
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in complete_pair_endpoints
    ]
    endpoint_index: dict[tuple[str, str], tuple[str, str]] = {}
    complete_rows_by_key: dict[
        tuple[str, str], Mapping[str, Any]
    ] = {}
    for row in complete_pair_endpoints:
        key = (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        if (
            key in endpoint_index
            or not key[0]
            or common.utf8_sort((left, right)) != [left, right]
            or key[1] != common.canonical_pair_uid(left, right)
        ):
            raise common.ContractError(
                "Support preflight complete-pair lineage drift"
            )
        endpoint_index[key] = (left, right)
        complete_rows_by_key[key] = row
    for row in candidate_pairs:
        key = (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        parent = complete_rows_by_key.get(key)
        if parent is None or any(
            str(row[name]) != str(parent[name])
            for name in complete_schema
        ):
            raise common.ContractError(
                "Support preflight C40 endpoint lineage drift"
            )
    if len(set(primary)) != len(primary) or len(set(secondary)) != len(
        secondary
    ):
        raise common.ContractError(
            "Support preflight pair universe contains duplicate keys"
        )
    if not set(primary).issubset(secondary):
        raise common.ContractError(
            "Support preflight C40 is outside the complete pair universe"
        )
    primary_counts = Counter(world_uid for world_uid, _pair_uid in primary)
    secondary_counts = Counter(
        world_uid for world_uid, _pair_uid in secondary
    )
    if (
        len(primary_counts) != 10
        or set(primary_counts.values()) != {40}
        or primary_counts.keys() != secondary_counts.keys()
        or set(secondary_counts.values()) != {378}
    ):
        raise common.ContractError(
            "Support preflight train pair counts drift"
        )
    return (
        sorted(primary, key=lambda value: (value[0].encode(), value[1].encode())),
        sorted(
            secondary,
            key=lambda value: (value[0].encode(), value[1].encode()),
        ),
        endpoint_index,
    )


def _stack(
    matrix: Mapping[tuple[str, str], np.ndarray],
    keys: Sequence[tuple[str, str]],
    *,
    label: str,
) -> np.ndarray:
    if set(matrix) != set(keys) and len(keys) == len(matrix):
        raise common.ContractError(f"{label} identity33 keyset drift")
    missing = set(keys) - set(matrix)
    if missing:
        raise common.ContractError(f"{label} identity33 lacks pair rows")
    output = np.vstack([matrix[key] for key in keys]).astype(
        np.float64, copy=False
    )
    if output.shape != (len(keys), 33) or not np.isfinite(output).all():
        raise common.ContractError(f"{label} matrix shape/domain drift")
    return output


def _shared_scale(m2_primary: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(m2_primary), axis=0))
    scale = np.where(scale <= 1e-12, 1.0, scale)
    if scale.shape != (33,) or not np.isfinite(scale).all():
        raise common.ContractError("Support preflight RMS scale is invalid")
    return scale


def _correlation_pair(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float | None, bool, tuple[int, int] | None]:
    left_mean = np.mean(left, axis=0)
    right_mean = np.mean(right, axis=0)
    left_cov = np.cov(left, rowvar=False, ddof=0)
    right_cov = np.cov(right, rowvar=False, ddof=0)
    left_std = np.sqrt(np.diag(left_cov))
    right_std = np.sqrt(np.diag(right_cov))
    left_zero = left_std <= 1e-12
    right_zero = right_std <= 1e-12
    if np.any(left_zero != right_zero):
        return None, False, None
    both_zero = left_zero & right_zero
    if np.any(
        both_zero
        & ~np.isclose(left_mean, right_mean, rtol=0.0, atol=1e-12)
    ):
        return None, False, None

    def one(covariance: np.ndarray, std: np.ndarray) -> np.ndarray:
        denominator = np.outer(std, std)
        output = np.zeros_like(covariance)
        np.divide(
            covariance,
            denominator,
            out=output,
            where=denominator > 1e-24,
        )
        np.fill_diagonal(output, 1.0)
        return output

    differences = np.abs(
        one(left_cov, left_std) - one(right_cov, right_std)
    )
    flat_index = int(np.argmax(differences))
    pair_index = tuple(
        int(value) for value in np.unravel_index(flat_index, differences.shape)
    )
    difference = float(differences[pair_index])
    return (
        difference,
        bool(math.isfinite(difference)),
        pair_index,
    )


def _fold_assignment(
    worlds: Sequence[str],
    *,
    random_seed: int,
    fold_count: int,
) -> dict[str, int]:
    prefix = str(random_seed).encode("ascii") + common.FIELD_SEPARATOR
    ordered = sorted(
        worlds,
        key=lambda world_uid: (
            hashlib.sha256(prefix + world_uid.encode("utf-8")).digest(),
            world_uid.encode("utf-8"),
        ),
    )
    assignment = {
        world_uid: ordinal % fold_count
        for ordinal, world_uid in enumerate(ordered)
    }
    if Counter(assignment.values()) != Counter(
        {fold: 2 for fold in range(fold_count)}
    ):
        raise common.ContractError(
            "Support preflight grouped-fold balance drift"
        )
    return assignment


def _two_sample_auc(
    policy: Mapping[str, Any],
    *,
    keys: Sequence[tuple[str, str]],
    m2: np.ndarray,
    m1: np.ndarray,
) -> tuple[float, str]:
    contract = policy["placebo"]["support_classifier"]
    fold_count = int(contract["world_grouped_folds"])
    random_seed = int(contract["random_seed"])
    worlds = common.utf8_sort({world_uid for world_uid, _pair_uid in keys})
    fold_by_world = _fold_assignment(
        worlds,
        random_seed=random_seed,
        fold_count=fold_count,
    )
    x_rows: list[np.ndarray] = []
    labels: list[int] = []
    folds: list[int] = []
    for row_index, (world_uid, _pair_uid) in enumerate(keys):
        for source_label, matrix in ((0, m2), (1, m1)):
            x_rows.append(matrix[row_index])
            labels.append(source_label)
            folds.append(fold_by_world[world_uid])
    x = np.vstack(x_rows)
    y = np.asarray(labels, dtype=np.int64)
    fold_array = np.asarray(folds, dtype=np.int64)
    scores = np.full(y.shape, np.nan, dtype=np.float64)
    for fold in range(fold_count):
        train = fold_array != fold
        test = fold_array == fold
        if (
            not train.any()
            or not test.any()
            or set(y[train]) != {0, 1}
            or set(y[test]) != {0, 1}
        ):
            raise common.ContractError(
                "Support classifier fold is empty or single-class"
            )
        model = LogisticRegression(
            solver=str(contract["solver"]),
            penalty=str(contract["penalty"]),
            C=float(contract["C"]),
            max_iter=int(contract["max_iter"]),
            tol=float(contract["tol"]),
            class_weight=contract["class_weight"],
            fit_intercept=bool(contract["fit_intercept"]),
            random_state=random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                model.fit(x[train], y[train])
            except ConvergenceWarning as exc:
                raise common.ContractError(
                    "Support classifier did not converge"
                ) from exc
        if (
            not np.isfinite(model.coef_).all()
            or not np.isfinite(model.intercept_).all()
            or np.any(model.n_iter_ >= int(contract["max_iter"]))
        ):
            raise common.ContractError(
                "Support classifier fit is nonfinite or at its iteration cap"
            )
        scores[test] = model.decision_function(x[test])
    if not np.isfinite(scores).all():
        raise common.ContractError("Support classifier OOF scores are nonfinite")
    auc = float(roc_auc_score(y, scores))
    symmetric = max(auc, 1.0 - auc)
    if not math.isfinite(symmetric):
        raise common.ContractError("Support classifier AUC is nonfinite")
    fold_hash = common.canonical_sha256(
        [
            {"world_uid": world_uid, "fold": fold_by_world[world_uid]}
            for world_uid in common.utf8_sort(fold_by_world)
        ]
    )
    return symmetric, fold_hash


def _comparison(
    policy: Mapping[str, Any],
    *,
    keys: Sequence[tuple[str, str]],
    m2_raw: np.ndarray,
    m1_raw: np.ndarray,
    scale: np.ndarray,
    validity_gate: bool,
) -> dict[str, Any]:
    m2 = m2_raw / scale
    m1 = m1_raw / scale
    thresholds = policy["placebo"]["support_comparability_thresholds"]
    m2_min = np.min(m2, axis=0)
    m2_max = np.max(m2, axis=0)
    m1_min = np.min(m1, axis=0)
    m1_max = np.max(m1, axis=0)
    feature_names = [
        str(value) for value in policy["history_features"]["feature_names"]
    ]
    range_excess_by_feature = np.maximum(
        np.maximum(m2_min - m1_min, 0.0),
        np.maximum(m1_max - m2_max, 0.0),
    )
    range_feature_index = int(np.argmax(range_excess_by_feature))
    range_excess = float(range_excess_by_feature[range_feature_index])
    mean_difference = np.mean(m1, axis=0) - np.mean(m2, axis=0)
    pooled_scale = np.sqrt(
        (np.var(m2, axis=0, ddof=0) + np.var(m1, axis=0, ddof=0))
        / 2.0
    )
    smd_valid = bool(
        np.all(
            (pooled_scale > 1e-12)
            | np.isclose(mean_difference, 0.0, rtol=0.0, atol=1e-12)
        )
    )
    smd = np.zeros(33, dtype=np.float64)
    np.divide(
        mean_difference,
        pooled_scale,
        out=smd,
        where=pooled_scale > 1e-12,
    )
    smd_feature_index = int(np.argmax(np.abs(smd)))
    maximum_smd = float(abs(smd[smd_feature_index])) if smd_valid else None
    zero_differences = np.abs(
        np.mean(m1 == 0.0, axis=0)
        - np.mean(m2 == 0.0, axis=0)
    )
    zero_feature_index = int(np.argmax(zero_differences))
    zero_rate_difference = float(zero_differences[zero_feature_index])
    probabilities = np.asarray(
        thresholds["quantiles"], dtype=np.float64
    )
    quantile_differences = np.abs(
        np.quantile(m1, probabilities, axis=0, method="linear")
        - np.quantile(m2, probabilities, axis=0, method="linear")
    )
    quantile_flat_index = int(np.argmax(quantile_differences))
    quantile_index, quantile_feature_index = (
        int(value)
        for value in np.unravel_index(
            quantile_flat_index, quantile_differences.shape
        )
    )
    quantile_difference = float(
        quantile_differences[quantile_index, quantile_feature_index]
    )
    covariance_differences = np.abs(
        np.cov(m1, rowvar=False, ddof=0)
        - np.cov(m2, rowvar=False, ddof=0)
    )
    covariance_flat_index = int(np.argmax(covariance_differences))
    covariance_pair = tuple(
        int(value)
        for value in np.unravel_index(
            covariance_flat_index, covariance_differences.shape
        )
    )
    covariance_difference = float(covariance_differences[covariance_pair])
    (
        correlation_difference,
        correlation_valid,
        correlation_pair,
    ) = _correlation_pair(m2, m1)
    auc_symmetric, fold_hash = _two_sample_auc(
        policy,
        keys=keys,
        m2=m2,
        m1=m1,
    )
    gates = {
        "range_slack_pass": (
            range_excess
            <= float(thresholds["range_slack_in_m2_scale_units"])
        ),
        "standardized_mean_difference_pass": (
            smd_valid
            and maximum_smd is not None
            and maximum_smd
            <= float(
                thresholds[
                    "maximum_absolute_standardized_mean_difference"
                ]
            )
        ),
        "zero_rate_difference_pass": (
            zero_rate_difference
            <= float(thresholds["maximum_absolute_zero_rate_difference"])
        ),
        "quantile_difference_pass": (
            quantile_difference
            <= float(
                thresholds[
                    "maximum_absolute_standardized_quantile_difference"
                ]
            )
        ),
        "covariance_difference_pass": (
            covariance_difference
            <= float(
                thresholds["maximum_absolute_covariance_difference"]
            )
        ),
        "correlation_difference_pass": (
            correlation_valid
            and correlation_difference is not None
            and correlation_difference
            <= float(
                thresholds["maximum_absolute_correlation_difference"]
            )
        ),
        "two_sample_auc_pass": (
            auc_symmetric
            <= float(thresholds["two_sample_auc_symmetric_maximum"])
        ),
    }
    return {
        "row_count_per_source": len(keys),
        "validity_gate": validity_gate,
        "maximum_range_excess": range_excess,
        "maximum_range_excess_feature": feature_names[
            range_feature_index
        ],
        "maximum_absolute_standardized_mean_difference": maximum_smd,
        "maximum_absolute_standardized_mean_difference_feature": (
            feature_names[smd_feature_index] if smd_valid else None
        ),
        "standardized_mean_difference_support_valid": smd_valid,
        "maximum_absolute_zero_rate_difference": zero_rate_difference,
        "maximum_absolute_zero_rate_difference_feature": feature_names[
            zero_feature_index
        ],
        "maximum_absolute_standardized_quantile_difference": (
            quantile_difference
        ),
        "maximum_absolute_standardized_quantile_difference_feature": (
            feature_names[quantile_feature_index]
        ),
        "maximum_absolute_standardized_quantile_difference_probability": (
            float(probabilities[quantile_index])
        ),
        "maximum_absolute_covariance_difference": covariance_difference,
        "maximum_absolute_covariance_difference_features": [
            feature_names[covariance_pair[0]],
            feature_names[covariance_pair[1]],
        ],
        "maximum_absolute_correlation_difference": correlation_difference,
        "maximum_absolute_correlation_difference_features": (
            [
                feature_names[correlation_pair[0]],
                feature_names[correlation_pair[1]],
            ]
            if correlation_pair is not None
            else None
        ),
        "correlation_support_valid": correlation_valid,
        "two_sample_auc_symmetric": auc_symmetric,
        "fold_assignment_sha256": fold_hash,
        "gates": gates,
        "all_thresholds_pass": all(gates.values()),
    }


def run_support_comparability_preflight(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    m2_identity33_all_pairs: Sequence[Mapping[str, Any]],
    candidate_pairs: Sequence[Mapping[str, Any]],
    complete_pair_endpoints: Sequence[Mapping[str, Any]],
    placebos: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare five M1 matrices with M2 without labels or oracle fields."""

    common.validate_policy(policy, mode=mode)
    if mode != "development_smoke" or split != "train":
        raise common.ContractError(
            "Support-comparability preflight is smoke-train only"
        )
    expected_seed_ids = _registered_seed_ids(policy, mode=mode)
    by_seed = {str(row.get("rewire_seed_id")): row for row in placebos}
    if (
        len(placebos) != len(expected_seed_ids)
        or set(by_seed) != set(expected_seed_ids)
    ):
        raise common.ContractError(
            "Support preflight placebo seed set drift"
        )
    primary_keys, secondary_keys, endpoint_index = _pair_universes(
        policy,
        candidate_pairs=candidate_pairs,
        complete_pair_endpoints=complete_pair_endpoints,
    )
    m2_index = _matrix_index(
        policy, m2_identity33_all_pairs, label="M2"
    )
    m2_rows_by_key = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"])): row
        for row in m2_identity33_all_pairs
    }
    if set(m2_index) != set(secondary_keys):
        raise common.ContractError("M2 identity33 complete keyset drift")
    m2_primary = _stack(m2_index, primary_keys, label="M2 C40")
    m2_secondary = _stack(m2_index, secondary_keys, label="M2 full378")
    scale = _shared_scale(m2_primary)
    seed_results: list[dict[str, Any]] = []
    for seed_id in expected_seed_ids:
        output = by_seed[seed_id]
        _validate_placebo_result(
            policy,
            output,
            expected_seed_id=seed_id,
            m2_rows_by_key=m2_rows_by_key,
            endpoint_index=endpoint_index,
            candidate_keys=set(primary_keys),
        )
        if "identity33_all_pairs" not in output:
            raise common.ContractError(
                "Support preflight placebo lacks its identity33 matrix"
            )
        m1_index = _matrix_index(
            policy,
            output["identity33_all_pairs"],
            label=f"M1 {seed_id}",
        )
        if set(m1_index) != set(secondary_keys):
            raise common.ContractError(
                "M1 identity33 complete keyset drift"
            )
        m1_primary = _stack(
            m1_index, primary_keys, label=f"M1 {seed_id} C40"
        )
        m1_secondary = _stack(
            m1_index, secondary_keys, label=f"M1 {seed_id} full378"
        )
        primary = _comparison(
            policy,
            keys=primary_keys,
            m2_raw=m2_primary,
            m1_raw=m1_primary,
            scale=scale,
            validity_gate=True,
        )
        secondary = _comparison(
            policy,
            keys=secondary_keys,
            m2_raw=m2_secondary,
            m1_raw=m1_secondary,
            scale=scale,
            validity_gate=False,
        )
        seed_results.append(
            {
                "rewire_seed_id": seed_id,
                "m1_identity33_sha256": common.canonical_sha256(
                    output["identity33_all_pairs"]
                ),
                "primary_c40": primary,
                "secondary_full378": secondary,
                "primary_validity_pass": primary["all_thresholds_pass"],
            }
        )
    result = {
        "version": PREFLIGHT_VERSION,
        "evidence_level": EVIDENCE_LEVEL,
        "mode": mode,
        "split": split,
        "feature_count": 33,
        "world_count": 10,
        "primary_pair_count_per_source": len(primary_keys),
        "secondary_pair_count_per_source": len(secondary_keys),
        "m2_identity33_sha256": common.canonical_sha256(
            m2_identity33_all_pairs
        ),
        "shared_m2_c40_rms_scale_sha256": common.canonical_sha256(
            [f"{value:.17g}" for value in scale]
        ),
        "seed_results": seed_results,
        "all_five_primary_validity_pass": all(
            row["primary_validity_pass"] for row in seed_results
        ),
        "labels_or_controller_inputs_read": False,
        "candidate_trigger_or_audit_inputs_read": False,
        "formal_use_forbidden": True,
    }
    result["canonical_self_hash"] = common.canonical_sha256(result)
    return result
