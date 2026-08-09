#!/usr/bin/env python3
"""Run the exact-scale v1.12 null-nuisance and join-shortcut preflight."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_preceremony as preceremony


ROOT = Path(__file__).resolve().parents[1]
NULL_SELLER_FEATURES = (
    "item_count",
    "title_missing_rate",
    "description_missing_rate",
    "time_bucket_probability_00",
    "time_bucket_probability_01",
    "time_bucket_probability_02",
    "time_bucket_probability_03",
)
NULL_PAIR_FEATURES = tuple(
    f"absdiff__{name}" for name in NULL_SELLER_FEATURES
) + tuple(f"sum__{name}" for name in NULL_SELLER_FEATURES)
JOIN_FEATURES = (
    "world_hash_01",
    "left_hash_01",
    "right_hash_01",
    "pair_hash_01",
    "pair_hash_02",
    "left_rank_01",
    "right_rank_01",
    "pair_rank_01",
    "absdiff_seller_hash_01",
    "sum_seller_hash_01",
)
FEATURES = (*NULL_PAIR_FEATURES, *JOIN_FEATURES)
ATTACK_MODELS = ("logistic_l2", "gradient_tree")


class ShortcutPreflightError(ValueError):
    """Raised when the exact shortcut preflight fails closed."""


@dataclass(frozen=True)
class LogisticArtifact:
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    audit: dict[str, Any]


def _unit_hash(domain: str, value: str) -> float:
    digest = hashlib.sha256(
        b"\x1f".join((domain.encode("ascii"), value.encode("utf-8")))
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False) / float(1 << 64)


def _seller_vectors(
    *,
    seller_uids: Sequence[str],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[float, ...]]:
    accumulators = {
        str(seller_uid): {
            "count": 0,
            "title_missing": 0,
            "description_missing": 0,
            "buckets": [0, 0, 0, 0],
        }
        for seller_uid in seller_uids
    }
    seen_items: set[str] = set()
    for row in items:
        seller_uid = str(row["seller_uid"])
        item_uid = str(row["item_uid"])
        if seller_uid not in accumulators or not item_uid or item_uid in seen_items:
            raise ShortcutPreflightError("Shortcut item/seller lineage drift")
        seen_items.add(item_uid)
        try:
            time_bucket = int(row["time_bucket"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ShortcutPreflightError(
                "Shortcut item lacks its private time bucket"
            ) from exc
        if not 0 <= time_bucket <= 3:
            raise ShortcutPreflightError("Shortcut item time bucket drift")
        if "description" in row:
            description = str(row["description"])
        elif "base_description" in row:
            description = str(row["base_description"])
        else:
            raise ShortcutPreflightError(
                "Shortcut item lacks description/base_description"
            )
        accumulator = accumulators[seller_uid]
        accumulator["count"] += 1
        accumulator["title_missing"] += int(str(row["title"]) == "")
        accumulator["description_missing"] += int(description == "")
        accumulator["buckets"][time_bucket] += 1
    output: dict[str, tuple[float, ...]] = {}
    for seller_uid, accumulator in accumulators.items():
        count = int(accumulator["count"])
        if not 2 <= count <= 8:
            raise ShortcutPreflightError("Shortcut seller item count drift")
        output[seller_uid] = (
            float(count),
            float(accumulator["title_missing"]) / count,
            float(accumulator["description_missing"]) / count,
            *(float(value) / count for value in accumulator["buckets"]),
        )
    return output


def _pair_feature_rows(
    *,
    world_uid: str,
    seller_uids: Sequence[str],
    items: Sequence[Mapping[str, Any]],
    controller_membership: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    import step28_v13_common as common

    ordered_sellers = sorted(
        (str(value) for value in seller_uids), key=lambda value: value.encode("utf-8")
    )
    if len(ordered_sellers) != 28 or len(set(ordered_sellers)) != 28:
        raise ShortcutPreflightError("Shortcut seller pool drift")
    seller_vectors = _seller_vectors(
        seller_uids=ordered_sellers,
        items=items,
    )
    seller_to_controller = {
        str(row["seller_uid"]): str(row["controller_uid"])
        for row in controller_membership
    }
    if set(seller_to_controller) != set(ordered_sellers):
        raise ShortcutPreflightError("Shortcut membership keyset drift")
    seller_rank = {
        seller_uid: index / 27.0
        for index, seller_uid in enumerate(ordered_sellers)
    }
    seller_hash = {
        seller_uid: _unit_hash("step28-v1.12-seller-hash-01", seller_uid)
        for seller_uid in ordered_sellers
    }
    pairs = list(itertools.combinations(ordered_sellers, 2))
    rows: list[dict[str, Any]] = []
    for pair_rank, (left, right) in enumerate(pairs):
        pair_uid = common.canonical_pair_uid(left, right)
        left_vector = seller_vectors[left]
        right_vector = seller_vectors[right]
        features: dict[str, float] = {}
        for index, name in enumerate(NULL_SELLER_FEATURES):
            features[f"absdiff__{name}"] = abs(
                left_vector[index] - right_vector[index]
            )
            features[f"sum__{name}"] = left_vector[index] + right_vector[index]
        features.update(
            {
                "world_hash_01": _unit_hash(
                    "step28-v1.12-world-hash-01", world_uid
                ),
                "left_hash_01": seller_hash[left],
                "right_hash_01": seller_hash[right],
                "pair_hash_01": _unit_hash(
                    "step28-v1.12-pair-hash-01", pair_uid
                ),
                "pair_hash_02": _unit_hash(
                    "step28-v1.12-pair-hash-02", pair_uid
                ),
                "left_rank_01": seller_rank[left],
                "right_rank_01": seller_rank[right],
                "pair_rank_01": pair_rank / 377.0,
                "absdiff_seller_hash_01": abs(
                    seller_hash[left] - seller_hash[right]
                ),
                "sum_seller_hash_01": seller_hash[left] + seller_hash[right],
            }
        )
        if set(features) != set(FEATURES) or not all(
            math.isfinite(value) for value in features.values()
        ):
            raise ShortcutPreflightError("Shortcut feature schema/value drift")
        rows.append(
            {
                "world_uid": world_uid,
                "canonical_pair_uid": pair_uid,
                "label": int(
                    seller_to_controller[left] == seller_to_controller[right]
                ),
                **features,
            }
        )
    if len(rows) != 378 or sum(int(row["label"]) for row in rows) != 20:
        raise ShortcutPreflightError("Shortcut full-pair label drift")
    return rows


def _fast_world_rows(
    *,
    execution_policy: dict[str, Any],
    template: dict[str, Any],
    style_profile: dict[str, Any],
    split: str,
    world_record: Mapping[str, Any],
    generator_capabilities: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Build exactly the label-free null fields without solving identities."""

    import step28_v13_nonidentity as nonidentity
    import step28_v13_structure as structure

    world_uid = str(world_record["world_uid"])
    structure_key = str(generator_capabilities["structure"])
    membership = structure.build_world_membership(
        execution_policy,
        mode="formal",
        world_uid=world_uid,
        structure_key_hex=structure_key,
    )
    controller_styles = nonidentity.world_controller_styles(
        policy=execution_policy,
        template=template,
        world_uid=world_uid,
        structure_key_hex=structure_key,
        controller_uids=membership["controller_uids"],
        mode="formal",
    )
    items: list[dict[str, Any]] = []
    for seller_uid in sorted(
        membership["seller_uids"], key=lambda value: value.encode("utf-8")
    ):
        controller_uid = membership["seller_to_controller"][seller_uid]
        effective_style = nonidentity.seller_effective_style(
            policy=execution_policy,
            template=template,
            mode="formal",
            seller_uid=seller_uid,
            controller_style=controller_styles[controller_uid],
        )
        items.extend(
            nonidentity.build_seller_items(
                policy=execution_policy,
                template=template,
                style_profile=style_profile,
                mode="formal",
                split=split,
                world_uid=world_uid,
                seller_uid=seller_uid,
                effective_style=effective_style,
            )
        )
    membership_rows = [
        {
            "world_uid": world_uid,
            "controller_uid": controller_uid,
            "seller_uid": seller_uid,
        }
        for controller_uid in sorted(
            membership["controller_members"], key=lambda value: value.encode("utf-8")
        )
        for seller_uid in membership["controller_members"][controller_uid]
    ]
    return _pair_feature_rows(
        world_uid=world_uid,
        seller_uids=membership["seller_uids"],
        items=items,
        controller_membership=membership_rows,
    )


def _full_bundle_feature_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    item_index = {
        str(row["item_uid"]): int(row["time_bucket"])
        for row in bundle["private"]["history_item_index"]
    }
    redacted_items = [
        {
            **dict(row),
            "time_bucket": item_index[str(row["item_uid"])],
        }
        for row in bundle["public"]["redacted_items"]
    ]
    return _pair_feature_rows(
        world_uid=str(bundle["world_uid"]),
        seller_uids=[
            str(row["seller_uid"]) for row in bundle["public"]["sellers"]
        ],
        items=redacted_items,
        controller_membership=bundle["private"]["controller_membership"],
    )


def validate_fast_path_parity() -> dict[str, Any]:
    validated = formal.load_and_validate_draft()
    draft = validated["draft"]
    master = bytes.fromhex(draft["randomness"]["design_only_master_hex"])
    capabilities = {
        split: formal.derive_capabilities(master, split=split)
        for split in formal.SPLITS
    }
    commitments = {
        split: formal.capability_commitments(capabilities[split])["generator"][
            "structure"
        ]
        for split in formal.SPLITS
    }
    hashes: list[dict[str, str]] = []
    for split in ("train", "development"):
        policy = formal.build_execution_policy(
            draft=draft,
            split=split,
            generator_capabilities=capabilities[split]["generator"],
            structure_commitments=commitments,
        )
        template, fixture, style_profile = formal.load_release_inputs(policy)
        record = formal.split_world_records(policy, split=split)[0]
        fast = _fast_world_rows(
            execution_policy=policy,
            template=template,
            style_profile=style_profile,
            split=split,
            world_record=record,
            generator_capabilities=capabilities[split]["generator"],
        )
        bundle = formal.materialize_world_bundle(
            execution_policy=policy,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
            split=split,
            world_record=record,
            generator_capabilities=capabilities[split]["generator"],
            historical_forbidden_hashes=validated["baseline"][
                "failed_identity_hashes"
            ],
            allocated_identity_hashes=set(),
            maximum_identity_counter=int(
                draft["identity_collision_resolution"]["maximum_counter"]
            ),
        )
        full = _full_bundle_feature_rows(bundle)
        if preceremony.canonical_json_bytes(fast) != preceremony.canonical_json_bytes(
            full
        ):
            raise ShortcutPreflightError(
                f"Fast/full null-nuisance feature parity failed: {split}"
            )
        hashes.append(
            {
                "split": split,
                "rows_sha256": preceremony.canonical_sha256(fast),
            }
        )
    return {
        "status": "PASS_FAST_FULL_NULL_NUISANCE_PARITY",
        "world_count": 2,
        "pair_count": 756,
        "split_hashes": hashes,
    }


def build_exact_scale_rows(
    *, split: str, world_count: int, progress_every: int = 25
) -> list[dict[str, Any]]:
    if split not in {"train", "development"} or world_count != 500:
        raise ShortcutPreflightError(
            "Exact numeric preflight requires 500 train/development worlds"
        )
    validated = formal.load_and_validate_draft()
    draft = validated["draft"]
    master = bytes.fromhex(draft["randomness"]["design_only_master_hex"])
    capabilities = {
        name: formal.derive_capabilities(master, split=name)
        for name in formal.SPLITS
    }
    commitments = {
        name: formal.capability_commitments(capabilities[name])["generator"][
            "structure"
        ]
        for name in formal.SPLITS
    }
    policy = formal.build_execution_policy(
        draft=draft,
        split=split,
        generator_capabilities=capabilities[split]["generator"],
        structure_commitments=commitments,
    )
    template, _fixture, style_profile = formal.load_release_inputs(policy)
    rows: list[dict[str, Any]] = []
    records = formal.split_world_records(policy, split=split)
    for index, record in enumerate(records):
        rows.extend(
            _fast_world_rows(
                execution_policy=policy,
                template=template,
                style_profile=style_profile,
                split=split,
                world_record=record,
                generator_capabilities=capabilities[split]["generator"],
            )
        )
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(f"SHORTCUT_ROWS_PROGRESS {split} {index + 1}/{world_count}")
    if (
        len(rows) != 189000
        or len({str(row["canonical_pair_uid"]) for row in rows}) != 189000
        or sum(int(row["label"]) for row in rows) != 10000
    ):
        raise ShortcutPreflightError("Exact-scale shortcut row closure failed")
    return rows


def _matrix(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x = np.asarray(
        [[float(row[name]) for name in FEATURES] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    worlds = [str(row["world_uid"]) for row in rows]
    if (
        x.shape != (len(rows), len(FEATURES))
        or y.shape != (len(rows),)
        or not np.all(np.isfinite(x))
        or set(np.unique(y).tolist()) != {0, 1}
    ):
        raise ShortcutPreflightError("Shortcut matrix validity failure")
    return x, y, worlds


def _fold_by_world(
    world_uids: Sequence[str], *, seed: int, fold_count: int
) -> dict[str, int]:
    worlds = sorted(set(world_uids), key=lambda value: value.encode("utf-8"))
    ranked = sorted(
        worlds,
        key=lambda world_uid: (
            hashlib.sha256(
                b"\x1f".join(
                    (str(seed).encode("ascii"), world_uid.encode("utf-8"))
                )
            ).digest(),
            world_uid.encode("utf-8"),
        ),
    )
    output = {world_uid: index % fold_count for index, world_uid in enumerate(ranked)}
    if set(Counter(output.values()).values()) != {len(worlds) // fold_count}:
        raise ShortcutPreflightError("Shortcut world fold balance drift")
    return output


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _objective_gradient_hessian(
    x: np.ndarray,
    y: np.ndarray,
    parameters: np.ndarray,
    *,
    l2: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    intercept = float(parameters[0])
    coefficients = parameters[1:]
    logits = intercept + x @ coefficients
    probability = _sigmoid(logits)
    residual = probability - y
    n = float(len(y))
    objective = float(
        np.mean(np.logaddexp(0.0, logits) - y * logits)
        + 0.5 * l2 * np.dot(coefficients, coefficients)
    )
    gradient = np.empty_like(parameters)
    gradient[0] = float(np.mean(residual))
    gradient[1:] = x.T @ residual / n + l2 * coefficients
    curvature = probability * (1.0 - probability)
    hessian = np.empty((len(parameters), len(parameters)), dtype=np.float64)
    hessian[0, 0] = float(np.mean(curvature))
    cross = x.T @ curvature / n
    hessian[0, 1:] = cross
    hessian[1:, 0] = cross
    hessian[1:, 1:] = x.T @ (curvature[:, None] * x) / n
    hessian[1:, 1:] += l2 * np.eye(x.shape[1], dtype=np.float64)
    if (
        not math.isfinite(objective)
        or not np.all(np.isfinite(gradient))
        or not np.all(np.isfinite(hessian))
    ):
        raise ShortcutPreflightError("Logistic objective/derivatives are nonfinite")
    return objective, gradient, hessian


def fit_exact_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    maximum_iterations: int,
    gradient_tolerance: float,
) -> LogisticArtifact:
    if (
        x.ndim != 2
        or y.shape != (x.shape[0],)
        or len(y) < 2
        or set(np.unique(y).tolist()) != {0, 1}
        or l2 <= 0.0
        or maximum_iterations < 2
        or gradient_tolerance <= 0.0
    ):
        raise ShortcutPreflightError("Exact logistic fit input drift")
    mean = np.mean(x, axis=0, dtype=np.float64)
    scale = np.std(x, axis=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)
    transformed = (x - mean) / scale
    parameters = np.zeros(x.shape[1] + 1, dtype=np.float64)
    prevalence = float(np.mean(y))
    parameters[0] = math.log(prevalence / (1.0 - prevalence))
    iteration_count = 0
    converged = False
    final_objective = math.inf
    final_gradient = math.inf
    while iteration_count < maximum_iterations:
        objective, gradient, hessian = _objective_gradient_hessian(
            transformed, y, parameters, l2=l2
        )
        gradient_norm = float(np.max(np.abs(gradient)))
        final_objective = objective
        final_gradient = gradient_norm
        if gradient_norm <= gradient_tolerance:
            converged = True
            break
        try:
            direction = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ShortcutPreflightError("Exact logistic Hessian is singular") from exc
        directional_derivative = float(np.dot(gradient, direction))
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            raise ShortcutPreflightError("Exact logistic Newton direction is invalid")
        step = 1.0
        accepted = False
        for _line_search in range(60):
            candidate = parameters + step * direction
            candidate_objective, _candidate_gradient, _candidate_hessian = (
                _objective_gradient_hessian(
                    transformed, y, candidate, l2=l2
                )
            )
            if candidate_objective <= objective + 1e-4 * step * directional_derivative:
                parameters = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise ShortcutPreflightError("Exact logistic line search failed")
        iteration_count += 1
    audit = {
        "solver_success": converged,
        "convergence_warning_count": 0,
        "iteration_count": iteration_count,
        "maximum_iterations": maximum_iterations,
        "normalized_gradient": final_gradient,
        "gradient_tolerance": gradient_tolerance,
        "objective_finite": math.isfinite(final_objective),
        "preceremony_exact_configuration": True,
    }
    preceremony.validate_optimizer_audit(audit)
    return LogisticArtifact(
        mean=mean,
        scale=scale,
        intercept=float(parameters[0]),
        coefficients=parameters[1:].copy(),
        audit={**audit, "objective": final_objective, "l2": l2},
    )


def score_exact_logistic(artifact: LogisticArtifact, x: np.ndarray) -> np.ndarray:
    transformed = (x - artifact.mean) / artifact.scale
    scores = _sigmoid(
        artifact.intercept + transformed @ artifact.coefficients
    )
    if scores.shape != (len(x),) or not np.all(np.isfinite(scores)):
        raise ShortcutPreflightError("Exact logistic score validity failure")
    return scores


def _fit_oof_and_development(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_worlds: Sequence[str],
    x_development: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    fold_count = int(config["fold_count"])
    fold_by_world = _fold_by_world(
        train_worlds,
        seed=int(config["fold_seed"]),
        fold_count=fold_count,
    )
    folds = np.asarray(
        [fold_by_world[world_uid] for world_uid in train_worlds],
        dtype=np.int8,
    )
    oof = {
        model: np.full(len(y_train), np.nan, dtype=np.float64)
        for model in ATTACK_MODELS
    }
    logistic_audits: list[dict[str, Any]] = []
    fold_audit: list[dict[str, Any]] = []
    for fold in range(fold_count):
        test = folds == fold
        train = ~test
        if set(np.unique(y_train[train]).tolist()) != {0, 1} or set(
            np.unique(y_train[test]).tolist()
        ) != {0, 1}:
            raise ShortcutPreflightError("Shortcut OOF fold is single-class")
        logistic = fit_exact_logistic(
            x_train[train],
            y_train[train],
            l2=float(config["logistic_l2"]),
            maximum_iterations=int(config["logistic_maximum_iterations"]),
            gradient_tolerance=float(config["logistic_gradient_tolerance"]),
        )
        oof["logistic_l2"][test] = score_exact_logistic(
            logistic, x_train[test]
        )
        tree = HistGradientBoostingClassifier(
            max_depth=int(config["tree_max_depth"]),
            max_iter=int(config["tree_max_iterations"]),
            learning_rate=0.03,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=int(config["fold_seed"]),
            class_weight=None,
        )
        tree.fit(x_train[train], y_train[train])
        oof["gradient_tree"][test] = tree.predict_proba(x_train[test])[:, 1]
        train_world_set = {
            train_worlds[index] for index in np.flatnonzero(train)
        }
        test_world_set = {
            train_worlds[index] for index in np.flatnonzero(test)
        }
        if train_world_set & test_world_set:
            raise ShortcutPreflightError("Shortcut OOF world leakage")
        logistic_audits.append(logistic.audit)
        fold_audit.append(
            {
                "fold": fold,
                "train_world_count": len(train_world_set),
                "test_world_count": len(test_world_set),
                "train_row_count": int(np.sum(train)),
                "test_row_count": int(np.sum(test)),
            }
        )
    if any(not np.all(np.isfinite(values)) for values in oof.values()):
        raise ShortcutPreflightError("Shortcut OOF score coverage failed")
    full_logistic = fit_exact_logistic(
        x_train,
        y_train,
        l2=float(config["logistic_l2"]),
        maximum_iterations=int(config["logistic_maximum_iterations"]),
        gradient_tolerance=float(config["logistic_gradient_tolerance"]),
    )
    full_tree = HistGradientBoostingClassifier(
        max_depth=int(config["tree_max_depth"]),
        max_iter=int(config["tree_max_iterations"]),
        learning_rate=0.03,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=int(config["fold_seed"]),
        class_weight=None,
    )
    full_tree.fit(x_train, y_train)
    development = {
        "logistic_l2": score_exact_logistic(full_logistic, x_development),
        "gradient_tree": full_tree.predict_proba(x_development)[:, 1].astype(
            np.float64
        ),
    }
    if any(not np.all(np.isfinite(values)) for values in development.values()):
        raise ShortcutPreflightError("Shortcut development score validity failed")
    return oof, development, {
        "fold_audit": fold_audit,
        "fold_logistic_optimizer_audits": logistic_audits,
        "full_train_logistic_optimizer_audit": full_logistic.audit,
    }


def _single_feature_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    maximum_auc = 0.5
    maximum_ap = float(np.mean(y))
    for index, name in enumerate(FEATURES):
        values = x[:, index]
        auc = float(roc_auc_score(y, values))
        symmetric = max(auc, 1.0 - auc)
        ap_forward = float(average_precision_score(y, values))
        ap_reverse = float(average_precision_score(y, -values))
        ap = max(ap_forward, ap_reverse)
        metrics[name] = {
            "roc_auc": auc,
            "symmetric_roc_auc": symmetric,
            "direction_free_average_precision": ap,
        }
        maximum_auc = max(maximum_auc, symmetric)
        maximum_ap = max(maximum_ap, ap)
    return {
        "features": metrics,
        "maximum_symmetric_roc_auc": maximum_auc,
        "maximum_direction_free_average_precision": maximum_ap,
    }


def _combined_metrics(
    scores: Mapping[str, np.ndarray], y: np.ndarray
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in ATTACK_MODELS:
        auc = float(roc_auc_score(y, scores[model]))
        output[model] = {
            "roc_auc": auc,
            "symmetric_roc_auc": max(auc, 1.0 - auc),
            "average_precision": float(
                average_precision_score(y, scores[model])
            ),
        }
    return output


def _draw_world_multiplicities(
    *, world_uids: Sequence[str], replicates: int, seed: int, split: str
) -> tuple[np.ndarray, np.ndarray, str]:
    ordered_worlds = sorted(set(world_uids), key=lambda value: value.encode("utf-8"))
    if len(ordered_worlds) != 500:
        raise ShortcutPreflightError("Shortcut bootstrap world count drift")
    ordinal = {world_uid: index for index, world_uid in enumerate(ordered_worlds)}
    row_world = np.asarray([ordinal[value] for value in world_uids], dtype=np.int16)
    split_seed = int.from_bytes(
        hashlib.sha256(
            b"\x1f".join((str(seed).encode("ascii"), split.encode("ascii")))
        ).digest()[:16],
        "big",
        signed=False,
    )
    generator = np.random.Generator(np.random.PCG64DXSM(split_seed))
    draws = generator.integers(
        0, 500, size=(replicates, 500), dtype=np.int16
    )
    multiplicities = np.zeros((replicates, 500), dtype=np.int16)
    for replicate in range(replicates):
        multiplicities[replicate] = np.bincount(
            draws[replicate], minlength=500
        ).astype(np.int16)
    draw_hash = hashlib.sha256(
        np.ascontiguousarray(draws.astype(">i2", copy=False)).tobytes()
    ).hexdigest()
    return multiplicities, row_world, draw_hash


def _bootstrap_rank_metrics(
    *,
    y: np.ndarray,
    scores: np.ndarray,
    multiplicities: np.ndarray,
    row_world: np.ndarray,
    replicate_chunk: int = 256,
    row_chunk: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact weighted ROC-AUC/AP for every world-bootstrap replicate."""

    if (
        y.shape != scores.shape
        or row_world.shape != y.shape
        or multiplicities.ndim != 2
        or multiplicities.shape[1] != 500
        or len(y) == 0
        or replicate_chunk < 1
        or row_chunk < 1
    ):
        raise ShortcutPreflightError("Bootstrap rank-metric input drift")
    replicate_count = multiplicities.shape[0]
    auc_values = np.empty(replicate_count, dtype=np.float64)
    ap_values = np.empty(replicate_count, dtype=np.float64)
    order_descending = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order_descending]
    group_boundaries = np.flatnonzero(
        np.r_[True, sorted_scores[1:] != sorted_scores[:-1], True]
    ).astype(np.int64, copy=False)
    if (
        len(group_boundaries) < 2
        or int(group_boundaries[0]) != 0
        or int(group_boundaries[-1]) != len(y)
    ):
        raise ShortcutPreflightError("Bootstrap score-group closure failed")

    positive_by_world = np.bincount(
        row_world, weights=y.astype(np.float64), minlength=500
    )
    negative_by_world = np.bincount(
        row_world, weights=1.0 - y.astype(np.float64), minlength=500
    )
    if (
        positive_by_world.shape != (500,)
        or negative_by_world.shape != (500,)
        or not np.all(positive_by_world == 20.0)
        or not np.all(negative_by_world == 358.0)
    ):
        raise ShortcutPreflightError("Bootstrap per-world class closure failed")

    for start_replicate in range(0, replicate_count, replicate_chunk):
        end_replicate = min(replicate_count, start_replicate + replicate_chunk)
        counts = multiplicities[start_replicate:end_replicate].astype(
            np.float64, copy=False
        )
        batch = end_replicate - start_replicate
        total_positive = counts @ positive_by_world
        total_negative = counts @ negative_by_world
        if np.any(total_positive <= 0.0) or np.any(total_negative <= 0.0):
            raise ShortcutPreflightError("Bootstrap produced a single class")

        auc_numerator = np.zeros(batch, dtype=np.float64)
        ap_numerator = np.zeros(batch, dtype=np.float64)
        cumulative_positive = np.zeros(batch, dtype=np.float64)
        cumulative_negative = np.zeros(batch, dtype=np.float64)
        cumulative_total = np.zeros(batch, dtype=np.float64)
        group_index = 0
        group_count = len(group_boundaries) - 1
        while group_index < group_count:
            row_start = int(group_boundaries[group_index])
            target = row_start + row_chunk
            group_stop = int(
                np.searchsorted(group_boundaries, target, side="right") - 1
            )
            group_stop = min(group_count, max(group_index + 1, group_stop))
            row_stop = int(group_boundaries[group_stop])
            indices = order_descending[row_start:row_stop]
            weights = counts[:, row_world[indices]]
            labels = y[indices].astype(np.float64, copy=False)
            local_starts = (
                group_boundaries[group_index:group_stop] - row_start
            ).astype(np.intp, copy=False)
            positive = np.add.reduceat(
                weights * labels[None, :], local_starts, axis=1
            )
            negative = np.add.reduceat(
                weights * (1.0 - labels)[None, :], local_starts, axis=1
            )
            total = positive + negative

            negative_before = cumulative_negative[:, None] + np.cumsum(
                negative, axis=1
            ) - negative
            auc_numerator += np.sum(
                positive
                * (
                    total_negative[:, None]
                    - negative_before
                    - 0.5 * negative
                ),
                axis=1,
            )

            cumulative_positive_block = cumulative_positive[:, None] + np.cumsum(
                positive, axis=1
            )
            cumulative_total_block = cumulative_total[:, None] + np.cumsum(
                total, axis=1
            )
            precision = np.divide(
                cumulative_positive_block,
                cumulative_total_block,
                out=np.zeros_like(cumulative_positive_block),
                where=cumulative_total_block > 0.0,
            )
            ap_numerator += np.sum(positive * precision, axis=1)
            cumulative_positive = cumulative_positive_block[:, -1]
            cumulative_negative = negative_before[:, -1] + negative[:, -1]
            cumulative_total = cumulative_total_block[:, -1]
            group_index = group_stop

        if not (
            np.allclose(cumulative_positive, total_positive, rtol=0.0, atol=1e-9)
            and np.allclose(
                cumulative_negative, total_negative, rtol=0.0, atol=1e-9
            )
        ):
            raise ShortcutPreflightError("Bootstrap rank traversal did not close")
        raw_auc = auc_numerator / (total_positive * total_negative)
        auc_values[start_replicate:end_replicate] = np.maximum(
            raw_auc, 1.0 - raw_auc
        )
        ap_values[start_replicate:end_replicate] = (
            ap_numerator / total_positive
        )
    if not np.all(np.isfinite(auc_values)) or not np.all(np.isfinite(ap_values)):
        raise ShortcutPreflightError("Bootstrap metric array is nonfinite")
    return auc_values, ap_values


def evaluate_exact_shortcut_rows(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    development_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one exact persisted 500+500-world train/development pair."""

    if len(train_rows) != 189000 or len(development_rows) != 189000:
        raise ShortcutPreflightError("Exact shortcut evaluation row-count drift")
    x_train, y_train, train_worlds = _matrix(train_rows)
    x_development, y_development, development_worlds = _matrix(
        development_rows
    )
    if (
        len(set(train_worlds)) != 500
        or len(set(development_worlds)) != 500
        or set(train_worlds) & set(development_worlds)
    ):
        raise ShortcutPreflightError("Exact shortcut world isolation drift")
    oof, development_scores, fit_audit = _fit_oof_and_development(
        x_train=x_train,
        y_train=y_train,
        train_worlds=train_worlds,
        x_development=x_development,
        config=config,
    )
    train_single = _single_feature_metrics(x_train, y_train)
    development_single = _single_feature_metrics(x_development, y_development)
    train_combined = _combined_metrics(oof, y_train)
    development_combined = _combined_metrics(
        development_scores, y_development
    )
    multiplicities, row_world, draw_hash = _draw_world_multiplicities(
        world_uids=development_worlds,
        replicates=int(config["bootstrap_replicates"]),
        seed=int(config["bootstrap_seed"]),
        split="development",
    )
    bootstrap_auc_by_model: dict[str, np.ndarray] = {}
    bootstrap_ap_by_model: dict[str, np.ndarray] = {}
    for model in ATTACK_MODELS:
        auc_values, ap_values = _bootstrap_rank_metrics(
            y=y_development,
            scores=development_scores[model],
            multiplicities=multiplicities,
            row_world=row_world,
        )
        bootstrap_auc_by_model[model] = auc_values
        bootstrap_ap_by_model[model] = ap_values
    max_auc = np.maximum.reduce(
        [bootstrap_auc_by_model[model] for model in ATTACK_MODELS]
    )
    baseline = 20.0 / 378.0
    max_ap_uplift = np.maximum.reduce(
        [bootstrap_ap_by_model[model] - baseline for model in ATTACK_MODELS]
    )
    upper_auc = float(np.quantile(max_auc, 0.95, method="higher"))
    upper_ap_uplift = float(
        np.quantile(max_ap_uplift, 0.95, method="higher")
    )
    point_single = max(
        float(train_single["maximum_symmetric_roc_auc"]),
        float(development_single["maximum_symmetric_roc_auc"]),
    )
    point_combined_auc = max(
        metric["symmetric_roc_auc"]
        for block in (train_combined, development_combined)
        for metric in block.values()
    )
    point_combined_ap_uplift = max(
        metric["average_precision"] - baseline
        for block in (train_combined, development_combined)
        for metric in block.values()
    )
    gates = {
        "single_feature_auc": point_single
        <= float(config["single_feature_maximum_symmetric_auc"]),
        "combined_auc": point_combined_auc
        <= float(config["combined_maximum_symmetric_auc"]),
        "combined_bootstrap_auc": upper_auc
        <= float(config["combined_bootstrap_95_upper_symmetric_auc"]),
        "combined_ap_uplift": point_combined_ap_uplift
        <= float(config["combined_maximum_ap_uplift"]),
        "combined_bootstrap_ap_uplift": upper_ap_uplift
        <= float(config["combined_bootstrap_95_upper_ap_uplift"]),
        "all_logistic_optimizer_audits": True,
    }
    return {
        "train_row_count": len(train_rows),
        "development_row_count": len(development_rows),
        "train_world_count": len(set(train_worlds)),
        "development_world_count": len(set(development_worlds)),
        "positive_count_per_split": 10000,
        "negative_count_per_split": 179000,
        "train_single_feature_metrics": train_single,
        "development_single_feature_metrics": development_single,
        "train_combined_metrics": train_combined,
        "development_combined_metrics": development_combined,
        "optimizer_audit": fit_audit,
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "bootstrap_draw_sha256": draw_hash,
        "bootstrap_95_upper_max_combined_symmetric_auc": upper_auc,
        "bootstrap_95_upper_max_combined_ap_uplift": upper_ap_uplift,
        "point_max_single_symmetric_auc": point_single,
        "point_max_combined_symmetric_auc": point_combined_auc,
        "point_max_combined_ap_uplift": point_combined_ap_uplift,
        "random_ap_baseline": baseline,
        "gates": gates,
    }


def run_exact_preflight(*, progress_every: int = 25) -> dict[str, Any]:
    start = time.perf_counter()
    parity = validate_fast_path_parity()
    train_rows = build_exact_scale_rows(
        split="train", world_count=500, progress_every=progress_every
    )
    development_rows = build_exact_scale_rows(
        split="development", world_count=500, progress_every=progress_every
    )
    config = formal.load_and_validate_draft()["draft"]["shortcut_preflight"]
    evaluation = evaluate_exact_shortcut_rows(
        train_rows=train_rows,
        development_rows=development_rows,
        config=config,
    )
    gates = evaluation["gates"]
    report = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-exact-shortcut-v1",
            "status": (
                "PASS_DESIGN_ONLY_EXACT_SHORTCUT_PREFLIGHT"
                if all(gates.values())
                else "FAIL_DESIGN_ONLY_EXACT_SHORTCUT_PREFLIGHT"
            ),
            "formal_authorization_used": False,
            "formal_seed_or_key_access": False,
            "formal_rows_produced": 0,
            "scientific_metrics_produced": False,
            "producer_path": (
                "scripts/step28_v13_v1_12_exact_shortcut_preflight.py"
            ),
            "producer_sha256": preceremony.sha256_file(Path(__file__)),
            "formal_common_sha256": preceremony.sha256_file(
                ROOT / "scripts" / "step28_v13_v1_12_formal_common.py"
            ),
            "formal_build_draft_sha256": preceremony.sha256_file(
                formal.DEFAULT_DRAFT_PATH
            ),
            "runtime_versions": formal.runtime_versions(),
            "feature_order": list(FEATURES),
            "feature_count": len(FEATURES),
            "fast_full_parity": parity,
            **evaluation,
            "elapsed_seconds": time.perf_counter() - start,
        }
    )
    if not all(gates.values()):
        raise ShortcutPreflightError(
            json.dumps(
                {
                    "status": report["status"],
                    "gates": gates,
                    "point_single": evaluation[
                        "point_max_single_symmetric_auc"
                    ],
                    "point_combined_auc": evaluation[
                        "point_max_combined_symmetric_auc"
                    ],
                    "point_combined_ap_uplift": evaluation[
                        "point_max_combined_ap_uplift"
                    ],
                    "upper_auc": evaluation[
                        "bootstrap_95_upper_max_combined_symmetric_auc"
                    ],
                    "upper_ap_uplift": evaluation[
                        "bootstrap_95_upper_max_combined_ap_uplift"
                    ],
                },
                sort_keys=True,
            )
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-fast-path-parity", action="store_true")
    parser.add_argument("--run-exact-preflight", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_fast_path_parity:
        report = validate_fast_path_parity()
        print(report["status"], report["world_count"], report["pair_count"])
        return
    if args.run_exact_preflight:
        if args.output is None:
            raise ShortcutPreflightError(
                "--run-exact-preflight requires a fresh --output receipt path"
            )
        report = run_exact_preflight(progress_every=args.progress_every)
        payload = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        preceremony.write_bytes_no_replace_long_path(args.output, payload)
        print(
            report["status"],
            report["train_row_count"],
            report["development_row_count"],
            preceremony.sha256_file(args.output),
        )
        return
    raise ShortcutPreflightError(
        "Choose --validate-fast-path-parity or --run-exact-preflight"
    )


if __name__ == "__main__":
    main()
