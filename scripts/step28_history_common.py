#!/usr/bin/env python3
"""Shared parser-observable history features for Step28/v4.

The same functions are used for synthetic worlds and the untouched real signal
table.  Latent controller IDs, recipes, raw identifier values, and labels never
enter ``history_features``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import step28_common as base


POLICY_PATH = base.ROOT / "schema" / "step28_transferable_identity_history_policy.json"
SUPPORTED_IDENTITY_TYPES = (
    "telegram",
    "email",
    "bat",
    "qq",
    "wechat",
    "phone",
    "crypto_wallet",
    "external_url",
)


def _deep_merge(parent: dict, child: dict) -> dict:
    merged = dict(parent)
    for key, value in child.items():
        if key == "_extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_policy(path: str | Path) -> dict:
    payload = base.load_json(path)
    parent_path = payload.get("_extends")
    if not parent_path:
        return payload
    parent = load_policy(parent_path)
    return _deep_merge(parent, payload)


def flag(row: dict, name: str) -> bool:
    return str(row.get(name, "")).strip().lower() in {"1", "true", "yes"}


def is_direct(row: dict) -> bool:
    return (
        flag(row, "direct_identity_eligible")
        and flag(row, "seller_facing_context")
        and not flag(row, "product_data_risk_context")
        and not flag(row, "support_only")
    )


def is_risky(row: dict) -> bool:
    return flag(row, "product_data_risk_context")


def is_support(row: dict) -> bool:
    return flag(row, "support_only")


def item_key(row: dict) -> str:
    return f"{row.get('source_dataset', '')}:{row.get('source_row_number', '')}"


def token_key(row: dict) -> tuple[str, str] | None:
    contact_type = str(row.get("contact_type", "")).strip().lower()
    value = str(row.get("normalized_value", "")).strip().lower()
    if not contact_type or not value:
        return None
    return contact_type, value


def build_signal_index(rows: list[dict]) -> tuple[dict, Counter]:
    by_seller: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sellers_by_token: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        seller = str(row.get("seller_uid", "")).strip()
        token = token_key(row)
        if not seller or token is None:
            continue
        by_seller[seller][token].append(row)
        sellers_by_token[token].add(seller)
    token_df = Counter({token: len(sellers) for token, sellers in sellers_by_token.items()})
    return by_seller, token_df


def _pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("Step28 pair endpoints must differ")
    return (left, right) if left < right else (right, left)


def _token_edge_record(
    left_occurrences: list[dict],
    right_occurrences: list[dict],
    seller_frequency: int,
    direct_frequency_maximum: int,
) -> dict:
    all_occurrences = [*left_occurrences, *right_occurrences]
    left_direct_items = {item_key(row) for row in left_occurrences if is_direct(row)}
    right_direct_items = {item_key(row) for row in right_occurrences if is_direct(row)}
    risky = any(is_risky(row) for row in all_occurrences)
    support = any(is_support(row) for row in all_occurrences)
    direct_both = bool(left_direct_items and right_direct_items)
    verified = (
        direct_both
        and not risky
        and not support
        and seller_frequency <= direct_frequency_maximum
    )
    return {
        "direct_both": direct_both,
        "verified": verified,
        "risky": risky,
        "support": support,
        "high_frequency": direct_both and seller_frequency > direct_frequency_maximum,
        "left_direct_item_count": len(left_direct_items),
        "right_direct_item_count": len(right_direct_items),
        "repeated_both": len(left_direct_items) >= 2 and len(right_direct_items) >= 2,
    }


def build_identity_graph(by_seller: dict, token_df: Counter, policy: dict) -> dict:
    direct_maximum = int(policy["generation"]["direct_token_seller_frequency_maximum"])
    weak_maximum = int(policy["generation"]["weak_graph_token_seller_frequency_maximum"])
    sellers_by_token: dict[tuple[str, str], list[str]] = defaultdict(list)
    for seller, token_rows in by_seller.items():
        for token in token_rows:
            sellers_by_token[token].append(seller)

    any_adjacency: dict[str, set[str]] = defaultdict(set)
    any_edge_tokens: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    strong_adjacency: dict[str, set[str]] = defaultdict(set)
    strong_edge_tokens: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    strong_edge_repeated: dict[tuple[str, str], bool] = defaultdict(bool)
    strong_edge_token_count: Counter = Counter()

    for token, sellers in sellers_by_token.items():
        unique = sorted(set(sellers))
        if not 2 <= len(unique) <= weak_maximum:
            continue
        for left_index in range(len(unique)):
            for right_index in range(left_index + 1, len(unique)):
                left, right = unique[left_index], unique[right_index]
                edge = (left, right)
                any_adjacency[left].add(right)
                any_adjacency[right].add(left)
                any_edge_tokens[edge].add(token)
                record = _token_edge_record(
                    by_seller[left][token],
                    by_seller[right][token],
                    token_df[token],
                    direct_maximum,
                )
                if record["verified"]:
                    strong_adjacency[left].add(right)
                    strong_adjacency[right].add(left)
                    strong_edge_tokens[edge].add(token)
                    strong_edge_repeated[edge] = bool(
                        strong_edge_repeated[edge] or record["repeated_both"]
                    )
                    strong_edge_token_count[edge] += 1

    return {
        "any_adjacency": any_adjacency,
        "any_edge_tokens": any_edge_tokens,
        "strong_adjacency": strong_adjacency,
        "strong_edge_tokens": strong_edge_tokens,
        "strong_edge_repeated": strong_edge_repeated,
        "strong_edge_token_count": strong_edge_token_count,
    }


def history_feature_details(
    left: str,
    right: str,
    by_seller: dict,
    token_df: Counter,
    graph: dict,
    policy: dict,
) -> tuple[dict[str, float], dict]:
    direct_maximum = int(policy["generation"]["direct_token_seller_frequency_maximum"])
    left_tokens = by_seller.get(left, {})
    right_tokens = by_seller.get(right, {})
    shared = sorted(set(left_tokens) & set(right_tokens))
    counts = Counter()
    bilateral_direct_items = 0
    shared_types: set[str] = set()
    shared_type_counts: Counter = Counter()
    verified_tokens: set[tuple[str, str]] = set()
    shared_token_hashes: list[str] = []

    for token in shared:
        shared_types.add(token[0])
        shared_type_counts[token[0]] += 1
        shared_token_hashes.append(hashlib.sha256(f"{token[0]}:{token[1]}".encode()).hexdigest()[:16])
        record = _token_edge_record(
            left_tokens[token], right_tokens[token], token_df[token], direct_maximum
        )
        if record["direct_both"] and (record["risky"] or record["support"]):
            counts["mixed"] += 1
        elif record["verified"]:
            counts["verified"] += 1
            verified_tokens.add(token)
            bilateral_direct_items += min(
                record["left_direct_item_count"], record["right_direct_item_count"]
            )
            counts["repeated"] += int(record["repeated_both"])
        elif record["risky"]:
            counts["risky"] += 1
        elif record["support"]:
            counts["support"] += 1
        elif record["high_frequency"]:
            counts["high_frequency"] += 1
        else:
            counts["ambiguous"] += 1

    strong_adjacency = graph["strong_adjacency"]
    any_adjacency = graph["any_adjacency"]
    strong_common = strong_adjacency.get(left, set()) & strong_adjacency.get(right, set())
    any_common = any_adjacency.get(left, set()) & any_adjacency.get(right, set())
    rotation_paths = 0
    corroborated_paths = 0
    same_token_paths = 0
    maximum_middle_degree = 0
    rotation_channel_pairs: list[str] = []
    rotation_edge_type_counts: Counter = Counter()
    for middle in strong_common:
        left_edge = _pair(left, middle)
        right_edge = _pair(middle, right)
        left_edge_tokens = graph["strong_edge_tokens"].get(left_edge, set())
        right_edge_tokens = graph["strong_edge_tokens"].get(right_edge, set())
        maximum_middle_degree = max(
            maximum_middle_degree, len(strong_adjacency.get(middle, set()))
        )
        if left_edge_tokens.isdisjoint(right_edge_tokens):
            rotation_paths += 1
            rotation_edge_type_counts.update(token[0] for token in left_edge_tokens)
            rotation_edge_type_counts.update(token[0] for token in right_edge_tokens)
            rotation_channel_pairs.extend(
                f"{left_token[0]}->{right_token[0]}"
                for left_token in left_edge_tokens
                for right_token in right_edge_tokens
            )
            left_corroborated = (
                graph["strong_edge_repeated"].get(left_edge, False)
                or graph["strong_edge_token_count"].get(left_edge, 0) >= 2
            )
            right_corroborated = (
                graph["strong_edge_repeated"].get(right_edge, False)
                or graph["strong_edge_token_count"].get(right_edge, 0) >= 2
            )
            corroborated_paths += int(left_corroborated and right_corroborated)
        else:
            same_token_paths += 1
    weak_paths = len(any_common - strong_common)

    log_verified = math.log1p(counts["verified"])
    log_repeated = math.log1p(counts["repeated"])
    log_high_frequency = math.log1p(counts["high_frequency"])
    log_risky = math.log1p(counts["risky"])
    log_support = math.log1p(counts["support"])
    log_rotation = math.log1p(rotation_paths)
    log_corroborated_rotation = math.log1p(corroborated_paths)
    log_uncorroborated_rotation = math.log1p(max(rotation_paths - corroborated_paths, 0))
    features = {
        "verified_direct_token_count_log1p": log_verified,
        "repeated_verified_token_count_log1p": log_repeated,
        "bilateral_direct_item_count_log1p": math.log1p(bilateral_direct_items),
        "shared_identifier_type_count_log1p": math.log1p(len(shared_types)),
        "mixed_context_token_count_log1p": math.log1p(counts["mixed"]),
        "risky_token_count_log1p": log_risky,
        "support_token_count_log1p": log_support,
        "high_frequency_token_count_log1p": log_high_frequency,
        "risky_without_verified": log_risky * float(counts["verified"] == 0),
        "support_without_verified": log_support * float(counts["verified"] == 0),
        "verified_x_risky": log_verified * log_risky,
        "verified_x_support": log_verified * log_support,
        "strong_rotation_path_count_log1p": log_rotation,
        "corroborated_rotation_path_count_log1p": log_corroborated_rotation,
        "uncorroborated_rotation_path_count_log1p": log_uncorroborated_rotation,
        "same_token_path_count_log1p": math.log1p(same_token_paths),
        "weak_path_count_log1p": math.log1p(weak_paths),
        "maximum_strong_middle_degree_log1p": math.log1p(maximum_middle_degree),
        "verified_x_repeat": log_verified * log_repeated,
        "rotation_x_corroboration": log_rotation * log_corroborated_rotation,
        "verified_x_high_frequency": log_verified * log_high_frequency,
        **{
            f"shared_{identity_type}_token_count_log1p": math.log1p(
                shared_type_counts[identity_type]
            )
            for identity_type in SUPPORTED_IDENTITY_TYPES
        },
        **{
            f"rotation_{identity_type}_edge_token_count_log1p": math.log1p(
                rotation_edge_type_counts[identity_type]
            )
            for identity_type in SUPPORTED_IDENTITY_TYPES
        },
    }
    details = {
        "shared_token_count": len(shared),
        "shared_token_hashes": sorted(shared_token_hashes),
        "shared_identifier_types": sorted(shared_types),
        "verified_token_hashes": sorted(
            hashlib.sha256(f"{token[0]}:{token[1]}".encode()).hexdigest()[:16]
            for token in verified_tokens
        ),
        "strong_common_middle_count": len(strong_common),
        "weak_common_middle_count": weak_paths,
        "rotation_channel_pairs": sorted(rotation_channel_pairs),
        "rotation_edge_identifier_types": dict(sorted(rotation_edge_type_counts.items())),
    }
    return features, details


def history_features(
    left: str,
    right: str,
    by_seller: dict,
    token_df: Counter,
    graph: dict,
    policy: dict,
) -> dict[str, float]:
    return history_feature_details(left, right, by_seller, token_df, graph, policy)[0]


def feature_vector(features: dict[str, float], policy: dict) -> np.ndarray:
    return np.asarray([features[name] for name in policy["model"]["feature_names"]], dtype=float)


def observable_state_values(values: np.ndarray | list[float]) -> tuple[float, ...]:
    """Canonicalize parser-observable features without labels or identifiers."""
    return tuple(round(float(value), 12) for value in np.asarray(values, dtype=float))


def observable_state_hash(values: np.ndarray | list[float]) -> str:
    payload = json.dumps(
        observable_state_values(values), separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def expansion_pairs(graph: dict, eligible_sellers: set[str]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    direct_pairs = {
        edge
        for edge in graph["any_edge_tokens"]
        if edge[0] in eligible_sellers and edge[1] in eligible_sellers
    }
    rotation_pairs: set[tuple[str, str]] = set()
    for middle, neighbors in graph["strong_adjacency"].items():
        eligible = sorted(set(neighbors) & eligible_sellers)
        for index in range(len(eligible)):
            for other in range(index + 1, len(eligible)):
                rotation_pairs.add(_pair(eligible[index], eligible[other]))
    return direct_pairs, rotation_pairs


def source_probability_from_cosine(cosine: float, policy: dict) -> float:
    return float(base.sigmoid(base.source_logit_from_cosine(float(cosine), policy)))


def predict_with_artifact(source_probability: np.ndarray, matrix: np.ndarray, artifact: dict) -> np.ndarray:
    source = np.asarray(source_probability, dtype=float)
    values = np.asarray(matrix, dtype=float)
    scales = np.asarray(artifact["feature_scales"], dtype=float)
    coefficients = np.asarray(artifact["coefficients"], dtype=float)
    correction = (values / scales) @ coefficients
    return np.asarray(base.sigmoid(base.logit(source) + correction), dtype=float)


def identity_correction(matrix: np.ndarray, artifact: dict) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return (values / np.asarray(artifact["feature_scales"], dtype=float)) @ np.asarray(
        artifact["coefficients"], dtype=float
    )


def binary_logloss(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(scores, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(labels, dtype=int)
    predicted = np.asarray(scores, dtype=float) >= float(threshold)
    tp = int(np.sum((y == 1) & predicted))
    fp = int(np.sum((y == 0) & predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-15),
    }


def weighted_confusion(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    sample_weight: np.ndarray,
) -> dict:
    y = np.asarray(labels, dtype=int)
    predicted = np.asarray(scores, dtype=float) >= float(threshold)
    weight = np.asarray(sample_weight, dtype=float)
    if y.shape != predicted.shape or y.shape != weight.shape:
        raise ValueError("weighted confusion inputs must have identical shapes")
    if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("weighted confusion requires finite non-negative weights")
    tp = float(np.sum(weight[(y == 1) & predicted]))
    fp = float(np.sum(weight[(y == 0) & predicted]))
    tn = float(np.sum(weight[(y == 0) & ~predicted]))
    fn = float(np.sum(weight[(y == 1) & ~predicted]))
    recall = tp / max(tp + fn, 1e-15)
    specificity = tn / max(tn + fp, 1e-15)
    precision = tp / max(tp + fp, 1e-15)
    return {
        "weighted_tp": tp,
        "weighted_fp": fp,
        "weighted_tn": tn,
        "weighted_fn": fn,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-15),
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    candidates = np.unique(np.asarray(scores, dtype=float))
    candidates = np.concatenate(([0.0], candidates, [1.0]))
    best_threshold = 0.5
    best = None
    for threshold in candidates:
        current = confusion(labels, scores, float(threshold))
        key = (current["balanced_accuracy"], current["f1"], -abs(float(threshold) - 0.5))
        if best is None or key > best[0]:
            best = (key, current)
            best_threshold = float(threshold)
    assert best is not None
    return best_threshold, best[1]


def choose_threshold_weighted(
    labels: np.ndarray, scores: np.ndarray, sample_weight: np.ndarray
) -> tuple[float, dict]:
    candidates = np.unique(np.asarray(scores, dtype=float))
    candidates = np.concatenate(([0.0], candidates, [1.0]))
    best_threshold = 0.5
    best = None
    for threshold in candidates:
        current = weighted_confusion(labels, scores, float(threshold), sample_weight)
        key = (
            current["balanced_accuracy"],
            current["f1"],
            -abs(float(threshold) - 0.5),
        )
        if best is None or key > best[0]:
            best = (key, current)
            best_threshold = float(threshold)
    assert best is not None
    return best_threshold, best[1]


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    return {
        "row_count": int(len(labels)),
        "positive_count": int(np.sum(labels)),
        "negative_count": int(len(labels) - np.sum(labels)),
        "roc_auc": base.roc_auc(labels, scores),
        "average_precision": base.average_precision(labels, scores),
        "logloss": binary_logloss(labels, scores),
        "threshold": float(threshold),
        **confusion(labels, scores, threshold),
    }


def weighted_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    sample_weight: np.ndarray,
) -> dict:
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(scores, dtype=float), 1e-12, 1.0 - 1e-12)
    weight = np.asarray(sample_weight, dtype=float)
    total_weight = float(np.sum(weight))
    if total_weight <= 0.0:
        raise ValueError("weighted metrics require positive total weight")
    weighted_logloss = float(
        np.sum(weight * (-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
        / total_weight
    )
    return {
        "raw_row_count": int(len(y)),
        "state_equal_weight_total": total_weight,
        "positive_weight": float(np.sum(weight[y == 1.0])),
        "negative_weight": float(np.sum(weight[y == 0.0])),
        "roc_auc": base.weighted_roc_auc(y, p, weight),
        "average_precision": base.weighted_average_precision(y, p, weight),
        "logloss": weighted_logloss,
        "threshold": float(threshold),
        **weighted_confusion(y, p, threshold, weight),
    }


def positive_review_eligible(
    *,
    identity_correction: float,
    out_of_support: bool,
    support: dict,
    policy: dict,
) -> bool:
    """Single production contract shared by synthetic safety audit and scoring."""
    scoring = policy["real_scoring"]
    if (
        scoring.get("review_queue_requires_positive_identity_correction", False)
        and float(identity_correction) <= 0.0
    ):
        return False
    if scoring.get("review_queue_requires_in_support", False) and out_of_support:
        return False
    if scoring.get(
        "review_queue_requires_positive_only_observable_state_support", False
    ):
        if support.get("status") != "positive_only":
            return False
        minimum = int(
            scoring.get("minimum_positive_observable_state_support_count", 1)
        )
        if int(support.get("positive_count", 0)) < minimum:
            return False
    if scoring.get(
        "review_queue_requires_support_in_both_train_and_development", False
    ) and set(support.get("splits", [])) != {
        "synthetic_train",
        "synthetic_development",
    }:
        return False
    return True


def canonical_pair_uid(left: str, right: str) -> str:
    ordered = _pair(left, right)
    return f"{ordered[0]}||{ordered[1]}"


def relative(path: Path) -> str:
    return str(path.relative_to(base.ROOT)).replace("\\", "/")
