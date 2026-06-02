from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "step7_transfer_safe_pair_feature_schema.json"
POLICY_PATH = ROOT / "schema" / "step7_training_policy.json"
FROZEN_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step5_en_frozen_silver_labels.csv",
    "zh_target_strict": ROOT / "reports" / "step5_zh_target_strict_frozen_silver_labels.csv",
}
PAIR_FEATURE_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step7_pair_features.en_content_train_pool.csv",
    "zh_target_strict": ROOT / "reports" / "step7_pair_features.zh_target_strict.csv",
}
EPS = 1e-9


def require_lightgbm():
    try:
        import lightgbm as lgb  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime
        raise SystemExit(
            "lightgbm is required for Step 7 fusion training. Install it, then rerun this script."
        ) from exc
    return lgb


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def label_to_int(label: str) -> int:
    return 1 if label == "positive" else 0


def to_float(value) -> float:
    if value in {"", None}:
        return math.nan
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Step-7 LightGBM fusion experiments from semantic-enriched pair features.")
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help="Experiment name from step7_training_policy.json. Repeat to run multiple experiments. Defaults to policy default_experiments.",
    )
    return parser.parse_args()


def join_frozen_with_features(frozen_rows: list[dict], feature_rows: list[dict]) -> list[dict]:
    feature_index = {row["pair_uid"]: row for row in feature_rows}
    joined = []
    missing = []
    for frozen in frozen_rows:
        feature_row = feature_index.get(frozen["pair_uid"])
        if feature_row is None:
            missing.append(frozen["pair_uid"])
            continue
        row = dict(feature_row)
        for key in (
            "balanced_review_rank",
            "review_status",
            "review_label",
            "reviewer_id",
            "review_notes",
            "soft_same_alias_continuity_bool",
            "usable_for_supervision",
            "usable_for_core_transfer",
            "split_name",
            "split_component_id",
            "split_component_size",
            "source_seller_raw_left",
            "source_seller_raw_right",
        ):
            row[key] = frozen[key]
        joined.append(row)
    if missing:
        raise ValueError(f"Missing semantic pair features for {len(missing)} frozen rows")
    return joined


def select_rows(rows: list[dict], split_name: str, require_core_transfer: bool) -> list[dict]:
    selected = []
    for row in rows:
        if row["review_label"] not in {"positive", "negative"}:
            continue
        if row["split_name"] != split_name:
            continue
        if row["usable_for_supervision"] != "1":
            continue
        if require_core_transfer and row["usable_for_core_transfer"] != "1":
            continue
        selected.append(row)
    return selected


def rows_to_matrix(rows: list[dict], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((len(rows), len(feature_names)), dtype=float)
    y = np.empty(len(rows), dtype=float)
    for i, row in enumerate(rows):
        y[i] = label_to_int(row["review_label"])
        for j, feature_name in enumerate(feature_names):
            x[i, j] = to_float(row.get(feature_name))
    return x, y


def balanced_sample_weights(y_true: np.ndarray) -> np.ndarray:
    positives = max(int(y_true.sum()), 1)
    negatives = max(int((1.0 - y_true).sum()), 1)
    total = len(y_true)
    pos_weight = total / (2.0 * positives)
    neg_weight = total / (2.0 * negatives)
    return np.where(y_true == 1.0, pos_weight, neg_weight)


def semantic_feature_names_for_experiment(policy: dict, experiment: dict, feature_names: list[str]) -> list[str]:
    semantic_names = policy["semantic_feature_sets"].get(experiment["semantic_feature_set"], [])
    return [name for name in semantic_names if name in feature_names]


def apply_semantic_activation_augmentation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    base_weights: np.ndarray,
    feature_names: list[str],
    policy: dict,
    experiment: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cfg = policy.get("semantic_activation_augmentation", {})
    enabled = bool(experiment.get("semantic_activation_augmentation_enabled", cfg.get("enabled", False)))
    semantic_feature_names = semantic_feature_names_for_experiment(policy, experiment, feature_names)
    diagnostics = {
        "enabled": False,
        "semantic_feature_names": semantic_feature_names,
        "semantic_feature_count": len(semantic_feature_names),
        "masked_non_semantic_feature_count": 0,
        "augmentation_weight": 0.0,
        "augmented_row_count": 0,
        "mask_value": str(cfg.get("mask_value", "nan")),
    }
    if not enabled or not semantic_feature_names:
        return x_train, y_train, base_weights, diagnostics

    non_semantic_indices = [idx for idx, name in enumerate(feature_names) if name not in set(semantic_feature_names)]
    augmentation_weight = float(cfg.get("augmentation_weight", 0.0))
    if augmentation_weight <= 0.0 or not non_semantic_indices:
        return x_train, y_train, base_weights, diagnostics

    mask_value = str(cfg.get("mask_value", "nan")).strip().lower()
    x_augmented = x_train.copy()
    if mask_value == "nan":
        x_augmented[:, non_semantic_indices] = np.nan
    elif mask_value == "zero":
        x_augmented[:, non_semantic_indices] = 0.0
    else:
        raise ValueError(f"Unsupported semantic_activation_augmentation.mask_value: {mask_value}")

    diagnostics.update(
        {
            "enabled": True,
            "masked_non_semantic_feature_count": len(non_semantic_indices),
            "augmentation_weight": round(float(augmentation_weight), 6),
            "augmented_row_count": int(len(x_augmented)),
            "mask_value": mask_value,
        }
    )
    return (
        np.concatenate([x_train, x_augmented], axis=0),
        np.concatenate([y_train, y_train], axis=0),
        np.concatenate([base_weights, base_weights * augmentation_weight], axis=0),
        diagnostics,
    )


def summarize_feature_family_gain(feature_importance: list[dict], semantic_feature_names: list[str]) -> dict:
    semantic_feature_set = set(semantic_feature_names)
    semantic_gain = 0.0
    semantic_split = 0
    non_semantic_gain = 0.0
    non_semantic_split = 0
    semantic_features_used = 0
    non_semantic_features_used = 0
    for item in feature_importance:
        gain = float(item["gain"])
        split = int(item["split"])
        is_semantic = item["feature_name"] in semantic_feature_set
        if is_semantic:
            semantic_gain += gain
            semantic_split += split
            if gain > 0.0 or split > 0:
                semantic_features_used += 1
        else:
            non_semantic_gain += gain
            non_semantic_split += split
            if gain > 0.0 or split > 0:
                non_semantic_features_used += 1
    total_gain = semantic_gain + non_semantic_gain
    total_split = semantic_split + non_semantic_split
    return {
        "semantic_feature_names": semantic_feature_names,
        "semantic_gain": round(float(semantic_gain), 6),
        "non_semantic_gain": round(float(non_semantic_gain), 6),
        "semantic_gain_share": round(float(semantic_gain / total_gain), 6) if total_gain > 0 else 0.0,
        "semantic_split_count": int(semantic_split),
        "non_semantic_split_count": int(non_semantic_split),
        "semantic_split_share": round(float(semantic_split / total_split), 6) if total_split > 0 else 0.0,
        "semantic_features_used": int(semantic_features_used),
        "non_semantic_features_used": int(non_semantic_features_used),
    }


def average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    pos = int(y_true.sum())
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = average_rank(y_score)
    pos_rank_sum = float(ranks[y_true == 1.0].sum())
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = int(y_true.sum())
    if positives == 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1.0)
    fp = np.cumsum(y_sorted == 0.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    return float(np.sum(precision[y_sorted == 1.0]) / positives)


def binary_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def threshold_grid(probabilities: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(probabilities, 6))
    if len(unique) >= 2:
        midpoints = (unique[:-1] + unique[1:]) / 2.0
    else:
        midpoints = np.array([], dtype=float)
    candidates = np.concatenate(([0.05], unique, midpoints, [0.95]))
    return np.clip(np.unique(candidates), 0.0, 1.0)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((y_true == 1.0) & (y_pred == 1.0)))
    tn = int(np.sum((y_true == 0.0) & (y_pred == 0.0)))
    fp = int(np.sum((y_true == 0.0) & (y_pred == 1.0)))
    fn = int(np.sum((y_true == 1.0) & (y_pred == 0.0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(float)
    counts = confusion_counts(y_true, y_pred)
    precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
    recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
    specificity = counts["tn"] / max(counts["tn"] + counts["fp"], 1)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    accuracy = (counts["tp"] + counts["tn"]) / max(len(y_true), 1)
    roc_auc = roc_auc_score(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    return {
        "row_count": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "negative_count": int(len(y_true) - y_true.sum()),
        "logloss": round(binary_logloss(y_true, y_prob), 6),
        "roc_auc": None if roc_auc is None else round(roc_auc, 6),
        "average_precision": None if average_precision is None else round(average_precision, 6),
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "specificity": round(float(specificity), 6),
        "balanced_accuracy": round(float(balanced_accuracy), 6),
        "f1": round(float(f1), 6),
        "confusion": counts,
    }


def threshold_tie_break_key(threshold: float, metrics: dict, tie_break_order: list[str]) -> tuple[float, ...]:
    key_parts: list[float] = []
    for name in tie_break_order:
        if name == "higher_threshold":
            key_parts.append(-float(threshold))
        elif name == "lower_threshold":
            key_parts.append(float(threshold))
        else:
            key_parts.append(-float(metrics.get(name, 0.0)))
    return tuple(key_parts)


def threshold_candidate_record(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, metric_name: str) -> dict:
    metrics = evaluate_probabilities(y_true, y_prob, float(threshold))
    primary = metrics.get(metric_name)
    if primary is None:
        raise ValueError(f"Unsupported threshold metric: {metric_name}")
    counts = metrics["confusion"]
    predicted_positive = int(counts["tp"] + counts["fp"])
    predicted_negative = int(counts["tn"] + counts["fn"])
    return {
        "threshold": float(threshold),
        "primary_metric": float(primary),
        "metrics": metrics,
        "predicted_positive_count": predicted_positive,
        "predicted_negative_count": predicted_negative,
    }


def select_threshold_from_candidates(candidate_records: list[dict], tie_break_order: list[str]) -> float:
    best_key = None
    best_threshold = 0.5
    for record in candidate_records:
        key = (-record["primary_metric"],) + threshold_tie_break_key(
            record["threshold"],
            record["metrics"],
            tie_break_order,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(record["threshold"])
    return best_threshold


def satisfies_prediction_guard(record: dict, guard_cfg: dict) -> bool:
    min_predicted_positive = int(guard_cfg.get("min_predicted_positive", 0))
    min_predicted_negative = int(guard_cfg.get("min_predicted_negative", 0))
    return (
        record["predicted_positive_count"] >= min_predicted_positive
        and record["predicted_negative_count"] >= min_predicted_negative
    )


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric_name: str, policy: dict) -> float:
    tie_break_order = policy["threshold_selection"].get(
        "tie_break_order",
        ["specificity", "precision", "f1", "higher_threshold"],
    )
    candidate_records = [
        threshold_candidate_record(y_true, y_prob, float(threshold), metric_name)
        for threshold in threshold_grid(y_prob)
    ]
    best_threshold = select_threshold_from_candidates(candidate_records, tie_break_order)

    guard_cfg = policy["threshold_selection"].get("degenerate_prediction_guard", {})
    if not guard_cfg or not bool(guard_cfg.get("enabled", False)):
        return best_threshold

    selected_record = next(
        record for record in candidate_records if abs(record["threshold"] - best_threshold) <= 1e-12
    )
    if satisfies_prediction_guard(selected_record, guard_cfg):
        return best_threshold

    best_primary = max(record["primary_metric"] for record in candidate_records)
    max_primary_metric_drop = float(guard_cfg.get("max_primary_metric_drop", 0.0))
    guarded_candidates = [
        record
        for record in candidate_records
        if satisfies_prediction_guard(record, guard_cfg)
        and best_primary - record["primary_metric"] <= max_primary_metric_drop + 1e-12
    ]
    if not guarded_candidates:
        return best_threshold

    guard_tie_break_order = guard_cfg.get(
        "tie_break_order",
        ["f1", "recall", "precision", "specificity", "lower_threshold"],
    )
    return select_threshold_from_candidates(guarded_candidates, guard_tie_break_order)


def rounded_unique_probability_count(probabilities: np.ndarray, decimals: int) -> int:
    if len(probabilities) == 0:
        return 0
    rounded = np.round(np.asarray(probabilities, dtype=float), int(decimals))
    return int(len(np.unique(rounded)))


def resolve_small_validation_guard(policy: dict, valid_row_count: int) -> tuple[dict, dict, dict | None]:
    cfg = policy.get("optimization", {}).get("small_validation_guard", {})
    diagnostics = {
        "enabled": bool(cfg.get("enabled", False)),
        "triggered": False,
        "valid_row_count": int(valid_row_count),
        "max_valid_rows": int(cfg.get("max_valid_rows", 0)),
        "mode": "standard",
        "applied_num_boost_round": int(policy["optimization"]["num_boost_round"]),
        "applied_early_stopping_rounds": int(policy["optimization"]["early_stopping_rounds"]),
        "applied_lightgbm_param_overrides": {},
        "force_post_train_iteration_scan": False,
    }
    if not diagnostics["enabled"]:
        return policy, diagnostics, None

    max_valid_rows = diagnostics["max_valid_rows"]
    if max_valid_rows <= 0 or int(valid_row_count) > max_valid_rows:
        return policy, diagnostics, None

    guarded_policy = json.loads(json.dumps(policy))
    fixed_tree_cfg = cfg.get("fixed_tree_training", {})
    diagnostics["triggered"] = True
    diagnostics["mode"] = "fixed_tree_training"

    if "num_boost_round" in fixed_tree_cfg:
        guarded_policy["optimization"]["num_boost_round"] = int(fixed_tree_cfg["num_boost_round"])
    if bool(fixed_tree_cfg.get("disable_early_stopping", False)):
        guarded_policy["optimization"]["early_stopping_rounds"] = 0

    lightgbm_param_overrides = fixed_tree_cfg.get("lightgbm_param_overrides", {})
    if lightgbm_param_overrides:
        guarded_policy["optimization"]["lightgbm_params"].update(lightgbm_param_overrides)

    diagnostics["applied_num_boost_round"] = int(guarded_policy["optimization"]["num_boost_round"])
    diagnostics["applied_early_stopping_rounds"] = int(guarded_policy["optimization"]["early_stopping_rounds"])
    diagnostics["applied_lightgbm_param_overrides"] = {
        str(key): guarded_policy["optimization"]["lightgbm_params"][key]
        for key in sorted(lightgbm_param_overrides.keys())
    }

    post_train_scan_cfg = json.loads(json.dumps(fixed_tree_cfg.get("post_train_iteration_scan", {})))
    if post_train_scan_cfg:
        post_train_scan_cfg["enabled"] = bool(post_train_scan_cfg.get("enabled", True))
    diagnostics["force_post_train_iteration_scan"] = bool(post_train_scan_cfg.get("enabled", False))
    return guarded_policy, diagnostics, post_train_scan_cfg if diagnostics["force_post_train_iteration_scan"] else None


def resolve_best_iteration_with_collapse_guard(
    booster,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    raw_best_iteration: int,
    policy: dict,
    cfg_override: dict | None = None,
    force_scan_reason: str | None = None,
) -> tuple[int, dict]:
    cfg = cfg_override or policy.get("optimization", {}).get("collapse_guard", {})
    metric_name = str(policy["threshold_selection"]["metric"])
    diagnostics = {
        "enabled": bool(cfg.get("enabled", False)) or bool(force_scan_reason),
        "triggered": False,
        "raw_best_iteration": int(raw_best_iteration),
        "resolved_best_iteration": int(raw_best_iteration),
        "trained_iteration_count": int(booster.current_iteration()),
        "trigger_reasons": [],
        "raw_unique_valid_probabilities": None,
        "resolved_unique_valid_probabilities": None,
        "selection": None,
    }
    if not diagnostics["enabled"]:
        return int(raw_best_iteration), diagnostics

    round_decimals = int(cfg.get("probability_round_decimals", 6))
    min_best_iteration = int(cfg.get("min_best_iteration", 1))
    min_unique_valid_probabilities = int(cfg.get("min_unique_valid_probabilities", 1))
    total_iterations = int(booster.current_iteration())
    if total_iterations <= 0:
        return int(raw_best_iteration), diagnostics

    raw_valid_prob = booster.predict(x_valid, num_iteration=int(raw_best_iteration))
    raw_unique_valid_probabilities = rounded_unique_probability_count(raw_valid_prob, round_decimals)
    diagnostics["raw_unique_valid_probabilities"] = int(raw_unique_valid_probabilities)

    trigger_reasons: list[str] = []
    if force_scan_reason:
        trigger_reasons.append(str(force_scan_reason))
    if int(raw_best_iteration) < min_best_iteration:
        trigger_reasons.append("best_iteration_below_minimum")
    if raw_unique_valid_probabilities < min_unique_valid_probabilities:
        trigger_reasons.append("insufficient_unique_valid_probabilities")
    if not trigger_reasons:
        diagnostics["resolved_unique_valid_probabilities"] = int(raw_unique_valid_probabilities)
        return int(raw_best_iteration), diagnostics

    selection_cfg = cfg.get("selection", {})
    scan_start_iteration = min(
        total_iterations,
        max(1, int(selection_cfg.get("min_iteration", min_best_iteration))),
    )
    max_primary_metric_drop = float(selection_cfg.get("max_primary_metric_drop", 0.03))

    scan_records: list[dict] = []
    for iteration in range(scan_start_iteration, total_iterations + 1):
        valid_prob = booster.predict(x_valid, num_iteration=iteration)
        threshold = choose_threshold(y_valid, valid_prob, metric_name, policy)
        metrics = evaluate_probabilities(y_valid, valid_prob, threshold)
        scan_records.append(
            {
                "iteration": int(iteration),
                "threshold": round(float(threshold), 6),
                "primary_metric": float(metrics[metric_name]),
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "unique_valid_probabilities": rounded_unique_probability_count(valid_prob, round_decimals),
            }
        )

    if not scan_records:
        diagnostics["resolved_unique_valid_probabilities"] = int(raw_unique_valid_probabilities)
        diagnostics["trigger_reasons"] = trigger_reasons
        return int(raw_best_iteration), diagnostics

    best_primary_metric = max(record["primary_metric"] for record in scan_records)
    eligible = [
        record
        for record in scan_records
        if record["primary_metric"] >= best_primary_metric - max_primary_metric_drop - 1e-12
    ]
    eligible_with_resolution = [
        record
        for record in eligible
        if record["unique_valid_probabilities"] >= min_unique_valid_probabilities
    ]
    candidate_pool = eligible_with_resolution or eligible
    selected = max(
        candidate_pool,
        key=lambda record: (
            int(record["unique_valid_probabilities"]),
            float(record["primary_metric"]),
            -1.0 if record["roc_auc"] is None else float(record["roc_auc"]),
            -1.0 if record["average_precision"] is None else float(record["average_precision"]),
            int(record["iteration"]),
        ),
    )

    diagnostics.update(
        {
            "triggered": True,
            "trigger_reasons": trigger_reasons,
            "resolved_best_iteration": int(selected["iteration"]),
            "resolved_unique_valid_probabilities": int(selected["unique_valid_probabilities"]),
            "selection": {
                "scan_start_iteration": int(scan_start_iteration),
                "scan_end_iteration": int(total_iterations),
                "max_primary_metric_drop": round(float(max_primary_metric_drop), 6),
                "min_best_iteration": int(min_best_iteration),
                "min_unique_valid_probabilities": int(min_unique_valid_probabilities),
                "best_primary_metric_across_scan": round(float(best_primary_metric), 6),
                "eligible_candidate_count": int(len(eligible)),
                "eligible_with_resolution_count": int(len(eligible_with_resolution)),
                "selected_iteration": int(selected["iteration"]),
                "selected_threshold": selected["threshold"],
                "selected_primary_metric": round(float(selected["primary_metric"]), 6),
                "selected_roc_auc": (
                    None if selected["roc_auc"] is None else round(float(selected["roc_auc"]), 6)
                ),
                "selected_average_precision": (
                    None
                    if selected["average_precision"] is None
                    else round(float(selected["average_precision"]), 6)
                ),
            },
        }
    )
    return int(selected["iteration"]), diagnostics


def apply_identifier_mask(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    mask_rate: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    if mask_rate <= 0.0:
        return x_train.copy(), 0
    mask_features = [
        "has_shared_contact_exact",
        "has_shared_pgp_fingerprint",
        "shared_contact_count_capped",
        "shared_pgp_fingerprint_count_capped",
    ]
    col_indices = [feature_names.index(name) for name in mask_features if name in feature_names]
    if not col_indices:
        return x_train.copy(), 0
    pos_indices = np.where(y_train == 1.0)[0]
    if len(pos_indices) == 0:
        return x_train.copy(), 0
    mask_count = int(round(len(pos_indices) * mask_rate))
    if mask_count <= 0:
        return x_train.copy(), 0
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pos_indices, size=mask_count, replace=False)
    masked = x_train.copy()
    masked[np.ix_(chosen, col_indices)] = 0.0
    return masked, mask_count


def ensure_non_empty(rows: list[dict], dataset_name: str) -> None:
    if not rows:
        raise ValueError(f"{dataset_name} is empty; Step 7 training cannot proceed")


def dataset_summary(rows: list[dict]) -> dict:
    return {
        "row_count": len(rows),
        "label_counts": dict(Counter(row["review_label"] for row in rows)),
        "review_stratum_counts": dict(Counter(row["review_stratum"] for row in rows)),
    }


def feature_names_for_experiment(schema: dict, policy: dict, experiment_name: str) -> list[str]:
    experiment = policy["experiments"][experiment_name]
    feature_names = []
    seen = set()
    for view_name in experiment["feature_views"]:
        for feature_name in schema["feature_views"][view_name]:
            if feature_name not in seen:
                feature_names.append(feature_name)
                seen.add(feature_name)
    for feature_name in policy["semantic_feature_sets"][experiment["semantic_feature_set"]]:
        if feature_name not in seen:
            feature_names.append(feature_name)
            seen.add(feature_name)
    return feature_names


def validate_feature_columns(rows: list[dict], feature_names: list[str], dataset_name: str) -> None:
    if not rows:
        return
    row_fields = set(rows[0].keys())
    missing = sorted(set(feature_names) - row_fields)
    if missing:
        raise SystemExit(
            f"{dataset_name} is missing required feature columns: {missing}. "
            "The pair-feature tables are likely stale for the current schema/policy. "
            "Rebuild Step 7 preview and semantic pair features before training."
        )


def finite_feature_count(rows: list[dict], feature_name: str) -> int:
    count = 0
    for row in rows:
        try:
            value = to_float(row.get(feature_name))
        except ValueError:
            continue
        if math.isfinite(value):
            count += 1
    return count


def validate_feature_values(rows: list[dict], feature_names: list[str], dataset_name: str) -> None:
    if not rows:
        return
    empty_features = [
        feature_name
        for feature_name in feature_names
        if finite_feature_count(rows, feature_name) == 0
    ]
    if empty_features:
        raise SystemExit(
            f"{dataset_name} has selected feature columns with zero finite values: {empty_features}. "
            "This would silently degrade the experiment. Rebuild Step 7 semantic pair features "
            "with every embedding/reranker required by the selected experiments, then rerun training."
        )


def random_seed_for_experiment(policy: dict, experiment: dict) -> int:
    return int(experiment.get("random_seed", policy["optimization"]["random_seed"]))


def fit_lightgbm(
    lgb,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_names: list[str],
    policy: dict,
    experiment: dict,
) -> dict:
    effective_policy, small_validation_guard, post_train_scan_cfg = resolve_small_validation_guard(
        policy,
        int(len(y_valid)),
    )
    random_seed = random_seed_for_experiment(policy, experiment)
    semantic_feature_names = semantic_feature_names_for_experiment(policy, experiment, feature_names)
    x_train_masked, masked_positive_rows = apply_identifier_mask(
        x_train,
        y_train,
        feature_names,
        float(experiment["identifier_positive_mask_rate"]),
        random_seed,
    )
    base_weights = balanced_sample_weights(y_train)
    x_train_final, y_train_final, train_weights, semantic_activation_augmentation = apply_semantic_activation_augmentation(
        x_train_masked,
        y_train,
        base_weights,
        feature_names,
        policy,
        experiment,
    )

    params = dict(effective_policy["optimization"]["lightgbm_params"])
    params["seed"] = random_seed
    params["feature_fraction_seed"] = random_seed
    params["bagging_seed"] = random_seed
    params["data_random_seed"] = random_seed

    train_set = lgb.Dataset(
        x_train_final,
        label=y_train_final,
        weight=train_weights,
        feature_name=feature_names,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        x_valid,
        label=y_valid,
        feature_name=feature_names,
        free_raw_data=False,
        reference=train_set,
    )
    eval_history: dict = {}
    callbacks = [
        lgb.log_evaluation(period=0),
        lgb.record_evaluation(eval_history),
    ]
    early_stopping_rounds = int(effective_policy["optimization"].get("early_stopping_rounds", 0))
    if early_stopping_rounds > 0:
        callbacks.insert(0, lgb.early_stopping(early_stopping_rounds, verbose=False))
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=int(effective_policy["optimization"]["num_boost_round"]),
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=callbacks,
    )
    raw_best_iteration = booster.best_iteration or int(booster.current_iteration()) or int(
        effective_policy["optimization"]["num_boost_round"]
    )
    best_iteration, collapse_guard = resolve_best_iteration_with_collapse_guard(
        booster,
        x_valid,
        y_valid,
        int(raw_best_iteration),
        effective_policy,
        cfg_override=post_train_scan_cfg,
        force_scan_reason=(
            "small_validation_guard_forced_iteration_scan"
            if bool(small_validation_guard.get("force_post_train_iteration_scan", False))
            else None
        ),
    )
    feature_gain = booster.feature_importance(importance_type="gain", iteration=best_iteration)
    feature_split = booster.feature_importance(importance_type="split", iteration=best_iteration)
    feature_importance = []
    for name, gain, split in zip(feature_names, feature_gain, feature_split, strict=True):
        feature_importance.append({"feature_name": name, "gain": round(float(gain), 6), "split": int(split)})
    feature_importance.sort(key=lambda item: (-item["gain"], -item["split"], item["feature_name"]))
    feature_family_gain = summarize_feature_family_gain(feature_importance, semantic_feature_names)
    return {
        "booster": booster,
        "best_iteration": int(best_iteration),
        "masked_positive_rows": int(masked_positive_rows),
        "masked_positive_rate_realized": round(masked_positive_rows / max(int(y_train.sum()), 1), 6),
        "random_seed": random_seed,
        "best_score": booster.best_score,
        "trained_iteration_count": int(booster.current_iteration()),
        "raw_best_iteration": int(raw_best_iteration),
        "collapse_guard": collapse_guard,
        "small_validation_guard": small_validation_guard,
        "feature_importance": feature_importance,
        "semantic_activation_augmentation": semantic_activation_augmentation,
        "feature_family_gain": feature_family_gain,
    }


def prediction_rows(rows: list[dict], probabilities: np.ndarray, threshold: float, experiment_name: str) -> list[dict]:
    predictions = (probabilities >= threshold).astype(int)
    output = []
    for row, probability, prediction in zip(rows, probabilities, predictions, strict=True):
        output.append(
            {
                "experiment_name": experiment_name,
                "pair_uid": row["pair_uid"],
                "data_bucket": row["data_bucket"],
                "split_name": row["split_name"],
                "review_label": row["review_label"],
                "y_true": label_to_int(row["review_label"]),
                "prob_positive": round(float(probability), 6),
                "pred_positive": int(prediction),
                "review_stratum": row["review_stratum"],
                "source_seller_raw_left": row["source_seller_raw_left"],
                "source_seller_raw_right": row["source_seller_raw_right"],
            }
        )
    return output


def output_path(template: str, experiment_name: str) -> Path:
    return ROOT / template.format(experiment_name=experiment_name)


def run_experiment(lgb, experiment_name: str, schema: dict, policy: dict, en_rows: list[dict], zh_rows: list[dict]) -> tuple[dict, list[dict], list[dict], object]:
    experiment = policy["experiments"][experiment_name]
    feature_names = feature_names_for_experiment(schema, policy, experiment_name)
    report_zh_zero_shot_metrics = bool(experiment.get("report_zh_zero_shot_metrics", experiment["zero_shot_safe"]))
    validate_feature_columns(en_rows, feature_names, "en_content_train_pool")
    if report_zh_zero_shot_metrics:
        validate_feature_columns(zh_rows, feature_names, "zh_target_strict")

    train_rows = select_rows(en_rows, policy["split_names"]["train"], require_core_transfer=True)
    valid_rows = select_rows(en_rows, policy["split_names"]["valid"], require_core_transfer=True)
    test_rows = select_rows(en_rows, policy["split_names"]["test"], require_core_transfer=True)
    ensure_non_empty(train_rows, f"{experiment_name}.train")
    ensure_non_empty(valid_rows, f"{experiment_name}.valid")
    ensure_non_empty(test_rows, f"{experiment_name}.test")
    validate_feature_values(
        train_rows + valid_rows + test_rows,
        feature_names,
        f"{experiment_name}.en_content_train_pool",
    )

    x_train, y_train = rows_to_matrix(train_rows, feature_names)
    x_valid, y_valid = rows_to_matrix(valid_rows, feature_names)
    x_test, y_test = rows_to_matrix(test_rows, feature_names)

    model = fit_lightgbm(lgb, x_train, y_train, x_valid, y_valid, feature_names, policy, experiment)
    valid_prob = model["booster"].predict(x_valid, num_iteration=model["best_iteration"])
    threshold = choose_threshold(y_valid, valid_prob, policy["threshold_selection"]["metric"], policy)
    test_prob = model["booster"].predict(x_test, num_iteration=model["best_iteration"])

    summary = {
        "feature_names": feature_names,
        "semantic_feature_set": experiment["semantic_feature_set"],
        "zero_shot_safe": bool(experiment["zero_shot_safe"]),
        "report_zh_zero_shot_metrics": report_zh_zero_shot_metrics,
        "train_dataset": dataset_summary(train_rows),
        "valid_dataset": dataset_summary(valid_rows),
        "test_dataset": dataset_summary(test_rows),
        "selected_threshold": round(float(threshold), 6),
        "threshold_metric": policy["threshold_selection"]["metric"],
        "threshold_selection_policy": policy["threshold_selection"],
        "valid_metrics": evaluate_probabilities(y_valid, valid_prob, threshold),
        "test_metrics": evaluate_probabilities(y_test, test_prob, threshold),
        "training_backend": "lightgbm",
        "masked_positive_rows": model["masked_positive_rows"],
        "masked_positive_rate_realized": model["masked_positive_rate_realized"],
        "random_seed": model["random_seed"],
        "best_iteration": model["best_iteration"],
        "raw_best_iteration": model["raw_best_iteration"],
        "trained_iteration_count": model["trained_iteration_count"],
        "best_score": model["best_score"],
        "collapse_guard": model["collapse_guard"],
        "small_validation_guard": model["small_validation_guard"],
        "semantic_activation_augmentation": model["semantic_activation_augmentation"],
        "feature_family_gain": model["feature_family_gain"],
        "top_feature_importance": model["feature_importance"][:20],
    }

    zh_predictions: list[dict] = []
    if report_zh_zero_shot_metrics:
        zh_test_rows = select_rows(zh_rows, policy["split_names"]["test"], require_core_transfer=True)
        ensure_non_empty(zh_test_rows, f"{experiment_name}.zh_test")
        validate_feature_values(zh_test_rows, feature_names, f"{experiment_name}.zh_target_strict")
        x_zh, y_zh = rows_to_matrix(zh_test_rows, feature_names)
        zh_prob = model["booster"].predict(x_zh, num_iteration=model["best_iteration"])
        summary["zh_test_dataset"] = dataset_summary(zh_test_rows)
        summary["zh_zero_shot_test_metrics"] = evaluate_probabilities(y_zh, zh_prob, threshold)
        zh_predictions = prediction_rows(zh_test_rows, zh_prob, threshold, experiment_name)

    en_predictions = prediction_rows(valid_rows, valid_prob, threshold, experiment_name) + prediction_rows(
        test_rows, test_prob, threshold, experiment_name
    )
    return summary, en_predictions, zh_predictions, model["booster"]


def main() -> None:
    args = parse_args()
    lgb = require_lightgbm()
    schema = load_json(SCHEMA_PATH)
    policy = load_json(POLICY_PATH)
    selected_experiments = args.experiments or policy["default_experiments"]
    unknown = sorted(set(selected_experiments) - set(policy["experiments"].keys()))
    if unknown:
        raise SystemExit(f"Unknown Step 7 experiment names: {unknown}")

    en_rows = join_frozen_with_features(
        load_csv(FROZEN_PATHS["en_content_train_pool"]),
        load_csv(PAIR_FEATURE_PATHS["en_content_train_pool"]),
    )
    zh_rows = join_frozen_with_features(
        load_csv(FROZEN_PATHS["zh_target_strict"]),
        load_csv(PAIR_FEATURE_PATHS["zh_target_strict"]),
    )

    summary = {
        "schema_path": str(SCHEMA_PATH.relative_to(ROOT)),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "input_dependencies": policy["input_dependencies"],
        "selected_experiments": selected_experiments,
        "backend": {"requested": "lightgbm", "used": "lightgbm"},
        "datasets": {
            "en_content_train_pool": {
                "reviewed_row_count": len(en_rows),
                "supervision_row_count": sum(1 for row in en_rows if row["usable_for_supervision"] == "1"),
            },
            "zh_target_strict": {
                "reviewed_row_count": len(zh_rows),
                "supervision_row_count": sum(1 for row in zh_rows if row["usable_for_supervision"] == "1"),
            },
        },
        "experiments": {},
    }

    for experiment_name in selected_experiments:
        experiment_summary, en_predictions, zh_predictions, booster = run_experiment(
            lgb,
            experiment_name,
            schema,
            policy,
            en_rows,
            zh_rows,
        )
        summary["experiments"][experiment_name] = experiment_summary
        write_csv(
            output_path(policy["output_templates"]["en_eval_predictions"], experiment_name),
            en_predictions,
            [
                "experiment_name",
                "pair_uid",
                "data_bucket",
                "split_name",
                "review_label",
                "y_true",
                "prob_positive",
                "pred_positive",
                "review_stratum",
                "source_seller_raw_left",
                "source_seller_raw_right",
            ],
        )
        if zh_predictions:
            write_csv(
                output_path(policy["output_templates"]["zh_test_predictions"], experiment_name),
                zh_predictions,
                [
                    "experiment_name",
                    "pair_uid",
                    "data_bucket",
                    "split_name",
                    "review_label",
                    "y_true",
                    "prob_positive",
                    "pred_positive",
                    "review_stratum",
                    "source_seller_raw_left",
                    "source_seller_raw_right",
                ],
            )
        model_path = output_path(policy["output_templates"]["model"], experiment_name)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(model_path), num_iteration=experiment_summary["best_iteration"])

    raw_control_name = "core_zero_shot_default_raw_style_gap_control"
    default_feature_names = (
        set(feature_names_for_experiment(schema, policy, "core_zero_shot_default"))
        if "core_zero_shot_default" in policy["experiments"]
        else set()
    )
    raw_control_feature_names = (
        set(feature_names_for_experiment(schema, policy, raw_control_name))
        if raw_control_name in policy["experiments"]
        else set()
    )
    relative_gap_features = set(schema["feature_views"].get("core_style_gap_only", []))
    raw_gap_features = set(schema["feature_views"].get("core_style_gap_raw_only", []))
    raw_control_defined = raw_control_name in policy["experiments"]

    summary["acceptance_checks"] = {
        "selected_experiments_known": True,
        "default_core_excludes_identifier_features": default_feature_names.isdisjoint(
            set(schema["feature_views"]["identifier_augmented"])
        ),
        "identifier_augmented_reported_separately": "identifier_augmented_default" in summary["experiments"],
        "raw_gap_control_declared_non_mainline": (
            not policy["experiments"][raw_control_name]["zero_shot_safe"] if raw_control_defined else True
        ),
        "raw_gap_control_replaces_relative_gap_block_only": (
            (
                default_feature_names - relative_gap_features
                == raw_control_feature_names - raw_gap_features
            )
            and raw_control_feature_names.isdisjoint(relative_gap_features)
            and raw_gap_features.issubset(raw_control_feature_names)
            if raw_control_defined and "core_zero_shot_default" in policy["experiments"]
            else True
        ),
    }
    write_json(output_path(policy["output_templates"]["summary"], "unused"), summary)
    print(json.dumps(summary["experiments"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
