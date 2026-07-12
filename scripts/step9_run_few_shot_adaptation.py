from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "step7_transfer_safe_pair_feature_schema.json"
STEP7_POLICY_PATH = ROOT / "schema" / "step7_training_policy.json"
STEP9_POLICY_PATH = ROOT / "schema" / "step9_training_policy.json"
STEP7_SUMMARY_PATH = ROOT / "reports" / "step7_training_summary.json"
FROZEN_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step5_en_frozen_silver_labels.csv",
    "zh_target_strict": ROOT / "reports" / "step5_zh_target_strict_frozen_silver_labels.csv",
}
PAIR_FEATURE_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step7_pair_features.en_content_train_pool.csv",
    "zh_target_strict": ROOT / "reports" / "step7_pair_features.zh_target_strict.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Step 9 few-shot adaptation on zh_target_strict using the fixed Step 7 feature tables "
            "and reviewed split containers."
        )
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help="Experiment name from schema/step9_training_policy.json. Defaults to policy default_experiments.",
    )
    parser.add_argument(
        "--ratio",
        action="append",
        dest="ratios",
        type=float,
        help="Chinese few-shot train ratio, e.g. 0.1. Repeat to run multiple ratios.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        help=(
            "Run seed used for both few-shot sampling and LightGBM randomization. "
            "Repeat to run multiple seeds. Defaults to policy default_seeds."
        ),
    )
    parser.add_argument(
        "--no-source-train",
        action="store_true",
        help="Disable English source-train rows and run pure target-only few-shot adaptation.",
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Optional isolated directory for every Step9 artifact and summary. Templates keep "
            "their canonical basenames but never overwrite reports/ canonical outputs."
        ),
    )
    parser.add_argument(
        "--en-pair-features",
        help="Optional isolated English pair-feature CSV; must be supplied with --zh-pair-features.",
    )
    parser.add_argument(
        "--zh-pair-features",
        help="Optional isolated Chinese pair-feature CSV; must be supplied with --en-pair-features.",
    )
    return parser.parse_args()


def redirect_output_templates(step9_policy: dict, output_root: str | None) -> None:
    if not output_root:
        return
    root = Path(output_root)
    if root.is_absolute():
        try:
            root = root.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError("Step9 isolated output root must remain inside the project") from exc
    if any(part == ".." for part in root.parts):
        raise ValueError("Step9 isolated output root must remain inside the project")
    rewritten = {}
    for key, template in step9_policy["output_templates"].items():
        rewritten[key] = str(root / Path(str(template)).name).replace("\\", "/")
    step9_policy["output_templates"] = rewritten
    step9_policy["runtime_output_root"] = str(root).replace("\\", "/")


def output_path(template: str, experiment_name: str, ratio_token: str, seed: int) -> Path:
    return ROOT / template.format(experiment_name=experiment_name, ratio_token=ratio_token, seed=seed)


def ratio_token(ratio: float) -> str:
    return f"{int(round(ratio * 100.0)):02d}pct"


def validate_ratios(ratios: list[float]) -> None:
    invalid = [ratio for ratio in ratios if ratio <= 0.0 or ratio > 1.0]
    if invalid:
        raise SystemExit(f"Step 9 ratios must satisfy 0 < ratio <= 1. Invalid values: {invalid}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context_file_record(path: Path) -> dict:
    if not path.exists():
        return {
            "path": str(path.relative_to(ROOT)),
            "exists": False,
            "size": None,
            "sha256": None,
        }
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def summary_context_fingerprints(step9_policy: dict, pair_feature_paths: dict[str, Path]) -> dict:
    context_paths = [
        Path(__file__).resolve(),
        Path(step7.__file__).resolve(),
        SCHEMA_PATH,
        STEP7_POLICY_PATH,
        STEP9_POLICY_PATH,
    ]
    for dependency in step9_policy["input_dependencies"]:
        path = Path(dependency)
        if not path.is_absolute():
            path = ROOT / path
        context_paths.append(path)
    context_paths.extend(pair_feature_paths.values())
    context_paths = list(dict.fromkeys(path.resolve() for path in context_paths))

    records = [context_file_record(path) for path in context_paths]
    combined = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        combined.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return {
        "fingerprint": combined.hexdigest(),
        "files": records,
    }


def ordered_union(left: list, right: list) -> list:
    result = []
    seen = set()
    for item in list(left or []) + list(right or []):
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def backup_existing_summary(path: Path, reason: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.stem}.codexbak.{reason}.{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def resolve_policy_path(path_value: str | None, default: Path) -> Path:
    if not path_value:
        return default
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def candidate_step7_summary_paths(
    preferred_path: Path | None = None,
    allow_archive_fallback: bool = False,
) -> list[Path]:
    reports_dir = ROOT / "reports"
    paths = []
    if preferred_path is not None:
        if preferred_path.exists():
            paths.append(preferred_path)
        elif not allow_archive_fallback:
            return []
    elif STEP7_SUMMARY_PATH.exists():
        paths.append(STEP7_SUMMARY_PATH)
    if not allow_archive_fallback:
        return paths
    if STEP7_SUMMARY_PATH.exists() and STEP7_SUMMARY_PATH not in paths:
        paths.append(STEP7_SUMMARY_PATH)
    for path in sorted(reports_dir.rglob("step7_training_summary.json")):
        if path not in paths:
            paths.append(path)
    return paths


def load_step7_summaries(
    preferred_path: Path | None = None,
    allow_archive_fallback: bool = False,
) -> list[tuple[Path, dict]]:
    summaries = [
        (path, step7.load_json(path))
        for path in candidate_step7_summary_paths(preferred_path, allow_archive_fallback=allow_archive_fallback)
    ]
    if not summaries:
        if preferred_path is not None and not allow_archive_fallback:
            raise SystemExit(
                "Step 9 could not find the Step 7 training summary required by policy at "
                f"{preferred_path}. Sync schema/step9_training_policy.json and reports/step7_training_summary.json, "
                "or intentionally point the policy at the archived snapshot you want to use."
            )
        raise SystemExit(
            "Step 9 could not find any Step 7 training summary record. "
            "Expected reports/step7_training_summary.json or an archived reports/**/step7_training_summary.json snapshot."
        )
    return summaries


def resolve_step7_experiment_summary(
    experiment_name: str,
    summary_records: list[tuple[Path, dict]],
) -> tuple[dict, Path]:
    for path, payload in summary_records:
        experiment = payload.get("experiments", {}).get(experiment_name)
        if experiment is not None:
            return experiment, path
    if len(summary_records) == 1:
        only_path = summary_records[0][0]
        raise SystemExit(
            "Step 9 could not resolve the required Step 7 experiment summary "
            f"for {experiment_name!r} inside {only_path}. Rebuild the requested Step 7 summary "
            "or point schema/step9_training_policy.json at the intended archived snapshot."
        )
    available = sorted(
        {
            name
            for _path, payload in summary_records
            for name in payload.get("experiments", {}).keys()
        }
    )
    raise SystemExit(
        "Step 9 could not resolve the required Step 7 experiment summary "
        f"for {experiment_name!r}. Available archived experiments: {available}"
    )


def minimal_prediction_fieldnames() -> list[str]:
    return [
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
    ]


def sampled_train_fieldnames() -> list[str]:
    return [
        "pair_uid",
        "split_name",
        "review_label",
        "review_stratum",
        "source_seller_raw_left",
        "source_seller_raw_right",
    ]


def synthetic_train_fieldnames() -> list[str]:
    return [
        "synthetic_pair_uid",
        "synthetic_split_name",
        "review_label",
        "synthetic_train_only",
        "source_pair_uid_left",
        "source_pair_uid_right",
        "lambda",
        "training_sample_weight",
        "source_review_stratum_left",
        "source_review_stratum_right",
        "source_seller_raw_left",
        "source_seller_raw_right",
    ]


def summarized_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "pair_uid": row["pair_uid"],
            "split_name": row["split_name"],
            "review_label": row["review_label"],
            "review_stratum": row["review_stratum"],
            "source_seller_raw_left": row["source_seller_raw_left"],
            "source_seller_raw_right": row["source_seller_raw_right"],
        }
        for row in rows
    ]


def dataset_summary(rows: list[dict]) -> dict:
    return {
        "row_count": len(rows),
        "label_counts": dict(Counter(row["review_label"] for row in rows)),
        "review_stratum_counts": dict(Counter(row["review_stratum"] for row in rows)),
    }


def label_count_summary_from_vector(y_true: np.ndarray) -> dict:
    positives = int(np.sum(np.asarray(y_true, dtype=float) == 1.0))
    negatives = int(np.sum(np.asarray(y_true, dtype=float) == 0.0))
    return {
        "positive": positives,
        "negative": negatives,
    }


def seller_uid_set(rows: list[dict]) -> set[str]:
    sellers: set[str] = set()
    for row in rows:
        left = str(row.get("seller_uid_left", "") or "").strip()
        right = str(row.get("seller_uid_right", "") or "").strip()
        if left:
            sellers.add(left)
        if right:
            sellers.add(right)
    return sellers


def seller_overlap_summary(left_rows: list[dict], right_rows: list[dict], label: str) -> dict:
    left_sellers = seller_uid_set(left_rows)
    right_sellers = seller_uid_set(right_rows)
    overlap = sorted(left_sellers & right_sellers)
    return {
        "label": label,
        "left_seller_count": len(left_sellers),
        "right_seller_count": len(right_sellers),
        "overlap_count": len(overlap),
        "overlap_examples": overlap[:10],
    }


def assert_zero_seller_overlap(left_rows: list[dict], right_rows: list[dict], label: str) -> dict:
    summary = seller_overlap_summary(left_rows, right_rows, label)
    if summary["overlap_count"] != 0:
        raise SystemExit(
            "Step 9 detected seller leakage across supervision containers for "
            f"{label}. Overlap examples: {summary['overlap_examples']}"
        )
    return summary


def clone_threshold_policy(threshold_policy: dict) -> dict:
    cloned = json.loads(json.dumps(threshold_policy))
    cloned.pop("mode", None)
    cloned.pop("bootstrap", None)
    return {"threshold_selection": cloned}


def choose_threshold_for_step9(
    y_valid: np.ndarray,
    valid_prob: np.ndarray,
    threshold_policy: dict,
    seed: int,
) -> tuple[float, dict]:
    mode = str(threshold_policy.get("mode", "single_valid"))
    base_policy = clone_threshold_policy(threshold_policy)
    metric_name = str(threshold_policy["metric"])
    single_valid_threshold = float(step7.choose_threshold(y_valid, valid_prob, metric_name, base_policy))

    diagnostics = {
        "mode": mode,
        "single_valid_threshold": round(single_valid_threshold, 6),
    }
    if mode == "single_valid":
        diagnostics["used_bootstrap"] = False
        return single_valid_threshold, diagnostics

    if mode != "bootstrap_valid":
        raise ValueError(f"Unsupported Step 9 threshold selection mode: {mode}")

    bootstrap_cfg = threshold_policy.get("bootstrap", {})
    num_resamples = int(bootstrap_cfg.get("num_resamples", 200))
    aggregation = str(bootstrap_cfg.get("aggregation", "median"))
    minimum_unique_thresholds = int(bootstrap_cfg.get("minimum_unique_thresholds_for_non_fallback", 3))
    if aggregation not in {"median", "mean"}:
        raise ValueError(f"Unsupported Step 9 bootstrap threshold aggregation: {aggregation}")
    if num_resamples <= 0:
        raise ValueError("Step 9 bootstrap threshold selection requires num_resamples > 0")

    pos_indices = np.where(y_valid == 1.0)[0]
    neg_indices = np.where(y_valid == 0.0)[0]
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        diagnostics.update(
            {
                "used_bootstrap": False,
                "fallback_to_single_valid": True,
                "fallback_reason": "missing_positive_or_negative_class_in_zh_valid",
            }
        )
        return single_valid_threshold, diagnostics

    rng = np.random.default_rng(int(seed) + 100003)
    sampled_thresholds: list[float] = []
    for _ in range(num_resamples):
        sampled_pos = rng.choice(pos_indices, size=len(pos_indices), replace=True)
        sampled_neg = rng.choice(neg_indices, size=len(neg_indices), replace=True)
        sampled_indices = np.concatenate([sampled_pos, sampled_neg])
        sampled_y = y_valid[sampled_indices]
        sampled_prob = valid_prob[sampled_indices]
        sampled_thresholds.append(float(step7.choose_threshold(sampled_y, sampled_prob, metric_name, base_policy)))

    rounded_unique_count = len({round(value, 6) for value in sampled_thresholds})
    if rounded_unique_count < minimum_unique_thresholds:
        diagnostics.update(
            {
                "used_bootstrap": False,
                "fallback_to_single_valid": True,
                "fallback_reason": "insufficient_bootstrap_threshold_diversity",
                "bootstrap": {
                    "num_resamples_requested": num_resamples,
                    "num_resamples_completed": len(sampled_thresholds),
                    "aggregation": aggregation,
                    "unique_threshold_count_rounded_6dp": rounded_unique_count,
                },
            }
        )
        return single_valid_threshold, diagnostics

    thresholds = np.array(sampled_thresholds, dtype=float)
    if aggregation == "median":
        selected_threshold = float(np.median(thresholds))
    else:
        selected_threshold = float(np.mean(thresholds))

    diagnostics.update(
        {
            "used_bootstrap": True,
            "fallback_to_single_valid": False,
            "bootstrap": {
                "num_resamples_requested": num_resamples,
                "num_resamples_completed": len(sampled_thresholds),
                "aggregation": aggregation,
                "unique_threshold_count_rounded_6dp": rounded_unique_count,
                "threshold_min": round(float(np.min(thresholds)), 6),
                "threshold_q25": round(float(np.quantile(thresholds, 0.25)), 6),
                "threshold_median": round(float(np.median(thresholds)), 6),
                "threshold_q75": round(float(np.quantile(thresholds, 0.75)), 6),
                "threshold_max": round(float(np.max(thresholds)), 6),
                "selected_threshold": round(selected_threshold, 6),
            },
        }
    )
    return selected_threshold, diagnostics

def group_key(row: dict, group_fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in group_fields)


def format_group_key(key: tuple[str, ...]) -> str:
    return "|".join(key)


def priority_group_keys(
    grouped: dict[tuple[str, ...], list[tuple[int, dict]]],
    group_fields: list[str],
    priority_group_order: list[dict],
) -> list[tuple[str, ...]]:
    ordered = []
    seen = set()
    for spec in priority_group_order:
        key = tuple(str(spec[field]) for field in group_fields)
        if key in grouped and key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def stratified_sample_rows(
    rows: list[dict],
    ratio: float,
    seed: int,
    group_fields: list[str],
    priority_group_order: list[dict],
    minimum_priority_group_count: int,
) -> list[dict]:
    if ratio >= 1.0:
        return list(rows)

    target_total = max(1, min(len(rows), int(round(len(rows) * ratio))))
    grouped: dict[tuple[str, ...], list[tuple[int, dict]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[group_key(row, group_fields)].append((idx, row))

    ordered_priority_keys = priority_group_keys(grouped, group_fields, priority_group_order)

    exact_counts: dict[tuple[str, ...], float] = {}
    counts: dict[tuple[str, ...], int] = {}
    for label, members in grouped.items():
        exact = len(members) * ratio
        exact_counts[label] = exact
        counts[label] = min(int(math.floor(exact)), len(members))

    current_total = sum(counts.values())
    protected_counts: dict[tuple[str, ...], int] = defaultdict(int)

    for key in ordered_priority_keys:
        if current_total >= target_total:
            break
        if counts[key] >= minimum_priority_group_count:
            protected_counts[key] = counts[key]
            continue
        counts[key] = min(len(grouped[key]), minimum_priority_group_count)
        protected_counts[key] = counts[key]
        current_total = sum(counts.values())

    if current_total < target_total:
        remainder_order = sorted(
            grouped.keys(),
            key=lambda label: (
                int(label in ordered_priority_keys),
                exact_counts[label] - math.floor(exact_counts[label]),
                len(grouped[label]),
                label,
            ),
            reverse=True,
        )
        while current_total < target_total:
            progressed = False
            for label in remainder_order:
                if counts[label] >= len(grouped[label]):
                    continue
                counts[label] += 1
                current_total += 1
                progressed = True
                if current_total == target_total:
                    break
            if not progressed:
                break
    elif current_total > target_total:
        reduction_order = sorted(
            grouped.keys(),
            key=lambda label: (
                int(label in ordered_priority_keys),
                exact_counts[label] - math.floor(exact_counts[label]),
                -len(grouped[label]),
                label,
            ),
        )
        while current_total > target_total:
            progressed = False
            for label in reduction_order:
                minimum_allowed = protected_counts.get(label, 0)
                if counts[label] <= minimum_allowed:
                    continue
                counts[label] -= 1
                current_total -= 1
                progressed = True
                if current_total == target_total:
                    break
            if not progressed:
                raise ValueError("Unable to reduce sampled counts to the requested Step 9 ratio")

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for label, members in grouped.items():
        choose = counts[label]
        member_indices = np.array([idx for idx, _ in members], dtype=int)
        chosen = rng.choice(member_indices, size=choose, replace=False)
        selected_indices.extend(int(idx) for idx in chosen.tolist())
    selected_indices.sort()
    return [rows[idx] for idx in selected_indices]


def feature_names_from_views(
    schema: dict,
    step7_policy: dict,
    feature_views: list[str],
    semantic_feature_set: str,
) -> list[str]:
    feature_names = []
    seen = set()
    for view_name in feature_views:
        for feature_name in schema["feature_views"][view_name]:
            if feature_name not in seen:
                feature_names.append(feature_name)
                seen.add(feature_name)
    for feature_name in step7_policy["semantic_feature_sets"][semantic_feature_set]:
        if feature_name not in seen:
            feature_names.append(feature_name)
            seen.add(feature_name)
    return feature_names


def safe_logit(probabilities: np.ndarray, clip_eps: float) -> np.ndarray:
    clipped = np.clip(np.array(probabilities, dtype=float), clip_eps, 1.0 - clip_eps)
    return np.log(clipped / (1.0 - clipped))


def safe_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.array(values, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_standardization(x_matrix: np.ndarray, enabled: bool) -> tuple[np.ndarray, dict]:
    x = np.array(x_matrix, dtype=float)
    if not enabled:
        clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return clean, {
            "enabled": False,
            "mean": [0.0 for _ in range(x.shape[1])],
            "scale": [1.0 for _ in range(x.shape[1])],
        }
    means = np.nanmean(x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    centered = np.where(np.isfinite(x), x, means)
    scales = np.nanstd(centered, axis=0)
    scales = np.where((np.isfinite(scales)) & (scales > 1e-12), scales, 1.0)
    standardized = (centered - means) / scales
    return standardized, {
        "enabled": True,
        "mean": [round(float(value), 12) for value in means],
        "scale": [round(float(value), 12) for value in scales],
    }


def apply_standardization(x_matrix: np.ndarray, standardization: dict) -> np.ndarray:
    x = np.array(x_matrix, dtype=float)
    means = np.array(standardization.get("mean", []), dtype=float)
    scales = np.array(standardization.get("scale", []), dtype=float)
    if means.size != x.shape[1] or scales.size != x.shape[1]:
        raise ValueError("Logistic artifact standardization dimensions do not match the feature matrix")
    scales = np.where(np.abs(scales) > 1e-12, scales, 1.0)
    clean = np.where(np.isfinite(x), x, means)
    return (clean - means) / scales


def logistic_sample_weights(y_true: np.ndarray, cfg: dict) -> np.ndarray:
    class_weight = str(cfg.get("class_weight", "balanced") or "balanced")
    if class_weight == "balanced":
        return step7.balanced_sample_weights(y_true)
    if class_weight in {"none", "uniform"}:
        return np.ones(len(y_true), dtype=float)
    raise ValueError(f"Unsupported Step 9 logistic class_weight: {class_weight}")


def apply_logistic_row_sample_weights(weights: np.ndarray, multipliers: np.ndarray | None) -> tuple[np.ndarray, dict]:
    if multipliers is None:
        return weights, {
            "enabled": False,
            "min_multiplier": 1.0,
            "mean_multiplier": 1.0,
            "max_multiplier": 1.0,
        }
    if len(multipliers) != len(weights):
        raise ValueError("Step 9 logistic sample-weight multiplier length mismatch")
    adjusted = weights.astype(float, copy=True) * multipliers.astype(float)
    before_mean = float(np.mean(weights)) if len(weights) else 0.0
    after_mean = float(np.mean(adjusted)) if len(adjusted) else 0.0
    if before_mean > 0.0 and after_mean > 0.0:
        adjusted *= before_mean / after_mean
    return adjusted, {
        "enabled": bool(np.any(np.abs(multipliers - 1.0) > 1e-12)),
        "min_multiplier": round(float(np.min(multipliers)), 6) if len(multipliers) else 1.0,
        "mean_multiplier": round(float(np.mean(multipliers)), 6) if len(multipliers) else 1.0,
        "max_multiplier": round(float(np.max(multipliers)), 6) if len(multipliers) else 1.0,
    }


def row_sample_weight_multipliers(rows: list[dict]) -> np.ndarray:
    return np.asarray([step7.row_training_sample_weight(row) for row in rows], dtype=float)


def fit_regularized_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    offset: np.ndarray | None = None,
    sample_weight_multipliers: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    if len(y_train) == 0:
        raise ValueError("Step 9 logistic fitting requires a non-empty training split")
    positives = int(np.sum(y_train == 1.0))
    negatives = int(np.sum(y_train == 0.0))
    if positives == 0 or negatives == 0:
        raise ValueError("Step 9 logistic fitting requires both positive and negative labels")

    standardize = bool(cfg.get("standardize_features", True))
    x_scaled, standardization = fit_standardization(x_train, standardize)
    y = np.array(y_train, dtype=float)
    weights = logistic_sample_weights(y, cfg)
    weights, row_sample_weight_summary = apply_logistic_row_sample_weights(weights, sample_weight_multipliers)
    offset_vector = np.zeros(len(y), dtype=float) if offset is None else np.array(offset, dtype=float)
    if len(offset_vector) != len(y):
        raise ValueError("Step 9 residual logistic offset length does not match y_train")

    l2_penalty = float(cfg.get("l2_penalty", 1.0))
    max_iter = int(cfg.get("max_iter", 100))
    tolerance = float(cfg.get("tolerance", 1e-8))
    if l2_penalty < 0.0:
        raise ValueError("Step 9 logistic l2_penalty must be non-negative")
    if max_iter <= 0:
        raise ValueError("Step 9 logistic max_iter must be positive")
    if tolerance <= 0.0:
        raise ValueError("Step 9 logistic tolerance must be positive")

    feature_count = x_scaled.shape[1]
    params = np.zeros(feature_count + 1, dtype=float)
    converged = False
    final_delta_norm = math.inf
    for iteration in range(1, max_iter + 1):
        logits = offset_vector + params[0] + x_scaled @ params[1:]
        probabilities = safe_sigmoid(logits)
        residual = (probabilities - y) * weights
        gradient = np.empty(feature_count + 1, dtype=float)
        gradient[0] = float(np.sum(residual))
        gradient[1:] = x_scaled.T @ residual + l2_penalty * params[1:]

        curvature = probabilities * (1.0 - probabilities) * weights
        weighted_x = x_scaled * curvature[:, None]
        hessian = np.empty((feature_count + 1, feature_count + 1), dtype=float)
        hessian[0, 0] = float(np.sum(curvature))
        hessian[0, 1:] = np.sum(weighted_x, axis=0)
        hessian[1:, 0] = hessian[0, 1:]
        hessian[1:, 1:] = x_scaled.T @ weighted_x
        if l2_penalty > 0.0:
            hessian[1:, 1:] += np.eye(feature_count) * l2_penalty

        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(hessian) @ gradient
        delta = np.clip(delta, -5.0, 5.0)
        params -= delta
        final_delta_norm = float(np.linalg.norm(delta))
        if final_delta_norm <= tolerance:
            converged = True
            break

    logits = offset_vector + params[0] + x_scaled @ params[1:]
    probabilities = safe_sigmoid(logits)
    artifact = {
        "standardization": standardization,
        "parameter_intercept": round(float(params[0]), 12),
        "parameter_coefficients": [round(float(value), 12) for value in params[1:]],
        "l2_penalty": round(float(l2_penalty), 12),
        "class_weight": str(cfg.get("class_weight", "balanced") or "balanced"),
        "row_sample_weight_multipliers": row_sample_weight_summary,
        "max_iter": max_iter,
        "tolerance": tolerance,
        "solver_iterations": int(iteration),
        "solver_converged": bool(converged),
        "solver_final_delta_norm": round(float(final_delta_norm), 12),
        "train_logloss": round(float(step7.binary_logloss(y, probabilities)), 6),
    }
    return artifact, probabilities


def apply_logistic_artifact_to_matrix(
    x_matrix: np.ndarray,
    artifact: dict,
    base_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    x_scaled = apply_standardization(x_matrix, artifact["standardization"])
    coefficients = np.array(artifact["parameter_coefficients"], dtype=float)
    if len(coefficients) != x_scaled.shape[1]:
        raise ValueError("Logistic artifact coefficient count does not match the feature matrix")
    logits = float(artifact["parameter_intercept"]) + x_scaled @ coefficients
    if base_probabilities is not None:
        clip_eps = float(artifact.get("base_probability_clip_eps", 1e-6))
        logits = safe_logit(base_probabilities, clip_eps) + logits
    return safe_sigmoid(logits)


def top_logistic_coefficients(feature_names: list[str], coefficients: list[float], limit: int = 20) -> list[dict]:
    rows = [
        {
            "feature_name": feature_name,
            "coefficient": round(float(coefficient), 6),
            "abs_coefficient": round(abs(float(coefficient)), 6),
        }
        for feature_name, coefficient in zip(feature_names, coefficients, strict=True)
    ]
    rows.sort(key=lambda row: (-row["abs_coefficient"], row["feature_name"]))
    return rows[:limit]


def step7_model_path(step7_policy: dict, experiment_name: str) -> Path:
    model_template = step7_policy.get("output_templates", {}).get("model")
    if not model_template:
        raise SystemExit("Step 9 could not resolve the Step 7 model output template.")
    return ROOT / model_template.format(experiment_name=experiment_name)


def predict_booster_rows(booster, rows: list[dict], feature_names: list[str]) -> np.ndarray:
    x_rows, _y_rows = step7.rows_to_matrix(rows, feature_names)
    return booster.predict(x_rows)


def hard_boundary_sample_rows(
    rows: list[dict],
    ratio: float,
    seed: int,
    base_probabilities: np.ndarray,
    strategy_cfg: dict,
    fallback_group_fields: list[str],
    fallback_priority_group_order: list[dict],
    fallback_minimum_priority_group_count: int,
) -> tuple[list[dict], dict]:
    if ratio >= 1.0:
        return list(rows), {
            "strategy": "hard_boundary",
            "ratio": round(float(ratio), 6),
            "target_total": len(rows),
            "selected_total": len(rows),
            "bucket_counts": {"full_target_train": len(rows)},
        }

    target_total = max(1, min(len(rows), int(round(len(rows) * ratio))))
    rng = np.random.default_rng(seed + 700001)
    jitter = rng.random(len(rows))
    base = np.array(base_probabilities, dtype=float)
    if len(base) != len(rows):
        raise ValueError("Hard-boundary sampling requires one base probability per target train row")

    negative_indices = [idx for idx, row in enumerate(rows) if row["review_label"] == "negative"]
    positive_indices = [idx for idx, row in enumerate(rows) if row["review_label"] == "positive"]
    hard_negative_share = float(strategy_cfg.get("hard_negative_share", 0.35))
    hard_positive_share = float(strategy_cfg.get("hard_positive_share", 0.35))
    hard_negative_count = min(len(negative_indices), max(1, int(round(target_total * hard_negative_share)))) if negative_indices else 0
    hard_positive_count = min(len(positive_indices), max(1, int(round(target_total * hard_positive_share)))) if positive_indices else 0
    while hard_negative_count + hard_positive_count > target_total:
        if hard_negative_count >= hard_positive_count and hard_negative_count > 0:
            hard_negative_count -= 1
        elif hard_positive_count > 0:
            hard_positive_count -= 1
        else:
            break
    typical_count = max(0, target_total - hard_negative_count - hard_positive_count)

    selected: list[int] = []
    selected_set: set[int] = set()

    hard_negative_order = sorted(
        negative_indices,
        key=lambda idx: (-float(base[idx]), float(jitter[idx]), rows[idx]["pair_uid"]),
    )
    hard_positive_order = sorted(
        positive_indices,
        key=lambda idx: (float(base[idx]), float(jitter[idx]), rows[idx]["pair_uid"]),
    )

    for idx in hard_negative_order[:hard_negative_count]:
        selected.append(idx)
        selected_set.add(idx)
    for idx in hard_positive_order[:hard_positive_count]:
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)

    median_by_label = {}
    for label, indices in (("negative", negative_indices), ("positive", positive_indices)):
        if indices:
            median_by_label[label] = float(np.median(base[indices]))
    typical_order = sorted(
        [idx for idx in range(len(rows)) if idx not in selected_set],
        key=lambda idx: (
            abs(float(base[idx]) - median_by_label.get(rows[idx]["review_label"], float(np.median(base)))),
            float(jitter[idx]),
            rows[idx]["pair_uid"],
        ),
    )
    for idx in typical_order[:typical_count]:
        selected.append(idx)
        selected_set.add(idx)

    if len(selected) < target_total:
        fallback_rows = stratified_sample_rows(
            rows,
            ratio,
            seed,
            fallback_group_fields,
            fallback_priority_group_order,
            fallback_minimum_priority_group_count,
        )
        uid_to_idx = {row["pair_uid"]: idx for idx, row in enumerate(rows)}
        for row in fallback_rows:
            idx = uid_to_idx[row["pair_uid"]]
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= target_total:
                break

    selected = sorted(selected[:target_total])
    selected_rows = [rows[idx] for idx in selected]
    selected_bucket = {}
    hard_negative_set = set(hard_negative_order[:hard_negative_count])
    hard_positive_set = set(hard_positive_order[:hard_positive_count])
    for idx in selected:
        if idx in hard_negative_set:
            bucket = "hard_negative_high_base_score"
        elif idx in hard_positive_set:
            bucket = "hard_positive_low_base_score"
        else:
            bucket = "typical_or_fallback_anchor"
        selected_bucket[rows[idx]["pair_uid"]] = bucket

    diagnostics = {
        "strategy": "hard_boundary",
        "ratio": round(float(ratio), 6),
        "target_total": int(target_total),
        "selected_total": int(len(selected_rows)),
        "base_probability_stats": {
            "min": round(float(np.min(base)), 6),
            "mean": round(float(np.mean(base)), 6),
            "median": round(float(np.median(base)), 6),
            "max": round(float(np.max(base)), 6),
        },
        "bucket_counts": dict(Counter(selected_bucket.values())),
        "selected_pair_buckets": selected_bucket,
    }
    return selected_rows, diagnostics


def sample_target_train_rows(
    rows: list[dict],
    ratio: float,
    seed: int,
    experiment_cfg: dict,
    step9_policy: dict,
    base_probabilities: np.ndarray | None,
    group_fields: list[str],
    priority_group_order: list[dict],
    minimum_priority_group_count: int,
) -> tuple[list[dict], dict]:
    strategy_name = str(
        experiment_cfg.get("sampling_strategy")
        or step9_policy.get("sampling", {}).get("default_strategy", "stratified")
    )
    if strategy_name == "stratified":
        selected = stratified_sample_rows(
            rows,
            ratio,
            seed,
            group_fields,
            priority_group_order,
            minimum_priority_group_count,
        )
        return selected, {
            "strategy": "stratified",
            "ratio": round(float(ratio), 6),
            "target_total": len(selected),
            "selected_total": len(selected),
        }
    if strategy_name == "hard_boundary":
        if base_probabilities is None:
            raise ValueError("Hard-boundary Step 9 sampling requires frozen base Step-7 probabilities")
        strategy_cfg = step9_policy.get("sampling", {}).get("strategies", {}).get("hard_boundary", {})
        return hard_boundary_sample_rows(
            rows,
            ratio,
            seed,
            base_probabilities,
            strategy_cfg,
            group_fields,
            priority_group_order,
            minimum_priority_group_count,
        )
    raise ValueError(f"Unsupported Step 9 sampling_strategy: {strategy_name}")


def truthy_flag(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def positive_mixup_enabled(experiment_cfg: dict) -> bool:
    return bool((experiment_cfg.get("positive_pair_mixup") or {}).get("enabled", False))


def positive_mixup_source_rows(rows: list[dict], cfg: dict) -> list[dict]:
    excluded_scopes = {str(value) for value in cfg.get("exclude_candidate_scopes", []) or []}
    excluded_splits = {str(value) for value in cfg.get("exclude_split_names", []) or []}
    minimum_source_weight = float(cfg.get("minimum_source_training_sample_weight", 0.0) or 0.0)
    selected = []
    for row in rows:
        if row.get("review_label") != "positive":
            continue
        if bool(cfg.get("require_usable_for_core_transfer", True)) and not truthy_flag(
            row.get("usable_for_core_transfer")
        ):
            continue
        if bool(cfg.get("require_core_transfer_eligible", True)) and not truthy_flag(
            row.get("core_transfer_eligible")
        ):
            continue
        if str(row.get("candidate_scope", "") or "") in excluded_scopes:
            continue
        if str(row.get("split_name", "") or "") in excluded_splits:
            continue
        if step7.row_training_sample_weight(row) + 1e-12 < minimum_source_weight:
            continue
        selected.append(row)
    return selected


def nearest_positive_neighbors(x_matrix: np.ndarray, k: int) -> list[list[int]]:
    if len(x_matrix) <= 1:
        return [[] for _ in range(len(x_matrix))]
    x = np.asarray(x_matrix, dtype=float)
    means = np.nanmean(x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    clean = np.where(np.isfinite(x), x, means)
    scales = np.nanstd(clean, axis=0)
    scales = np.where((np.isfinite(scales)) & (scales > 1e-12), scales, 1.0)
    scaled = (clean - means) / scales
    neighbors = []
    limit = max(1, min(int(k), len(x_matrix) - 1))
    for idx in range(len(x_matrix)):
        distances = np.sum((scaled - scaled[idx]) ** 2, axis=1)
        ranked = [int(candidate) for candidate in np.argsort(distances) if int(candidate) != idx]
        neighbors.append(ranked[:limit])
    return neighbors


def build_positive_pair_mixup_augmentation(
    sampled_zh_train_rows: list[dict],
    feature_names: list[str],
    experiment_cfg: dict,
    seed: int,
    experiment_name: str,
    run_key: str,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    cfg = dict(experiment_cfg.get("positive_pair_mixup") or {})
    diagnostics = {
        "enabled": bool(cfg.get("enabled", False)),
        "scope": "zh_train_sampled_positive_pairs_only",
        "synthetic_train_only": True,
        "source_positive_count": 0,
        "source_negative_count": 0,
        "eligible_positive_source_count": 0,
        "synthetic_row_count": 0,
        "skipped_reason": None,
    }
    if not diagnostics["enabled"]:
        diagnostics["skipped_reason"] = "disabled"
        return (
            np.empty((0, len(feature_names)), dtype=float),
            np.empty(0, dtype=float),
            [],
            diagnostics,
        )

    if len(feature_names) == 0:
        raise ValueError("Positive-pair mixup requires a non-empty feature representation")

    source_positive_rows = positive_mixup_source_rows(sampled_zh_train_rows, cfg)
    source_negative_count = sum(1 for row in sampled_zh_train_rows if row.get("review_label") == "negative")
    source_positive_count = sum(1 for row in sampled_zh_train_rows if row.get("review_label") == "positive")
    diagnostics.update(
        {
            "source_positive_count": int(source_positive_count),
            "source_negative_count": int(source_negative_count),
            "eligible_positive_source_count": int(len(source_positive_rows)),
        }
    )

    minimum_positive_sources = int(cfg.get("minimum_positive_sources", 2) or 2)
    if len(source_positive_rows) < minimum_positive_sources:
        diagnostics["skipped_reason"] = "insufficient_eligible_positive_sources"
        return (
            np.empty((0, len(feature_names)), dtype=float),
            np.empty(0, dtype=float),
            [],
            diagnostics,
        )

    target_positive_to_negative_ratio = float(cfg.get("target_positive_to_negative_ratio", 1.0) or 1.0)
    max_synthetic_per_real_positive = float(cfg.get("max_synthetic_per_real_positive", 2.0) or 2.0)
    desired_positive_count = int(round(source_negative_count * target_positive_to_negative_ratio))
    max_synthetic_count = int(math.floor(len(source_positive_rows) * max_synthetic_per_real_positive))
    synthetic_count = max(0, min(max_synthetic_count, desired_positive_count - source_positive_count))
    if synthetic_count <= 0:
        diagnostics["skipped_reason"] = "positive_count_already_meets_configured_target"
        diagnostics.update(
            {
                "target_positive_to_negative_ratio": round(float(target_positive_to_negative_ratio), 6),
                "max_synthetic_per_real_positive": round(float(max_synthetic_per_real_positive), 6),
                "desired_positive_count": int(desired_positive_count),
                "max_synthetic_count": int(max_synthetic_count),
            }
        )
        return (
            np.empty((0, len(feature_names)), dtype=float),
            np.empty(0, dtype=float),
            [],
            diagnostics,
        )

    x_positive, _y_positive = step7.rows_to_matrix(source_positive_rows, feature_names)
    neighbor_lists = nearest_positive_neighbors(x_positive, int(cfg.get("nearest_neighbor_k", 5) or 5))
    rng = np.random.default_rng(seed + 1900003)
    lambda_min = float(cfg.get("lambda_min", 0.2) or 0.2)
    lambda_max = float(cfg.get("lambda_max", 0.8) or 0.8)
    if not (0.0 <= lambda_min <= lambda_max <= 1.0):
        raise ValueError("positive_pair_mixup lambda_min/lambda_max must satisfy 0 <= min <= max <= 1")

    synthetic_features = np.empty((synthetic_count, len(feature_names)), dtype=float)
    synthetic_rows: list[dict] = []
    synthetic_weights: list[float] = []
    synthetic_weight_mode = str(cfg.get("synthetic_weight_mode", "minimum_parent_weight"))
    if synthetic_weight_mode != "minimum_parent_weight":
        raise ValueError(f"Unsupported positive-pair mixup synthetic_weight_mode: {synthetic_weight_mode}")
    for synthetic_idx in range(synthetic_count):
        left_idx = int(rng.integers(0, len(source_positive_rows)))
        neighbor_candidates = neighbor_lists[left_idx] or [
            idx for idx in range(len(source_positive_rows)) if idx != left_idx
        ]
        right_idx = int(rng.choice(neighbor_candidates))
        lambda_value = float(rng.uniform(lambda_min, lambda_max))
        synthetic_features[synthetic_idx, :] = (
            (1.0 - lambda_value) * x_positive[left_idx, :] + lambda_value * x_positive[right_idx, :]
        )
        left_row = source_positive_rows[left_idx]
        right_row = source_positive_rows[right_idx]
        synthetic_weight = min(
            step7.row_training_sample_weight(left_row),
            step7.row_training_sample_weight(right_row),
        )
        synthetic_weights.append(float(synthetic_weight))
        synthetic_rows.append(
            {
                "synthetic_pair_uid": f"synthetic_train_only::{experiment_name}::{run_key}::{synthetic_idx + 1:05d}",
                "synthetic_split_name": "synthetic_train_only",
                "review_label": "positive",
                "synthetic_train_only": "1",
                "source_pair_uid_left": left_row["pair_uid"],
                "source_pair_uid_right": right_row["pair_uid"],
                "lambda": round(float(lambda_value), 6),
                "training_sample_weight": round(float(synthetic_weight), 6),
                "source_review_stratum_left": left_row.get("review_stratum", ""),
                "source_review_stratum_right": right_row.get("review_stratum", ""),
                "source_seller_raw_left": left_row.get("source_seller_raw_left", ""),
                "source_seller_raw_right": right_row.get("source_seller_raw_right", ""),
            }
        )

    diagnostics.update(
        {
            "skipped_reason": None,
            "target_positive_to_negative_ratio": round(float(target_positive_to_negative_ratio), 6),
            "max_synthetic_per_real_positive": round(float(max_synthetic_per_real_positive), 6),
            "desired_positive_count": int(desired_positive_count),
            "max_synthetic_count": int(max_synthetic_count),
            "synthetic_row_count": int(synthetic_count),
            "lambda_min": round(float(lambda_min), 6),
            "lambda_max": round(float(lambda_max), 6),
            "nearest_neighbor_k": int(cfg.get("nearest_neighbor_k", 5) or 5),
            "minimum_source_training_sample_weight": round(
                float(cfg.get("minimum_source_training_sample_weight", 0.0) or 0.0), 6
            ),
            "synthetic_weight_mode": synthetic_weight_mode,
            "synthetic_weight_stats": {
                "min": round(float(np.min(synthetic_weights)), 6),
                "mean": round(float(np.mean(synthetic_weights)), 6),
                "max": round(float(np.max(synthetic_weights)), 6),
            },
            "exclusion_policy": {
                "require_usable_for_core_transfer": bool(cfg.get("require_usable_for_core_transfer", True)),
                "require_core_transfer_eligible": bool(cfg.get("require_core_transfer_eligible", True)),
                "exclude_candidate_scopes": list(cfg.get("exclude_candidate_scopes", []) or []),
                "exclude_split_names": list(cfg.get("exclude_split_names", []) or []),
            },
        }
    )
    return synthetic_features, np.ones(synthetic_count, dtype=float), synthetic_rows, diagnostics


def fit_for_step9(
    lgb,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_names: list[str],
    step7_policy_for_step9: dict,
    step7_experiment: dict,
    seed: int,
    train_rows: list[dict] | None = None,
) -> dict:
    train_policy = json.loads(json.dumps(step7_policy_for_step9))
    train_policy["optimization"]["random_seed"] = int(seed)
    train_experiment = json.loads(json.dumps(step7_experiment))
    train_experiment["random_seed"] = int(seed)
    return step7.fit_lightgbm(
        lgb,
        x_train,
        y_train,
        x_valid,
        y_valid,
        feature_names,
        train_policy,
        train_experiment,
        train_rows=train_rows,
    )


def apply_low_ratio_guard_to_training_policy(
    step9_train_policy: dict,
    step9_policy: dict,
    ratio: float,
    sampled_target_train_rows: list[dict],
    zh_valid_rows: list[dict],
) -> tuple[dict, dict]:
    cfg = json.loads(json.dumps(step9_policy.get("training", {}).get("low_ratio_guard", {})))
    diagnostics = {
        "enabled": bool(cfg.get("enabled", False)),
        "triggered": False,
        "ratio": round(float(ratio), 6),
        "sampled_target_train_row_count": int(len(sampled_target_train_rows)),
        "zh_valid_row_count": int(len(zh_valid_rows)),
        "max_ratio": None,
        "max_sampled_target_train_rows": None,
        "trigger_reasons": [],
        "applied_num_boost_round": int(step9_train_policy["optimization"]["num_boost_round"]),
        "applied_early_stopping_rounds": int(step9_train_policy["optimization"]["early_stopping_rounds"]),
        "applied_lightgbm_param_overrides": {},
        "applied_small_validation_guard_override": False,
    }
    if not diagnostics["enabled"]:
        return step9_train_policy, diagnostics

    max_ratio = float(cfg.get("max_ratio", 0.0))
    max_sampled_target_train_rows = int(cfg.get("max_sampled_target_train_rows", 0))
    diagnostics["max_ratio"] = round(float(max_ratio), 6)
    diagnostics["max_sampled_target_train_rows"] = int(max_sampled_target_train_rows)

    trigger_reasons: list[str] = []
    if max_ratio > 0.0 and float(ratio) <= max_ratio + 1e-12:
        trigger_reasons.append("ratio_below_or_equal_guard_max_ratio")
    if max_sampled_target_train_rows > 0 and len(sampled_target_train_rows) <= max_sampled_target_train_rows:
        trigger_reasons.append("sampled_target_train_rows_below_or_equal_guard_max_rows")
    if not trigger_reasons:
        return step9_train_policy, diagnostics

    guarded_policy = json.loads(json.dumps(step9_train_policy))
    fixed_tree_cfg = cfg.get("fixed_tree_training", {})
    if "num_boost_round" in fixed_tree_cfg:
        guarded_policy["optimization"]["num_boost_round"] = int(fixed_tree_cfg["num_boost_round"])
    if bool(fixed_tree_cfg.get("disable_early_stopping", False)):
        guarded_policy["optimization"]["early_stopping_rounds"] = 0

    lightgbm_param_overrides = fixed_tree_cfg.get("lightgbm_param_overrides", {})
    if lightgbm_param_overrides:
        guarded_policy["optimization"]["lightgbm_params"].update(lightgbm_param_overrides)

    post_train_scan_cfg = json.loads(json.dumps(fixed_tree_cfg.get("post_train_iteration_scan", {})))
    if post_train_scan_cfg:
        post_train_scan_cfg["enabled"] = bool(post_train_scan_cfg.get("enabled", True))
        small_validation_guard = guarded_policy["optimization"].setdefault("small_validation_guard", {})
        small_validation_guard["enabled"] = True
        small_validation_guard["max_valid_rows"] = max(
            int(len(zh_valid_rows)),
            int(small_validation_guard.get("max_valid_rows", 0)),
        )
        small_validation_guard["fixed_tree_training"] = post_train_scan_cfg and {
            "num_boost_round": int(guarded_policy["optimization"]["num_boost_round"]),
            "disable_early_stopping": int(guarded_policy["optimization"]["early_stopping_rounds"]) == 0,
            "lightgbm_param_overrides": lightgbm_param_overrides,
            "post_train_iteration_scan": post_train_scan_cfg,
        }

    diagnostics.update(
        {
            "triggered": True,
            "trigger_reasons": trigger_reasons,
            "applied_num_boost_round": int(guarded_policy["optimization"]["num_boost_round"]),
            "applied_early_stopping_rounds": int(guarded_policy["optimization"]["early_stopping_rounds"]),
            "applied_lightgbm_param_overrides": {
                str(key): guarded_policy["optimization"]["lightgbm_params"][key]
                for key in sorted(lightgbm_param_overrides.keys())
            },
            "applied_small_validation_guard_override": bool(post_train_scan_cfg),
        }
    )
    return guarded_policy, diagnostics


def build_step9_training_policy(step7_policy: dict, step9_policy: dict) -> tuple[dict, dict]:
    merged = json.loads(json.dumps(step7_policy))
    overrides = json.loads(json.dumps(step9_policy.get("training", {}).get("lightgbm_overrides", {})))
    if not overrides:
        return merged, {}

    if "num_boost_round" in overrides:
        merged["optimization"]["num_boost_round"] = int(overrides["num_boost_round"])
    if "early_stopping_rounds" in overrides:
        merged["optimization"]["early_stopping_rounds"] = int(overrides["early_stopping_rounds"])

    lightgbm_param_overrides = overrides.get("lightgbm_params", {})
    if lightgbm_param_overrides:
        merged["optimization"]["lightgbm_params"].update(lightgbm_param_overrides)

    applied = {}
    if "num_boost_round" in overrides:
        applied["num_boost_round"] = int(merged["optimization"]["num_boost_round"])
    if "early_stopping_rounds" in overrides:
        applied["early_stopping_rounds"] = int(merged["optimization"]["early_stopping_rounds"])
    if lightgbm_param_overrides:
        applied["lightgbm_params"] = {
            key: merged["optimization"]["lightgbm_params"][key]
            for key in sorted(lightgbm_param_overrides.keys())
        }
    return merged, applied

def summarize_metric_series(values: list[float]) -> dict:
    ordered = [float(value) for value in values if value is not None]
    if not ordered:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": round(float(np.mean(ordered)), 6),
        "min": round(float(np.min(ordered)), 6),
        "max": round(float(np.max(ordered)), 6),
    }


def collect_metric(runs: list[dict], metrics_key: str, metric_name: str) -> list[float]:
    values = []
    for run in runs:
        value = run.get(metrics_key, {}).get(metric_name)
        if value is not None:
            values.append(float(value))
    return values


def summarize_step9_experiment_runs(experiment_summary: dict) -> dict[str, dict]:
    grouped_runs: dict[str, list[dict]] = defaultdict(list)
    for run in experiment_summary["runs"].values():
        grouped_runs[str(run["ratio_token"])].append(run)

    aggregate: dict[str, dict] = {}
    for ratio_token, runs in sorted(grouped_runs.items()):
        adaptive_balanced_accuracy = collect_metric(runs, "zh_test_metrics", "balanced_accuracy")
        adaptive_roc_auc = collect_metric(runs, "zh_test_metrics", "roc_auc")
        adaptive_average_precision = collect_metric(runs, "zh_test_metrics", "average_precision")
        adaptive_pr_auc = collect_metric(runs, "zh_test_metrics", "pr_auc")
        adaptive_f1 = collect_metric(runs, "zh_test_metrics", "f1")
        adaptive_map = collect_metric(runs, "zh_test_metrics", "map")
        adaptive_mrr = collect_metric(runs, "zh_test_metrics", "mrr")
        fixed_balanced_accuracy = collect_metric(runs, "zh_test_metrics_fixed_base_threshold", "balanced_accuracy")
        fixed_roc_auc = collect_metric(runs, "zh_test_metrics_fixed_base_threshold", "roc_auc")
        fixed_average_precision = collect_metric(runs, "zh_test_metrics_fixed_base_threshold", "average_precision")
        fixed_pr_auc = collect_metric(runs, "zh_test_metrics_fixed_base_threshold", "pr_auc")
        fixed_f1 = collect_metric(runs, "zh_test_metrics_fixed_base_threshold", "f1")
        fixed_map = collect_metric(runs, "zh_test_metrics_fixed_base_threshold", "map")
        fixed_mrr = collect_metric(runs, "zh_test_metrics_fixed_base_threshold", "mrr")
        selected_thresholds = [run["selected_threshold"] for run in runs]

        aggregate[ratio_token] = {
            "ratio": round(float(runs[0]["ratio"]), 6),
            "run_count": len(runs),
            "selected_threshold": summarize_metric_series(selected_thresholds),
            "zh_test_metrics": {
                "balanced_accuracy": summarize_metric_series(adaptive_balanced_accuracy),
                "roc_auc": summarize_metric_series(adaptive_roc_auc),
                "average_precision": summarize_metric_series(adaptive_average_precision),
                "pr_auc": summarize_metric_series(adaptive_pr_auc),
                "f1": summarize_metric_series(adaptive_f1),
                "map": summarize_metric_series(adaptive_map),
                "mrr": summarize_metric_series(adaptive_mrr),
            },
            "zh_test_metrics_fixed_base_threshold": {
                "balanced_accuracy": summarize_metric_series(fixed_balanced_accuracy),
                "roc_auc": summarize_metric_series(fixed_roc_auc),
                "average_precision": summarize_metric_series(fixed_average_precision),
                "pr_auc": summarize_metric_series(fixed_pr_auc),
                "f1": summarize_metric_series(fixed_f1),
                "map": summarize_metric_series(fixed_map),
                "mrr": summarize_metric_series(fixed_mrr),
            },
        }
    return aggregate


def set_step9_acceptance_checks(summary: dict) -> None:
    zh_valid_count = int(summary["fixed_datasets"]["zh_valid"]["row_count"])
    zh_test_count = int(summary["fixed_datasets"]["zh_test"]["row_count"])
    summary["acceptance_checks"] = {
        "fixed_zh_valid_row_count": zh_valid_count,
        "fixed_zh_test_row_count": zh_test_count,
        "all_runs_keep_zh_test_fixed": all(
            run["zh_test_dataset"]["row_count"] == zh_test_count
            for experiment in summary["experiments"].values()
            for run in experiment["runs"].values()
        ),
        "all_runs_keep_zh_valid_fixed": all(
            run["zh_valid_dataset"]["row_count"] == zh_valid_count
            for experiment in summary["experiments"].values()
            for run in experiment["runs"].values()
        ),
    }


def merge_step9_experiment_summary(existing_experiment: dict | None, new_experiment: dict) -> dict:
    if not existing_experiment:
        return new_experiment
    merged = dict(new_experiment)
    merged_runs = dict(existing_experiment.get("runs", {}))
    merged_runs.update(new_experiment.get("runs", {}))
    merged["runs"] = merged_runs
    merged["aggregate_by_ratio"] = summarize_step9_experiment_runs(merged)
    return merged


def merge_with_existing_summary(summary_path: Path, new_summary: dict) -> dict:
    context = new_summary["summary_context_fingerprints"]
    if not summary_path.exists():
        new_summary["summary_merge_mode"] = "new_summary"
        new_summary["summary_merge_history"] = []
        return new_summary

    existing_summary = step7.load_json(summary_path)
    existing_context = existing_summary.get("summary_context_fingerprints", {})
    if existing_context.get("fingerprint") != context.get("fingerprint"):
        backup_path = backup_existing_summary(summary_path, "context_mismatch")
        new_summary["summary_merge_mode"] = "new_summary_after_context_mismatch"
        new_summary["summary_merge_history"] = list(existing_summary.get("summary_merge_history", [])) + [
            {
                "event": "existing_summary_backed_up_before_context_mismatch_replacement",
                "backup_path": str(backup_path.relative_to(ROOT)),
                "existing_fingerprint": existing_context.get("fingerprint"),
                "new_fingerprint": context.get("fingerprint"),
                "merge_policy": "do_not_merge_old_experiments_across_data_contexts",
            }
        ]
        set_step9_acceptance_checks(new_summary)
        return new_summary

    merged_summary = dict(new_summary)
    existing_experiments = existing_summary.get("experiments", {})
    merged_experiments = dict(existing_experiments)
    for experiment_name, experiment_summary in new_summary["experiments"].items():
        merged_experiments[experiment_name] = merge_step9_experiment_summary(
            existing_experiments.get(experiment_name),
            experiment_summary,
        )
    merged_summary["experiments"] = merged_experiments
    merged_summary["selected_experiments"] = ordered_union(
        existing_summary.get("selected_experiments", []),
        new_summary.get("selected_experiments", []),
    )
    merged_summary["selected_ratios"] = ordered_union(
        existing_summary.get("selected_ratios", []),
        new_summary.get("selected_ratios", []),
    )
    merged_summary["selected_seeds"] = ordered_union(
        existing_summary.get("selected_seeds", []),
        new_summary.get("selected_seeds", []),
    )
    merged_summary["summary_merge_mode"] = "merged_same_context"
    merged_summary["summary_merge_history"] = list(existing_summary.get("summary_merge_history", [])) + [
        {
            "event": "merged_same_context",
            "timestamp_local": datetime.now().replace(microsecond=0).isoformat(),
            "merged_experiments": new_summary.get("selected_experiments", []),
            "merged_ratios": new_summary.get("selected_ratios", []),
            "merged_seeds": new_summary.get("selected_seeds", []),
        }
    ]
    set_step9_acceptance_checks(merged_summary)
    return merged_summary


def main() -> None:
    args = parse_args()
    lgb = step7.require_lightgbm()
    schema = step7.load_json(SCHEMA_PATH)
    step7_policy = step7.load_json(STEP7_POLICY_PATH)
    step9_policy = step7.load_json(STEP9_POLICY_PATH)
    redirect_output_templates(step9_policy, args.output_root)
    if bool(args.en_pair_features) != bool(args.zh_pair_features):
        raise SystemExit("Step9 isolated feature inputs require both --en-pair-features and --zh-pair-features")
    pair_feature_paths = dict(PAIR_FEATURE_PATHS)
    if args.en_pair_features:
        pair_feature_paths = {
            "en_content_train_pool": resolve_policy_path(args.en_pair_features, PAIR_FEATURE_PATHS["en_content_train_pool"]),
            "zh_target_strict": resolve_policy_path(args.zh_pair_features, PAIR_FEATURE_PATHS["zh_target_strict"]),
        }
    for pool_name, path in pair_feature_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Step9 pair feature input is missing for {pool_name}: {path}")
    policy_step7_summary = step9_policy.get("baseline_reference", {}).get("step7_summary")
    preferred_step7_summary_path = resolve_policy_path(
        policy_step7_summary,
        STEP7_SUMMARY_PATH,
    )
    step7_summary_records = load_step7_summaries(
        preferred_step7_summary_path,
        allow_archive_fallback=not bool(policy_step7_summary),
    )
    step9_train_policy, training_overrides_applied = build_step9_training_policy(step7_policy, step9_policy)

    selected_experiments = args.experiments or step9_policy["default_experiments"]
    unknown_experiments = sorted(set(selected_experiments) - set(step9_policy["experiments"].keys()))
    if unknown_experiments:
        raise SystemExit(f"Unknown Step 9 experiment names: {unknown_experiments}")

    selected_ratios = args.ratios or step9_policy["adaptation_ratios"]
    validate_ratios(selected_ratios)
    selected_seeds = args.seeds or step9_policy["default_seeds"]
    sampling_group_fields = [str(field) for field in step9_policy["sampling"]["group_fields"]]
    priority_group_order = list(step9_policy["sampling"]["priority_group_order"])
    minimum_priority_group_count = int(step9_policy["sampling"]["minimum_priority_group_count_if_available"])
    include_source_train = bool(step9_policy["training"]["include_source_train"]) and not args.no_source_train

    en_rows = step7.join_frozen_with_features(
        step7.load_csv(FROZEN_PATHS["en_content_train_pool"]),
        step7.load_csv(pair_feature_paths["en_content_train_pool"]),
    )
    zh_rows = step7.join_frozen_with_features(
        step7.load_csv(FROZEN_PATHS["zh_target_strict"]),
        step7.load_csv(pair_feature_paths["zh_target_strict"]),
    )

    source_train_rows = step7.select_rows(en_rows, step9_policy["sampling"]["source_train_split"], require_core_transfer=True)
    source_test_rows = step7.select_rows(en_rows, step9_policy["sampling"]["source_test_split"], require_core_transfer=True)
    zh_train_rows = step7.select_rows(zh_rows, step9_policy["sampling"]["target_train_split"], require_core_transfer=True)
    zh_valid_rows = step7.select_rows(zh_rows, step9_policy["sampling"]["target_valid_split"], require_core_transfer=True)
    zh_test_rows = step7.select_rows(zh_rows, step9_policy["sampling"]["target_test_split"], require_core_transfer=True)

    step7.ensure_non_empty(zh_train_rows, "step9.zh_train")
    step7.ensure_non_empty(zh_valid_rows, "step9.zh_valid")
    step7.ensure_non_empty(zh_test_rows, "step9.zh_test")
    if include_source_train:
        step7.ensure_non_empty(source_train_rows, "step9.en_train")

    seller_overlap_checks = {
        "zh_train_vs_zh_valid": assert_zero_seller_overlap(zh_train_rows, zh_valid_rows, "zh_train_vs_zh_valid"),
        "zh_train_vs_zh_test": assert_zero_seller_overlap(zh_train_rows, zh_test_rows, "zh_train_vs_zh_test"),
        "zh_valid_vs_zh_test": assert_zero_seller_overlap(zh_valid_rows, zh_test_rows, "zh_valid_vs_zh_test"),
    }
    if include_source_train:
        seller_overlap_checks["source_train_vs_source_test"] = assert_zero_seller_overlap(
            source_train_rows,
            source_test_rows,
            "source_train_vs_source_test",
        )

    baseline_core, baseline_summary_path = resolve_step7_experiment_summary(
        "core_zero_shot_default",
        step7_summary_records,
    )
    summary = {
        "schema_path": str(SCHEMA_PATH.relative_to(ROOT)),
        "step7_policy_path": str(STEP7_POLICY_PATH.relative_to(ROOT)),
        "step9_policy_path": str(STEP9_POLICY_PATH.relative_to(ROOT)),
        "runtime_output_root": step9_policy.get("runtime_output_root"),
        "step7_summary_path": str(preferred_step7_summary_path.relative_to(ROOT)),
        "step7_summary_resolution_mode": (
            "explicit_policy_path" if policy_step7_summary else "top_level_or_archive_fallback"
        ),
        "step7_summary_search_paths": [str(path.relative_to(ROOT)) for path, _payload in step7_summary_records],
        "input_dependencies": step9_policy["input_dependencies"],
        "runtime_pair_feature_paths": {
            pool_name: str(path.relative_to(ROOT)).replace("\\", "/")
            for pool_name, path in pair_feature_paths.items()
        },
        "summary_context_fingerprints": summary_context_fingerprints(step9_policy, pair_feature_paths),
        "selected_experiments": selected_experiments,
        "selected_ratios": selected_ratios,
        "selected_seeds": selected_seeds,
        "include_source_train": include_source_train,
        "fixed_base_threshold_interpretation": (
            "diagnostic_cross_model_reference_only"
        ),
        "baseline_reference": {
            "experiment_name": "core_zero_shot_default",
            "resolved_from_step7_summary_path": str(baseline_summary_path.relative_to(ROOT)),
            "zh_zero_shot_test_metrics": baseline_core["zh_zero_shot_test_metrics"],
            "selected_threshold": baseline_core["selected_threshold"],
        },
        "fixed_datasets": {
            "source_train": dataset_summary(source_train_rows),
            "source_test": dataset_summary(source_test_rows),
            "zh_train_full": dataset_summary(zh_train_rows),
            "zh_valid": dataset_summary(zh_valid_rows),
            "zh_test": dataset_summary(zh_test_rows),
        },
        "seller_overlap_checks": seller_overlap_checks,
        "sampling_policy": {
            "stratify_by": step9_policy["sampling"]["stratify_by"],
            "group_fields": sampling_group_fields,
            "priority_group_order": priority_group_order,
            "minimum_priority_group_count_if_available": minimum_priority_group_count,
        },
        "training_policy_overrides_applied": training_overrides_applied,
        "experiments": {},
    }

    for experiment_name in selected_experiments:
        experiment_cfg = step9_policy["experiments"][experiment_name]
        base_step7_experiment_name = experiment_cfg["base_step7_experiment"]
        backend = str(experiment_cfg.get("backend", "legacy_lightgbm_mixed") or "legacy_lightgbm_mixed")
        if backend not in {"legacy_lightgbm_mixed", "residual_logistic", "logistic_regression_l2"}:
            raise SystemExit(f"Unsupported Step 9 backend for {experiment_name!r}: {backend!r}")
        if positive_mixup_enabled(experiment_cfg) and backend != "logistic_regression_l2":
            raise SystemExit(
                "Step 9 positive-pair mixup is implemented only for the logistic_regression_l2 backend. "
                f"Experiment {experiment_name!r} uses backend={backend!r}."
            )
        base_step7_experiment = step7_policy["experiments"][base_step7_experiment_name]
        base_step7_summary, base_step7_summary_path = resolve_step7_experiment_summary(
            base_step7_experiment_name,
            step7_summary_records,
        )
        fixed_base_threshold = float(base_step7_summary["selected_threshold"])
        base_feature_names = list(
            base_step7_summary.get("feature_names")
            or step7.feature_names_for_experiment(schema, step7_policy, base_step7_experiment_name)
        )
        base_model_path = step7_model_path(step7_policy, base_step7_experiment_name)
        if not base_model_path.exists():
            raise SystemExit(
                "Step 9 resolved the base Step 7 model path to "
                f"{base_model_path}, but that file does not exist."
            )
        base_booster = lgb.Booster(model_file=str(base_model_path))
        zh_train_base_prob = predict_booster_rows(base_booster, zh_train_rows, base_feature_names)
        zh_valid_base_prob = predict_booster_rows(base_booster, zh_valid_rows, base_feature_names)
        zh_test_base_prob = predict_booster_rows(base_booster, zh_test_rows, base_feature_names)
        source_test_base_prob = (
            predict_booster_rows(base_booster, source_test_rows, base_feature_names)
            if source_test_rows
            else np.array([], dtype=float)
        )
        zh_train_base_prob_by_uid = {
            row["pair_uid"]: float(probability)
            for row, probability in zip(zh_train_rows, zh_train_base_prob, strict=True)
        }

        if backend == "legacy_lightgbm_mixed":
            feature_names = step7.feature_names_for_experiment(schema, step7_policy, base_step7_experiment_name)
        elif backend == "residual_logistic":
            feature_names = feature_names_from_views(
                schema,
                step7_policy,
                list(experiment_cfg.get("residual_feature_views", base_step7_experiment["feature_views"])),
                str(experiment_cfg.get("residual_semantic_feature_set", base_step7_experiment["semantic_feature_set"])),
            )
        else:
            feature_names = feature_names_from_views(
                schema,
                step7_policy,
                list(experiment_cfg.get("feature_views", base_step7_experiment["feature_views"])),
                str(experiment_cfg.get("semantic_feature_set", base_step7_experiment["semantic_feature_set"])),
            )

        step7.validate_feature_columns(zh_train_rows, feature_names, f"step9.{experiment_name}.zh_train")
        step7.validate_feature_columns(zh_valid_rows, feature_names, f"step9.{experiment_name}.zh_valid")
        step7.validate_feature_columns(zh_test_rows, feature_names, f"step9.{experiment_name}.zh_test")
        if source_train_rows:
            step7.validate_feature_columns(source_train_rows, feature_names, f"step9.{experiment_name}.source_train")
        if source_test_rows:
            step7.validate_feature_columns(source_test_rows, feature_names, f"step9.{experiment_name}.source_test")

        experiment_summary = {
            "base_step7_experiment": base_step7_experiment_name,
            "resolved_from_step7_summary_path": str(base_step7_summary_path.relative_to(ROOT)),
            "training_backend": backend,
            "scoring_backend": backend,
            "role": str(experiment_cfg.get("role", "")),
            "feature_names": feature_names,
            "base_feature_names": base_feature_names,
            "base_model_path": str(base_model_path.relative_to(ROOT)),
            "fixed_base_threshold": round(fixed_base_threshold, 6),
            "fixed_base_threshold_role": "diagnostic_cross_model_reference_only",
            "runs": {},
        }

        for ratio in selected_ratios:
            ratio_key = ratio_token(ratio)
            for seed in selected_seeds:
                run_key = f"{ratio_key}_seed_{seed}"
                sampled_zh_train_rows, sampling_diagnostics = sample_target_train_rows(
                    zh_train_rows,
                    float(ratio),
                    int(seed),
                    experiment_cfg,
                    step9_policy,
                    zh_train_base_prob,
                    sampling_group_fields,
                    priority_group_order,
                    minimum_priority_group_count,
                )
                run_include_source_train = (
                    bool(experiment_cfg.get("include_source_train", include_source_train))
                    and include_source_train
                    and backend != "residual_logistic"
                )
                train_rows = list(sampled_zh_train_rows)
                if run_include_source_train:
                    train_rows = list(source_train_rows) + train_rows

                sampled_target_seller_overlap_checks = {
                    "sampled_zh_train_vs_zh_valid": assert_zero_seller_overlap(
                        sampled_zh_train_rows,
                        zh_valid_rows,
                        f"{experiment_name}.{run_key}.sampled_zh_train_vs_zh_valid",
                    ),
                    "sampled_zh_train_vs_zh_test": assert_zero_seller_overlap(
                        sampled_zh_train_rows,
                        zh_test_rows,
                        f"{experiment_name}.{run_key}.sampled_zh_train_vs_zh_test",
                    ),
                }

                x_valid, y_valid = step7.rows_to_matrix(zh_valid_rows, feature_names)
                x_zh_test, y_zh_test = step7.rows_to_matrix(zh_test_rows, feature_names)

                model = None
                scorer_artifact = None
                scorer_artifact_path = None
                low_ratio_guard = {
                    "enabled": False,
                    "triggered": False,
                    "reason": "not_applicable_to_non_lightgbm_backend",
                }
                best_iteration = 0
                raw_best_iteration = 0
                trained_iteration_count = 0
                best_score = None
                collapse_guard = None
                small_validation_guard = None
                masked_positive_rows = 0
                masked_positive_rate_realized = 0.0
                semantic_activation_augmentation = None
                feature_family_gain = None
                top_feature_importance: list[dict] = []
                top_coefficients: list[dict] = []
                training_random_seed = int(seed)
                positive_pair_mixup = {
                    "enabled": False,
                    "synthetic_train_only": True,
                    "synthetic_row_count": 0,
                    "skipped_reason": "not_configured_for_this_backend",
                }
                synthetic_train_rows: list[dict] = []
                synthetic_train_only_path = None

                if backend == "legacy_lightgbm_mixed":
                    run_train_policy, low_ratio_guard = apply_low_ratio_guard_to_training_policy(
                        step9_train_policy,
                        step9_policy,
                        float(ratio),
                        sampled_zh_train_rows,
                        zh_valid_rows,
                    )
                    x_train, y_train = step7.rows_to_matrix(train_rows, feature_names)
                    model = fit_for_step9(
                        lgb,
                        x_train,
                        y_train,
                        x_valid,
                        y_valid,
                        feature_names,
                        run_train_policy,
                        base_step7_experiment,
                        int(seed),
                        train_rows=train_rows,
                    )
                    valid_prob = model["booster"].predict(x_valid, num_iteration=model["best_iteration"])
                    zh_test_prob = model["booster"].predict(x_zh_test, num_iteration=model["best_iteration"])
                    best_iteration = int(model["best_iteration"])
                    raw_best_iteration = int(model["raw_best_iteration"])
                    trained_iteration_count = int(model["trained_iteration_count"])
                    best_score = model["best_score"]
                    collapse_guard = model.get("collapse_guard")
                    small_validation_guard = model.get("small_validation_guard")
                    masked_positive_rows = int(model["masked_positive_rows"])
                    masked_positive_rate_realized = float(model["masked_positive_rate_realized"])
                    semantic_activation_augmentation = model.get("semantic_activation_augmentation")
                    feature_family_gain = model.get("feature_family_gain")
                    top_feature_importance = model["feature_importance"][:20]
                elif backend == "logistic_regression_l2":
                    logistic_cfg = step9_policy["training"]["logistic_regression"]
                    x_train, y_train = step7.rows_to_matrix(train_rows, feature_names)
                    train_weight_multipliers = row_sample_weight_multipliers(train_rows)
                    if positive_mixup_enabled(experiment_cfg):
                        x_synthetic, y_synthetic, synthetic_train_rows, positive_pair_mixup = (
                            build_positive_pair_mixup_augmentation(
                                sampled_zh_train_rows,
                                feature_names,
                                experiment_cfg,
                                int(seed),
                                experiment_name,
                                run_key,
                            )
                        )
                        required_mixup_ratios = [
                            float(value)
                            for value in (
                                experiment_cfg.get("positive_pair_mixup", {}).get(
                                    "require_nonzero_synthetic_for_ratios", []
                                )
                                or []
                            )
                        ]
                        if any(abs(float(ratio) - required) <= 1e-12 for required in required_mixup_ratios):
                            synthetic_count = int(positive_pair_mixup.get("synthetic_row_count", 0))
                            if synthetic_count <= 0:
                                raise ValueError(
                                    f"{experiment_name}.{run_key} requires non-zero positive-pair mixup, "
                                    f"but augmentation was skipped: {positive_pair_mixup.get('skipped_reason')}"
                                )
                        if len(y_synthetic) > 0:
                            x_train = np.vstack([x_train, x_synthetic])
                            y_train = np.concatenate([y_train, y_synthetic])
                            train_weight_multipliers = np.concatenate(
                                [train_weight_multipliers, row_sample_weight_multipliers(synthetic_train_rows)]
                            )
                    scorer_artifact, train_prob = fit_regularized_logistic(
                        x_train,
                        y_train,
                        logistic_cfg,
                        sample_weight_multipliers=train_weight_multipliers,
                    )
                    scorer_artifact.update(
                        {
                            "artifact_type": "logistic_regression_l2",
                            "scoring_backend": backend,
                            "experiment_name": experiment_name,
                            "run_key": run_key,
                            "base_step7_experiment": base_step7_experiment_name,
                            "base_step7_model_path": str(base_model_path.relative_to(ROOT)),
                            "feature_names": feature_names,
                            "feature_mode": str(experiment_cfg.get("semantic_feature_set", "")),
                            "positive_pair_mixup": positive_pair_mixup,
                            "fit_split": "source_train_plus_sampled_zh_train"
                            if run_include_source_train
                            else "sampled_zh_train",
                            "train_probability_stats": {
                                "min": round(float(np.min(train_prob)), 6),
                                "mean": round(float(np.mean(train_prob)), 6),
                                "max": round(float(np.max(train_prob)), 6),
                            },
                        }
                    )
                    valid_prob = apply_logistic_artifact_to_matrix(x_valid, scorer_artifact)
                    zh_test_prob = apply_logistic_artifact_to_matrix(x_zh_test, scorer_artifact)
                    best_iteration = int(scorer_artifact["solver_iterations"])
                    raw_best_iteration = best_iteration
                    trained_iteration_count = best_iteration
                    best_score = {"train_logloss": scorer_artifact["train_logloss"]}
                    top_coefficients = top_logistic_coefficients(
                        feature_names,
                        scorer_artifact["parameter_coefficients"],
                    )
                    top_feature_importance = top_coefficients
                else:
                    residual_cfg = dict(step9_policy["training"]["residual_logistic"])
                    clip_eps = float(residual_cfg.get("base_probability_clip_eps", 1e-6))
                    x_train, y_train = step7.rows_to_matrix(sampled_zh_train_rows, feature_names)
                    sampled_base_prob = np.array(
                        [zh_train_base_prob_by_uid[row["pair_uid"]] for row in sampled_zh_train_rows],
                        dtype=float,
                    )
                    scorer_artifact, train_prob = fit_regularized_logistic(
                        x_train,
                        y_train,
                        residual_cfg,
                        offset=safe_logit(sampled_base_prob, clip_eps),
                        sample_weight_multipliers=row_sample_weight_multipliers(sampled_zh_train_rows),
                    )
                    scorer_artifact.update(
                        {
                            "artifact_type": "residual_logistic",
                            "calibrator_type": "residual_logistic",
                            "scoring_backend": backend,
                            "experiment_name": experiment_name,
                            "run_key": run_key,
                            "base_step7_experiment": base_step7_experiment_name,
                            "base_step7_model_path": str(base_model_path.relative_to(ROOT)),
                            "base_probability_clip_eps": round(float(clip_eps), 12),
                            "feature_names": feature_names,
                            "feature_mode": str(experiment_cfg.get("residual_semantic_feature_set", "")),
                            "fit_split": "sampled_zh_train",
                            "train_base_probability_stats": {
                                "min": round(float(np.min(sampled_base_prob)), 6),
                                "mean": round(float(np.mean(sampled_base_prob)), 6),
                                "max": round(float(np.max(sampled_base_prob)), 6),
                            },
                            "train_probability_stats": {
                                "min": round(float(np.min(train_prob)), 6),
                                "mean": round(float(np.mean(train_prob)), 6),
                                "max": round(float(np.max(train_prob)), 6),
                            },
                        }
                    )
                    valid_prob = apply_logistic_artifact_to_matrix(
                        x_valid,
                        scorer_artifact,
                        base_probabilities=zh_valid_base_prob,
                    )
                    zh_test_prob = apply_logistic_artifact_to_matrix(
                        x_zh_test,
                        scorer_artifact,
                        base_probabilities=zh_test_base_prob,
                    )
                    best_iteration = int(scorer_artifact["solver_iterations"])
                    raw_best_iteration = best_iteration
                    trained_iteration_count = best_iteration
                    best_score = {"train_logloss": scorer_artifact["train_logloss"]}
                    top_coefficients = top_logistic_coefficients(
                        feature_names,
                        scorer_artifact["parameter_coefficients"],
                    )
                    top_feature_importance = top_coefficients

                threshold, threshold_diagnostics = choose_threshold_for_step9(
                    y_valid,
                    valid_prob,
                    step9_policy["training"]["threshold_selection"],
                    int(seed),
                )
                zh_valid_metrics_fixed = step7.evaluate_probabilities(y_valid, valid_prob, fixed_base_threshold)
                zh_test_metrics_adaptive = step7.evaluate_probabilities(y_zh_test, zh_test_prob, threshold)
                zh_test_metrics_fixed = step7.evaluate_probabilities(y_zh_test, zh_test_prob, fixed_base_threshold)

                zh_valid_predictions = step7.prediction_rows(
                    zh_valid_rows, valid_prob, threshold, f"{experiment_name}_{run_key}"
                )
                zh_test_predictions = step7.prediction_rows(
                    zh_test_rows, zh_test_prob, threshold, f"{experiment_name}_{run_key}"
                )
                zh_test_predictions_fixed = step7.prediction_rows(
                    zh_test_rows,
                    zh_test_prob,
                    fixed_base_threshold,
                    f"{experiment_name}_{run_key}_fixed_base_threshold",
                )

                en_test_metrics = None
                en_test_metrics_fixed = None
                en_test_predictions: list[dict] = []
                en_test_predictions_fixed: list[dict] = []
                if source_test_rows:
                    x_en_test, y_en_test = step7.rows_to_matrix(source_test_rows, feature_names)
                    if backend == "legacy_lightgbm_mixed":
                        en_test_prob = model["booster"].predict(x_en_test, num_iteration=model["best_iteration"])
                    elif backend == "residual_logistic":
                        en_test_prob = apply_logistic_artifact_to_matrix(
                            x_en_test,
                            scorer_artifact,
                            base_probabilities=source_test_base_prob,
                        )
                    else:
                        en_test_prob = apply_logistic_artifact_to_matrix(x_en_test, scorer_artifact)
                    en_test_metrics = step7.evaluate_probabilities(y_en_test, en_test_prob, threshold)
                    en_test_metrics_fixed = step7.evaluate_probabilities(y_en_test, en_test_prob, fixed_base_threshold)
                    en_test_predictions = step7.prediction_rows(
                        source_test_rows, en_test_prob, threshold, f"{experiment_name}_{run_key}"
                    )
                    en_test_predictions_fixed = step7.prediction_rows(
                        source_test_rows,
                        en_test_prob,
                        fixed_base_threshold,
                        f"{experiment_name}_{run_key}_fixed_base_threshold",
                    )

                step7.write_csv(
                    output_path(step9_policy["output_templates"]["sampled_target_train"], experiment_name, ratio_key, int(seed)),
                    summarized_rows(sampled_zh_train_rows),
                    sampled_train_fieldnames(),
                )
                step7.write_csv(
                    output_path(step9_policy["output_templates"]["zh_valid_predictions"], experiment_name, ratio_key, int(seed)),
                    zh_valid_predictions,
                    minimal_prediction_fieldnames(),
                )
                step7.write_csv(
                    output_path(step9_policy["output_templates"]["zh_test_predictions"], experiment_name, ratio_key, int(seed)),
                    zh_test_predictions,
                    minimal_prediction_fieldnames(),
                )
                step7.write_csv(
                    output_path(
                        step9_policy["output_templates"]["zh_test_predictions_fixed_base_threshold"],
                        experiment_name,
                        ratio_key,
                        int(seed),
                    ),
                    zh_test_predictions_fixed,
                    minimal_prediction_fieldnames(),
                )
                if en_test_predictions:
                    step7.write_csv(
                        output_path(step9_policy["output_templates"]["en_test_predictions"], experiment_name, ratio_key, int(seed)),
                        en_test_predictions,
                        minimal_prediction_fieldnames(),
                    )
                    step7.write_csv(
                        output_path(
                            step9_policy["output_templates"]["en_test_predictions_fixed_base_threshold"],
                            experiment_name,
                            ratio_key,
                            int(seed),
                        ),
                        en_test_predictions_fixed,
                        minimal_prediction_fieldnames(),
                    )

                if synthetic_train_rows:
                    synthetic_train_only_path = output_path(
                        step9_policy["output_templates"]["synthetic_train_only"],
                        experiment_name,
                        ratio_key,
                        int(seed),
                    )
                    step7.write_csv(
                        synthetic_train_only_path,
                        synthetic_train_rows,
                        synthetic_train_fieldnames(),
                    )

                model_path = None
                if backend == "legacy_lightgbm_mixed":
                    model_path = output_path(step9_policy["output_templates"]["model"], experiment_name, ratio_key, int(seed))
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    model["booster"].save_model(str(model_path), num_iteration=model["best_iteration"])
                else:
                    scorer_artifact_path = output_path(
                        step9_policy["output_templates"]["artifact"],
                        experiment_name,
                        ratio_key,
                        int(seed),
                    )
                    step7.write_json(scorer_artifact_path, scorer_artifact)

                run_artifacts = {
                    "sampled_target_train": str(
                        output_path(
                            step9_policy["output_templates"]["sampled_target_train"],
                            experiment_name,
                            ratio_key,
                            int(seed),
                        ).relative_to(ROOT)
                    ),
                    "zh_valid_predictions": str(
                        output_path(
                            step9_policy["output_templates"]["zh_valid_predictions"],
                            experiment_name,
                            ratio_key,
                            int(seed),
                        ).relative_to(ROOT)
                    ),
                    "zh_test_predictions": str(
                        output_path(
                            step9_policy["output_templates"]["zh_test_predictions"],
                            experiment_name,
                            ratio_key,
                            int(seed),
                        ).relative_to(ROOT)
                    ),
                    "zh_test_predictions_fixed_base_threshold": str(
                        output_path(
                            step9_policy["output_templates"]["zh_test_predictions_fixed_base_threshold"],
                            experiment_name,
                            ratio_key,
                            int(seed),
                        ).relative_to(ROOT)
                    ),
                    "en_test_predictions": None
                    if not en_test_predictions
                    else str(
                        output_path(
                            step9_policy["output_templates"]["en_test_predictions"],
                            experiment_name,
                            ratio_key,
                            int(seed),
                        ).relative_to(ROOT)
                    ),
                    "en_test_predictions_fixed_base_threshold": None
                    if not en_test_predictions_fixed
                    else str(
                        output_path(
                            step9_policy["output_templates"]["en_test_predictions_fixed_base_threshold"],
                            experiment_name,
                            ratio_key,
                            int(seed),
                        ).relative_to(ROOT)
                    ),
                    "model": None if model_path is None else str(model_path.relative_to(ROOT)),
                    "scorer_artifact": None
                    if scorer_artifact_path is None
                    else str(scorer_artifact_path.relative_to(ROOT)),
                    "synthetic_train_only": None
                    if synthetic_train_only_path is None
                    else str(synthetic_train_only_path.relative_to(ROOT)),
                    "base_model": str(base_model_path.relative_to(ROOT)),
                }

                experiment_summary["runs"][run_key] = {
                    "ratio": float(ratio),
                    "ratio_token": ratio_key,
                    "seed": int(seed),
                    "sampling_seed": int(seed),
                    "base_step7_experiment": base_step7_experiment_name,
                    "training_backend": backend,
                    "scoring_backend": backend,
                    "role": str(experiment_cfg.get("role", "")),
                    "include_source_train": bool(run_include_source_train),
                    "actual_sampled_ratio_vs_zh_train": round(len(sampled_zh_train_rows) / len(zh_train_rows), 6),
                    "train_dataset": dataset_summary(train_rows),
                    "training_matrix_dataset": {
                        "row_count": int(len(y_train)) if "y_train" in locals() else len(train_rows),
                        "label_counts": label_count_summary_from_vector(y_train)
                        if "y_train" in locals()
                        else dataset_summary(train_rows)["label_counts"],
                        "real_row_count": int(len(train_rows)),
                        "synthetic_train_only_row_count": int(positive_pair_mixup.get("synthetic_row_count", 0)),
                    },
                    "sampled_zh_train_dataset": dataset_summary(sampled_zh_train_rows),
                    "sampled_zh_train_group_counts": dict(
                        Counter(format_group_key(group_key(row, sampling_group_fields)) for row in sampled_zh_train_rows)
                    ),
                    "sampling_diagnostics": sampling_diagnostics,
                    "sampled_target_seller_overlap_checks": sampled_target_seller_overlap_checks,
                    "zh_valid_dataset": dataset_summary(zh_valid_rows),
                    "zh_test_dataset": dataset_summary(zh_test_rows),
                    "selected_threshold": round(float(threshold), 6),
                    "fixed_base_threshold": round(float(fixed_base_threshold), 6),
                    "threshold_metric": step9_policy["training"]["threshold_selection"]["metric"],
                    "threshold_selection_policy": step9_policy["training"]["threshold_selection"],
                    "threshold_selection_diagnostics": threshold_diagnostics,
                    "low_ratio_guard": low_ratio_guard,
                    "zh_valid_metrics": step7.evaluate_probabilities(y_valid, valid_prob, threshold),
                    "zh_valid_metrics_fixed_base_threshold": zh_valid_metrics_fixed,
                    "zh_test_metrics": zh_test_metrics_adaptive,
                    "zh_test_metrics_fixed_base_threshold": zh_test_metrics_fixed,
                    "en_test_metrics": en_test_metrics,
                    "en_test_metrics_fixed_base_threshold": en_test_metrics_fixed,
                    "masked_positive_rows": masked_positive_rows,
                    "masked_positive_rate_realized": round(float(masked_positive_rate_realized), 6),
                    "semantic_activation_augmentation": semantic_activation_augmentation,
                    "positive_pair_mixup": positive_pair_mixup,
                    "feature_family_gain": feature_family_gain,
                    "training_random_seed": training_random_seed,
                    "best_iteration": best_iteration,
                    "raw_best_iteration": raw_best_iteration,
                    "trained_iteration_count": trained_iteration_count,
                    "best_score": best_score,
                    "collapse_guard": collapse_guard,
                    "small_validation_guard": small_validation_guard,
                    "top_feature_importance": top_feature_importance,
                    "top_logistic_coefficients": top_coefficients,
                    "sampled_target_train_pair_uids": [row["pair_uid"] for row in sampled_zh_train_rows],
                    "artifacts": run_artifacts,
                }

        experiment_summary["aggregate_by_ratio"] = summarize_step9_experiment_runs(experiment_summary)
        summary["experiments"][experiment_name] = experiment_summary

    set_step9_acceptance_checks(summary)

    summary_path = ROOT / step9_policy["output_templates"]["summary"]
    summary = merge_with_existing_summary(summary_path, summary)
    step7.write_json(summary_path, summary)
    print(json.dumps(summary["experiments"], ensure_ascii=False, indent=2))
    print(json.dumps({"summary_merge_mode": summary.get("summary_merge_mode")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
