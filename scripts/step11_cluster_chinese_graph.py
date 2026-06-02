from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step11_clustering_policy.json"
STEP7_SUMMARY_PATH = ROOT / "reports" / "step7_training_summary.json"
STEP7_POLICY_PATH = ROOT / "schema" / "step7_training_policy.json"
STEP9_SUMMARY_PATH = ROOT / "reports" / "step9_few_shot_summary.json"
STEP9_CALIBRATION_SUMMARY_PATH = ROOT / "reports" / "step9_calibration_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score all zh_target_strict candidate seller pairs with a selected synchronized "
            "Step 7, Step 9, or Step 9 calibration scorer, then extract connected-component clusters at "
            "the configured thresholds."
        )
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=POLICY_PATH,
        help="Path to schema/step11_clustering_policy.json.",
    )
    parser.add_argument(
        "--scorer-family",
        choices=("step7", "step9", "step9_calibration", "auto"),
        help=(
            "Scorer family to project into the Chinese graph. If omitted, Step 11 uses the "
            "policy default unless the provided CLI arguments imply a specific family."
        ),
    )
    parser.add_argument(
        "--step7-experiment",
        type=str,
        help=(
            "Optional Step 7 experiment name to use as the pair scorer. "
            "Defaults to scorer_selection.default_step7_experiment_name from "
            "schema/step11_clustering_policy.json."
        ),
    )
    parser.add_argument(
        "--step9-experiment",
        type=str,
        help=(
            "Optional Step 9 experiment name to use as the pair scorer. "
            "Must exist inside reports/step9_few_shot_summary.json."
        ),
    )
    parser.add_argument(
        "--step9-ratio",
        type=float,
        help=(
            "Optional Step 9 few-shot ratio to use for the selected Step 9 experiment, "
            "for example 0.2 or 0.5."
        ),
    )
    parser.add_argument(
        "--step9-seed",
        type=int,
        help="Optional Step 9 random seed to use for the selected Step 9 experiment.",
    )
    parser.add_argument(
        "--step9-calibration-experiment",
        type=str,
        help=(
            "Optional Step 9 calibration experiment name to use as the pair scorer. "
            "Must exist inside reports/step9_calibration_summary.json."
        ),
    )
    parser.add_argument(
        "--threshold",
        action="append",
        dest="thresholds",
        type=float,
        help=(
            "Optional explicit threshold. Repeat to restate the current Step 11 threshold set. "
            "The graph working threshold is resolved from schema/step11_clustering_policy.json "
            "override or the selected scorer summary pairwise threshold; if you want different "
            "sensitivity thresholds, update schema/step11_clustering_policy.json first."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def resolve_policy_path(path_value: str | None, default: Path) -> Path:
    if not path_value:
        return default
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def resolve_current_main_summary_path(
    policy: dict,
    configured_path_value: str | None,
    default_path: Path,
    label: str,
) -> Path:
    resolved_path = resolve_policy_path(configured_path_value, default_path)
    summary_resolution = policy.get("summary_resolution", {}) or {}
    strict_current_only = bool(summary_resolution.get("strict_current_main_summaries_only", True))
    if strict_current_only and resolved_path.resolve() != default_path.resolve():
        raise SystemExit(
            "Step 11 is configured to use only the current main synchronized summary files, but "
            f"{label} resolved to {resolved_path} instead of {default_path}. "
            "Update schema/step11_clustering_policy.json to point back to the main reports/*.json "
            "files rather than an archived snapshot."
        )
    return resolved_path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def append_unique(base: list[str], extra: list[str]) -> list[str]:
    seen = set(base)
    for item in extra:
        if item not in seen:
            base.append(item)
            seen.add(item)
    return base


def round_float(value: float) -> float:
    return round(float(value), 6)


def validate_thresholds(thresholds: list[float]) -> list[float]:
    rounded = sorted({round(float(threshold), 6) for threshold in thresholds})
    invalid = [threshold for threshold in rounded if threshold <= 0.0 or threshold > 1.0]
    if invalid:
        raise SystemExit(f"Step 11 thresholds must satisfy 0 < threshold <= 1. Invalid values: {invalid}")
    return rounded


def threshold_token(threshold: float) -> str:
    return f"{round(float(threshold), 6):.6f}".replace(".", "")


def threshold_field_name(threshold: float) -> str:
    return f"edge_at_threshold_{threshold_token(threshold)}"


def canonical_edge_key(seller_left: str, seller_right: str) -> tuple[str, str]:
    return (seller_left, seller_right) if seller_left <= seller_right else (seller_right, seller_left)


def expand_threshold_field_templates(fields: list[str], thresholds: list[float]) -> list[str]:
    expanded = []
    for field in fields:
        if "{threshold_token}" in field:
            expanded.extend(field.format(threshold_token=threshold_token(threshold)) for threshold in thresholds)
        else:
            expanded.append(field)
    return expanded


def output_path(template: str, **kwargs) -> Path:
    return ROOT / template.format(**kwargs)


def score_distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "min": round_float(np.min(array)),
        "p05": round_float(np.quantile(array, 0.05)),
        "p25": round_float(np.quantile(array, 0.25)),
        "median": round_float(np.median(array)),
        "mean": round_float(np.mean(array)),
        "p75": round_float(np.quantile(array, 0.75)),
        "p95": round_float(np.quantile(array, 0.95)),
        "max": round_float(np.max(array)),
    }


def truncate_text(value: object, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def extract_value_list(items: object, limit: int) -> list[str]:
    values = []
    for item in items or []:
        if isinstance(item, dict):
            value = str(item.get("value", "")).strip()
        else:
            value = str(item).strip()
        if not value:
            continue
        values.append(value)
        if len(values) >= limit:
            break
    return values


def build_contact_preview(profile: dict, per_type_limit: int = 2) -> str:
    contact_signals = profile.get("contact_signals") or {}
    preview_parts = []
    for contact_type in ("email", "telegram", "wickr", "wechat", "qq", "phone"):
        values = extract_value_list(contact_signals.get(contact_type), per_type_limit)
        if values:
            preview_parts.append(f"{contact_type}:{' | '.join(values)}")
    return truncate_text(" || ".join(preview_parts), 180)


def seller_preview_fields(profile: dict) -> dict:
    return {
        "source_dataset": str(profile.get("source_dataset", "") or ""),
        "source_market_raw": str(profile.get("source_market_raw", "") or ""),
        "source_seller_raw": str(profile.get("source_seller_raw", "") or ""),
        "alias_normalized": str(profile.get("alias_normalized", "") or ""),
        "item_count": int(profile.get("item_count", 0) or 0),
        "unique_title_count": int(profile.get("unique_title_count", 0) or 0),
        "unique_description_snippet_count": int(profile.get("unique_description_snippet_count", 0) or 0),
        "unique_category_count": int(profile.get("unique_category_count", 0) or 0),
        "contact_type_count": int(profile.get("contact_type_count", 0) or 0),
        "contact_token_count_total": int(profile.get("contact_token_count_total", 0) or 0),
        "top_category_preview": truncate_text(" || ".join(extract_value_list(profile.get("top_categories"), 3)), 180),
        "signature_title_preview": truncate_text(" || ".join(extract_value_list(profile.get("signature_titles"), 3)), 220),
        "signature_description_preview": truncate_text(
            " || ".join(extract_value_list(profile.get("signature_description_segments"), 2)),
            260,
        ),
        "contact_preview": build_contact_preview(profile),
    }


def seller_display_name(profile: dict) -> str:
    for key in ("alias_normalized", "source_seller_raw", "seller_uid"):
        value = str(profile.get(key, "") or "").strip()
        if value:
            return value
    return ""


def load_seller_index(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in load_jsonl(path):
        seller_uid = str(row.get("seller_uid", "") or "")
        if not seller_uid:
            raise ValueError(f"Seller profile without seller_uid in {path}")
        if seller_uid in index:
            raise ValueError(f"Duplicate seller_uid in {path}: {seller_uid}")
        index[seller_uid] = row
    return index


def ensure_unique_pair_uids(pair_rows: list[dict]) -> None:
    counts = Counter(row["pair_uid"] for row in pair_rows)
    duplicates = sorted(pair_uid for pair_uid, count in counts.items() if count > 1)
    if duplicates:
        preview = duplicates[:10]
        raise SystemExit(
            "Step 11 requires unique pair_uid values in reports/step7_pair_features.zh_target_strict.csv. "
            f"Found duplicates: {preview} (total duplicate ids: {len(duplicates)})"
        )


def ensure_pair_sellers_exist(pair_rows: list[dict], seller_index: dict[str, dict]) -> None:
    missing = set()
    for row in pair_rows:
        for key in ("seller_uid_left", "seller_uid_right"):
            seller_uid = row[key]
            if seller_uid not in seller_index:
                missing.add(seller_uid)
    if missing:
        preview = sorted(missing)[:10]
        raise SystemExit(
            "Step 11 seller-profile join failed. Missing seller_uids in zh_target_strict profiles: "
            f"{preview} (total missing: {len(missing)})"
        )


def ensure_core_transfer_eligible(pair_rows: list[dict]) -> list[dict]:
    valid_rows = []
    invalid = []
    for row in pair_rows:
        if str(row.get("core_transfer_eligible", "")).strip() != "1":
            invalid.append(row["pair_uid"])
        else:
            valid_rows.append(row)
    if invalid:
        preview = invalid[:10]
        print(
            "WARNING: Step 11 must score only core-transfer-eligible zh_target_strict pairs from the fixed Step 7 candidate universe. "
            f"Found non-eligible pair_uids: {preview} (total invalid: {len(invalid)}). Skipping them."
        )
    return valid_rows


def resolve_scorer_selection(policy: dict) -> dict:
    scorer_selection = policy.get("scorer_selection")
    if scorer_selection:
        return {
            "default_scorer_family": str(scorer_selection.get("default_scorer_family", "step7") or "step7"),
            "default_step7_experiment_name": scorer_selection.get("default_step7_experiment_name")
            or scorer_selection.get("default_experiment_name"),
            "default_step9_experiment_name": scorer_selection.get("default_step9_experiment_name"),
            "default_step9_calibration_experiment_name": scorer_selection.get(
                "default_step9_calibration_experiment_name"
            ),
            "default_step9_ratio": scorer_selection.get("default_step9_ratio"),
            "default_step9_seed": scorer_selection.get("default_step9_seed"),
            "pairwise_primary_threshold_source_template": scorer_selection.get(
                "pairwise_primary_threshold_source_template"
            )
            or scorer_selection.get("primary_threshold_source_template"),
            "graph_primary_threshold_overrides": scorer_selection.get("graph_primary_threshold_overrides", {}),
            "model_path_source": scorer_selection.get("model_path_source"),
            "sensitivity_thresholds": scorer_selection.get("sensitivity_thresholds", []),
            "sensitivity_threshold_resolution": scorer_selection.get("sensitivity_threshold_resolution", {}),
            "feature_source": scorer_selection.get("feature_source"),
            "scoring_backend": scorer_selection.get("scoring_backend"),
            "dynamic_mainline_candidates": scorer_selection.get("dynamic_mainline_candidates", {}),
        }

    legacy_fixed_scorer = policy.get("fixed_scorer")
    if legacy_fixed_scorer:
        return {
            "default_scorer_family": "step7",
            "default_step7_experiment_name": legacy_fixed_scorer.get("experiment_name"),
            "default_step9_experiment_name": None,
            "default_step9_calibration_experiment_name": None,
            "default_step9_ratio": None,
            "default_step9_seed": None,
            "pairwise_primary_threshold_source_template": legacy_fixed_scorer.get("primary_threshold_source"),
            "graph_primary_threshold_overrides": {},
            "model_path_source": legacy_fixed_scorer.get("model_path_source"),
            "sensitivity_thresholds": legacy_fixed_scorer.get("sensitivity_thresholds", []),
            "sensitivity_threshold_resolution": {},
            "feature_source": legacy_fixed_scorer.get("feature_source"),
            "scoring_backend": legacy_fixed_scorer.get("scoring_backend"),
            "dynamic_mainline_candidates": {},
        }

    raise SystemExit(
        "Step 11 policy must declare scorer_selection (or legacy fixed_scorer) inside "
        "schema/step11_clustering_policy.json."
    )


def normalize_step9_ratio(ratio: float | int | str | None) -> float | None:
    if ratio is None or ratio == "":
        return None
    normalized = float(ratio)
    if normalized <= 0.0 or normalized > 1.0:
        raise SystemExit(f"Step 11 Step 9 ratios must satisfy 0 < ratio <= 1. Received: {ratio}")
    return normalized


def get_nested_value(payload: dict, path: str) -> object:
    current: object = payload
    for token in str(path or "").split("."):
        if not token:
            continue
        if not isinstance(current, dict):
            return None
        current = current.get(token)
    return current


def to_candidate_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def dynamic_family_config(dynamic_cfg: dict, family: str) -> dict:
    return (dynamic_cfg.get("families", {}) or {}).get(family, {}) or {}


def experiment_allowed_by_dynamic_config(experiment_name: str, family_cfg: dict) -> bool:
    name = str(experiment_name or "")
    exclude_names = {str(item) for item in family_cfg.get("exclude_experiment_names", []) or []}
    exclude_prefixes = [str(item) for item in family_cfg.get("exclude_experiment_prefixes", []) or []]
    include_prefixes = [str(item) for item in family_cfg.get("include_experiment_prefixes", []) or []]
    if name in exclude_names:
        return False
    if any(name.startswith(prefix) for prefix in exclude_prefixes):
        return False
    if include_prefixes and not any(name.startswith(prefix) for prefix in include_prefixes):
        return False
    if bool(family_cfg.get("clean_only", False)) and name.startswith("identifier_augmented_"):
        return False
    return True


def build_candidate_record(
    *,
    scorer_family: str,
    scorer_token: str,
    source_experiment_name: str,
    source_run_key: str | None,
    source_ratio: float | None,
    source_ratio_token: str | None,
    source_seed: int | None,
    primary_metric_path: str,
    primary_metric: float,
    secondary_metric_paths: list[str],
    payload: dict,
    extra: dict | None = None,
) -> dict:
    secondary_metrics = {}
    for path in secondary_metric_paths:
        metric = to_candidate_float(get_nested_value(payload, path))
        if metric is not None:
            secondary_metrics[path] = round_float(metric)
    candidate = {
        "scorer_family": scorer_family,
        "scorer_token": scorer_token,
        "source_experiment_name": source_experiment_name,
        "source_run_key": source_run_key,
        "source_ratio": source_ratio,
        "source_ratio_token": source_ratio_token,
        "source_seed": source_seed,
        "primary_metric_path": primary_metric_path,
        "primary_metric": round_float(primary_metric),
        "secondary_metrics": secondary_metrics,
        "best_iteration": int(to_candidate_float(get_nested_value(payload, "best_iteration")) or 0),
        "trained_iteration_count": int(to_candidate_float(get_nested_value(payload, "trained_iteration_count")) or 0),
    }
    if extra:
        candidate.update(extra)
    return candidate


def sort_dynamic_candidates(candidates: list[dict], secondary_metric_paths: list[str]) -> list[dict]:
    def sort_key(candidate: dict) -> tuple:
        secondary = candidate.get("secondary_metrics", {}) or {}
        secondary_values = tuple(float(secondary.get(path, float("-inf"))) for path in secondary_metric_paths)
        ratio = candidate.get("source_ratio")
        ratio_sort = -float(ratio) if ratio not in ("", None) else float("-inf")
        return (
            -float(candidate.get("primary_metric", float("-inf"))),
            *tuple(-value for value in secondary_values),
            ratio_sort,
            str(candidate.get("scorer_token", "")),
        )

    return sorted(candidates, key=sort_key)


def ratio_aggregate_metric(
    experiment_summary: dict,
    ratio_token: str,
    metric_path: str,
) -> float | None:
    aggregate_by_ratio = (experiment_summary.get("aggregate_by_ratio", {}) or {}).get(ratio_token, {}) or {}
    return to_candidate_float(get_nested_value(aggregate_by_ratio, metric_path))


def resolve_step9_run_backend(experiment_summary: dict, run_summary: dict) -> str:
    for payload in (run_summary, experiment_summary):
        for key in ("scoring_backend", "training_backend", "backend"):
            value = str(payload.get(key, "") or "").strip()
            if value:
                return value
    return "legacy_lightgbm_mixed"


def resolve_step9_backend_iteration_filter(family_cfg: dict, backend: str) -> dict:
    fallback = {
        "min_best_iteration": int(family_cfg.get("min_best_iteration", 0) or 0),
        "min_trained_iteration_count": int(family_cfg.get("min_trained_iteration_count", 0) or 0),
    }
    backend_filters = family_cfg.get("backend_iteration_filters", {}) or {}
    backend_filter = backend_filters.get(backend) or backend_filters.get("*") or {}
    return {
        "min_best_iteration": int(backend_filter.get("min_best_iteration", fallback["min_best_iteration"]) or 0),
        "min_trained_iteration_count": int(
            backend_filter.get("min_trained_iteration_count", fallback["min_trained_iteration_count"]) or 0
        ),
        "iteration_role": str(backend_filter.get("iteration_role", "model_iteration") or "model_iteration"),
    }


def sort_step9_dynamic_candidates(
    candidates: list[dict],
    family_cfg: dict,
    secondary_metric_paths: list[str],
) -> list[dict]:
    prefer_ratio_aggregate = bool(family_cfg.get("prefer_ratio_aggregate_before_run_metric", False))
    ratio_aggregate_secondary_metric_paths = [
        str(path)
        for path in family_cfg.get("ratio_aggregate_secondary_metric_paths", ["zh_test_metrics.roc_auc.mean"]) or []
    ]

    def sort_key(candidate: dict) -> tuple:
        secondary = candidate.get("secondary_metrics", {}) or {}
        secondary_values = tuple(float(secondary.get(path, float("-inf"))) for path in secondary_metric_paths)
        ratio_secondary = candidate.get("ratio_aggregate_secondary_metrics", {}) or {}
        ratio_secondary_values = tuple(
            float(ratio_secondary.get(path, float("-inf"))) for path in ratio_aggregate_secondary_metric_paths
        )
        ratio = candidate.get("source_ratio")
        ratio_sort = -float(ratio) if ratio not in ("", None) else float("-inf")
        suspicious_perfect_small_test = int(bool(candidate.get("suspicious_perfect_small_test", False)))
        ratio_primary = float(candidate.get("ratio_aggregate_primary_metric_mean", float("-inf")))
        if not prefer_ratio_aggregate:
            ratio_primary = float("-inf")
            ratio_secondary_values = tuple(float("-inf") for _ in ratio_aggregate_secondary_metric_paths)
        return (
            suspicious_perfect_small_test,
            -ratio_primary,
            *tuple(-value for value in ratio_secondary_values),
            -float(candidate.get("primary_metric", float("-inf"))),
            *tuple(-value for value in secondary_values),
            ratio_sort,
            str(candidate.get("scorer_token", "")),
        )

    return sorted(candidates, key=sort_key)


def build_step7_dynamic_candidates(policy: dict, family_cfg: dict) -> list[dict]:
    baseline_reference = policy.get("baseline_reference", {})
    input_paths = policy.get("input_paths", {})
    step7_summary_path = resolve_current_main_summary_path(
        policy,
        baseline_reference.get("step7_summary") or input_paths.get("step7_summary"),
        STEP7_SUMMARY_PATH,
        "Step 7 summary",
    )
    if not step7_summary_path.exists():
        return []
    summary = step7.load_json(step7_summary_path)
    primary_metric_path = str(
        family_cfg.get("primary_metric_path", "zh_zero_shot_test_metrics.balanced_accuracy")
    )
    secondary_metric_paths = [
        str(path) for path in family_cfg.get("secondary_metric_paths", ["zh_zero_shot_test_metrics.roc_auc"]) or []
    ]
    min_best_iteration = int(family_cfg.get("min_best_iteration", 0) or 0)
    min_trained_iteration_count = int(family_cfg.get("min_trained_iteration_count", 0) or 0)
    candidates = []
    for experiment_name, experiment_summary in (summary.get("experiments", {}) or {}).items():
        if not experiment_allowed_by_dynamic_config(experiment_name, family_cfg):
            continue
        primary_metric = to_candidate_float(get_nested_value(experiment_summary, primary_metric_path))
        if primary_metric is None:
            continue
        best_iteration = int(to_candidate_float(get_nested_value(experiment_summary, "best_iteration")) or 0)
        trained_iteration_count = int(
            to_candidate_float(get_nested_value(experiment_summary, "trained_iteration_count")) or 0
        )
        if best_iteration < min_best_iteration or trained_iteration_count < min_trained_iteration_count:
            continue
        candidates.append(
            build_candidate_record(
                scorer_family="step7",
                scorer_token=experiment_name,
                source_experiment_name=experiment_name,
                source_run_key=None,
                source_ratio=None,
                source_ratio_token=None,
                source_seed=None,
                primary_metric_path=primary_metric_path,
                primary_metric=primary_metric,
                secondary_metric_paths=secondary_metric_paths,
                payload=experiment_summary,
                extra={
                    "selected_threshold": round_float(
                        float(experiment_summary.get("selected_threshold", 0.0) or 0.0)
                    ),
                },
            )
        )
    return sort_dynamic_candidates(candidates, secondary_metric_paths)


def build_step9_dynamic_candidates(policy: dict, family_cfg: dict) -> list[dict]:
    baseline_reference = policy.get("baseline_reference", {})
    input_paths = policy.get("input_paths", {})
    step9_summary_path = resolve_current_main_summary_path(
        policy,
        baseline_reference.get("step9_summary") or input_paths.get("step9_summary"),
        STEP9_SUMMARY_PATH,
        "Step 9 few-shot summary",
    )
    if not step9_summary_path.exists():
        return []
    summary = step7.load_json(step9_summary_path)
    primary_metric_path = str(family_cfg.get("primary_metric_path", "zh_test_metrics.balanced_accuracy"))
    secondary_metric_paths = [
        str(path) for path in family_cfg.get("secondary_metric_paths", ["zh_test_metrics.roc_auc", "best_iteration"])
        or []
    ]
    ratio_aggregate_primary_metric_path = str(
        family_cfg.get("ratio_aggregate_primary_metric_path", "zh_test_metrics.balanced_accuracy.mean")
    )
    perfect_small_test_row_count_max = int(family_cfg.get("perfect_small_test_row_count_max", 0) or 0)
    perfect_primary_metric_min = float(family_cfg.get("perfect_primary_metric_min", 1.0) or 1.0)
    perfect_secondary_metric_path = str(
        family_cfg.get("perfect_secondary_metric_path", "zh_test_metrics.roc_auc")
    )
    perfect_secondary_metric_min = float(family_cfg.get("perfect_secondary_metric_min", 1.0) or 1.0)
    candidates = []
    for experiment_name, experiment_summary in (summary.get("experiments", {}) or {}).items():
        if not experiment_allowed_by_dynamic_config(experiment_name, family_cfg):
            continue
        for run_key, run_summary in (experiment_summary.get("runs", {}) or {}).items():
            primary_metric = to_candidate_float(get_nested_value(run_summary, primary_metric_path))
            if primary_metric is None:
                continue
            backend = resolve_step9_run_backend(experiment_summary, run_summary)
            iteration_filter = resolve_step9_backend_iteration_filter(family_cfg, backend)
            best_iteration = int(to_candidate_float(get_nested_value(run_summary, "best_iteration")) or 0)
            trained_iteration_count = int(
                to_candidate_float(get_nested_value(run_summary, "trained_iteration_count")) or 0
            )
            if (
                best_iteration < iteration_filter["min_best_iteration"]
                or trained_iteration_count < iteration_filter["min_trained_iteration_count"]
            ):
                continue
            ratio = normalize_step9_ratio(run_summary.get("ratio"))
            seed = int(run_summary.get("seed", 0) or 0)
            ratio_token = str(run_summary.get("ratio_token", "") or "").strip() or (
                f"{int(round(float(ratio or 0.0) * 100))}pct" if ratio is not None else ""
            )
            ratio_aggregate_primary_metric_mean = ratio_aggregate_metric(
                experiment_summary,
                ratio_token,
                ratio_aggregate_primary_metric_path,
            )
            ratio_aggregate_secondary_metric_paths = [
                str(path)
                for path in family_cfg.get("ratio_aggregate_secondary_metric_paths", ["zh_test_metrics.roc_auc.mean"])
                or []
            ]
            ratio_aggregate_secondary_metrics = {}
            for metric_path in ratio_aggregate_secondary_metric_paths:
                metric = ratio_aggregate_metric(experiment_summary, ratio_token, metric_path)
                if metric is not None:
                    ratio_aggregate_secondary_metrics[metric_path] = round_float(metric)
            zh_test_row_count = int(to_candidate_float(get_nested_value(run_summary, "zh_test_dataset.row_count")) or 0)
            perfect_secondary_metric = to_candidate_float(get_nested_value(run_summary, perfect_secondary_metric_path))
            suspicious_perfect_small_test = (
                zh_test_row_count > 0
                and perfect_small_test_row_count_max > 0
                and zh_test_row_count <= perfect_small_test_row_count_max
                and float(primary_metric) >= perfect_primary_metric_min
                and (perfect_secondary_metric is not None and float(perfect_secondary_metric) >= perfect_secondary_metric_min)
            )
            candidates.append(
                build_candidate_record(
                    scorer_family="step9",
                    scorer_token=f"{experiment_name}_ratio_{ratio_token}_seed_{seed}",
                    source_experiment_name=experiment_name,
                    source_run_key=run_key,
                    source_ratio=ratio,
                    source_ratio_token=ratio_token,
                    source_seed=seed,
                    primary_metric_path=primary_metric_path,
                    primary_metric=primary_metric,
                    secondary_metric_paths=secondary_metric_paths,
                    payload=run_summary,
                    extra={
                        "selected_threshold": round_float(float(run_summary.get("selected_threshold", 0.0) or 0.0)),
                        "low_ratio_guard_triggered": bool(
                            ((run_summary.get("low_ratio_guard") or {}).get("triggered", False))
                        ),
                        "ratio_aggregate_primary_metric_mean": None
                        if ratio_aggregate_primary_metric_mean is None
                        else round_float(float(ratio_aggregate_primary_metric_mean)),
                        "ratio_aggregate_secondary_metrics": ratio_aggregate_secondary_metrics,
                        "zh_test_row_count": zh_test_row_count,
                        "suspicious_perfect_small_test": bool(suspicious_perfect_small_test),
                        "training_backend": str(run_summary.get("training_backend") or backend),
                        "scoring_backend": str(run_summary.get("scoring_backend") or backend),
                        "iteration_filter": iteration_filter,
                    },
                )
            )
    return sort_step9_dynamic_candidates(candidates, family_cfg, secondary_metric_paths)


def build_step9_calibration_dynamic_candidates(policy: dict, family_cfg: dict) -> list[dict]:
    baseline_reference = policy.get("baseline_reference", {})
    input_paths = policy.get("input_paths", {})
    calibration_summary_path = resolve_current_main_summary_path(
        policy,
        baseline_reference.get("step9_calibration_summary") or input_paths.get("step9_calibration_summary"),
        STEP9_CALIBRATION_SUMMARY_PATH,
        "Step 9 calibration summary",
    )
    if not calibration_summary_path.exists():
        return []
    summary = step7.load_json(calibration_summary_path)
    primary_metric_path = str(family_cfg.get("primary_metric_path", "zh_test_metrics.balanced_accuracy"))
    secondary_metric_paths = [
        str(path) for path in family_cfg.get("secondary_metric_paths", ["zh_test_metrics.roc_auc"]) or []
    ]
    max_abs_parameter_scale = family_cfg.get("max_abs_parameter_scale")
    candidates = []
    for experiment_name, experiment_summary in (summary.get("experiments", {}) or {}).items():
        if not experiment_allowed_by_dynamic_config(experiment_name, family_cfg):
            continue
        primary_metric = to_candidate_float(get_nested_value(experiment_summary, primary_metric_path))
        if primary_metric is None:
            continue
        parameter_scale = to_candidate_float(
            get_nested_value(experiment_summary, "calibrator_diagnostics.parameter_scale")
        )
        if max_abs_parameter_scale not in ("", None):
            limit = abs(float(max_abs_parameter_scale))
            if parameter_scale is None or abs(float(parameter_scale)) > limit:
                continue
        candidates.append(
            build_candidate_record(
                scorer_family="step9_calibration",
                scorer_token=experiment_name,
                source_experiment_name=experiment_name,
                source_run_key=None,
                source_ratio=None,
                source_ratio_token=None,
                source_seed=None,
                primary_metric_path=primary_metric_path,
                primary_metric=primary_metric,
                secondary_metric_paths=secondary_metric_paths,
                payload=experiment_summary,
                extra={
                    "selected_threshold": round_float(
                        float(
                            experiment_summary.get("primary_threshold")
                            or experiment_summary.get("selected_threshold")
                            or 0.0
                        )
                    ),
                    "parameter_scale": round_float(float(parameter_scale or 0.0)),
                    "base_step7_experiment": str(experiment_summary.get("base_step7_experiment", "") or ""),
                },
            )
        )
    return sort_dynamic_candidates(candidates, secondary_metric_paths)


def build_dynamic_mainline_candidates(policy: dict, scorer_selection: dict) -> dict:
    dynamic_cfg = scorer_selection.get("dynamic_mainline_candidates", {}) or {}
    snapshot = {
        "enabled": bool(dynamic_cfg.get("enabled", False)),
        "default_resolution_mode": str(dynamic_cfg.get("default_resolution_mode", "family_best") or "family_best"),
        "max_candidates_per_family": int(dynamic_cfg.get("max_candidates_per_family", 3) or 3),
        "overall_family_priority": [str(item) for item in dynamic_cfg.get("overall_family_priority", []) or []],
        "families": {},
        "best_overall": None,
    }
    if not snapshot["enabled"]:
        return snapshot

    builders = {
        "step7": build_step7_dynamic_candidates,
        "step9": build_step9_dynamic_candidates,
        "step9_calibration": build_step9_calibration_dynamic_candidates,
    }
    max_candidates = max(int(snapshot["max_candidates_per_family"]), 1)
    for family, builder in builders.items():
        family_cfg = dynamic_family_config(dynamic_cfg, family)
        candidates = builder(policy, family_cfg)
        snapshot["families"][family] = {
            "config": family_cfg,
            "candidate_count": len(candidates),
            "candidates": candidates[:max_candidates],
        }

    all_candidates = []
    for family, payload in snapshot["families"].items():
        for candidate in payload.get("candidates", []):
            all_candidates.append((family, candidate))
    if not all_candidates:
        return snapshot

    priority = snapshot["overall_family_priority"] or ["step9", "step9_calibration", "step7"]
    family_priority = {family: idx for idx, family in enumerate(priority)}
    ranked = sorted(
        all_candidates,
        key=lambda item: (
            -float(item[1].get("primary_metric", float("-inf"))),
            family_priority.get(item[0], len(family_priority)),
            str(item[1].get("scorer_token", "")),
        ),
    )
    snapshot["best_overall"] = ranked[0][1]
    return snapshot


def select_dynamic_candidate(
    dynamic_snapshot: dict,
    family: str,
    *,
    experiment_name: str | None = None,
    ratio: float | None = None,
    seed: int | None = None,
) -> tuple[dict | None, int | None]:
    family_payload = (dynamic_snapshot.get("families", {}) or {}).get(family, {}) or {}
    candidates = list(family_payload.get("candidates", []) or [])
    if not candidates:
        return None, None
    filtered = []
    for index, candidate in enumerate(candidates, start=1):
        if experiment_name and candidate.get("source_experiment_name") != experiment_name:
            continue
        candidate_ratio = candidate.get("source_ratio")
        if ratio is not None:
            if candidate_ratio is None or abs(float(candidate_ratio) - float(ratio)) > 1e-9:
                continue
        if seed is not None and int(candidate.get("source_seed") or 0) != int(seed):
            continue
        filtered.append((index, candidate))
    if not filtered:
        return None, None
    rank, candidate = filtered[0]
    return candidate, rank


def build_dynamic_candidates_for_family(policy: dict, family: str) -> list[dict]:
    scorer_selection = resolve_scorer_selection(policy)
    dynamic_cfg = scorer_selection.get("dynamic_mainline_candidates", {}) or {}
    family_cfg = dynamic_family_config(dynamic_cfg, family)
    builders = {
        "step7": build_step7_dynamic_candidates,
        "step9": build_step9_dynamic_candidates,
        "step9_calibration": build_step9_calibration_dynamic_candidates,
    }
    builder = builders.get(family)
    if builder is None:
        return []
    return builder(policy, family_cfg)


def select_fresh_dynamic_candidate(
    policy: dict,
    family: str,
    *,
    experiment_name: str | None = None,
    ratio: float | None = None,
    seed: int | None = None,
) -> tuple[dict | None, int | None]:
    candidates = build_dynamic_candidates_for_family(policy, family)
    if not candidates:
        return None, None
    snapshot = {
        "families": {
            family: {
                "candidates": candidates,
            }
        }
    }
    return select_dynamic_candidate(
        snapshot,
        family,
        experiment_name=experiment_name,
        ratio=ratio,
        seed=seed,
    )


def detect_requested_scorer_family(
    args: argparse.Namespace,
    scorer_selection: dict,
    dynamic_snapshot: dict | None = None,
) -> str:
    has_step7_request = bool(args.step7_experiment)
    has_step9_request = any(
        value is not None
        for value in (
            args.step9_experiment,
            args.step9_ratio,
            args.step9_seed,
        )
    )
    has_step9_calibration_request = bool(args.step9_calibration_experiment)
    request_count = sum(int(flag) for flag in (has_step7_request, has_step9_request, has_step9_calibration_request))
    if request_count > 1:
        raise SystemExit(
            "Step 11 scorer selection is ambiguous. Do not mix Step 7, Step 9, and Step 9 calibration "
            "selector arguments in the same command."
        )
    if args.scorer_family:
        if args.scorer_family == "step7" and (has_step9_request or has_step9_calibration_request):
            raise SystemExit(
                "Step 11 received Step 9 selector arguments together with --scorer-family step7. "
                "Remove the Step 9 arguments or switch --scorer-family to the matching family."
            )
        if args.scorer_family == "step9" and has_step7_request:
            raise SystemExit(
                "Step 11 received --step7-experiment together with --scorer-family step9. "
                "Remove --step7-experiment or switch --scorer-family to step7."
            )
        if args.scorer_family == "step9" and has_step9_calibration_request:
            raise SystemExit(
                "Step 11 received --step9-calibration-experiment together with --scorer-family step9. "
                "Remove the calibration selector or switch --scorer-family to step9_calibration."
            )
        if args.scorer_family == "step9_calibration" and (has_step7_request or has_step9_request):
            raise SystemExit(
                "Step 11 received Step 7 or Step 9 few-shot selectors together with --scorer-family "
                "step9_calibration. Remove the conflicting selectors or switch to the matching family."
            )
        if args.scorer_family == "auto":
            best_overall = (dynamic_snapshot or {}).get("best_overall")
            if best_overall:
                return str(best_overall["scorer_family"])
            raise SystemExit(
                "Step 11 received --scorer-family auto, but no dynamic mainline candidates were available. "
                "Pass an explicit family or enable scorer_selection.dynamic_mainline_candidates."
            )
        return str(args.scorer_family)
    if has_step7_request:
        return "step7"
    if has_step9_request:
        return "step9"
    if has_step9_calibration_request:
        return "step9_calibration"
    default_family = str(scorer_selection.get("default_scorer_family", "step7") or "step7")
    if default_family == "auto":
        best_overall = (dynamic_snapshot or {}).get("best_overall")
        if best_overall:
            return str(best_overall["scorer_family"])
        raise SystemExit(
            "Step 11 policy sets scorer_selection.default_scorer_family=auto, but no dynamic "
            "mainline candidates were available."
        )
    return default_family


def resolve_step7_scorer_reference(
    policy: dict,
    requested_experiment_name: str | None,
    dynamic_snapshot: dict | None = None,
) -> dict:
    baseline_reference = policy.get("baseline_reference", {})
    input_paths = policy.get("input_paths", {})
    step7_summary_path = resolve_current_main_summary_path(
        policy,
        baseline_reference.get("step7_summary") or input_paths.get("step7_summary"),
        STEP7_SUMMARY_PATH,
        "Step 7 summary",
    )
    step7_policy_path = resolve_policy_path(
        baseline_reference.get("step7_policy") or input_paths.get("step7_policy"),
        STEP7_POLICY_PATH,
    )
    if not step7_summary_path.exists():
        raise SystemExit(
            "Step 11 could not find the required Step 7 training summary at "
            f"{step7_summary_path}. Sync reports/step7_training_summary.json or update "
            "schema/step11_clustering_policy.json."
        )
    if not step7_policy_path.exists():
        raise SystemExit(
            "Step 11 could not find the required Step 7 training policy at "
            f"{step7_policy_path}. Sync schema/step7_training_policy.json or update "
            "schema/step11_clustering_policy.json."
        )

    scorer_selection = resolve_scorer_selection(policy)
    default_experiment_name = str(scorer_selection.get("default_step7_experiment_name", "") or "").strip()
    dynamic_candidate = None
    dynamic_candidate_rank = None
    if not requested_experiment_name and dynamic_snapshot:
        dynamic_candidate, dynamic_candidate_rank = select_fresh_dynamic_candidate(policy, "step7")
    experiment_name = str(
        requested_experiment_name
        or (dynamic_candidate or {}).get("source_experiment_name")
        or default_experiment_name
    ).strip()
    if not experiment_name:
        raise SystemExit(
            "Step 11 could not resolve a Step 7 scorer experiment name. Pass --step7-experiment "
            "or set scorer_selection.default_step7_experiment_name in schema/step11_clustering_policy.json."
        )

    step7_policy = step7.load_json(step7_policy_path)
    available_policy_experiments = sorted(step7_policy.get("experiments", {}).keys())
    if experiment_name not in step7_policy.get("experiments", {}):
        raise SystemExit(
            "Step 11 could not resolve the requested Step 7 experiment "
            f"{experiment_name!r} inside {step7_policy_path}. Available experiments: "
            f"{available_policy_experiments}"
        )

    step7_summary = step7.load_json(step7_summary_path)
    experiment_summary = step7_summary.get("experiments", {}).get(experiment_name)
    if experiment_summary is None:
        available = sorted(step7_summary.get("experiments", {}).keys())
        raise SystemExit(
            "Step 11 could not resolve the required Step 7 experiment summary "
            f"for {experiment_name!r} inside {step7_summary_path}. Available experiments: {available}"
        )

    selected_threshold = experiment_summary.get("selected_threshold")
    if selected_threshold is None:
        raise SystemExit(
            "Step 11 could not resolve the primary threshold from the Step 7 summary for "
            f"{experiment_name!r} inside {step7_summary_path}."
        )

    model_template = step7_policy.get("output_templates", {}).get("model")
    if not model_template:
        raise SystemExit(
            "Step 11 could not resolve the Step 7 model output template from "
            f"{step7_policy_path}."
        )
    model_path = ROOT / model_template.format(experiment_name=experiment_name)
    if not model_path.exists():
        raise SystemExit(
            "Step 11 resolved the current Step 7 model path to "
            f"{model_path}, but that file does not exist."
        )
    return {
        "scorer_family": "step7",
        "scorer_token": experiment_name,
        "source_experiment_name": experiment_name,
        "source_run_key": None,
        "source_ratio": None,
        "source_ratio_token": None,
        "source_seed": None,
        "summary_path": step7_summary_path,
        "policy_path": step7_policy_path,
        "model_path": model_path,
        "scoring_backend": "lightgbm",
        "primary_threshold": round_float(float(selected_threshold)),
        "selection_mode": "dynamic_family_best" if dynamic_candidate else ("explicit" if requested_experiment_name else "policy_default"),
        "dynamic_candidate_rank": dynamic_candidate_rank,
        "dynamic_candidate": dynamic_candidate,
    }


def resolve_step9_scorer_reference(
    policy: dict,
    requested_experiment_name: str | None,
    requested_ratio: float | None,
    requested_seed: int | None,
    dynamic_snapshot: dict | None = None,
) -> dict:
    baseline_reference = policy.get("baseline_reference", {})
    input_paths = policy.get("input_paths", {})
    step9_summary_path = resolve_current_main_summary_path(
        policy,
        baseline_reference.get("step9_summary") or input_paths.get("step9_summary"),
        STEP9_SUMMARY_PATH,
        "Step 9 few-shot summary",
    )
    if not step9_summary_path.exists():
        raise SystemExit(
            "Step 11 could not find the required Step 9 summary at "
            f"{step9_summary_path}. Sync reports/step9_few_shot_summary.json or update "
            "schema/step11_clustering_policy.json."
        )

    scorer_selection = resolve_scorer_selection(policy)
    default_experiment_name = str(scorer_selection.get("default_step9_experiment_name", "") or "").strip()
    normalized_requested_ratio = normalize_step9_ratio(requested_ratio)
    dynamic_candidate = None
    dynamic_candidate_rank = None
    if dynamic_snapshot:
        dynamic_candidate, dynamic_candidate_rank = select_fresh_dynamic_candidate(
            policy,
            "step9",
            experiment_name=requested_experiment_name,
            ratio=normalized_requested_ratio,
            seed=requested_seed,
        )
    experiment_name = str(
        requested_experiment_name
        or (dynamic_candidate or {}).get("source_experiment_name")
        or default_experiment_name
    ).strip()
    if not experiment_name:
        raise SystemExit(
            "Step 11 could not resolve a Step 9 scorer experiment name. Pass --step9-experiment "
            "or set scorer_selection.default_step9_experiment_name in schema/step11_clustering_policy.json."
        )

    summary = step7.load_json(step9_summary_path)
    experiments = summary.get("experiments", {})
    experiment_summary = experiments.get(experiment_name)
    if experiment_summary is None:
        available = sorted(experiments.keys())
        raise SystemExit(
            "Step 11 could not resolve the requested Step 9 experiment "
            f"{experiment_name!r} inside {step9_summary_path}. Available experiments: {available}"
        )

    ratio = normalize_step9_ratio(requested_ratio)
    if ratio is None:
        ratio = normalize_step9_ratio((dynamic_candidate or {}).get("source_ratio"))
    if ratio is None:
        ratio = normalize_step9_ratio(scorer_selection.get("default_step9_ratio"))
    seed = int(
        requested_seed
        if requested_seed is not None
        else (dynamic_candidate or {}).get("source_seed")
        or scorer_selection.get("default_step9_seed")
        or 0
    )
    if ratio is None or seed <= 0:
        available_runs = sorted(
            (
                run["ratio"],
                run.get("ratio_token"),
                run["seed"],
            )
            for run in experiment_summary.get("runs", {}).values()
        )
        raise SystemExit(
            "Step 11 requires an explicit Step 9 run selection. Pass --step9-ratio and --step9-seed, "
            "or set scorer_selection.default_step9_ratio/default_step9_seed in "
            "schema/step11_clustering_policy.json. Available runs: "
            f"{available_runs}"
        )

    matched_run_key = None
    matched_run = None
    for run_key, run in experiment_summary.get("runs", {}).items():
        run_ratio = normalize_step9_ratio(run.get("ratio"))
        run_seed = int(run.get("seed", 0) or 0)
        if run_ratio is None:
            continue
        if abs(run_ratio - ratio) > 1e-9 or run_seed != seed:
            continue
        matched_run_key = run_key
        matched_run = run
        break
    if matched_run is None or matched_run_key is None:
        available_runs = sorted(
            (
                normalize_step9_ratio(run.get("ratio")),
                run.get("ratio_token"),
                int(run.get("seed", 0) or 0),
            )
            for run in experiment_summary.get("runs", {}).values()
        )
        raise SystemExit(
            "Step 11 could not resolve the requested Step 9 run "
            f"for experiment={experiment_name!r}, ratio={ratio}, seed={seed} inside {step9_summary_path}. "
            f"Available runs: {available_runs}"
        )

    selected_threshold = matched_run.get("selected_threshold")
    if selected_threshold is None:
        raise SystemExit(
            "Step 11 could not resolve the primary threshold from the Step 9 summary for "
            f"{experiment_name!r} run {matched_run_key!r} inside {step9_summary_path}."
        )

    artifacts = matched_run.get("artifacts", {})
    scoring_backend = str(
        matched_run.get("scoring_backend")
        or matched_run.get("training_backend")
        or "legacy_lightgbm_mixed"
    )
    model_path = None
    scorer_artifact_path = None

    if scoring_backend in {"legacy_lightgbm_mixed", "lightgbm"}:
        model_rel_path = str(artifacts.get("model", "") or "").strip()
        if not model_rel_path:
            raise SystemExit(
                "Step 11 could not resolve the Step 9 model path from the Step 9 summary for "
                f"{experiment_name!r} run {matched_run_key!r}."
            )
        model_path = resolve_policy_path(model_rel_path, ROOT / model_rel_path)
        if not model_path.exists():
            raise SystemExit(
                "Step 11 resolved the Step 9 model path to "
                f"{model_path}, but that file does not exist."
            )
    elif scoring_backend in {"residual_logistic", "logistic_regression_l2"}:
        artifact_rel_path = str(artifacts.get("scorer_artifact", "") or "").strip()
        if not artifact_rel_path:
            raise SystemExit(
                "Step 11 could not resolve the Step 9 scorer artifact path from the Step 9 summary for "
                f"{experiment_name!r} run {matched_run_key!r}."
            )
        scorer_artifact_path = resolve_policy_path(artifact_rel_path, ROOT / artifact_rel_path)
        if not scorer_artifact_path.exists():
            raise SystemExit(
                "Step 11 resolved the Step 9 scorer artifact path to "
                f"{scorer_artifact_path}, but that file does not exist."
            )
        model_rel_path = str(artifacts.get("base_model", "") or "").strip()
        if model_rel_path:
            model_path = resolve_policy_path(model_rel_path, ROOT / model_rel_path)
            if not model_path.exists():
                raise SystemExit(
                    "Step 11 resolved the Step 9 base model path to "
                    f"{model_path}, but that file does not exist."
                )
        elif scoring_backend == "residual_logistic":
            raise SystemExit(
                "Step 11 residual logistic scoring requires artifacts.base_model in the Step 9 summary for "
                f"{experiment_name!r} run {matched_run_key!r}."
            )
    else:
        raise SystemExit(
            "Unsupported Step 9 scoring_backend in Step 11: "
            f"{scoring_backend!r} for {experiment_name!r} run {matched_run_key!r}."
        )

    policy_path = resolve_policy_path(summary.get("step9_policy_path"), ROOT / "schema" / "step9_training_policy.json")
    ratio_token = str(matched_run.get("ratio_token", "") or "").strip() or f"{int(round(ratio * 100))}pct"
    scorer_token = f"{experiment_name}_ratio_{ratio_token}_seed_{seed}"
    return {
        "scorer_family": "step9",
        "scorer_token": scorer_token,
        "source_experiment_name": experiment_name,
        "source_run_key": matched_run_key,
        "source_ratio": ratio,
        "source_ratio_token": ratio_token,
        "source_seed": seed,
        "summary_path": step9_summary_path,
        "policy_path": policy_path,
        "model_path": model_path,
        "scoring_backend": scoring_backend,
        "scorer_artifact_path": scorer_artifact_path,
        "primary_threshold": round_float(float(selected_threshold)),
        "selection_mode": "dynamic_family_best" if dynamic_candidate else ("explicit" if requested_experiment_name or requested_ratio is not None or requested_seed is not None else "policy_default"),
        "dynamic_candidate_rank": dynamic_candidate_rank,
        "dynamic_candidate": dynamic_candidate,
    }


def resolve_step9_calibration_scorer_reference(
    policy: dict,
    requested_experiment_name: str | None,
    dynamic_snapshot: dict | None = None,
) -> dict:
    baseline_reference = policy.get("baseline_reference", {})
    input_paths = policy.get("input_paths", {})
    calibration_summary_path = resolve_current_main_summary_path(
        policy,
        baseline_reference.get("step9_calibration_summary") or input_paths.get("step9_calibration_summary"),
        STEP9_CALIBRATION_SUMMARY_PATH,
        "Step 9 calibration summary",
    )
    if not calibration_summary_path.exists():
        raise SystemExit(
            "Step 11 could not find the required Step 9 calibration summary at "
            f"{calibration_summary_path}. Sync reports/step9_calibration_summary.json or update "
            "schema/step11_clustering_policy.json."
        )

    scorer_selection = resolve_scorer_selection(policy)
    default_experiment_name = str(
        scorer_selection.get("default_step9_calibration_experiment_name", "") or ""
    ).strip()
    dynamic_candidate = None
    dynamic_candidate_rank = None
    if not requested_experiment_name and dynamic_snapshot:
        dynamic_candidate, dynamic_candidate_rank = select_fresh_dynamic_candidate(policy, "step9_calibration")
    experiment_name = str(
        requested_experiment_name
        or (dynamic_candidate or {}).get("source_experiment_name")
        or default_experiment_name
    ).strip()
    if not experiment_name:
        raise SystemExit(
            "Step 11 could not resolve a Step 9 calibration scorer experiment name. Pass "
            "--step9-calibration-experiment or set "
            "scorer_selection.default_step9_calibration_experiment_name in "
            "schema/step11_clustering_policy.json."
        )

    summary = step7.load_json(calibration_summary_path)
    experiments = summary.get("experiments", {})
    experiment_summary = experiments.get(experiment_name)
    if experiment_summary is None:
        available = sorted(experiments.keys())
        raise SystemExit(
            "Step 11 could not resolve the requested Step 9 calibration experiment "
            f"{experiment_name!r} inside {calibration_summary_path}. Available experiments: {available}"
        )

    selected_threshold = experiment_summary.get("primary_threshold")
    if selected_threshold is None:
        selected_threshold = experiment_summary.get("selected_threshold")
    if selected_threshold is None:
        raise SystemExit(
            "Step 11 could not resolve the primary threshold from the Step 9 calibration summary for "
            f"{experiment_name!r} inside {calibration_summary_path}."
        )

    artifacts = experiment_summary.get("artifacts", {})
    calibrator_rel_path = str(artifacts.get("calibrator", "") or "").strip()
    if not calibrator_rel_path:
        raise SystemExit(
            "Step 11 could not resolve the Step 9 calibration artifact path from the summary for "
            f"{experiment_name!r}."
        )
    calibrator_path = resolve_policy_path(calibrator_rel_path, ROOT / calibrator_rel_path)
    if not calibrator_path.exists():
        raise SystemExit(
            "Step 11 resolved the Step 9 calibration artifact path to "
            f"{calibrator_path}, but that file does not exist."
        )

    model_rel_path = str(experiment_summary.get("step7_model_path", "") or "").strip()
    if not model_rel_path:
        raise SystemExit(
            "Step 11 could not resolve the Step 7 base model path from the Step 9 calibration summary for "
            f"{experiment_name!r}."
        )
    model_path = resolve_policy_path(model_rel_path, ROOT / model_rel_path)
    if not model_path.exists():
        raise SystemExit(
            "Step 11 resolved the Step 9 calibration base Step 7 model path to "
            f"{model_path}, but that file does not exist."
        )

    policy_path = resolve_policy_path(
        summary.get("step9_calibration_policy_path"),
        ROOT / "schema" / "step9_calibration_policy.json",
    )
    return {
        "scorer_family": "step9_calibration",
        "scorer_token": experiment_name,
        "source_experiment_name": experiment_name,
        "source_run_key": None,
        "source_ratio": None,
        "source_ratio_token": None,
        "source_seed": None,
        "summary_path": calibration_summary_path,
        "policy_path": policy_path,
        "model_path": model_path,
        "scoring_backend": "lightgbm_with_calibration",
        "primary_threshold": round_float(float(selected_threshold)),
        "calibration_artifact_path": calibrator_path,
        "calibrator_type": str(experiment_summary.get("calibrator_type", "") or ""),
        "selection_mode": "dynamic_family_best" if dynamic_candidate else ("explicit" if requested_experiment_name else "policy_default"),
        "dynamic_candidate_rank": dynamic_candidate_rank,
        "dynamic_candidate": dynamic_candidate,
    }


def resolve_scorer_reference(policy: dict, args: argparse.Namespace, dynamic_snapshot: dict | None = None) -> dict:
    scorer_selection = resolve_scorer_selection(policy)
    scorer_family = detect_requested_scorer_family(args, scorer_selection, dynamic_snapshot)
    if scorer_family == "step7":
        return resolve_step7_scorer_reference(policy, args.step7_experiment, dynamic_snapshot)
    if scorer_family == "step9":
        return resolve_step9_scorer_reference(
            policy,
            args.step9_experiment,
            args.step9_ratio,
            args.step9_seed,
            dynamic_snapshot,
        )
    if scorer_family == "step9_calibration":
        return resolve_step9_calibration_scorer_reference(
            policy,
            args.step9_calibration_experiment,
            dynamic_snapshot,
        )
    raise SystemExit(f"Unsupported Step 11 scorer family: {scorer_family}")


def resolve_graph_primary_threshold(scorer_selection: dict, scorer_reference: dict) -> tuple[float, str]:
    overrides = scorer_selection.get("graph_primary_threshold_overrides", {}) or {}
    scorer_token = str(scorer_reference["scorer_token"])
    pairwise_primary_threshold = float(scorer_reference["primary_threshold"])
    if scorer_token in overrides:
        return round_float(float(overrides[scorer_token])), "policy.graph_primary_threshold_overrides"
    return round_float(pairwise_primary_threshold), "selected scorer summary::pairwise_primary_threshold"


def resolve_thresholds_for_probabilities(
    graph_primary_threshold: float,
    scorer_selection: dict,
    probabilities: np.ndarray,
) -> tuple[list[float], dict]:
    requested_sensitivity_thresholds = validate_thresholds(
        list(scorer_selection.get("sensitivity_thresholds", []) or [])
    )
    resolution_cfg = scorer_selection.get("sensitivity_threshold_resolution", {}) or {}
    mode = str(resolution_cfg.get("mode", "static_absolute") or "static_absolute")
    resolved_thresholds = [round_float(float(graph_primary_threshold))]
    diagnostics = {
        "mode": mode,
        "graph_primary_threshold": round_float(float(graph_primary_threshold)),
        "requested_sensitivity_thresholds": requested_sensitivity_thresholds,
        "resolved_sensitivity_thresholds": [],
        "skipped_sensitivity_thresholds": [],
        "quantile_backfills": [],
        "score_distribution": score_distribution(list(np.asarray(probabilities, dtype=float))),
    }
    if not requested_sensitivity_thresholds:
        return validate_thresholds(resolved_thresholds), diagnostics

    if mode == "static_absolute":
        resolved_thresholds.extend(requested_sensitivity_thresholds)
        diagnostics["resolved_sensitivity_thresholds"] = requested_sensitivity_thresholds
        return validate_thresholds(resolved_thresholds), diagnostics

    if mode != "absolute_with_quantile_backfill":
        raise SystemExit(
            "Unsupported Step 11 sensitivity_threshold_resolution.mode: "
            f"{mode!r}. Supported values are 'static_absolute' and 'absolute_with_quantile_backfill'."
        )

    observed_max = float(np.max(np.asarray(probabilities, dtype=float)))
    epsilon = float(resolution_cfg.get("max_score_epsilon", 1e-9) or 1e-9)
    resolved_sensitivity_thresholds: list[float] = []
    skipped_sensitivity_thresholds: list[float] = []
    for threshold in requested_sensitivity_thresholds:
        if float(threshold) <= observed_max + epsilon:
            resolved_sensitivity_thresholds.append(round_float(float(threshold)))
        else:
            skipped_sensitivity_thresholds.append(round_float(float(threshold)))

    quantile_backfills = []
    quantile_fallbacks = [float(value) for value in resolution_cfg.get("quantile_fallbacks", []) or []]
    used_thresholds = set(resolved_thresholds + resolved_sensitivity_thresholds)
    remaining_backfills = len(skipped_sensitivity_thresholds)
    if remaining_backfills > 0:
        for quantile in quantile_fallbacks:
            if remaining_backfills <= 0:
                break
            if quantile <= 0.0 or quantile >= 1.0:
                raise SystemExit(
                    "Step 11 sensitivity_threshold_resolution.quantile_fallbacks values must satisfy 0 < q < 1. "
                    f"Received: {quantile}"
                )
            quantile_threshold = round_float(float(np.quantile(np.asarray(probabilities, dtype=float), quantile)))
            if quantile_threshold <= graph_primary_threshold + epsilon:
                continue
            if quantile_threshold > observed_max + epsilon:
                continue
            if quantile_threshold in used_thresholds:
                continue
            resolved_sensitivity_thresholds.append(quantile_threshold)
            used_thresholds.add(quantile_threshold)
            quantile_backfills.append(
                {
                    "quantile": round_float(quantile),
                    "resolved_threshold": quantile_threshold,
                }
            )
            remaining_backfills -= 1

    diagnostics["resolved_sensitivity_thresholds"] = sorted(resolved_sensitivity_thresholds)
    diagnostics["skipped_sensitivity_thresholds"] = skipped_sensitivity_thresholds
    diagnostics["quantile_backfills"] = quantile_backfills
    resolved_thresholds.extend(resolved_sensitivity_thresholds)
    return validate_thresholds(resolved_thresholds), diagnostics


def graph_threshold_diagnostics(
    graph_primary_threshold: float,
    threshold_views: dict,
    probabilities: np.ndarray,
    scorer_selection: dict,
) -> dict:
    distribution = score_distribution(list(np.asarray(probabilities, dtype=float)))
    observed_max = float(distribution.get("max", 0.0) or 0.0)
    epsilon = float(
        (scorer_selection.get("sensitivity_threshold_resolution", {}) or {}).get("max_score_epsilon", 1e-9)
        or 1e-9
    )
    primary_token = threshold_token(graph_primary_threshold)
    primary_view = threshold_views.get(primary_token, {}) or {}
    threshold_pass_edge_count = int(primary_view.get("threshold_pass_edge_count", 0) or 0)
    post_filter_edge_count = int(primary_view.get("edge_count", 0) or 0)
    exceeds_score_ceiling = float(graph_primary_threshold) > observed_max + epsilon
    return {
        "graph_primary_threshold": round_float(float(graph_primary_threshold)),
        "score_max": round_float(observed_max),
        "score_ceiling_margin": round_float(float(graph_primary_threshold) - observed_max),
        "graph_primary_threshold_exceeds_score_ceiling": exceeds_score_ceiling,
        "graph_primary_threshold_has_candidate_edges": threshold_pass_edge_count > 0,
        "graph_primary_threshold_has_post_filter_edges": post_filter_edge_count > 0,
        "threshold_pass_edge_count": threshold_pass_edge_count,
        "post_filter_edge_count": post_filter_edge_count,
    }


def safe_logit(probabilities: np.ndarray, clip_eps: float) -> np.ndarray:
    clipped = np.clip(np.array(probabilities, dtype=float), clip_eps, 1.0 - clip_eps)
    return np.log(clipped / (1.0 - clipped))


def safe_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.array(values, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def apply_calibration_artifact(probabilities: np.ndarray, artifact: dict) -> np.ndarray:
    calibrator_type = str(artifact.get("calibrator_type", "") or "")
    if calibrator_type != "platt_scaling":
        raise SystemExit(f"Unsupported Step 11 calibration artifact type: {calibrator_type}")
    clip_eps = float(artifact["clip_eps"])
    scale = float(artifact["parameter_scale"])
    bias = float(artifact["parameter_bias"])
    return safe_sigmoid(scale * safe_logit(probabilities, clip_eps) + bias)


def apply_standardization(x_matrix: np.ndarray, standardization: dict) -> np.ndarray:
    means = np.array(standardization.get("mean", []), dtype=float)
    scales = np.array(standardization.get("scale", []), dtype=float)
    if means.size != x_matrix.shape[1] or scales.size != x_matrix.shape[1]:
        raise SystemExit(
            "Step 11 scorer artifact standardization dimensions do not match the requested feature matrix."
        )
    scales = np.where(np.abs(scales) > 1e-12, scales, 1.0)
    clean = np.where(np.isfinite(x_matrix), x_matrix, means)
    return (clean - means) / scales


def apply_step9_logistic_artifact(
    pair_rows: list[dict],
    artifact: dict,
    base_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    artifact_type = str(artifact.get("artifact_type") or artifact.get("scoring_backend") or "")
    if artifact_type not in {"logistic_regression_l2", "residual_logistic"}:
        raise SystemExit(f"Unsupported Step 11 Step 9 scorer artifact type: {artifact_type}")
    feature_names = [str(name) for name in artifact.get("feature_names", [])]
    if not feature_names:
        raise SystemExit("Step 11 scorer artifact does not contain feature_names.")
    step7.validate_feature_columns(pair_rows, feature_names, "step11.zh_target_strict_pairs")
    x_pairs = rows_to_feature_matrix(pair_rows, feature_names)
    x_scaled = apply_standardization(x_pairs, artifact["standardization"])
    coefficients = np.array(artifact.get("parameter_coefficients", []), dtype=float)
    if coefficients.size != x_scaled.shape[1]:
        raise SystemExit(
            "Step 11 scorer artifact coefficient count does not match the requested feature matrix."
        )
    logits = float(artifact["parameter_intercept"]) + x_scaled @ coefficients
    if artifact_type == "residual_logistic":
        if base_probabilities is None:
            raise SystemExit("Step 11 residual logistic scoring requires base probabilities.")
        clip_eps = float(artifact.get("base_probability_clip_eps", 1e-6))
        logits = safe_logit(base_probabilities, clip_eps) + logits
    return safe_sigmoid(logits)


def rows_to_feature_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    matrix = np.empty((len(rows), len(feature_names)), dtype=float)
    for i, row in enumerate(rows):
        for j, feature_name in enumerate(feature_names):
            matrix[i, j] = step7.to_float(row.get(feature_name))
    return matrix


def ranked_index_map(probabilities: np.ndarray, pair_rows: list[dict]) -> dict[int, int]:
    order = sorted(range(len(pair_rows)), key=lambda idx: (-float(probabilities[idx]), pair_rows[idx]["pair_uid"]))
    return {idx: rank for rank, idx in enumerate(order, start=1)}


def row_float(row: dict, key: str, default: float = 0.0) -> float:
    value = step7.to_float(row.get(key))
    if not np.isfinite(value):
        return float(default)
    return float(value)


def row_flag(row: dict, key: str) -> bool:
    return str(row.get(key, "") or "").strip().lower() in {"1", "true", "yes", "y"}


def relation_reliability_config(policy: dict) -> dict:
    edge_filter_cfg = (policy.get("graph_policy", {}) or {}).get("graph_edge_filters", {}) or {}
    return edge_filter_cfg.get("relation_reliability_filter", {}) or {}


def resolve_relation_reliability_min_score(reliability_cfg: dict, scorer_token: str) -> float:
    overrides = reliability_cfg.get("minimum_score_overrides", {}) or {}
    if scorer_token in overrides:
        return round_float(float(overrides[scorer_token]))
    return round_float(float(reliability_cfg.get("minimum_score", 0.0) or 0.0))


def style_consistency_score(row: dict, reliability_cfg: dict) -> float:
    fields = reliability_cfg.get("style_gap_fields") or [
        "item_count_percentile_gap_abs",
        "price_median_percentile_gap_abs",
        "title_length_median_percentile_gap_abs",
        "description_length_median_percentile_gap_abs",
        "digit_ratio_mean_percentile_gap_abs",
        "punct_ratio_mean_percentile_gap_abs",
        "repeated_title_share_percentile_gap_abs",
        "repeated_description_share_percentile_gap_abs",
        "max_category_share_percentile_gap_abs",
        "uppercase_ratio_mean_percentile_gap_abs",
    ]
    values = [row_float(row, str(field), 0.0) for field in fields if str(field) in row]
    if not values:
        return 0.0
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return float(1.0 - np.median(clipped))


def max_semantic_similarity(row: dict) -> float:
    semantic_fields = [
        "embedding_cosine_gte_multilingual_base",
        "embedding_cosine_bge_m3",
        "embedding_cosine_multilingual_e5_large",
        "embedding_cosine_labse",
        "embedding_cosine_paraphrase_multilingual_mpnet",
        "reranker_score_gte_multilingual_reranker_base",
        "reranker_score_bge_reranker_v2_m3",
    ]
    return max((row_float(row, field, 0.0) for field in semantic_fields if field in row), default=0.0)


def compute_relation_reliability(row: dict, reliability_cfg: dict) -> dict:
    weights = reliability_cfg.get("weights", {}) or {}
    thresholds = reliability_cfg.get("thresholds", {}) or {}
    components: list[str] = []
    score = float(reliability_cfg.get("base_score", 0.2) or 0.0)

    has_pgp = row_flag(row, "has_shared_pgp_fingerprint") or row_float(row, "shared_pgp_fingerprint_count_capped") > 0.0
    has_contact = row_flag(row, "has_shared_contact_exact") or row_float(row, "shared_contact_count_capped") > 0.0
    has_direct_identity = bool(has_pgp or has_contact)
    if has_pgp:
        value = float(weights.get("shared_pgp_fingerprint", 0.45))
        score += value
        components.append(f"+shared_pgp_fingerprint:{value:.3f}")
    if has_contact:
        value = float(weights.get("shared_seller_contact", 0.35))
        score += value
        components.append(f"+shared_seller_contact:{value:.3f}")

    description_idf_mean = row_float(row, "shared_description_idf_mean")
    title_idf_mean = row_float(row, "shared_title_idf_mean")
    if row_flag(row, "has_shared_description_clone") and description_idf_mean >= float(
        thresholds.get("rare_description_idf_mean_min", 2.0)
    ):
        value = float(weights.get("rare_description_clone", 0.15))
        score += value
        components.append(f"+rare_description_clone:{value:.3f}")
    if row_flag(row, "has_shared_title_clone") and title_idf_mean >= float(
        thresholds.get("rare_title_idf_mean_min", 2.0)
    ):
        value = float(weights.get("rare_title_clone", 0.10))
        score += value
        components.append(f"+rare_title_clone:{value:.3f}")
    if row_float(row, "shared_rare_ngram_count") >= float(thresholds.get("rare_ngram_count_min", 2.0)):
        value = float(weights.get("rare_ngram_support", 0.08))
        score += value
        components.append(f"+rare_ngram_support:{value:.3f}")

    structural_score = row_float(row, "structural_support_score_raw")
    category_jaccard = row_float(row, "profile_category_jaccard")
    if structural_score >= float(thresholds.get("structural_support_min", 0.5)) or category_jaccard >= float(
        thresholds.get("category_jaccard_min", 0.5)
    ):
        value = float(weights.get("structural_support", 0.12))
        score += value
        components.append(f"+structural_support:{value:.3f}")

    style_score = style_consistency_score(row, reliability_cfg)
    if style_score >= float(thresholds.get("style_consistency_min", 0.72)):
        value = float(weights.get("style_consistency", 0.08))
        score += value
        components.append(f"+style_consistency:{value:.3f}")

    boilerplate_ratio_max = row_float(row, "boilerplate_ratio_max")
    shared_boilerplate_count = row_float(row, "shared_boilerplate_count")
    if boilerplate_ratio_max >= float(thresholds.get("boilerplate_ratio_max_penalty_min", 0.75)) or (
        shared_boilerplate_count >= float(thresholds.get("shared_boilerplate_count_penalty_min", 3.0))
    ):
        value = float(weights.get("boilerplate_template_penalty", -0.20))
        score += value
        components.append(f"{value:.3f}:boilerplate_template_penalty")

    clone_or_structural_support = bool(
        row_flag(row, "has_shared_description_clone")
        or row_flag(row, "has_shared_title_clone")
        or structural_score >= float(thresholds.get("structural_support_min", 0.5))
        or category_jaccard >= float(thresholds.get("category_jaccard_min", 0.5))
    )
    semantic_topic_only = (
        max_semantic_similarity(row) >= float(thresholds.get("semantic_topic_similarity_min", 0.75))
        and not has_direct_identity
        and not clone_or_structural_support
    )
    if semantic_topic_only:
        value = float(weights.get("semantic_topic_only_penalty", -0.25))
        score += value
        components.append(f"{value:.3f}:semantic_topic_only_penalty")

    score = float(np.clip(score, 0.0, 1.0))
    return {
        "score": round_float(score),
        "components": components,
        "has_direct_identity_support": has_direct_identity,
        "style_consistency_score": round_float(style_score),
        "max_semantic_similarity": round_float(max_semantic_similarity(row)),
        "semantic_topic_only": semantic_topic_only,
    }


def build_pair_records(
    pair_rows: list[dict],
    probabilities: np.ndarray,
    seller_index: dict[str, dict],
    thresholds: list[float],
    experiment_name: str,
    policy: dict,
) -> list[dict]:
    reliability_cfg = relation_reliability_config(policy)
    reliability_min_score = resolve_relation_reliability_min_score(reliability_cfg, experiment_name)
    reliability_enabled = bool(reliability_cfg.get("enabled", False))
    hard_keep_direct_identity = bool(reliability_cfg.get("hard_keep_direct_identity", True))
    rank_map = ranked_index_map(probabilities, pair_rows)
    records = []
    for idx, row in enumerate(pair_rows):
        left_profile = seller_index[row["seller_uid_left"]]
        right_profile = seller_index[row["seller_uid_right"]]
        probability = float(probabilities[idx])
        reliability = compute_relation_reliability(row, reliability_cfg)
        reliability_pass = (
            not reliability_enabled
            or float(reliability["score"]) >= reliability_min_score
            or (hard_keep_direct_identity and bool(reliability["has_direct_identity_support"]))
        )
        record = dict(row)
        record["source_seller_raw_left"] = str(left_profile.get("source_seller_raw", "") or "")
        record["source_seller_raw_right"] = str(right_profile.get("source_seller_raw", "") or "")
        record["alias_normalized_left"] = str(left_profile.get("alias_normalized", "") or "")
        record["alias_normalized_right"] = str(right_profile.get("alias_normalized", "") or "")
        record["scoring_experiment_name"] = experiment_name
        record["score_rank_desc"] = int(rank_map[idx])
        record["prob_positive"] = round_float(probability)
        record["relation_reliability_score"] = reliability["score"]
        record["relation_reliability_pass"] = int(reliability_pass)
        record["relation_reliability_direct_identity_support"] = int(reliability["has_direct_identity_support"])
        record["relation_reliability_semantic_topic_only"] = int(reliability["semantic_topic_only"])
        record["relation_reliability_style_consistency"] = reliability["style_consistency_score"]
        record["relation_reliability_components"] = " || ".join(reliability["components"])
        record["edge_score_final"] = round_float(probability * float(reliability["score"]))
        for threshold in thresholds:
            record[threshold_field_name(threshold)] = int(probability >= threshold)
        records.append(record)
    records.sort(key=lambda row: (int(row["score_rank_desc"]), row["pair_uid"]))
    return records


def build_scored_pair_fieldnames(base_fields: list[str], thresholds: list[float]) -> list[str]:
    extras = [
        "source_seller_raw_left",
        "source_seller_raw_right",
        "alias_normalized_left",
        "alias_normalized_right",
        "scoring_experiment_name",
        "score_rank_desc",
        "prob_positive",
        "relation_reliability_score",
        "relation_reliability_pass",
        "relation_reliability_direct_identity_support",
        "relation_reliability_semantic_topic_only",
        "relation_reliability_style_consistency",
        "relation_reliability_components",
        "edge_score_final",
    ]
    extras.extend(threshold_field_name(threshold) for threshold in thresholds)
    return append_unique(list(base_fields), extras)


def align_pair_records_to_fieldnames(records: list[dict], fieldnames: list[str]) -> None:
    for record in records:
        for field in fieldnames:
            record.setdefault(field, "")


def cluster_size_buckets(cluster_sizes: list[int]) -> dict[str, int]:
    buckets = {
        "size_2": 0,
        "size_3_to_5": 0,
        "size_6_to_10": 0,
        "size_11_plus": 0,
    }
    for size in cluster_sizes:
        if size == 2:
            buckets["size_2"] += 1
        elif 3 <= size <= 5:
            buckets["size_3_to_5"] += 1
        elif 6 <= size <= 10:
            buckets["size_6_to_10"] += 1
        elif size >= 11:
            buckets["size_11_plus"] += 1
    return buckets


def find_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    components = []
    seen = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        component = []
        seen.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        component.sort()
        components.append(component)
    return components


def top_k_neighbors(adjacency_scores: dict[str, dict[str, float]], top_k: int) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}
    for seller_uid, neighbors in adjacency_scores.items():
        ranked = sorted(neighbors.items(), key=lambda item: (-item[1], item[0]))
        selected[seller_uid] = {neighbor for neighbor, _score in ranked[:top_k]}
    return selected


def build_adjacency_scores(
    edges: list[dict],
    pair_score_lookup: dict[str, float],
) -> dict[str, dict[str, float]]:
    adjacency_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for edge in edges:
        seller_left = edge["seller_uid_left"]
        seller_right = edge["seller_uid_right"]
        score = float(pair_score_lookup[edge["pair_uid"]])
        adjacency_scores[seller_left][seller_right] = score
        adjacency_scores[seller_right][seller_left] = score
    return adjacency_scores


def build_adjacency_from_edges(edges: list[dict]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        seller_left = edge["seller_uid_left"]
        seller_right = edge["seller_uid_right"]
        adjacency[seller_left].add(seller_right)
        adjacency[seller_right].add(seller_left)
    return adjacency


def resolve_direct_keep_threshold(edge_filter_cfg: dict, scorer_token: str) -> float | None:
    overrides = edge_filter_cfg.get("direct_keep_threshold_overrides", {}) or {}
    if scorer_token not in overrides:
        return None
    value = overrides[scorer_token]
    if value in ("", None):
        return None
    return round_float(float(value))


def resolve_shared_neighbor_pruning_mode(edge_filter_cfg: dict) -> str:
    value = str(edge_filter_cfg.get("shared_neighbor_pruning_mode", "iterative") or "iterative").strip().lower()
    if value not in {"single_pass", "iterative"}:
        return "iterative"
    return value


def apply_graph_edge_filters(
    edges: list[dict],
    pair_score_lookup: dict[str, float],
    scorer_token: str,
    policy: dict,
) -> tuple[list[dict], dict]:
    graph_policy = policy.get("graph_policy", {})
    edge_filter_cfg = graph_policy.get("graph_edge_filters", {}) or {}
    enabled = bool(edge_filter_cfg.get("enabled", False))
    direct_keep_threshold = resolve_direct_keep_threshold(edge_filter_cfg, scorer_token)
    reliability_cfg = edge_filter_cfg.get("relation_reliability_filter", {}) or {}
    reliability_enabled = bool(reliability_cfg.get("enabled", False))
    reliability_min_score = resolve_relation_reliability_min_score(reliability_cfg, scorer_token)
    reliability_hard_keep_direct = bool(reliability_cfg.get("hard_keep_direct_identity", True))
    diagnostics = {
        "enabled": enabled,
        "require_shared_neighbor_count_min": int(edge_filter_cfg.get("require_shared_neighbor_count_min", 0) or 0),
        "reciprocal_top_k": int(edge_filter_cfg.get("reciprocal_top_k", 0) or 0),
        "require_triangle_participation": bool(edge_filter_cfg.get("require_triangle_participation", False)),
        "shared_neighbor_pruning_mode": resolve_shared_neighbor_pruning_mode(edge_filter_cfg),
        "shared_neighbor_pruning_max_passes": int(edge_filter_cfg.get("shared_neighbor_pruning_max_passes", 0) or 0),
        "shared_neighbor_pruning_passes": 0,
        "direct_keep_threshold": direct_keep_threshold,
        "relation_reliability_filter": {
            "enabled": reliability_enabled,
            "minimum_score": reliability_min_score,
            "hard_keep_direct_identity": reliability_hard_keep_direct,
            "pre_filter_edge_count": int(len(edges)),
            "post_filter_edge_count": int(len(edges)),
            "removed_by_relation_reliability": 0,
            "direct_identity_hard_kept_count": 0,
            "score_distribution": score_distribution(
                [row_float(edge, "relation_reliability_score") for edge in edges]
            ),
        },
        "pre_filter_edge_count": int(len(edges)),
        "after_relation_reliability_edge_count": int(len(edges)),
        "after_reciprocal_top_k_edge_count": int(len(edges)),
        "after_shared_neighbor_edge_count": int(len(edges)),
        "after_triangle_edge_count": int(len(edges)),
        "removed_by_relation_reliability": 0,
        "removed_by_reciprocal_top_k": 0,
        "removed_by_shared_neighbor": 0,
        "removed_by_triangle": 0,
        "post_filter_edge_count": int(len(edges)),
    }
    if not enabled or not edges:
        return edges, diagnostics

    filtered_edges = list(edges)
    if reliability_enabled and filtered_edges:
        reliability_edges = []
        direct_identity_hard_kept_count = 0
        for edge in filtered_edges:
            reliability_score = row_float(edge, "relation_reliability_score")
            has_direct_identity = row_flag(edge, "relation_reliability_direct_identity_support")
            if reliability_score >= reliability_min_score or (
                reliability_hard_keep_direct and has_direct_identity
            ):
                reliability_edges.append(edge)
                if reliability_score < reliability_min_score and has_direct_identity:
                    direct_identity_hard_kept_count += 1
        diagnostics["after_relation_reliability_edge_count"] = int(len(reliability_edges))
        diagnostics["removed_by_relation_reliability"] = int(len(filtered_edges) - len(reliability_edges))
        diagnostics["relation_reliability_filter"].update(
            {
                "post_filter_edge_count": int(len(reliability_edges)),
                "removed_by_relation_reliability": int(len(filtered_edges) - len(reliability_edges)),
                "direct_identity_hard_kept_count": int(direct_identity_hard_kept_count),
            }
        )
        filtered_edges = reliability_edges

    reciprocal_top_k = int(edge_filter_cfg.get("reciprocal_top_k", 0) or 0)
    if reciprocal_top_k > 0:
        adjacency_scores = build_adjacency_scores(filtered_edges, pair_score_lookup)
        top_k_map = top_k_neighbors(adjacency_scores, reciprocal_top_k)
        reciprocal_edges = []
        for edge in filtered_edges:
            seller_left = edge["seller_uid_left"]
            seller_right = edge["seller_uid_right"]
            if seller_right in top_k_map.get(seller_left, set()) and seller_left in top_k_map.get(seller_right, set()):
                reciprocal_edges.append(edge)
        diagnostics["after_reciprocal_top_k_edge_count"] = int(len(reciprocal_edges))
        diagnostics["removed_by_reciprocal_top_k"] = int(len(filtered_edges) - len(reciprocal_edges))
        filtered_edges = reciprocal_edges

    shared_neighbor_min = int(edge_filter_cfg.get("require_shared_neighbor_count_min", 0) or 0)
    if shared_neighbor_min > 0 and filtered_edges:
        pruning_mode = resolve_shared_neighbor_pruning_mode(edge_filter_cfg)
        max_passes = int(edge_filter_cfg.get("shared_neighbor_pruning_max_passes", 0) or 0)
        original_edge_count = len(filtered_edges)
        supported_edges = list(filtered_edges)
        pass_count = 0
        while supported_edges:
            pass_count += 1
            adjacency = build_adjacency_from_edges(supported_edges)
            next_supported_edges = []
            for edge in supported_edges:
                seller_left = edge["seller_uid_left"]
                seller_right = edge["seller_uid_right"]
                shared_neighbor_count = len(adjacency[seller_left] & adjacency[seller_right])
                score = float(pair_score_lookup[edge["pair_uid"]])
                if shared_neighbor_count >= shared_neighbor_min or (
                    direct_keep_threshold is not None and score >= direct_keep_threshold
                ):
                    next_supported_edges.append(edge)
            if len(next_supported_edges) == len(supported_edges):
                supported_edges = next_supported_edges
                break
            supported_edges = next_supported_edges
            if pruning_mode != "iterative":
                break
            if max_passes > 0 and pass_count >= max_passes:
                break
        diagnostics["shared_neighbor_pruning_passes"] = int(pass_count)
        diagnostics["after_shared_neighbor_edge_count"] = int(len(supported_edges))
        diagnostics["removed_by_shared_neighbor"] = int(original_edge_count - len(supported_edges))
        filtered_edges = supported_edges

    if bool(edge_filter_cfg.get("require_triangle_participation", False)) and filtered_edges:
        adjacency = build_adjacency_from_edges(filtered_edges)
        triangle_edges = []
        for edge in filtered_edges:
            seller_left = edge["seller_uid_left"]
            seller_right = edge["seller_uid_right"]
            if adjacency[seller_left] & adjacency[seller_right]:
                triangle_edges.append(edge)
        diagnostics["after_triangle_edge_count"] = int(len(triangle_edges))
        diagnostics["removed_by_triangle"] = int(len(filtered_edges) - len(triangle_edges))
        filtered_edges = triangle_edges

    diagnostics["post_filter_edge_count"] = int(len(filtered_edges))
    return filtered_edges, diagnostics


def build_cluster_rows_and_summary(
    pair_records: list[dict],
    pair_score_lookup: dict[str, float],
    seller_index: dict[str, dict],
    experiment_name: str,
    threshold: float,
    pair_universe_seller_count: int,
    minimum_cluster_size: int,
    scorer_token: str,
    policy: dict,
) -> tuple[list[dict], dict]:
    threshold_edges = [row for row in pair_records if pair_score_lookup[row["pair_uid"]] >= threshold]
    kept_edges, filter_diagnostics = apply_graph_edge_filters(
        threshold_edges,
        pair_score_lookup,
        scorer_token,
        policy,
    )
    adjacency = build_adjacency_from_edges(kept_edges)

    components = find_components(adjacency)
    component_payloads = []
    for nodes in components:
        if len(nodes) < minimum_cluster_size:
            continue
        node_set = set(nodes)
        component_edges = [
            edge
            for edge in kept_edges
            if edge["seller_uid_left"] in node_set and edge["seller_uid_right"] in node_set
        ]
        edge_scores = [pair_score_lookup[edge["pair_uid"]] for edge in component_edges]
        markets = sorted({str(seller_index[node].get("source_market_raw", "") or "") for node in nodes if node in seller_index})
        possible_edge_count = len(nodes) * (len(nodes) - 1) // 2
        component_adjacency = build_adjacency_from_edges(component_edges)
        member_degrees = [len(component_adjacency.get(node, set())) for node in nodes]
        leaf_member_count = sum(1 for degree in member_degrees if degree == 1)
        component_payloads.append(
            {
                "nodes": nodes,
                "edges": component_edges,
                "cluster_size": len(nodes),
                "cluster_edge_count": len(component_edges),
                "cluster_possible_edge_count": possible_edge_count,
                "cluster_density": 0.0 if possible_edge_count == 0 else len(component_edges) / possible_edge_count,
                "cluster_score_min": min(edge_scores) if edge_scores else 0.0,
                "cluster_score_mean": float(np.mean(edge_scores)) if edge_scores else 0.0,
                "cluster_score_max": max(edge_scores) if edge_scores else 0.0,
                "cluster_markets": markets,
                "leaf_member_count": leaf_member_count,
                "leaf_member_share": 0.0 if not nodes else leaf_member_count / len(nodes),
                "max_member_degree": max(member_degrees, default=0),
                "min_member_degree": min(member_degrees, default=0),
                "is_tree": bool(len(nodes) >= 2 and len(component_edges) == len(nodes) - 1),
            }
        )

    component_payloads.sort(
        key=lambda payload: (
            -payload["cluster_size"],
            -payload["cluster_score_max"],
            -payload["cluster_score_mean"],
            payload["nodes"][0],
        )
    )

    cluster_rows = []
    total_cluster_member_count = 0
    total_leaf_member_count = 0
    tree_cluster_count = 0
    for cluster_rank, payload in enumerate(component_payloads, start=1):
        payload["cluster_rank"] = cluster_rank
        payload["cluster_id"] = (
            f"zh_target_strict_{experiment_name}_thr_{threshold_token(threshold)}_c{cluster_rank:04d}"
        )
        neighbor_map: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in payload["edges"]:
            seller_left = edge["seller_uid_left"]
            seller_right = edge["seller_uid_right"]
            probability = pair_score_lookup[edge["pair_uid"]]
            neighbor_map[seller_left].append((seller_right, probability))
            neighbor_map[seller_right].append((seller_left, probability))
        total_cluster_member_count += payload["cluster_size"]
        total_leaf_member_count += int(payload["leaf_member_count"])
        tree_cluster_count += int(payload["is_tree"])

        member_order = sorted(
            payload["nodes"],
            key=lambda seller_uid: (
                -len(neighbor_map[seller_uid]),
                -max((probability for _, probability in neighbor_map[seller_uid]), default=0.0),
                seller_uid,
            ),
        )
        for member_rank, seller_uid in enumerate(member_order, start=1):
            profile = seller_index[seller_uid]
            preview = seller_preview_fields(profile)
            neighbors = sorted(neighbor_map[seller_uid], key=lambda item: (-item[1], item[0]))
            neighbor_scores = [probability for _, probability in neighbors]
            cluster_rows.append(
                {
                    "cluster_id": payload["cluster_id"],
                    "cluster_rank": payload["cluster_rank"],
                    "threshold": round_float(threshold),
                    "threshold_token": threshold_token(threshold),
                    "cluster_size": payload["cluster_size"],
                    "cluster_edge_count": payload["cluster_edge_count"],
                    "cluster_possible_edge_count": payload["cluster_possible_edge_count"],
                    "cluster_density": round_float(payload["cluster_density"]),
                    "cluster_score_min": round_float(payload["cluster_score_min"]),
                    "cluster_score_mean": round_float(payload["cluster_score_mean"]),
                    "cluster_score_max": round_float(payload["cluster_score_max"]),
                    "cluster_markets": " || ".join(payload["cluster_markets"]),
                    "member_rank_within_cluster": member_rank,
                    "member_degree": len(neighbors),
                    "member_edge_score_sum": round_float(sum(neighbor_scores)),
                    "member_edge_score_mean": round_float(np.mean(neighbor_scores)) if neighbor_scores else 0.0,
                    "member_edge_score_max": round_float(max(neighbor_scores)) if neighbor_scores else 0.0,
                    "strongest_neighbor_seller_uids": " || ".join(neighbor for neighbor, _ in neighbors[:3]),
                    "strongest_neighbor_scores": " || ".join(f"{probability:.6f}" for _, probability in neighbors[:3]),
                    "seller_uid": seller_uid,
                    **preview,
                }
            )

    cluster_sizes = [payload["cluster_size"] for payload in component_payloads]
    threshold_summary = {
        "threshold": round_float(threshold),
        "threshold_token": threshold_token(threshold),
        "threshold_pass_edge_count": len(threshold_edges),
        "edge_count": len(kept_edges),
        "edge_rate_within_candidate_pairs": round_float(len(kept_edges) / max(len(pair_records), 1)),
        "seller_count_with_edge": len(adjacency),
        "seller_count_without_edge_in_candidate_universe": pair_universe_seller_count - len(adjacency),
        "cluster_count": len(component_payloads),
        "largest_cluster_size": max(cluster_sizes, default=0),
        "largest_cluster_edge_count": max((payload["cluster_edge_count"] for payload in component_payloads), default=0),
        "cluster_size_distribution": {str(size): count for size, count in sorted(Counter(cluster_sizes).items())},
        "cluster_size_buckets": cluster_size_buckets(cluster_sizes),
        "edge_score_distribution": score_distribution([pair_score_lookup[edge["pair_uid"]] for edge in kept_edges]),
        "graph_edge_filtering": filter_diagnostics,
        "tree_cluster_count": int(tree_cluster_count),
        "tree_cluster_share": round_float(tree_cluster_count / max(len(component_payloads), 1)),
        "leaf_member_count_in_clusters": int(total_leaf_member_count),
        "cluster_member_count": int(total_cluster_member_count),
        "leaf_member_share_in_clusters": round_float(total_leaf_member_count / max(total_cluster_member_count, 1)),
        "top_clusters_by_size": [
            {
                "cluster_id": payload["cluster_id"],
                "cluster_rank": payload["cluster_rank"],
                "cluster_size": payload["cluster_size"],
                "cluster_edge_count": payload["cluster_edge_count"],
                "cluster_density": round_float(payload["cluster_density"]),
                "cluster_score_mean": round_float(payload["cluster_score_mean"]),
                "leaf_member_count": int(payload["leaf_member_count"]),
                "leaf_member_share": round_float(payload["leaf_member_share"]),
                "max_member_degree": int(payload["max_member_degree"]),
                "min_member_degree": int(payload["min_member_degree"]),
                "is_tree": bool(payload["is_tree"]),
                "cluster_markets": payload["cluster_markets"],
                "sample_members": [seller_display_name(seller_index[seller_uid]) for seller_uid in payload["nodes"][:5]],
            }
            for payload in component_payloads[:10]
        ],
    }
    return cluster_rows, threshold_summary


def main() -> None:
    args = parse_args()
    policy_path = args.policy.resolve()
    policy = step7.load_json(policy_path)
    scorer_selection = resolve_scorer_selection(policy)
    dynamic_mainline_candidates = build_dynamic_mainline_candidates(policy, scorer_selection)
    input_paths = policy.get("input_paths", {})
    pair_feature_path = ROOT / input_paths.get("pair_features", policy["input_dependencies"][0])
    seller_profile_path = ROOT / input_paths.get("seller_profiles", policy["input_dependencies"][1])
    scorer_reference = resolve_scorer_reference(policy, args, dynamic_mainline_candidates)
    scorer_token = str(scorer_reference["scorer_token"])
    pairwise_primary_threshold = float(scorer_reference["primary_threshold"])
    graph_primary_threshold, graph_primary_threshold_source = resolve_graph_primary_threshold(
        scorer_selection,
        scorer_reference,
    )
    summary_path_resolved = Path(scorer_reference["summary_path"])
    policy_path_resolved = Path(scorer_reference["policy_path"])
    model_path_value = scorer_reference.get("model_path")
    model_path = Path(model_path_value) if model_path_value is not None else None
    scoring_backend = str(scorer_reference.get("scoring_backend", "lightgbm") or "lightgbm")
    calibration_artifact_path = scorer_reference.get("calibration_artifact_path")
    scorer_artifact_path = scorer_reference.get("scorer_artifact_path")

    minimum_cluster_size = int(policy["graph_policy"]["minimum_cluster_size"])

    lgb = None if scoring_backend == "logistic_regression_l2" else step7.require_lightgbm()
    seller_index = load_seller_index(seller_profile_path)
    pair_rows = step7.load_csv(pair_feature_path)
    step7.ensure_non_empty(pair_rows, "step11.zh_target_strict_pairs")
    ensure_unique_pair_uids(pair_rows)
    ensure_pair_sellers_exist(pair_rows, seller_index)
    pair_rows = ensure_core_transfer_eligible(pair_rows)

    booster = None
    calibration_artifact = None
    scorer_artifact = None
    model_num_trees = None
    base_probabilities = None
    if scoring_backend in {"lightgbm", "legacy_lightgbm_mixed", "lightgbm_with_calibration"}:
        if model_path is None:
            raise SystemExit(f"Step 11 scoring backend {scoring_backend!r} requires a LightGBM model path.")
        booster = lgb.Booster(model_file=str(model_path))
        feature_names = list(booster.feature_name())
        if not feature_names:
            raise SystemExit("Step 11 could not recover feature names from the LightGBM model file.")
        step7.validate_feature_columns(pair_rows, feature_names, "step11.zh_target_strict_pairs")
        x_pairs = rows_to_feature_matrix(pair_rows, feature_names)
        probabilities = booster.predict(x_pairs)
        model_num_trees = int(booster.num_trees())
    elif scoring_backend == "residual_logistic":
        if model_path is None:
            raise SystemExit("Step 11 residual logistic scoring requires a base LightGBM model path.")
        if scorer_artifact_path is None:
            raise SystemExit("Step 11 residual logistic scoring requires a scorer artifact path.")
        booster = lgb.Booster(model_file=str(model_path))
        base_feature_names = list(booster.feature_name())
        if not base_feature_names:
            raise SystemExit("Step 11 could not recover feature names from the residual base LightGBM model file.")
        step7.validate_feature_columns(pair_rows, base_feature_names, "step11.zh_target_strict_pairs")
        base_matrix = rows_to_feature_matrix(pair_rows, base_feature_names)
        base_probabilities = booster.predict(base_matrix)
        scorer_artifact = step7.load_json(Path(scorer_artifact_path))
        feature_names = [str(name) for name in scorer_artifact.get("feature_names", [])]
        probabilities = apply_step9_logistic_artifact(
            pair_rows,
            scorer_artifact,
            base_probabilities=base_probabilities,
        )
        model_num_trees = int(booster.num_trees())
    elif scoring_backend == "logistic_regression_l2":
        if scorer_artifact_path is None:
            raise SystemExit("Step 11 logistic regression scoring requires a scorer artifact path.")
        scorer_artifact = step7.load_json(Path(scorer_artifact_path))
        feature_names = [str(name) for name in scorer_artifact.get("feature_names", [])]
        probabilities = apply_step9_logistic_artifact(pair_rows, scorer_artifact)
    else:
        raise SystemExit(f"Unsupported Step 11 scoring backend: {scoring_backend}")

    calibration_artifact = None
    if calibration_artifact_path is not None:
        calibration_artifact = step7.load_json(Path(calibration_artifact_path))
        probabilities = apply_calibration_artifact(probabilities, calibration_artifact)
    resolved_thresholds, threshold_resolution = resolve_thresholds_for_probabilities(
        graph_primary_threshold,
        scorer_selection,
        probabilities,
    )
    thresholds = args.thresholds or resolved_thresholds
    thresholds = validate_thresholds(thresholds)
    if thresholds != resolved_thresholds:
        raise SystemExit(
            "Step 11 currently uses the threshold set resolved from the selected scorer graph working threshold "
            "plus the sensitivity_threshold_resolution policy in schema/step11_clustering_policy.json. "
            f"Expected thresholds: {resolved_thresholds}; received: {thresholds}"
        )
    pair_score_lookup = {
        row["pair_uid"]: float(probability)
        for row, probability in zip(pair_rows, probabilities, strict=True)
    }
    pair_records = build_pair_records(pair_rows, probabilities, seller_index, thresholds, scorer_token, policy)
    pair_universe_sellers = sorted(
        {
            seller_uid
            for row in pair_rows
            for seller_uid in (row["seller_uid_left"], row["seller_uid_right"])
        }
    )

    scored_pair_output_path = output_path(
        policy["output_templates"]["scored_pairs"],
        experiment_name=scorer_token,
    )
    expected_scored_pair_fields = expand_threshold_field_templates(
        list(policy.get("scored_pair_output_fields", [])),
        thresholds,
    )
    if expected_scored_pair_fields:
        align_pair_records_to_fieldnames(pair_records, expected_scored_pair_fields)
        scored_pair_fieldnames = expected_scored_pair_fields
    else:
        scored_pair_fieldnames = build_scored_pair_fieldnames(list(pair_rows[0].keys()), thresholds)
    if expected_scored_pair_fields and set(scored_pair_fieldnames) != set(expected_scored_pair_fields):
        raise SystemExit(
            "Step 11 scored-pair fieldnames no longer match schema/step11_clustering_policy.json. "
            "Update the policy or the script before running on Linux."
        )
    step7.write_csv(
        scored_pair_output_path,
        pair_records,
        scored_pair_fieldnames,
    )

    cluster_output_paths = {}
    threshold_views = {}
    for threshold in thresholds:
        cluster_rows, threshold_summary = build_cluster_rows_and_summary(
            pair_records,
            pair_score_lookup,
            seller_index,
            scorer_token,
            threshold,
            len(pair_universe_sellers),
            minimum_cluster_size,
            scorer_token,
            policy,
        )
        cluster_path = output_path(
            policy["output_templates"]["clusters"],
            experiment_name=scorer_token,
            threshold_token=threshold_token(threshold),
        )
        step7.write_csv(cluster_path, cluster_rows, policy["cluster_output_fields"])
        cluster_output_paths[threshold_token(threshold)] = relative_path(cluster_path)
        threshold_views[threshold_token(threshold)] = threshold_summary

    graph_diagnostics = graph_threshold_diagnostics(
        graph_primary_threshold,
        threshold_views,
        probabilities,
        scorer_selection,
    )
    acceptance_checks = {
        "pair_rows_scored": len(pair_records) == len(pair_rows),
        "feature_names_resolved_for_scorer": len(feature_names) > 0,
        "all_pair_rows_core_transfer_eligible": all(
            str(row.get("core_transfer_eligible", "")).strip() == "1" for row in pair_rows
        ),
        "graph_primary_threshold_present": threshold_token(graph_primary_threshold) in threshold_views,
        "cluster_files_emitted_for_all_thresholds": len(cluster_output_paths) == len(thresholds),
        "graph_primary_threshold_not_above_score_ceiling": not graph_diagnostics[
            "graph_primary_threshold_exceeds_score_ceiling"
        ],
        "graph_primary_threshold_has_candidate_edges": graph_diagnostics[
            "graph_primary_threshold_has_candidate_edges"
        ],
        "graph_primary_threshold_has_post_filter_edges": graph_diagnostics[
            "graph_primary_threshold_has_post_filter_edges"
        ],
    }
    acceptance_checks_failed = [
        check_name for check_name, passed in acceptance_checks.items() if not bool(passed)
    ]

    summary_path = output_path(
        policy["output_templates"]["summary"],
        experiment_name=scorer_token,
    )
    summary = {
        "policy_path": relative_path(policy_path),
        "input_dependencies": policy["input_dependencies"],
        "output_paths": {
            "scored_pairs": relative_path(scored_pair_output_path),
            "clusters_by_threshold": cluster_output_paths,
            "summary": relative_path(summary_path),
        },
        "scorer_reference": {
            "scorer_family": scorer_reference["scorer_family"],
            "summary_path": relative_path(summary_path_resolved),
            "policy_path": relative_path(policy_path_resolved),
            "resolved_pairwise_primary_threshold": round_float(pairwise_primary_threshold),
            "resolved_graph_primary_threshold": round_float(graph_primary_threshold),
            "resolved_graph_primary_threshold_source": graph_primary_threshold_source,
            "scoring_backend": scoring_backend,
            "resolved_model_path": None if model_path is None else relative_path(model_path),
            "resolved_calibration_artifact_path": None
            if calibration_artifact_path is None
            else relative_path(Path(calibration_artifact_path)),
            "resolved_scorer_artifact_path": None
            if scorer_artifact_path is None
            else relative_path(Path(scorer_artifact_path)),
        },
        "selected_scorer": {
            "default_scorer_family": str(scorer_selection.get("default_scorer_family", "") or ""),
            "default_step7_experiment_name": str(scorer_selection.get("default_step7_experiment_name", "") or ""),
            "default_step9_experiment_name": str(scorer_selection.get("default_step9_experiment_name", "") or ""),
            "default_step9_calibration_experiment_name": str(
                scorer_selection.get("default_step9_calibration_experiment_name", "") or ""
            ),
            "default_step9_ratio": scorer_selection.get("default_step9_ratio"),
            "default_step9_seed": scorer_selection.get("default_step9_seed"),
            "requested_scorer_family": args.scorer_family,
            "requested_step7_experiment": args.step7_experiment,
            "requested_step9_experiment": args.step9_experiment,
            "requested_step9_ratio": args.step9_ratio,
            "requested_step9_seed": args.step9_seed,
            "requested_step9_calibration_experiment": args.step9_calibration_experiment,
            "scorer_family": scorer_reference["scorer_family"],
            "scorer_token": scorer_token,
            "source_experiment_name": scorer_reference["source_experiment_name"],
            "source_run_key": scorer_reference["source_run_key"],
            "source_ratio": scorer_reference["source_ratio"],
            "source_ratio_token": scorer_reference["source_ratio_token"],
            "source_seed": scorer_reference["source_seed"],
            "scoring_backend": scoring_backend,
            "model_path": None if model_path is None else relative_path(model_path),
            "calibration_artifact_path": None
            if calibration_artifact_path is None
            else relative_path(Path(calibration_artifact_path)),
            "scorer_artifact_path": None
            if scorer_artifact_path is None
            else relative_path(Path(scorer_artifact_path)),
            "calibrator_type": scorer_reference.get("calibrator_type"),
            "selection_mode": str(scorer_reference.get("selection_mode", "") or ""),
            "dynamic_candidate_rank": scorer_reference.get("dynamic_candidate_rank"),
            "dynamic_candidate": scorer_reference.get("dynamic_candidate"),
            "model_num_trees": model_num_trees,
            "model_feature_names": feature_names,
            "pairwise_primary_threshold": round_float(pairwise_primary_threshold),
            "graph_primary_threshold": round_float(graph_primary_threshold),
            "graph_primary_threshold_source": graph_primary_threshold_source,
            "primary_threshold": round_float(graph_primary_threshold),
            "thresholds": thresholds,
            "threshold_resolution": threshold_resolution,
        },
        "dynamic_mainline_candidates": dynamic_mainline_candidates,
        "graph_policy": policy["graph_policy"],
        "universes": {
            "seller_profile_count": len(seller_index),
            "pair_row_count": len(pair_rows),
            "pair_universe_unique_seller_count": len(pair_universe_sellers),
            "seller_profiles_not_in_candidate_universe": len(seller_index) - len(pair_universe_sellers),
            "candidate_universe_share_of_profiles": round_float(len(pair_universe_sellers) / max(len(seller_index), 1)),
        },
        "pair_table_diagnostics": {
            "review_stratum_counts": dict(Counter(row["review_stratum"] for row in pair_rows)),
            "review_priority_counts": dict(Counter(row["review_priority"] for row in pair_rows)),
            "candidate_scope_counts": dict(Counter(row["candidate_scope"] for row in pair_rows)),
            "core_transfer_eligible_counts": dict(Counter(str(row["core_transfer_eligible"]) for row in pair_rows)),
        },
        "pair_score_distribution": score_distribution(list(pair_score_lookup.values())),
        "relation_reliability_policy": relation_reliability_config(policy),
        "relation_reliability_distribution": score_distribution(
            [row_float(row, "relation_reliability_score") for row in pair_records]
        ),
        "edge_score_final_distribution": score_distribution(
            [row_float(row, "edge_score_final") for row in pair_records]
        ),
        "calibration_artifact": calibration_artifact,
        "scorer_artifact": scorer_artifact,
        "graph_threshold_diagnostics": graph_diagnostics,
        "threshold_views": threshold_views,
        "acceptance_checks": acceptance_checks,
        "acceptance_checks_failed": acceptance_checks_failed,
    }

    step7.write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
