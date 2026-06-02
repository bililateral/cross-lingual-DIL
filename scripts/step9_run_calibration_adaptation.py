from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "step7_transfer_safe_pair_feature_schema.json"
STEP7_POLICY_PATH = ROOT / "schema" / "step7_training_policy.json"
STEP9_CALIBRATION_POLICY_PATH = ROOT / "schema" / "step9_calibration_policy.json"
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
            "Run Step 9 frozen-score calibration adaptation using synchronized Step 7 model outputs "
            "and the fixed zh_target_strict reviewed split containers."
        )
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help=(
            "Experiment name from schema/step9_calibration_policy.json. Repeat to run multiple "
            "calibration experiments. Defaults to policy default_experiments."
        ),
    )
    return parser.parse_args()


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
                "Step 9 calibration could not find the Step 7 training summary required by policy at "
                f"{preferred_path}. Sync schema/step9_calibration_policy.json and "
                "reports/step7_training_summary.json, or intentionally point the policy at the archived "
                "snapshot you want to use."
            )
        raise SystemExit(
            "Step 9 calibration could not find any Step 7 training summary record. Expected "
            "reports/step7_training_summary.json or an archived reports/**/step7_training_summary.json snapshot."
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
            "Step 9 calibration could not resolve the required Step 7 experiment summary "
            f"for {experiment_name!r} inside {only_path}. Rebuild the requested Step 7 summary "
            "or point schema/step9_calibration_policy.json at the intended archived snapshot."
        )
    available = sorted(
        {
            name
            for _path, payload in summary_records
            for name in payload.get("experiments", {}).keys()
        }
    )
    raise SystemExit(
        "Step 9 calibration could not resolve the required Step 7 experiment summary "
        f"for {experiment_name!r}. Available archived experiments: {available}"
    )


def dataset_summary(rows: list[dict]) -> dict:
    return {
        "row_count": len(rows),
        "label_counts": dict(Counter(row["review_label"] for row in rows)),
        "review_stratum_counts": dict(Counter(row["review_stratum"] for row in rows)),
    }


def prediction_fieldnames() -> list[str]:
    return [
        "experiment_name",
        "pair_uid",
        "data_bucket",
        "split_name",
        "review_label",
        "y_true",
        "base_prob_positive",
        "calibrated_prob_positive",
        "pred_positive_primary",
        "pred_positive_fixed_half",
        "review_stratum",
        "source_seller_raw_left",
        "source_seller_raw_right",
    ]


def prediction_rows(
    rows: list[dict],
    base_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    primary_threshold: float,
    fixed_half_threshold: float,
    experiment_name: str,
) -> list[dict]:
    primary_predictions = (calibrated_probabilities >= primary_threshold).astype(int)
    fixed_half_predictions = (calibrated_probabilities >= fixed_half_threshold).astype(int)
    output = []
    for row, base_prob, calibrated_prob, primary_pred, fixed_pred in zip(
        rows,
        base_probabilities,
        calibrated_probabilities,
        primary_predictions,
        fixed_half_predictions,
        strict=True,
    ):
        output.append(
            {
                "experiment_name": experiment_name,
                "pair_uid": row["pair_uid"],
                "data_bucket": row["data_bucket"],
                "split_name": row["split_name"],
                "review_label": row["review_label"],
                "y_true": step7.label_to_int(row["review_label"]),
                "base_prob_positive": round(float(base_prob), 6),
                "calibrated_prob_positive": round(float(calibrated_prob), 6),
                "pred_positive_primary": int(primary_pred),
                "pred_positive_fixed_half": int(fixed_pred),
                "review_stratum": row["review_stratum"],
                "source_seller_raw_left": row["source_seller_raw_left"],
                "source_seller_raw_right": row["source_seller_raw_right"],
            }
        )
    return output


def output_path(template: str, experiment_name: str) -> Path:
    return ROOT / template.format(experiment_name=experiment_name)


def plain_output_path(template: str) -> Path:
    return ROOT / template


def clone_threshold_policy(threshold_policy: dict) -> dict:
    cloned = json.loads(json.dumps(threshold_policy))
    cloned.pop("mode", None)
    cloned.pop("bootstrap", None)
    return {"threshold_selection": cloned}


def probability_stats(probabilities: np.ndarray) -> dict:
    ordered = np.array(probabilities, dtype=float)
    return {
        "min": round(float(np.min(ordered)), 6),
        "q25": round(float(np.quantile(ordered, 0.25)), 6),
        "mean": round(float(np.mean(ordered)), 6),
        "q75": round(float(np.quantile(ordered, 0.75)), 6),
        "max": round(float(np.max(ordered)), 6),
    }


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - y_true) ** 2))


def safe_logit(probabilities: np.ndarray, clip_eps: float) -> np.ndarray:
    clipped = np.clip(np.array(probabilities, dtype=float), clip_eps, 1.0 - clip_eps)
    return np.log(clipped / (1.0 - clipped))


def safe_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_platt_scaling(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    calibrator_cfg: dict,
) -> tuple[dict, dict]:
    if len(probabilities) != len(y_true):
        raise ValueError("Step 9 calibration requires base probabilities and labels with matching lengths")
    if len(probabilities) == 0:
        raise ValueError("Step 9 calibration requires a non-empty fitting split")

    clip_eps = float(calibrator_cfg.get("clip_eps", 1e-6))
    l2_penalty = float(calibrator_cfg.get("l2_penalty", 1e-3))
    max_iter = int(calibrator_cfg.get("max_iter", 100))
    tolerance = float(calibrator_cfg.get("tolerance", 1e-8))
    if clip_eps <= 0.0 or clip_eps >= 0.5:
        raise ValueError("Step 9 calibration clip_eps must satisfy 0 < clip_eps < 0.5")
    if max_iter <= 0:
        raise ValueError("Step 9 calibration max_iter must be positive")
    if tolerance <= 0.0:
        raise ValueError("Step 9 calibration tolerance must be positive")

    y = np.array(y_true, dtype=float)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Step 9 calibration requires both positive and negative labels in the fit split")

    base_prob = np.array(probabilities, dtype=float)
    feature = safe_logit(base_prob, clip_eps)
    target = np.where(
        y == 1.0,
        (positives + 1.0) / (positives + 2.0),
        1.0 / (negatives + 2.0),
    )

    scale = 1.0
    bias = 0.0
    converged = False
    iterations_completed = 0
    for iteration in range(1, max_iter + 1):
        iterations_completed = iteration
        logits = scale * feature + bias
        calibrated = safe_sigmoid(logits)
        weights = calibrated * (1.0 - calibrated)

        grad_scale = float(np.sum((calibrated - target) * feature) + l2_penalty * scale)
        grad_bias = float(np.sum(calibrated - target) + l2_penalty * bias)
        h11 = float(np.sum(weights * feature * feature) + l2_penalty)
        h12 = float(np.sum(weights * feature))
        h22 = float(np.sum(weights) + l2_penalty)

        damping = 0.0
        determinant = h11 * h22 - h12 * h12
        while determinant <= 1e-12:
            damping = 1e-6 if damping == 0.0 else damping * 10.0
            if damping > 1.0:
                break
            determinant = (h11 + damping) * (h22 + damping) - h12 * h12
        if damping > 0.0:
            h11 += damping
            h22 += damping
            determinant = h11 * h22 - h12 * h12
        if determinant <= 1e-12:
            break

        delta_scale = (h22 * grad_scale - h12 * grad_bias) / determinant
        delta_bias = (-h12 * grad_scale + h11 * grad_bias) / determinant
        scale -= float(delta_scale)
        bias -= float(delta_bias)
        if max(abs(delta_scale), abs(delta_bias)) <= tolerance:
            converged = True
            break

    calibrated_prob = apply_platt_scaling(base_prob, scale, bias, clip_eps)
    diagnostics = {
        "type": "platt_scaling",
        "fit_row_count": int(len(y)),
        "fit_positive_count": positives,
        "fit_negative_count": negatives,
        "clip_eps": clip_eps,
        "l2_penalty": l2_penalty,
        "max_iter": max_iter,
        "tolerance": tolerance,
        "iterations_completed": iterations_completed,
        "converged": converged,
        "parameter_scale": round(float(scale), 6),
        "parameter_bias": round(float(bias), 6),
        "train_logloss_before": round(float(step7.binary_logloss(y, base_prob)), 6),
        "train_logloss_after": round(float(step7.binary_logloss(y, calibrated_prob)), 6),
        "train_brier_before": round(float(brier_score(y, base_prob)), 6),
        "train_brier_after": round(float(brier_score(y, calibrated_prob)), 6),
        "train_probability_stats_before": probability_stats(base_prob),
        "train_probability_stats_after": probability_stats(calibrated_prob),
    }
    artifact = {
        "version": "2026-04-08",
        "calibrator_type": "platt_scaling",
        "input_score_space": "step7_prob_positive",
        "clip_eps": clip_eps,
        "parameter_scale": round(float(scale), 12),
        "parameter_bias": round(float(bias), 12),
    }
    return artifact, diagnostics


def apply_platt_scaling(
    probabilities: np.ndarray,
    scale: float,
    bias: float,
    clip_eps: float,
) -> np.ndarray:
    feature = safe_logit(np.array(probabilities, dtype=float), clip_eps)
    return safe_sigmoid(scale * feature + bias)


def choose_threshold(
    y_valid: np.ndarray,
    calibrated_valid_probabilities: np.ndarray,
    threshold_policy: dict,
) -> float:
    mode = str(threshold_policy.get("mode", "single_valid"))
    if mode != "single_valid":
        raise SystemExit(
            "Step 9 calibration currently supports only threshold_selection.mode = single_valid. "
            "Use the calibration policy branch for low-variance thresholding instead of bootstrap."
        )
    metric_name = str(threshold_policy["metric"])
    return float(
        step7.choose_threshold(
            y_valid,
            calibrated_valid_probabilities,
            metric_name,
            clone_threshold_policy(threshold_policy),
        )
    )


def resolve_primary_threshold(
    primary_threshold_source: str,
    selected_threshold: float,
    fixed_calibrated_threshold: float,
) -> float:
    if primary_threshold_source == "zh_valid_selected_threshold":
        return float(selected_threshold)
    if primary_threshold_source == "fixed_calibrated_threshold":
        return float(fixed_calibrated_threshold)
    raise SystemExit(
        "Unsupported Step 9 calibration primary_threshold_source: "
        f"{primary_threshold_source!r}. Supported values are "
        "'zh_valid_selected_threshold' and 'fixed_calibrated_threshold'."
    )


def resolve_experiment_threshold_policy(
    calibration_cfg: dict,
    experiment_cfg: dict,
    experiment_name: str,
) -> tuple[str, float]:
    override_cfg = (calibration_cfg.get("experiment_threshold_overrides", {}) or {}).get(experiment_name, {}) or {}
    primary_threshold_source = str(
        override_cfg.get(
            "primary_threshold_source",
            experiment_cfg.get("primary_threshold_source", calibration_cfg["primary_threshold_source"]),
        )
    )
    fixed_calibrated_threshold = float(
        override_cfg.get(
            "fixed_calibrated_threshold",
            experiment_cfg.get("fixed_calibrated_threshold", calibration_cfg["fixed_calibrated_threshold"]),
        )
    )
    return primary_threshold_source, fixed_calibrated_threshold


def main() -> None:
    args = parse_args()
    lgb = step7.require_lightgbm()
    step7_policy = step7.load_json(STEP7_POLICY_PATH)
    calibration_policy = step7.load_json(STEP9_CALIBRATION_POLICY_PATH)
    policy_step7_summary = calibration_policy.get("baseline_reference", {}).get("step7_summary")
    preferred_step7_summary_path = resolve_policy_path(policy_step7_summary, STEP7_SUMMARY_PATH)
    step7_summary_records = load_step7_summaries(
        preferred_step7_summary_path,
        allow_archive_fallback=not bool(policy_step7_summary),
    )

    selected_experiments = args.experiments or calibration_policy["default_experiments"]
    unknown_experiments = sorted(set(selected_experiments) - set(calibration_policy["experiments"].keys()))
    if unknown_experiments:
        raise SystemExit(f"Unknown Step 9 calibration experiment names: {unknown_experiments}")

    en_rows = step7.join_frozen_with_features(
        step7.load_csv(FROZEN_PATHS["en_content_train_pool"]),
        step7.load_csv(PAIR_FEATURE_PATHS["en_content_train_pool"]),
    )
    zh_rows = step7.join_frozen_with_features(
        step7.load_csv(FROZEN_PATHS["zh_target_strict"]),
        step7.load_csv(PAIR_FEATURE_PATHS["zh_target_strict"]),
    )

    source_test_rows = step7.select_rows(
        en_rows,
        calibration_policy["splits"]["source_test_split"],
        require_core_transfer=True,
    )
    zh_train_rows = step7.select_rows(
        zh_rows,
        calibration_policy["splits"]["target_train_split"],
        require_core_transfer=True,
    )
    zh_valid_rows = step7.select_rows(
        zh_rows,
        calibration_policy["splits"]["target_valid_split"],
        require_core_transfer=True,
    )
    zh_test_rows = step7.select_rows(
        zh_rows,
        calibration_policy["splits"]["target_test_split"],
        require_core_transfer=True,
    )

    step7.ensure_non_empty(zh_train_rows, "step9_calibration.zh_train")
    step7.ensure_non_empty(zh_valid_rows, "step9_calibration.zh_valid")
    step7.ensure_non_empty(zh_test_rows, "step9_calibration.zh_test")

    summary = {
        "schema_path": str(SCHEMA_PATH.relative_to(ROOT)),
        "step7_policy_path": str(STEP7_POLICY_PATH.relative_to(ROOT)),
        "step9_calibration_policy_path": str(STEP9_CALIBRATION_POLICY_PATH.relative_to(ROOT)),
        "step7_summary_path": str(preferred_step7_summary_path.relative_to(ROOT)),
        "step7_summary_resolution_mode": (
            "explicit_policy_path" if policy_step7_summary else "top_level_or_archive_fallback"
        ),
        "step7_summary_search_paths": [str(path.relative_to(ROOT)) for path, _payload in step7_summary_records],
        "input_dependencies": calibration_policy["input_dependencies"],
        "selected_experiments": selected_experiments,
        "fixed_datasets": {
            "source_test": dataset_summary(source_test_rows),
            "zh_train": dataset_summary(zh_train_rows),
            "zh_valid": dataset_summary(zh_valid_rows),
            "zh_test": dataset_summary(zh_test_rows),
        },
        "calibration_policy": {
            "fit_split": calibration_policy["calibration"]["fit_split"],
            "threshold_selection": calibration_policy["threshold_selection"],
            "default_primary_threshold_source": calibration_policy["calibration"]["primary_threshold_source"],
            "fixed_calibrated_threshold": calibration_policy["calibration"]["fixed_calibrated_threshold"],
            "experiment_threshold_overrides": calibration_policy["calibration"].get(
                "experiment_threshold_overrides", {}
            ),
        },
        "experiments": {},
    }

    threshold_policy = calibration_policy["threshold_selection"]
    model_template = str(step7_policy["output_templates"]["model"])

    for experiment_name in selected_experiments:
        experiment_cfg = calibration_policy["experiments"][experiment_name]
        primary_threshold_source, fixed_calibrated_threshold = resolve_experiment_threshold_policy(
            calibration_policy["calibration"],
            experiment_cfg,
            experiment_name,
        )
        base_step7_experiment_name = str(experiment_cfg["base_step7_experiment"])
        calibrator_cfg = calibration_policy["calibration"]["calibrators"][experiment_cfg["calibrator_type"]]
        base_step7_summary, base_step7_summary_path = resolve_step7_experiment_summary(
            base_step7_experiment_name,
            step7_summary_records,
        )
        base_step7_threshold = float(base_step7_summary["selected_threshold"])
        base_step7_model_path = ROOT / model_template.format(experiment_name=base_step7_experiment_name)
        if not base_step7_model_path.exists():
            raise SystemExit(
                "Step 9 calibration could not find the required Step 7 model artifact at "
                f"{base_step7_model_path}. Sync the Linux outputs before running this branch."
            )

        booster = lgb.Booster(model_file=str(base_step7_model_path))
        feature_names = list(booster.feature_name())
        if not feature_names:
            raise SystemExit(
                "Step 9 calibration could not recover feature names from the synchronized Step 7 model file "
                f"{base_step7_model_path}."
            )
        step7.validate_feature_columns(zh_train_rows, feature_names, f"{experiment_name}.zh_train")
        step7.validate_feature_columns(zh_valid_rows, feature_names, f"{experiment_name}.zh_valid")
        step7.validate_feature_columns(zh_test_rows, feature_names, f"{experiment_name}.zh_test")
        step7.validate_feature_columns(source_test_rows, feature_names, f"{experiment_name}.en_test")

        x_zh_train, y_zh_train = step7.rows_to_matrix(zh_train_rows, feature_names)
        x_zh_valid, y_zh_valid = step7.rows_to_matrix(zh_valid_rows, feature_names)
        x_zh_test, y_zh_test = step7.rows_to_matrix(zh_test_rows, feature_names)
        x_en_test, y_en_test = step7.rows_to_matrix(source_test_rows, feature_names)

        zh_train_base_prob = booster.predict(x_zh_train)
        zh_valid_base_prob = booster.predict(x_zh_valid)
        zh_test_base_prob = booster.predict(x_zh_test)
        en_test_base_prob = booster.predict(x_en_test)

        calibrator_artifact, calibrator_diagnostics = fit_platt_scaling(
            zh_train_base_prob,
            y_zh_train,
            calibrator_cfg,
        )
        calibrated_clip_eps = float(calibrator_artifact["clip_eps"])
        calibrated_scale = float(calibrator_artifact["parameter_scale"])
        calibrated_bias = float(calibrator_artifact["parameter_bias"])

        zh_train_calibrated_prob = apply_platt_scaling(
            zh_train_base_prob,
            calibrated_scale,
            calibrated_bias,
            calibrated_clip_eps,
        )
        zh_valid_calibrated_prob = apply_platt_scaling(
            zh_valid_base_prob,
            calibrated_scale,
            calibrated_bias,
            calibrated_clip_eps,
        )
        zh_test_calibrated_prob = apply_platt_scaling(
            zh_test_base_prob,
            calibrated_scale,
            calibrated_bias,
            calibrated_clip_eps,
        )
        en_test_calibrated_prob = apply_platt_scaling(
            en_test_base_prob,
            calibrated_scale,
            calibrated_bias,
            calibrated_clip_eps,
        )

        selected_threshold = choose_threshold(
            y_zh_valid,
            zh_valid_calibrated_prob,
            threshold_policy,
        )
        primary_threshold = resolve_primary_threshold(
            primary_threshold_source,
            selected_threshold,
            fixed_calibrated_threshold,
        )

        calibrator_artifact.update(
            {
                "base_step7_experiment": base_step7_experiment_name,
                "step7_model_path": str(base_step7_model_path.relative_to(ROOT)),
                "step7_summary_path": str(base_step7_summary_path.relative_to(ROOT)),
                "primary_threshold_source": primary_threshold_source,
                "selected_threshold": round(float(selected_threshold), 6),
                "primary_threshold": round(float(primary_threshold), 6),
                "fixed_calibrated_threshold": round(float(fixed_calibrated_threshold), 6),
            }
        )

        calibrator_output_path = output_path(
            calibration_policy["output_templates"]["calibrator"],
            experiment_name,
        )
        step7.write_json(calibrator_output_path, calibrator_artifact)

        zh_train_predictions = prediction_rows(
            zh_train_rows,
            zh_train_base_prob,
            zh_train_calibrated_prob,
            primary_threshold,
            fixed_calibrated_threshold,
            experiment_name,
        )
        zh_valid_predictions = prediction_rows(
            zh_valid_rows,
            zh_valid_base_prob,
            zh_valid_calibrated_prob,
            primary_threshold,
            fixed_calibrated_threshold,
            experiment_name,
        )
        zh_test_predictions = prediction_rows(
            zh_test_rows,
            zh_test_base_prob,
            zh_test_calibrated_prob,
            primary_threshold,
            fixed_calibrated_threshold,
            experiment_name,
        )
        en_test_predictions = prediction_rows(
            source_test_rows,
            en_test_base_prob,
            en_test_calibrated_prob,
            primary_threshold,
            fixed_calibrated_threshold,
            experiment_name,
        )

        zh_train_predictions_path = output_path(
            calibration_policy["output_templates"]["zh_train_predictions"],
            experiment_name,
        )
        zh_valid_predictions_path = output_path(
            calibration_policy["output_templates"]["zh_valid_predictions"],
            experiment_name,
        )
        zh_test_predictions_path = output_path(
            calibration_policy["output_templates"]["zh_test_predictions"],
            experiment_name,
        )
        en_test_predictions_path = output_path(
            calibration_policy["output_templates"]["en_test_predictions"],
            experiment_name,
        )
        step7.write_csv(zh_train_predictions_path, zh_train_predictions, prediction_fieldnames())
        step7.write_csv(zh_valid_predictions_path, zh_valid_predictions, prediction_fieldnames())
        step7.write_csv(zh_test_predictions_path, zh_test_predictions, prediction_fieldnames())
        step7.write_csv(en_test_predictions_path, en_test_predictions, prediction_fieldnames())

        summary["experiments"][experiment_name] = {
            "base_step7_experiment": base_step7_experiment_name,
            "resolved_from_step7_summary_path": str(base_step7_summary_path.relative_to(ROOT)),
            "step7_model_path": str(base_step7_model_path.relative_to(ROOT)),
            "feature_names": feature_names,
            "calibrator_type": experiment_cfg["calibrator_type"],
            "calibrator_fit_split": calibration_policy["calibration"]["fit_split"],
            "primary_threshold_source": primary_threshold_source,
            "selected_threshold": round(float(selected_threshold), 6),
            "fixed_calibrated_threshold": round(float(fixed_calibrated_threshold), 6),
            "base_step7_selected_threshold": round(float(base_step7_threshold), 6),
            "threshold_metric": threshold_policy["metric"],
            "threshold_selection_policy": threshold_policy,
            "base_zh_zero_shot_test_metrics": base_step7_summary.get("zh_zero_shot_test_metrics"),
            "base_test_metrics": base_step7_summary.get("test_metrics"),
            "calibrator_diagnostics": calibrator_diagnostics,
            "zh_train_metrics": step7.evaluate_probabilities(
                y_zh_train,
                zh_train_calibrated_prob,
                primary_threshold,
            ),
            "zh_train_metrics_selected_threshold": step7.evaluate_probabilities(
                y_zh_train,
                zh_train_calibrated_prob,
                selected_threshold,
            ),
            "zh_train_metrics_fixed_half_threshold": step7.evaluate_probabilities(
                y_zh_train,
                zh_train_calibrated_prob,
                fixed_calibrated_threshold,
            ),
            "zh_valid_metrics": step7.evaluate_probabilities(
                y_zh_valid,
                zh_valid_calibrated_prob,
                primary_threshold,
            ),
            "zh_valid_metrics_selected_threshold": step7.evaluate_probabilities(
                y_zh_valid,
                zh_valid_calibrated_prob,
                selected_threshold,
            ),
            "zh_valid_metrics_fixed_half_threshold": step7.evaluate_probabilities(
                y_zh_valid,
                zh_valid_calibrated_prob,
                fixed_calibrated_threshold,
            ),
            "zh_test_metrics": step7.evaluate_probabilities(
                y_zh_test,
                zh_test_calibrated_prob,
                primary_threshold,
            ),
            "zh_test_metrics_selected_threshold": step7.evaluate_probabilities(
                y_zh_test,
                zh_test_calibrated_prob,
                selected_threshold,
            ),
            "zh_test_metrics_fixed_half_threshold": step7.evaluate_probabilities(
                y_zh_test,
                zh_test_calibrated_prob,
                fixed_calibrated_threshold,
            ),
            "en_test_metrics": step7.evaluate_probabilities(
                y_en_test,
                en_test_calibrated_prob,
                primary_threshold,
            ),
            "en_test_metrics_selected_threshold": step7.evaluate_probabilities(
                y_en_test,
                en_test_calibrated_prob,
                selected_threshold,
            ),
            "en_test_metrics_fixed_half_threshold": step7.evaluate_probabilities(
                y_en_test,
                en_test_calibrated_prob,
                fixed_calibrated_threshold,
            ),
            "base_probability_stats": {
                "zh_train": probability_stats(zh_train_base_prob),
                "zh_valid": probability_stats(zh_valid_base_prob),
                "zh_test": probability_stats(zh_test_base_prob),
                "en_test": probability_stats(en_test_base_prob),
            },
            "calibrated_probability_stats": {
                "zh_train": probability_stats(zh_train_calibrated_prob),
                "zh_valid": probability_stats(zh_valid_calibrated_prob),
                "zh_test": probability_stats(zh_test_calibrated_prob),
                "en_test": probability_stats(en_test_calibrated_prob),
            },
            "artifacts": {
                "calibrator": str(calibrator_output_path.relative_to(ROOT)),
                "zh_train_predictions": str(zh_train_predictions_path.relative_to(ROOT)),
                "zh_valid_predictions": str(zh_valid_predictions_path.relative_to(ROOT)),
                "zh_test_predictions": str(zh_test_predictions_path.relative_to(ROOT)),
                "en_test_predictions": str(en_test_predictions_path.relative_to(ROOT)),
            },
            "primary_threshold": round(float(primary_threshold), 6),
        }

    step7.write_json(plain_output_path(calibration_policy["output_templates"]["summary"]), summary)


if __name__ == "__main__":
    main()
