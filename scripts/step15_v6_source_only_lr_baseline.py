#!/usr/bin/env python3
"""Train the preregistered Step15-v6 source-only L2 logistic baseline."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7
import step15_train_incremental_hard_negative as step15


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step15_v6_paper_hardening_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--seed", action="append", dest="seeds", type=int)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[np.ndarray, float, dict]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=x_train.shape[1])
    b = 0.0
    mw = np.zeros_like(w)
    vw = np.zeros_like(w)
    mb = 0.0
    vb = 0.0
    best_w = w.copy()
    best_b = b
    best_metric = float("-inf")
    best_epoch = 0
    stale = 0
    lr = float(cfg["learning_rate"])
    l2 = float(cfg["l2_weight"])
    max_epochs = int(cfg["max_epochs"])
    patience = int(cfg["patience_epochs"])
    weight_sum = max(float(weights.sum()), 1e-12)
    for epoch in range(1, max_epochs + 1):
        prob = step15.sigmoid(x_train @ w + b)
        residual = (prob - y_train) * weights / weight_sum
        grad_w = x_train.T @ residual + l2 * w
        grad_b = float(np.sum(residual))
        mw = 0.9 * mw + 0.1 * grad_w
        vw = 0.999 * vw + 0.001 * (grad_w**2)
        mb = 0.9 * mb + 0.1 * grad_b
        vb = 0.999 * vb + 0.001 * (grad_b**2)
        mw_hat = mw / (1.0 - 0.9**epoch)
        vw_hat = vw / (1.0 - 0.999**epoch)
        mb_hat = mb / (1.0 - 0.9**epoch)
        vb_hat = vb / (1.0 - 0.999**epoch)
        w -= lr * mw_hat / (np.sqrt(vw_hat) + 1e-8)
        b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
        valid_prob = step15.sigmoid(x_valid @ w + b)
        metric = step7.average_precision_score(y_valid, valid_prob)
        metric_value = float(metric) if metric is not None else float("-inf")
        if metric_value > best_metric + 1e-10:
            best_metric = metric_value
            best_w = w.copy()
            best_b = b
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best_w, best_b, {
        "best_epoch": best_epoch,
        "trained_epoch_count": epoch,
        "best_source_valid_average_precision": round(best_metric, 8),
        "initialization_seed": seed,
    }


def prediction_rows(rows: list[dict], probabilities: np.ndarray, threshold: float, model_id: str) -> list[dict]:
    result = []
    for row, probability in zip(rows, probabilities, strict=True):
        result.append(
            {
                "model_id": model_id,
                "experiment_name": model_id,
                "pair_uid": row["pair_uid"],
                "split_component_id": row.get("split_component_id", ""),
                "review_label": row["review_label"],
                "y_true": step15.label_to_int(row["review_label"]),
                "prob_positive": float(probability),
                "threshold": float(threshold),
                "pred_positive": int(probability >= threshold),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    policy_path = step15.resolve_path(args.policy)
    policy = step7.load_json(policy_path)
    cfg = policy["source_only_lr_baseline"]
    seeds = args.seeds or policy["training"]["default_seeds"]
    if args.validate_config_only:
        if cfg["threshold_selection_domain"] != "en_content_train_pool_valid_only":
            raise SystemExit("Source-only LR threshold must be selected on English validation only")
        if cfg["feature_set"] != "strict_clean_30d":
            raise SystemExit("Source-only LR must use strict_clean_30d")
        if bool(cfg.get("target_identity_labels_used_for_training", True)):
            raise SystemExit("Source-only LR must never use target identity labels for training")
        if bool(cfg.get("strict_zero_shot_without_target_covariates", True)):
            raise SystemExit(
                "Source-only LR must disclose its unlabeled target train-reference preprocessing"
            )
        print(json.dumps({"status": "pass", "seeds": seeds, "config": cfg}, indent=2))
        return

    inductive_manifest = step15.validate_inductive_feature_lineage(policy_path, policy)
    input_manifest = step15.build_input_manifest(
        policy_path,
        policy,
        extra_paths=[Path(__file__).resolve()],
    )
    if inductive_manifest is not None:
        input_manifest["inductive_feature_manifest_sha256"] = inductive_manifest["manifest_sha256"]
    summary_path = ROOT / cfg["output_summary"]
    if summary_path.exists():
        previous = step7.load_json(summary_path)
        previous_manifest_sha = (previous.get("input_manifest") or {}).get("manifest_sha256")
        if previous.get("policy_version") != policy.get("version") or (
            previous_manifest_sha != input_manifest.get("manifest_sha256")
        ):
            raise ValueError(
                "Refusing to overwrite the source-only LR baseline across a different code/data manifest"
            )

    rows_by_pool = {
        pool_name: step15.load_pool(pool_name, pool_cfg)
        for pool_name, pool_cfg in policy["pools"].items()
    }
    feature_names = policy["feature_sets"][cfg["feature_set"]]
    step15.validate_features(rows_by_pool, feature_names)
    source_rows = rows_by_pool["en_content_train_pool"]
    target_rows = rows_by_pool["zh_target_strict"]
    train_rows = step15.select_eval_rows(source_rows, cfg["source_train_split"])
    source_valid_rows = step15.select_eval_rows(source_rows, cfg["source_valid_split"])
    source_test_rows = step15.select_eval_rows(source_rows, cfg["source_test_split"])
    target_test_rows = step15.select_eval_rows(target_rows, cfg["target_test_split"])
    if not train_rows or not source_valid_rows or not source_test_rows or not target_test_rows:
        raise ValueError("Source-only LR requires non-empty source train/valid/test and target test")

    standardizer = step15.fit_standardizer_bundle(train_rows, feature_names, {})
    x_train = step15.apply_standardizer_bundle(train_rows, feature_names, standardizer)
    x_source_valid = step15.apply_standardizer_bundle(source_valid_rows, feature_names, standardizer)
    x_source_test = step15.apply_standardizer_bundle(source_test_rows, feature_names, standardizer)
    x_target_test = step15.apply_standardizer_bundle(target_test_rows, feature_names, standardizer)
    y_train = step15.y_from_rows(train_rows)
    y_source_valid = step15.y_from_rows(source_valid_rows)
    y_source_test = step15.y_from_rows(source_test_rows)
    y_target_test = step15.y_from_rows(target_test_rows)

    runs = []
    for seed in seeds:
        weights = np.ones(len(train_rows), dtype=float)
        weights, component_diag = step15.apply_component_inverse_sqrt_weights(weights, train_rows)
        weights, row_diag = step15.apply_row_training_sample_weights(weights, train_rows)
        weights, class_diag = step15.apply_class_balance_multipliers(weights, y_train)
        w, b, diagnostics = fit_logistic(
            x_train,
            y_train,
            weights,
            x_source_valid,
            y_source_valid,
            cfg,
            int(seed),
        )
        source_valid_prob = step15.sigmoid(x_source_valid @ w + b)
        source_test_prob = step15.sigmoid(x_source_test @ w + b)
        target_test_prob = step15.sigmoid(x_target_test @ w + b)
        threshold = step7.choose_threshold(
            y_source_valid,
            source_valid_prob,
            policy["threshold_selection"]["metric"],
            policy,
        )
        model_id = f"{cfg['experiment_name']}_seed_{seed}"
        prediction_path = ROOT / cfg["target_predictions_template"].format(seed=seed)
        artifact_path = ROOT / cfg["artifact_template"].format(seed=seed)
        predictions = prediction_rows(target_test_rows, target_test_prob, threshold, model_id)
        step15.atomic_write_csv(prediction_path, predictions, list(predictions[0]))
        artifact = {
            "model_id": model_id,
            "feature_names": feature_names,
            "standardizer_bundle": standardizer,
            "weights": [round(float(value), 12) for value in w],
            "bias": round(float(b), 12),
            "training_diagnostics": {
                **diagnostics,
                "component_weights": component_diag,
                "row_weights": row_diag,
                "class_weights": class_diag,
            },
        }
        step15.atomic_write_json(artifact_path, artifact)
        runs.append(
            {
                "model_id": model_id,
                "seed": int(seed),
                "source_train_row_count": len(train_rows),
                "threshold_selection_scope": cfg["threshold_selection_domain"],
                "threshold": round(float(threshold), 8),
                "source_valid_metrics": step7.evaluate_probabilities(y_source_valid, source_valid_prob, threshold),
                "source_test_metrics": step7.evaluate_probabilities(y_source_test, source_test_prob, threshold),
                "target_test_metrics": step7.evaluate_probabilities(y_target_test, target_test_prob, threshold),
                "output_paths": {
                    "artifact": str(artifact_path.relative_to(ROOT)),
                    "target_predictions": str(prediction_path.relative_to(ROOT)),
                },
            }
        )
    summary = {
        "step": "step15_v6_source_only_lr_baseline",
        "policy": str(policy_path.relative_to(ROOT)),
        "policy_version": policy["version"],
        "experiment_name": cfg["experiment_name"],
        "scientific_role": cfg["scientific_role"],
        "strict_zero_shot_without_target_covariates": bool(
            cfg["strict_zero_shot_without_target_covariates"]
        ),
        "target_identity_labels_used_for_training": bool(
            cfg["target_identity_labels_used_for_training"]
        ),
        "target_split_membership_used_for_frozen_reference_statistics": bool(
            cfg["target_split_membership_used_for_frozen_reference_statistics"]
        ),
        "target_test_role": policy["evaluation_boundary"]["zh_test_role"],
        "input_manifest": input_manifest,
        "runs": runs,
    }
    step15.atomic_write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path.relative_to(ROOT)), "run_count": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
