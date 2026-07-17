#!/usr/bin/env python3
"""Run preregistered source-only and grouped-OOF Step24 comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import step15_v7_common as v7_common
import step7_train_baseline_models as step7
import step9_run_few_shot_adaptation as step9
import step24_common as common


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "row_count": int(len(labels)),
        "positive_count": int(np.sum(labels == 1.0)),
        "negative_count": int(np.sum(labels == 0.0)),
        "positive_prevalence": float(np.mean(labels)),
        "roc_auc": step7.roc_auc_score(labels, scores),
        "average_precision": step7.average_precision_score(labels, scores),
    }


def grouped_bootstrap(
    rows: list[dict],
    baseline: np.ndarray,
    candidate: np.ndarray,
    resamples: int,
    seed: int,
) -> dict:
    component_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        component_indices[row["step24_component_id"]].append(index)
    components = sorted(component_indices)
    labels = np.asarray(
        [1.0 if row["review_label"] == "positive" else 0.0 for row in rows],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    deltas = []
    skipped = 0
    for _ in range(resamples):
        sampled_indices = []
        for _component in components:
            sampled_component = components[int(rng.integers(0, len(components)))]
            sampled_indices.extend(component_indices[sampled_component])
        sampled_indices = np.asarray(sampled_indices, dtype=int)
        sampled_labels = labels[sampled_indices]
        if len(np.unique(sampled_labels)) < 2:
            skipped += 1
            continue
        deltas.append(
            step7.average_precision_score(sampled_labels, candidate[sampled_indices])
            - step7.average_precision_score(sampled_labels, baseline[sampled_indices])
        )
    if len(deltas) < max(100, int(resamples * 0.90)):
        raise ValueError("Step24 grouped bootstrap produced too few valid resamples")
    values = np.asarray(deltas, dtype=float)
    point = step7.average_precision_score(labels, candidate) - step7.average_precision_score(
        labels, baseline
    )
    return {
        "resamples_requested": resamples,
        "resamples_completed": len(deltas),
        "single_class_resamples_skipped": skipped,
        "group_count": len(components),
        "point_delta_average_precision": float(point),
        "mean_delta_average_precision": float(np.mean(values)),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "probability_delta_gt_zero": float(np.mean(values > 0.0)),
    }


def matrix_for_rows(rows: list[dict], features: dict[str, dict], names: list[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        feature_row = features.get(row["pair_uid"])
        if feature_row is None:
            raise ValueError(f"Step24 pair feature is missing: {row['pair_uid']}")
        matrix.append([float(feature_row[name]) for name in names])
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Step24 pair feature matrix contains non-finite values")
    return matrix


def label_counts(rows: list[dict]) -> dict:
    positive_count = sum(row["review_label"] == "positive" for row in rows)
    negative_count = sum(row["review_label"] == "negative" for row in rows)
    return {
        "row_count": len(rows),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_prevalence": positive_count / len(rows),
    }


def fit_and_score(
    train_rows: list[dict],
    train_matrix: np.ndarray,
    score_matrix: np.ndarray,
    feature_names: list[str],
    logistic_cfg: dict,
    weighting_cfg: dict,
) -> tuple[np.ndarray, dict]:
    labels = v7_common.labels_array(train_rows)
    weights, weight_summary = v7_common.factorized_evidence_weights(train_rows, weighting_cfg)
    artifact, _ = step9.fit_regularized_logistic(
        train_matrix,
        labels,
        logistic_cfg,
        sample_weight_multipliers=weights,
        sample_weight_target_total=float(len(train_rows)),
    )
    if not artifact["solver_converged"]:
        raise ValueError(f"Step24 LR/L2 did not converge for features={feature_names}")
    return step9.apply_logistic_artifact_to_matrix(score_matrix, artifact), {
        "feature_names": feature_names,
        "train_row_count": len(train_rows),
        "train_positive_count": int(np.sum(labels == 1.0)),
        "train_negative_count": int(np.sum(labels == 0.0)),
        "train_pair_uid_sha256": hashlib.sha256(
            "\n".join(sorted(row["pair_uid"] for row in train_rows)).encode("utf-8")
        ).hexdigest(),
        "logistic_artifact": artifact,
        "weight_summary": weight_summary,
    }


def slice_metrics(rows: list[dict], score_map: dict[str, np.ndarray]) -> dict:
    slices = {
        "all_canonical_train_oof": np.arange(len(rows), dtype=int),
        "canonical_non_silver": np.asarray(
            [index for index, row in enumerate(rows) if not common.bool_value(row.get("silver_train_only"))],
            dtype=int,
        ),
        "silver_train_only_secondary": np.asarray(
            [index for index, row in enumerate(rows) if common.bool_value(row.get("silver_train_only"))],
            dtype=int,
        ),
        "direct_component_positive_plus_all_negatives": np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["review_label"] == "negative"
                or row["evidence_type"]
                in {"same_controller_direct_identifier", "same_controller_component_anchor"}
            ],
            dtype=int,
        ),
        "soft_positive_plus_all_negatives": np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["review_label"] == "negative"
                or row["evidence_type"] == "same_controller_style_structural_soft"
            ],
            dtype=int,
        ),
    }
    labels = v7_common.labels_array(rows)
    output = {}
    for slice_name, indices in slices.items():
        if indices.size == 0 or len(np.unique(labels[indices])) < 2:
            output[slice_name] = {
                "row_count": int(indices.size),
                "status": "not_estimable_both_classes_required",
            }
            continue
        output[slice_name] = {
            "row_count": int(indices.size),
            "positive_count": int(np.sum(labels[indices] == 1.0)),
            "negative_count": int(np.sum(labels[indices] == 0.0)),
            "models": {
                model_name: metrics(labels[indices], scores[indices])
                for model_name, scores in score_map.items()
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path = common.resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    common.validate_policy(policy)
    eval_cfg = policy["evaluation"]
    model_feature_sets = eval_cfg["model_feature_sets"]
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "primary_model": eval_cfg["primary_model"],
                    "matched_baseline_model": eval_cfg["matched_baseline_model"],
                    "fold_count": eval_cfg["fold_count"],
                    "valid_or_test_selection_forbidden": True,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    output_root = common.resolve(policy["outputs_root"])
    feature_summary_path = output_root / policy["outputs"]["pair_feature_summary"]
    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    if feature_summary.get("valid_test_pair_count") != 0:
        raise ValueError("Step24 pair features contain valid/test rows")
    feature_paths = {
        "en_content_train_pool": output_root / policy["outputs"]["pair_features_en"],
        "zh_target_strict": output_root / policy["outputs"]["pair_features_zh"],
    }
    feature_indices = {}
    for pool_name, path in feature_paths.items():
        feature_rows = common.load_csv(path)
        feature_index = {row["pair_uid"]: row for row in feature_rows}
        if len(feature_index) != len(feature_rows):
            raise ValueError(f"Step24 pair feature file contains duplicates: {path}")
        feature_indices[pool_name] = feature_index
    rows_by_pool = common.load_canonical_train_rows(policy)
    en_rows = rows_by_pool["en_content_train_pool"]
    zh_rows = rows_by_pool["zh_target_strict"]
    if {row["step24_component_id"] for row in en_rows} & {
        row["step24_component_id"] for row in zh_rows
    }:
        raise ValueError("Step24 source and target component IDs overlap")
    if {
        row[field]
        for row in en_rows
        for field in ("seller_uid_left", "seller_uid_right")
    } & {
        row[field]
        for row in zh_rows
        for field in ("seller_uid_left", "seller_uid_right")
    }:
        raise ValueError("Step24 source and target seller IDs overlap")

    v7_policy = json.loads(
        common.resolve(policy["inputs"]["v7_policy"]).read_text(encoding="utf-8")
    )
    weighting_cfg = v7_policy["factorized_evidence_weighting"]
    logistic_cfg = eval_cfg["logistic"]
    all_feature_names = feature_summary["feature_names"]
    en_all = matrix_for_rows(en_rows, feature_indices["en_content_train_pool"], all_feature_names)
    zh_all = matrix_for_rows(zh_rows, feature_indices["zh_target_strict"], all_feature_names)
    column_index = {name: index for index, name in enumerate(all_feature_names)}

    scores = {
        f"raw_{name}": zh_all[:, column_index[name]].copy()
        for name in eval_cfg["raw_score_controls"]
    }
    artifacts = {"source_only": {}, "target_grouped_oof": {}}
    source_only_scores = {}
    for model_name, feature_names in model_feature_sets.items():
        columns = [column_index[name] for name in feature_names]
        model_scores, artifact = fit_and_score(
            en_rows,
            en_all[:, columns],
            zh_all[:, columns],
            feature_names,
            logistic_cfg,
            weighting_cfg,
        )
        key = f"source_only_{model_name}"
        scores[key] = model_scores
        source_only_scores[model_name] = model_scores
        artifacts["source_only"][model_name] = artifact

    fold_assignment = common.balanced_component_folds(
        zh_rows, int(eval_cfg["fold_count"]), int(eval_cfg["fold_seed"])
    )
    oof_scores = {name: np.full(len(zh_rows), np.nan, dtype=float) for name in model_feature_sets}
    for fold in range(int(eval_cfg["fold_count"])):
        held_indices = np.asarray(
            [
                index
                for index, row in enumerate(zh_rows)
                if fold_assignment[row["step24_component_id"]] == fold
            ],
            dtype=int,
        )
        held_set = set(held_indices.tolist())
        train_indices = np.asarray(
            [index for index in range(len(zh_rows)) if index not in held_set], dtype=int
        )
        fold_train_rows = en_rows + [zh_rows[index] for index in train_indices]
        fold_all = np.vstack([en_all, zh_all[train_indices]])
        fold_record = {
            "held_out_row_count": int(len(held_indices)),
            "held_out_positive_count": int(
                sum(zh_rows[index]["review_label"] == "positive" for index in held_indices)
            ),
            "held_out_negative_count": int(
                sum(zh_rows[index]["review_label"] == "negative" for index in held_indices)
            ),
            "held_out_component_ids": sorted(
                {zh_rows[index]["step24_component_id"] for index in held_indices}
            ),
            "models": {},
        }
        if not fold_record["held_out_positive_count"] or not fold_record["held_out_negative_count"]:
            raise ValueError(f"Step24 held-out fold {fold} is single-class")
        for model_name, feature_names in model_feature_sets.items():
            columns = [column_index[name] for name in feature_names]
            model_scores, artifact = fit_and_score(
                fold_train_rows,
                fold_all[:, columns],
                zh_all[held_indices][:, columns],
                feature_names,
                logistic_cfg,
                weighting_cfg,
            )
            oof_scores[model_name][held_indices] = model_scores
            fold_record["models"][model_name] = artifact
        artifacts["target_grouped_oof"][f"fold_{fold}"] = fold_record
    if any(np.any(~np.isfinite(values)) for values in oof_scores.values()):
        raise ValueError("Step24 failed to score every target OOF row exactly once")
    for model_name, values in oof_scores.items():
        scores[f"target_oof_{model_name}"] = values

    labels = v7_common.labels_array(zh_rows)
    metric_table = {name: metrics(labels, values) for name, values in scores.items()}
    primary_name = eval_cfg["primary_model"]
    baseline_name = eval_cfg["matched_baseline_model"]
    primary = oof_scores[primary_name]
    baseline = oof_scores[baseline_name]
    source_primary = source_only_scores[primary_name]
    source_baseline = source_only_scores[baseline_name]
    target_bootstrap = grouped_bootstrap(
        zh_rows,
        baseline,
        primary,
        int(eval_cfg["grouped_bootstrap_resamples"]),
        int(eval_cfg["grouped_bootstrap_seed"]),
    )
    source_bootstrap = grouped_bootstrap(
        zh_rows,
        source_baseline,
        source_primary,
        int(eval_cfg["grouped_bootstrap_resamples"]),
        int(eval_cfg["grouped_bootstrap_seed"]) + 1,
    )
    slices = slice_metrics(
        zh_rows,
        {
            name: values
            for name, values in scores.items()
            if name.startswith("target_oof_") or name.startswith("source_only_")
        },
    )
    slice_primary_key = f"target_oof_{primary_name}"
    slice_baseline_key = f"target_oof_{baseline_name}"
    source_slice_primary_key = f"source_only_{primary_name}"
    source_slice_baseline_key = f"source_only_{baseline_name}"

    def slice_ap_delta(slice_name: str, candidate_key: str, baseline_key: str) -> float:
        model_metrics = slices[slice_name]["models"]
        return float(
            model_metrics[candidate_key]["average_precision"]
            - model_metrics[baseline_key]["average_precision"]
        )

    direct_component_indices = np.asarray(
        [
            index
            for index, row in enumerate(zh_rows)
            if row["review_label"] == "positive"
            and row["evidence_type"]
            in {"same_controller_direct_identifier", "same_controller_component_anchor"}
        ],
        dtype=int,
    )
    if direct_component_indices.size == 0:
        raise ValueError("Step24 has no direct/component positive sensitivity rows")
    direct_component_mean_delta = float(
        np.mean(primary[direct_component_indices]) - np.mean(baseline[direct_component_indices])
    )
    negative_tails = {}
    maximum_tail_increases = {"mean": -math.inf, "q95": -math.inf, "top_decile_mean": -math.inf}
    for evidence_type in ("template_clone_not_controller", "semantic_topic_not_controller"):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(zh_rows)
                if row["review_label"] == "negative" and row["evidence_type"] == evidence_type
            ],
            dtype=int,
        )
        if indices.size == 0:
            raise ValueError(f"Step24 expected negative slice is empty: {evidence_type}")
        baseline_tail = common.negative_tail_metrics(baseline[indices])
        primary_tail = common.negative_tail_metrics(primary[indices])
        deltas = {
            key: float(primary_tail[key] - baseline_tail[key])
            for key in ("mean", "q95", "top_decile_mean")
        }
        for key, value in deltas.items():
            maximum_tail_increases[key] = max(maximum_tail_increases[key], value)
        negative_tails[evidence_type] = {
            "baseline": baseline_tail,
            "primary": primary_tail,
            "primary_minus_baseline": deltas,
        }

    gates_cfg = policy["promotion_gates"]
    target_ap_delta = float(metric_table[slice_primary_key]["average_precision"] - metric_table[slice_baseline_key]["average_precision"])
    source_ap_delta = float(
        metric_table[f"source_only_{primary_name}"]["average_precision"]
        - metric_table[f"source_only_{baseline_name}"]["average_precision"]
    )
    non_silver_delta = slice_ap_delta(
        "canonical_non_silver", slice_primary_key, slice_baseline_key
    )
    strong_delta = slice_ap_delta(
        "direct_component_positive_plus_all_negatives",
        slice_primary_key,
        slice_baseline_key,
    )
    source_non_silver_delta = slice_ap_delta(
        "canonical_non_silver", source_slice_primary_key, source_slice_baseline_key
    )
    source_strong_delta = slice_ap_delta(
        "direct_component_positive_plus_all_negatives",
        source_slice_primary_key,
        source_slice_baseline_key,
    )
    gate_results = {
        "target_oof_ap_gain": target_ap_delta
        >= float(gates_cfg["minimum_target_oof_ap_gain_over_e5_control"]),
        "target_oof_bootstrap_lower_bound": target_bootstrap["ci95_lower"]
        >= float(gates_cfg["minimum_target_oof_grouped_bootstrap_lower_bound"]),
        "source_only_ap_gain": source_ap_delta
        >= float(gates_cfg["minimum_source_only_ap_gain_over_e5_control"]),
        "source_only_bootstrap_lower_bound": source_bootstrap["ci95_lower"]
        >= float(gates_cfg["minimum_source_only_grouped_bootstrap_lower_bound"]),
        "source_only_non_silver_non_degradation": source_non_silver_delta
        >= float(gates_cfg["minimum_source_only_non_silver_ap_delta"]),
        "source_only_direct_component_non_degradation": source_strong_delta
        >= float(
            gates_cfg[
                "minimum_source_only_direct_component_plus_all_negatives_ap_delta"
            ]
        ),
        "non_silver_non_degradation": non_silver_delta
        >= float(gates_cfg["minimum_non_silver_ap_delta"]),
        "direct_component_slice_non_degradation": strong_delta
        >= float(gates_cfg["minimum_direct_component_plus_all_negatives_ap_delta"]),
        "direct_component_positive_mean_non_degradation": direct_component_mean_delta
        >= float(gates_cfg["minimum_direct_component_positive_mean_score_delta"]),
        "template_topic_mean_tail": maximum_tail_increases["mean"]
        <= float(gates_cfg["maximum_template_or_topic_mean_score_increase"]),
        "template_topic_q95_tail": maximum_tail_increases["q95"]
        <= float(gates_cfg["maximum_template_or_topic_q95_score_increase"]),
        "template_topic_top_decile_tail": maximum_tail_increases["top_decile_mean"]
        <= float(gates_cfg["maximum_template_or_topic_top_decile_mean_score_increase"]),
    }
    promotion_eligible = all(gate_results.values())

    prediction_rows = []
    ordered_score_names = sorted(scores)
    for index, row in enumerate(zh_rows):
        prediction_rows.append(
            {
                "pair_uid": row["pair_uid"],
                "pool": row["step24_pool"],
                "split_name": "train_oof",
                "component_id": row["step24_component_id"],
                "fold": fold_assignment[row["step24_component_id"]],
                "review_label": row["review_label"],
                "evidence_type": row["evidence_type"],
                "silver_train_only": row.get("silver_train_only", ""),
                **{name: f"{float(scores[name][index]):.12f}" for name in ordered_score_names},
            }
        )
    predictions_path = output_root / policy["outputs"]["oof_predictions"]
    common.write_csv_immutable(predictions_path, prediction_rows)
    artifacts_payload = {
        "step": "step24_model_artifacts",
        "version": policy["version"],
        "feature_names": all_feature_names,
        "fold_assignment": fold_assignment,
        "artifacts": artifacts,
        "policy_sha256": common.sha256_file(policy_path),
        "feature_summary_sha256": common.sha256_file(feature_summary_path),
        "producer_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    artifacts_payload["payload_sha256"] = common.canonical_hash(artifacts_payload)
    artifacts_path = output_root / policy["outputs"]["model_artifacts"]
    common.write_json_immutable(artifacts_path, artifacts_payload)

    summary = {
        "step": "step24_content_independent_authorship_evaluation",
        "version": policy["version"],
        "status": "pass",
        "promotion_eligible": promotion_eligible,
        "promotion_scope": gates_cfg["promotion_scope"],
        "method_selected_from_valid_or_test": False,
        "valid_test_labels_or_scores_used_for_selection": False,
        "valid_test_pair_features_scored": 0,
        "local_synthetic_label_count": 0,
        "encoder_parameters_updated": False,
        "dataset_counts": {
            "english_train": label_counts(en_rows),
            "chinese_train_oof": label_counts(zh_rows),
            "chinese_component_count": len({row["step24_component_id"] for row in zh_rows}),
            "chinese_silver_positive_count": sum(
                row["review_label"] == "positive" and common.bool_value(row.get("silver_train_only"))
                for row in zh_rows
            ),
            "chinese_non_silver_positive_count": sum(
                row["review_label"] == "positive" and not common.bool_value(row.get("silver_train_only"))
                for row in zh_rows
            ),
        },
        "model_metrics": metric_table,
        "slice_metrics": slices,
        "paired_grouped_bootstrap": {
            "target_grouped_oof_primary_minus_e5": target_bootstrap,
            "source_only_primary_minus_e5": source_bootstrap,
        },
        "negative_tail_audit": negative_tails,
        "key_deltas": {
            "target_oof_primary_minus_e5_ap": target_ap_delta,
            "source_only_primary_minus_e5_ap": source_ap_delta,
            "source_only_non_silver_primary_minus_e5_ap": source_non_silver_delta,
            "source_only_direct_component_primary_minus_e5_ap": source_strong_delta,
            "canonical_non_silver_primary_minus_e5_ap": non_silver_delta,
            "direct_component_slice_primary_minus_e5_ap": strong_delta,
            "direct_component_positive_mean_score_delta": direct_component_mean_delta,
            "maximum_template_topic_tail_increases": maximum_tail_increases,
        },
        "gate_results": gate_results,
        "gate_thresholds": gates_cfg,
        "outputs": {
            "predictions": str(predictions_path.relative_to(common.ROOT)).replace("\\", "/"),
            "model_artifacts": str(artifacts_path.relative_to(common.ROOT)).replace("\\", "/"),
        },
        "policy_sha256": common.sha256_file(policy_path),
        "feature_summary_sha256": common.sha256_file(feature_summary_path),
        "predictions_sha256": common.sha256_file(predictions_path),
        "model_artifacts_sha256": common.sha256_file(artifacts_path),
        "producer_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    summary["summary_sha256"] = common.canonical_hash(summary)
    summary_path = output_root / policy["outputs"]["evaluation_summary"]
    common.write_json_immutable(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "promotion_eligible": promotion_eligible,
                "target_oof_primary_minus_e5_ap": target_ap_delta,
                "source_only_primary_minus_e5_ap": source_ap_delta,
                "target_bootstrap_ci95": [
                    target_bootstrap["ci95_lower"],
                    target_bootstrap["ci95_upper"],
                ],
                "source_bootstrap_ci95": [
                    source_bootstrap["ci95_lower"],
                    source_bootstrap["ci95_upper"],
                ],
                "summary": str(summary_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
