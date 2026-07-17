#!/usr/bin/env python3
"""Evaluate Step25 raw versus template-decontaminated authorship representations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import step15_v7_common as v7_common
import step24_common as step24
import step24_evaluate_content_independent_authorship as step24_eval
import step25_common as common


def load_feature_index(path: Path) -> dict[str, dict]:
    rows = step24.load_csv(path)
    index = {row["pair_uid"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Step25 pair feature file contains duplicates: {path}")
    return index


def feature_matrix(rows: list[dict], index: dict[str, dict], names: list[str]) -> np.ndarray:
    return step24_eval.matrix_for_rows(rows, index, names)


def fold_metrics(rows: list[dict], scores: dict[str, np.ndarray], assignment: dict[str, int]) -> dict:
    labels = v7_common.labels_array(rows)
    output = {}
    for fold in sorted(set(assignment.values())):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if assignment[row["step24_component_id"]] == fold
            ],
            dtype=int,
        )
        output[f"fold_{fold}"] = {
            "row_count": int(len(indices)),
            "positive_count": int(np.sum(labels[indices] == 1.0)),
            "negative_count": int(np.sum(labels[indices] == 0.0)),
            "models": {
                name: step24_eval.metrics(labels[indices], value[indices])
                for name, value in scores.items()
            },
        }
    return output


def fit_grouped_oof(
    rows: list[dict],
    matrix: np.ndarray,
    feature_names: list[str],
    model_sets: dict[str, list[str]],
    logistic_cfg: dict,
    weighting_cfg: dict,
    fold_count: int,
    fold_seed: int,
    source_rows: list[dict] | None = None,
    source_matrix: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict, dict[str, int]]:
    assignment = step24.balanced_component_folds(rows, fold_count, fold_seed)
    column_index = {name: index for index, name in enumerate(feature_names)}
    oof = {name: np.full(len(rows), np.nan, dtype=float) for name in model_sets}
    artifacts = {}
    for fold in range(fold_count):
        held = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if assignment[row["step24_component_id"]] == fold
            ],
            dtype=int,
        )
        held_set = set(held.tolist())
        train = np.asarray(
            [index for index in range(len(rows)) if index not in held_set], dtype=int
        )
        fold_rows = [rows[index] for index in train]
        fold_matrix = matrix[train]
        if source_rows is not None:
            if source_matrix is None:
                raise ValueError("Step25 source rows were provided without a matrix")
            fold_rows = source_rows + fold_rows
            fold_matrix = np.vstack([source_matrix, fold_matrix])
        record = {
            "held_out_row_count": int(len(held)),
            "held_out_positive_count": int(
                sum(rows[index]["review_label"] == "positive" for index in held)
            ),
            "held_out_negative_count": int(
                sum(rows[index]["review_label"] == "negative" for index in held)
            ),
            "held_out_component_ids": sorted(
                {rows[index]["step24_component_id"] for index in held}
            ),
            "models": {},
        }
        if not record["held_out_positive_count"] or not record["held_out_negative_count"]:
            raise ValueError(f"Step25 grouped fold {fold} is single-class")
        for model_name, names in model_sets.items():
            columns = [column_index[name] for name in names]
            score, artifact = step24_eval.fit_and_score(
                fold_rows,
                fold_matrix[:, columns],
                matrix[held][:, columns],
                names,
                logistic_cfg,
                weighting_cfg,
            )
            oof[model_name][held] = score
            record["models"][model_name] = artifact
        artifacts[f"fold_{fold}"] = record
    if any(np.any(~np.isfinite(value)) for value in oof.values()):
        raise ValueError("Step25 failed to score every OOF row exactly once")
    return oof, artifacts, assignment


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    """Return average-tie empirical percentiles in [0, 1]."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        percentile = average_rank / max(len(values) - 1, 1)
        result[order[cursor:end]] = percentile
        cursor = end
    return result


def pairwise_violation_rate(negative_scores: np.ndarray, positive_scores: np.ndarray) -> float:
    comparisons = negative_scores[:, None] - positive_scores[None, :]
    return float(np.mean((comparisons > 0.0) + 0.5 * (comparisons == 0.0)))


def tail_audit(
    rows: list[dict],
    baseline: np.ndarray,
    primary: np.ndarray,
) -> tuple[dict, dict[str, float]]:
    output = {}
    maximums = {}
    baseline_rank = rank_percentiles(baseline)
    primary_rank = rank_percentiles(primary)
    strong_positive_indices = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["review_label"] == "positive"
            and row["evidence_type"]
            in {"same_controller_direct_identifier", "same_controller_component_anchor"}
        ],
        dtype=int,
    )
    if strong_positive_indices.size == 0:
        raise ValueError("Step25 tail audit requires direct/component positives")
    for evidence_type in (
        "template_clone_not_controller",
        "semantic_topic_not_controller",
        "public_contact_or_url_noise",
    ):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["review_label"] == "negative" and row["evidence_type"] == evidence_type
            ],
            dtype=int,
        )
        if indices.size == 0:
            raise ValueError(f"Step25 required negative slice is empty: {evidence_type}")
        base = step24.negative_tail_metrics(baseline[indices])
        candidate = step24.negative_tail_metrics(primary[indices])
        delta = {
            key: float(candidate[key] - base[key])
            for key in ("mean", "q95", "top_decile_mean")
        }
        baseline_rank_tail = {
            "mean_rank_percentile": float(np.mean(baseline_rank[indices])),
            "q95_rank_percentile": float(np.quantile(baseline_rank[indices], 0.95)),
            "top_decile_exposure": float(np.mean(baseline_rank[indices] >= 0.90)),
            "vs_strong_positive_violation_rate": pairwise_violation_rate(
                baseline[indices], baseline[strong_positive_indices]
            ),
        }
        primary_rank_tail = {
            "mean_rank_percentile": float(np.mean(primary_rank[indices])),
            "q95_rank_percentile": float(np.quantile(primary_rank[indices], 0.95)),
            "top_decile_exposure": float(np.mean(primary_rank[indices] >= 0.90)),
            "vs_strong_positive_violation_rate": pairwise_violation_rate(
                primary[indices], primary[strong_positive_indices]
            ),
        }
        rank_delta = {
            key: float(primary_rank_tail[key] - baseline_rank_tail[key])
            for key in baseline_rank_tail
        }
        output[evidence_type] = {
            "baseline_raw_style": base,
            "primary_decontaminated_style": candidate,
            "primary_minus_baseline": delta,
            "baseline_rank_tail": baseline_rank_tail,
            "primary_rank_tail": primary_rank_tail,
            "primary_minus_baseline_rank_tail": rank_delta,
        }
        for key, value in delta.items():
            maximums[f"{evidence_type}:{key}"] = value
        for key, value in rank_delta.items():
            maximums[f"{evidence_type}:{key}"] = value
    return output, maximums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path, policy, step24_policy = common.load_policy(args.policy)
    eval_cfg = policy["evaluation"]
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "boundary": eval_cfg["boundary"],
                    "primary_model": eval_cfg["primary_model"],
                    "matched_baseline_model": eval_cfg["matched_baseline_model"],
                    "publication_promotion_allowed": False,
                    "valid_test_selection_forbidden": True,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    output_root = common.resolve(policy["outputs_root"])
    pair_summary_path = output_root / policy["outputs"]["pair_feature_summary"]
    pair_summary = json.loads(pair_summary_path.read_text(encoding="utf-8"))
    if pair_summary["valid_test_pair_count"] != 0:
        raise ValueError("Step25 pair feature summary contains valid/test pairs")
    rows_by_pool = common.load_rows(policy, step24_policy)
    en_rows = rows_by_pool["en_content_train_pool"]
    zh_rows = rows_by_pool["zh_target_strict"]
    en_index = load_feature_index(output_root / policy["outputs"]["pair_features_en"])
    zh_index = load_feature_index(output_root / policy["outputs"]["pair_features_zh"])
    feature_names = pair_summary["feature_names"]
    en_matrix = feature_matrix(en_rows, en_index, feature_names)
    zh_matrix = feature_matrix(zh_rows, zh_index, feature_names)
    column_index = {name: index for index, name in enumerate(feature_names)}
    model_sets = eval_cfg["model_feature_sets"]
    v7_policy = json.loads(
        common.resolve(policy["inputs"]["v7_policy"]).read_text(encoding="utf-8")
    )
    weighting_cfg = v7_policy["factorized_evidence_weighting"]
    logistic_cfg = eval_cfg["logistic"]

    source_scores = {}
    source_artifacts = {}
    for model_name, names in model_sets.items():
        columns = [column_index[name] for name in names]
        score, artifact = step24_eval.fit_and_score(
            en_rows,
            en_matrix[:, columns],
            zh_matrix[:, columns],
            names,
            logistic_cfg,
            weighting_cfg,
        )
        source_scores[model_name] = score
        source_artifacts[model_name] = artifact

    en_oof, en_oof_artifacts, en_assignment = fit_grouped_oof(
        en_rows,
        en_matrix,
        feature_names,
        model_sets,
        logistic_cfg,
        weighting_cfg,
        int(eval_cfg["fold_count"]),
        int(eval_cfg["fold_seed"]) + 101,
    )
    target_oof, target_artifacts, zh_assignment = fit_grouped_oof(
        zh_rows,
        zh_matrix,
        feature_names,
        model_sets,
        logistic_cfg,
        weighting_cfg,
        int(eval_cfg["fold_count"]),
        int(eval_cfg["fold_seed"]),
        source_rows=en_rows,
        source_matrix=en_matrix,
    )
    primary_name = eval_cfg["primary_model"]
    baseline_name = eval_cfg["matched_baseline_model"]
    labels = v7_common.labels_array(zh_rows)
    metrics = {}
    all_scores = {}
    for name, values in source_scores.items():
        key = f"source_only_{name}"
        metrics[key] = step24_eval.metrics(labels, values)
        all_scores[key] = values
    for name, values in target_oof.items():
        key = f"target_oof_{name}"
        metrics[key] = step24_eval.metrics(labels, values)
        all_scores[key] = values
    source_primary = source_scores[primary_name]
    source_baseline = source_scores[baseline_name]
    target_primary = target_oof[primary_name]
    target_baseline = target_oof[baseline_name]
    source_bootstrap = step24_eval.grouped_bootstrap(
        zh_rows,
        source_baseline,
        source_primary,
        int(eval_cfg["grouped_bootstrap_resamples"]),
        int(eval_cfg["grouped_bootstrap_seed"]),
    )
    target_bootstrap = step24_eval.grouped_bootstrap(
        zh_rows,
        target_baseline,
        target_primary,
        int(eval_cfg["grouped_bootstrap_resamples"]),
        int(eval_cfg["grouped_bootstrap_seed"]) + 1,
    )
    slices = step24_eval.slice_metrics(zh_rows, all_scores)

    def slice_delta(scope: str, slice_name: str) -> float:
        table = slices[slice_name]["models"]
        return float(
            table[f"{scope}_{primary_name}"]["average_precision"]
            - table[f"{scope}_{baseline_name}"]["average_precision"]
        )

    source_ap_delta = float(
        metrics[f"source_only_{primary_name}"]["average_precision"]
        - metrics[f"source_only_{baseline_name}"]["average_precision"]
    )
    target_ap_delta = float(
        metrics[f"target_oof_{primary_name}"]["average_precision"]
        - metrics[f"target_oof_{baseline_name}"]["average_precision"]
    )
    step24_evaluation_path = common.resolve(policy["inputs"]["step24_outputs_root"]) / step24_policy[
        "outputs"
    ]["evaluation_summary"]
    step24_evaluation = json.loads(step24_evaluation_path.read_text(encoding="utf-8"))
    baseline_reproduction = {}
    reproduction_pairs = {
        "source_only": (
            metrics[f"source_only_{baseline_name}"],
            step24_evaluation["model_metrics"]["source_only_style_only_lr_l2_control"],
        ),
        "target_oof": (
            metrics[f"target_oof_{baseline_name}"],
            step24_evaluation["model_metrics"]["target_oof_style_only_lr_l2_control"],
        ),
    }
    for scope, (observed, expected) in reproduction_pairs.items():
        deltas = {
            metric_name: float(observed[metric_name] - expected[metric_name])
            for metric_name in ("roc_auc", "average_precision")
        }
        if any(abs(value) > 1e-10 for value in deltas.values()):
            raise ValueError(
                f"Step25 raw-style baseline does not reproduce frozen Step24 {scope}: {deltas}"
            )
        baseline_reproduction[scope] = {
            "expected_step24_model": f"{scope}_style_only_lr_l2_control",
            "metric_deltas": deltas,
            "tolerance": 1e-10,
            "status": "pass",
        }
    source_tails, source_tail_deltas = tail_audit(
        zh_rows, source_baseline, source_primary
    )
    target_tails, target_tail_deltas = tail_audit(
        zh_rows, target_baseline, target_primary
    )
    reliable_fraction = float(
        pair_summary["records"]["zh_target_strict"]["reliable_pair_fraction"]
    )
    gates = policy["d0_continuation_gates"]
    gate_results = {
        "source_only_ap_non_degradation": source_ap_delta
        >= float(gates["minimum_source_only_ap_delta_over_raw_style"]),
        "source_only_bootstrap_lower_bound": source_bootstrap["ci95_lower"]
        >= float(gates["minimum_source_only_grouped_bootstrap_lower_bound"]),
        "target_oof_ap_non_degradation": target_ap_delta
        >= float(gates["minimum_target_oof_ap_delta_over_raw_style"]),
        "target_oof_bootstrap_lower_bound": target_bootstrap["ci95_lower"]
        >= float(gates["minimum_target_oof_grouped_bootstrap_lower_bound"]),
        "source_only_non_silver_non_degradation": slice_delta(
            "source_only", "canonical_non_silver"
        )
        >= float(gates["minimum_source_only_non_silver_ap_delta"]),
        "source_only_direct_component_non_degradation": slice_delta(
            "source_only", "direct_component_positive_plus_all_negatives"
        )
        >= float(gates["minimum_source_only_direct_component_ap_delta"]),
        "target_non_silver_non_degradation": slice_delta(
            "target_oof", "canonical_non_silver"
        )
        >= float(gates["minimum_target_non_silver_ap_delta"]),
        "target_direct_component_non_degradation": slice_delta(
            "target_oof", "direct_component_positive_plus_all_negatives"
        )
        >= float(gates["minimum_target_direct_component_ap_delta"]),
        "source_template_mean_rank_reduction": source_tail_deltas[
            "template_clone_not_controller:mean_rank_percentile"
        ]
        <= float(gates["maximum_template_clone_mean_rank_percentile_delta"]),
        "source_template_q95_rank_reduction": source_tail_deltas[
            "template_clone_not_controller:q95_rank_percentile"
        ]
        <= float(gates["maximum_template_clone_q95_rank_percentile_delta"]),
        "source_template_top_decile_exposure_reduction": source_tail_deltas[
            "template_clone_not_controller:top_decile_exposure"
        ]
        <= float(gates["maximum_template_clone_top_decile_exposure_delta"]),
        "source_template_strong_positive_violation_reduction": source_tail_deltas[
            "template_clone_not_controller:vs_strong_positive_violation_rate"
        ]
        <= float(gates["maximum_template_clone_vs_strong_positive_violation_delta"]),
        "target_template_mean_rank_reduction": target_tail_deltas[
            "template_clone_not_controller:mean_rank_percentile"
        ]
        <= float(gates["maximum_template_clone_mean_rank_percentile_delta"]),
        "target_template_strong_positive_violation_reduction": target_tail_deltas[
            "template_clone_not_controller:vs_strong_positive_violation_rate"
        ]
        <= float(gates["maximum_template_clone_vs_strong_positive_violation_delta"]),
        "target_public_noise_rank_not_worse": target_tail_deltas[
            "public_contact_or_url_noise:mean_rank_percentile"
        ]
        <= float(gates["maximum_public_noise_mean_rank_percentile_delta"]),
        "target_semantic_topic_rank_not_worse": target_tail_deltas[
            "semantic_topic_not_controller:mean_rank_percentile"
        ]
        <= float(gates["maximum_semantic_topic_mean_rank_percentile_delta"]),
        "reliable_pair_coverage": reliable_fraction
        >= float(gates["minimum_reliable_pair_fraction"]),
    }
    d1_candidate_eligible = all(gate_results.values())

    zh_prediction_rows = []
    for index, row in enumerate(zh_rows):
        zh_prediction_rows.append(
            {
                "pair_uid": row["pair_uid"],
                "pool": row["step25_pool"],
                "split_name": "train_oof_d0",
                "component_id": row["step25_component_id"],
                "fold": zh_assignment[row["step24_component_id"]],
                "review_label": row["review_label"],
                "evidence_type": row["evidence_type"],
                "silver_train_only": row.get("silver_train_only", ""),
                **{
                    f"source_only_{name}": f"{float(values[index]):.12f}"
                    for name, values in source_scores.items()
                },
                **{
                    f"target_oof_{name}": f"{float(values[index]):.12f}"
                    for name, values in target_oof.items()
                },
            }
        )
    en_prediction_rows = []
    for index, row in enumerate(en_rows):
        en_prediction_rows.append(
            {
                "pair_uid": row["pair_uid"],
                "pool": row["step25_pool"],
                "split_name": "train_oof_d0",
                "component_id": row["step25_component_id"],
                "fold": en_assignment[row["step24_component_id"]],
                "review_label": row["review_label"],
                "evidence_type": row["evidence_type"],
                "silver_train_only": row.get("silver_train_only", ""),
                **{
                    f"source_oof_{name}": f"{float(values[index]):.12f}"
                    for name, values in en_oof.items()
                },
            }
        )
    zh_predictions_path = output_root / policy["outputs"]["oof_predictions_zh"]
    en_predictions_path = output_root / policy["outputs"]["oof_predictions_en"]
    step24.write_csv_immutable(zh_predictions_path, zh_prediction_rows)
    step24.write_csv_immutable(en_predictions_path, en_prediction_rows)
    artifacts_payload = {
        "step": "step25_model_artifacts",
        "version": policy["version"],
        "boundary": "d0_current_canonical_train",
        "feature_names": feature_names,
        "source_only": source_artifacts,
        "english_grouped_oof": en_oof_artifacts,
        "target_grouped_oof": target_artifacts,
        "english_fold_assignment": en_assignment,
        "target_fold_assignment": zh_assignment,
        "policy_sha256": step24.sha256_file(policy_path),
        "pair_feature_summary_sha256": step24.sha256_file(pair_summary_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    artifacts_payload["payload_sha256"] = step24.canonical_hash(artifacts_payload)
    artifacts_path = output_root / policy["outputs"]["model_artifacts"]
    step24.write_json_immutable(artifacts_path, artifacts_payload)
    summary = {
        "step": "step25_template_decontaminated_authorship_evaluation",
        "version": policy["version"],
        "status": "pass",
        "boundary": "d0_current_canonical_train",
        "hypothesis_informed_retrospective_analysis": True,
        "d1_candidate_eligible": d1_candidate_eligible,
        "publication_promotion_eligible": False,
        "method_selected_from_valid_or_test": False,
        "valid_test_labels_scores_or_pair_features_used": False,
        "local_synthetic_label_count": 0,
        "encoder_parameters_updated": False,
        "dataset_counts": {
            "english_train": step24_eval.label_counts(en_rows),
            "chinese_train_oof": step24_eval.label_counts(zh_rows),
            "english_component_count": len(set(en_assignment)),
            "chinese_component_count": len(set(zh_assignment)),
            "chinese_silver_positive_count": sum(
                row["review_label"] == "positive" and step24.bool_value(row.get("silver_train_only"))
                for row in zh_rows
            ),
            "chinese_non_silver_positive_count": sum(
                row["review_label"] == "positive" and not step24.bool_value(row.get("silver_train_only"))
                for row in zh_rows
            ),
        },
        "model_metrics": metrics,
        "step24_raw_style_baseline_reproduction": baseline_reproduction,
        "slice_metrics": slices,
        "fold_metrics": {
            "english_oof": fold_metrics(en_rows, en_oof, en_assignment),
            "target_oof": fold_metrics(zh_rows, target_oof, zh_assignment),
        },
        "paired_grouped_bootstrap": {
            "source_only_decontaminated_minus_raw_style": source_bootstrap,
            "target_oof_decontaminated_minus_raw_style": target_bootstrap,
        },
        "negative_tail_audit": {
            "source_only": source_tails,
            "target_oof": target_tails,
        },
        "key_deltas": {
            "source_only_primary_minus_raw_style_ap": source_ap_delta,
            "target_oof_primary_minus_raw_style_ap": target_ap_delta,
            "source_only_non_silver_ap": slice_delta("source_only", "canonical_non_silver"),
            "source_only_direct_component_ap": slice_delta(
                "source_only", "direct_component_positive_plus_all_negatives"
            ),
            "target_non_silver_ap": slice_delta("target_oof", "canonical_non_silver"),
            "target_direct_component_ap": slice_delta(
                "target_oof", "direct_component_positive_plus_all_negatives"
            ),
            "source_tail_deltas": source_tail_deltas,
            "target_tail_deltas": target_tail_deltas,
            "reliable_pair_fraction": reliable_fraction,
        },
        "d0_continuation_gate_results": gate_results,
        "d0_continuation_gate_thresholds": gates,
        "outputs": {
            "zh_predictions": str(zh_predictions_path.relative_to(common.ROOT)).replace(
                "\\", "/"
            ),
            "en_predictions": str(en_predictions_path.relative_to(common.ROOT)).replace(
                "\\", "/"
            ),
            "model_artifacts": str(artifacts_path.relative_to(common.ROOT)).replace(
                "\\", "/"
            ),
        },
        "policy_sha256": step24.sha256_file(policy_path),
        "pair_feature_summary_sha256": step24.sha256_file(pair_summary_path),
        "step24_evaluation_sha256": step24.sha256_file(step24_evaluation_path),
        "zh_predictions_sha256": step24.sha256_file(zh_predictions_path),
        "en_predictions_sha256": step24.sha256_file(en_predictions_path),
        "model_artifacts_sha256": step24.sha256_file(artifacts_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    summary["summary_sha256"] = step24.canonical_hash(summary)
    summary_path = output_root / policy["outputs"]["evaluation_summary"]
    step24.write_json_immutable(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "d1_candidate_eligible": d1_candidate_eligible,
                "publication_promotion_eligible": False,
                "source_only_primary_minus_raw_style_ap": source_ap_delta,
                "target_oof_primary_minus_raw_style_ap": target_ap_delta,
                "failed_d0_gates": [key for key, value in gate_results.items() if not value],
                "summary": str(summary_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
