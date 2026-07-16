#!/usr/bin/env python3
"""Evaluate Step23 item multi-instance representations with grouped target-train OOF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7
import step9_run_few_shot_adaptation as step9
import step15_v7_common as common


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step23_item_multi_instance_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_value(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_folds(rows: list[dict], fold_count: int, seed: int) -> dict[str, int]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["step23_component_id"]].append(row)
    if len(grouped) < fold_count:
        raise ValueError("Step23 has fewer target-train components than folds")
    totals = np.asarray([
        len(rows) / fold_count,
        sum(row["review_label"] == "positive" for row in rows) / fold_count,
        sum(row["review_label"] == "negative" for row in rows) / fold_count,
    ], dtype=float)
    records = []
    for component, component_rows in grouped.items():
        positives = sum(row["review_label"] == "positive" for row in component_rows)
        negatives = len(component_rows) - positives
        mass = max(
            len(component_rows) / max(totals[0], 1.0),
            positives / max(totals[1], 1.0),
            negatives / max(totals[2], 1.0),
        )
        tie = hashlib.sha256(f"{seed}|{component}".encode("utf-8")).hexdigest()
        records.append((component, len(component_rows), positives, negatives, mass, tie))
    records.sort(key=lambda item: (-item[4], -item[1], -item[2], item[5]))
    fold_counts = np.zeros((fold_count, 3), dtype=float)
    assignment = {}
    for component, count, positives, negatives, _mass, _tie in records:
        addition = np.asarray([count, positives, negatives], dtype=float)
        candidates = []
        for fold in range(fold_count):
            proposed = fold_counts.copy()
            proposed[fold] += addition
            error = (proposed - totals[None, :]) / np.maximum(totals[None, :], 1.0)
            candidates.append((float(np.sum(error**2)), fold_counts[fold, 0], fold_counts[fold, 1], fold))
        selected = min(candidates)[-1]
        assignment[component] = selected
        fold_counts[selected] += addition
    if any(row_count <= 0 or positives <= 0 or negatives <= 0 for row_count, positives, negatives in fold_counts):
        raise ValueError("Step23 grouped OOF emitted an empty or single-class fold")
    return assignment


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "row_count": int(len(labels)),
        "positive_count": int(np.sum(labels == 1.0)),
        "negative_count": int(np.sum(labels == 0.0)),
        "roc_auc": step7.roc_auc_score(labels, scores),
        "average_precision": step7.average_precision_score(labels, scores),
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def grouped_bootstrap(
    rows: list[dict], baseline: np.ndarray, candidate: np.ndarray, resamples: int, seed: int
) -> dict:
    component_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        component_indices[row["step23_component_id"]].append(index)
    components = sorted(component_indices)
    labels = common.labels_array(rows)
    rng = random.Random(seed)
    differences = []
    for _ in range(resamples):
        indices = []
        for _component in components:
            indices.extend(component_indices[rng.choice(components)])
        sampled_labels = labels[indices]
        if len(set(sampled_labels.tolist())) < 2:
            continue
        differences.append(
            step7.average_precision_score(sampled_labels, candidate[indices])
            - step7.average_precision_score(sampled_labels, baseline[indices])
        )
    if len(differences) < int(resamples * 0.95):
        raise ValueError("Step23 grouped bootstrap discarded too many single-class resamples")
    point = step7.average_precision_score(labels, candidate) - step7.average_precision_score(labels, baseline)
    return {
        "point_delta_average_precision": point,
        "grouped_bootstrap_95_ci": [percentile(differences, 0.025), percentile(differences, 0.975)],
        "bootstrap_probability_delta_gt_zero": sum(value > 0.0 for value in differences) / len(differences),
        "valid_resamples": len(differences),
    }


def matrix_for_rows(rows: list[dict], feature_index: dict[str, dict], feature_names: list[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        features = feature_index.get(row["pair_uid"])
        if features is None:
            raise ValueError(f"Step23 pair missing multi-instance features: {row['pair_uid']}")
        matrix.append([float(features[name]) for name in feature_names])
    return np.asarray(matrix, dtype=np.float64)


def fit_and_score(
    train_rows: list[dict],
    train_matrix_raw: np.ndarray,
    score_matrix_raw: np.ndarray,
    logistic_cfg: dict,
    weighting_cfg: dict,
) -> tuple[np.ndarray, dict]:
    imputation = common.fit_train_median_imputation(train_matrix_raw)
    train_matrix = common.apply_imputation(train_matrix_raw, imputation)
    score_matrix = common.apply_imputation(score_matrix_raw, imputation)
    _, standardization = step9.fit_standardization(train_matrix, True)
    labels = common.labels_array(train_rows)
    weights, weight_summary = common.factorized_evidence_weights(train_rows, weighting_cfg)
    artifact, _ = step9.fit_regularized_logistic(
        train_matrix,
        labels,
        logistic_cfg,
        sample_weight_multipliers=weights,
        sample_weight_target_total=float(len(train_rows)),
        precomputed_standardization=standardization,
    )
    return step9.apply_logistic_artifact_to_matrix(score_matrix, artifact), weight_summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Step23 prediction output is empty")
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    rendered = ("\ufeff" + buffer.getvalue()).encode("utf-8")
    if path.exists() and path.read_bytes() != rendered:
        raise ValueError(f"Refusing to overwrite different Step23 predictions: {path}")
    path.write_bytes(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    args = parser.parse_args()
    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    v7_policy_path = resolve(policy["inputs"]["v7_policy"])
    v7_policy = json.loads(v7_policy_path.read_text(encoding="utf-8"))
    output_root = resolve(policy["outputs_root"])
    output_cfg = policy["outputs"]
    feature_summary_path = output_root / output_cfg["pair_feature_summary"]
    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    feature_names = list(feature_summary["feature_names"])
    en_feature_path = output_root / output_cfg["pair_features_en"]
    zh_feature_path = output_root / output_cfg["pair_features_zh"]
    en_feature_index = {row["pair_uid"]: row for row in load_csv(en_feature_path)}
    zh_feature_index = {row["pair_uid"]: row for row in load_csv(zh_feature_path)}

    pools = common.load_joined_rows(v7_policy)
    en_rows = pools["en_content_train_pool"]
    zh_rows = pools["zh_target_strict"]
    assignments = {row["pair_uid"]: row for row in load_csv(resolve(policy["inputs"]["component_assignments"]))}
    for row in [*en_rows, *zh_rows]:
        assignment = assignments.get(row["pair_uid"])
        if assignment is None or assignment["split_name"] != row["split_name"]:
            raise ValueError(f"Step23 missing or inconsistent Step16I assignment: {row['pair_uid']}")
        if str(assignment.get("cross_split_component_leakage", "0")) == "1":
            raise ValueError(f"Step23 refuses a leaking component: {row['pair_uid']}")
        row["step23_component_id"] = assignment["recomputed_component_id"]
        row["v7_component_id"] = assignment["recomputed_component_id"]
    en_train = [row for row in en_rows if row["split_name"] == "train"]
    zh_train = [row for row in zh_rows if row["split_name"] == "train"]
    if len(en_train) != len(en_feature_index) or len(zh_train) != len(zh_feature_index):
        raise ValueError("Step23 feature rows do not exactly match canonical train rows")

    cosine_name = v7_policy["clean_semantic_encoder"]["output_feature"]
    en_aggregate = np.asarray([[float(row[cosine_name])] for row in en_train], dtype=np.float64)
    zh_aggregate = np.asarray([[float(row[cosine_name])] for row in zh_train], dtype=np.float64)
    en_item = matrix_for_rows(en_train, en_feature_index, feature_names)
    zh_item = matrix_for_rows(zh_train, zh_feature_index, feature_names)
    matrices = {
        "aggregate_redacted_e5_cosine": (en_aggregate, zh_aggregate),
        "item_multi_instance_only": (en_item, zh_item),
        "aggregate_plus_item_multi_instance": (
            np.concatenate([en_aggregate, en_item], axis=1),
            np.concatenate([zh_aggregate, zh_item], axis=1),
        ),
    }
    if list(matrices) != policy["evaluation"]["models"]:
        raise ValueError("Step23 model order/configuration drift")

    evaluation_cfg = policy["evaluation"]
    if not evaluation_cfg["valid_or_test_selection_forbidden"]:
        raise ValueError("Step23 valid/test selection prohibition was relaxed")
    expected_source_controls = [f"source_only_{name}" for name in matrices]
    if expected_source_controls != evaluation_cfg["source_only_controls"]:
        raise ValueError("Step23 source-only control order/configuration drift")
    fold_assignment = grouped_folds(zh_train, int(evaluation_cfg["fold_count"]), int(evaluation_cfg["fold_seed"]))
    logistic_cfg = dict(v7_policy["step9_latent_mixup"]["logistic"])
    weighting_cfg = v7_policy["factorized_evidence_weighting"]
    oof_scores = {name: np.full(len(zh_train), np.nan, dtype=float) for name in matrices}
    source_scores = {}
    source_weight_summaries = {}
    for model_name, (en_matrix, zh_matrix) in matrices.items():
        source_scores[model_name], source_weight_summaries[model_name] = fit_and_score(
            en_train, en_matrix, zh_matrix, logistic_cfg, weighting_cfg
        )

    fold_records = []
    for fold in range(int(evaluation_cfg["fold_count"])):
        held = np.asarray([
            index for index, row in enumerate(zh_train)
            if fold_assignment[row["step23_component_id"]] == fold
        ], dtype=int)
        held_set = set(held.tolist())
        train = np.asarray([index for index in range(len(zh_train)) if index not in held_set], dtype=int)
        train_rows = en_train + [zh_train[index] for index in train]
        for model_name, (en_matrix, zh_matrix) in matrices.items():
            train_matrix = np.concatenate([en_matrix, zh_matrix[train]], axis=0)
            oof_scores[model_name][held], _ = fit_and_score(
                train_rows, train_matrix, zh_matrix[held], logistic_cfg, weighting_cfg
            )
        fold_records.append({
            "fold": fold,
            "held_out_rows": len(held),
            "held_out_positives": int(sum(zh_train[index]["review_label"] == "positive" for index in held)),
            "held_out_negatives": int(sum(zh_train[index]["review_label"] == "negative" for index in held)),
            "held_out_component_count": len({zh_train[index]["step23_component_id"] for index in held}),
        })
    if any(np.any(~np.isfinite(scores)) for scores in oof_scores.values()):
        raise ValueError("Step23 OOF did not score every target-train row")

    labels = common.labels_array(zh_train)
    oof_metrics = {name: metrics(labels, scores) for name, scores in oof_scores.items()}
    source_metrics = {
        f"source_only_{name}": metrics(labels, scores) for name, scores in source_scores.items()
    }
    baseline_name = "aggregate_redacted_e5_cosine"
    bootstrap = {
        name: grouped_bootstrap(
            zh_train,
            oof_scores[baseline_name],
            scores,
            int(evaluation_cfg["grouped_bootstrap_resamples"]),
            int(evaluation_cfg["grouped_bootstrap_seed"]),
        )
        for name, scores in oof_scores.items()
        if name != baseline_name
    }
    candidate_names = [name for name in matrices if name != baseline_name]
    best_candidate_ap = max(float(oof_metrics[name]["average_precision"]) for name in candidate_names)
    tie_tolerance = float(evaluation_cfg["selection_tie_tolerance"])
    tied_candidates = {
        name for name in candidate_names
        if best_candidate_ap - float(oof_metrics[name]["average_precision"]) <= tie_tolerance
    }
    simplicity_order = list(evaluation_cfg["simplicity_tie_break_order"])
    if set(simplicity_order) != set(candidate_names):
        raise ValueError("Step23 simplicity tie-break order does not match candidate models")
    selected_name = next(name for name in simplicity_order if name in tied_candidates)
    baseline_ap = float(oof_metrics[baseline_name]["average_precision"])
    selected_ap = float(oof_metrics[selected_name]["average_precision"])

    slice_score_means = {}
    negative_slice_increases = []
    for evidence_type in ("template_clone_not_controller", "semantic_topic_not_controller"):
        indices = [index for index, row in enumerate(zh_train) if row.get("evidence_type") == evidence_type]
        if not indices:
            raise ValueError(f"Step23 expected target-train evidence slice is empty: {evidence_type}")
        baseline_mean = float(np.mean(oof_scores[baseline_name][indices]))
        selected_mean = float(np.mean(oof_scores[selected_name][indices]))
        increase = selected_mean - baseline_mean
        negative_slice_increases.append(increase)
        slice_score_means[evidence_type] = {
            "row_count": len(indices),
            "baseline_mean_score": baseline_mean,
            "selected_mean_score": selected_mean,
            "selected_minus_baseline": increase,
        }

    selected_bootstrap = bootstrap[selected_name]
    ap_gain = selected_ap - baseline_ap
    non_silver_indices = np.asarray([
        index for index, row in enumerate(zh_train) if not bool_value(row.get("silver_train_only"))
    ], dtype=int)
    strong_evidence_indices = np.asarray([
        index for index, row in enumerate(zh_train)
        if row["review_label"] == "negative"
        or row.get("evidence_type") in {
            "same_controller_direct_identifier",
            "same_controller_component_anchor",
        }
    ], dtype=int)
    sensitivity_metrics = {}
    for slice_name, indices in {
        "canonical_non_silver": non_silver_indices,
        "direct_component_positive_plus_all_negatives": strong_evidence_indices,
    }.items():
        slice_labels = labels[indices]
        if len(set(slice_labels.tolist())) < 2:
            raise ValueError(f"Step23 sensitivity slice is single-class: {slice_name}")
        baseline_metrics = metrics(slice_labels, oof_scores[baseline_name][indices])
        selected_metrics = metrics(slice_labels, oof_scores[selected_name][indices])
        sensitivity_metrics[slice_name] = {
            "baseline": baseline_metrics,
            "selected": selected_metrics,
            "selected_minus_baseline_ap": (
                selected_metrics["average_precision"] - baseline_metrics["average_precision"]
            ),
        }
    non_silver_ap_delta = sensitivity_metrics["canonical_non_silver"]["selected_minus_baseline_ap"]
    promotion = (
        ap_gain >= float(evaluation_cfg["minimum_ap_gain_over_aggregate"])
        and selected_bootstrap["grouped_bootstrap_95_ci"][0]
        >= float(evaluation_cfg["minimum_bootstrap_lower_bound"])
        and non_silver_ap_delta >= float(evaluation_cfg["minimum_non_silver_ap_delta"])
        and max(negative_slice_increases)
        <= float(evaluation_cfg["maximum_template_or_topic_mean_score_increase"])
    )

    prediction_rows = []
    for index, row in enumerate(zh_train):
        output = {
            "pair_uid": row["pair_uid"],
            "component_id": row["step23_component_id"],
            "review_label": row["review_label"],
            "evidence_type": row.get("evidence_type", ""),
            "fold": fold_assignment[row["step23_component_id"]],
        }
        for model_name in matrices:
            output[f"prob_oof_{model_name}"] = f"{oof_scores[model_name][index]:.12f}"
            output[f"prob_source_only_{model_name}"] = f"{source_scores[model_name][index]:.12f}"
        prediction_rows.append(output)
    prediction_path = output_root / output_cfg["oof_predictions"]
    write_csv(prediction_path, prediction_rows)

    input_hashes = {
        "policy": sha256_file(policy_path),
        "producer": sha256_file(Path(__file__)),
        "step7_dependency": sha256_file(Path(step7.__file__).resolve()),
        "step9_dependency": sha256_file(Path(step9.__file__).resolve()),
        "step15_v7_common_dependency": sha256_file(Path(common.__file__).resolve()),
        "v7_policy": sha256_file(v7_policy_path),
        "component_assignments": sha256_file(resolve(policy["inputs"]["component_assignments"])),
        "feature_summary": sha256_file(feature_summary_path),
        "features_en": sha256_file(en_feature_path),
        "features_zh": sha256_file(zh_feature_path),
        "v7_split_assignment": sha256_file(
            resolve(v7_policy["representative_validation"]["split_assignment_output"])
        ),
    }
    for pool_name, pool_cfg in v7_policy["pools"].items():
        for key in ("frozen_labels", "evidence_labels", "v7_pair_features"):
            input_hashes[f"{pool_name}:{key}"] = sha256_file(resolve(pool_cfg[key]))
    summary = {
        "step": "step23_item_multi_instance_grouped_oof_evaluation",
        "policy_version": policy["version"],
        "status": "target_train_grouped_oof_only",
        "valid_or_test_scores_used": False,
        "publication_holdout_untouched": True,
        "english_train_rows": len(en_train),
        "chinese_train_rows": len(zh_train),
        "chinese_train_components": len({row["step23_component_id"] for row in zh_train}),
        "feature_count": len(feature_names),
        "models": list(matrices),
        "source_only_metrics": source_metrics,
        "target_grouped_oof_metrics": oof_metrics,
        "grouped_bootstrap_vs_aggregate": bootstrap,
        "selection": {
            "selected_model": selected_name,
            "candidate_oof_average_precision": {
                name: oof_metrics[name]["average_precision"] for name in candidate_names
            },
            "best_candidate_oof_average_precision": best_candidate_ap,
            "tied_within_tolerance": [name for name in simplicity_order if name in tied_candidates],
            "tie_tolerance": tie_tolerance,
            "simplicity_tie_break_order": simplicity_order,
            "selection_scope": evaluation_cfg["selection_scope"],
            "selection_metric": evaluation_cfg["selection_metric"],
            "test_metrics_used": False,
        },
        "negative_slice_score_means": slice_score_means,
        "sensitivity_metrics": sensitivity_metrics,
        "promotion": {
            "selected_minus_aggregate_ap": ap_gain,
            "minimum_ap_gain_required": evaluation_cfg["minimum_ap_gain_over_aggregate"],
            "bootstrap_lower_bound": selected_bootstrap["grouped_bootstrap_95_ci"][0],
            "minimum_bootstrap_lower_bound": evaluation_cfg["minimum_bootstrap_lower_bound"],
            "non_silver_selected_minus_baseline_ap": non_silver_ap_delta,
            "minimum_non_silver_ap_delta": evaluation_cfg["minimum_non_silver_ap_delta"],
            "maximum_observed_template_or_topic_mean_score_increase": max(negative_slice_increases),
            "maximum_allowed_template_or_topic_mean_score_increase": evaluation_cfg[
                "maximum_template_or_topic_mean_score_increase"
            ],
            "eligible": promotion,
        },
        "folds": fold_records,
        "source_only_weight_summaries": source_weight_summaries,
        "input_hashes": input_hashes,
        "prediction_sha256": sha256_file(prediction_path),
    }
    summary_path = output_root / output_cfg["evaluation_summary"]
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if summary_path.exists() and summary_path.read_text(encoding="utf-8") != rendered:
        raise ValueError("Refusing to overwrite a different Step23 evaluation summary")
    summary_path.write_text(rendered, encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "selected_model": selected_name,
        "selected_minus_aggregate_ap": ap_gain,
        "promotion_eligible": promotion,
    }, indent=2))


if __name__ == "__main__":
    main()
