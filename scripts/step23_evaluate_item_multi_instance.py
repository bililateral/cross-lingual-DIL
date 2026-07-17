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


def negative_tail_metrics(scores: np.ndarray) -> dict:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("Step23 negative-tail metrics require at least one score")
    top_count = max(1, int(math.ceil(values.size * 0.10)))
    ordered = np.sort(values)
    return {
        "mean": float(np.mean(values)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "top_decile_mean": float(np.mean(ordered[-top_count:])),
        "maximum": float(np.max(values)),
        "row_count": int(values.size),
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
    feature_names: list[str],
) -> tuple[np.ndarray, dict, dict]:
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
    model_artifact = {
        "feature_names": list(feature_names),
        "imputation": imputation,
        "logistic_artifact": artifact,
        "weight_summary": weight_summary,
        "train_row_count": len(train_rows),
        "train_pair_uid_sha256": hashlib.sha256(
            "\n".join(sorted(row["pair_uid"] for row in train_rows)).encode("utf-8")
        ).hexdigest(),
    }
    return (
        step9.apply_logistic_artifact_to_matrix(score_matrix, artifact),
        weight_summary,
        model_artifact,
    )


def assert_domain_component_isolation(en_rows: list[dict], zh_rows: list[dict]) -> dict:
    en_components = {row["step23_component_id"] for row in en_rows}
    zh_components = {row["step23_component_id"] for row in zh_rows}
    component_overlap = sorted(en_components & zh_components)
    if component_overlap:
        raise ValueError(f"Step23 source/target component overlap: {component_overlap[0]}")
    en_sellers = {
        row[key]
        for row in en_rows
        for key in ("seller_uid_left", "seller_uid_right")
    }
    zh_sellers = {
        row[key]
        for row in zh_rows
        for key in ("seller_uid_left", "seller_uid_right")
    }
    seller_overlap = sorted(en_sellers & zh_sellers)
    if seller_overlap:
        raise ValueError(f"Step23 source/target seller overlap: {seller_overlap[0]}")
    return {
        "source_target_component_overlap": 0,
        "source_target_seller_overlap": 0,
    }


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


def write_json_immutable(path: Path, payload: dict) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different Step23 JSON artifact: {path}")
    path.write_text(rendered, encoding="utf-8")


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

    evaluation_cfg = policy["evaluation"]
    if not evaluation_cfg["valid_or_test_selection_forbidden"]:
        raise ValueError("Step23 valid/test selection prohibition was relaxed")
    if not evaluation_cfg["primary_model_preregistered"] or not evaluation_cfg["candidate_selection_forbidden"]:
        raise ValueError("Step23 v2 requires a preregistered primary without candidate selection")
    isolation = assert_domain_component_isolation(en_train, zh_train)
    model_feature_sets = {
        name: list(names) for name, names in evaluation_cfg["model_feature_sets"].items()
    }
    produced_feature_names = set(feature_names)
    for model_name, names in model_feature_sets.items():
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Step23 model feature set is empty or duplicated: {model_name}")
        missing = sorted(set(names) - produced_feature_names)
        if missing:
            raise ValueError(f"Step23 model feature set has missing features for {model_name}: {missing}")
    primary_name = evaluation_cfg["primary_model"]
    baseline_name = evaluation_cfg["matched_baseline_model"]
    if primary_name not in model_feature_sets or baseline_name not in model_feature_sets:
        raise ValueError("Step23 primary or matched baseline is absent from model_feature_sets")
    if not set(model_feature_sets[baseline_name]).issubset(model_feature_sets[primary_name]):
        raise ValueError("Step23 primary must contain every matched aggregate control feature")
    matrices = {
        model_name: (
            matrix_for_rows(en_train, en_feature_index, names),
            matrix_for_rows(zh_train, zh_feature_index, names),
        )
        for model_name, names in model_feature_sets.items()
    }
    fold_assignment = grouped_folds(zh_train, int(evaluation_cfg["fold_count"]), int(evaluation_cfg["fold_seed"]))
    logistic_cfg = dict(v7_policy["step9_latent_mixup"]["logistic"])
    weighting_cfg = v7_policy["factorized_evidence_weighting"]
    oof_scores = {name: np.full(len(zh_train), np.nan, dtype=float) for name in matrices}
    source_scores = {}
    source_weight_summaries = {}
    source_artifacts = {}
    for model_name, (en_matrix, zh_matrix) in matrices.items():
        (
            source_scores[model_name],
            source_weight_summaries[model_name],
            source_artifacts[model_name],
        ) = fit_and_score(
            en_train,
            en_matrix,
            zh_matrix,
            logistic_cfg,
            weighting_cfg,
            model_feature_sets[model_name],
        )

    fold_records = []
    oof_fold_artifacts = {name: [] for name in matrices}
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
            scores, _weight_summary, artifact = fit_and_score(
                train_rows,
                train_matrix,
                zh_matrix[held],
                logistic_cfg,
                weighting_cfg,
                model_feature_sets[model_name],
            )
            oof_scores[model_name][held] = scores
            artifact["fold"] = fold
            artifact["held_out_component_ids"] = sorted(
                {zh_train[index]["step23_component_id"] for index in held}
            )
            oof_fold_artifacts[model_name].append(artifact)
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
    baseline_ap = float(oof_metrics[baseline_name]["average_precision"])
    primary_ap = float(oof_metrics[primary_name]["average_precision"])

    negative_slice_tail_metrics = {}
    negative_mean_increases = []
    negative_q95_increases = []
    negative_top_decile_increases = []
    for evidence_type in ("template_clone_not_controller", "semantic_topic_not_controller"):
        indices = [index for index, row in enumerate(zh_train) if row.get("evidence_type") == evidence_type]
        if not indices:
            raise ValueError(f"Step23 expected target-train evidence slice is empty: {evidence_type}")
        baseline_tail = negative_tail_metrics(oof_scores[baseline_name][indices])
        primary_tail = negative_tail_metrics(oof_scores[primary_name][indices])
        deltas = {
            key: primary_tail[key] - baseline_tail[key]
            for key in ("mean", "q90", "q95", "top_decile_mean", "maximum")
        }
        negative_mean_increases.append(deltas["mean"])
        negative_q95_increases.append(deltas["q95"])
        negative_top_decile_increases.append(deltas["top_decile_mean"])
        negative_slice_tail_metrics[evidence_type] = {
            "baseline": baseline_tail,
            "primary": primary_tail,
            "primary_minus_baseline": deltas,
        }

    primary_bootstrap = bootstrap[primary_name]
    ap_gain = primary_ap - baseline_ap
    non_silver_indices = np.asarray([
        index for index, row in enumerate(zh_train) if not bool_value(row.get("silver_train_only"))
    ], dtype=int)
    silver_indices = np.asarray([
        index for index, row in enumerate(zh_train) if bool_value(row.get("silver_train_only"))
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
        "silver_train_only_secondary": silver_indices,
        "direct_component_positive_plus_all_negatives": strong_evidence_indices,
    }.items():
        slice_labels = labels[indices]
        if len(set(slice_labels.tolist())) < 2:
            raise ValueError(f"Step23 sensitivity slice is single-class: {slice_name}")
        baseline_metrics = metrics(slice_labels, oof_scores[baseline_name][indices])
        primary_metrics = metrics(slice_labels, oof_scores[primary_name][indices])
        sensitivity_metrics[slice_name] = {
            "baseline": baseline_metrics,
            "primary": primary_metrics,
            "primary_minus_baseline_ap": (
                primary_metrics["average_precision"] - baseline_metrics["average_precision"]
            ),
        }
    non_silver_ap_delta = sensitivity_metrics["canonical_non_silver"]["primary_minus_baseline_ap"]
    strong_evidence_ap_delta = sensitivity_metrics[
        "direct_component_positive_plus_all_negatives"
    ]["primary_minus_baseline_ap"]
    direct_component_positive_indices = np.asarray([
        index for index, row in enumerate(zh_train)
        if row["review_label"] == "positive"
        and row.get("evidence_type") in {
            "same_controller_direct_identifier",
            "same_controller_component_anchor",
        }
    ], dtype=int)
    if direct_component_positive_indices.size == 0:
        raise ValueError("Step23 has no direct/component positive sensitivity rows")
    direct_component_positive_mean_delta = float(
        np.mean(oof_scores[primary_name][direct_component_positive_indices])
        - np.mean(oof_scores[baseline_name][direct_component_positive_indices])
    )
    non_silver_rows = [zh_train[index] for index in non_silver_indices]
    non_silver_bootstrap = grouped_bootstrap(
        non_silver_rows,
        oof_scores[baseline_name][non_silver_indices],
        oof_scores[primary_name][non_silver_indices],
        int(evaluation_cfg["grouped_bootstrap_resamples"]),
        int(evaluation_cfg["grouped_bootstrap_seed"]) + 1,
    )
    label_quality = {
        "all_train": metrics(labels, oof_scores[baseline_name]),
        "canonical_non_silver": {
            "row_count": int(non_silver_indices.size),
            "positive_count": int(np.sum(labels[non_silver_indices] == 1.0)),
            "negative_count": int(np.sum(labels[non_silver_indices] == 0.0)),
        },
        "silver_train_only": {
            "row_count": int(silver_indices.size),
            "positive_count": int(np.sum(labels[silver_indices] == 1.0)),
            "negative_count": int(np.sum(labels[silver_indices] == 0.0)),
        },
        "direct_component_positive_count": int(direct_component_positive_indices.size),
        "all_label_oof_interpretation": "silver_supported_internal_development_only",
    }
    promotion = (
        ap_gain >= float(evaluation_cfg["minimum_ap_gain_over_matched_aggregate"])
        and primary_bootstrap["grouped_bootstrap_95_ci"][0]
        >= float(evaluation_cfg["minimum_bootstrap_lower_bound"])
        and non_silver_ap_delta >= float(evaluation_cfg["minimum_non_silver_ap_delta"])
        and strong_evidence_ap_delta >= float(evaluation_cfg["minimum_strong_evidence_ap_delta"])
        and direct_component_positive_mean_delta
        >= float(evaluation_cfg["minimum_direct_component_positive_mean_score_delta"])
        and max(negative_mean_increases)
        <= float(evaluation_cfg["maximum_template_or_topic_mean_score_increase"])
        and max(negative_q95_increases)
        <= float(evaluation_cfg["maximum_template_or_topic_q95_score_increase"])
        and max(negative_top_decile_increases)
        <= float(evaluation_cfg["maximum_template_or_topic_top_decile_mean_score_increase"])
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

    final_train_rows = en_train + zh_train
    final_train_artifacts = {}
    for model_name, (en_matrix, zh_matrix) in matrices.items():
        final_matrix = np.concatenate([en_matrix, zh_matrix], axis=0)
        _scores, _weight_summary, artifact = fit_and_score(
            final_train_rows,
            final_matrix,
            final_matrix,
            logistic_cfg,
            weighting_cfg,
            model_feature_sets[model_name],
        )
        final_train_artifacts[model_name] = artifact
    artifact_payload = {
        "step": "step23_v2_frozen_model_artifacts",
        "policy_version": policy["version"],
        "primary_model": primary_name,
        "matched_baseline_model": baseline_name,
        "valid_or_test_scores_used": False,
        "publication_holdout_untouched": True,
        "model_feature_sets": model_feature_sets,
        "source_only_artifacts": source_artifacts,
        "target_oof_fold_artifacts": oof_fold_artifacts,
        "final_all_train_artifacts": final_train_artifacts,
        "fold_assignment": dict(sorted(fold_assignment.items())),
        "input_hashes": input_hashes,
        "promotion_eligible": promotion,
    }
    artifact_path = output_root / output_cfg["model_artifacts"]
    write_json_immutable(artifact_path, artifact_payload)
    summary = {
        "step": "step23_item_multi_instance_grouped_oof_evaluation",
        "policy_version": policy["version"],
        "status": "target_train_grouped_oof_only",
        "valid_or_test_scores_used": False,
        "publication_holdout_untouched": True,
        "english_train_rows": len(en_train),
        "chinese_train_rows": len(zh_train),
        "chinese_train_components": len({row["step23_component_id"] for row in zh_train}),
        "domain_isolation": isolation,
        "feature_count": len(feature_names),
        "models": {
            name: {"feature_count": len(model_feature_sets[name]), "feature_names": model_feature_sets[name]}
            for name in matrices
        },
        "source_only_metrics": source_metrics,
        "target_grouped_oof_metrics": oof_metrics,
        "grouped_bootstrap_vs_matched_aggregate": bootstrap,
        "preregistration": {
            "primary_model": primary_name,
            "matched_baseline_model": baseline_name,
            "candidate_selection_performed": False,
            "development_scope": evaluation_cfg["development_scope"],
            "development_metric": evaluation_cfg["development_metric"],
            "test_metrics_used": False,
        },
        "negative_slice_tail_metrics": negative_slice_tail_metrics,
        "label_quality": label_quality,
        "sensitivity_metrics": sensitivity_metrics,
        "non_silver_grouped_bootstrap": non_silver_bootstrap,
        "direct_component_positive_mean_score_delta": direct_component_positive_mean_delta,
        "promotion": {
            "scope": evaluation_cfg["promotion_scope"],
            "primary_minus_matched_aggregate_ap": ap_gain,
            "minimum_ap_gain_required": evaluation_cfg["minimum_ap_gain_over_matched_aggregate"],
            "bootstrap_lower_bound": primary_bootstrap["grouped_bootstrap_95_ci"][0],
            "minimum_bootstrap_lower_bound": evaluation_cfg["minimum_bootstrap_lower_bound"],
            "non_silver_primary_minus_baseline_ap": non_silver_ap_delta,
            "minimum_non_silver_ap_delta": evaluation_cfg["minimum_non_silver_ap_delta"],
            "strong_evidence_primary_minus_baseline_ap": strong_evidence_ap_delta,
            "minimum_strong_evidence_ap_delta": evaluation_cfg["minimum_strong_evidence_ap_delta"],
            "direct_component_positive_mean_score_delta": direct_component_positive_mean_delta,
            "minimum_direct_component_positive_mean_score_delta": evaluation_cfg[
                "minimum_direct_component_positive_mean_score_delta"
            ],
            "maximum_observed_template_or_topic_mean_score_increase": max(negative_mean_increases),
            "maximum_allowed_template_or_topic_mean_score_increase": evaluation_cfg[
                "maximum_template_or_topic_mean_score_increase"
            ],
            "maximum_observed_template_or_topic_q95_score_increase": max(negative_q95_increases),
            "maximum_allowed_template_or_topic_q95_score_increase": evaluation_cfg[
                "maximum_template_or_topic_q95_score_increase"
            ],
            "maximum_observed_template_or_topic_top_decile_mean_score_increase": max(
                negative_top_decile_increases
            ),
            "maximum_allowed_template_or_topic_top_decile_mean_score_increase": evaluation_cfg[
                "maximum_template_or_topic_top_decile_mean_score_increase"
            ],
            "eligible": promotion,
        },
        "folds": fold_records,
        "source_only_weight_summaries": source_weight_summaries,
        "input_hashes": input_hashes,
        "prediction_sha256": sha256_file(prediction_path),
        "model_artifacts_sha256": sha256_file(artifact_path),
    }
    summary_path = output_root / output_cfg["evaluation_summary"]
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if summary_path.exists() and summary_path.read_text(encoding="utf-8") != rendered:
        raise ValueError("Refusing to overwrite a different Step23 evaluation summary")
    summary_path.write_text(rendered, encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "primary_model": primary_name,
        "primary_minus_matched_aggregate_ap": ap_gain,
        "promotion_eligible": promotion,
    }, indent=2))


if __name__ == "__main__":
    main()
