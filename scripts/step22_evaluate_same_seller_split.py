#!/usr/bin/env python3
"""Evaluate Step22 pseudo-alias augmentation on canonical Chinese train grouped OOF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7
import step9_run_few_shot_adaptation as step9
import step15_v7_common as common


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step22_same_seller_split_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_folds(rows: list[dict], fold_count: int, seed: int) -> dict[str, int]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["v7_component_id"]].append(row)
    if len(grouped) < fold_count:
        raise ValueError("Step22 grouped OOF has fewer components than folds")
    total_rows = len(rows)
    total_positives = sum(row["review_label"] == "positive" for row in rows)
    total_negatives = total_rows - total_positives
    targets = np.asarray(
        [total_rows / fold_count, total_positives / fold_count, total_negatives / fold_count],
        dtype=float,
    )
    records = []
    for component, component_rows in grouped.items():
        positives = sum(row["review_label"] == "positive" for row in component_rows)
        negatives = len(component_rows) - positives
        normalized_mass = max(
            len(component_rows) / max(targets[0], 1.0),
            positives / max(targets[1], 1.0),
            negatives / max(targets[2], 1.0),
        )
        digest = hashlib.sha256(f"{seed}|{component}".encode("utf-8")).hexdigest()
        records.append((component, len(component_rows), positives, negatives, normalized_mass, digest))
    records.sort(key=lambda item: (-item[4], -item[1], -item[2], item[5]))
    fold_counts = np.zeros((fold_count, 3), dtype=float)
    assignment: dict[str, int] = {}
    for component, count, positives, negatives, _, _ in records:
        addition = np.asarray([count, positives, negatives], dtype=float)
        candidates = []
        for index in range(fold_count):
            proposed = fold_counts.copy()
            proposed[index] += addition
            normalized_error = (proposed - targets[None, :]) / np.maximum(targets[None, :], 1.0)
            candidates.append(
                (
                    float(np.sum(normalized_error**2)),
                    float(fold_counts[index, 0]),
                    float(fold_counts[index, 1]),
                    index,
                )
            )
        fold = min(candidates)[-1]
        assignment[component] = fold
        fold_counts[fold] += addition
    if any(total == 0 for total in fold_counts[:, 0]):
        raise ValueError("Step22 grouped OOF emitted an empty fold")
    if any(positive == 0 for positive in fold_counts[:, 1]) or any(
        negative == 0 for negative in fold_counts[:, 2]
    ):
        raise ValueError("Step22 grouped OOF emitted a single-class held-out fold")
    return assignment


def pair_representation(
    rows: list[dict], seller_index: dict[str, int], embeddings: np.ndarray, latent_cfg: dict
) -> np.ndarray:
    projection = common.fixed_projection(
        2 * embeddings.shape[1],
        int(latent_cfg["projection_dimensions"]),
        int(latent_cfg["projection_seed"]),
    )
    output = []
    for row in rows:
        left = np.asarray(embeddings[seller_index[row["seller_uid_left"]]], dtype=np.float32)
        right = np.asarray(embeddings[seller_index[row["seller_uid_right"]]], dtype=np.float32)
        cosine = float(np.dot(left, right))
        symmetric = np.concatenate([np.abs(left - right), left * right])
        latent = np.asarray(symmetric @ projection, dtype=np.float64)
        output.append(np.concatenate([[cosine], latent]))
    return np.asarray(output, dtype=np.float64)


def real_representation(rows: list[dict], pool_cfg: dict, latent_cfg: dict, cosine_feature: str) -> np.ndarray:
    latent = common.projected_pair_latents(rows, pool_cfg, latent_cfg)
    cosine = np.asarray([[float(row[cosine_feature])] for row in rows], dtype=np.float64)
    return np.concatenate([cosine, latent], axis=1)


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "row_count": int(len(y_true)),
        "positive_count": int(np.sum(y_true == 1.0)),
        "negative_count": int(np.sum(y_true == 0.0)),
        "roc_auc": step7.roc_auc_score(y_true, scores),
        "average_precision": step7.average_precision_score(y_true, scores),
    }


def scaled_weights(base_weights: np.ndarray, target_total: float) -> np.ndarray:
    if target_total < 0:
        raise ValueError("Step22 received a negative synthetic weight budget")
    if len(base_weights) == 0:
        if target_total > 0:
            raise ValueError("Step22 cannot allocate a positive budget to an empty class")
        return np.asarray([], dtype=float)
    base = np.asarray(base_weights, dtype=float)
    if np.any(~np.isfinite(base)) or np.any(base < 0):
        raise ValueError("Step22 weight budget received invalid base weights")
    base_total = float(np.sum(base))
    if base_total <= 0:
        if target_total > 0:
            raise ValueError("Step22 cannot scale a zero-weight class to a positive budget")
        return np.zeros(len(base), dtype=float)
    return base * (target_total / base_total)


def duplicate_class(
    matrix: np.ndarray,
    labels: np.ndarray,
    base_weights: np.ndarray,
    class_value: float,
    target_total: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.where(labels == class_value)[0]
    if len(indices) == 0:
        raise ValueError(f"Step22 duplication control has no class={class_value} rows")
    return matrix[indices], labels[indices], scaled_weights(base_weights[indices], target_total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    v7_policy_path = resolve(policy["inputs"]["v7_policy"])
    v7_policy = json.loads(v7_policy_path.read_text(encoding="utf-8"))
    output_root = resolve(policy["outputs_root"])
    summary_path = output_root / policy["outputs"]["evaluation_summary"]
    prediction_path = output_root / policy["outputs"]["evaluation_predictions"]
    generation_summary_path = output_root / policy["outputs"]["summary"]
    pair_path = output_root / policy["outputs"]["pair_labels"]
    lineage_path = output_root / policy["outputs"]["pair_lineage"]
    embedding_path = output_root / policy["outputs"]["embedding_matrix"]
    embedding_metadata_path = output_root / policy["outputs"]["embedding_metadata"]
    producer_path = Path(__file__).resolve()
    required = [generation_summary_path, pair_path, lineage_path, embedding_path, embedding_metadata_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Step22 evaluation is missing inputs: {missing}")
    input_hashes = {
        "policy": sha256(policy_path),
        "producer": sha256(producer_path),
        "step7_dependency": sha256(Path(step7.__file__).resolve()),
        "step9_dependency": sha256(Path(step9.__file__).resolve()),
        "step15_v7_common_dependency": sha256(Path(common.__file__).resolve()),
        "v7_policy": sha256(v7_policy_path),
        "component_assignments": sha256(resolve(policy["inputs"]["component_assignments"])),
        "generation_summary": sha256(generation_summary_path),
        "pseudo_pair_labels": sha256(pair_path),
        "pseudo_pair_lineage": sha256(lineage_path),
        "pseudo_embedding_matrix": sha256(embedding_path),
        "pseudo_embedding_metadata": sha256(embedding_metadata_path),
    }
    if summary_path.is_file() and prediction_path.is_file() and not args.force:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("input_hashes") != input_hashes:
            raise FileExistsError("Existing Step22 evaluation belongs to different code or inputs")
        if existing.get("prediction_sha256") != sha256(prediction_path):
            raise ValueError("Existing Step22 prediction artifact drift")
        print(summary_path.read_text(encoding="utf-8"))
        return
    if summary_path.exists() != prediction_path.exists() and not args.force:
        raise FileExistsError("Incomplete Step22 evaluation output exists")

    pools = common.load_joined_rows(v7_policy)
    en_rows = pools["en_content_train_pool"]
    zh_rows = pools["zh_target_strict"]
    component_rows = load_csv(resolve(policy["inputs"]["component_assignments"]))
    component_index = {row["pair_uid"]: row for row in component_rows}
    for row in [*en_rows, *zh_rows]:
        assignment = component_index.get(row["pair_uid"])
        if assignment is None:
            raise ValueError(f"Missing Step16I assignment for {row['pair_uid']}")
        if assignment["split_name"] != row["split_name"]:
            raise ValueError(f"Canonical/Step16I split mismatch for {row['pair_uid']}")
        if str(assignment.get("cross_split_component_leakage", "0")) == "1":
            raise ValueError(f"Step16I component leakage for {row['pair_uid']}")
        row["step22_component_id"] = assignment["recomputed_component_id"]
        # Factorized weighting still reads the historical v7 field name. Replace
        # it with the same Step16I-v2 component used by the Step22 fold split.
        row["v7_component_id"] = assignment["recomputed_component_id"]
    en_train = [row for row in en_rows if row["split_name"] == "train"]
    zh_train = [row for row in zh_rows if row["split_name"] == "train"]
    evaluation_cfg = policy["evaluation"]
    fold_assignment = grouped_folds(
        [dict(row, v7_component_id=row["step22_component_id"]) for row in zh_train],
        int(evaluation_cfg["fold_count"]),
        int(evaluation_cfg["fold_seed"]),
    )
    latent_cfg = v7_policy["latent_pair_representation"]
    cosine_feature = v7_policy["clean_semantic_encoder"]["output_feature"]
    en_matrix_all = real_representation(en_rows, v7_policy["pools"]["en_content_train_pool"], latent_cfg, cosine_feature)
    zh_matrix_all = real_representation(zh_rows, v7_policy["pools"]["zh_target_strict"], latent_cfg, cosine_feature)
    en_index = {row["pair_uid"]: index for index, row in enumerate(en_rows)}
    zh_index = {row["pair_uid"]: index for index, row in enumerate(zh_rows)}
    en_train_matrix = en_matrix_all[[en_index[row["pair_uid"]] for row in en_train]]
    zh_train_matrix = zh_matrix_all[[zh_index[row["pair_uid"]] for row in zh_train]]

    pseudo_rows = load_csv(pair_path)
    lineage_index = {row["synthetic_pair_uid"]: row for row in load_csv(lineage_path)}
    metadata = json.loads(embedding_metadata_path.read_text(encoding="utf-8"))
    embeddings = np.load(embedding_path, mmap_mode="r")
    if list(embeddings.shape) != list(metadata["shape"]):
        raise ValueError("Step22 embedding shape mismatch")
    seller_index = {uid: index for index, uid in enumerate(metadata["seller_uids"])}
    pseudo_matrix = pair_representation(pseudo_rows, seller_index, embeddings, latent_cfg)
    pseudo_labels = common.labels_array(pseudo_rows)
    pseudo_components = []
    for row in pseudo_rows:
        lineage = lineage_index.get(row["pair_uid"])
        if lineage is None:
            raise ValueError(f"Missing Step22 lineage for {row['pair_uid']}")
        if row["split_name"] != "train" or row["benchmark_eligible"] != "0":
            raise ValueError(f"Step22 pseudo row escaped train-only scope: {row['pair_uid']}")
        pseudo_components.append(lineage["parent_component_id"])
    positive_pseudo_all = np.where(pseudo_labels == 1.0)[0]
    negative_pseudo_all = np.where(pseudo_labels == 0.0)[0]
    if len(positive_pseudo_all) < 100 or len(negative_pseudo_all) < 20:
        raise ValueError("Step22 evaluation has insufficient pseudo positive/negative parents")

    experiments = list(evaluation_cfg["experiments"])
    oof_scores = {name: np.full(len(zh_train), np.nan, dtype=float) for name in experiments}
    fold_records = []
    logistic_cfg = dict(v7_policy["step9_latent_mixup"]["logistic"])
    pos_fraction = float(policy["weighting"]["positive_synthetic_budget_fraction_of_real_zh_positive_weight"])
    neg_fraction = float(policy["weighting"]["negative_synthetic_budget_fraction_of_real_zh_negative_weight"])
    for fold in range(int(evaluation_cfg["fold_count"])):
        held_indices = np.asarray(
            [index for index, row in enumerate(zh_train) if fold_assignment[row["step22_component_id"]] == fold],
            dtype=int,
        )
        held_components = {zh_train[index]["step22_component_id"] for index in held_indices}
        train_indices = np.asarray([index for index in range(len(zh_train)) if index not in set(held_indices.tolist())], dtype=int)
        real_rows = en_train + [zh_train[index] for index in train_indices]
        real_matrix_raw = np.concatenate([en_train_matrix, zh_train_matrix[train_indices]], axis=0)
        y_real = common.labels_array(real_rows)
        real_weights, weight_summary = common.factorized_evidence_weights(real_rows, v7_policy["factorized_evidence_weighting"])
        zh_real_start = len(en_train)
        zh_fold_labels = common.labels_array([zh_train[index] for index in train_indices])
        zh_fold_weights = real_weights[zh_real_start:]
        positive_budget = pos_fraction * float(np.sum(zh_fold_weights[zh_fold_labels == 1.0]))
        negative_budget = neg_fraction * float(np.sum(zh_fold_weights[zh_fold_labels == 0.0]))
        eligible_pseudo = np.asarray(
            [index for index, component in enumerate(pseudo_components) if component not in held_components],
            dtype=int,
        )
        eligible_positive = eligible_pseudo[pseudo_labels[eligible_pseudo] == 1.0]
        eligible_negative = eligible_pseudo[pseudo_labels[eligible_pseudo] == 0.0]
        if len(eligible_positive) == 0 or len(eligible_negative) == 0:
            raise ValueError(f"Step22 fold {fold} has an empty eligible pseudo class")
        pseudo_positive_weights = scaled_weights(np.ones(len(eligible_positive)), positive_budget)
        pseudo_negative_weights = scaled_weights(np.ones(len(eligible_negative)), negative_budget)
        zh_train_raw = zh_train_matrix[train_indices]
        duplicate_positive_x, duplicate_positive_y, duplicate_positive_w = duplicate_class(
            zh_train_raw, zh_fold_labels, zh_fold_weights, 1.0, positive_budget
        )
        duplicate_negative_x, duplicate_negative_y, duplicate_negative_w = duplicate_class(
            zh_train_raw, zh_fold_labels, zh_fold_weights, 0.0, negative_budget
        )
        imputation = common.fit_train_median_imputation(real_matrix_raw)
        real_matrix = common.apply_imputation(real_matrix_raw, imputation)
        held_matrix = common.apply_imputation(zh_train_matrix[held_indices], imputation)
        _, standardization = step9.fit_standardization(real_matrix, True)
        additions = {
            "no_augmentation": (np.empty((0, real_matrix.shape[1])), np.asarray([], dtype=float), np.asarray([], dtype=float)),
            "equal_weight_duplication_positive_budget": (duplicate_positive_x, duplicate_positive_y, duplicate_positive_w),
            "same_seller_split_positive_only": (pseudo_matrix[eligible_positive], pseudo_labels[eligible_positive], pseudo_positive_weights),
            "equal_weight_duplication_full_budget": (
                np.concatenate([duplicate_positive_x, duplicate_negative_x], axis=0),
                np.concatenate([duplicate_positive_y, duplicate_negative_y]),
                np.concatenate([duplicate_positive_w, duplicate_negative_w]),
            ),
            "same_seller_split_plus_reviewed_negative_views": (
                np.concatenate([pseudo_matrix[eligible_positive], pseudo_matrix[eligible_negative]], axis=0),
                np.concatenate([pseudo_labels[eligible_positive], pseudo_labels[eligible_negative]]),
                np.concatenate([pseudo_positive_weights, pseudo_negative_weights]),
            ),
        }
        for experiment in experiments:
            if experiment not in additions:
                raise ValueError(f"Unknown Step22 experiment: {experiment}")
            added_raw, added_y, added_weights = additions[experiment]
            if len(added_y):
                added = common.apply_imputation(added_raw, imputation)
                x_train = np.concatenate([real_matrix, added], axis=0)
                y_train = np.concatenate([y_real, added_y])
                row_weights = np.concatenate([real_weights, added_weights])
            else:
                x_train, y_train, row_weights = real_matrix, y_real, real_weights
            artifact, _ = step9.fit_regularized_logistic(
                x_train,
                y_train,
                logistic_cfg,
                sample_weight_multipliers=row_weights,
                sample_weight_target_total=float(len(real_rows)),
                precomputed_standardization=standardization,
            )
            oof_scores[experiment][held_indices] = step9.apply_logistic_artifact_to_matrix(held_matrix, artifact)
        fold_records.append({
            "fold": fold,
            "held_out_rows": len(held_indices),
            "held_out_positives": int(np.sum(common.labels_array([zh_train[index] for index in held_indices]))),
            "eligible_pseudo_positives": len(eligible_positive),
            "eligible_pseudo_negatives": len(eligible_negative),
            "positive_synthetic_effective_weight": positive_budget,
            "negative_synthetic_effective_weight": negative_budget,
            "real_weight_summary": weight_summary,
        })
    if any(np.any(~np.isfinite(scores)) for scores in oof_scores.values()):
        raise ValueError("Step22 OOF did not score every Chinese train row")
    y_oof = common.labels_array(zh_train)
    experiment_metrics = {name: metrics(y_oof, scores) for name, scores in oof_scores.items()}
    no_ap = float(experiment_metrics["no_augmentation"]["average_precision"])
    dup_pos_ap = float(experiment_metrics["equal_weight_duplication_positive_budget"]["average_precision"])
    pseudo_pos_ap = float(experiment_metrics["same_seller_split_positive_only"]["average_precision"])
    dup_full_ap = float(experiment_metrics["equal_weight_duplication_full_budget"]["average_precision"])
    pseudo_full_ap = float(experiment_metrics["same_seller_split_plus_reviewed_negative_views"]["average_precision"])
    minimum_no_gain = float(evaluation_cfg["minimum_ap_gain_over_no_augmentation"])
    minimum_dup_gain = float(evaluation_cfg["minimum_ap_gain_over_matched_duplication"])
    positive_supported = pseudo_pos_ap - no_ap >= minimum_no_gain and pseudo_pos_ap - dup_pos_ap >= minimum_dup_gain
    full_supported = pseudo_full_ap - no_ap >= minimum_no_gain and pseudo_full_ap - dup_full_ap >= minimum_dup_gain
    summary = {
        "step": "step22_same_seller_split_grouped_oof_evaluation",
        "policy_version": policy["version"],
        "status": "train_grouped_oof_only",
        "publication_holdout_untouched": True,
        "valid_or_test_scores_used": False,
        "representation": evaluation_cfg["representation"],
        "real_chinese_train_rows": len(zh_train),
        "real_chinese_train_components": len({row["step22_component_id"] for row in zh_train}),
        "pseudo_positive_source_units": len(positive_pseudo_all),
        "reviewed_negative_source_units": len(negative_pseudo_all),
        "metrics": experiment_metrics,
        "comparisons": {
            "same_seller_positive_minus_no_augmentation_ap": pseudo_pos_ap - no_ap,
            "same_seller_positive_minus_matched_duplication_ap": pseudo_pos_ap - dup_pos_ap,
            "full_method_minus_no_augmentation_ap": pseudo_full_ap - no_ap,
            "full_method_minus_matched_duplication_ap": pseudo_full_ap - dup_full_ap,
            "full_method_minus_positive_only_ap": pseudo_full_ap - pseudo_pos_ap,
            "positive_representation_gain_supported": positive_supported,
            "full_representation_gain_supported": full_supported,
            "promotion_eligible": positive_supported or full_supported,
        },
        "folds": fold_records,
        "oof_score_hashes": {name: common.canonical_hash(scores.tolist()) for name, scores in oof_scores.items()},
        "input_hashes": input_hashes,
    }
    prediction_rows = []
    for index, row in enumerate(zh_train):
        output = {
            "pair_uid": row["pair_uid"],
            "component_id": row["step22_component_id"],
            "review_label": row["review_label"],
            "fold": fold_assignment[row["step22_component_id"]],
        }
        for experiment in experiments:
            output[f"prob_{experiment}"] = f"{oof_scores[experiment][index]:.12f}"
        prediction_rows.append(output)
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    summary["prediction_sha256"] = sha256(prediction_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
