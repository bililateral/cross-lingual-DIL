#!/usr/bin/env python3
"""Evaluate Step21 text augmentation against an equal-weight duplication control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7
import step9_run_few_shot_adaptation as step9
import step15_v7_common as common


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step21_synthetic_train_only_policy.json"


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
        raise ValueError("Step21 grouped OOF has fewer components than folds")
    records = []
    for component, component_rows in grouped.items():
        positives = sum(row["review_label"] == "positive" for row in component_rows)
        digest = hashlib.sha256(f"{seed}|{component}".encode("utf-8")).hexdigest()
        records.append((component, len(component_rows), positives, digest))
    records.sort(key=lambda item: (-item[1], -item[2], item[3]))
    fold_total = [0] * fold_count
    fold_positive = [0] * fold_count
    assignment = {}
    for component, count, positives, _ in records:
        fold = min(
            range(fold_count),
            key=lambda index: (fold_positive[index], fold_total[index], index),
        )
        assignment[component] = fold
        fold_total[fold] += count
        fold_positive[fold] += positives
    if any(total == 0 for total in fold_total):
        raise ValueError("Step21 grouped OOF emitted an empty fold")
    return assignment


def pair_latents(
    pair_rows: list[dict],
    seller_index: dict[str, int],
    embeddings: np.ndarray,
    latent_cfg: dict,
) -> np.ndarray:
    input_dim = 2 * embeddings.shape[1]
    projection = common.fixed_projection(
        input_dim,
        int(latent_cfg["projection_dimensions"]),
        int(latent_cfg["projection_seed"]),
    )
    output = []
    for row in pair_rows:
        left = np.asarray(embeddings[seller_index[row["seller_uid_left"]]], dtype=np.float32)
        right = np.asarray(embeddings[seller_index[row["seller_uid_right"]]], dtype=np.float32)
        symmetric = np.concatenate([np.abs(left - right), left * right])
        output.append(np.asarray(symmetric @ projection, dtype=np.float64))
    return np.asarray(output, dtype=np.float64)


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "row_count": int(len(y_true)),
        "positive_count": int(np.sum(y_true == 1.0)),
        "negative_count": int(np.sum(y_true == 0.0)),
        "roc_auc": step7.roc_auc_score(y_true, scores),
        "average_precision": step7.average_precision_score(y_true, scores),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--track", action="append", dest="tracks")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    v7_policy_path = resolve(policy["inputs"]["v7_policy"])
    v7_policy = json.loads(v7_policy_path.read_text(encoding="utf-8"))
    output_root = resolve(policy["outputs_root"])
    generated_summary_path = output_root / policy["outputs"]["summary"]
    generated_summary = json.loads(generated_summary_path.read_text(encoding="utf-8"))
    tracks = args.tracks or list(generated_summary["tracks"])
    unknown = sorted(set(tracks) - set(generated_summary["tracks"]))
    if unknown:
        raise ValueError(f"Unknown Step21 generated tracks: {unknown}")

    pools = common.load_joined_rows(v7_policy)
    en_rows = pools["en_content_train_pool"]
    zh_rows = pools["zh_target_strict"]
    component_rows = load_csv(resolve(policy["inputs"]["component_assignments"]))
    component_index = {row["pair_uid"]: row for row in component_rows}
    for row in [*en_rows, *zh_rows]:
        assignment = component_index.get(row["pair_uid"])
        if assignment is None:
            raise ValueError(f"Missing Step16I component assignment for {row['pair_uid']}")
        if assignment["split_name"] != row["split_name"]:
            raise ValueError(f"Canonical/Step16I split mismatch for {row['pair_uid']}")
        if str(assignment.get("cross_split_component_leakage", "0")) == "1":
            raise ValueError(f"Step16I component leakage for {row['pair_uid']}")
        row["v7_component_id"] = assignment["recomputed_component_id"]
    en_train = [row for row in en_rows if row["split_name"] == "train"]
    zh_train = [row for row in zh_rows if row["split_name"] == "train"]
    if any(row["split_name"] != "train" for row in zh_train):
        raise ValueError("Step21 OOF received a non-train Chinese row")

    feature_names = list(v7_policy["inductive_features"]["stable_strict_clean_features"])
    latent_cfg = v7_policy["latent_pair_representation"]
    en_clean = common.strict_clean_matrix(en_rows, feature_names)
    zh_clean = common.strict_clean_matrix(zh_rows, feature_names)
    en_latent = common.projected_pair_latents(
        en_rows, v7_policy["pools"]["en_content_train_pool"], latent_cfg
    )
    zh_latent = common.projected_pair_latents(
        zh_rows, v7_policy["pools"]["zh_target_strict"], latent_cfg
    )
    en_index = {row["pair_uid"]: index for index, row in enumerate(en_rows)}
    zh_index = {row["pair_uid"]: index for index, row in enumerate(zh_rows)}
    en_train_matrix = np.concatenate(
        [en_clean[[en_index[row["pair_uid"]] for row in en_train]], en_latent[[en_index[row["pair_uid"]] for row in en_train]]],
        axis=1,
    )
    zh_train_matrix = np.concatenate(
        [zh_clean[[zh_index[row["pair_uid"]] for row in zh_train]], zh_latent[[zh_index[row["pair_uid"]] for row in zh_train]]],
        axis=1,
    )
    zh_train_index = {row["pair_uid"]: index for index, row in enumerate(zh_train)}
    fold_assignment = grouped_folds(zh_train, args.folds, args.seed)
    logistic_cfg = dict(v7_policy["step9_latent_mixup"]["logistic"])
    experiments = list(policy["evaluation"]["experiments"])
    outputs = {}
    producer_path = Path(__file__).resolve()

    for track_name in tracks:
        track_root = output_root / policy["outputs"]["tracks_directory"] / track_name
        result_path = track_root / "step21_grouped_oof_evaluation.json"
        prediction_path = track_root / "step21_grouped_oof_predictions.csv"
        lineage_path = track_root / "synthetic_pair_lineage.csv"
        labels_path = track_root / "synthetic_pair_labels.step5_compatible.csv"
        lineage_rows = load_csv(lineage_path)
        synthetic_labels = {
            row["pair_uid"]: row
            for row in load_csv(labels_path)
        }
        metadata_path = track_root / "synthetic_e5_identifier_redacted.json"
        matrix_path = track_root / "synthetic_e5_identifier_redacted.npy"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        embeddings = np.load(matrix_path, mmap_mode="r")
        if list(embeddings.shape) != list(metadata["shape"]):
            raise ValueError(f"Step21 synthetic embedding shape mismatch: {track_name}")
        expected_input_hashes = {
            "policy": sha256(policy_path),
            "producer": sha256(producer_path),
            "v7_policy": sha256(v7_policy_path),
            "component_assignments": sha256(
                resolve(policy["inputs"]["component_assignments"])
            ),
            "generation_summary": sha256(generated_summary_path),
            "synthetic_lineage": sha256(lineage_path),
            "synthetic_labels": sha256(labels_path),
            "synthetic_embedding_metadata": sha256(metadata_path),
            "synthetic_embedding_matrix": sha256(matrix_path),
        }
        if result_path.is_file() and prediction_path.is_file() and not args.force:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("input_hashes") != expected_input_hashes:
                raise FileExistsError(
                    f"Existing Step21 evaluation uses different code or inputs: {track_name}"
                )
            if existing.get("prediction_sha256") != sha256(prediction_path):
                raise ValueError(f"Existing Step21 prediction artifact drift: {track_name}")
            outputs[track_name] = existing
            continue
        if result_path.is_file() and not prediction_path.is_file() and not args.force:
            raise FileExistsError(
                f"Incomplete Step21 evaluation has a summary without predictions: {track_name}"
            )
        seller_index = {uid: index for index, uid in enumerate(metadata["seller_uids"])}
        synthetic_rows = []
        parent_uids = []
        for lineage in lineage_rows:
            synthetic = synthetic_labels[lineage["synthetic_pair_uid"]]
            synthetic_rows.append(synthetic)
            parent_uids.append(lineage["parent_pair_uid"])
        if any(parent_uid not in zh_train_index for parent_uid in parent_uids):
            raise ValueError(f"Step21 track includes a parent outside canonical Chinese train: {track_name}")
        synthetic_latent = pair_latents(synthetic_rows, seller_index, embeddings, latent_cfg)
        synthetic_clean = np.asarray(
            [zh_clean[zh_index[parent_uid]].copy() for parent_uid in parent_uids],
            dtype=np.float64,
        )
        e5_feature_index = feature_names.index(
            v7_policy["clean_semantic_encoder"]["output_feature"]
        )
        for index, row in enumerate(synthetic_rows):
            left = np.asarray(embeddings[seller_index[row["seller_uid_left"]]], dtype=float)
            right = np.asarray(embeddings[seller_index[row["seller_uid_right"]]], dtype=float)
            synthetic_clean[index, e5_feature_index] = float(np.dot(left, right))
        synthetic_matrix = np.concatenate([synthetic_clean, synthetic_latent], axis=1)
        duplication_matrix = np.asarray(
            [zh_train_matrix[zh_train_index[parent_uid]] for parent_uid in parent_uids],
            dtype=np.float64,
        )

        oof_scores = {name: np.full(len(zh_train), np.nan, dtype=float) for name in experiments}
        fold_records = []
        for fold in range(args.folds):
            held_indices = np.asarray(
                [
                    index
                    for index, row in enumerate(zh_train)
                    if fold_assignment[row["v7_component_id"]] == fold
                ],
                dtype=int,
            )
            held_set = set(held_indices.tolist())
            train_indices = np.asarray(
                [index for index in range(len(zh_train)) if index not in held_set], dtype=int
            )
            real_rows = en_train + [zh_train[index] for index in train_indices]
            real_matrix_raw = np.concatenate(
                [en_train_matrix, zh_train_matrix[train_indices]], axis=0
            )
            y_real = common.labels_array(real_rows)
            real_weights, weight_summary = common.factorized_evidence_weights(
                real_rows, v7_policy["factorized_evidence_weighting"]
            )
            imputation = common.fit_train_median_imputation(real_matrix_raw)
            real_matrix = common.apply_imputation(real_matrix_raw, imputation)
            held_matrix = common.apply_imputation(zh_train_matrix[held_indices], imputation)
            _, standardization = step9.fit_standardization(real_matrix, True)
            parent_real_position = {
                row["pair_uid"]: len(en_train) + offset
                for offset, row in enumerate([zh_train[index] for index in train_indices])
            }
            eligible_synthetic_indices = [
                index for index, parent_uid in enumerate(parent_uids) if parent_uid in parent_real_position
            ]
            variants_by_parent = Counter(parent_uids[index] for index in eligible_synthetic_indices)
            synthetic_weights = np.asarray(
                [
                    real_weights[parent_real_position[parent_uids[index]]]
                    / variants_by_parent[parent_uids[index]]
                    for index in eligible_synthetic_indices
                ],
                dtype=float,
            )
            for experiment in experiments:
                x_train = real_matrix
                y_train = y_real
                row_weights = real_weights
                if experiment != "no_augmentation":
                    if experiment == "equal_effective_weight_duplication":
                        added_raw = duplication_matrix[eligible_synthetic_indices]
                    elif experiment == "identifier_redacted_text_augmentation":
                        added_raw = synthetic_matrix[eligible_synthetic_indices]
                    else:
                        raise ValueError(f"Unknown Step21 experiment: {experiment}")
                    added = common.apply_imputation(added_raw, imputation)
                    x_train = np.concatenate([real_matrix, added], axis=0)
                    y_train = np.concatenate(
                        [y_real, np.ones(len(eligible_synthetic_indices), dtype=float)]
                    )
                    row_weights = np.concatenate([real_weights, synthetic_weights])
                artifact, _ = step9.fit_regularized_logistic(
                    x_train,
                    y_train,
                    logistic_cfg,
                    sample_weight_multipliers=row_weights,
                    sample_weight_target_total=float(len(real_rows)),
                    precomputed_standardization=standardization,
                )
                oof_scores[experiment][held_indices] = step9.apply_logistic_artifact_to_matrix(
                    held_matrix, artifact
                )
            fold_records.append(
                {
                    "fold": fold,
                    "held_out_chinese_rows": len(held_indices),
                    "held_out_chinese_positives": int(
                        np.sum(common.labels_array([zh_train[index] for index in held_indices]))
                    ),
                    "eligible_synthetic_rows": len(eligible_synthetic_indices),
                    "eligible_synthetic_effective_weight": float(np.sum(synthetic_weights)),
                    "real_weight_summary": weight_summary,
                }
            )
        if any(np.any(~np.isfinite(scores)) for scores in oof_scores.values()):
            raise ValueError(f"Step21 OOF did not score every Chinese train row: {track_name}")
        y_oof = common.labels_array(zh_train)
        experiment_metrics = {name: metrics(y_oof, scores) for name, scores in oof_scores.items()}
        text_ap = float(experiment_metrics["identifier_redacted_text_augmentation"]["average_precision"])
        duplication_ap = float(experiment_metrics["equal_effective_weight_duplication"]["average_precision"])
        no_aug_ap = float(experiment_metrics["no_augmentation"]["average_precision"])
        summary = {
            "step": "step21_synthetic_train_only_grouped_oof_evaluation",
            "track": track_name,
            "selection_scope": policy["evaluation"]["selection_scope"],
            "valid_or_test_scores_used": False,
            "fold_count": args.folds,
            "fold_seed": args.seed,
            "real_chinese_train_rows": len(zh_train),
            "real_chinese_train_components": len({row["v7_component_id"] for row in zh_train}),
            "real_parent_components": generated_summary["tracks"][track_name]["effective_independent_sample_count"],
            "synthetic_rows_are_independent_samples": False,
            "metrics": experiment_metrics,
            "oof_score_hashes": {
                name: common.canonical_hash(scores.tolist())
                for name, scores in oof_scores.items()
            },
            "comparisons": {
                "text_augmentation_minus_no_augmentation_ap": text_ap - no_aug_ap,
                "text_augmentation_minus_equal_weight_duplication_ap": text_ap - duplication_ap,
                "interpretation": (
                    "representation_gain_supported"
                    if text_ap > duplication_ap and text_ap > no_aug_ap
                    else "no_representation_gain_over_required_controls"
                ),
            },
            "folds": fold_records,
            "input_hashes": expected_input_hashes,
        }
        prediction_rows = []
        for index, row in enumerate(zh_train):
            output = {
                "pair_uid": row["pair_uid"],
                "v7_component_id": row["v7_component_id"],
                "review_label": row["review_label"],
                "fold": fold_assignment[row["v7_component_id"]],
            }
            for experiment in experiments:
                output[f"prob_{experiment}"] = f"{oof_scores[experiment][index]:.12f}"
            prediction_rows.append(output)
        with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
        summary["prediction_sha256"] = sha256(prediction_path)
        result_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        outputs[track_name] = summary

    no_aug_hashes = {
        summary["oof_score_hashes"]["no_augmentation"] for summary in outputs.values()
    }
    if len(no_aug_hashes) != 1:
        raise ValueError("Step21 no-augmentation OOF scores changed across synthetic tracks")
    top_summary_path = output_root / "step21_synthetic_augmentation_evaluation_summary.json"
    top_summary_path.write_text(
        json.dumps(
            {
                "status": "train_grouped_oof_only",
                "publication_holdout_untouched": True,
                "tracks": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(top_summary_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
