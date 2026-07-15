#!/usr/bin/env python3
"""Run the train-OOF Step15-v8 B0-B3 representation bridge audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7
import step15_v7_common as v7
import step15_v8_common as common


ROOT = Path(__file__).resolve().parent.parent


def prediction_rows(
    rows: list[dict], scores: np.ndarray, threshold: float | None, split_name: str
) -> list[dict]:
    output = []
    for row, score in zip(rows, scores, strict=True):
        item = {
            "pair_uid": row["pair_uid"],
            "step15_pool": row["step15_pool"],
            "domain": row["domain"],
            "split_name": split_name,
            "review_label": row["review_label"],
            "evidence_type": row["evidence_type"],
            "v7_component_id": row["v7_component_id"],
            "prob_positive": f"{float(score):.12f}",
            "selected_threshold": "" if threshold is None else f"{float(threshold):.12f}",
            "predicted_label": "" if threshold is None else int(float(score) >= threshold),
        }
        output.append(item)
    return output


PREDICTION_FIELDS = [
    "pair_uid",
    "step15_pool",
    "domain",
    "split_name",
    "review_label",
    "evidence_type",
    "v7_component_id",
    "prob_positive",
    "selected_threshold",
    "predicted_label",
]


def select_matrix(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[indices], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path, policy, v7_policy = common.load_policy(args.policy)
    validation = common.validate_policy_contract(policy, v7_policy)
    if args.validate_config_only:
        print(json.dumps(validation, indent=2))
        return
    run_id = args.run_id or policy["default_run_id"]
    root = common.run_root(policy, run_id)
    final_root = root / policy["bridge_audit"]["output_subdirectory"]
    staging_root = final_root.with_name(f".{final_root.name}.incomplete")
    if final_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Step15-v8 bridge audit: {final_root} / {staging_root}"
        )
    if not (root / policy["clean_semantics"]["output_subdirectory"] / "clean_semantics_manifest.json").is_file():
        raise FileNotFoundError("Step15-v8 clean semantics must be built before the bridge audit")
    runtime_chain = common.verify_readiness_runtime_chain(policy, v7_policy)

    rows_by_pool = common.load_joined_rows(policy, v7_policy, root)
    splits = common.split_rows(rows_by_pool)
    train_rows = splits["train"]
    valid_rows = splits["valid"]
    test_rows = splits["internal_development_test"]
    expert_train_control_rows = splits["evidence_expert_train_controls"]
    expert_valid_control_rows = splits["evidence_expert_valid_controls"]
    expected_test = int(policy["evaluation"]["current_internal_test_row_count_expected"])
    if len(test_rows) != expected_test:
        raise ValueError(f"Internal test changed: expected={expected_test} observed={len(test_rows)}")

    latent_by_pair = {}
    latent_cfg = v7_policy["latent_pair_representation"]
    for pool_name, rows in rows_by_pool.items():
        pool_cfg = policy["pools"][pool_name]
        v7_pool = {
            "clean_e5_cache_metadata": pool_cfg["v7_clean_e5_metadata"],
            "clean_e5_cache_matrix": pool_cfg["v7_clean_e5_matrix"],
        }
        matrix = v7.projected_pair_latents(rows, v7_pool, latent_cfg)
        for row, vector in zip(rows, matrix, strict=True):
            latent_by_pair[(pool_name, row["pair_uid"])] = np.asarray(vector, dtype=float)

    def latents(rows: list[dict]) -> np.ndarray:
        if not rows:
            return np.empty(
                (0, int(latent_cfg["projection_dimensions"])), dtype=float
            )
        return np.asarray(
            [latent_by_pair[(row["step15_pool"], row["pair_uid"])] for row in rows],
            dtype=float,
        )

    train_latent = latents(train_rows)
    valid_latent = latents(valid_rows)
    test_latent = latents(test_rows)
    expert_train_control_latent = latents(expert_train_control_rows)
    expert_valid_control_latent = latents(expert_valid_control_rows)
    corpus_context = common.load_corpus_reference_context(policy, v7_policy)
    bridge_cfg = policy["bridge_audit"]
    seeds = [int(value) for value in bridge_cfg["seeds"]]
    feature_set_ids = list(bridge_cfg["feature_sets"])
    staging_root.mkdir(parents=True, exist_ok=False)
    oof_records = []
    oof_scores: dict[tuple[str, str, int], np.ndarray] = {}
    oof_paths = {}
    fold_manifests = []

    for feature_set_id in feature_set_ids:
        for seed in seeds:
            folds = common.seeded_component_group_folds(
                train_rows, int(bridge_cfg["group_folds"]), seed
            )
            scores = np.full(len(train_rows), np.nan, dtype=float)
            fold_records = []
            for fold_index, (fit_indices, held_indices) in enumerate(folds):
                fit_rows = [train_rows[index] for index in fit_indices]
                held_rows = [train_rows[index] for index in held_indices]
                corpus_reference = common.fit_corpus_reference(fit_rows, corpus_context)
                fit_feature_rows = common.apply_corpus_reference(
                    fit_rows, corpus_reference, corpus_context
                )
                held_feature_rows = common.apply_corpus_reference(
                    held_rows, corpus_reference, corpus_context
                )
                fit_latent = select_matrix(train_latent, fit_indices)
                held_latent = select_matrix(train_latent, held_indices)
                x_fit, transform = common.fit_feature_transform(
                    fit_feature_rows, feature_set_id, policy, v7_policy, fit_latent
                )
                x_held = common.apply_feature_transform(
                    held_feature_rows, policy, v7_policy, transform, held_latent
                )
                artifact = common.fit_lr(x_fit, fit_rows, policy, v7_policy)
                scores[held_indices] = common.apply_lr(x_held, artifact)
                fit_groups = {common.component_group_key(row) for row in fit_rows}
                held_groups = {common.component_group_key(row) for row in held_rows}
                if fit_groups & held_groups:
                    raise ValueError("Component leakage in persisted Step15-v8 OOF fold")
                fold_records.append(
                    {
                        "fold_index": fold_index,
                        "fit_count": len(fit_rows),
                        "held_count": len(held_rows),
                        "fit_component_count": len(fit_groups),
                        "held_component_count": len(held_groups),
                        "held_pair_uid_sha256": common.canonical_hash(
                            sorted(row["pair_uid"] for row in held_rows)
                        ),
                        "fold_train_corpus_reference_sha256": common.canonical_hash(
                            corpus_reference
                        ),
                        "transform": transform,
                        "model": artifact,
                    }
                )
            if not np.all(np.isfinite(scores)):
                raise ValueError(f"Incomplete OOF scores: {feature_set_id}:seed={seed}")
            macro_ap, by_domain_ap = common.macro_domain_average_precision(train_rows, scores)
            combined_ap = float(step7.average_precision_score(v7.labels_array(train_rows), scores))
            path = staging_root / "oof" / f"{feature_set_id}__lr_l2__seed_{seed}.train_oof.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(common.render_csv(prediction_rows(train_rows, scores, None, "train_oof"), PREDICTION_FIELDS))
            relative = str((final_root / path.relative_to(staging_root)).relative_to(ROOT)).replace("\\", "/")
            oof_paths[(feature_set_id, "lr_l2", seed)] = relative
            oof_scores[(feature_set_id, "lr_l2", seed)] = scores
            oof_records.append(
                {
                    "feature_set_id": feature_set_id,
                    "model_family": "lr_l2",
                    "seed": seed,
                    "macro_domain_average_precision": macro_ap,
                    "average_precision_by_domain": by_domain_ap,
                    "combined_average_precision": combined_ap,
                    "prediction_path": relative,
                }
            )
            fold_manifests.append(
                {
                    "feature_set_id": feature_set_id,
                    "model_family": "lr_l2",
                    "seed": seed,
                    "folds": fold_records,
                }
            )

    mean_oof = {
        feature_set_id: float(
            np.mean(
                [
                    row["macro_domain_average_precision"]
                    for row in oof_records
                    if row["feature_set_id"] == feature_set_id
                    and row["model_family"] == "lr_l2"
                ]
            )
        )
        for feature_set_id in feature_set_ids
    }
    best_score = max(mean_oof.values())
    tolerance = float(bridge_cfg["selection_tie_tolerance"])
    tied = [name for name in feature_set_ids if best_score - mean_oof[name] <= tolerance]
    simplicity = [
        "B1_v7_20d_e5_cosine_only",
        "B2_redacted_multiencoder_consensus",
        "B3_nonidentifier_retrieval_bridge",
        "B0_v7_20d_plus_e5_latent64",
    ]
    selected_feature_set = min(tied, key=simplicity.index)

    for seed in seeds:
        folds = common.seeded_component_group_folds(
            train_rows, int(bridge_cfg["group_folds"]), seed
        )
        scores = np.full(len(train_rows), np.nan, dtype=float)
        fold_records = []
        for fold_index, (fit_indices, held_indices) in enumerate(folds):
            fit_rows = [train_rows[index] for index in fit_indices]
            held_rows = [train_rows[index] for index in held_indices]
            corpus_reference = common.fit_corpus_reference(fit_rows, corpus_context)
            fit_feature_rows = common.apply_corpus_reference(
                fit_rows, corpus_reference, corpus_context
            )
            held_feature_rows = common.apply_corpus_reference(
                held_rows, corpus_reference, corpus_context
            )
            fit_latent = select_matrix(train_latent, fit_indices)
            held_latent = select_matrix(train_latent, held_indices)
            x_fit, transform = common.fit_feature_transform(
                fit_feature_rows, selected_feature_set, policy, v7_policy, fit_latent
            )
            x_held = common.apply_feature_transform(
                held_feature_rows, policy, v7_policy, transform, held_latent
            )
            artifact = common.fit_pairwise_ranker(
                x_fit, fit_rows, policy, v7_policy, seed + fold_index
            )
            scores[held_indices] = common.apply_pairwise_ranker(x_held, artifact)
            fold_records.append(
                {
                    "fold_index": fold_index,
                    "fit_count": len(fit_rows),
                    "held_count": len(held_rows),
                    "held_pair_uid_sha256": common.canonical_hash(
                        sorted(row["pair_uid"] for row in held_rows)
                    ),
                    "fold_train_corpus_reference_sha256": common.canonical_hash(
                        corpus_reference
                    ),
                    "transform": transform,
                    "model": artifact,
                }
            )
        macro_ap, by_domain_ap = common.macro_domain_average_precision(train_rows, scores)
        combined_ap = float(step7.average_precision_score(v7.labels_array(train_rows), scores))
        path = staging_root / "oof" / f"{selected_feature_set}__linear_pairwise_ranknet__seed_{seed}.train_oof.csv"
        path.write_bytes(common.render_csv(prediction_rows(train_rows, scores, None, "train_oof"), PREDICTION_FIELDS))
        relative = str((final_root / path.relative_to(staging_root)).relative_to(ROOT)).replace("\\", "/")
        oof_paths[(selected_feature_set, "linear_pairwise_ranknet", seed)] = relative
        oof_scores[(selected_feature_set, "linear_pairwise_ranknet", seed)] = scores
        oof_records.append(
            {
                "feature_set_id": selected_feature_set,
                "model_family": "linear_pairwise_ranknet",
                "seed": seed,
                "macro_domain_average_precision": macro_ap,
                "average_precision_by_domain": by_domain_ap,
                "combined_average_precision": combined_ap,
                "prediction_path": relative,
            }
        )
        fold_manifests.append(
            {
                "feature_set_id": selected_feature_set,
                "model_family": "linear_pairwise_ranknet",
                "seed": seed,
                "folds": fold_records,
            }
        )

    family_scores = {}
    for family in bridge_cfg["model_family_candidates"]:
        family_scores[family] = float(
            np.mean(
                [
                    row["macro_domain_average_precision"]
                    for row in oof_records
                    if row["feature_set_id"] == selected_feature_set
                    and row["model_family"] == family
                ]
            )
        )
    best_family_score = max(family_scores.values())
    tied_families = [
        family
        for family in bridge_cfg["model_family_candidates"]
        if best_family_score - family_scores[family] <= tolerance
    ]
    selected_family = "lr_l2" if "lr_l2" in tied_families else tied_families[0]

    y_valid = v7.labels_array(valid_rows)
    y_test = v7.labels_array(test_rows)
    final_records = []
    full_train_corpus_reference = common.fit_corpus_reference(train_rows, corpus_context)
    train_feature_rows = common.apply_corpus_reference(
        train_rows, full_train_corpus_reference, corpus_context
    )
    valid_feature_rows = common.apply_corpus_reference(
        valid_rows, full_train_corpus_reference, corpus_context
    )
    test_feature_rows = common.apply_corpus_reference(
        test_rows, full_train_corpus_reference, corpus_context
    )
    expert_train_control_feature_rows = common.apply_corpus_reference(
        expert_train_control_rows, full_train_corpus_reference, corpus_context
    )
    expert_valid_control_feature_rows = common.apply_corpus_reference(
        expert_valid_control_rows, full_train_corpus_reference, corpus_context
    )
    corpus_reference_path = staging_root / "full_train_corpus_reference.json"
    corpus_reference_path.write_text(
        json.dumps(full_train_corpus_reference, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    final_score_matrices: dict[str, dict[str, list[np.ndarray]]] = {
        "B0_lr_l2": {
            "valid": [],
            "internal_development_test": [],
            "evidence_expert_train_controls": [],
            "evidence_expert_valid_controls": [],
        },
        "selected_clean": {
            "valid": [],
            "internal_development_test": [],
            "evidence_expert_train_controls": [],
            "evidence_expert_valid_controls": [],
        },
    }
    for output_id, feature_set_id, family in (
        ("B0_lr_l2", "B0_v7_20d_plus_e5_latent64", "lr_l2"),
        ("selected_clean", selected_feature_set, selected_family),
    ):
        for seed in seeds:
            x_train, transform = common.fit_feature_transform(
                train_feature_rows, feature_set_id, policy, v7_policy, train_latent
            )
            x_valid = common.apply_feature_transform(
                valid_feature_rows, policy, v7_policy, transform, valid_latent
            )
            x_test = common.apply_feature_transform(
                test_feature_rows, policy, v7_policy, transform, test_latent
            )
            x_expert_train_controls = (
                common.apply_feature_transform(
                    expert_train_control_feature_rows,
                    policy,
                    v7_policy,
                    transform,
                    expert_train_control_latent,
                )
                if expert_train_control_rows
                else np.empty((0, x_train.shape[1]), dtype=float)
            )
            x_expert_valid_controls = (
                common.apply_feature_transform(
                    expert_valid_control_feature_rows,
                    policy,
                    v7_policy,
                    transform,
                    expert_valid_control_latent,
                )
                if expert_valid_control_rows
                else np.empty((0, x_train.shape[1]), dtype=float)
            )
            if family == "lr_l2":
                model = common.fit_lr(x_train, train_rows, policy, v7_policy)
            else:
                model = common.fit_pairwise_ranker(
                    x_train, train_rows, policy, v7_policy, seed
                )
            valid_scores = common.apply_model(x_valid, model)
            test_scores = common.apply_model(x_test, model)
            expert_train_control_scores = common.apply_model(
                x_expert_train_controls, model
            )
            expert_valid_control_scores = common.apply_model(
                x_expert_valid_controls, model
            )
            threshold = step7.choose_threshold(
                y_valid,
                valid_scores,
                policy["threshold_selection"]["metric"],
                policy,
            )
            valid_path = staging_root / "predictions" / f"{output_id}__seed_{seed}.zh_valid.csv"
            test_path = staging_root / "predictions" / f"{output_id}__seed_{seed}.internal_dev_test.csv"
            expert_train_control_path = (
                staging_root
                / "predictions"
                / f"{output_id}__seed_{seed}.evidence_expert_train_controls.csv"
            )
            expert_valid_control_path = (
                staging_root
                / "predictions"
                / f"{output_id}__seed_{seed}.evidence_expert_valid_controls.csv"
            )
            valid_path.parent.mkdir(parents=True, exist_ok=True)
            valid_path.write_bytes(
                common.render_csv(
                    prediction_rows(valid_rows, valid_scores, threshold, "representative_valid"),
                    PREDICTION_FIELDS,
                )
            )
            test_path.write_bytes(
                common.render_csv(
                    prediction_rows(test_rows, test_scores, threshold, "internal_development_test"),
                    PREDICTION_FIELDS,
                )
            )
            expert_train_control_path.write_bytes(
                common.render_csv(
                    prediction_rows(
                        expert_train_control_rows,
                        expert_train_control_scores,
                        threshold,
                        "evidence_expert_train_controls",
                    ),
                    PREDICTION_FIELDS,
                )
            )
            expert_valid_control_path.write_bytes(
                common.render_csv(
                    prediction_rows(
                        expert_valid_control_rows,
                        expert_valid_control_scores,
                        threshold,
                        "evidence_expert_valid_controls",
                    ),
                    PREDICTION_FIELDS,
                )
            )
            artifact_path = staging_root / "artifacts" / f"{output_id}__seed_{seed}.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact = {
                "artifact_type": "step15_v8_clean_bridge_model",
                "run_id": run_id,
                "output_id": output_id,
                "feature_set_id": feature_set_id,
                "model_family": family,
                "seed": seed,
                "feature_transform": transform,
                "model": model,
                "threshold_from_representative_valid": threshold,
                "train_pair_uid_sha256": common.canonical_hash(
                    sorted(row["pair_uid"] for row in train_rows)
                ),
                "full_train_corpus_reference_path": str(
                    (final_root / corpus_reference_path.relative_to(staging_root)).relative_to(ROOT)
                ).replace("\\", "/"),
                "full_train_corpus_reference_sha256": common.sha256(corpus_reference_path),
                "selection_source": "train_only_component_grouped_oof",
                "internal_test_used_for_selection": False,
            }
            artifact_path.write_text(
                json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            final_score_matrices[output_id]["valid"].append(valid_scores)
            final_score_matrices[output_id]["internal_development_test"].append(test_scores)
            final_score_matrices[output_id]["evidence_expert_train_controls"].append(
                expert_train_control_scores
            )
            final_score_matrices[output_id]["evidence_expert_valid_controls"].append(
                expert_valid_control_scores
            )
            final_records.append(
                {
                    "output_id": output_id,
                    "feature_set_id": feature_set_id,
                    "model_family": family,
                    "seed": seed,
                    "valid_metrics": step7.evaluate_probabilities(y_valid, valid_scores, threshold),
                    "internal_development_test_metrics": step7.evaluate_probabilities(
                        y_test, test_scores, threshold
                    ),
                    "artifact_path": str(
                        (final_root / artifact_path.relative_to(staging_root)).relative_to(ROOT)
                    ).replace("\\", "/"),
                    "valid_prediction_path": str(
                        (final_root / valid_path.relative_to(staging_root)).relative_to(ROOT)
                    ).replace("\\", "/"),
                    "internal_test_prediction_path": str(
                        (final_root / test_path.relative_to(staging_root)).relative_to(ROOT)
                    ).replace("\\", "/"),
                    "evidence_expert_train_control_prediction_path": str(
                        (
                            final_root
                            / expert_train_control_path.relative_to(staging_root)
                        ).relative_to(ROOT)
                    ).replace("\\", "/"),
                    "evidence_expert_valid_control_prediction_path": str(
                        (
                            final_root
                            / expert_valid_control_path.relative_to(staging_root)
                        ).relative_to(ROOT)
                    ).replace("\\", "/"),
                }
            )

    seed_mean_records = {}
    for output_id, matrices in final_score_matrices.items():
        valid_mean = np.mean(np.vstack(matrices["valid"]), axis=0)
        test_mean = np.mean(np.vstack(matrices["internal_development_test"]), axis=0)
        expert_train_control_mean = np.mean(
            np.vstack(matrices["evidence_expert_train_controls"]), axis=0
        )
        expert_valid_control_mean = np.mean(
            np.vstack(matrices["evidence_expert_valid_controls"]), axis=0
        )
        threshold = step7.choose_threshold(
            y_valid,
            valid_mean,
            policy["threshold_selection"]["metric"],
            policy,
        )
        valid_path = staging_root / "predictions" / f"{output_id}__seed_mean.zh_valid.csv"
        test_path = staging_root / "predictions" / f"{output_id}__seed_mean.internal_dev_test.csv"
        expert_train_control_path = (
            staging_root
            / "predictions"
            / f"{output_id}__seed_mean.evidence_expert_train_controls.csv"
        )
        expert_valid_control_path = (
            staging_root
            / "predictions"
            / f"{output_id}__seed_mean.evidence_expert_valid_controls.csv"
        )
        valid_path.write_bytes(
            common.render_csv(
                prediction_rows(valid_rows, valid_mean, threshold, "representative_valid"),
                PREDICTION_FIELDS,
            )
        )
        test_path.write_bytes(
            common.render_csv(
                prediction_rows(test_rows, test_mean, threshold, "internal_development_test"),
                PREDICTION_FIELDS,
            )
        )
        expert_train_control_path.write_bytes(
            common.render_csv(
                prediction_rows(
                    expert_train_control_rows,
                    expert_train_control_mean,
                    threshold,
                    "evidence_expert_train_controls",
                ),
                PREDICTION_FIELDS,
            )
        )
        expert_valid_control_path.write_bytes(
            common.render_csv(
                prediction_rows(
                    expert_valid_control_rows,
                    expert_valid_control_mean,
                    threshold,
                    "evidence_expert_valid_controls",
                ),
                PREDICTION_FIELDS,
            )
        )
        seed_mean_records[output_id] = {
            "threshold_from_representative_valid": threshold,
            "valid_metrics": step7.evaluate_probabilities(y_valid, valid_mean, threshold),
            "internal_development_test_metrics": step7.evaluate_probabilities(
                y_test, test_mean, threshold
            ),
            "valid_prediction_path": str(
                (final_root / valid_path.relative_to(staging_root)).relative_to(ROOT)
            ).replace("\\", "/"),
            "internal_test_prediction_path": str(
                (final_root / test_path.relative_to(staging_root)).relative_to(ROOT)
            ).replace("\\", "/"),
            "evidence_expert_train_control_prediction_path": str(
                (
                    final_root / expert_train_control_path.relative_to(staging_root)
                ).relative_to(ROOT)
            ).replace("\\", "/"),
            "evidence_expert_valid_control_prediction_path": str(
                (
                    final_root / expert_valid_control_path.relative_to(staging_root)
                ).relative_to(ROOT)
            ).replace("\\", "/"),
        }

    selection = {
        "feature_representation": {
            "candidate_mean_train_oof_macro_domain_average_precision": mean_oof,
            "best_score": best_score,
            "tied_within_tolerance": tied,
            "simplicity_tie_break_order": simplicity,
            "selected_feature_set_id": selected_feature_set,
            "selection_split": "train_oof_only",
        },
        "model_family": {
            "candidate_mean_train_oof_macro_domain_average_precision": family_scores,
            "best_score": best_family_score,
            "tied_within_tolerance": tied_families,
            "selected_model_family": selected_family,
            "selection_split": "train_oof_only",
        },
        "representative_valid_metrics_used_for_selection": False,
        "internal_test_metrics_used_for_selection": False,
    }
    summary = {
        "step": "step15_run_v8_bridge_audit",
        "version": policy["version"],
        "run_id": run_id,
        "split_counts": {
            name: {
                "total": len(rows),
                "positive": int(np.sum(v7.labels_array(rows))),
                "negative": int(len(rows) - np.sum(v7.labels_array(rows))),
                "component_count": len({common.component_group_key(row) for row in rows}),
            }
            for name, rows in splits.items()
        },
        "selection": selection,
        "oof_records": oof_records,
        "final_seed_records": final_records,
        "seed_mean": seed_mean_records,
        "current_internal_test_role": "diagnostic_only",
        "internal_test_satisfied_no_selection_or_promotion_gate": True,
        "inputs": {
            "policy_sha256": common.sha256(policy_path),
            "clean_semantics_manifest_sha256": common.sha256(
                root / policy["clean_semantics"]["output_subdirectory"] / "clean_semantics_manifest.json"
            ),
            "representative_validation_assignments_sha256": common.sha256(
                common.resolve(policy["frozen_dependencies"]["representative_validation_assignments"])
            ),
            "readiness_runtime_chain": runtime_chain,
        },
        "preprocessing_scope": {
            "oof_corpus_statistics": "fold_train_sellers_only",
            "oof_domain_normalization": "fold_train_rows_only",
            "oof_imputation": "fold_train_rows_only",
            "representative_valid_and_internal_test_reference": "full_train_sellers_only",
            "full_train_corpus_reference_path": str(
                (final_root / corpus_reference_path.relative_to(staging_root)).relative_to(ROOT)
            ).replace("\\", "/"),
            "full_train_corpus_reference_sha256": common.sha256(corpus_reference_path),
        },
    }
    summary["summary_sha256"] = common.canonical_hash(summary)
    summary_path = staging_root / "step15_v8_bridge_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fold_path = staging_root / "step15_v8_grouped_oof_fold_manifest.json"
    fold_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "group_key": bridge_cfg["group_key"],
                "folds": fold_manifests,
                "internal_test_used": False,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    staging_root.replace(final_root)
    print(
        json.dumps(
            {
                "status": "pass",
                "run_id": run_id,
                "selected_feature_set": selected_feature_set,
                "selected_model_family": selected_family,
                "summary": str((final_root / summary_path.name).relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
