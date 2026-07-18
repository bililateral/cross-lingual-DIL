#!/usr/bin/env python3
"""Evaluate the preregistered Step25-v3 copy-aware dual-channel models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import step15_v7_common as v7_common
import step24_common as step24
import step24_evaluate_content_independent_authorship as step24_eval
import step25_evaluate_template_decontaminated_authorship as step25_v1_eval
import step25_v2_common as step25_v2
import step25_v3_common as common


def load_feature_index(path: Path) -> dict[str, dict]:
    rows = step24.load_csv(path)
    index = {row["pair_uid"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Step25-v3 feature file contains duplicate pair_uid: {path}")
    return index


def fit_and_score_model(
    train_rows: list[dict],
    train_features: dict[str, dict],
    score_rows: list[dict],
    score_features: dict[str, dict],
    model_name: str,
    policy: dict,
    weighting_cfg: dict,
) -> tuple[np.ndarray, dict]:
    spec = policy["evaluation"]["model_specs"][model_name]
    feature_names = common.model_feature_names(policy, model_name)
    train_matrix = common.matrix_for_rows(train_rows, train_features, feature_names)
    score_matrix = common.matrix_for_rows(score_rows, score_features, feature_names)
    weights, weight_summary = common.normalized_factorized_weights(train_rows, weighting_cfg)
    artifact = common.fit_direction_constrained_logistic(
        train_matrix,
        v7_common.labels_array(train_rows),
        weights,
        feature_names,
        spec["coefficient_directions"],
        policy["evaluation"]["logistic"],
    )
    if not artifact["solver_converged"]:
        raise ValueError(f"Step25-v3 constrained LR/L2 did not converge: {model_name}")
    artifact.update(
        {
            "model_name": model_name,
            "role": spec["role"],
            "train_row_count": len(train_rows),
            "train_positive_count": int(
                np.sum(v7_common.labels_array(train_rows) == 1.0)
            ),
            "train_negative_count": int(
                np.sum(v7_common.labels_array(train_rows) == 0.0)
            ),
            "weight_summary": weight_summary,
            "train_pair_uid_sha256": step24.canonical_hash(
                sorted(row["pair_uid"] for row in train_rows)
            ),
        }
    )
    return common.apply_direction_constrained_logistic(score_matrix, artifact), artifact


def fit_grouped_oof(
    rows: list[dict],
    feature_index: dict[str, dict],
    policy: dict,
    weighting_cfg: dict,
    source_rows: list[dict] | None = None,
    source_features: dict[str, dict] | None = None,
) -> tuple[dict[str, np.ndarray], dict, dict[str, int]]:
    evaluation = policy["evaluation"]
    assignment = step24.balanced_component_folds(
        rows, int(evaluation["fold_count"]), int(evaluation["fold_seed"])
    )
    model_names = list(evaluation["model_specs"])
    scores = {name: np.full(len(rows), np.nan, dtype=np.float64) for name in model_names}
    artifacts = {}
    for fold in range(int(evaluation["fold_count"])):
        held = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if assignment[row["step24_component_id"]] == fold
            ],
            dtype=int,
        )
        held_set = set(held.tolist())
        train_rows = [row for index, row in enumerate(rows) if index not in held_set]
        train_features = feature_index
        if source_rows is not None:
            if source_features is None:
                raise ValueError("Step25-v3 source rows were provided without source features")
            train_rows = source_rows + train_rows
            train_features = {**source_features, **feature_index}
        held_rows = [rows[index] for index in held]
        labels = v7_common.labels_array(held_rows)
        if set(labels) != {0.0, 1.0}:
            raise ValueError(f"Step25-v3 grouped fold is single-class: {fold}")
        fold_record = {
            "held_out_row_count": len(held_rows),
            "held_out_positive_count": int(np.sum(labels == 1.0)),
            "held_out_negative_count": int(np.sum(labels == 0.0)),
            "held_out_component_ids": sorted(
                {row["step24_component_id"] for row in held_rows}
            ),
            "models": {},
        }
        for model_name in model_names:
            fold_scores, artifact = fit_and_score_model(
                train_rows,
                train_features,
                held_rows,
                feature_index,
                model_name,
                policy,
                weighting_cfg,
            )
            scores[model_name][held] = fold_scores
            fold_record["models"][model_name] = artifact
        artifacts[f"fold_{fold}"] = fold_record
    if any(np.any(~np.isfinite(values)) for values in scores.values()):
        raise ValueError("Step25-v3 failed to produce complete grouped-OOF scores")
    return scores, artifacts, assignment


def binary_feature_mask(
    rows: list[dict], feature_index: dict[str, dict], name: str, value: int
) -> np.ndarray:
    values = np.asarray(
        [float(feature_index[row["pair_uid"]][name]) for row in rows], dtype=float
    )
    if not np.all(np.isfinite(values)) or not set(np.unique(values)).issubset({0.0, 1.0}):
        raise ValueError(f"Step25-v3 binary feature is invalid: {name}")
    return values == float(value)


def masked_metrics(
    rows: list[dict], scores: dict[str, np.ndarray], mask: np.ndarray
) -> dict:
    indices = np.flatnonzero(mask)
    labels = v7_common.labels_array(rows)[indices]
    if len(indices) == 0 or len(np.unique(labels)) < 2:
        return {"status": "not_estimable_both_classes_required", "row_count": len(indices)}
    return {
        "status": "estimable",
        "row_count": int(len(indices)),
        "positive_count": int(np.sum(labels == 1.0)),
        "negative_count": int(np.sum(labels == 0.0)),
        "models": {
            name: step24_eval.metrics(labels, values[indices]) for name, values in scores.items()
        },
    }


def slice_ap_delta(slices: dict, name: str, primary: str, baseline: str) -> float:
    record = slices[name]
    if record.get("status") == "not_estimable_both_classes_required":
        raise ValueError(f"Step25-v3 required slice is not estimable: {name}")
    return float(
        record["models"][primary]["average_precision"]
        - record["models"][baseline]["average_precision"]
    )


def prediction_rows(
    rows: list[dict],
    assignment: dict[str, int],
    scores: dict[str, np.ndarray],
    feature_index: dict[str, dict],
    split_name: str,
) -> list[dict]:
    output = []
    for index, row in enumerate(rows):
        feature = feature_index[row["pair_uid"]]
        output.append(
            {
                "pair_uid": row["pair_uid"],
                "pool": row["step25_pool"],
                "split_name": split_name,
                "component_id": row["step24_component_id"],
                "fold": assignment[row["step24_component_id"]],
                "review_label": row["review_label"],
                "evidence_type": row["evidence_type"],
                "silver_train_only": row.get("silver_train_only", ""),
                "pair_local_style_reliable": feature["pair_local_style_reliable"],
                "global_style_reliable": feature["global_style_reliable"],
                "pair_local_maximum_mask_fraction": feature[
                    "pair_local_maximum_mask_fraction"
                ],
                **{
                    name: f"{float(values[index]):.12f}"
                    for name, values in sorted(scores.items())
                },
            }
        )
    return output


def assert_parent_fold_alignment(
    policy: dict, english_assignment: dict[str, int], target_assignment: dict[str, int]
) -> None:
    v2_root = common.resolve(policy["inputs"]["step25_v2_outputs_root"])
    v2_policy = json.loads(
        common.resolve(policy["inputs"]["step25_v2_policy"]).read_text(encoding="utf-8")
    )
    path = v2_root / v2_policy["outputs"]["model_artifacts"]
    parent = json.loads(path.read_text(encoding="utf-8"))
    if parent["english_fold_assignment"] != english_assignment:
        raise ValueError("Step25-v3 English folds differ from Step25-v2")
    if parent["target_fold_assignment"] != target_assignment:
        raise ValueError("Step25-v3 target folds differ from Step25-v2")


def missingness_corrected_design(
    train_rows: list[dict],
    train_index: dict[str, dict],
    score_rows: list[dict],
    score_index: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, dict]:
    names = [
        "decontaminated_pcm_multilingual_authorship_cosine",
        "decontaminated_mstyledistance_cosine",
    ]
    train_values = common.matrix_for_rows(train_rows, train_index, names)
    score_values = common.matrix_for_rows(score_rows, score_index, names)
    train_reliable = np.asarray(
        [bool(int(train_index[row["pair_uid"]]["decontaminated_pair_reliable"])) for row in train_rows]
    )
    score_reliable = np.asarray(
        [bool(int(score_index[row["pair_uid"]]["decontaminated_pair_reliable"])) for row in score_rows]
    )
    train_values = train_values.copy()
    score_values = score_values.copy()
    train_values[~train_reliable] = np.nan
    score_values[~score_reliable] = np.nan
    return step25_v2.matched_missingness_design(
        train_values,
        train_reliable,
        score_values,
        score_reliable,
        "fold_train_reliable_median_plus_indicator",
    )


def fit_missingness_control(
    train_rows: list[dict],
    train_index: dict[str, dict],
    score_rows: list[dict],
    score_index: dict[str, dict],
    logistic_cfg: dict,
    weighting_cfg: dict,
) -> tuple[np.ndarray, dict]:
    train_x, score_x, missingness = missingness_corrected_design(
        train_rows, train_index, score_rows, score_index
    )
    scores, artifact = step24_eval.fit_and_score(
        train_rows,
        train_x,
        score_x,
        [
            "global_clean_pcm_multilingual_authorship_cosine",
            "global_clean_mstyledistance_cosine",
            "global_style_reliable__indicator",
        ],
        logistic_cfg,
        weighting_cfg,
    )
    artifact["missingness"] = missingness
    return scores, artifact


def fit_missingness_grouped_oof(
    rows: list[dict],
    feature_index: dict[str, dict],
    assignment: dict[str, int],
    logistic_cfg: dict,
    weighting_cfg: dict,
    source_rows: list[dict] | None = None,
    source_index: dict[str, dict] | None = None,
) -> tuple[np.ndarray, dict]:
    scores = np.full(len(rows), np.nan, dtype=float)
    artifacts = {}
    for fold in sorted(set(assignment.values())):
        held = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if assignment[row["step24_component_id"]] == fold
            ],
            dtype=int,
        )
        held_set = set(held.tolist())
        train_rows = [row for index, row in enumerate(rows) if index not in held_set]
        train_index = feature_index
        if source_rows is not None:
            if source_index is None:
                raise ValueError("Step25-v3 missingness source index is absent")
            train_rows = source_rows + train_rows
            train_index = {**source_index, **feature_index}
        held_rows = [rows[index] for index in held]
        fold_scores, artifact = fit_missingness_control(
            train_rows,
            train_index,
            held_rows,
            feature_index,
            logistic_cfg,
            weighting_cfg,
        )
        scores[held] = fold_scores
        artifacts[f"fold_{fold}"] = artifact
    if np.any(~np.isfinite(scores)):
        raise ValueError("Step25-v3 missingness closure has incomplete OOF scores")
    return scores, artifacts


def load_frozen_v1_scores(policy: dict, pool_name: str) -> dict[str, dict]:
    root = common.resolve(policy["inputs"]["step25_v1_outputs_root"])
    name = (
        "step25_grouped_oof_predictions.en.csv"
        if pool_name == "en_content_train_pool"
        else "step25_grouped_oof_predictions.zh.csv"
    )
    rows = step24.load_csv(root / name)
    return {row["pair_uid"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    (
        policy_path,
        policy,
        v7_policy,
        step24_policy,
        step25_v1_policy,
        step25_v2_policy,
    ) = common.load_policy(args.policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "models": list(policy["evaluation"]["model_specs"]),
                    "primary": policy["evaluation"]["primary_model"],
                    "matched_baseline": policy["evaluation"]["matched_baseline_model"],
                    "valid_or_test_read": False,
                    "publication_promotion_allowed": False,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    parent_manifests = common.require_parent_manifests(policy)
    parent_references = common.frozen_parent_references(policy)
    root = common.resolve(policy["outputs_root"])
    feature_summary_path = root / policy["outputs"]["pair_feature_summary"]
    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    if feature_summary.get("valid_or_test_rows_read") != 0:
        raise ValueError("Step25-v3 feature builder read valid/test")
    if feature_summary.get("labels_or_evidence_types_used_as_features") is not False:
        raise ValueError("Step25-v3 features are not label-free")
    feature_paths = {
        "en_content_train_pool": root / policy["outputs"]["pair_features_en"],
        "zh_target_strict": root / policy["outputs"]["pair_features_zh"],
    }
    feature_indices = {name: load_feature_index(path) for name, path in feature_paths.items()}
    rows_by_pool = common.load_rows(step25_v2_policy, step24_policy, step25_v1_policy)
    en_rows = rows_by_pool["en_content_train_pool"]
    zh_rows = rows_by_pool["zh_target_strict"]
    for pool_name, rows in rows_by_pool.items():
        if set(feature_indices[pool_name]) != {row["pair_uid"] for row in rows}:
            raise ValueError(f"Step25-v3 feature boundary differs from canonical train: {pool_name}")
        forbidden = {"review_label", "evidence_type", "prob_positive", "model_score"}
        if any(forbidden & set(row) for row in feature_indices[pool_name].values()):
            raise ValueError(f"Step25-v3 feature table leaked supervision: {pool_name}")
    weighting_cfg = v7_policy["factorized_evidence_weighting"]
    model_names = list(policy["evaluation"]["model_specs"])
    source_scores = {}
    artifacts = {"source_only": {}, "english_grouped_oof": {}, "target_grouped_oof": {}}
    for model_name in model_names:
        scores, artifact = fit_and_score_model(
            en_rows,
            feature_indices["en_content_train_pool"],
            zh_rows,
            feature_indices["zh_target_strict"],
            model_name,
            policy,
            weighting_cfg,
        )
        source_scores[model_name] = scores
        artifacts["source_only"][model_name] = artifact
    english_oof, english_artifacts, english_assignment = fit_grouped_oof(
        en_rows,
        feature_indices["en_content_train_pool"],
        policy,
        weighting_cfg,
    )
    target_oof, target_artifacts, target_assignment = fit_grouped_oof(
        zh_rows,
        feature_indices["zh_target_strict"],
        policy,
        weighting_cfg,
        source_rows=en_rows,
        source_features=feature_indices["en_content_train_pool"],
    )
    artifacts["english_grouped_oof"] = english_artifacts
    artifacts["target_grouped_oof"] = target_artifacts
    assert_parent_fold_alignment(policy, english_assignment, target_assignment)

    en_labels = v7_common.labels_array(en_rows)
    zh_labels = v7_common.labels_array(zh_rows)
    model_metrics = {
        "english_grouped_oof": {
            name: step24_eval.metrics(en_labels, values) for name, values in english_oof.items()
        },
        "source_only_on_chinese_train": {
            name: step24_eval.metrics(zh_labels, values) for name, values in source_scores.items()
        },
        "target_grouped_oof": {
            name: step24_eval.metrics(zh_labels, values) for name, values in target_oof.items()
        },
    }
    fold_metrics = {
        "english_grouped_oof": step25_v1_eval.fold_metrics(
            en_rows, english_oof, english_assignment
        ),
        "target_grouped_oof": step25_v1_eval.fold_metrics(
            zh_rows, target_oof, target_assignment
        ),
    }
    source_named = {f"source_only_{name}": values for name, values in source_scores.items()}
    target_named = {f"target_oof_{name}": values for name, values in target_oof.items()}
    slices = step24_eval.slice_metrics(zh_rows, {**source_named, **target_named})
    reliable = binary_feature_mask(
        zh_rows, feature_indices["zh_target_strict"], "pair_local_style_reliable", 1
    )
    slices["pair_local_reliable_only"] = masked_metrics(
        zh_rows, {**source_named, **target_named}, reliable
    )
    slices["pair_local_unreliable_only"] = masked_metrics(
        zh_rows, {**source_named, **target_named}, ~reliable
    )

    primary = policy["evaluation"]["primary_model"]
    baseline = policy["evaluation"]["matched_baseline_model"]
    source_bootstrap = step24_eval.grouped_bootstrap(
        zh_rows,
        source_scores[baseline],
        source_scores[primary],
        int(policy["evaluation"]["grouped_bootstrap_resamples"]),
        int(policy["evaluation"]["grouped_bootstrap_seed"]) + 1,
    )
    target_bootstrap = step24_eval.grouped_bootstrap(
        zh_rows,
        target_oof[baseline],
        target_oof[primary],
        int(policy["evaluation"]["grouped_bootstrap_resamples"]),
        int(policy["evaluation"]["grouped_bootstrap_seed"]),
    )
    source_tail, _ = step25_v1_eval.tail_audit(
        zh_rows, source_scores[baseline], source_scores[primary]
    )
    target_tail, _ = step25_v1_eval.tail_audit(
        zh_rows, target_oof[baseline], target_oof[primary]
    )

    source_delta = float(
        model_metrics["source_only_on_chinese_train"][primary]["average_precision"]
        - model_metrics["source_only_on_chinese_train"][baseline]["average_precision"]
    )
    target_delta = float(
        model_metrics["target_grouped_oof"][primary]["average_precision"]
        - model_metrics["target_grouped_oof"][baseline]["average_precision"]
    )
    english_delta = float(
        model_metrics["english_grouped_oof"][primary]["average_precision"]
        - model_metrics["english_grouped_oof"][baseline]["average_precision"]
    )
    non_silver_delta = slice_ap_delta(
        slices, "canonical_non_silver", f"target_oof_{primary}", f"target_oof_{baseline}"
    )
    direct_delta = slice_ap_delta(
        slices,
        "direct_component_positive_plus_all_negatives",
        f"target_oof_{primary}",
        f"target_oof_{baseline}",
    )
    soft_delta = slice_ap_delta(
        slices,
        "soft_positive_plus_all_negatives",
        f"target_oof_{primary}",
        f"target_oof_{baseline}",
    )
    template_delta = target_tail["template_clone_not_controller"][
        "primary_minus_baseline_rank_tail"
    ]
    semantic_delta = target_tail["semantic_topic_not_controller"][
        "primary_minus_baseline_rank_tail"
    ]
    public_delta = target_tail["public_contact_or_url_noise"][
        "primary_minus_baseline_rank_tail"
    ]
    gates = policy["d0_to_d1_replication_gates"]
    gate_results = {
        "source_only_ap_non_degradation": source_delta
        >= float(gates["minimum_source_only_ap_delta_over_C0"]),
        "target_oof_ap_improvement": target_delta
        >= float(gates["minimum_target_oof_ap_delta_over_C0"]),
        "target_oof_bootstrap_lower_bound": target_bootstrap["ci95_lower"]
        >= float(gates["minimum_target_oof_grouped_bootstrap_lower_bound"]),
        "target_non_silver_non_degradation": non_silver_delta
        >= float(gates["minimum_target_non_silver_ap_delta"]),
        "target_direct_component_non_degradation": direct_delta
        >= float(gates["minimum_target_direct_component_ap_delta"]),
        "target_soft_positive_non_degradation": soft_delta
        >= float(gates["minimum_target_soft_positive_ap_delta"]),
        "target_template_violation_reduction": template_delta[
            "vs_strong_positive_violation_rate"
        ]
        <= float(gates["maximum_target_template_clone_violation_rate_delta"]),
        "target_template_mean_rank_nonincrease": template_delta["mean_rank_percentile"]
        <= float(gates["maximum_target_template_clone_mean_rank_percentile_delta"]),
        "target_semantic_mean_rank_bounded": semantic_delta["mean_rank_percentile"]
        <= float(gates["maximum_target_semantic_topic_mean_rank_percentile_delta"]),
        "target_public_mean_rank_nonincrease": public_delta["mean_rank_percentile"]
        <= float(gates["maximum_target_public_noise_mean_rank_percentile_delta"]),
        "english_oof_ap_non_degradation": english_delta
        >= float(gates["minimum_source_only_english_ap_delta_over_C0"]),
    }
    d1_eligible = all(gate_results.values())

    # Pure global-style missingness closure; it cannot influence C2 or any gate.
    v1_feature_root = common.resolve(policy["inputs"]["step25_v1_outputs_root"])
    v1_indices = {
        "en_content_train_pool": load_feature_index(
            v1_feature_root / step25_v1_policy["outputs"]["pair_features_en"]
        ),
        "zh_target_strict": load_feature_index(
            v1_feature_root / step25_v1_policy["outputs"]["pair_features_zh"]
        ),
    }
    closure_source, closure_source_artifact = fit_missingness_control(
        en_rows,
        v1_indices["en_content_train_pool"],
        zh_rows,
        v1_indices["zh_target_strict"],
        policy["evaluation"]["logistic"],
        weighting_cfg,
    )
    closure_english, closure_english_artifacts = fit_missingness_grouped_oof(
        en_rows,
        v1_indices["en_content_train_pool"],
        english_assignment,
        policy["evaluation"]["logistic"],
        weighting_cfg,
    )
    closure_target, closure_target_artifacts = fit_missingness_grouped_oof(
        zh_rows,
        v1_indices["zh_target_strict"],
        target_assignment,
        policy["evaluation"]["logistic"],
        weighting_cfg,
        source_rows=en_rows,
        source_index=v1_indices["en_content_train_pool"],
    )
    frozen_v1_en = load_frozen_v1_scores(policy, "en_content_train_pool")
    frozen_v1_zh = load_frozen_v1_scores(policy, "zh_target_strict")
    closure = {
        "selection_or_gate_use": False,
        "pair_local_reliability_intersection_used": False,
        "english_grouped_oof": {
            "fixed_zero_reference": step24_eval.metrics(
                en_labels,
                np.asarray(
                    [
                        float(frozen_v1_en[row["pair_uid"]]["source_oof_decontaminated_style_lr_l2_primary"])
                        for row in en_rows
                    ]
                ),
            ),
            "median_plus_indicator": step24_eval.metrics(en_labels, closure_english),
        },
        "source_only_on_chinese_train": {
            "fixed_zero_reference": step24_eval.metrics(
                zh_labels,
                np.asarray(
                    [
                        float(frozen_v1_zh[row["pair_uid"]]["source_only_decontaminated_style_lr_l2_primary"])
                        for row in zh_rows
                    ]
                ),
            ),
            "median_plus_indicator": step24_eval.metrics(zh_labels, closure_source),
        },
        "target_grouped_oof": {
            "fixed_zero_reference": step24_eval.metrics(
                zh_labels,
                np.asarray(
                    [
                        float(frozen_v1_zh[row["pair_uid"]]["target_oof_decontaminated_style_lr_l2_primary"])
                        for row in zh_rows
                    ]
                ),
            ),
            "median_plus_indicator": step24_eval.metrics(zh_labels, closure_target),
        },
    }
    artifacts["missingness_closure"] = {
        "source_only": closure_source_artifact,
        "english_grouped_oof": closure_english_artifacts,
        "target_grouped_oof": closure_target_artifacts,
    }

    en_path = root / policy["outputs"]["predictions_en"]
    zh_path = root / policy["outputs"]["predictions_zh"]
    step24.write_csv_immutable(
        en_path,
        prediction_rows(
            en_rows,
            english_assignment,
            {f"english_oof_{name}": values for name, values in english_oof.items()},
            feature_indices["en_content_train_pool"],
            "train_grouped_oof",
        ),
    )
    step24.write_csv_immutable(
        zh_path,
        prediction_rows(
            zh_rows,
            target_assignment,
            {**source_named, **target_named},
            feature_indices["zh_target_strict"],
            "train_source_and_grouped_oof",
        ),
    )
    artifact_payload = {
        "step": "step25_v3_copy_aware_dual_channel_model_artifacts",
        "version": policy["version"],
        "status": "pass",
        "english_fold_assignment": english_assignment,
        "target_fold_assignment": target_assignment,
        "artifacts": artifacts,
        "policy_sha256": step24.sha256_file(policy_path),
        "pair_feature_summary_sha256": step24.sha256_file(feature_summary_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    artifact_payload["payload_sha256"] = step24.canonical_hash(artifact_payload)
    artifact_path = root / policy["outputs"]["model_artifacts"]
    step24.write_json_immutable(artifact_path, artifact_payload)
    summary = {
        "step": "step25_v3_copy_aware_dual_channel",
        "version": policy["version"],
        "status": "pass",
        "boundary": policy["boundary"],
        "step25_v1_conclusion_reversed": False,
        "step25_v2_conclusion_reversed": False,
        "d1_replication_candidate_eligible": d1_eligible,
        "publication_promotion_eligible": False,
        "step11_or_step17_entry_allowed": False,
        "valid_or_test_rows_read_or_scored": 0,
        "model_or_threshold_selected_on_d0": False,
        "synthetic_text_pair_or_identity_label_count": 0,
        "frozen_parent_manifests": parent_manifests,
        "frozen_parent_references": parent_references,
        "dataset_counts": {
            "english_train": step24_eval.label_counts(en_rows),
            "chinese_train": step24_eval.label_counts(zh_rows),
            "chinese_pair_local_reliable_count": int(np.sum(reliable)),
            "chinese_pair_local_unreliable_count": int(np.sum(~reliable)),
        },
        "model_metrics": model_metrics,
        "fold_metrics": fold_metrics,
        "slice_metrics": slices,
        "paired_component_grouped_bootstrap": {
            "source_only_C2_minus_C0": source_bootstrap,
            "target_grouped_oof_C2_minus_C0": target_bootstrap,
        },
        "negative_tail_audit": {
            "source_only_C2_vs_C0": source_tail,
            "target_grouped_oof_C2_vs_C0": target_tail,
        },
        "key_deltas": {
            "source_only_C2_minus_C0_average_precision": source_delta,
            "target_grouped_oof_C2_minus_C0_average_precision": target_delta,
            "english_grouped_oof_C2_minus_C0_average_precision": english_delta,
            "target_non_silver_C2_minus_C0_average_precision": non_silver_delta,
            "target_direct_component_C2_minus_C0_average_precision": direct_delta,
            "target_soft_positive_C2_minus_C0_average_precision": soft_delta,
            "target_template_rank_tail_C2_minus_C0": template_delta,
            "target_semantic_rank_tail_C2_minus_C0": semantic_delta,
            "target_public_rank_tail_C2_minus_C0": public_delta,
        },
        "d0_to_d1_gate_results": gate_results,
        "d0_to_d1_gate_thresholds": gates,
        "missingness_only_closure_control": closure,
        "outputs": {
            "predictions_en": str(en_path.relative_to(common.ROOT)).replace("\\", "/"),
            "predictions_zh": str(zh_path.relative_to(common.ROOT)).replace("\\", "/"),
            "model_artifacts": str(artifact_path.relative_to(common.ROOT)).replace("\\", "/"),
        },
        "policy_sha256": step24.sha256_file(policy_path),
        "pair_feature_summary_sha256": step24.sha256_file(feature_summary_path),
        "predictions_en_sha256": step24.sha256_file(en_path),
        "predictions_zh_sha256": step24.sha256_file(zh_path),
        "model_artifacts_sha256": step24.sha256_file(artifact_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    summary["summary_sha256"] = step24.canonical_hash(summary)
    summary_path = root / policy["outputs"]["evaluation_summary"]
    step24.write_json_immutable(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "d1_replication_candidate_eligible": d1_eligible,
                "publication_promotion_eligible": False,
                "source_only_C2_minus_C0_ap": source_delta,
                "target_oof_C2_minus_C0_ap": target_delta,
                "summary": str(summary_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
