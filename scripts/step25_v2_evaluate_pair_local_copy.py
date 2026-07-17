#!/usr/bin/env python3
"""Evaluate the isolated Step25-v2 pair-local copy diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import step15_v7_common as v7_common
import step24_common as step24
import step24_evaluate_content_independent_authorship as step24_eval
import step25_evaluate_template_decontaminated_authorship as step25_v1_eval
import step25_v2_common as common


def load_feature_index(path: Path) -> dict[str, dict]:
    rows = step24.load_csv(path)
    index = {row["pair_uid"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Step25-v2 pair feature file contains duplicates: {path}")
    return index


def matrix_for_rows_allow_nan(
    rows: list[dict], index: dict[str, dict], names: list[str]
) -> np.ndarray:
    matrix = []
    for row in rows:
        feature_row = index.get(row["pair_uid"])
        if feature_row is None:
            raise ValueError(f"Step25-v2 pair feature is missing: {row['pair_uid']}")
        matrix.append([float(feature_row[name]) for name in names])
    return np.asarray(matrix, dtype=np.float64)


def reliability_for_rows(
    rows: list[dict], index: dict[str, dict], name: str
) -> np.ndarray:
    values = matrix_for_rows_allow_nan(rows, index, [name])[:, 0]
    if not np.all(np.isfinite(values)) or not set(np.unique(values)).issubset({0.0, 1.0}):
        raise ValueError(f"Step25-v2 reliability feature is not binary: {name}")
    return values.astype(bool)


def fit_and_score_spec(
    train_rows: list[dict],
    train_features: dict[str, dict],
    score_rows: list[dict],
    score_features: dict[str, dict],
    model_name: str,
    spec: dict,
    logistic_cfg: dict,
    weighting_cfg: dict,
) -> tuple[np.ndarray, dict]:
    style_names = list(spec["style_features"])
    reliability_name = spec["reliability_feature"]
    train_style = matrix_for_rows_allow_nan(train_rows, train_features, style_names)
    score_style = matrix_for_rows_allow_nan(score_rows, score_features, style_names)
    train_reliable = reliability_for_rows(train_rows, train_features, reliability_name)
    score_reliable = reliability_for_rows(score_rows, score_features, reliability_name)
    train_design, score_design, missingness_artifact = common.matched_missingness_design(
        train_style,
        train_reliable,
        score_style,
        score_reliable,
        spec["missingness_mode"],
    )
    design_names = style_names + [f"{reliability_name}__indicator"]
    scores, artifact = step24_eval.fit_and_score(
        train_rows,
        train_design,
        score_design,
        design_names,
        logistic_cfg,
        weighting_cfg,
    )
    artifact.update(
        {
            "model_name": model_name,
            "selection_role": spec["selection_role"],
            "raw_style_feature_names": style_names,
            "reliability_feature": reliability_name,
            "missingness": missingness_artifact,
        }
    )
    return scores, artifact


def fit_grouped_oof(
    rows: list[dict],
    feature_index: dict[str, dict],
    model_specs: dict[str, dict],
    logistic_cfg: dict,
    weighting_cfg: dict,
    fold_count: int,
    fold_seed: int,
    source_rows: list[dict] | None = None,
    source_features: dict[str, dict] | None = None,
) -> tuple[dict[str, np.ndarray], dict, dict[str, int]]:
    assignment = step24.balanced_component_folds(rows, fold_count, fold_seed)
    oof = {name: np.full(len(rows), np.nan, dtype=float) for name in model_specs}
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
        train_indices = [index for index in range(len(rows)) if index not in held_set]
        fold_rows = [rows[index] for index in train_indices]
        fold_features = feature_index
        if source_rows is not None:
            if source_features is None:
                raise ValueError("Step25-v2 source rows were provided without features")
            fold_rows = source_rows + fold_rows
            fold_features = {**source_features, **feature_index}
        held_rows = [rows[index] for index in held]
        held_positive = sum(row["review_label"] == "positive" for row in held_rows)
        held_negative = len(held_rows) - held_positive
        if not held_positive or not held_negative:
            raise ValueError(f"Step25-v2 grouped fold {fold} is single-class")
        fold_record = {
            "held_out_row_count": len(held_rows),
            "held_out_positive_count": held_positive,
            "held_out_negative_count": held_negative,
            "held_out_component_ids": sorted(
                {row["step24_component_id"] for row in held_rows}
            ),
            "models": {},
        }
        for model_name, spec in model_specs.items():
            fold_scores, artifact = fit_and_score_spec(
                fold_rows,
                fold_features,
                held_rows,
                feature_index,
                model_name,
                spec,
                logistic_cfg,
                weighting_cfg,
            )
            oof[model_name][held] = fold_scores
            fold_record["models"][model_name] = artifact
        artifacts[f"fold_{fold}"] = fold_record
    if any(np.any(~np.isfinite(scores)) for scores in oof.values()):
        raise ValueError("Step25-v2 failed to score every grouped-OOF row exactly once")
    return oof, artifacts, assignment


def reliable_only_metrics(
    rows: list[dict], scores: dict[str, np.ndarray], reliable: np.ndarray
) -> dict:
    indices = np.flatnonzero(reliable)
    labels = v7_common.labels_array(rows)[indices]
    if len(indices) == 0 or len(np.unique(labels)) < 2:
        return {
            "status": "not_estimable_both_classes_required",
            "row_count": int(len(indices)),
        }
    return {
        "status": "estimable",
        "row_count": int(len(indices)),
        "positive_count": int(np.sum(labels == 1.0)),
        "negative_count": int(np.sum(labels == 0.0)),
        "models": {
            name: step24_eval.metrics(labels, value[indices]) for name, value in scores.items()
        },
    }


def slice_ap_delta(
    slices: dict, slice_name: str, candidate_name: str, baseline_name: str
) -> float:
    record = slices[slice_name]
    if record.get("status") == "not_estimable_both_classes_required":
        raise ValueError(f"Step25-v2 required slice is not estimable: {slice_name}")
    models = record["models"]
    return float(
        models[candidate_name]["average_precision"]
        - models[baseline_name]["average_precision"]
    )


def prediction_rows(
    rows: list[dict],
    assignment: dict[str, int],
    scores: dict[str, np.ndarray],
    feature_index: dict[str, dict],
    split_name: str,
) -> list[dict]:
    ordered_names = sorted(scores)
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
                "global_and_pair_local_style_reliable": feature[
                    "global_and_pair_local_style_reliable"
                ],
                **{
                    name: f"{float(values[index]):.12f}"
                    for name, values in scores.items()
                    if name in ordered_names
                },
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path, policy, step24_policy, step25_v1_policy = common.load_policy(args.policy)
    evaluation = policy["evaluation"]
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "models": list(evaluation["model_specs"]),
                    "primary_diagnostic": evaluation["primary_model"],
                    "matched_baseline": evaluation["matched_baseline_model"],
                    "missing_cosine_zero_forbidden": True,
                    "fold_local_imputation": True,
                    "valid_or_test_read": False,
                    "publication_promotion_allowed": False,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    output_root = common.resolve(policy["outputs_root"])
    summary_path = output_root / policy["outputs"]["pair_feature_summary"]
    feature_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if feature_summary.get("valid_test_pair_count") != 0:
        raise ValueError("Step25-v2 pair features contain valid/test rows")
    feature_paths = {
        "en_content_train_pool": output_root / policy["outputs"]["pair_features_en"],
        "zh_target_strict": output_root / policy["outputs"]["pair_features_zh"],
    }
    feature_indices = {
        pool: load_feature_index(path) for pool, path in feature_paths.items()
    }
    required_feature_names = {
        name
        for spec in evaluation["model_specs"].values()
        for name in [*spec["style_features"], spec["reliability_feature"]]
    }
    missing_feature_names = required_feature_names - set(feature_summary["feature_names"])
    if missing_feature_names:
        raise ValueError(
            f"Step25-v2 pair feature summary is incomplete: {sorted(missing_feature_names)}"
        )
    if feature_summary.get("labels_evidence_types_or_scores_used_to_build_features") is not False:
        raise ValueError("Step25-v2 pair features were not built label-free")
    for pool_name, index in feature_indices.items():
        forbidden = {"review_label", "evidence_type", "prob_positive", "model_score"}
        for feature_row in index.values():
            if feature_row.get("pool") != pool_name or feature_row.get("split_name") != "train":
                raise ValueError(f"Step25-v2 pair feature boundary differs: {pool_name}")
            leaked = forbidden & set(feature_row)
            if leaked:
                raise ValueError(
                    f"Step25-v2 pair features leaked supervision fields: {pool_name}:{sorted(leaked)}"
                )
    rows_by_pool = common.load_rows(policy, step24_policy, step25_v1_policy)
    en_rows = rows_by_pool["en_content_train_pool"]
    zh_rows = rows_by_pool["zh_target_strict"]
    if {row["pair_uid"] for row in en_rows} & {row["pair_uid"] for row in zh_rows}:
        raise ValueError("Step25-v2 source and target pair UIDs overlap")
    if {row["step24_component_id"] for row in en_rows} & {
        row["step24_component_id"] for row in zh_rows
    }:
        raise ValueError("Step25-v2 source and target seller components overlap")
    if {
        row[field]
        for row in en_rows
        for field in ("seller_uid_left", "seller_uid_right")
    } & {
        row[field]
        for row in zh_rows
        for field in ("seller_uid_left", "seller_uid_right")
    }:
        raise ValueError("Step25-v2 source and target sellers overlap")
    if set(feature_indices["en_content_train_pool"]) != {
        row["pair_uid"] for row in en_rows
    }:
        raise ValueError("Step25-v2 English feature boundary differs from canonical train")
    if set(feature_indices["zh_target_strict"]) != {
        row["pair_uid"] for row in zh_rows
    }:
        raise ValueError("Step25-v2 Chinese feature boundary differs from canonical train")

    v7_policy = json.loads(
        common.resolve(policy["inputs"]["v7_policy"]).read_text(encoding="utf-8")
    )
    weighting_cfg = v7_policy["factorized_evidence_weighting"]
    model_specs = evaluation["model_specs"]
    logistic_cfg = evaluation["logistic"]
    source_scores = {}
    artifacts = {"source_only": {}, "english_grouped_oof": {}, "target_grouped_oof": {}}
    for model_name, spec in model_specs.items():
        values, artifact = fit_and_score_spec(
            en_rows,
            feature_indices["en_content_train_pool"],
            zh_rows,
            feature_indices["zh_target_strict"],
            model_name,
            spec,
            logistic_cfg,
            weighting_cfg,
        )
        source_scores[model_name] = values
        artifacts["source_only"][model_name] = artifact

    english_oof, english_artifacts, english_assignment = fit_grouped_oof(
        en_rows,
        feature_indices["en_content_train_pool"],
        model_specs,
        logistic_cfg,
        weighting_cfg,
        int(evaluation["fold_count"]),
        int(evaluation["fold_seed"]),
    )
    artifacts["english_grouped_oof"] = english_artifacts
    target_oof, target_artifacts, target_assignment = fit_grouped_oof(
        zh_rows,
        feature_indices["zh_target_strict"],
        model_specs,
        logistic_cfg,
        weighting_cfg,
        int(evaluation["fold_count"]),
        int(evaluation["fold_seed"]),
        source_rows=en_rows,
        source_features=feature_indices["en_content_train_pool"],
    )
    artifacts["target_grouped_oof"] = target_artifacts

    en_labels = v7_common.labels_array(en_rows)
    zh_labels = v7_common.labels_array(zh_rows)
    model_metrics = {
        "english_grouped_oof": {
            name: step24_eval.metrics(en_labels, values)
            for name, values in english_oof.items()
        },
        "source_only_on_chinese_train": {
            name: step24_eval.metrics(zh_labels, values)
            for name, values in source_scores.items()
        },
        "target_grouped_oof": {
            name: step24_eval.metrics(zh_labels, values)
            for name, values in target_oof.items()
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
    source_named = {f"source_only_{name}": value for name, value in source_scores.items()}
    target_named = {f"target_oof_{name}": value for name, value in target_oof.items()}
    slices = step24_eval.slice_metrics(zh_rows, {**source_named, **target_named})
    local_reliable = reliability_for_rows(
        zh_rows,
        feature_indices["zh_target_strict"],
        "pair_local_style_reliable",
    )
    p4_models = list(evaluation["P4_reliable_pair_only_sensitivity"]["compare"])
    reliable_only = {
        "source_only": reliable_only_metrics(
            zh_rows,
            {name: source_scores[name] for name in p4_models},
            local_reliable,
        ),
        "target_grouped_oof": reliable_only_metrics(
            zh_rows,
            {name: target_oof[name] for name in p4_models},
            local_reliable,
        ),
    }

    primary = evaluation["primary_model"]
    baseline = evaluation["matched_baseline_model"]
    source_bootstrap = step24_eval.grouped_bootstrap(
        zh_rows,
        source_scores[baseline],
        source_scores[primary],
        int(evaluation["grouped_bootstrap_resamples"]),
        int(evaluation["grouped_bootstrap_seed"]) + 1,
    )
    target_bootstrap = step24_eval.grouped_bootstrap(
        zh_rows,
        target_oof[baseline],
        target_oof[primary],
        int(evaluation["grouped_bootstrap_resamples"]),
        int(evaluation["grouped_bootstrap_seed"]),
    )
    source_tail, _source_tail_deltas = step25_v1_eval.tail_audit(
        zh_rows, source_scores[baseline], source_scores[primary]
    )
    target_tail, _target_tail_deltas = step25_v1_eval.tail_audit(
        zh_rows, target_oof[baseline], target_oof[primary]
    )

    source_ap_delta = float(
        model_metrics["source_only_on_chinese_train"][primary]["average_precision"]
        - model_metrics["source_only_on_chinese_train"][baseline]["average_precision"]
    )
    target_ap_delta = float(
        model_metrics["target_grouped_oof"][primary]["average_precision"]
        - model_metrics["target_grouped_oof"][baseline]["average_precision"]
    )
    target_direct_delta = slice_ap_delta(
        slices,
        "direct_component_positive_plus_all_negatives",
        f"target_oof_{primary}",
        f"target_oof_{baseline}",
    )
    template_delta = target_tail["template_clone_not_controller"][
        "primary_minus_baseline_rank_tail"
    ]
    reliable_fraction = float(np.mean(local_reliable))
    gate_cfg = policy["mechanism_diagnostic_gates"]
    gate_results = {
        "source_only_ap_non_degradation": source_ap_delta
        >= float(gate_cfg["minimum_source_only_ap_delta_over_matched_raw"]),
        "target_oof_ap_non_degradation": target_ap_delta
        >= float(gate_cfg["minimum_target_oof_ap_delta_over_matched_raw"]),
        "source_only_bootstrap_non_degradation": source_bootstrap["ci95_lower"]
        >= float(gate_cfg["minimum_source_only_bootstrap_lower_bound"]),
        "target_oof_bootstrap_non_degradation": target_bootstrap["ci95_lower"]
        >= float(gate_cfg["minimum_target_oof_bootstrap_lower_bound"]),
        "target_direct_component_non_degradation": target_direct_delta
        >= float(gate_cfg["minimum_target_direct_component_ap_delta"]),
        "target_template_clone_rank_reduction": template_delta["mean_rank_percentile"]
        <= float(gate_cfg["maximum_target_template_clone_mean_rank_percentile_delta"]),
        "target_template_clone_violation_reduction": template_delta[
            "vs_strong_positive_violation_rate"
        ]
        <= float(gate_cfg["maximum_target_template_clone_violation_rate_delta"]),
        "minimum_reliable_pair_fraction": reliable_fraction
        >= float(gate_cfg["minimum_reliable_pair_fraction"]),
    }
    mechanism_supported = all(gate_results.values())

    en_prediction_path = output_root / policy["outputs"]["predictions_en"]
    zh_prediction_path = output_root / policy["outputs"]["predictions_zh"]
    step24.write_csv_immutable(
        en_prediction_path,
        prediction_rows(
            en_rows,
            english_assignment,
            {f"english_oof_{key}": value for key, value in english_oof.items()},
            feature_indices["en_content_train_pool"],
            "train_grouped_oof",
        ),
    )
    step24.write_csv_immutable(
        zh_prediction_path,
        prediction_rows(
            zh_rows,
            target_assignment,
            {**source_named, **target_named},
            feature_indices["zh_target_strict"],
            "train_source_and_grouped_oof",
        ),
    )
    artifacts_payload = {
        "step": "step25_v2_pair_local_copy_model_artifacts",
        "version": policy["version"],
        "status": "pass",
        "fold_local_missingness_fit": True,
        "missing_style_encoded_as_fixed_zero": False,
        "english_fold_assignment": english_assignment,
        "target_fold_assignment": target_assignment,
        "artifacts": artifacts,
        "policy_sha256": step24.sha256_file(policy_path),
        "pair_feature_summary_sha256": step24.sha256_file(summary_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    artifacts_payload["payload_sha256"] = step24.canonical_hash(artifacts_payload)
    artifacts_path = output_root / policy["outputs"]["model_artifacts"]
    step24.write_json_immutable(artifacts_path, artifacts_payload)

    summary = {
        "step": "step25_v2_pair_local_copy_missingness_diagnostic",
        "version": policy["version"],
        "status": "pass",
        "boundary": policy["boundary"],
        "step25_v1_conclusion_reversed": False,
        "mechanism_hypothesis_supported": mechanism_supported,
        "d1_candidate_eligible": False,
        "publication_promotion_eligible": False,
        "step11_or_step17_entry_allowed": False,
        "method_selected_from_d0_valid_or_test": False,
        "valid_or_test_rows_read_or_scored": 0,
        "synthetic_text_pair_or_identity_label_count": 0,
        "encoder_parameters_updated": False,
        "missing_style_encoded_as_fixed_zero": False,
        "fold_local_imputation": True,
        "dataset_counts": {
            "english_train": step24_eval.label_counts(en_rows),
            "chinese_train": step24_eval.label_counts(zh_rows),
            "chinese_pair_local_reliable_count": int(np.sum(local_reliable)),
            "chinese_pair_local_unreliable_count": int(np.sum(~local_reliable)),
            "chinese_pair_local_reliable_fraction": reliable_fraction,
        },
        "model_metrics": model_metrics,
        "fold_metrics": fold_metrics,
        "slice_metrics": slices,
        "P4_reliable_pair_only_sensitivity": reliable_only,
        "paired_component_grouped_bootstrap": {
            "source_only_P2_minus_P0": source_bootstrap,
            "target_grouped_oof_P2_minus_P0": target_bootstrap,
        },
        "negative_tail_audit": {
            "source_only_P2_vs_P0": source_tail,
            "target_grouped_oof_P2_vs_P0": target_tail,
        },
        "key_deltas": {
            "source_only_P2_minus_P0_average_precision": source_ap_delta,
            "target_grouped_oof_P2_minus_P0_average_precision": target_ap_delta,
            "target_grouped_oof_direct_component_slice_P2_minus_P0_average_precision": target_direct_delta,
            "target_template_clone_rank_tail_P2_minus_P0": template_delta,
        },
        "mechanism_gate_results": gate_results,
        "mechanism_gate_thresholds": gate_cfg,
        "interpretation_constraint": gate_cfg["interpretation"],
        "outputs": {
            "predictions_en": str(en_prediction_path.relative_to(common.ROOT)).replace("\\", "/"),
            "predictions_zh": str(zh_prediction_path.relative_to(common.ROOT)).replace("\\", "/"),
            "model_artifacts": str(artifacts_path.relative_to(common.ROOT)).replace("\\", "/"),
        },
        "policy_sha256": step24.sha256_file(policy_path),
        "pair_feature_summary_sha256": step24.sha256_file(summary_path),
        "predictions_en_sha256": step24.sha256_file(en_prediction_path),
        "predictions_zh_sha256": step24.sha256_file(zh_prediction_path),
        "model_artifacts_sha256": step24.sha256_file(artifacts_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    summary["summary_sha256"] = step24.canonical_hash(summary)
    evaluation_path = output_root / policy["outputs"]["evaluation_summary"]
    step24.write_json_immutable(evaluation_path, summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "mechanism_hypothesis_supported": mechanism_supported,
                "d1_candidate_eligible": False,
                "publication_promotion_eligible": False,
                "source_only_P2_minus_P0_ap": source_ap_delta,
                "target_oof_P2_minus_P0_ap": target_ap_delta,
                "summary": str(evaluation_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
