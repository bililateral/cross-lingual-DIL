#!/usr/bin/env python3
"""Run Step9-v7 latent pair mixup and equal-effective-weight controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7
import step9_run_few_shot_adaptation as step9
import step15_v7_common as common


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step15_v7_two_stage_policy.json"


def render_csv(rows: list[dict], fields: list[str]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Step9-v7 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def row_index(rows: list[dict]) -> dict[str, int]:
    return {row["pair_uid"]: index for index, row in enumerate(rows)}


def select_rows(matrix: np.ndarray, index: dict[str, int], rows: list[dict]) -> np.ndarray:
    return np.asarray([matrix[index[row["pair_uid"]]] for row in rows], dtype=float)


def prediction_rows(
    rows: list[dict], probabilities: np.ndarray, threshold: float, split: str
) -> list[dict]:
    return [
        {
            "pair_uid": row["pair_uid"],
            "split_name": split,
            "review_label": row["review_label"],
            "evidence_type": row["evidence_type"],
            "v7_component_id": row["v7_component_id"],
            "prob_positive": f"{float(probability):.12f}",
            "selected_threshold": f"{threshold:.12f}",
            "predicted_label": int(float(probability) >= threshold),
        }
        for row, probability in zip(rows, probabilities, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--ratio", action="append", type=float)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path = common.resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    cfg = policy["step9_latent_mixup"]
    ratios = args.ratio or [float(value) for value in cfg["support_ratios"]]
    seeds = args.seed or [int(value) for value in cfg["seeds"]]
    experiments = list(cfg["experiments"])
    if experiments != [
        "no_augmentation",
        "equal_effective_weight_duplication",
        "latent_pair_embedding_mixup",
    ]:
        raise ValueError("Step9-v7 requires the preregistered three-way augmentation control")
    if args.validate_config_only:
        print(json.dumps({"status": "pass", "ratios": ratios, "seeds": seeds, "experiments": experiments}, indent=2))
        return
    run_id = args.run_id or cfg["default_run_id"]
    if not run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Step9-v7 run-id may contain only letters, digits, underscore, and hyphen")
    output_root = common.resolve(cfg["outputs_root"]) / run_id
    staging_root = output_root.with_name(f".{output_root.name}.incomplete")
    if output_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Step9-v7 final or incomplete run directory already exists: {output_root} / {staging_root}"
        )

    def staged(final_path: Path) -> Path:
        return staging_root / final_path.relative_to(output_root)

    pools = common.load_joined_rows(policy)
    en_rows = pools["en_content_train_pool"]
    zh_rows = pools["zh_target_strict"]
    en_train = [row for row in en_rows if row["v7_split_name"] == "train"]
    zh_train = [row for row in zh_rows if row["v7_split_name"] == "train"]
    zh_valid = [row for row in zh_rows if row["v7_split_name"] == "valid"]
    zh_test = [row for row in zh_rows if row["v7_split_name"] == "internal_development_test"]
    if len(zh_test) != int(policy["evaluation"]["current_zh_test_row_count_expected"]):
        raise ValueError("Current internal-development test row count changed")

    features = list(policy["inductive_features"]["stable_strict_clean_features"])
    latent_cfg = policy["latent_pair_representation"]
    en_clean = common.strict_clean_matrix(en_rows, features)
    zh_clean = common.strict_clean_matrix(zh_rows, features)
    en_latent = common.projected_pair_latents(en_rows, policy["pools"]["en_content_train_pool"], latent_cfg)
    zh_latent = common.projected_pair_latents(zh_rows, policy["pools"]["zh_target_strict"], latent_cfg)
    en_index = row_index(en_rows)
    zh_index = row_index(zh_rows)
    valid_clean_raw = select_rows(zh_clean, zh_index, zh_valid)
    test_clean_raw = select_rows(zh_clean, zh_index, zh_test)
    valid_latent = select_rows(zh_latent, zh_index, zh_valid)
    test_latent = select_rows(zh_latent, zh_index, zh_test)
    y_valid = common.labels_array(zh_valid)
    y_test = common.labels_array(zh_test)

    run_records = []
    schedule_weight_checks = []
    for ratio in ratios:
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"Invalid target support ratio: {ratio}")
        for seed in seeds:
            sampled_zh = common.stratified_support_sample(zh_train, ratio, seed)
            sampled_zh_strata = Counter(
                f"{row['review_label']}::{row['evidence_type']}" for row in sampled_zh
            )
            real_rows = en_train + sampled_zh
            real_clean_raw = select_rows(en_clean, en_index, en_train)
            real_latent = select_rows(en_latent, en_index, en_train)
            if sampled_zh:
                real_clean_raw = np.vstack(
                    [real_clean_raw, select_rows(zh_clean, zh_index, sampled_zh)]
                )
                real_latent = np.vstack(
                    [real_latent, select_rows(zh_latent, zh_index, sampled_zh)]
                )
            imputation = common.fit_train_median_imputation(real_clean_raw)
            real_clean = common.apply_imputation(real_clean_raw, imputation)
            valid_clean = common.apply_imputation(valid_clean_raw, imputation)
            test_clean = common.apply_imputation(test_clean_raw, imputation)
            real_weights, weight_diagnostics = common.factorized_evidence_weights(
                real_rows, policy["factorized_evidence_weighting"]
            )
            schedule, schedule_diagnostics = common.build_mixup_schedule(
                real_rows, real_latent, real_weights, cfg["mixup"], seed
            )
            if ratio >= 1.0 - 1e-12 and not schedule_diagnostics["schedule_budget_satisfied"]:
                raise ValueError(
                    "The preregistered 100% support mixup schedule cannot satisfy its "
                    "target-domain effective-weight budget"
                )
            augmented = {}
            manifests = {}
            for mode in experiments[1:]:
                syn_clean, syn_latent, syn_weight, manifest = common.augment_from_schedule(
                    real_clean, real_latent, real_rows, schedule, mode
                )
                augmented[mode] = (syn_clean, syn_latent, syn_weight)
                manifests[mode] = manifest
            duplicate_weight = float(np.sum(augmented[experiments[1]][2]))
            mixup_weight = float(np.sum(augmented[experiments[2]][2]))
            tolerance = float(cfg["duplication_control"]["required_total_synthetic_effective_weight_tolerance"])
            if abs(duplicate_weight - mixup_weight) > tolerance:
                raise ValueError("Duplication and latent mixup effective synthetic weights differ")
            schedule_weight_checks.append(
                {
                    "ratio": ratio,
                    "seed": seed,
                    "duplication_weight": duplicate_weight,
                    "mixup_weight": mixup_weight,
                    "absolute_difference": abs(duplicate_weight - mixup_weight),
                    "target_additional_positive_weight": schedule_diagnostics[
                        "target_additional_positive_weight"
                    ],
                    "schedule_budget_satisfied": schedule_diagnostics[
                        "schedule_budget_satisfied"
                    ],
                }
            )
            real_design_matrix = np.hstack([real_clean, real_latent])
            _, real_train_standardization = step9.fit_standardization(
                real_design_matrix,
                bool(cfg["logistic"]["standardize_features"]),
            )
            for experiment in experiments:
                x_real = real_design_matrix.copy()
                y_train = common.labels_array(real_rows)
                weights = real_weights.copy()
                if experiment != "no_augmentation":
                    syn_clean, syn_latent, syn_weight = augmented[experiment]
                    if len(syn_weight):
                        x_real = np.vstack([x_real, np.hstack([syn_clean, syn_latent])])
                        y_train = np.concatenate([y_train, np.ones(len(syn_weight), dtype=float)])
                        weights = np.concatenate([weights, syn_weight])
                logistic_artifact, _ = step9.fit_regularized_logistic(
                    x_real,
                    y_train,
                    cfg["logistic"],
                    sample_weight_multipliers=weights,
                    sample_weight_target_total=float(len(real_rows)),
                    precomputed_standardization=real_train_standardization,
                )
                x_valid = np.hstack([valid_clean, valid_latent])
                x_test = np.hstack([test_clean, test_latent])
                valid_prob = step9.apply_logistic_artifact_to_matrix(x_valid, logistic_artifact)
                test_prob = step9.apply_logistic_artifact_to_matrix(x_test, logistic_artifact)
                threshold = step7.choose_threshold(
                    y_valid, valid_prob, policy["threshold_selection"]["metric"], policy
                )
                valid_metrics = step7.evaluate_probabilities(y_valid, valid_prob, threshold)
                test_metrics = step7.evaluate_probabilities(y_test, test_prob, threshold)
                ratio_token = f"{int(round(ratio * 100)):03d}pct"
                run_key = f"{experiment}__ratio_{ratio_token}__seed_{seed}"
                prediction_fields = [
                    "pair_uid",
                    "split_name",
                    "review_label",
                    "evidence_type",
                    "v7_component_id",
                    "prob_positive",
                    "selected_threshold",
                    "predicted_label",
                ]
                valid_path = output_root / "predictions" / f"{run_key}.zh_valid.csv"
                test_path = output_root / "predictions" / f"{run_key}.internal_dev_test.csv"
                write_new(staged(valid_path), render_csv(prediction_rows(zh_valid, valid_prob, threshold, "valid"), prediction_fields))
                write_new(staged(test_path), render_csv(prediction_rows(zh_test, test_prob, threshold, "internal_development_test"), prediction_fields))
                manifest_path = None
                if experiment != "no_augmentation":
                    manifest_path = output_root / "manifests" / f"{run_key}.synthetic_train_only.csv"
                    manifest_fields = [
                        "synthetic_pair_uid",
                        "synthetic_train_only",
                        "mode",
                        "anchor_pair_uid",
                        "partner_pair_uid",
                        "domain",
                        "evidence_type",
                        "lambda_partner",
                        "augmentation_round",
                        "training_sample_weight",
                    ]
                    write_new(staged(manifest_path), render_csv(manifests[experiment], manifest_fields))
                artifact = {
                    "artifact_type": "step9_v7_latent_pair_logistic_l2",
                    "run_key": run_key,
                    "experiment": experiment,
                    "support_ratio": ratio,
                    "seed": seed,
                    "training_scope": {
                        "english_train_count": len(en_train),
                        "sampled_chinese_train_count": len(sampled_zh),
                        "sampled_chinese_pair_uids_sha256": common.canonical_hash(
                            sorted(row["pair_uid"] for row in sampled_zh)
                        ),
                        "sampled_chinese_label_evidence_stratum_counts": dict(
                            sorted(sampled_zh_strata.items())
                        ),
                        "support_sampling_policy": (
                            "deterministic_nested_label_x_evidence_type_stratified_"
                            "minimum_one_per_nonempty_stratum_for_positive_ratios"
                        ),
                        "real_train_count": len(real_rows),
                        "real_positive_count": int(np.sum(common.labels_array(real_rows))),
                        "real_negative_count": int(
                            len(real_rows) - np.sum(common.labels_array(real_rows))
                        ),
                        "real_train_pair_uids_sha256": common.canonical_hash(
                            sorted(row["pair_uid"] for row in real_rows)
                        ),
                    },
                    "feature_names": features + [f"e5_pair_latent_{index:03d}" for index in range(real_latent.shape[1])],
                    "clean_feature_imputation": imputation,
                    "latent_pair_representation": latent_cfg,
                    "factorized_weight_diagnostics": weight_diagnostics,
                    "mixup_schedule_diagnostics": schedule_diagnostics,
                    "logistic": logistic_artifact,
                    "selected_threshold_from_representative_valid": threshold,
                    "current_internal_test_used_for_selection": False,
                    "output_paths": {
                        "zh_valid": str(valid_path.relative_to(ROOT)).replace("\\", "/"),
                        "internal_development_test": str(test_path.relative_to(ROOT)).replace("\\", "/"),
                        "synthetic_train_only": None if manifest_path is None else str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
                    },
                }
                artifact_path = output_root / "artifacts" / f"{run_key}.json"
                write_new(staged(artifact_path), (json.dumps(artifact, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
                run_records.append(
                    {
                        "run_key": run_key,
                        "experiment": experiment,
                        "support_ratio": ratio,
                        "seed": seed,
                        "real_train_count": len(real_rows),
                        "english_train_count": len(en_train),
                        "sampled_chinese_train_count": len(sampled_zh),
                        "sampled_chinese_pair_uids_sha256": common.canonical_hash(
                            sorted(row["pair_uid"] for row in sampled_zh)
                        ),
                        "sampled_chinese_label_evidence_stratum_counts": dict(
                            sorted(sampled_zh_strata.items())
                        ),
                        "synthetic_train_count": 0 if experiment == "no_augmentation" else len(schedule),
                        "valid_metrics": valid_metrics,
                        "internal_development_test_metrics": test_metrics,
                        "mixup_schedule_diagnostics": schedule_diagnostics,
                        "artifact_path": str(artifact_path.relative_to(ROOT)).replace("\\", "/"),
                    }
                )

    selection = {}
    for ratio in ratios:
        candidates = {}
        for experiment in experiments:
            records = [
                row for row in run_records if row["experiment"] == experiment and abs(row["support_ratio"] - ratio) <= 1e-12
            ]
            candidates[experiment] = float(
                np.mean([row["valid_metrics"]["average_precision"] for row in records])
            )
        selected = max(experiments, key=lambda name: (candidates[name], -experiments.index(name)))
        selection[str(ratio)] = {
            "candidate_seed_mean_valid_average_precision": candidates,
            "selected_experiment": selected,
            "selection_split": "representative_zh_valid",
            "test_metrics_used_for_selection": False,
        }
    summary = {
        "step": "step9_v7_latent_pair_mixup",
        "version": policy["version"],
        "run_id": run_id,
        "run_count": len(run_records),
        "support_ratios": ratios,
        "seeds": seeds,
        "experiments": experiments,
        "selection": selection,
        "duplication_control_checks": schedule_weight_checks,
        "current_zh_test_role": "internal_development_test_only",
        "current_internal_test_used_for_selection": False,
        "runs": run_records,
        "policy": str(policy_path.relative_to(ROOT)).replace("\\", "/"),
        "policy_sha256": common.sha256(policy_path),
        "input_manifest": {
            "representative_validation_assignments": common.sha256(
                common.resolve(policy["representative_validation"]["split_assignment_output"])
            ),
            "v7_feature_manifest": common.sha256(
                common.resolve(policy["inductive_features"]["manifest_output"])
            ),
        },
    }
    summary["summary_sha256"] = common.canonical_hash(summary)
    summary_path = output_root / "step9_v7_latent_pair_mixup_summary.json"
    write_new(staged(summary_path), (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    staging_root.replace(output_root)
    print(json.dumps({"status": "pass", "summary": str(summary_path.relative_to(ROOT)), "run_count": len(run_records)}, indent=2))


if __name__ == "__main__":
    main()
