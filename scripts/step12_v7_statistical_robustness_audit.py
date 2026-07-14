#!/usr/bin/env python3
"""Audit Step9/15-v7 without selecting on the legacy internal-development test."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7
import step15_v7_common as common


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step12_v7_statistical_robustness_policy.json"
SPLIT_FILE_TOKEN = {
    "representative_validation": "zh_valid",
    "internal_development_test": "internal_dev_test",
}
SPLIT_ROW_TOKEN = {
    "representative_validation": "valid",
    "internal_development_test": "internal_development_test",
}


def render_csv(rows: list[dict], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Step12-v7 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def stable_seed(base: int, *tokens: str) -> int:
    digest = hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()
    return (int(base) + int(digest[:8], 16)) % (2**32 - 1)


def ranking_metric(metric: str, y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if metric == "average_precision":
        return step7.average_precision_score(y_true, scores)
    if metric == "roc_auc":
        return step7.roc_auc_score(y_true, scores)
    if metric == "pr_auc":
        return step7.precision_recall_auc_score(y_true, scores)
    raise ValueError(f"Unsupported Step12-v7 ranking metric: {metric}")


def grouped_indices(component_ids: list[str]) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, component_id in enumerate(component_ids):
        if not component_id:
            raise ValueError("Step12-v7 grouping requires a non-empty v7_component_id")
        grouped[component_id].append(index)
    groups = [np.asarray(indices, dtype=int) for _, indices in sorted(grouped.items())]
    covered = np.concatenate(groups) if groups else np.empty(0, dtype=int)
    if sorted(covered.tolist()) != list(range(len(component_ids))):
        raise ValueError("Step12-v7 component groups do not partition the evaluation rows")
    return groups


def sampled_group_indices(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    chosen = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[int(index)] for index in chosen])


def comparison_worker(payload: dict) -> dict:
    y_true = np.asarray(payload["y_true"], dtype=float)
    candidate_scores = np.asarray(payload["candidate_scores"], dtype=float)
    baseline_scores = np.asarray(payload["baseline_scores"], dtype=float)
    candidate_seed_scores = np.asarray(payload["candidate_seed_scores"], dtype=float)
    baseline_seed_scores = np.asarray(payload["baseline_seed_scores"], dtype=float)
    groups = [np.asarray(group, dtype=int) for group in payload["groups"]]
    metric = payload["metric"]
    candidate_point = ranking_metric(metric, y_true, candidate_scores)
    baseline_point = ranking_metric(metric, y_true, baseline_scores)
    if candidate_point is None or baseline_point is None:
        raise ValueError("Step12-v7 comparison metric is undefined on the full split")
    observed_difference = float(candidate_point - baseline_point)

    rng = np.random.default_rng(int(payload["bootstrap_seed"]))
    differences = []
    two_level_differences = []
    seed_ids_match = payload["candidate_seed_ids"] == payload["baseline_seed_ids"]
    for _ in range(int(payload["resamples"])):
        indices = sampled_group_indices(groups, rng)
        y_sample = y_true[indices]
        if len(np.unique(y_sample)) < 2:
            continue
        candidate_value = ranking_metric(metric, y_sample, candidate_scores[indices])
        baseline_value = ranking_metric(metric, y_sample, baseline_scores[indices])
        if candidate_value is not None and baseline_value is not None:
            differences.append(float(candidate_value - baseline_value))
        if seed_ids_match and candidate_seed_scores.shape[0] > 1:
            seed_sample = rng.integers(
                0, candidate_seed_scores.shape[0], size=candidate_seed_scores.shape[0]
            )
            candidate_two_level = np.mean(candidate_seed_scores[seed_sample], axis=0)
            baseline_two_level = np.mean(baseline_seed_scores[seed_sample], axis=0)
            candidate_value = ranking_metric(metric, y_sample, candidate_two_level[indices])
            baseline_value = ranking_metric(metric, y_sample, baseline_two_level[indices])
            if candidate_value is not None and baseline_value is not None:
                two_level_differences.append(float(candidate_value - baseline_value))

    confidence = float(payload["confidence"])
    alpha = (1.0 - confidence) / 2.0

    def interval(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        return (
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1.0 - alpha)),
        )

    ci_low, ci_high = interval(differences)
    two_level_low, two_level_high = interval(two_level_differences)
    rng_permutation = np.random.default_rng(int(payload["permutation_seed"]))
    extreme = 0
    for _ in range(int(payload["permutations"])):
        swapped_candidate = candidate_scores.copy()
        swapped_baseline = baseline_scores.copy()
        for group, swap in zip(
            groups,
            rng_permutation.integers(0, 2, size=len(groups), dtype=np.int8),
            strict=True,
        ):
            if bool(swap):
                swapped_candidate[group] = baseline_scores[group]
                swapped_baseline[group] = candidate_scores[group]
        null_candidate = ranking_metric(metric, y_true, swapped_candidate)
        null_baseline = ranking_metric(metric, y_true, swapped_baseline)
        if null_candidate is None or null_baseline is None:
            raise ValueError("Step12-v7 paired permutation produced an undefined metric")
        if abs(float(null_candidate - null_baseline)) >= abs(observed_difference) - 1e-15:
            extreme += 1
    permutations = int(payload["permutations"])
    return {
        "comparison_id": payload["comparison_id"],
        "evaluation_split": payload["evaluation_split"],
        "scope": "all_rows",
        "metric": metric,
        "candidate": payload["candidate"],
        "baseline": payload["baseline"],
        "candidate_value": candidate_point,
        "baseline_value": baseline_point,
        "difference": observed_difference,
        "grouped_bootstrap_ci_low": ci_low,
        "grouped_bootstrap_ci_high": ci_high,
        "valid_grouped_bootstrap_resamples": len(differences),
        "two_level_seed_component_ci_low": two_level_low,
        "two_level_seed_component_ci_high": two_level_high,
        "valid_two_level_resamples": len(two_level_differences),
        "paired_seed_ids": seed_ids_match,
        "permutation_p_value": (1.0 + extreme) / (permutations + 1.0),
        "permutation_extreme_count": extreme,
        "valid_permutations": permutations,
    }


def canonical_rows(policy: dict) -> dict[str, list[dict]]:
    pools = common.load_joined_rows(policy)
    zh_rows = pools["zh_target_strict"]
    result = {}
    for evaluation_split, row_split in SPLIT_ROW_TOKEN.items():
        rows = sorted(
            [row for row in zh_rows if row["v7_split_name"] == row_split],
            key=lambda row: row["pair_uid"],
        )
        if not rows:
            raise ValueError(f"Step12-v7 split is empty: {evaluation_split}")
        result[evaluation_split] = rows
    return result


def load_score_matrix(
    paths: list[Path],
    rows: list[dict],
    score_column: str,
) -> tuple[np.ndarray, list[float]]:
    pair_order = [row["pair_uid"] for row in rows]
    expected = {row["pair_uid"]: row for row in rows}
    matrices = []
    thresholds = []
    for path in paths:
        prediction_rows = common.load_csv(path)
        index = {row["pair_uid"]: row for row in prediction_rows}
        if len(index) != len(prediction_rows) or set(index) != set(expected):
            raise ValueError(f"Step12-v7 prediction universe mismatch: {path}")
        for pair_uid in pair_order:
            prediction = index[pair_uid]
            canonical = expected[pair_uid]
            if prediction.get("review_label") != canonical["review_label"]:
                raise ValueError(f"Step12-v7 prediction label mismatch: {path}:{pair_uid}")
            if prediction.get("evidence_type") != canonical["evidence_type"]:
                raise ValueError(f"Step12-v7 evidence slice mismatch: {path}:{pair_uid}")
            if prediction.get("v7_component_id") != canonical["v7_component_id"]:
                raise ValueError(f"Step12-v7 component mismatch: {path}:{pair_uid}")
        matrices.append(np.asarray([float(index[pair_uid][score_column]) for pair_uid in pair_order]))
        unique_thresholds = {float(row["selected_threshold"]) for row in prediction_rows}
        if len(unique_thresholds) != 1:
            raise ValueError(f"Step12-v7 prediction file has non-constant threshold: {path}")
        thresholds.append(next(iter(unique_thresholds)))
    return np.vstack(matrices), thresholds


def evidence_slice_metrics(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["evidence_type"]].append(index)
    result = {}
    y_true = common.labels_array(rows)
    for evidence_type, indices in sorted(grouped.items()):
        selected = np.asarray(indices, dtype=int)
        metrics = step7.evaluate_probabilities(y_true[selected], scores[selected], threshold)
        metrics["mean_score"] = float(np.mean(scores[selected]))
        metrics["max_score"] = float(np.max(scores[selected]))
        result[evidence_type] = metrics
    return result


def diagnostic_veto_slices(models: dict, rows_by_split: dict) -> list[dict]:
    output = []
    candidate = models["step15_v7_two_stage_veto"]
    baseline = models["step15_v7_clean_selected"]
    direct_types = {
        "same_controller_direct_identifier",
        "same_controller_component_anchor",
    }
    for split, rows in rows_by_split.items():
        y_true = common.labels_array(rows)
        evidence = [row["evidence_type"] for row in rows]
        for scope, mask, metric in (
            (
                "public_contact_or_url_noise",
                np.asarray([value == "public_contact_or_url_noise" for value in evidence]),
                "false_positive_rate",
            ),
            (
                "same_controller_direct_or_component",
                np.asarray([value in direct_types for value in evidence]),
                "recall",
            ),
        ):
            if not np.any(mask):
                output.append(
                    {
                        "evaluation_split": split,
                        "scope": scope,
                        "metric": metric,
                        "row_count": 0,
                        "status": "slice_absent",
                    }
                )
                continue
            candidate_scores = candidate["splits"][split]["scores"][mask]
            baseline_scores = baseline["splits"][split]["scores"][mask]
            labels = y_true[mask]
            candidate_pred = candidate_scores >= candidate["threshold"]
            baseline_pred = baseline_scores >= baseline["threshold"]
            if metric == "false_positive_rate":
                candidate_value = float(np.mean(candidate_pred))
                baseline_value = float(np.mean(baseline_pred))
            else:
                candidate_value = float(np.mean(candidate_pred[labels == 1.0]))
                baseline_value = float(np.mean(baseline_pred[labels == 1.0]))
            output.append(
                {
                    "evaluation_split": split,
                    "scope": scope,
                    "metric": metric,
                    "row_count": int(np.sum(mask)),
                    "candidate_value": candidate_value,
                    "baseline_value": baseline_value,
                    "difference": candidate_value - baseline_value,
                    "candidate_mean_score": float(np.mean(candidate_scores)),
                    "baseline_mean_score": float(np.mean(baseline_scores)),
                    "status": "diagnostic_internal_not_publication_promotion",
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--resamples", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=None)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path = common.resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    v7_policy_path = common.resolve(policy["v7_policy"])
    v7_policy = json.loads(v7_policy_path.read_text(encoding="utf-8"))
    if policy["selection"].get("test_metrics_used_for_selection") is not False:
        raise ValueError("Step12-v7 policy permits test-informed selection")
    if policy["publication_promotion"].get("eligible") is not False:
        raise ValueError("Internal-development results cannot publication-promote v7")
    if v7_policy["step9_latent_mixup"]["logistic"].get("class_weight") != "none":
        raise ValueError("V7 controls require uniform class weights for interpretable mixup isolation")
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "workers": args.workers,
                    "models": policy["models"],
                    "comparisons": [row["comparison_id"] for row in policy["paired_comparisons"]],
                },
                indent=2,
            )
        )
        return

    output_root = common.resolve(policy["outputs_root"])
    summary_path = common.resolve(policy["summary_output"])
    metrics_path = common.resolve(policy["model_metrics_output"])
    comparisons_path = common.resolve(policy["comparisons_output"])
    model_freeze_path = common.resolve(policy["model_freeze_manifest_output"])
    managed_paths = [summary_path, metrics_path, comparisons_path, model_freeze_path]
    if any(path.parent != output_root for path in managed_paths):
        raise ValueError("All Step12-v7 outputs, including the model freeze, must share outputs_root")
    staging_root = output_root.with_name(f".{output_root.name}.incomplete")
    if output_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Step12-v7 final or incomplete output already exists: {output_root} / {staging_root}"
        )

    def staged(final_path: Path) -> Path:
        return staging_root / final_path.relative_to(output_root)
    step20_policy = json.loads(
        common.resolve(policy["prospective_holdout_policy"]).read_text(encoding="utf-8")
    )
    prospective_labels = common.resolve(step20_policy["outputs"]["frozen_labels"])
    if prospective_labels.exists():
        raise ValueError("Cannot freeze v7 models after prospective labels have been unsealed")

    rows_by_split = canonical_rows(v7_policy)
    step9_root = (
        common.resolve(v7_policy["step9_latent_mixup"]["outputs_root"])
        / policy["step9_run_id"]
    )
    step9_summary_path = step9_root / "step9_v7_latent_pair_mixup_summary.json"
    step9_summary = json.loads(step9_summary_path.read_text(encoding="utf-8"))
    seeds = [int(value) for value in v7_policy["step9_latent_mixup"]["seeds"]]
    experiments = list(v7_policy["step9_latent_mixup"]["experiments"])
    ratio_token = f"{int(round(float(policy['support_ratio']) * 100)):03d}pct"
    models: dict[str, dict] = {}
    for experiment in experiments:
        model = {"model_id": experiment, "seed_ids": seeds, "splits": {}}
        for split, file_token in SPLIT_FILE_TOKEN.items():
            paths = [
                step9_root
                / "predictions"
                / f"{experiment}__ratio_{ratio_token}__seed_{seed}.{file_token}.csv"
                for seed in seeds
            ]
            matrix, per_seed_thresholds = load_score_matrix(
                paths, rows_by_split[split], "prob_positive"
            )
            model["splits"][split] = {
                "seed_scores": matrix,
                "scores": np.mean(matrix, axis=0),
                "source_paths": paths,
                "per_seed_thresholds": per_seed_thresholds,
            }
        models[experiment] = model

    source_only = {
        "model_id": "step9_v7_source_only_clean_fusion",
        "seed_ids": seeds,
        "splits": {},
        "training_scope": "english_labels_only_zero_percent_chinese_support",
    }
    for split, file_token in SPLIT_FILE_TOKEN.items():
        paths = [
            step9_root
            / "predictions"
            / f"no_augmentation__ratio_000pct__seed_{seed}.{file_token}.csv"
            for seed in seeds
        ]
        matrix, per_seed_thresholds = load_score_matrix(
            paths, rows_by_split[split], "prob_positive"
        )
        if not np.allclose(matrix, matrix[0][None, :], rtol=0.0, atol=1e-12):
            raise ValueError(
                "The deterministic source-only clean fusion unexpectedly varies by seed"
            )
        source_only["splits"][split] = {
            "seed_scores": matrix,
            "scores": np.mean(matrix, axis=0),
            "source_paths": paths,
            "per_seed_thresholds": per_seed_thresholds,
        }
    models["step9_v7_source_only_clean_fusion"] = source_only

    selection_key = str(float(policy["support_ratio"]))
    step9_selection = step9_summary["selection"].get(selection_key)
    if step9_selection is None or step9_selection.get("test_metrics_used_for_selection") is not False:
        raise ValueError("Step9-v7 selection is absent or test-informed")
    valid_rows = rows_by_split["representative_validation"]
    y_valid = common.labels_array(valid_rows)
    mean_per_seed_valid_ap = {
        experiment: float(
            np.mean(
                [
                    step7.average_precision_score(y_valid, seed_scores)
                    for seed_scores in models[experiment]["splits"]["representative_validation"]["seed_scores"]
                ]
            )
        )
        for experiment in experiments
    }
    tie_order = policy["selection"]["simplicity_tie_break_order"]
    selected_experiment = max(
        experiments,
        key=lambda name: (mean_per_seed_valid_ap[name], -tie_order.index(name)),
    )
    if selected_experiment != step9_selection["selected_experiment"]:
        raise ValueError(
            "Step12-v7 recomputed validation selection disagrees with Step9-v7: "
            f"{selected_experiment} vs {step9_selection['selected_experiment']}"
        )
    models["step15_v7_clean_selected"] = {
        **models[selected_experiment],
        "model_id": "step15_v7_clean_selected",
        "alias_of": selected_experiment,
    }

    step15_root = (
        common.resolve(v7_policy["outputs"]["two_stage_outputs_root"])
        / policy["step15_run_id"]
    )
    step15_summary_path = step15_root / "step15_v7_two_stage_summary.json"
    step15_summary = json.loads(step15_summary_path.read_text(encoding="utf-8"))
    if step15_summary["selected_clean_experiment"] != selected_experiment:
        raise ValueError("Step15-v7 and Step12-v7 clean selections disagree")
    two_stage = {"model_id": "step15_v7_two_stage_veto", "seed_ids": seeds, "splits": {}}
    for split in SPLIT_FILE_TOKEN:
        paths = [
            step15_root / "predictions" / f"two_stage_seed_{seed}.{SPLIT_ROW_TOKEN[split]}.csv"
            for seed in seeds
        ]
        matrix, per_seed_thresholds = load_score_matrix(
            paths, rows_by_split[split], "reliability_veto_prob_positive"
        )
        two_stage["splits"][split] = {
            "seed_scores": matrix,
            "scores": np.mean(matrix, axis=0),
            "source_paths": paths,
            "per_seed_thresholds": per_seed_thresholds,
        }
    models["step15_v7_two_stage_veto"] = two_stage

    raw_e5 = {"model_id": "raw_clean_e5_cosine", "seed_ids": [0], "splits": {}}
    for split, rows in rows_by_split.items():
        scores = np.asarray(
            [
                float(row["embedding_cosine_multilingual_e5_large_identifier_redacted"])
                for row in rows
            ]
        )
        raw_e5["splits"][split] = {"seed_scores": scores[None, :], "scores": scores}
    models["raw_clean_e5_cosine"] = raw_e5

    for model_id, model in models.items():
        valid_scores = model["splits"]["representative_validation"]["scores"]
        model["threshold"] = step7.choose_threshold(
            y_valid,
            valid_scores,
            v7_policy["threshold_selection"]["metric"],
            v7_policy,
        )
        for split, rows in rows_by_split.items():
            y_true = common.labels_array(rows)
            split_model = model["splits"][split]
            split_model["metrics"] = step7.evaluate_probabilities(
                y_true, split_model["scores"], model["threshold"]
            )
            split_model["evidence_slices"] = evidence_slice_metrics(
                rows, split_model["scores"], model["threshold"]
            )

    duplication_checks = [
        row
        for row in step9_summary["duplication_control_checks"]
        if abs(float(row["ratio"]) - float(policy["support_ratio"])) <= 1e-12
    ]
    if len(duplication_checks) != len(seeds) or any(
        float(row["absolute_difference"]) > 1e-10 for row in duplication_checks
    ):
        raise ValueError("Step9-v7 mixup and duplication effective weights are not exactly matched")
    if any(row.get("schedule_budget_satisfied") is not True for row in duplication_checks):
        raise ValueError("Step9-v7 100% support schedules did not satisfy their fixed weight budget")

    ranking_metrics = [policy["metrics"]["primary"], *policy["metrics"]["secondary"]]
    tasks = []
    bootstrap_resamples = int(args.resamples or policy["bootstrap"]["num_resamples"])
    permutations = int(args.permutations or policy["randomization_test"]["num_permutations"])
    for comparison in policy["paired_comparisons"]:
        candidate = models[comparison["candidate"]]
        baseline = models[comparison["baseline"]]
        for split, rows in rows_by_split.items():
            component_ids = [row["v7_component_id"] for row in rows]
            groups = grouped_indices(component_ids)
            y_true = common.labels_array(rows)
            for metric in ranking_metrics:
                token = (comparison["comparison_id"], split, metric)
                tasks.append(
                    {
                        "comparison_id": comparison["comparison_id"],
                        "candidate": comparison["candidate"],
                        "baseline": comparison["baseline"],
                        "evaluation_split": split,
                        "metric": metric,
                        "y_true": y_true.tolist(),
                        "candidate_scores": candidate["splits"][split]["scores"].tolist(),
                        "baseline_scores": baseline["splits"][split]["scores"].tolist(),
                        "candidate_seed_scores": candidate["splits"][split]["seed_scores"].tolist(),
                        "baseline_seed_scores": baseline["splits"][split]["seed_scores"].tolist(),
                        "candidate_seed_ids": candidate["seed_ids"],
                        "baseline_seed_ids": baseline["seed_ids"],
                        "groups": [group.tolist() for group in groups],
                        "resamples": bootstrap_resamples,
                        "confidence": policy["bootstrap"]["confidence_level"],
                        "bootstrap_seed": stable_seed(policy["bootstrap"]["random_seed"], *token),
                        "permutations": permutations,
                        "permutation_seed": stable_seed(
                            policy["randomization_test"]["random_seed"], *token
                        ),
                    }
                )
    workers = max(1, min(int(args.workers), len(tasks)))
    if workers == 1:
        comparison_rows = [comparison_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            comparison_rows = list(executor.map(comparison_worker, tasks))
    comparison_rows.sort(
        key=lambda row: (row["comparison_id"], row["evaluation_split"], row["metric"])
    )

    model_metric_rows = []
    for model_id in policy["models"]:
        model = models[model_id]
        for split in policy["evaluation_splits"]:
            metrics = model["splits"][split]["metrics"]
            model_metric_rows.append(
                {
                    "model_id": model_id,
                    "evaluation_split": split,
                    "row_count": metrics["row_count"],
                    "positive_count": metrics["positive_count"],
                    "negative_count": metrics["negative_count"],
                    "threshold_from_representative_validation": model["threshold"],
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "pr_auc": metrics["pr_auc"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "specificity": metrics["specificity"],
                }
            )

    veto_slice_diagnostics = diagnostic_veto_slices(models, rows_by_split)
    artifacts = {}
    for experiment in experiments:
        matching_runs = [
            row
            for row in step9_summary["runs"]
            if row["experiment"] == experiment
            and abs(float(row["support_ratio"]) - float(policy["support_ratio"])) <= 1e-12
        ]
        paths = [common.resolve(row["artifact_path"]) for row in matching_runs]
        if len(paths) != len(seeds):
            raise ValueError(f"Step12-v7 expected one artifact per seed for {experiment}")
        artifacts[experiment] = [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": common.sha256(path),
            }
            for path in sorted(paths)
        ]
    source_only_runs = [
        row
        for row in step9_summary["runs"]
        if row["experiment"] == "no_augmentation"
        and abs(float(row["support_ratio"])) <= 1e-12
    ]
    source_only_paths = [common.resolve(row["artifact_path"]) for row in source_only_runs]
    if len(source_only_paths) != len(seeds):
        raise ValueError("Step12-v7 expected one 0% support source-only artifact per seed")
    if any(int(row.get("sampled_chinese_train_count", -1)) != 0 for row in source_only_runs):
        raise ValueError("The v7 source-only control contains Chinese training support")
    source_only_artifacts = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": common.sha256(path),
        }
        for path in sorted(source_only_paths)
    ]

    joined_for_reference = common.load_joined_rows(v7_policy)["zh_target_strict"]
    identifier_frequency_reference_sellers = sorted(
        {
            row[key]
            for row in joined_for_reference
            if row["v7_split_name"] == "train"
            for key in ("seller_uid_left", "seller_uid_right")
        }
    )
    if common.canonical_hash(identifier_frequency_reference_sellers) != step15_summary.get(
        "identifier_frequency_reference_sellers_sha256"
    ):
        raise ValueError("Step15-v7 identifier-frequency train reference cannot be reproduced")

    model_freeze = {
        "step": "step15_v7_model_and_threshold_freeze",
        "version": policy["version"],
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "step9_run_id": policy["step9_run_id"],
        "step15_run_id": policy["step15_run_id"],
        "support_ratio": float(policy["support_ratio"]),
        "seed_ids": seeds,
        "selection_split": "representative_validation",
        "selection_metric": "mean_of_per_seed_valid_average_precision",
        "selection_candidates": mean_per_seed_valid_ap,
        "selected_clean_experiment": selected_experiment,
        "thresholds_from_representative_validation": {
            model_id: models[model_id]["threshold"] for model_id in policy["models"]
        },
        "step9_artifacts": artifacts,
        "step9_source_only_artifacts": source_only_artifacts,
        "step15_stage_b_policy": v7_policy["two_stage_method"]["stage_b"],
        "identifier_frequency_reference_scope": "v7_zh_train_sellers_only",
        "identifier_frequency_reference_sellers": identifier_frequency_reference_sellers,
        "identifier_frequency_reference_sellers_sha256": common.canonical_hash(
            identifier_frequency_reference_sellers
        ),
        "class_weight": v7_policy["step9_latent_mixup"]["logistic"]["class_weight"],
        "duplication_and_mixup_effective_weights_matched": True,
        "current_internal_test_used_for_model_selection": False,
        "prospective_holdout_required": True,
        "inputs": {
            "v7_policy": common.sha256(v7_policy_path),
            "step12_policy": common.sha256(policy_path),
            "step9_summary": common.sha256(step9_summary_path),
            "step15_summary": common.sha256(step15_summary_path),
            "v7_feature_manifest": common.sha256(
                common.resolve(v7_policy["inductive_features"]["manifest_output"])
            ),
            "representative_validation_manifest": common.sha256(
                common.resolve(v7_policy["representative_validation"]["manifest_output"])
            ),
        },
        "producer_sha256": common.sha256(Path(__file__).resolve()),
    }
    model_freeze["manifest_sha256"] = common.canonical_hash(model_freeze)

    summary = {
        "step": "step12_v7_statistical_robustness_audit",
        "version": policy["version"],
        "workers": workers,
        "bootstrap_resamples": bootstrap_resamples,
        "permutations": permutations,
        "selection": {
            "candidate_mean_per_seed_valid_average_precision": mean_per_seed_valid_ap,
            "selected_clean_experiment": selected_experiment,
            "selection_split": "representative_validation",
            "current_internal_test_used_for_selection": False,
        },
        "model_metrics": {
            model_id: {
                split: models[model_id]["splits"][split]["metrics"]
                for split in policy["evaluation_splits"]
            }
            for model_id in policy["models"]
        },
        "evidence_slice_metrics": {
            model_id: {
                split: models[model_id]["splits"][split]["evidence_slices"]
                for split in policy["evaluation_splits"]
            }
            for model_id in policy["models"]
        },
        "veto_slice_diagnostics": veto_slice_diagnostics,
        "paired_comparisons": comparison_rows,
        "publication_promotion": policy["publication_promotion"],
        "current_zh_test_role": "internal_development_test_only",
        "prospective_holdout_required": True,
        "model_freeze_manifest": str(model_freeze_path.relative_to(ROOT)).replace("\\", "/"),
        "policy": str(policy_path.relative_to(ROOT)).replace("\\", "/"),
        "policy_sha256": common.sha256(policy_path),
    }
    summary["summary_sha256"] = common.canonical_hash(summary)
    model_fields = list(model_metric_rows[0])
    comparison_fields = list(comparison_rows[0])
    write_new(staged(metrics_path), render_csv(model_metric_rows, model_fields))
    write_new(staged(comparisons_path), render_csv(comparison_rows, comparison_fields))
    write_new(
        staged(summary_path),
        (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    write_new(
        staged(model_freeze_path),
        (json.dumps(model_freeze, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    staging_root.replace(output_root)
    print(
        json.dumps(
            {
                "status": "pass",
                "selected_clean_experiment": selected_experiment,
                "publication_promotion_eligible": False,
                "prospective_holdout_required": True,
                "summary": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
                "model_freeze_manifest": str(model_freeze_path.relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
