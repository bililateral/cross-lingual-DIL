#!/usr/bin/env python3
"""Perform the only permitted label-unsealed evaluation of the Step20 holdout."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7
import step15_v7_common as common


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step20_prospective_holdout_policy.json"
RANKING_METRICS = ("average_precision", "roc_auc", "pr_auc")


def render_csv(rows: list[dict], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite one-time prospective evaluation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def create_lock_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(descriptor, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    finally:
        os.close(descriptor)


def replace_lock(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.complete.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ranking_metric(name: str, y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if name == "average_precision":
        return step7.average_precision_score(y_true, scores)
    if name == "roc_auc":
        return step7.roc_auc_score(y_true, scores)
    if name == "pr_auc":
        return step7.precision_recall_auc_score(y_true, scores)
    raise ValueError(f"Unsupported prospective ranking metric: {name}")


def percentile_interval(values: list[float], confidence: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def model_bootstrap(
    y_true: np.ndarray,
    scores: np.ndarray,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict:
    rng = np.random.default_rng(seed)
    distributions = {metric: [] for metric in RANKING_METRICS}
    for _ in range(resamples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        y_sample = y_true[indices]
        if len(np.unique(y_sample)) < 2:
            continue
        for metric in RANKING_METRICS:
            value = ranking_metric(metric, y_sample, scores[indices])
            if value is not None:
                distributions[metric].append(float(value))
    result = {}
    for metric, values in distributions.items():
        low, high = percentile_interval(values, confidence)
        result[metric] = {
            "ci_low": low,
            "ci_high": high,
            "valid_resamples": len(values),
        }
    return result


def paired_comparison(
    comparison: dict,
    y_true: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    resamples: int,
    bootstrap_seed: int,
    permutations: int,
    permutation_seed: int,
    confidence: float,
) -> list[dict]:
    rng = np.random.default_rng(bootstrap_seed)
    distributions = {metric: [] for metric in RANKING_METRICS}
    for _ in range(resamples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        y_sample = y_true[indices]
        if len(np.unique(y_sample)) < 2:
            continue
        for metric in RANKING_METRICS:
            candidate_value = ranking_metric(metric, y_sample, candidate[indices])
            baseline_value = ranking_metric(metric, y_sample, baseline[indices])
            if candidate_value is not None and baseline_value is not None:
                distributions[metric].append(float(candidate_value - baseline_value))
    observed = {
        metric: (
            float(ranking_metric(metric, y_true, candidate)),
            float(ranking_metric(metric, y_true, baseline)),
        )
        for metric in RANKING_METRICS
    }
    extreme = {metric: 0 for metric in RANKING_METRICS}
    rng_permutation = np.random.default_rng(permutation_seed)
    for _ in range(permutations):
        swap = rng_permutation.integers(0, 2, size=len(y_true), dtype=np.int8).astype(bool)
        null_candidate = np.where(swap, baseline, candidate)
        null_baseline = np.where(swap, candidate, baseline)
        for metric in RANKING_METRICS:
            null_candidate_value = ranking_metric(metric, y_true, null_candidate)
            null_baseline_value = ranking_metric(metric, y_true, null_baseline)
            if null_candidate_value is None or null_baseline_value is None:
                raise ValueError("Prospective paired permutation produced an undefined metric")
            observed_difference = observed[metric][0] - observed[metric][1]
            if abs(float(null_candidate_value - null_baseline_value)) >= abs(observed_difference) - 1e-15:
                extreme[metric] += 1
    rows = []
    for metric in RANKING_METRICS:
        low, high = percentile_interval(distributions[metric], confidence)
        candidate_value, baseline_value = observed[metric]
        rows.append(
            {
                "comparison_id": comparison["comparison_id"],
                "candidate": comparison["candidate"],
                "baseline": comparison["baseline"],
                "metric": metric,
                "candidate_value": candidate_value,
                "baseline_value": baseline_value,
                "difference": candidate_value - baseline_value,
                "paired_bootstrap_ci_low": low,
                "paired_bootstrap_ci_high": high,
                "valid_bootstrap_resamples": len(distributions[metric]),
                "permutation_p_value": (1.0 + extreme[metric]) / (permutations + 1.0),
                "permutation_extreme_count": extreme[metric],
                "valid_permutations": permutations,
                "grouping_unit": "seller_disjoint_pair",
            }
        )
    return rows


def slice_metrics(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["evidence_type"]].append(index)
    y_true = np.asarray([1.0 if row["review_label"] == "positive" else 0.0 for row in rows])
    output = {}
    for evidence_type, indices in sorted(grouped.items()):
        selected = np.asarray(indices, dtype=int)
        metrics = step7.evaluate_probabilities(y_true[selected], scores[selected], threshold)
        metrics["mean_score"] = float(np.mean(scores[selected]))
        metrics["max_score"] = float(np.max(scores[selected]))
        output[evidence_type] = metrics
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path = common.resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    outputs = {key: common.resolve(value) for key, value in policy["outputs"].items()}
    if policy["evaluation"].get("selection_on_holdout_forbidden") is not True:
        raise ValueError("Prospective holdout policy permits model selection")
    if policy["evaluation"].get("thresholds_must_come_from_representative_validation") is not True:
        raise ValueError("Prospective holdout policy permits threshold fitting")
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "models": policy["evaluation"]["models"],
                    "comparisons": policy["evaluation"]["paired_comparisons"],
                    "one_time_lock": str(outputs["evaluation_lock"].relative_to(ROOT)),
                },
                indent=2,
            )
        )
        return
    required = (
        outputs["frozen_pair_universe"],
        outputs["frozen_labels"],
        outputs["freeze_manifest"],
        outputs["frozen_model_scores"],
        outputs["frozen_score_manifest"],
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing one-time prospective evaluation input: {path}")
    if outputs["metrics"].exists() or outputs["predictions"].exists():
        raise FileExistsError("Prospective evaluation outputs already exist")
    if outputs["evaluation_lock"].exists():
        raise FileExistsError("Prospective holdout evaluation has already been attempted")
    score_manifest = json.loads(outputs["frozen_score_manifest"].read_text(encoding="utf-8"))
    if score_manifest["frozen_scores_file_sha256"] != common.sha256(outputs["frozen_model_scores"]):
        raise ValueError("Frozen prospective scores changed before label unsealing")
    if score_manifest["frozen_pair_universe_sha256"] != common.sha256(
        outputs["frozen_pair_universe"]
    ):
        raise ValueError("Frozen scores and prospective pair universe disagree")

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    in_progress_lock = {
        "status": "evaluation_in_progress_do_not_rerun",
        "started_at_utc": started,
        "policy_sha256": common.sha256(policy_path),
        "frozen_pair_universe_sha256": common.sha256(outputs["frozen_pair_universe"]),
        "frozen_labels_sha256": common.sha256(outputs["frozen_labels"]),
        "frozen_scores_sha256": common.sha256(outputs["frozen_model_scores"]),
        "score_manifest_sha256": common.sha256(outputs["frozen_score_manifest"]),
    }
    create_lock_once(outputs["evaluation_lock"], in_progress_lock)

    universe_rows = common.load_csv(outputs["frozen_pair_universe"])
    label_rows = common.load_csv(outputs["frozen_labels"])
    score_rows = common.load_csv(outputs["frozen_model_scores"])
    universe_index = {row["pair_uid"]: row for row in universe_rows}
    label_index = {row["pair_uid"]: row for row in label_rows}
    score_index = {row["pair_uid"]: row for row in score_rows}
    if (
        len(universe_index) != len(universe_rows)
        or len(label_index) != len(label_rows)
        or len(score_index) != len(score_rows)
        or set(universe_index) != set(label_index)
        or set(universe_index) != set(score_index)
    ):
        raise ValueError(
            "Prospective pair universe, labels, and frozen scores are not one-to-one"
        )
    for pair_uid, identity in universe_index.items():
        label = label_index[pair_uid]
        if (
            identity["seller_uid_left"] != label["seller_uid_left"]
            or identity["seller_uid_right"] != label["seller_uid_right"]
        ):
            raise ValueError(f"Prospective pair endpoints changed before evaluation: {pair_uid}")
    pair_order = sorted(label_index)
    rows = [label_index[pair_uid] for pair_uid in pair_order]
    y_true = np.asarray([1.0 if row["review_label"] == "positive" else 0.0 for row in rows])
    if len(np.unique(y_true)) != 2:
        raise ValueError("Prospective final holdout lacks both binary classes")
    models = list(policy["evaluation"]["models"])
    model_scores = {}
    thresholds = {}
    for model_id in models:
        model_scores[model_id] = np.asarray(
            [float(score_index[pair_uid][f"{model_id}__score"]) for pair_uid in pair_order]
        )
        observed_thresholds = {
            float(score_index[pair_uid][f"{model_id}__threshold"]) for pair_uid in pair_order
        }
        if len(observed_thresholds) != 1:
            raise ValueError(f"Prospective model threshold is not frozen/constant: {model_id}")
        thresholds[model_id] = next(iter(observed_thresholds))

    resamples = int(policy["evaluation"]["bootstrap_resamples"])
    confidence = float(policy["evaluation"]["confidence_level"])
    model_metrics = {}
    for offset, model_id in enumerate(models):
        point = step7.evaluate_probabilities(y_true, model_scores[model_id], thresholds[model_id])
        model_metrics[model_id] = {
            "point": point,
            "paired_seller_disjoint_bootstrap": model_bootstrap(
                y_true,
                model_scores[model_id],
                resamples,
                int(policy["evaluation"]["bootstrap_seed"]) + offset * 104729,
                confidence,
            ),
            "evidence_slices": slice_metrics(rows, model_scores[model_id], thresholds[model_id]),
        }
    comparisons = []
    for offset, comparison in enumerate(policy["evaluation"]["paired_comparisons"]):
        comparisons.extend(
            paired_comparison(
                comparison,
                y_true,
                model_scores[comparison["candidate"]],
                model_scores[comparison["baseline"]],
                resamples,
                int(policy["evaluation"]["bootstrap_seed"]) + (offset + 20) * 104729,
                int(policy["evaluation"]["permutation_count"]),
                int(policy["evaluation"]["permutation_seed"]) + offset * 130363,
                confidence,
            )
        )

    prediction_rows = []
    for index, pair_uid in enumerate(pair_order):
        label = label_index[pair_uid]
        score = score_index[pair_uid]
        output = {
            "pair_uid": pair_uid,
            "review_label": label["review_label"],
            "evidence_type": label["evidence_type"],
            "prospective_candidate_category": label["prospective_candidate_category"],
            "reliability_decision": score["reliability_decision"],
        }
        for model_id in models:
            output[f"{model_id}__score"] = f"{float(model_scores[model_id][index]):.12f}"
            output[f"{model_id}__threshold"] = f"{thresholds[model_id]:.12f}"
            output[f"{model_id}__predicted_label"] = int(
                model_scores[model_id][index] >= thresholds[model_id]
            )
        prediction_rows.append(output)

    summary = {
        "step": "step20_one_time_prospective_holdout_evaluation",
        "version": policy["version"],
        "evaluation_started_at_utc": started,
        "evaluation_completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "one_time_evaluation": True,
        "selection_or_threshold_tuning_on_holdout": False,
        "row_count": len(rows),
        "positive_count": int(np.sum(y_true)),
        "negative_count": int(len(y_true) - np.sum(y_true)),
        "seller_disjoint_pair_bootstrap": True,
        "model_metrics": model_metrics,
        "paired_comparisons": comparisons,
        "map": None,
        "mrr": None,
        "map_mrr_status": "not_applicable_without_preregistered_query_groups",
        "inputs": {
            "frozen_pair_universe_sha256": common.sha256(outputs["frozen_pair_universe"]),
            "frozen_labels_sha256": common.sha256(outputs["frozen_labels"]),
            "freeze_manifest_sha256": common.sha256(outputs["freeze_manifest"]),
            "frozen_scores_sha256": common.sha256(outputs["frozen_model_scores"]),
            "score_manifest_sha256": common.sha256(outputs["frozen_score_manifest"]),
        },
        "policy_sha256": common.sha256(policy_path),
        "producer_sha256": common.sha256(Path(__file__).resolve()),
    }
    summary["summary_sha256"] = common.canonical_hash(summary)
    prediction_payload = render_csv(prediction_rows, list(prediction_rows[0]))
    summary_payload = (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    write_new(outputs["predictions"], prediction_payload)
    write_new(outputs["metrics"], summary_payload)
    completed_lock = {
        **in_progress_lock,
        "status": "evaluation_complete_never_rerun",
        "completed_at_utc": summary["evaluation_completed_at_utc"],
        "predictions_sha256": common.sha256(outputs["predictions"]),
        "metrics_sha256": common.sha256(outputs["metrics"]),
        "summary_sha256": summary["summary_sha256"],
    }
    replace_lock(outputs["evaluation_lock"], completed_lock)
    print(
        json.dumps(
            {
                "status": "evaluation_complete_never_rerun",
                "row_count": len(rows),
                "positive_count": int(np.sum(y_true)),
                "negative_count": int(len(y_true) - np.sum(y_true)),
                "metrics": str(outputs["metrics"].relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
