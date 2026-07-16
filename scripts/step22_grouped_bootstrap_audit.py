#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "reports" / "step22_same_seller_split" / "v1_20260716"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives <= 0:
        raise ValueError("Average precision requires at least one positive")
    order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
    hits = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("Invalid percentile input")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "pair_uid",
        "component_id",
        "review_label",
        "prob_no_augmentation",
        "prob_equal_weight_duplication_positive_budget",
        "prob_same_seller_split_positive_only",
        "prob_equal_weight_duplication_full_budget",
        "prob_same_seller_split_plus_reviewed_negative_views",
    }
    missing = sorted(required - set(rows[0] if rows else []))
    if missing:
        raise ValueError(f"Step22 prediction columns missing: {missing}")
    if len({row["pair_uid"] for row in rows}) != len(rows):
        raise ValueError("Step22 OOF predictions contain duplicate pair_uid values")
    if {row["review_label"] for row in rows} != {"positive", "negative"}:
        raise ValueError("Step22 OOF predictions must contain both binary classes")
    return rows


def score_ap(rows: list[dict], score_column: str) -> float:
    labels = [int(row["review_label"] == "positive") for row in rows]
    scores = [float(row[score_column]) for row in rows]
    return average_precision(labels, scores)


def immutable_write_json(path: Path, payload: dict) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"Refusing to overwrite a different Step22 audit: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    predictions_path = root / "step22_grouped_oof_predictions.csv"
    evaluation_path = root / "step22_grouped_oof_evaluation.json"
    sync_manifest_path = root / "step22_sync_manifest.json"
    output_path = args.output or root / "step22_grouped_bootstrap_posthoc.json"
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if args.resamples < 1000:
        raise ValueError("Step22 grouped bootstrap requires at least 1000 resamples")

    sync_manifest = load_json(sync_manifest_path)
    expected_hashes = {entry["path"]: entry["sha256"] for entry in sync_manifest["files"]}
    for path in (predictions_path, evaluation_path):
        relative = path.relative_to(ROOT).as_posix()
        if expected_hashes.get(relative) != sha256_file(path):
            raise ValueError(f"Step22 frozen artifact hash mismatch: {relative}")

    evaluation = load_json(evaluation_path)
    if evaluation.get("valid_or_test_scores_used") is not False:
        raise ValueError("Step22 statistical audit refuses valid/test-selected predictions")
    if evaluation.get("publication_holdout_untouched") is not True:
        raise ValueError("Step22 publication holdout contract was not preserved")
    if evaluation.get("comparisons", {}).get("promotion_eligible") is not False:
        raise ValueError("This post-hoc audit is only defined for the frozen negative result")

    rows = load_predictions(predictions_path)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["component_id"]].append(row)
    components = sorted(groups)
    if len(components) != int(evaluation["real_chinese_train_components"]):
        raise ValueError("Component count does not reproduce the Step22 summary")

    comparisons = {
        "duplication_positive_minus_no_augmentation": (
            "prob_equal_weight_duplication_positive_budget", "prob_no_augmentation"
        ),
        "same_seller_positive_minus_no_augmentation": (
            "prob_same_seller_split_positive_only", "prob_no_augmentation"
        ),
        "same_seller_positive_minus_matched_duplication": (
            "prob_same_seller_split_positive_only", "prob_equal_weight_duplication_positive_budget"
        ),
        "duplication_full_minus_no_augmentation": (
            "prob_equal_weight_duplication_full_budget", "prob_no_augmentation"
        ),
        "full_method_minus_no_augmentation": (
            "prob_same_seller_split_plus_reviewed_negative_views", "prob_no_augmentation"
        ),
        "full_method_minus_matched_duplication": (
            "prob_same_seller_split_plus_reviewed_negative_views", "prob_equal_weight_duplication_full_budget"
        ),
    }
    point = {
        name: score_ap(rows, left) - score_ap(rows, right)
        for name, (left, right) in comparisons.items()
    }
    draws: dict[str, list[float]] = {name: [] for name in comparisons}
    rng = random.Random(args.seed)
    for _ in range(args.resamples):
        sampled_rows: list[dict] = []
        for _component_index in components:
            sampled_rows.extend(groups[rng.choice(components)])
        if {row["review_label"] for row in sampled_rows} != {"positive", "negative"}:
            continue
        for name, (left, right) in comparisons.items():
            draws[name].append(score_ap(sampled_rows, left) - score_ap(sampled_rows, right))

    comparison_output = {}
    for name in comparisons:
        values = draws[name]
        comparison_output[name] = {
            "point_delta_average_precision": point[name],
            "grouped_bootstrap_95_ci": [percentile(values, 0.025), percentile(values, 0.975)],
            "bootstrap_probability_delta_gt_zero": sum(value > 0.0 for value in values) / len(values),
            "valid_resamples": len(values),
        }

    largest_component = max(components, key=lambda component: len(groups[component]))
    reduced_rows = [row for row in rows if row["component_id"] != largest_component]
    payload = {
        "step": "step22_grouped_bootstrap_posthoc_negative_audit",
        "status": "posthoc_diagnostic_only_does_not_change_promotion",
        "seed": args.seed,
        "requested_resamples": args.resamples,
        "row_count": len(rows),
        "component_count": len(components),
        "comparisons": comparison_output,
        "largest_component_sensitivity": {
            "excluded_component_id": largest_component,
            "excluded_row_count": len(groups[largest_component]),
            "remaining_row_count": len(reduced_rows),
            "no_augmentation_ap": score_ap(reduced_rows, "prob_no_augmentation"),
            "same_seller_positive_ap": score_ap(reduced_rows, "prob_same_seller_split_positive_only"),
            "matched_full_duplication_ap": score_ap(reduced_rows, "prob_equal_weight_duplication_full_budget"),
            "full_method_ap": score_ap(reduced_rows, "prob_same_seller_split_plus_reviewed_negative_views"),
        },
        "promotion_eligible_before_and_after_audit": False,
        "scientific_scope": "negative_ablation_uncertainty_only_not_model_selection",
        "input_hashes": {
            "predictions": sha256_file(predictions_path),
            "evaluation": sha256_file(evaluation_path),
            "sync_manifest": sha256_file(sync_manifest_path),
            "producer": sha256_file(Path(__file__)),
        },
    }
    immutable_write_json(output_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "output": output_path.relative_to(ROOT).as_posix(),
        "promotion_eligible": False,
    }, indent=2))


if __name__ == "__main__":
    main()
