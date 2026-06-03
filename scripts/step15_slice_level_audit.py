#!/usr/bin/env python3
"""Step 15 fixed-test slice-level audit.

This audit is intentionally descriptive. It keeps the current zh_target_strict
test split fixed and checks whether Step 15 reduces the score assigned to the
specific evidence/noise slices it was designed to handle.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import step7_train_baseline_models as step7
import step12_statistical_robustness_audit as step12


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_PATH = ROOT / "schema" / "step15_evidence_type_policy.json"

DEFAULT_MODEL_IDS = [
    "raw_e5_cosine",
    "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean",
    "step15_identity_only_phase4_seed_mean",
    "step15_v2_identity_from_scratch_phase4_seed_mean",
    "step15_v2_warm_start_phase4_seed_mean",
    "step15_v2_domain_balanced_phase4_seed_mean",
    "step15_v2_zh_positive_mixup_phase4_seed_mean",
    "step15_v2_identifier_operational_phase4_seed_mean",
    "step15_v4_public_noise_robust_phase4_seed_mean",
    "step15_v4_domain_balanced_public_noise_robust_phase4_seed_mean",
    "step15_v4_identifier_public_noise_robust_phase4_seed_mean",
    "step15_v5_public_noise_weighted_strong_phase4_seed_mean",
    "step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Step15 scores by fixed zh_test evidence/noise slices.")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH), help="Path to Step15 policy JSON.")
    parser.add_argument("--model-id", action="append", dest="model_ids", help="Model id to audit. Repeatable.")
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_score_maps(pair_uids: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]], list[dict[str, str]]]:
    specs = [dict(spec) for spec in step12.MODEL_SPECS]
    score_maps: dict[str, dict[str, float]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []

    for spec in specs:
        if spec["kind"] == "feature":
            spec["path"] = step12.DEFAULT_FEATURES
        spec_path = resolve_path(spec["path"])
        if bool(spec.get("optional_until_generated", False)) and not spec_path.exists():
            skipped.append({"model_id": str(spec["model_id"]), "reason": "optional_prediction_file_not_found"})
            continue
        local_spec = dict(spec)
        local_spec["path"] = spec_path
        score_map = step12.load_score_map(local_spec)
        missing = [pair_uid for pair_uid in pair_uids if pair_uid not in score_map]
        if missing:
            raise ValueError(f"{spec['model_id']} is missing {len(missing)} zh_test pair scores; first={missing[0]}")
        score_maps[str(spec["model_id"])] = score_map
        metadata[str(spec["model_id"])] = {
            "role": spec["role"],
            "kind": spec["kind"],
            "path": str(spec_path.relative_to(ROOT)),
            "score_column": spec["score_column"],
        }

    for model_id, cfg in step12.ENSEMBLES.items():
        members = cfg["members"]
        missing_members = [member for member in members if member not in score_maps]
        if missing_members and bool(cfg.get("optional_until_generated", False)):
            skipped.append(
                {
                    "model_id": model_id,
                    "reason": "optional_ensemble_members_not_found",
                    "missing_members": "|".join(missing_members),
                }
            )
            continue
        if missing_members:
            raise ValueError(f"Ensemble {model_id} references missing members: {missing_members}")
        score_maps[model_id] = {
            pair_uid: sum(score_maps[member][pair_uid] for member in members) / len(members)
            for pair_uid in pair_uids
        }
        metadata[model_id] = {"role": cfg["role"], "kind": "seed_mean_ensemble", "members": "|".join(members)}

    return score_maps, metadata, skipped


def y_value(row: dict[str, str]) -> int:
    if row["review_label"] == "positive":
        return 1
    if row["review_label"] == "negative":
        return 0
    raise ValueError(f"Unsupported label in fixed zh_test audit: {row['review_label']}")


def metric_or_none(metric_name: str, y_true: list[int], scores: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    if metric_name == "roc_auc":
        value = step7.roc_auc_score(y, s)
    elif metric_name == "average_precision":
        value = step7.average_precision_score(y, s)
    else:
        raise ValueError(metric_name)
    return None if value is None else round(float(value), 6)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return round(float(np.quantile(np.asarray(values, dtype=float), q)), 6)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(np.asarray(values, dtype=float))), 6)


def build_slices(test_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    slices: dict[str, list[dict[str, str]]] = {"all": test_rows}
    by_evidence: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_stratum: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        by_evidence[row.get("evidence_type", "")].append(row)
        by_stratum[row.get("review_stratum", "") or "missing_review_stratum"].append(row)
    for evidence_type, rows in sorted(by_evidence.items()):
        slices[f"evidence_type::{evidence_type}"] = rows
    for review_stratum, rows in sorted(by_stratum.items()):
        slices[f"review_stratum::{review_stratum}"] = rows

    hard_negative_types = {
        "template_clone_not_controller",
        "semantic_topic_not_controller",
        "public_contact_or_url_noise",
    }
    direct_positive_types = {
        "same_controller_direct_identifier",
        "same_controller_component_anchor",
    }
    soft_positive_types = {"same_controller_style_structural_soft"}
    slices["hard_negative_any"] = [
        row for row in test_rows if row["review_label"] == "negative" and row.get("evidence_type") in hard_negative_types
    ]
    slices["negative_template_or_topic"] = [
        row
        for row in test_rows
        if row["review_label"] == "negative"
        and row.get("evidence_type") in {"template_clone_not_controller", "semantic_topic_not_controller"}
    ]
    slices["positive_direct_or_component_anchor"] = [
        row for row in test_rows if row["review_label"] == "positive" and row.get("evidence_type") in direct_positive_types
    ]
    slices["positive_style_structural_soft"] = [
        row for row in test_rows if row["review_label"] == "positive" and row.get("evidence_type") in soft_positive_types
    ]
    slices["identifier_present"] = [row for row in test_rows if row.get("has_direct_identifier_signal") == "1"]
    slices["identifier_absent"] = [row for row in test_rows if row.get("has_direct_identifier_signal") != "1"]
    return {name: rows for name, rows in slices.items() if rows}


def summarize_slice(model_id: str, rows: list[dict[str, str]], score_map: dict[str, float]) -> dict[str, Any]:
    y_true = [y_value(row) for row in rows]
    scores = [score_map[row["pair_uid"]] for row in rows]
    positive_scores = [score for score, y in zip(scores, y_true, strict=True) if y == 1]
    negative_scores = [score for score, y in zip(scores, y_true, strict=True) if y == 0]
    return {
        "model_id": model_id,
        "row_count": len(rows),
        "positive_count": sum(y_true),
        "negative_count": len(rows) - sum(y_true),
        "roc_auc": metric_or_none("roc_auc", y_true, scores),
        "average_precision": metric_or_none("average_precision", y_true, scores),
        "score_mean": mean_or_none(scores),
        "positive_score_mean": mean_or_none(positive_scores),
        "negative_score_mean": mean_or_none(negative_scores),
        "negative_score_p90": quantile(negative_scores, 0.9),
        "negative_score_max": max(negative_scores) if negative_scores else None,
        "positive_score_min": min(positive_scores) if positive_scores else None,
        "positive_score_p10": quantile(positive_scores, 0.1),
    }


def top_k_rows(model_id: str, test_rows: list[dict[str, str]], score_map: dict[str, float], k_values: list[int]) -> list[dict[str, Any]]:
    ranked = sorted(test_rows, key=lambda row: (-score_map[row["pair_uid"]], row["pair_uid"]))
    output = []
    for k in k_values:
        selected = ranked[: min(k, len(ranked))]
        label_counts = Counter(row["review_label"] for row in selected)
        evidence_counts = Counter(row.get("evidence_type", "") for row in selected)
        output.append(
            {
                "model_id": model_id,
                "top_k": k,
                "selected_count": len(selected),
                "positive_count": label_counts.get("positive", 0),
                "negative_count": label_counts.get("negative", 0),
                "precision_at_k": round(label_counts.get("positive", 0) / max(len(selected), 1), 6),
                "evidence_type_counts": dict(sorted(evidence_counts.items())),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    policy = step7.load_json(resolve_path(args.policy))
    zh_label_path = resolve_path(policy["pools"]["zh_target_strict"]["label_output"])
    label_rows = step7.load_csv(zh_label_path)
    test_rows = [
        row
        for row in label_rows
        if row.get("split_name") == "test"
        and row.get("review_label") in {"positive", "negative"}
        and row.get("identity_training_eligible") == "1"
    ]
    if not test_rows:
        raise SystemExit("No fixed zh_test rows available for Step15 slice audit.")
    pair_uids = [row["pair_uid"] for row in test_rows]
    score_maps, metadata, skipped = load_score_maps(pair_uids)
    selected_model_ids = args.model_ids or policy.get("slice_level_audit", {}).get("default_model_ids", DEFAULT_MODEL_IDS)
    selected_model_ids = [model_id for model_id in selected_model_ids if model_id in score_maps]
    if not selected_model_ids:
        raise SystemExit("No requested Step15 slice audit models are available.")

    slices = build_slices(test_rows)
    slice_rows: list[dict[str, Any]] = []
    for slice_name, rows in sorted(slices.items()):
        for model_id in selected_model_ids:
            record = summarize_slice(model_id, rows, score_maps[model_id])
            record["slice_name"] = slice_name
            slice_rows.append(record)

    top_k_summary = []
    for model_id in selected_model_ids:
        top_k_summary.extend(top_k_rows(model_id, test_rows, score_maps[model_id], [10, 20, 30]))

    outputs = policy.get("slice_level_audit", {}).get("outputs", {})
    out_csv = resolve_path(outputs.get("slice_metrics_csv", "reports/step15_slice_level_audit.csv"))
    out_json = resolve_path(outputs.get("summary_json", "reports/step15_slice_level_audit.json"))
    fieldnames = [
        "model_id",
        "slice_name",
        "row_count",
        "positive_count",
        "negative_count",
        "roc_auc",
        "average_precision",
        "score_mean",
        "positive_score_mean",
        "negative_score_mean",
        "negative_score_p90",
        "negative_score_max",
        "positive_score_min",
        "positive_score_p10",
    ]
    step7.write_csv(out_csv, slice_rows, fieldnames)
    summary = {
        "step": "step15_slice_level_audit",
        "policy": str(resolve_path(args.policy).relative_to(ROOT)),
        "policy_version": policy.get("version"),
        "scope": "zh_target_strict fixed test split",
        "test_dataset": {
            "row_count": len(test_rows),
            "label_counts": dict(sorted(Counter(row["review_label"] for row in test_rows).items())),
            "evidence_type_counts": dict(sorted(Counter(row.get("evidence_type", "") for row in test_rows).items())),
        },
        "model_ids": selected_model_ids,
        "model_metadata": {model_id: metadata[model_id] for model_id in selected_model_ids},
        "skipped_models": skipped,
        "slice_metrics_path": str(out_csv.relative_to(ROOT)),
        "top_k_summary": top_k_summary,
        "hard_rules": [
            "Fixed zh_test only; no train/valid/test mixing.",
            "Uses Step15 auxiliary evidence types for descriptive slicing only.",
            "Does not relabel Step5 rows.",
            "Does not use Step11 clusters as same-controller ground truth.",
        ],
    }
    step7.write_json(out_json, summary)
    print(json.dumps({"summary": str(out_json.relative_to(ROOT)), "slice_metrics": str(out_csv.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
