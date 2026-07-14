#!/usr/bin/env python3
"""Apply preregistered Step15-v8 validation gates and grouped bootstrap audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7
import step15_v7_common as v7
import step15_v8_common as common


ROOT = Path(__file__).resolve().parent.parent


def load_score_column(path: Path, rows: list[dict], column: str) -> np.ndarray:
    indexed = {row["pair_uid"]: row for row in common.load_csv(path)}
    expected = {row["pair_uid"] for row in rows}
    if set(indexed) != expected:
        raise ValueError(f"Step12-v8 score universe mismatch: {path}")
    return np.asarray([float(indexed[row["pair_uid"]][column]) for row in rows], dtype=float)


def load_prediction_index(path: Path, rows: list[dict]) -> dict[str, dict]:
    indexed = {row["pair_uid"]: row for row in common.load_csv(path)}
    expected = {row["pair_uid"] for row in rows}
    if set(indexed) != expected:
        raise ValueError(f"Step12-v8 prediction universe mismatch: {path}")
    return indexed


def grouped_bootstrap_delta(
    rows: list[dict],
    candidate: np.ndarray,
    baseline: np.ndarray,
    resamples: int,
    seed: int,
) -> dict:
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[common.component_group_key(row)].append(index)
    groups = [np.asarray(indices, dtype=int) for _, indices in sorted(grouped.items())]
    labels = v7.labels_array(rows)
    rng = np.random.default_rng(seed)
    deltas = []
    attempts = 0
    maximum_attempts = max(resamples * 20, 1000)
    while len(deltas) < resamples and attempts < maximum_attempts:
        attempts += 1
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        if len(set(labels[indices])) < 2:
            continue
        candidate_ap = step7.average_precision_score(labels[indices], candidate[indices])
        baseline_ap = step7.average_precision_score(labels[indices], baseline[indices])
        deltas.append(float(candidate_ap - baseline_ap))
    if len(deltas) != resamples:
        raise ValueError(
            f"Grouped bootstrap produced only {len(deltas)}/{resamples} valid resamples"
        )
    values = np.asarray(deltas, dtype=float)
    return {
        "resamples": resamples,
        "seed": seed,
        "component_count": len(groups),
        "mean_delta_average_precision": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "probability_candidate_below_baseline": float(np.mean(values < 0.0)),
    }


def subset_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    mask: np.ndarray,
    metric: str,
) -> tuple[float | None, int]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(labels):
        raise ValueError("Step12-v8 slice mask length differs from labels")
    y_true = labels[mask]
    if metric == "fpr":
        return common.false_positive_rate(y_true, scores[mask], threshold), int(np.sum(mask))
    if metric == "recall":
        return common.recall_at_threshold(y_true, scores[mask], threshold), int(np.sum(mask))
    raise ValueError(metric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026071403)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path, policy, v7_policy = common.load_policy(args.policy)
    validation = common.validate_policy_contract(policy, v7_policy)
    if args.validate_config_only:
        print(json.dumps(validation, indent=2))
        return
    run_id = args.run_id or policy["default_run_id"]
    root = common.run_root(policy, run_id)
    bridge_path = root / policy["bridge_audit"]["output_subdirectory"] / "step15_v8_bridge_audit_summary.json"
    expert_path = root / policy["occurrence_evidence_expert"]["output_subdirectory"] / "step15_v8_contextual_evidence_summary.json"
    if not bridge_path.is_file() or not expert_path.is_file():
        raise FileNotFoundError("Step12-v8 requires completed bridge and contextual evidence summaries")
    bridge = common.load_json(bridge_path)
    expert = common.load_json(expert_path)
    if bridge["selection"]["representative_valid_metrics_used_for_selection"] is not False:
        raise ValueError("Step12-v8 detected representative-valid model selection")
    if bridge["selection"]["internal_test_metrics_used_for_selection"] is not False:
        raise ValueError("Step12-v8 detected internal-test model selection")
    if expert["internal_test_used_for_model_fitting_selection_or_threshold"] is not False:
        raise ValueError("Step12-v8 detected internal-test evidence fitting")

    final_root = root / "step12"
    staging_root = final_root.with_name(f".{final_root.name}.incomplete")
    if final_root.exists() or staging_root.exists():
        raise FileExistsError(f"Refusing to overwrite Step12-v8: {final_root} / {staging_root}")
    rows_by_pool = common.load_joined_rows(policy, v7_policy, root)
    splits = common.split_rows(rows_by_pool)
    valid_rows = splits["valid"]
    test_rows = splits["internal_development_test"]
    b0_valid_path = common.resolve(bridge["seed_mean"]["B0_lr_l2"]["valid_prediction_path"])
    b0_test_path = common.resolve(
        bridge["seed_mean"]["B0_lr_l2"]["internal_test_prediction_path"]
    )
    expert_valid_path = common.resolve(expert["seed_mean"]["prediction_paths"]["zh_valid"])
    expert_test_path = common.resolve(
        expert["seed_mean"]["prediction_paths"]["internal_dev_test"]
    )
    b0_valid = load_score_column(b0_valid_path, valid_rows, "prob_positive")
    b0_test = load_score_column(b0_test_path, test_rows, "prob_positive")
    expert_valid_index = load_prediction_index(expert_valid_path, valid_rows)
    expert_test_index = load_prediction_index(expert_test_path, test_rows)
    clean_valid = np.asarray(
        [float(expert_valid_index[row["pair_uid"]]["clean_prob_positive"]) for row in valid_rows]
    )
    fused_valid = np.asarray(
        [
            float(
                expert_valid_index[row["pair_uid"]]["contextual_evidence_prob_positive"]
            )
            for row in valid_rows
        ]
    )
    clean_test = np.asarray(
        [float(expert_test_index[row["pair_uid"]]["clean_prob_positive"]) for row in test_rows]
    )
    fused_test = np.asarray(
        [
            float(expert_test_index[row["pair_uid"]]["contextual_evidence_prob_positive"])
            for row in test_rows
        ]
    )
    threshold = float(expert["clean_threshold_from_representative_valid"])
    y_valid = v7.labels_array(valid_rows)
    y_test = v7.labels_array(test_rows)
    b0_valid_ap = float(step7.average_precision_score(y_valid, b0_valid))
    clean_valid_ap = float(step7.average_precision_score(y_valid, clean_valid))
    fused_valid_ap = float(step7.average_precision_score(y_valid, fused_valid))

    persisted_valid_states = [
        expert_valid_index[row["pair_uid"]]["evidence_state"] for row in valid_rows
    ]
    zh_train_sellers = {
        str(row[key])
        for row in rows_by_pool["zh_target_strict"]
        if row["v7_split_name"] == "train"
        for key in ("seller_uid_left", "seller_uid_right")
    }
    occurrence_index, token_df = common.item_signal_index(
        common.resolve(policy["pools"]["zh_target_strict"]["item_identity_signals"]),
        zh_train_sellers,
    )
    frequency_threshold = int(
        policy["occurrence_evidence_expert"][
            "public_identifier_train_seller_frequency_threshold"
        ]
    )
    valid_states = [
        common.occurrence_evidence(
            row, occurrence_index, token_df, frequency_threshold
        )["evidence_state"]
        for row in valid_rows
    ]
    if persisted_valid_states != valid_states:
        first = next(
            index
            for index, (persisted, recomputed) in enumerate(
                zip(persisted_valid_states, valid_states, strict=True)
            )
            if persisted != recomputed
        )
        raise ValueError(
            "Step12-v8 recomputed occurrence state differs from the evidence prediction; "
            f"pair_uid={valid_rows[first]['pair_uid']} "
            f"persisted={persisted_valid_states[first]} recomputed={valid_states[first]}"
        )
    slice_masks = common.validation_slice_masks(valid_rows, valid_states)
    clean_public_fpr, public_count = subset_metric(
        y_valid,
        clean_valid,
        threshold,
        slice_masks["state_backed_public_noise_negative"],
        "fpr",
    )
    fused_public_fpr, _ = subset_metric(
        y_valid,
        fused_valid,
        threshold,
        slice_masks["state_backed_public_noise_negative"],
        "fpr",
    )
    clean_direct_recall, direct_count = subset_metric(
        y_valid,
        clean_valid,
        threshold,
        slice_masks["direct_or_component_positive"],
        "recall",
    )
    fused_direct_recall, _ = subset_metric(
        y_valid,
        fused_valid,
        threshold,
        slice_masks["direct_or_component_positive"],
        "recall",
    )
    clean_template_fpr, template_count = subset_metric(
        y_valid,
        clean_valid,
        threshold,
        slice_masks["template_clone_negative"],
        "fpr",
    )
    fused_template_fpr, _ = subset_metric(
        y_valid,
        fused_valid,
        threshold,
        slice_masks["template_clone_negative"],
        "fpr",
    )
    bootstrap_clean_vs_b0 = grouped_bootstrap_delta(
        valid_rows, clean_valid, b0_valid, args.resamples, args.seed
    )
    bootstrap_fusion_vs_clean = grouped_bootstrap_delta(
        valid_rows, fused_valid, clean_valid, args.resamples, args.seed
    )
    gates_cfg = policy["promotion_gates"]
    valid_counts = Counter(
        {
            key: int(np.sum(slice_masks[key]))
            for key in gates_cfg["minimum_valid_slice_counts"]
        }
    )
    data_readiness = {
        evidence_type: {
            "observed": int(valid_counts[evidence_type]),
            "required": int(required),
            "met": int(valid_counts[evidence_type]) >= int(required),
        }
        for evidence_type, required in gates_cfg["minimum_valid_slice_counts"].items()
    }

    gates = {
        "clean_ap_gain_over_B0": {
            "observed": clean_valid_ap - b0_valid_ap,
            "required_minimum": float(
                gates_cfg["clean_valid_average_precision_gain_over_B0_minimum"]
            ),
        },
        "public_noise_fpr_reduction": {
            "observed": (
                None
                if clean_public_fpr is None or fused_public_fpr is None
                else float(clean_public_fpr - fused_public_fpr)
            ),
            "required_minimum": float(
                gates_cfg["public_noise_valid_fpr_reduction_minimum"]
            ),
        },
        "direct_component_recall_drop": {
            "observed": (
                None
                if clean_direct_recall is None or fused_direct_recall is None
                else float(clean_direct_recall - fused_direct_recall)
            ),
            "allowed_maximum": float(
                gates_cfg["direct_or_component_valid_recall_drop_maximum"]
            ),
        },
        "template_fpr_increase": {
            "observed": (
                None
                if clean_template_fpr is None or fused_template_fpr is None
                else float(fused_template_fpr - clean_template_fpr)
            ),
            "allowed_maximum": float(
                gates_cfg["template_clone_valid_fpr_increase_maximum"]
            ),
        },
        "fusion_ap_drop": {
            "observed": clean_valid_ap - fused_valid_ap,
            "allowed_maximum": float(
                gates_cfg["fusion_valid_average_precision_drop_maximum"]
            ),
        },
        "grouped_bootstrap_clean_vs_B0_noninferiority": {
            "observed_ci_low": bootstrap_clean_vs_b0["ci_low"],
            "required_minimum": float(
                gates_cfg["grouped_bootstrap_clean_vs_B0_noninferiority_margin"]
            ),
        },
        "grouped_bootstrap_fusion_vs_clean_noninferiority": {
            "observed_ci_low": bootstrap_fusion_vs_clean["ci_low"],
            "required_minimum": float(
                gates_cfg["grouped_bootstrap_fusion_vs_clean_noninferiority_margin"]
            ),
        },
        "selection_never_read_internal_test": {
            "met": bridge["selection"]["internal_test_metrics_used_for_selection"] is False
            and expert["internal_test_used_for_model_fitting_selection_or_threshold"] is False
        },
    }
    gates["clean_ap_gain_over_B0"]["met"] = (
        gates["clean_ap_gain_over_B0"]["observed"]
        >= gates["clean_ap_gain_over_B0"]["required_minimum"]
    )
    gates["public_noise_fpr_reduction"]["met"] = (
        gates["public_noise_fpr_reduction"]["observed"] is not None
        and gates["public_noise_fpr_reduction"]["observed"]
        >= gates["public_noise_fpr_reduction"]["required_minimum"]
    )
    gates["direct_component_recall_drop"]["met"] = (
        gates["direct_component_recall_drop"]["observed"] is not None
        and gates["direct_component_recall_drop"]["observed"]
        <= gates["direct_component_recall_drop"]["allowed_maximum"]
    )
    gates["template_fpr_increase"]["met"] = (
        gates["template_fpr_increase"]["observed"] is not None
        and gates["template_fpr_increase"]["observed"]
        <= gates["template_fpr_increase"]["allowed_maximum"]
    )
    gates["fusion_ap_drop"]["met"] = (
        gates["fusion_ap_drop"]["observed"]
        <= gates["fusion_ap_drop"]["allowed_maximum"]
    )
    for gate_name in (
        "grouped_bootstrap_clean_vs_B0_noninferiority",
        "grouped_bootstrap_fusion_vs_clean_noninferiority",
    ):
        gates[gate_name]["met"] = (
            gates[gate_name]["observed_ci_low"]
            >= gates[gate_name]["required_minimum"]
        )
    method_gates_met = all(bool(item["met"]) for item in gates.values())
    data_readiness_met = all(bool(item["met"]) for item in data_readiness.values())
    promotion_eligible = method_gates_met and data_readiness_met

    metrics_rows = []
    for split_name, rows, labels, models in (
        (
            "representative_valid",
            valid_rows,
            y_valid,
            {"B0_v7_control": b0_valid, "selected_clean": clean_valid, "contextual_evidence": fused_valid},
        ),
        (
            "internal_development_test_diagnostic_only",
            test_rows,
            y_test,
            {"B0_v7_control": b0_test, "selected_clean": clean_test, "contextual_evidence": fused_test},
        ),
    ):
        for model_id, scores in models.items():
            metrics = step7.evaluate_probabilities(labels, scores, threshold)
            metrics_rows.append(
                {
                    "split": split_name,
                    "model_id": model_id,
                    "row_count": len(rows),
                    "positive_count": int(np.sum(labels)),
                    "negative_count": int(len(labels) - np.sum(labels)),
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "pr_auc": metrics.get("pr_auc"),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "f1": metrics["f1"],
                    "threshold": threshold,
                    "used_for_selection": False,
                }
            )

    summary = {
        "step": "step12_v8_statistical_robustness_audit",
        "version": policy["version"],
        "run_id": run_id,
        "selection": bridge["selection"],
        "representative_valid": {
            "row_count": len(valid_rows),
            "positive_count": int(np.sum(y_valid)),
            "negative_count": int(len(y_valid) - np.sum(y_valid)),
            "B0_average_precision": b0_valid_ap,
            "selected_clean_average_precision": clean_valid_ap,
            "contextual_evidence_average_precision": fused_valid_ap,
            "public_noise_count": public_count,
            "direct_component_count": direct_count,
            "template_clone_count": template_count,
            "state_backed_verified_direct_positive_count": int(
                np.sum(slice_masks["state_backed_verified_direct_positive"])
            ),
            "component_anchor_positive_count": int(
                np.sum(slice_masks["same_controller_component_anchor_positive"])
            ),
            "slice_definition": gates_cfg["validation_slice_definition"],
            "legacy_evidence_type_only_public_slice_used": False,
            "clean_public_noise_fpr": clean_public_fpr,
            "contextual_public_noise_fpr": fused_public_fpr,
            "clean_direct_component_recall": clean_direct_recall,
            "contextual_direct_component_recall": fused_direct_recall,
            "clean_template_fpr": clean_template_fpr,
            "contextual_template_fpr": fused_template_fpr,
        },
        "grouped_bootstrap_clean_minus_B0": bootstrap_clean_vs_b0,
        "grouped_bootstrap_contextual_minus_clean": bootstrap_fusion_vs_clean,
        "promotion": {
            "eligible": promotion_eligible,
            "method_gates_met": method_gates_met,
            "validation_data_readiness_met": data_readiness_met,
            "gates": gates,
            "validation_slice_readiness": data_readiness,
            "publication_claim_still_requires_step20": True,
            "internal_test_satisfied_no_gate": True,
        },
        "metrics": metrics_rows,
        "inputs": {
            "policy_sha256": common.sha256(policy_path),
            "bridge_summary_sha256": common.sha256(bridge_path),
            "contextual_evidence_summary_sha256": common.sha256(expert_path),
        },
    }
    summary["summary_sha256"] = common.canonical_hash(summary)
    staging_root.mkdir(parents=True, exist_ok=False)
    summary_path = staging_root / "step12_v8_statistical_robustness.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metric_fields = list(metrics_rows[0])
    metrics_path = staging_root / "step12_v8_model_metrics.csv"
    metrics_path.write_bytes(common.render_csv(metrics_rows, metric_fields))
    freeze = {
        "step": "step15_v8_model_and_threshold_freeze",
        "run_id": run_id,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_feature_set": bridge["selection"]["feature_representation"][
            "selected_feature_set_id"
        ],
        "selected_model_family": bridge["selection"]["model_family"][
            "selected_model_family"
        ],
        "threshold_from_representative_valid": threshold,
        "method_promotion_eligible": promotion_eligible,
        "step20_prospective_evaluation_required": True,
        "prospective_holdout_required": True,
        "internal_test_used_for_selection": False,
        "current_internal_test_used_for_model_selection": False,
        "inputs": {
            "v8_policy": common.sha256(policy_path),
            "bridge_summary": common.sha256(bridge_path),
            "contextual_evidence_summary": common.sha256(expert_path),
        },
        "summary_sha256": common.sha256(summary_path),
    }
    freeze["freeze_manifest_sha256"] = common.canonical_hash(freeze)
    freeze_path = staging_root / "step15_v8_model_freeze_manifest.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    staging_root.replace(final_root)
    print(
        json.dumps(
            {
                "status": "pass",
                "run_id": run_id,
                "promotion_eligible": promotion_eligible,
                "method_gates_met": method_gates_met,
                "validation_data_readiness_met": data_readiness_met,
                "summary": str((final_root / summary_path.name).relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
