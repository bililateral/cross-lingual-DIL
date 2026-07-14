#!/usr/bin/env python3
"""Select the Step9-v7 clean ranker on validation and apply the fixed reliability veto."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7
import step15_v7_common as common


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step15_v7_two_stage_policy.json"


def load_predictions(path: Path) -> list[dict]:
    return common.load_csv(path)


def render_csv(rows: list[dict], fields: list[str]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Step15-v7 two-stage artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def metric_slices(rows: list[dict], probabilities: np.ndarray, threshold: float) -> dict:
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["evidence_type"]].append(index)
    result = {}
    y_all = common.labels_array(rows)
    for evidence_type, indices in sorted(grouped.items()):
        selected = np.asarray(indices, dtype=int)
        result[evidence_type] = step7.evaluate_probabilities(
            y_all[selected], probabilities[selected], threshold
        )
    return result


def prediction_output_rows(
    rows: list[dict],
    clean: np.ndarray,
    vetoed: np.ndarray,
    decisions: list[dict],
    threshold: float,
    split: str,
) -> list[dict]:
    output = []
    for row, clean_score, vetoed_score, decision in zip(
        rows, clean, vetoed, decisions, strict=True
    ):
        output.append(
            {
                "pair_uid": row["pair_uid"],
                "split_name": split,
                "review_label": row["review_label"],
                "evidence_type": row["evidence_type"],
                "v7_component_id": row["v7_component_id"],
                "clean_prob_positive": f"{float(clean_score):.12f}",
                "reliability_veto_prob_positive": f"{float(vetoed_score):.12f}",
                "selected_threshold": f"{threshold:.12f}",
                "predicted_label": int(float(vetoed_score) >= threshold),
                "reliability_decision": decision["decision"],
                "reliability_score_multiplier": f"{float(decision['score_multiplier']):.6f}",
                "strong_direct_token_count": decision["strong_direct_token_count"],
                "public_noise_token_count": decision["public_noise_token_count"],
                "ambiguous_token_count": decision["ambiguous_token_count"],
                "shared_token_count": decision["shared_token_count"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--step9-run-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path = common.resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    method_cfg = policy["two_stage_method"]
    if method_cfg["stage_a"].get("auxiliary_evidence_head") is not False:
        raise ValueError("Step15-v7 clean ranker must not include an auxiliary evidence head")
    forbidden = set(method_cfg["stage_b"]["forbidden_inputs"])
    if not {"review_label", "evidence_type", "zh_test_membership"}.issubset(forbidden):
        raise ValueError("Stage-B inference input prohibition is incomplete")
    if args.validate_config_only:
        print(json.dumps({"status": "pass", "stage_a": method_cfg["stage_a"], "stage_b": method_cfg["stage_b"]}, indent=2))
        return

    step9_run_id = args.step9_run_id or method_cfg["step9_run_id"]
    run_id = args.run_id or method_cfg["default_run_id"]
    step9_root = common.resolve(policy["step9_latent_mixup"]["outputs_root"]) / step9_run_id
    step9_summary_path = step9_root / "step9_v7_latent_pair_mixup_summary.json"
    step9_summary = json.loads(step9_summary_path.read_text(encoding="utf-8"))
    selection = step9_summary["selection"].get("1.0")
    if selection is None or selection.get("test_metrics_used_for_selection") is not False:
        raise ValueError("Step9-v7 has no test-independent 100% support selection")
    selected_experiment = selection["selected_experiment"]
    seeds = [int(value) for value in policy["step9_latent_mixup"]["seeds"]]
    output_root = common.resolve(policy["outputs"]["two_stage_outputs_root"]) / run_id
    staging_root = output_root.with_name(f".{output_root.name}.incomplete")
    if output_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Step15-v7 final or incomplete run directory already exists: {output_root} / {staging_root}"
        )

    def staged(final_path: Path) -> Path:
        return staging_root / final_path.relative_to(output_root)

    pools = common.load_joined_rows(policy)
    zh_rows = pools["zh_target_strict"]
    zh_valid = [row for row in zh_rows if row["v7_split_name"] == "valid"]
    zh_test = [row for row in zh_rows if row["v7_split_name"] == "internal_development_test"]
    zh_train_sellers = sorted(
        {
            row[key]
            for row in zh_rows
            if row["v7_split_name"] == "train"
            for key in ("seller_uid_left", "seller_uid_right")
        }
    )
    reliability_pool_cfg = {
        **policy["pools"]["zh_target_strict"],
        "identifier_frequency_reference_sellers": zh_train_sellers,
    }
    row_maps = {
        "valid": {row["pair_uid"]: row for row in zh_valid},
        "internal_development_test": {row["pair_uid"]: row for row in zh_test},
    }
    pair_order = {split: sorted(rows) for split, rows in row_maps.items()}
    per_seed = []
    scores = {"valid": [], "internal_development_test": []}
    for seed in seeds:
        run_key = f"{selected_experiment}__ratio_100pct__seed_{seed}"
        paths = {
            "valid": step9_root / "predictions" / f"{run_key}.zh_valid.csv",
            "internal_development_test": step9_root / "predictions" / f"{run_key}.internal_dev_test.csv",
        }
        seed_scores = {}
        seed_decisions = {}
        seed_rows = {}
        for split, path in paths.items():
            prediction_rows = load_predictions(path)
            expected = row_maps[split]
            if set(row["pair_uid"] for row in prediction_rows) != set(expected):
                raise ValueError(f"Step9-v7 prediction universe mismatch: {path}")
            prediction_index = {row["pair_uid"]: row for row in prediction_rows}
            if len(prediction_index) != len(prediction_rows):
                raise ValueError(f"Step9-v7 prediction file contains duplicate pair UIDs: {path}")
            ordered_rows = [expected[pair_uid] for pair_uid in pair_order[split]]
            clean = np.asarray(
                [float(prediction_index[pair_uid]["prob_positive"]) for pair_uid in pair_order[split]]
            )
            vetoed, decisions, _ = common.apply_reliability_veto(
                ordered_rows,
                clean,
                reliability_pool_cfg,
                method_cfg["stage_b"],
            )
            seed_scores[split] = (clean, vetoed)
            seed_decisions[split] = decisions
            seed_rows[split] = ordered_rows
            scores[split].append(vetoed)
        y_valid = common.labels_array(seed_rows["valid"])
        threshold = step7.choose_threshold(
            y_valid,
            seed_scores["valid"][1],
            policy["threshold_selection"]["metric"],
            policy,
        )
        seed_record = {
            "seed": seed,
            "selected_clean_experiment": selected_experiment,
            "selection_source": "step9_v7_seed_mean_representative_valid_average_precision",
            "test_metrics_used_for_selection": False,
            "selected_threshold_from_representative_valid": threshold,
            "splits": {},
        }
        for split in ("valid", "internal_development_test"):
            ordered_rows = seed_rows[split]
            clean, vetoed = seed_scores[split]
            y_true = common.labels_array(ordered_rows)
            seed_record["splits"][split] = {
                "clean_metrics": step7.evaluate_probabilities(y_true, clean, threshold),
                "two_stage_metrics": step7.evaluate_probabilities(y_true, vetoed, threshold),
                "two_stage_evidence_slices": metric_slices(ordered_rows, vetoed, threshold),
                "reliability_decision_counts": dict(
                    sorted(Counter(item["decision"] for item in seed_decisions[split]).items())
                ),
            }
            output_path = output_root / "predictions" / f"two_stage_seed_{seed}.{split}.csv"
            fields = [
                "pair_uid",
                "split_name",
                "review_label",
                "evidence_type",
                "v7_component_id",
                "clean_prob_positive",
                "reliability_veto_prob_positive",
                "selected_threshold",
                "predicted_label",
                "reliability_decision",
                "reliability_score_multiplier",
                "strong_direct_token_count",
                "public_noise_token_count",
                "ambiguous_token_count",
                "shared_token_count",
            ]
            write_new(
                staged(output_path),
                render_csv(
                    prediction_output_rows(
                        ordered_rows, clean, vetoed, seed_decisions[split], threshold, split
                    ),
                    fields,
                ),
            )
        per_seed.append(seed_record)

    seed_mean = {}
    seed_mean_threshold = step7.choose_threshold(
        common.labels_array([row_maps["valid"][pair_uid] for pair_uid in pair_order["valid"]]),
        np.mean(np.vstack(scores["valid"]), axis=0),
        policy["threshold_selection"]["metric"],
        policy,
    )
    for split, rows in (("valid", zh_valid), ("internal_development_test", zh_test)):
        mean_scores = np.mean(np.vstack(scores[split]), axis=0)
        ordered_rows = [row_maps[split][pair_uid] for pair_uid in pair_order[split]]
        y_true = common.labels_array(ordered_rows)
        seed_mean[split] = {
            "selected_threshold_from_representative_valid": seed_mean_threshold,
            "metrics": step7.evaluate_probabilities(y_true, mean_scores, seed_mean_threshold),
            "evidence_slices": metric_slices(ordered_rows, mean_scores, seed_mean_threshold),
        }
        mean_path = output_root / "predictions" / f"two_stage_seed_mean.{split}.csv"
        mean_rows = []
        for row, score in zip(ordered_rows, mean_scores, strict=True):
            mean_rows.append(
                {
                    "pair_uid": row["pair_uid"],
                    "split_name": split,
                    "review_label": row["review_label"],
                    "evidence_type": row["evidence_type"],
                    "v7_component_id": row["v7_component_id"],
                    "prob_positive": f"{float(score):.12f}",
                    "selected_threshold": f"{seed_mean_threshold:.12f}",
                    "predicted_label": int(float(score) >= seed_mean_threshold),
                }
            )
        write_new(
            staged(mean_path),
            render_csv(
                mean_rows,
                [
                    "pair_uid",
                    "split_name",
                    "review_label",
                    "evidence_type",
                    "v7_component_id",
                    "prob_positive",
                    "selected_threshold",
                    "predicted_label",
                ],
            ),
        )

    summary = {
        "step": "step15_v7_two_stage",
        "version": policy["version"],
        "run_id": run_id,
        "selected_clean_experiment": selected_experiment,
        "clean_selection": selection,
        "auxiliary_evidence_head_used": False,
        "stage_b_uses_review_label_or_evidence_type_as_features": False,
        "stage_b_policy": method_cfg["stage_b"],
        "identifier_frequency_reference_scope": "v7_zh_train_sellers_only",
        "identifier_frequency_reference_seller_count": len(zh_train_sellers),
        "identifier_frequency_reference_sellers_sha256": common.canonical_hash(zh_train_sellers),
        "representative_validation_used_for_model_selection": True,
        "current_internal_test_used_for_model_selection": False,
        "current_zh_test_role": "internal_development_test_only",
        "seed_mean": seed_mean,
        "per_seed": per_seed,
        "inputs": {
            "step9_summary": str(step9_summary_path.relative_to(ROOT)).replace("\\", "/"),
            "step9_summary_sha256": common.sha256(step9_summary_path),
            "representative_validation_manifest": policy["representative_validation"]["manifest_output"],
        },
        "policy": str(policy_path.relative_to(ROOT)).replace("\\", "/"),
        "policy_sha256": common.sha256(policy_path),
    }
    summary["summary_sha256"] = common.canonical_hash(summary)
    summary_path = output_root / "step15_v7_two_stage_summary.json"
    write_new(staged(summary_path), (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    staging_root.replace(output_root)
    print(json.dumps({"status": "pass", "summary": str(summary_path.relative_to(ROOT)), "selected_clean_experiment": selected_experiment}, indent=2))


if __name__ == "__main__":
    main()
