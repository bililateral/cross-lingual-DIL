#!/usr/bin/env python3
"""Fail closed unless the complete Step15-v6 method-audit run satisfies its output contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step15_v6_paper_hardening_policy.json"


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scores(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = {row["pair_uid"]: float(row["prob_positive"]) for row in rows}
    if len(scores) != len(rows):
        raise ValueError(f"Duplicate pair_uid in prediction file: {path}")
    return scores


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_identical_scores(left_path: Path, right_path: Path, label: str) -> None:
    left = load_scores(left_path)
    right = load_scores(right_path)
    if left != right:
        differing = [pair_uid for pair_uid in left if left.get(pair_uid) != right.get(pair_uid)]
        raise ValueError(
            f"Matched Step15-v6 pre-treatment predictions differ for {label}: "
            f"coverage_equal={set(left) == set(right)} first_difference={differing[:1]}"
        )


def assert_identical_artifact_params(left_path: Path, right_path: Path, label: str) -> None:
    left = load_json(left_path).get("params")
    right = load_json(right_path).get("params")
    if left != right:
        raise ValueError(f"Matched Step15-v6 pre-treatment parameters differ for {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument(
        "--output",
        default="reports/step15_v6/manifests/step15_v6_output_validation_20260712.json",
    )
    args = parser.parse_args()
    policy_path = resolve(args.policy)
    policy = load_json(policy_path)
    summary_path = resolve(policy["outputs"]["summary_json"])
    source_summary_path = resolve(policy["source_only_lr_baseline"]["output_summary"])
    summary = load_json(summary_path)
    source_summary = load_json(source_summary_path)
    if summary.get("policy_version") != policy.get("version"):
        raise ValueError("Step15-v6 output validator found a summary/policy version mismatch")
    expected_seeds = [int(seed) for seed in policy["training"]["default_seeds"]]
    expected_experiments = [
        *policy["default_experiments"],
        "step15_v6_m3_normalized_retrieval_ablation",
        "step15_v6_m3_gold_only_ablation",
        "step15_v6_m3_gold_plus_high_confidence_silver_ablation",
    ]
    run_index: dict[tuple[str, str, int], dict] = {}
    for run in summary.get("runs", []):
        key = (str(run["experiment_name"]), str(run["phase_id"]), int(run["seed"]))
        if key in run_index:
            raise ValueError(f"Duplicate Step15-v6 summary run: {key}")
        run_index[key] = run
    expected_keys = {
        (experiment_name, phase_id, seed)
        for experiment_name in expected_experiments
        for phase_id in policy["experiments"][experiment_name]["phase_ids"]
        for seed in expected_seeds
    }
    actual_keys = {key for key in run_index if key[0] in set(expected_experiments)}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"Step15-v6 run coverage mismatch: missing={len(missing)} extra={len(extra)} "
            f"first_missing={missing[:1]} first_extra={extra[:1]}"
        )
    fixed_budget_experiments = {
        experiment_name
        for experiment_name in expected_experiments
        if policy["experiments"][experiment_name].get("fixed_update_budget_per_phase") is not None
    }
    selection = (summary.get("validation_only_model_selection") or {}).get(
        "m5_auxiliary_loss_weight", {}
    )
    selected_m5 = str(selection.get("selected_experiment", ""))
    m5_candidates = set(
        policy["validation_only_model_selection"]["m5_auxiliary_loss_weight"][
            "candidate_experiments"
        ]
    )
    if selected_m5 not in m5_candidates:
        raise ValueError("Step15-v6 M5 valid-only selection did not freeze one candidate")
    validated_paths = {policy_path, summary_path, source_summary_path}
    for key in sorted(expected_keys):
        run = run_index[key]
        configured_phases = policy["experiments"][key[0]]["phase_ids"]
        is_endpoint = key[1] == configured_phases[-1]
        test_path_value = (run.get("output_paths") or {}).get("zh_test_predictions")
        should_have_test = is_endpoint and (
            key[0] not in m5_candidates or key[0] == selected_m5
        )
        if should_have_test and not test_path_value:
            raise ValueError(f"Step15-v6 endpoint is missing zh_test prediction: {key}")
        if not should_have_test and test_path_value:
            raise ValueError(f"Step15-v6 non-selected/intermediate run exposed zh_test predictions: {key}")
        if should_have_test and key[0] == selected_m5:
            expected_test_role = "validation_selected_frozen_artifact_preregistered_endpoint_only"
        elif should_have_test:
            expected_test_role = "final_preregistered_endpoint_only"
        else:
            expected_test_role = "not_evaluated_intermediate_phase"
        if run.get("zh_test_evaluation_role") != expected_test_role:
            raise ValueError(f"Step15-v6 zh_test evaluation role mismatch for {key}")
        for value in (run.get("output_paths") or {}).values():
            if value:
                path = resolve(str(value))
                if not path.exists():
                    raise FileNotFoundError(f"Step15-v6 referenced output is missing: {value}")
                validated_paths.add(path)
        if key[0] in fixed_budget_experiments:
            expected_budget = int(policy["experiments"][key[0]]["fixed_update_budget_per_phase"])
            diagnostics = run["training_diagnostics"]
            if int(diagnostics.get("trained_epoch_count", -1)) != expected_budget:
                raise ValueError(f"Step15-v6 fixed update budget was not honored for {key}")
            if diagnostics.get("training_budget_mode") != "fixed_optimizer_updates_with_best_valid_checkpoint":
                raise ValueError(f"Step15-v6 training budget mode mismatch for {key}")
    m4 = "step15_v6_m4_trusted_positive_mixup"
    m4c = "step15_v6_m4c_matched_continuation_no_mixup"
    pre_treatment_phases = policy["experiments"][m4]["phase_ids"][:-1]
    for phase_id in pre_treatment_phases:
        for seed in expected_seeds:
            left_run = run_index[(m4, phase_id, seed)]
            right_run = run_index[(m4c, phase_id, seed)]
            left_valid = resolve(left_run["output_paths"]["zh_valid_predictions"])
            right_valid = resolve(right_run["output_paths"]["zh_valid_predictions"])
            assert_identical_scores(left_valid, right_valid, f"{phase_id}/seed={seed}/zh_valid")
            left_artifact = resolve(left_run["output_paths"]["artifact"])
            right_artifact = resolve(right_run["output_paths"]["artifact"])
            assert_identical_artifact_params(
                left_artifact, right_artifact, f"{phase_id}/seed={seed}/artifact"
            )
    phase4 = "phase4_add_trusted_positive_mixup"
    for seed in expected_seeds:
        m4_run = run_index[(m4, phase4, seed)]
        m4c_run = run_index[(m4c, phase4, seed)]
        if int(m4_run.get("synthetic_train_only_positive_mixup_count", 0)) <= 0:
            raise ValueError(f"Step15-v6 M4 generated no trusted positive mixup rows for seed {seed}")
        if int(m4c_run.get("synthetic_train_only_positive_mixup_count", -1)) != 0:
            raise ValueError(f"Step15-v6 M4c unexpectedly generated positive mixup rows for seed {seed}")
    candidate_records = selection.get("candidates") or []
    if len(candidate_records) != 2 or any(
        record.get("actual_seeds") != expected_seeds for record in candidate_records
    ):
        raise ValueError("Step15-v6 M5 validation selection lacks both complete ten-seed candidates")
    source_seeds = sorted(int(run["seed"]) for run in source_summary.get("runs", []))
    if source_seeds != expected_seeds:
        raise ValueError(
            f"Step15-v6 source-only run coverage mismatch: expected={expected_seeds} actual={source_seeds}"
        )
    for run in source_summary.get("runs", []):
        for value in (run.get("output_paths") or {}).values():
            if value:
                path = resolve(str(value))
                if not path.exists():
                    raise FileNotFoundError(f"Step15-v6 source-only output is missing: {value}")
                validated_paths.add(path)
    validated_inputs = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(validated_paths, key=lambda value: str(value))
    ]
    validated_manifest_sha = hashlib.sha256(
        json.dumps(
            {"validated_inputs": validated_inputs},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    output_path = resolve(args.output)
    payload = {
        "status": "pass",
        "policy": str(policy_path.relative_to(ROOT)),
        "policy_version": policy["version"],
        "expected_experiment_count": len(expected_experiments),
        "expected_run_count": len(expected_keys),
        "fixed_budget_run_count": sum(key[0] in fixed_budget_experiments for key in expected_keys),
        "m4_m4c_pre_treatment_contract": "exact_valid_predictions_and_artifact_parameters",
        "m4_positive_mixup_seed_count": len(expected_seeds),
        "m4c_zero_mixup_seed_count": len(expected_seeds),
        "m5_complete_candidate_count": len(candidate_records),
        "m5_validation_selected_experiment": selected_m5,
        "m5_nonselected_test_predictions_written": False,
        "source_only_seed_count": len(source_seeds),
        "validated_inputs": validated_inputs,
        "validated_inputs_manifest_sha256": validated_manifest_sha,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
