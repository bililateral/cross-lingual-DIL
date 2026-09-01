#!/usr/bin/env python3
"""Fit label-budget matched CPU controls before direct LaBSE fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_labse_finetune_common_v1 as fine_common
import step28_v13_v1_13_v9_4_1_model_training_core_v3 as core
import step28_v13_v1_13_v9_4_1_transfer_claim_controls_v4 as controls


OUTPUT_ROOT = (
    ROOT
    / "reports"
    / "step28_model_experiment"
    / "v9_4_1_same_budget_cpu_controls_v1_20260901"
)
MODEL_IDS = (
    "i0_identity_only",
    "t1_m0_plus_identity",
    "m3_base_fixed",
    "m3_joint_fixed",
)


class SameBudgetControlError(ValueError):
    """Raised when a same-budget CPU control violates its contract."""


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(value, dtype="<f8"), allow_pickle=False)


def _fixed_lightgbm(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    development_matrix: np.ndarray,
    grid: list[float | int],
) -> np.ndarray:
    medians = core.common_v1.finite_median_impute_fit(train_matrix)
    fitted_train = core.common_v1.impute_with_medians(train_matrix, medians)
    fitted_development = core.common_v1.impute_with_medians(
        development_matrix, medians
    )
    model = core._lightgbm_classifier(grid)
    model.fit(fitted_train, train_labels)
    result = np.ascontiguousarray(
        model.predict_proba(fitted_development)[:, 1], dtype="<f8"
    )
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise SameBudgetControlError("Fixed LightGBM produced invalid probabilities")
    return result


def _mask_for_worlds(
    row_worlds: list[str], selected_worlds: tuple[str, ...]
) -> np.ndarray:
    selected = set(selected_worlds)
    mask = np.fromiter(
        (world_uid in selected for world_uid in row_worlds),
        dtype=bool,
        count=len(row_worlds),
    )
    if int(np.sum(mask)) != len(selected_worlds) * 378:
        raise SameBudgetControlError("Budget row count does not equal worlds times 378")
    return mask


def validate_contract() -> dict[str, Any]:
    policy = fine_common.load_policy()
    controls.load_policy()
    return {
        "status": "PASSED_SAME_BUDGET_CPU_CONTROL_CONTRACT_NO_TRUTH_READ",
        "fine_tune_policy_canonical_self_hash": policy["canonical_self_hash"],
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }


def run() -> dict[str, Any]:
    policy = fine_common.load_policy()
    (
        _execution,
        _v3_policy,
        train,
        development,
        train_labels,
        development_labels,
        relevance,
    ) = controls._load_inputs(policy)
    if OUTPUT_ROOT.exists():
        raise SameBudgetControlError("Same-budget output already exists")
    building = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    try:
        world_order = fine_common.nested_world_subsets(
            sorted(set(train["world_uids"]), key=lambda value: value.encode("utf-8")),
            policy["training_budgets"]["world_counts"],
        )
        cpu = policy["same_budget_cpu_controls"]
        results: dict[str, Any] = {}
        for world_count in policy["training_budgets"]["world_counts"]:
            selected_worlds = world_order[int(world_count)]
            mask = _mask_for_worlds(train["world_uids"], selected_worlds)
            selected_identity = np.ascontiguousarray(
                train["identity33"][mask], dtype="<f8"
            )
            selected_labels = np.ascontiguousarray(train_labels[mask], dtype=np.int8)
            selected_world_rows = [
                world_uid
                for world_uid, keep in zip(train["world_uids"], mask)
                if keep
            ]
            scale, mu = core.common_v1.fit_identity_transform(
                selected_identity, selected_world_rows
            )
            train_phi, _ = core.common_v1.apply_identity_transform(
                selected_identity, scale, mu
            )
            development_phi, _ = core.common_v1.apply_identity_transform(
                development["identity33"], scale, mu
            )
            l2_i0 = float(cpu["i0_identity_only_l2"])
            l2_t1 = float(cpu["t1_m0_plus_identity_l2"])
            train_offset = controls.common_v2.raw_logit(train["m0_probability"][mask])
            development_offset = controls.common_v2.raw_logit(
                development["m0_probability"]
            )
            i0 = controls.fit_control(
                train_phi, selected_labels, l2_i0, offset=None
            )
            t1 = controls.fit_control(
                train_phi, selected_labels, l2_t1, offset=train_offset
            )
            predictions = {
                "i0_identity_only": controls.predict_control(
                    i0, development_phi, offset=None
                ),
                "t1_m0_plus_identity": controls.predict_control(
                    t1, development_phi, offset=development_offset
                ),
                "m3_base_fixed": _fixed_lightgbm(
                    train["base24"][mask],
                    selected_labels,
                    development["base24"],
                    list(cpu["m3_base_fixed_grid"]),
                ),
                "m3_joint_fixed": _fixed_lightgbm(
                    np.column_stack(
                        (train["base24"][mask], train["identity33"][mask])
                    ),
                    selected_labels,
                    np.column_stack(
                        (development["base24"], development["identity33"])
                    ),
                    list(cpu["m3_joint_fixed_grid"]),
                ),
            }
            if set(predictions) != set(MODEL_IDS):
                raise SameBudgetControlError("Same-budget prediction registry drift")
            model_reports: dict[str, Any] = {}
            thresholds: dict[str, float] = {}
            for model_id in MODEL_IDS:
                threshold, report = controls._evaluate_model(
                    predictions[model_id],
                    development_labels,
                    development,
                    relevance,
                )
                thresholds[model_id] = float(threshold)
                model_reports[model_id] = report
                save_array(
                    building
                    / "predictions"
                    / f"worlds_{int(world_count):03d}"
                    / f"{model_id}.npy",
                    predictions[model_id],
                )
            results[str(world_count)] = {
                "world_count": int(world_count),
                "pair_count": int(np.sum(mask)),
                "positive_pair_count": int(np.sum(selected_labels)),
                "world_selection_sha256": hashlib.sha256(
                    controls.canonical_json_bytes(list(selected_worlds))
                ).hexdigest(),
                "models": model_reports,
                "thresholds": thresholds,
                "ap_t1_minus_i0": float(
                    model_reports["t1_m0_plus_identity"]["pooled"][
                        "average_precision"
                    ]
                    - model_reports["i0_identity_only"]["pooled"][
                        "average_precision"
                    ]
                ),
                "fit": {
                    "i0_intercept": float(i0["theta"][0]),
                    "t1_intercept": float(t1["theta"][0]),
                    "i0_gradient_infinity_norm": float(
                        i0["gradient_infinity_norm"]
                    ),
                    "t1_gradient_infinity_norm": float(
                        t1["gradient_infinity_norm"]
                    ),
                },
            }
        write_json(
            building / "same_budget_evaluation.json",
            {
                "status": "SAME_BUDGET_CPU_CONTROLS_COMPLETE_AUDIT_TRUTH_SEALED",
                "fine_tune_policy_canonical_self_hash": policy[
                    "canonical_self_hash"
                ],
                "budgets": results,
                "truth_read_counts": {
                    "train_labels": 1,
                    "development_labels": 1,
                    "development_qrels": 1,
                    "audit_a_labels_or_qrels": 0,
                    "audit_b_labels_or_qrels": 0,
                },
            },
        )
        files = [
            controls.file_record(path, building)
            for path in sorted(building.rglob("*"), key=lambda item: item.as_posix())
            if path.is_file()
        ]
        write_json(
            building / "manifest.json",
            {
                "status": "SAME_BUDGET_CPU_CONTROLS_COMPLETE_AUDIT_TRUTH_SEALED",
                "producer": {
                    "path": Path(__file__).relative_to(ROOT).as_posix(),
                    "sha256": controls.sha256_file(Path(__file__)),
                },
                "files": files,
                "audit_a_truth_reads": 0,
                "audit_b_truth_reads": 0,
            },
        )
        building.replace(OUTPUT_ROOT)
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise
    compact = {
        count: {
            model_id: float(record["models"][model_id]["pooled"]["average_precision"])
            for model_id in MODEL_IDS
        }
        | {"ap_t1_minus_i0": float(record["ap_t1_minus_i0"])}
        for count, record in results.items()
    }
    return {
        "status": "SAME_BUDGET_CPU_CONTROLS_COMPLETE_AUDIT_TRUTH_SEALED",
        "output_root": OUTPUT_ROOT.relative_to(ROOT).as_posix(),
        "average_precision_by_budget": compact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-contract", "run"))
    args = parser.parse_args()
    result = validate_contract() if args.command == "validate-contract" else run()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
