#!/usr/bin/env python3
"""Resume Step7-v4 CPU selection with a pinned float64 solver correction."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import step7_v4_common as common
import step7_v4_select_source_model as parent_selector
import step7_v4_selection_core as corrected_solver


SCRIPT_PATH = Path(__file__).resolve()
PATCH_POLICY_PATH = (
    common.ROOT / "schema" / "step7_v4_selection_solver_patch_policy.json"
)
PARENT_POLICY_SHA256 = (
    "41193c9957fb80024a0e376c76ebcfe0851fe9fee8c15f597c436824fc3b2327"
)
PARENT_SELECTOR_SHA256 = (
    "d68c9bc14ea32723e6cfb7b7095d47c214f64d7026d7228e681eae07ee9a50c6"
)
PARENT_SOLVER_SHA256 = (
    "950edefad9e0b970e49a3db65ae9fe8a8cb441801a7bc4f2c2a2ed5960e2e86f"
)
SELECTION_OUTPUT_KEYS = [
    "selection_summary",
    "train_selection_lock",
    "model_artifacts",
    "train_oof_predictions",
    "blind_valid_predictions",
    "blind_scoring_lock",
    "valid_predictions",
]


def verify_file_pin(record: dict, role: str) -> dict:
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"Step7-v4 solver-patch file pin drift: {role}")
    path = common.resolve(record["path"])
    observed = common.file_record(path)
    if observed["sha256"] != record["sha256"]:
        raise ValueError(
            f"Step7-v4 solver-patch file drift: role={role} "
            f"expected={record['sha256']} observed={observed['sha256']}"
        )
    return observed


def load_patch_policy() -> dict:
    patch = common.load_json(PATCH_POLICY_PATH)
    if set(patch) != {
        "version",
        "scope",
        "parent_contract",
        "implementation",
        "solver_fix",
        "outputs",
    }:
        raise ValueError("Step7-v4 solver-patch policy schema drift")
    if (
        patch["version"]
        != "2026-07-24-step7-v4-selection-float64-stationarity-fix-v1"
        or patch["scope"]
        != "selection_only_reuse_verified_frozen_v2_gpu_features"
    ):
        raise ValueError("Step7-v4 solver-patch policy identity drift")

    parent = patch["parent_contract"]
    if set(parent) != {
        "policy",
        "selector",
        "selection_solver",
        "gpu_outputs_verified_by_unchanged_parent_selector",
        "gpu_reencoding_required",
        "training_or_validation_labels_changed",
    }:
        raise ValueError("Step7-v4 solver-patch parent contract drift")
    expected_parent = {
        "policy": (
            str(common.DEFAULT_POLICY.relative_to(common.ROOT)).replace(
                "\\", "/"
            ),
            PARENT_POLICY_SHA256,
        ),
        "selector": (
            "scripts/step7_v4_select_source_model.py",
            PARENT_SELECTOR_SHA256,
        ),
        "selection_solver": (
            "scripts/step7_v3_1_selection_core.py",
            PARENT_SOLVER_SHA256,
        ),
    }
    for role, (path, digest) in expected_parent.items():
        if parent.get(role) != {"path": path, "sha256": digest}:
            raise ValueError(
                f"Step7-v4 solver-patch parent pin drift: {role}"
            )
        verify_file_pin(parent[role], f"parent_{role}")
    if (
        parent["gpu_outputs_verified_by_unchanged_parent_selector"] is not True
        or parent["gpu_reencoding_required"] is not False
        or parent["training_or_validation_labels_changed"] is not False
    ):
        raise ValueError("Step7-v4 solver-patch scope was broadened")

    implementation = patch["implementation"]
    if set(implementation) != {"resume_selector", "corrected_solver"}:
        raise ValueError("Step7-v4 solver-patch implementation universe drift")
    for role, record in implementation.items():
        verify_file_pin(record, role)

    fix = patch["solver_fix"]
    if (
        set(fix)
        != {
            "primary_convergence_criterion",
            "fallback_convergence_criterion",
            "requested_normalized_gradient_tolerance",
            "float64_small_gradient_ceiling",
            "objective_resolution_multiplier",
            "parameter_resolution_multiplier",
            "armijo_unresolvable_full_newton_step_requires_strict_gradient_improvement",
            "genuine_nonconvergence_remains_fatal",
            "l2_grid_or_selection_metric_changed",
        }
        or fix["primary_convergence_criterion"]
        != "normalized_gradient_inf_norm_at_most_requested_tolerance"
        or fix["fallback_convergence_criterion"]
        != (
            "small_gradient_and_newton_objective_or_parameter_change_"
            "below_float64_resolution"
        )
        or float(fix["requested_normalized_gradient_tolerance"]) != 1e-9
        or float(fix["float64_small_gradient_ceiling"])
        != corrected_solver.FLOAT64_SQRT_EPSILON
        or float(fix["objective_resolution_multiplier"])
        != corrected_solver.OBJECTIVE_RESOLUTION_MULTIPLIER
        or float(fix["parameter_resolution_multiplier"])
        != corrected_solver.PARAMETER_RESOLUTION_MULTIPLIER
        or fix[
            "armijo_unresolvable_full_newton_step_requires_strict_gradient_improvement"
        ]
        is not True
        or fix["genuine_nonconvergence_remains_fatal"] is not True
        or fix["l2_grid_or_selection_metric_changed"] is not False
    ):
        raise ValueError("Step7-v4 numerical solver-patch contract drift")

    outputs = patch["outputs"]
    if set(outputs) != {"selection_root", *SELECTION_OUTPUT_KEYS, "patch_manifest"}:
        raise ValueError("Step7-v4 solver-patch output universe drift")
    root = str(outputs["selection_root"]).rstrip("/")
    expected_root = (
        "reports/step7_v4_raw_item_authorship_selection/v2_20260723/"
        "selection_float64_fix_v1_20260724"
    )
    if root != expected_root:
        raise ValueError("Step7-v4 solver-patch output root drift")
    for role, path in outputs.items():
        if role != "selection_root" and not str(path).startswith(root + "/"):
            raise ValueError(
                f"Step7-v4 solver-patch output escapes its root: {role}"
            )
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("Step7-v4 solver-patch output paths collide")
    return patch


def execution_policy(parent_policy: dict, patch: dict) -> dict:
    result = copy.deepcopy(parent_policy)
    result["training"]["solver"] = (
        "newton_with_armijo_backtracking_and_float64_stationarity_certificate"
    )
    for role in SELECTION_OUTPUT_KEYS:
        result["outputs"][role] = patch["outputs"][role]
    return result


def empty_solver_execution_audit() -> dict:
    return {
        "fit_count": 0,
        "convergence_criterion_counts": {},
        "float64_objective_resolution_step_fit_count": 0,
        "float64_objective_resolution_step_total": 0,
        "float64_stationarity_fallback_fit_count": 0,
        "maximum_final_normalized_gradient_inf_norm": 0.0,
        "maximum_solver_iterations": 0,
    }


def install_verified_patch(
    parent_policy: dict, solver_execution_audit: dict
) -> None:
    original_verify_gpu_outputs = parent_selector.verify_gpu_outputs

    def verify_parent_gpu_outputs(
        _execution_policy: dict,
        preparation_manifest: dict,
        preparation_bundle: dict,
    ):
        return original_verify_gpu_outputs(
            parent_policy, preparation_manifest, preparation_bundle
        )

    def audited_fit_logistic(*args, **kwargs):
        artifact = corrected_solver.fit_logistic(*args, **kwargs)
        criterion = artifact["solver_convergence_criterion"]
        counts = solver_execution_audit["convergence_criterion_counts"]
        counts[criterion] = int(counts.get(criterion, 0)) + 1
        resolution_steps = int(
            artifact["solver_float64_objective_resolution_step_count"]
        )
        solver_execution_audit["fit_count"] += 1
        solver_execution_audit[
            "float64_objective_resolution_step_fit_count"
        ] += int(resolution_steps > 0)
        solver_execution_audit[
            "float64_objective_resolution_step_total"
        ] += resolution_steps
        solver_execution_audit[
            "float64_stationarity_fallback_fit_count"
        ] += int(
            artifact["solver_used_float64_stationarity_fallback"]
        )
        solver_execution_audit[
            "maximum_final_normalized_gradient_inf_norm"
        ] = max(
            float(
                solver_execution_audit[
                    "maximum_final_normalized_gradient_inf_norm"
                ]
            ),
            float(
                artifact[
                    "solver_final_normalized_gradient_inf_norm"
                ]
            ),
        )
        solver_execution_audit["maximum_solver_iterations"] = max(
            int(solver_execution_audit["maximum_solver_iterations"]),
            int(artifact["solver_iterations"]),
        )
        return artifact

    parent_selector.verify_gpu_outputs = verify_parent_gpu_outputs
    parent_selector.solver.fit_logistic = audited_fit_logistic
    parent_selector.SELECTOR_SCRIPT = SCRIPT_PATH


def write_patch_manifest(
    patch: dict,
    run_policy: dict,
    summary: dict,
    solver_execution_audit: dict,
) -> dict:
    output_records = {
        role: common.file_record(common.resolve(patch["outputs"][role]))
        for role in SELECTION_OUTPUT_KEYS
    }
    parent_outputs = common.load_policy()["outputs"]
    payload = {
        "step": "step7_v4_selection_float64_solver_patch",
        "version": patch["version"],
        "scope": patch["scope"],
        "parent_contract": patch["parent_contract"],
        "patch_policy": common.file_record(PATCH_POLICY_PATH),
        "implementation": {
            role: common.file_record(common.resolve(record["path"]))
            for role, record in patch["implementation"].items()
        },
        "execution_policy_contract_sha256": common.canonical_hash(run_policy),
        "parent_gpu_output_manifest": common.file_record(
            common.resolve(parent_outputs["gpu_output_manifest"])
        ),
        "parent_gpu_sync_manifest": common.file_record(
            common.resolve(parent_outputs["gpu_sync_manifest"])
        ),
        "gpu_reencoding_performed": False,
        "gpu_features_reverified_with_unchanged_parent_selector": True,
        "solver_fix": patch["solver_fix"],
        "solver_execution_audit": {
            **solver_execution_audit,
            "convergence_criterion_counts": dict(
                sorted(
                    solver_execution_audit[
                        "convergence_criterion_counts"
                    ].items()
                )
            ),
        },
        "selection_status": summary["selection_decision"][
            "selection_status"
        ],
        "winner": summary["selection_decision"]["winner"],
        "all_formal_fits_converged": summary["nested_training_audit"][
            "all_formal_fits_converged"
        ],
        "valid_loaded_after_selection_lock": summary[
            "valid_loaded_after_selection_lock"
        ],
        "valid_loaded_after_blind_scoring_lock": summary[
            "valid_loaded_after_blind_scoring_lock"
        ],
        "historical_test_labels_read": summary[
            "historical_test_labels_read"
        ],
        "outputs": output_records,
    }
    payload["manifest_content_sha256"] = common.canonical_hash(payload)
    path = common.resolve(patch["outputs"]["patch_manifest"])
    common.write_json_immutable(path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    patch = load_patch_policy()
    parent_policy = common.load_policy()
    common.verify_implementation_files(parent_policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "gpu_reencoding_required": False,
                    "selection_root": patch["outputs"]["selection_root"],
                    "numerical_execution_performed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    run_policy = execution_policy(parent_policy, patch)
    solver_execution_audit = empty_solver_execution_audit()
    install_verified_patch(parent_policy, solver_execution_audit)
    summary = parent_selector.run_selection(run_policy)
    patch_manifest = write_patch_manifest(
        patch, run_policy, summary, solver_execution_audit
    )
    print(
        json.dumps(
            {
                "status": summary["selection_decision"][
                    "selection_status"
                ],
                "winner": summary["selection_decision"]["winner"],
                "gpu_reencoding_performed": False,
                "patch_manifest_content_sha256": patch_manifest[
                    "manifest_content_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
