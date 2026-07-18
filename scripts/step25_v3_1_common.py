#!/usr/bin/env python3
"""Step25-v3.1 policy overlay and fail-closed constrained LR/L2 solver."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

import step25_v3_common as v3
from step25_v3_common import *  # noqa: F401,F403


DEFAULT_POLICY = ROOT / "schema" / "step25_v3_1_solver_convergence_policy.json"


def load_policy(
    path: str | Path = DEFAULT_POLICY,
) -> tuple[Path, dict, dict, dict, dict, dict]:
    overlay_path = resolve(path)
    overlay = load_json(overlay_path)
    base_path = resolve(overlay["base_policy"])
    (
        _base_path,
        base,
        step15_v7_policy,
        step24_policy,
        step25_v1_policy,
        step25_v2_policy,
    ) = v3.load_policy(base_path)
    policy = copy.deepcopy(base)
    policy["version"] = overlay["version"]
    policy["objective"] = overlay["objective"]
    policy["outputs_root"] = overlay["outputs_root"]
    policy["outputs"] = copy.deepcopy(overlay["outputs"])
    repair = overlay["solver_repair"]
    logistic = policy["evaluation"]["logistic"]
    logistic.update(
        {
            "solver": repair["solver"],
            "max_iter": int(repair["maximum_iterations"]),
            "tolerance": float(repair["projected_gradient_tolerance"]),
            "projected_gradient_tolerance": float(
                repair["projected_gradient_tolerance"]
            ),
            "relative_loss_diagnostic_tolerance": float(
                repair["relative_loss_diagnostic_tolerance"]
            ),
            "relative_loss_is_a_convergence_criterion": False,
            "minimum_step_size": float(repair["minimum_step_size"]),
            "backtracking_factor": float(repair["backtracking_factor"]),
            "armijo_constant": float(repair["armijo_constant"]),
            "non_kkt_stagnation_action": repair["non_kkt_stagnation_action"],
        }
    )
    validate_solver_repair(overlay, policy, base)
    return (
        overlay_path,
        policy,
        step15_v7_policy,
        step24_policy,
        step25_v1_policy,
        step25_v2_policy,
    )


def validate_solver_repair(overlay: dict, policy: dict, base: dict) -> None:
    frozen = overlay["scientific_freeze"]
    if not all(frozen.values()):
        raise ValueError("Step25-v3.1 scientific freeze contains a relaxed constraint")
    if policy["outputs_root"] in {
        base["outputs_root"],
        base["inputs"]["step25_v1_outputs_root"],
        base["inputs"]["step25_v2_outputs_root"],
    }:
        raise ValueError("Step25-v3.1 output root is not isolated")
    for key in (
        "feature_contract",
        "missingness_closure_control",
        "operational_identifier_control",
        "d0_to_d1_replication_gates",
        "boundary",
        "immutable_parent_contract",
    ):
        if policy[key] != base[key]:
            raise ValueError(f"Step25-v3.1 changed a frozen contract: {key}")
    for key in (
        "canonical_split",
        "fold_count",
        "fold_seed",
        "model_specs",
        "primary_model",
        "matched_baseline_model",
        "grouped_bootstrap_resamples",
        "grouped_bootstrap_seed",
    ):
        if policy["evaluation"][key] != base["evaluation"][key]:
            raise ValueError(f"Step25-v3.1 changed frozen evaluation setting: {key}")
    base_logistic = base["evaluation"]["logistic"]
    logistic = policy["evaluation"]["logistic"]
    for key in (
        "l2_penalty",
        "class_weight",
        "standardize_features",
        "sample_weight_total_normalization",
        "backtracking_factor",
        "armijo_constant",
        "minimum_step_size",
        "max_iter",
    ):
        if logistic[key] != base_logistic[key]:
            raise ValueError(f"Step25-v3.1 changed frozen optimizer setting: {key}")
    repair = overlay["solver_repair"]
    if repair["relative_loss_is_a_convergence_criterion"] is not False:
        raise ValueError("Step25-v3.1 cannot converge from relative loss alone")
    if repair["non_kkt_stagnation_action"] != "fail_closed":
        raise ValueError("Step25-v3.1 numerical stagnation must fail closed")
    if float(repair["projected_gradient_tolerance"]) != float(
        base_logistic["tolerance"]
    ):
        raise ValueError("Step25-v3.1 changed the preregistered KKT tolerance")
    if len(policy["outputs"]) != len(set(policy["outputs"].values())):
        raise ValueError("Step25-v3.1 output map contains a collision")


def _objective_gradient(
    params: np.ndarray,
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    l2_penalty: float,
) -> tuple[float, np.ndarray]:
    return v3._objective_gradient(params, matrix, labels, weights, l2_penalty)


def _projected_gradient_residual(
    params: np.ndarray,
    gradient: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    projected = params - gradient
    projected[1:] = np.minimum(np.maximum(projected[1:], lower), upper)
    return float(np.max(np.abs(params - projected)))


def _objective_hessian(
    params: np.ndarray,
    matrix: np.ndarray,
    weights: np.ndarray,
    l2_penalty: float,
) -> np.ndarray:
    logits = params[0] + matrix @ params[1:]
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    curvature = weights * probabilities * (1.0 - probabilities)
    design = np.column_stack((np.ones(len(matrix), dtype=np.float64), matrix))
    hessian = design.T @ (curvature[:, None] * design)
    hessian[1:, 1:] += l2_penalty * np.eye(matrix.shape[1], dtype=np.float64)
    return hessian


def _active_set_newton_direction(
    params: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, bool]:
    free = np.ones(len(params), dtype=bool)
    boundary_tolerance = 1e-12
    for index in range(1, len(params)):
        coefficient = params[index]
        coefficient_gradient = gradient[index]
        lower_bound = lower[index - 1]
        upper_bound = upper[index - 1]
        at_lower_kkt = (
            np.isfinite(lower_bound)
            and coefficient <= lower_bound + boundary_tolerance
            and coefficient_gradient >= 0.0
        )
        at_upper_kkt = (
            np.isfinite(upper_bound)
            and coefficient >= upper_bound - boundary_tolerance
            and coefficient_gradient <= 0.0
        )
        if at_lower_kkt or at_upper_kkt:
            free[index] = False
    direction = np.zeros_like(params)
    free_indices = np.flatnonzero(free)
    if not len(free_indices):
        return direction, False
    free_hessian = hessian[np.ix_(free_indices, free_indices)]
    free_gradient = gradient[free_indices]
    try:
        direction[free_indices] = np.linalg.solve(free_hessian, -free_gradient)
    except np.linalg.LinAlgError:
        direction[free_indices] = np.linalg.lstsq(
            free_hessian, -free_gradient, rcond=None
        )[0]
    used_projected_gradient_fallback = False
    if not np.all(np.isfinite(direction)) or float(gradient @ direction) >= -1e-14:
        projected = params - gradient
        projected[1:] = np.minimum(np.maximum(projected[1:], lower), upper)
        direction = projected - params
        used_projected_gradient_fallback = True
    return direction, used_projected_gradient_fallback


def fit_direction_constrained_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    feature_names: list[str],
    directions: dict[str, str],
    cfg: dict,
) -> dict:
    matrix = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(labels):
        raise ValueError("Step25-v3.1 constrained logistic shape mismatch")
    if len(feature_names) != matrix.shape[1] or set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("Step25-v3.1 constrained logistic requires aligned two-class data")
    if np.any(sample_weights <= 0.0) or not np.all(np.isfinite(sample_weights)):
        raise ValueError("Step25-v3.1 constrained logistic weights must be finite and positive")
    if cfg.get("relative_loss_is_a_convergence_criterion") is not False:
        raise ValueError("Step25-v3.1 relative loss cannot terminate optimization")

    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    scaled = (matrix - means) / scales
    lower, upper = direction_bounds(feature_names, directions)
    params = np.zeros(matrix.shape[1] + 1, dtype=np.float64)
    prevalence = float(np.sum(sample_weights * labels) / np.sum(sample_weights))
    params[0] = math.log(max(prevalence, 1e-6) / max(1.0 - prevalence, 1e-6))

    tolerance = float(cfg["projected_gradient_tolerance"])
    relative_tolerance = float(cfg["relative_loss_diagnostic_tolerance"])
    converged = False
    termination_reason = "maximum_iterations_with_unmet_kkt"
    final_step = 0.0
    final_relative_loss = math.inf
    small_relative_loss_iterations = 0
    projected_gradient_fallback_count = 0
    iteration = 0
    for iteration in range(1, int(cfg["max_iter"]) + 1):
        loss, gradient = _objective_gradient(
            params, scaled, labels, sample_weights, float(cfg["l2_penalty"])
        )
        current_residual = _projected_gradient_residual(
            params, gradient, lower, upper
        )
        if current_residual <= tolerance:
            converged = True
            termination_reason = "projected_gradient_kkt_tolerance"
            break

        hessian = _objective_hessian(
            params, scaled, sample_weights, float(cfg["l2_penalty"])
        )
        direction, used_fallback = _active_set_newton_direction(
            params, gradient, hessian, lower, upper
        )
        projected_gradient_fallback_count += int(used_fallback)
        accepted = False
        trial_step = 1.0
        candidate = params.copy()
        candidate_loss = loss
        while trial_step >= float(cfg["minimum_step_size"]):
            candidate = params + trial_step * direction
            candidate[1:] = np.minimum(np.maximum(candidate[1:], lower), upper)
            displacement = candidate - params
            if not np.any(displacement):
                trial_step *= float(cfg["backtracking_factor"])
                continue
            candidate_loss, _ = _objective_gradient(
                candidate, scaled, labels, sample_weights, float(cfg["l2_penalty"])
            )
            armijo = loss + float(cfg["armijo_constant"]) * float(
                gradient @ displacement
            )
            if candidate_loss <= armijo + 1e-12:
                accepted = True
                break
            trial_step *= float(cfg["backtracking_factor"])
        if not accepted:
            termination_reason = "line_search_stagnation_with_unmet_kkt"
            break

        params = candidate
        final_step = trial_step
        final_relative_loss = abs(loss - candidate_loss) / max(1.0, abs(loss))
        if final_relative_loss <= relative_tolerance:
            small_relative_loss_iterations += 1

    final_objective, final_gradient = _objective_gradient(
        params, scaled, labels, sample_weights, float(cfg["l2_penalty"])
    )
    final_projected_gradient = _projected_gradient_residual(
        params, final_gradient, lower, upper
    )
    if final_projected_gradient <= tolerance:
        converged = True
        termination_reason = "projected_gradient_kkt_tolerance"
    elif converged:
        raise ValueError("Step25-v3.1 convergence state disagrees with final KKT residual")

    raw_coefficients = params[1:] / scales
    raw_intercept = params[0] - float(means @ raw_coefficients)
    for coefficient, name in zip(raw_coefficients, feature_names, strict=True):
        direction = directions[name]
        if direction == "nonnegative" and coefficient < -1e-10:
            raise ValueError(f"Step25-v3.1 violated nonnegative direction: {name}")
        if direction == "nonpositive" and coefficient > 1e-10:
            raise ValueError(f"Step25-v3.1 violated nonpositive direction: {name}")

    return {
        "model_family": "step25_v3_1_direction_constrained_logistic_l2",
        "solver_convergence_contract": "projected_gradient_kkt_only_fail_closed",
        "feature_names": feature_names,
        "coefficient_directions": directions,
        "standardization": {"means": means.tolist(), "scales": scales.tolist()},
        "parameter_intercept_standardized": float(params[0]),
        "parameter_coefficients_standardized": params[1:].tolist(),
        "parameter_intercept_raw": raw_intercept,
        "parameter_coefficients_raw": raw_coefficients.tolist(),
        "solver_iterations": iteration,
        "solver_converged": converged,
        "solver_termination_reason": termination_reason,
        "solver_final_objective": float(final_objective),
        "solver_final_gradient_infinity_norm": float(
            np.max(np.abs(final_gradient))
        ),
        "solver_final_projected_gradient": final_projected_gradient,
        "solver_final_relative_loss": float(final_relative_loss),
        "solver_final_accepted_step_size": float(final_step),
        "solver_small_relative_loss_iterations": small_relative_loss_iterations,
        "solver_projected_gradient_fallback_count": projected_gradient_fallback_count,
        "projected_gradient_tolerance": tolerance,
        "relative_loss_used_for_convergence": False,
        "l2_penalty": float(cfg["l2_penalty"]),
    }
