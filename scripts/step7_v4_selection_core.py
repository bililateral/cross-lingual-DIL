#!/usr/bin/env python3
"""Step7-v4 numerical helpers with float64-aware convergence certification."""

from __future__ import annotations

import math

import numpy as np

import step7_v3_1_selection_core as parent


FLOAT64_EPSILON = float(np.finfo(np.float64).eps)
FLOAT64_SQRT_EPSILON = math.sqrt(FLOAT64_EPSILON)
OBJECTIVE_RESOLUTION_MULTIPLIER = 64.0
PARAMETER_RESOLUTION_MULTIPLIER = 64.0


def _normalized_gradient_inf_norm(
    gradient: np.ndarray, weight_total: float
) -> float:
    return float(np.max(np.abs(gradient)) / weight_total)


def _objective_resolution(objective: float, weight_total: float) -> float:
    return float(
        OBJECTIVE_RESOLUTION_MULTIPLIER
        * FLOAT64_EPSILON
        * max(1.0, abs(float(objective)), float(weight_total))
    )


def _parameter_resolution(
    current: np.ndarray, proposed: np.ndarray
) -> float:
    return float(
        PARAMETER_RESOLUTION_MULTIPLIER
        * FLOAT64_EPSILON
        * max(
            1.0,
            float(np.linalg.norm(current)),
            float(np.linalg.norm(proposed)),
        )
    )


def _small_gradient_for_float64(
    normalized_gradient: float, requested_tolerance: float
) -> bool:
    return normalized_gradient <= max(
        float(requested_tolerance), FLOAT64_SQRT_EPSILON
    )


def fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    l2_penalty: float,
    max_iter: int,
    tolerance: float,
    armijo_c1: float,
    minimum_line_search_step: float,
) -> dict:
    """Fit deterministic L2 logistic regression without false float64 failures.

    The primary certificate remains the requested normalized-gradient
    tolerance. Near the optimum, however, an Armijo comparison can become
    unresolvable in float64 before that strict threshold is crossed. In that
    case the full Newton step is taken only if the recomputed analytic
    gradient strictly improves. A fit is accepted by the fallback certificate
    only when both conditions hold:

    * the normalized gradient is no larger than sqrt(float64 epsilon); and
    * either the Newton-predicted objective reduction is below the objective's
      float64 resolution, or the accepted parameter/objective changes are both
      below their float64 resolutions.

    This is a numerical-stationarity certificate, not a relaxed iteration or
    arbitrary decimal tolerance.
    """

    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    l2 = float(l2_penalty)
    requested_tolerance = float(tolerance)
    if x.ndim != 2 or len(x) != len(y) or len(w) != len(y):
        raise ValueError("Step7-v4 logistic input shape mismatch")
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(w))
    ):
        raise ValueError("Step7-v4 logistic input contains non-finite values")
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Step7-v4 logistic training requires both binary classes")
    if np.any(w <= 0.0):
        raise ValueError("Step7-v4 logistic weights must be positive")
    if (
        not math.isfinite(l2)
        or l2 < 0.0
        or int(max_iter) <= 0
        or not math.isfinite(requested_tolerance)
        or requested_tolerance <= 0.0
        or not 0.0 < float(armijo_c1) < 1.0
        or not 0.0 < float(minimum_line_search_step) <= 1.0
    ):
        raise ValueError("Step7-v4 logistic solver configuration is invalid")

    weight_total = float(np.sum(w))
    mean = np.sum(x * w[:, None], axis=0) / weight_total
    variance = np.sum(((x - mean) ** 2) * w[:, None], axis=0) / weight_total
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    params = np.zeros(z.shape[1] + 1, dtype=np.float64)

    def state(current: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        logits = current[0] + z @ current[1:]
        probabilities = parent.safe_sigmoid(logits)
        objective = float(
            np.sum(w * (np.logaddexp(0.0, logits) - y * logits))
            + 0.5 * l2 * np.dot(current[1:], current[1:])
        )
        residual = (probabilities - y) * w
        gradient = np.empty_like(current)
        gradient[0] = np.sum(residual)
        gradient[1:] = z.T @ residual + l2 * current[1:]
        curvature = probabilities * (1.0 - probabilities) * w
        if (
            not math.isfinite(objective)
            or not np.all(np.isfinite(gradient))
            or not np.all(np.isfinite(curvature))
        ):
            raise ValueError("Step7-v4 logistic numerical state is non-finite")
        return objective, gradient, curvature

    converged = False
    convergence_criterion = ""
    final_delta = math.inf
    final_normalized_gradient = math.inf
    final_objective = math.inf
    final_objective_change = math.inf
    final_predicted_decrease = math.inf
    final_objective_resolution = math.inf
    final_parameter_resolution = math.inf
    used_float64_stationarity = False
    float64_objective_resolution_step_count = 0

    for iteration in range(1, int(max_iter) + 1):
        objective, gradient, curvature = state(params)
        normalized_gradient = _normalized_gradient_inf_norm(
            gradient, weight_total
        )
        objective_resolution = _objective_resolution(objective, weight_total)
        if normalized_gradient <= requested_tolerance:
            converged = True
            convergence_criterion = (
                "normalized_gradient_inf_norm_at_most_requested_tolerance"
            )
            final_delta = 0.0
            final_normalized_gradient = normalized_gradient
            final_objective = objective
            final_objective_change = 0.0
            final_predicted_decrease = 0.0
            final_objective_resolution = objective_resolution
            final_parameter_resolution = _parameter_resolution(params, params)
            break

        weighted_z = z * curvature[:, None]
        hessian = np.empty((len(params), len(params)), dtype=np.float64)
        hessian[0, 0] = np.sum(curvature)
        hessian[0, 1:] = np.sum(weighted_z, axis=0)
        hessian[1:, 0] = hessian[0, 1:]
        hessian[1:, 1:] = z.T @ weighted_z
        hessian[1:, 1:] += np.eye(z.shape[1]) * l2
        try:
            newton_delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            newton_delta = np.linalg.pinv(hessian) @ gradient
        direction = -newton_delta
        directional_derivative = float(np.dot(gradient, direction))
        newton_direction = bool(
            np.all(np.isfinite(direction)) and directional_derivative < 0.0
        )
        if not newton_direction:
            direction = -gradient / max(float(np.linalg.norm(gradient)), 1.0)
            directional_derivative = float(np.dot(gradient, direction))
            if (
                not np.all(np.isfinite(direction))
                or not math.isfinite(directional_derivative)
                or directional_derivative >= 0.0
            ):
                raise ValueError(
                    "Step7-v4 logistic cannot construct a finite descent direction"
                )

        predicted_decrease = (
            0.5 * max(0.0, -directional_derivative)
            if newton_direction
            else math.inf
        )

        step_size = 1.0
        accepted = False
        proposed = params
        proposed_objective = objective
        while step_size >= float(minimum_line_search_step):
            candidate = params + step_size * direction
            candidate_objective, candidate_gradient, _candidate_curvature = (
                state(candidate)
            )
            candidate_normalized_gradient = _normalized_gradient_inf_norm(
                candidate_gradient, weight_total
            )
            if candidate_objective <= (
                objective
                + float(armijo_c1) * step_size * directional_derivative
            ):
                proposed = candidate
                proposed_objective = candidate_objective
                accepted = True
                break
            if (
                step_size == 1.0
                and newton_direction
                and abs(candidate_objective - objective)
                <= objective_resolution
                and candidate_normalized_gradient < normalized_gradient
            ):
                # Armijo's right-hand side can round below an objective value
                # whose true Newton improvement is smaller than one float64
                # objective unit. Taking the full Newton step is still
                # justified when the independently recomputed analytic
                # gradient strictly improves.
                proposed = candidate
                proposed_objective = candidate_objective
                accepted = True
                float64_objective_resolution_step_count += 1
                break
            step_size *= 0.5
        if not accepted:
            if (
                newton_direction
                and _small_gradient_for_float64(
                    normalized_gradient, requested_tolerance
                )
                and predicted_decrease <= objective_resolution
            ):
                converged = True
                convergence_criterion = (
                    "float64_armijo_unresolvable_at_objective_resolution"
                )
                used_float64_stationarity = True
                final_delta = 0.0
                final_normalized_gradient = normalized_gradient
                final_objective = objective
                final_objective_change = 0.0
                final_predicted_decrease = predicted_decrease
                final_objective_resolution = objective_resolution
                final_parameter_resolution = _parameter_resolution(
                    params, params
                )
                break
            raise ValueError(
                "Step7-v4 logistic Armijo line search failed before "
                f"certified convergence: l2={l2} "
                f"normalized_gradient={normalized_gradient} "
                f"predicted_decrease={predicted_decrease} "
                f"objective_resolution={objective_resolution}"
            )

        applied_delta = proposed - params
        delta_norm = float(np.linalg.norm(applied_delta))
        parameter_resolution = _parameter_resolution(params, proposed)
        objective_change = float(abs(proposed_objective - objective))
        params = proposed
        final_objective, final_gradient, _final_curvature = state(params)
        final_normalized_gradient = _normalized_gradient_inf_norm(
            final_gradient, weight_total
        )
        final_delta = delta_norm
        final_objective_change = objective_change
        final_predicted_decrease = predicted_decrease
        final_objective_resolution = objective_resolution
        final_parameter_resolution = parameter_resolution
        if final_normalized_gradient <= requested_tolerance:
            converged = True
            convergence_criterion = (
                "normalized_gradient_inf_norm_at_most_requested_tolerance"
            )
            break
        if (
            _small_gradient_for_float64(
                final_normalized_gradient, requested_tolerance
            )
            and delta_norm <= parameter_resolution
            and objective_change <= objective_resolution
        ):
            converged = True
            convergence_criterion = (
                "float64_parameter_and_objective_changes_below_resolution"
            )
            used_float64_stationarity = True
            break

    if not converged:
        raise ValueError(
            "Step7-v4 logistic solver did not reach certified convergence: "
            f"l2={l2} delta={final_delta} "
            f"normalized_gradient={final_normalized_gradient} "
            f"predicted_decrease={final_predicted_decrease} "
            f"objective_resolution={final_objective_resolution}"
        )
    return {
        "mean": [float(value) for value in mean],
        "scale": [float(value) for value in scale],
        "intercept": float(params[0]),
        "coefficients": [float(value) for value in params[1:]],
        "l2_penalty": l2,
        "solver_iterations": iteration,
        "solver_final_delta_norm": final_delta,
        "solver_final_normalized_gradient_inf_norm": (
            final_normalized_gradient
        ),
        "solver_final_objective": final_objective,
        "solver_final_objective_change_abs": final_objective_change,
        "solver_final_newton_predicted_objective_decrease": (
            final_predicted_decrease
        ),
        "solver_float64_objective_resolution": final_objective_resolution,
        "solver_float64_parameter_resolution": final_parameter_resolution,
        "solver_requested_normalized_gradient_tolerance": (
            requested_tolerance
        ),
        "solver_float64_small_gradient_ceiling": max(
            requested_tolerance, FLOAT64_SQRT_EPSILON
        ),
        "solver_convergence_criterion": convergence_criterion,
        "solver_used_float64_stationarity_fallback": (
            used_float64_stationarity
        ),
        "solver_float64_objective_resolution_step_count": (
            float64_objective_resolution_step_count
        ),
        "solver_line_search": "armijo_backtracking",
        "solver_converged": True,
        "sample_weight_total": weight_total,
    }
