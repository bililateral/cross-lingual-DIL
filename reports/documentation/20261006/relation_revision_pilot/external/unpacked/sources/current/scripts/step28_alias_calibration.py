"""Positive affine logit calibration on train-calibration scores only.

No formal input loader, neural model or test access. Raw ranking is preserved by
a positive slope, with a separate numerical check for ties and strict order.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

import step28_alias_ranking as ranking

data, metrics = ranking.data, ranking.metrics
SEEDS = ("s0", "s1", "s2")
ARMS = ("d", "hard")
RUNS = tuple(f"{seed}_{arm}" for seed in SEEDS for arm in ARMS)
VARIANTS = ("d_raw", "d_calibrated", "hard_raw", "hard_calibrated")
COMPARISONS = (("hard_calibrated", "d_calibrated"),
               ("hard_calibrated", "d_raw"),
               ("hard_calibrated", "hard_raw"),
               ("d_calibrated", "d_raw"))
BOUNDS = ((0.001, 100.0), (-100.0, 100.0))
OPTIONS = {"maxiter": 200, "maxfun": 2000, "maxls": 40,
           "maxcor": 10, "ftol": 1e-12, "gtol": 1e-8}
EVALUATION = {"valid_bootstrap_seed": 20260927, "bootstrap_replicates": 5000,
              "minimum_map_gain": .01,
              "scope": "Fixed models, calibration maps, paired seeds and synthetic groups; "
                       "not refitting, retraining, regeneration or repeated-valid selection uncertainty."}


def aligned(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if (x.shape != y.shape or x.ndim not in (1, 2) or not x.size
            or not np.isfinite(x).all() or not np.isin(y, [0., 1.]).all()
            or not 0 < y.sum() < y.size):
        raise ValueError("Aligned finite logits and both binary classes required")
    return x, y


def loss_gradient(parameters: np.ndarray, scores: np.ndarray,
                  labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Unweighted mean Bernoulli NLL; differentiating the unclipped logit loss."""
    a, b = np.asarray(parameters, dtype=np.float64)
    z = a * scores + b
    p = np.exp(-np.logaddexp(0., -z))
    residual = p - labels
    return (float(np.mean(np.logaddexp(0., z) - labels * z)),
            np.array([np.mean(residual * scores), np.mean(residual)]))


def projected_gradient(parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    result = gradient.copy()
    for i, (lower, upper) in enumerate(BOUNDS):
        if ((parameters[i] <= lower and gradient[i] > 0)
                or (parameters[i] >= upper and gradient[i] < 0)):
            result[i] = 0.
    return result


def parameters(record: dict) -> tuple[float, float]:
    a, b = float(record["a"]), float(record["b"])
    if (not np.isfinite([a, b]).all()
            or not BOUNDS[0][0] <= a <= BOUNDS[0][1]
            or not BOUNDS[1][0] <= b <= BOUNDS[1][1]):
        raise ValueError("Calibration parameters outside the positive-slope contract")
    return a, b


def transform(scores: np.ndarray, record: dict) -> np.ndarray:
    a, b = parameters(record)
    original = np.asarray(scores, dtype=np.float64)
    result = a * original + b
    if not original.size or not np.isfinite(original).all() or not np.isfinite(result).all():
        raise ValueError("Invalid calibration logits")
    return result


def preserve_order(original: np.ndarray, transformed: np.ndarray) -> dict:
    """Preserve every group's total pair order and exact ties, hence every query order."""
    x, z = np.asarray(original, dtype=np.float64), np.asarray(transformed, dtype=np.float64)
    if x.shape != z.shape or x.ndim != 2 or not x.size or not np.isfinite([x, z]).all():
        raise ValueError("Aligned finite group matrices required")
    order = np.argsort(x, axis=1, kind="stable")
    if not np.array_equal(order, np.argsort(z, axis=1, kind="stable")):
        raise ValueError("Calibration changed pair order")
    before = np.take_along_axis(x, order, axis=1)
    after = np.take_along_axis(z, order, axis=1)
    ties = before[:, 1:] == before[:, :-1]
    if (not np.array_equal(ties, after[:, 1:] == after[:, :-1])
            or not np.all(after[:, 1:][~ties] > after[:, :-1][~ties])):
        raise ValueError("Calibration changed exact ties or collapsed distinct logits")
    return {"groups": len(x), "pairs_per_group": x.shape[1],
            "exact_order_and_ties_preserved": True}


def fit(scores: np.ndarray, labels: np.ndarray, *, role: str,
        check: Callable[[], None] = lambda: None) -> dict:
    """One deterministic real optimizer fit; no retry or alternative method selection."""
    from scipy.optimize import minimize

    if role != "calibration":
        raise ValueError("Only train-calibration labels may fit a probability map")
    x, y = aligned(scores, labels)
    initial = np.array([1., 0.])
    initial_loss, _ = loss_gradient(initial, x, y)
    trajectory: list[dict] = []
    calls = 0

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal calls
        check()
        calls += 1
        if calls > OPTIONS["maxfun"]:
            raise RuntimeError("Calibration objective evaluation budget exhausted")
        return loss_gradient(theta, x, y)

    def callback(theta: np.ndarray) -> None:
        check()
        value, grad = loss_gradient(theta, x, y)
        trajectory.append({"iteration": len(trajectory) + 1, "a": float(theta[0]),
                           "b": float(theta[1]), "nll": value,
                           "projected_gradient_max": float(np.max(np.abs(projected_gradient(theta, grad))))})

    fitted = minimize(objective, initial, method="L-BFGS-B", jac=True,
                      bounds=BOUNDS, options=dict(OPTIONS), callback=callback)
    final_loss, gradient = loss_gradient(fitted.x, x, y)
    pg = float(np.max(np.abs(projected_gradient(fitted.x, gradient))))
    result = {"a": float(fitted.x[0]), "b": float(fitted.x[1]),
              "role": role, "objective": "unweighted_unclipped_bernoulli_nll",
              "initial_parameters": initial.tolist(), "bounds": [list(v) for v in BOUNDS],
              "options": dict(OPTIONS), "initial_nll": initial_loss, "final_nll": final_loss,
              "optimizer_success": bool(fitted.success), "optimizer_message": str(fitted.message),
              "optimizer_iterations": int(fitted.nit), "objective_calls": calls,
              "projected_gradient_max": pg, "trajectory": trajectory,
              "group_count": int(x.shape[0]) if x.ndim == 2 else None,
              "pair_count": int(x.size), "positive_count": int(y.sum())}
    parameters(result)
    passed = (fitted.success and fitted.nit <= OPTIONS["maxiter"]
              and calls <= OPTIONS["maxfun"] and np.isfinite(final_loss)
              and pg <= 1e-6 and final_loss <= initial_loss + 1e-12)
    result["status"] = "PASS_CALIBRATION_FIT" if passed else "FAILED_CALIBRATION_FIT_NO_RETRY"
    # Empirical calibration diagnostics do not choose among maps or alter the fit.
    for name, values in (("raw", x), ("calibrated", transform(x, result))):
        probabilities = np.exp(-np.logaddexp(0., -values))
        result[name + "_brier"] = float(np.mean((probabilities - y) ** 2))
    return result


def acceptance(primary: dict, against_raw: dict) -> dict:
    result = ranking.acceptance(primary, EVALUATION)
    checks = dict(result["checks"])
    for name in ("brier", "log_loss"):
        value = against_raw["metrics"][name]
        checks[name + "_mean_against_raw_a"] = value["mean"] <= 0
        checks[name + "_s0_against_raw_a"] = value["per_seed"][0] <= 0
    if len(checks) != 17:
        raise ValueError("Expected the confirmed seventeen performance guards")
    return {"passed": all(checks.values()), "checks": checks,
            "failed": [name for name, ok in checks.items() if not ok],
            "scope": result["scope"],
            "next": "SEPARATELY_REVIEWED_AND_AUTHORIZED_TEST" if all(checks.values())
                    else "RECORD_NEGATIVE_RESULT_NO_AUTOMATIC_RETRY"}


def summarize(matrices: dict[str, np.ndarray], domains: list[str]) -> dict[str, Any]:
    expected = {f"{seed}_{variant}" for seed in SEEDS for variant in VARIANTS}
    if set(matrices) != expected or any(v.shape != (60, 22) for v in matrices.values()):
        raise ValueError("Twelve complete valid matrices required before comparisons")
    comparisons = {}
    for candidate, reference in COMPARISONS:
        differences = np.stack([matrices[f"{seed}_{candidate}"] - matrices[f"{seed}_{reference}"]
                                for seed in SEEDS])
        comparisons[candidate + "_minus_" + reference] = ranking.paired_summary(
            differences, domains, EVALUATION)
    return {"comparisons": comparisons,
            "acceptance": acceptance(comparisons["hard_calibrated_minus_d_calibrated"],
                                     comparisons["hard_calibrated_minus_d_raw"])}
