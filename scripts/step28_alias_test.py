"""Fixed-pair test statistics; no text/label/model loading or fitting."""
from __future__ import annotations

from typing import Any

import numpy as np

import step28_alias_calibration as calibration

base = calibration.ranking.base
data, metrics = calibration.data, calibration.metrics
ROLES = ("A_raw", "A_cal", "C_raw", "C_cal")
COMPARISONS = (("C_cal", "A_cal"), ("C_cal", "A_raw"),
               ("C_cal", "C_raw"), ("A_cal", "A_raw"))
CRITERIA = (("T1", "C_cal_minus_A_cal", "map", "mean", ">=", .01),
            ("T2", "C_cal_minus_A_cal", "map", "lower", ">", 0.),
            ("T3", "C_cal_minus_A_cal", "recall_at_5", "mean", ">", 0.),
            ("T4", "C_cal_minus_A_cal", "average_precision", "mean", ">=", 0.),
            ("T5", "C_cal_minus_A_cal", "roc_auc", "mean", ">=", 0.),
            ("T6", "C_cal_minus_A_cal", "brier", "mean", "<=", 0.),
            ("T7", "C_cal_minus_A_cal", "log_loss", "mean", "<=", 0.),
            ("T8", "C_cal_minus_A_raw", "brier", "mean", "<=", 0.),
            ("T9", "C_cal_minus_A_raw", "log_loss", "mean", "<=", 0.))


def domain_rows(domains: list[str]) -> list[np.ndarray]:
    rows = [np.flatnonzero(np.asarray(domains) == name) for name in "ABC"]
    if len(domains) != 120 or any(len(row) != 40 for row in rows):
        raise ValueError("Expected all120 test groups,40 per domain")
    return rows


def bootstrap_draws() -> np.ndarray:
    return np.random.Generator(np.random.PCG64(20260928)).integers(
        0, 40, size=(5000, 3, 40))


def compare(candidate: np.ndarray, reference: np.ndarray, domains: list[str],
            draws: np.ndarray) -> dict[str, Any]:
    rows = domain_rows(domains)
    if (candidate.shape != (120, 22) or reference.shape != (120, 22)
            or not np.isfinite([candidate, reference]).all()
            or draws.shape != (5000, 3, 40) or draws.dtype.kind not in "iu"
            or np.any((draws < 0) | (draws >= 40))):
        raise ValueError("Complete paired test matrices and whole-group draws required")
    result = base.metric_comparison(candidate, reference, domains, draws)
    delta = candidate - reference
    for index, name in enumerate(metrics.COLUMNS):
        result[name]["by_domain"] = {d: float(delta[row, index].mean())
                                     for d, row in zip("ABC", rows, strict=True)}
    return result


def acceptance(comparisons: dict) -> dict:
    checks, observed = {}, {}
    for key, comparison, metric, statistic, operator, bound in CRITERIA:
        record = comparisons[comparison][metric]
        value = float(record["conditional_95pct_interval"][0]
                      if statistic == "lower" else record["mean"])
        if not np.isfinite(value):
            raise ValueError("Nonfinite acceptance value")
        checks[key] = bool(value >= bound if operator == ">=" else
                           value > bound if operator == ">" else value <= bound)
        observed[key] = {"value": value, "operator": operator, "bound": bound}
    return {"passed": all(checks.values()), "checks": checks, "observed": observed,
            "failed": [key for key, passed in checks.items() if not passed],
            "scope": "Fixeds0 same-generator120group observed guards and conditional intervals; "
                     "not population noninferiority or all-threshold protection."}


def summarize(matrices: dict[str, np.ndarray], domains: list[str],
              draws: np.ndarray) -> dict:
    if set(matrices) != set(ROLES):
        raise ValueError("Four complete model roles required")
    comparisons = {candidate + "_minus_" + reference:
                   compare(matrices[candidate], matrices[reference], domains, draws)
                   for candidate, reference in COMPARISONS}
    return {"comparisons": comparisons, "acceptance": acceptance(comparisons)}
