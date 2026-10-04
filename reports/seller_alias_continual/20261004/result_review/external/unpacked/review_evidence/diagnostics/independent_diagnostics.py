#!/usr/bin/env python3
"""Independent saved-diagnostics arithmetic; no training/model imports or labels.

Reads only twelve formal update summaries/matrices, the frozen risk policy and
already open native CPU summary. Independent math.fsum formulas reconstruct loss
decomposition, stage means, ranges and clipping counts. This does not regenerate
the supplied gradients from weights and does not assert a causal explanation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np


def mean(values):
    values = [float(v) for v in values]
    return math.fsum(values) / len(values)


def minmax(values):
    values = [float(v) for v in values]
    return {"minimum": min(values), "maximum": max(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()
    root = args.input_root.resolve()
    base = root / "reports/seller_alias_continual/20261004"
    provenance, checks, stages = [], [], []

    def read(path, array=False):
        payload = path.read_bytes()
        provenance.append({"path": str(path.relative_to(root)), "bytes": len(payload),
                           "sha256": hashlib.sha256(payload).hexdigest()})
        return np.load(path, allow_pickle=False) if array else json.loads(payload)

    def check(name, ok, detail=None):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    policy = read(base / "risk_result/source/schema/step28_risk_policy.json")
    check("frozen_risk_lambda", policy["arms"] == {"risk": 0.1})
    check("frozen_risk_gamma", policy["loss"]["risk_retention"] == 0.5)
    noises, grad = [], {k: [] for k in ("weighted_history", "rank", "positive", "negative")}
    for method, label in (("logit_tenth", "logit_low"), ("risk", "risk")):
        for order in ("ABC", "BCA", "CAB"):
            for stage in (2, 3):
                name = f"{order}_{method}_stage{stage}"
                path = base / f"{label}_result/job/run/updates/{name}.json"
                rec = read(path)
                matrix = read(path.with_suffix(".npy"), array=True)
                columns = rec["update_columns"]
                check(name + ".matrix_dimensions", matrix.shape == (288, len(columns)))
                check(name + ".all_finite", np.isfinite(matrix).all())
                check(name + ".stage_identity", rec["name"] == name and rec["stage"] == stage
                      and rec["order"] == order and rec["actual_domain"] == order[stage - 1])
                check(name + ".physical_updates", rec["updates"] == 288)
                col = {k: matrix[:, i].tolist() for i, k in enumerate(columns)}
                check(name + ".lambda_each_update", all(v == 0.1 for v in col["history_weight"]))
                check(name + ".four_observations", set(rec["observations"]) == {"1", "29", "30", "288"})
                for step in (1, 29, 30, 288):
                    obs = rec["observations"][str(step)]
                    check(name + f".observed_modules_{step}", all(
                        obs[m]["finite_nonzero_combined_gradient"]
                        and obs[m]["parameters_changed"] == (m == "head" or step != 288)
                        for m in ("encoder", "head")))
                error = {"current_base": 0.0, "history_base": 0.0,
                         "weighted_history": 0.0, "full_total": 0.0}
                if method == "risk":
                    error["channel_mean"] = 0.0
                    check(name + ".gamma_each_update", all(v == 0.5 for v in col["retention_weight"]))
                else:
                    check(name + ".mse_weight_each_update", all(v == 0.5 for v in col["logit_weight"]))
                independent_retention = []
                for i in range(288):
                    for role in ("current", "history"):
                        oracle = math.fsum([col[role + "_bce"][i], col[role + "_rank"][i],
                                            0.5 * col[role + "_hard"][i]])
                        error[role + "_base"] = max(error[role + "_base"], abs(oracle - col[role + "_total"][i]))
                    weighted_history = 0.1 * col["history_total"][i]
                    error["weighted_history"] = max(error["weighted_history"], abs(weighted_history - col["weighted_history_total"][i]))
                    if method == "risk":
                        raw = mean(col["retention_" + channel][i] for channel in ("rank", "positive", "negative"))
                        error["channel_mean"] = max(error["channel_mean"], abs(raw - col["retention"][i]))
                        retained = 0.5 * raw
                    else:
                        retained = 0.5 * col["logit_mse"][i]
                    independent_retention.append(retained)
                    full = math.fsum([col["current_total"][i], weighted_history, retained])
                    error["full_total"] = max(error["full_total"], abs(full - col["total"][i]))
                for component, delta in error.items():
                    check(name + ".decomposition_" + component, delta <= 3e-6, {"maximum_absolute_error": delta, "absolute_tolerance": 3e-6})
                summary = {"name": name, "method": method, "stage": stage, "order": order,
                           "updates": len(matrix), "combined_gradient_norm": minmax(col["gradient_norm"]),
                           "clip_count_gt_one": sum(x > 1.0 for x in col["gradient_norm"]),
                           "weighted_retention_mean": mean(independent_retention),
                           "raw_retention_mean": mean(col["retention"] if method == "risk" else col["logit_mse"]),
                           "current_total_mean": mean(col["current_total"]),
                           "weighted_history_total_mean": mean(col["weighted_history_total"]),
                           "total_mean": mean(col["total"]), "decomposition_maximum_errors": error}
                check(name + ".all_updates_clip", summary["clip_count_gt_one"] == 288)
                if method == "risk":
                    d = rec["risk_diagnostics"]
                    noise = d["train_against_same_model_eval"]
                    check(name + ".one_group_is_first_live", d["group_uid"] == rec["memory_after_training"]["members"][0])
                    check(name + ".four_draws_three_channels", len(noise) == 4 and all(len(x) == 3 for x in noise))
                    check(name + ".eval_self_zero", d["eval_self"] == [0.0, 0.0, 0.0])
                    check(name + ".finite_positive_mode_noise", all(math.isfinite(x) and x > 0 for row in noise for x in row))
                    check(name + ".gradient_components", set(d["first_update_head_gradient_norms"]) == set(grad))
                    check(name + ".scope_explicit", d["scope"] == "One retained group, fixed four seeds; noise diagnostic, no adaptive tuning")
                    noises.extend(noise)
                    for component, value in d["first_update_head_gradient_norms"].items():
                        grad[component].append(value)
                    summary.update({"component_raw_means": {key: mean(col["retention_" + key]) for key in ("rank", "positive", "negative")},
                                    "diagnostics": d,
                                    "first_update_history_uid": rec["history_ids"][0],
                                    "diagnostic_group_and_first_update_group_identical": d["group_uid"] == rec["history_ids"][0]})
                stages.append(summary)

    risks = [x for x in stages if x["method"] == "risk"]
    logits = [x for x in stages if x["method"] == "logit_tenth"]
    aggregates = {
        "mode_noise_range": {key: minmax(row[i] for row in noises) for i, key in enumerate(("rank", "positive", "negative"))},
        "head_component_norm_range": {k: minmax(v) for k, v in grad.items()},
        "stage_raw_retention_mean_range": minmax(x["raw_retention_mean"] for x in risks),
        "stage_weighted_risk_mean_range": minmax(x["weighted_retention_mean"] for x in risks),
        "stage_weighted_mse_mean_range": minmax(x["weighted_retention_mean"] for x in logits),
        "stage_channel_mean_range": {k: minmax(x["component_raw_means"][k] for x in risks) for k in ("rank", "positive", "negative")},
        "risk_clip_count": sum(x["clip_count_gt_one"] for x in risks),
        "logit_clip_count": sum(x["clip_count_gt_one"] for x in logits),
        "mode_draw_count": len(noises),
        "head_measurement_count": len(risks),
        "first_update_probe_group_same_as_mode_group_count": sum(x["diagnostic_group_and_first_update_group_identical"] for x in risks),
        "largest_independent_loss_decomposition_residual": max(v for x in stages for v in x["decomposition_maximum_errors"].values()),
    }

    # These are only rounded prose transcription checks, after reconstruction.
    # Different precision for each stated decimal/sci-notation is explicit.
    prose_ranges = [
        ("noise_rank", aggregates["mode_noise_range"]["rank"], [0.0128, 0.1048], [5e-5, 5e-5]),
        ("noise_positive", aggregates["mode_noise_range"]["positive"], [0.00628, 0.07465], [5e-6, 5e-6]),
        ("noise_negative", aggregates["mode_noise_range"]["negative"], [5.28e-8, 1.13e-5], [5e-11, 5e-8]),
        ("head_history", aggregates["head_component_norm_range"]["weighted_history"], [0.00968, 0.01645], [5e-6, 5e-6]),
        ("head_rank", aggregates["head_component_norm_range"]["rank"], [0, 0.01125], [0, 5e-6]),
        ("head_positive", aggregates["head_component_norm_range"]["positive"], [0, 0.02289], [0, 5e-6]),
        ("head_negative", aggregates["head_component_norm_range"]["negative"], [1.92e-7, 3.53e-6], [5e-10, 5e-9]),
        ("risk_raw_mean", aggregates["stage_raw_retention_mean_range"], [0.01181, 0.02744], [5e-6, 5e-6]),
        ("risk_weighted_mean", aggregates["stage_weighted_risk_mean_range"], [0.00591, 0.01372], [5e-6, 5e-6]),
        ("mse_weighted_mean", aggregates["stage_weighted_mse_mean_range"], [0.06933, 0.13022], [5e-6, 5e-6]),
        ("risk_rank_mean", aggregates["stage_channel_mean_range"]["rank"], [0.03464, 0.07584], [5e-6, 5e-6]),
        ("risk_positive_mean", aggregates["stage_channel_mean_range"]["positive"], [0.000315, 0.006424], [5e-7, 5e-7]),
        ("risk_negative_mean", aggregates["stage_channel_mean_range"]["negative"], [0.0000297, 0.000226], [5e-8, 5e-7]),
    ]
    for name, actual, expected, tolerance in prose_ranges:
        for i, bound in enumerate(("minimum", "maximum")):
            check("report." + name + "." + bound, abs(actual[bound] - expected[i]) <= tolerance[i],
                  {"actual": actual[bound], "reported": expected[i], "rounding_tolerance": tolerance[i]})
    check("report.clip_each_method_1728", aggregates["risk_clip_count"] == aggregates["logit_clip_count"] == 1728)
    native = read(root / "direct_reviews/risk_cpu/native.json")
    check("prior_native_explicit_three_updates", native["actual_native_updates"] == 3)
    check("prior_native_both_parameter_increments_nonzero", all(
        x["increment_norm"] > 0 and x["different_actual_parameter_update"]
        for x in native["retention_gradient_decomposition"].values()))
    result = {"status": "PASS" if all(x["pass"] for x in checks) else "FAIL",
              "scope": "Independent saved update arithmetic and diagnostic transcription; no weights, labels, Torch, training imports or server access",
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "argv": sys.argv},
              "aggregates": aggregates, "stages": stages, "checks": checks,
              "prior_native_summary_reused_not_rerun": native,
              "input_files": provenance,
              "limitations": ["Gradients are saved measurements, not regenerated from original model weights.",
                              "Mode probes cover one live group and four fixed seeds per continuation stage.",
                              "Gradient components cover the hidden-head weight at only the first update per stage.",
                              "Mode probe and first-update gradient probe generally use different retained groups.",
                              "Scalar loss magnitude and component norms cannot establish combined update strength, angles or causality.",
                              "No full-cache end-of-stage evaluation, no new calibration, no labels or model weights loaded."]}
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(checks),
                      "failed": [x for x in checks if not x["pass"]], "aggregates": aggregates},
                     ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
