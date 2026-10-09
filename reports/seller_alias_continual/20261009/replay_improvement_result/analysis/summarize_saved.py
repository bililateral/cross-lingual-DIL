"""Display saved five-arm results; no labels, model loading or new resampling."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np

ARMS = ("LOGIT0.1", "C", "S", "C_plus", "S_strong")
ORDERS = ("ABC", "BCA", "CAB")
ENDPOINTS = ("O", "N", "Z", "final_all", "A2", "F_first", "F", "G")
CLEAR = "未见明确迹象"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def formatted(value: dict) -> str:
    lo, hi = value["conditional_95pct_interval"]
    return f"{value['mean']:.9f} [{lo:.9f}, {hi:.9f}]"


def main(job: Path, old_job: Path, out: Path) -> None:
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    evaluation = read(job / "evaluation/evaluation.json")
    diagnostic = read(job / "evaluation/diagnostics/overfitting.json")
    manifest = read(job / "run/manifest.json")
    collected = read(job / "evaluation/collected.json")
    summary = {"scope": "Extract saved estimates/intervals and count saved decisions; no new metrics, bootstrap, labels or model inference.",
               "completion": read(job / "completion.json"), "primary": {},
               "comparisons": {}, "logical_diagnostic_counts": {},
               "physical_diagnostic_counts": {}, "supported_diagnostics": [],
               "historical_replay": {}, "first_stage_saved_update_comparison": {}}
    for arm in ARMS:
        summary["primary"][arm] = evaluation["endpoints"][arm]["primary"]
    for name, comparison in evaluation["comparisons"].items():
        summary["comparisons"][name] = {
            "passed_count": sum(comparison["checks"].values()),
            "total_checks": len(comparison["checks"]), "failed": comparison["assessment"]["failed"],
            "primary": comparison["delta"]["primary"],
            "raw_probability": {ep: {m: comparison["delta"]["raw"][ep][m] for m in ("brier", "log_loss")} for ep in ("O", "N", "Z")}}
    counts = Counter()
    curve_rows = []
    for point, roles in diagnostic["points"].items():
        for role, domains in roles.items():
            for domain, value in domains.items():
                if not isinstance(value, dict) or "interpretation" not in value:
                    continue
                for metric, label in value["interpretation"].items():
                    index = value["columns"].index(metric)
                    counts[role + "/" + metric + "/" + label] += 1
                    entry = {"point": point, "role": role, "domain": domain, "metric": metric,
                             "interpretation": label, "train_groups": value["fit_or_cache_groups"],
                             "valid_groups": value["valid_groups"],
                             "train_curve": [row[index] for row in value["train_raw_curve"]],
                             "valid_curve": [row[index] for row in value["valid_raw_curve"]]}
                    for field in ("epoch1_to_6_train_benefit", "epoch1_to_6_valid_benefit", "epoch1_to_6_gap_widening"):
                        entry[field] = {"mean": value[field]["mean"][index],
                                        "conditional_95pct_interval": value[field]["conditional_95pct_interval"][index]}
                    curve_rows.append(entry)
                    if label == "有相应迹象":
                        summary["supported_diagnostics"].append(entry)
    summary["physical_diagnostic_counts"] = dict(sorted(counts.items()))
    for arm in ARMS:
        counts = Counter()
        for order in ORDERS:
            for stage in (1, 2, 3):
                point = order + "_shared" if stage == 1 and arm != "LOGIT0.1" else f"{order}_{arm}_stage{stage}"
                for role, domains in diagnostic["points"][point].items():
                    for domain, value in domains.items():
                        if isinstance(value, dict) and "interpretation" in value:
                            for metric, label in value["interpretation"].items():
                                counts[role + "/" + label] += 1
        summary["logical_diagnostic_counts"][arm] = dict(sorted(counts.items()))
    for arm in ("C", "S", "LOGIT0.1"):
        values = [value for point, roles in evaluation["historical_replay"]["points"].items()
                  if "_" + arm + "_stage" in point for value in roles.values()]
        summary["historical_replay"][arm] = {"logical_role_matrices": len(values),
            "bitwise_equal": sum(value["bitwise_equal"] for value in values),
            "maximum_absolute_group_difference": max(value["maximum_absolute_group_difference"] for value in values)}
    historical = out / "historical"
    historical.mkdir()
    old_manifest = read(old_job / "run/manifest.json")
    summary["initial_logit_model_digest_equal"] = old_manifest["initial"]["model_state_sha256"] == manifest["initial_model_digests"]["LOGIT0.1"]
    shutil.copy2(old_job / "run/manifest.json", historical / "manifest.json")
    for order in ORDERS:
        old_log_path = old_job / "run/updates" / (order + "_shared.json")
        old_log = read(old_log_path)
        rec = old_log["update_file"]
        old_values_path = old_job / "run" / rec["path"]
        assert identity(old_values_path)["sha256"] == rec["sha256"]
        old_values = np.load(old_values_path, allow_pickle=False)
        new_log = read(job / "run/updates" / (order + "_LOGIT0.1_stage1.json"))
        columns = [name for name in old_log["update_columns"] if name != "logit_term_host_seconds"]
        mismatches = []
        for step, row in enumerate(new_log["rows"], 1):
            changed = {name: {"old": float(old_values[step-1, old_log["update_columns"].index(name)]), "new": row[name]}
                       for name in columns if old_values[step-1, old_log["update_columns"].index(name)] != row[name]}
            if changed:
                mismatches.append({"step": step, "fields": changed})
        summary["first_stage_saved_update_comparison"][order] = {
            "current_schedule_equal": old_log["current_ids"] == new_log["current_ids"],
            "steps_with_any_non_timing_difference": len(mismatches),
            "first_differing_step": mismatches[0] if mismatches else None,
            "diagnostic_first_occurs_after_step": 48}
        shutil.copy2(old_log_path, historical / old_log_path.name)
        shutil.copy2(old_values_path, historical / old_values_path.name)
    summary["exploration_failure"] = "One earlier interactive stdin summary failed before execution with Non-UTF-8 SyntaxError; replaced by this UTF-8 file. No labels or experimental state changed."
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "curves.json").write_text(json.dumps(curve_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["Saved results, original conditional 95% intervals; no new resampling.",
             "Primary: raw ranking/curve metrics plus stage-cal probability/classification metrics.",
             "O old; N newly learned; Z final new; A2 stage2 new; F_first/F lower is better; G higher is better.",
             "The three orders are not three seeds; shared first-stage curves are aliases."]
    for role in ("primary", "raw", "stage-cal", "first-cal"):
        for ep in ENDPOINTS:
            lines += ["", f"ABSOLUTE {role} {ep}", "metric\t" + "\t".join(ARMS)]
            for metric in collected["metric_columns"]:
                lines.append(metric + "\t" + "\t".join(formatted(evaluation["endpoints"][arm][role][ep][metric]) for arm in ARMS))
    for name, comparison in evaluation["comparisons"].items():
        lines += ["", f"COMPARISON {name}: {sum(comparison['checks'].values())}/23", "FAILED: " + ", ".join(comparison["assessment"]["failed"])]
        for role in ("primary", "raw", "stage-cal", "first-cal"):
            for ep in ENDPOINTS:
                lines.append(f"{role} {ep}")
                for metric in collected["metric_columns"]:
                    lines.append(metric + "\t" + formatted(comparison["delta"][role][ep][metric]))
    (out / "all_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["All 228 physical phase/role/domain/metric decisions, copied from saved diagnostics.",
             "Positive benefit = higher MAP/R5 or lower log_loss. Epoch1 -> epoch6 fixed in advance."]
    for row in curve_rows:
        lines += ["", " ".join(row[k] for k in ("point", "role", "domain", "metric", "interpretation")),
                  "train: " + ", ".join(f"{v:.9f}" for v in row["train_curve"]),
                  "valid: " + ", ".join(f"{v:.9f}" for v in row["valid_curve"])]
        for field in ("epoch1_to_6_train_benefit", "epoch1_to_6_valid_benefit", "epoch1_to_6_gap_widening"):
            lines.append(field + ": " + formatted(row[field]))
    (out / "all_curves.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    comparisons = ("C_plus-S", "C_plus-S_strong", "C_plus-LOGIT0.1")
    for ax, endpoint in zip(axes, ("O", "N", "Z")):
        for y, name in enumerate(comparisons):
            value = evaluation["comparisons"][name]["delta"]["primary"][endpoint]["map"]
            lo, hi = value["conditional_95pct_interval"]
            ax.errorbar(value["mean"], y, xerr=[[value["mean"]-lo], [hi-value["mean"]]], fmt="o", capsize=4, color=("#b54139" if hi < 0 else "#167865" if lo > 0 else "#64748b"))
        ax.axvline(0, color="#999999", linewidth=1)
        ax.set_title({"O": "Old domains (O)", "N": "New learning (N)", "Z": "Final new domain (Z)"}[endpoint])
        ax.set_xlabel("MAP difference (C_plus - comparator)")
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(range(3), ("vs S", "vs S_strong", "vs LOGIT0.1"))
    axes[0].invert_yaxis()
    fig.suptitle("Five-arm study: saved paired differences and conditional 95% intervals")
    fig.tight_layout()
    fig.savefig(out / "map_comparisons.png", dpi=180)
    fig.savefig(out / "map_comparisons.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, color in (("C", "#167865"), ("S", "#64748b"), ("C_plus", "#b54139"), ("S_strong", "#9a6c12")):
        value = diagnostic["points"][f"CAB_{arm}_stage3"]["fit"]["B"]
        for ax, role in zip(axes, ("train", "valid")):
            ax.plot(range(1, 7), [row[2] for row in value[role + "_raw_curve"]], marker="o", color=color, label=arm)
            ax.set_title("Current fit: 48 groups" if role == "train" else "Matched valid: 20 groups")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Unweighted raw log_loss (lower is better)")
            ax.grid(alpha=.2)
    axes[1].legend()
    fig.suptitle("CAB stage 3, domain B: fixed six-epoch probability-loss curves")
    fig.tight_layout()
    fig.savefig(out / "cab_fit_log_loss.png", dpi=180)
    fig.savefig(out / "cab_fit_log_loss.svg")
    plt.close(fig)
    execution = {"status": "SAVED_RESULT_DISPLAY_COMPLETE", "observed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                 "elapsed_seconds": time.monotonic()-started, "script": identity(Path(__file__)),
                 "inputs": [identity(job / p) for p in ("completion.json", "evaluation/evaluation.json", "evaluation/diagnostics/overfitting.json")],
                 "outputs": [identity(p) for p in sorted(out.rglob("*")) if p.is_file()]}
    (out / "execution.json").write_text(json.dumps(execution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": execution["status"], "elapsed_seconds": execution["elapsed_seconds"],
                      "diagnostic_decisions": len(curve_rows), "logical_counts": summary["logical_diagnostic_counts"],
                      "historical_first_updates": summary["first_stage_saved_update_comparison"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--old-job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.job, args.old_job, args.out)
