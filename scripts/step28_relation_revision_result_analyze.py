"""Summarize already-open saved results; no labels, text, model or training imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(job, old_job, verification, out):
    out.mkdir(exist_ok=False)
    result = read(job / "evaluation/evaluation.json")
    old = read(old_job / "evaluation/evaluation.json")
    verified = read(verification)
    assert verified["status"] == "PASS_SAVED_RESULT_INDEPENDENT_VERIFICATION"
    diagnostics = {"scope": "Descriptive saved-log diagnostics; no new comparison CI or causal attribution",
                   "input_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                    (job / "evaluation/evaluation.json", old_job / "evaluation/evaluation.json", verification)},
                   "training": {}, "old_configuration": {}, "stage2_predictions": {}}
    for name, point in verified["training_points"].items():
        rows = read(job / "run/updates" / (name + ".json"))["updates"]
        keys = ["current_bce", "current_rank", "current_hard", "current_total",
                "history_compressed", "history_rank", "history_total", "gradient_norm"]
        summary = {}
        for key in keys:
            values = np.array([r[key] for r in rows])
            summary[key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                            "first": float(values[0]), "last": float(values[-1]),
                            "first24_mean": float(values[:24].mean()), "last24_mean": float(values[-24:].mean())}
        summary["clip_active_count"] = sum(r["gradient_norm"] > 1 for r in rows)
        summary["current_sum_max_abs_error"] = max(abs(r["current_total"] - float(
            np.float32(np.float32(r["current_bce"]) + np.float32(r["current_rank"])) +
            np.float32(.5) * np.float32(r["current_hard"]))) for r in rows)
        assert summary["current_sum_max_abs_error"] < 1e-5
        old_rows = read(old_job / "run/updates" / (name.replace("relation_revision", "relation") + ".json"))["updates"]
        summary["old_history_first"] = old_rows[0]["history"]
        summary["old_gradient_median"] = float(np.median([r["gradient_norm"] for r in old_rows]))
        diagnostics["training"][name] = summary
        if name.endswith("stage2"):
            counts = read(job / "evaluation" / (name + "_stage-cal_counts.json"))
            diagnostics["stage2_predictions"][name] = {k: sum(r[k] for r in counts) for k in ("tp", "fp", "fn", "tn")}
    for ep in ("O", "N", "Z", "F_first", "F", "G", "final_all"):
        diagnostics["old_configuration"][ep] = {m: old["endpoints"]["relation"]["primary"][ep][m]["mean"]
                                                for m in ("map", "average_precision", "brier", "log_loss")}
    diagnostics["old_first_stage_map"] = {order: old["absolute_stage_results"][order + "_relation_stage1"]["raw"]["macro_by_domain"][order[0]]["map"] for order in ("ABC", "BCA", "CAB")}
    lines = ["# 完整主角色指标", "", "差值＝修订候选−LOGIT0.1；区间是固定单种子／三顺序下按实际域整群配对的条件95%区间。", "",
             "O/N/Z/final_all：Brier、log-loss越小越好，其余越大越好；F_first/F为遗忘，越小越好；G为获得，越大越好。", "",
             "|端点|指标|候选|LOGIT0.1|差值|条件95%区间|ABC差值|BCA差值|CAB差值|",
             "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    for ep, metrics in result["delta"].items():
        for metric, d in metrics.items():
            a = result["endpoints"]["relation"]["primary"][ep][metric]["mean"]
            b = result["endpoints"]["logit0.1"]["primary"][ep][metric]["mean"]
            lo, hi = d["conditional_95pct_interval"]
            values = "|".join(f"{d['per_order'][o]:+.9f}" for o in ("ABC", "BCA", "CAB"))
            lines.append(f"|{ep}|{metric}|{a:.9f}|{b:.9f}|{d['mean']:+.9f}|[{lo:+.9f}, {hi:+.9f}]|{values}|")
    (out / "metrics.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    colors = {"A": "#277a8a", "B": "#d38530", "C": "#9764a0"}
    for ax, order in zip(axes, ("ABC", "BCA", "CAB")):
        for domain in "ABC":
            for arm, style in (("relation", "-"), ("logit0.1", "--")):
                vals = [verified["stage_metrics"][arm][order][str(s)][domain]["map"] for s in (1, 2, 3)]
                ax.plot((1, 2, 3), vals, style, color=colors[domain], marker="o" if arm == "relation" else None,
                        label=f"{domain}: revised" if arm == "relation" else f"{domain}: LOGIT0.1")
        ax.set(title=order, xlabel="Completed training stage", xticks=(1, 2, 3), ylim=(.28, .52))
        ax.grid(alpha=.18)
    axes[0].set_ylabel("MAP on already-developed valid groups")
    axes[-1].legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle("Fixed candidate: stage trajectories (s0; three orders, not three seeds)")
    fig.tight_layout()
    fig.savefig(out / "stage_map.png", dpi=170)
    fig.savefig(out / "stage_map.svg")
    plt.close(fig)
    print(json.dumps({"status": "SAVED_DIAGNOSTICS_COMPLETE", "out": str(out), "training_points": 9}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("job", "old-job", "verification", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    main(args.job, args.old_job, args.verification, args.out)
