"""Independent saved-result audit; Linux py310 only, no labels/models/Memory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.workspace / "reports/job"
    run, evaluation = root / "run", root / "evaluation"
    manifest, completion = read(run / "manifest.json"), read(root / "completion.json")
    col, reported = read(evaluation / "collected.json"), read(evaluation / "evaluation.json")
    before, access = read(root / "before_valid.json"), read(root / "access.json")
    assert before["status"] == "ALL_BLIND_GATES_PASS"
    assert before["access"] == dict(train=1, valid=0, heldout=0, owners=0)
    assert access == completion["access"] == dict(train=1, valid=1, heldout=0, owners=0)
    assert completion["physical_updates"] == manifest["physical_updates"] == 1440
    assert completion["gradient_group_presentations"] == 2304
    assert not (root / "failure.json").exists()
    gate = read(args.workspace / "reports/documentation/20261012/crossfit_retention/qualification/formal_gate.json")
    source_files = completion["source_files"]
    for obj in [manifest, col, reported, before, read(root / "execution.json"), gate]:
        assert obj["source_files"] == source_files
    checked_files = []

    def verify(parent, rec):
        p = parent / rec["path"]
        assert p.stat().st_size == rec["bytes"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"]
        checked_files.append(str(p.relative_to(args.workspace)))
        return p

    for rec in source_files:
        verify(args.workspace, rec)
    verify(root, before["manifest"])
    assert manifest["starts"]["S"] == manifest["starts"]["C_cf"] == manifest["starts"]["C_perm"]
    common_members = manifest["starts"]["S"]["members"]
    rows = [np.flatnonzero(np.array(col["domains"]) == d) for d in "AB"]
    assert list(map(len, rows)) == [20, 20] and len(set(col["group_ids"])) == 40
    columns = col["columns"]
    points = ["A_shared", "B_S", "B_C_cf", "B_C_perm"]
    roles = ["raw", "stage-cal", "first-cal"]
    errors, scalar_checks, absolute, matrices, all_csv = [], 0, {}, {}, []

    def compare(actual, expected):
        nonlocal scalar_checks
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        difference = float(np.max(np.abs(a - b))) if a.size else 0.0
        errors.append(difference)
        scalar_checks += a.size
        assert difference <= 1e-12, difference

    # Reconstruct the same registered draw, not a new statistical experiment.
    draws = np.random.Generator(np.random.PCG64(20260930)).integers(20, size=(5000, 2, 20))
    frequencies = np.stack([np.stack([np.bincount(d, minlength=20) for d in draws[:, i]]) for i in range(2)])

    def stats(values, domain_index):
        sample_means = frequencies[domain_index] @ values / 20
        return values.mean(axis=0), np.percentile(sample_means, [2.5, 97.5], axis=0)

    def audit_stats(values, report, domain_index, kind, point, role, domain):
        mean, bounds = stats(values, domain_index)
        for j, metric in enumerate(columns):
            compare([mean[j], *bounds[:, j]], [report[metric]["mean"], *report[metric]["conditional_95pct_interval"]])
            all_csv.append([kind, point, role, domain, metric, mean[j], *bounds[:, j]])

    classification = {}
    rank_columns = [columns.index(k) for k in columns if k not in {
        "brier", "log_loss", "precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc"}]
    for point in points:
        p = manifest["points"][point]
        assert p["full_restore_verified"] and p["inference_restore_verified"]
        assert p["adam_step"] == (288 if point == "A_shared" else 576)
        assert p["memory_summary"]["members"] == common_members
        assert p["memory_summary"]["seen"] == 48 and p["memory_summary"]["bytes"] <= 1048576
        matrices[point], classification[point], absolute[point] = {}, {}, {}
        for role in roles:
            rec = col["points"][point][role]
            matrix = np.load(verify(evaluation, rec["matrix"]), allow_pickle=False)
            assert matrix.shape == (40, 22) and matrix.dtype == np.float64 and np.isfinite(matrix).all()
            matrices[point][role] = matrix
            target = reported["absolute"][point][role]
            for i, domain in enumerate("AB"):
                audit_stats(matrix[rows[i]], target["by_domain"][domain], i, "absolute", point, role, domain)
            compare(matrix.mean(0), [target["macro_all"][k] for k in columns])
            counts = read(verify(evaluation, rec["counts"]))
            assert len(counts) == 40 and all(r["tp"] + r["fn"] == 20 and r["fp"] + r["tn"] == 358 for r in counts)
            c = np.array([[r[k] for k in ("tp", "fp", "fn", "tn")] for r in counts], dtype=np.int64)
            classification[point][role] = {}
            for domain, selected in [("all", np.arange(40)), ("A", rows[0]), ("B", rows[1])]:
                tp, fp, fn, tn = map(int, c[selected].sum(0))
                values = dict(tp=tp, fp=fp, fn=fn, tn=tn, fpr=fp / (fp + tn), recall=tp / (tp + fn),
                              precision=tp / (tp + fp) if tp + fp else 0.0, f1=2 * tp / (2 * tp + fp + fn))
                for key, value in values.items():
                    compare(value, target["fixed_half_classification"][domain][key])
                classification[point][role][domain] = values
            scores = np.load(verify(run, p["scores"][role]), allow_pickle=False)
            assert scores.shape == (40, 378) and np.isfinite(scores).all()
            if role == "raw":
                raw = scores
            else:
                affine = p["maps"]["stage" if role == "stage-cal" else "first"]
                compare(scores, affine["a"] * raw.astype(np.float64) + affine["b"])
                assert np.array_equal(np.argsort(raw, axis=1, kind="stable"), np.argsort(scores, axis=1, kind="stable"))
                compare(matrix[:, rank_columns], matrices[point]["raw"][:, rank_columns])
            absolute[point][role] = {d: {k: float(matrix[rows[i], j].mean()) for j, k in enumerate(columns)} for i, d in enumerate("AB")}
    paired_groups = {}
    for contrast, other in [("C_cf-S", "B_S"), ("C_cf-C_perm", "B_C_perm")]:
        paired_groups[contrast] = {}
        for role in roles:
            delta = matrices["B_C_cf"][role] - matrices[other][role]
            for i, domain in enumerate("AB"):
                audit_stats(delta[rows[i]], reported["comparisons"][contrast][role][domain], i, "contrast", contrast, role, domain)
                if role == "raw":
                    v = delta[rows[i], columns.index("map")]
                    paired_groups[contrast][domain] = dict(positive=int((v > 0).sum()), negative=int((v < 0).sum()), zero=int((v == 0).sum()),
                        minimum=float(v.min()), maximum=float(v.max()), values=v.tolist())
    for arm in ["S", "C_cf", "C_perm"]:
        for role in roles:
            delta = matrices["B_" + arm][role] - matrices["A_shared"][role]
            audit_stats(delta[rows[0]], reported["old_changes"][arm][role], 0, "old_change", arm, role, "A")

    training = {p: read(verify(run, manifest["training"][p])) for p in points}
    assert all(len(t["rows"]) == t["updates"] == 288 for t in training.values())
    assert training["B_S"]["current_ids"] == training["B_C_cf"]["current_ids"] == training["B_C_perm"]["current_ids"]
    assert training["B_S"]["history_ids"] == training["B_C_cf"]["history_ids"] == training["B_C_perm"]["history_ids"]
    train_summary = {}
    for point, t in training.items():
        assert [r["step"] for r in t["rows"]] == list(range(1, 289))
        if point != "A_shared":
            for key in ("members", "seen", "tables", "origins"):
                assert t["memory_before"][key] == t["memory_after"][key]
            history = [r["history"] for r in t["rows"]]
            train_summary[point] = {
                "nonzero_distillation_steps": sum(r["distillation"] > 0 for r in history),
                "mean_distillation": float(np.mean([r["distillation"] for r in history])),
                "mean_history_supervised": float(np.mean([r["supervised"] for r in history])),
                "mean_current_supervised": float(np.mean([r["current"]["supervised"] for r in t["rows"]])),
                "mean_record_gradient_norm": float(np.mean([r["record_gradient_norm"] for r in history])),
                "mean_history_violations": float(np.mean([r["violations"] for r in history])) if point != "B_S" else None,
            }
    source_summary = {}
    for i, role in enumerate(["U0", "U1"]):
        source = read(verify(run, manifest["sources"][role]))
        assert source["updates"] == source["adam_step"] == len(source["rows"]) == 144
        assert source["initial"] == manifest["initial"]
        assert all(r["group_uid"] in manifest["folds"][1-i] and r["lr_step"] == 2*r["step"] for r in source["rows"])
        assert set(source["evaluated_ids"]).issubset(manifest["folds"][i])
        source_summary[role] = dict(updates=144, training_groups=len(set(r["group_uid"] for r in source["rows"])), evaluated_groups=len(source["evaluated_ids"]))

    overfit = read(root / "diagnostics_evaluation/overfitting.json")
    curves, residuals, diagnostic_draws = {}, {}, {}
    for point in points:
        captures = [read(verify(run / "diagnostics", r)) for r in manifest["diagnostics"][point]]
        assert [d["epoch"] for d in captures] == list(range(1, 7))
        residuals[point] = []
        fits, caches, valid_metrics = [], [], []
        for d in captures:
            assert d["valid_labels_used"] is False and d["valid_ids"] == col["group_ids"]
            fits.append(np.load(verify(run / "diagnostics", d["fit"]), allow_pickle=False))
            caches.append(np.load(verify(run / "diagnostics", d["cache"]), allow_pickle=False))
            rec = overfit["group_metrics"][f"{point}_epoch{d['epoch']}"]
            valid_metrics.append(np.load(verify(root / "diagnostics_evaluation", rec), allow_pickle=False))
            es = d["teacher_errors"]
            if es:
                keys = [k for k, v in es[0].items() if k != "uid" and isinstance(v, (int, float))]
                residuals[point].append({"epoch": d["epoch"], **{k: float(np.mean([r[k] for r in es])) for k in keys},
                                        "violations_total": sum(r.get("violations", 0) for r in es)})
        for i, domain in enumerate("AB"):
            curves.setdefault(point, {})[domain] = np.stack(valid_metrics)[:, rows[i]].mean(1).tolist()
        subset = [columns.index(k) for k in ("map", "recall_at_5", "log_loss")]
        compare(valid_metrics[-1], matrices[point]["raw"][:, subset])
        last_scores = np.load(verify(run / "diagnostics", captures[-1]["valid_blind"]), allow_pickle=False)
        endpoint_scores = np.load(run / manifest["points"][point]["scores"]["raw"]["path"], allow_pickle=False)
        assert np.array_equal(last_scores, endpoint_scores)
        stage = 1 if point == "A_shared" else 2
        for role, data, domain in [("fit", np.stack(fits), "A" if stage == 1 else "B"), ("cache", np.stack(caches), "A")]:
            if data.shape[1] == 0:
                continue
            key = (stage, role, domain)
            if key not in diagnostic_draws:
                payload = (json.dumps([20260930, "ABC", stage, role, domain, "diagnostics"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
                seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63-1)
                generator = np.random.Generator(np.random.PCG64(seed))
                diagnostic_draws[key] = (generator.integers(data.shape[1], size=(5000, data.shape[1])), generator.integers(20, size=(5000, 20)))
            td, vd = diagnostic_draws[key]
            v = np.stack(valid_metrics)[:, rows[0 if domain == "A" else 1]]
            signs = np.array([1., 1., -1.])
            target = overfit["points"][point][role][domain]
            compare(data.mean(1), target["train_raw_curve"])
            compare(v.mean(1), target["valid_raw_curve"])
            tb = (data[-1]-data[0])*signs
            vb = (v[-1]-v[0])*signs
            for values, draw, field in [(tb, td, "epoch1_to_6_train_benefit"), (vb, vd, "epoch1_to_6_valid_benefit")]:
                compare(values.mean(0), target[field]["mean"])
                compare(np.percentile(values[draw].mean(1), [2.5, 97.5], axis=0).T, target[field]["conditional_95pct_interval"])
            gap_draws = (data[:, td].mean(2)-v[:, vd].mean(2))*signs
            compare(np.percentile(gap_draws[-1]-gap_draws[0], [2.5, 97.5], axis=0).T, target["epoch1_to_6_gap_widening"]["conditional_95pct_interval"])

    with (args.output / "all_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "point_or_contrast", "role", "domain", "metric", "mean", "lower_95", "upper_95"])
        writer.writerows(all_csv)
    summary = dict(absolute=absolute, classification=classification, paired_map_groups=paired_groups,
                   training=train_summary, sources=source_summary, birth=read(run / "sources/birth.json"),
                   valid_curves=curves, residuals=residuals,
                   overfit_interpretations={p: {role: {d: v.get("interpretation") for d, v in domains.items() if isinstance(v, dict)} for role, domains in value.items()} for p, value in overfit["points"].items()})
    write(args.output / "summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"S": "#405f86", "C_cf": "#cf5c37", "C_perm": "#309687"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for i, domain in enumerate("AB"):
        ax = axes[0, i]
        for j, contrast in enumerate(["C_cf-S", "C_cf-C_perm"]):
            r = reported["comparisons"][contrast]["raw"][domain]["map"]
            lo, hi = r["conditional_95pct_interval"]
            ax.errorbar(r["mean"], j, xerr=[[r["mean"]-lo], [hi-r["mean"]]], fmt="o", capsize=5, color=colors["C_cf" if j == 0 else "C_perm"])
        ax.axvline(0, color="#999999", linewidth=1)
        ax.set_yticks([0, 1], ["C_cf - S", "C_cf - C_perm"])
        ax.set_xlim(-0.021, 0.011)
        ax.set_title(("Old domain A" if i == 0 else "New domain B") + ": final MAP difference")
        ax.set_xlabel("Paired mean and conditional 95% interval")
        ax = axes[1, i]
        for arm in ["S", "C_cf", "C_perm"]:
            values = [row[0] for row in curves["B_"+arm][domain]]
            ax.plot(range(1, 7), values, marker="o", label=arm, color=colors[arm])
        ax.axhline(absolute["A_shared"]["raw"][domain]["map"], linestyle="--", color="#888888", label="shared A endpoint")
        ax.set_xticks(range(1, 7)); ax.set_xlabel("B-stage epoch (fixed six points)")
        ax.set_ylabel("MAP"); ax.set_title("A unseen accounts" if i == 0 else "B unseen accounts")
        ax.legend(fontsize=8)
    fig.suptitle("C_cf fixed A-to-B development experiment | complete S comparator")
    fig.savefig(args.output / "map_comparison.png", dpi=180)
    fig.savefig(args.output / "map_comparison.pdf")
    plt.close(fig)
    audit = dict(status="PASS_SAVED_RESULT_AUDIT", observed_at=datetime.now().astimezone().isoformat(), python=platform.python_version(),
                 elapsed_seconds=time.perf_counter()-started, scalar_comparisons=int(scalar_checks), maximum_absolute_error=max(errors),
                 verified_source_count=len(source_files), verified_file_count=len(set(checked_files)),
                 bootstrap="Exact replay of registered PCG64(20260930), no new draw/seed/endpoint", diagnostic_draws="Exact registered group draws reused across arms",
                 access_before=access, access_after=read(root / "access.json"), model_loads=0, memory_deserializations=0, label_parses=0,
                 boundaries="Rechecks saved metrics/statistics/counts, source and execution records; does not independently regenerate metrics from labels or replay model updates.")
    assert audit["access_before"] == audit["access_after"]
    write(args.output / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
