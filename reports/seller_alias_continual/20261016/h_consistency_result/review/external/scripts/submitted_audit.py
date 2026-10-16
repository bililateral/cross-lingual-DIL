"""Independent H_cons saved-result audit. Linux py310 only; no labels/models/Memory.
Adapted from the closed H audit for H_cons two endpoints, 52 saved S/H inputs,
registered comparisons and corrected per-metric forgetting direction.
No production finalize/blind_gate is invoked; no new statistical choices.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
    before, access, execution = (read(root / n) for n in ("before_valid.json", "access.json", "execution.json"))
    assert before["status"] == "ALL_BLIND_GATES_PASS"
    assert before["access"] == dict(train=1, valid=0, heldout=0, owners=0)
    assert access == completion["access"] == dict(train=1, valid=1, heldout=0, owners=0)
    assert completion["physical_updates"] == manifest["physical_updates"] == 576
    assert completion["gradient_group_presentations"] == manifest["gradient_group_presentations"] == 864
    assert not (root / "failure.json").exists()
    gate = read(args.workspace / execution["gate"]["path"])
    policy = read(args.workspace / "schema/step28_h_consistency_policy.json")
    sources = completion["source_files"]
    assert len(sources) == 44
    for obj in (manifest, col, reported, before, gate):
        assert obj["source_files"] == sources
    assert execution["sources"] == sources
    checked_files = []

    def verify(parent, rec):
        p = parent / rec["path"]
        assert p.stat().st_size == rec["bytes"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"]
        checked_files.append(str(p.relative_to(args.workspace)))
        return p

    for r in sources:
        verify(args.workspace, r)
        verify(run / "sources", r)
    verify(root, before["manifest"])
    baseline = read(args.workspace / "reports/documentation/20261016/h_consistency/baseline_sources.json")
    assert len(baseline["files"]) == 52
    for r in baseline["files"] + baseline["unchanged_direct_sources"]:
        verify(args.workspace, r)
    old_runs = {}
    for info in baseline["baselines"].values():
        old_root = args.workspace / info["root"]
        old_col = read(old_root / "evaluation/collected.json")
        old_manifest = read(old_root / "run/manifest.json")
        assert col["group_ids"] == old_col["group_ids"] and col["domains"] == old_col["domains"] and col["columns"] == old_col["columns"]
        assert manifest["public_inputs"] == old_manifest["public_inputs"]
        assert read(run / "partition.json") == read(old_root / "run/partition.json")
        assert old_manifest["initials"][info["arm"]] == info["initial_identity"] == manifest["initials"]["H_cons"]
        old_runs[info["arm"]] = (old_root, old_col, old_manifest)
    hpoints = ["A_H_cons", "B_H_cons"]
    points = hpoints + [d+"_"+a for a in ("S", "G_half") for d in "AB"]
    roles = ["raw", "stage-cal", "first-cal"]
    contrasts = [("H_cons", "S"), ("H_cons", "G_half")]
    assert set(manifest["points"]) == set(hpoints)
    assert manifest["initials"]["H_cons"] == baseline["baselines"]["H"]["initial_identity"]
    assert manifest["starts"]["H_cons"] == manifest["points"]["A_H_cons"]["continuity"]
    members = manifest["points"]["A_H_cons"]["memory_summary"]["members"]
    rows = [np.flatnonzero(np.array(col["domains"]) == d) for d in "AB"]
    assert list(map(len, rows)) == [20, 20] and len(set(col["group_ids"])) == 40
    columns = col["columns"]
    assert len(columns) == 22
    errors, scalar_checks, all_csv = [], 0, []

    def compare(actual, expected):
        nonlocal scalar_checks
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b))
        assert not np.isinf(a).any() and not np.isinf(b).any()
        difference = float(np.max(np.abs(a[~np.isnan(a)] - b[~np.isnan(b)]))) if (~np.isnan(a)).any() else 0.0
        errors.append(difference)
        scalar_checks += a.size
        assert difference <= 1e-12, difference

    draws = np.random.Generator(np.random.PCG64(20260930)).integers(20, size=(5000, 2, 20))
    frequencies = np.stack([np.stack([np.bincount(d, minlength=20) for d in draws[:, i]]) for i in range(2)])

    def audit_stats(values, target, domain_index, kind, point, role, domain):
        for j, metric in enumerate(columns):
            x, t = values[:, j], target[metric]
            missing = int(np.isnan(x).sum())
            if missing:
                assert t["mean"] is None and t["conditional_95pct_interval"] is None
                assert t["undefined_groups"] == missing and t["total_groups"] == 20
                all_csv.append([kind, point, role, domain, metric, None, None, None])
            else:
                estimates = frequencies[domain_index] @ x / 20
                mean, bounds = x.mean(), np.percentile(estimates, [2.5, 97.5])
                compare([mean, *bounds], [t["mean"], *t["conditional_95pct_interval"]])
                all_csv.append([kind, point, role, domain, metric, mean, *bounds])

    matrices, absolute, classification = {}, {}, {}
    rank_columns = [i for i, k in enumerate(columns) if k not in {
        "brier", "log_loss", "precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc"}]
    for point in points:
        if point in hpoints:
            p, origin, point_col = manifest["points"][point], evaluation, col
        else:
            old_root, old_col, old_manifest = old_runs[point[2:]]
            p, origin, point_col = old_manifest["points"][point], old_root/"evaluation", old_col
        assert p["full_restore_verified"] and p["inference_restore_verified"]
        assert p["adam_step"] == (288 if point.startswith("A_") else 576)
        assert p["memory_summary"]["members"] == members and p["memory_summary"]["seen"] == 48
        assert p["memory_summary"]["bytes"] <= 1048576
        if point in hpoints:
            assert p["training"] == policy["training"] and p["pooling"] == policy["pooling"]
        matrices[point], absolute[point], classification[point] = {}, {}, {}
        for role in roles:
            classification[point][role] = {}
            rec = point_col["points"][point][role]
            saved = read(verify(origin, rec["matrix"]))
            assert saved["columns"] == columns
            matrix = np.asarray(saved["values"], dtype=float)
            assert matrix.shape == (40, 22) and not np.isinf(matrix).any()
            matrices[point][role] = matrix
            queries = np.load(verify(origin, rec["queries"]), allow_pickle=False)
            assert queries.shape == (40, 28, 10)
            compare(queries.mean(1), matrix[:, 12:])
            target = reported["absolute"][point][role]
            for i, domain in enumerate("AB"):
                audit_stats(matrix[rows[i]], target["by_domain"][domain], i, "absolute", point, role, domain)
            compare(matrix.mean(0), [target["macro_all"][k] for k in columns])
            counts = read(verify(origin, rec["counts"]))
            assert len(counts) == 40 and all(r["tp"]+r["fn"] == 20 and r["fp"]+r["tn"] == 358 for r in counts)
            for i, r in enumerate(counts):
                tp, fp, fn, tn = (r[k] for k in ("tp", "fp", "fn", "tn"))
                denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
                rates = [tp/(tp+fp) if tp+fp else None, tp/20, 2*tp/(2*tp+fp+fn), tn/358,
                         (tp/20+tn/358)/2, (tp*tn-fp*fn)/np.sqrt(denom) if denom else None]
                compare(matrix[i, 6:12], rates)
            c = np.asarray([[r[k] for k in ("tp", "fp", "fn", "tn")] for r in counts], np.int64)
            for domain, selected in [("all", np.arange(40)), ("A", rows[0]), ("B", rows[1])]:
                tp, fp, fn, tn = map(int, c[selected].sum(0))
                values = dict(tp=tp, fp=fp, fn=fn, tn=tn, fpr=fp/(fp+tn), recall=tp/(tp+fn),
                              precision=tp/(tp+fp) if tp+fp else None, f1=2*tp/(2*tp+fp+fn))
                for key, value in values.items():
                    compare(value, target["fixed_half_classification"][domain][key])
                classification[point][role][domain] = values
            if point in hpoints:
                scores = np.load(verify(run, p["scores"][role]), allow_pickle=False)
                assert scores.shape == (40, 378) and np.isfinite(scores).all()
                if role == "raw":
                    raw = scores
                    for suffix in ("raw", "calibration"):
                        original = np.load(verify(run, p["scores"][suffix]), allow_pickle=False)
                        for prefix in ("full", "inference"):
                            assert np.array_equal(original, np.load(verify(run, p["scores"][prefix+"-"+suffix]), allow_pickle=False))
                else:
                    affine = p["maps"]["stage" if role == "stage-cal" else "first"]
                    compare(scores, affine["a"] * raw.astype(np.float64) + affine["b"])
                    assert np.array_equal(np.argsort(raw, axis=1, kind="stable"), np.argsort(scores, axis=1, kind="stable"))
                    compare(matrix[:, rank_columns], matrices[point]["raw"][:, rank_columns])
            absolute[point][role] = {d: {k: target["by_domain"][d][k]["mean"] for k in columns} for d in "AB"}
        if point in hpoints:
            mapping = read(verify(run, p["mapping"]))
            assert mapping["status"] == "PASS_CALIBRATION_FIT" and mapping["optimizer_success"] and mapping["a"] > 0

    paired_groups = {}
    for candidate, other in contrasts:
        contrast = candidate+"-"+other
        paired_groups[contrast] = {}
        for role in roles:
            for stage, report_key, kind in [("B", "comparisons", "final_contrast"), ("A", "initial_comparisons", "initial_contrast")]:
                delta = matrices[stage+"_"+candidate][role] - matrices[stage+"_"+other][role]
                for i, domain in enumerate("AB"):
                    audit_stats(delta[rows[i]], reported[report_key][contrast][role][domain], i, kind, contrast, role, domain)
                    if stage == "B" and role == "raw":
                        v = delta[rows[i], columns.index("map")]
                        paired_groups[contrast][domain] = dict(positive=int((v > 0).sum()), negative=int((v < 0).sum()), zero=int((v == 0).sum()), values=v.tolist())
            delta = (matrices["A_"+candidate][role]-matrices["B_"+candidate][role]) - (matrices["A_"+other][role]-matrices["B_"+other][role])
            delta *= np.array([-1. if k in ("brier", "log_loss") else 1. for k in columns])
            audit_stats(delta[rows[0]], reported["forgetting_comparisons"][contrast][role], 0, "forgetting_contrast", contrast, role, "A")
    for arm in ("H_cons", "S", "G_half"):
        for role in roles:
            delta = matrices["B_"+arm][role] - matrices["A_"+arm][role]
            audit_stats(delta[rows[0]], reported["old_changes"][arm][role], 0, "old_change", arm, role, "A")

    training = {p: read(verify(run, manifest["training"][p])) for p in hpoints}
    train_summary = {}
    for point, t in training.items():
        assert t["updates"] == 288 and [r["step"] for r in t["rows"]] == list(range(1, 289))
        if point.startswith("B_"):
            for key in ("members", "seen", "tables", "origins"):
                assert t["memory_before"][key] == t["memory_after"][key]
            history = [r["history"] for r in t["rows"]]
            train_summary[point] = {"nonzero_distillation_steps": sum(r["distillation"] > 0 for r in history),
                "mean_distillation": float(np.mean([r["distillation"] for r in history])),
                "mean_current_supervised": float(np.mean([r["current"]["supervised"] for r in t["rows"]])),
                "mean_history_supervised": float(np.mean([r["supervised"] for r in history]))}
        components = {}
        for role, weight in (("current", 1.), ("history", .1)):
            if role == "history" and point.startswith("A_"):
                continue
            values = [r[role] for r in t["rows"]]
            components[role] = {"mean_JS": float(np.mean([r["consistency"] for r in values])),
                "min_JS": float(min(r["consistency"] for r in values)),
                "max_JS": float(max(r["consistency"] for r in values)),
                "nonzero_JS_steps": sum(r["consistency"] != 0 for r in values),
                "mean_weighted_JS": float(np.mean([r["consistency_contribution"] for r in values])),
                "mean_supervised_contribution": float(np.mean([r["supervised_contribution"] for r in values])),
                "mean_teacher_contribution": float(np.mean([r["teacher_contribution"] for r in values]))}
        train_summary.setdefault(point, {})["components"] = components
        for row in t["rows"]:
            total = 0.
            assert row["training"] == policy["training"] and row["pooling"] == policy["pooling"]
            for role, weight in (("current", 1.), ("history", .1)):
                if role == "history" and point.startswith("A_"):
                    assert "history" not in row
                    continue
                r = row[role]
                assert r["qualification_without_c"] is False and r["mse1"] is None
                assert np.isfinite(r["consistency"]) and r["record_gradient_norm"] > 0
                if role == "current":
                    assert r["mse0"] == 0
                expected = [weight*r["supervised"], weight*r["consistency"], .5*r["mse0"]]
                actual = [r[k] for k in ("supervised_contribution", "consistency_contribution", "teacher_contribution")]
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(r["weighted"], sum(expected), rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(r["distillation"], r["mse0"], rtol=2e-6, atol=2e-6)
                total += sum(expected)
            np.testing.assert_allclose(row["total"], total, rtol=2e-6, atol=2e-6)
        assert t["rows"][-1]["encoder_lr"] == 0 and t["rows"][-1]["head_lr"] == .001
    for stage in "AB":
        def trace(t):
            return (t["current_ids"], t["history_ids"], [[r["step"], r["encoder_lr"], r["head_lr"], r["current"]["assignment_sha256"], r.get("history",{}).get("assignment_sha256")] for r in t["rows"]])
        current = trace(training[stage+"_H_cons"])
        for other in ("S", "G_half"):
            old_root = old_runs[other][0]
            assert current == trace(read(old_root/"run/updates"/(stage+"_"+other+".json")))
    assert reported["success"]["passed"] == all(reported["comparisons"]["H_cons-S"]["raw"][d]["map"]["conditional_95pct_interval"][0] > 0 for d in "AB")
    assert reported["success"]["direct_H_new_B_positive"] == (reported["comparisons"]["H_cons-G_half"]["raw"]["B"]["map"]["conditional_95pct_interval"][0] > 0)

    independent_decisions, diagnostic_summary = {}, {}
    overfit = read(root / "diagnostics_evaluation/overfitting.json")
    curves, residuals, diagnostic_draws = {}, {}, {}
    for point in hpoints:
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
                residuals[point].append({"epoch": d["epoch"], **{k: float(np.mean([r[k] for r in es])) for k in keys}})
        for i, domain in enumerate("AB"):
            curves.setdefault(point, {})[domain] = np.stack(valid_metrics)[:, rows[i]].mean(1).tolist()
        subset = [columns.index(k) for k in ("map", "recall_at_5", "log_loss")]
        compare(valid_metrics[-1], matrices[point]["raw"][:, subset])
        last_scores = np.load(verify(run / "diagnostics", captures[-1]["valid_blind"]), allow_pickle=False)
        endpoint_scores = np.load(run / manifest["points"][point]["scores"]["raw"]["path"], allow_pickle=False)
        assert np.array_equal(last_scores, endpoint_scores)
        stage = 1 if point.startswith("A_") else 2
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
            gap = (data.mean(1)-v.mean(1))*signs
            gap_change = gap[-1]-gap[0]
            gap_ci = np.percentile(gap_draws[-1]-gap_draws[0], [2.5, 97.5], axis=0).T
            train_ci = np.percentile(tb[td].mean(1), [2.5, 97.5], axis=0).T
            valid_ci = np.percentile(vb[vd].mean(1), [2.5, 97.5], axis=0).T
            compare(gap, target["benefit_gap_curve"])
            compare(gap_change, target["epoch1_to_6_gap_widening"]["mean"])
            compare(np.percentile(gap_draws,[2.5,97.5],axis=1).transpose(1,2,0),target["benefit_gap_conditional_intervals"])
            decisions = {}
            for k, metric in enumerate(("map", "recall_at_5", "log_loss")):
                point_signal = tb[:,k].mean() > 0 and vb[:,k].mean() < 0 and gap_change[k] > 0
                supported = train_ci[k,0] > 0 and valid_ci[k,1] < 0 and gap_ci[k,0] > 0
                decisions[metric] = "有相应迹象" if supported else "证据不足" if point_signal else "未见明确迹象"
            assert decisions == target["interpretation"]
            independent_decisions.setdefault(point,{})[role] = {domain:decisions}
            diagnostic_summary.setdefault(point,{})[role] = {"domain":domain,"train_benefit":tb.mean(0).tolist(),"valid_benefit":vb.mean(0).tolist(),"gap_widening":gap_change.tolist(),"gap_interval":gap_ci.tolist(),"interpretation":decisions}


    with (args.output / "all_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "point_or_contrast", "role", "domain", "metric", "mean", "lower_95", "upper_95"])
        writer.writerows(all_csv)
    summary = dict(absolute=absolute, classification=classification, paired_map_groups=paired_groups,
                   training=train_summary, diagnostic_summary=diagnostic_summary, valid_curves=curves,
                   residuals=residuals, overfit_interpretations=independent_decisions,
                   success=reported["success"], baseline_files=52, forgetting_sign="High-good A-B; Brier/log-loss B-A; candidate minus other")
    write(args.output / "summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"H_cons-S":"#405f86", "H_cons-G_half":"#309687"}
    fig, axes = plt.subplots(2, 2, figsize=(11,7.5), constrained_layout=True)
    for i, domain in enumerate("AB"):
        ax = axes[0,i]
        for j, contrast in enumerate(("H_cons-S","H_cons-G_half")):
            r = reported["comparisons"][contrast]["raw"][domain]["map"]
            lo,hi=r["conditional_95pct_interval"]
            ax.errorbar(r["mean"],j,xerr=[[r["mean"]-lo],[hi-r["mean"]]],fmt="o",capsize=5,color=colors[contrast])
        ax.axvline(0,color="#999999",linewidth=1)
        ax.set_yticks(range(2),["H_cons - S","H_cons - H"])
        ax.set_title(("Old domain A" if i==0 else "New domain B")+": final MAP difference")
        ax.set_xlabel("Paired mean and conditional 95% interval")
        ax=axes[1,i]
        target=overfit["points"]["B_H_cons"]["cache" if domain=="A" else "fit"][domain]
        ax.plot(range(1,7),[r[0] for r in target["train_raw_curve"]],"o-",label="H_cons cache A" if domain=="A" else "H_cons fit B",color="#cf5c37")
        ax.plot(range(1,7),[r[0] for r in target["valid_raw_curve"]],"o-",label="H_cons development "+domain,color="#405f86")
        ax.set_title("During B training: "+domain+" MAP")
        ax.set_xlabel("Epoch");ax.set_ylabel("MAP");ax.legend()
    fig.suptitle("H_cons: one fixed seed, A to B, reused development groups")
    fig.savefig(args.output / "comparison.png", dpi=180)
    fig.savefig(args.output / "comparison.pdf")
    plt.close(fig)
    audit = dict(status="PASS_SAVED_RESULT_AUDIT", python=platform.python_version(), observed_epoch=time.time(),
                 elapsed_seconds=time.perf_counter()-started, source_count=44,
                 checked_files=len(set(checked_files)), scalar_checks=scalar_checks,
                 maximum_absolute_difference=max(errors), statistics_rows=len(all_csv),
                 diagnostics=12, diagnostic_decisions=9, updates=576,
                 baseline_files=52, success=reported["success"],
                 limits="Saved matrices/counts/queries/scores only; no labels/model/Memory loads. Reconstructs fixed registered bootstrap, not additional resampling choices. AP/ROC/Brier/log-loss not regenerated from unavailable labels. Qualification evidence reused; no new scientific candidate.")
    write(args.output / "audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
