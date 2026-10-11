"""Verify and summarize the completed three-arm run using saved results only."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import time

import numpy as np

ARMS = ("LOGIT0.1", "R0", "S")
ORDERS = ("ABC", "BCA", "CAB")
ROLES = ("raw", "stage-cal", "first-cal", "primary")
TERMS = {
    "O": ((3, 0, .5), (3, 1, .5)),
    "N": ((2, 1, .5), (3, 2, .5)),
    "Z": ((3, 2, 1.),),
    "F_first": ((1, 0, 1.), (3, 0, -1.)),
    "F": ((1, 0, .5), (3, 0, -.5), (2, 1, .5), (3, 1, -.5)),
    "G": ((2, 1, .5), (1, 1, -.5), (3, 2, .5), (2, 2, -.5)),
    "final_all": ((3, 0, 1/3), (3, 1, 1/3), (3, 2, 1/3)),
    "A2": ((2, 1, 1.),),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verified(root, rec):
    path = root / rec["path"]
    payload = path.read_bytes()
    assert len(payload) == rec["bytes"]
    assert hashlib.sha256(payload).hexdigest() == rec["sha256"], str(path)
    return path


def main(root):
    started = time.monotonic()
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    eroot = root / "job/evaluation"
    col = read(eroot / "collected.json")
    ev = read(eroot / "evaluation.json")
    diag = read(eroot / "diagnostics/overfitting.json")
    manifest = read(root / "job/run/manifest.json")
    complete = read(root / "job/completion.json")
    blind = read(root / "job/before_valid.json")
    assert complete["physical_updates"] == 7776 and complete["epoch_diagnostic_points"] == 162
    assert complete["access"] == dict(train=1, valid=1, heldout=0, owners=0)
    assert blind["access"] == dict(train=1, valid=0, heldout=0, owners=0)
    assert blind["states"] == 27 and blind["epochs"] == 162 and blind["initials"] == 2
    verified(root / "job", complete["evaluation"])
    verified(root / "job", complete["overfitting"])
    verified(root / "job", blind["manifest"])
    assert manifest["physical_updates"] == 7776 and manifest["gradient_group_presentations"] == 12960
    assert len(manifest["points"]) == 27 and len(diag["group_metrics"]) == 162
    assert len(manifest["starts"]) == 9 and len(manifest["initial"]) == 2
    assert all(v["full_restore_verified"] for v in manifest["points"].values())
    assert col["source_files"] == ev["source_files"] == manifest["source_files"]
    for rec in col["source_files"]:
        verified(root / "source", rec)
    columns = col["metric_columns"]
    assert len(columns) == 22 and len(set(col["group_ids"])) == 60
    domains = np.array(col["domains"])
    indices = {d: np.flatnonzero(domains == d) for d in "ABC"}
    assert all(len(v) == 20 for v in indices.values())
    rank = [columns.index(n) for n in columns if n not in
            ("brier", "log_loss", "precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc")]
    matrices = {}
    count_sets = 0
    for name, roles in col["points"].items():
        matrices[name] = {}
        for role, rec in roles.items():
            a = np.load(verified(eroot, rec["matrix"]), allow_pickle=False)
            assert a.shape == (60, 22) and a.dtype == np.float64 and np.isfinite(a).all()
            counts = read(verified(eroot, rec["counts"]))
            assert len(counts) == 60
            assert all(v["tp"] + v["fn"] == 20 and v["fp"] + v["tn"] == 358 for v in counts)
            count_sets += 1
            matrices[name][role] = a
        for role in ("stage-cal", "first-cal"):
            assert np.array_equal(matrices[name]["raw"][:, rank], matrices[name][role][:, rank])
        matrices[name]["primary"] = matrices[name]["stage-cal"].copy()
        matrices[name]["primary"][:, rank] = matrices[name]["raw"][:, rank]
    for rec in col["initial"].values():
        a = np.load(verified(eroot, rec["matrix"]), allow_pickle=False)
        assert a.shape == (60, 22) and np.isfinite(a).all()
        counts = read(verified(eroot, rec["counts"]))
        assert len(counts) == 60 and all(v["tp"]+v["fn"] == 20 and v["fp"]+v["tn"] == 358 for v in counts)
        count_sets += 1
    schedules = {}
    update_count = 0
    duration = Counter()
    for name, rec in manifest["training"].items():
        log = read(verified(root / "job/run", rec))
        assert len(log["rows"]) == 288 and log["adam_step"] == log["stage"] * 288
        order, arm, stage = log["order"], log["arm"], log["stage"]
        key = order, stage
        seq = (log["current_ids"], log["history_ids"])
        if key in schedules:
            assert schedules[key] == seq
        schedules[key] = seq
        update_count += len(log["rows"])
        duration[arm] += log["elapsed_seconds"]
    assert update_count == 7776
    # Directly index actual-domain rows and average the three fixed orders.
    # No call to the experiment's endpoint_fields, fields or summarize_field.
    draws = np.random.Generator(np.random.PCG64(20260930)).integers(0, 20, size=(5000, 3, 20))
    references = {}
    for arm in ARMS:
        references[arm] = {}
        for role in ROLES:
            references[arm][role] = {}
            for endpoint, terms in TERMS.items():
                per_order_fields = np.zeros((3, 3, 20, 22))
                for oi, order in enumerate(ORDERS):
                    for stage, arrival, weight in terms:
                        d = order[arrival]
                        values = matrices[f"{order}_{arm}_stage{stage}"][role][indices[d]]
                        per_order_fields[oi, "ABC".index(d)] += weight * values
                if endpoint in ("F_first", "F", "G"):
                    per_order_fields[..., [columns.index("brier"), columns.index("log_loss")]] *= -1
                references[arm][role][endpoint] = per_order_fields
    maximum_error = 0.
    checked_scalars = 0

    def compare(fields, saved):
        nonlocal maximum_error, checked_scalars
        estimates = np.zeros((5000, 22))
        for d in range(3):
            # Collapse order only; resample matching groups once per actual domain.
            groups = fields[:, d].sum(axis=0) / 3
            estimates += groups[draws[:, d]].sum(axis=1) / 20
        interval = np.quantile(estimates, [.025, .975], axis=0, method="linear")
        order_means = fields.sum(axis=1).mean(axis=1)
        means = order_means.mean(axis=0)
        for k, metric in enumerate(columns):
            expected = np.r_[means[k], order_means[:, k], interval[:, k]]
            item = saved[metric]
            observed = np.r_[item["mean"], [item["per_order"][o] for o in ORDERS], item["conditional_95pct_interval"]]
            error = float(np.max(np.abs(expected-observed)))
            maximum_error = max(maximum_error, error)
            checked_scalars += 6
            assert error < 1e-12, (metric, error)

    for arm in ARMS:
        for role in ROLES:
            for ep in TERMS:
                compare(references[arm][role][ep], ev["endpoints"][arm][role][ep])
    for name, candidate, other in (("R0-LOGIT0.1", "R0", "LOGIT0.1"), ("S-R0", "S", "R0"), ("S-LOGIT0.1", "S", "LOGIT0.1")):
        for role in ROLES:
            for ep in TERMS:
                compare(references[candidate][role][ep]-references[other][role][ep], ev["comparisons"][name]["delta"][role][ep])
    curves = []
    counts = {arm: Counter() for arm in ARMS}
    for point, roles in diag["points"].items():
        arm = point.split("_")[1]
        for role, ds in roles.items():
            for domain, result in ds.items():
                if not isinstance(result, dict) or "interpretation" not in result:
                    continue
                for metric, label in result["interpretation"].items():
                    k = result["columns"].index(metric)
                    entry = dict(point=point, arm=arm, role=role, domain=domain, metric=metric, interpretation=label,
                                 train_groups=result["fit_or_cache_groups"], valid_groups=result["valid_groups"],
                                 train_curve=[r[k] for r in result["train_raw_curve"]], valid_curve=[r[k] for r in result["valid_raw_curve"]])
                    for field in ("epoch1_to_6_train_benefit", "epoch1_to_6_valid_benefit", "epoch1_to_6_gap_widening"):
                        entry[field] = dict(mean=result[field]["mean"][k], conditional_95pct_interval=result[field]["conditional_95pct_interval"][k])
                    curves.append(entry)
                    counts[arm][role+"/"+metric+"/"+label] += 1
    write(out / "curves.json", curves)
    overview = {"status":"PASS_SAVED_RESULT_REFERENCE", "scope":"Saved matrices and records only; no source labels, Memory deserialization, model loading or new training.",
                "checked_statistics_scalars":checked_scalars, "maximum_absolute_statistical_error":maximum_error,
                "matrix_count":count_sets, "stage_count":27, "actual_update_rows":update_count,
                "paired_current_and_history_schedules":True, "diagnostic_count":162,
                "train_and_diagnostics_seconds_by_arm":dict(duration),
                "overfit_counts":{a:dict(v) for a,v in counts.items()},
                "map_endpoints":{a:{e:ev["endpoints"][a]["primary"][e]["map"] for e in TERMS} for a in ARMS},
                "map_comparisons":{n:{e:v["delta"]["primary"][e]["map"] for e in TERMS} for n,v in ev["comparisons"].items()},
                "initial_map":{a:{d:v["macro_by_domain"][d]["map"] for d in "ABC"} for a,v in ev["initial_raw"].items()},
                "first_and_final":{}}
    for order in ORDERS:
        overview["first_and_final"][order] = {}
        for arm in ARMS:
            a = ev["absolute_stage_results"]
            overview["first_and_final"][order][arm] = {
                "first_acquired":a[f"{order}_{arm}_stage1"]["raw"]["macro_by_domain"][order[0]]["map"],
                "second_acquired":a[f"{order}_{arm}_stage2"]["raw"]["macro_by_domain"][order[1]]["map"],
                "final":{d:a[f"{order}_{arm}_stage3"]["raw"]["macro_by_domain"][d]["map"] for d in "ABC"}}
    with (out / "all_endpoints.csv").open("w", encoding="utf-8", newline="") as f:
        w=csv.writer(f); w.writerow(("type","method_or_comparison","role","endpoint","metric","mean","lo95","hi95",*ORDERS))
        for kind, entries in (("absolute",ev["endpoints"]),("difference",{n:v["delta"] for n,v in ev["comparisons"].items()})):
            for name, roles in entries.items():
                for role, eps in roles.items():
                    for ep, metrics in eps.items():
                        for metric,v in metrics.items():
                            w.writerow((kind,name,role,ep,metric,v["mean"],*v["conditional_95pct_interval"],*[v["per_order"][o] for o in ORDERS]))
    with (out / "all_stages.csv").open("w", encoding="utf-8", newline="") as f:
        w=csv.writer(f); w.writerow(("point","role","actual_domain",*columns))
        for point,roles in ev["absolute_stage_results"].items():
            for role,result in roles.items():
                for domain,values in result["macro_by_domain"].items():
                    w.writerow((point,role,domain,*[values[c] for c in columns]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"LOGIT0.1":"#6b7280", "R0":"#2563eb", "S":"#059669"}
    fig,axes=plt.subplots(1,3,figsize=(12,4.2))
    names=("R0-LOGIT0.1","S-R0","S-LOGIT0.1")
    for ax,ep in zip(axes,("O","N","Z")):
        for y,n in enumerate(names):
            v=ev["comparisons"][n]["delta"]["primary"][ep]["map"]; lo,hi=v["conditional_95pct_interval"]
            ax.errorbar(v["mean"]*100,y,xerr=[[(v["mean"]-lo)*100],[(hi-v["mean"])*100]],fmt="o",capsize=4,color="#059669" if lo>0 else "#6b7280")
        ax.axvline(0,color="#aaaaaa",lw=1); ax.grid(axis="x",alpha=.2)
        ax.set_yticks(range(3),names); ax.invert_yaxis(); ax.set_title({"O":"Final old domains (O)","N":"New learning (N)","Z":"Final latest domain (Z)"}[ep]); ax.set_xlabel("MAP difference (percentage points)")
    fig.suptitle("Three-arm attribution: paired differences and conditional 95% intervals")
    fig.tight_layout(); fig.savefig(out/"map_comparisons.png",dpi=170); fig.savefig(out/"map_comparisons.svg"); plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(11,9),sharex=True,sharey=True)
    for oi,order in enumerate(ORDERS):
        for di,domain in enumerate("ABC"):
            ax=axes[oi,di]
            for arm in ARMS:
                values=[ev["absolute_stage_results"][f"{order}_{arm}_stage{t}"]["raw"]["macro_by_domain"][domain]["map"] for t in (1,2,3)]
                ax.plot((1,2,3),values,marker="o",label=arm,color=colors[arm])
            ax.axvline(order.index(domain)+1,color="#cccccc",ls=":")
            ax.set_title(f"Order {order}; domain {domain}"); ax.grid(alpha=.2); ax.set_xticks((1,2,3))
            if di==0: ax.set_ylabel("MAP")
            if oi==2: ax.set_xlabel("Completed training stage")
    axes[0,0].legend(); fig.suptitle("Absolute MAP trajectories; dotted line marks domain acquisition")
    fig.tight_layout(); fig.savefig(out/"map_trajectories.png",dpi=150); fig.savefig(out/"map_trajectories.svg"); plt.close(fig)
    overview["elapsed_seconds"] = time.monotonic()-started
    write(out/"summary.json",overview)
    print(json.dumps({k:overview[k] for k in ("status","checked_statistics_scalars","maximum_absolute_statistical_error","elapsed_seconds")},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    main(parser.parse_args().root)
