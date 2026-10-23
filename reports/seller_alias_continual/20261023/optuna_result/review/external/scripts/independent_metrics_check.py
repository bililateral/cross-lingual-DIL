"""Independent review of already-open saved metrics; never reads labels or models.

Usage: python independent_metrics_check.py INPUT_ROOT OUTPUT_DIRECTORY
Only Python stdlib and NumPy are imported. No submitted project code is executed.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
START = time.perf_counter()
READS = {}
FAILURES = []
CHECKS = {}
SCALARS = {}
MAXERR = {}
RUN_ROWS = []
METRIC_ROWS = []
DIAG_ROWS = []
SUBGROUP_ROWS = []
ROLE_ROWS = []
ARRAYS = {}
DIAGNOSTICS = {}


def touch(p, purpose):
    p = Path(p)
    key = str(p.relative_to(ROOT))
    row = READS.setdefault(key, {"path": key, "mode": "machine_parse_and_check", "purposes": set()})
    row["purposes"].add(purpose)
    return p


def js(p, purpose="saved_json"):
    return json.loads(touch(p, purpose).read_text(encoding="utf-8"))


def arr(p, purpose="saved_array"):
    return np.load(touch(p, purpose), allow_pickle=False)


def ok(condition, category, where):
    CHECKS[category] = CHECKS.get(category, 0) + 1
    if not condition:
        FAILURES.append({"category": category, "where": where})


def close(a, b, category, where, tolerance=1e-12):
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    SCALARS[category] = SCALARS.get(category, 0) + x.size
    shape = x.shape == y.shape
    finite = shape and not np.isinf(x).any() and not np.isinf(y).any()
    nan_equal = finite and np.array_equal(np.isnan(x), np.isnan(y))
    error = float(np.max(np.abs(x[~np.isnan(x)]-y[~np.isnan(y)]))) if nan_equal and (~np.isnan(x)).any() else 0.0
    MAXERR[category] = max(MAXERR.get(category, 0.0), error)
    ok(shape and finite and nan_equal and error <= tolerance, category, where)


def linked(base, record, purpose):
    p = touch(base/record["path"], purpose)
    b = p.read_bytes()
    ok(len(b) == record["bytes"] and hashlib.sha256(b).hexdigest() == record["sha256"], "linked_file_identity", str(p.relative_to(ROOT)))
    return p


def rates(ct):
    tp, fp, fn, tn = map(int, ct)
    den = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return [tp/(tp+fp) if tp+fp else np.nan, tp/(tp+fn),
            2*tp/(2*tp+fp+fn), tn/(tn+fp), (tp/(tp+fn)+tn/(tn+fp))/2,
            (tp*tn-fp*fn)/np.sqrt(den) if den else np.nan]


DRAWS = np.random.Generator(np.random.PCG64(20260930)).integers(20, size=(5000, 2, 20))


def summary_check(values, target, d, category, where):
    v = np.asarray(values, dtype=float)
    if np.isnan(v).any():
        ok(target["mean"] is None and target["conditional_95pct_interval"] is None,
           category+"_NA", where)
        if "undefined_groups" in target:
            ok(target["undefined_groups"] == int(np.isnan(v).sum()), category+"_NA_count", where)
        return [None, None, None]
    boot = v[DRAWS[:, "AB".index(d)]].mean(axis=1)
    result = [float(v.mean()), *np.quantile(boot, [0.025, 0.975]).tolist()]
    close(result, [target["mean"], *target["conditional_95pct_interval"]], category, where)
    return result


jobs = sorted(p for p in (ROOT/"result/job").iterdir() if p.is_dir() and (p/"evaluation/collected.json").exists())
ok(len(jobs) == 45, "run_count", "saved evaluation directories")
reference = None
for job in jobs:
    run_id = job.name
    col = js(job/"evaluation/collected.json", "group_column_alignment")
    evaluation = js(job/"evaluation/evaluation.json", "per_run_aggregate_and_subgroup_validation")
    spec = js(job/"execution.json", "trial_identity")["trial"]
    arm, seed = spec["arm"], spec["seed"]
    population = {k: col[k] for k in ("group_ids", "domains", "columns")}
    if reference is None:
        reference = population
    ok(population == reference, "fixed_population_alignment", run_id)
    columns = col["columns"]
    ok(len(columns) == 22 and len(col["group_ids"]) == len(set(col["group_ids"])) == 40,
       "fixed_population_shape", run_id)
    partition = js(job/"run/partition.json", "saved_group_order_only")
    ok([r["group_uid"] for r in partition["development"]] == col["group_ids"], "partition_order", run_id)
    rows = {d: np.flatnonzero(np.asarray(col["domains"]) == d) for d in "AB"}
    ok(all(len(x) == 20 for x in rows.values()), "domain_group_counts", run_id)
    deg = arr(linked(job/"evaluation", col["query_positive_counts"], "positive_count_aggregate"))
    ok(deg.shape == (40, 28) and np.all((deg == 1).sum(1) == 16) and np.all((deg == 2).sum(1) == 12),
       "positive_count_contract", run_id)
    mats, qs = {}, {}
    for point, records in col["points"].items():
        phase = point[0]
        pt = js(job/"run/points"/(point+".json"), "endpoint_score_maps_and_restore_records")
        maps = {"stage-cal": pt["maps"]["stage"], "first-cal": pt["maps"]["first"]}
        score_arrays = {k: arr(linked(job/"run", rec, "endpoint_saved_scores")) for k, rec in pt["scores"].items()}
        for key, values in score_arrays.items():
            expected = (12, 378) if "calibration" in key else (40, 378)
            ok(values.shape == expected and np.isfinite(values).all(), "score_shape_finite", run_id+"/"+point+"/"+key)
        for prefix in ("full", "inference"):
            close(score_arrays[prefix+"-raw"], score_arrays["raw"], "restore_raw_scores", run_id+"/"+point)
            close(score_arrays[prefix+"-calibration"], score_arrays["calibration"], "restore_calibration_scores", run_id+"/"+point)
        for role, mapping in maps.items():
            ok(mapping["a"] > 0 and np.isfinite([mapping["a"], mapping["b"]]).all(), "calibration_positive_slope", run_id+"/"+point+"/"+role)
            expected = score_arrays["raw"].astype(float)*mapping["a"]+mapping["b"]
            close(expected, score_arrays[role], "calibration_affine_transform", run_id+"/"+point+"/"+role)
            ok(np.array_equal(np.argsort(score_arrays["raw"],axis=1,kind="stable"), np.argsort(score_arrays[role],axis=1,kind="stable")),
               "calibration_order", run_id+"/"+point+"/"+role)
        mats[point], qs[point] = {}, {}
        for role, records_role in records.items():
            label = run_id+"/"+point+"/"+role
            matrix_file = js(linked(job/"evaluation", records_role["matrix"], "metric_matrix"))
            m = np.asarray(matrix_file["values"], dtype=float)
            q = arr(linked(job/"evaluation", records_role["queries"], "per_query_ranking"))
            counts = js(linked(job/"evaluation", records_role["counts"], "confusion_counts"))
            ct = np.asarray([[r[k] for k in ("tp","fp","fn","tn")] for r in counts], dtype=int)
            ok(matrix_file["columns"] == columns and m.shape == (40,22) and q.shape == (40,28,10), "metric_shapes", label)
            ok(np.isfinite(q).all() and not np.isinf(m).any(), "metric_finiteness", label)
            ok(np.all(ct >= 0) and np.all(ct[:,0]+ct[:,2] == 20) and np.all(ct[:,1]+ct[:,3] == 358), "confusion_margins", label)
            close(q.mean(1), m[:,12:], "query_to_group", label)
            close(np.asarray([rates(row) for row in ct]), m[:,6:12], "confusion_to_metrics", label)
            close((score_arrays[role] >= 0).sum(1), ct[:,0]+ct[:,1], "score_to_positive_prediction_count", label)
            target = evaluation["absolute"][point][role]
            for d in "AB":
                for k, metric in enumerate(columns):
                    stats = summary_check(m[rows[d],k], target["by_domain"][d][metric], d, "per_run_domain_summary", label+"/"+d+"/"+metric)
                    METRIC_ROWS.append([run_id, point, role, d, metric, *stats])
                expected_count = ct[rows[d]].sum(0)
                pooled = target["fixed_half_classification"][d]
                close(expected_count, [pooled[k] for k in ("tp","fp","fn","tn")], "pooled_counts", label+"/"+d)
                r = rates(expected_count)
                close([r[0],r[1],r[2],1-r[3]], [pooled[k] for k in ("precision","recall","f1","fpr")], "pooled_rates", label+"/"+d)
            for k, metric in enumerate(columns):
                expected_mean = None if np.isnan(m[:,k]).any() else float(m[:,k].mean())
                close(expected_mean, target["macro_all"][metric], "macro_all", label+"/"+metric)
                ok(target["macro_undefined_groups"][metric] == int(np.isnan(m[:,k]).sum()), "undefined_group_count", label+"/"+metric)
            mats[point][role], qs[point][role] = m, q
            ROLE_ROWS.append([run_id, point, role, int((ct[:,0]+ct[:,1]).sum()), int(np.isnan(m).sum()), float(score_arrays[role].min()), float(score_arrays[role].max())])
        for role in ("stage-cal","first-cal"):
            close(mats[point][role][:, [0,1,2,3,*range(12,22)]], mats[point]["raw"][:, [0,1,2,3,*range(12,22)]], "calibration_ranking_invariant", run_id+"/"+point+"/"+role)
            close(qs[point][role], qs[point]["raw"], "calibration_queries_invariant", run_id+"/"+point+"/"+role)
        for count in (1,2):
            subgroup = np.sum(qs[point]["raw"][:,:,0]*(deg==count),axis=1)/(deg==count).sum(1)
            for d in "AB":
                stats = summary_check(subgroup[rows[d]], evaluation["query_positive_count_MAP"][point][str(count)][d],d,"subgroup_per_run",run_id+"/"+point+"/"+str(count)+"/"+d)
                SUBGROUP_ROWS.append([run_id,point,count,d,*stats])
        if spec["phase"] == "replication":
            ARRAYS[arm,seed,phase] = mats[point]
    for role in ("raw","stage-cal","first-cal"):
        delta = mats["B_"+arm][role]-mats["A_"+arm][role]
        for k, metric in enumerate(columns):
            summary_check(delta[rows["A"],k],evaluation["old_changes"][role][metric],"A","old_change_per_run",run_id+"/"+role+"/"+metric)
    overfit = js(job/"diagnostics_evaluation/overfitting.json", "diagnostic_summary")
    all_diag = {}
    for phase in "AB":
        point = phase+"_"+arm
        entries = [js(job/"run/diagnostics"/(point+f"_epoch{e}.json"), "diagnostic_capture_metadata") for e in range(1,7)]
        fields = ("fit_ids","fit_domains","cache_ids","cache_domains","valid_ids")
        ok(all(all(ent[k] == entries[0][k] for k in fields) for ent in entries),"diagnostic_population_stability",run_id+"/"+point)
        ok(entries[0]["valid_ids"] == col["group_ids"],"diagnostic_valid_order",run_id+"/"+point)
        ok([e["epoch"] for e in entries] == list(range(1,7)),"diagnostic_epoch_coverage",run_id+"/"+point)
        curves = {"fit":[],"cache":[],"valid":[]}
        for epoch, ent in enumerate(entries,1):
            for role in ("fit","cache"):
                x = arr(linked(job/"run/diagnostics", ent[role], "diagnostic_fit_or_cache_group_metrics"))
                ok(x.shape == (len(ent[role+"_ids"]),3) and np.isfinite(x).all(),"diagnostic_metric_shape",run_id+"/"+point+"/"+role)
                curves[role].append(x)
            x = arr(linked(job/"run/diagnostics", ent["valid_blind"], "diagnostic_saved_logits"))
            ok(x.shape == (40,378) and np.isfinite(x).all(),"diagnostic_score_shape",run_id+"/"+point)
            vr = overfit["group_metrics"][point+f"_epoch{epoch}"]
            vm = arr(linked(job/"diagnostics_evaluation", vr, "diagnostic_valid_group_metrics"))
            ok(vm.shape == (40,3) and np.isfinite(vm).all(),"diagnostic_valid_metric_shape",run_id+"/"+point)
            curves["valid"].append(vm)
            if epoch == 6:
                endpoint = arr(job/"run/scores"/(point+"_raw.npy"), "diagnostic_endpoint_score_alignment")
                close(x,endpoint,"epoch6_endpoint_scores",run_id+"/"+point)
                close(vm,mats[point]["raw"][:,[12,16,5]],"epoch6_endpoint_metrics",run_id+"/"+point)
        for role, domain in (("fit",phase),("cache","A")):
            if role == "cache" and phase == "A":
                ok(overfit["points"][point][role] == {"status":"NOT_APPLICABLE_FIRST_STAGE"},"diagnostic_first_stage_NA",run_id)
                continue
            train = np.stack(curves[role]); valid = np.stack(curves["valid"])[:,rows[domain]]
            target = overfit["points"][point][role][domain]
            train_mean, valid_mean = train.mean(1), valid.mean(1)
            signs = np.array([1.,1.,-1.])
            gap = (train_mean-valid_mean)*signs
            tchange = (train_mean[-1]-train_mean[0])*signs
            vchange = (valid_mean[-1]-valid_mean[0])*signs
            close(train_mean,target["train_raw_curve"],"diagnostic_train_curves",run_id+"/"+point+"/"+role)
            close(valid_mean,target["valid_raw_curve"],"diagnostic_valid_curves",run_id+"/"+point+"/"+role)
            close(gap,target["benefit_gap_curve"],"diagnostic_benefit_gap",run_id+"/"+point+"/"+role)
            for key,value in (("train",tchange),("valid",vchange),("gap_widening",gap[-1]-gap[0])):
                field = "epoch1_to_6_"+(key+"_benefit" if key in ("train","valid") else key)
                close(value,target[field]["mean"],"diagnostic_epoch_change",run_id+"/"+point+"/"+role+"/"+field)
            all_diag[phase,role,domain] = target
            for k, metric in enumerate(("map","recall_at_5","log_loss")):
                DIAG_ROWS.append([run_id,phase,role,domain,metric,train.shape[1],valid.shape[1],tchange[k],vchange[k],gap[-1,k]-gap[0,k]])
        ok(all(ent["valid_labels_used"] is False for ent in entries),"diagnostic_blind_capture_marker",run_id+"/"+point)
    if spec["phase"] == "replication":
        DIAGNOSTICS[arm,seed] = all_diag
    RUN_ROWS.append([run_id,arm,seed,len(col["points"]),6,12])

joint = js(ROOT/"result/job/joint/evaluation.json", "replication_three_seed_aggregation")
for arm in ("S","S_pos","H_pos"):
    for phase in "AB":
        for role in ("raw","stage-cal","first-cal"):
            mean = np.mean([ARRAYS[arm,s,phase][role] for s in (1,2,3)],axis=0)
            for d in "AB":
                for k,metric in enumerate(columns):
                    summary_check(mean[rows[d],k],joint["three_seed_absolute"][arm][phase][role][d][metric],d,"three_seed_absolute",f"{arm}/{phase}/{role}/{d}/{metric}")
    for phase,role,domain in (("A","fit","A"),("B","fit","B"),("B","cache","A")):
        target = joint["diagnostics"][arm]["three_fixed_seed_point_means"][phase][role][domain]
        components = [DIAGNOSTICS[arm,s][phase,role,domain] for s in (1,2,3)]
        for field in ("train_raw_curve","valid_raw_curve","benefit_gap_curve"):
            close(np.mean([c[field] for c in components],axis=0),target[field],"three_seed_diagnostic_mean",f"{arm}/{phase}/{role}/{field}")
        for field in ("epoch1_to_6_train_benefit","epoch1_to_6_valid_benefit","epoch1_to_6_gap_widening"):
            close(np.mean([c[field]["mean"] for c in components],axis=0),target[field],"three_seed_diagnostic_change",f"{arm}/{phase}/{role}/{field}")

# Check the delivered CSV/Markdown views against independently checked records.
with touch(ROOT/"result/analysis/all_run_metrics.csv", "provided_CSV_against_recomputed_metrics").open(encoding="utf-8-sig",newline="") as f:
    delivered = list(csv.DictReader(f))
saved_index = {(r["run"],r["endpoint"],r["role"],r["domain"],r["metric"]):r for r in delivered}
ok(len(delivered) == len(saved_index) == len(METRIC_ROWS), "delivered_metric_CSV_coverage", "result/analysis/all_run_metrics.csv")
for run_id,point,role,domain,metric,mean,low,high in METRIC_ROWS:
    r=saved_index[run_id,point[0],role,domain,metric]
    close([mean,low,high],[None if r[k]=="" else float(r[k]) for k in ("mean","low","high")],
          "delivered_metric_CSV_values", "/".join((run_id,point,role,domain,metric)))
phase=role=None
table_cells=0
table_path=touch(ROOT/"result/tables.zh.md","provided_Markdown_table_against_verified_three_seed_mean")
for line in table_path.read_text(encoding="utf-8").splitlines():
    if line.startswith("## "):
        phase=line.split()[1];role=line.split(" / ")[1]
    if not line.startswith("| "):
        continue
    parts=[p.strip().replace("**","") for p in line.strip().strip("|").split("|")]
    metric=parts[0]
    if metric not in columns:
        continue
    for text_value,(arm,domain) in zip(parts[1:],[(a,d) for d in "AB" for a in ("S","S_pos","H_pos")]):
        value=joint["three_seed_absolute"][arm][phase][role][domain][metric]["mean"]
        expected="N/A" if value is None else f"{value:.6f}"
        ok(text_value==expected,"delivered_Markdown_table_values",f"{phase}/{role}/{domain}/{arm}/{metric}")
        table_cells+=1
ok(table_cells==792,"delivered_Markdown_table_coverage","result/tables.zh.md")

def export(name, headers, data):
    with (OUT/name).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(data)

export("run_metric_checks.csv",["run","arm","seed","endpoints","role_sets","diagnostic_captures"],RUN_ROWS)
export("independent_all_run_metrics.csv",["run","point","role","domain","metric","mean","low","high"],METRIC_ROWS)
export("independent_diagnostics.csv",["run","phase","role","domain","metric","train_groups","valid_groups","train_benefit","valid_benefit","gap_widening"],DIAG_ROWS)
export("independent_subgroup_metrics.csv",["run","point","positive_count","domain","mean","low","high"],SUBGROUP_ROWS)
export("score_role_checks.csv",["run","point","role","predicted_positive_total","undefined_metric_cells","minimum_logit","maximum_logit"],ROLE_ROWS)
scope = []
for path, row in sorted(READS.items()):
    p=ROOT/path
    scope.append({**row,"purposes":sorted(row["purposes"]),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/"machine_read_scope.json").write_text(json.dumps(scope,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
report = {"status":"PASS" if not FAILURES else "FAIL","environment":"Isolated external-review workspace; not project Linux execution or native qualification", "formal_label_reads":0,"model_loads":0,"submitted_code_executed":False,"run_count":len(jobs),"endpoint_role_sets":len(ROLE_ROWS),"domain_metric_rows":len(METRIC_ROWS),"diagnostic_captures":sum(r[-1] for r in RUN_ROWS),"diagnostic_comparison_rows":len(DIAG_ROWS),"subgroup_rows":len(SUBGROUP_ROWS),"parsed_files":len(READS),"check_counts":CHECKS,"numeric_scalar_comparisons":SCALARS,"maximum_absolute_error_by_category":MAXERR,"failures":FAILURES,"elapsed_seconds":time.perf_counter()-START,"limits":"Consistency and arithmetic from saved scores, group/query metrics and counts; no reconstruction of label-level AP/AUC/Brier/log-loss or fresh model inference."}
(OUT/"independent_metrics_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
