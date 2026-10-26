"""Bounded Linux-only audit of already opened five-method saved results.

No project imports, labels, model/Memory payloads, inference or new comparisons.
Reconstructs frozen endpoints and paired intervals independently from saved
group metrics; query arrays/counts check the corresponding metric aggregates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

METHODS = ("SEQ_ref", "ER0.1_ref", "LOGIT0.1_ref", "S_tuned", "H_pos_tuned")
ORDERS = ("ABC", "BCA", "CAB")
ROLES = ("raw", "stage-cal", "first-cal")
FIELDS = ("O", "N", "Z", "A_acq", "F_first", "F")
CELLS = ("1_1", "2_1", "2_2", "3_1", "3_2", "3_3")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


MAX_ERROR = 0.
CHECKS = 0


def same(actual, expected):
    global MAX_ERROR, CHECKS
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "Missing-value masks differ"
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.any():
        MAX_ERROR = max(MAX_ERROR, float(np.max(np.abs(a[ok]-b[ok]))))
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12, equal_nan=True)
    CHECKS += 1


def endpoints(cells):
    # Direct endpoint algebra, independent of production field_terms/contributions.
    a, b, c, d, e, f = [cells[key] for key in CELLS]
    result = dict(O=(d+e)/2, N=(c+f)/2, Z=f, A_acq=(a+c)/2,
                  F_first=a-d, F=((a-d)+(c-e))/2)
    if "1_2" in cells:
        result.update(D2=a-b, G2=c-cells["1_2"],
                      D3=((b-d)+(c-e))/2, G3=f-cells["2_3"])
    return result


def count_metrics(row):
    tp, fp, tn, fn = [row[k] for k in ("tp", "fp", "tn", "fn")]
    den = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return [tp/(tp+fp) if tp+fp else None, tp/(tp+fn),
            2*tp/(2*tp+fp+fn), tn/(tn+fp),
            (tp/(tp+fn)+tn/(tn+fp))/2,
            (tp*tn-fp*fn)/math.sqrt(den) if den else None]


def main(campaign, output):
    started = time.monotonic()
    started_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    def stop_at_limit(signum, frame):
        raise TimeoutError("Saved-results audit reached its 240-second bound")
    signal.signal(signal.SIGALRM, stop_at_limit)
    signal.alarm(240)
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    output.mkdir(parents=True, exist_ok=False)
    prior_audit_seconds = sum(read(p)["charged_seconds"] for p in output.parent.glob("audit*/execution.json"))
    status = "FAILED"
    result = {}
    try:
        completion = read(campaign/"completion.json")
        budget = read(campaign/"budget.json")
        assert completion["status"] == "ALL_45_AND_FROZEN_CONFIRMATION_ANALYSIS_COMPLETE"
        assert not budget["active"] and not budget["failed"]
        assert budget["used"]["analysis"]+prior_audit_seconds+240 < budget["limits"]["analysis"]
        saved = read(campaign/"analysis/result.json")
        confirmation = read(campaign/"confirmation/result.json")
        columns = saved["columns"]
        assert len(columns) == 22 and columns[12] == "map"
        ids = [f"{o}_s{s}_{m}" for o in ORDERS for s in (4, 5, 6) for m in METHODS]
        assert set(confirmation["runs"]) == set(completion["run_ids"]) == set(ids)
        assert confirmation["forwards"] == 11520
        assert len(set(confirmation["groups"])) == 120
        domain_ids = {d: [g for g, dom in zip(confirmation["groups"], confirmation["domains"], strict=True)
                          if dom == d] for d in "ABC"}
        assert all(len(v) == 40 for v in domain_ids.values())
        original_gate = read(campaign.parents[2]/"20261023/continual_implementation/formal_gate.json")
        # Exact frozen scientific workspace; never import or change it.
        workspace = campaign.parents[2]/"20261023/continual_implementation/workspace"
        for rec in original_gate["sources"]:
            got = identity(workspace/rec["path"])
            assert (got["bytes"], got["sha256"]) == (rec["bytes"], rec["sha256"])
        assert completion["sources"] == original_gate["sources"]
        arrays, counts, points, records = {}, {}, {}, {}
        query_checks = score_checks = restore_checks = updates = presentations = 0
        memory_bytes = []
        for rid in ids:
            record = read(campaign/rid/"run/completion.json")
            records[rid] = record
            assert record["status"] == "COMPLETE_864_UPDATES_3_ENDPOINTS_VALID_BLIND"
            assert record["updates"] == 864 and record["source_files"] == original_gate["sources"]
            assert record["access"] == dict(heldout=0, owners=0, train=1, valid=0, worker_train_reads=1)
            updates += record["updates"]
            presentations += record["gradient_group_presentations"]
            for stage, point in record["points"].items():
                assert point["adam_step"] == 288*int(stage)
                assert point["full_restore_verified"] and point["inference_restore_verified"]
                restore_checks += 2
                mem = point["memory_summary"]
                if mem is not None:
                    size = mem["bytes"] if "bytes" in mem else mem["serialized_bytes"]+mem["auxiliary_serialized_bytes"]
                    assert size <= 1048576 and len(mem["members"]) == 6
                    memory_bytes.append(size)
            expected = set(CELLS) | ({"1_2", "2_3"} if rid.endswith("SEQ_ref") else set())
            assert set(confirmation["runs"][rid]) == expected
            for role in ROLES:
                dense, plain, role_counts = {}, {}, {}
                for cell, point in confirmation["runs"][rid].items():
                    stage, pos = map(int, cell.split("_"))
                    domain = rid[:3][pos-1]
                    assert point["domain"] == domain and point["ids"] == domain_ids[domain]
                    rec = point["metrics"][role]["matrix"]
                    path = campaign/"confirmation"/rec["path"]
                    got = identity(path)
                    assert (got["bytes"], got["sha256"]) == (rec["bytes"], rec["sha256"])
                    matrix = read(path)
                    a = np.asarray(matrix["values"], dtype=float)
                    assert matrix["columns"] == columns and a.shape == (40, 22)
                    q = np.load(campaign/"confirmation"/point["metrics"][role]["queries"]["path"], allow_pickle=False)
                    assert q.shape == (40, 28, 10)
                    same(q.mean(axis=1), a[:, 12:])
                    query_checks += 1
                    for row, values in zip(matrix["counts"], a, strict=True):
                        assert row["tp"]+row["fn"] == 20 and row["tn"]+row["fp"] == 358
                        same(count_metrics(row), values[6:12])
                    scores = np.load(campaign/"confirmation"/point["scores"][role]["path"], allow_pickle=False)
                    assert scores.shape == (40, 378) and np.isfinite(scores).all()
                    same((scores >= 0).sum(axis=1), [x["tp"]+x["fp"] for x in matrix["counts"]])
                    if role != "raw":
                        raw = np.load(campaign/"confirmation"/point["scores"]["raw"]["path"], allow_pickle=False)
                        cal = record["points"][str(stage)]["maps"]["stage" if role == "stage-cal" else "first"]
                        assert cal["a"] > 0
                        same(scores, cal["a"]*raw.astype(float)+cal["b"])
                    score_checks += 1
                    # Only this actual domain contributes to the corresponding cell.
                    expanded = np.zeros((3, 40, 22), dtype=float)
                    expanded["ABC".index(domain)] = a
                    dense[cell], plain[cell], role_counts[cell] = expanded, a.mean(axis=0), matrix["counts"]
                    same(a.mean(axis=0), saved["per_run_actual_domain_cell_points"][f"{rid}_{role}_{cell}"]["values"])
                arrays[rid, role] = endpoints(dense)
                points[rid, role] = endpoints(plain)
                counts[rid, role] = role_counts
                for field, value in points[rid, role].items():
                    same(value, saved["per_run_endpoint_points"][f"{rid}_{role}"][field])
        # Paired reservoir membership is a saved-metadata check, not payload decoding.
        for order in ORDERS:
            for seed in (4, 5, 6):
                for stage in (1, 2, 3):
                    memberships = [records[f"{order}_s{seed}_{m}"]["points"][str(stage)]["memory_summary"]["members"]
                                   for m in METHODS[1:]]
                    assert all(v == memberships[0] for v in memberships)
        avg = {(m, role, f): np.mean([arrays[rid, role][f] for rid in ids if records[rid]["spec"]["name"] == m], axis=0)
               for m in METHODS for role in ROLES for f in FIELDS}
        for (m, role, field), a in avg.items():
            same(a.mean(axis=1).sum(axis=0), saved["absolute_endpoint_points"][m][role][field])
        # Independent multiplicity-weight implementation of the frozen same draws.
        draws = np.random.Generator(np.random.PCG64(2026102302)).integers(40, size=(5000, 3, 40))
        weights = np.zeros((5000, 3, 40), dtype=float)
        for domain in range(3):
            np.add.at(weights[:, domain], (np.arange(5000)[:, None], draws[:, domain]), 1/40)
        intervals_checked = 0
        for comparison, roles in saved["comparisons"].items():
            left, right = comparison.split("-", 1)
            for role, fields in roles.items():
                for field, metric_rows in fields.items():
                    delta = avg[left, role, field]-avg[right, role, field]
                    missing = np.isnan(delta).any(axis=(0, 1))
                    samples = weights.reshape(5000, 120) @ delta.reshape(120, 22)
                    interval = np.quantile(samples, [.025, .975], axis=0)
                    mean = delta.mean(axis=1).sum(axis=0)
                    for j, name in enumerate(columns):
                        row = metric_rows[name]
                        if missing[j]:
                            assert row["mean"] is None and row["conditional_95pct_interval"] is None
                        else:
                            same([mean[j], *interval[:, j]], [row["mean"], *row["conditional_95pct_interval"]])
                        intervals_checked += 1
        primary = saved["comparisons"]["H_pos_tuned-S_tuned"]["raw"]
        gate = dict(old_final=primary["O"]["map"]["conditional_95pct_interval"][0] > 0,
                    less_forgetting=primary["F"]["map"]["conditional_95pct_interval"][1] < 0,
                    new_mean=primary["N"]["map"]["mean"] >= 0,
                    last_new=primary["Z"]["map"]["mean"] >= 0)
        assert all(gate.values()) == saved["primary_pass"]
        per_pair = {}
        seq = {}
        for o in ORDERS:
            for s in (4, 5, 6):
                h, base = points[f"{o}_s{s}_H_pos_tuned", "raw"], points[f"{o}_s{s}_S_tuned", "raw"]
                per_pair[f"{o}_s{s}"] = {f: float(h[f][12]-base[f][12]) for f in FIELDS}
                seq[f"{o}_s{s}"] = {f: float(v[12]) for f, v in points[f"{o}_s{s}_SEQ_ref", "raw"].items()}
        groupings = {}
        for grouping, labels in (("order", ORDERS), ("seed", (4, 5, 6))):
            groupings[grouping] = {str(label): {f: float(np.mean([v[f] for key, v in per_pair.items()
                    if (key[:3] == label if grouping == "order" else key.endswith(f"s{label}"))])) for f in FIELDS}
                    for label in labels}
        actual_domains = {}
        for domain in "ABC":
            actual_domains[domain] = {}
            for method in METHODS:
                final, acquired, old, fresh = [], [], [], []
                for rid in ids:
                    if records[rid]["spec"]["name"] != method:
                        continue
                    position = rid[:3].index(domain)+1
                    p = saved["per_run_actual_domain_cell_points"]
                    final_value = p[f"{rid}_raw_3_{position}"]["values"][12]
                    final.append(final_value)
                    if position < 3:
                        old.append(final_value)
                        acquired.append(p[f"{rid}_raw_{position}_{position}"]["values"][12])
                    else:
                        fresh.append(final_value)
                actual_domains[domain][method] = dict(final_all_positions=np.mean(final), old_final=np.mean(old),
                      old_acquired=np.mean(acquired), old_forgetting=np.mean(acquired)-np.mean(old), new_last=np.mean(fresh))
        pooled_counts = {}
        # Descriptive pooled classification counts differ from group-macro metrics.
        for method in METHODS:
            pooled_counts[method] = {}
            for role in ROLES:
                total = dict(tp=0, fp=0, tn=0, fn=0)
                for rid in ids:
                    if records[rid]["spec"]["name"] == method:
                        for cell in ("3_1", "3_2", "3_3"):
                            for row in counts[rid, role][cell]:
                                for k in total:
                                    total[k] += row[k]
                pooled_counts[method][role] = dict(counts=total, classification=count_metrics(total))
        old_dev = read(campaign/"old_dev/result.json")
        old_means = {}
        for method in METHODS:
            rows = []
            for rid in ids:
                if records[rid]["spec"]["name"] != method:
                    continue
                cells = {}
                for cell in CELLS:
                    stage, pos = map(int, cell.split("_"))
                    meta = old_dev["runs"][rid][f"{stage}_raw"]["matrix"]
                    matrix = np.asarray(read(campaign/"old_dev"/meta["path"])["values"], dtype=float)
                    cells[cell] = matrix[np.array(old_dev["domains"]) == rid[:3][pos-1]].mean(axis=0)
                rows.append(endpoints(cells))
            old_means[method] = {f: float(np.mean([row[f][12] for row in rows])) for f in FIELDS}
        result = dict(status="SAVED_RESULTS_INDEPENDENT_AUDIT_COMPLETE", source_result=identity(campaign/"analysis/result.json"),
            script=identity(__file__), frozen_source_count=len(original_gate["sources"]), runs=45, updates=updates,
            gradient_group_presentations=presentations, endpoint_restore_flags=restore_checks,
            confirmation_cells=288, metric_matrices=query_checks, score_checks=score_checks,
            intervals_checked=intervals_checked, numeric_checks=CHECKS, max_abs_error=MAX_ERROR,
            memory_bytes_min=min(memory_bytes), memory_bytes_max=max(memory_bytes), primary_conditions=gate,
            primary_pass=all(gate.values()), map_absolute={m: {f: saved["absolute_endpoint_points"][m]["raw"][f][12] for f in FIELDS} for m in METHODS},
            map_comparisons={c: {f: roles["raw"][f]["map"] for f in FIELDS} for c, roles in saved["comparisons"].items()},
            h_minus_s_per_order_seed=per_pair, h_minus_s_groupings=groupings,
            actual_domain_map=actual_domains, seq_same_transition=seq,
            final_all_domains_pooled_counts=pooled_counts, old_development_map=old_means,
            scope="Saved score/metric/query/count arrays only. No label source reopening, model inference, Memory payload, training or new comparison/inference family. Existing paired bootstrap independently reconstructed.",
            limitations="AP/ROC/Brier/logloss are not recomputed from label sources; restore is checked from original receipts, not re-executed. Subgroups/count pools/old-dev summaries are descriptive only.")
        write(output/"audit.json", result)
        with (output/"tables.zh.txt").open("x", encoding="utf-8") as stream:
            stream.write("完整保存指标：固定五系统；macro为群平均，N/A不补零。所有非H-S比较及非主指标区间为探索性条件区间。\n\n")
            stream.write("绝对端点：method\trole\tfield\t"+"\t".join(columns)+"\n")
            for method in METHODS:
                for role in ROLES:
                    for field in FIELDS:
                        row = saved["absolute_endpoint_points"][method][role][field]
                        stream.write("\t".join([method, role, field]+["N/A" if v is None else f"{v:.12g}" for v in row])+"\n")
            stream.write("\n差值：comparison\trole\tfield\tmetric\tmean\tlower\tupper\n")
            for comparison, roles in saved["comparisons"].items():
                for role, fields in roles.items():
                    for field, metrics in fields.items():
                        for metric, row in metrics.items():
                            v = [None]*3 if row["mean"] is None else [row["mean"], *row["conditional_95pct_interval"]]
                            stream.write("\t".join([comparison, role, field, metric]+["N/A" if x is None else f"{x:.12g}" for x in v])+"\n")
        status = "SUCCESS"
    finally:
        elapsed = time.monotonic()-started
        write(output/"execution.json", dict(status=status, started_at=started_at,
            elapsed_seconds=elapsed, charged_seconds=math.ceil(elapsed), analysis_used_before=budget["used"]["analysis"],
            prior_audit_seconds=prior_audit_seconds,
            analysis_used_including_audit=budget["used"]["analysis"]+prior_audit_seconds+math.ceil(elapsed), analysis_limit=1800,
            time_limit=240, rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            threads=1, memory_limit_bytes=4*1024**3, frozen_budget_unchanged=True, script=identity(__file__)))
        signal.alarm(0)
    print(json.dumps({k: result[k] for k in ("status", "runs", "updates", "metric_matrices", "intervals_checked", "max_abs_error", "primary_pass")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.campaign, args.output)
