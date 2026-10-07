#!/usr/bin/env python3
"""Independent saved-output audit. No project imports, fitting, or label reads.

Input: the extracted review_input.zip. Output: JSON evidence in an audit directory.
Only public group IDs, saved scores, metric matrices, aggregate confusion counts,
calibration parameters, and receipts are read. Pair labels cannot be reconstructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.input_root.resolve()
    f = root / "reports/seller_alias_continual/20261007/function_memory_result"
    ref = root / "reports/seller_alias_continual/20261004/logit_low_result/job/evaluation"
    ev = f / "job/evaluation"
    inputs, checks, failures = {}, Counter(), []
    max_error = {}

    def record(p):
        p = p.resolve()
        b = p.read_bytes()
        inputs[str(p.relative_to(root))] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
        return b

    def read(p):
        return json.loads(record(p))

    def check(ok, kind, detail):
        checks[kind] += 1
        if not bool(ok):
            failures.append({"kind": kind, "detail": detail})

    def near(a, b, kind, detail, tol=1e-12):
        aa, bb = np.asarray(a), np.asarray(b)
        if aa.shape != bb.shape:
            check(False, kind, detail + ": shape")
            return
        err = float(np.max(np.abs(aa.astype(float) - bb.astype(float)))) if aa.size else 0.0
        max_error[kind] = max(max_error.get(kind, 0.0), err)
        checks[kind + "_numeric_values"] += aa.size
        check(np.isfinite(err) and err <= tol, kind, detail)

    def verify(base, rec):
        p = base / rec["path"]
        b = record(p)
        check(len(b) == rec["bytes"] and hashlib.sha256(b).hexdigest() == rec["sha256"],
              "file_record", str(p.relative_to(root)))
        return p

    def load(p):
        record(p)
        return np.load(p, allow_pickle=False)

    col = read(ev / "collected.json")
    previous = read(ref / "collected.json")
    shared = read(ref / "reference/collected.json")
    saved = read(ev / "evaluation.json")
    partition = read(f / "job/run/partition.json")
    cols = col["metric_columns"]
    domains = np.asarray(col["domains"])
    check(len(cols) == 22 and len(set(cols)) == 22, "columns", "22 unique names")
    check(len(col["group_ids"]) == len(set(col["group_ids"])) == 60, "groups", "60 unique IDs")
    for name, c in [("reference", previous), ("shared", shared)]:
        for key in ("group_ids", "domains", "metric_columns"):
            check(c[key] == col[key], "alignment", name + ":" + key)
    check(partition["development"] == [dict(domain=d, group_uid=g) for d, g in zip(col["domains"], col["group_ids"])],
          "partition_alignment", "actual domain and group ID in stored row order")
    for d in "ABC":
        check(int(np.sum(domains == d)) == 20, "domain_count", d)

    class_names = ["precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc"]
    non_rank = class_names + ["brier", "log_loss"]
    rank_cols = [cols.index(k) for k in cols if k not in non_rank]
    loaded, all_counts, details = {}, {}, []

    def classification(row):
        tp, fp, fn, tn = [float(row[k]) for k in ("tp", "fp", "fn", "tn")]
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        recall, specificity = tp / (tp + fn), tn / (tn + fp)
        return dict(precision=tp / (tp + fp) if tp + fp else 0., recall=recall,
                    f1=2 * tp / (2 * tp + fp + fn), specificity=specificity,
                    balanced_accuracy=(recall + specificity) / 2,
                    mcc=(tp * tn - fp * fn) / den if den else 0.)

    def pooled(rows):
        t = {k: sum(v[k] for v in rows) for k in ("tp", "fp", "fn", "tn")}
        c = classification(t)
        return {**t, "fpr": t["fp"] / (t["fp"] + t["tn"]),
                **{k: c[k] for k in ("precision", "recall", "f1")}}

    sets = []
    for name, roles in col["points"].items():
        for role, rec in roles.items():
            sets.append(("function_memory", ev, name, role, rec))
    for order in ("ABC", "BCA", "CAB"):
        for stage in (1, 2, 3):
            base = ref / "reference" if stage == 1 else ref
            collection = shared if stage == 1 else previous
            name = order + "_shared" if stage == 1 else f"{order}_logit_tenth_stage{stage}"
            for role, rec in collection["points"][name].items():
                sets.append(("logit0.1", base, name, role, rec))
    check(sum(v[0] == "function_memory" for v in sets) == 28, "collection_count", "candidate")
    check(sum(v[0] == "logit0.1" for v in sets) == 27, "collection_count", "reference")

    for arm, base, name, role, rec in sets:
        key = (arm, name, role)
        mat = load(verify(base, rec["matrix"]))
        counts = read(verify(base, rec["counts"]))
        loaded[key], all_counts[key] = mat, counts
        label = "/".join(key)
        check(mat.shape == (60, 22) and mat.dtype == np.dtype("float64") and np.isfinite(mat).all(), "matrix_shape_type", label)
        check(len(counts) == 60, "counts_shape", label)
        for i, c in enumerate(counts):
            check(set(c) == {"tp", "fp", "fn", "tn"} and all(type(v) is int and v >= 0 for v in c.values())
                  and c["tp"] + c["fn"] == 20 and c["fp"] + c["tn"] == 358,
                  "group_count_conservation", f"{label}/{i}")
            computed = classification(c)
            near([computed[k] for k in class_names], mat[i, [cols.index(k) for k in class_names]],
                 "classification_from_saved_counts", f"{label}/{i}")
        details.append({"arm": arm, "point": name, "role": role, "pooled": pooled(counts),
                        "zero_prediction_groups": sum(c["tp"] + c["fp"] == 0 for c in counts),
                        "macro_precision": float(mat[:, cols.index("precision")].mean())})
        if arm == "function_memory":
            absolute = saved["absolute_stage_results"][name][role]
            near([absolute["macro_all"][k] for k in cols], mat.mean(0), "all_22_macro", label)
            scopes = [("all", np.arange(60), absolute["pooled_fixed_half_classification"]["pooled"])]
            for d in "ABC":
                rows = np.flatnonzero(domains == d)
                near([absolute["macro_by_domain"][d][k] for k in cols], mat[rows].mean(0), "all_22_domain", label + "/" + d)
                scopes.append((d, rows, absolute["pooled_fixed_half_classification"]["by_domain"][d]))
            for scope, indices, target in scopes:
                got = pooled([counts[i] for i in indices])
                near([got[k] for k in got], [target[k] for k in got], "pooled_absolute", label + "/" + scope)
            check(absolute["pooled_fixed_half_classification"]["threshold"] == 0., "stored_logit_threshold", label)

    affine = []
    for order in ("ABC", "BCA", "CAB"):
        for stage in (1, 2, 3):
            name = f"{order}_function_memory_stage{stage}"
            point = read(f / f"job/run/points/{name}.json")
            mapping = read(verify(f / "job/run", point["mapping"]))
            near([mapping[k] for k in ("a", "b")], [point["maps"]["stage"][k] for k in ("a", "b")],
                 "map_receipt", name)
            raw = load(verify(f / "job/run", point["scores"]["raw"]))
            check(raw.shape == (60, 378) and np.isfinite(raw).all(), "score_shape_finite", name + "/raw")
            for role, mk in (("raw", None), ("stage-cal", "stage"), ("first-cal", "first")):
                score = raw if role == "raw" else load(verify(f / "job/run", point["scores"][role]))
                count = all_counts[("function_memory", name, role)]
                near((score >= 0).sum(1), [c["tp"] + c["fp"] for c in count], "predicted_positive_count", name + "/" + role, 0.)
                if mk is not None:
                    a, b = point["maps"][mk]["a"], point["maps"][mk]["b"]
                    expected = a * raw.astype(np.float64) + b
                    check(a > 0, "positive_affine", name + "/" + role)
                    near(score, expected, "affine_score_equality", name + "/" + role, 0.)
                    check(np.array_equal(np.argsort(raw, axis=1, kind="stable"), np.argsort(score, axis=1, kind="stable")),
                          "rank_order_preserved", name + "/" + role)
                    near(loaded[("function_memory", name, role)][:, rank_cols], loaded[("function_memory", name, "raw")][:, rank_cols],
                         "rank_curve_matrix_equality", name + "/" + role, 0.)
                    affine.append({"point": name, "role": role, "a": a, "b": b})
            if stage == 1:
                for role in ("raw", "stage-cal", "first-cal"):
                    near(loaded[("function_memory", name, role)], loaded[("logit0.1", order + "_shared", role)],
                         "shared_first_matrix_exact", order + "/" + role, 0.)
                    check(all_counts[("function_memory", name, role)] == all_counts[("logit0.1", order + "_shared", role)],
                          "shared_first_counts_exact", order + "/" + role)
    initial = load(f / "job/run/scores/initial_raw.npy")
    near((initial >= 0).sum(1), [c["tp"] + c["fp"] for c in all_counts[("function_memory", "initial", "raw")]],
         "predicted_positive_count", "initial/raw", 0.)

    endpoint_counts = {}
    for arm in ("function_memory", "logit0.1"):
        endpoint_counts[arm] = {}
        for ep, terms in {"O": [(3, 0), (3, 1)], "A2": [(2, 1)], "N": [(2, 1), (3, 2)],
                          "Z": [(3, 2)], "final_all": [(3, 0), (3, 1), (3, 2)]}.items():
            selected = []
            for order in ("ABC", "BCA", "CAB"):
                for stage, arrival in terms:
                    name = f"{order}_function_memory_stage{stage}" if arm == "function_memory" else f"{order}_logit_tenth_stage{stage}"
                    cs = all_counts[(arm, name, "stage-cal")]
                    selected.extend(cs[i] for i in np.flatnonzero(domains == order[arrival]))
            vals = pooled(selected)
            vals.update(group_contributions=len(selected),
                        zero_prediction_groups=sum(c["tp"] + c["fp"] == 0 for c in selected),
                        macro_precision=float(np.mean([classification(c)["precision"] for c in selected])),
                        note="Repeated appearances of the same actual group are contributions, not independent samples")
            endpoint_counts[arm][ep] = vals

    result = {"status": "PASS" if not failures else "FAIL", "scope": "Saved matrices/counts/scores/receipts only; no labels, model, Memory, fitting or project imports",
              "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
              "elapsed_seconds": time.perf_counter() - started, "checks": dict(checks), "maximum_absolute_errors": max_error,
              "failures": failures, "endpoint_counts": endpoint_counts, "all_sets": details, "positive_affine_maps": affine,
              "input_files": [{"path": p, **v} for p, v in sorted(inputs.items())]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "checks", "maximum_absolute_errors", "failures", "endpoint_counts")}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
