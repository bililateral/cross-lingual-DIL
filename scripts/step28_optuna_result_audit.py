"""Supplement the campaign audit from saved results only; Linux py310, no labels/models.

Checks the saved three-seed absolute, forgetting and positive-count summaries not
covered by the campaign's independent_analysis, and exports the complete tables.
Uses the already registered group bootstrap, with no new selection or criterion.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.name == "posix", "Linux only"
    started = time.perf_counter()
    root, out = args.job.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    completion = read(root / "completion.json")
    assert completion["status"] == "COMPLETE_36_SEARCH_9_REPLICATIONS_AND_ANALYSIS"
    assert completion["access"] == dict(train=46, valid=37, heldout=0, owners=0)
    prior = read(root / "analysis/audit.json")
    assert prior["status"] == "PASS_SAVED_ARITHMETIC"
    joint = read(root / "joint/evaluation.json")
    replicas = read(root / "replications.json")
    search = read(root / "search.json")
    arms, roles = ("S", "S_pos", "H_pos"), ("raw", "stage-cal", "first-cal")
    arrays, queries, degrees, saved = {}, {}, {}, {}
    errors, checks, identities, na = [], 0, [], 0

    def verify(base, rec):
        p = base / rec["path"]
        assert p.stat().st_size == rec["bytes"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"]
        identities.append(str(p.relative_to(root)))
        return p

    def close(a, b):
        nonlocal checks
        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        assert x.shape == y.shape and np.array_equal(np.isnan(x), np.isnan(y))
        assert not np.isinf(x).any() and not np.isinf(y).any()
        mask = ~np.isnan(x)
        error = float(np.max(np.abs(x[mask] - y[mask]))) if mask.any() else 0.0
        assert error <= 1e-12, error
        errors.append(error)
        checks += x.size

    specs = [r["spec"] for a in arms for r in search[a]] + [r["trial"] for r in replicas]
    table, count_table, seed_table, search_table = [], [], [], []
    reference_groups = None
    for spec in specs:
        arm, seed = spec["arm"], spec["seed"]
        run_id = f'{spec["phase"]}_{spec["number"]:02d}_{arm}_s{seed}'
        job = root / run_id
        col = read(job / "evaluation/collected.json")
        population = (col["group_ids"], col["domains"], col["columns"])
        if reference_groups is None:
            reference_groups = population
        assert population == reference_groups
        columns = col["columns"]
        rows = {d: np.flatnonzero(np.array(col["domains"]) == d) for d in "AB"}
        evaluation = read(job / "evaluation/evaluation.json")
        for point, role_records in col["points"].items():
            for role, rec in role_records.items():
                matrix = np.asarray(read(verify(job / "evaluation", rec["matrix"]))["values"], float)
                q = np.load(verify(job / "evaluation", rec["queries"]), allow_pickle=False)
                close(q.mean(1), matrix[:, 12:])
                counts = read(verify(job / "evaluation", rec["counts"]))
                for k, row in enumerate(counts):
                    tp, fp, fn, tn = (row[n] for n in ("tp", "fp", "fn", "tn"))
                    assert tp + fn == 20 and fp + tn == 358
                    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                    rates = [tp / (tp + fp) if tp + fp else None, tp / 20,
                             2 * tp / (2 * tp + fp + fn), tn / 358,
                             (tp / 20 + tn / 358) / 2,
                             (tp * tn - fp * fn) / np.sqrt(denom) if denom else None]
                    close(matrix[k, 6:12], rates)
                for d in "AB":
                    target = evaluation["absolute"][point][role]["by_domain"][d]
                    for metric in columns:
                        t = target[metric]
                        ci = t["conditional_95pct_interval"] or [None, None]
                        table.append([run_id, spec["phase"], arm, seed, point[0], role, d, metric, t["mean"], *ci])
                    pooled = evaluation["absolute"][point][role]["fixed_half_classification"][d]
                    actual = np.asarray([[r[k] for k in ("tp", "fp", "fn", "tn")] for r in counts])[rows[d]].sum(0)
                    close(actual, [pooled[k] for k in ("tp", "fp", "fn", "tn")])
                    count_table.append([run_id, point[0], role, d, *actual.tolist(), pooled["precision"], pooled["recall"], pooled["f1"], pooled["fpr"]])
                if spec["phase"] == "replication":
                    arrays[arm, seed, point[0], role] = matrix
                    if role == "raw":
                        queries[arm, seed, point[0]] = q
        if spec["phase"] == "replication":
            deg = np.load(verify(job / "evaluation", col["query_positive_counts"]), allow_pickle=False)
            assert deg.shape == (40, 28) and np.all((deg == 1).sum(1) == 16) and np.all((deg == 2).sum(1) == 12)
            degrees[arm, seed] = deg
            saved[arm, seed] = evaluation
            seed_table.append([arm, seed, evaluation["absolute"]["A_" + arm]["raw"]["by_domain"]["A"]["map"]["mean"],
                               evaluation["domains"]["A"], evaluation["domains"]["B"], evaluation["old_changes"]["raw"]["map"]["mean"]])
        else:
            r = search[arm][spec["number"]]
            search_table.append([arm, spec["number"], spec["lambda_rank"], spec["lr_multiplier"], r["domains"]["A"], r["domains"]["B"], r["value"]])

    draws = np.random.Generator(np.random.PCG64(20260930)).integers(20, size=(5000, 2, 20))
    frequencies = [np.stack([np.bincount(x, minlength=20) for x in draws[:, i]]) for i in range(2)]
    aggregate_rows, subgroup_rows = [], []

    def stats(values, target, domain, label, phase, role, names=columns):
        nonlocal na
        vals = np.asarray(values)
        if vals.ndim == 1:
            vals = vals[:, None]
        idx = "AB".index(domain)
        for k, metric in enumerate(names):
            v = vals[rows[domain], k]
            t = target[metric]
            if np.isnan(v).any():
                assert t["mean"] is None and t["conditional_95pct_interval"] is None
                na += 1
                summary = [None, None, None]
            else:
                summary = [float(v.mean()), *np.quantile(frequencies[idx] @ v / 20, [.025, .975]).tolist()]
                close(summary, [t["mean"], *t["conditional_95pct_interval"]])
            aggregate_rows.append([label, phase, role, domain, metric, *summary])

    signed = np.array([-1 if c in ("brier", "log_loss") else 1 for c in columns])
    old_change = {}
    for arm in arms:
        old_change[arm] = {}
        for phase in "AB":
            for role in roles:
                a = np.mean([arrays[arm, s, phase, role] for s in (1, 2, 3)], axis=0)
                for d in "AB":
                    stats(a, joint["three_seed_absolute"][arm][phase][role][d], d, arm, phase, role)
        delta = np.mean([arrays[arm, s, "B", "raw"] - arrays[arm, s, "A", "raw"] for s in (1, 2, 3)], axis=0)
        old_change[arm] = {c: float(delta[rows["A"], k].mean()) for k, c in enumerate(columns) if not np.isnan(delta[rows["A"], k]).any()}
    for arm, other in (("S_pos", "S"), ("H_pos", "S"), ("H_pos", "S_pos")):
        label = arm + "-" + other
        target = joint["comparisons"][label]
        for seed in (1, 2, 3):
            for role in roles:
                delta = ((arrays[arm, seed, "A", role] - arrays[arm, seed, "B", role]) -
                         (arrays[other, seed, "A", role] - arrays[other, seed, "B", role])) * signed
                stats(delta, target["by_seed"][str(seed)]["forgetting"][role], "A", label + f"/s{seed}", "forgetting", role)
        for role in roles:
            delta = np.mean([((arrays[arm, s, "A", role] - arrays[arm, s, "B", role]) -
                              (arrays[other, s, "A", role] - arrays[other, s, "B", role])) * signed for s in (1, 2, 3)], axis=0)
            stats(delta, target["three_seed_mean"]["forgetting"][role], "A", label, "forgetting", role)
        for count in (1, 2):
            differences = []
            for seed in (1, 2, 3):
                pair = []
                for candidate in (arm, other):
                    mask = degrees[candidate, seed] == count
                    pair.append((queries[candidate, seed, "B"][:, :, 0] * mask).sum(1) / mask.sum(1))
                differences.append(pair[0] - pair[1])
                for d in "AB":
                    t = joint["query_subgroup_MAP_comparisons"][label][str(count)]["by_seed"][str(seed)][d]
                    stats(differences[-1], {"map": t}, d, label + f"/positive{count}/s{seed}", "B", "raw", ["map"])
            for d in "AB":
                t = joint["query_subgroup_MAP_comparisons"][label][str(count)]["three_seed_mean"][d]
                stats(np.mean(differences, axis=0), {"map": t}, d, label + f"/positive{count}", "B", "raw", ["map"])
                subgroup_rows.append([label, count, d, t["mean"], *t["conditional_95pct_interval"]])

    def export(name, header, records):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(records)

    export("all_run_metrics.csv", ["run", "phase", "arm", "seed", "endpoint", "role", "domain", "metric", "mean", "low", "high"], table)
    export("classification_counts.csv", ["run", "endpoint", "role", "domain", "tp", "fp", "fn", "tn", "pooled_precision", "pooled_recall", "pooled_f1", "pooled_fpr"], count_table)
    export("replication_map.csv", ["arm", "seed", "A_end_A", "B_end_A", "B_end_B", "A_change"], seed_table)
    export("search.csv", ["arm", "trial", "lambda_rank", "lr_multiplier", "A_MAP", "B_MAP", "U"], search_table)
    export("aggregate_audit.csv", ["comparison", "endpoint", "role", "domain", "metric", "mean", "low", "high"], aggregate_rows)
    export("positive_count_comparisons.csv", ["comparison", "positives", "domain", "mean", "low", "high"], subgroup_rows)
    write(out / "summary.json", dict(selected=read(root / "selected.json"), by_seed=seed_table,
        three_seed_absolute=joint["three_seed_absolute"], comparisons=joint["comparisons"],
        direction_gate=joint["development_direction_consistency"], old_changes_raw=old_change,
        positive_count=joint["query_subgroup_MAP_comparisons"], diagnostics=joint["diagnostics"]))
    write(out / "audit.json", dict(status="PASS_SAVED_RESULT_SUPPLEMENT", checked_scalars=checks,
        maximum_absolute_error=max(errors), checked_files=len(set(identities)), undefined_aggregate_rows=na,
        all_run_metric_rows=len(table), aggregate_rows=len(aggregate_rows), elapsed_seconds=time.perf_counter()-started,
        formal_label_reads=0, model_loads=0, prior_audit=prior,
        limits="Saved metric/count/query summaries only; no independent label-level reconstruction, new data, or new selection."))
    print(json.dumps(dict(status="PASS", checked_scalars=checks, maximum_error=max(errors), metric_rows=len(table))))


if __name__ == "__main__":
    main()
