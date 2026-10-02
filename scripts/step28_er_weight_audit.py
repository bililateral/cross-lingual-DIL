"""Independently audit saved ER-weight results on Linux; no labels or models."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from step28_bge_continual_audit import Audit, METRICS, ORDERS, RANK, TERMS, bound, confusion, read, record

ARMS = {"half": .5, "quarter": .25}
ROLES = ("raw", "stage-cal", "first-cal", "primary")


def point(order: str, arm: str, stage: int) -> str:
    return order + "_shared" if stage == 1 else f"{order}_{arm}_stage{stage}"


def summarize(value: np.ndarray, weights: np.ndarray) -> dict:
    per_order = value.sum(axis=1) / 20
    means = per_order.mean(axis=0)
    ordered = np.sort(weights @ value.mean(axis=0), axis=0)
    intervals = []
    for fraction in (.025, .975):
        position = fraction * 4999
        lo, hi = math.floor(position), math.ceil(position)
        intervals.append(ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo]))
    return {m: {"mean": float(means[k]),
                "per_order": {o: float(per_order[i, k]) for i, o in enumerate(ORDERS)},
                "conditional_95pct_interval": [float(a[k]) for a in intervals]}
            for k, m in enumerate(METRICS)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "job", "baseline", "inventory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=47)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    os.nice(15)
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    project, job, baseline = args.project.resolve(), args.job.resolve(), args.baseline.resolve()
    run, ev, old_run = job / "run", job / "evaluation", baseline / "run"
    audit = Audit()
    for item in read(args.inventory)["files"]:
        bound(project, item)
        audit.checks["returned_files"] += 1
    manifest, collected, saved = read(run / "manifest.json"), read(ev / "collected.json"), read(ev / "evaluation.json")
    old_manifest, old_collected = read(old_run / "manifest.json"), read(baseline / "evaluation/collected.json")
    policy = read(project / "schema/step28_er_weight_policy.json")
    for item in policy["baseline"]["records"].values():
        bound(baseline, item)
    completion, gate = read(job / "completion.json"), read(job / "before_valid.json")
    assert (job / "exit_status.txt").read_text().strip() == "0" and not (job / "failure.json").exists()
    assert completion["physical_updates"] == manifest["physical_updates"] == 3456
    assert manifest["gradient_group_presentations"] == 6912
    assert read(job / "access.json") == completion["label_parses"] == dict(train=1, valid=1, heldout=0, owners=0)
    assert gate["status"] == "PASS_ER_WEIGHT_COMPLETE_BLIND_GATE"
    assert gate["label_parses"] == dict(train=1, valid=0, heldout=0, owners=0)
    bound(job, gate["manifest"])
    bound(job, completion["evaluation"])
    for item in read(job / "execution.json")["source_files"]:
        bound(project, item)
        audit.checks["unchanged_frozen_sources"] += 1
    assert collected["group_ids"] == old_collected["group_ids"]
    assert collected["domains"] == old_collected["domains"]
    assert tuple(collected["metric_columns"]) == METRICS
    assert len(set(collected["group_ids"])) == 60
    domains = np.asarray(collected["domains"])
    rows = {d: np.flatnonzero(domains == d) for d in "ABC"}
    assert all(len(r) == 20 for r in rows.values())
    partition = read(bound(run, manifest["partition"]))
    assert partition == read(old_run / "partition.json")
    uid_domain = {x["group_uid"]: x["domain"] for x in partition["fit"]}
    expected = {point(o, a, s) for o in ORDERS for a in ARMS for s in (2, 3)}
    assert set(manifest["points"]) == set(manifest["training"]) == set(collected["points"]) == expected
    diagnostics = {"training": {}, "memory": {}, "calibration": {}}
    scores = {}
    for name in sorted(expected):
        p = read(bound(run, manifest["points"][name]))
        t = read(bound(run, manifest["training"][name]))
        order, arm, stage = p["order"], name.split("_")[1], p["stage"]
        reference = read(bound(old_run, old_manifest["training"][point(order, "er", stage)]))
        assert p["history_weight"] == t["history_weight"] == ARMS[arm]
        assert p["completed_updates"] == t["adam_step"] == 288 * stage
        assert p["full_model_adam_and_rng_restore_verified"] is True and t["updates"] == 288
        for key in ("current_ids", "history_ids", "current_dropout_stream", "adam_step", "updates"):
            assert t[key] == reference[key], (name, key)
        current_ids = {uid for uid, domain in uid_domain.items() if domain == order[stage - 1]}
        assert Counter(t["current_ids"]) == Counter({uid: 6 for uid in current_ids}) and len(current_ids) == 48
        memory = read(run / "memory" / (name + "_budget.json"))
        assert 0 < memory["serialized_bytes"] <= 1048576 and len(memory["members"]) == 6
        assert memory["members"] == reference["memory_after_training"]["members"]
        assert set(t["history_ids"]) <= set(memory["members"]) and not set(memory["members"]) & current_ids
        x = np.load(bound(run, t["update_file"]), allow_pickle=False)
        assert x.shape == (288, 14) and np.isfinite(x).all()
        c = {column: x[:, k] for k, column in enumerate(t["update_columns"])}
        for role in ("current", "history"):
            assert np.allclose(c[role + "_total"], c[role + "_bce"] + c[role + "_rank"] + .5 * c[role + "_hard"], atol=3e-6, rtol=3e-6)
        audit.array(c["history_weight"], np.full(288, ARMS[arm]), name + "/lambda", 0.)
        audit.array(c["weighted_history_total"], ARMS[arm] * c["history_total"], name + "/weighted", 0.)
        audit.array(c["total"], c["current_total"] + ARMS[arm] * c["history_total"], name + "/total", 0.)
        audit.array(c["encoder_lr"], np.array([1e-5 * (s / 29 if s <= 29 else (288 - s) / 259) for s in range(1, 289)]), name + "/lr")
        audit.array(c["head_lr"], np.full(288, .001), name + "/head_lr", 0.)
        for step in (1, 29, 30, 288):
            for module in ("encoder", "head"):
                obs = t["observations"][str(step)][module]
                assert obs["finite_nonzero_combined_gradient"] and obs["parameters_changed"] == (module == "head" or step < 288)
        mapping = read(bound(run, p["mapping"]))
        assert mapping["status"] == "PASS_CALIBRATION_FIT" and mapping["a"] > 0
        assert mapping["calibration_group_ids"] == [g["group_uid"] for g in partition["calibration"] if g["domain"] == order[stage - 1]]
        assert (mapping["group_count"], mapping["pair_count"], mapping["positive_count"]) == (12, 4536, 240)
        assert mapping["model_state_sha256"] == p["model_state_sha256"] and mapping["score_source"] == p["scores"]["calibration"]
        values = {role: np.load(bound(run, item), allow_pickle=False) for role, item in p["scores"].items()}
        assert all(v.shape == ((12, 378) if r == "calibration" else (60, 378)) and np.isfinite(v).all() for r, v in values.items())
        for role, parameters in (("stage-cal", mapping), ("first-cal", p["first_map_parameters"])):
            raw = values["development"].astype(np.float64)
            audit.array(values[role], raw * parameters["a"] + parameters["b"], name + role, 0.)
            ordering = np.argsort(raw, axis=1, kind="stable")
            assert np.array_equal(ordering, np.argsort(values[role], axis=1, kind="stable"))
            assert np.array_equal(np.diff(np.take_along_axis(raw, ordering, 1)) == 0, np.diff(np.take_along_axis(values[role], ordering, 1)) == 0)
        scores[name] = values
        diagnostics["training"][name] = {"seconds": t["training_seconds"], "gradient_norm_gt_one": int((c["gradient_norm"] > 1).sum()), "gradient_norm_mean": float(c["gradient_norm"].mean())}
        diagnostics["memory"][name] = {"bytes": memory["serialized_bytes"], "domains": dict(Counter(uid_domain[uid] for uid in memory["members"]))}
        diagnostics["calibration"][name] = {k: mapping[k] for k in ("a", "b", "initial_nll", "final_nll")}
    for order in ORDERS:
        shared = read(bound(old_run, old_manifest["points"][order + "_shared"]))
        for arm in ARMS:
            start = manifest["restored_starts"][order + "_" + arm]
            assert start["full_checkpoint"] == shared["full_checkpoint"] and start["adam_step"] == 288
            assert start["model_state_sha256"] == shared["model_state_sha256"] and start["first_scores_replayed_exactly"]
            assert start["first_map"] == shared["first_map_parameters"]
            assert start["memory_source"] == old_manifest["memories"][order + "_er_after1"]["file"]
            retained = manifest["memories"][point(order, arm, 2)]
            assert retained["members"] == old_manifest["memories"][point(order, "er", 2)]["members"]
    matrices, counts = {}, {}
    for collection, folder in ((read(ev / "reference/collected.json"), ev / "reference"), (collected, ev)):
        for name, variants in collection["points"].items():
            matrices[name], counts[name] = {}, {}
            for role, info in variants.items():
                matrix = np.load(bound(folder, info["matrix"]), allow_pickle=False)
                assert matrix.shape == (60, 22) and matrix.dtype == np.float64 and np.isfinite(matrix).all()
                count = read(bound(folder, info["counts"]))
                assert len(count) == 60
                for i, row in enumerate(count):
                    assert all(type(row[k]) is int and row[k] >= 0 for k in ("tp", "fp", "fn", "tn"))
                    assert row["tp"] + row["fn"] == 20 and row["fp"] + row["tn"] == 358
                    for metric, value in confusion(row).items():
                        audit.equal(matrix[i, METRICS.index(metric)], value, name + role + metric)
                    if name in scores:
                        assert int((scores[name]["development" if role == "raw" else role][i] >= 0).sum()) == row["tp"] + row["fp"]
                if name not in expected:
                    assert info == old_collected["points"][name][role]
                matrices[name][role], counts[name][role] = matrix, count
                audit.checks["matrix_count_sets"] += 1
            for role in ROLES[1:3]:
                audit.array(matrices[name][role][:, RANK], matrices[name]["raw"][:, RANK], name + "/rank", 0.)
            matrices[name]["primary"] = matrices[name]["stage-cal"].copy()
    assert audit.checks["matrix_count_sets"] == 81
    draws = np.random.Generator(np.random.PCG64(20260930)).integers(0, 20, (5000, 3, 20))
    assert np.array_equal(draws, np.load(bound(ev, saved["draws"]), allow_pickle=False))
    weights = np.zeros((5000, 60))
    for i, domain in enumerate("ABC"):
        weights[:, rows[domain]] = np.stack([np.bincount(d, minlength=20) for d in draws[:, i]]) / 20

    def field(arm: str, role: str, endpoint: str) -> np.ndarray:
        value = np.zeros((3, 60, 22))
        for i, order in enumerate(ORDERS):
            for stage, arrival, coefficient in TERMS[endpoint]:
                selected = rows[order[arrival - 1]]
                value[i, selected] += coefficient * matrices[point(order, arm, stage)][role][selected]
        if endpoint in ("F_first", "F", "G"):
            value[:, :, [4, 5]] *= -1
        return value

    handmade = np.zeros((3, 60, 22))
    for domain, value in zip("ABC", (2, 4, 6)):
        handmade[:, rows[domain]] = value
    audit.equal(summarize(handmade, weights)["map"], {"mean": 12., "per_order": dict.fromkeys(ORDERS, 12.), "conditional_95pct_interval": [12., 12.]}, "handmade_bootstrap")
    assert confusion(dict(tp=2, fp=1, fn=2, tn=5))["recall"] == .5
    endpoints = {arm: {role: {ep: summarize(field(arm, role, ep), weights) for ep in TERMS} for role in ROLES} for arm in ("seq", "er", *ARMS)}
    audit.equal(saved["endpoints"], endpoints, "endpoints")
    comparisons = {}
    signs = dict(average_precision=1, roc_auc=1, brier=-1, log_loss=-1)
    for candidate in ARMS:
        for reference in ("er", "seq"):
            name = candidate + "_minus_" + reference
            delta = {ep: summarize(field(candidate, "primary", ep) - field(reference, "primary", ep), weights) for ep in TERMS}
            raw = {ep: summarize(field(candidate, "primary", ep) - field(reference, "raw", ep), weights) for ep in ("O", "N")}
            audit.equal(saved["comparisons"][name]["primary"], delta, name)
            audit.equal(saved["comparisons"][name]["against_raw_reference"], raw, name + "/raw")
            checks = {"old_map_improves": delta["O"]["map"]["mean"] > 0, "old_map_interval_above_zero": delta["O"]["map"]["conditional_95pct_interval"][0] > 0,
                      "old_recall5_improves": delta["O"]["recall_at_5"]["mean"] > 0, "new_map_non_decrease": delta["N"]["map"]["mean"] >= 0, "new_recall5_non_decrease": delta["N"]["recall_at_5"]["mean"] >= 0}
            for ep in ("O", "N"):
                checks.update({f"{ep}_{m}_non_degradation": sign * delta[ep][m]["mean"] >= 0 for m, sign in signs.items()})
                checks.update({f"{ep}_{m}_against_raw_reference": raw[ep][m]["mean"] <= 0 for m in ("brier", "log_loss")})
            checks.update({f"Z_{m}_non_degradation": sign * delta["Z"][m]["mean"] >= 0 for m, sign in dict(map=1, recall_at_5=1, **signs).items()})
            verdict = saved["comparisons"][name]["interpretation"]
            audit.equal(verdict["checks"], checks, name + "/checks")
            assert len(checks) == 23 and verdict["pilot_observed_checks_pass"] == all(checks.values())
            assert set(verdict["failed"]) == {k for k, value in checks.items() if not value}
            assert verdict["positive_old_map_orders"] == sum(v > 0 for v in delta["O"]["map"]["per_order"].values())
            comparisons[name] = {"passed": sum(checks.values()), "total": len(checks), "checks": checks}
    eligible = [a for a in ARMS if comparisons[a + "_minus_er"]["passed"] == 23]
    selected = max(eligible, key=lambda a: (endpoints[a]["primary"]["O"]["map"]["mean"], endpoints[a]["primary"]["O"]["recall_at_5"]["mean"], ARMS[a])) if eligible else "er"
    assert saved["selection"] == completion["selection"]
    assert saved["selection"]["selected"] == selected and saved["selection"]["eligible"] == eligible
    assert saved["selection"]["history_weight"] == ARMS.get(selected, 1.) and saved["selection"]["fallback_used"] == (not eligible)
    for name, roles in saved["absolute_stage_results"].items():
        for role, obj in roles.items():
            matrix = matrices[name][role]
            audit.equal(obj["macro_all"], dict(zip(METRICS, matrix.mean(0).tolist())), name + role)
            for domain, indices in rows.items():
                audit.equal(obj["macro_by_domain"][domain], dict(zip(METRICS, matrix[indices].mean(0).tolist())), name + role + domain)
            pooled = obj["pooled_fixed_half_classification"]
            assert pooled["threshold"] == 0
            for domain, indices in {**rows, "pooled": np.arange(60)}.items():
                total = {k: sum(counts[name][role][i][k] for i in indices) for k in ("tp", "fp", "fn", "tn")}
                rates = confusion(total)
                expected_count = {**total, **{k: rates[k] for k in ("precision", "recall", "f1")}, "fpr": total["fp"] / (total["fp"] + total["tn"])}
                audit.equal(pooled["pooled"] if domain == "pooled" else pooled["by_domain"][domain], expected_count, name + role + domain + "/counts")
    with bound(ev, saved["stage_metrics"]).open(encoding="utf-8", newline="") as stream:
        keys = set()
        for row in csv.DictReader(stream):
            name = point(row["order"], row["method"], int(row["stage"]))
            assert name == row["source_point"]
            key = tuple(row[k] for k in ("order", "method", "stage", "role", "actual_domain", "metric"))
            assert key not in keys
            keys.add(key)
            audit.equal(float(row["group_macro"]), matrices[name][row["role"]][rows[row["actual_domain"]], METRICS.index(row["metric"])].mean(), "stage_csv")
        assert len(keys) == 3 * 4 * 3 * 3 * 3 * 22
        audit.checks["stage_csv_rows"] = len(keys)
    result = {"status": "PASS_ER_WEIGHT_SAVED_RESULT_AUDIT", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "script": record(Path(__file__)), "helper": record(Path(__file__).with_name("step28_bge_continual_audit.py")),
              "numeric_comparisons": audit.numbers, "maximum_absolute_error": audit.max_error, "checks": dict(audit.checks), "comparisons": comparisons, "selection": saved["selection"],
              "new_label_parses": dict(train=0, valid=0, heldout=0, owners=0), "formal_texts_or_models_loaded": False,
              "limitations": ["Saved aggregate metrics audited; label-dependent AP/MAP not recomputed from truth.", "Native execution not repeated; weights hashed separately without model loading.", "Single seed, developed valid and selection-conditional intervals; no independent test or method novelty claim."]}
    for name, obj in (("audit.json", result), ("diagnostics.json", diagnostics)):
        (args.output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
