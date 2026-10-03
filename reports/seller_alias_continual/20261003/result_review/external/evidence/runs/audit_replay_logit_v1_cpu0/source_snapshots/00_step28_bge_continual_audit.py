"""Independent saved-result audit. No labels, texts, models, fitting or torch.

Run on Linux py310. Frozen experimental code is intentionally not imported.
This checks saved metric algebra and execution evidence; it does not reconstruct
label-dependent AP/MAP from blind scores or claim a second native execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ORDERS = ("ABC", "BCA", "CAB")
ARMS = ("frozen", "seq", "er", "logit")
ROLES = ("raw", "stage-cal", "first-cal", "primary")
METRICS = (
    "average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct",
    "brier", "log_loss", "precision", "recall", "f1", "specificity",
    "balanced_accuracy", "mcc", "map", "mrr", "recall_at_1", "recall_at_3",
    "recall_at_5", "recall_at_10", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10",
)
TERMS = {
    "O": [(3, 1, .5), (3, 2, .5)],
    "N": [(2, 2, .5), (3, 3, .5)],
    "Z": [(3, 3, 1.)],
    "F_first": [(1, 1, 1.), (3, 1, -1.)],
    "F": [(1, 1, .5), (3, 1, -.5), (2, 2, .5), (3, 2, -.5)],
    "G": [(2, 2, .5), (1, 2, -.5), (3, 3, .5), (2, 3, -.5)],
    "final_all": [(3, 1, 1 / 3), (3, 2, 1 / 3), (3, 3, 1 / 3)],
}
RANK = list(range(4)) + list(range(12, 22))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def record(path: Path) -> dict:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def bound(root: Path, info: dict) -> Path:
    path = (root / info["path"]).resolve()
    path.relative_to(root.resolve())
    if record(path) != {k: info[k] for k in ("bytes", "sha256")}:
        raise ValueError("File binding differs: " + str(path))
    return path


def point(order: str, arm: str, stage: int) -> str:
    return order + "_shared" if stage == 1 or arm == "frozen" else f"{order}_{arm}_stage{stage}"


def confusion(row: dict) -> dict:
    tp, fp, fn, tn = (int(row[k]) for k in ("tp", "fp", "fn", "tn"))
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    recall = tp / (tp + fn) if tp + fn else 0.
    spec = tn / (tn + fp) if tn + fp else 0.
    return {"precision": tp / (tp + fp) if tp + fp else 0., "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "specificity": spec, "balanced_accuracy": (recall + spec) / 2,
            "mcc": (tp * tn - fp * fn) / den if den else 0.}


class Audit:
    def __init__(self):
        self.numbers = 0
        self.max_error = 0.
        self.checks = Counter()

    def equal(self, actual, expected, name: str, tolerance: float = 3e-12):
        if isinstance(expected, dict):
            assert set(actual) == set(expected), (name, "keys")
            for key in expected:
                self.equal(actual[key], expected[key], name + "/" + key, tolerance)
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected), (name, "length")
            for i, value in enumerate(expected):
                self.equal(actual[i], value, name + "/" + str(i), tolerance)
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            error = abs(float(actual) - float(expected))
            assert math.isfinite(error) and error <= tolerance, (name, actual, expected, error)
            self.numbers += 1
            self.max_error = max(self.max_error, error)
        else:
            assert actual == expected, (name, actual, expected)

    def array(self, actual, expected, name: str, tolerance: float = 3e-12):
        assert actual.shape == expected.shape, name
        errors = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        maximum = float(errors.max(initial=0))
        assert np.isfinite(errors).all() and maximum <= tolerance, (name, maximum)
        self.numbers += actual.size
        self.max_error = max(self.max_error, maximum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=47)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    os.nice(15)
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    project, job = args.project.resolve(), args.job.resolve()
    run, ev = job / "run", job / "evaluation"
    inventory = read(args.inventory)
    for item in inventory["files"]:
        bound(project, item)
    audit.checks["returned_files"] = len(inventory["files"])
    manifest, collected, saved = read(run / "manifest.json"), read(ev / "collected.json"), read(ev / "evaluation.json")
    completion, gate = read(job / "completion.json"), read(job / "before_valid.json")
    assert completion["physical_updates"] == manifest["physical_updates"] == 6048
    assert manifest["gradient_group_presentations"] == 9504
    assert (job / "exit_status.txt").read_text().strip() == "0"
    assert not (job / "failure.json").exists()
    assert read(job / "access.json") == completion["label_parses"] == dict(train=1, valid=1, heldout=0, owners=0)
    assert gate["status"] == "PASS_COMPLETE_BLIND_GATE" and gate["points"] == 21
    assert gate["supervision"] == dict(train=1, valid=0, heldout=0, owners=0)
    bound(job, gate["manifest"])
    bound(job, completion["evaluation"])
    for item in read(job / "execution.json")["source_files"]:
        bound(project, item)
        audit.checks["unchanged_frozen_sources"] += 1
    assert tuple(collected["metric_columns"]) == METRICS
    domains = np.asarray(collected["domains"])
    groups = collected["group_ids"]
    assert len(groups) == len(set(groups)) == 60
    rows = {d: np.flatnonzero(domains == d) for d in "ABC"}
    assert all(len(r) == 20 for r in rows.values())
    partition = read(bound(run, manifest["partition"]))
    assert [x["group_uid"] for x in partition["development"]] == groups
    assert [x["domain"] for x in partition["development"]] == domains.tolist()
    uid_domain = {x["group_uid"]: x["domain"] for x in partition["fit"]}
    expected_points = {point(o, a, s) for o in ORDERS for a in ARMS for s in (1, 2, 3)}
    assert len(expected_points) == 21 and set(collected["points"]) == set(manifest["points"]) == expected_points
    matrices, counts, scores, points, mappings, training = {}, {}, {}, {}, {}, {}
    loss_error = 0.
    diagnostics = {"training": {}, "memory": {}, "calibration": {}}
    for name in sorted(expected_points):
        p = read(bound(run, manifest["points"][name]))
        points[name] = p
        assert p["name"] == name and p["completed_updates"] == 288 * p["stage"]
        assert p["full_model_adam_and_rng_restore_verified"] is True
        m = read(bound(run, p["mapping"]))
        mappings[name] = m
        assert m["a"] > 0 and m["status"] == "PASS_CALIBRATION_FIT" and m["optimizer_success"]
        assert (m["group_count"], m["pair_count"], m["positive_count"]) == (12, 4536, 240)
        assert m["actual_domain"] == p["actual_domain"] == p["order"][p["stage"] - 1]
        assert m["calibration_group_ids"] == [x["group_uid"] for x in partition["calibration"] if x["domain"] == p["actual_domain"]]
        assert m["model_state_sha256"] == p["model_state_sha256"]
        assert m["score_source"] == p["scores"]["calibration"]
        scores[name] = {r: np.load(bound(run, f), allow_pickle=False) for r, f in p["scores"].items()}
        for role, value in scores[name].items():
            assert value.shape == ((12, 378) if role == "calibration" else (60, 378)) and np.isfinite(value).all()
            assert value.dtype == (np.float32 if role in ("calibration", "development") else np.float64)
        raw = scores[name]["development"].astype(np.float64)
        for role, mapping in (("stage-cal", m), ("first-cal", p["first_map_parameters"])):
            transformed = raw * mapping["a"] + mapping["b"]
            audit.array(scores[name][role], transformed, name + role, 0.)
            ordering = np.argsort(raw, axis=1, kind="stable")
            assert np.array_equal(ordering, np.argsort(transformed, axis=1, kind="stable"))
            assert np.array_equal(np.diff(np.take_along_axis(raw, ordering, axis=1)) == 0,
                                  np.diff(np.take_along_axis(transformed, ordering, axis=1)) == 0)
            audit.checks["affine_group_orders_and_ties"] += 60
        diagnostics["calibration"][name] = {k: m[k] for k in ("a", "b", "initial_nll", "final_nll", "raw_brier", "calibrated_brier")}
        t = read(bound(run, manifest["training"][name]))
        training[name] = t
        assert t["updates"] == 288 and t["adam_step"] == p["completed_updates"]
        assert t["encoder_positive_lr_updates"] == 287
        expected_ids = {uid for uid, domain in uid_domain.items() if domain == p["actual_domain"]}
        assert len(expected_ids) == 48 and Counter(t["current_ids"]) == Counter({uid: 6 for uid in expected_ids})
        for start in range(0, 288, 48):
            assert set(t["current_ids"][start:start + 48]) == expected_ids
        x = np.load(bound(run, t["update_file"]), allow_pickle=False)
        assert x.shape == (288, 14) and np.isfinite(x).all()
        c = {column: x[:, k] for k, column in enumerate(t["update_columns"])}
        for prefix in ("current", "history"):
            delta = c[prefix + "_total"] - c[prefix + "_bce"] - c[prefix + "_rank"] - .5 * c[prefix + "_hard"]
            loss_error = max(loss_error, float(np.abs(delta).max()))
            assert np.allclose(c[prefix + "_total"], c[prefix + "_bce"] + c[prefix + "_rank"] + .5 * c[prefix + "_hard"], atol=2e-6, rtol=1e-6)
        audit.array(c["total"], c["current_total"] + c["history_total"] + .5 * c["logit_mse"], name + "/loss")
        lr = np.array([1e-5 * (s / 29 if s <= 29 else (288 - s) / 259) for s in range(1, 289)])
        audit.array(c["encoder_lr"], lr, name + "/lr")
        audit.array(c["head_lr"], np.full(288, .001), name + "/head_lr")
        for step in (1, 29, 30, 288):
            for module in ("encoder", "head"):
                obs = t["observations"][str(step)][module]
                assert obs["finite_nonzero_combined_gradient"] is True
                assert obs["parameters_changed"] == (module == "head" or step < 288)
        if name.endswith("shared") or "_seq_" in name:
            assert not t["history_ids"]
            assert np.count_nonzero(x[:, 4:9]) == 0
        else:
            assert len(t["history_ids"]) == 288
            memory = read(run / "memory" / (name + "_budget.json"))
            assert 0 < memory["serialized_bytes"] <= 1048576 and len(memory["members"]) == 6
            assert memory["seen"] == 48 * (t["stage"] - 1)
            assert set(t["history_ids"]) <= set(memory["members"])
            assert not set(memory["members"]) & expected_ids
            diagnostics["memory"][name] = {
                "bytes_including_auxiliary": memory["serialized_bytes"],
                "domain_composition": dict(Counter(uid_domain[uid] for uid in memory["members"])),
                "history_presentations_by_domain": dict(Counter(uid_domain[uid] for uid in t["history_ids"])),
                "draw_count_range": [min(Counter(t["history_ids"]).values()), max(Counter(t["history_ids"]).values())],
                "references": memory["references"], "origins": memory["reference_origins"],
            }
        diagnostics["training"][name] = {
            "seconds": t["training_seconds"],
            "epoch_means": [{col: float(values[e * 48:(e + 1) * 48].mean()) for col, values in c.items()} for e in range(6)],
        }
        audit.checks["physical_updates"] += 288
        audit.checks["history_gradient_presentations"] += len(t["history_ids"])
    for order in ORDERS:
        first = mappings[order + "_shared"]
        for name, p in points.items():
            if p["order"] == order:
                audit.equal(p["first_map_parameters"], {k: first[k] for k in ("a", "b")}, name + "/first_map")
        for stage in (2, 3):
            paired = [training[point(order, arm, stage)] for arm in ("seq", "er", "logit")]
            assert all(t["current_ids"] == paired[0]["current_ids"] and t["current_dropout_stream"] == paired[0]["current_dropout_stream"] for t in paired)
            assert paired[1]["history_ids"] == paired[2]["history_ids"]
            a, b = (diagnostics["memory"][point(order, arm, stage)] for arm in ("er", "logit"))
            assert a["domain_composition"] == b["domain_composition"]
        early = read(run / "memory" / (order + "_logit_stage2_budget.json"))
        late = read(run / "memory" / (order + "_logit_stage3_budget.json"))
        for uid in set(early["references"]) & set(late["references"]):
            assert early["references"][uid] == late["references"][uid] and late["reference_origins"][uid] == 1
    assert audit.checks["physical_updates"] == 6048 and audit.checks["history_gradient_presentations"] == 3456
    entries = {"initial": {"raw": collected["initial"]}, **collected["points"]}
    scores["initial"] = {"development": np.load(bound(run, manifest["initial"]["scores"]), allow_pickle=False)}
    for name, variants in entries.items():
        matrices[name], counts[name] = {}, {}
        for role, info in variants.items():
            value = np.load(bound(ev, info["matrix"]), allow_pickle=False)
            assert value.shape == (60, 22) and value.dtype == np.float64 and np.isfinite(value).all()
            rows_counts = read(bound(ev, info["counts"]))
            assert len(rows_counts) == 60
            score = scores[name]["development" if role == "raw" else role]
            for i, row in enumerate(rows_counts):
                assert all(type(row[k]) is int and row[k] >= 0 for k in ("tp", "fp", "fn", "tn"))
                assert row["tp"] + row["fn"] == 20 and row["fp"] + row["tn"] == 358
                assert int((score[i] >= 0).sum()) == row["tp"] + row["fp"]
                for metric, expected in confusion(row).items():
                    audit.equal(value[i, METRICS.index(metric)], expected, name + role + metric)
            matrices[name][role], counts[name][role] = value, rows_counts
            audit.checks["matrix_count_sets"] += 1
        if name != "initial":
            for role in ("stage-cal", "first-cal"):
                audit.array(matrices[name][role][:, RANK], matrices[name]["raw"][:, RANK], name + "/rank_invariance", 0.)
            matrices[name]["primary"] = matrices[name]["stage-cal"].copy()
            matrices[name]["primary"][:, RANK] = matrices[name]["raw"][:, RANK]
    draws = np.random.Generator(np.random.PCG64(20260930)).integers(0, 20, (5000, 3, 20))
    assert np.array_equal(draws, np.load(bound(ev, saved["draws"]), allow_pickle=False))
    weights = np.zeros((5000, 60))
    for d, domain in enumerate("ABC"):
        weights[:, rows[domain]] = np.stack([np.bincount(row, minlength=20) for row in draws[:, d]]) / 20

    def field(arm: str, role: str, endpoint: str) -> np.ndarray:
        result = np.zeros((3, 60, 22))
        for oi, order in enumerate(ORDERS):
            for stage, arrival, coefficient in TERMS[endpoint]:
                selected = rows[order[arrival - 1]]
                result[oi, selected] += coefficient * matrices[point(order, arm, stage)][role][selected]
        if endpoint in ("F_first", "F", "G"):
            result[:, :, [4, 5]] *= -1
        return result

    def summary(value: np.ndarray) -> dict:
        per_order = value.sum(axis=1) / 20
        average = per_order.mean(axis=0)
        replicates = weights @ value.mean(axis=0)
        ordered = np.sort(replicates, axis=0)
        ci = []
        for fraction in (.025, .975):
            position = fraction * 4999
            lo, hi = math.floor(position), math.ceil(position)
            ci.append(ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo]))
        return {m: {"mean": float(average[k]), "per_order": {o: float(per_order[i, k]) for i, o in enumerate(ORDERS)},
                    "conditional_95pct_interval": [float(ci[0][k]), float(ci[1][k])]} for k, m in enumerate(METRICS)}

    # Independent small algebra checks, without the experiment's helpers.
    assert confusion(dict(tp=2, fp=1, fn=2, tn=5))["recall"] == .5
    handmade = np.zeros((3, 60, 22))
    handmade[:, rows["A"]] = 2
    handmade[:, rows["B"]] = 4
    handmade[:, rows["C"]] = 6
    audit.equal(summary(handmade)["map"], {"mean": 12., "per_order": dict.fromkeys(ORDERS, 12.), "conditional_95pct_interval": [12., 12.]}, "handmade_bootstrap")
    recomputed = {arm: {role: {ep: summary(field(arm, role, ep)) for ep in TERMS} for role in ROLES} for arm in ARMS}
    audit.equal(saved["endpoints"], recomputed, "endpoints")
    comparisons = {}
    signs = dict(average_precision=1, roc_auc=1, brier=-1, log_loss=-1)
    for candidate, reference in (("er", "seq"), ("logit", "er")):
        name = candidate + "_minus_" + reference
        delta = {ep: summary(field(candidate, "primary", ep) - field(reference, "primary", ep)) for ep in TERMS}
        raw_delta = {ep: summary(field(candidate, "primary", ep) - field(reference, "raw", ep)) for ep in ("O", "N")}
        audit.equal(saved["comparisons"][name]["primary"], delta, name)
        audit.equal(saved["comparisons"][name]["against_raw_reference"], raw_delta, name + "/raw")
        checks = {
            "old_map_improves": delta["O"]["map"]["mean"] > 0,
            "old_map_interval_above_zero": delta["O"]["map"]["conditional_95pct_interval"][0] > 0,
            "old_recall5_improves": delta["O"]["recall_at_5"]["mean"] > 0,
            "new_map_non_decrease": delta["N"]["map"]["mean"] >= 0,
            "new_recall5_non_decrease": delta["N"]["recall_at_5"]["mean"] >= 0,
        }
        for ep in ("O", "N"):
            for m, sign in signs.items():
                checks[f"{ep}_{m}_non_degradation"] = sign * delta[ep][m]["mean"] >= 0
            for m in ("brier", "log_loss"):
                checks[f"{ep}_{m}_against_raw_reference"] = raw_delta[ep][m]["mean"] <= 0
        for m, sign in dict(map=1, recall_at_5=1, **signs).items():
            checks[f"Z_{m}_non_degradation"] = sign * delta["Z"][m]["mean"] >= 0
        verdict = saved["comparisons"][name]["interpretation"]
        audit.equal(verdict["checks"], checks, name + "/checks")
        assert verdict["pilot_observed_checks_pass"] == all(checks.values())
        assert set(verdict["failed"]) == {k for k, value in checks.items() if not value}
        assert verdict["positive_old_map_orders"] == sum(x > 0 for x in delta["O"]["map"]["per_order"].values())
        comparisons[name] = {"passed": sum(checks.values()), "total": len(checks), "checks": checks}
    first = recomputed["seq"]["raw"]["F_first"]["map"]
    gain = recomputed["seq"]["raw"]["G"]["map"]
    matched = [o for o in ORDERS if first["per_order"][o] > 0 and gain["per_order"][o] > 0]
    fg = dict(seq_first_MAP_loss_positive=first["mean"] > 0,
              seq_first_MAP_loss_interval_above_zero=first["conditional_95pct_interval"][0] > 0,
              same_order_path_forgetting_and_new_learning=len(matched) >= 2)
    audit.equal(saved["seq_forgetting"]["checks"], fg, "forgetting_checks")
    assert saved["seq_forgetting"]["matched_orders"] == matched
    assert saved["seq_forgetting"]["established"] == all(fg.values())
    for name in expected_points:
        for role in ROLES[:3]:
            value = matrices[name][role]
            obj = saved["absolute_stage_results"][name][role]
            audit.equal(obj["macro_all"], dict(zip(METRICS, value.mean(0).tolist())), name + role + "/macro")
            for d in "ABC":
                audit.equal(obj["macro_by_domain"][d], dict(zip(METRICS, value[rows[d]].mean(0).tolist())), name + role + d)
            pooled = obj["pooled_fixed_half_classification"]
            assert pooled["threshold"] == 0
            for d, selected in {**rows, "pooled": np.arange(60)}.items():
                total = {k: sum(counts[name][role][i][k] for i in selected) for k in ("tp", "fp", "fn", "tn")}
                rates = confusion(total)
                expected = {**total, **{k: rates[k] for k in ("precision", "recall", "f1")}, "fpr": total["fp"] / (total["fp"] + total["tn"])}
                audit.equal(pooled["pooled"] if d == "pooled" else pooled["by_domain"][d], expected, name + role + "/pooled" + d)
    for order in ORDERS:
        selected = rows[order[0]]
        initial, trained = matrices["initial"]["raw"][selected], matrices[order + "_shared"]["raw"][selected]
        for key, value in (("initial_raw", initial), ("first_raw", trained), ("raw_first_minus_initial", trained - initial)):
            audit.equal(saved["initial_to_first_learning"][order][key], dict(zip(METRICS, value.mean(0).tolist())), order + key)
    with bound(ev, saved["stage_metrics"]).open(encoding="utf-8", newline="") as stream:
        seen = set()
        for row in csv.DictReader(stream):
            name = row["source_point"]
            assert row["seed"] == "s0" and row["actual_domain"] == row["order"][int(row["arrival_index"]) - 1]
            if int(row["stage"]) == 0:
                assert name == row["method"] == "initial" and row["output_role"] == "raw"
            else:
                assert name == point(row["order"], row["method"], int(row["stage"]))
            key = tuple(row[k] for k in ("order", "actual_domain", "stage", "method", "output_role", "metric"))
            assert key not in seen
            seen.add(key)
            value = matrices[name][row["output_role"]][rows[row["actual_domain"]], METRICS.index(row["metric"])].mean()
            audit.equal(float(row["group_macro"]), float(value), "stage_table")
        assert len(seen) == 198 + 3 * 3 * 4 * 3 * 3 * 22
        audit.checks["stage_csv_rows"] = len(seen)
    with (args.output / "endpoints.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("method", "role", "endpoint", "metric", "mean", "ci_lower", "ci_upper", *ORDERS))
        for arm in ARMS:
            for role in ROLES:
                for ep, summary_value in recomputed[arm][role].items():
                    for metric, obj in summary_value.items():
                        writer.writerow((arm, role, ep, metric, obj["mean"], *obj["conditional_95pct_interval"], *(obj["per_order"][o] for o in ORDERS)))
    diagnostics["descriptive_logit_minus_seq"] = {
        ep: {m: recomputed["logit"]["primary"][ep][m]["mean"] - recomputed["seq"]["primary"][ep][m]["mean"] for m in METRICS}
        for ep in TERMS}
    diagnostics["descriptive_scope"] = "Absolute differences only; LOGIT-minus-SEQ is not an added prospective test or acceptance rule."
    result = {
        "status": "PASS_SAVED_RESULT_AUDIT", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "job": str(job.relative_to(project)), "script": record(Path(__file__)),
        "checks": dict(audit.checks), "numeric_comparisons": audit.numbers, "maximum_absolute_error": audit.max_error,
        "float32_objective_identity_max_error": loss_error, "comparisons": comparisons,
        "seq_first_forgetting": first, "seq_new_learning": gain,
        "new_label_parses": dict(train=0, valid=0, heldout=0, owners=0),
        "formal_texts_or_weights_loaded": False,
        "limitations": ["No label-dependent AP/MAP recomputation; saved matrices are the authorized collection.",
                        "No refit, native model rerun or full history payload read; execution evidence and summary bindings checked.",
                        "Conditional 5000-draw intervals, reused 60 development groups, one training seed.",
                        "No per-domain noninferiority, new method qualification or independent heldout claim."],
    }
    for filename, payload in (("audit.json", result), ("diagnostics.json", diagnostics)):
        with (args.output / filename).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
