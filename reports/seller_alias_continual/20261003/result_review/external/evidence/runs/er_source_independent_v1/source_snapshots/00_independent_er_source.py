#!/usr/bin/env python3
"""ER0.1 independent review: saved small evidence and handwritten CPU algebra only.

No project functions imported. No formal text/labels/cache bodies/weights are opened.
This does not execute a neural model or independently reproduce native training.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
# evidence/agents/er_source -> review_work
PROJECT = HERE.parents[2] / "input" / "project"
ER_REL = "reports/seller_alias_continual/20261002/er_low_execution/20261002_124018"
ER = PROJECT / ER_REL
OLD = PROJECT / "reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job"
WEIGHT = PROJECT / "reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job"
READ = {}

def read_bytes(path):
    raw = path.read_bytes()
    READ[str(path.relative_to(PROJECT))] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return raw

def read_json(path):
    return json.loads(read_bytes(path))

def read_array(path):
    import io
    return np.load(io.BytesIO(read_bytes(path)), allow_pickle=False)

def canonical(x):
    return (json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()

def fork_seed(root, *tags):
    digest = hashlib.sha256(canonical([root, *tags])).digest()
    return int(digest[:8].hex(), 16) % (2**63 - 1)

def close(x, y, tol=1e-12):
    d = float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
    assert d <= tol, (d, tol)
    return d

N = 28
EDGES = list(itertools.combinations(range(N), 2))
LOOKUP = {e: k for k, e in enumerate(EDGES)}
COMPONENT = np.repeat(np.arange(12), [3] * 4 + [2] * 8)
Y = np.array([COMPONENT[a] == COMPONENT[b] for a, b in EDGES], dtype=np.float64)

def loss_derivatives(scores, target=Y):
    """Independent adjacency-free edge traversal and analytic logit derivatives.

    BCE is a mean over 378 edges; every query's all-positive rank and hard
    terms are averaged over 28 queries. Edges incident to two queries collect
    both contributions. Hard index selection alone is treated as detached.
    """
    z = np.asarray(scores, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    assert z.shape == y.shape == (378,)
    bce = math.fsum(math.log1p(math.exp(-abs(float(s)))) + max(float(s), 0.) - float(t) * float(s)
                    for s, t in zip(z, y)) / len(EDGES)
    gbce = (1. / (1. + np.exp(-z)) - y) / len(EDGES)
    rank = hard = 0.
    grank, ghard = np.zeros(378), np.zeros(378)
    selected_all = []
    for q in range(N):
        candidates = [a for a in range(N) if a != q]
        ix = [LOOKUP[tuple(sorted((a, q)))] for a in candidates]
        pos = [k for k in ix if y[k] == 1]
        neg = sorted([(z[k], a, k) for a, k in zip(candidates, ix) if y[k] == 0],
                     key=lambda v: (-v[0], v[1]))[:5]
        selected = [k for _, _, k in neg]
        selected_all.append(selected)
        top = float(max(z[ix]))
        exp = np.exp(z[ix] - top)
        denom = math.fsum(exp.tolist())
        rank += (top + math.log(denom) - math.fsum(z[pos].tolist()) / len(pos)) / N
        for k, probability in zip(ix, exp / denom):
            grank[k] += probability / N
        for k in pos:
            grank[k] -= 1. / (N * len(pos))
        for p in pos:
            for n in selected:
                difference = float(z[n] - z[p])
                term = math.log1p(math.exp(-abs(difference))) + max(difference, 0.)
                scale = 1. / (N * len(pos) * len(selected))
                hard += term * scale
                derivative = 1. / (1. + math.exp(-difference)) * scale
                ghard[n] += derivative
                ghard[p] -= derivative
    return ({"bce": bce, "rank": rank, "hard": hard, "total": bce + rank + .5 * hard},
            {"bce": gbce, "rank": grank, "hard": ghard, "total": gbce + grank + .5 * ghard},
            selected_all)

def independent_checks():
    policy = read_json(PROJECT / "schema/step28_bge_continual_policy.json")
    low = read_json(PROJECT / "schema/step28_er_low_policy.json")
    manifest = read_json(ER / "job/run/manifest.json")
    partition = read_json(ER / "job/run/partition.json")
    baseline = read_json(OLD / "run/manifest.json")
    reference = read_json(WEIGHT / "run/manifest.json")
    assert len(manifest["source_files"]) == 31
    verified_sources = []
    for rec in manifest["source_files"]:
        raw = read_bytes(PROJECT / rec["path"])
        assert len(raw) == rec["bytes"] and hashlib.sha256(raw).hexdigest() == rec["sha256"]
        verified_sources.append(rec)
    assert low["arms"] == {"tenth": .1}
    assert manifest["physical_updates"] == 1728 and manifest["gradient_group_presentations"] == 3456
    assert partition == read_json(OLD / "run/partition.json") == read_json(WEIGHT / "run/partition.json")
    fit = {d: [r["group_uid"] for r in partition["fit"] if r["domain"] == d] for d in "ABC"}
    cal = {d: [r["group_uid"] for r in partition["calibration"] if r["domain"] == d] for d in "ABC"}
    all_ids = [r["group_uid"] for role in ("fit", "calibration", "development") for r in partition[role]]
    assert len(all_ids) == len(set(all_ids)) == 240
    for d in "ABC":
        assert (len(fit[d]), len(cal[d])) == (48, 12)
        ranked = sorted(fit[d] + cal[d], key=lambda uid: hashlib.sha256(canonical([20260918, d, uid])).hexdigest())
        assert set(ranked[:48]) == set(fit[d])
        assert len([r for r in partition["development"] if r["domain"] == d]) == 20
    stages = {}
    max_loss_error = 0.
    for order in ("ABC", "BCA", "CAB"):
        keep_rng = random.Random(fork_seed(policy["memory_seed"], order, "retention"))
        reservoir, seen = [], 0
        def insert(domain):
            nonlocal seen
            for uid in sorted(fit[domain]):
                seen += 1
                if seen <= 6:
                    reservoir.append(uid)
                else:
                    j = keep_rng.randrange(seen)
                    if j < 6:
                        reservoir[j] = uid
        insert(order[0])
        start = manifest["restored_starts"][order + "_tenth"]
        shared = read_json(OLD / "run" / baseline["points"][order + "_shared"]["path"])
        assert start["memory_summary"]["members"] == reservoir
        assert start["memory_source"] == baseline["memories"][order + "_er_after1"]["file"]
        assert start["full_checkpoint"] == shared["full_checkpoint"]
        assert start["model_state_sha256"] == shared["model_state_sha256"]
        assert start["first_map"] == shared["first_map_parameters"]
        assert start["adam_step"] == 288 and start["first_scores_replayed_exactly"] is True
        for arm in ("half", "quarter"):
            oldstart = reference["restored_starts"][order + "_" + arm]
            for key in ("full_checkpoint", "model_state_sha256", "memory_source", "first_map", "adam_step"):
                assert oldstart[key] == start[key]
        for stage in (2, 3):
            name = f"{order}_tenth_stage{stage}"
            log = read_json(ER / "job/run/updates" / (name + ".json"))
            point = read_json(ER / "job/run/points" / (name + ".json"))
            stream = fork_seed(policy["schedule_seed"], order, stage, "current")
            rng = random.Random(stream)
            schedule = []
            for epoch in range(6):
                batch = sorted(fit[order[stage - 1]])
                rng.shuffle(batch)
                schedule += batch
            draw_rng = random.Random(fork_seed(policy["memory_seed"], order, stage, "history_draws"))
            history = [reservoir[draw_rng.randrange(6)] for _ in range(288)]
            assert log["current_ids"] == schedule and log["current_dropout_stream"] == stream
            assert log["history_ids"] == history
            assert log["memory_after_training"]["members"] == reservoir
            assert log["memory_after_training"]["draw_count"] == 288
            assert not log["memory_after_training"]["with_logits"]
            assert log["memory_after_training"]["seen"] == 48 * (stage - 1)
            assert log["memory_after_training"]["serialized_bytes"] <= 1048576
            paired_records = [read_json(OLD / "run/updates" / f"{order}_{arm}_stage{stage}.json")
                              for arm in ("seq", "er")]
            paired_records += [read_json(WEIGHT / "run/updates" / f"{order}_{arm}_stage{stage}.json")
                               for arm in ("half", "quarter")]
            for j, paired in enumerate(paired_records):
                assert paired["current_ids"] == schedule and paired["current_dropout_stream"] == stream
                if j:
                    assert paired["history_ids"] == history
            values = read_array(ER / "job/run" / log["update_file"]["path"])
            assert values.dtype == np.float64 and values.shape == (288, 14) and np.isfinite(values).all()
            columns = dict(zip(log["update_columns"], values.T))
            for role in ("current", "history"):
                error = close(columns[role + "_total"], columns[role + "_bce"] + columns[role + "_rank"]
                              + .5 * columns[role + "_hard"], 3e-6)
                max_loss_error = max(error, max_loss_error)
            assert np.all(columns["history_weight"] == .1)
            close(columns["weighted_history_total"], .1 * columns["history_total"], 0.)
            close(columns["total"], columns["current_total"] + .1 * columns["history_total"], 0.)
            expected_lr = np.array([1e-5 * (s / 29 if s <= 29 else (288 - s) / 259) for s in range(1, 289)])
            close(columns["encoder_lr"], expected_lr, 0.)
            assert np.all(columns["head_lr"] == .001)
            assert log["adam_step"] == point["completed_updates"] == 288 * stage
            assert point["full_model_adam_and_rng_restore_verified"] is True
            assert point["first_map_parameters"] == start["first_map"]
            assert set(log["observations"]) == {"1", "29", "30", "288"}
            for s, module in log["observations"].items():
                for m in ("encoder", "head"):
                    assert module[m]["finite_nonzero_combined_gradient"] is True
                    assert module[m]["parameters_changed"] == (m == "head" or int(s) != 288)
            mapping = read_json(ER / "job/run" / point["mapping"]["path"])
            assert mapping["calibration_group_ids"] == cal[order[stage - 1]]
            assert mapping["actual_domain"] == order[stage - 1]
            assert mapping["pair_count"] == 4536 and mapping["positive_count"] == 240
            assert mapping["model_state_sha256"] == point["model_state_sha256"]
            assert mapping["score_source"] == point["scores"]["calibration"]
            assert mapping["a"] > 0 and mapping["optimizer_success"]
            assert mapping["projected_gradient_max"] <= 1e-6
            stages[name] = {"updates": len(schedule), "current_distinct": len(set(schedule)),
                            "history_distinct": len(set(history)), "history_members": reservoir.copy(),
                            "history_counts": {u: history.count(u) for u in reservoir},
                            "current_dropout_stream": stream,
                            "current_and_history_seed_example": [fork_seed(stream, 0, "dropout"),
                                fork_seed(policy["memory_seed"], order, stage, 0, "history_dropout")],
                            "adam_step": log["adam_step"], "encoder_positive_lr_updates": int((expected_lr > 0).sum()),
                            "memory_bytes": log["memory_after_training"]["serialized_bytes"],
                            "gradient_norm_range": [float(columns["gradient_norm"].min()), float(columns["gradient_norm"].max())],
                            "train_seconds": log["training_seconds"], "calibration_a_b": [mapping["a"], mapping["b"]]}
            if stage == 2:
                insert(order[1])
                assert manifest["memories"][name]["members"] == reservoir

    native = {}
    for arm, weight in (("quarter", .25), ("tenth", .1)):
        rec = read_json(ER / "cpu" / (arm + ".json"))
        independent = {}
        for role in ("current", "history"):
            losses, _, _ = loss_derivatives(rec[f"captured_{role}_logits"])
            for key, value in losses.items():
                close(value, rec["weighted_update"][role + "_" + key], 4e-6)
            independent[role] = losses
        total = independent["current"]["total"] + weight * independent["history"]["total"]
        error = close(total, rec["weighted_update"]["total"], 6e-6)
        assert rec["native_updates_actually_executed"] == 2
        assert len(rec["parameter_probes"]) == 29 and rec["backbone_layers"] == 24
        assert all(p["parameters_changed"] and p["gradient_norm_after_clip"] > 0 for p in rec["parameter_probes"])
        native[arm] = {"independent_losses": independent, "total": total,
                       "native_total": rec["weighted_update"]["total"], "total_abs_error": error,
                       "native_actual_updates": rec["native_updates_actually_executed"],
                       "native_counter_fixture": rec["counter_fixture"],
                       "probe_count": len(rec["parameter_probes"])}
    a, b = (read_json(ER / "cpu" / (arm + ".json")) for arm in ("quarter", "tenth"))
    for field in ("initial_model_state_sha256", "after_warm_model_state_sha256", "before_stage2_optimizer_sha256",
                  "captured_current_logits", "captured_history_logits"):
        assert a[field] == b[field]
    audit = read_json(ER / "cpu/audit.json")
    native["native_saved_gradient_scaling"] = audit["native_component_comparison"]
    # Hooks' saved norms corroborate scalar scaling. Elementwise native gradient
    # arrays were not delivered, so this review cannot independently recompare them.
    for name in a["gradient_components"]:
        ca, cb = a["gradient_components"][name], b["gradient_components"][name]
        assert ca["current"] == cb["current"]
        close(ca["weighted_history"]["norm"], 2.5 * cb["weighted_history"]["norm"], 1e-7)

    # One new handwritten graph with fixed deterministic scores and Jacobians.
    k = np.arange(378, dtype=float)
    current = .7 * np.sin(k * .317) + .002 * k - .4
    history = .9 * np.cos(k * .213) - .001 * k + .2
    lc, gc, _ = loss_derivatives(current)
    lh, gh, _ = loss_derivatives(history)
    h = 1e-5
    finite_differences = []
    for scores, gradients in ((current, gc), (history, gh)):
        for term in ("bce", "rank", "hard", "total"):
            for index in (0, 1, 17, 96, 177, 283, 377):
                positive, negative = scores.copy(), scores.copy()
                positive[index] += h
                negative[index] -= h
                numerical = (loss_derivatives(positive)[0][term] - loss_derivatives(negative)[0][term]) / (2 * h)
                finite_differences.append(close(numerical, gradients[term][index], 2e-9))
    xc = np.column_stack((np.sin(k / 13), np.cos(k / 17), np.ones(378)))
    xh = np.column_stack((np.cos(k / 19), np.sin(k / 23), .2 + k / 500))
    combined = xc.T @ gc["total"] + .1 * (xh.T @ gh["total"])
    def joint(theta):
        return loss_derivatives(current + xc @ theta)[0]["total"] + .1 * loss_derivatives(history + xh @ theta)[0]["total"]
    chain_errors = []
    for i in range(3):
        offset = np.zeros(3)
        offset[i] = h
        numeric = (joint(offset) - joint(-offset)) / (2 * h)
        chain_errors.append(close(numeric, combined[i], 2e-9))
    correct = lc["total"] + .1 * lh["total"]
    wrong_bce_only = lc["total"] + .1 * lh["bce"] + lh["rank"] + .5 * lh["hard"]
    wrong_normalization = correct / 1.1
    assert abs(correct - wrong_bce_only) > 1 and abs(correct - wrong_normalization) > .1
    clip_small = .01 / (np.linalg.norm(combined) + 1e-6)
    clipped = combined * min(1., clip_small)
    # Transparent one-step AdamW arithmetic from arbitrary handmade existing moments.
    theta = np.array([.3, -.7, .2])
    previous_m, previous_v = np.array([.01, -.02, .015]), np.array([.02, .03, .01])
    m = .9 * previous_m + .1 * clipped
    v = .999 * previous_v + .001 * clipped ** 2
    lr = np.array([1e-5 / 29, 1e-5 / 29, .001])
    decay = np.array([.01, .01, 0.])
    next_theta = theta * (1 - lr * decay) - lr * (m / (1 - .9 ** 289)) / (np.sqrt(v / (1 - .999 ** 289)) + 1e-8)
    assert np.all(next_theta != theta)
    handmade = {"new_handwritten_only": True, "logit_finite_difference_count": len(finite_differences),
                "logit_finite_difference_max_error": max(finite_differences),
                "shared_parameter_chain_count": 3, "chain_max_error": max(chain_errors),
                "current": lc, "history": lh, "correct_total": correct,
                "wrong_BCE_only_weight_total": wrong_bce_only, "wrong_divide_by_1_1_total": wrong_normalization,
                "combined_gradient": combined.tolist(), "explicit_clip_coefficient": float(min(1., clip_small)),
                "handwritten_adamw_next_theta": next_theta.tolist(),
                "scope": "Independent logit analytic derivatives and finite differences; NumPy AdamW example, not project Torch execution"}
    return {"source_records": verified_sources, "source_count": len(verified_sources), "stage_checks": stages,
            "all_1728_update_decomposition_max_abs_error": max_loss_error,
            "native_saved_evidence_reference": native, "handwritten": handmade}

def main():
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    result = independent_checks()
    result.update(status="PASS_INDEPENDENT_ER_SOURCE_SAVED_EVIDENCE_AND_HANDWRITTEN_ALGEBRA",
                  started_at_utc=started, ended_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  environment={"python": platform.python_version(), "numpy": np.__version__,
                               "platform": platform.platform(), "executable": sys.executable,
                               "affinity": sorted(os.sched_getaffinity(0)),
                               "torch_available": importlib.util.find_spec("torch") is not None},
                  input_files=READ,
                  boundaries={"formal_text_reads": 0, "formal_label_reads": 0, "cache_body_reads": 0,
                              "weight_reads": 0, "server_connections": 0, "training": 0,
                              "test_or_owners_reads": 0, "project_modules_imported": 0,
                              "native_training_rerun": False,
                              "native_gradient_arrays_independently_recomputed": False})
    path = HERE / "independent_er_source_result.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "source_count", "stage_checks",
        "all_1728_update_decomposition_max_abs_error", "native_saved_evidence_reference", "handwritten", "environment", "boundaries")},
        ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
