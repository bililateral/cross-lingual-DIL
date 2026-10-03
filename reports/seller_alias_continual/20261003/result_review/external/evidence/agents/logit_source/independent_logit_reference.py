#!/usr/bin/env python3
"""Independent LOGIT review: identifiers-only replay, target provenance, scalar loss.

Reads only submitted source/JSON metadata and saved handwritten CPU score vectors.
Does not import project code, Torch, a formal loader, text, labels, cache or weights.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys

import numpy as np


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def seed(base, *coordinates):
    # Re-express the frozen byte protocol, without calling the project helper.
    text = json.dumps([base, *coordinates], ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False) + "\n"
    return int(digest(text.encode())[:16], 16) % ((1 << 63) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    project, out = args.project.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    workspace = project / "reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace"
    job = workspace / "reports/job"
    original = project / "reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job"
    weight = project / "reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job"
    cpu = workspace.parent / "cpu"
    accesses = {}

    def read(path):
        path = path.resolve()
        payload = path.read_bytes()
        accesses[str(path.relative_to(project))] = {"bytes": len(payload), "sha256": digest(payload)}
        return json.loads(payload)

    def save(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    policy = read(workspace / "schema/step28_bge_continual_policy.json")
    logit_policy = read(workspace / "schema/step28_logit_weight_policy.json")
    manifest = read(job / "run/manifest.json")
    baseline = read(original / "run/manifest.json")
    quarter = read(weight / "run/manifest.json")
    partition = read(original / "run/partition.json")
    assert read(job / "run/partition.json") == partition

    # 34 actual source identities and exact correspondence to the separate ER root.
    source_rows = []
    for row in manifest["source_files"]:
        path = workspace / row["path"]
        payload = path.read_bytes()
        assert len(payload) == row["bytes"] and digest(payload) == row["sha256"], row["path"]
        common = project / row["path"]
        common_digest = digest(common.read_bytes()) if common.is_file() else None
        source_rows.append({**row, "actual_path": str(path.relative_to(project)),
                            "root_path": row["path"], "root_sha256": common_digest,
                            "same_bytes_as_root": common_digest == row["sha256"]})
    assert len(source_rows) == 34 and len({r["path"] for r in source_rows}) == 34
    save("source_correspondence.json", source_rows)

    fit = {d: sorted(r["group_uid"] for r in partition["fit"] if r["domain"] == d) for d in "ABC"}
    assert all(len(v) == 48 for v in fit.values())
    group_domain = {uid: d for d, values in fit.items() for uid in values}
    target_results, schedule_results, trace_rows = {}, {}, []
    for order in ("ABC", "BCA", "CAB"):
        # Algorithm R from identifiers only, not serialized sample payloads or RNG.
        retention_rng = random.Random(seed(policy["memory_seed"], order, "retention"))
        reservoir, snapshots = [], {}
        seen = 0
        for arrival, domain in enumerate(order[:2], 1):
            for uid in fit[domain]:
                seen += 1
                slot = len(reservoir) if len(reservoir) < 6 else retention_rng.randrange(seen)
                if slot < 6:
                    if slot == len(reservoir):
                        reservoir.append(uid)
                    else:
                        reservoir[slot] = uid
            snapshots[arrival] = reservoir.copy()
        start = manifest["restored_starts"][order + "_logit_quarter"]
        initial = baseline["memories"][order + "_logit_after1"]
        old_er = baseline["memories"][order + "_er_after1"]
        retained = manifest["memories"][order + "_logit_quarter_stage2"]
        original_old_logit = baseline["memories"][order + "_logit_stage2"]
        assert snapshots[1] == initial["members"] == old_er["members"] == start["memory_summary"]["members"]
        assert snapshots[2] == retained["members"] == baseline["memories"][order + "_er_stage2"]["members"]
        assert start["memory_source"] == initial["file"]
        for key in ("references", "reference_origins"):
            assert start["memory_summary"][key] == initial[key]
        assert set(initial["reference_origins"].values()) == {1}
        survivors = sorted(set(initial["references"]) & set(retained["references"]))
        newcomers = sorted(set(retained["references"]) - set(initial["references"]))
        assert survivors == sorted(set(snapshots[1]) & set(snapshots[2]))
        assert newcomers == sorted(set(snapshots[2]) - set(snapshots[1]))
        assert all(initial["references"][uid] == retained["references"][uid] for uid in survivors)
        assert all(retained["reference_origins"][uid] == 1 for uid in survivors)
        assert all(retained["reference_origins"][uid] == 2 for uid in newcomers)
        assert all(group_domain[uid] == order[origin - 1] for uid, origin in retained["reference_origins"].items())
        generation = retained["reference_update"]
        assert generation["mode"] == "eval" and generation["old_survivors_unchanged"] is True
        assert sorted(generation["new_target_ids"]) == newcomers
        stage2 = read(job / "run" / manifest["points"][order + "_logit_quarter_stage2"]["path"])
        assert generation["model_state_sha256"] == stage2["model_state_sha256"]
        assert set(retained["references"]) == set(snapshots[2])
        hashes_different_from_old_logit = {uid: retained["references"][uid] != original_old_logit["references"][uid]
                                         for uid in newcomers}
        assert all(hashes_different_from_old_logit.values())
        target_results[order] = {
            "first_members": snapshots[1], "stage2_members": snapshots[2],
            "old_survivors_unchanged": survivors, "new_target_ids": newcomers,
            "new_hashes_differ_from_old_LOGIT1": hashes_different_from_old_logit,
            "origin_stage2_model_sha256": generation["model_state_sha256"],
            "stage3_available_counts_by_domain": {d: sum(group_domain[x] == d for x in snapshots[2]) for d in "ABC"},
            "scope": "Target content is absent and not requested; verifies saved origins/digests and source causal order only."
        }
        for stage in (2, 3):
            name = f"{order}_logit_quarter_stage{stage}"
            log = read(job / "run" / manifest["training"][name]["path"])
            original_log = read(original / "run" / baseline["training"][f"{order}_er_stage{stage}"]["path"])
            quarter_log = read(weight / "run" / quarter["training"][f"{order}_quarter_stage{stage}"]["path"])
            stream = seed(policy["schedule_seed"], order, stage, "current")
            schedule_rng = random.Random(stream)
            current_ids = []
            for epoch in range(6):
                rows = fit[order[stage - 1]].copy()
                schedule_rng.shuffle(rows)
                current_ids.extend(rows)
            history_rng = random.Random(seed(policy["memory_seed"], order, stage, "history_draws"))
            history_ids = [snapshots[stage - 1][history_rng.randrange(6)] for _ in range(288)]
            assert current_ids == log["current_ids"] == original_log["current_ids"] == quarter_log["current_ids"]
            assert history_ids == log["history_ids"] == original_log["history_ids"] == quarter_log["history_ids"]
            assert stream == log["current_dropout_stream"] == original_log["current_dropout_stream"] == quarter_log["current_dropout_stream"]
            expected_targets = initial if stage == 2 else retained
            for key in ("references", "reference_origins", "members"):
                assert log["memory_after_training"][key] == expected_targets[key]
            for index, (current_uid, history_uid) in enumerate(zip(current_ids, history_ids)):
                trace_rows.append({"order": order, "stage": stage, "step": index + 1,
                    "current_id": current_uid, "history_id": history_uid,
                    "current_dropout_seed_from_protocol": seed(stream, index, "dropout"),
                    "history_dropout_seed_from_protocol": seed(policy["memory_seed"], order, stage, index, "history_dropout")})
            schedule_results[name] = {"current_rows": len(current_ids), "history_rows": len(history_ids),
                "matches_original_ER_and_ER025": True, "current_stream": stream,
                "cache_targets_match_initial_or_retained_stage": True}
    with (out / "independent_paired_schedule.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)
    assert len(trace_rows) == 1728
    save("target_and_schedule_results.json", {"targets": target_results, "schedules": schedule_results,
         "gradient_presentations": 2 * len(trace_rows),
         "randomness_scope": "Python Random schedule/draws reconstructed from only supplied fitting IDs; per-forward dropout seeds derived, no Torch kernels rerun."})

    # Handwritten relation: four triples then eight pairs. No formal truth is read.
    blocks = [list(range(i, i + 3)) for i in (0, 3, 6, 9)] + [list(range(i, i + 2)) for i in range(12, 28, 2)]
    positive_matrix = np.zeros((28, 28), dtype=bool)
    for block in blocks:
        for i in block:
            for j in block:
                positive_matrix[i, j] = i != j
    pairs = [(i, j) for i in range(28) for j in range(i + 1, 28)]
    edge = np.full((28, 28), -1, dtype=int)
    for k, (i, j) in enumerate(pairs):
        edge[i, j] = edge[j, i] = k
    y = np.array([positive_matrix[i, j] for i, j in pairs], dtype=float)
    assert len(y) == 378 and y.sum() == 20

    def sigmoid(values):
        return np.exp(-np.logaddexp(0., -np.asarray(values)))

    def base_and_gradient(z):
        z = np.asarray(z, dtype=float)
        bce = float(np.mean(np.logaddexp(0., z) - y * z))
        gradients = (sigmoid(z) - y) / 378
        rank, hard = 0., 0.
        for q in range(28):
            candidates = [j for j in range(28) if j != q]
            ids = edge[q, candidates]
            pos = [edge[q, j] for j in candidates if positive_matrix[q, j]]
            neg = sorted([j for j in candidates if not positive_matrix[q, j]], key=lambda j: (-z[edge[q, j]], j))[:5]
            val = z[ids]
            m = val.max()
            exp = np.exp(val - m)
            rank += (m + math.log(float(exp.sum())) - float(np.mean(z[pos]))) / 28
            gradients[ids] += exp / exp.sum() / 28
            gradients[pos] -= 1 / (28 * len(pos))
            for positive_edge in pos:
                for negative_account in neg:
                    negative_edge = edge[q, negative_account]
                    difference = z[negative_edge] - z[positive_edge]
                    divisor = 28 * len(pos) * 5
                    hard += float(np.logaddexp(0., difference)) / divisor
                    g = .5 * float(sigmoid(difference)) / divisor
                    gradients[negative_edge] += g
                    gradients[positive_edge] -= g
        return {"bce": bce, "rank": rank, "hard": hard, "total": bce + rank + .5 * hard}, gradients

    cpu_results = {}
    for arm in ("quarter", "logit_quarter"):
        native = read(cpu / (arm + ".json"))
        current = np.asarray(native["captured_current_logits"])
        history = np.asarray(native["captured_history_logits"])
        current_loss, gc = base_and_gradient(current)
        history_loss, gh = base_and_gradient(history)
        mse = 0.
        if arm == "logit_quarter":
            target = np.asarray(native["origin_eval_reference"])
            mse = float(np.dot(history - target, history - target) / 378)
        observed = native["weighted_update"]
        exact = current_loss["total"] + .25 * history_loss["total"] + .5 * mse
        differences = {role + "_" + key: abs(observed[role + "_" + key] - value)
                       for role, item in (("current", current_loss), ("history", history_loss))
                       for key, value in item.items()}
        differences["total"] = abs(observed["total"] - exact)
        differences["mse"] = abs(observed.get("logit_mse", 0.) - mse)
        assert max(differences.values()) < 1e-6
        cpu_results[arm] = {"independent_current": current_loss, "independent_history": history_loss,
                           "independent_mse": mse, "independent_total": exact,
                           "recorded_total": observed["total"], "absolute_errors": differences,
                           "not_verified": "No saved parameter-gradient tensors; native vector gradient equality is not independently recalculated here."}

    # Finite differences verify our independent analytical logit gradient and a
    # shared-parameter composite objective, using only an explicitly synthetic example.
    t = np.arange(378, dtype=float)
    zc = .8 * np.sin(t * .37) + t / 509
    zh = .9 * np.cos(t * .43) - t / 701
    target = .3 * np.cos(t * .29) + .1
    c_loss, gc = base_and_gradient(zc)
    h_loss, gh = base_and_gradient(zh)
    mse_gradient = (zh - target) / 378  # derivative of coefficient 0.5 mean square
    gradient = np.r_[gc, .25 * gh + mse_gradient]

    def composite(z):
        cur, hist = z[:378], z[378:]
        return (base_and_gradient(cur)[0]["total"] + .25 * base_and_gradient(hist)[0]["total"]
                + .5 * float(np.dot(hist - target, hist - target) / 378))

    joined = np.r_[zc, zh]
    coordinate_results = []
    for k in (0, 1, 8, 15, 33, 60, 120, 210, 300, 377, 378, 381, 390, 455, 576, 700, 750, 755):
        plus, minus = joined.copy(), joined.copy()
        plus[k] += 1e-6
        minus[k] -= 1e-6
        fd = (composite(plus) - composite(minus)) / (2e-6)
        coordinate_results.append({"coordinate": k, "finite_difference": fd,
                                   "analytic": float(gradient[k]), "absolute_error": abs(fd - gradient[k])})
    max_fd = max(float(r["absolute_error"]) for r in coordinate_results)
    assert max_fd < 1e-7
    # Seven linear parameters influence both current and history logits.
    design = np.stack([np.sin(np.arange(756) * (.013 + i * .019)) / (i + 1) for i in range(7)], axis=1)
    expected_parameter_gradient = design.T @ gradient
    parameter_results = []
    for k in range(7):
        delta = 1e-6 * design[:, k]
        fd = (composite(joined + delta) - composite(joined - delta)) / (2e-6)
        parameter_results.append({"parameter": k, "finite_difference": fd,
            "analytic": float(expected_parameter_gradient[k]), "absolute_error": abs(fd - expected_parameter_gradient[k])})
    max_param_fd = max(float(r["absolute_error"]) for r in parameter_results)
    assert max_param_fd < 1e-7
    wrong = np.r_[gc, .25 * gh + .25 * mse_gradient]
    gap = float(np.max(np.abs(gradient - wrong)))
    assert gap > 1e-4
    mathematical_results = {"current_loss": c_loss, "history_loss": h_loss,
        "independent_total": composite(joined), "finite_differences": coordinate_results,
        "linear_shared_parameter_finite_differences": parameter_results,
        "maximum_logit_fd_error": max_fd, "maximum_parameter_fd_error": max_param_fd,
        "wrong_MSE_coefficient_0125_logit_gradient_max_gap": gap,
        "scope": "NumPy mathematical example only, no model or optimizer is trained and no actual Torch/Adam/GPU gradient is re-created."}
    save("independent_loss_results.json", {"saved_handwritten_native_cpu": cpu_results,
                                          "synthetic_gradient_reference": mathematical_results})
    save("read_accesses.json", accesses)
    summary = {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "source_files_checked": 34, "source_files_same_bytes_as_root": sum(r["same_bytes_as_root"] for r in source_rows),
        "source_files_different_or_absent_in_root": [r["path"] for r in source_rows if not r["same_bytes_as_root"]],
        "reconstructed_updates": len(trace_rows), "target_survivor_counts": {o: len(v["old_survivors_unchanged"]) for o,v in target_results.items()},
        "new_target_counts": {o: len(v["new_target_ids"]) for o,v in target_results.items()},
        "maximum_native_cpu_scalar_error": max(max(v["absolute_errors"].values()) for v in cpu_results.values()),
        "maximum_synthetic_logit_fd_error": max_fd, "maximum_synthetic_parameter_fd_error": max_param_fd,
        "wrong_mse_coefficient_gradient_gap": gap, "status": "PASS_WITH_STATED_COVERAGE"}
    save("summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
