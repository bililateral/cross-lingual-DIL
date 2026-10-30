"""Two fixed positive-rank arms; shared record lifecycle, one joint valid opening."""
from __future__ import annotations

import argparse
import ast
from functools import partial
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import threading
import time
import traceback
import hashlib

import numpy as np

import step28_positive_ranking as method
import step28_g_half_run as single
import step28_h_consistency_run as baselines

record, data, parent, core = single.record, single.data, single.parent, single.core
base, previous, diag = single.base, single.previous, single.diag
POLICY = data.ROOT/"schema/step28_positive_ranking_policy.json"
BASELINE = baselines.BASELINE
AUTHORIZATION = data.ROOT/"reports/documentation/20261016/positive_ranking_implementation/authorization.json"
CONTRACT = data.ROOT/"docs/SELLER_ALIAS_POSITIVE_RANKING.zh.md"
POINTS = tuple(stage+"_"+arm for arm in method.ARMS for stage in "AB")
COMPLETE = "COMPLETE_1152_UPDATES_4_RESTORES_24_DIAGNOSTICS_VALID_BLIND"
COMPLETION_STATUS = "COMPLETE_S_POS_H_POS_FIXED_AB"
CPU_TESTS = ("test_rank_reference_and_rejected_mutants", "test_views_history_and_actual_vjp_adamw",
             "test_capture_restore_and_next_update", "test_dual_orchestration_and_joint_opening",
             "test_saved_comparisons_and_permissions", "test_stage_a_empty_cache",
             "test_native_continuation_preflight")
fresh, pretrained_archive = single.fresh, single.pretrained_archive
public_inputs, attach_labels, partition = single.public_inputs, single.attach_labels, single.partition
baseline_runs, baseline_inputs = baselines.baseline_runs, baselines.baseline_inputs
query_metrics, undefined_metrics = single.query_metrics, single.undefined_metrics
pooled_rates, summarize = single.pooled_rates, single.summarize
trajectory = single.trajectory


def policy():
    p = data.read_json(POLICY)
    old = single.policy()
    expected = {**old, "study": "fixed_positive_rank_S_and_H", "methods": list(method.ARMS),
        "physical_updates": 1152, "gradient_group_presentations": 1728,
        "physical_stage_states": 4, "epoch_diagnostic_points": 24,
        "cpu_verification": {**old["cpu_verification"], "maximum_seconds": 300},
        "native_verification": {**old["native_verification"], "maximum_seconds": 1200,
                                "per_arm_maximum_seconds": 600, "actual_updates": 12},
        "runtime": {**old["runtime"], "maximum_gpu_stage_seconds": 57600, "maximum_output_bytes": 16*2**30},
        "candidates": {arm: method.metadata(arm) for arm in method.ARMS},
        "evaluation": dict(bootstrap_replicates=5000, bootstrap_seed=20260930,
            comparisons=[["S_pos", "S"], ["H_pos", "S"], ["H_pos", "G_half"], ["H_pos", "S_pos"]],
            primary_metric="map", primary_role="raw",
            success="B_end_S_pos_minus_S_each_AB_MAP_CI_lower_strictly_positive")}
    del expected["pooling"]
    if p != expected:
        raise ValueError("Fixed dual positive-rank policy differs")
    return p


def config():
    c = record.config()
    c["runtime"] = policy()["runtime"]
    return c


def sources():
    paths = {data.ROOT/n for n in ("scripts/step28_positive_ranking_run.py",
        "scripts/step28_positive_ranking_verify.py", "tests/test_step28_positive_ranking.py",
        "tests/test_step28_g_half.py", "tests/test_step28_record_replay.py")}
    todo = list(paths)
    while todo:
        for node in ast.walk(ast.parse(todo.pop().read_text(encoding="utf-8"))):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                if name and name.startswith("step28_"):
                    child = data.ROOT/"scripts"/(name+".py")
                    if child not in paths:
                        paths.add(child)
                        todo.append(child)
    paths.update(data.ROOT/"schema"/n for n in ("step28_g_half_policy.json", "step28_record_replay_policy.json",
        "step28_bge_continual_policy.json", "step28_alias_ranking_policy.json", "step28_chinese_base_policy.json"))
    paths.update((POLICY, BASELINE, AUTHORIZATION, CONTRACT,
                  data.ROOT/"scripts/run_step28_positive_ranking_linux_20261018.sh"))
    return [data.record(p, data.ROOT) for p in sorted(paths)]


def loss_total(row, stage, arm):
    if stage not in (1, 2) or any(row.get(k) != v for k, v in method.metadata(arm).items()):
        raise ValueError("Wrong stage or update identity")
    total = 0.
    def close(a, b):
        if not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a, b, atol=2e-6, rtol=2e-6):
            raise ValueError("Positive-rank loss decomposition differs")
    for role, weight in (("current", 1.), ("history", .1)):
        if role == "history" and stage == 1:
            if role in row: raise ValueError("Stage A cannot use history")
            continue
        r = row[role]
        if r["qualification_old_rank"] is not False or r["mse1"] is not None or len(r["views"]) != 2:
            raise ValueError("Wrong rank/teacher/views in formal update")
        if role == "current" and r["mse0"] != 0:
            raise ValueError("Current used a birth teacher")
        for v in r["views"]:
            close(v["rank_used"], v["rank_pos"])
            close(v["total"], v["bce"]+v["rank_pos"]+.5*v["hard"])
        close(r["supervised"], sum(v["total"] for v in r["views"])/2)
        close(r["distillation"], r["mse0"])
        close(r["supervised_contribution"], weight*r["supervised"])
        close(r["teacher_contribution"], .5*r["mse0"])
        contribution = weight*r["supervised"]+.5*r["mse0"]
        close(r["weighted"], contribution)
        total += contribution
    close(row["total"], total)
    return total


def validate_diagnostic(d, arm):
    if any(d.get(k) != v for k, v in method.metadata(arm).items()):
        raise ValueError("Diagnostic scoring/training identity differs")


def native_objective_pair(off, on, stage):
    arm = on["training"]["arm"]
    loss_total(on, stage, arm)
    if any(off.get(k) != v for k, v in method.metadata(arm).items()):
        raise ValueError("Native control identity differs")
    details = {}
    for role in (("current",) if stage == 1 else ("current", "history")):
        a, b = off[role], on[role]
        if a["qualification_old_rank"] is not True or b["qualification_old_rank"] is not False:
            raise ValueError("Wrong old/new rank switch")
        for key in ("mse0", "mse1", "assignment_sha256", "teacher_contribution"):
            if a[key] != b[key]: raise ValueError("Rank comparison changed teacher/allocation")
        for va, vb in zip(a["views"], b["views"], strict=True):
            for key in ("bce", "hard", "rank_pos"):
                if va[key] != vb[key]: raise ValueError("Rank comparison changed other terms")
            if not np.isclose(va["total"], va["bce"]+va["rank_used"]+.5*va["hard"], atol=2e-6, rtol=2e-6):
                raise ValueError("Old-rank control arithmetic differs")
        details[role] = dict(old_rank=[v["rank_used"] for v in a["views"]],
            new_rank=[v["rank_pos"] for v in b["views"]], old_weighted=a["weighted"],new_weighted=b["weighted"],
            actual_total_difference=b["weighted"]-a["weighted"], scalar_difference_is_not_effect_gate=True)
    return details


class Budget(single.Budget):
    def __init__(self, root):
        super().__init__(root)
        self.limits = policy()["runtime"]


def projection(measurements):
    arms = {arm: single.projection(measurements[arm]) for arm in method.ARMS}
    seconds = sum(p["projected_total_seconds"] for p in arms.values())
    peak = math.ceil(1.1*(max(m["W_F"] for m in measurements.values())+
                           sum(2*m["W_I"] for m in measurements.values())+2*2**30))
    return dict(arms=arms, projected_total_seconds=seconds, projected_peak_output_bytes=peak,
                within_budget=seconds <= 57600 and peak <= 16*2**30)


def reviewed_gate(path, kind):
    gate = data.read_json(path)
    if (kind not in ("NATIVE", "FORMAL") or gate.get("kind") != kind
            or gate.get("status") != "APPROVED_NO_OPEN_BLOCKERS" or gate.get("source_files") != sources()
            or gate.get("candidates") != policy()["candidates"]
            or gate.get("implementation_authorization") != data.record(AUTHORIZATION, data.ROOT)):
        raise ValueError("Source-bound positive-rank implementation gate required")
    # Implementation approval is not GPU/data permission. The future permission is a separate exact record.
    permission = data.read_json(data.verify(data.ROOT/gate["execution_authorization"]["path"], gate["execution_authorization"]))
    if (permission.get("status") != "APPROVED" or kind not in permission.get("stages", [])
            or permission.get("contract") != data.record(CONTRACT, data.ROOT)
            or permission.get("source_files") != sources() or not permission.get("user_quote")):
        raise ValueError("Specific execution approval missing")
    for key in ("external_review", "primary_disposition", "cpu"):
        data.verify(data.ROOT/gate[key]["path"], gate[key])
    cpu = data.read_json(data.ROOT/gate["cpu"]["path"])
    if (cpu.get("status") != "PASS_HANDWRITTEN_ONLY" or cpu.get("mode") != "cpu"
            or cpu.get("source_files") != sources() or cpu.get("cumulative_charged_seconds", math.inf) > 300
            or cpu.get("unresolved_checks") != [] or set(cpu.get("qualified_tests", [])) != set(CPU_TESTS)):
        raise ValueError("Affected CPU evidence missing")
    return gate


def validate_gate(job, path):
    gate = reviewed_gate(path, "FORMAL")
    if (gate.get("job") != job.relative_to(data.ROOT).as_posix() or gate.get("runtime") != policy()["runtime"]
            or gate.get("supervision") != policy()["supervision"]):
        raise ValueError("Dual formal scope differs")
    gpu = data.read_json(data.verify(data.ROOT/gate["gpu"]["path"], gate["gpu"]))
    if (gpu.get("status") != "PASS_HANDWRITTEN_ONLY" or gpu.get("mode") != "gpu"
            or gpu.get("source_files") != sources() or gpu.get("cumulative_charged_seconds", math.inf) > 1200
            or set(gpu.get("arms", {})) != set(method.ARMS)):
        raise ValueError("Both native candidates must qualify")
    for arm, native in gpu["arms"].items():
        if (native.get("actual_updates") != 6 or native.get("charged_seconds", math.inf) > 600
                or native.get("maximum_records") != 224 or native.get("channel_tokens") != 256
                or native.get("both_phases_encoder_head_effect") is not True
                or native.get("next_update_bitwise_equal") is not True
                or native.get("pooling_restores_verified") is not True
                or native.get("rank_precision_verified") is not True
                or native.get("metadata") != method.metadata(arm)):
            raise ValueError("Native effects/restoration missing: "+arm)
    if not projection({arm: v["measurements"] for arm, v in gpu["arms"].items()})["within_budget"]:
        raise ValueError("Measured dual run exceeds budget")
    return gate


# These functions already receive an explicit entry: no global mutation or default-arm substitution.
for _name in ("train_stage", "memory_from", "restore_point", "checkpoint", "train", "open_valid"):
    globals()[_name] = partial(getattr(single, _name), entry=sys.modules[__name__])


def blind_gate(root: Path, manifest: dict, *, entry=None) -> tuple[dict, dict]:
    entry = entry or sys.modules[__name__]
    if (manifest["status"] != entry.COMPLETE or manifest["source_files"] != entry.sources()
            or manifest["physical_updates"] != 576*len(entry.method.ARMS) or manifest["gradient_group_presentations"] != 864*len(entry.method.ARMS)
            or any(set(manifest[k]) != set(entry.POINTS) for k in ("points", "training", "diagnostics"))
            or any(set(manifest[k]) != set(entry.method.ARMS) for k in ("starts", "initials"))):
        raise ValueError("Incomplete source-bound blind experiment")
    baseline, old_root, old_col, old_manifest = entry.baseline_inputs()
    if manifest.get("baseline_identity") != data.record(entry.BASELINE, data.ROOT):
        raise ValueError("Baseline provenance missing")
    split = data.read_json(data.verify(root/manifest["partition"]["path"], manifest["partition"]))
    if split != data.read_json(old_root/"run/partition.json"):
        raise ValueError("Frozen baseline partition differs")
    valid_ids = [r["group_uid"] for r in split["development"]]
    if len(valid_ids) != 40 or any(sum(r["domain"] == d for r in split["development"]) != 20 for d in "AB"):
        raise ValueError("AB valid coverage differs")
    fit = {r["group_uid"]: r["domain"] for r in split["fit"]}
    canonical = sorted(u for u, domain in fit.items() if domain == "A")
    if len(canonical) != 48:
        raise ValueError("Shared A fit scope differs")
    a_point = manifest["points"]["A_"+entry.method.ARMS[0]]
    shared = entry.memory_from(data.verify(root/a_point["memory"]["path"], a_point["memory"]).read_bytes(), entry.method.ARMS[0])
    members = shared.summary()["members"]
    roots = getattr(entry, "roots", dict(memory=20260930, schedule=20260918))
    rng = random.Random(data.seed_for(roots["memory"], "ABC", "retention"))
    retained = []
    for seen, uid in enumerate(sorted(canonical), 1):
        k = seen-1 if seen <= 6 else rng.randrange(seen)
        if k < 6:
            if k == len(retained):
                retained.append(uid)
            else:
                retained[k] = uid
    if retained != members:
        raise ValueError("Independent Algorithm R differs")
    for arm in entry.method.ARMS:
        initial_ok = (entry.validate_saved_initial(manifest["initials"][arm], baseline)
                      if hasattr(entry, "validate_saved_initial") else manifest["initials"][arm] == baseline["initial_identity"])
        if (not initial_ok
                or manifest["starts"][arm] != manifest["points"]["A_"+arm]["continuity"]):
            raise ValueError("Independent starts/own A continuation missing")
    scores, sequences = {}, {1: [], 2: []}
    for name in entry.POINTS:
        point = manifest["points"][name]
        arm, stage = name[2:], "AB".index(name[0])+1
        a_point = manifest["points"]["A_"+arm]
        if (point["arm"] != arm or point["stage"] != stage or point["adam_step"] != stage*288
                or point["full_restore_verified"] is not True or point["inference_restore_verified"] is not True
                or any(point.get(k) != v for k, v in entry.method.metadata(arm).items())):
            raise ValueError("Endpoint restoration missing")
        logrec = manifest["training"][name]
        log = data.read_json(data.verify(root/logrec["path"], logrec))
        expected_fit = [r["group_uid"] for r in split["fit"] if r["domain"] == "AB"[stage-1]]
        stream = data.seed_for(roots["schedule"], "ABC", stage, "current")
        expected = data.schedule([type("ID", (), {"uid": u})() for u in expected_fit], 6, stream)
        if log["updates"] != 288 or len(log["rows"]) != 288 or log["adam_step"] != stage*288 or log["current_ids"] != [g.uid for g in expected]:
            raise ValueError("Main schedule/update count differs")
        memory = entry.memory_from(data.verify(root/point["memory"]["path"], point["memory"]).read_bytes(), arm)
        if (memory.summary() != point["memory_summary"] or memory.reservoir.seen != 48 or memory.summary()["members"] != members
                or memory.maps["pooling"] != entry.method.identity(arm)
                or point["continuity"]["memory_sha256"] != memory.summary()["sha256"]):
            raise ValueError("Terminal memory unexpectedly retained B")
        if stage == 2:
            prior = entry.memory_from(data.verify(root/a_point["memory"]["path"], a_point["memory"]).read_bytes(), arm)
            prior.begin_stage(2)
            if prior.summary() != log["memory_before"]:
                raise ValueError("Wrong own A Memory")
            ids = [prior.draw()[0].uid for _ in range(288)]
            if ids != log["history_ids"] or prior.summary() != log["memory_after"]:
                raise ValueError("History draw state differs")
            if any(not np.array_equal(prior.tables[u], memory.tables[u]) for u in members):
                raise ValueError("Birth target refreshed")
        elif log["history_ids"] or log["memory_before"] is not None:
            raise ValueError("Shared A used history")
        assignments = []
        for step, row in enumerate(log["rows"], 1):
            total = entry.loss_total(row, stage, arm)
            hashes = [row["current"]["assignment_sha256"]]
            if stage == 2:
                h = row["history"]
                group = next(g for g in memory.reservoir.groups if g.uid == row["history_uid"])
                key = data.seed_for(roots["schedule"], "ABC", stage, step, "history", group.uid, "regroup")
                if h["assignment_sha256"] != hashlib.sha256(record.assignment(group, key).tobytes()).hexdigest():
                    raise ValueError("Historical A1 assignment differs")
                hashes.append(h["assignment_sha256"])
            assignments.append(hashes)
            rates = entry.spec.rates(step) if hasattr(entry, "spec") else (parent.stage_lr(step), .001)
            if (row["step"] != step or (row["encoder_lr"], row["head_lr"]) != rates
                    or row["pooling"] != entry.method.identity(arm)
                    or not np.isfinite(row["total"]) or not np.isclose(row["total"], total, rtol=2e-6, atol=2e-6)):
                raise ValueError("Actual loss/LR decomposition differs")
        sequences[stage].append((log["current_ids"], log["history_ids"], assignments))
        mapping = data.read_json(data.verify(root/point["mapping"]["path"], point["mapping"]))
        if mapping["status"] != "PASS_CALIBRATION_FIT" or point["maps"]["first"] != a_point["maps"]["first"]:
            raise ValueError("Calibration failed or first map changed")
        if point["calibration_ids"] != [r["group_uid"] for r in split["calibration"] if r["domain"] == "AB"[stage-1]]:
            raise ValueError("Calibration scope differs")
        data.verify(root/point["model"]["path"], point["model"])
        calibration = previous.array(root, point["scores"]["calibration"], (12, 378), np.float32)
        roles = {r: previous.array(root, point["scores"][r], (40, 378), np.float32 if r == "raw" else np.float64) for r in parent.ROLES}
        for prefix in ("full", "inference"):
            for role, expected_values in (("calibration", calibration), ("raw", roles["raw"])):
                restored = previous.array(root, point["scores"][prefix+"-"+role], expected_values.shape, np.float32)
                if not np.array_equal(restored, expected_values):
                    raise ValueError("Saved restored scores differ")
        for role, key in (("stage-cal", "stage"), ("first-cal", "first")):
            if not np.array_equal(roles[role], parent.calibration.transform(roles["raw"], point["maps"][key])):
                raise ValueError("Saved affine scores differ")
        captures = manifest["diagnostics"][name]
        if len(captures) != 6:
            raise ValueError("Missing complete epoch diagnostics")
        for epoch, rec in enumerate(captures, 1):
            droot = root/"diagnostics"
            d = data.read_json(data.verify(droot/rec["path"], rec))
            entry.validate_diagnostic(d, arm)
            cache_ids = [] if stage == 1 else members
            if (d["epoch"] != epoch or d["adam_step"] != (stage-1)*288+epoch*48 or d["stage_step"] != epoch*48
                    or d["stage"] != stage or d["arm"] != arm or d["order"] != "ABC" or d["valid_labels_used"] is not False
                    or d["fit_ids"] != expected_fit or d["cache_ids"] != cache_ids or d["valid_ids"] != valid_ids
                    or d["fit_domains"] != ["AB"[stage-1]]*48 or d["cache_domains"] != ["A"]*len(cache_ids)):
                raise ValueError("Diagnostic scope/epoch/identity differs")
            previous.array(droot, d["fit"], (48, 3), np.float64)
            previous.array(droot, d["cache"], (len(cache_ids), 3), np.float64)
            v = previous.array(droot, d["valid_blind"], (40, 378), np.float32)
            if epoch == 6 and not np.array_equal(v, roles["raw"]):
                raise ValueError("Final diagnostic and endpoint raw differ")
            if [e["uid"] for e in d["teacher_errors"]] != cache_ids:
                raise ValueError("History diagnostics missing")
            for e in d["teacher_errors"]:
                keys = ("D0", "D1", "residual_cross_mean", "residual_difference_mse")
                if not all(np.isfinite(e[k]) for k in keys):
                    raise ValueError("Nonfinite history diagnostic")
        scores[name] = roles
    for stage, sequences_here in (sequences.items() if roots["schedule"] == 20260918 and roots["memory"] == 20260930 else ()):
        for other, (old_root, _, _) in entry.baseline_runs().items():
            name = "AB"[stage-1]+"_"+other
            old_log = data.read_json(old_root/"run/updates"/(name+".json"))
            if sequences_here != [trajectory(old_log)]*len(entry.method.ARMS):
                raise ValueError("H actual schedule/A1/history differs from frozen "+other)
    return scores, split


def collect(root, scores, labelled, split, *, entry=None):
    entry = entry or sys.modules[__name__]
    root.mkdir()
    if [g.uid for g in labelled] != [r["group_uid"] for r in split["development"]]:
        raise ValueError("Opened valid alignment differs")
    truth = np.asarray([g.labels for g in labelled], np.uint8)
    result = {"source_files": entry.sources(), "group_ids": [g.uid for g in labelled],
        "domains": [r["domain"] for r in split["development"]], "columns": list(parent.metrics.COLUMNS), "points": {}}
    for name, roles in scores.items():
        result["points"][name] = {}
        for role, values in roles.items():
            matrix, counts = parent.metrics.group_metrics(truth, values)
            queries = query_metrics(truth, values)
            if queries.shape != (40, 28, 10) or not np.allclose(queries.mean(1), matrix[:, 12:], atol=1e-12, rtol=1e-12):
                raise ValueError("Query detail and retained ranking metrics disagree")
            if matrix.shape != (40, 22) or any(r["tp"]+r["fn"] != 20 or r["tn"]+r["fp"] != 358 for r in counts):
                raise ValueError("Full metric/confusion coverage differs")
            path = root/(name+"_"+role+"_counts.json")
            data.write_json(path, counts)
            nullable = matrix.astype(object)
            nullable[undefined_metrics(counts)] = None
            matrix_path = root/(name+"_"+role+"_metrics.json")
            data.write_json(matrix_path, {"columns": result["columns"], "values": nullable.tolist(),
                "na_reason": "Zero precision or MCC denominator; null is undefined, never zero."})
            result["points"][name][role] = {"matrix": data.record(matrix_path, root),
                "queries": previous.save_array(root/(name+"_"+role+"_queries.npy"), queries, root),
                "counts": data.record(path, root)}
    result["status"] = "ALL_12_POS_METRIC_COUNT_SETS_SAVED"
    left, right = np.triu_indices(28, 1)
    degrees = np.zeros((40, 28), np.uint8)
    for k, (a, b) in enumerate(zip(left, right)):
        degrees[:, a] += truth[:, k]
        degrees[:, b] += truth[:, k]
    if not np.all((degrees == 1).sum(1) == 16) or not np.all((degrees == 2).sum(1) == 12):
        raise ValueError("Query positive-count geometry differs")
    result["query_positive_counts"] = previous.save_array(root/"query_positive_counts.npy", degrees, root)
    data.write_json(root/"collected.json", result)
    return result


def finalize(root, *, entry=None):
    entry = entry or sys.modules[__name__]
    col = data.read_json(root/"collected.json")
    if col["status"] != "ALL_12_POS_METRIC_COUNT_SETS_SAVED" or set(col["points"]) != set(entry.POINTS) or col["source_files"] != entry.sources():
        raise ValueError("Complete saved collection required")
    old_runs = entry.baseline_runs()
    all_points = dict(col["points"])
    origins = {name: root for name in entry.POINTS}
    for arm, (old_root, old_col, _) in old_runs.items():
        if any(col[k] != old_col[k] for k in ("group_ids", "domains", "columns")):
            raise ValueError("New and frozen group correspondence differs")
        for stage in "AB":
            name = stage+"_"+arm
            all_points[name] = old_col["points"][name]
            origins[name] = old_root/"evaluation"
    rows = [np.flatnonzero(np.asarray(col["domains"]) == d) for d in "AB"]
    if any(len(r) != 20 for r in rows):
        raise ValueError("AB bootstrap scope differs")
    draws = np.random.Generator(np.random.PCG64(20260930)).integers(20, size=(5000, 2, 20))
    arrays, result = {}, {"absolute": {}, "comparisons": {}, "old_changes": {}, "columns": col["columns"]}
    for name, roles in all_points.items():
        origin = origins[name]
        arrays[name], result["absolute"][name] = {}, {}
        if set(roles) != set(parent.ROLES):
            raise ValueError("All calibration roles required")
        for role, rec in roles.items():
            saved = data.read_json(data.verify(origin/rec["matrix"]["path"], rec["matrix"]))
            v = np.asarray(saved["values"], np.float64)
            if saved["columns"] != col["columns"] or v.shape != (40, 22) or np.isinf(v).any():
                raise ValueError("Nullable metric matrix differs")
            q = previous.array(origin, rec["queries"], (40, 28, 10), np.float64)
            if not np.allclose(q.mean(1), v[:, 12:], atol=1e-12, rtol=1e-12):
                raise ValueError("Saved query and group rankings differ")
            counts = data.read_json(data.verify(origin/rec["counts"]["path"], rec["counts"]))
            if len(counts) != 40 or any(r["tp"]+r["fn"] != 20 or r["fp"]+r["tn"] != 358 for r in counts):
                raise ValueError("Classification group coverage differs")
            if not np.array_equal(np.isnan(v), undefined_metrics(counts)):
                raise ValueError("Undefined metric mask differs from confusion denominators")
            ct = np.asarray([[r[k] for k in ("tp", "fp", "fn", "tn")] for r in counts], np.int64)
            arrays[name][role] = v
            result["absolute"][name][role] = {"by_domain": summarize(v, rows, draws),
                "macro_all": {k: None if np.isnan(v[:, j]).any() else float(v[:, j].mean()) for j, k in enumerate(col["columns"])},
                "macro_undefined_groups": dict(zip(col["columns"], np.isnan(v).sum(0).tolist())),
                "fixed_half_classification": {"all": pooled_rates(ct), **{d: pooled_rates(ct[r]) for d, r in zip("AB", rows)}}}
    signs = np.array([-1. if key in ("brier", "log_loss") else 1. for key in col["columns"]])
    result["initial_comparisons"], result["forgetting_comparisons"] = {}, {}
    for candidate, other in entry.policy()["evaluation"]["comparisons"]:
        key = candidate+"-"+other
        result["comparisons"][key] = {role: summarize(arrays["B_"+candidate][role]-arrays["B_"+other][role], rows, draws) for role in parent.ROLES}
        result["initial_comparisons"][key] = {role: summarize(arrays["A_"+candidate][role]-arrays["A_"+other][role], rows, draws) for role in parent.ROLES}
        result["forgetting_comparisons"][key] = {role: summarize(
            ((arrays["A_"+candidate][role]-arrays["B_"+candidate][role])-(arrays["A_"+other][role]-arrays["B_"+other][role]))*signs, rows, draws)["A"] for role in parent.ROLES}
    for arm in (*entry.method.ARMS, *old_runs):
        result["old_changes"][arm] = {role: summarize(arrays["B_"+arm][role]-arrays["A_"+arm][role], rows, draws)["A"] for role in parent.ROLES}
    result["success"] = {"criterion": entry.policy()["evaluation"]["success"],
        "passed": all(result["comparisons"][entry.method.ARMS[0]+"-S"]["raw"][d]["map"]["conditional_95pct_interval"][0] > 0 for d in "AB"),
        "secondary_H_pos_minus_S_both_positive": all(result["comparisons"]["H_pos-S"]["raw"][d]["map"]["conditional_95pct_interval"][0] > 0 for d in "AB")}
    degrees = previous.array(root, col["query_positive_counts"], (40, 28), np.uint8)
    subgroup = {}
    for name in all_points:
        q = previous.array(origins[name], all_points[name]["raw"]["queries"], (40, 28, 10), np.float64)
        subgroup[name] = {m: (q[:, :, 0]*(degrees == m)).sum(1)/(degrees == m).sum(1) for m in (1, 2)}
    result["query_subgroup_MAP_comparisons"] = {}
    for candidate, other in entry.policy()["evaluation"]["comparisons"]:
        by_count = {}
        for m in (1, 2):
            delta = subgroup["B_"+candidate][m]-subgroup["B_"+other][m]
            by_count[str(m)] = {d: dict(mean=float(delta[r].mean()),
                conditional_95pct_interval=np.quantile(delta[r][draws[:, index, :]].mean(1), [.025, .975]).tolist())
                for index, (d, r) in enumerate(zip("AB", rows))}
        result["query_subgroup_MAP_comparisons"][candidate+"-"+other] = by_count
    result["query_subgroup_limits"] = "Descriptive raw B-end MAP differences only; same whole-group draws; no success gate or causal attribution. Counts derived once from the authorized joint valid opening."
    result.update(status="FIXED_AB_STATISTICS_COMPLETE", source_files=entry.sources(), primary_role="raw", primary_metric="map",
        forgetting_sign="High-good: A-B; brier/log_loss: B-A; arm difference candidate-other; old_changes always raw B-A",
        bootstrap="One PCG64(20260930) (5000,2,20) whole-group array shared by all contrasts and roles",
        undefined_policy="Null denotes N/A. Any undefined required group operand makes the fixed-group mean/interval N/A; no groups dropped, no per-arm nanmean. Pooled rates use summed counts first.",
        limits="Reused development; single seed AB; independent training trajectories; conditional system contrasts, not isolated retention or inference causality; primary S_pos-S dual-domain MAP gate; H_pos comparisons secondary and cannot replace a failed primary; no universal superiority.")
    data.write_json(root/"evaluation.json", result)
    return result


def execute(job, gate_path, *, entry=None):
    entry = entry or sys.modules[__name__]
    import torch
    from step28_record_replay_verify import preflight
    started = float(os.environ.get("RECORD_REPLAY_STARTED_EPOCH", "nan"))
    if not math.isfinite(started) or not 0 <= time.time()-started < entry.policy()["runtime"]["maximum_gpu_stage_seconds"]:
        raise ValueError("Bounded wrapper start required before any formal read")
    entry.validate_gate(job, gate_path)
    if job.exists() or not job.is_relative_to(data.ROOT/"reports"):
        raise ValueError("New independent output required")
    resources = preflight()
    if shutil.disk_usage(data.ROOT).free < 16*2**30:
        raise ValueError("Insufficient output disk")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    numerics = parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory, 0)
    job.mkdir(parents=True)
    budget = entry.Budget(job)
    data.write_json(job/"access.json", dict(train=0, valid=0, heldout=0, owners=0))
    data.write_json(job/"execution.json", {"gate": data.record(gate_path, data.ROOT), "sources": entry.sources(), "resources": resources, "numerics": numerics})
    stopped = threading.Event()
    def watchdog():
        while not stopped.wait(1):
            try:
                budget.check()
            except BaseException as exc:
                data.write_json(job/"failure.json", {"status": "BUDGET_STOP_NO_RETRY", "error": str(exc), "budget": budget.snapshot()})
                os._exit(2)
    def terminate(signum, frame):
        raise RuntimeError("TERM: retain scene, no automatic retry")
    signal.signal(signal.SIGTERM, terminate)
    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()
    try:
        budget.check()
        manifest, split, valid, metadata = entry.train(job, budget)
        budget.check()
        scores, split, labelled = entry.open_valid(job, manifest, valid, metadata, entry.config())
        budget.phase = "evaluation"
        entry.collect(job/"evaluation", scores, labelled, split)
        diag.evaluate(job/"run/diagnostics", manifest, labelled, split, job/"diagnostics_evaluation", scope="AB")
        entry.finalize(job/"evaluation")
        budget.check()
        result = {"status": entry.COMPLETION_STATUS, "source_files": entry.sources(), "physical_updates": 1152,
            "gradient_group_presentations": 1728, "access": data.read_json(job/"access.json"), "budget": budget.snapshot()}
        data.write_json(job/"completion.json", result)
        return result
    except BaseException as exc:
        data.write_json(job/"failure.json", {"status": "FAIL_NO_RETRY", "traceback": traceback.format_exc(),
            "budget": budget.snapshot()})
        raise
    finally:
        stopped.set()
        watcher.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "posix": parser.error("Scientific execution is Linux-only")
    execute(args.job.resolve(), args.gate.resolve())
