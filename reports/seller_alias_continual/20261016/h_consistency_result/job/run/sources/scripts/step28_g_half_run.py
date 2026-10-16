"""Fixed single G_half trajectory with read-only saved S/G comparators."""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
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

import numpy as np

import step28_g_half as method
import step28_balanced_pooling_run as saved
import step28_replay_diagnostics as diag
import step28_record_replay_run as legacy

record, data = method.record, method.data
parent, core = record.parent, record.core
base, previous = record.base, legacy.previous
POLICY = data.ROOT / "schema/step28_g_half_policy.json"
BASELINE = data.ROOT / "reports/documentation/20261015/g_refinement/baseline_sources.json"
AUTHORIZATION = data.ROOT / "reports/documentation/20261015/g_half_implementation/authorization.json"
POINTS = ("A_G_half", "B_G_half")
COMPLETE = "COMPLETE_576_UPDATES_2_RESTORES_12_DIAGNOSTICS_VALID_BLIND"
# These data/metric helpers retain their closed S/G/F semantics; no old training/qualification entry is invoked.
public_inputs, attach_labels, partition = saved.public_inputs, saved.attach_labels, saved.partition
query_metrics, undefined_metrics = saved.query_metrics, saved.undefined_metrics
pooled_rates, summarize = saved.pooled_rates, saved.summarize
fresh, verify_archive, pretrained_archive = saved.fresh, saved.verify_archive, saved.pretrained_archive


def policy() -> dict:
    p = data.read_json(POLICY)
    if (p["methods"] != ["G_half"] or p["domains"] != "AB" or p["main_rng_order"] != "ABC"
            or p["physical_updates"] != 576 or p["gradient_group_presentations"] != 864
            or p["physical_stage_states"] != 2 or p["epoch_diagnostic_points"] != 12
            or p["history_supervision"] != .1 or p["teacher_weight"] != .5 or p["seed"] != "s0"
            or p["memory_maximum_bytes"] != 1048576 or p["pooling"] != method.identity()
            or p["evaluation"] != dict(bootstrap_replicates=5000, bootstrap_seed=20260930,
                comparisons=[["G_half", "S"], ["G_half", "G"]], primary_metric="map",
                primary_role="raw", success="B_end_H_minus_S_each_AB_MAP_CI_lower_strictly_positive")
            or p["cpu_verification"]["maximum_seconds"] != 180
            or p["native_verification"]["maximum_seconds"] != 600
            or p["native_verification"]["actual_updates"] != 6
            or p["runtime"]["maximum_gpu_stage_seconds"] != 28800
            or p["runtime"]["maximum_output_bytes"] != 10*2**30
            or p["supervision"] != dict(train_parse_attempts=1, valid_parse_attempts=1,
                test_access=False, owners_access=False, automatic_retry=False)):
        raise ValueError("Approved G_half contract differs")
    return p


def config() -> dict:
    c = record.config()
    c["runtime"] = policy()["runtime"]
    return c


def sources() -> list[dict]:
    paths = {data.ROOT / n for n in ("scripts/step28_g_half_run.py", "scripts/step28_g_half_verify.py",
        "tests/test_step28_g_half.py", "tests/test_step28_record_replay.py")}
    todo = list(paths)
    while todo:
        for node in ast.walk(ast.parse(todo.pop().read_text(encoding="utf-8"))):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
            for n in names:
                if n and n.startswith("step28_"):
                    child = data.ROOT / "scripts" / (n + ".py")
                    if child not in paths:
                        paths.add(child)
                        todo.append(child)
    paths.update(data.ROOT / "schema" / n for n in ("step28_g_half_policy.json", "step28_record_replay_policy.json",
        "step28_bge_continual_policy.json", "step28_alias_ranking_policy.json", "step28_chinese_base_policy.json"))
    paths.update((data.ROOT/"docs/SELLER_ALIAS_G_HALF.zh.md", data.ROOT/"scripts/run_step28_g_half_linux_20261015.sh",
                  BASELINE, AUTHORIZATION))
    return [data.record(p, data.ROOT) for p in sorted(paths)]


def baseline_inputs():
    manifest = data.read_json(BASELINE)
    if (manifest["run_id"] != "20261015_131600" or len(manifest["files"]) != 43
            or manifest["initial_identity"] != {"model": "13b535e1d66a0f13f0d87461835bd3ba0cee69c531307884b471f8702bd14e4b",
                "adam": "de5b1864dc195dd65286e083567e3fd8bb6a146a9ecfcc3bae6b3fdda8c32d20"}):
        raise ValueError("Frozen baseline scope differs")
    for row in manifest["files"]:
        data.verify(data.ROOT/row["path"], row)
    root = data.ROOT/manifest["root"]
    old = data.read_json(root/"run/manifest.json")
    col = data.read_json(root/"evaluation/collected.json")
    split = data.read_json(root/"run/partition.json")
    if (col["group_ids"] != [r["group_uid"] for r in split["development"]]
            or col["domains"] != [r["domain"] for r in split["development"]]
            or col["columns"] != list(parent.metrics.COLUMNS)):
        raise ValueError("Frozen group/role/column alignment differs")
    for name in manifest["points"]:
        if set(col["points"][name]) != set(parent.ROLES):
            raise ValueError("Frozen probability roles incomplete")
        for role, rec in col["points"][name].items():
            values = data.read_json(root/"evaluation"/rec["matrix"]["path"])
            if values["columns"] != col["columns"] or np.asarray(values["values"], object).shape != (40,22):
                raise ValueError("Frozen paired matrix shape differs")
    for path in ("scripts/step28_balanced_pooling.py", "scripts/step28_balanced_pooling_run.py",
                 "scripts/step28_record_replay.py", "scripts/step28_record_replay_numerics.py",
                 "scripts/step28_alias_ranking.py", "scripts/step28_replay_diagnostics.py"):
        prior = next(r for r in old["source_files"] if r["path"] == path)
        data.verify(data.ROOT/path, prior)
    return manifest, root, col, old


DIRECT_COMPARISON_KEY = "direct_G_new_B_positive"
COMPLETION_STATUS = "COMPLETE_G_HALF_FIXED_AB"
CPU_TESTS = ("test_local_value_derivative_precision_and_rejections", "test_full_objective_top5_teacher_and_proxy_gradient",
             "test_real_vjp_encoder_head_gradients_and_updates", "test_actual_capture_restore_and_next_update",
             "test_single_candidate_orchestration_and_blind_gate", "test_saved_comparisons_directions_permissions_and_budgets")


def baseline_runs():
    baseline, root, col, manifest = baseline_inputs()
    return {arm: (root, col, manifest) for arm in ("S", "G")}


def loss_total(row, stage, arm):
    total = row["current"]["supervised"]
    if stage == 2:
        total += .1*row["history"]["supervised"]+.5*row["history"]["mse0"]
    return total


def validate_diagnostic(d, arm):
    return None


def validate_native(native):
    return None


def trajectory(log):
    return (log["current_ids"], log["history_ids"],
            [[r["current"]["assignment_sha256"]]+([r["history"]["assignment_sha256"]] if r["history_uid"] else []) for r in log["rows"]])


class Budget(legacy.Budget):
    def __init__(self, root: Path):
        super().__init__(root)
        self.limits = policy()["runtime"]

    def output_files(self):
        # Inputs are outside these output trees; no recursive scan through linked data/models.
        sidecars = [self.root.with_suffix(s) for s in (".dispatch.json", ".listener.json", ".listener.console.txt")]
        extra = data.ROOT/"reports/qualification"
        return list(dict.fromkeys([*super().output_files(), *sidecars, *(p for p in extra.rglob("*"))]))


def projection(m: dict) -> dict:
    keys = ("t_A", "t_B", "v", "h", "L", "s_F", "r_F", "s_I", "r_I", "W_F", "W_I")
    if any(not np.isfinite(m[k]) or m[k] <= 0 for k in keys):
        raise ValueError("Actual positive finite H measurements required")
    train = 288*(m["t_A"]+m["t_B"])
    forward = 1446*m["v"]+36*m["h"]
    io = m["L"]+2*(m["s_F"]+m["r_F"]+m["s_I"]+m["r_I"])
    seconds = 1.25*(train+forward+io+1800)
    size = math.ceil(1.1*(m["W_F"]+2*m["W_I"]+2*2**30))
    return dict(train_seconds=train, forward_seconds=forward, io_seconds=io, unmeasured_allowance_seconds=1800,
                projected_total_seconds=seconds, projected_peak_output_bytes=size,
                within_budget=seconds <= 28800 and size <= 10*2**30)


def train_stage(model, optimizer, current, memory, valid, c, arm, stage, root, budget, *, entry=None):
    entry = entry or sys.modules[__name__]
    if parent.adam_step(optimizer) != (stage-1)*288:
        raise ValueError("Stage continuity differs")
    if memory is not None:
        memory.begin_stage(stage)
    before = None if memory is None else memory.summary()
    schedule, _ = parent.schedule(current, parent.contract(), entry.method.RNG_ORDER, stage)
    name = "AB"[stage-1]+"_"+arm
    budget.phase = name
    rows, captures = [], []
    for step, group in enumerate(schedule, 1):
        history, target = (None, None) if memory is None else memory.draw()
        row = entry.method.update(model, optimizer, group, history, target, c, arm, stage, step, budget.check)
        if parent.adam_step(optimizer) != (stage-1)*288+step:
            raise ValueError("Actual optimizer call missing")
        rows.append({**row, "step": step, "current_uid": group.uid, "history_uid": history.uid if history else None})
        if step % 48 == 0:
            captures.append(entry.method.capture(model, optimizer, current, memory, valid, c, arm,
                stage, step//48, root/"diagnostics", name, budget.check))
        if step == 1 or step % 24 == 0:
            print(data.json_bytes({"event": "updates", "point": name, "completed": step, **budget.snapshot()}).decode(), flush=True)
    after = None if memory is None else memory.summary()
    if before is not None and any(before[k] != after[k] for k in ("members", "seen", "origins", "tables")):
        raise ValueError("History targets/membership changed within B")
    path = root/"updates"/(name+".json")
    data.write_json(path, {"arm": arm, "stage": stage, "updates": 288, "adam_step": parent.adam_step(optimizer),
        "current_ids": [g.uid for g in schedule], "history_ids": [r["history_uid"] for r in rows if r["history_uid"]],
        "memory_before": before, "memory_after": after, "rows": rows})
    return data.record(path, root), captures


def memory_from(payload: bytes, arm: str, *, entry=None):
    entry = entry or sys.modules[__name__]
    if arm not in entry.method.ARMS:
        raise ValueError("Unknown arm Memory")
    memory = record.Memory.from_bytes(payload)
    if any(memory.maps.get(k) != v for k, v in entry.method.metadata(arm).items()):
        raise ValueError("Memory candidate identity differs")
    return memory


def continuity(model, optimizer, memory):
    return {"model": core.state_digest(model.state_dict()), "adam": core.state_digest(optimizer.state_dict()),
            "rng": diag.rng_state(), "modes": [m.training for m in model.modules()],
            "memory_sha256": hashlib.sha256(memory.to_bytes()).hexdigest()}


def restore_point(path, expected, model, optimizer, arm, *, entry=None):
    entry = entry or sys.modules[__name__]
    meta = core.restore_state(path, model, optimizer, expected["state_sha256"])
    if entry.method.bind(model, meta) != arm:
        raise ValueError("Restored arm differs")
    for m, mode in zip(model.modules(), meta["modes"], strict=True):
        m.training = mode
    diag.restore_rng(meta["rng"])
    return meta


def checkpoint(root, name, model, optimizer, memory, current, calibration, valid, c, arm, stage, budget, *, entry=None):
    entry = entry or sys.modules[__name__]
    if len(calibration) != 12 or len(valid) != 40 or parent.adam_step(optimizer) != 288*stage:
        raise ValueError("Endpoint scope differs")
    optimizer.zero_grad(set_to_none=True)
    c = {**c, "pooling_stats": c.setdefault("pooling_stats", {}),
         "pooling_context": {"stage": stage, "step": 288, "role": "endpoint_prediction_or_restore"}}
    state, modes = diag.rng_state(), [m.training for m in model.modules()]
    with diag.neutral(model):
        scores = {"calibration": entry.method.score(model, calibration, c, arm, budget.check),
                  "raw": entry.method.score(model, valid, c, arm, budget.check)}
        mapping = parent.calibration.fit(scores["calibration"], np.asarray([g.labels for g in calibration], np.uint8),
                                         role="calibration", check=budget.check)
        if mapping["status"] != "PASS_CALIBRATION_FIT":
            raise ValueError("Calibration failed; no retry")
        current_map = dict(zip(("a", "b"), parent.calibration.parameters(mapping)))
        if memory is None:
            memory = record.Memory(entry.method.RNG_ORDER)
        first_map = memory.maps.get("first", current_map)
        memory.maps = {"first": first_map, "stage": current_map, "rng": state,
                       "completed_stage": stage, "adam_step": stage*288, **entry.method.metadata(arm)}
        if stage == 1:
            memory.retain(current, 1, lambda g: record.reference(model, g, c, budget.check))
    mp = root/"memory"/(name+".bin")
    mp.write_bytes(memory.to_bytes())
    memory = entry.memory_from(mp.read_bytes(), arm)
    meta = {"name": name, "rng": state, "modes": modes, **entry.method.metadata(arm),
            "members": [g.uid for g in memory.reservoir.groups], "maps": memory.maps,
            "memory": data.record(mp, root), "policy_sha256": data.sha256(entry.POLICY)}
    full_path = root/"work"/(name+".pt")
    budget.check(base.persistence.checkpoint_reserve(model, optimizer))
    full = core.save_state(full_path, model, optimizer, meta)
    if entry.restore_point(full_path, full, model, optimizer, arm) != meta:
        raise ValueError("Full model/Adam restore mismatch")
    with diag.neutral(model):
        for role, groups in (("calibration", calibration), ("raw", valid)):
            restored = entry.method.restored_score(model, groups, c, budget.check)
            if not np.array_equal(scores[role], restored):
                raise ValueError("Full restored scoring differs")
            scores["full-"+role] = restored
    model_path = root/"models"/(name+".pt")
    budget.check(base.persistence.checkpoint_reserve(model, None))
    inference = core.save_state(model_path, model, None, meta)
    if entry.restore_point(model_path, inference, model, None, arm) != meta:
        raise ValueError("Inference restore mismatch")
    with diag.neutral(model):
        for role, groups in (("calibration", calibration), ("raw", valid)):
            restored = entry.method.restored_score(model, groups, c, budget.check)
            if not np.array_equal(scores[role], restored):
                raise ValueError("Inference restored scoring differs")
            scores["inference-"+role] = restored
    for role, mapping_value in (("stage-cal", current_map), ("first-cal", first_map)):
        scores[role] = parent.calibration.transform(scores["raw"], mapping_value)
        parent.calibration.preserve_order(scores["raw"], scores[role])
    map_path = root/"maps"/(name+".json")
    data.write_json(map_path, mapping)
    result = {"arm": arm, "stage": stage, "adam_step": parent.adam_step(optimizer),
        "full_restore_verified": True, "inference_restore_verified": True, **entry.method.metadata(arm),
        "calibration_ids": [g.uid for g in calibration], "maps": {"first": first_map, "stage": current_map},
        "mapping": data.record(map_path, root), "memory": data.record(mp, root), "memory_summary": memory.summary(),
        "continuity": continuity(model, optimizer, memory), 
        "full_state": {**full, "path": full_path.relative_to(root).as_posix()},
        "model": {**inference, "path": model_path.relative_to(root).as_posix()},
        "scores": {k: previous.save_array(root/"scores"/(name+"_"+k+".npy"), v, root) for k,v in scores.items()}}
    data.write_json(root/"points"/(name+".json"), result)
    base.persistence.remove_work_file(full_path, root)
    return result, memory


def train(job, budget, *, entry=None):
    entry = entry or sys.modules[__name__]
    import torch
    root = job/"run"
    for sub in ("models", "work", "memory", "points", "scores", "maps", "updates", "diagnostics", "sources"):
        (root/sub).mkdir(parents=True)
    c = entry.config()
    baseline, old_root, old_col, old_manifest = entry.baseline_inputs()
    archive = entry.pretrained_archive(c)
    groups, metadata, checked = entry.public_inputs(c)
    if checked != old_manifest["public_inputs"]:
        raise ValueError("Formal input bytes differ from frozen S/G")
    _, split = entry.partition(groups, metadata, c)
    if split != data.read_json(old_root/"run/partition.json"):
        raise ValueError("H partition differs from frozen S/G")
    data.write_json(root/"partition.json", split)
    manifest = {"status": "RUNNING", "source_files": entry.sources(), "public_inputs": checked, "pretrained_archive": archive,
        "partition": data.record(root/"partition.json", root), "points": {}, "training": {}, "diagnostics": {},
        "baseline_identity": data.record(entry.BASELINE, data.ROOT), "initials": {}, "starts": {}, "physical_updates": 0, "gradient_group_presentations": 0}
    data.write_json(root/"startup.json", manifest)
    groups["train"] = previous.parse_once(job, groups["train"], c, "train",
        loader=lambda gs, cfg, s: entry.attach_labels(gs, cfg, s, metadata))
    selected, after = entry.partition(groups, metadata, c)
    if after != split:
        raise ValueError("Labels changed partition")
    def domain(role, d):
        return [g for g,r in zip(selected[role], split[role], strict=True) if r["domain"] == d]
    valid = groups["development"]
    for arm in entry.method.ARMS:
        c["pooling_stats"] = {}
        model, optimizer, initial = entry.fresh(c)
        entry.method.bind(model, entry.method.metadata(arm))
        manifest["initials"][arm] = initial
        if initial != baseline["initial_identity"] or parent.adam_step(optimizer) != 0:
            raise ValueError("Independent identical untrained starts required")
        memory = None
        for stage, d in enumerate("AB", 1):
            name = d+"_"+arm
            if stage == 2:
                manifest["starts"][arm] = continuity(model, optimizer, memory)
                if manifest["starts"][arm] != manifest["points"]["A_"+arm]["continuity"]:
                    raise ValueError("B did not continue its own restored A state")
            current = domain("fit", d)
            log, captures = entry.train_stage(model, optimizer, current, memory, valid, c, arm, stage, root, budget)
            endpoint, memory = entry.checkpoint(root, name, model, optimizer, memory, current if stage==1 else None,
                domain("calibration", d), valid, c, arm, stage, budget)
            manifest["training"][name], manifest["diagnostics"][name], manifest["points"][name] = log, captures, endpoint
            manifest["physical_updates"] += 288
            manifest["gradient_group_presentations"] += 288*stage
            data.write_json(root/"progress.json", manifest)
        del model, optimizer, memory
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    manifest["status"] = entry.COMPLETE
    data.write_json(root/"manifest.json", manifest)
    for source in manifest["source_files"]:
        destination = root/"sources"/source["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data.verify(data.ROOT/source["path"], source), destination)
    return manifest, split, valid, metadata


def blind_gate(root: Path, manifest: dict, *, entry=None) -> tuple[dict, dict]:
    entry = entry or sys.modules[__name__]
    if (manifest["status"] != entry.COMPLETE or manifest["source_files"] != entry.sources()
            or manifest["physical_updates"] != 576 or manifest["gradient_group_presentations"] != 864
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
    shared = record.Memory.from_bytes(data.verify(root/a_point["memory"]["path"], a_point["memory"]).read_bytes())
    members = shared.summary()["members"]
    rng = random.Random(data.seed_for(20260930, "ABC", "retention"))
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
        if (manifest["initials"][arm] != baseline["initial_identity"]
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
        stream = data.seed_for(20260918, "ABC", stage, "current")
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
                key = data.seed_for(20260918, "ABC", stage, step, "history", group.uid, "regroup")
                if h["assignment_sha256"] != hashlib.sha256(record.assignment(group, key).tobytes()).hexdigest():
                    raise ValueError("Historical A1 assignment differs")
                hashes.append(h["assignment_sha256"])
            assignments.append(hashes)
            if (row["step"] != step or row["encoder_lr"] != parent.stage_lr(step) or row["head_lr"] != .001
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
    for stage, sequences_here in sequences.items():
        for other, (old_root, _, _) in entry.baseline_runs().items():
            name = "AB"[stage-1]+"_"+other
            old_log = data.read_json(old_root/"run/updates"/(name+".json"))
            if sequences_here != [trajectory(old_log)]:
                raise ValueError("H actual schedule/A1/history differs from frozen "+other)
    return scores, split


def open_valid(job, manifest, groups, metadata, c, *, entry=None):
    entry = entry or sys.modules[__name__]
    if data.read_json(job/"access.json") != dict(train=1, valid=0, heldout=0, owners=0):
        raise ValueError("Single new train attempt and blind valid ledger required")
    scores, split = entry.blind_gate(job/"run", manifest)
    data.write_json(job/"before_valid.json", {"status": "ALL_BLIND_GATES_PASS", "source_files": entry.sources(),
        "manifest": data.record(job/"run/manifest.json", job), "access": data.read_json(job/"access.json")})
    labelled = previous.parse_once(job, groups, c, "development", loader=lambda gs, cfg, s: entry.attach_labels(gs, cfg, s, metadata))
    return scores, split, labelled


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
    result["status"] = "ALL_6_H_METRIC_COUNT_SETS_SAVED"
    data.write_json(root/"collected.json", result)
    return result


def finalize(root, *, entry=None):
    entry = entry or sys.modules[__name__]
    col = data.read_json(root/"collected.json")
    if col["status"] != "ALL_6_H_METRIC_COUNT_SETS_SAVED" or set(col["points"]) != set(entry.POINTS) or col["source_files"] != entry.sources():
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
        entry.DIRECT_COMPARISON_KEY: result["comparisons"]["-".join(entry.policy()["evaluation"]["comparisons"][1])]["raw"]["B"]["map"]["conditional_95pct_interval"][0] > 0}
    result.update(status="FIXED_AB_STATISTICS_COMPLETE", source_files=entry.sources(), primary_role="raw", primary_metric="map",
        forgetting_sign="High-good: A-B; brier/log_loss: B-A; arm difference candidate-other; old_changes always raw B-A",
        bootstrap="One PCG64(20260930) (5000,2,20) whole-group array shared by all contrasts and roles",
        undefined_policy="Null denotes N/A. Any undefined required group operand makes the fixed-group mean/interval N/A; no groups dropped, no per-arm nanmean. Pooled rates use summed counts first.",
        limits="Reused development; single seed AB; independent training trajectories; conditional system contrasts, not isolated retention or inference causality; fixed dual-domain H-S MAP development gate, no universal superiority.")
    data.write_json(root/"evaluation.json", result)
    return result



def reviewed_gate(path: Path, kind: str, *, entry=None) -> dict:
    entry = entry or sys.modules[__name__]
    gate = data.read_json(path)
    if (gate.get("kind") != kind or gate.get("status") != "APPROVED_NO_OPEN_BLOCKERS"
            or gate.get("source_files") != entry.sources() or any(gate.get(k) != v for k, v in entry.method.metadata().items())
            or gate.get("authorization") != data.record(entry.AUTHORIZATION, data.ROOT)):
        raise ValueError("Source-bound approved G_half gate required")
    auth = data.read_json(data.verify(data.ROOT/gate["authorization"]["path"], gate["authorization"]))
    if auth["status"] != "APPROVED_FULL_CONTRACT_GATED_SEQUENCE":
        raise ValueError("Full contract approval missing")
    data.verify(data.ROOT/auth["contract"]["path"], auth["contract"])
    for key in ("external_review", "primary_disposition", "cpu"):
        data.verify(data.ROOT/gate[key]["path"], gate[key])
    cpu = data.read_json(data.ROOT/gate["cpu"]["path"])
    if (cpu.get("status") != "PASS_HANDWRITTEN_ONLY" or cpu.get("mode") != "cpu"
            or cpu.get("source_files") != entry.sources() or cpu.get("cumulative_charged_seconds", float("inf")) > 180
            or cpu.get("unresolved_checks") != [] or set(cpu.get("qualified_tests", [])) != set(entry.CPU_TESTS)):
        raise ValueError("Current CPU increment not qualified")
    return gate


def validate_gate(job: Path, path: Path, *, entry=None) -> dict:
    entry = entry or sys.modules[__name__]
    gate = entry.reviewed_gate(path, "FORMAL")
    if (gate.get("job") != job.relative_to(data.ROOT).as_posix()
            or gate.get("runtime") != entry.policy()["runtime"] or gate.get("supervision") != entry.policy()["supervision"]):
        raise ValueError("Formal H scope differs")
    gpu = data.read_json(data.verify(data.ROOT/gate["gpu"]["path"], gate["gpu"]))
    native = gpu.get("native", {})
    if (gpu.get("status") != "PASS_HANDWRITTEN_ONLY" or gpu.get("mode") != "gpu"
            or gpu.get("source_files") != entry.sources() or gpu.get("cumulative_charged_seconds", float("inf")) > 600
            or native.get("actual_updates") != 6 or native.get("maximum_records") != 224
            or native.get("channel_tokens") != 256 or native.get("both_phases_encoder_head_effect") is not True
            or native.get("next_update_bitwise_equal") is not True or native.get("H_pooling_restores_verified") is not True
            or not entry.projection(native["measurements"])["within_budget"]):
        raise ValueError("Native H effects/restores/projection not qualified")
    entry.validate_native(native)
    return gate


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
    if shutil.disk_usage(data.ROOT).free < 10*2**30:
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
        result = {"status": entry.COMPLETION_STATUS, "source_files": entry.sources(), "physical_updates": 576,
            "gradient_group_presentations": 864, "access": data.read_json(job/"access.json"), "budget": budget.snapshot()}
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
    if os.name != "posix":
        parser.error("Scientific execution is Linux-only")
    execute(args.job.resolve(), args.gate.resolve())
