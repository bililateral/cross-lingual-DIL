"""Approved complete S/G/F independent s0 A->B experiment; no valid opening before complete gates."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import threading
import time
import traceback

import numpy as np

import step28_balanced_pooling as method
import step28_replay_diagnostics as diag
import step28_record_replay_run as legacy

record, data = method.record, method.data
parent, core = record.parent, record.core
base, previous = record.base, legacy.previous
POLICY = data.ROOT / "schema/step28_balanced_pooling_policy.json"
POINTS = tuple(stage+"_"+arm for arm in method.ARMS for stage in "AB")
COMPLETE = "COMPLETE_1728_UPDATES_6_RESTORES_36_DIAGNOSTICS_VALID_BLIND"


def policy() -> dict:
    p = data.read_json(POLICY)
    if (p["methods"] != list(method.ARMS) or p["domains"] != "AB" or p["main_rng_order"] != "ABC"
            or p["physical_updates"] != 1728 or p["gradient_group_presentations"] != 2592
            or p["physical_stage_states"] != 6 or p["epoch_diagnostic_points"] != 36
            or p["solver"] != method.SOLVER or p["native_verification"]["actual_updates"] != 3
            or p["history_supervision"] != .1 or p["teacher_weight"] != .5 or p["seed"] != "s0"
            or p["memory_maximum_bytes"] != 1048576
            or p["evaluation"]["comparisons"] != [["F", "S"], ["G", "S"], ["F", "G"]]
            or p["evaluation"]["overall_pass_gate"] is not False):
        raise ValueError("Approved balanced-pooling contract differs")
    n, r, s = p["native_verification"], p["runtime"], p["supervision"]
    if (n["prior_actual_updates"] != 8 or n["maximum_cumulative_updates"] != 11
            or n["prior_charged_seconds"] != 364 or n["maximum_seconds"] != 900
            or p["cpu_verification"]["prior_charged_seconds"] != 97 or p["cpu_verification"]["maximum_seconds"] != 300
            or r["prior_charged_seconds"] != 18428.570150192827 or r["maximum_cumulative_seconds"] != 86400
            or r["maximum_gpu_stage_seconds"] != math.floor(r["maximum_cumulative_seconds"]-r["prior_charged_seconds"])
            or s != dict(train_parse_attempts=1, valid_parse_attempts=1, test_access=False, owners_access=False,
                         automatic_retry=False, prior_train_parse_attempts=1, prior_valid_parse_attempts=0,
                         maximum_cumulative_train_parse_attempts=2, maximum_cumulative_valid_parse_attempts=1)):
        raise ValueError("Approved amendment ledger differs")
    return p


def config() -> dict:
    c = record.config()
    c["runtime"] = policy()["runtime"]
    return c


def sources() -> list[dict]:
    paths = {data.ROOT / n for n in (
        "scripts/step28_balanced_pooling_run.py", "scripts/step28_balanced_pooling_verify.py",
        "tests/test_step28_balanced_pooling.py", "tests/test_step28_record_replay.py")}
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
    paths.update(data.ROOT / "schema" / n for n in (
        "step28_balanced_pooling_policy.json", "step28_record_replay_policy.json",
        "step28_bge_continual_policy.json", "step28_alias_ranking_policy.json", "step28_chinese_base_policy.json"))
    paths.update(data.ROOT / n for n in ("docs/SELLER_ALIAS_BALANCED_POOLING.zh.md",
        "docs/SELLER_ALIAS_BALANCED_POOLING_AMENDMENT.zh.md",
        "scripts/run_step28_balanced_pooling_linux_20261015.sh"))
    return [data.record(p, data.ROOT) for p in sorted(paths)]


def public_inputs(c: dict) -> tuple[dict, list, dict]:
    root = data.ROOT / c["data_root"]
    for name, key in (("manifest.json", "data_manifest_sha256"), ("validation.json", "data_validation_sha256")):
        if data.sha256(root / name) != c[key]:
            raise ValueError("Pinned data identity differs")
    manifest = data.read_json(root / "manifest.json")
    if (manifest["study"] != "seller_alias_continual_expression_shift" or manifest["root_seed"] != 20260910
            or data.read_json(root / "validation.json")["status"] != "PASS_GENERATION_CONTRACT_NOT_MODEL_QUALIFICATION"):
        raise ValueError("Generation qualification missing")
    checked = {}
    def path(name):
        checked[name] = manifest["files"][name]
        return data.verify(root / name, checked[name])
    with path("groups.csv").open(encoding="utf-8", newline="") as stream:
        metadata = sorted((r for r in csv.DictReader(stream) if r["split"] in ("train", "development")),
                          key=lambda r: (r["domain"], int(r["group_index"])))
    result, seen_items, seen_sellers = {}, set(), set()
    for split, count in (("train", 60), ("development", 20)):
        rows = [r for r in metadata if r["split"] == split]
        known = {r["group_uid"] for r in rows}
        allowed = {r["group_uid"] for r in rows if r["domain"] in "AB"}
        if len(known) != 3*count or any(sorted(int(r["group_index"]) for r in rows if r["domain"] == d) != list(range(count)) for d in "ABC"):
            raise ValueError("Combined metadata partition differs")
        collected, scanned = defaultdict(lambda: defaultdict(list)), set()
        with path(split + "/items.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                if (set(r) != {"group_uid", "seller_uid", "item_uid", "title", "description"}
                        or r["group_uid"] not in known or r["item_uid"] in seen_items):
                    raise ValueError("Unknown/repeated public record")
                seen_items.add(r["item_uid"])
                scanned.add(r["group_uid"])
                if r["group_uid"] not in allowed:
                    continue  # C never reaches Group construction or tokenization.
                collected[r["group_uid"]][r["seller_uid"]].append((r["item_uid"], r["title"], r["description"]))
        if scanned != known or set(collected) != allowed:
            raise ValueError("Input coverage differs")
        result[split] = []
        for r in rows:
            if r["group_uid"] not in allowed:
                continue
            sellers = collected[r["group_uid"]]
            ids = tuple(sorted(sellers))
            if seen_sellers.intersection(ids):
                raise ValueError("Account reused across allowed groups")
            seen_sellers.update(ids)
            g = data.Group(r["group_uid"], ids, tuple(tuple(sorted(sellers[s])) for s in ids))
            g.validate()
            if int(r["accounts"]) != 28 or int(r["items"]) != sum(map(len, g.items)):
                raise ValueError("Record count differs")
            result[split].append(g)
    checked["technical_scan_scope"] = "ABC combined files/public metadata; only AB constructed; no heldout files"
    return result, metadata, checked


def attach_labels(groups: list, c: dict, split: str, metadata: list) -> list:
    if split not in ("train", "development"):
        raise ValueError("Forbidden supervision")
    known = {r["group_uid"] for r in metadata if r["split"] == split}
    allowed = {r["group_uid"] for r in metadata if r["split"] == split and r["domain"] in "AB"}
    if {g.uid for g in groups} != allowed or len(groups) != len(allowed):
        raise ValueError("Only AB groups may receive labels")
    root, name = data.ROOT / c["data_root"], split + "/supervision/pairs.csv"
    manifest = data.read_json(root / "manifest.json")
    rows, scanned = defaultdict(list), defaultdict(int)
    with data.verify(root / name, manifest["files"][name]).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["group_uid", "seller_uid_left", "seller_uid_right", "label"]:
            raise ValueError("Label schema differs")
        for r in reader:
            uid = r["group_uid"]
            if uid not in known:
                raise ValueError("Unknown label group")
            scanned[uid] += 1
            if uid in allowed:
                rows[uid].append(r)  # C label strings never reach numerical alignment.
    if set(scanned) != known or set(scanned.values()) != {378} or set(rows) != allowed:
        raise ValueError("Missing/duplicate label rows; no retry")
    result = []
    for g in groups:
        labels = data.align_labels(g, rows[g.uid])
        if sum(labels) != 20:
            raise ValueError("Expected 20 positive account pairs")
        result.append(data.Group(g.uid, g.sellers, g.items, labels))
    return result


def partition(groups: dict, metadata: list, c: dict) -> tuple[dict, dict]:
    by_id = {r["group_uid"]: r for r in metadata}
    fit = set()
    if len(groups["train"]) != 120 or len(groups["development"]) != 40:
        raise ValueError("AB scope required")
    for domain in "AB":
        ids = [g.uid for g in groups["train"] if by_id[g.uid]["domain"] == domain]
        if len(ids) != 60 or any(by_id[u]["split"] != "train" for u in ids):
            raise ValueError("Current domain count differs")
        ids.sort(key=lambda u: hashlib.sha256(data.json_bytes([c["partition"]["seed"], domain, u])).hexdigest())
        fit.update(ids[:48])
    selected = {"fit": [g for g in groups["train"] if g.uid in fit],
                "calibration": [g for g in groups["train"] if g.uid not in fit], "development": groups["development"]}
    rows = {k: [{"group_uid": g.uid, "domain": by_id[g.uid]["domain"]} for g in v] for k, v in selected.items()}
    if len({r["group_uid"] for v in rows.values() for r in v}) != 160:
        raise ValueError("Partition has duplicate/cross-role groups")
    for role, n in (("fit", 48), ("calibration", 12), ("development", 20)):
        if any(sum(r["domain"] == d for r in rows[role]) != n for d in "AB"):
            raise ValueError("AB per-role coverage differs")
    return selected, rows


class Budget(legacy.Budget):
    def __init__(self, root: Path):
        super().__init__(root)
        self.limits = policy()["runtime"]

    def output_files(self):
        # Only this isolated run's output trees; never scan formal input links.
        extra = (data.ROOT/"reports/qualification",
                 data.ROOT/"reports/documentation/20261015/balanced_pooling_repair")
        sidecars = [self.root.with_suffix(s) for s in (".dispatch.json", ".listener.json", ".listener.console.txt")]
        return list(dict.fromkeys([*super().output_files(), *sidecars,
                                   *(p for folder in extra for p in folder.rglob("*"))]))

    def snapshot(self) -> dict:
        result = super().snapshot()
        result.update(prior_charged_seconds=self.limits["prior_charged_seconds"],
                      cumulative_seconds=result["elapsed_seconds"]+self.limits["prior_charged_seconds"],
                      output_scope="job and wrapper/listener sidecars; qualification and amendment evidence in this isolated workspace")
        return result


def projection(m: dict) -> dict:
    keys = [f"{k}_{a}" for a in method.ARMS for k in ("t_A", "t_B", "v", "h")]
    keys += ["L", "s_F", "r_F", "s_I", "r_I", "W_F", "W_I"]
    if any(not np.isfinite(m[k]) or m[k] <= 0 for k in keys):
        raise ValueError("Actual positive finite measurements for every arm required")
    train = 288*sum(m["t_A_"+a]+m["t_B_"+a] for a in method.ARMS)
    forward = sum(1446*m["v_"+a]+36*m["h_"+a] for a in method.ARMS)
    io = 3*m["L"]+6*(m["s_F"]+m["r_F"]+m["s_I"]+m["r_I"])
    seconds = 1.25*(train+forward+io+3600)
    size = math.ceil(1.1*(m["W_F"]+6*m["W_I"]+2*2**30))
    return {"train_seconds": train, "forward_seconds": forward, "io_seconds": io,
            "projected_total_seconds": seconds, "projected_peak_output_bytes": size,
            "within_budget": seconds <= policy()["runtime"]["maximum_gpu_stage_seconds"] and size <= 16*2**30}


def train_stage(model, optimizer, current, memory, valid, c, arm, stage, root, budget):
    if parent.adam_step(optimizer) != (stage-1)*288:
        raise ValueError("Stage continuity differs")
    if memory is not None:
        memory.begin_stage(stage)
    before = None if memory is None else memory.summary()
    schedule, _ = parent.schedule(current, parent.contract(), method.RNG_ORDER, stage)
    name = "AB"[stage-1]+"_"+arm
    budget.phase = name
    rows, captures = [], []
    for step, group in enumerate(schedule, 1):
        history, target = (None, None) if memory is None else memory.draw()
        row = method.update(model, optimizer, group, history, target, c, arm, stage, step, budget.check)
        if parent.adam_step(optimizer) != (stage-1)*288+step:
            raise ValueError("Actual optimizer call missing")
        rows.append({**row, "step": step, "current_uid": group.uid, "history_uid": history.uid if history else None})
        if step % 48 == 0:
            captures.append(method.capture(model, optimizer, current, memory, valid, c, arm,
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


def memory_from(payload: bytes, arm: str):
    if arm not in method.ARMS:
        raise ValueError("Unknown arm Memory")
    return record.Memory.from_bytes(payload)


def continuity(model, optimizer, memory):
    return {"model": core.state_digest(model.state_dict()), "adam": core.state_digest(optimizer.state_dict()),
            "rng": diag.rng_state(), "modes": [m.training for m in model.modules()],
            "memory_sha256": hashlib.sha256(memory.to_bytes()).hexdigest()}


def restore_point(path, expected, model, optimizer, arm):
    meta = core.restore_state(path, model, optimizer, expected["state_sha256"])
    if method.bind(model, meta) != arm:
        raise ValueError("Restored arm differs")
    for m, mode in zip(model.modules(), meta["modes"], strict=True):
        m.training = mode
    diag.restore_rng(meta["rng"])
    return meta


def checkpoint(root, name, model, optimizer, memory, current, calibration, valid, c, arm, stage, budget):
    if len(calibration) != 12 or len(valid) != 40 or parent.adam_step(optimizer) != 288*stage:
        raise ValueError("Endpoint scope differs")
    optimizer.zero_grad(set_to_none=True)
    c = {**c, "pooling_stats": c.setdefault("pooling_stats", {}),
         "pooling_context": {"stage": stage, "step": 288, "role": "endpoint_prediction_or_restore"}}
    state, modes = diag.rng_state(), [m.training for m in model.modules()]
    with diag.neutral(model):
        scores = {"calibration": method.score(model, calibration, c, arm, budget.check),
                  "raw": method.score(model, valid, c, arm, budget.check)}
        mapping = parent.calibration.fit(scores["calibration"], np.asarray([g.labels for g in calibration], np.uint8),
                                         role="calibration", check=budget.check)
        if mapping["status"] != "PASS_CALIBRATION_FIT":
            raise ValueError("Calibration failed; no retry")
        current_map = dict(zip(("a", "b"), parent.calibration.parameters(mapping)))
        if memory is None:
            memory = record.Memory(method.RNG_ORDER)
        first_map = memory.maps.get("first", current_map)
        memory.maps = {"first": first_map, "stage": current_map, "rng": state,
                       "completed_stage": stage, "adam_step": stage*288, "pooling": method.identity(arm)}
        if stage == 1:
            memory.retain(current, 1, lambda g: record.reference(model, g, c, budget.check))
    mp = root/"memory"/(name+".bin")
    mp.write_bytes(memory.to_bytes())
    memory = memory_from(mp.read_bytes(), arm)
    meta = {"name": name, "rng": state, "modes": modes, "pooling": method.identity(arm),
            "members": [g.uid for g in memory.reservoir.groups], "maps": memory.maps,
            "memory": data.record(mp, root), "policy_sha256": data.sha256(POLICY)}
    full_path = root/"work"/(name+".pt")
    budget.check(base.persistence.checkpoint_reserve(model, optimizer))
    full = core.save_state(full_path, model, optimizer, meta)
    if restore_point(full_path, full, model, optimizer, arm) != meta:
        raise ValueError("Full model/Adam restore mismatch")
    with diag.neutral(model):
        for role, groups in (("calibration", calibration), ("raw", valid)):
            restored = method.restored_score(model, groups, c, budget.check)
            if not np.array_equal(scores[role], restored):
                raise ValueError("Full restored scoring differs")
            scores["full-"+role] = restored
    model_path = root/"models"/(name+".pt")
    budget.check(base.persistence.checkpoint_reserve(model, None))
    inference = core.save_state(model_path, model, None, meta)
    if restore_point(model_path, inference, model, None, arm) != meta:
        raise ValueError("Inference restore mismatch")
    with diag.neutral(model):
        for role, groups in (("calibration", calibration), ("raw", valid)):
            restored = method.restored_score(model, groups, c, budget.check)
            if not np.array_equal(scores[role], restored):
                raise ValueError("Inference restored scoring differs")
            scores["inference-"+role] = restored
    for role, mapping_value in (("stage-cal", current_map), ("first-cal", first_map)):
        scores[role] = parent.calibration.transform(scores["raw"], mapping_value)
        parent.calibration.preserve_order(scores["raw"], scores[role])
    map_path = root/"maps"/(name+".json")
    data.write_json(map_path, mapping)
    result = {"arm": arm, "stage": stage, "adam_step": parent.adam_step(optimizer),
        "full_restore_verified": True, "inference_restore_verified": True, "pooling": method.identity(arm),
        "calibration_ids": [g.uid for g in calibration], "maps": {"first": first_map, "stage": current_map},
        "mapping": data.record(map_path, root), "memory": data.record(mp, root), "memory_summary": memory.summary(),
        "continuity": continuity(model, optimizer, memory), "solver_summary": c.get("pooling_stats", {}).copy(),
        "full_state": {**full, "path": full_path.relative_to(root).as_posix()},
        "model": {**inference, "path": model_path.relative_to(root).as_posix()},
        "scores": {k: previous.save_array(root/"scores"/(name+"_"+k+".npy"), v, root) for k,v in scores.items()}}
    data.write_json(root/"points"/(name+".json"), result)
    base.persistence.remove_work_file(full_path, root)
    return result, memory


def fresh(c):
    random.seed(c["initialization_seed"])
    np.random.seed(c["initialization_seed"])
    model = record.load_model(c)
    optimizer = core.make_optimizer(model, c)
    identity = {"model": core.state_digest(model.state_dict()), "adam": core.state_digest(optimizer.state_dict())}
    return model, optimizer, identity


def verify_archive(archive: dict, c: dict) -> dict:
    expected = c["models"]["split_rank"]
    if not isinstance(archive, dict) or any(archive.get(k) != expected[k]
            for k in ("file_count", "total_size_bytes", "content_sha256")):
        raise ValueError("Pinned BGE archive differs or is missing")
    return archive


def pretrained_archive(c: dict) -> dict:
    return verify_archive(core.model_files(base.model_config(c, "split_rank")), c)


def train(job, budget):
    import torch
    root = job/"run"
    for sub in ("models", "work", "memory", "points", "scores", "maps", "updates", "diagnostics", "sources"):
        (root/sub).mkdir(parents=True)
    c = config()
    archive = pretrained_archive(c)
    groups, metadata, checked = public_inputs(c)
    _, split = partition(groups, metadata, c)
    data.write_json(root/"partition.json", split)
    manifest = {"status": "RUNNING", "source_files": sources(), "public_inputs": checked, "pretrained_archive": archive,
        "partition": data.record(root/"partition.json", root), "points": {}, "training": {}, "diagnostics": {},
        "initials": {}, "starts": {}, "physical_updates": 0, "gradient_group_presentations": 0}
    data.write_json(root/"startup.json", manifest)
    groups["train"] = previous.parse_once(job, groups["train"], c, "train",
        loader=lambda gs, cfg, s: attach_labels(gs, cfg, s, metadata))
    selected, after = partition(groups, metadata, c)
    if after != split:
        raise ValueError("Labels changed partition")
    def domain(role, d):
        return [g for g,r in zip(selected[role], split[role], strict=True) if r["domain"] == d]
    valid = groups["development"]
    for arm in method.ARMS:
        c["pooling_stats"] = {}
        model, optimizer, initial = fresh(c)
        method.bind(model, {"pooling": method.identity(arm)})
        manifest["initials"][arm] = initial
        if initial != manifest["initials"]["S"] or parent.adam_step(optimizer) != 0:
            raise ValueError("Independent identical untrained starts required")
        memory = None
        for stage, d in enumerate("AB", 1):
            name = d+"_"+arm
            if stage == 2:
                manifest["starts"][arm] = continuity(model, optimizer, memory)
                if manifest["starts"][arm] != manifest["points"]["A_"+arm]["continuity"]:
                    raise ValueError("B did not continue its own restored A state")
            current = domain("fit", d)
            log, captures = train_stage(model, optimizer, current, memory, valid, c, arm, stage, root, budget)
            endpoint, memory = checkpoint(root, name, model, optimizer, memory, current if stage==1 else None,
                domain("calibration", d), valid, c, arm, stage, budget)
            manifest["training"][name], manifest["diagnostics"][name], manifest["points"][name] = log, captures, endpoint
            manifest["physical_updates"] += 288
            manifest["gradient_group_presentations"] += 288*stage
            data.write_json(root/"progress.json", manifest)
        del model, optimizer, memory
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    manifest["status"] = COMPLETE
    data.write_json(root/"manifest.json", manifest)
    for source in manifest["source_files"]:
        destination = root/"sources"/source["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data.verify(data.ROOT/source["path"], source), destination)
    return manifest, split, valid, metadata


def solver_gate(stats: dict) -> None:
    """Check the recorded O/T partition; never replace raw P by rounded Q."""
    b, t, it = (stats.get(k) for k in ("blocks", "terminal_acceptances", "iterations"))
    if (any(type(v) is not int for v in (b, t, it, stats.get("failures")))
            or b <= 0 or not 0 <= t <= b or stats["failures"] != 0 or not 1 <= it <= 4096):
        raise ValueError("Missing/failed solver counts")
    p, q, gap = (stats.get(k) for k in ("marginal_error", "feasible_error", "gap_checked"))
    ordinary, terminal_min = stats.get("ordinary_marginal_error"), stats.get("terminal_min_iteration")
    if (any(not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in (p, q, gap))
            or q > method.SOLVER["feasible_tolerance"] or gap > method.SOLVER["gap_tolerance"]):
        raise ValueError("Solver output precision differs")
    if b > t:
        if (not isinstance(ordinary, (int, float)) or not np.isfinite(ordinary)
                or not 0 <= ordinary <= method.SOLVER["marginal_tolerance"] or ordinary > p):
            raise ValueError("Ordinary P evidence differs")
    elif ordinary is not None:
        raise ValueError("No ordinary blocks but non-null ordinary maximum")
    if (t == 0 and (terminal_min is not None or p > method.SOLVER["marginal_tolerance"])) or (t > 0 and (
            type(terminal_min) is not int or terminal_min != 4096 or it != 4096 or p <= method.SOLVER["marginal_tolerance"])):
        raise ValueError("Terminal acceptance iteration/P evidence differs")


def blind_gate(root: Path, manifest: dict) -> tuple[dict, dict]:
    if (manifest["status"] != COMPLETE or manifest["source_files"] != sources()
            or manifest["physical_updates"] != 1728 or manifest["gradient_group_presentations"] != 2592
            or any(set(manifest[k]) != set(POINTS) for k in ("points", "training", "diagnostics"))
            or any(set(manifest[k]) != set(method.ARMS) for k in ("starts", "initials"))):
        raise ValueError("Incomplete source-bound blind experiment")
    split = data.read_json(data.verify(root/manifest["partition"]["path"], manifest["partition"]))
    valid_ids = [r["group_uid"] for r in split["development"]]
    if len(valid_ids) != 40 or any(sum(r["domain"] == d for r in split["development"]) != 20 for d in "AB"):
        raise ValueError("AB valid coverage differs")
    fit = {r["group_uid"]: r["domain"] for r in split["fit"]}
    canonical = sorted(u for u, domain in fit.items() if domain == "A")
    if len(canonical) != 48:
        raise ValueError("Shared A fit scope differs")
    a_point = manifest["points"]["A_S"]
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
    for arm in method.ARMS:
        if (manifest["initials"][arm] != manifest["initials"]["S"]
                or manifest["starts"][arm] != manifest["points"]["A_"+arm]["continuity"]):
            raise ValueError("Independent starts/own A continuation missing")
    scores, sequences = {}, {1: [], 2: []}
    for name in POINTS:
        point = manifest["points"][name]
        arm, stage = name[2:], "AB".index(name[0])+1
        a_point = manifest["points"]["A_"+arm]
        if (point["arm"] != arm or point["stage"] != stage or point["adam_step"] != stage*288
                or point["full_restore_verified"] is not True or point["inference_restore_verified"] is not True
                or point["pooling"] != method.identity(arm)):
            raise ValueError("Endpoint restoration missing")
        logrec = manifest["training"][name]
        log = data.read_json(data.verify(root/logrec["path"], logrec))
        expected_fit = [r["group_uid"] for r in split["fit"] if r["domain"] == "AB"[stage-1]]
        stream = data.seed_for(20260918, "ABC", stage, "current")
        expected = data.schedule([type("ID", (), {"uid": u})() for u in expected_fit], 6, stream)
        if log["updates"] != 288 or len(log["rows"]) != 288 or log["adam_step"] != stage*288 or log["current_ids"] != [g.uid for g in expected]:
            raise ValueError("Main schedule/update count differs")
        memory = memory_from(data.verify(root/point["memory"]["path"], point["memory"]).read_bytes(), arm)
        if (memory.summary() != point["memory_summary"] or memory.reservoir.seen != 48 or memory.summary()["members"] != members
                or memory.maps["pooling"] != method.identity(arm)
                or point["continuity"]["memory_sha256"] != memory.summary()["sha256"]):
            raise ValueError("Terminal memory unexpectedly retained B")
        if stage == 2:
            prior = memory_from(data.verify(root/a_point["memory"]["path"], a_point["memory"]).read_bytes(), arm)
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
            total = row["current"]["supervised"]
            hashes = [row["current"]["assignment_sha256"]]
            if stage == 2:
                h = row["history"]
                total += .1*h["supervised"]+.5*h["mse0"]
                group = next(g for g in memory.reservoir.groups if g.uid == row["history_uid"])
                key = data.seed_for(20260918, "ABC", stage, step, "history", group.uid, "regroup")
                if h["assignment_sha256"] != hashlib.sha256(record.assignment(group, key).tobytes()).hexdigest():
                    raise ValueError("Historical A1 assignment differs")
                hashes.append(h["assignment_sha256"])
            assignments.append(hashes)
            if (row["step"] != step or row["encoder_lr"] != parent.stage_lr(step) or row["head_lr"] != .001
                    or row["pooling"] != method.identity(arm)
                    or not np.isfinite(row["total"]) or not np.isclose(row["total"], total, rtol=2e-6, atol=2e-6)):
                raise ValueError("Actual loss/LR decomposition differs")
        sequences[stage].append((log["current_ids"], log["history_ids"], assignments))
        if arm == "F":
            solver_gate(point["solver_summary"])
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
    if any(seq != [seq[0]]*3 for seq in sequences.values()):
        raise ValueError("Three arm schedules/A1/history draws are not paired")
    return scores, split


def open_valid(job, manifest, groups, metadata, c):
    if data.read_json(job/"access.json") != dict(train=1, valid=0, heldout=0, owners=0):
        raise ValueError("Current access ledger must retain the one prior train attempt")
    scores, split = blind_gate(job/"run", manifest)
    data.write_json(job/"before_valid.json", {"status": "ALL_BLIND_GATES_PASS", "source_files": sources(),
        "manifest": data.record(job/"run/manifest.json", job), "access": data.read_json(job/"access.json")})
    labelled = previous.parse_once(job, groups, c, "development", loader=lambda gs, cfg, s: attach_labels(gs, cfg, s, metadata))
    return scores, split, labelled


def query_metrics(labels, scores):
    """Preserve per-query detail; aggregate agrees with the retained metric core."""
    left, right = np.triu_indices(28, 1)
    adjacency = np.full((len(labels), 28, 28), -np.inf)
    relevant = np.zeros((len(labels), 28, 28), np.uint8)
    adjacency[:, left, right] = adjacency[:, right, left] = scores
    relevant[:, left, right] = relevant[:, right, left] = labels
    order = np.argsort(-adjacency, axis=2, kind="stable")[:, :, :27]
    ranked = np.take_along_axis(relevant, order, axis=2)
    total = relevant.sum(2)
    if not np.all((total >= 1) & (total <= 2)):
        raise ValueError("Query relevance differs")
    ranks = np.arange(1, 28)
    values = [(np.cumsum(ranked, axis=2)*ranked/ranks).sum(2)/total, 1./(np.argmax(ranked, axis=2)+1)]
    values.extend(ranked[:, :, :k].sum(2)/total for k in (1, 3, 5, 10))
    discount = 1./np.log2(ranks+1)
    for k in (1, 3, 5, 10):
        ideal = np.asarray([discount[:min(k, int(n))].sum() for n in total.ravel()]).reshape(total.shape)
        values.append((ranked[:, :, :k]*discount[:k]).sum(2)/ideal)
    return np.stack(values, axis=2)


def collect(root, scores, labelled, split):
    root.mkdir()
    if [g.uid for g in labelled] != [r["group_uid"] for r in split["development"]]:
        raise ValueError("Opened valid alignment differs")
    truth = np.asarray([g.labels for g in labelled], np.uint8)
    result = {"source_files": sources(), "group_ids": [g.uid for g in labelled],
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
    result["status"] = "ALL_18_METRIC_COUNT_SETS_SAVED"
    data.write_json(root/"collected.json", result)
    return result


def undefined_metrics(counts):
    """Only precision/MCC can be undefined for the contracted two-class groups."""
    mask = np.zeros((len(counts), 22), dtype=bool)
    for i, r in enumerate(counts):
        tp, fp, fn, tn = (r[k] for k in ("tp", "fp", "fn", "tn"))
        mask[i, 6] = tp+fp == 0
        mask[i, 11] = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) == 0
    return mask


def pooled_rates(counts):
    result = base.rates(counts)
    if result["tp"]+result["fp"] == 0:
        result.update(precision=None, precision_na_reason="No predicted positives")
    return result


def summarize(matrix, domain_rows, draws):
    result = {}
    for i, d in enumerate("AB"):
        values = matrix[domain_rows[i]]
        result[d] = {}
        for j, k in enumerate(parent.metrics.COLUMNS):
            missing = int(np.isnan(values[:, j]).sum())
            if missing:
                result[d][k] = {"mean": None, "conditional_95pct_interval": None,
                    "undefined_groups": missing, "total_groups": len(values),
                    "na_reason": "At least one required group operand is undefined; fixed groups retained."}
            else:
                estimates = values[:, j][draws[:, i]].mean(1)
                result[d][k] = {"mean": float(values[:, j].mean()),
                    "conditional_95pct_interval": np.quantile(estimates, [.025, .975]).tolist()}
    return result


def finalize(root):
    col = data.read_json(root/"collected.json")
    if col["status"] != "ALL_18_METRIC_COUNT_SETS_SAVED" or set(col["points"]) != set(POINTS) or col["source_files"] != sources():
        raise ValueError("Complete saved collection required")
    rows = [np.flatnonzero(np.asarray(col["domains"]) == d) for d in "AB"]
    if any(len(r) != 20 for r in rows):
        raise ValueError("AB bootstrap scope differs")
    draws = np.random.Generator(np.random.PCG64(20260930)).integers(20, size=(5000, 2, 20))
    arrays, result = {}, {"absolute": {}, "comparisons": {}, "old_changes": {}, "columns": col["columns"]}
    for name, roles in col["points"].items():
        arrays[name], result["absolute"][name] = {}, {}
        if set(roles) != set(parent.ROLES):
            raise ValueError("All calibration roles required")
        for role, rec in roles.items():
            saved = data.read_json(data.verify(root/rec["matrix"]["path"], rec["matrix"]))
            v = np.asarray(saved["values"], np.float64)
            if saved["columns"] != col["columns"] or v.shape != (40, 22) or np.isinf(v).any():
                raise ValueError("Nullable metric matrix differs")
            q = previous.array(root, rec["queries"], (40, 28, 10), np.float64)
            if not np.allclose(q.mean(1), v[:, 12:], atol=1e-12, rtol=1e-12):
                raise ValueError("Saved query and group rankings differ")
            counts = data.read_json(data.verify(root/rec["counts"]["path"], rec["counts"]))
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
    result["initial_comparisons"], result["forgetting_comparisons"] = {}, {}
    for candidate, other in policy()["evaluation"]["comparisons"]:
        key = candidate+"-"+other
        result["comparisons"][key] = {role: summarize(arrays["B_"+candidate][role]-arrays["B_"+other][role], rows, draws) for role in parent.ROLES}
        result["initial_comparisons"][key] = {role: summarize(arrays["A_"+candidate][role]-arrays["A_"+other][role], rows, draws) for role in parent.ROLES}
        result["forgetting_comparisons"][key] = {role: summarize(
            (arrays["A_"+candidate][role]-arrays["B_"+candidate][role])-(arrays["A_"+other][role]-arrays["B_"+other][role]), rows, draws)["A"] for role in parent.ROLES}
    for arm in method.ARMS:
        result["old_changes"][arm] = {role: summarize(arrays["B_"+arm][role]-arrays["A_"+arm][role], rows, draws)["A"] for role in parent.ROLES}
    result.update(status="FIXED_AB_STATISTICS_COMPLETE", source_files=sources(), primary_role="raw", primary_metric="map",
        forgetting_sign="A endpoint minus B-end A ability; old_changes uses the opposite signed ability change",
        bootstrap="One PCG64(20260930) (5000,2,20) whole-group array shared by all contrasts and roles",
        undefined_policy="Null denotes N/A. Any undefined required group operand makes the fixed-group mean/interval N/A; no groups dropped, no per-arm nanmean. Pooled rates use summed counts first.",
        limits="Reused development; single seed AB; independent training trajectories; conditional system contrasts, not isolated retention or inference causality; no overall pass gate.")
    data.write_json(root/"evaluation.json", result)
    return result


def prior_evidence():
    """Exact old receipts, not re-labelled qualification of current F."""
    p = policy()
    refs = p["prior_evidence"]
    loaded = {k: data.read_json(data.verify(data.ROOT/refs[k]["path"], refs[k]))
              for k in ("formal_gate", "failure", "access", "native_result", "qualification_state", "amendment_authorization")}
    data.verify(data.ROOT/refs["native_disposition"]["path"], refs["native_disposition"])
    old, native, ledger = loaded["formal_gate"], loaded["native_result"], loaded["qualification_state"]
    for key, reference in (("gpu", refs["native_result"]), ("native_disposition", refs["native_disposition"])):
        if any(old[key][field] != reference[field] for field in ("bytes", "sha256")):
            raise ValueError("Old qualification receipt identity differs")
    if (old["status"] != "APPROVED_BALANCED_POOLING_FORMAL" or old["review_disposition"] != "NO_OPEN_BLOCKERS"
            or native["status"] != "PASS_HANDWRITTEN_ONLY" or native["source_files"] != old["source_files"]
            or native["native"]["actual_updates"] != 8 or not all(native["native"].get(k) is True for k in
                ("next_update_bitwise_equal", "both_phases_encoder_head_effect", "three_pooling_restores_verified"))
            or ledger["native"]["actual_updates"] != 8 or ledger["native"]["charged_seconds"] != 364
            or loaded["failure"]["status"] != "FAIL_NO_RETRY"
            or loaded["failure"]["budget"]["elapsed_seconds"] != p["runtime"]["prior_charged_seconds"]
            or loaded["access"] != dict(train=1, valid=0, heldout=0, owners=0)):
        raise ValueError("Old failure, access or native ledger differs")
    amendment = data.ROOT/"docs/SELLER_ALIAS_BALANCED_POOLING_AMENDMENT.zh.md"
    data.verify(amendment, loaded["amendment_authorization"]["contract"])
    current = {r["path"]: r for r in sources()}
    changed = {"scripts/step28_balanced_pooling.py", "scripts/step28_balanced_pooling_run.py",
               "scripts/step28_balanced_pooling_verify.py", "tests/test_step28_balanced_pooling.py",
               "schema/step28_balanced_pooling_policy.json", "scripts/run_step28_balanced_pooling_linux_20261014.sh"}
    for r in old["source_files"]:
        if r["path"] not in changed and current.get(r["path"]) != r:
            raise ValueError("Inherited native dependency changed: "+r["path"])
    return loaded


def reviewed_gate(path, kind):
    gate = data.read_json(path)
    p = policy()
    prior_evidence()
    if (gate.get("status") != "APPROVED_BALANCED_POOLING_"+kind or gate.get("source_files") != sources()
            or gate.get("review_disposition") != "NO_OPEN_BLOCKERS" or gate.get("user_scope_approved") is not True
            or gate.get("prior_evidence") != p["prior_evidence"]
            or gate.get("authorization") != p["prior_evidence"]["amendment_authorization"]):
        raise ValueError("Reviewed, approved amendment source-bound gate required")
    for name in ("authorization", "review"):
        data.verify(data.ROOT/gate[name]["path"], gate[name])
    cpu = data.read_json(data.verify(data.ROOT/gate["cpu"]["path"], gate["cpu"]))
    if (cpu.get("mode") != "cpu" or cpu.get("scope") != "terminal-q" or cpu.get("status") != "PASS_HANDWRITTEN_ONLY"
            or cpu.get("source_files") != sources() or cpu.get("prior_compute_seconds", -1) < 97
            or not 97 < cpu.get("cumulative_charged_seconds", 0) <= 300):
        raise ValueError("Matching successful incremental CPU qualification required")
    return gate


def delta_native_gate(gpu, old):
    n = gpu.get("native", {})
    if (gpu.get("mode") != "gpu" or gpu.get("scope") != "terminal-q" or gpu.get("status") != "PASS_HANDWRITTEN_ONLY"
            or gpu.get("source_files") != sources() or gpu.get("prior_compute_seconds") != 364
            or not 364 < gpu.get("cumulative_charged_seconds", 0) <= 900
            or n.get("actual_updates") != 3 or n.get("cumulative_updates") != 11
            or any(n.get(k) is not True for k in ("next_update_bitwise_equal", "both_phases_encoder_head_effect",
                "F_pooling_restores_verified", "terminal_gpu_vjp_verified"))
            or n.get("inherited_native") != policy()["prior_evidence"]["native_result"]):
        raise ValueError("Matching three-update incremental F qualification required")
    verify_archive(n.get("pretrained_archive"), config())
    # Current report contains old S/G and conservative max IO explicitly, never invented timings.
    for arm in ("S", "G"):
        for key in ("t_A", "t_B", "v", "h"):
            name = key+"_"+arm
            if n["measurements"][name] != old["native"]["measurements"][name]:
                raise ValueError("Inherited S/G measurement differs")
    for key in ("L", "s_F", "r_F", "s_I", "r_I", "W_F", "W_I"):
        if n["measurements"][key] < old["native"]["measurements"][key]:
            raise ValueError("Inherited IO/size maximum lost")
    stress = n["terminal_stress"]
    for key in ("c_gpu", "c_cpu"):
        if not np.isfinite(stress[key]) or stress[key] <= 0:
            raise ValueError("Missing measured terminal stress")
    if stress["accounts"] != 28 or stress["records"] != 84 or stress["pairs"] != 378:
        raise ValueError("Terminal stress was not a complete table")
    for key, multiplier, cost in (("t_A_F", 2, "c_gpu"), ("t_B_F", 5, "c_gpu"),
                                  ("v_F", 1, "c_gpu"), ("h_F", 4, "c_cpu")):
        if n["measurements"][key] != n["natural_measurements"][key]+multiplier*stress[cost]:
            raise ValueError("Hard solver cost not propagated to full call count")
    if set(stress["solver_summaries"]) != {"cpu", "gpu"}:
        raise ValueError("Both measured terminal stress devices required")
    for s in stress["solver_summaries"].values():
        solver_gate(s)
        if s["blocks"] != 378 or s["terminal_acceptances"] != 378:
            raise ValueError("Terminal stress did not actually use hard blocks")
    solver_gate(n["solver_summary"])


def validate_gate(job, path):
    gate = reviewed_gate(path, "FORMAL")
    if gate.get("job") != job.relative_to(data.ROOT).as_posix() or gate.get("runtime") != policy()["runtime"] or gate.get("supervision") != policy()["supervision"]:
        raise ValueError("Formal scope differs")
    gpu = data.read_json(data.verify(data.ROOT/gate["gpu"]["path"], gate["gpu"]))
    delta_native_gate(gpu, prior_evidence()["native_result"])
    if not projection(gpu["native"]["measurements"])["within_budget"]:
        raise ValueError("New projection exceeds approved budget; no formal input opened")
    return gate


def execute(job, gate_path):
    import torch
    from step28_record_replay_verify import preflight
    started = float(os.environ.get("RECORD_REPLAY_STARTED_EPOCH", "nan"))
    if not math.isfinite(started) or not 0 <= time.time()-started < policy()["runtime"]["maximum_gpu_stage_seconds"]:
        raise ValueError("Bounded wrapper start required before any formal read")
    validate_gate(job, gate_path)
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
    budget = Budget(job)
    data.write_json(job/"access.json", dict(train=0, valid=0, heldout=0, owners=0))
    data.write_json(job/"execution.json", {"gate": data.record(gate_path, data.ROOT), "sources": sources(), "resources": resources, "numerics": numerics})
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
        manifest, split, valid, metadata = train(job, budget)
        budget.check()
        scores, split, labelled = open_valid(job, manifest, valid, metadata, config())
        budget.phase = "evaluation"
        collect(job/"evaluation", scores, labelled, split)
        diag.evaluate(job/"run/diagnostics", manifest, labelled, split, job/"diagnostics_evaluation", scope="AB")
        finalize(job/"evaluation")
        budget.check()
        result = {"status": "COMPLETE_BALANCED_POOLING_FIXED_AB", "source_files": sources(), "physical_updates": 1728,
            "gradient_group_presentations": 2592, "access": data.read_json(job/"access.json"), "budget": budget.snapshot()}
        data.write_json(job/"completion.json", result)
        return result
    except BaseException as exc:
        data.write_json(job/"failure.json", {"status": "FAIL_NO_RETRY", "traceback": traceback.format_exc(),
            "solver_failure": getattr(exc, "details", None), "budget": budget.snapshot()})
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
