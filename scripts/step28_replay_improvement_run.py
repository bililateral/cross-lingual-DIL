"""Five fixed arms, full first-stage reruns, neutral epoch diagnostics, one blind opening."""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import signal
import threading
import time
import traceback
import zipfile
from typing import Any, Callable

import numpy as np

import step28_record_replay as record
import step28_record_replay_run as old
import step28_er_weight as logit
import step28_replay_diagnostics as diagnostics

data, parent, base, core = record.data, record.parent, record.base, record.core
previous, evaluation = old.previous, old.evaluation
POLICY = data.ROOT/"schema/step28_replay_improvement_policy.json"
ARMS = ("LOGIT0.1", "C", "S", "C_plus", "S_strong")
RECORD_ARMS = ARMS[1:]


def policy() -> dict:
    p = data.read_json(POLICY)
    if (p["methods"] != list(ARMS) or p["physical_updates"] != 10368 or p["physical_stage_states"] != 36
            or p["epoch_diagnostic_points"] != 216 or p["gradient_group_presentations"] != 19008
            or p["teacher_coefficients"] != {"C": [.25,.25], "S": [.5,0.], "C_plus": [.5,.25], "S_strong": [.75,0.]}
            or p["orders"] != list(parent.ORDERS) or p["seed"] != "s0"):
        raise ValueError("Approved five-arm scope differs")
    return p


def config(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Unapproved arm")
    c = parent.config(parent.contract()) if arm == "LOGIT0.1" else record.config()
    c["runtime"] = policy()["runtime"]
    return c


def sources() -> list[dict]:
    paths = {data.ROOT/name for name in (
        "scripts/step28_replay_improvement_run.py", "scripts/step28_replay_improvement_verify.py",
        "scripts/step28_replay_diagnostics.py", "tests/test_step28_replay_improvement.py")}
    todo = list(paths)
    while todo:
        path = todo.pop()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [v.name for v in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                if name and name.startswith("step28_"):
                    child = data.ROOT/"scripts"/(name+".py")
                    if child not in paths:
                        paths.add(child)
                        todo.append(child)
    paths.update(data.ROOT/"schema"/name for name in (
        "step28_replay_improvement_policy.json", "step28_record_replay_policy.json",
        "step28_bge_continual_policy.json", "step28_alias_ranking_policy.json", "step28_chinese_base_policy.json"))
    paths.update(data.ROOT/name for name in ("docs/SELLER_ALIAS_REPLAY_IMPROVEMENT.zh.md",
        "scripts/run_step28_replay_improvement_linux_20261008.sh", "tests/test_step28_record_replay.py"))
    return [data.record(path, data.ROOT) for path in sorted(paths)]


def point_name(order: str, arm: str, stage: int) -> str:
    return order+"_shared" if stage == 1 and arm != "LOGIT0.1" else f"{order}_{arm}_stage{stage}"


def specifications() -> dict:
    return {point_name(order, arm, stage): (order, arm, stage)
            for order in parent.ORDERS for arm in ARMS for stage in (1,2,3)}


class Budget(old.Budget):
    def __init__(self, root: Path):
        super().__init__(root)
        self.limits = policy()["runtime"]


def memory_from(payload: bytes, arm: str) -> parent.Memory | record.Memory:
    return (parent.Memory if arm == "LOGIT0.1" else record.Memory).from_bytes(payload)


def update(model: Any, optimizer: Any, current: data.Group, history: data.Group | None,
           target: np.ndarray | None, c: dict, arm: str, order: str, stage: int,
           step: int, check: Callable) -> dict:
    if parent.adam_step(optimizer) != (stage-1)*288+step-1:
        raise ValueError("Actual update's Adam step differs")
    if arm != "LOGIT0.1":
        return record.update(model, optimizer, current, history, target, c, arm, order, stage, step, check)
    stream = data.seed_for(20260918, order, stage, "current")
    seeds = (data.seed_for(stream, step-1, "dropout"), data.seed_for(20260930, order, stage, step-1, "history_dropout"))
    if stage == 1:
        return parent.update(model, optimizer, current, None, None, c, "logit", stage, step, *seeds, check=check)
    return logit.update(model, optimizer, current, history, c, .1, stage, step, *seeds,
                        reference=target, logit_weight=.5, check=check)


def train_stage(model: Any, optimizer: Any, current: list, memory: Any, valid: list,
                c: dict, arm: str, order: str, stage: int, root: Path, budget: Any) -> tuple[dict,list]:
    if parent.adam_step(optimizer) != (stage-1)*288:
        raise ValueError("Adam continuity differs")
    if memory is not None:
        memory.begin_stage(stage)
    before = None if memory is None else memory.summary()
    schedule, _ = parent.schedule(current, parent.contract(), order, stage)
    rows, epoch_records = [], []
    name = point_name(order, arm, stage)
    budget.phase = name
    started = time.monotonic()
    for step, group in enumerate(schedule, 1):
        history, target = memory.draw() if memory is not None else (None,None)
        row = update(model, optimizer, group, history, target, c, arm, order, stage, step, budget.check)
        row.update(step=step, current_uid=group.uid, history_uid=None if history is None else history.uid)
        rows.append(row)
        if step % 48 == 0:
            epoch_records.append(diagnostics.capture(model, optimizer, current, memory, valid, c, arm,
                order, stage, step//48, root/"diagnostics", name, budget.check))
        if step == 1 or step % 24 == 0:
            print(data.json_bytes({"event":"updates", "point":name, "completed":step,
                  "stage_elapsed_seconds":time.monotonic()-started, **budget.snapshot()}).decode(), flush=True)
    after = None if memory is None else memory.summary()
    keys = ("members","seen","reference_origins","references") if arm == "LOGIT0.1" else ("members","seen","origins","tables")
    if memory is not None and any(before[k] != after[k] for k in keys):
        raise ValueError("Stage mutated retained membership or birth sources")
    result = {"order":order,"arm":arm,"stage":stage,"updates":288,"adam_step":parent.adam_step(optimizer),
              "current_ids":[g.uid for g in schedule],"history_ids":[r["history_uid"] for r in rows if r["history_uid"] is not None],
              "memory_before":before,"memory_after":after,"rows":rows,"elapsed_seconds":time.monotonic()-started}
    path = root/"updates"/(name+".json")
    data.write_json(path,result)
    return data.record(path,root), epoch_records


def checkpoint(root: Path, name: str, model: Any, optimizer: Any, memory: Any,
               current: list, calibration: list, valid: list, c: dict, arm: str,
               order: str, stage: int, budget: Any) -> tuple[dict, Any]:
    if len(calibration) != 12 or len(valid) != 60 or parent.adam_step(optimizer) != 288*stage:
        raise ValueError("Incomplete stage endpoint")
    state = diagnostics.rng_state()
    with diagnostics.neutral(model):
        scores = {"calibration":diagnostics.score(model,calibration,c,arm,budget.check),
                  "raw":diagnostics.score(model,valid,c,arm,budget.check)}
        mapping = parent.calibration.fit(scores["calibration"], np.asarray([g.labels for g in calibration],np.uint8),
                                         role="calibration", check=budget.check)
        map_path = root/"maps"/(name+".json")
        data.write_json(map_path,mapping)
        if mapping["status"] != "PASS_CALIBRATION_FIT":
            raise ValueError("Calibration failed; no retry")
        current_map = dict(zip(("a","b"),parent.calibration.parameters(mapping)))
        if memory is None:
            memory = parent.Memory(order,20260930,True,current_map) if arm == "LOGIT0.1" else record.Memory(order)
        first_map = memory.first_map if arm == "LOGIT0.1" else memory.maps.get("first",current_map)
        auxiliary = {"first":first_map,"stage":current_map,"rng":state,"completed_stage":stage,"adam_step":288*stage}
        if arm == "LOGIT0.1":
            memory.auxiliary = auxiliary
            if stage < 3:
                memory.retain(current,stage,lambda groups:diagnostics.score(model,groups,c,arm,budget.check))
        else:
            memory.maps = auxiliary
            if stage < 3:
                memory.retain(current,stage,lambda group:record.reference(model,group,c,budget.check))
    mp = root/"memory"/(name+".bin")
    mp.write_bytes(memory.to_bytes())
    memory = memory_from(mp.read_bytes(),arm)
    meta = {"name":name,"order":order,"stage":stage,"arm":arm,"rng":state,
            "memory":data.record(mp,root),"policy_sha256":data.sha256(POLICY)}
    full_path = root/"work"/(name+".pt")
    budget.check(base.persistence.checkpoint_reserve(model,optimizer))
    full = core.save_state(full_path,model,optimizer,meta)
    if core.restore_state(full_path,model,optimizer,full["state_sha256"]) != meta:
        raise ValueError("Full model/Adam metadata differs")
    diagnostics.restore_rng(state)
    with diagnostics.neutral(model):
        for role,groups in (("calibration",calibration),("raw",valid)):
            if not np.array_equal(scores[role],diagnostics.score(model,groups,c,arm,budget.check)):
                raise ValueError("Full restored scores differ")
    budget.check(base.persistence.checkpoint_reserve(model,None))
    model_path = root/"models"/(name+".pt")
    inference = core.save_state(model_path,model,None,meta)
    if core.restore_state(model_path,model,None,inference["state_sha256"]) != meta:
        raise ValueError("Inference restore differs")
    diagnostics.restore_rng(state)
    for role,m in (("stage-cal",current_map),("first-cal",first_map)):
        scores[role] = parent.calibration.transform(scores["raw"],m)
        parent.calibration.preserve_order(scores["raw"],scores[role])
    keep_full = stage == 1 and arm != "LOGIT0.1"
    result = {"order":order,"arm":arm,"stage":stage,"adam_step":stage*288,"full_restore_verified":True,
        "calibration_ids":[g.uid for g in calibration],"maps":{"first":first_map,"stage":current_map},
        "mapping":data.record(map_path,root),"memory":data.record(mp,root),"memory_summary":memory.summary(),
        "full_state":{**full,"path":full_path.relative_to(root).as_posix()},"full_retained":keep_full,
        "model":{**inference,"path":model_path.relative_to(root).as_posix()},
        "scores":{role:previous.save_array(root/"scores"/(name+"_"+role+".npy"),value,root) for role,value in scores.items()}}
    if not keep_full:
        base.persistence.remove_work_file(full_path,root)
    data.write_json(root/"points"/(name+".json"),result)
    return result,memory


def train(job: Path,budget: Any) -> tuple[dict,dict,list]:
    import torch
    root = job/"run"
    root.mkdir()
    for sub in ("models","work","memory","points","scores","maps","updates","diagnostics"):
        (root/sub).mkdir()
    groups,metadata,checked = base.public.public_inputs(config("C"))
    _,partition = base.partition(groups,metadata,config("C"))
    archive = core.model_files(base.model_config(config("C"),"split_rank"))
    if any(archive[k] != config("C")["models"]["split_rank"][k] for k in ("file_count","total_size_bytes","content_sha256")):
        raise ValueError("Pretrained BGE differs")
    data.write_json(root/"partition.json",partition)
    manifest = {"status":"RUNNING","source_files":sources(),"public_inputs":checked,"pretrained_archive":archive,
        "partition":data.record(root/"partition.json",root),"points":{},"training":{},"diagnostics":{},"starts":{},
        "physical_updates":0,"gradient_group_presentations":0}
    data.write_json(root/"startup.json",manifest)
    # Token lengths are checked by every real forward; no extra data or labels.
    groups["train"] = previous.parse_once(job,groups["train"],config("C"),"train")
    selected,after = base.partition(groups,metadata,config("C"))
    if after != partition:
        raise ValueError("Partition changed after supervision alignment")
    supply = parent.Supply(selected,partition)
    initial_digests = {}
    for order in parent.ORDERS:
        for family in ("LOGIT0.1","record"):
            c = config("C" if family == "record" else family)
            model = record.load_model(c)
            digest = core.state_digest(model.state_dict())
            if family in initial_digests and initial_digests[family] != digest:
                raise ValueError("Order initialization differs")
            initial_digests[family] = digest
            optimizer = core.make_optimizer(model,c)
            arms = ("LOGIT0.1",) if family == "LOGIT0.1" else ("shared",*RECORD_ARMS)
            shared = None
            for branch in arms:
                arm = "C" if branch == "shared" else branch
                path = order+"_"+branch
                if branch in RECORD_ARMS:
                    supply.branch_after_first(path,order+"_shared")
                    model = record.load_model(c)
                    optimizer = core.make_optimizer(model,c)
                    full = shared["full_state"]
                    meta = core.restore_state(data.verify(root/full["path"],full),model,optimizer,full["state_sha256"])
                    memory = memory_from(data.verify(root/shared["memory"]["path"],shared["memory"]).read_bytes(),arm)
                    diagnostics.restore_rng(meta["rng"])
                    manifest["starts"][path] = {"full_state_sha256":full["state_sha256"],"memory":memory.summary(),"rng":diagnostics.rng_state(),"adam_step":parent.adam_step(optimizer)}
                    stages = (2,3)
                else:
                    memory = None
                    stages = (1,) if branch == "shared" else (1,2,3)
                for stage in stages:
                    current,cal = supply.current(path,order,stage)
                    name = point_name(order,arm,stage)
                    log,diags = train_stage(model,optimizer,current,memory,groups["development"],c,arm,order,stage,root,budget)
                    point,memory = checkpoint(root,name,model,optimizer,memory,current,cal,groups["development"],c,arm,order,stage,budget)
                    manifest["training"][name],manifest["diagnostics"][name],manifest["points"][name] = log,diags,point
                    manifest["physical_updates"] += 288
                    manifest["gradient_group_presentations"] += 288 if stage == 1 else 576
                    if branch == "shared":
                        shared = point
                    data.write_json(root/"progress.json",manifest)
                del model,optimizer,current,cal,memory
                gc.collect()
                torch.cuda.empty_cache()
            if shared is not None:
                base.persistence.remove_work_file(root/shared["full_state"]["path"],root)
    manifest.update(status="ALL_10368_UPDATES_36_STATES_216_EPOCHS_VALID_BLIND",initial_model_digests=initial_digests)
    data.write_json(root/"manifest.json",manifest)
    return manifest,partition,groups["development"]


def check_diagnostics(root: Path, records: list, log: dict, point: dict, partition: dict) -> None:
    if len(records) != 6:
        raise ValueError("Missing epoch diagnostics")
    expected_fit = [r["group_uid"] for r in partition["fit"] if r["domain"] == point["order"][point["stage"]-1]]
    valid_ids = [r["group_uid"] for r in partition["development"]]
    domain_of = {r["group_uid"]:r["domain"] for r in partition["fit"]}
    expected_cache = [] if point["stage"] == 1 else log["memory_before"]["members"]
    for epoch,rec in enumerate(records,1):
        row = data.read_json(data.verify(root/rec["path"],rec))
        if (row["epoch"] != epoch or row["stage_step"] != 48*epoch or row["stage"] != point["stage"]
                or row["adam_step"] != (point["stage"]-1)*288+48*epoch or row["arm"] != point["arm"]
                or row["order"] != point["order"] or row["valid_labels_used"] is not False
                or row["fit_ids"] != expected_fit or row["cache_ids"] != expected_cache or row["valid_ids"] != valid_ids
                or row["fit_domains"] != [domain_of[u] for u in expected_fit]
                or row["cache_domains"] != [domain_of[u] for u in expected_cache]):
            raise ValueError("Diagnostic scope, identity or epoch differs")
        previous.array(root,row["fit"],(48,3),np.float64)
        previous.array(root,row["cache"],(len(expected_cache),3),np.float64)
        if [r["uid"] for r in row["teacher_errors"]] != expected_cache:
            raise ValueError("Cache source-diagnostic identity differs")
        for error in row["teacher_errors"]:
            numeric = ("D0",) if point["arm"] == "LOGIT0.1" else ("D0","D1","residual_cross_mean","residual_difference_mse")
            if not all(np.isfinite(error[k]) for k in numeric):
                raise ValueError("Nonfinite source-error diagnostic")
            if point["arm"] == "LOGIT0.1" and any(error[k] is not None for k in ("D1","residual_cross_mean","residual_difference_mse")):
                raise ValueError("LOGIT has no A1 source")
        values = previous.array(root,row["valid_blind"],(60,378),np.float32)
        if epoch == 6 and not np.array_equal(values,previous.array(root.parent,point["scores"]["raw"],(60,378),np.float32)):
            raise ValueError("Epoch6 and stage-end raw predictions differ")


def blind_gate(root: Path,manifest: dict) -> tuple[dict,dict]:
    specs = specifications()
    if (manifest["status"] != "ALL_10368_UPDATES_36_STATES_216_EPOCHS_VALID_BLIND" or manifest["source_files"] != sources()
            or manifest["physical_updates"] != 10368 or manifest["gradient_group_presentations"] != 19008
            or any(set(manifest[k]) != set(specs) for k in ("points","training","diagnostics"))):
        raise ValueError("Incomplete source-bound five-arm experiment")
    partition = data.read_json(data.verify(root/manifest["partition"]["path"],manifest["partition"]))
    fit = {r["group_uid"]:r["domain"] for r in partition["fit"]}
    for order in parent.ORDERS:
        if any(manifest["starts"][order+"_"+arm] != manifest["starts"][order+"_C"] for arm in RECORD_ARMS):
            raise ValueError("Record arms did not share a complete first state")
    scores, sequences = {},{}
    for name,(order,arm,stage) in specs.items():
        point = manifest["points"][name]
        if point["order"] != order or point["stage"] != stage or point["arm"] != ("C" if name.endswith("_shared") else arm):
            raise ValueError("Physical point identity differs")
        # The physical shared point uses C; all four logical arms alias its bytes.
        arm = point["arm"]
        log = data.read_json(data.verify(root/manifest["training"][name]["path"],manifest["training"][name]))
        ids = sorted(u for u,d in fit.items() if d == order[stage-1])
        rng = random.Random(data.seed_for(20260918,order,stage,"current"))
        expected = []
        for _ in range(6):
            copy = ids.copy()
            rng.shuffle(copy)
            expected.extend(copy)
        if (log["updates"] != 288 or log["adam_step"] != stage*288 or log["current_ids"] != expected
                or len(log["rows"]) != 288 or not point["full_restore_verified"] or point["adam_step"] != stage*288):
            raise ValueError("Stage schedule, updates or restoration differs")
        memory = memory_from(data.verify(root/point["memory"]["path"],point["memory"]).read_bytes(),arm)
        if memory.summary() != point["memory_summary"]:
            raise ValueError("Saved history differs")
        retained,seen = [],0
        rr = random.Random(data.seed_for(20260930,order,"retention"))
        for arrival in order[:min(stage,2)]:
            for uid in sorted(u for u,d in fit.items() if d == arrival):
                seen += 1
                i = len(retained) if len(retained)<6 else rr.randrange(seen)
                if i < 6:
                    if i == len(retained):
                        retained.append(uid)
                    else:
                        retained[i] = uid
        if retained != memory.summary()["members"] or seen != memory.reservoir.seen:
            raise ValueError("Independent Algorithm R membership differs")
        origins = memory.reference_origins if arm == "LOGIT0.1" else memory.origins
        if any(fit[u] != order[s-1] for u,s in origins.items()):
            raise ValueError("Birth domain differs")
        if stage > 1:
            prior = manifest["points"][point_name(order,arm,stage-1)]["memory"]
            old_memory = memory_from(data.verify(root/prior["path"],prior).read_bytes(),arm)
            old_memory.begin_stage(stage)
            if old_memory.summary() != log["memory_before"]:
                raise ValueError("Wrong previous memory")
            draws = [old_memory.draw()[0].uid for _ in range(288)]
            if draws != log["history_ids"] or old_memory.summary() != log["memory_after"]:
                raise ValueError("History sampling differs")
            targets = lambda m:m.references if arm == "LOGIT0.1" else m.tables
            for uid in targets(old_memory).keys() & targets(memory).keys():
                if not np.array_equal(targets(old_memory)[uid],targets(memory)[uid]):
                    raise ValueError("Surviving birth target changed")
            pair = (log["current_ids"],draws,log["memory_before"]["members"])
            if (order,stage) in sequences and sequences[order,stage] != pair:
                raise ValueError("Five-arm samples unpaired")
            sequences[order,stage] = pair
        elif log["history_ids"] or log["memory_before"] is not None:
            raise ValueError("First stage used history")
        for step,row in enumerate(log["rows"],1):
            if row["encoder_lr"] != parent.stage_lr(step) or row["head_lr"] != .001 or row["step"] != step:
                raise ValueError("Learning schedule differs")
            if arm == "LOGIT0.1":
                value = row["current_total"] + (.1*row["history_total"]+.5*row["logit_mse"] if stage>1 else 0.)
            else:
                value = row["current"]["supervised"]
                if stage > 1:
                    a,b = policy()["teacher_coefficients"][arm]
                    h = row["history"]
                    value += .1*h["supervised"]+a*h["mse0"]+b*h["mse1"]
            if not np.isfinite(row["total"]) or not np.isclose(value,row["total"],rtol=2e-6,atol=2e-6):
                raise ValueError("Actual loss decomposition differs")
        cal_ids = [r["group_uid"] for r in partition["calibration"] if r["domain"] == order[stage-1]]
        if cal_ids != point["calibration_ids"]:
            raise ValueError("Calibration scope differs")
        data.verify(root/point["model"]["path"],point["model"])
        mapping = data.read_json(data.verify(root/point["mapping"]["path"],point["mapping"]))
        if mapping["status"] != "PASS_CALIBRATION_FIT":
            raise ValueError("Calibration failed")
        previous.array(root,point["scores"]["calibration"],(12,378),np.float32)
        roles = {role:previous.array(root,point["scores"][role],(60,378),np.float32 if role=="raw" else np.float64) for role in parent.ROLES}
        for role,key in (("stage-cal","stage"),("first-cal","first")):
            if not np.array_equal(roles[role],parent.calibration.transform(roles["raw"],point["maps"][key])):
                raise ValueError("Saved affine role differs")
        check_diagnostics(root/"diagnostics",manifest["diagnostics"][name],log,point,partition)
        scores[name] = roles
    return scores,partition


def collect(root: Path,scores: dict,labelled: list,partition: dict) -> dict:
    root.mkdir()
    if [g.uid for g in labelled] != [r["group_uid"] for r in partition["development"]]:
        raise ValueError("Valid alignment differs")
    truth = np.asarray([g.labels for g in labelled],np.uint8)
    result = {"source_files":sources(),"metric_columns":list(parent.metrics.COLUMNS),"group_ids":[g.uid for g in labelled],
              "domains":[r["domain"] for r in partition["development"]],"points":{},
              "logical_points":{arm:{f"{order}_stage{stage}":point_name(order,arm,stage)
                  for order in parent.ORDERS for stage in (1,2,3)} for arm in ARMS}}
    for name,roles in scores.items():
        result["points"][name] = {}
        for role,values in roles.items():
            matrix,counts = parent.metrics.group_metrics(truth,values)
            cp = root/(name+"_"+role+"_counts.json")
            data.write_json(cp,counts)
            result["points"][name][role] = {"matrix":previous.save_array(root/(name+"_"+role+".npy"),matrix,root),"counts":data.record(cp,root)}
    result["status"] = "ALL_108_STAGE_METRIC_COUNT_SETS_SAVED"
    data.write_json(root/"collected.json",result)
    return result


def finalize(root: Path) -> dict:
    col = data.read_json(root/"collected.json")
    if col["status"] != "ALL_108_STAGE_METRIC_COUNT_SETS_SAVED" or col["source_files"] != sources() or set(col["points"]) != set(specifications()):
        raise ValueError("Incomplete collection")
    arrays = {arm:{} for arm in ARMS}
    for arm in ARMS:
        for order in parent.ORDERS:
            for stage in (1,2,3):
                arrays[arm][evaluation.point_name(order,"er",stage)] = old.read_roles(root,col["points"][point_name(order,arm,stage)])
    draws = evaluation.bootstrap_draws()
    endpoints = (*evaluation.ENDPOINTS,"A2")
    fields = {arm:{role:{ep:old.fields(a,col["domains"],role,ep) for ep in endpoints} for role in (*parent.ROLES,"primary")} for arm,a in arrays.items()}
    result = {"endpoints":{arm:{role:{ep:evaluation.summarize_field(v,draws) for ep,v in eps.items()} for role,eps in roles.items()} for arm,roles in fields.items()},"comparisons":{}}
    for candidate,other in policy()["evaluation"]["comparisons"]:
        delta = {role:{ep:evaluation.summarize_field(fields[candidate][role][ep]-fields[other][role][ep],draws) for ep in endpoints} for role in (*parent.ROLES,"primary")}
        versus_raw = {ep:evaluation.summarize_field(fields[candidate]["stage-cal"][ep]-fields[other]["raw"][ep],draws) for ep in endpoints}
        checks = evaluation.comparison_checks(delta["primary"],versus_raw)
        result["comparisons"][candidate+"-"+other] = {"delta":delta,"candidate_cal_minus_other_raw":versus_raw,
            "assessment":checks,"checks":checks["checks"],"pass":checks["pilot_observed_checks_pass"]}
    result["development_criteria_pass"] = all(result["comparisons"]["C_plus-"+a]["pass"] for a in ("S","S_strong","LOGIT0.1"))
    result["absolute_stage_results"] = {}
    rows = evaluation.domain_rows(col["domains"])
    for name,roles in col["points"].items():
        result["absolute_stage_results"][name] = {}
        for role,rec in roles.items():
            matrix,counts = evaluation.read_matrix(root,rec["matrix"]),evaluation.read_counts(root,rec["counts"])
            result["absolute_stage_results"][name][role] = {"macro_all":dict(zip(parent.metrics.COLUMNS,matrix.mean(0).tolist())),
                "macro_by_domain":{d:dict(zip(parent.metrics.COLUMNS,matrix[r].mean(0).tolist())) for d,r in zip("ABC",rows)},"fixed_half_classification":base.fixed_classification(counts,col["domains"])}
    result.update(status="STATISTICS_COMPLETE_REQUIRES_VALID_COMPLETION",source_files=sources(),automatic_followup=False)
    result["historical_replay"] = historical_replay(col,arrays)
    evaluation.write_once(root/"evaluation.json",data.json_bytes(result))
    return result


def historical_replay(col: dict,arrays: dict) -> dict:
    """Registered saved metrics only: rerun-old group/domain differences, no old labels/models."""
    spec = policy()["historical_results"]
    path = data.verify(data.ROOT/spec["path"],spec)
    result = {"source":spec,"scope":"Descriptive new-minus-historical replay; no automatic retry or replacement", "points":{}}
    with zipfile.ZipFile(path) as archive:
        def read(name,rec=None):
            payload = archive.read(name)
            if rec is not None and (len(payload)!=rec["bytes"] or hashlib.sha256(payload).hexdigest()!=rec["sha256"]):
                raise ValueError("Historical payload differs")
            return payload
        current = json.loads(read("result/job/evaluation/collected.json"))
        old_policy = json.loads(read("result/source/schema/step28_record_replay_policy.json"))
        refs = {name:json.loads(read("reference/"+name,rec)) for name,rec in old_policy["reference"]["collections"].items()}
        for registry in (current,*refs.values()):
            if any(registry[k]!=col[k] for k in ("group_ids","domains","metric_columns")):
                raise ValueError("Historical/current group pairing differs")
        rows = evaluation.domain_rows(col["domains"])
        for arm in ("C","S","LOGIT0.1"):
            for order in parent.ORDERS:
                for stage in (1,2,3):
                    folder,registry = "result/job/evaluation",current
                    name = order+"_shared" if stage==1 else f"{order}_{arm}_stage{stage}"
                    if arm=="LOGIT0.1":
                        folder = "reference/reference" if stage==1 else "reference"
                        registry = refs["reference/collected.json" if stage==1 else "collected.json"]
                        name = order+"_shared" if stage==1 else f"{order}_logit_tenth_stage{stage}"
                    key = f"{order}_{arm}_stage{stage}"
                    result["points"][key] = {}
                    for role in parent.ROLES:
                        rec = registry["points"][name][role]["matrix"]
                        before = np.load(io.BytesIO(read(folder+"/"+rec["path"],rec)),allow_pickle=False)
                        if before.shape!=(60,22) or before.dtype!=np.float64 or not np.isfinite(before).all():
                            raise ValueError("Historical metrics incomplete")
                        after = arrays[arm][evaluation.point_name(order,"er",stage)][role]
                        difference = after-before
                        result["points"][key][role] = {"bitwise_equal":bool(np.array_equal(before,after)),
                            "maximum_absolute_group_difference":float(np.abs(difference).max()),
                            "mean_difference_by_actual_domain":{d:dict(zip(parent.metrics.COLUMNS,difference[r].mean(0).tolist())) for d,r in zip("ABC",rows)}}
    return result


def validate_gate(job: Path,gate_path: Path) -> dict:
    p = policy()
    gate = data.read_json(gate_path)
    if (gate.get("status") != "APPROVED_REPLAY_IMPROVEMENT_FORMAL" or gate.get("source_files") != sources()
            or gate.get("job") != job.relative_to(data.ROOT).as_posix() or gate.get("runtime") != p["runtime"]
            or gate.get("supervision") != p["supervision"] or gate.get("review_disposition") != "NO_OPEN_BLOCKERS"):
        raise ValueError("Source-bound reviewed execution gate differs")
    for mode in ("cpu","gpu"):
        rec = gate[mode]
        evidence = data.read_json(data.verify(data.ROOT/rec["path"],rec))
        if evidence.get("mode") != mode or evidence.get("status") != "PASS_HANDWRITTEN_ONLY" or evidence.get("source_files") != sources():
            raise ValueError("Missing matching verification")
        if mode == "gpu":
            native = evidence.get("native",{})
            estimate = native.get("projected_total_seconds",float("inf"))
            if (not np.isfinite(estimate) or not 0<estimate<=172800 or native.get("projected_peak_output_bytes",float("inf"))>64*2**30
                    or native.get("neutrality_both_architectures") is not True or set(native.get("architectures",{})) != {"record","LOGIT0.1"}):
                raise ValueError("Native update/diagnostic qualification or resource projection missing")
    return p


def execute(job: Path,gate_path: Path) -> dict:
    import torch
    from step28_record_replay_verify import preflight
    p = validate_gate(job,gate_path)
    if job.exists() or not job.is_relative_to(data.ROOT/"reports"):
        raise ValueError("New independent output required")
    resources = preflight()
    historical = p["historical_results"]
    data.verify(data.ROOT/historical["path"],historical)
    if shutil.disk_usage(data.ROOT).free < p["runtime"]["maximum_output_bytes"]:
        raise ValueError("Insufficient output space")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    job.mkdir(parents=True)
    budget = Budget(job)
    data.write_json(job/"access.json",dict(train=0,valid=0,heldout=0,owners=0))
    data.write_json(job/"execution.json",{"gate":data.record(gate_path,data.ROOT),"sources":sources(),"resources":resources})
    stopped = threading.Event()
    def watchdog():
        while not stopped.wait(1):
            try:
                budget.check()
            except BaseException as exc:
                data.write_json(job/"failure.json",{"status":"BUDGET_STOP_NO_RETRY","error":str(exc),"budget":budget.snapshot()})
                os._exit(2)
    def terminate(signum,frame):
        raise RuntimeError("TERM: preserve scene; no automatic retry")
    signal.signal(signal.SIGTERM,terminate)
    watcher = threading.Thread(target=watchdog,daemon=True)
    watcher.start()
    try:
        manifest,partition,valid = train(job,budget)
        budget.phase = "blind_gate"
        scores,verified = blind_gate(job/"run",manifest)
        if verified != partition:
            raise ValueError("Partition differs")
        data.write_json(job/"before_valid.json",{"status":"PASS_COMPLETE_BLIND_GATE","states":36,"epochs":216,
            "manifest":data.record(job/"run/manifest.json",job),"access":data.read_json(job/"access.json")})
        budget.check()
        budget.phase = "collect"
        labelled = previous.parse_once(job,valid,config("C"),"development")
        collect(job/"evaluation",scores,labelled,partition)
        diagnostics.evaluate(job/"run/diagnostics",manifest,labelled,partition,job/"evaluation/diagnostics")
        del labelled,valid,scores
        budget.phase = "statistics"
        result = finalize(job/"evaluation")
        access = data.read_json(job/"access.json")
        if access != dict(train=1,valid=1,heldout=0,owners=0):
            raise ValueError("Access ledger differs")
        completion = {"status":"COMPLETE_REPLAY_IMPROVEMENT_DEVELOPMENT","physical_updates":10368,"epoch_diagnostic_points":216,
            "access":access,"budget":budget.snapshot(),"development_criteria_pass":result["development_criteria_pass"],
            "evaluation":data.record(job/"evaluation/evaluation.json",job),"overfitting":data.record(job/"evaluation/diagnostics/overfitting.json",job)}
        with budget.lock:
            budget.check(16384)
            data.write_json(job/"completion.json",completion)
            try:
                budget.check(1)
            except BaseException:
                (job/"completion.json").unlink(missing_ok=True)
                raise
        return completion
    except BaseException:
        data.write_json(job/"failure.json",{"status":"FAILED_NO_AUTOMATIC_RETRY","traceback":traceback.format_exc(),"budget":budget.snapshot()})
        raise
    finally:
        stopped.set()
        watcher.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--gate",type=Path,required=True)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Existing Linux py310 only")
    execute(args.out.resolve(),args.gate.resolve())
