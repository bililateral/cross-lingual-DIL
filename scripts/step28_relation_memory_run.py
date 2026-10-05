"""Fixed relation-memory pilot; formal access requires a source-bound review gate."""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import os
import random
from pathlib import Path
import shutil
import threading
import time
import traceback

import numpy as np
import psutil

import step28_relation_memory as method
import step28_bge_continual_run as previous
import step28_bge_continual_evaluate as evaluation
import step28_relation_memory_verify as admission

data, base, core, parent = method.data, method.base, method.core, method.parent
POLICY = data.ROOT / "schema/step28_relation_memory_policy.json"
ROLES = parent.ROLES
COMPLETE = "COMPLETE_RELATION_2592_UPDATES_9_ENDPOINTS_VALID_BLIND"


def sources() -> list[dict]:
    # Exact local scientific import closure, never a recursive data-directory scan.
    paths = {Path(__file__), Path(method.__file__), Path(admission.__file__),
             data.ROOT / "tests/test_step28_relation_memory_run.py",
             data.ROOT / "tests/test_step28_relation_memory.py"}
    todo = list(paths)
    while todo:
        path = todo.pop()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([n.name for n in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                if name and name.startswith("step28_"):
                    child = data.ROOT / "scripts" / (name + ".py")
                    if child not in paths:
                        paths.add(child); todo.append(child)
    paths.update(data.ROOT / "schema" / n for n in (
        "step28_relation_memory_policy.json", "step28_bge_continual_policy.json",
        "step28_alias_ranking_policy.json", "step28_chinese_base_policy.json"))
    paths.update(data.ROOT / n for n in (
        "docs/SELLER_ALIAS_RELATION_MEMORY_PILOT.zh.md",
        "docs/SELLER_ALIAS_RELATION_MEMORY_EXECUTION.zh.md",
        "scripts/run_step28_relation_memory_pilot_linux_20261004.sh"))
    return [data.record(p, data.ROOT) for p in sorted(paths)]


def policy() -> dict:
    p = data.read_json(POLICY)
    if (p["orders"] != list(parent.ORDERS) or p["physical_updates"] != 2592
            or p["metric_count_sets"] != 28 or p["supervision"] != dict(train=1,valid=1,heldout=0,owners=0)):
        raise ValueError("Fixed pilot scope differs")
    return p


def point_name(order: str, stage: int) -> str:
    return f"{order}_relation_stage{stage}"


def expected_points() -> list[str]:
    return [point_name(o,s) for o in parent.ORDERS for s in (1,2,3)]


class Budget(base.persistence.Budget):
    def __init__(self, root: Path, p: dict):
        super().__init__(root, p)
        self.started -= max(0., time.time()-float(os.environ["RELATION_STARTED_EPOCH"]))
        self.peak_rss = 0

    def check(self, reserve: int = 0) -> None:
        import torch
        super().check(reserve)
        siblings=sum(p.stat().st_size for p in (self.root.with_suffix(".console.txt"),self.root.with_suffix(".wrapper.txt")) if p.is_file())
        if self.peak_bytes+siblings+reserve>self.config["maximum_output_bytes"]:
            raise RuntimeError("Job plus launcher evidence exceeds output budget")
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        if self.peak_rss > 64*2**30:
            raise RuntimeError("RSS exceeds 64GiB")
        if torch.cuda.is_initialized() and torch.cuda.max_memory_reserved() > 28*2**30:
            raise RuntimeError("CUDA reserved peak exceeds 28GiB")


def checkpoint(root: Path, name: str, model, optimizer, memory, current, current_cal,
               valid, c: dict, stage: int, budget) -> tuple[dict, method.Memory]:
    """Actual full model/Adam/RNG/memory restoration before next-stage consumption."""
    if len(current_cal)!=12 or len(valid)!=60 or parent.adam_step(optimizer)!=stage*288:
        raise ValueError("Incomplete stage endpoint")
    scores = {"calibration":parent.ranking.score(model,current_cal,c,budget.check),
              "raw":parent.ranking.score(model,valid,c,budget.check)}
    mapping = parent.calibration.fit(scores["calibration"],
        np.asarray([g.labels for g in current_cal],np.uint8),role="calibration",check=budget.check)
    data.write_json(root/"maps"/(name+".json"),mapping)
    if mapping["status"] != "PASS_CALIBRATION_FIT":
        raise ValueError("Calibration failed; no alternative fit")
    current_map = dict(zip(("a","b"),parent.calibration.parameters(mapping)))
    first_map = memory.maps.get("first",current_map)
    for role, m in (("stage-cal",current_map),("first-cal",first_map)):
        scores[role] = parent.calibration.transform(scores["raw"],m)
        parent.calibration.preserve_order(scores["raw"],scores[role])
    rng = previous.rng_state()
    memory.maps = {"first":first_map,"stage":current_map,"torch_rng":rng,
                   "completed_stage":stage,"adam_step":stage*288}
    memory.to_bytes()
    retention = (memory.consolidate(current,stage,lambda g:method.reference(model,g,c,budget.check))
                 if stage<3 else None)
    previous.restore_rng(rng)
    payload = memory.to_bytes()
    mp = root/"memory"/(name+".bin")
    mp.write_bytes(payload)
    restored = method.Memory.from_bytes(mp.read_bytes())
    if restored.to_bytes()!=payload:
        raise ValueError("Memory file restoration differs")
    meta = {"name":name,"order":memory.order,"stage":stage,"rng":rng,
            "memory":data.record(mp,root),"policy_sha256":data.sha256(POLICY)}
    full_path = root/"work"/(name+".pt")
    budget.check(base.persistence.checkpoint_reserve(model,optimizer))
    full = core.save_state(full_path,model,optimizer,meta)
    if core.restore_state(full_path,model,optimizer,full["state_sha256"])!=meta:
        raise ValueError("Full checkpoint metadata differs")
    previous.restore_rng(restored.maps["torch_rng"])
    for role,groups in (("calibration",current_cal),("raw",valid)):
        if not np.array_equal(scores[role],parent.ranking.score(model,groups,c,budget.check)):
            raise ValueError("Complete blind score replay differs")
    budget.check(base.persistence.checkpoint_reserve(model,None))
    inference_path = root/"models"/(name+".pt")
    inference = core.save_state(inference_path,model,None,meta)
    if core.restore_state(inference_path,model,None,inference["state_sha256"]) != meta:
        raise ValueError("Inference checkpoint differs")
    previous.restore_rng(rng)
    result = {"stage":stage,"order":memory.order,"adam_step":parent.adam_step(optimizer),
              "scores":{role:previous.save_array(root/"scores"/(name+"_"+role+".npy"),v,root)
                        for role,v in scores.items()},
              "mapping":data.record(root/"maps"/(name+".json"),root),
              "calibration_ids":[g.uid for g in current_cal],"maps":{ "first":first_map,"stage":current_map},
              "memory":data.record(mp,root),"memory_bytes":len(payload),"retention":retention,
              "memory_summary":{"members":[g.uid for g in memory.reservoir.groups],
                  "count":memory.count,"seen":memory.reservoir.seen,"stage":memory.stage},
              "full_restore_verified":True,"full_state":full,
              "model":{**inference,"path":inference_path.relative_to(root).as_posix()}}
    base.persistence.remove_work_file(full_path,root)
    result["intermediate_deleted_bytes"] = full["bytes"]
    data.write_json(root/"points"/(name+".json"),result)
    return result, restored


def train(job: Path, p: dict, budget) -> tuple[dict,dict,list]:
    import torch
    root=job/"run"; root.mkdir()
    for sub in ("models","work","memory","points","scores","maps","updates"):
        (root/sub).mkdir()
    c=method.config()
    groups,metadata,checked=base.public.public_inputs(c)
    _,partition=base.partition(groups,metadata,c)
    files=core.model_files(base.model_config(c,"split_rank"))
    if any(files[k]!=c["models"]["split_rank"][k] for k in ("file_count","total_size_bytes","content_sha256")):
        raise ValueError("Pretrained archive mismatch")
    data.write_json(root/"partition.json",partition)
    manifest={"status":"RUNNING","source_files":sources(),"partition":data.record(root/"partition.json",root),
              "public_inputs":checked,"pretrained_archive":files,"points":{},"training":{}}
    groups["train"]=previous.parse_once(job,groups["train"],c,"train")
    selected,labelled_partition=base.partition(groups,metadata,c)
    if labelled_partition!=partition:
        raise ValueError("Label alignment changed partition")
    supply=parent.Supply(selected,partition)
    initial_digest=None
    for order in parent.ORDERS:
        model=method.load_model(c)
        digest=core.state_digest(model.state_dict())
        if initial_digest is None:
            initial_digest=digest
            scores=parent.ranking.score(model,groups["development"],c,budget.check)
            manifest["initial"]=previous.save_array(root/"scores/initial_raw.npy",scores,root)
            manifest["initial_digest"]=digest
        elif digest!=initial_digest:
            raise ValueError("Order initial model differs")
        optimizer=method.make_optimizer(model,c); memory=method.Memory(order)
        for stage in (1,2,3):
            current,current_cal=supply.current(order,order,stage)
            sequence,_=parent.schedule(current,parent.contract(),order,stage)
            if stage>1: memory.begin_stage(stage)
            rows=[]; history=[]
            for i,g in enumerate(sequence,1):
                record=method.update(model,optimizer,g,memory,c,stage,i,check=budget.check)
                rows.append({k:v for k,v in record.items() if k!="history_uid"})
                history.append(record["history_uid"])
                if i%24==0:
                    print(data.json_bytes({"order":order,"stage":stage,"step":i,**budget.state()}).decode(),flush=True)
            name=point_name(order,stage)
            tr={"order":order,"stage":stage,"current_ids":[g.uid for g in sequence],
                "history_ids":history,"updates":rows,"adam_step":parent.adam_step(optimizer)}
            data.write_json(root/"updates"/(name+".json"),tr)
            _,memory=checkpoint(root,name,model,optimizer,memory,current,current_cal,
                                groups["development"],c,stage,budget)
            manifest["points"][name]=data.record(root/"points"/(name+".json"),root)
            manifest["training"][name]=data.record(root/"updates"/(name+".json"),root)
            manifest["physical_updates"]=len(manifest["points"])*288
            data.write_json(root/"progress.json",manifest)
            del current,current_cal,sequence,rows,history,tr
        del model,optimizer,memory
        gc.collect(); torch.cuda.empty_cache()
    manifest.update(status=COMPLETE,gradient_group_presentations=4320)
    data.write_json(root/"manifest.json",manifest)
    return manifest,partition,groups["development"]


def blind_gate(root: Path, manifest: dict) -> tuple[dict,dict]:
    if (manifest["status"]!=COMPLETE or manifest["physical_updates"]!=2592
            or manifest["gradient_group_presentations"]!=4320
            or set(manifest["points"])!=set(expected_points()) or manifest["source_files"]!=sources()):
        raise ValueError("Incomplete blind training gate")
    partition=data.read_json(data.verify(root/manifest["partition"]["path"],manifest["partition"]))
    scores={"initial":{"raw":previous.array(root,manifest["initial"],(60,378),np.float32)}}
    for order in parent.ORDERS:
        old_members=[]; seen=0
        retention_rng=random.Random(data.seed_for(20260930,order,"retention"))
        for stage in (1,2,3):
            name=point_name(order,stage)
            point=data.read_json(data.verify(root/manifest["points"][name]["path"],manifest["points"][name]))
            tr=data.read_json(data.verify(root/manifest["training"][name]["path"],manifest["training"][name]))
            fit=[r["group_uid"] for r in partition["fit"] if r["domain"]==order[stage-1]]
            dummy=[type("Scheduled",(),{"uid":uid,"labels":(1,)})() for uid in fit]
            expected,_=parent.schedule(dummy,parent.contract(),order,stage)
            draw_rng=random.Random(data.seed_for(20260930,order,stage,"history_draws"))
            expected_history=([old_members[draw_rng.randrange(6)] for _ in range(288)] if stage>1 else [None]*288)
            if (tr["current_ids"]!=[g.uid for g in expected] or len(tr["updates"])!=288
                    or tr["adam_step"]!=stage*288 or point["adam_step"]!=stage*288
                    or not point["full_restore_verified"]
                    or tr["history_ids"]!=expected_history):
                raise ValueError("Schedule/history/real update evidence differs")
            data.verify(root/point["model"]["path"],point["model"])
            data.verify(root/point["memory"]["path"],point["memory"])
            if point["memory_bytes"]!=point["memory"]["bytes"] or point["memory_bytes"]>method.MAXIMUM_BYTES:
                raise ValueError("Complete memory budget differs")
            if stage<3:
                for uid in sorted(fit):
                    seen+=1
                    index=len(old_members) if len(old_members)<6 else retention_rng.randrange(seen)
                    if index<6:
                        if index==len(old_members): old_members.append(uid)
                        else: old_members[index]=uid
            if point["memory_summary"]!={"members":old_members,"count":seen,"seen":seen,"stage":min(stage,2)}:
                raise ValueError("Independent reservoir/retention evidence differs")
            cal_ids=[r["group_uid"] for r in partition["calibration"] if r["domain"]==order[stage-1]]
            mapping=data.read_json(data.verify(root/point["mapping"]["path"],point["mapping"]))
            if point["calibration_ids"]!=cal_ids or mapping["status"]!="PASS_CALIBRATION_FIT":
                raise ValueError("Current-only calibration differs")
            raw=previous.array(root,point["scores"]["raw"],(60,378),np.float32)
            previous.array(root,point["scores"]["calibration"],(12,378),np.float32)
            scores[name]={"raw":raw}
            for role,key in (("stage-cal","stage"),("first-cal","first")):
                values=previous.array(root,point["scores"][role],(60,378),np.float64)
                if not np.array_equal(values,parent.calibration.transform(raw,point["maps"][key])):
                    raise ValueError("Saved mapping transform differs")
                parent.calibration.preserve_order(raw,values)
                scores[name][role]=values
    return scores,partition


def collect(root: Path, scores: dict, labelled: list, partition: dict) -> dict:
    root.mkdir()
    if set(scores)!={"initial",*expected_points()} or [g.uid for g in labelled]!=[r["group_uid"] for r in partition["development"]]:
        raise ValueError("Incomplete aligned collection")
    truth=np.asarray([g.labels for g in labelled],np.uint8)
    result={"group_ids":[g.uid for g in labelled],"domains":[r["domain"] for r in partition["development"]],
            "metric_columns":list(parent.metrics.COLUMNS),"points":{},"source_files":sources()}
    for name,roles in scores.items():
        result["points"][name]={}
        for role,values in roles.items():
            matrix,counts=parent.metrics.group_metrics(truth,values)
            mr=previous.save_array(root/(name+"_"+role+".npy"),matrix,root)
            cp=root/(name+"_"+role+"_counts.json"); data.write_json(cp,counts)
            result["points"][name][role]={"matrix":mr,"counts":data.record(cp,root)}
    if sum(map(len,result["points"].values()))!=28:
        raise ValueError("Incomplete 28 metric sets")
    result["status"]="ALL_28_RELATION_METRIC_COUNT_SETS_SAVED"
    data.write_json(root/"collected.json",result)
    return result


def read_roles(root: Path, record: dict) -> dict:
    out={}
    if set(record)!=set(ROLES): raise ValueError("Incomplete metric roles")
    for role,files in record.items():
        out[role]=evaluation.read_matrix(root,files["matrix"])
        evaluation.read_counts(root,files["counts"])
    for role in ("stage-cal","first-cal"):
        if not np.array_equal(out[role][:,evaluation.RANK_COLUMNS],out["raw"][:,evaluation.RANK_COLUMNS]):
            raise ValueError("Calibration changed rank/curve metrics")
    return out


def comparisons(candidate: dict, reference: dict, domains: list) -> dict:
    # Local logical views: never change old artifacts or share candidate first states.
    draws=evaluation.bootstrap_draws()
    fields={}; summaries={}
    for arm,arrays in (("relation",candidate),("logit0.1",reference)):
        fields[arm]={role:{ep:evaluation.endpoint_fields(arrays,domains,"er",role,ep)
                          for ep in evaluation.ENDPOINTS} for role in (*ROLES,"primary")}
        summaries[arm]={role:{ep:evaluation.summarize_field(f,draws) for ep,f in fs.items()}
                        for role,fs in fields[arm].items()}
    delta={ep:evaluation.summarize_field(fields["relation"]["primary"][ep]-fields["logit0.1"]["primary"][ep],draws)
           for ep in evaluation.ENDPOINTS}
    checks={"O_map_positive":delta["O"]["map"]["mean"]>0}
    checks.update({f"{ep}_map_non_decrease":delta[ep]["map"]["mean"]>=0 for ep in ("N","Z")})
    checks.update({f"{ep}_AP_non_decrease":delta[ep]["average_precision"]["mean"]>=0 for ep in ("O","N","Z")})
    return {"endpoints":summaries,"delta":delta,"continuation_checks":checks,
            "worth_matched_replay":all(checks.values()),
            "old_map_conditional_positive":delta["O"]["map"]["conditional_95pct_interval"][0]>0,
            "automatic_followup":False,"scope":"Developed valid; single seed, three orders; configuration effect, not mechanism attribution"}


def finalize(root: Path) -> dict:
    p=policy(); collected=data.read_json(root/"collected.json")
    if (collected["status"]!="ALL_28_RELATION_METRIC_COUNT_SETS_SAVED" or collected["source_files"]!=sources()
            or set(collected["points"])!={"initial",*expected_points()}
            or collected["metric_columns"]!=list(parent.metrics.COLUMNS)):
        raise ValueError("Complete source-matched saved collection required")
    baseline=data.ROOT/p["reference"]["linux_root"]
    reference_collections=[data.read_json(data.verify(baseline/r["path"],r)) for r in p["reference"]["collections"]]
    for col in reference_collections:
        if any(col[k]!=collected[k] for k in ("group_ids","domains","metric_columns")):
            raise ValueError("Reference group/domain/metric pairing differs")
    candidate={}; reference={}
    for order in parent.ORDERS:
        for stage in (1,2,3):
            logical=evaluation.point_name(order,"er",stage)
            candidate[logical]=read_roles(root,collected["points"][point_name(order,stage)])
            if stage==1:
                reference[logical]=read_roles(baseline/"reference",reference_collections[1]["points"][order+"_shared"])
            else:
                reference[logical]=read_roles(baseline,reference_collections[0]["points"][f"{order}_logit_tenth_stage{stage}"])
    # Initial scores remain a separate diagnostic, not a trained shared first stage.
    initial=collected["points"]["initial"]["raw"]
    evaluation.read_matrix(root,initial["matrix"]); evaluation.read_counts(root,initial["counts"])
    result=comparisons(candidate,reference,collected["domains"])
    result["absolute_stage_results"]={}
    rows=evaluation.domain_rows(collected["domains"])
    for name,roles in collected["points"].items():
        result["absolute_stage_results"][name]={}
        for role,rec in roles.items():
            matrix=evaluation.read_matrix(root,rec["matrix"])
            counts=evaluation.read_counts(root,rec["counts"])
            result["absolute_stage_results"][name][role]={"macro_all":dict(zip(parent.metrics.COLUMNS,matrix.mean(0).tolist())),
                "macro_by_domain":{d:dict(zip(parent.metrics.COLUMNS,matrix[r].mean(0).tolist())) for d,r in zip("ABC",rows)},
                "pooled_fixed_half_classification":base.fixed_classification(counts,collected["domains"])}
    result.update(status="COMPLETE_RELATION_FIXED_POINT",collected=data.record(root/"collected.json",root),
                  reference=p["reference"],source_files=sources())
    evaluation.write_once(root/"evaluation.json",data.json_bytes(result))
    return result


def execute(job: Path, gate_path: Path) -> dict:
    import torch
    p=policy(); gate=data.read_json(gate_path)
    if (gate.get("status")!="APPROVED_RELATION_PILOT" or gate.get("source_files")!=sources()
            or gate.get("job")!=job.relative_to(data.ROOT).as_posix()
            or gate.get("runtime")!=p["runtime"] or gate.get("supervision")!=p["supervision"]):
        raise ValueError("Matching reviewed formal gate required")
    for key in ("native","integration_cpu"):
        r=gate[key]; evidence=data.read_json(data.verify(data.ROOT/r["path"],r))
        if evidence["status"]!="PASS_HANDWRITTEN_ONLY": raise ValueError("Missing passed evidence")
        if key=="native" and evidence["scientific_sources"]["scripts/step28_relation_memory.py"]!=data.sha256(Path(method.__file__)):
            raise ValueError("Native core source differs")
        if key=="integration_cpu" and evidence["source_files"]!=sources():
            raise ValueError("Integration verification source differs")
    if gate["review_disposition"]!="NO_OPEN_BLOCKERS": raise ValueError("External/main review incomplete")
    if job.exists() or not job.is_relative_to(data.ROOT/"reports"):
        raise ValueError("New independent reports directory required")
    resources=admission.preflight()
    for record in p["reference"]["collections"]:
        data.verify(data.ROOT/p["reference"]["linux_root"]/record["path"],record)
    if shutil.disk_usage(data.ROOT).free<26*2**30: raise ValueError("Insufficient disk")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    job.mkdir(parents=True)
    budget=Budget(job,p); stopped=threading.Event()
    data.write_json(job/"access.json",dict(train=0,valid=0,heldout=0,owners=0))
    data.write_json(job/"execution.json",{"gate":data.record(gate_path,data.ROOT),"sources":sources(),"resources":resources})
    def watchdog():
        while not stopped.wait(1):
            try: budget.check()
            except Exception as exc:
                data.write_json(job/"failure.json",{"status":"BUDGET_STOP_NO_RETRY","error":str(exc)})
                os._exit(2)
    threading.Thread(target=watchdog,daemon=True).start()
    try:
        manifest,partition,valid=train(job,p,budget)
        scores,verified_partition=blind_gate(job/"run",manifest)
        if verified_partition!=partition: raise ValueError("Partition changed")
        data.write_json(job/"before_valid.json",{"status":"PASS_COMPLETE_BLIND_GATE","points":9,
            "manifest":data.record(job/"run/manifest.json",job),"access":data.read_json(job/"access.json")})
        labelled=previous.parse_once(job,valid,method.config(),"development")
        collect(job/"evaluation",scores,labelled,partition)
        del labelled,scores,valid
        result=finalize(job/"evaluation")
        budget.check(1)
        completion={"status":result["status"],"physical_updates":2592,"access":data.read_json(job/"access.json"),
                    "budget":budget.state(),"peak_rss_bytes":budget.peak_rss,
                    "evaluation":data.record(job/"evaluation/evaluation.json",job)}
        data.write_json(job/"completion.json",completion)
        return completion
    except BaseException:
        data.write_json(job/"failure.json",{"status":"FAILED_NO_AUTOMATIC_RETRY","traceback":traceback.format_exc(),
                                            "access":data.read_json(job/"access.json")})
        raise
    finally: stopped.set()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("execute","finalize"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--gate",type=Path)
    args=parser.parse_args()
    if os.name!="posix": parser.error("Existing Linux py310 only")
    if args.action=="execute":
        if args.gate is None: parser.error("Reviewed gate required")
        execute(args.out.resolve(),args.gate.resolve())
    else: finalize(args.out.resolve())
