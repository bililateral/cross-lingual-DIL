"""Specific pre-data resource recovery; no retry after model/data admission."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3

import psutil

from step28_optuna import ARMS, TrialSpec
from step28_optuna_formal_wait import resources


def formal_preflight(workspace):
    process = psutil.Process()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or len(process.cpu_affinity()) != 1:
        raise ValueError("Physical GPU0 and exactly one CPU affinity required")
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise ValueError("Single-thread environment must precede imports")
    if os.environ.get("TOKENIZERS_PARALLELISM") != "false":
        raise ValueError("Tokenizer parallelism must be disabled")
    state = resources(workspace)
    if not state["ready"]:
        # execute_trial calls this before mkdir, CUDA context/model or data reads.
        print(json.dumps(dict(status="RESOURCE_DEFERRED_BEFORE_DATA",resources=state,
                              model_loads=0,train=0,valid=0,updates=0)),flush=True)
        raise SystemExit(75)
    return state


def validate_deferral(log, path):
    if path.exists():
        raise ValueError("Resource retry forbidden after run directory/data admission")
    rows=[]
    for line in log.read_text().splitlines():
        if line.startswith("{"):
            try: rows.append(json.loads(line))
            except ValueError: pass
    if (len(rows)!=1 or rows[0].get("status")!="RESOURCE_DEFERRED_BEFORE_DATA"
        or rows[0].get("resources",{}).get("ready") is not False
        or any(rows[0].get(k)!=0 for k in ("model_loads","train","valid","updates"))):
        raise ValueError("Missing exact pre-data resource deferral receipt")
    return rows[0]


def restore_prior(search, ledger, job, approval, data):
    """Reconstruct the three real anchors and the one already-drawn candidate."""
    if approval.get("status")!="APPROVED_SPECIFIC_PRE_DATA_RECOVERY":
        raise ValueError("Specific resource recovery authorization required")
    root=Path(approval["prior_root"])
    prior=root/"workspace/reports/job"
    files=approval["files"]
    for row in files:
        data.verify(root/row["path"],row)
    actual={p.relative_to(root).as_posix() for p in prior.rglob("*") if p.is_file()}
    recorded={r["path"] for r in files if r["path"].startswith("workspace/reports/job/")}
    if actual!=recorded:
        raise ValueError("Prior output set differs; do not omit failed attempts")
    failure=data.read_json(prior/"failure.json")
    if failure.get("status")!="STOP_NO_RETRY" or failure.get("access")!=dict(train=3,valid=3,heldout=0,owners=0):
        raise ValueError("Only the observed three-anchor resource stop is recoverable")
    failed=TrialSpec.from_dict(data.read_json(prior/"search_01_S_pos_s0.json"))
    if (failed.arm,failed.number,failed.phase)!=("S_pos",1,"search") or (prior/failed.run_id).exists():
        raise ValueError("Failed candidate already reached data admission")
    text=(prior/(failed.run_id+".console.txt")).read_text()
    if "Resource gate failed:" not in text or "preflight" not in text:
        raise ValueError("Not the approved preflight failure")
    cleanup=json.loads(text.splitlines()[-1])
    if cleanup.get("status")!="FAILED_OWN_GROUP_CLEANUP" or cleanup.get("resources_released") is not True or cleanup.get("residual_pids")!=[]:
        raise ValueError("Failed process group not released")
    with sqlite3.connect((prior/"studies.sqlite3").as_uri()+"?mode=ro",uri=True) as db:
        states=db.execute("SELECT s.study_name,t.number,t.state FROM trials t JOIN studies s ON s.study_id=t.study_id ORDER BY t.trial_id").fetchall()
    if states!=[("S",0,"COMPLETE"),("S_pos",0,"COMPLETE"),("H_pos",0,"COMPLETE"),("S_pos",1,"FAIL")]:
        raise ValueError("Prior Optuna states differ")
    lines=(root/"execution/formal.console.txt").read_text().splitlines()
    finals=[json.loads(l[len("FINAL_ACCOUNTING "):]) for l in lines if l.startswith("FINAL_ACCOUNTING ")]
    if len(finals)!=1:raise ValueError("Original final accounting required")
    old=finals[0]
    runs=[r for r in old["intervals"] if r["kind"]=="search"]
    if ([r["run_id"] for r in runs]!=[TrialSpec(a,"search",0).run_id for a in ARMS]+[failed.run_id]
        or [r["error"] for r in runs]!=[None,None,None,"RuntimeError"]
        or old["used"]["replication"] or old["used"]["evaluation"] or old["used"]["analysis"]
        or old["qualification_seconds"]!=1000 or old["failed"] is not True):
        raise ValueError("Prior accounting scope differs")
    for kind,value in old["used"].items():
        if abs(sum(r["seconds"] for r in old["intervals"] if r["kind"]==kind)-value)>1e-8:
            raise ValueError("Prior intervals do not reproduce charged time")
    for arm in ARMS:
        if abs(sum(r["seconds"] for r in runs if r["arm"]==arm)-old["by_arm"]["search"][arm])>1e-8:
            raise ValueError("Prior arm charges differ")
    if "pre_dispatch_failure" in approval:
        stopped=approval["pre_dispatch_failure"];stopped_root=Path(stopped["root"])
        for row in stopped["files"]:data.verify(stopped_root/row["path"],row)
        before=data.read_json(stopped_root/"workspace/reports/job/failure.json")
        if before.get("access")!=dict(train=0,valid=0,heldout=0,owners=0) or (stopped_root/"workspace/reports/job/studies.sqlite3").exists():
            raise ValueError("Only the source-order failure before dispatch may carry control cost")
        records=[json.loads(l[len("FINAL_ACCOUNTING "):]) for l in (stopped_root/"execution/formal.console.txt").read_text().splitlines() if l.startswith("FINAL_ACCOUNTING ")]
        if len(records)!=1:raise ValueError("Pre-dispatch accounting missing")
        charged=records[0]
        if (charged["qualification_seconds"]!=0 or charged["failed"] is not True
            or any(v for k,v in charged["used"].items() if k!="control")
            or any(r["kind"]!="control" for r in charged["intervals"])
            or abs(sum(r["seconds"] for r in charged["intervals"])-charged["used"]["control"])>1e-8):
            raise ValueError("Pre-dispatch attempt exceeded control-only scope")
        ledger.used["control"]+=charged["used"]["control"]
        ledger.intervals+=charged["intervals"]
    # The new controller has already charged its imports before this active
    # control interval. Inherit old time without discarding that new work.
    ledger.used={k:old["used"][k]+v for k,v in ledger.used.items()}
    ledger.arm_used={k:v.copy() for k,v in old["by_arm"].items()}
    ledger.intervals=list(old["intervals"])+ledger.intervals
    ledger.calendar_wait_seconds+=old.get("calendar_wait_seconds",0.)
    if ledger.active is not None:
        kind,started,allowed=ledger.active
        if kind!="control":raise ValueError("Recovery must be charged to control")
        allowed=min(allowed,ledger.remaining("control"))
        ledger.active=(kind,started,allowed)
        ledger.check()
        signal.setitimer(signal.ITIMER_REAL,max(.001,allowed-(ledger.clock()-started)))
    # Original FAIL and SQLite are immutable. The new study reconstructs the
    # same logical candidate sequence from real completed evidence, not zeros.
    for arm in ARMS:
        spec=search.ask(arm,0);source=prior/spec.run_id
        completion=data.read_json(source/"completion.json")
        if (completion.get("source_files")!=approval["prior_sources"]
            or (source/"failure.json").exists()
            or data.read_json(source/"before_valid.json").get("status")!="ALL_BLIND_GATES_PASS"):
            raise ValueError("Prior completed source/blind evidence differs")
        shutil.copytree(source,job/spec.run_id)
        search.tell({**completion, "child_exit_code":0,"resources_released":True,
                     "completion_identity":data.record(source/"completion.json",root)})
    pending=search.ask("S_pos",1)
    if pending.identity()!=failed.identity():
        raise ValueError("Replayed sampler did not reproduce the exact failed parameters")
    # The new controller is allowed to proceed; the historical error interval
    # remains charged and visible. Qualification is supplied by the new CPU gate.
    data.write_json(job/"recovery.json",dict(status="RESTORED_THREE_REAL_ANCHORS",
        approval=approval,prior_final_accounting=old,pending=failed.identity(),
        original_failure_preserved=True,completed_runs_retrained=0))
    data.write_json(job/"search.json",search.rows)
    return pending
