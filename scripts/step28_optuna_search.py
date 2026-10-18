"""Sequential 36+9 production dispatch. Explicit reviewed permission is mandatory."""
from __future__ import annotations

import time
MODULE_STARTED=time.monotonic()

import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import step28_optuna_run as run
from step28_optuna import ARMS, TrialSpec
from step28_optuna_control import Ledger, Search, S0, projected_gate, replication_specs, joint_opening

data=run.data


def validate_gate(path):
    gate=data.read_json(path)
    current=run.sources()
    if (gate.get("kind")!="FORMAL" or gate.get("status")!="APPROVED_NO_OPEN_BLOCKERS"
        or gate.get("source_files")!=current or gate.get("policy")!=data.record(run.POLICY,data.ROOT)):
        raise ValueError("Current reviewed formal gate required")
    def bound(key):
        return data.read_json(data.verify(data.ROOT/gate[key]["path"],gate[key]))
    permission=bound("execution_authorization")
    if (permission.get("status")!="APPROVED" or permission.get("stages")!=["FORMAL"]
        or permission.get("source_files")!=current or not permission.get("user_quote")
        or permission.get("search_weight_lifecycle") is not True or permission.get("job")!=gate.get("job")):
        raise ValueError("Specific 45-run data/budget/lifecycle permission missing")
    for key in ("external_review","primary_disposition"):
        data.verify(data.ROOT/gate[key]["path"],gate[key])
    cpu,native=bound("cpu"),bound("native")
    if (cpu.get("status")!="PASS_HANDWRITTEN_ONLY" or cpu.get("source_files")!=current
        or cpu.get("cumulative_charged_seconds",math.inf)>300 or cpu.get("unresolved_checks")!=[]
        or set(cpu.get("qualified_tests",[]))!=set(run.CPU_TESTS)
        or native.get("status")!="PASS_HANDWRITTEN_ONLY" or native.get("source_files")!=current
        or native.get("cumulative_charged_seconds",math.inf)>1800 or set(native.get("arms",{}))!=set(ARMS)):
        raise ValueError("Current bounded CPU/native evidence missing")
    for arm,row in native["arms"].items():
        if (row.get("actual_updates")!=6 or row.get("charged_seconds",math.inf)>600
            or not all(row.get(k) is True for k in ("current_history_lambda_effect","actual_lr_effect",
                "next_update_bitwise_equal","inference_restore_equal","maximum_shape_verified"))):
            raise ValueError("Native actual effects missing: "+arm)
    projection=bound("time_qualification")
    if projection.get("status")!="NEW_9000_SECOND_QUALIFICATION" or projection.get("source_files")!=current:
        raise ValueError("Old mathematical PASS is not new time qualification")
    for record in projection["measurement_evidence"]:
        data.verify(data.ROOT/record["path"],record)
    if not projection["measurement_evidence"]:
        raise ValueError("Actual conservative cost evidence required")
    expected={arm:{key:run.single.projection(native["arms"][arm]["measurements"])["projected_total_seconds"]
                   for key in ("search","replication")} for arm in ARMS}
    if projection.get("cost_model")!="legacy_1.25_allowance1800" or projection["arms"]!=expected:
        raise ValueError("Projection must recompute the retained cost model; a revised model needs a reviewed amendment")
    for key in ("control","evaluation","analysis"):
        bound=projection["shared_evidence"][key]
        data.verify(data.ROOT/bound["evidence"]["path"],bound["evidence"])
        if bound["conservative_seconds"]!=projection["shared"][key]:raise ValueError("Unbound shared-stage projection")
    projected_gate(projection)
    run.policy()
    return gate,cpu,native


def fixed_reference():
    base=data.ROOT/"reports/seller_alias_continual"
    summary=base/"20261018/positive_ranking_result/analysis/summary.json"
    if summary.stat().st_size!=92383 or data.sha256(summary)!="6ad47620c634f35275621ff006a5838a55fbe5650884fa4d5d9c3de4814365d1":
        raise ValueError("Fixed S0 summary differs")
    refs={d:data.read_json(summary)["absolute"]["B_S"]["raw"][d]["map"] for d in "AB"}
    root=base/"20261015/balanced_pooling_result/job"
    expected={"evaluation/B_S_raw_metrics.json":(12880,"9caa6d26f5f843563140dbacde20599e8a912dc603043cd32d206eb27e83a4f1"),
        "evaluation/collected.json":(13818,"6e9a04882abe9c7d9cc10ddef9b5e9e179bc80f4ae866553391f9d041cdc7882"),
        "run/partition.json":(9962,"a828b9cd7b2acade501a8bed7209282ae7f916c4fa2944426f49f755fcf01c51")}
    for name,(size,sha) in expected.items():
        p=root/name
        if p.stat().st_size!=size or data.sha256(p)!=sha:raise ValueError("Fixed S0 group source differs")
    col=data.read_json(root/"evaluation/collected.json");split=data.read_json(root/"run/partition.json")
    if col["group_ids"]!=[r["group_uid"] for r in split["development"]] or col["domains"]!=[r["domain"] for r in split["development"]]:
        raise ValueError("S0 ordered group alignment differs")
    import numpy as np
    values=np.asarray(data.read_json(root/"evaluation/B_S_raw_metrics.json")["values"],np.float64)
    reduced={d:float(values[np.asarray(col["domains"])==d,12].mean(dtype=np.float64)) for d in "AB"}
    if refs!=S0 or reduced!=refs:raise ValueError("Full precision S0 reduction differs; no tolerance substitution")
    return refs


@contextmanager
def charged(ledger,kind,**kwargs):
    def expired(signum,frame):raise TimeoutError("Hard campaign interval deadline")
    with ledger.interval(kind,**kwargs) as seconds:
        prior=signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,seconds)
        try:yield seconds
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)


def launch(command,log,seconds,env):
    """Bound the owned process group including import, exit and worker release."""
    import psutil
    started=time.monotonic();deadline=started+seconds
    with log.open("x",encoding="utf-8") as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            while True:
                if time.monotonic()>=deadline:raise TimeoutError("Child hard deadline")
                code=child.poll()
                members=[]
                for p in psutil.process_iter(["pid"]):
                    try:
                        if os.getpgid(p.pid)==child.pid:members.append(p.pid)
                    except (ProcessLookupError,PermissionError):pass
                if code is not None and not members:break
                time.sleep(.1)
            if code!=0:raise RuntimeError(f"Owned child failed with exit {code}")
            return dict(child_pid=child.pid,child_exit_code=code,resources_released=True,child_seconds=time.monotonic()-started)
        except BaseException:
            # Only this freshly launched process group's members can be terminated.
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
            raise


def access_total(job):
    result=dict(train=0,valid=0,heldout=0,owners=0)
    for path in job.glob("*/access.json"):
        row=data.read_json(path)
        for k in result:result[k]+=row[k]
    if result["train"]>45 or result["valid"]>37 or result["heldout"] or result["owners"]:
        raise ValueError("Campaign access exceeded; preserve attempts")
    return result


def clean_search_weights(job):
    manifest=data.read_json(job/"run/manifest.json")
    paths=[job/"run"/p["model"]["path"] for p in manifest["points"].values()]
    if len(paths)!=2 or any(p.parent!=(job/"run/models") for p in paths):raise ValueError("Unexpected weight lifecycle targets")
    receipt=[data.record(p,job) for p in paths]
    data.write_json(job/"search_weight_retirement.json",dict(status="VERIFIED_BEFORE_DELETE",files=receipt))
    for path in paths:path.unlink()
    data.write_json(job/"search_weight_retirement.json",dict(status="DELETED_AFTER_COMPLETE_RESULT",files=receipt))


def execute(job,gate_path):
    ledger=Ledger();search=None
    ledger.used["control"]=time.monotonic()-MODULE_STARTED
    if job.exists() or not job.is_relative_to(data.ROOT/"reports"):
        raise ValueError("New independent campaign output required")
    job.mkdir(parents=True)
    try:
        with charged(ledger,"control"):
            gate,cpu,native=validate_gate(gate_path)
            if gate["job"]!=job.relative_to(data.ROOT).as_posix():raise ValueError("Approved campaign path differs")
            ledger.qualification=cpu["cumulative_charged_seconds"]+native["cumulative_charged_seconds"]
            search=Search("sqlite:///"+str(job/"studies.sqlite3"),fixed_reference())
            data.write_json(job/"startup.json",dict(source_files=run.sources(),gate=data.record(gate_path,data.ROOT)))
        def trial(spec):
            path=job/spec.run_id
            with charged(ledger,"control"):
                spec_path=job/(spec.run_id+".json");data.write_json(spec_path,spec.identity())
                access_total(job)
            with charged(ledger,spec.phase,arm=spec.arm,run_id=spec.run_id) as seconds:
                env={**os.environ,"RECORD_REPLAY_STARTED_EPOCH":str(time.time())}
                receipt=launch([sys.executable,str(data.ROOT/"scripts/step28_optuna_run.py"),"--job",str(path),
                    "--spec",str(spec_path),"--gate",str(gate_path)],job/(spec.run_id+".console.txt"),seconds,env)
                result=data.read_json(path/"completion.json")
                if result.get("source_files")!=run.sources() or (path/"failure.json").exists():raise ValueError("Invalid child completion")
                result.update(receipt,completion_identity=data.record(path/"completion.json",data.ROOT))
            return result
        for number in range(12):
            for arm in Search.order(number):
                with charged(ledger,"control"):spec=search.ask(arm,number)
                result=trial(spec)
                with charged(ledger,"control"):
                    search.tell(result)
                    clean_search_weights(job/spec.run_id)
                    data.write_json(job/"search.json",search.rows)
                    data.write_json(job/"ledger.json",ledger.snapshot())
        with charged(ledger,"control"):
            selected=search.selected();data.write_json(job/"selected.json",selected)
            specs=replication_specs(selected);records=[]
        for spec in specs:
            result=trial(spec)
            with charged(ledger,"control"):
                records.append(result);data.write_json(job/"replications.json",records)
        with charged(ledger,"evaluation") as seconds:
            joint_opening(records,specs)
            launch([sys.executable,str(data.ROOT/"scripts/step28_optuna_run.py"),"--job",str(job),"--gate",str(gate_path),"--joint"],
                job/"joint.console.txt",seconds,{**os.environ,"CUDA_VISIBLE_DEVICES":""})
        with charged(ledger,"analysis") as seconds:
            launch([sys.executable,str(data.ROOT/"scripts/step28_optuna_verify.py"),"--analyze",str(job)],
                job/"analysis.console.txt",seconds,{**os.environ,"CUDA_VISIBLE_DEVICES":""})
        with charged(ledger,"control"):
            access=access_total(job)
            if access!=dict(train=45,valid=37,heldout=0,owners=0):raise ValueError("Incomplete final access coverage")
            data.write_json(job/"completion.json",dict(status="COMPLETE_36_SEARCH_9_REPLICATIONS_AND_ANALYSIS",access=access,source_files=run.sources()))
    except BaseException:
        ledger.failed=True
        if search is not None:search.fail(traceback.format_exc())
        data.write_json(job/"failure.json",dict(status="STOP_NO_RETRY",traceback=traceback.format_exc(),access=access_total(job)))
        raise
    finally:data.write_json(job/"ledger.json",ledger.snapshot())


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--job",type=Path,required=True);parser.add_argument("--gate",type=Path,required=True)
    args=parser.parse_args()
    if os.name!="posix":parser.error("Linux only")
    execute(args.job.resolve(),args.gate.resolve())
