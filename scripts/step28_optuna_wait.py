"""One authorized native attempt; no data access, retry, or implicit formal gate."""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def now():
    return datetime.datetime.now().astimezone().isoformat()


def verify(root,rows):
    for row in rows:
        path=root/row["path"]
        if path.stat().st_size!=row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]:
            raise ValueError("Approved source changed: "+row["path"])


def resources(workspace):
    gpu=subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True,timeout=15)
    apps=subprocess.check_output(["nvidia-smi","-i","0","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True,timeout=15)
    ancestors={os.getpid(),*(p.pid for p in psutil.Process().parents())}
    jobs=[p.info for p in psutil.process_iter(["pid","username","cmdline"])
        if p.pid not in ancestors and any("step28" in x for x in (p.info["cmdline"] or []))]
    free=int(gpu.strip());host=psutil.virtual_memory().available;disk=shutil.disk_usage(workspace).free
    return dict(gpu_free_mib=free,compute_pids=apps.split(),project_jobs=jobs,host_free_bytes=host,disk_free_bytes=disk,
        ready=free>=30720 and host>=72*2**30 and disk>=10*2**30 and not apps.strip() and not jobs)


def main(workspace,gate_path,output):
    active=Path(__file__).resolve();base=output.parent
    gate=json.loads(gate_path.read_text())
    if active!=base/"active_wait.py" or output.exists():raise ValueError("Fresh owned native dispatch required")
    verify(workspace,gate["source_files"])
    verify(workspace,gate["dispatch_sources"])
    verify(workspace,[gate[k] for k in ("cpu","primary_disposition","external_review","execution_authorization")])
    with (base/"native_dispatch.json").open("x",encoding="utf-8") as stream:
        json.dump(dict(status="CLAIMED_ONCE",pid=os.getpid(),at=now(),output=str(output),gate=str(gate_path)),stream)
    status=base/"native_wait_status.json"
    while True:
        state=resources(workspace)
        write(status,dict(status="WAITING_RESOURCES",at=now(),**state))
        if state["ready"]:break
        time.sleep(30)
    verify(workspace,gate["source_files"]);verify(workspace,gate["dispatch_sources"])
    command=["bash",str(workspace/"scripts/run_step28_optuna_native_linux_20261019.sh"),str(workspace),str(gate_path),str(output)]
    with (base/"native_launcher.log").open("x") as log:
        child=subprocess.Popen(command,cwd=workspace,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(status,dict(status="NATIVE_STARTED_ONCE",at=now(),child_pid=child.pid,resources=state))
    removed=False
    while child.poll() is None:
        result=output/"result.json"
        if result.exists() and not removed:
            try:value=json.loads(result.read_text())
            except json.JSONDecodeError:value={}
            if value.get("actual_updates",0)>0 or any(a.get("actual_updates",0)>0 for a in value.get("arms",{}).values()):
                active.unlink();removed=True
                write(status,dict(status="ACTUAL_NATIVE_UPDATE_SELF_REMOVED",at=now(),child_pid=child.pid))
        time.sleep(5)
    if child.returncode:
        write(status,dict(status="NATIVE_FAILED_NO_RETRY",at=now(),exit_code=child.returncode,active_removed=removed,formal_started=False))
        return child.returncode
    value=json.loads((output/"result.json").read_text())
    wrapper=dict(line.split("=",1) for line in output.with_suffix(".wrapper.txt").read_text().splitlines())
    if (value["status"]!="PASS_HANDWRITTEN_ONLY" or value["source_files"]!=gate["source_files"]
            or int(wrapper["charged_seconds"])>1800 or set(value["arms"])!={"S","S_pos","H_pos"}):
        raise ValueError("Native completion/source/budget differs")
    for arm,row in value["arms"].items():
        if row["actual_updates"]!=6 or len(row["updates"])!=6 or row["charged_seconds"]>600 or not all(row[k] is True for k in
                ("current_history_lambda_effect","actual_lr_effect","next_update_bitwise_equal","inference_restore_equal","maximum_shape_verified")):
            raise ValueError("Native arm evidence incomplete: "+arm)
    projected={a:r["old_cost_projection"]["projected_total_seconds"] for a,r in value["arms"].items()}
    if any(not math.isfinite(p) or not 0<p<=9000 for p in projected.values()):
        write(status,dict(status="NATIVE_NUMERICAL_PASS_NEW_TIME_GATE_FAILED",at=now(),projected_seconds=projected,
            single_run_limit=9000,formal_started=False,automatic_retry=False,active_removed=removed))
        return 0
    formal=gate_path.parent/"formal_gate.json"
    if not formal.exists():
        write(status,dict(status="NATIVE_NUMERICAL_PASS_FORMAL_TIME_EVIDENCE_INCOMPLETE",at=now(),projected_seconds=projected,
            missing="Current complete time qualification, including measured C300/E2400/A600 capacity, and its bound formal gate",
            formal_started=False,automatic_retry=False,active_removed=removed))
        return 0
    # The production entry independently validates every CPU/native/time/data/source gate.
    # No default or fabricated formal gate is created by the listener.
    formal_value=json.loads(formal.read_text());formal_job=workspace/formal_value["job"]
    with (base/"formal_dispatch.json").open("x") as stream:
        json.dump(dict(at=now(),gate=str(formal),job=str(formal_job)),stream)
    env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
        "CUDA_VISIBLE_DEVICES":"0","TOKENIZERS_PARALLELISM":"false","CUBLAS_WORKSPACE_CONFIG":":4096:8","PYTHONDONTWRITEBYTECODE":"1"}
    with (base/"formal.console.txt").open("x") as log:
        formal_child=subprocess.Popen(["taskset","-c","0",sys.executable,"-B",str(workspace/"scripts/step28_optuna_search.py"),
            "--job",str(formal_job),"--gate",str(formal)],cwd=workspace,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(status,dict(status="FORMAL_ENTRY_LAUNCHED_AFTER_NATIVE_AND_TIME_GATES",at=now(),child_pid=formal_child.pid,automatic_retry=False))
    return 0


if __name__=="__main__":
    workspace,gate,output=(Path(p).resolve() for p in sys.argv[1:])
    try:raise SystemExit(main(workspace,gate,output))
    except Exception:
        import traceback
        write(output.parent/"native_wait_failure.json",dict(at=now(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
