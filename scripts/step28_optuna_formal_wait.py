"""One formal-only dispatch under the approved 272-hour gate; no retry."""
from __future__ import annotations

import datetime
import hashlib
import json
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
    free=int(subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True,timeout=15).strip())
    apps=subprocess.check_output(["nvidia-smi","-i","0","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True,timeout=15).split()
    ancestors={os.getpid(),*(p.pid for p in psutil.Process().parents())}
    owner=psutil.Process().username()
    jobs=[p.info for p in psutil.process_iter(["pid","username","cmdline"])
        if p.pid not in ancestors and p.info["username"]==owner
        and any("step28" in x for x in (p.info["cmdline"] or []))]
    host=psutil.virtual_memory().available;disk=shutil.disk_usage(workspace).free
    return dict(gpu_free_mib=free,compute_pids=apps,project_jobs=jobs,host_free_bytes=host,disk_free_bytes=disk,
        ready=free>=9216 and host>=72*2**30 and disk>=64*2**30 and not jobs)


def first_update(job):
    for log in job.glob("*.console.txt"):
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.startswith("{"):continue
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if row.get("event")=="updates" and row.get("completed",0)>0:
                return dict(log=str(log),event=row)
    return None


def main(workspace,gate_path,base):
    active=Path(__file__).resolve();gate=json.loads(gate_path.read_text());job=workspace/gate["job"]
    if active!=base/"active_formal_wait.py" or job.exists():raise ValueError("Fresh owned formal dispatch required")
    source=workspace/"scripts/step28_optuna_formal_wait.py"
    if active.read_bytes()!=source.read_bytes():raise ValueError("Active listener differs from bound source")
    dependencies=[gate[k] for k in ("cpu","native_reuse","time_qualification","execution_authorization","external_review","primary_disposition")]
    if "recovery" in gate:dependencies.append(gate["recovery"])
    verify(workspace,gate["source_files"]+dependencies)
    with (base/"formal_dispatch.json").open("x",encoding="utf-8") as stream:
        json.dump(dict(status="CLAIMED_ONCE",pid=os.getpid(),at=now(),job=str(job),gate=str(gate_path)),stream)
    status=base/"formal_wait_status.json"
    while True:
        state=resources(workspace);write(status,dict(status="WAITING_RESOURCES",at=now(),**state))
        if state["ready"]:break
        time.sleep(30)
    verify(workspace,gate["source_files"]+dependencies)
    env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
        "CUDA_VISIBLE_DEVICES":"0","TOKENIZERS_PARALLELISM":"false","CUBLAS_WORKSPACE_CONFIG":":4096:8","PYTHONDONTWRITEBYTECODE":"1"}
    env.pop("RECORD_REPLAY_STARTED_EPOCH",None)
    command=["taskset","-c","0",sys.executable,"-B",str(workspace/"scripts/step28_optuna_search.py"),"--job",str(job),"--gate",str(gate_path)]
    with (base/"formal.console.txt").open("x") as log:
        child=subprocess.Popen(command,cwd=workspace,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(base/"formal_start.json",dict(at=now(),child_pid=child.pid,resources=state,command=command))
    write(status,dict(status="FORMAL_CONTROLLER_STARTED",at=now(),child_pid=child.pid))
    removed=False
    while child.poll() is None:
        update=first_update(job) if not removed else None
        if update:
            active.unlink();removed=True
            write(status,dict(status="ACTUAL_FORMAL_UPDATE_SELF_REMOVED",at=now(),child_pid=child.pid,**update))
        time.sleep(5)
    write(base/"formal_exit.json",dict(at=now(),exit_code=child.returncode,active_removed=removed,automatic_retry=False))
    return child.returncode


if __name__=="__main__":
    workspace,gate,base=(Path(p).resolve() for p in sys.argv[1:])
    try:raise SystemExit(main(workspace,gate,base))
    except Exception:
        import traceback
        write(base/"formal_wait_failure.json",dict(at=now(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
