"""Prepare, arm and observe the single authorized Chinese start on Linux."""
import time
ENTRY = time.monotonic()

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import psutil

START = Path(__file__).resolve().parent
ROOT, WS = START.parent, START.parent/"workspace"
PROJECT = ROOT.parents[3]
EN = ROOT.parent/"vendorlink_market_training/execution"
GATE, CAMPAIGN, WAIT = ROOT/"formal_gate.json", ROOT/"campaign", ROOT/"wait"
sys.path.insert(0,str(WS/"scripts"))
import step28_history_benchmark as b
import step28_benchmark_wait as w
from step30_evolution_io import PYTHON, read, record, sha, verify, write

GIB = 2**30
SOURCE_GATE = "7607f7cf20834a5fa4f9a34ebda0d92543537d33b87c704abf91d812b0555b1f"
SOURCE_LAUNCH = "766901a7d37b98064ba16f1d21e23845d297227a92ba00bb29f87a78cd01daff"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def freeze(p, value):
    if p.exists():
        if read(p) != value:
            raise ValueError("Existing frozen file differs: "+str(p))
    else:
        with p.open("x",encoding="utf-8") as out:
            json.dump(value,out,ensure_ascii=False,indent=2,allow_nan=False)
            out.write("\n")


def inputs(install=False):
    return w.training_inputs(WS,PROJECT,install)


def accepted_review(document, manifest_sha, qualification_sha, fix_sha):
    first = document.splitlines()[0] if document else ""
    if not (first.startswith("<!-- ") and first.endswith(" -->")):
        raise ValueError("Missing structured independent review disposition")
    signed = json.loads(first[5:-4])
    if (signed.get("review_closed") is not True or signed.get("experiment") != b.EXPERIMENT or
            signed.get("manifest_sha256") != manifest_sha or signed.get("qualification_sha256") != qualification_sha
            or signed.get("fix_qualification_sha256") != fix_sha):
        raise ValueError("Review is open or covers different sources or qualifications")


def production_sources():
    manifest = read(START/"source_manifest.json")
    expected = {r["path"]:r for r in manifest["files"]}
    actual = []
    for p in b.source_paths():
        row=record(p)
        m=expected[p.relative_to(WS).as_posix()]
        if (m["bytes"],m["sha256"])!=(row["bytes"],row["sha256"]):
            raise ValueError("Deployed source differs: "+str(p))
        actual.append(row)
    return actual


def prepare():
    if WAIT.exists() or CAMPAIGN.exists() or GATE.exists():
        raise FileExistsError("An existing gate, waiter or campaign cannot be re-created")
    if sha(EN/"formal_gate.json")!=SOURCE_GATE or sha(EN/"launch.json")!=SOURCE_LAUNCH:
        raise ValueError("Wrong deployed VendorLink gate or launch")
    src=production_sources()
    template=WS/"scripts/once_wait.py"
    manifest=read(START/"source_manifest.json")
    if (sha(template)!=manifest["one_shot_sha256"] or sha(Path(__file__))!=manifest["operations_sha256"]
            or sha(ROOT/"scope.zh.md")!=manifest["analysis_plan_sha256"]
            or sha(START/"authorization.zh.md")!=manifest["authorization_text_sha256"]):
        raise ValueError("One-shot launcher, authorization, plan or operations differ")
    conclusion=START/"review.zh.md"
    q=read(START/"qualification.json")
    fix=read(START/"qualification_costs.json")
    accepted_review(conclusion.read_text(encoding="utf-8"),sha(START/"source_manifest.json"),
                    sha(START/"qualification.json"),sha(START/"qualification_costs.json"))
    if q["passed"] is not True or fix["passed"] is not True:
        raise ValueError("Listener qualification incomplete")
    links=inputs(True)
    hours=dict(zip(b.NAMES,(6,5,5,46,6,4),strict=True))
    # The historical 30-second ledger tail is separate. Sleeper time is free.
    preflight_seconds=math.ceil(time.monotonic()-ENTRY)
    cpu=q["prior_cpu_seconds"]+q["listener_cpu_seconds"]+fix["charged_cpu_seconds"]
    control=cpu+preflight_seconds+q["operational_reserve_seconds"]
    authorized=dict(experiment=b.EXPERIMENT,runs=b.plan(),cohort=b.cohort(),actions=["training",*b.POST],
        sources=src,workers=3,seconds_per_method={k:v*3600 for k,v in hours.items()},
        budget_seconds={k:v*3600 for k,v in dict(training=700,control=8,generation=1,confirmation=10,analysis=1).items()},
        prior_used_seconds=dict(training=0,control=control,generation=0,confirmation=q["prior_gpu_seconds"],analysis=0),
        post_seconds=dict(generation=2700,confirmation=28800,analysis=2700),cpu_test_seconds=cpu,
        worker_failure_policy=b.dispatch.POLICY,analysis_plan=record(ROOT/"scope.zh.md"),
        limits=dict(rss_bytes=196*GIB,disk_bytes=320*GIB,small_bytes=8*GIB,free_reserve_bytes=64*GIB,
          free_gpu_bytes=41*GIB//4,allocator_bytes=37*GIB//4,process_gpu_bytes=41*GIB//4,worker_rss_bytes=64*GIB,
          free_ram_bytes=72*GIB,save_reserve_bytes=8*GIB),
        listener=record(template),predecessor=record(EN/"launch.json"),other_gpu_commitments=[],
        allow_observed_foreign_gpu_processes=True)
    auth=ROOT/"authorization.json"
    freeze(auth,dict(authorized,authorize_execution=True,decision=record(START/"authorization.zh.md"),
                     review=record(conclusion),qualification=record(START/"qualification.json"),
                     fix_qualification=record(START/"qualification_costs.json")))
    freeze(GATE,dict(authorized,approved=True,review_closed=True,automatic_retry=False,workspace=str(WS),
                     campaign=str(CAMPAIGN),authorization=record(auth)))
    b.read_gate(GATE)
    value=dict(at=now(),gate=record(GATE),authorization=record(auth),sources=src,inputs=links,
               predecessor=authorized["predecessor"],listener=authorized["listener"],
               preflight_seconds=preflight_seconds,prior=authorized["prior_used_seconds"],repo_git=manifest["basis_git"])
    freeze(START/"production.json",value)
    return dict(gate=value["gate"],source_count=len(src),prior=value["prior"])


def arm():
    prod=read(START/"production.json")
    if verify(prod["gate"])!=GATE or CAMPAIGN.exists() or WAIT.exists():
        raise ValueError("Expected a fresh source-bound gate and unused single-use wait path")
    g=b.read_gate(GATE)
    if verify(g["listener"])!=WS/"scripts/once_wait.py":
        raise ValueError("Wrong listener path")
    # Exclusive console creation reserves the only launcher even before wait.mkdir().
    with (START/"listener_console.txt").open("xb") as console:
        argv=[PYTHON,"-B","-u",g["listener"]["path"],"--gate",str(GATE)]
        proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=console,stderr=subprocess.STDOUT,
              cwd=WS,start_new_session=True,umask=0o077,
              env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false"))
        info=psutil.Process(proc.pid)
        receipt=dict(at=now(),gate=record(GATE),listener=g["listener"],identity=dict(pid=proc.pid,
            owner=info.username(),cwd=info.cwd(),command=argv,**w.kernel_identity(proc.pid)))
        freeze(START/"armed.json",receipt)
    return receipt


def observe():
    out=dict(at=now(),gate=GATE.exists(),campaign=CAMPAIGN.exists(),wait=WAIT.exists())
    if (START/"armed.json").is_file():
        identity=read(START/"armed.json")["identity"]
        out.update(identity=identity,identity_matches=w.matches(identity))
    for label,path in (("wait_status",WAIT/"status.json"),("ready",WAIT/"ready.json"),
                       ("first_update",WAIT/"first_update.json"),("self_deleted",CAMPAIGN/"waiter_deleted.json"),
                       ("wait_failure",WAIT/"failure.json"),("campaign_failure",CAMPAIGN/"failure.json")):
        if path.is_file():
            out[label]=read(path)
    out["listener_exists"]=(WS/"scripts/once_wait.py").is_file()
    if (CAMPAIGN/"dispatch.json").exists():
        d=read(CAMPAIGN/"dispatch.json")
        out["dispatch"]=dict(status=d["status"],pending=len(d["pending"]),failures=d["failures"],
            successful=d.get("successful_tasks",[]),active=[dict(slot=s,identity=i,matches=w.matches(i))
                                                          for s,i in d["active"].items()])
    if (CAMPAIGN/"budget.json").exists():
        out["settled"]=read(CAMPAIGN/"budget.json")["used_seconds"]
    write(START/"observation.json",out)
    return out


def reconcile():
    if not (WAIT/"first_update.json").is_file():
        raise ValueError("An existing first-update proof is required")
    w.retire(b.read_gate(GATE),CAMPAIGN,{})
    return read(CAMPAIGN/"waiter_deleted.json")


if __name__=="__main__":
    if sys.platform!="linux" or Path(sys.executable).resolve()!=Path(PYTHON).resolve():
        raise RuntimeError("Use Linux py310")
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("inputs","prepare","check","arm","observe","reconcile"))
    action=p.parse_args().action
    actions={"inputs":inputs,"prepare":prepare,"check":lambda:dict(sources=len(production_sources()),inputs=inputs()),
             "arm":arm,"observe":observe,"reconcile":reconcile}
    print(json.dumps(actions[action](),ensure_ascii=False))
