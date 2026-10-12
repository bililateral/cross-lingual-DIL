"""One-shot resource listener, deployed as an expendable copy after formal qualification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def once(job, active, ready, launch, sleep=time.sleep):
    """Claim before waiting; a failed launch never clears the durable claim."""
    claim = job.with_suffix(".dispatch.json")
    with claim.open("x", encoding="utf-8") as stream:
        json.dump({"listener_pid": os.getpid(), "job": str(job), "active": str(active), "claimed_at": time.time()}, stream)
    log = job.with_suffix(".listener.json")
    while not ready():
        sleep(30)
    child = launch()
    log.write_text(json.dumps({"status": "LAUNCHED_ONCE", "child_pid": child.pid}), encoding="utf-8")
    removed = False
    while True:
        console = job.with_suffix(".console.txt")
        if not removed and console.exists():
            for line in console.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "updates" and event.get("completed", 0) >= 1:
                    active.unlink()
                    removed = True
                    log.write_text(json.dumps({"status": "ACTUAL_FORMAL_UPDATE_SELF_REMOVED", "child_pid": child.pid,
                                               "active_exists": active.exists()}), encoding="utf-8")
                    break
        code = child.poll()
        if code is not None:
            log.write_text(json.dumps({"status": "EXIT_NO_RETRY", "exit_code": code,
                                      "child_pid": child.pid, "self_removed_after_actual_update": removed}), encoding="utf-8")
            return code
        sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    workspace, job, active = args.workspace.resolve(), args.job.resolve(), Path(__file__).resolve()
    if os.name != "posix" or not active.is_relative_to(workspace/"reports") or job.exists() or not job.parent.is_dir():
        parser.error("Deploy only an expendable listener inside this independent workspace/reports")
    sys.path.insert(0, str(workspace/"scripts"))
    import shutil
    import psutil
    import step28_crossfit_retention_run as run
    from step28_record_replay_verify import preflight
    os.chdir(workspace)
    psutil.Process().cpu_affinity([args.cpu])
    run.validate_gate(job, args.gate.resolve())  # No formal files opened by validation.
    def ready():
        try:
            preflight()
            return shutil.disk_usage(workspace).free >= 16*2**30 and psutil.cpu_percent(interval=1, percpu=True)[args.cpu] < 20
        except ValueError:
            return False
    def launch():
        return subprocess.Popen(["bash", str(workspace/"scripts/run_step28_crossfit_retention_linux_20261012.sh"),
                                 str(args.cpu), str(args.gate.resolve()), str(job)], cwd=workspace)
    raise SystemExit(once(job, active, ready, launch))


if __name__ == "__main__":
    main()
