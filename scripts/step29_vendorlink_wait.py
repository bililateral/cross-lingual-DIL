"""One-shot resource wait for the already approved native qualification."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from step29_vendorlink_native_control import GIB, gpu_snapshot, sha, write


def wait_reason(target: dict, snapshot: dict, ledger: dict, ram: int, disk: int,
                current_created: float | None, initial_free: int) -> str | None:
    if current_created == target["created"] and target["pid"] in snapshot["processes"]:
        return "TARGET_STILL_USING_GPU"
    required = 3*ledger["g2_admission"]["per_slot"] if "g2_admission" in ledger else initial_free
    if snapshot["free"] < required:
        return "INSUFFICIENT_FREE_VRAM"
    if ram < (72 if "g2_admission" in ledger else 24)*GIB or disk < 32*GIB:
        return "INSUFFICIENT_RAM_OR_DISK"
    return None


def run(config_path: Path, permit_path: Path, home: Path) -> None:
    permit = json.loads(permit_path.read_text())
    config = json.loads(config_path.read_text())
    job = Path(permit["job"])
    if (sys.executable != "/home/yongpeng/miniconda3/envs/py310/bin/python"
            or not permit["approved"] or not permit["implementation_review_closed"]
            or sha(config_path) != permit["config_sha256"]):
        raise ValueError("Approved unchanged qualification required")
    # The open lock remains held even after the active script deletes itself.
    with (home/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (home/"started.json").exists():
            raise ValueError("This one-shot wait was already started")
        write(home/"started.json", dict(pid=os.getpid(), created=psutil.Process().create_time(),
              config_sha256=sha(config_path), target=permit["wait_target"]))
        status = dict(status="WAITING", active_check_seconds=0., charged_seconds=0)
        try:
            while True:
                started = time.monotonic()
                ledger = json.loads((job/"ledger.json").read_text())
                if ledger["status"] != "WAIT_RESOURCE" or ledger["config_sha256"] != sha(config_path):
                    raise ValueError("Only an unchanged resource wait may continue")
                if ledger["aux"] >= 298:
                    raise TimeoutError("No auxiliary balance for another resource check")
                try:
                    target = permit["wait_target"]
                    try:
                        created = psutil.Process(target["pid"]).create_time()
                    except psutil.NoSuchProcess:
                        created = None
                    snapshot = gpu_snapshot(timeout=min(10, 298-ledger["aux"]))
                    reason = wait_reason(target, snapshot, ledger, psutil.virtual_memory().available,
                                         psutil.disk_usage(job).free, created,
                                         config["limits"]["initial_free_bytes"])
                    status.update(reason=reason, resource=snapshot, at=time.time())
                finally:
                    status["active_check_seconds"] += time.monotonic()-started
                    charged = math.ceil(status["active_check_seconds"])
                    ledger["aux"] += charged-status["charged_seconds"]
                    status["charged_seconds"] = charged
                    write(job/"ledger.json", ledger)
                    write(home/"status.json", status)
                if ledger["aux"] >= 298:
                    raise TimeoutError("Auxiliary balance exhausted; no dispatch")
                if reason:
                    time.sleep(300)
                    continue
                command = [sys.executable, "-B", str(Path(__file__).with_name("step29_vendorlink_native_control.py")),
                           "--config", str(config_path), "--permit", str(permit_path), "--job", str(job)]
                status.update(status="QUALIFICATION_RUNNING", command=command)
                write(home/"status.json", status)
                with (home/"control_console.txt").open("a") as output:
                    child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                    status["child_pid"] = child.pid
                    write(home/"status.json", status)
                    while child.poll() is None:
                        # Rows are appended only after a real optimizer update returns.
                        updates = [p for p in (job/"g1").glob("*/updates.jsonl") if p.stat().st_size]
                        if updates and Path(__file__).exists():
                            Path(__file__).unlink()
                            status["self_deleted_after_update"] = dict(at=time.time(), log=str(updates[0]))
                            write(home/"status.json", status)
                        time.sleep(1)
                ledger = json.loads((job/"ledger.json").read_text())
                if child.returncode != 0 or ledger["status"] != "WAIT_RESOURCE":
                    status.update(status=ledger["status"], returncode=child.returncode)
                    if child.returncode != 0:
                        status["next_attempt_minimum_free_bytes"] = 17*GIB
                        status["automatic_retry"] = False
                    write(home/"status.json", status)
                    return
                status["status"] = "WAITING"
        except BaseException as error:
            status.update(status="STOP_NO_AUTOMATIC_RETRY", error=repr(error), at=time.time())
            write(home/"status.json", status)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--permit", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.permit, args.home)
