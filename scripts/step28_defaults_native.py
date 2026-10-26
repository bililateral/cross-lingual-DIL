"""Two sequential BGE checks; requires a separate, source-bound GPU authorization."""
from __future__ import annotations

import time
STARTED = time.monotonic()
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import traceback

import step28_continual_defaults as m
import step28_continual_defaults_run as entry
import step28_continual_five_verify as verified

GIB = 2**30
LIMITS = dict(allocator_bytes=int(9.25*GIB), rss_bytes=32*GIB, disk_bytes=12*GIB,
              small_bytes=256*2**20, free_reserve_bytes=64*GIB, free_gpu_bytes=int(10.25*GIB))
METHODS = m.METHODS


def sources() -> list[dict]:
    return entry.sources(("scripts/step28_defaults_native.py", "tests/test_step28_defaults_native.py"))


class Budget(verified.run.Budget):
    def check(self, reserve: int = 0) -> None:
        super().check(reserve)
        if (self.peak_rss > LIMITS["rss_bytes"] or self.peak_disk+reserve > LIMITS["disk_bytes"]
                or self.peak_small > LIMITS["small_bytes"]):
            raise RuntimeError("Original-method native RSS/storage limit exceeded")


def authorization(path: Path, job: Path) -> dict:
    row = m.data.read_json(path)
    seconds = row.get("total_seconds")
    if (row.get("status") != "APPROVED_ORIGINAL_DEFAULTS_NATIVE"
            or row.get("job") != str(job.resolve()) or row.get("methods") != list(METHODS)
            or row.get("limits") != LIMITS or row.get("sources") != sources()
            or row.get("max_update_calls") != 6 or row.get("handwritten_only") is not True
            or row.get("sharing") not in ("idle_only", "current_comparison")
            or type(seconds) not in (int, float) or not 60 < seconds < float("inf")):
        raise ValueError("Separate GPU scope, allocation and exact sources must be approved")
    return row


def admission(sharing: str) -> dict:
    import psutil
    row = verified.resources()
    fields = row["gpu"].split(",")
    free = int(fields[2].strip())*2**20
    if (free < LIMITS["free_gpu_bytes"] or row["host_available_bytes"] < LIMITS["rss_bytes"]
            or row["disk_available_bytes"] < LIMITS["disk_bytes"]+LIMITS["free_reserve_bytes"]):
        raise RuntimeError("Current native resources insufficient; no waiting or launch")
    if row["compute_apps"] and sharing == "idle_only":
        raise RuntimeError("GPU is occupied; idle-only scope cannot launch")
    peers = []
    for line in row["compute_apps"].splitlines():
        pid = int(line.split(",")[0])
        process = psutil.Process(pid)
        command, cwd = process.cmdline(), process.cwd()
        expected = "/home/yongpeng/cross-lingual/reports/documentation/20261026/continual_comparison/workspace"
        if (process.username() != "yongpeng" or cwd != expected
                or expected+"/scripts/step28_continual_comparison_run.py" not in command):
            raise RuntimeError("Unapproved co-resident GPU process; no process is stopped")
        peers.append(dict(pid=pid, owner=process.username(), command=command, cwd=cwd))
    return {**row, "free_gpu_bytes": free, "peers": peers, "instantaneous_not_reserved": True}


def worker(job: Path, name: str, deadline: float, approval_path: Path) -> dict:
    approval = authorization(approval_path, job.parent)
    if name not in METHODS or job.name != name or not job.is_dir():
        raise ValueError("Wrong fixed native worker")
    if deadline <= time.monotonic()+15 or deadline > STARTED+approval["total_seconds"]:
        raise ValueError("Missing or expanded parent deadline")
    return verified.native(job, m.Spec(name, "ABC", 4), deadline, None,
        implementation=m, allocator_bytes=LIMITS["allocator_bytes"], compare_legacy=False,
        source_files=approval["sources"], budget_factory=Budget)


def controller(job: Path, approval_path: Path) -> dict:
    approval = authorization(approval_path, job)
    if job.exists():
        raise ValueError("Existing native attempt is never overwritten or retried")
    deadline = STARTED+approval["total_seconds"]
    job.mkdir(parents=True)
    receipts = []
    result = dict(status="RUNNING", sources=approval["sources"], receipts=receipts,
                  deadline_monotonic=deadline, total_seconds=approval["total_seconds"])
    # Lock only this supplemental lane; never lock or modify another campaign.
    with (job.parent/"defaults_native.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for name in METHODS:
                resource = admission(approval["sharing"])
                child_job = job/name
                child_job.mkdir()
                m.data.write_json(child_job/"resource_before.json", resource)
                remaining = deadline-time.monotonic()-5
                if remaining <= 15:
                    raise TimeoutError("No remaining native allowance; first-arm cost is retained")
                command = [sys.executable, "-B", str(Path(__file__).resolve()), "--job", str(child_job),
                    "--authorization", str(approval_path.resolve()), "--worker", name, "--deadline", str(deadline)]
                environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                    TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
                started, code, failure = time.monotonic(), None, None
                m.data.write_json(job/"status.json", {**result, "active": name})
                try:
                    with (child_job/"console.txt").open("w") as log:
                        # subprocess.run kills and waits only for this owned child on timeout.
                        code = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                            timeout=remaining, check=False).returncode
                except subprocess.TimeoutExpired:
                    failure = "PARENT_DEADLINE"
                finally:
                    receipts.append(dict(method=name, exit_code=code, failure=failure,
                        elapsed_seconds=time.monotonic()-started))
                    m.data.write_json(job/"status.json", result)
                if code != 0:
                    raise RuntimeError("Native worker failed; no retry or second arm")
                outcome = m.data.read_json(child_job/"result.json")
                if (outcome.get("status") != "PASS_NATIVE_HANDWRITTEN" or outcome.get("attempted_calls") != 3
                        or outcome.get("actual_calls") != 3 or outcome.get("spec") != m.Spec(name, "ABC", 4).identity()
                        or outcome.get("source_files") != approval["sources"]
                        or outcome.get("outer_checkpoint_equals_legacy") is not False
                        or not outcome.get("actual_next_update_equal") or not outcome.get("inference_equal")):
                    raise ValueError("Native worker receipt does not prove the required fixed scope")
            result["status"] = "COMPLETE_TWO_ORIGINAL_NATIVE_CHECKS"
        except BaseException:
            result.update(status="FAILED_STOP_NO_RETRY", traceback=traceback.format_exc())
            raise
        finally:
            result["charged_seconds"] = time.monotonic()-STARTED
            m.data.write_json(job/"status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--worker", choices=METHODS)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    try:
        if args.worker:
            worker(args.job, args.worker, args.deadline, args.authorization)
        else:
            controller(args.job, args.authorization)
    except BaseException:
        if args.job.is_dir():
            (args.job/"failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
