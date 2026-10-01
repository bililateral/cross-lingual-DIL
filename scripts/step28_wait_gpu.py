"""One-use queue: reviewed base models first, then approved sensitivity training.

Scheduling only. No dataset parsing, model loading, evaluation, or automatic retry.
"""
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INTERVAL = 60
BASE_COMPLETE = "COMPLETE_BASE_MODELS_2592_CALIBRATED_VALID_BLIND"
OLD_SOURCES_SHA256 = "a0acd5e65bed29fcc10b8314b53c69dd2c649bb9112c9949e18b61278bf1dac2"


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resources(minimum_disk_bytes: int) -> dict:
    gpu = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    jobs = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    if gpu.returncode or jobs.returncode:
        raise RuntimeError("GPU resource query failed")
    free = int(gpu.stdout.strip())
    pids = [int(x.strip()) for x in jobs.stdout.splitlines() if x.strip()]
    memory = next(int(x.split()[1]) * 1024 for x in Path("/proc/meminfo").read_text().splitlines()
                  if x.startswith("MemAvailable:"))
    disk = shutil.disk_usage(ROOT).free
    return {"gpu_free_mib": free, "compute_pids": pids, "host_free_bytes": memory,
            "disk_free_bytes": disk, "required_disk_bytes": minimum_disk_bytes,
            "ready": not pids and free >= 24576 and memory >= 16 * 1024**3
            and disk >= minimum_disk_bytes}


def approved(queue: Path, stage: str, sensitivity: Path) -> list[dict] | None:
    if stage == "base_models":
        path = queue / "base_review_ready.json"
        if not path.exists():
            return None
        value = read(path)
        if (value["status"] != "EXTERNAL_AND_PRIMARY_REVIEW_COMPLETE"
                or not value.get("review_reply_id") or value.get("model_slug") != "gpt-6-pro"):
            raise ValueError("Base-model scientific review is not complete")
        rows = value["sources"]
    elif stage == "weight_sensitivity":
        path = sensitivity / "approved_sources.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != OLD_SOURCES_SHA256:
            raise ValueError("Original sensitivity approval changed")
        rows = read(path)
        if len(rows) != 21:
            raise ValueError("Original sensitivity source count differs")
    else:
        raise ValueError("Unknown queue stage")
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Missing or repeated approved sources")
    return rows


def verify_sources(rows: list[dict]) -> None:
    for row in rows:
        path = ROOT / row["path"]
        if (path.stat().st_size != row["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise RuntimeError("Approved source changed: " + row["path"])


def wait_ready(queue: Path, stage: str, sensitivity: Path, disk_bytes: int) -> list[dict]:
    consecutive = 0
    while True:
        rows = approved(queue, stage, sensitivity)
        try:
            state = resources(disk_bytes)
        except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            state = {"ready": False, "error": str(error)}
        state["review_ready"] = rows is not None
        state["resource_ready"] = state["ready"]
        state["ready"] = state["ready"] and rows is not None
        consecutive = consecutive + 1 if state["ready"] else 0
        state.update(stage=stage, time=now(), consecutive_ready=consecutive,
                     status="WAITING_RESOURCES" if rows is not None else "WAITING_BASE_REVIEW")
        write(queue / "wait_status.json", state)
        if consecutive >= 2:
            verify_sources(rows)
            return rows
        time.sleep(INTERVAL)


def launch(base: Path, script: str, rows: list[dict]) -> Any:
    verify_sources(rows)
    if (base / "job").exists():
        raise RuntimeError("Job directory exists; no restart")
    base.mkdir(parents=True, exist_ok=True)
    with (base / "launch.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "LAUNCH_ATTEMPT", "time": now(), "script": script}, stream)
    with (base / "launcher.log").open("ab") as log:
        process = subprocess.Popen(["bash", str(ROOT / script), str(base / "job")],
            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(base / "launch.json", {"status": "LAUNCHER_STARTED", "pid": process.pid,
                                 "time": now(), "script": script, "gpu": 0})
    return process


def check_base_completion(base: Path, rows: list[dict]) -> None:
    job, run = base / "job", base / "job/run"
    if ((run / "failure.json").exists() or (job / "exit_status.txt").read_text().strip() != "0"):
        raise RuntimeError("Base training failed; sensitivity will not be launched")
    manifest, completion = read(run / "manifest.json"), read(run / "completion.json")
    digest = hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()
    models = read(run / "model_verification.json")
    if (manifest["status"] != BASE_COMPLETE or manifest["physical_updates"] != 2592
            or set(manifest["arms"]) != {"labse", "multilingual_e5_large", "bge_m3"}
            or manifest["source_files"] != rows or completion["status"] != BASE_COMPLETE
            or completion["manifest_sha256"] != digest or models["manifest_sha256"] != digest
            or models["status"] != "SIX_MODELS_FRESH_SIZE_SHA_VERIFIED" or len(models["files"]) != 6):
        raise RuntimeError("Base output is incomplete; sensitivity will not be launched")


def run_queue(queue: Path) -> None:
    config = read(queue / "queue.json")
    base = ROOT / config["base_models"]
    sensitivity = ROOT / config["weight_sensitivity"]
    for path in (queue, base, sensitivity):
        if not path.resolve().is_relative_to(ROOT / "reports"):
            raise ValueError("Queue outputs must stay inside project reports")
    # A queue is started once. A crash or restart requires explicit inspection.
    with (queue / "queue_start.json").open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "time": now(), "order": ["base_models", "weight_sensitivity"]}, stream)
    rows = wait_ready(queue, "base_models", sensitivity, 32 * 1024**3)
    process = launch(base, "scripts/run_step28_base_model_linux_20260918.sh", rows)
    while process.poll() is None:
        write(queue / "wait_status.json", {"status": "BASE_TRAINING_RUNNING", "pid": process.pid,
                                          "time": now(), "sensitivity_started": False})
        time.sleep(INTERVAL)
    if process.returncode != 0:
        raise RuntimeError("Base launcher failed; no sensitivity launch or retry")
    check_base_completion(base, rows)
    write(queue / "base_completed.json", {"status": BASE_COMPLETE, "time": now(),
                                          "valid_evaluation": "Not invoked by scheduler"})
    rows = wait_ready(queue, "weight_sensitivity", sensitivity, 24 * 1024**3)
    process = launch(sensitivity, "scripts/run_step28_sensitivity_linux_20260916.sh", rows)
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError("Sensitivity launcher exited before startup; no retry")
        if (sensitivity / "job/run/startup.json").is_file():
            script = Path(__file__).resolve()
            if script != ROOT / "scripts/step28_wait_gpu.py":
                raise RuntimeError("Unexpected waiter self-delete path")
            script.unlink()
            receipt = {"status": "BASE_COMPLETE_SENSITIVITY_ENTRY_STARTED_WAITER_REMOVED",
                       "time": now(), "pid": process.pid, "waiter_removed": not script.exists(),
                       "sensitivity_training_complete": False, "valid_evaluations_invoked": 0}
            write(queue / "completion.json", receipt)
            write(sensitivity / "launch.json", receipt)
            return
        time.sleep(1)
    raise RuntimeError("Sensitivity startup not observed in 120 seconds; inspect recorded PID, no retry")


if __name__ == "__main__":
    queue = Path(sys.argv[1]).resolve()
    try:
        run_queue(queue)
    except Exception as error:
        if queue.is_relative_to(ROOT / "reports") and queue.is_dir():
            write(queue / "wait_failure.json", {"error": str(error), "time": now(), "automatic_retry": False})
        raise
