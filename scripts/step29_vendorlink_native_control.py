"""Bounded export/G1/G2 supervisor; no formal trajectory dispatch or auto retry."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil

GIB = 2**30


def write(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def gpu_snapshot():
    # One XML capture supplies global free and all process occupancies together.
    raw = subprocess.check_output(["nvidia-smi", "-q", "-x", "-i", "0"], timeout=10)
    gpu = ET.fromstring(raw).find("gpu")
    mib = lambda text: int(text.split()[0])*2**20
    return dict(at=time.monotonic(), free=mib(gpu.findtext("fb_memory_usage/free")),
                total=mib(gpu.findtext("fb_memory_usage/total")),
                processes={int(p.findtext("pid")): mib(p.findtext("used_memory"))
                           for p in gpu.findall("processes/process_info")})


def fits(snapshot, reservations, requested):
    unreflected = sum(max(0, cap-snapshot["processes"].get(pid, 0))
                      for pid, cap in reservations.items())
    return snapshot["free"] >= requested+unreflected


def overlap(rows):
    """Any nonempty interval during which all three workers performed a real step."""
    return any(max(a["started"], b["started"], c["started"]) < min(a["ended"], b["ended"], c["ended"])
               for a in rows[0] for b in rows[1] for c in rows[2])


def source_check(config):
    for row in config["sources"]+config["model_files"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Frozen source/model identity differs: "+str(path))


def supervise(commands, folders, seconds, phase, job, environment, ledger, config, reserve=0):
    children, handles, starts, stops = [], [], [], {}
    usage, nvml, failure = [], {}, None
    watch_started = time.monotonic()
    try:
        reservations = {}
        for command, folder in zip(commands, folders):
            if reserve and not fits(gpu_snapshot(), reservations, reserve):
                raise RuntimeError("Resource changed during reserved launch; no automatic retry")
            handle = (job/(folder.name+"_console.txt")).open("w")
            handles.append(handle)
            starts.append(time.monotonic())
            child = subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            children.append(child)
            reservations[child.pid] = reserve
        gate_open = False
        next_sample = 0.
        while any(p.poll() is None for p in children):
            now = time.monotonic()
            for p in children:
                if p.poll() is not None:
                    stops.setdefault(p.pid, now)
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError("Worker failed; stop this qualification")
            if any(p.poll() is None and now-start >= seconds-2 for p, start in zip(children, starts)):
                raise TimeoutError("Task time cap; exit tail reserved")
            rss = 0
            for child in children:
                try:
                    process = psutil.Process(child.pid)
                    own_rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)])
                    if own_rss > 24*GIB:
                        raise MemoryError("Worker exceeds 24 GiB RAM")
                    rss += own_rss
                except psutil.NoSuchProcess:
                    pass
            if rss > 72*GIB:
                raise MemoryError("Qualification exceeds 72 GiB RAM")
            if now >= next_sample:
                if sum(p.stat().st_size for p in job.rglob("*") if p.is_file()) > 32*GIB:
                    raise MemoryError("Qualification outputs exceed 32 GiB")
                if phase != "aux":
                    sample = gpu_snapshot()
                    active = "g2"
                    if phase == "g1" and (folders[0]/"active.json").exists():
                        active = json.loads((folders[0]/"active.json").read_text())["method"]
                    nvml[active] = max(nvml.get(active, 0), *(sample["processes"].get(p.pid, 0) for p in children))
                    with (job/(phase+"_gpu.jsonl")).open("a") as out:
                        out.write(json.dumps(sample)+"\n")
                next_sample = now+1
            if phase == "g2" and not gate_open and all((f/"ready.json").exists() for f in folders):
                write(job/"g2_gate.json", dict(at=now, ready_pids=[p.pid for p in children]))
                gate_open = True
            time.sleep(.1)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError("Qualification worker exited unsuccessfully")
    except BaseException as error:
        failure = repr(error)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
    finally:
        for child, start in zip(children, starts):
            child.wait()
            ended = stops.setdefault(child.pid, time.monotonic())
            usage.append(dict(pid=child.pid, returncode=child.returncode,
                              wall_seconds=ended-start, charged_seconds=math.ceil(ended-start)))
        for handle in handles:
            handle.close()
        ledger[phase] += sum(r["charged_seconds"] for r in usage)
        ledger["attempts"].append(dict(phase=phase, tasks=usage, failure=failure, nvml=nvml,
                                        supervision_wall_seconds=time.monotonic()-watch_started))
        ledger["status"] = "STOP_NO_AUTOMATIC_RETRY" if failure else "PHASE_FINISHED"
        write(job/"ledger.json", ledger)
    if failure:
        raise RuntimeError(failure)
    return nvml, time.monotonic()-watch_started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--permit", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    config, permit = json.loads(args.config.read_text()), json.loads(args.permit.read_text())
    if (sys.platform != "linux" or sys.executable != "/home/yongpeng/miniconda3/envs/py310/bin/python"
            or permit.get("operation") != "vendorlink_native_qualification" or not permit.get("approved")
            or not permit.get("implementation_review_closed") or permit.get("config_sha256") != sha(args.config)
            or permit.get("job") != str(args.job.resolve())):
        raise ValueError("Exact reviewed/approved scope and Linux py310 required")
    root = Path(__file__).resolve().parent
    started = time.monotonic()
    if args.job.exists():
        ledger = json.loads((args.job/"ledger.json").read_text())
        if ledger["status"] != "WAIT_RESOURCE" or ledger["config_sha256"] != sha(args.config):
            raise ValueError("Only an unchanged resource wait may continue automatically")
    else:
        args.job.mkdir(parents=True)
        ledger = dict(status="PREPARING", config_sha256=sha(args.config), aux=0, g1=0, g2=0, attempts=[])
    attempt_start = len(ledger["attempts"])
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false", TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    base = ["--config", str(args.config), "--permit", str(args.permit)]
    try:
        source_check(config)
        if ledger["aux"] >= 290:
            raise TimeoutError("Insufficient auxiliary balance for qualification supervision")
        if "export_manifest_sha256" not in ledger:
            _, elapsed = supervise([[sys.executable, "-B", str(root/"step29_vendorlink_export.py"), *base]],
                [Path(config["export"])], 295-ledger["aux"], "aux", args.job, environment, ledger, config)
            ledger["export_manifest_sha256"] = sha(config["export_manifest"])
        if sha(config["export_manifest"]) != ledger["export_manifest_sha256"]:
            raise ValueError("Export manifest changed")
        sample = gpu_snapshot()
        write(args.job/"resource.json", sample)
        if psutil.virtual_memory().available < 24*GIB or psutil.disk_usage(args.job).free < 32*GIB:
            ledger["status"] = "WAIT_RESOURCE"
            return
        g1 = args.job/"g1"
        if not (g1/"result.json").exists():
            if sample["free"] < 17*GIB:
                ledger["status"] = "WAIT_RESOURCE"
                return
            nvml, elapsed = supervise([[sys.executable, "-B", str(root/"step29_vendorlink_native.py"),
                *base, "--mode", "g1", "--job", str(g1)]], [g1], 900-ledger["g1"], "g1", args.job,
                environment, ledger, config, reserve=17*GIB)
            ledger["g1_nvml"] = nvml
        result = json.loads((g1/"result.json").read_text())
        if result["status"] != "G1_PASSED" or result["updates"] > 15:
            raise ValueError("G1 qualification incomplete")
        peaks = {m: max(r["peak"]["reserved"], ledger["g1_nvml"].get(m, 0)) for m, r in result["methods"].items()}
        method = max(peaks, key=peaks.get)
        reserve = math.ceil(peaks[method]+max(GIB, peaks[method]*.1))
        sample = gpu_snapshot()
        ledger["g2_admission"] = dict(method=method, per_slot=reserve, resource=sample)
        if sample["free"] < 3*reserve or psutil.virtual_memory().available < 72*GIB:
            ledger["status"] = "WAIT_RESOURCE"
            return
        folders = [args.job/f"g2_{i}" for i in range(3)]
        commands = [[sys.executable, "-B", str(root/"step29_vendorlink_native.py"), *base,
            "--mode", "g2", "--job", str(f), "--method", method, "--checkpoint", result["methods"][method]["checkpoint"],
            "--gate", str(args.job/"g2_gate.json")] for f in folders]
        _, elapsed = supervise(commands, folders, min(300, (900-ledger["g2"])//3), "g2", args.job,
            environment, ledger, config, reserve=reserve)
        results = [json.loads((f/"result.json").read_text()) for f in folders]
        if any(r["status"] != "G2_WORKER_PASSED" for r in results) or not overlap([r["rows"] for r in results]):
            raise AssertionError("No actual three-way optimizer-update overlap")
        ledger["status"] = "NATIVE_QUALIFIED_NOT_FORMAL_TRAINING"
        ledger["optimizer_updates"] = result["updates"]+sum(r["updates"] for r in results)
    except BaseException as error:
        ledger["status"], ledger["error"] = "STOP_NO_AUTOMATIC_RETRY", repr(error)
        raise
    finally:
        # Parent supervision overlaps workers; only nonoverlapping preparation/finish time is extra.
        child_union = sum(r["supervision_wall_seconds"] for r in ledger["attempts"][attempt_start:])
        ledger["aux"] += math.ceil(max(0, time.monotonic()-started-child_union))
        if ledger["aux"] > 300 or ledger["g1"] > 900 or ledger["g2"] > 900:
            ledger["status"] = "BUDGET_EXCEEDED_STOP"
        write(args.job/"ledger.json", ledger)
        print(json.dumps(ledger), flush=True)


if __name__ == "__main__":
    main()
