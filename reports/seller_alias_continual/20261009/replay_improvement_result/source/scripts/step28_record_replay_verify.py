"""Budgeted handwritten CPU checks and reviewed native worst-shape first update."""
from __future__ import annotations

import time
STARTED = time.monotonic()

import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import unittest

import psutil
import torch

import step28_record_replay as method
import step28_record_replay_run as run


def preflight() -> dict:
    process = psutil.Process()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or len(process.cpu_affinity()) != 1:
        raise ValueError("Physical GPU0 and exactly one CPU affinity required")
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise ValueError("Single-thread environment must precede imports")
    if os.environ.get("TOKENIZERS_PARALLELISM") != "false":
        raise ValueError("Tokenizer parallelism must be disabled")
    gpu = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=index,uuid,memory.total,memory.free,memory.used",
                                   "--format=csv,noheader,nounits"], text=True).strip()
    apps = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    ancestors = {process.pid, *(p.pid for p in process.parents())}
    jobs = [p.info for p in psutil.process_iter(["pid", "username", "cmdline"])
            if p.pid not in ancestors and any("step28" in x for x in (p.info["cmdline"] or []))]
    available = psutil.virtual_memory().available
    if int(gpu.split(",")[-2]) < 30*1024 or available < 72*2**30 or apps or jobs:
        raise ValueError(f"Resource gate failed: gpu={gpu}, apps={apps}, host={available}, jobs={jobs}")
    return {"gpu": gpu, "compute_apps": apps, "host_available_bytes": available, "project_jobs": jobs}


def native(check, progress, handmade) -> dict:
    resources = preflight()
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory, 0)
    c = method.config()
    model = method.load_model(c)
    auto = model.encoder[0].auto_model
    if not auto.is_gradient_checkpointing or not torch.cuda.is_bf16_supported():
        raise ValueError("Required native execution mode missing")
    current, history = [handmade.handmade_group("native_"+role, 8, 254) for role in ("current", "history")]
    for group in (current, history):
        lengths = [len(v) for v in model.encoder.tokenizer(method.base.record_texts(group, "separate_moments"),
                   padding=False, truncation=False)["input_ids"]]
        if len(lengths) != 448 or set(lengths) != {256}:
            raise ValueError("Handwritten probe is not the actual N224/token256 worst shape")
    progress("native_source_eval", resources=resources)
    tick = time.monotonic()
    teacher = method.reference(model, history, c, check)
    torch.cuda.synchronize()
    source_seconds = time.monotonic()-tick
    if teacher.shape != (24976,):
        raise ValueError("Incomplete native source table")
    with torch.no_grad():
        model.head[0].weight.add_(.003)
    probes = {"encoder_query": auto.encoder.layer[0].attention.self.query.weight,
              "head_hidden": model.head[0].weight, "head_output": model.head[2].weight}
    before = {k: p.detach().clone() for k, p in probes.items()}
    gradient_snapshots, increments = {}, {}
    def observe(role, log):
        increments[role] = {}
        for key, param in probes.items():
            if param.grad is None:
                raise ValueError("Actual encoder/head branch has no gradient")
            old = gradient_snapshots.get(key, torch.zeros_like(param.grad))
            difference = param.grad.detach()-old
            norm = float(difference.norm())
            if not torch.isfinite(difference).all() or norm <= 0:
                raise ValueError("Native branch did not affect the actual encoder/head gradient")
            increments[role][key] = norm
            gradient_snapshots[key] = param.grad.detach().clone()
    forwards = [0]
    def count_forward(module, args):
        forwards[0] += 1
    handle = auto.encoder.layer[0].register_forward_pre_hook(count_forward)
    optimizer = method.core.make_optimizer(model, c)
    progress("native_first_optimizer_update", source_seconds=source_seconds)
    tick = time.monotonic()
    try:
        log = method.update(model, optimizer, current, history, teacher, c, "C", "ABC", 2, 1, check, observe)
    finally:
        handle.remove()
    torch.cuda.synchronize()
    update_seconds = time.monotonic()-tick
    if forwards[0] <= 224:
        raise ValueError("Native activation checkpoint recomputation not observed")
    changed = {k: bool(torch.any(p.detach() != before[k])) for k, p in probes.items()}
    if not all(changed.values()) or method.parent.adam_step(optimizer) != 1:
        raise ValueError("Actual first Adam update missing")
    tensors = [*model.parameters(), *(p.grad for p in model.parameters() if p.grad is not None),
               *(s[k] for s in optimizer.state.values() for k in ("exp_avg", "exp_avg_sq"))]
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("Native parameters/gradients/Adam moments must remain FP32")
    # Conservative shape-only projection: all gradient groups treated as half this
    # paired update, all 2,274 eval/source group passes at worst-shape source cost;
    # 15*2*(12+60)+60+3*6+6*6 = 2,274. Add checkpoint time and a 25% margin.
    estimate = 1.25*(7776*update_seconds/2 + 2274*source_seconds + 1800)
    return {"kind": "native_record_replay_first_optimizer_step", "records_per_group": 224,
            "tokens_per_channel": 256, "source_table_values": 24976, "source_seconds": source_seconds,
            "paired_update_seconds": update_seconds, "gradient_increments": increments, "parameters_changed": changed,
            "first_layer_forward_calls": forwards[0], "adam_step": 1, "log": log,
            "parameter_gradient_adam_dtype": "float32", "shape_upper_projection_seconds": estimate,
            "within_formal_24h_projection": estimate <= 86400,
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--prior-seconds", type=float, default=0)
    args = parser.parse_args()
    if os.name != "posix" or args.output.exists() or args.prior_seconds < 0:
        parser.error("Linux, new output and honest cumulative time required")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(method.data.ROOT/"tests"))
    import test_step28_record_replay as handwritten
    os.environ["RECORD_REPLAY_TEST_ROOT"] = str(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    report = {"status": "RUNNING", "mode": args.mode, "source_files": run.sources(),
              "prior_compute_seconds": args.prior_seconds, "formal_inputs": False, "formal_labels": False,
              "peak_rss_bytes": 0}
    limit = 180 if args.mode == "cpu" else 600
    stopped = threading.Event()
    lock = threading.RLock()
    def write():
        with lock:
            report["elapsed_seconds"] = time.monotonic()-STARTED
            report["cumulative_seconds"] = args.prior_seconds+report["elapsed_seconds"]
            if args.mode == "gpu" and torch.cuda.is_initialized():
                report["max_reserved_bytes"] = torch.cuda.max_memory_reserved()
                report["max_allocated_bytes"] = torch.cuda.max_memory_allocated()
            method.data.write_json(args.output/"result.json", report)
    def check():
        report["peak_rss_bytes"] = max(report["peak_rss_bytes"], process.memory_info().rss)
        used = 0
        for path in args.output.parent.rglob("*"):
            try:
                if path.is_file():
                    used += path.stat().st_size
            except FileNotFoundError:
                pass
        if (time.monotonic()-STARTED+args.prior_seconds > limit
                or report["peak_rss_bytes"] > (2 if args.mode == "cpu" else 64)*2**30
                or used > (16 if args.mode == "cpu" else 128)*2**20):
            raise RuntimeError("Approved handwritten verification budget exceeded")
        if args.mode == "gpu" and torch.cuda.is_initialized() and torch.cuda.max_memory_reserved() > 28*2**30:
            raise RuntimeError("CUDA reserved exceeds 28 GiB")
    def watchdog():
        while not stopped.wait(.5):
            try:
                check()
            except BaseException as exc:
                try:
                    report.update(status="BUDGET_STOP", error=str(exc))
                    write()
                finally:
                    os._exit(2)
    def progress(stage, **details):
        report.update(stage=stage, **details)
        write()
    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()
    try:
        if len(process.cpu_affinity()) != 1 or any(os.environ.get(k) != "1" for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
            raise ValueError("One CPU and pre-import thread limits required")
        progress("started")
        if args.mode == "cpu":
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise ValueError("CPU checks must disable CUDA")
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(handwritten.RecordReplayTests)
            with (args.output/"unittest.txt").open("w", encoding="utf-8") as stream:
                result = unittest.TextTestRunner(stream=stream, verbosity=2, failfast=True).run(suite)
            report.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
            if not result.wasSuccessful() or result.skipped:
                raise RuntimeError("Handwritten checks failed")
        else:
            if args.gate is None:
                raise ValueError("Native GPU needs reviewed implementation and CPU gate")
            gate = method.data.read_json(args.gate)
            if (gate.get("status") != "APPROVED_RECORD_REPLAY_NATIVE" or gate.get("source_files") != run.sources()
                    or gate.get("review_disposition") != "NO_OPEN_BLOCKERS"):
                raise ValueError("Native review gate differs")
            rec = gate["cpu"]
            evidence = method.data.read_json(method.data.verify(method.data.ROOT/rec["path"], rec))
            if evidence.get("status") != "PASS_HANDWRITTEN_ONLY" or evidence.get("mode") != "cpu" or evidence.get("source_files") != run.sources():
                raise ValueError("Matching CPU evidence required")
            report["native"] = native(check, progress, handwritten)
        check()
        report["status"] = "PASS_HANDWRITTEN_ONLY"
    except BaseException:
        report.update(status="FAIL", traceback=traceback.format_exc())
        raise
    finally:
        stopped.set()
        watcher.join(timeout=2)
        write()


if __name__ == "__main__":
    main()
