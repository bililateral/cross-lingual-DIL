"""Bounded handwritten qualification, sharing the actual update/save/launcher."""
from __future__ import annotations

import time
STARTED = time.monotonic()
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import unittest

import numpy as np
import psutil

import step28_continual_five as m
import step28_continual_five_run as run
import step28_continual_five_control as ctl

ALLOCATOR_BYTES = int(9.25*2**30)


def resources() -> dict:
    gpu = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True).strip()
    apps = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid,used_memory",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    return dict(time=time.time(), gpu=gpu, compute_apps=apps,
                host_available_bytes=psutil.virtual_memory().available,
                disk_available_bytes=__import__("shutil").disk_usage(m.data.ROOT).free)


def native(job: Path, spec: m.Spec, deadline: float, barrier: Path | None) -> dict:
    import torch
    sys.path.insert(0, str(m.data.ROOT/"tests"))
    from test_step28_record_replay import handmade_group
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m.parent.configure_numerics()
    allocator = ALLOCATOR_BYTES if barrier else 28*2**30
    torch.cuda.set_per_process_memory_fraction(allocator/torch.cuda.get_device_properties(0).total_memory)
    budget = run.Budget(job.parent, deadline, tail=15)
    c, method = m.config(spec), m.Method(spec)
    measurements, calls = {}, []
    def progress(event):
        m.data.write_json(job/"progress.json", dict(event=event, attempted_calls=len(calls), calls=calls,
                                                  measurements=measurements, resources=budget.snapshot()))
    def timed(name, fn):
        budget.check()
        torch.cuda.synchronize()
        tick = time.monotonic()
        value = fn()
        torch.cuda.synchronize()
        measurements[name] = time.monotonic()-tick
        progress(name)
        return value
    def synchronize_phase(name):
        if barrier is None:
            return
        (barrier/(name+"_"+spec.name)).write_text(str(time.monotonic()))
        while len(list(barrier.glob(name+"_*"))) != 3:
            budget.check()
            time.sleep(.1)
    archive = m.core.model_files(m.base.model_config(c, "split_rank"))
    if any(archive[k] != c["models"]["split_rank"][k] for k in ("file_count", "total_size_bytes", "content_sha256")):
        raise ValueError("Native pretrained archive differs")
    model, optimizer = timed("load", lambda: method.fresh(c))
    if barrier is None:
        model.encoder._five_outer_checkpoint = False  # Named qualification comparator only.
    if not model.encoder[0].auto_model.is_gradient_checkpointing or not torch.cuda.is_bf16_supported():
        raise ValueError("Required BF16/checkpointing unavailable")
    current, history = [handmade_group("native_"+role, 8, 254) for role in ("current", "history")]
    for group in (current, history):
        lengths = [len(v) for v in model.encoder.tokenizer(m.base.record_texts(group, "separate_moments"),
                                                          padding=False, truncation=False)["input_ids"]]
        if len(lengths) != 448 or set(lengths) != {256}:
            raise ValueError("Maximum 224-record/256-token shape absent")
    def update(stage, step, target=None):
        if len(calls) >= 3:
            raise ValueError("Native worker's three-call cap exhausted")
        calls.append(dict(stage=stage, step=step, status="STARTED"))
        progress("call_started")
        result = method.update(model, optimizer, current, history if target is not None else None,
                               target, c, stage, step, budget.check)
        calls[-1].update(status="COMPLETE", adam_step=m.parent.adam_step(optimizer), row=result)
        progress("call_complete")
        return result
    synchronize_phase("loaded")
    timed("current_update", lambda: update(1, 1))
    # Synthetic arriving-stage state, explicitly unrelated to formal path results.
    for state in optimizer.state.values():
        state["step"].fill_(624)
    with m.diag.neutral(model):
        raw = timed("forward", lambda: method.score(model, [history], c, budget.check))
        teacher = (timed("teacher", lambda: m.record.reference(model, history, c, budget.check))
                   if spec.is_record else raw[0].copy())
    teacher = (teacher+2*np.sin(np.arange(len(teacher))*.31)).astype(np.float32)
    (job/"work").mkdir()
    path = job/"work/full.pt"
    meta = {**method.metadata(), "rng": m.diag.rng_state(), "modes": [v.training for v in model.modules()],
            "fixture": "handwritten Adam step624, not a formal trained state"}
    synchronize_phase("resident")
    full = timed("save_full", lambda: run.save(path, model, optimizer, meta, budget))
    synchronize_phase("full_saved")
    first = timed("replay_update", lambda: update(3, 49, teacher))
    digest = m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
    rng = m.diag.rng_state()
    timed("restore_full", lambda: method.restore(path, full, model, optimizer))
    model.encoder._five_outer_checkpoint = True
    synchronize_phase("restored")
    second = timed("replay_repeat", lambda: update(3, 49, teacher))
    if (first != second or rng != m.diag.rng_state()
            or digest != m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))):
        raise ValueError("Actual full restore/next update differs")
    with m.diag.neutral(model):
        before = timed("forward_after", lambda: method.score(model, [history], c, budget.check))
    inference_path = job/"work/inference.pt"
    inference = timed("save_inference", lambda: run.save(inference_path, model, None, meta, budget))
    timed("restore_inference", lambda: method.restore(inference_path, inference, model, None))
    with m.diag.neutral(model):
        np.testing.assert_array_equal(before, method.score(model, [history], c, budget.check))
    synchronize_phase("done")
    files = [m.data.record(p, job) for p in (path, inference_path)]
    result = dict(status="PASS_NATIVE_HANDWRITTEN", spec=spec.identity(), attempted_calls=len(calls),
        actual_calls=sum(r["status"] == "COMPLETE" for r in calls), measurements=measurements,
        native_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        native_peak_reserved_bytes=torch.cuda.max_memory_reserved(), allocator_cap_bytes=allocator,
        full_bytes=full["bytes"], inference_bytes=inference["bytes"], resources=budget.snapshot(),
        actual_next_update_equal=True, outer_checkpoint_equals_legacy=barrier is None,
        inference_equal=True, maximum_shape=[224, 256], archive=archive,
        retired_temporary_states=files, source_files=run.sources())
    m.data.write_json(job/"result.json", result)
    for p in (path, inference_path):
        m.base.persistence.remove_work_file(p, job)
    return result


def native_batch(root: Path, ledger_path: Path) -> dict:
    prior_calls = sum(m.data.read_json(p)["attempted_calls"] for p in root.parent.glob("native*/**/progress.json"))
    if prior_calls+15 > 36:
        raise ValueError("Remaining aggregate qualification calls cannot cover this attempt")
    observation = resources()
    if (observation["compute_apps"] or int(observation["gpu"].split(",")[2]) < 31*1024
            or observation["host_available_bytes"] < 208*2**30 or observation["disk_available_bytes"] < 288*2**30):
        raise RuntimeError("Native three-worker initial resource gate unavailable")
    root.mkdir()
    m.data.write_json(root/"preflight.json", observation)
    ledger = ctl.Ledger(ledger_path)
    if ledger_path.exists():
        prior = m.data.read_json(ledger_path)
        if prior["active"]:
            raise ValueError("Unsettled prior qualification; reconcile before any new attempt")
        ledger.used.update(prior["used"])
        ledger.completed = prior["completed"]
    ledger.save()
    stop = threading.Event()
    monitor_seconds = []
    def monitor():
        with (root/"resource_samples.jsonl").open("w", encoding="utf-8") as stream:
            while not stop.is_set():
                tick = time.monotonic()
                row = resources()
                row["owned_rss_bytes"] = run.resident_bytes(run.owned_processes(os.getpid()))
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                monitor_seconds.append(time.monotonic()-tick)
                stop.wait(1)
    thread = threading.Thread(target=monitor, daemon=True)
    ledger.charge("control", time.monotonic()-STARTED)
    thread.start()
    try:
        # Serial S and account LOGIT provide a contemporaneous contention proxy.
        for name in ("S_tuned", "LOGIT0.1_ref"):
            spec = m.Spec(name, "CAB", 4)
            row = dict(id=root.name+"_single_"+name, spec=spec.identity(), allowance=600.)
            def command(q, job, deadline):
                m.data.write_json(job/"spec.json", q["spec"])
                return [sys.executable, "-B", str(Path(__file__)), "native", "--job", str(job),
                        "--spec", str(job/"spec.json"), "--deadline", str(deadline)]
            ctl.dispatch([row], root, ledger, "native", command, max_workers=1, tail=15, poll_seconds=1)
        barrier = root/"barrier"
        barrier.mkdir()
        rows = [dict(id=root.name+"_parallel_"+name, spec=m.Spec(name, "CAB", 4).identity(), allowance=1200.)
                for name in ("S_tuned", "H_pos_tuned", "LOGIT0.1_ref")]
        def command(q, job, deadline):
            m.data.write_json(job/"spec.json", q["spec"])
            return [sys.executable, "-B", str(Path(__file__)), "native", "--job", str(job),
                    "--spec", str(job/"spec.json"), "--deadline", str(deadline), "--barrier", str(barrier)]
        ctl.dispatch(rows, root, ledger, "native", command, tail=15, poll_seconds=1)
        results = {p.parent.name: m.data.read_json(p) for p in sorted(root.glob("*/result.json"))}
        if len(results) != 5 or sum(r["actual_calls"] for r in results.values()) != 15:
            raise ValueError("Native complete-call coverage differs")
        all_attempted = sum(m.data.read_json(p)["attempted_calls"] for p in root.parent.glob("native*/**/progress.json"))
        if all_attempted > 36:
            raise ValueError("Aggregate qualification call limit exceeded")
        result = dict(status="PASS_LIMITED_THREE_WAY_NATIVE", results=results, ledger=ledger.snapshot(),
                      total_attempted_calls=all_attempted, sources=run.sources())
        m.data.write_json(root/"result.json", result)
        return result
    except BaseException as error:
        m.data.write_json(root/"failure.json", dict(error=repr(error), traceback=traceback.format_exc(), ledger=ledger.snapshot()))
        raise
    finally:
        stop.set()
        thread.join(timeout=10)
        ledger.charge("control", sum(monitor_seconds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cpu", "native", "native-batch"))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--barrier", type=Path)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    if args.mode == "cpu":
        sys.path.insert(0, str(m.data.ROOT/"tests"))
        os.environ["FIVE_TEST_ROOT"] = str(args.job)
        names = os.environ.get("FIVE_TEST_NAMES", "test_step28_continual_five").split(",")
        suite = unittest.defaultTestLoader.loadTestsFromNames(names)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        m.data.write_json(args.job/"result.json", dict(status="PASS_CPU" if result.wasSuccessful() else "FAIL_CPU",
            tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), elapsed_seconds=time.monotonic()-STARTED))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.mode == "native":
        try:
            native(args.job, m.Spec.from_dict(m.data.read_json(args.spec)), args.deadline, args.barrier)
        except BaseException as error:
            m.data.write_json(args.job/"failure.json", dict(error=repr(error), traceback=traceback.format_exc()))
            raise
    else:
        native_batch(args.job, args.ledger)


if __name__ == "__main__":
    main()
