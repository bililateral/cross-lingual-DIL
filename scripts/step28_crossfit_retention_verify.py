"""Cumulative Linux handwritten CPU/native qualification; no formal inputs."""
from __future__ import annotations

import time
STARTED = time.monotonic()

import argparse
import gc
import os
from pathlib import Path
import sys
import threading
import traceback
import unittest

import numpy as np
import psutil
import torch

import step28_crossfit_retention_run as run
from step28_record_replay_verify import preflight
from step28_record_replay_numerics import cpu_copy, equal_state

method, record, diag, core, data = run.method, run.record, run.diag, run.core, run.data


def state(model, optimizer):
    return cpu_copy({"model": model.state_dict(), "adam": optimizer.state_dict(),
                     "grads": [p.grad for p in model.parameters()]})


def restore(model, optimizer, saved, rng, modes):
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["adam"])
    for p, g in zip(model.parameters(), saved["grads"], strict=True):
        p.grad = None if g is None else g.to(p.device)
    for m, mode in zip(model.modules(), modes, strict=True):
        m.training = mode
    diag.restore_rng(rng)


def native(check, progress, handwritten, out):
    resources = preflight()
    numerics = run.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory, 0)
    measurements, updates = {}, []
    def timed(name, fn):
        check()
        torch.cuda.synchronize()
        start = time.monotonic()
        value = fn()
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        measurements[name] = max(measurements.get(name, 0.), elapsed)
        return value
    c = run.config()
    progress("load_native_BGE")
    model, optimizer, initial = timed("L", lambda: run.fresh(c))
    if not model.encoder[0].auto_model.is_gradient_checkpointing or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16/checkpointing mode missing")
    current, history = [handwritten.handmade_group("native_"+s, 8, 254) for s in ("current", "history")]
    for g in (current, history):
        lengths = [len(v) for v in model.encoder.tokenizer(run.base.record_texts(g, "separate_moments"), padding=False, truncation=False)["input_ids"]]
        if len(lengths) != 448 or set(lengths) != {256}:
            raise ValueError("Native fixture not the declared N224/token256 maximum")
    before = state(model, optimizer)
    modes0, rng0 = [m.training for m in model.modules()], diag.rng_state()
    progress("main_first_allocation")
    updates.append(timed("t_A", lambda: method.update(model, optimizer, current, None, None, c, "S", 1, 1, check)))
    if run.parent.adam_step(optimizer) != 1:
        raise ValueError("First actual main update missing")
    with diag.neutral(model):
        full_table = timed("f", lambda: record.reference(model, history, c, check))
        full_scores = record.account_logits(torch.from_numpy(full_table).to("cuda"), record.assignment(history)).cpu().numpy().copy()
    for s in optimizer.state.values():
        s["step"].fill_(288)  # Explicit synthetic continuation count, one real update's moments.
    model.train()
    model.head.eval()
    baseline = state(model, optimizer)
    rng, modes = diag.rng_state(), [m.training for m in model.modules()]
    restore(model, optimizer, before, rng0, modes0)
    if {"model": core.state_digest(model.state_dict()), "adam": core.state_digest(optimizer.state_dict())} != initial:
        raise ValueError("Handwritten source did not reset to pre-A/empty Adam")
    progress("source_first_lr_index_2")
    updates.append(timed("t_U", lambda: method.source_update(model, optimizer, current, c, "U0", 1, check)))
    if run.parent.adam_step(optimizer) != 1 or updates[-1]["encoder_lr"] != run.parent.stage_lr(2):
        raise ValueError("Source actual Adam/LR clocks confused")
    del before
    with diag.neutral(model):
        unlabeled = data.Group(history.uid, history.sellers, history.items)
        o = timed("f", lambda: diag.score(model, [unlabeled], c, "S", check))[0]
        # Exercise both evaluation roles, not two claimed fully trained native sources.
        o2 = timed("f", lambda: diag.score(model, [unlabeled], c, "S", check))[0]
    if not np.array_equal(o, o2):
        raise ValueError("Handwritten source role eval inconsistent")
    def birth():
        f = torch.from_numpy(full_scores).to("cuda")
        b, capacity, detail = method.birth(history, f, torch.from_numpy(o).to("cuda"))
        p = method.permute(history, b, capacity, np.random.Generator(np.random.PCG64(20261012)))
        loss, pd = method.floor_loss(f, history, p)
        if float(loss) or pd["violations"]:
            raise ValueError("Handwritten permutation birth not feasible")
        return {**detail, "changed": int(np.count_nonzero(p != b)), "full_source": "one main update", "outside_source": "one auxiliary update; role evaluation only"}
    born = timed("h", birth)
    floor = np.linspace(1., 2., 1016, dtype=np.float32)
    inactive = np.zeros(1016, np.float32)
    # Deliberately active term witness, separate from the honest source-birth above.
    outcomes, on = {}, None
    for arm, label, target in (("S", "S", full_table), ("C_cf", "on", floor),
                               ("C_cf", "off", inactive), ("C_cf", "repeat", floor), ("C_perm", "perm", floor[::-1].copy())):
        restore(model, optimizer, baseline, rng, modes)
        if label == "repeat":
            progress("native_neutral_diagnostic")
            before_hash = core.state_digest({"model": model.state_dict(), "adam": optimizer.state_dict(), "grad": [p.grad for p in model.parameters()]})
            with diag.neutral(model):
                for g in (current, history):
                    v = timed("f", lambda: diag.score(model, [g], c, arm, check))
                    if g.uid == history.uid:
                        method.floor_loss(torch.from_numpy(v[0]), g, floor)
            if (before_hash != core.state_digest({"model": model.state_dict(), "adam": optimizer.state_dict(), "grad": [p.grad for p in model.parameters()]})
                    or rng != diag.rng_state() or modes != [m.training for m in model.modules()]):
                raise ValueError("Native diagnostic changed model/Adam/grad/RNG/modes")
        progress("paired_"+label)
        row = timed("t_S" if arm == "S" else "t_perm" if arm == "C_perm" else "t_cf",
                    lambda: method.update(model, optimizer, current, history, target, c, arm, 2, 1, check))
        updates.append(row)
        if run.parent.adam_step(optimizer) != 289:
            raise ValueError("Actual native paired update missing")
        outcomes[label] = row
        if label == "on":
            on, on_rng = state(model, optimizer), diag.rng_state()
            if row["history"]["violations"] <= 0 or row["history"]["distillation"] <= 0:
                raise ValueError("New-term fixture was not active")
        elif label == "off":
            effects = {}
            parameters = dict(model.named_parameters())
            for component in ("encoder", "head"):
                selected = [(i, n, p) for i, (n, p) in enumerate(parameters.items()) if n.startswith(component+".")]
                grad_changed = any(not torch.equal(on["grads"][i], p.grad.detach().cpu()) for i, n, p in selected)
                update_changed = any(not torch.equal(on["model"][n], p.detach().cpu()) for i, n, p in selected)
                finite = all(torch.isfinite(p.grad).all() for i, n, p in selected)
                if not grad_changed or not update_changed or not finite:
                    raise ValueError("New floor has no distinguishable finite encoder/head update effect")
                effects[component] = {"added_gradient_distinguishable": grad_changed, "actual_update_distinguishable": update_changed, "finite": bool(finite)}
        elif label == "repeat":
            equality = {"tensors": 0, "elements": 0}
            equal_state(on, state(model, optimizer), equality)
            if on_rng != diag.rng_state() or row != outcomes["on"]:
                raise ValueError("Inserted diagnostic changed next actual loss/RNG")
        check()
    if len(updates) != 7:
        raise ValueError("Native physical update scope differs")
    tensors = [*model.parameters(), *(p.grad for p in model.parameters()), *(s[k] for s in optimizer.state.values() for k in ("exp_avg", "exp_avg_sq"))]
    if any(t is None or t.dtype != torch.float32 or not torch.isfinite(t).all() for t in tensors):
        raise ValueError("Native parameters/grad/Adam must be finite FP32")
    del baseline, on, tensors
    gc.collect()
    memory = handwritten.shared_memory()
    memory.maps = {"first": {"a": 1., "b": 0.}, "stage": {"a": 1., "b": 0.}, "rng": diag.rng_state(), "completed_stage": 1, "adam_step": 288}
    own = method.Memory.from_shared(memory, "C_cf", {g.uid: floor.copy() for g in memory.reservoir.groups},
                                    handwritten.source_ids(memory))
    payload = own.to_bytes()
    if method.Memory.from_bytes(payload).to_bytes() != payload:
        raise ValueError("Native own Memory roundtrip differs")
    meta = {"fixture": "handwritten native only", "rng": diag.rng_state(), "modes": [m.training for m in model.modules()],
            "memory_sha256": __import__("hashlib").sha256(payload).hexdigest()}
    work = out/"work"
    work.mkdir()
    full_path, infer_path = work/"full.pt", work/"inference.pt"
    progress("actual_checkpoint_IO_and_restore")
    full = timed("s_F", lambda: core.save_state(full_path, model, optimizer, meta))
    if timed("r_F", lambda: core.restore_state(full_path, model, optimizer, full["state_sha256"])) != meta:
        raise ValueError("Full native restoration failed")
    inference = timed("s_I", lambda: core.save_state(infer_path, model, None, meta))
    if timed("r_I", lambda: core.restore_state(infer_path, model, None, inference["state_sha256"])) != meta:
        raise ValueError("Inference native restoration failed")
    measurements.update(W_F=full_path.stat().st_size, W_I=infer_path.stat().st_size)
    removal = {"full": full, "inference": inference, "purpose": "verified qualification IO; no subsequent dependency", "removed_after_verification": False}
    data.write_json(out/"temporary_checkpoint_identity.json", removal)
    full_path.unlink()
    infer_path.unlink()
    work.rmdir()
    removal["removed_after_verification"] = not full_path.exists() and not infer_path.exists()
    data.write_json(out/"temporary_checkpoint_identity.json", removal)
    result = {"resources": resources, "numerics": numerics, "actual_updates": 7, "updates": updates,
        "actual_source_birth": born, "measurements": measurements, "projection": run.projection(measurements),
        "new_term_encoder_head_effect": True, "effects": effects, "next_update_bitwise_equal": True, "equality_counts": equality,
        "own_memory_bytes": len(payload), "maximum_records": 224, "channel_tokens": 256,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "fixture": "One main and one auxiliary first update; five real paired updates from one synthetic counter-288 state. No formal inputs, no claim of 144-step native sources."}
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--prior-seconds", type=float, default=0.)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if os.name != "posix" or args.output.exists() or args.prior_seconds < 0 or (args.mode == "gpu" and args.prior_seconds):
        parser.error("Linux, new evidence and cumulative CPU ledger required; GPU has no automatic retry")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(data.ROOT/"tests"))
    import test_step28_crossfit_retention as handwritten
    os.environ["CROSSFIT_TEST_ROOT"] = str(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    numerics = run.parent.configure_numerics()
    process = psutil.Process()
    limits = run.policy()["cpu_verification" if args.mode == "cpu" else "native_verification"]
    report = {"status": "RUNNING", "mode": args.mode, "source_files": run.sources(), "formal_inputs": False,
        "formal_labels": False, "prior_compute_seconds": args.prior_seconds, "peak_rss_bytes": 0, "peak_work_bytes": 0, "numerics": numerics}
    stopped, lock = threading.Event(), threading.RLock()
    def write():
        with lock:
            report["elapsed_seconds"] = time.monotonic()-STARTED
            report["cumulative_seconds"] = args.prior_seconds+report["elapsed_seconds"]
            if args.mode == "gpu" and torch.cuda.is_initialized():
                report["max_reserved_bytes"] = torch.cuda.max_memory_reserved()
            data.write_json(args.output/"result.json", report)
    def check():
        report["peak_rss_bytes"] = max(report["peak_rss_bytes"], process.memory_info().rss)
        total, evidence = 0, 0
        for path in args.output.parent.rglob("*"):
            try:
                if path.is_file():
                    size = path.stat().st_size
                    total += size
                    if not path.is_relative_to(args.output/"work"):
                        evidence += size
            except FileNotFoundError:
                pass
        report["peak_work_bytes"] = max(report["peak_work_bytes"], total)
        if (time.monotonic()-STARTED+args.prior_seconds > limits["maximum_seconds"]
                or evidence > limits["maximum_evidence_bytes"] or total > limits.get("maximum_work_bytes", limits["maximum_evidence_bytes"])
                or report["peak_rss_bytes"] > limits["maximum_rss_bytes"]
                or (args.mode == "gpu" and torch.cuda.is_initialized() and torch.cuda.max_memory_reserved() > 28*2**30)):
            raise RuntimeError("Approved cumulative qualification budget exceeded")
    def watchdog():
        while not stopped.wait(.5):
            try:
                check()
            except BaseException as exc:
                report.update(status="BUDGET_STOP", error=str(exc))
                write()
                os._exit(2)
    def progress(stage):
        report["stage"] = stage
        write()
    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()
    try:
        if (len(process.cpu_affinity()) != 1 or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
                or os.environ.get("TOKENIZERS_PARALLELISM") != "false"):
            raise ValueError("Pre-import single CPU/thread environment required")
        if args.mode == "cpu":
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise ValueError("CPU qualification must disable GPU")
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(handwritten.CrossfitTests)
            with (args.output/"unittest.txt").open("w", encoding="utf-8") as stream:
                result = unittest.TextTestRunner(stream=stream, verbosity=2, failfast=True).run(suite)
            report.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
            if not result.wasSuccessful() or result.skipped:
                raise RuntimeError("Affected handwritten qualification failed")
        else:
            if args.gate is None:
                raise ValueError("Implementation review gate required")
            run.reviewed_gate(args.gate, "NATIVE")
            report["native"] = native(check, progress, handwritten, args.output)
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
