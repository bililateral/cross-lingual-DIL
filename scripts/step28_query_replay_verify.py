"""Cumulative Linux handwritten CPU/native qualification; no formal inputs."""
from __future__ import annotations

import time
STARTED = time.monotonic()

import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
import threading
import traceback
import unittest

import numpy as np
import psutil
import torch

import step28_query_replay_run as run
from step28_record_replay_verify import preflight
from step28_record_replay_numerics import cpu_copy, equal_state

method, record, diag, core, data = run.method, run.record, run.diag, run.core, run.data


def state(model, optimizer):
    return cpu_copy({"model": model.state_dict(), "adam": optimizer.state_dict(),
                     "grads": [p.grad for p in model.parameters()]})


def restore(model, optimizer, saved, rng, modes):
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(cpu_copy(saved["adam"]))
    for p, g in zip(model.parameters(), saved["grads"], strict=True):
        p.grad = None if g is None else g.to(p.device)
    for m, mode in zip(model.modules(), modes, strict=True):
        m.training = mode
    diag.restore_rng(rng)


def component_effects(model, on):
    """Compare active paths; unused parameters must keep both mask and value."""
    parameters = list(model.named_parameters())
    effects = {}
    for component in ("encoder", "head"):
        active, inactive = [], []
        for i, (name, parameter) in enumerate(parameters):
            if not name.startswith(component+"."):
                continue
            gradient, previous = parameter.grad, on["grads"][i]
            if (gradient is None) != (previous is None):
                raise ValueError("On/off gradient participation changed: "+name)
            if gradient is None:
                if not torch.equal(on["model"][name], parameter.detach().cpu()):
                    raise ValueError("Unused parameter changed: "+name)
                inactive.append(name)
            else:
                if not torch.isfinite(gradient).all() or not torch.isfinite(previous).all():
                    raise ValueError("Nonfinite on/off gradient: "+name)
                active.append((i, name, parameter))
        gradient_delta = max((float((on["grads"][i]-p.grad.detach().cpu()).abs().max()) for i, n, p in active), default=0.)
        parameter_delta = max((float((on["model"][n]-p.detach().cpu()).abs().max()) for i, n, p in active), default=0.)
        if not active or gradient_delta == 0 or parameter_delta == 0:
            raise ValueError("No distinguishable encoder/head active update effect")
        effects[component] = {"post_clip_gradient_distinguishable": True, "actual_update_distinguishable": True,
            "finite": True, "active_grad_parameters": len(active), "inactive_grad_parameters": inactive,
            "active_parameter_names": [n for i, n, p in active],
            "post_clip_gradient_max_delta": gradient_delta, "parameter_max_delta": parameter_delta}
    return effects


def checked_fresh(c):
    archive = run.pretrained_archive(c)
    model, optimizer, initial = run.fresh(c)
    return model, optimizer, initial, archive


def capture_handwritten_diagnostic(model, optimizer, fit, memory, valid, c, arm, out, check=lambda: None):
    root = out/"diagnostics"
    root.mkdir()
    return diag.capture(model, optimizer, fit, memory, valid, c, arm, "ABC", 2, 1,
                        root, "handwritten", check, scope="AB", table_diagnostic=method.diagnostic)


def native(check, progress, handwritten, out):
    resources = preflight()
    if shutil.disk_usage(out).free < run.policy()["native_verification"]["maximum_work_bytes"]:
        raise ValueError("Insufficient native temporary disk")
    numerics = run.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory, 0)
    measurements, updates = {}, []
    def timed(name, fn):
        check()
        torch.cuda.synchronize()
        start = time.monotonic()
        value = fn()
        torch.cuda.synchronize()
        measurements[name] = max(measurements.get(name, 0.), time.monotonic()-start)
        return value
    c = run.config()
    progress("load_native_BGE")
    model, optimizer, initial, archive = timed("L", lambda: checked_fresh(c))
    if not model.encoder[0].auto_model.is_gradient_checkpointing or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16/checkpointing required")
    current, history = [handwritten.handmade_group("native_"+s, 8, 254) for s in ("current", "history")]
    for g in (current, history):
        lengths = [len(v) for v in model.encoder.tokenizer(run.base.record_texts(g, "separate_moments"), padding=False, truncation=False)["input_ids"]]
        if len(lengths) != 448 or set(lengths) != {256}:
            raise ValueError("Maximum N224/token256 qualification fixture differs")
    progress("first_S_current_update")
    updates.append(timed("t_A", lambda: method.update(model, optimizer, current, None, None, c, "S", 1, 1, check)))
    progress("first_complete", actual_updates=len(updates), completed_updates=updates, measurements=measurements)
    if run.parent.adam_step(optimizer) != 1:
        raise ValueError("First optimizer update missing")
    with diag.neutral(model):
        full_table = timed("f", lambda: record.reference(model, history, c, check))
        timed("f", lambda: record.score(model, [current], c, check))
    # Explicit handwritten active-response target, not claimed to be trained teacher scores.
    owner = record.assignment(history)
    pattern = np.sin(np.arange(len(owner), dtype=np.float64))
    for a in range(28):
        pattern[owner == a] -= pattern[owner == a].mean()
    i, j = np.triu_indices(len(owner), 1)
    target = (full_table+3*(pattern[i]+pattern[j])).astype(np.float32)
    def table_diagnostics():
        rows = []
        for seed in (None, data.seed_for(20261008, "ABC", 2, history.uid, "diagnostic_regroup")):
            slots = record.assignment(history, seed)
            rows.append((record.account_logits(torch.from_numpy(full_table), slots)-record.account_logits(torch.from_numpy(target), slots)).numpy().astype(np.float64))
        e0, e1 = rows
        return {"D0": float(np.mean(e0**2)), "D1": float(np.mean(e1**2)),
                "residual_cross_mean": float(np.mean(e0*e1)), "residual_difference_mse": float(np.mean((e1-e0)**2)),
                **method.diagnostic(full_table, history, target)}
    target_details = timed("h", table_diagnostics)
    if target_details["R_q"] <= 0:
        raise ValueError("Handwritten response witness is inactive")
    for value in optimizer.state.values():
        value["step"].fill_(336)  # One real allocation; synthetic epoch-1 continuation count.
    model.train()
    model.head.eval()
    baseline, rng, modes = state(model, optimizer), diag.rng_state(), [m.training for m in model.modules()]
    memory = handwritten.shared_memory()
    memory.begin_stage(2)
    for _ in range(48):
        memory.draw()
    payload = memory.to_bytes()
    fit = [handwritten.handmade_group(f"native_fit_{i}") for i in range(48)]
    valid = [data.Group(g.uid, g.sellers, g.items) for g in [handwritten.handmade_group(f"native_valid_{i}") for i in range(40)]]
    on, off, outcomes, effects, equality, query_effect = None, None, {}, {}, {}, {}
    for arm, label, timing in (("S","S","t_S"),("Q","Q","t_Q"),("C_q","on","t_q"),
                               ("C_b","bootstrap","t_b"),("C_q","repeat","t_q")):
        restore(model, optimizer, baseline, rng, modes)
        if label == "repeat":
            progress("actual_native_epoch_diagnostic")
            before_hash = core.state_digest(state(model, optimizer))
            capture_handwritten_diagnostic(model, optimizer, fit, memory, valid, c, arm, out, check)
            if (before_hash != core.state_digest(state(model, optimizer)) or rng != diag.rng_state()
                    or modes != [m.training for m in model.modules()] or payload != memory.to_bytes()):
                raise ValueError("Actual diagnostic changed model/Adam/grad/RNG/modes/Memory")
        progress("paired_"+label)
        row = timed(timing, lambda: method.update(model, optimizer, current, history, target, c, arm, 2, 49, check))
        updates.append(row)
        outcomes[label] = row
        progress("paired_"+label+"_complete", actual_updates=len(updates), completed_updates=updates, measurements=measurements)
        if run.parent.adam_step(optimizer) != 337:
            raise ValueError("Paired native Adam continuation differs")
        if label == "S":
            original = state(model, optimizer)
        elif label == "Q":
            difference = row["total"]-outcomes["S"]["total"]
            if difference == 0:
                raise ValueError("Handwritten Q/S objective witness is inactive")
            query_effect = {"target_difference": difference, "effects": component_effects(model, original)}
            del original
            off = state(model, optimizer)
        elif label == "on":
            if (row["history"]["response"] <= 0 or row["total"] == outcomes["Q"]["total"]
                    or row["history"]["assignment_sha256"] != outcomes["Q"]["history"]["assignment_sha256"]):
                raise ValueError("Response inactive or Q/C_q allocations differ")
            effects = component_effects(model, off)
            del off
            on, on_rng = state(model, optimizer), diag.rng_state()
        elif label == "repeat":
            equality = {"tensors": 0, "elements": 0}
            equal_state(on, state(model, optimizer), equality)
            if on_rng != diag.rng_state() or row != outcomes["on"] or memory.to_bytes() != payload:
                raise ValueError("Actual diagnostic changed next candidate update")
        check()
    if len(updates) != 6:
        raise ValueError("Six physical native updates required")
    tensors = [*model.parameters(), *(p.grad for p in model.parameters() if p.grad is not None),
               *(s[k] for s in optimizer.state.values() for k in ("exp_avg", "exp_avg_sq"))]
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in tensors):
        raise ValueError("Finite FP32 parameters/gradients/Adam required")
    del baseline, on, tensors
    gc.collect()
    meta = {"fixture": "handwritten native", "rng": diag.rng_state(), "modes": [m.training for m in model.modules()],
            "memory_sha256": __import__("hashlib").sha256(payload).hexdigest()}
    work = out/"work"
    work.mkdir()
    full_path, infer_path = work/"full.pt", work/"inference.pt"
    progress("actual_checkpoint_IO_and_restore")
    with diag.neutral(model):
        before_scores = timed("f", lambda: record.score(model, [history], c, check))
    full = timed("s_F", lambda: core.save_state(full_path, model, optimizer, meta))
    if timed("r_F", lambda: core.restore_state(full_path, model, optimizer, full["state_sha256"])) != meta:
        raise ValueError("Full restoration failed")
    with diag.neutral(model):
        if not np.array_equal(before_scores, timed("f", lambda: record.score(model, [history], c, check))):
            raise ValueError("Full restored scores differ")
    inference = timed("s_I", lambda: core.save_state(infer_path, model, None, meta))
    if timed("r_I", lambda: core.restore_state(infer_path, model, None, inference["state_sha256"])) != meta:
        raise ValueError("Inference restoration failed")
    with diag.neutral(model):
        if not np.array_equal(before_scores, timed("f", lambda: record.score(model, [history], c, check))):
            raise ValueError("Inference restored scores differ")
    measurements.update(W_F=full_path.stat().st_size, W_I=infer_path.stat().st_size)
    removal = {"full": full, "inference": inference, "purpose": "qualification IO verified; no later dependency", "removed_after_verification": False}
    data.write_json(out/"temporary_checkpoint_identity.json", removal)
    full_path.unlink()
    infer_path.unlink()
    work.rmdir()
    removal["removed_after_verification"] = not full_path.exists() and not infer_path.exists()
    data.write_json(out/"temporary_checkpoint_identity.json", removal)
    result = {"resources": resources, "numerics": numerics, "actual_updates": 6, "updates": updates,
        "pretrained_archive": archive, "initial_state": initial,
        "measurements": measurements, "projection": run.projection(measurements), "active_handwritten_response": target_details,
        "new_term_encoder_head_effect": True, "effects": effects, "next_update_bitwise_equal": True, "equality_counts": equality,
        "query_encoder_head_effect": True, "Q_minus_S": query_effect,
        "C_q_minus_Q_target_difference": outcomes["on"]["total"]-outcomes["Q"]["total"],
        "effect_scope": "Actual Q versus S and C_q versus Q updates and post-clip gradients; independent direct VJP separately on CPU",
        "memory_bytes": len(payload), "maximum_records": 224, "channel_tokens": 256,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "fixture": "One actual S allocation plus five paired updates from one explicitly synthetic counter-336 state; full real epoch diagnostic before repeat. No formal inputs."}
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
    parser.add_argument("--cpu-tests", nargs="+", help="Affected QueryReplayTests methods; prior evidence is retained separately")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if os.name != "posix" or args.output.exists() or args.prior_seconds < 0:
        parser.error("Linux, new evidence and cumulative ledger required; GPU has no automatic retry")
    if args.cpu_tests and args.mode != "cpu":
        parser.error("Selected tests are only for affected CPU qualification")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(data.ROOT/"tests"))
    import test_step28_query_replay as handwritten
    os.environ["QUERY_REPLAY_TEST_ROOT"] = str(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    numerics = run.parent.configure_numerics()
    process = psutil.Process()
    limits = run.policy()["cpu_verification" if args.mode == "cpu" else "native_verification"]
    report = {"status": "RUNNING", "mode": args.mode, "source_files": run.sources(), "formal_inputs": False, "torch": torch.__version__,
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
    def progress(stage, **details):
        report.update(stage=stage, **details)
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
            available = unittest.defaultTestLoader.getTestCaseNames(handwritten.QueryReplayTests)
            selected = args.cpu_tests or available
            if len(set(selected)) != len(selected) or any(name not in available for name in selected):
                raise ValueError("Unknown or duplicate affected CPU test")
            report["selected_tests"] = selected
            suite = unittest.TestSuite(handwritten.QueryReplayTests(name) for name in selected)
            with (args.output/"unittest.txt").open("w", encoding="utf-8") as stream:
                result = unittest.TextTestRunner(stream=stream, verbosity=2, failfast=True).run(suite)
            report.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
            if not result.wasSuccessful() or result.skipped:
                raise RuntimeError("Affected handwritten qualification failed")
        else:
            if args.gate is None:
                raise ValueError("Implementation review gate required")
            gate = run.reviewed_gate(args.gate, "NATIVE")
            if args.prior_seconds != gate.get("native_prior_seconds", 0):
                raise ValueError("Prior GPU budget must match the explicitly approved attempt gate")
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
