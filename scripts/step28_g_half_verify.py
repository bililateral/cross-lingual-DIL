"""Cumulative Linux handwritten CPU/native qualification; no formal inputs."""
from __future__ import annotations

import time
import os
import math
STARTED = time.monotonic()-max(0.,time.time()-float(os.environ.get("QUALIFICATION_STARTED_EPOCH",time.time())))

import argparse
import gc
from pathlib import Path
import shutil
import sys
import threading
import traceback
import unittest

import numpy as np
import psutil
import torch

import step28_g_half_run as run
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


def adam_effects(model, optimizer, saved):
    """Record on/off Adam states by actual optimizer parameter ID and model name."""
    current = optimizer.state_dict()
    if [g["params"] for g in current["param_groups"]] != [g["params"] for g in saved["param_groups"]]:
        raise ValueError("On/off Adam parameter mapping differs")
    names = {id(p): name for name, p in model.named_parameters()}
    result = {component: {"finite": True, "parameters": []} for component in ("encoder", "head")}
    for live, packed in zip(optimizer.param_groups, current["param_groups"], strict=True):
        for parameter, pid in zip(live["params"], packed["params"], strict=True):
            name = names[id(parameter)]
            component = name.split(".", 1)[0]
            before, after = saved["state"].get(pid), current["state"].get(pid)
            row = {"name": name, "optimizer_parameter_id": pid, "state_present": after is not None}
            if (before is None) != (after is None) or (after is None and parameter.grad is not None):
                raise ValueError("On/off Adam participation differs: "+name)
            if after is not None:
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    if key not in before or key not in after:
                        raise ValueError("Missing on/off Adam field: "+name+"/"+key)
                    a, b = before[key], after[key].detach().cpu()
                    if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                        raise ValueError("Invalid on/off Adam field: "+name+"/"+key)
                    row[key] = {"max_abs_delta": float((a-b).abs().max())}
                    if key == "step":
                        row[key].update(off=float(a), on=float(b))
            result[component]["parameters"].append(row)
    return result


def component_effects(model, on, *, optimizer=None):
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
    if optimizer is not None:
        for component, value in adam_effects(model, optimizer, on["adam"]).items():
            effects[component]["adam"] = value
    return effects


def checked_fresh(c):
    archive = run.pretrained_archive(c)
    model, optimizer, initial = run.fresh(c)
    return model, optimizer, initial, archive


def disk_reserve(path, total, reserve, maximum):
    if total+reserve > maximum or shutil.disk_usage(path).free < reserve:
        raise RuntimeError("Qualification checkpoint reserve exceeds available disk or approved work bytes")


def native_save(path, model, optimizer, metadata, check):
    check(run.base.persistence.checkpoint_reserve(model, optimizer))
    return core.save_state(path, model, optimizer, metadata)


def native(check, progress, handwritten, out, *, entry=None, consistency=False):
    run = entry or sys.modules["step28_g_half_run"]
    method = run.method
    active, control = method.ARMS[0], "H" if consistency else "G"
    import step28_balanced_pooling as original
    resources = preflight()
    if shutil.disk_usage(out).free < 10*2**30:
        raise ValueError("Insufficient native temporary disk")
    numerics = run.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    measurements, updates = {}, []
    def timed(name, fn):
        check(); torch.cuda.synchronize(); started = time.monotonic()
        value = fn(); torch.cuda.synchronize()
        measurements[name] = max(measurements.get(name,0.),time.monotonic()-started)
        return value
    c = run.config()
    if consistency:
        c["handwritten_qualification"] = True
    model, optimizer, initial, archive = timed("L", lambda: checked_fresh(c))
    if not model.encoder[0].auto_model.is_gradient_checkpointing or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16/checkpointing required")
    if consistency and initial != {"model": "13b535e1d66a0f13f0d87461835bd3ba0cee69c531307884b471f8702bd14e4b",
            "adam": "de5b1864dc195dd65286e083567e3fd8bb6a146a9ecfcc3bae6b3fdda8c32d20"}:
        raise ValueError("H_cons native initial identity differs")
    method.bind(model, method.metadata())
    current, history = [handwritten.handmade_group("native_"+s,8,254) for s in ("current","history")]
    for g in (current,history):
        lengths = [len(v) for v in model.encoder.tokenizer(run.base.record_texts(g,"separate_moments"),padding=False,truncation=False)["input_ids"]]
        if len(lengths)!=448 or set(lengths)!={256}:
            raise ValueError("Maximum N224/token256 fixture differs")
    probes = {"encoder":model.encoder[0].auto_model.encoder.layer[0].attention.self.query.weight,"head":model.head[0].weight}
    gradients, effects, table_checks, objective_pairs = {}, {}, [], {}
    def updated(arm, phase, step, h=None, target=None):
        if len(updates)>=6: raise ValueError("Six actual native calls exhausted")
        pre = {}
        def observe(role, log):
            if role == ("current" if h is None else "history"):
                for key,p in probes.items():
                    if p.grad is None or not torch.isfinite(p.grad).all(): raise ValueError("Missing native gradient")
                    pre[key] = p.grad.detach().cpu().clone()
        scorer = method if consistency or arm == active else original
        captured = []
        options = {}
        if consistency:
            options = dict(qualification_without_c=arm == control,
                table_observer=(lambda table, group, seed: captured.append((table.clone(), group, seed))) if arm == active else None)
        progress("native_call_started", attempted_calls=len(updates)+1, actual_updates=len(updates))
        row = timed(("t_"+phase) if arm == active else "control_"+phase,
            lambda: scorer.update(model,optimizer,current,h,target,c,active if consistency else arm,
                                  1 if h is None else 2,step,check,observe,**options))
        updates.append(dict(arm=arm,phase=phase,step=step,log=row))
        gradients[len(updates)] = pre
        progress("native_update",actual_updates=len(updates),updates=updates,measurements=measurements)
        for table, group, seed in captured:
            table_checks.append(handwritten.verify_native_table(table, group, seed))
            progress("native_table_precision", table_precision=table_checks)
        return row
    # The only actual allocation update. Subsequent fixture clocks are explicit, not claimed training.
    updated(control,"allocation",1)
    work=out/"work";work.mkdir(); identities={}
    def metadata(label):
        return dict(fixture=label,**method.metadata(),rng=diag.rng_state(),modes=[m.training for m in model.modules()])
    common_path=work/"common.pt";common_meta=metadata("One actual "+control+" update; common current fixture Adam step1")
    common = timed("s_F",lambda:native_save(common_path,model,optimizer,common_meta,check))
    identities["common"] = common
    def reload(path,receipt):
        return timed("r_F",lambda:run.restore_point(path,receipt,model,optimizer,active))
    reload(common_path,common); updated(control,"A",2); gcurrent=state(model,optimizer)
    reload(common_path,common); hrow=updated(active,"A",2)
    effects["current"] = component_effects(model,gcurrent,optimizer=optimizer if consistency else None)
    for key in probes:
        delta=float((gradients[2][key]-gradients[3][key]).abs().max())
        if delta<=0:raise ValueError("Candidate-control preclip current gradient not distinguishable")
        effects["current"][key]["pre_clip_gradient_max_delta"]=delta
    progress("native_effects", effects=effects)
    if consistency:
        objective_pairs["current"] = run.native_objective_pair(updates[1]["log"], hrow, 1)
        progress("native_objective_pair", objective_pairs=objective_pairs)
    elif updates[1]["log"]["total"]==hrow["total"]:
        raise ValueError("Current candidate-control objective indistinguishable")
    del gcurrent;gc.collect()
    reload(common_path,common)
    # Same trained-one-step parameters/moments, but legal B epoch1 counter336 for the paired fixture.
    for value in optimizer.state.values(): value["step"].fill_(336)
    with diag.neutral(model):table=timed("v",lambda:record.reference(model,history,c,check))
    i,j=np.triu_indices(224,1);target=(table+2*np.sin(i*.31+j*.71)).astype(np.float32)
    paired_path=work/"paired.pt";paired_meta=metadata("Synthetic Adam step336; parameters/moments from one real "+control+" allocation update")
    paired=timed("s_F",lambda:native_save(paired_path,model,optimizer,paired_meta,check));identities["paired"]=paired
    reload(paired_path,paired);updated(control,"B",49,history,target);gpaired=state(model,optimizer)
    reload(paired_path,paired);first=updated(active,"B",49,history,target)
    if first["history"]["mse0"]<=0 or first["history"]["supervised"]<=0:raise ValueError("Inactive D/history")
    effects["paired"]=component_effects(model,gpaired,optimizer=optimizer if consistency else None)
    for key in probes:
        delta=float((gradients[4][key]-gradients[5][key]).abs().max())
        if delta<=0:raise ValueError("Candidate-control preclip paired gradient not distinguishable")
        effects["paired"][key]["pre_clip_gradient_max_delta"]=delta
    progress("native_effects", effects=effects)
    if consistency:
        objective_pairs["paired"] = run.native_objective_pair(updates[3]["log"], first, 2)
        progress("native_objective_pair", objective_pairs=objective_pairs)
    elif updates[3]["log"]["total"]==first["total"]:
        raise ValueError("Paired candidate-control objective indistinguishable")
    del gpaired;gc.collect()
    expected, expected_rng = state(model,optimizer), diag.rng_state()
    reload(paired_path,paired)
    memory=handwritten.shared_memory();memory.begin_stage(2)
    for _ in range(48):memory.draw()
    payload=memory.to_bytes();rng=diag.rng_state();modes=[m.training for m in model.modules()]
    fit=[handwritten.handmade_group(f"native_fit_{i}") for i in range(48)]
    valid=[data.Group(g.uid,g.sellers,g.items) for g in [handwritten.handmade_group(f"native_valid_{i}") for i in range(40)]]
    before=core.state_digest(state(model,optimizer));droot=out/"diagnostics";droot.mkdir()
    progress("actual_H_full_role_capture", actual_updates=len(updates))
    method.capture(model,optimizer,fit,memory,valid,c,active,2,1,droot,"handwritten",check)
    if before!=core.state_digest(state(model,optimizer)) or rng!=diag.rng_state() or payload!=memory.to_bytes():
        raise ValueError("Actual H capture mutated state")
    repeated=updated(active,"B",49,history,target)
    equality={"tensors":0,"elements":0};equal_state(expected,state(model,optimizer),equality)
    if expected_rng!=diag.rng_state() or payload!=memory.to_bytes() or modes!=[m.training for m in model.modules()] or first!=repeated:
        raise ValueError("Capture changed repeated next update")
    del expected;gc.collect()
    if len(updates)!=6 or run.parent.adam_step(optimizer)!=337:raise ValueError("Native call accounting differs")
    def residual():
        errors=[]
        for seed in (None,data.seed_for(20261008,"ABC",2,history.uid,"diagnostic_regroup")):
            slots=record.assignment(history,seed)
            errors.append((method.account_logits(torch.from_numpy(table),slots,active)-
                           method.account_logits(torch.from_numpy(target),slots,active)).numpy().astype(np.float64))
        return dict(D0=float(np.mean(errors[0]**2)),D1=float(np.mean(errors[1]**2)))
    residuals=timed("h",residual)
    # Persist the actual H-updated state, releasing the obsolete common work file first.
    data.write_json(out/"temporary_checkpoint_identity.json",dict(files=identities,removed=False))
    common_path.unlink()
    actual_meta=metadata("After actual H update6; H full-state restore")
    actual=timed("s_F",lambda:native_save(common_path,model,optimizer,actual_meta,check))
    identities["H_full"]=actual
    with diag.neutral(model):
        timed("v",lambda:record.reference(model,history,c,check))
        before_scores=timed("v",lambda:method.restored_score(model,[history],c,check))
    if reload(common_path,actual)!=actual_meta:raise ValueError("Full H metadata restore mismatch")
    with diag.neutral(model):
        if not np.array_equal(before_scores,timed("v",lambda:method.restored_score(model,[history],c,check))):
            raise ValueError("Full H scores not bitwise restored")
    inference_path=work/"H_inference.pt";infer_meta=metadata("Actual H inference restore fixture")
    infer=timed("s_I",lambda:native_save(inference_path,model,None,infer_meta,check));identities["inference"]=infer
    if timed("r_I",lambda:run.restore_point(inference_path,infer,model,None,active))!=infer_meta:
        raise ValueError("Inference H metadata restore mismatch")
    with diag.neutral(model):
        if not np.array_equal(before_scores,timed("v",lambda:method.restored_score(model,[history],c,check))):
            raise ValueError("Inference H scores not bitwise restored")
    measurements["W_F"]=max(common_path.stat().st_size,paired_path.stat().st_size)
    measurements["W_I"]=inference_path.stat().st_size
    data.write_json(out/"temporary_checkpoint_identity.json",dict(files=identities,removed=False))
    for path in (common_path,paired_path,inference_path):path.unlink()
    work.rmdir();data.write_json(out/"temporary_checkpoint_identity.json",dict(files=identities,removed=True))
    return dict(resources=resources,numerics=numerics,pretrained_archive=archive,initial_state=initial,
        actual_updates=6,updates=updates,measurements=measurements,projection=run.projection(measurements),effects=effects,
        both_phases_encoder_head_effect=True,H_pooling_restores_verified=True,next_update_bitwise_equal=True,
        training_identity=method.metadata().get("training"), consistency_precision_verified=bool(table_checks),
        table_precision=table_checks, objective_pairs=objective_pairs,
        equality_counts=equality,maximum_records=224,channel_tokens=256,memory_bytes=len(payload),residuals=residuals,
        fixture="Six actual updates only. Common model/moments after call1; paired counter336 explicitly synthetic. Calls5/6 differ only by actual capture. No formal input.")


def main(*, entry=None, handwritten_module="test_step28_g_half", test_class_name="GHalfTests", consistency=False):
    import importlib
    run = entry or sys.modules["step28_g_half_run"]
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--tests", nargs="*")
    parser.add_argument("--prior-seconds", type=float, default=0.)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if os.name != "posix" or args.output.exists() or args.prior_seconds < 0:
        parser.error("Linux, new evidence and cumulative ledger required; GPU has no automatic retry")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(data.ROOT/"tests"))
    handwritten = importlib.import_module(handwritten_module)
    os.environ["G_HALF_TEST_ROOT"] = str(args.output)
    os.environ["H_CONS_TEST_ROOT"] = str(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    numerics = run.parent.configure_numerics()
    process = psutil.Process()
    limits = run.policy()["cpu_verification" if args.mode == "cpu" else "native_verification"]
    report = {"status": "RUNNING", "mode": args.mode, "scope": run.method.ARMS[0]+"_increment", "source_files": run.sources(), "formal_inputs": False, "torch": torch.__version__,
        "formal_labels": False, "prior_compute_seconds": args.prior_seconds, "peak_rss_bytes": 0, "peak_work_bytes": 0, "numerics": numerics}
    stopped, lock = threading.Event(), threading.RLock()
    def write():
        with lock:
            report["elapsed_seconds"] = time.monotonic()-STARTED
            report["cumulative_seconds"] = args.prior_seconds+report["elapsed_seconds"]
            report["cumulative_charged_seconds"] = args.prior_seconds+math.ceil(report["elapsed_seconds"])
            if "prior_actual_updates" in report:
                report["cumulative_actual_updates"] = report["prior_actual_updates"]+report.get("actual_updates", 0)
            if args.mode == "gpu" and torch.cuda.is_initialized():
                report["max_reserved_bytes"] = torch.cuda.max_memory_reserved()
            data.write_json(args.output/"result.json", report)
    def check(reserve=0):
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
        disk_reserve(args.output, total, reserve, limits.get("maximum_work_bytes", limits["maximum_evidence_bytes"]))
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
        if args.prior_seconds < limits["prior_charged_seconds"]:
            raise ValueError("Prior charged budget cannot be reset")
        if args.mode == "cpu":
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise ValueError("CPU qualification must disable GPU")
            test_class = getattr(handwritten, test_class_name)
            selected = args.tests or unittest.defaultTestLoader.getTestCaseNames(test_class)
            report["selected_tests"] = selected
            suite = unittest.TestSuite(test_class(name) for name in selected)
            with (args.output/"unittest.txt").open("w", encoding="utf-8") as stream:
                result = unittest.TextTestRunner(stream=stream, verbosity=2, failfast=True).run(suite)
            report.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
            if not result.wasSuccessful() or result.skipped:
                raise RuntimeError("Affected handwritten qualification failed")
        else:
            if args.gate is None:
                raise ValueError("Implementation review gate required")
            gate = run.reviewed_gate(args.gate, "NATIVE")
            if consistency:
                report["prior_actual_updates"] = run.validate_native_requalification(gate, args.prior_seconds, args.output)
            elif args.prior_seconds != 0:
                raise ValueError("Unique GPU attempt; no automatic retry")
            report["native"] = native(check, progress, handwritten, args.output, entry=run, consistency=consistency)
            if consistency:
                run.validate_native(report["native"])
        check()
        report["status"] = "PASS_HANDWRITTEN_ONLY"
    except BaseException as exc:
        report.update(status="FAIL", traceback=traceback.format_exc())
        raise
    finally:
        stopped.set()
        watcher.join(timeout=2)
        write()


if __name__ == "__main__":
    main()
