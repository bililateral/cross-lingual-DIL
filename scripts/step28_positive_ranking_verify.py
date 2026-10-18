"""Bounded dual positive-rank CPU/native qualification; no formal data."""
from __future__ import annotations
from step28_g_half_verify import STARTED, state, restore, component_effects, checked_fresh, disk_reserve, native_save
import argparse
import gc
import math
import os
import shutil
from pathlib import Path
import sys
import threading
import time
import traceback
import unittest
import numpy as np
import psutil
import torch
import step28_positive_ranking_run as run
from step28_record_replay_verify import preflight
from step28_record_replay_numerics import equal_state
method,record,diag,core,data=run.method,run.record,run.diag,run.core,run.data


def native(check, progress, handwritten, out, arm):
    method = run.method
    active, control = arm, arm+"_old_rank"
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
    c["handwritten_qualification"] = True
    model, optimizer, initial, archive = timed("L", lambda: checked_fresh(c))
    if not model.encoder[0].auto_model.is_gradient_checkpointing or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16/checkpointing required")
    if initial != {"model": "13b535e1d66a0f13f0d87461835bd3ba0cee69c531307884b471f8702bd14e4b",
            "adam": "de5b1864dc195dd65286e083567e3fd8bb6a146a9ecfcc3bae6b3fdda8c32d20"}:
        raise ValueError("positive-rank native initial identity differs")
    method.bind(model, method.metadata(active))
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
            pre[role] = {}
            for key,p in probes.items():
                if p.grad is None or not torch.isfinite(p.grad).all(): raise ValueError("Missing native gradient")
                pre[role][key] = p.grad.detach().cpu().clone()
        captured = []
        options = dict(qualification_old_rank=arm == control,
            table_observer=(lambda table, group, seed, reference: captured.append((table.clone(), group, seed, reference))) if arm == active else None)
        progress("native_call_started", attempted_calls=len(updates)+1, actual_updates=len(updates))
        row = timed(("t_"+phase) if arm == active else "control_"+phase,
            lambda: method.update(model,optimizer,current,h,target,c,active,
                                  1 if h is None else 2,step,check,observe,**options))
        updates.append(dict(arm=arm,phase=phase,step=step,log=row))
        gradients[len(updates)] = pre
        progress("native_update",actual_updates=len(updates),updates=updates,measurements=measurements)
        for table, group, seed, reference in captured:
            table_checks.append(handwritten.verify_native_table(table, group, seed, reference, active))
            progress("native_table_precision", table_precision=table_checks)
        return row
    # The only actual allocation update. Subsequent fixture clocks are explicit, not claimed training.
    updated(control,"allocation",1)
    work=out/"work";work.mkdir(); identities={}
    def metadata(label):
        return dict(fixture=label,**method.metadata(active),rng=diag.rng_state(),modes=[m.training for m in model.modules()])
    common_path=work/"common.pt";common_meta=metadata("One actual "+control+" update; common current fixture Adam step1")
    common = timed("s_F",lambda:native_save(common_path,model,optimizer,common_meta,check))
    identities["common"] = common
    def reload(path,receipt):
        return timed("r_F",lambda:run.restore_point(path,receipt,model,optimizer,active))
    reload(common_path,common); updated(control,"A",2); gcurrent=state(model,optimizer)
    reload(common_path,common); hrow=updated(active,"A",2)
    effects["current"] = component_effects(model,gcurrent,optimizer=optimizer)
    for key in probes:
        delta=float((gradients[2]["current"][key]-gradients[3]["current"][key]).abs().max())
        if delta<=0:raise ValueError("Candidate-control preclip current gradient not distinguishable")
        effects["current"][key]["pre_clip_gradient_max_delta"]=delta
    progress("native_effects", effects=effects)
    objective_pairs["current"] = run.native_objective_pair(updates[1]["log"], hrow, 1)
    progress("native_objective_pair", objective_pairs=objective_pairs)
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
    effects["paired"]=component_effects(model,gpaired,optimizer=optimizer)
    for key in probes:
        delta=float((gradients[4]["history"][key]-gradients[5]["history"][key]).abs().max())
        if delta<=0:raise ValueError("Candidate-control preclip paired gradient not distinguishable")
        effects["paired"][key]["pre_clip_gradient_max_delta"]=delta
        branch_deltas = {}
        for role in ("current", "history"):
            off = gradients[4][role][key]-(gradients[4]["current"][key] if role == "history" else 0)
            on = gradients[5][role][key]-(gradients[5]["current"][key] if role == "history" else 0)
            branch_deltas[role] = float((off-on).abs().max())
            if branch_deltas[role] <= 0:
                raise ValueError("Missing isolated native rank effect: "+role+"/"+key)
        effects["paired"][key]["isolated_branch_preclip_deltas"] = branch_deltas
    progress("native_effects", effects=effects)
    objective_pairs["paired"] = run.native_objective_pair(updates[3]["log"], first, 2)
    progress("native_objective_pair", objective_pairs=objective_pairs)
    del gpaired;gc.collect()
    expected, expected_rng = state(model,optimizer), diag.rng_state()
    reload(paired_path,paired)
    memory=handwritten.shared_memory(active);memory.begin_stage(2)
    for _ in range(48):memory.draw()
    payload=memory.to_bytes();rng=diag.rng_state();modes=[m.training for m in model.modules()]
    fit=[handwritten.handmade_group(f"native_fit_{i}") for i in range(48)]
    valid=[data.Group(g.uid,g.sellers,g.items) for g in [handwritten.handmade_group(f"native_valid_{i}") for i in range(40)]]
    before=core.state_digest(state(model,optimizer));droot=out/"diagnostics";droot.mkdir()
    progress("actual_candidate_full_role_capture", actual_updates=len(updates))
    method.capture(model,optimizer,fit,memory,valid,c,active,2,1,droot,"handwritten",check)
    if before!=core.state_digest(state(model,optimizer)) or rng!=diag.rng_state() or payload!=memory.to_bytes():
        raise ValueError("Actual candidate capture mutated state")
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
    # Persist the actual candidate-updated state, releasing the obsolete common work file first.
    data.write_json(out/"temporary_checkpoint_identity.json",dict(files=identities,removed=False))
    common_path.unlink()
    actual_meta=metadata("After actual candidate update6; H full-state restore")
    actual=timed("s_F",lambda:native_save(common_path,model,optimizer,actual_meta,check))
    identities["H_full"]=actual
    with diag.neutral(model):
        timed("v",lambda:record.reference(model,history,c,check))
        before_scores=timed("v",lambda:method.restored_score(model,[history],c,check))
    if reload(common_path,actual)!=actual_meta:raise ValueError("Full H metadata restore mismatch")
    with diag.neutral(model):
        if not np.array_equal(before_scores,timed("v",lambda:method.restored_score(model,[history],c,check))):
            raise ValueError("Full H scores not bitwise restored")
    inference_path=work/"H_inference.pt";infer_meta=metadata("Actual candidate inference restore fixture")
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
        actual_updates=6,updates=updates,measurements=measurements,projection=run.single.projection(measurements),effects=effects,
        both_phases_encoder_head_effect=True,pooling_restores_verified=True,next_update_bitwise_equal=True,
        metadata=method.metadata(active), rank_precision_verified=bool(table_checks),
        table_precision=table_checks, objective_pairs=objective_pairs,
        equality_counts=equality,maximum_records=224,channel_tokens=256,memory_bytes=len(payload),residuals=residuals,
        fixture="Six actual updates only. Common model/moments after call1; paired counter336 explicitly synthetic. Calls5/6 differ only by actual capture. No formal input.")

def main():
    import importlib
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
    handwritten = importlib.import_module("test_step28_positive_ranking")
    os.environ["G_HALF_TEST_ROOT"] = str(args.output)
    os.environ["H_CONS_TEST_ROOT"] = str(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    numerics = run.parent.configure_numerics()
    process = psutil.Process()
    limits = run.policy()["cpu_verification" if args.mode == "cpu" else "native_verification"]
    report = {"status": "RUNNING", "mode": args.mode, "scope": "S_pos_H_pos_increment", "source_files": run.sources(), "formal_inputs": False, "torch": torch.__version__,
        "formal_labels": False, "prior_compute_seconds": args.prior_seconds, "peak_rss_bytes": 0, "peak_work_bytes": 0, "numerics": numerics}
    stopped, lock = threading.Event(), threading.RLock()
    arm_started = None
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
                    if "work" not in path.relative_to(args.output.parent).parts:
                        evidence += size
            except FileNotFoundError:
                pass
        report["peak_work_bytes"] = max(report["peak_work_bytes"], total)
        disk_reserve(args.output, total, reserve, limits.get("maximum_work_bytes", limits["maximum_evidence_bytes"]))
        if (time.monotonic()-STARTED+args.prior_seconds > limits["maximum_seconds"]
                or (arm_started is not None and time.monotonic()-arm_started > limits["per_arm_maximum_seconds"])
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
            test_class = handwritten.PositiveRankingTests
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
            if args.prior_seconds != 0:
                raise ValueError("Unique dual native attempt; no automatic retry")
            report["arms"] = {}
            for arm in method.ARMS:
                arm_started = time.monotonic()
                arm_out = args.output/arm
                arm_out.mkdir()
                def arm_check(reserve=0):
                    check(reserve)
                    if time.monotonic()-arm_started > limits["per_arm_maximum_seconds"]:
                        raise RuntimeError("Per-arm native budget exceeded")
                def arm_progress(stage, **details):
                    if "actual_updates" in details:
                        details["active_arm_actual_updates"] = details["actual_updates"]
                        details["actual_updates"] += sum(v["actual_updates"] for v in report["arms"].values())
                    progress(stage, active_arm=arm, **details)
                value = native(arm_check, arm_progress, handwritten, arm_out, arm)
                arm_check()
                value["charged_seconds"] = math.ceil(time.monotonic()-arm_started)
                report["arms"][arm] = value
                arm_started = None
                write()
            report["actual_updates"] = sum(v["actual_updates"] for v in report["arms"].values())
        check()
        report["status"] = "PASS_HANDWRITTEN_ONLY"
        if args.mode == "cpu":
            report.update(qualified_tests=selected, unresolved_checks=[])
    except BaseException as exc:
        report.update(status="FAIL", traceback=traceback.format_exc())
        raise
    finally:
        stopped.set()
        watcher.join(timeout=2)
        write()

if __name__ == "__main__":
    main()
