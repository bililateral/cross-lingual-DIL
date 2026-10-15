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

import step28_balanced_pooling_run as run
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


def disk_reserve(path, total, reserve, maximum):
    if total+reserve > maximum or shutil.disk_usage(path).free < reserve:
        raise RuntimeError("Qualification checkpoint reserve exceeds available disk or approved work bytes")


def native_save(path, model, optimizer, metadata, check):
    check(run.base.persistence.checkpoint_reserve(model, optimizer))
    return core.save_state(path, model, optimizer, metadata)


def native(check, progress, handwritten, out):
    resources = preflight()
    if shutil.disk_usage(out).free < 10*2**30:
        raise ValueError("Insufficient native temporary disk")
    numerics = run.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    old = run.prior_evidence()["native_result"]["native"]
    measurements, updates = {}, []
    def timed(name, fn):
        check(); torch.cuda.synchronize(); started=time.monotonic()
        result=fn(); torch.cuda.synchronize()
        measurements[name]=max(measurements.get(name,0.),time.monotonic()-started)
        return result
    c=run.config()
    progress("load_native_BGE")
    model,optimizer,initial,archive=timed("L",lambda:checked_fresh(c))
    if not model.encoder[0].auto_model.is_gradient_checkpointing or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16/checkpointing required")
    method.bind(model,{"pooling":method.identity("F")})
    current,history=[handwritten.handmade_group("native_"+s,8,254) for s in ("current","history")]
    for g in (current,history):
        lengths=[len(v) for v in model.encoder.tokenizer(run.base.record_texts(g,"separate_moments"),padding=False,truncation=False)["input_ids"]]
        if len(lengths)!=448 or set(lengths)!={256}:
            raise ValueError("Maximum N224/token256 fixture differs")
    probes={"encoder":model.encoder[0].auto_model.encoder.layer[0].attention.self.query.weight,"head":model.head[0].weight}
    effects={}
    def updated(phase,stage,step,h,target):
        before={k:p.detach().cpu().clone() for k,p in probes.items()}; gradients={}
        def observe(role,log):
            if role==("current" if h is None else "history"):
                for key,p in probes.items():
                    if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError("Missing native gradient")
                    gradients[key]=float(p.grad.float().norm())
        row=timed("t_"+phase+"_F",lambda:method.update(model,optimizer,current,h,target,c,"F",stage,step,check,observe))
        updates.append(row)
        delta={k:float((p.detach().cpu()-before[k]).abs().max()) for k,p in probes.items()}
        if any(x<=0 for x in [*gradients.values(),*delta.values()]) or len(gradients)!=2:
            raise ValueError("Native encoder/head update inactive")
        effects[phase]=dict(pre_clip_gradient_norm=gradients,parameter_max_delta=delta)
        progress("incremental_update",actual_updates=len(updates),completed_updates=updates,measurements=measurements)
        return row
    updated("A",1,1,None,None)
    if run.parent.adam_step(optimizer)!=1:raise ValueError("Real Adam allocation missing")
    with diag.neutral(model):table=timed("v_F",lambda:record.reference(model,history,c,check))
    i,j=np.triu_indices(224,1)
    target=(table+2*np.sin(i*.31+j*.71)).astype(np.float32)
    for value in optimizer.state.values():value["step"].fill_(336)
    baseline=state(model,optimizer);rng=diag.rng_state();modes=[m.training for m in model.modules()]
    memory=handwritten.shared_memory("F");memory.begin_stage(2)
    for _ in range(48):memory.draw()
    payload=memory.to_bytes()
    first=updated("B",2,49,history,target)
    if first["history"]["supervised"]<=0 or first["history"]["mse0"]<=0:raise ValueError("Inactive paired history")
    expected,expected_rng=state(model,optimizer),diag.rng_state()
    restore(model,optimizer,baseline,rng,modes)
    fit=[handwritten.handmade_group(f"native_fit_{i}") for i in range(48)]
    valid=[data.Group(g.uid,g.sellers,g.items) for g in [handwritten.handmade_group(f"native_valid_{i}") for i in range(40)]]
    before=core.state_digest(state(model,optimizer)); droot=out/"diagnostics";droot.mkdir()
    progress("actual_F_epoch_diagnostic")
    method.capture(model,optimizer,fit,memory,valid,c,"F",2,1,droot,"handwritten",check)
    if before!=core.state_digest(state(model,optimizer)) or rng!=diag.rng_state() or memory.to_bytes()!=payload:
        raise ValueError("Actual F diagnostic changed learner state")
    repeated=updated("B",2,49,history,target)
    equality={"tensors":0,"elements":0};equal_state(expected,state(model,optimizer),equality)
    if expected_rng!=diag.rng_state() or memory.to_bytes()!=payload or modes!=[m.training for m in model.modules()]:
        raise ValueError("Diagnostic changed next state")
    if {k:v for k,v in first.items() if k!="solver_summary"}!={k:v for k,v in repeated.items() if k!="solver_summary"}:
        raise ValueError("Repeated F update differs")
    if len(updates)!=3 or run.parent.adam_step(optimizer)!=337:raise ValueError("Fixed three updates missing")
    del baseline,expected;gc.collect()
    tensors=[*model.parameters(),*(p.grad for p in model.parameters() if p.grad is not None),
             *(s[k] for s in optimizer.state.values() for k in ("exp_avg","exp_avg_sq"))]
    if any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in tensors):raise ValueError("Finite FP32 state required")
    del tensors
    def residual():
        errors=[]
        for seed in (None,data.seed_for(20261008,"ABC",2,history.uid,"diagnostic_regroup")):
            slots=record.assignment(history,seed)
            errors.append((method.account_logits(torch.from_numpy(table),slots,"F",c["pooling_stats"])
                -method.account_logits(torch.from_numpy(target),slots,"F",c["pooling_stats"])).numpy().astype(np.float64))
        return {"D0":float(np.mean(errors[0]**2)),"D1":float(np.mean(errors[1]**2))}
    timed("h_F",residual)
    meta={"fixture":"handwritten native","pooling":method.identity("F"),"rng":diag.rng_state(),"modes":[m.training for m in model.modules()]}
    work=out/"work";work.mkdir();identities={}
    with diag.neutral(model):
        timed("v_F",lambda:record.reference(model,history,c,check))
        before_scores=timed("v_F",lambda:method.restored_score(model,[history],c,check))
    for kind,opt,skey,rkey,wkey in (("full",optimizer,"s_F","r_F","W_F"),("inference",None,"s_I","r_I","W_I")):
        path=work/("F_"+kind+".pt")
        saved=timed(skey,lambda:native_save(path,model,opt,meta,check))
        if timed(rkey,lambda:run.restore_point(path,saved,model,opt,"F"))!=meta:raise ValueError("F restore metadata differs")
        with diag.neutral(model):
            if not np.array_equal(before_scores,timed("v_F",lambda:method.restored_score(model,[history],c,check))):
                raise ValueError("F restored native scores differ")
        measurements[wkey]=path.stat().st_size;identities[kind]=saved
        data.write_json(out/"temporary_checkpoint_identity.json",dict(files=identities,removed=False))
        path.unlink()
    work.rmdir();data.write_json(out/"temporary_checkpoint_identity.json",dict(files=identities,removed=True))
    natural=measurements.copy()
    gpu_stats={};captured=[]
    def hard_gpu():
        table,slots=handwritten.terminal_table("cuda:0");table.requires_grad_()
        def keep(tensor):
            if tensor.shape==(378,3,3):captured.append(tensor.detach())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(keep,lambda t:t):
            values=method.account_logits(table,slots,"F",gpu_stats)
        upstream=torch.linspace(-.7,1.3,378,device=table.device)
        gradient,=torch.autograd.grad(values,table,upstream)
        return values.detach(),gradient,upstream
    values,gradient,upstream=timed("c_gpu",hard_gpu)
    if len(captured)!=1:raise ValueError("Actual Envelope saved Q missing")
    plan=captured[0];reference,refp=handwritten.rounded_reference(handwritten.terminal_block().float().double().numpy())
    if float((values-reference).abs().max())>1e-8+1e-7*max(1,abs(reference)) or float((plan.double()-torch.tensor(refp,device=plan.device)).abs().sum((1,2)).max())>2e-4:
        raise ValueError("Actual GPU terminal output precision failed")
    i,j=torch.triu_indices(84,84,1,device=gradient.device);expected_gradient=torch.zeros_like(gradient)
    for pair,(a,b) in enumerate(zip(*np.triu_indices(28,1))):
        mask=(i//3==int(a))&(j//3==int(b));expected_gradient[mask]=upstream[pair]*plan[pair,i[mask]%3,j[mask]%3]
    if not torch.equal(gradient,expected_gradient):raise ValueError("Actual full-table GPU VJP did not use returned Q")
    cpu_stats={}
    def hard_cpu():
        table,slots=handwritten.terminal_table()
        return method.account_logits(table,slots,"F",cpu_stats)
    timed("c_cpu",hard_cpu)
    stress=dict(c_gpu=measurements.pop("c_gpu"),c_cpu=measurements.pop("c_cpu"),accounts=28,records=84,pairs=378,
                solver_summaries=dict(gpu=gpu_stats,cpu=cpu_stats),actual_gpu_reference_value=reference)
    for s in stress["solver_summaries"].values():
        run.solver_gate(s)
        if s["terminal_acceptances"]!=378:raise ValueError("Hard full-table did not trigger terminal")
    for arm in ("S","G"):
        for key in ("t_A","t_B","v","h"):measurements[key+"_"+arm]=old["measurements"][key+"_"+arm]
    for key in ("L","s_F","r_F","s_I","r_I","W_F","W_I"):measurements[key]=max(measurements[key],old["measurements"][key])
    for key,mult,cost in (("t_A_F",2,"c_gpu"),("t_B_F",5,"c_gpu"),("v_F",1,"c_gpu"),("h_F",4,"c_cpu")):
        measurements[key]+=mult*stress[cost]
    result=dict(resources=resources,numerics=numerics,pretrained_archive=archive,initial_state=initial,
        actual_updates=3,cumulative_updates=11,updates=updates,measurements=measurements,natural_measurements=natural,
        inherited_native=run.policy()["prior_evidence"]["native_result"],terminal_stress=stress,
        projection=run.projection(measurements),effects=effects,both_phases_encoder_head_effect=True,
        F_pooling_restores_verified=True,terminal_gpu_vjp_verified=True,next_update_bitwise_equal=True,
        equality_counts=equality,maximum_records=224,channel_tokens=256,memory_bytes=len(payload),solver_summary=c["pooling_stats"],
        fixture="Three actual F updates, synthetic B counter336, actual diagnostic and next-update comparison; GPU 3x3 terminal stress is separate from natural BGE text. No formal inputs.")
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--scope", choices=("full", "terminal-q"), default="full")
    parser.add_argument("--prior-seconds", type=float, default=0.)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if os.name != "posix" or args.output.exists() or args.prior_seconds < 0:
        parser.error("Linux, new evidence and cumulative ledger required; GPU has no automatic retry")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(data.ROOT/"tests"))
    import test_step28_balanced_pooling as handwritten
    os.environ["BALANCED_POOLING_TEST_ROOT"] = str(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    numerics = run.parent.configure_numerics()
    process = psutil.Process()
    limits = run.policy()["cpu_verification" if args.mode == "cpu" else "native_verification"]
    report = {"status": "RUNNING", "mode": args.mode, "scope": args.scope, "source_files": run.sources(), "formal_inputs": False, "torch": torch.__version__,
        "formal_labels": False, "prior_compute_seconds": args.prior_seconds, "peak_rss_bytes": 0, "peak_work_bytes": 0, "numerics": numerics}
    stopped, lock = threading.Event(), threading.RLock()
    def write():
        with lock:
            report["elapsed_seconds"] = time.monotonic()-STARTED
            report["cumulative_seconds"] = args.prior_seconds+report["elapsed_seconds"]
            report["cumulative_charged_seconds"] = args.prior_seconds+math.ceil(report["elapsed_seconds"])
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
            test_class = handwritten.TerminalPoolingTests if args.scope == "terminal-q" else handwritten.BalancedPoolingTests
            selected = (["test_terminal_precision_vjp_and_batch", "test_terminal_real_chunked_update",
                         "test_terminal_gate_budget_and_inherited_evidence", "test_independent_three_arm_orchestration_and_blind_gate"]
                        if args.scope == "terminal-q" else unittest.defaultTestLoader.getTestCaseNames(test_class))
            report["selected_tests"] = selected
            suite = unittest.TestSuite(test_class(name) for name in selected)
            with (args.output/"unittest.txt").open("w", encoding="utf-8") as stream:
                result = unittest.TextTestRunner(stream=stream, verbosity=2, failfast=True).run(suite)
            report.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
            if not result.wasSuccessful() or result.skipped:
                raise RuntimeError("Affected handwritten qualification failed")
        else:
            if args.scope != "terminal-q":
                raise ValueError("Current native allowance is the fixed three-update delta only")
            if args.gate is None:
                raise ValueError("Implementation review gate required")
            gate = run.reviewed_gate(args.gate, "NATIVE")
            if args.prior_seconds != 364 or args.prior_seconds != gate.get("native_prior_seconds"):
                raise ValueError("Prior GPU budget must match the explicitly approved attempt gate")
            report["native"] = native(check, progress, handwritten, args.output)
        check()
        report["status"] = "PASS_HANDWRITTEN_ONLY"
    except BaseException as exc:
        report.update(status="FAIL", traceback=traceback.format_exc(), solver_failure=getattr(exc, "details", None))
        raise
    finally:
        stopped.set()
        watcher.join(timeout=2)
        write()


if __name__ == "__main__":
    main()
