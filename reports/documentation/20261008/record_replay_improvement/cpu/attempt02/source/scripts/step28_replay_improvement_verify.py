"""Cumulative Linux-only handwritten qualification for the approved five-arm experiment."""
from __future__ import annotations

import time
STARTED = time.monotonic()

import argparse
import copy
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

import step28_replay_improvement_run as run
import step28_replay_diagnostics as diag
from step28_record_replay_verify import preflight

data,record,core = run.data,run.record,run.core


def host_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:host_copy(v) for k,v in value.items()}
    if isinstance(value,list):
        return [host_copy(v) for v in value]
    if isinstance(value,tuple):
        return tuple(host_copy(v) for v in value)
    return copy.deepcopy(value)


def native(check,progress,handmade,out: Path) -> dict:
    resources = preflight()
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    result = {"architectures":{},"resources":resources,"neutrality_both_architectures":False}
    for arm,family in (("C_plus","record"),("LOGIT0.1","LOGIT0.1")):
        c = run.config(arm)
        progress("native_load_"+family)
        model = record.load_model(c)
        auto = model.encoder[0].auto_model
        if not auto.is_gradient_checkpointing or not torch.cuda.is_bf16_supported():
            raise ValueError("Native BF16/checkpointing mode missing")
        current,history = [handmade.handmade_group("native_"+role,8,254) for role in ("current","history")]
        for group in (current,history):
            lengths = [len(v) for v in model.encoder.tokenizer(run.base.record_texts(group,"separate_moments"),padding=False,truncation=False)["input_ids"]]
            if len(lengths)!=448 or set(lengths)!={256}:
                raise ValueError("Not the declared N224/token256 worst shape")
        optimizer = core.make_optimizer(model,c)
        tick = time.monotonic()
        with diag.neutral(model):
            target = record.reference(model,history,c,check) if arm!="LOGIT0.1" else diag.score(model,[history],c,arm,check)[0]
        torch.cuda.synchronize()
        source_seconds = time.monotonic()-tick
        progress("native_first_allocation_"+family)
        tick = time.monotonic()
        run.update(model,optimizer,current,None,None,c,arm,"ABC",1,1,check)
        torch.cuda.synchronize()
        first_seconds = time.monotonic()-tick
        # Explicit handwritten continuation fixture: one real update's Adam moments,
        # phase counter set to 288. This is not a claim to have trained an old stage.
        for state in optimizer.state.values():
            state["step"].fill_(288)
        model.train()
        model.head.eval()
        initial_model,initial_optimizer = host_copy(model.state_dict()),host_copy(optimizer.state_dict())
        initial_grads = [p.grad.detach().cpu().clone() for p in model.parameters()]
        state,modes = diag.rng_state(),[m.training for m in model.modules()]
        initial_hashes = (core.state_digest(model.state_dict()),core.state_digest(optimizer.state_dict()),core.state_digest(initial_grads))
        progress("native_paired_baseline_"+family)
        tick = time.monotonic()
        row = run.update(model,optimizer,current,history,target,c,arm,"ABC",2,1,check)
        torch.cuda.synchronize()
        paired_seconds = time.monotonic()-tick
        baseline_hashes = (core.state_digest(model.state_dict()),core.state_digest(optimizer.state_dict()))
        if baseline_hashes[0] == initial_hashes[0] or run.parent.adam_step(optimizer)!=289:
            raise ValueError("Native paired Adam update did not execute")
        model.load_state_dict(initial_model)
        optimizer.load_state_dict(initial_optimizer)
        for p,g in zip(model.parameters(),initial_grads):
            p.grad = g.to(p.device)
        for m,mode in zip(model.modules(),modes):
            m.training = mode
        diag.restore_rng(state)
        progress("native_diagnostic_"+family)
        tick = time.monotonic()
        with diag.neutral(model):
            values = diag.score(model,[current,history],c,arm,check)
        torch.cuda.synchronize()
        score_seconds = (time.monotonic()-tick)/2
        tick = time.monotonic()
        run.previous.save_array(out/(family+"_handwritten_scores.npy"),values,out)
        save_seconds = time.monotonic()-tick
        after_hashes = (core.state_digest(model.state_dict()),core.state_digest(optimizer.state_dict()),
                        core.state_digest([p.grad for p in model.parameters()]))
        if initial_hashes != after_hashes or state != diag.rng_state() or modes != [m.training for m in model.modules()]:
            raise ValueError("Native eval diagnostic mutated model/grad/Adam/RNG/modes")
        repeat = run.update(model,optimizer,current,history,target,c,arm,"ABC",2,1,check)
        if baseline_hashes != (core.state_digest(model.state_dict()),core.state_digest(optimizer.state_dict())):
            raise ValueError("Native next update differs with inserted diagnostics")
        tensors = [*model.parameters(),*(p.grad for p in model.parameters()),*(s[k] for s in optimizer.state.values() for k in ("exp_avg","exp_avg_sq"))]
        if any(t.dtype!=torch.float32 for t in tensors):
            raise ValueError("Parameters/gradients/Adam must remain FP32")
        parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
        result["architectures"][family] = {"arm":arm,"source_seconds":source_seconds,"worst_eval_group_seconds":score_seconds,
            "first_current_allocation_seconds":first_seconds,"paired_update_seconds":paired_seconds,"save_two_groups_seconds":save_seconds,
            "parameter_bytes":parameter_bytes,"maximum_records":224,"channel_tokens":256,"loss":row,"repeat_loss":repeat,
            "native_state_unchanged":True,"next_model_adam_bitwise_equal":True,
            "fixture":"One real first update; synthetic phase counter 288; two actual paired next updates from identical model/Adam/RNG; no formal inputs."}
        del model,optimizer,initial_model,initial_optimizer,initial_grads,tensors,auto
        gc.collect()
        torch.cuda.empty_cache()
        check()
    r,l = result["architectures"]["record"],result["architectures"]["LOGIT0.1"]
    # Actual presentation counts: 14,688 record and 4,320 LOGIT gradient groups.
    # Record forward-only groups: 18,360 epoch scores + 864 extra cache tables
    # + 3,888 checkpoint/restored scores + <=90 new sources. LOGIT: 6,048+1,296+36.
    train_seconds = 14688*max(r["paired_update_seconds"]/2,r["first_current_allocation_seconds"])+4320*max(l["paired_update_seconds"]/2,l["first_current_allocation_seconds"])
    forward_seconds = 23202*max(r["worst_eval_group_seconds"],r["source_seconds"])+7380*max(l["worst_eval_group_seconds"],l["source_seconds"])
    result["projected_total_seconds"] = 1.25*(train_seconds+forward_seconds+3600+24408*max(r["save_two_groups_seconds"],l["save_two_groups_seconds"])/2)
    # 36 inference states plus at most shared+current full states, and 2GiB small
    # evidence allowance, then 10% serialization/headroom. No epoch weights.
    result["projected_peak_output_bytes"] = int(1.1*(27*r["parameter_bytes"]+9*l["parameter_bytes"]+6*max(r["parameter_bytes"],l["parameter_bytes"])+2*2**30))
    result.update(neutrality_both_architectures=True,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                  projection_scope="Worst legal shape for every group, measured epoch forwards/saves, 3600s checkpoint/statistics allowance, 25% time margin")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("cpu","gpu"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gate",type=Path)
    parser.add_argument("--prior-seconds",type=float,default=0)
    args = parser.parse_args()
    if os.name!="posix" or args.output.exists() or args.prior_seconds<0:
        parser.error("Linux, new evidence and honest cumulative time required")
    args.output.mkdir(parents=True)
    sys.path.insert(0,str(data.ROOT/"tests"))
    import test_step28_replay_improvement as handmade
    os.environ["RECORD_REPLAY_TEST_ROOT"] = str(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    limits = run.policy()["cpu_verification" if args.mode=="cpu" else "native_verification"]
    report = {"status":"RUNNING","mode":args.mode,"source_files":run.sources(),"formal_inputs":False,"formal_labels":False,
              "prior_compute_seconds":args.prior_seconds,"peak_rss_bytes":0}
    stopped,lock = threading.Event(),threading.RLock()
    def write():
        with lock:
            report["elapsed_seconds"] = time.monotonic()-STARTED
            report["cumulative_seconds"] = args.prior_seconds+report["elapsed_seconds"]
            if args.mode=="gpu" and torch.cuda.is_initialized():
                report["max_reserved_bytes"] = torch.cuda.max_memory_reserved()
            data.write_json(args.output/"result.json",report)
    def check():
        report["peak_rss_bytes"] = max(report["peak_rss_bytes"],process.memory_info().rss)
        used = 0
        for path in args.output.parent.rglob("*"):
            try:
                if path.is_file():
                    used += path.stat().st_size
            except FileNotFoundError:
                pass
        if (time.monotonic()-STARTED+args.prior_seconds > limits["maximum_seconds"]
                or used>limits["maximum_evidence_bytes"] or report["peak_rss_bytes"]>(2 if args.mode=="cpu" else 64)*2**30
                or (args.mode=="gpu" and torch.cuda.is_initialized() and torch.cuda.max_memory_reserved()>28*2**30)):
            raise RuntimeError("Approved cumulative verification budget exceeded")
    def watchdog():
        while not stopped.wait(.5):
            try:
                check()
            except BaseException as exc:
                report.update(status="BUDGET_STOP",error=str(exc))
                write()
                os._exit(2)
    def progress(stage):
        report["stage"] = stage
        write()
    watcher = threading.Thread(target=watchdog,daemon=True)
    watcher.start()
    try:
        if len(process.cpu_affinity())!=1 or any(os.environ.get(k)!="1" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")):
            raise ValueError("One CPU and pre-import single thread required")
        if args.mode=="cpu":
            if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
                raise ValueError("CPU verification must disable CUDA")
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(handmade.ImprovementTests)
            with (args.output/"unittest.txt").open("w",encoding="utf-8") as stream:
                result = unittest.TextTestRunner(stream=stream,verbosity=2,failfast=True).run(suite)
            report.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
            if not result.wasSuccessful() or result.skipped:
                raise RuntimeError("Affected handwritten checks failed")
        else:
            if args.gate is None:
                raise ValueError("Reviewed source gate required before native GPU")
            gate = data.read_json(args.gate)
            if gate.get("status")!="APPROVED_REPLAY_IMPROVEMENT_NATIVE" or gate.get("source_files")!=run.sources() or gate.get("review_disposition")!="NO_OPEN_BLOCKERS":
                raise ValueError("Reviewed native gate differs")
            rec = gate["cpu"]
            evidence = data.read_json(data.verify(data.ROOT/rec["path"],rec))
            if evidence.get("mode")!="cpu" or evidence.get("status")!="PASS_HANDWRITTEN_ONLY" or evidence.get("source_files")!=run.sources():
                raise ValueError("Matching CPU evidence required")
            report["native"] = native(check,progress,handmade,args.output)
        check()
        report["status"] = "PASS_HANDWRITTEN_ONLY"
    except BaseException:
        report.update(status="FAIL",traceback=traceback.format_exc())
        raise
    finally:
        stopped.set()
        watcher.join(timeout=2)
        write()


if __name__ == "__main__":
    main()
