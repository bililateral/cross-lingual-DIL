"""Bounded handwritten CPU checks or one native GPU update; no formal loader."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import unittest
import subprocess

STARTED = time.monotonic()

import numpy as np
import psutil
import torch

import step28_function_memory as method

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import test_step28_function_memory as handwritten


def preflight() -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Native verification must map physical GPU0")
    process = psutil.Process()
    if len(process.cpu_affinity()) != 1:
        raise ValueError("Exactly one CPU affinity required")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError(f"Missing pre-import single-thread setting: {key}")
    if os.environ.get("TOKENIZERS_PARALLELISM") != "false":
        raise ValueError("Tokenizer threads must be disabled before import")
    gpu = subprocess.check_output(["nvidia-smi", "--id=0",
        "--query-gpu=index,uuid,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    apps = subprocess.check_output(["nvidia-smi", "--id=0",
        "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True).strip()
    ancestors = {p.pid for p in process.parents()} | {process.pid}
    jobs = [p.info for p in psutil.process_iter(["pid", "username", "cmdline"])
            if p.pid not in ancestors and any("step28" in x for x in (p.info["cmdline"] or []))]
    available = psutil.virtual_memory().available
    if int(gpu.split(",")[-2].strip()) < 30*1024 or available < 72*2**30 or apps or jobs:
        raise ValueError(f"Resource preflight failed: gpu={gpu}, apps={apps}, available={available}, jobs={jobs}")
    return {"gpu":gpu, "compute_apps":apps, "host_available_bytes":available, "project_jobs":jobs}


def assert_update_dtypes(model, optimizer) -> dict:
    parameters = list(model.parameters())
    gradients = [p.grad for p in parameters if p.grad is not None]
    moments = [s[key] for s in optimizer.state.values() for key in ("exp_avg", "exp_avg_sq") if key in s]
    if (not gradients or not moments or any(v.dtype != torch.float32 for v in parameters + gradients + moments)):
        raise ValueError("Native parameters, nonempty gradients and Adam moments must be FP32")
    return {"parameters":len(parameters), "gradients":len(gradients), "adam_moments":len(moments), "dtype":"float32"}


def native(check, progress) -> dict:
    c = method.config()
    progress("preflight", preflight=preflight())
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    progress("load_model")
    model = method.load_model(c)
    auto = model.encoder[0].auto_model
    if not auto.is_gradient_checkpointing:
        raise ValueError("Native checkpointing not enabled")
    current=handwritten.handmade_group("native_current",8,254)
    old=handwritten.handmade_group("native_history",8,254)
    lengths=[]
    for g in (current,old):
        texts=method.base.record_texts(g,"separate_moments")
        tokens=model.encoder.tokenizer(texts,padding=False,truncation=False)
        shape=[len(ids) for ids in tokens["input_ids"]]
        if len(shape)!=448 or set(shape)!={256}:
            raise ValueError("Handwritten native case does not cover 448 x 256")
        lengths.append({"texts":len(shape),"minimum":min(shape),"maximum":max(shape)})
    progress("reference_forward", input_shapes=lengths)
    x=method.reference(model,old,c,check)
    # Explicit initial-model handwritten memory fixture, not a trained first stage.
    h,b,constant=method.statistics(torch.from_numpy(x),old.labels,method.weight(model).detach().cpu())
    memory=method.Memory("ABC")
    memory.h,memory.b,memory.constant,memory.count=h.numpy(),b.numpy(),constant,1
    # Check no_grad versus live canonical coordinates, release this graph before step.
    progress("live_reference_comparison")
    with method.deterministic_history(model):
        y=method.relation_features(model,old,c,check)
    difference=float((y.detach().cpu()-torch.from_numpy(x)).abs().max())
    assert np.array_equal(y.detach().cpu().numpy(), x.astype(np.float32))
    del y
    progress("reference_verified", reference_max_difference=difference)
    with torch.no_grad():
        model.head[2].weight.add_(.003)
    # The created function target has zero residual; this fixed perturbation tests active retention.
    query_name,query_parameter=next((n,p) for n,p in model.encoder.named_parameters() if "query.weight" in n)
    probes=(query_parameter,model.head[0].weight,model.head[2].weight)
    before=[p.detach().clone() for p in probes]
    history_gradients=[]
    def inspect_history(base_loss, loss):
        grads=torch.autograd.grad(loss,probes,retain_graph=True)
        norms=[float(g.float().norm()) for g in grads]
        if not all(np.isfinite(norms)) or not all(v>0 for v in norms):
            raise ValueError("Historical objective did not reach native encoder/head/w")
        history_gradients.extend(norms)
        progress("history_gradient_verified", history_only_gradient_norms=norms)
        del grads
    calls=[0]
    def count_forward(module,args):
        calls[0]+=1
    handle=auto.encoder.layer[0].register_forward_pre_hook(count_forward)
    optimizer=method.make_optimizer(model,c)
    tick=time.monotonic()
    row=method.optimization_step(model,optimizer,current,old,x,memory,c,1e-5,
                                 current_seed=51,history_seed=71,check=check,observe_history=inspect_history,observe_stage=progress)
    torch.cuda.synchronize()
    elapsed=time.monotonic()-tick
    handle.remove()
    changes=[float((p.detach()-v).abs().max()) for p,v in zip(probes,before)]
    if not all(v>0 for v in changes) or calls[0] <= 224:
        raise ValueError("Missing parameter change or checkpoint recomputation")
    dtypes = assert_update_dtypes(model, optimizer)
    return {"kind":"native_handwritten_first_optimizer_step", "official_data":False,
            "optimizer":row,"input_shapes":lengths,"reference_max_difference":difference,
            "encoder_probe":query_name,"history_only_gradient_norms":history_gradients,
            "probe_max_parameter_changes":changes,"first_layer_calls":calls[0],
            "checkpointing":auto.is_gradient_checkpointing,"update_seconds":elapsed,
            "max_allocated_bytes":torch.cuda.max_memory_allocated(),
            "max_reserved_bytes":torch.cuda.max_memory_reserved(),
            "adam_parameters_with_state":len(optimizer.state),
            "dtypes":dtypes,
            "note":"Extra history-only derivative diagnosis counted in budget; no native weights saved."}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("cpu","gpu"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gate",type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ["FUNCTION_MEMORY_TEST_ROOT"]=str(args.output)
    started=STARTED
    limit=600
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    process=psutil.Process()
    peak_rss=[0]
    stopped=threading.Event()
    report={"mode":args.mode,"started_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python":sys.version,"torch":torch.__version__,"numpy":np.__version__,
            "affinity":process.cpu_affinity(),"seconds_limit":limit,
            "scientific_sources":{str(p.relative_to(method.data.ROOT)):method.data.sha256(p)
                                  for p in (Path(method.__file__),Path(__file__),Path(handwritten.__file__))}}
    import step28_function_memory_run as run
    import test_step28_function_memory_run as integration
    report["source_files"]=run.sources()
    def write():
        report["elapsed_seconds"]=time.monotonic()-started
        report["peak_rss_bytes"]=peak_rss[0]
        if args.mode == "gpu" and torch.cuda.is_initialized():
            report["max_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["max_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (args.output/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    def check():
        rss=process.memory_info().rss
        peak_rss[0]=max(peak_rss[0],rss)
        used=0
        for path in args.output.parent.rglob("*"):
            try:
                if path.is_file(): used+=path.stat().st_size
            except FileNotFoundError:
                pass  # Disposable handwritten checkpoint files can be removed during the scan.
        if (time.monotonic()-started>limit or rss>(2 if args.mode=="cpu" else 64)*2**30
                or used>(16 if args.mode=="cpu" else 128)*2**20):
            raise RuntimeError("Handwritten verification budget exceeded")
        if args.mode=="gpu" and torch.cuda.is_initialized() and torch.cuda.max_memory_reserved()>28*2**30:
            raise RuntimeError("Reserved allocator peak exceeds 28GiB")
    def watchdog():
        while not stopped.wait(.5):
            try:
                check()
            except BaseException as exc:
                report.update(status="BUDGET_STOP",error=str(exc)); write(); os._exit(2)
    threading.Thread(target=watchdog,daemon=True).start()
    def progress(stage, **evidence):
        report.update(stage=stage, **evidence)
        write()
    progress("started")
    try:
        if len(process.cpu_affinity())!=1 or any(os.environ.get(k)!="1" for k in
                ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")):
            raise ValueError("Exactly one CPU and pre-import thread limits required")
        if args.mode=="cpu":
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise ValueError("CPU verification requires disabled CUDA")
            suite=unittest.defaultTestLoader.loadTestsFromTestCase(handwritten.FunctionMemoryTests)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(integration.IntegrationTests))
            with (args.output/"unittest.txt").open("w",encoding="utf-8") as stream:
                result=unittest.TextTestRunner(stream=stream,verbosity=2,failfast=True).run(suite)
            report.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                          skipped=len(result.skipped))
            if not result.wasSuccessful():
                raise RuntimeError("Handwritten CPU verification failed")
        else:
            if args.gate is None:
                raise ValueError("Native verification requires the reviewed implementation/CPU gate")
            gate=method.data.read_json(args.gate)
            if (gate.get("status")!="APPROVED_FUNCTION_MEMORY_NATIVE" or gate.get("source_files")!=run.sources()
                    or gate.get("review_disposition")!="NO_OPEN_BLOCKERS"):
                raise ValueError("Native implementation review gate differs")
            cpu=gate["integration_cpu"]
            evidence=method.data.read_json(method.data.verify(method.data.ROOT/cpu["path"],cpu))
            if evidence["status"]!="PASS_HANDWRITTEN_ONLY" or evidence["mode"]!="cpu" or evidence["source_files"]!=run.sources():
                raise ValueError("Source-bound CPU evidence required before native step")
            report["native"]=native(check, progress)
        check()
        report["status"]="PASS_HANDWRITTEN_ONLY"
    except BaseException:
        report.update(status="FAIL",traceback=traceback.format_exc())
        raise
    finally:
        stopped.set(); write()


if __name__=="__main__":
    main()
