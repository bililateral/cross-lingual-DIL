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

import numpy as np
import psutil
import torch

import step28_relation_memory as method

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import test_step28_relation_memory as handwritten


def native(check) -> dict:
    c = method.config()
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    model = method.load_model(c)
    auto = model.encoder[0].auto_model
    if not auto.is_gradient_checkpointing:
        raise ValueError("Native checkpointing not enabled")
    if any(m.p != 0 for m in model.modules() if isinstance(m,torch.nn.modules.dropout._DropoutNd)):
        raise ValueError("Active dropout in native model")
    if auto.config.hidden_dropout_prob != 0 or auto.config.attention_probs_dropout_prob != 0:
        raise ValueError("Active BGE config dropout")
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
    x=method.reference(model,old,c,check)
    # Explicit initial-model handwritten memory fixture, not a trained first stage.
    h,b,constant=method.statistics(torch.from_numpy(x),old.labels)
    memory=method.Memory("ABC")
    memory.h,memory.b,memory.constant,memory.count=h.numpy(),b.numpy(),constant,1
    # Check no_grad versus live canonical coordinates, release this graph before step.
    y=method.relation_features(model,old,c,check)
    difference=float((y.detach().cpu()-torch.from_numpy(x)).abs().max())
    torch.testing.assert_close(y.detach().cpu(),torch.from_numpy(x),atol=1e-5,rtol=1e-4)
    del y
    query_name,query_parameter=next((n,p) for n,p in model.encoder.named_parameters() if "query.weight" in n)
    probes=(query_parameter,model.head[0].weight,model.weight)
    before=[p.detach().clone() for p in probes]
    history_gradients=[]
    def inspect_history(loss):
        grads=torch.autograd.grad(loss,probes,retain_graph=True)
        norms=[float(g.float().norm()) for g in grads]
        if not all(np.isfinite(norms)) or not all(v>0 for v in norms):
            raise ValueError("Historical objective did not reach native encoder/head/w")
        history_gradients.extend(norms)
        del grads
    calls=[0]
    def count_forward(module,args):
        calls[0]+=1
    handle=auto.encoder.layer[0].register_forward_pre_hook(count_forward)
    optimizer=method.make_optimizer(model,c)
    tick=time.monotonic()
    row=method.optimization_step(model,optimizer,current,old,x,memory,c,1e-5,
                                 check=check,observe_history=inspect_history)
    torch.cuda.synchronize()
    elapsed=time.monotonic()-tick
    handle.remove()
    changes=[float((p.detach()-v).abs().max()) for p,v in zip(probes,before)]
    if not all(v>0 for v in changes) or calls[0] <= 224:
        raise ValueError("Missing parameter change or checkpoint recomputation")
    return {"kind":"native_handwritten_first_optimizer_step", "official_data":False,
            "optimizer":row,"input_shapes":lengths,"reference_max_difference":difference,
            "encoder_probe":query_name,"history_only_gradient_norms":history_gradients,
            "probe_max_parameter_changes":changes,"first_layer_calls":calls[0],
            "checkpointing":auto.is_gradient_checkpointing,"update_seconds":elapsed,
            "max_allocated_bytes":torch.cuda.max_memory_allocated(),
            "max_reserved_bytes":torch.cuda.max_memory_reserved(),
            "adam_parameters_with_state":len(optimizer.state),
            "parameters_fp32":all(p.dtype==torch.float32 for p in model.parameters()),
            "note":"Extra history-only derivative diagnosis counted in budget; no native weights saved."}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("cpu","gpu"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    limit=1800 if args.mode=="cpu" else 3600
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
    def write():
        report["elapsed_seconds"]=time.monotonic()-started
        report["peak_rss_bytes"]=peak_rss[0]
        (args.output/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    def check():
        rss=process.memory_info().rss
        peak_rss[0]=max(peak_rss[0],rss)
        if (time.monotonic()-started>limit or rss>64*2**30
                or sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())>128*2**20):
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
    try:
        if args.mode=="cpu":
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise ValueError("CPU verification requires disabled CUDA")
            suite=unittest.defaultTestLoader.loadTestsFromTestCase(handwritten.RelationMemoryTests)
            with (args.output/"unittest.txt").open("w",encoding="utf-8") as stream:
                result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
            report.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                          skipped=len(result.skipped))
            if not result.wasSuccessful():
                raise RuntimeError("Handwritten CPU verification failed")
        else:
            report["native"]=native(check)
        check()
        report["status"]="PASS_HANDWRITTEN_ONLY"
    except BaseException:
        report.update(status="FAIL",traceback=traceback.format_exc())
        raise
    finally:
        stopped.set(); write()


if __name__=="__main__":
    main()
