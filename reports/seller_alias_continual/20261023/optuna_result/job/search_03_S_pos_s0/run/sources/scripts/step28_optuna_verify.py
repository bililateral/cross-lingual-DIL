"""Bounded handwritten qualification and independent saved-result arithmetic."""
from __future__ import annotations
import time
STARTED=time.monotonic()
import argparse
import math
import os
from pathlib import Path
import sys
import threading
import traceback
import unittest

import numpy as np
import psutil
import torch

import step28_optuna_run as run
from step28_optuna import TrialSpec, Memory
from step28_optuna_control import ARMS

data,core,diag,record=run.data,run.core,run.diag,run.method.record


def independent_feedback(job,spec,feedback):
    """Derive the two feedback values from bound B/raw group MAP, not the selector."""
    from step28_optuna_control import S0
    root=job/"evaluation"
    col=data.read_json(root/"collected.json")
    partition=data.read_json(job/"run/partition.json")["development"]
    if (col["group_ids"]!=[g["group_uid"] for g in partition]
            or col["domains"]!=[g["domain"] for g in partition]):raise ValueError("Feedback group alignment differs")
    rec=col["points"]["B_"+spec.arm]["raw"]["matrix"]
    if rec["path"]!="B_"+spec.arm+"_raw_metrics.json":raise ValueError("Feedback must name B/raw, not A/cal")
    matrix=data.read_json(data.verify(root/rec["path"],rec))
    values=np.asarray(matrix["values"],dtype=np.float64)
    if matrix["columns"]!=list(run.parent.metrics.COLUMNS) or values.shape!=(40,22):raise ValueError("Feedback B/raw matrix differs")
    expected={}
    for domain in "AB":
        indices=[i for i,d in enumerate(col["domains"]) if d==domain]
        if len(indices)!=20:raise ValueError("Feedback population differs")
        expected[domain]=float(np.sum(values[indices,matrix["columns"].index("map")],dtype=np.float64)/20)
    if not all(math.isfinite(x) for x in expected.values()):raise ValueError("Nonfinite feedback MAP")
    evaluation=data.read_json(root/"evaluation.json")
    completion=data.read_json(job/"completion.json")
    if completion.get("trial")!=spec.identity() or any(row.get("domains")!=expected for row in (evaluation,completion,feedback)):
        raise ValueError("Feedback is not the saved B/raw domain MAP")
    u=min(expected["A"]-S0["A"],expected["B"]-S0["B"])
    if "value" in feedback and feedback["value"]!=u:raise ValueError("Independent objective differs")
    return dict(domains=expected,value=u,matrix=rec)


def independent_analysis(job):
    """Saved group matrices only; independent means/quantiles, no model or labels."""
    selected=data.read_json(job/"selected.json")
    search=data.read_json(job/"search.json")
    from step28_optuna_control import S0
    for arm in ARMS:
        rows=search[arm]
        if len(rows)!=12 or sorted(r["number"] for r in rows)!=list(range(12)):raise ValueError("Missing search slot")
        for row in rows:
            spec=TrialSpec.from_dict(row["spec"])
            if row["state"]!="COMPLETE" or spec.arm!=arm or row["number"]!=spec.number:raise ValueError("Search identity differs")
            independent_feedback(job/spec.run_id,spec,row)
        winner=max(rows,key=lambda r:(min(r["domains"][d]-S0[d] for d in "AB"),
            ((r["domains"]["A"]-S0["A"])+(r["domains"]["B"]-S0["B"]))/2,-r["number"]))
        if selected[arm]!=winner:raise ValueError("Independent frozen selector differs")
    specs=[TrialSpec.from_dict(r["trial"]) for r in data.read_json(job/"replications.json")]
    rows_checked=0;maximum_error=0.
    def check(value,expected):
        nonlocal rows_checked,maximum_error
        if expected is None:
            if value is not None:raise ValueError("Undefined value was silently imputed")
        else:
            delta=float(np.max(np.abs(np.asarray(value)-np.asarray(expected))))
            maximum_error=max(maximum_error,delta)
            if delta>1e-12:raise ValueError("Saved analysis arithmetic differs")
        rows_checked+=1
    rng=np.random.Generator(np.random.PCG64(20260930));draws=rng.integers(20,size=(5000,2,20))
    matrices={};groups=None
    def audit_summary(values,saved,rows,domains="AB"):
        for dindex,(domain,indices) in enumerate(zip("AB",rows)):
            if domain not in domains:continue
            for k,name in enumerate(run.parent.metrics.COLUMNS):
                v=values[indices,k];undefined=np.isnan(v).any()
                check(saved[domain][name]["mean"],None if undefined else float(np.sum(v)/20))
                check(saved[domain][name]["conditional_95pct_interval"],None if undefined else
                      np.quantile(np.sum(v[draws[:,dindex,:]],axis=1)/20,[.025,.975]).tolist())
    all_specs=[TrialSpec.from_dict(r["spec"]) for arm in ARMS for r in search[arm]]+specs
    for spec in all_specs:
        root=job/spec.run_id/"evaluation";col=data.read_json(root/"collected.json")
        if groups is None:groups=(col["group_ids"],col["domains"])
        if (col["group_ids"],col["domains"])!=groups:raise ValueError("Saved populations differ")
        rows=[np.flatnonzero(np.asarray(col["domains"])==d) for d in "AB"]
        arrays=run.load_arrays(root,col);saved=data.read_json(root/"evaluation.json")
        for point,roles in arrays.items():
            for role,values in roles.items():audit_summary(values,saved["absolute"][point][role]["by_domain"],rows)
        for role in run.parent.ROLES:
            delta=arrays["B_"+spec.arm][role]-arrays["A_"+spec.arm][role]
            audit_summary(delta,{"A":saved["old_changes"][role]},rows,domains="A")
        if spec.phase=="replication":matrices[spec.arm,spec.seed]=arrays
    joint=data.read_json(job/"joint/evaluation.json")
    for arm,opponent in (("S_pos","S"),("H_pos","S"),("H_pos","S_pos")):
        contrasts=joint["comparisons"][arm+"-"+opponent];six=[]
        for seed in (1,2,3):
            for phase in "AB":
                for role in run.parent.ROLES:
                    delta=matrices[arm,seed][phase+"_"+arm][role]-matrices[opponent,seed][phase+"_"+opponent][role]
                    audit_summary(delta,contrasts["by_seed"][str(seed)][phase][role],rows)
                    if phase=="B" and role=="raw":six.extend(float(np.sum(delta[r,12])/20)>0 for r in rows)
        for phase in "AB":
            for role in run.parent.ROLES:
                delta=sum(matrices[arm,s][phase+"_"+arm][role]-matrices[opponent,s][phase+"_"+opponent][role] for s in (1,2,3))/3
                audit_summary(delta,contrasts["three_seed_mean"][phase][role],rows)
        if opponent=="S" and joint["development_direction_consistency"][arm]!=all(six):raise ValueError("Six-cell development rule differs")
    for arm in ARMS:
        diagnostic=joint["diagnostics"][arm]
        if set(diagnostic["by_seed"])!={"1","2","3"} or set(diagnostic["three_fixed_seed_point_means"])!={"A","B"}:
            raise ValueError("Missing seed/phase diagnostic")
        for seed in (1,2,3):
            saved=diagnostic["by_seed"][str(seed)]
            original=data.read_json(data.verify(job/saved["source"]["path"],saved["source"]))
            if saved["points"]!=original["points"]:raise ValueError("Per-seed diagnostic source differs")
            spec=next(s for s in specs if s.arm==arm and s.seed==seed)
            root=job/spec.run_id;manifest=data.read_json(root/"run/manifest.json")
            for phase in "AB":
                rec=manifest["diagnostics"][phase+"_"+arm][0]
                capture=data.read_json(data.verify(root/"run/diagnostics"/rec["path"],rec))
                if saved["populations"][phase]!={k:capture[k] for k in ("fit_ids","fit_domains","cache_ids","cache_domains","valid_ids")}:
                    raise ValueError("Seed diagnostic populations differ")
        for phase,roles in diagnostic["three_fixed_seed_point_means"].items():
            if set(roles)!={"fit","cache"} or set(roles["fit"])!={phase} or set(roles["cache"])!=({"A"} if phase=="B" else set()):
                raise ValueError("Missing diagnostic comparison")
            for role,domains in roles.items():
                for domain,fields in domains.items():
                    original=[diagnostic["by_seed"][str(s)]["points"][phase+"_"+arm][role][domain] for s in (1,2,3)]
                    expected_keys={"train_raw_curve","valid_raw_curve","benefit_gap_curve",
                        "epoch1_to_6_train_benefit","epoch1_to_6_valid_benefit","epoch1_to_6_gap_widening"}
                    if set(fields)!=expected_keys:raise ValueError("Missing descriptive diagnostic fields")
                    for key,value in fields.items():
                        vals=[r[key]["mean"] if key.startswith("epoch") else r[key] for r in original]
                        check(value,(np.asarray(vals[0])+np.asarray(vals[1])+np.asarray(vals[2]))/3)
    out=job/"analysis";out.mkdir()
    data.write_json(out/"audit.json",dict(status="PASS_SAVED_ARITHMETIC",checked_fields=rows_checked,
        maximum_absolute_error=maximum_error,source_files=run.sources(),formal_label_reads=0,model_loads=0,
        limits="Checks saved group-metric aggregation and selection; does not independently reconstruct label-level metrics or claim independent holdout confirmation."))


def native_arm(out,arm,check,progress,continuing):
    """Exactly six real calls; Q0/QB cross-parameter replay is a handwritten fixture only."""
    from step28_positive_ranking_verify import native_preflight
    from step28_g_half_verify import native_save,state,disk_reserve
    from step28_record_replay_numerics import equal_state
    from test_step28_record_replay import handmade_group
    resources=native_preflight(continuing)
    disk_reserve(out,0,10*2**30,10*2**30)
    run.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    entries=[run.Entry(TrialSpec(arm,"handwritten",i,w,lr)) for i,(w,lr) in enumerate(((1,1),(.5,1),(.5,.5)))]
    measurements={};updates=[];gradients={};effects={}
    def timed(name,fn):
        check();torch.cuda.synchronize();start=time.monotonic();value=fn();torch.cuda.synchronize()
        measurements[name]=max(measurements.get(name,0.),time.monotonic()-start);return value
    c=entries[0].config()
    def checked_fresh():
        archive=entries[0].pretrained_archive(c)
        return (*entries[0].fresh(c),archive)
    model,optimizer,initial,archive=timed("L",checked_fresh)
    if not entries[0].validate_initial(model,optimizer,initial,entries[0].baseline_inputs()[0]):raise ValueError("Native initial differs")
    entries[0].method.bind(model,entries[0].method.metadata(arm))
    current,history=[handmade_group("native_"+role,8,254) for role in ("current","history")]
    for g in (current,history):
        lengths=[len(x) for x in model.encoder.tokenizer(run.old.base.record_texts(g,"separate_moments"),padding=False,truncation=False)["input_ids"]]
        if len(lengths)!=448 or set(lengths)!={256}:raise ValueError("Native maximum N224/token256 shape differs")
    if not model.encoder[0].auto_model.is_gradient_checkpointing or not torch.cuda.is_bf16_supported():raise ValueError("Native execution mode differs")
    probes={"encoder":model.encoder[0].auto_model.encoder.layer[0].attention.self.query.weight,"head":model.head[0].weight}
    work=out/"work";work.mkdir();receipts={}
    def save(name,entry):
        meta=dict(**entry.method.metadata(arm),rng=diag.rng_state(),modes=[v.training for v in model.modules()],fixture=name)
        receipt=timed("s_F",lambda:native_save(work/(name+".pt"),model,optimizer,meta,check))
        receipts[name]=(receipt,meta);return name
    def reload(name,entry):
        receipt,expected=receipts[name]
        meta=timed("r_F",lambda:core.restore_state(work/(name+".pt"),model,optimizer,receipt["state_sha256"]))
        if meta!=expected:raise ValueError("Native fixture restore differs")
        for module,mode in zip(model.modules(),meta["modes"],strict=True):module.training=mode
        diag.restore_rng(meta["rng"])
        # Deliberately enumerated Q0/QB fixture comparison; production Entry never permits this rebind.
        entry.method.bind(model,entry.method.metadata(arm))
    def update(entry,stage,step,target=None):
        if len(updates)>=6:raise ValueError("Six native calls exhausted")
        pre={}
        def observe(role,log):pre[role]={k:p.grad.detach().cpu().clone() for k,p in probes.items()}
        progress("call_started",actual_updates=len(updates),attempted_calls=len(updates)+1)
        row=timed("t_"+"AB"[stage-1],lambda:entry.method.update(model,optimizer,current,history if target is not None else None,target,
            entry.config(),arm,stage,step,check,observe))
        entry.loss_total(row,stage,arm);updates.append(row);gradients[len(updates)]=pre
        progress("call_completed",actual_updates=len(updates),updates=updates,measurements=measurements)
        return row
    parameter_snapshots={};moment_snapshots={}
    def parameters():return {k:p.detach().cpu().clone() for k,p in probes.items()}
    def moments():return {k:{field:optimizer.state[p][field].detach().cpu().clone() for field in ("step","exp_avg","exp_avg_sq")} for k,p in probes.items()}
    def lambda_state_effect(call,component):
        p=probes[component]
        deltas={key:float((moment_snapshots[call][component][key]-optimizer.state[p][key].detach().cpu()).abs().max()) for key in ("exp_avg","exp_avg_sq")}
        if deltas["exp_avg"]<=0:raise ValueError("Lambda changed no actual Adam first moment")
        return dict(parameter_delta=float((parameter_snapshots[call][component]-p.detach().cpu()).abs().max()),adam_moment_deltas=deltas)
    save("Q0",entries[0]);update(entries[0],1,1);parameter_snapshots[1]=parameters();moment_snapshots[1]=moments();save("Q1",entries[0])
    reload("Q0",entries[1]);update(entries[1],1,1)
    for component in probes:
        delta=float((gradients[1]["current"][component]-gradients[2]["current"][component]).abs().max())
        if delta<=0:raise ValueError("Missing actual A lambda gradient effect")
        effects[component]={"A_current_preclip_delta":delta}
        effects[component]["A_state_effect"]=lambda_state_effect(1,component)
    data.write_json(out/"Q0_retirement.json",data.record(work/"Q0.pt",out));(work/"Q0.pt").unlink()
    reload("Q1",entries[0])
    for value in optimizer.state.values():value["step"].fill_(336)
    with diag.neutral(model):table=timed("v",lambda:record.reference(model,history,c,check))
    i,j=np.triu_indices(224,1);target=(table+2*np.sin(i*.31+j*.71)).astype(np.float32)
    save("QB",entries[0])
    data.write_json(out/"Q1_retirement.json",data.record(work/"Q1.pt",out));(work/"Q1.pt").unlink()
    update(entries[0],2,49,target);parameter_snapshots[3]=parameters();moment_snapshots[3]=moments()
    reload("QB",entries[1]);update(entries[1],2,49,target)
    for component in probes:
        for role in ("current","history"):
            a=gradients[3][role][component]-(gradients[3]["current"][component] if role=="history" else 0)
            b=gradients[4][role][component]-(gradients[4]["current"][component] if role=="history" else 0)
            delta=float((a-b).abs().max())
            if delta<=0:raise ValueError("Missing actual B branch lambda effect")
            effects[component]["B_"+role+"_preclip_delta"]=delta
        effects[component]["B_state_effect"]=lambda_state_effect(3,component)
    lr_full={k:p.detach().cpu().clone() for k,p in probes.items()}
    lr_moments=moments()
    reload("QB",entries[2]);first=update(entries[2],2,49,target)
    for k,p in probes.items():
        if not torch.equal(gradients[4]["history"][k],gradients[5]["history"][k]):raise ValueError("LR-only forward/gradient changed")
        for field in ("step","exp_avg","exp_avg_sq"):
            if not torch.equal(lr_moments[k][field],optimizer.state[p][field].detach().cpu()):raise ValueError("LR-only Adam moments changed")
        delta=float((lr_full[k]-p.detach().cpu()).abs().max())
        if delta<=0:raise ValueError("LR had no actual update effect")
        effects[k]["LR_parameter_delta"]=delta
    expected=state(model,optimizer);expected_rng=diag.rng_state()
    reload("QB",entries[2]);entry=entries[2]
    fit=[handmade_group(f"native_fit_{k}") for k in range(48)];valid=[data.Group(g.uid,g.sellers,g.items) for g in [handmade_group(f"native_valid_{k}") for k in range(40)]]
    memory=Memory(entry.spec);memory.retain(fit,1,lambda g:np.sin(np.arange(1540)).astype(np.float32));memory.maps.update(entry.method.metadata(arm));memory.begin_stage(2)
    for _ in range(48):memory.draw()
    payload=memory.to_bytes();memory=entry.memory_from(payload,arm)
    droot=out/"diagnostics";droot.mkdir()
    entry.method.capture(model,optimizer,fit,memory,valid,entry.config(),arm,2,1,droot,"handwritten",check)
    second=update(entry,2,49,target);equality={"tensors":0,"elements":0};equal_state(expected,state(model,optimizer),equality)
    if first!=second or expected_rng!=diag.rng_state() or payload!=memory.to_bytes():raise ValueError("Next update/RNG/Memory differs")
    del expected
    with diag.neutral(model):scores=timed("v",lambda:entry.method.restored_score(model,[history],entry.config(),check))
    save("actual",entry);reload("actual",entry)
    with diag.neutral(model):np.testing.assert_array_equal(scores,timed("v",lambda:entry.method.restored_score(model,[history],entry.config(),check)))
    infer_path=work/"inference.pt";meta=receipts["actual"][1]
    infer=timed("s_I",lambda:native_save(infer_path,model,None,meta,check))
    timed("r_I",lambda:entry.restore_point(infer_path,infer,model,None,arm))
    with diag.neutral(model):np.testing.assert_array_equal(scores,entry.method.restored_score(model,[history],entry.config(),check))
    def residual():
        result=[]
        for seed in (None,data.seed_for(20261008,"ABC",2,history.uid,"diagnostic_regroup")):
            slots=record.assignment(history,seed)
            result.append(float((entry.method.account_logits(torch.from_numpy(table),slots,arm)-entry.method.account_logits(torch.from_numpy(target),slots,arm)).double().square().mean()))
        return result
    residuals=timed("h",residual)
    measurements["W_F"]=max(p.stat().st_size for p in work.glob("*.pt"));measurements["W_I"]=infer_path.stat().st_size
    identities=[data.record(p,out) for p in sorted(work.glob("*.pt"))];data.write_json(out/"temporary_states.json",identities)
    for p in work.glob("*.pt"):p.unlink()
    work.rmdir()
    return dict(actual_updates=len(updates),resources=resources,pretrained_archive=archive,initial=initial,updates=updates,effects=effects,
        measurements=measurements,old_cost_projection=run.single.projection(measurements),residuals=residuals,
        current_history_lambda_effect=True,actual_lr_effect=True,next_update_bitwise_equal=True,inference_restore_equal=True,
        maximum_shape_verified=True,equality=equality,fixture="Six real calls; QB step336 is synthetic. Old projection is diagnostic, not new 9000-second qualification.")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path);parser.add_argument("--prior-seconds",type=float,default=0)
    parser.add_argument("--tests",nargs="*");parser.add_argument("--analyze",type=Path)
    parser.add_argument("--native-gate",type=Path)
    args=parser.parse_args()
    if os.name!="posix":parser.error("Linux existing py310 only")
    if args.analyze:
        with run.shared_resources(args.analyze,"analysis"):
            independent_analysis(args.analyze)
        return
    out=args.output.resolve()
    if out.exists() or not 0<=args.prior_seconds<300:parser.error("New output and cumulative prior required")
    native=args.native_gate is not None
    out.mkdir(parents=True);sys.path.insert(0,str(data.ROOT/"tests"));os.environ["OPTUNA_TEST_ROOT"]=str(out)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);process=psutil.Process()
    started=STARTED-max(0.,time.time()-float(os.environ.get("QUALIFICATION_STARTED_EPOCH",time.time()))-(time.monotonic()-STARTED))
    report=dict(status="RUNNING",mode="gpu" if native else "cpu",source_files=run.sources(),prior_charged_seconds=args.prior_seconds,
        peak_rss_bytes=0,peak_output_bytes=0,formal_inputs=False,formal_labels=False)
    stop=threading.Event();limit=1800 if native else 300;arm_deadline=[None]
    def write():
        report.update(elapsed_seconds=time.monotonic()-started,cumulative_charged_seconds=args.prior_seconds+math.ceil(time.monotonic()-started))
        data.write_json(out/"result.json",report)
    def check(reserve=0):
        report["peak_rss_bytes"]=max(report["peak_rss_bytes"],process.memory_info().rss)
        size=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        report["peak_output_bytes"]=max(report["peak_output_bytes"],size+reserve)
        if native:
            from step28_g_half_verify import disk_reserve
            disk_reserve(out,size,reserve,10*2**30)
        if (args.prior_seconds+time.monotonic()-started>=limit or arm_deadline[0] is not None and time.monotonic()>=arm_deadline[0]
            or report["peak_rss_bytes"]>(64 if native else 2)*2**30
            or size+reserve>(10*2**30 if native else 64*2**20)):raise RuntimeError("Qualification cumulative resource limit")
    def watchdog():
        while not stop.wait(.2):
            try:check()
            except BaseException:
                report.update(status="BUDGET_STOP",traceback=traceback.format_exc());write();os._exit(2)
    watcher=threading.Thread(target=watchdog,daemon=True);watcher.start()
    try:
        if len(process.cpu_affinity())!=1 or any(os.environ.get(k)!="1" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")) or os.environ.get("TOKENIZERS_PARALLELISM")!="false":raise ValueError("Pre-import single CPU/thread required")
        if not native:
            if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU qualification requires hidden GPU")
            from test_step28_optuna import OptunaTests
            tests=args.tests or list(run.CPU_TESTS)
            with (out/"unittest.txt").open("w",encoding="utf-8") as stream:
                result=unittest.TextTestRunner(stream=stream,verbosity=2,failfast=True).run(unittest.TestSuite(OptunaTests(n) for n in tests))
            report.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
            if not result.wasSuccessful() or result.skipped:raise RuntimeError("Affected CPU qualification failed")
            report.update(qualified_tests=tests,unresolved_checks=[])
        else:
            gate=data.read_json(args.native_gate)
            if args.prior_seconds or gate.get("kind")!="NATIVE" or gate.get("status")!="APPROVED_NO_OPEN_BLOCKERS" or gate.get("source_files")!=run.sources():raise ValueError("Unique source-bound native permission required")
            for key in ("external_review","primary_disposition","cpu","execution_authorization"):
                data.verify(data.ROOT/gate[key]["path"],gate[key])
            permission=data.read_json(data.ROOT/gate["execution_authorization"]["path"])
            if permission.get("status")!="APPROVED" or permission.get("stages")!=["NATIVE"] or permission.get("source_files")!=run.sources() or not permission.get("user_quote"):raise ValueError("Specific native execution permission required")
            cpu=data.read_json(data.ROOT/gate["cpu"]["path"])
            if cpu.get("status")!="PASS_HANDWRITTEN_ONLY" or cpu.get("source_files")!=run.sources() or cpu.get("cumulative_charged_seconds",math.inf)>300 or cpu.get("unresolved_checks")!=[] or set(cpu.get("qualified_tests",[]))!=set(run.CPU_TESTS):raise ValueError("Current bounded CPU evidence required")
            report["arms"]={}
            for arm in ARMS:
                tick=time.monotonic();arm_deadline[0]=tick+600;arm_out=out/arm;arm_out.mkdir()
                def arm_check(reserve=0):
                    check(reserve)
                    if time.monotonic()-tick>=600:raise RuntimeError("Per-arm native deadline")
                def progress(stage,**details):report.update(stage=stage,active_arm=arm,**details);write()
                value=native_arm(arm_out,arm,arm_check,progress,bool(report["arms"]))
                arm_check();value["charged_seconds"]=math.ceil(time.monotonic()-tick);report["arms"][arm]=value;arm_deadline[0]=None;write()
        check();report["status"]="PASS_HANDWRITTEN_ONLY"
    except BaseException:
        report.update(status="FAIL",traceback=traceback.format_exc());raise
    finally:stop.set();watcher.join();write()


if __name__=="__main__":main()
