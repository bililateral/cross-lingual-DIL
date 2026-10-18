"""Affected trial paths only; handmade inputs, no corpus/model archive access."""
import copy
import itertools
import math
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import step28_optuna as m
import step28_optuna_control as ctl
import step28_optuna_run as run
from test_step28_record_replay import handmade_group, tiny_model
from test_step28_positive_ranking import independent_pool, independent_rank, independent_loss, PAIRS, LOOKUP
from step28_g_half_verify import state, restore

data,record,core,diag=m.data,m.record,m.core,m.diag


def reference(table,group,seed,spec,target=None,weight=None):
    predictions=[independent_pool(table,a,spec.arm) for a in (record.assignment(group),record.assignment(group,seed))]
    losses=[]
    for scores in predictions:
        new=independent_rank(scores,group.labels)
        rank=new
        if spec.arm=="S":
            terms=[]
            for q in range(28):
                ids=[LOOKUP[tuple(sorted((q,j)))] for j in range(28) if j!=q]
                pos=[k for k in ids if group.labels[k]]
                # Original supervised contrastive rank averages positive log probabilities.
                terms.append(torch.logsumexp(scores[ids],0)-scores[pos].mean())
            rank=torch.stack(terms).mean()
        losses.append(independent_loss(scores,group)-new+(spec.lambda_rank if weight is None else weight)*rank)
    result=sum(losses)/2
    if target is not None:
        teacher=independent_pool(torch.as_tensor(target,dtype=table.dtype),record.assignment(group),spec.arm)
        result=.1*result+.5*(predictions[0]-teacher).square().mean()
    return result


class OptunaTests(unittest.TestCase):
    def temporary(self):
        t=tempfile.TemporaryDirectory(dir=os.environ["OPTUNA_TEST_ROOT"]);self.addCleanup(t.cleanup)
        return Path(t.name)

    def evidence(self,name,value):data.write_json(Path(os.environ["OPTUNA_TEST_ROOT"])/(name+".json"),value)

    def test_actual_objective_vjp_lr_and_rejected_omissions(self):
        current,history=handmade_group("current"),handmade_group("history")
        target=(np.sin(np.arange(1540)/7)*3).astype(np.float32);evidence={}
        for arm in m.ARMS:
            spec=m.TrialSpec(arm,"handwritten",1,.5,.5,1);method=m.Method(spec);c=m.config(spec)
            proxy=torch.tensor(np.sin(np.arange(1540)*.3),dtype=torch.float64,requires_grad=True)
            for g,t in ((current,None),(history,target)):
                actual,_=method.objective(proxy,g,17,arm,t);expected=reference(proxy,g,17,spec,t)
                torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
                ga,=torch.autograd.grad(actual,proxy,retain_graph=True);ge,=torch.autograd.grad(expected,proxy,retain_graph=True)
                torch.testing.assert_close(ga,ge,atol=1e-12,rtol=1e-12)
                mutant=reference(proxy,g,17,spec,t,weight=1.)
                gm,=torch.autograd.grad(mutant,proxy,retain_graph=True)
                self.assertGreater(float((ga-gm).abs().max()),1e-6)
            model=tiny_model();direct=copy.deepcopy(model);method.bind(model,method.metadata(arm))
            oa,ob=[core.make_optimizer(v,c) for v in (model,direct)]
            observed={}
            def observe(role,log):observed[role]={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            row=method.update(model,oa,current,history,target,c,arm,2,49,observe=observe)
            run.Entry(spec).loss_total(row,2,arm)
            direct.train();ob.zero_grad(set_to_none=True);errors={}
            for role,g,t in (("current",current,None),("history",history,target)):
                torch.manual_seed(data.seed_for(spec.roots["schedule"],"ABC",2,49,role,"dropout"))
                z=record.record_vectors(direct,g,c);i,j=torch.triu_indices(len(z),len(z),1)
                table=direct.head(torch.cat(((z[i]-z[j]).abs(),z[i]*z[j]),1)).flatten()
                seed=data.seed_for(spec.roots["schedule"],"ABC",2,49,role,g.uid,"regroup")
                reference(table,g,seed,spec,t).backward()
                errors[role]=0.
                for n,p in direct.named_parameters():
                    torch.testing.assert_close(observed[role][n],p.grad,atol=2e-6,rtol=2e-5)
                    errors[role]=max(errors[role],float((observed[role][n]-p.grad).abs().max()))
            torch.nn.utils.clip_grad_norm_(direct.parameters(),1.)
            ob.param_groups[0]["lr"],ob.param_groups[1]["lr"]=spec.rates(49);ob.step()
            for a,b in zip(model.parameters(),direct.parameters()):
                torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
                for key in ("step","exp_avg","exp_avg_sq"):
                    torch.testing.assert_close(oa.state[a][key],ob.state[b][key],atol=2e-6,rtol=2e-5)
            rates={}
            for step in (1,29,30,288):
                oa.param_groups[0]["lr"],oa.param_groups[1]["lr"]=99.,99.
                r=method.update(model,oa,current,None,None,c,arm,1,step)
                factor=step/29 if step<=29 else (288-step)/259
                self.assertEqual((r["encoder_lr"],r["head_lr"]),(.5*1e-5*factor,.0005))
                self.assertEqual(tuple(g["lr"] for g in oa.param_groups),spec.rates(step));rates[str(step)]=spec.rates(step)
            default=m.Method(m.TrialSpec(arm,"handwritten",0))
            actual,_=default.objective(proxy,current,17,arm)
            prior,_=(record.objective(proxy,current,17,"S") if arm=="S" else m.positive.objective(proxy,current,17,arm))
            torch.testing.assert_close(actual,prior,atol=1e-12,rtol=1e-12)
            evidence[arm]=dict(independent_preclip_errors=errors,rates=rates,actual_optimizer_calls=5,
                lambda_log_only_and_history_omission_rejected=True,default_objective_compatible=True)
        self.evidence("actual_objective_vjp",evidence)

    def test_seeds_own_memory_capture_and_restore(self):
        out=self.temporary();groups=[handmade_group(f"fit_{i:02}") for i in range(48)]
        evidence={};streams=[]
        for seed in range(4):
            spec=m.TrialSpec("S_pos","handwritten",seed,.5,.5,seed);c=m.config(spec)
            schedule,_=m.parent.schedule(groups,c,"ABC",1)
            expected=data.schedule(groups,6,data.seed_for(spec.roots["schedule"],"ABC",1,"current"))
            self.assertEqual([g.uid for g in schedule],[g.uid for g in expected]);streams.append([g.uid for g in schedule])
            memory=m.Memory(spec);memory.retain(groups,1,lambda g:np.sin(np.arange(1540)).astype(np.float32))
            memory.maps.update(m.Method(spec).metadata(spec.arm));members=[]
            rng=random.Random(data.seed_for(spec.roots["memory"],"ABC","retention"))
            for seen,g in enumerate(sorted(groups,key=lambda g:g.uid),1):
                k=seen-1 if seen<=6 else rng.randrange(seen)
                if k<6:
                    if k==len(members):members.append(g.uid)
                    else:members[k]=g.uid
            self.assertEqual(memory.summary()["members"],members)
            memory.begin_stage(2);expected_rng=random.Random(data.seed_for(spec.roots["memory"],"ABC",2,"history_draws"))
            ids=[memory.draw()[0].uid for _ in range(48)]
            self.assertEqual(ids,[expected_rng.choice(members) for _ in range(48)])
            payload=memory.to_bytes();restored=m.Memory.from_bytes(payload,spec)
            self.assertEqual(memory.draw()[0].uid,restored.draw()[0].uid)
            with self.assertRaises(ValueError):m.Memory.from_bytes(payload,m.TrialSpec("S_pos","handwritten",seed,.6,.5,seed))
            evidence[str(seed)]=dict(roots=spec.roots,retained=members,schedule_digest=__import__("hashlib").sha256(data.json_bytes(streams[-1])).hexdigest())
        self.assertEqual(len({tuple(s) for s in streams}),4)
        spec=m.TrialSpec("H_pos","handwritten",1,.5,.5,1);entry=run.Entry(spec);method=entry.method;c=entry.config()
        model=tiny_model();method.bind(model,method.metadata(spec.arm));optimizer=core.make_optimizer(model,c)
        current,history=groups[0],groups[1];target=np.sin(np.arange(1540)).astype(np.float32)
        method.update(model,optimizer,current,None,None,c,spec.arm,1,1)
        for v in optimizer.state.values():v["step"].fill_(336)
        memory=m.Memory(spec);memory.retain(groups,1,lambda g:target);memory.maps.update(method.metadata(spec.arm));memory.begin_stage(2)
        for _ in range(48):memory.draw()
        payload=memory.to_bytes();meta={**method.metadata(spec.arm),"rng":diag.rng_state(),"modes":[x.training for x in model.modules()]}
        p=out/"full.pt";receipt=core.save_state(p,model,optimizer,meta)
        first=method.update(model,optimizer,current,history,target,c,spec.arm,2,49)
        expected=core.state_digest(state(model,optimizer));rng=diag.rng_state()
        entry.restore_point(p,receipt,model,optimizer,spec.arm)
        valid=[data.Group(g.uid,g.sellers,g.items) for g in groups[:40]]
        capture=method.capture(model,optimizer,groups,memory,valid,c,spec.arm,2,1,out,"handwritten")
        entry.validate_diagnostic(data.read_json(out/capture["path"]),spec.arm)
        self.assertEqual(memory.to_bytes(),payload)
        second=method.update(model,optimizer,current,history,target,c,spec.arm,2,49)
        self.assertEqual(first,second);self.assertEqual(expected,core.state_digest(state(model,optimizer)));self.assertEqual(rng,diag.rng_state())
        with diag.neutral(model):before=method.restored_score(model,[current],c)
        p=out/"infer.pt";receipt=core.save_state(p,model,None,meta);entry.restore_point(p,receipt,model,None,spec.arm)
        with diag.neutral(model):np.testing.assert_array_equal(before,method.restored_score(model,[current],c))
        wrong=run.Entry(m.TrialSpec("H_pos","handwritten",1,.6,.5,1))
        with self.assertRaises(ValueError):wrong.restore_point(p,receipt,model,None,spec.arm)
        heads=[run.Entry(m.TrialSpec("S","handwritten",s,1,1,s)).expected_head() for s in range(4)]
        self.assertEqual(len(set(heads)),4)
        evidence["restore"]=dict(actual_optimizer_calls=3,full_inference_and_next_update_equal=True,actual_capture=[48,6,40],foreign_identity_rejected=True,heads=heads)
        self.evidence("seeds_memory_restore",evidence)

    def test_real_optuna_selection_and_stop(self):
        search=ctl.Search("sqlite:///"+str(self.temporary()/"studies.sqlite3"),ctl.S0)
        for number in range(12):
            for arm in search.order(number):
                spec=search.ask(arm,number)
                result=dict(status="SEARCH_COMPLETE",trial=spec.identity(),child_exit_code=0,resources_released=True,
                    access=dict(train=1,valid=1,heldout=0,owners=0),
                    anchor_compatible=True,domains={"A":.50+number*.001,"B":.51+number*.002},completion_identity={"handwritten":True})
                search.tell(result)
        selected=search.selected()
        for number in (1,2,3):
            self.assertEqual(len({(search.rows[a][number]["spec"]["lambda_rank"],search.rows[a][number]["spec"]["lr_multiplier"]) for a in m.ARMS}),1)
        self.assertEqual(len({id(s.sampler) for s in search.studies.values()}),3)
        rows=[dict(number=i,state="COMPLETE",domains={"A":0.,"B":float(i)},value=0.) for i in range(12)]
        self.assertEqual(ctl.select_trial(rows,{"A":0.,"B":0.})["number"],11)
        rows[10]["domains"]["B"]=11.;self.assertEqual(ctl.select_trial(rows,{"A":0.,"B":0.})["number"],10)
        specs=ctl.replication_specs(selected)
        records=[dict(status="REPLICATION_BLIND_COMPLETE",trial=s.identity(),access=dict(train=1,valid=0,heldout=0,owners=0),child_exit_code=0,resources_released=True) for s in specs]
        ctl.joint_opening(records,specs)
        with self.assertRaises(ValueError):ctl.joint_opening(records[:-1],specs[:-1])
        records[0]["access"]["valid"]=1
        with self.assertRaises(ValueError):ctl.joint_opening(records,specs)
        bad=ctl.Search(None,ctl.S0);s=bad.ask("S",0)
        result.update(trial=s.identity(),domains={"A":float("nan"),"B":.5})
        with self.assertRaises(ValueError):bad.tell(result)
        self.assertTrue(bad.failed);self.assertEqual(bad.studies["S"].trials[0].state.name,"FAIL")
        with self.assertRaises(ValueError):bad.ask("S_pos",0)
        self.evidence("optuna_control",dict(actual_handwritten_slots=36,selected=selected,rows=search.rows,nan_stops_all=True,ties_use_mean_then_number=True,nine_blind_gate=True))

    def test_disjoint_budgets_and_real_launcher_failure(self):
        now=[0.];ledger=ctl.Ledger(lambda:now[0],prior_cpu=20,prior_native=100)
        for i in range(8):
            with ledger.interval("replication",arm=m.ARMS[i%3],run_id=str(i)):now[0]+=100
            now[0]+=1000
        self.assertEqual(ledger.used["replication"],800)
        with ledger.interval("evaluation"):now[0]+=200
        self.assertEqual(sum(ledger.used.values()),1000)
        with self.assertRaises(ValueError):
            with ledger.interval("control"):
                with ledger.interval("analysis"):pass
        self.assertTrue(ledger.failed)
        ledger=ctl.Ledger(lambda:now[0])
        with self.assertRaises(TimeoutError):
            with ledger.interval("control"):now[0]+=301
        self.assertEqual(ledger.used["control"],301);self.assertTrue(ledger.failed)
        projected=dict(arms={a:dict(search=8000,replication=8000) for a in m.ARMS},shared=dict(ctl.SHARED))
        ctl.projected_gate(projected);projected["arms"]["S_pos"]["search"]=18844.60961239063
        with self.assertRaises(ValueError):ctl.projected_gate(projected)
        from step28_optuna_search import launch
        import sys
        out=self.temporary()
        receipt=launch([sys.executable,"-c","print('handwritten child exits')"],out/"ok.txt",5,os.environ.copy())
        self.assertEqual(receipt["child_exit_code"],0)
        with self.assertRaises(RuntimeError):launch([sys.executable,"-c","raise SystemExit(3)"],out/"fail.txt",5,os.environ.copy())
        with self.assertRaises(TimeoutError):launch([sys.executable,"-c","import time; time.sleep(5)"],out/"timeout.txt",.2,os.environ.copy())
        self.evidence("budgets",dict(waiting_not_charged=True,evaluation_once=True,old_pass_rejected=True,hard_timeout_and_failure=True,failed_seconds_retained=301))

    def test_entry_orchestration_and_access_negative_controls(self):
        # This isolates the real 288-step caller and actual config; updates here are deliberately simulated.
        spec=m.TrialSpec("S","handwritten",1,.5,.5,2);entry=run.Entry(spec);out=self.temporary();(out/"updates").mkdir()
        groups=[handmade_group(f"current_{i}") for i in range(48)];calls=[]
        class Budget:
            phase=""
            def check(self,*a):pass
            def snapshot(self):return {}
        def update(*args):
            calls.append(args[2].uid);return dict(current={},history_uid=None)
        with mock.patch.object(entry.method,"update",side_effect=update),mock.patch.object(entry.method,"capture",return_value={}),mock.patch.object(run.parent,"adam_step",side_effect=lambda o:len(calls)):
            log,captures=entry.train_stage(None,None,groups,None,[],entry.config(),"S",1,out,Budget())
        expected,_=m.parent.schedule(groups,entry.config(),"ABC",1)
        self.assertEqual(calls,[g.uid for g in expected]);self.assertEqual(len(captures),6)
        # No valid loader may run if the actual entry's blind gate rejects an incomplete B.
        data.write_json(out/"access.json",dict(train=1,valid=0,heldout=0,owners=0))
        with mock.patch.object(entry,"blind_gate",side_effect=ValueError("incomplete B")),mock.patch.object(entry,"attach_labels") as loader:
            with self.assertRaises(ValueError):entry.open_valid(out,{},[],[],entry.config())
            loader.assert_not_called()
        self.evidence("entry",dict(simulated_updates=288,actual_config_schedule_used=True,incomplete_B_prevents_supervision=True,actual_optimizer_calls=0))

    def test_nine_saved_evaluation_and_independent_analysis(self):
        campaign=self.temporary();groups=[handmade_group(f"g{i:02}") for i in range(40)]
        split={"development":[dict(group_uid=g.uid,domain="A" if i<20 else "B") for i,g in enumerate(groups)]}
        specs=[m.TrialSpec(arm,"replication",s-1,1,1,s) for s in (1,2,3) for arm in m.ARMS]
        rng=np.random.Generator(np.random.PCG64(117));base=rng.normal(size=(40,378)).astype(np.float32)
        base[0]=-10  # Undefined precision/MCC must remain N/A in the saved aggregates.
        for spec in specs:
            entry=run.Entry(spec);root=campaign/spec.run_id;root.mkdir()
            values=base+spec.seed*(0.05 if spec.arm=="S_pos" else -.05 if spec.arm=="H_pos" else 0)*np.asarray([g.labels for g in groups])
            values=values.astype(np.float32)
            scores={point:{role:values if role=="raw" else values.astype(np.float64) for role in run.parent.ROLES} for point in entry.POINTS}
            col=run.old.collect(root/"evaluation",scores,groups,split,entry=entry)
            col["status"]="ALL_6_TRIAL_METRIC_COUNT_SETS_SAVED";data.write_json(root/"evaluation/collected.json",col)
            run.statistics(root/"evaluation",col,spec.arm)
            # Handwritten saved diagnostic curves with distinct seed populations and values.
            manifest={"diagnostics":{}};points={}
            for phase in "AB":
                point=phase+"_"+spec.arm
                capture=dict(fit_ids=[f"fit_{spec.seed}"],fit_domains=[phase],cache_ids=[] if phase=="A" else [f"cache_{spec.seed}"],
                    cache_domains=[] if phase=="A" else ["A"],valid_ids=[g.uid for g in groups])
                target=root/"run/diagnostics"/(point+".json");target.parent.mkdir(parents=True,exist_ok=True);data.write_json(target,capture)
                manifest["diagnostics"][point]=[data.record(target,root/"run/diagnostics")]
                row=dict(status="DESCRIPTIVE_CONDITIONAL_DEVELOPMENT_DIAGNOSIS",fit_or_cache_groups=1,valid_groups=20,
                    train_raw_curve=[[spec.seed+epoch]*3 for epoch in range(6)],valid_raw_curve=[[spec.seed-epoch]*3 for epoch in range(6)],
                    benefit_gap_curve=[[2*epoch]*3 for epoch in range(6)],interpretation={"map":"未见明确迹象"})
                for key in ("epoch1_to_6_train_benefit","epoch1_to_6_valid_benefit","epoch1_to_6_gap_widening"):
                    row[key]=dict(mean=[spec.seed]*3,conditional_95pct_interval=[[spec.seed-1,spec.seed+1]]*3)
                points[point]=dict(fit={phase:copy.deepcopy(row)},cache={"A":copy.deepcopy(row)} if phase=="B" else {"status":"NOT_APPLICABLE_FIRST_STAGE"})
            data.write_json(root/"run/manifest.json",manifest)
            data.write_json(root/"run/partition.json",split)
            (root/"diagnostics_evaluation").mkdir()
            data.write_json(root/"diagnostics_evaluation/overfitting.json",dict(points=points))
        (campaign/"joint").mkdir();joint=run.replication_statistics(campaign,specs)
        self.assertTrue(joint["development_direction_consistency"]["S_pos"])
        self.assertFalse(joint["development_direction_consistency"]["H_pos"])
        self.assertIn("H_pos-S_pos",joint["comparisons"])
        self.assertEqual(joint["diagnostics"]["H_pos"]["three_fixed_seed_point_means"]["B"]["cache"]["A"]["train_raw_curve"][0],[2.]*3)
        self.assertNotEqual(joint["diagnostics"]["S"]["by_seed"]["1"]["populations"]["B"]["cache_ids"],
            joint["diagnostics"]["S"]["by_seed"]["2"]["populations"]["B"]["cache_ids"])
        with self.assertRaises(ValueError):run.replication_statistics(campaign,specs[:-1])
        for arm in ("S_pos","H_pos"):
            for d in "AB":
                rows=range(20) if d=="A" else range(20,40)
                values=[]
                for s in (1,2,3):
                    def read(candidate):
                        spec=next(x for x in specs if x.seed==s and x.arm==candidate)
                        return np.asarray(data.read_json(campaign/spec.run_id/"evaluation"/("B_"+candidate+"_raw_metrics.json"))["values"],float)[list(rows),12]
                    values.append(read(arm)-read("S"))
                self.assertAlmostEqual(joint["comparisons"][arm+"-S"]["three_seed_mean"]["B"]["raw"][d]["map"]["mean"],float(np.mean(values)),places=14)
                self.assertNotEqual(float(np.mean(values)),float(np.mean(values[0])))
        self.assertIsNone(joint["three_seed_absolute"]["S"]["B"]["raw"]["A"]["precision"]["mean"])
        # Run the actual independent analyzer on a deliberately complete handwritten 36-slot ledger.
        search={a:[] for a in m.ARMS}
        import shutil
        for arm in m.ARMS:
            origin=next(s for s in specs if s.arm==arm and s.seed==1)
            domains=data.read_json(campaign/origin.run_id/"evaluation/evaluation.json")["domains"]
            for number in range(12):
                spec=m.TrialSpec(arm,"search",number)
                shutil.copytree(campaign/origin.run_id/"evaluation",campaign/spec.run_id/"evaluation")
                (campaign/spec.run_id/"run").mkdir()
                data.write_json(campaign/spec.run_id/"run/partition.json",split)
                data.write_json(campaign/spec.run_id/"completion.json",dict(trial=spec.identity(),domains=domains))
                search[arm].append(dict(number=number,state="COMPLETE",spec=spec.identity(),domains=domains,value=ctl.objective(domains,ctl.S0)))
        data.write_json(campaign/"search.json",search)
        data.write_json(campaign/"selected.json",{a:ctl.select_trial(search[a],ctl.S0) for a in m.ARMS})
        data.write_json(campaign/"replications.json",[dict(trial=s.identity()) for s in specs])
        from step28_optuna_verify import independent_analysis
        independent_analysis(campaign)
        audit=data.read_json(campaign/"analysis/audit.json");self.assertEqual(audit["status"],"PASS_SAVED_ARITHMETIC")
        from step28_optuna_verify import independent_feedback
        target=m.TrialSpec("S","search",0);root=campaign/target.run_id
        feedback=copy.deepcopy(search["S"][0]);feedback["domains"]["A"]+=.1
        feedback["value"]=ctl.objective(feedback["domains"],ctl.S0)
        completion=data.read_json(root/"completion.json");completion["domains"]=feedback["domains"];data.write_json(root/"completion.json",completion)
        evaluation=data.read_json(root/"evaluation/evaluation.json");evaluation["domains"]=feedback["domains"];data.write_json(root/"evaluation/evaluation.json",evaluation)
        with self.assertRaisesRegex(ValueError,"B/raw"):independent_feedback(root,target,feedback)
        # Internal feedback/selection agreement cannot conceal a wrong saved source.
        changed=copy.deepcopy(search);changed["S"][0]=feedback
        data.write_json(campaign/"search.json",changed)
        data.write_json(campaign/"selected.json",{a:ctl.select_trial(changed[a],ctl.S0) for a in m.ARMS})
        with self.assertRaisesRegex(ValueError,"B/raw"):independent_analysis(campaign)
        col=data.read_json(root/"evaluation/collected.json")
        col["points"]["B_S"]["raw"]["matrix"]=col["points"]["A_S"]["raw"]["matrix"]
        data.write_json(root/"evaluation/collected.json",col)
        with self.assertRaisesRegex(ValueError,"A/cal"):independent_feedback(root,target,feedback)
        self.evidence("saved_evaluation",dict(actual_handwritten_collections=9,synthetic_analysis_slots=45,audit=audit,
            synthetic_copies_are_not_training=True,actual_optimizer_calls=0))

    def test_cpu_six_call_fixture_orchestration(self):
        from contextlib import ExitStack
        from types import SimpleNamespace
        from test_step28_record_replay import TinyEncoder
        from step28_optuna_verify import native_arm
        class FixtureEncoder(TinyEncoder):
            def __getitem__(self,index):
                auto=SimpleNamespace(is_gradient_checkpointing=True,encoder=SimpleNamespace(layer=[SimpleNamespace(
                    attention=SimpleNamespace(self=SimpleNamespace(query=self.embedding)))]))
                return SimpleNamespace(auto_model=auto)
            def tokenizer(self,texts,**kwargs):
                if kwargs.get("padding") is False:return {"input_ids":[[0]*256 for _ in texts]}
                return super().tokenizer(texts,**kwargs)
        archive_times=[]
        def archive(c):
            import time
            time.sleep(.02);archive_times.append(time.monotonic())
            return {"handwritten_no_archive":True}
        def fresh(c):
            self.assertTrue(archive_times)
            torch.manual_seed(c["initialization_seed"])
            model=core.build_model(FixtureEncoder(),8,5);optimizer=core.make_optimizer(model,c)
            return model,optimizer,dict(model=core.state_digest(model.state_dict()),adam=core.state_digest(optimizer.state_dict()))
        with ExitStack() as stack:
            for target,kwargs in (
                ("step28_positive_ranking_verify.native_preflight",dict(return_value={"CPU_fixture_only":True})),
                ("torch.cuda.synchronize",dict(return_value=None)),("torch.cuda.set_per_process_memory_fraction",dict(return_value=None)),
                ("torch.cuda.get_device_properties",dict(return_value=SimpleNamespace(total_memory=48*2**30))),
                ("torch.cuda.is_bf16_supported",dict(return_value=True)),
                ("step28_optuna_run.Entry.pretrained_archive",dict(side_effect=archive)),
                ("step28_optuna_run.Entry.fresh",dict(side_effect=fresh)),
                ("step28_optuna_run.Entry.validate_initial",dict(return_value=True)),
                ("step28_optuna_run.Entry.baseline_inputs",dict(return_value=({},None,None,None))),
                ("step28_optuna_run.parent.configure_numerics",dict(return_value={}))):stack.enter_context(mock.patch(target,**kwargs))
            def progress(stage,**value):self.evidence("six_call_progress",dict(stage=stage,**value,CPU_fixture_only=True))
            value=native_arm(self.temporary(),"H_pos",lambda reserve=0:None,progress,False)
        self.assertEqual(value["actual_updates"],6)
        self.assertGreaterEqual(value["measurements"]["L"],.02)
        self.evidence("six_call_fixture",dict(actual_CPU_optimizer_calls=6,native_BGE_GPU_evidence=False,
            maximum_shape_tokenizer_check_mocked=True,initial_and_resource_checks_mocked=True,
            actual_next_update_equal=value["next_update_bitwise_equal"],effects=value["effects"],updates=value["updates"]))

    def test_repair_resources_and_anchor(self):
        from types import SimpleNamespace
        from step28_optuna_verify import native_arm
        out=self.temporary()
        with mock.patch("step28_positive_ranking_verify.native_preflight",return_value={}), \
                mock.patch("step28_g_half_verify.shutil.disk_usage",return_value=SimpleNamespace(free=1)), \
                mock.patch.object(run.Entry,"fresh") as load:
            with self.assertRaisesRegex(RuntimeError,"disk"):native_arm(out,"S",lambda *a:None,lambda *a,**k:None,False)
            load.assert_not_called()
        from step28_g_half_verify import disk_reserve
        with mock.patch("step28_g_half_verify.shutil.disk_usage",return_value=SimpleNamespace(free=1)):
            with self.assertRaises(RuntimeError):disk_reserve(out,0,2,10)
        for phase in ("evaluation","analysis"):
            root=self.temporary()
            with mock.patch.object(run,"EVIDENCE_LIMIT",100000):
                with self.assertRaisesRegex(RuntimeError,"evidence"):
                    with run.shared_resources(root,phase):
                        # Final check must not reuse the preceding <10-second cached scan.
                        (root/"large.txt").write_bytes(b"x"*100001)
            with self.assertRaisesRegex(RuntimeError,"budget"):
                with run.shared_resources(root,phase) as budget:
                    budget.limits["maximum_rss_bytes"]=1
                    budget.check()
        entry=run.Entry(m.TrialSpec("S_pos","search",0))
        with mock.patch.object(data,"verify",side_effect=ValueError("unbound old manifest")) as verify, \
                mock.patch.object(run.previous,"array") as array:
            with self.assertRaisesRegex(ValueError,"unbound"):run.anchor_compatibility(entry,{})
            self.assertEqual(verify.call_args.args[1]["sha256"],"ef6416de2070376c4e45613493a23bd92083293f9b5b8bad8fabf430d5742d35")
            array.assert_not_called()
        self.evidence("repair_resources",dict(native_low_actual_free_rejected_before_load=True,
            E_A_force_final_evidence_and_actual_RSS_rejected=True,old_manifest_checked_before_arrays=True,actual_optimizer_calls=0))

    def test_repair_controller_boundaries(self):
        import contextlib
        import io
        import json
        import signal
        import sys
        import time
        import step28_optuna_search as search
        out=self.temporary()
        worker_pid=out/"worker.pid"
        code="import os,time; p=os.fork(); " + "open("+repr(str(worker_pid))+",'w').write(str(os.getpid())) if p==0 else None; time.sleep(10)"
        with self.assertRaises(TimeoutError):search.launch([sys.executable,"-c",code],out/"worker.log",.3,os.environ.copy())
        pid=int(worker_pid.read_text())
        import psutil
        self.assertTrue(not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE)
        cleanup=json.loads((out/"worker.log").read_text().splitlines()[-1])
        self.assertTrue(cleanup["resources_released"]);self.assertEqual(cleanup["residual_pids"],[])
        # Execute the actual pre-import prefix with an intentionally slow import and scaled test deadline.
        prefix=(data.ROOT/"scripts/step28_optuna_search.py").read_text().split("\ndata=run.data")[0]
        code="""import builtins,signal,time
original_import=builtins.__import__
original_timer=signal.setitimer
def delayed(name,*args,**kwargs):
    if name=='step28_optuna_run':time.sleep(.2)
    return original_import(name,*args,**kwargs)
def timer(kind,seconds,*args):return original_timer(kind,min(seconds,.05),*args)
builtins.__import__=delayed
signal.setitimer=timer
exec(compile(SOURCE,'actual_controller_prefix','exec'),{'__name__':'__main__'})
""".replace("SOURCE",repr(prefix))
        with self.assertRaises(RuntimeError):search.launch([sys.executable,"-c",code],out/"import.log",5,os.environ.copy())
        self.assertIn("CONTROLLER_IMPORT_FAILED_NO_DISPATCH",(out/"import.log").read_text())
        elapsed={}
        for failure in (False,True):
            root=self.temporary();ledger=ctl.Ledger();ledger.control_since=time.monotonic();ledger.continuous_control=True
            original=data.write_json
            def slow_write(path,value):
                if path.name in ("ledger.json","failure.json","completion.json"):time.sleep(.02)
                return original(path,value)
            output=io.StringIO()
            try:
                with contextlib.redirect_stdout(output),mock.patch.object(data,"write_json",side_effect=slow_write):
                    search.finalize(ledger,root,True,None,"handwritten failure" if failure else None,{})
            finally:signal.setitimer(signal.ITIMER_REAL,0)
            receipt=json.loads(output.getvalue().split("FINAL_ACCOUNTING ",1)[1])
            self.assertGreaterEqual(receipt["used"]["control"],.04)
            self.assertEqual(receipt["used"]["search"],0)
            elapsed[str(failure)]=receipt["used"]["control"]
        self.evidence("repair_controller",dict(own_worker_release=cleanup,pre_import_hard_deadline=True,
            successful_and_failed_final_writes_control_seconds=elapsed,actual_optimizer_calls=0))
