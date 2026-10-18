"""Changed-objective evidence only, entirely handwritten; Linux py310."""
from __future__ import annotations
import copy
import hashlib
import itertools
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch

import step28_positive_ranking_run as run
from test_step28_record_replay import handmade_group, tiny_model
from test_step28_g_half import literal_pool as h_pool
from step28_g_half_verify import state, restore, component_effects

method, record, core, data, diag = run.method, run.record, run.core, run.data, run.diag
PAIRS = list(itertools.combinations(range(28), 2))
LOOKUP = {p: k for k, p in enumerate(PAIRS)}
ATOL, RTOL = 2e-6, 2e-5


def independent_rank(values, labels, mutant=None):
    """Separate candidate enumeration and logaddexp formulation."""
    terms = []
    for q in range(28):
        ids = [LOOKUP[tuple(sorted((q, j)))] for j in range(28) if j != q]
        positives = [k for k in ids if labels[k]]
        negatives = [k for k in ids if not labels[k]]
        if mutant == "constant_only":
            term = torch.logsumexp(values[ids], 0)-values[positives].mean()-math.log(len(positives))
        else:
            if mutant == "omit_second": positives = positives[:1]
            pieces = []
            for p in positives:
                ns = negatives if mutant != "other_positive_negative" else [k for k in ids if k != p]
                pieces.append(torch.logaddexp(values.new_zeros(()), torch.logsumexp(values[ns], 0)-values[p]))
            term = torch.stack(pieces).mean()
        terms.append(term)
    return torch.stack(terms).mean()/(27 if mutant == "extra_average" else 1)


def analytic_rank(values, labels):
    """Plain Python scalar log-sum and analytical edge gradient, no production helper."""
    gradient = np.zeros(378, np.float64)
    losses, query_gradients = [], []
    for q in range(28):
        ids = [LOOKUP[tuple(sorted((q,j)))] for j in range(28) if j != q]
        pos, neg = [k for k in ids if labels[k]], [k for k in ids if not labels[k]]
        qgrad = np.zeros(378, np.float64)
        pieces = []
        for p in pos:
            differences = [float(values[n])-float(values[p]) for n in neg]
            peak = max(0., *differences)
            weights = [math.exp(d-peak) for d in differences]
            den = math.exp(-peak)+math.fsum(weights)
            pieces.append(peak+math.log(den))
            for n, w in zip(neg, weights): qgrad[n] += w/den/len(pos)
            qgrad[p] -= math.fsum(weights)/den/len(pos)
        losses.append(math.fsum(pieces)/len(pos))
        query_gradients.append(qgrad)
        gradient += qgrad/28
    return math.fsum(losses)/28, gradient, query_gradients


def independent_pool(table, slots, arm):
    if arm == "H_pos": return h_pool(table, slots)
    lookup = {p: k for k, p in enumerate(itertools.combinations(range(len(slots)), 2))}
    return torch.stack([table[[lookup[tuple(sorted((int(i),int(j))))]
        for i in np.flatnonzero(slots == a) for j in np.flatnonzero(slots == b)]].mean()
        for a, b in PAIRS])


def independent_loss(values, group):
    bce = torch.stack([torch.nn.functional.softplus(v)-float(group.labels[k])*v for k,v in enumerate(values)]).mean()
    hard = []
    for q in range(28):
        ids = [(j, LOOKUP[tuple(sorted((q,j)))]) for j in range(28) if j != q]
        pos = [k for j,k in ids if group.labels[k]]
        neg = sorted([(j,k) for j,k in ids if not group.labels[k]], key=lambda jk:(-float(values[jk[1]].detach()),jk[0]))[:5]
        hard.append(torch.stack([torch.nn.functional.softplus(values[n]-values[p]) for p in pos for j,n in neg]).mean())
    return bce+independent_rank(values, group.labels)+.5*torch.stack(hard).mean()


def independent_objective(table, group, seed, arm, target=None):
    a0, a1 = record.assignment(group), record.assignment(group, seed)
    v0, v1 = [independent_pool(table, slots, arm) for slots in (a0,a1)]
    supervised = (independent_loss(v0, group)+independent_loss(v1, group))/2
    return supervised if target is None else .1*supervised+.5*(v0-independent_pool(torch.as_tensor(target,device=table.device,dtype=table.dtype),a0,arm)).square().mean()


def shared_memory(arm):
    groups = [handmade_group(f"A_fit_{i:02}") for i in range(48)]
    memory = record.Memory("ABC")
    memory.retain(groups, 1, lambda g: (np.sin(np.arange(1540)/7)*2).astype(np.float32))
    memory.maps.update(method.metadata(arm))
    return memory


def verify_native_table(table, group, seed, reference, arm):
    """Actual FP32 table -> both views -> independent rank values/gradients."""
    rows = []
    for name, slots in (("A0",record.assignment(group)),("A1",record.assignment(group,seed))):
        scores = method.account_logits(table,slots,arm).detach().requires_grad_(True)
        rank = method.objectives(scores,torch.as_tensor(group.labels,device=scores.device,dtype=scores.dtype))["rank_pos"]
        gradient, = torch.autograd.grad(rank,scores)
        value, grad, _ = analytic_rank(scores.detach().double().cpu().tolist(),group.labels)
        np.testing.assert_allclose(float(rank.detach()),value,atol=ATOL,rtol=RTOL)
        np.testing.assert_allclose(gradient.detach().cpu().numpy(),grad,atol=ATOL,rtol=RTOL)
        rows.append(dict(view=name,value=float(rank.detach()),reference=value,gradient_max_error=float(np.max(np.abs(gradient.detach().cpu().numpy()-grad)))))
    return dict(arm=arm,uid=group.uid,role="history" if reference is not None else "current",views=rows)


class PositiveRankingTests(unittest.TestCase):
    def temporary(self):
        t = tempfile.TemporaryDirectory(dir=os.environ["G_HALF_TEST_ROOT"])
        self.addCleanup(t.cleanup)
        return Path(t.name)

    def evidence(self,name,value):
        data.write_json(Path(os.environ["G_HALF_TEST_ROOT"])/(name+".json"),value)

    def test_rank_reference_and_rejected_mutants(self):
        g = handmade_group("rank")
        labels = np.asarray(g.labels)
        rows = []
        for dtype in (torch.float64, torch.float32):
            for tag, values in (("unequal",np.sin(np.arange(378)*.7)*4), ("equal",np.zeros(378)),
                                ("extreme",np.where(np.arange(378)%3,1000.,-1000.))):
                x = torch.tensor(values,dtype=dtype,requires_grad=True)
                actual = method.objectives(x,torch.tensor(labels,dtype=dtype))["rank_pos"]
                gradient, = torch.autograd.grad(actual,x)
                ref, grad, qgrad = analytic_rank(x.detach().tolist(),labels)
                atol,rtol = (1e-12,1e-12) if dtype == torch.float64 else (ATOL,RTOL)
                np.testing.assert_allclose(float(actual.detach()),ref,atol=atol,rtol=rtol)
                np.testing.assert_allclose(gradient.numpy(),grad,atol=atol,rtol=rtol)
                self.assertTrue(np.all(grad[labels.astype(bool)] <= 0))
                self.assertTrue(np.all(grad[~labels.astype(bool)] >= 0))
                for k,(a,b) in enumerate(PAIRS): self.assertAlmostEqual(grad[k],(qgrad[a][k]+qgrad[b][k])/28,places=12)
                if tag == "unequal" and dtype == torch.float64:
                    for k in (0,1,53,97,377):
                        xp,xm=values.copy(),values.copy();xp[k]+=1e-5;xm[k]-=1e-5
                        fd=(analytic_rank(xp,labels)[0]-analytic_rank(xm,labels)[0])/2e-5
                        self.assertLessEqual(abs(fd-grad[k]),2e-8)
                rows.append(dict(case=tag,dtype=str(dtype),value=float(actual.detach()),reference=ref,
                                 gradient_max_error=float(np.max(np.abs(gradient.numpy()-grad)))))
        x=torch.tensor(np.sin(np.arange(378)*.7)*4,dtype=torch.float64,requires_grad=True)
        correct=independent_rank(x,labels);cg,=torch.autograd.grad(correct,x,retain_graph=True)
        rejected={}
        for name in ("constant_only","omit_second","other_positive_negative","extra_average"):
            bad=independent_rank(x,labels,name);bg,=torch.autograd.grad(bad,x,retain_graph=True)
            delta=float((bg-cg).abs().max());self.assertGreater(delta,1e-5);rejected[name]=delta
        # Real-number single-positive identity and double-positive equal-score strengthening.
        neg=torch.arange(26,dtype=torch.float64)/9-2;p=torch.tensor(1.3,dtype=torch.float64)
        torch.testing.assert_close(torch.logsumexp(torch.cat((p[None],neg)),0)-p,
                                  torch.logaddexp(p.new_zeros(()),torch.logsumexp(neg,0)-p),atol=1e-12,rtol=1e-12)
        zero=torch.zeros(378,dtype=torch.float64,requires_grad=True)
        old=record.parent.ranking.objectives(zero,torch.tensor(labels,dtype=zero.dtype),.5)["rank"]
        new=method.objectives(zero,torch.tensor(labels,dtype=zero.dtype))["rank_pos"]
        og,=torch.autograd.grad(old,zero,retain_graph=True);ng,=torch.autograd.grad(new,zero)
        # q0 and q3 are both K3; this negative edge receives two strengthened contributions.
        k=LOOKUP[(0,3)];self.assertGreater(float(ng[k]),float(og[k]))
        self.evidence("rank_reference",dict(cases=rows,rejected_mutants=rejected,undirected_two_query_gradient=True))

    def test_views_history_and_actual_vjp_adamw(self):
        c={**run.config(),"handwritten_qualification":True}
        current,history=handmade_group("current"),handmade_group("history")
        target=(np.sin(np.arange(1540)/7)*3).astype(np.float32)
        evidence={}
        for arm in method.ARMS:
            proxy=(torch.sin(torch.arange(1540)*.39)*3).requires_grad_()
            for ref in (None,target):
                actual,log=method.objective(proxy,current,29,arm,ref)
                independent=independent_objective(proxy,current,29,arm,ref)
                ga,=torch.autograd.grad(actual,proxy,retain_graph=True);gi,=torch.autograd.grad(independent,proxy,retain_graph=True)
                torch.testing.assert_close(actual,independent,atol=ATOL,rtol=RTOL)
                torch.testing.assert_close(ga,gi,atol=ATOL,rtol=RTOL)
            model=tiny_model();method.bind(model,method.metadata(arm));opt=core.make_optimizer(model,c)
            # One real warm-up gives common nonempty Adam moments, not an invented state.
            method.update(model,opt,current,None,None,c,arm,1,1)
            direct=copy.deepcopy(model);od=core.make_optimizer(direct,c);od.load_state_dict(copy.deepcopy(opt.state_dict()))
            old=copy.deepcopy(model);oo=core.make_optimizer(old,c);oo.load_state_dict(copy.deepcopy(opt.state_dict()))
            observed={};tables=[]
            def observe(role,log):observed[role]={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            row=method.update(model,opt,current,history,target,c,arm,2,49,observe=observe,
                table_observer=lambda t,g,s,r:tables.append(verify_native_table(t,g,s,r,arm)))
            run.loss_total(row,2,arm)
            direct.train();od.zero_grad(set_to_none=True);losses=[];pre_errors={}
            for role,g,t in (("current",current,None),("history",history,target)):
                torch.manual_seed(data.seed_for(20260918,"ABC",2,49,role,"dropout"))
                z=record.record_vectors(direct,g,c);i,j=torch.triu_indices(len(z),len(z),1)
                table=direct.head(torch.cat(((z[i]-z[j]).abs(),z[i]*z[j]),1)).flatten()
                seed=data.seed_for(20260918,"ABC",2,49,role,g.uid,"regroup")
                loss=independent_objective(table,g,seed,arm,t);losses.append(float(loss.detach()));loss.backward()
                pre_errors[role]={}
                for n,p in direct.named_parameters():
                    torch.testing.assert_close(observed[role][n],p.grad,atol=ATOL,rtol=RTOL)
                    pre_errors[role][n]=float((observed[role][n]-p.grad).abs().max())
            torch.nn.utils.clip_grad_norm_(direct.parameters(),1.)
            od.param_groups[0]["lr"],od.param_groups[1]["lr"]=run.parent.stage_lr(49),.001;od.step()
            self.assertAlmostEqual(row["total"],sum(losses),places=5)
            for (n,a),b in zip(model.named_parameters(),direct.parameters()):
                torch.testing.assert_close(a,b,atol=ATOL,rtol=RTOL);torch.testing.assert_close(a.grad,b.grad,atol=ATOL,rtol=RTOL)
                for key in ("step","exp_avg","exp_avg_sq"):
                    torch.testing.assert_close(opt.state[a][key],od.state[b][key],atol=ATOL,rtol=RTOL)
            old_observed={}
            def old_observe(role,log):old_observed[role]={n:p.grad.clone() for n,p in old.named_parameters() if p.grad is not None}
            method.update(old,oo,current,history,target,c,arm,2,49,observe=old_observe,qualification_old_rank=True)
            effects=component_effects(model,state(old,oo),optimizer=opt)
            branch_delta={}
            for role in ("current","history"):
                # History callback contains accumulated current+history; subtract current to isolate its own effect.
                branch_delta[role]={}
                for component in ("encoder","head"):
                    deltas=[]
                    for n in observed[role]:
                        if n.startswith(component+"."):
                            a=observed[role][n]-(observed["current"][n] if role=="history" else 0)
                            b=old_observed[role][n]-(old_observed["current"][n] if role=="history" else 0)
                            deltas.append(float((a-b).abs().max()))
                    branch_delta[role][component]=max(deltas);self.assertGreater(max(deltas),0)
            # Omit new history rank but retain new current: independent reference detects this fault.
            new_hist,_=method.objective(proxy,history,29,arm,target)
            old_hist,_=method.objective(proxy,history,29,arm,target,old_rank=True)
            nh,=torch.autograd.grad(new_hist,proxy,retain_graph=True);oh,=torch.autograd.grad(old_hist,proxy)
            self.assertGreater(float((nh-oh).abs().max()),1e-5)
            evidence[arm]=dict(total=row["total"],independent_total=sum(losses),preclip_errors=pre_errors,
                old_new_effects=effects,isolated_branch_preclip_deltas=branch_delta,tables=tables,
                actual_optimizer_calls=3,history_only_rank_fault_rejected=True)
        self.evidence("actual_vjp_adamw",evidence)

    def test_capture_restore_and_next_update(self):
        c=run.config();out=self.temporary();evidence={}
        current,history=handmade_group("current"),handmade_group("history")
        target=(3*np.sin(np.arange(1540)/7)).astype(np.float32)
        for arm in method.ARMS:
            model=tiny_model();method.bind(model,method.metadata(arm));optimizer=core.make_optimizer(model,c)
            method.update(model,optimizer,current,None,None,c,arm,1,1)
            for v in optimizer.state.values():v["step"].fill_(336)
            saved,rng,modes=state(model,optimizer),diag.rng_state(),[m.training for m in model.modules()]
            first=method.update(model,optimizer,current,history,target,c,arm,2,49)
            expected,expected_rng=state(model,optimizer),diag.rng_state()
            restore(model,optimizer,saved,rng,modes)
            memory=shared_memory(arm);memory.begin_stage(2)
            for _ in range(48):memory.draw()
            payload=memory.to_bytes();fit=[handmade_group(f"fit{i}") for i in range(48)]
            valid=[data.Group(g.uid,g.sellers,g.items) for g in [handmade_group(f"valid{i}") for i in range(40)]]
            capture=method.capture(model,optimizer,fit,memory,valid,c,arm,2,1,out,arm)
            result=data.read_json(out/capture["path"]);run.validate_diagnostic(result,arm)
            with diag.neutral(model):
                raw=method.score(model,valid,c,arm)
                np.testing.assert_array_equal(run.previous.array(out,result["valid_blind"],(40,378),np.float32),raw)
                for role,gs in (("fit",fit),("cache",memory.reservoir.groups)):
                    np.testing.assert_array_equal(run.previous.array(out,result[role],(len(gs),3),np.float64),diag.metrics(gs,method.score(model,gs,c,arm)))
                g=memory.reservoir.groups[0];live=record.reference(model,g,c);old=memory.tables[g.uid]
                for key,seed in (("D0",None),("D1",data.seed_for(20261008,"ABC",2,g.uid,"diagnostic_regroup"))):
                    slots=record.assignment(g,seed)
                    error=(independent_pool(torch.from_numpy(live),slots,arm)-independent_pool(torch.from_numpy(old),slots,arm)).detach().numpy().astype(np.float64)
                    self.assertAlmostEqual(result["teacher_errors"][0][key],float(np.mean(error**2)),places=6)
            self.assertEqual(core.state_digest(saved),core.state_digest(state(model,optimizer)))
            self.assertEqual((rng,payload,modes),(diag.rng_state(),memory.to_bytes(),[m.training for m in model.modules()]))
            repeated=method.update(model,optimizer,current,history,target,c,arm,2,49)
            self.assertEqual(first,repeated);self.assertEqual(core.state_digest(expected),core.state_digest(state(model,optimizer)))
            self.assertEqual(expected_rng,diag.rng_state())
            meta={**method.metadata(arm),"rng":diag.rng_state(),"modes":[m.training for m in model.modules()]}
            with diag.neutral(model):before=method.restored_score(model,[current],c)
            for kind,opt in (("full",optimizer),("inference",None)):
                p=out/(arm+kind+".pt");receipt=core.save_state(p,model,opt,meta)
                model.training_identity={"arm":"incorrect"}
                run.restore_point(p,receipt,model,opt,arm)
                with diag.neutral(model):np.testing.assert_array_equal(before,method.restored_score(model,[current],c))
            wrong="H_pos" if arm=="S_pos" else "S_pos"
            with self.assertRaises(ValueError):method.bind(model,{**meta,"pooling":method.identity(wrong)})
            with self.assertRaises(ValueError):run.memory_from(payload,wrong)
            evidence[arm]=dict(actual_optimizer_calls=3,synthetic_clock=336,next_update_bitwise_equal=True,
                actual_capture_populations=[48,6,40],teacher_both_views=True,full_inference_restores=True,own_identity_rejections=True)
        self.evidence("capture_restore",evidence)

    def test_stage_a_empty_cache(self):
        c=run.config();out=self.temporary();evidence={}
        fit=[handmade_group(f"A_fit{i}") for i in range(48)]
        valid=[data.Group(g.uid,g.sellers,g.items) for g in [handmade_group(f"blind{i}") for i in range(40)]]
        for arm in method.ARMS:
            model=tiny_model();method.bind(model,method.metadata(arm));optimizer=core.make_optimizer(model,c)
            method.update(model,optimizer,fit[0],None,None,c,arm,1,1)
            for value in optimizer.state.values():value["step"].fill_(48)
            saved,rng,modes=state(model,optimizer),diag.rng_state(),[m.training for m in model.modules()]
            empty=method.score(model,[],c,arm)
            self.assertEqual((empty.shape,empty.dtype),((0,378),np.dtype(np.float32)))
            with self.assertRaisesRegex(ValueError,"Scoring candidate differs"):
                method.score(model,[],c,"H_pos" if arm=="S_pos" else "S_pos")
            capture=method.capture(model,optimizer,fit,None,valid,c,arm,1,1,out,"A_"+arm)
            row=data.read_json(out/capture["path"]);run.validate_diagnostic(row,arm)
            self.assertEqual((row["stage"],row["stage_step"],row["adam_step"]),(1,48,48))
            self.assertEqual((row["cache_ids"],row["cache_domains"],row["teacher_errors"]),([],[],[]))
            run.previous.array(out,row["cache"],(0,3),np.float64)
            run.previous.array(out,row["fit"],(48,3),np.float64)
            run.previous.array(out,row["valid_blind"],(40,378),np.float32)
            self.assertEqual(core.state_digest(saved),core.state_digest(state(model,optimizer)))
            self.assertEqual((rng,modes),(diag.rng_state(),[m.training for m in model.modules()]))
            # Reproduce the removed S dispatch failure without altering the frozen scorer.
            with diag.neutral(model),self.assertRaisesRegex(ValueError,"need at least one array"):
                record.score(model,[],c)
            evidence[arm]=dict(actual_optimizer_calls=1,synthetic_clock=48,
                capture_populations=[48,0,40],empty_score_shape=[0,378],cache_metric_shape=[0,3],
                state_unchanged=True,wrong_arm_rejected=True,old_empty_dispatch_failure_reproduced=True)
        self.evidence("stage_a_empty_cache",evidence)

    def test_native_continuation_preflight(self):
        import step28_positive_ranking_verify as verify
        from types import SimpleNamespace
        process=mock.Mock(pid=413);process.cpu_affinity.return_value=[0];process.parents.return_value=[]
        sample={"apps":"", "free":31*1024, "host":80*2**30, "jobs":[], "allocated":0, "reserved":0}
        def query(args,**kwargs):
            return (sample["apps"] if "--query-compute-apps=pid,process_name,used_memory" in args
                    else f"0, handwritten-GPU, 32768, {sample['free']}, 1024")
        env={"CUDA_VISIBLE_DEVICES":"0","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
             "OPENBLAS_NUM_THREADS":"1","TOKENIZERS_PARALLELISM":"false"}
        with mock.patch.dict(os.environ,env),mock.patch.object(verify.psutil,"Process",return_value=process), \
             mock.patch.object(verify.psutil,"process_iter",side_effect=lambda *a:sample["jobs"]), \
             mock.patch.object(verify.psutil,"virtual_memory",side_effect=lambda:SimpleNamespace(available=sample["host"])), \
             mock.patch.object(verify.subprocess,"check_output",side_effect=query), \
             mock.patch.object(verify.gc,"collect") as collect, \
             mock.patch.object(verify.torch.cuda,"synchronize"),mock.patch.object(verify.torch.cuda,"empty_cache") as empty, \
             mock.patch.object(verify.torch.cuda,"memory_allocated",side_effect=lambda:sample["allocated"]), \
             mock.patch.object(verify.torch.cuda,"memory_reserved",side_effect=lambda:sample["reserved"]):
            first=verify.native_preflight(False)
            self.assertEqual(first["compute_apps"],"");empty.assert_not_called()
            sample["apps"]="413, python, 512"
            with self.assertRaisesRegex(ValueError,"Resource gate failed"):
                verify.native_preflight(False)
            second=verify.native_preflight(True)
            self.assertEqual(second["allowed_context_pid"],413);collect.assert_called_once();empty.assert_called_once()
            rejected=[]
            for key,value in (("apps","413, python, 512\n414, python, 1"),("free",30*1024-1),
                    ("host",72*2**30-1),("allocated",4),("reserved",4),
                    ("jobs",[SimpleNamespace(pid=415,info={"pid":415,"username":"same-user","cmdline":["step28_other.py"]})])):
                previous=sample[key];sample[key]=value
                with self.assertRaises(ValueError):verify.native_preflight(True)
                sample[key]=previous;rejected.append(key)
            # Reach the real native entry's resource call, stopping before model loading/GPU work.
            class StopBeforeGPU(Exception):pass
            with mock.patch.object(verify,"native_preflight",side_effect=StopBeforeGPU) as entry:
                for continuing,arm in ((False,"S_pos"),(True,"H_pos")):
                    with self.assertRaises(StopBeforeGPU):
                        verify.native(lambda:None,lambda *a,**k:None,None,None,arm,continuing=continuing)
                self.assertEqual(entry.call_args_list,[mock.call(False),mock.call(True)])
        self.evidence("native_continuation_preflight",dict(first_strict=True,own_context_only_second=True,
            rejected_cases=rejected,resource_thresholds_unchanged=True,real_native_entry_checked=True,
            cuda_and_process_state="mocked handwritten fixtures",actual_gpu_calls=0))

    def test_dual_orchestration_and_joint_opening(self):
        root=self.temporary();groups={};metadata=[]
        for split,count in (("train",60),("development",20)):
            groups[split]=[]
            for d in "AB":
                for i in range(count):
                    g=handmade_group(f"{d}_{split}_{i:02}")
                    metadata.append(dict(group_uid=g.uid,domain=d,split=split,group_index=i))
                    groups[split].append(g if split=="train" else data.Group(g.uid,g.sellers,g.items))
        loads,calls=[],[];c=run.config()
        _,split_expected=run.partition(groups,metadata,c)
        oldroot=root/"frozen_fixture"; (oldroot/"run/updates").mkdir(parents=True)
        data.write_json(oldroot/"run/partition.json",split_expected)
        tiny=tiny_model(); tinyopt=core.make_optimizer(tiny,c)
        fake_baseline={"initial_identity":{"model":core.state_digest(tiny.state_dict()),"adam":core.state_digest(tinyopt.state_dict())}}
        del tiny,tinyopt
        class Budget:
            phase="handwritten orchestration"
            def check(self,*args):pass
            def snapshot(self):return {}
        def load(cfg):
            loads.append("fresh");return tiny_model()
        def update(model,optimizer,current,history,target,cfg,arm,stage,step,check):
            self.assertTrue(current.uid.startswith("AB"[stage-1]+"_"))
            self.assertEqual(run.parent.adam_step(optimizer),(stage-1)*288+step-1)
            calls.append((arm,stage,step,current.uid,None if history is None else history.uid))
            if history:self.assertTrue(history.uid.startswith("A_"))
            t=(stage-1)*288+step
            for p in model.parameters():
                optimizer.state[p]={"step":torch.tensor(float(t)),"exp_avg":torch.zeros_like(p),"exp_avg_sq":torch.zeros_like(p)}
            with torch.no_grad():model.head[2].bias.fill_(t*.001+method.ARMS.index(arm)*.3)
            row=dict(total=1.,encoder_lr=run.parent.stage_lr(step),head_lr=.001,**method.metadata(arm))
            for role,g in (("current",current),("history",history)):
                if g is not None:
                    seed=data.seed_for(20260918,"ABC",stage,step,role,g.uid,"regroup")
                    row[role]=dict(supervised=1.,weighted=1. if role=="current" else .1,mse0=0.,mse1=None,distillation=0.,
                        supervised_contribution=1. if role=="current" else .1,teacher_contribution=0.,qualification_old_rank=False,
                        views=[dict(bce=.2,rank_pos=.3,rank_used=.3,hard=1.,total=1.) for _ in range(2)],
                        assignment_sha256=hashlib.sha256(record.assignment(g,seed).tobytes()).hexdigest())
            if history:row["total"]=1.1
            return row
        truth=np.asarray(handmade_group("geometry").labels);tables={}
        def reference(model,g,cfg,check=lambda:None):
            if g.uid not in tables:
                slots=record.assignment(g)
                lookup={p:k for k,p in enumerate(itertools.combinations(range(28),2))}
                values=[]
                for i,j in itertools.combinations(range(len(slots)),2):
                    a,b=sorted((int(slots[i]),int(slots[j])))
                    k=0 if a==b else lookup[a,b]
                    values.append((0 if a==b else truth[k]*3)+math.sin(i*.7+j*.3))
                tables[g.uid]=np.asarray(values,np.float32)
            return tables[g.uid]+float(model.head[2].bias.detach())
        def score(model,gs,cfg,arm,check=lambda:None):
            if not gs:return production_score(model,gs,cfg,arm,check)
            return np.stack([method.account_logits(torch.from_numpy(reference(model,g,cfg)),record.assignment(g),arm).numpy() for g in gs]).astype(np.float32)
        production_score = method.score
        with mock.patch.object(run,"baseline_inputs",return_value=(fake_baseline,oldroot,{},{"public_inputs":{"handwritten":True}})), \
             mock.patch.object(run,"baseline_runs",side_effect=lambda:{a:(oldroot,{}, {}) for a in ("S","G_half")}), \
             mock.patch.object(run,"pretrained_archive",return_value={"handwritten":True}), \
             mock.patch.object(run,"public_inputs",return_value=(groups,metadata,{"handwritten":True})), \
             mock.patch.object(run,"attach_labels",side_effect=lambda gs,cfg,s,meta:gs), \
             mock.patch.object(core,"model_files",return_value=c["models"]["split_rank"]), \
             mock.patch.object(record,"load_model",side_effect=load), \
             mock.patch.object(method,"update",side_effect=update), \
             mock.patch.object(record,"reference",side_effect=reference), \
             mock.patch.object(method,"score",side_effect=score), \
             mock.patch.object(diag,"metrics",side_effect=lambda gs,vs:np.zeros((len(gs),3),np.float64)), \
             mock.patch.object(run.parent.calibration,"fit",return_value={"status":"PASS_CALIBRATION_FIT","a":1.,"b":0.}), \
             mock.patch("builtins.print"):
            data.write_json(root/"access.json",dict(train=0,valid=0,heldout=0,owners=0))
            manifest,split,valid,meta=run.train(root,Budget())
            self.assertEqual((len(loads),len(calls)),(2,1152))
            self.assertEqual(manifest["physical_updates"],1152)
            self.assertEqual(manifest["gradient_group_presentations"],1728)
            self.assertEqual(sum(map(len,manifest["diagnostics"].values())),24)
            self.assertEqual(len(list((root/"run/memory").glob("*.bin"))),4)
            self.assertFalse(list((root/"run/work").glob("*.pt")))
            for stage in "AB":
                log=data.read_json(root/"run/updates"/(stage+"_S_pos.json"))
                for other in ("S","G_half"):data.write_json(oldroot/"run/updates"/(stage+"_"+other+".json"),log)
            scores,checked=run.blind_gate(root/"run",manifest)
            self.assertEqual((set(scores),checked),(set(run.POINTS),split))
            for arm in method.ARMS:
                self.assertEqual(manifest["starts"][arm],manifest["points"]["A_"+arm]["continuity"])
            for collection,key in (("points","A_S_pos"),("diagnostics","B_H_pos"),("training","A_S_pos"),("starts","H_pos")):
                damaged=copy.deepcopy(manifest);del damaged[collection][key]
                with self.assertRaises(ValueError):run.open_valid(root,damaged,valid,meta,c)
                self.assertEqual(data.read_json(root/"access.json")["valid"],0)
            damaged=copy.deepcopy(manifest);damaged["starts"]["H_pos"]["model"]="wrong_A"
            with self.assertRaises(ValueError):run.open_valid(root,damaged,valid,meta,c)
            damaged=copy.deepcopy(manifest);damaged["points"]["B_H_pos"]["pooling"]={"arm":"S"}
            with self.assertRaises(ValueError):run.open_valid(root,damaged,valid,meta,c)
            self.assertEqual(data.read_json(root/"access.json")["valid"],0)
            badlog=data.read_json(oldroot/"run/updates/B_G_half.json")
            badlog["rows"][0]["current"]["assignment_sha256"]="wrong"
            data.write_json(oldroot/"run/updates/B_G_half.json",badlog)
            with self.assertRaises(ValueError):run.open_valid(root,manifest,valid,meta,c)
            self.assertEqual(data.read_json(root/"access.json")["valid"],0)
        self.evidence("orchestration",dict(fixture="Actual scheduler, pooling, Memory and checkpoint/diagnostic IO; mocked update/forward/metrics/calibration only",
            scheduler_calls=len(calls),real_updates_in_this_fixture=0,independent_initializations=len(loads),
            diagnostics=24,endpoints=4,own_A_states=True,blind_rejections=True,
            scope="Control-flow evidence only. Real tiny gradient/update and actual diagnostic tests are separate."))

    def test_saved_comparisons_and_permissions(self):
        labelled=[handmade_group(f"{d}_valid_{i:02}") for d in "AB" for i in range(20)]
        split={"development":[dict(group_uid=g.uid,domain=g.uid[0]) for g in labelled]}
        raw=np.stack([np.asarray(g.labels,np.float32)*2-1+np.sin(np.arange(378)+i).astype(np.float32)*.2 for i,g in enumerate(labelled)])
        scores={p:{"raw":raw.copy(),"stage-cal":raw.astype(np.float64),"first-cal":raw.astype(np.float64)} for p in run.POINTS}
        temp=self.temporary();new=temp/"new";old=temp/"old";old.mkdir()
        col=run.collect(new,scores,labelled,split)
        oldcol=copy.deepcopy(col)
        (old/"evaluation").mkdir()
        import shutil
        for p in new.iterdir(): shutil.copyfile(p,old/"evaluation"/p.name)
        columns=col["columns"]
        def set_values(origin,collection,point,m,loss):
            for role,rec in collection["points"][point].items():
                path=origin/rec["matrix"]["path"];v=data.read_json(path)
                for row in v["values"]:
                    row[columns.index("map")]=m
                    for key in ("brier","log_loss"):row[columns.index(key)]=loss
                data.write_json(path,v);rec["matrix"]=data.record(path,origin)
                qp=origin/rec["queries"]["path"];q=np.load(qp,allow_pickle=False);q[:,:,0]=m;qp.unlink()
                rec["queries"]=run.previous.save_array(qp,q,origin)
        for arm,a,b,la,lb in (("S_pos",.5,.4,.4,.5),("H_pos",.5,.7,.4,.3)):
            set_values(new,col,"A_"+arm,a,la);set_values(new,col,"B_"+arm,b,lb)
        for arm in method.ARMS:
            for stage in "AB":set_values(old/"evaluation",oldcol,stage+"_"+arm,.4,.3)
        oldcol["points"]={stage+"_"+arm:copy.deepcopy(oldcol["points"][stage+"_S_pos"]) for stage in "AB" for arm in ("S","G_half")}
        data.write_json(new/"collected.json",col)
        with mock.patch.object(run,"baseline_runs",return_value={a:(old,oldcol,{}) for a in ("S","G_half")}):
            result=run.finalize(new)
            self.assertFalse(result["success"]["passed"])
            self.assertTrue(result["success"]["secondary_H_pos_minus_S_both_positive"])
            self.assertEqual(set(result["comparisons"]),{"S_pos-S","H_pos-S","H_pos-G_half","H_pos-S_pos"})
            for key in ("S_pos-S","H_pos-S_pos"):
                self.assertEqual(set(result["comparisons"][key]),set(run.parent.ROLES))
            self.assertAlmostEqual(result["forgetting_comparisons"]["S_pos-S"]["raw"]["map"]["mean"],.1)
            for key in ("brier","log_loss"):
                self.assertAlmostEqual(result["forgetting_comparisons"]["S_pos-S"]["raw"][key]["mean"],.1)
            for m in ("1","2"):
                self.assertAlmostEqual(result["query_subgroup_MAP_comparisons"]["H_pos-S_pos"][m]["B"]["mean"],.3)
            oldcol["group_ids"]=oldcol["group_ids"][::-1]
            with self.assertRaises(ValueError):run.finalize(new)
        gate=temp/"gate.json";data.write_json(gate,{"kind":"NATIVE","status":"APPROVED_NO_OPEN_BLOCKERS"})
        with self.assertRaises(ValueError):run.reviewed_gate(gate,"NATIVE")
        m={k:1. for k in ("t_A","t_B","v","h","L","s_F","r_F","s_I","r_I")};m.update(W_F=1000,W_I=500)
        projection=run.projection({arm:m for arm in method.ARMS})
        self.assertEqual(projection["projected_total_seconds"],2*1.25*(576+1482+9+1800))
        self.assertFalse(run.projection({arm:{**m,"t_B":1000} for arm in method.ARMS})["within_budget"])
        self.evidence("comparison_permissions",dict(success=result["success"],all_four_contrasts=True,
            primary_not_replaced_by_secondary=True,wrong_uid_rejected=True,missing_permission_rejected=True,
            subgroup_MAP=result["query_subgroup_MAP_comparisons"],projection=projection,
            scope="Artificial saved-metric arithmetic, no formal labels or saved baseline payloads."))
