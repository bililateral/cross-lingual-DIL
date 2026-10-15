"""Independent handwritten references and affected live paths; Linux only."""
from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import step28_balanced_pooling as method
import step28_balanced_pooling_run as run
from test_step28_record_replay import handmade_group, tiny_model, literal_scores

record, core, data, diag = run.record, run.core, run.data, run.diag


def reference_2xm(k):
    """Eliminate column constraints and solve ONE scalar; no matrix scaling."""
    k = np.asarray(k, np.float64)
    if k.shape == (2,2):
        delta=(k[0,0]+k[1,1]-k[0,1]-k[1,0])/4
        x=.5/(1+math.exp(-2*delta))
        return float(k.mean()+np.logaddexp(delta,-delta)-math.log(2)),np.array([[x,.5-x],[.5-x,x]])
    d = k[0]-k[1]
    lo, hi = float(d.min()), float(d.max())
    for _ in range(200):
        if hi-lo <= 1e-13:
            break
        mid = (lo+hi)/2
        first = 1/(1+np.exp(mid-d))
        if first.mean() > .5: lo = mid
        else: hi = mid
    else:
        raise AssertionError("Independent bisection failed")
    p0 = 1/(1+np.exp((lo+hi)/2-d))/k.shape[1]
    p = np.stack((p0, 1/k.shape[1]-p0))
    nz = p > 0
    value = (p*k).sum()-(p[nz]*np.log(p[nz]*k.size)).sum()
    return float(value), p


def literal_pool(table, slots, arm):
    """Enumerate pairs, and use closed-form 2x2 F on this handwritten group."""
    lookup = {pair: table[k] for k,pair in enumerate(itertools.combinations(range(len(slots)),2))}
    result = []
    for a,b in itertools.combinations(range(28),2):
        terms = [lookup[tuple(sorted((int(i),int(j))))] for i in np.flatnonzero(slots==a)
                 for j in np.flatnonzero(slots==b)]
        v = torch.stack(terms)
        if arm == "S": score = v.mean()
        elif arm == "G": score = (torch.logsumexp(v.double(),0)-math.log(4)).to(table.dtype)
        else:
            delta = (v[0].double()+v[3]-v[1]-v[2])/4
            score = (v.double().mean()+torch.logaddexp(delta,-delta)-math.log(2)).to(table.dtype)
        result.append(score)
    return torch.stack(result)


def literal_loss(values, group):
    pairs = list(itertools.combinations(range(28),2))
    lookup = {p:k for k,p in enumerate(pairs)}
    bce = torch.stack([torch.nn.functional.softplus(values[k])-group.labels[k]*values[k] for k in range(378)]).mean()
    ranks, hard = [], []
    for q in range(28):
        candidates = [j for j in range(28) if j != q]
        scores = {j:values[lookup[tuple(sorted((q,j)))]] for j in candidates}
        pos = [j for j in candidates if group.labels[lookup[tuple(sorted((q,j)))]]]
        neg = [j for j in candidates if j not in pos]
        top = sorted(neg,key=lambda j:(-float(scores[j].detach()),j))[:5]
        ranks.append(torch.logsumexp(torch.stack(list(scores.values())),0)-torch.stack([scores[j] for j in pos]).mean())
        hard.append(torch.stack([torch.nn.functional.softplus(scores[n]-scores[p]) for p in pos for n in top]).mean())
    return bce+torch.stack(ranks).mean()+.5*torch.stack(hard).mean()


def literal_objective(table, group, seed, arm, target=None):
    a0, a1 = record.assignment(group), record.assignment(group,seed)
    v0, v1 = literal_pool(table,a0,arm), literal_pool(table,a1,arm)
    loss = (literal_loss(v0,group)+literal_loss(v1,group))/2
    if target is not None:
        old = literal_pool(torch.as_tensor(target,dtype=table.dtype),a0,arm)
        loss = .1*loss+.5*(v0-old).square().mean()
    return loss


def shared_memory(arm="F"):
    groups = [handmade_group(f"A_fit_{i:02}") for i in range(48)]
    memory = record.Memory("ABC")
    memory.retain(groups,1,lambda g:(np.sin(np.arange(1540)/7)*2).astype(np.float32))
    memory.maps["pooling"] = method.identity(arm)
    return memory


class BalancedPoolingTests(unittest.TestCase):
    def temporary(self):
        t = tempfile.TemporaryDirectory(dir=os.environ["BALANCED_POOLING_TEST_ROOT"])
        self.addCleanup(t.cleanup)
        return Path(t.name)

    def evidence(self, name, value):
        data.write_json(Path(os.environ["BALANCED_POOLING_TEST_ROOT"])/(name+".json"),value)

    def test_independent_values_plans_gradients_and_fail_closed(self):
        cases = [np.array(v,np.float64) for v in (
            [[0,0],[0,0]], [[2,0],[0,2]], [[2,2],[0,0]], [[4,0],[0,0]],
            [[20,0],[0,0]], [[8,1,-4],[2,-2,3]])]
        cases += [np.stack((np.arange(m)*.7-2, np.sin(np.arange(m))*3+.2)) for m in range(2,9)]
        evidence = []
        for k in cases:
            expected, p = reference_2xm(k)
            for dtype in (torch.float64,torch.float32):
                x = torch.tensor(k,dtype=dtype)[None].requires_grad_()
                value = method.block_scores(x,"F")
                grad, = torch.autograd.grad(value.sum(),x)
                # For FP32, the reference uses the actual rounded input.
                ref, refp = reference_2xm(x.detach().double()[0].numpy())
                bound = 1e-8 if dtype==torch.float64 else 1e-8+1e-7*max(1,abs(ref))
                self.assertLessEqual(abs(float(value)-ref),bound)
                self.assertLessEqual(float(np.abs(grad[0].numpy()-refp).sum()),2e-4)
                if k.shape==(2,2): self.assertLessEqual(float(np.abs(grad[0].numpy()-refp).sum()),1e-7)
                xt=x.detach().transpose(1,2).requires_grad_()
                vt=method.block_scores(xt,"F")
                gt,=torch.autograd.grad(vt.sum(),xt)
                self.assertLessEqual(abs(float(vt.detach())-ref),bound)
                self.assertLessEqual(float(np.abs(gt[0].numpy()-refp.T).sum()),2e-4)
            value,plan,info = method.solve(torch.from_numpy(k)[None])
            self.assertLessEqual(abs(float(value)-expected),1e-8)
            fd = np.empty_like(k)
            for i,j in itertools.product(range(2),range(k.shape[1])):
                plus,minus=k.copy(),k.copy();plus[i,j]+=1e-5;minus[i,j]-=1e-5
                fd[i,j]=(reference_2xm(plus)[0]-reference_2xm(minus)[0])/2e-5
            self.assertLessEqual(float(np.abs(plan[0].numpy()-fd).sum()),2e-4)
            evidence.append(dict(input=k.tolist(),reference=expected,actual=float(value),plan=plan[0].tolist(),diagnostics=info))
        # Symmetric 8x8 has an independent closed form.
        k = torch.eye(8,dtype=torch.float64)*3
        ref = math.log((math.exp(3)+7)/8)
        value,p,_=method.solve(k[None])
        expected=torch.full((8,8),1/(8*(math.exp(3)+7)),dtype=torch.float64)
        expected.fill_diagonal_(math.exp(3)/(8*(math.exp(3)+7)))
        self.assertLessEqual(abs(float(value)-ref),1e-8)
        self.assertLessEqual(float((p[0]-expected).abs().sum()),2e-4)
        with self.assertRaises(ValueError): method.solve(torch.tensor([[[8.,1.,-4.],[2.,-2.,3.]]]),maximum_iterations=1)
        with self.assertRaises(ValueError): method.solve(torch.full((1,2,3),float("nan")))
        with self.assertRaises(ValueError): method.solve(torch.zeros(1,1,2))
        cover=method.block_scores(torch.tensor([[[2.,0.],[0.,2.]]]),"F")
        isolated=method.block_scores(torch.tensor([[[4.,0.],[0.,0.]]]),"F")
        torch.testing.assert_close(cover,isolated,atol=1e-7,rtol=0)
        self.evidence("independent_solver",dict(scope="handwritten independent scalar/closed forms",cases=evidence,
            truncated_nonfinite_shape_rejected=True,symmetric_8x8_value=float(value)))

    def test_all_shapes_equivariance_and_additive_identity(self):
        rows=[]
        for n,m in itertools.product(range(2,9),repeat=2):
            i,j=torch.arange(n,dtype=torch.float64)[:,None],torch.arange(m,dtype=torch.float64)[None,:]
            k=torch.sin(i*1.3+j*.9)*2+i*.11-j*.2
            f,p,info=method.solve(k[None])
            ft,pt,_=method.solve(k.T[None])
            fp,pp,_=method.solve(k.flip((0,1))[None])
            shifted=k+i*.3+j*.2-1
            fs,_,_=method.solve(shifted[None])
            self.assertLessEqual(abs(float(f-ft)),1e-8)
            self.assertLessEqual(abs(float(f-fp)),1e-8)
            self.assertLessEqual(float((p[0]-pt[0].T).abs().sum()),2e-4)
            self.assertLessEqual(float((p[0]-pp[0].flip((0,1))).abs().sum()),2e-4)
            self.assertLessEqual(abs(float(fs-f)-float((i*.3+j*.2-1).mean())),1e-8)
            self.assertGreaterEqual(float(f-k.mean()),-1e-8)
            self.assertLessEqual(float(f-method.block_scores(k[None],"G")),1e-8)
            rounded=k.float()
            reference64,_,_=method.solve(rounded.double()[None])
            output32=method.block_scores(rounded[None],"F")
            self.assertLessEqual(abs(float(output32-reference64)),1e-8+1e-7*max(1,abs(float(reference64))))
            rows.append(dict(n=n,m=m,**info))
        self.evidence("all_shapes",rows)

    def test_full_objective_proxy_and_mutation_witnesses(self):
        group=handmade_group("objective")
        table=(torch.sin(torch.arange(1540,dtype=torch.float64)*.39)*3).requires_grad_()
        target=(np.cos(np.arange(1540)*.27)*2).astype(np.float64)
        rows={}
        for arm in method.ARMS:
            actual,_=method.objective(table,group,29,arm,target)
            independent=literal_objective(table,group,29,arm,target)
            ga,=torch.autograd.grad(actual,table,retain_graph=True)
            gi,=torch.autograd.grad(independent,table,retain_graph=True)
            torch.testing.assert_close(actual,independent,atol=1e-8,rtol=1e-8)
            torch.testing.assert_close(ga,gi,atol=2e-6,rtol=2e-5)
            rows[arm]=dict(value=float(actual),gradient=ga.tolist())
        a0=record.assignment(group);v=literal_pool(table,a0,"F")
        wrong_teacher=literal_pool(torch.as_tensor(target),a0,"S")
        right_teacher=literal_pool(torch.as_tensor(target),a0,"F")
        self.assertGreater(float(((v-wrong_teacher).square()-(v-right_teacher).square()).abs().max()),1e-3)
        self.assertNotEqual(rows["S"]["value"],rows["G"]["value"])
        self.assertNotEqual(rows["G"]["value"],rows["F"]["value"])
        no_a1=.1*literal_loss(v,group)+.5*(v-right_teacher).square().mean()
        self.assertGreater(abs(float(no_a1)-rows["F"]["value"]),1e-5)
        k=torch.tensor([[[2.,0.],[0.,2.]]],dtype=torch.float64)
        fv,plan,_=method.solve(k)
        self.assertGreater(abs(float(fv-(plan*k).sum())),.1)
        x=table.detach().clone().requires_grad_()
        with mock.patch.object(method,"account_logits",side_effect=lambda t,s,a,stats=None,context=None:record.account_logits(t,s)):
            mutant,_=method.objective(x,group,29,"F",target)
        self.assertGreater(abs(float(mutant)-rows["F"]["value"]),1e-3)
        self.evidence("objective_witnesses",dict(values={k:v["value"] for k,v in rows.items()},
            incorrect_mean_teacher_and_missing_A1_and_entropy_detectable=True,independent_proxy_gradient=True))

    def test_chunked_real_updates_match_independent_single_graph(self):
        c=run.config(); current,history=handmade_group("current"),handmade_group("history")
        target=(np.sin(np.arange(1540)/7)*3).astype(np.float32)
        effects={}
        for arm in method.ARMS:
            model=tiny_model();direct=copy.deepcopy(model)
            oa,ob=[core.make_optimizer(m,c) for m in (model,direct)]
            row=method.update(model,oa,current,history,target,c,arm,2,49)
            direct.train();ob.zero_grad(set_to_none=True)
            values=[]
            for role,g,t in (("current",current,None),("history",history,target)):
                torch.manual_seed(data.seed_for(20260918,"ABC",2,49,role,"dropout"))
                z=record.record_vectors(direct,g,c)
                i,j=torch.triu_indices(len(z),len(z),1)
                u,v=z[i],z[j]
                table=direct.head(torch.cat(((u-v).abs(),u*v),1)).flatten()
                seed=data.seed_for(20260918,"ABC",2,49,role,g.uid,"regroup")
                loss=literal_objective(table,g,seed,arm,t); values.append(float(loss.detach()));loss.backward()
            torch.nn.utils.clip_grad_norm_(direct.parameters(),1.)
            ob.param_groups[0]["lr"],ob.param_groups[1]["lr"]=run.parent.stage_lr(49),.001
            ob.step()
            self.assertAlmostEqual(row["total"],sum(values),places=5)
            for a,b in zip(model.parameters(),direct.parameters()):
                torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)
            self.assertGreater(float(model.encoder.embedding.weight.grad.norm()),0)
            effects[arm]=row["total"]
        self.evidence("single_graph_vjp",dict(objectives=effects,all_parameters_and_gradients_checked=True,
            scope="actual tiny encoder/head updates, not BGE"))

    def test_actual_scoring_diagnostics_restoration_and_next_update(self):
        from step28_balanced_pooling_verify import state,restore,component_effects
        c=run.config();model=tiny_model();optimizer=core.make_optimizer(model,c)
        current,history=handmade_group("current"),handmade_group("history")
        method.update(model,optimizer,current,None,None,c,"S",1,1)
        for v in optimizer.state.values():v["step"].fill_(336)
        saved,rng,modes=state(model,optimizer),diag.rng_state(),[m.training for m in model.modules()]
        target=(3*np.sin(np.arange(1540)/7)).astype(np.float32)
        outcomes,states,effects={},{},{}
        for arm in method.ARMS:
            restore(model,optimizer,saved,rng,modes)
            row=method.update(model,optimizer,current,history,target,c,arm,2,49)
            outcomes[arm]=row;states[arm]=state(model,optimizer)
            if arm=="G":effects["G-S"]=component_effects(model,states["S"])
            if arm=="F":effects["F-G"]=component_effects(model,states["G"])
        expected_rng=diag.rng_state()
        restore(model,optimizer,saved,rng,modes)
        record.update(model,optimizer,current,history,target,c,"S","ABC",2,49)
        self.assertEqual(core.state_digest(states["S"]),core.state_digest(state(model,optimizer)))
        restore(model,optimizer,saved,rng,modes)
        memory=shared_memory();memory.begin_stage(2)
        for _ in range(48):memory.draw()
        payload=memory.to_bytes()
        fit=[handmade_group(f"fit{i}") for i in range(48)]
        valid=[data.Group(g.uid,g.sellers,g.items) for g in [handmade_group(f"valid{i}") for i in range(40)]]
        out=self.temporary()
        capture=method.capture(model,optimizer,fit,memory,valid,c,"F",2,1,out,"fixture")
        result=data.read_json(out/capture["path"])
        with diag.neutral(model):
            raw=method.score(model,valid,c,"F")
            mean=record.score(model,valid,c)
            np.testing.assert_array_equal(run.previous.array(out,result["valid_blind"],(40,378),np.float32),raw)
            self.assertFalse(np.array_equal(raw,mean))
            g=memory.reservoir.groups[0]; live=record.reference(model,g,c); old=memory.tables[g.uid]
            for key,seed in (("D0",None),("D1",data.seed_for(20261008,"ABC",2,g.uid,"diagnostic_regroup"))):
                slots=record.assignment(g,seed)
                error=(literal_pool(torch.from_numpy(live),slots,"F")-literal_pool(torch.from_numpy(old),slots,"F")).detach().numpy().astype(np.float64)
                self.assertAlmostEqual(result["teacher_errors"][0][key],float(np.mean(error**2)),places=6)
        self.assertEqual(core.state_digest(saved),core.state_digest(state(model,optimizer)))
        self.assertEqual((rng,payload,modes),(diag.rng_state(),memory.to_bytes(),[m.training for m in model.modules()]))
        method.update(model,optimizer,current,history,target,c,"F",2,49)
        self.assertEqual(core.state_digest(states["F"]),core.state_digest(state(model,optimizer)))
        self.assertEqual(expected_rng,diag.rng_state())
        for arm in method.ARMS:
            meta={"pooling":method.identity(arm),"rng":diag.rng_state(),"modes":[m.training for m in model.modules()]}
            method.bind(model,meta)
            birth=record.Memory("ABC")
            with diag.neutral(model):
                birth.retain(fit,1,lambda g:record.reference(model,g,c))
                for g in birth.reservoir.groups:
                    np.testing.assert_array_equal(birth.tables[g.uid],record.reference(model,g,c))
            self.assertEqual(birth.to_bytes(),record.Memory.from_bytes(birth.to_bytes()).to_bytes())
            with diag.neutral(model):before=method.restored_score(model,[current],c)
            for kind,opt in (("full",optimizer),("inference",None)):
                p=out/(arm+kind+".pt");saved_file=core.save_state(p,model,opt,meta)
                method.bind(model,{"pooling":method.identity("S")})
                run.restore_point(p,saved_file,model,opt,arm)
                with diag.neutral(model):np.testing.assert_array_equal(before,method.restored_score(model,[current],c))
            with self.assertRaises(ValueError):method.bind(model,{"pooling":{**meta["pooling"],"tau":2.}})
            with self.assertRaises(ValueError):method.bind(model,{})
        self.evidence("actual_paths",dict(effects=effects,complete_S_unchanged=True,actual_F_capture_and_D0_D1=True,
            next_update_bitwise_equal=True,three_scoring_restores=True,actual_birth_tables=True,
            fixture="synthetic counter 336; actual tiny updates"))

    def test_projection_and_saved_evaluation(self):
        m={f"{k}_{a}":1. for a in method.ARMS for k in ("t_A","t_B","v","h")}
        m.update(L=1.,s_F=1.,r_F=1.,s_I=1.,r_I=1.,W_F=1000,W_I=500)
        p=run.projection(m)
        self.assertEqual(p["train_seconds"],1728)
        self.assertEqual(p["forward_seconds"],4338+108)
        self.assertEqual(p["io_seconds"],27)
        for key in m:
            with self.assertRaises(ValueError):run.projection({**m,key:0})
        self.assertFalse(run.projection({**m,"t_B_F":1000})["within_budget"])
        root=self.temporary()/"evaluation"
        labelled=[handmade_group(f"{d}_valid_{i:02}") for d in "AB" for i in range(20)]
        split={"development":[dict(group_uid=g.uid,domain=g.uid[0]) for g in labelled]}
        raw=np.stack([np.asarray(g.labels,np.float32)*2+np.sin(np.arange(378)+i).astype(np.float32) for i,g in enumerate(labelled)])
        scores={p:{"raw":raw.copy(),"stage-cal":raw.astype(np.float64)*1.2-.3,"first-cal":raw.astype(np.float64)*.8+.1} for p in run.POINTS}
        saved=run.collect(root,scores,labelled,split);result=run.finalize(root)
        self.assertEqual(saved["status"],"ALL_18_METRIC_COUNT_SETS_SAVED")
        for key in ("comparisons","initial_comparisons","forgetting_comparisons"):
            self.assertEqual(set(result[key]),{"F-S","G-S","F-G"})
        for comparison in result["comparisons"].values():
            for role in comparison.values():
                for domain in role.values():
                    for metric in domain.values():self.assertEqual(metric,{"mean":0.,"conditional_95pct_interval":[0.,0.]})

    def test_undefined_saved_metrics_preserve_fixed_groups(self):
        labelled=[handmade_group(f"{d}_valid_{i:02}") for d in "AB" for i in range(20)]
        split={"development":[dict(group_uid=g.uid,domain=g.uid[0]) for g in labelled]}
        perfect=np.stack([np.asarray(g.labels,np.float64)*2-1 for g in labelled])
        scores={}
        for point in run.POINTS:
            mixed=perfect.copy()
            if point[-1]=="S":mixed[0]=-1
            if point[-1]=="F":mixed[1]=-1
            scores[point]={"raw":-np.ones((40,378)),"stage-cal":np.ones((40,378)),"first-cal":mixed}
        root=self.temporary()/"nullable_evaluation"
        saved=run.collect(root,scores,labelled,split)
        result=run.finalize(root)
        negative=result["absolute"]["B_S"]["raw"]
        positive=result["absolute"]["B_S"]["stage-cal"]
        for key in ("precision","mcc"):
            self.assertIsNone(negative["by_domain"]["A"][key]["mean"])
            self.assertIsNone(negative["by_domain"]["A"][key]["conditional_95pct_interval"])
            self.assertEqual(negative["by_domain"]["A"][key]["undefined_groups"],20)
            self.assertIsNone(negative["macro_all"][key])
            self.assertEqual(negative["macro_undefined_groups"][key],40)
            paired=result["comparisons"]["F-S"]["first-cal"]["A"][key]
            self.assertEqual(paired["undefined_groups"],2)
            self.assertEqual(paired["total_groups"],20)
            self.assertIsNone(paired["conditional_95pct_interval"])
            self.assertIsNone(result["forgetting_comparisons"]["F-S"]["first-cal"][key]["mean"])
        self.assertEqual(negative["by_domain"]["A"]["recall"],{"mean":0.,"conditional_95pct_interval":[0.,0.]})
        self.assertEqual(negative["by_domain"]["A"]["f1"]["mean"],0.)
        self.assertIsNone(negative["fixed_half_classification"]["all"]["precision"])
        self.assertAlmostEqual(positive["by_domain"]["A"]["precision"]["mean"],20/378)
        self.assertIsNone(positive["by_domain"]["A"]["mcc"]["mean"])
        self.assertEqual(result["comparisons"]["F-S"]["first-cal"]["B"]["precision"]["mean"],0.)
        # Pooled precision is defined by summed counts even when one group is N/A.
        self.assertEqual(result["absolute"]["B_S"]["first-cal"]["fixed_half_classification"]["A"]["precision"],1.)
        rec=saved["points"]["B_S"]["raw"]["matrix"]
        values=data.read_json(root/rec["path"])["values"]
        self.assertIsNone(values[0][6]);self.assertIsNone(values[0][11]);self.assertEqual(values[0][7],0.)
        self.evidence("nullable_metrics",dict(all_negative=negative,all_positive=positive,
            paired_mismatched_groups=result["comparisons"]["F-S"]["first-cal"],fixture="Handwritten only; role names exercise persistence, not calibration claims"))

    def test_solver_failure_context_and_native_write_reserve(self):
        import step28_balanced_pooling_verify as verify
        g=handmade_group("failure_history")
        slots=record.assignment(g)
        left,right=np.triu_indices(len(slots),1)
        table=torch.zeros(len(left))
        bad=int(np.flatnonzero((slots[left]==0)&(slots[right]==1))[0])
        target=table.numpy().copy();target[bad]=np.nan
        stats={}
        with self.assertRaises(method.SolverFailure) as caught:
            method.objective(table,g,3,"F",target,stats,{"stage":2,"step":7,"role":"history"})
        details=caught.exception.details
        self.assertEqual({k:details[k] for k in ("arm","stage","step","role","view","group_uid","failure_count")},
            dict(arm="F",stage=2,step=7,role="history",view="teacher_A0",group_uid=g.uid,failure_count=1))
        self.assertEqual(len(details["failed_blocks"]),1)
        self.assertEqual(details["failed_blocks"][0],dict(batch_index=0,n=2,m=2,iteration=0,marginal_error=None,
            feasible_error=None,gap_checked=None,account_pair_index=0,account_slots=[0,1]))
        with self.assertRaises(method.SolverFailure) as short:
            method.solve(torch.tensor([[[8.,1.,-4.],[2.,-2.,3.]]]),maximum_iterations=1)
        row=short.exception.details["failed_blocks"][0]
        self.assertEqual((row["n"],row["m"],row["iteration"]),(2,3,1))
        self.assertGreater(row["marginal_error"],method.SOLVER["marginal_tolerance"])
        out=self.temporary()
        data.write_json(out/"failure.json",{"solver_failure":details})
        self.assertEqual(data.read_json(out/"failure.json")["solver_failure"],details)
        model=object()
        with mock.patch.object(verify.run.base.persistence,"checkpoint_reserve",return_value=6), \
                mock.patch.object(verify.core,"save_state") as save, \
                mock.patch.object(verify.shutil,"disk_usage",return_value=mock.Mock(free=50)) as disk:
            with self.assertRaises(RuntimeError):
                verify.native_save(out/"too_large.pt",model,None,{},lambda reserve:verify.disk_reserve(out,7,reserve,12))
            save.assert_not_called()
            disk.return_value.free=4
            with self.assertRaises(RuntimeError):
                verify.native_save(out/"too_little_free.pt",model,None,{},lambda reserve:verify.disk_reserve(out,7,reserve,20))
            save.assert_not_called()
            disk.return_value.free=50
            verify.native_save(out/"fits.pt",model,None,{},lambda reserve:verify.disk_reserve(out,7,reserve,20))
            save.assert_called_once()
        self.evidence("failure_and_reserve",dict(solver_failure=details,iteration_failure=short.exception.details,
            no_write_before_approved_reserve=True,fixture="Actual handwritten solver failure; mocked filesystem/save to prove refusal ordering, no large payload"))

    def test_AB_filter_before_construction_alignment_and_parse_ledger(self):
        root = self.temporary()
        rows, originals, files = [], {}, {}
        c = run.config()
        c["data_root"] = str(root)
        for split, count in (("train",60),("development",20)):
            groups = []
            for d in "ABC":
                for i in range(count):
                    g = handmade_group(f"{d}_{split}_{i:02}")
                    rows.append(dict(group_uid=g.uid,domain=d,split=split,group_index=i,accounts=28,items=56))
                    groups.append(g)
            originals[split] = groups
            (root/split/"supervision").mkdir(parents=True)
            with (root/split/"items.jsonl").open("w",encoding="utf-8") as stream:
                for g in groups:
                    for seller, items in zip(g.sellers,g.items):
                        for uid,title,desc in items:
                            stream.write(json.dumps(dict(group_uid=g.uid,seller_uid=seller,item_uid=uid,title=title,description=desc),ensure_ascii=False)+"\n")
            with (root/split/"supervision/pairs.csv").open("w",encoding="utf-8",newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["group_uid","seller_uid_left","seller_uid_right","label"])
                for g in groups:
                    for k,(i,j) in enumerate(itertools.combinations(range(28),2)):
                        # C values are deliberately nonnumeric: numerical alignment would fail.
                        writer.writerow([g.uid,g.sellers[i],g.sellers[j],"FORBIDDEN_C_NUMERIC" if g.uid.startswith("C_") else g.labels[k]])
            for name in (split+"/items.jsonl",split+"/supervision/pairs.csv"):
                files[name] = data.record(root/name,root)
        with (root/"groups.csv").open("w",encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files["groups.csv"] = data.record(root/"groups.csv",root)
        data.write_json(root/"manifest.json",dict(study="seller_alias_continual_expression_shift",root_seed=20260910,files=files))
        data.write_json(root/"validation.json",dict(status="PASS_GENERATION_CONTRACT_NOT_MODEL_QUALIFICATION"))
        c["data_manifest_sha256"],c["data_validation_sha256"] = data.sha256(root/"manifest.json"),data.sha256(root/"validation.json")
        constructed, aligned = [], []
        real_group, real_align = data.Group,data.align_labels
        def construct(uid,*args,**kwargs):
            constructed.append(uid)
            self.assertFalse(uid.startswith("C_"))
            return real_group(uid,*args,**kwargs)
        def align(g,rs):
            aligned.append(g.uid)
            self.assertFalse(g.uid.startswith("C_"))
            return real_align(g,rs)
        with mock.patch.object(data,"Group",side_effect=construct),mock.patch.object(data,"align_labels",side_effect=align):
            groups,meta,_ = run.public_inputs(c)
            job = root/"job"
            job.mkdir()
            data.write_json(job/"access.json",dict(train=0,valid=0,heldout=0,owners=0))
            def loader(gs,cfg,s):
                self.assertEqual(data.read_json(job/"access.json")["train"],1)
                return run.attach_labels(gs,cfg,s,meta)
            groups["train"] = run.previous.parse_once(job,groups["train"],c,"train",loader=loader)
            with self.assertRaises(ValueError):
                run.previous.parse_once(job,groups["train"],c,"train",loader=loader)
        self.assertEqual((len(groups["train"]),len(groups["development"])),(120,40))
        self.assertEqual(len(aligned),120)
        selected,split = run.partition(groups,meta,c)
        old_selected,old_split = run.base.partition({"train":originals["train"],"development":originals["development"]},rows,c)
        for role in split:
            self.assertEqual(split[role],[r for r in old_split[role] if r["domain"] in "AB"])
        # Failed opening is consumed before loader error; missing blind prerequisites never opens.
        data.write_json(job/"access.json",dict(train=0,valid=0,heldout=0,owners=0))
        with self.assertRaises(RuntimeError):
            run.previous.parse_once(job,[],c,"train",loader=mock.Mock(side_effect=RuntimeError("handwritten loader failure")))
        self.assertEqual(data.read_json(job/"access.json")["train"],1)
        with self.assertRaises((ValueError,KeyError)):
            run.open_valid(job,{},groups["development"],meta,c)
        self.assertEqual(data.read_json(job/"access.json")["valid"],0)

    def test_independent_three_arm_orchestration_and_blind_gate(self):
        root=self.temporary();groups={};metadata=[]
        for split,count in (("train",60),("development",20)):
            groups[split]=[]
            for d in "AB":
                for i in range(count):
                    g=handmade_group(f"{d}_{split}_{i:02}")
                    metadata.append(dict(group_uid=g.uid,domain=d,split=split,group_index=i))
                    groups[split].append(g if split=="train" else data.Group(g.uid,g.sellers,g.items))
        loads,calls=[],[];c=run.config()
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
            row=dict(total=1.,encoder_lr=run.parent.stage_lr(step),head_lr=.001,pooling=method.identity(arm))
            for role,g in (("current",current),("history",history)):
                if g is not None:
                    seed=data.seed_for(20260918,"ABC",stage,step,role,g.uid,"regroup")
                    row[role]=dict(supervised=1.,weighted=1. if role=="current" else .1,mse0=0.,
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
            if not gs:return np.empty((0,378),np.float32)
            return np.stack([method.account_logits(torch.from_numpy(reference(model,g,cfg)),record.assignment(g),arm,
                cfg.setdefault("pooling_stats",{})).numpy() for g in gs]).astype(np.float32)
        with mock.patch.object(run,"public_inputs",return_value=(groups,metadata,{"handwritten":True})), \
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
            self.assertEqual((len(loads),len(calls)),(3,1728))
            self.assertEqual(manifest["physical_updates"],1728)
            self.assertEqual(manifest["gradient_group_presentations"],2592)
            self.assertEqual(sum(map(len,manifest["diagnostics"].values())),36)
            self.assertEqual(len(list((root/"run/memory").glob("*.bin"))),6)
            self.assertFalse(list((root/"run/work").glob("*.pt")))
            scores,checked=run.blind_gate(root/"run",manifest)
            self.assertEqual((set(scores),checked),(set(run.POINTS),split))
            for arm in method.ARMS:
                self.assertEqual(manifest["starts"][arm],manifest["points"]["A_"+arm]["continuity"])
            self.assertNotEqual(manifest["starts"]["S"]["model"],manifest["starts"]["F"]["model"])
            for collection,key in (("points","A_F"),("diagnostics","B_G"),("training","A_S"),("starts","F")):
                damaged=copy.deepcopy(manifest);del damaged[collection][key]
                with self.assertRaises(ValueError):run.open_valid(root,damaged,valid,meta,c)
                self.assertEqual(data.read_json(root/"access.json")["valid"],0)
            damaged=copy.deepcopy(manifest);damaged["starts"]["F"]=damaged["starts"]["S"]
            with self.assertRaises(ValueError):run.open_valid(root,damaged,valid,meta,c)
            damaged=copy.deepcopy(manifest);damaged["points"]["B_F"]["pooling"]=method.identity("S")
            with self.assertRaises(ValueError):run.open_valid(root,damaged,valid,meta,c)
            self.assertEqual(data.read_json(root/"access.json")["valid"],0)
        self.evidence("orchestration",dict(fixture="Actual scheduler, pooling, Memory and checkpoint/diagnostic IO; mocked update/forward/metrics/calibration only",
            scheduler_calls=len(calls),real_updates_in_this_fixture=0,independent_initializations=len(loads),
            diagnostics=36,endpoints=6,own_A_states=True,blind_rejections=True,
            scope="Control-flow evidence only. Real tiny gradient/update and actual diagnostic tests are separate."))



# Fixed terminal witnesses; no formal inputs or model files.
def terminal_block(e=1e-4):
    p = np.array([[1/3-.05-e, .05, e], [.05, 1/3-.05-e, e], [e, e, 1/3-2*e]], np.float64)
    return torch.from_numpy(np.log(9*p))


def rounded_reference(k):
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 80
        d = lambda x: Decimal.from_float(float(x))
        a, b, c, z = (d(k[i,j]) for i,j in ((0,0),(0,1),(0,2),(2,2)))
        assert k[0,0] == k[1,1] and k[0,1] == k[1,0]
        assert all(k[i,j] == k[0,2] for i,j in ((1,2),(2,0),(2,1)))
        s = Decimal(1)/3
        beta = (2*c-z).exp()/(a.exp()+b.exp())
        x = 2*beta*s/(3*beta+(beta*beta+4*beta).sqrt())
        u = (s-x)/(1+(b-a).exp()); v = s-x-u
        p = [[u,v,x],[v,u,x],[x,x,s-2*x]]
        value = sum(p[i][j]*d(k[i,j])-p[i][j]*(9*p[i][j]).ln() for i in range(3) for j in range(3))
        return float(value), np.array([[float(t) for t in row] for row in p])


def terminal_table(device="cpu"):
    slots = np.repeat(np.arange(28), 3)
    i,j = torch.triu_indices(84,84,1,device=device)
    return terminal_block().to(device).float()[i%3,j%3], slots


def terminal_group(uid, role):
    """Edge-colour the fixed A0/A1 incidence so each account has all three types.

    This constructs one deterministic handwritten input, without choosing a seed
    or searching model/text outcomes. The actual regroup function remains active.
    """
    g = handmade_group(uid,3)
    seed = data.seed_for(20260918,"ABC",2,49,role,g.uid,"regroup")
    a0,a1 = record.assignment(g),record.assignment(g,seed)
    remaining = set(range(84)); colors = np.full(84,-1)
    for color in range(3):
        matched = {}
        def augment(account, seen):
            for edge in sorted(remaining):
                dest = int(a1[edge])
                if a0[edge] != account or dest in seen: continue
                seen.add(dest)
                if dest not in matched or augment(int(a0[matched[dest]]),seen):
                    matched[dest]=edge; return True
            return False
        for account in range(28):
            if not augment(account,set()): raise AssertionError("Handwritten incidence is not regular")
        edges=set(matched.values());remaining-=edges
        for edge in edges:colors[edge]=color
    for slots in (a0,a1):
        assert all(sorted(colors[slots==i].tolist())==[0,1,2] for i in range(28))
    items=tuple(tuple((item[0],f"{colors[i*3+j]}/{i}",f"{colors[i*3+j]}/{i}") for j,item in enumerate(rows)) for i,rows in enumerate(g.items))
    return data.Group(g.uid,g.sellers,items,g.labels),colors


def terminal_model():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            embedding=torch.zeros(84,31)
            for account in range(28):
                for color in range(3):embedding[3*account+color,color]=1;embedding[3*account+color,3+account]=1
            self.embedding=torch.nn.Embedding.from_pretrained(embedding,freeze=False)
        def tokenizer(self,texts,**kwargs):
            return {"input_ids":torch.tensor([[int(t.split("/")[0])+3*int(t.split("/")[1])] for t in texts])}
        def forward(self,batch):
            return {"sentence_embedding":self.embedding(batch["input_ids"][:,0])}
    model=core.build_model(Encoder(),62)
    model.head=torch.nn.Linear(124,1)
    k=terminal_block().float()
    a,b,c=(float(k[i,j])*2 for i,j in ((0,1),(0,2),(1,2)))
    with torch.no_grad():
        model.head.weight.zero_();model.head.bias.zero_()
        model.head.weight[0,:3]=torch.tensor([(a+b-c)/2,(a+c-b)/2,(b+c-a)/2])
        model.head.weight[0,62:65]=4*k.diagonal()
        # Row/column-only offsets distinguish hard-negative ranks without changing
        # the exact transport plan. Both different/same source-account pairs agree.
        offsets=torch.arange(28)*.01
        model.head.weight[0,3:31]=2*offsets
        model.head.weight[0,65:93]=8*offsets
    return model


class TerminalPoolingTests(BalancedPoolingTests):
    def test_terminal_precision_vjp_and_batch(self):
        evidence=[]
        for dtype in (torch.float64,torch.float32):
            k=terminal_block().to(dtype)
            variants=torch.stack((k,k.T,k.flip((0,1)),torch.zeros_like(k)))
            stats={};values,q,info=method.solve(variants,stats)
            ref,refp=rounded_reference(k.double().numpy())
            self.assertLessEqual(abs(float(values[0])-ref),1e-8)
            self.assertLessEqual(float((q[0]-torch.from_numpy(refp)).abs().sum()),2e-4)
            self.assertEqual(stats["terminal_acceptances"],3)
            self.assertEqual(stats["terminal_min_iteration"],4096)
            run.solver_gate(stats)
            for j,plan in ((1,q[0].T),(2,q[0].flip((0,1)))):
                torch.testing.assert_close(q[j],plan,atol=1e-12,rtol=0)
            x=variants.clone().requires_grad_();upstream=x.new_tensor([-.7,1.3,.2,-2.])
            output=method.block_scores(x,"F");gradient,=torch.autograd.grad(output,x,upstream)
            torch.testing.assert_close(gradient,upstream[:,None,None]*q.to(dtype),atol=0,rtol=0)
            bound=1e-8 if dtype==torch.float64 else 1e-8+1e-7*max(1,abs(ref))
            self.assertLessEqual(abs(float(output[0])-ref),bound)
            self.assertLessEqual(float((q[0].to(dtype).double()-torch.from_numpy(refp)).abs().sum()),2e-4)
            before=copy.deepcopy(stats)
            with self.assertRaises(method.SolverFailure):
                method.solve(torch.stack((torch.zeros_like(k),terminal_block(1e-8).to(dtype))),stats)
            self.assertEqual(stats,{**before,"failures":1})
            with self.assertRaises(method.SolverFailure):method.solve(k[None],maximum_iterations=1)
            # Successful calls sum T; a failed mixed call commits no partial success.
            method.solve(k[None],before)
            self.assertEqual(before["terminal_acceptances"],4)
            evidence.append(dict(dtype=str(dtype),reference_value=ref,reference_plan=refp.tolist(),value=float(values[0]),
                plan=q[0].tolist(),actual_vjp=gradient.tolist(),upstream=upstream.tolist(),stats=info,
                failed_mixed_call_kept_success_counts=True,early_terminal_rejected=True))
        self.evidence("terminal_precision",evidence)

    def test_terminal_real_chunked_update(self):
        c=run.config(); model=terminal_model();direct=copy.deepcopy(model)
        current,cc=terminal_group("terminal_current","current")
        history,hc=terminal_group("terminal_history","history")
        oa,ob=[core.make_optimizer(m,c) for m in (model,direct)]
        with torch.no_grad():
            initial=record.pair_table(model,record.record_vectors(model,current,c))
        lookup={(i,j):idx for idx,(i,j) in enumerate(itertools.combinations(range(84),2))}
        first=[int(np.flatnonzero(cc[:3]==i)[0]) for i in range(3)]
        second=[int(np.flatnonzero(cc[3:6]==i)[0])+3 for i in range(3)]
        k=np.array([[float(initial[lookup[min(i,j),max(i,j)]]) for j in second] for i in first])
        value,plan=rounded_reference(k)
        ti,tj=np.triu_indices(84,1)
        target=(k[hc[ti],hc[tj]]+.3).astype(np.float32)
        actual_proxy=[];actual_objective=method.objective
        def observed(table,*args,**kwargs):
            table.register_hook(lambda grad:actual_proxy.append(grad.detach().clone()))
            return actual_objective(table,*args,**kwargs)
        # Observation-only hook: no replaced solve, value, gradient or update.
        with mock.patch.object(method,"objective",side_effect=observed):
            row=method.update(model,oa,current,history,target,c,"F",2,49)
        direct.train();ob.zero_grad(set_to_none=True);totals=[];expected_proxy=[]
        def independent_pool(table,slots,colors,offset=0.):
            values=[]
            for a,b in itertools.combinations(range(28),2):
                terms=[];weights=[]
                for i in np.flatnonzero(slots==a):
                    for j in np.flatnonzero(slots==b):
                        terms.append(table[lookup[min(i,j),max(i,j)]])
                        weights.append(plan[colors[i],colors[j]])
                p=torch.tensor(weights,dtype=table.dtype)
                values.append((torch.stack(terms)*p).sum()-(p*torch.log(9*p)).sum())
            return torch.stack(values)
        for role,g,colors,t in (("current",current,cc,None),("history",history,hc,target)):
            z=record.record_vectors(direct,g,c);i,j=torch.triu_indices(84,84,1)
            table=direct.head(torch.cat(((z[i]-z[j]).abs(),z[i]*z[j]),1)).flatten()
            seed=data.seed_for(20260918,"ABC",2,49,role,g.uid,"regroup")
            v0,v1=[independent_pool(table,record.assignment(g,s),colors) for s in (None,seed)]
            loss=(literal_loss(v0,g)+literal_loss(v1,g))/2
            if t is not None:
                old=independent_pool(torch.from_numpy(t),record.assignment(g),colors)
                loss=.1*loss+.5*(v0-old).square().mean()
            expected_proxy.append(torch.autograd.grad(loss,table,retain_graph=True)[0])
            totals.append(float(loss.detach()));loss.backward()
        torch.nn.utils.clip_grad_norm_(direct.parameters(),1.)
        ob.param_groups[0]["lr"],ob.param_groups[1]["lr"]=run.parent.stage_lr(49),.001;ob.step()
        self.assertAlmostEqual(row["total"],sum(totals),places=5)
        proxy_delta=max(float((a-b).abs().max()) for a,b in zip(actual_proxy,expected_proxy))
        self.evidence("terminal_update_proxy",dict(max_delta=proxy_delta,actual_objective=row["total"],independent_objective=sum(totals),solver_summary=c["pooling_stats"]))
        for a,b in zip(actual_proxy,expected_proxy):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        changes={}
        for (name,a),b in zip(model.named_parameters(),direct.parameters()):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)
            self.assertGreater(float(a.grad.norm()),0)
            changes[name]=dict(gradient_norm=float(a.grad.norm()),gradient_max_delta=float((a.grad-b.grad).abs().max()),parameter_max_delta=float((a-b).abs().max()))
        self.assertGreater(row["history"]["supervised"],0);self.assertGreater(row["history"]["mse0"],0)
        self.assertEqual(c["pooling_stats"]["terminal_acceptances"],5*378)
        self.assertEqual(run.parent.adam_step(oa),1)
        self.evidence("terminal_real_update",dict(actual_input=k.tolist(),reference_value=value,reference_plan=plan.tolist(),
            update=row,proxy_max_delta=proxy_delta,parameters=changes,actual_optimizer_calls=1,
            fixture="Trainable type/account-token embedding and linear symmetric head with fixed row/column offsets; actual A0/A1, objectives, 512 VJP and AdamW. Independent analytic envelope reference; observation-only proxy hook."))

    def test_terminal_gate_budget_and_inherited_evidence(self):
        s=dict(blocks=2,failures=0,iterations=4096,marginal_error=4e-7,ordinary_marginal_error=0.,
               terminal_acceptances=1,terminal_min_iteration=4096,feasible_error=1e-16,gap_checked=4e-10)
        run.solver_gate(s)
        for key,value in (("terminal_min_iteration",1),("iterations",4095),("terminal_acceptances",0),
                          ("marginal_error",0.),("ordinary_marginal_error",2e-10),("gap_checked",1e-8),
                          ("feasible_error",1e-9),("failures",1),("terminal_acceptances",3)):
            with self.assertRaises(ValueError):run.solver_gate({**s,key:value})
        old=method.identity("F");old["solver"]["version"]="log_sinkhorn_round_v1"
        with self.assertRaises(ValueError):method.bind(tiny_model(),{"pooling":old})
        inherited=run.prior_evidence()
        m={f"{k}_{a}":1. for a in method.ARMS for k in ("t_A","t_B","v","h")}
        m.update(L=1.,s_F=1.,r_F=1.,s_I=1.,r_I=1.,W_F=1000,W_I=500)
        initial=run.projection(m)["projected_total_seconds"]
        m["t_B_F"]+=(80000-initial)/(1.25*288)
        projected=run.projection(m)
        self.assertAlmostEqual(projected["projected_total_seconds"],80000)
        self.assertFalse(projected["within_budget"])
        budget_root=self.temporary();budget=run.Budget(budget_root);snap=budget.snapshot()
        self.assertEqual(snap["prior_charged_seconds"],18428.570150192827)
        self.assertEqual(snap["cumulative_seconds"],snap["elapsed_seconds"]+18428.570150192827)
        budget.started-=67972
        budget.check()  # The former cutoff is no longer the approved cutoff.
        budget.started-=10800
        with self.assertRaises(RuntimeError):budget.check()
        p=run.policy();gpu=copy.deepcopy(inherited["native_result"])
        gpu.update(mode="gpu",scope="terminal-q",source_files=run.terminal_q_sources(),prior_compute_seconds=364,cumulative_charged_seconds=365)
        n=gpu["native"];natural=copy.deepcopy(n["measurements"])
        stress=dict(c_gpu=.001,c_cpu=.001,accounts=28,records=84,pairs=378,
                    solver_summaries={k:{**s,"blocks":378,"terminal_acceptances":378,"ordinary_marginal_error":None} for k in ("cpu","gpu")})
        n.update(actual_updates=3,cumulative_updates=11,F_pooling_restores_verified=True,terminal_gpu_vjp_verified=True,
                 inherited_native=p["prior_evidence"]["native_result"],terminal_stress=stress,
                 natural_measurements=natural,solver_summary=s)
        for key,mult,cost in (("t_A_F",2,"c_gpu"),("t_B_F",5,"c_gpu"),("v_F",1,"c_gpu"),("h_F",4,"c_cpu")):
            n["measurements"][key]=natural[key]+mult*stress[cost]
        run.delta_native_gate(gpu,inherited["native_result"])
        for bad in (8,11):
            damaged=copy.deepcopy(gpu);damaged["native"]["actual_updates"]=bad
            with self.assertRaises(ValueError):run.delta_native_gate(damaged,inherited["native_result"])
        damaged=copy.deepcopy(gpu);damaged["source_files"]=inherited["native_result"]["source_files"]
        with self.assertRaises(ValueError):run.delta_native_gate(damaged,inherited["native_result"])
        adjust=(80000-run.projection(n["measurements"])["projected_total_seconds"])/(1.25*288)
        n["natural_measurements"]["t_B_F"]+=adjust
        n["measurements"]["t_B_F"]=n["natural_measurements"]["t_B_F"]+5*stress["c_gpu"]
        gpu_path=budget_root/"gpu_fixture.json";data.write_json(gpu_path,gpu)
        job=budget_root/"not_created_job"
        gate=dict(job=job.relative_to(data.ROOT).as_posix(),runtime=p["runtime"],supervision=p["supervision"],gpu=data.record(gpu_path,data.ROOT))
        with mock.patch.object(run,"reviewed_gate",return_value=gate),mock.patch.object(run,"public_inputs") as reader, \
             mock.patch.dict(os.environ,{"RECORD_REPLAY_STARTED_EPOCH":str(__import__("time").time())}):
            with self.assertRaisesRegex(ValueError,"projection"):run.execute(job,budget_root/"fixture_gate")
            reader.assert_not_called();self.assertFalse(job.exists())
        self.evidence("terminal_gate_budget",dict(inherited_updates=inherited["native_result"]["native"]["actual_updates"],
            remaining_seconds=78771,over_budget_projection=projected,new_directory_budget=snap,
            inconsistent_terminal_rejected=True,old_identity_rejected=True))

    def test_runtime_inheritance(self):
        frozen = run.terminal_q_sources()
        repair = data.ROOT/"reports/documentation/20261015/balanced_pooling_repair"
        gpu = data.read_json(repair/"qualification/gpu/attempt01/result/result.json")
        cpu = data.read_json(repair/"qualification/cpu/attempt02/result/result.json")
        self.assertEqual(cpu["source_files"], frozen)
        self.assertEqual(gpu["source_files"], frozen)
        run.delta_native_gate(gpu, run.prior_evidence()["native_result"])
        projection = run.projection(gpu["native"]["measurements"])
        self.assertEqual(projection["projected_total_seconds"], 77073.03956648684)
        self.assertTrue(projection["within_budget"])
        current = run.sources()
        scratch = self.temporary()
        # A labelled unit-test receipt exercises the gate, never authorizes a run.
        receipt = scratch/"synthetic_runtime_cpu.json"
        data.write_json(receipt, dict(status="PASS_RUNTIME_ONLY", source_files=current,
            prior_charged_seconds=168, cumulative_charged_seconds=169, synthetic_fixture=True))
        gate = data.read_json(repair/"qualification/native_gate.json")
        gate.update(status="APPROVED_BALANCED_POOLING_FORMAL", source_files=current,
            runtime_authorization=data.record(repair/"runtime/authorization.json", data.ROOT),
            runtime_cpu=data.record(receipt, data.ROOT))
        gate_path = scratch/"synthetic_formal_gate.json"
        data.write_json(gate_path, gate)
        run.reviewed_gate(gate_path, "FORMAL")
        for field, value in (("source_files", frozen), ("runtime_cpu", gate["cpu"])):
            data.write_json(gate_path, {**gate, field: value})
            with self.assertRaises(ValueError):
                run.reviewed_gate(gate_path, "FORMAL")
        for action in ("solver_change", "extra_source", "missing_source"):
            damaged = copy.deepcopy(current)
            if action == "solver_change":
                next(r for r in damaged if r["path"] == "scripts/step28_balanced_pooling.py")["sha256"] = "changed"
            elif action == "extra_source":
                damaged.append(dict(path="scripts/unapproved.py", bytes=1, sha256="changed"))
            else:
                damaged.pop()
            with mock.patch.object(run, "sources", return_value=damaged):
                with self.assertRaisesRegex(ValueError, "dependency"):
                    run.terminal_q_sources()
        self.evidence("runtime_inheritance", dict(inherited_source_count=len(frozen), current_source_count=len(current),
            real_saved_native_validated=True, projection=projection,
            solver_change_extra_and_missing_sources_rejected=True, stale_gate_and_cpu_rejected=True,
            reviewed_gate_check="Synthetic runtime receipt in a removed scratch directory; no formal gate issued",
            model_or_optimizer_calls=0))
