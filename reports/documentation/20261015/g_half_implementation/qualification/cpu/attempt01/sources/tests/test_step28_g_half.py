"""Handwritten H references and changed production paths; execute only on Linux."""
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
import step28_g_half as method
import step28_g_half_run as run
from test_step28_record_replay import handmade_group, tiny_model
record, core, data, diag = run.record, run.core, run.data, run.diag


def literal_pool(table, slots, mode="H"):
    lookup = {pair: table[k] for k, pair in enumerate(itertools.combinations(range(len(slots)), 2))}
    result = []
    for a, b in itertools.combinations(range(28), 2):
        values = torch.stack([lookup[tuple(sorted((int(i), int(j))))] for i in np.flatnonzero(slots == a)
                              for j in np.flatnonzero(slots == b)]).double()
        peak = values.max()
        g = peak + torch.log(torch.exp(values-peak).sum()/len(values))
        result.append((g if mode == "G" else values.mean() if mode == "S" else (values.sum()/len(values)+g)/2).to(table.dtype))
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


def literal_objective(table, group, seed, target=None):
    a0, a1 = record.assignment(group), record.assignment(group, seed)
    v0, v1 = literal_pool(table, a0), literal_pool(table, a1)
    supervised = (literal_loss(v0, group)+literal_loss(v1, group))/2
    if target is None:
        return supervised
    teacher = literal_pool(torch.as_tensor(target, dtype=table.dtype), a0)
    return .1*supervised+.5*(v0-teacher).square().mean()


def shared_memory():
    groups = [handmade_group(f"A_fit_{i:02}") for i in range(48)]
    memory = record.Memory("ABC")
    memory.retain(groups, 1, lambda g: (np.sin(np.arange(1540)/7)*2).astype(np.float32))
    memory.maps["pooling"] = method.identity()
    return memory


class GHalfTests(unittest.TestCase):
    def temporary(self):
        t = tempfile.TemporaryDirectory(dir=os.environ["G_HALF_TEST_ROOT"])
        self.addCleanup(t.cleanup)
        return Path(t.name)

    def evidence(self, name, value):
        data.write_json(Path(os.environ["G_HALF_TEST_ROOT"])/(name+".json"), value)

    def test_local_value_derivative_precision_and_rejections(self):
        cases = [np.zeros((2,2)), np.array([[4,0],[0,0]]), np.array([[-12,0],[0,0]])]
        cases += [np.sin(np.arange(n*m).reshape(n,m)*.7)*5 for n,m in ((2,8),(8,2),(8,8))]
        rows = []
        for k in cases:
            for dtype in (torch.float32, torch.float64):
                x = torch.tensor(k, dtype=dtype)[None].requires_grad_()
                v = x.detach().double().numpy().ravel().tolist()
                peak = max(v); weights = [math.exp(t-peak) for t in v]; den = math.fsum(weights)
                ref = .5*math.fsum(v)/len(v)+.5*(peak+math.log(den/len(v)))
                q = np.asarray([.5/len(v)+.5*w/den for w in weights]).reshape(k.shape)
                actual = method.block_scores(x)
                gradient, = torch.autograd.grad(actual.sum(), x)
                if dtype == torch.float64:
                    np.testing.assert_allclose(actual.detach().numpy(), [ref], atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(gradient[0].numpy(), q, atol=1e-12, rtol=1e-12)
                else:
                    self.assertLessEqual(abs(float(actual)-ref), 1e-8+1e-7*max(1,abs(ref)))
                    self.assertLessEqual(float(np.abs(gradient[0].numpy()-q).sum()), 2e-7)
                torch.testing.assert_close(method.block_scores(x.transpose(1,2)), actual, atol=1e-12, rtol=1e-12)
                shifted = method.block_scores(x.double()+3.25)
                np.testing.assert_allclose(shifted.detach().numpy(), [ref+3.25], atol=1e-12, rtol=1e-12)
                rows.append(dict(shape=list(k.shape), dtype=str(dtype), value=float(actual), reference=ref,
                                 gradient_l1_error=float(np.abs(gradient[0].numpy()-q).sum())))
        for bad in (torch.zeros(1,1,2), torch.full((1,2,2), float('nan')), torch.full((1,2,2), float('inf'))):
            with self.assertRaises(ValueError): method.block_scores(bad)
        self.evidence("local_precision", rows)

    def test_full_objective_top5_teacher_and_proxy_gradient(self):
        g = handmade_group("objective")
        slots = record.assignment(g)
        table = (torch.sin(torch.arange(1540, dtype=torch.float32)*.39)*3).requires_grad_()
        target = (np.cos(np.arange(1540)*.27)*2).astype(np.float32)
        actual, log = method.objective(table, g, 29, "G_half", target)
        independent = literal_objective(table, g, 29, target)
        ga, = torch.autograd.grad(actual, table, retain_graph=True)
        gi, = torch.autograd.grad(independent, table, retain_graph=True)
        torch.testing.assert_close(actual, independent, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(ga, gi, atol=2e-6, rtol=2e-5)
        s = literal_pool(table, slots); t = literal_pool(torch.from_numpy(target), slots)
        no_a1 = .1*literal_loss(s,g)+.5*(s-t).square().mean()
        d1 = (literal_pool(table,record.assignment(g,29))-literal_pool(torch.from_numpy(target),record.assignment(g,29))).square().mean()
        self.assertGreater(abs(float(actual-no_a1)), 1e-5)
        self.assertGreater(float(d1), 0)
        for wrong in ("S", "G"):
            self.assertGreater(abs(float((s-literal_pool(torch.from_numpy(target),slots,wrong)).square().mean())-log["mse0"]), 1e-4)
        # Sparse nonconstant teacher: mix scores before squaring, not branch losses.
        i,j = np.triu_indices(56,1)
        birth = np.where((i%2==0)&(j%2==0),4.,0.).astype(np.float32)
        href = .5+.5*(math.log(math.exp(4)+3)-math.log(4))
        student = torch.full((1540,), href, requires_grad=True)
        _, zero = method.objective(student,g,29,"G_half",birth)
        self.assertLess(zero["mse0"],1e-12)
        wrong_d = .5*((literal_pool(student,slots,"S")-literal_pool(torch.from_numpy(birth),slots,"S"))**2+
                       (literal_pool(student,slots,"G")-literal_pool(torch.from_numpy(birth),slots,"G"))**2).mean()
        self.assertGreater(float(wrong_d), .69)
        # Four high negatives plus the sparse/flat pair force an actual fifth/sixth swap for query0.
        values = torch.full((1540,),-3.)
        for b in (3,4,5,6): values[torch.from_numpy((i//2==0)&(j//2==b))] = 5.
        mask = (i//2==0)&(j//2==7)
        values[torch.from_numpy(mask)] = torch.tensor([4.,0.,0.,0.])
        values[torch.from_numpy((i//2==0)&(j//2==8))] = 2.
        values.requires_grad_()
        h = method.account_logits(values,slots,"G_half")
        gs = literal_pool(values,slots,"G")
        truth = torch.tensor(g.labels,dtype=torch.float32)
        ranking = record.parent.ranking
        htop = ranking.hard_candidates(h,truth)[2][0].tolist()
        gtop = ranking.hard_candidates(gs,truth)[2][0].tolist()
        self.assertEqual(htop,[3,4,5,6,8]); self.assertEqual(gtop,[3,4,5,6,7])
        hard = ranking.objectives(h,truth,.5)["hard"]
        hard_grad, = torch.autograd.grad(hard,values)
        self.assertGreater(float(hard_grad[torch.from_numpy((i//2==0)&(j//2==8))].abs().sum()),0)
        reference_loss = literal_loss(h,g)
        live = ranking.objectives(h,truth,.5)["total"]
        torch.testing.assert_close(live,reference_loss,atol=2e-6,rtol=2e-5)
        self.evidence("objective_witnesses", dict(value=float(actual), independent=float(independent), proxy_max_error=float((ga-gi).abs().max()),
            no_a1_difference=float(actual-no_a1), correct_sparse_D=zero["mse0"], wrong_mixture_D=float(wrong_d), H_top5=htop,G_top5=gtop,
            history_weight=.1,teacher_weight=.5,teacher_A0_only=True))

    def test_real_vjp_encoder_head_gradients_and_updates(self):
        c=run.config(); current,history=handmade_group("current"),handmade_group("history")
        target=(np.sin(np.arange(1540)/7)*3).astype(np.float32)
        model=tiny_model(); direct=copy.deepcopy(model)
        oa,ob=[core.make_optimizer(m,c) for m in (model,direct)]
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        observed={}
        def observe(role, log): observed[role]={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        row=method.update(model,oa,current,history,target,c,"G_half",2,49,observe=observe)
        direct.train(); ob.zero_grad(set_to_none=True); values=[]; components={}
        for role,g,t in (("current",current,None),("history",history,target)):
            torch.manual_seed(data.seed_for(20260918,"ABC",2,49,role,"dropout"))
            z=record.record_vectors(direct,g,c)
            i,j=torch.triu_indices(len(z),len(z),1); u,v=z[i],z[j]
            table=direct.head(torch.cat(((u-v).abs(),u*v),1)).flatten()
            seed=data.seed_for(20260918,"ABC",2,49,role,g.uid,"regroup")
            loss=literal_objective(table,g,seed,t); values.append(float(loss.detach())); loss.backward()
            for n,p in direct.named_parameters():
                torch.testing.assert_close(observed[role][n],p.grad,atol=2e-6,rtol=2e-5)
        norm=torch.nn.utils.clip_grad_norm_(direct.parameters(),1.)
        ob.param_groups[0]["lr"],ob.param_groups[1]["lr"]=run.parent.stage_lr(49),.001
        ob.step()
        self.assertAlmostEqual(row["total"],sum(values),places=5)
        for (n,a),b in zip(model.named_parameters(),direct.parameters()):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)
            components[n]=dict(preclip_norm=float(observed["history"][n].norm()),postclip_norm=float(a.grad.norm()),delta=float((a-before[n]).abs().max()))
        self.assertGreater(components["encoder.embedding.weight"]["delta"],0)
        self.assertGreater(components["head.0.weight"]["delta"],0)
        self.evidence("independent_full_graph",dict(objective=row["total"],reference=sum(values),gradient_norm=float(norm),parameters=components,
            scope="One actual tiny paired update; per-role preclip, postclip and parameters compared; no BGE/formal data"))

    def test_actual_capture_restore_and_next_update(self):
        from step28_g_half_verify import state,restore
        c=run.config(); model=tiny_model(); optimizer=core.make_optimizer(model,c)
        current,history=handmade_group("current"),handmade_group("history")
        method.update(model,optimizer,current,None,None,c,"G_half",1,1)
        for v in optimizer.state.values():v["step"].fill_(336)
        saved,rng,modes=state(model,optimizer),diag.rng_state(),[m.training for m in model.modules()]
        target=(3*np.sin(np.arange(1540)/7)).astype(np.float32)
        first=method.update(model,optimizer,current,history,target,c,"G_half",2,49)
        expected,expected_rng=state(model,optimizer),diag.rng_state()
        restore(model,optimizer,saved,rng,modes)
        memory=shared_memory();memory.begin_stage(2)
        for _ in range(48):memory.draw()
        payload=memory.to_bytes();fit=[handmade_group(f"fit{i}") for i in range(48)]
        valid=[data.Group(g.uid,g.sellers,g.items) for g in [handmade_group(f"valid{i}") for i in range(40)]]
        out=self.temporary()
        capture=method.capture(model,optimizer,fit,memory,valid,c,"G_half",2,1,out,"fixture")
        result=data.read_json(out/capture["path"])
        with diag.neutral(model):
            raw=method.score(model,valid,c,"G_half");mean=record.score(model,valid,c)
            np.testing.assert_array_equal(run.previous.array(out,result["valid_blind"],(40,378),np.float32),raw)
            self.assertFalse(np.array_equal(raw,mean))
            for role,gs in (("fit",fit),("cache",memory.reservoir.groups)):
                live=method.score(model,gs,c,"G_half")
                np.testing.assert_array_equal(run.previous.array(out,result[role],(len(gs),3),np.float64),diag.metrics(gs,live))
            g=memory.reservoir.groups[0];live=record.reference(model,g,c);old=memory.tables[g.uid]
            for key,seed in (("D0",None),("D1",data.seed_for(20261008,"ABC",2,g.uid,"diagnostic_regroup"))):
                slots=record.assignment(g,seed)
                error=(literal_pool(torch.from_numpy(live),slots)-literal_pool(torch.from_numpy(old),slots)).detach().numpy().astype(np.float64)
                self.assertAlmostEqual(result["teacher_errors"][0][key],float(np.mean(error**2)),places=6)
                wrong=(literal_pool(torch.from_numpy(live),slots,"S")-literal_pool(torch.from_numpy(old),slots,"S")).detach().numpy().astype(np.float64)
                self.assertGreater(abs(result["teacher_errors"][0][key]-float(np.mean(wrong**2))),1e-5)
        self.assertEqual(core.state_digest(saved),core.state_digest(state(model,optimizer)))
        self.assertEqual((rng,payload,modes),(diag.rng_state(),memory.to_bytes(),[m.training for m in model.modules()]))
        repeated=method.update(model,optimizer,current,history,target,c,"G_half",2,49)
        self.assertEqual(first,repeated);self.assertEqual(core.state_digest(expected),core.state_digest(state(model,optimizer)))
        self.assertEqual(expected_rng,diag.rng_state())
        meta={"pooling":method.identity(),"rng":diag.rng_state(),"modes":[m.training for m in model.modules()]}
        method.bind(model,meta)
        with diag.neutral(model):before=method.restored_score(model,[current],c)
        for kind,opt in (("full",optimizer),("inference",None)):
            p=out/(kind+".pt");receipt=core.save_state(p,model,opt,meta)
            model.pooling_identity={"arm":"S"}
            run.restore_point(p,receipt,model,opt,"G_half")
            with diag.neutral(model):np.testing.assert_array_equal(before,method.restored_score(model,[current],c))
        for meta_bad in ({},{"pooling":{**method.identity(),"mean_weight":.6}},{"pooling":{**method.identity(),"arm":"G"}}):
            with self.assertRaises(ValueError):method.bind(model,meta_bad)
        self.evidence("actual_capture_restore",dict(next_update_bitwise_equal=True,all_roles_H=True,D0_D1_H=True,
            full_inference_restores=True,metadata_rejections=True,fixture="Counter336 synthetic, three real tiny updates; actual 48/6/40 capture."))

    def test_single_candidate_orchestration_and_blind_gate(self):
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
        with mock.patch.object(run,"baseline_inputs",return_value=(fake_baseline,oldroot,{},{"public_inputs":{"handwritten":True}})), \
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
            self.assertEqual((len(loads),len(calls)),(1,576))
            self.assertEqual(manifest["physical_updates"],576)
            self.assertEqual(manifest["gradient_group_presentations"],864)
            self.assertEqual(sum(map(len,manifest["diagnostics"].values())),12)
            self.assertEqual(len(list((root/"run/memory").glob("*.bin"))),2)
            self.assertFalse(list((root/"run/work").glob("*.pt")))
            for stage in "AB":
                log=data.read_json(root/"run/updates"/(stage+"_G_half.json"))
                for other in ("S","G"):data.write_json(oldroot/"run/updates"/(stage+"_"+other+".json"),log)
            scores,checked=run.blind_gate(root/"run",manifest)
            self.assertEqual((set(scores),checked),(set(run.POINTS),split))
            for arm in method.ARMS:
                self.assertEqual(manifest["starts"][arm],manifest["points"]["A_"+arm]["continuity"])
            for collection,key in (("points","A_G_half"),("diagnostics","B_G_half"),("training","A_G_half"),("starts","G_half")):
                damaged=copy.deepcopy(manifest);del damaged[collection][key]
                with self.assertRaises(ValueError):run.open_valid(root,damaged,valid,meta,c)
                self.assertEqual(data.read_json(root/"access.json")["valid"],0)
            damaged=copy.deepcopy(manifest);damaged["starts"]["G_half"]["model"]="wrong_A"
            with self.assertRaises(ValueError):run.open_valid(root,damaged,valid,meta,c)
            damaged=copy.deepcopy(manifest);damaged["points"]["B_G_half"]["pooling"]={"arm":"S"}
            with self.assertRaises(ValueError):run.open_valid(root,damaged,valid,meta,c)
            self.assertEqual(data.read_json(root/"access.json")["valid"],0)
            badlog=data.read_json(oldroot/"run/updates/B_G.json")
            badlog["rows"][0]["current"]["assignment_sha256"]="wrong"
            data.write_json(oldroot/"run/updates/B_G.json",badlog)
            with self.assertRaises(ValueError):run.open_valid(root,manifest,valid,meta,c)
            self.assertEqual(data.read_json(root/"access.json")["valid"],0)
        self.evidence("orchestration",dict(fixture="Actual scheduler, pooling, Memory and checkpoint/diagnostic IO; mocked update/forward/metrics/calibration only",
            scheduler_calls=len(calls),real_updates_in_this_fixture=0,independent_initializations=len(loads),
            diagnostics=12,endpoints=2,own_A_states=True,blind_rejections=True,
            scope="Control-flow evidence only. Real tiny gradient/update and actual diagnostic tests are separate."))



    def test_saved_comparisons_directions_permissions_and_budgets(self):
        labelled=[handmade_group(f"{d}_valid_{i:02}") for d in "AB" for i in range(20)]
        split={"development":[dict(group_uid=g.uid,domain=g.uid[0]) for g in labelled]}
        raw=np.stack([np.asarray(g.labels,np.float32)*2-1+np.sin(np.arange(378)+i).astype(np.float32)*.2 for i,g in enumerate(labelled)])
        score={p:{"raw":raw.copy(),"stage-cal":raw.astype(np.float64),"first-cal":raw.astype(np.float64)} for p in run.POINTS}
        temp=self.temporary();hroot=temp/"H";oldroot=temp/"old";oldroot.mkdir()
        col=run.collect(hroot,score,labelled,split)
        oldcol=run.collect(oldroot/"evaluation",score,labelled,split)
        # Explicit saved-metric algebra fixture, not a performance claim or recomputation of old results.
        columns=col["columns"]
        def set_values(origin,collection,point,map_value,loss_value):
            for role,rec in collection["points"][point].items():
                p=origin/rec["matrix"]["path"];v=data.read_json(p)
                for row in v["values"]:
                    row[columns.index("map")]=map_value
                    row[columns.index("brier")]=loss_value
                    row[columns.index("log_loss")]=loss_value
                data.write_json(p,v);rec["matrix"]=data.record(p,origin)
                qpath=origin/rec["queries"]["path"];q=np.load(qpath,allow_pickle=False);q[:,:,0]=map_value
                rec["queries"]=run.previous.save_array(qpath,q,origin)
        for origin,collection,a,b,la,lb in ((hroot,col,.5,.6,.4,.5),(oldroot/"evaluation",oldcol,.4,.4,.3,.3)):
            set_values(origin,collection,"A_G_half",a,la);set_values(origin,collection,"B_G_half",b,lb)
        oldcol["points"]={stage+"_"+arm:copy.deepcopy(oldcol["points"][stage+"_G_half"]) for stage in "AB" for arm in ("S","G")}
        data.write_json(hroot/"collected.json",col)
        baseline={"points":list(oldcol["points"])}
        with mock.patch.object(run,"baseline_inputs",return_value=(baseline,oldroot,oldcol,{})):
            result=run.finalize(hroot)
            self.assertTrue(result["success"]["passed"]);self.assertTrue(result["success"]["direct_G_new_B_positive"])
            for other in ("S","G"):
                self.assertAlmostEqual(result["comparisons"]["G_half-"+other]["raw"]["A"]["map"]["mean"],.2)
                self.assertAlmostEqual(result["forgetting_comparisons"]["G_half-"+other]["raw"]["map"]["mean"],-.1)
                for key in ("brier","log_loss"):
                    self.assertAlmostEqual(result["forgetting_comparisons"]["G_half-"+other]["raw"][key]["mean"],.1)
            oldcol["group_ids"]=oldcol["group_ids"][::-1]
            with self.assertRaises(ValueError):run.finalize(hroot)
        m={k:1. for k in ("t_A","t_B","v","h","L","s_F","r_F","s_I","r_I")};m.update(W_F=1000,W_I=500)
        projection=run.projection(m)
        self.assertEqual(projection["train_seconds"],576);self.assertEqual(projection["forward_seconds"],1482)
        self.assertEqual(projection["io_seconds"],9);self.assertEqual(projection["projected_total_seconds"],1.25*(576+1482+9+1800))
        self.assertFalse(run.projection({**m,"t_B":1000})["within_budget"])
        for key in m:
            with self.assertRaises(ValueError):run.projection({**m,key:0})
        job=temp/"job";job.mkdir();data.write_json(job/"access.json",dict(train=0,valid=0,heldout=0,owners=0))
        with self.assertRaises(RuntimeError):
            run.previous.parse_once(job,[],run.config(),"train",loader=mock.Mock(side_effect=RuntimeError("handwritten read failure")))
        with self.assertRaises(ValueError):
            run.previous.parse_once(job,[],run.config(),"train",loader=mock.Mock())
        with self.assertRaises((ValueError,KeyError)):run.open_valid(job,{},[],[],run.config())
        self.assertEqual(data.read_json(job/"access.json"),dict(train=1,valid=0,heldout=0,owners=0))
        budget=run.Budget(job);budget.started-=28801
        with self.assertRaises(RuntimeError):budget.check()
        from step28_g_half_verify import disk_reserve
        with self.assertRaises(RuntimeError):disk_reserve(temp,8,3,10)
        self.evidence("comparison_and_permissions",dict(success=result["success"],forgetting=result["forgetting_comparisons"]["G_half-S"]["raw"],
            fixture="Handwritten saved-metric signs/UID and real collection; no baseline payload read",parse_failure_consumed=True,blind_open_blocked=True,projection=projection))
