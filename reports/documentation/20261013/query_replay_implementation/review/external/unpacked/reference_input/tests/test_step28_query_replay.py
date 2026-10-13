"""Handwritten independent math and actual affected paths; Linux qualification only."""
from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import step28_query_replay as method
import step28_query_replay_run as run
from test_step28_record_replay import handmade_group, tiny_model, literal_scores

record, core, data, diag = run.record, run.core, run.data, run.diag


def shared_memory():
    groups = [handmade_group(f"A_fit_{i:02}") for i in range(48)]
    memory = record.Memory("ABC")
    memory.retain(groups, 1, lambda g: np.zeros(1540, np.float32))
    return memory


def natural_indices(group):
    ends = np.cumsum([0]+list(map(len, group.items)))
    return [list(range(int(ends[a]), int(ends[a+1]))) for a in range(28)]


def literal_u(table, group):
    indices = natural_indices(group)
    lookup = {pair: table[k] for k, pair in enumerate(itertools.combinations(range(sum(map(len, group.items))), 2))}
    return {(q, r): torch.stack([lookup[tuple(sorted((s, r)))] for s in indices[q]]).mean()
            for q in range(28) for a in range(28) if a != q for r in indices[a]}


def literal_directed(table, group, rows):
    means = literal_u(table, group)
    return torch.stack([torch.stack([table.sum()*0 if j == q else torch.stack([means[q, r] for r in rows[q][j]]).mean()
                                    for j in range(28)]) for q in range(28)])


def literal_row_loss(scores, group):
    lookup = {pair: group.labels[k] for k, pair in enumerate(itertools.combinations(range(28), 2))}
    bce, rank, hard = [], [], []
    for q in range(28):
        all_j = [j for j in range(28) if j != q]
        pos = [j for j in all_j if lookup[tuple(sorted((q, j)))]]
        neg = [j for j in all_j if j not in pos]
        top = sorted(neg, key=lambda j: (-float(scores[q, j].detach()), j))[:5]
        bce.extend(torch.nn.functional.softplus(scores[q, j])-lookup[tuple(sorted((q, j)))]*scores[q, j] for j in all_j)
        rank.append(torch.logsumexp(scores[q, all_j], 0)-scores[q, pos].mean())
        hard.append(torch.stack([torch.nn.functional.softplus(scores[q, n]-scores[q, p]) for p in pos for n in top]).mean())
    return torch.stack(bce).mean()+torch.stack(rank).mean()+.5*torch.stack(hard).mean()


def literal_response(table, group, target):
    indices = natural_indices(group)
    u, old = literal_u(table, group), literal_u(torch.as_tensor(target, dtype=table.dtype).detach(), group)
    # The handwritten fixture's component layout, independent of implementation discovery.
    components = [list(range(3*i, 3*i+3)) for i in range(4)]+[[i, i+1] for i in range(12, 28, 2)]
    terms = []
    for q in range(28):
        for component in components:
            accounts = [a for a in component if a != q]
            population = sum(len(indices[a]) for a in accounts)
            b = sum((population-len(indices[a]))/(len(indices[a])*population*(population-1)) for a in accounts)
            for a in accounts:
                errors = torch.stack([u[q, r]-old[q, r] for r in indices[a]])
                terms.append(b*((errors-errors.mean())**2).sum())
    return torch.stack(terms).sum()/756


class QueryReplayTests(unittest.TestCase):
    def temporary(self):
        t = tempfile.TemporaryDirectory(dir=os.environ["QUERY_REPLAY_TEST_ROOT"])
        self.addCleanup(t.cleanup)
        return Path(t.name)

    def test_candidate_exclusion_count_multiset_keys_and_rng(self):
        full = handmade_group("varying", 8)
        group = data.Group(full.uid, full.sellers, tuple(v[:2+i%7] for i, v in enumerate(full.items)), full.labels)
        natural = natural_indices(group)
        global_rng = diag.rng_state()
        rows = method.candidates(group, 2, 17, "query_replay")
        bootstrap = method.candidates(group, 2, 17, "within_account_bootstrap")
        self.assertEqual(global_rng, diag.rng_state())
        self.assertEqual(rows, method.candidates(group, 2, 17, "query_replay"))
        components = [list(range(3*i, 3*i+3)) for i in range(4)]+[[i, i+1] for i in range(12, 28, 2)]
        changed, repeated = False, False
        for q in range(28):
            self.assertEqual(sorted(sum(rows[q], [])), sorted(r for a in range(28) if a != q for r in natural[a]))
            for k, component in enumerate(components):
                accounts = [a for a in component if a != q]
                pool = sum([natural[a] for a in accounts], [])
                rng = random.Random(data.seed_for(20260918, "query_replay", "ABC", 2, 17, group.uid, q, k))
                if len(accounts) > 1:
                    rng.shuffle(pool)
                cursor = 0
                for a in accounts:
                    self.assertEqual(rows[q][a], pool[cursor:cursor+len(natural[a])])
                    cursor += len(natural[a])
                    changed |= set(rows[q][a]) != set(natural[a])
                    self.assertEqual(len(bootstrap[q][a]), len(natural[a]))
                    self.assertTrue(set(bootstrap[q][a]) <= set(natural[a]))
                    repeated |= len(set(bootstrap[q][a])) < len(natural[a])
            if q >= 12:
                mate = q+1 if q % 2 == 0 else q-1
                self.assertEqual(rows[q][mate], natural[mate])
        self.assertTrue(changed and repeated)
        self.assertNotEqual(rows, method.candidates(group, 2, 18, "query_replay"))

    def test_directed_pooling_and_row_loss_independent_gradient(self):
        group = handmade_group("directed")
        table = torch.sin(torch.arange(1540, dtype=torch.float64)/9).requires_grad_()
        rows = method.candidates(group, 2, 13, "query_replay")
        actual, expected = method.directed_scores(table, group, rows), literal_directed(table, group, rows)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        self.assertGreater(float((actual-actual.T).abs().max()), .01)
        loss, ref = method.row_objective(actual, group), literal_row_loss(expected, group)
        torch.testing.assert_close(loss, ref, atol=1e-10, rtol=1e-9)
        a, = torch.autograd.grad(loss, table, retain_graph=True)
        b, = torch.autograd.grad(ref, table)
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
        scores = torch.zeros((28, 28), dtype=torch.float64, requires_grad=True)
        # Exact ties exercise stable top5 and live selected-logit gradients.
        x, y = method.row_objective(scores, group), literal_row_loss(scores, group)
        torch.testing.assert_close(torch.autograd.grad(x, scores, retain_graph=True)[0], torch.autograd.grad(y, scores)[0], atol=1e-10, rtol=1e-9)
        values = torch.cos(torch.arange(378, dtype=torch.float64))
        matrix = values.new_zeros((28, 28))
        i, j = torch.triu_indices(28, 28, 1)
        matrix[i, j] = matrix[j, i] = values
        original = run.parent.ranking.objectives(values, torch.tensor(group.labels, dtype=torch.float64), .5)["total"]
        torch.testing.assert_close(method.row_objective(matrix, group), original, atol=1e-10, rtol=1e-9)

    def test_exact_finite_population_expectation_and_symmetric_derivative(self):
        group = handmade_group("finite")
        lookup = {p:k for k,p in enumerate(itertools.combinations(range(56), 2))}
        table = torch.zeros(1540, dtype=torch.float64)
        e = [-1., 1., -2., 2.]
        for s in (0, 1):
            for r, value in zip(range(24, 28), e):
                table[lookup[s, r]] = value
        square = []
        for first in itertools.combinations(range(4), 2):
            second = [r for r in range(4) if r not in first]
            square.append((sum(e[r] for r in first)/2)**2+(sum(e[r] for r in second)/2)**2)
        actual = method.response_loss(table, group, np.zeros(1540))
        self.assertAlmostEqual(float(actual), sum(square)/len(square)/756, places=12)
        live = torch.sin(torch.arange(1540, dtype=torch.float64)/7).requires_grad_()
        target = torch.cos(torch.arange(1540, dtype=torch.float64)/11)
        ref = literal_response(live, group, target)
        value = method.response_loss(live, group, target)
        torch.testing.assert_close(value, ref, atol=1e-10, rtol=1e-9)
        grad, = torch.autograd.grad(value, live, retain_graph=True)
        independent, = torch.autograd.grad(ref, live)
        torch.testing.assert_close(grad, independent, atol=1e-10, rtol=1e-9)
        self.assertGreater(float(grad.abs().max()), 0.)
        # FP32 is the actual table format; not just a float64 reference.
        v32 = method.response_loss(live.detach().float(), group, target.float())
        torch.testing.assert_close(v32.double(), value.detach(), atol=1e-6, rtol=1e-5)

    def test_whole_history_blind_spaces_bias_and_same_source_gradient(self):
        group = handmade_group("blind")
        lookup = {p:k for k,p in enumerate(itertools.combinations(range(56), 2))}
        base = torch.zeros(1540, dtype=torch.float64)
        for qa, aa, block in ((0, 3, [[1., -1.], [-1., 1.]]), (12, 13, [[3., 1.], [-1., -3.]])):
            changed = base.clone()
            for s in range(2):
                for r in range(2):
                    changed[lookup[2*qa+s, 2*aa+r]] = block[s][r]
            for arm in (("Q", "C_q", "C_b") if qa == 0 else ("Q", "C_q")):
                x = method.objective(base, group, 19, arm, base.numpy(), step=5)[0]
                y = method.objective(changed, group, 19, arm, base.numpy(), step=5)[0]
                torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-9)
            if qa == 12:
                # Bootstrap can see K2 marginal changes; this larger blind space is Q/C_q only.
                rows = method.candidates(group, 2, 5, "within_account_bootstrap")
                self.assertGreater(float(method.directed_scores(changed, group, rows).abs().max()), 0.)
        t = torch.tensor([-1., 1.]*28, dtype=torch.float64)
        biased = torch.stack([t[i]+t[j] for i, j in itertools.combinations(range(56), 2)])
        self.assertEqual(float(record.account_logits(biased, record.assignment(group)).square().sum()), 0.)
        self.assertGreater(float(method.response_loss(biased, group, base)), 0.)
        self.assertEqual(float(method.response_loss(biased, group, biased)), 0.)
        biased.requires_grad_()
        loss = method.objective(biased, group, 19, "C_q", base.numpy(), step=5)[0]
        gradient, = torch.autograd.grad(loss, biased)
        same = torch.tensor([i//2 == j//2 for i,j in itertools.combinations(range(56),2)])
        self.assertEqual(float(gradient[same].abs().sum()), 0.)
        self.assertGreater(float(gradient[~same].abs().sum()), 0.)

    def test_actual_chunked_update_matches_independent_single_graph(self):
        c = run.config()
        current, history = handmade_group("current"), handmade_group("history")
        model = tiny_model()
        direct = copy.deepcopy(model)
        oa, ob = [core.make_optimizer(m, c) for m in (model, direct)]
        target = np.sin(np.arange(1540)/7).astype(np.float32)*3
        row = method.update(model, oa, current, history, target, c, "C_q", 2, 1)
        direct.train()
        ob.zero_grad(set_to_none=True)
        values = []
        for role, group, old in (("current", current, None), ("history", history, target)):
            torch.manual_seed(data.seed_for(20260918,"ABC",2,1,role,"dropout"))
            z = record.record_vectors(direct, group, c)
            i, j = torch.triu_indices(len(z), len(z), 1)
            u, v = z[i], z[j]
            table = direct.head(torch.cat(((u-v).abs(),u*v),1)).flatten()
            seed = data.seed_for(20260918,"ABC",2,1,role,group.uid,"regroup")
            a0 = literal_scores(table, np.repeat(np.arange(28),2))
            truth = torch.tensor(group.labels,dtype=torch.float32)
            natural = run.parent.ranking.objectives(a0,truth,.5)["total"]
            if old is None:
                a1 = literal_scores(table,record.assignment(group,seed))
                loss = (natural+run.parent.ranking.objectives(a1,truth,.5)["total"])/2
            else:
                rows = method.candidates(group,2,1,"query_replay")
                augmented = literal_row_loss(literal_directed(table,group,rows),group)
                teacher = literal_scores(torch.from_numpy(old),np.repeat(np.arange(28),2))
                loss = .1*(natural+augmented)/2+.5*(a0-teacher).square().mean()+.25*literal_response(table,group,old)
            values.append(float(loss.detach()))
            loss.backward()
        torch.nn.utils.clip_grad_norm_(direct.parameters(),1.)
        ob.param_groups[0]["lr"],ob.param_groups[1]["lr"] = run.parent.stage_lr(1),.001
        ob.step()
        self.assertAlmostEqual(row["total"],sum(values),places=5)
        for a,b in zip(model.parameters(),direct.parameters()):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)

    def test_actual_R_effect_complete_S_and_diagnostic_neutrality(self):
        from step28_query_replay_verify import state, restore, component_effects
        c = run.config()
        model = tiny_model()
        optimizer = core.make_optimizer(model,c)
        current,history = handmade_group("current"),handmade_group("history")
        method.update(model,optimizer,current,None,None,c,"S",1,1)
        for v in optimizer.state.values():
            v["step"].fill_(336)
        saved,rng,modes = state(model,optimizer),diag.rng_state(),[m.training for m in model.modules()]
        target = (3*np.sin(np.arange(1540)/7)).astype(np.float32)
        method.update(model,optimizer,current,history,target,c,"S",2,49)
        original = state(model,optimizer)
        restore(model,optimizer,saved,rng,modes)
        record.update(model,optimizer,current,history,target,c,"S","ABC",2,49)
        self.assertEqual(core.state_digest(original),core.state_digest(state(model,optimizer)))
        restore(model,optimizer,saved,rng,modes)
        q = method.update(model,optimizer,current,history,target,c,"Q",2,49)
        off = state(model,optimizer)
        restore(model,optimizer,saved,rng,modes)
        on = method.update(model,optimizer,current,history,target,c,"C_q",2,49)
        self.assertEqual(q["history"]["assignment_sha256"],on["history"]["assignment_sha256"])
        self.assertGreater(on["history"]["response"],0.)
        effects = component_effects(model,off)
        expected,expected_rng = state(model,optimizer),diag.rng_state()
        restore(model,optimizer,saved,rng,modes)
        memory = shared_memory()
        memory.begin_stage(2)
        for _ in range(48):memory.draw()
        payload = memory.to_bytes()
        fit = [handmade_group(f"fit{i}") for i in range(48)]
        valid = [data.Group(g.uid,g.sellers,g.items) for g in [handmade_group(f"valid{i}") for i in range(40)]]
        path = self.temporary()
        diag.capture(model,optimizer,fit,memory,valid,c,"C_q","ABC",2,1,path,"hand",scope="AB",table_diagnostic=method.diagnostic)
        self.assertEqual(core.state_digest(saved),core.state_digest(state(model,optimizer)))
        self.assertEqual((rng,payload,modes),(diag.rng_state(),memory.to_bytes(),[m.training for m in model.modules()]))
        after = method.update(model,optimizer,current,history,target,c,"C_q",2,49)
        self.assertEqual(after,on)
        self.assertEqual(core.state_digest(expected),core.state_digest(state(model,optimizer)))
        self.assertEqual(expected_rng,diag.rng_state())
        data.write_json(Path(os.environ["QUERY_REPLAY_TEST_ROOT"])/"actual_update_evidence.json",{"effects":effects,"actual_full_diagnostic_neutral":True,"next_update_equal":True,"scope":"handwritten lightweight encoder/head; not BGE qualification"})

    def test_projection_and_entry_gate_fail_closed(self):
        m = dict(t_A=1.,t_S=2.,t_Q=3.,t_q=4.,t_b=5.,f=.1,h=.2,L=7.,s_F=8.,r_F=9.,s_I=10.,r_I=11.,W_F=1000,W_I=500)
        result = run.projection(m)
        self.assertEqual(result["train_seconds"],288*15)
        self.assertAlmostEqual(result["forward_seconds"],345.4+28.8)
        self.assertEqual(result["io_seconds"],7+40+81+50+55)
        self.assertFalse(run.projection({**m,"t_q":1000})["within_budget"])
        self.assertFalse(run.projection({**m,"W_F":20*2**30})["within_budget"])
        for k in m:
            with self.assertRaises(ValueError):run.projection({**m,k:0})
        root = self.temporary()
        gate = root/"gate.json"
        data.write_json(gate,{"status":"APPROVED_QUERY_REPLAY_NATIVE","source_files":[],"user_scope_approved":True,"review_disposition":"NO_OPEN_BLOCKERS"})
        with self.assertRaises(ValueError):run.reviewed_gate(gate,"NATIVE")

    def test_saved_queries_metrics_and_fixed_contrasts(self):
        root = self.temporary()/"evaluation"
        labelled = [handmade_group(f"{domain}_valid_{i:02}") for domain in "AB" for i in range(20)]
        split = {"development":[{"group_uid":g.uid,"domain":g.uid[0]} for g in labelled]}
        raw = np.stack([np.asarray(g.labels,dtype=np.float32)*2+np.sin(np.arange(378)+i).astype(np.float32) for i,g in enumerate(labelled)])
        scores = {p:{"raw":raw.copy(),"stage-cal":raw.astype(np.float64)*1.2-.3,"first-cal":raw.astype(np.float64)*.8+.1} for p in run.POINTS}
        saved = run.collect(root,scores,labelled,split)
        report = run.finalize(root)
        self.assertEqual(saved["status"],"ALL_15_METRIC_COUNT_SETS_SAVED")
        self.assertEqual(set(report["comparisons"]),{"C_q-S","Q-S","C_q-Q","C_q-C_b"})
        for comparison in report["comparisons"].values():
            for role in comparison.values():
                for domain in role.values():
                    for value in domain.values():
                        self.assertEqual(value,{"mean":0.,"conditional_95pct_interval":[0.,0.]})
        for point in saved["points"].values():
            for role in point.values():
                q = run.previous.array(root,role["queries"],(40,28,10),np.float64)
                self.assertTrue(np.isfinite(q).all())

    def test_native_restore_isolation_and_output_accounting(self):
        from step28_query_replay_verify import state,restore
        model = tiny_model()
        c = run.config()
        optimizer = core.make_optimizer(model,c)
        current = handmade_group("restore")
        method.update(model,optimizer,current,None,None,c,"S",1,1)
        saved,rng,modes = state(model,optimizer),diag.rng_state(),[m.training for m in model.modules()]
        digest = core.state_digest(saved)
        restore(model,optimizer,saved,rng,modes)
        method.update(model,optimizer,current,None,None,c,"S",1,2)
        self.assertEqual(core.state_digest(saved),digest)
        root = self.temporary()
        job = root/"reports/job"
        job.mkdir(parents=True)
        qualification = root/"reports/qualification/cpu"
        qualification.mkdir(parents=True)
        (qualification/"kept.txt").write_bytes(b"qualification")
        (job/"file.txt").write_bytes(b"result")
        job.with_suffix(".console.txt").write_bytes(b"console")
        with mock.patch.object(data,"ROOT",root):
            budget = run.Budget(job)
            files = {p.resolve() for p in budget.output_files() if p.is_file()}
            self.assertTrue({(qualification/"kept.txt").resolve(),(job/"file.txt").resolve(),job.with_suffix(".console.txt").resolve()} <= files)

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

    def test_actual_AB_orchestration_restores_own_memory_and_blind_rejections(self):
        root = self.temporary()
        groups, metadata = {}, []
        for split,count in (("train",60),("development",20)):
            groups[split] = []
            for d in "AB":
                for i in range(count):
                    g = handmade_group(f"{d}_{split}_{i:02}")
                    metadata.append(dict(group_uid=g.uid,domain=d,split=split,group_index=i))
                    groups[split].append(g if split=="train" else data.Group(g.uid,g.sellers,g.items))
        model_calls, update_calls = [], []
        c = run.config()
        class Budget:
            phase = "handwritten orchestration"
            def check(self,*args):
                pass
            def snapshot(self):
                return {}
        def load(cfg):
            model_calls.append("fresh")
            return tiny_model()
        def advance(model,optimizer,t):
            for p in model.parameters():
                optimizer.state[p] = {"step":torch.tensor(float(t)),"exp_avg":torch.zeros_like(p),"exp_avg_sq":torch.zeros_like(p)}
            with torch.no_grad():
                model.head[2].bias.fill_(t*.001)
        def update(model,optimizer,current,history,target,cfg,arm,stage,step,check):
            self.assertTrue(current.uid.startswith("A_" if stage==1 else "B_"))
            self.assertEqual(run.parent.adam_step(optimizer),(stage-1)*288+step-1)
            update_calls.append((arm,stage,step,current.uid,history.uid if history else None))
            if history:
                self.assertTrue(history.uid.startswith("A_"))
                self.assertEqual(target.shape,(1540,))
            advance(model,optimizer,(stage-1)*288+step)
            current_log = dict(supervised=1.,weighted=1.,mse0=0.,mse1=0.,distillation=0.)
            row = dict(current=current_log,total=1.,encoder_lr=run.parent.stage_lr(step),head_lr=.001)
            if history:
                historical = {**current_log,"weighted":.1}
                if arm != "S":
                    rule = "within_account_bootstrap" if arm == "C_b" else "query_replay"
                    historical.update(response=0.,response_weight=0. if arm=="Q" else .25,
                        weighted_response=0.,augmentation_rule=rule,
                        assignment_sha256=hashlib.sha256(data.json_bytes(method.candidates(history,stage,step,rule))).hexdigest())
                row.update(history=historical,total=1.1)
            return row
        truth = np.asarray(handmade_group("geometry").labels)
        table_cache = {}
        def reference(model,g,cfg,check=lambda:None):
            if g.uid not in table_cache:
                slots = record.assignment(g)
                lookup = {p:k for k,p in enumerate(itertools.combinations(range(28),2))}
                values = []
                for i,j in itertools.combinations(range(len(slots)),2):
                    a,b = sorted((int(slots[i]),int(slots[j])))
                    k = 0 if a==b else lookup[a,b]
                    values.append((0 if a==b else truth[k]*3)+k*.0001)
                table_cache[g.uid] = np.asarray(values,np.float32)
            return table_cache[g.uid]*(1+float(model.head[2].bias.detach()))
        def score(model,gs,cfg,check=lambda:None):
            return np.stack([record.account_logits(torch.from_numpy(reference(model,g,cfg)),record.assignment(g)).numpy() for g in gs]).astype(np.float32)
        with mock.patch.object(run,"public_inputs",return_value=(groups,metadata,{"handwritten":True})), \
             mock.patch.object(run,"attach_labels",side_effect=lambda gs,cfg,s,meta:gs), \
             mock.patch.object(core,"model_files",return_value=c["models"]["split_rank"]), \
             mock.patch.object(record,"load_model",side_effect=load), \
             mock.patch.object(method,"update",side_effect=update), \
             mock.patch.object(record,"reference",side_effect=reference), \
             mock.patch.object(record,"score",side_effect=score), \
             mock.patch.object(diag,"metrics",side_effect=lambda gs,vs:np.zeros((len(gs),3),np.float64)), \
             mock.patch.object(run.parent.calibration,"fit",return_value={"status":"PASS_CALIBRATION_FIT","a":1.,"b":0.}), \
             mock.patch("builtins.print"):
            data.write_json(root/"access.json",dict(train=0,valid=0,heldout=0,owners=0))
            manifest,split,valid,meta = run.train(root,Budget())
            self.assertEqual(len(update_calls),1440)
            self.assertEqual(len(model_calls),1)
            self.assertEqual(manifest["physical_updates"],1440)
            self.assertEqual(manifest["gradient_group_presentations"],2592)
            self.assertEqual(sum(len(v) for v in manifest["diagnostics"].values()),30)
            self.assertEqual(manifest["status"],run.COMPLETE)
            self.assertEqual(len(list((root/"run/memory").glob("*.bin"))),5)
            self.assertFalse(list((root/"run/work").glob("*.pt")))
            scores,checked_split = run.blind_gate(root/"run",manifest)
            self.assertEqual(checked_split,split)
            self.assertEqual(set(scores),set(run.POINTS))
            for collection,key in (("points","B_C_q"),("diagnostics","B_S"),("training","B_Q"),("starts","C_b")):
                damaged = copy.deepcopy(manifest)
                del damaged[collection][key]
                with self.assertRaises(ValueError):
                    run.open_valid(root,damaged,valid,meta,c)
                self.assertEqual(data.read_json(root/"access.json")["valid"],0)
            damaged = copy.deepcopy(manifest)
            damaged["points"]["B_C_q"]["full_restore_verified"] = False
            with self.assertRaises(ValueError):
                run.open_valid(root,damaged,valid,meta,c)
            self.assertEqual(data.read_json(root/"access.json")["valid"],0)
        # This fixture checks actual orchestration/serialization, not 1440 real BGE updates.
        evidence = {"fixture":"Actual scheduler with explicitly mocked tiny updates/forward/metrics; real checkpoint/Memory/diagnostic IO and blind gate",
                    "main_update_calls":len(update_calls),"diagnostic_points":30,
                    "shared_A_restores_identical":True,"five_memories":True,"scope":"mock updates validate orchestration only; real mathematical and update tests are separate"}
        data.write_json(Path(os.environ["QUERY_REPLAY_TEST_ROOT"])/"scheduler_evidence.json",evidence)

