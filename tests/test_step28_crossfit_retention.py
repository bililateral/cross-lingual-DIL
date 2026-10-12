"""Independent handwritten math, actual updates, and the affected AB lifecycle."""
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

import step28_crossfit_retention as method
import step28_crossfit_retention_run as run
import step28_crossfit_retention_wait as listener
from test_step28_record_replay import handmade_group, tiny_model, literal_scores

record, core, data, diag = run.record, run.core, run.data, run.diag


def shared_memory(records=2, text_length=0):
    groups = [handmade_group(f"A_fit_{i:02}", records, text_length) for i in range(48)]
    memory = record.Memory("ABC")
    memory.retain(groups, 1, lambda g: np.zeros(sum(map(len, g.items))*(sum(map(len, g.items))-1)//2, np.float32))
    return memory


def source_ids(memory):
    return {g.uid: {"fold": i%2, "role": "U"+str(i%2), "source_sha256": "0"*64,
                    "full_source_sha256": "1"*64} for i,g in enumerate(memory.reservoir.groups)}


def literal_floor(scores, group, values):
    pairs = list(itertools.combinations(range(28), 2))
    positions = {p: k for k,p in enumerate(pairs)}
    lookup = lambda a,b: positions[tuple(sorted((a,b)))]
    terms, k = [], 0
    for q in range(28):
        positive = [p for p in range(28) if p != q and group.labels[lookup(q,p)] == 1]
        negative = [n for n in range(28) if n != q and group.labels[lookup(q,n)] == 0]
        for p in positive:
            for n in negative:
                if values[k] > 0:
                    error = torch.maximum(scores.new_zeros(()), scores.new_tensor(float(values[k]))-(scores[lookup(q,p)]-scores[lookup(q,n)]))
                    terms.append(error**2/(28*len(positive)*len(negative)))
                k += 1
    assert k == 1016
    return sum(terms, scores.sum()*0)


class CrossfitTests(unittest.TestCase):
    def temporary(self):
        t = tempfile.TemporaryDirectory(dir=os.environ["CROSSFIT_TEST_ROOT"])
        self.addCleanup(t.cleanup)
        return Path(t.name)

    def test_floor_independent_loss_gradient_and_numeric_boundaries(self):
        group = handmade_group("math")
        s = torch.linspace(-.7, 1.3, 378, requires_grad=True)
        b = np.linspace(0, .9, 1016, dtype=np.float32)
        b[::3] = 0
        live, details = method.floor_loss(s, group, b)
        reference = literal_floor(s, group, b)
        g, = torch.autograd.grad(live, s, retain_graph=True)
        r, = torch.autograd.grad(reference, s)
        torch.testing.assert_close(live, reference, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(g, r, rtol=3e-6, atol=2e-8)
        self.assertGreater(details["violations"], 0)
        self.assertGreater(float(live.detach()), 0)
        self.assertAlmostEqual(float(method.indices(group)[2].sum()), 1., places=6)
        zero, d = method.floor_loss(s.detach(), group, np.zeros(1016, np.float32))
        self.assertEqual((float(zero), d["violations"]), (0., 0))
        # Inactive zero targets cannot constrain an incorrect/negative ordering.
        f = torch.tensor(np.asarray(group.labels)*3., dtype=torch.float32)
        born, cap, d = method.birth(group, f, f*.5)
        self.assertTrue(np.all(born == 1.5))
        self.assertTrue(np.all(cap == 3))
        self.assertEqual(d["violations"], 0)
        self.assertEqual(float(method.floor_loss(f*2, group, born)[0]), 0.)
        absent = method.birth(group, f, -f)[0]
        self.assertFalse(absent.any())
        # Finite operands can overflow their FP32 subtraction.
        overflow = torch.tensor(np.where(group.labels, np.finfo(np.float32).max, -np.finfo(np.float32).max))
        with self.assertRaises(ValueError):
            method.birth(group, overflow, overflow)
        tiny = np.full(1016, np.nextafter(np.float32(0), np.float32(1)), np.float32)
        loss, d = method.floor_loss(torch.zeros(378), group, tiny)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(d["violations"], 1016)  # Strict count survives squared-underflow.

    def test_chunked_new_term_matches_direct_autograd_and_actual_adam(self):
        current, history, c = handmade_group("current"), handmade_group("history"), run.config()
        chunked = tiny_model()
        direct = copy.deepcopy(chunked)
        oa, ob = [core.make_optimizer(m, c) for m in (chunked, direct)]
        b = np.linspace(.1, 1.4, 1016, dtype=np.float32)
        b[::4] = 0
        # Low-level actual update permits empty Adam; main counter guards are checked separately.
        row = record.update(chunked, oa, current, history, b, c, "C_cf", "ABC", 2, 1, objective_fn=method.objective)
        direct.train()
        ob.zero_grad(set_to_none=True)
        losses = []
        for role, group, target in (("current", current, None), ("history", history, b)):
            torch.manual_seed(data.seed_for(20260918, "ABC", 2, 1, role, "dropout"))
            z = record.record_vectors(direct, group, c)
            i,j = torch.triu_indices(len(z), len(z), 1)
            u,v = z[i],z[j]
            table = direct.head(torch.cat(((u-v).abs(),u*v),1)).flatten()
            seed = data.seed_for(20260918, "ABC", 2, 1, role, group.uid, "regroup")
            scores = [literal_scores(table, record.assignment(group, seed)) for seed in (None, seed)]
            truth = torch.tensor(group.labels, dtype=torch.float32)
            supervised = sum(run.parent.ranking.objectives(v,truth,.5)["total"] for v in scores)/2
            loss = supervised if target is None else .1*supervised+.5*literal_floor(scores[0], group, target)
            losses.append(float(loss.detach()))
            loss.backward()
        torch.nn.utils.clip_grad_norm_(direct.parameters(), 1., error_if_nonfinite=True)
        for a,bp in zip(chunked.parameters(), direct.parameters(), strict=True):
            torch.testing.assert_close(a.grad, bp.grad, rtol=8e-5, atol=3e-7)
        ob.param_groups[0]["lr"], ob.param_groups[1]["lr"] = run.parent.stage_lr(1), .001
        ob.step()
        for a,bp in zip(chunked.parameters(), direct.parameters(), strict=True):
            torch.testing.assert_close(a, bp, rtol=4e-5, atol=3e-7)
        self.assertAlmostEqual(row["total"], sum(losses), places=5)
        # A detached encoder or inactive/bilateral/renormalized floor would fail the reference above.
        self.assertGreater(row["history"]["record_gradient_norm"], 0)

    def test_complete_S_default_and_source_actual_lr_clock(self):
        c, group = run.config(), handmade_group("current")
        a, b = tiny_model(), tiny_model()
        oa, ob = [core.make_optimizer(m,c) for m in (a,b)]
        expected = record.update(a,oa,group,None,None,c,"S","ABC",1,1)
        actual = method.update(b,ob,group,None,None,c,"S",1,1)
        self.assertEqual(actual,expected)
        self.assertEqual(core.state_digest(a.state_dict()),core.state_digest(b.state_dict()))
        self.assertEqual(core.state_digest(oa.state_dict()),core.state_digest(ob.state_dict()))
        history = handmade_group("history")
        with diag.neutral(a):
            target = record.reference(a,history,c)
        for optimizer in (oa,ob):
            for s in optimizer.state.values():
                s["step"].fill_(288)
        expected = record.update(a,oa,group,history,target,c,"S","ABC",2,1)
        actual = method.update(b,ob,group,history,target,c,"S",2,1)
        self.assertEqual(actual,expected)
        self.assertEqual(core.state_digest(a.state_dict()),core.state_digest(b.state_dict()))
        self.assertEqual(core.state_digest(oa.state_dict()),core.state_digest(ob.state_dict()))
        model = tiny_model()
        optimizer = core.make_optimizer(model,c)
        row = method.source_update(model,optimizer,group,c,"U0",1)
        self.assertEqual(run.parent.adam_step(optimizer),1)
        self.assertEqual(row["lr_step"],2)
        self.assertEqual(row["encoder_lr"],run.parent.stage_lr(2))
        self.assertNotEqual(row["encoder_lr"],run.parent.stage_lr(1))
        with self.assertRaises(ValueError):
            method.source_update(model,optimizer,group,c,"U0",1)

    def test_feasible_permutation_memory_and_degenerate_stops(self):
        memory = shared_memory()
        groups = memory.reservoir.groups
        scores = np.asarray(groups[0].labels,np.float32)*3+np.linspace(0,.2,378,dtype=np.float32)
        full, outside = {g.uid:scores for g in groups}, {g.uid:scores*.5 for g in groups}
        memories, result = run.target_memories(memory,full,outside,source_ids(memory),"cpu")
        self.assertEqual(result["status"],"NONDEGENERATE")
        a,b = memories["C_cf"],memories["C_perm"]
        for g in groups:
            values,perm = a.tables[g.uid],b.tables[g.uid]
            capacity = method.margins(torch.from_numpy(scores),g)[0].numpy()
            for block in method.indices(g)[3]:
                np.testing.assert_array_equal(np.sort(values[block]),np.sort(perm[block]))
            self.assertTrue(np.all(perm[perm>0]<=capacity[perm>0]))
        payload = a.to_bytes()
        restored = method.Memory.from_bytes(payload)
        self.assertEqual(payload,restored.to_bytes())
        self.assertLessEqual(len(payload),1048576)
        self.assertTrue(all(v.shape==(1016,) for v in restored.tables.values()))
        a.begin_stage(2)
        restored.begin_stage(2)
        self.assertEqual([a.draw()[0].uid for _ in range(288)],[restored.draw()[0].uid for _ in range(288)])
        with self.assertRaises(ValueError):
            restored.retain(groups,2,None)
        zeros = {g.uid:np.zeros(378,np.float32) for g in groups}
        self.assertEqual(run.target_memories(memory,full,zeros,source_ids(memory),"cpu")[1]["status"],"DEGENERATE_STOP_BEFORE_B_VALID")
        same = {g.uid:np.asarray(g.labels,np.float32)*3 for g in groups}
        identical = run.target_memories(memory,same,same,source_ids(memory),"cpu")[1]
        self.assertEqual(identical["status"],"DEGENERATE_STOP_BEFORE_B_VALID")
        self.assertGreater(identical["active_total"],0)
        with self.assertRaises(ValueError):
            method.Memory.from_shared(memory,"C_cf",a.tables,{})
        restored.maps["oversize"] = "x"*1048576
        with self.assertRaises(ValueError):
            restored.to_bytes()

    def test_fold_schedule_independent_seed_and_heldout_supply(self):
        groups = [handmade_group(f"A{i:02}") for i in range(48)]
        state = diag.rng_state()
        folds = method.folds(groups)
        expected = sorted(groups,key=lambda g:(hashlib.sha256(("C_cf/fold/s0/"+g.uid).encode("utf-8")).digest(),g.uid.encode("utf-8")))
        self.assertEqual([[g.uid for g in f] for f in folds],[[g.uid for g in expected[:24]],[g.uid for g in expected[24:]]])
        for i,role in enumerate(("U0","U1")):
            schedule = method.source_schedule(folds[1-i],role)
            self.assertEqual(len(schedule),144)
            self.assertFalse({g.uid for g in schedule}&{g.uid for g in folds[i]})
            self.assertTrue(all(sum(x.uid==g.uid for x in schedule)==6 for g in folds[1-i]))
        self.assertEqual(diag.rng_state(),state)
        with self.assertRaises(ValueError):
            run.auxiliary(None,None,folds[0],[folds[0][0]],"U0",{},lambda:None)

    def test_single_shot_listener_wait_launch_delete_and_failure(self):
        root = self.temporary()
        active = root/"active.py"
        active.write_text("handwritten expendable listener")
        job = root/"job"
        calls = []
        checks = iter((False,True))
        class Child:
            pid = 42
            def poll(self):
                return 0
        def launch():
            calls.append("launch")
            job.with_suffix(".console.txt").write_text(json.dumps({"event":"updates","completed":1})+"\n")
            return Child()
        self.assertEqual(listener.once(job,active,lambda:next(checks),launch,lambda s:calls.append(s)),0)
        self.assertEqual(calls,[30,"launch"])
        self.assertFalse(active.exists())
        with self.assertRaises(FileExistsError):
            listener.once(job,active,lambda:True,launch)
        self.assertEqual(calls,[30,"launch"])
        active.write_text("failure must retain listener")
        class Failed:
            pid = 43
            def poll(self):
                return 7
        self.assertEqual(listener.once(root/"failed",active,lambda:True,lambda:Failed(),lambda s:None),7)
        self.assertTrue(active.exists())

    def test_projection_complete_call_counts_and_excess_stop(self):
        m = dict(t_A=10.,t_U=11.,t_S=20.,t_cf=21.,t_perm=22.,f=2.,h=.1,L=3.,s_F=4.,r_F=5.,s_I=6.,r_I=7.,W_F=3*2**30,W_I=1*2**30)
        p = run.projection(m)
        self.assertEqual(p["projected_total_seconds"],1.25*(288*84+2684*2+.6+9+16+35+24+28+3600))
        self.assertTrue(p["within_budget"])
        self.assertFalse(run.projection({**m,"f":100.})["within_budget"])
        self.assertFalse(run.projection({**m,"W_F":8*2**30})["within_budget"])
        with self.assertRaises(ValueError):
            run.projection({**m,"r_F":float("nan")})

    def test_native_restore_keeps_saved_adam_immutable(self):
        import step28_crossfit_retention_verify as verify
        model = tiny_model()
        optimizer = core.make_optimizer(model,run.config())
        current,history = handmade_group("restore_current"),handmade_group("restore_history")
        method.update(model,optimizer,current,None,None,run.config(),"S",1,1)
        for state in optimizer.state.values():
            state["step"].fill_(288)
        saved = verify.state(model,optimizer)
        expected = core.state_digest(saved)
        rng,modes = diag.rng_state(),[m.training for m in model.modules()]
        # Reproduce the old direct-load alias with actual installed AdamW on CPU.
        unprotected = copy.deepcopy(saved)
        optimizer.load_state_dict(unprotected["adam"])
        method.update(model,optimizer,current,history,np.ones(1016,np.float32),run.config(),"C_cf",2,1)
        original_changed = core.state_digest(unprotected) != expected
        self.assertTrue(original_changed)
        self.assertEqual({int(s["step"]) for s in unprotected["adam"]["state"].values()},{289})
        trials = []
        for _ in range(3):
            verify.restore(model,optimizer,saved,rng,modes)
            self.assertEqual(run.parent.adam_step(optimizer),288)
            row = method.update(model,optimizer,current,history,np.ones(1016,np.float32),run.config(),"C_cf",2,1)
            self.assertEqual(run.parent.adam_step(optimizer),289)
            self.assertEqual(core.state_digest(saved),expected)
            trials.append({"state":core.state_digest(verify.state(model,optimizer)),"row":row,"rng":diag.rng_state()})
        self.assertEqual(trials[0],trials[1])
        self.assertEqual(trials[1],trials[2])
        data.write_json(Path(os.environ["CROSSFIT_TEST_ROOT"])/"restore_isolation.json",
            {"torch":torch.__version__,"old_direct_load_polluted_saved_step":original_changed,
             "fixed_real_updates":3,"before_steps":[288]*3,"after_steps":[289]*3,
             "entire_saved_state_unchanged":True,"repeated_results_identical":True})

    def test_budget_counts_qualification_and_listener_outputs(self):
        root = self.temporary()
        job = root/"reports/job"
        job.mkdir(parents=True)
        qualified = root/"reports/qualification/cpu/attempt03"
        qualified.mkdir(parents=True)
        (job/"payload.bin").write_bytes(b"J"*7000)
        (qualified/"evidence.bin").write_bytes(b"Q"*4000)
        # The inherited default remains job-only for unchanged historical callers.
        old = run.legacy.Budget(job)
        self.assertNotIn(qualified/"evidence.bin",old.output_files())
        with mock.patch.object(data,"ROOT",root):
            budget = run.Budget(job)
            budget.limits = {**budget.limits,"maximum_output_bytes":10000}
            with self.assertRaisesRegex(RuntimeError,"budget exceeded"):
                budget.check()
            self.assertEqual(budget.peak_bytes,11000)
            side = job.with_suffix(".listener.json")
            side.write_bytes(b"L"*2000)
            inputs = root/"reports/seller_alias_continual/inputs"
            inputs.mkdir(parents=True)
            (inputs/"not_an_output.bin").write_bytes(b"I"*20000)
            budget.limits["maximum_output_bytes"] = 20000
            budget.last_disk = -float("inf")
            budget.check()
            actual = sum(p.stat().st_size for p in budget.output_files() if p.is_file())
            self.assertLess(actual,20000)
            with self.assertRaisesRegex(RuntimeError,"budget exceeded"):
                budget.check(reserve=8000)
            self.assertEqual(len(budget.output_files()),len(set(budget.output_files())))
            data.write_json(Path(os.environ["CROSSFIT_TEST_ROOT"])/"output_scope.json",
                {"job_bytes":7000,"qualification_bytes":4000,"limit":10000,"combined_rejected":True,
                 "listener_counted":side in budget.output_files(),"input_tree_excluded":True,"reserve_rejected":True})

    def test_native_effects_preserve_unused_parameter_checks(self):
        import step28_crossfit_retention_verify as verify
        model = tiny_model()
        model.encoder.register_parameter("unused_probe",torch.nn.Parameter(torch.ones(2)))
        optimizer = core.make_optimizer(model,run.config())
        current,history = handmade_group("effects_current"),handmade_group("effects_history")
        method.update(model,optimizer,current,None,None,run.config(),"S",1,1)
        for value in optimizer.state.values():
            value["step"].fill_(288)
        saved = verify.state(model,optimizer)
        rng,modes = diag.rng_state(),[m.training for m in model.modules()]
        method.update(model,optimizer,current,history,np.ones(1016,np.float32),run.config(),"C_cf",2,1)
        on = verify.state(model,optimizer)
        verify.restore(model,optimizer,saved,rng,modes)
        method.update(model,optimizer,current,history,np.zeros(1016,np.float32),run.config(),"C_cf",2,1)
        effects = verify.component_effects(model,on)
        self.assertEqual(effects["encoder"]["inactive_grad_parameters"],["encoder.unused_probe"])
        with self.assertRaisesRegex(ValueError,"active update effect"):
            verify.component_effects(model,verify.state(model,optimizer))
        parameter = next(p for n,p in model.named_parameters() if n.startswith("encoder.") and p.grad is not None)
        grad = parameter.grad
        parameter.grad = None
        with self.assertRaisesRegex(ValueError,"participation"):
            verify.component_effects(model,on)
        parameter.grad = grad
        with torch.no_grad():
            model.encoder.unused_probe.add_(1)
        with self.assertRaisesRegex(ValueError,"Unused parameter changed"):
            verify.component_effects(model,on)
        data.write_json(Path(os.environ["CROSSFIT_TEST_ROOT"])/"native_effects_scope.json",
            {"effects":effects,"same_update_rejected":True,"missing_active_gradient_rejected":True,
             "changed_unused_parameter_rejected":True,"real_tiny_updates":3,"formal_model_loaded":False})

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
        model_calls, update_calls, source_calls = [], [], []
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
                self.assertEqual(target.shape,(1540,) if arm=="S" else (1016,))
            advance(model,optimizer,(stage-1)*288+step)
            current_log = dict(supervised=1.,weighted=1.,mse0=0.,mse1=0.,distillation=0.)
            row = dict(current=current_log,total=1.,encoder_lr=run.parent.stage_lr(step),head_lr=.001)
            if history:
                row.update(history={**current_log,"weighted":.1},total=1.1)
            return row
        def source(model,optimizer,g,cfg,role,step,check):
            self.assertEqual(run.parent.adam_step(optimizer),step-1)
            source_calls.append((role,step,g.uid))
            advance(model,optimizer,step)
            return dict(step=step,lr_step=2*step,encoder_lr=run.parent.stage_lr(2*step),head_lr=.001,group_uid=g.uid)
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
             mock.patch.object(method,"source_update",side_effect=source), \
             mock.patch.object(record,"reference",side_effect=reference), \
             mock.patch.object(record,"score",side_effect=score), \
             mock.patch.object(diag,"metrics",side_effect=lambda gs,vs:np.zeros((len(gs),3),np.float64)), \
             mock.patch.object(run.parent.calibration,"fit",return_value={"status":"PASS_CALIBRATION_FIT","a":1.,"b":0.}), \
             mock.patch("builtins.print"):
            data.write_json(root/"access.json",dict(train=0,valid=0,heldout=0,owners=0))
            manifest,split,valid,meta = run.train(root,Budget())
            self.assertEqual(len(update_calls),1152)
            self.assertEqual(len(source_calls),288)
            self.assertEqual(len(model_calls),3)
            self.assertEqual(manifest["physical_updates"],1440)
            self.assertEqual(manifest["gradient_group_presentations"],2304)
            self.assertEqual(sum(len(v) for v in manifest["diagnostics"].values()),24)
            self.assertEqual(manifest["status"],run.COMPLETE)
            self.assertEqual(len(list((root/"run/memory").glob("*.bin"))),6)
            self.assertFalse(list((root/"run/work").glob("*.pt")))
            scores,checked_split = run.blind_gate(root/"run",manifest)
            self.assertEqual(checked_split,split)
            self.assertEqual(set(scores),set(run.POINTS))
            for collection,key in (("points","B_C_cf"),("diagnostics","B_S"),("sources","U1"),("starts","C_perm")):
                damaged = copy.deepcopy(manifest)
                del damaged[collection][key]
                with self.assertRaises(ValueError):
                    run.open_valid(root,damaged,valid,meta,c)
                self.assertEqual(data.read_json(root/"access.json")["valid"],0)
            damaged = copy.deepcopy(manifest)
            damaged["points"]["B_C_cf"]["full_restore_verified"] = False
            with self.assertRaises(ValueError):
                run.open_valid(root,damaged,valid,meta,c)
            self.assertEqual(data.read_json(root/"access.json")["valid"],0)
        # This fixture checks actual orchestration/serialization, not 1440 real BGE updates.
        evidence = {"fixture":"Actual scheduler with explicitly mocked tiny updates/forward/metrics; real checkpoint/Memory/diagnostic IO and blind gate",
                    "main_update_calls":len(update_calls),"source_update_calls":len(source_calls),"diagnostic_points":24,
                    "shared_A_restores_identical":True,"six_memories":True,"source_train_ids":source_calls}
        data.write_json(Path(os.environ["CROSSFIT_TEST_ROOT"])/"scheduler_evidence.json",evidence)

    def test_diagnostic_neutrality_and_next_candidate_update(self):
        c = run.config()
        model = tiny_model()
        optimizer = core.make_optimizer(model,c)
        current,history = handmade_group("current"),handmade_group("history")
        method.update(model,optimizer,current,None,None,c,"S",1,1)
        for s in optimizer.state.values():
            s["step"].fill_(336)  # Explicit epoch-1 continuation fixture.
        shared = shared_memory()
        memory = method.Memory.from_shared(shared,"C_cf",{g.uid:np.ones(1016,np.float32) for g in shared.reservoir.groups},source_ids(shared))
        memory.begin_stage(2)
        for _ in range(48):
            memory.draw()
        model.train()
        model.head.eval()
        state = copy.deepcopy({"model":model.state_dict(),"adam":optimizer.state_dict(),"grads":[p.grad for p in model.parameters()]})
        modes,rng,mem = [m.training for m in model.modules()],diag.rng_state(),memory.to_bytes()
        floor = np.ones(1016,np.float32)
        before_row = method.update(model,optimizer,current,history,floor,c,"C_cf",2,49)
        expected = core.state_digest({"model":model.state_dict(),"adam":optimizer.state_dict(),"grads":[p.grad for p in model.parameters()]})
        expected_rng = diag.rng_state()
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["adam"])
        for p,g in zip(model.parameters(),state["grads"]):
            p.grad = g.clone()
        for m,mode in zip(model.modules(),modes):
            m.training = mode
        diag.restore_rng(rng)
        fit = [handmade_group(f"fit{i}") for i in range(48)]
        valid = [data.Group(g.uid,g.sellers,g.items) for g in [handmade_group(f"valid{i}") for i in range(40)]]
        droot = self.temporary()
        # Full 48/6/40 call wiring above; here one actual group per role is sufficient for neutrality.
        def repeated_score(m,gs,cfg,arm,check):
            one = record.score(m,gs[:1],cfg,check)
            return np.repeat(one,len(gs),axis=0)
        with mock.patch.object(diag,"score",side_effect=repeated_score), \
             mock.patch.object(diag,"metrics",side_effect=lambda gs,vs:np.zeros((len(gs),3),np.float64)):
            diag.capture(model,optimizer,fit,memory,valid,c,"C_cf","ABC",2,1,droot,"hand",scope="AB")
        self.assertEqual(core.state_digest(state),core.state_digest({"model":model.state_dict(),"adam":optimizer.state_dict(),"grads":[p.grad for p in model.parameters()]}))
        self.assertEqual(memory.to_bytes(),mem)
        self.assertEqual(diag.rng_state(),rng)
        self.assertEqual([m.training for m in model.modules()],modes)
        after_row = method.update(model,optimizer,current,history,floor,c,"C_cf",2,49)
        self.assertEqual(after_row,before_row)
        self.assertEqual(core.state_digest({"model":model.state_dict(),"adam":optimizer.state_dict(),"grads":[p.grad for p in model.parameters()]}),expected)
        self.assertEqual(diag.rng_state(),expected_rng)

    def test_saved_AB_metrics_paired_intervals_roles_and_pooled_counts(self):
        root = self.temporary()/"evaluation"
        labelled = [handmade_group(f"{d}_valid_{i}") for d in "AB" for i in range(20)]
        split = {"development":[{"group_uid":g.uid,"domain":g.uid[0]} for g in labelled]}
        truth = np.asarray([g.labels for g in labelled],np.uint8)
        raw = truth.astype(np.float32)+np.linspace(-.4,.4,378,dtype=np.float32)
        scores = {n:{"raw":raw.copy(),"stage-cal":raw.astype(np.float64)+.2,"first-cal":raw.astype(np.float64)-.1} for n in run.POINTS}
        run.collect(root,scores,labelled,split)
        result = run.finalize(root)
        for contrast,roles in result["comparisons"].items():
            for ds in roles.values():
                for metrics in ds.values():
                    for v in metrics.values():
                        self.assertEqual(v,{"mean":0.,"conditional_95pct_interval":[0.,0.]})
        counts = result["absolute"]["A_shared"]["raw"]["fixed_half_classification"]
        self.assertEqual(counts["all"]["tp"]+counts["all"]["fn"],800)
        self.assertEqual(counts["all"]["fp"]+counts["all"]["tn"],14320)
        self.assertNotEqual(result["absolute"]["A_shared"]["stage-cal"]["macro_all"]["brier"],result["absolute"]["A_shared"]["first-cal"]["macro_all"]["brier"])
        delta = np.zeros((40,22))
        delta[:,0] = np.arange(40)*.01
        draws = np.random.Generator(np.random.PCG64(20260930)).integers(20,size=(5000,2,20))
        actual = run.summarize(delta,[np.arange(20),np.arange(20,40)],draws)
        manual = np.array([sum(delta[20+int(k),0] for k in sample)/20 for sample in draws[:,1]])
        np.testing.assert_allclose(actual["B"][run.parent.metrics.COLUMNS[0]]["conditional_95pct_interval"],np.quantile(manual,[.025,.975]),rtol=0,atol=1e-15)
