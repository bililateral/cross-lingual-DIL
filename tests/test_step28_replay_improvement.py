"""Independent loss/gradient checks and real diagnostic/epoch entry tests on handmade groups."""
from __future__ import annotations

import copy
import itertools
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import step28_replay_improvement_run as run
import step28_replay_diagnostics as diag
import step28_record_replay as record
from test_step28_record_replay import handmade_group, tiny_model, TinyEncoder, literal_scores


def tiny(arm):
    torch.manual_seed(41)
    return record.core.build_model(TinyEncoder(),16 if arm=="LOGIT0.1" else 8,5)


def unlabelled(group):
    return record.data.Group(group.uid,group.sellers,group.items)


def set_step(optimizer,step):
    # Explicit synthetic phase fixture; this is not evidence of earlier updates.
    for state in optimizer.state.values():
        state["step"].fill_(step)


class ImprovementTests(unittest.TestCase):
    def test_four_actual_losses_and_vjp_match_independent_full_graph(self):
        c = run.config("C")
        current,history = [handmade_group(s) for s in ("current","history")]
        for arm,(w0,w1) in {"C":(.25,.25),"S":(.5,0),"C_plus":(.5,.25),"S_strong":(.75,0)}.items():
            model = tiny_model()
            target = record.reference(model,history,c)
            with torch.no_grad():
                model.head[0].weight.add_(.08)
            direct = copy.deepcopy(model)
            oa,ob = [record.core.make_optimizer(m,c) for m in (model,direct)]
            row = record.update(model,oa,current,history,target,c,arm,"ABC",2,1)
            direct.train()
            ob.zero_grad(set_to_none=True)
            literal_total = 0.
            for role,g,t in (("current",current,None),("history",history,target)):
                torch.manual_seed(record.data.seed_for(20260918,"ABC",2,1,role,"dropout"))
                z = record.record_vectors(direct,g,c)
                i,j = torch.triu_indices(len(z),len(z),1)
                table = direct.head(torch.cat(((z[i]-z[j]).abs(),z[i]*z[j]),1)).flatten()
                seed = record.data.seed_for(20260918,"ABC",2,1,role,g.uid,"regroup")
                slots = [record.assignment(g),record.assignment(g,seed)]
                scores = [literal_scores(table,a) for a in slots]
                truth = torch.tensor(g.labels,dtype=torch.float32)
                supervised = sum(record.parent.ranking.objectives(v,truth,.5)["total"] for v in scores)/2
                loss = supervised
                if t is not None:
                    errors = [((v-literal_scores(torch.from_numpy(t),a))**2).mean() for v,a in zip(scores,slots)]
                    loss = .1*supervised+w0*errors[0]+w1*errors[1]
                literal_total += float(loss.detach())
                loss.backward()
            self.assertAlmostEqual(row["total"],literal_total,delta=2e-6)
            torch.nn.utils.clip_grad_norm_(direct.parameters(),1.)
            for a,b in zip(model.parameters(),direct.parameters()):
                torch.testing.assert_close(a.grad,b.grad,rtol=5e-5,atol=2e-7)
            ob.param_groups[0]["lr"],ob.param_groups[1]["lr"] = record.parent.stage_lr(1),.001
            ob.step()
            for a,b in zip(model.parameters(),direct.parameters()):
                torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-7)

    def test_new_regroup_term_survives_original_view_nullspace(self):
        group = handmade_group("nullspace")
        a0,a1 = record.assignment(group),record.assignment(group,17)
        k = next(i for i,(l,r) in enumerate(itertools.combinations(range(56),2)) if a0[l]==a0[r] and a1[l]!=a1[r])
        target = np.zeros(1540,np.float32)
        target[k] = 3
        values = torch.linspace(-.2,.3,1540,requires_grad=True)
        gradients = {}
        for arm in run.RECORD_ARMS:
            loss,_ = record.objective(values,group,17,arm,target)
            control,_ = record.objective(values,group,17,arm,np.zeros_like(target))
            gradients[arm], = torch.autograd.grad(loss-control,values)
        self.assertLess(float(gradients["S_strong"].abs().max()),1e-8)
        self.assertAlmostEqual(float(gradients["C_plus"][k]),-3/(32*378),delta=1e-8)
        torch.testing.assert_close(gradients["C"],gradients["C_plus"],rtol=1e-5,atol=2e-8)

    def test_capture_both_architectures_preserves_state_and_next_update(self):
        for arm in ("C_plus","LOGIT0.1"):
            c,model = run.config(arm),tiny(arm)
            optimizer = record.core.make_optimizer(model,c)
            current = [handmade_group(f"current{i:02}") for i in range(48)]
            history = [handmade_group(f"history{i:02}") for i in range(48)]
            valid = [unlabelled(handmade_group(f"valid{i:02}")) for i in range(60)]
            run.update(model,optimizer,current[0],None,None,c,arm,"ABC",1,1,lambda:None)
            memory = (record.parent.Memory("ABC",20260930,True,{"a":1.,"b":0.}) if arm=="LOGIT0.1" else record.Memory("ABC"))
            with diag.neutral(model):
                callback = (lambda gs:diag.score(model,gs,c,arm,lambda:None)) if arm=="LOGIT0.1" else (lambda g:record.reference(model,g,c))
                memory.retain(history,1,callback)
            memory.begin_stage(2)
            set_step(optimizer,336)
            model.train()
            model.head.eval()  # Mixed child modes must restore individually.
            before = (record.core.state_digest(model.state_dict()),record.core.state_digest(optimizer.state_dict()),
                      [p.grad.clone() for p in model.parameters()],diag.rng_state(),memory.to_bytes(),[m.training for m in model.modules()])
            clone,other,m2 = copy.deepcopy(model),None,run.memory_from(memory.to_bytes(),arm)
            other = record.core.make_optimizer(clone,c)
            other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            with tempfile.TemporaryDirectory(dir=os.environ["RECORD_REPLAY_TEST_ROOT"]) as tmp:
                root = Path(tmp)
                rec = diag.capture(model,optimizer,current,memory,valid,c,arm,"ABC",2,1,root,"toy")
                entry = record.data.read_json(root/rec["path"])
                self.assertFalse(entry["valid_labels_used"])
                self.assertEqual(len(entry["cache_ids"]),6)
                self.assertEqual(entry["cache_domains"],["A"]*6)
                self.assertEqual(np.load(root/entry["fit"]["path"]).shape,(48,3))
            self.assertEqual(before[0],record.core.state_digest(model.state_dict()))
            self.assertEqual(before[1],record.core.state_digest(optimizer.state_dict()))
            for g,p in zip(before[2],model.parameters()):
                self.assertTrue(torch.equal(g,p.grad))
            self.assertEqual(before[3],diag.rng_state())
            self.assertEqual(before[4],memory.to_bytes())
            self.assertEqual(before[5],[m.training for m in model.modules()])
            h,t = memory.draw()
            h2,t2 = m2.draw()
            self.assertEqual(h.uid,h2.uid)
            run.update(model,optimizer,current[1],h,t,c,arm,"ABC",2,49,lambda:None)
            run.update(clone,other,current[1],h2,t2,c,arm,"ABC",2,49,lambda:None)
            self.assertEqual(record.core.state_digest(model.state_dict()),record.core.state_digest(clone.state_dict()))
            self.assertEqual(record.core.state_digest(optimizer.state_dict()),record.core.state_digest(other.state_dict()))

    def test_real_epoch_loop_and_checkpoint_for_both_architectures(self):
        for arm in ("C_plus","LOGIT0.1"):
            c,model = run.config(arm),tiny(arm)
            optimizer = record.core.make_optimizer(model,c)
            current = [handmade_group(f"fit{i:02}",2) for i in range(48)]
            cal = [handmade_group(f"cal{i:02}",2) for i in range(12)]
            valid = [unlabelled(handmade_group(f"valid{i:02}",2)) for i in range(60)]
            budget = mock.Mock(snapshot=mock.Mock(return_value={}))
            with tempfile.TemporaryDirectory(dir=os.environ["RECORD_REPLAY_TEST_ROOT"]) as tmp:
                root = Path(tmp)
                for sub in ("models","work","memory","points","scores","maps","updates","diagnostics"):
                    (root/sub).mkdir()
                log,diags = run.train_stage(model,optimizer,current,None,valid,c,arm,"ABC",1,root,budget)
                self.assertEqual(len(diags),6)
                point,memory = run.checkpoint(root,run.point_name("ABC",arm,1),model,optimizer,None,current,cal,valid,c,arm,"ABC",1,budget)
                self.assertTrue(point["full_restore_verified"])
                self.assertEqual(memory.reservoir.seen,48)
                partition = {"fit":[{"group_uid":g.uid,"domain":"A"} for g in current],
                             "development":[{"group_uid":g.uid,"domain":"ABC"[i//20]} for i,g in enumerate(valid)]}
                run.check_diagnostics(root/"diagnostics",diags,record.data.read_json(root/log["path"]),point,partition)
                with self.assertRaises(ValueError):
                    run.check_diagnostics(root/"diagnostics",diags[:-1],{},point,partition)
                # Full actual trajectory outputs enter the end-only diagnostic evaluator.
                labelled = [handmade_group(g.uid,2) for g in valid]
                report = diag.evaluate(root/"diagnostics",{"diagnostics":{"point":diags}},labelled,partition,root/"evaluated")
                self.assertEqual(report["points"]["point"]["cache"]["status"],"NOT_APPLICABLE_FIRST_STAGE")

    def test_metrics_and_independent_group_gap_with_fixed_epoch_changes(self):
        g = handmade_group("metrics")
        y = np.asarray(g.labels)
        values = diag.metrics([g],(2*y-1).reshape(1,-1).astype(np.float32))
        np.testing.assert_allclose(values[0],[1.,1.,np.log1p(np.exp(-1))],rtol=1e-12,atol=1e-12)
        train = np.tile(np.array([.5,.5,.5]),(6,48,1))
        valid = np.tile(np.array([.5,.5,.5]),(6,20,1))
        train[:, :, :] += np.arange(6)[:,None,None]*np.array([.04,.04,-.04])
        valid[:, :, :] += np.arange(6)[:,None,None]*np.array([-.02,-.02,.02])
        result = diag.curve_comparison(train,valid,7)
        self.assertEqual(list(result["interpretation"].values()),["有相应迹象"]*3)
        np.testing.assert_allclose(result["epoch1_to_6_gap_widening"]["mean"],[.3]*3)
        self.assertEqual(diag.curve_comparison(train[:,:0],valid,7)["status"],"NOT_APPLICABLE_NO_SURVIVING_GROUPS")

    def test_scope_and_blind_qualification_reject_missing_or_wrong_native(self):
        specs = run.specifications()
        self.assertEqual(len(specs),36)
        self.assertEqual(len(specs)*288,10368)
        self.assertEqual(len(specs)*6,216)
        self.assertEqual(sum(288 if s==1 else 576 for _,_,s in specs.values()),19008)
        with self.assertRaises(ValueError):
            run.blind_gate(Path("absent"),{"status":"PARTIAL"})
        p,source = run.policy(),run.sources()
        job,gate_path = record.data.ROOT/"reports/job",record.data.ROOT/"gate.json"
        evidence = {"cpu":{"mode":"cpu","status":"PASS_HANDWRITTEN_ONLY","source_files":source},
            "gpu":{"mode":"gpu","status":"PASS_HANDWRITTEN_ONLY","source_files":source,
                   "native":{"projected_total_seconds":172800,"projected_peak_output_bytes":64*2**30,
                             "neutrality_both_architectures":True,"architectures":{"record":{},"LOGIT0.1":{}}}}}
        gate = {"status":"APPROVED_REPLAY_IMPROVEMENT_FORMAL","source_files":source,"job":"reports/job",
                "runtime":p["runtime"],"supervision":p["supervision"],"review_disposition":"NO_OPEN_BLOCKERS",
                "cpu":{"path":"cpu.json"},"gpu":{"path":"gpu.json"}}
        for seconds,ok in ((172800,True),(172801,False),(float("nan"),False)):
            evidence["gpu"]["native"]["projected_total_seconds"] = seconds
            lookup = {run.POLICY:p,gate_path:gate,**{record.data.ROOT/(m+".json"):e for m,e in evidence.items()}}
            with mock.patch.object(record.data,"read_json",side_effect=lambda path:lookup[path]),mock.patch.object(record.data,"verify",side_effect=lambda path,rec:path):
                if ok:
                    self.assertEqual(run.validate_gate(job,gate_path),p)
                else:
                    with self.assertRaises(ValueError):
                        run.validate_gate(job,gate_path)

    def test_saved_collection_routes_all_23_checks_without_truthy_container_bug(self):
        labelled = [handmade_group(f"valid{i:02}") for i in range(60)]
        partition = {"development":[{"group_uid":g.uid,"domain":"ABC"[i//20]} for i,g in enumerate(labelled)]}
        values = np.zeros((60,378),np.float32)
        scores = {name:{role:values.astype(np.float32 if role=="raw" else np.float64) for role in run.parent.ROLES} for name in run.specifications()}
        with tempfile.TemporaryDirectory(dir=os.environ["RECORD_REPLAY_TEST_ROOT"]) as tmp:
            root = Path(tmp)/"evaluation"
            run.collect(root,scores,labelled,partition)
            # Historical replay is a separate input dependency, absent for handmade data.
            with mock.patch.object(run,"historical_replay",return_value={}) as historical:
                result = run.finalize(root)
            historical.assert_called_once()
        self.assertFalse(result["development_criteria_pass"])
        self.assertEqual(len(result["comparisons"]),7)
        for comparison in result["comparisons"].values():
            self.assertEqual(len(comparison["checks"]),23)
            self.assertEqual(sum(comparison["checks"].values()),20)
            self.assertFalse(comparison["pass"])


if __name__ == "__main__":
    unittest.main()
