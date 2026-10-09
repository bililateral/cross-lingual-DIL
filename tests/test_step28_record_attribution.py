"""Independent loss/gradient checks and real diagnostic/epoch entry tests on handmade groups."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import step28_record_attribution_run as run
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


class AttributionTests(unittest.TestCase):
    def test_r0_and_s_losses_gradients_and_adam_match_literal_reference(self):
        c = run.config("R0")
        current,history = [handmade_group(s) for s in ("current","history")]
        for arm in ("R0", "S"):
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
                slots = [record.assignment(g)]
                if arm == "S":
                    slots.append(record.assignment(g,seed))
                scores = [literal_scores(table,a) for a in slots]
                truth = torch.tensor(g.labels,dtype=torch.float32)
                supervised = sum(record.parent.ranking.objectives(v,truth,.5)["total"] for v in scores)/len(scores)
                loss = supervised
                if t is not None:
                    errors = [((v-literal_scores(torch.from_numpy(t),a))**2).mean() for v,a in zip(scores,slots)]
                    loss = .1*supervised+.5*errors[0]
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

    def test_r0_never_reads_a1_and_s_supervision_has_an_independent_gradient(self):
        group = handmade_group("ablation")
        values = torch.linspace(-1., 2., 1540, requires_grad=True)
        original = record.assignment
        def forbid_a1(g, seed=None):
            if seed is not None:
                raise AssertionError("R0 requested regrouped supervision")
            return original(g)
        target = np.linspace(.7, -.2, 1540, dtype=np.float32)
        with mock.patch.object(record, "assignment", side_effect=forbid_a1):
            r0, log = record.objective(values, group, 17, "R0", target)
        self.assertIsNone(log["mse1"])
        s, _ = record.objective(values, group, 17, "S", target)
        gr, = torch.autograd.grad(r0, values, retain_graph=True)
        gs, = torch.autograd.grad(s, values)
        # An S implementation silently replacing augmentation with A0 must fail.
        a0, a1 = original(group), original(group, 17)
        truth = torch.tensor(group.labels, dtype=torch.float32)
        difference = .05*(record.parent.ranking.objectives(literal_scores(values, a1), truth, .5)["total"]
                          -record.parent.ranking.objectives(literal_scores(values, a0), truth, .5)["total"])
        expected, = torch.autograd.grad(difference, values)
        self.assertGreater(float(expected.abs().max()), 1e-5)
        torch.testing.assert_close(gs-gr, expected, rtol=1e-4, atol=2e-8)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(torch.zeros_like(expected), expected, rtol=1e-4, atol=2e-8)

    def test_capture_all_arms_preserves_state_and_next_update(self):
        for arm in run.ARMS:
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

    def test_real_epoch_loop_and_checkpoint_for_all_arms(self):
        for arm in run.ARMS:
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
        self.assertEqual(len(specs),27)
        self.assertEqual(len(specs)*288,7776)
        self.assertEqual(len(specs)*6,162)
        self.assertEqual(sum(288 if s==1 else 576 for _,_,s in specs.values()),12960)
        with self.assertRaises(ValueError):
            run.blind_gate(Path("absent"),{"status":"PARTIAL"})
        p,source = run.policy(),run.sources()
        job,gate_path = record.data.ROOT/"reports/job",record.data.ROOT/"gate.json"
        evidence = {"cpu":{"mode":"cpu","status":"PASS_HANDWRITTEN_ONLY","source_files":source},
            "gpu":{"mode":"gpu","status":"PASS_HANDWRITTEN_ONLY","source_files":source,
                   "native":{"projected_total_seconds":172800,"projected_peak_output_bytes":48*2**30,
                             "neutrality_all_arms":True,"architectures":{a:{} for a in run.ARMS}}}}
        gate = {"status":"APPROVED_RECORD_ATTRIBUTION_FORMAL","source_files":source,"job":"reports/job",
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

    def test_saved_metrics_keep_all_comparisons_without_inventing_a_total_pass(self):
        labelled = [handmade_group(f"valid{i:02}") for i in range(60)]
        partition = {"development":[{"group_uid":g.uid,"domain":"ABC"[i//20]} for i,g in enumerate(labelled)]}
        values = np.zeros((60,378),np.float32)
        scores = {name:{role:values.astype(np.float32 if role=="raw" else np.float64)
                       for role in run.parent.ROLES} for name in run.specifications()}
        initial = {family: values for family in ("record", "LOGIT0.1")}
        with tempfile.TemporaryDirectory(dir=os.environ["RECORD_REPLAY_TEST_ROOT"]) as tmp:
            root = Path(tmp)/"evaluation"
            run.collect(root,scores,labelled,partition,initial)
            result = run.finalize(root)
        self.assertNotIn("development_criteria_pass", result)
        self.assertEqual(set(result["comparisons"]), {"R0-LOGIT0.1", "S-R0", "S-LOGIT0.1"})
        self.assertEqual(set(result["initial_raw"]), {"record", "LOGIT0.1"})
        for comparison in result["comparisons"].values():
            self.assertNotIn("pass", comparison)
            self.assertEqual(comparison["map_interpretation"], {"O": "方向未确定", "N": "方向未确定"})
        case = lambda lo,hi: {"map": {"conditional_95pct_interval": [lo,hi]}}
        self.assertEqual(run.map_interpretation({"O": case(.01,.03), "N": case(-.03,-.01)}),
                         {"O": "正向证据", "N": "负向证据"})
        self.assertEqual(run.map_interpretation({"O": case(0.,.03), "N": case(-.03,0.)}),
                         {"O": "方向未确定", "N": "方向未确定"})

    def test_actual_dispatch_creates_nine_fresh_paths_and_never_branches(self):
        fit = [handmade_group(f"fit{d}{i:02}") for d in "ABC" for i in range(48)]
        cal = [handmade_group(f"cal{d}{i:02}") for d in "ABC" for i in range(12)]
        valid = [unlabelled(handmade_group(f"valid{i:02}")) for i in range(60)]
        partition = {"fit": [{"group_uid":g.uid,"domain":g.uid[3]} for g in fit],
                     "calibration": [{"group_uid":g.uid,"domain":g.uid[3]} for g in cal],
                     "development": [{"group_uid":g.uid,"domain":"ABC"[i//20]} for i,g in enumerate(valid)]}
        public = {"train": fit+cal, "development": valid}
        selected = {"fit": fit, "calibration": cal}
        archive = {k: run.config("R0")["models"]["split_rank"][k]
                   for k in ("file_count", "total_size_bytes", "content_sha256")}
        seen, models = [], []
        def load(c):
            model = tiny("LOGIT0.1" if c["interventions"]["split_rank"]["account_dim"] == 4096 else "R0")
            models.append(model)
            return model
        def stage(model, optimizer, current, memory, unseen, c, arm, order, index, root, budget):
            seen.append((arm, order, index, id(model), [g.uid for g in current], memory))
            self.assertIsNone(memory) if index == 1 else self.assertEqual(memory, (arm, order, index-1))
            return {}, [{}]*6
        def checkpoint(root, name, model, optimizer, memory, current, calibration, unseen, c, arm, order, index, budget):
            with torch.no_grad():
                model.head[0].weight.add_(.01)
            return {}, (arm, order, index)
        with tempfile.TemporaryDirectory(dir=os.environ["RECORD_REPLAY_TEST_ROOT"]) as tmp, \
                mock.patch.object(run.base.public, "public_inputs", return_value=(public, {}, {})), \
                mock.patch.object(run.base, "partition", return_value=(selected, partition)), \
                mock.patch.object(run.core, "model_files", return_value=archive), \
                mock.patch.object(run.previous, "parse_once", return_value=fit+cal) as labels, \
                mock.patch.object(record, "load_model", side_effect=load), \
                mock.patch.object(run, "train_stage", side_effect=stage), \
                mock.patch.object(run, "checkpoint", side_effect=checkpoint), \
                mock.patch.object(run.parent.Supply, "branch_after_first", side_effect=AssertionError("No shared trained state")):
            manifest, _, _ = run.train(Path(tmp), mock.Mock())
        labels.assert_called_once()
        self.assertEqual(len(models), 9)
        self.assertEqual([(a,o,s) for a,o,s,*_ in seen],
                         [(a,o,s) for o in run.parent.ORDERS for a in run.ARMS for s in (1,2,3)])
        self.assertEqual(len({m for _,_,_,m,_,_ in seen}), 9)
        self.assertEqual(manifest["physical_updates"], 7776)
        self.assertEqual(manifest["gradient_group_presentations"], 12960)
        for order in run.parent.ORDERS:
            self.assertEqual(manifest["starts"][order+"_R0"], manifest["starts"][order+"_S"])
        self.assertEqual(set(manifest["initial"]), {"record", "LOGIT0.1"})

    def test_actual_execute_does_not_open_valid_after_a_failed_blind_gate(self):
        with tempfile.TemporaryDirectory(dir=os.environ["RECORD_REPLAY_TEST_ROOT"]) as tmp:
            job, gate = Path(tmp)/"job", Path(tmp)/"gate.json"
            gate.write_text("{}", encoding="utf-8")
            with mock.patch.object(run, "validate_gate", return_value=run.policy()), \
                    mock.patch("step28_record_replay_verify.preflight", return_value={}), \
                    mock.patch.object(torch, "set_num_threads"), mock.patch.object(torch, "set_num_interop_threads"), \
                    mock.patch.object(torch.cuda, "set_per_process_memory_fraction"), \
                    mock.patch.object(torch.cuda, "get_device_properties", return_value=mock.Mock(total_memory=32*2**30)), \
                    mock.patch.object(run, "train", return_value=({}, {}, [])), \
                    mock.patch.object(run, "blind_gate", side_effect=ValueError("Missing final blind predictions")), \
                    mock.patch.object(run.previous, "parse_once") as labels:
                with self.assertRaisesRegex(ValueError, "Missing final blind predictions"):
                    run.execute(job, gate)
                labels.assert_not_called()
            observed = run.data.read_json(job/"execution.json")["numerics"]
            self.assertTrue(observed["deterministic_algorithms"])
            self.assertFalse(observed["cuda_matmul_allow_tf32"])
            self.assertFalse((job/"before_valid.json").exists())

if __name__ == "__main__":
    unittest.main()
