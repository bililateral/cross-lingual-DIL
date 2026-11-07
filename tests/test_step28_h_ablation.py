"""Retirable bounded Linux CPU qualification; handwritten inputs only."""
import copy
from contextlib import ExitStack
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import step28_h_ablation as m
import step28_h_ablation_run as run
import step28_h_ablation_control as control
import step28_h_ablation_evaluate as evaluate
from test_step28_record_replay import handmade_group, TinyEncoder
from test_step28_positive_ranking import independent_pool, independent_loss, independent_rank, LOOKUP


def literal_account(model, z, slots):
    rows = [torch.stack([z[i] for i in range(len(z)) if slots[i] == a]).mean(0) for a in range(28)]
    return torch.cat([model.head(torch.cat(((a-b).abs(), a*b)).unsqueeze(0)).flatten()
                      for a, b in itertools.combinations(rows, 2)])


def reference(values, group, seed, spec, role, target=None, model=None):
    natural = m.record.assignment(group)
    def pool(v, slots):
        if spec.name == "AccountPool":
            return literal_account(model, v, slots)
        if spec.name == "MeanPool":
            return independent_pool(v.double(), slots, "mean").to(v.dtype)
        return independent_pool(v, slots, "H_pos")
    v0 = pool(values, natural)
    supervised = values.new_zeros(())
    if role == "current" or spec.alpha:
        predictions = [v0] if spec.name == "NoRegroup" else [v0, pool(values, m.record.assignment(group, seed))]
        pieces = []
        for predicted in predictions:
            pos = independent_rank(predicted, group.labels)
            rank = pos
            if spec.name == "StandardRank":
                terms = []
                for q in range(28):
                    ids = [LOOKUP[tuple(sorted((q, j)))] for j in range(28) if j != q]
                    positive = [k for k in ids if group.labels[k]]
                    terms.append(torch.logsumexp(predicted[ids], 0)-predicted[positive].mean())
                rank = torch.stack(terms).mean()
            pieces.append(independent_loss(predicted, group)-pos+spec.lambda_rank*rank)
        supervised = sum(pieces)/len(pieces)
    kd = values.new_zeros(())
    if target is not None:
        t = torch.as_tensor(target, dtype=v0.dtype)
        teacher = t if spec.name == "AccountPool" else pool(t, natural)
        kd = (v0-teacher).square().mean()
    return (1 if role == "current" else spec.alpha)*supervised+spec.beta*kd


def fresh(method, c):
    torch.manual_seed(41)
    model = m.core.build_model(TinyEncoder(), 8, 5)
    m.five.checkpoint_encoder(model.encoder)
    method.bind(model, method.metadata())
    return model, m.core.make_optimizer(model, c)


class AblationTests(unittest.TestCase):
    def root(self):
        temp = tempfile.TemporaryDirectory(dir=os.environ["ABLATION_TEST_ROOT"])
        self.addCleanup(temp.cleanup)
        return Path(temp.name)

    def evidence(self, name, value):
        m.data.write_json(Path(os.environ["ABLATION_TEST_ROOT"])/(name+".json"), value)

    def test_objective_gradients_deleted_branches_and_mother(self):
        group = handmade_group("handwritten")
        errors, losses = {}, {}
        for name in m.NAMES:
            spec = m.Spec(name, "ABC", 4)
            method, c = m.Method(spec), m.config(spec)
            model, _ = fresh(method, c)
            values = (torch.sin(torch.arange(56*8).reshape(56, 8)*.31).double() if name == "AccountPool"
                      else torch.sin(torch.arange(1540).double()*.3)).requires_grad_()
            if name == "AccountPool":
                model.double()
            for role in ("current", "history") if spec.uses_history else ("current",):
                target = (np.sin(np.arange(378 if name == "AccountPool" else 1540)/7)*3).astype(np.float32) if role == "history" and spec.beta else None
                g = replace(group, labels=None) if name == "KD" and role == "history" else group
                actual, log = method.objective(values, g, 17, role, target, model)
                expected = reference(values, g, 17, spec, role, target, model)
                ga, = torch.autograd.grad(actual, values, retain_graph=True)
                ge, = torch.autograd.grad(expected, values, retain_graph=True)
                torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
                torch.testing.assert_close(ga, ge, atol=1e-11, rtol=1e-11)
                errors[name+"_"+role] = float((ga-ge).abs().max())
                losses[name+"_"+role] = float(actual.detach())
                if name == "KD" and role == "history":
                    assignment = m.record.assignment
                    def no_regroup(g, seed=None):
                        self.assertIsNone(seed)
                        return assignment(g)
                    with mock.patch.object(m.record, "assignment", side_effect=no_regroup), mock.patch.object(m.five.tuned.positive, "objectives", side_effect=AssertionError("KD history supervision forbidden")):
                        method.objective(values, g, 17, role, target, model)
                    self.assertEqual(log["views"], [])
                if name == "Full":
                    mother = m.history.Method(m.history.Spec("replication", 3, "ABC", 4, **m.SELECTED))
                    old, _ = mother.tuned.objective(values, group, 17, "H_pos", target)
                    torch.testing.assert_close(actual, old, atol=0, rtol=0)
        self.assertNotEqual(losses["Full_current"], losses["NoRegroup_current"])
        self.assertNotEqual(losses["Full_current"], losses["MeanPool_current"])
        self.assertNotEqual(losses["Full_current"], losses["StandardRank_current"])
        self.assertLess(losses["ER_history"], losses["ER_current"]*.1)
        # Nonlinear matching and mean cannot be interchanged.
        method = m.Method(m.Spec("AccountPool", "ABC", 4)); model, _ = fresh(method, m.config(method.spec))
        z = torch.sin(torch.arange(56*8).reshape(56, 8)*.31)
        difference = (m.account_scores(model, z, m.record.assignment(group))-m.mean_scores(m.record.pair_table(model, z), m.record.assignment(group))).abs().max()
        self.assertGreater(float(difference), 1e-4)
        self.evidence("objectives", dict(independent_gradient_errors=errors, losses=losses,
            mother_exact=True, kd_history_labels_none=True, forbidden_history_operations_not_called=True,
            account_vs_record_mean_difference=float(difference.detach())))

    def test_actual_updates_optimizer_restore_and_own_memory(self):
        evidence = {}
        current, history = handmade_group("current"), handmade_group("history")
        for name in m.NAMES:
            spec = m.Spec(name, "ABC", 4); method, c = m.Method(spec), m.config(spec)
            model, optimizer = fresh(method, c)
            for p in model.parameters():
                optimizer.state[p] = dict(step=torch.tensor(624.), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
            direct = copy.deepcopy(model); direct.encoder._five_outer_checkpoint = False
            other = m.core.make_optimizer(direct, c); other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            before = copy.deepcopy(model.state_dict())
            target = (np.sin(np.arange(378 if name == "AccountPool" else 1540)/7)*3).astype(np.float32) if spec.beta else None
            root = self.root(); path = root/"full.pt"
            meta = {**method.metadata(), "rng": m.diag.rng_state(), "modes": [v.training for v in model.modules()]}
            rec = m.core.save_state(path, model, optimizer, meta)
            actual = method.update(model, optimizer, current, history if spec.uses_history else None, target, c, 3, 49)
            direct.train(); other.zero_grad(set_to_none=True)
            for role, group, teacher in (("current", current, None), ("history", history, target)):
                if role == "history" and not spec.uses_history: continue
                torch.manual_seed(m.data.seed_for(spec.roots["schedule"], spec.order, 3, 49, role, "dropout"))
                z = m.record.record_vectors(direct, group, c)
                i,j = torch.triu_indices(len(z),len(z),1)
                values = z if name == "AccountPool" else direct.head(torch.cat(((z[i]-z[j]).abs(), z[i]*z[j]), 1)).flatten()
                seed = m.data.seed_for(spec.roots["schedule"], spec.order, 3, 49, role, group.uid, "regroup")
                reference(values, group, seed, spec, role, teacher, direct).backward()
            torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.)
            for pg, lr in zip(other.param_groups, spec.rates(49), strict=True): pg["lr"] = lr
            other.step()
            for p, q in zip(model.parameters(), direct.parameters(), strict=True):
                torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(optimizer.state[p][key], other.state[q][key], atol=2e-6, rtol=2e-5)
            for prefix in ("encoder.", "head."):
                self.assertTrue(any(not torch.equal(p, before[k]) for k,p in model.state_dict().items() if k.startswith(prefix)))
            digest = m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
            method.restore(path, rec, model, optimizer)
            repeated = method.update(model, optimizer, current, history if spec.uses_history else None, target, c, 3, 49)
            self.assertEqual(actual, repeated)
            self.assertEqual(digest, m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict())))
            memory = method.new_memory(dict(a=1., b=0.))
            if name == "SEQ":
                self.assertIsNone(memory)
            else:
                fit = [handmade_group(f"A_{i}") for i in range(48)]
                with m.diag.neutral(model): method.retain(memory, model, fit, c, 1, {}, lambda: None)
                if spec.beta:
                    for g in memory.reservoir.groups:
                        expected = method.score(model, [g], c)[0] if name == "AccountPool" else m.record.reference(model, g, c)
                        np.testing.assert_array_equal(memory.tables[g.uid], expected)
                payload = memory.to_bytes(); restored = method.memory_from(payload)
                self.assertEqual(payload, restored.to_bytes())
                memory.begin_stage(2); restored.begin_stage(2)
                for _ in range(288):
                    a, ta = memory.draw(); b, tb = restored.draw()
                    self.assertEqual(a.uid, b.uid); np.testing.assert_array_equal(ta, tb)
                old = {k: v.tobytes() for k,v in memory.tables.items()}
                with m.diag.neutral(model): method.retain(memory, model, [handmade_group(f"B_{i}") for i in range(48)], c, 2, {}, lambda: None)
                for uid in old.keys() & memory.tables.keys(): self.assertEqual(old[uid], memory.tables[uid].tobytes())
                self.assertEqual(len(memory.reservoir.groups), 6)
                self.assertLessEqual(len(memory.to_bytes()), 1048576)
                with self.assertRaises(ValueError): m.Memory.from_bytes(payload, m.Spec("SEQ", "ABC", 4))
            evidence[name] = dict(actual_updates=2, independent_updates=1, exact_resume=True, memory=None if memory is None else memory.summary())
        self.evidence("updates_memory", evidence)

    def test_closed_gate_ram_credit_wait_and_failed_slot(self):
        root = self.root(); closed = root/"closed.json"
        m.data.write_json(closed, dict(experiment="H_HISTORY_ABLATION", approved=False))
        with mock.patch.object(run, "sources", side_effect=AssertionError("No downstream reads")):
            for action in run.ACTIONS:
                with self.assertRaisesRegex(ValueError, "closed"): run.gate(closed, action)
        self.assertEqual(len(m.specifications()), 63)
        self.assertEqual(sum(864 if s.name == "SEQ" else 1440 for s in m.specifications()), 85536)
        allowed = dict(workers=3, sources=[], campaign=str(root), workspace=str(root), limits=run.LIMITS)
        with mock.patch.object(run.queue, "resources", return_value=dict(ready=True, free_ram_bytes=71*2**30)):
            self.assertFalse(control.admission(allowed, dict(free=32*2**30),32*2**30,0)["ready"])
        with mock.patch.object(run.queue, "resources", return_value=dict(ready=True, free_ram_bytes=72*2**30)):
            self.assertTrue(control.admission(allowed, dict(free=32*2**30),32*2**30,2)["ready"])
            self.assertFalse(control.admission(allowed, dict(free=32*2**30),16*2**30,1)["ready"])
        with mock.patch.object(control.inherited.time, "monotonic", side_effect=[100.,101.,101.,111.,112.]), mock.patch.object(control.time, "sleep"):
            ledger = control.Ledger(root, dict(training=0.,auxiliary=0.),dict(training=100.,auxiliary=100.))
            ledger.pause(10, active=False)
            self.assertEqual(ledger.used["auxiliary"],32.)
        limits = {**run.LIMITS,"rss_bytes":4*2**30,"disk_bytes":2**30,"free_reserve_bytes":0}
        allowed.update(limits=limits)
        ledger = control.Ledger(root,dict(training=0.,auxiliary=0.),dict(training=100.,auxiliary=100.))
        rows = [dict(id=f"r{i}",bucket="training",seconds=10.,shutdown_reserve=.1,gpu=False) for i in range(7)]
        def command(row,job,deadline):
            return [sys.executable,"-c","import time;time.sleep(.2);raise SystemExit("+str(int(row["id"]=="r0"))+")"]
        with self.assertRaisesRegex(RuntimeError,"Incomplete batch"):
            control.inherited.dispatch(root,rows,command,lambda r,j:None,allowed,ledger,poll_seconds=.02,admit=control.admission)
        receipt=m.data.read_json(root/"dispatch.json")
        self.assertEqual(len(receipt["completed"]),7)
        self.assertEqual(receipt["retired_slots"],[0])
        self.assertEqual([r["id"] for r in receipt["completed"] if r["slot"]==0],["r0"])
        self.assertEqual(sum(r["success"] and r["released"] for r in receipt["completed"]),6)
        self.evidence("control",dict(gate_closed=True,ram_boundary_tested=True,slot_credit_tested=True,
            inactive_sleep_excluded=True,actual_cpu_process_dispatch=receipt,gpu_calls=0))

    def test_actual_domain_fields_lower_loss_orientation_and_na(self):
        columns = list(m.parent.metrics.COLUMNS)
        evidence = {}
        for order in m.ORDERS:
            spec = m.Spec("Full", order, 4)
            cells = {f"{stage}_{position}": np.full((40, 22), stage*10.+position) for stage,position in evaluate.CELLS}
            cells["3_1"][0, columns.index("precision")] = np.nan
            fields = {f: evaluate.field_values(spec, cells, f) for f in evaluate.FIELDS}
            points = {f: a.mean(1).sum(0) for f,a in fields.items()}
            self.assertEqual(points["O"][columns.index("map")],31.5)
            self.assertEqual(points["F"][columns.index("map")],-15.)
            self.assertEqual(points["F"][columns.index("brier")],15.)
            self.assertEqual(points["F_first"][columns.index("log_loss")],20.)
            self.assertEqual(points["N"][columns.index("map")],27.5)
            self.assertTrue(np.isnan(points["O"][columns.index("precision")]))
            self.assertEqual(fields["O"]["ABC".index(order[0]),1,columns.index("map")],15.5)
            evidence[order] = {f: evaluate.prior.nullable(v) for f,v in points.items()}
        self.evidence("field_delta",dict(actual_domain_mapping=True,lower_loss_forgetting_reversed=True,
            undefined_propagates=True,independent_handwritten_points=evidence))

    def test_ram_gate_is_in_actual_dispatch(self):
        root = self.root()
        limits = {**run.LIMITS, "rss_bytes":4*2**30, "disk_bytes":2**30, "free_reserve_bytes":0}
        allowed = dict(workers=3,sources=[],campaign=str(root),workspace=str(root),limits=limits)
        ledger = control.Ledger(root,dict(training=0.,auxiliary=0.),dict(training=100.,auxiliary=100.))
        observations = []
        real_resources = run.queue.resources
        def resources(*args,**kwargs):
            value = real_resources(*args,**kwargs)
            value["free_ram_bytes"] = (71 if len(observations)<2 else 72)*2**30
            self.assertFalse((root/"handwritten_child").exists())
            observations.append(value["free_ram_bytes"])
            return value
        class HandwrittenGpuObservation:
            def snapshot(self): return dict(free=32*2**30,processes={})
        rows=[dict(id="handwritten_child",bucket="training",seconds=10.,shutdown_reserve=.1,gpu=True)]
        with mock.patch.object(run.queue,"resources",side_effect=resources):
            result=control.inherited.dispatch(root,rows,
                lambda row,job,deadline:[sys.executable,"-c","import time;time.sleep(.05)"],
                lambda row,job:None,allowed,ledger,monitor=HandwrittenGpuObservation(),
                poll_seconds=.01,admit=control.admission)
        self.assertEqual(observations,[71*2**30,71*2**30,72*2**30])
        self.assertEqual(result["status"],"ALL_COMPLETE_RELEASED")
        # Check that the real whole-campaign entry supplies this same admission callback.
        with tempfile.TemporaryDirectory(dir=m.data.ROOT) as temporary:
            campaign=Path(temporary)/"closed_fixture"
            gate=Path(temporary)/"handwritten_gate.json";m.data.write_json(gate,{})
            handwritten={**allowed,"campaign":str(campaign),"actions":list(run.ACTIONS),
                "prior_used_seconds":dict(training=0.,confirmation=0.,auxiliary=0.)}
            monitor=mock.Mock()
            def dispatch(*args,**kwargs):
                self.assertIs(kwargs["admit"],control.admission)
                self.assertEqual(len(args[1]),63)
                raise RuntimeError("HANDWRITTEN_STOP_BEFORE_FORMAL_WORK")
            with mock.patch.object(run,"gate",return_value=handwritten),mock.patch.object(control,"predecessors"),mock.patch.object(run,"references"),mock.patch.object(run.queue,"GpuMemory",return_value=monitor),mock.patch.object(control.inherited,"dispatch",side_effect=dispatch):
                with self.assertRaisesRegex(RuntimeError,"HANDWRITTEN_STOP_BEFORE_FORMAL_WORK"):
                    control.execute(gate)
            timing=m.data.read_json(campaign/"execution_time.json")
            self.assertGreaterEqual(timing["elapsed_calendar_seconds"],0.)
            self.assertGreaterEqual(timing["finished_epoch"],timing["started_epoch"])
        self.evidence("dispatch_admission_delta",dict(actual_cpu_child_processes=1,
            observed_ram_bytes=observations,withheld_until_72GiB=True,whole_entry_injects_same_callback=True,
            whole_entry_fixed_list=63,production_sources_unchanged=True,real_gpu_calls=0,
            substitutes=["handwritten GPU resource observation","available-RAM observation","formal entry authorization/reference and dispatch capture"]))

    def test_observed_gpu_cost_record_without_gpu(self):
        root=self.root()
        inner=mock.Mock()
        samples=[dict(free=16*2**30,processes={123:size*2**30,999:40*2**30}) for size in (4,3,8)]
        inner.snapshot.side_effect=samples
        child=mock.Mock(pid=123);child.create_time.return_value=11.
        with mock.patch.object(control.inherited.psutil.Process,"children",return_value=[child]):
            tracker=control.ObservedGpuMemory(inner,root)
            for sample in samples:self.assertEqual(tracker.snapshot(),sample)
            tracker.close()
        saved=m.data.read_json(root/"gpu_observed_peaks.json")
        self.assertFalse(saved["continuous_lifetime_maximum"])
        self.assertEqual(saved["by_process"],{"123@11.0":dict(pid=123,created=11.,observed_peak_gpu_bytes=8*2**30)})
        inner.close.assert_called_once()
        self.evidence("cost_record_delta",dict(real_gpu_calls=0,unchanged_admission_observations=True,
            foreign_gpu_process_excluded=True,lower_sample_does_not_reduce_peak=True,persisted=saved))


    def test_actual_confirmation_restore_score_then_single_label_opening(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory(dir=os.environ["ABLATION_TEST_ROOT"]) as temp:
            campaign = Path(temp)
            generation, job = campaign/"generation", campaign/"confirmation"
            generation.mkdir()
            job.mkdir()
            groups = [handmade_group(f"handwritten_{d}_{i}") for d in "ABC" for i in range(40)]
            domains = [d for d in "ABC" for _ in range(40)]
            # The real align_labels parser consumes only these handwritten pair rows.
            with (generation/"pairs.jsonl").open("w") as stream:
                import itertools
                for group in groups:
                    rows = [dict(group_uid=group.uid, seller_uid_left=group.sellers[i], seller_uid_right=group.sellers[j], label=group.labels[k])
                        for k, (i, j) in enumerate(itertools.combinations(range(28), 2))]
                    stream.write(json.dumps(rows)+"\n")
            m.data.write_json(generation/"result.json", dict(status="GENERATED_VALIDATED_OWNERS_SEALED", namespace=evaluate.NAMESPACE, root_seed=evaluate.CONFIRM_SEED,
                sources=[], files={"pairs.jsonl": m.data.record(generation/"pairs.jsonl", generation)}))
            def fresh(method, c):
                torch.manual_seed(method.spec.seed)
                model = m.core.build_model(TinyEncoder(), 8, 5)
                m.five.checkpoint_encoder(model.encoder)
                method.bind(model, method.metadata())
                return model, m.core.make_optimizer(model, c)
            records = []
            specs = [m.Spec("Full", order, seed) for order in m.ORDERS for seed in m.SEEDS]
            for name in evaluate.NAMES:
                for spec in specs:
                    spec = m.history.Spec("replication", 3, spec.order, spec.seed, **m.SELECTED) if name == "Full" else m.Spec(name, spec.order, spec.seed)
                    method = m.history.Method(spec) if name == "Full" else m.Method(spec)
                    model, optimizer = fresh(method, m.config(spec))
                    root = campaign/f"{spec.order}_{spec.seed}_{name}"
                    root.mkdir()
                    points = {}
                    for stage in (1, 2, 3):
                        with torch.no_grad():
                            for p in model.parameters():
                                p.add_(.0001)
                        maps = dict(first=dict(a=1.1, b=.01), stage=dict(a=1.+stage*.1, b=stage*.01))
                        meta = {**method.metadata(), "stage": stage, "maps": maps, "rng": m.diag.rng_state(),
                            "modes": [v.training for v in model.modules()]}
                        path = root/f"stage{stage}.pt"
                        receipt = m.core.save_state(path, model, None, meta)
                        points[str(stage)] = dict(model={**receipt, "path": path.name}, maps=maps)
                    records.append(dict(name=name, spec=spec, root=root, record=dict(points=points)))
                    del model, optimizer
            accesses = []
            real_access = evaluate.prior.access
            def access(root, role, keys):
                if role == "ablation_confirmation_joint":
                    blind = m.data.read_json(job/"all_blind_complete.json")
                    evaluate.validate_cells(blind, records)
                    self.assertEqual(blind["forwards"], 17280)
                accesses.append(role)
                real_access(root, role, keys)
            budget = type("Budget", (), {"check": lambda self: None})()
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(m.five.Method, "fresh", fresh))
                from dataclasses import replace
                stack.enter_context(mock.patch.object(evaluate.prior, "confirmation_public", return_value=([replace(g, labels=None) for g in groups], domains)))
                stack.enter_context(mock.patch.object(evaluate.prior, "access", side_effect=access))
                stack.enter_context(mock.patch.object(torch, "set_num_interop_threads"))
                stack.enter_context(mock.patch.object(torch.cuda, "set_per_process_memory_fraction"))
                stack.enter_context(mock.patch.object(torch.cuda, "get_device_properties", return_value=type("Device", (), {"total_memory": 32*2**30})()))
                stack.enter_context(mock.patch.object(torch.cuda, "empty_cache"))
                evaluate.confirmation(campaign, job, records, budget, dict(sources=[], limits=run.LIMITS))
            result = m.data.read_json(job/"result.json")
            self.assertEqual(result["status"], "ALL_ABLATION_CONFIRMATION_METRICS_COMPLETE")
            self.assertEqual(accesses, ["ablation_blind_inference", "ablation_confirmation_joint"])
            self.assertEqual(len(result["runs"]), 72)
            summary = evaluate.summarize(job, result, budget)
            self.assertFalse(summary["primary_pass"])
            self.assertEqual(len(summary["contrasts"]), 9)
            self.assertEqual(len(summary["ranking_query_subgroups"]), 324)
            map_index = list(m.parent.metrics.COLUMNS).index("map")
            for name in m.NAMES:
                fields = summary["absolute"][name]["raw"]
                self.assertAlmostEqual(fields["F"][map_index], fields["A_acq"][map_index]-fields["O"][map_index])
            broken = copy.deepcopy(result)
            broken["runs"].pop(next(iter(broken["runs"])))
            with self.assertRaises(ValueError): evaluate.validate_cells(broken, records)
            self.evidence("summary", dict(bootstrap=summary["bootstrap"], primary=summary["primary_checks"],
                contrasts=list(summary["contrasts"]), query_subgroup_cells=len(summary["ranking_query_subgroups"]),
                metric_columns=summary["metric_columns"], fixed_identity_missing_arm_rejected=True))
        self.evidence("actual_confirmation", dict(actual_tiny_model_restores=216, actual_handwritten_group_forwards=17280,
            actual_metric_matrices=1296, label_openings_after_all_blind=1, native_model=False, gpu=False,
            formal_data=False, generator_calls=0, substitutes=["TinyEncoder fresh", "handwritten public groups", "CUDA setup only"]))


if __name__ == "__main__":
    if sys.platform != "linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Only bounded Linux handwritten CPU qualification")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    unittest.main()
