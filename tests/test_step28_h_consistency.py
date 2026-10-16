"""Independent handwritten H_cons references; Linux qualification only."""
from __future__ import annotations

import copy
import hashlib
import inspect
import itertools
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import step28_h_consistency as method
import step28_h_consistency_run as run
import step28_g_half_verify as qualification
from test_step28_g_half import literal_pool, literal_loss, GHalfTests
from test_step28_record_replay import handmade_group, tiny_model

record, core, data, diag = run.record, run.core, run.data, run.diag
PAIRS = list(itertools.combinations(range(28), 2))
PAIR_INDEX = {pair: i for i, pair in enumerate(PAIRS)}
QUERY_INDEX = np.array([[PAIR_INDEX[tuple(sorted((q, j)))] for j in range(28) if j != q] for q in range(28)])


def js_reference(z0, z1):
    """Scalar log-normalizers and analytic derivatives, independent of torch JS."""
    values, gradients = [], [np.zeros((28, 27)), np.zeros((28, 27))]
    for q in range(28):
        logs, probabilities = [], []
        for z in (z0[q], z1[q]):
            peak = max(z)
            normalizer = peak+math.log(math.fsum(math.exp(float(t)-peak) for t in z))
            log = np.asarray(z, np.float64)-normalizer
            logs.append(log)
            probabilities.append(np.exp(log))
        lm = np.array([max(a, b)+math.log((math.exp(a-max(a, b))+math.exp(b-max(a, b)))/2)
                       for a, b in zip(*logs)])
        kl = [math.fsum(float(p*(l-m)) for p, l, m in zip(prob, log, lm)) for prob, log in zip(probabilities, logs)]
        values.append(sum(kl)/2)
        for v in range(2):
            gradients[v][q] = probabilities[v]*(logs[v]-lm-kl[v])/56
    return math.fsum(values)/28, gradients


def table_reference(values, slots):
    """Explicit record-pair membership and analytic H derivative; no production pool."""
    edges = list(itertools.combinations(range(len(slots)), 2))
    ids = {pair: i for i, pair in enumerate(edges)}
    scores, blocks = [], []
    for a, b in PAIRS:
        block = [ids[tuple(sorted((int(i), int(j))))] for i in np.flatnonzero(slots == a) for j in np.flatnonzero(slots == b)]
        x = values[block].astype(np.float64)
        peak = float(max(x)); weights = np.exp(x-peak); den = math.fsum(weights)
        scores.append(.5*math.fsum(x)/len(x)+.5*(peak+math.log(den/len(x))))
        blocks.append((block, .5/len(x)+.5*weights/den))
    return np.array(scores), blocks


def table_precision(table, group, seed):
    """Actual device/dtype H/JS versus scalar FP64 value and complete pullback."""
    x = table.detach().clone().requires_grad_(True)
    slots = [record.assignment(group), record.assignment(group, seed)]
    scores = [method.account_logits(x, s) for s in slots]
    rows = [method.query_logits(v) for v in scores]
    loss = method.consistency(*rows)
    query_grads = torch.autograd.grad(loss, rows, retain_graph=True)
    branch_grads = [torch.autograd.grad(v, x, g, retain_graph=True)[0] for v, g in zip(rows, query_grads)]
    total, = torch.autograd.grad(loss, x)
    raw = x.detach().cpu().numpy()
    local = [table_reference(raw, s) for s in slots]
    for values, (expected, _) in zip(scores, local):
        actual = values.detach().cpu().numpy().astype(np.float64)
        bound = 1e-8+1e-7*np.maximum(1, np.abs(expected)) if x.dtype == torch.float32 else 1e-12+1e-12*np.abs(expected)
        if np.any(np.abs(actual-expected) > bound):
            raise AssertionError("H output precision exceeded")
    # FP32 H output is the real JS input; casting does not undo that rounding.
    inputs = [v.detach().cpu().numpy().astype(np.float64) for v in rows]
    expected, qg = js_reference(*inputs)
    tolerance = 1e-8+1e-7*abs(expected) if x.dtype == torch.float32 else 1e-12+1e-12*abs(expected)
    if abs(float(loss)-expected) > tolerance:
        raise AssertionError("JS scalar precision exceeded")
    pulls, l1 = [], []
    for view, (actual, (_, blocks)) in enumerate(zip(query_grads, local)):
        error = float(np.abs(actual.detach().cpu().numpy()-qg[view]).sum())
        if x.dtype == torch.float32:
            if error > 2e-7: raise AssertionError("FP32 per-view gradient L1 exceeded")
        else:
            np.testing.assert_allclose(actual.detach().cpu().numpy(), qg[view], atol=1e-12, rtol=1e-12)
        l1.append(error)
        directed = np.zeros((28, 28))
        directed[~np.eye(28, dtype=bool)] = qg[view].ravel()
        pull = np.zeros(len(raw))
        for (a, b), (indices, derivative) in zip(PAIRS, blocks):
            pull[indices] += (directed[a, b]+directed[b, a])*derivative
        pulls.append(pull)
        if x.dtype == torch.float64:
            np.testing.assert_allclose(branch_grads[view].detach().cpu().numpy(), pull, atol=1e-12, rtol=1e-12)
    error = float(np.abs(total.detach().cpu().numpy()-sum(pulls)).sum())
    if x.dtype == torch.float32:
        if error > 5e-7: raise AssertionError("FP32 shared K gradient L1 exceeded")
    else:
        np.testing.assert_allclose(total.detach().cpu().numpy(), sum(pulls), atol=1e-12, rtol=1e-12)
    return dict(dtype=str(x.dtype), device=str(x.device), js=float(loss), scalar_error=abs(float(loss)-expected),
                view_gradient_l1=l1, shared_gradient_l1=error, branch_max=[float(np.max(np.abs(p))) for p in pulls]), pulls


def verify_native_table(table, group, seed):
    result, _ = table_precision(table, group, seed)
    return result


def independent_js(v0, v1):
    # Independent full graph uses entropy of explicit normalized probabilities.
    p = [torch.stack([v[PAIR_INDEX[tuple(sorted((q, j)))]] for j in range(28) if j != q]).double()
         for v in (v0, v1) for q in range(28)]
    p = [torch.exp(z-torch.logsumexp(z, 0)) for z in p]
    values = []
    for q in range(28):
        a, b = p[q], p[q+28]; m = (a+b)/2
        values.append((a*a.log()).sum()/2+(b*b.log()).sum()/2-(m*m.log()).sum())
    return (torch.stack(values).sum()/28).to(v0.dtype)


def independent_objective(table, group, seed, target=None):
    slots = [record.assignment(group), record.assignment(group, seed)]
    v0, v1 = [literal_pool(table, s) for s in slots]
    sup = (literal_loss(v0, group)+literal_loss(v1, group))/2
    c = independent_js(v0, v1)
    d = table.new_zeros(()) if target is None else (v0-literal_pool(torch.as_tensor(target, dtype=table.dtype), slots[0])).square().mean()
    w = 1. if target is None else .1
    return w*sup+w*c+.5*d, dict(supervised=float(sup.detach()), consistency=float(c.detach()), mse0=float(d.detach()))


def shared_memory():
    groups = [handmade_group(f"A_fit_{i:02}") for i in range(48)]
    memory = record.Memory("ABC")
    memory.retain(groups, 1, lambda g: (np.sin(np.arange(1540)/7)*2).astype(np.float32))
    memory.maps.update(method.metadata())
    return memory


class HConsistencyTests(unittest.TestCase):
    def temporary(self):
        t = tempfile.TemporaryDirectory(dir=os.environ["H_CONS_TEST_ROOT"])
        self.addCleanup(t.cleanup)
        return Path(t.name)

    def evidence(self, name, value):
        data.write_json(Path(os.environ["H_CONS_TEST_ROOT"])/(name+".json"), value)

    def test_js_value_gradient_precision_and_faults(self):
        scores = torch.sin(torch.arange(378, dtype=torch.float64)*.31)*3
        np.testing.assert_array_equal(method.query_logits(scores).numpy(), scores.numpy()[QUERY_INDEX])
        rows = []
        z0 = np.sin(np.arange(756).reshape(28, 27)*.17)*2
        z1 = np.cos(np.arange(756).reshape(28, 27)*.23)*3
        reference, expected = js_reference(z0, z1)
        for dtype in (torch.float64, torch.float32):
            a, b = [torch.tensor(v, dtype=dtype, requires_grad=True) for v in (z0, z1)]
            value = method.consistency(a, b)
            ref, gradients = js_reference(a.detach().double().numpy(), b.detach().double().numpy())
            actual = torch.autograd.grad(value, (a, b), retain_graph=True)
            self.assertLessEqual(abs(float(value)-ref), 1e-12+1e-12*abs(ref) if dtype == torch.float64 else 1e-8+1e-7*abs(ref))
            for v, g in zip(actual, gradients):
                if dtype == torch.float64: np.testing.assert_allclose(v.numpy(), g, atol=1e-12, rtol=1e-12)
                else: self.assertLessEqual(float(np.abs(v.numpy()-g).sum()), 2e-7)
            # Locate the actual live mixture node, with both log-softmax parents.
            todo, seen, mixtures = [value.grad_fn], set(), []
            while todo:
                node = todo.pop()
                if node is None or node in seen: continue
                seen.add(node)
                if type(node).__name__.startswith("LogaddexpBackward"):
                    mixtures.append([type(n).__name__ for n, _ in node.next_functions])
                todo.extend(n for n, _ in node.next_functions)
            self.assertEqual(len(mixtures), 1)
            self.assertTrue(all(n.startswith("LogSoftmaxBackward") for n in mixtures[0]))
            self.assertNotIn("detach", inspect.getsource(method.consistency))
            rows.append(dict(dtype=str(dtype), js=float(value), scalar_error=abs(float(value)-ref), mixture_parents=mixtures[0]))
        self.assertAlmostEqual(float(method.consistency(torch.tensor(z0), torch.tensor(z0))), 0., places=14)
        q = method.query_logits(scores)
        self.assertLess(abs(float(method.consistency(q, method.query_logits(scores+7)))), 1e-12)
        star = scores.clone();star[:27] += 1
        self.assertGreater(float(method.consistency(q, method.query_logits(star))), 1e-5)
        self.assertGreater(abs(float(method.consistency(torch.tensor(z0), torch.tensor(z1[:, ::-1].copy())))-reference), 1e-5)
        self.assertGreater(abs(reference-reference/27), 1e-5)
        extreme = torch.full((28, 27), -1000., dtype=torch.float64);extreme[:, 0] = 1000.
        other = torch.roll(extreme, 1, 1)
        self.assertAlmostEqual(float(method.consistency(extreme, other)), math.log(2), places=12)
        with self.assertRaises(ValueError):method.consistency(torch.zeros(28, 28), torch.zeros(28, 28))
        with self.assertRaises(ValueError):method.consistency(torch.full((28, 27), float("nan")), torch.zeros(28, 27))
        self.evidence("js_precision", dict(cases=rows, canonical_columns=True, global_offset_only=True,
            mixture_live_graph=True, extreme_logit_bound=math.log(2), scope="First-order m-only detach is equivalent; no Hessian requirement"))

    def test_shared_table_pullback_and_finite_difference(self):
        group = handmade_group("pullback")
        values = np.sin(np.arange(1540)*.17)*3+np.cos(np.arange(1540)*.07)
        results = []
        for dtype in (torch.float64, torch.float32):
            result, branches = table_precision(torch.tensor(values, dtype=dtype), group, 17)
            self.assertTrue(all(float(np.max(np.abs(g))) > 5e-7 for g in branches))
            x = torch.tensor(values, dtype=dtype, requires_grad=True)
            view = [method.query_logits(method.account_logits(x, record.assignment(group, seed))) for seed in (None, 17)]
            complete, = torch.autograd.grad(method.consistency(*view), x, retain_graph=True)
            for missing in (0, 1):
                detached = [v.detach() if i == missing else v for i, v in enumerate(view)]
                wrong, = torch.autograd.grad(method.consistency(*detached), x, retain_graph=True)
                self.assertGreater(float((complete-wrong).abs().sum()), 5e-7)
            results.append(result)
        slots = [record.assignment(group, seed) for seed in (None, 17)]
        def scalar(v):
            q = [table_reference(v, s)[0][QUERY_INDEX] for s in slots]
            return js_reference(*q)[0]
        _, branches = table_precision(torch.tensor(values, dtype=torch.float64), group, 17)
        total = sum(branches);differences=[]
        for i in (2, 11, 53, 79, 123, 301, 511, 777, 1001, 1403):
            self.assertGreater(abs(total[i]), 1e-10)
            a, b = values.copy(), values.copy();a[i] += 1e-5;b[i] -= 1e-5
            observed = (scalar(a)-scalar(b))/2e-5
            self.assertLessEqual(abs(observed-total[i]), 2e-9)
            differences.append(dict(coordinate=i, analytical=float(total[i]), finite_difference=observed, error=abs(observed-total[i])))
        zero = torch.ones(1540, dtype=torch.float64)
        v = [method.query_logits(method.account_logits(zero, s)) for s in slots]
        self.assertLess(abs(float(method.consistency(*v))), 1e-12)
        native = []
        for role, stage, step in (("current",1,2),("current",2,49),("history",2,49)):
            group = handmade_group("native_"+role, 8, 254)
            self.assertEqual(sum(map(len, group.items)), 224)
            self.assertEqual({len(text) for records in group.items for item in records for text in item[1:]}, {254})
            seed = data.seed_for(20260918,"ABC",stage,step,role,group.uid,"regroup")
            natural, regrouped = record.assignment(group), record.assignment(group,seed)
            np.testing.assert_array_equal(np.bincount(natural),np.bincount(regrouped))
            self.assertFalse(np.array_equal(natural,regrouped))
            i,j=np.triu_indices(224,1)
            hand_table=(np.sin(.17*i+.31*j)*3+np.cos(.07*i-.19*j)).astype(np.float32)
            check,_=table_precision(torch.from_numpy(hand_table),group,seed)
            self.assertGreater(check["js"],1e-6)
            target=(hand_table+2*np.sin(.31*i+.71*j)).astype(np.float32)
            _,parts=method.objective(torch.from_numpy(hand_table),group,seed,"H_cons",target)
            self.assertGreater(parts["mse0"],1e-6)
            native.append(dict(role=role,stage=stage,step=step,seed=seed,assignment_sha256=hashlib.sha256(regrouped.tobytes()).hexdigest(),
                               precision=check,synthetic_D0=parts["mse0"]))
        self.evidence("shared_K_precision", dict(cases=results, finite_difference=differences, missing_students_detected=True,
            legal_assignment_zero=True, native_handwritten_geometry=native,
            native_fixture=dict(current_uid="native_current",history_uid="native_history",
                records_per_account=8,text_length=254,expected_tokens=256,paired_clock=336,
                teacher="W_eval_K + 2*sin(.31*i+.71*j)", fixed_steps=[1,2,2,49,49,49])))

    def test_complete_objective_vjp_and_actual_updates(self):
        c = run.config();c["handwritten_qualification"] = True
        current, history = handmade_group("current"), handmade_group("history")
        model = tiny_model();method.bind(model, method.metadata());optimizer = core.make_optimizer(model, c)
        method.update(model, optimizer, current, None, None, c, "H_cons", 1, 1, qualification_without_c=True)
        initial, rng, modes = qualification.state(model, optimizer), diag.rng_state(), [m.training for m in model.modules()]
        with diag.neutral(model): old = record.reference(model, history, c)
        i, j = np.triu_indices(56, 1); target = (old+2*np.sin(.31*i+.71*j)).astype(np.float32)
        effects = {}
        for stage, step, historic in ((1, 2, False), (2, 49, True)):
            qualification.restore(model, optimizer, initial, rng, modes)
            if historic:
                for v in optimizer.state.values():v["step"].fill_(336)
            start = qualification.state(model, optimizer)
            direct = copy.deepcopy(model);direct_opt = core.make_optimizer(direct, c);direct_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            branch = history if historic else None; teacher = target if historic else None
            pre_off = {}
            def observe_off(role, log):
                pre_off[role] = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            off_log = method.update(model, optimizer, current, branch, teacher, c, "H_cons", stage, step,
                                    observe=observe_off, qualification_without_c=True)
            off = qualification.state(model, optimizer)
            qualification.restore(model, optimizer, start, rng, modes)
            observed = {}
            def observe(role, log):observed[role] = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            row = method.update(model, optimizer, current, branch, teacher, c, "H_cons", stage, step, observe=observe)
            run.loss_total(row, stage, "H_cons")
            direct.train();direct_opt.zero_grad(set_to_none=True);values=[]
            for role, group, ref in (("current",current,None), ("history",branch,teacher)):
                if group is None:continue
                torch.manual_seed(data.seed_for(20260918,"ABC",stage,step,role,"dropout"))
                z = record.record_vectors(direct,group,c);a,b=torch.triu_indices(len(z),len(z),1)
                u,v=z[a],z[b];table=direct.head(torch.cat(((u-v).abs(),u*v),1)).flatten()
                seed=data.seed_for(20260918,"ABC",stage,step,role,group.uid,"regroup")
                loss, components=independent_objective(table,group,seed,ref);values.append(float(loss.detach()))
                for key,value in components.items():self.assertTrue(np.isclose(row[role][key],value,atol=2e-6,rtol=2e-5))
                loss.backward()
                for name,p in direct.named_parameters():torch.testing.assert_close(observed[role][name],p.grad,atol=2e-6,rtol=2e-5)
            torch.nn.utils.clip_grad_norm_(direct.parameters(),1.)
            direct_opt.param_groups[0]["lr"],direct_opt.param_groups[1]["lr"]=run.parent.stage_lr(step),.001
            direct_opt.step()
            self.assertTrue(np.isclose(row["total"],sum(values),atol=2e-6,rtol=2e-5))
            for (name,p),q in zip(model.named_parameters(),direct.parameters()):
                torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)
                for key in ("step","exp_avg","exp_avg_sq"):
                    torch.testing.assert_close(optimizer.state[p][key],direct_opt.state[q][key],atol=2e-6,rtol=2e-5)
            effect = qualification.component_effects(model,off)
            role = "history" if historic else "current"
            for component in ("encoder","head"):
                delta=max(float((observed[role][n]-pre_off[role][n]).abs().max()) for n in observed[role] if n.startswith(component+"."))
                self.assertGreater(delta,0);effect[component]["preclip_max_delta"]=delta
            extra=row["current"]["consistency"]+(.1*row["history"]["consistency"] if historic else 0)
            self.assertTrue(np.isclose(row["total"]-off_log["total"],extra,atol=2e-6,rtol=2e-6))
            self.assertGreater(extra,2e-6)
            bad=copy.deepcopy(row);bad["total"]-=extra
            with self.assertRaises(ValueError):run.loss_total(bad,stage,"H_cons")
            bad=copy.deepcopy(row);bad.pop("training")
            with self.assertRaises(ValueError):run.loss_total(bad,stage,"H_cons")
            if historic:
                self.assertGreater(row["history"]["mse0"],0)
                bad=copy.deepcopy(row);bad["history"]["consistency_contribution"]=bad["history"]["consistency"]
                with self.assertRaises(ValueError):run.loss_total(bad,stage,"H_cons")
            effects["paired" if historic else "current"] = dict(components=effect,total=row["total"],without_c=off_log["total"],C_contribution=extra)
        with self.assertRaises(ValueError):method.update(model,optimizer,current,None,None,run.config(),"H_cons",1,2,qualification_without_c=True)
        self.evidence("complete_graph_and_update",dict(effects=effects,independent_full_graph=True,adam_moments_compared=True,
            scope="Real tiny-model updates; paired clock336 synthetic, no formal model or inputs"))

    def test_actual_capture_restore_and_training_identity(self):
        c=run.config();model=tiny_model();method.bind(model,method.metadata());optimizer=core.make_optimizer(model,c)
        current,history=handmade_group("current"),handmade_group("history")
        method.update(model,optimizer,current,None,None,c,"H_cons",1,1)
        for v in optimizer.state.values():v["step"].fill_(336)
        start,rng,modes=qualification.state(model,optimizer),diag.rng_state(),[m.training for m in model.modules()]
        target=(3*np.sin(np.arange(1540)/7)).astype(np.float32)
        first=method.update(model,optimizer,current,history,target,c,"H_cons",2,49)
        expected,expected_rng=qualification.state(model,optimizer),diag.rng_state()
        qualification.restore(model,optimizer,start,rng,modes)
        memory=shared_memory();memory.begin_stage(2)
        for _ in range(48):memory.draw()
        payload=memory.to_bytes();fit=[handmade_group(f"fit{i}") for i in range(48)]
        valid=[data.Group(g.uid,g.sellers,g.items) for g in [handmade_group(f"valid{i}") for i in range(40)]]
        out=self.temporary();rec=method.capture(model,optimizer,fit,memory,valid,c,"H_cons",2,1,out,"fixture")
        capture=data.read_json(out/rec["path"]);run.validate_diagnostic(capture,"H_cons")
        with diag.neutral(model):
            raw=method.score(model,valid,c,"H_cons")
            np.testing.assert_array_equal(run.previous.array(out,capture["valid_blind"],(40,378),np.float32),raw)
            for role,groups in (("fit",fit),("cache",memory.reservoir.groups)):
                live=method.score(model,groups,c,"H_cons")
                np.testing.assert_array_equal(run.previous.array(out,capture[role],(len(groups),3),np.float64),diag.metrics(groups,live))
            group=memory.reservoir.groups[0];live=record.reference(model,group,c);old=memory.tables[group.uid]
            for key,seed in (("D0",None),("D1",data.seed_for(20261008,"ABC",2,group.uid,"diagnostic_regroup"))):
                slots=record.assignment(group,seed)
                residual=(literal_pool(torch.from_numpy(live),slots)-literal_pool(torch.from_numpy(old),slots)).numpy().astype(np.float64)
                self.assertAlmostEqual(capture["teacher_errors"][0][key],float(np.mean(residual**2)),places=6)
        self.assertEqual(core.state_digest(start),core.state_digest(qualification.state(model,optimizer)))
        self.assertEqual((rng,modes,payload),(diag.rng_state(),[m.training for m in model.modules()],memory.to_bytes()))
        repeated=method.update(model,optimizer,current,history,target,c,"H_cons",2,49)
        self.assertEqual(first,repeated);self.assertEqual(expected_rng,diag.rng_state())
        self.assertEqual(core.state_digest(expected),core.state_digest(qualification.state(model,optimizer)))
        meta={**method.metadata(),"rng":diag.rng_state(),"modes":[m.training for m in model.modules()]}
        with diag.neutral(model):before=method.restored_score(model,[current],c)
        for kind,opt in (("full",optimizer),("inference",None)):
            p=out/(kind+".pt");receipt=core.save_state(p,model,opt,meta);model.training_identity={"arm":"H"}
            run.restore_point(p,receipt,model,opt,"H_cons")
            with diag.neutral(model):np.testing.assert_array_equal(before,method.restored_score(model,[current],c))
        for bad in ({"pooling":method.identity()}, {**method.metadata(),"training":{}}):
            with self.assertRaises(ValueError):method.bind(model,bad)
        bad=copy.deepcopy(capture);bad.pop("training")
        with self.assertRaises(ValueError):run.validate_diagnostic(bad,"H_cons")
        self.assertEqual(run.memory_from(payload,"H_cons").to_bytes(),payload)
        memory.maps.pop("training")
        with self.assertRaises(ValueError):run.memory_from(memory.to_bytes(),"H_cons")
        self.evidence("capture_restore_identity",dict(actual_roles=[48,6,40],all_H_scores=True,D0_D1_H=True,
            next_update_bitwise_equal=True,full_inference_restored=True,missing_training_rejected=True))

    def test_unchanged_single_arm_defaults(self):
        # Only shared runner/default paths changed. Reuse their existing witnesses.
        for name in ("test_single_candidate_orchestration_and_blind_gate",
                     "test_saved_comparisons_directions_permissions_and_budgets"):
            result=unittest.TestResult();GHalfTests(name).run(result)
            self.assertTrue(result.wasSuccessful(),str(result.errors)+str(result.failures))
        self.evidence("old_default_increment",dict(tests=2,scope="Existing H control-flow and saved-statistic fixtures only"))

    def test_loss_gate_and_saved_comparison_interfaces(self):
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
            row=dict(total=1.2,encoder_lr=run.parent.stage_lr(step),head_lr=.001,**method.metadata(arm))
            for role,g in (("current",current),("history",history)):
                if g is not None:
                    seed=data.seed_for(20260918,"ABC",stage,step,role,g.uid,"regroup")
                    row[role]=dict(supervised=1.,consistency=.2,weighted=1.2 if role=="current" else .12,mse0=0.,mse1=None,distillation=0.,
                        qualification_without_c=False,supervised_contribution=1. if role=="current" else .1,
                        consistency_contribution=.2 if role=="current" else .02,teacher_contribution=0.,
                        assignment_sha256=hashlib.sha256(record.assignment(g,seed).tobytes()).hexdigest())
            if history:row["total"]=1.32
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
        with mock.patch.object(run,"baseline_runs",return_value={a:(oldroot,{}, {}) for a in ("S","G_half")}), \
             mock.patch.object(run,"baseline_inputs",return_value=(fake_baseline,oldroot,{},{"public_inputs":{"handwritten":True}})), \
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
                log=data.read_json(root/"run/updates"/(stage+"_H_cons.json"))
                for other in ("S","G_half"):data.write_json(oldroot/"run/updates"/(stage+"_"+other+".json"),log)
            scores,checked=run.blind_gate(root/"run",manifest)
            self.assertEqual((set(scores),checked),(set(run.POINTS),split))
            for arm in method.ARMS:
                self.assertEqual(manifest["starts"][arm],manifest["points"]["A_"+arm]["continuity"])
            for collection,key in (("points","A_H_cons"),("diagnostics","B_H_cons"),("training","A_H_cons"),("starts","H_cons")):
                damaged=copy.deepcopy(manifest);del damaged[collection][key]
                with self.assertRaises(ValueError):run.open_valid(root,damaged,valid,meta,c)
                self.assertEqual(data.read_json(root/"access.json")["valid"],0)
            damaged=copy.deepcopy(manifest);damaged["starts"]["H_cons"]["model"]="wrong_A"
            with self.assertRaises(ValueError):run.open_valid(root,damaged,valid,meta,c)
            damaged=copy.deepcopy(manifest);damaged["points"]["B_H_cons"]["pooling"]={"arm":"S"}
            with self.assertRaises(ValueError):run.open_valid(root,damaged,valid,meta,c)
            self.assertEqual(data.read_json(root/"access.json")["valid"],0)
            badlog=data.read_json(oldroot/"run/updates/B_G_half.json")
            badlog["rows"][0]["current"]["assignment_sha256"]="wrong"
            data.write_json(oldroot/"run/updates/B_G_half.json",badlog)
            with self.assertRaises(ValueError):run.open_valid(root,manifest,valid,meta,c)
            self.assertEqual(data.read_json(root/"access.json")["valid"],0)
        self.check_saved_comparisons()
        self.evidence("h_cons_orchestration",dict(fixture="Actual scheduler, pooling, Memory and checkpoint/diagnostic IO; mocked update/forward/metrics/calibration only",
            scheduler_calls=len(calls),real_updates_in_this_fixture=0,independent_initializations=len(loads),
            diagnostics=12,endpoints=2,own_A_states=True,blind_rejections=True,
            scope="Control-flow evidence only. Real tiny gradient/update and actual diagnostic tests are separate."))

    def check_saved_comparisons(self):
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
                qpath.unlink()  # Replace only this explicitly constructed temporary algebra fixture.
                rec["queries"]=run.previous.save_array(qpath,q,origin)
        for origin,collection,a,b,la,lb in ((hroot,col,.5,.6,.4,.5),(oldroot/"evaluation",oldcol,.4,.4,.3,.3)):
            set_values(origin,collection,"A_H_cons",a,la);set_values(origin,collection,"B_H_cons",b,lb)
        oldcol["points"]={stage+"_"+arm:copy.deepcopy(oldcol["points"][stage+"_H_cons"]) for stage in "AB" for arm in ("S","G_half")}
        data.write_json(hroot/"collected.json",col)
        baseline={"points":list(oldcol["points"])}
        with mock.patch.object(run,"baseline_runs",return_value={a:(oldroot,oldcol,{}) for a in ("S","G_half")}):
            result=run.finalize(hroot)
            self.assertTrue(result["success"]["passed"]);self.assertTrue(result["success"]["direct_H_new_B_positive"])
            for other in ("S","G_half"):
                self.assertAlmostEqual(result["comparisons"]["H_cons-"+other]["raw"]["A"]["map"]["mean"],.2)
                self.assertAlmostEqual(result["forgetting_comparisons"]["H_cons-"+other]["raw"]["map"]["mean"],-.1)
                for key in ("brier","log_loss"):
                    self.assertAlmostEqual(result["forgetting_comparisons"]["H_cons-"+other]["raw"][key]["mean"],.1)
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
        self.evidence("h_cons_comparison_and_permissions",dict(success=result["success"],forgetting=result["forgetting_comparisons"]["H_cons-S"]["raw"],
            fixture="Handwritten saved-metric signs/UID and real collection; no baseline payload read",parse_failure_consumed=True,blind_open_blocked=True,projection=projection))
