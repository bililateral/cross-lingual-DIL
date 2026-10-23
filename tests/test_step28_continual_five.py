"""Affected actual entries on handwritten data; executed only by Linux py310."""
import copy
import hashlib
import os
from pathlib import Path
import random
import tempfile
import unittest
import sys
import json
import itertools
import psutil
from unittest import mock

import numpy as np
import torch

import step28_continual_five as m
import step28_continual_five_run as run
import step28_continual_five_control as ctl
import step28_continual_five_evaluate as evaluation
from test_step28_record_replay import handmade_group, TinyEncoder
from test_step28_optuna import reference


def fixture(name, order="CAB", seed=4, step=624):
    spec = m.Spec(name, order, seed)
    method, c = m.Method(spec), m.config(spec)
    torch.manual_seed(41)
    model = m.core.build_model(TinyEncoder(), 8 if spec.is_record else 16, 5)
    m.checkpoint_encoder(model.encoder)
    method.bind(model, method.metadata())
    optimizer = m.core.make_optimizer(model, c)
    # Explicit handwritten Adam state; not a completed training trajectory.
    for p in model.parameters():
        optimizer.state[p] = dict(step=torch.tensor(float(step)), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
    return spec, method, c, model, optimizer


def original_objective(scores, group):
    from test_step28_positive_ranking import independent_loss, independent_rank, LOOKUP
    original = []
    for q in range(28):
        ids = [LOOKUP[tuple(sorted((q, j)))] for j in range(28) if j != q]
        pos = [k for k in ids if group.labels[k]]
        original.append(torch.logsumexp(scores[ids], 0)-scores[pos].mean())
    return independent_loss(scores, group)-independent_rank(scores, group.labels)+torch.stack(original).mean()


class FiveTests(unittest.TestCase):
    def evidence(self, name, row):
        m.data.write_json(Path(os.environ["FIVE_TEST_ROOT"])/(name+".json"), row)

    def test_actual_five_updates_and_omission_controls(self):
        current, history = handmade_group("current"), handmade_group("history")
        evidence = {}
        for name in m.METHODS:
            spec, method, c, model, optimizer = fixture(name)
            direct = copy.deepcopy(model)
            self.assertIs(direct.encoder._five_original_forward.__self__, direct.encoder)
            direct.encoder._five_outer_checkpoint = False
            other = m.core.make_optimizer(direct, c)
            other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            target = np.sin(np.arange(1540 if spec.is_record else 378)/7).astype(np.float32)*3
            if name in ("SEQ_ref", "ER0.1_ref"):
                target = None
            h = None if name == "SEQ_ref" else history
            row = method.update(model, optimizer, current, h, target, c, 3, 49)
            direct.train()
            other.zero_grad(set_to_none=True)
            omission_gradients = []
            for role, group, teacher in (("current", current, None), ("history", h, target)):
                if group is None:
                    continue
                torch.manual_seed(m.data.seed_for(spec.roots["schedule"], spec.order, 3, 49, role, "dropout"))
                if spec.is_record:
                    z = m.record.record_vectors(direct, group, c)
                    i, j = torch.triu_indices(len(z), len(z), 1)
                    table = direct.head(torch.cat(((z[i]-z[j]).abs(), z[i]*z[j]), 1)).flatten()
                    regroup = m.data.seed_for(spec.roots["schedule"], spec.order, 3, 49, role, group.uid, "regroup")
                    objective = reference(table, group, regroup, spec, teacher)
                    if teacher is not None:
                        mutant = .1*reference(table, group, regroup, spec, None)
                        real_grad, = torch.autograd.grad(objective, table, retain_graph=True)
                        wrong_grad, = torch.autograd.grad(mutant, table, retain_graph=True)
                        self.assertGreater(float((real_grad-wrong_grad).abs().max()), 1e-7)
                else:
                    scores = m.base.logits(direct, group, c, "split_rank")
                    supervised = original_objective(scores, group)
                    objective = supervised if role == "current" else .1*supervised
                    if teacher is not None:
                        objective = objective+.5*(scores-torch.from_numpy(teacher)).square().mean()
                    if role == "history":
                        real_grad, = torch.autograd.grad(objective, scores, retain_graph=True)
                        without_teacher, = torch.autograd.grad(.1*supervised, scores, retain_graph=True)
                        self.assertGreater(float(real_grad.abs().max()), 1e-7)
                        omission_gradients.append(float(real_grad.abs().max()))
                        if teacher is not None:
                            self.assertGreater(float((real_grad-without_teacher).abs().max()), 1e-7)
                            wrongly_scaled, = torch.autograd.grad(.1*(supervised+.5*(scores-torch.from_numpy(teacher)).square().mean()), scores, retain_graph=True)
                            self.assertGreater(float((real_grad-wrongly_scaled).abs().max()), 1e-7)
                objective.backward()
            errors = []
            # The actual path has clipped once; compare its complete pre-step effect
            # against independently assembled current/history gradients and Adam.
            torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.)
            for group, rate in zip(other.param_groups, spec.rates(49)):
                group["lr"] = rate
            other.step()
            for p, q in zip(model.parameters(), direct.parameters()):
                torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5)
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(optimizer.state[p][field], other.state[q][field], atol=3e-6, rtol=3e-5)
                errors.append(float((p-q).detach().abs().max()))
            self.assertEqual(m.parent.adam_step(optimizer), 625)
            self.assertEqual((row["encoder_lr"], row["head_lr"]), spec.rates(49))
            with self.assertRaises(ValueError):
                method.update(model, optimizer, current, h, target, c, 3, 49)
            evidence[name] = dict(max_parameter_error=max(errors), actual_adam_step=625,
                                  omitted_history_or_teacher_detected=name != "SEQ_ref", roots=spec.roots)
        self.evidence("actual_five_updates", evidence)

    def test_three_stage_memory_orders_seeds_and_birth_targets(self):
        first = [handmade_group(f"first_{i:02}") for i in range(48)]
        second = [handmade_group(f"second_{i:02}") for i in range(48)]
        evidence = []
        for order, seed in (("ABC", 4), ("BCA", 5), ("CAB", 6)):
            summaries = []
            for name in m.METHODS[1:]:
                spec, method, c, model, _ = fixture(name, order, seed)
                memory = method.new_memory(dict(a=1., b=0.))
                length = 1540 if spec.is_record else 378
                def birth(group):
                    return np.full(length, 1. if group.uid.startswith("first") else 2., np.float32)
                with mock.patch.object(m.record, "reference", side_effect=lambda mod, g, config, check: birth(g)), \
                     mock.patch.object(method, "score", side_effect=lambda mod, gs, config, check: np.stack([birth(g) for g in gs])):
                    method.retain(memory, model, first, c, 1, dict(first=dict(a=1., b=0.)))
                    memory.begin_stage(2)
                    for _ in range(288):
                        memory.draw()
                    method.retain(memory, model, second, c, 2, dict(first=dict(a=1., b=0.)))
                memory.begin_stage(3)
                payload = memory.to_bytes()
                restored = method.memory_from(payload)
                allowed = set(memory.summary()["members"])
                observed = []
                for _ in range(8):
                    g, target = memory.draw()
                    rg, rt = restored.draw()
                    self.assertEqual(g.uid, rg.uid)
                    self.assertIn(g.uid, allowed)
                    if target is not None:
                        np.testing.assert_array_equal(target, birth(g))
                        np.testing.assert_array_equal(target, rt)
                    observed.append(g.uid)
                self.assertEqual(memory.to_bytes(), restored.to_bytes())
                with self.assertRaises(ValueError):
                    m.Method(m.Spec(name, order, 5 if seed != 5 else 4)).memory_from(payload)
                summaries.append((allowed, observed))
                evidence.append(dict(spec=spec.identity(), members=sorted(allowed), draws=observed, bytes=len(payload)))
            self.assertTrue(all(row == summaries[0] for row in summaries))
            spec = m.Spec("S_tuned", order, seed)
            seq, _ = m.parent.schedule(second, m.config(spec), order, 3)
            expected = m.data.schedule(second, 6, m.data.seed_for(spec.roots["schedule"], order, 3, "current"))
            self.assertEqual([g.uid for g in seq], [g.uid for g in expected])
        self.evidence("memory_stage3", evidence)

    def test_real_restore_next_update_and_budget(self):
        out = Path(os.environ["FIVE_TEST_ROOT"])
        spec, method, c, model, optimizer = fixture("H_pos_tuned")
        current, history = handmade_group("current"), handmade_group("history")
        target = np.sin(np.arange(1540)).astype(np.float32)
        metadata = {**method.metadata(), "rng": m.diag.rng_state(), "modes": [v.training for v in model.modules()]}
        with tempfile.TemporaryDirectory(dir=out) as temporary:
            root = Path(temporary)
            budget = run.Budget(root, __import__("time").monotonic()+60, tail=1)
            full = run.save(root/"full.pt", model, optimizer, metadata, budget)
            first = method.update(model, optimizer, current, history, target, c, 3, 49)
            digest = m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
            method.restore(root/"full.pt", full, model, optimizer)
            second = method.update(model, optimizer, current, history, target, c, 3, 49)
            self.assertEqual(first, second)
            self.assertEqual(digest, m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict())))
            with m.diag.neutral(model):
                before = method.score(model, [current], c)
            inference = run.save(root/"infer.pt", model, None, metadata, budget)
            method.restore(root/"infer.pt", inference, model, None)
            with m.diag.neutral(model):
                np.testing.assert_array_equal(before, method.score(model, [current], c))
        now = [0.]
        ledger = ctl.Ledger(clock=lambda: now[0])
        for i in range(3):
            ledger.reserve(str(i), "training", 100.)
        now[0] = 40.
        self.assertEqual(ledger.available("training"), ctl.LIMITS["training"]-300)
        ledger.settle("1", success=True)
        self.assertEqual(ledger.available("training"), ctl.LIMITS["training"]-240)
        ledger.used["training"] = ctl.LIMITS["training"]-300
        with self.assertRaises(ValueError):
            ledger.reserve("over", "training", 60, remaining_projection=41)
        now[0] = 101.
        with self.assertRaises(RuntimeError):
            ledger.settle("0", success=True)
        self.assertEqual(ledger.completed[-1]["actual_seconds"], 101)
        self.assertTrue(ledger.failed)
        with mock.patch.object(run.time, "monotonic", return_value=100.):
            with self.assertRaises(TimeoutError):
                run.Budget(out, 101., 2.).check()
        self.evidence("restore_budget", dict(actual_next_update_bitwise_equal=True, inference_equal=True,
            same_absolute_child_deadline=True, active_elapsed_not_double_counted=True, actual_overrun_not_clipped=True))

    def test_exact_complete_id_gate(self):
        rows = []
        for spec in m.specifications():
            rows.append(dict(status=run.COMPLETE, spec=spec.identity(), source_files=[], updates=864,
                initial=dict(model=f"family{spec.is_record}_s{spec.seed}"),
                gradient_group_presentations=864 if spec.name == "SEQ_ref" else 1440,
                access=dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1),
                training={str(i): {} for i in (1, 2, 3)}, points={str(i): dict(continual_spec=spec.identity(), adam_step=i*288,
                    full_restore_verified=True, inference_restore_verified=True, development_ids=list(range(60)),
                    calibration_ids=list(range(12)), memory_summary=dict(members=list(range(6))),
                    scores={k: {} for k in ("calibration", "raw", "full-calibration", "full-raw",
                        "inference-calibration", "inference-raw", "stage-cal", "first-cal")}) for i in (1, 2, 3)}))
        random.Random(19).shuffle(rows)
        ctl.joint_opening(rows, [])
        for mutant in (rows[:-1], rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):
                ctl.joint_opening(mutant, [])
        broken = copy.deepcopy(rows)
        broken[0]["points"]["3"]["full_restore_verified"] = False
        with self.assertRaises(ValueError):
            ctl.joint_opening(broken, [])
        broken = copy.deepcopy(rows)
        broken[0]["initial"]["model"] = "foreign_initial_state"
        with self.assertRaises(ValueError):
            ctl.joint_opening(broken, [])
        broken = copy.deepcopy(rows)
        broken[0]["access"]["valid"] = 1
        with self.assertRaises(ValueError):
            ctl.joint_opening(broken, [])
        self.evidence("complete_ids", dict(exact_45_unordered=True, duplicate_missing_bad_restore_and_early_open_rejected=True))

    def test_real_child_timeout_failure_and_pending_projection(self):
        out = Path(os.environ["FIVE_TEST_ROOT"])
        with tempfile.TemporaryDirectory(dir=out) as temporary:
            root = Path(temporary)
            ledger = ctl.Ledger(root/"ledger.json")
            queue = [dict(id=str(i), allowance=5., projected_seconds=1.) for i in range(3)]
            # Three real, CPU-only subprocesses; production dispatch and reservations.
            process = psutil.Process()
            original_affinity = process.cpu_affinity()
            controller_cores = psutil.Process(os.getppid()).cpu_affinity()[:3]
            try:
                # The CPU-test wrapper is pinned to one core. Only this controller
                # fixture exposes three available cores; each real child stays on one.
                process.cpu_affinity(controller_cores)
                ctl.dispatch(queue, root, ledger, "cpu", lambda row, job, deadline: [sys.executable, "-c",
                    "import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))"], tail=1, poll_seconds=.1)
            finally:
                process.cpu_affinity(original_affinity)
            affinities = [json.loads((root/str(i)/"console.txt").read_text()) for i in range(3)]
            self.assertTrue(all(len(v) == 1 for v in affinities))
            self.assertEqual(len({v[0] for v in affinities}), 3)
            self.assertEqual(len(ledger.completed), 3)
            self.assertFalse(ledger.active)
            failed = ctl.Ledger(root/"failed_ledger.json")
            with self.assertRaises(TimeoutError):
                ctl.dispatch([dict(id="timeout", allowance=1.8)], root, failed, "cpu",
                             lambda row, job, deadline: [sys.executable, "-c", "import time; time.sleep(15)"],
                             max_workers=1, tail=1.4)
            self.assertTrue(failed.failed)
            self.assertFalse(failed.active)
            self.assertGreater(failed.used["cpu"], 0)
            self.assertLess(failed.used["cpu"], 1.8)
        self.evidence("real_launcher", dict(three_real_children_completed=True, owned_timeout_stopped=True,
                                             failure_cost_preserved=True, pending_projection_distinct_from_active_reservation=True))

    def test_domain_paired_statistics_and_access_attempts(self):
        spec = m.Spec("SEQ_ref", "BCA", 4)
        cells = {f"{i}_{j}": np.full((40, 22), i*10+j, np.float64)
                 for i, j in evaluation.CORE_CELLS+((1, 2), (2, 3))}
        fields = {f: evaluation.contributions(spec, cells, f, 40)
                  for f in ("O", "A_acq", "F", "D2", "D3", "G2", "G3")}
        np.testing.assert_array_equal(fields["O"], fields["A_acq"]-fields["F"])
        np.testing.assert_array_equal(fields["F"], fields["D2"]/2+fields["D3"])
        self.assertEqual(float(fields["O"].mean(1).sum(0)[12]), 31.5)
        self.assertEqual(float(fields["O"][0].sum()), 0.)  # BCA old domains are B and C.
        draws = np.random.Generator(np.random.PCG64(evaluation.BOOTSTRAP_SEED)).integers(40, size=(5000, 3, 40))
        intervals = evaluation.interval(fields["O"], draws)
        self.assertEqual(intervals["map"]["conditional_95pct_interval"], [31.5, 31.5])
        with tempfile.TemporaryDirectory(dir=os.environ["FIVE_TEST_ROOT"]) as temporary:
            root = Path(temporary)
            evaluation.access(root, "new_confirmation_joint", ("new_pairs",))
            with self.assertRaises(ValueError):
                evaluation.access(root, "new_confirmation_joint", ("new_pairs",))
        self.evidence("paired_statistics", dict(actual_domain_mapping=True, acquisition_forgetting_identity=True,
            same_transition_seq_identity=True, constant_bootstrap_exact=True, repeated_label_attempt_rejected=True,
            no_generated_confirmation_or_formal_labels_used=True))

    def test_actual_endpoint_calibration_restore_and_stage3_supply(self):
        old = [handmade_group(f"old_{i:02}") for i in range(48)]
        current = [handmade_group(f"new_{i:02}") for i in range(48)]
        calibration = [handmade_group(f"cal_{i:02}") for i in range(12)]
        valid = [m.data.Group(g.uid, g.sellers, g.items) for g in [handmade_group(f"dev_{i:02}") for i in range(60)]]
        evidence = {}
        for name in ("SEQ_ref", "LOGIT0.1_ref", "H_pos_tuned"):
            spec, method, c, model, optimizer = fixture(name, step=576)
            first = dict(a=1., b=0.)
            memory = method.new_memory(first)
            with m.diag.neutral(model):
                method.retain(memory, model, old, c, 1, dict(first=first))
            if memory is not None:
                memory.begin_stage(2)
                for _ in range(288):
                    memory.draw()
            with tempfile.TemporaryDirectory(dir=os.environ["FIVE_TEST_ROOT"]) as temporary:
                root = Path(temporary)
                for folder in ("models", "work", "memory", "points", "maps", "scores"):
                    (root/folder).mkdir()
                budget = run.Budget(root, __import__("time").monotonic()+120, tail=1)
                point, memory, first_returned = run.checkpoint(root, method, model, optimizer, memory, current,
                    calibration, valid, c, 2, first, budget)
                self.assertEqual(first, first_returned)
                self.assertTrue(point["full_restore_verified"] and point["inference_restore_verified"])
                self.assertFalse((root/point["full_state"]["path"]).exists())
                self.assertTrue((root/point["model"]["path"]).exists())
                if memory is not None:
                    self.assertEqual(memory.reservoir.seen, 96)
                    memory.begin_stage(3)
                    g, teacher = memory.draw()
                    self.assertIn(g.uid, memory.summary()["members"])
                    self.assertIsNotNone(teacher)
                else:
                    self.assertIsNone(point["memory"])
                evidence[name] = dict(actual_stage2_endpoint=True, restored_predictions_equal=True,
                    first_calibration_preserved=True, full_retired_after_verification=True,
                    stage3_history_ready=memory is not None)
        self.evidence("actual_endpoint", evidence)

    def test_confirmation_restore_cells_and_joint_numeric_labels(self):
        # Exercise the new adapter on two representatives and handwritten groups.
        # The separate 45-ID test covers global admission. No generator is called.
        specs = [m.Spec("SEQ_ref", "BCA", 4), m.Spec("H_pos_tuned", "CAB", 5)]
        groups = [handmade_group(f"confirmation_{d}_{i:02}") for d in "ABC" for i in range(40)]
        with tempfile.TemporaryDirectory(dir=os.environ["FIVE_TEST_ROOT"]) as temporary:
            root = Path(temporary)
            generated, job = root/"generation", root/"confirmation"
            generated.mkdir()
            job.mkdir()
            with (generated/"groups.jsonl").open("w") as public, (generated/"pairs.jsonl").open("w") as pairs:
                for n, group in enumerate(groups):
                    items = [dict(seller_uid=seller, item_uid=item, title=title, description=description)
                             for seller, records in zip(group.sellers, group.items)
                             for item, title, description in records]
                    public.write(json.dumps(dict(metadata=dict(group_uid=group.uid, domain="ABC"[n//40]), items=items))+"\n")
                    rows = [dict(group_uid=group.uid, seller_uid_left=a, seller_uid_right=b, label=int(label))
                            for (a, b), label in zip(itertools.combinations(group.sellers, 2), group.labels)]
                    pairs.write(json.dumps(rows)+"\n")
            m.data.write_json(generated/"result.json", dict(status="GENERATED_VALIDATED_OWNERS_SEALED",
                fixture="handwritten only; no generation or owner file", files={p.name: m.data.record(p, generated)
                    for p in generated.glob("*.jsonl")}))
            records, expected = [], {}
            for spec in specs:
                _, method, c, model, _ = fixture(spec.name, spec.order, spec.seed)
                run_root = root/spec.run_id/"run"
                run_root.mkdir(parents=True)
                points = {}
                for stage in (1, 2, 3):
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.add_(.013)
                    maps = dict(stage=dict(a=.7+stage*.1, b=-.2*stage), first=dict(a=.8, b=-.2))
                    meta = dict(**method.metadata(), stage=stage, adam_step=288*stage, maps=maps,
                                rng=m.diag.rng_state(), modes=[v.training for v in model.modules()])
                    path = run_root/f"stage{stage}.pt"
                    rec = m.core.save_state(path, model, None, meta)
                    points[str(stage)] = dict(model=dict(rec, path=path.name), maps=maps)
                    for position in (1, 2, 3):
                        domain = spec.order[position-1]
                        selected = groups[40*"ABC".index(domain):40*("ABC".index(domain)+1)]
                        with m.diag.neutral(model):
                            expected[spec.run_id, stage, position] = method.score(model, selected, c)
                records.append(dict(spec=spec.identity(), points=points))
            def cpu_fresh(method, config, device="cuda:0"):
                return fixture(method.spec.name, method.spec.order, method.spec.seed)[3:]
            budget = run.Budget(root, __import__("time").monotonic()+300, tail=1)
            with mock.patch.object(m, "specifications", return_value=specs), \
                 mock.patch.object(m.Method, "fresh", cpu_fresh), \
                 mock.patch.object(torch, "set_num_interop_threads"), \
                 mock.patch.object(torch.cuda, "set_per_process_memory_fraction"), \
                 mock.patch.object(torch.cuda, "get_device_properties", return_value=type("Device", (), {"total_memory": 2**35})()):
                evaluation.confirmation(root, job, records, budget, dict(allocator_bytes=2**30))
            result = m.data.read_json(job/"result.json")
            self.assertEqual(result["forwards"], 560)
            for spec in specs:
                for cell, point in result["runs"][spec.run_id].items():
                    stage, position = map(int, cell.split("_"))
                    actual = np.load(job/point["scores"]["raw"]["path"])
                    np.testing.assert_array_equal(actual, expected[spec.run_id, stage, position])
                    mapping = records[specs.index(spec)]["points"][str(stage)]["maps"]["stage"]
                    np.testing.assert_allclose(np.load(job/point["scores"]["stage-cal"]["path"]),
                                               actual.astype(np.float64)*mapping["a"]+mapping["b"], rtol=1e-12, atol=1e-12)
            events = m.data.read_json(root/"evaluation_access.json")["events"]
            self.assertEqual([v["role"] for v in events], ["new_confirmation_joint"])
            self.assertTrue((job/"all_blind_complete.json").is_file())
            self.assertFalse((generated/"owners.jsonl").exists())
        self.evidence("confirmation_adapter", dict(handwritten_groups=120, representative_methods=2,
            actual_model_restores=6, actual_group_forwards=560, exact_stage_domain_scores=True,
            numeric_json_label_alignment=True, all_cells_before_one_label_open=True, owner_file_absent=True,
            cpu_encoder_substituted=True, formal_generation_and_data_used=False))
