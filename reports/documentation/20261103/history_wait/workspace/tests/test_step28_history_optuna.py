"""Delta acceptance: handwritten CPU graphs, actual processes and actual Optuna."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

import step28_history_optuna as m
import step28_history_optuna_run as run
import step28_history_optuna_control as control
import step28_history_optuna_evaluate as evaluate
from test_step28_record_replay import handmade_group, TinyEncoder
from test_step28_positive_ranking import independent_pool, independent_loss, independent_rank


def reference(table, group, seed, spec, target=None):
    slots = [m.five.record.assignment(group), m.five.record.assignment(group, seed)]
    values = [independent_pool(table, a, "H_pos") for a in slots]
    losses = [independent_loss(v, group)+(spec.lambda_rank-1)*independent_rank(v, group.labels) for v in values]
    supervised = sum(losses)/2
    if target is None:
        return supervised
    teacher = independent_pool(torch.as_tensor(target, dtype=table.dtype), slots[0], "H_pos")
    return spec.history_weight*supervised+spec.teacher_weight*(values[0]-teacher).square().mean()


def fixture(spec, step=624):
    method, c = m.Method(spec), m.config(spec)
    torch.manual_seed(41)
    model = m.core.build_model(TinyEncoder(), 8, 5)
    m.five.checkpoint_encoder(model.encoder)
    method.bind(model, method.metadata())
    optimizer = m.core.make_optimizer(model, c)
    for p in model.parameters():
        if step:
            optimizer.state[p] = dict(step=torch.tensor(float(step)), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
    return method, c, model, optimizer


class HistoryTests(unittest.TestCase):
    def evidence(self, name, value):
        m.data.write_json(Path(os.environ["HISTORY_TEST_ROOT"])/(name+".json"), value)

    def test_slot_credit_counterexamples_and_actual_dispatch(self):
        outcomes = []
        with tempfile.TemporaryDirectory(dir=os.environ["HISTORY_TEST_ROOT"]) as temp:
            root = Path(temp)
            limits = {**run.LIMITS, "rss_bytes": 4*2**30, "disk_bytes": 2**30, "free_reserve_bytes": 0}
            allowed = dict(workers=3, sources=[], campaign=str(root), workspace=str(root), limits=limits)
            s = run.LIMITS["process_gpu_bytes"]
            mixed = dict(free=16*2**30, processes={123: 8*2**30})
            self.assertTrue(run.queue.admission(mixed, [123], s)["ready"])
            self.assertFalse(control.slot_admission(allowed, mixed, 16*2**30, 1)["ready"])
            # A larger observed free/zero usage cannot return credit while a group lives.
            self.assertFalse(control.slot_admission(allowed, dict(free=32*2**30, processes={}), 16*2**30, 1)["ready"])
            self.assertTrue(control.slot_admission(allowed, mixed, 16*2**30, 0)["ready"])
            # The frozen credit does not replace the latest free-space requirement.
            self.assertFalse(control.slot_admission(allowed, dict(free=8*2**30, processes={}), 32*2**30, 0)["ready"])
            for occupied in range(3):
                self.assertTrue(control.slot_admission(allowed, mixed, 32*2**30, occupied)["ready"])
            self.assertFalse(control.slot_admission(allowed, mixed, 32*2**30, 3)["ready"])
            for capacity, expected_peak in ((16*2**30, 1), (32*2**30, 3)):
                campaign = root/str(expected_peak)
                campaign.mkdir()
                local = {**allowed, "campaign": str(campaign)}
                ledger = control.Ledger(campaign, dict(training=0., auxiliary=0.), dict(training=100., auxiliary=100.))
                rows = [dict(id=f"r{i}", bucket="training", seconds=10., shutdown_reserve=.1, gpu=True) for i in range(4)]
                released = []
                actual_release = control.release
                def release(process):
                    time.sleep(.02)
                    result = actual_release(process)
                    released.append(dict(pid=process.pid, at=time.monotonic()))
                    return result
                class HandwrittenMonitor:
                    def snapshot(self):
                        return dict(free=capacity, processes={p.pid: 8*2**30 for p in control.psutil.Process().children()})
                def command(row, job, deadline):
                    code = "import json,time; from pathlib import Path; start=time.monotonic(); time.sleep(.2); Path('"+str(job/"result.json")+"').write_text(json.dumps(dict(start=start,end=time.monotonic())))"
                    return [sys.executable, "-c", code]
                with mock.patch.object(control, "release", side_effect=release):
                    result = control.dispatch(campaign, rows, command, lambda row, job: m.data.read_json(job/"result.json"),
                        local, ledger, monitor=HandwrittenMonitor(), poll_seconds=.01)
                intervals = [m.data.read_json(campaign/f"r{i}/result.json") for i in range(4)]
                peak = max(sum(p["start"] <= r["start"] < p["end"] for p in intervals) for r in intervals)
                self.assertEqual(peak, expected_peak)
                self.assertGreaterEqual(intervals[expected_peak]["start"], released[0]["at"])
                self.assertEqual(control.group_members(released[-1]["pid"]), [])
                outcomes.append(dict(capacity=capacity, peak=peak, intervals=intervals, released=released, dispatch=result))
        self.evidence("slot_credit", dict(mixed_observation_old_ready_new_rejected=True,
            live_group_credit_never_refunded=True, latest_free_still_required=True,
            actual_cpu_dispatch=outcomes, gpu_calls=0))

    def test_close_failure_settles_and_finite_tail_is_precharged(self):
        with tempfile.TemporaryDirectory(dir=os.environ["HISTORY_TEST_ROOT"]) as temp:
            root = Path(temp)
            with mock.patch.object(control.time, "monotonic", side_effect=[100., 107., 109.]):
                ledger = control.Ledger(root, dict(training=0., auxiliary=89.), dict(training=100., auxiliary=200.))
                self.assertEqual(ledger.used["auxiliary"], 119.)
                monitor = mock.Mock()
                monitor.close.side_effect = RuntimeError("handwritten close failure")
                with self.assertRaisesRegex(RuntimeError, "handwritten close failure"):
                    control.close_and_settle(monitor, ledger)
                self.assertEqual(ledger.used["auxiliary"], 126.)
                control.close_and_settle(None, ledger)
                self.assertEqual(ledger.used["auxiliary"], 128.)
            saved = m.data.read_json(root/"budget.json")
            self.assertEqual(saved["used_seconds"]["auxiliary"], 89.+9.+30.)
            self.assertEqual(len(saved["events"]), 1)
            self.assertEqual(saved["events"][0]["conservative_seconds"], 30)
            # A handwritten 12-second terminal write fits the already charged 30.
            self.assertGreaterEqual(saved["used_seconds"]["auxiliary"], 89.+9.+12.)
            short = control.Ledger(root, dict(training=0., auxiliary=89.), dict(training=100., auxiliary=110.))
            with self.assertRaises(TimeoutError):
                short.control()
        self.evidence("final_accounting", dict(close_exception_preserved=True, failed_close_still_settles=True,
            one_conservative_tail_seconds=30, existing_auxiliary_cap_enforced=True,
            handwritten_elapsed_seconds=9, handwritten_terminal_seconds=12, saved=saved))

    def test_real_coefficients_and_anchor_equivalence(self):
        group = handmade_group("current")
        target = (np.sin(np.arange(1540)/7)*3).astype(np.float32)
        table = torch.tensor(np.sin(np.arange(1540)*.3), dtype=torch.float64, requires_grad=True)
        anchor = m.Spec("replication", 0, "ABC", 4, **m.ANCHOR)
        changed = m.Spec("replication", 1, "ABC", 4, history_weight=.23, teacher_weight=.83)
        original = m.five.tuned.Method(m.five.Spec("H_pos_tuned", "ABC", 4))
        old_values, new_values = [], []
        evidence = {}
        for spec in (anchor, changed):
            for teacher in (None, target):
                actual, log = m.Method(spec).tuned.objective(table, group, 17, spec.arm, teacher)
                expected = reference(table, group, 17, spec, teacher)
                ga, = torch.autograd.grad(actual, table, retain_graph=True)
                ge, = torch.autograd.grad(expected, table, retain_graph=True)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-12)
                if spec == anchor:
                    old, _ = original.objective(table, group, 17, "H_pos", teacher)
                    torch.testing.assert_close(actual, old, atol=0, rtol=0)
                    old_values.append(float(actual.detach()))
                else:
                    new_values.append(float(actual.detach()))
                if teacher is not None:
                    mutants = dict(no_history=spec.teacher_weight*table.new_tensor(0),
                        no_teacher=spec.history_weight*reference(table, group, 17, spec),
                        old_coefficients=reference(table, group, 17, anchor, teacher))
                    # Teacher-only mutant keeps a differentiable, correct teacher operation.
                    mutants["no_history"] = actual-spec.history_weight*reference(table, group, 17, spec)
                    for name, value in mutants.items():
                        if name == "old_coefficients" and spec == anchor:
                            continue
                        gm, = torch.autograd.grad(value, table, retain_graph=True)
                        error = float((ga-gm).abs().max())
                        self.assertGreater(error, 1e-7)
                        evidence[f"{spec.number}_{name}"] = error
        self.assertEqual(old_values[0], new_values[0])
        self.assertNotEqual(old_values[1], new_values[1])
        self.evidence("objectives", dict(gradient_mutants=evidence, current_weight_unchanged=True, anchor_matches_original=True))

    def test_actual_update_restore_and_parameter_identity(self):
        spec = m.Spec("replication", 1, "ABC", 4, history_weight=.23, teacher_weight=.83)
        method, c, model, optimizer = fixture(spec)
        current, history = handmade_group("current"), handmade_group("history")
        target = (np.sin(np.arange(1540)/7)*3).astype(np.float32)
        direct = copy.deepcopy(model)
        direct.encoder._five_outer_checkpoint = False
        other = m.core.make_optimizer(direct, c)
        other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        meta = {**method.metadata(), "rng": m.diag.rng_state(), "modes": [v.training for v in model.modules()]}
        with tempfile.TemporaryDirectory(dir=os.environ["HISTORY_TEST_ROOT"]) as temp:
            path = Path(temp)/"full.pt"
            rec = m.core.save_state(path, model, optimizer, meta)
            actual = method.update(model, optimizer, current, history, target, c, 3, 49)
            direct.train()
            other.zero_grad(set_to_none=True)
            for role, group, teacher in (("current", current, None), ("history", history, target)):
                torch.manual_seed(m.data.seed_for(spec.roots["schedule"], spec.order, 3, 49, role, "dropout"))
                z = m.five.record.record_vectors(direct, group, c)
                i, j = torch.triu_indices(len(z), len(z), 1)
                table = direct.head(torch.cat(((z[i]-z[j]).abs(), z[i]*z[j]), 1)).flatten()
                seed = m.data.seed_for(spec.roots["schedule"], spec.order, 3, 49, role, group.uid, "regroup")
                reference(table, group, seed, spec, teacher).backward()
            torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.)
            for group, rate in zip(other.param_groups, spec.rates(49), strict=True):
                group["lr"] = rate
            other.step()
            changes, errors = {}, []
            for (name, p), q in zip(model.named_parameters(), direct.parameters(), strict=True):
                torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(optimizer.state[p][field], other.state[q][field], atol=2e-6, rtol=2e-5)
                errors.append(float((p-q).detach().abs().max()))
                changes[name] = float((p.detach()-before[name]).abs().max())
            for prefix in ("encoder.", "head."):
                self.assertTrue(any(value > 0 for name, value in changes.items() if name.startswith(prefix)))
            digest = m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
            method.restore(path, rec, model, optimizer)
            repeated = method.update(model, optimizer, current, history, target, c, 3, 49)
            self.assertEqual(actual, repeated)
            self.assertEqual(digest, m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict())))
            wrong = m.Method(m.Spec("replication", 1, "ABC", 4, history_weight=.22, teacher_weight=.83))
            with self.assertRaises(ValueError):
                wrong.restore(path, rec, model, optimizer)
            payload = method.new_memory(dict(a=1., b=0.)).to_bytes()
            self.assertEqual(method.memory_from(payload).to_bytes(), payload)
            with self.assertRaises(ValueError):
                wrong.memory_from(payload)
        method, c, model, optimizer = fixture(spec, step=0)
        first = method.update(model, optimizer, current, None, None, c, 1, 1)
        self.assertNotIn("history", first)
        self.assertEqual(m.parent.adam_step(optimizer), 1)
        self.evidence("updates", dict(production_optimizer_steps=3, independent_optimizer_steps=1,
            maximum_parameter_error=max(errors), changes=changes, full_restore_repeat_exact=True,
            foreign_coefficients_rejected=True, formal_model_loads=0))

    def test_three_process_workers_and_healthy_slot_continuation(self):
        outcomes = []
        for failure in (False, True):
            with tempfile.TemporaryDirectory(dir=os.environ["HISTORY_TEST_ROOT"]) as temp:
                root = Path(temp)
                limits = {**run.LIMITS, "rss_bytes": 4*2**30, "disk_bytes": 2**30, "free_reserve_bytes": 0}
                allowed = dict(workers=3, sources=[], campaign=str(root), workspace=str(root), limits=limits)
                ledger = control.Ledger(root, dict(training=0., auxiliary=0.), dict(training=100., auxiliary=100.))
                rows = [dict(id=f"r{i}", bucket="training", seconds=10., shutdown_reserve=.1, gpu=False) for i in range(9)]
                def command(row, job, deadline):
                    fail = failure and row["id"] == "r0"
                    code = "import json,time; from pathlib import Path; start=time.monotonic(); time.sleep(.3); Path('"+str(job/"result.json")+"').write_text(json.dumps(dict(start=start,end=time.monotonic()))); raise SystemExit("+str(int(fail))+")"
                    return [sys.executable, "-c", code]
                def validate(row, job):
                    m.data.read_json(job/"result.json")
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "Incomplete batch"):
                        control.dispatch(root, rows, command, validate, allowed, ledger, poll_seconds=.02)
                else:
                    control.dispatch(root, rows, command, validate, allowed, ledger, poll_seconds=.02)
                receipt = m.data.read_json(root/"dispatch.json")
                self.assertEqual(len(receipt["completed"]), 9)
                self.assertTrue(all(r["released"] for r in receipt["completed"]))
                times = [m.data.read_json(root/f"r{i}/result.json") for i in range(3)]
                self.assertLess(max(r["start"] for r in times), min(r["end"] for r in times))
                if failure:
                    self.assertEqual(receipt["retired_slots"], [0])
                    self.assertEqual([r["id"] for r in receipt["completed"] if r["slot"] == 0], ["r0"])
                    self.assertEqual(sum(r["success"] for r in receipt["completed"]), 8)
                else:
                    self.assertEqual(set(r["slot"] for r in receipt["completed"]), {0, 1, 2})
                self.assertGreater(ledger.used["training"], max(r["end"] for r in times)-min(r["start"] for r in times))
                outcomes.append(dict(failure=failure, dispatch=receipt, fees=ledger.used))
        # An exited leader with a surviving child must not count as successful release.
        child = subprocess.Popen([sys.executable, "-c", "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])"], start_new_session=True)
        child.wait(timeout=5)
        self.assertTrue(control.release(child))
        self.assertEqual(control.group_members(child.pid), [])
        self.evidence("three_workers", dict(batches=outcomes, orphan_detected_and_released=True, gpu=False))

    def test_actual_optuna_finite_sequence_selection_and_failed_candidate(self):
        search = m.Search()
        for number in range(m.TRIALS):
            specs = search.ask()
            if number == 0:
                self.assertEqual(specs[0].parameters, m.ANCHOR)
            with self.assertRaises(ValueError):
                search.ask()
            feedback = dict(status="THREE_ORDERS_VERIFIED_DEV_COMPLETE", specs=[s.identity() for s in specs],
                final_map_by_order={o: .5 if number != 5 else .6 for o in m.ORDERS}, result_identity={"handwritten_unit_fixture": True}, all_workers_released=True)
            incomplete = copy.deepcopy(feedback)
            incomplete["final_map_by_order"].pop("CAB")
            with self.assertRaises(ValueError):
                search.tell(incomplete)
            search.tell(feedback)
        self.assertEqual(search.selected()["number"], 5)
        with self.assertRaises(ValueError):
            search.ask()
        tied = copy.deepcopy(search.rows)
        for row in tied:
            row["value"] = .5
        self.assertEqual(m.select_trial(tied)["number"], 0)
        failed = m.Search()
        failed.ask()
        failed.fail("handwritten failure")
        with self.assertRaises(ValueError):
            failed.selected()
        self.evidence("optuna", dict(rows=search.rows, chosen=5, tied_anchor=0, no_partial_feedback=True,
            synthetic_feedback_only=True, formal_training=0))

    def test_closed_entries_before_data_or_outputs(self):
        with tempfile.TemporaryDirectory(dir=os.environ["HISTORY_TEST_ROOT"]) as temp:
            root = Path(temp)
            gate = root/"gate.json"
            m.data.write_json(gate, dict(experiment="HISTORY_OPTUNA", approved=False, workers=3))
            with mock.patch.object(run, "sources", side_effect=AssertionError("sources should not be touched")):
                for action in run.ACTIONS:
                    with self.assertRaisesRegex(ValueError, "formal execution is closed"):
                        run.gate(gate, action)
            with self.assertRaisesRegex(ValueError, "formal execution is closed"):
                control.execute(gate)
            self.assertEqual(list(root.iterdir()), [gate])
        self.evidence("closed_gate", dict(all_actions_rejected=True, no_formal_inputs=True, cuda_available_not_queried=True))

    def test_saved_raw_map_binding_and_missing_confirmation_cell(self):
        with tempfile.TemporaryDirectory(dir=os.environ["HISTORY_TEST_ROOT"]) as temp:
            root = Path(temp)
            allowed = dict(campaign=str(root), sources=[], _gate_sha256="handwritten_gate")
            specs = m.specifications("search", 0, m.ANCHOR)
            batch = root/"batch.json"
            m.data.write_json(batch, dict(status="APPROVED_IMPLEMENTATION_AND_RESOURCES", sources=[],
                runs=[s.identity() for s in specs], allocator_bytes=run.LIMITS["allocator_bytes"], main_gate_sha256="handwritten_gate"))
            job = root/"evaluation"
            job.mkdir()
            ids = [f"handwritten_{d}_{i}" for d in "ABC" for i in range(20)]
            domains = [d for d in "ABC" for _ in range(20)]
            labels = np.asarray([handmade_group("fixture").labels]*20, np.uint8)
            scores = np.asarray([np.cos(np.arange(378)/17)]*20)
            raw_rec = evaluate.save_array(root/"raw.npy", np.tile(scores, (3, 1)), root)
            records = [dict(name=m.NAME, spec=s, root=root, record=dict(partition=dict(development=[dict(group_uid=uid, domain=d) for uid, d in zip(ids, domains)]),
                points={"3": dict(development_ids=ids, scores={"raw": raw_rec})})) for s in specs]
            result = dict(status="ALL_DEVELOPMENT_METRICS_COMPLETE", sources=[], groups=ids, domains=domains, runs={})
            expected, _ = m.parent.metrics.group_metrics(labels, scores)
            for item in records:
                s = item["spec"]
                cells = {}
                for position in (1, 2, 3):
                    domain = s.order[position-1]
                    metric = evaluate.prior.metric_set(job/f"{s.order}_{position}", labels, scores)
                    cells[f"3_{position}"] = dict(domain=domain, ids=[uid for uid, d in zip(ids, domains) if d == domain],
                        metrics={"raw": metric}, scores={"raw": run.queue.record(root/"raw.npy")})
                result["runs"][evaluate.identity(item)] = dict(spec=s.identity(), cells=cells)
            m.data.write_json(job/"result.json", result)
            # This test supplies handwritten completion metadata only; it exercises the
            # real saved metric/query/RAW binding, not qualification of a training run.
            with mock.patch.object(evaluate, "complete_batch", return_value=records):
                feedback = evaluate.verified_feedback(job, batch, allowed)
                for value in feedback["final_map_by_order"].values():
                    self.assertAlmostEqual(value, expected[:, list(m.parent.metrics.COLUMNS).index("map")].mean(), places=12)
                first = result["runs"][evaluate.identity(records[0])]["cells"]["3_1"]
                first["domain"] = "B"
                m.data.write_json(job/"result.json", result)
                with self.assertRaisesRegex(ValueError, "RAW score binding"):
                    evaluate.verified_feedback(job, batch, allowed)
                first["domain"] = "A"
                first["scores"]["raw"]["sha256"] = "0"*64
                m.data.write_json(job/"result.json", result)
                with self.assertRaisesRegex(ValueError, "RAW score binding"):
                    evaluate.verified_feedback(job, batch, allowed)
            rep_specs = m.specifications("replication", 1, dict(history_weight=.2, teacher_weight=.7))
            reps = [dict(name="H_pos_tuned", spec=m.five.Spec("H_pos_tuned", s.order, s.seed)) for s in rep_specs]
            reps += [dict(name=m.NAME, spec=s) for s in rep_specs]
            confirmation_ids = [f"fake_confirm_{d}_{i}" for d in "ABC" for i in range(40)]
            saved = dict(groups=confirmation_ids, domains=[d for d in "ABC" for _ in range(40)], forwards=4320, runs={})
            for item in reps:
                spec = item["spec"]
                saved["runs"][evaluate.identity(item)] = dict(spec=spec.identity(), cells={f"{stage}_{pos}":
                    dict(domain=spec.order[pos-1], ids=[uid for uid in confirmation_ids if f"_{spec.order[pos-1]}_" in uid],
                        scores={r: {} for r in m.parent.ROLES}) for stage, pos in evaluate.CELLS})
            evaluate.validate_cells(saved, reps)
            saved["runs"][evaluate.identity(reps[0])]["cells"].pop("2_1")
            with self.assertRaises(ValueError):
                evaluate.validate_cells(saved, reps)
        self.evidence("saved_metrics", dict(actual_metric_query_correspondence=True, changed_domain_rejected=True,
            foreign_raw_hash_rejected=True, missing_blind_cell_rejected=True, fake_metadata_only=True,
            formal_label_reads=0, generated_groups=0))

    def test_full_previous_chain_and_action_dispatch_index(self):
        import step28_formal_wait as wait
        with tempfile.TemporaryDirectory(dir=os.environ["HISTORY_TEST_ROOT"]) as temp:
            root = Path(temp)
            state = root/"state"
            state.mkdir()
            binding = dict(state=str(state))
            process = dict(owner="handwritten_owner")
            for key, experiment, count, extras in (("forever_gate", "FOREVER", 9, ["generation", "confirmation", "analysis"]),
                                                  ("vendor_gate", "VendorLink", 36, ["scoring", "analysis"])):
                campaign = root/experiment
                campaign.mkdir()
                runs = ([dict(order="ABC", seed=i, name="fixture") for i in range(count)] if count == 9
                        else [dict(id=f"vendor_{i}") for i in range(count)])
                ids = [f'ABC_s{i}_fixture' if count == 9 else f'vendor_{i}' for i in range(count)]+extras
                gate = root/(key+".json")
                m.data.write_json(gate, dict(experiment=experiment, campaign=str(campaign), runs=runs, sources=[]))
                binding[key] = process[key] = run.queue.record(gate)
                m.data.write_json(campaign/"completion.json", dict(status="ALL_FORMAL_ACTIONS_COMPLETE", sources=[],
                    completed=[dict(id=i, success=True) for i in ids], active={}, used=dict(control=1.), limits=dict(control=2.)))
            m.data.write_json(state/"process.json", process)
            binding["process"] = run.queue.record(state/"process.json")
            with mock.patch.object(wait, "relevant_processes", return_value=[]):
                with self.assertRaisesRegex(ValueError, "whole FOREVER/VendorLink"):
                    control.chain_complete(binding)
                m.data.write_json(state/"completion.json", dict(status="BOTH_FORMAL_BATCHES_COMPLETE"))
                control.chain_complete(binding)
                with mock.patch.object(wait, "relevant_processes", return_value=[dict(pid=123)]):
                    with self.assertRaisesRegex(ValueError, "has not exited"):
                        control.chain_complete(binding)
                old = m.data.read_json(root/"VendorLink/completion.json")
                old["completed"].pop()
                m.data.write_json(root/"VendorLink/completion.json", old)
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    control.chain_complete(binding)
            # A real followup process must preserve the training completion gate.
            original = b'{"status":"HANDWRITTEN_TRAINING_COMPLETE"}\n'
            (root/"dispatch.json").write_bytes(original)
            index = root/"evaluation_dispatch"
            index.mkdir()
            allowed = dict(workers=3, sources=[], campaign=str(root), workspace=str(root), dispatch_index=str(index),
                limits={**run.LIMITS, "rss_bytes": 4*2**30, "disk_bytes": 2**30, "free_reserve_bytes": 0})
            ledger = control.Ledger(root, dict(auxiliary=0.), dict(auxiliary=100.))
            control.dispatch(root, [dict(id="evaluation", bucket="auxiliary", seconds=5., shutdown_reserve=.1, gpu=False)],
                lambda row, job, end: [sys.executable, "-c", "print('handwritten followup')"], lambda row, job: None,
                allowed, ledger, poll_seconds=.02)
            self.assertEqual((root/"dispatch.json").read_bytes(), original)
            self.assertEqual(m.data.read_json(index/"dispatch.json")["status"], "ALL_COMPLETE_RELEASED")
        self.evidence("previous_chain", dict(both_batches_required=True, active_process_rejected=True,
            incomplete_vendor_rejected=True, action_preserves_training_gate=True, handwritten_only=True))

    def test_all_metrics_summary_and_fixed_conditional_interval(self):
        with tempfile.TemporaryDirectory(dir=os.environ["HISTORY_TEST_ROOT"]) as temp:
            root = Path(temp)
            labels = np.asarray([handmade_group("fixture").labels]*40, np.uint8)
            old_scores = np.full((40, 378), -2.)  # No predicted positives: undefined Precision/MCC.
            new_scores = np.where(labels, 3., -3.)
            metrics = {name: evaluate.prior.metric_set(root/name, labels, scores)
                for name, scores in zip(evaluate.NAMES, (old_scores, new_scores))}
            specs = m.specifications("replication", 1, dict(history_weight=.2, teacher_weight=.7))
            saved = dict(runs={})
            for name in evaluate.NAMES:
                for spec in specs:
                    spec = m.five.Spec("H_pos_tuned", spec.order, spec.seed) if name == evaluate.NAMES[0] else spec
                    item = dict(name=name, spec=spec)
                    saved["runs"][evaluate.identity(item)] = dict(name=name, spec=spec.identity(), cells={f"{stage}_{pos}":
                        dict(domain=spec.order[pos-1], metrics={r: metrics[name] for r in m.parent.ROLES}) for stage, pos in evaluate.CELLS})
            budget = type("Budget", (), {"check": lambda self: None})()
            result = evaluate.summarize(root, saved, budget, confirmation=True)
            self.assertTrue(result["primary_final_map_improved"])
            index = list(m.parent.metrics.COLUMNS).index("map")
            self.assertAlmostEqual(result["absolute"][m.NAME]["raw"]["final_mean"][index], 1.)
            difference = result["paired_intervals"]["raw"]["final_mean"]["map"]
            self.assertAlmostEqual(difference["conditional_interval"][0], difference["mean"])
            self.assertAlmostEqual(difference["conditional_interval"][1], difference["mean"])
            key = "ABC_s4_H_pos_tuned_raw_3_1"
            cell = result["per_cell"][key]
            self.assertIsNone(cell["pooled_precision"])
            self.assertIsNone(cell["pooled_mcc"])
            self.assertEqual(cell["counts"], dict(tp=0, tn=14320, fp=0, fn=800))
            for name in evaluate.NAMES:
                for role in m.parent.ROLES:
                    fields = result["absolute"][name][role]
                    self.assertAlmostEqual(fields["F"][index], fields["A_acq"][index]-fields["O"][index])
        self.evidence("summary", dict(all_22_metrics=True, fixed_5000_paired_interval=True,
            independently_known_perfect_map=1., undefined_precision_mcc_preserved=True,
            formal_results=False, generated_groups=0))

    def test_actual_confirmation_restore_score_then_single_label_opening(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory(dir=os.environ["HISTORY_TEST_ROOT"]) as temp:
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
            m.data.write_json(generation/"result.json", dict(namespace=evaluate.NAMESPACE, root_seed=evaluate.CONFIRM_SEED,
                sources=[], files={"pairs.jsonl": m.data.record(generation/"pairs.jsonl", generation)}))
            def fresh(method, c):
                torch.manual_seed(method.spec.seed)
                model = m.core.build_model(TinyEncoder(), 8, 5)
                m.five.checkpoint_encoder(model.encoder)
                method.bind(model, method.metadata())
                return model, m.core.make_optimizer(model, c)
            records = []
            specs = m.specifications("replication", 1, dict(history_weight=.2, teacher_weight=.7))
            for name in evaluate.NAMES:
                for spec in specs:
                    spec = m.five.Spec("H_pos_tuned", spec.order, spec.seed) if name == evaluate.NAMES[0] else spec
                    method = m.five.Method(spec) if name == evaluate.NAMES[0] else m.Method(spec)
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
                if role == "history_confirmation_joint":
                    blind = m.data.read_json(job/"all_blind_complete.json")
                    evaluate.validate_cells(blind, records)
                    self.assertEqual(blind["forwards"], 4320)
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
            self.assertEqual(result["status"], "ALL_HISTORY_CONFIRMATION_METRICS_COMPLETE")
            self.assertEqual(accesses, ["history_blind_inference", "history_confirmation_joint"])
            self.assertEqual(len(result["runs"]), 18)
        self.evidence("actual_confirmation", dict(actual_tiny_model_restores=54, actual_handwritten_group_forwards=4320,
            actual_metric_matrices=324, label_openings_after_all_blind=1, native_model=False, gpu=False,
            formal_data=False, generator_calls=0, substitutes=["TinyEncoder fresh", "handwritten public groups", "CUDA setup only"]))


if __name__ == "__main__":
    if sys.platform != "linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Only the bounded Linux CPU handwritten qualification is permitted")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main()
