"""Targeted handwritten Linux-only evidence for the new FOREVER behavior."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import step28_forever as m
import step28_forever_run as run
import step28_forever_qualification as qualification
from test_step28_continual_comparison import Encoder, group

EVIDENCE = {}


def fixture():
    spec = m.Spec('ABC', 4)
    torch.manual_seed(7)
    model = m.core.build_model(Encoder(), 8, 5)
    m.five.checkpoint_encoder(model.encoder)
    c = dict(input=dict(microbatch=16, token_budget=256), schedule_seed=2026102904,
             optimizer=dict(encoder_lr=1e-5, head_lr=.001, encoder_weight_decay=.01,
                            head_weight_decay=0., betas=[.9, .999], eps=1e-8, foreach=False))
    return m.Learner(spec, model, m.core.make_optimizer(model, c), c)


def dispatcher(spec):
    values, partition = dict(fit=[], calibration=[]), dict(fit=[], calibration=[])
    for d, domain in enumerate('ABC'):
        for role, count, start in (('fit', 48, d*60), ('calibration', 12, d*60+48)):
            for i in range(start, start+count):
                g = group(i)
                values[role].append(g)
                partition[role].append(dict(group_uid=g.uid, domain=domain))
    return run.Supply(m.parent.Supply(values, partition), spec)


def stage2_fixture():
    learner = fixture()
    learner.update(group(300), 'fixture')  # A real preparatory Adam update, never a forged 288.
    learner.anchor = m.snapshot(learner.model)
    learner.old.add_stage([group(i) for i in range(48)])
    learner.stage = 2
    supply = dispatcher(learner.spec)
    # Explicit synthetic stage-2 data boundary for qualification only.
    supply.supply.next_stage[learner.spec.run_id] = 2
    current, calibration = supply.current(2)
    return learner, supply, current, calibration


class ForeverTests(unittest.TestCase):
    def test_proven_unused_bert_pooler_kept_in_full_parameter_set(self):
        import inspect
        from sentence_transformers.models import Transformer, Pooling, Normalize
        from transformers import BertConfig, BertModel
        tr = Transformer.__new__(Transformer)
        torch.nn.Module.__init__(tr)
        tr.model = BertModel(BertConfig(vocab_size=16, hidden_size=8,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=16))
        tr.modality_config = {'text': dict(method='forward', method_output_name='last_hidden_state')}
        tr.module_output_name = 'token_embeddings'
        tr.model_forward_params = set(inspect.signature(tr.model.forward).parameters)
        model = m.core.build_model(torch.nn.Sequential(tr, Pooling(8, pooling_mode='cls'), Normalize()), 8, 5)
        inputs = dict(input_ids=torch.tensor([[1, 2, 3], [1, 4, 5]]), attention_mask=torch.ones(2, 3, dtype=torch.long))
        embeddings = model.encoder(inputs)['sentence_embedding']
        model.head(torch.cat((embeddings, embeddings), dim=1)).sum().backward()
        values = m.parameters(model)
        missing = {n for n, p in values.items() if p.grad is None}
        self.assertEqual(missing, {'encoder.0.model.pooler.dense.weight', 'encoder.0.model.pooler.dense.bias'})
        count = len(values)
        completed = m.complete_task_gradients(model)
        self.assertEqual(set(completed), missing)
        self.assertEqual(len(m.parameters(model)), count)
        self.assertTrue(all(torch.count_nonzero(values[n].grad) == 0 for n in completed))
        anchor = m.snapshot(model)
        for n in completed:
            anchor[n].add_(.25)
        m.regularize(model, anchor, .001)
        self.assertTrue(all(torch.allclose(values[n].grad, torch.full_like(values[n], -.0005)) for n in completed))
        next(model.head.parameters()).grad = None
        with self.assertRaises(ValueError):
            m.complete_task_gradients(model)
        EVIDENCE['unused_pooler'] = dict(random_tiny_bert=True, actual_transformer_cls_forward=True,
            missing_names=completed, full_parameter_count=count, full_p_retained=True,
            task_zero_and_regularizer_reference_pass=True, broken_head_still_rejected=True)

    def test_production_configuration_without_corpus_or_model_loading(self):
        c = m.config(m.Spec('CAB', 6))
        self.assertEqual(c['initialization_seed'], 2026101906)
        self.assertEqual(c['schedule_seed'], 2026102906)
        self.assertEqual(c['input']['token_budget'], 256)
        self.assertEqual(c['interventions']['split_rank']['account_dim'], 2048)
        EVIDENCE['production_config'] = dict(initialization_seed=c['initialization_seed'],
            schedule_seed=c['schedule_seed'], input=c['input'], model_loaded=False, corpus_read=False)

    def test_clock_against_hand_count_and_last_step(self):
        c, hits = m.Clock(), []
        for t in range(1, 289):
            c.observe(1.)
            while c.due(True):
                hits.append(t)
                c.next_threshold += 1
        self.assertEqual(hits, [24, 48, 96, 168])
        self.assertEqual(2*6*len(hits)+6, 54)
        self.assertEqual(288+54, 342)
        EVIDENCE['hand_clock'] = dict(actual_trigger_steps=hits, current_steps=288, replay_group_steps=54,
                                      independent_expected=[24, 48, 96, 168])
        self.assertFalse(c.due(False))
        final = m.Clock(current=287, tau=23., tau_day=24., mu0=1., mu=1.)
        final.observe(100.)
        crossed = []
        while final.due(True):
            crossed.append(final.next_threshold)
            final.next_threshold += 1
        self.assertEqual(crossed, [0, 1, 2])
        self.assertEqual(final.current, 288)
        self.assertEqual(final.beta(), .003)
        c = m.Clock(current=24, tau=24., tau_day=24., mu0=1., mu=1.)
        earlier = c.beta()
        for _ in range(10):
            c.observe(.1)
        self.assertLess(c.beta(), earlier)  # An end event reusing the stale beta would fail.
        EVIDENCE['end_beta'] = dict(previous_event=earlier, recomputed_end=c.beta(), last_step_crossings=crossed)
        empty = m.Clock()
        with self.assertRaises(ValueError):
            for _ in range(24):
                empty.observe(0.)

    def test_actual_displacement_not_gradient_or_parameterwise_norm(self):
        learner = fixture()
        before = m.snapshot(learner.model)
        with torch.no_grad():
            list(learner.model.encoder.parameters())[0].flatten()[0] += 3
            list(learner.model.head.parameters())[0].flatten()[0] += 4
        actual = m.delta(learner.model, before)
        self.assertAlmostEqual(actual['total'], 5., places=6)
        self.assertAlmostEqual(actual['encoder'], 3., places=6)
        self.assertAlmostEqual(actual['head'], 4., places=6)
        EVIDENCE['actual_displacement'] = actual

    def test_regularizer_autograd_actual_update_and_missing_gradient(self):
        a, b = fixture(), fixture()
        anchor = m.snapshot(a.model)
        for value in anchor.values():
            value.add_(.2)
        immutable = m.core.state_digest(anchor)
        for learner in (a, b):
            learner.optimizer.zero_grad(set_to_none=True)
            m.record.backward_group(learner.model, group(), learner.c, 'R0', 111, 0, None)
        task = {n: p.grad.clone() for n, p in m.parameters(a.model).items()}
        logged = m.regularize(a.model, anchor, .001)
        loss = .001*sum(((p-anchor[n])**2).sum() for n, p in m.parameters(b.model).items())
        loss.backward()
        changed = dict(encoder=False, head=False)
        max_error = 0.
        for name, p in m.parameters(a.model).items():
            self.assertTrue(torch.allclose(p.grad, dict(b.model.named_parameters())[name].grad, atol=2e-7, rtol=2e-6))
            role = 'encoder' if name.startswith('encoder.') else 'head'
            max_error = max(max_error, float((p.grad-dict(b.model.named_parameters())[name].grad).abs().max()))
            changed[role] |= not torch.equal(p.grad, task[name])
        self.assertTrue(all(changed.values()))
        self.assertAlmostEqual(logged['penalty'], float(loss.detach()), places=6)
        for learner in (a, b):
            torch.nn.utils.clip_grad_norm_(learner.model.parameters(), 1.)
            learner.optimizer.step()
        for name, p in a.model.named_parameters():
            self.assertTrue(torch.allclose(p, dict(b.model.named_parameters())[name], atol=1e-7, rtol=1e-6))
        self.assertEqual(m.core.state_digest(anchor), immutable)
        EVIDENCE['regularizer'] = dict(gradient_max_abs_error=max_error, gradient_changed_by_role=changed,
                                      logged_penalty=logged['penalty'], autograd_penalty=float(loss.detach()),
                                      updated_models_agree=True, anchor_unchanged=True)
        next(a.model.encoder.parameters()).grad = None
        with self.assertRaises(ValueError):
            m.regularize(a.model, anchor, .001)

    def test_regularizer_crosses_real_chunk_boundary(self):
        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(1025, 1024, bias=False)
        model.head = torch.nn.Linear(1, 1)
        anchor = m.snapshot(model)
        for value in anchor.values():
            value.add_(.25)
        reference = .001*sum(((p-anchor[n])**2).sum() for n, p in m.parameters(model).items())
        reference.backward()
        expected = {n: p.grad.clone() for n, p in m.parameters(model).items()}
        for p in model.parameters():
            p.grad.zero_()
        m.regularize(model, anchor, .001)
        errors = {n: float((p.grad-expected[n]).abs().max()) for n, p in m.parameters(model).items()}
        self.assertEqual(max(errors.values()), 0.)
        EVIDENCE['multiple_chunks'] = dict(encoder_elements=model.encoder.weight.numel(), chunk_elements=1048576,
                                         per_parameter_gradient_errors=errors, final_chunk_checked=True)

    def test_real_entry_event_and_next_update_restore(self):
        learner, supply, current, _ = stage2_fixture()
        rows, physical = [], []
        old_rng = learner.old.rng.getstate()
        source_order = [g.uid for g in m.parent.schedule(current, learner.c, 'ABC', 2)[0]]
        with tempfile.TemporaryDirectory(dir=os.environ.get('FOREVER_TEST_TMP')) as temp:
            identity = dict(spec=learner.spec.identity(), config=learner.c, qualification_only=True)
            store = run.Store(Path(temp), identity)
            anchor_hash = m.core.state_digest(learner.anchor)
            for i in range(24):
                learner.advance(current[i], before_step=lambda: physical.append(1), log=rows.append,
                                commit=lambda: store.commit(learner, supply))
            self.assertEqual(learner.clock.current, 24)
            self.assertEqual(learner.clock.next_threshold, 1)
            self.assertEqual(learner.counts, dict(current=24, replay=12, fixture=1))
            self.assertEqual(len(physical), 36)
            self.assertEqual(learner.old.rng.getstate(), old_rng)
            self.assertEqual(source_order, [g.uid for g in m.parent.schedule(current, learner.c, 'ABC', 2)[0]])
            self.assertEqual(m.core.state_digest(learner.anchor), anchor_hash)
            replay = [r for r in rows if r['role'] == 'replay']
            self.assertEqual({r['group'] for r in replay}, {g.uid for g in learner.old.groups})
            self.assertTrue(any(r['regularizer']['penalty'] > 0 for r in replay))
            self.assertTrue(all(r['encoder_lr'] == 1e-5 for r in replay))
            store.commit(learner, supply)
            learner.update(current[24], 'current')
            expected = m.core.state_digest(dict(model=learner.model.state_dict(), optimizer=learner.optimizer.state_dict(),
                                               small=learner.small(), rng=m.diag.rng_state()))
            restored = fixture()
            supplied = dispatcher(restored.spec)
            reloaded = run.Store(Path(temp), identity)
            got, _ = reloaded.restore(restored, supplied)
            self.assertEqual({g.uid for g in got}, {g.uid for g in current})
            restored.update(got[24], 'current')
            actual = m.core.state_digest(dict(model=restored.model.state_dict(), optimizer=restored.optimizer.state_dict(),
                                             small=restored.small(), rng=m.diag.rng_state()))
            self.assertEqual(expected, actual)
            EVIDENCE['real_entry_and_restore'] = dict(current=24, replay=12, fixture_predecessor=1,
                next_update_branches=2, real_clock=asdict(restored.clock), replay_only_survivors=True,
                retention_rng_unchanged=True, current_schedule_unchanged=True, model_adam_rng_next_exact=True,
                final_adam_step=m.parent.adam_step(restored.optimizer), parameter_mapping=m.mapping(restored.model, restored.optimizer))

    def test_resume_drains_multiple_due_events_before_next_current(self):
        learner, supply, current, _ = stage2_fixture()
        # Boundary fixture with three due events; each event still performs real gradient updates.
        learner.clock = m.Clock(current=25, tau=5., tau_day=1., mu0=.04, mu=.04)
        rows = []
        with tempfile.TemporaryDirectory(dir=os.environ.get('FOREVER_TEST_TMP')) as temp:
            identity = dict(spec=learner.spec.identity(), config=learner.c, qualification_only=True)
            store = run.Store(Path(temp), identity)
            def first_event():
                store.commit(learner, supply)
                raise RuntimeError('interrupted after first of three due events')
            with self.assertRaises(RuntimeError):
                learner.drain_events(log=rows.append, commit=first_event)
            self.assertEqual(learner.clock.current, 25)
            self.assertEqual(learner.clock.next_threshold, 1)
            remaining = []
            learner.advance(current[25], log=remaining.append)
            expected = m.core.state_digest(dict(model=learner.model.state_dict(), optimizer=learner.optimizer.state_dict(),
                                               small=learner.small(), rng=m.diag.rng_state()))
            restored, supplied = fixture(), dispatcher(learner.spec)
            fit, _ = run.Store(Path(temp), identity).restore(restored, supplied)
            actual_rows = []
            restored.advance(fit[25], log=actual_rows.append)
            actual = m.core.state_digest(dict(model=restored.model.state_dict(), optimizer=restored.optimizer.state_dict(),
                                             small=restored.small(), rng=m.diag.rng_state()))
            self.assertEqual(expected, actual)
            self.assertEqual([r['role'] for r in actual_rows], ['replay']*24+['current'])
            self.assertEqual([r.get('event') for r in actual_rows[:24]], ['event1']*12+['event2']*12)
            self.assertEqual(restored.clock.current, 26)
            EVIDENCE['multiple_event_resume'] = dict(prior_event_replay_steps=len(rows),
                restored_sequence=[r['role'] for r in actual_rows], next_current=26,
                model_adam_clock_rng_exact=True, boundary_fixture_not_full_stage=True)

    def test_stage_transaction_before_after_publication_and_eviction(self):
        learner, supply, current, calibration = stage2_fixture()
        # Boundary fixture, explicitly not evidence of 288 real predecessor updates.
        learner.clock = m.Clock(current=288, tau=24., tau_day=24., mu0=1., mu=.7, next_threshold=1)
        with tempfile.TemporaryDirectory(dir=os.environ.get('FOREVER_TEST_TMP')) as temp:
            identity = dict(spec=learner.spec.identity(), config=learner.c, qualification_only=True)
            store = run.Store(Path(temp), identity)
            store.commit(learner, supply)
            unrelated = Path(temp)/'models'/'keep_user_note.txt'
            unrelated.write_text('not a Store generation')
            previous = (Path(temp)/'latest.json').read_bytes()
            def fail_before(phase):
                if phase == 'before_publish':
                    raise RuntimeError('handwritten interruption before publication')
            with self.assertRaises(RuntimeError):
                run.stage(learner, supply, store, current, calibration, hook=fail_before)
            self.assertEqual((Path(temp)/'latest.json').read_bytes(), previous)
            restored, supplied = fixture(), dispatcher(learner.spec)
            recovered = run.Store(Path(temp), identity)
            fit, cal = recovered.restore(restored, supplied)
            self.assertEqual(len(list((Path(temp)/'work').glob('*.pt'))), 1)
            self.assertEqual(len(list((Path(temp)/'anchors').glob('*.pt'))), 1)
            self.assertEqual(len(list((Path(temp)/'models').glob('*.pt'))), 0)
            self.assertEqual(len(list(Path(temp).glob('latest.*.json'))), 0)
            self.assertEqual(restored.stage, 2)
            self.assertEqual(restored.old.seen, 48)
            def fail_after(phase):
                if phase == 'after_publish':
                    raise RuntimeError('handwritten interruption after publication')
            with self.assertRaises(RuntimeError):
                run.stage(restored, supplied, recovered, fit, cal, hook=fail_after)
            final, final_supply = fixture(), dispatcher(learner.spec)
            self.assertIsNone(run.Store(Path(temp), identity).restore(final, final_supply))
            self.assertEqual(final.stage, 3)
            self.assertEqual(final.phase, 'await_current')
            self.assertEqual(final.old.seen, 96)
            self.assertEqual(len(list((Path(temp)/'work').glob('*.pt'))), 1)
            self.assertEqual(len(list((Path(temp)/'anchors').glob('*.pt'))), 1)
            self.assertEqual(len(list((Path(temp)/'memory').glob('*.bin'))), 1)
            self.assertEqual(len(list((Path(temp)/'models').glob('*.pt'))), 1)
            self.assertEqual(len(list((Path(temp)/'maps').glob('*.json'))), 1)
            self.assertTrue(unrelated.is_file())
            self.assertTrue((Path(temp)/'retired.jsonl').is_file())
            expected = m.memory(final.spec)
            expected.add_stage([group(i) for i in range(48)])
            expected.add_stage(sorted(current, key=lambda g: g.uid))
            self.assertEqual(final.old.to_bytes(), expected.to_bytes())
            allowed, _ = final_supply.current(3)
            self.assertTrue(all(g.uid in {group(i).uid for i in range(120, 168)} for g in allowed))
            with self.assertRaises(ValueError):
                final_supply.current(2)
            EVIDENCE['stage_transaction'] = dict(before_publish_old_latest_preserved=True,
                after_publish_only_new_latest=True, next_stage=final.stage, retained_seen=final.old.seen,
                next_current_only=True, memory_matches_independent_algorithm_r=True,
                before_and_after_publish_orphans_removed=True, committed_endpoint_preserved=True,
                unrelated_file_preserved=True,
                boundary_fixture_not_288_real_updates=True)

    def test_qualification_budget_admission_and_compute_only_overlap(self):
        with self.assertRaises(TimeoutError):
            qualification.budget_delay(3600, 3590, 3)
        self.assertEqual(qualification.budget_delay(3600, 3500, 3), 29.)
        g = qualification.GIB
        enough = dict(free_gpu_bytes=24*g, ram_available_bytes=190*g, disk_free_bytes=159*g)
        admission = qualification.auxiliary_admission(enough, 6*g, 2*g, g)
        self.assertTrue(admission['ready'])
        self.assertFalse(qualification.auxiliary_admission(dict(enough, free_gpu_bytes=24*g-1), 6*g, 2*g, g)['ready'])
        self.assertFalse(qualification.auxiliary_admission(dict(enough, ram_available_bytes=190*g-1), 6*g, 2*g, g)['ready'])
        self.assertFalse(qualification.auxiliary_admission(dict(enough, disk_free_bytes=159*g-1), 6*g, 2*g, g)['ready'])
        interval = lambda a, b: dict(started=a, ended=b)
        main = [interval(0, 1), interval(5, 6)]  # A long checkpoint lies between, not computation.
        self.assertEqual(qualification.compute_overlap([main, [interval(2, 4)], [interval(2, 4)]]), 0)
        self.assertEqual(qualification.compute_overlap([main, [interval(.5, 2)], [interval(.75, 3)]]), .25)
        children = []
        with tempfile.TemporaryDirectory(dir=os.environ.get('FOREVER_TEST_TMP')) as temp:
            for i in range(3):
                ready = Path(temp)/str(i)
                code = 'import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path('+repr(str(ready))+').write_text("ready"); time.sleep(30)'
                begin = time.monotonic()
                p = subprocess.Popen([sys.executable, '-c', code], start_new_session=True)
                children.append(dict(process=p, started=begin, ended=None))
            try:
                until = time.monotonic()+2
                while not all((Path(temp)/str(i)).exists() for i in range(3)) and time.monotonic() < until:
                    time.sleep(.01)
                self.assertTrue(all((Path(temp)/str(i)).exists() for i in range(3)))
                begin = time.monotonic()
                qualification.stop_children(children)
                tail = time.monotonic()-begin
                self.assertLess(tail, qualification.EXIT_GRACE+qualification.EXIT_KILL_ALLOWANCE)
                self.assertTrue(all(r['process'].returncode == -signal.SIGKILL for r in children))
            finally:
                qualification.stop_children(children)
        EVIDENCE['qualification_controls'] = dict(exit_tail_wall_seconds=tail,
            three_term_ignoring_workers_reaped=True, actual_worker_seconds=qualification.elapsed_workers(children),
            concurrent_exit_reserve_seconds=13, delayed_admission=admission,
            gpu_ram_disk_shortfall_rejected=True, checkpoint_only_overlap_rejected=True,
            compute_only_overlap_reference_seconds=.25)

    def test_actual_supervisor_refuses_delayed_auxiliary_launch(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('FOREVER_TEST_TMP')) as temp:
            root = Path(temp)
            args = SimpleNamespace(mode='gpu', job=root/'job', permit=root/'permit.json')
            permit = dict(status='APPROVED_MINIMAL_IMPLEMENTATION_CPU_GPU_ONLY', sources=[],
                cpu_seconds=600, gpu_seconds=3600, optimizer_calls=48, formal_training=False,
                formal_data=False, implementation_review_closed=True, vendorlink_predecessor=str(root))
            args.permit.write_text(json.dumps(permit))
            launched = []
            class Child:
                pid, returncode = 12345, None
                def poll(self):
                    return self.returncode
            child = Child()
            def launch(command, **kwargs):
                launched.append(command)
                (args.job/'main_ready.json').write_text('{}')
                return child
            def kill(pid, sig):
                self.assertEqual(pid, child.pid)
                child.returncode = -sig
            g = qualification.GIB
            initial = dict(free_gpu_bytes=30*g, ram_available_bytes=192*g, disk_free_bytes=160*g)
            reduced = dict(initial, free_gpu_bytes=24*g-1)
            process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=2*g), children=lambda **kw: [])
            with patch.object(qualification, 'predecessor_ready', return_value=True), \
                 patch.object(qualification.psutil, 'process_iter', return_value=[]), \
                 patch.object(qualification.psutil, 'Process', return_value=process), \
                 patch.object(qualification, 'resources', side_effect=[initial, reduced]), \
                 patch.object(qualification.subprocess, 'check_output', return_value='12345, 6144\n'), \
                 patch.object(qualification.subprocess, 'Popen', side_effect=launch), \
                 patch.object(qualification.os, 'killpg', side_effect=kill):
                with self.assertRaises(RuntimeError):
                    qualification.supervise(args)
            receipt = json.loads((args.job/'receipt.json').read_text())
            self.assertEqual(len(launched), 1)
            self.assertEqual(receipt['reason'], 'AUXILIARY_RESOURCE_CHANGED')
            self.assertEqual(receipt['status'], 'STOP')
            self.assertGreaterEqual(receipt['charged_seconds'], 1)
            EVIDENCE['supervisor_delayed_dispatch'] = dict(actual_supervise_called=True,
                handwritten_fake_processes=True, initial_free_gib=30, later_free_bytes=24*g-1,
                launch_count=1, auxiliary_launches=0, stopped_reason=receipt['reason'],
                outer_test_cpu_charge_only=True)

    def test_vendorlink_priority_gate(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('FOREVER_TEST_TMP')) as temp:
            root = Path(temp)
            self.assertFalse(qualification.predecessor_ready(root, []))
            for name, status in [('ledger.json', 'NATIVE_QUALIFIED_NOT_FORMAL_TRAINING'),
                                 ('g1/result.json', 'G1_PASSED'),
                                 *[(f'g2_{i}/result.json', 'G2_WORKER_PASSED') for i in range(3)]]:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(status=status)))
            self.assertTrue(qualification.predecessor_ready(root, []))
            self.assertFalse(qualification.predecessor_ready(root, ['python step29_vendorlink_native.py']))
            self.assertTrue(qualification.predecessor_ready(root, ['python unrelated_cuda_training.py']))
            (root/'ledger.json').write_text(json.dumps(dict(status='STOP_NO_AUTOMATIC_RETRY')))
            self.assertFalse(qualification.predecessor_ready(root, []))
            self.assertFalse(qualification.predecessor_ready(root, [], True))
            (root/'ledger.json').write_text(json.dumps(dict(status='SINGLE_SLOT_QUALIFIED_G2_NOT_QUALIFIED')))
            for i in range(3):
                (root/f'g2_{i}/result.json').unlink()
            self.assertFalse(qualification.predecessor_ready(root, []))
            self.assertTrue(qualification.predecessor_ready(root, [], True))
            self.assertFalse(qualification.predecessor_ready(root, ['python step29_vendorlink_native.py'], True))
            (root/'g1/result.json').write_text(json.dumps(dict(status='FAIL')))
            self.assertFalse(qualification.predecessor_ready(root, [], True))
            EVIDENCE['vendorlink_priority'] = dict(incomplete_blocked=True, success_and_exit_required=True,
                failed_blocked=True, unrelated_cuda_process_not_a_blocker=True,
                explicit_single_slot_scope_requires_g1_success_and_exit=True,
                single_slot_acceptance_does_not_manufacture_g2_success=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m.parent.configure_numerics()
    suite = unittest.main(verbosity=2, exit=False)
    if os.environ.get('FOREVER_TEST_REPORT'):
        Path(os.environ['FOREVER_TEST_REPORT']).write_text(json.dumps(EVIDENCE, indent=2), encoding='utf-8')
    raise SystemExit(0 if suite.result.wasSuccessful() else 1)
