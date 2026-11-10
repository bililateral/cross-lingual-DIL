"""Only handwritten increment checks; execute under existing Linux py310."""
import time
STARTED = time.monotonic()
import argparse
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import psutil
import torch
from transformers import BertConfig, BertModel, BertTokenizer

import step29_vendorlink_methods as old
from step29_vendorlink_data import Ad, View, Group, seed_for
from step29_vendorlink_run import set_rates
import step30_evolution_control as queue
from step30_evolution_io import Ledger, write, record
import step35_vendorlink_account_policy as policy
import step35_vendorlink_account_methods as methods
import step35_vendorlink_account_run as run
import step35_vendorlink_account_evaluate as evaluate
import step35_vendorlink_account_control as control


def accounts(domain, count=4):
    return [dict(account_uid=f'{domain}:{k:04}', records=[dict(uid=f'{domain}:{k}:{r}',
        title=f'blue item {k}', description=f'text sample {r}') for r in range(12)]) for k in range(count)]


def tiny(directory, settings=None):
    settings = settings or replace(policy.AccountSettings('H_full'), pooling='mean_last4_sum',
        encoder_lr=2.5e-5, head_lr=3e-4, lr_multiplier=1., head_schedule='encoder',
        microbatch=64, channel_tokens=16, vendor_tokens=24)
    vocab = directory/'vocab.txt'
    vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'blue', 'item', 'text', 'sample']
                               +[str(i) for i in range(32)])+'\n')
    tokenizer = BertTokenizer(vocab_file=str(vocab), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=4,
        num_attention_heads=2, intermediate_size=12, max_position_embeddings=32,
        hidden_dropout_prob=.1, attention_probs_dropout_prob=.1)
    torch.manual_seed(123)
    return methods.AccountModel(BertModel(cfg), tokenizer, settings)


def dense_backward(model, group, seed, history=False, teacher=None):
    torch.manual_seed(seed)
    z = model.records(group)
    loss, _ = old.h_objective(model.table(z), group, seed_for(seed, 'regroup'), model.settings, teacher, history)
    if not history and model.settings.gamma:
        labels = torch.tensor([model.auxiliary_classes.index(v.identity) for v in group.views for _ in v.records])
        logits = model.auxiliary_head(z)
        # Independent CE formula and ordinary autograd, no detached leaf/chunk path.
        ce = (logits.logsumexp(1)-logits[torch.arange(len(labels)), labels]).mean()
        loss = loss + model.settings.gamma*ce
    loss.backward()


def development(pass_d=True):
    values = {}
    for task in policy.tasks('D'):
        candidate = task['candidate']
        v = .81 if candidate == 'V0' else .80
        if candidate.startswith('H'):
            v = (.80 if pass_d else .70) if candidate == 'H_CE' else .65
            rows = {'H_SEQ': [dict(seed=41, R11=v)]}
        else:
            rows = {m: [dict(seed=41, R11=v if m == 'V_CLS' else (.95 if candidate == 'V1' else .82))]
                    for m in ('V_CLS', 'V_SET')}
        values[task['id']] = rows
    return values


def matrices():
    definitions = {
        'H_SEQ': (.80, .84, .86, .76, .80), 'H_ER': (.80, .84, .86, .77, .80),
        'H_full': (.80, .84, .86, .80, .83), 'V_CLS': (.81, .85, .87, .70, .72),
        'V_SET': (.80, .84, .86, .73, .75)}
    return {name: [dict(seed=seed, **policy.summarize(dict(R01=.2, R11=a, R12=.60, R21=.7,
        R22=b, R23=.60, R31=x, R32=y, R33=c))) for seed in policy.SEEDS]
        for name, (a, b, c, x, y) in definitions.items()}


class Increment(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['STEP35_TEST_ROOT'])
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_dense_gradient_actual_update_and_gamma_zero(self):
        current = run.groups_for(accounts('B'), 51, 'B', 0)[0]
        history = run.groups_for(accounts('A'), 51, 'A', 0)[0]
        for gamma in (0., 1.):
            model = tiny(self.root)
            model.settings = replace(model.settings, gamma=gamma)
            model.start_account_stage([a['account_uid'] for a in accounts('B')], 76)
            reference = copy.deepcopy(model)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            teacher = model.teacher(history)+.1
            opt, ref_opt = old.make_optimizer(model), old.make_optimizer(reference)
            logs = methods.update_account(model, opt, current, history, teacher, 1234)
            reference.train()
            dense_backward(reference, current, seed_for(1234, 'current'))
            dense_backward(reference, history, seed_for(1234, 'history'), True, teacher)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
            ref_opt.step()
            if gamma:
                reference.auxiliary_optimizer.step()
            for (name, actual), (other_name, expected) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
                self.assertEqual(name, other_name)
                if actual.grad is not None or expected.grad is not None:
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=3e-6)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            self.assertNotIn('ce', logs['history'])
            self.assertGreater(logs['current_encoder_gradient_norm'], 0)
            self.assertGreater(logs['history']['distillation'], 0)
            if gamma:
                self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()
                                    if n.startswith('auxiliary_head.')))
                no_ce = tiny(self.root, replace(model.settings, gamma=0.))
                no_ce.start_account_stage(model.auxiliary_classes, 76)
                no_ce.load_state_dict(before)
                methods.update_account(no_ce, old.make_optimizer(no_ce), current, history, teacher, 1234)
                self.assertTrue(any(not torch.allclose(p.grad, dict(no_ce.named_parameters())[n].grad, atol=1e-6, rtol=1e-4)
                                    for n, p in model.named_parameters() if n.startswith('encoder.')))
            else:
                legacy = old.Model(copy.deepcopy(model.encoder), model.tokenizer, model.settings)
                legacy.load_state_dict({n: v for n, v in before.items() if not n.startswith('auxiliary_head.')})
                old.update_h(legacy, old.make_optimizer(legacy), current, history, teacher, 1234)
                for n, p in legacy.named_parameters():
                    torch.testing.assert_close(p, dict(model.named_parameters())[n], rtol=0, atol=0)

    def test_initialization_state_boundary_and_recovery(self):
        model = tiny(self.root)
        initial_rng = torch.get_rng_state().clone()
        initial_head = copy.deepcopy(model.head.state_dict())
        model.start_account_stage([a['account_uid'] for a in accounts('B')], 76)
        self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
        for n, p in model.head.state_dict().items():
            self.assertTrue(torch.equal(p, initial_head[n]))
        opt = old.make_optimizer(model)
        cur = run.groups_for(accounts('B'), 51, 'B', 0)[0]
        hist = run.groups_for(accounts('A'), 51, 'A', 0)[0]
        memory = old.Memory(14, model.settings)
        memory.retain([hist], 1, model)
        h, t = memory.draw(99)
        methods.update_account(model, opt, cur, h, t, 5)
        state_before = {id(p): {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}
                        for p, state in opt.state.items()}
        model.start_account_stage([a['account_uid'] for a in accounts('C')], 77)
        self.assertEqual(len(model.auxiliary_optimizer.state), 0)
        for p, state in opt.state.items():
            for k, v in state.items():
                if torch.is_tensor(v):
                    self.assertTrue(torch.equal(v, state_before[id(p)][k]))
        cur = run.groups_for(accounts('C'), 51, 'C', 0)[0]
        methods.set_account_rates(model, opt, 2, 10)
        methods.update_account(model, opt, cur, h, t, 6)
        self.assertTrue(all(int(s['step']) == 2 for s in opt.state.values()))
        self.assertTrue(all(int(s['step']) == 1 for s in model.auxiliary_optimizer.state.values()))
        path = self.root/'latest.pt'
        methods.save_account_state(path, model, opt, memory, {'stage': 3, 'next_epoch': 1}, {'test': 1})
        expected = copy.deepcopy(model)
        expected_opt = old.make_optimizer(expected)
        expected_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        methods.update_account(expected, expected_opt, cur, h, t, 7)
        restored = tiny(self.root)
        restored_mem = old.Memory(14, restored.settings)
        restored_opt, cursor = methods.load_account_state(path, restored, restored_mem, {'test': 1})
        self.assertEqual(cursor['stage'], 3)
        methods.update_account(restored, restored_opt, cur, h, t, 7)
        for n, p in expected.state_dict().items():
            torch.testing.assert_close(p, restored.state_dict()[n], rtol=0, atol=0)

    def test_actual_dream_cold_training_scoring_and_phase_stops(self):
        task = next(t for t in policy.tasks('D') if t['id'] == 'D_Dream_H_CE')
        model = tiny(self.root, replace(policy.AccountSettings(**task['settings']), channel_tokens=16,
                                       vendor_tokens=24, microbatch=64))
        task = dict(task, epochs=2, settings=asdict(model.settings))
        loaded = []
        def load(domain):
            loaded.append(domain)
            return accounts(domain)
        result = run.execute(model, load, self.root/'training', task, {'test': 2}, {'test': 1})
        self.assertEqual(loaded, ['Dream'])
        self.assertEqual(result['updates'], 6)
        control.validate_training(task, result, {'counts': {'Dream': {'eligible_accounts': 4}}}, {'test': 2})
        manifest = dict(counts={'Dream': {'eligible_accounts': 4}}, files={})
        for role, count in (('gallery', 4), ('dev', 2)):
            rows = accounts('Dream')
            for row in rows:
                row['records'] = [dict(r, uid=r['uid']+':'+role) for r in row['records'][:count]]
            path = self.root/(role+'.json')
            write(path, rows)
            manifest['files']['Dream:'+role] = record(path)
        def restore(*args):
            return run.restore(*args, factory=lambda binding, selected: tiny(self.root, policy.AccountSettings(**selected['settings'])))
        with patch.object(evaluate, 'restore', side_effect=restore):
            scored = evaluate.score(task, result, manifest, {'test': 1}, self.root/'scores', 'dev', lambda: None)
        self.assertEqual(set(scored['sources']), {'Dream:gallery', 'Dream:dev'})
        self.assertEqual(scored['cells'], ['R11'])
        self.assertEqual(policy.select(development())['selected'], {'H': 'H_CE', 'V': 'V0'})
        visited = []
        def fail_d(phase, selected):
            visited.append(phase)
            return development(False)
        self.assertEqual(control.pipeline(fail_d, lambda *args: None)['status'], 'STOP_ADAPTATION')
        self.assertEqual(visited, ['D'])
        result_rows = matrices()
        self.assertEqual(policy.decide(result_rows, True)['status'], 'PASS_SYSTEM')
        weak = copy.deepcopy(result_rows)
        for row in weak['H_full']:
            row['R11'] = .75
        self.assertFalse(policy.decide(weak, True)['checks']['initial_protection'])
        no_forgetting = copy.deepcopy(result_rows)
        for row in no_forgetting['H_SEQ']:
            row['F'] = .0349
        self.assertFalse(policy.decide(no_forgetting)['checks']['meaningful_forgetting'])
        with self.assertRaises(ValueError):
            policy.decide({**result_rows, 'H_SEQ': result_rows['H_SEQ'][:2]})
        self.assertEqual(policy.CAPS, dict(training=288000, gpu=14400, auxiliary=14400))
        jobs = control.make_jobs(self.root, 'M', policy.tasks('M', {'H': 'H_CE', 'V': 'V0'}), 'train')
        self.assertEqual(len(jobs), 6)
        self.assertTrue(all(j['seconds'] == 36000 for j in jobs))
        ledger = Ledger(self.root/'ledger', dict(training=(6+19+4*46/6)*3600, gpu=0, auxiliary=0), 0, caps=policy.CAPS)
        ledger.reserve('training', 2*36000)
        with patch.object(torch, 'set_num_interop_threads'), patch.object(torch, 'set_num_threads'), \
             patch.object(torch.cuda, 'set_per_process_memory_fraction', side_effect=AssertionError('No allocator cap')):
            control.configure_compute()
        with self.assertRaises(ValueError):
            closed = self.root/'gate.json'
            write(closed, dict(protocol=policy.PROTOCOL, approved=False))
            control.permit(closed)

    def test_real_dispatch_oom_health_no_cap_and_incomplete_barrier(self):
        root = self.root/'queue'
        root.mkdir()
        child = self.root/'child.py'
        child.write_text('import sys,time,json\nfrom pathlib import Path\np=Path(sys.argv[1]);kind=sys.argv[2]\n'
            'if kind=="oom":\n (p/"worker_failure.json").write_text(json.dumps({"kind":"CUDA_OOM"}));sys.exit(9)\n'
            'time.sleep(.5)\n(p/"done").write_text("ok")\n')
        caps = dict(training=300, gpu=300, auxiliary=300)
        ledger = Ledger(root, dict(training=0, gpu=0, auxiliary=0), 0, caps=caps)
        limits = dict(policy.LIMITS, rss_bytes=2**30)
        runner = queue.Runner(dict(experiment_root=str(root), workspace=str(self.root), other_gpu_commitments=[]),
            self.root/'gate.json', root, ledger, worker_entry=child, caps=caps, limits=limits,
            wait_for_resources=False, halt_dispatch_on_failure=True, continue_after_oom=True)
        jobs = [dict(action='train', target=root/name, bucket='training', seconds=30,
                     extra=[kind], gpu_family='H', reserved_bytes=5*2**30)
                for name, kind in (('failed', 'oom'), ('healthy1', 'ok'), ('healthy2', 'ok'))]
        def command(job, slot, deadline):
            return [sys.executable, '-B', str(child), str(job['target']), job['extra'][0]]
        class FakeMemory:
            def snapshot(self):
                occupied = {p.pid: 20*2**30 for p in psutil.Process().children() if p.is_running()}
                return dict(free=80*2**30, total=128*2**30, processes=occupied)
            def close(self):
                pass
        with patch.object(runner, 'command', side_effect=command), \
             patch('step28_formal_queue.GpuMemory', FakeMemory):
            with self.assertRaisesRegex(RuntimeError, 'Incomplete phase'):
                runner.run_many(jobs, 2, lambda job: self.assertTrue((job['target']/'done').exists()))
        state = json.loads((root/'dispatch.json').read_text())
        self.assertEqual(state['status'], 'INCOMPLETE')
        self.assertEqual(len(state['failures']), 1)
        self.assertEqual(state['failures'][0]['kind'], 'CUDA_OOM')
        self.assertEqual(state['pending'], [])
        one = json.loads((root/'healthy1'/'dispatch_receipt.json').read_text())
        two = json.loads((root/'healthy2'/'dispatch_receipt.json').read_text())
        self.assertTrue(one['success'] and two['success'])
        self.assertEqual(one['slot'], two['slot'])
        self.assertGreater(one['peak_gpu_bytes'], 16*2**30)
        self.assertEqual(len([e for e in ledger.events if e['operation'] == 'failed']), 1)
        with self.assertRaisesRegex(RuntimeError, 'downstream'):
            runner.run_many([], 1, lambda job: None)
        write(Path(os.environ['STEP35_TEST_ROOT'])/'dispatch_evidence.json',
              dict(actual_child_processes=True, state=state, healthy_receipts=[one, two], budget=ledger.used))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or Path(sys.executable).resolve() != Path('/home/yongpeng/miniconda3/envs/py310/bin/python').resolve():
        raise RuntimeError('Existing Linux py310 only')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['STEP35_TEST_ROOT'] = str(args.output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Increment))
    seconds = time.monotonic()-STARTED
    write(args.output/'receipt.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), seconds=seconds,
        charged_seconds=__import__('math').ceil(seconds), bucket='auxiliary', real_data_read=False, gpu_runs=0))
    sys.exit(0 if result.wasSuccessful() else 1)
