"""One-off CPU/native GPU checks for the deterministic birth branch only."""
import argparse
import copy
from dataclasses import asdict, replace
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import BertConfig, BertModel, BertTokenizer

from step29_vendorlink_data import Ad, View, Group, assignment, seed_for
from step29_vendorlink_methods import h_objective, h_pool, make_optimizer
from step35_vendorlink_account_methods import AccountModel, deterministic_student, update_account
from step35_vendorlink_account_policy import (AccountSettings, settings, deterministic_tasks, tasks, summarize,
    DETERMINISTIC_PROTOCOL, DETERMINISTIC_CAPS, DETERMINISTIC_LIMITS)
from step35_vendorlink_account_run import execute, restore, make_model
from step35_vendorlink_account_io import record
from step35_vendorlink_account_control import (pipeline, make_jobs, validate_training, deterministic_baseline,
    configure_compute)


def handwritten_group(accounts=4, long=False):
    views = []
    for i in range(accounts):
        for v in range(2):
            records = tuple(Ad(f'a{i}_{v}_{r}',
                ('one two three four five ' * 100 if long else f'one two {i} {r}'),
                ('five four three two one ' * 100 if long else f'three four {v} {i}')) for r in range(2))
            views.append(View(f'v{i}_{v}', records, f'account{i:03d}'))
    return Group('handwritten', 'AlphaBay', 'fit', tuple(views))


def direct_pool(table, slots):
    # Independent indexing/aggregation reference, without production h_pool.
    entries, k = {}, 0
    for i in range(len(slots)):
        for j in range(i+1, len(slots)):
            entries[i, j] = table[k]
            k += 1
    output = []
    for a in range(max(slots)+1):
        for b in range(a+1, max(slots)+1):
            values = torch.stack([entries[min(i,j), max(i,j)]
                for i, s in enumerate(slots) if s == a for j, t in enumerate(slots) if t == b]).double()
            output.append((.5*values.mean()+.5*torch.log(torch.exp(values).mean())).to(table))
    return torch.stack(output)


def dropout_state(encoder):
    return [(m.training, getattr(m, 'p', None), getattr(m, 'dropout_prob', None)) for m in encoder.modules()]


def max_state_difference(a, b, prefix=''):
    pairs = [(v, b.state_dict()[k]) for k, v in a.state_dict().items() if k.startswith(prefix)]
    return max(float((u-v).abs().max()) for u, v in pairs)


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        vocab = self.root/'vocab.txt'
        vocab.write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','one','two','three','four','five']))
        self.tokenizer = BertTokenizer(str(vocab), do_lower_case=True)

    def model(self, setting=None):
        torch.manual_seed(712)
        c = BertConfig(vocab_size=10, hidden_size=8, num_hidden_layers=4,
                       num_attention_heads=2, intermediate_size=16, max_position_embeddings=512)
        c._attn_implementation = 'sdpa'
        encoder = BertModel(c)
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        s = setting or AccountSettings(**dict(settings('H_CE_RMS', 'H_full'), deterministic_history=True))
        s = replace(s, channel_tokens=12)
        model = AccountModel(encoder, self.tokenizer, s)
        model.start_account_stage([f'account{i:03d}' for i in range(4)], 994)
        return model

    def test_birth_gradient_and_checkpoint_context(self):
        m = self.model()
        group = handwritten_group()
        teacher = m.teacher(group)
        teacher += torch.linspace(-.2, .3, len(teacher))
        target_copy = teacher.clone()
        ref = copy.deepcopy(m)
        m.train()
        before = dropout_state(m.encoder)
        rng = torch.random.get_rng_state().clone()
        calls = []
        checkpoint = m.encoder.encoder._gradient_checkpointing_func
        def observed(*args, **kwargs):
            calls.append(1)
            return checkpoint(*args, **kwargs)
        m.encoder.encoder._gradient_checkpointing_func = observed
        logs = m.backward_birth(group, teacher)
        self.assertTrue(calls)
        self.assertEqual(before, dropout_state(m.encoder))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(target_copy, teacher))
        ref.eval()
        scores = ref.table(ref.records(group))
        loss = ref.settings.beta*(direct_pool(scores, assignment(group))-direct_pool(teacher, assignment(group))).square().mean()
        loss.backward()
        errors = [float((p.grad-q.grad).abs().max()) for p, q in zip(m.parameters(), ref.parameters())
                  if p.grad is not None and q.grad is not None]
        self.assertLess(max(errors), 2e-6)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.encoder.parameters() if p.grad is not None), 0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.head.parameters() if p.grad is not None), 0)
        self.assertAlmostEqual(logs['weighted'], float(loss), places=6)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with deterministic_student(m.encoder):
                raise RuntimeError('intentional')
        self.assertEqual(before, dropout_state(m.encoder))
        EVIDENCE['gradient_reference_max_abs'] = max(errors)

    def test_random_paths_legacy_and_beta_zero(self):
        group = handwritten_group()
        m = self.model(replace(self.model().settings, deterministic_history=False))
        old_file = ARGS.old_workspace/'scripts/step35_vendorlink_account_methods.py'
        spec = importlib.util.spec_from_file_location('frozen_account_methods', old_file)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        legacy = old.AccountModel(copy.deepcopy(m.encoder), self.tokenizer, m.settings)
        legacy.start_account_stage(list(m.auxiliary_classes), 994)
        legacy.load_state_dict(m.state_dict())
        teacher = m.teacher(group)+.15
        logs = update_account(m, make_optimizer(m), group, group, teacher, 561)
        old_logs = old.update_account(legacy, make_optimizer(legacy), group, group, teacher, 561)
        self.assertEqual(logs, old_logs)
        self.assertEqual(max_state_difference(m, legacy), 0.)
        det = self.model(replace(m.settings, deterministic_history=True, beta=0.))
        er = copy.deepcopy(det)
        er.settings = replace(er.settings, method='H_ER', deterministic_history=False)
        with patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
            result = update_account(det, make_optimizer(det), group, group, teacher, 992)
            self.assertEqual(clip.call_count, 1)
        state = torch.random.get_rng_state().clone()
        update_account(er, make_optimizer(er), group, group, teacher, 992)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(max_state_difference(det, er), 0.)
        self.assertEqual(result['optimizer_updates'], 1)
        self.assertEqual(result['auxiliary_optimizer_updates'], 1)
        EVIDENCE['legacy_and_beta_zero_update_max_abs'] = 0.

    def test_actual_execute_restore_and_missing_branch_rejection(self):
        task = deterministic_tasks(dict(H='H_CE_RMS', V='V1'))[0]
        task['epochs'] = 1
        task['settings']['channel_tokens'] = 12
        model = self.model(AccountSettings(**task['settings']))
        def fit(domain):
            return [dict(account_uid=f'account{i:03d}', records=[dict(uid=f'{domain}_{i}_{r}',
                title='one two', description='three four') for r in range(12)]) for i in range(4)]
        identity, binding = {'handwritten': True}, {'handwritten': True}
        manifest = {'counts': {d: {'eligible_accounts': 4} for d in task['domains']}}
        completion = execute(model, fit, self.root/'run', task, identity, binding)
        validate_training(task, completion, manifest, identity)
        self.assertEqual(completion['workload']['deterministic_history_groups'], 6)
        restored, _ = restore(binding, task, completion['endpoints']['3']['path'], identity, 3,
                              factory=lambda b, t: self.model(AccountSettings(**t['settings'])))
        self.assertEqual(max_state_difference(model, restored, 'encoder.'), 0.)
        self.assertEqual(max_state_difference(model, restored, 'head.'), 0.)
        broken = copy.deepcopy(completion)
        broken['workload']['deterministic_history_groups'] -= 1
        with self.assertRaises(ValueError):
            validate_training(task, broken, manifest, identity)
        EVIDENCE['handwritten_execute'] = dict(updates=completion['updates'], deterministic_history_groups=6,
                                             endpoint_count=len(completion['endpoints']))

    def test_three_tasks_original_gates_and_baseline(self):
        selected = dict(H='H_CE_RMS', V='V1')
        fixed = deterministic_tasks(selected)
        self.assertEqual([x['seed'] for x in fixed], [51,52,53])
        jobs = make_jobs(self.root, 'M', fixed, 'train')
        self.assertEqual([j['seconds'] for j in jobs], [46800]*3)
        self.assertLess(sum(j['seconds'] for j in jobs), DETERMINISTIC_CAPS['training'])
        self.assertEqual(DETERMINISTIC_LIMITS['disk_bytes'], 12*2**30)
        old_root = ARGS.old_workspace.parent
        old_gate = json.loads((old_root/'gate.json').read_text())
        old_campaign = Path(old_gate['campaign'])
        prior_m = {k: record(p) for k, p in dict(gate=old_root/'gate.json',
            batch=old_campaign/'M/batch.json', completion=old_campaign/'completion.json',
            budget=old_campaign/'budget.json', results=old_campaign/'M/analysis/result/results.json').items()}
        gate = dict(protocol=DETERMINISTIC_PROTOCOL, prior_P=old_gate['prior_P'], prior_M=prior_m,
            baseline=old_gate['baseline'], manifest=old_gate['manifest'], model_binding=old_gate['model_binding'],
            experiment_root=str(self.root/'new'))
        baseline = deterministic_baseline(gate)
        candidate = {'H_full': copy.deepcopy(baseline['H_full'])}
        phases, saved = [], {}
        def phase(name, selection):
            phases.append((name, selection))
            return candidate
        result = pipeline(phase, saved.__setitem__, baseline, deterministic=True)
        self.assertEqual(phases, [('M', selected)])
        self.assertFalse(result['confirmation'])
        self.assertFalse(result['application'])
        self.assertFalse(result['automatic_followup'])
        self.assertFalse(result['candidate_M']['checks']['meaningful_forgetting'])
        self.assertEqual(result['comparisons']['H_full']['means']['O'], 0.)
        candidate['H_full'].pop()
        with self.assertRaises(ValueError):
            pipeline(phase, saved.__setitem__, baseline, deterministic=True)
        EVIDENCE['fixed_scope'] = dict(training_tasks=3, max_training_reservation_seconds=140400,
                                      reused_baseline_methods=sorted(baseline), downstream=False)


def gpu_check():
    g = json.loads((ARGS.old_workspace.parent/'gate.json').read_text())
    for b in g['model_binding']['files']:
        if record(b['path']) != b:
            raise ValueError('Pretrained source identity mismatch')
    task = deterministic_tasks(dict(H='H_CE_RMS', V='V1'))[0]
    m = make_model(g['model_binding'], task)
    m.start_account_stage([f'account{i:03d}' for i in range(875)], 994)
    group = handwritten_group(11, long=True)
    m.eval()
    with torch.no_grad():
        expected = m.table(m.records(group))
    teacher = expected.cpu()+torch.linspace(-.2, .3, len(expected))
    m.train()
    states = dropout_state(m.encoder)
    torch.cuda.reset_peak_memory_stats()
    with deterministic_student(m.encoder), torch.no_grad():
        actual = m.table(m.records(group))
    error = float((actual-expected).abs().max())
    if error > 2e-6:
        raise AssertionError(('Deterministic/eval mismatch', error))
    m.zero_grad(set_to_none=True)
    birth = m.backward_birth(group, teacher)
    encoder_norm = float(torch.stack([p.grad.float().norm() for p in m.encoder.parameters() if p.grad is not None]).norm())
    pair_norm = float(torch.stack([p.grad.float().norm() for p in m.head.parameters() if p.grad is not None]).norm())
    if min(encoder_norm, pair_norm) <= 0 or states != dropout_state(m.encoder):
        raise AssertionError('Gradient or context restoration failed')
    before_encoder = m.encoder.embeddings.word_embeddings.weight.detach().clone()
    before_pair = m.head[2].weight.detach().clone()
    optimizer = make_optimizer(m)
    with patch.object(optimizer, 'step', wraps=optimizer.step) as main_step, \
         patch.object(m.auxiliary_optimizer, 'step', wraps=m.auxiliary_optimizer.step) as aux_step, \
         patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
        logs = update_account(m, optimizer, group, group, teacher, 345)
        if (main_step.call_count, aux_step.call_count, clip.call_count) != (1,1,1):
            raise AssertionError('One joint update required')
    encoder_change = float((m.encoder.embeddings.word_embeddings.weight-before_encoder).abs().max())
    pair_change = float((m.head[2].weight-before_pair).abs().max())
    if min(encoder_change, pair_change) <= 0 or states != dropout_state(m.encoder):
        raise AssertionError('Actual parameter update or mode restoration failed')
    return dict(status='PASS', handwritten=True, pretrained_model=True, real_data_read=False,
        records=44, channel_tokens=256, auxiliary_classes=875, actual_updates=1,
        deterministic_eval_max_abs=error, isolated_birth=birth,
        isolated_birth_encoder_gradient_norm=encoder_norm, isolated_birth_pair_gradient_norm=pair_norm,
        encoder_update_max_abs=encoder_change, pair_update_max_abs=pair_change, logs=logs,
        gradient_checkpointing=m.encoder.is_gradient_checkpointing,
        attention_implementation=m.encoder.config._attn_implementation,
        modes_restored=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), no_artificial_gpu_cap=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('cpu','gpu'), required=True)
    parser.add_argument('--old-workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    ARGS = parser.parse_args()
    EVIDENCE = {}
    configure_compute()
    started = time.monotonic()
    if ARGS.mode == 'cpu':
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        value = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                     failures=len(result.failures), errors=len(result.errors), evidence=EVIDENCE,
                     formal_data_read=False, gpu_used=False)
    else:
        value = gpu_check()
    value['body_seconds'] = time.monotonic()-started
    ARGS.output.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(value, allow_nan=False), flush=True)
    raise SystemExit(0 if value['status']=='PASS' else 1)
