"""Necessary handwritten checks through actual new execution/evaluation paths."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from step29_vendorlink_methods import Model, h_objective, make_optimizer, update_h
from step34_vendorlink_account_data import write, record
from step34_vendorlink_account_policy import AccountSettings, DOMAINS, settings, tasks, decide, summarize
from step34_vendorlink_account_run import groups_for, execute, restore
import step34_vendorlink_account_evaluate as evaluation
from step34_vendorlink_account_control import pipeline, validate_training
from step30_evolution_control import Runner
from step30_evolution_io import Ledger


def accounts(domain, count=4):
    return [dict(account_uid=f'{domain}-{i:02d}', records=[dict(uid=f'{domain}-{i}-{j}',
        title=f'blue item {i}', description=f'text sample {j}') for j in range(12)]) for i in range(count)]


def tiny(task, root, checkpoint=True):
    vocab = root/'vocab.txt'
    if not vocab.exists():
        vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'blue', 'item', 'text', 'sample']+
                                  [str(i) for i in range(20)])+'\n')
    tokenizer = BertTokenizer(vocab_file=str(vocab), do_lower_case=True)
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=4,
        num_attention_heads=2, intermediate_size=12, max_position_embeddings=32,
        hidden_dropout_prob=.1, attention_probs_dropout_prob=.1)
    torch.manual_seed(41)
    encoder = BertModel(config)
    if checkpoint and task['method'] != 'VendorLink_SEQ':
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return Model(encoder, tokenizer, AccountSettings(**task['settings']))


def small_task(method, seed=43):
    candidate = 'V1' if method == 'VendorLink_SEQ' else 'H1'
    value = settings(candidate, method)
    value.update(channel_tokens=16, vendor_tokens=24)
    return dict(id=method+'_fixture', phase='P', method=method, candidate=candidate,
                seed=seed, stages=3, epochs=2, settings=value)


class IncrementTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_checkpoint_and_weight_gradients(self):
        task = small_task('H_full')
        left, right = tiny(task, self.root, False), tiny(task, self.root, True)
        group = groups_for(accounts('A'), 43, 'A', 0)[0]
        teacher = left.teacher(group)+.2
        opts = [make_optimizer(m) for m in (left, right)]
        for step in (17, 18):
            a = update_h(left, opts[0], group, group, teacher, step)
            b = update_h(right, opts[1], group, group, teacher, step)
            self.assertAlmostEqual(a['history']['weighted'], b['history']['weighted'], places=6)
        for x, y in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-6)
        table = torch.linspace(-.3, .4, 120, requires_grad=True)
        s = AccountSettings(**task['settings'])
        total, _ = h_objective(table, group, 2, s, teacher=torch.zeros(120), history=True)
        er, _ = h_objective(table, group, 2, replace(s, method='H_ER', alpha=1., beta=0.), history=True)
        full0, _ = h_objective(table, group, 2, replace(s, alpha=0., beta=1.), teacher=torch.zeros(120), history=True)
        reference = s.alpha*er+s.beta*full0
        torch.testing.assert_close(torch.autograd.grad(total, table, retain_graph=True)[0],
                                   torch.autograd.grad(reference, table)[0])

    def test_training_scoring_and_old_heads(self):
        manifest = dict(counts={d: {'eligible_accounts': 4} for d in DOMAINS}, files={})
        for domain in DOMAINS:
            for role, n in (('gallery', 4), ('dev', 2), ('confirm', 2)):
                rows = [dict(account_uid=a['account_uid'], records=[dict(uid=f'{role}-{a["account_uid"]}-{j}',
                    title=f'blue item {i}', description=f'{role} text {j}') for j in range(n)])
                    for i, a in enumerate(accounts(domain))]
                path = self.root/f'{domain}_{role}.json'
                write(path, rows)
                manifest['files'][domain+':'+role] = record(path)
        binding, streams, initial = {'fixture': True}, {}, {}
        for method in ('H_SEQ', 'H_ER', 'H_full', 'VendorLink_SEQ'):
            task = small_task(method)
            model = tiny(task, self.root)
            calls = []
            def load(domain):
                calls.append(domain)
                return accounts(domain)
            job = self.root/method
            identity = {'fixture_task': task}
            result = execute(model, load, job, task, identity, binding)
            validate_training(task, result, manifest, identity)
            self.assertEqual(calls, list(DOMAINS))
            self.assertEqual(result['workload']['current_ads'], 288)
            updates = [json.loads(line) for line in (job/'updates.jsonl').read_text().splitlines()]
            streams[method] = [r['current'] for r in updates]
            endpoint = torch.load(job/'stage1.pt', weights_only=False)
            initial[method] = endpoint['encoder']
            if method != 'VendorLink_SEQ':
                self.assertEqual(result['updates'], 18)
                self.assertEqual(result['workload']['history_groups'], 0 if method == 'H_SEQ' else 12)
            else:
                last = torch.load(job/'stage3.pt', weights_only=False)
                self.assertEqual(set(last['vendor_heads']), set(DOMAINS))
                for k, v in endpoint['vendor_heads']['AlphaBay']['state'].items():
                    torch.testing.assert_close(v, last['vendor_heads']['AlphaBay']['state'][k], rtol=0, atol=0)
            def restoring(b, t, ep, ident, stage):
                return restore(b, t, ep, ident, stage, factory=lambda b, t: tiny(t, self.root))
            with patch.object(evaluation, 'restore', side_effect=restoring):
                scored = evaluation.score(task, result, manifest, binding, job/'score', 'dev', lambda: None)
            analyzed = evaluation.analyze([scored], [task], job/'analysis')
            self.assertIn('R12', analyzed['V_CLS' if method == 'VendorLink_SEQ' else method][0])
            if method == 'VendorLink_SEQ':
                # Old head + changed current encoder must change old predictions.
                ids, query = evaluation.load_role(manifest['files']['AlphaBay:dev'], 'AlphaBay', 'dev', 4)
                old = last['vendor_heads']['AlphaBay']
                before, ads = evaluation.classification(model, query, ids, old, 43, 'AlphaBay')
                np.testing.assert_allclose(before, ads.reshape(4, 2, 4).mean(1))
                with torch.no_grad():
                    model.encoder.pooler.dense.bias.add_(1.)
                after, _ = evaluation.classification(model, query, ids, old, 43, 'AlphaBay')
                self.assertGreater(float(np.abs(after-before).max()), 1e-6)
        self.assertEqual(streams['H_SEQ'], streams['H_ER'])
        self.assertEqual(streams['H_SEQ'], streams['H_full'])
        for name in initial['H_SEQ']:
            torch.testing.assert_close(initial['H_SEQ'][name], initial['H_full'][name], rtol=0, atol=0)
            torch.testing.assert_close(initial['H_SEQ'][name], initial['H_ER'][name], rtol=0, atol=0)

    def test_rectangular_roles_and_macro_f1(self):
        model = tiny(small_task('H_SEQ'), self.root)
        q, g = torch.randn(3, 2, 16), torch.randn(4, 4, 16)
        actual = evaluation.rectangular(model, q, g)
        expected = np.zeros((3, 4))
        with torch.no_grad():
            for i in range(3):
                for j in range(4):
                    logits = torch.stack([model.head(torch.cat(((a-b).abs(), a*b))).reshape(()) for a in q[i] for b in g[j]]).double()
                    expected[i, j] = .5*logits.mean()+.5*logits.exp().mean().log()
        np.testing.assert_allclose(actual, expected, atol=2e-7)
        self.assertEqual(evaluation.metrics(np.eye(4), list(range(4)), 4)['macro_F1'], 1.)
        x = np.zeros((4, 4));x[:, 0] = 1
        self.assertAlmostEqual(evaluation.metrics(x, list(range(4)), 4)['macro_F1'], .1)
        self.assertEqual(evaluation.metrics(x, list(range(4)), 4)['Top1'], .25)

    def test_gate_stop_prevents_downstream(self):
        d = {}
        for c in ('H0', 'H1', 'V0', 'V1'):
            d['D_'+c] = {m: [dict(R11=.8)] for m in (('H_SEQ',) if c[0] == 'H' else ('V_CLS', 'V_SET'))}
        bad = {m: [dict(seed=s, **summarize(dict(R11=.8, R22=.8, R33=.8, R31=.8, R32=.8, R12=.6, R23=.6)))
                   for s in (43, 44, 45)] for m in ('H_SEQ', 'V_CLS', 'V_SET')}
        called = []
        def phase(name, selected):
            called.append(name)
            if name == 'D':return d
            if name == 'P':return bad
            self.fail('Downstream was opened after no meaningful forgetting')
        result = pipeline(phase, lambda name, value: None)
        self.assertEqual(result['status'], 'STOP_NO_MEANINGFUL_FORGETTING')
        self.assertEqual(called, ['D', 'P'])
        for row in bad['H_SEQ']:row['R11'] = .1
        self.assertEqual(decide(bad)['status'], 'STOP_LEARNING')

    def test_actual_dispatch_failure_and_reservations(self):
        caps = dict(training=100, gpu=100, auxiliary=180)
        ledger = Ledger(self.root, dict(training=0, gpu=0, auxiliary=0), 0, caps=caps)
        with self.assertRaises(TimeoutError):ledger.reserve('training', 101)
        limits = dict(workers=2, free_ram_bytes=1024, disk_bytes=2**30, free_disk_reserve=1024,
                      free_gpu_bytes=0, process_gpu_bytes=0, rss_bytes=2**30)
        allowed = dict(experiment_root=str(self.root), workspace=str(self.root), other_gpu_commitments=[])
        class FixtureRunner(Runner):
            def command(self, job, slot, deadline):
                return [sys.executable, '-c', 'import time,sys;time.sleep(float(sys.argv[1]));sys.exit(int(sys.argv[2]))',
                        '0.3' if job['target'].name == 'bad' else '0.8', '1' if job['target'].name == 'bad' else '0']
        runner = FixtureRunner(allowed, self.root/'gate', self.root, ledger, caps=caps, limits=limits,
                               halt_dispatch_on_failure=True)
        jobs = [dict(action='fixture', target=self.root/n, bucket='auxiliary', seconds=10, extra=[])
                for n in ('bad', 'healthy', 'never_dispatched')]
        with self.assertRaises(RuntimeError):runner.run_many(jobs, 2, lambda j: None)
        self.assertTrue(json.loads((self.root/'healthy/dispatch_receipt.json').read_text())['success'])
        self.assertFalse((self.root/'never_dispatched/console.txt').exists())
        self.assertFalse(json.loads((self.root/'dispatch.json').read_text())['active'])


if __name__ == '__main__':
    unittest.main()
