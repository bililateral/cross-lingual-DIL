"""Incremental handwritten Linux checks of English adaptation and two slots."""
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import BertConfig, BertModel, BertTokenizer

import step28_formal_queue as queue
import step29_vendorlink_data as data
import step29_vendorlink_methods as method
import step29_vendorlink_run as run
import step29_english_abc as policy

EVIDENCE = {}


def group(domain='AlphaBay'):
    views = []
    for identity in range(4):
        for side in range(2):
            ads = tuple(data.Ad(f'{domain}-{identity}-{side}-{k}', f'blue item {identity}',
                               f'text {side} sample {k}') for k in range(2))
            views.append(data.View(f'{domain}:{identity}:{side}', ads, f'{domain}:{identity}'))
    return data.Group(domain+':fixture', domain, 'fit', tuple(views))


def tiny(root, settings):
    vocab = root/'vocab.txt'
    vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'blue', 'item',
                               'text', 'sample']+[str(i) for i in range(8)])+'\n')
    tokenizer = BertTokenizer(vocab_file=str(vocab), do_lower_case=True)
    torch.manual_seed(41)
    bert = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=4,
        num_attention_heads=2, intermediate_size=12, max_position_embeddings=32,
        hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
    return method.Model(bert, tokenizer, replace(settings, channel_tokens=16, vendor_tokens=24, microbatch=64))


class EnglishABC(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['ABC_TEST_ROOT'])
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_pool_independent_values_padding_and_each_layer_gradients(self):
        mask = torch.tensor([[1, 1, 0]])
        layers = [torch.tensor([[[float(i), 2.*i], [3.*i, 4.*i], [999., 999.]]], requires_grad=True)
                  for i in range(1, 6)]
        output = SimpleNamespace(last_hidden_state=layers[-1], hidden_states=tuple(layers))
        expected = {'cls': [5., 10.], 'mean_last': [10., 15.], 'mean_last4_sum': [28., 42.]}
        for name, reference in expected.items():
            actual = method.text_pool(output, mask, name)
            torch.testing.assert_close(actual, torch.tensor([reference]), atol=0, rtol=0)
        result = method.text_pool(output, mask, 'mean_last4_sum')
        result.sum().backward()
        self.assertIsNone(layers[0].grad)
        for layer in layers[1:]:
            torch.testing.assert_close(layer.grad, torch.tensor([[[.5,.5],[.5,.5],[0.,0.]]]), atol=0, rtol=0)
        EVIDENCE['pool'] = {'hand_computed': expected, 'four_layer_gradients': True, 'padding_gradient_zero': True}

    def test_actual_records_reach_four_layers_and_saved_endpoints(self):
        settings = replace(policy.baseline(), pooling='mean_last4_sum', head_schedule='encoder',
                           bce_weight=.25, hard_weight=0.)
        model = tiny(self.root, settings)
        seen = []
        original = model.encode
        def capture(*args, **kwargs):
            output, mask = original(*args, **kwargs)
            if kwargs.get('hidden') and torch.is_grad_enabled():
                for value in output.hidden_states[-4:]:
                    value.retain_grad()
                seen.append(output.hidden_states[-4:])
            return output, mask
        model.encode = capture
        g = group()
        teacher = model.teacher(g)+.2
        optimizer = method.make_optimizer(model)
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        logs = method.update_h(model, optimizer, g, g, teacher, 51)
        self.assertTrue(all(t.grad is not None and t.grad.abs().sum() > 0 for states in seen[-2:] for t in states))
        self.assertTrue(any(not torch.equal(before[k], p) for k,p in model.named_parameters() if k.startswith('encoder.')))
        self.assertTrue(any(not torch.equal(before[k], p) for k,p in model.named_parameters() if k.startswith('head.')))
        # Real three-stage execute path on handwritten data; no formal corpus.
        model.encode = original
        job = self.root/'trajectory'
        run.execute(model, lambda domain: [group(domain)], job, order=['AlphaBay','Dream','Agora'],
                    seed=7, epochs=2, identity={'fixture': 'english_abc'}, inference_endpoints=True)
        model.eval()
        expected = model.records(g)
        loaded = tiny(self.root, settings)
        state = torch.load(job/'stage3.pt', weights_only=False)
        self.assertEqual(state['settings'], asdict(model.settings))
        loaded.load_state_dict(state['model']); loaded.eval()
        torch.testing.assert_close(expected, loaded.records(g), atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'identity'):
            wrong = tiny(self.root, replace(settings, pooling='cls'))
            run.load_state(job/'latest.pt', wrong, method.Memory(1, wrong.settings), {'fixture':'english_abc'})
        EVIDENCE['actual_trajectory'] = {'updates':6, 'endpoints':3, 'settings_restored':True,
                                       'each_layer_received_gradient':True, 'current_and_history_updated':True}

    def test_loss_weights_affect_both_branches_and_independent_bce_gradient(self):
        g = group()
        nrecords = sum(len(v.records) for v in g.views)
        base = torch.linspace(-.4, .8, nrecords*(nrecords-1)//2, dtype=torch.float64)
        for history in (False, True):
            gradients = {}
            for bce in (1., .25):
                scores = base.clone().requires_grad_()
                loss, _ = method.h_objective(scores, g, 73, replace(policy.baseline(), bce_weight=bce),
                                             teacher=base+.1, history=history)
                gradients[bce] = torch.autograd.grad(loss, scores)[0]
            scores = base.clone().requires_grad_()
            natural, regrouped = data.assignment(g), data.assignment(g, 73)
            labels = scores.new_tensor(g.labels())
            bces = []
            for slots in (natural, regrouped):
                pooled = method.h_pool(scores, slots)
                bces.append(torch.where(labels.bool(), F.softplus(-pooled), F.softplus(pooled)).mean())
            independent = torch.autograd.grad(sum(bces)/2, scores)[0]
            factor = .1 if history else 1.
            torch.testing.assert_close(gradients[1.]-gradients[.25], .75*factor*independent, atol=1e-12, rtol=1e-10)
            scores = base.clone().requires_grad_()
            full, _ = method.h_objective(scores, g, 73, policy.baseline(), teacher=base+.1, history=history)
            nohard, _ = method.h_objective(scores, g, 73, replace(policy.baseline(), hard_weight=0.), teacher=base+.1, history=history)
            delta = torch.autograd.grad(full-nohard, scores)[0]
            self.assertGreater(float(delta.abs().sum()), 0.)
        EVIDENCE['loss_weights'] = {'current_and_history_bce_gradient_reference':True, 'hard_gradient_removed_both':True}

    def test_effective_rates_and_adam_parameter_effect(self):
        s = policy.candidate('L3', policy.candidate('L2', policy.candidate('L1', policy.baseline())))
        model = tiny(self.root, s)
        optimizer = method.make_optimizer(model)
        rates = []
        for step, factor in ((1,.5),(2,1.),(11,.5),(20,0.)):
            run.set_rates(optimizer, s, step, 20)
            self.assertEqual([p['lr'] for p in optimizer.param_groups], [2.5e-5*factor,3e-4*factor])
            rates.append([p['lr'] for p in optimizer.param_groups])
        before = [p.detach().clone() for p in model.parameters()]
        for p in model.parameters(): p.grad = torch.ones_like(p)
        optimizer.step()
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,model.parameters())))
        run.set_rates(optimizer,s,2,20); optimizer.step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())))
        EVIDENCE['rates'] = {'effective_values':rates, 'zero_factor_no_update':True, 'nonzero_factor_real_adam_update':True}

    def test_selection_rejects_low_initial_learning_and_missing_pairs(self):
        old = dict(B2=.95,B3=.93,C3=.9,A3=.9)
        good = dict(B2=.95,B3=.947,C3=.9,A3=.9)
        low = dict(B2=.89,B3=.89,C3=.9,A3=.9)
        self.assertFalse(policy.basic_quality(low,old))
        self.assertTrue(policy.success([good]*3,[old]*3,[old]*3,final=True)['passed'])
        self.assertFalse(policy.success([low]*3,[old]*3,[old]*3,final=True)['passed'])
        bad_seed = [good,good,{**good,'C3':.889}]
        self.assertFalse(policy.success(bad_seed,[old]*3,[old]*3,final=True)['passed'])
        self.assertFalse(policy.precheck([{**old,'B3':.96}]*3))
        with self.assertRaises(ValueError): policy.success([good]*2,[old]*3,[old]*3,final=True)
        self.assertLess(policy.selection_key('P0',policy.baseline(),old,old,old),
                        policy.selection_key('P1',policy.candidate('P1'),old,old,old))
        EVIDENCE['selection'] = {'false_no_forgetting_rejected':True,'bad_agora_seed_rejected':True,
                                 'missing_seed_rejected':True,'no_vendor_forgetting_stops':True,'tie_favors_simple':True}

    def test_two_real_children_failure_retirement_and_refill(self):
        # Only NVML/resource readings are handwritten; child process execution is real.
        import test_step28_formal_wait as old
        allowed = old.allowance(self.root, 3)
        allowed['workers'] = 2
        with patch.object(queue, 'GpuMemory', old.Memory):
            with self.assertRaisesRegex(RuntimeError, 'Incomplete training'):
                queue.dispatch(allowed,[dict(id=str(i),seconds=100,exit=2 if i==0 else 0) for i in range(5)],
                               old.command,lambda r,p:self.assertTrue((p/'ran').exists()),poll_seconds=.03)
        result = json.loads((self.root/'batch/failure.json').read_text())
        self.assertEqual(result['retired_slots'],[0])
        self.assertEqual(len(result['completed']),5)
        self.assertEqual(sum(r['success'] for r in result['completed']),4)
        self.assertTrue(all(r['slot']==1 for r in result['completed'] if int(r['id'])>=2))
        self.assertFalse((self.root/'batch/training_complete.json').exists())
        gib=2**30
        reserve=queue.admission(dict(free=20*gib,processes={10:2*gib}),[10],15*gib)
        self.assertFalse(reserve['ready']);self.assertEqual(reserve['required_gpu_bytes'],28*gib)
        EVIDENCE['two_slot_dispatch'] = {'actual_children':result['completed'],'reservation':reserve,
             'scope':'dispatch kernel only; new formal adaptive campaign entry not claimed deployed'}


if __name__ == '__main__':
    if sys.platform != 'linux': raise SystemExit('Linux qualification only')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:
        unittest.main(verbosity=2)
    finally:
        Path(os.environ['ABC_TEST_ROOT'],'evidence.json').write_text(json.dumps(EVIDENCE,indent=2)+'\n')
