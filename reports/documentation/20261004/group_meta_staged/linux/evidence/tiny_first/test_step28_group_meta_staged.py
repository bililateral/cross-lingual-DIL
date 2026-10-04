"""Staged-gradient checks on handmade groups and an independent nonlinear scalar."""
import copy
from dataclasses import replace
import os
import unittest
from unittest import mock

import torch

import test_step28_group_meta as fixtures
import step28_group_meta as direct
import step28_group_meta_staged as staged
from step28_group_meta_gpu_check import assert_state_close


class StagedTests(unittest.TestCase):
    def test_nonlinear_chain_rates_and_unused_adam(self):
        dtype = torch.float64
        p = {n: torch.tensor(v, dtype=dtype, requires_grad=True)
             for n, v in zip(('x', 'y', 'unused', 'zero'), (.3, -.2, .7, .1))}
        rates = dict(x=.03, y=.11, unused=.07, zero=.02)
        support = (lambda p: .5 * (torch.sin(p['x'] * p['y']) - .8).square() + p['zero'] * 0,
                   lambda p: .1 * (p['x'] + 2 * p['y'] - 1).square())
        query = (lambda p: .5 * (p['x'] + p['y'] + .4).square(),
                 lambda p: .3 * (p['x'] * p['y'] + .9).square())
        total, _, _, _ = direct.lookahead(p, lambda q: sum(f(q) for f in support),
                                          lambda q: sum(f(q) for f in query), rates, .7)
        expected = torch.autograd.grad(total, tuple(p.values()), allow_unused=True)
        actual, _ = staged.staged_gradient(p, support, query, rates, .7)
        for name, value in zip(p, expected):
            if value is None:
                self.assertIsNone(actual[name])
            else:
                torch.testing.assert_close(actual[name], value, rtol=1e-10, atol=1e-12)
        self.assertIsNone(actual['unused'])
        self.assertIsNotNone(actual['zero'])
        # Independent numerical inner gradient and outer central difference.
        import math
        def scalar(x, y):
            z = x * y
            common = (math.sin(z) - .8) * math.cos(z)
            gx = common * y + .2 * (x + 2*y - 1)
            gy = common * x + .4 * (x + 2*y - 1)
            a, b = x - .03*gx, y - .11*gy
            return .5*(math.sin(z)-.8)**2 + .1*(x+2*y-1)**2 + .7*(.5*(a+b+.4)**2+.3*(a*b+.9)**2)
        eps = 1e-6
        for name, delta in (('x', (eps, 0)), ('y', (0, eps))):
            fd = (scalar(.3+delta[0], -.2+delta[1])-scalar(.3-delta[0], -.2-delta[1]))/(2*eps)
            self.assertAlmostEqual(actual[name].item(), fd, places=8)

    def test_complete_gradient_dropout_and_two_adam_steps(self):
        device = os.environ.get('STAGED_TEST_DEVICE', 'cpu')
        c, episode, model, settings = fixtures.setup()
        model.to(device)
        model.encoder.register_parameter('unused', torch.nn.Parameter(torch.tensor(.7, device=device)))
        optimizer = direct.core.make_optimizer(model, c)
        fixtures.fixtures.toy_prior_step(model, optimizer, episode.history_support, c)
        # Pre-existing moments expose accidental materialization of a zero grad.
        optimizer.state[model.encoder.unused] = {
            'step': torch.tensor(288.), 'exp_avg': torch.tensor(.2, device=device),
            'exp_avg_sq': torch.tensor(.1, device=device)}
        initial_model, initial_adam = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        for setting in (settings, direct.Settings(1e-5, 1e-3, .1, 1.), replace(settings, outer_weight=0.)):
            model.load_state_dict(initial_model)
            optimizer.load_state_dict(copy.deepcopy(initial_adam))
            for step in (1, 2):
                model.train()
                seeds = tuple(100 + step*4 + i for i in range(4))
                rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone() if device.startswith('cuda') else None
                before = copy.deepcopy(model.state_dict())
                loss, _, _ = direct.objective(model, episode, c, setting, seeds)
                expected = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
                loss_value = float(loss.detach())
                del loss
                actual, terms = staged.gradient(model, episode, c, setting, seeds)
                for (name, _), g in zip(model.named_parameters(), expected):
                    if g is None:
                        self.assertIsNone(actual['model.'+name])
                    else:
                        torch.testing.assert_close(actual['model.'+name], g, rtol=2e-5, atol=2e-6)
                self.assertAlmostEqual(terms['total'], loss_value, delta=3e-6)
                self.assertEqual(terms['physical_group_forwards'], 2 if not setting.outer_weight else 6)
                assert_state_close(before, model.state_dict())
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                if cuda_rng is not None:
                    self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
                old_adam = copy.deepcopy(optimizer.state_dict())
                direct.update(model, optimizer, episode, c, setting, 2, step, seeds)
                reference_model, reference_adam = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
                model.load_state_dict(before)
                optimizer.load_state_dict(old_adam)
                with mock.patch.object(optimizer, 'step', wraps=optimizer.step) as opt_step, \
                        mock.patch.object(torch.nn.utils, 'clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
                    staged.update(model, optimizer, episode, c, setting, 2, step, seeds)
                self.assertEqual((opt_step.call_count, clip.call_count), (1, 1))
                assert_state_close(reference_model, model.state_dict())
                assert_state_close(reference_adam, optimizer.state_dict())
                self.assertEqual(optimizer.state[model.encoder.unused]['step'].item(), 288)


if __name__ == '__main__':
    unittest.main()
