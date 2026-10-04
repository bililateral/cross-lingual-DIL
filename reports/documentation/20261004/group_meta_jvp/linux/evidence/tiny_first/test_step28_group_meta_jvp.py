"""Reuse the shipped scalar and full-group update checks, also compare each HVP."""
import unittest
from unittest import mock

import torch

import step28_group_meta_jvp as candidate
import test_step28_group_meta_staged as prior


class JvpTests(unittest.TestCase):
    def compare_case(self, case: str) -> None:
        actual_hvp = candidate.hvp
        calls = []

        def compared(loss, params, vector):
            devices = sorted({p.device.index for p in params.values() if p.is_cuda})
            # Restore RNG after the independent reverse-over-reverse reference,
            # so the actual JVP sees exactly the same dropout stream.
            with torch.random.fork_rng(devices=devices):
                first = torch.autograd.grad(loss(params), tuple(params.values()),
                                            create_graph=True, allow_unused=True)
                active = [(g, v) for g, v in zip(first, vector, strict=True)
                          if g is not None and g.requires_grad]
                expected = (torch.autograd.grad(tuple(g for g, _ in active), tuple(params.values()),
                                               grad_outputs=tuple(v for _, v in active), allow_unused=True)
                            if active else (None,) * len(params))
            result = actual_hvp(loss, params, vector)
            for p, wanted, got in zip(params.values(), expected, result, strict=True):
                torch.testing.assert_close(got, torch.zeros_like(p) if wanted is None else wanted,
                                           rtol=2e-5, atol=2e-6)
                self.assertFalse(got.requires_grad)
            calls.append(1)
            return result

        with mock.patch.object(prior, 'staged', candidate), mock.patch.object(candidate, 'hvp', compared):
            getattr(prior.StagedTests(), case)()
        self.assertGreaterEqual(len(calls), 2)

    def test_scalar_full_chain_and_unused_adam(self):
        self.compare_case('test_nonlinear_chain_rates_and_unused_adam')

    def test_each_hvp_full_gradient_dropout_and_two_updates(self):
        self.compare_case('test_complete_gradient_dropout_and_two_adam_steps')


if __name__ == '__main__':
    unittest.main()
