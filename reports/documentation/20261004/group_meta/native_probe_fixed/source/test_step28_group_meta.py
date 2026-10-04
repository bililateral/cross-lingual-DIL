"""Handwritten evidence for the full look-ahead graph and whole-group routing."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_group_meta as method
import test_step28_bge_continual_contracts as fixtures


def setup():
    c = fixtures.config()
    c["input"]["activation_checkpointing"] = False
    groups = [fixtures.handmade_group(f"meta_{i}", i * 2) for i in range(4)]
    model = fixtures.tiny_model()
    model.eval()
    with torch.no_grad():
        ref = method.base.logits(model, groups[1], c, "split_rank").numpy().copy()
    episode = method.Episode(*groups, ref, groups[1].uid)
    # Deliberately large inner rates on the tiny model make missing Hessian paths
    # detectable. They are not proposed hyperparameters for BGE training.
    settings = method.Settings(.03, .04, .3, .7)
    return c, episode, model, settings


class GroupMetaTests(unittest.TestCase):
    def test_analytic_counterexample_and_detach_mutation(self):
        p = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        support = lambda x: .5 * (x["p"] - 1).square()
        query = lambda x: .5 * (x["p"] + 1).square()
        total, ls, lq, fast = method.lookahead({"p": p}, support, query, {"p": .2}, .7)
        actual = torch.autograd.grad(total, p)[0].item()
        self.assertAlmostEqual(fast["p"].item(), .44, places=12)
        self.assertLess(support(fast).item(), ls.item())
        self.assertGreater(lq.item(), query({"p": p}).item())
        self.assertAlmostEqual(actual, .1064, places=12)
        # Independent scalar finite difference through the actual inner formula.
        fn = lambda x: .5 * (x - 1) ** 2 + .7 * .5 * (x - .2 * (x - 1) + 1) ** 2
        finite = (fn(.300001) - fn(.299999)) / .000002
        self.assertAlmostEqual(actual, finite, places=8)
        detached_gradient = -.7 + .7 * 1.44
        unadapted_gradient = -.7 + .7 * 1.3
        self.assertGreater(abs(actual - detached_gradient), .1)
        self.assertGreater(abs(actual - unadapted_gradient), .1)
        self.assertGreater(actual, 0)  # Outer correction reverses support-only descent.

    def test_whole_group_identity_and_reference_binding(self):
        _, episode, _, _ = setup()
        episode.validate()
        with self.assertRaises(ValueError):
            replace(episode, current_query=episode.current_support).validate()
        impostor = replace(episode.current_query, sellers=episode.current_support.sellers)
        with self.assertRaises(ValueError):
            replace(episode, current_query=impostor).validate()
        with self.assertRaises(ValueError):
            replace(episode, reference_uid=episode.history_query.uid).validate()
        with self.assertRaises(ValueError):
            replace(episode, reference=np.zeros(378, dtype=np.float64)).validate()

    def test_actual_four_group_graph_hessian_and_query_routing(self):
        c, episode, model, settings = setup()
        model.eval()  # Full autograd remains enabled; removes stochastic fixtures.
        original = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        loss, terms, fast = method.objective(model, episode, c, settings, (1, 2, 3, 4))
        params = tuple(model.parameters())
        direct = torch.autograd.grad(terms["support"], params, retain_graph=True)
        full = torch.autograd.grad(loss, params, retain_graph=True)
        # q-gradient w.r.t. fast params is the first-order approximation. Compare
        # its composed update to the full gradient: the Hessian must matter.
        q_fast = torch.autograd.grad(terms["query"], tuple(fast.values()), retain_graph=True)
        for prefix in ("encoder.", "head."):
            indices = [i for i, (name, _) in enumerate(model.named_parameters()) if name.startswith(prefix)]
            increment = sum(float((full[i] - direct[i]).square().sum()) for i in indices)
            correction = sum(float((full[i] - direct[i] - settings.outer_weight * q_fast[i]).square().sum())
                             for i in indices)
            self.assertGreater(increment, 1e-12)
            self.assertGreater(correction, 1e-14)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        # A different historical query must change the objective and gradients.
        altered = replace(episode, history_query=fixtures.handmade_group("different", 9))
        other, _, _ = method.objective(model, altered, c, settings, (1, 2, 3, 4))
        other_grad = torch.autograd.grad(other, tuple(model.parameters()))
        self.assertGreater(sum(float((a - b).square().sum()) for a, b in zip(full, other_grad)), 1e-10)
        # Numeric differentiation of the actual four-group scalar objective.
        parameter = model.head[-1].bias
        analytic = full[-1].item()
        original_bias = parameter.detach().clone()
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                parameter.copy_(original_bias + sign * .002)
            value, _, _ = method.objective(model, episode, c, settings, (1, 2, 3, 4))
            values.append(float(value.detach()))
        with torch.no_grad():
            parameter.copy_(original_bias)
        self.assertAlmostEqual(analytic, (values[0] - values[1]) / .004, delta=.002)

    def test_zero_beta_exact_logit_gradient_and_single_adam_update(self):
        c, episode, model, settings = setup()
        model.train()
        zero = replace(settings, outer_weight=0.)
        loss, _, _ = method.objective(model, episode, c, zero, (17, 18, 19, 20))
        actual = torch.autograd.grad(loss, tuple(model.parameters()))
        torch.manual_seed(17)
        cs = method.base.logits(model, episode.current_support, c, "split_rank")
        lc = method.parent.ranking.objectives(cs, torch.tensor(episode.current_support.labels, dtype=cs.dtype), .5)["total"]
        torch.manual_seed(18)
        hs = method.base.logits(model, episode.history_support, c, "split_rank")
        lh = method.parent.ranking.objectives(hs, torch.tensor(episode.history_support.labels, dtype=hs.dtype), .5)["total"]
        expected = torch.autograd.grad(lc + .1 * lh + .5 * (hs - torch.tensor(episode.reference)).square().mean(),
                                       tuple(model.parameters()))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-7)
        optimizer = method.core.make_optimizer(model, c)
        fixtures.toy_prior_step(model, optimizer, episode.current_support, c)
        before = copy.deepcopy(model.state_dict())
        with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step, \
                mock.patch.object(torch.nn.utils, "clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
            result = method.update(model, optimizer, episode, c, settings, 2, 1, (21, 22, 23, 24))
        self.assertEqual((step.call_count, clip.call_count, result["adam_step"]), (1, 1, 289))
        for prefix in ("encoder.", "head."):
            self.assertTrue(any(not torch.equal(before[n], v) for n, v in model.state_dict().items()
                                if n.startswith(prefix)))
        # Full-state restoration produces the identical next meta update.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.pt"
            saved = method.core.save_state(path, model, optimizer, {"step": 289})
            expected = method.update(model, optimizer, episode, c, settings, 2, 2, (25, 26, 27, 28))
            digest = method.core.state_digest(model.state_dict())
            method.core.restore_state(path, model, optimizer, saved["state_sha256"])
            restored = method.update(model, optimizer, episode, c, settings, 2, 2, (25, 26, 27, 28))
            self.assertEqual(expected, restored)
            self.assertEqual(digest, method.core.state_digest(model.state_dict()))

    def test_stage_routing_pairing_and_memory_lifecycle(self):
        c, episode, model, settings = setup()
        optimizer = method.core.make_optimizer(model, c)
        fixtures.toy_prior_step(model, optimizer, episode.current_support, c)
        memory = method.parent.Memory("ABC", 20260930, True, {"a": 1., "b": 0.})
        first = [fixtures.handmade_group(f"old_{i:02}") for i in range(48)]
        current = [fixtures.handmade_group(f"new_{i:02}") for i in range(48)]
        memory.retain(first, 1, lambda rows: np.zeros((len(rows), 378), np.float32))
        origins = copy.deepcopy(memory.references)
        calls = []
        def spy(model, optimizer, episode, config, settings, stage, step, seeds, check):
            episode.validate()
            self.assertEqual(settings, method.Settings(.03, .04, .3, .7))
            np.testing.assert_array_equal(episode.reference, memory.references[episode.history_support.uid])
            calls.append(episode)
            return {"stage": stage, "step": step}
        with mock.patch.object(method, "update", side_effect=spy):
            result = method.train_stage(model, optimizer, current, memory, c, "ABC", 2, settings)
        self.assertEqual(result["physical_updates"], 288)
        self.assertEqual(result["gradient_group_presentations"], 1152)
        for role in (0, 2):
            counts = {g.uid: sum(e.groups()[role].uid == g.uid for e in calls) for g in current}
            self.assertEqual(set(counts.values()), {6})
        for name, value in origins.items():
            np.testing.assert_array_equal(value, memory.references[name])
        restored = method.parent.Memory.from_bytes(memory.to_bytes())
        self.assertEqual(memory.to_bytes(), restored.to_bytes())
        memory.retain(current, 2, lambda rows: np.ones((len(rows), 378), np.float32))
        for uid in set(origins) & set(memory.references):
            np.testing.assert_array_equal(memory.references[uid], origins[uid])
        self.assertLessEqual(len(memory.to_bytes()), 1048576)
        # Spy is routing evidence only, explicitly not 288 real meta updates.


if __name__ == "__main__":
    unittest.main()
