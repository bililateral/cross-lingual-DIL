"""Handwritten discriminating evidence for risk replay, never formal samples."""
from __future__ import annotations

import copy
import itertools
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_risk_replay as method
import test_step28_bge_continual_contracts as fixtures


def scalar_risks(scores: np.ndarray, labels: tuple) -> tuple[np.ndarray, np.ndarray]:
    pairs = list(itertools.combinations(range(28), 2))
    result, degrees = [], []
    for query in range(28):
        selected = [k for k, pair in enumerate(pairs) if query in pair]
        pos = [float(scores[k]) for k in selected if labels[k]]
        neg = [float(scores[k]) for k in selected if not labels[k]]
        maximum = max(pos + neg)
        rank = maximum + math.log(sum(math.exp(v - maximum) for v in pos + neg)) - sum(pos) / len(pos)
        result.append([rank, sum(float(np.logaddexp(0, -v)) for v in pos) / len(pos),
                       sum(float(np.logaddexp(0, v)) for v in neg) / len(neg)])
        degrees.append(len(pos))
    return np.asarray(result), np.asarray(degrees)


def scalar_penalty(scores: np.ndarray, labels: tuple, ref: dict) -> float:
    risks, degrees = scalar_risks(scores, labels)
    total = 0.
    for degree in (1, 2):
        for channel in range(3):
            ordered = sorted(risks[degrees == degree, channel].tolist())
            for k, value in enumerate(ordered):
                total += max(0., value - ref[str(degree)][k][channel]) ** 2 / 84
    return total


def prepared() -> tuple:
    c, old, new = fixtures.config(), fixtures.handmade_group("old"), fixtures.handmade_group("new", 2)
    model = fixtures.tiny_model()
    optimizer = method.core.make_optimizer(model, c)
    fixtures.toy_prior_step(model, optimizer, old, c)
    return c, old, new, model, optimizer


class RiskReplayTests(unittest.TestCase):
    def test_scalar_formula_and_all_logit_derivatives(self):
        group = fixtures.handmade_group()
        values = np.random.default_rng(19).normal(size=378)
        ref = method.make_reference(np.zeros(378, dtype=np.float32), group.labels)
        predicted = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        risks, counts = method.query_risks(predicted, torch.tensor(group.labels))
        expected, degrees = scalar_risks(values, group.labels)
        np.testing.assert_allclose(risks.detach(), expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(counts, degrees)
        loss = method.distribution_penalty(risks, counts, ref).mean()
        self.assertAlmostEqual(float(loss), scalar_penalty(values, group.labels, ref), places=12)
        loss.backward()
        differences = []
        for index in range(378):
            delta = np.zeros(378); delta[index] = 1e-5
            differences.append((scalar_penalty(values + delta, group.labels, ref)
                                - scalar_penalty(values - delta, group.labels, ref)) / 2e-5)
        np.testing.assert_allclose(predicted.grad, differences, rtol=1e-5, atol=2e-9)

    def test_tail_improvement_permutation_strata_and_channels(self):
        counts = torch.tensor([1] * 16 + [2] * 12)
        risks = torch.arange(1, 85, dtype=torch.float64).reshape(28, 3) / 100
        ref = {"1": risks[:16].tolist(), "2": risks[16:].tolist()}
        for candidate in (risks, risks * .8, torch.cat((risks[:16].flip(0), risks[16:].flip(0)))):
            self.assertEqual(float(method.distribution_penalty(candidate, counts, ref).sum()), 0.)
        tail = risks.clone(); tail[0, 0] -= .005; tail[15, 0] += .005
        self.assertAlmostEqual(float(tail[:, 0].mean()), float(risks[:, 0].mean()))
        penalties = method.distribution_penalty(tail, counts, ref)
        self.assertAlmostEqual(float(penalties[0]), .005 ** 2 / 28)
        self.assertEqual(float(penalties[1:].sum()), 0.)
        swapped = risks.clone(); swapped[[0, 27]] = swapped[[27, 0]]
        self.assertGreater(float(method.distribution_penalty(swapped, counts, ref).sum()), 0.)
        for channel in range(3):
            degraded = risks.clone(); degraded[:, channel] += 1
            result = method.distribution_penalty(degraded, counts, ref)
            self.assertAlmostEqual(float(result[channel]), 1.)
            self.assertAlmostEqual(float(result.sum()), 1.)

    def test_real_update_gradient_decomposition_and_optimizer(self):
        c, old, new, model, optimizer = prepared()
        reference = {"1": [[0.] * 3 for _ in range(16)], "2": [[0.] * 3 for _ in range(12)]}
        expected = copy.deepcopy(model)
        expected_optimizer = method.core.make_optimizer(expected, c)
        expected_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for pg, rate in zip(expected_optimizer.param_groups, (method.parent.stage_lr(1), .001)):
            pg["lr"] = rate
        expected.train(); expected_optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(31)
        cs = method.base.logits(expected, new, c, "split_rank")
        current_loss = method.parent.ranking.objectives(cs, torch.tensor(new.labels), .5)["total"]
        torch.manual_seed(32)
        hs = method.base.logits(expected, old, c, "split_rank")
        old_loss = method.parent.ranking.objectives(hs, torch.tensor(old.labels), .5)["total"]
        rr, counts = method.query_risks(hs, torch.tensor(old.labels))
        keep = method.distribution_penalty(rr, counts, reference).mean()
        params = tuple(expected.parameters())
        grads = torch.autograd.grad(.5 * keep, params, retain_graph=True)
        self.assertGreater(float(grads[0].norm()), 0.)
        self.assertGreater(sum(float(g.norm()) for g in grads[1:]), 0.)
        (current_loss + .1 * old_loss + .5 * keep).backward()
        full_gradients = [p.grad.clone() for p in params]
        torch.nn.utils.clip_grad_norm_(expected.parameters(), c["optimizer"]["clip_norm"])
        expected_optimizer.step()
        captured = []
        original = torch.nn.utils.clip_grad_norm_
        def capture(parameters, *args, **kwargs):
            parameters = list(parameters)
            captured.extend([p.grad.clone() for p in parameters])
            return original(parameters, *args, **kwargs)
        with mock.patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=capture) as clip, \
                mock.patch.object(optimizer, "step", wraps=optimizer.step) as step, \
                mock.patch.object(method.base, "logits", wraps=method.base.logits) as forwards:
            row = method.update(model, optimizer, new, old, reference, c, 2, 1, 31, 32)
        self.assertEqual((clip.call_count, step.call_count, forwards.call_count), (1, 1, 2))
        for actual, target in zip(captured, full_gradients):
            torch.testing.assert_close(actual, target, rtol=2e-5, atol=2e-6)
        for actual, target in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(actual, target, rtol=0, atol=1e-7)
        self.assertAlmostEqual(row["retention"], float(keep), places=6)
        self.assertEqual(method.parent.adam_step(optimizer), 289)

    def test_zero_weight_matches_er_and_term_changes_both_modules(self):
        import step28_er_weight as er
        c, old, new, initial, opt = prepared()
        ref = {"1": [[0.] * 3 for _ in range(16)], "2": [[0.] * 3 for _ in range(12)]}
        outputs = []
        for variant in ("er", "zero", "risk"):
            model = copy.deepcopy(initial)
            optimizer = method.core.make_optimizer(model, c)
            optimizer.load_state_dict(copy.deepcopy(opt.state_dict()))
            if variant == "er":
                er.update(model, optimizer, new, old, c, .1, 2, 1, 3, 4)
            else:
                method.update(model, optimizer, new, old, ref, c, 2, 1, 3, 4,
                              retention_weight=.5 if variant == "risk" else 0.)
            outputs.append(model)
        for a, b in zip(outputs[0].parameters(), outputs[1].parameters()):
            self.assertTrue(torch.equal(a, b))
        for name in ("encoder", "head"):
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
                getattr(outputs[1], name).parameters(), getattr(outputs[2], name).parameters())))

    def test_actual_stage_memory_lifecycle_and_restore(self):
        c, old, new, model, optimizer = prepared()
        policy = method.parent.contract()
        first = [fixtures.handmade_group(f"a{i:02}") for i in range(48)]
        second = [fixtures.handmade_group(f"b{i:02}", 1) for i in range(48)]
        memory = method.RiskMemory("ABC", policy["memory_seed"], {"a": 1., "b": 0.})
        memory.retain(first, 1, model, optimizer, c)
        origins = copy.deepcopy(memory.cache.auxiliary["risk_replay"]["groups"])
        baseline = method.parent.Memory("ABC", policy["memory_seed"], False, {"a": 1., "b": 0.})
        baseline.retain(first, 1, None); baseline.begin_stage(2)
        with mock.patch.object(method, "update", wraps=method.update) as calls:
            trained = method.train_stage(model, optimizer, second, memory, c, "ABC", 2)
        self.assertEqual(calls.call_count, 288)
        self.assertEqual(trained["history_ids"], [baseline.draw()[0].uid for _ in range(288)])
        self.assertEqual(method.parent.adam_step(optimizer), 576)
        self.assertEqual(trained["updates"][-1]["encoder_lr"], 0.)
        with mock.patch.object(method.parent.ranking, "score", wraps=method.parent.ranking.score) as score:
            memory.retain(second, 2, model, optimizer, c)
        live = memory.cache.auxiliary["risk_replay"]["groups"]
        self.assertEqual(set(live), {g.uid for g in memory.cache.reservoir.groups})
        for uid in set(origins) & set(live):
            self.assertEqual(origins[uid], live[uid])
        self.assertEqual({g.uid for g in score.call_args.args[1]}, set(live) - set(origins))
        restored = method.RiskMemory.from_bytes(memory.to_bytes())
        self.assertEqual(restored.to_bytes(), memory.to_bytes())
        self.assertLess(len(memory.to_bytes()), 1048576)
        memory.begin_stage(3); restored.begin_stage(3)
        for _ in range(3):
            a, ar = memory.draw(); b, br = restored.draw()
            self.assertEqual((a.uid, ar), (b.uid, br))
        restored.cache.auxiliary["oversized"] = "x" * 1048576
        with self.assertRaises(ValueError):
            restored.to_bytes()

    def test_invalid_reference_and_labels_fail_before_update(self):
        c, old, new, model, optimizer = prepared()
        with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaises(ValueError):
                method.update(model, optimizer, new, old, {"1": []}, c, 2, 1, 1, 2)
            self.assertEqual(step.call_count, 0)
        with self.assertRaises(ValueError):
            method.query_risks(torch.zeros(378), torch.zeros(378))

    def test_model_adam_and_memory_restore_preserves_next_update(self):
        c, old, new, model, optimizer = prepared()
        policy = method.parent.contract()
        memory = method.RiskMemory("ABC", policy["memory_seed"], {"a": 1., "b": 0.})
        memory.retain([fixtures.handmade_group(f"a{i:02}") for i in range(48)], 1,
                      model, optimizer, c)
        memory.begin_stage(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            saved = method.core.save_state(path, model, optimizer, {"handwritten": True})
            payload = memory.to_bytes()
            group, ref = memory.draw()
            first = method.update(model, optimizer, new, group, ref, c, 2, 1, 11, 12)
            after = method.core.state_digest(model.state_dict())
            after_opt = method.core.state_digest(optimizer.state_dict())
            method.core.restore_state(path, model, optimizer, saved["state_sha256"])
            restored = method.RiskMemory.from_bytes(payload)
            group, ref = restored.draw()
            second = method.update(model, optimizer, new, group, ref, c, 2, 1, 11, 12)
            self.assertEqual(first, second)
            self.assertEqual(after, method.core.state_digest(model.state_dict()))
            self.assertEqual(after_opt, method.core.state_digest(optimizer.state_dict()))
            self.assertEqual(memory.to_bytes(), restored.to_bytes())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
