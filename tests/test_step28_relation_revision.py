"""Incremental handwritten checks; run only on Linux py310, without formal data."""
from __future__ import annotations

import copy
from dataclasses import replace
import itertools
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_relation_revision as revision
from test_step28_relation_memory import handmade_group, tiny_model, random_z

old = revision.original


def explicit_losses(scores, labels):
    edges = list(itertools.combinations(range(28), 2))
    bce = sum(math.log1p(math.exp(s)) - y * s for s, y in zip(scores, labels)) / 378
    query, hard, all_pairs = [], [], []
    for q in range(28):
        pos = [scores[i] for i, edge in enumerate(edges) if q in edge and labels[i]]
        neg = [scores[i] for i, edge in enumerate(edges) if q in edge and not labels[i]]
        query.append(math.log(sum(math.exp(s) for s in pos + neg)) - sum(pos) / len(pos))
        hard.append(sum(math.log1p(math.exp(n-p)) for p in pos
                        for n in sorted(neg, reverse=True)[:5]) / (len(pos)*5))
        all_pairs.append(sum(math.log1p(math.exp(n-p)) for p in pos for n in neg)
                         / (len(pos)*len(neg)))
    return bce + sum(query)/28 + .5*sum(hard)/28, sum(all_pairs)/28


def fixture(model, group, c):
    x = old.reference(model, group, c)
    h, b, constant = old.statistics(torch.from_numpy(x), group.labels)
    memory = old.Memory("ABC")
    # Kernel-only synthetic one-group statistic, not a serialized stage state.
    memory.h, memory.b, memory.constant, memory.count = h.numpy(), b.numpy(), constant, 1
    return memory, x


class RevisionTests(unittest.TestCase):
    def test_independent_objectives_and_margin_direction(self):
        group = handmade_group("formula")
        scores = torch.linspace(-1.3, 1.4, 378, dtype=torch.float64, requires_grad=True)
        expected_current, expected_rank = explicit_losses(scores.detach().tolist(), group.labels)
        self.assertAlmostEqual(float(revision.current_objective(scores, group.labels)["total"]), expected_current, places=11)
        self.assertAlmostEqual(float(revision.historical_ranking(scores, group.labels)), expected_rank, places=11)
        # All positive margins already exceed 2; larger correct margins stay beneficial.
        labels = torch.tensor(group.labels, dtype=torch.float64)
        confident = (3 * (2*labels-1)).requires_grad_()
        gradient = torch.autograd.grad(revision.historical_ranking(confident, group.labels), confident)[0]
        self.assertTrue(bool((gradient[labels.bool()] < 0).all()))
        self.assertTrue(bool((gradient[~labels.bool()] > 0).all()))

    def test_projection_blind_counterexample_is_seen_by_rank(self):
        group = handmade_group("blind")
        edges = list(itertools.combinations(range(28), 2))
        x = torch.stack((.5*(2*torch.tensor(group.labels, dtype=torch.float64)-1), torch.ones(378, dtype=torch.float64)))
        idx = [edges.index(e) for e in [(0, 3), (0, 1), (4, 5)]]
        x[0, idx[1]] = 0
        direction = torch.zeros_like(x)
        direction[0, idx] = torch.tensor([.2, -.4, .2], dtype=torch.float64)
        w = torch.tensor([1., 0.], dtype=torch.float64)
        h, b, c = old.statistics(x, group.labels)
        torch.testing.assert_close(old.historical_loss(x+direction, w, x, h, b, c, 1),
                                   old.historical_loss(x, w, x, h, b, c, 1))
        amplitude = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        penalty = revision.historical_ranking((x+amplitude*direction).T@w, group.labels)
        self.assertGreater(float(penalty), float(revision.historical_ranking(x.T@w, group.labels)))
        self.assertGreater(float(torch.autograd.grad(penalty, amplitude)[0]), 0.)

    def test_serial_kernel_matches_joint_and_both_history_paths_are_live(self):
        c = old.config()
        model = tiny_model()
        joint = copy.deepcopy(model)
        previous, current = handmade_group("history"), handmade_group("current")
        controls = [i for i, n in enumerate([3]*4 + [2]*8) for _ in range(n)]
        controls = controls[1:] + controls[:1]
        previous = replace(previous, labels=tuple(
            int(controls[i] == controls[j]) for i, j in itertools.combinations(range(28), 2)))
        previous.validate()
        self.assertNotEqual(previous.labels, current.labels)
        self.assertEqual(sum(previous.labels), 20)
        memory, x = fixture(model, previous, c)
        # Nonzero drift makes the compressed historical path observable.
        with torch.no_grad():
            model.weight.add_(.04)
            joint.weight.add_(.04)
        joint_optimizer = old.make_optimizer(joint, c)
        y = old.relation_features(joint, previous, c)
        terms = revision.history_objective(y, joint.weight, x, memory, previous.labels)
        probes = (joint.encoder.embedding.weight, joint.head[0].weight, joint.weight)
        for name in ("compressed", "rank"):
            grads = torch.autograd.grad(terms[name], probes, retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() and float(g.norm()) > 0 for g in grads))
        z = old.relation_features(joint, current, c)
        # Independent aggregation: never use the production history total/weight.
        # The unchanged Q kernel and B components have separate formula evidence.
        q = old.historical_loss(y, joint.weight, x, memory.h, memory.b,
                                memory.constant, memory.count)
        scores = y.T @ joint.weight
        edges = list(itertools.combinations(range(28), 2))
        query_losses = []
        for query in range(28):
            positive = [i for i, edge in enumerate(edges) if query in edge and previous.labels[i]]
            negative = [i for i, edge in enumerate(edges) if query in edge and not previous.labels[i]]
            query_losses.append(torch.stack([
                torch.nn.functional.softplus(scores[n] - scores[p])
                for p in positive for n in negative]).mean())
        r = torch.stack(query_losses).mean()
        current_terms = revision.current_objective(z.T@joint.weight, current.labels)
        b = current_terms["bce"] + current_terms["rank"] + .5 * current_terms["hard"]
        total = b + q + .1 * r
        total.backward()
        expected = {name: p.grad.detach().clone() for name, p in joint.named_parameters()}
        torch.nn.utils.clip_grad_norm_(joint.parameters(), 1.)
        joint_optimizer.param_groups[0]["lr"] = 1e-5
        joint_optimizer.step()
        original_history = revision.history_objective
        original_clip = torch.nn.utils.clip_grad_norm_

        def execute(mode):
            network = copy.deepcopy(model)
            optimizer = old.make_optimizer(network, c)
            captured = {}

            def capture(parameters, *args, **kwargs):
                captured.update({name: p.grad.detach().clone()
                                 for name, p in network.named_parameters()})
                return original_clip(parameters, *args, **kwargs)

            def history(*args):
                values = original_history(*args[:-1], current.labels if mode == "wrong_labels" else args[-1])
                if mode == "omit_rank":
                    values["total"] = values["compressed"]
                elif mode == "detach_rank":
                    values["total"] = values["compressed"] + .1 * values["rank"].detach()
                elif mode == "wrong_weight":
                    values["total"] = values["compressed"] + .2 * values["rank"]
                return values

            with mock.patch.object(revision, "history_objective", side_effect=history), \
                    mock.patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=capture) as clip, \
                    mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
                record = revision.optimization_step(network, optimizer, current, previous, x, memory, c, 1e-5)
            self.assertEqual(clip.call_count, 1)
            self.assertEqual(step.call_count, 1)
            self.assertEqual(captured.keys(), expected.keys())
            return network, record, captured

        def compare_gradients(actual):
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=2e-7, rtol=3e-5,
                                           msg=lambda text: f"preclip gradient {name}: {text}")

        network, record, captured = execute("correct")
        compare_gradients(captured)
        self.assertEqual(record["adam_step"], 1)
        self.assertAlmostEqual(record["total"], float(total.detach()), places=5)
        self.assertAlmostEqual(record["history_compressed"], float(q.detach()), places=6)
        self.assertAlmostEqual(record["history_rank"], float(r.detach()), places=6)
        self.assertAlmostEqual(record["history_total"], float((q+.1*r).detach()), places=6)
        self.assertEqual(record["history_rank_weight"], .1)
        self.assertEqual(record["history_uid"], previous.uid)
        for a, b in zip(network.parameters(), joint.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-4)
        for fault in ("omit_rank", "detach_rank", "wrong_weight", "wrong_labels"):
            with self.subTest(fault=fault):
                _, _, gradients = execute(fault)
                with self.assertRaisesRegex(AssertionError, "preclip gradient"):
                    compare_gradients(gradients)
                print(f"K1 rejected {fault} by preclip gradient", flush=True)

    def test_cumulative_statistic_changes_gradient_and_actual_update(self):
        c = old.config()
        group, current = handmade_group("history"), handmade_group("current")
        model = tiny_model()
        memory, x = fixture(model, group, c)
        changed = copy.deepcopy(memory)
        # Quadratic-statistic kernel fixture, not two consistent consolidated histories.
        h, b, constant = old.statistics(torch.from_numpy(random_z(91)), group.labels)
        changed.h, changed.b, changed.constant = h.numpy(), b.numpy(), constant
        y = old.relation_features(model, group, c)
        first = revision.history_objective(y, model.weight, x, memory, group.labels)
        second = revision.history_objective(y, model.weight, x, changed, group.labels)
        torch.testing.assert_close(first["rank"], second["rank"])
        g1 = torch.autograd.grad(first["total"], model.weight, retain_graph=True)[0]
        g2 = torch.autograd.grad(second["total"], model.weight)[0]
        self.assertGreater(float((g1-g2).norm()), 1e-5)
        alternative = copy.deepcopy(model)
        for network, state in ((model, memory), (alternative, changed)):
            revision.optimization_step(network, old.make_optimizer(network, c), current,
                                       group, x, state, c, 1e-5)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(model.parameters(), alternative.parameters())))

    def test_stage_entry_consolidation_and_restore(self):
        c, model, memory = old.config(), tiny_model(), old.Memory("ABC")
        optimizer = old.make_optimizer(model, c)
        for stage in (1, 2):
            groups = [handmade_group(f"stage{stage}_{i:02}") for i in range(48)]
            records = revision.train_stage(model, optimizer, groups, memory, c, stage)
            self.assertEqual(len(records), 288)
            self.assertEqual(records[-1]["adam_step"], stage*288)
            self.assertEqual(memory.count, stage*48)
            self.assertTrue(all((r["history_uid"] is None) == (stage == 1) for r in records))
            if stage == 2:
                self.assertTrue(all(r["history_rank"] > 0 for r in records))
            payload = memory.to_bytes()
            self.assertLessEqual(len(payload), 1048576)
            self.assertEqual(old.Memory.from_bytes(payload).to_bytes(), payload)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
