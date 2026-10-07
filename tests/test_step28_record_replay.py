"""Independent handwritten computation and affected lifecycle checks; Linux only."""
from __future__ import annotations

import copy
import hashlib
import itertools
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import step28_record_replay as method
import step28_record_replay_run as run


def handmade_group(uid: str, records: int = 2, text_length: int = 0) -> method.data.Group:
    controllers = [i for i, size in enumerate([3]*4+[2]*8) for _ in range(size)]
    labels = tuple(int(controllers[i] == controllers[j]) for i, j in itertools.combinations(range(28), 2))
    def text(i, j, channel):
        if text_length:
            prefix = f"{'旧' if 'history' in uid else '新'}款{chr(0x4e00+i)}{chr(0x4e40+j)}{chr(0x4e60+channel)}"
            return prefix + "物"*(text_length-len(prefix))
        return f"手写{uid}账号{i}记录{j}通道{channel}"
    group = method.data.Group(uid, tuple(f"{uid}_s{i:02}" for i in range(28)),
        tuple(tuple((f"{uid}_r{i:02}_{j}", text(i, j, 0), text(i, j, 1)) for j in range(records))
              for i in range(28)), labels)
    group.validate()
    return group


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 4)
        self.dropout = torch.nn.Dropout(.3)

    def tokenizer(self, texts, **kwargs):
        return {"input_ids": torch.tensor([[v % 64 for v in hashlib.sha256(t.encode()).digest()[:5]] for t in texts])}

    def forward(self, batch):
        return {"sentence_embedding": self.dropout(self.embedding(batch["input_ids"])).mean(1)}


def tiny_model():
    torch.manual_seed(41)
    return method.core.build_model(TinyEncoder(), 8, 5)


def literal_scores(table, slots):
    """Enumerate cross-account record pairs; no matrix pooling formula reused."""
    pairs = {pair: table[k] for k, pair in enumerate(itertools.combinations(range(len(slots)), 2))}
    result = []
    for a, b in itertools.combinations(range(28), 2):
        terms = [pairs[tuple(sorted((int(i), int(j))))] for i in np.flatnonzero(slots == a)
                 for j in np.flatnonzero(slots == b)]
        result.append(torch.stack(terms).mean())
    return torch.stack(result)


class RecordReplayTests(unittest.TestCase):
    def test_regroup_indivisible_conservation_and_same_account_terms(self):
        group = handmade_group("geometry")
        a0, a1 = method.assignment(group), method.assignment(group, 17)
        self.assertFalse(np.array_equal(a0, a1))
        self.assertTrue(np.array_equal(np.bincount(a0), np.bincount(a1)))
        controllers = np.array([i for i, size in enumerate([3]*4+[2]*8) for _ in range(size)])
        self.assertTrue(np.array_equal(controllers[a0], controllers[a1]))
        values = torch.arange(1540, dtype=torch.float64)/100
        torch.testing.assert_close(method.account_logits(values, a1), literal_scores(values, a1), rtol=1e-12, atol=1e-12)
        inside = torch.tensor([a0[i] == a0[j] for i, j in itertools.combinations(range(56), 2)])
        missing = values.clone()
        missing[inside] = 0
        self.assertTrue(torch.equal(method.account_logits(values, a0), method.account_logits(missing, a0)))
        self.assertGreater(float((method.account_logits(values, a1)-method.account_logits(missing, a1)).abs().max()), 0)
        unlabelled = method.data.Group(group.uid, group.sellers, group.items)
        np.testing.assert_array_equal(method.assignment(unlabelled), a0)
        with self.assertRaises(ValueError):
            method.assignment(unlabelled, 1)

    def test_chunked_vjp_every_parameter_and_actual_adam_match_direct_autograd(self):
        c = method.config()
        current, history = handmade_group("current"), handmade_group("history")
        chunked = tiny_model()
        teacher = method.reference(chunked, history, c)
        with torch.no_grad():
            chunked.head[0].weight.add_(.08)
        direct = copy.deepcopy(chunked)
        oa, ob = [method.core.make_optimizer(m, c) for m in (chunked, direct)]
        row = method.update(chunked, oa, current, history, teacher, c, "C", "ABC", 2, 1)
        direct.train()
        ob.zero_grad(set_to_none=True)
        for role, group, target in (("current", current, None), ("history", history, teacher)):
            torch.manual_seed(method.data.seed_for(20260918, "ABC", 2, 1, role, "dropout"))
            z = method.record_vectors(direct, group, c)
            # Unchunked graph through the actual head and encoder, with literal bag means.
            i, j = torch.triu_indices(len(z), len(z), 1)
            u, v = z[i], z[j]
            table = direct.head(torch.cat(((u-v).abs(), u*v), 1)).flatten()
            seed = method.data.seed_for(20260918, "ABC", 2, 1, role, group.uid, "regroup")
            slots = [method.assignment(group), method.assignment(group, seed)]
            logits = [literal_scores(table, a) for a in slots]
            truth = torch.tensor(group.labels, dtype=torch.float32)
            supervised = sum(method.parent.ranking.objectives(logit, truth, .5)["total"] for logit in logits)/2
            loss = supervised
            if target is not None:
                mse = sum(((logit-literal_scores(torch.from_numpy(target), a))**2).mean()
                          for logit, a in zip(logits, slots))/2
                loss = .1*supervised+.5*mse
            loss.backward()
        torch.nn.utils.clip_grad_norm_(direct.parameters(), 1., error_if_nonfinite=True)
        for a, b in zip(chunked.parameters(), direct.parameters()):
            self.assertIsNotNone(a.grad)
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-5, atol=2e-7)
        ob.param_groups[0]["lr"] = method.parent.stage_lr(1)
        ob.param_groups[1]["lr"] = .001
        ob.step()
        for a, b in zip(chunked.parameters(), direct.parameters()):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
        self.assertGreater(row["history"]["record_gradient_norm"], 0)
        self.assertGreater(row["history"]["mse1"], 0)

    def test_augmented_constraint_changes_source_gradient_in_original_nullspace(self):
        group = handmade_group("constraint")
        original, regroup = method.assignment(group), method.assignment(group, 17)
        pairs = list(itertools.combinations(range(56), 2))
        k = next(k for k, (i, j) in enumerate(pairs) if original[i] == original[j] and regroup[i] != regroup[j])
        target = np.zeros(len(pairs), np.float32)
        target[k] = 3
        table = torch.linspace(-.2, .3, len(pairs), requires_grad=True)
        target0 = np.zeros_like(target)
        grads = {}
        for arm in ("C", "S"):
            loss, _ = method.objective(table, group, 17, arm, target)
            control, _ = method.objective(table, group, 17, arm, target0)
            grads[arm], = torch.autograd.grad(loss-control, table)
        # Two separately accumulated FP32 graphs can leave roundoff on subtraction.
        self.assertLess(float(grads["S"].abs().max()), 1e-8)
        self.assertAlmostEqual(float(grads["C"][k]), -3/(32*378), delta=1e-8)

    def test_memory_origin_survivors_eviction_draw_and_byte_cap(self):
        first = [handmade_group(f"a{i:02}") for i in range(48)]
        second = [handmade_group(f"b{i:02}") for i in range(48)]
        memory = method.Memory("ABC")
        calls = []
        def teacher(group):
            calls.append(group.uid)
            return np.full(1540, 1 if group.uid.startswith("a") else 2, np.float32)
        memory.retain(first, 1, teacher)
        self.assertEqual(len(calls), 6)
        originals = copy.deepcopy(memory.tables)
        memory.begin_stage(2)
        for _ in range(31):
            memory.draw()
        restored = method.Memory.from_bytes(memory.to_bytes())
        self.assertEqual([memory.draw()[0].uid for _ in range(257)], [restored.draw()[0].uid for _ in range(257)])
        prior_calls = len(calls)
        retention = memory.retain(second, 2, teacher)
        self.assertEqual(len(calls)-prior_calls, len(retention["created"]))
        self.assertGreater(len(retention["evicted"]), 0)
        for uid in originals.keys() & memory.tables.keys():
            np.testing.assert_array_equal(originals[uid], memory.tables[uid])
            self.assertEqual(memory.origins[uid], 1)
        self.assertEqual(memory.to_bytes(), method.Memory.from_bytes(memory.to_bytes()).to_bytes())
        memory.maps["excess"] = "x"*1048576
        with self.assertRaises(ValueError):
            memory.to_bytes()

    def test_full_checkpoint_memory_rng_restore_reproduces_next_real_update(self):
        c = method.config()
        current, history = handmade_group("current"), handmade_group("history")
        model = tiny_model()
        optimizer = method.core.make_optimizer(model, c)
        target = method.reference(model, history, c)
        method.update(model, optimizer, current, history, target, c, "C", "ABC", 2, 1)
        rng = run.previous.rng_state()
        with tempfile.TemporaryDirectory(dir=os.environ["RECORD_REPLAY_TEST_ROOT"]) as tmp:
            path = Path(tmp)/"state.pt"
            state = method.core.save_state(path, model, optimizer, {"rng": rng})
            clone = tiny_model()
            other = method.core.make_optimizer(clone, c)
            meta = method.core.restore_state(path, clone, other, state["state_sha256"])
            run.previous.restore_rng(meta["rng"])
            for m, o in ((model, optimizer), (clone, other)):
                method.update(m, o, current, history, target, c, "C", "ABC", 2, 2)
            self.assertEqual(method.core.state_digest(model.state_dict()), method.core.state_digest(clone.state_dict()))
            self.assertEqual(method.core.state_digest(optimizer.state_dict()), method.core.state_digest(other.state_dict()))

    def test_actual_stage_schedule_and_complete_checkpoint_path(self):
        c = method.config()
        model = tiny_model()
        optimizer = method.core.make_optimizer(model, c)
        current = [handmade_group(f"train{i:02}") for i in range(48)]
        calibration = [handmade_group(f"cal{i:02}") for i in range(12)]
        valid = [handmade_group(f"valid{i:02}") for i in range(60)]
        budget = mock.Mock(snapshot=mock.Mock(return_value={}))
        with tempfile.TemporaryDirectory(dir=os.environ["RECORD_REPLAY_TEST_ROOT"]) as tmp:
            root = Path(tmp)
            for sub in ("updates", "memory", "work", "models", "points", "scores", "maps"):
                (root/sub).mkdir()
            # A real 288-update handwritten first stage exercises the production loop.
            log_rec = run.train_stage(model, optimizer, current, None, c, "C", "ABC", 1, root, budget)
            log = method.data.read_json(root/log_rec["path"])
            self.assertEqual(log["updates"], 288)
            self.assertEqual(method.parent.adam_step(optimizer), 288)
            self.assertEqual(len(set(log["current_ids"])), 48)
            point, memory = run.checkpoint(root, "ABC_shared", model, optimizer, method.Memory("ABC"),
                current, calibration, valid, c, 1, budget)
            self.assertTrue(point["full_restore_verified"])
            self.assertEqual(memory.maps["adam_step"], 288)
            self.assertEqual(memory.reservoir.seen, 48)
            self.assertEqual(point["scores"]["raw"]["bytes"], 90848)

    def test_qualification_gate_requires_distinct_source_matched_cpu_gpu(self):
        p, source = method.policy(), run.sources()
        job = method.data.ROOT/"reports/handwritten_never_execute"
        gate_path = method.data.ROOT/"handwritten_gate.json"
        cpu_path, gpu_path = [method.data.ROOT/(name+".json") for name in ("cpu", "gpu")]
        cpu = {"status": "PASS_HANDWRITTEN_ONLY", "mode": "cpu", "source_files": source}
        gpu = {**cpu, "mode": "gpu", "native": {"kind": "native_record_replay_first_optimizer_step"}}
        gate = {"status": "APPROVED_RECORD_REPLAY_PILOT", "source_files": source, "job": job.relative_to(method.data.ROOT).as_posix(),
                "runtime": p["runtime"], "supervision": p["supervision"], "review_disposition": "NO_OPEN_BLOCKERS",
                "cpu": {"path": cpu_path.name}, "gpu": {"path": gpu_path.name}}
        for evidence, accepts in ((gpu, True), (cpu, False), ({**gpu, "source_files": []}, False)):
            lookup = {method.POLICY: p, gate_path: gate, cpu_path: cpu, gpu_path: evidence}
            with mock.patch.object(run.data, "read_json", side_effect=lambda path: lookup[path]), \
                    mock.patch.object(run.data, "verify", side_effect=lambda path, rec: path):
                if accepts:
                    self.assertEqual(run.validate_gate(job, gate_path), p)
                else:
                    with self.assertRaises(ValueError):
                        run.validate_gate(job, gate_path)

    def test_paired_endpoints_use_actual_domain_and_A2_not_N(self):
        domains = [d for d in "ABC" for _ in range(20)]
        arrays = {}
        for order in method.parent.ORDERS:
            for stage in (1, 2, 3):
                values = np.array([[100*stage+ord(domain)-ord("A")]*22 for domain in domains], np.float64)
                arrays[run.evaluation.point_name(order, "er", stage)] = {role: values.copy() for role in method.parent.ROLES}
        draws = run.evaluation.bootstrap_draws()
        for ep, expected in (("O", 301.), ("N", 251.), ("Z", 301.), ("A2", 201.), ("final_all", 301.)):
            result = run.evaluation.summarize_field(run.fields(arrays, domains, "primary", ep), draws)
            self.assertAlmostEqual(result["map"]["mean"], expected)
            np.testing.assert_allclose(result["map"]["conditional_95pct_interval"], [expected, expected], rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
