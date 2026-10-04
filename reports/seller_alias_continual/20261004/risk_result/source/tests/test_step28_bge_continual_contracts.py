"""Handmade contracts for the new sequential/replay computation, not formal data."""
from __future__ import annotations

import copy
import hashlib
import itertools
import math
from pathlib import Path
from typing import Any
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_bge_continual as method
import step28_bge_continual_run as runner
import step28_bge_continual_evaluate as evaluation


def handmade_group(uid: str = "handmade", shift: int = 0) -> method.data.Group:
    owners = [i for i, n in enumerate([3] * 4 + [2] * 8) for _ in range(n)]
    labels = tuple(int(owners[i] == owners[j]) for i, j in itertools.combinations(range(28), 2))
    group = method.data.Group(uid, tuple(f"{uid}_seller_{i:02}" for i in range(28)),
                              tuple(tuple((f"{uid}_item_{i:02}_{j}",
                                           f"人工商品{(i + shift) % 9}款式{j}",
                                           f"手写描述，款式{owners[i]}，记录{i}，内容{j + shift}。")
                                          for j in range(2 + i % 2)) for i in range(28)), labels)
    group.validate()
    return group


def scalar_losses(values: np.ndarray, labels: tuple[int, ...]) -> dict:
    """Independent scalar edge traversal, inherited formula expressed without Torch."""
    pairs = list(itertools.combinations(range(28), 2))
    scores = np.asarray(values, dtype=np.float64)
    bce = sum(float(np.logaddexp(0., s)) - y * s for s, y in zip(scores, labels, strict=True)) / 378
    ranks, hard = [], []
    for query in range(28):
        edges = [(i, v if u == query else u) for i, (u, v) in enumerate(pairs) if query in (u, v)]
        positive = [i for i, _ in edges if labels[i] == 1]
        negative = sorted(((i, other) for i, other in edges if labels[i] == 0),
                          key=lambda row: (-scores[row[0]], row[1]))[:5]
        maximum = max(scores[i] for i, _ in edges)
        ranks.append(maximum + math.log(sum(math.exp(scores[i] - maximum) for i, _ in edges))
                     - sum(scores[i] for i in positive) / len(positive))
        hard.append(sum(float(np.logaddexp(0., scores[neg] - scores[pos]))
                        for pos in positive for neg, _ in negative) / (5 * len(positive)))
    rank, difficult = sum(ranks) / 28, sum(hard) / 28
    return {"bce": bce, "rank": rank, "hard": difficult, "total": bce + rank + .5 * difficult}


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 4)
        self.dropout = torch.nn.Dropout(.2)

    def tokenizer(self, texts: list[str], **kwargs) -> dict:
        return {"input_ids": torch.tensor([[v % 64 for v in hashlib.sha256(t.encode()).digest()[:5]]
                                            for t in texts])}

    def forward(self, features: dict) -> dict:
        return {"sentence_embedding": self.dropout(self.embedding(features["input_ids"])).mean(1)}


def tiny_model() -> Any:
    torch.manual_seed(93)
    return method.core.build_model(TinyEncoder(), 16, 7)


def config() -> dict:
    result = method.config(method.contract())
    result["input"]["encoder_bf16"] = False
    return result


def toy_prior_step(model: Any, optimizer: Any, group: Any, c: dict, count: int = 288) -> None:
    """One real update then a declared handmade Adam-counter fixture, not 288 updates.

    Moment tensors come from actual optimization. Counter rebasing isolates the
    next-stage operation cheaply; another test exercises all actual 288 calls.
    """
    method.update(model, optimizer, group, None, None, c, "seq", 1, 1, 51, 52)
    for state in optimizer.state.values():
        state["step"].fill_(count)


def synthetic_matrices() -> tuple[dict, list[str]]:
    domains = ["ABC"[i % 3] for i in range(60)]
    arrays = {}
    for j, point in enumerate(evaluation.expected_points()):
        values = np.zeros((60, 22), dtype=np.float64)
        for g in range(60):
            values[g] = .2 + .001 * j + .0001 * g + np.arange(22) * .00001
        arrays[point] = {role: values.copy() for role in method.ROLES}
    return arrays, domains


class HandmadeBudget:
    def check(self, reserve: int = 0) -> None:
        pass


def gate_fixture(root: Path) -> tuple[dict, dict]:
    """Explicit file-identity fixtures, NOT model training or checkpoint evidence."""
    p = method.contract()
    for folder in ("scores", "maps", "points", "updates", "models", "branches", "memory"):
        (root / folder).mkdir()
    groups = {role: [handmade_group(f"{role}_{d}_{i:02}") for d in "ABC" for i in range(count)]
              for role, count in (("fit", 48), ("calibration", 12), ("development", 20))}
    partition = {role: [{"group_uid": g.uid, "domain": g.uid.split("_")[1]} for g in rows]
                 for role, rows in groups.items()}
    method.data.write_json(root / "partition.json", partition)
    raw = np.tile(np.linspace(-1, 1, 378, dtype=np.float32), (60, 1))
    manifest = {"status": runner.COMPLETE, "source_files": runner.sources(), "physical_updates": 6048,
                "gradient_group_presentations": 9504, "points": {}, "training": {}, "memories": {},
                "partition": method.data.record(root / "partition.json", root),
                "initial": {"scores": runner.save_array(root / "scores/initial.npy", raw, root)}}
    memories = {}
    for order in method.ORDERS:
        first = [g for g in groups["fit"] if g.uid.split("_")[1] == order[0]]
        for arm in ("er", "logit"):
            memory = method.Memory(order, p["memory_seed"], arm == "logit", {"a": 1., "b": 0.})
            memory.retain(first, 1, lambda gs: np.zeros((len(gs), 378), np.float32))
            memories[order, arm] = memory
            key = order + "_" + arm + "_after1"
            manifest["memories"][key] = runner.save_memory(root, key, memory)
    for name in evaluation.expected_points():
        order, stage = name[:3], 1 if name.endswith("_shared") else int(name[-1])
        arm = "shared" if stage == 1 else name.split("_")[1]
        current = [g for g in groups["fit"] if g.uid.split("_")[1] == order[stage - 1]]
        sequence, stream = method.schedule(current, p, order, stage)
        raw_rec = runner.save_array(root / "scores" / (name + "_raw.npy"), raw, root)
        cal_rec = runner.save_array(root / "scores" / (name + "_calibration.npy"), raw[:12], root)
        mapping = {"a": 1., "b": 0., "status": "PASS_CALIBRATION_FIT", "group_count": 12,
                   "pair_count": 4536, "positive_count": 240, "actual_domain": order[stage - 1],
                   "model_state_sha256": "handmade_fixture_not_weights", "score_source": cal_rec,
                   "calibration_group_ids": [g.uid for g in groups["calibration"] if g.uid.split("_")[1] == order[stage - 1]]}
        map_path = root / "maps" / (name + ".json")
        method.data.write_json(map_path, mapping)
        weight = root / "models" / (name + ".pt")
        weight.write_bytes(b"handmade file identity fixture, not torch weights")
        point = {"name": name, "stage": stage, "order": order, "actual_domain": order[stage - 1],
                 "completed_updates": stage * 288, "full_model_adam_and_rng_restore_verified": True,
                 "model_state_sha256": "handmade_fixture_not_weights", "model": method.data.record(weight, root),
                 "mapping": method.data.record(map_path, root), "first_map_parameters": {"a": 1., "b": 0.},
                 "scores": {"development": raw_rec, "calibration": cal_rec}}
        if stage == 1:
            full = root / "branches" / (name + ".pt")
            full.write_bytes(b"handmade full file identity fixture, not torch state")
            point["full_checkpoint"] = method.data.record(full, root)
        for role in ("stage-cal", "first-cal"):
            point["scores"][role] = runner.save_array(root / "scores" / (name + "_" + role + ".npy"), raw.astype(np.float64), root)
        history, memory_summary = [], None
        if arm in ("er", "logit"):
            memory = memories[order, arm]
            memory.begin_stage(stage)
            history = [memory.draw()[0].uid for _ in range(288)]
            memory_summary = memory.summary()
        values = np.zeros((288, len(method.STEP_COLUMNS)), np.float64)
        values[:, 0:3] = 1.
        values[:, 3] = 2.5
        if history:
            values[:, 4:7], values[:, 7] = 1., 2.5
        values[:, 9] = values[:, 3] + values[:, 7]
        values[:, 10] = [method.stage_lr(s) for s in range(1, 289)]
        values[:, 11], values[:, 12] = .001, 1.
        log = {"order": order, "stage": stage, "method": arm, "actual_domain": order[stage - 1],
               "updates": 288, "adam_step": 288 * stage, "update_columns": list(method.STEP_COLUMNS),
               "current_ids": [g.uid for g in sequence], "history_ids": history,
               "memory_after_training": memory_summary,
               "update_file": runner.save_array(root / "updates" / (name + ".npy"), values, root)}
        if arm in ("er", "logit") and stage == 2:
            memories[order, arm].retain(current, 2, lambda gs: np.zeros((len(gs), 378), np.float32))
            manifest["memories"][name] = runner.save_memory(root, name, memories[order, arm])
        point_path, log_path = root / "points" / (name + ".json"), root / "updates" / (name + ".json")
        method.data.write_json(point_path, point)
        method.data.write_json(log_path, log)
        manifest["points"][name], manifest["training"][name] = method.data.record(point_path, root), method.data.record(log_path, root)
    return manifest, p


class ContinualContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_confirmed_contract_and_stage_schedule(self) -> None:
        p = method.contract()
        self.assertEqual(p["physical_updates"], 3 * 288 + 3 * 3 * 2 * 288)
        self.assertEqual(sum(method.stage_lr(i) > 0 for i in range(1, 289)), 287)
        self.assertAlmostEqual(sum(method.stage_lr(i) for i in range(1, 289)), .00144, places=15)
        self.assertEqual(method.stage_lr(288), 0)
        self.assertEqual(method.stage_lr(29), 1e-5)
        for bad in (0, 289, 1.0):
            with self.assertRaises(ValueError):
                method.stage_lr(bad)

    def test_supply_cannot_rewind_or_skip(self) -> None:
        groups, partition = {}, {}
        for role, count in (("fit", 48), ("calibration", 12)):
            groups[role] = [handmade_group(f"{role}_{d}_{i:02}") for d in "ABC" for i in range(count)]
            partition[role] = [{"group_uid": g.uid, "domain": d}
                               for d in "ABC" for g in groups[role] if g.uid.startswith(role + "_" + d)]
        supply = method.Supply(groups, partition)
        with self.assertRaises(ValueError):
            supply.current("ABC_shared", "ABC", 2)
        fit, cal = supply.current("ABC_shared", "ABC", 1)
        self.assertEqual((len(fit), len(cal)), (48, 12))
        self.assertTrue(all(g.uid.startswith("fit_A") for g in fit))
        supply.branch_after_first("ABC_er", "ABC_shared")
        with self.assertRaises(ValueError):
            supply.current("ABC_er", "BCA", 2)
        fit, _ = supply.current("ABC_er", "ABC", 2)
        self.assertTrue(all(g.uid.startswith("fit_B") for g in fit))
        with self.assertRaises(ValueError):
            supply.current("ABC_er", "ABC", 1)

    def test_reservoir_pairing_restore_and_nonrefresh(self) -> None:
        first = [handmade_group(f"A_{i:02}") for i in range(48)]
        second = [handmade_group(f"B_{i:02}") for i in range(48)]
        mem = [method.Memory("ABC", 20260930, use, {"a": .5, "b": -.2}) for use in (False, True)]
        first_values = np.arange(378, dtype=np.float32)
        for m in mem:
            m.retain(first, 1, lambda groups: np.stack([first_values for _ in groups]))
        self.assertEqual(mem[0].summary()["members"], mem[1].summary()["members"])
        old = {uid: row.copy() for uid, row in mem[1].references.items()}
        for m in mem:
            m.begin_stage(2)
        for _ in range(288):
            self.assertEqual(mem[0].draw()[0].uid, mem[1].draw()[0].uid)
        for m in mem:
            with self.assertRaises(ValueError):
                m.draw()
            restored = method.Memory.from_bytes(m.to_bytes())
            self.assertEqual(restored.to_bytes(), m.to_bytes())
            m.retain(second, 2, lambda groups: np.stack([first_values + 10 for _ in groups]))
        self.assertEqual(mem[0].summary()["members"], mem[1].summary()["members"])
        for uid, values in mem[1].references.items():
            np.testing.assert_array_equal(values, old[uid] if uid in old else first_values + 10)
        self.assertEqual(mem[1].reservoir.seen, 96)
        self.assertLess(len(mem[1].to_bytes()), 1048576)

    def test_complete_memory_bytes_not_just_reference_array(self) -> None:
        m = method.Memory("ABC", 1, False, {"a": 1, "b": 0})
        groups = [handmade_group(f"A_{i:02}") for i in range(48)]
        m.retain(groups, 1, None)
        self.assertGreater(len(m.to_bytes()), 6 * 378 * 4)
        g = m.reservoir.groups[0]
        giant = tuple(tuple((uid, title, description + "大" * 10000) for uid, title, description in rows)
                      for rows in g.items)
        m.reservoir.groups[0] = method.data.Group(g.uid, g.sellers, giant, g.labels)
        with self.assertRaises(ValueError):
            m.to_bytes()

    def test_current_schedule_and_dropout_are_paired(self) -> None:
        p = method.contract()
        current = [handmade_group(f"B_{i:02}") for i in range(48)]
        a, stream = method.schedule(current, p, "ABC", 2)
        b, other = method.schedule(list(reversed(current)), p, "ABC", 2)
        self.assertEqual([g.uid for g in a], [g.uid for g in b])
        self.assertEqual(stream, other)
        self.assertTrue(all(sum(g.uid == uid for g in a) == 6 for uid in {g.uid for g in current}))

    def test_live_combined_gradient_matches_independent_graph(self) -> None:
        c, current, history = config(), handmade_group("B", 2), handmade_group("A", 1)
        trained = tiny_model()
        opt = method.core.make_optimizer(trained, c)
        toy_prior_step(trained, opt, history, c)
        reference_model, reference_opt = copy.deepcopy(trained), None
        reference_opt = method.core.make_optimizer(reference_model, c)
        reference_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        target = np.linspace(-.8, .7, 378).astype(np.float32)
        calls = []
        original_logits = method.base.logits

        def observe(model, group, *args, **kwargs):
            calls.append(group.uid)
            return original_logits(model, group, *args, **kwargs)

        with mock.patch.object(method.base, "logits", side_effect=observe):
            got = method.update(trained, opt, current, history, target, c, "logit", 2, 1, 18, 19, observe=True)
        self.assertEqual(calls, ["B", "A"])
        reference_model.train()
        reference_opt.zero_grad(set_to_none=True)
        torch.manual_seed(18)
        pc = original_logits(reference_model, current, c, "split_rank")
        lc = method.ranking.objectives(pc, torch.tensor(current.labels, dtype=torch.float32), .5)["total"]
        torch.manual_seed(19)
        ph = original_logits(reference_model, history, c, "split_rank")
        lh = method.ranking.objectives(ph, torch.tensor(history.labels, dtype=torch.float32), .5)["total"]
        penalty = torch.sum((ph - torch.tensor(target)) ** 2) / 378
        (lc + lh + .5 * penalty).backward()
        torch.nn.utils.clip_grad_norm_(reference_model.parameters(), 1., error_if_nonfinite=True)
        for group, rate in zip(reference_opt.param_groups, (1e-5 / 29, .001), strict=True):
            group["lr"] = rate
        reference_opt.step()
        self.assertAlmostEqual(got["total"], float((lc + lh + .5 * penalty).detach()), places=5)
        for x, y in zip(trained.parameters(), reference_model.parameters(), strict=True):
            torch.testing.assert_close(x, y, rtol=1e-6, atol=2e-7)
        self.assertEqual(method.adam_step(opt), 289)

    def test_history_logit_term_has_its_own_encoder_and_head_gradient(self) -> None:
        model, group = tiny_model(), handmade_group()
        c = config()
        model.train()
        predicted = method.base.logits(model, group, c, "split_rank")
        target = predicted.detach() + torch.linspace(.1, .8, 378)
        penalty = (predicted - target).square().mean()
        params = [model.encoder.embedding.weight, model.head[0].weight]
        for gradient in torch.autograd.grad(penalty, params):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()), 0)
        self.assertFalse(target.requires_grad)

    def test_current_forward_same_with_or_without_replay(self) -> None:
        c, current, old = config(), handmade_group("B"), handmade_group("A")
        first = tiny_model()
        optimizer = method.core.make_optimizer(first, c)
        toy_prior_step(first, optimizer, old, c)
        captures = []
        for arm in ("seq", "er", "logit"):
            model = copy.deepcopy(first)
            opt = method.core.make_optimizer(model, c)
            opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            logits = []
            hook = model.head.register_forward_hook(lambda _, __, out: logits.append(out.detach().clone()))
            try:
                method.update(model, opt, current, old if arm != "seq" else None,
                              np.zeros(378, np.float32) if arm == "logit" else None,
                              c, arm, 2, 1, 107, 108)
            finally:
                hook.remove()
            captures.append(logits[0])
            self.assertEqual(len(logits), 1 if arm == "seq" else 2)
        for got in captures[1:]:
            self.assertTrue(torch.equal(got, captures[0]))

    def test_actual_288_calls_last_encoder_zero_and_adam_continues(self) -> None:
        c, group = config(), handmade_group()
        model = tiny_model()
        optimizer = method.core.make_optimizer(model, c)
        for step in range(1, 289):
            log = method.update(model, optimizer, group, None, None, c, "seq", 1, step,
                                step, step + 1000, observe=step == 288)
        self.assertEqual(method.adam_step(optimizer), 288)
        self.assertFalse(log["modules"]["encoder"]["parameters_changed"])
        self.assertTrue(log["modules"]["head"]["parameters_changed"])
        method.update(model, optimizer, group, None, None, c, "seq", 2, 1, 19, 20)
        self.assertEqual(method.adam_step(optimizer), 289)
        with self.assertRaises(ValueError):
            method.update(model, optimizer, group, None, None, c, "seq", 2, 1, 19, 20)

    def test_full_disk_adam_rng_branch_next_update(self) -> None:
        c, group = config(), handmade_group()
        model = tiny_model()
        opt = method.core.make_optimizer(model, c)
        toy_prior_step(model, opt, group, c)
        state = runner.rng_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared.pt"
            receipt = method.core.save_state(path, model, opt, {"rng": state, "completed_updates": 288})
            clone = tiny_model()
            other_opt = method.core.make_optimizer(clone, c)
            metadata = method.core.restore_state(path, clone, other_opt, receipt["state_sha256"])
            runner.restore_rng(metadata["rng"])
            for m, o in ((model, opt), (clone, other_opt)):
                method.update(m, o, group, None, None, c, "seq", 2, 1, 19, 20)
            self.assertEqual(method.core.state_digest(model.state_dict()), method.core.state_digest(clone.state_dict()))
            self.assertEqual(method.core.state_digest(opt.state_dict()), method.core.state_digest(other_opt.state_dict()))

    def test_actual_checkpoint_path_calibration_and_branch_restore(self) -> None:
        c, group = config(), handmade_group()
        model, budget = tiny_model(), HandmadeBudget()
        optimizer = method.core.make_optimizer(model, c)
        toy_prior_step(model, optimizer, group, c)
        calibration = [handmade_group(f"current_cal_{i:02}", i % 3) for i in range(12)]
        valid = [handmade_group(f"valid_{i:02}", i % 4) for i in range(60)]
        observed, original_score = [], method.ranking.score

        def score(model, rows, *args):
            observed.append([g.uid for g in rows])
            return original_score(model, rows, *args)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("branches", "work", "models", "scores", "maps", "points"):
                (root / folder).mkdir()
            with mock.patch.object(method.ranking, "score", side_effect=score):
                point = runner.checkpoint(root, "ABC_shared", model, optimizer, c, "ABC", 1,
                                          calibration, valid, None, budget)
            self.assertEqual(observed, [[g.uid for g in gs] for gs in (calibration, valid, calibration, valid)])
            self.assertTrue(point["full_checkpoint_retained"])
            with mock.patch.object(method.base, "load_model", side_effect=lambda *a: tiny_model()):
                clone, other = runner.restore_branch(root, point, c)
            self.assertEqual(method.core.state_digest(optimizer.state_dict()), method.core.state_digest(other.state_dict()))
            for m, o in ((model, optimizer), (clone, other)):
                method.update(m, o, group, None, None, c, "seq", 2, 1, 9, 10)
            self.assertEqual(method.core.state_digest(model.state_dict()), method.core.state_digest(clone.state_dict()))
            self.assertEqual(method.core.state_digest(optimizer.state_dict()), method.core.state_digest(other.state_dict()))

    def test_blind_gate_complete_then_rejects_changed_weight_and_wrong_current_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, p = gate_fixture(root)
            scores = runner.blind_gate(root, manifest, p)
            self.assertEqual(len(scores["points"]), 21)
            weight = root / "models/ABC_shared.pt"
            original = weight.read_bytes()
            weight.write_bytes(original + b"changed")
            with self.assertRaises(ValueError):
                runner.blind_gate(root, manifest, p)
            weight.write_bytes(original)
            path = root / "updates/ABC_seq_stage2.json"
            log = method.data.read_json(path)
            log["current_ids"] = [uid.replace("fit_B_", "fit_C_") for uid in log["current_ids"]]
            method.data.write_json(path, log)
            changed = copy.deepcopy(manifest)
            changed["training"]["ABC_seq_stage2"] = method.data.record(path, root)
            with self.assertRaises(ValueError):
                runner.blind_gate(root, changed, p)

    def test_blind_gate_rejects_old_calibration_and_wrong_history_draw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, p = gate_fixture(root)
            path = root / "maps/ABC_seq_stage2.json"
            original_map = path.read_bytes()
            mapping = method.data.read_json(path)
            mapping["calibration_group_ids"][0] = "calibration_A_00"
            method.data.write_json(path, mapping)
            point_path = root / "points/ABC_seq_stage2.json"
            original_point = point_path.read_bytes()
            point = method.data.read_json(point_path)
            point["mapping"] = method.data.record(path, root)
            method.data.write_json(point_path, point)
            changed = copy.deepcopy(manifest)
            changed["points"]["ABC_seq_stage2"] = method.data.record(point_path, root)
            with self.assertRaises(ValueError):
                runner.blind_gate(root, changed, p)
            path.write_bytes(original_map)
            point_path.write_bytes(original_point)
            log_path = root / "updates/ABC_er_stage2.json"
            log = method.data.read_json(log_path)
            log["history_ids"][0] = "fit_C_00"
            method.data.write_json(log_path, log)
            changed = copy.deepcopy(manifest)
            changed["training"]["ABC_er_stage2"] = method.data.record(log_path, root)
            with self.assertRaises(ValueError):
                runner.blind_gate(root, changed, p)

    def test_single_parse_attempt_written_before_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            method.data.write_json(root / "access.json", {"train": 0, "valid": 0, "heldout": 0, "owners": 0})
            def failure(*args):
                self.assertEqual(method.data.read_json(root / "access.json")["valid"], 1)
                raise RuntimeError("handmade failure before any real label")
            with self.assertRaises(RuntimeError):
                runner.parse_once(root, [], {}, "development", failure)
            with self.assertRaises(ValueError):
                runner.parse_once(root, [], {}, "development", failure)
            with self.assertRaises(ValueError):
                runner.parse_once(root, [], {}, "heldout", failure)

    def test_endpoint_group_weights_with_interleaved_domains(self) -> None:
        arrays, domains = synthetic_matrices()
        fields = evaluation.endpoint_fields(arrays, domains, "er", "raw", "O")
        k = method.metrics.COLUMNS.index("map")
        independent = []
        for order in method.ORDERS:
            last = arrays[f"{order}_er_stage3"]["raw"]
            independent.append(sum(sum(last[i, k] for i, d in enumerate(domains) if d == old) / 20
                                   for old in order[:2]) / 2)
        np.testing.assert_allclose(fields[..., k].mean(2).sum(1), independent, rtol=0, atol=1e-15)
        z = evaluation.endpoint_fields(arrays, domains, "er", "raw", "Z")
        all_ = evaluation.endpoint_fields(arrays, domains, "er", "raw", "final_all")
        np.testing.assert_allclose(all_, (2 * fields + z) / 3, rtol=0, atol=1e-15)

    def test_actual_domain_bootstrap_against_frequency_reference(self) -> None:
        arrays, domains = synthetic_matrices()
        fields = evaluation.endpoint_fields(arrays, domains, "logit", "raw", "F_first")
        draws = evaluation.bootstrap_draws()
        got = evaluation.summarize_field(fields, draws)["map"]
        k = method.metrics.COLUMNS.index("map")
        values = fields[..., k].mean(0)
        counts = np.asarray([[np.bincount(draws[b, d], minlength=20) for d in range(3)] for b in range(5000)])
        boot = (counts * values).sum((1, 2)) / 20
        np.testing.assert_allclose(got["conditional_95pct_interval"], np.quantile(boot, [.025, .975]), atol=1e-15, rtol=0)
        self.assertEqual(draws.shape, (5000, 3, 20))

    def test_forgetting_loss_sign_and_frozen_zero(self) -> None:
        arrays, domains = synthetic_matrices()
        for role in method.ROLES:
            self.assertTrue(np.all(evaluation.endpoint_fields(arrays, domains, "frozen", role, "F_first") == 0))
        fields = evaluation.endpoint_fields(arrays, domains, "seq", "raw", "F_first")
        b, m = (method.metrics.COLUMNS.index(n) for n in ("brier", "map"))
        np.testing.assert_allclose(fields[..., b], -fields[..., m], rtol=0, atol=1e-15)

    def test_latest_domain_each_guard_cannot_be_masked_by_N(self) -> None:
        def record(value):
            return {"mean": value, "conditional_95pct_interval": [.001, .1],
                    "per_order": {o: .01 for o in method.ORDERS}}
        delta = {e: {n: record(.01 if n not in ("brier", "log_loss") else -.01)
                     for n in method.metrics.COLUMNS} for e in evaluation.ENDPOINTS}
        raw = copy.deepcopy(delta)
        self.assertTrue(evaluation.comparison_checks(delta, raw)["pilot_observed_checks_pass"])
        for name in ("map", "recall_at_5", "average_precision", "roc_auc", "brier", "log_loss"):
            changed = copy.deepcopy(delta)
            changed["Z"][name]["mean"] *= -1
            self.assertFalse(evaluation.comparison_checks(changed, raw)["pilot_observed_checks_pass"])
        changed = copy.deepcopy(raw)
        changed["O"]["brier"]["mean"] = .001
        self.assertFalse(evaluation.comparison_checks(delta, changed)["pilot_observed_checks_pass"])

    def test_same_path_gate_is_not_same_transition_claim(self) -> None:
        first = {"map": {"mean": .1, "conditional_95pct_interval": [.05, .15],
                         "per_order": {o: .1 for o in method.ORDERS}}}
        gain = {"map": {"per_order": {o: (-.1 + .3) / 2 for o in method.ORDERS}}}
        result = evaluation.forgetting_checks(first, gain)
        self.assertTrue(result["established"])
        self.assertIn("not necessarily same transition", result["claim"])

    def test_collection_survives_first_bootstrap_failure(self) -> None:
        groups = [handmade_group(f"valid_{i:02}") for i in range(60)]
        partition = {"development": [{"group_uid": g.uid, "domain": "ABC"[i % 3]} for i, g in enumerate(groups)]}
        raw = np.tile(np.linspace(-2, 2, 378).astype(np.float32), (60, 1))
        scores = {"initial": raw, "points": {p: {"raw": raw, "stage-cal": raw.astype(float),
                                                 "first-cal": raw.astype(float)} for p in evaluation.expected_points()}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "evaluation"
            runner.collect(root, scores, groups, partition, [])
            self.assertEqual(len(list(root.glob("*.npy"))), 64)
            self.assertEqual(len(list(root.glob("*_counts.json"))), 64)
            with mock.patch.object(evaluation, "bootstrap_draws", side_effect=RuntimeError("handmade statistics fault")):
                with self.assertRaises(RuntimeError):
                    evaluation.finalize(root)
            self.assertTrue((root / "collected.json").is_file())
            self.assertFalse((root / "evaluation.json").exists())

    def test_array_identity_and_schema_before_use(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = runner.save_array(root / "scores.npy", np.zeros((60, 378), np.float32), root)
            runner.array(root, record, (60, 378), np.float32)
            with self.assertRaises(ValueError):
                runner.array(root, record, (60, 378), np.float64)
            raw = bytearray((root / "scores.npy").read_bytes())
            raw[-1] ^= 1
            (root / "scores.npy").write_bytes(raw)
            with self.assertRaises(ValueError):
                runner.array(root, record, (60, 378), np.float32)

    def test_finalization_never_replaces_existing_different_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            evaluation.write_once(path, b"original")
            evaluation.write_once(path, b"original")
            with self.assertRaises(FileExistsError):
                evaluation.write_once(path, b"changed")
            self.assertEqual(path.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
