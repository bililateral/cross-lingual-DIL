"""Handmade inputs only: objectives, actual updates, restore and evaluation boundaries."""
from __future__ import annotations

import copy
import hashlib
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
import step28_alias_ranking as method
import step28_alias_ranking_run as runner

PAIRS = list(itertools.combinations(range(28), 2))


def handmade_group() -> method.data.Group:
    components = [k for k, size in enumerate([3] * 4 + [2] * 8) for _ in range(size)]
    labels = tuple(int(components[i] == components[j]) for i, j in PAIRS)
    items = tuple(tuple((f"item_{i:02}_{j}", f"手工商品{i % 7}类型{j}",
                         f"人工样本描述{components[i]}，文本条目{i}，内容{j * j}。")
                        for j in range(2 + i % 2)) for i in range(28))
    group = method.data.Group("handmade_ranking", tuple(f"seller_{i:02}" for i in range(28)), items, labels)
    group.validate()
    return group


def scalar_objectives(values: np.ndarray, labels: tuple[int, ...], weight: float) -> dict:
    """Independent scalar edge traversal; no production candidate selector/reducer."""
    scores = np.asarray(values, dtype=np.float64)
    bce = sum(float(np.logaddexp(0., s)) - y * s for s, y in zip(scores, labels, strict=True)) / 378
    ranks, hard = [], []
    gradient = np.zeros(378, dtype=np.float64)
    for query in range(28):
        edges = [(index, pair[1] if pair[0] == query else pair[0])
                 for index, pair in enumerate(PAIRS) if query in pair]
        positives = [index for index, _ in edges if labels[index] == 1]
        negatives = sorted(((index, other) for index, other in edges if labels[index] == 0),
                           key=lambda row: (-scores[row[0]], row[1]))[:5]
        local = [scores[index] for index, _ in edges]
        maximum = max(local)
        ranks.append(maximum + math.log(sum(math.exp(s - maximum) for s in local))
                     - sum(scores[index] for index in positives) / len(positives))
        comparisons = []
        for pos in positives:
            for neg, _ in negatives:
                difference = scores[neg] - scores[pos]
                comparisons.append(float(np.logaddexp(0., difference)))
                derivative = math.exp(-float(np.logaddexp(0., -difference))) / (28 * 5 * len(positives))
                gradient[neg] += derivative
                gradient[pos] -= derivative
        hard.append(sum(comparisons) / len(comparisons))
    rank, difficult = sum(ranks) / 28, sum(hard) / 28
    return {"bce": bce, "rank": rank, "hard": difficult,
            "total": bce + rank + weight * difficult, "hard_gradient": gradient}


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)

    def tokenizer(self, texts: list[str], **kwargs) -> dict:
        return {"input_ids": torch.tensor([[hashlib.sha256(text.encode()).digest()[i] % 32
                                            for i in range(3)] for text in texts])}

    def forward(self, features: dict) -> dict:
        return {"sentence_embedding": self.embedding(features["input_ids"]).mean(1)}


def tiny_model() -> torch.nn.Module:
    torch.manual_seed(71)
    return method.core.build_model(TinyEncoder(), 16, 7)


def handmade_run_files(root: Path, policy: dict) -> tuple[dict, dict, list[Path]]:
    """Receipt fixtures, explicitly not trained models or formal execution evidence."""
    group = handmade_group()
    selected, partition = {}, {}
    for role, count in method.base.ROLE_SIZES.items():
        selected[role] = [method.data.Group(f"hand_{role}_{i:03}", group.sellers, group.items) for i in range(count)]
        partition[role] = [{"group_uid": g.uid, "domain": "ABC"[i // (count // 3)]}
                           for i, g in enumerate(selected[role])]
    method.data.write_json(root / "partition.json", partition)
    receipt = root / "handmade_receipt.json"
    method.data.write_json(receipt, {"status": "HANDMADE_FIXTURE_NOT_AUTHORIZATION_OR_TRAINING"})
    preflights = {run: {"initial_state_sha256": "handmade_equal_initial_state"} for run in method.RUNS}
    manifest = {"status": runner.COMPLETE, "source_files": runner.sources(), "policy": policy,
                "inputs": {"handmade": True}, "runs": {}, "physical_updates": 7776,
                "label_parses": {"train": 1, "development": 0, "heldout": 0, "owners": 0},
                "native_verification": method.data.record(receipt, method.data.ROOT),
                "authorization": method.data.record(receipt, method.data.ROOT),
                "partition": method.data.record(root / "partition.json", root), "preflights": preflights}
    weights = []
    for run_id in method.RUNS:
        seed, kind = method.split_run(run_id)
        config = method.reference_config(policy, seed)
        folder = root / run_id
        (folder / "models").mkdir(parents=True)
        (folder / "scores").mkdir()
        schedule, stream = method.base.schedule(selected["fit"], config)
        updates = np.zeros((864, 6), dtype=np.float64)
        updates[:, :2] = 1.
        updates[:, 2] = 1. if kind == "hard" else 0.
        updates[:, 3] = 2.5 if kind == "hard" else 2.
        updates[:, 4] = [method.encoder_lr(policy, kind, step) for step in range(1, 865)]
        updates[:, 5] = .001
        np.save(folder / "updates.npy", updates, allow_pickle=False)
        arm = {"run_id": run_id, "reference_config": config, "updates": 864, "label_parses": 0,
               "dropout_stream": stream, "group_schedule_sha256": hashlib.sha256(
                   method.data.json_bytes([g.uid for g in schedule])).hexdigest(),
               "fit_group_ids": [g.uid for g in selected["fit"]],
               "calibration_group_ids": [g.uid for g in selected["calibration"]],
               "preflight": preflights[run_id], "training": [], "points": {},
               "update_columns": list(runner.UPDATE_COLUMNS), "update_log": method.data.record(folder / "updates.npy", folder),
               "encoder_positive_lr_updates": int((updates[:, 4] > 0).sum())}
        for start in (0, 432):
            observation = {name: {"finite_nonzero_gradient": True, "parameters_changed": True}
                           for name in ("encoder", "head")}
            means = updates[start:start + 432, :4].reshape(3, 144, 4).mean(1)
            arm["training"].append({"start": start, "stop": start + 432, "updates": 432,
                "observations": {str(s): observation for s in runner.OBSERVED_STEPS if start < s <= start + 432},
                "mean_losses_by_epoch": {name: means[:, i].tolist() for i, name in enumerate(method.LOSS_NAMES)}})
        for epoch in (3, 6):
            step = epoch * 144
            weight = folder / "models" / f"epoch{epoch}.pt"
            weight.write_bytes(b"HANDMADE FILE: NOT A MODEL")
            weights.append(weight)
            point = {"run_id": run_id, "epoch": epoch, "metadata": {
                "run_id": run_id, "epoch": epoch, "completed_updates": step,
                "policy_sha256": method.POLICY_SHA256, "reference_config": config,
                "optimizer_lrs": [method.encoder_lr(policy, kind, step), .001],
                "next_encoder_lr": method.encoder_lr(policy, kind, step + 1) if step < 864 else None},
                "full_model_and_adam_reloaded": True, "model_state_sha256": "handmade_state",
                "model": {**method.data.record(weight, folder), "actual_reload_verified": True},
                "scores": {}, "train_metrics": {}}
            for role, count in method.base.ROLE_SIZES.items():
                path = folder / "scores" / f"epoch{epoch}_{role}.npy"
                scores = np.tile(np.where(group.labels, 2., -2.).astype(np.float32), (count, 1))
                np.save(path, scores, allow_pickle=False)
                point["scores"][role] = method.data.record(path, folder)
                if role != "development":
                    path = folder / "scores" / f"epoch{epoch}_{role}_metrics.npy"
                    np.save(path, np.zeros((count, 22), dtype=np.float64), allow_pickle=False)
                    point["train_metrics"][role] = {"file": method.data.record(path, folder)}
            arm["points"][str(epoch)] = point
        labels = np.tile(group.labels, (36, 1)).astype(np.uint8)
        scores = np.where(labels, 2., -2.).astype(np.float32)
        calibrated = method.base.calibrate(labels, scores, [r["domain"] for r in partition["calibration"]])
        calibrated.update(epoch=6, model_state_sha256="handmade_state",
                          score_sha256=arm["points"]["6"]["scores"]["calibration"]["sha256"])
        method.data.write_json(folder / "calibration.json", calibrated)
        arm["calibration"] = method.data.record(folder / "calibration.json", folder)
        method.data.write_json(folder / "manifest.json", arm)
        manifest["runs"][run_id] = {"manifest": method.data.record(folder / "manifest.json", root), "updates": 864}
    method.data.write_json(root / "manifest.json", manifest)
    method.data.write_json(root / "completion.json", {
        "status": runner.COMPLETE, "manifest_sha256": method.data.sha256(root / "manifest.json")})
    return selected, partition, weights


class RankingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.policy = method.contract()
        cls.config = method.reference_config(cls.policy, "s0")
        cls.group = handmade_group()

    def test_all_three_losses_against_scalar_and_analytic_gradient(self):
        scores = np.random.default_rng(73).normal(size=378)
        tensor = torch.tensor(scores, requires_grad=True)
        actual = method.objectives(tensor, torch.tensor(self.group.labels, dtype=torch.float64), .5)
        expected = scalar_objectives(scores, self.group.labels, .5)
        for name in method.LOSS_NAMES:
            self.assertAlmostEqual(float(actual[name]), expected[name], places=12)
        gradient, = torch.autograd.grad(actual["hard"], tensor)
        np.testing.assert_allclose(gradient.numpy(), expected["hard_gradient"], atol=1e-15, rtol=1e-12)
        self.assertTrue((gradient.numpy()[np.asarray(self.group.labels) == 1] < 0).all())
        self.assertTrue((gradient.numpy()[np.asarray(self.group.labels) == 0] >= 0).all())
        self.assertTrue(np.any(gradient.numpy() == 0))

    def test_query_macro_is_not_unweighted_comparison_mean(self):
        labels = np.asarray(self.group.labels)
        scores = np.asarray([2. if y and i < 12 else (-2. if y else 0.)
                             for (i, _), y in zip(PAIRS, labels, strict=True)])
        result = method.objectives(torch.tensor(scores), torch.tensor(labels, dtype=torch.float64), .5)
        triple_loss, pair_loss = float(np.logaddexp(0., -2.)), float(np.logaddexp(0., 2.))
        self.assertAlmostEqual(float(result["hard"]), (12 * triple_loss + 16 * pair_loss) / 28, places=12)
        self.assertGreater(abs(float(result["hard"]) - (24 * triple_loss + 16 * pair_loss) / 40), .1)

    def test_known_negative_exclusion_and_stable_ties(self):
        labels = torch.tensor(self.group.labels)
        scores = torch.where(labels == 1, 100., 0.)
        _, positives, selected = method.hard_candidates(scores, labels)
        for query in range(28):
            expected = [i for i in range(28) if i != query and not bool(positives[query, i])][:5]
            self.assertEqual(selected[query].tolist(), expected)
        self.assertEqual(sum(int(positives[q].sum()) * 5 for q in range(28)), 200)

    def test_unique_score_account_permutation_preserves_loss(self):
        scores = np.random.default_rng(81).normal(size=378)
        permutation = np.random.default_rng(82).permutation(28)
        edge = {pair: index for index, pair in enumerate(PAIRS)}
        indices = [edge[tuple(sorted((int(permutation[i]), int(permutation[j]))))] for i, j in PAIRS]
        labels = np.asarray(self.group.labels)
        a = method.objectives(torch.tensor(scores), torch.tensor(labels, dtype=torch.float64), .5)
        b = method.objectives(torch.tensor(scores[indices]), torch.tensor(labels[indices], dtype=torch.float64), .5)
        for name in method.LOSS_NAMES:
            self.assertAlmostEqual(float(a[name]), float(b[name]), places=12)

    def test_invalid_supervision_and_unapproved_hyperparameters_rejected(self):
        labels = torch.tensor(self.group.labels, dtype=torch.float32)
        scores = torch.zeros(378)
        for weight in (-.5, .1, 1.):
            with self.assertRaises(ValueError):
                method.objectives(scores, labels, weight)
        for bad in (torch.zeros(378), torch.full((378,), 2.)):
            with self.assertRaises(ValueError):
                method.objectives(scores, bad, .5)
        scores[0] = torch.nan
        with self.assertRaises(ValueError):
            method.objectives(scores, labels, .5)

    def test_original_d_exact_actual_adam_and_random_stream(self):
        original, wrapped = tiny_model(), tiny_model()
        left = method.core.make_optimizer(original, self.config)
        right = method.core.make_optimizer(wrapped, self.config)
        original_log = method.base.update(original, left, self.group, self.config, "split_rank", 19, observe=True)
        original_rng = torch.random.get_rng_state().clone()
        wrapped_log = method.update(wrapped, right, self.group, self.config, self.policy, "d", 1, 19, observe=True)
        for key in original_log:
            self.assertEqual(original_log[key], wrapped_log[key])
        self.assertTrue(torch.equal(original_rng, torch.random.get_rng_state()))
        self.assertEqual(method.core.state_digest(original.state_dict()), method.core.state_digest(wrapped.state_dict()))
        self.assertEqual(method.core.state_digest(left.state_dict()), method.core.state_digest(right.state_dict()))

    def test_schedule_all_rates_and_zero_encoder_last_step(self):
        for arm in ("schedule", "hard"):
            rates = [method.encoder_lr(self.policy, arm, step) for step in range(1, 865)]
            self.assertAlmostEqual(rates[0], 1e-5 / 87, places=20)
            self.assertEqual(rates[86], 1e-5)
            self.assertAlmostEqual(rates[87], 1e-5 * 776 / 777, places=20)
            self.assertEqual(rates[-1], 0.)
            self.assertEqual(sum(lr > 0 for lr in rates), 863)
            self.assertTrue(all(a < b for a, b in zip(rates[:86], rates[1:87])))
            self.assertTrue(all(a > b for a, b in zip(rates[86:-1], rates[87:])))
        self.assertTrue(all(method.encoder_lr(self.policy, "d", t) == 2e-5 for t in range(1, 865)))
        for bad in (0, 865, 1.5):
            with self.assertRaises(ValueError):
                method.encoder_lr(self.policy, "hard", bad)
        model = tiny_model()
        optimizer = method.core.make_optimizer(model, self.config)
        before = [method.core.state_digest(module.state_dict()) for module in (model.encoder, model.head)]
        method.set_learning_rate(optimizer, self.policy, "hard", 864)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertEqual(before[0], method.core.state_digest(model.encoder.state_dict()))
        self.assertNotEqual(before[1], method.core.state_digest(model.head.state_dict()))
        self.assertEqual(optimizer.param_groups[1]["lr"], .001)

    def test_actual_hard_training_components_and_optimizer_coverage(self):
        model = tiny_model()
        optimizer = method.core.make_optimizer(model, self.config)
        params = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len({id(p) for p in params}), len(params))
        self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters()})
        model.train()
        predicted = method.base.logits(model, self.group, self.config, "split_rank")
        losses = method.objectives(predicted, torch.tensor(self.group.labels, dtype=torch.float32), .5)
        targets = [model.encoder.embedding.weight, model.head[0].weight]
        for name in ("bce", "rank", "hard"):
            gradients = torch.autograd.grad(losses[name], targets, retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() and float(g.norm()) > 0 for g in gradients))
        for step in (1, 2):
            log = method.update(model, optimizer, self.group, self.config, self.policy, "hard", step, 21 + step, observe=True)
            self.assertTrue(all(r["parameters_changed"] for r in log["modules"].values()))
            self.assertAlmostEqual(log["total"], log["bce"] + log["rank"] + .5 * log["hard"], places=5)
        with self.assertRaises(ValueError):
            method.update(model, optimizer, self.group, self.config, self.policy, "hard", 4, 25)

    def test_actual_restore_continues_adam_and_warmup_boundary(self):
        model = tiny_model()
        optimizer = method.core.make_optimizer(model, self.config)
        for step in range(1, 87):
            method.update(model, optimizer, self.group, self.config, self.policy, "hard", step, 500 + step)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handmade.pt"
            metadata = {"completed_updates": 86, "next_encoder_lr": 1e-5}
            saved = method.core.save_state(path, model, optimizer, metadata)
            restored = tiny_model()
            restored_optimizer = method.core.make_optimizer(restored, self.config)
            self.assertEqual(method.core.restore_state(path, restored, restored_optimizer, saved["state_sha256"]), metadata)
            for step in (87, 88):
                first = method.update(model, optimizer, self.group, self.config, self.policy, "hard", step, 500 + step)
                second = method.update(restored, restored_optimizer, self.group, self.config, self.policy, "hard", step, 500 + step)
                self.assertEqual(first, second)
                self.assertEqual(method.core.state_digest(model.state_dict()), method.core.state_digest(restored.state_dict()))
                self.assertEqual(method.core.state_digest(optimizer.state_dict()), method.core.state_digest(restored_optimizer.state_dict()))
            np.testing.assert_array_equal(method.score(model, [self.group], self.config),
                                          method.score(restored, [self.group], self.config))

    def test_text_only_inputs_and_paired_fixed_schedules(self):
        records = [record for account in self.group.items for record in account]
        texts = method.base.record_texts(self.group, "separate_moments")
        self.assertEqual(texts, [r[1] for r in records] + [r[2] for r in records])
        self.assertNotIn(self.group.uid, " ".join(texts))
        self.assertTrue(all(seller not in " ".join(texts) for seller in self.group.sellers))
        groups = [method.data.Group(f"hand_{i:03}", self.group.sellers, self.group.items) for i in range(144)]
        for seed in method.SEEDS:
            config = method.reference_config(self.policy, seed)
            schedule, stream = method.base.schedule(groups, config)
            self.assertEqual(len(schedule), 864)
            for epoch in range(6):
                self.assertEqual({g.uid for g in schedule[144 * epoch:144 * (epoch + 1)]}, {g.uid for g in groups})
            for arm in method.ARMS:
                paired, paired_stream = method.base.schedule(groups, method.reference_config(self.policy, seed))
                self.assertEqual(([g.uid for g in schedule], stream), ([g.uid for g in paired], paired_stream))
                self.assertEqual(method.split_run(f"{seed}_{arm}"), (seed, arm))

    def test_bootstrap_against_independent_frequency_weights(self):
        values = np.random.default_rng(90).normal(size=(3, 60, 22))
        domains = [d for d in "ABC" for _ in range(20)]
        actual = method.paired_summary(values, domains, self.policy["evaluation"])
        e = self.policy["evaluation"]
        draws = np.random.default_rng(e["valid_bootstrap_seed"]).integers(0, 20, size=(5000, 3, 20))
        weights = np.stack([np.apply_along_axis(lambda x: np.bincount(x, minlength=20), 1, draws[:, d])
                            for d in range(3)], axis=1).reshape(5000, 60)
        for index, name in enumerate(method.metrics.COLUMNS):
            samples = weights @ values[:, :, index].sum(0) / 180
            expected = np.quantile(samples, [.025, .975], method="linear")
            np.testing.assert_allclose(actual["metrics"][name]["conditional_95pct_interval"], expected, atol=1e-15)
            np.testing.assert_allclose(actual["metrics"][name]["per_seed"], values[:, :, index].mean(1), atol=1e-15)
        with self.assertRaises(ValueError):
            method.paired_summary(values, ["A"] * 60, e)

    def test_thirteen_primary_conditions_and_strict_recall(self):
        rows = {name: {"mean": .02, "per_seed": [.02] * 3, "conditional_95pct_interval": [.001, .03]}
                for name in method.metrics.COLUMNS}
        for name in ("brier", "log_loss"):
            rows[name] = {"mean": -.02, "per_seed": [-.02] * 3, "conditional_95pct_interval": [-.03, -.001]}
        summary = {"metrics": rows}
        accepted = method.acceptance(summary, self.policy["evaluation"])
        self.assertTrue(accepted["passed"])
        self.assertEqual(len(accepted["checks"]), 13)
        for field in ("mean", "per_seed"):
            changed = copy.deepcopy(summary)
            changed["metrics"]["recall_at_5"][field] = 0. if field == "mean" else [0., .02, .02]
            self.assertFalse(method.acceptance(changed, self.policy["evaluation"])["passed"])
        for name, sign in method.GUARDS.items():
            changed = copy.deepcopy(summary)
            changed["metrics"][name]["per_seed"][0] = -sign * .0001
            self.assertFalse(method.acceptance(changed, self.policy["evaluation"])["passed"])
        changed = copy.deepcopy(summary)
        changed["metrics"]["map"]["per_seed"][2] = 0.
        self.assertFalse(method.acceptance(changed, self.policy["evaluation"])["passed"])

    def test_all_matrices_survive_statistics_failure_without_reparse(self):
        truth = np.tile(self.group.labels, (60, 1)).astype(np.uint8)
        scores = np.where(truth, 2., -2.).astype(np.float32)
        arrays = {run: {"points": {"3": scores, "6": scores}, "threshold": 0.} for run in method.RUNS}
        partition = {"development": [{"group_uid": f"hand_{i}", "domain": "ABC"[i // 20]} for i in range(60)]}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            runner.collect(truth, arrays, partition, out, self.policy, {"handmade": True})
            self.assertEqual(len(list(out.glob("*/epoch*_metrics.npy"))), 18)
            with mock.patch.object(method, "paired_summary", side_effect=RuntimeError("handmade statistics fault")):
                with self.assertRaisesRegex(RuntimeError, "handmade statistics fault"):
                    runner.finalize(out)
            self.assertTrue((out / "collected.json").exists())
            self.assertFalse((out / "evaluation.json").exists())
            with mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No labels")), \
                 mock.patch.object(method.base.public, "public_inputs", side_effect=AssertionError("No inputs")):
                report = runner.finalize(out)
            self.assertFalse(report["acceptance"]["passed"])
            self.assertEqual(len(report["comparisons"]), 3)
            self.assertTrue(all(row["mean"] == 0 for row in report["comparisons"]["hard_minus_d"]["metrics"].values()))
            with self.assertRaises(FileExistsError):
                runner.finalize(out)

    def test_formal_authorization_missing_and_no_test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not_authorized.json"
            method.data.write_json(path, {"status": "PREPARATION_ONLY"})
            with self.assertRaises(ValueError):
                runner.verify_authorization(path, self.policy)
        self.assertIs(self.policy["supervision"]["formal_access_currently_authorized"], False)
        self.assertIs(self.policy["supervision"]["test_access"], False)
        with mock.patch.object(sys, "argv", ["ranking", "test", "--out", "unused"]):
            with self.assertRaises(SystemExit):
                runner.main()

    def test_each_of_eighteen_files_rejected_before_label_access(self):
        with tempfile.TemporaryDirectory(dir=method.data.ROOT / "reports") as directory:
            root = Path(directory)
            selected, partition, weights = handmade_run_files(root, self.policy)
            budget = mock.Mock()
            with mock.patch.object(method.base, "partition", return_value=(selected, partition)), \
                 mock.patch.object(method.base.public, "public_inputs", return_value=({}, [], {"handmade": True})), \
                 mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("Must not parse")) as labels:
                arrays, _, result = runner.validate_run(root, self.policy, {}, [], {"handmade": True})
                self.assertEqual(len(arrays), 9)
                self.assertEqual(len(result["fresh_model_files_verified_before_valid"]), 18)
                for weight in weights:
                    original = weight.read_bytes()
                    weight.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                    try:
                        with self.assertRaises(ValueError):
                            runner.evaluate(root, root / "evaluation", self.policy, budget)
                        self.assertFalse((root / "evaluation").exists())
                    finally:
                        weight.write_bytes(original)
                labels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
