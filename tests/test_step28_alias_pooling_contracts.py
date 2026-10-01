"""Hand-checkable pooling, gradients, paired inference and nondegradation guards."""
from __future__ import annotations

import copy
import hashlib
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_alias_pooling as method
import step28_alias_pooling_run as runner


def handmade_group() -> method.data.Group:
    owners = [k for k, size in enumerate([3] * 4 + [2] * 8) for _ in range(size)]
    labels = tuple(int(owners[i] == owners[j]) for i, j in itertools.combinations(range(28), 2))
    items = tuple(tuple((f"item_{i:02}_{j}", f"手工商品{i % 7}类型{j}",
                         f"人工样本描述{owners[i]}，文本条目{i}，内容{j * j}。")
                        for j in range(2 + i % 2)) for i in range(28))
    group = method.data.Group("handmade_pooling", tuple(f"seller_{i:02}" for i in range(28)), items, labels)
    group.validate()
    return group


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)

    def tokenizer(self, texts: list[str], **kwargs) -> dict:
        return {"input_ids": torch.tensor([[hashlib.sha256(text.encode()).digest()[i] % 32
                                            for i in range(3)] for text in texts])}

    def forward(self, features: dict) -> dict:
        return {"sentence_embedding": self.embedding(features["input_ids"]).mean(1)}


def tiny_model(policy: dict, weighted: bool = False) -> torch.nn.Module:
    torch.manual_seed(71)
    model = method.core.build_model(TinyEncoder(), 16, 7)
    if weighted:
        method.attach_aggregation(model, policy, 20260909, dimension=4)
    return model


class PoolingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.policy = method.contract()
        cls.config = method.reference_config(cls.policy, "s0")

    def test_uniform_initialization_rng_and_parameter_count(self):
        torch.manual_seed(57)
        previous = torch.random.get_rng_state().clone()
        pool = method.make_aggregation(1024, self.policy, 20260909)
        self.assertTrue(torch.equal(previous, torch.random.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in pool.parameters()), 131200)
        for n in (2, 3, 8):
            vectors = torch.arange(2 * n * 1024, dtype=torch.float32).reshape(2 * n, 1024) + 1
            original = method.base.pool_channels(vectors, [n], "separate_moments")
            candidate = method.pool_channels(vectors, [n], pool)
            torch.testing.assert_close(original, candidate, atol=1e-6, rtol=1e-5)
            weights = pool(vectors[:n], vectors[n:])
            torch.testing.assert_close(weights, torch.full((n,), 1 / n))

    def test_weighted_moments_against_independent_scalar_formula(self):
        raw = np.array([[1, 2], [3, -1], [2, 4], [-2, 3], [4, 1],
                        [2, -1], [1, 3], [4, -2], [2, 5], [3, 2]], dtype=np.float64)
        normalized = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        reference = []
        offset = 0
        for n, weights in ((2, [.3, .7]), (3, [.2, .3, .5])):
            parts = []
            for shift in (0, 5):
                x = normalized[shift + offset:shift + offset + n]
                means = [sum(weights[i] * x[i, j] for i in range(n)) for j in range(2)]
                stds = [np.sqrt(sum(weights[i] * (x[i, j] - means[j])**2 for i in range(n)) + 1e-8) for j in range(2)]
                parts.extend(means + stds)
            reference.append(np.asarray(parts) / np.linalg.norm(parts))
            offset += n

        def weights(title, description):
            return title.new_tensor([.3, .7] if len(title) == 2 else [.2, .3, .5])

        actual = method.pool_channels(torch.tensor(raw, dtype=torch.float32), [2, 3], weights)
        np.testing.assert_allclose(actual.detach().numpy(), reference, atol=2e-7, rtol=2e-6)

    def test_joint_permutation_and_weight_bounds(self):
        pool = method.make_aggregation(4, self.policy, 41)
        with torch.no_grad():
            pool.output.weight.fill_(.3)
        vectors = torch.arange(40, dtype=torch.float32).reshape(10, 4) / 9 - 1
        original = method.pool_channels(vectors, [2, 3], pool)
        permutation = torch.tensor([1, 0, 4, 2, 3])
        shuffled = torch.cat((vectors[:5][permutation], vectors[5:][permutation]))
        torch.testing.assert_close(original, method.pool_channels(shuffled, [2, 3], pool))
        for n in (2, 3):
            w = pool(vectors[:n], vectors[5:5+n])
            self.assertTrue(bool((w >= .5 / n).all()))
            self.assertAlmostEqual(float(w.sum()), 1., places=6)
        with self.assertRaises(ValueError):
            method.pool_channels(vectors, [2, 2], pool)

    def test_all_text_pairing_and_no_identity_input(self):
        group = handmade_group()
        records = [record for account in group.items for record in account]
        texts = method.base.record_texts(group, "separate_moments")
        self.assertEqual(texts, [r[1] for r in records] + [r[2] for r in records])
        self.assertNotIn(group.uid, " ".join(texts))
        self.assertTrue(all(seller not in " ".join(texts) for seller in group.sellers))

    def test_original_d_and_wrapper_really_match_one_adam_update(self):
        group = handmade_group()
        original, wrapped = tiny_model(self.policy), tiny_model(self.policy)
        optimizer = method.core.make_optimizer(original, self.config)
        wrapped_optimizer = method.make_optimizer(wrapped, self.config, self.policy)
        left = method.base.update(original, optimizer, group, self.config, "split_rank", 19, observe=True)
        right = method.update(wrapped, wrapped_optimizer, group, self.config, 19, observe=True)
        self.assertEqual(left, right)
        self.assertEqual(method.core.state_digest(original.state_dict()), method.core.state_digest(wrapped.state_dict()))
        self.assertEqual(method.core.state_digest(optimizer.state_dict()), method.core.state_digest(wrapped_optimizer.state_dict()))

    def test_new_module_optimizer_gradients_and_two_real_updates(self):
        model = tiny_model(self.policy, weighted=True)
        optimizer = method.make_optimizer(model, self.config, self.policy)
        optimized = [p for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(optimized), len({id(p) for p in optimized}))
        self.assertEqual({id(p) for p in optimized}, {id(p) for p in model.parameters()})
        log = method.update(model, optimizer, handmade_group(), self.config, 21, observe=True)
        norms = log["modules"]["aggregation"]["parameter_gradient_norms_before_clip"]
        self.assertEqual(norms["hidden.weight"], 0.)
        self.assertEqual(norms["hidden.bias"], 0.)
        self.assertGreater(norms["output.weight"], 0.)
        log = method.update(model, optimizer, handmade_group(), self.config, 22, observe=True)
        for value in log["modules"]["aggregation"]["parameter_gradient_norms_before_clip"].values():
            self.assertGreater(value, 0.)
        self.assertTrue(all(float(state["step"]) == 2 for state in optimizer.state.values()))

    def test_checkpoint_restores_aggregation_and_adam_not_only_encoder(self):
        model = tiny_model(self.policy, weighted=True)
        optimizer = method.make_optimizer(model, self.config, self.policy)
        method.update(model, optimizer, handmade_group(), self.config, 25)
        prediction = method.score(model, [handmade_group()], self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handmade.pt"
            saved = method.core.save_state(path, model, optimizer, {"handmade": True})
            with torch.no_grad():
                model.aggregation.output.weight.add_(1)
            self.assertEqual(method.core.restore_state(path, model, optimizer, saved["state_sha256"]), {"handmade": True})
            np.testing.assert_array_equal(prediction, method.score(model, [handmade_group()], self.config))

    def test_shared_common_initialization_and_fixed_partition(self):
        plain = tiny_model(self.policy)
        augmented = tiny_model(self.policy, weighted=True)
        self.assertEqual(method.common_digest(plain), method.common_digest(augmented))
        for seed in method.SEED_IDS:
            config = method.reference_config(self.policy, seed)
            self.assertEqual(config["partition"], self.config["partition"])
            self.assertEqual(config["optimizer"], self.config["optimizer"])
        self.assertEqual(len(method.NEW_RUNS) * 144 * 6, 4320)
        self.assertNotIn("s0_d", method.NEW_RUNS)

    def test_bootstrap_averages_seeds_before_groups_hand_case(self):
        delta = np.zeros((3, 60, 22))
        for seed, shift in enumerate((-.003, 0, .003)):
            for domain, value in enumerate((.01, .02, .03)):
                delta[seed, domain*20:(domain+1)*20, :] = value + shift
        result = method.paired_summary(delta, list("A"*20 + "B"*20 + "C"*20), self.policy["evaluation"])
        for record in result["metrics"].values():
            self.assertAlmostEqual(record["mean"], .02)
            np.testing.assert_allclose(record["per_seed"], [.017, .02, .023], atol=1e-15)
            np.testing.assert_allclose(record["conditional_95pct_interval"], [.02, .02], atol=1e-15)
        with self.assertRaises(ValueError):
            method.paired_summary(delta[:, :-1], list("A"*20 + "B"*20 + "C"*19), self.policy["evaluation"])

    def test_acceptance_guards_basic_identification_and_fixed_s0(self):
        record = {"metrics": {key: {"mean": .01 * sign, "per_seed": [.01 * sign] * 3,
                                    "conditional_95pct_interval": [-.1, .1]}
                               for key, sign in method.GUARD_SIGNS.items()}}
        record["metrics"]["map"] = {"mean": .02, "per_seed": [.02]*3, "conditional_95pct_interval": [.005, .035]}
        self.assertTrue(method.acceptance(record, self.policy["evaluation"])["passed"])
        for metric, sign in method.GUARD_SIGNS.items():
            with self.subTest(metric=metric):
                changed = copy.deepcopy(record)
                changed["metrics"][metric]["mean"] = -.00001 * sign
                self.assertFalse(method.acceptance(changed, self.policy["evaluation"])["passed"])
                changed = copy.deepcopy(record)
                changed["metrics"][metric]["per_seed"][0] = -.00001 * sign
                self.assertFalse(method.acceptance(changed, self.policy["evaluation"])["passed"])
        for field, value in (("mean", .009), ("per_seed", [.04, .03, -.001]), ("conditional_95pct_interval", [0., .04])):
            changed = copy.deepcopy(record)
            changed["metrics"]["map"][field] = value
            self.assertFalse(method.acceptance(changed, self.policy["evaluation"])["passed"])

    def test_collection_saved_before_uncertainty_failure(self):
        group = handmade_group()
        truth = np.tile(group.labels, (60, 1)).astype(np.uint8)
        scores = np.where(truth, 2., -2.).astype(np.float32)
        arrays = {run_id: {"points": {"3": scores, "6": scores}, "threshold": 0.}
                  for run_id in method.ALL_RUNS}
        part = {"development": [{"group_uid": f"hand_{i}", "domain": "ABC"[i // 20]} for i in range(60)]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "evaluation"
            destination.mkdir()
            policy = copy.deepcopy(self.policy)
            policy["historical_d"]["evaluation_root"] = "old"
            with mock.patch.object(method, "paired_summary", side_effect=RuntimeError("injected uncertainty failure")):
                collected = runner.collect(truth, arrays, part, destination, policy, {"handmade": True})
                self.assertEqual(len(list(destination.glob("*/epoch*_metrics.npy"))), 12)
                self.assertTrue((destination / "collected.json").is_file())
                (root / "old/split_rank").mkdir(parents=True)
                policy["historical_d"]["files"] = []
                for epoch in ("3", "6"):
                    path = root / "old/split_rank" / f"epoch{epoch}_metrics.npy"
                    np.save(path, np.load(destination / "s0_d" / f"epoch{epoch}_metrics.npy", allow_pickle=False), allow_pickle=False)
                    policy["historical_d"]["files"].append(method.data.record(path, root))
                path = root / "old/collected.json"
                method.data.write_json(path, {"group_ids": collected["group_ids"], "domains": collected["domains"],
                                             "arms": {"split_rank": collected["runs"]["s0_d"]}})
                policy["historical_d"]["files"].append(method.data.record(path, root))
                collected["policy"] = policy
                method.data.write_json(destination / "collected.json", collected)
                with mock.patch.object(method.data, "ROOT", root), mock.patch.object(method, "contract", return_value=policy), \
                     mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No labels in saved-result processing")):
                    with self.assertRaisesRegex(RuntimeError, "injected uncertainty"):
                        runner.finalize(destination)
                    self.assertTrue((destination / "historical_alignment.json").is_file())
                    self.assertFalse((destination / "evaluation.json").exists())
                self.assertEqual(len(collected["runs"]), 6)
                for run in collected["runs"].values():
                    self.assertEqual(len(run["automatic_classification"]["counts_by_group"]), 60)


if __name__ == "__main__":
    unittest.main()
