"""Hand-checkable contracts for the new pure-text population experiment."""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_continual_population as core
import step28_continual_population_data as data
import step28_continual_population_evaluate as evaluate
import step28_continual_population_run as runner

TORCH = importlib.util.find_spec("torch") is not None


def fixture(uid: str = "group") -> data.Group:
    sellers = tuple(f"seller{i:02}" for i in range(28))
    owners = [i // 2 for i in range(16)] + [8 + i // 3 for i in range(12)]
    labels = tuple(int(owners[i] == owners[j]) for i, j in itertools.combinations(range(28), 2))
    items = tuple(tuple((f"{uid}_{i:02}_{j}", f"title {i}", f"description {i} {j}")
                        for j in range(2)) for i in range(28))
    return data.Group(uid, sellers, items, labels)


class DataContracts(unittest.TestCase):
    def test_complete_pair_identity_alignment_not_csv_order(self):
        group = fixture()
        rows = [{"group_uid": group.uid, "seller_uid_left": a, "seller_uid_right": b, "label": str(y)}
                for (a, b), y in zip(itertools.combinations(group.sellers, 2), group.labels)]
        random.Random(9).shuffle(rows)
        self.assertEqual(data.align_labels(group, rows), group.labels)
        self.assertEqual(sum(group.labels), 20)
        wrong = copy.deepcopy(rows)
        wrong[0]["seller_uid_right"] = "unknown"
        with self.assertRaises(ValueError):
            data.align_labels(group, wrong)
        with self.assertRaises(ValueError):
            data.align_labels(group, rows + [rows[0]])

    def test_reversed_edge_and_nonbinary_label_rejected(self):
        group = fixture()
        rows = [{"group_uid": group.uid, "seller_uid_left": b, "seller_uid_right": a, "label": "1"}
                for a, b in itertools.combinations(group.sellers, 2)]
        with self.assertRaises(ValueError):
            data.align_labels(group, rows)
        rows[0]["label"] = "0.0"
        with self.assertRaises(ValueError):
            data.align_labels(group, rows)

    def test_exact_uniform_reservoir_small_enumeration(self):
        # For n=4,k=2 Algorithm R has 3*4 equally likely draw sequences.
        # Each of the six subsets must occur exactly twice, independently of seed.
        counts = {subset: 0 for subset in itertools.combinations(range(4), 2)}
        for choices in itertools.product(range(3), range(4)):
            draws = iter(choices)
            mem = data.Memory(0, capacity=2)
            with patch.object(mem.rng, "randrange", side_effect=lambda n: next(draws)):
                mem.add_stage([fixture(str(i)) for i in range(4)])
            counts[tuple(sorted(int(g.uid) for g in mem.groups))] += 1
        self.assertEqual(set(counts.values()), {2})

    def test_memory_roundtrip_next_selection_and_self_contained_text(self):
        mem = data.Memory(31)
        mem.add_stage([fixture(str(i)) for i in range(11)])
        payload = mem.to_bytes()
        restored = data.Memory.from_bytes(payload)
        self.assertEqual(restored.to_bytes(), payload)
        for g in restored.groups:
            g.validate()
            self.assertEqual(g, fixture(g.uid))
        current = [fixture(str(i)) for i in range(11, 19)]
        mem.add_stage(current)
        restored.add_stage(current)
        self.assertEqual(mem.to_bytes(), restored.to_bytes())
        self.assertEqual(mem.seen, 19)
        with self.assertRaises(ValueError):
            mem.add_stage([mem.groups[0]])

    def test_memory_byte_cap_includes_json_text_labels_rng(self):
        mem = data.Memory(1)
        mem.add_stage([fixture(str(i)) for i in range(6)])
        payload = mem.to_bytes()
        self.assertLess(len(payload), 1048576)
        self.assertIn(b'"rng":', payload)
        self.assertIn(b'"labels":', payload)
        self.assertIn(b'description', payload)
        mem.maximum_bytes = 100
        with self.assertRaises(ValueError):
            mem.to_bytes()

    def test_current_schedule_paired_independent_of_replay_draws(self):
        groups = [fixture(str(i)) for i in range(4)]
        seed = data.seed_for(20260909, "ABC", 2, "current")
        first = data.schedule(groups, 3, seed)
        rng = random.Random(data.seed_for(20260909, "ABC", 2, "replay_draw"))
        for _ in range(100):
            rng.choice(groups)
        second = data.schedule(list(reversed(groups)), 3, seed)
        self.assertEqual([g.uid for g in first], [g.uid for g in second])
        for start in range(0, 12, 4):
            self.assertEqual({g.uid for g in first[start:start + 4]}, {g.uid for g in groups})

    def test_real_new_train_audit_no_valid_test_owners_or_old_labels(self):
        config = data.policy()
        root = data.ROOT / config["data_root"]
        open_original = Path.open
        accessed = []

        def guarded(path, *args, **kwargs):
            resolved = path.resolve()
            self.assertTrue(resolved.is_relative_to(root.resolve()))
            relative = resolved.relative_to(root.resolve()).as_posix()
            self.assertNotIn(relative, {f"{s}/supervision/owners.csv" for s in data.SPLITS} |
                             {"development/supervision/pairs.csv", "heldout/supervision/pairs.csv"})
            accessed.append(relative)
            return open_original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            archive = data.Archive(config, train_labels=True)
        self.assertEqual(archive.train_label_parses, 1)
        self.assertEqual([len(archive.groups(s)) for s in data.SPLITS], [180, 60, 120])
        self.assertEqual(sum(sum(g.labels) for g in archive.groups("train")), 3600)
        measured = []
        for order in config["orders"]:
            mem = data.Memory(data.seed_for(20260909, order, "memory"))
            for domain in order:
                mem.add_stage(archive.groups("train", domain))
                measured.append(len(mem.to_bytes()))
                self.assertEqual(len(mem.groups), 6)
        # Also bound the actual six largest serialized groups, not only chosen ones.
        largest = sorted(archive.groups("train"), key=lambda g: len(data.json_bytes(g.payload())), reverse=True)[:6]
        worst = data.Memory(1)
        worst.add_stage(largest)
        self.assertLessEqual(len(worst.to_bytes()), 1048576)
        print("Actual train audit: 180 groups, 68040 aligned labels; memory bytes by stage:",
              measured, "six largest:", len(worst.to_bytes()))


class MetricContracts(unittest.TestCase):
    def test_ap_pr_auc_roc_ties_against_independent_sklearn(self):
        from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_auc_score
        labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)
        scores = np.array([2., 2., 1., 0., 0., -1.])
        actual = evaluate.curve_metrics(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        self.assertAlmostEqual(actual["average_precision"], average_precision_score(labels, scores))
        self.assertAlmostEqual(actual["trapezoidal_pr_auc"], auc(recall, precision))
        self.assertAlmostEqual(actual["roc_auc"], roc_auc_score(labels, scores))
        self.assertNotAlmostEqual(actual["average_precision"], actual["trapezoidal_pr_auc"])

    def test_logit_threshold_probability_and_confusion(self):
        labels = np.array([1, 0, 1, 0], dtype=np.uint8)
        scores = np.array([0., 0., -np.log(3), np.log(3)])
        result = evaluate.classification(labels, scores)
        self.assertEqual(result["confusion"], {"tp": 1, "fp": 2, "fn": 1, "tn": 0})
        self.assertAlmostEqual(result["brier"], (.25 + .25 + .5625 + .5625) / 4)
        self.assertAlmostEqual(result["log_loss"], (-2 * np.log(.5) - 2 * np.log(.25)) / 4)

    def test_query_metrics_hand_ranked_graph_ties_and_self_exclusion(self):
        # Four accounts, true groups (0,1) and (2,3); all scores tied.
        y = np.array([[1, 0, 0, 0, 0, 1]], dtype=np.uint8)
        observed = evaluate.retrieval(y, np.zeros((1, 6)), 4)[0]
        self.assertAlmostEqual(observed[0], (1 + 1 + 1 / 3 + 1 / 3) / 4)
        self.assertAlmostEqual(observed[1], observed[0])
        self.assertEqual(observed[2], .5)  # Recall@1
        self.assertEqual(observed[3], 1.)  # Recall@3
        self.assertEqual(observed[6], .5)  # NDCG@1
        self.assertEqual(observed[7], .75)  # NDCG@3

    def test_bootstrap_pairs_orders_and_equal_domains(self):
        config = data.policy()["evaluation"]
        # Within each domain the contrast is constant. Unequal group counts must
        # not change equal-domain weighting. Opposite order offsets cancel first.
        means = np.array([1., 3., 3., 5., 5., 5.])
        delta = np.stack((means - 100, means, means + 100))
        rows = [np.array([0]), np.array([1, 2]), np.array([3, 4, 5])]
        result = evaluate.paired_interval(delta, rows, config)
        self.assertEqual(result["mean"], 3.)
        np.testing.assert_allclose(result["conditional_95pct_interval"], [3., 3.])
        self.assertEqual(result["per_order"], [-97., 3., 103.])

    def test_failed_run_stops_before_truth_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "failure.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Failed run"):
                evaluate.validate_run(root, data.policy(), None)
            self.assertFalse((root / "access.json").exists())

    def test_domain_forgetting_takes_max_after_group_aggregation(self):
        history = np.array([[1., 0.], [0., 1.]])
        current = np.array([.4, .4])
        actual = evaluate.old_domain_change(current, history[0], history, np.array([0, 1]))
        self.assertAlmostEqual(actual["current_minus_when_learned"], -.1)
        self.assertAlmostEqual(actual["best_previous_minus_current"], .1)
        self.assertNotAlmostEqual(actual["best_previous_minus_current"], .6)

    def test_nan_and_wrong_pair_count_rejected(self):
        with self.assertRaises(ValueError):
            evaluate.curve_metrics(np.array([1, 0]), np.array([0., np.nan]))
        with self.assertRaises(ValueError):
            evaluate.retrieval(np.zeros((1, 377)), np.zeros((1, 377)), 28)

    def test_all_three_required_cache_points_must_exist(self):
        points = {"initial": {}, "shared": {"memory": {}},
                  "sequential_stage2": {}, "sequential_stage3": {},
                  "er_stage2": {"memory": {}}, "er_stage3": {"memory": {}},
                  "cumulative_stage2": {}, "cumulative_stage3": {}}
        evaluate.require_memory_points(points)
        for name in ("shared", "er_stage2", "er_stage3"):
            with self.subTest(missing=name):
                incomplete = copy.deepcopy(points)
                del incomplete[name]["memory"]
                with self.assertRaisesRegex(ValueError, "required historical memory"):
                    evaluate.require_memory_points(incomplete)
        points["initial"]["memory"] = {}
        with self.assertRaises(ValueError):
            evaluate.require_memory_points(points)
        del points["initial"]["memory"]
        del points["er_stage3"]["memory"]
        config = data.policy()
        manifest = {"status": runner.COMPLETE, "config": config, "source_files": runner.sources(),
                    "label_reads": {"new_train_csv_offline_packaging": 1, "development": 0,
                                    "heldout": 0, "owners": 0, "old_assets": 0},
                    "physical_updates": 5400,
                    "orders": [{"order": o, "points": points, "models": {a: {} for a in config["arms"]}}
                               for o in config["orders"]],
                    "runtime_check": {"status": "PASS_REAL_LABSE_NEW_INPUT_HEAD_MEMORY_ADAM_RELOAD",
                                      "torch_contracts": {"passed": 3, "skipped": 0, "failed": 0},
                                      "formal_initialization_restored": True},
                    "budget": {"elapsed_seconds": 1, "peak_observed_bytes": 1},
                    "evaluation_group_ids": {"development": [], "heldout": []},
                    "verified_input_files": {"train/supervision/pairs.csv": {}},
                    "pretrained_model": config["model"]}
        archive = SimpleNamespace(checked={}, manifest={"files": {"train/supervision/pairs.csv": {}}},
                                  groups=lambda _: [])
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data, "read_json", side_effect=lambda p: manifest if p.name == "manifest.json" else {}):
                with self.assertRaisesRegex(ValueError, "required historical memory"):
                    evaluate.validate_run(Path(tmp), config, archive)


@unittest.skipUnless(TORCH, "Windows environment has no PyTorch; real semantics remain pending Linux")
class TorchContracts(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(111)

    def test_record_multiplicity_symmetric_pairs_and_encoder_gradient(self):
        import torch
        raw = torch.tensor([[2., 0.], [0., 3.], [0., 4.], [1., 0.]], requires_grad=True)
        pooled = core.pool_records(raw, [3, 1])
        expected = torch.tensor([[1., 2.], [1., 0.]])
        expected = torch.nn.functional.normalize(expected, dim=1)
        torch.testing.assert_close(pooled, expected)
        model = core.build_model(torch.nn.Linear(2, 2), 2, 3)
        torch.testing.assert_close(model.pair_logits(pooled), model.pair_logits(pooled.flip(0)))
        pooled[0, 0].backward()
        self.assertGreater(float(raw.grad.norm()), 0.)

    def toy_model(self):
        import torch

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(127, 4)
                self.dropout = torch.nn.Dropout(.25)

            def tokenizer(self, texts, **kwargs):
                self.last_texts = list(texts)
                tokens = [[ord(c) % 127 for c in text] for text in texts]
                length = max(map(len, tokens))
                return {"input_ids": torch.tensor([row + [0] * (length - len(row)) for row in tokens]),
                        "attention_mask": torch.tensor([[1] * len(row) + [0] * (length - len(row)) for row in tokens])}

            def forward(self, features):
                mask = features["attention_mask"].unsqueeze(-1)
                v = self.dropout(self.embedding(features["input_ids"]))
                return {"sentence_embedding": (v * mask).sum(dim=1) / mask.sum(dim=1)}

        return core.build_model(Encoder(), 4, 5)

    def test_separate_branches_match_independent_sum_and_one_adam_update(self):
        import torch
        cfg = data.policy()
        model = self.toy_model()
        reference = copy.deepcopy(model)
        current, replay = fixture("current"), fixture("replay")
        # Change old texts so the historical gradient is not a duplicate branch.
        replay = data.Group(replay.uid, replay.sellers,
                            tuple(tuple((u, t + " OLD", d) for u, t, d in rows) for rows in replay.items), replay.labels)
        opt = core.make_optimizer(model, cfg)
        ref_opt = core.make_optimizer(reference, cfg)
        current_seed, old_seed = 3, 9
        reference.train()
        ref_opt.zero_grad(set_to_none=True)
        torch.manual_seed(current_seed)
        current_logits = core.logits(reference, current, cfg)
        current_loss = torch.nn.functional.binary_cross_entropy_with_logits(current_logits, torch.tensor(current.labels).float())
        torch.manual_seed(old_seed)
        old_logits = core.logits(reference, replay, cfg)
        old_loss = torch.nn.functional.binary_cross_entropy_with_logits(old_logits, torch.tensor(replay.labels).float())
        (current_loss + old_loss).backward()
        expected_grads = [p.grad.clone() for p in reference.parameters()]
        model.train()
        opt.zero_grad(set_to_none=True)
        core.backward_group(model, current, cfg, current_seed)
        core.backward_group(model, replay, cfg, old_seed)
        for p, expected in zip(model.parameters(), expected_grads):
            torch.testing.assert_close(p.grad, expected)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        ref_opt.step()
        core.update(model, opt, current, replay, cfg, current_seed, old_seed)
        for p, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, expected)
        self.assertTrue(all(float(s["step"]) == 1 for s in opt.state.values()))

    def test_real_torch_save_fresh_reload_adam_next_update_and_token_limit(self):
        import torch
        cfg = data.policy()
        model = self.toy_model()
        optimizer = core.make_optimizer(model, cfg)
        group = fixture()
        core.update(model, optimizer, group, None, cfg, 4, 8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            saved = core.save_state(path, model, optimizer, {"stage": 1})
            restored = self.toy_model()
            restored_opt = core.make_optimizer(restored, cfg)
            core.restore_state(path, restored, restored_opt, saved["state_sha256"])
            np.testing.assert_array_equal(core.score(model, [group], cfg), core.score(restored, [group], cfg))
            core.update(model, optimizer, group, None, cfg, 5, 9)
            core.update(restored, restored_opt, group, None, cfg, 5, 9)
            self.assertEqual(core.state_digest(model.state_dict()), core.state_digest(restored.state_dict()))
            self.assertEqual(core.state_digest(optimizer.state_dict()), core.state_digest(restored_opt.state_dict()))
        cfg["model"]["token_budget"] = 2
        with self.assertRaisesRegex(ValueError, "token budget"):
            core.account_vectors(model, group, cfg)


if __name__ == "__main__":
    unittest.main()
