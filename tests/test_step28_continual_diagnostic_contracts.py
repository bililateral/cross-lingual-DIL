"""Diagnostic-specific semantic checks; no project supervision is parsed."""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_continual_diagnostic as diag
import step28_continual_diagnostic_evaluate as evaluation
import step28_continual_diagnostic_run as runner
import step28_continual_population as core
import step28_continual_population_data as data
import step28_continual_population_evaluate as metrics


def fixture(uid: str = "group") -> data.Group:
    owners = [i // 2 for i in range(16)] + [8 + i // 3 for i in range(12)]
    labels = tuple(int(owners[i] == owners[j]) for i, j in itertools.combinations(range(28), 2))
    sellers = tuple(f"seller{i:02}" for i in range(28))
    items = tuple(tuple((f"{uid}_{i}_{j}", f"title {i}", f"description {j} {i}")
                        for j in range(2)) for i in range(28))
    return data.Group(uid, sellers, items, labels)


class Contracts(unittest.TestCase):
    def test_public_archive_never_opens_truth_or_test_text(self):
        original = Path.open
        opened = []
        config = diag.policy()
        root = (data.ROOT / config["data_root"]).resolve()

        def guarded(path, *args, **kwargs):
            rel = path.resolve().relative_to(root).as_posix()
            self.assertNotIn("supervision", rel)
            self.assertFalse(rel.startswith("heldout/"))
            opened.append(rel)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            archive = diag.Archive(config, train_labels=False)
        self.assertEqual([len(archive.groups(s)) for s in diag.SPLITS], [180, 60])
        self.assertEqual(archive.train_label_parses, 0)
        self.assertEqual(set(opened), {"manifest.json", "validation.json", "groups.csv",
                                       "train/items.jsonl", "development/items.jsonl"})
        for path in ("heldout/items.jsonl", "development/supervision/pairs.csv", "train/supervision/owners.csv"):
            with self.assertRaises(ValueError):
                archive.path(path)

    def test_same_positions_and_dropout_stream_with_different_domains(self):
        config = diag.policy()
        first = diag.positions(60, 3, diag.segment_seed(config, "A", "branch"))
        second = diag.positions(60, 3, diag.segment_seed(config, "A", "branch"))
        self.assertEqual(first, second)
        for offset in (0, 60, 120):
            self.assertEqual(sorted(first[offset:offset + 60]), list(range(60)))
        self.assertNotEqual(first, diag.positions(60, 3, diag.segment_seed(config, "A", "shared")))

    def comparison_fixture(self):
        config = diag.policy()
        rows = {"A": np.array([0, 1]), "B": np.array([2, 3]), "C": np.array([4, 5])}
        values = {a: {p: np.zeros((6, len(metrics.COLUMNS))) for p in diag.POINTS} for a in "ABC"}
        return config, rows, values

    def test_correct_domain_and_no_factor_of_three_dilution(self):
        cfg, rows, values = self.comparison_fixture()
        for i, (a, b) in enumerate(cfg["rotations"]):
            values[a]["same"][rows[a], :] = i + 1
            values[a]["cross"][rows[a], :] = -(i + 1)
            values[a]["cross"][rows[b], :] = 10 * (i + 1)
        result = diag.comparisons(values, rows, cfg)["average_precision"]
        self.assertEqual(result["same_old_gain"]["mean"], 2.)
        self.assertEqual(result["cross_new_gain"]["mean"], 20.)
        self.assertEqual(result["cross_old_change"]["mean"], -2.)
        self.assertEqual(result["cross_minus_same_old"]["mean"], -4.)
        self.assertEqual(result["cross_new_gain"]["conditional_95pct_interval"], [20., 20.])

    def test_bootstrap_reuses_same_domain_draws_across_contrasts(self):
        cfg, rows, values = self.comparison_fixture()
        for a, b in cfg["rotations"]:
            values[a]["cross"][rows[b][0], :] = 1.
            values[a]["cross"][rows[b][1], :] = 3.
            values[a]["same"][rows[a][0], :] = 2.
            values[a]["same"][rows[a][1], :] = 6.
        result = diag.comparisons(values, rows, cfg)["average_precision"]
        np.testing.assert_allclose(result["same_old_gain"]["conditional_95pct_interval"],
                                   2 * np.array(result["cross_new_gain"]["conditional_95pct_interval"]))
        # Independent bootstrap reference for the selected-domain equal mean.
        rng = np.random.default_rng(cfg["evaluation"]["bootstrap_seed"])
        ref = np.mean([np.array([1., 3.])[rng.integers(0, 2, size=(5000, 2))].mean(axis=1)
                       for _ in "ABC"], axis=0)
        np.testing.assert_allclose(result["cross_new_gain"]["conditional_95pct_interval"],
                                   np.quantile(ref, [.025, .975]))

    def test_worse_than_same_is_not_forgetting(self):
        cfg, rows, values = self.comparison_fixture()
        for a, b in cfg["rotations"]:
            values[a]["same"][rows[a], :] = .2
            values[a]["cross"][rows[a], :] = .1
            values[a]["cross"][rows[b], :] = .3
        result = diag.comparisons(values, rows, cfg)
        self.assertLess(result["average_precision"]["cross_minus_same_old"]["mean"], 0)
        self.assertFalse(diag.decision(result)["ap_conflict_supported"])
        self.assertTrue(diag.decision(result)["remaining_learning_room_supported"])

    def test_two_same_directions_and_map_agreement_required(self):
        cfg, rows, values = self.comparison_fixture()
        for a, b in cfg["rotations"]:
            values[a]["cross"][rows[a], :] = -.1
            values[a]["cross"][rows[b], :] = .2
        result = diag.comparisons(values, rows, cfg)
        self.assertTrue(diag.decision(result)["ap_conflict_supported"])
        self.assertTrue(diag.decision(result)["action"].startswith("CONFLICT"))
        result["map"]["cross_old_change"]["per_direction"] = [.1, .1, .1]
        self.assertTrue(diag.decision(result)["action"].startswith("STOP"))
        # Positive/negative aggregate intervals alone do not replace matching directions.
        result["average_precision"]["cross_new_gain"]["per_direction"] = [.9, -.01, -.01]
        result["average_precision"]["cross_old_change"]["per_direction"] = [.01, -.9, -.9]
        self.assertFalse(diag.decision(result)["ap_conflict_supported"])

    def test_failed_or_missing_run_never_parses_valid(self):
        with tempfile.TemporaryDirectory(prefix="seller_alias_diagnostic_") as tmp:
            path = Path(tmp)
            (path / "failure.json").write_text("{}", encoding="utf-8")
            with patch.object(diag, "Archive", return_value=SimpleNamespace()), patch.object(diag, "read_labels") as read:
                with self.assertRaises(ValueError):
                    evaluation.evaluate(path, path / "evaluation")
                read.assert_not_called()
                self.assertFalse((path / "evaluation").exists())

    def gate_fixture(self, root):
        cfg = diag.policy()
        grouped = {s: [fixture(f"{d}_{i}_{s}") for d in "ABC"
                       for i in range(cfg["groups_per_domain"][s])] for s in diag.SPLITS}
        archive = SimpleNamespace(checked={}, manifest={"files": {"train/supervision/pairs.csv": {}}},
            metadata=[{"group_uid": g.uid, "domain": g.uid[0]} for gs in grouped.values() for g in gs],
            groups=lambda split, domain="ABC": [g for g in grouped[split] if g.uid[0] in domain])
        manifest = {"status": diag.COMPLETE, "config": cfg, "source_files": [], "physical_updates": 1620,
            "label_reads": {"train": 1, "development": 0, "heldout": 0, "owners": 0, "old_assets": 0},
            "torch_contracts": {"passed": 2, "skipped": 0, "failed": 0},
            "intermediate_states_removed_after_reload": True, "pretrained_model": cfg["model"],
            "budget": {"elapsed_seconds": 100, "peak_observed_bytes": 10000},
            "verified_input_files": {"train/supervision/pairs.csv": {}},
            "group_ids": {s: [g.uid for g in gs] for s, gs in grouped.items()}, "rotations": []}
        (root / "scores").mkdir()
        (root / "metrics").mkdir()
        for a, b in cfg["rotations"]:
            r = {"anchor": a, "target": b, "initial_model_sha256": "original", "training": {}, "points": {},
                 "branch_starts": {arm: {"model": a + "shared", "optimizer": a + "adam"} for arm in diag.ARMS}}
            for arm in diag.POINTS:
                phase = "shared" if arm == "shared" else "branch"
                seed = diag.segment_seed(cfg, a, phase)
                positions = diag.positions(60, 3, seed)
                current = archive.groups("train", b if arm == "cross" else a)
                r["training"][arm] = {"updates": 180, "position_order": positions,
                    "group_ids": [current[i].uid for i in positions], "dropout_stream": seed,
                    "adam_step": 180 if arm == "shared" else 360, "online_bce_per_epoch": [.3, .2, .1],
                    "training_seconds": 10, "module_state_before": {"encoder": "e0", "head": "h0"},
                    "module_state_after": {"encoder": "e1", "head": "h1"},
                    "first_update_clipped_gradient_norms": {"encoder": .1, "head": .2}}
                point = {"full_model_and_adam_reloaded": True, "scores": {}, "model_state_sha256": a + arm}
                for split in diag.SPLITS:
                    path = root / "scores" / f"{a}_{arm}_{split}.npy"
                    np.save(path, np.zeros((len(grouped[split]), 378), dtype=np.float32), allow_pickle=False)
                    point["scores"][split] = data.record(path, root)
                labels = np.array([g.labels for g in grouped["train"]], dtype=np.uint8)
                matrix, counts = metrics.group_metrics(labels, np.zeros(labels.shape))
                path = root / "metrics" / f"{a}_{arm}_train.npy"
                np.save(path, matrix, allow_pickle=False)
                point["train_metrics"] = {**diag.summarize(matrix, diag.domain_rows(archive, "train")),
                    "columns": list(metrics.COLUMNS), "per_group_confusion": counts, "per_group_metrics": data.record(path, root)}
                r["points"][arm] = point
            manifest["rotations"].append(r)
        return cfg, archive, manifest

    def test_complete_gate_and_missing_points_before_truth(self):
        with tempfile.TemporaryDirectory(prefix="seller_alias_diagnostic_") as tmp:
            root = Path(tmp)
            cfg, archive, manifest = self.gate_fixture(root)
            data.write_json(root / "manifest.json", manifest)
            with patch.object(diag, "sources", return_value=[]):
                _, scores = evaluation.validate(root, cfg, archive)
                self.assertEqual(set(scores), set("ABC"))
                for arm in diag.POINTS:
                    changed = copy.deepcopy(manifest)
                    del changed["rotations"][0]["points"][arm]
                    data.write_json(root / "manifest.json", changed)
                    with patch.object(diag, "Archive", return_value=archive), patch.object(diag, "read_labels") as read:
                        with self.assertRaises(ValueError):
                            evaluation.evaluate(root, root / "eval")
                        read.assert_not_called()

    def test_wrong_domain_adam_mapping_and_score_corruption_rejected(self):
        with tempfile.TemporaryDirectory(prefix="seller_alias_diagnostic_") as tmp:
            root = Path(tmp)
            cfg, archive, manifest = self.gate_fixture(root)
            variants = []
            changed = copy.deepcopy(manifest)
            changed["rotations"][0]["training"]["cross"]["group_ids"] = changed["rotations"][0]["training"]["same"]["group_ids"]
            variants.append(changed)
            changed = copy.deepcopy(manifest)
            changed["rotations"][0]["branch_starts"]["cross"]["optimizer"] = "wrong_adam"
            variants.append(changed)
            changed = copy.deepcopy(manifest)
            changed["group_ids"]["development"].reverse()
            variants.append(changed)
            with patch.object(diag, "sources", return_value=[]):
                for changed in variants:
                    data.write_json(root / "manifest.json", changed)
                    with self.assertRaises(ValueError):
                        evaluation.validate(root, cfg, archive)
                data.write_json(root / "manifest.json", manifest)
                path = root / "scores/A_cross_development.npy"
                payload = bytearray(path.read_bytes())
                payload[-1] ^= 1
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    evaluation.validate(root, cfg, archive)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Actual Torch contracts require original Linux py310")
class TorchContracts(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(111)

    def model(self):
        import torch

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(127, 4)
                self.dropout = torch.nn.Dropout(.25)

            def tokenizer(self, texts, **kwargs):
                tokens = [[ord(c) % 127 for c in text] for text in texts]
                size = max(map(len, tokens))
                return {"input_ids": torch.tensor([t + [0] * (size - len(t)) for t in tokens]),
                        "attention_mask": torch.tensor([[1] * len(t) + [0] * (size - len(t)) for t in tokens])}

            def forward(self, features):
                mask = features["attention_mask"].unsqueeze(-1)
                v = self.dropout(self.embedding(features["input_ids"]))
                return {"sentence_embedding": (v * mask).sum(1) / mask.sum(1)}

        return core.build_model(Encoder(), 4, 5)

    def test_actual_segment_branch_reload_preserves_nonempty_adam(self):
        cfg = diag.policy()
        model = self.model()
        optimizer = core.make_optimizer(model, cfg)
        groups = [fixture("one"), fixture("two")]
        first = runner.train_segment(model, optimizer, groups, cfg, "A", "shared", lambda: None)
        self.assertEqual(first["adam_step"], 6)
        with tempfile.TemporaryDirectory(prefix="seller_alias_diagnostic_") as tmp:
            checkpoint = Path(tmp) / "shared.pt"
            saved = core.save_state(checkpoint, model, optimizer, {"test": True})
            ends = []
            for _ in range(2):
                opt = core.make_optimizer(model, cfg)
                core.restore_state(checkpoint, model, opt, saved["state_sha256"])
                log = runner.train_segment(model, opt, groups, cfg, "A", "branch", lambda: None)
                self.assertEqual(log["adam_step"], 12)
                ends.append((core.state_digest(model.state_dict()), core.state_digest(opt.state_dict())))
            self.assertEqual(ends[0], ends[1])

    def test_extra_scoring_cannot_change_next_dropout_update(self):
        cfg = diag.policy()
        left = self.model()
        right = copy.deepcopy(left)
        lopt, ropt = core.make_optimizer(left, cfg), core.make_optimizer(right, cfg)
        group = fixture()
        for m, o in ((left, lopt), (right, ropt)):
            core.update(m, o, group, None, cfg, 23, 0)
        before = core.state_digest(right.state_dict())
        core.score(right, [group], cfg)
        core.score(right, [group, group], cfg)
        self.assertEqual(before, core.state_digest(right.state_dict()))
        for m, o in ((left, lopt), (right, ropt)):
            core.update(m, o, group, None, cfg, 37, 0)
        self.assertEqual(core.state_digest(left.state_dict()), core.state_digest(right.state_dict()))
        self.assertEqual(core.state_digest(lopt.state_dict()), core.state_digest(ropt.state_dict()))


if __name__ == "__main__":
    unittest.main()
