"""Hand-built supervision only; never opens project training/valid labels."""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_chinese_base as base


def truth28() -> np.ndarray:
    owners = list(itertools.chain.from_iterable([[i] * 3 for i in range(4)] + [[i] * 2 for i in range(4, 12)]))
    return np.asarray([int(owners[i] == owners[j]) for i, j in itertools.combinations(range(28), 2)], dtype=np.uint8)


def public_fixture() -> tuple[dict, list]:
    groups, metadata = {"train": [], "development": []}, []
    for split, n in (("train", 60), ("development", 20)):
        for domain in "ABC":
            for i in range(n):
                uid = f"{domain}_{split}_{i:03}"
                sellers = tuple(f"{uid}_s{j:02}" for j in range(28))
                items = tuple(tuple((f"{s}_i{k}", f"title {j}", f"description {k}") for k in range(2))
                              for j, s in enumerate(sellers))
                groups[split].append(base.data.Group(uid, sellers, items))
                metadata.append({"group_uid": uid, "domain": domain, "split": split, "group_index": str(i)})
    return groups, metadata


def fake_run(out: Path) -> tuple[dict, dict, list, dict]:
    """Synthetic receipt fixture, not evidence of actual model execution."""
    c = base.contract()
    groups, metadata = public_fixture()
    selected, part = base.partition(groups, metadata, c)
    rows, seed = base.schedule(selected["fit"], c)
    out.mkdir(parents=True)
    for name, value in (("exit_status.txt", "0\n"), ("started.txt", "2026-09-18T10:00:00+08:00"),
                        ("finished.txt", "2026-09-18T11:00:00+08:00"), ("resource_usage.log", "fixture only")):
        (out.parent / name).write_text(value)
    base.data.write_json(out / "partition.json", part)
    preflights = {}
    for arm in base.ARMS:
        preflights[arm] = {"pretrained": {k: c["models"][arm][k] for k in ("file_count", "total_size_bytes", "content_sha256")},
                          "initial_model_state_sha256": "fixture", "updates": 0, "supervision_reads": 0,
                          "max_tokens": 50, "records_checked": 24103 * (2 if arm.startswith("split_") else 1), "actual_forward_shape": [1, 378]}
    checked = {"fixture": "public input only"}
    r = {"status": base.COMPLETE, "config": c, "source_files": base.sources(), "arms": {},
         "inputs": checked, "preflights": preflights, "torch_contracts": {"passed": base.CPU_CONTRACT_COUNT, "skipped": 0},
         "physical_updates": 3456, "formal_training_seconds": 10.,
         "label_parses": {"train": 1, "development": 0, "heldout": 0, "owners": 0},
         "budget": {"elapsed_seconds": 100., "peak_observed_bytes": 100000},
         "partition": base.data.record(out / "partition.json", out)}
    t = truth28()
    for arm in base.ARMS:
        ar = out / arm
        (ar / "scores").mkdir(parents=True)
        (ar / "models").mkdir()
        a = {"arm": arm, "updates": 864, "label_parses": 0, "points": {},
             "preflight": preflights[arm], "dropout_stream": seed,
             "fit_group_ids": [g.uid for g in selected["fit"]],
             "calibration_group_ids": [g.uid for g in selected["calibration"]],
             "group_schedule_sha256": base.hashlib.sha256(base.data.json_bytes([g.uid for g in rows])).hexdigest(),
             "training": [{"start": start, "stop": start + 432, "updates": 432,
                           "losses_by_epoch": {"bce": [0.5]*3, "rank": [2.]*3, "total": [0.5 + 2*base.contract()["interventions"][arm]["rank_weight"]]*3}, "first_update_modules": {k: {"finite_nonzero_gradient": True, "parameters_changed": True}
                                                    for k in ("encoder", "head")}} for start in (0, 432)]}
        for epoch in (3, 6):
            p = {"arm": arm, "epoch": epoch, "full_model_and_adam_reloaded": True,
                 "model_state_sha256": f"fixture_{arm}_{epoch}", "scores": {}, "train_metrics": {}}
            for role in base.ROLES:
                scores = np.tile(np.where(t == 1, 2., -2.).astype(np.float32), (base.ROLE_SIZES[role], 1))
                path = ar / "scores" / f"epoch{epoch}_{role}.npy"
                np.save(path, scores, allow_pickle=False)
                p["scores"][role] = base.data.record(path, ar)
                if role != "development":
                    path = ar / "scores" / f"epoch{epoch}_{role}_metrics.npy"
                    np.save(path, np.zeros((base.ROLE_SIZES[role], 22), np.float64), allow_pickle=False)
                    p["train_metrics"][role] = {"file": base.data.record(path, ar)}
            path = ar / "models" / f"epoch{epoch}.pt"
            path.write_bytes(f"untrained fixture {arm} {epoch}".encode())
            p["model"] = {**base.data.record(path, ar), "actual_reload_verified": True}
            a["points"][str(epoch)] = p
        scores = np.tile(np.where(t == 1, 2., -2.).astype(np.float32), (36, 1))
        cal = base.calibrate(np.tile(t, (36, 1)), scores, [x["domain"] for x in part["calibration"]])
        cal.update(epoch=6, model_state_sha256=a["points"]["6"]["model_state_sha256"],
                   score_sha256=a["points"]["6"]["scores"]["calibration"]["sha256"])
        base.data.write_json(ar / "calibration.json", cal)
        a["calibration"] = base.data.record(ar / "calibration.json", ar)
        base.data.write_json(ar / "manifest.json", a)
        r["arms"][arm] = {"updates": 864, "manifest": base.data.record(ar / "manifest.json", out)}
    base.data.write_json(out / "manifest.json", r)
    base.verify_models(out)
    base.data.write_json(out / "completion.json", {"status": base.COMPLETE, "budget": r["budget"],
                                                   "manifest_sha256": base.data.sha256(out / "manifest.json")})
    return c, groups, metadata, checked


class ScientificContracts(unittest.TestCase):
    def test_partition_and_schedule_keep_calibration_out(self):
        groups, metadata = public_fixture()
        selected, part = base.partition(groups, metadata, base.contract())
        schedule, _ = base.schedule(selected["fit"], base.contract())
        self.assertEqual(len(schedule), 864)
        self.assertEqual(len(schedule) * len(base.ARMS), 3456)
        self.assertTrue(set(g.uid for g in schedule).isdisjoint(g.uid for g in selected["calibration"]))
        for epoch in range(6):
            self.assertEqual({g.uid for g in schedule[144*epoch:144*(epoch+1)]},
                             {g.uid for g in selected["fit"]})
        self.assertEqual([sum(r["domain"] == d for r in part["calibration"]) for d in "ABC"], [12]*3)

    def test_inputs_only_text_and_exact_channel_order(self):
        group = base.data.Group("secret_group", ("secret_a", "secret_b"),
            ((("id_1", "标题甲", "描述甲"), ("id_2", "标题乙", "描述乙")),
             (("id_3", "标题丙", "描述丙"),)))
        self.assertEqual(base.record_texts(group, "combined_mean"),
                         ["标题甲\n描述甲", "标题乙\n描述乙", "标题丙\n描述丙"])
        self.assertEqual(base.record_texts(group, "separate_moments"),
                         ["标题甲", "标题乙", "标题丙", "描述甲", "描述乙", "描述丙"])
        with self.assertRaises(ValueError):
            base.record_texts(group, "unknown")

    def test_calibration_strict_ties_and_domain_gates(self):
        truth = np.array([[1, 0, 0]]*3)
        scores = np.array([[5, 1, 1], [5, 3, 2], [5, -1, -2]], dtype=np.float32)
        result = base.calibrate(truth, scores, list("ABC"))
        self.assertEqual(result["threshold"], np.nextafter(np.float64(3), np.inf))
        self.assertEqual(int(base.binary_counts(truth, scores, result["threshold"])[:, 1].sum()), 0)
        counts = np.tile([10, 0, 10, 358], (60, 1))
        draws = np.tile(np.arange(20), (5, 3, 1))
        self.assertTrue(base.automatic_report(counts, list("A"*20 + "B"*20 + "C"*20), draws)["all_domains_pass"])
        counts[0] = [10, 8, 10, 350]
        self.assertFalse(base.automatic_report(counts, list("A"*20 + "B"*20 + "C"*20), draws)["all_domains_pass"])
        # Unequal group counts expose the difference between pooled and macro rates.
        saved = [{"tp": 1, "fp": 0, "fn": 1, "tn": 8},
                 {"tp": 9, "fp": 9, "fn": 1, "tn": 1}] * 3
        pooled = base.fixed_classification(saved, list("AABBCC"))
        self.assertEqual([pooled["pooled"][key] for key in ("tp", "fp", "fn", "tn")], [30, 27, 6, 27])
        for point in [pooled["pooled"], *pooled["by_domain"].values()]:
            self.assertAlmostEqual(point["precision"], 10/19)
            self.assertAlmostEqual(point["recall"], 10/12)
            self.assertAlmostEqual(point["f1"], 20/31)
            self.assertAlmostEqual(point["fpr"], 0.5)
        self.assertNotAlmostEqual(pooled["pooled"]["precision"], (1 + 0.5)/2)

    def test_complete_four_arm_gate_and_changed_model(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "job" / "run"
            c, groups, metadata, checked = fake_run(out)
            scores, _, _ = base.validated_run(out, c, groups, metadata, checked)
            self.assertEqual(set(scores), set(base.ARMS))
            path = out / "split_rank/models/epoch6.pt"
            payload = path.read_bytes()
            path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
            with mock.patch.object(base.platform, "system", return_value="Linux"), \
                 mock.patch.object(base.public, "public_inputs", return_value=(groups, metadata, checked)), \
                 mock.patch.object(base.public, "attach_labels") as parse:
                with self.assertRaises(ValueError):
                    base.evaluate(out, Path(directory) / "rejected_evaluation")
            parse.assert_not_called()

    def test_evaluation_saves_all_matrices_with_one_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "job" / "run"
            c, groups, metadata, checked = fake_run(out)
            labelled = [base.data.Group(g.uid, g.sellers, g.items, tuple(truth28())) for g in groups["development"]]
            reference = {"scores": np.tile(np.where(truth28(), 2., -2.).astype(np.float32), (60, 1)),
                         "threshold": 0., "files": {"fixture": "no historical files read"}}
            with mock.patch.object(base.platform, "system", return_value="Linux"), \
                 mock.patch.object(base.public, "public_inputs", return_value=(groups, metadata, checked)), \
                 mock.patch.object(base.public, "attach_labels", return_value=labelled) as parse, \
                 mock.patch.object(base, "historical_reference", return_value=reference):
                result = base.evaluate(out, Path(directory) / "evaluation")
            self.assertEqual(parse.call_count, 1)
            self.assertEqual(set(result["arms"]), set(base.ARMS))
            self.assertEqual(len(result["comparisons"]), 7)
            self.assertEqual(len(list((Path(directory)/"evaluation").rglob("*.npy"))), 9)
            for arm in base.ARMS:
                self.assertTrue(result["arms"][arm]["automatic_classification"]["all_domains_pass"])
                for epoch in ("3", "6"):
                    self.assertEqual(result["arms"][arm]["points"][epoch]["fixed_classification"]["pooled"]["tp"], 1200)
            # Independent fault injection: the FIRST uncertainty call fails, but
            # every matrix and confusion count must already be recoverable.
            failed = Path(directory) / "uncertainty_failure"
            with mock.patch.object(base.platform, "system", return_value="Linux"), \
                 mock.patch.object(base.public, "public_inputs", return_value=(groups, metadata, checked)), \
                 mock.patch.object(base.public, "attach_labels", return_value=labelled) as parse, \
                 mock.patch.object(base, "historical_reference", return_value=reference), \
                 mock.patch.object(base, "automatic_report", side_effect=RuntimeError("injected bootstrap failure")):
                with self.assertRaisesRegex(RuntimeError, "injected bootstrap"):
                    base.evaluate(out, failed)
            self.assertEqual(parse.call_count, 1)
            self.assertTrue((failed / "failure.json").is_file())
            self.assertFalse((failed / "evaluation.json").exists())
            saved = base.data.read_json(failed / "collected.json")
            self.assertEqual(saved["status"], "CHINESE_BASE_METRICS_COLLECTED_BEFORE_UNCERTAINTY")
            self.assertEqual(saved["domains"], list("A"*20 + "B"*20 + "C"*20))
            self.assertEqual(len(saved["group_ids"]), 60)
            self.assertEqual(len(list(failed.rglob("*.npy"))), 9)
            for arm in base.ARMS:
                np.testing.assert_array_equal(saved["arms"][arm]["automatic_classification"]["counts_by_group"],
                                              result["arms"][arm]["automatic_classification"]["counts_by_group"])
                for epoch in ("3", "6"):
                    self.assertEqual(saved["arms"][arm]["points"][epoch]["counts_at_logit_zero"],
                                     result["arms"][arm]["points"][epoch]["counts_at_logit_zero"])
                    self.assertEqual(saved["arms"][arm]["points"][epoch]["fixed_classification"],
                                     result["arms"][arm]["points"][epoch]["fixed_classification"])
            self.assertEqual(len(saved["historical_labse"]["counts_by_group"]), 60)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires existing Linux Torch")
class TorchContracts(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        torch.manual_seed(29)

    def test_uniform_logits_have_log27_rank_and_log2_bce(self):
        torch = self.torch
        z = torch.zeros(378, dtype=torch.float64, requires_grad=True)
        terms = base.objectives(z, torch.tensor(truth28(), dtype=torch.float64), 1)
        self.assertAlmostEqual(float(terms["rank"].detach()), float(np.log(27)), places=13)
        self.assertAlmostEqual(float(terms["bce"].detach()), float(np.log(2)), places=13)
        terms["total"].backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_rank_matches_independent_scalar_and_analytic_edge_gradient(self):
        torch = self.torch
        z = torch.linspace(-1.3, 2.7, 378, dtype=torch.float64, requires_grad=True)
        y = truth28()
        terms = base.objectives(z, torch.tensor(y, dtype=torch.float64), 1)
        logits = z.detach().numpy()
        pairs = list(itertools.combinations(range(28), 2))
        gradient, losses = np.zeros(378), []
        for account in range(28):
            indices = [k for k, pair in enumerate(pairs) if account in pair]
            values = logits[indices]
            positive = y[indices].astype(bool)
            maximum = float(max(values))
            exp = np.exp(values - maximum)
            losses.append(maximum + np.log(exp.sum()) - values[positive].mean())
            gradient[indices] += (exp / exp.sum() - positive / positive.sum()) / 28
        actual = torch.autograd.grad(terms["rank"], z)[0].numpy()
        self.assertAlmostEqual(float(terms["rank"].detach()), float(np.mean(losses)), places=13)
        np.testing.assert_allclose(actual, gradient, rtol=1e-12, atol=1e-14)

    def test_weight_changes_actual_gradient_and_bad_supervision_rejected(self):
        torch = self.torch
        z = torch.linspace(-1., 1., 378, dtype=torch.float64, requires_grad=True)
        y = torch.tensor(truth28(), dtype=torch.float64)
        terms = base.objectives(z, y, 1)
        gradients = {name: torch.autograd.grad(value, z, retain_graph=True)[0] for name, value in terms.items()}
        torch.testing.assert_close(gradients["total"], gradients["bce"] + gradients["rank"])
        self.assertGreater(float(gradients["rank"].norm()), 0)
        zero = base.objectives(z, y, 0)
        torch.testing.assert_close(torch.autograd.grad(zero["total"], z)[0], gradients["bce"])
        for weight in (-1, .5, 2):
            with self.assertRaises(ValueError):
                base.objectives(z, y, weight)
        with self.assertRaises(ValueError):
            base.objectives(z, torch.zeros_like(y), 1)

    def test_population_moments_hand_case_permutation_and_duplication(self):
        torch = self.torch
        vectors = torch.tensor([[1., 0.], [0., 1.], [3., 4.], [4., 3.]], requires_grad=True)
        actual = base.pool_channels(vectors, [2], "separate_moments")
        raw = vectors.detach().numpy().astype(np.float64)
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        expected = np.concatenate([v for channel in (raw[:2], raw[2:]) for v in
                                   (channel.mean(0), np.sqrt(((channel-channel.mean(0))**2).mean(0)+1e-8))])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(actual.detach().numpy()[0], expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(base.pool_channels(vectors[[1, 0, 3, 2]], [2], "separate_moments"), actual)
        torch.testing.assert_close(base.pool_channels(vectors[[0, 1, 0, 1, 2, 3, 2, 3]], [4], "separate_moments"), actual)
        actual[0, 0].backward()
        self.assertTrue(torch.isfinite(vectors.grad).all())

    def test_account_permutation_and_pair_symmetry(self):
        torch = self.torch
        model = base.core.build_model(torch.nn.Identity(), 7, 8)
        accounts = torch.randn(28, 7)
        logits = model.pair_logits(accounts)
        left, right = torch.triu_indices(28, 28, 1)
        matrix = torch.zeros(28, 28)
        matrix[left, right] = logits
        matrix[right, left] = logits
        permutation = torch.randperm(28)
        torch.testing.assert_close(model.pair_logits(accounts[permutation]), matrix[permutation[left], permutation[right]])

    def test_all_four_updates_and_real_model_adam_restore_on_tiny_encoder(self):
        torch = self.torch

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(64, 8)
                self.projection = torch.nn.Linear(8, 8)

            def tokenizer(self, texts, **kwargs):
                tokens = [[ord(x) % 63 + 1 for x in text] for text in texts]
                width = max(map(len, tokens))
                return {"input_ids": torch.tensor([t + [0]*(width-len(t)) for t in tokens]),
                        "attention_mask": torch.tensor([[1]*len(t) + [0]*(width-len(t)) for t in tokens])}

            def forward(self, features):
                mask = features["attention_mask"].unsqueeze(-1)
                vector = (self.embedding(features["input_ids"]) * mask).sum(1) / mask.sum(1)
                return {"sentence_embedding": self.projection(vector)}

        groups, _ = public_fixture()
        source = groups["train"][0]
        group = base.data.Group(source.uid, source.sellers, source.items, tuple(truth28()))
        c = base.contract()
        for arm in base.ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(31)
                dimension = 32 if arm.startswith("split_") else 8
                model = base.core.build_model(Encoder(), dimension, 12)
                optimizer = base.core.make_optimizer(model, c)
                before = base.core.state_digest(model.encoder.state_dict())
                result = base.update(model, optimizer, group, c, arm, 71, observe=True)
                self.assertNotEqual(before, base.core.state_digest(model.encoder.state_dict()))
                self.assertAlmostEqual(result["total"], result["bce"] +
                                       c["interventions"][arm]["rank_weight"] * result["rank"], places=5)
                target = Path(directory) / "checkpoint.pt"
                saved = base.core.save_state(target, model, optimizer, {"arm": arm})
                restored = base.core.build_model(Encoder(), dimension, 12)
                other = base.core.make_optimizer(restored, c)
                self.assertEqual(base.core.restore_state(target, restored, other, saved["state_sha256"]), {"arm": arm})
                base.update(model, optimizer, group, c, arm, 72)
                base.update(restored, other, group, c, arm, 72)
                self.assertEqual(base.core.state_digest(model.state_dict()), base.core.state_digest(restored.state_dict()))
                self.assertEqual(base.core.state_digest(optimizer.state_dict()), base.core.state_digest(other.state_dict()))

    def test_native_pooling_api_modes_do_not_override_model(self):
        from types import SimpleNamespace
        self.assertEqual(base.pooling_modes(SimpleNamespace(pooling_mode="cls")), ["cls"])
        self.assertEqual(base.pooling_modes(SimpleNamespace(pooling_mode_cls_token=True)), ["cls"])
        self.assertNotEqual(base.pooling_modes(SimpleNamespace(pooling_mode="mean")), ["cls"])


if __name__ == "__main__":
    unittest.main()
