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
import step28_base_model as base


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
                          "max_tokens": 50, "records_checked": 24103, "actual_forward_shape": [1, 378]}
    checked = {"fixture": "public input only"}
    r = {"status": base.COMPLETE, "config": c, "source_files": base.sources(), "arms": {},
         "inputs": checked, "preflights": preflights, "torch_contracts": {"passed": 2, "skipped": 0},
         "physical_updates": 2592, "formal_training_seconds": 10.,
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
                           "first_update_modules": {k: {"finite_nonzero_gradient": True, "parameters_changed": True}
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


class NumericalContracts(unittest.TestCase):
    def test_native_pooling_current_and_legacy_api(self):
        from types import SimpleNamespace

        for mode in ("cls", "mean"):
            self.assertEqual(base.pooling_modes(SimpleNamespace(pooling_mode=mode)), [mode])
            self.assertEqual(base.pooling_modes(SimpleNamespace(pooling_mode=(mode,))), [mode])
        self.assertEqual(base.pooling_modes(SimpleNamespace(
            pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)), ["cls"])
        self.assertEqual(base.pooling_modes(SimpleNamespace(
            pooling_mode_cls_token=False, pooling_mode_mean_tokens=True)), ["mean"])
        self.assertNotEqual(base.pooling_modes(SimpleNamespace(pooling_mode=("cls", "mean"))), ["cls"])
        self.assertNotEqual(base.pooling_modes(SimpleNamespace(pooling_mode="max")), ["mean"])
        self.assertEqual(base.pooling_modes(SimpleNamespace()), [])
        with self.assertRaises(ValueError):
            base.pooling_modes(SimpleNamespace(pooling_mode=42))

    def test_pooled_intervals_use_pooled_counts_with_same_stratified_draws(self):
        counts = np.array([[5 + i % 13, i % 5, 15 - i % 13, 358 - i % 5] for i in range(60)])
        draws = np.random.default_rng(20260918).integers(0, 20, size=(5000, 3, 20))
        unchanged = draws.copy()
        result = base.automatic_report(counts, list("A"*20 + "B"*20 + "C"*20), draws)
        reference = []
        for sample in draws:
            total = np.zeros(4, dtype=np.int64)
            for d in range(3):
                frequencies = np.bincount(sample[d], minlength=20)
                total += frequencies @ counts[20*d:20*(d+1)]
            tp, fp, fn, tn = total
            reference.append([fp/(fp+tn), tp/(tp+fn), tp/(tp+fp)])
        for i, name in enumerate(("fpr", "recall", "precision")):
            self.assertAlmostEqual(result["pooled"][name], base.rates(counts)[name])
            np.testing.assert_allclose(result["pooled"]["conditional_95pct_intervals"][name],
                                       np.quantile(np.asarray(reference)[:, i], [.025, .975]), rtol=0, atol=1e-15)
        np.testing.assert_array_equal(draws, unchanged)

    def test_paired_automatic_differences_have_hand_computable_signs(self):
        reference = np.tile([10, 0, 10, 358], (60, 1))
        candidate = np.tile([12, 1, 8, 357], (60, 1))
        draws = np.tile(np.arange(20), (7, 3, 1))
        result = base.automatic_comparison(candidate, reference, list("A"*20 + "B"*20 + "C"*20), draws)
        for scope in (*"ABC", "pooled"):
            for name, expected in (("fpr", 1/358), ("recall", .1), ("precision", 12/13 - 1)):
                self.assertAlmostEqual(result[name][scope]["difference"], expected)
                np.testing.assert_allclose(result[name][scope]["conditional_95pct_interval"], [expected, expected])

    def test_partition_is_disjoint_group_based_and_label_blind(self):
        c = base.contract()
        groups, meta = public_fixture()
        result, record = base.partition(groups, meta, c)
        self.assertEqual({k: len(v) for k, v in result.items()}, base.ROLE_SIZES)
        self.assertEqual(len(set(x["group_uid"] for role in record.values() for x in role)), 240)
        changed = {**groups, "train": [base.data.Group(g.uid, g.sellers, g.items, tuple(truth28())) for g in groups["train"]]}
        self.assertEqual(base.partition(changed, list(reversed(meta)), c)[1], record)
        reversed_groups = {**groups, "train": list(reversed(groups["train"]))}
        other, _ = base.partition(reversed_groups, meta, c)
        self.assertEqual({g.uid for g in result["fit"]}, {g.uid for g in other["fit"]})

    def test_shared_six_epoch_schedule_never_uses_calibration(self):
        c = base.contract()
        groups, meta = public_fixture()
        selected, _ = base.partition(groups, meta, c)
        rows, _ = base.schedule(selected["fit"], c)
        self.assertEqual(len(rows), 864)
        self.assertTrue(set(g.uid for g in rows).isdisjoint(g.uid for g in selected["calibration"]))
        for epoch in range(6):
            self.assertEqual(set(g.uid for g in rows[epoch*144:(epoch+1)*144]), set(g.uid for g in selected["fit"]))
        self.assertEqual(3 * len(rows), 2592)

    def test_query_prefix_is_symmetric_and_does_not_contain_identifiers(self):
        g = base.data.Group("private_group", ("private_a", "private_b"),
                            ((("i1", "商品一", "描述一"),), (("i2", "商品二", "描述二"),)))
        self.assertEqual(base.record_texts(g, "query: "), ["query: 商品一\n描述一", "query: 商品二\n描述二"])
        self.assertNotIn("private", " ".join(base.record_texts(g, "")))

    def test_calibration_hand_case_strict_ties_and_global_threshold(self):
        y = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        s = np.array([[5., 1., 1.], [5., 3., 2.], [5., -1., -2.]], dtype=np.float32)
        r = base.calibrate(y, s, list("ABC"))
        self.assertEqual(r["threshold"], np.nextafter(np.float64(3.), np.inf))
        self.assertEqual(r["counts_by_domain"], {d: [1, 0, 0, 2] for d in "ABC"})
        # Keeping float32 in classification would round the threshold down to 3.
        count = base.binary_counts(y, s, r["threshold"])
        self.assertEqual(int(count[:, 1].sum()), 0)

    def test_calibration_exact_integer_false_positive_allowance(self):
        y = np.tile(np.r_[1, np.zeros(1000, dtype=int)], (3, 1))
        s = np.tile(np.r_[2000., np.arange(1000.)], (3, 1))
        r = base.calibrate(y, s, list("ABC"))
        self.assertEqual(r["threshold"], np.nextafter(np.float64(998.), np.inf))
        self.assertEqual([r["counts_by_domain"][d][1] for d in "ABC"], [1, 1, 1])
        s[:, -2:] = 999.  # A tie cannot be split to manufacture exactly one FP.
        r = base.calibrate(y, s, list("ABC"))
        self.assertEqual([r["counts_by_domain"][d][1] for d in "ABC"], [0, 0, 0])

    def test_calibration_does_not_optimize_against_positive_scores(self):
        y = np.tile([1, 0, 0], (3, 1))
        s = np.array([[2., 0., 1.]] * 3)
        first = base.calibrate(y, s, list("ABC"))["threshold"]
        s[:, 0] = -1000.
        self.assertEqual(base.calibrate(y, s, list("ABC"))["threshold"], first)

    def test_calibration_rejects_nan_missing_domain_and_nonbinary_truth(self):
        y = np.tile([1, 0], (3, 1))
        s = np.tile([2., 0.], (3, 1))
        with self.assertRaises(ValueError):
            base.calibrate(y, s, ["A", "A", "B"])
        s[0, 0] = np.nan
        with self.assertRaises(ValueError):
            base.calibrate(y, s, list("ABC"))
        with self.assertRaises(ValueError):
            base.binary_counts(np.array([[2, 0]]), np.array([[1., 0.]]), 0.)

    def test_each_domain_must_pass_no_pooled_cancellation(self):
        # 20 groups per domain, each 20 positives and 358 negatives.
        counts = np.tile([10, 0, 10, 358], (60, 1))
        counts[40:, 0], counts[40:, 2] = 9, 11
        draws = np.tile(np.arange(20), (5, 3, 1))
        result = base.automatic_report(counts, list("A"*20 + "B"*20 + "C"*20), draws)
        self.assertFalse(result["all_domains_pass"])
        self.assertFalse(result["by_domain"]["C"]["passes_point_gates"])
        counts[40:, 0], counts[40:, 2] = 10, 10
        counts[0, 1], counts[0, 3] = 8, 350  # 8/7160 exceeds 0.1%.
        self.assertFalse(base.automatic_report(counts, list("A"*20 + "B"*20 + "C"*20), draws)["all_domains_pass"])
        counts[0, 1], counts[0, 3] = 7, 351
        self.assertTrue(base.automatic_report(counts, list("A"*20 + "B"*20 + "C"*20), draws)["all_domains_pass"])

    def test_policy_tampering_is_rejected(self):
        with mock.patch.object(base.data, "sha256", return_value="wrong"):
            with self.assertRaises(ValueError):
                base.contract()


class CompleteGateContracts(unittest.TestCase):
    def test_missing_actual_model_blocks_before_valid_parse(self):
        self.assert_model_change_blocks_valid("delete")

    def test_same_size_actual_model_change_blocks_before_valid_parse(self):
        self.assert_model_change_blocks_valid("same_size_change")

    def assert_model_change_blocks_valid(self, change):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "job" / "run"
            _, groups, meta, checked = fake_run(out)
            path = out / "bge_m3/models/epoch6.pt"
            if change == "delete":
                path.unlink()
            else:
                value = path.read_bytes()
                path.write_bytes(bytes([value[0] ^ 1]) + value[1:])
            with mock.patch.object(base.platform, "system", return_value="Linux"), \
                 mock.patch.object(base.public, "public_inputs", return_value=(groups, meta, checked)), \
                 mock.patch.object(base.public, "attach_labels") as parse:
                with self.assertRaises((ValueError, FileNotFoundError)):
                    base.evaluate(out, Path(tmp) / "evaluation")
            parse.assert_not_called()

    def test_complete_fixture_gate_then_exactly_one_valid_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "job" / "run"
            c, groups, meta, checked = fake_run(out)
            values, _, _ = base.validated_run(out, c, groups, meta, checked)
            self.assertEqual(set(values), set(base.ARMS))
            labelled = [base.data.Group(g.uid, g.sellers, g.items, tuple(int(v) for v in truth28()))
                        for g in groups["development"]]
            with mock.patch.object(base.platform, "system", return_value="Linux"), \
                 mock.patch.object(base.public, "public_inputs", return_value=(groups, meta, checked)), \
                 mock.patch.object(base.public, "attach_labels", return_value=labelled) as parse:
                result = base.evaluate(out, Path(tmp) / "evaluation")
            parse.assert_called_once_with(groups["development"], c, "development")
            self.assertTrue(all(v["automatic_classification"]["all_domains_pass"] for v in result["arms"].values()))
            self.assertEqual(len(result["columns"]), 22)

    def test_partial_model_run_cannot_consume_valid_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "job" / "run"
            _, groups, meta, checked = fake_run(out)
            (out / "bge_m3" / "manifest.json").unlink()
            with mock.patch.object(base.platform, "system", return_value="Linux"), \
                 mock.patch.object(base.public, "public_inputs", return_value=(groups, meta, checked)), \
                 mock.patch.object(base.public, "attach_labels") as parse:
                with self.assertRaises((ValueError, FileNotFoundError)):
                    base.evaluate(out, Path(tmp) / "evaluation")
            parse.assert_not_called()

    def test_wrong_point_threshold_and_corrupt_scores_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "job" / "run"
            c, groups, meta, checked = fake_run(out)
            p = out / "labse" / "calibration.json"
            value = base.data.read_json(p)
            value["epoch"] = 3
            base.data.write_json(p, value)
            with self.assertRaises(ValueError):
                base.validated_run(out, c, groups, meta, checked)

    def test_failed_outer_job_blocks_even_complete_inner_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "job" / "run"
            c, groups, meta, checked = fake_run(out)
            (out.parent / "exit_status.txt").write_text("124")
            with self.assertRaises(ValueError):
                base.validated_run(out, c, groups, meta, checked)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Real CPU Torch is not installed locally")
class TorchContracts(unittest.TestCase):
    def fixture(self):
        import torch

        class Tokenizer:
            def __call__(self, texts, *, padding, truncation, return_tensors="pt"):
                rows = [[1] + [2 + ord(ch) % 20 for ch in text] + [2] for text in texts]
                width = max(map(len, rows))
                return {"input_ids": torch.tensor([r + [0] * (width-len(r)) for r in rows]),
                        "attention_mask": torch.tensor([[1]*len(r) + [0]*(width-len(r)) for r in rows])}

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(24, 4)
                self.linear = torch.nn.Linear(4, 4)
                self.tokenizer = Tokenizer()

            def forward(self, batch):
                mask = batch["attention_mask"].unsqueeze(-1)
                x = (self.embedding(batch["input_ids"]) * mask).sum(1) / mask.sum(1)
                return {"sentence_embedding": self.linear(x)}

        torch.manual_seed(123)
        model = base.core.build_model(Encoder(), 4, 3)
        c = copy.deepcopy(base.contract())
        c["input"].update(microbatch=4, encoder_bf16=False)
        group = base.data.Group("hand_group", ("a", "b", "c"),
            ((("a1", "甲", "内容一"), ("a2", "乙", "内容二")),
             (("b1", "丙", "内容三"), ("b2", "丁", "内容四")),
             (("c1", "戊", "内容五"), ("c2", "己", "内容六"))), (1, 0, 0))
        return model, c, group

    def test_real_loss_gradients_two_adam_steps_and_disk_restore(self):
        import torch
        model, c, group = self.fixture()
        reference = copy.deepcopy(model)
        optimizer = base.core.make_optimizer(model, c)
        ref_optimizer = torch.optim.AdamW([
            {"params": reference.encoder.parameters(), "lr": 2e-5, "weight_decay": .01},
            {"params": reference.head.parameters(), "lr": .001, "weight_decay": 0.}],
            betas=(.9, .999), eps=1e-8, foreach=False)
        with tempfile.TemporaryDirectory() as tmp:
            for step in (1, 2):
                log = base.update(model, optimizer, group, c, "multilingual_e5_large", step, observe=True)
                ref_optimizer.zero_grad(set_to_none=True)
                texts = ["query: " + t + "\n" + d for records in group.items for _, t, d in records]
                raw = reference.encoder(reference.encoder.tokenizer(texts, padding=True, truncation=False))["sentence_embedding"]
                records = raw / raw.norm(dim=1, keepdim=True)
                account = torch.stack([records[0:2].mean(0), records[2:4].mean(0), records[4:6].mean(0)])
                account = account / account.norm(dim=1, keepdim=True)
                pairs = [(0, 1), (0, 2), (1, 2)]
                features = torch.stack([torch.cat([torch.abs(account[i]-account[j]), account[i]*account[j]]) for i,j in pairs])
                z = reference.head(features).flatten()
                y = torch.tensor([1., 0., 0.])
                loss = (torch.logaddexp(torch.zeros_like(z), z) - y*z).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
                for p, q in zip(model.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=2e-5)
                ref_optimizer.step()
                self.assertAlmostEqual(log["bce"], float(loss), places=6)
                for p, q in zip(model.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
                if step == 1:
                    path = Path(tmp) / "state.pt"
                    rec = base.core.save_state(path, model, optimizer, {"step": 1})
                    fresh, _, _ = self.fixture()
                    fresh_optimizer = base.core.make_optimizer(fresh, c)
                    base.core.restore_state(path, fresh, fresh_optimizer, rec["state_sha256"])
                    model, optimizer = fresh, fresh_optimizer
            self.assertTrue(all(float(x["step"]) == 2 for x in optimizer.state.values()))

    def test_real_prefix_symmetry_and_no_silent_truncation(self):
        import torch
        model, c, group = self.fixture()
        model.eval()
        original = base.logits(model, group, c, "multilingual_e5_large")
        swapped = base.data.Group(group.uid, tuple(reversed(group.sellers)), tuple(reversed(group.items)), group.labels)
        reversed_values = base.logits(model, swapped, c, "multilingual_e5_large")
        torch.testing.assert_close(original, reversed_values.flip(0))
        no_prefix = base.logits(model, group, c, "labse")
        self.assertFalse(torch.allclose(original, no_prefix))
        c["input"]["token_budget"] = 2
        with self.assertRaises(ValueError):
            base.logits(model, group, c, "multilingual_e5_large")


if __name__ == "__main__":
    unittest.main()
