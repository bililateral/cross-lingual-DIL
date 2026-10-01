from __future__ import annotations

import copy
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import step28_continual_text as core
import step28_labse_finetune_common as common

try:
    import torch
except ImportError:
    torch = None


def example_batch(count: int = 3, offset: int = 0, width: int = 1) -> core.PairBatch:
    """Hand-created verification inputs, never project supervision or a dataset."""
    accounts = {f"check_{offset + i:04d}": core.canonical_fields({
        "title": [f"普通商品说明{i}。" * width, f"包装备注{i}。"],
        "description": [f"这是用于核对计算的文本{i}。" * width],
    }) for i in range(count)}
    edges = tuple(itertools.combinations(sorted(accounts), 2))
    rng = np.random.default_rng(offset + 79)
    return core.PairBatch(accounts, edges, rng.normal(0, .3, (len(edges), 51)),
                          (np.arange(len(edges)) % 3 == 0).astype(np.uint8))


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        def ids(value):
            return [1, *[3 + ord(char) % 250 for char in value], 2]
        if isinstance(text, str):
            return {"input_ids": ids(text)}
        rows = [ids(value) for value in text]
        maximum = max(map(len, rows))
        return {"input_ids": torch.tensor([row + [0] * (maximum - len(row)) for row in rows]),
                "attention_mask": torch.tensor([[1] * len(row) + [0] * (maximum - len(row)) for row in rows])}


def toy_encoder():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(256, 7)
            self.dropout = torch.nn.Dropout(.2)

        def forward(self, features):
            values = self.dropout(self.embedding(features["input_ids"]))
            mask = features["attention_mask"].unsqueeze(2)
            return {"sentence_embedding": (values * mask).sum(1) / mask.sum(1)}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(47)
        return Encoder()


class TextMemoryContracts(unittest.TestCase):
    def test_full_fields_preserve_text_and_canonicalize_duplicates(self):
        fields = core.canonical_fields({"title": ["  标题\n", "另一个", "  标题\n"], "description": ["描述"]})
        self.assertEqual(set(fields["title"]), {"  标题\n", "另一个"})
        with self.assertRaises(ValueError):
            core.canonical_fields({**fields, "controller": ["forbidden"]})

    def test_sparse_edge_roundtrip_is_self_contained_and_float64_exact(self):
        batch = example_batch()
        batch.numeric[0, 5] = np.nan
        batch.numeric[0, 6] = 1.000000000000001
        payload = core.edge_payload(batch, 0)
        restored = core.batch_from_payloads([payload])
        self.assertEqual(restored.edges, (batch.edges[0],))
        self.assertEqual(set(restored.accounts), set(batch.edges[0]))
        self.assertEqual(core.edge_payload(restored, 0), payload)
        self.assertNotEqual(restored.numeric[0, 6], float(np.float32(restored.numeric[0, 6])))
        with self.assertRaises(ValueError):
            restored.validate(complete_world=True)

    def test_missing_history_and_conflicting_endpoint_fail(self):
        first, second = example_batch(), example_batch()
        del second.accounts[second.edges[0][0]]
        with self.assertRaises(ValueError):
            second.validate()
        second = example_batch()
        second.accounts[second.edges[1][0]]["description"] = ("冲突描述",)
        with self.assertRaises(ValueError):
            core.batch_from_payloads([core.edge_payload(first, 0), core.edge_payload(second, 1)])

    def test_current_world_requires_all_378_edges(self):
        batch = example_batch(28)
        batch.validate(training=True, complete_world=True)
        batch.edges = batch.edges[:-1]
        batch.numeric, batch.labels = batch.numeric[:-1], batch.labels[:-1]
        with self.assertRaises(ValueError):
            batch.validate(complete_world=True)

    def test_streaming_random_prefix_matches_independent_full_history_reference(self):
        for seed in (11, 23, 37):
            memory = core.ByteMemory(6500, seed)
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            all_rows = []
            for stage in range(1, 5):
                batches = [example_batch(2, stage * 100 + i * 2, 1 + i % 7) for i in range(17)]
                for batch in batches:
                    payload = core.edge_payload(batch, 0)
                    all_rows.append((int(rng.bit_generator.random_raw()), len(all_rows) + 1, payload))
                expected, size = [], 1024
                for entry in sorted(all_rows, key=lambda row: row[:2]):
                    size += 20 + len(entry[2])
                    if size > 6500:
                        break  # deliberate prefix, NOT skip-to-fit packing
                    expected.append(entry)
                memory.update_after_stage(batches, stage)
                self.assertEqual(memory.entries, expected)
                self.assertEqual(len(memory.to_bytes()), memory.used_bytes)
                self.assertLessEqual(memory.used_bytes, 6500)

    def test_memory_restore_preserves_future_sampling_and_selection(self):
        memory = core.ByteMemory(6500, 11)
        memory.update_after_stage([example_batch(4)], 1)
        memory.sample(2)
        restored = core.ByteMemory.from_bytes(memory.to_bytes())
        self.assertEqual(memory.sample(3).edges, restored.sample(3).edges)
        for obj in (memory, restored):
            obj.update_after_stage([example_batch(5, 100)], 2)
        self.assertEqual(memory.to_bytes(), restored.to_bytes())
        self.assertEqual(len(restored.sample(999).edges), len(restored.entries))

    def test_selection_does_not_use_labels_or_optimization_repetitions(self):
        first, second = core.ByteMemory(5000, 23), core.ByteMemory(5000, 23)
        batch, changed = example_batch(6), example_batch(6)
        changed.labels = 1 - changed.labels
        first.update_after_stage([batch], 1)
        second.update_after_stage([changed], 1)
        self.assertEqual([x[:2] for x in first.entries], [x[:2] for x in second.entries])
        with self.assertRaises(ValueError):
            first.update_after_stage([batch], 1)

    def test_oversized_full_edge_and_truncated_memory_fail(self):
        memory = core.ByteMemory(1500)
        with self.assertRaises(ValueError):
            memory.update_after_stage([example_batch(2, width=100)], 1)
        other = core.ByteMemory(6500)
        other.update_after_stage([example_batch()], 1)
        with self.assertRaises(ValueError):
            core.ByteMemory.from_bytes(other.to_bytes()[:-1])

    def test_columns_and_first_only_scaler_reference(self):
        self.assertEqual(core.load_policy()["epochs_per_stage"], 2)
        numeric = np.tile(np.arange(51), (2, 1)).astype(float)
        dynamic = np.tile(np.arange(6) + 100, (2, 1)).astype(float)
        raw = core.combine_numpy(numeric, dynamic)
        np.testing.assert_array_equal(raw[0], [*range(18), *range(100, 106), *range(18, 51)])
        raw[0, 0], raw[1, 0] = np.nan, 4
        scaler = core.Preprocessor.fit(raw)
        self.assertEqual(scaler.medians[0], 4)
        self.assertEqual(scaler.scales[0], 1)
        self.assertTrue(np.isfinite(scaler.transform(raw)).all())

    def test_chunk_plan_reconstructs_long_original_text(self):
        batch = example_batch(2, width=40)
        chunks, spans = core.prepare_text(batch, CharacterTokenizer())
        self.assertTrue(any(span.stop - span.start > 1 for span in spans.values()))
        for text, span in spans.items():
            self.assertEqual("".join(chunks[span]), text)
            self.assertTrue(all(len(CharacterTokenizer()(value)["input_ids"]) <= 256 for value in chunks[span]))


@unittest.skipIf(torch is None, "PyTorch unavailable locally; actual runtime remains required")
class TextTrainingContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.encoder, self.head = toy_encoder(), core.make_head()
        self.tokenizer = CharacterTokenizer()
        self.current = example_batch()
        self.history = example_batch(2, 100)
        self.scaler = core.Preprocessor(np.zeros(57), np.zeros(57), np.ones(57))

    def test_hierarchy_and_derivative_match_independent_numpy(self):
        batch = example_batch(2)
        _chunks, original_spans = core.prepare_text(batch, self.tokenizer)
        # Force unequal numbers of chunks per text, so flattening would be wrong.
        spans, cursor = {}, 0
        for i, text in enumerate(original_spans):
            spans[text] = slice(cursor, cursor + i % 3 + 1)
            cursor = spans[text].stop
        values = np.random.default_rng(81).normal(size=(cursor, 7))
        tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        def independent(array):
            text_vectors = {}
            for text, span in spans.items():
                unit = array[span] / np.linalg.norm(array[span], axis=1, keepdims=True)
                mean = unit.mean(0)
                text_vectors[text] = mean / np.linalg.norm(mean)
            accounts = {uid: {field: np.stack([text_vectors[text] for text in fields[field]])
                              for field in core.FIELDS} for uid, fields in batch.accounts.items()}
            return common.six_pair_aggregates(accounts[batch.edges[0][0]], accounts[batch.edges[0][1]])
        actual = core.aggregate_chunks(batch, tensor, spans)
        np.testing.assert_allclose(actual.detach().numpy()[0], independent(values), atol=2e-15, rtol=0)
        actual.sum().backward()
        for row, column in ((0, 0), (cursor // 2, 3), (cursor - 1, 6)):
            plus, minus = values.copy(), values.copy()
            plus[row, column] += 1e-6
            minus[row, column] -= 1e-6
            derivative = (independent(plus).sum() - independent(minus).sum()) / 2e-6
            self.assertAlmostEqual(float(tensor.grad[row, column]), derivative, places=7)

    def test_affine_preserves_dynamic_gradient_and_column_order(self):
        dynamic = torch.arange(18, dtype=torch.float64).reshape(3, 6).requires_grad_(True)
        self.scaler.scales[18:24] = np.arange(1, 7)
        observed = core.scaled_features(self.current.numeric, dynamic, self.scaler)
        expected = self.scaler.transform(core.combine_numpy(self.current.numeric, dynamic.detach().numpy()))
        np.testing.assert_array_equal(observed.detach().numpy(), expected)
        observed[:, 18:24].sum().backward()
        np.testing.assert_allclose(dynamic.grad.numpy(), np.tile(1 / np.arange(1, 7), (3, 1)), rtol=0, atol=0)

    def test_extra_history_forward_does_not_change_current_dropout(self):
        state = torch.get_rng_state().clone()
        def logits(seed, batch):
            with core.paired_rng(torch.device("cpu"), seed):
                return core.batch_logits(self.encoder, self.head, self.tokenizer, batch,
                                          self.scaler, bf16=False).detach()
        before = logits(81, self.current)
        logits(42, self.history)
        after = logits(81, self.current)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_sequential_backward_matches_summed_objective_and_full_update(self):
        reference_encoder, reference_head = copy.deepcopy(self.encoder), copy.deepcopy(self.head)
        actual_optimizer = core.make_text_optimizer(self.encoder, self.head)
        reference_optimizer = core.make_text_optimizer(reference_encoder, reference_head)
        losses = []
        for batch, seed in ((self.current, 31), (self.history, 73)):
            with core.paired_rng(torch.device("cpu"), seed):
                logits = core.batch_logits(reference_encoder, reference_head, self.tokenizer,
                                           batch, self.scaler, bf16=False)
                # Independent stable BCE expression, unequal 3-vs-1 branch sizes.
                labels = torch.tensor(batch.labels, dtype=logits.dtype)
                losses.append((torch.logaddexp(torch.zeros_like(logits), logits) - labels * logits).mean())
        sum(losses).backward()
        reference_parameters = list(reference_encoder.parameters()) + list(reference_head.parameters())
        torch.nn.utils.clip_grad_norm_(reference_parameters, 1., error_if_nonfinite=True)
        reference_optimizer.step()
        core.train_update(self.encoder, self.head, actual_optimizer, self.tokenizer, self.current,
                          self.history, self.scaler, current_seed=31, history_seed=73, bf16=False)
        for actual, expected in zip(list(self.encoder.parameters()) + list(self.head.parameters()), reference_parameters):
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-8, rtol=2e-6)
            torch.testing.assert_close(actual, expected, atol=3e-8, rtol=2e-6)

    def test_initial_eval_scaler_does_not_train_or_use_labels(self):
        state = copy.deepcopy(self.encoder.state_dict())
        self.current.labels = None
        self.encoder.train()
        actual = core.fit_first_scaler(self.encoder, self.tokenizer, [self.current], bf16=False)
        self.assertTrue(self.encoder.training)
        self.encoder.eval()
        with torch.no_grad():
            dynamic = core.dynamic_features(self.encoder, self.tokenizer, self.current, bf16=False).numpy()
        expected = core.Preprocessor.fit(core.combine_numpy(self.current.numeric, dynamic))
        np.testing.assert_array_equal(actual.means, expected.means)
        for name, value in self.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]))
        self.assertTrue(all(p.grad is None for p in self.encoder.parameters()))

    def test_nonempty_optimizer_and_memory_reload_reproduce_next_update(self):
        optimizer = core.make_text_optimizer(self.encoder, self.head)
        core.train_update(self.encoder, self.head, optimizer, self.tokenizer, self.current,
                          None, self.scaler, current_seed=31, history_seed=73, bf16=False)
        memory = core.ByteMemory(6500)
        memory.update_after_stage([self.history], 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.pt"
            core.save_state(path, self.encoder, self.head, optimizer, self.scaler, {"next_stage": 2}, memory)
            before = core.score_batch(self.encoder, self.head, self.tokenizer, self.current, self.scaler, bf16=False)
            other_encoder, other_head = toy_encoder(), core.make_head()
            other_optimizer = core.make_text_optimizer(other_encoder, other_head)
            scaler, progress, restored = core.restore_state(path, other_encoder, other_head, other_optimizer)
            for name in ("medians", "means", "scales"):
                np.testing.assert_array_equal(getattr(scaler, name), getattr(self.scaler, name))
            self.assertEqual(progress, {"next_stage": 2})
            self.assertTrue(other_optimizer.state)
            np.testing.assert_array_equal(before, core.score_batch(other_encoder, other_head, self.tokenizer,
                                                                   self.current, scaler, bf16=False))
            for encoder, head, opt, prep, cache in ((self.encoder, self.head, optimizer, self.scaler, memory),
                                                   (other_encoder, other_head, other_optimizer, scaler, restored)):
                core.train_update(encoder, head, opt, self.tokenizer, self.current, cache.sample(16), prep,
                                  current_seed=41, history_seed=83, bf16=False)
            for actual, expected in zip(list(self.encoder.parameters()) + list(self.head.parameters()),
                                        list(other_encoder.parameters()) + list(other_head.parameters())):
                self.assertTrue(torch.equal(actual, expected))
            for actual, expected in zip(optimizer.state.values(), other_optimizer.state.values()):
                for key in actual:
                    self.assertTrue(torch.equal(actual[key], expected[key]))

    def test_stage_schedule_retains_only_prior_memory_and_pairs_current_order(self):
        memory = core.ByteMemory(6500)
        memory.update_after_stage([self.history], 1)
        worlds = [example_batch(28, 300)]
        summaries = []
        for cache in (None, memory):
            encoder, head = copy.deepcopy(self.encoder), copy.deepcopy(self.head)
            summaries.append(core.train_stage(encoder, head, core.make_text_optimizer(encoder, head),
                                               self.tokenizer, worlds, self.scaler, order_seed=11,
                                               stage=2, epochs=1, memory=cache, bf16=False))
        self.assertEqual(summaries[0]["current_order_sha256"], summaries[1]["current_order_sha256"])
        self.assertEqual(summaries[1]["current_presentations"], 378)
        self.assertEqual(summaries[1]["replay_presentations"], 1)
        self.assertEqual(memory.last_stage, 1)
        self.assertEqual(memory.seen, 1)


if __name__ == "__main__":
    unittest.main()
