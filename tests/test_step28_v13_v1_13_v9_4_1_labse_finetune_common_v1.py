from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_labse_finetune_common_v1 as common
import step7_v4_common as step7_common


class CharacterTokenizer:
    def __call__(self, text: str, **_kwargs):
        return {"input_ids": [101, *[ord(value) for value in text], 102]}


class LabseFinetuneCommonV1Tests(unittest.TestCase):
    def test_policy_keeps_audit_truth_and_raw_identity_text_forbidden(self) -> None:
        policy = common.load_policy()
        self.assertEqual(
            policy["canonical_self_hash"], common.canonical_self_hash(policy)
        )
        self.assertEqual(policy["forbidden_splits"], ["audit_a", "audit_b"])
        self.assertEqual(policy["text_input"]["file_role"], "redacted_items.jsonl")
        self.assertFalse(policy["text_input"]["whole_document_truncation_allowed"])
        self.assertIsNone(policy["text_input"]["maximum_chunks_per_text"])
        self.assertEqual(policy["formal_layout"]["items_per_world"], 99)
        for split in policy["allowed_splits"]:
            path = common.verified_redacted_items_path(policy, split)
            self.assertTrue(path.is_file())
            self.assertIn(f"{split}\\observed\\redacted_items.jsonl", str(path))

    def test_exact_chunking_reconstructs_every_character_without_overlap(self) -> None:
        tokenizer = CharacterTokenizer()
        text = "第一段较长文字，需要分块。\n第二段也必须完整保留；不能截断。"
        chunks = common.chunk_text_exact(tokenizer, text, budget=12)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(common.tokenizer_length(tokenizer, row) <= 12 for row in chunks))

    def test_nested_budget_subsets_are_label_free_and_nested(self) -> None:
        worlds = [f"world_{index:03d}" for index in range(10)]
        forward = common.nested_world_subsets(worlds, [1, 3, 10])
        reverse = common.nested_world_subsets(list(reversed(worlds)), [1, 3, 10])
        self.assertEqual(forward, reverse)
        self.assertTrue(set(forward[1]) <= set(forward[3]) <= set(forward[10]))

    def test_redacted_index_uses_all_unique_text_and_requires_both_fields(self) -> None:
        rows = []
        for world in ("w0", "w1"):
            for seller in ("s0", "s1"):
                rows.extend(
                    [
                        {
                            "world_uid": world,
                            "seller_uid": f"{world}_{seller}",
                            "item_uid": f"{world}_{seller}_i0",
                            "title": "相同标题",
                            "description": f"{seller}描述一",
                        },
                        {
                            "world_uid": world,
                            "seller_uid": f"{world}_{seller}",
                            "item_uid": f"{world}_{seller}_i1",
                            "title": "相同标题",
                            "description": f"{seller}描述二",
                        },
                    ]
                )
        index = common.build_redacted_text_index(
            rows,
            expected_worlds=2,
            expected_sellers_per_world=2,
            expected_items_per_world=4,
        )
        self.assertEqual(len(index), 2)
        self.assertEqual(len(index["w0"]["w0_s0"]["title"]), 1)
        self.assertEqual(len(index["w0"]["w0_s0"]["description"]), 2)

    def test_six_aggregates_match_identical_seller_vectors(self) -> None:
        seller = {
            "title": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f8"),
            "description": np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype="<f8"),
        }
        observed = common.six_pair_aggregates(seller, seller, top_k=3)
        np.testing.assert_allclose(
            observed,
            np.asarray([1.0, 0.5, 1.0, 0.5, 1.0, 0.5]),
            rtol=0.0,
            atol=1e-12,
        )

    def test_six_aggregates_match_the_frozen_step7_definition(self) -> None:
        rng = np.random.default_rng(17)
        left = {
            "title": rng.normal(size=(4, 7)),
            "description": rng.normal(size=(3, 7)),
        }
        right = {
            "title": rng.normal(size=(2, 7)),
            "description": rng.normal(size=(5, 7)),
        }
        frozen_left = {
            field: (matrix, np.ones(len(matrix), dtype=np.int64))
            for field, matrix in left.items()
        }
        frozen_right = {
            field: (matrix, np.ones(len(matrix), dtype=np.int64))
            for field, matrix in right.items()
        }
        frozen, _weighted = step7_common.aggregate_pair_vectors(
            frozen_left, frozen_right, top_k=3
        )
        expected = np.asarray(
            [frozen[name] for name in step7_common.AGGREGATE_SUFFIXES],
            dtype="<f8",
        )
        observed = common.six_pair_aggregates(left, right, top_k=3)
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-15)

    def test_differentiable_aggregates_match_numpy_and_have_gradients(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        rng = np.random.default_rng(29)
        left_numpy = {
            "title": rng.normal(size=(3, 5)),
            "description": rng.normal(size=(4, 5)),
        }
        right_numpy = {
            "title": rng.normal(size=(2, 5)),
            "description": rng.normal(size=(3, 5)),
        }
        left = {
            field: torch.tensor(value, dtype=torch.float64, requires_grad=True)
            for field, value in left_numpy.items()
        }
        right = {
            field: torch.tensor(value, dtype=torch.float64, requires_grad=True)
            for field, value in right_numpy.items()
        }
        observed = common.torch_six_pair_aggregates(left, right, top_k=3)
        expected = common.six_pair_aggregates(left_numpy, right_numpy, top_k=3)
        np.testing.assert_allclose(
            observed.detach().numpy(), expected, rtol=0.0, atol=2e-15
        )
        observed.sum().backward()
        for matrix in (*left.values(), *right.values()):
            self.assertIsNotNone(matrix.grad)
            self.assertTrue(torch.isfinite(matrix.grad).all())


if __name__ == "__main__":
    unittest.main()
