from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_character_baseline as baseline


class CharacterBaselineContracts(unittest.TestCase):
    def test_independent_character_counts_idf_and_cosine(self):
        train = ["aaab aaab", "aaab bbbc", "bbbc aaab"]
        dev = ["aaab aaab", "bbbc aaab", "cccc"]

        def grams(text):
            return Counter(text[i:i+n] for n in (3, 4, 5)
                           for i in range(len(text)-n+1))

        df = Counter()
        for text in train:
            df.update(grams(text).keys())
        expected_names = sorted(g for g, count in df.items() if count >= 2)
        idf = np.asarray([1 + math.log(4 / (1 + df[g])) for g in expected_names])
        manual = []
        for text in dev:
            counts = grams(text)
            values = np.asarray([(1 + math.log(counts[g])) * weight if counts[g] else 0
                                 for g, weight in zip(expected_names, idf)])
            norm = np.linalg.norm(values)
            manual.append(values / norm if norm else values)
        manual = np.asarray(manual)
        vectorizer, matrix = baseline.fit_vectors(train, dev)
        self.assertEqual(list(vectorizer.get_feature_names_out()), expected_names)
        expected_vocabulary = [[name, index] for index, name in enumerate(expected_names)]
        expected_hash = hashlib.sha256(json.dumps(expected_vocabulary, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(baseline.vocabulary_digest(vectorizer), expected_hash)
        np.testing.assert_allclose(vectorizer.idf_, idf, rtol=0, atol=1e-15)
        np.testing.assert_allclose(matrix.toarray(), manual, rtol=0, atol=1e-15)
        actual = baseline.score_pairs(matrix, np.array([0, 0]), np.array([1, 2]))
        np.testing.assert_allclose(actual, [manual[0] @ manual[1], 0], rtol=0, atol=1e-15)

    def test_development_cannot_extend_training_vocabulary(self):
        vectorizer, matrix = baseline.fit_vectors(["zzzy", "zzzx"], ["aaab", "aaac"])
        self.assertEqual(set(vectorizer.vocabulary_), {"zzz"})
        self.assertEqual(matrix.nnz, 0)
        before = vectorizer.idf_.copy()
        vectorizer.transform(["aaa aaa aaa"])
        np.testing.assert_array_equal(vectorizer.idf_, before)
        self.assertEqual(set(vectorizer.vocabulary_), {"zzz"})
        self.assertEqual(baseline.score_pairs(matrix, np.array([0]), np.array([1]))[0], 0)

    def test_pair_order_orientation_and_batches(self):
        matrix = sparse.csr_matrix([[1., 0.], [.6, .8], [0., 0.]])
        left, right = np.array([0, 1, 0, 1]), np.array([1, 0, 2, 1])
        first = baseline.score_pairs(matrix, left, right, batch_size=1)
        np.testing.assert_allclose(first, [.6, .6, 0, 1], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(first, baseline.score_pairs(matrix, left, right, batch_size=3))
        order = np.array([3, 1, 0, 2])
        np.testing.assert_array_equal(first[order], baseline.score_pairs(matrix, left[order], right[order]))

    def test_pair_index_failures(self):
        matrix = sparse.eye(2, format="csr")
        for left, right, batch in (([0], [2], 1), ([-1], [0], 1), ([0], [1, 0], 1),
                                   ([0.5], [1], 1), ([0], [1], 0)):
            with self.subTest(left=left, right=right, batch=batch), self.assertRaises(ValueError):
                baseline.score_pairs(matrix, np.asarray(left), np.asarray(right), batch)

    def test_nonempty_corpus_required(self):
        for train, dev in (([], ["aaa"]), (["aaa"], []), ([""], ["aaa"]), ([None], ["aaa"])):
            with self.subTest(train=train, dev=dev), self.assertRaises(ValueError):
                baseline.fit_vectors(train, dev)

    def test_inherited_projection_and_sealed_split_guard(self):
        self.assertEqual(baseline.style.transferable_style_projection("商品12，确认。"), "W2N2.W2.")
        self.assertEqual(baseline.style.transferable_style_projection("e\u0301"), "W1")
        for split in ("audit_a", "audit_b", "test"):
            with self.subTest(split=split), self.assertRaises(baseline.style.StyleTransferContractError):
                baseline.style.load_chinese_style_streams({}, split)


if __name__ == "__main__":
    unittest.main()
