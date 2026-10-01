"""Numerical and retained-history contracts, runnable without PyTorch."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_continual_core as core


def examples(count: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((count, core.FEATURE_COUNT), dtype=np.float32)
    values[:, 0] = np.arange(count)
    return values, (np.arange(count) % 2).astype(np.uint8)


class ContinualCoreContracts(unittest.TestCase):
    def test_preprocessing_uses_first_stage_and_population_scale(self):
        first = np.zeros((3, core.FEATURE_COUNT))
        first[:, 0] = [1., np.nan, 5.]
        first[:, 1] = np.nan
        first[:, 2] = 7.
        prep = core.Preprocessor.fit(first)
        self.assertEqual(prep.medians[0], 3.)
        self.assertAlmostEqual(prep.scales[0], np.sqrt(8 / 3))
        self.assertEqual(prep.medians[1], 0.)
        self.assertEqual(prep.scales[1], 1.)
        np.testing.assert_array_equal(prep.transform(first)[:, 1:], 0.)
        snapshot = [value.copy() for value in (prep.medians, prep.means, prep.scales)]
        future = first.copy()
        future[0, 0] = 1000.
        prep.transform(future)
        for before, after in zip(snapshot, (prep.medians, prep.means, prep.scales), strict=True):
            np.testing.assert_array_equal(before, after)

    def test_feature_label_shapes_and_values_rejected(self):
        x, y = examples(3)
        for values, labels in ((x, y[:2]), (x[:, :-1], y), (np.full_like(x, np.inf), y),
                                (x, np.array([0, 1, 2]))):
            with self.subTest(shape=values.shape, labels=labels.tolist()):
                with self.assertRaises(ValueError):
                    core.validate_examples(values, labels)

    def test_uniform_reservoir_chunking_preserves_same_arrival_sequence(self):
        x, y = examples(101)
        whole, chunked = core.Reservoir(2048, 19), core.Reservoir(2048, 19)
        whole.update_after_stage(x, y)
        chunked.update_after_stage(x[:13], y[:13])
        chunked.update_after_stage(x[13:70], y[13:70])
        chunked.update_after_stage(x[70:], y[70:])
        self.assertEqual(whole.seen, 101)
        self.assertEqual(whole.serialized_parts(), chunked.serialized_parts())
        self.assertEqual(len(np.unique(whole.records["features"][:, 0])), whole.capacity)

    def test_retained_state_includes_rng_and_fits_real_byte_budget(self):
        memory = core.Reservoir(core.BUDGET, 11)
        memory.update_after_stage(*examples(4100))
        self.assertEqual(memory.records.dtype.itemsize, 229)
        self.assertEqual(memory.capacity, 2284)
        array_bytes, state_bytes = memory.serialized_parts()
        self.assertLessEqual(len(array_bytes) + len(state_bytes), core.BUDGET)
        self.assertGreater(len(array_bytes) + len(state_bytes), memory.records.nbytes)

    def test_disk_format_replays_future_sampling_and_retention(self):
        first = core.Reservoir(4096, 37)
        first.update_after_stage(*examples(30))
        first.sample(4)
        restored = core.Reservoir.restore(*first.serialized_parts())
        for _ in range(3):
            a, b = first.sample(7), restored.sample(7)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
        new_x, new_y = examples(11)
        new_x[:, 0] += 30
        first.update_after_stage(new_x, new_y)
        restored.update_after_stage(new_x, new_y)
        self.assertEqual(first.serialized_parts(), restored.serialized_parts())

    def test_short_memory_replay_is_without_replacement_and_never_inserts(self):
        memory = core.Reservoir(core.BUDGET, 11)
        self.assertIsNone(memory.sample(256))
        memory.update_after_stage(*examples(3))
        samples = memory.sample(256)
        self.assertEqual(len(samples[1]), 3)
        np.testing.assert_array_equal(np.sort(samples[0][:, 0]), [0, 1, 2])
        self.assertEqual(memory.seen, 3)

    def test_extra_memory_draws_cannot_change_current_sample_order(self):
        wanted = core.current_permutation(83, 23, 2, 1)
        memory = core.Reservoir(4096, 23)
        memory.update_after_stage(*examples(90))
        for _ in range(11):
            memory.sample(2)
        np.testing.assert_array_equal(wanted, core.current_permutation(83, 23, 2, 1))
        self.assertFalse(np.array_equal(wanted, core.current_permutation(83, 23, 2, 2)))

    def test_bce_and_all_numpy_gradient_entries_match_finite_differences(self):
        # A 2-input/2-hidden hand fixture exercises the general verification equations.
        parameters = [np.array([[.2, -.1], [.3, .4]]), np.array([.8, .6]),
                      np.array([[.5, -.2]]), np.array([.1])]
        x, y = np.array([[1., 2.], [-1., 1.], [.5, -.4]]), np.array([1., 0., 1.])
        loss, gradient = core.independent_bce_gradient(parameters, x, y)
        logits = np.maximum(x @ parameters[0].T + parameters[1], 0.) @ parameters[2].T + parameters[3]
        probability = 1 / (1 + np.exp(-logits.ravel()))
        self.assertAlmostEqual(loss, float(np.mean(-y*np.log(probability)-(1-y)*np.log1p(-probability))))
        for index, parameter in enumerate(parameters):
            for coordinate in range(parameter.size):
                plus, minus = [p.copy() for p in parameters], [p.copy() for p in parameters]
                plus[index].flat[coordinate] += 1e-6
                minus[index].flat[coordinate] -= 1e-6
                difference = (core.independent_bce_gradient(plus, x, y)[0]
                              - core.independent_bce_gradient(minus, x, y)[0]) / 2e-6
                self.assertAlmostEqual(difference, gradient[index].flat[coordinate], places=8)


if __name__ == "__main__":
    unittest.main()
