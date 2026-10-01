"""Hand-checkable partition, model mapping and once-only diagnostic boundaries."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import step28_continual_replay_probe as probe
from test_step28_continual_expression_run_contracts import hand_labels


def fixture(root: Path) -> tuple:
    out = root / 'run'
    out.mkdir()
    selected = {o: [SimpleNamespace(uid=o + str(i)) for i in range(60)] for o in probe.pilot.ORDERS}
    valid = [SimpleNamespace(uid='valid' + str(i)) for i in range(60)]
    partitions = {o: probe.partition([g.uid for g in rows], [g.uid for g in rows[:6]],
                                    [g.uid for g in rows[:3]]) for o, rows in selected.items()}
    registry, points = {}, {}
    scores = np.zeros((60, 378), dtype=np.float32)
    original = root / 'original.npy'
    np.save(original, scores, allow_pickle=False)
    for n in probe.NAMES:
        registry[n] = {'valid': probe.data.record(original, root)}
        files = {}
        for split in ('train', 'development'):
            p = out / f'{n}_{split}.npy'
            np.save(p, scores, allow_pickle=False)
            files[split] = probe.data.record(p, out)
        points[n] = {'source': registry[n], 'original_valid_exact': True,
                     'model_unchanged': True, 'scores': files}
    ctx = {'config': probe.contract(), 'selected': selected, 'groups': {'development': valid,
           'train': [g for rows in selected.values() for g in rows]}, 'inputs': {},
           'partitions': partitions, 'registry': registry, 'model_config': {},
           'valid_metrics': {n: np.zeros((60, 22)) for n in probe.NAMES}}
    r = {'status': probe.COMPLETE, 'config': ctx['config'], 'source_files': probe.sources(),
         'points': points, 'group_forwards': 1080, 'optimizer_updates': 0, 'label_parses': 0,
         'inputs': {}, 'partitions': partitions, 'train_ids': {o: [g.uid for g in rows] for o, rows in selected.items()},
         'valid_ids': [g.uid for g in valid], 'total_seconds': 1.}
    probe.data.write_json(out / 'manifest.json', r)
    return out, ctx, r


class ProbeContracts(unittest.TestCase):
    def test_evicted_groups_are_still_exposed(self):
        ids = [str(i) for i in range(60)]
        p = probe.partition(ids, ids[:6], ids[:3] + ['new1', 'new2', 'new3'])
        self.assertEqual(p['ever_replayed'], list(range(6)))
        self.assertEqual(p['evicted_after_replay'], [3, 4, 5])
        self.assertEqual(p['never_replayed'], list(range(6, 60)))
        with self.assertRaises(ValueError):
            probe.partition(ids, ids[:5] + ids[:1], ids[:3])
        with self.assertRaises(ValueError):
            probe.partition(ids, ids[:6], ['8'])

    def test_partition_preserves_public_row_order(self):
        ids = [str(i) for i in reversed(range(60))]
        p = probe.partition(ids, [str(i) for i in range(6)], ['0'])
        self.assertEqual(p['ever_replayed'], list(range(54, 60)))
        self.assertEqual(p['retained_until_end'], [59])

    def test_unequal_subset_sizes_and_order_equal_summary(self):
        arrays, masks, valid = {}, {}, {}
        for j, o in enumerate(probe.pilot.ORDERS):
            masks[o] = {'ever_replayed': list(range(6)), 'never_replayed': list(range(6, 60))}
            for role in probe.ROLES:
                a = np.zeros((60, 23))
                if role == 'shared':
                    a[:] = .2
                if role == 'sequential':
                    a[:] = .3
                if role == 'er':
                    a[:6] = .4 + .1 * j
                    a[6:] = .1
                arrays[o + '_' + role] = a
                valid[o + '_' + role] = np.full((60, 22), .9)
        r = probe.summarize(arrays, masks, valid)
        self.assertAlmostEqual(r['order_equal_er_minus_seq']['ever_replayed']['average_precision'], .2)
        self.assertAlmostEqual(r['order_equal_er_minus_seq']['never_replayed']['average_precision'], -.2)
        self.assertAlmostEqual(r['ever_minus_never_contrast']['average_precision'], .4)
        self.assertAlmostEqual(r['by_order']['ABC']['train']['ever_replayed']['changes']['er_minus_shared']['average_precision'], .2)
        self.assertEqual(r['by_order']['ABC']['train']['never_replayed']['groups'], 54)

    def test_wrong_model_metadata_and_valid_fail_before_train_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'valid.npy'
            original = np.zeros((60, 378), dtype=np.float32)
            np.save(path, original, allow_pickle=False)
            spec = {'model': {'path': 'unused', 'state_sha256': 's'}, 'valid': {'path': 'valid.npy'},
                    'point_name': 'ABC_shared', 'model_state_sha256': 'model'}
            model = Mock()
            with patch.object(probe.data, 'verify', return_value=path), \
                 patch.object(probe.core, 'restore_state', return_value={'point': 'WRONG', 'config': {}}), \
                 patch.object(probe.core, 'score') as score:
                with self.assertRaises(ValueError):
                    probe.score_one(model, spec, [], [], {}, lambda: None)
                score.assert_not_called()
            with patch.object(probe.data, 'verify', return_value=path), \
                 patch.object(probe.core, 'restore_state', return_value={'point': 'ABC_shared', 'config': {}}), \
                 patch.object(probe.core, 'state_digest', return_value='model'), \
                 patch.object(probe.core, 'score', return_value=original + 1) as score:
                with self.assertRaises(ValueError):
                    probe.score_one(model, spec, ['train'], ['valid'], {}, lambda: None)
                self.assertEqual(score.call_count, 1)
                self.assertEqual(score.call_args.args[1], ['valid'])

    def test_complete_gate_rejects_missing_point_id_and_score_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, ctx, original = fixture(root)
            with patch.object(probe.data, 'ROOT', root), patch.object(probe, 'sources', return_value=original['source_files']):
                arrays, _ = probe.validated_scores(out, ctx)
                self.assertEqual(len(arrays), 9)
                for field, value in [('group_forwards', 1079), ('label_parses', 1), ('optimizer_updates', 1)]:
                    r = copy.deepcopy(original)
                    r[field] = value
                    probe.data.write_json(out / 'manifest.json', r)
                    with self.assertRaises(ValueError): probe.validated_scores(out, ctx)
                r = copy.deepcopy(original)
                r['points'].pop('CAB_er')
                probe.data.write_json(out / 'manifest.json', r)
                with self.assertRaises(ValueError): probe.validated_scores(out, ctx)
                r = copy.deepcopy(original)
                r['train_ids']['ABC'].reverse()
                probe.data.write_json(out / 'manifest.json', r)
                with self.assertRaises(ValueError): probe.validated_scores(out, ctx)
                probe.data.write_json(out / 'manifest.json', original)
                (out / 'ABC_er_train.npy').write_bytes(b'corrupt')
                with self.assertRaises(ValueError): probe.validated_scores(out, ctx)

    def test_failed_gate_does_not_parse_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'evaluation'
            with patch.object(probe.platform, 'system', return_value='Windows'), \
                 patch.object(probe, 'context', return_value={}), \
                 patch.object(probe, 'validated_scores', side_effect=ValueError('gate')), \
                 patch.object(probe.pilot, 'attach_labels') as labels:
                with self.assertRaises(ValueError): probe.evaluate(Path(tmp), dest)
                labels.assert_not_called()
                self.assertFalse(dest.exists())

    def test_once_train_only_after_all_nine_and_exact_bce(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, ctx, manifest = fixture(root)
            label_calls = []
            def attach(groups, config, split):
                self.assertTrue((root / 'evaluation/access.json').exists())
                label_calls.append(split)
                return [SimpleNamespace(uid=g.uid, labels=tuple(hand_labels())) for g in reversed(groups)]
            with patch.object(probe.platform, 'system', return_value='Windows'), \
                 patch.object(probe.data, 'ROOT', root), \
                 patch.object(probe, 'sources', return_value=manifest['source_files']), \
                 patch.object(probe, 'context', return_value=ctx), \
                 patch.object(probe.pilot, 'attach_labels', side_effect=attach):
                r = probe.evaluate(out, root / 'evaluation')
                self.assertEqual(label_calls, ['train'])
                self.assertEqual(len(r['points']), 9)
                a = np.load(root / 'evaluation/ABC_er_metrics.npy', allow_pickle=False)
                np.testing.assert_allclose(a[:, -1], np.log(2), rtol=0, atol=1e-15)
                np.testing.assert_allclose(a[:, 0], 20/378, rtol=0, atol=1e-15)
                with self.assertRaises(ValueError): probe.evaluate(out, root / 'evaluation')
                self.assertEqual(label_calls, ['train'])

    def test_labels_align_by_uid_even_when_parser_order_is_reversed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, ctx, manifest = fixture(root)
            left, right = np.triu_indices(28, 1)
            adjacency = np.zeros((28, 28), dtype=np.uint8)
            adjacency[left, right] = hand_labels()
            adjacency[right, left] = hand_labels()
            truth = np.asarray([np.roll(np.roll(adjacency, i, axis=0), i, axis=1)[left, right]
                                for i in range(60)])
            for name in probe.NAMES:
                p = out / f'{name}_train.npy'
                np.save(p, truth.astype(np.float32) * 2 - 1, allow_pickle=False)
                manifest['points'][name]['scores']['train'] = probe.data.record(p, out)
            probe.data.write_json(out / 'manifest.json', manifest)
            def attach(groups, config, split):
                self.assertEqual(split, 'train')
                return [SimpleNamespace(uid=g.uid, labels=tuple(truth[int(g.uid[3:])]))
                        for g in reversed(groups)]
            with patch.object(probe.platform, 'system', return_value='Windows'), \
                 patch.object(probe.data, 'ROOT', root), \
                 patch.object(probe, 'sources', return_value=manifest['source_files']), \
                 patch.object(probe, 'context', return_value=ctx), \
                 patch.object(probe.pilot, 'attach_labels', side_effect=attach):
                r = probe.evaluate(out, root / 'evaluation')
                a = np.load(root / 'evaluation/ABC_er_metrics.npy', allow_pickle=False)
                np.testing.assert_array_equal(a[:, 0], np.ones(60))
                np.testing.assert_allclose(a[:, -1], np.log1p(np.exp(-1)), atol=1e-15, rtol=0)


if __name__ == '__main__':
    unittest.main()
