"""Joint replay, continuous schedule and label gates using hand-made fixtures.

No project supervision, model loading or CUDA is used. Runtime training evidence
comes from the inherited unchanged core and this stage's mandatory exact replay.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import step28_continual_joint as joint
from step28_continual_joint import data, metrics, pilot
from test_step28_continual_expression_run_contracts import fixtures, hand_labels, synthetic_result


def fixture_result(root: Path) -> tuple:
    old_root, old_c, groups, meta, checked = synthetic_result(root)
    old = data.read_json(old_root / 'manifest.json')
    old['pretrained'] = {'fixture': 'pretrained'}
    old['config'] = old_c
    old['inputs'] = {'fixture': 'same_public_inputs'}
    old_scores = {}
    for name, p in old['points'].items():
        p['model_state_sha256'] = 'model_' + name
        old_scores[name] = np.load(old_root / p['scores']['development']['path'], allow_pickle=False)
    for name, m in old['models'].items():
        m['state_sha256'] = 'inference_' + name
    out = root / 'new'
    (out / 'scores').mkdir(parents=True)
    c = joint.contract()
    c['origin_evaluation'] = str(root / 'old_evaluation.json')
    r = {'status': joint.COMPLETE, 'config': c, 'source_files': joint.sources(),
         'physical_updates': 1080, 'training': {}, 'points': {}, 'train_metrics': {},
         'models': {'joint_six': {'path': 'models/joint_six.pt', 'bytes': 1, 'sha256': 'fixture',
                                'actual_loaded_model_equals_replayed_state': True}},
         'label_parses': {'train': 1, 'development': 0, 'heldout': 0, 'owners': 0},
         'original_replays': dict.fromkeys((n for n in joint.NAMES if n != 'joint_six'), True),
         'initial_state_sha256': old['points']['initial']['checkpoint']['state_sha256'],
         'inputs': old['inputs'], 'pretrained': old['pretrained'],
         'scored_model_groups': 840, 'group_forwards_with_replay': 1680,
         'budget': {'elapsed_seconds': 1, 'peak_observed_bytes': 1},
         'original_model_files': {n: {'path': (Path(c['origin_run']) / m['path']).as_posix(),
                                    'bytes': m['bytes'], 'sha256': m['sha256']}
                                  for n, m in old['models'].items() if n in joint.NAMES}}
    schedule, seed = joint.joint_schedule(groups['train'], meta, old['config'])
    for name, start in (('joint', 0), ('joint_six', 540)):
        r['training'][name] = joint.training_record([g.uid for g in schedule[start:start+540]], seed, start,
            [.1]*540, [1.]*540, {k: {'finite_nonzero_gradient': True, 'parameters_changed': True}
                                for k in ('encoder', 'head')}, 1.)
    for name in joint.NAMES:
        subset = joint.train_groups(name, groups['train'], meta)
        state = ('fixture_joint' if name == 'joint' else 'inference_' + name)
        p = {'checkpoint': {'state_sha256': state}, 'model_state_sha256': 'model_' + name,
             'full_model_and_adam_reloaded': True, 'scores': {}}
        for split, count in (('train', len(subset)), ('development', 60)):
            a = (old_scores[name] if split == 'development' and name in old_scores
                 else np.zeros((count, 378), dtype=np.float32))
            path = out / 'scores' / f'{name}_{split}.npy'
            np.save(path, a, allow_pickle=False)
            p['scores'][split] = data.record(path, out)
        r['points'][name] = p
        path = out / 'scores' / f'{name}_train_metrics.npy'
        np.save(path, np.zeros((len(subset), 2), dtype=np.float64), allow_pickle=False)
        r['train_metrics'][name] = {'columns': list(joint.TRAIN_COLUMNS),
            'group_ids': [g.uid for g in subset], 'file': data.record(path, out)}
    write_result(out, r)
    old_eval = {'points': {}}
    truth = np.tile(hand_labels(), (60, 1))
    for name in joint.NAMES:
        if name == 'joint_six':
            continue
        score = old_scores[name]
        rows = [metrics.classification(y, x) for y, x in zip(truth, score)]
        matrix = np.column_stack((np.asarray([[r[k] for k in metrics.CLASS_KEYS] for r in rows]),
                                  metrics.retrieval(truth, score, 28)))
        path = root / f'old_{name}_metrics.npy'
        np.save(path, matrix, allow_pickle=False)
        old_eval['points'][name] = {'file': data.record(path, root)}
    data.write_json(Path(c['origin_evaluation']), old_eval)
    return out, c, groups, meta, checked, old, old_scores, old_eval, r


def write_result(out: Path, r: dict) -> None:
    data.write_json(out / 'manifest.json', r)
    m = r['models']['joint_six']
    data.write_json(out / 'model_verification.json', {
        'manifest_sha256': data.sha256(out / 'manifest.json'),
        'files': [{k: m[k] for k in ('path', 'bytes', 'sha256')}]})


class JointContracts(unittest.TestCase):
    def test_schedule_matches_independent_six_epoch_shuffle(self):
        groups, meta = fixtures()
        c = pilot.contract()
        rows, seed = joint.joint_schedule(groups['train'], meta, c)
        rng = random.Random(seed)
        expected = []
        for _ in range(6):
            epoch = sorted(groups['train'], key=lambda g: g.uid)
            rng.shuffle(epoch)
            expected.extend(g.uid for g in epoch)
        self.assertEqual([g.uid for g in rows], expected)
        self.assertEqual([g.uid for g in rows[:540]], [g.uid for g in pilot.segment_schedule(groups['train'], meta, c, 'JOINT', 1)[0]])
        self.assertNotEqual(expected[:540], expected[540:])

    def test_static_risk_is_stable_bce_and_step_ap_not_pr_area(self):
        groups, _ = fixtures()
        g = groups['train'][0]
        y = tuple(hand_labels())
        labelled = data.Group(g.uid, g.sellers, g.items, y)
        scores = np.zeros((1, 378), dtype=np.float32)
        result = joint.static_train_metrics([labelled], scores)
        self.assertAlmostEqual(result[0, 0], np.log(2))
        self.assertAlmostEqual(result[0, 1], 20/378)
        bad = np.where(np.asarray(y), -1000., 1000.).astype(np.float32)[None, :]
        self.assertEqual(joint.static_train_metrics([labelled], bad)[0, 0], 1000.)
        with self.assertRaises(ValueError):
            joint.static_train_metrics([g], scores)

    def test_subset_keeps_original_domain_order(self):
        groups, meta = fixtures()
        counts = []
        for n in joint.NAMES:
            subset = joint.train_groups(n, groups['train'], meta)
            counts.append(len(subset) + 60)
            if n not in joint.NAMES[:2]:
                self.assertTrue(all('_' + n[0] + '_' in g.uid for g in subset))
        self.assertEqual(sum(counts), 840)

    def test_joint_replay_requires_adam_and_full_scores(self):
        old = {'points': {'joint': {'checkpoint': {'state_sha256': 'full'}, 'model_state_sha256': 'model'}}}
        p = copy.deepcopy(old['points']['joint'])
        a = np.zeros((60, 378), dtype=np.float32)
        joint.check_original_point('joint', p, old, a, a)
        p['checkpoint']['state_sha256'] = 'same_model_wrong_adam'
        with self.assertRaises(ValueError):
            joint.check_original_point('joint', p, old, a, a)
        p['checkpoint']['state_sha256'] = 'full'
        b = a.copy()
        b[-1, -1] = 1
        with self.assertRaises(ValueError):
            joint.check_original_point('joint', p, old, b, a)

    def test_domain_matching_and_equal_weight_contrasts(self):
        arrays = {n: np.zeros((60, len(metrics.COLUMNS))) for n in joint.NAMES}
        for i, name in enumerate(joint.NAMES[2:]):
            arrays[name][:] = -100
            arrays[name][i*20:(i+1)*20] = i + 1
        arrays['joint'][:] = 2
        arrays['joint_six'][:] = 3
        draws = np.tile(np.arange(20), (2, 3, 1))
        r = joint.compare(arrays, draws)['average_precision']
        self.assertEqual(r['six_minus_single']['by_domain_or_first_domain'], [2, 1, 0])
        self.assertEqual(r['six_minus_three']['conditional_95pct_interval'], [1, 1])

    def test_real_complete_gate_rejects_changed_state_scores_mapping_and_schedule(self):
        with tempfile.TemporaryDirectory(prefix='joint_contract_') as tmp:
            out, c, groups, meta, checked, old, old_scores, old_eval, baseline = fixture_result(Path(tmp))
            with patch.object(joint, 'origin', return_value=(old, old_scores, old_eval)):
                scores, risks, _ = joint.validated_scores(out, c, groups, meta, checked)
                self.assertEqual(set(scores), set(joint.NAMES))
                self.assertEqual(risks['joint_six'].shape, (180, 2))
                mutations = [
                    lambda r: r['training']['joint_six'].update(first_adam_step=1),
                    lambda r: r['training']['joint_six'].update(dropout_index_start=0),
                    lambda r: r['training']['joint_six']['group_ids'].reverse(),
                    lambda r: r['points']['joint']['checkpoint'].update(state_sha256='wrong_adam'),
                    lambda r: r['points']['ABC_shared']['scores']['development'].update(path='scores/BCA_shared_development.npy'),
                    lambda r: r['train_metrics']['BCA_shared']['group_ids'].reverse(),
                    lambda r: r['models']['joint_six'].update(actual_loaded_model_equals_replayed_state=False),
                    lambda r: r['label_parses'].update(heldout=1),
                    lambda r: r['budget'].update(peak_observed_bytes=13*1024**3),
                ]
                for mutation in mutations:
                    r = copy.deepcopy(baseline)
                    mutation(r)
                    write_result(out, r)
                    with self.assertRaises(ValueError):
                        joint.validated_scores(out, c, groups, meta, checked)
                write_result(out, baseline)
                (out / 'failure.json').write_text('{}', encoding='utf-8')
                with self.assertRaises(ValueError):
                    joint.validated_scores(out, c, groups, meta, checked)

    def test_actual_toy_evaluation_reads_valid_once_and_matches_independent_ap(self):
        from sklearn.metrics import average_precision_score
        with tempfile.TemporaryDirectory(prefix='joint_contract_') as tmp:
            out, c, groups, meta, checked, old, old_scores, old_eval, _ = fixture_result(Path(tmp))
            target = Path(tmp) / 'evaluated'
            with patch.object(joint, 'contract', return_value=c), \
                 patch.object(pilot, 'contract', return_value=old['config']), \
                 patch.object(pilot, 'public_inputs', return_value=(groups, meta, checked)), \
                 patch.object(joint, 'origin', return_value=(old, old_scores, old_eval)), \
                 patch.object(joint.platform, 'system', return_value='Windows'), \
                 patch.object(pilot, 'attach_labels', wraps=pilot.attach_labels) as parser:
                result = joint.evaluate(out, target)
                self.assertEqual(parser.call_count, 1)
                self.assertEqual(parser.call_args.args[2], 'development')
                with self.assertRaises(ValueError):
                    joint.evaluate(out, target)
                self.assertEqual(parser.call_count, 1)
            self.assertEqual(len(result['columns']), 22)
            for n in joint.NAMES:
                m = np.load(target / f'{n}_metrics.npy', allow_pickle=False)
                x = np.load(out / 'scores' / f'{n}_development.npy', allow_pickle=False)
                expected = [average_precision_score(hand_labels(), row) for row in x]
                np.testing.assert_allclose(m[:, 0], expected, atol=1e-15, rtol=0)

    def test_training_window_uses_global_dropout_and_continuous_adam(self):
        # This is orchestration evidence, not actual Torch/LaBSE training.
        from types import SimpleNamespace
        class Parameter:
            grad = None
        class Gradient:
            def detach(self): return self
            def float(self): return self
            def norm(self): return 1.
        class Module:
            def __init__(self): self.value, self.parameter = 0, Parameter()
            def state_dict(self): return {'value': self.value}
            def parameters(self): return [self.parameter]
        groups, meta = fixtures()
        old_c = pilot.contract()
        rows, seed = joint.joint_schedule(groups['train'], meta, old_c)
        model = SimpleNamespace(encoder=Module(), head=Module())
        optimizer = SimpleNamespace(state={0: {'step': 540}})
        budget = SimpleNamespace(check=lambda: None, state=lambda: {})
        observed = []
        def update(m, o, current, replay, config, current_seed, replay_seed, check):
            observed.append((current.uid, current_seed))
            o.state[0]['step'] += 1
            for module in (m.encoder, m.head):
                module.value += 1
                module.parameter.grad = Gradient()
            return {'current_bce': .1, 'gradient_norm_before_clip': 1.}
        with patch.object(joint.core, 'update', side_effect=update), \
             patch.object(joint.core, 'state_digest', side_effect=lambda x: str(x)), \
             patch('builtins.print'):
            r = joint.train_window(model, optimizer, rows, seed, 540, old_c, budget)
        self.assertEqual(observed, [(rows[i].uid, data.seed_for(seed, i, 'dropout_current')) for i in range(540, 1080)])
        self.assertEqual(r['first_adam_step'], 541)
        self.assertEqual(optimizer.state[0]['step'], 1080)

    def test_valid_parser_is_unreachable_when_complete_gate_fails(self):
        with tempfile.TemporaryDirectory(prefix='joint_contract_') as tmp:
            groups, meta = fixtures()
            with patch.object(joint.platform, 'system', return_value='Windows'), \
                 patch.object(pilot, 'public_inputs', return_value=(groups, meta, {})), \
                 patch.object(joint, 'validated_scores', side_effect=ValueError('incomplete')), \
                 patch.object(pilot, 'attach_labels') as labels:
                with self.assertRaises(ValueError):
                    joint.evaluate(Path(tmp) / 'run', Path(tmp) / 'evaluation')
                labels.assert_not_called()
                self.assertFalse((Path(tmp) / 'evaluation').exists())


if __name__ == '__main__':
    unittest.main()
