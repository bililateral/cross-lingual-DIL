"""Hand fixtures only: bounded replay, paired metrics, fail-before-label gate.

One small actual Torch test is required on Linux before supervision. Local lack
of Torch is reported as a skip, never as evidence of actual training correctness.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import step28_continual_replay as replay
from step28_continual_replay import core, data, metrics, pilot
from test_step28_continual_expression_run_contracts import fixtures, hand_labels, synthetic_result


def labelled(groups: list) -> list:
    return [data.Group(g.uid, g.sellers, g.items, tuple(hand_labels())) for g in groups]


def write_result(out: Path, r: dict) -> None:
    data.write_json(out / 'manifest.json', r)
    data.write_json(out / 'model_verification.json', {
        'manifest_sha256': data.sha256(out / 'manifest.json'),
        'files': [{k: m[k] for k in ('path', 'bytes', 'sha256')} for _, m in sorted(r['models'].items())]})


def fixture_result(root: Path) -> tuple:
    old_root, old_c, groups, meta, checked = synthetic_result(root)
    old = data.read_json(old_root / 'manifest.json')
    old_scores, old_arrays = {}, {}
    truth = np.tile(hand_labels(), (60, 1))
    for name, p in old['points'].items():
        p['model_state_sha256'] = 'model_' + name
        x = np.load(old_root / p['scores']['development']['path'], allow_pickle=False)
        old_scores[name] = x
        rows = [metrics.classification(y, s) for y, s in zip(truth, x)]
        old_arrays[name] = np.column_stack(([[row[k] for k in metrics.CLASS_KEYS] for row in rows],
            metrics.retrieval(truth, x, 28)))
    c = replay.contract()
    out = root / 'new'
    for directory in ('scores', 'memory'):
        (out / directory).mkdir(parents=True)
    r = {'status': replay.COMPLETE, 'config': c, 'source_files': replay.sources(),
        'physical_updates': 1620, 'group_forwards_with_replay': 1080,
        'label_parses': {'train': 1, 'development': 0, 'heldout': 0, 'owners': 0},
        'torch_contracts': {'passed': 1, 'skipped': 0}, 'inputs': old['inputs'], 'pretrained': old['pretrained'],
        'points': {}, 'training': {}, 'memory': {}, 'models': {},
        'starts': {o: old['points']['initial']['checkpoint']['state_sha256'] for o in pilot.ORDERS},
        'budget': {'elapsed_seconds': 1, 'peak_observed_bytes': 1}, 'formal_training_seconds': 1}
    for order in pilot.ORDERS:
        r['starts'][order + '_continuation'] = old['points'][order + '_shared']['checkpoint']['state_sha256']
        history = replay.expected_history(meta, order)
        for stage in (1, 2, 3):
            old_name, _, prior = pilot.segment(order, stage)
            name = f'{order}_shared' if stage == 1 else f'{order}_er_stage{stage}'
            p = copy.deepcopy(old['points'][old_name])
            p['checkpoint']['path'] = f'work/{name}.pt'
            path = out / 'scores' / f'{name}_development.npy'
            np.save(path, old_scores[old_name], allow_pickle=False)
            p['scores']['development'] = data.record(path, out)
            r['points'][name] = p
            t = copy.deepcopy(old['training'][old_name])
            t['training_seconds'] = 1
            if stage > 1:
                t.update({'replay_group_ids': history[stage - 1]['replay_group_ids'],
                    'reservoir_unchanged': True, 'final_index': 180,
                    'current_pair_presentations': 68040, 'replay_pair_presentations': 68040,
                    'maximum_history_bytes': 200000, 'current_bce_by_epoch': [.1] * 3,
                    'replay_bce_by_epoch': [.2] * 3,
                    'first_update': {'encoder_and_head_changed': True, 'branches_squared_hook_norm': {
                        branch: {'encoder': 1., 'head': 1.} for branch in ('current', 'replay')}}})
            r['training'][name] = t
            if stage < 3:
                # Opaque fixture bytes: Windows gate MUST hash, not parse them.
                path = out / 'memory' / f'{name}.json'
                path.write_bytes(b'opaque cache fixture: never deserialize on Windows')
                h = history[stage]
                r['memory'][name] = {k: v for k, v in h.items() if k != 'replay_group_ids'}
                r['memory'][name].update({'file': data.record(path, out), 'next_draw': h['replay_group_ids'][0],
                    'disk_roundtrip_exact': True, 'full_state_bytes': path.stat().st_size})
            if stage == 3:
                r['models'][name] = {'path': f'models/{name}.pt', 'bytes': 1, 'sha256': 'fixture',
                    'actual_loaded_model_equals_replayed_state': True}
    write_result(out, r)
    return out, c, groups, meta, checked, old, old_scores, old_arrays, r, old_c


class ReplayContracts(unittest.TestCase):
    def test_approved_constants_fail_closed(self):
        c = replay.contract()
        for parent, key, value in [('memory', 'capacity', 7), ('memory', 'replay_coefficient', .5),
                ('evaluation', 'new_difference_floor', -.1), ('runtime', 'maximum_output_bytes', 1)]:
            wrong = copy.deepcopy(c)
            wrong[parent][key] = value
            with self.subTest(key=key), patch.object(data, 'read_json', return_value=wrong):
                with self.assertRaises(ValueError):
                    replay.contract()

    def test_algorithm_r_two_stages_and_next_draw_restore(self):
        groups, meta = fixtures()
        train = labelled(groups['train'])
        for order in pilot.ORDERS:
            mem = data.Memory(data.seed_for(20260909, order, 'memory'))
            expected = replay.expected_history(meta, order)
            for stage in (1, 2):
                current = replay.current_groups(train, meta, order[stage - 1])
                for group in current:
                    mem.add_stage([group])
                    replay.ReplayState(mem, order, stage + 1).to_bytes()
                state = replay.ReplayState(mem, order, stage + 1)
                self.assertEqual([g.uid for g in mem.groups], expected[stage]['retained_groups'])
                self.assertEqual(mem.seen, 60 * stage)
                before = mem.to_bytes()
                ids = [state.draw().uid for _ in range(73)]
                restored = replay.ReplayState.from_bytes(state.to_bytes())
                actual_next = restored.draw()
                self.assertEqual(state.draw(), actual_next)
                ids.append(actual_next.uid)
                ids += [restored.draw().uid for _ in range(106)]
                self.assertEqual(ids, expected[stage]['replay_group_ids'])
                self.assertEqual(before, mem.to_bytes())
                with self.assertRaises(ValueError):
                    restored.draw()

    def test_envelope_byte_cap_counts_rng_beyond_memory_body(self):
        groups, meta = fixtures()
        mem = data.Memory(1)
        mem.add_stage(labelled(groups['train'][:60]))
        state = replay.ReplayState(mem, 'ABC', 2)
        n = len(mem.to_bytes())
        self.assertGreater(len(state.to_bytes()), n + 5000)
        mem.maximum_bytes = n + 100
        self.assertLess(len(mem.to_bytes()), mem.maximum_bytes)
        with self.assertRaisesRegex(ValueError, 'including replay RNG'):
            state.to_bytes()

    def test_saved_memory_is_actual_source_and_disk_roundtrips(self):
        groups, meta = fixtures()
        current = labelled(groups['train'][:60])
        memory = data.Memory(data.seed_for(20260909, 'ABC', 'memory'))
        memory.add_stage(current)
        state = replay.ReplayState(memory, 'ABC', 2)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'memory').mkdir()
            budget = SimpleNamespace(check=lambda reserve=0: None)
            restored, receipt = replay.save_memory(state, current, 'ABC_shared', out, budget)
            self.assertIsNot(restored.memory, memory)
            self.assertEqual(restored.to_bytes(), state.to_bytes())
            self.assertEqual(receipt['next_draw'], restored.draw().uid)
            data.verify(out / receipt['file']['path'], receipt['file'])

    def test_train_uses_only_current_and_cache_with_paired_seeds(self):
        groups, meta = fixtures()
        train = labelled(groups['train'])
        mem = data.Memory(data.seed_for(20260909, 'ABC', 'memory'))
        mem.add_stage(replay.current_groups(train, meta, 'A'))
        state = replay.ReplayState(mem, 'ABC', 2)
        current = replay.current_groups(train, meta, 'B')
        opt = SimpleNamespace(state={'x': {'step': 180}})
        calls = []

        def update(model, optimizer, now, old, config, cs, rs, check):
            optimizer.state['x']['step'] += 1
            calls.append((now.uid, old.uid, cs, rs))
            return {'current_bce': .1, 'replay_bce': .2, 'gradient_norm_before_clip': 1.,
                'branches_squared_hook_norm': {k: {'encoder': 1., 'head': 1.} for k in ('current', 'replay')},
                'encoder_and_head_changed': True}

        with patch.object(replay, 'observed_update', side_effect=update), patch.object(core, 'update', side_effect=update), patch('builtins.print'):
            log = replay.train_replay(None, opt, current, state, pilot.contract(),
                SimpleNamespace(check=lambda: None, state=lambda: {}))
        rows, seed = pilot.segment_schedule(groups['train'], meta, pilot.contract(), 'ABC', 2)
        ids = replay.expected_history(meta, 'ABC')[1]['replay_group_ids']
        self.assertEqual(calls, [(g.uid, ids[i], data.seed_for(seed, i, 'dropout_current'),
            data.seed_for(seed, i, 'dropout_replay')) for i, g in enumerate(rows)])
        self.assertEqual(log['last_adam_step'], 360)
        self.assertEqual(log['current_pair_presentations'] + log['replay_pair_presentations'], 136080)
        with self.assertRaises(ValueError):
            replay.train_replay(None, opt, current, state, pilot.contract(), None)

    def test_new_capability_uses_absolute_after_not_different_start_gain(self):
        old = {n: np.zeros((60, len(metrics.COLUMNS))) for n in pilot.POINTS}
        new = {n: np.zeros_like(old['initial']) for n in replay.NAMES}
        for o in pilot.ORDERS:
            old[o + '_shared'][:] = .2
            new[o + '_shared'][:] = .2
            old[o + '_stage2'][:] = .4
            old[o + '_stage3'][:] = .3
            new[o + '_er_stage2'][:] = .4
            new[o + '_er_stage3'][:] = .4
        draws = np.zeros((5000, 3, 20), dtype=int)
        v = replay.compare(new, old, draws, replay.contract())['metrics']['average_precision']
        self.assertAlmostEqual(v['contrasts']['H']['mean'], .1)
        self.assertAlmostEqual(v['contrasts']['N']['mean'], .05)
        self.assertAlmostEqual(v['contrasts']['G_er']['mean'], .1)
        # Increase ER stage2 score on stage3's future domain, keeping post scores.
        for o in pilot.ORDERS:
            d = 'ABC'.index(o[2])
            new[o + '_er_stage2'][20*d:20*(d+1)] += .15
        changed = replay.compare(new, old, draws, replay.contract())['metrics']['average_precision']
        self.assertEqual(changed['contrasts']['N'], v['contrasts']['N'])
        self.assertLess(changed['contrasts']['G_er']['mean'], v['contrasts']['G_er']['mean'])

    def test_bootstrap_reuses_underlying_domain_across_transitions_orders(self):
        rng = np.random.default_rng(147)
        old = {n: rng.normal(size=(60, len(metrics.COLUMNS))) for n in pilot.POINTS}
        new = {n: rng.normal(size=(60, len(metrics.COLUMNS))) for n in replay.NAMES}
        draws = rng.integers(0, 20, (5000, 3, 20))
        result = replay.compare(new, old, draws, replay.contract())['metrics']['average_precision']['contrasts']['N']
        col = metrics.COLUMNS.index('average_precision')
        samples = []
        for b in range(5000):
            transition_means = []
            for o in pilot.ORDERS:
                for s in (2, 3):
                    d = 'ABC'.index(o[s - 1])
                    idx = 20*d + draws[b, d]
                    transition_means.append((new[f'{o}_er_stage{s}'][idx, col] - old[f'{o}_stage{s}'][idx, col]).mean())
            samples.append(np.mean(transition_means))
        np.testing.assert_allclose(result['conditional_95pct_interval'], np.quantile(samples, [.025, .975]), atol=1e-15)

    def test_same_orders_gate_cannot_mix_different_orders(self):
        old = {n: np.full((60, len(metrics.COLUMNS)), .3) for n in pilot.POINTS}
        new = {n: np.full_like(old['initial'], .3) for n in replay.NAMES}
        for i, o in enumerate(pilot.ORDERS):
            # ABC/BCA keep old better but learn nothing; CAB learns new only.
            d = 'ABC'.index(o[0])
            new[f'{o}_er_stage3'][20*d:20*(d+1)] += .03 if i < 2 else -.01
            if i == 2:
                for s in (2, 3):
                    d = 'ABC'.index(o[s - 1])
                    new[f'{o}_er_stage{s}'][20*d:20*(d+1)] += .3
        gate = replay.compare(new, old, np.zeros((5000, 3, 20), dtype=int), replay.contract())['gate']
        self.assertTrue(gate['checks']['retention'])
        self.assertTrue(gate['checks']['new_capability'])
        self.assertFalse(gate['passed'])
        self.assertEqual(gate['same_orders'], [])

    def test_complete_gate_hashes_cache_without_deserializing_and_rejects_mutations(self):
        with tempfile.TemporaryDirectory() as temp:
            out, c, groups, meta, checked, old, old_scores, old_arrays, r, old_c = fixture_result(Path(temp))
            with patch.object(replay, 'origin', return_value=(old, old_scores, old_arrays)):
                scores, _, _ = replay.validated_scores(out, c, groups, meta, checked)
                self.assertEqual(set(scores), set(replay.NAMES))
                for mutation in ('missing_memory', 'wrong_history', 'reset_adam', 'wrong_start', 'missing_branch', 'oversize'):
                    wrong = copy.deepcopy(r)
                    if mutation == 'missing_memory':
                        del wrong['memory']['ABC_shared']
                    elif mutation == 'wrong_history':
                        wrong['training']['ABC_er_stage2']['replay_group_ids'][0] = 'future_group'
                    elif mutation == 'reset_adam':
                        wrong['training']['ABC_er_stage2']['first_adam_step'] = 1
                    elif mutation == 'wrong_start':
                        wrong['starts']['ABC_continuation'] = 'inference_only'
                    elif mutation == 'missing_branch':
                        wrong['training']['ABC_er_stage2']['first_update']['branches_squared_hook_norm']['replay']['encoder'] = 0
                    else:
                        wrong['training']['ABC_er_stage2']['maximum_history_bytes'] = 1048577
                    write_result(out, wrong)
                    with self.subTest(mutation=mutation), patch.object(replay, 'contract', return_value=c), patch.object(pilot, 'contract', return_value=old_c), patch.object(pilot, 'public_inputs', return_value=(groups, meta, checked)), patch.object(pilot, 'attach_labels') as labels, patch.object(replay.platform, 'system', return_value='Windows'):
                        with self.assertRaises(ValueError):
                            replay.evaluate(out, Path(temp) / 'evaluation')
                        labels.assert_not_called()

    def test_evaluation_hand_labels_once_and_shared_metrics_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out, c, groups, meta, checked, old, old_scores, old_arrays, r, old_c = fixture_result(root)
            original_attach = pilot.attach_labels
            with patch.object(replay, 'origin', return_value=(old, old_scores, old_arrays)), patch.object(replay, 'contract', return_value=c), patch.object(pilot, 'contract', return_value=old_c), patch.object(pilot, 'public_inputs', return_value=(groups, meta, checked)), patch.object(pilot, 'attach_labels', wraps=original_attach) as labels, patch.object(replay.platform, 'system', return_value='Windows'):
                result = replay.evaluate(out, root / 'evaluation')
                self.assertEqual(labels.call_count, 1)
                self.assertEqual(labels.call_args.args[2], 'development')
                self.assertEqual(result['comparison']['metrics']['average_precision']['contrasts']['H']['mean'], 0.)
                self.assertFalse(result['comparison']['gate']['passed'])
                for o in pilot.ORDERS:
                    np.testing.assert_array_equal(np.load(root / 'evaluation' / f'{o}_shared_metrics.npy'), old_arrays[f'{o}_shared'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Local Python has no Torch; required Linux CPU fixture remains pending')
class TorchContracts(unittest.TestCase):
    def test_observed_update_matches_independent_sum_and_full_adam_continuation(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(171)
        model = core.build_model(torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2)), 4, 5)
        reference = copy.deepcopy(model)
        unchanged = copy.deepcopy(model)
        c = pilot.contract()
        opts = [core.make_optimizer(m, c) for m in (model, reference, unchanged)]
        current = SimpleNamespace(labels=(0, 1, 0, 1, 0, 0), offset=.1)
        history = SimpleNamespace(labels=(1, 0, 0, 0, 1, 0), offset=.7)

        def logits(m, g, config, check=lambda: None):
            return m.pair_logits(m.encoder(torch.arange(12).reshape(4, 3).float() / 10 + g.offset))

        with patch.object(core, 'logits', side_effect=logits):
            for step in (1, 2):
                reference.train()
                opts[1].zero_grad(set_to_none=True)
                torch.manual_seed(19 + step)
                a = torch.nn.functional.binary_cross_entropy_with_logits(logits(reference, current, c), torch.tensor(current.labels).float())
                torch.manual_seed(39 + step)
                b = torch.nn.functional.binary_cross_entropy_with_logits(logits(reference, history, c), torch.tensor(history.labels).float())
                (a + b).backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), c['optimizer']['clip_norm'], error_if_nonfinite=True)
                opts[1].step()
                log = replay.observed_update(model, opts[0], current, history, c, 19 + step, 39 + step, lambda: None)
                core.update(unchanged, opts[2], current, history, c, 19 + step, 39 + step)
                self.assertEqual(core.state_digest(model.state_dict()), core.state_digest(unchanged.state_dict()))
                self.assertEqual(core.state_digest(opts[0].state_dict()), core.state_digest(opts[2].state_dict()))
                for p, q in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)
                self.assertTrue(all(float(s['step']) == step for s in opts[0].state.values()))
                self.assertGreater(log['branches_squared_hook_norm']['replay']['encoder'], 0.)
                if step == 1:
                    with tempfile.TemporaryDirectory() as temp:
                        path = Path(temp) / 'state.pt'
                        saved = core.save_state(path, model, opts[0], {'point': 'hand_fixture'})
                        fresh = copy.deepcopy(model)
                        fresh_opt = core.make_optimizer(fresh, c)
                        core.restore_state(path, fresh, fresh_opt, saved['state_sha256'])
                        model, opts[0] = fresh, fresh_opt


if __name__ == '__main__':
    unittest.main()
