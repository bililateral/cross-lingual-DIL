"""Hand-created fixtures only; no formal supervision, weights or server access."""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import step28_continual_distillation as distill
from step28_continual_distillation import baseline, core, data, metrics, pilot
from test_step28_continual_replay_contracts import fixtures, labelled, fixture_result, write_result


def targets_for(memory, point='ABC_shared'):
    result = {}
    for i, g in enumerate(memory.groups):
        raw = np.full(378, i / 10, dtype='<f4').tobytes()
        result[g.uid] = {'float32_base64': base64.b64encode(raw).decode('ascii'),
            'sha256': hashlib.sha256(raw).hexdigest(), 'point': point, 'model_state_sha256': 'a' * 64,
            'pair_order_sha256': hashlib.sha256(data.json_bytes(g.sellers)).hexdigest()}
    return result


def state_fixture():
    groups, meta = fixtures()
    train = labelled(groups['train'])
    memory = data.Memory(data.seed_for(20260909, 'ABC', 'memory'))
    memory.add_stage(baseline.current_groups(train, meta, 'A'))
    return distill.ReplayState(memory, 'ABC', 2, targets_for(memory)), train, meta


def new_result(root):
    out, _, groups, meta, checked, old, old_scores, old_arrays, r, old_c = fixture_result(root)
    c = distill.contract()
    r.update(status=distill.COMPLETE, config=c, source_files=distill.sources(),
             torch_contracts={'passed': 3, 'skipped': 0}, teacher_group_forwards=0)
    for key in ('points', 'training', 'memory', 'models'):
        r[key] = {name.replace('_er_stage', '_distill_stage'): value for name, value in r[key].items()}
    for order in pilot.ORDERS:
        previous = {}
        for stage in (1, 2, 3):
            name = f'{order}_shared' if stage == 1 else f'{order}_distill_stage{stage}'
            p = r['points'][name]
            p['checkpoint']['path'] = f'work/{name}.pt'
            source = out / p['scores']['development']['path']
            target = out / 'scores' / f'{name}_development.npy'
            target.write_bytes(source.read_bytes())
            p['scores']['development'] = data.record(target, out)
            if stage > 1:
                r['training'][name].update(replay_mse_by_epoch=[.2] * 3,
                    distillation_coefficient=.5, targets_unchanged=True)
            if stage == 3:
                r['models'][name]['path'] = f'models/{name}.pt'
            else:
                m = r['memory'][name]
                path = out / 'memory' / f'{name}.json'
                path.write_bytes(b'opaque cache bytes; no Windows label decoding')
                m.update(file=data.record(path, out), full_state_bytes=path.stat().st_size)
                new = [uid for uid in m['retained_groups'] if uid in m['added_groups']]
                public = {g.uid: g for g in groups['train']}
                targets = {uid: ({'point': name, 'model_state_sha256': p['model_state_sha256'],
                    'pair_order_sha256': hashlib.sha256(data.json_bytes(public[uid].sellers)).hexdigest(),
                    'sha256': 'b' * 64} if uid in new else previous[uid]) for uid in m['retained_groups']}
                m['teacher'] = {'new_groups': new, 'group_forwards': len(new), 'seconds': 1,
                    'model_state_unchanged': True, 'maximum_insertion_history_bytes': 200000, 'targets': targets}
                previous = targets
                r['teacher_group_forwards'] += len(new)
    write_result(out, r)
    return out, c, groups, meta, checked, old, old_scores, old_arrays, r, old_c


class DistillationContracts(unittest.TestCase):
    def test_fixed_policy_and_reference(self):
        c = distill.contract()
        for key, value in [('coefficient', 1), ('refresh_survivors', True), ('maximum_target_group_forwards', 60)]:
            bad = copy.deepcopy(c)
            bad['distillation'][key] = value
            with patch.object(baseline, 'contract', return_value=baseline.contract()), patch.object(data, 'read_json', return_value=bad):
                with self.assertRaises(ValueError):
                    distill.contract()

    def test_target_bytes_order_and_actual_next_draw_roundtrip(self):
        state, _, meta = state_fixture()
        expected = baseline.expected_history(meta, 'ABC')[1]['replay_group_ids']
        for _ in range(73):
            state.draw()
        restored = distill.ReplayState.from_bytes(state.to_bytes())
        self.assertEqual([restored.draw().uid for _ in range(107)], expected[73:])
        for g in state.memory.groups:
            np.testing.assert_array_equal(state.target(g), restored.target(g))
        wrong = copy.deepcopy(state)
        first = wrong.memory.groups[0]
        wrong.targets[first.uid]['pair_order_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            distill.ReplayState.from_bytes(wrong.to_bytes())
        wrong = copy.deepcopy(state)
        del wrong.targets[first.uid]
        with self.assertRaises(ValueError):
            wrong.draw()

    def test_teacher_state_is_counted_and_rejects_corruption(self):
        state, _, _ = state_fixture()
        plain = len(baseline.ReplayState(state.memory, 'ABC', 2).to_bytes())
        self.assertGreater(len(state.to_bytes()) - plain, 9072)
        state.memory.maximum_bytes = plain + 500
        with self.assertRaises(ValueError):
            state.to_bytes()
        state, _, _ = state_fixture()
        row = state.targets[state.memory.groups[0].uid]
        raw = np.full(378, np.nan, dtype='<f4').tobytes()
        row.update(float32_base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError):
            state.validate_targets()

    def test_disk_restored_targets_are_the_supply(self):
        state, train, meta = state_fixture()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'memory').mkdir()
            restored, receipt = distill.save_memory(state, baseline.current_groups(train, meta, 'A'),
                'ABC_shared', out, SimpleNamespace(check=lambda *args: None))
            self.assertIsNot(restored, state)
            self.assertEqual(restored.to_bytes(), state.to_bytes())
            self.assertEqual(restored.draw().uid, receipt['next_draw'])

    def test_same_schedule_target_alignment_and_single_step(self):
        state, train, meta = state_fixture()
        current = baseline.current_groups(train, meta, 'B')
        optimizer = SimpleNamespace(state={'x': {'step': 180}})
        calls = []
        def update(model, opt, now, old, config, cs, rs, check, target, observe):
            calls.append((now.uid, old.uid, cs, rs, observe))
            np.testing.assert_array_equal(target, state.target(old))
            opt.state['x']['step'] += 1
            return {'current_bce': .1, 'replay_bce': .2, 'replay_mse': .3,
                'gradient_norm_before_clip': .1, 'encoder_and_head_changed': True,
                'branches_squared_hook_norm': {k: {'encoder': 1., 'head': 1.} for k in ('current', 'replay')}}
        with patch.object(distill, 'observed_update', side_effect=update), patch('builtins.print'):
            result = distill.train_replay(None, optimizer, current, state, pilot.contract(),
                SimpleNamespace(check=lambda: None, state=lambda: {}))
        rows, seed = pilot.segment_schedule(train, meta, pilot.contract(), 'ABC', 2)
        history = baseline.expected_history(meta, 'ABC')[1]['replay_group_ids']
        self.assertEqual(calls, [(g.uid, history[i], data.seed_for(seed, i, 'dropout_current'),
            data.seed_for(seed, i, 'dropout_replay'), i == 0) for i, g in enumerate(rows)])
        self.assertTrue(result['targets_unchanged'])
        np.testing.assert_allclose(result['replay_mse_by_epoch'], [.3] * 3)

    def test_gate_rejects_wrong_teacher_before_label_access(self):
        with tempfile.TemporaryDirectory() as temp:
            out, c, groups, meta, checked, old, scores, arrays, r, old_c = new_result(Path(temp))
            with patch.object(distill, 'origin', return_value=(old, scores, arrays)):
                distill.validated_scores(out, c, groups, meta, checked)
                for kind in ('future_teacher', 'refreshed', 'wrong_coefficient', 'extra_forward', 'wrong_pairs'):
                    bad = copy.deepcopy(r)
                    target = next(iter(bad['memory']['ABC_shared']['teacher']['targets'].values()))
                    if kind == 'future_teacher':
                        target['point'] = 'ABC_distill_stage2'
                    elif kind == 'wrong_pairs':
                        target['pair_order_sha256'] = '0' * 64
                    elif kind == 'wrong_coefficient':
                        bad['training']['ABC_distill_stage2']['distillation_coefficient'] = 0
                    elif kind == 'extra_forward':
                        bad['teacher_group_forwards'] += 1
                    else:
                        m = bad['memory']['ABC_distill_stage2']['teacher']
                        uid = next(uid for uid in m['targets'] if uid not in m['new_groups'])
                        m['targets'][uid]['sha256'] = '0' * 64
                    write_result(out, bad)
                    with self.subTest(kind=kind), patch.object(distill, 'contract', return_value=c), patch.object(pilot, 'contract', return_value=old_c), patch.object(pilot, 'public_inputs', return_value=(groups, meta, checked)), patch.object(pilot, 'attach_labels') as labels, patch.object(distill.platform, 'system', return_value='Windows'):
                        with self.assertRaises(ValueError):
                            distill.evaluate(out, Path(temp) / 'evaluation')
                        labels.assert_not_called()

    def test_evaluation_one_hand_parse_roles_and_incremental_contrast(self):
        with tempfile.TemporaryDirectory() as temp:
            out, c, groups, meta, checked, old, scores, arrays, r, old_c = new_result(Path(temp))
            reference = copy.deepcopy(arrays)
            for name in reference:
                if '_stage' in name:
                    reference[name] = reference[name] - .1
            attach = pilot.attach_labels
            with patch.object(distill, 'origin', return_value=(old, scores, arrays)), patch.object(distill, 'reference_baseline', return_value=reference), patch.object(distill, 'contract', return_value=c), patch.object(pilot, 'contract', return_value=old_c), patch.object(pilot, 'public_inputs', return_value=(groups, meta, checked)), patch.object(pilot, 'attach_labels', wraps=attach) as labels, patch.object(distill.platform, 'system', return_value='Windows'):
                result = distill.evaluate(out, Path(temp) / 'evaluation')
                self.assertEqual(labels.call_count, 1)
                ap = result['comparison']['metrics']['average_precision']
                self.assertAlmostEqual(ap['contrasts']['H']['mean'], 0)
                self.assertIn('G_distill', ap['contrasts'])
                self.assertFalse(result['comparison']['gate']['passed'])
                self.assertTrue(result['incremental_retention_passed'])
                self.assertAlmostEqual(result['versus_random_er']['average_precision']['contrasts']['N']['mean'], .1)
                self.assertIn('G_random_er', result['versus_random_er']['average_precision']['contrasts'])
                self.assertNotIn('G_seq', result['versus_random_er']['average_precision']['contrasts'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Local Torch absent; real CPU evidence required before formal labels')
class TorchContracts(unittest.TestCase):
    def test_hand_loss_gradient_and_detached_target(self):
        import torch
        x = torch.tensor([0., 2.], requires_grad=True)
        y, z = torch.tensor([0., 1.]), torch.tensor([-1., 0.])
        loss, bce, mse = distill.historical_loss(x, y, z, .5)
        self.assertAlmostEqual(float(mse), 2.5)
        self.assertAlmostEqual(float(loss), float(bce) + 1.25, places=6)
        loss.backward()
        expected = (torch.sigmoid(x.detach()) - y) / 2 + (x.detach() - z) / 2
        torch.testing.assert_close(x.grad, expected)
        with self.assertRaises(ValueError):
            distill.historical_loss(x, y, z.requires_grad_(), .5)

    def test_real_update_zero_equivalence_nonzero_reference_and_reload(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(171)
        initial = core.build_model(torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2)), 4, 5)
        current = SimpleNamespace(labels=(0, 1, 0, 1, 0, 0), offset=.1)
        history = SimpleNamespace(labels=(1, 0, 0, 0, 1, 0), offset=.7)
        targets = np.linspace(-1, 1, 6, dtype=np.float32)
        c = pilot.contract()
        def logits(m, g, config, check=lambda: None):
            return m.pair_logits(m.encoder(torch.arange(12).reshape(4, 3).float() / 10 + g.offset))
        for coefficient in (0., .5):
            model, reference, plain = [copy.deepcopy(initial) for _ in range(3)]
            opt, refopt, plainopt = [core.make_optimizer(m, c) for m in (model, reference, plain)]
            with patch.object(core, 'logits', side_effect=logits):
                for step in (1, 2):
                    reference.train()
                    refopt.zero_grad(set_to_none=True)
                    torch.manual_seed(19 + step)
                    a = torch.nn.functional.binary_cross_entropy_with_logits(logits(reference, current, c), torch.tensor(current.labels).float())
                    torch.manual_seed(39 + step)
                    predicted = logits(reference, history, c)
                    b = torch.nn.functional.binary_cross_entropy_with_logits(predicted, torch.tensor(history.labels).float())
                    term = a + b + coefficient * ((predicted - torch.tensor(targets)) ** 2).mean()
                    term.backward()
                    torch.nn.utils.clip_grad_norm_(reference.parameters(), c['optimizer']['clip_norm'], error_if_nonfinite=True)
                    refopt.step()
                    log = distill.observed_update(model, opt, current, history, c, 19 + step, 39 + step, lambda: None, targets, coefficient=coefficient)
                    if coefficient == 0:
                        core.update(plain, plainopt, current, history, c, 19 + step, 39 + step)
                        self.assertEqual(core.state_digest(model.state_dict()), core.state_digest(plain.state_dict()))
                        self.assertEqual(core.state_digest(opt.state_dict()), core.state_digest(plainopt.state_dict()))
                    for p, q in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)
                    for actual, expected in zip(opt.state.values(), refopt.state.values()):
                        for key in ('exp_avg', 'exp_avg_sq'):
                            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-8)
                    self.assertGreater(log['branches_squared_hook_norm']['replay']['encoder'], 0)
                    self.assertTrue(all(float(s['step']) == step for s in opt.state.values()))
                    if step == 1:
                        with tempfile.TemporaryDirectory() as temp:
                            path = Path(temp) / 'state.pt'
                            saved = core.save_state(path, model, opt, {'point': 'hand'})
                            model = copy.deepcopy(initial)
                            opt = core.make_optimizer(model, c)
                            core.restore_state(path, model, opt, saved['state_sha256'])

    def test_actual_teacher_scores_stage_eviction_rng_and_modes(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(33)
        model = core.build_model(torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2)), 4, 5)
        model.train()
        state, train, meta = state_fixture()
        memory = data.Memory(data.seed_for(20260909, 'ABC', 'memory'))
        state = distill.ReplayState(memory, 'ABC', 2)
        budget = SimpleNamespace(check=lambda *args: None)
        def logits(m, g, config, check=lambda: None):
            self.assertFalse(m.training)
            self.assertTrue(torch.is_inference_mode_enabled())
            return m.pair_logits(m.encoder(torch.arange(84).reshape(28, 3).float() / 84))
        with patch.object(core, 'logits', side_effect=logits):
            rng = torch.random.get_rng_state().clone()
            receipt = distill.populate_targets(model, state, baseline.current_groups(train, meta, 'A'), 'ABC_shared', pilot.contract(), budget)
            self.assertEqual(receipt['group_forwards'], 6)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            previous = copy.deepcopy(state.targets)
            with torch.no_grad():
                model.head[-1].bias.add_(.5)
            state = distill.ReplayState(state.memory, 'ABC', 3, state.targets)
            receipt = distill.populate_targets(model, state, baseline.current_groups(train, meta, 'B'), 'ABC_distill_stage2', pilot.contract(), budget)
            expected = baseline.expected_history(meta, 'ABC')[2]
            self.assertEqual([g.uid for g in state.memory.groups], expected['retained_groups'])
            self.assertEqual(set(state.targets), set(expected['retained_groups']))
            for uid in set(previous) & set(state.targets):
                self.assertEqual(state.targets[uid], previous[uid])
            for uid in receipt['new_groups']:
                self.assertEqual(state.targets[uid]['point'], 'ABC_distill_stage2')
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
