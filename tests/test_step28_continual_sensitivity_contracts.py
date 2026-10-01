"""Hand-generated fixtures only. No project supervision, models or server access."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import step28_continual_sensitivity as sens
from step28_continual_sensitivity import distill, core, data, pilot, persistence
from test_step28_continual_distillation_contracts import new_result, state_fixture
from test_step28_continual_replay_contracts import write_result


def fixture(root):
    out = root / 'stage'
    out.mkdir()
    c = sens.contract()
    stage = {'status': sens.COMPLETE, 'config': c, 'source_files': sens.sources(), 'arms': {},
        'physical_updates': 3240, 'formal_training_seconds': 18,
        'torch_contracts': {'passed': 2, 'skipped': 0},
        'label_parses': {'train': 1, 'development': 0, 'heldout': 0, 'owners': 0},
        'budget': {'elapsed_seconds': 10, 'peak_observed_bytes': 100}}
    for arm in sens.ARMS:
        values = new_result(root / ('fixture_' + arm))
        source, _, groups, meta, checked, old, scores, arrays, record, old_c = values
        target = out / arm
        shutil.copytree(source, target)
        record.update(status=sens.ARM_COMPLETE, config=c, source_files=sens.sources(),
            arm=arm, weights=sens.WEIGHTS[arm], torch_contracts=stage['torch_contracts'],
            label_parses={'train': 0, 'development': 0, 'heldout': 0, 'owners': 0},
            supervision_source='parent_stage_once_only_train_parse', formal_training_seconds=9)
        for name, t in record['training'].items():
            if '_distill_stage' in name:
                t.update(replay_coefficient=sens.WEIGHTS[arm]['alpha'],
                         distillation_coefficient=sens.WEIGHTS[arm]['beta'])
        write_result(target, record)
        stage['arms'][arm] = {'manifest': data.record(target / 'manifest.json', out),
                             'physical_updates': 1620, 'formal_training_seconds': 9}
    data.write_json(out / 'manifest.json', stage)
    data.write_json(out / 'access.json', {'train_parse_attempts': 1, 'development': 0, 'heldout': 0, 'owners': 0})
    return out, c, groups, meta, checked, old, scores, arrays, old_c


class SensitivityContracts(unittest.TestCase):
    def test_policy_rejects_extra_arm_weights_and_stage_budget(self):
        c, original = sens.contract(), distill.contract()
        for kind in ('alpha', 'beta', 'third_arm', 'budget', 'parses'):
            bad = copy.deepcopy(c)
            if kind == 'alpha': bad['configurations']['weaker_replay']['alpha'] = .25
            elif kind == 'beta': bad['configurations']['weaker_distillation']['beta'] = .5
            elif kind == 'third_arm': bad['configurations']['extra'] = {'alpha': .5, 'beta': .25}
            elif kind == 'budget': bad['runtime']['maximum_gpu_stage_seconds'] *= 2
            else: bad['label_parses']['train'] = 2
            with self.subTest(kind=kind), patch.object(distill, 'contract', side_effect=lambda: copy.deepcopy(original)), patch.object(data, 'read_json', return_value=bad):
                with self.assertRaises(ValueError): sens.contract()

    def test_training_passes_both_weights_to_every_update(self):
        for arm, weights in sens.WEIGHTS.items():
            state, train, meta = state_fixture()
            current = sens.current_groups(train, meta, 'B')
            opt = SimpleNamespace(state={'p': {'step': 180}})
            calls = []

            def update(model, optimizer, now, replay, config, cs, rs, check, target, *, observe, alpha, beta):
                calls.append((now.uid, replay.uid, cs, rs, observe, alpha, beta))
                np.testing.assert_array_equal(target, state.target(replay))
                optimizer.state['p']['step'] += 1
                return {'current_bce': .1, 'replay_bce': .2, 'replay_mse': .3,
                    'gradient_norm_before_clip': .4, 'encoder_and_head_changed': True,
                    'branches_squared_hook_norm': {b: {'encoder': 1., 'head': 1.} for b in ('current', 'replay')}}

            with patch.object(sens, 'observed_update', side_effect=update), patch('builtins.print'):
                r = sens.train_replay(None, opt, current, state, pilot.contract(),
                    SimpleNamespace(check=lambda: None, state=lambda: {}), weights)
            rows, seed = pilot.segment_schedule(train, meta, pilot.contract(), 'ABC', 2)
            draws = sens.expected_history(meta, 'ABC')[1]['replay_group_ids']
            expected = [(g.uid, draws[i], data.seed_for(seed, i, 'dropout_current'),
                         data.seed_for(seed, i, 'dropout_replay'), i == 0, weights['alpha'], weights['beta']) for i, g in enumerate(rows)]
            self.assertEqual(calls, expected)
            self.assertEqual(r['replay_coefficient'], weights['alpha'])
            self.assertEqual(r['distillation_coefficient'], weights['beta'])
            self.assertEqual(opt.state['p']['step'], 360)

    def test_both_arms_gate_rejects_weights_or_cross_arm_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            out, c, groups, meta, checked, old, scores, arrays, _ = fixture(Path(temp))
            good = data.read_json(out / 'manifest.json')
            with patch.object(sens, 'origin', return_value=(old, scores, arrays)):
                sens.validated_stage(out, c, groups, meta, checked)
                arm = sens.ARMS[1]
                original = data.read_json(out / arm / 'manifest.json')
                for kind in ('alpha', 'beta', 'wrong_arm', 'teacher'):
                    r = copy.deepcopy(original)
                    if kind == 'alpha': r['training']['ABC_distill_stage2']['replay_coefficient'] = .5
                    elif kind == 'beta': r['training']['ABC_distill_stage2']['distillation_coefficient'] = .5
                    elif kind == 'wrong_arm': r['arm'] = sens.ARMS[0]
                    else: next(iter(r['memory']['ABC_distill_stage2']['teacher']['targets'].values()))['model_state_sha256'] = 'wrong_branch'
                    write_result(out / arm, r)
                    parent = copy.deepcopy(good)
                    parent['arms'][arm]['manifest'] = data.record(out / arm / 'manifest.json', out)
                    data.write_json(out / 'manifest.json', parent)
                    with self.subTest(kind=kind), self.assertRaises(ValueError):
                        sens.validated_stage(out, c, groups, meta, checked)

    def test_partial_stage_stops_before_valid_parser(self):
        with tempfile.TemporaryDirectory() as temp:
            out, c, groups, meta, checked, old, scores, arrays, _ = fixture(Path(temp))
            record = data.read_json(out / 'manifest.json')
            del record['arms'][sens.ARMS[1]]
            data.write_json(out / 'manifest.json', record)
            with patch.object(sens.platform, 'system', return_value='Windows'), patch.object(pilot, 'public_inputs', return_value=(groups, meta, checked)), patch.object(pilot, 'attach_labels') as labels:
                with self.assertRaises(ValueError): sens.evaluate(out, Path(temp) / 'evaluation')
                labels.assert_not_called()

    def test_complete_evaluation_parses_hand_truth_once_for_both_arms(self):
        with tempfile.TemporaryDirectory() as temp:
            out, c, groups, meta, checked, old, scores, arrays, old_c = fixture(Path(temp))
            original_distill = copy.deepcopy(arrays)
            for name in original_distill:
                if '_stage' in name: original_distill[name] += .1
            attach = pilot.attach_labels
            with patch.object(sens.platform, 'system', return_value='Windows'), patch.object(pilot, 'contract', return_value=old_c), patch.object(pilot, 'public_inputs', return_value=(groups, meta, checked)), patch.object(sens, 'origin', return_value=(old, scores, arrays)), patch.object(distill, 'reference_baseline', return_value=arrays), patch.object(sens, 'reference_distillation', return_value=original_distill), patch.object(pilot, 'attach_labels', wraps=attach) as labels:
                result = sens.evaluate(out, Path(temp) / 'evaluation')
            self.assertEqual(labels.call_count, 1)
            self.assertEqual(set(result['arms']), set(sens.ARMS))
            for arm in sens.ARMS:
                comparisons = result['arms'][arm]['comparisons']
                self.assertIn('gate', comparisons['sequential'])
                self.assertNotIn('gate', comparisons['original_distillation'])
                self.assertAlmostEqual(comparisons['original_distillation']['metrics']['average_precision']['contrasts']['N']['mean'], -.1)
                self.assertEqual(len(result['arms'][arm]['points']), 9)

    def test_budget_counts_two_arm_directories_together(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for arm in sens.ARMS:
                (root / arm).mkdir()
                (root / arm / 'payload').write_bytes(b'x' * 60)
            c = {'runtime': {'maximum_gpu_stage_seconds': 10, 'maximum_output_bytes': 100}}
            budget = persistence.Budget(root, c)
            with self.assertRaisesRegex(RuntimeError, 'output budget'): budget.check(1)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Local Torch absent; real CPU gate before formal labels')
class TorchContracts(unittest.TestCase):
    def test_weighted_hand_gradient_and_separate_term_effects(self):
        import torch
        gradients = {}
        for alpha, beta in ((1., .5), (.5, .5), (1., .25)):
            x = torch.tensor([0., 2.], requires_grad=True)
            y, z = torch.tensor([0., 1.]), torch.tensor([-1., 0.])
            loss, bce, mse = sens.historical_loss(x, y, z, alpha, beta)
            self.assertAlmostEqual(float(loss), alpha * float(bce) + beta * 2.5, places=6)
            loss.backward()
            expected = alpha * (torch.sigmoid(x.detach()) - y) / 2 + beta * (x.detach() - z)
            torch.testing.assert_close(x.grad, expected)
            gradients[alpha, beta] = x.grad.clone()
        torch.testing.assert_close(gradients[1., .5] - gradients[.5, .5], .25 * (torch.sigmoid(torch.tensor([0., 2.])) - y))
        torch.testing.assert_close(gradients[1., .5] - gradients[1., .25], .25 * (torch.tensor([0., 2.]) - z))
        with self.assertRaises(ValueError): sens.historical_loss(x, y, z.requires_grad_(), 1., .25)

    def test_real_two_updates_adam_reload_and_original_equivalence(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(171)
        initial = core.build_model(torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2)), 4, 5)
        current = SimpleNamespace(labels=(0, 1, 0, 1, 0, 0), offset=.1)
        history = SimpleNamespace(labels=(1, 0, 0, 0, 1, 0), offset=.7)
        target = np.linspace(-1, 1, 6, dtype=np.float32)
        c = pilot.contract()

        def logits(m, g, config, check=lambda: None):
            return m.pair_logits(m.encoder(torch.arange(12).reshape(4, 3).float() / 10 + g.offset))

        for alpha, beta in ((1., .5), (.5, .5), (1., .25)):
            model, reference, original = [copy.deepcopy(initial) for _ in range(3)]
            opt, ropt, oopt = [core.make_optimizer(m, c) for m in (model, reference, original)]
            with patch.object(core, 'logits', side_effect=logits):
                for step in (1, 2):
                    reference.train(); ropt.zero_grad(set_to_none=True)
                    torch.manual_seed(19 + step)
                    now = torch.nn.functional.binary_cross_entropy_with_logits(logits(reference, current, c), torch.tensor(current.labels).float())
                    torch.manual_seed(39 + step)
                    pred = logits(reference, history, c)
                    hist = torch.nn.functional.binary_cross_entropy_with_logits(pred, torch.tensor(history.labels).float())
                    (now + alpha * hist + beta * (pred - torch.tensor(target)).square().mean()).backward()
                    torch.nn.utils.clip_grad_norm_(reference.parameters(), c['optimizer']['clip_norm'], error_if_nonfinite=True)
                    ropt.step()
                    log = sens.observed_update(model, opt, current, history, c, 19 + step, 39 + step, lambda: None, target, observe=(step == 1), alpha=alpha, beta=beta)
                    for actual, expected in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
                    for actual, expected in zip(opt.state.values(), ropt.state.values()):
                        self.assertEqual(float(actual['step']), step)
                        for key in ('exp_avg', 'exp_avg_sq'): torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-8)
                    if (alpha, beta) == (1., .5):
                        distill.observed_update(original, oopt, current, history, c, 19 + step, 39 + step, lambda: None, target, observe=(step == 1))
                        self.assertEqual(core.state_digest(model.state_dict()), core.state_digest(original.state_dict()))
                        self.assertEqual(core.state_digest(opt.state_dict()), core.state_digest(oopt.state_dict()))
                    if step == 2:
                        self.assertTrue(all(v == 0 for branch in log['branches_squared_hook_norm'].values() for v in branch.values()))
                    if step == 1:
                        with tempfile.TemporaryDirectory() as tmp:
                            p = Path(tmp) / 'state.pt'
                            record = core.save_state(p, model, opt, {'point': 'hand'})
                            model = copy.deepcopy(initial); opt = core.make_optimizer(model, c)
                            core.restore_state(p, model, opt, record['state_sha256'])


if __name__ == '__main__':
    unittest.main()
