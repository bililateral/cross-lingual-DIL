"""One-shot CPU checks of the P-to-M amendment; no formal data or model loads."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import step35_vendorlink_account_control as control
from step35_vendorlink_account_evaluate import application_completion
from step35_vendorlink_account_io import record, write
from step35_vendorlink_account_policy import CAPS, LIMITS, DOMAINS, SEEDS, tasks, summarize, decide
from step30_evolution_io import Ledger
from step29_vendorlink_data import group_chunks

SELECTED = dict(H='H_CE_RMS', V='V1')


def baseline():
    h = summarize(dict(R01=.1, R11=.82, R12=.84, R21=.81, R22=.881,
                       R23=.80, R31=.81, R32=.871, R33=.89))
    v = summarize(dict(R01=.1, R11=.81, R12=.80, R21=.80, R22=.87,
                       R23=.75, R31=.78, R32=.85, R33=.88))
    return {m: [dict(seed=s, **row) for s in SEEDS]
            for m, row in [('H_SEQ', h), ('V_CLS', v), ('V_SET', v)]}


class Checks(unittest.TestCase):
    def test_four_point_and_actual_pipeline(self):
        p = baseline()
        self.assertTrue(decide(p)['checks']['learning'])
        self.assertFalse(decide(p)['checks']['meaningful_forgetting'])
        below = copy.deepcopy(p)
        for row in below['H_SEQ']:
            row['GB'] = .039999
        self.assertFalse(decide(below)['checks']['learning'])
        m = {name: copy.deepcopy(p['H_SEQ']) for name in ['H_ER', 'H_full']}
        for row in m['H_full']:
            row['R31'] += .035
            row['R32'] += .035
            row.update(summarize({k: v for k, v in row.items() if k.startswith('R')}))
        calls, saved = [], {}
        def run(phase, selected):
            calls.append((phase, selected))
            return m
        answer = control.pipeline(run, saved.__setitem__, p)
        self.assertEqual(calls, [('M', SELECTED)])
        self.assertEqual(answer['status'], 'EXPLORATORY_M_COMPLETE')
        self.assertFalse(answer['confirmation'])
        self.assertTrue(answer['application'])
        self.assertTrue(answer['M']['checks']['retention'])
        self.assertTrue(answer['M']['old_score_increment_conditions'])
        self.assertFalse(answer['M']['old_score_increment'])
        self.assertEqual(saved['comparison_results']['H_SEQ'], p['H_SEQ'])
        m['H_full'].pop()
        with self.assertRaises(ValueError):
            control.pipeline(run, lambda *a: None, p)

    def test_prior_identity_and_cumulative_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = dict(experiment_root=str(root), campaign=str(root/'old_campaign'),
                       manifest={}, model_binding={}, caps=CAPS, limits=LIMITS)
            write(root/'old_gate.json', old)
            prior_gate = record(root/'old_gate.json')
            decision = dict(status='STOP_LEARNING', checks=dict(learning=False, meaningful_forgetting=False))
            original = dict(gate=old, batch=dict(tasks=tasks('P', SELECTED), gate=prior_gate),
                            decision=decision, completion=dict(status='STOP_LEARNING', P=decision),
                            budget=dict(used_seconds=dict(training=66942, gpu=1305, auxiliary=1198)),
                            results=baseline())
            bindings = {}
            for key, value in original.items():
                path = root/(key+'.json')
                if key == 'gate':
                    path = root/'old_gate.json'
                else:
                    write(path, value)
                bindings[key] = record(path)
            g = dict(continuation_after_p_stop=True, prior_P=bindings,
                     experiment_root=str(root), campaign=str(root/'mechanism/campaign'),
                     prior_campaign=old['campaign'], manifest={}, model_binding={}, caps=CAPS,
                     limits=LIMITS, prior_seconds=dict(training=66942, gpu=1305, auxiliary=1235),
                     baseline=bindings['results'])
            before = {b['path']: Path(b['path']).read_bytes() for b in bindings.values()}
            self.assertEqual(control.continued_baseline(g), baseline())
            ledger = Ledger(root/'budget_test', g['prior_seconds'], 3, caps=CAPS)
            ledger.reserve('training', 6*10*3600)
            self.assertEqual(ledger.used['training'], 66942)
            bad = copy.deepcopy(g)
            bad['prior_seconds']['training'] = 0
            with self.assertRaises(ValueError):
                control.continued_baseline(bad)
            self.assertEqual(before, {p: Path(p).read_bytes() for p in before})
            write(Path(bindings['results']['path']), {})
            with self.assertRaises(ValueError):
                control.continued_baseline(g)

    def test_actual_worker_and_application_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root/'gate.json'
            write(gate, {})
            manifest = root/'manifest.json'
            write(manifest, {})
            task = tasks('M', SELECTED)[0]
            campaign = root/'mechanism/campaign'
            write(campaign/'M/batch.json', dict(tasks=tasks('M', SELECTED), selected=SELECTED,
                  manifest=record(manifest), gate=record(gate)))
            g = dict(experiment_root=str(root), campaign=str(campaign), manifest=record(manifest),
                     model_binding=dict(files=[]), prior_campaign=str(root/'old_campaign'))
            args = SimpleNamespace(phase='M', action='train', task=task['id'],
                                   target=campaign/'M/training'/task['id'], gate=gate, deadline=1e30)
            class ReachedTraining(Exception):
                pass
            def reached(model, load_fit, origin, actual, identity, binding, check):
                self.assertEqual(actual, task)
                self.assertEqual(origin, args.target/'run')
                raise ReachedTraining()
            with patch.object(control, 'configure_compute'), patch.object(control, 'Check'), \
                    patch('step35_vendorlink_account_run.make_model', return_value=object()), \
                    patch('step35_vendorlink_account_run.execute', side_effect=reached):
                with self.assertRaises(ReachedTraining):
                    control.worker(g, args)
                args.phase = 'C'
                with self.assertRaises(ValueError):
                    control.worker(g, args)
            v = next(t for t in tasks('P', SELECTED) if t['method'] == 'VendorLink_SEQ')
            self.assertTrue(application_completion(g, v).is_relative_to(root/'old_campaign'))
            self.assertTrue(application_completion(g, task).is_relative_to(campaign))

    def test_six_jobs_and_required_history_consumption(self):
        specs = tasks('M', SELECTED)
        self.assertEqual([(t['method'], t['seed']) for t in specs],
                         [(m, s) for s in SEEDS for m in ('H_ER', 'H_full')])
        jobs = control.make_jobs(Path('/tmp/new_campaign'), 'M', specs, 'train')
        self.assertTrue(all(j['seconds'] == 36000 and j['gpu_family'] == 'H' for j in jobs))
        self.assertIsNone(LIMITS['allocator_bytes'])
        self.assertIsNone(LIMITS['process_gpu_bytes'])
        manifest = dict(counts={d: dict(eligible_accounts=n) for d, n in zip(DOMAINS, [546, 157, 875])})
        counts = [3*len(group_chunks(list(range(manifest['counts'][d]['eligible_accounts'])), 8))*40 for d in DOMAINS]
        with tempfile.TemporaryDirectory() as directory:
            endpoint = Path(directory)/'endpoint.txt'
            endpoint.write_text('synthetic metadata only')
            for task in specs[:2]:
                result = dict(task=task, identity={}, status='TRAINING_COMPLETE_EVALUATION_CLOSED',
                              endpoints={str(s): record(endpoint) for s in range(4)}, updates=sum(counts),
                              workload=dict(current_ads=(546+157+875)*12*40,
                                            history_groups=sum(counts[1:]), history_ads=1, teacher_groups=6))
                control.validate_training(task, result, manifest, {})
                result['workload']['history_groups'] = 0
                with self.assertRaisesRegex(ValueError, 'consume retained history'):
                    control.validate_training(task, result, manifest, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
