"""Incremental Linux CPU evidence; no model, official data, CUDA or live signals."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parent
W = Path('/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace')
P = Path('/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_pause/repair/final_sources')
sys.path.insert(0, str(W/'scripts'))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retry = load(ROOT/'workspace/scripts/retry_once.py', 'retry_under_test')
dispatch = load(P/'step28_history_optuna_control.py', 'actual_reviewed_dispatch')
control = load(P/'step28_h_ablation_control.py', 'actual_reviewed_controller')


def fixture():
    names = ['good', retry.TARGET, 'next', 'last', 'tail']
    snapshot = dict(sources=['frozen-science'], active={}, retired_slots=[0],
        completed=[dict(id='good', index=0, slot=1, success=True, released=True),
                   dict(id=retry.TARGET, index=1, slot=0, success=False, released=True)],
        pending=names[2:])
    return names, snapshot


class IncrementalTests(unittest.TestCase):
    def test_one_retry_transform_preserves_history_and_rejects_others(self):
        names, prior = fixture()
        original = copy.deepcopy(prior)
        actual, failed = retry.retry_continuation(prior, names)
        self.assertEqual(prior, original)
        self.assertEqual(actual['pending'], [retry.TARGET, 'next', 'last', 'tail'])
        self.assertEqual(actual['retired_slots'], [])
        self.assertEqual(actual['completed'], [original['completed'][0]])
        self.assertEqual(failed, original['completed'][1])
        for mutation in ('other_failure', 'active', 'wrong_slot', 'wrong_pending'):
            bad = copy.deepcopy(original)
            if mutation == 'other_failure': bad['completed'][0]['success'] = False
            if mutation == 'active': bad['active'] = {'1': {'id': 'next', 'pid': 1}}
            if mutation == 'wrong_slot': bad['completed'][1]['slot'] = 2
            if mutation == 'wrong_pending': bad['pending'] = []
            with self.assertRaises(ValueError): retry.retry_continuation(bad, names)

    def test_actual_cpu_retry_uses_slot0_then_slot0_claims_next(self):
        names, prior = fixture()
        effective, _ = retry.retry_continuation(prior, names)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = control.Ledger(root, dict(training=15., confirmation=0., auxiliary=0.),
                                    dict(training=1000., confirmation=1000., auxiliary=1000.))
            original_events = copy.deepcopy(ledger.events)
            rows = [dict(id=n, seconds=10, bucket='training', gpu=False, shutdown_reserve=0) for n in names]
            allowed = dict(sources=['frozen-science'], workers=3, workspace=str(root), campaign=str(root))
            def command(row, job, deadline):
                duration = .05 if row['id'] == retry.TARGET else .45
                return [sys.executable, '-c', 'import time,sys;time.sleep(float(sys.argv[1]))', str(duration)]
            with patch.object(dispatch.queue, 'check_running', lambda *a: None):
                result = dispatch.dispatch(root, rows, command, lambda *a: None, allowed, ledger,
                    poll_seconds=.01, continuation=effective)
            self.assertEqual(result['status'], 'ALL_COMPLETE_RELEASED')
            receipts = {r['id']:r for r in result['completed']}
            self.assertEqual(receipts[retry.TARGET]['slot'], 0)
            self.assertEqual(receipts['tail']['slot'], 0)
            self.assertTrue(all(r['success'] and r['released'] for r in result['completed']))
            self.assertFalse((root/'good').exists())
            self.assertGreater(ledger.used['training'], 15.)
            self.assertEqual(ledger.events[:len(original_events)], original_events)

    def test_new_failure_retires_again_and_blocks_downstream(self):
        names, prior = fixture()
        effective, _ = retry.retry_continuation(prior, names)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = control.Ledger(root, dict(training=15., confirmation=0., auxiliary=0.),
                                    dict(training=1000., confirmation=1000., auxiliary=1000.))
            rows = [dict(id=n, seconds=10, bucket='training', gpu=False, shutdown_reserve=0) for n in names]
            allowed = dict(sources=['frozen-science'], workers=3, workspace=str(root), campaign=str(root))
            def command(row, job, deadline):
                return [sys.executable, '-c', 'import sys;sys.exit(int(sys.argv[1]))',
                        '7' if row['id'] == retry.TARGET else '0']
            with patch.object(dispatch.queue, 'check_running', lambda *a: None):
                with self.assertRaisesRegex(RuntimeError, 'Incomplete batch'):
                    dispatch.dispatch(root, rows, command, lambda *a: None, allowed, ledger,
                                      poll_seconds=.01, continuation=effective)
            self.assertEqual(retry.read(root/'dispatch.json')['retired_slots'], [0])
            self.assertEqual(len([e for e in ledger.events if e['id'] == retry.TARGET]), 1)

    def test_beta_binding_only_changes_supervisor_and_carries_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = Path(temporary)/'request.json'
            retry.write(request, {'approved': True})
            gate = dict(prior_seconds=dict(auxiliary=92, training=0, gpu=0),
                dependency=dict(request={}, controller={}, state='old', campaign='unchanged'),
                caps=dict(training=144000, gpu=3600, auxiliary=3600), sources=['unchanged'])
            actual = retry.beta_replacement_gate(gate, request, {'pid': 2}, Path(temporary), 185)
            self.assertEqual(actual['prior_seconds']['auxiliary'], 185)
            self.assertEqual(actual['caps'], gate['caps'])
            self.assertEqual(actual['sources'], gate['sources'])
            self.assertEqual(actual['dependency']['campaign'], gate['dependency']['campaign'])
            self.assertEqual(actual['dependency']['request'], retry.record(request))
            with self.assertRaises(ValueError):
                retry.beta_replacement_gate(gate, request, {}, Path(temporary), 0)

    def test_closed_entry_cannot_signal_or_create_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = Path(temporary)/'request.json'
            retry.write(request, dict(approved=False, review_closed=False))
            with patch.object(retry, 'load', side_effect=AssertionError('Premature runtime import')):
                with self.assertRaises(ValueError): retry.execute(request)
            self.assertFalse((Path(temporary)/'state').exists())

    def test_unchanged_budget_reserves_complete_retry_and_rejects_shortfall(self):
        caps = dict(training=378*3600, confirmation=43200, auxiliary=14400)
        budget = dict(caps=caps, used_seconds=dict(training=152265., confirmation=0., auxiliary=12180.))
        retry.check_allowance(budget, caps, 1200, 2)
        self.assertEqual(budget['used_seconds']['auxiliary'], 12180.)
        for auxiliary, training, preparation in ((13100, 152265, 1200), (12180, 1350000, 1), (12180, 152265, 1201)):
            bad = copy.deepcopy(budget)
            bad['used_seconds'].update(auxiliary=auxiliary, training=training)
            with self.assertRaises(TimeoutError): retry.check_allowance(bad, caps, preparation, 2)


if __name__ == '__main__':
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IncrementalTests))
    retry.write(ROOT/'cpu02.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun, errors=len(result.errors), failures=len(result.failures),
        seconds=time.monotonic()-started, sources=[retry.record(Path(__file__)),
        retry.record(ROOT/'workspace/scripts/retry_once.py'), retry.record(P/'step28_history_optuna_control.py')],
        scope='Real CPU retry/slot0 handoff; reused freeze/drain, no GPU/model or official data'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
