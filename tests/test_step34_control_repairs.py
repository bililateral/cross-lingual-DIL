"""Only the five concrete operational review findings; no models or real data."""
import json
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from step30_evolution_control import Runner, resource_observation
from step30_evolution_io import Ledger
import step34_vendorlink_account_control as control
from step34_vendorlink_account_data import Check, SharedDiskLimit
from step34_vendorlink_account_policy import LIMITS


class Repairs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, exhaust=False):
        caps = dict(training=1000, gpu=1000, auxiliary=600)
        ledger = Ledger(self.root, dict(training=0, gpu=0, auxiliary=0), 0, caps=caps)
        limits = dict(workers=2, free_ram_bytes=1024, disk_bytes=2**30, free_disk_reserve=1024,
                      free_gpu_bytes=0, process_gpu_bytes=0, rss_bytes=2**30, save_reserve_bytes=1024)
        allowed = dict(experiment_root=str(self.root), workspace=str(self.root), other_gpu_commitments=[])
        class FixtureRunner(Runner):
            def command(self, job, slot, deadline):
                if exhaust:
                    self.ledger.used['auxiliary'] = self.caps['auxiliary']-59-(90 if exhaust == 'reserved' else 0)
                if job['target'].name == 'shared':
                    code = 'import time,pathlib,sys; time.sleep(.3); pathlib.Path(sys.argv[1]).write_text("{}"); sys.exit(1)'
                    return [sys.executable, '-c', code, str(job['target']/'shared_failure.json')]
                return [sys.executable, '-c', 'import time; time.sleep(60)']
        return FixtureRunner(allowed, self.root/'unused_gate', self.root, ledger, caps=caps,
                             limits=limits, halt_dispatch_on_failure=True), ledger

    def jobs(self, *names):
        return [dict(action='fixture', target=self.root/n, bucket='auxiliary', seconds=90, extra=[]) for n in names]

    def test_64gib_dispatch_gate(self):
        allowed = dict(experiment_root=str(self.root), other_gpu_commitments=[])
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=40*2**30)):
            self.assertFalse(resource_observation(allowed, self.root, False, limits=LIMITS)['ready'])
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=64*2**30)):
            self.assertTrue(resource_observation(allowed, self.root, False, limits=LIMITS)['ready'])

    def test_atomic_rename_check_reuses_safe_counter(self):
        class Gone:
            def is_file(self): return True
            def stat(self): raise FileNotFoundError('normal atomic rename')
        check = Check(self.root, 60)
        with patch.object(Path, 'rglob', return_value=iter([Gone()])), patch('shutil.disk_usage', return_value=SimpleNamespace(free=128*2**30)):
            check()  # Old is_file()+stat() implementation raises here.

    def test_launch_exception_stops_same_dispatch_tick(self):
        runner, ledger = self.runner()
        with patch('step30_evolution_control.subprocess.Popen', side_effect=OSError('fixture launch failure')) as launch:
            with self.assertRaises(RuntimeError):
                runner.run_many(self.jobs('bad_start', 'must_not_start'), 2, lambda job: None)
            self.assertEqual(launch.call_count, 1)
        self.assertFalse((self.root/'must_not_start/console.txt').exists())

    def test_shared_disk_marker_and_parent_stop(self):
        marker_path = self.root/'campaign/marker/shared_failure.json'
        argv = ['control', '--gate', str(self.root/'unused'), '--action', 'train', '--target', str(marker_path.parent)]
        with patch.object(control, '__file__', str(self.root/'workspace/scripts/control.py')), \
                patch.object(sys, 'argv', argv), patch.object(control, 'permit', return_value={'campaign': str(self.root/'campaign')}), \
                patch.object(control, 'worker', side_effect=SharedDiskLimit('shared disk fixture')):
            with self.assertRaises(SharedDiskLimit):control.main()
        marker = json.loads(marker_path.read_text())
        self.assertEqual(marker['status'], 'SHARED_DISK_FAILURE')
        with patch.object(control, '__file__', str(self.root/'workspace/scripts/control.py')), \
                patch.object(sys, 'argv', argv), patch.object(control, 'permit', side_effect=ValueError('bound source differs')):
            with self.assertRaises(ValueError):control.main()
        self.assertEqual(json.loads(marker_path.read_text())['status'], 'SOURCE_VALIDITY_FAILURE')
        runner, ledger = self.runner()
        with self.assertRaises(RuntimeError):runner.run_many(self.jobs('shared', 'healthy', 'never'), 2, lambda job: None)
        receipt = json.loads((self.root/'healthy/dispatch_receipt.json').read_text())
        self.assertFalse(receipt['success'])
        self.assertTrue(receipt['released'])
        self.assertFalse((self.root/'never/console.txt').exists())
        self.assertFalse(json.loads((self.root/'dispatch.json').read_text())['active'])

    def test_auxiliary_hard_stop_releases_owned_child(self):
        runner, ledger = self.runner(exhaust=True)
        with self.assertRaisesRegex(TimeoutError, 'cleanup reserve'):
            runner.run_many(self.jobs('active'), 1, lambda job: self.fail('Aborted job cannot complete'))
        dispatch = json.loads((self.root/'dispatch.json').read_text())
        self.assertFalse(dispatch['active'])
        self.assertIn('Auxiliary monitoring cleanup reserve', dispatch['controller_error'])
        self.assertTrue(json.loads((self.root/'active/dispatch_receipt.json').read_text())['released'])
        self.assertLessEqual(ledger.used['auxiliary'], ledger.caps['auxiliary'])

    def test_active_auxiliary_commitment_is_protected(self):
        runner, ledger = self.runner(exhaust='reserved')
        with self.assertRaisesRegex(TimeoutError, 'cleanup reserve'):
            runner.run_many(self.jobs('active'), 1, lambda job: self.fail('Reserved balance was consumed'))
        self.assertFalse(json.loads((self.root/'dispatch.json').read_text())['active'])
        self.assertLessEqual(ledger.used['auxiliary'], ledger.caps['auxiliary'])

    def test_phase_decisions_are_measured_without_worker_time(self):
        active, recorded = [], []
        @contextmanager
        def measure(name):
            active.append(name)
            yield
            recorded.append(active.pop())
        def phase(name, selected):
            self.assertFalse(active)
            return {}
        def save(name, value):
            self.assertTrue(active)
        with patch.object(control, 'select', return_value={}), patch.object(control, 'decide', return_value={'status': 'STOP_LEARNING'}):
            result = control.pipeline(phase, save, measure)
        self.assertEqual(result['status'], 'STOP_LEARNING')
        self.assertEqual(recorded, ['select_D', 'decide_P'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
