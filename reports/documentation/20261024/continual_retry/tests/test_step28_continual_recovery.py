"""Handwritten recovery orchestration checks; Linux py310 only, no BGE/data."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(os.environ['RECOVERY_FROZEN_WORKSPACE'])/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import step28_continual_five_recover as recovery
import step28_continual_five_control as ctl
import step28_continual_refill as refill


def handwritten_failure():
    return dict(status='STOP_NO_RETRY', budget=dict(active={}, failed=True,
        used={**dict.fromkeys(ctl.LIMITS, 0.), 'training':1000., 'cpu':68., 'control':224.},
        completed=[dict(id=s.run_id, bucket='training', success=i<24, actual_seconds=10.)
                   for i,s in enumerate(ctl.m.specifications()[:27])]))


def handwritten_ready(root, gate):
    (root/'tests').mkdir()
    (root/'scripts').mkdir()
    (root/'tests/test_step28_continual_recovery.py').write_bytes(Path(__file__).read_bytes())
    (root/'scripts/step28_continual_refill.py').write_bytes(Path(refill.__file__).read_bytes())
    paths = [Path(recovery.__file__).resolve(), root/'scripts/step28_continual_refill.py',
             root/'tests/test_step28_continual_recovery.py', root/gate['launcher_name']]
    sources = [dict(path=str(p), sha256=recovery.sha(p)) for p in paths]
    recovery.write(root/'cpu.json', dict(exit_code=0, charged_seconds=1, recovery_cumulative_seconds=1,
        campaign_cumulative_cpu_seconds=69, formal_data_reads=0, sources=sources))
    ready = dict(status='REVIEW_CLOSED_AND_CPU_PASSED',
        authorization_sha256=recovery.sha(root/'authorization.json'), sources=sources,
        cpu_receipt=str(root/'cpu.json'), cpu_receipt_sha256=recovery.sha(root/'cpu.json'))
    recovery.write(root/'ready.json', ready)
    return ready


class RecoveryTests(unittest.TestCase):
    def test_exact_queue_and_real_dispatch_preserve_costs(self):
        failure = handwritten_failure()
        original = copy.deepcopy(failure)
        success, pending = recovery.split_attempts(failure, ctl.m.specifications())
        self.assertEqual(len(success), 24)
        self.assertEqual([s.run_id for s in pending[:3]], list(recovery.FAILED_IDS))
        with tempfile.TemporaryDirectory(dir=os.environ['RECOVERY_TEST_ROOT']) as tmp:
            root = Path(tmp)
            ledger = recovery.inherited_ledger(ctl, root/'budget.json', failure, 3., 2.)
            self.assertEqual(ledger.used['training'], 1000.)
            self.assertEqual(ledger.used['cpu'], 71.)
            self.assertEqual(ledger.used['control'], 226.)
            queue = [dict(id=s.run_id, allowance=10., projected_seconds=0.) for s in pending]
            def command(row, job, deadline):
                return [sys.executable, '-c', 'from pathlib import Path; Path("'+str(job/'ran')+'").write_text("ran")']
            ctl.dispatch(queue, root, ledger, 'training', command, max_workers=1, tail=.1, poll_seconds=.01)
            self.assertEqual({p.parent.name for p in root.glob('*/ran')}, {s.run_id for s in pending})
            self.assertFalse(any((root/s).exists() for s in success))
            self.assertEqual(len(ledger.completed), 45)
            self.assertGreater(ledger.used['training'], 1000.)
            self.assertEqual(failure, original)
        broken = copy.deepcopy(failure)
        broken['budget']['completed'][-1]['success'] = True
        with self.assertRaises(ValueError):
            recovery.split_attempts(broken, ctl.m.specifications())

    def test_hardlink_custody_is_counted_by_actual_budget(self):
        with tempfile.TemporaryDirectory(dir=os.environ['RECOVERY_TEST_ROOT']) as tmp:
            root = Path(tmp)
            old, aggregate = root/'old', root/'aggregate'
            old.mkdir(); aggregate.mkdir()
            (old/'model.pt').write_bytes(b'model'*100)
            (old/'evidence.json').write_text('{}')
            recovery.hardlink_tree(old, aggregate/'successful')
            recovery.hardlink_tree(old, aggregate/'prior_failed')
            self.assertTrue(os.path.samefile(old/'model.pt', aggregate/'successful/model.pt'))
            self.assertEqual(recovery.sha(old/'model.pt'), recovery.sha(aggregate/'successful/model.pt'))
            budget = ctl.run.Budget(aggregate, __import__('time').monotonic()+30., tail=1.)
            budget.check()
            self.assertEqual(budget.peak_disk, 1004)
            self.assertEqual(budget.peak_small, 4)
            with self.assertRaises(ValueError):
                recovery.hardlink_tree(old, aggregate/'successful')

    def test_gpu_foreign_pid_and_review_are_both_gates(self):
        row = dict(free_gpu_bytes=31*2**30, compute_apps='', host_available_bytes=220*2**30,
                   disk_free_bytes=300*2**30)
        self.assertTrue(recovery.resource_ready(row))
        self.assertFalse(recovery.resource_ready({**row, 'compute_apps':'569359, 27450'}))
        self.assertFalse(recovery.resource_ready({**row, 'free_gpu_bytes':4438*2**20}))
        with tempfile.TemporaryDirectory(dir=os.environ['RECOVERY_TEST_ROOT']) as tmp:
            root=Path(tmp)
            gate=dict(status='AUTHORIZED_WAIT_BASED_RECOVERY', review_ready=str(root/'absent.json'),
                      prior_control_seconds=224., pending_ids=list(recovery.FAILED_IDS), campaign=str(root/'formal'))
            recovery.write(root/'authorization.json',gate)
            with mock.patch.object(recovery,'resources',return_value=row), \
                 mock.patch.object(recovery.subprocess,'Popen') as launch, \
                 mock.patch.object(recovery.time,'sleep',side_effect=InterruptedError('end handwritten wait')):
                with self.assertRaises(InterruptedError):
                    recovery.wait_for_recovery(root/'authorization.json',root/'wait.sh')
                launch.assert_not_called()
            self.assertFalse(recovery.read(root/'waiter_status.json')['review_ready'])
            with self.assertRaises(FileExistsError):
                recovery.wait_for_recovery(root/'authorization.json',root/'wait.sh')

    def test_real_child_update_causes_only_launcher_self_removal(self):
        with tempfile.TemporaryDirectory(dir=os.environ['RECOVERY_TEST_ROOT']) as tmp:
            root=Path(tmp)
            launcher=root/'wait.sh'; launcher.write_text('handwritten launcher')
            gate=dict(status='AUTHORIZED_WAIT_BASED_RECOVERY', review_ready=str(root/'ready.json'),
                      prior_control_seconds=224., pending_ids=['handwritten_new'], campaign=str(root/'formal'),
                      launcher_name=launcher.name)
            recovery.write(root/'authorization.json',gate)
            handwritten_ready(root, gate)
            code=('import time; from pathlib import Path; p=Path('+repr(str(root/'formal/handwritten_new'))+'); '
                  'p.mkdir(parents=True); (p/"console.txt").write_text(\'{"event":"updates","completed":1}\\n\'); time.sleep(1)')
            actual_popen=subprocess.Popen
            children=[]
            def launch_handwritten(*args,**kwargs):
                child=actual_popen([sys.executable,'-c',code],**kwargs); children.append(child); return child
            actual_sleep=__import__('time').sleep
            row=dict(free_gpu_bytes=31*2**30,compute_apps='',host_available_bytes=220*2**30,disk_free_bytes=300*2**30)
            with mock.patch.object(recovery,'resources',return_value=row), \
                 mock.patch.object(recovery.subprocess,'Popen',side_effect=launch_handwritten), \
                 mock.patch.object(recovery.time,'sleep',side_effect=lambda n:actual_sleep(.02)):
                recovery.wait_for_recovery(root/'authorization.json',launcher)
            for child in children: child.wait(timeout=5)
            self.assertEqual(len(children),1)
            self.assertFalse(launcher.exists())
            self.assertTrue((root/'authorization.json').exists())
            self.assertEqual(recovery.read(root/'waiter_status.json')['status'],'TRAINING_UPDATED_WAITER_REMOVED')


class ReviewRepairTests(unittest.TestCase):
    def test_required_sources_must_match_tested_sources(self):
        with tempfile.TemporaryDirectory(dir=os.environ['RECOVERY_TEST_ROOT']) as tmp:
            root = Path(tmp)
            gate = dict(review_ready=str(root/'ready.json'), launcher_name='wait.sh')
            (root/'wait.sh').write_text('handwritten launcher')
            recovery.write(root/'authorization.json', gate)
            ready = handwritten_ready(root, gate)
            self.assertIsNotNone(recovery.validate_ready(gate, root/'authorization.json'))
            recovery.write(root/'ready.json', {**ready, 'sources': []})
            with self.assertRaisesRegex(ValueError, 'coverage'):
                recovery.validate_ready(gate, root/'authorization.json')
            cpu = recovery.read(root/'cpu.json')
            cpu['sources'][0]['sha256'] = 'wrong tested source'
            recovery.write(root/'cpu.json', cpu)
            recovery.write(root/'ready.json', {**ready, 'cpu_receipt_sha256': recovery.sha(root/'cpu.json')})
            with self.assertRaisesRegex(ValueError, 'CPU evidence'):
                recovery.validate_ready(gate, root/'authorization.json')

    def test_wait_exception_is_charged_once_and_sleep_is_excluded(self):
        for fault in ('resource', 'review', 'budget', 'sleep'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(dir=os.environ['RECOVERY_TEST_ROOT']) as tmp:
                root = Path(tmp)
                gate = dict(status='AUTHORIZED_WAIT_BASED_RECOVERY', review_ready=str(root/'absent.json'),
                            prior_control_seconds=1739. if fault == 'budget' else 224., prior_wait_control_seconds=4.)
                recovery.write(root/'authorization.json', gate)
                clock = [0.]
                def resource():
                    clock[0] += 15. if fault == 'resource' else 2.
                    if fault == 'resource': raise TimeoutError('handwritten resource timeout')
                    return dict(free_gpu_bytes=31*2**30,compute_apps='',host_available_bytes=220*2**30,disk_free_bytes=300*2**30)
                def review(*args):
                    if fault == 'review':
                        clock[0] += 3.
                        raise ValueError('handwritten review mismatch')
                    return None
                def sleep(seconds):
                    clock[0] += seconds
                    raise InterruptedError('end handwritten sleep')
                with mock.patch.object(recovery, 'resources', side_effect=resource), \
                     mock.patch.object(recovery, 'validate_ready', side_effect=review), \
                     mock.patch.object(recovery.time, 'monotonic', side_effect=lambda: clock[0]), \
                     mock.patch.object(recovery.time, 'sleep', side_effect=sleep), \
                     mock.patch.object(recovery.subprocess, 'Popen') as launch:
                    with self.assertRaises((TimeoutError, ValueError, RuntimeError, InterruptedError)):
                        recovery.wait_for_recovery(root/'authorization.json', root/'wait.sh')
                    launch.assert_not_called()
                charge = recovery.read(root/'waiter_failure.json')['control_seconds']
                self.assertEqual(charge, {'resource':19., 'review':9., 'budget':6., 'sleep':6.}[fault])
                print('Wait accounting handwritten case:', fault, 'charged=', charge, flush=True)

    def test_preparation_overlimit_is_charged_once_and_failure_is_saved(self):
        for preparation_fails in (False, True):
            with self.subTest(preparation_fails=preparation_fails), tempfile.TemporaryDirectory(dir=os.environ['RECOVERY_TEST_ROOT']) as tmp:
                root = Path(tmp)
                old = root/'old'; old.mkdir(); (old/'prior_failed').mkdir()
                failure = handwritten_failure(); failure['budget']['used']['control'] = 1730.
                recovery.write(old/'failure.json', failure)
                _, pending = recovery.split_attempts(failure, ctl.m.specifications())
                gate = dict(status='AUTHORIZED_WAIT_BASED_RECOVERY', workspace=os.environ['RECOVERY_FROZEN_WORKSPACE'],
                    original_gate=str(root/'original_gate.json'), original_campaign=str(old), campaign=str(root/'formal'),
                    pending_ids=[s.run_id for s in pending], inherited_recovery_cpu_seconds=28)
                recovery.write(root/'authorization.json', gate)
                recovery.write(root/'original_gate.json', dict(sources=['handwritten']))
                recovery.write(root/'cpu.json', dict(recovery_cumulative_seconds=29))
                actual_sha = recovery.sha
                def identity(path):
                    return recovery.OLD_FAILURE_SHA if Path(path) == old/'failure.json' else actual_sha(path)
                with mock.patch.object(recovery, 'validate_ready', return_value=dict(cpu_receipt=str(root/'cpu.json'))), \
                     mock.patch.object(recovery, 'sha', side_effect=identity), \
                     mock.patch.object(ctl.run, 'sources', return_value=['handwritten']), \
                     mock.patch.object(ctl, 'resource_admission', return_value={'handwritten': True}), \
                     mock.patch.object(recovery, 'verify_reused', side_effect=ValueError('handwritten preparation failure') if preparation_fails else None), \
                     mock.patch.object(recovery, 'hardlink_tree') as links, \
                     mock.patch.object(recovery.time, 'monotonic', side_effect=[100., 180.]), \
                     mock.patch.object(refill, 'dispatch') as launch:
                    with self.assertRaises((RuntimeError, ValueError)):
                        recovery.execute(root/'authorization.json', 0.)
                    launch.assert_not_called()
                record = recovery.read(root/'formal/failure.json')
                self.assertEqual(record['budget']['used']['control'], 1840.)
                self.assertEqual(record['budget']['active'], {})
                self.assertEqual(record['budget']['used']['training'], 1000.)
                self.assertEqual(record['budget']['used']['cpu'], 69.)
                if not preparation_fails:
                    links.assert_any_call(old/'prior_failed', root/'formal/prior_failed/earlier')
                self.assertEqual(record['accounting_error'] is not None, preparation_fails)
                if preparation_fails:
                    self.assertIn('handwritten preparation failure', record['error'])
                print('Preparation case:', preparation_fails, 'one charge=', 110., 'failure saved', flush=True)


class RefillTests(unittest.TestCase):
    def test_refill_exact_single_slot_threshold_ignores_foreign_pids(self):
        with tempfile.TemporaryDirectory(dir=os.environ['RECOVERY_TEST_ROOT']) as tmp:
            root = Path(tmp)
            gate = dict(minimum_free_slot_bytes=10*2**30)
            def query(command, **kwargs):
                if '--query-compute-apps=pid,used_memory' in command:
                    return '999999, 4096'  # A different CUDA process would exist.
                return str(free_mib)
            with mock.patch.object(refill.subprocess, 'check_output', side_effect=query), \
                 mock.patch.object(refill.psutil, 'virtual_memory', return_value=mock.Mock(available=80*2**30)):
                free_mib = 10239
                self.assertFalse(refill.slot_available(gate, 2, root))
                free_mib = 10240
                self.assertTrue(refill.slot_available(gate, 2, root))
                # Even with more than one vacant slot, refill checks only the next worker.
                self.assertTrue(refill.slot_available(gate, 0, root))
            self.assertEqual(recovery.read(root/'admission_status.json')['required_free_gpu_bytes'], 10*2**30)

    def test_refill_wait_preserves_two_live_workers_then_fills_vacancy(self):
        with tempfile.TemporaryDirectory(dir=os.environ['RECOVERY_TEST_ROOT']) as tmp:
            root = Path(tmp)
            ledger = ctl.Ledger(root/'budget.json')
            queue = [dict(id=f'child{i}', allowance=10., projected_seconds=0.) for i in range(4)]
            release = root/'release'
            child_code = ('import sys,time; from pathlib import Path; p=Path(sys.argv[1]); '
                          'r=Path(sys.argv[2]); (p/"alive").write_text("alive"); '
                          'start=time.monotonic();\n'
                          'while not r.exists() and time.monotonic()-start<5: time.sleep(.01)\n'
                          '(p/"done").write_text("done")')
            def command(row, job, deadline):
                if row['id'] == 'child0':
                    return [sys.executable, '-c', 'pass']
                return [sys.executable, '-c', child_code, str(job), str(release)]
            observations = []
            def admit(active):
                started = len(list(root.glob('child*/process.json')))
                free = 31*1024 if started < 3 else (9*1024 if len(observations) < 3 else 10*1024)
                with mock.patch.object(refill.subprocess, 'check_output', return_value=str(free)), \
                     mock.patch.object(refill.psutil, 'virtual_memory', return_value=mock.Mock(available=80*2**30)):
                    allowed = refill.slot_available(dict(minimum_free_slot_bytes=10*2**30), active, root)
                if started == 3 and not allowed:
                    pids = [recovery.read(root/f'child{i}/process.json')['pid'] for i in (1, 2)]
                    self.assertTrue(all(__import__('psutil').pid_exists(pid) for pid in pids))
                    self.assertFalse((root/'child3').exists())
                    observations.append(dict(active=active, live_pids=pids, new_worker_started=False))
                elif started == 3 and allowed:
                    release.write_text('release')
                return allowed
            # Three real child processes share the single authorized test CPU core.
            core = __import__('psutil').Process().cpu_affinity()[0]
            with mock.patch.object(refill.psutil.Process, 'cpu_affinity', return_value=[core]*3):
                refill.dispatch(ctl, queue, root, ledger, 'training', command,
                                admit=admit, tail=.1, poll_seconds=.02)
            self.assertEqual(len(observations), 3)
            self.assertTrue(all(row['active'] == 2 for row in observations))
            self.assertTrue(all(row['success'] for row in ledger.completed))
            self.assertEqual(len(ledger.completed), 4)
            self.assertTrue((root/'child3/done').exists())
            print('Low-GPU refill observations; two workers retained, vacant slot not launched:', observations, flush=True)

    def test_refill_empty_active_set_waits_and_real_failure_still_stops(self):
        with tempfile.TemporaryDirectory(dir=os.environ['RECOVERY_TEST_ROOT']) as tmp:
            root = Path(tmp)
            ledger = ctl.Ledger(root/'budget.json')
            query = mock.Mock(side_effect=[False, False, True])
            with mock.patch.object(refill.time, 'sleep', wraps=__import__('time').sleep) as sleep:
                refill.dispatch(ctl, [dict(id='after_wait', allowance=10.)], root, ledger, 'training',
                    lambda row, job, deadline: [sys.executable, '-c', 'pass'],
                    admit=query, max_workers=1, tail=.1, poll_seconds=.01)
            self.assertGreaterEqual(sleep.call_count, 2)
            self.assertEqual(len(ledger.completed), 1)
            failed_root = root/'genuine_failure'; failed_root.mkdir()
            failed_ledger = ctl.Ledger(failed_root/'budget.json')
            with self.assertRaisesRegex(RuntimeError, 'Failed/over-budget'):
                refill.dispatch(ctl, [dict(id='bad', allowance=10.), dict(id='never', allowance=10.)],
                    failed_root, failed_ledger, 'training',
                    lambda row, job, deadline: [sys.executable, '-c', 'raise SystemExit(1)'],
                    admit=lambda active: True, max_workers=1, tail=.1, poll_seconds=.01)
            self.assertTrue(failed_ledger.failed)
            self.assertFalse((failed_root/'never').exists())


if __name__ == '__main__':
    unittest.main()
