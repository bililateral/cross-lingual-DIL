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


def handwritten_failure():
    return dict(status='STOP_NO_RETRY', budget=dict(active={}, failed=True,
        used={**dict.fromkeys(ctl.LIMITS, 0.), 'training':1000., 'cpu':68., 'control':224.},
        completed=[dict(id=s.run_id, bucket='training', success=i<24, actual_seconds=10.)
                   for i,s in enumerate(ctl.m.specifications()[:26])]))


class RecoveryTests(unittest.TestCase):
    def test_exact_queue_and_real_dispatch_preserve_costs(self):
        failure = handwritten_failure()
        original = copy.deepcopy(failure)
        success, pending = recovery.split_attempts(failure, ctl.m.specifications())
        self.assertEqual(len(success), 24)
        self.assertEqual([s.run_id for s in pending[:2]], list(recovery.FAILED_IDS))
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
            recovery.write(root/'cpu.json',dict(exit_code=0,charged_seconds=1,formal_data_reads=0))
            recovery.write(root/'ready.json',dict(status='REVIEW_CLOSED_AND_CPU_PASSED',
                authorization_sha256=recovery.sha(root/'authorization.json'),sources=[],
                cpu_receipt=str(root/'cpu.json'),cpu_receipt_sha256=recovery.sha(root/'cpu.json')))
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


if __name__ == '__main__':
    unittest.main()
