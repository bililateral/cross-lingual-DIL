"""Only new queue, resource deferral, directory ownership and budget behavior."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import step28_comparison_control as control


class ComparisonControlTest(unittest.TestCase):
    def test_failed_slot_retires_but_healthy_slots_refill(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            script = campaign/'worker.py'
            script.write_text("import sys,time,os,json\nfrom pathlib import Path\n"
                "p=Path(sys.argv[1]);p.mkdir()\n"
                "(p/'start.json').write_text(json.dumps({'time':time.monotonic(),'cpu':list(os.sched_getaffinity(0))}))\n"
                "time.sleep(.12 if p.name=='bad' else .35)\n"
                "(p/'end.json').write_text(json.dumps({'time':time.monotonic()}))\n"
                "raise SystemExit(3 if p.name=='bad' else 0)\n")
            rows = [dict(id=n, allowance=120.) for n in ('bad', 'other1', 'other2', 'next1', 'next2')]
            ledger = control.Ledger(campaign/'budget.json', dict(training=1000, control=30), {})
            command = lambda row, job, deadline, metadata: [sys.executable, '-B', str(script), str(job)]
            with self.assertRaisesRegex(RuntimeError, 'after active siblings finished: bad'):
                control.dispatch(rows, campaign, ledger, 'training', command,
                                 retire_failed_slots=True, poll_seconds=.02)
            self.assertEqual({r['id']: r['success'] for r in ledger.completed},
                             dict(bad=False, other1=True, other2=True, next1=True, next2=True))
            starts = {r['id']: control.data.read_json(campaign/r['id']/'start.json') for r in rows}
            ends = {r['id']: control.data.read_json(campaign/r['id']/'end.json')['time'] for r in rows}
            for identity in ('next1', 'next2'):
                self.assertNotEqual(starts[identity]['cpu'], starts['bad']['cpu'])
                previous = next(n for n in ('other1', 'other2') if starts[n]['cpu'] == starts[identity]['cpu'])
                self.assertGreaterEqual(starts[identity]['time'], ends[previous])
            self.assertTrue(ledger.failed)
            self.assertFalse(ledger.budget_exceeded)
            self.assertEqual(control.data.read_json(campaign/'status.json')['not_dispatched'], [])
            print('Retired-slot evidence:', dict(outcomes={r['id']: r['success'] for r in ledger.completed},
                  cpu_assignment={k:v['cpu'] for k,v in starts.items()}, no_failed_slot_refill=True), flush=True)

    def test_all_failed_slots_leave_remaining_queue_unstarted(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            ledger = control.Ledger(campaign/'budget.json', dict(training=1000, control=30), {})
            command = lambda row, job, deadline, metadata: [sys.executable, '-c', 'import time;time.sleep(.1);raise SystemExit(3)']
            with self.assertRaisesRegex(RuntimeError, 'after active siblings finished'):
                control.dispatch([dict(id=n, allowance=120.) for n in ('a', 'b', 'c', 'pending')],
                                 campaign, ledger, 'training', command, retire_failed_slots=True, poll_seconds=.02)
            self.assertEqual(len(ledger.completed), 3)
            self.assertFalse((campaign/'dispatch/pending').exists())
            with self.assertRaises(ValueError):
                ledger.reserve('a', 'training', 120, allow_failed_campaign=True)

    def test_slot_reserves_unmaterialized_allocations(self):
        gib = 2**30
        self.assertTrue(control.slot_admission(23*gib, [], set(), 0)['slot_ready'])
        row = control.slot_admission(23*gib, [], {10}, 2)
        self.assertFalse(row['slot_ready'])
        self.assertEqual(row['required_free_gpu_bytes'], int(25.5*gib))
        self.assertFalse(control.slot_admission(11*gib, [(10,6*gib),(11,6*gib)], {10,11}, 2)['slot_ready'])
        self.assertTrue(control.slot_admission(10*gib, [(10,int(7.75*gib))], {10}, 1)['slot_ready'])

    def test_recovery_keeps_old_failures_cost_and_only_exact_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, campaign = root/'old', root/'formal'
            old.mkdir(); campaign.mkdir()
            pending = ['CAB_s6_L2R_text_fixed','CAB_s4_L2R_text_fixed','CAB_s5_L2R_text_fixed','CAB_s6_ER_GCR_group_fixed']
            specs = control.m.specifications(new_only=True)
            success = [s for s in specs if s.run_id not in pending]
            source = [dict(path='fixed_algorithm.py', bytes=1, sha256='fixed')]
            old_gate = dict(campaign=str(old), sources=source, budget_seconds=dict(training=1000,control=30))
            control.data.write_json(root/'old_gate.json', old_gate)
            completed = [dict(id=s.run_id,success=True,actual_seconds=10.) for s in success]
            completed += [dict(id=n,success=False,actual_seconds=15.) for n in pending[:3]]
            prior = dict(training=185.,control=3.)
            failure = dict(status='STOP_NO_RETRY',budget=dict(active={},completed=completed,used=prior,failed=True))
            control.data.write_json(root/'failure.json',failure)
            reused = {}
            for spec in success:
                path = old/spec.run_id/'run';path.mkdir(parents=True)
                (path/'payload').write_bytes(b'handwritten artifact')
                payload = control.data.record(path/'payload',path)
                points = {str(i):dict(adam_step=i*288,full_restore_verified=True,inference_restore_verified=True,
                                      model=payload,mapping=payload,memory=payload,calibration_scores=payload)
                          for i in (1,2,3)}
                control.data.write_json(path/'completion.json',dict(status=control.run.COMPLETE,updates=864,
                    spec=spec.identity(),source_files=source,points=points,
                    access=dict(train=1,valid=0,heldout=0,owners=0,worker_train_reads=1)))
                reused[spec.run_id] = control.data.record(path/'completion.json',path)
            for name in pending[:3]:
                (old/name).mkdir();(old/name/'failure.txt').write_text('old failure retained')
            pin = lambda p: {**control.data.record(p,root),'path':str(p)}
            gate = dict(sources=source,budget_seconds=old_gate['budget_seconds'],prior_used_seconds=prior,
                recovery=dict(original_gate=pin(root/'old_gate.json'),original_failure=pin(root/'failure.json'),
                              pending_ids=pending,reused=reused,failure_policy='retire_failed_slot_continue_healthy'))
            ledger = control.Ledger(campaign/'budget.json',gate['budget_seconds'],prior)
            self.assertEqual(control.adopt_recovery(campaign,gate,ledger),pending)
            self.assertEqual(ledger.used,prior)
            self.assertEqual(len(ledger.completed),14)
            self.assertTrue(all(r['success'] for r in ledger.completed))
            self.assertFalse(ledger.failed)
            self.assertEqual(len(control.data.read_json(campaign/'recovery.json')['prior_attempts']),17)
            import os
            self.assertTrue(os.path.samefile(old/success[0].run_id/'run/payload',campaign/success[0].run_id/'run/payload'))
            self.assertTrue((campaign/'prior_failed'/pending[0]/'failure.txt').is_file())
            gate['prior_used_seconds']={**prior,'training':0.}
            with self.assertRaisesRegex(ValueError,'inherited charges'):
                control.recovery_attempts(gate)

    def test_long_reservation_retains_failures_and_whole_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = [0.]
            ledger = control.Ledger(Path(directory)/'budget.json',
                dict(training=800*3600, control=3600, cpu=1200, native=3000),
                dict(cpu=42, native=2479), clock=lambda: clock[0])
            ledger.reserve('l2r', 'training', 72*3600, remaining=684*3600)
            self.assertEqual(ledger.active['l2r']['deadline'], 72*3600)
            with self.assertRaises(ValueError):
                ledger.reserve('extra', 'training', 50*3600, remaining=684*3600)
            clock[0] = 19.
            ledger.settle('l2r', False)
            self.assertEqual(ledger.used['training'], 19.)
            self.assertEqual((ledger.used['cpu'], ledger.used['native']), (42., 2479.))
            self.assertFalse(ledger.completed[0]['success'])
            self.assertTrue(ledger.failed)
            self.assertFalse(ledger.budget_exceeded)
            ledger.charge('control', .1)  # A worker failure is not a budget overrun.
            with self.assertRaises(ValueError):
                ledger.reserve('l2r', 'training', 72*3600)

    def test_real_three_children_own_dirs_and_low_slot_waits(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            script = campaign/'worker.py'
            script.write_text("from pathlib import Path\nimport sys,time,os,json\n"
                "p=Path(sys.argv[1]);p.mkdir();time.sleep(.2)\n"
                "(p/'receipt.json').write_text(json.dumps({'cores':list(os.sched_getaffinity(0)),"
                "'threads':os.environ['OMP_NUM_THREADS']}))\n")
            ledger = control.Ledger(campaign/'budget.json', dict(training=1000, control=30), {})
            observed = []
            def admit(active):
                observed.append(active)
                return len(observed) > 2
            def command(row, job, deadline, metadata):
                self.assertFalse(job.exists())
                return [sys.executable, '-B', str(script), str(job)]
            rows = [dict(id=f'worker{i}', allowance=120.) for i in range(4)]
            control.dispatch(rows, campaign, ledger, 'training', command,
                             admit=admit, poll_seconds=.03)
            self.assertEqual(len(ledger.completed), 4)
            self.assertTrue(all(row['success'] for row in ledger.completed))
            receipts = [json.loads((campaign/row['id']/'receipt.json').read_text()) for row in rows]
            self.assertTrue(all(len(row['cores']) == 1 and row['threads'] == '1' for row in receipts))
            self.assertEqual(len({row['cores'][0] for row in receipts[:3]}), 3)
            self.assertEqual(observed[:2], [0, 0])
            self.assertEqual(ledger.active, {})

    def test_real_failed_child_drains_two_siblings_and_keeps_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            script = campaign/'worker.py'
            script.write_text("import sys,time,json\nfrom pathlib import Path\n"
                "p=Path(sys.argv[1]);p.mkdir()\n"
                "if sys.argv[2]=='bad':\n"
                " time.sleep(.1);raise SystemExit(3)\n"
                "deadline=time.monotonic()+3\n"
                "while not (p.parent/'dispatch/bad/failure.json').exists():\n"
                " if time.monotonic()>deadline: raise SystemExit(9)\n"
                " time.sleep(.01)\n"
                "time.sleep(.1)\n"
                "(p/'finished_after_failure.json').write_text(json.dumps({'finished':True}))\n")
            ledger = control.Ledger(campaign/'budget.json', dict(training=1000, control=30), {})
            rows = [dict(id=name, allowance=120.) for name in ('bad', 'other1', 'other2', 'never')]
            command = lambda row, job, deadline, metadata: [sys.executable, '-B', str(script), str(job), row['id']]
            with self.assertRaisesRegex(RuntimeError, 'after active siblings finished: bad'):
                control.dispatch(rows, campaign, ledger, 'training', command,
                                 poll_seconds=.03)
            results = {row['id']: row['success'] for row in ledger.completed}
            self.assertEqual(results, dict(bad=False, other1=True, other2=True))
            for identity in ('other1', 'other2'):
                self.assertTrue((campaign/identity/'finished_after_failure.json').exists())
            self.assertGreater(ledger.used['training'], .1)
            self.assertTrue(ledger.failed)
            self.assertFalse(ledger.budget_exceeded)
            self.assertEqual(ledger.active, {})
            self.assertFalse((campaign/'never').exists())
            status = control.data.read_json(campaign/'status.json')
            self.assertEqual(status['status'], 'FAILED_WORKERS_DRAINED_NO_RETRY')
            self.assertEqual(status['not_dispatched'], ['never'])
            self.assertEqual(status['failed_workers'][0]['returncode'], 3)

    def test_deadline_stops_only_the_expired_child(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            script = campaign/'worker.py'
            script.write_text("import sys,time\nfrom pathlib import Path\n"
                "p=Path(sys.argv[1]);p.mkdir()\n"
                "time.sleep(3 if sys.argv[2]=='expired' else .6)\n"
                "(p/'finished').write_text('done')\n")
            ledger = control.Ledger(campaign/'budget.json', dict(training=1000, control=30), {})
            rows = [dict(id='expired', allowance=60.25), dict(id='other', allowance=120.)]
            command = lambda row, job, deadline, metadata: [sys.executable, '-B', str(script), str(job), row['id']]
            with self.assertRaisesRegex(RuntimeError, 'after active siblings finished: expired'):
                control.dispatch(rows, campaign, ledger, 'training', command, max_workers=2, poll_seconds=.02)
            self.assertTrue((campaign/'other/finished').exists())
            self.assertFalse((campaign/'expired/finished').exists())
            failure = control.data.read_json(campaign/'dispatch/expired/failure.json')
            self.assertTrue(failure['deadline_stopped'])
            self.assertEqual({r['id']: r['success'] for r in ledger.completed}, dict(expired=False, other=True))

    def test_spawn_error_drains_running_child(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            script = campaign/'worker.py'
            script.write_text("import sys,time\nfrom pathlib import Path\n"
                              "p=Path(sys.argv[1]);p.mkdir();time.sleep(.4);(p/'finished').touch()\n")
            ledger = control.Ledger(campaign/'budget.json', dict(training=1000, control=30), {})
            popen = control.subprocess.Popen
            def launch(command, **kwargs):
                if command[-1].endswith('/bad'):
                    raise OSError('injected worker launch failure')
                return popen(command, **kwargs)
            rows = [dict(id=name, allowance=120.) for name in ('other', 'bad', 'never')]
            command = lambda row, job, deadline, metadata: [sys.executable, '-B', str(script), str(job)]
            with mock.patch.object(control.subprocess, 'Popen', side_effect=launch):
                with self.assertRaisesRegex(RuntimeError, 'after active siblings finished: bad'):
                    control.dispatch(rows, campaign, ledger, 'training', command, max_workers=2, poll_seconds=.02)
            self.assertTrue((campaign/'other/finished').exists())
            self.assertFalse((campaign/'never').exists())
            self.assertEqual({r['id']: r['success'] for r in ledger.completed}, dict(bad=False, other=True))

    def test_global_error_still_stops_children_and_is_not_masked_by_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            ledger = control.Ledger(campaign/'budget.json', dict(training=1000, control=30), {})
            command = lambda row, job, deadline, metadata: [sys.executable, '-c', 'import time;time.sleep(10)']
            calls = []
            def check():
                calls.append(1)
                if len(calls) == 2:
                    ledger.limits['control'] = 0  # Cleanup also fails; the original error must survive.
                    raise ValueError('shared campaign disk limit')
            with self.assertRaisesRegex(ValueError, 'shared campaign disk limit'):
                control.dispatch([dict(id='child', allowance=120.)], campaign, ledger, 'training',
                                 command, max_workers=1, check=check, poll_seconds=.02)
            self.assertEqual(ledger.active, {})
            self.assertFalse(ledger.completed[0]['success'])
            self.assertTrue(ledger.budget_exceeded)
            pid = control.data.read_json(campaign/'dispatch/child/process.json')['pid']
            self.assertFalse(control.psutil.pid_exists(pid))

    def test_failed_dispatch_blocks_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)/'formal'
            gate = dict(campaign=str(campaign), budget_seconds=dict(training=1000, control=30, preparation=30),
                        prior_used_seconds={}, launcher=str(Path(directory)/'absent'), limits={},
                        per_run_seconds=dict(L2R_text_fixed=120., ER_GCR_group_fixed=120.))
            import step28_continual_comparison_evaluate as evaluation
            with mock.patch.object(control, 'validate_gate', return_value=gate), \
                 mock.patch.object(control, 'reference_check'), \
                 mock.patch.object(control, 'resources', return_value=True), \
                 mock.patch.object(control, 'dispatch', side_effect=RuntimeError('Worker failure retained after drain')), \
                 mock.patch.object(evaluation, 'full_gate') as full_gate:
                with self.assertRaisesRegex(RuntimeError, 'Worker failure retained after drain'):
                    control.execute(Path(directory)/'gate.json')
            full_gate.assert_not_called()
            self.assertFalse((campaign/'training_complete.json').exists())
            self.assertFalse((campaign/'completion.json').exists())
            self.assertIn('Worker failure retained after drain', control.data.read_json(campaign/'failure.json')['error'])

    def test_success_cost_only_once_and_active_reservations_are_not_spent(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = [10.]
            ledger = control.Ledger(Path(directory)/'budget.json', dict(training=100, control=30),
                                    {}, clock=lambda: clock[0])
            ledger.reserve('a', 'training', 40, remaining=40)
            ledger.reserve('b', 'training', 40)
            self.assertEqual(ledger.used['training'], 0.)
            self.assertEqual(ledger.available('training'), 20.)
            clock[0] = 15.
            ledger.settle('a', True)
            clock[0] = 17.
            ledger.settle('b', True)
            self.assertEqual(ledger.used['training'], 12.)
            self.assertEqual(ledger.available('training'), 88.)
            with self.assertRaises(ValueError):
                ledger.reserve('a', 'training', 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
