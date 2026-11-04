"""Linux CPU checks for the changed dispatch/chain, with real tiny subprocesses."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import step28_formal_queue as q
import step28_formal_wait as w

EVIDENCE = {}


class Memory:
    def snapshot(self):
        return dict(free=32*2**30, total=32*2**30, processes={})
    def close(self):
        pass


def allowance(root, workers=3):
    return dict(campaign=str(root/'batch'), workspace=str(root), workers=workers,
        worker_failure_policy='retire_failed_slot_continue_healthy' if workers == 3 else 'stop',
        prior_used_seconds=dict(training=7.), budget_seconds=dict(training=2000, control=30, analysis=100),
        sources=[], limits=dict(rss_bytes=4*2**30, disk_bytes=64*2**20,
        free_reserve_bytes=1, free_gpu_bytes=10*2**30, process_gpu_bytes=10*2**30))


def command(row, job, deadline):
    code = "import pathlib,time,sys;time.sleep(.12);pathlib.Path('"+str(job/'ran')+"').write_text('yes');sys.exit("+str(row.get('exit', 0))+")"
    return [sys.executable, '-B', '-c', code]


class FormalWaitTests(unittest.TestCase):
    def test_actual_forever_budget_tolerates_peer_retirement_preserves_caps(self):
        import step28_forever_formal as formal
        class Retired:
            suffix = '.pt'
            def is_file(self): return True
            def is_symlink(self): return False
            def stat(self): raise FileNotFoundError('Peer legitimately retired this checkpoint')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kept = root/'kept.txt'
            kept.write_bytes(b'abc')
            limits=dict(rss_bytes=4*2**30,disk_bytes=100,small_bytes=100,
                        free_reserve_bytes=1,allocator_bytes=9*2**30)
            old=formal.ComparisonBudget(root,time.monotonic()+600,limits.copy())
            actual=formal.Budget(root,time.monotonic()+600,limits.copy())
            with patch.object(Path,'rglob',return_value=[Retired(),kept]):
                with self.assertRaises(FileNotFoundError): old.check()
                actual.check()
                self.assertEqual(actual.peak_disk,3)
                self.assertEqual(q.campaign_size(root),3)
                for key in ('disk_bytes','small_bytes','rss_bytes'):
                    bounded=formal.Budget(root,time.monotonic()+600,{**limits,key:1})
                    with self.assertRaisesRegex(RuntimeError,'resource allowance'): bounded.check()
                bounded=formal.Budget(root,time.monotonic()+600,limits)
                with self.assertRaisesRegex(RuntimeError,'resource allowance'): bounded.check(reserve=100)
                with self.assertRaises(TimeoutError): formal.Budget(root,time.monotonic()+30,limits).check()
                with patch.object(formal.shutil,'disk_usage',return_value=type('Disk',(),{'free':0})()):
                    with self.assertRaisesRegex(RuntimeError,'resource allowance'):
                        formal.Budget(root,time.monotonic()+600,limits).check()
            import step28_forever_evaluate as evaluation
            self.assertIs(evaluation.formal.Budget,formal.Budget)
            EVIDENCE['worker_budget_retirement'] = dict(old_raises=True,new_actual_class_passes=True,
                kept_bytes=3,disk_small_rss_reserve_free_deadline_checks_preserved=True,
                training_and_confirmation_resolve_same_budget=True)

    def test_forever_actual_parent_passes_three_slots_and_all_followups(self):
        import step28_forever_formal as formal
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = allowance(root)
            allowed.update(runs=[s.identity() for s in formal.specs()],epochs=6,
                extra_full_parameter_anchor=True,per_run_seconds=43200,
                early_evaluation_opening=False,automatic_retry=False)
            allowed['budget_seconds'].update(training=108*3600,generation=600,confirmation=21600,analysis=900)
            def actual_dispatch(scope, rows, factory, validate, followups, **kwargs):
                self.assertEqual(scope['workers'],3)
                self.assertEqual(len(rows),9)
                self.assertEqual([r['action'] for r in followups],['generate','confirmation','analyze'])
                argv = factory(rows[0], root/'job', 123.)
                self.assertIn('worker',argv)
                self.assertTrue(argv[2].endswith('step28_forever_formal.py'))
                EVIDENCE['actual_forever_entry'] = dict(rows=9,workers=scope['workers'],followups=followups,command=argv)
            with patch.object(q,'gate',return_value=allowed), patch.object(q,'dispatch',side_effect=actual_dispatch), \
                 patch.object(sys,'argv',['step28_forever_formal.py','train','--gate',str(root/'gate.json')]):
                formal.main()

    def test_failure_retires_slot_and_healthy_slots_take_remaining_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = allowance(root)
            rows = [dict(id=str(i), seconds=100, exit=2 if i == 0 else 0) for i in range(9)]
            with patch.object(q, 'GpuMemory', Memory):
                with self.assertRaisesRegex(RuntimeError, 'Incomplete training'):
                    q.dispatch(allowed, rows, command, lambda r, p: self.assertTrue((p/'ran').exists()),
                        [dict(id='analysis', seconds=90, bucket='analysis', gpu=False)], poll_seconds=.03)
            receipt = w.read(root/'batch/failure.json')
            self.assertEqual(len(receipt['completed']), 9)
            self.assertEqual(sum(r['success'] for r in receipt['completed']), 8)
            self.assertEqual(receipt['retired_slots'], [0])
            self.assertTrue(all(r['slot'] in (1, 2) for r in receipt['completed'] if int(r['id']) >= 3))
            self.assertGreater(receipt['used']['training'], 7)
            self.assertFalse((root/'batch/analysis').exists())
            self.assertFalse((root/'batch/training_complete.json').exists())
            EVIDENCE['healthy_refill'] = receipt['completed']

    def test_process_limit_stops_only_offender_then_refills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class Over(Memory):
                def snapshot(self):
                    value = super().snapshot()
                    path = root/'batch/0/process.json'
                    if path.exists():
                        value['processes'][w.read(path)['pid']] = 11*2**30
                    return value
            rows = [dict(id=str(i), seconds=100) for i in range(5)]
            def slow(row, job, deadline):
                return [sys.executable, '-c', 'import time;time.sleep(30)'] if row['id'] == '0' else command(row, job, deadline)
            with patch.object(q, 'GpuMemory', Over):
                with self.assertRaisesRegex(RuntimeError, 'Incomplete training'):
                    q.dispatch(allowance(root), rows, slow, lambda r, p: None, poll_seconds=.03)
            result = w.read(root/'batch/failure.json')
            self.assertEqual(sum(r['success'] for r in result['completed']), 4)
            self.assertEqual(w.read(root/'batch/0/gpu_limit.json')['observed_bytes'], 11*2**30)
            self.assertEqual(next(r for r in result['completed'] if r['id'] == '0')['error'], 'PROCESS_GPU_LIMIT_EXCEEDED')
            EVIDENCE['total_process_limit'] = result['completed']

    def test_reservations_and_resource_wait_then_three_admissions(self):
        gib = 2**30
        a = q.admission(dict(free=20*gib, processes={11:2*gib, 12:2*gib}), [11,12], 10*gib)
        self.assertFalse(a['ready'])
        self.assertEqual(a['required_gpu_bytes'], 26*gib)
        self.assertTrue(q.admission(dict(free=26*gib, processes={11:2*gib,12:2*gib}), [11,12], 10*gib)['ready'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class Growing(Memory):
                calls = 0
                def snapshot(self):
                    self.calls += 1
                    v = super().snapshot()
                    if self.calls < 5:
                        v['free'] = 10*gib
                    return v
            with patch.object(q, 'GpuMemory', Growing):
                result = q.dispatch(allowance(root), [dict(id=str(i), seconds=100) for i in range(3)],
                    command, lambda r,p: None, poll_seconds=.02)
            self.assertEqual({r['slot'] for r in result['completed']}, {0,1,2})
            EVIDENCE['pending_reservations'] = a

    def test_single_vendor_slot_and_serial_followups_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def validate(row, job):
                if row['id'] == 'analysis':
                    self.assertEqual(len(w.read(root/'batch/training_complete.json')['completed']), 2)
            with patch.object(q, 'GpuMemory', Memory):
                result = q.dispatch(allowance(root, 1), [dict(id=str(i), seconds=100) for i in range(2)],
                    command, validate, [dict(id='analysis', seconds=90, bucket='analysis', gpu=False)], poll_seconds=.02)
            self.assertEqual([r['id'] for r in result['completed']], ['0','1','analysis'])
            self.assertTrue(all(r['slot'] == 0 for r in result['completed']))
            self.assertEqual(result['status'], 'ALL_FORMAL_ACTIONS_COMPLETE')
            EVIDENCE['serial_followups'] = result['completed']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(q, 'GpuMemory', Memory):
                with self.assertRaisesRegex(RuntimeError, 'Worker failed'):
                    q.dispatch(allowance(root, 1), [dict(id='fail',seconds=100,exit=2), dict(id='later',seconds=100)],
                               command, lambda r,p: None, poll_seconds=.02)
            self.assertFalse((root/'batch/later').exists())

    def test_chain_gate_missing_failed_and_valid_negative_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root/'previous'
            campaign.mkdir()
            gate = dict(campaign=str(campaign), sources=[], runs=[dict(order=o,seed=s,name='F') for o in ('ABC','BCA','CAB') for s in (4,5,6)])
            q.write(root/'gate.json',gate)
            binding = dict(kind='FOREVER',gate=q.record(root/'gate.json'),owner='fixture')
            with patch.object(w,'relevant_processes',return_value=[{'pid':123}]):
                self.assertFalse(w.predecessor_state(binding)['ready'])
            with patch.object(w,'relevant_processes',return_value=[]):
                with self.assertRaisesRegex(RuntimeError,'without complete'):
                    w.predecessor_state(binding)
                ids = [f"{r['order']}_s{r['seed']}_F" for r in gate['runs']]+['generation','confirmation','analysis']
                done = dict(status='ALL_FORMAL_ACTIONS_COMPLETE',sources=[],active={},
                    completed=[dict(id=i,success=True) for i in ids],used={'training':1},limits={'training':100})
                q.write(campaign/'completion.json',done)
                for name,status in [('confirmation','ALL_FOREVER_METRICS_COMPLETE'),('analysis','FOREVER_FIXED_COMPARISON_COMPLETE')]:
                    (campaign/name).mkdir()
                    q.write(campaign/name/'result.json',dict(status=status,passed=False))
                self.assertTrue(w.predecessor_state(binding)['ready'])
                done['completed'].pop()
                q.write(campaign/'completion.json',done)
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    w.predecessor_state(binding)
                q.write(campaign/'failure.json',{})
                with self.assertRaisesRegex(RuntimeError,'failed'):
                    w.predecessor_state(binding)
            EVIDENCE['chain_gate'] = dict(missing_blocks=True, incomplete_blocks=True, failure_blocks=True, scientific_false_can_continue=True)

    def test_hs_full_gate_checks_all_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = dict(campaign=str(root),sources=[],runs=[dict(order=o,seed=s,name=n)
                for o in ('ABC','BCA','CAB') for s in (4,5,6) for n in ('H','S')])
            q.write(root/'gate.json',gate)
            records=[]
            for spec in gate['runs']:
                identity=f"{spec['order']}_s{spec['seed']}_{spec['name']}"
                run=root/identity/'run'
                run.mkdir(parents=True)
                (run/'payload').write_bytes(b'x')
                payload=dict(path='payload',bytes=1)
                point=dict(full_restore_verified=True,inference_restore_verified=True,
                    model=payload,mapping=payload,scores={k:payload for k in ('raw','stage-cal','first-cal')})
                q.write(run/'completion.json',dict(source_files=[],updates=864,points={str(i):point for i in (1,2,3)}))
                rec=q.record(run/'completion.json')
                rec['path']=str((run/'completion.json').relative_to(root))
                records.append(dict(id=identity,completion=rec))
            budget=dict(active={},failed=False,completed=[dict(id=r['id'],success=True) for r in records],used={'training':1},limits={'training':100})
            q.write(root/'completion.json',dict(status='ALL_18_ORIGINAL_RUNS_AND_54_ENDPOINTS_BLIND_COMPLETE',sources=[],endpoints=54,
                evaluation_labels_opened=False,new_confirmation_generated=False,runs=records,budget=budget))
            binding=dict(kind='H/S',gate=q.record(root/'gate.json'),owner='fixture')
            with patch.object(w,'relevant_processes',return_value=[]):
                self.assertTrue(w.predecessor_state(binding)['ready'])
                (run/'payload').unlink()
                with self.assertRaises(FileNotFoundError):
                    w.predecessor_state(binding)
            EVIDENCE['hs_full_gate'] = dict(handwritten_runs=18,points=54,missing_payload_rejected=True)

    def test_wait_integrated_and_listener_deletes_only_after_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            listener = root/'listener.py'
            listener.write_text('retained handwritten source')
            allowed = allowance(root,1)
            allowed.update(predecessor={'fixture':True}, listener=str(listener),listener_source=q.record(listener))
            with patch.object(q,'GpuMemory',Memory), patch.object(w,'predecessor_state',side_effect=RuntimeError('blocked predecessor')):
                with self.assertRaisesRegex(RuntimeError,'blocked predecessor'):
                    q.dispatch(allowed,[dict(id='must_not_run',seconds=100)],command,lambda r,p:None)
            self.assertFalse((root/'batch/must_not_run').exists())
            self.assertTrue(listener.exists())
            log = root/'batch/real/run/updates.jsonl'
            log.parent.mkdir(parents=True)
            log.write_text(json.dumps({'role':'current','counts':{'current':1}})+'\n')
            w.observe_first_update(allowed,root/'batch')
            self.assertFalse(listener.exists())
            self.assertTrue((root/'batch/first_update.json').exists())
            EVIDENCE['predecessor_entry_and_self_delete'] = True


if __name__ == '__main__':
    if sys.platform != 'linux':
        raise SystemExit('Linux py310 only')
    suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:], sys.modules[__name__]) if len(sys.argv)>1 else unittest.defaultTestLoader.loadTestsFromTestCase(FormalWaitTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    q.write(Path(os.environ['FORMAL_TEST_EVIDENCE']),dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),evidence=EVIDENCE))
    raise SystemExit(not result.wasSuccessful())
