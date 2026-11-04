"""Linux-only handwritten dispatch checks; no CUDA/model/data access."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import step29_english_gpu_wait as wait


class Snapshot:
    def __init__(self, values):
        self.values, self.calls = list(values), 0

    def snapshot(self):
        self.calls += 1
        value = self.values.pop(0) if len(self.values) > 1 else self.values[0]
        return dict(free=value, total=32*2**30, processes={})


class WaitTests(unittest.TestCase):
    def test_below_threshold_waits_and_accepts_exact_15_gib(self):
        memory = Snapshot([wait.THRESHOLD-1, wait.THRESHOLD])
        seen = []
        budget = wait.Auxiliary(14)
        with patch.object(wait.time, 'sleep') as sleep:
            answer = wait.wait_ready(memory, budget, seen.append)
        self.assertEqual(answer['free'], 15360*2**20)
        self.assertEqual(len(seen), 1)
        sleep.assert_called_once_with(.1)

    def exercise(self, fail_second):
        with tempfile.TemporaryDirectory(dir=os.environ.get('ABC_TEST_ROOT')) as temporary:
            root = Path(temporary)
            listener = root/'active_gpu_wait.py'
            listener.write_text('handwritten active copy')
            permit = dict(root=str(root/'probes'), workspace=str(root), sources=[], model_files=[],
                          prior_auxiliary_seconds=14, prior_gpu_seconds=14,
                          single_process_gpu_bytes=int(14.75*2**30), listener=wait.queue.record(listener))
            marker = root/'started.txt'
            def command(pooling, job, deadline):
                # Real subprocesses exercise launch order, errors and one-shot state.
                code = ("import json,pathlib,sys,os; p=pathlib.Path(sys.argv[1]); p.mkdir(); "
                        "f=pathlib.Path(sys.argv[2]); f.open('a').write(sys.argv[3]+'\\n'); "
                        "u={'pid':os.getpid(),'intervals':[{'begin':1,'end':2,'logs':{'optimizer_updates':1}}]}; "
                        "(p/'updates.json').write_text(json.dumps(u)); "
                        "v={'status':'HANDWRITTEN_GPU_COMPLETE','updates':2,'changed':{'encoder':True,'head':True},"
                        "'formal_data_reads':0,'pooling':sys.argv[3]}; (p/'result.json').write_text(json.dumps(v)); "
                        "sys.exit(int(sys.argv[4]))")
                return [sys.executable, '-B', '-c', code, str(job), str(marker), pooling,
                        '2' if fail_second and pooling == 'mean_last' else '0']
            # Initial detection followed by a lost window must return to waiting.
            memory = Snapshot([wait.THRESHOLD, wait.THRESHOLD-1, wait.THRESHOLD-1, wait.THRESHOLD])
            with patch.object(wait.psutil, 'virtual_memory', return_value=type('RAM', (), {'available':64*2**30})()):
                if fail_second:
                    with self.assertRaisesRegex(RuntimeError, 'exit 2'):
                        wait.execute(permit, memory, command)
                else:
                    wait.execute(permit, memory, command)
                with self.assertRaises(FileExistsError):
                    wait.execute(permit, memory, command)
            ledger = json.loads((root/'state/status.json').read_text())
            started = marker.read_text().splitlines()
            self.assertEqual(started, list(wait.POOLINGS[:2 if fail_second else 3]))
            self.assertEqual(ledger['status'], 'FAILED_NO_RETRY' if fail_second else 'SINGLE_GPU_CHECKS_COMPLETE')
            self.assertEqual(len(ledger['completed']), 1 if fail_second else 3)
            self.assertGreater(ledger['new_gpu_seconds'], 0)
            self.assertEqual(ledger['cumulative_gpu_seconds'], 14+ledger['new_gpu_seconds'])
            self.assertIsNone(ledger['active'])
            self.assertFalse(ledger['priority_guaranteed'])
            self.assertFalse(listener.exists())
            self.assertTrue(json.loads((root/'state/first_update.json').read_text())['listener_deleted'])

    def test_no_self_delete_until_actual_update_receipt(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('ABC_TEST_ROOT')) as temporary:
            root=Path(temporary)
            (root/'state').mkdir()
            job=root/'probe';job.mkdir()
            listener=root/'active_gpu_wait.py';listener.write_text('active copy')
            retained=root/'retained.py';retained.write_text('active copy')
            permit=dict(root=str(root/'probes'),listener=wait.queue.record(listener))
            wait.observe_update(permit,job,17)
            self.assertTrue(listener.exists())
            value=dict(pid=17,intervals=[dict(begin=1,end=2,logs={'optimizer_updates':0})])
            (job/'updates.json').write_text(json.dumps(value))
            wait.observe_update(permit,job,17)
            self.assertTrue(listener.exists())
            value['intervals'][0]['logs']['optimizer_updates']=1
            (job/'updates.json').write_text(json.dumps(value))
            wait.observe_update(permit,job,17)
            self.assertFalse(listener.exists())
            self.assertTrue(retained.exists())

    def test_update_lands_at_monitor_error_is_observed_after_cleanup(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('ABC_TEST_ROOT')) as temporary:
            root=Path(temporary)
            listener=root/'active_gpu_wait.py';listener.write_text('active copy')
            permit=dict(root=str(root/'probes'),workspace=str(root),sources=[],model_files=[],
                        prior_auxiliary_seconds=18,prior_gpu_seconds=14,
                        single_process_gpu_bytes=int(14.75*2**30),listener=wait.queue.record(listener))
            class LateUpdate(Snapshot):
                def snapshot(self):
                    value=super().snapshot()
                    if self.calls==3:
                        active=json.loads((root/'state/status.json').read_text())['active']
                        job=Path(active['job']);job.mkdir()
                        wait.queue.write(job/'updates.json',dict(pid=active['pid'],intervals=[
                            dict(begin=1,end=2,logs={'optimizer_updates':1})]))
                        raise RuntimeError('handwritten monitor failure after update')
                    return value
            command=lambda *args: [sys.executable,'-c','import time; time.sleep(10)']
            with patch.object(wait.psutil,'virtual_memory',return_value=type('RAM',(),{'available':64*2**30})()):
                with self.assertRaisesRegex(RuntimeError,'handwritten monitor failure'):
                    wait.execute(permit,LateUpdate([wait.THRESHOLD]),command)
            self.assertFalse(listener.exists())
            status=json.loads((root/'state/status.json').read_text())
            self.assertIn('handwritten monitor failure',status['error'])
            self.assertEqual(status['status'],'FAILED_NO_RETRY')
            self.assertGreater(status['new_gpu_seconds'],0)
            self.assertIsNone(status['active'])

    def test_three_serial_children_and_no_duplicate_start(self):
        self.exercise(False)

    def test_second_child_failure_stops_without_retry_or_third_child(self):
        self.exercise(True)


if __name__ == '__main__':
    if sys.platform != 'linux':
        raise SystemExit('Research checks require Linux py310')
    unittest.main(verbosity=2)
