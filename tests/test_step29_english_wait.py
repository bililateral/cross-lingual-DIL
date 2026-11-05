"""Handwritten metadata and real CPU child checks; no model, data or CUDA."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import step29_english_wait as wait
import step29_english_control as control


class WaitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def history(self, anchor=True):
        root, state = self.root/'history', self.root/'state'
        root.mkdir(); state.mkdir()
        gate = dict(campaign=str(root), sources=[dict(path='frozen')],
                    prior_used_seconds=dict(training=0, confirmation=0, auxiliary=10))
        wait.write(self.root/'gate.json', gate)
        binding = dict(gate=control.queue.record(self.root/'gate.json'), state=str(state))
        used = dict(training=100, confirmation=0 if anchor else 10, auxiliary=50)
        wait.write(root/'budget.json', dict(used_seconds=used))
        wait.write(root/'completion.json', dict(sources=gate['sources'], used_seconds=used,
            selected=dict(number=0 if anchor else 3, parameters=dict(history_weight=.1 if anchor else .2, teacher_weight=.5)),
            status='COMPLETE_ANCHOR_SELECTED_NO_NEW_CONFIRMATION' if anchor else 'COMPLETE_PENDING_RESULT_REVIEW_MODELS_RETAINED'))
        wait.write(state/'completion.json', dict(status='FORMAL_CONTROLLER_COMPLETED'))
        def dispatch(folder, ids):
            folder.mkdir(parents=True, exist_ok=True)
            wait.write(folder/'dispatch.json', dict(status='ALL_COMPLETE_RELEASED', sources=gate['sources'],
                pending=[], active={}, retired_slots=[], completed=[dict(id=i, success=True, error=None,
                    released=True, actual_seconds=1) for i in ids]))
        for n in range(12):
            dispatch(root/f'trial_{n:02}', [f'search_{n:02}_{o}_s7_H_pos_history' for o in ('ABC','BCA','CAB')])
            dispatch(root/f'trial_{n:02}/evaluation_dispatch', ['evaluation'])
        if not anchor:
            dispatch(root/'replication', [f'replication_03_{o}_s{s}_H_pos_history' for o in ('ABC','BCA','CAB') for s in (4,5,6)])
            for folder, job in ((root/'replication','evaluation'),(root,'generation'),(root,'confirmation'),(root,'analysis')):
                dispatch(folder/(job+'_dispatch'), [job])
        return binding, root, state

    def test_whole_history_and_failure(self):
        binding, root, state = self.history(False)
        self.assertFalse(wait.predecessor(binding, lambda _: [123])['ready'])
        self.assertTrue(wait.predecessor(binding, lambda _: [])['ready'])
        missing = root/'analysis_dispatch/dispatch.json'
        original = missing.read_bytes(); missing.unlink()
        with self.assertRaises(FileNotFoundError): wait.predecessor(binding, lambda _: [])
        missing.write_bytes(original)
        bad = wait.read(root/'trial_11/dispatch.json'); bad['completed'].pop()
        wait.write(root/'trial_11/dispatch.json', bad)
        with self.assertRaises(ValueError): wait.predecessor(binding, lambda _: [])
        wait.write(state/'failure.json', dict(error='synthetic'))
        with self.assertRaises(RuntimeError): wait.predecessor(binding, lambda _: [123])

    def test_anchor_and_real_process_exit(self):
        binding, root, state = self.history()
        child = subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'], cwd=root)
        try:
            import psutil
            p = psutil.Process(child.pid)
            binding.update(process=dict(pid=p.pid, created=p.create_time(), owner=p.username(),
                                        command=p.cmdline(), cwd=p.cwd()), config='none1',listener='none2')
            self.assertFalse(wait.predecessor(binding)['ready'])
        finally:
            child.terminate(); child.wait(timeout=3)
        self.assertTrue(wait.predecessor(binding)['ready'])
        (state/'completion.json').unlink()
        with self.assertRaises(RuntimeError): wait.predecessor(binding)

    def test_actual_runner_callback_and_retirement(self):
        root = self.root/'campaign'; root.mkdir()
        state = self.root/'state'; state.mkdir()
        listener = self.root/'active.py'; listener.write_text('synthetic active copy\n')
        config = dict(listener=str(listener), state=str(state), active_source=control.queue.record(listener))
        allowed = dict(campaign=str(root), workspace=str(self.root), workers=2, limits=control.formal.LIMITS)
        ledger = control.Ledger(root, dict(training=0,gpu=0,auxiliary=0), control.formal.CAPS)
        runner = control.Runner(allowed, ledger, None, poll_seconds=.02,
                               observe=lambda: wait.observe_update(config, root))
        wait.observe_update(config, root); self.assertTrue(listener.exists())
        payload = "from pathlib import Path;import json,sys,time;p=Path(sys.argv[1])/'run';p.mkdir();(p/'updates.jsonl').write_text(json.dumps(dict(stage=1,total_updates=1,optimizer_updates=1))+'\\n');time.sleep(.1)"
        runner.dispatch(root, [dict(id='handwritten_cpu',bucket='training',seconds=65,gpu=False)],
            lambda row, job, deadline: [sys.executable,'-c',payload,str(job)], lambda row, job: None)
        self.assertFalse(listener.exists())
        self.assertEqual(wait.read(state/'retirement.json')['reason'], 'FIRST_REAL_OPTIMIZER_UPDATE')
        self.assertEqual(wait.read(root/'dispatch.json')['status'], 'COMPLETE')

    def test_wait_activation_and_precheck_stop(self):
        state, listener = self.root/'state', self.root/'active.py'
        listener.write_text('synthetic listener\n')
        authorization=self.root/'auth.json'; wait.write(authorization,dict(approved=True))
        review=self.root/'review.txt'; review.write_text('synthetic closed review')
        gate=dict(workspace=str(self.root),campaign=str(self.root/'formal'),
                  prior_used_seconds=dict(training=0,gpu=37,auxiliary=10), authorization=control.queue.record(authorization))
        template=self.root/'gate.json';wait.write(template,gate)
        config=dict(approved=True,review_closed=True,state=str(state),listener=str(listener),
                    prior_auxiliary_seconds=10,authorization=gate['authorization'],review_evidence=control.queue.record(review),
                    gate_template=control.queue.record(template),active_source=control.queue.record(listener),predecessor={})
        path=self.root/'config.json';wait.write(path,config)
        clock=[0.]; sleeps=[]; calls=[]
        class Monitor:
            def snapshot(self): return {}
            def close(self): pass
        def sleep(seconds): sleeps.append(seconds);clock[0]+=seconds
        def launch(path, observe=None):
            calls.append(wait.read(path));Path(gate['campaign']).mkdir()
            wait.write(Path(gate['campaign'])/'completion.json',dict(status='STOP_PRECHECK_NO_VENDOR_DREAM_FORGETTING'))
        cwd, argv=os.getcwd(),sys.argv[:]
        try:
            with patch.object(wait,'__file__',str(listener)), patch.object(wait,'predecessor',side_effect=[dict(ready=False),dict(ready=True),dict(ready=True)]), patch.object(wait.time,'monotonic',lambda:clock[0]), patch.object(wait.time,'sleep',sleep), patch.object(control.formal,'gate',return_value=gate), patch.object(control.queue,'GpuMemory',Monitor), patch.object(control.queue,'resources',side_effect=[dict(ready=False),dict(ready=True)]), patch.object(control,'execute',launch):
                wait.execute(path)
                with self.assertRaises(FileExistsError): wait.execute(path)
        finally:
            os.chdir(cwd);sys.argv=argv
        self.assertEqual(sleeps,[5,5]);self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['prior_used_seconds']['auxiliary'],10)
        self.assertFalse(listener.exists())
        self.assertEqual(wait.read(state/'retirement.json')['reason'],'PRECHECK_ENDED_WITHOUT_TRAINING')


if __name__=='__main__':
    unittest.main(verbosity=2)
