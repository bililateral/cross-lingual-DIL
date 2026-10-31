"""Real orphaned CPU workers: repaired slot refill, failure isolation, fee carry."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import psutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import step28_comparison_continue as c


class ContinueTest(unittest.TestCase):
    def setup_case(self, root, fail):
        old,repair,campaign=root/'old',root/'repair',root/'continued'
        old.mkdir();repair.mkdir()
        pin=lambda p:{**c.data.record(p,root),'path':str(p)}
        ids={s.run_id:s for s in c.m.specifications(new_only=True)}
        templates={identity:dict(status=c.run.COMPLETE,updates=864,spec=spec.identity(),
            source_files=['fixed'],access=dict(train=1,valid=0,heldout=0,owners=0,worker_train_reads=1),
            points={str(i):dict(adam_step=i*288,full_restore_verified=True,inference_restore_verified=True) for i in (1,2,3)})
            for identity,spec in ids.items()}
        c.shared.write(root/'templates.json',templates)
        script=root/'orphan.py'
        script.write_text("import os,sys,json,time\nfrom pathlib import Path\n"
            "root=Path(sys.argv[1]);identity=sys.argv[2];job=Path(sys.argv[3]);job.mkdir()\n"
            "(job/'start.json').write_text(json.dumps({'time':time.monotonic(),'cpu':list(os.sched_getaffinity(0))}))\n"
            "time.sleep(float(sys.argv[4]))\n"
            "if sys.argv[5]=='fail':\n"
            " (job/'failure.json').write_text('{}');raise SystemExit(3)\n"
            "(job/'run').mkdir();(job/'run/completion.json').write_text(json.dumps(json.loads((root/'templates.json').read_text())[identity]))\n"
            "(job/'end.json').write_text(json.dumps({'time':time.monotonic()}))\n")
        pending_script=old/'workspace/scripts/step28_continual_comparison_run.py'
        pending_script.parent.mkdir(parents=True)
        pending_script.write_text("import sys,json,os,time\nfrom pathlib import Path\n"
            "a=sys.argv;job=Path(a[a.index('--job')+1]);job.mkdir()\n"
            "(job/'start.json').write_text(json.dumps({'time':time.monotonic(),'cpu':list(os.sched_getaffinity(0))}))\n"
            "root=job.parent.parent;assert not (root/'gcr_started_once').exists();(root/'gcr_started_once').write_text('once')\n"
            "(job/'run').mkdir();(job/'run/completion.json').write_text(json.dumps(json.loads((root/'templates.json').read_text())[job.name]))\n")
        post=root/'post.py'
        post.write_text("import sys\nfrom pathlib import Path\n"
            "p=Path(sys.argv[sys.argv.index('--campaign')+1]);assert (p/'training_complete.json').exists()\n"
            "with (p/'post_order.txt').open('a') as f:f.write(sys.argv[1]+'\\n')\n")
        old_gate=dict(campaign=str(old),workspace=str(old/'workspace'),sources=['fixed'])
        repair_gate=dict(campaign=str(repair),workspace=str(root),sources=['fixed'])
        c.shared.write(root/'old_gate.json',old_gate);c.shared.write(root/'repair_gate.json',repair_gate)
        adoption={identity:dict(completion=str((repair if identity==c.TARGET else old)/identity/'run/completion.json'),
                      source_gate=pin(root/('repair_gate.json' if identity==c.TARGET else 'old_gate.json'))) for identity in ids}
        done=set(ids)-c.ACTIVE-{c.PENDING}
        for identity in done:
            p=old/identity/'run/completion.json';p.parent.mkdir(parents=True);c.shared.write(p,templates[identity])
        cores=psutil.Process().cpu_affinity()[:3]
        jobs=[]
        ordered=[c.TARGET,'CAB_s4_L2R_text_fixed','CAB_s5_L2R_text_fixed']
        started=time.monotonic()-5
        for slot,identity in enumerate(ordered):
            jobs.append(dict(id=identity,job=str((repair if identity==c.TARGET else old)/identity),slot=slot,
                command=['taskset','-c',str(cores[slot]),sys.executable,'-B',str(script),str(root),identity,
                         str((repair if identity==c.TARGET else old)/identity),str(1.8+slot*.6),
                         'fail' if fail and slot==0 else 'ok']))
        c.shared.write(root/'jobs.json',{'jobs':jobs})
        parent_code=("import subprocess,json,sys,psutil;from pathlib import Path;"
                     "rows=json.loads(Path(sys.argv[1]).read_text())['jobs'];out=[]\n"
                     "for row in rows:\n"
                     " p=subprocess.Popen(row['command'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);q=psutil.Process(p.pid)\n"
                     " out.append(dict(id=row['id'],job=row['job'],slot=row['slot'],pid=p.pid,created=q.create_time(),owner=q.username(),cwd=q.cwd()))\n"
                     "print(json.dumps(out),flush=True)\n")
        process=subprocess.run([sys.executable,'-B','-c',parent_code,str(root/'jobs.json')],capture_output=True,text=True,check=True)
        workers={r['id']:r for r in json.loads(process.stdout)}
        for record in workers.values():
            q=psutil.Process(record['pid'])
            # taskset exec is brief; record the actual worker command after it has run.
            deadline=time.monotonic()+2
            while not (Path(record['job'])/'start.json').exists():
                if time.monotonic()>deadline:raise RuntimeError('Handwritten child did not start')
                time.sleep(.01)
            record['command']=q.cmdline()
        reservation=lambda k:dict(id=k,bucket='training',started=started,deadline=started+120,allowance=120)
        budget=dict(used={k:0. for k in c.LIMITS},completed=[dict(id=k,success=True) for k in done]+[dict(id=c.TARGET,success=False)],
                    active={k:reservation(k) for k in c.ACTIVE-{c.TARGET}},failed=True,budget_exceeded=False)
        budget['used'].update(training=100,cpu=98)
        new=dict(used={k:0. for k in c.LIMITS},completed=[],active={c.TARGET:reservation(c.TARGET)},failed=False,budget_exceeded=False)
        new['used']['cpu']=8
        claim=dict(budgets=dict(donor=budget,repair=new),workers=workers,controllers={})
        c.shared.write(root/'claim.json',claim)
        gate=dict(takeover=pin(root/'claim.json'),donor_gate=pin(root/'old_gate.json'),repair_gate=pin(root/'repair_gate.json'),
                  campaign=str(campaign),pending_ids=[c.PENDING],adoption=adoption,additional_used_seconds={},sources=['fixed'])
        c.shared.write(root/'gate.json',gate)
        return gate,claim,post

    def run_case(self, fail):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);gate,claim,post=self.setup_case(root,fail);campaign=Path(gate['campaign'])
            original_dispatch=c.shared.dispatch
            def fast(*args,**kwargs):
                return original_dispatch(*args,**kwargs,poll_seconds=.02)
            def publish_ready(path,value):
                # Use the real write; only the external transfer signal is a fixture.
                original_write(path,value)
                if path.name=='ready.json':
                    original_write(campaign/'activate.json',{'takeover':gate['takeover']})
            original_write=c.shared.write
            before={p:p.read_bytes() for p in (root/'old').glob('*/run/completion.json')}
            try:
                with patch.object(c,'validate',return_value=gate),patch.object(c,'resources',return_value=set()), \
                     patch.object(c.shared,'resources',return_value=True),patch.object(c.shared,'dispatch',side_effect=fast), \
                     patch.object(c.shared,'write',side_effect=publish_ready),patch.object(c.evaluation,'__file__',str(post)), \
                     patch.object(c.evaluation,'full_gate',return_value=list(range(36))) as complete:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'after active siblings finished'):
                            c.execute(root/'gate.json')
                        complete.assert_not_called()
                    else:
                        c.execute(root/'gate.json')
                        complete.assert_called_once()
                starts={k:c.data.read_json(Path(v['job'])/'start.json') for k,v in claim['workers'].items()}
                gcr=c.data.read_json(root/'old'/c.PENDING/'start.json')
                budget=c.data.read_json(campaign/'budget.json')
                self.assertEqual(budget['used']['cpu'],106)
                self.assertGreater(budget['used']['training'],115)
                self.assertEqual(before,{p:p.read_bytes() for p in before})
                outcome=c.data.read_json(campaign/'dispatch'/c.TARGET/'outcome.json')
                self.assertIsNone(outcome['actual_exit_code'])
                self.assertEqual(outcome['success'],not fail)
                if fail:
                    self.assertNotEqual(gcr['cpu'],starts[c.TARGET]['cpu'])
                    self.assertEqual(sum(not r['success'] for r in budget['completed']),1)
                    self.assertFalse((campaign/'completion.json').exists())
                    self.assertFalse((campaign/'post_order.txt').exists())
                else:
                    self.assertEqual(gcr['cpu'],starts[c.TARGET]['cpu'])
                    self.assertGreaterEqual(gcr['time'],c.data.read_json(root/'repair'/c.TARGET/'end.json')['time'])
                    self.assertEqual((campaign/'post_order.txt').read_text(),'generate\nconfirmation\nanalyze\n')
                    self.assertEqual(len(budget['completed']),21)
                    self.assertEqual(len(list(campaign.glob('*/run/completion.json'))),18)
                    for identity in gate['adoption']:
                        self.assertEqual(c.evaluation.trajectory_sources(gate,identity,campaign/identity/'run/completion.json'),['fixed'])
                print('Attached-slot evidence:',dict(failed_repair=fail,repair_cpu=starts[c.TARGET]['cpu'],
                      subsequent_gcr_cpu=gcr['cpu'],gcr_single_dispatch=True,original_fees_kept=True),flush=True)
            finally:
                for record in claim['workers'].values():
                    process=c.exact_process(record)
                    if process is not None:
                        os.killpg(process.pid,signal.SIGKILL)

    def test_repaired_successful_slot_receives_next_real_task(self):
        self.run_case(False)

    def test_repaired_failed_slot_retires_healthy_slots_still_refill(self):
        self.run_case(True)

    def test_missing_completion_is_failure_without_fabricating_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            worker=c.AttachedWorker(dict(pid=99999999,created=0,job=directory,id=c.TARGET),{'sources':['fixed']})
            self.assertEqual(worker.poll(),'EXITED')
            self.assertFalse(worker.successful)
            self.assertIsNone(worker.returncode)
            self.assertIn('error',worker.completion_check)

    def test_exit_during_identity_read_is_distinct_from_live_identity_change(self):
        class Process:
            def __init__(self, exits):
                self.calls=0;self.exits=exits
            def create_time(self):return 1.
            def status(self):
                self.calls+=1
                return c.psutil.STATUS_ZOMBIE if self.exits and self.calls>1 else 'running'
            def username(self):return 'owner'
            def cwd(self):return '/fixed'
            def cmdline(self):return []
            def is_running(self):return True
        record=dict(pid=123,created=1.,owner='owner',cwd='/fixed',command=['worker'])
        with patch.object(c.psutil,'Process',return_value=Process(True)):
            self.assertIsNone(c.exact_process(record))
        with patch.object(c.psutil,'Process',return_value=Process(False)):
            with self.assertRaisesRegex(RuntimeError,'identity changed'):
                c.exact_process(record)


if __name__=='__main__':
    unittest.main()
