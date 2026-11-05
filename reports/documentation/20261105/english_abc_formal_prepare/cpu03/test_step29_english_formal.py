"""Incremental Linux CPU checks of the actual new ABC batch wiring."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

import step29_english_formal as f
import step29_english_control as c
from test_step29_english_abc import group, tiny

EVIDENCE = {}


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['ABC_TEST_ROOT'])
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def allowed(self):
        return dict(campaign=str(self.root), workspace=str(self.root), workers=2,
            limits=dict(rss_bytes=4*2**30,disk_bytes=10*2**30,small_bytes=2**30,
                        free_reserve_bytes=0,free_gpu_bytes=f.SLOT_BYTES,process_gpu_bytes=f.SLOT_BYTES))

    def test_real_train_restore_dev_scoring_and_gate_rejection(self):
        allowed = self.allowed()
        allowed.update(epochs=2,sources=[],fit_counts=dict.fromkeys(f.DOMAINS,1),
                       dev_counts=dict.fromkeys(f.DOMAINS,1),dev_queries=dict.fromkeys(f.DOMAINS,8),
                       fit_sources={},dev_sources={},dev_qrels={})
        for d in f.DOMAINS:
            g = group(d)
            for role,value in (('fit_sources',[asdict(g)]),
                               ('dev_sources',[asdict(replace(g,split='dev').blind())]),
                               ('dev_qrels',{g.uid:dict(view_ids=[v.uid for v in g.evaluation_order().views],
                                                       labels=g.evaluation_order().labels())})):
                p=self.root/f'{d}_{role}.json'; f.queue.write(p,value)
                allowed[role][d]=f.queue.record(p)
        shape = replace(f.policy.baseline(),pooling='mean_last4_sum',head_schedule='encoder',
                        bce_weight=.25,hard_weight=0.,channel_tokens=16,vendor_tokens=24,microbatch=64)
        row=f.spec(shape,7)
        job=self.root/row['id'];job.mkdir()
        def factory(a,settings):return tiny(self.root,settings)
        budget=f.Budget(allowed,time.monotonic()+1000)
        f.train(job,row,allowed,budget,factory)
        complete=f.validate_completed(row,job/'run',allowed)
        self.assertFalse((job/'run/latest.pt').exists())
        self.assertEqual(complete['access']['fit_domains'],list(f.DOMAINS))
        batch=dict(name='fixture',training_root=str(self.root),runs=[row])
        evaluation=self.root/'evaluation';evaluation.mkdir()
        value=f.evaluate(evaluation,batch,allowed,budget,factory)
        m=f.validate_evaluation(value,[row],'fixture',allowed)
        self.assertEqual(len(value['runs'][row['id']]['cells']),6)
        self.assertEqual(f.read(evaluation/'access.json'),dict(dev_text_attempted=True,dev_qrels_attempted=True,heldout=0,owners=0))
        # A missing endpoint fails before a second evaluator opens even dev text.
        (job/'run/stage2.pt').unlink()
        refused=self.root/'refused';refused.mkdir()
        with self.assertRaises(FileNotFoundError):f.evaluate(refused,batch,allowed,budget,factory)
        self.assertFalse((refused/'access.json').exists())
        # Actual CLI refuses a preparation gate, without model or data paths present.
        gate=self.root/'gate.json';f.queue.write(gate,dict(approved=False))
        result=subprocess.run([sys.executable,'-B',c.__file__,'--gate',str(gate)],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('explicitly approved',result.stderr)
        EVIDENCE['actual_worker_and_evaluator']=dict(updates=complete['updates'],cells=6,metrics=m,
            closed_cli_rejected=True,missing_endpoint_refused_before_dev=True,access=f.read(evaluation/'access.json'))

    def test_query_weighting_seed_and_old_default_boundaries(self):
        row=f.spec(f.policy.baseline(),7)
        allowed=dict(dev_counts=dict.fromkeys(f.DOMAINS,1),dev_queries=dict.fromkeys(f.DOMAINS,8))
        cells={f'{a}_{b}':dict(domain=f.DOMAINS[b-1],groups=1,query_count=8,metrics=[.5]*10) for a,b in f.CELLS}
        value=dict(status='COMPLETE_DEV_ROUND',name='test',columns=f.METRICS,runs={row['id']:dict(spec=row,cells=cells)})
        f.validate_evaluation(value,[row],'test',allowed)
        wrong=json.loads(json.dumps(value));wrong['runs'][row['id']]['spec']['seed']=8
        with self.assertRaisesRegex(ValueError,'pairing'):f.validate_evaluation(wrong,[row],'test',allowed)
        wrong=json.loads(json.dumps(value));wrong['runs'][row['id']]['cells']['3_3']['query_count']=7
        with self.assertRaisesRegex(ValueError,'weighting'):f.validate_evaluation(wrong,[row],'test',allowed)
        old=asdict(f.Settings('H_full'))
        for key in ('pooling','head_schedule','bce_weight','hard_weight'):old.pop(key)
        self.assertEqual(asdict(f.Settings(**old)),asdict(f.Settings('H_full')))
        # Independent ranking: first query's one positive is second, AP=1/2.
        result=f.query_metrics([.4,.9,.1,.1,.9,.4],[1,0,0,0,0,1],['a','b','c','d'])
        self.assertEqual(result[0,0],.5)
        EVIDENCE['pairing_and_metrics']=dict(wrong_seed_rejected=True,wrong_query_count_rejected=True,
            old_settings_defaults=asdict(f.Settings(**old)),hand_rank_ap=float(result[0,0]))

    def test_full_finite_pipeline_and_precheck_stop(self):
        base=dict(B2=.85,B3=.75,C3=.9,A3=.9)
        vendor=dict(B2=.86,B3=.76,C3=.9,A3=.9)
        old={f'ABC_s{s}_{m}':vendor for s in (4,5,6) for m in ('H_full','H_ER','H_SEQ','VendorLink_SEQ')}
        calls=[];saves={}
        def action(name,rows):
            calls.append((name,rows))
            out={}
            for r in rows:
                settings=r['settings']
                if settings['method']=='VendorLink_SEQ':value=vendor
                elif settings['pooling']=='cls':value=base
                else:value=dict(B2=.85,B3=.84 if settings['pooling']=='mean_last4_sum' else .83,C3=.9,A3=.9)
                out[r['id']]=value
            return out
        result=c.pipeline(lambda:old,action,lambda name,value:saves.update({name:value}))
        self.assertTrue(result['verdict']['passed'])
        self.assertEqual(result['selected'],'P2')
        self.assertEqual([name for name,_ in calls],['representations','L1','L2','L3','T1','T2','seed8','final'])
        all_rows=[r for _,rows in calls for r in rows]
        self.assertEqual(len(all_rows),28)
        self.assertEqual(len({r['id'] for r in all_rows}),28)
        self.assertEqual({r['seed'] for r in calls[-1][1]},{9,10,11})
        self.assertEqual(len(calls[-1][1]),15)
        self.assertEqual(saves['frozen']['settings']['pooling'],'mean_last4_sum')
        no_forgetting={k:dict(v,B3=v['B2']) for k,v in old.items()}
        stop=c.pipeline(lambda:no_forgetting,lambda *a:self.fail('No training after failed precheck'),lambda *a:None)
        self.assertEqual(stop['status'],'STOP_PRECHECK_NO_VENDOR_DREAM_FORGETTING')
        def missing(name,rows):return {r['id']:base for r in rows[1:]}
        with self.assertRaisesRegex(ValueError,'paired'):c.pipeline(lambda:old,missing,lambda *a:None)
        EVIDENCE['finite_pipeline']=dict(rounds=[dict(name=n,count=len(r)) for n,r in calls],
            selected=result['selected'],verdict=result['verdict'],precheck_stop=stop)

    def test_real_two_slot_failure_health_and_persistent_retirement(self):
        allowed=self.allowed()
        ledger=c.Ledger(self.root,dict(training=0,gpu=0,auxiliary=0),dict(training=100,gpu=100,auxiliary=100))
        class Monitor:
            def snapshot(self):return dict(free=32*2**30,total=32*2**30,processes={})
        runner=c.Runner(allowed,ledger,Monitor(),poll_seconds=.01)
        folder=self.root/'round';folder.mkdir()
        code="import json,sys,time;from pathlib import Path;p=Path(sys.argv[1]);time.sleep(.05);(p/'ran').write_text('done');sys.exit(int(sys.argv[2]))"
        def command(row,job,deadline):return [sys.executable,'-c',code,str(job),str(row.get('exit',0))]
        rows=[dict(id=str(i),bucket='training',seconds=3,shutdown_reserve=.1,exit=2 if i==0 else 0) for i in range(5)]
        with self.assertRaisesRegex(RuntimeError,'Incomplete round'):
            runner.dispatch(folder,rows,command,lambda r,p:self.assertTrue((p/'ran').exists()))
        observed=f.read(folder/'dispatch.json')
        self.assertEqual(len(observed['completed']),5)
        self.assertEqual(runner.retired,{0})
        self.assertTrue(all(r['slot']==1 for r in observed['completed'] if int(r['id'])>=2))
        # A separate caller cannot inadvertently restore slot 0 by reusing this runner.
        next_round=self.root/'next';next_round.mkdir()
        runner.dispatch(next_round,[dict(id='next',bucket='training',seconds=3,shutdown_reserve=.1)],command,lambda *a:None)
        self.assertEqual(f.read(next_round/'dispatch.json')['completed'][0]['slot'],1)
        self.assertEqual(ledger.used['training'],6)  # Each of six short child lifecycles rounds up.
        EVIDENCE['real_dispatch']=dict(first=observed,second=f.read(next_round/'dispatch.json'),budget=ledger.used)

    def test_shared_scan_first_update_and_unreflected_reservation(self):
        allowed=self.allowed()
        budget=f.Budget(allowed,time.monotonic()+100)
        class Gone:
            suffix='.pt'
            def is_file(self):return True
            def is_symlink(self):return False
            def stat(self):raise FileNotFoundError('Concurrent completed-checkpoint retirement')
        with patch.object(Path,'rglob',return_value=iter([Gone()])):
            budget.check()  # Actual callback -> queue.campaign_files, not a job-only substitute.
        job=self.root/'job';(job/'run').mkdir(parents=True)
        update=dict(stage=1,total_updates=1,optimizer_updates=1)
        (job/'run/updates.jsonl').write_text(json.dumps(update)+'\n')
        from types import SimpleNamespace
        c.observe_first_update(self.root,{1:dict(job=job,process=SimpleNamespace(pid=123),created=42)})
        self.assertEqual(f.read(self.root/'first_update.json')['update'],update)
        reservation=f.queue.admission(dict(free=20*2**30,processes={123:2*2**30}),[123],f.SLOT_BYTES)
        self.assertFalse(reservation['ready'])
        self.assertEqual(reservation['pending_gpu_bytes'],f.SLOT_BYTES-2*2**30)
        EVIDENCE['shared_scan_and_receipt']=dict(retirement_race_accepted=True,first_update=f.read(self.root/'first_update.json'),reservation=reservation)

    def test_launch_failure_retires_only_that_slot(self):
        allowed=self.allowed()
        ledger=c.Ledger(self.root,dict(training=0,gpu=0,auxiliary=0),dict(training=100,gpu=100,auxiliary=100))
        runner=c.Runner(allowed,ledger,None,poll_seconds=.01)
        folder=self.root/'launch';folder.mkdir()
        actual=subprocess.Popen
        calls=0
        def launch(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==1:raise OSError('Handwritten process creation failure')
            return actual(*args,**kwargs)
        rows=[dict(id=str(i),bucket='training',seconds=3,shutdown_reserve=.1,gpu=False) for i in range(3)]
        with patch.object(c.subprocess,'Popen',side_effect=launch):
            with self.assertRaisesRegex(RuntimeError,'Incomplete round'):
                runner.dispatch(folder,rows,lambda *args:[sys.executable,'-c','import time;time.sleep(.02)'],lambda *args:None)
        value=f.read(folder/'dispatch.json')
        self.assertTrue(value['completed'][0]['launch_failed'])
        self.assertEqual([(r['slot'],r['success']) for r in value['completed']],[(0,False),(1,True),(1,True)])
        EVIDENCE['launch_failure']=value

    def test_release_sleep_excluded_only_from_parent_not_child(self):
        ledger=c.Ledger(self.root,dict(training=0,gpu=0,auxiliary=0),dict(training=100,gpu=100,auxiliary=100))
        ready=self.root/'ready'
        code="import signal,sys,time;from pathlib import Path;signal.signal(signal.SIGTERM,lambda *a:(time.sleep(.15),sys.exit(0)));Path(sys.argv[1]).write_text('ready');time.sleep(10)"
        child=subprocess.Popen([sys.executable,'-c',code,str(ready)],start_new_session=True)
        begin=time.monotonic()
        try:
            while not ready.exists() and time.monotonic()-begin < 3:time.sleep(.01)
            self.assertTrue(ready.exists())
            ledger.control();before=ledger.used['auxiliary'];waited=[]
            def pause(seconds):
                start=time.monotonic();ledger.exclude_sleep(seconds);waited.append(time.monotonic()-start)
            start=time.monotonic()
            c.release(child,pause)
            end=time.monotonic()
            ledger.control()
            control=ledger.used['auxiliary']-before
            self.assertGreater(sum(waited),.05)
            self.assertAlmostEqual(end-start-control,sum(waited),delta=.03)
            ledger.finish(dict(id='cleanup_fixture',bucket='training'),begin,end)
            self.assertGreaterEqual(ledger.events[-1]['actual_seconds'],sum(waited))
            self.assertEqual(child.returncode,0)
            EVIDENCE['release_wait_accounting']=dict(release_wall=end-start,excluded_sleep=sum(waited),
                parent_control_charged=control,child_actual_seconds=ledger.events[-1]['actual_seconds'],
                child_charged_seconds=ledger.events[-1]['charged_seconds'],child_released=True)
        finally:
            if child.poll() is None:c.release(child)


if __name__=='__main__':
    if sys.platform!='linux':raise SystemExit('Linux only')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:unittest.main(verbosity=2)
    finally:Path(os.environ['ABC_TEST_ROOT'],'evidence.json').write_text(json.dumps(EVIDENCE,indent=2)+'\n')
