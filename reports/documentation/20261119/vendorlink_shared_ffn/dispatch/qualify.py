"""Handwritten CPU exercise of the staged controller and its real worker entry."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

import numpy as np
import psutil


def run_suite(root, ref_file):
    import torch
    import step35_ffn_stages as stages
    import step35_market_queue as queue
    import step35_market_worker as worker
    import step35_vendorlink_account_run as native
    import step35_vendorlink_account_control as ctrl
    import step35_vendorlink_account_evaluate as ev
    from step29_vendorlink_data import group_chunks
    from step30_evolution_io import read, record, verify, write
    from step35_market_adapters import SHARED_FFN_FORMAT
    from step35_vendorlink_account_policy import CELLS, DOMAINS, SEEDS

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and len(os.sched_getaffinity(0)) == 1
    ref = read(ref_file)
    previous = read(verify(ref['gate']))
    external_gate = dict(vendorlink_reference=record(ref_file), manifest=previous['manifest'],
                         model_binding=previous['model_binding'])
    readers, old, refs = stages.vendor_reference(external_gate)
    assert tuple(readers) == ('V_SET','V_CLS') and all([r['seed'] for r in rows] == list(SEEDS) for rows in readers.values())
    for key in ('vendorlink_reference','manifest','model_binding'):
        try:
            stages.vendor_reference(dict(external_gate, **{key: {'sha256':'wrong'}}))
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted a mismatched external '+key)

    campaign = root/'campaign'
    gate_path = root/'formal_gate.json'  # This entire fixture lives in TemporaryDirectory.
    files = {}
    counts = {d:dict(eligible_accounts=100) for d in DOMAINS}
    for d in DOMAINS:
        for role in ('fit','dev'):
            path = root/'fixture'/f'{d}_{role}.json'
            write(path, [f'handwritten-{d}-{role}'])
            files[d+':'+role] = record(path)
    manifest = dict(counts=counts, files=files)
    manifest_file = root/'fixture/manifest.json'
    write(manifest_file, manifest)
    note = root/'fixture/synthetic_only.json'
    write(note, dict(scope='ephemeral synthetic receipts only; no GPU, market data or formal authorization'))
    sources = [record(p) for p in sorted((root/'workspace/scripts').glob('*.py'))]
    gpu = root/'fixture/mock_gpu_receipt.json'
    write(gpu, dict(fixture=True, status='PASS', device='cuda', market_adapters=SHARED_FFN_FORMAT,
          model_sources={Path(b['path']).name:b['sha256'] for b in sources if Path(b['path']).name in
                         ('step35_market_adapters.py','step35_vendorlink_account_run.py')}))
    tasks = worker.market_tasks(shared_ffn=True)
    task_file = root/'fixture/tasks.json'
    write(task_file, tasks)
    g = dict(protocol=stages.PROTOCOL, training_authorized=True, review_closed=True, sources=sources,
             authorization=record(note), scope=record(note), review=record(note), qualification=[record(gpu)],
             gpu_qualification=record(gpu), vendorlink_reference=record(ref_file),
             model_binding=dict(directory='SYNTHETIC_NO_MODEL',files=[]), manifest=record(manifest_file),
             tasks=record(task_file), worker_entry=record(Path(worker.__file__)), caps=stages.CAPS,
             prior_seconds={b:0 for b in stages.CAPS}, cpu_test_seconds=0, prior_operation_ids=[],
             limits=dict(previous['limits'],free_disk_reserve=0), workspace=str(root/'workspace'),
             campaign=str(campaign), experiment_root=str(root))
    scenario = ''
    events = []
    results = {}

    def fake_train(model, load_fit, job, task, identity, binding, check):
        assert model is None and binding == g['model_binding']
        for d in DOMAINS:
            assert load_fit(d) == [f'handwritten-{d}-fit']
        job.mkdir()
        endpoints = {}
        for i in range(4):
            f = job/f'endpoint_{i}.json'
            write(f,dict(synthetic=True,stage=i,task=task['id']))
            endpoints[str(i)] = record(f)
        per = 3*len(group_chunks(list(range(100)),8))*task['epochs']
        replay = task['method'] != 'H_SEQ'
        workload = dict(current_ads=3*100*12*task['epochs'],history_groups=2*per if replay else 0,
                        history_ads=2*per*16 if replay else 0,teacher_groups=6 if replay else 0)
        if task['method'] == 'H_full':
            workload['deterministic_history_groups'] = 2*per
        out = dict(status='TRAINING_COMPLETE_EVALUATION_CLOSED',task=task,identity=identity,endpoints=endpoints,
                   updates=3*per,workload=workload,current_stream='current-'+str(task['seed']),
                   history_stream='replay-'+str(task['seed']),survivors={d:list(range(6)) for d in DOMAINS[:2]} if replay else {},
                   history_bytes={s:500 for s in ('end_1','enter_2','end_2','enter_3','end_3')})
        write(job/'completion.json',out)
        return out

    def fake_score(task, end, data, binding, dest, role, check):
        assert role == 'dev' and data == manifest and end['task'] == task and not dest.exists()
        original = {'R01':40,'R11':90,'R12':70,'R21':83,'R22':93,'R23':80,'R33':91}
        method = task['method']
        original['R31'],original['R32'] = {'H_SEQ':(81,82),'H_ER':(84,85),'H_full':(89,91)}[method]
        if (scenario=='seq_stable' and method=='H_SEQ') or (scenario=='er_stable' and method=='H_ER'):
            original.update(R31=90,R32=93)
        if scenario=='seq_learning' and method=='H_SEQ':
            original['R22']=71
        if scenario=='er_learning' and method=='H_ER':
            original['R33']=81
        if scenario=='full_no_increment' and method=='H_full':
            original.update(R31=84,R32=85)
        if scenario=='unequal_start' and method=='H_ER' and task['seed']==52:
            original['R11']=88
        dest.mkdir(parents=True)
        exports = {}
        for t,j in CELLS:
            cell = f'R{t}{j}'
            k = original[cell]
            prediction = np.arange(100)
            prediction[k:] = np.roll(prediction[k:],1)
            matrix = np.zeros((100,100),dtype=np.float64)
            matrix[np.arange(100),prediction] = 1
            if scenario=='short_matrix' and task['id']=='H_SEQ_s53' and cell=='R33':
                matrix=matrix[:-1]
            path = dest/f'{method}_{cell}.npy'
            np.save(path,matrix,allow_pickle=False)
            exports[method+':'+cell] = record(path)
        value = dict(status='BLIND_SCORES_COMPLETE', task=task, role=role, training_identity=end['identity'],
                     files=exports, cells=[f'R{t}{j}' for t,j in CELLS],sources=files,reference_role='fit',
                     reference_records=12,diagnostics={})
        write(dest/'completion.json',value)
        return value

    def child(action, dest, name=None, expected_error=None):
        # A real child must match the owning controller's PID and Linux start identity.
        pid = os.fork()
        if pid == 0:
            if action in ('train','score'):
                def access_boundary(event, arguments):
                    if event == 'open' and isinstance(arguments[0], str):
                        path = arguments[0]
                        if ('/analysis/result/' in path or (action == 'train' and
                                ('/scoring/' in path or path.endswith(('_dev.json','_confirm.json'))))):
                            raise AssertionError('Worker tried to read development evidence: '+path)
                sys.addaudithook(access_boundary)
                # Check the boundary itself independently of which files this worker normally reads.
                try: (campaign/'stages/H_SEQ/analysis/result/results.json').read_bytes()
                except AssertionError as exc: assert 'development evidence' in str(exc)
                else: os._exit(2)
            try:
                worker.worker(gate_path,action,dest,name,time.monotonic()+300)
                if expected_error is not None:
                    raise AssertionError('Worker admitted an invalid operation: '+expected_error)
            except (ValueError, FileNotFoundError) as exc:
                if expected_error is None or expected_error not in str(exc):
                    traceback.print_exc(); os._exit(1)
            except BaseException:
                traceback.print_exc(); os._exit(1)
            os._exit(0)
        _,status = os.waitpid(pid,0)
        if os.waitstatus_to_exitcode(status):
            raise AssertionError((scenario,action,name,str(dest)))

    def drive(self,jobs,slots,completed,*,recovery=None):
        action = jobs[0]['action']
        assert slots == (3 if action=='train' else 1)
        for i,job in enumerate(jobs):
            name = job.get('task_id')
            if name:
                assert name.startswith(('H_SEQ_','H_ER_','H_full_'))
            if scenario=='missing_train' and action=='train' and i==2:
                continue
            if scenario=='missing_score' and action=='score' and i==2:
                continue
            target = job['target']
            target.mkdir(parents=True)
            if scenario=='pass_all' and name=='H_SEQ_s51' and action=='train':
                child('train',campaign/'training/H_ER_s51/attempt-1','H_ER_s51','closed.json')
                child('score',campaign/'scoring/H_SEQ_s51','H_SEQ_s51','training_complete.json')
            # Exercise the production argv builder without binding fixture children to other CPUs.
            with patch.object(psutil.Process,'cpu_affinity',return_value=list(range(6))):
                argv=self.command(job,i if recovery else 0,time.monotonic()+300)
            assert Path(argv[6])==Path(worker.__file__) and Path(argv[argv.index('--gate')+1])==gate_path
            child(argv[argv.index('--action')+1],Path(argv[argv.index('--target')+1]),
                  argv[argv.index('--task')+1] if '--task' in argv else None)
            completed(job)
            receipt = dict(success=True,released=True,slot=i)
            if recovery:
                receipt.update(task_id=name,attempt=1)
            write(target/'dispatch_receipt.json',receipt)
            if recovery:
                recovery.finished(i,job,receipt)
            events.append([action,name or target.parent.name])
            if scenario=='pass_all' and action=='score' and name=='H_SEQ_s51':
                child('analyze',campaign/'stages/H_SEQ/analysis',expected_error='scoring_complete.json')

    def qualify(case, error=None, change=None, other_tasks=None):
        nonlocal scenario
        scenario=case
        events.clear()
        write(task_file,tasks if other_tasks is None else other_tasks)
        write(gate_path,dict(g,tasks=record(task_file),**(change or {})))
        try:
            queue.campaign(gate_path)
        except (ValueError,FileNotFoundError,RuntimeError) as exc:
            if error is None or error not in str(exc):
                raise
        else:
            assert error is None,case
        root_exists=campaign.is_dir()
        opened=[m for m in stages.METHODS if (campaign/'stages'/m/'analysis/result/access.json').is_file()]
        data=dict(opened=opened,dispatched=events.copy(),campaign_created=root_exists)
        if (campaign/'completion.json').is_file():
            closed=read(campaign/'completion.json')
            assert closed['confirm_open'] is False and list(closed['stages'])==opened
            data.update(status=closed['status'])
            for m in opened:
                analysis=campaign/'stages'/m/'analysis/result'
                assert read(analysis/'access.json')['tasks']==stages.batch_tasks(tasks,m)
                assert len(read(analysis/'metrics.json'))==27 and len(read(analysis/'results.json')[m])==3
            if 'H_full' in opened:
                folder=campaign/'stages/H_full/analysis/result'
                full=read(folder/'decision.json')
                assert set(('V_CLS','V_SET','original_H_ER','original_beta_Full'))<=set(full['summaries'])
                pairs=read(folder/'metric_pairs.json')['full_minus_comparator']
                assert len(pairs)==36 and all(len(v)==13 for v in pairs.values())
                data['full_er']=full['full_er']
                data['external']=full['external']
            if case=='pass_all':
                child('train',campaign/'training/H_SEQ_s51/attempt-2','H_SEQ_s51','not active')
                child('score',campaign/'scoring/H_ER_s51','H_ER_s51','not active')
                try: queue.campaign(gate_path)
                except FileExistsError: pass
                else: raise AssertionError('Existing campaign was recreated')
            if case in ('seq_stable','er_stable'):
                next_method='H_ER' if case=='seq_stable' else 'H_full'
                child('train',campaign/'training'/f'{next_method}_s51'/'attempt-1',
                      f'{next_method}_s51','did not authorize')
        else:
            data.update(status='REJECTED')
            assert not opened
        if root_exists:
            campaign.rename(root/('archived_fixture_'+case))
        results[case]=data

    with ExitStack() as ctx:
        # The frozen reference was verified independently above. Fixtures substitute only the
        # dataset/model identity so no scientific input, model or CUDA job can be accessed.
        ctx.enter_context(patch.object(stages,'vendor_reference',lambda fake: (readers,old,refs)))
        ctx.enter_context(patch.object(queue.MarketRunner,'run_many',drive))
        ctx.enter_context(patch.object(ctrl,'configure_compute',lambda:None))
        ctx.enter_context(patch.object(native,'make_model',lambda b,t:None))
        ctx.enter_context(patch.object(native,'execute',fake_train))
        ctx.enter_context(patch.object(ev,'score',fake_score))
        qualify('no_authorization','approved',change=dict(training_authorized=False))
        qualify('wrong_gpu','CUDA qualification',change=dict(qualification=[]))
        qualify('old_twelve','frozen tasks',other_tasks=worker.market_tasks())
        mixed=tasks.copy()
        mixed[0]=dict(mixed[0],market_adapters='fit12-h-market-qv-r8-v1')
        qualify('mixed_format','frozen tasks',other_tasks=mixed)
        wrong=tasks.copy(); wrong[1]=tasks[0]
        qualify('duplicate_seed','frozen tasks',other_tasks=wrong)
        qualify('missing_train','three fixed training winners')
        qualify('missing_score','dispatch_receipt.json')
        qualify('short_matrix','Incomplete Q2/FIT12')
        for case,count,outcome in [('seq_stable',1,'STOP_NO_MEANINGFUL_FORGETTING'),('seq_learning',1,'STOP_LEARNING'),
                ('er_stable',2,'STOP_ER_FORGETTING'),('er_learning',2,'STOP_ER_LEARNING'),
                ('unequal_start',2,'HOLD_PREHISTORY_IDENTITY'),('full_no_increment',3,'STOP_FULL_APPLICATION'),
                ('pass_all',3,'PASS_FULL_APPLICATION')]:
            qualify(case)
            actual=results[case]
            assert actual['status']==outcome and actual['opened']==list(stages.METHODS[:count]),(case,actual)
            expect=[]
            for method in stages.METHODS[:count]:
                expect.extend([action,f'{method}_s{s}'] for action in ('train','score') for s in SEEDS)
                expect.append(['analyze',method])
            assert actual['dispatched']==expect
    assert all(not results[x]['campaign_created'] for x in ('no_authorization','wrong_gpu','old_twelve','mixed_format','duplicate_seed'))
    assert [j[0] for j in results['missing_train']['dispatched']]==['train']*2
    assert [j[0] for j in results['missing_score']['dispatched']]==['train']*3+['score']*2
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',device='cpu',gpu_initialized=False,formal_training_tasks=0,synthetic_cases=results,
                official_aggregate_binding=refs['results']['sha256'],external_readers=list(readers),seeds=list(SEEDS),
                rejected_wrong_reference_fields=['bundle hash','manifest','pretrained model'],
                training_and_blind_workers_read_development_values=False,closed_stage_reentry_rejected=True,
                boundaries='Real campaign, child worker, receipt checks, native 13-metric analysis and decisions; GPU execution, model outputs and timing are synthetic.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('sources',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    start=time.monotonic()
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        root=Path(tmp)
        path=root/'workspace/scripts'
        shutil.copytree(args.sources,path)
        sys.path.insert(0,str(path))
        result=run_suite(root,args.reference)
        from step30_evolution_io import write
        from step35_market_worker import market_tasks
        write(args.output.with_name('tasks.json'),market_tasks(shared_ffn=True))
    result.update(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
