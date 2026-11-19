from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import sys
import psutil

base = Path('/home/yongpeng/cross-lingual/reports/documentation/20261118')
en = base/'vendorlink_market_training/execution'
zh = base/'h_history_benchmark'
camp = en/'campaign'
zc = zh/'campaign'
def read(p):
    return json.loads(p.read_bytes())
def binding(p):
    data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
def checked(b, expected=None):
    p = Path(b['path'])
    if expected is not None and p != expected:
        raise ValueError(str(p))
    if binding(p) != {k:b[k] for k in ('path','bytes','sha256')}:
        raise ValueError('Identity mismatch '+str(p))
    return p
def stamp(p):
    return dt(p.stat().st_mtime) if p.is_file() else None
def dt(sec=None):
    tz = timezone(timedelta(hours=8))
    v = datetime.now(tz) if sec is None else datetime.fromtimestamp(sec,tz)
    return v.isoformat(timespec='seconds')
def match(x):
    try:
        p = psutil.Process(x['pid'])
        ticks = int(Path(f"/proc/{p.pid}/stat").read_text().rsplit(')',1)[1].split()[19])
        return (p.status()!=psutil.STATUS_ZOMBIE and p.username()==x['owner'] and p.cwd()==x['cwd']
                and p.cmdline()==x['command'] and ticks==x['start_ticks']
                and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==x['boot_id'])
    except (psutil.NoSuchProcess,psutil.ZombieProcess,FileNotFoundError):
        return False
assert binding(en/'formal_gate.json')['sha256']=='7607f7cf20834a5fa4f9a34ebda0d92543537d33b87c704abf91d812b0555b1f'
assert binding(zh/'formal_gate.json')['sha256']=='4fd8bd8ecd9b3dad90d6b03cff86f60b53335b82aa8ac51b528f2619c97767fa'
g=read(en/'formal_gate.json')
launch=read(en/'launch.json')
checked(launch['gate'],en/'formal_gate.json')
for b in g['sources']:
    checked(b)
sys.path.insert(0,str(en/'workspace/scripts'))
from step35_market_accounting import billed_operation, MarketLedger
def settle():
    if not match(launch['controller']):
        MarketLedger.settle_closed(camp)
def fetch(p,target):
    files[target]=binding(p)
files={}
with billed_operation(en,'result_evidence_and_handoff',settle):
    zd=read(zc/'dispatch.json') if (zc/'dispatch.json').is_file() else {}
    armed=read(zh/'start/armed.json')['identity']
    zs=dict(observed=dt(),gate=binding(zh/'formal_gate.json'),controller_pid=armed['pid'],
        controller_identity_matches=match(armed),wait_status=read(zh/'wait/status.json').get('status'),
        ready_at=stamp(zh/'wait/ready.json'),campaign_exists=zc.is_dir(),first_update_at=stamp(zh/'wait/first_update.json'),
        retired_at=stamp(zc/'waiter_deleted.json'),once_file_exists=(zh/'workspace/scripts/once_wait.py').exists(),
        campaign_status=zd.get('status'),successful_tasks=zd.get('successful_tasks',[]),
        pending_count=len(zd.get('pending',[])),failures=zd.get('failures',[]),budget_stop=zd.get('budget_stop'),
        terminal_failure=read(zc/'failure.json') if (zc/'failure.json').exists() else None,
        wait_failure=read(zh/'wait/failure.json') if (zh/'wait/failure.json').exists() else None,
        complete=(zc/'completion.json').is_file(),active=[dict(slot=k,task=v.get('task_id'),attempt=v.get('attempt'),
            pid=v['pid'],identity_matches=match(v)) for k,v in zd.get('active',{}).items()])
    for remote,local in [('wait/ready.json','ready.json'),('wait/first_update.json','first_update.json'),('campaign/waiter_deleted.json','waiter_deleted.json')]:
        if (zh/remote).is_file():
            fetch(zh/remote,'zh/'+local)
    ed=read(camp/'dispatch.json')
    es=dict(observed=dt(),gate=binding(en/'formal_gate.json'),production_git=g['source_git'],
        source_count=len(g['sources']),controller_pid=launch['controller']['pid'],controller_identity_matches=match(launch['controller']),
        dispatch_status=ed['status'],active=list(ed['active']),failure=read(camp/'failure.json') if (camp/'failure.json').exists() else None,
        training_complete_at=stamp(camp/'training_complete.json'),scoring_complete_at=stamp(camp/'scoring_complete.json'),
        analysis_receipt_at=stamp(camp/'analysis/dispatch_receipt.json'),batch_completed_at=stamp(camp/'completion.json'),
        launched=launch['started'])
    tasks=read(checked(g['tasks']))
    released=False
    if es['failure'] is None and all(es[k] for k in ('training_complete_at','scoring_complete_at','batch_completed_at')):
        done=read(camp/'completion.json')
        tr=read(camp/'training_complete.json')
        sc=read(camp/'scoring_complete.json')
        assert done['status']=='COMPLETE' and done['dev_opening']=='whole_batch_once' and done['confirm_open'] is False
        assert tr['tasks']==sc['tasks']==tasks and len(tasks)==12 and len(sc['completions'])==12
        assert set(tr['winners'])=={t['id'] for t in tasks}
        decision_path=checked(done['decision'],camp/'analysis/result/decision.json')
        assert read(camp/'analysis/dispatch_receipt.json')['success'] and ed['status']=='COMPLETE' and not ed['active']
        assert read(camp/'analysis/result/access.json')['opening']=='whole_batch_once'
        runtimes={}
        streamdata={}
        cell_counts={}
        for t,b in zip(tasks,sc['completions'],strict=True):
            name=t['id']
            w=tr['winners'][name]
            td=camp/'training'/name/f"attempt-{w['attempt']}"
            rec=read(checked(w['receipt'],td/'dispatch_receipt.json'))
            train=read(checked(w['completion'],td/'run/completion.json'))
            score=read(checked(b,camp/'scoring'/name/'result/completion.json'))
            assert rec['success'] and train['status']=='TRAINING_COMPLETE_EVALUATION_CLOSED' and train['identity']['task']==t
            assert score['status']=='BLIND_SCORES_COMPLETE' and score['role']=='dev' and score['task']==t
            assert train['identity']['gate']==launch['gate'] and score['training_identity']==train['identity']
            cells=['R01','R11','R12','R21','R22','R23','R31','R32','R33']
            readers=['V_SET','V_CLS','V_CLS_ads'] if name.startswith('VendorLink') else [t['method']]
            assert set(score['cells'])==set(cells) and set(score['files'])=={m+':'+c for m in readers for c in cells}
            assert score['reference_role']=='fit' and score['reference_records']==12
            assert read(camp/'scoring'/name/'dispatch_receipt.json')['success']
            cell_counts[name]=len(score['files'])
            runtimes[name]=dict(attempt=w['attempt'],updates=train['workload']['current_updates'],
                history_groups=train['workload']['history_groups'],history_bytes=train.get('history_bytes',{}),
                scored_matrices=cell_counts[name])
            if not name.startswith('VendorLink'):
                streamdata[name]=(train['current_stream'],train['history_stream'],train['survivors'])
        streams={}
        for seed in [51,52,53]:
            a,b,c=[streamdata[f'{m}_s{seed}'] for m in ['H_SEQ','H_ER','H_full']]
            streams[str(seed)]=dict(current_equal=a[0]==b[0]==c[0],history_equal=b[1]==c[1],
                survivor_lists_equal=b[2]==c[2],counts={d:len(l) for d,l in c[2].items()})
        assert all(x['current_equal'] and x['history_equal'] and x['survivor_lists_equal'] and sorted(x['counts'].values())==[6,6] for x in streams.values())
        result_dir=camp/'analysis/result'
        metrics=read(result_dir/'metrics.json')
        assert len(metrics)==sum(cell_counts.values()) and all(len(v)==13 for v in metrics.values())
        sizes=[n for x in runtimes.values() for n in x['history_bytes'].values()]
        assert max(sizes)<=2**20
        es.update(release=done,training=runtimes,paired_streams=streams,matrix_count=len(metrics),metrics_per_cell=13,
            history_bytes_max=max(sizes),failed_training_attempt_count=sum((camp/'training'/t['id']/f'attempt-{a}'/'worker_failure.json').is_file() for t in tasks for a in (1,2)))
        for n in ('training_complete','scoring_complete','completion'):
            fetch(camp/(n+'.json'),'en/'+n+'.json')
        fetch(camp/'analysis/dispatch_receipt.json','en/analysis_receipt.json')
        for n in ('results','metrics','metric_summary','metric_pairs','decision','access'):
            fetch(result_dir/(n+'.json'),'en/'+n+'.json')
        released=True
    else:
        es.update(successful_tasks=ed.get('successful_tasks',[]),score_receipts={t['id']:(camp/'scoring'/t['id']/'dispatch_receipt.json').is_file() for t in tasks},pending=ed.get('pending'),dispatch_failures=ed.get('failures'))
budget=read(camp/'budget.json')
es['settled_seconds']=budget['used_seconds']
es['caps_seconds']=budget['caps']
es['last_internal_operations']=[dict(operation=x['operation'],bucket=x['bucket'],success=x.get('success')) for x in budget['events'] if x['operation'] not in budget.get('operation_ids',[])][-3:]
if released: fetch(camp/'budget.json','en/budget.json')
print(json.dumps(dict(english=es,chinese=zs,files=files,released=released),ensure_ascii=True))
