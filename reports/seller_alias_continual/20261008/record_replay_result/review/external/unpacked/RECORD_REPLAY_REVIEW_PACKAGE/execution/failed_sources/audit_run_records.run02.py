"""Read-only review of supplied execution/provenance/cleanup records.

Uses only standard library and NumPy; imports the reviewer-owned Reader helper,
never imports project code. No remote access, project execute/native, training,
label/text parsing, model load, or Memory deserialization.
"""
from pathlib import Path
from collections import Counter
import argparse, datetime, difflib, hashlib, io, json, math, random, sys, time, zipfile
import numpy as np
from audit_saved_results import Reader, require, ORDERS

def derived_seed(value,*parts):
    return int.from_bytes(hashlib.sha256((json.dumps([value,*parts],separators=(',',':'))+'\n').encode()).digest()[:8],'big')%(2**63-1)

def main(root,out,archive=None):
    started=time.perf_counter(); out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rd=Reader(root); findings={}; diffs={}; omitted=[]
    def verify(prefix,rec,purpose):
        return rd.read(prefix+rec['path'],purpose,rec)
    def close(a,b,label,rtol=2e-6,atol=2e-6):
        require(math.isfinite(a) and math.isfinite(b),'Nonfinite '+label)
        diffs[label]=max(diffs.get(label,0.),abs(a-b))
        require(abs(a-b)<=atol+rtol*abs(b),'Numerical log mismatch '+label)
    manifest=rd.json('package_manifest.json','All 334 registered payload identities')
    require(len(manifest)==334 and len({r['path'] for r in manifest})==334,'Archive payload count/uniqueness')
    supplied={str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file()}
    require(supplied=={r['path'] for r in manifest}|{'package_manifest.json'},'Archive file membership')
    for rec in manifest: verify('',rec,'Package registered byte count and SHA-256')
    archive_id=None
    if archive:
        b=Path(archive).read_bytes();archive_id={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            require(len(z.namelist())==len(set(z.namelist()))==335,'Original ZIP 335 unique members')
            for n in z.namelist(): require(z.read(n)==rd.read(n,'Extracted bytes equal uploaded archive'),'Archive extraction mismatch '+n)
    findings['package']={'payloads':334,'total_members':335,'all_payload_identities_match':True,'archive':archive_id}

    hpath='history/implementation_review_input.zip';hb=rd.read(hpath,'Historical implementation archive identities and minimal R1/R2 diff')
    nested=[];source_changes=[]
    with zipfile.ZipFile(io.BytesIO(hb)) as z:
        require(len(z.namelist())==159 and hashlib.sha256(hb).hexdigest()=='a4903d4a6a310729012b958a172424c5e1d218564be9ed1bd05c80f5436849e1','Historical archive binding')
        oldmanifest=json.loads(z.read('manifest.json'))
        require(len(oldmanifest)==158,'Historical manifest payload count')
        for rec in oldmanifest:
            b=z.read(rec['path']);require(len(b)==rec['bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256'],'Historical payload '+rec['path'])
        for n in z.namelist():
            b=z.read(n);nested.append({'path':hpath+'!/'+n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
                                      'mode':'machine identity only; semantic range recorded separately'})
        for p in sorted((Path(root)/'result/source').rglob('*')):
            if p.is_file():
                rel=str(p.relative_to(Path(root)/'result/source'));old=z.read('implementation/'+rel);new=p.read_bytes()
                if old!=new:
                    source_changes.append(rel)
                    patch=''.join(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),
                                                     fromfile='historical/implementation/'+rel,tofile='actual/result/source/'+rel))
                    (out/(p.name+'.diff.txt')).write_text(patch,encoding='utf-8')
        require(set(source_changes)=={'scripts/step28_record_replay_run.py','tests/test_step28_record_replay.py'},'Only R1/R2 implementation changes expected')
    (out/'nested_inventory.json').write_text(json.dumps(nested,ensure_ascii=False,indent=2),encoding='utf-8')
    findings['history']={'members':159,'payload_hashes_verified':158,'actual_changed_files':source_changes,'historical_core_reexecution':False}

    execution=rd.json('result/job/execution.json','Actual launch source identities and gate')
    run=rd.json('result/job/run/manifest.json','All physical points, starts, source identities and registered outputs')
    col=rd.json('result/job/evaluation/collected.json','Completed metric registry source binding')
    ev=rd.json('result/job/evaluation/evaluation.json','Completed evaluation binding')
    gate=rd.json('result/execution_gate.json','Actual qualification and frozen budget')
    nativegate=rd.json('qualification/native_gate.json','Historical native approval source bindings')
    source_inventory=rd.json('qualification/execution_source_inventory.json','Actual 21-source inventory')
    cpu=rd.json('qualification/cpu/attempt03/result.json','Latest CPU evidence and source identities')
    gpu=rd.json('qualification/gpu/attempt01/result.json','Existing native evidence and source identities')
    startup=rd.json('result/job/run/startup.json','Startup source and public-input identities')
    progress=rd.json('result/job/run/progress.json','Final saved progress membership and counters')
    src=execution['sources'];require(len(src)==21,'Source count')
    for name,seq in [('run',run['source_files']),('collection',col['source_files']),('evaluation',ev['source_files']),
        ('gate',gate['source_files']),('nativegate',nativegate['source_files']),('inventory',source_inventory),
        ('cpu',cpu['source_files']),('gpu',gpu['source_files']),('startup',startup['source_files']),('progress',progress['source_files'])]:
        require(seq==src,'Source inventory disagreement '+name)
    for rec in src: verify('result/source/',rec,'Actual frozen source equality across all 11 records')
    rd.read('result/execution_gate.json','Execution binds actual gate',execution['gate'])
    gate_refs={'cpu':'qualification/cpu/attempt03/result.json','gpu':'qualification/gpu/attempt01/result.json',
               'main_review':'qualification/report.zh.txt','native_disposition':'qualification/native_disposition.zh.txt'}
    for k,p in gate_refs.items(): rd.read(p,'Gate referenced qualification record',gate[k])
    for k in ('cpu','main_review'): rd.read(gate_refs[k],'Native gate referenced evidence',nativegate[k])
    require(cpu['status']==gpu['status']=='PASS_HANDWRITTEN_ONLY' and cpu['tests_run']==9 and cpu['failures']==cpu['errors']==cpu['skipped']==0,'Qualification status')
    require(gpu['native']['adam_step']==1 and gpu['native']['records_per_group']==224 and gpu['native']['tokens_per_channel']==256,'Native evidence shape')
    n=gpu['native']; estimate=1.25*(7776*n['paired_update_seconds']/2+2274*n['source_seconds']+1800)
    close(n['shape_upper_projection_seconds'],estimate,'qualification_projection',rtol=0,atol=1e-9)
    require(0<estimate<=86400 and n['within_formal_24h_projection'] is True,'Native projection within fixed bound')
    require(all(n['parameters_changed'].values()) and all(v>0 and math.isfinite(v) for d in n['gradient_increments'].values() for v in d.values()),'Recorded actual native component update')
    cpu_times=[]
    for i in (1,2,3):
        p=f'qualification/cpu/attempt{i:02d}/'
        v=rd.json(p+'result.json','Retained CPU attempt facts')
        t=rd.read(f'qualification/cpu/attempt{i:02d}.wrapper.txt','CPU wrapper wall clock').decode().splitlines()
        dt=(datetime.datetime.fromisoformat(t[2])-datetime.datetime.fromisoformat(t[0])).total_seconds();cpu_times.append(dt)
        rd.read(p+'unittest.txt','Historical CPU actual test report; not executed')
    require(sum(cpu_times)==gate['cpu_outer_cumulative_seconds']==25,'CPU cumulative external wall time')
    findings['sources_and_qualification']={'frozen_sources':21,'agreeing_identity_records':11,'cpu_outer_seconds':sum(cpu_times),
        'cpu_latest_tests':9,'native_outer_seconds':gate['gpu_outer_cumulative_seconds'],'projection_seconds':estimate,
        'native_is_supplied_prior_evidence_not_reexecuted':True}

    completion=rd.json('result/job/completion.json','Final completion, resources and original criteria')
    before=rd.json('result/job/before_valid.json','All stages restored before valid access')
    access=rd.json('result/job/access.json','Final observed label-parse ledger')
    resource=rd.json('result/job/resource.json','Final resource observations')
    wrapper=rd.read('result/job.wrapper.txt','Original wrapper exit and wall clock').decode().splitlines()
    require(wrapper[1]=='exit_code=0','Original exit code')
    wall=(datetime.datetime.fromisoformat(wrapper[2])-datetime.datetime.fromisoformat(wrapper[0])).total_seconds()
    require(completion['status']=='COMPLETE_RECORD_REPLAY_DEVELOPMENT','Completion status')
    require(access==completion['access']=={'heldout':0,'owners':0,'train':1,'valid':1},'Final access counts')
    require(before['access']=={'heldout':0,'owners':0,'train':1,'valid':0} and before['points']==15 and before['status']=='PASS_COMPLETE_BLIND_GATE','Blind gate')
    rd.read('result/job/'+before['manifest']['path'],'Blind-gate manifest SHA',before['manifest'])
    rd.read('result/job/'+completion['evaluation']['path'],'Completion-evaluation SHA',completion['evaluation'])
    require(not any('failure' in p for p in supplied if p.startswith('result/job/')),'Unexpected formal failure file')
    require(run['physical_updates']==progress['physical_updates']==4320 and run['gradient_group_presentations']==progress['gradient_group_presentations']==7776,'Progress totals')
    for k in ('points','starts','training','partition','public_inputs','pretrained_archive','initial'):
        require(progress[k]==run[k],'Progress/manifest '+k)
    limits={'elapsed_seconds':86400,'peak_output_bytes':32*2**30,'peak_cuda_reserved_bytes':28*2**30,'peak_rss_bytes':64*2**30}
    for k,v in limits.items(): require(completion['budget'][k]<=v and resource[k]<=v,'Budget '+k)
    require(wall==27882 and abs(wall-resource['elapsed_seconds'])<2,'Recorded wrapper/resource time')
    partition=rd.json('result/job/run/'+run['partition']['path'],'Full public partition group IDs and domains',run['partition'])
    pools={r:{d:[x['group_uid'] for x in partition[r] if x['domain']==d] for d in 'ABC'} for r in ('fit','calibration','development')}
    for r,size in [('fit',48),('calibration',12),('development',20)]:
        require(all(len(v)==size for v in pools[r].values()),'Partition count '+r)
    all_ids=[x['group_uid'] for v in partition.values() for x in v]
    require(len(all_ids)==len(set(all_ids))==240,'Partition uniqueness and disjointness')
    ud={x['group_uid']:x['domain'] for x in partition['fit']}
    logs={};point_diagnostics={};all_scores={};score_values=0
    for order in ORDERS:
        require(run['starts'][order+'_C']==run['starts'][order+'_S'],'Full paired starts '+order)
        firstpoint=run['points'][order+'_shared']
        require(run['starts'][order+'_C']['full_state_sha256']==firstpoint['full_state']['state_sha256'],'Start full-state binding')
        require(run['starts'][order+'_C']['memory']==firstpoint['memory_summary'],'Start memory binding')
        for arm in ('C','S'):
            reservoir=[];origins={};seen=0;previous=None;retention=random.Random(derived_seed(20260930,order,'retention'))
            for stage in (1,2,3):
                name=order+'_shared' if stage==1 else f'{order}_{arm}_stage{stage}'
                point=run['points'][name];local=rd.json('result/job/run/points/'+name+'.json','Full stage checkpoint metadata')
                require(local==point,'Point file differs from manifest '+name)
                tr=rd.json('result/job/run/'+run['training'][name]['path'],'All 288 update rows and memory snapshots',run['training'][name])
                fit=sorted(pools['fit'][order[stage-1]]);sch=random.Random(derived_seed(20260918,order,stage,'current'))
                schedule=[]
                for epoch in range(6):
                    ids=fit.copy();sch.shuffle(ids);schedule.extend(ids)
                require(tr['current_ids']==schedule and len(tr['rows'])==tr['updates']==288,'Full current schedule '+name)
                hist_rng=random.Random(derived_seed(20260930,order,stage,'history_draws'))
                history=[reservoir[hist_rng.randrange(6)] for _ in range(288)] if stage>1 else []
                require(tr['history_ids']==history,'History draw sequence '+name)
                require(point['full_restore_verified'] is True and tr['adam_step']==point['adam_step']==stage*288,'Recorded restore/Adam '+name)
                if stage>1:
                    for k in ('members','seen','origins','tables'):
                        require(tr['memory_before'][k]==tr['memory_after'][k]==previous[k],'Stage memory frozen '+name+'/'+k)
                    require(tr['memory_before']['count']==0 and tr['memory_after']['count']==288,'History draw counter '+name)
                for i,row in enumerate(tr['rows'],1):
                    require(row['step']==i and row['current_uid']==schedule[i-1] and row['history_uid']==(history[i-1] if history else None),'Row schedule '+name)
                    for role in ('current','history') if stage>1 else ('current',):
                        require(all(math.isfinite(v) for v in row[role].values()),'Finite row losses '+name)
                        require(row[role]['record_gradient_norm']>=0,'Recorded group gradient '+name)
                    close(row['current']['weighted'],row['current']['supervised'],'current_weighted')
                    total=row['current']['supervised']
                    if stage>1:
                        h=row['history'];d=(h['mse0']+h['mse1'])/2 if arm=='C' else h['mse0']
                        close(h['distillation'],d,'distillation_definition')
                        close(h['weighted'],.1*h['supervised']+.5*d,'history_weighted')
                        total+=.1*h['supervised']+.5*d
                    close(row['total'],total,'total_loss')
                    lr=1e-5*min(i/29,(288-i)/259) if i>29 else 1e-5*i/29
                    close(row['encoder_lr'],lr,'encoder_lr',rtol=1e-12,atol=1e-16)
                    require(row['head_lr']==.001 and math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0,'Head LR and global norm')
                if stage<3:
                    for uid in fit:
                        seen+=1;j=len(reservoir) if len(reservoir)<6 else retention.randrange(seen)
                        if j<6:
                            if j==len(reservoir):reservoir.append(uid)
                            else:origins.pop(reservoir[j]);reservoir[j]=uid
                            origins[uid]=stage
                m=point['memory_summary']
                require(m['members']==reservoir and m['origins']==origins and m['seen']==seen,'Algorithm R end members/origins '+name)
                require(m['bytes']==point['memory']['bytes']<=2**20 and m['sha256']==point['memory']['sha256'],'Recorded memory byte cap '+name)
                require(set(m['tables'])==set(reservoir),'Memory teacher identities')
                if previous:
                    for u in set(previous['tables'])&set(m['tables']):require(previous['tables'][u]==m['tables'][u],'Survivor source refreshed '+name)
                    if stage==3:require(point['retention'] is None and previous['tables']==m['tables'],'Unneeded stage3 new teacher')
                mapping=rd.json('result/job/run/'+point['mapping']['path'],'All calibration metadata and optimizer trajectory',point['mapping'])
                require(mapping['status']=='PASS_CALIBRATION_FIT' and 0<mapping['a']<=100 and math.isfinite(mapping['b']),'Calibration status')
                require(mapping['group_count']==12 and mapping['pair_count']==4536 and mapping['positive_count']==240,'Calibration counts')
                require(point['calibration_ids']==pools['calibration'][order[stage-1]],'Current-domain-only calibration')
                require(point['maps']['stage']=={k:mapping[k] for k in ('a','b')},'Stage map binding')
                require(point['maps']['first']==firstpoint['maps']['first'],'First calibration preserved')
                scores={}
                for role,rec in point['scores'].items():
                    scores[role]=rd.array('result/job/run/'+rec['path'],'Full saved blind score matrix, affine roles and predicted counts',rec)
                    expected=(12,378) if role=='calibration' else (60,378)
                    require(scores[role].shape==expected and np.isfinite(scores[role]).all(),'Blind score shape/finite '+name+'/'+role)
                for role,key in [('stage-cal','stage'),('first-cal','first')]:
                    a,b=(point['maps'][key][k] for k in ('a','b'))
                    require(a>0 and np.array_equal(scores[role],scores['raw'].astype(np.float64)*a+b),'Exact positive affine '+name+'/'+role)
                    idx=np.argsort(scores['raw'],axis=1,kind='stable');cal_sort=np.take_along_axis(scores[role],idx,axis=1);raw_sort=np.take_along_axis(scores['raw'],idx,axis=1)
                    require(np.array_equal(np.diff(raw_sort,axis=1)==0,np.diff(cal_sort,axis=1)==0) and np.all(np.diff(cal_sort,axis=1)>=0),'Order and ties '+name+'/'+role)
                if name not in logs:
                    score_values+=sum(a.size for a in scores.values());all_scores[name]=scores
                    point_diagnostics[name]={'stage':stage,'adam_step':point['adam_step'],'updates':288,'history_presentations':len(history),
                        'memory_bytes_recorded':m['bytes'],'members_by_domain':dict(Counter(ud[u] for u in reservoir)),
                        'clipped_steps':sum(r['gradient_norm']>1 for r in tr['rows']),'training_seconds':tr['training_seconds'],
                        'encoder_positive_lr_steps':sum(r['encoder_lr']>0 for r in tr['rows']),
                        'calibration':{k:mapping[k] for k in ('a','b','final_nll','projected_gradient_max')}}
                logs[name]=tr;previous=m
        for stage in (2,3):
            for k in ('current_ids','history_ids'):
                require(logs[f'{order}_C_stage{stage}'][k]==logs[f'{order}_S_stage{stage}'][k],'Cross-arm paired schedule')
    require(len(logs)==15 and sum(t['updates'] for t in logs.values())==4320,'15 physical states/4320 updates')
    require(sum(len(t['history_ids']) for t in logs.values())+4320==7776,'7776 presentations')
    initial=rd.array('result/job/run/'+run['initial']['scores']['path'],'Common initial blind score array',run['initial']['scores'])
    require(initial.shape==(60,378) and np.isfinite(initial).all(),'Initial raw score array');score_values+=initial.size
    all_scores['initial']={'raw':initial}
    # Predicted-positive counts use score signs only; no labels reconstructed/read.
    predicted_count_groups=0
    for name,roles in col['points'].items():
        for role,recs in roles.items():
            c=rd.json('result/job/evaluation/'+recs['counts']['path'],'Score-sign predicted totals vs existing count table',recs['counts'])
            require(np.array_equal((all_scores[name][role]>=0).sum(1),np.array([v['tp']+v['fp'] for v in c])),'Threshold count vs score sign '+name+'/'+role)
            predicted_count_groups+=60
    console=rd.read('result/job.console.txt','All 410 console lines, all progress and resource observations').decode().splitlines()
    events=[json.loads(x) for x in console if x.startswith('{')]
    require(len(events)==195,'15x13 logged progress observations')
    require(all(a['elapsed_seconds']<b['elapsed_seconds'] for a,b in zip(events,events[1:])),'Console time monotonicity')
    for name in logs:
        items=[x for x in events if x['point']==name]
        require([x['completed'] for x in items]==[1,*range(24,289,24)],'Console progress count '+name)
        require(items[-1]['adam_step']==logs[name]['adam_step'],'Final console Adam '+name)
    for x in events:
        for k,v in limits.items():require(x[k]<=resource[k]<=v,'Observed resource trajectory '+k)
    findings['run_records']={'physical_updates':4320,'current_group_presentations':4320,'history_group_presentations':3456,
        'total_group_presentations':7776,'physical_states':15,'logical_points':18,'complete_paired_starts':3,
        'recorded_full_restores':15,'blind_score_arrays':61,'blind_score_values_checked':score_values,
        'score_sign_vs_count_group_checks':predicted_count_groups,'positive_affine_and_ties_exact':True,
        'console_progress_observations':195,'wrapper_seconds':wall,'wrapper_start':wrapper[0],'wrapper_finish':wrapper[2],
        'resource':resource,'before_valid_access':before['access'],'final_access':access,
        'memory_body_verification':'Not available in supplied package; compared retained identities/byte records only'}
    findings['points']=point_diagnostics

    inventory=rd.json('result/inventory.json','All 239 export file identities')
    for rec in inventory['files']:verify('result/',rec,'Actual exported evidence identity')
    pre=rd.json('maintenance/predelete.json','All authorized weight identities, nonweights and exception metadata')
    post=rd.json('maintenance/receipt.json','All deletion entries and retained-evidence assertions')
    rd.read('maintenance/predelete.json','Receipt binds predelete SHA',{'bytes':(Path(root)/'maintenance/predelete.json').stat().st_size,'sha256':post['predelete_sha256']})
    require(rd.read('result/custody/predelete.json','Exported duplicate predelete identity')==rd.read('maintenance/predelete.json','Maintenance predelete identity'),'Predelete duplicate')
    require(post['status']=='COMPLETE' and len(post['deleted'])==len(pre['weights'])==15,'15 recorded deletions')
    require(pre['weights']==post['weights'] and pre['nonweight_files']==post['nonweight_files'] and pre['retained_exception_metadata']==post['retained_exception_metadata'],'Before/after retained metadata lists')
    for p in run['points'].values():
        w=next(x for x in pre['weights'] if x['path']=='reports/job/run/'+p['model']['path'])
        require(all(w[k]==p['model'][k] for k in ('bytes','sha256')),'Weight identity binds stage')
        deleted=next(x for x in post['deleted'] if x['absolute_path']==w['absolute_path'])
        require(deleted['bytes']==w['bytes'] and deleted['sha256']==w['sha256'] and deleted['absent'] is True,'Recorded delete identity')
        require(post['started_at']<=deleted['deleted_at']<=post['finished_at'],'Recorded deletion window')
    present=0
    for rec in pre['nonweight_files']:
        name='result/'+rec['path'].removeprefix('reports/')
        if (Path(root)/name).is_file():rd.read(name,'Nonweight preservation vs predelete identity',rec);present+=1
        else:
            require(name.startswith('result/job/run/memory/') and name.endswith('.bin'),'Unexpected missing preserved file '+name)
            omitted.append({'path':name,'bytes':rec['bytes'],'sha256':rec['sha256'],'reason':'Memory body intentionally excluded by user scope'})
            point=run['points'][Path(name).stem]
            require(point['memory']['sha256']==rec['sha256'] and point['memory']['bytes']==rec['bytes'],'Omitted Memory identity consistency')
    require(present==209 and len(omitted)==15 and len(pre['retained_exception_metadata'])==23,'Preserved nonweight scope')
    released=sum(x['bytes'] for x in post['deleted'])
    require(released==post['logical_bytes_released']==19565561724,'Logical released bytes')
    require(all(post[k] is True for k in ('all_targets_absent','nonweight_sha256_unchanged','retained_exception_metadata_unchanged')),'Recorded deletion final assertions')
    findings['cleanup']={'exported_files_verified':239,'nonweight_hashes_recomputed':209,'nonweight_memory_identity_only':15,
        'retained_exception_metadata_records':23,'deleted_weights':15,'logical_bytes_released':released,
        'weight_bytes_not_available_and_not_requested':True,'remote_absence_is_recorded_not_independently_observed':True,
        'current_result_requires_model_loading':False}
    findings['status']='PASS_SUPPLIED_RUN_RECORDS'
    findings['elapsed_seconds']=time.perf_counter()-started;findings['numeric_max_errors']=diffs
    findings['scope']='Independent read-only operations in reviewer sandbox; all project execution and recovery evidence is supplied historical evidence.'
    findings['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name,obj in [('run_records.json',findings),('machine_read_scope.json',list(rd.events.values())),('intentional_omissions.json',omitted)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in findings.items() if k not in ('points',)},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path)
    a=p.parse_args();main(a.input,a.output,a.archive)
