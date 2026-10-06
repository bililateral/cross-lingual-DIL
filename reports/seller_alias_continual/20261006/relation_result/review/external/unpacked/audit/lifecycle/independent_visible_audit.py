"""Attachment-only audit. Imports no project module, Torch, training or label loader."""
from __future__ import annotations
import ast
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import numpy as np

PACKAGE = Path('/workspace/scratch/f4d639b3b473/relation_memory_result_review')
RESULT = PACKAGE / 'reports/seller_alias_continual/20261006/relation_result'
SOURCE = RESULT / 'source'
RUN = RESULT / 'job/run'
OUT = Path(__file__).parent
checks, failures, verified_files = [], [], set()

def jread(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def canonical(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode('utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def expect(label, result, detail=None):
    checks.append(label)
    if not bool(result): failures.append({'check':label,'detail':detail})

def verify(path, record, label=None):
    path = Path(path)
    label = label or str(path.relative_to(PACKAGE))
    expect(label+':exists', path.is_file())
    if not path.is_file(): return
    expect(label+':bytes', path.stat().st_size == record['bytes'])
    expect(label+':sha256', sha(path) == record['sha256'])
    verified_files.add(str(path.relative_to(PACKAGE)))

def seed(seed0,*parts):
    return int.from_bytes(hashlib.sha256(canonical([seed0,*parts])).digest()[:8], 'big') % (2**63-1)

def load_array(record, shape, dtype, label):
    path = RUN/record['path']
    verify(path,record,label)
    array = np.load(path,allow_pickle=False)
    expect(label+':shape',array.shape == shape)
    expect(label+':dtype',array.dtype == dtype)
    expect(label+':finite',np.isfinite(array).all())
    return array

gate = jread(RESULT/'gate.json')
execution = jread(RESULT/'job/execution.json')
manifest = jread(RUN/'manifest.json')
before = jread(RESULT/'job/before_valid.json')
access = jread(RESULT/'job/access.json')
completion = jread(RESULT/'job/completion.json')
collected = jread(RESULT/'job/evaluation/collected.json')
evaluation = jread(RESULT/'job/evaluation/evaluation.json')
inventory = jread(RESULT/'inventory.json')
sync = jread(RESULT/'sync.json')
policy = jread(SOURCE/'schema/step28_relation_memory_policy.json')

# Recreate the source import closure using AST only, then hash its actual bodies.
relative = {'scripts/step28_relation_memory_run.py','scripts/step28_relation_memory.py',
            'scripts/step28_relation_memory_verify.py','tests/test_step28_relation_memory_run.py',
            'tests/test_step28_relation_memory.py'}
todo = list(relative)
while todo:
    name = todo.pop()
    for node in ast.walk(ast.parse((SOURCE/name).read_text(encoding='utf-8'))):
        imports = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
        for module in imports:
            if module and module.startswith('step28_'):
                child = 'scripts/'+module+'.py'
                if child not in relative:
                    relative.add(child); todo.append(child)
relative.update('schema/'+name for name in ['step28_relation_memory_policy.json',
    'step28_bge_continual_policy.json','step28_alias_ranking_policy.json','step28_chinese_base_policy.json'])
relative.update(['docs/SELLER_ALIAS_RELATION_MEMORY_PILOT.zh.md',
    'docs/SELLER_ALIAS_RELATION_MEMORY_EXECUTION.zh.md','scripts/run_step28_relation_memory_pilot_linux_20261004.sh'])
actual_sources = [{'path':name,'bytes':(SOURCE/name).stat().st_size,'sha256':sha(SOURCE/name)} for name in sorted(relative)]
expect('AST import closure is 23 files',len(actual_sources)==23)
for name,records in [('gate',gate['source_files']),('execution',execution['sources']),
                     ('manifest',manifest['source_files']),('collected',collected['source_files']),
                     ('evaluation',evaluation['source_files'])]:
    expect(name+':actual frozen 23 source bytes equal bindings',records==actual_sources)
for rec in actual_sources: verified_files.add(str((SOURCE/rec['path']).relative_to(PACKAGE)))
verify(RESULT/'gate.json',execution['gate'],'execution gate binding (relocated reports/gate.json)')
for name in ['native','integration_cpu','review_main','external_review']:
    verify(PACKAGE/gate[name]['path'],gate[name],'gate prerequisite '+name)
native = jread(PACKAGE/gate['native']['path'])
integration = jread(PACKAGE/gate['integration_cpu']['path'])
expect('native core hash matches actual frozen implementation',native['scientific_sources']['scripts/step28_relation_memory.py']==sha(SOURCE/'scripts/step28_relation_memory.py'))
expect('native handwritten status',native['status']=='PASS_HANDWRITTEN_ONLY')
expect('integration 23 sources match actual bodies',integration['source_files']==actual_sources)
expect('integration status',integration['status']=='PASS_HANDWRITTEN_ONLY')
expect('gate qualification',gate['status']=='APPROVED_RELATION_PILOT' and gate['review_disposition']=='NO_OPEN_BLOCKERS')
expect('gate policy identity',gate['runtime']==policy['runtime'] and gate['supervision']==policy['supervision'])

verify(RUN/manifest['partition']['path'],manifest['partition'])
partition = jread(RUN/'partition.json')
domain = {row['group_uid']:row['domain'] for role in partition.values() for row in role}
all_ids = [r['group_uid'] for rows in partition.values() for r in rows]
expect('all 240 fit/calibration/valid group IDs distinct',len(all_ids)==240 and len(set(all_ids))==240)
for role,count in [('fit',48),('calibration',12),('development',20)]:
    expect(role+':domain sizes',collections.Counter(r['domain'] for r in partition[role])==dict.fromkeys('ABC',count))
partition_seed = jread(SOURCE/'schema/step28_chinese_base_policy.json')['partition']['seed']
for d in 'ABC':
    union = [r['group_uid'] for k in ['fit','calibration'] for r in partition[k] if r['domain']==d]
    expected_fit = set(sorted(union,key=lambda uid:hashlib.sha256(canonical([partition_seed,d,uid])).hexdigest())[:48])
    expect(d+':deterministic public-ID fit/cal partition',expected_fit=={r['group_uid'] for r in partition['fit'] if r['domain']==d})
expect('collected valid identity/order',collected['group_ids']==[r['group_uid'] for r in partition['development']])
expect('collected valid domain order',collected['domains']==[r['domain'] for r in partition['development']])
for rec in policy['reference']['collections']:
    reference_path = PACKAGE/policy['reference']['local_root']/rec['path']
    verify(reference_path,rec,'saved baseline collection '+rec['path'])
    baseline = jread(reference_path)
    expect('baseline pairing '+rec['path'],all(baseline[k]==collected[k] for k in ['group_ids','domains','metric_columns']))

points = [f'{order}_relation_stage{stage}' for order in ['ABC','BCA','CAB'] for stage in [1,2,3]]
expect('manifest point coverage',set(manifest['points'])==set(points))
expect('manifest trace coverage',set(manifest['training'])==set(points))
initial = load_array(manifest['initial'],(60,378),np.dtype('float32'),'initial blind scores')
expect('initial model digest format',len(manifest['initial_digest'])==64)
point_summaries=[]
total_updates=0
total_presentations=0
memory_receipts=[]
model_receipts=[]
stage1_raw_hashes=[]
schedule_seed = jread(SOURCE/'schema/step28_bge_continual_policy.json')['schedule_seed']
for order in ['ABC','BCA','CAB']:
    survivors=[]
    seen=0
    retention_rng=random.Random(seed(20260930,order,'retention'))
    first_map=None
    for stage in [1,2,3]:
        name=f'{order}_relation_stage{stage}'
        verify(RUN/manifest['points'][name]['path'],manifest['points'][name])
        verify(RUN/manifest['training'][name]['path'],manifest['training'][name])
        point=jread(RUN/manifest['points'][name]['path'])
        tr=jread(RUN/manifest['training'][name]['path'])
        fit=[r['group_uid'] for r in partition['fit'] if r['domain']==order[stage-1]]
        current_rng=random.Random(seed(schedule_seed,order,stage,'current'))
        schedule=[]
        for _ in range(6):
            epoch=sorted(fit);current_rng.shuffle(epoch);schedule.extend(epoch)
        expect(name+':full independently reconstructed current sequence',tr['current_ids']==schedule)
        expect(name+':each fit group appears exactly six times',collections.Counter(tr['current_ids'])==dict.fromkeys(fit,6))
        draw_rng=random.Random(seed(20260930,order,stage,'history_draws'))
        history=[survivors[draw_rng.randrange(6)] for _ in range(288)] if stage>1 else [None]*288
        expect(name+':full independently reconstructed history sequence',tr['history_ids']==history)
        expect(name+':no current/previous overlap',not(set(fit)&set(survivors)))
        expect(name+':identity and actual step endpoint',all(point[k]==tr[k]==v for k,v in [('order',order),('stage',stage),('adam_step',stage*288)]))
        expect(name+':288 row updates',len(tr['updates'])==288)
        for i,row in enumerate(tr['updates'],1):
            expect(name+f':update {i} phase',row['stage']==stage and row['step']==i and row['adam_step']==(stage-1)*288+i)
            lr=1e-5*(i/29 if i<=29 else (288-i)/259)
            expect(name+f':update {i} learning rates',row['encoder_lr']==lr and row['head_lr']==.001)
            expect(name+f':update {i} finite gradients/loss',all(math.isfinite(row[k]) for k in ['current','history','total','gradient_norm']) and row['gradient_norm']>0)
            expect(name+f':update {i} objective sum',row['total']==row['current']+row['history'] and (stage>1 or row['history']==0.))
        total_updates+=len(tr['updates'])
        total_presentations+=len(tr['updates'])+sum(uid is not None for uid in history)
        old_members=list(survivors)
        old_count=seen
        if stage<3:
            for uid in sorted(fit):
                seen+=1
                slot=len(survivors) if len(survivors)<6 else retention_rng.randrange(seen)
                if slot<6:
                    if slot==len(survivors):survivors.append(uid)
                    else:survivors[slot]=uid
        expect(name+':independent Algorithm R complete summary',point['memory_summary']=={'members':survivors,'count':seen,'seen':seen,'stage':min(stage,2)})
        expect(name+':six real fitting-group IDs in past domains',len(survivors)==6 and len(set(survivors))==6 and all(domain[uid] in order[:min(stage,2)] for uid in survivors))
        if stage<3:
            ret=point['retention']
            expect(name+':retention counters and members',all(ret[k]==v for k,v in [('old_count',old_count),('old_members',old_members),('new_count',seen),('new_members',survivors),('reservoir_seen',seen),('serialized_bytes',point['memory_bytes'])]))
            expect(name+':reported H PSD gate',math.isfinite(ret['h_min_eigenvalue']) and ret['h_min_eigenvalue']>=-ret['h_psd_tolerance'])
        else:
            expect(name+':no terminal successor consolidation',point['retention'] is None and seen==96)
        expect(name+':complete memory reported <= 1MiB',0<point['memory_bytes']==point['memory']['bytes']<=1048576)
        expect(name+':reported full restore and disposable full file',point['full_restore_verified'] is True and point['full_state']['bytes']==point['intermediate_deleted_bytes'] and point['full_state']['bytes']>point['model']['bytes'])
        expect(name+':memory body intentionally absent',not(RUN/point['memory']['path']).exists())
        expect(name+':model body intentionally absent',not(RUN/point['model']['path']).exists())
        memory_receipts.append({'path':'job/run/'+point['memory']['path'],'bytes':point['memory']['bytes'],'sha256':point['memory']['sha256']})
        model_receipts.append(point['model'])
        verify(RUN/point['mapping']['path'],point['mapping'])
        mapping=jread(RUN/point['mapping']['path'])
        cal_ids=[r['group_uid'] for r in partition['calibration'] if r['domain']==order[stage-1]]
        expect(name+':current-only calibration IDs',point['calibration_ids']==cal_ids)
        expect(name+':calibration scope and success',mapping['status']=='PASS_CALIBRATION_FIT' and mapping['group_count']==12 and mapping['pair_count']==4536 and mapping['positive_count']==240 and mapping['role']=='calibration')
        expect(name+':map identity',point['maps']['stage']=={k:mapping[k] for k in ['a','b']})
        if stage==1:first_map=point['maps']['stage']
        expect(name+':own first-stage map persists',point['maps']['first']==first_map)
        raw=load_array(point['scores']['raw'],(60,378),np.dtype('float32'),name+' raw')
        load_array(point['scores']['calibration'],(12,378),np.dtype('float32'),name+' current calibration')
        if stage==1:
            stage1_raw_hashes.append(point['scores']['raw']['sha256'])
            expect(name+':trained scores differ from initialization',not np.array_equal(raw,initial))
        for role,key in [('stage-cal','stage'),('first-cal','first')]:
            arr=load_array(point['scores'][role],(60,378),np.dtype('float64'),name+' '+role)
            a,b=point['maps'][key]['a'],point['maps'][key]['b']
            expect(name+':'+role+' positive finite bounded affine map',math.isfinite(a) and math.isfinite(b) and .001<=a<=100 and -100<=b<=100)
            expect(name+':'+role+' exact independent affine transform',np.array_equal(arr,a*raw.astype(np.float64)+b))
            ix=np.argsort(raw,axis=1,kind='stable')
            expect(name+':'+role+' preserves full group order',np.array_equal(ix,np.argsort(arr,axis=1,kind='stable')))
            rsorted=np.take_along_axis(raw,ix,axis=1)
            asorted=np.take_along_axis(arr,ix,axis=1)
            expect(name+':'+role+' preserves exact ties',np.array_equal(np.diff(rsorted,axis=1)==0,np.diff(asorted,axis=1)==0))
        point_summaries.append({'name':name,'updates':len(tr['updates']),'historical_presentations':sum(uid is not None for uid in history),
            'adam_step':tr['adam_step'],'fit_domain':order[stage-1],'historical_domain_counts':dict(collections.Counter(domain[u] for u in survivors)),
            'memory_bytes_reported':point['memory_bytes'],'statistic_count_reported':seen,'full_restore_reported':point['full_restore_verified'],
            'first_map':point['maps']['first'],'stage_map':point['maps']['stage']})

expect('three own first stages have distinct blind score files',len(set(stage1_raw_hashes))==3)
expect('all physical updates independently counted',total_updates==2592==manifest['physical_updates']==completion['physical_updates'])
expect('all gradient group presentations independently counted',total_presentations==4320==manifest['gradient_group_presentations'])
expect('complete blind manifest status',manifest['status']=='COMPLETE_RELATION_2592_UPDATES_9_ENDPOINTS_VALID_BLIND')
verify(RESULT/'job'/before['manifest']['path'],before['manifest'],'before_valid bound completed manifest')
expect('before_valid gate identity',before['status']=='PASS_COMPLETE_BLIND_GATE' and before['points']==9)
expect('before_valid ledger',before['access']==dict(train=1,valid=0,heldout=0,owners=0))
expect('final ledger equals completion',access==completion['access']==dict(train=1,valid=1,heldout=0,owners=0))
expect('final completion status',completion['status']=='COMPLETE_RELATION_FIXED_POINT')
verify(RESULT/'job'/completion['evaluation']['path'],completion['evaluation'],'completion bound evaluation')
verify(RESULT/'job/evaluation'/evaluation['collected']['path'],evaluation['collected'],'evaluation bound collection')
expect('statistics cannot independently promote completion',evaluation['status']=='STATISTICS_COMPLETE_REQUIRES_VALID_COMPLETION')
expect('completion decision equals evidence',completion['worth_matched_replay']==evaluation['observed_continuation_checks_pass']==all(evaluation['continuation_checks'].values()))
expect('28 score-derived metric count sets',sum(len(v) for v in collected['points'].values())==28)
expect('all expected collected endpoints',set(collected['points'])==set(points+['initial']))
metric_count_sets=0
for name,roles in collected['points'].items():
    expect(name+':collected roles',set(roles)==({'raw'} if name=='initial' else {'raw','stage-cal','first-cal'}))
    for role,records in roles.items():
        for kind,rec in records.items():verify(RESULT/'job/evaluation'/rec['path'],rec)
        arr=np.load(RESULT/'job/evaluation'/records['matrix']['path'],allow_pickle=False)
        expect(name+' '+role+':metric matrix shape finite',arr.shape==(60,len(collected['metric_columns'])) and np.isfinite(arr).all())
        metric_count_sets+=1

# Inventory identities of absent bodies are receipts, never fresh body hashes.
inventory_index={rec['path']:rec for rec in inventory['files']}
absent_inventory=[]
for rec in inventory['files']:
    path=RESULT/rec['path']
    if path.exists():verify(path,rec,'inventory '+rec['path'])
    else:absent_inventory.append(rec)
expect('only nine cache bodies absent from inventory payload',sorted(absent_inventory,key=lambda r:r['path'])==sorted(memory_receipts,key=lambda r:r['path']))
for rec in memory_receipts:
    expect('cache receipt metadata agrees with server inventory '+rec['path'],rec==inventory_index[rec['path']])
retained_models=inventory['retained_model_files']
expect('nine inference model receipts',len(retained_models)==9)
for rec in model_receipts:
    matches=[r for r in retained_models if r['path'].endswith('/job/run/'+rec['path'])]
    expect('model receipt matches server inventory '+rec['path'],len(matches)==1 and all(matches[0][k]==rec[k] for k in ['bytes','sha256']))
expect('no failure/recovery artifacts in supplied inventory',not any('failure' in r['path'] or 'recovery_from' in r['path'] for r in inventory['files']))
wrapper=dict(line.split('=',1) for line in (RESULT/'job.wrapper.txt').read_text().splitlines() if '=' in line)
expect('wrapper normal completion matches sync',wrapper['exit_code']=='0' and sync['formal_exit_code']==0 and wrapper['started_at']==sync['formal_started_at'] and wrapper['ended_at']==sync['formal_ended_at'])
console=[]
for line in (RESULT/'job.console.txt').read_text().splitlines():
    if line.startswith('{'):console.append(json.loads(line))
expected_progress=[(o,s,i) for o in ['ABC','BCA','CAB'] for s in [1,2,3] for i in range(24,289,24)]
expect('108 chronological console checkpoints',[(r['order'],r['stage'],r['step']) for r in console]==expected_progress)
expect('console elapsed increases',all(a['elapsed_seconds']<b['elapsed_seconds'] for a,b in zip(console,console[1:])))
expect('console ends before completion snapshot',console[-1]['elapsed_seconds']<completion['budget']['elapsed_seconds'])

result={'scope':'Independent attachment-only Python standard library plus NumPy; no project imports, Torch, Linux server, training, model/cache bodies or label access.',
        'status':'PASS_VISIBLE_LIFECYCLE_EVIDENCE' if not failures else 'ISSUES',
        'checks':len(checks),'failures':failures,'unique_present_file_hashes_verified':len(verified_files),
        'source_files_recreated_by_AST':len(actual_sources),'physical_updates_independently_counted':total_updates,
        'gradient_group_presentations_independently_counted':total_presentations,'metric_count_sets_hashed':metric_count_sets,
        'console_progress_rows':len(console),'point_summaries':point_summaries,
        'memory_bytes_min_reported':min(p['memory_bytes_reported'] for p in point_summaries),
        'memory_bytes_max_reported':max(p['memory_bytes_reported'] for p in point_summaries),
        'absent_body_scope':{'cache_bodies':len(absent_inventory),'model_bodies':len(retained_models),
           'freshly_hashed':False,'actual_state_or_statistic_reconstruction':False,
           'timing_scope':'Frozen control flow plus before_valid/final ledger and monotonic console; not an OS-level file-open audit or wall timestamp for the valid parse.'},
        'verified_present_files':sorted(verified_files)}
(OUT/'independent_visible_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['point_summaries','verified_present_files']},ensure_ascii=False,indent=2))
sys.exit(bool(failures))
