#!/usr/bin/env python3
"""Read-only replay of saved positive-ranking wiring and native evidence.

Uses the Python standard library only. It does not import project modules,
PyTorch, models, labels, or Memory payloads and does not execute training.
"""
import collections,datetime,hashlib,json,math,resource,shlex,sys,time,traceback
from pathlib import Path

ROOT=Path('/workspace/scratch/0f307e99d302/review_input')
W=ROOT/'workspace'
OUT=Path(__file__).parent
START=time.monotonic(); CPU=time.process_time()
STAMP=datetime.datetime.now(datetime.timezone.utc).isoformat()
inputs={}; checks=[]; failures=[]; result={}

def digest(path,purpose):
    blob=path.read_bytes()
    name=path.relative_to(ROOT).as_posix()
    inputs[name]={'path':name,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),
                  'mode':'mechanical_parse_or_byte_identity','purpose':purpose}
    return blob

def read(rel,purpose):
    return json.loads(digest(W/rel,purpose))

def check(name,condition,detail=None):
    row={'check':name,'passed':bool(condition)}
    if detail is not None:row['detail']=detail
    checks.append(row)
    if not condition:failures.append(row)

def source_record(path,purpose):
    b=digest(path,purpose)
    return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def close(a,b):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-6+2e-6*abs(b)

def extremes(values):
    return {'count':len(values),'min':min(values),'max':max(values),'mean':math.fsum(values)/len(values)}

try:
    manifest=read('reports/job/run/manifest.json','Formal source identities, own-state continuity and endpoint metadata')
    policy=read('schema/step28_positive_ranking_policy.json','Frozen candidate metadata and budgets')
    native=read('reports/qualification/native/result.json','All saved native call logs and parameter/Adam evidence')
    native_wall=read('reports/launch/native_wallclock.json','Outer native exit, timing and PID')
    acceptance=read('reports/launch/native_acceptance.json','Native receipt and saved projection')
    gate=read('reports/documentation/20261016/positive_ranking_implementation/qualification/formal_gate.json','Source-bound formal admission and native identity')
    execution=read('reports/job/execution.json','Actual executed formal gate and frozen sources')
    source_checks=[]
    for s in manifest['source_files']:
        expected={k:s[k] for k in ('bytes','sha256')}
        live=source_record(W/s['path'],'40-source byte identity; semantic source reading recorded separately')
        frozen=source_record(W/'reports/job/run/sources'/s['path'],'40 frozen run sources byte identity')
        source_checks.append({'path':s['path'],'expected':expected,'workspace':live,'frozen':frozen,
                              'passed':live==frozen==expected})
    check('40 sources each match workspace and frozen run copy',len(source_checks)==40 and all(x['passed'] for x in source_checks))
    for name,values in [('native',native['source_files']),('formal_gate',gate['source_files']),('execution',execution['sources'])]:
        check('source list equality: '+name,values==manifest['source_files'])
    result['sources']={'count':len(source_checks),'identities':source_checks}
    check('formal gate refers to exact native result',gate['gpu']=={'path':'reports/qualification/native/result.json',**source_record(W/'reports/qualification/native/result.json','Exact formal-gate native receipt')})
    check('native acceptance refers to exact result',acceptance['gpu']==gate['gpu'])
    check('native outer successful within 1200 seconds',native_wall['exit_code']==0 and native_wall['charged_seconds']<=1200)
    check('native 12 actual calls, no formal inputs/labels',native['actual_updates']==12 and native['formal_inputs'] is False and native['formal_labels'] is False)
    result['native_outer']={k:native_wall[k] for k in ['exit_code','elapsed_seconds','charged_seconds','pid','command']}
    result['native_internal']={k:native[k] for k in ['elapsed_seconds','cumulative_charged_seconds','max_reserved_bytes','peak_rss_bytes','peak_work_bytes','torch','status','numerics']}
    native_results={}
    for arm in ['S_pos','H_pos']:
        a=native['arms'][arm]; calls=a['updates']
        expected=[(arm+'_old_rank','allocation',1),(arm+'_old_rank','A',2),(arm,'A',2),
                  (arm+'_old_rank','B',49),(arm,'B',49),(arm,'B',49)]
        check(arm+' exactly six intended optimizer calls',a['actual_updates']==6 and len(calls)==6 and [(x['arm'],x['phase'],x['step']) for x in calls]==expected)
        check(arm+' restored/capture repeated call logs identical',calls[4]['log']==calls[5]['log'])
        check(arm+' source metadata independent initialization',a['metadata']==policy['candidates'][arm] and a['initial_state']==manifest['initials'][arm])
        check(arm+' per-arm time bound',a['charged_seconds']<=600)
        pair_checks=[]
        for phase,off_idx,on_idx in [('current',1,2),('paired',3,4)]:
            off,on=calls[off_idx]['log'],calls[on_idx]['log']
            for role in ['current']+(['history'] if phase=='paired' else []):
                r0,r1=off[role],on[role]
                equal_fields=['mse0','mse1','assignment_sha256','teacher_contribution']
                okay=(r0['qualification_old_rank'] is True and r1['qualification_old_rank'] is False
                      and all(r0[k]==r1[k] for k in equal_fields)
                      and all(all(v0[k]==v1[k] for k in ['bce','hard','rank_pos']) for v0,v1 in zip(r0['views'],r1['views'])))
                check(arm+' native old/new controlled terms '+phase+'/'+role,okay)
                pair_checks.append({'phase':phase,'role':role,'teacher_contribution':r1['teacher_contribution'],
                                    'old_rank_used':[v['rank_used'] for v in r0['views']],
                                    'new_rank_used':[v['rank_used'] for v in r1['views']],
                                    'old_weighted':r0['weighted'],'new_weighted':r1['weighted']})
        effects={}
        for phase,components in a['effects'].items():
            effects[phase]={}
            for component,e in components.items():
                adam=e['adam']['parameters']; active=[r for r in adam if r['state_present']]
                clock=2. if phase=='current' else 337.
                check(arm+'/'+phase+'/'+component+' parameter/gradient effects positive',e['parameter_max_delta']>0 and e['post_clip_gradient_max_delta']>0 and e['pre_clip_gradient_max_delta']>0 and e['finite'])
                check(arm+'/'+phase+'/'+component+' complete participating Adam states and clock',len(active)==e['active_grad_parameters'] and all(r['step']['off']==r['step']['on']==clock and r['step']['max_abs_delta']==0 for r in active))
                check(arm+'/'+phase+'/'+component+' Adam moment effects',max(r['exp_avg']['max_abs_delta'] for r in active)>0 and max(r['exp_avg_sq']['max_abs_delta'] for r in active)>0)
                if phase=='paired':check(arm+'/'+component+' isolated current/history probes change',all(v>0 for v in e['isolated_branch_preclip_deltas'].values()))
                effects[phase][component]={k:v for k,v in e.items() if k not in ['active_parameter_names','adam']}
                effects[phase][component]['adam_summary']={'entries':len(adam),'participating':len(active),
                    'exp_avg_max_delta':max(r['exp_avg']['max_abs_delta'] for r in active),
                    'exp_avg_sq_max_delta':max(r['exp_avg_sq']['max_abs_delta'] for r in active),
                    'off_and_on_step':clock,
                    'parameters_with_nonzero_exp_avg_delta':sum(r['exp_avg']['max_abs_delta']>0 for r in active),
                    'parameters_with_nonzero_exp_avg_sq_delta':sum(r['exp_avg_sq']['max_abs_delta']>0 for r in active)}
        rows=a['table_precision']
        check(arm+' 5 native table checks, each A0/A1',len(rows)==5 and all([v['view'] for v in r['views']]==['A0','A1'] for r in rows))
        check(arm+' saved rank value/gradient errors finite and within declared tolerance',all(close(v['value'],v['reference']) and math.isfinite(v['gradient_max_error']) and v['gradient_max_error']<=2e-6 for r in rows for v in r['views']))
        diag=read(f'reports/qualification/native/{arm}/diagnostics/handwritten_epoch1.json','Real handwritten native 48/6/40 diagnostic metadata')
        check(arm+' saved real native diagnostic scope',len(diag['fit_ids'])==48 and len(diag['cache_ids'])==6 and len(diag['valid_ids'])==40 and diag['valid_labels_used'] is False and diag['adam_step']==336)
        identity=read(f'reports/qualification/native/{arm}/temporary_checkpoint_identity.json','Native temporary checkpoint identities and successful cleanup evidence')
        check(arm+' temporary payload cleanup recorded',identity['removed'] is True)
        native_results[arm]={k:a[k] for k in ['actual_updates','charged_seconds','metadata','initial_state','resources','measurements','equality_counts','next_update_bitwise_equal','pooling_restores_verified','fixture','residuals']}
        native_results[arm].update(effects=effects,objective_pairs=pair_checks,table_precision=rows)
    check('second native arm reports exact own context and zero allocated/reserved',native_results['H_pos']['resources']['allowed_context_pid']==native_wall['pid'] and native_results['H_pos']['resources']['live_allocated_bytes']==native_results['H_pos']['resources']['reserved_bytes']==0)
    result['native']=native_results
    formal={}; all_logs={}; arithmetic_max=collections.defaultdict(float); branch_count=0; view_count=0
    for name in ['A_S_pos','B_S_pos','A_H_pos','B_H_pos']:
        log=read('reports/job/run/updates/'+name+'.json','Independent arithmetic over every saved formal optimizer update')
        all_logs[name]=log; stage=1 if name[0]=='A' else 2; arm=name[2:]
        check(name+' row and optimizer counts',len(log['rows'])==log['updates']==288 and log['adam_step']==stage*288)
        check(name+' own A to B continuity',stage==1 or manifest['starts'][arm]==manifest['points']['A_'+arm]['continuity'])
        measurements=collections.defaultdict(list); mismatches=[]
        for index,row in enumerate(log['rows'],1):
            if row['step']!=index or any(row[k]!=v for k,v in policy['candidates'][arm].items()):mismatches.append([index,'identity'])
            expected_total=0.
            for role,weight in [('current',1.)]+([('history',.1)] if stage==2 else []):
                branch_count+=1;r=row[role]
                if r['qualification_old_rank'] is not False or r['mse1'] is not None or len(r['views'])!=2:mismatches.append([index,role,'rank/teacher/views'])
                if role=='current' and (r['mse0']!=0 or r['teacher_contribution']!=0):mismatches.append([index,role,'current_teacher'])
                expected_supervised=0.
                for vi,v in enumerate(r['views']):
                    view_count+=1
                    wanted=v['bce']+v['rank_pos']+.5*v['hard']
                    arithmetic_max['view_total']=max(arithmetic_max['view_total'],abs(v['total']-wanted))
                    if v['rank_used']!=v['rank_pos'] or not close(v['total'],wanted):mismatches.append([index,role,vi,'rank_used/total'])
                    expected_supervised+=v['total']/2
                    measurements[role+'_rank_pos'].append(v['rank_pos'])
                formulas=[('supervised',expected_supervised),('distillation',r['mse0']),('supervised_contribution',weight*r['supervised']),('teacher_contribution',.5*r['mse0']),('weighted',weight*r['supervised']+.5*r['mse0'])]
                for key,wanted in formulas:
                    arithmetic_max[key]=max(arithmetic_max[key],abs(r[key]-wanted))
                    if not close(r[key],wanted):mismatches.append([index,role,key])
                expected_total+=weight*r['supervised']+.5*r['mse0']
                for key in ['record_gradient_norm','supervised_contribution','teacher_contribution','mse0']:
                    measurements[role+'_'+key].append(r[key])
            arithmetic_max['optimizer_total']=max(arithmetic_max['optimizer_total'],abs(row['total']-expected_total))
            if not close(row['total'],expected_total):mismatches.append([index,'total'])
            if stage==1 and ('history' in row or row['history_uid'] is not None):mismatches.append([index,'unexpected_history'])
            measurements['gradient_norm'].append(row['gradient_norm'])
        check(name+' all saved loss terms and weights',not mismatches,mismatches)
        point=manifest['points'][name]
        check(name+' restored scoring source identities',all(point['scores'][prefix+'-'+role]['sha256']==point['scores'][role]['sha256'] for prefix in ['full','inference'] for role in ['raw','calibration']))
        check(name+' endpoint metadata',point['adam_step']==stage*288 and point['full_restore_verified'] and point['inference_restore_verified'] and all(point[k]==v for k,v in policy['candidates'][arm].items()))
        if stage==2:
            a=manifest['points']['A_'+arm]['memory_summary'];b=point['memory_summary']
            check(name+' own unrefreshed teacher table identities',all(a[k]==b[k]==log['memory_before'][k]==log['memory_after'][k] for k in ['members','origins','tables','seen']) and a['count']==0 and b['count']==288)
            check(name+' all history teacher terms nonzero',min(measurements['history_teacher_contribution'])>0)
        formal[name]={'updates':288,'adam_step':log['adam_step'],'measurements':{k:extremes(v) for k,v in measurements.items()},
                      'memory_summary':point['memory_summary'],'full_restore_verified':point['full_restore_verified'],'inference_restore_verified':point['inference_restore_verified']}
    for stage in ['A','B']:
        a,b=all_logs[stage+'_S_pos'],all_logs[stage+'_H_pos']
        check(stage+' shared current/history schedule and A1 hashes',a['current_ids']==b['current_ids'] and a['history_ids']==b['history_ids'] and all(x['current']['assignment_sha256']==y['current']['assignment_sha256'] and (stage=='A' or x['history']['assignment_sha256']==y['history']['assignment_sha256']) for x,y in zip(a['rows'],b['rows'])))
    check('total formal groups/views',branch_count==1728 and view_count==3456)
    result['formal']={'optimizer_updates':sum(v['updates'] for v in formal.values()),'gradient_group_logs':branch_count,'rank_view_logs':view_count,'maximum_scalar_rounding_errors':dict(arithmetic_max),'points':formal}
    result['status']='PASS_SAVED_WIRING_REPLAY' if not failures else 'FAIL_SAVED_WIRING_REPLAY'
except Exception:
    result['status']='AUDITOR_SCRIPT_ERROR'
    result['traceback']=traceback.format_exc()
    failures.append({'check':'script execution','passed':False,'detail':result['traceback']})

result['checks']=checks;result['failures']=failures
result['scope_limits']=['No project code imported; no torch or numpy imported.','No formal text, labels, Memory or weights accessed.','No training, GPU operation, network or project-server access.','Native gradient/Adam maxima and restore counts are read from preserved project evidence; they are not recalculated from model tensors in this review.','Formal scalar arithmetic and byte identities are independently recomputed from saved evidence.']
result['resource']={'start_utc':STAMP,'wall_seconds':time.monotonic()-START,'cpu_seconds':time.process_time()-CPU,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'python':sys.version}
(OUT/'wiring_replay_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(OUT/'mechanically_read_inputs.json').write_text(json.dumps(list(inputs.values()),ensure_ascii=False,indent=2)+'\n')
summary={k:result[k] for k in ['status','failures','native_outer','native_internal','formal','resource'] if k in result}
summary['checks_count']=len(checks);summary['mechanically_read_inputs']=len(inputs)
output=json.dumps(summary,ensure_ascii=False,indent=2)+'\n'
print(output,end='')
code=0 if not failures else 1
(OUT/'wiring_replay.stdout.txt').write_text(output)
(OUT/'wiring_replay.execution.json').write_text(json.dumps({'command':shlex.join([sys.executable,str(Path(__file__).resolve())]),'cwd':str(Path.cwd()),'exit_code':code,'stdout':'wiring_replay.stdout.txt','stderr':'','resource':result['resource'],'input_list':'mechanically_read_inputs.json'},ensure_ascii=False,indent=2)+'\n')
sys.exit(code)
