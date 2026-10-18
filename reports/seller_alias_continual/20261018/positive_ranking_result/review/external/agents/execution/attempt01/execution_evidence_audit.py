#!/usr/bin/env python3
"""Read-only, standard-library replay of saved execution evidence. No production imports."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time
import zipfile

ROOT = Path('/workspace/scratch/0f307e99d302/review_input')
W = ROOT / 'workspace'
OUT = Path(__file__).resolve().parent
START = time.perf_counter()
READS = {}
CHECKS = []
SUMMARY = {}

def read(path, mode='mechanical_byte_identity', fields=None):
    path = Path(path)
    blob = path.read_bytes()
    key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    row = READS.setdefault(key, {'path': key, 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest(), 'read_modes': [], 'fields': []})
    if mode not in row['read_modes']: row['read_modes'].append(mode)
    if fields:
        for field in fields:
            if field not in row['fields']: row['fields'].append(field)
    return blob

def j(path, fields=None):
    return json.loads(read(path, 'structured_saved_evidence', fields))

def identity(path):
    b = read(path)
    return {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}

def check(name, cond, details=None):
    CHECKS.append({'name': name, 'pass': bool(cond), **({'details': details} if details is not None else {})})

def file_rec(base, r, name):
    path = base/r['path']
    check(name + ':exists', path.is_file(), r['path'])
    if path.is_file():
        got = identity(path)
        check(name+':identity', got == {'bytes': r['bytes'], 'sha256': r['sha256']}, {'path': str(path.relative_to(ROOT)), 'actual': got})

def npy_meta(path):
    b = read(path, 'npy_header_and_byte_identity')
    if b[:6] != b'\x93NUMPY': raise ValueError('NPY header absent: '+str(path))
    major = b[6]
    sz = 2 if major == 1 else 4
    n = int.from_bytes(b[8:8+sz], 'little')
    m = ast.literal_eval(b[8+sz:8+sz+n].decode('latin1').strip())
    m['payload_sha256'] = hashlib.sha256(b[8+sz+n:]).hexdigest()
    m['payload_bytes'] = len(b)-8-sz-n
    return m

def compact_effect(v):
    rows = v['adam']['parameters']
    active = [x for x in rows if x['state_present']]
    return {
        'active_grad_parameters': v['active_grad_parameters'],
        'pre_clip_gradient_max_delta': v['pre_clip_gradient_max_delta'],
        'post_clip_gradient_max_delta': v['post_clip_gradient_max_delta'],
        'parameter_max_delta': v['parameter_max_delta'],
        'isolated_branch_preclip_deltas': v.get('isolated_branch_preclip_deltas'),
        'adam_parameter_records': len(rows), 'adam_active_records': len(active),
        'adam_max_exp_avg_delta': max(x['exp_avg']['max_abs_delta'] for x in active),
        'adam_max_exp_avg_sq_delta': max(x['exp_avg_sq']['max_abs_delta'] for x in active),
        'adam_steps_on_off': sorted(set((x['step']['on'], x['step']['off']) for x in active))}

def main():
    zpath = Path('/workspace/scratch/0f307e99d302/upload/positive_ranking_result_review.zip')
    ident = identity(zpath)
    with zipfile.ZipFile(zpath) as z:
        ident['zip_members'] = len(z.infolist())
        ident['bad_crc_member'] = z.testzip()
    check('authorized_archive_identity', ident == {'bytes': 5481875, 'sha256': '4eeeb5df0e93b6c32bf352f42b656b945a8d4e4f71078b73fa4df0e015ae63f7', 'zip_members': 389, 'bad_crc_member': None}, ident)
    SUMMARY['archive_identity'] = ident
    manifests = j(ROOT/'manifest.json')
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    check('package_exact_members', len(manifests) == 388 and actual == {x['path'] for x in manifests}|{'manifest.json'})
    for k,r in enumerate(manifests): file_rec(ROOT,r,'package_'+str(k))
    SUMMARY['package_payloads'] = len(manifests)

    # Required documents were read in full in the review session, not treated as hash-only.
    for rel in ['request.zh.txt','workspace/docs/SELLER_ALIAS_POSITIVE_RANKING.zh.md','workspace/reports/result_sync/report.zh.txt']:
        read(ROOT/rel,'human_full_text')
    q = W/'reports/documentation/20261016/positive_ranking_implementation/qualification'
    gate = j(q/'formal_gate.json')
    source = gate['source_files']
    source_names = {r['path'] for r in source}
    check('40_unique_sources',len(source)==len(source_names)==40)
    bindings = [(q/'native_gate.json','source_files'),(q/'cpu_gate_workspace.json','source_files'),(q/'execution_authorization.json','source_files'),
                (W/'reports/qualification/native/result.json','source_files'),(W/'reports/job/execution.json','sources'),
                (W/'reports/job/before_valid.json','source_files'),(W/'reports/job/completion.json','source_files'),
                (W/'reports/job/run/startup.json','source_files'),(W/'reports/job/run/progress.json','source_files'),
                (W/'reports/job/run/manifest.json','source_files'),(W/'reports/job/evaluation/collected.json','source_files'),
                (W/'reports/job/evaluation/evaluation.json','source_files')]
    for path,key in bindings: check('source_binding:'+str(path.relative_to(W)),j(path,[key])[key]==source)
    for r in source:
        file_rec(W,r,'current_source:'+r['path'])
        file_rec(W/'reports/job/run/sources',r,'frozen_source:'+r['path'])
    actual_copies = {str(p.relative_to(W/'reports/job/run/sources')) for p in (W/'reports/job/run/sources').rglob('*') if p.is_file()}
    check('exact_frozen_source_copy_set',actual_copies==source_names)
    # Independently reconstruct the existing source closure without importing a training module.
    pending = {'scripts/step28_positive_ranking_run.py','scripts/step28_positive_ranking_verify.py','tests/test_step28_positive_ranking.py','tests/test_step28_g_half.py','tests/test_step28_record_replay.py'}
    seen=set(pending)
    while pending:
        rel=pending.pop()
        for node in ast.walk(ast.parse(read(W/rel,'AST_import_closure').decode())):
            names = [x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for name in names:
                if name and name.startswith('step28_'):
                    child='scripts/'+name+'.py'
                    if child not in seen: seen.add(child);pending.add(child)
    seen.update('schema/'+n for n in ['step28_g_half_policy.json','step28_record_replay_policy.json','step28_bge_continual_policy.json','step28_alias_ranking_policy.json','step28_chinese_base_policy.json','step28_positive_ranking_policy.json'])
    seen.update(['reports/documentation/20261016/h_consistency/baseline_sources.json','reports/documentation/20261016/positive_ranking_implementation/authorization.json','docs/SELLER_ALIAS_POSITIVE_RANKING.zh.md','scripts/run_step28_positive_ranking_linux_20261018.sh'])
    check('independent_source_closure',seen==source_names,{'count':len(seen),'missing':sorted(seen-source_names),'extra':sorted(source_names-seen)})
    for kg in ['native_gate.json','formal_gate.json']:
        g=j(q/kg)
        for key in ['implementation_authorization','execution_authorization','external_review','primary_disposition','cpu']:
            file_rec(W,g[key],kg+':'+key)
        check(kg+':approval_status',g['status']=='APPROVED_NO_OPEN_BLOCKERS')
    permission=j(q/'execution_authorization.json')
    check('specific_execution_authorization',permission['status']=='APPROVED' and permission['stages']==['NATIVE','FORMAL'] and bool(permission['user_quote']))
    file_rec(W,gate['gpu'],'formal_gate_native_packet')
    SUMMARY['sources']={'count':40,'bindings_checked':len(bindings)+1,'independent_AST_closure_count':len(seen)}

    inv=j(W/'reports/result_sync/inventory.json')
    for r in inv['files']: file_rec(W,r,'formal_returned:'+r['path'])
    check('255_returned_originals',len(inv['files'])==inv['count']==255 and sum(r['bytes'] for r in inv['files'])==inv['bytes']==8685752)
    job_actual={str(p.relative_to(W)) for p in (W/'reports/job').rglob('*') if p.is_file()}
    check('all_job_files_in_inventory',job_actual=={r['path'] for r in inv['files'] if r['path'].startswith('reports/job/')})
    for r in inv['excluded_payload_metadata_only']:
        check('excluded_payload_absent:'+r['path'],not (W/r['path']).exists())
    analysis_inv=j(W/'reports/result_sync/analysis_inventory.json')
    for r in analysis_inv['files']: file_rec(W,r,'analysis_returned:'+r['path'])
    base=j(W/'reports/documentation/20261016/h_consistency/baseline_sources.json')
    for r in base['files']:file_rec(W,r,'baseline:'+r['path'])
    check('52_baseline_files',len(base['files'])==base['file_count']==52 and sum(r['bytes'] for r in base['files'])==base['total_bytes']==2940294)
    SUMMARY['files']={'returned_originals':len(inv['files']),'returned_original_bytes':inv['bytes'],'job_subtree_files':len(job_actual),'excluded_payload_metadata':len(inv['excluded_payload_metadata_only']),'baselines':len(base['files']),'baseline_bytes':base['total_bytes']}

    native=j(W/'reports/qualification/native/result.json')
    native_wall=j(W/'reports/launch/native_wallclock.json')
    native_start=j(W/'reports/launch/native_start.json')
    accept=j(W/'reports/launch/native_acceptance.json')
    check('native_outer_exit_budget',native_wall['exit_code']==0 and math.ceil(native_wall['elapsed_seconds'])==native_wall['charged_seconds']==650 and native_wall['charged_seconds']<=1200)
    check('native_total_actual_updates',native['actual_updates']==12 and len(native['arms'])==2)
    native_summary={}
    for arm in ['S_pos','H_pos']:
        a=native['arms'][arm]
        check(arm+':native_six_calls',a['actual_updates']==len(a['updates'])==6 and [(v['phase'],v['step']) for v in a['updates']]==[('allocation',1),('A',2),('A',2),('B',49),('B',49),('B',49)])
        check(arm+':native_budget',a['charged_seconds']<=600)
        check(arm+':native_independent_initial',a['initial_state']==base['baselines']['S']['initial_identity'])
        check(arm+':native_effect_and_restore',all(a[k] for k in ['both_phases_encoder_head_effect','next_update_bitwise_equal','pooling_restores_verified','rank_precision_verified']))
        effect_summary={}
        for phase in ['current','paired']:
            effect_summary[phase]={}
            for module in ['encoder','head']:
                v=a['effects'][phase][module];c=compact_effect(v);effect_summary[phase][module]=c
                check(arm+':'+phase+':'+module+':real_effect',v['finite'] and v['adam']['finite'] and all(c[k]>0 for k in ['pre_clip_gradient_max_delta','post_clip_gradient_max_delta','parameter_max_delta','adam_max_exp_avg_delta','adam_max_exp_avg_sq_delta']))
                check(arm+':'+phase+':'+module+':adam_clock',c['adam_steps_on_off']==[(2.0,2.0)] if phase=='current' else c['adam_steps_on_off']==[(337.0,337.0)])
                if phase=='paired':check(arm+':'+module+':isolated_history_effect',all(x>0 for x in c['isolated_branch_preclip_deltas'].values()))
        m=a['measurements'];train=288*(m['t_A']+m['t_B']);forward=1446*m['v']+36*m['h'];io=m['L']+2*(m['s_F']+m['r_F']+m['s_I']+m['r_I']);seconds=1.25*(train+forward+io+1800)
        check(arm+':projection_formula',seconds==a['projection']['projected_total_seconds'])
        native_summary[arm]={'actual_calls':len(a['updates']),'charged_seconds':a['charged_seconds'],'effects':effect_summary,'projection_seconds':seconds,'restore_equality_counts':a['equality_counts'],'resources':a['resources']}
    hres=native['arms']['H_pos']['resources']
    check('native_second_arm_zero_memory_own_pid',hres['live_allocated_bytes']==hres['reserved_bytes']==0 and hres['allowed_context_pid']==native_start['pid'])
    seconds=sum(x['projection_seconds'] for x in native_summary.values())
    ms=[a['measurements'] for a in native['arms'].values()]
    peak=math.ceil(1.1*(max(m['W_F'] for m in ms)+sum(2*m['W_I'] for m in ms)+2*2**30))
    check('dual_projection',accept['projection']=={'projected_total_seconds':seconds,'projected_peak_output_bytes':peak,'within_budget':seconds<=57600 and peak<=16*2**30})
    SUMMARY['native']={'outer_charged_seconds':native_wall['charged_seconds'],'inner_charged_seconds':native['cumulative_charged_seconds'],'projected_seconds':seconds,'projected_peak_bytes':peak,'arms':native_summary}

    job=W/'reports/job';run=job/'run';manifest=j(run/'manifest.json');split=j(run/'partition.json');before=j(job/'before_valid.json');done=j(job/'completion.json');access=j(job/'access.json')
    file_rec(job,before['manifest'],'before_valid_manifest')
    check('pre_valid_access',before['status']=='ALL_BLIND_GATES_PASS' and before['access']=={'train':1,'valid':0,'heldout':0,'owners':0})
    check('final_access',done['access']==access=={'train':1,'valid':1,'heldout':0,'owners':0})
    check('manifest_complete',manifest['status']=='COMPLETE_1152_UPDATES_4_RESTORES_24_DIAGNOSTICS_VALID_BLIND')
    check('partition_groups',Counter(r['domain'] for r in split['fit'])=={'A':48,'B':48} and Counter(r['domain'] for r in split['calibration'])=={'A':12,'B':12} and Counter(r['domain'] for r in split['development'])=={'A':20,'B':20})
    for kind,b in base['baselines'].items():
        check('baseline_partition:'+kind,j(W/b['root']/'run/partition.json')==split)
    current_ids_by_stage={};history_ids_by_stage={};updates=groups=captures=0;update_summary={};diagnostics=[]
    for arm in ['S_pos','H_pos']:
        check(arm+':formal_initial_identity',manifest['initials'][arm]==base['baselines']['S']['initial_identity'])
        check(arm+':A_continuation',manifest['starts'][arm]==manifest['points']['A_'+arm]['continuity'])
        for stage_idx,stage in enumerate('AB',1):
            name=stage+'_'+arm;p=manifest['points'][name];log=j(run/'updates'/f'{name}.json')
            check(name+':point_json_manifest_same',j(run/'points'/f'{name}.json')==p)
            file_rec(run,manifest['training'][name],name+':update_source')
            check(name+':288_rows',log['updates']==len(log['rows'])==len(log['current_ids'])==288)
            check(name+':adam_clock',log['adam_step']==p['adam_step']==stage_idx*288)
            check(name+':ordered_steps',[r['step'] for r in log['rows']]==list(range(1,289)))
            expected_fit=[r['group_uid'] for r in split['fit'] if r['domain']==stage]
            check(name+':every_fit_six',Counter(log['current_ids'])==Counter({u:6 for u in expected_fit}))
            check(name+':row_uids',[r['current_uid'] for r in log['rows']]==log['current_ids'])
            hist=[r['history_uid'] for r in log['rows'] if r['history_uid'] is not None]
            check(name+':history_count',hist==log['history_ids'] and len(hist)==(288 if stage=='B' else 0))
            check(name+':both_restore_flags',p['full_restore_verified'] is True and p['inference_restore_verified'] is True)
            for role in ['raw','calibration']:
                original=npy_meta(run/p['scores'][role]['path'])
                for prefix in ['full','inference']:
                    check(name+':'+prefix+'_'+role+':saved_restore_payload',npy_meta(run/p['scores'][prefix+'-'+role]['path'])==original)
            check(name+':score_shape',npy_meta(run/p['scores']['raw']['path'])['shape']==(40,378))
            check(name+':calibration_shape',npy_meta(run/p['scores']['calibration']['path'])['shape']==(12,378))
            members=p['memory_summary']['members']
            check(name+':memory_budget_members',p['memory_summary']['bytes']<=2**20 and len(members)==6 and set(members)<=set(r['group_uid'] for r in split['fit'] if r['domain']=='A'))
            if stage=='B':
                check(name+':own_prior_memory',set(hist)<=set(manifest['points']['A_'+arm]['memory_summary']['members']) and p['memory_summary']['tables']==manifest['points']['A_'+arm]['memory_summary']['tables'])
            logs=[]
            for rec in manifest['diagnostics'][name]:
                file_rec(run/'diagnostics',rec,name+':capture')
                d=j(run/'diagnostics'/rec['path']);logs.append(d)
                for key in ['fit','cache','valid_blind']:file_rec(run/'diagnostics',d[key],name+':'+key)
                check(name+':diag_clock:'+str(d['epoch']),d['adam_step']==(stage_idx-1)*288+48*d['epoch'] and d['stage_step']==48*d['epoch'])
                check(name+':diag_scope:'+str(d['epoch']),d['valid_labels_used'] is False and d['fit_ids']==expected_fit and d['valid_ids']==[r['group_uid'] for r in split['development']] and d['cache_ids']==([] if stage=='A' else members))
                check(name+':diag_shapes:'+str(d['epoch']),npy_meta(run/'diagnostics'/d['fit']['path'])['shape']==(48,3) and npy_meta(run/'diagnostics'/d['cache']['path'])['shape']==(0 if stage=='A' else 6,3) and npy_meta(run/'diagnostics'/d['valid_blind']['path'])['shape']==(40,378))
                if d['epoch']==6:check(name+':diag6_endpoint_payload',npy_meta(run/'diagnostics'/d['valid_blind']['path'])==npy_meta(run/p['scores']['raw']['path']))
                diagnostics.append({'point':name,'epoch':d['epoch'],'stage_step':d['stage_step'],'adam_step':d['adam_step'],'fit_groups':len(d['fit_ids']),'cache_groups':len(d['cache_ids']),'valid_groups':len(d['valid_ids']),'valid_labels_used':d['valid_labels_used']})
            check(name+':all_six_epochs',[x['epoch'] for x in logs]==list(range(1,7)))
            updates+=len(log['rows']);groups+=len(log['rows'])+len(hist);captures+=len(logs)
            seq=(log['current_ids'],log['history_ids'],[(r['current']['assignment_sha256'],r.get('history',{}).get('assignment_sha256')) for r in log['rows']])
            if stage in current_ids_by_stage:check(name+':cross_arm_schedule_assignment',seq==current_ids_by_stage[stage])
            else:current_ids_by_stage[stage]=seq
            for kind,b in base['baselines'].items():
                old=j(W/b['root']/'run/updates'/f'{stage}_{b["arm"]}.json')
                oldseq=(old['current_ids'],old['history_ids'],[(r['current']['assignment_sha256'],r.get('history',{}).get('assignment_sha256')) for r in old['rows']])
                check(name+':baseline_schedule_assignment:'+kind,seq==oldseq)
            update_summary[name]={'row_count':len(log['rows']),'history_count':len(hist),'adam_step':log['adam_step'],'diagnostic_captures':len(logs),'memory_bytes':p['memory_summary']['bytes']}
    check('1152_1728_4_24',updates==manifest['physical_updates']==done['physical_updates']==1152 and groups==manifest['gradient_group_presentations']==done['gradient_group_presentations']==1728 and len(manifest['points'])==4 and captures==24)
    over=j(job/'diagnostics_evaluation/overfitting.json');decisions=[]
    for name,branches in over['points'].items():
        for branch,domains in branches.items():
            if domains.get('status')=='NOT_APPLICABLE_FIRST_STAGE':continue
            for domain,v in domains.items():
                for metric,text in v['interpretation'].items(): decisions.append({'point':name,'branch':branch,'domain':domain,'metric':metric,'interpretation':text})
    check('18_strict_interpretations',len(decisions)==18 and Counter(r['interpretation'] for r in decisions)=={'未见明确迹象':17,'证据不足':1})
    for key,rec in over['group_metrics'].items():file_rec(job/'diagnostics_evaluation',rec,'diagnostic_evaluation:'+key)
    check('24_diag_eval_matrices',len(over['group_metrics'])==24)
    SUMMARY['formal_counts']={'updates':updates,'gradient_group_presentations':groups,'endpoints':len(manifest['points']),'diagnostic_captures':captures,'strict_interpretations':len(decisions),'points':update_summary,'diagnostics':diagnostics,'decisions':decisions,'pre_valid_access':before['access'],'final_access':access}
    wrapper=read(W/'reports/job.wrapper.txt','human_full_text').decode().splitlines()
    seconds=(datetime.fromisoformat(wrapper[2])-datetime.fromisoformat(wrapper[0])).total_seconds()
    check('wrapper_exit_and_clock',wrapper[1]=='exit_code=0' and seconds==12661)
    aw=j(W/'reports/result_sync/analysis_wallclock.json')
    check('formal_plus_analysis_clock',math.ceil(aw['elapsed_seconds'])==aw['charged_seconds']==3 and aw['formal_training_charged_seconds']==seconds and aw['formal_cumulative_seconds']==seconds+aw['charged_seconds']==12664 and 12664<=aw['formal_maximum_seconds']==57600)
    check('analysis_script_identity',identity(W/'scripts/step28_positive_ranking_audit.py')['sha256']==aw['script_sha256'])
    budget=done['budget']
    check('reported_resources_within_budget',budget['elapsed_seconds']<seconds and budget['peak_cuda_reserved_bytes']<=budget['limits']['maximum_cuda_reserved_bytes'] and budget['peak_rss_bytes']<=budget['limits']['maximum_rss_bytes'] and budget['peak_output_bytes']<=budget['limits']['maximum_output_bytes'])
    console=read(W/'reports/job.console.txt','console_saved_log_parse').decode().splitlines();events=[json.loads(s) for s in console if s.startswith('{')]
    check('console_four_stage_records',{x['point'] for x in events}==set(manifest['points']))
    for name in manifest['points']:
        counts=[x['completed'] for x in events if x['point']==name]
        check(name+':console_checkpoints',counts==[1]+list(range(24,289,24)))
    check('console_monotone_wallclock',all(a['elapsed_seconds']<b['elapsed_seconds'] for a,b in zip(events,events[1:])))
    close=j(W/'reports/result_sync/closeout.json')
    check('closeout_completion_identity',close['completion_sha256']==identity(job/'completion.json')['sha256'])
    check('closeout_four_model_four_memory_metadata',len(close['models'])==4 and len(close['memory'])==4 and close['failure_exists'] is False and close['run_pids_absent'] is True)
    SUMMARY['formal_resources']={'wrapper_start':wrapper[0],'wrapper_end':wrapper[2],'wrapper_exit':0,'wrapper_seconds':seconds,'analysis_seconds':aw['elapsed_seconds'],'charged_with_analysis':aw['formal_cumulative_seconds'],'budget':budget,'console_update_checkpoints':len(events),'closeout_is_saved_observation_not_live_verification':True}
    SUMMARY['scope_limits']=['No project-server connection, network, training, model loading, formal text/label parsing or Memory loading.', 'Bytes and metadata of excluded payloads are preserved source assertions; actual payload content could not be reverified in this package.', 'Source identities, saved score restore equality, update/capture records and fixed result accounting are replayed locally; label-derived metric truth is not recomputed.', 'Native tensor-effect entries are preserved project-native evidence; this local execution did not generate the underlying tensor effects.']

if __name__=='__main__':
    exit_code=0
    try: main()
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code=1
    SUMMARY['check_count']=len(CHECKS)
    SUMMARY['failed_checks']=[c for c in CHECKS if not c['pass']]
    if SUMMARY['failed_checks']:exit_code=2
    SUMMARY['local_resources']={'elapsed_seconds':time.perf_counter()-START,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'python':sys.version,'exit_code':exit_code,'recorded_at':datetime.now(timezone.utc).isoformat()}
    (OUT/'checks.json').write_text(json.dumps(CHECKS,ensure_ascii=False,indent=2)+'\n')
    (OUT/'mechanical_summary.json').write_text(json.dumps(SUMMARY,ensure_ascii=False,indent=2)+'\n')
    (OUT/'read_scope.json').write_text(json.dumps({'scope_note':'Human full-text readings are explicitly distinguished from source-identity/AST/structured mechanical reads. Entire archives or source files hashed are not claimed as full semantic reading.','files':list(READS.values())},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'check_count':len(CHECKS),'failed_checks':SUMMARY['failed_checks'],'resources':SUMMARY['local_resources'],'output':str(OUT)},ensure_ascii=False))
    sys.exit(exit_code)
