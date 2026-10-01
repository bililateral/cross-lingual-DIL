"""Independent label-free audit of recorded source, updates, reservoir and native evidence.
Does not import project modules or open full memory payloads, project CSVs, texts or weights.
Group UIDs are opaque scheduling tokens only; no controller identity is reconstructed.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv, hashlib, json, random, time
import numpy as np

START = time.perf_counter()
R = Path('/mnt/data/bge_input')
J = R/'reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job'
U = J/'run'
A = R/'reports/seller_alias_continual/20261001/bge_continual_result'
C = R/'reports/seller_alias_continual/20260930/bge_continual_cpu/20260930_142608/job'
OUT = Path('/mnt/data/bge_review_evidence/outputs')
checks = Counter()

def load(path): return json.loads(path.read_text(encoding='utf-8'))
def canonical(obj): return (json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(b): return hashlib.sha256(b).hexdigest()
def equal(a,b,name):
    assert a == b, (name,a,b)
    checks[name] += 1

def verify(base, rec):
    p=base/rec['path']; b=p.read_bytes()
    equal(len(b),rec['bytes'],'referenced_size')
    equal(digest(b),rec['sha256'],'referenced_sha256')
    return p

def seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical(list(parts))).digest()[:8],'big')%(2**63-1)
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

m=load(U/'manifest.json'); exe=load(J/'execution.json'); before=load(J/'before_valid.json'); complete=load(J/'completion.json'); p=load(R/'schema/step28_bge_continual_policy.json')
source=m['source_files']; equal(len(source),18,'source_count')
for rec in source: verify(R,rec)
for obj in [exe,load(U/'startup.json'),load(verify(R,exe['authorization'])),load(verify(R,exe['audit']))]:
    equal(obj['source_files'],source,'source_chain_identical')
equal(m['policy_sha256'],digest((R/'schema/step28_bge_continual_policy.json').read_bytes()),'policy_digest')
verify(J,before['manifest']); verify(J,complete['evaluation']); verify(U,m['partition'])
equal(before['supervision'],dict(train=1,valid=0,heldout=0,owners=0),'before_valid_counts')
equal(complete['label_parses'],dict(train=1,valid=1,heldout=0,owners=0),'completion_counts')
equal(load(J/'access.json'),complete['label_parses'],'access_complete_equal')
equal(before['points'],21,'before_valid_points')
equal((J/'exit_status.txt').read_text().strip(),'0','original_exit')
ret=load(A/'return_inventory.json')
for rec in ret['files']: verify(R,rec)
equal(len(ret['files']),ret['file_count'],'returned_count')
equal(sum(x['bytes'] for x in ret['files']),ret['total_bytes'],'returned_bytes')
part=load(U/'partition.json')
uid_domain={row['group_uid']:row['domain'] for role in part.values() for row in role}
equal(len(uid_domain),240,'distinct_partition_groups')
fit={d:sorted(x['group_uid'] for x in part['fit'] if x['domain']==d) for d in 'ABC'}
for role,n in [('fit',48),('calibration',12),('development',20)]:
    equal(Counter(row['domain'] for row in part[role]),Counter({d:n for d in 'ABC'}),'partition_counts')
# Reconstruct public-ID hash partition from the original 60 training IDs/domain, no labels.
basep=load(R/'schema/step28_chinese_base_policy.json')
for d in 'ABC':
    ids=[x['group_uid'] for role in ('fit','calibration') for x in part[role] if x['domain']==d]
    ordered=sorted(ids,key=lambda uid:digest(canonical([basep['partition']['seed'],d,uid])))
    equal(set(ordered[:48]),set(fit[d]),'independent_partition_hash')

updates={}; points={}; histories={}; memory_rows=[]; update_rows=[]; cal_rows=[]
loss_error=0.; stored_memory_bytes=[]; aux_bytes=[]
columns=['current_bce','current_rank','current_hard','current_total','history_bce','history_rank','history_hard','history_total','logit_mse','total','encoder_lr','head_lr','gradient_norm','logit_term_host_seconds']
steps=np.arange(1,289)
expected_lr=np.array([1e-5*(int(t)/29 if t<=29 else (288-int(t))/259) for t in steps])
for name,rec in sorted(m['training'].items()):
    x=load(verify(U,rec)); z=np.load(verify(U,x['update_file']),allow_pickle=False)
    equal(z.shape,(288,14),'update_array_shape'); equal(z.dtype,np.dtype('float64'),'update_dtype'); assert np.isfinite(z).all()
    equal(x['update_columns'],columns,'update_columns'); equal(x['updates'],288,'updates_288'); equal(x['adam_step'],288*x['stage'],'continuous_adam_record')
    equal(x['actual_domain'],x['order'][x['stage']-1],'current_actual_domain')
    equal(x['seed'],'s0','training_seed')
    equal(x['encoder_positive_lr_updates'],287,'encoder_positive_lr_updates')
    equal(np.array_equal(z[:,10],expected_lr),True,'all_stage_learning_rates_exact')
    equal(np.array_equal(z[:,11],np.full(288,.001)),True,'head_lr_exact')
    stream=seed(p['schedule_seed'],x['order'],x['stage'],'current')
    equal(stream,x['current_dropout_stream'],'current_stream')
    rng=random.Random(stream); current=[]
    for epoch in range(6):
        rows=list(fit[x['actual_domain']]);rng.shuffle(rows);current.extend(rows)
    equal(current,x['current_ids'],'current_sequence_independent')
    equal(digest(canonical(current)),x['current_schedule_sha256'],'current_schedule_digest')
    equal(Counter(current),Counter({uid:6 for uid in fit[x['actual_domain']]}),'current_six_presentations')
    equal(set(x['observations']),{'1','29','30','288'},'observation_steps')
    for s,obs in x['observations'].items():
        for module in ('encoder','head'):
            equal(obs[module]['finite_nonzero_combined_gradient'],True,'recorded_nonzero_combined_gradient')
            equal(obs[module]['parameters_changed'],module=='head' or s!='288','recorded_parameter_change_scope')
    for offset in [0,4]:
        e=float(np.max(np.abs(z[:,offset+3]-(z[:,offset]+z[:,offset+1]+.5*z[:,offset+2]))))
        loss_error=max(loss_error,e)
        assert e<=1e-6,(name,'float32_component_sum',e)
    equal(np.array_equal(z[:,9],z[:,3]+z[:,7]+.5*z[:,8]),True,'combined_recorded_objective_exact')
    if x['method'] in ('seq','shared'):
        equal(x['history_ids'],[],'no_seq_history');equal(x['memory_after_training'],None,'no_seq_memory')
        equal(bool(np.all(z[:,4:9]==0)),True,'no_seq_history_loss')
    else:
        equal(len(x['history_ids']),288,'history_draw_count')
        summary=x['memory_after_training'];stored_memory_bytes.append(summary['serialized_bytes'])
        assert summary['serialized_bytes']<=1048576
        equal(summary['members'],list(dict.fromkeys(summary['members'])),'six_unique_memory_slots');equal(len(summary['members']),6,'six_memory_groups')
        equal(summary['draw_count'],288,'recorded_history_count')
    if x['method']!='logit':equal(bool(np.all(z[:,8]==0)),True,'zero_nonlogit_mse')
    assert np.all(z[:,12]>0)
    point=load(verify(U,m['points'][name]));points[name]=point
    equal(point['completed_updates'],x['adam_step'],'point_adam_step')
    equal(point['full_model_adam_and_rng_restore_verified'],True,'recorded_full_restore_verified')
    equal(point['full_checkpoint_retained'],x['stage']==1,'shared_full_checkpoint_retention')
    auxiliary=point['learner_auxiliary'];size=len(canonical(auxiliary))
    equal(size,point['learner_auxiliary_serialized_bytes'],'auxiliary_actual_byte_remeasure')
    aux_bytes.append(size)
    mapping=load(verify(U,point['mapping']))
    equal(mapping['role'],'calibration','calibration_only_role')
    equal(mapping['options'],p['calibration']['options'],'calibration_fixed_options')
    equal(mapping['status'],'PASS_CALIBRATION_FIT','calibration_recorded_status')
    equal(mapping['optimizer_success'],True,'calibration_success')
    assert mapping['a']>0 and mapping['projected_gradient_max']<=1e-6
    assert mapping['final_nll']<=mapping['initial_nll']+1e-12
    equal(len(mapping['trajectory']),mapping['optimizer_iterations'],'calibration_trajectory_count')
    last=mapping['trajectory'][-1]
    equal([mapping['a'],mapping['b']],[last['a'],last['b']],'map_matches_last_iteration')
    cal_rows.append(dict(point=name,a=mapping['a'],b=mapping['b'],iterations=mapping['optimizer_iterations'],calls=mapping['objective_calls'],projected_gradient_max=mapping['projected_gradient_max'],initial_nll=mapping['initial_nll'],final_nll=mapping['final_nll'],optimizer_message=mapping['optimizer_message']))
    updates[name]=(x,z)
    update_rows.append(dict(point=name,method=x['method'],stage=x['stage'],current_total_mean=float(z[:,3].mean()),history_total_mean=float(z[:,7].mean()),current_last48_mean=float(z[-48:,3].mean()),history_last48_mean=float(z[-48:,7].mean()),mse_mean=float(z[:,8].mean()),half_mse_mean=float(.5*z[:,8].mean()),clip_fraction=float((z[:,12]>basep['optimizer']['clip_norm']).mean()),gradient_norm_mean=float(z[:,12].mean()),training_seconds=x['training_seconds'],mse_host_enqueue_seconds=float(z[:,13].sum())))
equal(len(updates),21,'actual_training_endpoints')
equal(sum(x['updates'] for x,z in updates.values()),6048,'total_physical_updates')
equal(6048+sum(len(x['history_ids']) for x,z in updates.values()),9504,'gradient_group_presentations')
# Standard Algorithm R driven only by the frozen opaque IDs, with one insertion per current group.
for order in ('ABC','BCA','CAB'):
    rng=random.Random(seed(p['memory_seed'],order,'retention'));slots=[];seen=0
    previous_targets={}
    for completed_stage in (1,2):
        for uid in fit[order[completed_stage-1]]:
            seen+=1
            index=len(slots) if seen<=6 else rng.randrange(seen)
            if index<6:
                if len(slots)<6:slots.append(uid)
                else:slots[index]=uid
        suffix='after1' if completed_stage==1 else 'stage2'
        for arm in ('er','logit'):
            summary=m['memories'][order+'_'+arm+'_'+suffix]
            equal(summary['members'],slots,'algorithm_R_retained_slots_independent')
            equal(summary['seen'],seen,'algorithm_R_seen')
            equal(summary['serialized_bytes'],summary['file']['bytes'],'memory_recorded_bytes_binding')
            equal(summary['sha256'],summary['file']['sha256'],'memory_recorded_hash_binding')
            equal(summary['draw_count'],0,'retention_resets_draw_count')
            stored_memory_bytes.append(summary['serialized_bytes']);assert summary['serialized_bytes']<=1048576
            if arm=='er':equal(summary['references'],{},'ER_no_reference_logits')
            else:
                equal(set(summary['references']),set(slots),'LOGIT_reference_coverage')
                for uid in slots:
                    equal(summary['reference_origins'][uid],1 if uid_domain[uid]==order[0] else 2,'reference_origin_stage')
                    if uid in previous_targets:equal(summary['references'][uid],previous_targets[uid],'surviving_reference_not_refreshed')
                previous_targets=dict(summary['references'])
        stage=completed_stage+1
        dr=random.Random(seed(p['memory_seed'],order,stage,'history_draws'))
        sequence=[slots[dr.randrange(6)] for _ in range(288)]
        er=updates[f'{order}_er_stage{stage}'][0];logit=updates[f'{order}_logit_stage{stage}'][0];sq=updates[f'{order}_seq_stage{stage}'][0]
        equal(er['history_ids'],sequence,'independent_history_draw_sequence')
        equal(logit['history_ids'],sequence,'paired_history_draw_sequence')
        equal(er['current_ids'],logit['current_ids'],'ER_LOGIT_current_pairing');equal(er['current_ids'],sq['current_ids'],'ER_SEQ_current_pairing')
        counts=Counter(sequence)
        for arm in ('er','logit'):
            name=f'{order}_{arm}_stage{stage}'
            recorded=load(U/'memory'/f'{name}_budget.json')
            train=updates[name][0]['memory_after_training']
            for k in ('members','seen','draw_count','draw_stage','references','reference_origins','with_logits'):
                equal(recorded[k],train[k],'post_cal_memory_same_historical_state')
            equal(recorded['auxiliary_serialized_bytes'],points[name]['learner_auxiliary_serialized_bytes'],'memory_auxiliary_binding')
            equal(recorded['serialized_bytes']-train['serialized_bytes'],recorded['auxiliary_serialized_bytes']-train['auxiliary_serialized_bytes'],'memory_size_change_is_auxiliary_change')
            stored_memory_bytes.append(recorded['serialized_bytes']);assert recorded['serialized_bytes']<=1048576
        memory_rows.append(dict(order=order,training_stage=stage,old_domains=order[:completed_stage],domain_counts=dict(Counter(uid_domain[u] for u in slots)),draws_by_domain=dict(Counter(uid_domain[u] for u in sequence)),minimum_draws_per_group=min(counts.values()),maximum_draws_per_group=max(counts.values()),group_draws=dict(counts),retained_members=list(slots)))
# Original log chronology, not a recreated training log.
rawlog=(J/'train.log').read_text();events=[];nonjson=[]
for line in rawlog.splitlines():
    if line.startswith('{'):events.append(json.loads(line))
    elif line.strip():nonjson.append(line)
ue=[x for x in events if x.get('event')=='updates'];equal(len(ue),252,'original_update_log_events')
assert all(b['elapsed_seconds']>=a['elapsed_seconds'] for a,b in zip(ue,ue[1:]))
for order in ('ABC','BCA','CAB'):
    for stage,arm in [(1,'seq')]+[(s,a) for a in ('seq','er','logit') for s in (2,3)]:
        es=[x for x in ue if (x['order'],x['stage'],x['method'])==(order,stage,arm)]
        equal([x['stage_updates'] for x in es],list(range(24,289,24)),'recorded_24_step_checkpoints')
        equal([x['logical_updates'] for x in es],[(stage-1)*288+i for i in range(24,289,24)],'recorded_log_Adam_continuity')
# Read actual saved native CPU evidence, do not rerun real BGE.
native=load(C/'audit.json');equal(native['native_updates_actually_executed'],6,'native_actual_updates')
equal(native['contracts'],dict(passed=22,failed=0,skipped=0),'native_handmade_contracts')
equal(native['source_files'],source,'native_sources_bound')
ns={arm:load(verify(C,rec)) for arm,rec in native['native'].items()}
for arm,x in ns.items():
    equal(x['native_updates_actually_executed'],2,'native_per_arm_updates');equal(len(x['parameter_probes']),29,'native_probe_count')
    equal(x['stage2_update']['step'],1,'native_second_stage_local_step')
    for q in x['parameter_probes']:
        assert q['gradient_norm_after_clip']>0;equal(q['parameters_changed'],True,'native_probe_changed');equal(q['adam_step'],289.,'native_counter_fixture_289')
    for field in ('initial_model_state_sha256','after_warm_model_state_sha256','before_stage2_optimizer_sha256','captured_current_logits','origin_eval_reference'):
        equal(x[field],ns['seq'][field],'native_shared_state_pairing')
equal(ns['er']['captured_history_logits'],ns['logit']['captured_history_logits'],'native_history_forward_pairing')
result=dict(status='PASS_INDEPENDENT_RECORDED_TRAINING_EVIDENCE',checks=dict(checks),source_files=18,returned_files=ret['file_count'],returned_bytes=ret['total_bytes'],updates=6048,historical_gradient_presentations=3456,total_gradient_presentations=9504,maximum_float32_loss_component_sum_error=loss_error,recorded_memory_min_bytes=min(stored_memory_bytes),recorded_memory_max_bytes=max(stored_memory_bytes),actual_auxiliary_min_bytes=min(aux_bytes),actual_auxiliary_max_bytes=max(aux_bytes),memory_snapshots_examined=len(stored_memory_bytes),original_log_events=len(ue),original_non_json_log_lines=nonjson,original_start=(J/'started.txt').read_text().strip(),original_finish=(J/'finished.txt').read_text().strip(),original_completion_budget=complete['budget'],original_native_environment=native['environment'],native_actual_updates=6,native_counter_fixture_not_288_actual_updates=True,formal_labels_read=0,project_modules_imported=0,elapsed_seconds=time.perf_counter()-START,unverified=['Full formal memory payload bytes and reference numerical values excluded; byte summaries and surviving reference hashes verified, auxiliary metadata bytes actually remeasured.','Formal checkpoint tensors excluded; source control flow and saved actual restore attestations inspected, not independently reloaded.','Calibration objective/gradient/labels and scalar gradients are not recomputed from formal labels.','Recorded boundary counts are normal-program execution evidence, not an operating-system trace of every file read.'])
save('independent_training_evidence.json',result);save('independent_memory_diagnostics.json',memory_rows);save('independent_update_diagnostics.json',update_rows);save('independent_calibration_records.json',cal_rows)
for fn,rows in [('independent_update_diagnostics.csv',update_rows),('independent_calibration_records.csv',cal_rows)]:
    with (OUT/fn).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(result,ensure_ascii=False,indent=2));print('MEMORY_DOMAIN_DRAW_DIAGNOSTICS');print(json.dumps([{k:v for k,v in x.items() if k not in ('group_draws','retained_members')} for x in memory_rows],ensure_ascii=False,indent=2))
