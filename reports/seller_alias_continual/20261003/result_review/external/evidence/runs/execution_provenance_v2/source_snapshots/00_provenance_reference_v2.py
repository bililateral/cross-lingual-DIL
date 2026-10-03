#!/usr/bin/env python3
"""Independent audit of permitted provenance JSON/logs/NPY only.
No project imports, no model loading, no labels/cache bodies, no network.
Identity checks authenticate supplied bytes, not remote execution history.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import numpy as np


def digest(b):
    return hashlib.sha256(b).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    a = parser.parse_args()
    p = a.project.resolve(); out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    opened = {}; checks = []; detail = {}
    def note(path, mode):
        path = path.resolve()
        # This review only reads submitted files; all paths are below project root.
        if not path.is_relative_to(p):
            raise ValueError(f'Outside submitted project: {path}')
        if path.suffix in {'.pt','.bin','.safetensors'}:
            raise ValueError('Forbidden native weight load')
        b = path.read_bytes()
        key = path.relative_to(p).as_posix()
        opened.setdefault(key, {'path':key,'bytes':len(b),'sha256':digest(b),'modes':[]})['modes'].append(mode)
        return b
    def jread(path):
        return json.loads(note(path, 'full_json_parse'))
    def tread(path):
        return note(path, 'full_text_read').decode('utf-8')
    def nread(path):
        # Only saved small update logs in this program; no label-bearing NPY.
        if path.parent.name != 'updates':
            raise ValueError('Only update NPY permitted in provenance program')
        import io
        return np.load(io.BytesIO(note(path,'all_update_array_values')), allow_pickle=False)
    def ck(name, value, **extra):
        checks.append({'name':name,'passed':bool(value), **extra})
    def verify(path, r, name):
        if not path.is_file():
            ck(name,False,missing=str(path.relative_to(p)))
            return
        b = note(path,'file_identity')
        ck(name,len(b)==r['bytes'] and digest(b)==r['sha256'], path=path.relative_to(p).as_posix(),actual_bytes=len(b),actual_sha256=digest(b))
    def srcmap(rows):
        ck('unique_source_paths',len(rows)==len({r['path'] for r in rows}))
        return {r['path']:r for r in rows}
    def chrono(start,end):
        return (datetime.fromisoformat(end.strip())-datetime.fromisoformat(start.strip())).total_seconds()
    def resource(text):
        d={}
        for line in text.splitlines():
            if ': ' in line:
                key,v=line.strip().rsplit(': ',1)
                d[key]=v
        return d
    base = p/'reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job'
    weight = p/'reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job'
    bman = jread(base/'run/manifest.json'); wman=jread(weight/'run/manifest.json')
    bexec = jread(base/'execution.json'); wexec=jread(weight/'execution.json')
    bpoints={o:jread(base/f'run/points/{o}_shared.json') for o in ['ABC','BCA','CAB']}
    bpartition=jread(base/'run/partition.json')
    configs=[
        {'study':'low','result':'er_low','arm':'tenth','run':p/'reports/seller_alias_continual/20261002/er_low_execution/20261002_124018','root':p,'weight':0.1,'sources':31,'sets':99,'returned':270,'policy':'schema/step28_er_low_policy.json'},
        {'study':'logit','result':'logit_weight','arm':'logit_quarter','run':p/'reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200','root':p/'reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace','weight':0.25,'sources':34,'sets':63,'returned':198,'policy':'schema/step28_logit_weight_policy.json'}]
    allsource={}
    for cfg in configs:
        study=cfg['study']; root=cfg['root']; run=cfg['run']; job=(run/'job') if study=='low' else root/'reports/job'
        result=p/f"reports/seller_alias_continual/20261003/{cfg['result']}_result"
        cpu=run/'cpu'; arm=cfg['arm']; man=jread(job/'run/manifest.json'); exe=jread(job/'execution.json')
        start=jread(job/'run/startup.json'); progress=jread(job/'run/progress.json'); cad=jread(cpu/'audit.json')
        cpolicy=jread(root/cfg['policy'])
        source=srcmap(exe['source_files']); allsource[study]=source
        ck(f'{study}_source_count',len(source)==cfg['sources'],count=len(source))
        for r in source.values(): verify(root/r['path'],r,f'{study}_source_identity')
        for stage,record in [('manifest',man),('startup',start),('progress',progress),('native_cpu',cad)]:
            ck(f'{study}_{stage}_sources_match_execution',srcmap(record['source_files'])==source)
        ck(f'{study}_policy_sha',digest(note(root/cfg['policy'],'policy_digest'))==man['policy_sha256'])
        for r in bman['source_files']: verify(root/r['path'],r,f'{study}_original_18_unchanged')
        ck(f'{study}_same_runtime_baseline',all(exe[k]==bexec[k]==wexec[k] for k in ['python','torch','cuda','gpu']))
        ck(f'{study}_one_cpu_recorded',len(exe['cpu_affinity'])==1)
        # authorization.json and separate freeze.json intentionally not recreated.
        missing_auth=(root/exe['authorization']['path']).is_file() is False
        check_cpu_path = cpu/'audit.json'  # LOGIT raw execution path is workspace/reports/cpu; returned as sibling cpu.
        verify(check_cpu_path,exe['audit'],f'{study}_execution_cpu_audit_bound')
        for rel,r in man['baseline_records'].items(): verify(base/rel,r,f'{study}_pinned_baseline')
        for rel,r in cpolicy['weight_reference']['records'].items() if isinstance(cpolicy.get('weight_reference',{}).get('records'),dict) else []:
            verify(weight/rel,r,f'{study}_pinned_previous_weight')
        ck(f'{study}_partition_matches_baseline',jread(job/'run/partition.json')==bpartition)
        ck(f'{study}_public_input_metadata_matches',man['public_inputs']==bman['public_inputs']==wman['public_inputs'])
        ck(f'{study}_pretrained_archive_metadata_matches',man['pretrained_archive']==bman['pretrained_archive']==cad['archive'])
        expected={f'{o}_{arm}_stage{s}' for o in ['ABC','BCA','CAB'] for s in [2,3]}
        ck(f'{study}_six_point_names',set(man['points'])==expected)
        ck(f'{study}_six_training_names',set(man['training'])==expected)
        ck(f'{study}_count_record',man['physical_updates']==1728 and man['gradient_group_presentations']==3456)
        restored=[]
        for o in ['ABC','BCA','CAB']:
            r=man['restored_starts'][f'{o}_{arm}']; shared=bpoints[o]
            ck(f'{study}_{o}_shared_full_record',r['full_checkpoint']==shared['full_checkpoint'])
            ck(f'{study}_{o}_shared_model_digest',r['model_state_sha256']==shared['model_state_sha256'])
            ck(f'{study}_{o}_shared_mapping',r['first_map']==shared['first_map_parameters'])
            ck(f'{study}_{o}_shared_Adam288',r['adam_step']==shared['completed_updates']==288)
            ck(f'{study}_{o}_first_scores_replay_record',r['first_scores_replayed_exactly'] is True)
            memname=f'{o}_{"er" if study=="low" else "logit"}_after1'
            mem=bman['memories'][memname]
            ck(f'{study}_{o}_first_memory_source',r['memory_source']==mem['file'])
            ck(f'{study}_{o}_first_memory_members',r['memory_summary']['members']==mem['members'])
            for oldarm in ['half','quarter']:
                prev=wman['restored_starts'][f'{o}_{oldarm}']
                for k in ['full_checkpoint','model_state_sha256','adam_step','first_map']:
                    ck(f'{study}_{o}_{oldarm}_same_start_{k}',r[k]==prev[k])
            restored.append({'order':o,'full_checkpoint':r['full_checkpoint'],'model_state_sha256':r['model_state_sha256'],'first_map':r['first_map'],'memory_source':r['memory_source'],'recorded_replayed_60_groups':r['first_scores_replayed_exactly']})
        update_details=[]; points={}; all_rows=[]
        for name in sorted(expected):
            for kind in ['points','training']:
                verify(job/'run'/man[kind][name]['path'],man[kind][name],f'{study}_{name}_{kind}_identity')
            point=jread(job/f'run/points/{name}.json'); points[name]=point
            u=jread(job/f'run/updates/{name}.json'); arr=nread(job/'run'/u['update_file']['path']);verify(job/'run'/u['update_file']['path'],u['update_file'],f'{study}_{name}_update_npy_identity')
            o=u['order']; s=u['stage']; old=jread(base/f'run/updates/{o}_er_stage{s}.json')
            ck(f'{study}_{name}_288_rows_columns',arr.shape==(288,14 if study=='low' else 16) and len(u['update_columns'])==arr.shape[1])
            ck(f'{study}_{name}_finite_update_values',np.all(np.isfinite(arr)))
            ck(f'{study}_{name}_paired_metadata',all(u[k]==old[k] for k in ['order','stage','actual_domain','current_ids','history_ids','current_dropout_stream','adam_step','updates']))
            ck(f'{study}_{name}_Adam_continues',u['adam_step']==288*s==point['completed_updates'])
            ck(f'{study}_{name}_six_epochs',len(Counter(u['current_ids']))==48 and set(Counter(u['current_ids']).values())=={6})
            hfreq=Counter(u['history_ids']);ck(f'{study}_{name}_six_history_groups',len(hfreq)==6 and sum(hfreq.values())==288)
            col={n:arr[:,i] for i,n in enumerate(u['update_columns'])}
            first_err=float(np.max(np.abs(col['current_total']-(col['current_bce']+col['current_rank']+.5*col['current_hard']))))
            hist_err=float(np.max(np.abs(col['history_total']-(col['history_bce']+col['history_rank']+.5*col['history_hard']))))
            wh_err=float(np.max(np.abs(col['weighted_history_total']-cfg['weight']*col['history_total'])))
            total_reference=col['current_total']+cfg['weight']*col['history_total']
            if study=='logit': total_reference=total_reference+.5*col['logit_mse']
            total_err=float(np.max(np.abs(col['total']-total_reference)))
            ck(f'{study}_{name}_whole_base_loss',first_err<1e-6 and hist_err<1e-6,current_max_error=first_err,history_max_error=hist_err)
            ck(f'{study}_{name}_weighted_history',wh_err==0 and np.all(col['history_weight']==cfg['weight']),max_error=wh_err)
            ck(f'{study}_{name}_combined_total',total_err==0,max_error=total_err)
            if study=='logit': ck(f'{study}_{name}_independent_MSE_coefficient',np.all(col['logit_weight']==.5) and np.all(col['logit_mse']>=0))
            lr=np.array([1e-5*k/29 if k<=29 else 1e-5*(288-k)/259 for k in range(1,289)])
            lr_error=float(np.max(np.abs(lr-col['encoder_lr'])))
            ck(f'{study}_{name}_LR',lr_error<1e-20 and np.all(col['head_lr']==.001) and col['encoder_lr'][-1]==0,maximum_error=lr_error)
            for local,obs in u['observations'].items():
                ck(f'{study}_{name}_observed_gradient_{local}',all(obs[z]['finite_nonzero_combined_gradient'] is True for z in ['encoder','head']))
                ck(f'{study}_{name}_observed_parameter_change_{local}',obs['head']['parameters_changed'] is True and obs['encoder']['parameters_changed']==(local!='288'))
            ck(f'{study}_{name}_checkpoint_restored_record',point['full_model_adam_and_rng_restore_verified'] is True)
            ck(f'{study}_{name}_checkpoint_policy',point['policy_sha256']==man['policy_sha256'] and point['history_weight']==cfg['weight'])
            ck(f'{study}_{name}_first_mapping',point['first_map_parameters']==bpoints[o]['first_map_parameters'])
            for which,r in point['scores'].items(): verify(job/'run'/r['path'],r,f'{study}_{name}_{which}_saved_score_identity')
            verify(job/'run'/point['mapping']['path'],point['mapping'],f'{study}_{name}_mapping_identity')
            # No read of native weights or body of memory. Only permitted budget summary.
            memory=jread(job/f'run/memory/{name}_budget.json')
            training_memory = u['memory_after_training']
            identity_fields = set(memory) - {'auxiliary_serialized_bytes','serialized_bytes','sha256'}
            ck(f'{study}_{name}_budget_summary_identity_and_auxiliary_delta',set(memory)==set(training_memory) and all(memory[k]==training_memory[k] for k in identity_fields) and memory['serialized_bytes']-training_memory['serialized_bytes']==memory['auxiliary_serialized_bytes']-training_memory['auxiliary_serialized_bytes'] and memory['auxiliary_serialized_bytes']==point['learner_auxiliary_serialized_bytes'])
            ck(f'{study}_{name}_memory_budget',memory['serialized_bytes']<1024**2 and memory['auxiliary_serialized_bytes']<memory['serialized_bytes'] and len(memory['members'])==6)
            auxiliary_bytes=len((json.dumps(point['learner_auxiliary'],ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode())
            ck(f'{study}_{name}_auxiliary_serialization',auxiliary_bytes==point['learner_auxiliary_serialized_bytes'],actual_bytes=auxiliary_bytes)
            update_details.append({'name':name,'rows':len(arr),'columns':u['update_columns'],'actual_domain':u['actual_domain'],'current_group_count':len(Counter(u['current_ids'])),'current_frequency_range':[min(Counter(u['current_ids']).values()),max(Counter(u['current_ids']).values())],'history_frequency_range':[min(hfreq.values()),max(hfreq.values())],'history_member_count':len(hfreq),'current_dropout_stream':u['current_dropout_stream'],'adam_step':u['adam_step'],'current_base_decomposition_error':first_err,'history_base_decomposition_error':hist_err,'weighted_history_error':wh_err,'combined_total_error':total_err,'lr_max_error':lr_error,'gradient_norm_min':float(col['gradient_norm'].min()),'gradient_norm_max':float(col['gradient_norm'].max()),'clipped_steps':int((col['gradient_norm']>1).sum()),'cuda_allocator':u['cuda_allocator'],'training_seconds':u['training_seconds'],'memory_serialized_bytes':memory['serialized_bytes'],'auxiliary_bytes_recomputed':auxiliary_bytes,'checkpoint_full_restore_recorded':point['full_model_adam_and_rng_restore_verified']})
            all_rows.append(arr)
        ck(f'{study}_rows_independently_summed',sum(len(ar) for ar in all_rows)==1728)
        ck(f'{study}_group_presentations_from_ID_positions',sum(len(jread(job/f'run/updates/{name}.json')[k]) for name in expected for k in ['current_ids','history_ids'])==3456)
        access=jread(job/'access.json'); before=jread(job/'before_valid.json'); completion=jread(job/'completion.json')
        ck(f'{study}_before_valid_counts',before['label_parses']=={'train':1,'valid':0,'heldout':0,'owners':0})
        ck(f'{study}_complete_access_counts',access==completion['label_parses']=={'train':1,'valid':1,'heldout':0,'owners':0})
        verify(job/'run/manifest.json',before['manifest'],f'{study}_gate_bound_complete_manifest')
        verify(job/'evaluation/evaluation.json',completion['evaluation'],f'{study}_completion_bound_evaluation')
        collected=jread(job/'evaluation/collected.json'); refs=jread(job/'evaluation/reference/collected.json')
        started=tread(job/'started.txt');finished=tread(job/'finished.txt'); raw_resource=tread(job/'resource_usage.log');res=resource(raw_resource)
        cstarted=tread(cpu/'started.txt');cfinished=tread(cpu/'finished.txt');cpures=resource(tread(cpu/'resource_usage.log'))
        ck(f'{study}_formal_exit0',tread(job/'exit_status.txt').strip()=='0' and res['Exit status']=='0')
        ck(f'{study}_formal_hard_time',chrono(started,finished)<43200 and completion['budget']['elapsed_seconds']<43200)
        ck(f'{study}_output_byte_budget',man['budget']['peak_observed_bytes']==completion['budget']['peak_observed_bytes']<16*1024**3)
        lines=tread(job/'train.log').splitlines(); events=[];warnings=[]
        for i,line in enumerate(lines,1):
            if line.startswith('{'): events.append({'line':i,**json.loads(line)})
            elif line.strip(): warnings.append({'line':i,'text':line})
        updates=[x for x in events if x.get('event')=='updates']
        ck(f'{study}_72_logged_events',len(updates)==72)
        ck(f'{study}_log_times_monotone',all(y['elapsed_seconds']>x['elapsed_seconds'] for x,y in zip(updates,updates[1:])))
        for o in ['ABC','BCA','CAB']:
            for s in [2,3]:
                e=[x for x in updates if x['order']==o and x['stage']==s]
                ck(f'{study}_{o}_{s}_logged_steps',[x['stage_updates'] for x in e]==list(range(24,289,24)) and all(x['logical_updates']==288*(s-1)+x['stage_updates'] and x['arm']==arm for x in e))
        ck(f'{study}_no_logged_production_exception',all('FutureWarning:' in x['text'] or 'if pooling_modes(' in x['text'] for x in warnings))
        ck(f'{study}_completion_log_status',events[-1].get('status')==completion['status'])
        ck(f'{study}_no_failure_file',(job/'failure.json').is_file() is False)
        # Full supplied return list -> Linux member path; returned_path is Windows display alias.
        inv=jread(result/'return_inventory.json');retpaths=[]
        for r in inv['files']:
            verify(p/r['path'],r,f'{study}_returned_file_identity')
            retpaths.append(r['returned_path'])
            ck(f'{study}_linux_windows_path_mapping',r['path']==job.relative_to(p).as_posix()+'/'+r['returned_path'].removeprefix('job/'))
        ck(f'{study}_return_totals',len(inv['files'])==inv['file_count']==cfg['returned'] and sum(r['bytes'] for r in inv['files'])==inv['total_bytes'])
        ck(f'{study}_return_unique_paths',len(retpaths)==len(set(retpaths))==len({r['path'] for r in inv['files']}))
        sync=jread(result/'return_sync.json');ck(f'{study}_sync_record_totals',sync['files_verified']==inv['file_count'] and sync['bytes_verified']==inv['total_bytes'])
        custody=jread(result/'payload_custody.json'); ck(f'{study}_six_native_weight_records',len(custody['weights_retained_linux'])==6)
        for r in custody['weights_retained_linux']:
            n=Path(r['path']).stem; pr=points[n]['model']; expected_path=(job/'run'/pr['path']).relative_to(p).as_posix()
            ck(f'{study}_{n}_native_weight_metadata',r['path']==expected_path and all(r[k]==pr[k] for k in ['bytes','sha256']))
        for r in custody['excluded_memory']:
            n=Path(r['path']).stem; ck(f'{study}_{n}_custody_memory_bytes',r['bytes']==man['memories'][n]['serialized_bytes'])
        # Native hand-written CPU evidence: parse raw streams and saved numerical records.
        cout=tread(cpu/'stdout.log'); cerr=tread(cpu/'stderr.log'); ccontracts=jread(cpu/'contracts.json')
        ck(f'{study}_cpu_exit0',tread(cpu/'exit_status.txt').strip()=='0' and cpures['Exit status']=='0')
        ck(f'{study}_cpu_contract_count',len(re.findall(r'^test_.*\.\.\. ok$',cerr,re.M))==(22 if study=='low' else 28)==ccontracts['passed'] and ccontracts['failed']==ccontracts['skipped']==0)
        ck(f'{study}_cpu_no_forbidden',cad['formal_inputs'] is False and cad['formal_labels'] is False and cad['formal_updates']==0 and cad['gpu'] is False)
        nd={}
        for r in [*cad['native'].values(),*cad['native_reference'].values()]:
            verify(cpu/r['path'],r,f'{study}_native_raw_record_identity')
            value=jread(cpu/r['path']);nd[value['arm']]=value
            ck(f'{study}_{value["arm"]}_actual_native_updates',value['native_updates_actually_executed']==2 and value['warm_update']['adam_step']==1 and value['weighted_update']['adam_step']==289)
            ck(f'{study}_{value["arm"]}_native_parameter_probes',len(value['parameter_probes'])==29 and all(x['parameters_changed'] is True and math.isfinite(x['gradient_norm_after_clip']) and x['gradient_norm_after_clip']>0 for x in value['parameter_probes']))
        ref=nd['quarter']; candidate=nd[arm]
        for k in ['initial_model_state_sha256','after_warm_model_state_sha256','before_stage2_optimizer_sha256','captured_current_logits','captured_history_logits']:
            ck(f'{study}_native_paired_{k}',ref[k]==candidate[k])
        ck(f'{study}_cpu_four_native_updates',cad['native_updates_actually_executed']==sum(x['native_updates_actually_executed'] for x in nd.values())==4)
        native_summary={'environment':cad['environment'],'started':cstarted.strip(),'finished':cfinished.strip(),'timestamp_elapsed_seconds':chrono(cstarted,cfinished),'resource':cpures,'contracts_raw_stream_count':len(re.findall(r'^test_.*\.\.\. ok$',cerr,re.M)),'native_updates_actually_executed':cad['native_updates_actually_executed'],'counter_fixture':candidate['counter_fixture'],'parameter_probes_per_arm':{k:len(v['parameter_probes']) for k,v in nd.items()},'gradient_vector_values_available':False,'component_comparison_recorded':cad['native_component_comparison']}
        if study=='low':
            ratios={}
            for part in ['encoder_first_query','head_hidden']:
                cr=ref['gradient_components'][part];cc=candidate['gradient_components'][part]
                ck(f'{study}_{part}_native_current_summary_matches',cr['current']==cc['current'])
                ratio=cr['weighted_history']['norm']/cc['weighted_history']['norm']; ratios[part]=ratio
                # Diagnostic only: independently reduced float32 norms cannot
                # recover or certify the per-element scaling of absent vectors.
            native_summary['recomputed_norm_ratios_only']=ratios
        else:
            observed=np.asarray(candidate['captured_history_logits']);target=np.asarray(candidate['origin_eval_reference'])
            mse=float(np.dot(observed-target,observed-target)/len(observed))
            ck(f'{study}_native_saved_handwritten_MSE',len(target)==378 and abs(mse-candidate['independent_logit_mse'])<1e-16,recomputed=mse)
            native_summary['recomputed_saved_handwritten_mse']=mse
            native_summary['elementwise_gradient_increment_not_recomputable']='Only norms and hashes provided, no gradient vectors. The maximum_difference fields belong to the native CPU computation.'
        # Supplied result audit original streams, provenance, references, not its PASS as a conclusion.
        audit_exec=jread(result/'audit_execution.json');aa=jread(result/'audit/audit.json');astd=json.loads(tread(result/'audit_stdout.log'));asterr=note(result/'audit_stderr.log','complete_original_stderr')
        ck(f'{study}_audit_stdout_exact_bytes',note(result/'audit_stdout.log','stream_byte_comparison')==note(result/'audit/audit.json','stream_byte_comparison'))
        ck(f'{study}_audit_stdout_json',astd==aa)
        ck(f'{study}_audit_exit_and_stderr',audit_exec['exit_code']==0 and len(asterr)==0)
        for which,filename in [('script','scripts/step28_er_weight_audit.py'),('helper','scripts/step28_bge_continual_audit.py')]:
            b=note(p/filename,'audit_source_identity');ck(f'{study}_audit_{which}_identity',len(b)==aa[which]['bytes'] and digest(b)==aa[which]['sha256'])
        for r in jread(result/'audit_return_inventory.json')['files']: verify(result/r['path'],r,f'{study}_audit_return_identity')
        # Read diagnostics fully but do not substitute its conclusions for above checks.
        diagnostics=jread(result/'audit/diagnostics.json')
        detail[study]={'source_root':root.relative_to(p).as_posix(),'job':job.relative_to(p).as_posix(),'source_files':list(source.values()),'separate_freeze_json_in_submission':False,'authorization_raw_in_submission':not missing_auth,'execution_authorization_record':exe['authorization'],'restored_starts':restored,'runtime':{k:exe[k] for k in ['python','torch','cuda','gpu','cpu_affinity']},'started':started.strip(),'finished':finished.strip(),'timestamp_elapsed_seconds':chrono(started,finished),'gnu_resource':res,'manifest_budget':man['budget'],'completion_budget':completion['budget'],'new_updates_from_rows':sum(len(ar) for ar in all_rows),'new_gradient_presentations_from_ID_lists':3456,'new_points':len(points),'new_score_arrays':sum(len(x['scores']) for x in points.values()),'complete_status':completion['status'],'startup_status':start['status'],'progress_snapshot_status':progress['status'],'access':access,'before_valid':before,'update_details':update_details,'log_events':updates,'non_json_log_lines':warnings,'native_cpu':native_summary,'native_weight_records':custody['weights_retained_linux'],'native_weights_total_bytes':sum(x['bytes'] for x in custody['weights_retained_linux']),'return_file_count':inv['file_count'],'return_total_bytes':inv['total_bytes'],'result_audit_execution':audit_exec,'result_audit_recorded_numeric_comparisons':aa['numeric_comparisons'],'result_audit_recorded_maximum_absolute_error':aa['maximum_absolute_error'],'result_audit_recorded_maximum_base_loss_decomposition_error':aa['maximum_base_loss_decomposition_error'],'historical_handwritten_evidence_full_archives_in_submission':False,'old_quarter_update_logs_provided':False,'paired_comparison_scope':'Current logs directly compared to original ER1; old quarter paired status inherited from original result review and manifest bindings; no direct row comparison of absent quarter logs'}
    changed=[]; added=[]
    for name,row in allsource['logit'].items():
        if name not in allsource['low']: added.append(name)
        elif row!=allsource['low'][name]: changed.append(name)
    ck('two_source_roots_7_changed_3_added',len(changed)==7 and len(added)==3,changed=changed,added=added)
    detail['source_delta']={'changed':changed,'added':added}
    # Historical full report and disposition reads: substantive interpretation in notes.md.
    manual=['AGENTS.md','docs/RESEARCH_DISCIPLINE.zh.md','docs/SELLER_ALIAS_ER_LOW.zh.md','docs/SELLER_ALIAS_LOGIT_WEIGHT.zh.md','reports/documentation/20261002/er_low/review/external/REVIEW.zh.md','reports/documentation/20261002/er_low/review/report.zh.md','reports/documentation/20261002/logit_weight/review/external/REVIEW.zh.md','reports/documentation/20261002/logit_weight/review/report.zh.md','reports/seller_alias_continual/20261001/bge_continual_result/review/external/REVIEW.zh.md','reports/seller_alias_continual/20261001/bge_continual_result/review/report.zh.md','reports/seller_alias_continual/20261002/er_weight_result/review/external/REVIEW.zh.md','reports/seller_alias_continual/20261002/er_weight_result/review/report.zh.md']
    for name in manual:note(p/name,'full_text_also_manually_read_in_tool_outputs')
    detail['environment']={'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'formal_models_loaded':False,'new_training':False,'formal_label_parses':0,'network_used':False}
    detail['failed_checks']=[x for x in checks if not x['passed']]
    detail['passed_check_count']=sum(x['passed'] for x in checks)
    detail['check_count']=len(checks)
    (out/'provenance_result.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2)+'\n')
    (out/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    for x in opened.values():x['modes']=sorted(set(x['modes']))
    (out/'read_coverage.json').write_text(json.dumps(sorted(opened.values(),key=lambda x:x['path']),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'check_count':len(checks),'failed_checks':detail['failed_checks'],'source_roots':{x:len(allsource[x]) for x in allsource},'read_files':len(opened),'per_study':{x:{k:detail[x][k] for k in ['started','finished','timestamp_elapsed_seconds','new_updates_from_rows','new_gradient_presentations_from_ID_lists','new_points','new_score_arrays','return_file_count','return_total_bytes','native_weights_total_bytes']} for x in ['low','logit']}},ensure_ascii=False,indent=2))
    return 1 if detail['failed_checks'] else 0

if __name__=='__main__':raise SystemExit(main())
