"""Independent audit of already-open saved evidence, in the web review workspace.

No project imports, model loading, training, label reconstruction, or network.
Reproduces ONLY the frozen endpoints, 5000-draw PCG64 statistics, 23 checks,
epoch1-to-6 diagnoses, and registered historical matrix comparisons.
Original files are read-only and are never replaced by these calculations.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
import traceback
import zipfile
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
INPUT = BASE / 'input'
ORDERS = ('ABC','BCA','CAB')
ARMS = ('LOGIT0.1','C','S','C_plus','S_strong')
ROLES = ('raw','stage-cal','first-cal','primary')
METRICS = ('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct',
           'brier','log_loss','precision','recall','f1','specificity',
           'balanced_accuracy','mcc','map','mrr','recall_at_1','recall_at_3',
           'recall_at_5','recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')
RANK = [0,1,2,3,*range(12,22)]
DIAG = ('map','recall_at_5','log_loss')
TERMS = {
    'O': ((3,1,.5),(3,2,.5)), 'N': ((2,2,.5),(3,3,.5)),
    'Z': ((3,3,1.),), 'A2': ((2,2,1.),),
    'F_first': ((1,1,1.),(3,1,-1.)),
    'F': ((1,1,.5),(3,1,-.5),(2,2,.5),(3,2,-.5)),
    'G': ((2,2,.5),(1,2,-.5),(3,3,.5),(2,3,-.5)),
    'final_all': ((3,1,1/3),(3,2,1/3),(3,3,1/3)),
}
SEEN = {}
CHECKS = Counter()
STATS = defaultdict(lambda: {'numeric_values':0,'unequal_values':0,'max_abs_difference':0.,'max_path':None})

def save(name, obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def expect(value, description):
    if not value:
        raise AssertionError(description)
    CHECKS[description.split(':')[0]] += 1

def payload(path, mode='saved_evidence_read'):
    path = Path(path).resolve()
    raw = path.read_bytes()
    rel = str(path.relative_to(BASE))
    rec = SEEN.setdefault(rel, {'path':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'modes':[]})
    if mode not in rec['modes']:
        rec['modes'].append(mode)
    return raw

def js(path):
    return json.loads(payload(path, 'all_JSON_fields_processed'))

def verified(root, rec):
    path = root/rec['path']
    raw = payload(path)
    expect(len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['sha256'], 'linked_file_identity:'+str(path))
    return path

def arr(path, shape=None, dtype=None):
    v=np.load(io.BytesIO(payload(path,'all_saved_array_values_processed')),allow_pickle=False)
    expect(np.isfinite(v).all(), 'finite_array:'+str(path))
    if shape is not None: expect(v.shape==shape, 'array_shape:'+str(path))
    if dtype is not None: expect(v.dtype==np.dtype(dtype), 'array_dtype:'+str(path))
    return v

def compare(a,b,category,path=''):
    if isinstance(a,dict):
        expect(set(a)==set(b),'same_fields:'+category+path)
        for k in a: compare(a[k],b[k],category,path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        expect(len(a)==len(b),'same_length:'+category+path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,category,path+'/'+str(i))
    elif isinstance(a,(int,float,np.number)) and not isinstance(a,(bool,np.bool_)):
        d=abs(float(a)-float(b)); s=STATS[category];s['numeric_values']+=1
        s['unequal_values']+=int(a!=b)
        if d>s['max_abs_difference']:s.update(max_abs_difference=d,max_path=path)
        expect(d<=2e-12, 'numeric_agreement:'+category+path)
    else: expect(a==b,'exact_value:'+category+path)

def seed(seed,*parts):
    raw=(json.dumps([seed,*parts],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%(2**63-1)

def point(order,arm,stage):
    return order+'_shared' if stage==1 and arm!='LOGIT0.1' else f'{order}_{arm}_stage{stage}'

def loaded_matrices():
    root=INPUT/'result/job/evaluation'; col=js(root/'collected.json')
    expect(col['metric_columns']==list(METRICS),'frozen_metric_columns')
    expect(len(col['group_ids'])==len(set(col['group_ids']))==60,'sixty_unique_valid_groups')
    expect(Counter(col['domains'])==dict(A=20,B=20,C=20),'actual_domain_counts')
    matrices={}
    for name,roles in col['points'].items():
        matrices[name]={r:arr(verified(root,rec['matrix']),(60,22),'float64') for r,rec in roles.items()}
        expect(set(roles)==set(ROLES[:3]),'three_saved_roles:'+name)
        for r in ROLES[1:3]:
            expect(np.array_equal(matrices[name][r][:,RANK],matrices[name]['raw'][:,RANK]),'affine_rank_invariance:'+name+r)
    return col,matrices

def make_field(matrices,col,arm,role,endpoint):
    f=np.zeros((3,3,20,22),dtype=np.float64)
    indices={d:np.flatnonzero(np.asarray(col['domains'])==d) for d in 'ABC'}
    for oi,order in enumerate(ORDERS):
        for stage,arrival,weight in TERMS[endpoint]:
            item=matrices[point(order,arm,stage)]
            v=item['stage-cal'].copy() if role=='primary' else item[role]
            if role=='primary':v[:,RANK]=item['raw'][:,RANK]
            d=order[arrival-1];di='ABC'.index(d)
            f[oi,di]+=weight*v[indices[d]]
    if endpoint in ('F_first','F','G'):f[...,[4,5]]*=-1
    return f

def summarize(f,draws):
    # Orders are fixed conditions. Average them BEFORE cluster resampling.
    avg=f.mean(axis=0)
    means=avg.mean(axis=1).sum(axis=0)
    by_order=f.mean(axis=2).sum(axis=1)
    boots=sum(avg[d][draws[:,d]].mean(axis=1) for d in range(3))
    cis=np.quantile(boots,[.025,.975],axis=0,method='linear')
    return {m:{'mean':float(means[i]),'per_order':dict(zip(ORDERS,by_order[:,i].tolist())),
               'conditional_95pct_interval':cis[:,i].tolist()} for i,m in enumerate(METRICS)}

def criteria(delta,rawref):
    c={'old_map_improves':delta['O']['map']['mean']>0,
       'old_map_interval_above_zero':delta['O']['map']['conditional_95pct_interval'][0]>0,
       'old_recall5_improves':delta['O']['recall_at_5']['mean']>0,
       'new_map_non_decrease':delta['N']['map']['mean']>=0,
       'new_recall5_non_decrease':delta['N']['recall_at_5']['mean']>=0}
    guard={'average_precision':1,'roc_auc':1,'brier':-1,'log_loss':-1}
    for ep in ('O','N'):
        for m,s in guard.items():c[f'{ep}_{m}_non_degradation']=s*delta[ep][m]['mean']>=0
        for m in ('brier','log_loss'):c[f'{ep}_{m}_against_raw_reference']=rawref[ep][m]['mean']<=0
    for m,s in {'map':1,'recall_at_5':1,**guard}.items():c[f'Z_{m}_non_degradation']=s*delta['Z'][m]['mean']>=0
    expect(len(c)==23,'twenty_three_frozen_checks')
    return c

def classification(count):
    tp,fp,fn,tn=(count[k] for k in ('tp','fp','fn','tn'))
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    sp=tn/(tn+fp) if tn+fp else 0.;den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return dict(precision=p,recall=r,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                specificity=sp,balanced_accuracy=(r+sp)/2,mcc=(tp*tn-fp*fn)/den if den else 0.)

def execution_audit():
    root=INPUT/'result/job';run=root/'run'
    man=js(run/'manifest.json');part=js(run/'partition.json');comp=js(root/'completion.json')
    ev=js(root/'evaluation/evaluation.json');col,mats=loaded_matrices()
    pol=js(INPUT/'result/source/schema/step28_replay_improvement_policy.json')
    refs=[js(root/'execution.json')['sources'],col['source_files'],ev['source_files'],
          js(INPUT/'result/qualification/formal_gate.json')['source_files'],
          js(INPUT/'background/cpu/result.json')['source_files'],js(INPUT/'background/gpu/result.json')['source_files']]
    for vals in refs:expect(vals==man['source_files'],'twenty_eight_sources_agree')
    expect(len(man['source_files'])==28,'source_count')
    for rec in man['source_files']:verified(INPUT/'result/source',rec)
    b=js(root/'before_valid.json');verified(root,b['manifest'])
    for k in ('evaluation','overfitting'):verified(root,comp[k])
    expect(b['access']==dict(train=1,valid=0,heldout=0,owners=0),'blind_gate_access')
    expect(comp['access']==js(root/'access.json')==dict(train=1,valid=1,heldout=0,owners=0),'final_access')
    for k,n in [('fit',48),('calibration',12),('development',20)]:
        expect(Counter(r['domain'] for r in part[k])=={d:n for d in 'ABC'},'partition_counts:'+k)
    ids=[r['group_uid'] for role in part.values() for r in role]
    expect(len(ids)==len(set(ids))==240,'partition_identity_separation')
    expect(col['group_ids']==[r['group_uid'] for r in part['development']],'collected_identity_order')
    specifications={point(o,a,s):(o,'C' if point(o,a,s).endswith('_shared') else a,s) for o in ORDERS for a in ARMS for s in (1,2,3)}
    for k in ('points','training','diagnostics'):expect(set(man[k])==set(specifications),'physical_coverage:'+k)
    for a in ARMS:
        expect(col['logical_points'][a]=={f'{o}_stage{s}':point(o,a,s) for o in ORDERS for s in (1,2,3)},'logical_mapping:'+a)
    for o in ORDERS:
        for a in ARMS[1:]:expect(man['starts'][o+'_'+a]==man['starts'][o+'_C'],'shared_complete_start:'+o+a)
        expect(man['starts'][o+'_C']['full_state_sha256']==man['points'][o+'_shared']['full_state']['state_sha256'],'shared_full_digest:'+o)
    domain={r['group_uid']:r['domain'] for r in part['fit']}
    schedules={};retained={};logs={};memory_bytes=[];loss_max=0.;grad_presentations=0
    for o in ORDERS:
        rng=random.Random(seed(20260930,o,'retention'));members=[];seen=0
        for st,d in enumerate(o[:2],1):
            for u in sorted(u for u,dd in domain.items() if dd==d):
                seen+=1;j=len(members) if len(members)<6 else rng.randrange(seen)
                if j<6:
                    if j==len(members):members.append(u)
                    else:members[j]=u
            retained[o,st]=members.copy()
    for name,(o,a,s) in specifications.items():
        p=man['points'][name];log=js(verified(run,man['training'][name]));logs[name]=log
        expect(js(run/'points'/(name+'.json'))==p,'point_receipt_matches:'+name)
        expect((p['order'],p['arm'],p['stage'])==(o,a,s),'point_identity:'+name)
        expect(p['full_restore_verified'] is True and p['adam_step']==288*s,'saved_restore_receipt:'+name)
        expect(log['adam_step']==288*s and log['updates']==len(log['rows'])==288,'stage_update_count:'+name)
        rng=random.Random(seed(20260918,o,s,'current'));seq=[]
        for epoch in range(6):
            block=sorted(u for u,d in domain.items() if d==o[s-1]);rng.shuffle(block);seq+=block
        expect(seq==log['current_ids']==[r['current_uid'] for r in log['rows']],'independent_current_schedule:'+name)
        expect(p['calibration_ids']==[r['group_uid'] for r in part['calibration'] if r['domain']==o[s-1]],'calibration_ids:'+name)
        summ=p['memory_summary'];member=retained[o,min(s,2)]
        expect(summ['members']==member and summ['seen']==48*min(s,2),'independent_algorithm_R:'+name)
        size=summ.get('serialized_bytes',summ.get('bytes'));memory_bytes.append(size)
        expect(size==p['memory']['bytes'] and summ['sha256']==p['memory']['sha256'] and size<=1048576,'memory_receipt_budget:'+name)
        origin_key='reference_origins' if a=='LOGIT0.1' else 'origins'
        source_key='references' if a=='LOGIT0.1' else 'tables'
        expect(set(summ[origin_key])==set(member)==set(summ[source_key]),'memory_source_coverage:'+name)
        expect(all(domain[u]==o[st-1] for u,st in summ[origin_key].items()),'memory_birth_domain:'+name)
        if s>1:
            before,after=log['memory_before'],log['memory_after'];prior=man['points'][point(o,a,s-1)]['memory_summary']
            for key in ('members','seen',origin_key,source_key):
                expect(before[key]==after[key]==prior[key],'memory_stable_during_stage:'+name+key)
            for u in set(after[source_key])&set(summ[source_key]):expect(after[source_key][u]==summ[source_key][u],'survivor_source_hash_unchanged:'+name+u)
            rng=random.Random(seed(20260930,o,s,'history_draws'));want=[before['members'][rng.randrange(6)] for _ in range(288)]
            expect(log['history_ids']==want==[r['history_uid'] for r in log['rows']],'independent_history_draws:'+name)
            pair=(seq,want,before['members'])
            expect(schedules.get((o,s),pair)==pair,'cross_arm_pairing:'+name);schedules[o,s]=pair
            memory_bytes += [x.get('serialized_bytes',x.get('bytes')) for x in (before,after)]
            expect(all(x.get('serialized_bytes',x.get('bytes'))<=1048576 for x in (before,after)),'before_after_memory_budget:'+name)
        else:expect(log['history_ids']==[] and log['memory_before'] is None and log['memory_after'] is None,'no_history_first_stage:'+name)
        for step,row in enumerate(log['rows'],1):
            expect(row['step']==step and row['head_lr']==.001 and row['encoder_lr']==1e-5*(step/29 if step<=29 else (288-step)/259),'actual_learning_rate:'+name)
            expect(np.isfinite(row['gradient_norm']) and row['gradient_norm']>0,'finite_nonzero_gradient_norm:'+name)
            if a=='LOGIT0.1':value=row['current_total']+(.1*row['history_total']+.5*row['logit_mse'] if s>1 else 0)
            else:
                value=row['current']['supervised']
                if s>1:
                    ca,cb=pol['teacher_coefficients'][a];h=row['history'];value+=.1*h['supervised']+ca*h['mse0']+cb*h['mse1']
            loss_max=max(loss_max,abs(value-row['total']))
            expect(np.isclose(value,row['total'],rtol=2e-6,atol=2e-6),'saved_objective_decomposition:'+name)
            grad_presentations+=1+int(row['history_uid'] is not None)
        mapping=js(verified(run,p['mapping']));expect(mapping['status']=='PASS_CALIBRATION_FIT','calibration_receipt:'+name)
        raw=arr(verified(run,p['scores']['raw']),(60,378),'float32')
        arr(verified(run,p['scores']['calibration']),(12,378),'float32')
        for role,mkey in [('stage-cal','stage'),('first-cal','first')]:
            values=arr(verified(run,p['scores'][role]),(60,378),'float64');m=p['maps'][mkey]
            expect(m['a']>0 and np.array_equal(values,raw.astype(np.float64)*m['a']+m['b']),'saved_affine_transform:'+name+role)
        first=man['points'][point(o,a,1)]['maps']['first']
        expect(p['maps']['first']==first,'first_map_fixed:'+name)
        entries=[]
        for epoch,rec in enumerate(man['diagnostics'][name],1):
            e=js(verified(run/'diagnostics',rec));entries.append(e)
            expect((e['point'],e['order'],e['arm'],e['stage'],e['epoch'],e['stage_step'],e['adam_step'])==(name,o,a,s,epoch,48*epoch,288*(s-1)+48*epoch),'diagnostic_identity_step:'+name)
            expect(e['valid_labels_used'] is False and e['mode']=='A0_eval_no_autograd' and e['seed']=='s0','diagnostic_blind_receipt:'+name)
            expected_fit=[r['group_uid'] for r in part['fit'] if r['domain']==o[s-1]]
            cache=[] if s==1 else log['memory_before']['members']
            expect(e['fit_ids']==expected_fit and e['fit_domains']==[o[s-1]]*48 and e['cache_ids']==cache and e['cache_domains']==[domain[u] for u in cache] and e['valid_ids']==col['group_ids'],'diagnostic_population_identity:'+name)
            expect([t['uid'] for t in e['teacher_errors']]==cache,'diagnostic_teacher_identity:'+name)
            for role in ('fit','cache'):arr(verified(run/'diagnostics',e[role]),(len(e[role+'_ids']),3),'float64')
            v=arr(verified(run/'diagnostics',e['valid_blind']),(60,378),'float32')
            if epoch==6:expect(np.array_equal(v,raw),'epoch6_equals_stage_raw:'+name)
        expect(len(entries)==6,'six_epochs:'+name)
    expect(sum(l['updates'] for l in logs.values())==man['physical_updates']==10368,'total_updates')
    expect(grad_presentations==man['gradient_group_presentations']==19008,'gradient_group_presentations')
    expect(comp['budget']['limits']==pol['runtime'],'current_72_hour_limits')
    for a,bk in [('elapsed_seconds','maximum_gpu_stage_seconds'),('peak_cuda_reserved_bytes','maximum_cuda_reserved_bytes'),('peak_rss_bytes','maximum_rss_bytes'),('peak_output_bytes','maximum_output_bytes')]:
        expect(0<comp['budget'][a]<=pol['runtime'][bk],'recorded_actual_budget:'+a)
    excluded=sum(p[k]['bytes'] for p in man['points'].values() for k in ('model','memory'))
    result={'physical_stages':len(man['points']),'logical_stages':sum(map(len,col['logical_points'].values())),
            'physical_epochs':sum(map(len,man['diagnostics'].values())),'logical_epochs':270,
            'updates':10368,'gradient_group_presentations':grad_presentations,'valid_unique_groups':60,
            'memory_recorded_min_bytes':min(memory_bytes),'memory_recorded_max_bytes':max(memory_bytes),
            'maximum_objective_absolute_roundoff':loss_max,'excluded_model_memory_payload_bytes':excluded,
            'budget':comp['budget'],'direct_model_or_memory_payload_checks':'SKIPPED_BY_USER_SCOPE; receipt hashes/summaries only',
            'cross_method_pairing':'same current groups, history draws and group ordering; no claim of cross-architecture dropout identity'}
    save('execution_audit.json',result);print('EXECUTION',json.dumps(result,ensure_ascii=False),flush=True)

def metrics_audit():
    col,mats=loaded_matrices();saved=js(INPUT/'result/job/evaluation/evaluation.json')
    pol=js(INPUT/'result/source/schema/step28_replay_improvement_policy.json')
    draws=np.random.Generator(np.random.PCG64(20260930)).integers(0,20,size=(5000,3,20))
    fs={a:{r:{ep:make_field(mats,col,a,r,ep) for ep in TERMS} for r in ROLES} for a in ARMS}
    end={};deltas={};rawref={};check_rows={}
    for a in ARMS:
        end[a]={r:{ep:summarize(fs[a][r][ep],draws) for ep in TERMS} for r in ROLES}
        compare(end[a],saved['endpoints'][a],'endpoint_summaries',a)
        print('METRIC endpoints',a,flush=True)
    for a,b in pol['evaluation']['comparisons']:
        key=a+'-'+b
        d={r:{ep:summarize(fs[a][r][ep]-fs[b][r][ep],draws) for ep in TERMS} for r in ROLES}
        v={ep:summarize(fs[a]['stage-cal'][ep]-fs[b]['raw'][ep],draws) for ep in TERMS}
        compare(d,saved['comparisons'][key]['delta'],'paired_deltas',key)
        compare(v,saved['comparisons'][key]['candidate_cal_minus_other_raw'],'cross_role_comparisons',key)
        checks=criteria(d['primary'],v);compare(checks,saved['comparisons'][key]['checks'],'criterion_booleans',key)
        expect(all(checks.values())==saved['comparisons'][key]['pass'],'comparison_pass:'+key)
        expect(set(k for k,ok in checks.items() if not ok)==set(saved['comparisons'][key]['assessment']['failed']),'criterion_failed_names:'+key)
        check_rows[key]={'passed':sum(checks.values()),'total':23,'checks':checks,'failed':[k for k,ok in checks.items() if not ok]}
        deltas[key]=d;rawref[key]=v
        print('METRIC comparison',key,sum(checks.values()),flush=True)
    passed=all(check_rows['C_plus-'+a]['passed']==23 for a in pol['evaluation']['continue_requires_all_23_against'])
    expect(passed==saved['development_criteria_pass']==False,'frozen_overall_false')
    count_values=0
    for name,roles in col['points'].items():
        for r,info in roles.items():
            counts=js(verified(INPUT/'result/job/evaluation',info['counts']))
            expect(len(counts)==60,'count_rows')
            for i,c in enumerate(counts):
                expect(all(type(c[k]) is int and c[k]>=0 for k in ('tp','fp','fn','tn')) and c['tp']+c['fn']==20 and c['fp']+c['tn']==358,'valid_confusion_counts')
                for m,val in classification(c).items():compare(val,float(mats[name][r][i,METRICS.index(m)]),'count_derived_metrics',name+'/'+r+'/'+str(i)+'/'+m)
                count_values+=4
            actual=saved['absolute_stage_results'][name][r]
            compare(dict(zip(METRICS,mats[name][r].mean(0).tolist())),actual['macro_all'],'absolute_macro_all',name+'/'+r)
            for dom in 'ABC':
                idx=np.flatnonzero(np.asarray(col['domains'])==dom)
                compare(dict(zip(METRICS,mats[name][r][idx].mean(0).tolist())),actual['macro_by_domain'][dom],'absolute_macro_by_domain',name+'/'+r+'/'+dom)
    save('recomputed_metrics.json',{'scope':'Web-workspace reproduction of frozen statistics, not replacement estimates','endpoints':end,'comparisons':deltas,'cross_role':rawref,'checks':check_rows,'development_criteria_pass':passed})
    result={'matrix_count':108,'matrix_numeric_values':108*60*22,'confusion_count_values':count_values,
            'summary_metric_objects':(5*4*8+7*4*8+7*8)*22,'frozen_summary_numeric_values':58080,
            'checks':check_rows,'development_criteria_pass':passed,'numeric_comparison':dict(STATS)}
    save('metrics_audit.json',result);print('METRIC_SUMMARY',json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False),flush=True)

def diagnostic_curve(t,v,s):
    if t.shape[1]==0:return {'status':'NOT_APPLICABLE_NO_SURVIVING_GROUPS'}
    rng=np.random.Generator(np.random.PCG64(s));td=rng.integers(t.shape[1],size=(5000,t.shape[1]));vd=rng.integers(v.shape[1],size=(5000,v.shape[1]))
    sign=np.array([1.,1.,-1.]);dt=(t[-1]-t[0])*sign;dv=(v[-1]-v[0])*sign
    def summary(a,draw):return {'mean':a.mean(0).tolist(),'conditional_95pct_interval':np.quantile(a[draw].mean(1),[.025,.975],axis=0).T.tolist()}
    st,sv=summary(dt,td),summary(dv,vd)
    gap=(t.mean(1)-v.mean(1))*sign;boot=(t[:,td].mean(2)-v[:,vd].mean(2))*sign
    widening=np.quantile(boot[-1]-boot[0],[.025,.975],axis=0).T
    labels={}
    for k,m in enumerate(DIAG):
        supported=st['conditional_95pct_interval'][k][0]>0 and sv['conditional_95pct_interval'][k][1]<0 and widening[k,0]>0
        signal=st['mean'][k]>0 and sv['mean'][k]<0 and gap[-1,k]>gap[0,k]
        labels[m]='有相应迹象' if supported else '证据不足' if signal else '未见明确迹象'
    return {'train_raw_curve':t.mean(1).tolist(),'valid_raw_curve':v.mean(1).tolist(),
            'benefit_gap_curve':gap.tolist(),'benefit_gap_conditional_intervals':np.quantile(boot,[.025,.975],axis=1).transpose(1,2,0).tolist(),
            'epoch1_to_6_train_benefit':st,'epoch1_to_6_valid_benefit':sv,
            'epoch1_to_6_gap_widening':{'mean':(gap[-1]-gap[0]).tolist(),'conditional_95pct_interval':widening.tolist()},
            'interpretation':labels,'fit_or_cache_groups':t.shape[1],'valid_groups':v.shape[1]}

def diagnostics_audit():
    run=INPUT/'result/job/run';out=INPUT/'result/job/evaluation/diagnostics'
    man=js(run/'manifest.json');saved=js(out/'overfitting.json');col,mats=loaded_matrices()
    all_result={};counts=Counter();contexts=Counter();missing=[];teacher=[];draw_populations={};scalar_identity_max=0.
    for p,recs in man['diagnostics'].items():
        rows=[js(verified(run/'diagnostics',rec)) for rec in recs]
        first=rows[0];curve={role:[] for role in ('fit','cache','valid')}
        for e in rows:
            for k in ('fit_ids','cache_ids','fit_domains','cache_domains','valid_ids'):expect(e[k]==first[k],'stable_epoch_population:'+p+k)
            for role in ('fit','cache'):curve[role].append(arr(verified(run/'diagnostics',e[role]),(len(e[role+'_ids']),3),'float64'))
            curve['valid'].append(arr(verified(out,saved['group_metrics'][f'{p}_epoch{e["epoch"]}']),(60,3),'float64'))
            if e['teacher_errors']:
                values=e['teacher_errors'];means={}
                for k in ('D0','D1','residual_cross_mean','residual_difference_mse'):
                    means[k]=sum(x[k] for x in values)/len(values) if values[0][k] is not None else None
                teacher.append(dict(point=p,arm=e['arm'],order=e['order'],stage=e['stage'],epoch=e['epoch'],cache_groups=len(values),**means))
                for x in values:
                    if x['D1'] is not None:
                        d=abs(x['D0']+x['D1']-2*x['residual_cross_mean']-x['residual_difference_mse']);scalar_identity_max=max(scalar_identity_max,d)
                        expect(d<=2e-12,'residual_scalar_identity:'+p)
                    else:expect(e['arm']=='LOGIT0.1' and x['residual_cross_mean'] is None and x['residual_difference_mse'] is None,'logit_no_A1:'+p)
        curve={k:np.stack(v) for k,v in curve.items()}
        expect(np.array_equal(curve['valid'][-1],mats[p]['raw'][:,[12,16,5]]),'diagnostic_epoch6_metric_link:'+p)
        all_result[p]={}
        for role in ('fit','cache'):
            domains=[first['order'][first['stage']-1]] if role=='fit' else list(first['order'][:first['stage']-1])
            all_result[p][role]={}
            if not domains:
                expect(saved['points'][p][role]=={'status':'NOT_APPLICABLE_FIRST_STAGE'},'first_stage_cache_NA:'+p)
            for dom in domains:
                sel=np.flatnonzero(np.asarray(first[role+'_domains'])==dom);vr=np.flatnonzero(np.asarray(col['domains'])==dom)
                pop=([first[role+'_ids'][i] for i in sel],[col['group_ids'][i] for i in vr]);key=(first['order'],first['stage'],role,dom)
                expect(draw_populations.get(key,pop)==pop,'cross_arm_diagnostic_group_pairing:'+p+role+dom);draw_populations[key]=pop
                expect(not(set(pop[0])&set(pop[1])),'independent_fit_valid_identities:'+p+role+dom)
                result=diagnostic_curve(curve[role][:,sel],curve['valid'][:,vr],seed(20260930,first['order'],first['stage'],role,dom,'diagnostics'))
                expected=saved['points'][p][role][dom]
                if 'interpretation' in result:
                    compare(result,{k:expected[k] for k in result},'diagnostic_arrays_and_intervals',p+'/'+role+'/'+dom)
                    contexts[role]+=1
                    for m,l in result['interpretation'].items():counts[role+'/'+m+'/'+l]+=1
                else:
                    expect(result==expected,'no_survivor_NA:'+p+role+dom);missing.append([p,role,dom])
                all_result[p][role][dom]=result
    original_teacher=js(INPUT/'result/analysis/output/teacher_errors.json')
    teacher.sort(key=lambda x:(x['point'],x['epoch']))
    compare(teacher,sorted(original_teacher['rows'],key=lambda x:(x['point'],x['epoch'])),'teacher_scalar_means')
    logical={}
    for a in ARMS:
        counter=Counter()
        for o in ORDERS:
            for s in (1,2,3):
                for role,domains in all_result[point(o,a,s)].items():
                    for d,value in domains.items():
                        for m,label in value.get('interpretation',{}).items():counter[role+'/'+label]+=1
        logical[a]=dict(sorted(counter.items()))
    summary=js(INPUT/'result/analysis/output/summary.json')
    compare(dict(sorted(counts.items())),summary['physical_diagnostic_counts'],'physical_diagnostic_counts')
    compare(logical,summary['logical_diagnostic_counts'],'logical_diagnostic_counts')
    save('recomputed_diagnostics.json',{'scope':'Exact frozen six-epoch arrays and original resampling reproduced in web workspace','points':all_result,'teacher':teacher})
    result={'contexts':dict(contexts),'physical_metric_decisions':sum(counts.values()),'physical_counts':dict(sorted(counts.items())),
            'logical_counts':logical,'N_A_empty_domain_contexts':missing,'teacher_contexts':len(teacher),
            'maximum_residual_identity_absolute_error':scalar_identity_max,
            'numeric_comparison':dict(STATS),'diagnostic_raw_numeric_values':216*48*3+180*6*3+216*60*3}
    save('diagnostics_audit.json',result);print('DIAGNOSTICS',json.dumps(result,ensure_ascii=False),flush=True)

def historical_audit():
    col,mats=loaded_matrices();saved=js(INPUT/'result/job/evaluation/evaluation.json')
    outer=zipfile.ZipFile(io.BytesIO(payload(INPUT/'background/historical_result_input.zip','nested_archive_identity_and_selected_members')))
    inner_raw=outer.read('background/original_result_input.zip')
    expect(len(inner_raw)==9614123 and hashlib.sha256(inner_raw).hexdigest()=='dbed856f06482e29b444ce45b5b44973083bbd28964516ae2cb5ccf5c8be3c38','registered_original_result_package')
    z=zipfile.ZipFile(io.BytesIO(inner_raw));hist_access=[]
    def read(name,rec=None):
        b=z.read(name);h=hashlib.sha256(b).hexdigest();hist_access.append({'path':name,'bytes':len(b),'sha256':h})
        if rec:expect(len(b)==rec['bytes'] and h==rec['sha256'],'historical_link_identity:'+name)
        return b
    oldcol=json.loads(read('result/job/evaluation/collected.json'));pol=json.loads(read('result/source/schema/step28_record_replay_policy.json'))
    refs={n:json.loads(read('reference/'+n,rec)) for n,rec in pol['reference']['collections'].items()}
    for registry in (oldcol,*refs.values()):
        expect(all(registry[k]==col[k] for k in ('group_ids','domains','metric_columns')),'historical_group_pairing')
    results={};older={};max_maps={};endpoint_drift={}
    for a in ('C','S','LOGIT0.1'):
        results[a]={'role_matrices':0,'array_equal':0,'bitwise_equal':0,'maximum_absolute_any_metric':0.,'maximum_absolute_MAP':0.}
        older[a]={}
        for o in ORDERS:
            for s in (1,2,3):
                folder,reg='result/job/evaluation',oldcol;n=o+'_shared' if s==1 else f'{o}_{a}_stage{s}'
                if a=='LOGIT0.1':
                    folder='reference/reference' if s==1 else 'reference';reg=refs['reference/collected.json' if s==1 else 'collected.json'];n=o+'_shared' if s==1 else f'{o}_logit_tenth_stage{s}'
                name=point(o,a,s);older[a][name]={}
                for role in ROLES[:3]:
                    rec=reg['points'][n][role]['matrix'];before=np.load(io.BytesIO(read(folder+'/'+rec['path'],rec)),allow_pickle=False);after=mats[name][role]
                    expect(before.shape==(60,22) and before.dtype==np.float64 and np.isfinite(before).all(),'historical_matrix_schema')
                    older[a][name][role]=before
                    delta=after-before;res=results[a];res['role_matrices']+=1;res['array_equal']+=int(np.array_equal(before,after));res['bitwise_equal']+=int(before.tobytes()==after.tobytes());res['maximum_absolute_any_metric']=max(res['maximum_absolute_any_metric'],float(np.abs(delta).max()));res['maximum_absolute_MAP']=max(res['maximum_absolute_MAP'],float(np.abs(delta[:,12]).max()))
                    item={'bitwise_equal':bool(np.array_equal(before,after)),'maximum_absolute_group_difference':float(np.abs(delta).max()),
                          'mean_difference_by_actual_domain':{d:dict(zip(METRICS,delta[np.asarray(col['domains'])==d].mean(0).tolist())) for d in 'ABC'}}
                    compare(item,saved['historical_replay']['points'][f'{o}_{a}_stage{s}'][role],'registered_historical_comparison',f'{o}_{a}_stage{s}/{role}')
        endpoint_drift[a]={}
        for role in ROLES:
            endpoint_drift[a][role]={}
            for ep in TERMS:
                old=make_field(older[a],col,a,role,ep).mean(0).mean(1).sum(0)
                new=make_field(mats,col,a,role,ep).mean(0).mean(1).sum(0)
                endpoint_drift[a][role][ep]={m:{'old':float(old[i]),'new':float(new[i]),'new_minus_old':float(new[i]-old[i])} for i,m in enumerate(METRICS)}
    hr=INPUT/'result/analysis/output/historical';manifest=js(hr/'manifest.json');current=js(INPUT/'result/job/run/manifest.json')
    expect(manifest['initial']['model_state_sha256']==current['initial_model_digests']['LOGIT0.1'],'historical_initial_digest_match')
    expect(manifest['public_inputs']==current['public_inputs'] and manifest['partition']==current['partition'] and manifest['pretrained_archive']==current['pretrained_archive'],'historical_declared_inputs_match')
    entryrec=next(x for x in manifest['source_files'] if x['path']=='scripts/step28_bge_continual_run.py')
    b=payload(hr/'step28_bge_continual_run.py','historical_entry_source_identity');expect(len(b)==entryrec['bytes'] and hashlib.sha256(b).hexdigest()==entryrec['sha256'],'historical_entry_bound_by_manifest')
    updates={}
    for o in ORDERS:
        old=js(hr/(o+'_shared.json'));new=js(INPUT/'result/job/run/updates'/(o+'_LOGIT0.1_stage1.json'))
        raw=payload(hr/(o+'_shared.json'));rec=manifest['training'][o+'_shared'];expect(len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['sha256'],'historical_update_metadata_binding:'+o)
        vals=arr(hr/(o+'_shared.npy'),(288,len(old['update_columns'])),'float64')
        raw=payload(hr/(o+'_shared.npy'));rec=old['update_file'];expect(len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['sha256'],'historical_update_array_binding:'+o)
        expect(old['current_ids']==new['current_ids'],'historical_first_domain_schedule:'+o)
        expect(old['current_dropout_stream']==seed(20260918,o,1,'current'),'historical_dropout_seed_derivation:'+o)
        diff=[]
        for step,row in enumerate(new['rows'],1):
            fields={k:{'old':float(vals[step-1,i]),'new':row[k]} for i,k in enumerate(old['update_columns']) if k!='logit_term_host_seconds' and vals[step-1,i]!=row[k]}
            if fields:diff.append({'step':step,'fields':fields})
        updates[o]={'current_schedule_equal':True,'steps_with_any_non_timing_difference':len(diff),'first_differing_step':diff[0] if diff else None,'diagnostic_first_occurs_after_step':48,'all_differences':diff}
        expect(diff[0]['step']==1 and set(diff[0]['fields'])=={'gradient_norm'},'pre_diagnostic_first_gradient_difference:'+o)
    save('historical_matrix_drift.json',{'matrices':results,'endpoint_descriptive_drift':endpoint_drift,'scope':'Existing fixed roles/endpoints; arithmetic differences only, no new statistical selection or confidence interval'})
    save('historical_update_drift.json',updates)
    save('historical_inputs.json',{'outer_file':'input/background/historical_result_input.zip','inner_member':'background/original_result_input.zip','inner_bytes':len(inner_raw),'inner_sha256':hashlib.sha256(inner_raw).hexdigest(),'inner_members':len(z.infolist()),'selected_members':hist_access})
    print('HISTORY',json.dumps({'matrices':results,'first_updates':{o:{k:v for k,v in x.items() if k!='all_differences'} for o,x in updates.items()},'LOGIT_primary_MAP_endpoint_drift':{ep:v['map'] for ep,v in endpoint_drift['LOGIT0.1']['primary'].items()},'numeric_comparison':dict(STATS)},ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['execution','metrics','diagnostics','historical']);args=ap.parse_args()
    begin=time.monotonic();started=datetime.now(timezone.utc).isoformat()
    fn={'execution':execution_audit,'metrics':metrics_audit,'diagnostics':diagnostics_audit,'historical':historical_audit}[args.phase]
    status='PASS';error=None
    try:fn()
    except BaseException:
        status='FAIL';error=traceback.format_exc();print(error,flush=True)
    result={'phase':args.phase,'status':status,'started_at_UTC':started,'elapsed_seconds':time.monotonic()-begin,
            'environment':'ChatGPT web review workspace; NOT project Linux','python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),
            'project_imports':False,'project_server_connection':False,'training':False,'weights_or_memory_payloads':False,'formal_text_or_labels':False,
            'check_counts':dict(CHECKS),'numeric_comparisons':dict(STATS),'error':error,'input_files':list(SEEN.values())}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    save(f'{args.phase}_run_{stamp}.json',result)
    print('RUN_RESULT',json.dumps({k:v for k,v in result.items() if k!='input_files'},ensure_ascii=False),flush=True)
    sys.exit(0 if status=='PASS' else 1)

if __name__=='__main__':main()
