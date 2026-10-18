#!/usr/bin/env python3
"""Independent, saved-evidence-only diagnostic/statistical review.

Does not import project modules or its supplied audit script; never reads models,
Memory, formal text or original labels. Fixed resampling is reproduced with
frequency matrices and manual linear empirical quantiles.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
import platform
import resource
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np


P = argparse.ArgumentParser()
P.add_argument('--workspace', type=Path, required=True)
P.add_argument('--output', type=Path, required=True)
args = P.parse_args()
W, O = args.workspace.resolve(), args.output.resolve()
O.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()
reads = {}
checks = []
max_error = 0.0
compared_numbers = 0


def encoded(v):
    return (json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def out(name, v):
    (O/name).write_bytes(encoded(v))


def payload(p, mode):
    p = p.resolve()
    assert p.is_relative_to(W), p
    b = p.read_bytes()
    k = p.relative_to(W).as_posix()
    reads.setdefault(k, {'path': 'workspace/'+k, 'bytes':len(b),
                        'sha256':hashlib.sha256(b).hexdigest(), 'modes':[]})
    if mode not in reads[k]['modes']:
        reads[k]['modes'].append(mode)
    return b


def jread(p):
    return json.loads(payload(p, 'full_json_parsed'))


def nread(p):
    a = np.load(io.BytesIO(payload(p, 'full_array_loaded_allow_pickle_false')), allow_pickle=False)
    return a


def record_path(root, rec):
    p = root/rec['path']
    b = payload(p, 'referenced_size_sha256_verified')
    assert len(b) == rec['bytes']
    assert hashlib.sha256(b).hexdigest() == rec['sha256']
    return p


def eq(actual, expected, label, tolerance=2e-12):
    global max_error, compared_numbers
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b)), label
    assert not np.isinf(a).any() and not np.isinf(b).any(), label
    mask = np.isfinite(a)
    err = float(np.max(np.abs(a[mask]-b[mask]))) if mask.any() else 0.0
    max_error = max(max_error, err)
    compared_numbers += int(a.size)
    assert err <= tolerance, (label, err)


def freqs(draws, n):
    f = np.zeros((len(draws), n), dtype=np.int64)
    np.add.at(f, (np.repeat(np.arange(len(draws)), draws.shape[1]), draws.ravel()), 1)
    assert np.all(f.sum(1) == draws.shape[1])
    return f


def quantiles(a):
    # Numpy's default linear percentile rule implemented directly.
    s = np.sort(a, axis=0)
    values = []
    for p in (.025, .975):
        at = (len(s)-1)*p
        low, high = math.floor(at), math.ceil(at)
        frac = at-low
        values.append(s[low]*(1-frac)+s[high]*frac)
    return np.stack(values, axis=-1)


def stats(v, f):
    if np.isnan(v).any():
        return {'mean':None, 'conditional_95pct_interval':[None,None]}
    boot = np.einsum('rg,g->r', f, v)/v.size
    return {'mean':float(np.mean(v)), 'conditional_95pct_interval':quantiles(boot).tolist()}


def compare_stat(value, target, label):
    eq(value['mean'], target['mean'], label+'/mean')
    eq(value['conditional_95pct_interval'],target['conditional_95pct_interval'],label+'/ci')


job = W/'reports/job'
evaluation = jread(job/'evaluation/evaluation.json')
collected = jread(job/'evaluation/collected.json')
manifest = jread(job/'run/manifest.json')
partition = jread(job/'run/partition.json')
overfit = jread(job/'diagnostics_evaluation/overfitting.json')
baseline = jread(W/'reports/documentation/20261016/h_consistency/baseline_sources.json')
columns = collected['columns']
assert len(columns) == 22
domains = np.asarray(collected['domains'])
rows = {d:np.flatnonzero(domains == d) for d in 'AB'}
assert all(len(v) == 20 for v in rows.values())
assert collected['group_ids'] == [r['group_uid'] for r in partition['development']]
assert collected['domains'] == [r['domain'] for r in partition['development']]
roles = ['raw','stage-cal','first-cal']
points = ['A_S_pos','B_S_pos','A_H_pos','B_H_pos']
origins = {p:job/'evaluation' for p in points}
point_records = dict(collected['points'])
for key, arm in [('S','S'),('H','G_half')]:
    root = W/baseline['baselines'][key]['root']/'evaluation'
    c = jread(root/'collected.json')
    for k in ['group_ids','domains','columns']:
        assert c[k] == collected[k], (key,k)
    for stage in 'AB':
        point = stage+'_'+arm
        origins[point] = root
        point_records[point] = c['points'][point]

matrices, queries, counts, classification = {}, {}, {}, {}
rank_indices = [i for i,k in enumerate(columns) if k not in
    ['brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc']]
for point, records in point_records.items():
    matrices[point], queries[point], counts[point], classification[point] = {}, {}, {}, {}
    for role in roles:
        rec = records[role]
        m = jread(record_path(origins[point],rec['matrix']))
        q = nread(record_path(origins[point],rec['queries']))
        c = jread(record_path(origins[point],rec['counts']))
        a = np.asarray(m['values'],dtype=np.float64)
        assert m['columns']==columns and a.shape==(40,22) and q.shape==(40,28,10)
        eq(q.mean(1), a[:,12:], point+'/'+role+'/query_to_group')
        assert len(c)==40
        for i,x in enumerate(c):
            tp,fp,fn,tn = [x[k] for k in ['tp','fp','fn','tn']]
            assert tp+fn==20 and fp+tn==358
            denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            calc = {
                'precision':tp/(tp+fp) if tp+fp else None,
                'recall':tp/(tp+fn),
                'f1':2*tp/(2*tp+fp+fn),
                'specificity':tn/(tn+fp),
                'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2,
                'mcc':(tp*tn-fp*fn)/denominator if denominator else None}
            for k,v in calc.items():
                eq(v,a[i,columns.index(k)],point+'/'+role+'/'+str(i)+'/'+k)
        matrices[point][role], queries[point][role], counts[point][role] = a,q,c
        classification[point][role]={}
        for domain, r in rows.items():
            t = np.asarray([[c[i][k] for k in ['tp','fp','fn','tn']] for i in r]).sum(0)
            tp,fp,fn,tn = map(int,t)
            classification[point][role][domain] = {'counts':t.tolist(),
                'pooled_precision':tp/(tp+fp) if tp+fp else None,
                'pooled_recall':tp/(tp+fn), 'pooled_f1':2*tp/(2*tp+fp+fn),
                'pooled_fpr':fp/(fp+tn),
                'undefined_macro_columns':[k for k in columns if np.isnan(a[r,columns.index(k)]).any()]}
    for role in roles[1:]:
        eq(matrices[point][role][:,rank_indices],matrices[point]['raw'][:,rank_indices],point+'/'+role+'/ranking_invariance',0)
        eq(queries[point][role],queries[point]['raw'],point+'/'+role+'/query_invariance',0)

draws = np.random.Generator(np.random.PCG64(20260930)).integers(20,size=(5000,2,20))
f = {d:freqs(draws[:,i],20) for i,d in enumerate('AB')}
contrasts = [('S_pos','S'),('H_pos','S'),('H_pos','G_half'),('H_pos','S_pos')]
final, initial, old_change, forgetting = {}, {}, {}, {}
for c,o in contrasts:
    key = c+'-'+o
    final[key],initial[key],forgetting[key] = {}, {}, {}
    for role in roles:
        final[key][role],initial[key][role],forgetting[key][role] = {}, {}, {}
        for d,r in rows.items():
            final[key][role][d],initial[key][role][d] = {}, {}
            for k in ['map','brier','log_loss']:
                i=columns.index(k)
                delta=matrices['B_'+c][role][r,i]-matrices['B_'+o][role][r,i]
                z=stats(delta,f[d])
                compare_stat(z,evaluation['comparisons'][key][role][d][k],key+'/'+role+'/'+d+'/'+k)
                final[key][role][d][k]=z
                delta=matrices['A_'+c][role][r,i]-matrices['A_'+o][role][r,i]
                z=stats(delta,f[d])
                compare_stat(z,evaluation['initial_comparisons'][key][role][d][k],'initial/'+key+'/'+role+'/'+d+'/'+k)
                initial[key][role][d][k]=z
        for k in ['map','brier','log_loss']:
            i=columns.index(k)
            r=rows['A']
            sign=-1 if k in ['brier','log_loss'] else 1
            delta=((matrices['A_'+c][role][r,i]-matrices['B_'+c][role][r,i])-
                   (matrices['A_'+o][role][r,i]-matrices['B_'+o][role][r,i]))*sign
            z=stats(delta,f['A'])
            compare_stat(z,evaluation['forgetting_comparisons'][key][role][k],'forgetting/'+key+'/'+role+'/'+k)
            forgetting[key][role][k]=z
for arm in ['S','G_half','S_pos','H_pos']:
    old_change[arm]={}
    for role in roles:
        old_change[arm][role]={}
        for k in ['map','brier','log_loss']:
            i=columns.index(k);r=rows['A']
            z=stats(matrices['B_'+arm][role][r,i]-matrices['A_'+arm][role][r,i],f['A'])
            compare_stat(z,evaluation['old_changes'][arm][role][k],'old_change/'+arm+'/'+role+'/'+k)
            old_change[arm][role][k]=z

degrees=nread(record_path(job/'evaluation',collected['query_positive_counts']))
assert degrees.shape==(40,28) and np.all((degrees==1).sum(1)==16) and np.all((degrees==2).sum(1)==12)
subgroups={}
for point,q in queries.items():
    subgroups[point]={}
    for count in [1,2]:
        subgroups[point][count]=np.asarray([q['raw'][g,degrees[g]==count,0].mean() for g in range(40)])
    eq((16*subgroups[point][1]+12*subgroups[point][2])/28,matrices[point]['raw'][:,columns.index('map')],point+'/16_12_weighted_MAP')
subgroup_contrasts={}
for c,o in contrasts:
    key=c+'-'+o
    subgroup_contrasts[key]={}
    for count in [1,2]:
        subgroup_contrasts[key][count]={}
        for d,r in rows.items():
            z=stats((subgroups['B_'+c][count]-subgroups['B_'+o][count])[r],f[d])
            compare_stat(z,evaluation['query_subgroup_MAP_comparisons'][key][str(count)][d],'subgroup/'+key+'/'+str(count)+'/'+d)
            subgroup_contrasts[key][count][d]=z

calibration=[]
for point in points:
    pr=jread(job/'run/points'/f'{point}.json')
    mp=jread(record_path(job/'run',pr['mapping']))
    raw=nread(record_path(job/'run',pr['scores']['raw']))
    eq(raw,nread(job/'run/diagnostics'/f'{point}_epoch6_valid.npy'),point+'/endpoint_to_last_blind',0)
    for role,mapkey in [('stage-cal','stage'),('first-cal','first')]:
        a,b=pr['maps'][mapkey]['a'],pr['maps'][mapkey]['b']
        assert a>0 and math.isfinite(a) and math.isfinite(b)
        transformed=nread(record_path(job/'run',pr['scores'][role]))
        eq(a*raw.astype(np.float64)+b,transformed,point+'/'+role+'/transform',0)
        assert np.array_equal(np.argsort(raw,axis=1,kind='stable'),np.argsort(transformed,axis=1,kind='stable'))
        ix=np.argsort(raw,axis=1,kind='stable')
        x,z=np.take_along_axis(raw,ix,1),np.take_along_axis(transformed,ix,1)
        assert np.array_equal(x[:,1:]==x[:,:-1],z[:,1:]==z[:,:-1])
        calibration.append({'point':point,'role':role,'a':a,'b':b,'exact_affine_score_match':True,'exact_order_and_ties_preserved':True})
    for k in ['a','b']:
        eq(mp[k],pr['maps']['stage'][k],point+'/mapping_parameter/'+k,0)
    assert len(pr['calibration_ids'])==12
    if point.startswith('A_'):
        assert pr['maps']['first']==pr['maps']['stage']
    else:
        earlier=jread(job/'run/points'/('A_'+point[2:]+'.json'))
        assert pr['maps']['first']==earlier['maps']['first']

diagnostics=[]
valid_curves={}
residual_checks=0
capture_count=0
signs=np.array([1.,1.,-1.])
for point in points:
    captures=[];curves={k:[] for k in ['fit','cache','valid']}
    stage=1 if point.startswith('A_') else 2
    for ordinal,rec in enumerate(manifest['diagnostics'][point],1):
        d=jread(record_path(job/'run/diagnostics',rec));captures.append(d);capture_count+=1
        assert d['epoch']==ordinal and d['stage']==stage and d['stage_step']==48*ordinal
        assert d['adam_step']==(stage-1)*288+48*ordinal and d['point']==point
        assert d['valid_labels_used'] is False and d['mode']=='A0_eval_no_autograd'
        assert d['valid_ids']==collected['group_ids']
        assert len(d['fit_ids'])==48 and len(d['cache_ids'])==(0 if stage==1 else 6)
        assert d['metric_columns']==['map','recall_at_5','log_loss']
        for key in ['fit_ids','cache_ids','fit_domains','cache_domains','valid_ids']:
            assert d[key]==captures[0][key],(point,ordinal,key)
        for role in ['fit','cache']:
            a=nread(record_path(job/'run/diagnostics',d[role]));assert a.shape==(len(d[role+'_ids']),3)
            assert np.isfinite(a).all();curves[role].append(a)
        v=nread(record_path(job/'diagnostics_evaluation',overfit['group_metrics'][f'{point}_epoch{ordinal}']))
        assert v.shape==(40,3) and np.isfinite(v).all();curves['valid'].append(v)
        blind=nread(record_path(job/'run/diagnostics',d['valid_blind']))
        assert blind.shape==(40,378) and blind.dtype==np.float32 and np.isfinite(blind).all()
        assert len(d['teacher_errors'])==len(d['cache_ids'])
        for e in d['teacher_errors']:
            assert e['D0']>=0 and e['D1']>=0 and e['residual_difference_mse']>=0
            eq(e['residual_difference_mse'],e['D0']+e['D1']-2*e['residual_cross_mean'],point+'/residual_identity')
            residual_checks+=1
    assert len(captures)==6
    curves={k:np.stack(v) for k,v in curves.items()}
    eq(curves['valid'][-1],matrices[point]['raw'][:,[columns.index(k) for k in ['map','recall_at_5','log_loss']]],point+'/last_diagnostic_to_endpoint_metrics',0)
    valid_curves[point]={d:curves['valid'][:,r].mean(1).tolist() for d,r in rows.items()}
    for role,domain in [('fit','A' if stage==1 else 'B')]+([] if stage==1 else [('cache','A')]):
        x=curves[role];v=curves['valid'][:,rows[domain]];n=x.shape[1]
        seed_data=(json.dumps([20260930,'ABC',stage,role,domain,'diagnostics'],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        seed=int.from_bytes(hashlib.sha256(seed_data).digest()[:8],'big')%(2**63-1)
        rng=np.random.Generator(np.random.PCG64(seed))
        ft=freqs(rng.integers(n,size=(5000,n)),n);fv=freqs(rng.integers(20,size=(5000,20)),20)
        dt=(x[-1]-x[0])*signs;dv=(v[-1]-v[0])*signs
        bt=np.einsum('rg,gk->rk',ft,dt)/n;bv=np.einsum('rg,gk->rk',fv,dv)/20
        g=(x.mean(1)-v.mean(1))*signs
        boots=(np.einsum('rg,egk->erk',ft,x)/n-np.einsum('rg,egk->erk',fv,v)/20)*signs
        tc,vc,gc=quantiles(bt),quantiles(bv),quantiles(bt-bv)
        target=overfit['points'][point][role][domain]
        eq(x.mean(1),target['train_raw_curve'],point+'/'+role+'/train_curve')
        eq(v.mean(1),target['valid_raw_curve'],point+'/'+role+'/valid_curve')
        eq(g,target['benefit_gap_curve'],point+'/'+role+'/gap_curve')
        eq(np.stack([quantiles(b) for b in boots]),target['benefit_gap_conditional_intervals'],point+'/'+role+'/gap_curve_intervals')
        for a,name,ci in [(dt,'train',tc),(dv,'valid',vc)]:
            eq(a.mean(0),target['epoch1_to_6_'+name+'_benefit']['mean'],point+'/'+role+'/'+name+'_benefit')
            eq(ci,target['epoch1_to_6_'+name+'_benefit']['conditional_95pct_interval'],point+'/'+role+'/'+name+'_ci')
        eq(dt.mean(0)-dv.mean(0),target['epoch1_to_6_gap_widening']['mean'],point+'/'+role+'/gap_mean')
        eq(gc,target['epoch1_to_6_gap_widening']['conditional_95pct_interval'],point+'/'+role+'/gap_ci')
        for i,k in enumerate(['map','recall_at_5','log_loss']):
            tmean,vmean=float(dt[:,i].mean()),float(dv[:,i].mean())
            widen=tmean-vmean
            strong=bool(tc[i,0]>0 and vc[i,1]<0 and gc[i,0]>0)
            point_signal=tmean>0 and vmean<0 and widen>0
            decision='有相应迹象' if strong else '证据不足' if point_signal else '未见明确迹象'
            assert decision==target['interpretation'][k]
            diagnostics.append({'point':point,'role':role,'domain':domain,'metric':k,
                'train_groups':n,'valid_groups':20,'bootstrap_seed':seed,
                'train_benefit':tmean,'train_ci':tc[i].tolist(),
                'valid_benefit':vmean,'valid_ci':vc[i].tolist(),
                'gap_widening':widen,'gap_ci':gc[i].tolist(),
                'decision':decision,'gap_epoch1':float(g[0,i]),'gap_epoch6':float(g[-1,i])})
assert capture_count==24 and len(diagnostics)==18
decision_counts=dict(Counter(d['decision'] for d in diagnostics))
assert decision_counts=={'未见明确迹象':17,'证据不足':1}

result={'status':'SAVED_DIAGNOSTICS_AND_TARGETED_STATISTICS_INDEPENDENTLY_REPRODUCED',
    'formal_labels_read':False,'project_server_connected':False,'model_or_memory_loaded':False,
    'project_modules_imported':False,'supplied_audit_executed':False,
    'capture_count':capture_count,'strict_decision_count':len(diagnostics),'decision_counts':decision_counts,
    'all_18_gap_lower_bounds_positive':all(d['gap_ci'][0]>0 for d in diagnostics),
    'diagnostic_rows':diagnostics,'valid_curves':valid_curves,
    'final_comparisons':final,'initial_comparisons':initial,'old_change':old_change,
    'forgetting_comparisons':forgetting,'subgroup_contrasts':subgroup_contrasts,
    'classification':classification,'calibration_checks':calibration,
    'query_positive_geometry':{'groups':40,'per_group_single_positive_queries':16,'per_group_two_positive_queries':12,'per_domain_single_positive_queries':320,'per_domain_two_positive_queries':240},
    'residual_identity_checks':residual_checks,
    'numerical_values_compared':compared_numbers,'maximum_absolute_discrepancy':max_error,
    'bootstrap':{'main':'PCG64(20260930), one (5000,2,20) array','diagnostics':'seed_for payload preserved; same trajectories across epochs; independent fit/cache vs valid resampling; arm excluded from seed','recompute':'frequency matrix/einsum plus manual linear empirical quantiles'},
    'limitations':['AP/ROC/Brier/log-loss original values are saved label-derived inputs, not independently rebuilt from original labels.',
                  'Diagnostic epoch 1 is after 48 stage updates; epoch 1-to-6 does not cover the first epoch or pretraining-to-stage entry.',
                  'Cache comparisons use six surviving training groups, not the complete historical fit domain.',
                  'Conditional group intervals do not include training seed or repeated development selection uncertainty.',
                  'No causal effect on unseen tasks or exclusive subgroup interaction claim follows from these fixed contrasts.']}
out('independent_results.json',result)
out('read_scope.json',{'kind':'actual_files_fully_parsed_or_loaded_by_this_script','count':len(reads),'files':list(reads.values())})
elapsed=time.perf_counter()-start
out('runtime.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
    'argv':sys.argv,'elapsed_seconds':elapsed,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES']},
    'not_project_native_execution':True})
print(json.dumps({'status':result['status'],'captures':capture_count,'strict_decisions':len(diagnostics),
    'decision_counts':decision_counts,'all_18_gap_lower_bounds_positive':result['all_18_gap_lower_bounds_positive'],
    'saved_values_compared':compared_numbers,'max_absolute_discrepancy':max_error,'actual_input_files':len(reads),
    'elapsed_seconds':elapsed,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},ensure_ascii=False,indent=2))

