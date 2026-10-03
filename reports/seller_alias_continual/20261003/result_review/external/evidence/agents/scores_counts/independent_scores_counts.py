#!/usr/bin/env python3
"""Independent small-result audit. Never loads formal labels, text or weights.

Class rates are independently computed from the permitted saved confusion counts.
Calibration/log-loss identities condition on saved truth-derived NLL scalars and
class totals; no per-pair truth is reconstructed or inferred.
"""
from pathlib import Path
from collections import Counter
import ast
import hashlib
import io
import json
import math
import os
import platform
import sys
import numpy as np

P = Path('/workspace/scratch/ca13b63db3f0/review_work/input/project')
OUT = Path(__file__).resolve().parent
ROOTS = {
    'low': P/'reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job',
    'logit': P/'reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/reports/job',
    'base': P/'reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job',
    'weight': P/'reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job',
}
ORDERS = ('ABC','BCA','CAB')
ROLES = ('raw','stage-cal','first-cal')
COLS = ('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct',
        'brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc',
        'map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10',
        'ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')
RANK = [0,1,2,3,*range(12,22)]
coverage = {}
failures = []
counters = Counter()
maxerr = Counter()

def require(ok, description):
    if not bool(ok):
        failures.append(description)

def blob(path, purpose, record=None):
    path = path.resolve()
    # Deliberately constrain actual content reads to permitted artifact types.
    if path.suffix not in ('.json','.npy','.py','.md') or not path.is_relative_to(P):
        raise ValueError('Read outside declared small-result/source scope: '+str(path))
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rec = coverage.setdefault(str(path.relative_to(P)),
        {'path':str(path.relative_to(P)),'bytes':len(raw),'sha256':digest,'purposes':[]})
    if purpose not in rec['purposes']:
        rec['purposes'].append(purpose)
    if record is not None:
        require(record['bytes'] == len(raw) and record['sha256']==digest,
                'Record mismatch: '+str(path.relative_to(P)))
        counters['records_hash_and_size_checked'] += 1
    return raw

def js(path, purpose='small record', record=None):
    return json.loads(blob(path,purpose,record))

def arr(path, shape, dtype, record=None):
    a = np.load(io.BytesIO(blob(path,'saved numeric array',record)),allow_pickle=False)
    require(a.shape==shape and a.dtype==dtype and np.isfinite(a).all(),
            'Bad shape/dtype/finiteness: '+str(path))
    return a

def err(label, observed, expected, tolerance=2e-12):
    a=np.asarray(observed,dtype=np.float64)
    b=np.asarray(expected,dtype=np.float64)
    delta=float(np.max(np.abs(a-b))) if a.size else 0.0
    maxerr[label]=max(maxerr[label],delta)
    require(delta<=tolerance,f'{label}: {delta} exceeds {tolerance}')
    return delta

def softplus(x):
    x=np.asarray(x,dtype=np.float64)
    return np.maximum(x,0)+np.log1p(np.exp(-np.abs(x)))

def sigmoid(x):
    x=np.asarray(x,dtype=np.float64)
    e=np.exp(-np.abs(x))
    return np.where(x>=0,1/(1+e),e/(1+e))

def class_rates(c):
    tp,fp,fn,tn=(int(c[k]) for k in ('tp','fp','fn','tn'))
    sens=tp/(tp+fn) if tp+fn else 0.0
    spec=tn/(tn+fp) if tn+fp else 0.0
    divisor=math.prod((tp+fp,tp+fn,tn+fp,tn+fn))
    return {'precision':tp/(tp+fp) if tp+fp else 0.0,'recall':sens,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            'specificity':spec,'balanced_accuracy':(sens+spec)/2,
            'mcc':(tp*tn-fp*fn)/math.sqrt(divisor) if divisor else 0.0}

def pooled(counts):
    c={k:sum(int(row[k]) for row in counts) for k in ('tp','fp','fn','tn')}
    r=class_rates(c)
    return {**c,'fpr':c['fp']/(c['fp']+c['tn']),
            **{k:r[k] for k in ('recall','precision','f1')}}

def check_collection(root,c,evaluation):
    require(c['metric_columns']==list(COLS),'22-column schema differs')
    require(len(c['group_ids'])==60 and len(set(c['group_ids']))==60,'Group identities not 60 unique')
    require(Counter(c['domains'])=={'A':20,'B':20,'C':20},'Actual-domain group counts differ')
    matrices,counts={},{}
    for name,roles in c['points'].items():
        require(set(roles)==set(ROLES),'Incomplete metric roles: '+name)
        matrices[name],counts[name]={},{}
        for role,recs in roles.items():
            m=arr(root/recs['matrix']['path'],(60,22),np.float64,recs['matrix'])
            cc=js(root/recs['counts']['path'],'saved aggregate group confusion',recs['counts'])
            require(len(cc)==60,'Wrong confusion rows: '+name)
            for i,row in enumerate(cc):
                require(all(type(row[k]) is int and row[k]>=0 for k in ('tp','fp','fn','tn')),
                        'Invalid confusion integer: '+name)
                require(row['tp']+row['fn']==20 and row['fp']+row['tn']==358,'Bad class support: '+name)
                rates=class_rates(row)
                for key,value in rates.items():
                    err('group_classification_rates',m[i,COLS.index(key)],value)
            bounded=[j for j in range(22) if COLS[j] not in ('log_loss','mcc')]
            require(np.all((m[:,bounded]>=0)&(m[:,bounded]<=1)),'Metric range differs: '+name)
            require(np.all((m[:,11]>=-1)&(m[:,11]<=1)) and np.all(m[:,5]>=0),'MCC/log-loss range differs')
            saved=evaluation['absolute_stage_results'][name][role]
            # Sum counts first, independently derive pooled rates; compare all reported cells.
            pool=saved['pooled_fixed_half_classification']
            require(pool['threshold']==0.0,'Fixed probability .5 / logit 0 changed')
            for group_label,indices in [('pooled',list(range(60))),
                                        *[(d,[i for i,x in enumerate(c['domains']) if x==d]) for d in 'ABC']]:
                wanted=pooled([cc[i] for i in indices])
                got=pool['pooled'] if group_label=='pooled' else pool['by_domain'][group_label]
                for key,value in wanted.items():
                    err('pooled_classification',got[key],value)
                require(wanted['tp']+wanted['fn']==20*len(indices),'Pooled positive class total differs')
            # Fsum makes accumulation independent from NumPy mean used by producer.
            for j,key in enumerate(COLS):
                err('macro_metrics',saved['macro_all'][key],math.fsum(m[:,j].tolist())/60)
                for d in 'ABC':
                    rows=[i for i,x in enumerate(c['domains']) if x==d]
                    err('macro_metrics',saved['macro_by_domain'][d][key],math.fsum(m[rows,j].tolist())/20)
            matrices[name][role]=m
            counts[name][role]=cc
            counters['metric_count_sets_including_study_reuse']+=1
            counters['group_confusion_rows_including_study_reuse']+=60
        for role in ROLES[1:]:
            require(np.array_equal(matrices[name][role][:,RANK],matrices[name]['raw'][:,RANK]),
                    'Affine calibration changed saved rank/curve columns: '+name+'/'+role)
    return matrices,counts

def conditional_calibration(raw,maprec):
    # Only two saved truth-derived aggregate scalars: initial NLL and positive count.
    # They determine the sufficient moments for this two-parameter affine NLL.
    x=raw.astype(np.float64)
    mu_y=maprec['positive_count']/x.size
    mu_yx=math.fsum(softplus(x).ravel().tolist())/x.size-maprec['initial_nll']
    traj=[]
    for t in [*maprec['trajectory'],
              {'a':maprec['a'],'b':maprec['b'],'nll':maprec['final_nll'],
               'projected_gradient_max':maprec['projected_gradient_max'],'iteration':'final'}]:
        a,b=t['a'],t['b']
        z=a*x+b
        prob=sigmoid(z)
        loss=math.fsum(softplus(z).ravel().tolist())/x.size-a*mu_yx-b*mu_y
        gradient=[math.fsum((prob*x).ravel().tolist())/x.size-mu_yx,
                  math.fsum(prob.ravel().tolist())/x.size-mu_y]
        projected=[]
        for value,g,(lower,upper) in zip((a,b),gradient,((.001,100),(-100,100))):
            projected.append(0.0 if (value<=lower and g>0) or (value>=upper and g<0) else g)
        pg=max(map(abs,projected))
        err('conditional_calibration_nll',t['nll'],loss)
        err('conditional_calibration_projected_gradient',t['projected_gradient_max'],pg)
        traj.append({'iteration':t['iteration'],'nll':loss,'projected_gradient_max':pg})
    require(traj[-1]['projected_gradient_max']<=1e-6,'Calibration KKT tolerance unmet')
    require(maprec['final_nll']<=maprec['initial_nll']+1e-12,'Calibration objective increased')
    counters['calibration_trajectory_rows_including_final']+=len(traj)
    return {'sufficient_statistic_mean_y':mu_y,'sufficient_statistic_mean_y_times_logit':mu_yx,
            'final_independent_nll':traj[-1]['nll'],'final_independent_projected_gradient':traj[-1]['projected_gradient_max'],
            'trajectory_rows_checked':len(traj),'raw_logit_threshold_for_probability_half':-maprec['b']/maprec['a'],
            'condition':'Original saved initial_nll and positive_count are assumed; no true labels independently available.'}

def main():
    base=js(ROOTS['base']/'evaluation/collected.json','original frozen collection identities')
    weight=js(ROOTS['weight']/'evaluation/collected.json','original frozen weight collection identities')
    base_manifest=js(ROOTS['base']/'run/manifest.json','shared start metadata')
    shared={o:js(ROOTS['base']/'run'/base_manifest['points'][o+'_shared']['path'],'shared start metadata',base_manifest['points'][o+'_shared']) for o in ORDERS}
    all_new={}
    studies={}
    all_refs={}
    calibrations=[]
    for study,arm,reuse_number in [('low','tenth',81),('logit','logit_quarter',45)]:
        root=ROOTS[study]
        c=js(root/'evaluation/collected.json','new collection identities')
        ref=js(root/'evaluation/reference/collected.json','frozen reused collection identities')
        ev=js(root/'evaluation/evaluation.json','published absolute values and metadata')
        manifest=js(root/'run/manifest.json','point and start identities')
        partition=js(root/'run'/manifest['partition']['path'],'public fit/calibration/development partition',manifest['partition'])
        require(c['group_ids']==base['group_ids']==weight['group_ids']==ref['group_ids'],'Group identity misalignment')
        require(c['domains']==base['domains']==weight['domains']==ref['domains'],'Actual-domain row misalignment')
        require(c['metric_columns']==base['metric_columns']==weight['metric_columns']==ref['metric_columns'],'Metric schema misalignment')
        require(len(c['points'])*3==18 and len(ref['points'])*3==reuse_number,'18+81/18+45 count differs')
        require(ev['new_metric_count_sets']==18 and ev['reused_metric_count_sets']==reuse_number,'Published reuse totals differ')
        require(set(c['points'])=={f'{o}_{arm}_stage{s}' for o in ORDERS for s in (2,3)},'New endpoint names differ')
        expected_reference_arms=('seq','er','half','quarter') if study=='low' else ('seq','quarter')
        require(set(ref['points'])=={o+'_shared' for o in ORDERS}|{f'{o}_{a}_stage{s}' for o in ORDERS for a in expected_reference_arms for s in (2,3)},'Reused reference set differs')
        # Original collections are included even when duplicate original matrix bytes are omitted.
        for name,roles in ref['points'].items():
            parent=weight if '_half_' in name or '_quarter_' in name else base
            require(roles==parent['points'][name],'Reuse differs from original descriptor: '+name)
            counters['reused_sets_compared_to_original_descriptors']+=3
        new_m,new_c=check_collection(root/'evaluation',c,ev)
        ref_m,ref_c=check_collection(root/'evaluation/reference',ref,ev)
        all_new[study]=(new_m,new_c)
        all_refs[study]=(ref,ref_m,ref_c)
        stage_points=[]
        for name in c['points']:
            point=js(root/'run'/manifest['points'][name]['path'],'new point metadata',manifest['points'][name])
            o,s=point['order'],point['stage']
            start=manifest['restored_starts'][o+'_'+arm]
            first=shared[o]['first_map_parameters']
            require(start['first_map']==first==point['first_map_parameters'],'First calibration map not frozen')
            require(start['full_checkpoint']==shared[o]['full_checkpoint'],'Shared full checkpoint descriptor differs')
            require(start['model_state_sha256']==shared[o]['model_state_sha256'],'Shared model state digest differs')
            require(start['adam_step']==288 and start['first_scores_replayed_exactly'],'Original start replay evidence differs')
            maps=js(root/'run'/point['mapping']['path'],'current-only calibration full trajectory',point['mapping'])
            require(maps['name']==name and maps['actual_domain']==o[s-1],'Calibration point/domain differs')
            require(maps['calibration_group_ids']==[r['group_uid'] for r in partition['calibration'] if r['domain']==o[s-1]],'Calibration not current-domain IDs')
            require(maps['group_count']==12 and maps['pair_count']==4536 and maps['positive_count']==240,'Calibration totals differ')
            require(maps['score_source']==point['scores']['calibration'] and maps['model_state_sha256']==point['model_state_sha256'],'Calibration model/score binding differs')
            require(maps['optimizer_success'] and maps['optimizer_iterations']==len(maps['trajectory'])<=200 and maps['objective_calls']<=2000,'Calibration solver trace differs')
            require(maps['initial_parameters']==[1.0,0.0] and maps['bounds']==[[.001,100.0],[-100.0,100.0]],'Calibration starting point or bounds differs')
            require(.001<=maps['a']<=100 and -100<=maps['b']<=100,'Calibration parameters outside bounds')
            score_arrays={}
            for role,recs in point['scores'].items():
                shape=(12,378) if role=='calibration' else (60,378)
                dtype=np.float32 if role in ('calibration','development') else np.float64
                score_arrays[role]=arr(root/'run'/recs['path'],shape,dtype,recs)
                counters['new_score_arrays']+=1
            raw=score_arrays['development'].astype(np.float64)
            record={'study':study,'point':name,'raw_logit_range':[float(raw.min()),float(raw.max())],
                    'stage_calibration':conditional_calibration(score_arrays['calibration'],maps),
                    'first_raw_logit_threshold_for_probability_half':-first['b']/first['a']}
            raw_soft=softplus(raw).mean(axis=1)
            raw_loss=new_m[name]['raw'][:,COLS.index('log_loss')]
            # The supplied scores in this packet are safely inside clipping bounds.
            raw_p=sigmoid(raw)
            require(np.all((raw_p>1e-15)&(raw_p<1-1e-15)),'Raw log-loss clipping prevents moment transfer')
            # Per-group conditional moment, not an inferred individual label vector.
            yx=raw_soft-raw_loss
            record['role_predicted_positive_totals']={}
            for role,use_map in [('raw',{'a':1.,'b':0.}),('stage-cal',maps),('first-cal',first)]:
                scored=raw if role=='raw' else score_arrays[role]
                if role!='raw':
                    z=raw*float(use_map['a'])+float(use_map['b'])
                    require(np.array_equal(scored,z),'Saved affine score values differ')
                    err('saved_affine_values',scored,z,0.0)
                    for i in range(60):
                        # Python sorted order is independent of producer's np.argsort path.
                        order=sorted(range(378),key=lambda j:(float(raw[i,j]),j))
                        order_z=sorted(range(378),key=lambda j:(float(scored[i,j]),j))
                        require(order==order_z,'Calibration changed sort order')
                        require(all((raw[i,j]==raw[i,k])==(scored[i,j]==scored[i,k]) for j,k in zip(order[:-1],order[1:])),'Calibration changed ties')
                    pred=sigmoid(scored)
                    require(np.all((pred>1e-15)&(pred<1-1e-15)),'Affine log-loss clipping prevents moment transfer')
                    cond_loss=softplus(scored).mean(axis=1)-use_map['a']*yx-use_map['b']*(20/378)
                    err('conditional_affine_valid_log_loss',new_m[name][role][:,5],cond_loss)
                    counters['conditional_valid_logloss_group_rows']+=60
                predicted=(scored>=0).sum(axis=1)
                recorded=np.array([r['tp']+r['fp'] for r in new_c[name][role]])
                require(np.array_equal(predicted,recorded),'Predicted-positive count differs from scores: '+name+'/'+role)
                require(np.array_equal(scored>=0,sigmoid(scored)>=.5),'Logit and probability thresholds disagree')
                counters['blind_score_predicted_positive_group_rows']+=60
                record['role_predicted_positive_totals'][role]=int(predicted.sum())
            calibrations.append(record)
            stage_points.append(name)
        studies[study]={'new_sets':18,'reused_sets':reuse_number,'points':stage_points,
                        'group_count':60,'per_group_unordered_pairs':378,'per_group_positive_pairs':20,
                        'new_formal_truth_read_for_review':0}
    # The same 45 reference sets appear in both requested studies.
    for name in all_refs['logit'][0]['points']:
        require(all_refs['logit'][0]['points'][name]==all_refs['low'][0]['points'][name],'Cross-study reused identity differs')
        for role in ROLES:
            require(np.array_equal(all_refs['logit'][1][name][role],all_refs['low'][1][name][role]),'Cross-study reused matrix values differ')
            require(all_refs['logit'][2][name][role]==all_refs['low'][2][name][role],'Cross-study reused confusion differs')
            counters['cross_study_same_reference_sets']+=1
    report={'status':'PASS_INDEPENDENT_SAVED_SCORES_COUNTS_CHECKS' if not failures else 'FAIL',
            'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0))},
            'studies':studies,'counts':dict(counters),'maximum_absolute_errors':dict(maxerr),
            'calibrations':calibrations,'failures':failures,
            'unverified':['No formal labels were read; AP/MAP/MRR/Recall@K/NDCG@K/AUC were not independently recomputed from formal labels.',
                          'Confusion TP/FP membership and Brier values were not independently regenerated from formal labels.',
                          'Calibration trajectory identities condition on original saved initial_nll and positive_count; no independent-label fit or gradient verification.',
                          'Affine valid log-loss identity conditions on original saved raw log_loss and group class totals.',
                          'Shared model/Adam/RNG records are audited metadata, not a fresh native weight reload.']}
    (OUT/'numeric_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (OUT/'read_coverage.json').write_text(json.dumps({'purpose':'Computational content reads; source human reading spans are separately listed in notes.md.','files':sorted(coverage.values(),key=lambda r:r['path'])},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','studies','counts','maximum_absolute_errors','failures')},ensure_ascii=False,indent=2))
    if failures:
        raise SystemExit(1)

if __name__=='__main__':
    main()
