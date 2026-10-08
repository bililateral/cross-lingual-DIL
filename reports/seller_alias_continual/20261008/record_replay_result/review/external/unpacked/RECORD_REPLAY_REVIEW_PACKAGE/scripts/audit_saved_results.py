"""Independent review of saved metrics only. Does not import any project module.

Usage: python audit_saved_results.py --input EXTRACTED_ZIP --output NEW_DIRECTORY
Reads only supplied JSON, NPY and text evidence. Never reads labels, raw text,
weights, Memory bodies; no fitting, model evaluation, project execute/native.
Bootstrap uses multinomial occurrence weights, separate from project indexing.
"""
from pathlib import Path
from collections import Counter
import argparse, csv, datetime, hashlib, io, json, platform, re, sys, time
import numpy as np

ORDERS = ('ABC', 'BCA', 'CAB')
ROLES = ('raw', 'stage-cal', 'first-cal', 'primary')
ARMS = ('C', 'S', 'LOGIT0.1')
COLUMNS = ('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct',
           'brier','log_loss','precision','recall','f1','specificity','balanced_accuracy',
           'mcc','map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10',
           'ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')
CLASS_KEYS = ('precision','recall','f1','specificity','balanced_accuracy','mcc')
RANK_INDEX = [i for i,k in enumerate(COLUMNS) if k not in CLASS_KEYS+('brier','log_loss')]
TERMS = {
 'O': ((3,1,.5),(3,2,.5)), 'N': ((2,2,.5),(3,3,.5)), 'Z': ((3,3,1.),),
 'F_first': ((1,1,1.),(3,1,-1.)),
 'F': ((1,1,.5),(3,1,-.5),(2,2,.5),(3,2,-.5)),
 'G': ((2,2,.5),(1,2,-.5),(3,3,.5),(2,3,-.5)),
 'final_all': ((3,1,1/3),(3,2,1/3),(3,3,1/3)), 'A2': ((2,2,1.),)
}

def require(condition, message):
    if not condition: raise AssertionError(message)

def classification(c):
    tp,fp,fn,tn = (int(c[k]) for k in ('tp','fp','fn','tn'))
    den = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** .5
    return dict(precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn),
                f1=2*tp/(2*tp+fp+fn), specificity=tn/(tn+fp),
                balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,
                mcc=(tp*tn-fp*fn)/den if den else 0.)

class Reader:
    def __init__(self, root): self.root=Path(root); self.events={}
    def read(self, name, purpose, record=None):
        name=str(name); data=(self.root/name).read_bytes(); digest=hashlib.sha256(data).hexdigest()
        entry=self.events.setdefault(name,dict(path=name,bytes=len(data),sha256=digest,purposes=[]))
        if purpose not in entry['purposes']: entry['purposes'].append(purpose)
        if record is not None:
            require(len(data)==record['bytes'] and digest==record['sha256'], 'Identity mismatch: '+name)
            entry['registered_identity_verified']=True
        return data
    def json(self, name, purpose, record=None): return json.loads(self.read(name,purpose,record))
    def array(self, name, purpose, record):
        a=np.load(io.BytesIO(self.read(name,purpose,record)),allow_pickle=False)
        return a

def main(root, out):
    started=time.perf_counter(); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    rd=Reader(root); checks={}; deviations={}; results={}; stage_rows=[]; stat_rows=[]
    def near(a,b,name,atol=1e-12):
        x=np.asarray(a,dtype=np.float64); y=np.asarray(b,dtype=np.float64)
        require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),name+' invalid numeric shape/value')
        diff=float(np.max(np.abs(x-y))) if x.size else 0.
        d=deviations.setdefault(name,{'numbers':0,'maximum_absolute_error':0.})
        d['numbers']+=int(x.size); d['maximum_absolute_error']=max(d['maximum_absolute_error'],diff)
        require(diff<=atol,name+' differs by '+repr(diff))

    current=rd.json('result/job/evaluation/collected.json','Full metric registry and row pairing')
    refs={0:rd.json('reference/reference/collected.json','Required baseline first-domain registry'),
          1:rd.json('reference/collected.json','Required baseline continuation registry')}
    policy=rd.json('result/source/schema/step28_record_replay_policy.json','Frozen reference bindings and original criteria')
    for path,rec in policy['reference']['collections'].items():
        rd.read('reference/'+path,'Frozen baseline registry SHA',rec)
    saved=rd.json('result/job/evaluation/evaluation.json','All saved statistics and original checks')
    complete=rd.json('result/job/completion.json','Completed original decision')
    rd.read('result/job/evaluation/evaluation.json','Completion binds evaluation',complete['evaluation'])
    rd.read('result/job/evaluation/collected.json','Evaluation binds collection',saved['collected'])
    require(current['metric_columns']==list(COLUMNS),'22 metric columns/order differ')
    require(len(set(current['group_ids']))==60 and len(current['domains'])==60,'Invalid paired groups')
    for ref in refs.values():
        for k in ('group_ids','domains','metric_columns'): require(ref[k]==current[k],'Baseline pairing '+k)
    domains=np.asarray(current['domains']); rows={d:np.flatnonzero(domains==d) for d in 'ABC'}
    require(all(len(v)==20 for v in rows.values()),'Need 20 groups per actual domain')
    partition=rd.json('result/job/run/partition.json','Public group partition to metric registry alignment')
    require(partition['development']==[dict(domain=d,group_uid=g) for d,g in zip(current['domains'],current['group_ids'])], 'Partition/metric order')
    cache={}; arrays={a:{} for a in ARMS}; counts={a:{} for a in ARMS}
    matrix_catalog=[]
    def load_point(folder, name, roles):
        cache_key=(folder,name)
        if cache_key in cache: return cache[cache_key]
        av={}; cv={}
        for role,records in roles.items():
            matrix_path=folder+'/'+records['matrix']['path']; count_path=folder+'/'+records['counts']['path']
            a=rd.array(matrix_path,'All 60x22 metric values: shape, finite, classifications, statistics',records['matrix'])
            c=rd.json(count_path,'All 60 confusion-count rows and derived metrics',records['counts'])
            require(a.shape==(60,22) and a.dtype==np.dtype('<f8') and np.isfinite(a).all(),matrix_path+' shape/type')
            require(len(c)==60,count_path+' row count')
            for i,row in enumerate(c):
                require(set(row)=={'tp','fp','fn','tn'} and all(type(x) is int and x>=0 for x in row.values()),count_path+' count types')
                require(row['tp']+row['fn']==20 and row['fp']+row['tn']==358,count_path+' full-pair totals')
                cl=classification(row)
                near([a[i,COLUMNS.index(k)] for k in CLASS_KEYS],[cl[k] for k in CLASS_KEYS],'count_to_six_metrics')
            av[role]=a; cv[role]=c
            matrix_catalog.append({'path':matrix_path,'count_path':count_path,'shape':[60,22],
                                   'dtype':str(a.dtype),'rows_checked':60,'values_checked':1320})
        if 'stage-cal' in av:
            for role in ('stage-cal','first-cal'):
                require(np.array_equal(av['raw'][:,RANK_INDEX],av[role][:,RANK_INDEX]),'Ranking changed after calibration '+str(cache_key))
            av['primary']=av['stage-cal'].copy(); av['primary'][:,RANK_INDEX]=av['raw'][:,RANK_INDEX]
        cache[cache_key]=(av,cv); return av,cv

    for name,roles in current['points'].items(): load_point('result/job/evaluation',name,roles)
    require(len(matrix_catalog)==46,'Expected current 46 sets')
    for arm in ARMS:
        for order in ORDERS:
            for s in (1,2,3):
                if arm=='LOGIT0.1':
                    folder='reference/reference' if s==1 else 'reference'
                    col=refs[0 if s==1 else 1]
                    name=order+'_shared' if s==1 else f'{order}_logit_tenth_stage{s}'
                else:
                    folder='result/job/evaluation'; col=current
                    name=order+'_shared' if s==1 else f'{order}_{arm}_stage{s}'
                arrays[arm][order,s], counts[arm][order,s]=load_point(folder,name,col['points'][name])
    require(len(matrix_catalog)==73,'Expected total 73 sets')
    checks['registries_and_sets']={'current':46,'baseline':27,'total':73,'metric_numbers':73*60*22,
                                    'count_rows':73*60,'count_numbers':73*60*4,'classification_numbers':73*60*6,
                                    'actual_domain_groups':{k:len(v) for k,v in rows.items()},
                                    'same_group_ids_domains_and_columns':True,'ranking_columns_bitwise_unchanged':14}

    # Each draw is shared across orders, stages, arms, roles, metrics, endpoints.
    draw=np.random.Generator(np.random.PCG64(20260930)).integers(20,size=(5000,3,20))
    occurrence=np.stack([(draw==i).sum(axis=2) for i in range(20)],axis=2).astype(np.float64)/20
    require(np.allclose(occurrence.sum(2),1.),'Invalid bootstrap weights')
    def contribution(arm,role,ep):
        f=np.zeros((3,3,20,22),dtype=np.float64)
        for oi,order in enumerate(ORDERS):
            for stage,arrival,coef in TERMS[ep]:
                domain=order[arrival-1]; di='ABC'.index(domain)
                f[oi,di]+=coef*arrays[arm][order,stage][role][rows[domain]]
        if ep in ('F_first','F','G'): f[..., [4,5]] *= -1.
        return f
    def summarize(f,expected,label,role,ep):
        per_order=f.sum(axis=(1,2))/20
        means=per_order.mean(axis=0)
        bootstrap=np.einsum('bdg,dgm->bm',occurrence,f.mean(axis=0),optimize=False)
        lo,hi=np.quantile(bootstrap,[.025,.975],axis=0,method='linear')
        result={}
        for i,metric in enumerate(COLUMNS):
            got=[means[i],lo[i],hi[i],*per_order[:,i]]
            exp=expected[metric]; target=[exp['mean'],*exp['conditional_95pct_interval'],*[exp['per_order'][o] for o in ORDERS]]
            near(got,target,'all_endpoint_statistics')
            result[metric]={'mean':float(means[i]),'conditional_95pct_interval':[float(lo[i]),float(hi[i])],
                            'per_order':dict(zip(ORDERS,per_order[:,i].tolist()))}
            stat_rows.append([label,role,ep,metric,*got])
        return result
    fields={}
    for arm in ARMS:
        results[arm]={}
        for role in ROLES:
            results[arm][role]={}
            for ep in TERMS:
                fields[arm,role,ep]=contribution(arm,role,ep)
                results[arm][role][ep]=summarize(fields[arm,role,ep],saved['endpoints'][arm][role][ep],arm,role,ep)
    decision={}
    for other in ('S','LOGIT0.1'):
        label='C-'+other; results[label]={}
        for role in ROLES:
            results[label][role]={}
            for ep in TERMS:
                f=fields['C',role,ep]-fields[other,role,ep]
                results[label][role][ep]=summarize(f,saved['comparisons'][label]['delta'][role][ep],label,role,ep)
        r=results[label]['primary']; lower=lambda ep,metric:r[ep][metric]['conditional_95pct_interval'][0]
        decision[label]={'O_MAP_lower_positive':lower('O','map')>0,
                        'final_all_MAP_mean_positive':r['final_all']['map']['mean']>0,
                        'A2_MAP_lower_ge_minus_point01':lower('A2','map')>=-.01,
                        'Z_MAP_lower_ge_minus_point01':lower('Z','map')>=-.01,
                        'O_AP_lower_ge_minus_point01':lower('O','average_precision')>=-.01}
        require(decision[label]==saved['comparisons'][label]['checks'],label+' original checks')
        require(all(decision[label].values())==saved['comparisons'][label]['pass'],label+' original pass')
    overall=all(all(d.values()) for d in decision.values())
    require(overall==saved['development_criteria_pass']==complete['development_criteria_pass'],'Original overall decision')

    fixed={}; stage_paths={}
    for arm in ARMS:
        fixed[arm]={}; stage_paths[arm]={}
        for ep,select in {'O':((3,1),(3,2)),'N':((2,2),(3,3)),'Z':((3,3),)}.items():
            total=Counter()
            for order in ORDERS:
                for stage,arrival in select:
                    for i in rows[order[arrival-1]]: total.update(counts[arm][order,stage]['stage-cal'][i])
            fixed[arm][ep]={**total,**classification(total),'fpr':total['fp']/(total['fp']+total['tn'])}
        for order in ORDERS:
            stage_paths[arm][order]={}
            for stage in (1,2,3):
                stage_paths[arm][order][stage]={}
                for d in 'ABC':
                    values=arrays[arm][order,stage]['primary'][rows[d]].mean(0)
                    stage_paths[arm][order][stage][d]=dict(zip(COLUMNS,values.tolist()))
                    for k,v in zip(COLUMNS,values): stage_rows.append([arm,order,stage,d,k,float(v)])

    # All current absolute stage results: four macro groups, plus pooled counts/rates.
    for name,roles in current['points'].items():
        av,cv=cache['result/job/evaluation',name]
        for role in roles:
            target=saved['absolute_stage_results'][name][role]
            near(av[role].mean(0),[target['macro_all'][k] for k in COLUMNS],'stage_macro_numbers')
            for d in 'ABC': near(av[role][rows[d]].mean(0),[target['macro_by_domain'][d][k] for k in COLUMNS],'stage_macro_numbers')
            fc=target['fixed_half_classification']; require(fc['threshold']==0.,'Fixed logit threshold')
            for label,idx in [('pooled',np.arange(60)),*[(d,rows[d]) for d in 'ABC']]:
                total=Counter()
                for i in idx: total.update(cv[role][i])
                got={**total,**classification(total),'fpr':total['fp']/(total['fp']+total['tn'])}
                ex=fc['pooled'] if label=='pooled' else fc['by_domain'][label]
                near([got[k] for k in ex],[ex[k] for k in ex],'stage_pooled_counts_rates')

    existing=rd.json('result/verification/attempt01/result.json','Existing Linux audit comparison, not trusted as substitute')
    for arm in ARMS:
        for ep in fixed[arm]:
            near([fixed[arm][ep][k] for k in existing['fixed_half'][arm][ep]],list(existing['fixed_half'][arm][ep].values()),'existing_pooled_endpoint_table')
        for o in ORDERS:
            for s in (1,2,3):
                for d in 'ABC':
                    for m,v in existing['stage_metrics'][arm][o][str(s)][d].items():
                        near(stage_paths[arm][o][s][d][m],v,'existing_stage_diagnostics')
    require(existing['checks']==decision,'Existing verifier decision')
    require(existing['script_sha256']==hashlib.sha256(rd.read('result/analysis/step28_record_replay_result_verify.py','Existing verifier self hash')).hexdigest(),'Existing script identity')
    text=rd.read('result/verification/attempt01/tables.zh.txt','All 704 rendered metric rows vs independent statistics').decode('utf-8')
    role=ep=None; table_rows=0
    for line in text.splitlines():
        if ' / ' in line and line.split(' / ')[0] in ROLES: role,ep=line.split(' / ')
        elif ' | ' in line and line.split(' | ')[0] in COLUMNS:
            pieces=line.split(' | '); metric=pieces[0]
            printed=[float(x) for x in re.findall(r'[-+]?\d+\.\d+', ' | '.join(pieces[1:]))]
            expected=[results[a][role][ep][metric]['mean'] for a in ARMS]
            for label in ('C-S','C-LOGIT0.1'):
                r=results[label][role][ep][metric]; expected += [r['mean'],*r['conditional_95pct_interval']]
            near(printed,expected,'printed_all_22_metric_tables',atol=5.001e-10); table_rows+=1
    require(table_rows==4*8*22,'Full printed table coverage')

    payload={'status':'PASS_INDEPENDENT_SAVED_STATISTICS','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_seconds':time.perf_counter()-started,'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()},
        'scope':'Current reviewer sandbox CPU; independent saved-array aggregation only; no project module/execute/native, labels, raw text, model or Memory loads.',
        'checks':checks,'numeric_comparisons':deviations,'five_conditions':decision,'development_criteria_pass':overall,
        'bootstrap':{'replicates':5000,'seed':20260930,'engine':'PCG64','shape':[5000,3,20],
                     'method':'Shared actual-domain whole-group occurrence weights, fixed orders averaged',
                     'draw_data_sha256':hashlib.sha256(draw.astype('<i8').tobytes()).hexdigest(),
                     'interval':'marginal conditional 95%, linear percentile; not retraining/selection/generation uncertainty'},
        'fixed_half':fixed,'stage_paths':stage_paths,'results':results,
        'limitations':['Underlying label-to-ranking/probability matrices not regenerated.','No independent remote server or deleted-model observation.'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name,obj in [('statistics.json',payload),('machine_read_scope.json',list(rd.events.values())),('matrix_catalog.json',matrix_catalog)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,header,data in [('all_statistics.tsv',['comparison_or_arm','role','endpoint','metric','mean','ci_low','ci_high',*ORDERS],stat_rows),
                             ('all_stage_domain_metrics.tsv',['arm','order','stage','actual_domain','metric','macro'],stage_rows)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f,delimiter='\t',lineterminator='\n'); writer.writerow(header);writer.writerows(data)
    print(json.dumps({k:payload[k] for k in ('status','elapsed_seconds','checks','numeric_comparisons','five_conditions','development_criteria_pass')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.input,args.output)
