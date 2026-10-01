#!/usr/bin/env python3
"""Independent NumPy/stdlib audit of the submitted fixed-s0 saved results.

No project modules, labels, text, model weights, training or fitting are used.
Truth-dependent per-group AP/MAP/Brier/etc. are not recomputed. All numeric
expectations below follow the specified statistical definitions or saved counts.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, importlib.metadata, json, math, os
from pathlib import Path
import platform, sys, traceback
import numpy as np

COLS = ('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct',
        'brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc',
        'map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10',
        'ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')
ROLES = ('A_raw','A_cal','C_raw','C_cal')
PAIRS = (('C_cal','A_cal'),('C_cal','A_raw'),('C_cal','C_raw'),('A_cal','A_raw'))
KEYS = ('tp','fp','fn','tn')
CKEYS = ('precision','recall','f1','specificity','balanced_accuracy','mcc')
RANK = (0,1,2,3,*range(12,22))
TOL=1e-12

class Checks:
    def __init__(self, root: Path):
        self.root=root; self.numeric_count=0; self.exact_count=0; self.maximum=0.; self.records=[]; self.files={}
    def equal(self,a,b,name,exact=False):
        x,y=np.asarray(a),np.asarray(b)
        if x.shape!=y.shape: raise AssertionError(f'{name}: shapes {x.shape} vs {y.shape}')
        if exact:
            if not np.array_equal(x,y): raise AssertionError(f'{name}: exact mismatch')
            self.exact_count+=x.size; err=0.
        else:
            x=x.astype(np.float64);y=y.astype(np.float64)
            if not np.isfinite(x).all() or not np.isfinite(y).all(): raise AssertionError(name+': nonfinite')
            err=float(np.max(np.abs(x-y))) if x.size else 0.
            if err>TOL: raise AssertionError(f'{name}: difference {err:.17g} exceeds {TOL}')
            self.numeric_count+=x.size; self.maximum=max(err,self.maximum)
        self.records.append({'check':name,'values':int(x.size),'exact':exact,'max_abs_difference':err})
    def require(self,ok,name):
        if not ok: raise AssertionError(name)
        self.records.append({'check':name,'boolean':True})
    def file(self,base,rec):
        p=(base/rec['path']).resolve()
        self.require(p.is_relative_to(self.root),'input confined to submission: '+rec['path'])
        self.require(p.suffix not in ('.pt','.safetensors') and '/supervision/' not in str(p)
                     and p.name!='items.jsonl','no privileged/formal raw input: '+rec['path'])
        b=p.read_bytes();d=hashlib.sha256(b).hexdigest()
        self.require(len(b)==rec['bytes'] and d==rec['sha256'],'file receipt: '+str(p.relative_to(self.root)))
        self.files[str(p.relative_to(self.root))]={'bytes':len(b),'sha256':d}
        return p
    def array(self,base,rec,shape,dtype):
        v=np.load(self.file(base,rec),allow_pickle=False)
        self.require(v.shape==shape and v.dtype==dtype and np.isfinite(v).all(),'array schema: '+rec['path'])
        return v

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def avg(v):
    x=np.asarray(v,dtype=np.float64)
    if x.ndim==1:return math.fsum(map(float,x))/len(x)
    return np.array([math.fsum(map(float,x[:,k]))/len(x) for k in range(x.shape[1])])
def ci(v):
    x=np.sort(np.asarray(v),axis=0); out=[]
    for q in (.025,.975):
        h=(len(x)-1)*q; lo=math.floor(h); f=h-lo
        out.append(x[lo]+f*(x[math.ceil(h)]-x[lo]))
    return np.asarray(out)
def cstats(row):
    tp,fp,fn,tn=map(int,row);pp=tp+fp;pos=tp+fn;neg=fp+tn
    precision=tp/pp if pp else 0.;recall=tp/pos if pos else 0.;specificity=tn/neg if neg else 0.
    den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'fpr':fp/neg if neg else 0.,
            'specificity':specificity,'balanced_accuracy':.5*(recall+specificity),
            'mcc':(tp*tn-fp*fn)/den if den else 0.}
def carray(raw):
    return np.array([[r[k] for k in KEYS] if isinstance(r,dict) else r for r in raw],dtype=np.int64)
def acceptance(comps):
    p=comps['C_cal_minus_A_cal'];r=comps['C_cal_minus_A_raw']
    specs=[('T1',p['map']['mean'],'>=',.01),('T2',p['map']['conditional_95pct_interval'][0],'>',0.),
           ('T3',p['recall_at_5']['mean'],'>',0.),('T4',p['average_precision']['mean'],'>=',0.),
           ('T5',p['roc_auc']['mean'],'>=',0.),('T6',p['brier']['mean'],'<=',0.),
           ('T7',p['log_loss']['mean'],'<=',0.),('T8',r['brier']['mean'],'<=',0.),('T9',r['log_loss']['mean'],'<=',0.)]
    checks={k:bool(v>=b if op=='>=' else v>b if op=='>' else v<=b) for k,v,op,b in specs}
    return {'checks':checks,'passed':all(checks.values()),'failed':[k for k,v in checks.items() if not v],
            'observed':{k:{'value':v,'operator':op,'bound':b} for k,v,op,b in specs}}

def run(root,out):
    check=Checks(root);out.mkdir(parents=True,exist_ok=False)
    job=root/'reports/seller_alias_continual/20260928/test_execution/20260928_182700/execution/job'
    result={'status':'RUNNING','no_project_modules_imported':True,'formal_label_parses':0,'formal_text_parses':0,
            'model_loads':0,'gpu_execution':0,'training_updates':0,'new_fits':0,'label_reconstruction':False,
            'truth_dependent_group_metrics_recomputed':False}
    try:
        environment={'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
            'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            'actual_threads':int(next(s.split(':')[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('Threads:'))),
            'packages_metadata_only':{}}
        for pkg in ('numpy','scipy','torch','sentence-transformers','transformers'):
            try:environment['packages_metadata_only'][pkg]=importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:environment['packages_metadata_only'][pkg]=None
        check.require(len(environment['cpu_affinity'])==1 and environment['actual_threads']==1,'actual single CPU and thread')
        write(out/'environment.json',environment)
        ev=load(job/'evaluation/evaluation.json');col=load(job/'evaluation/collected.json')
        blind=load(job/'blind.json');done=load(job/'completion.json');prep=load(job/'preparation.json')
        access=load(job/'heldout_access.json');align=load(job/'alignment.json');policy=load(root/'schema/step28_alias_test_policy.json')
        check.require(tuple(ev['columns'])==COLS and ev['columns']==col['columns'],'all 22 columns in fixed order')
        ids=ev['group_ids']; domains=np.array(ev['domains']); dr={d:np.flatnonzero(domains==d) for d in 'ABC'}
        check.require(len(set(ids))==120 and all(len(x)==40 for x in dr.values()),'120 unique groups and 40/domain')
        for obj in (blind,col,align):check.require(obj['group_ids']==ids,'group identity correspondence')
        check.require([r['group_uid'] for r in prep['group_metadata']]==ids,'public group rows align')
        check.require([r['domain'] for r in prep['group_metadata']]==domains.tolist(),'public domains align')
        check.require(all(len(align['sellers'][g])==28 and align['sellers'][g]==sorted(align['sellers'][g]) for g in ids),'28 ordered seller IDs per group')
        check.require(len({s for g in ids for s in align['sellers'][g]})==3360,'seller IDs unique across groups')
        check.require(align['queries']==3360 and align['pairs']==45360 and align['positive_pairs']==2400 and align['labels_saved'] is False,'counts/identity only; no true labels saved')
        check.file(job,access['blind']);check.file(job,col['blind']);check.file(job,done['evaluation'])
        check.require(access['blind']==col['blind'],'labels bound to complete blind capture')
        check.require(access['parse_attempts']==1 and done['label_parses']=={'train':0,'development':0,'heldout':1,'owners':0},'one successful-stage label attempt record')
        check.require(all(access[k]==done['label_parses'][k] for k in done['label_parses']),'access tallies correspond')
        for obj in (prep,col,ev):check.require(obj['source_files']==done['source_files'],'same frozen34 snapshot')
        for rec in done['source_files']:check.file(root,rec)
        check.require(len(done['source_files'])==34,'34 sources')
        for obj in (blind,col,ev):check.require(obj['policy_sha256']==hashlib.sha256((root/'schema/step28_alias_test_policy.json').read_bytes()).hexdigest(),'policy digest')
        check.require(prep['policy']==policy and ev['statistics']==policy['statistics'],'snapshot policy exact')
        check.require(col['status']=='ALL_FOUR_TEST_MATRICES_SAVED_BEFORE_STATISTICS','complete capture before statistics record')
        check.require(done['status']==ev['status']=='FIXED_PAIR_TEST_EVALUATED_NO_AUTOMATIC_RETRY','completed result status')
        check.require(done['training_updates']==0 and done['calibration_fits']==0,'no formal updates/refits')
        check.require((job.parent/'exit_status.txt').read_text().strip()=='0','formal process exit0')
        print('Verified saved source/role/group/label-attempt correspondence',flush=True)
        scores={}; mats={}; counts={}; fixed={};means={}; per_domain={};mapping_info={};order_info={};threshold_checks={}
        comparisons={}; strict={}; crows=[]; rowmetrics=[];pred_checks=0
        for model in ('A','C'):
            rec=policy['models'][model];fit=load(check.file(root,rec['calibration']))
            check.require(fit['a']==rec['a'] and fit['b']==rec['b'] and fit['a']>0,'fixed positive parameters '+model)
            manifest=load(check.file(root,rec['source_manifest']));point=manifest['points']['6']
            check.require(point['epoch']==6 and point['run_id']==rec['run_id']==('s0_d' if model=='A' else 's0_hard'),'fixed checkpoint '+model)
            for k in ('bytes','sha256'):check.require(point['model'][k]==rec['payload'][k],'checkpoint file '+k)
            check.require(point['model']['state_sha256']==rec['payload']['payload_state_sha256'],'payload digest distinct '+model)
            check.require(point['model_state_sha256']==rec['payload']['model_parameters_sha256'],'pure parameter digest '+model)
            check.require(fit['origin']['model']==point['model'] and fit['origin']['model_state_sha256']==point['model_state_sha256'],'fit origin '+model)
            actual=blind['actual_inference'][model]
            check.require(actual['model_file']=={k:rec['payload'][k] for k in ('path','bytes','sha256')},'actual inference file receipt '+model)
            check.require(actual['parameters_sha256']==rec['payload']['model_parameters_sha256'] and actual['groups']==120 and actual['training_updates']==0,'actual inference restored state receipt '+model)
            check.require(blind['models'][model]['origin']=={'model':rec['payload'],'map':rec['calibration']},'blind provenance '+model)
            for suffix,dtype in (('raw',np.float32),('cal',np.float64)):
                role=model+'_'+suffix;scores[role]=check.array(job,blind['models'][model]['files'][role],(120,378),dtype)
                er=ev['points'][role];mats[role]=check.array(job/'evaluation',er['file'],(120,22),np.float64)
                check.require(er['file']==col['points'][role]['file'] and er['counts']==col['points'][role]['counts'],'pre/post stats artifact '+role)
                raw=er['counts'];cc=carray(raw)
                check.require(cc.shape==(120,4) and np.all(cc>=0),'confusion schema '+role)
                check.equal(cc[:,0]+cc[:,2],np.repeat(20,120),role+' positive count',True)
                check.equal(cc[:,1]+cc[:,3],np.repeat(358,120),role+' negative count',True)
                check.equal([sum(float(s)>=0. for s in group) for group in scores[role]],cc[:,0]+cc[:,1],role+' blind predicted total',True);pred_checks+=120
                cs=np.array([[cstats(row)[k] for k in CKEYS] for row in cc])
                check.equal(mats[role][:,[COLS.index(k) for k in CKEYS]],cs,role+' six classification columns from saved counts')
                counts[role]=cc;means[role]=dict(zip(COLS,avg(mats[role]).tolist()));per_domain[role]={}
                check.equal([er['mean'][k] for k in COLS],list(means[role].values()),role+' means fsum')
                fixed[role]={}
                check.require(er['fixed_classification']['threshold']==0.,'fixed probability0.5 logit0 '+role)
                for domain,index in {'pooled':np.arange(120),**dr}.items():
                    if domain!='pooled':
                        mm=avg(mats[role][index]);per_domain[role][domain]=dict(zip(COLS,mm.tolist()))
                        check.equal([er['by_domain'][domain][k] for k in COLS],mm,role+'/domain/'+domain)
                    stats=cstats(cc[index].sum(axis=0));fixed[role][domain]=stats
                    saved=er['fixed_classification']['pooled'] if domain=='pooled' else er['fixed_classification']['by_domain'][domain]
                    for key,value in saved.items():check.equal(value,stats[key],role+'/fixed/'+domain+'/'+key)
                    crows.append({'role':role,'scope':domain,**stats})
                rowmetrics.append({'role':role,'groups':120,'queries':3360,'total_pairs':45360,'positive_pairs':2400})
            a,b=float(fit['a']),float(fit['b']);x=scores[model+'_raw'];y=scores[model+'_cal']
            mapped=np.array([[a*float(v)+b for v in row] for row in x],dtype=np.float64)
            check.equal(y,mapped,model+' scalar affine replay',True)
            pairs=[(i,j) for i in range(28) for j in range(i+1,28)]
            edgemap={pair:e for e,pair in enumerate(pairs)};ties=0;query_count=0
            for g,(u,v) in enumerate(zip(x,y)):
                o=sorted(range(378),key=lambda e:(float(u[e]),e));n=sorted(range(378),key=lambda e:(float(v[e]),e))
                check.require(o==n,model+f'/group{g}/ascending stable order')
                tx=[float(u[o[i]])==float(u[o[i-1]]) for i in range(1,378)];ty=[float(v[n[i]])==float(v[n[i-1]]) for i in range(1,378)]
                check.require(tx==ty,model+f'/group{g}/tie partition');ties+=sum(tx)
                for q in range(28):
                    others=[j for j in range(28) if j!=q]
                    get=lambda row,j:float(row[edgemap[tuple(sorted((q,j)))]])
                    qo=sorted(others,key=lambda j:(-get(u,j),j));qn=sorted(others,key=lambda j:(-get(v,j),j))
                    check.require(qo==qn,model+f'/group{g}/query{q}/descending order');query_count+=1
            check.equal(mats[model+'_raw'][:,RANK],mats[model+'_cal'][:,RANK],model+' all14 curve/retrieval invariance',True)
            old=load(check.file(root,rec['old_threshold_source']));t=float(old['threshold']);diag=ev['automatic'][model]
            check.require(old['model_state_sha256']==rec['payload']['model_parameters_sha256'],'strict threshold model binding '+model)
            check.require(diag['counts']==col['automatic'][model]['counts'] and diag['original_threshold']==t and diag['mapped_threshold']==a*t+b,'strict threshold original capture '+model)
            check.require(diag['decision']=='compare_original_logits_with_original_threshold_no_refit','strict raw-space decision '+model)
            cc=carray(diag['counts']);check.require(cc.shape==(120,4) and np.all(cc>=0),'strict count schema '+model)
            check.equal(cc[:,0]+cc[:,2],np.repeat(20,120),model+' strict pos',True);check.equal(cc[:,1]+cc[:,3],np.repeat(358,120),model+' strict neg',True)
            rawpred=np.array([[float(s)>=t for s in group] for group in x])
            check.equal(rawpred.sum(1),cc[:,0]+cc[:,1],model+' strict rawfloat64 decision counts',True);pred_checks+=120
            wrong=y>=float(a*t+b)
            threshold_checks[model]={'original_threshold':t,'mapped_threshold':a*t+b,'wrong_mapped_comparison_disagreement_pairs':int(np.count_nonzero(wrong!=rawpred)),
                                   'groups_with_disagreement':int(np.any(wrong!=rawpred,axis=1).sum()),
                                   'raw_at_threshold':int(np.equal(x.astype(np.float64),t).sum())}
            strict[model]={'counts':cc,'report':{}}
            mapping_info[model]={'a':a,'b':b,'raw_zero_for_calibrated_half':-b/a,'score_seconds_from_log':actual['score_seconds']}
            order_info[model]={'groups':120,'pairs_per_group':378,'exact_tie_adjacencies':ties,'queries':query_count,'replayed_values':int(y.size)}
        print('Verified scalar mapping, all group/query orders, raw-score thresholds, and count-derived classification',flush=True)
        draws=np.random.Generator(np.random.PCG64(20260928)).integers(0,40,size=(5000,3,40))
        saved=np.load(check.file(job/'evaluation',ev['bootstrap']),allow_pickle=False)
        check.equal(saved,draws,'complete PCG64 index tensor',True)
        # Independently derive frequency weights, with no call to production metrics/statistics.
        weights=np.zeros((5000,120),dtype=np.int64)
        for k in range(5000):
            for d,domain in enumerate('ABC'): weights[k,dr[domain]]=np.bincount(draws[k,d],minlength=40)
        check.equal(weights.sum(1),np.repeat(120,5000),'whole groups per repetition',True)
        for index in dr.values():check.equal(weights[:,index].sum(1),np.repeat(40,5000),'40 groups/domain/repetition',True)
        distributions={};comparison_rows=[]; allmean_rows=[];domain_rows=[]
        for cand,ref in PAIRS:
            name=cand+'_minus_'+ref;delta=mats[cand]-mats[ref]
            # einsum reductions rather than production indexed means or submitted BLAS @.
            boot=np.einsum('rg,gm->rm',weights.astype(np.float64),delta,optimize=False)/120.
            # Independent indexed domain means cross-check every replicate, not just the CI.
            indexed=np.zeros_like(boot)
            for d,domain in enumerate('ABC'):indexed+=delta[dr[domain]][draws[:,d]].mean(axis=1)/3.
            check.equal(boot,indexed,name+' all5000x22 independent sampling identities')
            for k in (0,124,125,2500,4874,4875,4999):
                explicit=np.array([math.fsum(float(delta[dr[domain][j],c]) for d,domain in enumerate('ABC') for j in draws[k,d])/120 for c in range(22)])
                check.equal(boot[k],explicit,name+f'/replicate{k}/scalar fsum')
            bounds=ci(boot);point=avg(delta);comparisons[name]={};distributions[name]=boot
            for c,metric in enumerate(COLS):
                r={'mean':float(point[c]),'conditional_95pct_interval':bounds[:,c].tolist(),
                   'by_domain':{d:float(avg(delta[index,c])) for d,index in dr.items()}}
                stored=ev['comparisons'][name][metric]
                check.equal(stored['mean'],r['mean'],name+'/'+metric+'/point')
                check.equal(stored['conditional_95pct_interval'],r['conditional_95pct_interval'],name+'/'+metric+'/CI')
                check.equal([stored['by_domain'][d] for d in 'ABC'],[r['by_domain'][d] for d in 'ABC'],name+'/'+metric+'/domains')
                comparisons[name][metric]=r
                comparison_rows.append({'comparison':name,'metric':metric,'difference':r['mean'],'lower':bounds[0,c],'upper':bounds[1,c],**r['by_domain']})
        for model in ('A','C'):
            cc=strict[model].pop('counts');domain_pass=[]
            for scope,index in {'pooled':np.arange(120),**dr}.items():
                st=cstats(cc[index].sum(0));samples=np.einsum('rg,gc->rc',weights[:,index],cc[index],optimize=False)
                intervals={}
                for key,num,other in (('precision',0,1),('recall',0,2),('fpr',1,3)):
                    vals=np.array([float(r[num])/int(r[num]+r[other]) if r[num]+r[other] else 0. for r in samples])
                    intervals[key]=ci(vals).tolist()
                stored=ev['automatic'][model]['report']['pooled'] if scope=='pooled' else ev['automatic'][model]['report']['by_domain'][scope]
                for key in ('tp','fp','fn','tn','precision','recall','fpr','f1'):check.equal(stored[key],st[key],model+'/strict/'+scope+'/'+key)
                for key in intervals:check.equal(stored['conditional_95pct_intervals'][key],intervals[key],model+'/strict/'+scope+'/'+key+'/CI')
                st['conditional_95pct_intervals']=intervals
                if scope!='pooled':
                    passed=(st['fp']*1000<=st['fp']+st['tn'] and st['tp']*2>=st['tp']+st['fn'])
                    check.require(stored['passes_point_gates']==passed,model+'/strict/'+scope+'/gate');domain_pass.append(passed);st['passes_point_gates']=passed
                strict[model]['report'][scope]=st
            check.require(ev['automatic'][model]['report']['all_domains_pass']==all(domain_pass),model+'/all strict domains gate')
            strict[model]['all_domains_pass']=all(domain_pass)
        decision=acceptance(comparisons)
        for rec in (ev['acceptance'],done['acceptance']):
            check.require(rec['checks']==decision['checks'] and rec['failed']==decision['failed'] and rec['passed']==decision['passed'],'all nine machine verdicts exact')
            for key,r in decision['observed'].items():
                check.require(rec['observed'][key]['operator']==r['operator'] and rec['observed'][key]['bound']==r['bound'],key+' frozen operator and bound')
                check.equal(rec['observed'][key]['value'],r['value'],key+' unrounded observation')
        user=load(root/'reports/seller_alias_continual/20260928/test_result/user_acceptance.json')
        check.require(user['original_contract_outcome']['passed'] is False and user['original_contract_outcome']['failed']==decision['failed'],'postresult decision does not rewrite original verdict')
        check.equal(user['original_contract_outcome']['map_difference'],comparisons['C_cal_minus_A_cal']['map']['mean'],'user receipt original MAP')
        # Only a handcrafted numerical boundary; no formal scores or labels are altered.
        t=math.nextafter(1.,math.inf);a=.75;b=10.;x=1.
        check.require((x>=t) is False and (a*x+b>=a*t+b) is True,'handcrafted mapped-threshold rounding caveat')
        # Performance tolerances remain exactly zero; numerical matching has separate tolerance.
        check.require(.01>=.01 and not(math.nextafter(.01,0)>=.01) and not(0.>0.),'acceptance threshold boundary semantics')
        for c,metric in enumerate(COLS):
            r={'metric':metric,**{role:means[role][metric] for role in ROLES},'C_cal_minus_A_cal':comparisons['C_cal_minus_A_cal'][metric]['mean'],
               'lower':comparisons['C_cal_minus_A_cal'][metric]['conditional_95pct_interval'][0],'upper':comparisons['C_cal_minus_A_cal'][metric]['conditional_95pct_interval'][1]};allmean_rows.append(r)
            for domain in 'ABC':domain_rows.append({'domain':domain,'metric':metric,**{role:per_domain[role][domain][metric] for role in ROLES},
                **{name:co[metric]['by_domain'][domain] for name,co in comparisons.items()}})
        for filename,rows in [('all22_metrics.csv',allmean_rows),('four_comparisons.csv',comparison_rows),('per_domain_metrics.csv',domain_rows),('fixed_threshold_counts.csv',crows)]:
            with (out/filename).open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        np.savez_compressed(out/'independent_bootstrap_outputs.npz',draws=draws,**distributions)
        result.update(status='PASS_INDEPENDENT_SAVED_RESULT_AUDIT',numeric_values_checked=int(check.numeric_count),
            exact_values_checked=int(check.exact_count),maximum_numeric_difference=check.maximum,numeric_tolerance=TOL,performance_tolerance=0,
            decision=decision,means=means,by_domain=per_domain,comparisons=comparisons,fixed=fixed,automatic=strict,
            mapping=mapping_info,order=order_info,threshold_checks=threshold_checks,blind_count_rows=pred_checks,
            files_verified=check.files,environment=environment,
            scope='Saved receipts, score transformations/order, confusion algebra, group summaries/bootstrap/acceptance. No per-pair label recovery, model execution or truth-based recomputation of AP/MAP/Brier.',
            repeated_labelled_criteria='Original nine conditions, T1 failure preserved; user post-result acceptance is a separate research-direction decision.')
        print('Verified four full comparisons, all22 group bootstrap, original9 guards and separately recorded user acceptance',flush=True)
        write(out/'check_details.json',check.records);write(out/'independent_results.json',result)
        print(json.dumps({k:result[k] for k in ('status','numeric_values_checked','exact_values_checked','maximum_numeric_difference','blind_count_rows','decision')},ensure_ascii=False,indent=2))
    except Exception as exc:
        result.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                      numeric_values_checked=int(check.numeric_count),maximum_numeric_difference=check.maximum)
        write(out/'check_details.json',check.records);write(out/'independent_results.json',result)
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.root.resolve(),args.out.resolve())
