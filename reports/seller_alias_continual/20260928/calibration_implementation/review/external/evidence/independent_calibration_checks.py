"""Reviewer-authored references. All scores/labels here are handmade, never formal.

The imported project functions are systems under test, not reference calculators.
Does not call execute(), historical(), a model loader, or any formal CSV reader.
"""
from __future__ import annotations
from pathlib import Path
import copy,csv,decimal,hashlib,itertools,json,math,os,platform,shutil,sys,tempfile,traceback
from unittest import mock
import numpy as np
import scipy

E=Path(__file__).resolve().parent; SRC=E.parent/'submission'; OUT=E/'independent_outputs'
OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(SRC/'scripts'))
import step28_alias_calibration as sut
import step28_alias_calibration_run as runner
LOAD=np.load; array_loads=[]
def guarded_load(path,*args,**kwargs):
    q=Path(path).resolve()
    if q.is_relative_to(SRC): raise AssertionError('Submitted formal arrays must not be parsed')
    array_loads.append(str(q)); return LOAD(path,*args,**kwargs)
np.load=guarded_load
RESULTS={}; COUNTS={}; MAXIMA={}
def close(name,a,b,atol=1e-12):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    assert a.shape==b.shape,(name,a.shape,b.shape)
    err=float(np.max(np.abs(a-b))) if a.size else 0.
    assert np.isfinite(a).all() and np.isfinite(b).all() and err<=atol,(name,err,atol)
    MAXIMA[name]=max(MAXIMA.get(name,0),err); COUNTS[name]=COUNTS.get(name,0)+a.size
    return err

def sigmoid(x):
    if x>=0: return 1./(1.+math.exp(-x))
    t=math.exp(x); return t/(1.+t)

def scalar_reference(theta,x,y):
    a,b=map(float,theta); pairs=list(zip(np.ravel(x),np.ravel(y),strict=True))
    terms=[]; ga=[]; gb=[]; h00=[]; h01=[]; h11=[]
    for xi,yi in pairs:
        xi,yi=float(xi),int(yi); z=a*xi+b; p=sigmoid(z)
        # Bernoulli losses chosen by label, avoiding cancellation of z - z.
        t=-z if yi else z
        terms.append(max(t,0.)+math.log1p(math.exp(-abs(t))))
        r=p-yi; ga.append(r*xi); gb.append(r)
        w=p*(1-p); h00.append(w*xi*xi); h01.append(w*xi); h11.append(w)
    n=len(pairs)
    H=np.array([[math.fsum(h00),math.fsum(h01)],[math.fsum(h01),math.fsum(h11)]])/n
    return math.fsum(terms)/n,np.array([math.fsum(ga),math.fsum(gb)])/n,H

def handmade_truth(groups):
    own=np.array([i for i in range(8) for _ in range(2)]+[i for i in range(8,12) for _ in range(3)])
    labels=[]; rng=np.random.default_rng(71029)
    for g in range(groups):
        c=own[rng.permutation(28)]
        labels.append([int(c[i]==c[j]) for i in range(28) for j in range(i+1,28)])
    return np.asarray(labels,dtype=np.uint8)

# Explicit independent definitions, not sklearn and not project metric code.
NAMES=('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct','brier','log_loss',
       'precision','recall','f1','specificity','balanced_accuracy','mcc','map','mrr',
       'recall_at_1','recall_at_3','recall_at_5','recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')

def metric_probability_reference(z):
    # High positive logits amplify 1-ULP probability differences inside log(1-p).
    # Use an independent 90-digit oracle and then round to float64, not a
    # reciprocal sigmoid whose intermediate rounding can be a different result.
    if z >= 8.:
        with decimal.localcontext() as context:
            context.prec=90
            value=decimal.Decimal.from_float(float(z))
            return float(1/(1+(-value).exp()))
    return sigmoid(float(z))


def ref_group(y,x):
    y=np.asarray(y); x=np.asarray(x,dtype=float); n=len(y); positives=int(sum(y)); negatives=n-positives
    indices=sorted(range(n),key=lambda i:(-float(x[i]),i)); tp=fp=0; oldrec=oldfpr=0.; oldprec=1.; ap=trap=roc=rfpr=0.
    for _, block in itertools.groupby(indices,key=lambda i:float(x[i])):
        ids=list(block); tp+=sum(int(y[i]) for i in ids); fp+=len(ids)-sum(int(y[i]) for i in ids)
        recall=tp/positives; fpr=fp/negatives; prec=tp/(tp+fp)
        ap+=(recall-oldrec)*prec; trap+=(recall-oldrec)*(prec+oldprec)/2.; roc+=(fpr-oldfpr)*(recall+oldrec)/2.
        if fpr<=.01: rfpr=max(rfpr,recall)
        oldrec,oldfpr,oldprec=recall,fpr,prec
    # AUC independently also checked as positive/negative comparisons including ties.
    auc=math.fsum(float(x[i]>x[j])+.5*float(x[i]==x[j]) for i in np.flatnonzero(y) for j in np.flatnonzero(1-y))/(positives*negatives)
    assert abs(roc-auc)<1e-12
    # The specified historical metric first rounds exp(-logaddexp(0,-z))
    # to float64, then clips. Reproduce that numeric interface independently;
    # probability accuracy itself is separately compared to a 90-digit oracle.
    ps=[float(np.exp(-np.logaddexp(0.,-np.float64(z)))) for z in x]
    brier=math.fsum((p-int(v))**2 for p,v in zip(ps,y,strict=True))/n
    pc=[min(max(p,1e-15),1-1e-15) for p in ps]
    logloss=-math.fsum(math.log(p) if v else math.log1p(-p) for p,v in zip(pc,y,strict=True))/n
    ct={'tp':0,'fp':0,'fn':0,'tn':0}
    for yi,xi in zip(y,x,strict=True):
        ct['tp' if yi and xi>=0 else 'fp' if not yi and xi>=0 else 'fn' if yi else 'tn']+=1
    tp,fp,fn,tn=(ct[k] for k in ('tp','fp','fn','tn'))
    pr=tp/(tp+fp) if tp+fp else 0.; re=tp/(tp+fn); sp=tn/(tn+fp); f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); mcc=(tp*tn-fp*fn)/den if den else 0.
    vals=[ap,trap,auc,rfpr,brier,logloss,pr,re,f1,sp,(re+sp)/2,mcc]
    edges=list(itertools.combinations(range(28),2)); table={edge:(float(z),int(v)) for edge,z,v in zip(edges,x,y,strict=True)}
    qr=[]
    for q in range(28):
        cand=[(table[tuple(sorted((q,r)))][0],r,table[tuple(sorted((q,r)))][1]) for r in range(28) if r!=q]
        cand.sort(key=lambda t:(-t[0],t[1])); ranks=[k for k,t in enumerate(cand,1) if t[2]]; total=len(ranks)
        assert total in (1,2)
        row=[math.fsum(j/r for j,r in enumerate(ranks,1))/total,1./ranks[0]]
        row += [sum(r<=k for r in ranks)/total for k in (1,3,5,10)]
        row += [math.fsum(1/math.log2(r+1) for r in ranks if r<=k)/math.fsum(1/math.log2(j+1) for j in range(1,min(k,total)+1)) for k in (1,3,5,10)]
        qr.append(row)
    vals+= [math.fsum(row[j] for row in qr)/28 for j in range(10)]
    return np.asarray(vals),ct

def ref_matrix(y,x):
    pairs=[ref_group(a,b) for a,b in zip(y,x,strict=True)]
    return np.stack([r[0] for r in pairs]),[r[1] for r in pairs]

def test_scalar_derivatives():
    rng=np.random.default_rng(831)
    x=rng.normal(0,3,(4,13)); y=(rng.uniform(size=x.shape)<.23).astype(float)
    records=[]
    for theta in ([1.,0.],[.001,-2.],[1.7,.9],[5.,-10.]):
        value,grad=sut.loss_gradient(np.array(theta),x,y); rv,rg,H=scalar_reference(theta,x,y)
        close('scalar_nll',value,rv); close('scalar_gradient',grad,rg)
        close('hessian_symmetry',H,H.T)
        assert np.linalg.eigvalsh(H).min()>=-1e-14
        for k in range(2):
            step=np.eye(2)[k]*1e-5
            fd=(scalar_reference(np.array(theta)+step,x,y)[0]-scalar_reference(np.array(theta)-step,x,y)[0])/2e-5
            close('finite_difference_gradient',grad[k],fd,1e-8)
            hfd=(sut.loss_gradient(np.array(theta)+step,x,y)[1]-sut.loss_gradient(np.array(theta)-step,x,y)[1])/2e-5
            close('finite_difference_hessian',H[:,k],hfd,1e-8)
        records.append({'theta':theta,'nll':value,'gradient':grad.tolist(),'hessian_min_eigenvalue':float(np.linalg.eigvalsh(H).min())})
    RESULTS['scalar_derivatives']=records

def test_independent_optima_and_boundaries():
    cases=[]
    # p(-2)=1/5; p(.5)=4/5. Both parameters have an analytic unique optimum.
    x=np.repeat([-2.,.5],5); y=np.array([0,0,0,0,1,0,1,1,1,1])
    target_a=2*math.log(4)/2.5; target_b=-math.log(4)+2*target_a
    r=sut.fit(x,y,role='calibration'); assert r['status']=='PASS_CALIBRATION_FIT'
    close('interior_known_optimum',[r['a'],r['b']],[target_a,target_b],2e-6)
    cases.append({'case':'interior_p02_p08','target':[target_a,target_b],'actual':r})
    # Anticorrelation: symmetric intercept zero and slope at strictly positive lower bound.
    x=np.repeat([-2.,2.],5); y=np.array([0,1,1,1,1,0,0,0,0,1])
    r=sut.fit(x,y,role='calibration'); assert r['status']=='PASS_CALIBRATION_FIT'
    _,g,_=scalar_reference([r['a'],r['b']],x,y)
    close('lower_bound',[r['a'],r['b']],[.001,0.],2e-6); assert g[0]>0
    cases.append({'case':'negative_association','gradient':g.tolist(),'actual':r})
    # Width .01 implies unconstrained slope 219.7, outside a<=100.
    x=np.repeat([0.,.01],4); y=np.array([0,0,0,1,0,1,1,1])
    r=sut.fit(x,y,role='calibration'); assert r['status']=='PASS_CALIBRATION_FIT'
    _,g,_=scalar_reference([r['a'],r['b']],x,y)
    close('upper_bound',[r['a'],r['b']],[100.,-.5],2e-6); assert g[0]<0
    cases.append({'case':'positive_slope_upper_bound','gradient':g.tolist(),'actual':r})
    # Constant x: probability identified, a and b separately not identified.
    x=np.full(10,2.); y=np.array([1,1]+[0]*8)
    r=sut.fit(x,y,role='calibration'); assert r['status']=='PASS_CALIBRATION_FIT'
    close('constant_logit_optimum',2*r['a']+r['b'],math.log(.2/.8),2e-6)
    cases.append({'case':'constant_nonidentifiable_parameters','actual':r})
    RESULTS['analytical_scipy_fits']=cases

def test_probability_and_ranking_semantics():
    y=handmade_truth(5); rng=np.random.default_rng(725)
    x=np.stack([rng.normal(size=378),rng.integers(-2,3,378),np.zeros(378),np.linspace(80,90,378),np.where(y[4],.2,-.8)]).astype(float)
    probability_checks=[]
    for a,b in ((.001,-3.),(.7,40.),(100.,-100.)):
        z=sut.transform(x,{'a':a,'b':b}); sut.preserve_order(x,z)
        before,_=sut.metrics.group_metrics(y,x); after,counts=sut.metrics.group_metrics(y,z)
        ref,ct=ref_matrix(y,z); close('independent_22_metrics',after,ref)
        # Analyze ideal real sigmoid vs the frozen rounded probability separately.
        actual_p=np.exp(-np.logaddexp(0.,-z))
        oracle_p=np.array([[metric_probability_reference(float(t)) for t in row] for row in z])
        # Absolute probability error, not a relaxed tolerance for group metrics.
        probability_error=float(np.max(np.abs(actual_p-oracle_p)))
        assert probability_error<=4*np.finfo(float).eps
        cp=np.clip(actual_p,1e-15,1-1e-15); cq=np.clip(oracle_p,1e-15,1-1e-15)
        ideal_metric=-np.mean(y*np.log(cq)+(1-y)*np.log1p(-cq),axis=1)
        interface_metric=-np.mean(y*np.log(cp)+(1-y)*np.log1p(-cp),axis=1)
        # Mean-value-theorem bound for propagating rounded p into log terms.
        bound=np.mean(y*np.abs(cp-cq)/np.minimum(cp,cq)+(1-y)*np.abs(cp-cq)/np.minimum(1-cp,1-cq),axis=1)
        propagated=np.abs(interface_metric-ideal_metric)
        assert np.all(propagated<=bound+1e-13)
        probability_checks.append({'map':[a,b],'max_probability_error_vs_independent_oracle':probability_error,
                                   'max_rounded_vs_oracle_probability_logloss_difference':float(propagated.max()),
                                   'max_propagated_probability_rounding_bound':float(bound.max())})
        assert counts==ct
        rankidx=[i for i,n in enumerate(NAMES) if n in sut.metrics.CURVE_KEYS+sut.metrics.RETRIEVAL_KEYS]
        assert np.array_equal(before[:,rankidx],after[:,rankidx])
    extreme_x=np.array([-1000.,1000.]); extreme_y=np.array([1,0])
    value,_=sut.loss_gradient(np.array([1.,0.]),extreme_x,extreme_y)
    reported=sut.metrics.classification(extreme_y,extreme_x)
    close('extreme_fit_nll',value,1000.); assert reported['log_loss']<35 and reported['brier']==1.
    collapsed=np.array([[0.,float(np.nextafter(np.float32(0),np.float32(1))) ]])
    z=sut.transform(collapsed,{'a':.001,'b':100.})
    try: sut.preserve_order(collapsed,z)
    except ValueError: pass
    else: raise AssertionError('Collapsed new tie accepted')
    # Canonical old-score threshold avoids rounding a threshold into a score.
    raw=np.array([[1.,2.]]); threshold=np.nextafter(1.,np.inf); z=sut.transform(raw,{'a':.001,'b':100.})
    sut.preserve_order(raw,z)
    mapped=.001*threshold+100.
    assert bool((raw>=threshold)[0,0]) is False and bool((z>=mapped)[0,0]) is True
    RESULTS['probability_rank_semantics']={'independent_metric_rows':15,'rank_columns_preserved':len(rankidx),'unclipped_nll_extreme':value,'clipped_logloss_extreme':reported['log_loss'],'new_tie_refused':True,'threshold_roundoff_example':{'raw':1.,'old_threshold':threshold,'mapped_score':z[0,0],'mapped_threshold':mapped,'old_decision':False,'rounded_mapped_decision':True},'saturated_probabilities_are_not_used_for_ranking':True,'probability_oracle_checks':probability_checks,'metric_reference_scope':'Frozen float64 probability interface independently aggregated; high-precision sigmoid accuracy checked separately, not conflated with clipped metric.'}

def reference_acceptance(primary,raw):
    m=primary['metrics']; vals=[m['map']['mean']>=.01,m['map']['conditional_95pct_interval'][0]>0,all(v>0 for v in m['map']['per_seed']),m['recall_at_5']['mean']>0,m['recall_at_5']['per_seed'][0]>0]
    for name in ('average_precision','roc_auc','brier','log_loss'):
        sign=-1 if name in ('brier','log_loss') else 1
        vals.extend([sign*m[name]['mean']>=0,sign*m[name]['per_seed'][0]>=0])
    for name in ('brier','log_loss'): vals.extend([raw['metrics'][name]['mean']<=0,raw['metrics'][name]['per_seed'][0]<=0])
    return vals

def test_seventeen_guards():
    base={'metrics':{n:{'mean':.02 if n in ('map','recall_at_5') else 0.,'per_seed':[.02 if n in ('map','recall_at_5') else 0.]*3,'conditional_95pct_interval':[.01,.03]} for n in NAMES}}
    p=copy.deepcopy(base); r=copy.deepcopy(base); assert sut.acceptance(p,r)['passed']
    # Each guard must independently be able to fail; equals are permitted only where specified.
    cases=[(False,'map','mean',.009),(False,'map','conditional_95pct_interval',[0,.03]),(False,'map','per_seed',[.02,0,.02]),(False,'recall_at_5','mean',0),(False,'recall_at_5','per_seed',[0,.02,.02])]
    for n in ('average_precision','roc_auc','brier','log_loss'):
        v=.001 if n in ('brier','log_loss') else -.001
        cases.extend([(False,n,'mean',v),(False,n,'per_seed',[v,0.,0.])])
    for n in ('brier','log_loss'): cases.extend([(True,n,'mean',.001),(True,n,'per_seed',[.001,0,0])])
    for raw,n,f,v in cases:
        p,r=copy.deepcopy(base),copy.deepcopy(base); (r if raw else p)['metrics'][n][f]=v
        got=sut.acceptance(p,r); assert list(got['checks'].values())==reference_acceptance(p,r)
        assert len(got['failed'])==1
    p,r=copy.deepcopy(base),copy.deepcopy(base)
    for n in ('brier','log_loss'):
        p['metrics'][n].update(mean=-.01,per_seed=[-.01]*3)
        r['metrics'][n].update(mean=.01,per_seed=[.01]*3)
    assert len(sut.acceptance(p,r)['failed'])==4
    RESULTS['seventeen_guards']={'individual_failures_checked':len(cases),'bad_calibrated_A_cannot_pass':True,'reference_booleans_match':True}

def linear_quantile(values,q):
    s=np.sort(values,axis=0); pos=(len(s)-1)*q; i=int(math.floor(pos)); t=pos-i
    return s[i]*(1-t)+s[min(i+1,len(s)-1)]*t

def ref_bootstrap(delta,domains):
    rows=[[i for i,d in enumerate(domains) if d==dom] for dom in 'ABC']; rng=np.random.default_rng(20260927)
    weights=np.zeros((5000,60))
    for rep in range(5000):
        for ids in rows:
            choices=rng.integers(0,20,size=20)
            for k in choices: weights[rep,ids[int(k)]]+=1./60.
    avg=np.mean(delta,axis=0); samples=weights@avg
    return avg.mean(0),np.array([linear_quantile(samples,q) for q in (.025,.975)]),delta.mean(1),{d:avg[ids].mean(0) for d,ids in zip('ABC',rows,strict=True)}

def test_bootstrap_all_comparisons():
    rng=np.random.default_rng(271128); domains=['ABC'[i%3] for i in range(60)]; matrices={}
    common=rng.normal(.3,.04,(60,22))
    for i,seed in enumerate(sut.SEEDS):
        for j,variant in enumerate(sut.VARIANTS): matrices[f'{seed}_{variant}']=common + rng.normal(.001*(i+1)*(j-1),.018,(60,22))
    got=sut.summarize(matrices,domains)
    for c,r in sut.COMPARISONS:
        delta=np.stack([matrices[f'{s}_{c}']-matrices[f'{s}_{r}'] for s in sut.SEEDS])
        means,ci,ps,bd=ref_bootstrap(delta,domains); record=got['comparisons'][c+'_minus_'+r]
        for i,n in enumerate(NAMES):
            m=record['metrics'][n]; close('bootstrap_means',m['mean'],means[i]); close('bootstrap_intervals',m['conditional_95pct_interval'],ci[:,i]); close('bootstrap_per_seed',m['per_seed'],ps[:,i])
            for d in 'ABC': close('bootstrap_by_domain',m['by_domain'][d],bd[d][i])
    assert list(got['acceptance']['checks'].values())==reference_acceptance(got['comparisons']['hard_calibrated_minus_d_calibrated'],got['comparisons']['hard_calibrated_minus_d_raw'])
    (OUT/'bootstrap_reference_checks.json').write_text(json.dumps(got,indent=2))
    RESULTS['bootstrap']={'replicates':5000,'comparisons':4,'columns':22,'same_group_seeds_averaged_before_resampling':True,'domains_interleaved':True}

BUNDLE={}
def test_full_scale_handmade_save_restore():
    rng=np.random.default_rng(92288); yc=handmade_truth(36); yv=handmade_truth(60)
    part={'calibration':[{'group_uid':f'manual_cal_{i}','domain':'ABC'[i//12]} for i in range(36)],'development':[{'group_uid':f'manual_valid_{i}','domain':'ABC'[i%3]} for i in range(60)]}
    saved={}
    for i,run_id in enumerate(sut.RUNS):
        score={'calibration':(rng.normal(-1.6,.9,(36,378))+.8*yc+.02*i).astype(np.float32),'development':(rng.normal(-1.6,.9,(60,378))+.8*yv+.02*i).astype(np.float32)}
        old={role:ref_matrix(y,score[role])[0] for role,y in (('calibration',yc),('development',yv))}
        saved[run_id]={'scores':score,'old_metrics':old,'threshold':float(np.nextafter(np.float64(.5),np.inf)),'origin':{'handmade_run':run_id,'formal_source':False}}
    job=OUT/'handmade_pipeline'; job.mkdir(); fitted=runner.fit_all(job,saved,yc,part)
    arrays=runner.restored_scores(job,saved,part)
    assert len(fitted['runs'])==6 and len(arrays)==12
    fits={}
    for rid in sut.RUNS:
        record=json.loads((job/rid/'fit.json').read_text()); fits[rid]=record
        rv,rg,_=scalar_reference([record['a'],record['b']],saved[rid]['scores']['calibration'],yc)
        close('six_fit_scalar_nll',record['final_nll'],rv); close('six_fit_kkt',record['projected_gradient_max'],np.max(np.abs(rg)),1e-10)
    BUNDLE.update(saved=saved,part=part,yc=yc,yv=yv,job=job,arrays=arrays,fitted=fitted,fits=fits)
    RESULTS['handmade_full_scale']={'fits':6,'calibration_shape':[36,378],'development_shape':[60,378],'fit_records':{r:{k:v[k] for k in ('a','b','initial_nll','final_nll','optimizer_iterations','objective_calls','projected_gradient_max')} for r,v in fits.items()},'all_twelve_scores_reloaded':True}

def test_map_and_score_corruption():
    b=BUNDLE; job=b['job']; tested=[]
    records=[]
    for r in b['fitted']['runs'].values(): records.extend([r['fit'],*r['scores'].values()])
    for rec in records:
        p=job/rec['path']; raw=p.read_bytes(); corrupt=bytearray(raw); corrupt[-2]^=1; p.write_bytes(corrupt)
        try:
            with mock.patch.object(runner,'parse_once',side_effect=AssertionError('No parsing permitted')):
                try: runner.restored_scores(job,b['saved'],b['part'])
                except ValueError: tested.append(rec['path'])
                else: raise AssertionError('Corruption accepted '+rec['path'])
        finally: p.write_bytes(raw)
    assert len(tested)==18
    RESULTS['corruption']={'six_map_plus_twelve_score_files_rejected':tested}

def test_collect_faults_and_saved_only_recovery():
    b=BUNDLE; job=b['job']; collected=runner.collect(job,b['yv'],b['arrays'],b['saved'],b['part'])
    assert len(collected['points'])==12
    for point,rec in collected['points'].items():
        ref,counts=ref_matrix(b['yv'],b['arrays'][point]); got=np.load(job/'evaluation'/rec['file']['path'],allow_pickle=False)
        close('pipeline_valid_22_reference',got,ref); assert counts==rec['counts_at_logit_zero']
    faults=[(sut,'summarize'),(sut.ranking,'paired_summary'),(sut.ranking.base,'metric_comparison'),(sut.ranking.base,'automatic_report')]
    passed=[]
    for obj,name in faults:
        with mock.patch.object(obj,name,side_effect=RuntimeError('reviewer post-collection fault '+name)),mock.patch.object(runner,'parse_once',side_effect=AssertionError('No labels')),mock.patch.object(sut,'fit',side_effect=AssertionError('No fitting')):
            try: runner.finalize(job)
            except RuntimeError as e:
                assert 'reviewer post-collection' in str(e); passed.append(name)
            else: raise AssertionError('Injected fault not reached')
        assert not (job/'evaluation/evaluation.json').exists()
        for rec in collected['points'].values(): sut.data.verify(job/'evaluation'/rec['file']['path'],rec['file'])
    with mock.patch.object(runner,'parse_once',side_effect=AssertionError('No labels')),mock.patch.object(sut,'fit',side_effect=AssertionError('No fitting')),mock.patch.object(runner,'historical',side_effect=AssertionError('No formal inputs')):
        result=runner.finalize(job)
    assert len(result['acceptance']['checks'])==17 and result['maximum_raw_metric_difference']<1e-12
    # Compare four real handmade output differences to independent resampling.
    mats={p:np.load(job/'evaluation'/r['file']['path'],allow_pickle=False) for p,r in collected['points'].items()}
    domains=collected['domains']
    for c,r in sut.COMPARISONS:
        delta=np.stack([mats[f'{s}_{c}']-mats[f'{s}_{r}'] for s in sut.SEEDS]); means,ci,ps,bd=ref_bootstrap(delta,domains)
        for i,n in enumerate(NAMES):
            rec=result['comparisons'][c+'_minus_'+r]['metrics'][n]; close('pipeline_bootstrap_mean',rec['mean'],means[i]); close('pipeline_bootstrap_ci',rec['conditional_95pct_interval'],ci[:,i])
    # A published result cannot silently be overwritten.
    try: runner.finalize(job)
    except FileExistsError: pass
    else: raise AssertionError('Published evaluation overwritten')
    RESULTS['collection_recovery']={'post_collection_faults':passed,'all_twelve_matrices_preserved':True,'recovered_without_labels_or_fitting':True,'raw_metric_max_difference':result['maximum_raw_metric_difference'],'handmade_acceptance_only':result['acceptance']}

def test_bad_old_metrics_and_optimizer_failure():
    b=BUNDLE; saved=copy.deepcopy(b['saved']); saved['s0_d']['old_metrics']['calibration'][0,0]+=.01
    out=OUT/'handmade_old_metric_mismatch'; out.mkdir()
    # Reuse already measured manual optimizer results to exercise this branch without six unnecessary refits.
    with mock.patch.object(sut,'fit',side_effect=[copy.deepcopy(b['fits'][r]) for r in sut.RUNS]) as calls, mock.patch.object(runner,'parse_once',side_effect=AssertionError('No labels')):
        try: runner.fit_all(out,saved,b['yc'],b['part'])
        except ValueError as e: assert 'Calibration relevance/order' in str(e)
        else: raise AssertionError('Old raw metric mismatch accepted')
        assert calls.call_count==6 and (out/'fitted.json').exists()
    # Actual failure status must be saved and stop before a complete fitted marker.
    failed=copy.deepcopy(b['fits']['s0_d']); failed['status']='FAILED_CALIBRATION_FIT_NO_RETRY'; failed['optimizer_success']=False
    out2=OUT/'handmade_injected_fit_failure'; out2.mkdir()
    with mock.patch.object(sut,'fit',return_value=failed) as calls:
        try: runner.fit_all(out2,b['saved'],b['yc'],b['part'])
        except ValueError: pass
        else: raise AssertionError('Failed optimizer accepted')
        assert calls.call_count==1 and (out2/'s0_d/fit.json').exists() and not (out2/'fitted.json').exists()
    RESULTS['failed_fit_and_alignment']={'old_metric_failure_saves_all_six_records':True,'existing_handmade_solver_outputs_reused_for_branch_test':True,'injected_failed_status_preserved_and_no_retry':True}

def test_supplied_cpu_evidence_numerics():
    # Reconstruct only explicitly published handmade fixtures, not any formal scores/labels.
    source=SRC/'reports/seller_alias_continual/20260928/calibration_implementation/cpu/20260928_121900/evidence/handmade.json'
    records=json.loads(source.read_text())['actual_fits']
    own=[c for c in range(8) for _ in range(2)]+[c for c in range(8,12) for _ in range(3)]
    y=np.tile(np.array([int(own[i]==own[j]) for i,j in itertools.combinations(range(28),2)]),(36,1))
    for i,rid in enumerate(sut.RUNS):
        x=(np.random.default_rng(1700+i).normal(-2.4,1.1,(36,378))+.7*y).astype(np.float32).astype(float); r=records[rid]
        v,g,_=scalar_reference([r['a'],r['b']],x,y); v0,_,_=scalar_reference([1,0],x,y)
        close('project_cpu_initial_nll',r['initial_nll'],v0); close('project_cpu_final_nll',r['final_nll'],v); close('project_cpu_final_gradient',r['projected_gradient_max'],np.max(np.abs(g)),1e-10)
        assert r['status']=='PASS_CALIBRATION_FIT' and r['objective_calls']<=2000 and r['optimizer_iterations']<=200
    RESULTS['project_cpu_handmade_numeric_evidence']={'published_handmade_fits_recomputed':6,'formal_scores_used':False}

TESTS=[test_scalar_derivatives,test_independent_optima_and_boundaries,test_probability_and_ranking_semantics,test_seventeen_guards,test_bootstrap_all_comparisons,test_full_scale_handmade_save_restore,test_map_and_score_corruption,test_collect_faults_and_saved_only_recovery,test_bad_old_metrics_and_optimizer_failure,test_supplied_cpu_evidence_numerics]
passed=[]; failed=[]
for test in TESTS:
    print('RUN',test.__name__,flush=True)
    try: test(); passed.append(test.__name__); print('PASS',test.__name__,flush=True)
    except Exception:
        error=traceback.format_exc(); failed.append({'test':test.__name__,'traceback':error}); print(error,file=sys.stderr,flush=True)
        # Continue to retain independent failures; dependent bundle cases explain their own failures.
summary={'status':'PASS_INDEPENDENT_HANDMADE_REVIEW' if not failed else 'FAILED_INDEPENDENT_REVIEW','tests_passed':passed,'tests_failed':failed,'numeric_counts':COUNTS,'maximum_errors':MAXIMA,'results':RESULTS,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'affinity':sorted(os.sched_getaffinity(0)),'threads_line':[s.strip() for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('Threads:')]},'boundaries':{'formal_numpy_arrays_loaded':0,'formal_labels_or_text_read':0,'formal_execute_called':False,'model_loads':0,'training_updates':0,'handmade_array_loads':len(array_loads)}}
(OUT/'independent_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)); print(json.dumps({'status':summary['status'],'passed':len(passed),'failed':len(failed),'maximum_errors':MAXIMA},indent=2),flush=True)
sys.exit(bool(failed))
