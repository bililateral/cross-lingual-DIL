"""Independent saved-output calibration result audit; NumPy + Python stdlib only.

No project module is imported. No labels/text/model files or optimizer are used.
Formal per-group truth metrics are NOT recomputed. Matrices/counts are evidence.
Scalar intercept stationarity uses only the disclosed total positive count, never
individual labels or inferred relevance. No formal mapping is re-fitted.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,importlib.metadata,itertools,json,math,os,platform,sys,traceback
from pathlib import Path
import numpy as np

COLS=('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct','brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc','map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')
SEEDS=('s0','s1','s2');ARMS=('d','hard');VARIANTS=('d_raw','d_calibrated','hard_raw','hard_calibrated')
CMP=(('hard_calibrated','d_calibrated'),('hard_calibrated','d_raw'),('hard_calibrated','hard_raw'),('d_calibrated','d_raw'))
RANK=(0,1,2,3,*range(12,22)); CLASS=('precision','recall','f1','specificity','balanced_accuracy','mcc');CK=('tp','fp','fn','tn')
OPTIONS={'maxiter':200,'maxfun':2000,'maxls':40,'maxcor':10,'ftol':1e-12,'gtol':1e-8}

class Audit:
 def __init__(self,root,out):
  self.root=root;self.out=out;self.num=0;self.maxdiff=0.;self.files={};self.narrays={};self.blind_rows=0;self.groups=0;self.queries=0;self.scalar_replays=0
 def equal(self,a,b,name,tol=1e-12):
  x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float)
  if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise AssertionError('Shape/finiteness '+name)
  v=float(np.max(abs(x-y))) if x.size else 0.
  self.num+=x.size;self.maxdiff=max(self.maxdiff,v)
  if v>tol:
   (self.out/'first_numerical_mismatch.json').write_text(json.dumps({'name':name,'max_abs':v,'tolerance':tol,'actual':x.tolist(),'expected':y.tolist()},indent=2))
   raise AssertionError(f'{name}: {v} > {tol}')
 def file(self,base,rec):
  p=(base/rec['path']).resolve();assert p.is_relative_to(self.root)
  assert p.suffix not in ('.pt','.safetensors') and p.name not in ('pairs.csv','items.jsonl','owners.json'),p
  b=p.read_bytes();assert len(b)==rec['bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256'],p
  self.files[str(p.relative_to(self.root))]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return p
 def array(self,base,rec,shape,dtype):
  p=self.file(base,rec);assert p.suffix=='.npy';a=np.load(p,allow_pickle=False)
  assert a.shape==shape and a.dtype==dtype and np.isfinite(a).all(),p
  self.narrays[str(p.relative_to(self.root))]={'shape':list(a.shape),'dtype':str(a.dtype)};return a
 def mapping(self,actual,expected,name):
  for k,v in expected.items():
   if isinstance(v,dict): self.mapping(actual[k],v,name+'/'+k)
   elif isinstance(v,(str,bool)):assert actual[k]==v,(name,k)
   else:self.equal(actual[k],v,name+'/'+k)

def read(p):return json.loads(p.read_text())
def mean_cols(a):return np.array([math.fsum(map(float,a[:,j]))/len(a) for j in range(a.shape[1])])
def interval(a):
 a=np.asarray(a);v=np.sort(a,axis=0);res=[]
 for p in (.025,.975):
  h=(len(v)-1)*p;l=math.floor(h);u=math.ceil(h);res.append(v[l]+(v[u]-v[l])*(h-l))
 return np.asarray(res)
def counts(raw):
 a=np.array([[r[k] for k in CK] if isinstance(r,dict) else r for r in raw]);assert a.dtype.kind in 'iu' and a.ndim==2 and a.shape[1]==4
 assert (a>=0).all() and np.all(a[:,0]+a[:,2]==20) and np.all(a[:,1]+a[:,3]==358);return a

def cm(row):
 tp,fp,fn,tn=map(int,row);p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn);s=tn/(tn+fp);f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
 den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=p,recall=r,specificity=s,fpr=fp/(fp+tn),f1=f,balanced_accuracy=(r+s)/2,mcc=(tp*tn-fp*fn)/den if den else 0.)
def scalar_sigmoid(x):
 if x>=0:return 1/(1+math.exp(-x))
 e=math.exp(x);return e/(1+e)
def verify_counts(A,raw,scores,matrix,th,name):
 c=counts(raw);blind=np.array([sum(float(x)>=th for x in row) for row in scores]);A.equal(c[:,0]+c[:,1],blind,name+'/blind',tol=0);A.blind_rows+=len(c)
 if matrix is not None:A.equal(matrix[:,6:12],[[cm(r)[k] for k in CLASS] for r in c],name+'/class')
 return c

def main(root,out):
 out.mkdir(parents=True,exist_ok=False);A=Audit(root,out)
 result={'status':'RUNNING','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  base=root/'reports/seller_alias_continual/20260928';job=base/'calibration_execution/20260928_140752/job'
  E=read(job/'evaluation/evaluation.json');C=read(job/'evaluation/collected.json');F=read(job/'fitted.json');P=read(job/'preparation.json');done=read(job/'completion.json');old=root/E['policy']['historical_job']
  OC=read(old/'evaluation/collected.json');OM=read(old/'run/manifest.json');part=read(old/'run/partition.json')
  assert tuple(E['columns'])==COLS and E['group_ids']==OC['group_ids'] and E['domains']==OC['domains'];assert P['partition']==part
  assert E['acceptance']==done['acceptance'];assert len(set(E['group_ids']))==60
  domains=np.array(E['domains']);idx={d:np.flatnonzero(domains==d) for d in 'ABC'};assert all(len(v)==20 for v in idx.values())
  assert E['group_ids']==[r['group_uid'] for r in part['development']]
  for rec in done['source_files']:A.file(root,rec)
  mats={};score={};zero_counts={};fit_info={};cal_means={};auto={};group_positive_total=0
  pairs=list(itertools.combinations(range(28),2));qindices=[[(k,v if u==q else u) for k,(u,v) in enumerate(pairs) if q in (u,v)] for q in range(28)]
  for seed in SEEDS:
   for arm in ARMS:
    rid=seed+'_'+arm;r=F['runs'][rid];fr=read(A.file(job,r['fit']));orig=fr['origin'];ar=old/'run'/rid;am=read(A.file(old/'run',orig['manifest']));pt=am['points']['6']
    assert am['run_id']==pt['run_id']==rid and pt['epoch']==6 and am['updates']==864
    assert orig=={'manifest':OM['runs'][rid]['manifest'],'model_state_sha256':pt['model_state_sha256'],'model':pt['model'],'scores':pt['scores']}
    assert fr['group_ids']==am['calibration_group_ids']==[r0['group_uid'] for r0 in part['calibration']]
    assert fr['role']=='calibration' and fr['objective']=='unweighted_unclipped_bernoulli_nll'
    assert fr['initial_parameters']==[1.,0.] and fr['bounds']==[[.001,100.],[-100.,100.]] and fr['options']==OPTIONS
    assert (fr['group_count'],fr['pair_count'],fr['positive_count'])==(36,13608,720)
    a,b=fr['a'],fr['b'];assert .001<a<100. and -100.<b<100.
    assert fr['optimizer_success'] and fr['status']=='PASS_CALIBRATION_FIT'
    assert fr['optimizer_iterations']<=200 and fr['objective_calls']<=2000 and fr['projected_gradient_max']<=1e-6 and fr['final_nll']<=fr['initial_nll']+1e-12
    tr=fr['trajectory'];assert len(tr)==fr['optimizer_iterations'] and [t['iteration'] for t in tr]==list(range(1,len(tr)+1))
    for k in ('a','b','projected_gradient_max'):A.equal(fr[k],tr[-1][k],rid+'/last/'+k,tol=0)
    A.equal(fr['final_nll'],tr[-1]['nll'],rid+'/last/nll',tol=0)
    stop='gtol' if 'NORM OF PROJECTED GRADIENT' in fr['optimizer_message'] else 'ftol';relative_reduction=(tr[-2]['nll']-tr[-1]['nll'])/max(abs(tr[-2]['nll']),abs(tr[-1]['nll']),1.)
    if stop=='gtol':assert fr['projected_gradient_max']<=1e-8
    else:assert 'RELATIVE REDUCTION' in fr['optimizer_message'] and relative_reduction<=1e-12
    fit_info[rid]={k:fr[k] for k in ('a','b','initial_nll','final_nll','raw_brier','calibrated_brier','optimizer_iterations','objective_calls','projected_gradient_max','optimizer_message')}
    fit_info[rid].update(stop=stop,last_relative_nll_reduction=relative_reduction,raw_logit_for_probability_half=-b/a)
    for role,n in [('calibration',36),('development',60)]:
     raw=A.array(ar,orig['scores'][role],(n,378),np.float32).astype(np.float64);cal=A.array(job,r['scores'][role],(n,378),np.float64)
     # Python scalar arithmetic is independent of project vectorized transform.
     replay=np.array([a*float(z)+b for z in raw.flat],dtype=np.float64).reshape(raw.shape)
     A.equal(cal,replay,rid+'/'+role+'/scalar_replay',tol=0);A.scalar_replays+=raw.size
     for x,z in zip(raw,cal):
      order=sorted(range(378),key=lambda k:(float(x[k]),k));after=sorted(range(378),key=lambda k:(float(z[k]),k));assert order==after
      assert all((x[u]==x[v])==(z[u]==z[v]) for u,v in zip(order,order[1:]));A.groups+=1
      for qi in qindices:
       ox=sorted(qi,key=lambda t:(-float(x[t[0]]),t[1]));oz=sorted(qi,key=lambda t:(-float(z[t[0]]),t[1]));assert ox==oz;A.queries+=1
     mm={}
     for variant,values in [('raw',raw),('calibrated',cal)]:
      point=rid+'_'+variant
      if role=='calibration': rec=r['calibration_'+variant];mat=A.array(job,rec['file'],(n,22),np.float64);cr=rec['counts']
      else:
       rec=E['points'][point];assert {k:rec[k] for k in C['points'][point]}==C['points'][point]
       mat=A.array(job/'evaluation',rec['file'],(n,22),np.float64);cr=rec['counts_at_logit_zero'];mats[point]=mat;score[point]=values
      c=verify_counts(A,cr,values,mat,0.,point+'/'+role);mm[variant]=mat
      if role=='calibration':
       cal_means[point]=dict(zip(COLS,mean_cols(mat).tolist()));A.equal(fr[variant+'_brier'],mean_cols(mat)[4],point+'/fitted_brier')
      else:
       zero_counts[point]=c
       for scope,ix in [('mean',np.arange(60)),*idx.items()]:A.mapping(rec['mean'] if scope=='mean' else rec['by_domain'][scope],dict(zip(COLS,mean_cols(mat[ix]).tolist())),point+'/'+scope)
       for scope,ix in [('pooled',np.arange(60)),*idx.items()]:
        actual=rec['fixed_classification']['pooled'] if scope=='pooled' else rec['fixed_classification']['by_domain'][scope];ref=cm(c[ix].sum(0));A.mapping(actual,{k:ref[k] for k in actual},point+'/fixed/'+scope)
     assert np.array_equal(mm['raw'][:,RANK],mm['calibrated'][:,RANK]),rid+role+' rank metrics'
     oldrec=pt['train_metrics']['calibration']['file'] if role=='calibration' else OC['runs'][rid]['points']['6']['file']
     oldmat=A.array(ar if role=='calibration' else old/'evaluation',oldrec,(n,22),np.float64)
     A.equal(oldmat,mm['raw'],rid+'/'+role+'/historical',tol=0)
     if role=='development':
      prior=A.array(job/'evaluation',C['old_raw_metrics'][rid],(60,22),np.float64);A.equal(prior,oldmat,rid+'/old_copy',tol=0)
      A.equal(zero_counts[rid+'_raw'],counts(OC['runs'][rid]['points']['6']['counts_at_logit_zero']),rid+'/old_counts',tol=0)
     else:
      probs=np.array([scalar_sigmoid(float(z)) for z in cal.flat]);mean_probability=math.fsum(map(float,probs))/probs.size
      intercept=mean_probability-fr['positive_count']/fr['pair_count'];assert abs(intercept)<=fr['projected_gradient_max']+5e-16
      fit_info[rid]['aggregate_only_intercept_gradient']=intercept
      fit_info[rid]['calibration_logit_range']=[float(cal.min()),float(cal.max())]
      fit_info[rid]['nll_vs_stored_calibration_log_loss']={v: (fr['initial_nll'] if v=='raw' else fr['final_nll'])-cal_means[rid+'_'+v]['log_loss'] for v in ('raw','calibrated')}
      # This compares TWO SAVED truth-dependent summaries; not a fresh truth loss.
      for v in ('raw','calibrated'):A.equal(fr['initial_nll'] if v=='raw' else fr['final_nll'],cal_means[rid+'_'+v]['log_loss'],rid+'/'+v+'/saved_nll_vs_log_loss')
    diag=E['automatic_diagnostics'][rid];oc=OC['runs'][rid]['automatic_classification'];oldth=read(ar/'calibration.json')['threshold'];assert oldth==oc['threshold']==diag['original_threshold']
    A.equal(diag['mapped_threshold'],a*oldth+b,rid+'/mapped_threshold',tol=0)
    c=verify_counts(A,diag['counts'],score[rid+'_raw'],None,oldth,rid+'/old_threshold')
    A.equal(c,counts(oc['counts_by_group']),rid+'/old_decisions',tol=0);auto[rid]=c
    fit_info[rid]['mapped_threshold_decision_mismatches']=int(np.count_nonzero((score[rid+'_raw']>=oldth)!=(score[rid+'_calibrated']>=diag['mapped_threshold'])))
    print('VERIFIED',rid,stop,'NLL',fr['initial_nll'],'->',fr['final_nll'],'pg',fr['projected_gradient_max'])
  # Sequential domain draws, not the project's prebuilt indexed-mean loop or
  # the submitted analyst's np.add.at construction. Frequency weights reuse group units.
  rng=np.random.Generator(np.random.PCG64(20260927));W=np.zeros((5000,60),dtype=np.int64)
  for rnum in range(5000):
   for d in 'ABC':
    drawn=rng.integers(0,20,size=20)
    for local in drawn:W[rnum,idx[d][local]]+=1
  assert np.all(W.sum(1)==60) and all(np.all(W[:,ix].sum(1)==20) for ix in idx.values())
  np.save(out/'bootstrap_group_frequency_weights.npy',W,allow_pickle=False)
  comparisons={};per_seed={};boot_saved={};table=[]
  for cand,ref in CMP:
   name=cand+'_minus_'+ref;dif=np.stack([mats[s+'_'+cand]-mats[s+'_'+ref] for s in SEEDS]);avg=np.array([[math.fsum(float(dif[s,g,k]) for s in range(3))/3 for k in range(22)] for g in range(60)])
   bs=W@avg/60.;boot_saved[name]=bs;ci=interval(bs);point=mean_cols(avg);seedmeans=np.stack([mean_cols(dif[s]) for s in range(3)]);seedboots=[W@dif[s]/60. for s in range(3)]
   comparisons[name]={};per_seed[name]={s:{} for s in SEEDS}
   for k,m in enumerate(COLS):
    v={'mean':float(point[k]),'per_seed':seedmeans[:,k].tolist(),'by_domain':{d:float(mean_cols(avg[ix])[k]) for d,ix in idx.items()},'conditional_95pct_interval':ci[:,k].tolist()};A.mapping(E['comparisons'][name]['metrics'][m],v,name+'/'+m);comparisons[name][m]=v
    row=dict(comparison=name,metric=m,difference=v['mean'],low=ci[0,k],high=ci[1,k]);row.update({s:float(seedmeans[i,k]) for i,s in enumerate(SEEDS)});table.append(row)
    for i,s in enumerate(SEEDS):
     v={'mean':float(seedmeans[i,k]),'conditional_95pct_interval':interval(seedboots[i][:,k]).tolist()};A.mapping(E['per_seed_comparisons'][name][s][m],v,name+'/'+s+'/'+m);per_seed[name][s][m]=v
   # Independent scalar fsum check of selected WHOLE GROUP replicate outcomes.
   for rep in (0,1,2499,4999):
    expanded=[g for g in range(60) for _ in range(int(W[rep,g]))]
    A.equal(bs[rep],[math.fsum(float(avg[g,k]) for g in expanded)/60 for k in range(22)],name+'/scalar_rep/'+str(rep))
   print('BOOTSTRAP',name,'MAP',comparisons[name]['map'],'Brier',comparisons[name]['brier'])
  np.savez_compressed(out/'bootstrap_mean_differences.npz',**boot_saved)
  # Old diagnostic interval check was absent from the submitted analyst's code.
  automatic_results={}
  for rid,c in auto.items():
   actual=E['automatic_diagnostics'][rid]['report'];calc={};pooled_boot=W@c
   for scope,ix in [('pooled',np.arange(60)),*idx.items()]:
    countsboot=pooled_boot if scope=='pooled' else W[:,ix]@c[ix];ref=cm(c[ix].sum(0));a=actual['pooled'] if scope=='pooled' else actual['by_domain'][scope]
    A.mapping(a,{k:ref[k] for k in ('tp','fp','fn','tn','fpr','precision','recall','f1')},rid+'/auto/'+scope)
    ints={}
    for m,u,v in [('fpr',1,3),('recall',0,2),('precision',0,1)]:
     den=countsboot[:,u]+countsboot[:,v];vals=np.divide(countsboot[:,u],den,out=np.zeros(5000),where=den>0);ints[m]=interval(vals).tolist();A.equal(a['conditional_95pct_intervals'][m],ints[m],rid+'/auto/'+scope+'/'+m)
    calc[scope]={**ref,'conditional_95pct_intervals':ints}
    if scope!='pooled':
     passed=ref['fp']*1000<=ref['fp']+ref['tn'] and ref['tp']*2>=ref['tp']+ref['fn'];assert a['passes_point_gates']==passed;calc[scope]['passes_point_gates']=passed
   assert actual['all_domains_pass']==all(calc[d]['passes_point_gates'] for d in 'ABC');automatic_results[rid]=calc
  p=comparisons['hard_calibrated_minus_d_calibrated'];q=comparisons['hard_calibrated_minus_d_raw']
  checks={'map_minimum_observed_gain':p['map']['mean']>=.01,'map_interval_above_zero':p['map']['conditional_95pct_interval'][0]>0,'map_improves_each_seed':all(x>0 for x in p['map']['per_seed']),'recall_at_5_mean_strictly_improves':p['recall_at_5']['mean']>0,'recall_at_5_s0_strictly_improves':p['recall_at_5']['per_seed'][0]>0}
  for m,sgn in [('average_precision',1),('roc_auc',1),('brier',-1),('log_loss',-1)]:
   checks[m+'_mean_non_degradation']=sgn*p[m]['mean']>=0;checks[m+'_s0_non_degradation']=sgn*p[m]['per_seed'][0]>=0
  for m in ('brier','log_loss'):
   checks[m+'_mean_against_raw_a']=q[m]['mean']<=0;checks[m+'_s0_against_raw_a']=q[m]['per_seed'][0]<=0
  assert len(checks)==17 and checks==E['acceptance']['checks'];assert all(checks.values())==E['acceptance']['passed'];assert E['acceptance']['failed']==[k for k,v in checks.items() if not v]
  means={v:dict(zip(COLS,mean_cols(np.concatenate([mats[s+'_'+v] for s in SEEDS])).tolist())) for v in VARIANTS}
  by_point={pt:{'mean':dict(zip(COLS,mean_cols(m).tolist())),'by_domain':{d:dict(zip(COLS,mean_cols(m[ix]).tolist())) for d,ix in idx.items()}} for pt,m in mats.items()}
  fixed={pt:{'pooled':cm(c.sum(0)),'macro':{k:float(mean_cols(np.array([[cm(row)[k] for k in CLASS] for row in c]))[i]) for i,k in enumerate(CLASS)},'by_domain':{d:cm(c[ix].sum(0)) for d,ix in idx.items()},'zero_predicted_groups':int(np.sum(c[:,0]+c[:,1]==0))} for pt,c in zero_counts.items()}
  mean_table=[dict(metric=m,**{v:means[v][m] for v in VARIANTS}) for m in COLS]
  for name,rows in [('all_22_means.csv',mean_table),('four_comparisons.csv',table),('fixed_half_classification.csv',[dict(point=pt,**r['pooled'],macro_precision=r['macro']['precision'],macro_f1=r['macro']['f1'],zero_predicted_groups=r['zero_predicted_groups']) for pt,r in fixed.items()])]:
   with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  result.update(status='PASS',numeric_comparisons=int(A.num),maximum_absolute_difference=A.maxdiff,metric_tolerance=1e-12,exact_mapping_scalar_values=A.scalar_replays,order_preserved_groups=A.groups,query_rank_lists_checked=A.queries,blind_count_rows=A.blind_rows,files=A.files,arrays=A.narrays,acceptance=checks,fits=fit_info,means=means,by_point=by_point,comparisons=comparisons,per_seed_intervals=per_seed,automatic_diagnostics=automatic_results,fixed_classification=fixed,calibration_means=cal_means,
    environment={'python':sys.version,'numpy':np.__version__,'scipy_installed_not_used':importlib.metadata.version('scipy'),'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'proc_threads':next(line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('Threads:')),'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}},
    scope={'project_imports':0,'formal_labels_read':0,'formal_text_read':0,'model_loads':0,'refits':0,'new_truth_metrics':0,'aggregate_only_intercept_stationarity':True,'limitations':'Full slope gradient/true NLL and per-group AP/MAP/Brier cannot be independently recomputed without truth; saved-record consistency only. No server access.'})
  assert not any(k.startswith('step28_') or k in ('torch','scipy.optimize') for k in sys.modules)
 except Exception as e:
  result.update(status='FAIL',exception_type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),numeric_comparisons=int(A.num),maximum_absolute_difference=A.maxdiff)
  raise
 finally:
  (out/'independent_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ('status','numeric_comparisons','maximum_absolute_difference','exact_mapping_scalar_values','order_preserved_groups','query_rank_lists_checked','blind_count_rows','acceptance','environment')},ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();main(args.root.resolve(),args.out.resolve())
