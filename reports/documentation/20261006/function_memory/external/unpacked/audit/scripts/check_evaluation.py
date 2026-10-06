"""Independent saved-matrix arithmetic using symbolic, non-experimental matrices."""
from pathlib import Path
import sys,os,json,copy
R=Path(__file__).resolve().parents[2];I=R/'input';O=R/'audit/outputs'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path[:0]=[str(I/'scripts'),str(I/'tests')]
import numpy as np
import step28_function_memory_run as run
orders=['ABC','BCA','CAB'];roles=['raw','stage-cal','first-cal','primary'];cols=list(run.parent.metrics.COLUMNS)
rank=list(run.evaluation.RANK_COLUMNS);nonrank=[i for i in range(22) if i not in rank];domains=list('CAB')*20
rows={d:np.flatnonzero(np.array(domains)==d) for d in 'ABC'}
rng=np.random.default_rng(8080);arrays=[]
for arm in range(2):
 a={}
 for i,order in enumerate(orders):
  for stage in (1,2,3):
   raw=rng.uniform(.25,.75,(60,22));a[run.evaluation.point_name(order,'er',stage)]={}
   for k,role in enumerate(roles[:3]):
    x=raw.copy();x[:,nonrank]+=.01*k
    a[run.evaluation.point_name(order,'er',stage)][role]=x
 arrays.append(a)
actual=run.comparisons(*arrays,domains)
# Our endpoint map and bootstrap use direct weighted rows plus frequency contraction.
terms={'O':[(3,1,.5),(3,2,.5)],'N':[(2,2,.5),(3,3,.5)],'Z':[(3,3,1.)],
 'F_first':[(1,1,1.),(3,1,-1.)],'F':[(1,1,.5),(3,1,-.5),(2,2,.5),(3,2,-.5)],
 'G':[(2,2,.5),(1,2,-.5),(3,3,.5),(2,3,-.5)],'final_all':[(3,1,1/3),(3,2,1/3),(3,3,1/3)],'A2':[(2,2,1.)]}
draws=np.random.Generator(np.random.PCG64(20260930)).integers(0,20,(5000,3,20));assert np.array_equal(draws,run.evaluation.bootstrap_draws())
freq=np.eye(20,dtype=np.float64)[draws].sum(2)/20
fields={};checked=0;maximum=0.
def summarize(field):
 per=field.sum(1).mean(1);value=per.mean(0);mf=field.mean(0);samples=np.einsum('rdg,dgm->rm',freq,mf);ci=np.quantile(samples,[.025,.975],axis=0,method='linear')
 return value,per,ci
def compare(field,saved):
 global checked,maximum
 mean,per,ci=summarize(field)
 for j,name in enumerate(cols):
  reference=np.array([mean[j],*per[:,j],*ci[:,j]]);record=saved[name]
  got=np.array([record['mean'],*[record['per_order'][o] for o in orders],*record['conditional_95pct_interval']])
  maximum=max(maximum,float(np.max(np.abs(got-reference))));checked+=len(reference)
for name,a in zip(['function_memory','logit0.1'],arrays):
 fields[name]={}
 for role in roles:
  fields[name][role]={}
  for ep,tt in terms.items():
   f=np.zeros((3,3,20,22))
   for i,order in enumerate(orders):
    for stage,arrival,weight in tt:
     rec=a[run.evaluation.point_name(order,'er',stage)]
     x=rec['stage-cal'].copy() if role=='primary' else rec[role].copy()
     if role=='primary':x[:,rank]=rec['raw'][:,rank]
     domain=order[arrival-1];f[i,'ABC'.index(domain)]+=weight*x[rows[domain]]
   if ep in ('F_first','F','G'):f[...,[cols.index('brier'),cols.index('log_loss')]]*=-1
   fields[name][role][ep]=f
   compare(f,actual['endpoints'][name][role][ep])
for ep in terms:compare(fields['function_memory']['primary'][ep]-fields['logit0.1']['primary'][ep],actual['delta'][ep])
assert maximum<1e-12
# Five fixed rule boundaries: distinct hypothesis tests with existing stage matrices.
reference={}
for order in orders:
 for stage in (1,2,3):reference[run.evaluation.point_name(order,'er',stage)]={role:np.full((60,22),.5) for role in roles[:3]}
def add(candidate,stage,arrivals,metric,change):
 for order in orders:
  for arrival in arrivals:
   for role in roles[:3]:candidate[run.evaluation.point_name(order,'er',stage)][role][rows[order[arrival-1]],cols.index(metric)]+=change
cases={};same=run.comparisons(reference,reference,domains);cases['equal_no_old_gain']=same['continuation_checks'];assert not same['development_criteria_pass']
positive=copy.deepcopy(reference);add(positive,3,[1,2,3],'map',.02);add(positive,3,[1,2],'average_precision',.02)
r=run.comparisons(positive,reference,domains);cases['qualified_symbolic_gain']=r['continuation_checks'];assert r['development_criteria_pass']
for label,stage,arrivals,metric,change,key in [
 ('second_new_cost',2,[2],'map',-.011,'A2_map_lower_ge_minus_point01'),
 ('latest_new_cost',3,[3],'map',-.04,'Z_map_lower_ge_minus_point01'),
 ('old_AP_cost',3,[1,2],'average_precision',-.04,'O_AP_lower_ge_minus_point01')]:
 cand=copy.deepcopy(positive);add(cand,stage,arrivals,metric,change);r=run.comparisons(cand,reference,domains);cases[label]=r['continuation_checks'];assert not r['continuation_checks'][key]
out={'scope':'SYMBOLIC_METRIC_MATRICES_NOT_MODEL_RESULTS','paired_groups':60,'actual_domain_rows':{k:v.tolist() for k,v in rows.items()},
'draws':5000,'generator':'PCG64','seed':20260930,'numbers_compared':checked,'maximum_absolute_error':maximum,'cases':cases,
'no_labels_no_models':True,'no_success_threshold_modified':True}
(O/'evaluation_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='actual_domain_rows'},ensure_ascii=False,indent=2))
