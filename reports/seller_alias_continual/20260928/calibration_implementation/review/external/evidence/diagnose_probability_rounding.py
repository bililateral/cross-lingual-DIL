from pathlib import Path
import math, decimal, json
import numpy as np
E=Path(__file__).resolve().parent
own=np.array([i for i in range(8) for _ in range(2)]+[i for i in range(8,12) for _ in range(3)])
rng=np.random.default_rng(71029); ys=[]
for g in range(5):
 c=own[rng.permutation(28)]; ys.append([int(c[i]==c[j]) for i in range(28) for j in range(i+1,28)])
y=np.array(ys); rng=np.random.default_rng(725)
x=np.stack([rng.normal(size=378),rng.integers(-2,3,378),np.zeros(378),np.linspace(80,90,378),np.where(y[4],.2,-.8)]).astype(float)
def sig(z):
 if z>=0:return 1/(1+math.exp(-z))
 e=math.exp(z);return e/(1+e)
def clipped_loss(ps,labels):
 p=np.clip(ps,1e-15,1-1e-15);return -(labels*np.log(p)+(1-labels)*np.log1p(-p))
all_records=[]
for a,b in ((.001,-3.),(.7,40.),(100.,-100.)):
 z=a*x+b; p=np.exp(-np.logaddexp(0.,-z)); q=np.array([[sig(float(t)) for t in row] for row in z])
 lp=clipped_loss(p,y);lq=clipped_loss(q,y); dif=lp.mean(1)-lq.mean(1)
 print('MAP',a,b,'GROUP DIFF',dif,'MAX PROB DIFF',np.max(np.abs(p-q)))
 if np.max(np.abs(dif))>1e-12:
  row=int(np.argmax(np.abs(dif))); inds=np.argsort(-np.abs(lp[row]-lq[row]))[:6]; examples=[]
  for k in inds:
   with decimal.localcontext() as ctx:
    ctx.prec=90; t=decimal.Decimal.from_float(float(z[row,k])); truep=1/(1+(-t).exp()); high=float(truep)
    logloss= -(truep.ln() if y[row,k] else (1-truep).ln())
   example={'row':row,'pair':int(k),'logit':float(z[row,k]),'handmade_label':int(y[row,k]),'production_probability':float(p[row,k]),'reviewer_v1_probability':float(q[row,k]),'high_precision_probability_rounded_to_float64':high,'production_ulp_error':float((p[row,k]-high)/np.spacing(high)),'v1_ulp_error':float((q[row,k]-high)/np.spacing(high)),'production_probability_clipped_loss':float(lp[row,k]),'v1_probability_clipped_loss':float(lq[row,k]),'high_precision_unclipped_loss':str(logloss)}
   examples.append(example)
  all_records.append({'map':[a,b],'group_difference':dif.tolist(),'max_probability_difference':float(np.max(np.abs(p-q))),'examples':examples})
result={'classification':'REVIEWER_REFERENCE_NUMERICAL_SEMANTICS_MISMATCH_NOT_PROVEN_PROJECT_DEFECT','observations':all_records,'note':'Legacy evaluation explicitly computes float64 exp(-logaddexp(0,-z)), then clips p; a direct reciprocal sigmoid is algebraically equivalent but can differ by 1 ulp near 1. Do not relax metric tolerance or modify frozen implementation. Separate the mandated float64 probability algorithm from exact Bernoulli logit NLL.'}
(E/'probability_rounding_diagnosis.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
