"""Handmade 22-metric reference: no project inputs, labels, score archives or models.
Curve/ranking/classification aggregation is independent. The frozen NumPy sigmoid
expression is intentionally shared for probability numerics and explicitly recorded.
"""
from pathlib import Path
import itertools,json,math,sys
import numpy as np
R=Path('/mnt/data/bge_continual_external_audit');sys.path.insert(0,str(R/'source/scripts'))
import step28_bge_continual as m
pairs=list(itertools.combinations(range(28),2)); owners=sum(([i]*n for i,n in enumerate([3]*4+[2]*8)),[])
y=np.array([owners[a]==owners[b] for a,b in pairs],dtype=np.uint8)
def ref(z):
 x=[float(v) for v in z];label=list(map(int,y));pos=sum(label);neg=len(label)-pos
 order=sorted(range(378),key=lambda i:(-x[i],i));blocks=[]
 for i in order:
  if not blocks or x[i]!=x[blocks[-1][0]]:blocks.append([i])
  else:blocks[-1].append(i)
 tp=fp=0;prev_r=prev_f=0.;prev_p=1.;ap=pr=auc=atfpr=0.
 for block in blocks:
  tp+=sum(label[i] for i in block);fp+=len(block)-sum(label[i] for i in block)
  r=tp/pos;f=fp/neg;p=tp/(tp+fp)
  ap+=(r-prev_r)*p;pr+=(r-prev_r)*(p+prev_p)/2;auc+=(f-prev_f)*(r+prev_r)/2
  if f<=.01:atfpr=max(atfpr,r)
  prev_r,prev_f,prev_p=r,f,p
 prob=np.exp(-np.logaddexp(0.,-np.array(x,dtype=np.float64)));clip=np.clip(prob,1e-15,1-1e-15)
 tp=sum(label[i] and x[i]>=0 for i in range(378));fp=sum(not label[i] and x[i]>=0 for i in range(378));fn=pos-tp;tn=neg-fp
 precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/pos;specificity=tn/neg;den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
 out={'average_precision':ap,'trapezoidal_pr_auc':pr,'roc_auc':auc,'recall_at_fpr_1pct':atfpr,'brier':math.fsum((float(p)-lab)**2 for p,lab in zip(prob,label))/378,'log_loss':-math.fsum(lab*math.log(float(p))+(1-lab)*math.log1p(-float(p)) for p,lab in zip(clip,label))/378,'precision':precision,'recall':recall,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'specificity':specificity,'balanced_accuracy':(recall+specificity)/2,'mcc':(tp*tn-fp*fn)/den if den else 0.}
 rows=[];hits={k:[] for k in (1,3,5,10)}
 for q in range(28):
  ix=[(i,b if a==q else a) for i,(a,b) in enumerate(pairs) if q in (a,b)];ix.sort(key=lambda t:(-x[t[0]],t[1]))
  rel=[label[i] for i,_ in ix];total=sum(rel);ranks=[i+1 for i,l in enumerate(rel) if l]
  one={'map':math.fsum((j+1)/rank for j,rank in enumerate(ranks))/total,'mrr':1/ranks[0]}
  for k in (1,3,5,10):
   n=sum(rel[:k]);one['recall_at_'+str(k)]=n/total;hits[k].append(float(n>0))
   one['ndcg_at_'+str(k)]=math.fsum(v/math.log2(i+2) for i,v in enumerate(rel[:k]))/math.fsum(1/math.log2(i+2) for i in range(min(k,total)))
  rows.append(one)
 for key in rows[0]:out[key]=math.fsum(row[key] for row in rows)/28
 return out,{k:math.fsum(v)/28 for k,v in hits.items()},dict(tp=tp,fp=fp,fn=fn,tn=tn)
rng=np.random.Generator(np.random.PCG64(3012026))
cases={'all_tied':np.zeros(378),'continuous':rng.normal(size=378)*2,'partial_ties':np.round(rng.normal(size=378),1),'extreme':np.linspace(-100,100,378),'all_negative':np.full(378,-2.),'perfect_positive_top':y.astype(float)*2-1}
results=[];maxerror=0.
for name,z in cases.items():
 expected,hits,counts=ref(z);matrix,returned_counts=m.metrics.group_metrics(y[None,:],z[None,:]);got=matrix[0];assert returned_counts[0]==counts;diff={k:float(got[i]-expected[k]) for i,k in enumerate(m.metrics.COLUMNS)}
 err=max(map(abs,diff.values()));maxerror=max(maxerror,err)
 results.append({'name':name,'metrics':expected,'hit_at_k_not_recall':hits,'confusion':counts,'max_absolute_error':err,'differences':diff});assert err<=1e-12,(name,diff)
assert any(abs(x['metrics']['recall_at_1']-x['hit_at_k_not_recall'][1])>1e-6 for x in results)
result={'status':'PASS','handmade_cases':len(cases),'metric_comparisons':len(cases)*22,'maximum_error':maxerror,'cases':results,'probability_numeric_scope':'Frozen np.exp(-np.logaddexp(0.,-z)) and clipped log-loss probabilities deliberately same; independent aggregation/curves/counts/ranking','formal_input_label_model_access':False}
(R/'outputs/independent_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
