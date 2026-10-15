"""Postprocess only supplied handwritten CPU evidence; no model or solver execution."""
import argparse, json, time, sys, platform, hashlib
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('input_root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
t0=time.perf_counter();b=a.input_root/'reports/documentation/20261015/balanced_pooling_repair/qualification/cpu'
read=lambda f:json.loads(f.read_text())
s1=b/'attempt01/result/terminal_precision.json';s2=b/'attempt02/result/terminal_precision.json'
assert s1.read_bytes()==s2.read_bytes()
rows=[]
for e in read(s2):
 q=np.array(e['plan']);ref=np.array(e['reference_plan']);dtype=np.float32 if e['dtype']=='torch.float32' else np.float64
 rows.append({'dtype':e['dtype'],'saved_internal_value_error':abs(e['value']-e['reference_value']),
  'saved_plan_l1_error':float(np.abs(q-ref).sum()),'saved_plan_cast_l1_error':float(np.abs(q.astype(dtype).astype(float)-ref).sum()),
  'saved_output_cast_error':abs(float(dtype(e['value']))-e['reference_value'])})
 assert rows[-1]['saved_internal_value_error']<=1e-8
 assert rows[-1]['saved_plan_l1_error']<=2e-4 and rows[-1]['saved_plan_cast_l1_error']<=2e-4
 assert rows[-1]['saved_output_cast_error'] <= (1e-8 if dtype==np.float64 else 1e-8+1e-7*max(1,abs(e['reference_value'])))
f=b/'attempt02/result/terminal_real_update.json';e=read(f)
# Solve x^2 = beta*(s-x)*(s-2*x), independently of production Sinkhorn.
with localcontext() as ctx:
 ctx.prec=80;D=lambda x:Decimal.from_float(float(x));k=[[D(x) for x in r] for r in e['actual_input']];s=Decimal(1)/3
 assert k[0][0]==k[1][1] and k[0][1]==k[1][0]
 assert all(k[i][j]==k[0][2] for i,j in [(1,2),(2,0),(2,1)])
 A,B,C,Z=k[0][0],k[0][1],k[0][2],k[2][2]
 beta=(2*C-Z).exp()/(A.exp()+B.exp())
 x=(2*beta*s)/(3*beta+(beta*beta+4*beta).sqrt())
 u=(s-x)/(1+(B-A).exp());v=s-x-u;plan=[[u,v,x],[v,u,x],[x,x,s-2*x]]
 value=sum(plan[i][j]*k[i][j]-plan[i][j]*(9*plan[i][j]).ln() for i in range(3) for j in range(3))
 ref=np.array([[float(x) for x in row] for row in plan]);pv=float(value)
result={'scope':'Saved handwritten evidence arithmetic only. No project Linux, GPU, BGE or formal inputs.',
 'precision_evidence_identical_across_attempts':True,'precision_rows':rows,
 'actual_fp32_symmetric_input_reference':{'value':pv,'value_difference_from_saved':pv-e['reference_value'],
  'plan_l1_difference_from_saved':float(np.abs(ref-np.array(e['reference_plan'])).sum()),'decimal_precision':80},
 'elapsed_seconds':time.perf_counter()-t0,'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__},
 'input_files':[{'path':str(f.relative_to(a.input_root)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in [s1,s2,b/'attempt02/result/terminal_real_update.json']]}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
