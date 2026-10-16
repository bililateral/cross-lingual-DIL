"""Check the supplied prose tables against independently re-computed existing results.
Only formatting comparisons of already registered outputs; no new scientific statistics.
"""
import argparse,csv,json,re
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--return-root',type=Path,required=True);a=p.parse_args()
R=a.input;O=a.return_root;D=O/'reviewer_independent_v3'
lines=(R/'report.zh.txt').read_text().splitlines()
rows=list(csv.DictReader((D/'independent_all_metrics.csv').open()))
stat={tuple(r[k] for k in ['kind','point_or_contrast','role','domain','metric']):r for r in rows}
res=json.loads((D/'reviewer_checks.json').read_text()); checked=[]
def fmt(v):return 'N/A' if v is None or (isinstance(v,float) and np.isnan(v)) else f'{float(v):.9f}'
def val(k):return stat[k]
def trip(k):
 r=val(k);return f"{float(r['mean']):.9f} [{float(r['lower_95']):.9f}, {float(r['upper_95']):.9f}]"
def check(actual,expected,n,field):
 assert actual==expected,dict(line=n,field=field,actual=actual,expected=expected)
 checked.append(dict(line=n,field=field,value=actual))
section=None
for n,line in enumerate(lines,1):
 cells=[x.strip() for x in line.split('|')]
 if line.startswith('比较 | 域 | 差值'):section='primary';continue
 if line.startswith('指标 | S/A'):section='raw';continue
 if line.startswith('比较 | 域 | 指标'):section='auxiliary';continue
 if line.startswith('比较 | 角色 | 域 | Brier'):section='probability';continue
 if line.startswith('方法 | 角色 | 域 | TP'):section='counts';continue
 if line.startswith('方法 | A末旧A MAP'):section='acquisition';continue
 if line.startswith('阶段/角色/域 | 指标'):section='diagnostics';continue
 if line.startswith('阶段 | 分支 | 非零JS'):section='JS';continue
 if '|' not in line:section=None;continue
 if section=='primary':
  comp,d,t,sg=cells;check(t,trip(('final_contrast',comp,'raw',d,'map')),n,'MAP triplet')
  # Original saved values, not label regeneration.
  base=json.loads((R/'evidence/reports/documentation/20261016/h_consistency/baseline_sources.json').read_text())
  roots={'H_cons':R/'evidence/reports/job',**{v['arm']:R/'evidence'/v['root'] for v in base['baselines'].values()}}
  vv=[]
  for arm in ['H_cons',comp.removeprefix('H_cons-')]:
   root=roots[arm];col=json.loads((root/'evaluation/collected.json').read_text());m=json.loads((root/'evaluation'/col['points']['B_'+arm]['raw']['matrix']['path']).read_text());values=np.array(m['values'],float)
   vv.append(values[np.array(col['domains'])==d,m['columns'].index('map')])
  diff=vv[0]-vv[1];check(sg,'/'.join(str(int(f(diff).sum())) for f in [lambda x:x>0,lambda x:x<0,lambda x:x==0]),n,'group signs')
 elif section=='raw':
  metric,*values=cells
  for s,(d,arm) in zip(values,[(d,arm) for d in 'AB' for arm in ['S','G_half','H_cons']],strict=True):
   v=val(('absolute','B_'+arm,'raw',d,metric))['mean'];check(s,fmt(float(v) if v else None),n,arm+'/'+d+'/'+metric)
 elif section=='auxiliary':
  c,d,k,t=cells;check(t,trip(('final_contrast',c,'raw',d,k)),n,k)
 elif section=='probability':
  c,r,d,b,l=cells
  for x,k in [(b,'brier'),(l,'log_loss')]:check(x,trip(('final_contrast',c,r,d,k)),n,r+'/'+d+'/'+k)
 elif section=='counts':
  arm,role,d,*vv=cells;rr=res['classification']['B_'+arm][role][d]
  for text,k in zip(vv,['tp','fp','fn','tn','precision','recall','f1','fpr'],strict=True):check(text,str(rr[k]) if k in ('tp','fp','fn','tn') else fmt(rr[k]),n,k)
 elif section=='acquisition':
  arm,aa,bb,change=cells
  for text,stage in [(aa,'A'),(bb,'B')]:check(text,fmt(float(val(('absolute',stage+'_'+arm,'raw','A','map'))['mean'])),n,stage+' MAP')
  check(change,trip(('old_change',arm,'raw','A','map')),n,'old_change')
 elif section=='diagnostics':
  ps,k,t,v,g,decision=cells;point,role,d=ps.split('/')
  r=next(x for x in res['diagnostics'] if (x['point'],x['role'],x['domain'],x['metric'])==(point,role,d,k))
  check(t,fmt(r['train_benefit']),n,'train benefit');check(v,fmt(r['valid_benefit']),n,'valid benefit')
  check(g,f"{r['gap_widening']:.9f} [{r['gap_CI'][0]:.9f}, {r['gap_CI'][1]:.9f}]",n,'gap CI');check(decision,r['interpretation'],n,'decision')
 elif section=='JS':
  point,role,nz,mean,minmax,weighted,sup,teacher=cells
  tt=json.loads((R/'evidence/reports/job/run/updates'/f'{point}.json').read_text())['rows'];rr=[r[role] for r in tt]
  cc=np.array([r['consistency'] for r in rr]);check(nz,str(int(np.count_nonzero(cc))),n,'nonzero JS');check(mean,fmt(cc.mean()),n,'JS mean')
  check(minmax,f'{cc.min():.9f}/{cc.max():.9f}',n,'JS minmax')
  for text,k in [(weighted,'consistency_contribution'),(sup,'supervised_contribution'),(teacher,'teacher_contribution')]:check(text,fmt(np.mean([r[k] for r in rr])),n,k)
assert len(checked)>500
summary=dict(status='PASS_ALL_SUPPLIED_NUMERIC_TABLE_CELLS',checked_cells=len(checked),source_report='report.zh.txt',against='reviewer_independent_v3/independent_all_metrics.csv and independently checked saved counts/diagnostics/update components',table_rows=len(set(x['line'] for x in checked)),checks=checked)
(O/'report_table_check.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='checks'},ensure_ascii=False,indent=2))
