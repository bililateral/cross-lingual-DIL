#!/usr/bin/env python3
"""Saved-only report reconciliation; no project imports or new resampling choices."""
from pathlib import Path
import argparse,csv,hashlib,json,platform,time,traceback,zipfile
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
a=p.parse_args();r=a.input.resolve();o=a.review.resolve();dest=o/'checks/run03';dest.mkdir(parents=True,exist_ok=True)
start=time.perf_counter();reads={};count=0;maxdiff=0.;res={}
def load(path):
 b=path.read_bytes();reads[str(path)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return json.loads(b)
def eq(x,y,where):
 global count,maxdiff
 count+=1
 if isinstance(x,dict):
  assert set(x)==set(y),(where,'keys')
  for k in x:eq(x[k],y[k],where+'/'+k)
 elif isinstance(x,(list,tuple)):
  assert len(x)==len(y),(where,'length')
  for i,(u,v) in enumerate(zip(x,y)):eq(u,v,where+'/'+str(i))
 elif x is None or isinstance(x,(str,bool)):
  assert x==y,(where,x,y)
 else:
  delta=abs(float(x)-float(y));maxdiff=max(maxdiff,delta);assert delta<=1e-12,(where,x,y)
try:
 with zipfile.ZipFile(a.archive) as z:
  infos=z.infolist();assert len(infos)==219 and len({i.filename for i in infos})==219
  assert all(not i.is_dir() and not Path(i.filename).is_absolute() and '..' not in Path(i.filename).parts for i in infos)
  assert z.testzip() is None
  rows=[]
  for i in infos:
   b=z.read(i);assert b==(r/i.filename).read_bytes(),i.filename
   rows.append({'path':i.filename,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 (o/'evidence/input_members.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
 summary=load(r/'result/analysis/summary.json');audit=load(o/'checks/run01/audit_result.json')
 with (o/'checks/run01/recomputed_statistics.csv').open() as f:stats=list(csv.DictReader(f))
 lookup={tuple(x[k] for k in ['kind','point_or_contrast','role','domain','metric']):x for x in stats}
 for point,roles in summary['absolute'].items():
  for role,domains in roles.items():
   for domain,metrics in domains.items():
    for metric,v in metrics.items():
     row=lookup[('absolute',point,role,domain,metric)];eq(v,float(row['mean']) if row['mean'] else None,'summary absolute')
 eq(summary['classification'],audit['classification'],'classification')
 eq(summary['success'],audit['statistics']['success'],'success');eq(summary['baseline_files'],43,'baseline files')
 c=load(r/'result/job/evaluation/collected.json');old_col=load(r/'baseline/evaluation/collected.json');cols=c['columns'];domains=np.array(c['domains']);group_out={}
 for other in ['S','G']:
  h=load(r/'result/job/evaluation'/c['points']['B_G_half']['raw']['matrix']['path']);b=load(r/'baseline/evaluation'/old_col['points']['B_'+other]['raw']['matrix']['path'])
  d=np.asarray(h['values'],float)[:,cols.index('map')]-np.asarray(b['values'],float)[:,cols.index('map')]
  group_out['G_half-'+other]={}
  for domain in 'AB':
   v=d[domains==domain];obj={'positive':int((v>0).sum()),'negative':int((v<0).sum()),'zero':int((v==0).sum()),'values':v.tolist()}
   eq(obj,summary['paired_map_groups']['G_half-'+other][domain],'per group MAP '+other+domain);group_out['G_half-'+other][domain]=obj
 tr=audit['training']['B_G_half'];eq(summary['training'],{'B_G_half':{'nonzero_distillation_steps':tr['positive_D0_steps'],'mean_distillation':tr['mean_D0'],'mean_current_supervised':tr['current_supervised_mean'],'mean_history_supervised':tr['history_supervised_mean']}},'training')
 diagnoses=load(o/'checks/run01/diagnostics.json');metrics=['map','recall_at_5','log_loss']
 for point,roles in summary['diagnostic_summary'].items():
  for role,x in roles.items():
   d=[next(y for y in diagnoses if y['point']==point and y['role']==role and y['metric']==k) for k in metrics]
   expected={'domain':d[0]['domain'],'train_benefit':[v['train_benefit'] for v in d],'valid_benefit':[v['valid_benefit'] for v in d],'gap_widening':[v['gap_widening'] for v in d],'gap_interval':[v['gap_ci'] for v in d],'interpretation':{v['metric']:v['interpretation'] for v in d}}
   eq(x,expected,'diagnostic summary '+point+role);eq(x['interpretation'],summary['overfit_interpretations'][point][role][x['domain']],'interpretations')
 with (o/'checks/run01/diagnostic_curves.csv').open() as f:curves=list(csv.DictReader(f))
 for point,ds in summary['valid_curves'].items():
  for domain,values in ds.items():
   expected=[[float(next(v['mean'] for v in curves if v['point']==point and v['domain']==domain and v['role']=='valid' and int(v['epoch'])==epoch and v['metric']==k)) for k in metrics] for epoch in range(1,7)]
   eq(values,expected,'valid curves '+point+domain)
 expected_res={'A_G_half':[],'B_G_half':[{k:v for k,v in row.items() if k!='point'} for row in load(o/'checks/run01/residuals.json')]};eq(summary['residuals'],expected_res,'residuals')
 console=[json.loads(s) for s in (r/'result/job.console.txt').read_text().splitlines() if s.startswith('{')]
 eq(len(console),26,'26 progress events')
 expected_clock=[1]+list(range(24,289,24))
 eq([x['completed'] for x in console if x['point']=='A_G_half'],expected_clock,'A progress')
 eq([x['completed'] for x in console if x['point']=='B_G_half'],expected_clock,'B progress')
 assert all(b['elapsed_seconds']>u['elapsed_seconds'] for u,b in zip(console,console[1:]))
 assert console[-1]['elapsed_seconds']<6048
 gate=load(r/'context/formal_gate.json');policy=load(r/'result/job/run/sources/schema/step28_g_half_policy.json');auth=load(r/'result/job/run/sources/reports/documentation/20261015/g_half_implementation/authorization.json')
 eq(gate['runtime'],policy['runtime'],'formal runtime');eq(gate['supervision'],policy['supervision'],'formal supervision');eq(gate['kind'],'FORMAL','formal kind');eq(auth['status'],'APPROVED_FULL_CONTRACT_GATED_SEQUENCE','full authorization')
 cpu=load(r/'context/cpu_receipt.json');native=load(r/'context/native_result.json')
 eq(len(set(cpu['qualified_tests'])),6,'six unique CPU tests');eq(sorted([t for attempt in cpu['attempts'] for t in attempt['passed']]),sorted(cpu['qualified_tests']),'CPU inherited complete tests')
 eq(native['native']['actual_updates'],6,'six native actual calls');eq(len(native['native']['updates']),6,'six native update rows')
 correction=list(csv.DictReader((r/'context/baseline_loss_correction.csv').open()));eq(len(correction),18,'18 frozen direction corrections')
 for row in correction:
  eq(float(row['loss_worsening_mean_BminusA']),-float(row['saved_mean_AminusB']),'historical mean sign')
  eq(float(row['loss_worsening_low']),-float(row['saved_high']),'historical low sign')
  eq(float(row['loss_worsening_high']),-float(row['saved_low']),'historical high sign')
 keys=['kind','point_or_contrast','role','domain','metric','mean','lower_95','upper_95']
 subsets={
 'key_map_statistics.csv':[v for v in stats if v['metric']=='map' and ((v['kind']=='absolute' and v['point_or_contrast'].startswith('B_')) or v['kind']!='absolute') and v['role']=='raw'],
 'final_probability_contrasts.csv':[v for v in stats if v['kind']=='final_contrast' and v['metric'] in ['brier','log_loss']],
 'loss_forgetting_contrasts.csv':[v for v in stats if v['kind']=='forgetting_contrast' and v['metric'] in ['brier','log_loss']],
 'selected_secondary_contrasts.csv':[v for v in stats if v['kind']=='final_contrast' and v['role']=='raw' and v['domain']=='B' and v['metric'] in ['recall_at_5','recall_at_fpr_1pct','average_precision','trapezoidal_pr_auc']]}
 for name,rows in subsets.items():
  with (o/'evidence'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 res={'status':'PASS_SAVED_REPORT_RECONCILIATION','checks':count,'maximum_numeric_difference':maxdiff,'archive_crc_ok':True,'input_members_equal_extraction':219,'summary_all_sections_reconciled':True,'progress_events':26,'historical_loss_corrections_checked':18,'new_loss_forgetting_rows':len(subsets['loss_forgetting_contrasts.csv']),'gate_text_inconsistency':{'field':'context/formal_gate.json#/scope','literal':gate['scope'],'structured_kind':gate['kind'],'runtime_matches_formal_policy':True,'full_user_authorization':auth['user_verbatim'],'impact':'Non-blocking descriptive metadata mismatch; do not rewrite frozen gate or rerun training.'},'group_map_wins':group_out}
except Exception as exc:
 res={'status':'FAIL_REVIEWER_RECONCILIATION','error':str(exc)};traceback.print_exc();raise
finally:
 res.update(elapsed_seconds=time.perf_counter()-start,environment={'python':platform.python_version(),'numpy':np.__version__,'project_native':False,'new_resampling_choices':False})
 (dest/'result.json').write_text(json.dumps(res,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 (dest/'read_files.json').write_text(json.dumps(reads,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in res.items() if k!='group_map_wins'},ensure_ascii=False,indent=2))
