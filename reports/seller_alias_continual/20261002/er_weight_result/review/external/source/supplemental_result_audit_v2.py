"""ID-only independent replay schedule and full report-table crosscheck.
No project functions imported; no text/label/cache-body/native-model files read.
"""
from __future__ import annotations
import argparse,collections,csv,datetime,hashlib,json,os,pathlib,random,re
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--project',type=pathlib.Path,required=True);p.add_argument('--independent',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--cpu',type=int,default=0);a=p.parse_args();a.output.mkdir(exist_ok=False);os.sched_setaffinity(0,{a.cpu})
R=a.project;N=R/'reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job';B=R/'reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job';S=R/'reports/seller_alias_continual/20261002/er_weight_result'
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def seed(*parts):return int.from_bytes(hashlib.sha256((json.dumps(list(parts),ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()).digest()[:8],'big')%(2**63-1)
def digest(v):return hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
pol=read(R/'schema/step28_bge_continual_policy.json');partition=read(N/'run/partition.json');nm=read(N/'run/manifest.json');bm=read(B/'run/manifest.json');uid_dom={x['group_uid']:x['domain'] for x in partition['fit']};fit={d:sorted(u for u in uid_dom if uid_dom[u]==d) for d in 'ABC'}
checks=[];seedrows=[]
for order in ('ABC','BCA','CAB'):
 reservoir=[];seen=0;rng=random.Random(seed(pol['memory_seed'],order,'retention'))
 for arrival in (1,2):
  for u in fit[order[arrival-1]]:
   seen+=1;j=len(reservoir) if len(reservoir)<6 else rng.randrange(seen)
   if j<6:
    if j==len(reservoir):reservoir.append(u)
    else:reservoir[j]=u
  source=order+'_er_after1' if arrival==1 else order+'_er_stage2'
  assert reservoir==bm['memories'][source]['members']
  stage=arrival+1;cs=seed(pol['schedule_seed'],order,stage,'current');current=[];crng=random.Random(cs)
  for epoch in range(6):
   ids=fit[order[stage-1]].copy();crng.shuffle(ids);current+=ids
  hrng=random.Random(seed(pol['memory_seed'],order,stage,'history_draws'));history=[reservoir[hrng.randrange(6)] for _ in range(288)]
  for i in range(288):seedrows.append(dict(order=order,stage=stage,update=i+1,current_uid=current[i],history_uid=history[i],current_dropout_seed=seed(cs,i,'dropout'),history_dropout_seed=seed(pol['memory_seed'],order,stage,i,'history_dropout')))
  for arm,base in [('er',B),('half',N),('quarter',N)]:
   name=f'{order}_{arm}_stage{stage}';t=read(base/'run/updates'/f'{name}.json');assert t['current_dropout_stream']==cs and t['current_ids']==current and t['history_ids']==history
   assert t['memory_after_training']['members']==reservoir and t['memory_after_training']['draw_count']==288
   checks.append(dict(name=name,source=source,seen=seen,members=reservoir.copy(),member_domain_counts=dict(collections.Counter(uid_dom[u] for u in reservoir)),stream=cs,current_sha256=digest(current),history_sha256=digest(history),checked_current_ids=len(current),checked_history_ids=len(history)))
with (a.output/'id_schedule_and_derived_dropout_seeds.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(seedrows[0]));w.writeheader();w.writerows(seedrows)
save(a.output/'id_schedule_checks.json',checks)
log=(N/'train.log').read_text();events=[json.loads(line) for line in log.splitlines() if line.startswith('{')];updates=[e for e in events if e.get('event')=='updates'];assert len(updates)==144
for order in ('ABC','BCA','CAB'):
 for arm in ('half','quarter'):
  for stage in (2,3):
   selected=[e for e in updates if (e['order'],e['arm'],e['stage'])==(order,arm,stage)]
   assert [v['stage_updates'] for v in selected]==list(range(24,289,24))
   assert all(v['logical_updates']==(stage-1)*288+v['stage_updates'] for v in selected)
assert all(x['elapsed_seconds']<y['elapsed_seconds'] for x,y in zip(updates,updates[1:]))
assert events[-1]['status']=='COMPLETE_ER_WEIGHT_DEVELOPMENT_COMPARISON'
assert 'Traceback (most recent call last)' not in log
start=datetime.datetime.fromisoformat((N/'started.txt').read_text().strip());end=datetime.datetime.fromisoformat((N/'finished.txt').read_text().strip());assert (end-start).total_seconds()==38369
usage=(N/'resource_usage.log').read_text();assert '10:39:29' in usage and '7092740' in usage and 'Exit status: 0' in usage
project_audit=read(S/'audit/audit.json');assert json.loads((S/'audit_stdout.log').read_text())==project_audit and (S/'audit_stderr.log').stat().st_size==0
for name,file in [('script','scripts/step28_er_weight_audit.py'),('helper','scripts/step28_bge_continual_audit.py')]:
 data=(R/file).read_bytes();assert project_audit[name]==dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
local_audit=read(a.independent.parent/'supplied_audit_cpu0/audit.json');assert local_audit['numeric_comparisons']==project_audit['numeric_comparisons']==672510 and local_audit['maximum_absolute_error']==project_audit['maximum_absolute_error']
ends=read(a.independent/'independent_endpoints.json');comps=read(a.independent/'independent_comparisons.json');doc=(R/'docs/SELLER_ALIAS_ER_WEIGHT_RESULT.zh.md').read_text();rows=[]
def compare(label,displayed,actual,digits,line):
 v=float(displayed.replace('−','-').replace('+',''));tol=.5*10**-digits+2e-14;err=abs(v-actual);assert err<=tol,(label,v,actual,err,tol)
 rows.append(dict(label=label,report_line=line,displayed=v,independent=actual,absolute_error=err,rounding_tolerance=tol))
def numbers(s):return re.findall(r'[+−-]?\d+\.\d+',s)
method_names={'SEQ':'seq','ER λ=1':'er','ER λ=0.5':'half','ER λ=0.25':'quarter'};comparison_names={'λ0.5−原ER':'half_minus_er','λ0.25−原ER':'quarter_minus_er','λ0.5−SEQ':'half_minus_seq','λ0.25−SEQ':'quarter_minus_seq'}
section='';lines=doc.splitlines()
for line_no,line in enumerate(lines,1):
 if line.startswith('### ') and '端点的完整指标' in line:section=line.split()[1]
 if not line.startswith('|'):continue
 cells=[c.strip() for c in line.strip('|').split('|')];label=cells[0]
 if label in method_names:
  arm=method_names[label];pairs=[('O','map'),('O','recall_at_5'),('N','map'),('N','recall_at_5'),('Z','map'),('Z','recall_at_5'),('F_first','map')]
  assert len(cells)==8
  for cell,(ep,m) in zip(cells[1:],pairs):compare(f'core/{arm}/{ep}/{m}',cell,ends[arm]['primary'][ep][m]['mean'],6,line_no)
 elif label in comparison_names and len(cells)==6:
  name=comparison_names[label];d=comps[name]['primary'];om=d['O']['map'];ns=numbers(cells[1]);assert len(ns)==3
  for val,exp,k in zip(ns,[om['mean'],*om['conditional_95pct_interval']],['mean','low','high']):compare(name+'/O/map/'+k,val,exp,6,line_no)
  for cell,(ep,m) in zip(cells[2:5],[('O','recall_at_5'),('N','map'),('Z','map')]):compare(name+'/'+ep+'/'+m,cell,d[ep][m]['mean'],6,line_no)
  assert cells[5]==str(comps[name]['passed'])+'/23'
 elif label in comparison_names and len(cells)==4:
  name=comparison_names[label]
  for cell,order in zip(cells[1:],('ABC','BCA','CAB')):compare(name+'/order/'+order,cell,comps[name]['primary']['O']['map']['per_order'][order],6,line_no)
 elif section in ('O','N','Z') and label in ends['seq']['primary'][section] and len(cells)==5:
  for cell,arm in zip(cells[1:],('seq','er','half','quarter')):compare(f'full/{section}/{label}/{arm}',cell,ends[arm]['primary'][section][label]['mean'],9,line_no)
assert len([x for x in rows if x['label'].startswith('full/')])==264
# Prose quantities not already covered by the full tables: critical conditional CIs.
prose_checks=[('quarter_minus_seq','O','recall_at_5','[−0.035417,−0.003274]'),('quarter_minus_seq','N','map','[−0.035226,−0.019692]'),('quarter_minus_seq','Z','map','[−0.043346,−0.018854]'),('quarter_minus_seq','F_first','map','[−0.016057,0.014143]'),('half_minus_er','O','recall_at_5','[−0.008631,0.018155]')]
for name,ep,m,token in prose_checks:
 assert token in doc;line_no=next(i+1 for i,s in enumerate(lines) if token in s)
 for val,exp,k in zip(numbers(token),comps[name]['primary'][ep][m]['conditional_95pct_interval'],('low','high')):compare(f'prose/{name}/{ep}/{m}/{k}',val,exp,6,line_no)
assert comps['quarter_minus_seq']['primary']['O']['map']['conditional_95pct_interval'][0]<0<comps['quarter_minus_seq']['primary']['O']['map']['conditional_95pct_interval'][1]
assert comps['quarter_minus_seq']['primary']['F_first']['map']['conditional_95pct_interval'][0]<0<comps['quarter_minus_seq']['primary']['F_first']['map']['conditional_95pct_interval'][1]
for ep,m in [('O','recall_at_5'),('N','map'),('Z','map')]:assert comps['quarter_minus_seq']['primary'][ep][m]['conditional_95pct_interval'][1]<0
with (a.output/'report_numeric_checks.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Aggregate protection is not domain/order-wide protection. Export all final-domain differences.
col=read(N/'evaluation/collected.json');ref=read(N/'evaluation/reference/collected.json');groups=col['group_ids'];print('GROUP_IDS_TYPE',type(groups).__name__)
# group_ids are already bound and independently aligned; select rows using partition development order.
guids=[v['group_uid'] for v in partition['development']];domain_rows={d:[i for i,v in enumerate(partition['development']) if v['domain']==d] for d in 'ABC'}
assert guids==groups==ref['group_ids']
metric_names=list(ends['seq']['primary']['O']);assert metric_names==col['metric_columns']==ref['metric_columns'];domain_diffs=[]
for cand in ('half','quarter'):
 for reference in ('er','seq'):
  for order in ('ABC','BCA','CAB'):
   ca=np.load(N/'evaluation'/col['points'][f'{order}_{cand}_stage3']['stage-cal']['matrix']['path'],allow_pickle=False)
   ra=np.load(N/'evaluation/reference'/ref['points'][f'{order}_{reference}_stage3']['stage-cal']['matrix']['path'],allow_pickle=False)
   for domain,ix in domain_rows.items():
    delta=(ca[ix]-ra[ix]).mean(0)
    for k,m in enumerate(metric_names):domain_diffs.append(dict(comparison=cand+'_minus_'+reference,order=order,actual_domain=domain,arrival_role='old' if domain in order[:2] else 'latest',metric=m,delta=float(delta[k])))
with (a.output/'all_final_domain_differences.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(domain_diffs[0]));w.writeheader();w.writerows(domain_diffs)
summary=dict(status='PASS_ID_ONLY_SCHEDULE_AND_REPORT_CROSSCHECK',cpu_affinity=sorted(os.sched_getaffinity(0)),schedule_records=len(checks),current_and_history_id_comparisons=len(checks)*288*2,derived_seed_rows=len(seedrows),per_step_seed_evidence='Derived from frozen policy and IDs; actual GPU masks not recorded or rerun.',train_progress_events=len(updates),report_numeric_cells=len(rows),full_22_metric_cells=264,maximum_report_rounding_error=max(v['absolute_error'] for v in rows),project_audit_stdout_matches=True,project_audit_numbers=672510,project_audit_maximum=project_audit['maximum_absolute_error'],final_domain_cells=len(domain_diffs),all_inputs_saved_results_only=True)
save(a.output/'summary.json',summary);print(json.dumps(summary,indent=2))
