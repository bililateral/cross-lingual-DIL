"""Check displayed report values and a saved-score threshold rounding edge; no truth."""
from pathlib import Path
from decimal import Decimal
import argparse,json,re,math,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--results',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
r=Path(args.root);d=json.loads(args.results.read_text());out=args.out;out.mkdir(exist_ok=False,parents=True)
doc=r/'docs/SELLER_ALIAS_CALIBRATION_RESULT.zh.md';text=doc.read_text();tables=[];cur=[]
for line in text.splitlines()+['']:
 if line.startswith('|'):cur.append([c.strip() for c in line.strip('|').split('|')])
 elif cur:tables.append(cur);cur=[]
assert len(tables)==5
checked=[]
def num(s,actual,name):
 s=s.replace('**','').replace('−','-').replace('，',',').strip();dec=Decimal(s);tol=float(Decimal(5).scaleb(dec.as_tuple().exponent-1));v=float(dec)
 assert abs(v-float(actual))<=tol+1e-17,(name,s,actual,tol)
 checked.append({'label':name,'display':s,'reference':float(actual),'rounding_tolerance':tol})
cols=list(d['means']['d_raw']);vs=('d_raw','d_calibrated','hard_raw','hard_calibrated')
assert len(tables[0][2:])==22
for k,row in zip(cols,tables[0][2:],strict=True):
 for v,s in zip(vs,row[1:],strict=True):num(s,d['means'][v][k],'22means/'+k+'/'+v)
p=d['comparisons']['hard_calibrated_minus_d_calibrated']
for k,row in zip(('map','recall_at_5','brier','log_loss'),tables[1][2:],strict=True):
 num(row[1],p[k]['mean'],'primary/'+k);lo,hi=row[2].strip('[]').split(',');num(lo,p[k]['conditional_95pct_interval'][0],'primary/'+k+'/low');num(hi,p[k]['conditional_95pct_interval'][1],'primary/'+k+'/high')
for i,(seed,row) in enumerate(zip(('s0','s1','s2'),tables[2][2:],strict=True)):
 for k,s in zip(('map','recall_at_5','brier','log_loss'),row[1:],strict=True):num(s,p[k]['per_seed'][i],'seed/'+seed+'/'+k)
for rid,row in zip(('s0_d','s0_hard','s1_d','s1_hard','s2_d','s2_hard'),tables[3][2:],strict=True):
 fit=d['fits'][rid];num(row[1],fit['a'],rid+'/a');num(row[2],fit['b'],rid+'/b');it,call=row[3].split('／');num(it,fit['optimizer_iterations'],rid+'/iterations');num(call,fit['objective_calls'],rid+'/calls');num(row[4],fit['projected_gradient_max'],rid+'/pg');assert row[5]==fit['stop']
for v,row in zip(vs,tables[4][2:],strict=True):
 for k,s in zip(('tp','fp','precision','recall','f1'),row[1:],strict=True):num(s,d['fixed_classification']['s0_'+v]['pooled'][k],'s0/'+v+'/'+k)
base=r/'reports/seller_alias_continual/20260928';job=base/'calibration_execution/20260928_140752/job';old=r/'reports/seller_alias_continual/20260927/ranking_execution/20260927_114646/job'
e=json.loads((job/'evaluation/evaluation.json').read_text());rid='s1_d';fit=json.loads((job/rid/'fit.json').read_text());raw=np.load(old/'run'/rid/'scores/epoch6_development.npy',allow_pickle=False).astype(np.float64);cal=np.load(job/rid/'development_scores.npy',allow_pickle=False);diag=e['automatic_diagnostics'][rid];original=raw>=diag['original_threshold'];naive=cal>=diag['mapped_threshold'];changed=original!=naive
edge={'run_id':rid,'original_threshold':diag['original_threshold'],'mapped_threshold':diag['mapped_threshold'],'different_decisions_if_naively_comparing_mapped_threshold':int(changed.sum()),'added_predicted_positives':int((~original&naive).sum()),'removed_predicted_positives':int((original&~naive).sum()),'affected_groups':int(np.any(changed,axis=1).sum()),'formal_code_uses_original_space':diag['decision'],'original_predicted_positive_pairs':int(original.sum()),'naive_predicted_positive_pairs':int(naive.sum()),'qualification':'No labels used: added predicted positives must NOT be described as verified false positives.'}
assert edge['different_decisions_if_naively_comparing_mapped_threshold']==d['fits'][rid]['mapped_threshold_decision_mismatches']==93
# Cross-environment comparison of the submitted analyst's only current outputs.
a=base/'calibration_result/analysis';b=r/'reports/calibration_external_saved_audit';csv_equal={n:(a/n).read_bytes()==(b/n).read_bytes() for n in ('mean_metrics.csv','paired_metrics.csv')};assert all(csv_equal.values())
# Byte budget concerns outputs, not RSS. Actual final job contents are small.
resource=(job.parent/'resource_usage.log').read_text();assert '0:01.40' in resource and 'Maximum resident set size (kbytes): 99228' in resource and 'Exit status: 0' in resource
bytes_final=sum(p.stat().st_size for p in job.rglob('*') if p.is_file());assert bytes_final<268435456
result={'status':'PASS','displayed_numeric_values_checked':len(checked),'tables_checked':5,'display_checks':checked,'threshold_rounding_edge':edge,'s2_log_loss_primary_ci':d['per_seed_intervals']['hard_calibrated_minus_d_calibrated']['s2']['log_loss'],'s2_fixed_classification_primary_differences':{k:p[k]['per_seed'][2] for k in ('recall','f1','precision','specificity')},'submitted_replay_csv_bytes_equal':csv_equal,'formal_job_final_files':len([p for p in job.rglob('*') if p.is_file()]),'formal_job_final_bytes':bytes_final,'gnu_time_seconds':1.40,'rss_kib':99228,'labels_or_models_read':0}
(out/'report_edge_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='display_checks'},ensure_ascii=False,indent=2))
