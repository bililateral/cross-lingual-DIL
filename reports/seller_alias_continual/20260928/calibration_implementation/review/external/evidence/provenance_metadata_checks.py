"""Metadata/hash-only audit; no NumPy array or formal supervision parsing."""
from pathlib import Path
import json,hashlib
E=Path(__file__).resolve().parent; S=E.parent/'submission'
def read(p):return json.loads(p.read_text())
def rec(p):
 b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def check(p,r):assert rec(p)=={'bytes':r['bytes'],'sha256':r['sha256']},str(p)
p=read(S/'schema/step28_alias_calibration_policy.json');J=S/p['historical_job']
for path,row in p['historical_records'].items(): check(J/path,row)
manifest=read(J/'run/manifest.json'); partrec=manifest['partition'];check(J/'run'/partrec['path'],partrec);part=read(J/'run'/partrec['path'])
col=read(J/'evaluation/collected.json');assert col['group_ids']==[r['group_uid'] for r in part['development']]
assert col['domains']==[r['domain'] for r in part['development']]
assert len(col['columns'])==22
report={'method_runs':p['runs'],'partition':{},'bindings':[]}
seen=set()
for role,n in [('fit',144),('calibration',36),('development',60)]:
 ids=[r['group_uid'] for r in part[role]];counts={d:sum(r['domain']==d for r in part[role]) for d in 'ABC'}
 assert len(ids)==len(set(ids))==n and not seen.intersection(ids);seen.update(ids);assert set(counts.values())=={n//3}
 report['partition'][role]={'groups':n,'domain_counts':counts}
for rid in p['runs']:
 armrec=manifest['runs'][rid]['manifest'];assert armrec['path']==f'{rid}/manifest.json';check(J/'run'/armrec['path'],armrec)
 arm=read(J/'run'/armrec['path']);point=arm['points']['6'];assert point['run_id']==rid and point['epoch']==6 and arm['updates']==864 and point['full_model_and_adam_reloaded']
 assert arm['calibration_group_ids']==[r['group_uid'] for r in part['calibration']]
 result={'run_id':rid,'epoch':6,'model_record_only':point['model'],'sources':{}}
 for role in ('calibration','development'):
  sr=point['scores'][role];assert sr['path']==f'scores/epoch6_{role}.npy';check(J/'run'/rid/sr['path'],sr)
  if role=='calibration': mr=point['train_metrics'][role]['file'];mp=J/'run'/rid/mr['path'];assert mr['path']=='scores/epoch6_calibration_metrics.npy'
  else: mr=col['runs'][rid]['points']['6']['file'];mp=J/'evaluation'/mr['path'];assert mr['path']==f'{rid}/epoch6_metrics.npy'
  check(mp,mr);result['sources'][role]={'score':sr,'old_metric':mr,'raw_bytes_verified_only':True}
 tr=arm['calibration'];check(J/'run'/rid/tr['path'],tr);th=read(J/'run'/rid/tr['path'])
 assert th['model_state_sha256']==point['model_state_sha256'] and th['score_sha256']==point['scores']['calibration']['sha256']
 result['old_threshold_record_matches']=True;report['bindings'].append(result)
base=read(S/'schema/step28_chinese_base_policy.json')
for name,key in [('manifest.json','data_manifest_sha256'),('validation.json','data_validation_sha256')]:
 assert rec(S/base['data_root']/name)['sha256']==base[key]
report.update(raw_score_files_hashed=12,old_metric_files_hashed=12,formal_arrays_parsed=0,formal_labels_parsed=0,model_payloads_opened=0,dataset_manifest_and_validation_hashes_match=True)
# Numeric package differences in replay of identical public handmade fixture.
local=read(S/'reports/seller_alias_continual/20260928/calibration_implementation/cpu/20260928_121900/evidence/handmade.json')['actual_fits']
web=read(E/'supplied_audit_artifacts/handmade.json')['actual_fits']
fields=('a','b','initial_nll','final_nll','projected_gradient_max')
report['identical_handmade_cross_environment_max_abs_difference']={f:max(abs(local[r][f]-web[r][f]) for r in local) for f in fields}
report['identical_handmade_iteration_call_counts_match']=all(local[r]['optimizer_iterations']==web[r]['optimizer_iterations'] and local[r]['objective_calls']==web[r]['objective_calls'] for r in local)
report['non_gtol_convergence_example']={'run':'s0_d_handmade_only','termination':local['s0_d']['optimizer_message'],'projected_gradient_max':local['s0_d']['projected_gradient_max'],'solver_gtol':1e-8,'explicit_fit_acceptance_limit':1e-6}
(E/'provenance_metadata_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='bindings'},ensure_ascii=False,indent=2))
