"""Web standard-library evidence check; no project data/model/train is loaded."""
from pathlib import Path
import hashlib, json
root=Path('/workspace/scratch/f4d639b3b473/relation_memory_result_review')
base=root/'reports/seller_alias_continual/20261006/relation_result'
source=base/'source'
prep=root/'reports/documentation/20261005/relation_memory'
old=Path('/workspace/scratch/f4d639b3b473/relation_memory_pilot_review')
def read(p): return json.loads(p.read_text())
def rec(p,path): return {'bytes':p.stat().st_size,'path':path,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
gate=read(prep/'gate.json'); deployment=read(prep/'deployment.json'); cpu=read(prep/'cpu/result.json'); execution=read(base/'job/execution.json'); collection=read(base/'job/evaluation/collected.json'); evaluation=read(base/'job/evaluation/evaluation.json'); completion=read(base/'job/completion.json')
actual=[rec(source/r['path'],r['path']) for r in gate['source_files']]
assert len(actual)==23 and len({r['path'] for r in actual})==23
bindings={k: actual==v for k,v in [('gate',gate['source_files']),('deployment',deployment['source_files']),('cpu',cpu['source_files']),('job_execution',execution['sources']),('collected',collection['source_files']),('evaluation',evaluation['source_files'])]}
assert all(bindings.values())
gate_record=rec(prep/'gate.json','reports/gate.json')
assert gate_record==deployment['gate']==execution['gate']
evidence={}
for key in ('native','integration_cpu','review_main','external_review'):
 r=gate[key]; got=rec(root/r['path'],r['path']); evidence[key]=got==r; assert got==r
assert gate['runtime']==read(source/'schema/step28_relation_memory_policy.json')['runtime']
assert gate['supervision']==read(source/'schema/step28_relation_memory_policy.json')['supervision']
assert completion['evaluation']==rec(base/'job/evaluation/evaluation.json','evaluation/evaluation.json')
changed=[]
for r in actual:
 p=old/r['path']
 if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']: changed.append(r['path'])
assert changed==['scripts/step28_relation_memory_run.py','tests/test_step28_relation_memory_run.py']
assert evaluation['status']=='STATISTICS_COMPLETE_REQUIRES_VALID_COMPLETION'
assert 'worth_matched_replay' not in evaluation
assert completion['status']=='COMPLETE_RELATION_FIXED_POINT'
assert completion['worth_matched_replay']==evaluation['observed_continuation_checks_pass']==False
budget=completion['budget']; policy=read(source/'schema/step28_relation_memory_policy.json')
assert 0<budget['elapsed_seconds']<policy['runtime']['maximum_gpu_stage_seconds']
assert budget['peak_observed_bytes']<policy['runtime']['maximum_output_bytes']
assert budget['peak_rss_bytes']<64*2**30 and budget['peak_cuda_reserved_bytes']<28*2**30
assert not list((base/'job').glob('failure*.json')) and not list((base/'job').glob('recovery_from_*.json'))
result={'scope':'web stdlib identity/status check; no Torch, model, label or training execution','source_count':23,'source_bindings_equal':bindings,'gate_record_equal':True,'gate_evidence_records_equal':evidence,'only_changed_vs_review10':changed,'core_native_sha_equal':hashlib.sha256((source/'scripts/step28_relation_memory.py').read_bytes()).hexdigest()==read(root/gate['native']['path'])['scientific_sources']['scripts/step28_relation_memory.py'],'evaluation_status':evaluation['status'],'evaluation_has_worth':False,'completion_status':completion['status'],'completion_evaluation_record_equal':True,'worth_matched_replay':False,'completion_budget':budget,'failure_or_recovery_receipts':0,'cpu_original':{k:cpu[k] for k in ('status','tests_run','failures','errors','skipped','elapsed_seconds','formal_data','full_three_order_small_model_updates')}}
print(json.dumps(result,ensure_ascii=False,indent=2))
