from pathlib import Path
import json
import math
p=Path('/workspace/scratch/ca13b63db3f0/review_work/input/project')
result=[]
for study,job,arm in [('low',p/'reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job','tenth'),('logit',p/'reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/reports/job','logit_quarter')]:
 for name in sorted((job/'run/updates').glob('*.json')):
  u=json.loads(name.read_text());b=json.loads((job/f'run/memory/{name.stem}_budget.json').read_text());t=u['memory_after_training']
  point=json.loads((job/f'run/points/{name.stem}.json').read_text())
  row={'name':name.stem,'changed_fields':[k for k in b if b[k]!=t[k]],'training_auxiliary_bytes':t['auxiliary_serialized_bytes'],'checkpoint_auxiliary_bytes':b['auxiliary_serialized_bytes'],'full_memory_byte_delta':b['serialized_bytes']-t['serialized_bytes'],'auxiliary_byte_delta':b['auxiliary_serialized_bytes']-t['auxiliary_serialized_bytes'],'point_auxiliary_bytes':point['learner_auxiliary_serialized_bytes']}
  result.append(row)
low=p/'reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/cpu'
a=json.loads((low/'quarter.json').read_text());b=json.loads((low/'tenth.json').read_text())
ratios={}
for k in a['gradient_components']:
 x=a['gradient_components'][k]['weighted_history']['norm'];y=b['gradient_components'][k]['weighted_history']['norm']
 ratios[k]={'quarter_norm':x,'tenth_norm':y,'norm_ratio':x/y,'norm_difference_to_2_5_multiple':x-2.5*y,'gradient_vectors_provided':False,'diagnosis':'float32 independently reduced norms are not exact vector ratios; v1 arbitrary 2e-6 ratio cutoff was unsupported'}
print(json.dumps({'budget_differences':result,'native_norms':ratios,'code_locations':{'training_summary':'step28_er_weight_run.py:77-95','checkpoint_new_auxiliary':'step28_er_weight_run.py:153-168','post_checkpoint_auxiliary_update':'step28_er_weight_run.py:244-248','native_elementwise_check':'step28_er_weight_check.py:155-164'}},indent=2))
