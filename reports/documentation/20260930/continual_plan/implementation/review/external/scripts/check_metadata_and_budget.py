"""Read-only local metadata, exact budget arithmetic, no data/model loading."""
from pathlib import Path
import ast,hashlib,json,sys
from fractions import Fraction
R=Path('/mnt/data/bge_continual_external_audit');S=R/'source'
sys.path.insert(0,str(S/'scripts'))
import step28_bge_continual as m
import step28_bge_continual_run as r
p=m.contract();static=json.loads((S/'reports/documentation/20260930/continual_plan/implementation/static_check.json').read_text())
checks=[]
for row in static['frozen_source_files']:
 b=(S/row['path']).read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];checks.append(row)
assert r.sources()==static['frozen_source_files']
astfiles=[]
for row in static['syntax_sources']:
 path=S/row['path'];b=path.read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];ast.parse(b.decode('utf-8'),feature_version=(3,10));astfiles.append(row['path'])
rate=[Fraction(1,100000)*(Fraction(t,29) if t<=29 else Fraction(288-t,259)) for t in range(1,289)]
original=[Fraction(1,100000)*(Fraction(t,87) if t<=87 else Fraction(864-t,777)) for t in range(1,865)]
assert sum(rate)*3==sum(original)
summary={'physical_updates':3*288+3*3*2*288,'historical_presentations':3*2*2*288,'gradient_group_presentations':3*288+3*3*2*288+3*2*2*288,'distinct_endpoints':3+3*3*2,'matrices':1+(3+3*3*2)*3,'score_files_expected':1+(3+3*3*2)*4,'active_inference_weight_files_expected':21,'retained_shared_model_Adam_files':3,'reference_numeric_bytes':6*378*4,'local_positive_encoder_steps':sum(v>0 for v in rate),'physical_positive_encoder_steps':21*sum(v>0 for v in rate),'lr_area_per_stage':float(sum(rate)),'lr_area_three_stages':float(3*sum(rate)),'old_864_area':float(sum(original)),'not_equal_optimization_trajectory':True,'current_calibration_pairs':12*378,'current_calibration_positives':12*20,'epochs_per_current_group':6,'expected_draws_per_memory_group':288/6,'memory_group_expected_repeat_ratio_to_current':(288/6)/6,'formal_resource_actuals_not_measured':True}
assert summary['physical_updates']==p['physical_updates']==6048;assert summary['gradient_group_presentations']==p['gradient_group_presentations']==9504
out={'status':'PASS','bound_source_count':len(checks),'bound_sources':checks,'python310_AST_checked_files':astfiles,'source_log_sets_not_misreported_as_52_runtime_sources':True,'budget_arithmetic':summary,'user_choices':json.loads((S/'reports/documentation/20260930/continual_plan/implementation_decision.json').read_text()),'old_history_not_rerun':True}
(R/'outputs/metadata_budget.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':out['status'],'bound_source_count':len(checks),'AST_files':len(astfiles),'budget_arithmetic':summary},ensure_ascii=False,indent=2))
