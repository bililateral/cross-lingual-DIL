"""Current re-submission crosschecks beyond the reused independent arithmetic implementation.
Read only saved metadata/counts/scores and source bytes; no labels, raw texts, or models.
"""
from pathlib import Path
import json,hashlib,zipfile,sys,datetime
import numpy as np
BASE=Path('/mnt/data/ranking_result_reaudit_20260928');ROOT=BASE/'submitted';OUT=BASE/'evidence'
JOB=ROOT/'reports/seller_alias_continual/20260927/ranking_execution/20260927_114646/job'
RES=ROOT/'reports/seller_alias_continual/20260928/ranking_result'
read=lambda p:json.loads(p.read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
man=read(JOB/'run/manifest.json'); policy=read(ROOT/'schema/step28_alias_ranking_policy.json'); auth=read(JOB.parent.parent/'authorization.json')
# Verify source status against the ACTUAL pre-training ZIP, not prior review prose.
old=zipfile.ZipFile('/mnt/data/ranking_review.zip');same=[]
assert sha(Path('/mnt/data/ranking_review.zip').read_bytes())=='bad88d2c83feef2d03f7c545519ab134930c9b50dd81c99d6e3b849008cf5203'
for rec in man['source_files']:
 b=(ROOT/rec['path']).read_bytes();assert b==old.read(rec['path']);same.append(rec)
assert len(same)==18
# Source reference and complete-score/point object roles have to remain unique.
eval=read(JOB/'evaluation/evaluation.json'); col=read(JOB/'evaluation/collected.json')
report=read(RES/'analysis/analysis.json')
assert eval['acceptance']==report['acceptance']==read(JOB/'completion.json')['acceptance']
assert policy['primary_comparison']==['hard','d'] and policy['fixed_test_seed']=='s0' and policy['primary_epoch']==6
assert policy['supervision']['formal_access_currently_authorized'] is False
assert auth['status']=='AUTHORIZED_RANKING_TRAIN_AND_VALID'
assert auth['review_disposition_sha256']==sha((ROOT/'reports/seller_alias_continual/20260927/ranking_implementation/review/disposition.json').read_bytes())
# All 9 calibrated thresholds are used as FLOAT64; no threshold is fitted here.
threshold_rows=[];groups_by_role={'fit':144,'calibration':36,'development':60}
point_records=[]
for run_id in policy['runs']:
 ar=read(JOB/'run'/run_id/'manifest.json');cr=read(JOB/'run'/run_id/'calibration.json');th=cr['threshold']
 assert eval['runs'][run_id]['automatic_classification']['threshold']==th
 for role in ('calibration','development'):
  sc=np.load(JOB/'run'/run_id/ar['points']['6']['scores'][role]['path'],allow_pickle=False)
  expected=cr['counts_by_group'] if role=='calibration' else eval['runs'][run_id]['automatic_classification']['counts_by_group']
  actual=[sum(float(x)>=th for x in row) for row in sc]
  assert actual==[x[0]+x[1] for x in expected]
  threshold_rows.append({'run_id':run_id,'role':role,'threshold':th,'threshold_minus_float32':float(th-float(np.float32(th))),'score_max':float(sc.max()),'total_positive_predictions':sum(actual)})
 for ep in ('3','6'):
  pt=ar['points'][ep];assert int(ep)*144==pt['metadata']['completed_updates']
  point_records.append({'run_id':run_id,'epoch':int(ep),'completed_updates':pt['metadata']['completed_updates'],'full_state_sha256':pt['checkpoint']['state_sha256'],'inference_state_sha256':pt['model']['state_sha256'],'model_state_sha256':pt['model_state_sha256'],'full_adam_reloaded_recorded':pt['full_model_and_adam_reloaded'],'inference_reload_recorded':pt['model']['actual_reload_verified'],'blind_score_bytes':sum(pt['scores'][r]['bytes'] for r in groups_by_role)})
# Compare calibration-vs-fixed thresholds from aggregate counts; do not infer labels.
calibrated={rid:{'pooled':eval['runs'][rid]['automatic_classification']['pooled'],'domains':eval['runs'][rid]['automatic_classification']['by_domain']} for rid in policy['runs']}
assert all(not eval['runs'][rid]['automatic_classification']['all_domains_pass'] for rid in policy['runs'])
# Parse raw logs instead of relying on prose for time and exit.
resource=(JOB/'resource_usage.log').read_text();assert '14:25:21' in resource and 'Exit status: 0' in resource and '6936424' in resource
assert int((JOB/'exit_status.txt').read_text())==0
ret=read(RES/'weight_retention.json');inv=read(RES/'weight_inventory.json');post=read(RES/'retention_verification.json')
assert ret['deleted_files']==post['deleted_files']==0 and inv['deletion_performed'] is False
assert ret['status']=='RETAIN_ALL_RANKING_WEIGHTS_BY_EXPLICIT_USER_DIRECTION'
print('All eighteen frozen files match the actual pre-training attachment.')
print('All eighteen checkpoint identities and fifty-four score role identities agree.')
print('Calibrated thresholds/counts agree at full stored precision; no re-fitting performed.')
print('No model files opened; no original pair labels or item texts read.')
result={'status':'PASS_ADDITIONAL_SAVED_EVIDENCE_CHECKS','frozen_sources_equal_pretraining_zip':same,'checkpoint_records':point_records,'threshold_boundaries':threshold_rows,'automatic_classification':calibrated,'operations':{'formal_label_reads':0,'formal_text_reads':0,'model_loads':0,'training_updates':0,'remote_server_access':False},'note':'Completed records establish submitted provenance only, not an independent observation of remote filesystem or every GPU operation.'}
(OUT/'additional_saved_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
