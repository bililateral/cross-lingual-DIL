#!/usr/bin/env python3
"""Independent saved-evidence audit; never imports project modules or formal labels.

Usage: python audit_search.py INPUT_ROOT OUTPUT_DIR
Only uses stdlib and NumPy for saved metric reductions.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sqlite3
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
reads = {}
checks = []
ARMS = ('S', 'S_pos', 'H_pos')
S0 = dict(A=0.49413171588434307, B=0.5058704368257183)

def note(p, method, purpose):
    p = Path(p)
    rel = str(p.relative_to(root))
    reads.setdefault(rel, set()).add((method, purpose))

def jread(rel, purpose):
    p = root / rel
    note(p, 'machine_json_fields', purpose)
    return json.loads(p.read_text())

def check(name, good, details=None):
    checks.append(dict(check=name, passed=bool(good), details=details))

def record(p):
    p = Path(p)
    note(p, 'machine_bytes_sha256', 'identity binding')
    b = p.read_bytes()
    return dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest())

def domain_maps(run_id, arm):
    rel = f'result/job/{run_id}'
    col = jread(rel+'/evaluation/collected.json', 'group/domain order and B/raw binding')
    partition = jread(rel+'/run/partition.json', 'fixed development identity/order')
    check(run_id+':group_order', col['group_ids'] == [r['group_uid'] for r in partition['development']]
          and col['domains'] == [r['domain'] for r in partition['development']])
    bind = col['points']['B_'+arm]['raw']['matrix']
    matrix_rel = rel+'/evaluation/'+bind['path']
    check(run_id+':B_raw_binding', bind['path'] == 'B_'+arm+'_raw_metrics.json'
          and record(root/matrix_rel) == {k:bind[k] for k in ('bytes','sha256')})
    matrix = jread(matrix_rel, 'reduce saved group MAP in original order, no labels')
    values = np.asarray(matrix['values'], dtype=np.float64)
    check(run_id+':matrix_shape', values.shape == (40,22) and matrix['columns'][12] == 'map')
    means = {}
    for domain in 'AB':
        indices = np.array([i for i,d in enumerate(col['domains']) if d == domain])
        check(run_id+':twenty_'+domain, len(indices)==20)
        means[domain] = float(np.sum(values[indices,12],dtype=np.float64)/20)
    return means

search = jread('result/job/search.json', '36 search slots, values and identity')
selected = jread('result/job/selected.json', 'frozen unique selected parameters')
summary = jread('result/analysis/summary.json', 'direction gate and selected fields only')
replications = jread('result/job/replications.json', 'nine blind completion and exact identities')
check('arms', set(search)==set(ARMS))

db_path = root/'result/job/studies.sqlite3'
note(db_path,'machine_sqlite_read_only','studies, trial states/values/attrs/params/distributions/order')
db = sqlite3.connect(db_path.as_uri()+'?mode=ro',uri=True)
db.row_factory = sqlite3.Row
study_rows = [dict(r) for r in db.execute('select * from studies')]
db_trials = [dict(r) for r in db.execute('select t.*,s.study_name from trials t join studies s on t.study_id=s.study_id order by t.trial_id')]
version = [dict(r) for r in db.execute('select * from version_info')]
directions = [dict(r) for r in db.execute('select * from study_directions')]
check('sqlite_three_independent_studies', [r['study_name'] for r in study_rows]==list(ARMS)
      and len({r['study_id'] for r in study_rows})==3)
check('sqlite_version_4_6_0', version[0]['library_version']=='4.6.0')
check('sqlite_directions',len(directions)==3 and all(r['direction']=='MAXIMIZE' and r['objective']==0 for r in directions))
check('sqlite_36_complete',len(db_trials)==36 and all(r['state']=='COMPLETE' for r in db_trials))
check('sqlite_no_intermediate_values',db.execute('select count(*) from trial_intermediate_values').fetchone()[0]==0)
expected_order = [(arm,n) for n in range(12) for arm in ARMS[n%3:]+ARMS[:n%3]]
check('sqlite_rotating_logical_order',[(r['study_name'],r['number']) for r in db_trials]==expected_order)
db_by_key = {(r['study_name'],r['number']):r for r in db_trials}
rows_out=[]
selections=[]
raw_hashes=[]

for arm in ARMS:
    rows=search[arm]
    check(arm+':12_slots',len(rows)==12 and [r['number'] for r in rows]==list(range(12)))
    for row in rows:
        n=row['number']; spec=row['spec']; run_id=f'search_{n:02}_{arm}_s0'
        means=domain_maps(run_id,arm)
        completion=jread(f'result/job/{run_id}/completion.json','own successful feedback and default anchor flag')
        evaluation=jread(f'result/job/{run_id}/evaluation/evaluation.json','own domain feedback fields only')
        u=min(means[d]-S0[d] for d in 'AB')
        avg=sum(means[d]-S0[d] for d in 'AB')/2
        check(run_id+':own_feedback_exact',row['domains']==means==completion['domains']==evaluation['domains']
              and row['value']==u and row['state']=='COMPLETE' and completion['status']=='SEARCH_COMPLETE')
        check(run_id+':identity', completion['trial']==spec and spec['arm']==arm and spec['phase']=='search'
              and spec['number']==n and spec['seed']==0)
        check(run_id+':access',completion['access']==dict(train=1,valid=1,heldout=0,owners=0))
        if n==0:
            check(run_id+':default_anchor_record',spec['lambda_rank']==spec['lr_multiplier']==1
                  and completion['anchor_compatible'] is True)
        t=db_by_key[(arm,n)]; tid=t['trial_id']
        val=[dict(r) for r in db.execute('select * from trial_values where trial_id=?',(tid,))]
        attrs={r['key']:json.loads(r['value_json']) for r in db.execute('select * from trial_user_attributes where trial_id=?',(tid,))}
        params={r['param_name']:dict(r) for r in db.execute('select * from trial_params where trial_id=?',(tid,))}
        check(run_id+':sqlite_feedback_exact',len(val)==1 and val[0]['value']==u and val[0]['value_type']=='FINITE'
              and attrs['domains']==means and attrs['trial']==spec)
        identity=attrs['completion_identity']; actual_identity=record(root/f'result/job/{run_id}/completion.json')
        check(run_id+':sqlite_completion_sha',actual_identity=={k:identity[k] for k in ('bytes','sha256')}
              and Path(identity['path']).parent.name==run_id)
        for key,lo,hi in [('lambda_rank',.25,2.),('lr_multiplier',.5,2.)]:
            p=params[key];dist=json.loads(p['distribution_json'])
            check(run_id+':sqlite_'+key,p['param_value']==spec[key] and lo<=p['param_value']<=hi
                  and dist==dict(name='FloatDistribution',attributes=dict(step=None,low=lo,high=hi,log=True)))
        raw=record(root/f'result/job/{run_id}/run/scores/B_{arm}_raw.npy')
        raw_hashes.append(raw['sha256'])
        rows_out.append(dict(arm=arm,trial=n,lambda_rank=spec['lambda_rank'],lr_multiplier=spec['lr_multiplier'],
                             A_MAP=means['A'],B_MAP=means['B'],U=u,mean_improvement=avg,
                             own_completion_sha256=actual_identity['sha256'],B_raw_score_sha256=raw['sha256']))
    winner=max(rows,key=lambda r:(min(r['domains'][d]-S0[d] for d in 'AB'),sum(r['domains'][d]-S0[d] for d in 'AB')/2,-r['number']))
    check(arm+':exact_frozen_selection',selected[arm]==winner==summary['selected'][arm])
    selections.append(dict(arm=arm,trial=winner['number'],lambda_rank=winner['spec']['lambda_rank'],
                           lr_multiplier=winner['spec']['lr_multiplier'],U=winner['value']))
check('raw_B_scores_36_distinct',len(set(raw_hashes))==36)
check('twelve_parameter_pairs_shared_by_all_arms',all(len({(search[a][n]['spec']['lambda_rank'],search[a][n]['spec']['lr_multiplier']) for a in ARMS})==1 for n in range(12)))
check('no_repeated_parameter_pair_within_arm',all(len({(r['spec']['lambda_rank'],r['spec']['lr_multiplier']) for r in search[a]})==12 for a in ARMS))

# Explanatory check, not a fresh Optuna execution. Fixed 4.6.0 default_gamma:
# https://optuna.readthedocs.io/en/v4.6.0/_modules/optuna/samplers/_tpe/sampler.html
# The external reference confirms gamma=min(ceil(.1*n),25) and rank-based splits.
splits=[]
for n in range(4,12):
    k=min(math.ceil(.1*n),25)
    good={a:sorted(r['number'] for r in sorted(search[a][:n],key=lambda r:r['value'],reverse=True)[:k]) for a in ARMS}
    check(f'tpe_{n}:identical_good_membership',len({tuple(v) for v in good.values()})==1)
    splits.append(dict(proposed_trial=n,prior_complete=n,n_good=k,good_trials_by_arm=good))

anchor_path=root/'result/job/search_00_S_s0/evaluation/B_S_raw_metrics.json'
historical_s_map_sha='9caa6d26f5f843563140dbacde20599e8a912dc603043cd32d206eb27e83a4f1'
check('S0_bound_historical_group_matrix_hash',record(anchor_path)==dict(bytes=12880,sha256=historical_s_map_sha))
check('S0_full_precision_reduction',domain_maps('search_00_S_s0','S')==S0)
recovery=jread('result/job/recovery.json','bound pending restart exact parameters')
bound=jread('context/reports/documentation/20261021/optuna_reboot/recovery_bound.json','recovery approval and 25 prior completion hashes')
check('reboot_pending_not_redrawn',recovery['pending']==bound['pending']==search['S'][8]['spec'])
prior_bind={r['path']:r for r in bound['files']}
preserved=[]
for arm,n in expected_order[:25]:
    run_id=f'search_{n:02}_{arm}_s0'
    # Unchanged completion binding supports the reused observed results.
    b=prior_bind[run_id+'/completion.json']
    actual=record(root/f'result/job/{run_id}/completion.json')
    passed=actual=={k:b[k] for k in ('bytes','sha256')}
    check(run_id+':preserved_before_reboot',passed)
    preserved.append(dict(run_id=run_id,completion_sha256=actual['sha256'],matches_pre_reboot_binding=passed))

check('nine_prescribed_replications',len(replications)==9 and {(r['trial']['arm'],r['trial']['seed']) for r in replications}=={(a,s) for a in ARMS for s in (1,2,3)})
rep_means={}
expected_rep_order=[(a,s) for s in (1,2,3) for a in ARMS[(s-1)%3:]+ARMS[:(s-1)%3]]
check('replication_rotating_order',[(r['trial']['arm'],r['trial']['seed']) for r in replications]==expected_rep_order)
rep_rows=[]
for r in replications:
    t=r['trial'];a=t['arm'];s=t['seed'];run_id=f'replication_{s-1:02}_{a}_s{s}'
    expected_roots=dict(initialization=2026101900+s,schedule=2026102900+s,memory=2026103900+s,
                        partition=20260918,diagnostic=20261008,statistics=20260930)
    check(run_id+':frozen_config_seed',t['lambda_rank']==selected[a]['spec']['lambda_rank']
          and t['lr_multiplier']==selected[a]['spec']['lr_multiplier'] and t['phase']=='replication'
          and t['number']==s-1 and t['roots']==expected_roots and t['rng_order']=='ABC')
    completion=jread(f'result/job/{run_id}/completion.json','immutable own blind completion')
    check(run_id+':blind_exit',r['status']==completion['status']=='REPLICATION_BLIND_COMPLETE'
          and r['access']==completion['access']==dict(train=1,valid=0,heldout=0,owners=0)
          and r['child_exit_code']==0 and r['resources_released'] is True and completion['trial']==t)
    check(run_id+':completion_sha',record(root/f'result/job/{run_id}/completion.json')=={k:r['completion_identity'][k] for k in ('bytes','sha256')})
    rep_means[a,s]=domain_maps(run_id,a)
    rep_rows.append(dict(arm=a,seed=s,**rep_means[a,s]))
gates={};delta_rows=[]
for a in ('S_pos','H_pos'):
    cells=[]
    for s in (1,2,3):
        for d in 'AB':
            delta=rep_means[a,s][d]-rep_means['S',s][d]
            cells.append(delta)
            delta_rows.append(dict(candidate=a,seed=s,domain=d,candidate_MAP=rep_means[a,s][d],S_tuned_MAP=rep_means['S',s][d],delta=delta,positive=delta>0))
    gates[a]=dict(positive_cells=sum(x>0 for x in cells),total_cells=6,passed=all(x>0 for x in cells),
                  mean_A_delta=float(np.mean([rep_means[a,s]['A']-rep_means['S',s]['A'] for s in (1,2,3)])),
                  mean_B_delta=float(np.mean([rep_means[a,s]['B']-rep_means['S',s]['B'] for s in (1,2,3)])))
    check(a+':six_cell_gate',gates[a]['passed']==summary['direction_gate'][a])

def csvout(name, rows):
    with (out/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

csvout('search_independent.csv',rows_out)
csvout('replication_map_independent.csv',rep_rows)
csvout('six_cell_gate_independent.csv',delta_rows)
(out/'tpe_partition_evidence.json').write_text(json.dumps(dict(method='rank membership only; no Optuna execution',official_version='4.6.0',source='https://optuna.readthedocs.io/en/v4.6.0/_modules/optuna/samplers/_tpe/sampler.html',splits=splits),ensure_ascii=False,indent=2)+'\n')
result=dict(scope='external review sandbox; saved public evidence only; no project code imported, no model or formal labels read',
            reference=S0,selected=selections,six_cell_gate=gates,checks=len(checks),passed=sum(c['passed'] for c in checks),
            failures=[c for c in checks if not c['passed']],sqlite_version=version,sqlite_studies=study_rows,
            raw_score_distinct_count=len(set(raw_hashes)),pre_reboot_preserved=preserved,
            limitations=['No installed Optuna in review environment; no sampler execution or package installation.',
                         'No historical raw-score copy provided for independent six-array original-vs-anchor equality.',
                         'S0 historical group metric matrix is independently bound by its frozen historical SHA256.'])
(out/'search_audit_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(out/'search_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
read_rows=[]
for rel,methods in sorted(reads.items()):
    b=(root/rel).read_bytes()
    for method,purpose in sorted(methods):
        read_rows.append(dict(path=rel,method=method,purpose=purpose,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
csvout('machine_read_scope.csv',read_rows)
db.close()
print(json.dumps({k:result[k] for k in ('checks','passed','failures','selected','six_cell_gate','raw_score_distinct_count')},ensure_ascii=False,indent=2))
