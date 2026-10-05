"""No formal data or model: execute actual NumPy evaluation and pure comparisons.

The run module imports a Torch admission module at top level. This script loads
only its unchanged pure `comparisons` FunctionDef using AST, explicitly rather
than mocking Torch. The computation uses handwritten metric matrices.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent/'relation_memory_pilot_review'
sys.path.insert(0,str(PROJECT/'scripts'))
import step28_bge_continual_evaluate as evaluation

source=PROJECT/'scripts/step28_relation_memory_run.py'
tree=ast.parse(source.read_text(encoding='utf-8'),filename=str(source))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='comparisons')
scope={'evaluation':evaluation,'parent':evaluation.method,'ROLES':evaluation.method.ROLES}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
compare=scope['comparisons']

COLS=evaluation.metrics.COLUMNS
ROLES=evaluation.method.ROLES
ORDERS=evaluation.method.ORDERS
rank_cols=np.asarray(evaluation.RANK_COLUMNS)
cost_cols=np.asarray([COLS.index('brier'),COLS.index('log_loss')])
map_col=COLS.index('map')
ap_col=COLS.index('average_precision')
report={'python':sys.version,'numpy':np.__version__,'torch_used':False,
        'formal_inputs':False,'BGE_or_GPU':False,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'comparisons_load':'Unmodified comparisons AST function; no run-module execution or fake Torch',
        'matrix_origin':'Handwritten numeric metric arrays; not output of a trained model'}

# Deliberately interleave actual domains, rather than relying on contiguous rows.
domains=list('CAB')*20
rows={d:np.flatnonzero(np.asarray(domains)==d) for d in 'ABC'}
draws=evaluation.bootstrap_draws()
expected_draws=np.random.Generator(np.random.PCG64(20260930)).integers(0,20,size=(5000,3,20))
assert np.array_equal(draws,expected_draws)
report['bootstrap_spec']={'shape':list(draws.shape),'generator':'PCG64','seed':20260930,
                          'replicates':5000,'percentiles':[0.025,0.975],'quantile':'linear'}

rng=np.random.default_rng(105)
reference={};candidate={}
for order in ORDERS:
    for stage in (1,2,3):
        key=evaluation.point_name(order,'er',stage)
        raw=rng.uniform(.15,.8,(60,22))
        changed=raw+rng.uniform(-.05,.06,(60,22))
        reference[key]={};candidate[key]={}
        for role in ROLES:
            reference[key][role]=raw.copy()
            candidate[key][role]=changed.copy()
            if role!='raw':
                # All raw ordering/curve metrics stay identical across roles.
                nonrank=np.setdiff1d(np.arange(22),rank_cols)
                shift=.012 if role=='stage-cal' else -.018
                reference[key][role][:,nonrank]+=shift
                candidate[key][role][:,nonrank]-=shift/2

def primary(arrays,key):
    out=arrays[key]['stage-cal'].copy()
    out[:,rank_cols]=arrays[key]['raw'][:,rank_cols]
    return out

def manual_endpoint(arrays,order,endpoint,boot=False):
    def value(stage,arrival):
        domain=order[arrival-1]
        data=primary(arrays,evaluation.point_name(order,'er',stage))[rows[domain]]
        if boot:
            return data[draws[:,'ABC'.index(domain)]].mean(axis=1)
        return data.mean(axis=0)
    # Independently express the declared arrival semantics, including own stage 1.
    if endpoint=='O': out=(value(3,1)+value(3,2))/2
    elif endpoint=='N': out=(value(2,2)+value(3,3))/2
    elif endpoint=='Z': out=value(3,3)
    elif endpoint=='F_first': out=value(1,1)-value(3,1)
    elif endpoint=='F': out=((value(1,1)-value(3,1))+(value(2,2)-value(3,2)))/2
    elif endpoint=='G': out=((value(2,2)-value(1,2))+(value(3,3)-value(2,3)))/2
    elif endpoint=='final_all': out=(value(3,1)+value(3,2)+value(3,3))/3
    else: raise ValueError(endpoint)
    if endpoint in ('F_first','F','G'):
        out[...,cost_cols]*=-1
    return out

result=compare(candidate,reference,domains)
errors={}
for ep in evaluation.ENDPOINTS:
    manual_per_order=np.asarray([manual_endpoint(candidate,o,ep)-manual_endpoint(reference,o,ep)
                                for o in ORDERS])
    manual_point=manual_per_order.mean(axis=0)
    # Same draw per actual domain is reused for both methods and all arrival orders.
    manual_boot=np.mean([manual_endpoint(candidate,o,ep,True)-manual_endpoint(reference,o,ep,True)
                         for o in ORDERS],axis=0)
    intervals=np.quantile(manual_boot,[.025,.975],axis=0,method='linear')
    actual_point=np.asarray([result['delta'][ep][k]['mean'] for k in COLS])
    actual_per_order=np.asarray([[result['delta'][ep][k]['per_order'][o] for k in COLS] for o in ORDERS])
    actual_ci=np.asarray([result['delta'][ep][k]['conditional_95pct_interval'] for k in COLS]).T
    errors[ep]={'point_max_abs':float(np.max(np.abs(manual_point-actual_point))),
                'per_order_max_abs':float(np.max(np.abs(manual_per_order-actual_per_order))),
                'interval_max_abs':float(np.max(np.abs(intervals-actual_ci)))}
    assert max(errors[ep].values())<2e-14
report['all_7_endpoints_22_columns_manual_comparison']=errors

def uniform_cases():
    baseline={};current={}
    for order in ORDERS:
        for stage in (1,2,3):
            key=evaluation.point_name(order,'er',stage)
            baseline[key]={role:np.full((60,22),.5) for role in ROLES}
            current[key]={role:np.full((60,22),.5) for role in ROLES}
    return current,baseline

current,baseline=uniform_cases()
for order in ORDERS:
    for role in ROLES:
        current[evaluation.point_name(order,'er',1)][role][rows[order[0]],map_col]=.9
        for d in order[:2]:
            current[evaluation.point_name(order,'er',3)][role][rows[d],map_col]=.6
case=compare(current,baseline,domains)
assert np.isclose(case['delta']['O']['map']['mean'],.1)
assert np.isclose(case['endpoints']['relation']['raw']['F_first']['map']['mean'],.3)
assert case['worth_matched_replay']
report['own_first_endpoint']={'O_map_delta':case['delta']['O']['map']['mean'],
                             'candidate_F_first':case['endpoints']['relation']['raw']['F_first']['map']['mean'],
                             'baseline_F_first':case['endpoints']['logit0.1']['raw']['F_first']['map']['mean'],
                             'six_observation_checks':case['continuation_checks']}

# Independent current/old grouping prevents a Z deterioration from being confused
# with an O deterioration: change only ABC's newest domain C at the final endpoint.
for role in ROLES:
    current['ABC_er_stage3'][role][rows['C'],map_col]=.49
latest=compare(current,baseline,domains)
assert latest['delta']['O']['map']['mean']>0
assert latest['delta']['Z']['map']['mean']<0
assert not latest['worth_matched_replay']
report['latest_only_deterioration_rejected']={
    'O_map_delta':latest['delta']['O']['map']['mean'],
    'N_map_delta':latest['delta']['N']['map']['mean'],
    'Z_map_delta':latest['delta']['Z']['map']['mean'],
    'six_observation_checks':latest['continuation_checks']}

# Positive observed O gain with a crossing-zero conditional interval must remain
# eligible under this explicitly observational continuation rule.
current,baseline=uniform_cases()
delta=np.array([-.25]*9+[.25]*11)
for order in ORDERS:
    for role in ROLES:
        for d in order[:2]:
            current[evaluation.point_name(order,'er',3)][role][rows[d],map_col]+=delta
crossing=compare(current,baseline,domains)
assert crossing['worth_matched_replay']
assert not crossing['old_map_conditional_positive']
assert crossing['delta']['O']['map']['conditional_95pct_interval'][0]<0
report['observation_not_significance_gate']={
    'O_map_delta':crossing['delta']['O']['map']['mean'],
    'O_map_interval':crossing['delta']['O']['map']['conditional_95pct_interval'],
    'worth_matched_replay':crossing['worth_matched_replay'],
    'old_map_conditional_positive':crossing['old_map_conditional_positive'],
    'automatic_followup':crossing['automatic_followup'],
    'six_observation_checks':crossing['continuation_checks'],
}

# AP is a separate lower-level condition, not a synonym for MAP.
for role in ROLES:
    current['ABC_er_stage3'][role][rows['C'],ap_col]=.49
ap_failure=compare(current,baseline,domains)
assert not ap_failure['worth_matched_replay']
assert ap_failure['continuation_checks']['O_map_positive']
assert not ap_failure['continuation_checks']['Z_AP_non_decrease']
report['AP_gate_separate_from_MAP']=ap_failure['continuation_checks']

report['status']='PASS_NUMPY_METRIC_AGGREGATION_ONLY'
(ROOT/'endpoint_metric_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
