"""Check saved aggregates and pairing; no labels, models, or resampling."""
from pathlib import Path
import hashlib
import json
import math
import signal
import time
import numpy as np

started = time.monotonic()
signal.alarm(60)
root = Path('/home/yongpeng/cross-lingual/reports/documentation/20261031/comparison_continue/formal')
read = lambda path: json.loads(path.read_text())
saved = read(root / 'confirmation/result.json')
analysis = read(root / 'analysis/result.json')
gate = read(root.parent / 'formal_gate.json')
done = read(root / 'completion.json')
assert done['budget']['used']['analysis'] + 60 <= done['budget']['limits']['analysis']
assert not done['budget']['failed'] and not done['budget']['active']
assert len(done['budget']['completed']) == 21 and all(r['success'] for r in done['budget']['completed'])
assert saved['status'] == 'ALL_COMPARISON_METRICS_COMPLETE'
assert analysis['status'] == 'FIXED_COMPARISON_ANALYSIS_COMPLETE'
assert saved['forwards'] == 8640 and len(set(saved['groups'])) == 120
assert saved['domains'] == [d for d in 'ABC' for _ in range(40)]
assert saved['sources'] == gate['sources']
methods = ['H_pos_tuned', 'S_tuned', 'L2R_text_fixed', 'ER_GCR_group_fixed']
expected = {f'{order}_s{seed}_{method}' for order in ['ABC','BCA','CAB'] for seed in [4,5,6] for method in methods}
assert set(saved['runs']) == expected
columns = analysis['columns']
terms = {'O': {'3_1': .5, '3_2': .5}, 'F': {'1_1': .5, '3_1': -.5, '2_2': .5, '3_2': -.5},
         'A_acq': {'1_1': .5, '2_2': .5}, 'N': {'2_2': .5, '3_3': .5}, 'Z': {'3_3': 1.},
         'F_first': {'1_1': 1., '3_1': -1.}, 'final_mean': {'3_1': 1/3, '3_2': 1/3, '3_3': 1/3}}
cell_names = {'1_1','2_1','2_2','3_1','3_2','3_3'}
max_error = 0.
comparisons_checked = 0
matrices = 0
na_counts = {name: 0 for name in columns}

def same(actual, expected):
    global max_error
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b))
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.any():
        error = float(np.max(np.abs(a[mask] - b[mask])))
        max_error = max(max_error, error)
        assert error < 1e-10, error

def verified(rec):
    p = root / 'confirmation' / rec['path']
    raw = p.read_bytes()
    assert len(raw) == rec['bytes'] and hashlib.sha256(raw).hexdigest() == rec['sha256']
    return p

aggregates = {}
for identity, cells in saved['runs'].items():
    order, seed, method = identity.split('_', 2)
    assert set(cells) == cell_names
    roles = {role: {} for role in ['raw','stage-cal','first-cal']}
    for cell, point in cells.items():
        stage, position = map(int, cell.split('_'))
        domain = order[position - 1]
        assert point['domain'] == domain
        assert point['ids'] == [uid for uid, d in zip(saved['groups'], saved['domains']) if d == domain]
        score_arrays = {}
        for role, rec in point['scores'].items():
            values = np.load(verified(rec), allow_pickle=False)
            assert values.shape == (40,378) and np.isfinite(values).all()
            score_arrays[role] = values
        raw = score_arrays['raw']
        idx = np.argsort(raw, axis=1, kind='stable')
        ordered_raw = np.take_along_axis(raw, idx, axis=1)
        for role in ['stage-cal','first-cal']:
            ordered = np.take_along_axis(score_arrays[role], idx, axis=1)
            assert np.array_equal(np.diff(ordered_raw,axis=1)==0, np.diff(ordered,axis=1)==0)
            assert (np.diff(ordered,axis=1)>=0).all()
        for role, records in point['metrics'].items():
            matrix = read(verified(records['matrix']))
            values = np.asarray(matrix['values'], dtype=float)
            queries = np.load(verified(records['queries']), allow_pickle=False)
            assert matrix['columns'] == columns and values.shape == (40,22) and queries.shape == (40,28,10)
            assert all(x['tp']+x['fn']==20 and x['tn']+x['fp']==358 for x in matrix['counts'])
            same(values[:,12:], queries.mean(axis=1))
            mean = values.mean(axis=0)
            same(mean, analysis['per_run_actual_domain_cell_points'][f'{identity}_{role}_{cell}']['values'])
            for j, name in enumerate(columns):na_counts[name] += int(np.isnan(values[:,j]).sum())
            roles[role][cell] = mean
            matrices += 1
    for role, cells in roles.items():
        fields = {field: sum(weight*cells[cell] for cell,weight in weights.items()) for field,weights in terms.items()}
        for field, value in fields.items():same(value, analysis['per_run_endpoint_points'][identity+'_'+role][field])
        same(fields['F'], fields['A_acq']-fields['O'])
        aggregates.setdefault(method,{}).setdefault(role,[]).append(fields)
absolute = {}
for method, roles in aggregates.items():
    absolute[method] = {}
    for role, runs in roles.items():
        assert len(runs) == 9
        absolute[method][role] = {field: np.mean([row[field] for row in runs],axis=0) for field in terms}
        for field, value in absolute[method][role].items():same(value,analysis['absolute_endpoint_points'][method][role][field])
for comparison, roles in analysis['comparisons'].items():
    a,b = comparison.split('-')
    for role, fields in roles.items():
        for field, metrics in fields.items():
            difference = absolute[a][role][field]-absolute[b][role][field]
            for j, name in enumerate(columns):
                stat = metrics[name]
                same([stat['mean']], [difference[j]])
                expected_q = [.0125,.9875] if a=='H_pos_tuned' and b in methods[2:] and field in ['O','F'] else [.025,.975]
                assert stat['quantiles'] == expected_q
                interval = stat['conditional_interval']
                if math.isnan(difference[j]):assert interval is None
                else:assert interval is not None and len(interval)==2 and all(math.isfinite(x) for x in interval) and interval[0]<=interval[1]
                comparisons_checked += 1
checks = {}
for b in methods[2:]:
    v=analysis['comparisons']['H_pos_tuned-'+b]['raw']
    checks['H_pos_tuned-'+b] = dict(old_retention=v['O']['map']['conditional_interval'][0]>0,
        forgetting=v['F']['map']['conditional_interval'][1]<0,
        acquisition_point=v['N']['map']['mean']>=0, final_domain_point=v['Z']['map']['mean']>=0)
assert analysis['primary'] == dict(passed=all(x for row in checks.values() for x in row.values()),checks=checks)
elapsed = time.monotonic()-started
print(json.dumps(dict(status='SAVED_FIELDS_PAIRING_ARITHMETIC_VERIFIED',matrices=matrices,score_arrays=648,
    query_arrays=648,statistic_entries_checked=comparisons_checked,max_absolute_arithmetic_error=max_error,
    primary=analysis['primary'],na_counts=na_counts,elapsed_seconds=elapsed,billed_seconds=math.ceil(elapsed),
    scope='Saved arrays/metrics, identities, means, endpoint formulas, contrasts and decision only; interval generation reused, not independently resampled.',
    labels_opened=0,models_loaded=0,training=0,bootstrap_replicates_added=0),ensure_ascii=False))
