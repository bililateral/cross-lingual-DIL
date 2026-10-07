#!/usr/bin/env python3
"""Read-only independent audit of saved group metrics, never labels or models.

This implementation imports no project code. It resamples each stage's actual
domain groups first, combines endpoint formulas second, and then averages the
three fixed orders. Identical domain/group draws are reused for both methods,
every role, every stage and every order. The observed result is trial 0, with
each of the 20 domain groups included exactly once. Trials 1..5000 are PCG64
conditional bootstrap replicates with the frozen seed and linear quantiles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


ORDERS = ('ABC', 'BCA', 'CAB')
ROLES = ('raw', 'stage-cal', 'first-cal', 'primary')
ENDPOINTS = ('O', 'N', 'Z', 'F_first', 'F', 'G', 'final_all', 'A2')
METRICS = ('average_precision', 'trapezoidal_pr_auc', 'roc_auc',
           'recall_at_fpr_1pct', 'brier', 'log_loss', 'precision', 'recall',
           'f1', 'specificity', 'balanced_accuracy', 'mcc', 'map', 'mrr',
           'recall_at_1', 'recall_at_3', 'recall_at_5', 'recall_at_10',
           'ndcg_at_1', 'ndcg_at_3', 'ndcg_at_5', 'ndcg_at_10')
RANK = [i for i, m in enumerate(METRICS)
        if m in METRICS[:4] or m in METRICS[12:]]
READS: dict[str, dict] = {}


def read_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    READS[str(path)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return data


def read_json(path: Path):
    return json.loads(read_bytes(path))


def check_record(path: Path, record: dict):
    data = read_bytes(path)
    assert len(data) == record['bytes'], path
    assert hashlib.sha256(data).hexdigest() == record['sha256'], path
    return data


def matrix(root: Path, record: dict) -> np.ndarray:
    path = root / record['path']
    check_record(path, record)
    a = np.load(path, allow_pickle=False)
    assert a.shape == (60, 22) and a.dtype == np.dtype('<f8'), path
    assert np.isfinite(a).all(), path
    return a


def summary(ordered_trials: np.ndarray) -> dict:
    # shape: fixed order x (observed + 5000 replicates) x metric.
    across_orders = ordered_trials.mean(axis=0)
    ci = np.quantile(across_orders[1:], [.025, .975], axis=0, method='linear')
    return {m: {'mean': float(across_orders[0, k]),
                'per_order': {o: float(ordered_trials[i, 0, k])
                              for i, o in enumerate(ORDERS)},
                'conditional_95pct_interval': ci[:, k].tolist()}
            for k, m in enumerate(METRICS)}


def endpoint_formula(stage_domain_means: dict, endpoint: str) -> np.ndarray:
    """Indices in this dict mean stage and arrival, never actual-domain rows."""
    s = stage_domain_means
    if endpoint == 'O':
        z = (s[3, 1] + s[3, 2]) / 2
    elif endpoint == 'A2':
        z = s[2, 2]
    elif endpoint == 'Z':
        z = s[3, 3]
    elif endpoint == 'N':
        z = (s[2, 2] + s[3, 3]) / 2
    elif endpoint == 'final_all':
        z = (s[3, 1] + s[3, 2] + s[3, 3]) / 3
    elif endpoint == 'F_first':
        z = s[1, 1] - s[3, 1]
    elif endpoint == 'F':
        z = ((s[1, 1] - s[3, 1]) + (s[2, 2] - s[3, 2])) / 2
    elif endpoint == 'G':
        z = ((s[2, 2] - s[1, 2]) + (s[3, 3] - s[2, 3])) / 2
    else:
        raise ValueError(endpoint)
    z = z.copy()
    if endpoint in ('F_first', 'F', 'G'):
        z[:, METRICS.index('brier')] *= -1
        z[:, METRICS.index('log_loss')] *= -1
    return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    start = time.perf_counter()
    root = args.input_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    current = root / 'reports/seller_alias_continual/20261007/function_memory_result'
    eroot = current / 'job/evaluation'
    policy = read_json(current / 'source/schema/step28_function_memory_policy.json')
    read_bytes(current / 'review/context.zh.md')
    read_bytes(root / 'docs/SELLER_ALIAS_FUNCTION_MEMORY_RESULT.zh.md')
    read_bytes(current / 'source/docs/SELLER_ALIAS_FUNCTION_MEMORY_PILOT.zh.md')
    # Source read solely to verify inherited definitions, never executed/imported.
    for name in ('step28_bge_continual_evaluate.py', 'step28_function_memory_run.py',
                 'step28_continual_population_evaluate.py'):
        read_bytes(current / 'source/scripts' / name)
    coll = read_json(eroot / 'collected.json')
    reported = read_json(eroot / 'evaluation.json')
    reference_root = root / policy['reference']['local_root']
    reference_collections = []
    for rec in policy['reference']['collections']:
        reference_collections.append(json.loads(check_record(reference_root / rec['path'], rec)))
    assert coll['metric_columns'] == list(METRICS)
    assert len(coll['group_ids']) == len(set(coll['group_ids'])) == 60
    assert len(coll['domains']) == 60
    for c in reference_collections:
        for key in ('group_ids', 'domains', 'metric_columns'):
            assert c[key] == coll[key], ('reference alignment', key)
    ids = coll['group_ids']
    domains = coll['domains']
    per_domain = {d: [uid for uid, domain in zip(ids, domains) if domain == d]
                  for d in 'ABC'}
    assert all(len(v) == 20 for v in per_domain.values())
    row_by_uid = {uid: i for i, uid in enumerate(ids)}
    matrices = {'function_memory': {}, 'logit0.1': {}}
    count_records = {'function_memory': {}, 'logit0.1': {}}
    matrix_sets = {'function_memory': 0, 'logit0.1': 0}
    for arm in matrices:
        for o in ORDERS:
            matrices[arm][o] = {}
            count_records[arm][o] = {}
            for st in (1, 2, 3):
                if arm == 'function_memory':
                    collection, rr, name = coll, eroot, f'{o}_function_memory_stage{st}'
                elif st == 1:
                    collection, rr, name = reference_collections[1], reference_root / 'reference', f'{o}_shared'
                else:
                    collection, rr, name = reference_collections[0], reference_root, f'{o}_logit_tenth_stage{st}'
                role_records = collection['points'][name]
                assert set(role_records) == set(ROLES[:3]), name
                values, counts = {}, {}
                for role in ROLES[:3]:
                    values[role] = matrix(rr, role_records[role]['matrix'])
                    counts[role] = json.loads(check_record(rr / role_records[role]['counts']['path'], role_records[role]['counts']))
                    assert len(counts[role]) == 60
                    for cnt in counts[role]:
                        assert all(type(cnt[k]) is int and cnt[k] >= 0 for k in ('tp','fp','fn','tn'))
                        assert cnt['tp'] + cnt['fn'] == 20 and cnt['fp'] + cnt['tn'] == 358
                    matrix_sets[arm] += 1
                for role in ROLES[1:3]:
                    assert np.array_equal(values[role][:, RANK], values['raw'][:, RANK])
                values['primary'] = values['stage-cal'].copy()
                values['primary'][:, RANK] = values['raw'][:, RANK]
                matrices[arm][o][st] = values
                count_records[arm][o][st] = counts
    initial = coll['points']['initial']['raw']
    initial_values = matrix(eroot, initial['matrix'])
    initial_counts = json.loads(check_record(eroot / initial['counts']['path'], initial['counts']))
    assert len(initial_counts) == 60
    matrix_sets['function_memory'] += 1
    assert matrix_sets == {'function_memory': 28, 'logit0.1': 27}

    shared = []
    for o in ORDERS:
        for role in ROLES:
            assert np.array_equal(matrices['function_memory'][o][1][role], matrices['logit0.1'][o][1][role])
            shared.append(f'{o}/{role}: all 60 x 22 values identical')

    # Domain d controls the bootstrap axis regardless of its arrival in an order.
    rng = np.random.Generator(np.random.PCG64(20260930))
    draws = rng.integers(0, 20, size=(5000, 3, 20))
    sampled_indices = {}
    for di, d in enumerate('ABC'):
        local_rows = np.array([row_by_uid[uid] for uid in per_domain[d]])
        sampled_indices[d] = local_rows[np.vstack((np.arange(20), draws[:, di]))]
    endpoint_trials, summaries = {}, {}
    for arm, order_matrices in matrices.items():
        endpoint_trials[arm], summaries[arm] = {}, {}
        for role in ROLES:
            order_endpoints = {ep: [] for ep in ENDPOINTS}
            for o, stage_matrices in order_matrices.items():
                sampled = {(st, arrival): stage_matrices[st][role][sampled_indices[domain]].mean(axis=1)
                           for st in (1, 2, 3) for arrival, domain in enumerate(o, 1)}
                for ep in ENDPOINTS:
                    order_endpoints[ep].append(endpoint_formula(sampled, ep))
            endpoint_trials[arm][role] = {ep: np.stack(ts) for ep, ts in order_endpoints.items()}
            summaries[arm][role] = {ep: summary(ts) for ep, ts in endpoint_trials[arm][role].items()}
    deltas = {ep: summary(endpoint_trials['function_memory']['primary'][ep] -
                          endpoint_trials['logit0.1']['primary'][ep]) for ep in ENDPOINTS}
    comparisons = []
    def compare_summary(found, expected, prefix):
        for metric in METRICS:
            one, two = found[metric], expected[metric]
            comparisons.append((prefix + '/' + metric + '/mean', one['mean'], two['mean']))
            for o in ORDERS:
                comparisons.append((prefix + '/' + metric + '/per_order/' + o,
                                    one['per_order'][o], two['per_order'][o]))
            for k in (0, 1):
                comparisons.append((prefix + '/' + metric + '/ci/' + str(k),
                                    one['conditional_95pct_interval'][k], two['conditional_95pct_interval'][k]))
    for arm in matrices:
        for role in ROLES:
            for ep in ENDPOINTS:
                compare_summary(summaries[arm][role][ep], reported['endpoints'][arm][role][ep], arm+'/'+role+'/'+ep)
    for ep in ENDPOINTS:
        compare_summary(deltas[ep], reported['delta'][ep], 'delta/'+ep)
    assert len(comparisons) == 9504
    differences = [(path, abs(x-y)) for path, x, y in comparisons]
    biggest = max(differences, key=lambda x:x[1])
    mismatches = [(path,x,y) for path,x,y in comparisons if abs(x-y)>1e-12]

    def lo(ep, metric):
        return deltas[ep][metric]['conditional_95pct_interval'][0]
    acceptance = {
        'O_map_lower_positive': lo('O','map') > 0,
        'final_all_map_observed_positive': deltas['final_all']['map']['mean'] > 0,
        'A2_map_lower_ge_minus_point01': lo('A2','map') >= -.01,
        'Z_map_lower_ge_minus_point01': lo('Z','map') >= -.01,
        'O_AP_lower_ge_minus_point01': lo('O','average_precision') >= -.01,
    }
    assert acceptance == reported['continuation_checks']
    assert all(acceptance.values()) == reported['development_criteria_pass']
    assert (lo('O','map') > 0) == reported['old_map_conditional_positive']
    assert reported['automatic_followup'] is False
    key_results = {ep:{m:{'candidate':summaries['function_memory']['primary'][ep][m],
                         'reference':summaries['logit0.1']['primary'][ep][m],
                         'delta':deltas[ep][m]}
                    for m in ('map','average_precision','brier','log_loss')}
                   for ep in ENDPOINTS}
    stage_domain = {}
    for arm in matrices:
        stage_domain[arm] = {}
        for o in ORDERS:
            stage_domain[arm][o] = {}
            for st in (1, 2, 3):
                stage_domain[arm][o][str(st)] = {}
                for d in 'ABC':
                    rows = [row_by_uid[uid] for uid in per_domain[d]]
                    stage_domain[arm][o][str(st)][d] = {
                        m: float(matrices[arm][o][st]['primary'][rows, METRICS.index(m)].mean())
                        for m in METRICS}
    result = {
        'status': 'PASS' if not mismatches else 'FAIL',
        'scope': 'Saved group matrices/counts only; no truth/text/model/Memory/project execution',
        'matrix_sets': matrix_sets,
        'group_alignment': {'unique_actual_groups':60, 'per_domain':{d:len(v) for d,v in per_domain.items()},
                            'identical_id_domain_metric_order_across_collections':True,
                            'bootstrap_pairing':'same actual group draws across all methods/orders/stages/roles'},
        'shared_first':shared,
        'statistics': {'metric_entries':len(comparisons)//6, 'numeric_values':len(comparisons),
                       'tolerance':1e-12, 'mismatch_count':len(mismatches),
                       'max_absolute_error':biggest[1], 'max_error_path':biggest[0]},
        'acceptance':acceptance, 'passed':sum(acceptance.values()), 'total':len(acceptance),
        'key_results':key_results, 'stage_domain_metrics':stage_domain,
        'environment': {'python':sys.version, 'numpy':np.__version__, 'platform':platform.platform()},
        'elapsed_seconds':time.perf_counter()-start,
        'limitations':[
            'No independent re-derivation of curve/retrieval metrics from per-pair labels; none available/used.',
            'Conditional intervals treat the 60 groups as sampling units and the three orders as fixed.',
            'Intervals exclude refit randomness, development selection, and real-market validity.',
            'First-stage identities are checked from saved matrices, not fresh model execution.',
        ],
    }
    recomputed = {'endpoints':summaries,'delta':deltas,'continuation_checks':acceptance}
    for filename, value in [('result.json',result), ('all_recomputed_statistics.json',recomputed),
                            ('input_read_manifest.json',READS)]:
        (out/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','matrix_sets','statistics','acceptance','passed','total','elapsed_seconds')},ensure_ascii=False,indent=2))
    for ep in ENDPOINTS:
        for metric in ('map','average_precision'):
            print(ep,metric,json.dumps(key_results[ep][metric],ensure_ascii=False))
    if mismatches:
        print('MISMATCHES',json.dumps(mismatches[:30]))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
