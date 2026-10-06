#!/usr/bin/env python3
"""Independent audit of saved group metrics; no project imports or model access.

Contract implementation: add endpoint coefficients to 60 per-group rows, average
the three fixed orders, then resample whole groups within each actual domain.
Only stdlib and NumPy are used. Pair labels/scores are not in this audit package,
so AP/MAP/other ranking metric *generation* is explicitly outside this check.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np


WORK = Path('/workspace/scratch/f4d639b3b473')
PACKAGE = WORK / 'relation_memory_result_review'
RESULT = PACKAGE / 'reports/seller_alias_continual/20261006/relation_result'
OUT = WORK / 'result_metrics_audit'
EVAL_DIR = RESULT / 'job/evaluation'
SOURCE = RESULT / 'source'
ORDERS = ('ABC', 'BCA', 'CAB')
DOMAINS = ('A', 'B', 'C')
ROLES = ('raw', 'stage-cal', 'first-cal', 'primary')
ENDPOINTS = ('O', 'N', 'Z', 'F_first', 'F', 'G', 'final_all')
COLS = ('average_precision', 'trapezoidal_pr_auc', 'roc_auc',
        'recall_at_fpr_1pct', 'brier', 'log_loss', 'precision', 'recall',
        'f1', 'specificity', 'balanced_accuracy', 'mcc', 'map', 'mrr',
        'recall_at_1', 'recall_at_3', 'recall_at_5', 'recall_at_10',
        'ndcg_at_1', 'ndcg_at_3', 'ndcg_at_5', 'ndcg_at_10')
RANK_COLS = (0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260930
ATOL = 1e-12


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


catalogue = []


def checked_file(base, record, label):
    path = base / record['path']
    assert path.resolve().is_relative_to(base.resolve()), (label, 'path traversal')
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert len(data) == record['bytes'], (label, 'byte count')
    assert digest == record['sha256'], (label, 'sha256')
    catalogue.append({'label': label, 'local_path': str(path),
                      'bytes': len(data), 'sha256': digest})
    return path


saved = read_json(EVAL_DIR / 'evaluation.json')
policy = read_json(SOURCE / 'schema/step28_relation_memory_policy.json')
assert tuple(policy['orders']) == ORDERS
assert saved['reference'] == policy['reference']
candidate_path = checked_file(EVAL_DIR, saved['collected'], 'candidate collection')
candidate = read_json(candidate_path)
reference_dir = PACKAGE / policy['reference']['local_root']
refs = {record['path']: read_json(checked_file(reference_dir, record,
        'reference collection ' + record['path']))
        for record in policy['reference']['collections']}
late_ref = refs['collected.json']
early_ref = refs['reference/collected.json']

assert len(candidate['group_ids']) == 60
assert len(set(candidate['group_ids'])) == 60
assert tuple(candidate['metric_columns']) == COLS
domain_labels = np.array(candidate['domains'])
domain_rows = {d: np.flatnonzero(domain_labels == d) for d in DOMAINS}
assert set(candidate['domains']) == set(DOMAINS)
assert all(len(rows) == 20 for rows in domain_rows.values())
for ref in refs.values():
    for header in ('group_ids', 'domains', 'metric_columns'):
        assert ref[header] == candidate[header], ('header alignment', header)

assert saved['source_files'] == candidate['source_files']
assert len(candidate['source_files']) == 23
for record in candidate['source_files']:
    checked_file(SOURCE, record, 'frozen source ' + record['path'])

assert set(candidate['points']) == {'initial'} | {
    f'{order}_relation_stage{stage}' for order in ORDERS for stage in (1, 2, 3)}
assert set(candidate['points']['initial']) == {'raw'}

count_comparisons = []
count_summaries = []
roles_rank_difference = []
matrix_sets = {'relation': {}, 'logit0.1': {}}
candidate_all_points = {}


def count_metrics(counts):
    c = np.array([[row[k] for k in ('tp', 'fp', 'fn', 'tn')]
                  for row in counts], dtype=np.float64)
    tp, fp, fn, tn = c.T
    precision = np.divide(tp, tp + fp, out=np.zeros(60), where=(tp + fp) != 0)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denom,
                    out=np.zeros(60), where=denom != 0)
    return np.column_stack((precision, recall, f1, specificity,
                            (recall + specificity) / 2, mcc))


def load_set(base, record, label):
    matrix_file = checked_file(base, record['matrix'], label + ' matrix')
    counts_file = checked_file(base, record['counts'], label + ' counts')
    matrix = np.load(matrix_file, allow_pickle=False)
    counts = read_json(counts_file)
    assert matrix.shape == (60, 22), (label, matrix.shape)
    assert matrix.dtype == np.dtype('float64'), (label, str(matrix.dtype))
    assert np.isfinite(matrix).all(), (label, 'non-finite')
    assert len(counts) == 60
    for row in counts:
        assert set(row) == {'tp', 'fp', 'fn', 'tn'}
        assert all(type(v) is int and v >= 0 for v in row.values())
        assert row['tp'] + row['fn'] == 20
        assert row['fp'] + row['tn'] == 358
    derived = count_metrics(counts)
    diff = np.abs(derived - matrix[:, 6:12])
    flat = int(np.argmax(diff))
    row, col = np.unravel_index(flat, diff.shape)
    count_comparisons.append({'label': label, 'max_abs_error': float(diff[row, col]),
                              'group_row': int(row), 'metric': COLS[6 + col]})
    assert float(diff.max()) <= ATOL, ('counts/metrics', label, float(diff.max()))
    count_summaries.append({'label': label, 'group_count': len(counts),
                            'all_pairs_per_group': 378,
                            'positive_pairs_per_group': 20,
                            'negative_pairs_per_group': 358})
    return matrix


for point, records in candidate['points'].items():
    candidate_all_points[point] = {
        role: load_set(EVAL_DIR, record, f'relation/{point}/{role}')
        for role, record in records.items()}

for order in ORDERS:
    for stage in (1, 2, 3):
        point = f'{order}_relation_stage{stage}'
        assert set(candidate['points'][point]) == set(ROLES[:3])
        matrix_sets['relation'][(order, stage)] = candidate_all_points[point]
        if stage == 1:
            point, collection, base = f'{order}_shared', early_ref, reference_dir / 'reference'
        else:
            point, collection, base = f'{order}_logit_tenth_stage{stage}', late_ref, reference_dir
        records = collection['points'][point]
        assert set(records) == set(ROLES[:3])
        matrix_sets['logit0.1'][(order, stage)] = {
            role: load_set(base, records[role], f'logit0.1/{point}/{role}')
            for role in ROLES[:3]}

for arm, points in matrix_sets.items():
    for (order, stage), roles in points.items():
        for role in ('stage-cal', 'first-cal'):
            diff = float(np.max(np.abs(roles[role][:, RANK_COLS] - roles['raw'][:, RANK_COLS])))
            roles_rank_difference.append({'arm': arm, 'order': order, 'stage': stage,
                                          'role': role, 'max_abs_error': diff})
            assert diff == 0, (arm, order, stage, role, 'ranking changed')
        if stage == 1:
            assert np.array_equal(roles['stage-cal'], roles['first-cal'])
        primary = roles['stage-cal'].copy()
        primary[:, RANK_COLS] = roles['raw'][:, RANK_COLS]
        roles['primary'] = primary


def endpoint_terms(name, order):
    """Weights from frozen scientific contract, specified as (stage, domain, weight)."""
    first, second, third = tuple(order)
    return {
        'O': [(3, first, .5), (3, second, .5)],
        'N': [(2, second, .5), (3, third, .5)],
        'Z': [(3, third, 1.)],
        'F_first': [(1, first, 1.), (3, first, -1.)],
        'F': [(1, first, .5), (3, first, -.5),
              (2, second, .5), (3, second, -.5)],
        'G': [(2, second, .5), (1, second, -.5),
              (3, third, .5), (2, third, -.5)],
        'final_all': [(3, first, 1/3), (3, second, 1/3), (3, third, 1/3)],
    }[name]


def contributions(arm, role, endpoint):
    by_order = []
    for order in ORDERS:
        rows = np.zeros((60, 22), dtype=np.float64)
        for stage, domain, weight in endpoint_terms(endpoint, order):
            ix = domain_rows[domain]
            rows[ix] += weight * matrix_sets[arm][(order, stage)][role][ix]
        if endpoint in ('F_first', 'F', 'G'):
            rows[:, [4, 5]] *= -1  # Positive forgetting or gain for the two losses.
        by_order.append(rows)
    return np.stack(by_order)


rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
draws = rng.integers(0, 20, size=(BOOTSTRAP_N, 3, 20))
# Each group row receives its frequency among 20 draws in its actual domain.
# This allows a direct linear weighted sum, independent of project endpoint tensors.
resampling_weights = np.zeros((BOOTSTRAP_N, 60), dtype=np.float64)
for d, domain in enumerate(DOMAINS):
    frequencies = (draws[:, d, :, None] == np.arange(20)[None, None, :]).sum(axis=1)
    resampling_weights[:, domain_rows[domain]] = frequencies / 20.0
assert np.allclose(resampling_weights.sum(axis=1), 3.0, atol=1e-15)


def summarize(blocks):
    common = blocks.mean(axis=0)  # Three fixed orders, never three independent seeds.
    order_values = blocks.sum(axis=1) / 20.0
    point = common.sum(axis=0) / 20.0
    sampled = resampling_weights @ common
    interval = np.quantile(sampled, [.025, .975], axis=0, method='linear')
    return {metric: {
        'mean': float(point[j]),
        'per_order': {order: float(order_values[i, j]) for i, order in enumerate(ORDERS)},
        'conditional_95pct_interval': [float(interval[0, j]), float(interval[1, j])],
    } for j, metric in enumerate(COLS)}


blocks = {arm: {role: {endpoint: contributions(arm, role, endpoint)
                       for endpoint in ENDPOINTS} for role in ROLES}
          for arm in matrix_sets}
computed_endpoints = {arm: {role: {endpoint: summarize(value)
                                 for endpoint, value in role_blocks.items()}
                            for role, role_blocks in arm_blocks.items()}
                      for arm, arm_blocks in blocks.items()}
computed_delta = {endpoint: summarize(blocks['relation']['primary'][endpoint]
                                    - blocks['logit0.1']['primary'][endpoint])
                  for endpoint in ENDPOINTS}

comparison_count = 0
max_error = -1.0
max_location = None
max_values = None
violations = []


def compare(got, expected, location):
    global comparison_count, max_error, max_location, max_values
    if isinstance(got, dict):
        assert set(got) == set(expected), (location, 'keys')
        for key in got:
            compare(got[key], expected[key], location + '/' + key)
    elif isinstance(got, list):
        assert len(got) == len(expected), (location, 'length')
        for i, (g, e) in enumerate(zip(got, expected)):
            compare(g, e, location + '/' + str(i))
    else:
        error = abs(float(got) - float(expected))
        comparison_count += 1
        if error > max_error:
            max_error, max_location, max_values = error, location, [got, expected]
        if error > ATOL:
            violations.append({'location': location, 'got': got, 'expected': expected,
                               'abs_error': error})


compare(computed_endpoints, saved['endpoints'], 'endpoints')
compare(computed_delta, saved['delta'], 'delta')
endpoint_compare = {'numeric_leaves': comparison_count, 'max_abs_error': max_error,
                    'max_error_location': max_location, 'independent_saved_values': max_values,
                    'violations_above_1e_12': list(violations)}
assert comparison_count == 8316

# Separately verify all candidate absolute macro rows, including the initial raw set.
comparison_count = 0
max_error = -1.0
max_location = None
max_values = None
violations = []
for point, roles in candidate_all_points.items():
    for role, matrix in roles.items():
        reported = saved['absolute_stage_results'][point][role]
        macro_all = dict(zip(COLS, matrix.mean(axis=0).tolist()))
        macro_domain = {domain: dict(zip(COLS, matrix[ix].mean(axis=0).tolist()))
                        for domain, ix in domain_rows.items()}
        compare(macro_all, reported['macro_all'], f'absolute/{point}/{role}/macro_all')
        compare(macro_domain, reported['macro_by_domain'], f'absolute/{point}/{role}/macro_by_domain')
absolute_compare = {'numeric_leaves': comparison_count, 'max_abs_error': max_error,
                    'max_error_location': max_location, 'independent_saved_values': max_values,
                    'violations_above_1e_12': list(violations)}
assert comparison_count == 2464

checks = {
    'O_map_positive': computed_delta['O']['map']['mean'] > 0,
    'N_map_non_decrease': computed_delta['N']['map']['mean'] >= 0,
    'Z_map_non_decrease': computed_delta['Z']['map']['mean'] >= 0,
    'O_AP_non_decrease': computed_delta['O']['average_precision']['mean'] >= 0,
    'N_AP_non_decrease': computed_delta['N']['average_precision']['mean'] >= 0,
    'Z_AP_non_decrease': computed_delta['Z']['average_precision']['mean'] >= 0,
}
assert checks == saved['continuation_checks']
assert all(checks.values()) == saved['observed_continuation_checks_pass']
old_conditional_positive = computed_delta['O']['map']['conditional_95pct_interval'][0] > 0
assert old_conditional_positive == saved['old_map_conditional_positive']
assert saved['automatic_followup'] is False

max_count = max(count_comparisons, key=lambda x: x['max_abs_error'])
summary = {
    'audit_scope': 'Saved metric/count aggregation only; Python/NumPy in web workspace, no project evaluation imports, labels, models, training or GPU.',
    'python': platform.python_version(), 'numpy': np.__version__,
    'candidate_metric_count_sets': sum(len(x) for x in candidate['points'].values()),
    'reference_metric_count_sets': 27,
    'required_matrix_and_count_files_checked': 110,
    'unique_groups': len(candidate['group_ids']),
    'domain_group_sizes': {d: len(x) for d, x in domain_rows.items()},
    'headers_identical': True,
    'frozen_source_files_hash_checked': len(candidate['source_files']),
    'baseline_path_mapping': policy['reference'],
    'group_count_metric_comparison': {'sets': len(count_comparisons),
                                       'numeric_values': 55 * 60 * 6,
                                       'maximum': max_count},
    'ranking_columns_equal_across_roles': all(r['max_abs_error'] == 0 for r in roles_rank_difference),
    'endpoint_comparison': endpoint_compare,
    'absolute_macro_comparison': absolute_compare,
    'bootstrap': {'method': 'PCG64 whole-group paired by actual A/B/C, shared draws, three fixed orders averaged',
                  'resamples': BOOTSTRAP_N, 'seed': BOOTSTRAP_SEED,
                  'draw_shape': list(draws.shape), 'numpy_dtype': str(draws.dtype),
                  'draws_sha256': hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
                  'interval': 'linear percentile [.025,.975]'},
    'continuation_checks': checks, 'number_passed': sum(checks.values()),
    'observed_continuation_checks_pass': all(checks.values()),
    'old_map_conditional_positive': old_conditional_positive,
    'automatic_followup': False,
    'limitations': [
        'Pair labels and scores are absent; MAP/AP/AUC/ranking metric generation and Brier/log-loss cannot be recomputed from truth.',
        'Matching recorded headers/hashes proves package alignment, not unrecorded original model predictions.',
        'Intervals condition on these 60 developed-valid groups and fixed orders/s0; no seed or independent-valid uncertainty.',
        'No claim about project Linux/GPU execution or mechanism attribution.'],
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(OUT / 'recomputed_endpoints.json').write_text(json.dumps({
    'endpoints': computed_endpoints, 'delta': computed_delta,
    'continuation_checks': checks}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(OUT / 'file_identity_and_counts.json').write_text(json.dumps({
    'checked_files': catalogue, 'counts': count_summaries,
    'count_metric_comparisons': count_comparisons,
    'role_ranking_comparisons': roles_rank_difference}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
assert not endpoint_compare['violations_above_1e_12']
assert not absolute_compare['violations_above_1e_12']
