"""Independent saved-matrix/count interpretation audit; stdlib + NumPy only.

Does not import project evaluation or verification functions, access raw labels,
load models/memories, fit calibration, train, or reconstruct bootstrap intervals.
Intervals are inspected from the saved evaluation; a separate auditor recomputes
them. Matrices and counts are independently aggregated here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

ROOT = Path('/workspace/scratch/f4d639b3b473/relation_memory_result_review')
OUT = Path('/workspace/scratch/f4d639b3b473/result_interpretation_audit')
RESULT = ROOT / 'reports/seller_alias_continual/20261006/relation_result'
CAND = RESULT / 'job/evaluation'
REF = ROOT / 'reports/seller_alias_continual/20261004/logit_low_result/job/evaluation'
ORDERS = ('ABC', 'BCA', 'CAB')
ROLES = ('raw', 'stage-cal', 'first-cal')
COLUMNS = ['average_precision', 'trapezoidal_pr_auc', 'roc_auc', 'recall_at_fpr_1pct',
           'brier', 'log_loss', 'precision', 'recall', 'f1', 'specificity', 'balanced_accuracy',
           'mcc', 'map', 'mrr', 'recall_at_1', 'recall_at_3', 'recall_at_5', 'recall_at_10',
           'ndcg_at_1', 'ndcg_at_3', 'ndcg_at_5', 'ndcg_at_10']
RANK = COLUMNS[:4] + COLUMNS[12:]
PROB = ('brier', 'log_loss')
CLASS = ('precision', 'recall', 'f1', 'specificity', 'balanced_accuracy', 'mcc')
TERMS = {
    'O': [(3, 1, .5), (3, 2, .5)],
    'N': [(2, 2, .5), (3, 3, .5)],
    'Z': [(3, 3, 1.)],
    'F_first': [(1, 1, 1.), (3, 1, -1.)],
    'F': [(1, 1, .5), (3, 1, -.5), (2, 2, .5), (3, 2, -.5)],
    'G': [(2, 2, .5), (1, 2, -.5), (3, 3, .5), (2, 3, -.5)],
    'final_all': [(3, 1, 1/3), (3, 2, 1/3), (3, 3, 1/3)],
}
tick = time.monotonic()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


verified = []


def checked(folder, rec, kind):
    path = folder / rec['path']
    payload = path.read_bytes()
    assert len(payload) == rec['bytes']
    assert hashlib.sha256(payload).hexdigest() == rec['sha256']
    verified.append(str(path.relative_to(ROOT)))
    if kind == 'matrix':
        value = np.load(path, allow_pickle=False)
        assert value.shape == (60, 22) and value.dtype == np.float64 and np.isfinite(value).all()
        return value
    value = json.loads(payload)
    assert len(value) == 60
    for row in value:
        assert set(row) == {'tp', 'fp', 'fn', 'tn'}
        assert all(type(v) is int and v >= 0 for v in row.values())
        assert row['tp'] + row['fn'] == 20 and row['fp'] + row['tn'] == 358
    return value


collection = read_json(CAND / 'collected.json')
reference = [read_json(REF / 'collected.json'), read_json(REF / 'reference/collected.json')]
evaluation = read_json(CAND / 'evaluation.json')
assert collection['metric_columns'] == COLUMNS
for col in reference:
    assert all(col[k] == collection[k] for k in ('metric_columns', 'group_ids', 'domains'))
domains = np.array(collection['domains'])
ids = collection['group_ids']
assert len(set(ids)) == 60 and all(sum(domains == d) == 20 for d in 'ABC')
matrices = {'relation': {}, 'logit0.1': {}}
counts = {'relation': {}, 'logit0.1': {}}
rank_columns = [COLUMNS.index(c) for c in RANK]
for arm in matrices:
    for order in ORDERS:
        for stage in (1, 2, 3):
            if arm == 'relation':
                folder = CAND
                rec = collection['points'][f'{order}_relation_stage{stage}']
            elif stage == 1:
                folder = REF / 'reference'
                rec = reference[1]['points'][order + '_shared']
            else:
                folder = REF
                rec = reference[0]['points'][f'{order}_logit_tenth_stage{stage}']
            assert set(rec) == set(ROLES)
            matrices[arm][order, stage] = {role: checked(folder, rec[role]['matrix'], 'matrix') for role in ROLES}
            counts[arm][order, stage] = {role: checked(folder, rec[role]['counts'], 'counts') for role in ROLES}
            for role in ('stage-cal', 'first-cal'):
                assert np.array_equal(matrices[arm][order, stage]['raw'][:, rank_columns],
                                      matrices[arm][order, stage][role][:, rank_columns])
            primary = matrices[arm][order, stage]['stage-cal'].copy()
            primary[:, rank_columns] = matrices[arm][order, stage]['raw'][:, rank_columns]
            matrices[arm][order, stage]['primary'] = primary
initial = collection['points']['initial']['raw']
checked(CAND, initial['matrix'], 'matrix')
checked(CAND, initial['counts'], 'counts')
assert len(verified) == 110 and len(set(verified)) == 110

# Direct domain means and signed endpoint sums, independent of production code.
means, per_order = {}, {}
max_error = 0.
for arm in matrices:
    means[arm], per_order[arm] = {}, {}
    for role in (*ROLES, 'primary'):
        means[arm][role], per_order[arm][role] = {}, {}
        for endpoint, terms in TERMS.items():
            rows = []
            for order in ORDERS:
                v = sum(weight * matrices[arm][order, stage][role][domains == order[arrival - 1]].mean(0)
                        for stage, arrival, weight in terms)
                if endpoint in ('F_first', 'F', 'G'):
                    v[[COLUMNS.index(c) for c in PROB]] *= -1
                rows.append(v)
            result = np.stack(rows)
            means[arm][role][endpoint] = dict(zip(COLUMNS, result.mean(0).tolist()))
            per_order[arm][role][endpoint] = {o: dict(zip(COLUMNS, row.tolist())) for o, row in zip(ORDERS, result)}
            for col, c in enumerate(COLUMNS):
                saved = evaluation['endpoints'][arm][role][endpoint][c]
                max_error = max(max_error, abs(saved['mean'] - result[:, col].mean()))
                max_error = max(max_error, *(abs(saved['per_order'][o] - result[i, col]) for i, o in enumerate(ORDERS)))
assert max_error < 1e-12


def divide(x, y):
    return x / y if y else 0.


def classification(row):
    tp, fp, fn, tn = (row[k] for k in ('tp', 'fp', 'fn', 'tn'))
    precision, recall = divide(tp, tp + fp), divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return {'precision': precision, 'recall': recall, 'f1': divide(2 * tp, 2 * tp + fp + fn),
            'specificity': specificity, 'fpr': divide(fp, fp + tn),
            'balanced_accuracy': .5 * (recall + specificity),
            'mcc': divide(tp * tn - fp * fn,
                          math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))}


pooled = {}
for arm in matrices:
    pooled[arm] = {}
    for role in ROLES:
        pooled[arm][role] = {}
        for endpoint in ('O', 'N', 'Z', 'final_all'):
            selected, group_ids = [], []
            for order in ORDERS:
                for stage, arrival, weight in TERMS[endpoint]:
                    assert weight > 0
                    for i in np.flatnonzero(domains == order[arrival - 1]):
                        selected.append(counts[arm][order, stage][role][i])
                        group_ids.append(ids[i])
            totals = {k: sum(r[k] for r in selected) for k in ('tp', 'fp', 'fn', 'tn')}
            direct_group_means = {c: float(np.mean([classification(r)[c] for r in selected])) for c in CLASS}
            assert all(abs(direct_group_means[c] - means[arm][role][endpoint][c]) < 1e-12 for c in CLASS)
            pooled[arm][role][endpoint] = {
                'count_totals': totals, 'pooled': classification(totals),
                'macro': direct_group_means, 'group_endpoint_occurrences': len(selected),
                'unique_groups': len(set(group_ids)),
                'group_occurrences_without_positive_predictions': sum(r['tp'] + r['fp'] == 0 for r in selected),
            }

expected_report_counts = {
    ('relation', 'O'): [213, 221, 2187, 42739], ('logit0.1', 'O'): [195, 82, 2205, 42878],
    ('relation', 'N'): [139, 75, 2261, 42885], ('logit0.1', 'N'): [180, 78, 2220, 42882],
    ('relation', 'Z'): [101, 62, 1099, 21418], ('logit0.1', 'Z'): [104, 44, 1096, 21436],
}
for (arm, endpoint), values in expected_report_counts.items():
    assert list(pooled[arm]['stage-cal'][endpoint]['count_totals'].values()) == values

trajectory = {}
for order in ORDERS:
    trajectory[order] = {}
    for arm in matrices:
        trajectory[order][arm] = {domain: {stage: {c: float(matrices[arm][order, stage]['primary'][domains == domain, COLUMNS.index(c)].mean())
                                                for c in ('map', 'average_precision', 'brier', 'log_loss', 'f1', 'recall')}
                                                for stage in (1, 2, 3)} for domain in 'ABC'}

probability_roles = {}
for arm in matrices:
    probability_roles[arm] = {}
    for endpoint in ('O', 'N', 'Z', 'final_all'):
        probability_roles[arm][endpoint] = {role: {c: means[arm][role][endpoint][c] for c in
                                          ('brier', 'log_loss', 'precision', 'recall', 'f1')}
                                          for role in ROLES}
        probability_roles[arm][endpoint]['stage_minus_raw'] = {c:
            means[arm]['stage-cal'][endpoint][c] - means[arm]['raw'][endpoint][c] for c in PROB}
        probability_roles[arm][endpoint]['stage_minus_first'] = {c:
            means[arm]['stage-cal'][endpoint][c] - means[arm]['first-cal'][endpoint][c] for c in PROB}

delta_means = {ep: {c: means['relation']['primary'][ep][c] - means['logit0.1']['primary'][ep][c]
                     for c in COLUMNS} for ep in TERMS}
for ep in TERMS:
    assert all(abs(delta_means[ep][c] - evaluation['delta'][ep][c]['mean']) < 1e-12 for c in COLUMNS)

# Inspect all 22-column signs. Intervals are read, not independently recomputed here.
sign_inventory = {}
for endpoint in ('O', 'N', 'Z', 'final_all'):
    sign_inventory[endpoint] = {'beneficial_point_estimate': [], 'adverse_point_estimate': [],
                                'adverse_interval_excludes_zero': [], 'interval_contains_zero': []}
    for c in COLUMNS:
        d = delta_means[endpoint][c]
        lo, hi = evaluation['delta'][endpoint][c]['conditional_95pct_interval']
        beneficial = d < 0 if c in PROB else d > 0
        sign_inventory[endpoint]['beneficial_point_estimate' if beneficial else 'adverse_point_estimate'].append(c)
        if lo <= 0 <= hi:
            sign_inventory[endpoint]['interval_contains_zero'].append(c)
        elif (lo > 0 if c in PROB else hi < 0):
            sign_inventory[endpoint]['adverse_interval_excludes_zero'].append(c)

# Check all 154 rows in metrics.md against saved evaluation six-decimal formatting.
metric_table_errors, rows_seen = [], []
endpoint = None
for line in (RESULT / 'metrics.zh.md').read_text(encoding='utf-8').splitlines():
    if line.startswith('## '):
        endpoint = line[3:].strip()
    if not line.startswith('| ') or endpoint not in TERMS:
        continue
    cells = [x.strip() for x in line.strip().strip('|').split('|')]
    if cells[0] not in COLUMNS:
        continue
    c = cells[0]
    wanted = [f"{means['relation']['primary'][endpoint][c]:.6f}",
              f"{means['logit0.1']['primary'][endpoint][c]:.6f}",
              f"{delta_means[endpoint][c]:.6f}"]
    lo, hi = evaluation['delta'][endpoint][c]['conditional_95pct_interval']
    wanted.append(f'[{lo:.6f}, {hi:.6f}]')
    if cells[1:] != wanted:
        metric_table_errors.append({'endpoint': endpoint, 'metric': c, 'actual': cells[1:], 'expected': wanted})
    rows_seen.append((endpoint, c))
assert len(rows_seen) == 154 and len(set(rows_seen)) == 154
assert not metric_table_errors

main_table_errors = []
aliases = {'MAP': 'map', 'AP': 'average_precision', 'Recall@5': 'recall_at_5', 'Brier': 'brier', 'log-loss': 'log_loss'}
for lineno, line in enumerate((ROOT / 'docs/SELLER_ALIAS_RELATION_MEMORY_RESULT.zh.md').read_text(encoding='utf-8').splitlines(), 1):
    match = re.match(r'^\| (O|N|Z|final_all) (MAP|AP|Recall@5|Brier|log-loss) \|', line)
    if not match:
        continue
    ep, c = match[1], aliases[match[2]]
    cells = [x.strip().lstrip('+') for x in line.strip().strip('|').split('|')]
    lo, hi = evaluation['delta'][ep][c]['conditional_95pct_interval']
    expected = [f"{means['relation']['primary'][ep][c]:.6f}", f"{means['logit0.1']['primary'][ep][c]:.6f}",
                f"{delta_means[ep][c]:.6f}", f'[{lo:.6f}, {hi:.6f}]']
    if cells[1:] != expected:
        main_table_errors.append({'line': lineno, 'endpoint': ep, 'metric': c, 'actual': cells[1:], 'expected': expected})

six_rules = {'O_map_positive': delta_means['O']['map'] > 0,
             **{f'{ep}_map_non_decrease': delta_means[ep]['map'] >= 0 for ep in ('N', 'Z')},
             **{f'{ep}_AP_non_decrease': delta_means[ep]['average_precision'] >= 0 for ep in ('O', 'N', 'Z')}}
assert six_rules == evaluation['continuation_checks'] and not any(six_rules.values())

report = {
    'status': 'PASS_SAVED_MATRIX_INTERPRETATION_WITH_MINOR_DISPLAY_CORRECTIONS',
    'environment': {'python': sys.executable, 'version': platform.python_version(), 'numpy': np.__version__,
                    'torch_spec_available': importlib.util.find_spec('torch') is not None},
    'coverage': {'verified_matrix_count_files': len(verified), 'metric_table_rows_checked': len(rows_seen),
                 'all_rank_columns_identical_across_roles': True, 'max_mean_or_per_order_error': max_error,
                 'project_evaluation_imported': False, 'labels_or_models_loaded': False,
                 'bootstrap_recomputed_here': False},
    'six_rules': six_rules, 'pooled': pooled, 'probability_roles': probability_roles,
    'first_domain_trajectories': {o: {arm: trajectory[o][arm][o[0]] for arm in matrices} for o in ORDERS},
    'all_domain_trajectories': trajectory, 'independent_means': means, 'independent_per_order': per_order,
    'independent_primary_delta_means': delta_means, '22_metric_direction_inventory': sign_inventory,
    'metrics_table_errors': metric_table_errors, 'main_result_table_display_errors': main_table_errors,
    'input_files': verified, 'elapsed_seconds': time.monotonic() - tick,
}
(OUT / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
summary = {'status': report['status'], 'environment': report['environment'], 'coverage': report['coverage'],
           'six_rules': six_rules, 'main_result_table_display_errors': main_table_errors,
           'pooled_stage_cal': {a: pooled[a]['stage-cal'] for a in matrices},
           'probability_roles': probability_roles, 'first_domain_trajectories': report['first_domain_trajectories'],
           '22_metric_direction_inventory': sign_inventory, 'elapsed_seconds': report['elapsed_seconds']}
print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
