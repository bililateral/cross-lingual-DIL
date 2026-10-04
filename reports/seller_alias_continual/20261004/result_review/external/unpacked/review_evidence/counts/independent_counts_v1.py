#!/usr/bin/env python3
"""Independent saved-result audit: NumPy + standard library only, never imports project code.

Input: saved metric/count arrays, blind logits, calibration aggregates and provenance JSON.
No labels, native model, cache, server, train operation, or audit implementation is opened.
AP/MAP/Brier are not reconstructed from truth. Calibrated NLL is checked conditional
on saved raw NLL via the affine-logit sufficient-statistic identity, explicitly.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

COLS = ['average_precision', 'trapezoidal_pr_auc', 'roc_auc', 'recall_at_fpr_1pct',
        'brier', 'log_loss', 'precision', 'recall', 'f1', 'specificity',
        'balanced_accuracy', 'mcc', 'map', 'mrr', 'recall_at_1', 'recall_at_3',
        'recall_at_5', 'recall_at_10', 'ndcg_at_1', 'ndcg_at_3', 'ndcg_at_5', 'ndcg_at_10']
RANK = COLS[:4] + COLS[12:]
CLASS = ['precision', 'recall', 'f1', 'specificity', 'balanced_accuracy', 'mcc']
ROLES = ['raw', 'stage-cal', 'first-cal']
ORDERS = ['ABC', 'BCA', 'CAB']
INPUTS, FAILURES, ERROR_MAX, CHECKS = {}, [], {}, collections.Counter()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def blob(p):
    p = p.resolve()
    p.relative_to(ROOT)
    b = p.read_bytes()
    INPUTS[str(p.relative_to(ROOT))] = {'bytes': len(b), 'sha256': sha(b)}
    return b


def read(p):
    return json.loads(blob(p))


def array(p):
    return np.load(io.BytesIO(blob(p)), allow_pickle=False)


def check(condition, category, key, detail=None):
    CHECKS[category] += 1
    if not bool(condition):
        FAILURES.append({'category': category, 'key': key, 'detail': detail})


def close(actual, expected, category, key, tol=2e-12):
    av, ev = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if av.shape != ev.shape:
        check(False, category, key, {'shapes': [list(av.shape), list(ev.shape)]})
        return
    diff = float(np.max(np.abs(av - ev))) if av.size else 0.0
    ERROR_MAX[category] = max(ERROR_MAX.get(category, 0.0), diff)
    check(np.isfinite(av).all() and np.isfinite(ev).all() and diff <= tol,
          category, key, {'max_abs_error': diff, 'tolerance': tol})


def descriptor(base, rec, category, key):
    b = blob(base / rec['path'])
    check(len(b) == rec['bytes'] and sha(b) == rec['sha256'], category, key)


def rates(c):
    tp, fp, fn, tn = [int(c[k]) for k in ('tp', 'fp', 'fn', 'tn')]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(c, precision=p, recall=r, specificity=sp,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                fpr=fp / (fp + tn) if fp + tn else 0.0,
                accuracy=(tp + tn) / (tp + fp + fn + tn),
                balanced_accuracy=(r + sp) / 2,
                mcc=(tp * tn - fp * fn) / mcc_den if mcc_den else 0.0,
                predicted_positive=tp + fp)


def pooled(rows):
    return rates({k: sum(int(c[k]) for c in rows) for k in ('tp', 'fp', 'fn', 'tn')})


def average(x):
    return [math.fsum(float(v) for v in x[:, k]) / len(x) for k in range(x.shape[1])]


def original_collection(point, new_logit):
    if '_logit_tenth_' in point:
        return new_logit
    if '_logit_quarter_' in point:
        return ORIG['logit_quarter']
    if '_tenth_' in point:
        return ORIG['tenth']
    return ORIG['shared']


def run_job(name, expected_reference_sets, new_logit):
    folder = ROOT / f'reports/seller_alias_continual/20261004/{name}_result'
    job, ev = folder / 'job', folder / 'job/evaluation'
    new = read(ev / 'collected.json')
    ref = read(ev / 'reference/collected.json')
    evaluation = read(ev / 'evaluation.json')
    method = 'logit_tenth' if name == 'logit_low' else 'risk'
    methods = ['seq', 'tenth', 'logit_quarter', 'logit_tenth'] + (['risk'] if name == 'risk' else [])
    expected_new_points = {f'{o}_{method}_stage{s}' for o in ORDERS for s in (2, 3)}
    expected_ref_points = {f'{o}_shared' for o in ORDERS} | {
        f'{o}_{m}_stage{s}' for o in ORDERS for m in methods if m != method for s in (2, 3)}
    check(set(new['points']) == expected_new_points, 'collection_layout', name + ':new')
    check(set(ref['points']) == expected_ref_points, 'collection_layout', name + ':reference')
    check(len(ref['points']) * 3 == expected_reference_sets, 'collection_layout', name + ':count')
    check(evaluation['new_metric_count_sets'] == 18 and
          evaluation['reused_metric_count_sets'] == expected_reference_sets,
          'collection_layout', name + ':evaluation_set_counts')
    for role, coll in [('new', new), ('reference', ref)]:
        check(coll['metric_columns'] == COLS, 'collection_layout', name + ':' + role + ':columns')
        check(len(coll['group_ids']) == len(set(coll['group_ids'])) == 60,
              'collection_layout', name + ':' + role + ':60_unique_groups')
        check(collections.Counter(coll['domains']) == {'A': 20, 'B': 20, 'C': 20},
              'collection_layout', name + ':' + role + ':20_each_domain')
        check(coll['group_ids'] == ORIG['shared']['group_ids'] and
              coll['domains'] == ORIG['shared']['domains'], 'group_alignment', name + ':' + role)
    for point, roles in ref['points'].items():
        check(roles == original_collection(point, new_logit)['points'][point],
              'reference_identity', name + ':' + point)

    matrices, counts, recomputed = {}, {}, {}
    for origin, coll in [(ev, new), (ev / 'reference', ref)]:
        for point, roles in coll['points'].items():
            matrices[point], counts[point], recomputed[point] = {}, {}, {}
            check(set(roles) == set(ROLES), 'collection_layout', name + ':' + point + ':roles')
            for role in ROLES:
                key = name + ':' + point + ':' + role
                desc = roles[role]
                descriptor(origin, desc['matrix'], 'matrix_descriptor', key)
                descriptor(origin, desc['counts'], 'counts_descriptor', key)
                x = array(origin / desc['matrix']['path'])
                c = read(origin / desc['counts']['path'])
                matrices[point][role], counts[point][role] = x, c
                check(x.shape == (60, 22) and x.dtype == np.float64 and np.isfinite(x).all(),
                      'matrix_structure', key)
                check(len(c) == 60 and all(set(row) == {'tp', 'fp', 'fn', 'tn'} and
                    all(type(v) is int and v >= 0 for v in row.values()) for row in c),
                      'counts_structure', key)
                check(all(row['tp'] + row['fn'] == 20 and row['tn'] + row['fp'] == 358 for row in c),
                      'counts_class_totals', key)
                for metric in CLASS:
                    close(x[:, COLS.index(metric)], [rates(row)[metric] for row in c],
                          'group_classification', key + ':' + metric)
                saved_abs = evaluation['absolute_stage_results'][point][role]
                close([saved_abs['macro_all'][m] for m in COLS], average(x),
                      'absolute_macro', key + ':all')
                rec = {'pooled': pooled(c), 'by_domain': {}}
                saved_pool = saved_abs['pooled_fixed_half_classification']
                check(saved_pool['threshold'] == 0, 'threshold_semantics', key)
                for metric, value in saved_pool['pooled'].items():
                    close(value, rec['pooled'][metric], 'absolute_pooled', key + ':all:' + metric)
                for d in 'ABC':
                    indices = [i for i, dom in enumerate(coll['domains']) if dom == d]
                    close([saved_abs['macro_by_domain'][d][m] for m in COLS], average(x[indices]),
                          'absolute_macro', key + ':' + d)
                    rec['by_domain'][d] = pooled([c[i] for i in indices])
                    for metric, value in saved_pool['by_domain'][d].items():
                        close(value, rec['by_domain'][d][metric], 'absolute_pooled', key + ':' + d + ':' + metric)
                recomputed[point][role] = rec
            for role in ['stage-cal', 'first-cal']:
                idx = [COLS.index(m) for m in RANK]
                check(np.array_equal(matrices[point]['raw'][:, idx], matrices[point][role][:, idx]),
                      'metric_rank_invariance', name + ':' + point + ':' + role)
            if point.endswith('_shared'):
                check(np.array_equal(matrices[point]['first-cal'], matrices[point]['stage-cal']) and
                      counts[point]['first-cal'] == counts[point]['stage-cal'],
                      'first_shared_role_equality', name + ':' + point)

    check(set(evaluation['absolute_stage_results']) == set(matrices),
          'absolute_coverage', name)
    raw_csv = blob(ev / 'stage_metrics.csv').decode('utf-8')
    rows = list(csv.DictReader(io.StringIO(raw_csv)))
    expected_keys = {(o, m, str(s), role, d, metric) for o in ORDERS for m in methods
                     for s in (1, 2, 3) for role in ROLES for d in 'ABC' for metric in COLS}
    observed_keys = []
    weights = {'seq': 0., 'tenth': .1, 'logit_quarter': .25, 'logit_tenth': .1, 'risk': .1}
    for row in rows:
        key = tuple(row[k] for k in ['order', 'method', 'stage', 'role', 'actual_domain', 'metric'])
        observed_keys.append(key)
        o, m, s, role, d, metric = key
        point = o + '_shared' if s == '1' else f'{o}_{m}_stage{s}'
        check(point == row['source_point'] and float(row['history_weight']) == weights[m],
              'csv_source_and_weight', name + ':' + ':'.join(key))
        indices = [i for i, dom in enumerate(new['domains']) if dom == d]
        want = math.fsum(float(matrices[point][role][i, COLS.index(metric)]) for i in indices) / 20
        close(float(row['group_macro']), want, 'csv_macro', name + ':' + ':'.join(key))
    check(len(observed_keys) == len(set(observed_keys)) and set(observed_keys) == expected_keys,
          'csv_coverage', name)

    partition = read(job / 'run/partition.json')
    shared_partition = read(ROOT / 'dependencies/shared/run/partition.json')
    check(partition == shared_partition, 'calibration_partition', name + ':shared_partition')
    check([(g['group_uid'], g['domain']) for g in partition['development']] ==
          list(zip(new['group_ids'], new['domains'])), 'calibration_partition', name + ':valid_alignment')
    cal_stats = {}
    expected_score_files = set()
    for point in sorted(expected_new_points):
        p = read(job / f'run/points/{point}.json')
        o, s = point[:3], int(point[-1])
        first = read(ROOT / f'dependencies/shared/run/points/{o}_shared.json')['first_map_parameters']
        check(p['first_map_parameters'] == first, 'first_map_identity', name + ':' + point)
        descriptor(job / 'run', p['mapping'], 'calibration_descriptor', name + ':' + point)
        mapping = read(job / 'run' / p['mapping']['path'])
        check(mapping['name'] == point and mapping['actual_domain'] == o[s - 1] and
              mapping['model_state_sha256'] == p['model_state_sha256'],
              'calibration_identity', name + ':' + point)
        expected_cal = [g['group_uid'] for g in partition['calibration'] if g['domain'] == o[s - 1]]
        check(mapping['calibration_group_ids'] == expected_cal and len(expected_cal) == 12 and
              mapping['group_count'] == 12 and mapping['pair_count'] == 4536 and
              mapping['positive_count'] == 240 and mapping['role'] == 'calibration',
              'calibration_partition', name + ':' + point)
        check(not set(expected_cal) & set(new['group_ids']) and
              not set(expected_cal) & {g['group_uid'] for g in partition['fit']},
              'calibration_disjoint', name + ':' + point)
        check(mapping['score_source'] == p['scores']['calibration'],
              'calibration_identity', name + ':' + point + ':score')
        check(set(p['scores']) == {'development', 'calibration', 'stage-cal', 'first-cal'},
              'score_structure', name + ':' + point + ':four_roles')
        scores = {}
        for role, desc in p['scores'].items():
            descriptor(job / 'run', desc, 'score_descriptor', name + ':' + point + ':' + role)
            expected_score_files.add(desc['path'])
            scores[role] = array(job / 'run' / desc['path'])
            shape, dtype = ((12, 378), np.float32) if role == 'calibration' else (
                ((60, 378), np.float32) if role == 'development' else ((60, 378), np.float64))
            check(scores[role].shape == shape and scores[role].dtype == dtype and
                  np.isfinite(scores[role]).all(), 'score_structure', name + ':' + point + ':' + role)
        raw = scores['development'].astype(np.float64)
        point_stats = {'stage_map': {k: mapping[k] for k in ['a', 'b', 'group_count', 'pair_count', 'positive_count']},
                       'first_map': first, 'score_roles': {}}
        for role in ROLES:
            z = raw if role == 'raw' else scores[role]
            predicted = (z >= 0).sum(axis=1)
            expected_positive = [c['tp'] + c['fp'] for c in counts[point][role]]
            close(predicted, expected_positive, 'predicted_positive', name + ':' + point + ':' + role, 0)
            probabilities = np.exp(-np.logaddexp(0, -z))
            check(np.array_equal(z >= 0, probabilities >= .5),
                  'threshold_semantics', name + ':' + point + ':' + role + ':logit_probability')
            rs = {'prediction_positive_by_group': predicted.tolist(),
                  'prediction_positive_total': int(predicted.sum()),
                  'score_min': float(z.min()), 'score_max': float(z.max())}
            if role != 'raw':
                mm = mapping if role == 'stage-cal' else first
                a, b = mm['a'], mm['b']
                check(.001 <= a <= 100 and -100 <= b <= 100, 'positive_affine_bounds', name + ':' + point + ':' + role)
                close(z, a * raw + b, 'affine_score_exact', name + ':' + point + ':' + role, 0)
                ties, mismatch, minimum_distinct_gap = 0, 0, math.inf
                for rowx, rowz in zip(raw, z):
                    ordered = sorted(range(378), key=lambda j: (float(rowx[j]), j))
                    for left, right in zip(ordered[:-1], ordered[1:]):
                        tied = rowx[left] == rowx[right]
                        ties += int(tied)
                        if tied:
                            mismatch += int(rowz[left] != rowz[right])
                        else:
                            gap = float(rowz[right] - rowz[left])
                            mismatch += int(gap <= 0)
                            minimum_distinct_gap = min(minimum_distinct_gap, gap)
                check(mismatch == 0, 'score_order_and_ties', name + ':' + point + ':' + role)
                rs.update(exact_tied_adjacencies=ties, changed_order_or_ties=mismatch,
                          minimum_distinct_gap=minimum_distinct_gap,
                          equivalent_raw_threshold=-b / a)
                # Conditional on saved raw NLL and positive count, with no truth reopening:
                # E[y*x] = E[softplus(x)] - raw_NLL; E[y] = 20/378.
                # E[NLL(a*x+b)] = E[softplus(a*x+b)] - a*E[y*x] - b*E[y].
                check(np.all((probabilities > 1e-15) & (probabilities < 1 - 1e-15)),
                      'affine_nll_unclipped_domain', name + ':' + point + ':' + role)
                raw_loss = matrices[point]['raw'][:, COLS.index('log_loss')]
                moment = np.logaddexp(0, raw).mean(axis=1) - raw_loss
                derived_nll = np.logaddexp(0, z).mean(axis=1) - a * moment - b * (20 / 378)
                close(matrices[point][role][:, COLS.index('log_loss')], derived_nll,
                      'calibrated_nll_from_saved_raw_nll', name + ':' + point + ':' + role)
            point_stats['score_roles'][role] = rs
        # Calibration fit consistency uses only blind calibration scores, the declared
        # positive count, and saved initial NLL as sufficient statistics, not labels.
        x = scores['calibration'].astype(np.float64)
        prevalence = mapping['positive_count'] / x.size
        yx = float(np.logaddexp(0, x).mean()) - mapping['initial_nll']
        for idx, theta in enumerate([mapping] + mapping['trajectory']):
            a, b = theta['a'], theta['b']
            z = a * x + b
            prob = np.exp(-np.logaddexp(0, -z))
            nll = float(np.logaddexp(0, z).mean()) - a * yx - b * prevalence
            ga, gb = float((prob * x).mean()) - yx, float(prob.mean()) - prevalence
            gradient = [ga, gb]
            for k, (param, (lower, upper)) in enumerate(zip([a, b], [[.001, 100], [-100, 100]])):
                if (param <= lower and gradient[k] > 0) or (param >= upper and gradient[k] < 0):
                    gradient[k] = 0.0
            pg = max(abs(v) for v in gradient)
            close(theta['final_nll'] if idx == 0 else theta['nll'], nll,
                  'calibration_nll_from_saved_initial_nll', name + ':' + point + ':' + str(idx))
            close(theta['projected_gradient_max'], pg,
                  'calibration_gradient_from_saved_initial_nll', name + ':' + point + ':' + str(idx))
            if idx == 0:
                point_stats.update(calibration_predicted_mean=float(prob.mean()),
                                   calibration_declared_prevalence=prevalence,
                                   calibration_intercept_gradient=gb,
                                   calibration_gradient_max_conditional=pg)
        cal_stats[point] = point_stats
    actual_score_files = {str(p.relative_to(job / 'run')) for p in (job / 'run/scores').glob('*.npy')}
    check(expected_score_files == actual_score_files and len(actual_score_files) == 24,
          'score_coverage', name)
    return {'new_matrix_count': 18, 'reference_matrix_count': expected_reference_sets,
            'total_matrices': len(matrices) * 3, 'matrix_shape': [60, 22],
            'count_rows': len(matrices) * 3 * 60, 'stage_csv_rows': len(rows),
            'score_arrays': len(actual_score_files), 'calibration': cal_stats,
            'recomputed_working_points': recomputed}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    ROOT = Path(args.input_root).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    ORIG = {name: read(ROOT / f'dependencies/{name}/evaluation/collected.json')
            for name in ['shared', 'tenth', 'logit_quarter']}
    new_logit = read(ROOT / 'reports/seller_alias_continual/20261004/logit_low_result/job/evaluation/collected.json')
    jobs = {name: run_job(name, n, new_logit) for name, n in [('logit_low', 63), ('risk', 81)]}
    status = 'PASS_SAVED_RESULTS_WITH_DECLARED_TRUTH_BOUNDARY' if not FAILURES else 'FAIL_REVIEW_CHECKS'
    summary = {'status': status, 'input_root': str(ROOT),
               'argv': [sys.executable, *sys.argv], 'python': platform.python_version(),
               'numpy': np.__version__, 'platform': platform.platform(),
               'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
               'elapsed_seconds': time.time() - start, 'check_counts': dict(CHECKS),
               'max_absolute_errors_by_category': ERROR_MAX, 'failure_count': len(FAILURES),
               'failures': FAILURES,
               'scope': {'project_code_imported': False, 'torch_imported': 'torch' in sys.modules,
                         'truth_labels_loaded': False, 'model_loaded': False, 'cache_loaded': False,
                         'bootstrap_redone': False,
                         'ap_map_label_level_recomputed': False,
                         'brier_label_level_recomputed': False,
                         'calibrated_nll_recomputed_conditional_on_saved_raw_nll': True},
               'jobs': {k: {kk: vv for kk, vv in v.items() if kk not in ['recomputed_working_points', 'calibration']}
                        for k, v in jobs.items()}}
    for name, result in jobs.items():
        (out / f'{name}_working_points.json').write_text(json.dumps(result['recomputed_working_points'], indent=2, ensure_ascii=False) + '\n')
        (out / f'{name}_calibration.json').write_text(json.dumps(result['calibration'], indent=2, ensure_ascii=False) + '\n')
    (out / 'input_hashes.json').write_text(json.dumps(INPUTS, indent=2, sort_keys=True) + '\n')
    (out / 'machine_verdict.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    sys.exit(0 if not FAILURES else 1)
