"""Reviewer's independent saved-statistic reconstruction; no project modules imported.

Uses the exact registered draws, direct group indexing, math.fsum means and an
explicit sorted linear percentile. It never reconstructs or loads formal labels.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

START = time.perf_counter()
ROOT = Path('/workspace/scratch/0f307e99d302/review_input/workspace')
OUT = Path('/workspace/scratch/0f307e99d302/audit_output/independent_statistics')
OUT.mkdir(parents=True, exist_ok=False)
READS = {}
CHECKS = 0
MAX_ERROR = 0.0


def raw_file(path, purpose):
    path = Path(path)
    data = path.read_bytes()
    key = str(path.relative_to(ROOT))
    READS.setdefault(key, {'path': key, 'bytes': len(data),
                          'sha256': hashlib.sha256(data).hexdigest(), 'purposes': []})
    if purpose not in READS[key]['purposes']:
        READS[key]['purposes'].append(purpose)
    return data


def read_json(path, purpose):
    return json.loads(raw_file(path, purpose))


def read_array(path, purpose):
    import io
    value = np.load(io.BytesIO(raw_file(path, purpose)), allow_pickle=False)
    READS[str(Path(path).relative_to(ROOT))].update(shape=list(value.shape), dtype=str(value.dtype))
    assert not np.isinf(value).any()
    return value


def check(a, b, tolerance=1e-12):
    global CHECKS, MAX_ERROR
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert x.shape == y.shape, (x.shape, y.shape)
    assert np.array_equal(np.isnan(x), np.isnan(y))
    valid = ~np.isnan(x)
    error = float(np.max(np.abs(x[valid] - y[valid]))) if valid.any() else 0.0
    CHECKS += x.size
    MAX_ERROR = max(MAX_ERROR, error)
    assert error <= tolerance, error


def manual_quantile(samples, probability):
    ordered = np.sort(samples)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return float(ordered[lower] + weight * (ordered[upper] - ordered[lower]))


job = ROOT / 'reports/job'
collected = read_json(job / 'evaluation/collected.json', 'group/column/record alignment')
reported = read_json(job / 'evaluation/evaluation.json', 'compare independently derived statistics')
run_manifest = read_json(job / 'run/manifest.json', 'score and calibration provenance')
baseline = read_json(ROOT / 'reports/documentation/20261016/h_consistency/baseline_sources.json', 'fixed baseline resolution')
columns = collected['columns']
assert len(columns) == 22
indices = {d: np.flatnonzero(np.asarray(collected['domains']) == d) for d in 'AB'}
assert [len(indices[d]) for d in 'AB'] == [20, 20]
assert len(set(collected['group_ids'])) == 40
origins = {a: (job, collected) for a in ('S_pos', 'H_pos')}
for description in baseline['baselines'].values():
    old_root = ROOT / description['root']
    old_col = read_json(old_root / 'evaluation/collected.json', 'fixed baseline group/role alignment')
    for key in ('group_ids', 'domains', 'columns'):
        assert old_col[key] == collected[key]
    origins[description['arm']] = (old_root, old_col)
assert set(origins) == {'S', 'G_half', 'S_pos', 'H_pos'}

draws = np.random.Generator(np.random.PCG64(20260930)).integers(20, size=(5000, 2, 20))
np.save(OUT / 'registered_draws.npy', draws, allow_pickle=False)
stats_rows = []
stats_lookup = {}
null_rows = 0


def statistics(vector, domain):
    assert vector.shape == (20,)
    missing = int(np.isnan(vector).sum())
    if missing:
        return {'mean': None, 'conditional_95pct_interval': None, 'undefined_groups': missing, 'total_groups': 20}
    sample_means = vector[draws[:, 'AB'.index(domain), :]].sum(axis=1, dtype=np.float64) / 20
    return {'mean': math.fsum(map(float, vector)) / 20,
            'conditional_95pct_interval': [manual_quantile(sample_means, p) for p in (0.025, 0.975)]}


def compare_field(matrix, target, kind, name, role, domain):
    global null_rows
    for j, metric in enumerate(columns):
        rec = statistics(matrix[indices[domain], j], domain)
        saved = target[metric]
        if rec['mean'] is None:
            assert saved['mean'] is None and saved['conditional_95pct_interval'] is None
            assert saved['undefined_groups'] == rec['undefined_groups'] and saved['total_groups'] == 20
            null_rows += 1
            low = high = None
        else:
            low, high = rec['conditional_95pct_interval']
            check([rec['mean'], low, high], [saved['mean'], *saved['conditional_95pct_interval']])
        key = (kind, name, role, domain, metric)
        stats_lookup[key] = rec
        stats_rows.append([*key, rec['mean'], low, high])


matrices, queries, totals, threshold_checks = {}, {}, {}, []
role_order = ('raw', 'stage-cal', 'first-cal')
for arm in ('S_pos', 'H_pos', 'S', 'G_half'):
    origin, col = origins[arm]
    for stage in 'AB':
        point = stage + '_' + arm
        matrices[point], queries[point], totals[point] = {}, {}, {}
        for role in role_order:
            rec = col['points'][point][role]
            saved = read_json(origin / 'evaluation' / rec['matrix']['path'], 'all 40x22 metric entries')
            assert saved['columns'] == columns
            matrix = np.asarray(saved['values'], float)
            assert matrix.shape == (40, 22)
            q = read_array(origin / 'evaluation' / rec['queries']['path'], 'all 40x28x10 query statistics')
            assert q.shape == (40, 28, 10) and np.isfinite(q).all()
            assert ((q >= 0) & (q <= 1)).all()
            check(q.mean(axis=1), matrix[:, 12:])
            if role != 'raw':
                check(q, queries[point]['raw'])
                check(matrix[:, list(range(4)) + list(range(12, 22))],
                      matrices[point]['raw'][:, list(range(4)) + list(range(12, 22))])
            counts = read_json(origin / 'evaluation' / rec['counts']['path'], 'rebuild classification from 40 confusion records')
            c = np.array([[r[k] for k in ('tp', 'fp', 'fn', 'tn')] for r in counts], dtype=np.int64)
            assert c.shape == (40, 4) and (c >= 0).all()
            assert (c[:, 0] + c[:, 2] == 20).all() and (c[:, 1] + c[:, 3] == 358).all()
            for i, (tp, fp, fn, tn) in enumerate(c.tolist()):
                denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                reconstructed = [tp / (tp + fp) if tp + fp else None,
                                 tp / (tp + fn), 2 * tp / (2 * tp + fp + fn),
                                 tn / (tn + fp), (tp / (tp + fn) + tn / (tn + fp)) / 2,
                                 (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else None]
                check(reconstructed, matrix[i, 6:12])
            totals[point][role] = {d: c[indices[d]].sum(axis=0).tolist() for d in 'AB'}
            for domain, selected in [('all', np.arange(40)), *indices.items()]:
                tp, fp, fn, tn = map(int, c[selected].sum(axis=0))
                pooled = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                          'fpr': fp / (fp + tn), 'recall': tp / (tp + fn),
                          'precision': tp / (tp + fp) if tp + fp else None,
                          'f1': 2 * tp / (2 * tp + fp + fn)}
                for k, v in pooled.items():
                    check(v, reported['absolute'][point][role]['fixed_half_classification'][domain][k])
            for d in 'AB':
                compare_field(matrix, reported['absolute'][point][role]['by_domain'][d], 'absolute', point, role, d)
            check(matrix.mean(axis=0), [reported['absolute'][point][role]['macro_all'][k] for k in columns])
            assert matrix.shape == (40, 22) and not np.isinf(matrix).any()
            matrices[point][role], queries[point][role] = matrix, q
            if arm.endswith('_pos'):
                meta = run_manifest['points'][point]
                score = read_array(job / 'run' / meta['scores'][role]['path'], 'calibration, threshold prediction-count correspondence')
                assert score.shape == (40, 378) and np.isfinite(score).all()
                assert np.array_equal((score >= 0).sum(axis=1), c[:, 0] + c[:, 1])
                threshold_checks.append({'point': point, 'role': role, 'groups': 40, 'predicted_positive_counts_match': True})
                if role == 'raw':
                    raw = score
                else:
                    m = meta['maps']['stage' if role == 'stage-cal' else 'first']
                    assert m['a'] > 0
                    check(score, m['a'] * raw.astype(float) + m['b'])
                    order = np.argsort(raw, axis=1, kind='stable')
                    assert np.array_equal(order, np.argsort(score, axis=1, kind='stable'))
                    raw_sorted = np.take_along_axis(raw, order, axis=1)
                    score_sorted = np.take_along_axis(score, order, axis=1)
                    assert np.array_equal(np.diff(raw_sorted, axis=1) == 0, np.diff(score_sorted, axis=1) == 0)

contrasts = (('S_pos', 'S'), ('H_pos', 'S'), ('H_pos', 'G_half'), ('H_pos', 'S_pos'))
sign = np.asarray([-1 if k in ('brier', 'log_loss') else 1 for k in columns])
map_summary = []
for candidate, comparator in contrasts:
    name = candidate + '-' + comparator
    for role in role_order:
        for stage, section, kind in [('A', 'initial_comparisons', 'initial_contrast'), ('B', 'comparisons', 'final_contrast')]:
            delta = matrices[stage + '_' + candidate][role] - matrices[stage + '_' + comparator][role]
            for d in 'AB':
                compare_field(delta, reported[section][name][role][d], kind, name, role, d)
                if stage == 'B' and role == 'raw':
                    values = delta[indices[d], columns.index('map')]
                    map_summary.append({'contrast': name, 'domain': d, **statistics(values, d),
                                        'positive_groups': int((values > 0).sum()), 'negative_groups': int((values < 0).sum()),
                                        'zero_groups': int((values == 0).sum())})
        difference = (matrices['A_' + candidate][role] - matrices['B_' + candidate][role]
                      - matrices['A_' + comparator][role] + matrices['B_' + comparator][role]) * sign
        compare_field(difference, reported['forgetting_comparisons'][name][role], 'forgetting_contrast', name, role, 'A')
for arm in ('S_pos', 'H_pos', 'S', 'G_half'):
    for role in role_order:
        change = matrices['B_' + arm][role] - matrices['A_' + arm][role]
        compare_field(change, reported['old_changes'][arm][role], 'old_change', arm, role, 'A')

degrees = read_array(job / 'evaluation' / collected['query_positive_counts']['path'], 'pre-registered 1/2-positive subgroup membership')
assert degrees.shape == (40, 28)
assert ((degrees == 1).sum(axis=1) == 16).all() and ((degrees == 2).sum(axis=1) == 12).all()
subgroups, subgroup_rows = {}, []
for point in matrices:
    ap = queries[point]['raw'][:, :, 0]
    subgroups[point] = {m: np.asarray([math.fsum(map(float, ap[i][degrees[i] == m])) / int((degrees[i] == m).sum()) for i in range(40)]) for m in (1, 2)}
    check((16 * subgroups[point][1] + 12 * subgroups[point][2]) / 28, matrices[point]['raw'][:, 12])
for candidate, comparator in contrasts:
    name = candidate + '-' + comparator
    for m in (1, 2):
        delta = subgroups['B_' + candidate][m] - subgroups['B_' + comparator][m]
        for d in 'AB':
            r = statistics(delta[indices[d]], d)
            target = reported['query_subgroup_MAP_comparisons'][name][str(m)][d]
            check([r['mean'], *r['conditional_95pct_interval']], [target['mean'], *target['conditional_95pct_interval']])
            subgroup_rows.append({'contrast': name, 'positives': m, 'domain': d, **r})

assert len(stats_rows) == 2640
saved_csv_path = ROOT / 'reports/result_sync/analysis/all_metrics.csv'
raw_file(saved_csv_path, 'all 2640 saved independent-audit CSV rows')
with saved_csv_path.open(newline='') as f:
    saved_rows = list(csv.DictReader(f))
assert len(saved_rows) == 2640
for row in saved_rows:
    key = tuple(row[k] for k in ('kind', 'point_or_contrast', 'role', 'domain', 'metric'))
    value = stats_lookup[key]
    expected = [value['mean'], *(value['conditional_95pct_interval'] or [None, None])]
    actual = [float(row[k]) if row[k] else None for k in ('mean', 'lower_95', 'upper_95')]
    check(actual, expected)

with (OUT / 'all_metrics_independent.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['kind', 'point_or_contrast', 'role', 'domain', 'metric', 'mean', 'lower_95', 'upper_95'])
    writer.writerows(stats_rows)

primary = {r['domain']: r for r in map_summary if r['contrast'] == 'S_pos-S'}
passed = all(primary[d]['conditional_95pct_interval'][0] > 0 for d in 'AB')
assert passed is False and reported['success']['passed'] is False
summary = {'status': 'INDEPENDENT_SAVED_STATISTICS_RECONSTRUCTION_MATCHES',
           'python': platform.python_version(), 'numpy': np.__version__,
           'statistical_rows': len(stats_rows), 'undefined_rows': null_rows,
           'numeric_comparisons': CHECKS, 'maximum_absolute_difference': MAX_ERROR,
           'subgroup_comparisons': len(subgroup_rows), 'primary_passed': passed,
           'primary_A_lower_17digits': format(primary['A']['conditional_95pct_interval'][0], '.17g'),
           'primary_A_lower_hex': primary['A']['conditional_95pct_interval'][0].hex(),
           'draws_array_bytes_sha256': hashlib.sha256(draws.tobytes(order='C')).hexdigest(),
           'map_contrasts': map_summary, 'subgroup_contrasts': subgroup_rows,
           'threshold_prediction_count_checks': threshold_checks, 'confusion_totals': totals,
           'read_file_count': len(READS), 'elapsed_seconds': time.perf_counter() - START,
           'limits': 'Saved AP/ROC/Brier/log-loss/query entries are input evidence, not recomputed from labels. No model/Memory loads, training, production finalize/blind_gate, network or project-server access.'}
(OUT / 'result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
(OUT / 'read_scope.json').write_text(json.dumps(list(READS.values()), ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k not in ('confusion_totals', 'threshold_prediction_count_checks', 'subgroup_contrasts')}, ensure_ascii=False, indent=2))
