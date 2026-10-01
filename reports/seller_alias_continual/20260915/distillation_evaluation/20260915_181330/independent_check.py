"""Audit saved metrics, counts and blind predictions; never load supervision."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
ORDERS = ('ABC', 'BCA', 'CAB')
STUDY = ROOT / 'reports/seller_alias_continual/20260915'
NEW = json.loads((HERE / 'evaluation.json').read_text(encoding='utf-8'))
RUN = ROOT / 'reports/seller_alias_continual/20260915/distillation_execution/20260915_164430/job/run'
references = {
    'distillation': (HERE, NEW, ('shared', 'distill_stage2', 'distill_stage3')),
    'sequential': (ROOT / Path(NEW['config']['origin_evaluation']).parent,
                   json.loads((ROOT / NEW['config']['origin_evaluation']).read_text()),
                   ('shared', 'stage2', 'stage3')),
    'random_er': (ROOT / Path(NEW['config']['baseline_evaluation']).parent,
                  json.loads((ROOT / NEW['config']['baseline_evaluation']).read_text()),
                  ('shared', 'er_stage2', 'er_stage3')),
}
columns = NEW['columns']
compared = 0
maximum_difference = 0.0


def close(actual, expected):
    global compared, maximum_difference
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    maximum_difference = max(maximum_difference, error)
    compared += actual.size
    assert np.allclose(actual, expected, atol=1e-12, rtol=0), error


def load_matrix(base, info):
    raw = (base / info['path']).read_bytes()
    assert len(raw) == info['bytes'] and hashlib.sha256(raw).hexdigest() == info['sha256']
    return np.load(base / info['path'], allow_pickle=False)


def confusion(counts):
    tp, fp, fn, tn = [float(counts[k]) for k in ('tp', 'fp', 'fn', 'tn')]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    s = tn / (tn + fp) if tn + fp else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {'precision': p, 'recall': r, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'specificity': s, 'balanced_accuracy': (r + s) / 2,
            'mcc': (tp * tn - fp * fn) / denom if denom else 0.}


matrices = {}
first_domain = {}
micro = {}
for arm, (base, evidence, suffixes) in references.items():
    tensor = np.stack([np.stack([load_matrix(base, evidence['points'][f'{order}_{suffix}']['file'])
                                for suffix in suffixes]) for order in ORDERS])
    assert tensor.shape == (3, 3, 60, 22)
    matrices[arm] = tensor
    first_domain[arm] = np.stack([tensor[i, 2, 20*i:20*(i+1)] for i in range(3)]).mean((0, 1))
    counts = np.zeros(4, dtype=np.int64)
    for i, order in enumerate(ORDERS):
        point = evidence['points'][f'{order}_{suffixes[-1]}']
        for row in point['confusion'][20*i:20*(i+1)]:
            counts += [row[k] for k in ('tp', 'fp', 'fn', 'tn')]
    micro[arm] = confusion(dict(zip(('tp', 'fp', 'fn', 'tn'), counts.tolist())))
    micro[arm]['counts'] = dict(zip(('tp', 'fp', 'fn', 'tn'), counts.tolist()))

close(matrices['distillation'][:, 0], matrices['sequential'][:, 0])
close(matrices['distillation'][:, 0], matrices['random_er'][:, 0])
blind_checks = 0
for name, point in NEW['points'].items():
    matrix = load_matrix(HERE, point['file'])
    scores = np.load(RUN / f'scores/{name}_development.npy', allow_pickle=False)
    for group, counts in enumerate(point['confusion']):
        assert sum(counts.values()) == 378
        assert counts['tp'] + counts['fp'] == int(np.count_nonzero(scores[group] >= 0))
        blind_checks += 1
        for key, value in confusion(counts).items():
            close(value, matrix[group, columns.index(key)])
    for i, domain in enumerate('ABC'):
        close(matrix[20*i:20*(i+1)].mean(0), [point['by_domain'][domain][k] for k in columns])

# Independent bootstrap calculation uses occurrence weights, not array resampling.
draws = np.random.default_rng(20260910).integers(0, 20, size=(5000, 3, 20))
weights = np.stack([[np.bincount(row, minlength=20) / 20 for row in draws[:, d]] for d in range(3)], axis=1)
audit_contrasts = {}
for reference in ('sequential', 'random_er'):
    a, b = matrices['distillation'], matrices[reference]
    role = 'seq' if reference == 'sequential' else 'random_er'
    sets = {key: [[] for _ in 'ABC'] for key in (
        'H', 'N', 'F_distill', f'F_{role}', 'G_distill', f'G_{role}',
        'second_distill_change', f'second_{role}_change', f'final_distill_minus_{role}')}
    for i, order in enumerate(ORDERS):
        first = 'ABC'.index(order[0])
        sl = slice(20*first, 20*(first+1))
        sets['H'][first].append(a[i, 2, sl] - b[i, 2, sl])
        sets['F_distill'][first].append(a[i, 0, sl] - a[i, 2, sl])
        sets[f'F_{role}'][first].append(b[i, 0, sl] - b[i, 2, sl])
        for stage in (1, 2):
            d = 'ABC'.index(order[stage])
            sl = slice(20*d, 20*(d+1))
            sets['N'][d].append(a[i, stage, sl] - b[i, stage, sl])
            sets['G_distill'][d].append(a[i, stage, sl] - a[i, stage-1, sl])
            sets[f'G_{role}'][d].append(b[i, stage, sl] - b[i, stage-1, sl])
        second = 'ABC'.index(order[1])
        sl = slice(20*second, 20*(second+1))
        sets['second_distill_change'][second].append(a[i, 2, sl] - a[i, 1, sl])
        sets[f'second_{role}_change'][second].append(b[i, 2, sl] - b[i, 1, sl])
        for d in range(3):
            sl = slice(20*d, 20*(d+1))
            sets[f'final_distill_minus_{role}'][d].append(a[i, 2, sl] - b[i, 2, sl])
    expected = NEW['comparison']['metrics'] if reference == 'sequential' else NEW['versus_random_er']
    audit_contrasts[reference] = {}
    for key, observations in sets.items():
        groups = np.stack([np.mean(v, axis=0) for v in observations])
        samples = np.einsum('rdg,dgc->rc', weights, groups) / 3
        intervals = np.quantile(samples, [.025, .975], axis=0)
        for j, metric in enumerate(columns):
            record = expected[metric]['contrasts'][key]
            close(groups[:, :, j].mean(), record['mean'])
            close(groups[:, :, j].mean(1), record['by_domain_or_first_domain'])
            close(intervals[:, j], record['conditional_95pct_interval'])
        audit_contrasts[reference][key] = {
            'mean': dict(zip(columns, groups.mean((0, 1)).tolist())),
            'conditional_interval': dict(zip(columns, intervals.T.tolist()))}

ap = columns.index('average_precision')
current = matrices['distillation']
per_order = {}
for i, order in enumerate(ORDERS):
    sl = slice(20*i, 20*(i+1))
    gains = []
    for stage in (1, 2):
        d = 'ABC'.index(order[stage]); window = slice(20*d, 20*(d+1))
        gains.append(float((current[i, stage, window, ap] - current[i, stage-1, window, ap]).mean()))
    h = float((current[i, 2, sl, ap] - matrices['sequential'][i, 2, sl, ap]).mean())
    per_order[order] = {'H': h, 'G': float(np.mean(gains)), 'passes': bool(h > 0 and np.mean(gains) >= .02)}
h = audit_contrasts['sequential']['H']
n = audit_contrasts['sequential']['N']
gates = {'retention': h['mean']['average_precision'] >= .01 and h['conditional_interval']['average_precision'][0] > 0,
         'new_capability': n['conditional_interval']['average_precision'][0] >= -.01,
         'same_order_learning_and_retention': sum(v['passes'] for v in per_order.values()) >= 2}
assert gates == NEW['comparison']['gate']['checks']
assert all(gates.values()) == NEW['comparison']['gate']['passed']
output = {'status': 'PASS', 'saved_numeric_values_compared': int(compared),
          'maximum_absolute_difference': maximum_difference, 'blind_predicted_positive_count_checks': blind_checks,
          'formal_label_parses': 0, 'first_domain_final': {k: dict(zip(columns, v.tolist())) for k,v in first_domain.items()},
          'first_domain_micro': micro, 'independent_gates': gates, 'per_order': per_order,
          'scope': 'Saved matrices/contrasts/intervals/counts and blind predicted-positive counts; does not recompute AP from sealed truth.'}
(HERE / 'independent_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: output[k] for k in ('status', 'saved_numeric_values_compared', 'maximum_absolute_difference', 'blind_predicted_positive_count_checks', 'independent_gates')}))
