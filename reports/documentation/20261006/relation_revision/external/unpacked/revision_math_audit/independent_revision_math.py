#!/usr/bin/env python3
"""Independent NumPy/stdlib handwritten objective audit, no project imports.

No Torch, model, formal data, server, or production evaluator is used. Feature
and score examples are algebraic fixtures, not claims of BGE reachability.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
eps = .001
edges = list(itertools.combinations(range(28), 2))
controllers = [i for i, n in enumerate([3] * 4 + [2] * 8) for _ in range(n)]
labels = np.array([int(controllers[i] == controllers[j]) for i, j in edges])
queries = []
for q in range(28):
    incident = np.array([i for i, edge in enumerate(edges) if q in edge])
    queries.append((incident[labels[incident] == 1], incident[labels[incident] == 0]))

checks = []


def compare(name, actual, expected, atol=1e-8):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    checks.append({'name': name, 'max_abs_error': error, 'atol': atol, 'pass': error <= atol})


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -np.asarray(x)))


def losses_and_score_gradients(p):
    bce = float(np.mean(np.logaddexp(0., p) - labels * p))
    gb = (sigmoid(p) - labels) / 378
    query = hard = rank = square_rank = 0.
    gq, gh, gr = (np.zeros(378) for _ in range(3))
    all_comparison_losses = []
    for pos, neg in queries:
        incident = np.concatenate((pos, neg))
        maximum = p[incident].max()
        exponentials = np.exp(p[incident] - maximum)
        denominator = exponentials.sum()
        query += (maximum + np.log(denominator) - p[pos].mean()) / 28
        gq[incident] += exponentials / denominator / 28
        gq[pos] -= 1 / len(pos) / 28
        selected = sorted(neg, key=lambda j: (-float(p[j]), int(j)))[:5]
        selected = np.asarray(selected)
        hard_gap = p[selected][None, :] - p[pos][:, None]
        hard += float(np.logaddexp(0., hard_gap).mean()) / 28
        hard_derivative = sigmoid(hard_gap) / (28 * len(pos) * 5)
        gh[pos] -= hard_derivative.sum(axis=1)
        gh[selected] += hard_derivative.sum(axis=0)
        gap = p[neg][None, :] - p[pos][:, None]
        pair_losses = np.logaddexp(0., gap)
        rank += float(pair_losses.mean()) / 28
        all_comparison_losses.extend(pair_losses.ravel().tolist())
        derivative = sigmoid(gap) / (28 * len(pos) * len(neg))
        gr[pos] -= derivative.sum(axis=1)
        gr[neg] += derivative.sum(axis=0)
        square_rank += float(((p[pos][:, None] - p[neg][None, :] - 2) ** 2).mean()) / 28
    return {
        'bce': bce, 'query': query, 'hard': hard,
        'B': bce + query + .5 * hard, 'R': rank,
        'square_proxy': float(np.mean((p - (2 * labels - 1)) ** 2)) + square_rank,
        'wrong_global_comparison_R': float(np.mean(all_comparison_losses)),
    }, {'B': gb + gq + .5 * gh, 'R': gr}


def scalar_reference(p):
    # Separate scalar loop, no use of the vectorized objective above.
    bce = sum(math.log1p(math.exp(float(s))) - int(y) * float(s)
              for s, y in zip(p, labels)) / 378
    query, hard, rank = [], [], []
    for q in range(28):
        pos = [float(p[i]) for i, edge in enumerate(edges) if q in edge and labels[i] == 1]
        neg = [float(p[i]) for i, edge in enumerate(edges) if q in edge and labels[i] == 0]
        query.append(math.log(sum(math.exp(s) for s in pos + neg)) - sum(pos) / len(pos))
        hard.append(sum(math.log1p(math.exp(n - a)) for a in pos
                        for n in sorted(neg, reverse=True)[:5]) / (5 * len(pos)))
        rank.append(sum(math.log1p(math.exp(n - a)) for a in pos for n in neg)
                    / (len(pos) * len(neg)))
    return {'B': bce + sum(query) / 28 + .5 * sum(hard) / 28,
            'R': sum(rank) / 28}


def statistics(z):
    # Construct the square loss as weighted regression rows; H=A.T@A, b=A.T@t.
    a_rows = [z.T / np.sqrt(378)]
    t_rows = [(2 * labels - 1) / np.sqrt(378)]
    for pos, neg in queries:
        denominator = np.sqrt(28 * len(pos) * len(neg))
        a_rows.append(np.array([(z[:, a] - z[:, n]) / denominator
                                for a in pos for n in neg]))
        t_rows.append(np.full(len(pos) * len(neg), 2 / denominator))
    a, target = np.concatenate(a_rows), np.concatenate(t_rows)
    return a.T @ a, a.T @ target, float(target @ target)


def compressed(y, w, x, h, b, c, n):
    k = x @ x.T / 378 + eps * np.eye(x.shape[0])
    v = np.linalg.solve(k, x @ (y.T @ w) / 378 + eps * w)
    return float((v @ h @ v - 2 * b @ v + c) / n)


def analytic_history(y, w, x, h, b, c, n):
    k = x @ x.T / 378 + eps * np.eye(x.shape[0])
    p = y.T @ w
    v = np.linalg.solve(k, x @ p / 378 + eps * w)
    a = 2 * (h @ v - b) / n
    t = np.linalg.solve(k.T, a)
    qp = x.T @ t / 378
    rterms, rgrad = losses_and_score_gradients(p)
    combined_score = qp + .1 * rgrad['R']
    return {
        'Q': float((v @ h @ v - 2 * b @ v + c) / n), 'R': rterms['R'],
        'Y': np.outer(w, combined_score),
        'w': y @ combined_score + eps * t,
        'Q_Y': np.outer(w, qp), 'Q_w_via_p': y @ qp, 'Q_w_direct': eps * t,
        'R_Y': np.outer(w, rgrad['R']), 'R_w': y @ rgrad['R'],
    }


def full_history(y, w, x, h, b, c, n):
    r = losses_and_score_gradients(y.T @ w)[0]['R']
    return compressed(y, w, x, h, b, c, n) + .1 * r


scores = np.linspace(-1.3, 1.4, 378)
terms, grads = losses_and_score_gradients(scores)
scalar = scalar_reference(scores.tolist())
for name in ('B', 'R'):
    compare('scalar_vs_vector_' + name, terms[name], scalar[name], 2e-13)
zero, _ = losses_and_score_gradients(np.zeros(378))
compare('zero_B_log27_plus_1p5_log2', zero['B'], math.log(27) + 1.5 * math.log(2), 2e-14)
compare('zero_R_log2', zero['R'], math.log(2), 2e-14)
direction = np.cos(np.arange(378)) / np.sqrt(378)
step = 1e-6
for name in ('B', 'R'):
    plus = losses_and_score_gradients(scores + step * direction)[0][name]
    minus = losses_and_score_gradients(scores - step * direction)[0][name]
    compare('score_directional_derivative_' + name, (plus - minus) / (2 * step),
            grads[name] @ direction, 3e-9)

rng = np.random.default_rng(20261006)
x, y, z1, z2 = [rng.uniform(-.7, .7, (32, 378)) for _ in range(4)]
for z in (x, y, z1, z2):
    z[-1] = 1
x[-2] *= .002  # An observed but poorly conditioned feature direction.
w = rng.uniform(-.3, .3, 32)
h1, b1, c1 = statistics(z1)
h2, b2, c2 = statistics(z2)
h, b, c, n = h1 + h2, b1 + b2, c1 + c2, 2
analytic = analytic_history(y, w, x, h, b, c, n)
dy = rng.normal(size=y.shape)
dy[-1] = 0  # The feature's constant row is not a trainable feature direction.
dy /= np.linalg.norm(dy)
dw = rng.normal(size=32)
dw /= np.linalg.norm(dw)
fd_y = (full_history(y + step * dy, w, x, h, b, c, n)
        - full_history(y - step * dy, w, x, h, b, c, n)) / (2 * step)
fd_w = (full_history(y, w + step * dw, x, h, b, c, n)
        - full_history(y, w - step * dw, x, h, b, c, n)) / (2 * step)
compare('combined_history_Y_direction', fd_y, np.sum(analytic['Y'] * dy), 3e-9)
compare('combined_history_w_direction', fd_w, analytic['w'] @ dw, 3e-9)

k = x @ x.T / 378 + eps * np.eye(32)
p_fixed = y.T @ w


def separate_w_paths(w_argument, direct):
    p = p_fixed if direct else y.T @ w_argument
    regularizer_w = w_argument if direct else w
    v = np.linalg.solve(k, x @ p / 378 + eps * regularizer_w)
    return (v @ h @ v - 2 * b @ v + c) / n


for name, direct in (('Q_w_via_p', False), ('Q_w_direct', True)):
    fd = (separate_w_paths(w + step * dw, direct)
          - separate_w_paths(w - step * dw, direct)) / (2 * step)
    compare(name + '_finite_difference', fd, analytic[name] @ dw, 3e-9)
compare('Q_statistics_constant', c, 10., 2e-13)
compare('Q_division_preserves_replicated_history',
        compressed(y, w, x, 48 * h, 48 * b, 48 * c, 48 * n), analytic['Q'], 3e-13)

# Canonical 32-dimensional embedding of the already reviewed 2D null example.
blind_x = np.zeros((32, 378))
blind_x[0] = .5 * (2 * labels - 1)
blind_x[-1] = 1
idx = [edges.index(e) for e in ((0, 3), (0, 1), (4, 5))]
blind_x[0, idx[1]] = 0
delta = np.zeros(378)
delta[idx] = [.2, -.4, .2]
blind_w = np.zeros(32)
blind_w[0] = 1
blind_y = blind_x.copy()
blind_y[0] += delta
blind_h, blind_b, blind_c = statistics(blind_x)
before = losses_and_score_gradients(blind_x.T @ blind_w)[0]
after, after_grad = losses_and_score_gradients(blind_y.T @ blind_w)
q_before = compressed(blind_x, blind_w, blind_x, blind_h, blind_b, blind_c, 1)
q_after = compressed(blind_y, blind_w, blind_x, blind_h, blind_b, blind_c, 1)
compare('old_blind_X_delta_zero', blind_x @ delta, np.zeros(32), 2e-15)
compare('old_blind_Q_same', q_after, q_before, 3e-13)
blind_example = {
    'edges': [list(edges[i]) for i in idx], 'labels': labels[idx].tolist(),
    'scores_before': (blind_x.T @ blind_w)[idx].tolist(),
    'scores_after': (blind_y.T @ blind_w)[idx].tolist(),
    'Q_before': q_before, 'Q_after': q_after,
    'R_before': before['R'], 'R_after': after['R'],
    'dR_d_amplitude_at_1': float(after_grad['R'] @ delta),
    'weighted_dR_d_amplitude_at_1': float(.1 * after_grad['R'] @ delta),
    'Q_and_R_values_are_not_MAP': True,
    'feature_values_inside_tanh_range_except_constant_row': bool(np.max(np.abs(blind_y[:-1])) < 1),
}
checks.append({'name': 'R_sees_old_harmful_null_example',
               'pass': after['R'] > before['R'] and float(after_grad['R'] @ delta) > 0})

# Aggregate ranking loss can improve while one already-correct comparison flips.
# The extra two positive edges trade scores while staying in ker(X); this is a
# score-space limitation, not a claim that a particular encoder realizes it.
tradeoff_y = blind_x.copy()
bad_positive, good_positive = edges.index((12, 13)), edges.index((14, 15))
tradeoff_y[0, [bad_positive, good_positive]] = [-.8, .8]
tradeoff_delta = delta.copy()
tradeoff_delta[[bad_positive, good_positive]] += [.7, -.7]
tradeoff_after_y = tradeoff_y.copy()
tradeoff_after_y[0] += tradeoff_delta
tradeoff_before = losses_and_score_gradients(tradeoff_y[0])[0]
tradeoff_after = losses_and_score_gradients(tradeoff_after_y[0])[0]
tradeoff = {
    'maximum_X_delta': float(np.max(np.abs(blind_x @ tradeoff_delta))),
    'Q_before': compressed(tradeoff_y, blind_w, blind_x, blind_h, blind_b, blind_c, 1),
    'Q_after': compressed(tradeoff_after_y, blind_w, blind_x, blind_h, blind_b, blind_c, 1),
    'R_before': tradeoff_before['R'], 'R_after': tradeoff_after['R'],
    'specific_positive_minus_negative_before': float(tradeoff_y[0, idx[1]] - tradeoff_y[0, idx[0]]),
    'specific_positive_minus_negative_after': float(tradeoff_after_y[0, idx[1]] - tradeoff_after_y[0, idx[0]]),
    'feature_max_abs_nonconstant': float(np.max(np.abs(tradeoff_after_y[:-1]))),
}
compare('aggregate_tradeoff_Q_same', tradeoff['Q_after'], tradeoff['Q_before'], 3e-13)

# R is exactly invariant to common score shifts. Its finite-score Hessian is a
# positive-weight comparison-graph Laplacian; connectivity implies rank 377.
compare('R_common_shift_invariance', losses_and_score_gradients(scores + .31)[0]['R'], terms['R'], 2e-14)
adjacency = [set() for _ in edges]
for pos, neg in queries:
    for a in pos:
        for negative in neg:
            adjacency[int(a)].add(int(negative))
            adjacency[int(negative)].add(int(a))
seen = {0}
frontier = [0]
while frontier:
    node = frontier.pop()
    for other in adjacency[node] - seen:
        seen.add(other)
        frontier.append(other)

# Exact valid-group conflict: p=+-d/2, X=Y with one +-0.5 feature and a constant.
# Then T=I, Q=1.25*(d-2)^2 and R=softplus(-d) for every query.
conflict_x = np.zeros((32, 378))
conflict_x[0] = .5 * (2 * labels - 1)
conflict_x[-1] = 1
conflict_h, conflict_b, conflict_c = statistics(conflict_x)
margin = 6.
conflict_w = np.zeros(32)
conflict_w[0] = margin
conflict_Q = compressed(conflict_x, conflict_w, conflict_x,
                        conflict_h, conflict_b, conflict_c, 1)
conflict_R = losses_and_score_gradients(conflict_x.T @ conflict_w)[0]['R']
compare('valid_group_Q_conflict_formula', conflict_Q, 1.25 * (margin - 2) ** 2, 1e-12)
compare('valid_group_R_conflict_formula', conflict_R, math.log1p(math.exp(-margin)), 1e-14)
conflict = {
    'margin': margin, 'Q': conflict_Q, 'R': conflict_R,
    'dQ_d_margin': 2.5 * (margin - 2),
    'd_0p1R_d_margin': -.1 / (1 + math.exp(margin)),
    'Q_to_weighted_R_derivative_magnitude_ratio': 2.5 * (margin - 2) / (.1 / (1 + math.exp(margin))),
    'combined_history_dJ_d_margin': 2.5 * (margin - 2) - .1 / (1 + math.exp(margin)),
    'weighted_R_score_gradient_l1_bound': .2,
    'bound_is_not_a_parameter_gradient_or_objective_balance_guarantee': True,
}

result = {
    'status': 'PASS' if all(c['pass'] for c in checks) else 'FAIL',
    'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                    'torch_used': False, 'project_imports': False, 'formal_data_used': False,
                    'scope': 'Web workspace algebraic handwritten fixtures; not project Linux/BGE evidence'},
    'fixture': {'accounts': 28, 'edges': len(edges), 'positive_edges': int(labels.sum()),
                'query_positive_degrees': [len(p) for p, _ in queries],
                'total_query_comparisons': sum(len(p) * len(n) for p, n in queries)},
    'checks': checks,
    'objective_reduction': {'vectorized': terms, 'independent_stdlib': scalar,
                            'correct_R_minus_wrong_global_R': terms['R'] - terms['wrong_global_comparison_R'],
                            'zero_scores': zero},
    'history_gradients': {'Q': analytic['Q'], 'R': analytic['R'],
        'Q_w_via_p_norm': float(np.linalg.norm(analytic['Q_w_via_p'])),
        'Q_w_direct_norm': float(np.linalg.norm(analytic['Q_w_direct'])),
        'R_Y_norm': float(np.linalg.norm(analytic['R_Y'])), 'R_w_norm': float(np.linalg.norm(analytic['R_w'])),
        'combined_Y_direction_fd': float(fd_y), 'combined_Y_direction_analytic': float(np.sum(analytic['Y'] * dy)),
        'combined_w_direction_fd': float(fd_w), 'combined_w_direction_analytic': float(analytic['w'] @ dw),
        'constant_feature_row_was_not_perturbed': True},
    'old_Q_blind_example': blind_example,
    'aggregate_tradeoff_example': tradeoff,
    'R_invariance': {'score_nodes': 378, 'connected_component_from_first_node': len(seen),
                    'finite_score_hessian_kernel': 'span(all-ones), by connected weighted-Laplacian argument',
                    'does_not_certify_unseen_or_evicted_group_behavior': True},
    'large_margin_conflict': conflict,
    'limitations': [
        'Independent formulas and finite differences do not execute Torch autograd or certify the production model.',
        'The new current B differs from the historical square proxy; H/b/c/N do not summarize B or R.',
        'Score/feature-space examples do not assert exact reachability by the fixed BGE/head parameterization.',
        'R sees some harmful cache changes but does not protect every comparison or cache-external examples.',
        'No coefficient tuning, effects training, formal-data read, or authorization was performed.'],
}
(out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result['status'] == 'PASS' else 1)
