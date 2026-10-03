"""Independent scientific oracle. Handwritten data only; no archive or model load.

Risks, analytic logit derivatives, hard-negative objective and AdamW arithmetic
are implemented independently in NumPy/scalar operations. PyTorch is used for
the real production text-to-logit graph, and to transport independently derived
logit gradients to model parameters. This is a Tiny CPU check, not native BGE.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from unittest import mock

import numpy as np
import torch

INPUT = Path(os.environ.get('RISK_AUDIT_INPUT', Path(__file__).resolve().parents[2] / 'input')).resolve()
sys.path.insert(0, str(INPUT / 'scripts'))
import step28_risk_replay as production

N = 28
EDGES = list(itertools.combinations(range(N), 2))
CLASSES = [i // 2 for i in range(16)] + [8 + (i - 16) // 3 for i in range(16, 28)]
LABELS = tuple(int(CLASSES[i] == CLASSES[j]) for i, j in EDGES)
INCIDENT = [[e for e, (i, j) in enumerate(EDGES) if q == i or q == j] for q in range(N)]
OUT = None
RESULTS = {}


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -x))


def scalar_risks(values, labels=LABELS, jacobian=False):
    """Enumerate incident undirected edges; no dense Torch scores/masks."""
    values = np.asarray(values, dtype=np.float64)
    risks = np.zeros((N, 3), dtype=np.float64)
    degrees = np.zeros(N, dtype=np.int64)
    jac = np.zeros((N, 3, len(EDGES)), dtype=np.float64)
    for q, ids in enumerate(INCIDENT):
        yes = [e for e in ids if labels[e] == 1]
        no = [e for e in ids if labels[e] == 0]
        pos, neg, all_scores = values[yes], values[no], values[ids]
        maximum = float(max(all_scores))
        exponentials = np.exp(all_scores - maximum)
        partition = math.fsum(exponentials)
        risks[q, 0] = maximum + math.log(partition) - math.fsum(pos) / len(yes)
        risks[q, 1] = math.fsum(np.logaddexp(0.0, -pos)) / len(yes)
        risks[q, 2] = math.fsum(np.logaddexp(0.0, neg)) / len(no)
        degrees[q] = len(yes)
        if jacobian:
            jac[q, 0, ids] = exponentials / partition
            jac[q, 0, yes] -= 1.0 / len(yes)
            jac[q, 1, yes] = -sigmoid(-pos) / len(yes)
            jac[q, 2, no] = sigmoid(neg) / len(no)
    return (risks, degrees, jac) if jacobian else (risks, degrees)


def reference_from_risks(risks, degrees):
    return {str(m): [[float(x) for x in row] for row in
                    np.sort(risks[degrees == m], axis=0, kind='stable')]
            for m in (1, 2)}


def scalar_distribution(risks, degrees, reference, derivative=False):
    """Each channel independently pairs scalar empirical order statistics."""
    total = [0.0, 0.0, 0.0]
    grad = np.zeros((N, 3), dtype=np.float64)
    for m in (1, 2):
        members = [q for q in range(N) if degrees[q] == m]
        for c in range(3):
            order = sorted(members, key=lambda q: (float(risks[q, c]), q))
            for k, q in enumerate(order):
                excess = max(0.0, float(risks[q, c]) - float(reference[str(m)][k][c]))
                total[c] += excess * excess / N
                grad[q, c] = 2.0 * excess / (3 * N)
    return (np.asarray(total), grad) if derivative else np.asarray(total)


def risk_value_grad(values, reference, labels=LABELS):
    risks, degrees, jac = scalar_risks(values, labels, jacobian=True)
    channels, derivative = scalar_distribution(risks, degrees, reference, derivative=True)
    return float(channels.mean()), np.einsum('qc,qce->e', derivative, jac), channels


def scalar_base(values, labels=LABELS, derivative=False):
    values = np.asarray(values, dtype=np.float64)
    risks, degrees, jac = scalar_risks(values, labels, jacobian=True)
    bce = float(np.logaddexp(0.0, values).mean() - np.dot(labels, values) / len(values))
    rank = float(risks[:, 0].mean())
    g = (sigmoid(values) - np.asarray(labels)) / len(values) + jac[:, 0].mean(axis=0)
    hard = 0.0
    for q, ids in enumerate(INCIDENT):
        yes = [e for e in ids if labels[e] == 1]
        no = [e for e in ids if labels[e] == 0]
        # The secondary key is the candidate index, independent of pair storage.
        def key(e):
            i, j = EDGES[e]
            return (-float(values[e]), j if i == q else i)
        hardest = sorted(no, key=key)[:5]
        for pe in yes:
            for ne in hardest:
                delta = values[ne] - values[pe]
                hard += float(np.logaddexp(0.0, delta)) / (N * len(yes) * 5)
                amount = 0.5 * float(sigmoid(delta)) / (N * len(yes) * 5)
                g[ne] += amount
                g[pe] -= amount
    result = {'bce': bce, 'rank': rank, 'hard': hard, 'total': bce + rank + 0.5 * hard}
    return (result, g) if derivative else result


def finite_differences(fn, values, epsilon=1e-5):
    out = np.zeros_like(values, dtype=np.float64)
    for e in range(len(values)):
        left, right = np.array(values, dtype=np.float64), np.array(values, dtype=np.float64)
        left[e] -= epsilon
        right[e] += epsilon
        out[e] = (fn(right) - fn(left)) / (2 * epsilon)
    return out


def group(uid, shift=0):
    # Group IDs and seller IDs are alignment metadata. Only these handmade title
    # and description strings enter the production tokenizer/encoder path.
    rows = []
    for i in range(N):
        rows.append(tuple((f'{uid}_item_{i:02d}_{j:02d}',
                           f'款{shift % 53}型号{(i * 7 + shift) % 29}版本{j}人工测试商品',
                           f'批{shift}色{(i + shift) % 17}容量{j + 2} 手写描述主题{CLASSES[i]}。')
                          for j in range(2 + (i + shift) % 3)))
    result = production.data.Group(uid, tuple(f'{uid}_seller_{i:02d}' for i in range(N)), tuple(rows), LABELS)
    result.validate()
    return result


class IndependentTinyEncoder(torch.nn.Module):
    """Different fixture from the submitted Tiny: Unicode tokenizer + tanh map."""
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(101, 5)
        self.project = torch.nn.Linear(5, 6)
        self.drop = torch.nn.Dropout(0.17)

    def tokenizer(self, texts, **kwargs):
        token_rows = [[(ord(ch) * 7 + i) % 101 for i, ch in enumerate(t)] for t in texts]
        width = max(map(len, token_rows))
        ids = torch.tensor([row + [0] * (width - len(row)) for row in token_rows])
        mask = torch.tensor([[1.0] * len(row) + [0.0] * (width - len(row)) for row in token_rows])
        return {'input_ids': ids, 'attention_mask': mask}

    def forward(self, batch):
        emb = self.drop(torch.tanh(self.project(self.emb(batch['input_ids']))))
        mask = batch['attention_mask'].unsqueeze(-1)
        return {'sentence_embedding': (emb * mask).sum(1) / mask.sum(1)}


def prepared():
    torch.manual_seed(47031)
    model = production.core.build_model(IndependentTinyEncoder(), 24, 11)
    c = production.parent.config(production.parent.contract())
    c['input']['encoder_bf16'] = False
    c['optimizer']['clip_norm'] = 0.08  # Deliberately active clipping probe.
    optimizer = production.core.make_optimizer(model, c)
    old, current = group('reference_old', 3), group('reference_current', 26)
    production.parent.update(model, optimizer, old, None, None, c, 'seq', 1, 1, 1701, 1702)
    # One real warm update, then an explicitly artificial step-counter fixture.
    for state in optimizer.state.values():
        state['step'].fill_(288)
    teacher_scores = production.parent.ranking.score(model, [old], c)[0]
    risks, counts = scalar_risks(teacher_scores)
    ref = reference_from_risks(risks * np.array([0.76, 0.67, 0.55]), counts)
    return model, optimizer, c, old, current, ref


def assert_allclose(actual, expected, *, atol, rtol=0.0, label=''):
    a, e = np.asarray(actual), np.asarray(expected)
    try:
        np.testing.assert_allclose(a, e, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise AssertionError(f'{label}: max error {np.max(np.abs(a-e))}\n{error}') from error


def record(name, data):
    RESULTS[name] = data
    print(json.dumps({'check': name, **data}, ensure_ascii=False, allow_nan=False), flush=True)


def test_formula(method):
    rng = np.random.default_rng(83811)
    scores = rng.normal(0.2, 2.2, 378)
    teacher = rng.normal(0.9, 1.5, 378)
    teacher_risks, counts = scalar_risks(teacher)
    ref = reference_from_risks(teacher_risks * [0.89, 0.77, 0.96], counts)
    expected, expected_counts, _ = scalar_risks(scores, jacobian=True)
    live = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
    got, degrees = method.query_risks(live, torch.tensor(LABELS))
    assert_allclose(got.detach().numpy(), expected, atol=2e-14, label='risk values')
    assert_allclose(degrees.numpy(), expected_counts, atol=0, label='strata')
    penalty = method.distribution_penalty(got, degrees, ref)
    val, grad, channels = risk_value_grad(scores, ref)
    assert_allclose(penalty.detach().numpy(), channels, atol=2e-14, label='channel penalties')
    penalty.mean().backward()
    numeric = finite_differences(lambda x: risk_value_grad(x, ref)[0], scores)
    assert_allclose(live.grad.numpy(), grad, atol=1e-14, label='Torch risk logit gradient')
    assert_allclose(grad, numeric, atol=3e-9, rtol=2e-5, label='378 independent finite differences')
    base_values, base_gradient = scalar_base(scores, derivative=True)
    base_live = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
    actual_base = method.parent.ranking.objectives(base_live, torch.tensor(LABELS, dtype=torch.float64), .5)
    actual_base['total'].backward()
    for key in base_values:
        assert_allclose(actual_base[key].item(), base_values[key], atol=2e-14, label=key)
    assert_allclose(base_live.grad.numpy(), base_gradient, atol=2e-14, label='full base gradient')
    numeric_base = finite_differences(lambda x: scalar_base(x)['total'], scores)
    assert_allclose(base_gradient, numeric_base, atol=3e-9, rtol=2e-5, label='base finite differences')
    # Extreme finite logits test stable logsumexp and softplus, not just mild inputs.
    extremes = rng.choice([-700.0, 700.0], size=378)
    er, ec = scalar_risks(extremes)
    xr, xc = method.query_risks(torch.tensor(extremes), torch.tensor(LABELS))
    assert_allclose(xr.numpy(), er, atol=5e-12, label='extreme logits')
    np.savez(OUT / 'formula_inputs_outputs.npz', scores=scores, teacher=teacher,
             labels=LABELS, expected_risks=expected, expected_grad=grad,
             torch_grad=live.grad.numpy(), finite_diff_grad=numeric, base_grad=base_gradient,
             finite_diff_base=numeric_base, extreme_logits=extremes)
    (OUT / 'formula_reference.json').write_text(json.dumps(ref, indent=2))
    record('formula', {'passed': True, 'logits': 378, 'risk_values_max_error': float(np.max(abs(got.detach().numpy()-expected))),
                       'analytic_vs_torch_gradient_max_error': float(np.max(abs(live.grad.numpy()-grad))),
                       'finite_difference_gradient_max_error': float(np.max(abs(numeric-grad))),
                       'base_finite_difference_max_error': float(np.max(abs(numeric_base-base_gradient))),
                       'channels': channels.tolist(), 'penalty_mean': val, 'dtype': 'float64'})


def test_properties(method):
    counts = np.array([1] * 16 + [2] * 12)
    risks = np.column_stack((np.arange(28) / 30 + 1.0, np.arange(28) / 70 + .1,
                             np.arange(28) / 90 + .01))
    ref = reference_from_risks(risks, counts)
    def calc(x, reference=ref):
        return method.distribution_penalty(torch.tensor(x), torch.tensor(counts), reference).numpy()
    assert_allclose(calc(risks), [0, 0, 0], atol=0, label='identity')
    assert_allclose(calc(risks * .8), [0, 0, 0], atol=0, label='improvement')
    swapped = risks.copy(); swapped[[0, 15]] = swapped[[15, 0]]
    assert_allclose(calc(swapped), [0, 0, 0], atol=0, label='individual swap')
    tail = risks.copy(); tail[0, 0] -= .01; tail[15, 0] += .01
    assert_allclose(tail[:, 0].mean(), risks[:, 0].mean(), atol=1e-15)
    assert_allclose(calc(tail), [.01 ** 2 / 28, 0, 0], atol=1e-16, label='equal mean tail degradation')
    across = risks.copy(); across[[0, 27]] = across[[27, 0]]
    expected = scalar_distribution(across, counts, ref)
    assert np.all(expected > 0)
    assert_allclose(calc(across), expected, atol=1e-15, label='cross-stratum no compensation')
    joint = risks.copy(); joint[:16, 1] = risks[:16, 1][::-1]
    assert_allclose(calc(joint), [0, 0, 0], atol=0, label='joint not protected')
    for c in range(3):
        changed = risks.copy(); changed[:, c] += .3
        target = np.zeros(3); target[c] = .09
        assert_allclose(calc(changed), target, atol=1e-15, label='independent channel')
    # At x=target, squared positive part is C1: derivative is zero on both sides.
    x = torch.tensor(risks, requires_grad=True)
    method.distribution_penalty(x, torch.tensor(counts), ref).mean().backward()
    assert_allclose(x.grad.numpy(), np.zeros((28, 3)), atol=0, label='squared-hinge derivative at zero')
    # Two equal live risks matched to different reference order statistics:
    # stable sort gives one limiting gradient; a central difference averages them.
    tie_ref_risk = risks.copy(); tie_ref_risk[0, 0] = .1; tie_ref_risk[1, 0] = .2
    tie_ref = reference_from_risks(tie_ref_risk, counts)
    tied = risks.copy(); tied[0, 0] = tied[1, 0] = .4
    x = torch.tensor(tied, requires_grad=True)
    method.distribution_penalty(x, torch.tensor(counts), tie_ref).mean().backward()
    eps = 1e-7
    center = calc(tied, tie_ref).mean()
    plus, minus = tied.copy(), tied.copy(); plus[0, 0] += eps; minus[0, 0] -= eps
    left = (center - calc(minus, tie_ref).mean()) / eps
    right = (calc(plus, tie_ref).mean() - center) / eps
    central = (calc(plus, tie_ref).mean() - calc(minus, tie_ref).mean()) / (2 * eps)
    branch = float(x.grad[0, 0])
    assert abs(branch - left) < 2e-9 and abs(left - right) > .001
    record('properties', {'passed': True, 'mean_unchanged_tail_penalty': calc(tail).tolist(),
             'individual_swap_penalty': calc(swapped).tolist(), 'joint_reassignment_penalty': calc(joint).tolist(),
             'cross_stratum_penalty': expected.tolist(), 'squared_hinge_zero_derivative': True,
             'sort_tie': {'stable_branch_gradient': branch, 'left_derivative': left,
                          'right_derivative': right, 'central_difference': central,
                          'classical_derivative_exists': False,
                          'interpretation': 'Stable sort selects a valid limiting branch; do not assert central finite differences at exact ties.'}})


def independent_parameter_gradient(model, current, old, c, ref, lam, gamma, cs=815, hs=927):
    expected = copy.deepcopy(model)
    expected.train(); expected.zero_grad(set_to_none=True)
    torch.manual_seed(cs)
    s_current = production.base.logits(expected, current, c, 'split_rank')
    lc, gc = scalar_base(s_current.detach().numpy(), derivative=True)
    s_current.backward(torch.tensor(gc, dtype=s_current.dtype))
    torch.manual_seed(hs)
    s_history = production.base.logits(expected, old, c, 'split_rank')
    lh, gh = scalar_base(s_history.detach().numpy(), derivative=True)
    keep, gr, channels = risk_value_grad(s_history.detach().numpy(), ref)
    increment = torch.autograd.grad(s_history, tuple(expected.parameters()),
                                    grad_outputs=torch.tensor(gamma * gr, dtype=s_history.dtype), retain_graph=True)
    s_history.backward(torch.tensor(lam * gh + gamma * gr, dtype=s_history.dtype))
    gradients = {n: p.grad.detach().numpy().copy() for n, p in expected.named_parameters()}
    pieces = {'current_base': lc, 'history_base': lh, 'channels': channels.tolist(),
              'retention': keep, 'total': lc['total'] + lam * lh['total'] + gamma * keep,
              'retention_gradient_norms': {n: float(v.norm()) for (n, p), v in zip(expected.named_parameters(), increment)}}
    return gradients, pieces


def manual_adam(model, optimizer, gradients, c, step=1):
    # NumPy float64 arithmetic for the documented AdamW update; references copied
    # from the actual prior optimizer state, not recomputed by another AdamW.
    norm = math.sqrt(math.fsum(float(np.square(g.astype(np.float64)).sum()) for g in gradients.values()))
    coefficient = min(1.0, float(c['optimizer']['clip_norm']) / (norm + 1e-6))
    names = {id(p): n for n, p in model.named_parameters()}
    target, target_state = {}, {}
    rates = [1e-5 * (step / 29 if step <= 29 else (288 - step) / 259), .001]
    for pg, lr in zip(optimizer.param_groups, rates):
        b1, b2 = pg['betas']; wd, eps = pg['weight_decay'], pg['eps']
        for param in pg['params']:
            name, state = names[id(param)], optimizer.state[param]
            theta = param.detach().numpy().astype(np.float64)
            g = gradients[name].astype(np.float64) * coefficient
            t = float(state['step']) + 1
            m = b1 * state['exp_avg'].numpy().astype(np.float64) + (1-b1) * g
            v = b2 * state['exp_avg_sq'].numpy().astype(np.float64) + (1-b2) * g*g
            target[name] = (theta * (1-lr*wd) - lr * (m / (1-b1**t)) / (np.sqrt(v / (1-b2**t)) + eps))
            target_state[name] = {'step': t, 'exp_avg': m, 'exp_avg_sq': v}
    return target, target_state, norm, coefficient


def test_update(method):
    model, optimizer, c, old, current, ref = prepared()
    lam, gamma = .23, .71
    gradients, pieces = independent_parameter_gradient(model, current, old, c, ref, lam, gamma)
    target, target_state, norm, coefficient = manual_adam(model, optimizer, gradients, c)
    assert coefficient < .95, 'The oracle must exercise actual clipping'
    captured, counts = {}, {'backward': 0, 'clip': 0, 'step': 0, 'forwards': []}
    original_clip = torch.nn.utils.clip_grad_norm_
    original_backward = torch.Tensor.backward
    original_step = optimizer.step
    original_logits = production.base.logits
    def capture(parameters, *a, **kw):
        parameters = list(parameters); counts['clip'] += 1
        captured.update({n: p.grad.detach().numpy().copy() for n, p in model.named_parameters()})
        return original_clip(parameters, *a, **kw)
    def backward(tensor, *a, **kw):
        counts['backward'] += 1
        return original_backward(tensor, *a, **kw)
    def step(*a, **kw):
        counts['step'] += 1
        return original_step(*a, **kw)
    def logits(m, g, *a, **kw):
        counts['forwards'].append({'uid': g.uid, 'training': bool(m.training)})
        return original_logits(m, g, *a, **kw)
    with mock.patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=capture), \
         mock.patch.object(torch.Tensor, 'backward', new=backward), \
         mock.patch.object(optimizer, 'step', side_effect=step), \
         mock.patch.object(production.base, 'logits', side_effect=logits):
        row = method.update(model, optimizer, current, old, ref, c, 2, 1, 815, 927,
                            history_weight=lam, retention_weight=gamma)
    assert counts['backward'] == 2 and counts['clip'] == 1 and counts['step'] == 1
    assert [x['uid'] for x in counts['forwards']] == [current.uid, old.uid]
    assert all(x['training'] for x in counts['forwards'])
    checks = {}
    # Write raw evidence before asserting; rejected mutants retain their evidence.
    np.savez(OUT / 'update_gradients.npz', **{'actual_'+n: g for n, g in captured.items()},
             **{'expected_'+n: g for n, g in gradients.items()})
    (OUT / 'update_row_and_reference.json').write_text(json.dumps({'row': row, 'expected': pieces, 'counts': counts,
                                            'lambda': lam, 'gamma': gamma, 'reference': ref}, indent=2))
    for name, p in model.named_parameters():
        state = optimizer.state[p]
        checks[name] = {'gradient_max_error': float(np.max(abs(captured[name]-gradients[name]))),
                        'parameter_max_error': float(np.max(abs(p.detach().numpy()-target[name]))),
                        'first_moment_max_error': float(np.max(abs(state['exp_avg'].numpy()-target_state[name]['exp_avg']))),
                        'second_moment_max_error': float(np.max(abs(state['exp_avg_sq'].numpy()-target_state[name]['exp_avg_sq'])))}
    (OUT / 'update_comparison.json').write_text(json.dumps(checks, indent=2))
    for name, p in model.named_parameters():
        assert_allclose(captured[name], gradients[name], atol=2e-6, rtol=2e-5, label=f'{name} preclip gradient')
        assert_allclose(p.detach().numpy(), target[name], atol=1.5e-7, label=f'{name} manual AdamW parameters')
        for key in ('exp_avg', 'exp_avg_sq'):
            assert_allclose(optimizer.state[p][key].numpy(), target_state[name][key], atol=2e-8, rtol=2e-5, label=f'{name} {key}')
        assert float(optimizer.state[p]['step']) == 289
    assert_allclose(row['total'], pieces['total'], atol=3e-6, label='logged scalar total')
    assert_allclose(row['gradient_norm'], norm, atol=3e-6, label='common gradient norm')
    enc_norm = sum(v for n, v in pieces['retention_gradient_norms'].items() if n.startswith('encoder.'))
    head_norm = sum(v for n, v in pieces['retention_gradient_norms'].items() if n.startswith('head.'))
    assert enc_norm > 1e-6 and head_norm > 1e-6
    record('real_update', {'passed': True, 'backend': torch.__version__, 'actual_updates_in_probe': 2,
                           'counter_fixture': 'One real warm update; Adam counter explicitly rebased 1->288; one risk update.',
                           'reference_fixture': 'Independent eval risk reference multiplied by [0.76,0.67,0.55] to activate retention; not a scientific stage target.',
                           'lambda': lam, 'gamma': gamma, 'preclip_norm': norm, 'clip_scale': coefficient,
                           'counts': counts, 'retention_encoder_norm_sum': enc_norm, 'retention_head_norm_sum': head_norm,
                           'per_parameter_comparison': checks})


def test_zero_gamma(method):
    import step28_er_weight as er
    start, opt, c, old, current, ref = prepared()
    models, states = [], []
    for tag in ('er', 'risk'):
        model = copy.deepcopy(start)
        optimizer = production.core.make_optimizer(model, c)
        optimizer.load_state_dict(copy.deepcopy(opt.state_dict()))
        if tag == 'er':
            er.update(model, optimizer, current, old, c, .1, 2, 1, 777, 778)
        else:
            method.update(model, optimizer, current, old, ref, c, 2, 1, 777, 778,
                          history_weight=.1, retention_weight=0.0)
        models.append(production.core.state_digest(model.state_dict()))
        states.append(production.core.state_digest(optimizer.state_dict()))
    assert models[0] == models[1] and states[0] == states[1]
    record('zero_gamma', {'passed': True, 'lambda': .1, 'model_equal': True, 'adam_equal': True})


def test_stage(method):
    model, optimizer, c, old, current, _ = prepared()
    p = production.parent.contract()
    first = [group(f'phase1_{k:02d}', k + 1) for k in range(48)]
    second = [group(f'phase2_{k:02d}', k + 100) for k in range(48)]
    memory = method.RiskMemory('ABC', p['memory_seed'], {'a': 1.0, 'b': 0.0})
    model.train()
    memory.retain(first, 1, model, optimizer, c)
    assert model.training
    origins = copy.deepcopy(memory.cache.auxiliary['risk_replay']['groups'])
    reference_error = 0.0
    # Direct per-UID independent reference check: groups have different texts.
    for g in memory.cache.reservoir.groups:
        score = production.parent.ranking.score(model, [g], c)[0]
        r, m = scalar_risks(score)
        expected = reference_from_risks(r, m)
        for degree in ('1', '2'):
            actual = origins[g.uid]['values'][degree]
            reference_error = max(reference_error, float(np.max(abs(np.asarray(actual)-expected[degree]))))
            assert_allclose(actual, expected[degree], atol=8e-7, label=f'group-reference alignment {g.uid}')
    baseline = production.parent.Memory('ABC', p['memory_seed'], False, {'a': 1.0, 'b': 0.0})
    baseline.retain(first, 1, None); baseline.begin_stage(2)
    expected_sequence, stream = production.parent.schedule(second, p, 'ABC', 2)
    observed, independent_first = [], {}
    original_update = method.update
    def inspect_update(m, o, cur, hist, ref, cc, stage, step, cs, hs, **kw):
        expected_g, _ = baseline.draw()
        assert hist.uid == expected_g.uid
        assert ref == origins[hist.uid]['values']
        assert cur.uid == expected_sequence[step-1].uid
        assert cs == production.data.seed_for(stream, step-1, 'dropout')
        assert hs == production.data.seed_for(p['memory_seed'], 'ABC', 2, step-1, 'history_dropout')
        assert kw['history_weight'] == .23, 'train_stage lost nondefault lambda'
        assert kw['retention_weight'] == .71, 'train_stage lost nondefault gamma'
        if step == 1:
            gradients, details = independent_parameter_gradient(m, cur, hist, cc, ref, .23, .71, cs, hs)
            target, state, norm, scale = manual_adam(m, o, gradients, cc)
            independent_first.update({'target': target, 'details': details})
        row = original_update(m, o, cur, hist, ref, cc, stage, step, cs, hs, **kw)
        if step == 1:
            for name, param in m.named_parameters():
                assert_allclose(param.detach().numpy(), independent_first['target'][name], atol=2e-7,
                                label='real train_stage first update '+name)
        observed.append({'step': step, 'current': cur.uid, 'history': hist.uid,
                         'lambda': kw['history_weight'], 'gamma': kw['retention_weight'], 'retention': row['retention']})
        return row
    with mock.patch.object(method, 'update', new=inspect_update):
        result = method.train_stage(model, optimizer, second, memory, c, 'ABC', 2,
                                    history_weight=.23, retention_weight=.71)
    assert len(observed) == 288 and production.parent.adam_step(optimizer) == 576
    assert result['physical_updates'] == 288 and result['updates'][-1]['encoder_lr'] == 0
    before = copy.deepcopy(memory.cache.auxiliary['risk_replay']['groups'])
    model.train()
    memory.retain(second, 2, model, optimizer, c)
    assert model.training
    live = memory.cache.auxiliary['risk_replay']['groups']
    assert set(live) == {g.uid for g in memory.cache.reservoir.groups}
    for uid in set(before) & set(live):
        assert live[uid] == before[uid], 'old reference refreshed'
    for g in memory.cache.reservoir.groups:
        if g.uid not in before:
            score = production.parent.ranking.score(model, [g], c)[0]
            r, degrees = scalar_risks(score)
            expected = reference_from_risks(r, degrees)
            for degree in ('1', '2'):
                assert_allclose(live[g.uid]['values'][degree], expected[degree], atol=1e-6,
                                label=f'newcomer reference alignment {g.uid}')
    payload = memory.to_bytes()
    assert len(payload) < 1048576
    restored = method.RiskMemory.from_bytes(payload)
    assert restored.to_bytes() == payload
    memory.begin_stage(3); restored.begin_stage(3)
    for _ in range(3):
        g, r = memory.draw(); gg, rr = restored.draw()
        assert g.uid == gg.uid and r == rr
    assert memory.to_bytes() == restored.to_bytes()
    (OUT / 'stage_updates.json').write_text(json.dumps(result, indent=2))
    (OUT / 'stage_observed_transmission.json').write_text(json.dumps(observed, indent=2))
    (OUT / 'memory_after_stage2.json').write_bytes(payload)
    record('real_stage', {'passed': True, 'actual_stage_updates': 288, 'lambda': .23, 'gamma': .71,
                         'adam_step_final': 576, 'first_stage_counter_fixture': True,
                         'handwritten_text_groups_distinct': True, 'max_reference_error': reference_error,
                         'surviving_old_references': len(set(before) & set(live)),
                         'new_surviving_references': len(set(live) - set(before)),
                         'serialized_bytes': len(payload), 'new_reference_per_uid_checked': True,
                         'old_references_not_refreshed': True, 'draws_restore_equal': True})


def test_restore(method):
    model, optimizer, c, old, current, ref = prepared()
    p = production.parent.contract()
    memory = method.RiskMemory('ABC', p['memory_seed'], {'a': 1.0, 'b': 0.0})
    first = [group(f'restore_{k:02d}', 500+k) for k in range(48)]
    memory.retain(first, 1, model, optimizer, c); memory.begin_stage(2)
    path = OUT / 'tiny_checkpoint.pt'
    torch.manual_seed(5591); np.random.seed(7329); random.seed(4321)
    rng = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate()}
    # core save/restore accepts metadata; NumPy RNG is kept as primitive lists to
    # satisfy torch weights_only, reproducing no formal runner/checkpoint path.
    metadata = {'handwritten': True, 'rng': {'torch': rng['torch'], 'python': rng['python'],
                   'numpy': [rng['numpy'][0], rng['numpy'][1].tolist(), rng['numpy'][2], rng['numpy'][3], rng['numpy'][4]]}}
    receipt = production.core.save_state(path, model, optimizer, metadata)
    payload = memory.to_bytes()
    outcomes = []
    for repeat in range(2):
        if repeat:
            loaded = production.core.restore_state(path, model, optimizer, receipt['state_sha256'])
            memory = method.RiskMemory.from_bytes(payload)
            torch.set_rng_state(loaded['rng']['torch']); random.setstate(loaded['rng']['python'])
            nr = loaded['rng']['numpy']; np.random.set_state((nr[0], np.array(nr[1], np.uint32), nr[2], nr[3], nr[4]))
        samples = [float(torch.rand(())), float(np.random.random()), random.random()]
        hist, reference = memory.draw()
        row = method.update(model, optimizer, current, hist, reference, c, 2, 1, 9182, 2391,
                            history_weight=.23, retention_weight=.71)
        outcomes.append({'rng_samples': samples, 'history_uid': hist.uid, 'row': row,
                         'model': production.core.state_digest(model.state_dict()),
                         'optimizer': production.core.state_digest(optimizer.state_dict()),
                         'memory': hashlib.sha256(memory.to_bytes()).hexdigest()})
    assert outcomes[0] == outcomes[1]
    (OUT / 'restore_outcomes.json').write_text(json.dumps(outcomes, indent=2))
    record('restore', {'passed': True, 'next_update_and_model_adam_memory_rng_identical': True,
                        'actual_risk_updates_in_probe': 2, 'checkpoint': str(path.name)})


TESTS = {'formula': test_formula, 'properties': test_properties, 'update': test_update,
         'zero_gamma': test_zero_gamma, 'stage': test_stage, 'restore': test_restore}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tests', default=','.join(TESTS))
    args = parser.parse_args()
    OUT = args.output; OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Explicit CPU-only environment required')
    if args.module:
        spec = importlib.util.spec_from_file_location('oracle_target', args.module)
        target = importlib.util.module_from_spec(spec); spec.loader.exec_module(target)
    else:
        target = production
    print(json.dumps({'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                       'cuda_available': torch.cuda.is_available(), 'threads': torch.get_num_threads(),
                       'module': str(Path(target.__file__).resolve()),
                       'module_sha256': hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest(),
                       'scope': 'handwritten Tiny CPU only; no native BGE or formal data'}, indent=2), flush=True)
    started = time.monotonic()
    try:
        # These loaders must remain uncalled, even if a future regression attempts
        # to accidentally expand this test into formal-data or real-model access.
        with (mock.patch.object(production.base.public, 'public_inputs', side_effect=AssertionError('No formal texts')),
              mock.patch.object(production.base.public, 'attach_labels', side_effect=AssertionError('No formal labels')),
              mock.patch.object(production.data, 'Archive', side_effect=AssertionError('No formal archive')),
              mock.patch.object(production.base, 'load_model', side_effect=AssertionError('No real model download/load'))):
            for name in args.tests.split(','):
                TESTS[name](target)
    finally:
        (OUT / 'results.json').write_text(json.dumps({'results': RESULTS, 'seconds': time.monotonic()-started},
                                                    ensure_ascii=False, indent=2, allow_nan=False))
    print('ALL SELECTED INDEPENDENT CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
