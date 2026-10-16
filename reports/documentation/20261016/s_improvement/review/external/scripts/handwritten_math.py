#!/usr/bin/env python3
"""Reviewer-only algebra checks. No project imports, model, optimizer or data reads.

Input contains explicitly constructed score coordinates and a legal handwritten graph.
Finite-difference tolerances are fixed here before the first execution.
This is NOT an S_pos/H_pos implementation or project-native qualification.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATOL64 = 2e-12
ATOL32 = 3e-5
FD_EPS = 1e-5
FD_ATOL = 2e-8

def logsumexp_weights(x: np.ndarray) -> tuple[np.generic, np.ndarray]:
    shift = np.max(x)
    weights = np.exp(x-shift)
    total = np.sum(weights, dtype=x.dtype)
    return shift+np.log(total), weights/total

def query(z, positives, kind: str, dtype=np.float64):
    z = np.asarray(z, dtype=dtype)
    positives = np.asarray(positives, dtype=np.int64)
    assert z.shape == (27,) and len(positives) in (1, 2)
    assert len(set(positives.tolist())) == len(positives)
    assert np.isfinite(z).all()
    negative = np.asarray([j for j in range(27) if j not in positives], dtype=np.int64)
    m = dtype(len(positives))
    if kind == 'old':
        value, grad = logsumexp_weights(z)
        value = value-np.mean(z[positives], dtype=dtype)
        grad[positives] -= dtype(1)/m
    elif kind == 'new':
        value = dtype(0)
        grad = np.zeros_like(z)
        for p in positives:
            v, w = logsumexp_weights(np.r_[np.asarray([0], dtype=dtype), z[negative]-z[p]])
            value += v/m
            grad[p] -= np.sum(w[1:], dtype=dtype)/m
            grad[negative] += w[1:]/m
    else:
        raise ValueError(kind)
    return float(value), grad

def graph(edges, kind: str):
    mat = np.zeros((28, 28), dtype=np.float64)
    lab = np.zeros((28, 28), dtype=np.int64)
    for i, j, score, label in edges:
        mat[i, j] = mat[j, i] = score
        lab[i, j] = lab[j, i] = label
    directed = np.zeros_like(mat)
    losses = []
    for q in range(28):
        candidates = np.asarray([j for j in range(28) if j != q])
        pos = np.flatnonzero(lab[q, candidates])
        value, grad = query(mat[q, candidates], pos, kind)
        losses.append(value)
        directed[q, candidates] = grad/28
    return float(np.mean(losses)), np.asarray([directed[i, j]+directed[j, i] for i,j,_,_ in edges]), losses

def main() -> None:
    started = time.perf_counter()
    raw = (ROOT/'checks/handwritten_inputs.json').read_bytes()
    inputs = json.loads(raw)
    output = {'scope': 'Reviewer handwritten scalar algebra only; no project/native qualification, no performance statistics',
              'input_sha256': hashlib.sha256(raw).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'tolerances': {'float64_identity_atol': ATOL64, 'float32_identity_atol': ATOL32,
                             'finite_difference_epsilon': FD_EPS, 'finite_difference_atol': FD_ATOL},
              'queries': [], 'finite_differences': [], 'expected_invalid_variants': {}}
    for case in inputs['queries']:
        item = {'name': case['name'], 'by_dtype': {}}
        for dt in (np.float64, np.float32):
            vo, go = query(case['scores'], case['positive_indices'], 'old', dt)
            vn, gn = query(case['scores'], case['positive_indices'], 'new', dt)
            assert np.isfinite([vo,vn]).all() and np.isfinite(go).all() and np.isfinite(gn).all()
            if len(case['positive_indices']) == 1:
                atol = ATOL64 if dt == np.float64 else ATOL32
                # Large offset cancellation is tested only for finiteness in FP32.
                if max(abs(x) for x in case['scores']) < 100:
                    assert abs(vo-vn) < atol and np.max(np.abs(go-gn)) < atol
            item['by_dtype'][dt.__name__] = {'old_loss': vo, 'new_loss': vn,
                'old_gradient': go.tolist(), 'new_gradient': gn.tolist(),
                'old_gradient_sum': float(go.sum()), 'new_gradient_sum': float(gn.sum())}
        output['queries'].append(item)
    for case in inputs['queries'][:3]:
        z = np.asarray(case['scores'], dtype=np.float64)
        for kind in ('old','new'):
            _, analytic = query(z, case['positive_indices'], kind)
            numeric = np.zeros(27)
            for j in range(27):
                zp, zm = z.copy(), z.copy()
                zp[j] += FD_EPS; zm[j] -= FD_EPS
                numeric[j] = (query(zp, case['positive_indices'], kind)[0]-query(zm, case['positive_indices'], kind)[0])/(2*FD_EPS)
            err = float(np.max(np.abs(numeric-analytic)))
            assert err < FD_ATOL, (case['name'], kind, err)
            output['finite_differences'].append({'case': case['name'], 'kind': kind, 'coordinates': 27, 'max_abs_error': err})
    asym = inputs['queries'][1]
    equal = inputs['queries'][2]
    vo, go = query(asym['scores'], [0,1], 'old')
    vn, gn = query(asym['scores'], [0,1], 'new')
    assert go[0] > 0 and gn[0] < 0 and gn[1] < 0
    assert abs(gn[1]) < abs(go[1])  # harder positive may receive LESS attraction
    assert np.all(gn[2:] > go[2:])
    _, geo = query(equal['scores'], [0,1], 'old')
    _, gen = query(equal['scores'], [0,1], 'new')
    assert geo[0] < 0 and geo[1] < 0 and np.all(gen[2:] > geo[2:])
    # Independent algebraic decomposition in a moderate, non-overflow example.
    z = np.asarray(asym['scores'], dtype=np.float64)
    Q = float(np.exp(z[:2]).sum()); A = float(np.exp(z[2:]).sum())
    r = np.exp(z[:2])/Q
    kl = float(np.mean(np.log(.5/r)))
    reconstructed = float(np.log1p(A/Q)+np.log(2)+kl)
    assert abs(reconstructed-vo) < ATOL64
    output['old_loss_decomposition'] = {'log_one_plus_negative_over_positive': float(np.log1p(A/Q)),
        'log_m': float(np.log(2)), 'uniform_to_positive_KL': kl, 'sum': reconstructed, 'old_loss': vo}
    # Expected wrong expressions, not failed scientific trials.
    omitted_z = z.copy()
    neg = np.arange(2,27)
    omitted_v, omitted_w = logsumexp_weights(np.r_[0., omitted_z[neg]-omitted_z[0]])
    omitted_g = np.zeros(27); omitted_g[0] = -omitted_w[1:].sum(); omitted_g[neg] = omitted_w[1:]
    mutants = {'old_minus_log_m': (vo-np.log(2),go),
               'missing_second_positive_not_relabelled': (float(omitted_v), omitted_g),
               'other_true_positive_in_each_denominator': (vo,go),
               'extra_divide_by_27': (vn/27,gn/27)}
    for name, (v,g) in mutants.items():
        err = float(np.max(np.abs(g-gn)))
        assert err > 1e-6  # derivative check alone rejects each on this fixed case
        output['expected_invalid_variants'][name] = {'loss':float(v),'max_gradient_difference_from_new':err,
                                                   'rejected_by_derivative':True}
    edges = inputs['graph']['edges']
    assert len(edges) == 378 and sum(int(e[3]) for e in edges) == 20
    degree = np.zeros(28,dtype=int)
    for i,j,_,label in edges:
        degree[i] += label; degree[j] += label
    assert (degree==1).sum()==16 and (degree==2).sum()==12
    results = {}
    for kind in ('old','new'):
        value, analytic, query_values = graph(edges,kind)
        checked = []
        for i,j in inputs['graph']['finite_difference_edges']:
            k = next(k for k,e in enumerate(edges) if e[:2]==[i,j])
            plus = [e.copy() for e in edges]; minus = [e.copy() for e in edges]
            plus[k][2] += FD_EPS; minus[k][2] -= FD_EPS
            numerical = (graph(plus,kind)[0]-graph(minus,kind)[0])/(2*FD_EPS)
            error = abs(numerical-analytic[k])
            assert error < FD_ATOL, (kind,i,j,error)
            checked.append({'edge':[i,j], 'two_query_gradient':float(analytic[k]),
                            'finite_difference':numerical,'absolute_error':error})
        results[kind] = {'loss':value,'query_values':query_values,'checked_edges':checked,
                         'all_edge_gradients':analytic.tolist()}
    assert abs(results['old']['query_values'][0]-results['new']['query_values'][0]) < ATOL64
    mixed = next(k for k,e in enumerate(edges) if e[:2]==[0,16])
    assert abs(results['old']['all_edge_gradients'][mixed]-results['new']['all_edge_gradients'][mixed]) > 1e-6
    output['graph'] = {'degrees':degree.tolist(), 'results':results,
                      'single_positive_query_0_value_unchanged_but_mixed_edge_0_16_changed':True}
    output['environment'] = {'python':sys.version, 'numpy':np.__version__, 'platform':platform.platform(),
        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),
        'imports_scientific_packages':['numpy'], 'project_modules_imported':False,
        'optimizer_calls':0, 'model_loads':0, 'formal_data_reads':0}
    output['status']='PASS_HANDWRITTEN_ALGEBRA_ONLY'
    output['elapsed_seconds']=time.perf_counter()-started
    (ROOT/'checks/handwritten_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':output['status'],'elapsed_seconds':output['elapsed_seconds'],
        'asymmetric':output['queries'][1]['by_dtype']['float64'],
        'max_query_fd_error':max(v['max_abs_error'] for v in output['finite_differences']),
        'max_graph_fd_error':max(v['absolute_error'] for r in results.values() for v in r['checked_edges'])},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
