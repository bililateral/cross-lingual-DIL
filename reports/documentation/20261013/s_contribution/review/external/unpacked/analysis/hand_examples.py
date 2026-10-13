#!/usr/bin/env python3
"""Exact, small, hand-specified algebra checks. Standard library only; no project imports.
Usage: python hand_examples.py hand_examples_inputs.json OUTPUT_DIR
Outputs are fractions, not experimental estimates. Any failure is logged and re-raised.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import pathlib
import platform
import sys
import time
import traceback
from fractions import Fraction as Q

def partitions(records: tuple[int, ...], counts: tuple[int, ...]):
    if len(counts) == 1:
        assert len(records) == counts[0]
        yield (records,)
        return
    for chosen in itertools.combinations(records, counts[0]):
        left = tuple(x for x in records if x not in chosen)
        for tail in partitions(left, counts[1:]):
            yield (chosen,) + tail

def mean(values):
    values = list(values)
    if not values:
        raise ValueError('Empty mean')
    return sum(values, Q(0)) / len(values)

def score(x, y, kernel):
    assert set(x).isdisjoint(y)
    return mean(kernel(i, j) for i in x for j in y)

def arbitrary(i, j):
    return Q((i+1)*(j+1)+(i+j)%3, 7) if i != j else Q(0)

def run(inp):
    result = {'nature': inp['kind'], 'claims': []}
    counts = tuple(inp['toy_origin_counts'])
    origin = tuple(a for a, n in enumerate(counts) for _ in range(n))
    kernel = lambda i, j: Q(int(i != j and origin[i] == origin[j]))
    rows = []
    for bags in partitions(tuple(range(sum(counts))), counts):
        mixed = all(len({origin[r] for r in bag}) > 1 for bag in bags)
        valid_pairs = [(i,j) for i in bags[0] for j in bags[1] if origin[i] != origin[j]]
        cross = mean(kernel(i,j) for i,j in valid_pairs) if valid_pairs else None
        rows.append({'bags': bags, 'mixed_both': mixed, 'score': score(*bags, kernel),
                     'cross_origin_score': cross})
    p_mixed = Q(sum(row['mixed_both'] for row in rows), len(rows))
    expected_score = mean(row['score'] for row in rows)
    assert len(rows) == 6 and p_mixed == Q(2,3) and expected_score == Q(1,3)
    assert all(row['cross_origin_score'] == 0 for row in rows)
    result['two_by_two'] = {'allocations': rows, 'mixed_probability':p_mixed,
                           'expected_score':expected_score, 'original_score':Q(0)}
    cases = []
    for raw_counts in inp['partition_cases']:
        counts = tuple(raw_counts); n = sum(counts)
        origin = tuple(a for a, number in enumerate(counts) for _ in range(number))
        assignments = list(partitions(tuple(range(n)), counts))
        expectation = mean(arbitrary(i,j) for i in range(n) for j in range(n) if i != j)
        rho_expected = Q(sum(m*(m-1) for m in counts), n*(n-1))
        edge_results = []
        for a,b in itertools.combinations(range(len(counts)),2):
            actual = mean(score(bags[a], bags[b], arbitrary) for bags in assignments)
            actual_rho = mean(mean(Q(int(origin[i] == origin[j]))
                                  for i in bags[a] for j in bags[b]) for bags in assignments)
            assert actual == expectation and actual_rho == rho_expected
            edge_results.append({'slots':[a,b],'score_mean':actual, 'rho_mean':actual_rho})
        zero_overlap = []
        for bags in assignments:
            ok = all(not ({origin[r] for r in bags[a]} & {origin[r] for r in bags[b]})
                     for a,b in itertools.combinations(range(len(counts)),2))
            if ok:
                assert all(len({origin[r] for r in bag}) == 1 for bag in bags)
                assert all(tuple(r for r in range(n) if origin[r] == origin[bag[0]]) == bag for bag in bags)
                zero_overlap.append(bags)
        cases.append({'counts':counts,'allocation_count':len(assignments),
                      'score_formula':expectation,'rho_formula':rho_expected,'edges':edge_results,
                      'zero_overlap_allocations':zero_overlap,
                      'zero_overlap_only_whole_account_permutations':True})
    result['expectation_and_degeneracy'] = cases
    m = inp['mixture']; alpha = Q(m['alpha'])
    direct = score(tuple(m['Xa']+m['Xb']), tuple(m['Y']), arbitrary)
    mixture = alpha*score(m['Xa'],m['Y'],arbitrary)+(1-alpha)*score(m['Xb'],m['Y'],arbitrary)
    assert direct == mixture
    result['mixture_linearity'] = {'direct':direct, 'mixture':mixture, 'alpha':alpha}
    t = inp['teacher_nullspace']; pair = set(t['delta_unordered_pair'])
    delta = lambda i,j: Q(t['delta_value']) if {i,j} == pair else Q(0)
    d0 = score(*t['A0'], delta); d1 = score(*t['A1'], delta)
    assert d0 == 0 and d1 == Q(1,4)
    result['teacher_projection'] = {'delta_A0':d0, 'delta_A1':d1,
        'scope':'table-space example only; not two trained networks or equal full supervised objectives'}
    result['claims'] = ['2/3 mixing and 1/3 expected score', 'uniform allocation expectation',
                        'same-origin pair fraction formula', 'zero-overlap degeneration',
                        'empirical-mixture affinity', 'teacher A0 nullspace']
    return result

def encode(value):
    if isinstance(value, Q):
        return f'{value.numerator}/{value.denominator}'
    raise TypeError(type(value).__name__)

def main():
    if len(sys.argv) != 3:
        raise SystemExit('Usage: hand_examples.py INPUT_JSON OUTPUT_DIR')
    src = pathlib.Path(sys.argv[1]); dest = pathlib.Path(sys.argv[2]); dest.mkdir(parents=True, exist_ok=True)
    started = time.time()
    execution = {'python':sys.version,'platform':platform.platform(), 'argv':sys.argv,
                 'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                 'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
                 'environment_role':'ChatGPT working container, NOT project Linux native qualification',
                 'attempt':'01','failures':[]}
    try:
        result = run(json.loads(src.read_text(encoding='utf-8')))
        (dest/'hand_examples_output.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=encode)+'\n',encoding='utf-8')
        execution['status'] = 'PASS'; execution['exit_code'] = 0
        print('PASS: 6 exact finite algebra checks; no project code, data, labels, model or saved statistics loaded.')
        print('toy: 6 allocations, mixed=2/3, E[score]=1/3')
        for c in result['expectation_and_degeneracy']:
            print(f"counts={c['counts']}, allocations={c['allocation_count']}, E[rho]={c['rho_formula']}, zero-overlap={len(c['zero_overlap_allocations'])}")
    except BaseException:
        execution['status'] = 'FAIL'; execution['exit_code'] = 1
        execution['failures'].append(traceback.format_exc())
        raise
    finally:
        execution['elapsed_seconds'] = time.time()-started
        (dest/'hand_examples_execution.json').write_text(json.dumps(execution,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__ == '__main__':
    main()
