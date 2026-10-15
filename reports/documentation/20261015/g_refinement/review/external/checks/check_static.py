#!/usr/bin/env python3
"""Web-isolated, standard-library-only STATIC arithmetic audit.

No production source is imported or executed. Saved CSV summaries are checked
for arithmetic/record correspondence, NOT regenerated from labels or resampled.
Run from the returned archive root:
  python checks/check_static.py --input-zip input/review_input.zip \
      --toy-input checks/inputs.json --output checks/output.json
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def flatten(block: list[list[float]]) -> list[float]:
    require(bool(block) and bool(block[0]), 'Empty handwritten block')
    require(all(len(row) == len(block[0]) for row in block), 'Ragged block')
    result = [float(x) for row in block for x in row]
    require(all(math.isfinite(x) for x in result), 'Nonfinite handwritten input')
    return result


def lae(values: list[float], tau: float = 1.0) -> float:
    require(bool(values) and tau > 0, 'Invalid scalar audit arguments')
    z = [x / tau for x in values]
    peak = max(z)
    return tau * (peak + math.log(math.fsum(math.exp(x - peak) for x in z))
                  - math.log(len(values)))


def half(values: list[float]) -> float:
    return 0.5 * (math.fsum(values) / len(values) + lae(values))


def scalar_record(block: list[list[float]]) -> dict:
    values = flatten(block)
    n = len(values)
    peak = max(values)
    exp_values = [math.exp(x - peak) for x in values]
    denom = math.fsum(exp_values)
    p = [x / denom for x in exp_values]
    q = [0.5 / n + 0.5 * x for x in p]
    mu, g, h = math.fsum(values) / n, lae(values), half(values)
    tol = 1e-12
    require(mu <= h + tol and h <= g + tol, 'Jensen sandwich failed')
    require(abs(math.fsum(q) - 1) < tol, 'Derivative weights do not sum to 1')
    require(all(0.5/n <= x <= 0.5 + 0.5/n for x in q), 'Local derivative bound failed')
    deviation_error = max(abs((v - 1/n) - 0.5*(u - 1/n)) for u,v in zip(p,q))
    require(deviation_error < tol, 'Deviation-halving identity failed')
    width = len(block[0])
    return dict(shape=[len(block),width], N=n, mean=mu, G=g, G_half=h,
                softmax=p, derivative_half=q,
                row_derivative_sums=[math.fsum(q[i:i+width]) for i in range(0,n,width)],
                lower_bound=0.5/n, upper_bound=0.5+0.5/n,
                deviation_halving_max_error=deviation_error)


def run(zip_path: Path, toy_path: Path) -> dict:
    toy = json.loads(toy_path.read_text(encoding='utf-8'))
    original = zip_path.read_bytes()
    expected = toy['expected_zip']
    require(len(original) == expected['bytes'], 'ZIP byte count mismatch')
    require(sha256(original) == expected['sha256'], 'ZIP SHA256 mismatch')
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        infos = archive.infolist()
        names = archive.namelist()
        require(len(infos) == expected['members'], 'ZIP member count mismatch')
        require(len(set(names)) == len(names), 'Duplicate ZIP member')
        require(archive.testzip() is None, 'ZIP CRC failure')
        manifest = json.loads(archive.read('source_manifest.json'))
        payload_checks = []
        for rec in manifest['files']:
            content = archive.read(rec['entry'])
            actual_hash = sha256(content)
            require(len(content) == rec['bytes'], 'Payload size: ' + rec['entry'])
            require(actual_hash == rec['sha256'], 'Payload SHA: ' + rec['entry'])
            payload_checks.append(dict(entry=rec['entry'], bytes=len(content), sha256=actual_hash))
        require(set(names) == {x['entry'] for x in manifest['files']} | {'source_manifest.json'},
                'Manifest membership mismatch')
        csv_content = archive.read('result/all_metrics.csv').decode('utf-8-sig')
        rows = list(csv.DictReader(io.StringIO(csv_content)))
        # Each source record occupies one physical line in this supplied CSV.
        require(len(csv_content.splitlines()) == len(rows)+1, 'CSV physical line mapping invalid')
        keyed = {}
        for line, row in enumerate(rows,2):
            key = (row['kind'],row['point_or_contrast'],row['role'],row['domain'],row['metric'])
            require(key not in keyed, 'Duplicate metric key')
            keyed[key] = dict(source_line=line, **row)
        selected_map_rows = [dict(source_line=i, **r) for i,r in enumerate(rows,2)
                             if r['metric']=='map' and r['role']=='raw']
        def mean(kind, point, domain, metric='map', role='raw'):
            return float(keyed[(kind,point,role,domain,metric)]['mean'])
        residuals = []
        for contrast in ('F-S','G-S','F-G'):
            a,b = contrast.split('-')
            for kind,stage in (('final_contrast','B'),('initial_contrast','A')):
                for domain in ('A','B'):
                    calculated = mean('absolute',stage+'_'+a,domain)-mean('absolute',stage+'_'+b,domain)
                    saved = mean(kind,contrast,domain)
                    residuals.append(abs(calculated-saved))
            calculated = ((mean('absolute','A_'+a,'A')-mean('absolute','B_'+a,'A'))
                          -(mean('absolute','A_'+b,'A')-mean('absolute','B_'+b,'A')))
            residuals.append(abs(calculated-mean('forgetting_contrast',contrast,'A')))
        require(max(residuals) < 1e-14, 'MAP summary arithmetic discrepancy')
        correction_content = archive.read('result/loss_forgetting_direction.csv').decode('utf-8-sig')
        corrections = list(csv.DictReader(io.StringIO(correction_content)))
        require(len(corrections) == 18, 'Expected 18 prior loss direction corrections')
        correction_checks = []
        for line,c in enumerate(corrections,2):
            key = (c['kind'],c['contrast'],c['role'],c['domain'],c['metric'])
            old = keyed[key]
            triples = [float(c[x]) for x in ('saved_mean_AminusB','saved_low','saved_high')]
            require(triples == [float(old[x]) for x in ('mean','lower_95','upper_95')],
                    'Correction does not map to old summary')
            fixed = [float(c[x]) for x in ('loss_worsening_mean_BminusA',
                                          'loss_worsening_low','loss_worsening_high')]
            require(fixed == [-triples[0],-triples[2],-triples[1]], 'Sign/CI flip mismatch')
            require(fixed[1] < 0 < fixed[2], 'Expected crossing-zero correction')
            correction_checks.append(dict(correction_line=line, original_line=old['source_line'],
                                          contrast=c['contrast'],role=c['role'],metric=c['metric'],
                                          saved=triples,corrected=fixed))
    numerical = {name: scalar_record(block) for name,block in toy['blocks'].items()}
    derivative_checks = []
    eps = toy['finite_difference_step']
    for name in toy['gradient_check_names']:
        values = flatten(toy['blocks'][name])
        finite_difference = []
        for i in range(len(values)):
            plus,minus = values.copy(),values.copy()
            plus[i] += eps
            minus[i] -= eps
            finite_difference.append((half(plus)-half(minus))/(2*eps))
        analytic = numerical[name]['derivative_half']
        err = max(abs(x-y) for x,y in zip(finite_difference,analytic))
        require(err < toy['finite_difference_tolerance'], 'Scalar derivative failed: '+name)
        derivative_checks.append(dict(name=name, finite_difference=finite_difference,
                                      analytic=analytic,max_absolute_error=err))
    sparse, flat = numerical['sparse'],numerical['flat']
    require(2 < sparse['G'] < 3, 'Proposed 2 < G(sparse) < 3 failed')
    require(sparse['G'] > flat['G'] and sparse['G_half'] < flat['G_half'], 'Ranking reversal failed')
    require(numerical['two_sided']['G_half'] == numerical['one_row']['G_half'],
            'All-entry permutation counterexample failed')
    shifted = [x+toy['translation'] for x in flatten(toy['blocks']['sparse'])]
    shift_error = abs(half(shifted) - (sparse['G_half']+toy['translation']))
    require(shift_error < 1e-10, 'Translation identity failed')
    rectangular = toy['blocks']['rectangular']
    transposed = [list(row) for row in zip(*rectangular)]
    transpose_error = abs(half(flatten(rectangular))-half(flatten(transposed)))
    require(transpose_error < 1e-12, 'Transpose identity failed')
    c = sparse['G_half']
    student = [c,c,c,c]
    a,b = c-sparse['mean'], c-sparse['G']
    distillation = dict(teacher=toy['blocks']['sparse'],student=[[c,c],[c,c]],
                        mixed_then_squared=(half(student)-sparse['G_half'])**2,
                        average_component_squared=0.5*a*a+0.5*b*b,
                        mean_error=a,G_error=b)
    require(distillation['mixed_then_squared'] < 1e-25 and
            distillation['average_component_squared'] > 0.1,'D mixing distinction failed')
    diag = toy['diagnostic_report_values']
    gap_error = abs(diag['fit_change']-diag['valid_change']-diag['gap_change'])
    require(gap_error < 1e-12,'Reported diagnostic gap arithmetic failed')
    return {
        'status':'PASS_STATIC_ARITHMETIC_ONLY',
        'scope_limits':['No project source imported/executed','No model, optimizer, autograd, GPU or training',
                        'No formal text, labels, Memory or credentials','No resampling or per-group metric reconstruction',
                        'Saved CIs are read and direction-corrected, not independently re-estimated',
                        'Binary64 Python math is not a qualification of proposed torch FP64/FP32 implementation'],
        'input_zip':dict(bytes=len(original),members=len(infos),sha256=sha256(original),crc='PASS'),
        'manifest_payload_checks':payload_checks,'manifest_payload_count':len(payload_checks),
        'saved_summary':dict(total_rows=len(rows),selected_raw_map_rows=selected_map_rows,
                            map_arithmetic_max_residual=max(residuals),arithmetic_relations_checked=len(residuals),
                            direction_corrections=correction_checks,
                            reported_gap_arithmetic_residual=gap_error),
        'handwritten_blocks':numerical,'finite_difference_checks':derivative_checks,
        'translation_error':shift_error,'transpose_error':transpose_error,
        'temperature_2_analytic_comparator_only':dict(G_tau2_sparse=lae(flatten(toy['blocks']['sparse']),2.0),
                                                    G_half_sparse=sparse['G_half'],not_a_new_candidate=True),
        'distillation_square_placement_counterexample':distillation,
        'global_shape_bound':dict(N_min=4,N_max=64,min_local_weight=1/128,max_local_weight=5/8)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-zip',type=Path,required=True)
    parser.add_argument('--toy-input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = run(args.input_zip,args.toy_input)
    text = json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    args.output.write_text(text,encoding='utf-8')
    print(text,end='')

if __name__ == '__main__':
    main()
