#!/usr/bin/env python3
"""Check published report tables against already authorized saved results.

This is a display/interpretation consistency check, not an independent
label-dependent metric computation and not a rerun of native training.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sys

PROJECT = Path('/workspace/scratch/ca13b63db3f0/review_work/input/project')
OUT = Path(__file__).resolve().parent
READS = {}
CHECKS = []


def get_text(path: Path) -> str:
    raw = path.read_bytes()
    READS[str(path)] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return raw.decode('utf-8')


def get_json(path: Path):
    return json.loads(get_text(path))


def numeric(label, displayed, value, digits=6):
    observed = float(displayed)
    error = abs(observed - value)
    tolerance = 0.5 * 10**(-digits) + 1e-13
    ok = math.isfinite(error) and error <= tolerance
    CHECKS.append({'label': label, 'displayed': displayed, 'saved': value,
                   'absolute_rounding_error': error, 'tolerance': tolerance, 'ok': ok})
    assert ok, CHECKS[-1]


def tables(text):
    lines = text.splitlines()
    result = []
    current = []
    first = 0
    for lineno, line in enumerate(lines + [''], 1):
        if line.startswith('|'):
            if not current:
                first = lineno
            current.append([c.strip() for c in line.strip().strip('|').split('|')])
        elif current:
            result.append((first, current[0], current[2:]))
            current = []
    return result


DATA = {
    'low': {
        'job': 'reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job',
        'doc': 'docs/SELLER_ALIAS_ER_LOW_RESULT.zh.md',
        'methods': {'SEQ / λ=0':'seq', 'ER1':'er', 'ER0.5':'half', 'ER0.25':'quarter', 'ER0.1':'tenth'},
        'comparisons': {'ER0.1−ER0.25':'tenth_minus_quarter', 'ER0.1−SEQ':'tenth_minus_seq'},
    },
    'logit': {
        'job': 'reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/reports/job',
        'doc': 'docs/SELLER_ALIAS_LOGIT_WEIGHT_RESULT.zh.md',
        'methods': {'SEQ':'seq', 'ER0.25':'quarter', 'LOGIT0.25':'logit_quarter'},
        'comparisons': {'LOGIT0.25−ER0.25':'logit_quarter_minus_quarter', 'ER0.25−SEQ':'quarter_minus_seq', 'LOGIT0.25−SEQ':'logit_quarter_minus_seq'},
    },
}


def main():
    results = {}
    for study, cfg in DATA.items():
        doc_path = PROJECT / cfg['doc']
        doc = get_text(doc_path)
        ev = get_json(PROJECT / cfg['job'] / 'evaluation/evaluation.json')
        section = iter(('O', 'N', 'Z'))
        counts = {'core_values':0, 'endpoint_values':0, 'comparison_values':0,
                  'per_order_values':0, 'boolean_checks':0}
        for lineno, header, rows in tables(doc):
            if header[0] == '方法':
                columns = [('O','map'), ('O','recall_at_5'), ('N','map'),
                           ('N','recall_at_5'), ('Z','map'), ('Z','recall_at_5'),
                           ('F_first','map'), ('final_all','map')]
                assert len(header) == len(columns)+1
                for row in rows:
                    arm = cfg['methods'][row[0]]
                    for shown, (ep, metric) in zip(row[1:],columns):
                        numeric(f'{study}/line{lineno}/{arm}/{ep}/{metric}', shown, ev['endpoints'][arm]['primary'][ep][metric]['mean'])
                        counts['core_values']+=1
            elif header[0] == '指标':
                ep = next(section)
                arms = [cfg['methods'][h] for h in header[1:]]
                for row in rows:
                    metric = row[0]
                    for arm, shown in zip(arms, row[1:]):
                        numeric(f'{study}/line{lineno}/{arm}/{ep}/{metric}', shown, ev['endpoints'][arm]['primary'][ep][metric]['mean'])
                        counts['endpoint_values']+=1
            elif header[:2] == ['比较','端点']:
                for row in rows:
                    comparison = cfg['comparisons'][row[0]]
                    ep, metric = [x.strip() for x in row[1].split('/')]
                    obj = ev['comparisons'][comparison]['primary'][ep][metric]
                    numeric(f'{study}/line{lineno}/{comparison}/{ep}/{metric}/mean',row[2],obj['mean'])
                    interval = row[3].strip('[]').split(',')
                    assert len(interval)==2
                    for i, shown in enumerate(interval):
                        numeric(f'{study}/line{lineno}/{comparison}/{ep}/{metric}/ci{i}',shown,obj['conditional_95pct_interval'][i])
                    counts['comparison_values']+=3
            elif header[0] == 'O MAP差值':
                assert header[1:]==['ABC','BCA','CAB']
                for row in rows:
                    comparison = cfg['comparisons'][row[0]]
                    obj = ev['comparisons'][comparison]['primary']['O']['map']['per_order']
                    for order, shown in zip(header[1:],row[1:]):
                        numeric(f'{study}/line{lineno}/{comparison}/O/map/{order}',shown,obj[order])
                        counts['per_order_values']+=1
            elif header[0] == '原判据':
                pairs = [cfg['comparisons'][h] for h in header[1:]]
                for row in rows:
                    for pair, shown in zip(pairs,row[1:]):
                        checks = ev['comparisons'][pair]['interpretation']['checks']
                        expected = f'{sum(checks.values())}/{len(checks)}' if row[0]=='合计' else ('通过' if checks[row[0]] else '未通过')
                        assert shown == expected, (study,row[0],pair,shown,expected)
                        CHECKS.append({'label':f'{study}/line{lineno}/{pair}/{row[0]}','displayed':shown,'saved':expected,'ok':True})
                        counts['boolean_checks']+=1
        assert counts['core_values'] == 8*len(cfg['methods'])
        assert counts['endpoint_values'] == 3*22*len(cfg['methods'])
        assert counts['comparison_values'] == 8*3*len(cfg['comparisons'])
        assert counts['per_order_values'] == 3*len(cfg['comparisons'])
        assert counts['boolean_checks'] == 24*len(cfg['comparisons'])
        original = get_json(PROJECT / f'reports/seller_alias_continual/20261003/{"er_low" if study=="low" else "logit_weight"}_result/audit/audit.json')
        replayed = get_json(OUT / study / 'audit.json')
        excluded = {'completed_at_utc','elapsed_seconds','cpu_affinity'}
        original_stable = {k:v for k,v in original.items() if k not in excluded}
        replayed_stable = {k:v for k,v in replayed.items() if k not in excluded}
        assert original_stable == replayed_stable
        original_diag = get_json(PROJECT / f'reports/seller_alias_continual/20261003/{"er_low" if study=="low" else "logit_weight"}_result/audit/diagnostics.json')
        replayed_diag = get_json(OUT / study / 'diagnostics.json')
        assert original_diag == replayed_diag
        results[study] = {'report_table_counts':counts,'audit_replay_stable_fields_identical':True,
                          'diagnostics_identical':True,'dynamic_fields_excluded':sorted(excluded),
                          'passed_counts':{k:sum(v['interpretation']['checks'].values()) for k,v in ev['comparisons'].items()}}
    payload = {'status':'PASS_REPORT_TABLE_AND_REPLAY_CONSISTENCY','python':sys.version,'results':results,
               'checked_entries':len(CHECKS),'checks':CHECKS,'inputs':READS,
               'limitations':['No labels, models, calibration fitting, or native execution.',
                              'Compared published table values against authorized saved result JSON; independent endpoint recomputation is outside this script.',
                              'Audit replay stable-field identity does not independently establish scientific correctness.']}
    (OUT/'report_consistency.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('checks','inputs')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
