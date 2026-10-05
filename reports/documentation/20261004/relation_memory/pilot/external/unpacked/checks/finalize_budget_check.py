"""Exact extracted production finalizer on fully handwritten saved-matrix files.

No pilot entry-point import (torch absent), no CLI/execute/train/model/real
reference/data access. Two globals are fixture providers: policy() redirects
references to handwritten files; sources() returns the verified package source
records. Every extracted function body is unchanged from the reviewed source.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(os.environ.get("RELATION_REVIEW_SOURCE_ROOT", str(Path(__file__).resolve().parents[1] / "source")))
OUT = Path(os.environ.get("RELATION_REVIEW_OUTPUT", str(Path(__file__).resolve().parents[1] / "replay_results" / "finalize_budget_check")))
OUT.mkdir(parents=True, exist_ok=True)
FIXTURE = OUT / 'finalize_budget_fixture'
FIXTURE.mkdir(exist_ok=False)
sys.path.insert(0, str(ROOT / 'scripts'))
import step28_bge_continual as parent
import step28_bge_continual_evaluate as evaluation
import step28_bge_continual_run as previous

data, base = parent.data, parent.base
source_path = ROOT / 'scripts/step28_relation_memory_run.py'
source_text = source_path.read_text(encoding='utf-8')
names = ('point_name', 'expected_points', 'read_roles', 'comparisons', 'finalize')
functions = [n for n in ast.parse(source_text).body if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in functions} == set(names)
extracted = [{'name': n.name, 'first_line': n.lineno, 'last_line': n.end_lineno,
              'source_sha256': hashlib.sha256(ast.get_source_segment(source_text, n).encode()).hexdigest()}
             for n in functions]
data.write_json(OUT / 'extracted_functions.json', {'file': data.record(source_path, ROOT),
    'functions': extracted, 'unchanged_bodies': True,
    'substituted_globals': ['policy: handwritten reference path and identities',
                           'sources: verified reviewed-package source records']})

source_records = data.read_json(ROOT / 'reports/documentation/20261004/relation_memory/pilot/cpu/result.json')['source_files']
p = data.read_json(ROOT / 'schema/step28_relation_memory_policy.json')
job = FIXTURE / 'job'
saved = job / 'evaluation'
baseline = FIXTURE / 'handwritten_reference'
saved.mkdir(parents=True)
(baseline / 'reference').mkdir(parents=True)

# Real 22 metrics and confusion counts from one invented complete relation group.
# No CSV, text dataset, model, or formal labels are loaded.
controls = [i for i, n in enumerate([3] * 4 + [2] * 8) for _ in range(n)]
truth = np.array([[int(controls[i] == controls[j])
                   for i, j in itertools.combinations(range(28), 2)]], dtype=np.uint8)
zero_scores = np.zeros((1, 378), dtype=np.float32)
perfect_scores = (2 * truth.astype(np.float32) - 1)
base_metric, base_counts = parent.metrics.group_metrics(truth, zero_scores)
good_metric, good_counts = parent.metrics.group_metrics(truth, perfect_scores)
domains = [d for d in 'ABC' for _ in range(20)]
group_ids = [f'handwritten_valid_{d}_{i:02}' for d in 'ABC' for i in range(20)]


def save_role(folder, prefix, improved_domains=()):
    improved = [d in improved_domains for d in domains]
    matrix = np.stack([good_metric[0] if good else base_metric[0] for good in improved])
    counts = [copy.deepcopy(good_counts[0] if good else base_counts[0]) for good in improved]
    mr = previous.save_array(folder / (prefix + '.npy'), matrix, folder)
    cp = folder / (prefix + '_counts.json')
    data.write_json(cp, counts)
    return {'matrix': mr, 'counts': data.record(cp, folder)}


common = {'group_ids': group_ids, 'domains': domains, 'metric_columns': list(parent.metrics.COLUMNS)}
collected = {**copy.deepcopy(common), 'source_files': source_records, 'points': {}}
collected['points']['initial'] = {'raw': save_role(saved, 'initial_raw')}
refcols = [{**copy.deepcopy(common), 'points': {}} for _ in range(2)]
for order in parent.ORDERS:
    for stage in (1, 2, 3):
        name = f'{order}_relation_stage{stage}'
        # Candidate improves only old domains at stage3. New/current domains equal reference.
        improved = order[:2] if stage == 3 else ()
        collected['points'][name] = {role: save_role(saved, name + '_' + role, improved)
                                    for role in parent.ROLES}
        index = int(stage == 1)
        folder = baseline / 'reference' if index else baseline
        refname = order + '_shared' if stage == 1 else f'{order}_logit_tenth_stage{stage}'
        refcols[index]['points'][refname] = {role: save_role(folder, refname + '_' + role)
                                           for role in parent.ROLES}
assert sum(map(len, collected['points'].values())) == 28
collected['status'] = 'ALL_28_RELATION_METRIC_COUNT_SETS_SAVED'
data.write_json(saved / 'collected.json', collected)
refs = []
for folder, refcol in zip((baseline, baseline / 'reference'), refcols):
    data.write_json(folder / 'collected.json', refcol)
    refs.append(data.record(folder / 'collected.json', baseline))
p['reference'] = {'linux_root': str(baseline), 'collections': refs,
                  'fixture_only': 'No real LOGIT0.1 reference was accessed.'}

# These are synthetic existing failure/consumption records, not statements that
# an actual approved run happened. They isolate inherited recovery eligibility.
failure = {'status': 'BUDGET_STOP_NO_RETRY',
           'error': 'Approved GPU-stage time budget reached',
           'fixture_only': True, 'observed_elapsed_seconds': 86401}
data.write_json(job / 'failure.json', failure)
data.write_json(job / 'budget.json', {'fixture_only': True,
    'maximum_gpu_stage_seconds': 86400, 'elapsed_seconds': 86401, 'remaining_seconds': 0})
data.write_json(job / 'access.json', {'train': 1, 'valid': 1, 'heldout': 0, 'owners': 0,
                                   'fixture_only': 'No label loader was called.'})
namespace = {'__name__': 'extracted_relation_finalizer_audit', 'Path': Path,
    'np': np, 'data': data, 'base': base, 'parent': parent, 'evaluation': evaluation,
    'ROLES': parent.ROLES, 'policy': lambda: copy.deepcopy(p),
    'sources': lambda: copy.deepcopy(source_records)}
module = ast.Module(body=functions, type_ignores=[])
exec(compile(module, str(source_path), 'exec'), namespace)
tick = time.monotonic()
result = namespace['finalize'](saved)
elapsed = time.monotonic() - tick
assert result['status'] == 'COMPLETE_RELATION_FIXED_POINT'
assert result['worth_matched_replay'] is True
assert all(result['continuation_checks'].values())
assert data.read_json(job / 'failure.json') == failure
assert (saved / 'evaluation.json').is_file()
assert not (job / 'completion.json').exists()
assert 'torch' not in sys.modules
report = {
    'status': 'REPRODUCED_FINALIZE_IGNORES_EXISTING_BUDGET_FAILURE',
    'environment': {'python': sys.version, 'numpy': np.__version__,
                    'torch_spec_available': importlib.util.find_spec('torch') is not None},
    'executed': 'Five exact production function bodies with real dependency functions; fixture policy/sources providers.',
    'formal_entry_called': False, 'formal_data_or_reference_read': False,
    'label_parser_calls': 0, 'torch_model_calls': 0,
    'saved_metric_count_sets': 28, 'preexisting_failure': failure,
    'preexisting_budget': data.read_json(job / 'budget.json'),
    'finalizer_status': result['status'], 'worth_matched_replay': result['worth_matched_replay'],
    'continuation_checks': result['continuation_checks'],
    'delta_O_map': result['delta']['O']['map']['mean'],
    'delta_N_map': result['delta']['N']['map']['mean'],
    'delta_Z_map': result['delta']['Z']['map']['mean'],
    'failure_unchanged': True, 'completion_json_exists': False,
    'evaluation_json': data.record(saved / 'evaluation.json', OUT),
    'elapsed_seconds_this_synthetic_check': elapsed,
    'limitation': 'No actual budget overrun or approved run occurred. This isolates missing eligibility checks '
                  'in finalize; it does not claim that comparisons math itself should implement runtime budgets.',
}
data.write_json(OUT / 'finalize_budget_result.json', report)
print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
