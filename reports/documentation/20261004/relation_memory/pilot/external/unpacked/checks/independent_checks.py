"""Limited local evidence audit: real stdlib/NumPy production paths only.

Never imports the torch-bound pilot entry point or test module, never calls
train/execute/public_inputs/attach_labels/load_model. No fake torch, installation,
GPU access, formal examples, or original package mutations.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

ROOT = Path(os.environ.get("RELATION_REVIEW_SOURCE_ROOT", str(Path(__file__).resolve().parents[1] / "source")))
OUT = Path(os.environ.get("RELATION_REVIEW_OUTPUT", str(Path(__file__).resolve().parents[1] / "replay_results" / "independent_checks")))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts'))
import step28_bge_continual as parent
import step28_bge_continual_run as previous
import step28_relation_memory as method

data = parent.data
rows = []


def record(name, **facts):
    item = {'name': name, **facts}
    rows.append(item)
    print(json.dumps(item, ensure_ascii=False, sort_keys=True), flush=True)


def rejects(call, expected):
    try:
        call()
    except Exception as exc:
        assert isinstance(exc, ValueError), (type(exc).__name__, str(exc))
        assert expected in str(exc), str(exc)
        return {'exception': type(exc).__name__, 'message': str(exc)}
    raise AssertionError('Expected rejection did not occur')


environment = {
    'python': sys.executable, 'python_version': platform.python_version(),
    'platform': platform.platform(), 'numpy': np.__version__,
    'torch_spec_available': importlib.util.find_spec('torch') is not None,
    'runtime_python_env': os.environ.get('CODEX_PRIMARY_RUNTIME_PYTHON'),
    'runtime_root_env': os.environ.get('CODEX_PRIMARY_RUNTIME_ROOT'),
    'thread_env': {k: os.environ.get(k) for k in
                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
    'claim': 'Web workspace Python; not project Linux py310 or submitted CPU rerun.',
}
record('environment', **environment)

# Hash all submitted integration-CPU source records without importing its runner.
evidence_root = ROOT / 'reports/documentation/20261004/relation_memory/pilot'
cpu = data.read_json(evidence_root / 'cpu/result.json')
checks = []
for rec in cpu['source_files']:
    actual = data.record(ROOT / rec['path'], ROOT)
    checks.append({'path': rec['path'], 'matches': actual == rec})
assert all(c['matches'] for c in checks)

# Independently rederive the documented local import closure from source ASTs.
paths = {ROOT / p for p in (
    'scripts/step28_relation_memory_run.py', 'scripts/step28_relation_memory.py',
    'scripts/step28_relation_memory_verify.py',
    'tests/test_step28_relation_memory_run.py', 'tests/test_step28_relation_memory.py')}
todo = list(paths)
while todo:
    path = todo.pop()
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        names = ([v.name for v in node.names] if isinstance(node, ast.Import)
                 else [node.module] if isinstance(node, ast.ImportFrom) else [])
        for name in names:
            if name and name.startswith('step28_'):
                child = ROOT / 'scripts' / (name + '.py')
                if child not in paths:
                    paths.add(child)
                    todo.append(child)
paths.update(ROOT / 'schema' / name for name in (
    'step28_relation_memory_policy.json', 'step28_bge_continual_policy.json',
    'step28_alias_ranking_policy.json', 'step28_chinese_base_policy.json'))
paths.update(ROOT / p for p in (
    'docs/SELLER_ALIAS_RELATION_MEMORY_PILOT.zh.md',
    'docs/SELLER_ALIAS_RELATION_MEMORY_EXECUTION.zh.md',
    'scripts/run_step28_relation_memory_pilot_linux_20261004.sh'))
closure = [data.record(p, ROOT) for p in sorted(paths)]
assert closure == cpu['source_files']
record('submitted_cpu_source_identity', count=len(checks), all_hashes_match=True,
       independent_import_closure_matches=True, files=checks)

progress = [json.loads(line) for line in
            (evidence_root / 'cpu.console.txt').read_text().splitlines() if line.strip()]
observed = [(r['order'], r['stage'], r['step']) for r in progress]
expected = [(order, stage, step) for order in parent.ORDERS
            for stage in (1, 2, 3) for step in range(24, 289, 24)]
assert observed == expected
assert all(b['elapsed_seconds'] >= a['elapsed_seconds']
           for a, b in zip(progress, progress[1:]))
record('submitted_cpu_console_consistency', progress_records=len(progress),
       stage_endpoint_records=sum(r['step'] == 288 for r in progress),
       exact_order_stage_every_24_schedule=True,
       final_progress_elapsed=progress[-1]['elapsed_seconds'],
       submitted_suite_tests=cpu['tests_run'], submitted_suite_status=cpu['status'],
       update_report_field=cpu['full_three_order_small_model_updates'],
       limit='Log/source consistency only; no independent torch rerun or per-update observation.')

# Use actual production parse_once, with an explicitly handwritten loader only.
# Failure is charged before any loader call; forbidden split never calls loader.
with tempfile.TemporaryDirectory(prefix='parse_', dir=OUT) as temp:
    job = Path(temp)
    calls = []
    data.write_json(job / 'access.json', dict(train=0, valid=0, heldout=0, owners=0))

    def failed_loader(groups, config, split):
        calls.append({'split': split, 'access_at_loader_entry': data.read_json(job / 'access.json')})
        raise ValueError('deliberate handwritten loader failure')

    rejection = rejects(lambda: previous.parse_once(job, [], {}, 'train', failed_loader),
                        'deliberate handwritten')
    assert calls[0]['access_at_loader_entry']['train'] == 1
    second = rejects(lambda: previous.parse_once(job, [], {}, 'train', failed_loader),
                     'already been attempted')
    assert len(calls) == 1
    forbidden = {}
    for split in ('heldout', 'owners'):
        forbidden[split] = rejects(lambda s=split: previous.parse_once(job, [], {}, s, failed_loader),
                                   'Forbidden supervision')
    assert len(calls) == 1
    record('production_parse_once_failure_and_reentry', first_failure=rejection,
           reentry=second, forbidden=forbidden, loader_calls=calls,
           final_access=data.read_json(job / 'access.json'))

with tempfile.TemporaryDirectory(prefix='early_parse_', dir=OUT) as temp:
    job = Path(temp)
    data.write_json(job / 'access.json', dict(train=0, valid=0, heldout=0, owners=0))
    calls = []

    def handwritten_loader(groups, config, split):
        calls.append(split)
        return ['handwritten-only']

    result = previous.parse_once(job, [], {}, 'development', handwritten_loader)
    assert result == ['handwritten-only'] and calls == ['development']
    record('parse_once_is_not_the_complete_blind_gate', accepted_without_blind_manifest=True,
           final_access=data.read_json(job / 'access.json'),
           interpretation='Expected helper scope: parse_once enforces split/attempt only. '
           'Current execute orders training and blind_gate before this call. '
           'This observation alone is not an actual early-valid-access defect.')

# Construct legal complete public-shaped handwritten groups without the torch fixture.
controls = [i for i, n in enumerate([3] * 4 + [2] * 8) for _ in range(n)]
labels = tuple(int(controls[i] == controls[j])
               for i, j in itertools.combinations(range(28), 2))


def group(uid):
    result = data.Group(uid, tuple(f'{uid}_s{i:02}' for i in range(28)),
        tuple(tuple((f'{uid}_i{i:02}_{j}', f'手写标题{uid}-{i}-{j}',
                    f'手写描述{uid}-{i}-{j}') for j in range(2)) for i in range(28)), labels)
    result.validate()
    return result


groups, metadata = {'train': [], 'development': []}, []
for split, count in (('train', 60), ('development', 20)):
    for domain in 'ABC':
        for index in range(count):
            uid = f'{split}_{domain}_{index:02}'
            groups[split].append(group(uid))
            metadata.append(dict(group_uid=uid, domain=domain, split=split, group_index=str(index)))
selected, partition = parent.base.partition(groups, metadata, method.config())
supply = parent.Supply(selected, partition)
future = rejects(lambda: supply.current('ABC', 'ABC', 2), 'Future, old or repeated')
stage_checks = []
for order in parent.ORDERS:
    for stage in (1, 2, 3):
        fit, calibration = supply.current(order, order, stage)
        assert len(fit) == 48 and len(calibration) == 12
        expected_fit = [r['group_uid'] for r in partition['fit'] if r['domain'] == order[stage - 1]]
        expected_cal = [r['group_uid'] for r in partition['calibration'] if r['domain'] == order[stage - 1]]
        assert [g.uid for g in fit] == expected_fit
        assert [g.uid for g in calibration] == expected_cal
        assert not set(expected_fit) & set(expected_cal)
        sequence, stream = parent.schedule(fit, parent.contract(), order, stage)
        assert len(sequence) == 288 and set(Counter(g.uid for g in sequence).values()) == {6}
        stage_checks.append({'order': order, 'stage': stage, 'domain': order[stage - 1],
                             'fit': 48, 'calibration': 12, 'presentations': len(sequence),
                             'fit_presentations_each': 6})
        rejects(lambda o=order, s=stage: supply.current(o, o, s), 'Future, old or repeated')
    rejects(lambda o=order: supply.current(o, o, 4), 'No fourth domain')
fresh = parent.Supply(selected, partition)
fresh.current('same_path', 'ABC', 1)
changed_order = rejects(lambda: fresh.current('same_path', 'BCA', 2), 'Future, old or repeated')
record('production_supply_and_schedule_on_handwritten_groups', stages=stage_checks,
       future_stage_rejected=future, path_order_change_rejected=changed_order,
       repeated_stages_and_fourth_stage_rejected=True,
       limit='No learner calls, gradient, memory, BGE or formal data. Supplies protect requested path/order, '
             'not an operating-system isolation boundary for the offline dispatcher.')

# Small exact production calibration transform and metric example (no fitting).
# Positive affine transforms preserve rank while classification can change.
x = np.asarray([np.linspace(-2.3, 1.7, 378, dtype=np.float32)], dtype=np.float32)
z = parent.calibration.transform(x, {'a': 1.5, 'b': -1.0})
parent.calibration.preserve_order(x, z)
raw_matrix, raw_counts = parent.metrics.group_metrics(np.array([labels], np.uint8), x)
cal_matrix, cal_counts = parent.metrics.group_metrics(np.array([labels], np.uint8), z)
import step28_bge_continual_evaluate as evaluation
assert np.array_equal(raw_matrix[:, evaluation.RANK_COLUMNS], cal_matrix[:, evaluation.RANK_COLUMNS])
record('production_metric_transform_small_example', matrix_shape=list(raw_matrix.shape),
       rank_columns_equal=True, raw_counts=raw_counts, calibrated_counts=cal_counts,
       limit='One handwritten group; no L-BFGS fit, no 28-set integration replay.')

assert 'torch' not in sys.modules
data.write_json(OUT / 'results.json', {
    'status': 'PASS_LIMITED_STDLIB_NUMPY_AUDIT', 'checks': rows,
    'not_run': ['submitted two torch integration tests', 'torch CPU training', 'formal execute or train',
                'BGE/tokenizer/native model load', 'GPU', 'formal data or supervision'],
    'original_package_modified': False,
})
print('PASS_LIMITED_STDLIB_NUMPY_AUDIT', flush=True)
