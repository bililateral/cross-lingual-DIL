"""Web-workspace stdlib-only probes of extracted resource-control code.

No Torch/BGE load, no formal inputs, no training, no process termination.
Only the two Budget class ASTs and the local watchdog function are executed
with a synthetic torch/psutil surface and temporary tiny files.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import os
import shutil
import sys
import tempfile
import time
from unittest import mock

ROOT = Path('/workspace/scratch/f4d639b3b473/relation_memory_pilot_review')
OUT = Path(__file__).parent
base_tree = ast.parse((ROOT / 'scripts/step28_continual_population_run.py').read_text())
pilot_tree = ast.parse((ROOT / 'scripts/step28_relation_memory_run.py').read_text())

def only_def(tree, name):
    return next(node for node in tree.body if getattr(node, 'name', None) == name)

base_ns = dict(time=time, shutil=shutil, Path=Path)
exec(compile(ast.Module(body=[only_def(base_tree, 'Budget')], type_ignores=[]),
             '<production base Budget AST>', 'exec'), base_ns)
synthetic_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
assert 'torch' not in sys.modules
sys.modules['torch'] = synthetic_torch
pilot_ns = dict(base=SimpleNamespace(persistence=SimpleNamespace(Budget=base_ns['Budget'])),
                time=time, os=os, Path=Path,
                psutil=SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0))))
exec(compile(ast.Module(body=[only_def(pilot_tree, 'Budget')], type_ignores=[]),
             '<production pilot Budget AST>', 'exec'), pilot_ns)

results = {'scope': 'Web-workspace stdlib-only; extracted budget/control AST; no project Linux/GPU/Torch/BGE/training',
           'path_cases': []}
os.environ['RELATION_STARTED_EPOCH'] = str(time.time())
with tempfile.TemporaryDirectory(prefix='relation_resource_edges_') as temp:
    for index, name in enumerate(('job', 'job.v1')):
        root = Path(temp) / str(index) / name
        root.mkdir(parents=True)
        actual = [Path(str(root) + suffix) for suffix in ('.console.txt', '.wrapper.txt')]
        for path in actual:
            path.write_bytes(b'x' * 10)
        budget = pilot_ns['Budget'](root, {'runtime': {'maximum_gpu_stage_seconds': 60,
                                                     'maximum_output_bytes': 16}})
        try:
            budget.check()
            outcome = 'PASS'
        except RuntimeError as exc:
            outcome = str(exc)
        results['path_cases'].append({'job_name': name, 'actual_launcher_bytes': 20,
             'configured_limit_bytes': 16, 'actual_files': [p.name for p in actual],
             'checked_files': [root.with_suffix(s).name for s in ('.console.txt', '.wrapper.txt')],
             'observed_check': outcome})

    race_root = Path(temp) / 'race'
    (race_root / 'work').mkdir(parents=True)
    disposable = race_root / 'work' / 'already_verified_full.pt'
    disposable.write_bytes(b'x' * 10)
    race_budget = pilot_ns['Budget'](race_root, {'runtime': {'maximum_gpu_stage_seconds': 60,
                                                           'maximum_output_bytes': 1024}})
    original_stat = Path.stat
    seen = [0]
    def stat_with_concurrent_cleanup(path, *args, **kwargs):
        if path == disposable:
            seen[0] += 1
            if seen[0] == 2:
                # Controlled scheduling: main's already-verified cleanup occurs
                # between the monitor's is_file() and its separate stat().
                disposable.unlink()
        return original_stat(path, *args, **kwargs)
    try:
        with mock.patch.object(Path, 'stat', stat_with_concurrent_cleanup):
            race_budget.check()
        race_outcome = 'PASS'
    except FileNotFoundError as exc:
        race_outcome = type(exc).__name__
    results['concurrent_verified_temp_cleanup'] = {'limit_bytes': 1024,
        'bytes_before_cleanup': 10, 'bytes_after_cleanup': 0,
        'observed_check': race_outcome,
        'note': 'Deterministic interleaving simulation; not an observed project failure.'}

execute = only_def(pilot_tree, 'execute')
watchdog = next(node for node in execute.body if isinstance(node, ast.FunctionDef) and node.name == 'watchdog')
exit_calls = []
def fail_check():
    raise RuntimeError('synthetic budget breach')
def fail_write(*args, **kwargs):
    raise OSError('synthetic failure-ledger write error')
wd_ns = dict(stopped=SimpleNamespace(wait=lambda seconds: False),
             budget=SimpleNamespace(check=fail_check),
             data=SimpleNamespace(write_json=fail_write),
             job=Path('/synthetic-not-written'),
             os=SimpleNamespace(_exit=lambda code: exit_calls.append(code)))
exec(compile(ast.Module(body=[watchdog], type_ignores=[]),
             '<production watchdog AST>', 'exec'), wd_ns)
try:
    wd_ns['watchdog']()
except OSError as exc:
    results['watchdog_write_failure'] = {'uncaught': str(exc), 'process_exit_called': bool(exit_calls)}

assert results['path_cases'][0]['observed_check'] != 'PASS'
assert results['path_cases'][1]['observed_check'] == 'PASS'
assert results['watchdog_write_failure']['process_exit_called'] is False
assert results['concurrent_verified_temp_cleanup']['observed_check'] == 'FileNotFoundError'
assert sys.modules['torch'] is synthetic_torch
del sys.modules['torch']
(OUT / 'resource_edges.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(results, ensure_ascii=False, indent=2))
