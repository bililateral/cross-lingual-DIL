"""Operation bindings, single-use access receipts and bounded worker checks."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import time

import psutil

from step30_evolution_policy import CAPS, LIMITS

PYTHON = '/home/yongpeng/miniconda3/envs/py310/bin/python'
SOURCES = tuple('step30_evolution_'+x+'.py' for x in ('policy', 'data', 'run', 'evaluate', 'io', 'control')) + (
    'step29_vendorlink_data.py', 'step29_vendorlink_methods.py', 'step29_vendorlink_run.py')


def write(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def record(path: Path) -> dict:
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def verify(row: dict) -> Path:
    path = Path(row['path'])
    if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise ValueError('Bound source identity mismatch: '+str(path))
    return path


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pinned(row: dict):
    return read(verify(row))


def qualification_directory(allowed: dict) -> Path:
    directory = Path(allowed['qualification_directory']).resolve()
    bindings = {Path(row['path']).resolve(): row for row in allowed['qualification_sources']}
    required = {directory/(name+'.py') for name in ('audit', 'followup')}
    if len(allowed['qualification_sources']) != 2 or set(bindings) != required:
        raise ValueError('Bind the exact audit/followup files that will be imported')
    if not directory.is_relative_to(Path(allowed['workspace']).resolve()):
        raise ValueError('Qualification imports must use the isolated workspace')
    for row in bindings.values():
        verify(row)
    return directory


def access(root: Path, role: str, sources: list[dict]) -> None:
    path = root/'access'/f'{role}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Persist ATTEMPT before the first source open; failure does not refund it.
    with path.open('x', encoding='utf-8') as output:
        json.dump(dict(role=role, attempted_at=time.time(), sources=sources), output)


def permit(path: Path, action: str) -> dict:
    value = read(path)
    if (sys.platform != 'linux' or Path(sys.executable).resolve() != Path(PYTHON).resolve()
            or value.get('experiment') != 'Evolution_pilot_v1' or value.get('approved') is not True
            or value.get('review_closed') is not True or action not in value.get('actions', [])
            or value.get('limits') != LIMITS or value.get('budget_seconds') != CAPS
            or value.get('automatic_retry') is not False or value.get('automatic_wait') is not False):
        raise ValueError('Explicit source-bound reviewed Evolution operation permit required')
    verify(value['authorization'])
    verify(value['proposal'])
    workspace = Path(value['workspace']).resolve()
    experiment = Path(value['experiment_root']).resolve()
    if not workspace.is_relative_to(experiment) or not Path(value['campaign']).resolve().is_relative_to(experiment):
        raise ValueError('All new source/data/output belong to the bound experiment directory')
    actual = Path(__file__).resolve().parent
    bindings = {Path(r['path']).resolve(): r for r in value['sources']}
    required = {actual/name for name in SOURCES}
    if not required <= set(bindings):
        raise ValueError('Actual entry and direct dependencies must be source-bound')
    for p, row in bindings.items():
        if not p.is_relative_to(workspace):
            raise ValueError('Use an isolated deployment; no source outside bound workspace')
        verify(row)
    return value


def dataset(allowed: dict) -> dict:
    if not Path(allowed['dataset_manifest']['path']).resolve().is_relative_to(Path(allowed['experiment_root']).resolve()):
        raise ValueError('New dataset must count against the same experiment disk cap')
    manifest = pinned(allowed['dataset_manifest'])
    if (manifest['protocol'] != 'evolution-pilot-v1' or manifest['counts'] != {'fit': 160, 'dev': 64, 'test': 96}
            or manifest['domains'] != ['2014-Q3', '2014-Q4', '2015-Q1']):
        raise ValueError('Dataset protocol or quotas differ')
    language = pinned(allowed['language_receipt'])
    if (not language['passed'] or language['manifest'] != allowed['dataset_manifest']
            or not manifest.get('language_samples')
            or language.get('language_samples') != manifest['language_samples']):
        raise ValueError('Same frozen dataset must pass the predetermined manual language gate')
    return manifest


def directory_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob('*'):
        try:
            item = path.stat()
        except FileNotFoundError:
            # Atomic writers may rename an enumerated .tmp before this stat.
            # Other I/O failures remain failures; the save reserve is unchanged.
            continue
        if stat.S_ISREG(item.st_mode):
            total += item.st_size
    return total


class Budget:
    def __init__(self, root: Path, deadline: float, rss: int = 32*2**30):
        self.root, self.deadline, self.rss = root, deadline, rss
        self.last = 0.

    def check(self, force: bool = False):
        now = time.monotonic()
        if now >= self.deadline-10:
            raise TimeoutError('Worker deadline/shutdown reserve')
        if psutil.Process().memory_info().rss > self.rss:
            raise RuntimeError('Worker RSS limit')
        if force or now-self.last >= 2:
            self.last = now
            # Reserve one compact model and metadata BEFORE an atomic save.
            if (directory_bytes(self.root)+2**30 > LIMITS['disk_bytes']
                    or shutil.disk_usage(self.root).free < LIMITS['free_disk_reserve']+2**30):
                raise RuntimeError('Experiment disk cap/shared free-space reserve')


class Ledger:
    def __init__(self, root: Path, prior: dict, cpu_tests: int):
        if set(prior) != set(CAPS) or any(not math.isfinite(v) or v < 0 or v >= CAPS[k] for k, v in prior.items()):
            raise ValueError('Complete nonnegative prior costs required')
        if not 0 <= cpu_tests <= 1200 or prior['auxiliary'] < cpu_tests:
            raise ValueError('CPU tests are a subset of the auxiliary cumulative budget')
        self.root, self.used, self.events = root, dict(prior), []
        self.cpu_tests = cpu_tests
        self.used['auxiliary'] += 30  # Explicit conservative final save/cleanup fee.
        self.save()

    def save(self):
        write(self.root/'budget.json', dict(used_seconds=self.used, caps=CAPS, cpu_test_seconds=self.cpu_tests,
                                            conservative_tail_seconds=30, events=self.events))

    def reserve(self, bucket, seconds):
        if seconds <= 0 or self.used[bucket]+seconds > CAPS[bucket]:
            raise TimeoutError('Cannot reserve full next operation within cumulative bucket')

    def charge(self, bucket, name, seconds, success):
        charged = math.ceil(seconds)
        self.used[bucket] += charged
        self.events.append(dict(operation=name, bucket=bucket, actual_seconds=seconds,
                                charged_seconds=charged, success=success))
        self.save()
        if self.used[bucket] > CAPS[bucket]:
            raise TimeoutError('Cumulative budget exceeded; no further dispatch')

    @contextmanager
    def active(self, name):
        start, ok = time.monotonic(), False
        try:
            yield
            ok = True
        finally:
            self.charge('auxiliary', name, time.monotonic()-start, ok)
