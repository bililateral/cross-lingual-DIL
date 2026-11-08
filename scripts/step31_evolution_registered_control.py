"""Two-slot, six-trajectory Registered execution. Every downstream phase is finite."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import shutil
import sys
import time

import psutil

from step30_evolution_io import (PYTHON, Budget, Ledger, access, pinned, read, record, verify, write)
from step30_evolution_control import Runner, configure_gpu
from step31_evolution_registered_policy import PROTOCOL, DOMAINS, CAPS, LIMITS, tasks, identity

SOURCES = ('step31_evolution_registered_policy.py', 'step31_evolution_registered_data.py',
           'step31_evolution_registered_evaluate.py', 'step31_evolution_registered_control.py',
           'step30_evolution_control.py', 'step30_evolution_run.py', 'step30_evolution_io.py',
           'step30_evolution_policy.py', 'step30_evolution_evaluate.py', 'step30_evolution_data.py',
           'step29_vendorlink_data.py', 'step29_vendorlink_methods.py', 'step29_vendorlink_run.py',
           'step28_formal_queue.py')


def permit(path, action):
    gate = read(path)
    if (sys.platform != 'linux' or Path(sys.executable).resolve() != Path(PYTHON).resolve()
            or gate.get('protocol') != PROTOCOL or gate.get('approved') is not True
            or gate.get('review_closed') is not True or action not in gate.get('actions', [])
            or gate.get('caps') != CAPS or gate.get('limits') != LIMITS
            or gate.get('automatic_retry') is not False
            or gate.get('worker_failure_policy') != 'retire_failed_slot_continue_healthy'
            or gate.get('automatic_next_phase') is not False):
        raise ValueError('Source-bound reviewed Registered permit required')
    for k in ('authorization', 'scope', 'review_disposition'):
        verify(gate[k])
    workspace, root = Path(gate['workspace']).resolve(), Path(gate['experiment_root']).resolve()
    if not workspace.is_relative_to(root) or Path(__file__).resolve().parent != workspace/'scripts':
        raise ValueError('Use this new isolated workspace')
    expected = {workspace/'scripts'/name for name in SOURCES}
    if {Path(r['path']).resolve() for r in gate['sources']} != expected:
        raise ValueError('Actual entry and dependencies must all be bound')
    for row in gate['sources']:
        verify(row)
    for name in ('dataset', 'campaign'):
        if not Path(gate[name]).resolve().is_relative_to(root):
            raise ValueError('All new data and outputs must stay in this experiment')
    return gate


def dataset(gate):
    root = Path(gate['dataset'])
    language = read(root/'language_receipt.json')
    manifest = pinned(language['manifest'])
    if (not language['passed'] or language['manifest']['path'] != str(root/'manifest.json')
            or language['language_samples'] != manifest['language_samples']
            or manifest['protocol'] != PROTOCOL or manifest['domains'] != list(DOMAINS)
            or manifest['accounts_per_batch'] != 96 or manifest['confirmation_exported'] is not False):
        raise ValueError('Frozen development data and original language gate must pass')
    return manifest


def gpu_admission():
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for formal training/scoring')
    configure_gpu()


def worker(gate, args):
    from step29_vendorlink_data import Group
    from step30_evolution_run import execute, make_model
    from step30_evolution_evaluate import training_complete
    from step31_evolution_registered_evaluate import score_one, analyze
    root = Path(gate['experiment_root'])
    check = Budget(root, args.deadline).check
    check(True)
    manifest = dataset(gate)
    batch = read(Path(gate['campaign'])/'batch.json')
    if batch['specs'] != tasks() or batch['manifest'] != record(Path(gate['dataset'])/'manifest.json'):
        raise ValueError('Frozen batch/manifest identity mismatch')
    binding = gate['model_binding']
    if args.action in ('train', 'score'):
        for row in binding['files']:
            verify(row)
        gpu_admission()
    def factory(task):
        return make_model(Path(binding['directory']), task, domains=DOMAINS)
    if args.action in ('train', 'score'):
        task = next((t for t in tasks() if t['id'] == args.task), None)
        if task is None:
            raise ValueError('Unknown fixed task')
    if args.action == 'train':
        expected_target = Path(gate['campaign'])/'training'/task['id']
        if args.target != expected_target:
            raise ValueError('Exact fresh training destination required')
        model = factory(task)
        run_identity = dict(protocol=PROTOCOL, task=task, manifest=batch['manifest'], gate=batch['gate'])
        requested = []
        def load_fit(domain):
            if domain != DOMAINS[len(requested)]:
                raise ValueError('Only one current fit batch at a time, in fixed order')
            requested.append(domain)
            source = manifest['files'][domain+':fit']
            access(args.target, 'current_fit_'+domain, [source])
            return [Group.from_dict(g) for g in pinned(source)]
        complete = execute(model, load_fit, args.target/'run', task, run_identity, binding,
                           write, record, check, expected_groups=12, domains=DOMAINS)
        training_complete(task, complete, run_identity, fit_groups=12)
    elif args.action == 'score':
        if args.target != Path(gate['campaign'])/'scoring'/task['id']:
            raise ValueError('Exact fresh scoring destination required')
        score_one(args.target/'scores', task, batch, manifest, factory, binding, check)
    elif args.action == 'analyze':
        if args.target != Path(gate['campaign'])/'analysis':
            raise ValueError('Exact analysis destination required')
        receipts = read(Path(gate['campaign'])/'scoring_complete.json')
        analyze(args.target, batch, manifest, receipts, check)
    else:
        raise ValueError('Unknown worker action')
    check(True)


def campaign(gate, gate_path):
    from step30_evolution_evaluate import training_complete
    from step31_evolution_registered_evaluate import validate_scores
    root = Path(gate['campaign'])
    if root.exists():
        raise ValueError('No repeated campaign, queue or automatic recovery')
    root.mkdir(parents=True)
    preparation = read(Path(gate['experiment_root'])/'preparation_budget.json')
    if preparation['used_seconds']['training'] or preparation['used_seconds']['gpu']:
        raise ValueError('No prior formal training or scoring permitted')
    if preparation['cpu_test_seconds'] > LIMITS['cpu_test_seconds']:
        raise ValueError('The new CPU increment budget is exhausted')
    ledger = Ledger(root, preparation['used_seconds'], preparation['cpu_test_seconds'], caps=CAPS)
    runner = Runner(gate, gate_path, root, ledger, worker_entry=Path(__file__), caps=CAPS)
    try:
        with ledger.active('prepare_registered_campaign'):
            if gate.get('waiter'):
                verify(gate['waiter'])
            if shutil.disk_usage(root).free < LIMITS['free_disk_start']:
                raise ValueError('Insufficient starting shared disk reserve')
            manifest = dataset(gate)
            batch = dict(specs=tasks(), manifest=record(Path(gate['dataset'])/'manifest.json'),
                         gate=record(gate_path), completed=[])
            write(root/'batch.json', batch)
            jobs = []
            for task in tasks():
                target = root/'training'/task['id']; target.mkdir(parents=True)
                jobs.append(dict(action='train', target=target, bucket='training', seconds=1800,
                                 extra=['--task', task['id']]))
        def complete_training(job):
            task = next(t for t in tasks() if t['id'] == job['target'].name)
            bound = record(job['target']/'run/completion.json')
            complete = pinned(bound)
            expected_identity = dict(protocol=PROTOCOL, task=task, manifest=batch['manifest'], gate=batch['gate'])
            training_complete(task, complete, expected_identity, fit_groups=12)
            batch['completed'].append(dict(task_id=task['id'], completion=bound, identity=expected_identity))
            write(root/'batch.json', batch)
        runner.run_many(jobs, 2, complete_training)
        scored = []
        for task in tasks():
            target = root/'scoring'/task['id']; target.mkdir(parents=True)
            def complete_score(job):
                bound = record(job['target']/'scores/blind_complete.json')
                validate_scores(pinned(bound))
                scored.append(dict(task_id=job['target'].name, completion=bound))
                write(root/'scoring_complete.json', scored)
            runner.run_many([dict(action='score', target=target, bucket='gpu', seconds=600,
                                 extra=['--task', task['id']])], 1, complete_score)
        target = root/'analysis'; target.mkdir()
        runner.run('analyze', target, 'auxiliary', 300, [])
        result = read(target/'result.json')
        write(root/'completion.json', dict(status=result['decision']['status'],
              premise_passed=result['decision']['premise_passed'], result=record(target/'result.json'),
              automatic_next_phase=False, training_trajectories=6, scoring_trajectories=6))
    except BaseException as exc:
        write(root/'failure.json', dict(error=repr(exc), automatic_retry=False,
                                       automatic_next_phase=False))
        raise
    finally:
        ledger.save()


@contextmanager
def preparation_charge(gate, operation):
    root = Path(gate['experiment_root'])
    path = root/'preparation_budget.json'
    value = read(path)
    start, ok = time.monotonic(), False
    try:
        yield
        ok = True
    finally:
        elapsed = time.monotonic()-start
        value['used_seconds']['auxiliary'] += math.ceil(elapsed)
        value['events'].append(dict(operation=operation, elapsed_seconds=elapsed, success=ok))
        write(path, value)
        if value['used_seconds']['auxiliary'] >= CAPS['auxiliary']:
            raise TimeoutError('Auxiliary budget exhausted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--action', choices=('build', 'language', 'campaign', 'train', 'score', 'analyze'), required=True)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--task')
    parser.add_argument('--decisions', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    gate = permit(args.gate, args.action)
    if args.action == 'campaign':
        campaign(gate, args.gate)
    elif args.action in ('build', 'language'):
        from step31_evolution_registered_data import build, language_adjudicate
        process = psutil.Process()
        process.cpu_affinity(process.cpu_affinity()[:2])
        with preparation_charge(gate, args.action):
            remaining = CAPS['auxiliary']-read(Path(gate['experiment_root'])/'preparation_budget.json')['used_seconds']['auxiliary']
            check = Budget(Path(gate['experiment_root']), time.monotonic()+min(180, remaining-60), rss=8*2**30).check
            check(True)
            if args.action == 'build':
                build(Path(gate['dataset']), gate['qualified_cache'], check)
            else:
                language_adjudicate(Path(gate['dataset']), read(args.decisions))
            check(True)
    else:
        if args.deadline is None or args.target is None:
            raise ValueError('Controller-bound worker lifecycle required')
        try:
            worker(gate, args)
        except BaseException as exc:
            write(args.target/'failure.json', dict(error=repr(exc), automatic_retry=False))
            raise


if __name__ == '__main__':
    main()
