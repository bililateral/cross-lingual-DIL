"""Four approved frozen-encoder head trials; reuse the qualified two-slot Runner."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

from step30_evolution_control import Runner
from step30_evolution_io import Ledger
from step35_vendorlink_account_io import Check, SharedDiskLimit, PYTHON, pinned, record, write
from step35_vendorlink_account_policy import LIMITS
from step35_vendorlink_account_control import SOURCES as ORIGINAL_SOURCES, configure_compute

PROTOCOL = 'vendorlink-head-alignment-20261110'
CAPS = dict(training=14400, gpu=1800, auxiliary=1800)
SOURCES = ORIGINAL_SOURCES + ('step35_vendorlink_head_adapt.py', 'step35_vendorlink_head_control.py')


def read(path):
    return json.loads(Path(path).read_text())


def tasks(g):
    out = []
    for candidate in ('FULL_POOL', 'LOCAL'):
        for domain in ('AlphaBay', 'Dream'):
            base = read(pinned(g['bases'][domain]))['task']
            settings = dict(base['settings'], diagnostics=False)
            out.append(dict(id='D_'+domain+'_'+candidate, phase='D', domains=[domain], stages=1,
                method='H_SEQ', candidate=candidate, seed=41, epochs=10, settings=settings))
    return out


def permit(path):
    g = read(path)
    if (sys.platform != 'linux' or Path(sys.executable).resolve() != Path(PYTHON).resolve()
            or g['protocol'] != PROTOCOL or g['approved'] is not True or g['review_waived'] is not True
            or g['primary_closed'] is not True or g['caps'] != CAPS or g['limits'] != LIMITS
            or g['automatic_retry'] is not False
            or g['worker_failure_policy'] != 'retire_oom_continue_current_phase_no_retry'):
        raise ValueError('Approved source-bound four-head scope required')
    root, workspace = Path(g['experiment_root']).resolve(), Path(g['workspace']).resolve()
    if (not workspace.is_relative_to(root) or not Path(g['campaign']).resolve().is_relative_to(root)
            or Path(__file__).resolve() != workspace/'scripts'/Path(__file__).name):
        raise ValueError('Use the new isolated workspace')
    if {Path(b['path']).resolve() for b in g['sources']} != {workspace/'scripts'/n for n in SOURCES}:
        raise ValueError('Bind actual entry and all direct dependencies')
    for b in g['sources']+[g['scope'], g['manifest'], g['cpu'], g['gpu'], g['baseline'], g['rms_results'], g['primary']]+list(g['bases'].values()):
        pinned(b)
    for key in ('cpu', 'gpu'):
        result = read(g[key]['path'])
        if result['status'] != 'PASS' or result.get('failures', 0) or result.get('errors', 0):
            raise ValueError('Necessary incremental verification incomplete')
    return g


def jobs(g, specs, action):
    root = Path(g['campaign'])/'D'
    if action == 'cache':
        return [dict(action='cache', target=root/'cache'/d, bucket='gpu', seconds=300,
            extra=['--domain', d], gpu_family='H', reserved_bytes=LIMITS['free_gpu_bytes'])
            for d in ('AlphaBay', 'Dream')]
    return [dict(action=action, target=root/('training' if action == 'train' else 'scoring')/t['id'],
        bucket='training' if action == 'train' else 'gpu', seconds=3600 if action == 'train' else 240,
        extra=['--task', t['id']], gpu_family='H', reserved_bytes=LIMITS['free_gpu_bytes']) for t in specs]


def validate_training(g, task, completion):
    from step29_vendorlink_data import group_chunks
    manifest = read(pinned(g['manifest']))
    n = manifest['counts'][task['domains'][0]]['eligible_accounts']
    if (completion['status'] != 'HEAD_TRAINING_COMPLETE' or completion['task'] != task
            or completion['base'] != g['bases'][task['domains'][0]]
            or completion['updates'] != 10*3*len(group_chunks(list(range(n)), 8))
            or completion['workload'] != dict(current_ads=120*n, current_updates=completion['updates'],
                                              encoder_updates=0, auxiliary_updates=0)):
        raise ValueError('Fixed complete head exposure required')
    pinned(completion['endpoint'])


def decide(result, baseline, rms):
    checks = {}
    for d in ('AlphaBay', 'Dream'):
        local = result['D_'+d+'_LOCAL']['H_SEQ'][0]['R11']
        full = result['D_'+d+'_FULL_POOL']['H_SEQ'][0]['R11']
        v = baseline['D_'+d+'_V1']['V_CLS'][0]['R11']
        old = rms['D_'+d+'_H_CE_RMS']['H_SEQ'][0]['R11']
        checks[d] = dict(LOCAL=local, FULL_POOL=full, V1_CLS=v, RMS=old,
                         FULL_minus_LOCAL=full-local, FULL_minus_RMS=full-old,
                         learning_pass=full >= .70 and full-v >= -.02)
    return dict(status='PASS_LEARNING' if all(c['learning_pass'] for c in checks.values()) else 'STOP_LEARNING',
        checks=checks, positive_alignment_difference=all(c['FULL_minus_LOCAL'] > 0 for c in checks.values()),
        next_phase_authorized=False, application=False)


def worker(g, args):
    from step35_vendorlink_head_adapt import cache_fit, train_head, score_head
    from step35_vendorlink_account_evaluate import analyze
    root = Path(g['campaign'])/'D'
    manifest = read(pinned(g['manifest']))
    specs = tasks(g)
    frozen = read(root/'batch.json')
    if frozen != dict(tasks=specs, gate=record(args.gate), manifest=g['manifest']):
        raise ValueError('Fixed four-task batch identity differs')
    check = Check(Path(g['experiment_root']), max(1, args.deadline-time.monotonic()))
    if args.action != 'analyze':
        configure_compute()
    if args.action == 'cache':
        if args.domain not in ('AlphaBay', 'Dream') or args.target != root/'cache'/args.domain:
            raise ValueError('Only two approved fit caches')
        fit = manifest['files'][args.domain+':fit']
        write(args.target/'access.json', dict(role='fit', source=fit, purpose='frozen_encoder_cache'))
        for b in g['model_binding']['files']:
            pinned(b)
        cache_fit(g['bases'][args.domain], fit, g['model_binding'], args.domain, args.target/'result',
                  dict(gate=record(args.gate), domain=args.domain), check)
    elif args.action in ('train', 'score'):
        task = next(t for t in specs if t['id'] == args.task)
        domain = task['domains'][0]
        origin = root/'training'/task['id']/'run'
        identity = dict(protocol=PROTOCOL, task=task, gate=record(args.gate), manifest=g['manifest'])
        if args.action == 'train':
            if args.target/'run' != origin:
                raise ValueError('Wrong training destination')
            cached = read(root/'cache_complete.json')
            if cached['domains'] != ['AlphaBay', 'Dream']:
                raise ValueError('Complete paired caches required')
            for b in cached['completions']:
                pinned(b)
            fit = manifest['files'][domain+':fit']
            write(args.target/'access.json', dict(role='fit', source=fit, encoder_frozen=True))
            result = train_head(task, record(root/'cache'/domain/'result/completion.json'), fit,
                                identity, origin, check)
            validate_training(g, task, result)
        else:
            if args.target != root/'scoring'/task['id']:
                raise ValueError('Wrong scoring destination')
            frozen = read(root/'training_complete.json')
            if frozen['tasks'] != specs:
                raise ValueError('All four head fits required before scoring')
            for b in frozen['completions']:
                pinned(b)
            completion = read(origin/'completion.json')
            validate_training(g, task, completion)
            if completion['identity'] != identity:
                raise ValueError('Wrong current training identity')
            score_head(task, completion, manifest, g['model_binding'], args.target/'result', check)
    else:
        if args.action != 'analyze' or args.target != root/'analysis':
            raise ValueError('Unknown or out-of-scope action')
        frozen = read(root/'scoring_complete.json')
        if frozen['tasks'] != specs:
            raise ValueError('Whole fixed four-task release only')
        analyze([read(pinned(b)) for b in frozen['completions']], specs, args.target/'result')


def campaign(g, gate_path, started):
    root = Path(g['campaign'])
    if root.exists():
        raise ValueError('No duplicate controller or automatic restart')
    root.mkdir(parents=True)
    ledger = Ledger(root, g['prior_seconds'], g['cpu_test_seconds'], caps=CAPS)
    runner = Runner(g, gate_path, root, ledger, worker_entry=Path(__file__), caps=CAPS, limits=LIMITS,
                    halt_dispatch_on_failure=True, continue_after_oom=True)
    specs = tasks(g)
    write(root/'D/batch.json', dict(tasks=specs, gate=record(gate_path), manifest=g['manifest']))
    ledger.charge('auxiliary', 'controller_setup', time.monotonic()-started, True)
    @contextmanager
    def measure(name):
        ledger.reserve('auxiliary', 60)
        with ledger.active(name):
            yield
    def complete(job):
        if job['action'] == 'train':
            c = read(job['target']/'run/completion.json')
            validate_training(g, c['task'], c)
        elif job['action'] == 'cache':
            c = read(job['target']/'result/completion.json')
            if c['status'] != 'FIT_CACHE_COMPLETE':
                raise ValueError('Incomplete fit cache')
            pinned(c['file'])
        elif job['action'] == 'score':
            c = read(job['target']/'result/completion.json')
            if c['status'] != 'BLIND_SCORES_COMPLETE' or set(c['files']) != {'H_SEQ:R11'} or c['role'] != 'dev':
                raise ValueError('Incomplete fixed head reading')
            pinned(c['files']['H_SEQ:R11'])
        elif not (job['target']/'result/results.json').is_file():
            raise ValueError('Incomplete analysis')
    try:
        runner.run_many(jobs(g, specs, 'cache'), 2, complete)
        with measure('cache_manifest'):
            write(root/'D/cache_complete.json', dict(domains=['AlphaBay', 'Dream'],
                completions=[record(root/'D/cache'/d/'result/completion.json') for d in ('AlphaBay', 'Dream')]))
        runner.run_many(jobs(g, specs, 'train'), 2, complete)
        with measure('training_manifest'):
            write(root/'D/training_complete.json', dict(tasks=specs,
                completions=[record(root/'D/training'/t['id']/'run/completion.json') for t in specs]))
        runner.run_many(jobs(g, specs, 'score'), 2, complete)
        with measure('scoring_manifest'):
            write(root/'D/scoring_complete.json', dict(tasks=specs,
                completions=[record(root/'D/scoring'/t['id']/'result/completion.json') for t in specs]))
        runner.run_many([dict(action='analyze', target=root/'D/analysis', bucket='auxiliary', seconds=600, extra=[])], 1, complete)
        with measure('decision_and_completion'):
            result = decide(read(root/'D/analysis/result/results.json'), read(pinned(g['baseline'])), read(pinned(g['rms_results'])))
            write(root/'decisions/D.json', result)
            write(root/'completion.json', result)
    except BaseException as exc:
        write(root/'failure.json', dict(error=repr(exc), automatic_retry=False,
              results_released=(root/'D/analysis/result/results.json').exists()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--action', choices=['campaign', 'cache', 'train', 'score', 'analyze'], required=True)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--domain')
    parser.add_argument('--task')
    args = parser.parse_args()
    started = time.monotonic()
    if args.action == 'campaign':
        campaign(permit(args.gate), args.gate, started)
        return
    expected = Path(__file__).resolve().parents[2]/'campaign'
    if args.target is None or not args.target.resolve().is_relative_to(expected):
        raise ValueError('Worker output outside isolated campaign')
    try:
        worker(permit(args.gate), args)
    except (ValueError, OSError, SharedDiskLimit) as exc:
        write(args.target/'shared_failure.json', dict(status='SOURCE_DATA_OR_DISK_VALIDITY_FAILURE', error=repr(exc)))
        raise
    except RuntimeError as exc:
        import torch
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or 'CUDA out of memory' in str(exc)
        write(args.target/'worker_failure.json', dict(kind='CUDA_OOM' if oom else 'OTHER', error=repr(exc), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
