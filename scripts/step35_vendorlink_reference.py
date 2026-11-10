"""Approved fixed fit4/fit12 reference diagnostic; scoring only, no training."""
import argparse
import json
from pathlib import Path
import sys
import time

from step30_evolution_control import Runner
from step30_evolution_io import Ledger, directory_bytes
from step35_vendorlink_account_control import SOURCES as ORIGINAL_SOURCES, configure_compute
from step35_vendorlink_account_io import PYTHON, pinned, record, write
from step35_vendorlink_account_policy import LIMITS as ORIGINAL_LIMITS

PROTOCOL = 'vendorlink-reference-diagnostic-20261110'
CAPS = dict(gpu=1200, auxiliary=600)
LIMITS = dict(ORIGINAL_LIMITS, rss_bytes=8*2**30, free_ram_bytes=10*2**30,
              disk_bytes=256*2**20, save_reserve_bytes=32*2**20)
SOURCES = ORIGINAL_SOURCES + ('step35_vendorlink_head_adapt.py', 'step35_vendorlink_reference.py')
DOMAINS = ('AlphaBay', 'Dream')


def read(path):
    return json.loads(Path(path).read_text())


def tasks():
    # These describe readers, not newly trained models. Base training identity is separate.
    return [dict(id='D_'+d+'_REFERENCE', phase='D', domains=[d], seed=41,
                 method='H_SEQ', settings=dict(diagnostics=False)) for d in DOMAINS]


def reference_vectors(cache, cache_completion, fit_binding, accounts, qids, query):
    import torch
    ids = [a['account_uid'] for a in accounts]
    records = [r['uid'] for a in accounts for r in a['records']]
    if (ids != sorted(set(ids)) or qids != ids or cache['classes'] != ids
            or any(len(a['records']) != 12 for a in accounts)
            or len(records) != len(set(records)) or cache['records'] != records
            or cache['fit'] != fit_binding or cache_completion['fit'] != fit_binding
            or cache['base'] != cache_completion['base'] or cache['identity'] != cache_completion['identity']
            or cache['encoder_frozen'] is not True or cache['encoder_mode'] != 'eval'):
        raise ValueError('Exact frozen fit order and encoder identity required')
    qrecords = [r.uid for v in query for r in v.records]
    if (len(query) != len(ids) or any(len(v.records) != 2 for v in query)
            or len(qrecords) != len(set(qrecords)) or set(qrecords) & set(records)):
        raise ValueError('Dev Q2 must be disjoint from every fit reference')
    z = cache['vectors']
    if z.ndim != 3 or z.shape[:2] != (len(ids), 12) or not torch.isfinite(z).all():
        raise ValueError('Complete finite fit12 vectors required')
    return z.detach()


def score_reference_vectors(model, query, reference, check=lambda: None):
    from step35_vendorlink_account_evaluate import rectangular
    if query.shape != (len(reference), 2, reference.shape[-1]) or reference.shape[1] != 12:
        raise ValueError('Fixed Q2 and fit12 reference shape required')
    return {'H_FIT'+str(k)+':R11': rectangular(model, query, reference[:, :k], check) for k in (4, 12)}


def validate_score(completion, task, g):
    domain = task['domains'][0]
    if (completion['task'] != task or completion['status'] != 'BLIND_SCORES_COMPLETE'
            or completion['role'] != 'dev' or set(completion['files']) != {'H_FIT4:R11', 'H_FIT12:R11'}
            or completion['base'] != g['bases'][domain]
            or completion['cache'] != g['caches'][domain] or completion['training_updates'] != 0):
        raise ValueError('Both fixed reference readings required per market')
    for binding in completion['files'].values():
        pinned(binding)


def decide(result, baseline, rms):
    checks = {}
    for d in DOMAINS:
        row = result['D_'+d+'_REFERENCE']
        fit4, fit12 = (row['H_FIT'+str(k)][0]['R11'] for k in (4, 12))
        old = rms['D_'+d+'_H_CE_RMS']['H_SEQ'][0]['R11']
        vendor = baseline['D_'+d+'_V1']['V_CLS'][0]['R11']
        checks[d] = dict(G4=old, FIT4=fit4, FIT12=fit12, V1_CLS=vendor,
            FIT4_minus_G4=fit4-old, FIT12_minus_FIT4=fit12-fit4,
            FIT12_minus_V1_CLS=fit12-vendor, candidate_pass=fit12 >= .70 and fit12-vendor >= -.02)
    return dict(status='PASS_REFERENCE_DIAGNOSTIC' if all(x['candidate_pass'] for x in checks.values())
                else 'STOP_REFERENCE_DIAGNOSTIC', checks=checks, next_phase_authorized=False,
                original_G4_failure_unchanged=True, training_updates=0)


def permit(path):
    g = read(path)
    root, workspace = Path(g['experiment_root']).resolve(), Path(g['workspace']).resolve()
    if (sys.platform != 'linux' or Path(sys.executable).resolve() != Path(PYTHON).resolve()
            or g['protocol'] != PROTOCOL or not g['approved'] or not g['review_waived']
            or not g['primary_closed'] or g['caps'] != CAPS or g['limits'] != LIMITS
            or g['automatic_retry'] is not False
            or g['worker_failure_policy'] != 'retire_oom_continue_current_phase_no_retry'
            or Path(__file__).resolve() != workspace/'scripts'/Path(__file__).name
            or not workspace.is_relative_to(root) or not Path(g['campaign']).resolve().is_relative_to(root)):
        raise ValueError('Approved isolated scoring-only scope required')
    if {Path(b['path']).resolve() for b in g['sources']} != {workspace/'scripts'/n for n in SOURCES}:
        raise ValueError('Complete current sources required')
    for b in g['sources']+[g['scope'], g['cpu'], g['manifest'], g['baseline'], g['rms_results']]+list(g['bases'].values())+list(g['caches'].values()):
        pinned(b)
    if read(g['cpu']['path'])['status'] != 'PASS':
        raise ValueError('Necessary CPU check incomplete')
    return g


def worker(g, args):
    import psutil
    import numpy as np
    from step35_vendorlink_account_evaluate import analyze, encode, load_role
    root = Path(g['campaign'])/'D'
    if read(root/'batch.json') != dict(tasks=tasks(), gate=record(args.gate), manifest=g['manifest']):
        raise ValueError('Fixed diagnostic batch differs')
    last_disk = 0.
    def check():
        nonlocal last_disk
        now = time.monotonic()
        if now >= args.deadline-5 or psutil.Process().memory_info().rss > LIMITS['rss_bytes']:
            raise RuntimeError('Scoring lifecycle or RSS cap')
        if now-last_disk > 5:
            last_disk = now
            if directory_bytes(Path(g['experiment_root'])) > LIMITS['disk_bytes']:
                raise OSError('Shared diagnostic disk cap')
    if args.action == 'analyze':
        if args.target != root/'analysis':
            raise ValueError('Wrong analysis destination')
        completed = read(root/'scoring_complete.json')
        if completed['tasks'] != tasks() or len(completed['completions']) != 2:
            raise ValueError('Both markets and all four readings required')
        values = [read(pinned(b)) for b in completed['completions']]
        for c, task in zip(values, tasks(), strict=True):
            validate_score(c, task, g)
        analyze(values, tasks(), args.target/'result')
        check()
        return
    import torch
    from step35_vendorlink_head_adapt import base_model
    configure_compute()
    task = next(t for t in tasks() if t['domains'][0] == args.domain)
    if args.action != 'score' or args.target != root/'scoring'/task['id']:
        raise ValueError('Only two fixed market scoring jobs')
    manifest = read(pinned(g['manifest']))
    sources = {role: manifest['files'][args.domain+':'+role] for role in ('fit', 'dev')}
    write(args.target/'access.json', dict(sources=sources, purpose='fit_as_registered_reference', confirm=False))
    cached = read(pinned(g['caches'][args.domain]))
    if cached['base'] != g['bases'][args.domain] or cached['status'] != 'FIT_CACHE_COMPLETE':
        raise ValueError('Cache must belong to the unchanged RMS endpoint')
    cache = torch.load(pinned(cached['file']), map_location='cpu', weights_only=False)
    accounts = read(pinned(sources['fit']))
    qids, query = load_role(sources['dev'], args.domain, 'dev', len(accounts))
    reference = reference_vectors(cache, cached, sources['fit'], accounts, qids, query)
    for b in g['model_binding']['files']:
        pinned(b)
    model = base_model(g['bases'][args.domain], g['model_binding'])
    scores = score_reference_vectors(model, encode(model, query, check), reference.to('cuda:0'), check)
    target = args.target/'result'
    if target.exists():
        raise ValueError('Fresh score output required')
    target.mkdir()
    files = {}
    for name, matrix in scores.items():
        check()
        p = target/(name.replace(':', '_')+'.npy')
        np.save(p, matrix)
        files[name] = record(p)
    write(target/'completion.json', dict(task=task, role='dev', status='BLIND_SCORES_COMPLETE', cells=['R11'],
        files=files, sources=sources, base=g['bases'][args.domain], cache=g['caches'][args.domain],
        selection='first 4 / all 12 in frozen fit order', training_updates=0))


def campaign(g, gate_path):
    root = Path(g['campaign'])
    if root.exists():
        raise ValueError('No duplicate campaign or retry')
    root.mkdir()
    ledger = Ledger(root, g['prior_seconds'], g['cpu_test_seconds'], caps=CAPS)
    runner = Runner(g, gate_path, root, ledger, worker_entry=Path(__file__), caps=CAPS, limits=LIMITS,
                    wait_for_resources=False, halt_dispatch_on_failure=True, continue_after_oom=True)
    with ledger.active('setup'):
        write(root/'D/batch.json', dict(tasks=tasks(), gate=record(gate_path), manifest=g['manifest']))
    def complete(job):
        if job['action'] == 'score':
            c = read(job['target']/'result/completion.json')
            validate_score(c, next(t for t in tasks() if t['id'] == job['target'].name), g)
        elif not (job['target']/'result/results.json').is_file():
            raise ValueError('Incomplete whole-batch analysis')
    try:
        jobs = [dict(action='score', target=root/'D/scoring'/t['id'], bucket='gpu', seconds=600,
                     extra=['--domain', t['domains'][0]], reserved_bytes=LIMITS['free_gpu_bytes']) for t in tasks()]
        runner.run_many(jobs, 2, complete)
        with ledger.active('score_manifest'):
            write(root/'D/scoring_complete.json', dict(tasks=tasks(), completions=[record(j['target']/'result/completion.json') for j in jobs]))
        runner.run_many([dict(action='analyze', target=root/'D/analysis', bucket='auxiliary', seconds=120, extra=[])], 1, complete)
        with ledger.active('decision'):
            outcome = decide(read(root/'D/analysis/result/results.json'), read(pinned(g['baseline'])), read(pinned(g['rms_results'])))
            write(root/'decisions/D.json', outcome)
            write(root/'completion.json', outcome)
    except BaseException as exc:
        write(root/'failure.json', dict(error=repr(exc), automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--action', choices=['campaign', 'score', 'analyze'], required=True)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--domain', choices=DOMAINS)
    args = parser.parse_args()
    g = permit(args.gate)
    if args.action == 'campaign':
        campaign(g, args.gate)
        return
    if args.target is None or not args.target.resolve().is_relative_to(Path(g['campaign']).resolve()):
        raise ValueError('Worker destination outside new campaign')
    try:
        worker(g, args)
    except (ValueError, OSError) as exc:
        write(args.target/'shared_failure.json', dict(error=repr(exc)))
        raise
    except RuntimeError as exc:
        import torch
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or 'CUDA out of memory' in str(exc)
        write(args.target/'worker_failure.json', dict(kind='CUDA_OOM' if oom else 'OTHER', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
