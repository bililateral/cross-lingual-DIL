"""Resume the approved finite pipeline without rewriting any prior identity."""
import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import sys
import time

import psutil

import step34_vendorlink_account_control as native
from step34_vendorlink_account_data import Check, SharedDiskLimit, pinned, record, write
from step34_vendorlink_account_policy import PROTOCOL, tasks
from step30_evolution_control import Runner
from step30_evolution_io import Ledger

read = native.read
GLOBAL_CAPS = dict(training=178200, gpu=21600, auxiliary=16200)
HOLD = dict(training=28800, gpu=0, auxiliary=1800)


def limits(g, task=None):
    result = dict(g['limits'])
    if task and task['method'] == 'VendorLink_SEQ':
        result.update(allocator_bytes=15*2**30, process_gpu_bytes=16*2**30,
                      free_gpu_bytes=16*2**30)
    return result


def permit(path):
    g = read(path)
    r = g['resumption']
    if not r['approved'] or not r['review_closed'] or r['caps'] != GLOBAL_CAPS:
        raise ValueError('Source-bound queue resumption must be approved and closed')
    if record(Path(__file__)) != r['source']:
        raise ValueError('Actual resumption entry differs')
    for b in (r['authorization'], r['scope'], r['review_disposition'], r['cpu']):
        pinned(b)
    return native.permit(path)  # Original 12 scientific sources remain byte-identical.


def sourced_result(g, task, row):
    """Verify the original gate and endpoint identity, never relabel a checkpoint."""
    origin = read(pinned(row['gate']))
    if (origin['protocol'] != PROTOCOL or not origin['approved'] or not origin['review_closed']
            or origin['manifest'] != g['manifest'] or origin['model_binding'] != g['model_binding']):
        raise ValueError('Foreign training source or model/data identity')
    if 'resumption' in origin and not (origin['resumption']['approved'] and origin['resumption']['review_closed']):
        raise ValueError('Unclosed resumed training source')
    if {Path(b['path']).name for b in origin['sources']} != set(native.SOURCES):
        raise ValueError('Incomplete original scientific source bindings')
    for b in origin['sources']:
        pinned(b)
    result = read(pinned(row['completion']))
    identity = dict(protocol=PROTOCOL, task=task, manifest=g['manifest'], gate=row['gate'])
    native.validate_training(task, result, read(pinned(g['manifest'])), identity)
    return result


def collect_training(g, specs, origins):
    if set(origins) != {t['id'] for t in specs}:
        raise ValueError('Whole phase training provenance is incomplete')
    for task in specs:
        sourced_result(g, task, origins[task['id']])
    return dict(tasks=specs, origins=origins)


def inherited_ledger(root, prior):
    ledger = Ledger.__new__(Ledger)
    ledger.root = root
    ledger.caps = {k: GLOBAL_CAPS[k]-HOLD[k] for k in GLOBAL_CAPS}
    ledger.used, ledger.events = dict(prior['used_seconds']), list(prior['events'])
    ledger.cpu_tests = prior['cpu_test_seconds']
    if any(ledger.used[k] > ledger.caps[k] for k in ledger.caps):
        raise ValueError('No room for the full existing H45 reservation')
    ledger.save()
    return ledger


def merge_h45(ledger, root, external):
    marker = root/'H45_accounting_merged.json'
    if marker.exists():
        raise ValueError('H45 incremental ledger already merged')
    if external['caps'] != dict(training=28800, gpu=0, auxiliary=1800):
        raise ValueError('External H45 quota differs')
    if any(external['used_seconds'][k] > HOLD[k] for k in HOLD):
        raise ValueError('External H45 exceeded its reserved quota')
    combined = {k: ledger.used[k]+external['used_seconds'][k] for k in GLOBAL_CAPS}
    if any(combined[k] > GLOBAL_CAPS[k] for k in GLOBAL_CAPS):
        raise ValueError('Merged global budget exceeded')
    ledger.used, ledger.caps = combined, dict(GLOBAL_CAPS)
    ledger.events.extend(external['events'])
    ledger.cpu_tests += external['cpu_test_seconds']
    ledger.save()
    write(marker, dict(incremental_used=external['used_seconds'], duplicate_history=False))


def process_live(row):
    try:
        p = psutil.Process(row['pid'])
        if (p.create_time() != row['created'] or p.username() != row['owner']
                or p.cmdline() != row['command'] or p.cwd() != row['cwd']):
            raise ValueError('Existing controller identity changed')
        return True
    except psutil.NoSuchProcess:
        return False


def worker(g, args):
    root = Path(g['campaign'])
    task = None
    if args.action in ('train', 'score'):
        batch = read(root/args.phase/'batch.json')
        expected = tasks(args.phase, batch['selected']) if args.phase != 'C' else tasks('P', batch['selected'])+tasks('M', batch['selected'])
        if batch['tasks'] != expected or batch['manifest'] != g['manifest'] or batch['gate'] != record(args.gate):
            raise ValueError('Frozen resumed phase differs')
        task = next(t for t in expected if t['id'] == args.task)
    effective = limits(g, task)
    write(args.target/'effective_resources.json', dict(limits=effective, task=args.task))
    # The unchanged real worker reads this dictionary at its actual CUDA setter.
    native.LIMITS = effective
    if args.action != 'score':
        return native.worker(g, args)
    if args.target != root/args.phase/'scoring'/task['id']:
        raise ValueError('Wrong resumed blind scoring destination')
    if args.phase in ('M', 'C') and read(root/'decisions/P.json')['status'] != 'PASS_P':
        raise ValueError('P gate closed')
    if args.phase == 'C' and read(root/'decisions/M.json')['status'] != 'PASS_SYSTEM':
        raise ValueError('M gate closed')
    frozen = read(root/args.phase/'training_complete.json')
    if frozen['tasks'] != expected:
        raise ValueError('Whole batch training must finish before scoring')
    if set(frozen['origins']) != {t['id'] for t in expected}:
        raise ValueError('Whole phase completion bindings are incomplete')
    for row in frozen['origins'].values():
        pinned(row['completion'])
    completion = sourced_result(g, task, frozen['origins'][task['id']])
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(effective['allocator_bytes']/torch.cuda.get_device_properties(0).total_memory)
    from step34_vendorlink_account_evaluate import score
    check = Check(Path(g['experiment_root']), max(1, args.deadline-time.monotonic()))
    score(task, completion, read(pinned(g['manifest'])), g['model_binding'], args.target/'result',
          'confirm' if args.phase == 'C' else 'dev', check)


def campaign(g, gate):
    here, root = Path(g['workspace']).parent, Path(g['campaign'])
    if (here/'claim').exists() or root.exists():
        raise ValueError('No duplicate resumption or automatic retry')
    with (here/'claim').open('x') as out:
        out.write(str(psutil.Process().pid))
    r = g['resumption']
    primary, h45 = r['primary'], r['H45']
    wait_active = 0.
    while process_live(primary['controller']):
        tick = time.monotonic()
        write(here/'state.json', dict(status='WAIT_PRIMARY_SETTLEMENT', task='P_VendorLink_SEQ_s45'))
        wait_active += time.monotonic()-tick
        if wait_active+r['preparation_seconds'] >= 600:
            raise TimeoutError('Finite predecessor-wait activity quota')
        time.sleep(30)
    if read(Path(primary['campaign'])/'dispatch.json')['active']:
        raise ValueError('Primary worker release unconfirmed')
    primary_prior = read(Path(primary['campaign'])/'budget.json')
    root.mkdir()
    ledger = inherited_ledger(root, primary_prior)
    write(root/'budget_basis.json', dict(primary=record(Path(primary['campaign'])/'budget.json'),
          external_H45_reserved=HOLD, external_H45_campaign=h45['campaign'],
          original_tail_not_recharged=True))
    ledger.cpu_tests += r['cpu_seconds']
    ledger.charge('auxiliary', 'resumption_preparation_and_active_wait', r['preparation_seconds']+wait_active, True)
    manifest = read(pinned(g['manifest']))
    selected = read(pinned(r['selection']))
    if selected != {'H': 'H1', 'V': 'V0'}:
        raise ValueError('Original D selection differs')
    phase_origins = {'P': dict(r['completed_P']), 'M': {}}

    class ResumingRunner(Runner):
        def retire_waiter(self, active):
            super().retire_waiter(active)
            # A shared validity failure in the existing H45 stops only our new children.
            if (Path(h45['target'])/'shared_failure.json').exists():
                raise ValueError('External H45 shared validity failure')

    def runner(profile):
        allowed = dict(g)
        allowed['other_gpu_commitments'] = [h45['worker']] if process_live(h45['controller']) else []
        return ResumingRunner(allowed, gate, root, ledger, worker_entry=Path(__file__),
                              caps=ledger.caps, limits=profile, halt_dispatch_on_failure=True)

    def complete(job):
        if job['action'] == 'train':
            task = read(job['target']/'run/completion.json')['task']
            b = dict(completion=record(job['target']/'run/completion.json'), gate=record(gate))
            sourced_result(g, task, b)
            latest = job['target']/'run/latest.pt'
            old = record(latest)
            latest.unlink()
            write(job['target']/'latest_retired.json', old)
            phase_origins[task['phase']][task['id']] = b
        elif job['action'] in ('score', 'application'):
            if not (job['target']/'result/completion.json').exists():
                raise ValueError('Missing scoring completion')
        elif not (job['target']/'result/results.json').exists():
            raise ValueError('Missing analysis completion')

    def jobs(phase, specs, action):
        for task in specs:
            target = root/phase/('training' if action == 'train' else 'scoring')/task['id']
            yield dict(action=action, target=target, bucket='training' if action == 'train' else 'gpu',
                       seconds=(14400 if task['method'] == 'VendorLink_SEQ' else 28800) if action == 'train' else 3600,
                       extra=['--phase', phase, '--task', task['id']])

    @contextmanager
    def measure(name):
        ledger.reserve('auxiliary', 60)
        with ledger.active(name):
            yield

    def save(name, value):
        if name == 'selection' and value != selected:
            raise ValueError('Replayed D selection differs from the frozen selection')
        write(root/'decisions'/f'{name}.json', value)

    def run_phase(phase, selection):
        if phase == 'D':
            return read(pinned(r['D_results']))
        specs = tasks(phase, selection) if phase != 'C' else tasks('P', selection)+tasks('M', selection)
        write(root/phase/'batch.json', dict(tasks=specs, selected=selection, manifest=g['manifest'], gate=record(gate)))
        if phase == 'P':
            # Accept V44 only from the successful15GiB attempt, never the deleted OOM endpoint.
            v44 = next(t for t in specs if t['id'] == 'P_VendorLink_SEQ_s44')
            phase_origins['P'][v44['id']] = dict(completion=record(Path(primary['target'])/'run/completion.json'), gate=primary['gate'])
            for task in specs:
                if task['id'] in phase_origins['P']:
                    sourced_result(g, task, phase_origins['P'][task['id']])
            if (Path(h45['target'])/'dispatch_receipt.json').exists() and not read(Path(h45['target'])/'dispatch_receipt.json')['success']:
                raise ValueError('Existing H45 failed; no new dispatch')
            pending = [t for t in specs if t['id'] == 'P_VendorLink_SEQ_s45']
            if (Path(r['original_campaign'])/'P/training/P_VendorLink_SEQ_s45/process.json').exists():
                raise ValueError('Original pending V45 has already been dispatched')
            write(here/'state.json', dict(status='P_VENDOR45', automatic_handoff=True))
            runner(limits(g, pending[0])).run_many(list(jobs('P', pending, 'train')), 1, complete)
            active = 0.
            while process_live(h45['controller']):
                tick = time.monotonic()
                ledger.reserve('auxiliary', math.ceil(active)+60)
                write(here/'state.json', dict(status='WAIT_H45_SETTLEMENT'))
                active += time.monotonic()-tick
                time.sleep(30)
            ledger.charge('auxiliary', 'H45_settlement_wait_activity', active, True)
            if read(Path(h45['campaign'])/'dispatch.json')['active']:
                raise ValueError('H45 worker release unconfirmed')
            merge_h45(ledger, root, read(Path(h45['campaign'])/'budget.json'))
            htask = next(t for t in specs if t['id'] == 'P_H_SEQ_s45')
            phase_origins['P'][htask['id']] = dict(completion=record(Path(h45['target'])/'run/completion.json'), gate=h45['gate'])
        elif phase == 'M':
            runner(limits(g)).run_many(list(jobs(phase, specs, 'train')), 2, complete)
        origins = {**phase_origins['P'], **phase_origins['M']} if phase == 'C' else phase_origins[phase]
        with measure(phase+'_training_manifest'):
            write(root/phase/'training_complete.json', collect_training(g, specs, origins))
        # Original task order; scoring uses the corresponding14/16GiB resource profile.
        # Serial scoring prevents reserving32GiB for two VendorLink scorers on this card.
        for task, job in zip(specs, jobs(phase, specs, 'score')):
            runner(limits(g, task)).run_many([job], 1, complete)
        with measure(phase+'_scoring_manifest'):
            scoring = [record(root/phase/'scoring'/t['id']/'result/completion.json') for t in specs]
            write(root/phase/'scoring_complete.json', dict(tasks=specs, completions=scoring))
        job = dict(action='analyze', target=root/phase/'analysis', bucket='auxiliary', seconds=1800,
                   extra=['--phase', phase])
        runner(limits(g)).run_many([job], 1, complete)
        with measure(phase+'_analysis_read'):
            return read(root/phase/'analysis/result/results.json')

    try:
        result = native.pipeline(run_phase, save, measure)
        with measure('quantitative_completion'):
            write(root/'quantitative_completion.json', result)
        if result['application']:
            job = dict(action='application', target=root/'application', bucket='gpu', seconds=3600,
                       extra=['--phase', 'M'])
            runner(limits(g)).run_many([job], 1, complete)
        with measure('completion'):
            write(root/'completion.json', result)
        write(here/'state.json', dict(status='COMPLETE', automatic_retry=False))
    except BaseException as error:
        write(root/'failure.json', dict(error=repr(error), automatic_retry=False))
        write(here/'state.json', dict(status='FAILED', error=repr(error), automatic_retry=False))
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--action', choices=['campaign', 'train', 'score', 'analyze', 'application'], required=True)
    p.add_argument('--target', type=Path)
    p.add_argument('--deadline', type=float)
    p.add_argument('--phase')
    p.add_argument('--task')
    args = p.parse_args()
    if args.action == 'campaign':
        g = permit(args.gate)
        return campaign(g, args.gate)
    expected = Path(__file__).resolve().parents[2]/'campaign'
    if args.target is None or not args.target.resolve().is_relative_to(expected):
        raise ValueError('Worker target outside approved resumed campaign')
    try:
        g = permit(args.gate)
        worker(g, args)
    except (ValueError, SharedDiskLimit, OSError) as error:
        write(args.target/'shared_failure.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    main()
