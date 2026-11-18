"""Four fixed FIT12 training arms, with a manual same-controller OOM handoff."""
import argparse
import os
from pathlib import Path
import sys

import psutil

from step30_evolution_control import Runner, kernel_identity
from step30_evolution_io import Ledger, PYTHON, read, record, verify, write


PROTOCOL = 'vendorlink-fit12-market-adapters'
SEEDS = (51, 52, 53)
V_ORDER = [('VendorLink_SEQ', seed) for seed in SEEDS]
H_ORDER = [('H_full', 51), ('H_ER', 51), ('H_SEQ', 51), ('H_full', 52), ('H_full', 53),
           ('H_ER', 52), ('H_ER', 53), ('H_SEQ', 52), ('H_SEQ', 53)]
HOURS = {'VendorLink_SEQ': 3, 'H_SEQ': 8, 'H_ER': 12, 'H_full': 14}
CAPS = dict(training=111*3600, gpu=5*3600, auxiliary=4*3600)


def attempt_job(job, number):
    task = job['task_id']
    return dict(job, attempt=number, target=job['task_dir']/f'attempt-{number}',
                operation=f'{task}/attempt-{number}')


class OomHandoff:
    """Only the live dispatch loop consumes a request; workers never own the queue."""
    def __init__(self, jobs, slots):
        self.planned = {job['task_id'] for job in jobs}
        self.slots = slots
        self.second_slot_seconds = jobs[1]['seconds'] if slots > 1 else 0
        self.started = False
        self.warmed = set()
        self.suspended = {}
        self.winners = {}

    def observe_updates(self, active):
        if self.slots == 1 or len(self.warmed) >= 2:
            return
        for slot, item in active.items():
            path = item['job']['target']/'run/workload.json'
            if slot < 2 and slot not in self.warmed and path.is_file() and read(path)['current_updates'] > 0:
                self.warmed.add(slot)

    def finished(self, slot, job, receipt):
        path = job['target']/'dispatch_receipt.json'
        if receipt['success']:
            task = job['task_id']
            if task in self.winners:
                raise ValueError('Two successful attempts for one logical trajectory')
            self.winners[task] = dict(attempt=job['attempt'], receipt=record(path),
                                      completion=record(job['target']/'run/completion.json'))
            if job['attempt'] == 2:
                self.winners[task]['first_oom_receipt'] = record(job['task_dir']/'attempt-1'/'dispatch_receipt.json')
            self.warmed.add(slot)
        elif receipt['failure_kind'] == 'CUDA_OOM' and receipt['released']:
            binding = record(path)
            self.suspended[slot] = (job, dict(task=job['task_id'], attempt=job['attempt'], slot=slot,
                receipt=binding, pid=receipt['process']['pid'],
                start_ticks=receipt['process']['start_ticks'], boot_id=receipt['process']['boot_id']))

    def waiting(self, ordinary_pending):
        if self.slots == 1 and any(job['attempt'] == 2 for job, _ in self.suspended.values()):
            return False
        return any(ordinary_pending or job['attempt'] == 1 for job, _ in self.suspended.values())

    def complete(self):
        return self.planned == self.winners.keys()

    def dispatch_details(self):
        return dict(oom_slots=[row for _, row in self.suspended.values()], successful_tasks=sorted(self.winners))

    def select(self, slot, ordinary, retired, snapshot):
        if slot == 2 and len(self.warmed) < 2:
            return None
        if not self.started and self.slots > 1 and len(psutil.Process().cpu_affinity()) < 4:
            return None
        if len(psutil.Process().cpu_affinity()) < 2*(slot+1):
            return None
        if slot not in retired:
            return ordinary
        if slot not in self.suspended:
            return None
        job, row = self.suspended[slot]
        if not request_marker(row).is_file():
            return None
        # waitpid/release() ran before the failure receipt. Check both kernel and NVML
        # again at admission: a finished process may still appear in the GPU sample.
        try:
            if kernel_identity(row['pid']) == {k: row[k] for k in ('boot_id', 'start_ticks')}:
                return None
        except psutil.NoSuchProcess:
            pass
        if snapshot is None or row['pid'] in snapshot['processes']:
            return None
        if job['attempt'] == 1:
            return attempt_job(job, 2)
        return ordinary  # A second OOM can release capacity, never a third attempt.

    def admission_limits(self, limits):
        if self.started or self.slots == 1:
            return limits
        values = dict(limits)
        for name in ('free_gpu_bytes', 'free_ram_bytes', 'save_reserve_bytes'):
            values[name] *= 2
        return values

    def initial_pair_seconds(self):
        return 0 if self.started else self.second_slot_seconds

    def admitted(self, slot, job, retired, observation):
        self.started = True
        if slot not in self.suspended:
            return
        _, row = self.suspended[slot]
        path = verify(row['receipt'])
        write(path.parent/'slot_reopened.json', dict(previous=row, next_task=job['task_id'],
              next_attempt=job['attempt'], admission=observation))
        del self.suspended[slot]
        retired.remove(slot)


def request_marker(row):
    receipt = row['receipt']
    return Path(receipt['path']).parent/('reopen-'+receipt['sha256']+'.request')


def request_recovery(campaign, task, attempt):
    """Submit an idempotent request for a particular failed attempt, never launch a process."""
    campaign = Path(campaign).resolve()
    if task not in {f'{m}_s{s}' for m, s in V_ORDER+H_ORDER} or attempt not in (1, 2):
        raise ValueError('Unknown fixed trajectory or attempt')
    path = campaign/'training'/task/f'attempt-{attempt}'/'dispatch_receipt.json'
    receipt = read(path)
    if (receipt['task_id'] != task or receipt['attempt'] != attempt or receipt['success']
            or receipt['failure_kind'] != 'CUDA_OOM' or not receipt['released']):
        raise ValueError('A released, recorded CUDA OOM is required')
    binding = record(path)
    marker = request_marker(dict(receipt=binding))
    if marker.exists():
        return marker
    parent = read(campaign/'controller.json')
    process = psutil.Process(parent['pid'])
    if (parent['protocol'] != PROTOCOL or kernel_identity(parent['pid']) !=
            {k: parent[k] for k in ('start_ticks', 'boot_id')} or
            (process.username(), process.cwd(), process.cmdline()) !=
            (parent['owner'], parent['cwd'], parent['command'])):
        raise ValueError('Original controller is no longer live')
    state = read(campaign/'dispatch.json')
    if (state['status'] not in ('RUNNING', 'WAITING_OOM_RECOVERY') or state['budget_stop']
            or not any(row['receipt'] == binding for row in state['oom_slots'])):
        raise ValueError('This attempt is not awaiting recovery in the current block')
    try:
        marker.touch(exist_ok=False)
    except FileExistsError:
        pass  # Concurrent submissions express the same request.
    return marker


class MarketRunner(Runner):
    def __init__(self, *args, source_bindings, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_bindings = source_bindings

    def command(self, job, slot, deadline):
        for binding in self.source_bindings:
            verify(binding)
        cores = psutil.Process().cpu_affinity()[2*slot:2*slot+2]
        if len(cores) != 2:
            raise RuntimeError('Two available cores required for the selected slot')
        return ['taskset', '-c', ','.join(map(str, cores)), PYTHON, '-B', '-u', str(self.worker_entry),
                '--gate', str(self.gate_path), '--action', 'train', '--target', str(job['target']),
                '--deadline', str(deadline), '--task', job['task_id']]


def run_training(runner, tasks, completed):
    """V completes serially, then the H pool. Publish only validated winning attempts."""
    ordered = V_ORDER+H_ORDER
    ordered_ids = [f'{method}_s{seed}' for method, seed in ordered]
    if ([t['id'] for t in tasks] != ordered_ids or [(t['method'], t['seed']) for t in tasks] != ordered
            or any(t['epochs'] != 40 or t['stages'] != 3 or
                   t['domains'] != ['AlphaBay', 'Dream', 'Agora'] for t in tasks)):
        raise ValueError('Exactly the four frozen arms, three paired seeds and all three markets required')
    root = runner.root
    parent = psutil.Process(os.getpid())
    write(root/'controller.json', dict(protocol=PROTOCOL, pid=parent.pid, owner=parent.username(),
          cwd=parent.cwd(), command=parent.cmdline(), **kernel_identity(parent.pid)))
    winners = {}
    for batch, slots in ((tasks[:3], 1), (tasks[3:], 3)):
        jobs = []
        for task in batch:
            name, method = task['id'], task['method']
            vendor = method == 'VendorLink_SEQ'
            jobs.append(attempt_job(dict(task_id=name, task_dir=root/'training'/name, action='train',
                bucket='training', seconds=HOURS[method]*3600, gpu_family='V' if vendor else 'H',
                reserved_bytes=runner.limits['vendor_reserve_bytes' if vendor else 'free_gpu_bytes']), 1))
        handoff = OomHandoff(jobs, slots)
        runner.run_many(jobs, slots, completed, recovery=handoff)
        winners.update(handoff.winners)
    if set(winners) != set(ordered_ids):
        raise RuntimeError('Incomplete training batch')
    manifest = dict(tasks=tasks, winners=winners, completions=[winners[t]['completion'] for t in ordered_ids])
    for value in winners.values():
        verify(value['completion'])
        verify(value['receipt'])
        if value['attempt'] == 2:
            verify(value['first_oom_receipt'])
    write(root/'training_complete.json', manifest)
    return manifest


def campaign(gate):
    gate = Path(gate).resolve()
    g = read(gate)
    if (sys.platform != 'linux' or Path(sys.executable).resolve() != Path(PYTHON).resolve() or
            g['protocol'] != PROTOCOL or g['training_authorized'] is not True or g['review_closed'] is not True
            or g['caps'] != CAPS or g['limits']['workers'] != 3):
        raise ValueError('An approved, reviewed four-arm Linux training gate is required')
    from step35_vendorlink_account_control import validate_training
    sources = [verify(item) for item in g['sources']]
    worker = verify(g['worker_entry'])
    if (worker not in sources or Path(__file__).resolve() not in sources
            or Path(sys.modules[Runner.__module__].__file__).resolve() not in sources):
        raise ValueError('Bind the actual controller, dispatch loop and training worker')
    root = Path(g['campaign']).resolve()
    data = read(verify(g['manifest']))
    tasks = read(verify(g['tasks']))
    root.mkdir(exist_ok=False)
    ledger = Ledger(root, g['prior_seconds'], g['cpu_test_seconds'], caps=CAPS)
    gate_binding = record(gate)
    runner = MarketRunner(g, gate, root, ledger, worker_entry=worker, limits=g['limits'], caps=CAPS,
                          source_bindings=[gate_binding, g['tasks'], g['manifest'], *g['sources']],
                          halt_dispatch_on_failure=True, continue_after_oom=True)
    by_id = {t['id']: t for t in tasks}

    def completed(job):
        task = by_id[job['task_id']]
        result = read(job['target']/'run/completion.json')
        identity = dict(protocol=PROTOCOL, task=task, manifest=g['manifest'], gate=gate_binding)
        validate_training(task, result, data, identity)

    return run_training(runner, tasks, completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    start = sub.add_parser('campaign')
    start.add_argument('--gate', type=Path, required=True)
    reopen = sub.add_parser('request-oom')
    reopen.add_argument('--campaign', type=Path, required=True)
    reopen.add_argument('--task', required=True)
    reopen.add_argument('--attempt', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    if args.action == 'request-oom':
        print(request_recovery(args.campaign, args.task, args.attempt))
    else:
        campaign(args.gate)


if __name__ == '__main__':
    main()
