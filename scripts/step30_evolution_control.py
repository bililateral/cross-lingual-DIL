"""Finite Evolution queue: up to two slots, failed slots retire, no retry."""
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from step30_evolution_io import (Budget, Ledger, PYTHON, access, dataset, directory_bytes,
                                  permit, pinned, qualification_directory, read, record, sha, verify, write)
from step30_evolution_policy import CAPS, DOMAINS, LIMITS, decision, select_a, tasks


def resource_observation(allowed: dict, root: Path, gpu: bool, owned=(), nv=None) -> dict:
    import shutil
    root = Path(allowed['experiment_root'])
    ram_pending = 0
    for row in owned:
        try:
            ram_pending += max(0, LIMITS['free_ram_bytes']-psutil.Process(row['pid']).memory_info().rss)
        except psutil.NoSuchProcess:
            pass
    save_reserve = (len(owned)+1)*2**30
    observation = dict(ram_available=psutil.virtual_memory().available,
                       ram_pending=ram_pending, disk_free=shutil.disk_usage(root).free,
                       experiment_bytes=directory_bytes(root), save_reserve=save_reserve)
    observation['ready'] = (observation['ram_available']-ram_pending >= LIMITS['free_ram_bytes'] and
                            observation['disk_free'] >= LIMITS['free_disk_reserve']+save_reserve and
                            observation['experiment_bytes']+save_reserve <= LIMITS['disk_bytes'])
    if gpu:
        from step28_formal_queue import GpuMemory
        temporary_reader = nv is None
        if temporary_reader:
            nv = GpuMemory()
        try:
            memory = nv.snapshot()
            used = memory['processes']
            unreflected, commitments = 0, []
            for row in [*allowed['other_gpu_commitments'], *owned]:
                try:
                    process = psutil.Process(row['pid'])
                    if process.create_time() != row['created'] or process.username() != row['owner']:
                        raise ValueError('Declared active commitment process identity changed')
                    if row not in owned and (process.cwd() != row['cwd'] or process.cmdline() != row['command']):
                        raise ValueError('Declared active commitment command/cwd changed')
                    remaining = max(0, row['reserved_bytes']-used.get(row['pid'], 0))
                    unreflected += remaining
                    commitments.append(dict(**row, unreflected_bytes=remaining))
                except psutil.NoSuchProcess:
                    commitments.append(dict(**row, exited=True))
            if set(used)-{r['pid'] for r in commitments if not r.get('exited')}:
                raise ValueError('Unrecognized GPU job; its commitment must be checked before admission')
            observation.update(gpu_free=memory['free'], unreflected_commitments=unreflected, commitments=commitments)
            observation['ready'] &= memory['free']-unreflected >= LIMITS['free_gpu_bytes']
        finally:
            if temporary_reader:
                nv.close()
    return observation


def release(process):
    """Terminate only the newly created worker process group; never other jobs."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    # A worker is not allowed to leave descendants behind, even after exit.
    remaining = []
    for p in psutil.process_iter(['pid', 'username', 'status']):
        try:
            if os.getpgid(p.pid) == process.pid and p.status() != psutil.STATUS_ZOMBIE:
                if p.username() != psutil.Process().username():
                    raise RuntimeError('Unexpected foreign process group member')
                remaining.append(p.pid)
        except (ProcessLookupError, psutil.NoSuchProcess):
            pass
    if remaining:
        os.killpg(process.pid, signal.SIGKILL)
        raise RuntimeError('Unexpected worker descendants; campaign stops')


class Runner:
    def __init__(self, allowed: dict, gate_path: Path, root: Path, ledger: Ledger):
        self.allowed, self.gate_path, self.root, self.ledger = allowed, gate_path, root, ledger
        self.failed = False

    def retire_waiter(self, active):
        binding = self.allowed.get('waiter')
        if not binding or (self.root/'waiter_deleted.json').exists():
            return
        for item in active.values():
            path = item['job']['target']/'run/updates.jsonl'
            if item['job']['action'] != 'train' or not path.exists():
                continue
            import json
            with path.open() as source:
                first = source.readline()
            try:
                update = json.loads(first)
            except json.JSONDecodeError:
                continue  # The writer may still be completing its first line.
            if update.get('optimizer_updates') == 1 and update.get('total_updates', 0) >= 1:
                script = verify(binding)
                if script != Path(self.allowed['workspace'])/'scripts/once_wait.py':
                    raise ValueError('Only this campaign one-shot wrapper may delete itself')
                script.unlink()
                write(self.root/'waiter_deleted.json', dict(source=binding, first_update=str(path),
                      total_updates=update['total_updates'], deleted=True, queue_continues=True))
                return

    def command(self, job: dict, slot: int, deadline: float):
        cores = psutil.Process().cpu_affinity()
        if len(cores) < 2*LIMITS['workers']:
            raise RuntimeError('Four allowed CPU cores required for two slots')
        chosen = cores[-4:][slot*2:slot*2+2]
        return ['taskset', '-c', ','.join(map(str, chosen)), PYTHON, '-B', '-u', str(Path(__file__).resolve()),
                '--gate', str(self.gate_path), '--action', job['action'], '--target', str(job['target']),
                '--deadline', str(deadline), *job['extra']]

    def run_many(self, jobs: list[dict], slots: int, completed):
        """One parent owns dispatch and accounting; healthy slots keep taking work."""
        import shutil
        from step28_formal_queue import GpuMemory
        if self.failed:
            raise RuntimeError('Incomplete phase cannot start downstream work')
        if not 1 <= slots <= LIMITS['workers']:
            raise ValueError('Unapproved slot count')
        active, retired, failures = {}, set(), []
        next_job, parent_active, budget_stop = 0, 0., None
        gpu = any(j['bucket'] in ('training', 'gpu') for j in jobs)
        nv = GpuMemory() if gpu else None
        def finish(slot, error=None):
            item = active.pop(slot)
            p, job = item['process'], item['job']
            try:
                release(p)
            except BaseException as exc:
                error = error or repr(exc)
            elapsed = time.monotonic()-item['started']
            if p.returncode or elapsed > job['seconds']:
                error = error or f'Worker exit {p.returncode} or lifecycle cap exceeded'
            if error is None:
                try:
                    completed(job)
                except BaseException as exc:
                    error = repr(exc)
            receipt = dict(success=error is None, error=error, seconds=elapsed, slot=slot,
                           released=p.poll() is not None, peak_gpu_bytes=item['peak_gpu'], peak_rss_bytes=item['peak_rss'])
            write(job['target']/'dispatch_receipt.json', receipt)
            self.ledger.charge(job['bucket'], job['target'].name, elapsed, error is None)
            if error is not None:
                retired.add(slot)
                failures.append(dict(task=job['target'].name, slot=slot, error=error))
        try:
            while active or (next_job < len(jobs) and len(retired) < slots and budget_stop is None):
                tick = time.monotonic()
                try:
                    snapshot = nv.snapshot() if nv else None
                    self.retire_waiter(active)
                    for slot, item in list(active.items()):
                        p, job = item['process'], item['job']
                        error = None
                        if p.poll() is None:
                            try:
                                rss = psutil.Process(p.pid).memory_info().rss
                                occupied = snapshot['processes'].get(p.pid, 0) if snapshot else 0
                                item['peak_rss'] = max(item['peak_rss'], rss)
                                item['peak_gpu'] = max(item['peak_gpu'], occupied)
                                if time.monotonic() >= item['deadline']-15:
                                    error = 'Full worker lifecycle deadline'
                                elif rss > LIMITS['rss_bytes'] or occupied > LIMITS['process_gpu_bytes']:
                                    error = 'Worker RAM/GPU limit'
                                elif (directory_bytes(Path(self.allowed['experiment_root'])) > LIMITS['disk_bytes']
                                      or shutil.disk_usage(self.root).free < LIMITS['free_disk_reserve']):
                                    error = 'Shared disk limit'
                            except psutil.NoSuchProcess:
                                pass
                        if p.poll() is not None or error:
                            finish(slot, error)
                    for slot in range(slots):
                        if slot in active or slot in retired or next_job == len(jobs) or budget_stop:
                            continue
                        job = jobs[next_job]
                        outstanding = sum(x['job']['seconds'] for x in active.values() if x['job']['bucket'] == job['bucket'])
                        try:
                            self.ledger.reserve(job['bucket'], outstanding+job['seconds'])
                            auxiliary_reserved = sum(x['job']['seconds'] for x in active.values() if x['job']['bucket'] == 'auxiliary')
                            self.ledger.reserve('auxiliary', auxiliary_reserved+math.ceil(parent_active)+60+
                                                (job['seconds'] if job['bucket'] == 'auxiliary' else 0))
                        except TimeoutError as exc:
                            if not active:
                                budget_stop = str(exc)
                                break
                            continue  # In-flight reservations can be released after completion.
                        owned = [x['identity'] for x in active.values()]
                        observation = resource_observation(self.allowed, self.root, gpu, owned, nv)
                        write(job['target']/'admission.json', observation)
                        if not observation['ready']:
                            continue
                        next_job += 1  # A launched/failed launch is consumed, never retried.
                        started = time.monotonic()
                        deadline = started+job['seconds']
                        process = None
                        try:
                            argv = self.command(job, slot, deadline)
                            environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                                TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                                CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='0' if gpu else '')
                            with (job['target']/'console.txt').open('x') as output:
                                process = subprocess.Popen(argv, env=environment, cwd=self.allowed['workspace'],
                                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True, umask=0o077)
                            p = psutil.Process(process.pid)
                            identity = dict(pid=p.pid, owner=p.username(), created=p.create_time(), command=argv,
                                            cwd=p.cwd(), reserved_bytes=LIMITS['free_gpu_bytes'])
                            # taskset execs in place, so the command after exec omits its own wrapper.
                            identity['command'] = argv[3:]
                            active[slot] = dict(process=process, job=job, identity=identity, started=started,
                                                deadline=deadline, peak_gpu=0, peak_rss=0)
                            write(job['target']/'process.json', dict(**identity, launch_command=argv, slot=slot))
                        except BaseException as exc:
                            if slot in active:
                                finish(slot, repr(exc))
                            else:
                                if process is not None:
                                    release(process)
                                elapsed = time.monotonic()-started
                                write(job['target']/'dispatch_receipt.json', dict(success=False, error=repr(exc),
                                      seconds=elapsed, slot=slot, released=True))
                                self.ledger.charge(job['bucket'], job['target'].name, elapsed, False)
                                retired.add(slot)
                                failures.append(dict(task=job['target'].name, slot=slot, error=repr(exc)))
                    write(self.root/'dispatch.json', dict(active={s: x['identity'] for s, x in active.items()},
                          retired=sorted(retired), failures=failures, pending=[j['target'].name for j in jobs[next_job:]],
                          budget_stop=budget_stop, status='RUNNING' if active else 'WAITING_RESOURCES'))
                finally:
                    parent_active += time.monotonic()-tick
                if self.ledger.used['auxiliary']+math.ceil(parent_active)+60 >= CAPS['auxiliary']:
                    budget_stop = 'Auxiliary monitoring budget exhausted'
                if active or (next_job < len(jobs) and len(retired) < slots and budget_stop is None):
                    time.sleep(.2 if active else 30)
        finally:
            cleanup = time.monotonic()
            try:
                for slot in list(active):
                    try:
                        finish(slot, 'Controller interrupted; no automatic retry')
                    finally:
                        self.failed = True
                if nv:
                    nv.close()
            finally:
                parent_active += time.monotonic()-cleanup
                self.ledger.charge('auxiliary', 'dispatch_monitor', parent_active, not failures and not budget_stop)
        self.failed |= bool(failures or budget_stop or next_job != len(jobs))
        write(self.root/'dispatch.json', dict(status='INCOMPLETE' if self.failed else 'COMPLETE', active={},
              retired=sorted(retired), failures=failures, pending=[j['target'].name for j in jobs[next_job:]], budget_stop=budget_stop))
        if self.failed:
            raise RuntimeError('Incomplete phase; healthy slots drained, downstream closed')

    def run(self, action: str, target: Path, bucket: str, seconds: int, extra: list[str]):
        self.run_many([dict(action=action, target=target, bucket=bucket, seconds=seconds, extra=extra)], 1, lambda job: None)


def pipeline(run_phase, save, choose=select_a, judge=decision):
    """The real controller and handwritten gate tests use the same dispatch flow."""
    a_rows = run_phase('A', None)
    a = choose(a_rows)
    save('A', a)
    if not a['passed']:
        return dict(status='STOP_A_NO_QUALIFIED_ENGLISH_START', decisions={'A': a})
    selected = a['selected']
    b_rows = run_phase('B', selected)
    b = judge(b_rows, selected, 'B')
    save('B', b)
    if not b['passed']:
        return dict(status='STOP_B_NO_JOINT_FORGETTING', selected=selected, decisions={'A': a, 'B': b})
    c_rows = run_phase('C', selected)
    c = judge(b_rows+c_rows, selected, 'C')
    save('C', c)
    if not c['passed']:
        return dict(status='STOP_C_FULL_MECHANISM_NOT_SUPPORTED', selected=selected, decisions={'A': a, 'B': b, 'C': c})
    d_rows = run_phase('D', selected)
    d = judge(d_rows, selected, 'D')
    save('D', d)
    return dict(status='CONFIRMATION_COMPLETE', strong_goal=d['passed'], selected=selected,
                decisions={'A': a, 'B': b, 'C': c, 'D': d})


def campaign(allowed: dict, gate_path: Path):
    import shutil
    from step30_evolution_evaluate import training_complete
    root = Path(allowed['campaign'])
    if root.exists():
        raise ValueError('Fresh independent campaign required; no automatic recovery')
    root.mkdir(parents=True)
    ledger = Ledger(root, allowed['prior_seconds'], allowed['cpu_test_seconds'])
    runner = Runner(allowed, gate_path, root, ledger)
    try:
        with ledger.active('prepare'):
            if allowed.get('waiter'):
                verify(allowed['waiter'])
            if shutil.disk_usage(root).free < LIMITS['free_disk_start']:
                raise RuntimeError('At least 128 GiB free at campaign start')
            manifest = dataset(allowed)
            gate_binding = record(gate_path)
        def run_phase(phase, selected):
            with ledger.active('freeze_'+phase):
                specs = tasks(phase, selected)
                target = root/phase
                target.mkdir()
                batch = dict(phase=phase, selected=selected, specs=specs, gate=gate_binding, completed=[])
                write(target/'batch.json', batch)
            jobs = []
            for task in specs:
                with ledger.active('prepare_'+task['id']):
                    job = target/task['id']
                    job.mkdir()
                    identity = dict(gate=gate_binding, spec=task, dataset=allowed['dataset_manifest'])
                    write(job/'spec.json', dict(task=task, identity=identity))
                jobs.append(dict(action='train', target=job, bucket='training',
                                 seconds=900 if task['stages'] == 1 else 3600, extra=['--batch', str(target/'batch.json')]))
            def completed(job):
                task = next(t for t in specs if t['id'] == job['target'].name)
                identity = read(job['target']/'spec.json')['identity']
                with ledger.active('validate_'+task['id']):
                    completion = job['target']/'run/completion.json'
                    training_complete(task, read(completion), identity)
                    batch['completed'].append(dict(task_id=task['id'], identity=identity, completion=record(completion)))
                    write(target/'batch.json', batch)
            runner.run_many(jobs, LIMITS['workers'], completed)
            # Only a fully complete prescribed phase may start any blind scoring.
            runner.run('score', target, 'gpu', CAPS['gpu']-ledger.used['gpu'], ['--batch', str(target/'batch.json')])
            analysis = target/'analysis'
            with ledger.active('prepare_analysis_'+phase):
                analysis.mkdir()
            runner.run('analyze', analysis, 'auxiliary', CAPS['auxiliary']-ledger.used['auxiliary']-60, ['--batch', str(target/'batch.json')])
            return read(target/'metrics.json')
        def choose(rows):
            with ledger.active('A_selection'):
                return select_a(rows)
        def judge(rows, selected, phase):
            with ledger.active(phase+'_decision'):
                budget = Budget(Path(allowed['experiment_root']), time.monotonic()+CAPS['auxiliary']-ledger.used['auxiliary'])
                return decision(rows, selected, phase, budget.check)
        result = pipeline(run_phase, lambda phase, result: write(root/phase/'decision.json', result), choose, judge)
        with ledger.active('complete'):
            write(root/'completion.json', result)
    except BaseException as exc:
        write(root/'failure.json', dict(error=repr(exc), automatic_retry=False, downstream_closed=True))
        raise


def configure_gpu():
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(LIMITS['allocator_bytes']/total, 0)


def worker(allowed: dict, gate_path: Path, action: str, target: Path, batch_path: Path, deadline: float):
    from step30_evolution_data import validate_group
    from step30_evolution_evaluate import analyze_batch, score_batch
    from step30_evolution_run import execute, make_model
    from step29_vendorlink_data import Group
    root = Path(allowed['campaign']).resolve()
    if not target.resolve().is_relative_to(root) or not batch_path.resolve().is_relative_to(root):
        raise ValueError('Worker outside this independent campaign')
    batch = read(batch_path)
    if batch['gate'] != record(gate_path) or batch['specs'] != tasks(batch['phase'], batch.get('selected')):
        raise ValueError('Batch gate/tasks differ')
    budget = Budget(Path(allowed['experiment_root']), deadline)
    budget.check()
    manifest = dataset(allowed)
    binding = allowed['model_binding']
    if action in ('train', 'score'):
        directory = Path(binding['directory']).resolve()
        if directory != Path('/home/yongpeng/cross-lingual/models/step29/bert_base_cased').resolve():
            raise ValueError('Use the already approved local English backbone')
        model_sources = {Path(r['path']).resolve() for r in binding['files']}
        actual = {p.resolve() for p in directory.iterdir() if p.is_file() and p.suffix in ('.json', '.txt', '.bin', '.safetensors')}
        if (model_sources != actual or not {'config.json', 'vocab.txt'} <= {p.name for p in actual}
                or not {'pytorch_model.bin', 'model.safetensors'} & {p.name for p in actual}):
            raise ValueError('Complete actual model/tokenizer file identities required')
        for source in binding['files']:
            verify(source)
        configure_gpu()
    factory = lambda task: make_model(Path(binding['directory']), task)
    if action == 'train':
        request = read(target/'spec.json')
        task, identity = request['task'], request['identity']
        if (task not in batch['specs'] or identity != dict(gate=batch['gate'], spec=task, dataset=allowed['dataset_manifest'])
                or target.name != task['id']):
            raise ValueError('Worker task/identity differs from frozen phase')
        model = factory(task)
        def load_fit(domain):
            row = manifest['files'][domain+':fit']
            access(target, 'fit_'+domain, [row])
            return [Group.from_dict(g) for g in pinned(row)]
        execute(model, load_fit, target/'run', task, identity, binding, write, record, budget.check)
    elif action == 'score':
        score_batch(batch_path.parent, batch, manifest, factory, binding, budget.check)
    elif action == 'analyze':
        analyze_batch(batch_path.parent, batch, manifest)
        budget.check()
    else:
        raise ValueError('Unknown worker action')


def build(allowed: dict):
    from step30_evolution_data import clean_archive, export_dataset
    target = Path(allowed['build_output'])
    if target.exists():
        raise ValueError('One new custodian build; no reuse/retry')
    accounting = target.parent/(target.name+'_accounting')
    if accounting.exists():
        raise ValueError('Build was already attempted')
    accounting.mkdir(parents=True)
    ledger = Ledger(accounting, allowed['prior_seconds'], allowed['cpu_test_seconds'])
    begin, ok = time.monotonic(), False
    experiment = Path(allowed['experiment_root']).resolve()
    if not target.resolve().is_relative_to(experiment):
        raise ValueError('New build output must be inside the experiment disk cap')
    budget = Budget(experiment, begin+CAPS['auxiliary']-ledger.used['auxiliary'], 8*2**30)
    try:
        qualification = qualification_directory(allowed)
        budget.check()
        access(accounting, 'official_four_table_build', [allowed['archive']])
        archive = verify(allowed['archive'])
        expected = pinned(allowed['qualification_summary'])['analysis']
        data, cleaning = clean_archive(archive, qualification, expected, budget.check)
        export_dataset(target, data, cleaning, write, record, budget.check)
        budget.check(force=True)
        ok = True
    finally:
        ledger.charge('auxiliary', 'official_build', time.monotonic()-begin, ok)


def adjudicate_language(allowed: dict):
    from step30_evolution_data import language_result
    target = Path(allowed['language_output'])
    if target.exists():
        raise ValueError('Language adjudication already attempted; no selective re-audit')
    target.mkdir(parents=True)
    ledger = Ledger(target, allowed['prior_seconds'], allowed['cpu_test_seconds'])
    with ledger.active('fixed_manual_language_count'):
        manifest = pinned(allowed['dataset_manifest'])
        if manifest.get('language_samples') != allowed['language_samples']:
            raise ValueError('Language sample must be bound by this exact dataset manifest')
        access(target, 'manual_fit_language_judgments', [allowed['language_samples'], allowed['language_decisions']])
        result = language_result(pinned(allowed['language_samples']), pinned(allowed['language_decisions']))
        write(target/'language_receipt.json', dict(**result, manifest=allowed['dataset_manifest'],
                                                  language_samples=allowed['language_samples']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--action', choices=('campaign', 'build', 'language', 'train', 'score', 'analyze'), required=True)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    action = 'campaign' if args.action in ('train', 'score', 'analyze') else args.action
    allowed = permit(args.gate, action)
    if args.action == 'campaign':
        campaign(allowed, args.gate)
    elif args.action == 'build':
        build(allowed)
    elif args.action == 'language':
        adjudicate_language(allowed)
    else:
        if args.target is None or args.batch is None or args.deadline is None:
            raise ValueError('Worker requires controller-provided target, batch and deadline')
        worker(allowed, args.gate, args.action, args.target, args.batch, args.deadline)


if __name__ == '__main__':
    main()
