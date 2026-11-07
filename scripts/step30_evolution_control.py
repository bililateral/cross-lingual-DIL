"""One finite Evolution worker, A -> B -> C -> D, no listener or retry."""
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
                                  permit, pinned, read, record, sha, verify, write)
from step30_evolution_policy import CAPS, DOMAINS, LIMITS, decision, select_a, tasks


def resource_observation(allowed: dict, root: Path, gpu: bool) -> dict:
    import shutil
    observation = dict(ram_available=psutil.virtual_memory().available,
                       disk_free=shutil.disk_usage(root).free, experiment_bytes=directory_bytes(root))
    observation['ready'] = (observation['ram_available'] >= LIMITS['free_ram_bytes'] and
                            observation['disk_free'] >= LIMITS['free_disk_reserve']+2**30 and
                            observation['experiment_bytes']+2**30 <= LIMITS['disk_bytes'])
    if gpu:
        import pynvml as nv
        nv.nvmlInit()
        try:
            device = nv.nvmlDeviceGetHandleByIndex(0)
            memory = nv.nvmlDeviceGetMemoryInfo(device)
            used = {p.pid: p.usedGpuMemory for p in nv.nvmlDeviceGetComputeRunningProcesses(device)}
            unreflected, commitments = 0, []
            for row in allowed['other_gpu_commitments']:
                try:
                    process = psutil.Process(row['pid'])
                    if process.create_time() != row['created'] or process.username() != row['owner']:
                        raise ValueError('Declared active commitment process identity changed')
                    remaining = max(0, row['reserved_bytes']-used.get(row['pid'], 0))
                    unreflected += remaining
                    commitments.append(dict(**row, unreflected_bytes=remaining))
                except psutil.NoSuchProcess:
                    commitments.append(dict(**row, exited=True))
            observation.update(gpu_free=memory.free, unreflected_commitments=unreflected, commitments=commitments)
            observation['ready'] &= memory.free-unreflected >= LIMITS['free_gpu_bytes']
        finally:
            nv.nvmlShutdown()
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

    def run(self, action: str, target: Path, bucket: str, seconds: int, extra: list[str]):
        if self.failed:
            raise RuntimeError('Failed single worker cannot dispatch another operation')
        with self.ledger.active('admission_'+target.name):
            self.ledger.reserve(bucket, seconds)
            observation = resource_observation(self.allowed, self.root, bucket in ('training', 'gpu'))
            write(target/'admission.json', observation)
            if not observation['ready']:
                raise RuntimeError('Resource gate closed; no automatic waiting/listener')
            cores = psutil.Process().cpu_affinity()
            if len(cores) < 2:
                raise RuntimeError('Two allowed CPU cores required')
        start, process, ok, parent_active = time.monotonic(), None, False, 0.
        deadline = start+seconds
        argv = ['taskset', '-c', ','.join(map(str, cores[-2:])), PYTHON, '-B', '-u', str(Path(__file__).resolve()),
                '--gate', str(self.gate_path), '--action', action, '--target', str(target), '--deadline', str(deadline), *extra]
        environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                           TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='0' if bucket in ('training', 'gpu') else '')
        error = None
        try:
            with (target/'console.txt').open('x') as output:
                process = subprocess.Popen(argv, env=environment, cwd=self.allowed['workspace'],
                                           stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                p = psutil.Process(process.pid)
                write(target/'process.json', dict(pid=p.pid, owner=p.username(), created=p.create_time(), command=argv, cwd=p.cwd()))
                while process.poll() is None:
                    active_start = time.monotonic()
                    try:
                        if time.monotonic() >= deadline-5:
                            raise TimeoutError('Full worker lifecycle deadline')
                        if p.memory_info().rss > LIMITS['rss_bytes']:
                            raise RuntimeError('Worker RAM limit')
                        if directory_bytes(self.root) > LIMITS['disk_bytes']:
                            raise RuntimeError('Experiment disk limit')
                    finally:
                        # Parent monitoring is auxiliary active work; pure sleep
                        # is excluded and worker lifecycle is charged separately.
                        parent_active += time.monotonic()-active_start
                    time.sleep(.5)
                if process.returncode:
                    raise RuntimeError(f'Worker exit {process.returncode}')
                ok = True
        except BaseException as exc:
            error, self.failed = repr(exc), True
            raise
        finally:
            try:
                if process is not None:
                    release(process)
            except BaseException as exc:
                error, ok, self.failed = repr(exc), False, True
                raise
            finally:
                elapsed = time.monotonic()-start
                if elapsed > seconds:
                    ok, self.failed = False, True
                    error = error or 'Worker lifecycle overrun'
                write(target/'dispatch_receipt.json', dict(success=ok, error=error, seconds=elapsed, released=True))
                self.ledger.charge(bucket, target.name, elapsed, ok)
                self.ledger.charge('auxiliary', 'monitor_'+target.name, parent_active, ok)
        if not ok:
            raise RuntimeError(error or 'Worker failed; no retry')


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
            for task in specs:
                with ledger.active('prepare_'+task['id']):
                    job = target/task['id']
                    job.mkdir()
                    identity = dict(gate=gate_binding, spec=task, dataset=allowed['dataset_manifest'])
                    write(job/'spec.json', dict(task=task, identity=identity))
                runner.run('train', job, 'training', 900 if task['stages'] == 1 else 3600, ['--batch', str(target/'batch.json')])
                with ledger.active('validate_'+task['id']):
                    completion = job/'run/completion.json'
                    training_complete(task, read(completion), identity)
                    batch['completed'].append(dict(task_id=task['id'], identity=identity, completion=record(completion)))
                    write(target/'batch.json', batch)
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
                budget = Budget(root, time.monotonic()+CAPS['auxiliary']-ledger.used['auxiliary'])
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
    budget = Budget(root, deadline)
    budget.check()
    manifest = dataset(allowed)
    binding = allowed['model_binding']
    if action in ('train', 'score'):
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
    budget = Budget(accounting, begin+CAPS['auxiliary']-ledger.used['auxiliary'], 8*2**30)
    try:
        access(accounting, 'official_four_table_build', [allowed['archive']])
        archive = verify(allowed['archive'])
        expected = pinned(allowed['qualification_summary'])['analysis']
        for row in allowed['qualification_sources']:
            verify(row)
        data, cleaning = clean_archive(archive, Path(allowed['qualification_directory']), expected, budget.check)
        export_dataset(target, data, cleaning, write, record)
        ok = True
    finally:
        ledger.charge('auxiliary', 'official_build', time.monotonic()-begin, ok)


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
        from step30_evolution_data import language_result
        target = Path(allowed['language_output'])
        if target.exists():
            raise ValueError('Language adjudication already attempted; no selective re-audit')
        target.mkdir(parents=True)
        ledger = Ledger(target, allowed['prior_seconds'], allowed['cpu_test_seconds'])
        with ledger.active('fixed_manual_language_count'):
            access(target, 'manual_fit_language_judgments', [allowed['language_samples'], allowed['language_decisions']])
            result = language_result(pinned(allowed['language_samples']), pinned(allowed['language_decisions']))
            write(target/'language_receipt.json', dict(**result, manifest=allowed['dataset_manifest']))
    else:
        if args.target is None or args.batch is None or args.deadline is None:
            raise ValueError('Worker requires controller-provided target, batch and deadline')
        worker(allowed, args.gate, args.action, args.target, args.batch, args.deadline)


if __name__ == '__main__':
    main()
