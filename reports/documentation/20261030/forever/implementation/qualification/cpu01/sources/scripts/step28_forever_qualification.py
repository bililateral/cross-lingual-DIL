"""Bounded handwritten qualification supervisor. No formal data or train action."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

GIB = 2**30


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def verify(root: Path, permit: dict) -> None:
    if (permit['status'] != 'APPROVED_MINIMAL_IMPLEMENTATION_CPU_GPU_ONLY'
            or permit['cpu_seconds'] != 600 or permit['gpu_seconds'] != 3600
            or permit['optimizer_calls'] != 48 or permit['formal_training'] is not False
            or permit['formal_data'] is not False):
        raise ValueError('Qualification scope differs')
    for row in permit['sources']:
        raw = (root/row['path']).read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError(f'Deployed source differs: {row["path"]}')


def resources() -> dict:
    import shutil
    values = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free,memory.total',
                                      '--format=csv,noheader,nounits'], text=True)
    free, total = [int(v)*2**20 for v in values.strip().split(',')]
    return dict(epoch=time.time(), free_gpu_bytes=free, total_gpu_bytes=total,
                ram_available_bytes=psutil.virtual_memory().available,
                disk_free_bytes=shutil.disk_usage(Path.cwd()).free)


def predecessor_ready(root: Path, live_commands: list[str]) -> bool:
    """VendorLink G1 plus all G2 workers must finish successfully and exit first."""
    paths = [root/'ledger.json', root/'g1/result.json', *[root/f'g2_{i}/result.json' for i in range(3)]]
    if not all(p.is_file() for p in paths):
        return False
    ledger, first, *workers = [json.loads(p.read_text()) for p in paths]
    return (ledger.get('status') == 'NATIVE_QUALIFIED_NOT_FORMAL_TRAINING'
            and first.get('status') == 'G1_PASSED'
            and all(r.get('status') == 'G2_WORKER_PASSED' for r in workers)
            and not any('vendorlink' in cmd and ('native' in cmd or 'step29_linux_20261029' in cmd)
                        for cmd in live_commands))


def supervise(args) -> None:
    root = Path(__file__).resolve().parents[1]
    permit = json.loads(args.permit.read_text())
    verify(root, permit)
    if args.job.exists():
        raise FileExistsError('Never overwrite/restart an existing qualification attempt')
    args.job.mkdir(parents=True)
    # The ledger is external to recoverable learner checkpoints and never rolls back.
    ledger_path = args.permit.parent/'qualification_accounting.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(cpu_seconds=0, gpu_seconds=0, optimizer_calls=0, attempts=[])
    if args.mode == 'gpu':
        live = []
        for p in psutil.process_iter(['cmdline']):
            try:
                live.append(' '.join(p.info['cmdline'] or []))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if not predecessor_ready(Path(permit['vendorlink_predecessor']), live):
            write(args.job/'status.json', dict(status='WAITING_VENDORLINK_NO_WORKER_STARTED',
                  reason='User requires successful complete VendorLink native qualification before FOREVER GPU'))
            return
        if not permit.get('implementation_review_closed'):
            write(args.job/'status.json', dict(status='WAITING_IMPLEMENTATION_REVIEW_NO_WORKER_STARTED'))
            return
        resource = resources()
        write(args.job/'admission.json', resource)
        if (resource['free_gpu_bytes'] < 30*GIB or resource['ram_available_bytes'] < 192*GIB
                or resource['disk_free_bytes'] < 160*GIB):
            write(args.job/'status.json', dict(status='WAITING_RESOURCES_NO_WORKER_STARTED',
                  reason='Three-slot qualification reserves 10 GiB per pending slot, 192 GiB RSS and 96+64 GiB disk',
                  formal_training=False))
            return
    bucket = 'cpu_seconds' if args.mode == 'cpu' else 'gpu_seconds'
    cap = permit[bucket]-ledger[bucket]
    if cap <= 0:
        raise RuntimeError('Qualification bucket exhausted')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    children, logs, started = [], [], time.monotonic()
    if args.mode == 'cpu':
        env.update(CUDA_VISIBLE_DEVICES='', FOREVER_TEST_TMP=str(args.job.resolve()))
        commands = [[sys.executable, '-B', str(root/'tests/test_step28_forever.py')]]
    else:
        env['CUDA_VISIBLE_DEVICES'] = '0'
        commands = [[sys.executable, '-B', str(Path(__file__).resolve()), '--mode', 'worker',
                     '--job', str(args.job), '--slot', str(i), '--deadline', str(time.monotonic()+cap),
                     '--permit', str(args.permit)] for i in range(3)]
    peaks = dict(rss_bytes=0, disk_bytes=0)
    stopped = None
    try:
        def launch(index):
            command = commands[index]
            stream = (args.job/f'console{index}.txt').open('w')
            logs.append(stream)
            child = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            children.append(dict(process=child, started=time.monotonic(), ended=None, slot=index))
        launch(0)
        while any(row['process'].poll() is None for row in children):
            if args.mode == 'gpu' and len(children) == 1 and (args.job/'main_ready.json').exists():
                launch(1)
                launch(2)
            elapsed, rss = 0., 0
            for row in children:
                p = row['process']
                if p.poll() is not None and row['ended'] is None:
                    row['ended'] = time.monotonic()
                elapsed += (row['ended'] or time.monotonic())-row['started']
                if p.poll() is None:
                    try:
                        process = psutil.Process(p.pid)
                        resident = sum(v.memory_info().rss for v in [process, *process.children(recursive=True)])
                        rss += resident
                        if resident > (4 if args.mode == 'cpu' else 64)*GIB:
                            stopped = 'RSS_CAP'
                    except psutil.NoSuchProcess:
                        pass
                elif p.returncode != 0:
                    stopped = 'CHILD_FAILURE'
            files = [p for p in args.job.rglob('*') if p.is_file()]
            disk = sum(p.stat().st_size for p in files)
            peaks['rss_bytes'], peaks['disk_bytes'] = max(peaks['rss_bytes'], rss), max(peaks['disk_bytes'], disk)
            calls = sum(len(p.read_text().splitlines()) for p in args.job.glob('slot*/calls.jsonl'))
            if elapsed >= cap-3 or calls+ledger['optimizer_calls'] > 48:
                stopped = 'TIME_OR_CALL_CAP'
            if disk > (64*2**20 if args.mode == 'cpu' else 96*GIB):
                stopped = 'DISK_CAP'
            if stopped:
                for row in children:
                    if row['process'].poll() is None:
                        os.killpg(row['process'].pid, signal.SIGTERM)
                break
            time.sleep(.2)
    finally:
        for row in children:
            try:
                row['process'].wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(row['process'].pid, signal.SIGKILL)
                row['process'].wait()
            row['ended'] = row['ended'] or time.monotonic()
        for stream in logs:
            stream.close()
        charged = math.ceil(sum(row['ended']-row['started'] for row in children))
        calls = sum(len(p.read_text().splitlines()) for p in args.job.glob('slot*/calls.jsonl'))
        ledger[bucket] += charged
        ledger['optimizer_calls'] += calls
        receipt = dict(mode=args.mode, job=str(args.job), charged_seconds=charged, optimizer_calls=calls,
                       children=[dict(slot=r['slot'], pid=r['process'].pid, exit_code=r['process'].returncode,
                                      elapsed_seconds=r['ended']-r['started']) for r in children],
                       status='PASS' if stopped is None and all(r['process'].returncode == 0 for r in children) else 'STOP',
                       reason=stopped, peaks=peaks, wall_seconds=time.monotonic()-started,
                       formal_data_reads=0, formal_training=False)
        ledger['attempts'].append(receipt)
        write(ledger_path, ledger)
        write(args.job/'receipt.json', receipt)
    if receipt['status'] != 'PASS':
        raise RuntimeError('Bounded qualification stopped; failures remain charged')


def worker(args) -> None:
    # Imported only inside explicitly authorized Linux GPU workers.
    import torch
    import step28_forever as m
    import step28_forever_run as run
    from step28_comparison_native import group
    from test_step28_forever import dispatcher
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(8.5*GIB/torch.cuda.get_device_properties(0).total_memory)
    job = args.job/f'slot{args.slot}'
    job.mkdir()
    allowed_calls = 39 if args.slot == 0 else 3
    calls, rows = [], []
    def check(reserve=0):
        if time.monotonic() >= args.deadline-5:
            raise TimeoutError('Qualification deadline')
        if psutil.Process().memory_info().rss > 64*GIB:
            raise RuntimeError('Worker RSS cap')
    def physical():
        import fcntl
        if len(calls) >= allowed_calls:
            raise RuntimeError('Fixed per-slot physical update allocation exhausted')
        with (args.job/'calls.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger_path = args.permit.parent/'qualification_accounting.json'
            prior = json.loads(ledger_path.read_text()).get('optimizer_calls', 0) if ledger_path.exists() else 0
            total = sum(len(p.read_text().splitlines()) for p in args.job.glob('slot*/calls.jsonl'))
            if prior+total >= 48:
                raise RuntimeError('Shared physical optimizer call allowance exhausted')
            calls.append(dict(attempt=len(calls)+1, epoch=time.time()))
            with (job/'calls.jsonl').open('a') as stream:
                stream.write(json.dumps(calls[-1])+'\n')
    def log(row):
        rows.append(row)
        with (job/'updates.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(epoch=time.time(), **row))+'\n')
    def timed(label, fn):
        check()
        torch.cuda.synchronize()
        begin = time.monotonic()
        write(job/'phase.json', dict(phase=label, epoch=time.time()))
        value = fn()
        torch.cuda.synchronize()
        timings[label] = time.monotonic()-begin
        return value
    timings = {}
    spec = m.Spec('ABC', 4)
    c = m.config(spec)
    # Pretrained identity is checked before loading; only the existing archive is used.
    archive = m.core.model_files(m.record.base.model_config(c, 'split_rank'))
    if any(archive[k] != c['models']['split_rank'][k] for k in ('file_count', 'total_size_bytes', 'content_sha256')):
        raise ValueError('Pinned BGE archive differs')
    model, optimizer = timed('load', lambda: m.fresh(spec, c))
    learner = m.Learner(spec, model, optimizer, c)
    current = group('forever_current', 0)
    old = [group(f'forever_old_{i}', 48+i) for i in range(6)]
    for g in [current, *old]:
        ids = model.encoder.tokenizer(m.record.base.record_texts(g, 'separate_moments'),
                                     padding=False, truncation=False)['input_ids']
        if len(ids) != 448 or set(map(len, ids)) != {256}:
            raise ValueError('Actual maximum 224-record/two-channel/256-token shape absent')
    timed('fixture', lambda: log(learner.update(old[0], 'fixture', check=check, before_step=physical)))
    learner.anchor = m.snapshot(model)
    learner.old.add_stage(old)
    learner.old.seen = 48  # Explicit qualification-only stage-2 reservoir fixture, no invented Adam steps.
    learner.stage = 2
    supply = dispatcher(spec)  # Handwritten tiny dispatcher only, never the formal archive.
    supply.supply.next_stage[spec.run_id] = 2
    supply.current(2)
    identity = dict(spec=spec.identity(), config=c, qualification_only=True,
                    sources=json.loads(args.permit.read_text())['sources'])
    store = run.Store(job/'state', identity, check)
    if args.slot == 0:
        for i in range(23):
            timed(f'current{i+1}', lambda: learner.advance(current, check, physical, log))
        write(args.job/'main_ready.json', dict(epoch=time.time()))
        while not all((args.job/f'aux_ready{i}.json').exists() for i in (1, 2)):
            check()
            time.sleep(.1)
        write(args.job/'compute_start.json', dict(epoch=time.time()))
        timed('current24_and_replay', lambda: learner.advance(current, check, physical, log,
                                                             lambda: store.commit(learner, supply)))
        if learner.clock.next_threshold != 1 or learner.counts != dict(current=24, replay=12, fixture=1):
            raise ValueError('Full production calibration and first two replay epochs missing')
        timed('full_commit', lambda: store.commit(learner, supply))
        timed('uninterrupted_next', lambda: log(learner.update(current, 'current', check=check, before_step=physical)))
        expected = m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                           state=learner.small(), rng=m.diag.rng_state()))
        # Restore in place, without a second GPU model or fabricated optimizer counter.
        supply = dispatcher(spec)
        timed('full_restore', lambda: store.restore(learner, supply))
        timed('restored_next', lambda: log(learner.update(current, 'current', check=check, before_step=physical)))
        actual = m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                         state=learner.small(), rng=m.diag.rng_state()))
        if actual != expected:
            raise ValueError('Actual next BGE model/Adam/clock/cursor/RNG differ after restoration')
        write(args.job/'main_done.json', dict(epoch=time.time()))
    else:
        write(args.job/f'aux_ready{args.slot}.json', dict(epoch=time.time()))
        while not (args.job/'compute_start.json').exists():
            check()
            time.sleep(.1)
        timed('overlap_current', lambda: log(learner.update(current, 'current', check=check, before_step=physical)))
        # Auxiliary replay tests the same production kernel at fixed beta; no calibrated-clock claim.
        timed('overlap_replay', lambda: log(learner.update(old[0], 'replay', ('aux',), .001, check, physical)))
        while not (args.job/'main_done.json').exists():
            check()
            time.sleep(.1)  # Deliberate residency through main save/restore; this wall time is charged.
    write(job/'result.json', dict(status='PASS_BOUNDED_NATIVE_SLOT', slot=args.slot, qualification_only=True,
          fixture_predecessor_updates=1, physical_calls=len(calls), logical_counts=learner.counts,
          actual_adam_step=m.parent.adam_step(optimizer), timings=timings, parameters=m.mapping(model, optimizer),
          anchor_bytes=sum(t.numel()*t.element_size() for t in learner.anchor.values()),
          history_bytes=learner.historical_bytes(), maximum_group=dict(records=224, channels=2, tokens=256),
          peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
          formal_data_reads=0, untested='Remaining production events, full stages, formal data and end calibration'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('cpu', 'gpu', 'worker'), required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--permit', type=Path, required=True)
    parser.add_argument('--slot', type=int, choices=(0, 1, 2))
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.mode == 'worker':
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
        verify(Path(__file__).resolve().parents[1], json.loads(args.permit.read_text()))
        worker(args)
    else:
        supervise(args)
