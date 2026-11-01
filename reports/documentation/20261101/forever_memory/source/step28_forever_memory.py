"""One approved FOREVER memory measurement; no formal data or queue dispatch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import psutil

GIB = 2**30
SECONDS = 300
PRIOR_NATIVE_SECONDS = 1274


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*2**20), b''):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def gpu() -> dict:
    free, total = [int(x.strip())*2**20 for x in subprocess.check_output(
        ['nvidia-smi', '--id=0', '--query-gpu=memory.free,memory.total',
         '--format=csv,noheader,nounits'], text=True, timeout=2).strip().split(',')]
    apps = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
        text=True, timeout=2)
    return dict(epoch=time.time(), free_bytes=free, total_bytes=total,
                processes={int(p): int(v.strip())*2**20 for line in apps.splitlines()
                           if line.strip() for p, v in [line.split(',')]})


def worker(args) -> None:
    sys.path[:0] = [str(args.workspace/'scripts'), str(args.workspace/'tests')]
    import torch
    import step28_forever as m
    import step28_forever_run as run
    from step28_comparison_native import group
    from test_step28_forever import dispatcher

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(8.5*GIB/torch.cuda.get_device_properties(0).total_memory)
    phases, updates = [], []

    def check(reserve=0):
        if time.monotonic()+reserve >= args.deadline-10:
            raise TimeoutError('Approved memory measurement time exhausted')
        if psutil.Process().memory_info().rss > 64*GIB:
            raise RuntimeError('Worker RSS limit exceeded')

    def timed(name, action):
        check()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        write(args.job/'phase.json', dict(phase=name, epoch=time.time()))
        value = action()
        torch.cuda.synchronize()
        phases.append(dict(phase=name, seconds=time.monotonic()-tick,
            allocated_peak_bytes=torch.cuda.max_memory_allocated(),
            reserved_peak_bytes=torch.cuda.max_memory_reserved(),
            allocated_end_bytes=torch.cuda.memory_allocated(),
            reserved_end_bytes=torch.cuda.memory_reserved()))
        write(args.job/'phases.json', dict(phases=phases))
        return value

    spec = m.Spec('ABC', 4)
    config = m.config(spec)
    archive = m.core.model_files(m.record.base.model_config(config, 'split_rank'))
    if any(archive[k] != config['models']['split_rank'][k]
           for k in ('file_count', 'total_size_bytes', 'content_sha256')):
        raise ValueError('Pinned BGE identity differs')
    model, optimizer = timed('load', lambda: m.fresh(spec, config))
    if not torch.cuda.is_bf16_supported() or not model.encoder[0].auto_model.is_gradient_checkpointing:
        raise ValueError('BF16 and production gradient checkpointing required')
    learner = m.Learner(spec, model, optimizer, config)
    current = group('forever_current', 0)
    history = [group(f'forever_old_{i}', 48+i) for i in range(6)]
    for value in [current, *history]:
        tokens = model.encoder.tokenizer(m.record.base.record_texts(value, 'separate_moments'),
                                        padding=False, truncation=False)['input_ids']
        if len(tokens) != 448 or set(map(len, tokens)) != {256}:
            raise ValueError('Maximum handwritten input shape differs')

    def update(value, role, beta=None):
        row = learner.update(value, role, ('memory_measurement',) if beta else (), beta, check)
        updates.append(row)
        write(args.job/'updates.json', dict(updates=updates))
        return row

    timed('prepare_update', lambda: update(history[0], 'fixture'))
    learner.anchor = timed('anchor_snapshot', lambda: m.snapshot(model))
    learner.old.add_stage(history)
    learner.old.seen = 48  # Handwritten stage-2 fixture, not 48 claimed training updates.
    learner.stage = 2
    timed('current_update', lambda: update(current, 'current'))
    timed('replay_update', lambda: update(history[0], 'replay', .001))
    if (updates[1]['displacement']['encoder'] <= 0 or updates[1]['displacement']['head'] <= 0
            or updates[2]['regularizer']['penalty'] <= 0):
        raise ValueError('Actual current update or anchor replay is missing')
    supply = dispatcher(spec)
    supply.supply.next_stage[spec.run_id] = 2
    supply.current(2)
    store = run.Store(args.job/'state', dict(spec=spec.identity(), config=config,
                                           memory_measurement_only=True), check)
    timed('save', lambda: store.commit(learner, supply))
    timed('restore', lambda: store.restore(learner, dispatcher(spec)))
    if m.parent.adam_step(optimizer) != 3 or learner.counts != dict(current=1, replay=1, fixture=1):
        raise ValueError('Actual update/restoration counts differ')
    write(args.job/'result.json', dict(status='MEASUREMENT_COMPLETE', phases=phases,
        spec=spec.identity(), actual_adam_step=3, counts=learner.counts,
        maximum_group=dict(records=224, channels=2, tokens=256),
        allocator_limit_bytes=int(8.5*GIB), anchor_bytes=sum(t.numel()*t.element_size()
        for t in learner.anchor.values()), historical_bytes=learner.historical_bytes(),
        parameter_count=sum(p.numel() for p in m.parameters(model, optimizer).values()),
        formal_data_reads=0, scope='one worker; fixed beta replay kernel; no clock or full-stage qualification'))


def supervise(args) -> None:
    args.job.mkdir()  # Never overwrite or silently retry an attempt.
    tick = time.monotonic()
    process, error, samples = None, None, []
    try:
        gate = json.loads(args.gate.read_text())
        if gate['experiment'] != 'FOREVER' or Path(gate['workspace']).resolve() != args.workspace:
            raise ValueError('FOREVER frozen workspace differs')
        for expected in gate['sources']:
            actual = record(Path(expected['path']))
            if any(actual[k] != expected[k] for k in ('bytes', 'sha256')):
                raise ValueError('Frozen source mismatch: '+expected['path'])
        observation = gpu()
        write(args.job/'admission.json', dict(**observation,
            ram_available_bytes=psutil.virtual_memory().available,
            disk_free_bytes=shutil.disk_usage(args.job).free, sources=gate['sources'],
            entry=record(Path(__file__)), gate=record(args.gate)))
        if (observation['free_bytes'] < 10*GIB or psutil.virtual_memory().available < 64*GIB
                or shutil.disk_usage(args.job).free < 96*GIB):
            raise RuntimeError('Insufficient actual resources; no wait or retry')
        argv = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                '--job', str(args.job), '--workspace', str(args.workspace),
                '--deadline', str(tick+SECONDS)]
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'CUDA_VISIBLE_DEVICES': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        with (args.job/'console.txt').open('x', encoding='utf-8') as out:
            process = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env,
                                       cwd=args.workspace, start_new_session=True)
            p = psutil.Process(process.pid)
            write(args.job/'process.json', dict(pid=p.pid, created=p.create_time(),
                owner=p.username(), cwd=str(args.workspace), command=argv))
            while process.poll() is None:
                if time.monotonic() >= tick+SECONDS-15:
                    raise TimeoutError('Five-minute measurement limit including shutdown reserve')
                path = args.job/'phase.json'
                before = json.loads(path.read_text())['phase'] if path.exists() else 'startup'
                sample = gpu()
                after = json.loads(path.read_text())['phase'] if path.exists() else 'startup'
                sample.update(phase=before if before == after else 'transition',
                              worker_bytes=sample['processes'].get(process.pid, 0))
                samples.append(sample)
                with (args.job/'gpu_samples.jsonl').open('a', encoding='utf-8') as out:
                    out.write(json.dumps(sample)+'\n')
                try:
                    if p.memory_info().rss > 64*GIB or shutil.disk_usage(args.job).free < 64*GIB:
                        raise RuntimeError('Worker RAM or shared disk reserve exceeded')
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.5)
        if process.returncode != 0 or not (args.job/'result.json').exists():
            raise RuntimeError('Measurement worker failed; no automatic retry')
    except BaseException:
        error = traceback.format_exc()
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
        elapsed = time.monotonic()-tick
        write(args.job/'receipt.json', dict(status='FAILED' if error else 'MEASURED', error=error,
            returncode=process.returncode if process else None, wall_seconds=elapsed,
            charged_seconds=math.ceil(elapsed), prior_native_seconds=PRIOR_NATIVE_SECONDS,
            cumulative_native_seconds=PRIOR_NATIVE_SECONDS+math.ceil(elapsed), native_limit_seconds=3600,
            sampled_process_peak_bytes=max((s['worker_bytes'] for s in samples), default=0),
            sampling_sleep_seconds=.5, samples=len(samples), entry=record(Path(__file__)),
            five_minute_limit_met=elapsed <= SECONDS, automatic_retry=False))
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Use existing Linux py310 only')
    args.workspace, args.job = args.workspace.resolve(), args.job.resolve()
    if args.worker:
        try:
            worker(args)
        except BaseException:
            write(args.job/'worker_failure.json', dict(error=traceback.format_exc()))
            raise
    else:
        supervise(args)
