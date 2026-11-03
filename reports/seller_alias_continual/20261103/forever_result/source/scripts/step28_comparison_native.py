"""Bounded BGE qualification on handwritten text, never a formal trajectory."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
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
LIMITS = dict(rss_bytes=32*GIB, disk_bytes=12*GIB, small_bytes=256*2**20,
              free_reserve_bytes=64*GIB, allocator_bytes=int(7.75*GIB))
NATIVE_SECONDS = 1800
METHODS = ('ER_GCR_group_fixed', 'L2R_text_fixed')


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def transient_snapshot(stage) -> dict:
    return copy.deepcopy(stage.transient())


def resources() -> dict:
    fields = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    free, total = [int(x.strip())*2**20 for x in fields.split(',')]
    return dict(epoch=time.time(), free_gpu_bytes=free, total_gpu_bytes=total, compute_apps=apps,
                available_host_bytes=psutil.virtual_memory().available,
                available_disk_bytes=shutil.disk_usage(Path.cwd()).free)


def ready(row: dict) -> bool:
    return (row['free_gpu_bytes'] >= int(8.5*GIB)
            and row['available_host_bytes'] >= 40*GIB and row['available_disk_bytes'] >= 80*GIB)


def group(uid: str, offset: int = 0):
    import step28_continual_comparison as m
    owners = [i for i, size in enumerate([3]*4+[2]*8) for _ in range(size)]
    labels = tuple(int(owners[i] == owners[j]) for i, j in itertools.combinations(range(28), 2))
    def text(i, j, channel):
        # Each punctuation mark is a token; short UTF-8 keeps full paid history below 1 MiB.
        prefix = '商品'+chr(0x4e00+(i+offset) % 128)+chr(0x4e80+j)+chr(0x4e90+channel)
        return prefix+'!'*(254-len(prefix))
    value = m.data.Group(uid, tuple(f'{uid}_s{i:02}' for i in range(28)),
        tuple(tuple((f'{uid}_r{i:02}_{j}', text(i,j,0), text(i,j,1)) for j in range(8))
              for i in range(28)), labels)
    value.validate()
    return value


def worker(job: Path, name: str, deadline: float) -> None:
    import numpy as np
    import torch
    import step28_continual_comparison as m
    import step28_continual_comparison_run as run
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(LIMITS['allocator_bytes']/torch.cuda.get_device_properties(0).total_memory)
    budget = run.Budget(job, deadline, LIMITS)
    spec = m.Spec(name, 'ABC', 4)
    c, method = m.config(spec), m.Method(spec)
    timings, calls = {}, []
    def timed(label, action):
        budget.check()
        write(job/'progress.json', dict(phase=label, status='STARTED', timings=timings, calls=calls))
        torch.cuda.synchronize()
        tick = time.monotonic()
        result = action()
        torch.cuda.synchronize()
        timings[label] = time.monotonic()-tick
        write(job/'progress.json', dict(phase=label, status='COMPLETE', timings=timings, calls=calls))
        return result
    archive = m.core.model_files(m.base.model_config(c, 'split_rank'))
    if any(archive[k] != c['models']['split_rank'][k] for k in
           ('file_count', 'total_size_bytes', 'content_sha256')):
        raise ValueError('Pinned BGE archive differs')
    model, optimizer = timed('load', lambda: method.fresh(c))
    if not model.encoder[0].auto_model.is_gradient_checkpointing or not torch.cuda.is_bf16_supported():
        raise ValueError('Required native BF16/checkpointing absent')
    current = [group(f'native_current_{i:02}', i) for i in range(48)]
    history = [group(f'native_history_{i:02}', i+48) for i in range(6)]
    calibration = [group(f'native_cal_{i:02}', i+64) for i in range(12)]
    for g in current+history+calibration:
        ids = model.encoder.tokenizer(m.base.record_texts(g, 'separate_moments'),
                                     padding=False, truncation=False)['input_ids']
        if len(ids) != 448 or set(map(len, ids)) != {256}:
            raise ValueError('Full 224-record, two-channel, 256-token fixture absent')
    def update(stage):
        if len(calls) >= 3:
            raise ValueError('Three actual update attempts exhausted')
        calls.append(dict(stage=stage.stage, step=1, status='STARTED'))
        value = method.update(model, optimizer, stage, current[0], c, 1, budget.check)
        if value['gradient_norm'] <= 0:
            raise ValueError('Native update has no gradient')
        calls[-1].update(status='COMPLETE', adam_step=m.parent.adam_step(optimizer), value=value)
        return value
    initial = m.core.state_digest(model.encoder.state_dict())
    stage1 = method.stage(1, current, method.memory(), model)
    timed('current_update', lambda: update(stage1))
    if initial == m.core.state_digest(model.encoder.state_dict()):
        raise ValueError('Actual BGE encoder parameters did not change')
    memory = method.memory()
    memory.stage = 1
    if method.algorithm is m.gcr:
        memory.seen = 48
        memory.groups = {g.uid: g for g in history}
        memory.weights = {g.uid: 1.25+.25*i for i,g in enumerate(history)}
    else:
        memory.seen = 1344
        rows = [a for g in history for a in m.l2r.accounts(g, 'A')]
        for q in rows[:14]:
            positive = next(a for a in rows if a.uid != q.uid and m.l2r.same_identity(a,q))
            negative = [a for a in rows if not m.l2r.same_identity(a,q)][:10]
            memory.packages[q.uid] = dict(query=q, positive=positive, negatives=negative)
    payload = memory.to_bytes()
    if memory.summary()['slots'] != 168:
        raise ValueError('Full 168 paid historical slots absent')
    memory = method.memory_from(payload)
    stage = method.stage(2, current, memory, model)
    # Explicit synthetic boundary, not a claim to have performed 288 prior updates.
    for state in optimizer.state.values():
        state['step'].fill_(288)
    for directory in ('work','models','memory','maps','scores','points'):
        (job/directory).mkdir()
    metadata = {**method.metadata(), 'rng': m.diag.rng_state(),
                'modes': [v.training for v in model.modules()],
                'fixture': 'handwritten Adam counter 288 and historical memory, not a trained endpoint'}
    before_path = job/'work/before.pt'
    before = timed('save_before', lambda: run.save(before_path, model, optimizer, metadata, budget))
    transient = transient_snapshot(stage)
    first = timed('history_update', lambda: update(stage))
    if method.algorithm is m.gcr:
        if first.get('replay',{}).get('pool') != 'history_pool' or first['replay']['absolute_weight'] == 1:
            raise ValueError('Nonunit historical replay was not exercised')
    elif first.get('history',{}).get('queries') != 14 or first['current']['positive_items'] != 40:
        raise ValueError('Full current/history L2R update absent')
    digest = m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
    rng = m.diag.rng_state()
    after_transient = transient_snapshot(stage)
    timed('restore_before', lambda: method.restore(before_path, before, model, optimizer))
    stage.restore_transient(transient)
    second = timed('history_repeat', lambda: update(stage))
    equality = dict(loss_logs=first == second, rng=rng == m.diag.rng_state(),
                    transient=after_transient == stage.transient(),
                    model_and_adam=digest == m.core.state_digest(
                        dict(model=model.state_dict(), optimizer=optimizer.state_dict())))
    write(job/'restore_equality.json', equality)
    if not all(equality.values()):
        raise ValueError('Restored next update/model/Adam/RNG/transient state differs')
    retired = [m.data.record(before_path, job)]
    m.base.persistence.remove_work_file(before_path, job)
    envelope = None
    if method.algorithm is m.l2r:
        # Current block <=28 queries/positives +40*5 negatives =228 accounts.
        rows = [a for g in current for a in m.l2r.accounts(g,'B')][:228]
        def maximum_graph():
            optimizer.zero_grad(set_to_none=True)
            model.train()
            z = m.l2r.encode(model, rows, c, budget.check)
            z.square().mean().backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            if norm <= 0:
                raise ValueError('Maximum graph gradient missing')
            optimizer.zero_grad(set_to_none=True)
            return dict(accounts=228, records=1824, token_sequences=3648,
                        tokens_per_sequence=256, gradient_norm=norm, optimizer_updates=0)
        envelope = timed('maximum_encoding_graph', maximum_graph)
        # Handwritten completed candidate exposure, solely for native maintenance qualification.
        all_rows = list(stage.index.by_id.values())
        queries = all_rows+[p['query'] for p in memory.packages.values()]
        stage.mc = {q.uid: {a.uid for a in itertools.islice(
            (a for a in all_rows if not m.l2r.same_identity(q,a)), 10)} for q in queries}
    else:
        stage.entered = set(stage.current)
        stage.candidates = sorted(stage.current)[:6]
    for state in optimizer.state.values():
        state['step'].fill_(576)
    point, retained, _ = timed('actual_stage_checkpoint', lambda: run.checkpoint(
        job, method, model, optimizer, stage, calibration, c, None, budget))
    if method.algorithm is m.gcr:
        if len(point['maintenance']['candidate_ids']) != 12:
            raise ValueError('GCR full 12-column native selection absent')
    retired.append(m.data.record(job/'models/stage2.pt', job))
    result = dict(status='PASS_NATIVE_HANDWRITTEN', spec=spec.identity(), timings=timings, calls=calls,
        maximum_group=dict(records=224, channels=2, tokens=256), l2r_encoding_envelope=envelope,
        initial_history_bytes=len(payload), history_slots=168, retained_memory=retained.summary(),
        stage_checkpoint=point, full_next_update_equal=True, encoder_actually_updated=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), budget=budget.summary(), archive=archive,
        retired_temporary_states=retired, formal_data_reads=0,
        boundary='3 real updates; synthetic counters 288/576 and candidate exposure; no formal trajectory')
    write(job/'result.json', result)
    # The actual endpoint is under models/, outside remove_work_file's deliberately
    # narrow work/ guard. Retire it mechanically after reviewing the saved evidence.


def controller(args) -> None:
    root = Path(__file__).resolve().parents[1]
    approval = json.loads(args.authorization.read_text(encoding='utf-8'))
    allocation = approval.get('worker_seconds')
    if (approval.get('status') != 'APPROVED_NATIVE_HANDWRITTEN'
        or allocation not in (NATIVE_SECONDS, 3000) or approval.get('wait_seconds') != 0
        or approval.get('limits') != LIMITS or args.job.exists()):
        raise ValueError('Concrete native authorization absent or run already exists')
    manifest = json.loads((root/'native_sources.json').read_text(encoding='utf-8'))
    if approval.get('sources') != manifest:
        raise ValueError('Authorization source identity differs')
    for row in manifest:
        raw = (root/row['path']).read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Native source differs')
    args.job.mkdir(parents=True)
    charged, receipts = approval.get('prior_charged_seconds', 0), []
    if not 0 <= charged < allocation:
        raise ValueError('Invalid carried native charges')
    methods = tuple(approval.get('remaining_methods', METHODS))
    if methods not in (METHODS, ('L2R_text_fixed',)):
        raise ValueError('Only the fixed pair or remaining L2R qualification is allowed')
    for name in methods:
        resource = resources()
        write(args.job/'resource.json', dict(method=name, state='ADMISSION_CHECK', **resource))
        if not ready(resource):
            raise RuntimeError('Available resources below the bounded shared-GPU trial gate')
        other_pids = []
        for line in resource['compute_apps'].splitlines():
            pid = int(line.split(',')[0])
            p = psutil.Process(pid)
            if (p.username() != 'yongpeng' or not any('step28_continual_five_run.py' in s for s in p.cmdline())
                or p.cwd() != '/home/yongpeng/cross-lingual/reports/documentation/20261023/continual_implementation/workspace'):
                raise RuntimeError('Unrecognized concurrent GPU process; do not compete with it')
            other_pids.append(dict(pid=pid, user=p.username(), started=p.create_time(), cwd=p.cwd(), command=p.cmdline()))
        resource['verified_concurrent_processes'] = other_pids
        if charged >= allocation-60:
            raise TimeoutError('Remaining native allocation exhausted before next load')
        job = args.job/name
        job.mkdir()
        started = time.monotonic()
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command = [sys.executable, '-B', __file__, '--worker', name, '--job', str(job),
                   '--deadline', str(started+allocation-charged)]
        peak, reason = 0, None
        with (job/'console.txt').open('w') as output:
            child = subprocess.Popen(command, cwd=root, env=environment, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                try:
                    p = psutil.Process(child.pid)
                    rss = sum(x.memory_info().rss for x in [p,*p.children(recursive=True)] if x.is_running())
                    peak = max(peak,rss)
                    used = sum(x.stat().st_size for x in job.rglob('*') if x.is_file())
                    if time.monotonic()-started >= allocation-charged:
                        reason = 'TOTAL_NATIVE_TIME_LIMIT'
                    elif rss > LIMITS['rss_bytes'] or used > LIMITS['disk_bytes']:
                        reason = 'NATIVE_RESOURCE_LIMIT'
                    if reason:
                        os.killpg(child.pid, signal.SIGKILL)
                        break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.2)
            code = child.wait()
        elapsed = time.monotonic()-started
        charged += math.ceil(elapsed)
        receipt = dict(method=name, exit_code=code, reason=reason, wall_seconds=elapsed,
                       charged_seconds=math.ceil(elapsed), cumulative_seconds=charged,
                       peak_rss_bytes=peak, admission_resource=resource)
        write(job/'receipt.json', receipt)
        receipts.append(receipt)
        write(args.job/'status.json', dict(status='FAILED_STOP' if code else 'WORKER_COMPLETE',
                                          receipts=receipts, sources=manifest))
        if code:
            raise RuntimeError('Native attempt failed; no retry or next worker')
    write(args.job/'status.json', dict(status='COMPLETE_NATIVE_HANDWRITTEN', receipts=receipts, sources=manifest))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--worker', choices=METHODS)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    try:
        if args.worker:
            worker(args.job, args.worker, args.deadline)
        else:
            controller(args)
    except Exception:
        if args.job.is_dir():
            (args.job/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
