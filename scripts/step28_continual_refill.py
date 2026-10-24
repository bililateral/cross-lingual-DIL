"""Recovery-only refill fix; frozen training, startup and failure handling unchanged.

Dispatch body copied from frozen controller SHA256
95da8b50ab412efc4b81a9675eb9d9ab4b6ad3e85b2fe1e3263c03224575c3eb.
Only a false admission result defers a vacant slot, including an empty queue of
active workers. Real child failures, deadlines and budget exceptions still stop.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

import psutil


def slot_available(gate, active, campaign):
    free = int(subprocess.check_output(
        ['nvidia-smi', '--id=0', '--query-gpu=memory.free',
         '--format=csv,noheader,nounits'], text=True).strip())*2**20
    # Keep the original host-memory failure; only GPU refill logic is amended.
    if psutil.virtual_memory().available < 64*2**30:
        raise RuntimeError('Actual host resource admission unavailable; no launch')
    available = free >= gate['minimum_free_slot_bytes']
    # Import the recovery writer lazily; it has no model/data side effects.
    from step28_continual_five_recover import write
    write(Path(campaign)/'admission_status.json', dict(
        status='READY' if available else 'WAITING_FOR_ONE_SLOT',
        epoch=time.time(), active_workers=active, free_gpu_bytes=free,
        required_free_gpu_bytes=gate['minimum_free_slot_bytes']))
    return available


def dispatch(ctl, queue: list[dict], campaign: Path, ledger, bucket: str,
             command, *, max_workers: int = 3, tail: float = 60., admit=None,
             poll_seconds: float = 30.) -> list[dict]:
    """Fixed queue, no replacement or retry. Wait time is not controller work."""
    run, data, terminate_owned = ctl.run, ctl.data, ctl.terminate_owned
    active, completed, next_index = {}, [], 0
    cores = psutil.Process().cpu_affinity()[:max_workers]
    if len(cores) != max_workers:
        raise RuntimeError("Distinct single-core workers unavailable")
    pending_error, tick = None, None
    try:
        while active or next_index < len(queue):
            tick = time.monotonic()
            for identity, (process, log, slot) in list(active.items()):
                if process.poll() is None and time.monotonic() >= ledger.active[identity]["deadline"]-tail:
                    tick += terminate_owned(process)
                    raise TimeoutError("Child stopped within its reserved termination tail")
                if process.poll() is not None:
                    log.close()
                    del active[identity]
                    completed.append(ledger.settle(identity, success=process.returncode == 0))
            owned = run.owned_processes(os.getpid())
            rss = run.resident_bytes(owned)
            if rss > 192*2**30:
                raise RuntimeError("Campaign owned-PID-union RSS exceeded")
            while len(active) < max_workers and next_index < len(queue):
                row = queue[next_index]
                if admit:
                    if not admit(len(active)):
                        break
                remaining = sum(q.get("projected_seconds", q["allowance"]) for q in queue[next_index+1:])
                allowance = min(row["allowance"], ledger.available(bucket)-remaining)
                if allowance < row.get("projected_seconds", row["allowance"]):
                    raise RuntimeError("Available allowance cannot cover this run and the pending campaign")
                reservation = ledger.reserve(row["id"], bucket, allowance, remaining_projection=remaining)
                slot = next(v for v in range(max_workers) if v not in [a[2] for a in active.values()])
                job = campaign/row["id"]
                job.mkdir()
                command_line = ["taskset", "-c", str(cores[slot]), *command(row, job, reservation["deadline"])]
                data.write_json(job/"dispatch.json", dict(reservation=reservation, command=command_line,
                                                        source_files=run.sources()))
                log = (job/"console.txt").open("w", encoding="utf-8")
                env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "0",
                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONDONTWRITEBYTECODE": "1"}
                try:
                    process = subprocess.Popen(command_line, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                except BaseException:
                    log.close()
                    ledger.settle(row["id"], success=False)
                    raise
                active[row["id"]] = (process, log, slot)
                data.write_json(job/"process.json", dict(pid=process.pid, owner=psutil.Process(process.pid).username(),
                                                       created=psutil.Process(process.pid).create_time()))
                next_index += 1
            elapsed = time.monotonic()-tick
            tick = None
            ledger.charge("control", elapsed)
            if active or next_index < len(queue):
                next_stop = min((ledger.active[identity]["deadline"]-tail for identity in active),
                                default=time.monotonic()+poll_seconds)
                time.sleep(max(0., min(poll_seconds, next_stop-time.monotonic())))
    except BaseException as error:
        pending_error = error
    finally:
        if tick is not None:
            try:
                ledger.charge("control", time.monotonic()-tick)
            except RuntimeError as error:
                pending_error = error
        cleanup_started, cleanup_waited = time.monotonic(), 0.
        for identity, (process, log, _) in list(active.items()):
            try:
                cleanup_waited += terminate_owned(process)
            finally:
                log.close()
                try:
                    ledger.settle(identity, success=False)
                except RuntimeError:
                    pass
        # A timeout can remove the child from active only after settling it.
        for identity in list(ledger.active):
            try:
                ledger.settle(identity, success=False)
            except RuntimeError:
                pass
        ledger.charge("control", time.monotonic()-cleanup_started-cleanup_waited)
    if pending_error:
        raise pending_error
    return completed
