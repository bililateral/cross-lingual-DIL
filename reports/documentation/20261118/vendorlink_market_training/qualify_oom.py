"""Linux CPU qualification of the four-arm dispatcher with real child processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE/'scripts'
sys.path.insert(0, str(SCRIPTS))
PY = '/home/yongpeng/miniconda3/envs/py310/bin/python'


def read(path):
    return json.loads(path.read_text())


def wait(check):
    deadline = time.monotonic()+40
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(.04)
    raise TimeoutError(f'Qualification condition at line {check.__code__.co_firstlineno} was not met')


def child(args):
    # Only the authored fixture executes in the child. No tensor, CUDA or formal inputs.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from step30_evolution_io import write
    root, target = args.gate.parent, args.target
    scenario = read(args.gate)['case']
    attempt = int(target.name.split('-')[-1])
    run = target/'run'
    write(run/'workload.json', dict(current_updates=1))
    if scenario == 'handoff' and args.task.startswith('H_'):
        if (args.task, attempt) == ('H_full_s51', 1):
            wait(lambda: (root/'fail_H').exists())
        elif (args.task, attempt) == ('H_full_s51', 2):
            wait(lambda: (root/'finish_retry').exists())
        elif args.task == 'H_ER_s51':
            wait(lambda: (root/'finish_healthy').exists())
        elif args.task in ('H_SEQ_s51', 'H_full_s52', 'H_full_s53'):
            wait(lambda: (root/'finish_three').exists())
    fail = args.task == 'VendorLink_SEQ_s51' and (attempt == 1 or scenario == 'repeat')
    fail |= scenario == 'handoff' and (args.task, attempt) == ('H_full_s51', 1)
    if fail:
        kind = 'OTHER' if scenario == 'other' else 'CUDA_OOM'
        write(target/'worker_failure.json', dict(kind=kind, error='injected CPU fixture '+kind))
        raise RuntimeError('injected '+kind)
    write(run/'completion.json', dict(fixture=True, task=args.task, seed=int(args.task.split('_s')[-1])))


def supervise(gate):
    import step28_formal_queue as hardware
    import step30_evolution_control as dispatch
    from step30_evolution_io import Ledger, record, write
    from step35_market_queue import CAPS, H_ORDER, MarketRunner, V_ORDER, run_training
    root = gate.parent
    scale = 2**20

    def terminate(signum, frame):
        raise KeyboardInterrupt('Stop and release qualification children')

    signal.signal(signal.SIGTERM, terminate)

    class FixtureNvml:
        def snapshot(self):
            held = root/'lingering_pid'
            pids = {int(held.read_text()): scale} if held.exists() else {}
            return dict(free=(2 if (root/'limit_gpu').exists() else 64)*scale, processes=pids)

        def close(self):
            pass

    hardware.GpuMemory = FixtureNvml
    # Faster observation only. Real monotonic lifetimes, resource accounting and
    # fork/exit, Linux identities, process groups and resource admission stay active.
    dispatch.time = SimpleNamespace(monotonic=time.monotonic, sleep=lambda duration: time.sleep(min(duration, .06)))
    limits = dict(workers=3, free_gpu_bytes=scale, vendor_reserve_bytes=3*scale,
        rss_bytes=1024*scale, free_ram_bytes=32*scale, save_reserve_bytes=scale,
        free_disk_reserve=16*scale, disk_bytes=64*scale)
    allowed = dict(workspace=str(HERE), experiment_root=str(root), other_gpu_commitments=[])
    prior = dict.fromkeys(CAPS, 0)
    if read(gate)['case'] == 'budget':
        prior['training'] = CAPS['training']-3*3600
    ledger = Ledger(root, prior, 0, caps=CAPS)
    runner = MarketRunner(allowed, gate, root, ledger, worker_entry=Path(__file__), limits=limits,
                          caps=CAPS, source_bindings=[record(gate), record(Path(__file__))],
                          halt_dispatch_on_failure=True, continue_after_oom=True)
    tasks = [dict(id=f'{method}_s{seed}', method=method, seed=seed, stages=3, epochs=40,
                  domains=['AlphaBay', 'Dream', 'Agora']) for method, seed in V_ORDER+H_ORDER]

    def completed(job):
        saved = read(job['target']/'run/completion.json')
        assert saved == dict(fixture=True, task=job['task_id'], seed=int(job['task_id'].split('_s')[-1]))

    try:
        run_training(runner, tasks, completed)
    except RuntimeError as exc:
        if 'Incomplete phase' not in str(exc):
            raise
        write(root/'fixture_incomplete.json', dict(error=str(exc)))


def exercise(root, scenario):
    from step30_evolution_io import record, write
    from step35_market_queue import request_recovery
    root.mkdir()
    gate = root/'fixture.json'
    write(gate, dict(case=scenario))
    out = (root/'parent.log').open('x')
    parent = subprocess.Popen([PY, '-B', str(Path(__file__).resolve()), '--supervise', str(gate)],
                              stdout=out, stderr=subprocess.STDOUT)
    def state():
        return read(root/'dispatch.json') if (root/'dispatch.json').exists() else {}
    def directory(task, attempt):
        return root/'training'/task/f'attempt-{attempt}'
    def stopped(task, attempt):
        return next((row for row in state().get('oom_slots', []) if
                     (row['task'], row['attempt']) == (task, attempt)), None)
    def live(task):
        return next((item for item in state().get('active', {}).values() if item['task_id'] == task), None)
    def request(task, attempt):
        return str(request_recovery(root, task, attempt))
    v, full, healthy, seq, following = ('VendorLink_SEQ_s51', 'H_full_s51', 'H_ER_s51', 'H_SEQ_s51', 'H_full_s52')
    evidence = dict(case=scenario)
    try:
        if scenario != 'other':
            row = wait(lambda: stopped(v, 1))
            assert row['slot'] == 0 and not state()['active'] and not (root/'training_complete.json').exists()
            if scenario == 'handoff':
                (root/'lingering_pid').write_text(str(row['pid']))
            command = [PY, '-B', str(SCRIPTS/'step35_market_queue.py'), 'request-oom',
                       '--campaign', str(root), '--task', v, '--attempt', '1']
            subprocess.run(command, check=True, capture_output=True, timeout=15)
            if scenario == 'handoff':
                time.sleep(.4)
                assert not (directory(v, 2)/'process.json').exists() and not state()['active']
                assert request(v, 1) == str(next(directory(v, 1).glob('*.request')))
                (root/'lingering_pid').unlink()
                initial = wait(lambda: live(full) and live(healthy) and live(seq) and state()['active'])
                (root/'fail_H').touch()
                h_oom = wait(lambda: stopped(full, 1))
                first_receipt = record(directory(full, 1)/'dispatch_receipt.json')
                assert read(directory(full, 1)/'worker_failure.json')['kind'] == 'CUDA_OOM'
                (root/'finish_healthy').touch()
                wait(lambda: live(following) and live(seq))
                before = {k: v['pid'] for k, v in state()['active'].items()}
                assert set(before) == {'1', '2'} and before['2'] == initial['2']['pid']
                assert before['1'] != initial['1']['pid']
                (root/'limit_gpu').touch()
                with ThreadPoolExecutor(2) as pool:
                    sent = list(pool.map(lambda _: request(full, 1), range(2)))
                assert sent[0] == sent[1]
                admission = directory(full, 2)/'admission.json'
                wait(lambda: admission.exists() and not read(admission)['ready'])
                assert not (directory(full, 2)/'process.json').exists()
                (root/'limit_gpu').unlink()
                together = wait(lambda: live(full) and live(full)['attempt'] == 2 and
                                len(state()['active']) == 3 and state()['active'])
                wait(lambda: (directory(full, 2)/'run/workload.json').is_file())
                assert all(together[k]['pid'] == pid for k, pid in before.items())
                assert together['0']['pid'] != h_oom['pid']
                assert (directory(full, 1)/'slot_reopened.json').exists()
                (root/'finish_retry').touch()
                next_on_restored = wait(lambda: live('H_full_s53') and state()['active'].get('0'))
                assert next_on_restored['task_id'] == 'H_full_s53'
                wait(lambda: (directory('H_full_s53', 1)/'run/workload.json').is_file())
                assert all(state()['active'][k]['pid'] == pid for k, pid in before.items())
                (root/'finish_three').touch()
                evidence.update(healthy_slot_next=[healthy, following], old_failure_pid=h_oom['pid'],
                    healthy_pids_during_rejoin=before, concurrent_pids={k:r['pid'] for k,r in together.items()},
                    concurrent_tasks={k:r['task_id'] for k,r in together.items()},
                    restored_slot_next=next_on_restored['task_id'],
                    duplicate_request_same_marker=True, rejected_lingering_gpu_pid=True, held_low_vram=True)
        parent.wait(timeout=35)
        assert parent.returncode == 0, (root/'parent.log').read_text()
        state_end = state()
        events = [e for e in read(root/'budget.json')['events'] if e['bucket'] == 'training']
        assert all(e['charged_seconds'] == math.ceil(e['actual_seconds']) and e['charged_seconds']>0 for e in events)
        assert len({e['operation'] for e in events}) == len(events)
        receipts = [read(f) for f in (root/'training').glob('*/*/dispatch_receipt.json')]
        assert len(events) == len(receipts)
        charges = {e['operation']:e for e in events}
        for receipt in receipts:
            event = charges[f"{receipt['task_id']}/attempt-{receipt['attempt']}"]
            assert event['actual_seconds'] == receipt['seconds'] and event['success'] == receipt['success']
        result_path = root/'training_complete.json'
        assert result_path.exists() == (scenario == 'handoff')
        if scenario == 'handoff':
            winners = read(result_path)['winners']
            assert len(winners) == 12 and len(events) == 14 and state_end['status'] == 'COMPLETE'
            assert sorted(t for t,w in winners.items() if w['attempt']==2) == sorted([full,v])
            assert record(directory(full, 1)/'dispatch_receipt.json') == first_receipt
            for task, w in winners.items():
                assert w['completion'] == record(directory(task,w['attempt'])/'run/completion.json')
            evidence['winning_retries'] = {task:winners[task]['attempt'] for task in (v,full)}
        else:
            assert state_end['status'] == 'INCOMPLETE'
            assert len(events) == (2 if scenario == 'repeat' else 1)
            assert not any(p.name.startswith('H_') for p in (root/'training').iterdir())
            assert not directory(v, 3).exists()
            if scenario == 'other':
                try:
                    request(v, 1)
                    raise AssertionError('Non-OOM accepted')
                except ValueError:
                    pass
            elif scenario == 'budget':
                assert state_end['budget_stop'] and not (directory(v,2)/'process.json').exists()
            else:
                assert all(not event['success'] for event in events)
        evidence.update(status=state_end['status'], published_winners=12 if result_path.exists() else 0,
            attempts=len(events), actual_job_charged_seconds=sum(e['charged_seconds'] for e in events),
            oom_attempt_fees=[{k:e[k] for k in ('operation','success','charged_seconds')} for e in events
                              if e['operation'].split('/')[0] in (v,full)],
            budget_stop=state_end['budget_stop'])
        return evidence
    finally:
        if parent.poll() is None:
            for name in ('fail_H','finish_healthy','finish_retry','finish_three'):
                (root/name).touch(exist_ok=True)
            parent.terminate()
            parent.wait(timeout=6)
        out.close()


def qualify():
    if Path(sys.executable).resolve() != Path(PY).resolve() or sys.platform != 'linux':
        raise RuntimeError('Use the existing Linux py310')
    if len(os.sched_getaffinity(0)) < 6:
        raise RuntimeError('Qualification needs six available CPU affinities for the three real two-core slots')
    out = HERE/'evidence'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    rows = [exercise(out/case,case) for case in ('handoff','repeat','other','budget')]
    sources = [Path(__file__).resolve(), *[SCRIPTS/n for n in ('step35_market_queue.py','step30_evolution_control.py',
        'step30_evolution_io.py','step30_evolution_policy.py','step28_formal_queue.py')]]
    result = dict(status='PASS', observed_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-started,
        python=sys.executable, inputs='authored CPU subprocesses; structured OOM injection; synthetic NVML',
        model_updates=0, formal_data_access=False, gpu_kernels=0, cases=rows,
        source_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (HERE/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'])))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--supervise',type=Path)
    p.add_argument('--gate',type=Path)
    p.add_argument('--action',choices=['train'])
    p.add_argument('--target',type=Path)
    p.add_argument('--deadline',type=float)
    p.add_argument('--task')
    args=p.parse_args()
    if args.action:
        child(args)
    elif args.supervise:
        supervise(args.supervise)
    else:
        qualify()
