"""Bounded handwritten CPU checks of the actual Evolution two-slot controller."""
from pathlib import Path
import argparse
import json
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--permit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    allowed = json.loads(args.permit.read_text())
    if (sys.platform != 'linux' or str(Path(sys.executable).resolve()) !=
            str(Path('/home/yongpeng/miniconda3/envs/py310/bin/python').resolve())
            or allowed.get('operation') != 'Evolution_two_worker_cpu' or allowed.get('approved') is not True):
        raise ValueError('Approved Linux handwritten CPU scope required')
    sys.path.insert(0, allowed['scripts'])
    import step30_evolution_control as control
    import step28_formal_queue as queue
    from step30_evolution_io import Ledger, PYTHON, read, record, verify, write
    from step30_evolution_policy import LIMITS, CAPS, tasks
    for row in [allowed['authorization'], *allowed['sources']]:
        verify(row)
    args.output.mkdir(parents=True, exist_ok=False)
    def alarm(*_):
        raise TimeoutError('CPU qualification lifecycle cap')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(90)
    checks, success = [], False
    started = time.monotonic()
    real_reader, original_command = queue.GpuMemory, control.Runner.command
    original_dataset, original_tasks = control.dataset, control.tasks

    class Snapshot:
        over_target = None
        def snapshot(self):
            processes = {}
            if self.over_target and (self.over_target/'process.json').exists():
                processes[read(self.over_target/'process.json')['pid']] = 15*2**30
            return dict(free=64*2**30, total=80*2**30, processes=processes)
        def close(self):
            pass

    def command(self, job, slot, deadline):
        # Real subprocesses; no torch/model/GPU import inside children. Completion
        # metadata are declared synthetic and only exercise controller barriers.
        code = '''import json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]); bad=sys.argv[2]=='1'; duration=float(sys.argv[3]); action=sys.argv[4]
start=time.monotonic()
(p/'started.json').write_text(json.dumps(dict(start=start,pid=os.getpid(),action=action)))
if action=='train':
    (p/'run').mkdir(exist_ok=True)
    (p/'run/updates.jsonl').write_text(json.dumps(dict(optimizer_updates=1,total_updates=1))+'\\n')
time.sleep(duration)
if bad: sys.exit(3)
if (p/'spec.json').exists():
    r=json.loads((p/'spec.json').read_text());t=r['task'];n=t['stages']*t['epochs']*20
    w=dict(current_updates=n,current_records=32*n,history_groups=0,history_records=0,teacher_groups=0,teacher_records=0,stage_model_saves=t['stages'])
    c=dict(status='TRAINING_COMPLETE_EVALUATION_CLOSED',task=t,identity=r['identity'],updates=n,workload=w,endpoints={str(i):{} for i in range(t['stages']+1)})
    (p/'run/completion.json').write_text(json.dumps(c))
(p/'finished.json').write_text(json.dumps(dict(start=start,end=time.monotonic(),pid=os.getpid())))
'''
        fail = job['target'].name.endswith('s11') and job['target'].parent.name == 'A' and 'H0_' in job['target'].name
        duration = .5 if fail else 1.
        if job['target'].name == 'over':
            duration = 3.
        cores = control.psutil.Process().cpu_affinity()[-4:][slot*2:slot*2+2]
        return ['taskset', '-c', ','.join(map(str, cores)), PYTHON, '-B', '-c', code,
                str(job['target']), str(int(fail)), str(duration), job['action']]

    try:
        # Existing NVML reader is exercised without creating a CUDA context.
        nv = real_reader()
        physical = nv.snapshot()
        nv.close()
        checks.append(dict(name='existing_nvml_read_only', passed=physical['total'] > 0))
        queue.GpuMemory = Snapshot
        control.Runner.command = command
        root = args.output/'campaign'
        workspace = args.output/'workspace'
        (workspace/'scripts').mkdir(parents=True)
        waiter = workspace/'scripts/once_wait.py'
        waiter.write_text('# handwritten one-shot copy for deletion-trigger test\n')
        gate = args.output/'handwritten_gate.json'
        write(gate, dict(handwritten=True))
        scope = dict(campaign=str(root), experiment_root=str(args.output), workspace=str(workspace),
                     prior_seconds=dict(training=0, gpu=0, auxiliary=0), cpu_test_seconds=0,
                     other_gpu_commitments=[], dataset_manifest=dict(handwritten=True), waiter=record(waiter))
        fixed = tasks('A')[:4]
        control.tasks = lambda phase, selected=None: fixed
        control.dataset = lambda _: dict(handwritten=True)
        try:
            control.campaign(scope, gate)
            raise AssertionError('Incomplete actual campaign accepted')
        except RuntimeError as exc:
            assert 'Incomplete phase' in str(exc), str(exc)
        state = read(root/'dispatch.json')
        batch = read(root/'A/batch.json')
        assert len(state['failures']) == 1 and len(batch['completed']) == 3 and not state['pending']
        rows = [read(root/'A'/t['id']/'dispatch_receipt.json') for t in fixed]
        assert [r['slot'] for r in rows] == [0, 1, 1, 1]
        assert [r['success'] for r in rows] == [False, True, True, True]
        times = [read(root/'A'/t['id']/'started.json')['start'] for t in fixed]
        assert abs(times[0]-times[1]) < .5
        assert times[2] >= read(root/'A'/fixed[1]['id']/'finished.json')['end']
        assert not (root/'A/console.txt').exists() and not (root/'A/analysis').exists() and not (root/'B').exists()
        assert not waiter.exists() and read(root/'waiter_deleted.json')['deleted']
        ledger = read(root/'budget.json')
        assert ledger['used_seconds']['training'] == sum(__import__('math').ceil(r['seconds']) for r in rows)
        checks.append(dict(name='actual_campaign_overlap_failure_retirement_healthy_refill_and_label_barrier',
                           slots=[r['slot'] for r in rows], passed=True, first_update_deletes_waiter=True))

        self_process = control.psutil.Process()
        identity = dict(pid=self_process.pid, owner=self_process.username(), created=self_process.create_time(),
                        cwd=self_process.cwd(), command=self_process.cmdline(), reserved_bytes=14*2**30)
        class Tight(Snapshot):
            def snapshot(self):
                return dict(free=20*2**30, total=40*2**30, processes={self_process.pid: 7*2**30})
        observation = control.resource_observation(scope, args.output, True, [identity], Tight())
        assert observation['unreflected_commitments'] == 7*2**30 and not observation['ready']
        checks.append(dict(name='unreflected_reservation_prevents_double_spend', passed=True))

        over = args.output/'over'
        over.mkdir()
        Snapshot.over_target = over
        runner = control.Runner(scope, gate, args.output/'limit_state', Ledger(args.output/'limit_state',
            dict(training=0, gpu=0, auxiliary=0), 0))
        try:
            runner.run('train', over, 'training', 60, [])
            raise AssertionError('Whole PID cap not enforced')
        except RuntimeError as exc:
            assert 'Incomplete phase' in str(exc)
        receipt = read(over/'dispatch_receipt.json')
        assert 'RAM/GPU limit' in receipt['error'] and receipt['released'] and not (over/'finished.json').exists()
        checks.append(dict(name='whole_pid_14_gib_cap_stops_real_owned_child', passed=True, synthetic_nvml=True))
        Snapshot.over_target = None

        reserve_root = args.output/'reservation'
        reserve_root.mkdir()
        ledger = Ledger(reserve_root, dict(training=CAPS['training']-100, gpu=0, auxiliary=0), 0)
        runner = control.Runner(scope, gate, reserve_root, ledger)
        jobs = []
        for name in ('one', 'two'):
            target = reserve_root/name
            target.mkdir()
            jobs.append(dict(action='train', target=target, bucket='training', seconds=60, extra=[]))
        runner.run_many(jobs, 2, lambda job: None)
        assert read(reserve_root/'two/started.json')['start'] >= read(reserve_root/'one/finished.json')['end']
        checks.append(dict(name='inflight_full_caps_reserved_and_released_before_second_start', passed=True))
        success = True
    finally:
        queue.GpuMemory, control.Runner.command = real_reader, original_command
        control.dataset, control.tasks = original_dataset, original_tasks
        signal.alarm(0)
        write(args.output/'receipt.json', dict(status='PASS' if success else 'FAIL', checks=checks,
              body_seconds=time.monotonic()-started, permit=record(args.permit),
              handwritten=True, real_data_read=False, model_loaded=False, cuda_allocated=False,
              synthetic_training_receipts_are_not_experiments=True))


if __name__ == '__main__':
    main()
