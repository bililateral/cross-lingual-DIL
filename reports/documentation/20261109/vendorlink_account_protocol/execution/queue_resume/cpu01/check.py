"""Handwritten CPU-only provenance, budget, resource, and predecessor checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'workspace/scripts'))
import step34_vendorlink_account_resume as resume
from step34_vendorlink_account_data import record, write
from step34_vendorlink_account_policy import tasks, DOMAINS, LIMITS, PROTOCOL
from step30_evolution_control import resource_observation


def rejected(action):
    try:
        action()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('Required negative case was accepted')


started = time.monotonic()
checks = []
with tempfile.TemporaryDirectory(prefix='resume_cpu_', dir=HERE) as temporary:
    root = Path(temporary)
    manifest = root/'manifest.json'
    write(manifest, dict(counts={d: dict(eligible_accounts=8) for d in DOMAINS}))
    sources = []
    for name in resume.native.SOURCES:
        p = root/name
        p.write_text('CPU source-binding fixture; never imported as a scientific module\n')
        sources.append(record(p))
    g = dict(manifest=record(manifest), model_binding=dict(directory='CPU fixture only'), limits=LIMITS)
    specs = tasks('P', {'H': 'H1', 'V': 'V0'})
    origins = {}
    for i, task in enumerate(specs):
        gate = root/f'gate{i}.json'
        write(gate, dict(protocol=PROTOCOL, approved=True, review_closed=True, manifest=g['manifest'],
                         model_binding=g['model_binding'], sources=sources))
        gate_binding = record(gate)
        identity = dict(protocol=PROTOCOL, task=task, manifest=g['manifest'], gate=gate_binding)
        endpoints = {}
        for stage in range(4):
            endpoint = root/f'cpu_endpoint_{i}_{stage}.json'
            write(endpoint, dict(identity=identity, stage=stage, CPU_only_not_a_model=True))
            endpoints[str(stage)] = record(endpoint)
        completion = root/f'completion{i}.json'
        write(completion, dict(task=task, identity=identity,
              status='TRAINING_COMPLETE_EVALUATION_CLOSED', endpoints=endpoints,
              updates=360, workload=dict(current_ads=11520)))
        origins[task['id']] = dict(gate=gate_binding, completion=record(completion))
    frozen = resume.collect_training(g, specs, origins)
    assert len(frozen['origins']) == 6
    bad = copy.deepcopy(origins)
    bad[specs[0]['id']]['gate'] = origins[specs[1]['id']]['gate']
    rejected(lambda: resume.collect_training(g, specs, bad))
    bad = dict(origins)
    bad.pop(specs[-1]['id'])
    rejected(lambda: resume.collect_training(g, specs, bad))
    checks.append('Six heterogeneous gate-bound completions required; swapped gate and missing task rejected')
    task = specs[3]
    p = root/'failed_attempt.json'
    failed = json.loads(Path(origins[task['id']]['completion']['path']).read_text())
    failed['updates'] = 1
    write(p, failed)
    rejected(lambda: resume.sourced_result(g, task, dict(gate=origins[task['id']]['gate'], completion=record(p))))
    checks.append('Incomplete failed-attempt completion rejected rather than used for scoring')
    prior = dict(used_seconds=dict(training=41230, gpu=88, auxiliary=1955), events=[], cpu_test_seconds=322)
    ledger = resume.inherited_ledger(root/'account', prior)
    assert ledger.used == prior['used_seconds'] and ledger.caps['training'] == 149400
    external = dict(caps=dict(training=28800, gpu=0, auxiliary=1800),
                    used_seconds=dict(training=13000, gpu=0, auxiliary=300), events=[], cpu_test_seconds=0)
    resume.merge_h45(ledger, root/'account', external)
    assert ledger.used == dict(training=54230, gpu=88, auxiliary=2255)
    assert ledger.caps == resume.GLOBAL_CAPS
    rejected(lambda: resume.merge_h45(ledger, root/'account', external))
    checks.append('Full H45 reservation subtracted first; actual incremental costs merged once; old history not doubled')
    h = resume.limits(g, specs[0]); v = resume.limits(g, specs[1])
    assert h['allocator_bytes'] == 13*2**30 and h['free_gpu_bytes'] == 14*2**30
    assert v['allocator_bytes'] == 15*2**30 and v['process_gpu_bytes'] == v['free_gpu_bytes'] == 16*2**30
    this = psutil.Process()
    commitment = dict(pid=this.pid, created=this.create_time(), owner=this.username(),
                      command=this.cmdline(), cwd=this.cwd(), reserved_bytes=14*2**30)
    allowed = dict(experiment_root=str(root), other_gpu_commitments=[commitment])
    class NV:
        def snapshot(self):
            return dict(free=23*2**30, processes={this.pid: 8*2**30})
    assert resource_observation(allowed, root, True, nv=NV(), limits=v)['ready']
    commitment['reserved_bytes'] = 16*2**30
    assert not resource_observation(allowed, root, True, nv=NV(), limits=v)['ready']
    checks.append('H14+V16 commitment admitted; V16+V16 blocked on31GiB fixture, without CUDA allocation')
    proc = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'])
    try:
        p = psutil.Process(proc.pid)
        row = dict(pid=p.pid, created=p.create_time(), owner=p.username(), command=p.cmdline(), cwd=p.cwd())
        assert resume.process_live(row)
        bad = dict(row, created=row['created']+1)
        rejected(lambda: resume.process_live(bad))
    finally:
        proc.terminate(); proc.wait(timeout=5)
    assert not resume.process_live(row)
    checks.append('Real CPU predecessor identity checked; changed identity rejected; actual exit releases wait condition')
result = dict(status='PASS', checks=checks, seconds=time.monotonic()-started,
              source=record(Path(resume.__file__)), check_source=record(Path(__file__)),
              no_model_loading=True, no_CUDA=True, no_formal_data=True,
              scope='Only new provenance/accounting/resource/predecessor glue; original science and phase pipeline reused')
write(HERE/'cpu01/receipt.json', result)
print(json.dumps(result, ensure_ascii=False))
