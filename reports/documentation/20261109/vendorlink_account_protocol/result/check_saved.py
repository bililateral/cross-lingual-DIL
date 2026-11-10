"""Check only released metadata and saved metrics; no model, labels, or predictions."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
import statistics as st
import subprocess
import time

import psutil

START = time.monotonic()
BASE = Path('/home/yongpeng/cross-lingual/reports/documentation/20261109/vendorlink_account_protocol/execution')
OUT = BASE.parent / 'result'
RUN = BASE / 'queue_resume/campaign'
TZ = timezone(timedelta(hours=8))
RECEIVE = {}


def read(path):
    path = Path(path)
    value = json.loads(path.read_text())
    relative = path.relative_to(BASE).as_posix()
    data = path.read_bytes()
    RECEIVE[relative] = dict(source=str(path), relative='source/' + relative,
                             bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return value


def pinned(binding):
    p = Path(binding['path'])
    data = p.read_bytes()
    assert len(data) == binding['bytes'] and hashlib.sha256(data).hexdigest() == binding['sha256'], str(p)
    return p


def write(name, value):
    path = OUT / name
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def stats(values):
    return dict(mean=st.mean(values), sample_sd=st.stdev(values), minimum=min(values), maximum=max(values))


assert '/envs/py310/' in __import__('sys').executable
OUT.mkdir(exist_ok=True)
done = read(RUN / 'completion.json')
assert done['status'] == 'STOP_LEARNING' and not done['application']
assert not (RUN / 'failure.json').exists()
assert all(not (RUN / p).exists() for p in ('M', 'C', 'application'))
dispatch = read(RUN / 'dispatch.json')
assert dispatch['status'] == 'COMPLETE' and not dispatch['active'] and not dispatch['failures']
read(BASE / 'queue_resume/state.json')
gate = read(BASE / 'queue_resume/gate.json')
for binding in gate['sources'] + [gate['resumption']['source']]:
    pinned(binding)
read(pinned(gate['manifest']))
train = read(RUN / 'P/training_complete.json')
scoring = read(RUN / 'P/scoring_complete.json')
access = read(RUN / 'P/analysis/result/access.json')
results = read(RUN / 'P/analysis/result/results.json')
metrics = read(RUN / 'P/analysis/result/metrics.json')
assert len(train['tasks']) == 6 and train['tasks'] == scoring['tasks'] == access['tasks']
assert access['opening'] == 'whole_batch_once'
assert set(results) == {'H_SEQ', 'V_CLS', 'V_SET', 'V_CLS_ads'}
cells = ('R01', 'R11', 'R12', 'R21', 'R22', 'R23', 'R31', 'R32', 'R33')
train_rows, files = [], 0
for task, binding in zip(train['tasks'], scoring['completions']):
    origin = train['origins'][task['id']]
    original_gate = read(pinned(origin['gate']))
    for source in original_gate['sources']:
        pinned(source)
    c = read(pinned(origin['completion']))
    assert c['task'] == task and c['identity'] == dict(protocol=gate['protocol'], task=task,
        manifest=gate['manifest'], gate=origin['gate'])
    assert c['status'] == 'TRAINING_COMPLETE_EVALUATION_CLOSED'
    assert c['updates'] == (23640 if task['method'] == 'H_SEQ' else 23720)
    assert c['workload']['current_updates'] == c['updates'] and c['workload']['current_ads'] == 757440
    assert all(c['workload'][k] == 0 for k in ('history_ads', 'history_groups', 'teacher_ads', 'teacher_groups'))
    assert set(c['endpoints']) == {'0', '1', '2', '3'}
    for endpoint in c['endpoints'].values():
        p = Path(endpoint['path'])
        assert p.exists() and p.stat().st_size == endpoint['bytes']  # No weight read or rehash.
    receipt = read(Path(origin['completion']['path']).parent.parent / 'dispatch_receipt.json')
    assert receipt['released']
    if not receipt['success']:
        assert task['id'] in ('P_VendorLink_SEQ_s44', 'P_H_SEQ_s45')
        assert receipt['error'] == 'TypeError("\'PosixPath\' object is not subscriptable")'
    score = read(pinned(binding))
    assert score['task'] == task and score['training_identity'] == c['identity']
    assert score['role'] == 'dev' and score['status'] == 'BLIND_SCORES_COMPLETE'
    assert score['cells'] == list(cells)
    expected_methods = ['H_SEQ'] if task['method'] == 'H_SEQ' else ['V_SET', 'V_CLS', 'V_CLS_ads']
    assert set(score['files']) == {m + ':' + cell for m in expected_methods for cell in cells}
    for name, f in score['files'].items():
        p = Path(f['path'])
        assert p.exists() and p.stat().st_size == f['bytes']  # No prediction contents read.
        saved = metrics[task['id'] + ':' + name]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in saved.values())
        assert saved['MRR'] == saved['MAP'] and saved['Top1'] == saved['micro_F1'] == saved['Recall@1']
    score_receipt = read(Path(binding['path']).parent.parent / 'dispatch_receipt.json')
    assert score_receipt['success'] and score_receipt['released']
    files += len(score['files'])
    train_rows.append(dict(task=task['id'], updates=c['updates'], workload=c['workload'],
        completed=datetime.fromtimestamp(Path(origin['completion']['path']).stat().st_mtime, TZ).isoformat(),
        receipt=receipt, scoring_receipt=score_receipt))
assert files == len(metrics) == 108
summary, differences, metric_means = {}, {}, {}
for method, rows in results.items():
    assert [row['seed'] for row in rows] == [43, 44, 45]
    for row in rows:
        expected = dict(Q=(row['R11']+row['R22'])/2, O=(row['R31']+row['R32'])/2,
            N=(row['R22']+row['R33'])/2, Z=row['R33'], GB=row['R22']-row['R12'], GC=row['R33']-row['R23'])
        expected['F'] = expected['Q']-expected['O']
        assert all(abs(row[k]-v) < 1e-12 for k, v in expected.items())
        for cell in cells:
            task = f'P_{"H_SEQ" if method == "H_SEQ" else "VendorLink_SEQ"}_s{row["seed"]}'
            assert row[cell] == metrics[f'{task}:{method}:{cell}']['macro_F1']
    summary[method] = {k: stats([row[k] for row in rows]) for k in rows[0] if k != 'seed'}
    for domain, initial, final in [('AlphaBay', 'R11', 'R31'), ('Dream', 'R22', 'R32')]:
        summary[method]['F_' + domain] = stats([row[initial]-row[final] for row in rows])
    metric_means[method] = {cell: {k: stats([metrics[f'P_{"H_SEQ" if method == "H_SEQ" else "VendorLink_SEQ"}_s{s}:{method}:{cell}'][k] for s in (43,44,45)])
        for k in next(iter(metrics.values()))} for cell in cells}
for method in ('V_CLS', 'V_SET'):
    differences[method] = {k: stats([a[k]-b[k] for a,b in zip(results['H_SEQ'], results[method])])
                            for k in results['H_SEQ'][0] if k != 'seed'}
learning = {}
for cell in ('R11', 'R22', 'R33'):
    h = summary['H_SEQ'][cell]
    best = max(summary[m][cell]['mean'] for m in ('V_CLS', 'V_SET'))
    learning[cell] = dict(H_mean=h['mean'], H_min=h['minimum'], V_best_mean=best,
        relative_gap=h['mean']-best, absolute_mean_pass=h['mean'] >= .70,
        absolute_seed_pass=h['minimum'] >= .60, relative_pass=h['mean'] >= best-.05, V_pass=best >= .70)
gains = {k: dict(**summary['H_SEQ'][k], passed=summary['H_SEQ'][k]['mean'] >= .05 and summary['H_SEQ'][k]['minimum'] >= 0) for k in ('GB','GC')}
forgetting = dict(**summary['H_SEQ']['F'], mean_pass=summary['H_SEQ']['F']['mean'] >= .05,
    seeds_at_least_3pp=sum(row['F'] >= .03 for row in results['H_SEQ']), all_nonnegative=all(row['F'] >= 0 for row in results['H_SEQ']))
decision = read(RUN / 'decisions/P.json')
computed_learning = all(all(v[k] for k in ('absolute_mean_pass','absolute_seed_pass','relative_pass','V_pass')) for v in learning.values()) and all(v['passed'] for v in gains.values())
computed_forgetting = forgetting['mean_pass'] and forgetting['seeds_at_least_3pp'] >= 2 and forgetting['all_nonnegative']
assert decision['checks'] == dict(learning=computed_learning, meaningful_forgetting=computed_forgetting) == done['P']['checks']
for rel in ['P/batch.json', 'P/analysis/dispatch_receipt.json', 'quantitative_completion.json', 'decisions/selection.json', 'H45_accounting_merged.json']:
    read(RUN / rel)
for rel in ['campaign/D/analysis/result/results.json', 'campaign/D/analysis/result/metrics.json', 'parallel_h45/campaign/dispatch.json', 'parallel_h45/campaign/budget.json', 'queue_resume/accounting_origin.json']:
    p = BASE / rel
    if p.exists():
        read(p)
budget = read(RUN / 'budget.json')
launch = read(BASE / 'launch.json')
read(BASE / 'queue_resume/launch.json')
old_processes = []
for rel in ('launch.json', 'queue_resume/launch.json', 'parallel_h45/launch.json', 'retry_s44_15g/launch.json'):
    p = BASE / rel
    if p.exists():
        row = read(p)
        pid = row.get('pid')
        old_processes.append(dict(source=rel, pid=pid, exists=psutil.pid_exists(pid)))
active = []
for p in psutil.process_iter(['pid','username','create_time','cmdline','cwd']):
    try:
        if p.pid != os.getpid() and str(BASE) in ' '.join(p.info['cmdline'] or []):
            active.append(p.info)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
assert not active
chinese_path = Path('/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/training/dispatch.json')
chinese = json.loads(chinese_path.read_text())
cp = psutil.Process(1412412)
chinese_state = dict(status=chinese['status'], completed=len(chinese['completed']), active=len(chinese['active']), pending=len(chinese['pending']),
    supervisor=dict(pid=cp.pid, owner=cp.username(), created=cp.create_time(), command=cp.cmdline(), cwd=cp.cwd()))
finished = datetime.fromtimestamp((RUN / 'completion.json').stat().st_mtime, TZ)
started = datetime.fromisoformat(launch['started_server'])
elapsed = time.monotonic()-START
charged = math.ceil(elapsed)
check = dict(observed=datetime.now(TZ).isoformat(), result_status=done['status'],
    training=train_rows, checked_sources='All referenced scientific sources and resumption entry matched',
    complete_training=6, complete_scoring=6, score_matrices=108, saved_metrics_per_matrix=len(next(iter(metrics.values()))),
    independent_saved_field_arithmetic='PASS', original_gate_checks=decision, learning=learning, gains=gains, forgetting=forgetting,
    weights_and_scores='Existence/size only; original validated digests retained, no contents loaded or rehashed',
    new_training=0, new_model_loads=0, raw_text_label_prediction_reads=0, old_processes=old_processes, active_processes=active,
    chinese=chinese_state, gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),
    completed=finished.isoformat(), started=started.isoformat(), elapsed_campaign_seconds=(finished-started).total_seconds(),
    original_used_seconds=budget['used_seconds'], original_caps=budget['caps'], original_cpu_test_seconds=budget['cpu_test_seconds'],
    result_check_seconds=elapsed, result_check_charged_auxiliary_seconds=charged,
    combined_used_seconds={**budget['used_seconds'], 'auxiliary':budget['used_seconds']['auxiliary']+charged},
    check_source=dict(path=str(Path(__file__)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
write('check.json',check)
write('summary.json',dict(methods=summary, H_minus= differences, all_metrics=metric_means))
write('transfer_manifest.json',dict(files=list(RECEIVE.values()), count=len(RECEIVE), bytes=sum(r['bytes'] for r in RECEIVE.values())))
print(json.dumps({k:check[k] for k in ('result_status','completed','elapsed_campaign_seconds','complete_training','complete_scoring','score_matrices','result_check_seconds','combined_used_seconds')},indent=2))
print('TRANSFER',len(RECEIVE),sum(r['bytes'] for r in RECEIVE.values()))
