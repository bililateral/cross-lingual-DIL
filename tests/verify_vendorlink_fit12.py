"""One-off Linux CPU checks for the FIT12 protocol delta, using synthetic data."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent/'workspace/scripts'))
import step35_vendorlink_account_evaluate as ev
from step35_vendorlink_account_control import pipeline, make_jobs
from step35_vendorlink_account_policy import tasks, CAPS, LIMITS, SEEDS, CELLS, summarize
from step35_vendorlink_account_run import restore
from step35_vendorlink_account_io import record, write

started = time.monotonic()
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
process = psutil.Process()
process.cpu_affinity(process.cpu_affinity()[:2])
torch.manual_seed(71)
selection = dict(H='H_CE_RMS', V='V1')
calls = []


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.settings = SimpleNamespace(method='H_SEQ')
        self.encoder = torch.nn.Linear(3, 3, bias=False)
        self.head = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1))

    def records(self, group, check):
        check()
        calls.append((float(self.encoder.weight[0, 0]), len(group.views[0].records)))
        x = torch.tensor([json.loads(r.title) for v in group.views for r in v.records])
        return torch.nn.functional.normalize(self.encoder(x), dim=1)


def reference(model, q, g):
    out = torch.empty(len(q), len(g), dtype=torch.float64)
    with torch.no_grad():
        for i, left in enumerate(q):
            for j, right in enumerate(g):
                pairs = torch.stack([model.head(torch.cat(((a-b).abs(), a*b))).reshape(())
                                     for a in left for b in right]).double()
                out[i, j] = .5*pairs.mean()+.5*torch.log(torch.exp(pairs).mean())
    return out.numpy()


def rejection(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError('Required invalid-input rejection did not happen')


checks = []
with tempfile.TemporaryDirectory(prefix='fit12_cpu_') as temp:
    root = Path(temp)
    model = Tiny().eval()
    q = torch.randn(4, 2, 3)
    g = torch.randn(4, 12, 3)
    actual, expected = ev.rectangular(model, q, g), reference(model, q, g)
    error = float(np.max(np.abs(actual-expected)))
    assert error < 2e-6
    changed = g.clone(); changed[:, 11] += 3
    assert np.max(np.abs(ev.rectangular(model, q, changed)-actual)) > 1e-5
    vendor = SimpleNamespace(settings=SimpleNamespace(method='VendorLink_SEQ'))
    v = ev.rectangular(vendor, q, g)
    vr = np.array([[2*float(a.mean(0)@b.mean(0))/(float(a.mean(0).square().sum())+
                   float(b.mean(0).square().sum())) for b in g] for a in q])
    assert np.max(np.abs(v-vr)) < 2e-6
    checks.append('independent_H_and_V_FIT12_formulas_and_twelfth_record')

    manifest = dict(files={}, counts={})
    for di, domain in enumerate(('AlphaBay', 'Dream', 'Agora')):
        manifest['counts'][domain] = dict(eligible_accounts=4)
        for role, count in (('fit', 12), ('dev', 2), ('confirm', 2)):
            rows = []
            for a in range(4):
                rows.append(dict(account_uid=f'a{a}', records=[dict(uid=f'{domain}/{role}/{a}/{j}',
                    title=json.dumps([.1+a+.1*j, .4+di+.17*j, .3+a*.7+.13*j]), description='handwritten')
                    for j in range(count)]))
            path = root/f'{domain}_{role}.json'; write(path, rows)
            manifest['files'][domain+':'+role] = record(path)
    task = tasks('P', selection)[0]
    identity = dict(test='handwritten_only', task=task)
    binding = dict(test='no_pretrained_model')
    endpoints = {}
    initial = copy.deepcopy(model.state_dict())
    for stage in range(4):
        model.load_state_dict(initial)
        with torch.no_grad():
            model.encoder.weight.copy_(torch.tensor([[1.+stage, .2, .1], [.3, 1., .1+stage], [.1, .2, 1.]]))
        value = dict(identity=identity, task=task, model_binding=binding, stage=stage,
                     encoder=model.encoder.state_dict(), head=model.head.state_dict(), vendor_heads={})
        path = root/f'stage{stage}.pt'; torch.save(value, path); endpoints[str(stage)] = record(path)
    def factory(_binding, _task):
        m = Tiny(); m.load_state_dict(initial)
        with torch.no_grad():
            m.encoder.weight.copy_(torch.tensor([[1., .2, .1], [.3, 1., .1], [.1, .2, 1.]]))
        return m
    def restored(b, t, p, i, s):
        return restore(b, t, p, i, s, factory=factory)
    completion = dict(status='TRAINING_COMPLETE_EVALUATION_CLOSED', identity=identity, endpoints=endpoints)
    calls.clear()
    with patch.object(ev, 'restore', restored):
        scored = ev.score(task, completion, manifest, binding, root/'dev_score', 'dev', lambda: None)
    used = list(calls)
    assert len(used) == 18 and [n for _, n in used] == [2, 12]*9
    assert [x for x, n in used if n == 12] == [1., 2., 2., 3., 3., 3., 4., 4., 4.]
    assert np.max(np.abs(np.load(root/'dev_score/H_SEQ_R31.npy')-np.load(root/'dev_score/H_SEQ_R11.npy'))) > 1e-5
    for stage, j in CELLS:
        d = task['domains'][j-1]
        _, query = ev.load_role(manifest['files'][d+':dev'], d, 'dev', 4)
        _, gallery = ev.load_role(manifest['files'][d+':fit'], d, 'fit', 4)
        m, _ = restored(binding, task, endpoints[str(stage)]['path'], identity, stage)
        qz, gz = ev.encode(m, query), ev.encode(m, gallery)
        assert np.max(np.abs(np.load(root/f'dev_score/H_SEQ_R{stage}{j}.npy')-reference(m, qz, gz))) < 2e-6
    ev.validate_scores(scored)
    analyzed = ev.analyze([scored], [task], root/'analysis')
    assert set(analyzed) == {'H_SEQ'}
    assert len(json.loads((root/'analysis/metrics.json').read_text())) == 9
    checks.append('actual_restore_score_analyze_all_nine_cells_current_encoder')

    with patch.object(ev, 'restore', restored):
        confirmed = ev.score(task, completion, manifest, binding, root/'confirm_score', 'confirm', lambda: None)
    ev.validate_scores(confirmed)
    missing = copy.deepcopy(scored); del missing['files']['H_SEQ:R21']
    rejection(lambda: ev.analyze([missing], [task], root/'must_not_open'))
    assert not (root/'must_not_open').exists()
    wrong = copy.deepcopy(manifest)
    fp = Path(manifest['files']['AlphaBay:fit']['path'])
    rows = json.loads(fp.read_text()); query = json.loads(Path(manifest['files']['AlphaBay:dev']['path']).read_text())
    rows[0]['records'][11]['uid'] = query[0]['records'][0]['uid']
    path = root/'overlap.json'; write(path, rows); wrong['files']['AlphaBay:fit'] = record(path)
    rejection(lambda: ev.score(task, completion, wrong, binding, root/'overlap_score', 'dev', lambda: None))
    rows.reverse(); path = root/'order.json'; write(path, rows)
    rejection(lambda: ev.load_role(record(path), 'AlphaBay', 'fit', 4))
    checks.append('confirm_role_and_missing_cell_overlap_class_order_rejections')

    def row(seed, old):
        return dict(seed=seed, **summarize(dict(R01=.1, R11=.85, R12=.15, R21=.82,
                    R22=.86, R23=.15, R31=old, R32=old+.01, R33=.88)))
    p = {m: [row(s, old) for s in SEEDS] for m, old in (('H_SEQ', .79), ('V_CLS', .75), ('V_SET', .76))}
    m = {key: [row(s, old) for s in SEEDS] for key, old in (('H_ER', .81), ('H_full', .84))}
    baseline = dict(status='PASS_REFERENCE_DIAGNOSTIC', checks={d: dict(candidate_pass=True) for d in task['domains'][:2]})
    def invoke(p_rows, m_rows, c_rows=None):
        phases, decisions = [], {}
        def run(phase, selected):
            assert selected == selection
            phases.append(phase)
            return {'P': p_rows, 'M': m_rows, 'C': {**p_rows, **m_rows} if c_rows is None else c_rows}[phase]
        result = pipeline(run, lambda name, value: decisions.update({name: value}), baseline)
        return result, phases, decisions
    result, phases, decisions = invoke(p, m)
    assert result['status'] == 'CONFIRMATION_PASS' and phases == ['P', 'M', 'C']
    assert decisions['C']['old_score_increment']
    no_f = copy.deepcopy(p); no_f['H_SEQ'] = [row(s, .85) for s in SEEDS]
    result, phases, _ = invoke(no_f, m)
    assert result['status'] == 'STOP_NO_MEANINGFUL_FORGETTING' and phases == ['P'] and not result['application']
    weak_m = copy.deepcopy(m); weak_m['H_full'] = [row(s, .79) for s in SEEDS]
    result, phases, _ = invoke(p, weak_m)
    assert result['status'] == 'STOP_NO_H_ADVANTAGE' and phases == ['P', 'M'] and result['application']
    incomplete = copy.deepcopy(p); incomplete['H_SEQ'].pop()
    rejection(lambda: invoke(incomplete, m))
    checks.append('real_pipeline_fixed_selection_P_stop_M_stop_C_and_missing_seed')

    specs = tasks('P', selection)+tasks('M', selection)
    assert len(specs) == 12 and sum(t['stages'] for t in specs) == 36
    assert all(t['epochs'] == 40 and not t['settings']['diagnostics'] for t in specs)
    assert all(t['settings']['ce_scale_mode'] == 'rms' for t in specs if t['method'].startswith('H'))
    rejection(lambda: tasks('D', selection))
    rejection(lambda: tasks('P', dict(H='H_base', V='V1')))
    assert CAPS == dict(training=288000, gpu=14400, auxiliary=14400)
    assert LIMITS['allocator_bytes'] is None and LIMITS['process_gpu_bytes'] is None
    assert LIMITS['disk_bytes'] == 32*2**30
    jobs = make_jobs(root, 'P', tasks('P', selection), 'train')
    assert [j['seconds'] for j in jobs] == [21600, 14400]*3
    assert all(j['seconds'] == 36000 for j in make_jobs(root, 'M', tasks('M', selection), 'train'))
    checks.append('actual_fixed_task_generation_and_budget_reservations')

print(json.dumps(dict(status='PASS', tests=len(checks), errors=0, failures=0, checks=checks,
    max_independent_H_error=error, score_cells=9, fit_encoder_calls=9,
    elapsed_seconds=time.monotonic()-started, final_rss_bytes=process.memory_info().rss,
    gpu_used=False, formal_data_used=False), sort_keys=True))
