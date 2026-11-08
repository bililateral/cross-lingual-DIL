"""Limited Linux CPU evidence for changed Registered paths; handwritten inputs only."""
import argparse
from dataclasses import asdict
from datetime import date, timedelta
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from step29_vendorlink_data import Ad, View, Group
from step29_vendorlink_methods import Model, Settings
from step30_evolution_run import execute, restore
from step30_evolution_io import Ledger, read, write, record
from step30_evolution_control import Runner
from step31_evolution_registered_policy import DOMAINS, CAPS, CELLS, tasks, cell_key, decide
from step31_evolution_registered_data import partition
from step31_evolution_registered_evaluate import encode_views, rectangular, query_metrics, validate_scores, analyze


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, hidden_dropout_prob=0., initializer_range=.02)
        self.embedding = nn.Embedding(128, 4)

    def forward(self, input_ids, attention_mask, **kwargs):
        z = self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=z, pooler_output=z.mean(1), hidden_states=(z,)+tuple(z*k for k in (1, 2, 3, 4)))


def tokenizer(texts, **kwargs):
    rows = [[ord(c) % 128 for c in t[:12]] for t in texts]
    n = max(map(len, rows))
    return dict(input_ids=torch.tensor([r+[0]*(n-len(r)) for r in rows]),
                attention_mask=torch.tensor([[1]*len(r)+[0]*(n-len(r)) for r in rows]))


def fixture_rows():
    result = []
    for i in range(64):
        day = date(2020, 1, 1)+timedelta(days=i % 10)
        records = [dict(lid=f'{i}:{j}', vid=f'fixture{i}', observed=str(day+timedelta(days=(0 if j < 4 else 20 if j < 6 else 40))),
                        title=f'title {i} {j}', body=f'handwritten text {i} {j}', body_key=f'body{i}:{j}') for j in range(8)]
        prior = 'test' if i < 16 else 'unused' if i < 32 else 'fit' if i < 56 else 'dev'
        result.append(dict(vid=f'fixture{i}', prior_usage=prior,
                           roles=dict(fit=records[:4], gallery=records[4:6], query=records[6:])))
    return result


def main(root):
    if sys.platform != 'linux':
        raise RuntimeError('Linux-only handwritten CPU evidence')
    root.mkdir(parents=True)
    started = time.monotonic()
    passed = []
    rows = fixture_rows()
    exports, qrels, language, selection = partition(rows, quota=8, confirmation_count=24)
    assert partition(list(reversed(rows)), quota=8, confirmation_count=24) == (exports, qrels, language, selection)
    assert set(selection['reserved_ids']).isdisjoint(selection['development_ids'])
    assert not any(x in selection['development_ids'] for x in [f'fixture{i}' for i in range(16)])
    assert selection['development_records'] == 192
    for bad in ('dates', 'duplicate'):
        changed = json.loads(json.dumps(rows))
        for r in changed:
            if bad == 'dates':
                r['roles']['query'][0]['observed'] = r['roles']['gallery'][0]['observed']
            else:
                r['roles']['query'][0]['body_key'] = r['roles']['fit'][0]['body_key']
        try:
            partition(changed, quota=8, confirmation_count=24)
            raise AssertionError('Expected invalid role rejection')
        except ValueError:
            pass
    passed.append('deterministic_reservation_roles_dates_and_duplicates')
    for family in ('H_SEQ', 'VendorLink_SEQ'):
        task = dict(id='handwritten_'+family, stages=3, epochs=1, seed=31,
                    system='V_SEQ' if family == 'VendorLink_SEQ' else family,
                    settings=asdict(Settings(method=family, pooling='mean_last4_sum')))
        def factory(_):
            torch.manual_seed(123)
            return Model(Encoder(), tokenizer, Settings(**task['settings']))
        model = factory(task)
        before = model.encoder.embedding.weight.detach().clone()
        calls = []
        def loader(domain):
            calls.append(domain)
            return [Group.from_dict(g) for g in exports[domain]['fit']]
        job = root/family
        complete = execute(model, loader, job, task, {'fixture': True}, {'fixture': True}, write, record,
                           expected_groups=1, domains=DOMAINS)
        assert calls == list(DOMAINS) and complete['updates'] == 3
        assert complete['workload']['history_groups'] == 0 and complete['workload']['stage_model_saves'] == 3
        assert not torch.equal(before, model.encoder.embedding.weight)
        restored = restore(factory, task, job/'stage3.pt', {'fixture': True}, {'fixture': True}, 3)
        first = Group.from_dict(exports['E1']['fit'][0]).blind().views[:2]
        q, g = encode_views(restored, first[:1], lambda: None), encode_views(restored, first[1:], lambda: None)
        actual = rectangular(restored, q, g)[0, 0]
        pair = Group('pair', 'E1', 'dev', first)
        reference = restored.score(pair).numpy()[0]
        assert np.isclose(actual, reference, rtol=1e-6, atol=1e-6), (actual, reference)
        # Independent scalar aggregation catches accidental sigmoid/ordinary cosine substitutions.
        with torch.no_grad():
            if family == 'H_SEQ':
                logits = [float(restored.head(torch.cat(((a-b).abs(), a*b))).item()) for a in q[0] for b in g[0]]
                reference2 = .5*np.mean(logits)+.5*(np.log(np.exp(np.array(logits)-max(logits)).sum())+max(logits)-np.log(4))
            else:
                u, v = q[0].numpy().mean(0), g[0].numpy().mean(0)
                reference2 = 2*np.dot(u, v)/(np.dot(u, u)+np.dot(v, v))
        assert np.isclose(actual, reference2, rtol=1e-6, atol=1e-6)
        passed.append(family+'_actual_three_domain_updates_compact_restore_rectangular_equivalence')
    sample = dict(query_ids=['q1', 'q2'], gallery_ids=['a', 'b', 'c'], scores=[[.1, .2, .2], [.2, .2, .1]])
    measured = query_metrics(sample, {'q1': 'c', 'q2': 'a'})
    assert measured['ranks'] == [2, 1] and measured['mean']['map'] == .75
    try:
        query_metrics(sample, {'q1': 'missing', 'q2': 'a'})
        raise AssertionError('Expected qrel mismatch rejection')
    except ValueError:
        pass
    base = []
    for task in tasks():
        cells = {}
        for t, d, pool in CELLS:
            value = .65 if t < d else .7
            if t == 3 and d < 3:
                value = .62
            cells[cell_key(t, d, pool)] = dict(mean={'map': value, 'recall@1': .5})
        base.append(dict(task=task, cells=cells))
    result = decide(base)
    assert result['premise_passed'] and result['automatic_next_phase'] is False
    for row in base:
        if row['task']['system'] == 'V_SEQ':
            for d in (1, 2):
                row['cells'][cell_key(3, d, d)]['mean']['map'] = .72
    assert decide(base)['status'] == 'STOP_NO_JOINT_FORGETTING'
    try:
        analyze(root/'closed_labels', {'specs': tasks(), 'completed': []}, {}, [], lambda: None)
        raise AssertionError('Missing training must block labels')
    except ValueError:
        assert not (root/'closed_labels/access').exists()
    passed.append('metric_alignment_ties_complete_batch_gate_and_joint_premise')
    # Real CPU subprocesses exercise inherited failure isolation under the NEW ledger caps.
    dispatch = root/'dispatch'; dispatch.mkdir()
    ledger = Ledger(dispatch, dict(training=0, gpu=0, auxiliary=352), 0, caps=CAPS)
    class FixtureRunner(Runner):
        def command(self, job, slot, deadline):
            code = "import time,sys; time.sleep(float(sys.argv[1])); sys.exit(int(sys.argv[2]))"
            return [sys.executable, '-c', code, '.12' if job['target'].name == '0' else '.35',
                    '1' if job['target'].name == '0' else '0']
    runner = FixtureRunner(dict(experiment_root=str(dispatch), workspace=str(root)), root/'unused.json', dispatch,
                           ledger, worker_entry=Path(__file__), caps=CAPS)
    jobs = []
    for i in range(4):
        target = dispatch/str(i); target.mkdir()
        jobs.append(dict(action='fixture', target=target, bucket='auxiliary', seconds=30, extra=[]))
    try:
        runner.run_many(jobs, 2, lambda job: None)
        raise AssertionError('Failed slot must fail the whole phase')
    except RuntimeError:
        state = read(dispatch/'dispatch.json')
        assert state['status'] == 'INCOMPLETE' and state['retired'] == [0] and not state['active']
        assert all(read(dispatch/str(i)/'dispatch_receipt.json')['success'] for i in (1, 2, 3))
        assert all(read(dispatch/str(i)/'process.json')['slot'] == 1 for i in (1, 2, 3))
        assert ledger.caps == CAPS and ledger.used['auxiliary'] > 352
    passed.append('actual_dual_cpu_children_failed_slot_retires_healthy_continues_new_caps')
    write(root/'receipt.json', dict(status='PASS', fixture_only=True, pretrained_loaded=False, gpu=False,
          checks=passed, elapsed_seconds=time.monotonic()-started,
          scope='Changed data/rectangle/domain/budget wiring only; no full historical regression'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
