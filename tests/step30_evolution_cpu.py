"""Bounded one-off Evolution integration qualification, handwritten inputs only.

Run only with Linux py310, CUDA hidden, an exact source manifest and fresh output.
No official archive, pretrained model, real texts, qrels or campaign are opened.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import step30_evolution_control as control
import step30_evolution_data as data
import step30_evolution_evaluate as evaluate
import step30_evolution_io as io
import step30_evolution_policy as policy
import step30_evolution_run as run
from step29_vendorlink_data import Ad, Group, View, seed_for
from step29_vendorlink_methods import Model, Settings, Memory, make_optimizer


def must_fail(call, message):
    try:
        call()
    except (ValueError, RuntimeError, FileExistsError, TimeoutError, KeyError):
        return
    raise AssertionError('Expected rejection: '+message)


def group(d, index=0, split='fit'):
    return Group(f'{d}_{index}', d, split, tuple(
        View(f'{d}_{index}_{a}_{v}', tuple(Ad(f'{d}_{index}_{a}_{v}_{r}',
             ('Apple' if a % 2 else 'apple')+' '+str(a),
             'the and with offer '+str((a+index+v+r) % 8)) for r in range(2)),
             f'{d}_{index}_{a}') for a in range(8) for v in range(2)))


def test_data(temp, qualification):
    records = [data.Record(f'{d}_v{v}', f'{d}_{v}_{r}', '1', day, r+1, 'fixture',
                          'Apple title', 'the and with sample text', f'body-{d}-{v}-{r}')
               for d, day in zip(policy.DOMAINS, ('2014-07-01', '2014-10-01', '2015-01-01'))
               for v in range(26) for r in range(4)]
    exported, custody = data.partition(records, dict(fit=8, dev=8, test=8))
    reversed_export, _ = data.partition(list(reversed(records)), dict(fit=8, dev=8, test=8))
    assert exported == reversed_export
    assert len(exported) == 9 and len(custody['mapping']) == 288
    for (d, split), groups in exported.items():
        assert len(groups) == 1
        data.validate_group(groups[0], split == 'fit')
        if split != 'fit':
            truth = custody['qrels'][d, split][groups[0].uid]
            assert sum(truth['labels']) == 8 and truth['view_ids'] == [v.uid for v in groups[0].views]
    eligible = {policy.DOMAINS[0]: {'a', 'b'}, policy.DOMAINS[1]: {'a'}, policy.DOMAINS[2]: {'b', 'c'}}
    matched = data.match_accounts(eligible, 1)
    assert len({v for vs in matched.values() for v in vs}) == 3 and matched[policy.DOMAINS[1]] == ['a']
    must_fail(lambda: data.match_accounts({d: {'a'} for d in policy.DOMAINS}, 1), 'capacity no retry')
    base, _ = data.qualification_modules(qualification, lambda: None)
    earliest = {}
    original = dict(vid='v', lid='l', mscrape_id='1', title='Apple', description='The AND with Case')
    later = dict(original, mscrape_id='2', title='apple', description='the and with case')
    dates = {'1': '2014-07-01', '2': '2014-10-01'}
    data.remember_snapshot(earliest, later, 1, dates, base, {})
    data.remember_snapshot(earliest, original, 2, dates, base, {})
    data.remember_snapshot(earliest, dict(original, title='APPLE'), 3, dates, base, {})
    selected = earliest['v', 'l']
    assert selected[1:3] == ('1', 2) and selected[4:] == ('Apple', 'The AND with Case')
    samples = {d: [dict(id=str(i)) for i in range(32)] for d in policy.DOMAINS}
    decisions = {d: {str(i): 'english' if i < 29 else 'unclear' for i in range(32)} for d in policy.DOMAINS}
    assert data.language_result(samples, decisions)['passed']
    decisions[policy.DOMAINS[1]]['0'] = 'other'
    assert not data.language_result(samples, decisions)['passed']
    return dict(exported_groups=9, records=288, minimum_four_not_twenty=True,
                same_snapshot_source_row=selected[2], language_boundary=[29, 28])


def tiny_factory(temp):
    model_dir = temp/'tiny_model'
    model_dir.mkdir(parents=True)
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'Apple', 'apple', 'the', 'and', 'with', 'offer']+[str(i) for i in range(8)]
    (model_dir/'vocab.txt').write_text('\n'.join(vocab)+'\n')
    tokenizer = BertTokenizer(vocab_file=str(model_dir/'vocab.txt'), do_lower_case=False)
    tokenizer.save_pretrained(model_dir)
    torch.manual_seed(871)
    encoder = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=4,
                                  num_attention_heads=2, intermediate_size=12, hidden_dropout_prob=0.,
                                  attention_probs_dropout_prob=0., max_position_embeddings=512))
    encoder.save_pretrained(model_dir)
    binding = dict(directory=str(model_dir), files=[io.record(p) for p in sorted(model_dir.iterdir())])
    def factory(task):
        return run.make_model(model_dir, task, 'cpu', expected_hidden=8)
    return factory, binding


def test_updates(temp):
    factory, binding = tiny_factory(temp/'updates_fixture')
    task = policy.spec('B', 'H_SEQ', 'H3', 13)
    m = factory(task)
    assert m.tokenizer(['Apple', 'apple'])['input_ids'][0] != m.tokenizer(['Apple', 'apple'])['input_ids'][1]
    initial_embedding = m.encoder.embeddings.word_embeddings.weight.detach().clone()
    reports, histories, current_h, v_current = {}, {}, {}, {}
    for system in ('H_SEQ', 'H_ER', 'H_full', 'V_SEQ', 'V_ER'):
        task = policy.spec('B', system, 'V0' if system.startswith('V') else 'H3', 13)
        task = dict(task, epochs=2)  # Reduced exposure ONLY in this handwritten internal API test.
        m = factory(task)
        initial_scores = m.score(group(policy.DOMAINS[0]).blind()).clone()
        requests = []
        def loader(domain):
            assert domain == policy.DOMAINS[len(requests)]
            requests.append(domain)
            return [group(domain)]
        identity = dict(fixture=True, system=system)
        result = run.execute(m, loader, temp/system, task, identity, binding, io.write, io.record, expected_groups=1)
        assert requests == list(policy.DOMAINS) and result['updates'] == 6
        assert result['workload']['stage_model_saves'] == 3
        assert not (temp/system/'latest.pt').exists()
        assert len(list((temp/system).glob('*.pt'))) == 4
        assert not torch.equal(m.encoder.embeddings.word_embeddings.weight, initial_embedding)
        start = run.restore(factory, task, Path(result['endpoints']['0']['path']), identity, binding, 0)
        assert torch.equal(start.score(group(policy.DOMAINS[0]).blind()), initial_scores)
        values = []
        for t, d in policy.CELLS:
            restored = run.restore(factory, task, Path(result['endpoints'][str(t)]['path']), identity, binding, t)
            g = group(policy.DOMAINS[d-1]).evaluation_order()
            scores = restored.score(g.blind())
            values.append(float(scores.mean()))
            metrics = evaluate.query_metrics(scores.numpy(), g.labels(), [v.uid for v in g.views])
            assert len(metrics) == 10 and metrics['map'] == metrics['mrr']
            assert metrics['recall@1'] == metrics['ndcg@1']
        logs = [json.loads(x) for x in (temp/system/'updates.jsonl').read_text().splitlines()]
        histories[system] = [x['history_group'] for x in logs]
        (v_current if system.startswith('V') else current_h)[system] = [x['current_ids'] for x in logs]
        if system == 'H_full':
            assert max(x.get('history', {}).get('distillation', 0) for x in logs) > 0
        if system == 'V_ER':
            assert len(m.classes) == 24 and all('history_ce' in x for x in logs[2:])
            assert result['workload']['teacher_groups'] == 0
        for stage in (1, 2, 3):
            state = torch.load(temp/system/f'stage{stage}.pt', weights_only=False)
            assert set(state) == {'identity', 'task', 'model_binding', 'stage', 'updates', 'encoder', 'head'}
        reports[system] = dict(workload=result['workload'], evaluated_cells=len(values), initial_restored_exactly=True)
    assert histories['H_ER'] == histories['H_full'] == histories['V_ER']
    assert current_h['H_SEQ'] == current_h['H_ER'] == current_h['H_full']
    assert v_current['V_SEQ'] == v_current['V_ER']
    # A genuinely completes one stage, without interrupting a three-stage run.
    a = dict(policy.spec('A', 'H_SEQ', 'H0', 11), epochs=1)
    a_result = run.execute(factory(a), lambda d: [group(d)], temp/'single', a, {'single': True}, binding,
                           io.write, io.record, expected_groups=1)
    assert a_result['updates'] == 1 and set(a_result['endpoints']) == {'0', '1'}
    # Compare the actual V_ER accumulated gradient to an independent one-graph
    # CE_current + .1*CE_history reference; remove history to prove sensitivity.
    task = policy.spec('B', 'V_ER', 'V0', 13)
    m = factory(task)
    cur, hist = group(policy.DOMAINS[1]), group(policy.DOMAINS[0])
    classes = sorted({v.identity for g in (cur, hist) for v in g.views})
    m.start_vendor_stage(classes, 811)
    reference, ablated = copy.deepcopy(m), copy.deepcopy(m)
    examples = [(r, v.identity) for v in cur.views for r in v.records]
    captured = {}
    optimizer = make_optimizer(m)
    original_step = optimizer.step
    def capture_step():
        captured.update({n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None})
        original_step()
    optimizer.step = capture_step
    run.update_vendor(m, optimizer, examples, hist, 771, lambda: None)
    def reference_gradient(model, weight):
        model.train()
        model.zero_grad()
        lookup = {v: k for k, v in enumerate(model.classes)}
        terms = []
        for g, w in ((cur, 1.), (hist, weight)):
            records = [(r, v.identity) for v in g.views for r in v.records]
            output, _ = model.encode([r.title+' [SEP] '+r.description for r, _ in records], 512)
            labels = torch.tensor([lookup[v] for _, v in records])
            terms.append(w*torch.nn.functional.cross_entropy(model.classifier(output.pooler_output), labels))
        sum(terms).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        return {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    expected, without = reference_gradient(reference, .1), reference_gradient(ablated, 0.)
    max_error = max(float((captured[n]-expected[n]).abs().max()) for n in expected)
    history_delta = max(float((captured[n]-without[n]).abs().max()) for n in expected if n.startswith('encoder.'))
    assert max_error < 2e-6 and history_delta > 1e-7
    # Algorithm R teachers are born only for surviving new groups and never refreshed.
    hsettings = Settings(method='H_full')
    hmem, vmem = Memory(321, hsettings), run.VendorMemory(321, Settings(method='VendorLink_SEQ'))
    class Teacher:
        def __init__(self): self.calls = []
        def teacher(self, g, check):
            self.calls.append(g.uid)
            return torch.full((496,), float(len(self.calls)))
    teacher = Teacher()
    old_teachers = {}
    for stage in (1, 2):
        groups = [group(policy.DOMAINS[stage-1], i) for i in range(20)]
        hmem.retain(groups, stage, teacher)
        vmem.retain(groups, stage)
        assert [r['group']['uid'] for r in hmem.entries] == [r['group']['uid'] for r in vmem.entries]
        for row in hmem.entries:
            uid = row['group']['uid']
            if uid in old_teachers: assert torch.equal(old_teachers[uid], row['teacher'])
            old_teachers[uid] = row['teacher'].clone()
        assert len(hmem.bytes()) < 1048576 and len(vmem.bytes()) < 1048576
    assert len(teacher.calls) <= 12 and len(teacher.calls) == len(set(teacher.calls))
    small = run.VendorMemory(1, replace(Settings(method='VendorLink_SEQ'), memory_bytes=1))
    must_fail(lambda: small.retain([cur], 1), 'real serialized memory cap')
    return dict(trajectories=reports, single_stage_complete=True, cased_token_ids_differ=True,
                vendor_gradient_max_error=max_error, encoder_history_gradient_difference=history_delta,
                paired_history_draws=True, teacher_calls=len(teacher.calls), teacher_refreshes=0)


def test_evaluation(temp):
    factory, binding = tiny_factory(temp/'evaluation_fixture')
    root = temp/'evaluation'
    root.mkdir()
    manifest = dict(files={})
    for d in policy.DOMAINS:
        g = group(d, split='test').evaluation_order()
        path, truth = root/(d+'_blind.json'), root/(d+'_qrels.json')
        io.write(path, [asdict(g.blind())])
        io.write(truth, {g.uid: dict(view_ids=[v.uid for v in g.views], labels=g.labels())})
        manifest['files'][d+':test'] = io.record(path)
        manifest['files'][d+':test:qrels'] = io.record(truth)
    specs = [dict(policy.spec('D', m, 'H3' if m == 'H_full' else 'V0', 21), epochs=2) for m in ('H_full', 'V_ER')]
    batch = dict(phase='D', specs=specs, selected=dict(H='H3', V='V0'), completed=[])
    for task in specs:
        identity = dict(fixture=True, system=task['system'])
        job = root/task['id']
        result = run.execute(factory(task), lambda d: [group(d)], job, task, identity,
                             binding, io.write, io.record, expected_groups=1)
        evaluate.training_complete(task, result, identity, fit_groups=1)
        batch['completed'].append(dict(task_id=task['id'], identity=identity, completion=io.record(job/'completion.json')))
    blind = evaluate.score_batch(root, batch, manifest, factory, binding, lambda: None,
                                 fixture_specs=specs, fixture_groups=1)
    assert blind['forwards'] == 18
    label_access = root.parent/'access/D_qrels.json'
    assert not label_access.exists()
    first = blind['runs'][0]['cells']['0_1']
    path = Path(first['path'])
    good = io.read(path)
    bad = copy.deepcopy(good)
    next(iter(bad.values()))['scores'].pop()
    io.write(path, bad)
    blind['runs'][0]['cells']['0_1'] = io.record(path)
    io.write(root/'blind_complete.json', blind)
    must_fail(lambda: evaluate.analyze_batch(root, batch, manifest, fixture_specs=specs, fixture_groups=1), 'no label open after incomplete blind output')
    assert not label_access.exists()
    io.write(path, good)
    blind['runs'][0]['cells']['0_1'] = io.record(path)
    io.write(root/'blind_complete.json', blind)
    metrics = evaluate.analyze_batch(root, batch, manifest, fixture_specs=specs, fixture_groups=1)
    assert len(metrics) == 2 and all(len(r['cells']) == 9 for r in metrics) and label_access.exists()
    must_fail(lambda: evaluate.analyze_batch(root, batch, manifest, fixture_specs=specs, fixture_groups=1), 'no second D label open')
    return dict(actual_trajectories=2, blind_cells=18, label_open_after_all_scores=True,
                missing_score_blocks_open=True, second_open_rejected=True, metrics_per_cell=10)


def fake_row(task, P=(.3, .3, .3), A=(.7, .7, .7), final=(.6, .6, .7), groups=12):
    # Paper metrics for decision tests only: never claimed to be model output.
    entries = {}
    for t, d in policy.cells(task['stages']):
        value = P[d-1] if t == d-1 else (A[d-1] if t == d else final[d-1])
        query = {m: [value if m not in ('recall@1', 'ndcg@1') else .5]*16 for m in policy.METRICS}
        entries[f'{t}_{d}'] = dict(mean={m: float(np.mean(v)) for m, v in query.items()},
             groups=[dict(uid=f'd{d}g{g}', view_ids=[str(i) for i in range(16)], queries=copy.deepcopy(query)) for g in range(groups)])
    return dict(spec=task, cells=entries)


def test_decisions(temp):
    selected = dict(H='H3', V='V0')
    assert [len(policy.tasks(p, None if p == 'A' else selected)) for p in 'ABCD'] == [16, 6, 12, 18]
    assert [len(policy.tasks(p, dict(H='H0', V='V0'))) for p in 'BCD'] == [6, 9, 15]
    a = [fake_row(s) for s in policy.tasks('A')]
    assert policy.select_a(a)['selected'] == {'H': 'H0', 'V': 'V0'}
    must_fail(lambda: policy.select_a(a[:-1]), 'missing A seed')
    b = [fake_row(s) for s in policy.tasks('B', selected)]
    assert policy.decision(b, selected, 'B')['passed']
    failed_b = [fake_row(s, final=(.7, .7, .7)) if s['system'] == 'V_SEQ' else fake_row(s) for s in policy.tasks('B', selected)]
    assert not policy.decision(failed_b, selected, 'B')['passed']
    def summary(p, a):
        pp, aa = np.full((3, 3), p), np.full((3, 3), a)
        return dict(P=pp, A=aa, G=aa-pp)
    cases = [(.5, .7, .7, .8, True), (.6, .65, .9, .65, False),
             (.5, .7, .68, .69, False), (.5, .7, .5, .69, True), (.5, .7, .8, .8, False)]
    for pre, after, pf, af, expected in cases:
        assert policy.learning_paths(summary(pf, af), summary(pre, after))['passed'] == expected
    mixed = summary(.7, .8)
    mixed['G'][1, 2] = -.001
    assert not policy.learning_paths(mixed, summary(.5, .7))['passed']
    mixed = summary(.7, .8)
    mixed['A'][0:3, 2] = .69
    mixed['G'] = mixed['A']-mixed['P']
    assert not policy.learning_paths(mixed, summary(.5, .7))['passed']
    def mechanism_rows(phase):
        result = []
        for s in policy.tasks(phase, selected):
            if s['system'] == 'H_full':
                result.append(fake_row(s, P=(.5, .5, .5), A=(.72, .72, .72), final=(.71, .71, .72)))
            elif s['system'] in ('H_ER', 'V_ER'):
                result.append(fake_row(s, final=(.65, .65, .7)))
            else:
                result.append(fake_row(s))
        return result
    c = mechanism_rows('C')
    assert policy.decision(b+c, selected, 'C')['passed']
    bad_c = [fake_row(r['spec']) if r['spec']['system'] == 'H_full' else r for r in c]
    assert not policy.decision(b+bad_c, selected, 'C')['passed']
    # Dispatch through the actual common flow: stop B prevents C and D calls.
    calls = []
    def phase_stop(phase, choice):
        calls.append(phase)
        if phase == 'A': return a
        # A ties choose H0; construct exact corresponding B specs.
        return [fake_row(s, final=(.7, .7, .7)) if s['system'] == 'V_SEQ' else fake_row(s)
                for s in policy.tasks('B', choice)]
    result = control.pipeline(phase_stop, lambda *args: None)
    assert calls == ['A', 'B'] and result['status'] == 'STOP_B_NO_JOINT_FORGETTING'
    # C failure stops D with exact selected configs, not an artificial skip flag.
    calls = []
    def phase_c_stop(phase, choice):
        calls.append(phase)
        if phase == 'A': return a
        return [fake_row(s) for s in policy.tasks(phase, choice)]
    result = control.pipeline(phase_c_stop, lambda *args: None)
    assert calls == ['A', 'B', 'C'] and result['status'] == 'STOP_C_FULL_MECHANISM_NOT_SUPPORTED'
    drows = mechanism_rows('D')
    d = policy.decision(drows, selected, 'D')
    assert d['passed'] and d['internal_support'] and all(d['support'].values())
    for m in policy.REFERENCES:
        for endpoint in ('F', 'O'):
            assert abs(d['intervals'][m][endpoint][0]-d['intervals'][m][endpoint][1]) < 1e-12
    noisy = copy.deepcopy(drows)
    for row in noisy:
        if row['spec']['system'] == 'V_ER':
            for key in ('3_1', '3_2'):
                for i, g in enumerate(row['cells'][key]['groups']):
                    g['queries']['map'] = [.45 if i < 6 else .85]*16
    uncertain = policy.decision(noisy, selected, 'D')
    assert uncertain['internal_support'] and not uncertain['support']['V_ER'] and not uncertain['passed']
    must_fail(lambda: policy.decision(drows[:-1], selected, 'D'), 'missing confirm seed')
    return dict(paper_cases=len(cases)+2, stop_dispatch=['A_only_capability', 'B_joint', 'C_full'],
                no_d_after_c_failure=True, paired_bootstrap_replicates=5000, all_internal_external_support_checked=True)


def test_boundaries(temp):
    path = temp/'access_check'
    io.access(path, 'D_qrels', [])
    must_fail(lambda: io.access(path, 'D_qrels', []), 'once-only test label attempt')
    gate = temp/'closed.json'
    io.write(gate, dict(experiment='Evolution_pilot_v1', approved=False))
    must_fail(lambda: io.permit(gate, 'campaign'), 'formal remains closed')
    ledger = io.Ledger(temp/'ledger', dict(training=7, gpu=3, auxiliary=50), 10)
    ledger.charge('training', 'failed', 1.1, False)
    assert ledger.used['training'] == 9 and ledger.cpu_tests == 10
    must_fail(lambda: ledger.reserve('training', policy.CAPS['training']), 'prior fees survive')
    must_fail(lambda: io.Budget(temp, time.monotonic()+5).check(), 'deadline actual callback')
    # Real subprocess dispatch exits at the closed gate; the same Runner must
    # retain failure costs and reject the next queued operation, not retry it.
    runner_root = temp/'runner'
    runner_root.mkdir()
    failed_job = runner_root/'failed_worker'
    failed_job.mkdir()
    runner_ledger = io.Ledger(runner_root, dict(training=0, gpu=0, auxiliary=0), 0)
    runner = control.Runner(dict(workspace=str(temp), experiment_root=str(temp)), gate, runner_root, runner_ledger)
    must_fail(lambda: runner.run('analyze', failed_job, 'auxiliary', 20, []), 'actual worker failure')
    assert runner.failed and not io.read(failed_job/'dispatch_receipt.json')['success']
    used = runner_ledger.used.copy()
    must_fail(lambda: runner.run('analyze', runner_root/'forbidden_next', 'auxiliary', 20, []), 'no next dispatch')
    assert not (runner_root/'forbidden_next').exists() and runner_ledger.used == used
    # Malformed blinded predictions are rejected before qrels loader is called.
    spec = policy.tasks('A')[0]
    score = temp/'scores.json'
    io.write(score, {'g': dict(view_ids=[str(i) for i in range(16)], scores=[0.]*119)})
    invalid = dict(specs=[spec], runs=[dict(spec=spec, cells={'0_1': io.record(score), '1_1': io.record(score)})],
                   inventory={policy.DOMAINS[0]: {'g': [str(i) for i in range(16)]}}, forwards=2)
    must_fail(lambda: evaluate.validate_blind(invalid, [spec]), 'incomplete finite blind matrix')
    # Exact rank fixture: give the sole positive higher score than all negatives.
    g = group(policy.DOMAINS[0]).evaluation_order()
    result = evaluate.query_metrics(g.labels(), g.labels(), [v.uid for v in g.views])
    assert all(all(x == 1 for x in v) for v in result.values())
    return dict(one_open_attempt=True, formal_closed=True, failed_fees_preserved=True,
                invalid_blind_rejected=True, known_ranking_correct=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--prior-seconds', type=int, default=0)
    parser.add_argument('--only', choices=('all', 'remaining', 'final', 'data', 'updates', 'evaluation', 'decisions', 'boundaries'), default='all')
    args = parser.parse_args()
    if sys.platform != 'linux' or Path(sys.executable).resolve() != Path(io.PYTHON).resolve() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Linux py310 with CUDA hidden required')
    if args.output.exists() or not 0 <= args.prior_seconds < 1200:
        raise ValueError('Fresh evidence and unspent cumulative CPU cap required')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('Cumulative 1200-second CPU limit')))
    signal.alarm(1200-args.prior_seconds)
    result = dict(status='RUNNING', official_data_opened=False, pretrained_model_opened=False, gpu=False,
                  prior_seconds=args.prior_seconds, sources=io.read(args.sources), checks={})
    try:
        for row in result['sources']:
            io.verify(row)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory(dir=args.output) as directory:
            temp = Path(directory)
            for name, fn in (('data', lambda: test_data(temp, args.qualification)),
                             ('updates', lambda: test_updates(temp)), ('evaluation', lambda: test_evaluation(temp)),
                             ('decisions', lambda: test_decisions(temp)),
                             ('boundaries', lambda: test_boundaries(temp))):
                if (args.only in ('all', name) or (args.only == 'remaining' and name in ('decisions', 'boundaries'))
                        or (args.only == 'final' and name in ('evaluation', 'decisions', 'boundaries'))):
                    result['checks'][name] = fn()
                    io.write(args.output/'progress.json', result)
                    print('PASS', name, flush=True)
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        elapsed = time.monotonic()-started
        result.update(actual_seconds=elapsed, charged_seconds=math.ceil(elapsed),
                      cumulative_cpu_seconds=args.prior_seconds+math.ceil(elapsed))
        io.write(args.output/'receipt.json', result)
        print(json.dumps({k: result[k] for k in ('status', 'charged_seconds', 'cumulative_cpu_seconds')}), flush=True)


if __name__ == '__main__':
    main()
