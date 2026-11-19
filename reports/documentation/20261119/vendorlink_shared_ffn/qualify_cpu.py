"""Synthetic-only CPU qualification through the FIT12 training and scoring entries."""
import argparse
import gc
import hashlib
import io
import json
import os
import resource
import tempfile
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

import step35_vendorlink_account_evaluate as ev
import step35_vendorlink_account_run as run
from step29_vendorlink_data import Group, assignment, seed_for
from step29_vendorlink_methods import Memory, h_objective, h_pool, make_optimizer
from step35_market_adapters import CAP, MarketH, restore_history, restore_scoring_endpoint
from step35_market_worker import market_tasks
from step35_vendorlink_account_io import pinned, record, write
from step35_vendorlink_account_policy import CELLS, DOMAINS


def noop():
    pass


def digest(items):
    h = hashlib.sha256()
    for item in items:
        array = item.detach().contiguous().cpu().numpy() if torch.is_tensor(item) else item
        h.update(array.tobytes())
    return h.hexdigest()


def fixture(root):
    rows, views = {}, {}
    manifest = dict(files={}, counts={})
    for market, count in zip(DOMAINS, (12, 4, 4)):
        manifest['counts'][market] = dict(eligible_accounts=count)
        for role, width in (('fit', 12), ('dev', 2)):
            accounts = []
            for i in range(count):
                owner = f'fixture-{market}-{i:02d}'
                ads = [dict(uid=f'{owner}-{role}-{j:02d}', title=f'{role} pencil {i} {j}',
                            description=f'paper {market} {i} blue {j}') for j in range(width)]
                accounts.append(dict(account_uid=owner, records=ads))
            path = root/f'{market}-{role}.json'
            write(path, accounts)
            source = record(path)
            manifest['files'][market+':'+role] = source
            views[market, role] = ev.load_role(source, market, role, count)[1]
            if role == 'fit':
                rows[market] = accounts
    return manifest, rows, views


def scored(model, views, domain, small=False):
    q, fit = views[domain, 'dev'], views[domain, 'fit']
    return ev.score_market(model, domain, q[:2] if small else q, fit[:2] if small else fit, noop)


def perturb_routes(model, views):
    before = {d: scored(model, views, d, True) for d in DOMAINS}
    result = {}
    for name, param in [('shared_ffn', model.shared_ffn_parameters()['output.dense.bias']),
                        ('Agora_adapter', model.projections['11.value'].banks['Agora'].up.weight)]:
        old = param.detach().clone()
        with torch.no_grad():
            param[:16].add_(0.5)
        try:
            result[name] = {}
            for domain in DOMAINS:
                after = scored(model, views, domain, True)
                diffs = [float(np.abs(np.asarray(a)-np.asarray(b)).max()) for a, b in zip(after, before[domain])]
                required = name == 'shared_ffn' or domain == 'Agora'
                assert all(x > 0 if required else x == 0 for x in diffs), (name, domain, diffs)
                result[name][domain] = dict(zip(('Q2_max_abs', 'FIT12_max_abs', 'score_max_abs'), diffs))
        finally:
            with torch.no_grad():
                param.copy_(old)
    return result


def history_gradient(model, entry, birth):
    group, teacher = Group.from_dict(entry['group']), entry['teacher']
    shared = model.shared_ffn_parameters()
    params = dict(shared, **{'head.'+k: p for k, p in model.head.named_parameters()})
    seed = 837
    model.zero_grad(set_to_none=True)
    model.train(not birth)
    torch.manual_seed(seed)
    # Keep the origin route active during checkpoint recomputation in direct autograd.
    with model.routing(group.domain):
        table = model.h.table(model.records(group, noop))
        if birth:
            slots = assignment(group)
            loss = model.settings.beta*(h_pool(table, slots)-h_pool(teacher, slots)).square().mean()
        else:
            loss, _ = h_objective(table, group, seed_for(seed, 'regroup'), model.settings, history=True)
        reference = torch.autograd.grad(loss, list(params.values()))
    model.train()
    rng = torch.random.get_rng_state()
    if birth:
        model.backward_birth(group, teacher, noop)
        assert torch.equal(rng, torch.random.get_rng_state())
    else:
        model.backward_h(group, seed, history=True)
    shared_ids = {id(p) for p in shared.values()}
    assert all(p.grad is None for p in model.encoder.parameters() if id(p) not in shared_ids)
    opt = make_optimizer(model.h)
    before = {k: p.detach().clone() for k, p in params.items()}
    result = {}
    for (name, p), expected in zip(params.items(), reference):
        norm = float(expected.norm())
        assert p.grad is not None and norm > 0 and p.grad.norm().isfinite(), name
        rel = float((p.grad-expected).norm())/norm
        assert rel < 0.001, (name, rel, norm)
        result[name] = dict(reference_norm=norm, relative_gradient_error=rel)
    torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True)
    opt.step()
    for name, param in params.items():
        group_opt = opt.param_groups[0 if name in shared else 1]
        decay_only = before[name]*(1-group_opt['lr']*group_opt['weight_decay'])
        change = float((param.detach()-decay_only).abs().max())
        assert change > 0, name
        result[name]['movement_beyond_decay'] = change
    return dict(origin=group.domain, true_retained_teacher=birth, loss=float(loss.detach()), parameters=result)


def check_arm(root, template, binding, manifest, rows, views):
    task = dict(template, epochs=1)
    method = task['method']
    identity = hashlib.sha256(('synthetic-fit12-'+method).encode()).hexdigest()
    folder = root/method
    live_scores, restored_scores, ffns, bases, dream_banks = {}, {}, {}, {}, {}
    live = []
    original_init, atomic, score_market = MarketH.__init__, run.atomic_tensor, ev.score_market

    def observe_init(model, *args, **kwargs):
        original_init(model, *args, **kwargs)
        live.append(model)

    def observe_save(path, value, check, replace=False):
        atomic(path, value, check, replace)
        if path.stem not in ('stage0', 'stage1', 'stage2', 'stage3'):
            return
        model, stage = live[0], value['stage']
        was_training = model.training
        for t, j in CELLS:
            if t == stage:
                live_scores[t, DOMAINS[j-1]] = tuple(digest([x]) for x in scored(model, views, DOMAINS[j-1]))
        model.train(was_training)
        assert ('shared_ffn' in value) == (stage >= 2)
        if stage:
            shared = model.shared_ffn_parameters()
            assert len(shared) == 4
            ffns[stage] = {k: digest([p]) for k, p in shared.items()}
            ids = {id(p) for p in shared.values()}
            bases[stage] = digest(p for n, p in model.encoder.named_parameters() if '.banks.' not in n and id(p) not in ids)
        if stage >= 2:
            dream_banks[stage] = digest(p for k, p in model.adapter_tensors().items() if '.Dream.' in k)

    with patch.object(MarketH, '__init__', observe_init), patch.object(run, 'atomic_tensor', observe_save):
        end = run.execute(run.make_model(binding, task, 'cpu'), rows.__getitem__, folder, task, identity, binding)
    assert len(live) == 1
    model = live.pop()
    assert len(set(bases.values())) == 1 and dream_banks[2] == dream_banks[3]
    assert all(len({ffns[t][k] for t in (1, 2, 3)}) == 3 for k in ffns[1])

    def observe_score(restored, domain, query, fit, check):
        values = score_market(restored, domain, query, fit, check)
        restored_scores[restored.stage, domain] = tuple(digest([x]) for x in values)
        return values

    with patch.object(ev, 'score_market', observe_score):
        result = ev.score(task, end, manifest, binding, folder/'score', 'dev', noop,
                          model_factory=lambda b, t: run.make_model(b, t, 'cpu'))
    ev.validate_scores(result)
    assert len(restored_scores) == 9 and restored_scores == live_scores
    for t, j in CELLS:
        saved = np.load(pinned(result['files'][f'{method}:R{t}{j}']), allow_pickle=False)
        assert digest([saved]) == live_scores[t, DOMAINS[j-1]][2]

    ffn_count = sum(p.numel() for p in model.shared_ffn_parameters().values())
    report = dict(updates=end['updates'], workload=end['workload'], history_bytes=end['history_bytes'],
                  live_reloaded_identical_cells=result['cells'], all_other_base_parameters_unchanged=True,
                  Dream_bank_frozen_in_Agora=True, four_FFN_tensors_updated_in_both_later_stages=True,
                  shared_FFN_parameters=ffn_count)
    assert end['updates'] == 12 and max(end['history_bytes'].values()) <= CAP
    assert set(end['history_bytes']) == {'end_1','enter_2','end_2','enter_3','end_3'}
    sizes = end['history_bytes']
    payload = (folder/'market_state_3.zip').read_bytes()
    assert len(payload) == sizes['end_3']
    replay = method != 'H_SEQ'
    mem = Memory(0, model.settings)
    restore_history(model, payload, identity, mem)
    with ZipFile(io.BytesIO(payload)) as archive:
        adapters = torch.load(io.BytesIO(archive.read('adapters.pt')), weights_only=True)
        assert len(adapters) == 32 and set(adapters) == set(model.adapter_tensors())
        if replay:
            saved = torch.load(io.BytesIO(archive.read('memory.pt')), weights_only=True)
            assert mem.seen == saved['seen'] == 9 and mem.rng.getstate() == saved['rng']
            assert len(mem.entries) == len(saved['entries']) == 6
            for a, b in zip(mem.entries, saved['entries']):
                assert a['group'] == b['group'] and torch.equal(a['teacher'], b['teacher']) and a['teacher'].isfinite().all()
        else:
            assert not mem.entries and 'memory.pt' not in archive.namelist()
    assert end['workload']['history_groups'] == (6 if replay else 0)
    assert end['workload'].get('deterministic_history_groups', 0) == (6 if method == 'H_full' else 0)
    if method == 'H_full':
        report['routing_interventions'] = perturb_routes(model, views)
        bad = torch.load(pinned(end['endpoints']['2']), weights_only=False)
        del bad['shared_ffn']
        try:
            restore_scoring_endpoint(model.h, bad, identity, 2)
        except ValueError as exc:
            assert 'FFN snapshot' in str(exc)
        else:
            raise AssertionError('Missing stage-specific FFN was accepted')
        report['missing_stage_FFN_rejected'] = True
    if replay:
        entry = max(mem.entries, key=lambda e: e['origin'])
        report['isolated_history_gradient'] = history_gradient(model, entry, method == 'H_full')
        report['restored_replay_groups'] = len(mem.entries)
    pair = dict(current=end['current_stream'], replay=end['history_stream'], survivors=end['survivors'],
                early={k: v for k, v in live_scores.items() if k[0] <= 1})
    return report, pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('official_gate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and len(os.sched_getaffinity(0)) == 1 and not torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    binding = json.loads(args.official_gate.read_text())['model_binding']
    for source in binding['files']:
        pinned(source)
    tasks = market_tasks(shared_ffn=True)
    assert len(tasks) == 9 and len(market_tasks()) == 12
    reports, pairs = {}, {}
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix='synthetic-') as temp:
        root = Path(temp)
        manifest, rows, views = fixture(root)
        for task in [t for t in tasks if t['seed'] == 51]:
            method = task['method']
            print('Qualifying '+method, flush=True)
            reports[method], pairs[method] = check_arm(root, task, binding, manifest, rows, views)
            gc.collect()
    seq, er, full = (pairs[k] for k in ('H_SEQ', 'H_ER', 'H_full'))
    assert seq['current'] == er['current'] == full['current'] and seq['early'] == er['early'] == full['early']
    assert er['replay'] == full['replay'] and er['survivors'] == full['survivors'] and not seq['survivors']
    assert not torch.cuda.is_initialized()
    write(args.output, dict(status='PASS', device='cpu', logical_cores=len(os.sched_getaffinity(0)),
          cuda_initialized=False, data='generated stationery text only', fit_accounts=[12,4,4], seed=51,
          epochs_per_synthetic_market=1, current_groups_identical=True, ER_Full_replay_and_survivors_identical=True,
          shared_initial_and_AlphaBay_scores_identical=True, peak_rss_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          arms=reports))


if __name__ == '__main__':
    main()
