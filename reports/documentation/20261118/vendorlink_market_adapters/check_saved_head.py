"""Distinguish saved weights from fresh initialization on the ordinary restore/score path."""
import argparse
from datetime import datetime
from functools import partial
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import torch

from check_routing import (DOMAINS, FORMAT, IDENTITY, MarketH, Memory, beta_tasks, differences, groups_for,
                           handmade, load_role, make_model, pack_history, read_model, record, shift_bank, write)
from step30_evolution_io import sha
from step35_vendorlink_account_run import restore


@torch.no_grad()
def run(work, gate):
    binding = gate['model_binding']
    task = dict(beta_tasks(dict(H='H_CE_RMS', V='V1'))[0], id='synthetic_saved_head', market_adapters=FORMAT)
    factory = partial(make_model, device='cpu')
    plain = factory(binding, task)
    roles = {}
    for market in DOMAINS:
        roles[market] = []
        for role in ('dev', 'fit'):
            path = work / f'{market}-{role}.json'
            write(path, handmade(market, role, 3))
            roles[market].append(load_role(record(path), market, role, 3)[1])
    original = read_model(plain, 'AlphaBay', roles)
    # A deliberately identifiable synthetic shared encoder also detects an omitted base restore.
    plain.encoder.encoder.layer[-1].output.LayerNorm.bias[:24].add_(1.5)
    base_path = work / 'shared.pt'
    torch.save(plain.encoder.state_dict(), base_path)
    model = MarketH(plain)
    model.enter(1)
    memory = Memory(1814, model.settings)
    memory.retain(groups_for(handmade('AlphaBay', 'fit', 16), task['seed'], 'AlphaBay', 0), 1, model)
    model.enter(2)
    shift_bank(model, 'Dream', 1811)
    model.enter(3)
    shift_bank(model, 'Agora', 1812)
    model.eval()
    history_file = work / 'history.zip'
    history_file.write_bytes(pack_history(model, memory, IDENTITY))
    common = dict(identity=IDENTITY, task=task, model_binding=binding, stage=3, vendor_heads={},
                  shared_encoder=record(base_path), market_state=record(history_file))
    direct, files = {}, []
    for name in ('before', 'after'):
        if name == 'after':
            last = model.head[-1].weight
            last.add_(torch.randn(last.shape, generator=torch.Generator().manual_seed(1813)) * 0.2)
        endpoint = work / f'{name}.pt'
        torch.save(dict(common, head=model.head.state_dict()), endpoint)
        files.append(endpoint)
        direct[name] = {m: read_model(model, m, roles) for m in DOMAINS}
    banks_distinct = {f'{a}-{b}': differences(direct['before'][a], direct['before'][b])
                      for a, b in (('AlphaBay', 'Dream'), ('Dream', 'Agora'), ('AlphaBay', 'Agora'))}
    base_distinct = differences(original, direct['before']['AlphaBay'])
    assert all(min(d.values()) > 0 for d in [base_distinct, *banks_distinct.values()])
    del model, plain
    actual, errors = {}, {}
    for name, endpoint in zip(('before', 'after'), files):
        loaded, _ = restore(binding, task, endpoint, IDENTITY, 3, factory=factory)
        actual[name] = {m: read_model(loaded, m, roles) for m in DOMAINS}
        errors[name] = {m: differences(direct[name][m], actual[name][m]) for m in DOMAINS}
        assert not any(v for row in errors[name].values() for v in row.values())
        del loaded
    effect = {m: differences(actual['before'][m], actual['after'][m]) for m in DOMAINS}
    assert all(row['Q2'] == row['FIT12'] == 0 and min(row['scores'], row['within_query_score_gap']) > 0
               for row in effect.values())
    return dict(status='PASS', training_updates=0, base_vs_initial=base_distinct,
                three_distinct_routes=banks_distinct, restored_vs_direct=errors, only_saved_head_changes=effect,
                same_shared_base_sha256=common['shared_encoder']['sha256'],
                same_market_state_sha256=common['market_state']['sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_available()
    start = time.monotonic()
    with TemporaryDirectory(dir=args.output.parent, prefix='synthetic-head-') as directory:
        result = run(Path(directory), json.loads(args.gate.read_text()))
    result.update(ended=datetime.now().astimezone().isoformat(timespec='seconds'),
                  wall_seconds=time.monotonic() - start, source_sha256=sha(Path(__file__)),
                  check_routing_sha256=sha(Path(__file__).with_name('check_routing.py')))
    write(args.output, result)
    print(result['status'], result['ended'], flush=True)
