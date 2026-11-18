"""Synthetic, CPU-only interventions through the normal full H Q2/FIT12 score path."""
import argparse
from dataclasses import asdict
from datetime import datetime
from functools import partial
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory
import time
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import torch

from step29_vendorlink_data import Group, seed_for
from step29_vendorlink_methods import Memory
from step35_market_adapters import CAP, DOMAINS, FORMAT, MarketH, pack_history, restore_history
from step35_vendorlink_account_evaluate import load_role, score, score_market, validate_scores
from step35_vendorlink_account_io import pinned, record, write
from step35_vendorlink_account_policy import beta_tasks
from step35_vendorlink_account_run import groups_for, make_model


ITEMS = ('linen', 'oak', 'cotton', 'steel', 'paper', 'copper', 'wool', 'glass')
SHAPES = ('woven mat', 'storage box', 'travel bag', 'desk lamp')
COLORS = ('red', 'blue', 'green', 'gold', 'white', 'black')
IDENTITY = hashlib.sha256(b'synthetic FIT12 market routing qualification').hexdigest()


def handmade(market, role, n):
    rows = []
    for account in range(n):
        records = []
        for i in range(12 if role == 'fit' else 2):
            token = f'{market}/{role}/{account}/{i}'.encode()
            item, shape = ITEMS[account % 8], SHAPES[account % 4]
            color = COLORS[(i + (2 if role == 'dev' else 0) + account) % 6]
            records.append(dict(uid=hashlib.sha256(token).hexdigest(), title=f'{color} {item} {shape}',
                                description=f'The {shape} has a {item} surface. Edition {i + 1}, color {color}.'))
        rows.append(dict(account_uid=f'synthetic-{market}-{account:02}', records=records))
    return rows


def max_delta(x, y):
    a, b = np.asarray(x), np.asarray(y)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.max(np.abs(a - b)))


def differences(before, after):
    q, fit, scores = (max_delta(x, y) for x, y in zip(before, after, strict=True))
    delta = after[2] - before[2]
    return dict(Q2=q, FIT12=fit, scores=scores, within_query_score_gap=float(np.ptp(delta, axis=1).max()))


@torch.no_grad()
def shift_bank(model, market, seed):
    gen = torch.Generator().manual_seed(seed)
    for p in model.projections.values():
        weight = p.banks[market].up.weight
        weight.add_(torch.randn(weight.shape, generator=gen) * 0.08)


def read_model(model, market, roles):
    q, fit, matrix = score_market(model, market, *roles[market], lambda: None)
    assert q.shape == (3, 2, 1536) and fit.shape == (3, 12, 1536) and matrix.shape == (3, 3)
    return q.cpu().numpy(), fit.cpu().numpy(), matrix


def check_dense(factory, binding, task, routed, market, roles, expected):
    # Independent algebraic reference: replace W by W+up@down on a plain BERT,
    # without route contexts, adapter forwards or changing the relation head.
    plain = factory(binding, task)
    for site, p in routed.projections.items():
        layer, name = site.split('.')
        weight = getattr(plain.encoder.encoder.layer[int(layer)].attention.self, name).weight
        bank = p.banks[market]
        weight.add_(bank.up.weight @ bank.down.weight)
    actual = differences(expected, read_model(plain, market, roles))
    assert max(actual.values()) < 5e-6, (market, actual)
    return actual


def run(work: Path, gate: dict) -> dict:
    task = dict(beta_tasks(dict(H='H_CE_RMS', V='V1'))[0], id='synthetic_route', market_adapters=FORMAT)
    binding = gate['model_binding']
    for item in binding['files']:
        pinned(item)
    factory = partial(make_model, device='cpu')
    model = factory(binding, task)
    assert not torch.cuda.is_available() and model.tokenizer.do_lower_case
    base_path = work / 'one_synthetic_shared_encoder.pt'
    torch.save(model.encoder.state_dict(), base_path)
    base = record(base_path)
    manifest, roles = dict(files={}, counts={}), {}
    for market in DOMAINS:
        roles[market] = []
        manifest['counts'][market] = dict(eligible_accounts=3)
        for role in ('dev', 'fit'):
            path = work / f'{market}-{role}.json'
            write(path, handmade(market, role, 3))
            source = record(path)
            manifest['files'][f'{market}:{role}'] = source
            roles[market].append(load_role(source, market, role, 3)[1])
    plain = read_model(model, 'AlphaBay', roles)
    model = MarketH(model).eval()
    memory = Memory(seed_for(task['seed'], DOMAINS, 'reservoir'), model.settings)
    endpoints = {}

    def save_endpoint(stage, pack=None):
        value = dict(identity=IDENTITY, task=task, model_binding=binding, stage=stage,
                     head=model.head.state_dict(), vendor_heads={})
        if stage:
            value['shared_encoder'] = base
        if pack is not None:
            path = work / f'history-{stage}.zip'
            path.write_bytes(pack)
            value['market_state'] = record(path)
        path = work / f'endpoint-{stage}.pt'
        torch.save(value, path)
        endpoints[str(stage)] = record(path)

    save_endpoint(0)
    model.enter(1)
    model.eval()
    save_endpoint(1)
    for stage, market in enumerate(DOMAINS[:2], 1):
        if stage == 2:
            model.enter(2)
            fresh = differences(plain, read_model(model, market, roles))
            assert not any(fresh.values()), fresh
            shift_bank(model, market, 1811)
        groups = groups_for(handmade(market, 'fit', 16), task['seed'], market, 0)
        memory.retain(groups, stage, model)
    save_endpoint(2, pack_history(model, memory, IDENTITY))
    model.enter(3)
    for p in model.projections.values():
        for k, v in p.banks['Dream'].named_parameters():
            other = p.banks['Agora'].get_parameter(k)
            assert v.data_ptr() != other.data_ptr() and torch.equal(v, other)
    model.eval()
    initial = {m: read_model(model, m, roles) for m in DOMAINS}
    a_identity = differences(plain, initial['AlphaBay'])
    dream_active = differences(initial['AlphaBay'], initial['Dream'])
    copy_effect = differences(initial['Dream'], initial['Agora'])
    assert not any(a_identity.values()) and not any(copy_effect.values())
    assert all(v > 0 for v in dream_active.values()), dream_active
    saved = {k: p.detach().clone() for k, p in model.adapter_tensors().items()}
    shift_bank(model, 'Agora', 1812)
    changed = {m: read_model(model, m, roles) for m in DOMAINS}
    c_effect = {m: differences(initial[m], changed[m]) for m in DOMAINS}
    assert not any(c_effect['AlphaBay'].values()) and not any(c_effect['Dream'].values())
    assert all(v > 0 for v in c_effect['Agora'].values()), c_effect
    dense = {m: check_dense(factory, binding, task, model, m, roles, changed[m]) for m in DOMAINS[1:]}
    bundle = pack_history(model, memory, IDENTITY)
    save_endpoint(3, bundle)
    with ZipFile(io.BytesIO(bundle)) as archive:
        members = {i.filename: i.file_size for i in archive.infolist()}
        assert all(i.compress_type == ZIP_STORED and i.compress_size == i.file_size for i in archive.infolist())
        assert archive.read('memory.pt') == memory.bytes()
    head_last = model.head[-1].weight
    saved_head = head_last.clone()
    model.load_adapters(saved)
    assert all(torch.equal(p, saved[k]) for k, p in model.adapter_tensors().items())
    gen = torch.Generator().manual_seed(1813)
    head_last.add_(torch.randn(head_last.shape, generator=gen) * 0.2)
    head_effect = {m: differences(initial[m], read_model(model, m, roles)) for m in DOMAINS}
    assert all(d['Q2'] == d['FIT12'] == 0 and d['scores'] > 0 and d['within_query_score_gap'] > 0
               for d in head_effect.values()), head_effect
    head_last.copy_(saved_head)
    recovered = Memory(0, model.settings)
    restore_history(model, bundle, IDENTITY, memory=recovered)
    roundtrip = differences(changed['Agora'], read_model(model, 'Agora', roles))
    assert not any(roundtrip.values())
    assert (len(recovered.entries) == 6 and recovered.seen == memory.seen
            and recovered.rng.getstate() == memory.rng.getstate())
    teachers = []
    for expected, row in zip(memory.entries, recovered.entries, strict=True):
        assert expected['group'] == row['group'] and expected['origin'] == row['origin']
        group, teacher = Group.from_dict(row['group']), row['teacher']
        assert teacher.dtype == torch.float32 and teacher.shape == (496,)
        assert torch.isfinite(teacher).all() and teacher.min() < teacher.max()
        assert torch.equal(teacher, expected['teacher']) and torch.equal(teacher, model.teacher(group, lambda: None))
        teachers.append(dict(origin=row['origin'], records=sum(len(v.records) for v in group.views),
                             teacher_floats=teacher.numel()))
    assert {t['origin'] for t in teachers} == {1, 2}
    for key in range(12):
        g1, t1 = memory.draw(key)
        g2, t2 = recovered.draw(key)
        assert g1 == g2 and torch.equal(t1, t2)
    recovered.entries[0]['group']['views'][0]['records'][0]['description'] += 'x' * CAP
    try:
        pack_history(model, recovered, IDENTITY)
    except ValueError as e:
        if 'exceeds the 1 MiB cap' not in str(e):
            raise
        oversized = str(e)
    else:
        raise AssertionError('Oversized history was accepted')

    # Drive the original manifest -> role reader -> endpoint restore -> cell export.
    # Every completion and input below is a temporary, explicitly synthetic fixture.
    completion = dict(status='TRAINING_COMPLETE_EVALUATION_CLOSED', identity=IDENTITY, endpoints=endpoints)
    exported = score(task, completion, manifest, binding, work / 'synthetic_scores', 'dev', lambda: None,
                     model_factory=factory)
    validate_scores(exported)
    cell_errors = {}
    for key, file in exported['files'].items():
        cell = key.split(':')[1]
        stage, j = int(cell[1]), int(cell[2])
        expected = plain if stage < 2 else changed[DOMAINS[min(j, stage) - 1]]
        cell_errors[cell] = max_delta(np.load(pinned(file), allow_pickle=False), expected[2])
    assert len(cell_errors) == 9 and not any(cell_errors.values()), cell_errors
    parameters = {m: sum(v.numel() for k, v in model.adapter_tensors().items() if f'.{m}.' in k)
                  for m in DOMAINS[1:]}
    assert list(parameters.values()) == [98304, 98304]
    disk_bytes = (work / 'history-3.zip').stat().st_size
    assert disk_bytes == len(bundle) <= CAP
    return dict(status='PASS', validation_inputs='handwritten_synthetic_only', training_updates=0,
                official_model_files={Path(x['path']).name: x['sha256'] for x in binding['files']},
                tokenizer_do_lower_case=model.h.tokenizer.do_lower_case,
                settings=asdict(model.settings), query_shape=[3, 2, 1536], fit_shape=[3, 12, 1536],
                score_shape=[3, 3], raw_score='full H account relation logit',
                AlphaBay_matches_unadapted=a_identity, Dream_at_birth_matches_AlphaBay=fresh,
                Dream_nonzero_vs_base=dream_active, Agora_at_birth_matches_Dream=copy_effect,
                Agora_only=c_effect, head_weights_only=head_effect, independent_dense=dense,
                restored_Agora=roundtrip, full_score_export_max_errors=cell_errors,
                capacity=dict(parameters=parameters, dtype='float32', payload_bytes=sum(parameters.values()) * 4,
                              members=members, framing_bytes=disk_bytes - sum(members.values()), total_bytes=disk_bytes,
                              cap_bytes=CAP, remaining_bytes=CAP - disk_bytes, sha256=hashlib.sha256(bundle).hexdigest(),
                              groups=teachers, reservoir_seen=memory.seen, rng_restored=True, draws_equal=12,
                              oversized_rejection=oversized, memory_bytes_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start, cpu = time.monotonic(), time.process_time()
    when = lambda: datetime.now().astimezone().isoformat(timespec='seconds')
    started = when()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with TemporaryDirectory(dir=args.output.parent, prefix='synthetic-') as dirname, torch.no_grad():
        receipt = run(Path(dirname), json.loads(args.gate.read_text()))
    from step30_evolution_io import sha
    receipt.update(sources={Path(m.__file__).name: sha(Path(m.__file__)) for name, m in sys.modules.items()
                            if name.startswith(('step29_', 'step30_', 'step35_'))},
                   check_sha256=sha(Path(__file__)), gate_sha256=sha(args.gate), started=started, ended=when(),
                   wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu,
                   rss_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   affinity=sorted(os.sched_getaffinity(0)), python=sys.executable, torch=torch.__version__,
                   cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write(args.output, receipt)
    print(receipt['status'], receipt['ended'], receipt['capacity']['total_bytes'], flush=True)
