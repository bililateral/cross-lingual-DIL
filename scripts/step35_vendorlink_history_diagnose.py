"""One approved, forward-only examination of nine frozen FIT12 endpoints."""
import argparse
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import psutil


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError('No overwriting diagnostic evidence')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def binding(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def verify(value):
    actual = binding(value['path'])
    if any(actual[k] != value[k] for k in ('bytes', 'sha256')):
        raise ValueError('Source identity mismatch: '+value['path'])
    return Path(value['path'])


def decompose(samples, reference, deterministic):
    samples, reference, deterministic = [np.asarray(x, dtype=np.float64)
                                          for x in (samples, reference, deterministic)]
    if samples.ndim != 2 or samples.shape[1:] != reference.shape or deterministic.shape != reference.shape:
        raise ValueError('Aligned draws/pairs required')
    average = samples.mean(0)
    mse = float(np.square(samples-reference).mean())
    variance = float(np.square(samples-average).mean())
    bias = float(np.square(average-reference).mean())
    error = deterministic-reference
    residual = abs(mse-variance-bias)
    if residual > 1e-10*max(1., mse):
        raise ValueError('Variance decomposition identity failed')
    return dict(stochastic_mse=mse, stochastic_variance=variance, squared_mean_shift=bias,
                deterministic_mse=float(np.square(error).mean()),
                train_mean_vs_eval_mse=float(np.square(average-deterministic).mean()),
                deterministic_shift=float(error.mean()),
                deterministic_shift_squared=float(error.mean()**2),
                deterministic_centered_mse=float(np.square(error-error.mean()).mean()),
                decomposition_residual=residual)


def margins(scores, labels, n):
    i, j = np.triu_indices(n, 1)
    table = np.full((n, n), -np.inf)
    positives = np.zeros((n, n), dtype=bool)
    table[i, j] = table[j, i] = scores
    positives[i, j] = positives[j, i] = labels
    result = []
    for q in range(n):
        negative = ~positives[q]
        negative[q] = False
        result.extend((table[q, positives[q]]-table[q, negative].max()).tolist())
    return dict(mean=float(np.mean(result)), minimum=float(min(result)),
                nonpositive_fraction=float(np.mean(np.asarray(result) <= 0)))


def self_test():
    started = time.monotonic()
    d = decompose([[0., 2.], [2., 4.]], [1., 1.], [2., 2.])
    expected = dict(stochastic_mse=3., stochastic_variance=1., squared_mean_shift=2.,
                    deterministic_mse=1., train_mean_vs_eval_mse=1., deterministic_centered_mse=0.)
    assert all(d[k] == v for k, v in expected.items())
    zero = decompose([[1., -1.]]*16, [1., -1.], [1., -1.])
    assert all(v == 0. for v in zero.values())
    # Views 0/1 and 2/3 are positive; every positive beats every negative by 3.
    m = margins(np.array([2., -1., -1., -1., -1., 2.]), np.array([1, 0, 0, 0, 0, 1], bool), 4)
    assert m == dict(mean=3., minimum=3., nonpositive_fraction=0.)
    return dict(status='PASS', seconds=time.monotonic()-started,
                checks=['independent_hand_calculation', 'zero_drift', 'positive_negative_margin'])


def run(args, started):
    root = args.root.resolve()
    base = root.parent
    gate = read(base/'gate.json')
    for value in gate['sources']+[gate['manifest']]:
        verify(value)
    for value in gate['model_binding']['files']:
        verify(value)
    sources = [binding(__file__), binding(root/'scope.zh.md')]+gate['sources']
    sys.path.insert(0, str(Path(gate['workspace'])/'scripts'))
    import torch
    import transformers
    from step29_vendorlink_data import assignment, seed_for
    from step29_vendorlink_methods import h_pool
    from step35_vendorlink_account_run import groups_for, restore
    from step35_vendorlink_account_policy import DOMAINS
    if sys.platform != 'linux' or Path(sys.executable).resolve() != Path('/home/yongpeng/miniconda3/envs/py310/bin/python').resolve():
        raise RuntimeError('Only the existing Linux py310 is allowed')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free_mib = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                          '--format=csv,noheader,nounits'], text=True).strip())
    if free_mib < 5120 or psutil.virtual_memory().available < 8*2**30:
        raise RuntimeError('Insufficient observed resource; no automatic waiting or retry')
    manifest = read(gate['manifest']['path'])
    fit_binding = manifest['files']['AlphaBay:fit']
    fit = read(verify(fit_binding))
    if len(fit) != 546:
        raise ValueError('Unexpected frozen fit count')
    budget = dict(peak_rss_bytes=0, forwards=0)

    def check():
        budget['peak_rss_bytes'] = max(budget['peak_rss_bytes'], psutil.Process().memory_info().rss)
        if time.monotonic()-started > 2695:
            raise TimeoutError('GPU lifecycle budget')
        if budget['peak_rss_bytes'] > 8*2**30:
            raise RuntimeError('Approved RSS limit')
        if sum(p.stat().st_size for p in root.iterdir() if p.is_file()) > 64*2**20:
            raise RuntimeError('Approved new-output limit')

    rows, identities, counts, versions = [], [], [], []
    dropout_modules = None
    with torch.no_grad():
        for seed in (51, 52, 53):
            candidates = groups_for(fit, seed, 'AlphaBay', 0)
            selected, rng = [], random.Random(seed_for(seed, DOMAINS, 'reservoir'))
            for seen, group in enumerate(sorted(candidates, key=lambda g: g.uid), 1):
                slot = len(selected) if len(selected) < 6 else rng.randrange(seen)
                if slot < 6:
                    if slot == len(selected):
                        selected.append(group)
                    else:
                        selected[slot] = group
            index_path = base/f'campaign/M/training/M_H_full_s{seed}/run/memory_stage1.json'
            er_index_path = base/f'campaign/M/training/M_H_ER_s{seed}/run/memory_stage1.json'
            index, er_index = read(index_path), read(er_index_path)
            expected = [dict(uid=g.uid, origin=1) for g in selected]
            if index['groups'] != expected or er_index['groups'] != expected or len(selected) != 6:
                raise ValueError('Actual retained group identity mismatch')
            identities.extend([binding(index_path), binding(er_index_path)])
            counts.append(dict(seed=seed, groups=6, candidates=len(candidates),
                               distinct_accounts=len({v.identity for g in selected for v in g.views}),
                               records=sum(len(v.records) for g in selected for v in g.views)))
            references = []
            for method, stage in (('H_full', 1), ('H_full', 2), ('H_ER', 2)):
                check()
                directory = base/f'campaign/M/training/M_{method}_s{seed}/run'
                completion = read(directory/'completion.json')
                endpoint = completion['endpoints'][str(stage)]
                verify(endpoint)
                identities.extend([binding(directory/'completion.json'), endpoint])
                model, _ = restore(gate['model_binding'], completion['task'], endpoint['path'],
                                   completion['identity'], stage)
                before = [(name, id(p), p._version) for name, p in model.named_parameters()]
                if dropout_modules is None:
                    dropout_modules = [dict(name=name, type=type(m).__name__, p=float(m.p))
                                       for name, m in model.named_modules() if isinstance(m, torch.nn.Dropout)]
                    versions = dict(torch=torch.__version__, transformers=transformers.__version__,
                                    attention_implementation=getattr(model.encoder.config, '_attn_implementation', None))
                for k, group in enumerate(selected):
                    def forward(training, draw):
                        check()
                        model.train(training)
                        torch.manual_seed(seed_for(seed, 'history-diagnosis', group.uid, draw))
                        table = model.table(model.records(group, check))
                        result = h_pool(table, assignment(group)).cpu().numpy().astype(np.float64)
                        if not np.isfinite(result).all():
                            raise ValueError('Nonfinite diagnostic score')
                        budget['forwards'] += 1
                        return result
                    deterministic = forward(False, 'deterministic')
                    repeat = forward(False, 'repeat')
                    repeat_difference = float(np.max(np.abs(deterministic-repeat)))
                    if repeat_difference > 1e-6:
                        raise ValueError('Deterministic repeat mismatch')
                    if stage == 1:
                        actual_teacher = h_pool(model.teacher(group, check).to('cuda:0'),
                                                assignment(group)).cpu().numpy().astype(np.float64)
                        budget['forwards'] += 1
                        if not np.array_equal(actual_teacher, deterministic):
                            raise ValueError('Actual teacher path mismatch')
                        references.append(actual_teacher)
                    reference = references[k]
                    draws = np.stack([forward(True, r) for r in range(16)])
                    labels = np.asarray(group.labels(), dtype=bool)
                    row = dict(seed=seed, method=method, stage=stage, group_number=k,
                               records=sum(len(v.records) for v in group.views), pairs=len(labels),
                               positives=int(labels.sum()), repeat_max_difference=repeat_difference,
                               subsets={name: decompose(draws[:, mask], reference[mask], deterministic[mask])
                                        for name, mask in (('all', np.ones(len(labels), bool)),
                                                           ('positive', labels), ('negative', ~labels))},
                               reference_margin=margins(reference, labels, len(group.views)),
                               deterministic_margin=margins(deterministic, labels, len(group.views)),
                               deterministic_positive_score=float(deterministic[labels].mean()),
                               deterministic_negative_score=float(deterministic[~labels].mean()))
                    rows.append(row)
                after = [(name, id(p), p._version) for name, p in model.named_parameters()]
                if before != after or any(p.grad is not None for p in model.parameters()):
                    raise ValueError('Unexpected parameter mutation/gradient')
                print(json.dumps(dict(progress='endpoint_complete', seed=seed, method=method, stage=stage,
                                      endpoints_completed=len(rows)//6, seconds=time.monotonic()-started)), flush=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()
    if len(rows) != 54 or len(counts) != 3:
        raise ValueError('Nine endpoints and all six groups required')
    summary = []
    for seed in (51, 52, 53):
        for method, stage in (('H_full', 1), ('H_full', 2), ('H_ER', 2)):
            selected_rows = [r for r in rows if (r['seed'], r['method'], r['stage']) == (seed, method, stage)]
            summary.append(dict(seed=seed, method=method, stage=stage,
                subsets={name: {key: float(np.mean([r['subsets'][name][key] for r in selected_rows]))
                                for key in selected_rows[0]['subsets'][name]}
                         for name in ('all', 'positive', 'negative')},
                reference_margin={key: float(np.mean([r['reference_margin'][key] for r in selected_rows]))
                                  for key in selected_rows[0]['reference_margin']},
                deterministic_margin={key: float(np.mean([r['deterministic_margin'][key] for r in selected_rows]))
                                      for key in selected_rows[0]['deterministic_margin']}))
    check()
    torch.cuda.synchronize()
    return dict(status='COMPLETE', sources=sources, input_role=fit_binding, inputs=identities,
                reconstruction=counts, versions=versions, dropout_modules=dropout_modules,
                rows=rows, summary=summary, optimizer_updates=0, parameter_mutations=0,
                formal_metrics_recomputed=False, confirmation_opened=False,
                cost=dict(**budget, seconds=time.monotonic()-started,
                          sampled_cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(),
                          sampled_cuda_max_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:2])
    if args.self_test:
        result = self_test()
        result.update(source=binding(__file__), lifecycle_seconds=time.monotonic()-started)
        write(args.root/'cpu.json', result)
        print(json.dumps(result))
    else:
        if (args.root/'started.json').exists():
            raise RuntimeError('One approved diagnostic attempt; no automatic retry')
        write(args.root/'started.json', dict(pid=os.getpid(), created=process.create_time(),
                                            owner=process.username(), command=process.cmdline(), cwd=process.cwd(),
                                            start=datetime.datetime.now().astimezone().isoformat()))
        def timeout(signum, frame):
            raise TimeoutError('2700-second lifecycle limit')
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(2700)
        try:
            result = run(args, started)
            write(args.root/'results.json', result)
            write(args.root/'completion.json', dict(status='COMPLETE', endpoints=9, groups_per_endpoint=6,
                results=binding(args.root/'results.json'), seconds=time.monotonic()-started,
                finish=datetime.datetime.now().astimezone().isoformat()))
            print(json.dumps(dict(status='COMPLETE', seconds=time.monotonic()-started)), flush=True)
        except BaseException as exc:
            write(args.root/'failure.json', dict(error=repr(exc), automatic_retry=False,
                                                seconds=time.monotonic()-started))
            raise
        finally:
            signal.alarm(0)
