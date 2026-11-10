"""Summarize only saved D numeric training logs; never load models or scores."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import statistics
import sys
import time

import psutil


def describe(values):
    return dict(count=len(values), mean=statistics.fmean(values),
                median=statistics.median(values), minimum=min(values), maximum=max(values))


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    python = Path('/home/yongpeng/miniconda3/envs/py310/bin/python')
    if sys.platform != 'linux' or Path(sys.executable).resolve() != python.resolve():
        raise ValueError('Existing Linux py310 only')
    proc = psutil.Process()
    proc.cpu_affinity(proc.cpu_affinity()[-2:])
    resource.setrlimit(resource.RLIMIT_AS, (4 * 2**30, 4 * 2**30))
    signal.alarm(600)
    root = args.root.resolve(strict=True)
    if args.output.exists():
        raise ValueError('Do not overwrite a prior diagnosis')
    decision = json.loads((root / 'campaign/decisions/D.json').read_text())
    if decision['status'] != 'STOP_ADAPTATION':
        raise ValueError('Only the completed released D run is in scope')
    summaries = {}
    sources = []
    for domain, classes, h_steps, v_steps in (('AlphaBay', 546, 204, 205), ('Dream', 157, 60, 59)):
        for candidate in ('H_base', 'H_CE', 'V0', 'V1'):
            task = f'D_{domain}_{candidate}'
            path = root / 'campaign/D/training' / task / 'run/updates.jsonl'
            h = candidate.startswith('H')
            gamma = float(candidate == 'H_CE')
            per_epoch = h_steps if h else v_steps
            total = 40 * per_epoch
            buckets = defaultdict(lambda: defaultdict(list))
            epoch_ce = defaultdict(lambda: [0.0, 0])
            digest = hashlib.sha256()
            auxiliary_updates = exposures = 0
            maximum_rate_error = maximum_encoder_norm_discrepancy = 0.0
            first = last = None
            with path.open('rb') as stream:
                for count, raw in enumerate(stream, 1):
                    digest.update(raw)
                    row = json.loads(raw)
                    epoch = row['epoch']
                    if (epoch != (count - 1) // per_epoch or row['stage'] != 1
                            or any(row[key] != count for key in ('stage_update', 'updates', 'total_updates'))
                            or row['optimizer_updates'] != 1):
                        raise ValueError(f'Unexpected saved update counters: {task}')
                    values = {'gradient_norm': row['gradient_norm'],
                              'clip_applied': float(row['gradient_norm'] > 1),
                              'clip_factor': min(1., 1. / (row['gradient_norm'] + 1e-6))}
                    if h:
                        current = row['current']
                        if current['gamma'] != gamma or current['distillation'] != 0:
                            raise ValueError(f'Wrong D intervention: {task}')
                        if row['auxiliary_optimizer_updates'] != int(gamma):
                            raise ValueError(f'Missing/extraneous auxiliary update: {task}')
                        auxiliary_updates += row['auxiliary_optimizer_updates']
                        records = current['ce_records']
                        values.update(ce=current['ce'], h_supervised=current['supervised'],
                                      record_h_grad=current['record_gradient_H_norm'],
                                      record_ce_grad=current['record_gradient_CE_norm'],
                                      encoder_grad=row['current_encoder_gradient_norm'])
                        if values['record_h_grad'] > 0:
                            values['ce_to_h_record_gradient_ratio'] = values['record_ce_grad'] / values['record_h_grad']
                        maximum_encoder_norm_discrepancy = max(maximum_encoder_norm_discrepancy,
                            abs(row['current_encoder_gradient_norm'] - row['total_encoder_gradient_norm']))
                        warmup = math.ceil(.1 * total)
                        factor = count / warmup if count <= warmup else (total - count) / (total - warmup)
                        expected_rates = [2.5e-5 * factor, 3e-4 * factor]
                    else:
                        records = min(32, 12 * classes - ((count - 1) % per_epoch) * 32)
                        values['ce'] = row['cross_entropy']
                        expected_rates = [(4e-5 if candidate == 'V0' else 2e-5) * (total - count + 1) / total]
                    exposures += records
                    epoch_ce[epoch][0] += values['ce'] * records
                    epoch_ce[epoch][1] += records
                    maximum_rate_error = max(maximum_rate_error,
                        max(abs(a-b) for a, b in zip(row['rates'], expected_rates, strict=True)))
                    if not all(math.isfinite(value) for value in values.values()):
                        raise ValueError(f'Nonfinite saved diagnostic: {task}')
                    for key, value in values.items():
                        buckets[epoch][key].append(value)
                    projection = dict(update=count, epoch=epoch, rates=row['rates'], **values)
                    first = projection if first is None else first
                    last = projection
                    if count % 1000 == 0 and (time.monotonic() - started > 590 or proc.memory_info().rss > 4 * 2**30):
                        raise RuntimeError('Diagnosis CPU/RSS budget exhausted')
            if count != total or exposures != 40 * 12 * classes or maximum_rate_error > 1e-12:
                raise ValueError(f'Exposure/schedule mismatch: {task}')
            epochs = []
            for epoch in range(40):
                ce = epoch_ce[epoch][0] / epoch_ce[epoch][1]
                epochs.append(dict(epoch=epoch + 1, weighted_ce=ce,
                    geometric_mean_true_class_probability=math.exp(-ce),
                    values={key: describe(values) for key, values in buckets[epoch].items()}))
            summaries[task] = dict(classes=classes, uniform_ce=math.log(classes), updates=count,
                record_exposures=exposures, auxiliary_updates=auxiliary_updates,
                maximum_learning_rate_error=maximum_rate_error,
                maximum_encoder_norm_discrepancy=maximum_encoder_norm_discrepancy if h else None,
                first=first, last=last, epochs=epochs)
            sources.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest()))
    elapsed = time.monotonic() - started
    value = dict(observed_at=datetime.now(timezone.utc).isoformat(), status='COMPLETE',
        source=dict(path=str(Path(__file__).resolve()), bytes=Path(__file__).stat().st_size,
                    sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        input_sources=sources, tasks=summaries,
        cost=dict(seconds=elapsed, charged_seconds=math.ceil(elapsed), cap_seconds=600,
                  cpu_affinity=proc.cpu_affinity(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        boundaries=['Saved numeric log summaries only; no model/data/prediction reads or metric recomputation.',
                    'Record-gradient ratios are before normalization Jacobians and clipping; not encoder contribution or cosine.',
                    'exp(-mean CE) is geometric mean true-class probability, never accuracy.',
                    'H_base auxiliary head is untrained; its CE is a random-head diagnostic.',
                    'Single seed; no estimate of between-training uncertainty.'])
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    if len(payload) > 20 * 2**20:
        raise RuntimeError('Output budget exceeded')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(payload)
    print(json.dumps(dict(status=value['status'], tasks=len(summaries), bytes=len(payload), cost=value['cost'])))


if __name__ == '__main__':
    main()
