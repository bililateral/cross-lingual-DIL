"""Summarize already released RMS numeric logs; no models, labels, or scores."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time


def binding(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (4 * 2**30, 4 * 2**30))
    root = args.root.resolve()
    target = root / 'result' / 'log_summary.json'
    if target.exists():
        raise ValueError('Preserve original summary')
    gate = json.loads((root / 'gate.json').read_text())
    completion = json.loads((root / 'campaign/completion.json').read_text())
    if completion['status'] not in ('STOP_ADAPTATION', 'PASS_D'):
        raise ValueError('Whole completed D required')
    sources = []
    for item in gate['sources']:
        actual = binding(Path(item['path']))
        if actual != item:
            raise ValueError('Frozen runtime source changed')
        sources.append(actual)
    summaries = {}
    for domain, accounts, updates in [('AlphaBay', 546, 8160), ('Dream', 157, 2400)]:
        job = root / 'campaign/D/training' / ('D_' + domain + '_H_CE_RMS') / 'run'
        rows = [json.loads(line) for line in (job / 'updates.jsonl').read_text().splitlines()]
        if len(rows) != updates or [r['updates'] for r in rows] != list(range(1, updates + 1)):
            raise ValueError('Missing or repeated updates')
        epochs = []
        for epoch in range(40):
            group = [r for r in rows if r['epoch'] == epoch]
            if len(group) != updates // 40:
                raise ValueError('Unexpected epoch size')
            current = [r['current'] for r in group]
            count = sum(r['ce_records'] for r in current)
            if count != 12 * accounts:
                raise ValueError('Exposure changed')
            if any(r['ce_scale'] != math.sqrt(1536) or r['gamma'] != 1 for r in current):
                raise ValueError('Scale intervention missing')
            if any(r['optimizer_updates'] != 1 or r['auxiliary_optimizer_updates'] != 1 for r in group):
                raise ValueError('Optimizer update missing')
            epochs.append(dict(epoch=epoch + 1, updates=len(group), records=count,
                ce=sum(r['ce'] * r['ce_records'] for r in current) / count,
                ce_accuracy=sum(r['ce_correct'] for r in current) / count,
                margin=sum(r['ce_margin_sum'] for r in current) / count,
                logit_std=sum(r['ce_logit_std_sum'] for r in current) / count,
                supervised=statistics.mean(r['supervised'] for r in current),
                auxiliary_row_norm=statistics.mean(r['auxiliary_row_norm_mean'] for r in current),
                auxiliary_effective_row_norm=statistics.mean(r['auxiliary_effective_row_norm_mean'] for r in current),
                record_ce_h_gradient_ratio_median=statistics.median(
                    r['record_gradient_CE_norm'] / r['record_gradient_H_norm'] for r in current),
                clip_fraction=statistics.mean(float(r['gradient_norm'] > 1) for r in group)))
        summaries[domain] = dict(log=binding(job / 'updates.jsonl'), updates=len(rows), epochs=epochs)
    elapsed = time.monotonic() - started
    result = dict(source=binding(Path(__file__).resolve()), scope='saved_numeric_logs_only',
        no_model_or_score_or_role_load=True, runtime_sources=sources, markets=summaries,
        actual_seconds=elapsed, charged_seconds=math.ceil(elapsed),
        cpu_affinity=sorted(os.sched_getaffinity(0)), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        cost_note='Supplement to completed 7633-second campaign; original budget.json preserved')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(actual_seconds=elapsed, charged_seconds=math.ceil(elapsed),
        last_epoch={d: v['epochs'][-1] for d, v in summaries.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
