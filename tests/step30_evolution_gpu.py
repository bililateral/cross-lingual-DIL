"""Authorized one-worker, handwritten-input GPU qualification; no real data."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time


def fixture_groups():
    from step29_vendorlink_data import Ad, Group, View
    from step30_evolution_data import validate_group
    from step30_evolution_policy import DOMAINS

    # Both H channels reach 256 tokens; V concatenation reaches 512 tokens.
    # Keep the original six-group/one-MiB memory bound, including raw text.
    filler = 'red blue green white black ' * 60
    result = {}
    for domain in DOMAINS:
        groups = []
        for group_index in range(20):
            views = []
            for account in range(8):
                owner = f'fixture:{domain}:{group_index}:{account}'
                for side in range(2):
                    records = tuple(Ad(
                        f'{owner}:record:{side * 2 + j}',
                        f'Example title account {account} item {side * 2 + j}. ' + filler,
                        f'Example description group {group_index} item {side * 2 + j}. ' + filler,
                    ) for j in range(2))
                    views.append(View(f'{owner}:view:{side}', records, owner))
            group = Group(f'fixture:{domain}:group:{group_index:02d}', domain, 'fit', tuple(views))
            validate_group(group, True)
            groups.append(group)
        result[domain] = groups
    assert len({v.identity for groups in result.values() for g in groups for v in g.views}) == 480
    assert len({r.uid for groups in result.values() for g in groups for v in g.views for r in v.records}) == 1920
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--permit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--fixture-only', action='store_true')
    args = parser.parse_args()
    permitted = json.loads(args.permit.read_text(encoding='utf-8'))
    approved_python = Path('/home/yongpeng/miniconda3/envs/py310/bin/python').resolve()
    if (sys.platform != 'linux' or Path(sys.executable).resolve() != approved_python
            or permitted.get('operation') != 'Evolution_single_worker_gpu_qualification'
            or permitted.get('approved') is not True or permitted.get('workers') != 1
            or permitted.get('automatic_retry') is not False):
        raise ValueError('Source-bound approved single-worker qualification required')
    source_root = Path(permitted['source_root']).resolve()
    sys.path.insert(0, str(source_root))
    from step30_evolution_io import Budget, record, verify, write

    required = {source_root / ('step30_evolution_' + name + '.py')
                for name in ('policy', 'data', 'run', 'evaluate', 'io', 'control')}
    required |= {source_root / ('step29_vendorlink_' + name + '.py')
                 for name in ('data', 'methods', 'run')}
    if {Path(row['path']).resolve() for row in permitted['sources']} != required:
        raise ValueError('Complete actual production source binding required')
    for row in [permitted['authorization'], permitted['entry'], *permitted['sources']]:
        verify(row)
    if verify(permitted['entry']).resolve() != Path(__file__).resolve():
        raise ValueError('Probe entry differs from the approved source')
    if args.output.exists() or not args.output.resolve().is_relative_to(Path(permitted['root']).resolve()):
        raise ValueError('Fresh output inside the qualification directory required')
    args.output.mkdir(parents=True)
    source_binding = record(args.permit)
    data = fixture_groups()
    if args.fixture_only:
        write(args.output / 'receipt.json', dict(status='PASS', handwritten=True,
              domains=3, groups=60, accounts=480, records=1920, gpu=False,
              pretrained_loaded=False, permit=source_binding))
        return

    import psutil
    import torch
    from step30_evolution_control import configure_gpu
    from step30_evolution_evaluate import training_complete
    from step30_evolution_policy import CELLS, DOMAINS, settings
    from step30_evolution_run import execute, make_model, restore

    binding = permitted['model_binding']
    if Path(binding['directory']).resolve() != Path('/home/yongpeng/cross-lingual/models/step29/bert_base_cased').resolve():
        raise ValueError('Existing approved English backbone only')
    actual_files = {p.resolve() for p in Path(binding['directory']).iterdir()
                    if p.is_file() and p.suffix in ('.json', '.txt', '.bin', '.safetensors')}
    if actual_files != {Path(row['path']).resolve() for row in binding['files']}:
        raise ValueError('Complete actual model and tokenizer binding required')
    for row in binding['files']:
        verify(row)
    configure_gpu()
    total_gpu = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(permitted['allocator_bytes'] / total_gpu, 0)
    budget = Budget(Path(permitted['experiment_root']), args.deadline, 32 * 2**30)
    metrics = dict(cuda_peak_allocated_bytes=0, cuda_peak_reserved_bytes=0,
                   sampled_rss_bytes=0, encode_calls=0, token_lengths={}, runs=[])
    last_save = 0.

    def check():
        nonlocal last_save
        budget.check()
        metrics['cuda_peak_allocated_bytes'] = max(metrics['cuda_peak_allocated_bytes'], torch.cuda.max_memory_allocated())
        metrics['cuda_peak_reserved_bytes'] = max(metrics['cuda_peak_reserved_bytes'], torch.cuda.max_memory_reserved())
        metrics['sampled_rss_bytes'] = max(metrics['sampled_rss_bytes'], psutil.Process().memory_info().rss)
        if time.monotonic() - last_save >= 2:
            write(args.output / 'progress.json', metrics)
            last_save = time.monotonic()

    def factory(task):
        model = make_model(Path(binding['directory']), task)
        original_encode = model.encode

        def encode(texts, limit, hidden=False):
            output, mask = original_encode(texts, limit, hidden)
            if mask.shape[1] != limit:
                raise ValueError('Handwritten fixture failed to reach the requested maximum length')
            key = f'{task["system"]}:{limit}:{len(texts)}:{hidden}'
            metrics['token_lengths'][key] = metrics['token_lengths'].get(key, 0) + 1
            metrics['encode_calls'] += 1
            return output, mask

        model.encode = encode
        return model

    started, success = time.monotonic(), False
    try:
        for system, recipe in (('H_full', 'H3'), ('V_ER', 'V0')):
            check()
            task = dict(id='gpu_fixture_' + system, phase='GPU_FIXTURE', system=system,
                        recipe=recipe, seed=107, stages=3, epochs=1, settings=settings(system, recipe))
            identity = dict(purpose='handwritten_gpu_qualification_not_effect_evidence',
                            permit=source_binding, task=task, actual_data_access=False)
            target = args.output / system
            model = factory(task)
            before = model.encoder.encoder.layer[0].attention.self.query.weight.detach().cpu().clone()
            zero_scores = model.score(data[DOMAINS[0]][0].blind(), check).clone()
            completion = execute(model, lambda d: data[d], target, task, identity, binding, write, record, check)
            training_complete(task, completion, identity)
            delta = float((model.encoder.encoder.layer[0].attention.self.query.weight.detach().cpu() - before).abs().max())
            if not math.isfinite(delta) or delta <= 0:
                raise ValueError('Encoder parameter did not actually change')
            final_scores = model.score(data[DOMAINS[0]][0].blind(), check).clone()
            updates = [json.loads(line) for line in (target / 'updates.jsonl').read_text().splitlines()]
            if len(updates) != 60 or any(row['optimizer_updates'] != 1 for row in updates):
                raise ValueError('Expected sixty actual optimizer updates')
            if any(not math.isfinite(row['gradient_norm']) or row['gradient_norm'] <= 0 for row in updates):
                raise ValueError('Missing finite nonzero update gradients')
            if system == 'H_full':
                history = [row['history'] for row in updates if 'history' in row]
                if len(history) != 40 or not any(row['distillation'] > 0 for row in history):
                    raise ValueError('H replay and birth-score penalty must actually execute')
            elif sum('history_ce' in row for row in updates) != 40:
                raise ValueError('V replay classification must actually execute')
            check()
            # The instrumentation captures this model; collect its reference cycle
            # before constructing a restored model, so no two GPU models coexist.
            del model, before
            gc.collect()
            torch.cuda.empty_cache()
            scored = []
            for stage in range(4):
                check()
                restored = restore(factory, task, verify(completion['endpoints'][str(stage)]), identity, binding, stage)
                for t, position in CELLS:
                    if stage != t:
                        continue
                    scores = restored.score(data[DOMAINS[position - 1]][0].blind(), check)
                    if scores.shape != (120,) or not torch.isfinite(scores).all():
                        raise ValueError('Incomplete or nonfinite restored GPU scores')
                    if stage == 0:
                        torch.testing.assert_close(scores, zero_scores, rtol=0, atol=0)
                    if stage == 3 and position == 1:
                        torch.testing.assert_close(scores, final_scores, rtol=0, atol=0)
                    scored.append(dict(stage=stage, domain=position, pairs=scores.numel()))
                del restored
                gc.collect()
                torch.cuda.empty_cache()
            if len(scored) != 9:
                raise ValueError('Nine predeclared cells required')
            saved = sorted(p.name for p in target.glob('*.pt'))
            if saved != ['initial.pt', 'stage1.pt', 'stage2.pt', 'stage3.pt']:
                raise ValueError('Unexpected periodic or missing compact model files')
            metrics['runs'].append(dict(system=system, recipe=recipe, updates=60,
                  encoder_max_abs_change=delta, workload=completion['workload'], restored_cells=scored,
                  initial_scores_exact=True, final_scores_exact=True, saved_models=saved))
            write(args.output / 'progress.json', metrics)
        budget.check(force=True)
        success = True
    except BaseException as error:
        write(args.output / 'failure.json', dict(error=repr(error), permit=source_binding,
              metrics=metrics, automatic_retry=False, effect_claim=False))
        raise
    finally:
        write(args.output / 'receipt.json', dict(status='PASS' if success else 'FAIL',
              permit=source_binding, measured_body_seconds=time.monotonic() - started,
              metrics=metrics, handwritten=True, real_data_read=False, effect_claim=False,
              accounting='Use the separate complete outer-process lifecycle receipt, including failure/import/save/exit.'))


if __name__ == '__main__':
    main()
