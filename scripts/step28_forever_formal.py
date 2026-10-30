"""Nine FOREVER trajectories, one worker, existing stage mathematics unchanged.

Training and new-confirmation access are distinct. This entry produces all 27
inference endpoints; it does not open old or new evaluation labels.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

ENTRY_STARTED = time.monotonic()

import step28_forever_run as run
import step28_formal_queue as queue
from step28_continual_comparison_run import Budget
from step28_continual_five_run import COMPLETE as H_COMPLETE

m, data = run.m, run.m.data
COMPLETE = 'COMPLETE_FOREVER_864_CURRENT_3_ENDPOINTS_BLIND'


def specs():
    return [m.Spec(order, seed) for order in ('ABC', 'BCA', 'CAB') for seed in (4, 5, 6)]


def validate_scope(allowed):
    if (allowed['runs'] != [s.identity() for s in specs()]
            or allowed['epochs'] != 6 or allowed['extra_full_parameter_anchor'] is not True
            or allowed['budget_seconds']['training'] != 108 * 3600
            or allowed['per_run_seconds'] != 12 * 3600
            or allowed.get('early_evaluation_opening') is not False
            or allowed.get('automatic_retry') is not False):
        raise ValueError('Fixed nine-run FOREVER scope differs')


def verify_reference(allowed):
    """Only already-approved completion/calibration metadata and model identities."""
    if len(allowed['h_reference_runs']) != 9:
        raise ValueError('Nine existing H trajectories required; never retrain missing H')
    old_gate = data.read_json(queue.verify(allowed['h_reference_gate']))
    seen = set()
    for item in allowed['h_reference_runs']:
        row = data.read_json(queue.verify(item['completion']))
        identity = row['spec']
        key = (identity['order'], identity['seed'])
        if (row['status'] != H_COMPLETE or row['source_files'] != old_gate['sources']
                or identity['name'] != 'H_pos_tuned' or key in seen
                or key not in {(s.order, s.seed) for s in specs()}
                or row['updates'] != 864 or set(row['points']) != {'1', '2', '3'}):
            raise ValueError('Pinned H reference is incomplete/foreign')
        seen.add(key)
        root = Path(item['completion']['path']).parent
        for point in row['points'].values():
            for name in ('model', 'mapping'):
                data.verify(root/point[name]['path'], point[name])
            if not point['inference_restore_verified'] or not point['full_restore_verified']:
                raise ValueError('H endpoint restoration evidence missing')


def validate_completion(spec, root, allowed):
    row = data.read_json(root/'completion.json')
    if (row['status'] != COMPLETE or row['spec'] != spec.identity()
            or row['source_files'] != allowed['sources'] or set(row['points']) != {'1', '2', '3'}
            or row['counts']['current'] != 864 or row['counts']['fixture'] != 0
            or not 0 <= row['counts']['replay'] <= 228
            or row['access'] != dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)):
        raise ValueError('FOREVER complete training/count/source gate failed')
    for number in (1, 2, 3):
        point = row['points'][str(number)]
        expected = [r['group_uid'] for r in row['partition']['calibration']
                    if r['domain'] == spec.order[number - 1]]
        if (point['stage'] != number or point['counts']['current'] != 288 * number
                or point['calibration_ids'] != expected or len(expected) != 12
                or not point['inference_restore_verified']):
            raise ValueError('FOREVER endpoint or current calibration differs')
        data.verify(root/point['model']['path'], point['model'])
        fitted = data.read_json(data.verify(root/point['calibration']['path'], point['calibration']))
        if (fitted['status'] != 'PASS_CALIBRATION_FIT'
                or dict(zip(('a', 'b'), m.parent.calibration.parameters(fitted))) != point['maps']['stage']):
            raise ValueError('FOREVER fitted calibration differs')
    return row


def retire_completed(store, spec, allowed):
    """Retire only completed run's recovery payloads after all three endpoints verify."""
    validate_completion(spec, store.root, allowed)
    latest = data.read_json(store.root/'latest.json')
    if latest['metadata']['state']['phase'] != 'complete':
        raise ValueError('Incomplete run still needs its full state')
    retired = []
    for rec in (latest['full']['file'], latest['metadata']['anchor']):
        if rec:
            path = store.verify(rec)
            if path.parent.name not in ('work', 'anchors'):
                raise ValueError('Unexpected recovery path')
            retired.append(rec)
    # Retain provenance before deletion. Completed inference models/Memory remain.
    data.write_json(store.root/'recovery_retirement.json', dict(reason='complete_no_automatic_resume',
        latest=latest, retired=retired, status='PLANNED'))
    for rec in retired:
        store.verify(rec).unlink()
    (store.root/'latest.json').unlink()
    data.write_json(store.root/'recovery_retirement.json', dict(reason='complete_no_automatic_resume',
        latest=latest, retired=retired, status='DELETED'))


def execute(job, spec, deadline, allowed):
    import torch
    root = job/'run'
    if root.exists():
        raise ValueError('Fresh run only; recovery requires a separate explicit scope')
    root.mkdir()
    budget = Budget(Path(allowed['campaign']), deadline, allowed['limits'])
    budget.check()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(allowed['limits']['allocator_bytes'] /
        torch.cuda.get_device_properties(0).total_memory)
    c = m.config(spec)
    archive = m.core.model_files(m.five.base.model_config(c, 'split_rank'))
    if any(archive[k] != c['models']['split_rank'][k]
           for k in ('file_count', 'total_size_bytes', 'content_sha256')):
        raise ValueError('BGE pretrained source differs')
    access = dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)
    data.write_json(job/'access.json', access)  # Attempts count before parsing.
    supplied, partition, checked = run.training_inputs(c)
    supply = run.Supply(supplied, spec)
    model, optimizer = m.fresh(spec, c)
    learner = m.Learner(spec, model, optimizer, c)
    identity = dict(spec=spec.identity(), source_files=allowed['sources'], config=c)
    store = run.Store(root, identity, budget.check)
    result = dict(status='TRAINING', spec=spec.identity(), source_files=allowed['sources'],
        access=access, partition=partition, public_inputs=checked, pretrained_archive=archive, points={})
    data.write_json(root/'startup.json', result)
    def log(row):
        with (root/'updates.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(dict(wall_time_epoch=time.time(), **row)) + '\n')
        if learner.counts['current'] % 24 == 0:
            print(json.dumps(dict(event='updates', stage=learner.stage, counts=learner.counts)), flush=True)
    for number in (1, 2, 3):
        result['points'][str(number)] = run.stage(learner, supply, store, log=log)
        data.write_json(root/'progress.json', {**result, 'counts': learner.counts})
    result.update(status=COMPLETE, counts=learner.counts.copy(), resources=budget.summary())
    data.write_json(root/'completion.json', result)
    retire_completed(store, spec, allowed)
    del learner, model, optimizer, supply, supplied
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('train', 'worker'))
    parser.add_argument('--gate', required=True, type=Path)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--order')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    allowed = queue.gate(args.gate, 'FOREVER', 'train')
    validate_scope(allowed)
    if args.action == 'worker':
        spec = m.Spec(args.order, args.seed)
        if args.job.resolve() != Path(allowed['campaign']).resolve()/spec.run_id:
            raise ValueError('Worker path differs from its approved trajectory')
        try:
            execute(args.job, spec, args.deadline, allowed)
        except BaseException as error:
            data.write_json(args.job/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
            raise
    else:
        rows = [dict(id=s.run_id, order=s.order, seed=s.seed, seconds=allowed['per_run_seconds']) for s in specs()]
        followups = [dict(id=directory, action=action, bucket=bucket, gpu=gpu,
            seconds=allowed['budget_seconds'][bucket] - allowed['prior_used_seconds'].get(bucket, 0))
            for directory, action, bucket, gpu in (('generation', 'generate', 'generation', False),
                ('confirmation', 'confirmation', 'confirmation', True), ('analysis', 'analyze', 'analysis', False))]
        def command(row, job, deadline):
            if 'action' in row:
                return [sys.executable, '-B', str(Path(__file__).with_name('step28_forever_evaluate.py')),
                        row['action'], '--gate', str(args.gate.resolve()), '--deadline', str(deadline)]
            return [sys.executable, '-B', __file__, 'worker', '--gate', str(args.gate.resolve()),
                    '--job', str(job), '--order', row['order'], '--seed', str(row['seed']), '--deadline', str(deadline)]
        def validate(row, job):
            if 'action' not in row:
                return validate_completion(m.Spec(row['order'], row['seed']), job/'run', allowed)
            expected = dict(generate='GENERATED_VALIDATED_OWNERS_SEALED',
                            confirmation='ALL_FOREVER_METRICS_COMPLETE', analyze='FOREVER_FIXED_COMPARISON_COMPLETE')
            if data.read_json(job/'result.json')['status'] != expected[row['action']]:
                raise ValueError('Post-training action did not complete')
        queue.dispatch(allowed, rows, command, validate, followups,
                       preflight=lambda: verify_reference(allowed), started=ENTRY_STARTED)


if __name__ == '__main__':
    main()
