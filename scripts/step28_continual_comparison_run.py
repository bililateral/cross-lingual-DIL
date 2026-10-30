"""Independent fixed-comparison trajectory. Requires its own approved runtime gate."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import psutil

import step28_continual_comparison as m
from step28_bge_continual_run import save_array

data = m.data
COMPLETE = 'COMPLETE_COMPARISON_864_UPDATES_3_ENDPOINTS_BLIND'
DESIGN = 'reports/documentation/20261024/continual_comparison'


def sources() -> list[dict]:
    pending = [data.ROOT/'scripts'/name for name in (
        'step28_continual_comparison.py', 'step28_comparison_l2r.py', 'step28_comparison_gcr.py',
        'step28_continual_comparison_run.py', 'step28_continual_comparison_evaluate.py')]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names = ([n.name for n in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) else [])
            pending.extend(data.ROOT/'scripts'/(name+'.py') for name in names if name and name.startswith('step28_'))
    seen.update(data.ROOT/'schema'/(name+'_policy.json') for name in (
        'step28_chinese_base', 'step28_alias_ranking', 'step28_record_replay',
        'step28_bge_continual', 'step28_continual_expression'))
    seen.add(data.ROOT/'schema/step28_continual_expression_catalog.json')
    seen.add(data.ROOT/'tests/test_step28_continual_comparison.py')
    seen.update(data.ROOT/DESIGN/name for name in (
        'proposal.zh.txt', 'review/primary_disposition.zh.txt', 'review/revision_disposition.zh.txt',
        'review/closure/primary.zh.txt', 'implementation/review/numerical_amendment.proposed.zh.txt',
        'implementation/review/numerical_authorization.json'))
    return [data.record(path, data.ROOT) for path in sorted(seen)]


def gate(path: Path, action: str) -> dict:
    row = data.read_json(path)
    if (row.get('status') != 'APPROVED_COMPARISON_IMPLEMENTATION_AND_RESOURCES'
            or row.get('sources') != sources() or action not in row.get('approved_actions', [])
            or row.get('new_runs') != [s.identity() for s in m.specifications(new_only=True)]):
        raise ValueError('Independent comparison implementation/resource/data gate missing')
    return row


class Budget:
    """Actual new allocation, never the previous campaign's 490-hour pool."""
    def __init__(self, root: Path, deadline: float, limits: dict):
        self.root, self.deadline, self.limits = root, deadline, limits
        self.started, self.last = time.monotonic(), 0.
        self.peak_rss = self.peak_disk = 0
        for name in ('rss_bytes', 'disk_bytes', 'small_bytes', 'free_reserve_bytes', 'allocator_bytes'):
            if not isinstance(limits[name], int) or limits[name] <= 0:
                raise ValueError('A positive separately approved resource allocation is required')

    def check(self, reserve: int = 0):
        now = time.monotonic()
        if now >= self.deadline-60:
            raise TimeoutError('Allocated deadline reached with shutdown reserve')
        if now-self.last < 2 and not reserve:
            return
        self.last = now
        process = psutil.Process()
        rss = 0
        for p in [process, *process.children(recursive=True)]:
            try:
                rss += p.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        files = [(p.suffix, p.stat().st_size) for p in self.root.rglob('*') if p.is_file() and not p.is_symlink()]
        used, small = sum(size for _, size in files), sum(size for suffix, size in files if suffix != '.pt')
        self.peak_rss, self.peak_disk = max(self.peak_rss, rss), max(self.peak_disk, used)
        if (rss > self.limits['rss_bytes'] or used+reserve > self.limits['disk_bytes']
                or small > self.limits['small_bytes']
                or shutil.disk_usage(self.root).free < self.limits['free_reserve_bytes']+reserve):
            raise RuntimeError('Independent comparison resource allowance exceeded')

    def summary(self) -> dict:
        return dict(wall_seconds=time.monotonic()-self.started, peak_rss_bytes=self.peak_rss,
                    peak_disk_bytes=self.peak_disk, deadline_monotonic=self.deadline)


def training_inputs(c: dict) -> tuple:
    """Train-only dispatcher: never opens old dev text, labels, owners or test."""
    root = data.ROOT/c['data_root']
    for name, key in (('manifest.json', 'data_manifest_sha256'), ('validation.json', 'data_validation_sha256')):
        if data.sha256(root/name) != c[key]:
            raise ValueError('Pinned training archive differs')
    manifest = data.read_json(root/'manifest.json')
    checked = {}
    def path(name):
        checked[name] = manifest['files'][name]
        return data.verify(root/name, checked[name])
    with path('groups.csv').open(encoding='utf-8', newline='') as stream:
        metadata = sorted((r for r in csv.DictReader(stream) if r['split'] == 'train'),
                          key=lambda r: (r['domain'], int(r['group_index'])))
    collected, sellers_seen, items_seen = defaultdict(lambda: defaultdict(list)), set(), set()
    with path('train/items.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if set(item) != {'group_uid', 'seller_uid', 'item_uid', 'title', 'description'} or item['item_uid'] in items_seen:
                raise ValueError('Training item identity/schema differs')
            items_seen.add(item['item_uid'])
            collected[item['group_uid']][item['seller_uid']].append((item['item_uid'], item['title'], item['description']))
    if len(metadata) != 180 or set(collected) != {r['group_uid'] for r in metadata}:
        raise ValueError('Complete train-only group coverage required')
    groups = []
    for row in metadata:
        items = collected[row['group_uid']]
        sellers = tuple(sorted(items))
        if sellers_seen & set(sellers):
            raise ValueError('Training identities overlap groups')
        sellers_seen.update(sellers)
        group = data.Group(row['group_uid'], sellers, tuple(tuple(sorted(items[s])) for s in sellers))
        group.validate()
        if int(row['accounts']) != 28 or int(row['items']) != sum(map(len, group.items)):
            raise ValueError('Training shape differs')
        groups.append(group)
    labelled = m.base.public.attach_labels(groups, c, 'train')
    checked['train/supervision/pairs.csv'] = manifest['files']['train/supervision/pairs.csv']
    fit_ids = set()
    for domain in 'ABC':
        rows = [r for r in metadata if r['domain'] == domain]
        if sorted(int(r['group_index']) for r in rows) != list(range(60)):
            raise ValueError('Fixed domain stream differs')
        ids = sorted((r['group_uid'] for r in rows), key=lambda uid: hashlib.sha256(
            data.json_bytes([c['partition']['seed'], domain, uid])).hexdigest())
        fit_ids.update(ids[:48])
    partition = {role: [dict(group_uid=g.uid, domain=row['domain']) for g, row in zip(labelled, metadata, strict=True)
                        if (g.uid in fit_ids) == (role == 'fit')] for role in ('fit', 'calibration')}
    values = {role: [g for g in labelled if (g.uid in fit_ids) == (role == 'fit')] for role in partition}
    return m.parent.Supply(values, partition), partition, checked


def save(path, model, optimizer, metadata, budget):
    budget.check(m.base.persistence.checkpoint_reserve(model, optimizer))
    return m.core.save_state(path, model, optimizer, metadata)


def checkpoint(root, method, model, optimizer, stage, calibration, c, first_map, budget):
    import torch
    number = stage.stage
    if len(calibration) != 12 or m.parent.adam_step(optimizer) != 288*number:
        raise ValueError('Current calibration/stage endpoint differs')
    optimizer.zero_grad(set_to_none=True)
    rng, modes = m.diag.rng_state(), [v.training for v in model.modules()]
    diagnostic_path = None
    if c.get('diagnostic_calibration') and number == 1:
        if method.spec.run_id != 'CAB_s6_L2R_text_fixed':
            raise ValueError('Diagnostic state is limited to the failed CAB_s6 first stage')
        diagnostic_path = root/'work'/'stage1_precalibration.pt'
        diagnostic_metadata = {**method.metadata(), 'stage': 1, 'adam_step': 288,
                               'rng': rng, 'modes': modes, 'transient': stage.transient(),
                               'memory': stage.memory.to_bytes().decode('utf-8'),
                               'calibration_ids': [g.uid for g in calibration]}
        diagnostic = save(diagnostic_path, model, optimizer, diagnostic_metadata, budget)
        data.write_json(root/'precalibration.json', dict(state=diagnostic,
                        purpose='One authorized CAB_s6 calibration diagnosis; no automatic retry'))
    with m.diag.neutral(model):
        scores = method.score(model, calibration, c, budget.check)
        if diagnostic_path is not None:
            save_array(root/'scores'/'stage1_diagnostic.npy', scores, root)
        fitted = m.parent.calibration.fit(scores, np.asarray([g.labels for g in calibration], np.uint8),
                                          role='calibration', check=budget.check)
        if diagnostic_path is not None:
            data.write_json(root/'maps'/'stage1_diagnostic.json', fitted)
            print(json.dumps(dict(event='diagnostic_calibration', run_id=method.spec.run_id,
                  status=fitted['status'], optimizer_message=fitted['optimizer_message'],
                  projected_gradient_max=fitted['projected_gradient_max'])), flush=True)
        if fitted['status'] != 'PASS_CALIBRATION_FIT':
            # Keep the returned diagnostics before unwinding; do not refit or relax the gate.
            failure_path = root/'maps'/f'stage{number}_failed.json'
            data.write_json(failure_path, fitted)
            save_array(root/'scores'/f'stage{number}_calibration_failed.npy', scores, root)
            raise ValueError('Current calibration failed; no retry or confirmation refit; '
                             f'diagnostics={failure_path}; optimizer={fitted["optimizer_message"]}; '
                             f'projected_gradient_max={fitted["projected_gradient_max"]}')
        current_map = dict(zip(('a', 'b'), m.parent.calibration.parameters(fitted)))
        first_map = current_map if first_map is None else first_map
        # GCR maintenance differentiates the real head; neutral() disables autograd.
        with torch.inference_mode(False), torch.enable_grad():
            maintenance = stage.retain(model, c, budget.check)
    memory_path = root/'memory'/f'stage{number}.bin'
    memory_path.write_bytes(stage.memory.to_bytes())
    memory = method.memory_from(memory_path.read_bytes())
    metadata = {**method.metadata(), 'stage': number, 'adam_step': number*288, 'rng': rng, 'modes': modes,
                'maps': {'first': first_map, 'stage': current_map}, 'memory': data.record(memory_path, root)}
    full_path, inference_path = root/'work'/f'stage{number}.pt', root/'models'/f'stage{number}.pt'
    full = save(full_path, model, optimizer, metadata, budget)
    if method.restore(full_path, full, model, optimizer) != metadata:
        raise ValueError('Full restored metadata differs')
    with m.diag.neutral(model):
        if not np.array_equal(scores, method.score(model, calibration, c, budget.check)):
            raise ValueError('Full restored calibration scores differ')
    inference = save(inference_path, model, None, metadata, budget)
    if method.restore(inference_path, inference, model, None) != metadata:
        raise ValueError('Inference restored metadata differs')
    with m.diag.neutral(model):
        if not np.array_equal(scores, method.score(model, calibration, c, budget.check)):
            raise ValueError('Inference restored calibration scores differ')
    map_path = root/'maps'/f'stage{number}.json'
    data.write_json(map_path, fitted)
    point = {**metadata, 'calibration_ids': [g.uid for g in calibration],
             'mapping': data.record(map_path, root), 'maintenance': maintenance, 'memory_summary': memory.summary(),
             'full_restore_verified': True, 'inference_restore_verified': True,
             'full_state': {**full, 'path': full_path.relative_to(root).as_posix()},
             'model': {**inference, 'path': inference_path.relative_to(root).as_posix()},
             'calibration_scores': save_array(root/'scores'/f'stage{number}_calibration.npy', scores, root)}
    data.write_json(root/'points'/f'stage{number}.json', point)
    m.base.persistence.remove_work_file(full_path, root)
    if diagnostic_path is not None:
        m.base.persistence.remove_work_file(diagnostic_path, root)
        data.write_json(root/'precalibration_retired.json', dict(reason='Stage passed and both restored endpoints verified'))
    return point, memory, first_map


def execute(job: Path, spec: m.Spec, deadline: float, gate_path: Path) -> dict:
    import torch
    allowed = gate(gate_path, 'train')
    if spec.name not in m.NEW_METHODS or allowed.get('train_data') is not True or job.exists():
        raise ValueError('Only authorized new training in a fresh directory is allowed')
    job.mkdir(parents=True)
    budget = Budget(job, deadline, allowed['limits'])
    budget.check()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(allowed['limits']['allocator_bytes']/torch.cuda.get_device_properties(0).total_memory)
    method, c = m.Method(spec), m.config(spec)
    if allowed.get('recovery') and spec.run_id == 'CAB_s6_L2R_text_fixed':
        c['diagnostic_calibration'] = True
    archive = m.core.model_files(m.base.model_config(c, 'split_rank'))
    if any(archive[k] != c['models']['split_rank'][k] for k in ('file_count', 'total_size_bytes', 'content_sha256')):
        raise ValueError('Pretrained BGE source differs')
    access = dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)
    data.write_json(job/'access.json', access)  # Count attempt before train parsing.
    supply, partition, checked = training_inputs(c)
    root = job/'run'
    for name in ('models', 'work', 'memory', 'maps', 'scores', 'points', 'updates'):
        (root/name).mkdir(parents=True)
    model, optimizer = method.fresh(c)
    memory, first_map = method.memory(), None
    result = dict(status='RUNNING', spec=spec.identity(), source_files=allowed['sources'], access=access,
                  partition=partition, public_inputs=checked, pretrained_archive=archive, points={}, updates=0)
    data.write_json(root/'startup.json', result)
    for number in (1, 2, 3):
        current, calibration = supply.current(spec.run_id, spec.order, number)
        stage = method.stage(number, current, memory, model)
        sequence, seed = m.parent.schedule(current, c, spec.order, number)
        rows = []
        for step, group in enumerate(sequence, 1):
            row = method.update(model, optimizer, stage, group, c, step, budget.check)
            rows.append(dict(step=step, current_uid=group.uid, **row))
            if step == 1 or step % 24 == 0:
                print(json.dumps(dict(event='updates', run_id=spec.run_id, stage=number, completed=step,
                                      **budget.summary())), flush=True)
        data.write_json(root/'updates'/f'stage{number}.json', dict(schedule_seed=seed, rows=rows))
        point, memory, first_map = checkpoint(root, method, model, optimizer, stage, calibration,
                                              c, first_map, budget)
        result['points'][str(number)] = point
        result['updates'] += 288
        data.write_json(root/'progress.json', result)
        del stage, current, calibration, sequence
    result.update(status=COMPLETE, resources=budget.summary())
    data.write_json(root/'completion.json', result)
    del model, optimizer, memory, supply
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    try:
        execute(args.job, m.Spec.from_dict(data.read_json(args.spec)), args.deadline, args.gate)
    except BaseException as error:
        if args.job.is_dir():
            data.write_json(args.job/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                                                        status='STOP_NO_RETRY'))
        raise
