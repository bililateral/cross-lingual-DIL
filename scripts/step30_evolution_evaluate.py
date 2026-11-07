"""Nine-cell blind scoring, complete-batch label release, saved query metrics."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from step29_vendorlink_data import Group
from step30_evolution_data import validate_group
from step30_evolution_io import access, pinned, read, record, verify, write
from step30_evolution_policy import DOMAINS, METRICS, SPLITS, cells, tasks
from step30_evolution_run import restore


def training_complete(task: dict, completion: dict, expected_identity: dict, *, fit_groups: int = 20) -> None:
    stages, epochs = task['stages'], task['epochs']
    expected_updates = fit_groups*epochs*stages
    work = completion['workload']
    replay = task['system'] in ('H_ER', 'H_full', 'V_ER')
    if (completion['status'] != 'TRAINING_COMPLETE_EVALUATION_CLOSED' or completion['task'] != task
            or completion['identity'] != expected_identity or completion['updates'] != expected_updates
            or set(completion['endpoints']) != {str(i) for i in range(stages+1)}
            or work['current_updates'] != expected_updates or work['current_records'] != 32*expected_updates
            or work['stage_model_saves'] != stages or work['history_groups'] != ((stages-1)*epochs*fit_groups if replay else 0)
            or work['history_records'] != 32*work['history_groups']):
        raise ValueError('Incomplete or mismatched trajectory; downstream remains closed')
    if (work['teacher_groups'] > (12 if task['system'] in ('H_ER', 'H_full') else 0)
            or work['teacher_records'] != 32*work['teacher_groups']):
        raise ValueError('Teacher workload differs from contract')


def score_batch(root: Path, batch: dict, manifest: dict, factory, model_binding: dict, check,
                *, fixture_specs=None, fixture_groups=None) -> dict:
    phase, specs = batch['phase'], batch['specs']
    expected_specs = tasks(phase, batch.get('selected')) if fixture_specs is None else fixture_specs
    if specs != expected_specs or len(batch['completed']) != len(specs):
        raise ValueError('Complete fixed training batch required before blind scoring')
    completed = {r['task_id']: r for r in batch['completed']}
    if set(completed) != {s['id'] for s in specs}:
        raise ValueError('Missing/duplicate batch identities')
    for task in specs:
        row = completed[task['id']]
        training_complete(task, pinned(row['completion']), row['identity'], fit_groups=20 if fixture_groups is None else fixture_groups)
    split = 'test' if phase == 'D' else 'dev'
    domains = DOMAINS[:1] if phase == 'A' else DOMAINS
    sources = [manifest['files'][d+':'+split] for d in domains]
    access(root.parent, phase+'_blind_text', sources)
    public, inventory = {}, {}
    for d, source in zip(domains, sources):
        groups = [Group.from_dict(x) for x in pinned(source)]
        if (len(groups) != (SPLITS[split]//8 if fixture_groups is None else fixture_groups) or len({g.uid for g in groups}) != len(groups)
                or any(g.domain != d or g.split != split for g in groups)):
            raise ValueError('Blind quota/domain/partition differs')
        for g in groups:
            validate_group(g, False)
        public[d] = groups
        inventory[d] = {g.uid: [v.uid for v in g.views] for g in groups}
    result = dict(phase=phase, specs=specs, inventory=inventory, blind_sources=sources,
                  model_binding=model_binding, runs=[], forwards=0)
    for task in specs:
        complete = pinned(completed[task['id']]['completion'])
        run = dict(spec=task, cells={})
        for stage in range(task['stages']+1):
            check()
            model = restore(factory, task, verify(complete['endpoints'][str(stage)]),
                            complete['identity'], model_binding, stage)
            for t, position in cells(task['stages']):
                if t != stage:
                    continue
                d, values = DOMAINS[position-1], {}
                for group in public[d]:
                    check()
                    scores = model.score(group, check).numpy().astype(np.float64)
                    if scores.shape != (120,) or not np.isfinite(scores).all():
                        raise ValueError('Nonfinite or incomplete blind pair scores')
                    values[group.uid] = dict(view_ids=[v.uid for v in group.views], scores=scores.tolist())
                target = root/f'{task["id"]}_{stage}_{position}.json'
                write(target, values)
                run['cells'][f'{stage}_{position}'] = record(target)
                result['forwards'] += len(values)
            del model
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        result['runs'].append(run)
    validate_blind(result, specs)
    write(root/'blind_complete.json', result)
    return result


def validate_blind(result: dict, specs: list[dict]) -> None:
    if result['specs'] != specs or len(result['runs']) != len(specs):
        raise ValueError('Blind batch specifications differ')
    by_id = {r['spec']['id']: r for r in result['runs']}
    if set(by_id) != {s['id'] for s in specs}:
        raise ValueError('Missing/extra blind task')
    forwards = 0
    for task in specs:
        run = by_id[task['id']]
        if run['spec'] != task or set(run['cells']) != {f'{t}_{d}' for t, d in cells(task['stages'])}:
            raise ValueError('Missing/extra/rebound blind cell')
        for t, d in cells(task['stages']):
            values = pinned(run['cells'][f'{t}_{d}'])
            expected = result['inventory'][DOMAINS[d-1]]
            if set(values) != set(expected):
                raise ValueError('Blind candidate groups differ')
            for uid, row in values.items():
                a = np.asarray(row['scores'])
                if row['view_ids'] != expected[uid] or a.shape != (120,) or not np.isfinite(a).all():
                    raise ValueError('Blind IDs/shape/finiteness failure before label opening')
            forwards += len(values)
    if forwards != result['forwards']:
        raise ValueError('Blind forward count differs')


def query_metrics(scores, labels, ids: list[str]) -> dict:
    if len(ids) != 16 or len(set(ids)) != 16:
        raise ValueError('Sixteen distinct view IDs required')
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    if scores.shape != (120,) or labels.shape != (120,) or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Finite complete 120-pair scores and binary qrels required')
    table, truth = np.zeros((16, 16)), np.zeros((16, 16), dtype=bool)
    upper = np.triu_indices(16, 1)
    table[upper], truth[upper] = scores, labels
    table += table.T
    truth |= truth.T
    if not (truth.sum(1) == 1).all():
        raise ValueError('Exactly one positive among fifteen candidates per query')
    ranks = []
    for q in range(16):
        order = sorted((i for i in range(16) if i != q), key=lambda i: (-table[q, i], ids[i].encode('utf-8')))
        ranks.append(next(k+1 for k, i in enumerate(order) if truth[q, i]))
    ranks = np.asarray(ranks)
    result = dict(map=(1/ranks).tolist(), mrr=(1/ranks).tolist())
    for k in (1, 3, 5, 10):
        result[f'recall@{k}'] = (ranks <= k).astype(float).tolist()
        result[f'ndcg@{k}'] = np.where(ranks <= k, 1/np.log2(ranks+1), 0.).tolist()
    return result


def analyze_batch(root: Path, batch: dict, manifest: dict, *, fixture_specs=None, fixture_groups=None) -> list[dict]:
    phase, specs = batch['phase'], batch['specs']
    blind = read(root/'blind_complete.json')
    expected_specs = tasks(phase, batch.get('selected')) if fixture_specs is None else fixture_specs
    if specs != expected_specs or blind['phase'] != phase:
        raise ValueError('Label release phase/configuration differs')
    validate_blind(blind, specs)  # ALL scores checked before ANY qrels access.
    split = 'test' if phase == 'D' else 'dev'
    domains = DOMAINS[:1] if phase == 'A' else DOMAINS
    if (set(blind['inventory']) != set(domains) or
            blind['blind_sources'] != [manifest['files'][d+':'+split] for d in domains] or
            any(len(blind['inventory'][d]) != (SPLITS[split]//8 if fixture_groups is None else fixture_groups) for d in domains)):
        raise ValueError('Blind dataset sources/domains/quotas differ before qrels opening')
    sources = [manifest['files'][d+':'+split+':qrels'] for d in domains]
    access(root.parent, phase+'_qrels', sources)
    qrels = {d: pinned(row) for d, row in zip(domains, sources)}
    for d in domains:
        if set(qrels[d]) != set(blind['inventory'][d]):
            raise ValueError('Qrels candidate group identity differs')
    output = []
    for run in blind['runs']:
        item = dict(spec=run['spec'], cells={})
        for t, position in cells(run['spec']['stages']):
            d, key = DOMAINS[position-1], f'{t}_{position}'
            values, groups = pinned(run['cells'][key]), []
            for uid in sorted(values):
                row, truth = values[uid], qrels[d][uid]
                if row['view_ids'] != truth['view_ids']:
                    raise ValueError('Qrels and blind score order differ')
                metrics = query_metrics(row['scores'], truth['labels'], row['view_ids'])
                groups.append(dict(uid=uid, view_ids=row['view_ids'], queries=metrics))
            means = {m: float(np.mean([g['queries'][m] for g in groups])) for m in METRICS}
            item['cells'][key] = dict(mean=means, groups=groups)
        output.append(item)
    write(root/'metrics.json', output)
    return output
