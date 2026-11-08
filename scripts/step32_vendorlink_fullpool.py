"""Fixed-market dev diagnosis of six existing ABC trajectories; never trains."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil
import torch

from step29_vendorlink_data import Group
from step29_vendorlink_methods import Model, Settings
from step30_evolution_control import Runner, configure_gpu
from step30_evolution_io import Ledger, PYTHON, access, directory_bytes, pinned, read, record, verify, write
from step31_evolution_registered_evaluate import encode_views, rectangular

PROTOCOL = 'vendorlink-fullpool-dev-20261108'
DOMAINS = ('AlphaBay', 'Dream', 'Agora')
COUNTS = dict(AlphaBay=386, Dream=120, Agora=502)
GROUPS = dict(AlphaBay=24, Dream=8, Agora=31)
CELLS = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3))
METRICS = ('MAP', 'MRR', 'Recall@1', 'Recall@3', 'Recall@5', 'Recall@10',
           'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10')
METHODS = ('H_SEQ', 'VendorLink_SEQ')
CAPS = dict(gpu=5400, auxiliary=1800)
LIMITS = dict(workers=2, free_gpu_bytes=14*2**30, process_gpu_bytes=14*2**30,
              allocator_bytes=13*2**30, rss_bytes=8*2**30, free_ram_bytes=8*2**30,
              disk_bytes=2*2**30, free_disk_reserve=2*2**30, save_reserve_bytes=16*2**20)
SOURCES = ('step32_vendorlink_fullpool.py', 'step31_evolution_registered_evaluate.py',
           'step31_evolution_registered_policy.py', 'step30_evolution_control.py',
           'step30_evolution_io.py', 'step30_evolution_policy.py', 'step30_evolution_data.py',
           'step30_evolution_evaluate.py', 'step30_evolution_run.py',
           'step29_vendorlink_data.py', 'step29_vendorlink_methods.py',
           'step29_vendorlink_run.py', 'step28_formal_queue.py')


def tasks():
    return [dict(id=f'ABC_s{s}_{m}', seed=s, method=m, order='ABC')
            for s in (4, 5, 6) for m in METHODS]


class Check:
    def __init__(self, root, deadline):
        self.root, self.deadline, self.last = Path(root), deadline, 0.

    def __call__(self):
        now = time.monotonic()
        if now >= self.deadline-10:
            raise TimeoutError('Full worker lifecycle/shutdown reserve')
        if now-self.last >= 2:
            self.last = now
            if (psutil.Process().memory_info().rss > LIMITS['rss_bytes']
                    or directory_bytes(self.root)+LIMITS['save_reserve_bytes'] > LIMITS['disk_bytes']
                    or shutil.disk_usage(self.root).free < LIMITS['free_disk_reserve']):
                raise RuntimeError('Bounded inference RAM/disk limit')


def permit(path):
    gate = read(path)
    if (sys.platform != 'linux' or Path(sys.executable).resolve() != Path(PYTHON).resolve()
            or gate.get('protocol') != PROTOCOL or gate.get('approved') is not True
            or gate.get('review_exempt_by_user') is not True or gate.get('caps') != CAPS
            or gate.get('limits') != LIMITS or gate.get('automatic_retry') is not False
            or gate.get('automatic_next_phase') is not False or gate.get('automatic_wait') is not False):
        raise ValueError('Exact approved diagnostic scope and user review exemption required')
    root, workspace = Path(gate['experiment_root']).resolve(), Path(gate['workspace']).resolve()
    if workspace != root/'workspace' or Path(__file__).resolve().parent != workspace/'scripts':
        raise ValueError('Use the isolated diagnostic deployment')
    if {Path(r['path']).resolve() for r in gate['sources']} != {workspace/'scripts'/x for x in SOURCES}:
        raise ValueError('All actual entry/dependencies must be source-bound')
    for row in [*gate['sources'], gate['authorization'], gate['scope'], gate['inputs']]:
        verify(row)
    inputs = pinned(gate['inputs'])
    if ({r['id'] for r in inputs['old_runs']} != {t['id'] for t in tasks()}
            or len(inputs['old_runs']) != 6 or inputs['dev_queries'] != COUNTS):
        raise ValueError('Exactly the six old ABC trajectories and three dev pools required')
    exported = {r['path']: r for r in pinned(inputs['export_manifest'])['files']}
    for kind, suffix in (('dev_sources', 'dev.json'), ('dev_qrels', 'dev_qrels.json')):
        if set(inputs[kind]) != set(DOMAINS):
            raise ValueError('Only the three existing dev exports are permitted')
        for domain, row in inputs[kind].items():
            if exported.get(row['path']) != row or Path(row['path']).name != f'{domain}_{suffix}':
                raise ValueError('Data binding differs from the original export manifest')
    return gate, inputs


def old_completion(binding, check=lambda: None):
    value = pinned(binding['completion'])
    if (value['identity']['spec'] != binding['spec'] or binding['spec']['id'] != binding['id']
            or binding['spec'] not in tasks() or value['identity']['epochs'] != 40
            or value['identity']['sources'] != binding['sources'] or value['updates'] != 10280
            or value['inference_endpoints'] is not True
            or value['stage_endpoints'] != ['stage1.pt', 'stage2.pt', 'stage3.pt']
            or set(value['endpoint_files']) != {'1', '2', '3'}
            or value['access'] != dict(fit_domains=list(DOMAINS), dev=0, heldout=0, qrels=0, owners=0)):
        raise ValueError('Old completion identity/settings/access mismatch')
    for stage, source in value['endpoint_files'].items():
        check()
        if Path(source['path']).resolve() != Path(binding['completion']['path']).parent/f'stage{stage}.pt':
            raise ValueError('Foreign endpoint path')
        verify(source)
    return value


def load_public(inputs, target):
    access(target, 'dev_blind_text', [inputs['dev_sources'][d] for d in DOMAINS])
    pools, groups = {}, {}
    for domain in DOMAINS:
        original = [Group.from_dict(g) for g in pinned(inputs['dev_sources'][domain])]
        items = sorted((v for g in original for v in g.views), key=lambda v: v.uid)
        uids = [v.uid for v in items]
        records = [r.uid for v in items for r in v.records]
        if (len(original) != GROUPS[domain] or len({g.uid for g in original}) != len(original)
                or len(items) != COUNTS[domain] or len(set(uids)) != len(uids)
                or len(set(records)) != len(records)
                or any(g.domain != domain or g.split != 'dev' for g in original)
                or any(v.identity is not None or len(v.records) != 2 for v in items)):
            raise ValueError('Fixed blind dev views/counts/record isolation differ')
        pools[domain], groups[domain] = items, original
    return pools, groups


def make_model(inputs, settings):
    from transformers import BertModel, BertTokenizer
    return Model(BertModel.from_pretrained(inputs['model_path'], local_files_only=True),
                 BertTokenizer.from_pretrained(inputs['model_path'], local_files_only=True,
                                               do_lower_case=True), settings).to('cuda:0')


def score_one(target, task, inputs, check, factory=make_model):
    binding = next(r for r in inputs['old_runs'] if r['id'] == task['id'])
    completion = old_completion(binding, check)
    pools, groups = load_public(inputs, target)
    settings = Settings(method=task['method'])
    model = factory(inputs, settings)
    result = dict(protocol=PROTOCOL, task=task, original_completion=binding['completion'],
                  dev_sources=inputs['dev_sources'], cells={}, checkpoints=[], old_group_check=None)
    for stage in (1, 2, 3):
        check()
        source = completion['endpoint_files'][str(stage)]
        state = torch.load(verify(source), map_location='cpu', weights_only=False)
        if (state['identity'] != completion['identity'] or state['stage'] != stage
                or asdict(Settings(**state['settings'])) != asdict(settings)):
            raise ValueError('Restored old endpoint identity/configuration differs')
        if task['method'] == 'VendorLink_SEQ':
            model.start_vendor_stage(state['classes'], 0)
        model.load_state_dict(state['model'], strict=True)
        del state
        result['checkpoints'].append(source)
        for _, position in (c for c in CELLS if c[0] == stage):
            check()
            domain, key = DOMAINS[position-1], f'{stage}_{position}'
            items = pools[domain]
            ids = [v.uid for v in items]
            encoded = encode_views(model, items, check)
            scores = rectangular(model, encoded, encoded, check)
            if not np.allclose(scores, scores.T, atol=2e-5, rtol=2e-5):
                raise ValueError('Symmetric account scoring mismatch')
            # One existing group per trajectory verifies the actual new inference path.
            if result['old_group_check'] is None:
                g = groups[domain][0]
                positions = [ids.index(v.uid) for v in g.views]
                within = scores[np.ix_(positions, positions)][np.triu_indices(len(positions), 1)]
                reference = model.score(g, check).numpy()
                maximum = float(np.max(np.abs(within-reference)))
                if not np.allclose(within, reference, atol=2e-5, rtol=2e-5):
                    raise ValueError('New full pool disagrees with original within-group scoring')
                result['old_group_check'] = dict(domain=domain, pairs=len(reference),
                    max_absolute_difference=maximum, atol=2e-5, rtol=2e-5, passed=True)
            path = target/f'{key}.npz'
            np.savez(path, ids=np.array(ids), scores=scores)
            result['cells'][key] = dict(domain=domain, query_count=len(ids), candidate_count=len(ids)-1,
                                       matrix=record(path))
            write(target/'progress.json', dict(task=task['id'], completed_cells=len(result['cells']),
                                              total_cells=8, effects_open=False))
            del encoded, scores
    del model
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    validate_scores(result, task)
    write(target/'blind_complete.json', result)


def validate_scores(result, task):
    if (result['protocol'] != PROTOCOL or result['task'] != task
            or set(result['cells']) != {f'{t}_{d}' for t, d in CELLS}
            or len(result['checkpoints']) != 3 or not result['old_group_check']['passed']):
        raise ValueError('Incomplete or foreign trajectory scores')
    identities = {}
    for key, cell in result['cells'].items():
        domain = DOMAINS[int(key[-1])-1]
        if cell['domain'] != domain or cell['query_count'] != COUNTS[domain] or cell['candidate_count'] != COUNTS[domain]-1:
            raise ValueError('Candidate pool metadata changed')
        with np.load(verify(cell['matrix']), allow_pickle=False) as data:
            ids, scores = data['ids'].tolist(), data['scores']
            if (ids != sorted(ids) or len(set(ids)) != COUNTS[domain]
                    or scores.shape != (COUNTS[domain], COUNTS[domain]) or not np.isfinite(scores).all()):
                raise ValueError('Blind matrix IDs/shape/finite check failed before labels')
        if domain in identities and identities[domain] != ids:
            raise ValueError('Pool changed between model stages')
        identities[domain] = ids
    return identities


def positive_map(groups, qrels):
    if set(qrels) != {g.uid for g in groups}:
        raise ValueError('Only the original dev qrels are accepted')
    positive = {}
    for g in groups:
        row, ids = qrels[g.uid], [v.uid for v in g.views]
        labels = np.asarray(row['labels'])
        if row['view_ids'] != ids or labels.shape != (len(ids)*(len(ids)-1)//2,) or not np.isin(labels, [0, 1]).all():
            raise ValueError('Original qrels alignment differs')
        a, b = np.triu_indices(len(ids), 1)
        for i, j in zip(a[labels == 1], b[labels == 1]):
            if ids[i] in positive or ids[j] in positive:
                raise ValueError('Each observation view needs exactly one distinct partner')
            positive[ids[i]], positive[ids[j]] = ids[j], ids[i]
        if set(ids)-set(positive):
            raise ValueError('Missing positive view')
    return positive


def query_metrics(scores, ids, positive):
    n = len(ids)
    if (ids != sorted(ids) or len(set(ids)) != n or set(ids) != set(positive)
            or scores.shape != (n, n) or not np.isfinite(scores).all()
            or any(positive.get(positive[x]) != x or positive[x] == x for x in ids)):
        raise ValueError('Complete fixed pool and symmetric nonself positives required')
    values = np.array(scores, dtype=float, copy=True)
    np.fill_diagonal(values, -np.inf)
    index = {x: i for i, x in enumerate(ids)}
    order = np.argsort(-values, axis=1, kind='stable')
    partner = np.array([index[positive[x]] for x in ids])
    ranks = np.argmax(order == partner[:, None], axis=1)+1
    return np.column_stack([1/ranks, 1/ranks, *[(ranks <= k).astype(float) for k in (1, 3, 5, 10)],
                            *[np.where(ranks <= k, 1/np.log2(ranks+1), 0.) for k in (1, 3, 5, 10)]])


def analyze(target, inputs, receipts, check=lambda: None):
    if len(receipts) != 6 or {r['task_id'] for r in receipts} != {t['id'] for t in tasks()}:
        raise ValueError('All six trajectories required before any dev qrels read')
    by_id = {r['task_id']: pinned(r['completion']) for r in receipts}
    reference_ids = None
    for task in tasks():
        check()
        result = by_id[task['id']]
        if result['dev_sources'] != inputs['dev_sources']:
            raise ValueError('Scoring input changed')
        ids = validate_scores(result, task)
        if reference_ids is not None and ids != reference_ids:
            raise ValueError('Candidate pools differ between trajectories')
        reference_ids = ids
    pools, groups = load_public(inputs, target)
    if {d: [v.uid for v in pools[d]] for d in DOMAINS} != reference_ids:
        raise ValueError('Input views differ from complete blind scores')
    access(target, 'dev_qrels_joint_opening', [inputs['dev_qrels'][d] for d in DOMAINS])
    positives = {d: positive_map(groups[d], pinned(inputs['dev_qrels'][d])) for d in DOMAINS}
    summary = dict(status='DESCRIPTIVE_DEV_DIAGNOSTIC_COMPLETE', protocol=PROTOCOL,
                   metrics=METRICS, training_updates=0, scored_trajectories=6, checkpoints=18, cells=48,
                   counts=COUNTS, runs=[], methods={}, automatic_next_phase=False,
                   inference='reused_original_CLS_H_and_original_VendorLink',
                   scope='fixed_seen_development_data_observed_account_proxy_no_confirmatory_claim')
    for task in tasks():
        cells = {}
        for key, cell in by_id[task['id']]['cells'].items():
            check()
            with np.load(verify(cell['matrix']), allow_pickle=False) as data:
                per_query = query_metrics(data['scores'], data['ids'].tolist(), positives[cell['domain']])
            path = target/f'{task["id"]}_{key}_queries.npy'
            np.save(path, per_query)
            cells[key] = dict(domain=cell['domain'], query_count=len(per_query),
                              candidate_count=cell['candidate_count'], mean=per_query.mean(0).tolist(),
                              query_metrics=record(path))
        r = {key: np.array(value['mean']) for key, value in cells.items()}
        q, o = (r['1_1']+r['2_2'])/2, (r['3_1']+r['3_2'])/2
        roles = dict(Q=q, O=o, F=q-o, N=r['3_3'], final=(2*o+r['3_3'])/3,
                     F_A=r['1_1']-r['3_1'], F_B=r['2_2']-r['3_2'],
                     G_B=r['2_2']-r['1_2'], G_C=r['3_3']-r['2_3'])
        summary['runs'].append(dict(task=task, cells=cells, roles={k: v.tolist() for k, v in roles.items()}))
    for method in METHODS:
        rows = [x for x in summary['runs'] if x['task']['method'] == method]
        summary['methods'][method] = dict(
            cells={k: np.mean([x['cells'][k]['mean'] for x in rows], axis=0).tolist() for k in rows[0]['cells']},
            roles={k: np.mean([x['roles'][k] for x in rows], axis=0).tolist() for k in rows[0]['roles']})
    write(target/'result.json', summary)
    lines = ['# VendorLink 固定全市场候选池：完整开发指标', '',
             '所有列均为0—1原始值；F正值为下降，G正值为新域学习增量。无新训练，无独立确认或统计验收。', '']
    for row in summary['runs']:
        lines += ['## '+row['task']['id'], '', '|阶段/市场|'+ '|'.join(METRICS)+'|',
                  '|---|'+'---:|'*len(METRICS)]
        for key, cell in row['cells'].items():
            lines.append('|'+key+' '+cell['domain']+'|'+'|'.join(f'{x:.9f}' for x in cell['mean'])+'|')
        for key, values in row['roles'].items():
            lines.append('|'+key+'|'+'|'.join(f'{x:.9f}' for x in values)+'|')
        lines.append('')
    (target/'tables.zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def campaign(gate, inputs, gate_path):
    root = Path(gate['experiment_root'])/'campaign'
    root.mkdir()  # Refuse reruns, including campaigns that failed before scoring.
    prior = pinned(gate['preparation_budget'])
    ledger = Ledger(root, prior['used_seconds'], prior['cpu_test_seconds'], caps=CAPS)
    runner = Runner(gate, gate_path, root, ledger, worker_entry=Path(__file__), caps=CAPS,
                    limits=LIMITS, wait_for_resources=False)
    receipts = []
    try:
        with ledger.active('metadata_and_fixed_batch'):
            for binding in inputs['old_runs']:
                old_completion(binding)
            for row in inputs['model_files']:
                verify(row)
            write(root/'batch.json', dict(protocol=PROTOCOL, tasks=tasks(), cells=CELLS,
                                         gate=record(gate_path), inputs=gate['inputs']))
            jobs = []
            for task in tasks():
                target = root/'scoring'/task['id']; target.mkdir(parents=True)
                jobs.append(dict(action='score', target=target, bucket='gpu', seconds=900,
                                 extra=['--task', task['id']]))
        def complete(job):
            binding = record(job['target']/'blind_complete.json')
            task = next(t for t in tasks() if t['id'] == job['target'].name)
            validate_scores(pinned(binding), task)
            receipts.append(dict(task_id=task['id'], completion=binding))
            write(root/'scoring_complete.json', receipts)
        runner.run_many(jobs, 2, complete)
        target = root/'analysis'; target.mkdir()
        runner.run('analyze', target, 'auxiliary', 300, [])
        result = read(target/'result.json')
        write(root/'completion.json', dict(status=result['status'], result=record(target/'result.json'),
                                           new_training=0, automatic_next_phase=False))
    except BaseException as exc:
        write(root/'failure.json', dict(error=repr(exc), automatic_retry=False, automatic_next_phase=False))
        raise
    finally:
        ledger.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--action', choices=('campaign', 'score', 'analyze'), required=True)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--task')
    args = parser.parse_args()
    os.umask(0o077)
    gate, inputs = permit(args.gate)
    if args.action == 'campaign':
        campaign(gate, inputs, args.gate)
        return
    root = Path(gate['experiment_root'])
    if args.target is None or args.deadline is None:
        raise ValueError('Controller-bound worker required')
    check = Check(root, args.deadline)
    check()
    try:
        if args.action == 'score':
            task = next(t for t in tasks() if t['id'] == args.task)
            if args.target.resolve() != root/'campaign/scoring'/task['id']:
                raise ValueError('Exact scoring destination required')
            configure_gpu()  # Existing deterministic FP32 and 13GiB allocator configuration.
            score_one(args.target, task, inputs, check)
        else:
            if args.target.resolve() != root/'campaign/analysis':
                raise ValueError('Exact analysis destination required')
            analyze(args.target, inputs, read(root/'campaign/scoring_complete.json'), check)
        check()
    except BaseException as exc:
        write(args.target/'failure.json', dict(error=repr(exc), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
