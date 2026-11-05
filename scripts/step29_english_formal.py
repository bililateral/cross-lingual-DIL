"""ABC development workers. No data/model access without a closed formal gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import psutil
import torch

import step28_formal_queue as queue
import step29_english_abc as policy
import step29_vendorlink_run as run
from step29_vendorlink_data import Group, seed_for
from step29_vendorlink_methods import Model, Settings
from step29_vendorlink_evaluate import CELLS, METRICS, query_metrics

DOMAINS = ('AlphaBay', 'Dream', 'Agora')
COUNTS = dict(zip(DOMAINS, (97, 34, 126)))
DEV_COUNTS = dict(zip(DOMAINS, (24, 8, 31)))
CAPS = dict(training=172800, gpu=10800, auxiliary=14400)
# Larger applicable single-worker peak: old VendorLink 14.150390625 GiB.
# Uniform reservation includes max(1 GiB, 10%); no separate dual GPU test.
SLOT_BYTES = math.ceil(14.150390625 * 2**30 * 1.1)
LIMITS = dict(rss_bytes=32*2**30, disk_bytes=96*2**30,
              small_bytes=4*2**30, free_reserve_bytes=64*2**30,
              free_gpu_bytes=SLOT_BYTES, process_gpu_bytes=SLOT_BYTES,
              allocator_bytes=SLOT_BYTES-2**30)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def spec(settings, seed):
    values = asdict(settings)
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return dict(id=f'{settings.method}_s{seed}_{digest[:16]}', seed=seed,
                order='ABC', settings=values)


def gate(path):
    allowed = read(path)
    if (sys.platform != 'linux' or allowed.get('approved') is not True
            or allowed.get('review_closed') is not True
            or allowed.get('experiment') != 'English_ABC_development'
            or allowed.get('workers') != 2 or allowed.get('epochs') != 40
            or allowed.get('budget_seconds') != CAPS or allowed.get('limits') != LIMITS
            or allowed.get('automatic_retry') is not False
            or allowed.get('dual_gpu_test_required') is not False
            or allowed.get('fit_counts', COUNTS) != COUNTS
            or allowed.get('dev_counts', DEV_COUNTS) != DEV_COUNTS
            or allowed.get('conclusion_scope') != 'FIXED_DATA_DEVELOPMENT_ONLY'):
        raise ValueError('Reviewed, explicitly approved ABC development gate required')
    for kind in ('fit_sources', 'dev_sources', 'dev_qrels'):
        if set(allowed[kind]) != set(DOMAINS):
            raise ValueError('Exactly three pinned current-fit/dev domains required')
    manifest = read(queue.verify(allowed['export_manifest']))
    exported = {r['path']: r for r in manifest['files']}
    for domain in DOMAINS:
        for kind, suffix in (('fit_sources','fit.json'), ('dev_sources','dev.json'), ('dev_qrels','dev_qrels.json')):
            row = allowed[kind][domain]
            if Path(row['path']).name != f'{domain}_{suffix}' or exported.get(row['path']) != row:
                raise ValueError('Only the original fit and dev exports may be bound')
        if (manifest['counts'][domain+':dev']['groups'] != DEV_COUNTS[domain]
                or allowed['dev_queries'][domain] != manifest['counts'][domain+':dev']['views']):
            raise ValueError('Dev query weighting differs from frozen export metadata')
    # Binding checks here open source/authorization metadata, not fit/dev labels.
    queue.verify(allowed['authorization'])
    workspace = Path(allowed['workspace']).resolve()
    sources = set()
    for row in allowed['sources']:
        source = queue.verify(row).resolve()
        if not source.is_relative_to(workspace):
            raise ValueError('Source outside isolated workspace')
        sources.add(source)
    required = {Path(__file__).resolve(), Path(sys.argv[0]).resolve(), Path(queue.__file__).resolve(),
                Path(policy.__file__).resolve(), Path(run.__file__).resolve()}
    if not required <= sources:
        raise ValueError('Actual entry/dependencies are not pinned')
    if len(allowed['old_runs']) != 12 or {r['id'] for r in allowed['old_runs']} != {
            f'ABC_s{s}_{m}' for s in (4, 5, 6) for m in ('H_full', 'H_ER', 'H_SEQ', 'VendorLink_SEQ')}:
        raise ValueError('Old ABC four-method three-seed precheck must be complete')
    return allowed


class Budget:
    """The actual model/save callback uses the shared, retirement-safe scan."""
    def __init__(self, allowed, deadline):
        self.allowed, self.deadline, self.last = allowed, deadline, 0.

    def check(self):
        now = time.monotonic()
        if now >= self.deadline-60:
            raise TimeoutError('Worker shutdown reserve reached')
        if now-self.last < 2:
            return
        self.last = now
        root, limits = Path(self.allowed['campaign']), self.allowed['limits']
        files = queue.campaign_files(root)
        # Two atomic latest checkpoints can coexist with their previous versions.
        reserve = self.allowed['workers'] * 3 * 2**30
        if (psutil.Process().memory_info().rss > limits['rss_bytes']
                or sum(size for _, size in files)+reserve > limits['disk_bytes']
                or sum(size for suffix, size in files if suffix != '.pt') > limits['small_bytes']
                or shutil.disk_usage(root).free < limits['free_reserve_bytes']+reserve):
            raise RuntimeError('Shared campaign/checkpoint/RAM resource limit')


def configure(allowed):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(allowed['limits']['allocator_bytes']/total, 0)
    for row in allowed['model_files']:
        queue.verify(row)


def make_model(allowed, settings):
    from transformers import BertModel, BertTokenizer
    return Model(BertModel.from_pretrained(allowed['model_path'], local_files_only=True),
                 BertTokenizer.from_pretrained(allowed['model_path'], local_files_only=True,
                                               do_lower_case=True), settings).to('cuda:0')


def identity(row, allowed):
    return dict(spec=row, epochs=allowed['epochs'], sources=allowed['sources'],
                task='observed_account_views_not_verified_aliases')


def validate_completed(row, root, allowed):
    value = read(root/'completion.json')
    if (value['status'] != 'THREE_STAGES_TRAINED_NOT_EVALUATED'
            or value['identity'] != identity(row, allowed)
            or value['updates'] != sum(allowed.get('fit_counts', COUNTS).values())*allowed['epochs']
            or value['stage_endpoints'] != ['stage1.pt', 'stage2.pt', 'stage3.pt']
            or value['inference_endpoints'] is not True
            or value['access'] != dict(fit_domains=list(DOMAINS), dev=0, heldout=0, qrels=0, owners=0)
            or set(value['endpoint_files']) != {'1', '2', '3'}):
        raise ValueError('Incomplete/mismatched ABC training endpoints or access')
    for stage, record in value['endpoint_files'].items():
        if Path(record['path']).resolve() != (root/f'stage{stage}.pt').resolve():
            raise ValueError('Endpoint escaped its trajectory')
        queue.verify(record)
    return value


def train(job, row, allowed, budget, factory=make_model):
    seed = seed_for(row['seed'], 'initialization')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    settings = Settings(**row['settings'])
    model = factory(allowed, settings)
    access = dict(fit_domains=[], dev=0, heldout=0, qrels=0, owners=0)
    def load_fit(domain):
        if domain != DOMAINS[len(access['fit_domains'])]:
            raise ValueError('Only the next current domain may be opened')
        access['fit_domains'].append(domain)
        queue.write(job/'access.json', access)  # Failed reads are still attempts.
        groups = [Group.from_dict(g) for g in read(queue.verify(allowed['fit_sources'][domain]))]
        if (len(groups) != allowed.get('fit_counts', COUNTS)[domain]
                or any(g.domain != domain or g.split != 'fit' for g in groups)):
            raise ValueError('Current fit export differs')
        return groups
    root = job/'run'
    run.execute(model, load_fit, root, order=DOMAINS, seed=row['seed'], epochs=allowed['epochs'],
                identity=identity(row, allowed), check=budget.check, inference_endpoints=True)
    value = read(root/'completion.json')
    points = {}
    for stage in (1, 2, 3):
        budget.check()
        point = root/f'stage{stage}.pt'
        state = torch.load(point, map_location='cpu', weights_only=False)
        if (state['identity'] != identity(row, allowed) or state['settings'] != row['settings']
                or state['stage'] != stage
                or state['updates'] != sum(allowed.get('fit_counts', COUNTS)[d] for d in DOMAINS[:stage])*allowed['epochs']):
            raise ValueError('Saved endpoint configuration/stage/exposure differs')
        if settings.method == 'VendorLink_SEQ':
            model.start_vendor_stage(state['classes'], 0)
        model.load_state_dict(state['model'], strict=True)
        points[str(stage)] = queue.record(point)
        del state
    value.update(endpoint_files=points, access=access)
    queue.write(root/'completion.json', value)
    validate_completed(row, root, allowed)
    # Only the successful trajectory's redundant full recovery state retires.
    latest = queue.record(root/'latest.pt')
    queue.write(root/'recovery_retirement.json', dict(file=latest, status='PLANNED'))
    (root/'latest.pt').unlink()
    queue.write(root/'recovery_retirement.json', dict(file=latest, status='DELETED'))
    del model
    gc.collect()


def old_completed(binding):
    value = read(queue.verify(binding['completion']))
    row = value['identity']['spec']
    if (row != binding['spec'] or row['id'] != binding['id'] or row['order'] != 'ABC'
            or row['seed'] not in (4, 5, 6) or value['identity']['epochs'] != 40
            or value['identity']['sources'] != binding['sources']
            or value['updates'] != 10280 or not value['inference_endpoints']
            or value['stage_endpoints'] != ['stage1.pt', 'stage2.pt', 'stage3.pt']
            or set(value['endpoint_files']) != {'1', '2', '3'}
            or value['access'] != dict(fit_domains=list(DOMAINS), dev=0, heldout=0, qrels=0, owners=0)):
        raise ValueError('Old ABC completion is incomplete or foreign')
    for record in value['endpoint_files'].values():
        queue.verify(record)
    return value


def evaluation_specs(batch, allowed):
    if batch['name'] == 'precheck':
        return [(dict(id=r['id'], seed=r['spec']['seed'], order='ABC',
                      settings=asdict(Settings(r['spec']['method']))), old_completed(r))
                for r in allowed['old_runs']]
    return [(row, validate_completed(row, Path(batch['training_root'])/row['id']/'run', allowed))
            for row in batch['runs']]


def evaluate(job, batch, allowed, budget, factory=make_model):
    pairs = evaluation_specs(batch, allowed)  # All completions before any dev access.
    access = dict(dev_text_attempted=True, dev_qrels_attempted=False, heldout=0, owners=0)
    queue.write(job/'access.json', access)
    public = {d: [Group.from_dict(g) for g in read(queue.verify(allowed['dev_sources'][d]))] for d in DOMAINS}
    for domain, groups in public.items():
        if (len(groups) != allowed.get('dev_counts', DEV_COUNTS)[domain]
                or len({g.uid for g in groups}) != len(groups)
                or any(g.domain != domain or g.split != 'dev' or any(v.identity is not None for v in g.views) for g in groups)):
            raise ValueError('Blinded fixed dev export only')
    blind = {}
    for row, completed in pairs:
        budget.check()
        settings = Settings(**row['settings'])
        model = factory(allowed, settings)
        cells = {}
        for stage in (1, 2, 3):
            budget.check()
            state = torch.load(queue.verify(completed['endpoint_files'][str(stage)]),
                               map_location='cpu', weights_only=False)
            # Old endpoints acquire ONLY the explicit original defaults for new fields.
            restored = asdict(Settings(**state['settings']))
            if (state['identity'] != completed['identity'] or state['stage'] != stage
                    or restored != row['settings']):
                raise ValueError('Endpoint settings/stage/identity mismatch')
            if settings.method == 'VendorLink_SEQ':
                model.start_vendor_stage(state['classes'], 0)
            model.load_state_dict(state['model'], strict=True)
            del state
            for _, position in (c for c in CELLS if c[0] == stage):
                domain, key = DOMAINS[position-1], f'{stage}_{position}'
                scores = run.evaluate(model, public[domain], budget.check)
                target = job/f'{row["id"]}_{key}_scores.json'
                queue.write(target, scores)
                cells[key] = dict(domain=domain, scores=queue.record(target))
        blind[row['id']] = dict(spec=row, cells=cells)
        del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    queue.write(job/'blind_complete.json', dict(runs=blind))
    # Validate every score table before opening qrels for the round.
    for item in blind.values():
        for cell in item['cells'].values():
            values = read(queue.verify(cell['scores']))
            groups = public[cell['domain']]
            if set(values) != {g.uid for g in groups}:
                raise ValueError('Incomplete blind group set')
            for g in groups:
                v = values[g.uid]
                if (v['view_ids'] != [x.uid for x in g.views]
                        or np.asarray(v['scores']).shape != (len(g.views)*(len(g.views)-1)//2,)
                        or not np.isfinite(v['scores']).all()):
                    raise ValueError('Blind ordering/shape/finiteness differs')
    access['dev_qrels_attempted'] = True
    queue.write(job/'access.json', access)
    truth = {d: read(queue.verify(allowed['dev_qrels'][d])) for d in DOMAINS}
    for d in DOMAINS:
        if set(truth[d]) != {g.uid for g in public[d]}:
            raise ValueError('Dev qrels group set differs')
    result = dict(status='COMPLETE_DEV_ROUND', name=batch['name'], columns=METRICS, runs={})
    for key, item in blind.items():
        cells = {}
        for cell_key, cell in item['cells'].items():
            budget.check()
            values = read(queue.verify(cell['scores']))
            queries = {}
            for uid, scores in values.items():
                qrel = truth[cell['domain']][uid]
                if scores['view_ids'] != qrel['view_ids']:
                    raise ValueError('Qrels and score view order differ')
                queries[uid] = query_metrics(scores['scores'], qrel['labels'], scores['view_ids']).tolist()
            all_queries = [v for group in queries.values() for v in group]
            target = job/f'{key}_{cell_key}_queries.json'
            queue.write(target, dict(columns=METRICS, queries=queries))
            cells[cell_key] = dict(domain=cell['domain'], metrics=np.mean(all_queries, axis=0).tolist(),
                                   query_count=len(all_queries), groups=len(queries), queries=queue.record(target))
        result['runs'][key] = dict(spec=item['spec'], cells=cells)
    validate_evaluation(result, [r for r, _ in pairs], batch['name'], allowed)
    queue.write(job/'result.json', result)  # Nothing usable for selection before complete.
    return result


def validate_evaluation(value, specs, name, allowed):
    if (value['status'] != 'COMPLETE_DEV_ROUND' or value['name'] != name
            or tuple(value['columns']) != METRICS or len({r['id'] for r in specs}) != len(specs)
            or set(value['runs']) != {r['id'] for r in specs}):
        raise ValueError('Incomplete/duplicate development round')
    result = {}
    for row in specs:
        item = value['runs'][row['id']]
        if item['spec'] != row or set(item['cells']) != {f'{a}_{b}' for a,b in CELLS}:
            raise ValueError('Seed/settings pairing or six cells differ')
        for stage, position in CELLS:
            cell = item['cells'][f'{stage}_{position}']
            domain = DOMAINS[position-1]
            if (cell['domain'] != domain or cell['groups'] != allowed.get('dev_counts', DEV_COUNTS)[domain]
                    or cell['query_count'] != allowed['dev_queries'][domain]
                    or len(cell['metrics']) != len(METRICS)
                    or not all(math.isfinite(v) and 0 <= v <= 1 for v in cell['metrics'])):
                raise ValueError('Market/query weighting/metric completeness differs')
        result[row['id']] = {key: item['cells'][cell]['metrics'][0]
                             for key, cell in dict(B2='2_2', B3='3_2', C3='3_3', A3='3_1').items()}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('train', 'evaluate'))
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--id')
    p.add_argument('--deadline', type=float, required=True)
    args = p.parse_args()
    allowed = gate(args.gate)
    batch = read(args.batch)
    if (batch['gate_sha256'] != run.file_sha256(args.gate)
            or args.batch.resolve() != Path(allowed['campaign'])/batch['name']/'batch.json'):
        raise ValueError('Round is not bound to this campaign/gate')
    expected = args.batch.parent/(args.id if args.action == 'train' else 'evaluation')
    if args.job.resolve() != expected.resolve() or any(
            f.name not in ('console.txt', 'process.json') for f in args.job.iterdir()):
        raise ValueError('Fresh dispatched worker only; no retry')
    try:
        budget = Budget(allowed, args.deadline)
        budget.check()
        configure(allowed)
        if args.action == 'train':
            row = next(r for r in batch['runs'] if r['id'] == args.id)
            train(args.job, row, allowed, budget)
        else:
            evaluate(args.job, batch, allowed, budget)
    except BaseException as exc:
        queue.write(args.job/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
