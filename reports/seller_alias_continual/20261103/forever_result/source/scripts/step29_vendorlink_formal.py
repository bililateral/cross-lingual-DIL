"""Fixed 36-run VendorLink batch; no development or held-out labels during training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ENTRY_STARTED = time.monotonic()

import numpy as np
import psutil
import torch

import step28_formal_queue as queue
import step29_vendorlink_run as run
from step29_vendorlink_data import Group, seed_for
from step29_vendorlink_methods import METHODS, Model, Settings

DOMAINS = dict(A='AlphaBay', B='Dream', C='Agora')
GROUPS = dict(AlphaBay=97, Dream=34, Agora=126)


def specifications():
    return [dict(id=f'{order}_s{seed}_{method}', order=order, seed=seed, method=method)
            for order in ('ABC', 'BCA', 'CAB') for seed in (4, 5, 6) for method in METHODS]


def validate_scope(allowed):
    if (allowed['runs'] != specifications() or allowed['epochs'] != 40
            or allowed['per_run_seconds'] != 8 * 3600
            or allowed['budget_seconds']['training'] != 288 * 3600
            or allowed.get('early_evaluation_opening') is not False
            or allowed.get('conclusion_scope') != 'DESCRIPTIVE_FIXED_DATA_ONLY'
            or allowed.get('automatic_retry') is not False
            or set(allowed['fit_sources']) != set(DOMAINS.values())):
        raise ValueError('Fixed four-arm/36-run/40-epoch proposal differs')


class Budget:
    def __init__(self, allowed, deadline):
        self.allowed, self.deadline = allowed, deadline
        self.started, self.last = time.monotonic(), 0.
        self.peak_rss = self.peak_disk = 0

    def check(self):
        now = time.monotonic()
        if now >= self.deadline - 60:
            raise TimeoutError('Trajectory shutdown reserve reached')
        if now - self.last < 2:
            return
        self.last = now
        root, limits = Path(self.allowed['campaign']), self.allowed['limits']
        files = [(p.suffix, p.stat().st_size) for p in root.rglob('*') if p.is_file() and not p.is_symlink()]
        disk = sum(n for _, n in files)
        rss = psutil.Process().memory_info().rss
        self.peak_rss, self.peak_disk = max(self.peak_rss, rss), max(self.peak_disk, disk)
        # New+old latest can coexist during its atomic replacement.
        if (rss > limits['rss_bytes'] or disk + 3 * 2**30 > limits['disk_bytes']
                or sum(n for suffix, n in files if suffix != '.pt') > limits['small_bytes']
                or shutil.disk_usage(root).free < limits['free_reserve_bytes'] + 3 * 2**30):
            raise RuntimeError('VendorLink actual resource cap exceeded')


def validate_completion(spec, root, allowed):
    value = json.loads((root/'completion.json').read_text(encoding='utf-8'))
    expected = dict(spec=spec, epochs=allowed['epochs'], sources=allowed['sources'],
        task='observed_account_views_not_verified_aliases')
    if (value['status'] != 'THREE_STAGES_TRAINED_NOT_EVALUATED' or value['identity'] != expected
            or value['updates'] != 257 * allowed['epochs'] or not value['inference_endpoints']
            or value['stage_endpoints'] != ['stage1.pt', 'stage2.pt', 'stage3.pt']
            or value.get('access', {}).get('fit_domains') != [DOMAINS[d] for d in spec['order']]
            or any(value['access'][k] for k in ('dev', 'heldout', 'qrels', 'owners'))):
        raise ValueError('Incomplete/foreign VendorLink trajectory')
    if set(value.get('endpoint_files', {})) != {'1', '2', '3'}:
        raise ValueError('All three actual endpoint files required')
    for rec in value['endpoint_files'].values():
        queue.verify(rec)
    return value


def execute(job, spec, deadline, allowed):
    from transformers import BertModel, BertTokenizer
    budget = Budget(allowed, deadline)
    budget.check()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed_for(spec['seed'], 'initialization'))
    np.random.seed(seed_for(spec['seed'], 'initialization'))
    torch.manual_seed(seed_for(spec['seed'], 'initialization'))
    for row in allowed['model_files']:
        queue.verify(row)
    settings = Settings(method=spec['method'])
    bert = BertModel.from_pretrained(allowed['model_path'], local_files_only=True)
    tokenizer = BertTokenizer.from_pretrained(allowed['model_path'], local_files_only=True, do_lower_case=True)
    model = Model(bert, tokenizer, settings).to('cuda:0')
    root = job/'run'
    access = dict(fit_domains=[], dev=0, heldout=0, qrels=0, owners=0)
    def load_fit(domain):
        if access['fit_domains'] != [DOMAINS[d] for d in spec['order'][:len(access['fit_domains'])]]:
            raise ValueError('Current-domain stream differs')
        expected = DOMAINS[spec['order'][len(access['fit_domains'])]]
        if domain != expected:
            raise ValueError('Future/history fit read refused')
        access['fit_domains'].append(domain)
        queue.write(job/'access.json', access)
        path = queue.verify(allowed['fit_sources'][domain])
        groups = [Group.from_dict(g) for g in json.loads(path.read_text(encoding='utf-8'))]
        if len(groups) != GROUPS[domain] or any(g.domain != domain or g.split != 'fit' for g in groups):
            raise ValueError('Fixed exported fit groups differ')
        return groups
    identity = dict(spec=spec, epochs=allowed['epochs'], sources=allowed['sources'],
                    task='observed_account_views_not_verified_aliases')
    run.execute(model, load_fit, root, order=[DOMAINS[d] for d in spec['order']], seed=spec['seed'],
        epochs=allowed['epochs'], identity=identity, check=budget.check, inference_endpoints=True)
    value = json.loads((root/'completion.json').read_text(encoding='utf-8'))
    endpoints = {}
    for number in (1, 2, 3):
        budget.check()
        path = root/f'stage{number}.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        if (state['identity'] != identity or state['settings'] != asdict(settings)
                or state['stage'] != number
                or state['updates'] != sum(GROUPS[DOMAINS[d]] for d in spec['order'][:number]) * allowed['epochs']):
            raise ValueError('Saved inference endpoint identity/count differs')
        if settings.method == 'VendorLink_SEQ':
            model.start_vendor_stage(state['classes'], 0)
        model.load_state_dict(state['model'], strict=True)
        endpoints[str(number)] = queue.record(path)
        del state
    value.update(endpoint_files=endpoints, access=access,
        resources=dict(wall_seconds=time.monotonic()-budget.started,
                       peak_rss_bytes=budget.peak_rss, peak_disk_bytes=budget.peak_disk))
    queue.write(root/'completion.json', value)
    validate_completion(spec, root, allowed)
    # Stage weights support all later blind cells. Only this complete run's
    # no-longer-needed full recovery payload is retired, with identity retained.
    latest = root/'latest.pt'
    retired = queue.record(latest)
    queue.write(root/'recovery_retirement.json', dict(file=retired, status='PLANNED'))
    latest.unlink()
    queue.write(root/'recovery_retirement.json', dict(file=retired, status='DELETED'))
    del model, bert
    gc.collect()
    torch.cuda.empty_cache()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('train', 'worker'))
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--id')
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    allowed = queue.gate(args.gate, 'VendorLink', 'train')
    validate_scope(allowed)
    if args.action == 'worker':
        spec = next(s for s in specifications() if s['id'] == args.id)
        if args.job.resolve() != Path(allowed['campaign']).resolve()/spec['id']:
            raise ValueError('Worker job is outside the fixed queue')
        try:
            execute(args.job, spec, args.deadline, allowed)
        except BaseException as error:
            queue.write(args.job/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
            raise
    else:
        rows = [{**s, 'seconds': allowed['per_run_seconds']} for s in specifications()]
        followups = [dict(id=directory, action=action, bucket=bucket, gpu=gpu,
            seconds=allowed['budget_seconds'][bucket] - allowed['prior_used_seconds'].get(bucket, 0))
            for directory, action, bucket, gpu in (('scoring', 'score', 'confirmation', True),
                ('analysis', 'analyze', 'analysis', False))]
        def command(row, job, deadline):
            if 'action' in row:
                return [sys.executable, '-B', str(Path(__file__).with_name('step29_vendorlink_evaluate.py')),
                        row['action'], '--gate', str(args.gate.resolve()), '--deadline', str(deadline)]
            return [sys.executable, '-B', __file__, 'worker', '--gate', str(args.gate.resolve()),
                    '--job', str(job), '--id', row['id'], '--deadline', str(deadline)]
        def validate(row, job):
            if 'action' not in row:
                return validate_completion({k: row[k] for k in ('id', 'order', 'seed', 'method')}, job/'run', allowed)
            name = dict(score='all_blind_complete.json', analyze='result.json')[row['action']]
            value = json.loads((job/name).read_text())
            if row['action'] == 'score' and value['forwards'] != 4104:
                raise ValueError('Full blind scoring did not complete')
            if row['action'] == 'analyze' and value['status'] != 'VENDORLINK_FIXED_ANALYSIS_COMPLETE':
                raise ValueError('Analysis did not complete')
        queue.dispatch(allowed, rows, command, validate, followups,
                       preflight=lambda: queue.verify(allowed['dependency_metadata']), started=ENTRY_STARTED)


if __name__ == '__main__':
    main()
