"""Frozen RMS encoder, original H head, and fit-only full-pool head supervision."""
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from step29_vendorlink_data import Ad, View, Group, seed_for
from step29_vendorlink_methods import h_objective
from step29_vendorlink_run import set_rates
from step35_vendorlink_account_policy import AccountSettings
from step35_vendorlink_account_io import pinned, record, write
from step35_vendorlink_account_run import groups_for, restore, atomic_tensor
from step35_vendorlink_account_evaluate import encode, rectangular, load_role


def read(binding):
    return json.loads(pinned(binding).read_text())


def base_model(base, model_binding):
    completion = read(base)
    model, _ = restore(model_binding, completion['task'], pinned(completion['endpoints']['1']),
                       completion['identity'], 1)
    model.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def cache_fit(base, fit, model_binding, domain, target, identity, check=lambda: None):
    if target.exists():
        raise ValueError('Fresh fit cache required')
    accounts = read(fit)
    groups_for(accounts, 41, domain, 40)  # Validate twelve-record fit accounts.
    ids = [a['account_uid'] for a in accounts]
    records = [r['uid'] for a in accounts for r in a['records']]
    if len(records) != len(set(records)):
        raise ValueError('Fit record identities must be globally unique')
    model = base_model(base, model_binding)
    views = [View(a['account_uid'], tuple(Ad(**r) for r in a['records']), a['account_uid']) for a in accounts]
    vectors = encode(model, views, check).detach().cpu()
    if vectors.shape[:2] != (len(accounts), 12) or not torch.isfinite(vectors).all():
        raise ValueError('Complete finite fit-only features required')
    target.mkdir(parents=True)
    path = target/'features.pt'
    atomic_tensor(path, dict(identity=identity, base=base, fit=fit, classes=ids,
        records=records, vectors=vectors, head={k: v.cpu() for k, v in model.head.state_dict().items()},
        encoder_frozen=True, encoder_mode='eval'), check)
    result = dict(status='FIT_CACHE_COMPLETE', identity=identity, base=base, fit=fit,
                  file=record(path), accounts=len(accounts), records=len(records))
    write(target/'completion.json', result)
    return result


def make_head(state, dimension, device):
    head = nn.Sequential(nn.Linear(2*dimension, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
    head.load_state_dict(state, strict=True)
    return head


def original_table(head, z):
    i, j = torch.triu_indices(len(z), len(z), 1, device=z.device)
    return head(torch.cat(((z[i]-z[j]).abs(), z[i]*z[j]), 1)).flatten()


def full_pool_logits(head, query, galleries, check=lambda: None):
    """One Q2 against every G4; exact original mean/log-mean-exp H score."""
    if query.ndim != 2 or len(query) != 2 or galleries.ndim != 3 or galleries.shape[1] != 4:
        raise ValueError('Q2/G4 required')
    parts = []
    for start in range(0, len(galleries), 64):
        check()
        right = galleries[start:start+64]
        left = query[None, :, None, :].expand(len(right), 2, 4, -1)
        right = right[:, None, :, :].expand(-1, 2, -1, -1)
        pair = torch.cat(((left-right).abs(), left*right), -1).reshape(-1, 2*query.shape[-1])
        values = head(pair).reshape(len(left), 8).double()
        parts.append((.5*values.mean(1)+.5*(values.logsumexp(1)-math.log(8))).float())
    return torch.cat(parts)


def gallery_indices(accounts, group, lookup, step_seed):
    excluded = {}
    for view in group.views:
        excluded.setdefault(view.identity, set()).update(r.uid for r in view.records)
    indices = []
    for account in accounts:
        choices = sorted(r['uid'] for r in account['records'] if r['uid'] not in excluded.get(account['account_uid'], set()))
        if len(choices) < 4:
            raise ValueError('Insufficient disjoint fit gallery')
        chosen = random.Random(seed_for(step_seed, account['account_uid'], 'gallery')).sample(choices, 4)
        indices.append([lookup[r] for r in chosen])
    return indices


def head_update(head, optimizer, features, accounts, group, lookup, settings, step_seed, full_pool, check=lambda: None):
    if features.requires_grad:
        raise ValueError('Frozen encoder features required')
    optimizer.zero_grad(set_to_none=True)
    z = features[[lookup[r.uid] for v in group.views for r in v.records]]
    loss, logs = h_objective(original_table(head, z), group, seed_for(step_seed, 'regroup'), settings)
    loss.backward()
    global_loss, correct = 0., 0
    if full_pool:
        index = gallery_indices(accounts, group, lookup, step_seed)
        galleries = features[index]
        classes = {a['account_uid']: k for k, a in enumerate(accounts)}
        for view in group.views:
            check()
            query = features[[lookup[r.uid] for r in view.records]]
            logits = full_pool_logits(head, query, galleries, check)
            label = classes[view.identity]
            ce = F.cross_entropy(logits[None], torch.tensor([label], device=logits.device))/len(group.views)
            ce.backward()  # Sum exact per-query full-pool CE gradients before ONE update.
            global_loss += float(ce.detach())
            correct += int(logits.argmax().item() == label)
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(original_supervised=logs['supervised'], global_ce=global_loss,
        global_queries=len(group.views) if full_pool else 0, global_correct=correct,
        candidates=len(accounts) if full_pool else 0, gradient_norm=float(norm),
        encoder_updates=0, auxiliary_updates=0, optimizer_updates=1)


def train_head(task, cache_binding, fit, identity, target, check=lambda: None, device='cuda:0'):
    if target.exists():
        raise ValueError('Fresh head training required')
    cache_completion = read(cache_binding)
    cache = torch.load(pinned(cache_completion['file']), map_location='cpu', weights_only=False)
    if cache['identity'] != cache_completion['identity'] or cache['fit'] != fit or cache['base'] != cache_completion['base']:
        raise ValueError('Fit cache provenance differs')
    accounts = read(fit)
    if cache['classes'] != [a['account_uid'] for a in accounts] or cache['records'] != [r['uid'] for a in accounts for r in a['records']]:
        raise ValueError('Cached feature/record alignment differs')
    features = cache['vectors'].flatten(0, 1).to(device).detach()
    lookup = {uid: i for i, uid in enumerate(cache['records'])}
    head = make_head(cache['head'], features.shape[1], device)
    settings = AccountSettings(**task['settings'])
    # Empty encoder group preserves the qualified schedule without an encoder parameter.
    optimizer = torch.optim.AdamW([{'params': [], 'lr': settings.encoder_lr},
        {'params': list(head.parameters()), 'lr': settings.head_lr}], weight_decay=settings.weight_decay, eps=1e-8)
    total = len(groups_for(accounts, task['seed'], task['domains'][0], 40))*task['epochs']
    updates, records = 0, 0
    target.mkdir(parents=True)
    for epoch in range(task['epochs']):
        for group in groups_for(accounts, task['seed'], task['domains'][0], 40+epoch):
            check()
            updates += 1
            set_rates(optimizer, settings, updates, total)
            logs = head_update(head, optimizer, features, accounts, group, lookup, settings,
                seed_for(task['seed'], task['domains'][0], 40+epoch, updates, 'head_alignment'),
                task['candidate'] == 'FULL_POOL', check)
            records += sum(len(v.records) for v in group.views)
            with (target/'updates.jsonl').open('a') as output:
                output.write(json.dumps(dict(stage=1, epoch=epoch, total_updates=updates, updates=updates,
                    stage_update=updates, rates=[g['lr'] for g in optimizer.param_groups], **logs), allow_nan=False)+'\n')
    if updates != total or records != 12*len(accounts)*task['epochs']:
        raise ValueError('Full fixed exposure required')
    endpoint = target/'head.pt'
    atomic_tensor(endpoint, dict(identity=identity, task=task, base=cache['base'], cache=cache_binding,
        head={k: v.detach().cpu() for k, v in head.state_dict().items()}, updates=updates,
        encoder_updates=0, auxiliary_updates=0), check)
    result = dict(status='HEAD_TRAINING_COMPLETE', identity=identity, task=task,
        endpoint=record(endpoint), base=cache['base'], cache=cache_binding, updates=updates,
        workload=dict(current_ads=records, current_updates=updates, encoder_updates=0, auxiliary_updates=0))
    write(target/'workload.json', result['workload'])
    write(target/'completion.json', result)
    return result


@torch.no_grad()
def score_head(task, completion, manifest, model_binding, target, check=lambda: None):
    if target.exists():
        raise ValueError('Fresh score output required')
    payload = torch.load(pinned(completion['endpoint']), map_location='cpu', weights_only=False)
    if payload['identity'] != completion['identity'] or payload['task'] != task or payload['base'] != completion['base']:
        raise ValueError('Head/encoder provenance differs')
    domain = task['domains'][0]
    n = manifest['counts'][domain]['eligible_accounts']
    sources = {role: manifest['files'][domain+':'+role] for role in ('gallery', 'dev')}
    target.mkdir(parents=True)
    write(target/'access.json', dict(role='dev', sources=sources))
    qids, query = load_role(sources['dev'], domain, 'dev', n)
    gids, gallery = load_role(sources['gallery'], domain, 'gallery', n)
    if qids != gids or {r.uid for v in query for r in v.records} & {r.uid for v in gallery for r in v.records}:
        raise ValueError('Query/gallery class or text isolation differs')
    model = base_model(completion['base'], model_binding)
    model.head.load_state_dict(payload['head'], strict=True)
    values = rectangular(model, encode(model, query, check), encode(model, gallery, check), check)
    path = target/'H_SEQ_R11.npy'
    np.save(path, values)
    result = dict(task=task, role='dev', status='BLIND_SCORES_COMPLETE', cells=['R11'],
        files={'H_SEQ:R11': record(path)}, sources=sources, training_identity=completion['identity'])
    write(target/'completion.json', result)
    return result
