"""Blind rectangular retrieval and frozen-encoder comparisons, then one label release."""
from __future__ import annotations

from dataclasses import asdict
import gc
from pathlib import Path
import math

import numpy as np
import torch

from step29_vendorlink_data import Ad, View, Group
from step29_vendorlink_methods import vendor_embeddings
from step30_evolution_io import access, pinned, read, record, write, verify
from step30_evolution_evaluate import training_complete
from step30_evolution_run import restore
from step31_evolution_registered_policy import DOMAINS, CELLS, METRICS, tasks, cell_key, decide


def views(values):
    result = [View(v['uid'], tuple(Ad(**r) for r in v['records']), v.get('identity')) for v in values]
    if (any(v.identity is not None or len(v.records) != 2 for v in result)
            or len({v.uid for v in result}) != len(result)
            or [v.uid for v in result] != sorted(v.uid for v in result)):
        raise ValueError('Sorted distinct blind two-record views required')
    records = [r.uid for v in result for r in v.records]
    if len(set(records)) != len(records):
        raise ValueError('Records overlap blind views')
    return result


@torch.no_grad()
def encode_views(model, items, check):
    model.eval()
    if model.settings.method != 'VendorLink_SEQ':
        encoded = model.records(Group('rectangular', 'evaluation', 'dev', tuple(items)), check)
    else:
        texts = [r.title+' [SEP] '+r.description for v in items for r in v.records]
        parts = []
        for start in range(0, len(texts), model.settings.microbatch):
            check()
            output, mask = model.encode(texts[start:start+model.settings.microbatch],
                                        model.settings.vendor_tokens, True)
            parts.append(vendor_embeddings(output.hidden_states, mask))
        encoded = torch.cat(parts)
        if (encoded.norm(dim=1) == 0).any():
            raise ValueError('Zero advertisement representation')
        encoded = torch.nn.functional.normalize(encoded.float(), dim=1)
    return encoded.reshape(len(items), 2, -1)


@torch.no_grad()
def rectangular(model, q, g, check=lambda: None):
    if q.ndim != 3 or g.ndim != 3 or q.shape[1] != 2 or g.shape[1] != 2:
        raise ValueError('Two records per query and gallery view required')
    if model.settings.method == 'VendorLink_SEQ':
        u, v = q.mean(1), g.mean(1)
        denominator = u.square().sum(1)[:, None]+v.square().sum(1)[None, :]
        if (denominator <= 0).any():
            raise ValueError('Zero norm in original VendorLink multi-record denominator')
        scores = 2*(u@v.T)/denominator
    else:
        qflat, gflat = q.flatten(0, 1), g.flatten(0, 1)
        count, width = len(qflat)*len(gflat), len(gflat)
        chunks = []
        for start in range(0, count, 512):
            check()
            index = torch.arange(start, min(start+512, count), device=q.device)
            left, right = qflat[index//width], gflat[index%width]
            chunks.append(model.head(torch.cat(((left-right).abs(), left*right), 1)).flatten())
        logits = torch.cat(chunks).reshape(len(q), 2, len(g), 2).permute(0, 2, 1, 3).flatten(2).double()
        scores = (.5*logits.mean(2)+.5*(torch.logsumexp(logits, 2)-math.log(4))).to(q.dtype)
    if scores.shape != (len(q), len(g)) or not torch.isfinite(scores).all():
        raise ValueError('Nonfinite or incomplete rectangular scores')
    return scores.cpu().numpy().astype(np.float64)


def validate_training(batch):
    if batch['specs'] != tasks() or len(batch['completed']) != 6:
        raise ValueError('All six fixed trainings must finish before any scoring')
    completed = {r['task_id']: r for r in batch['completed']}
    if set(completed) != {t['id'] for t in tasks()}:
        raise ValueError('Missing/duplicate training result')
    for task in tasks():
        row = completed[task['id']]
        training_complete(task, pinned(row['completion']), row['identity'], fit_groups=12)
    return completed


def score_one(root, task, batch, manifest, factory, model_binding, check):
    completed = validate_training(batch)
    if task not in tasks() or root.exists():
        raise ValueError('Fresh fixed scoring trajectory required')
    root.mkdir(parents=True)
    sources = [manifest['files'][f'{d}:{role}'] for d in DOMAINS for role in ('query', 'gallery')]
    access(root, 'blind_query_gallery', sources)
    query = {i: views(pinned(manifest['files'][f'{d}:query'])) for i, d in enumerate(DOMAINS, 1)}
    gallery = {i: views(pinned(manifest['files'][f'{d}:gallery'])) for i, d in enumerate(DOMAINS, 1)}
    if any(len(v) != 96 for v in [*query.values(), *gallery.values()]):
        raise ValueError('Exactly 96 query and gallery views per batch')
    all_views = sum([query[i]+gallery[i] for i in (1, 2, 3)], [])
    if len({r.uid for v in all_views for r in v.records}) != 1152:
        raise ValueError('Evaluation roles overlap')
    complete = pinned(completed[task['id']]['completion'])
    result = dict(task=task, manifest_identity=batch['manifest'], endpoint_identity=complete['identity'], cells={})
    for t in range(4):
        model = restore(factory, task, verify(complete['endpoints'][str(t)]), complete['identity'], model_binding, t)
        needed = [c for c in CELLS if c[0] == t]
        qs = {d: encode_views(model, query[d], check) for d in sorted({c[1] for c in needed})}
        largest = max(c[2] for c in needed)
        gs = {d: encode_views(model, gallery[d], check) for d in range(1, largest+1)}
        for _, d, pool in needed:
            candidates = [(v.uid, i, j) for i in range(1, pool+1) for j, v in enumerate(gallery[i])]
            candidates.sort()
            g = torch.stack([gs[i][j] for _, i, j in candidates])
            scores = rectangular(model, qs[d], g, check)
            k = cell_key(t, d, pool)
            value = dict(cell=[t, d, pool], query_ids=[v.uid for v in query[d]],
                         gallery_ids=[uid for uid, _, _ in candidates], scores=scores.tolist())
            path = root/(k+'.json'); write(path, value); result['cells'][k] = record(path)
        del model, qs, gs, g
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    validate_scores(result)
    write(root/'blind_complete.json', result)
    return result


def validate_scores(result):
    if set(result['cells']) != {cell_key(*c) for c in CELLS}:
        raise ValueError('All fixed, growing and frozen cells required')
    for cell in CELLS:
        value = pinned(result['cells'][cell_key(*cell)])
        q, g, scores = value['query_ids'], value['gallery_ids'], np.asarray(value['scores'])
        if (value['cell'] != list(cell) or len(q) != 96 or len(g) != 96*cell[2]
                or len(set(q)) != len(q) or len(set(g)) != len(g) or set(q)&set(g)
                or q != sorted(q) or g != sorted(g) or scores.shape != (len(q), len(g))
                or not np.isfinite(scores).all()):
            raise ValueError('Blind shape/IDs/order/finiteness mismatch before qrels')


def query_metrics(value, qrels):
    qids, gids = value['query_ids'], value['gallery_ids']
    if (set(qids) != set(qrels) or len(set(gids)) != len(gids) or gids != sorted(gids)
            or not set(qrels.values()) <= set(gids) or len(set(qrels.values())) != len(qrels)):
        raise ValueError('Every query must align to exactly one distinct gallery positive')
    scores = np.asarray(value['scores'], dtype=float)
    if scores.shape != (len(qids), len(gids)) or not np.isfinite(scores).all():
        raise ValueError('Finite aligned rectangular scores required')
    positive = np.array([gids.index(qrels[q]) for q in qids])
    order = np.argsort(-scores, axis=1, kind='stable')  # Already byte-sorted ASCII hashes break ties.
    ranks = np.argmax(order == positive[:, None], axis=1)+1
    metrics = dict(map=1/ranks, mrr=1/ranks)
    for k in (1, 3, 5, 10):
        metrics[f'recall@{k}'] = (ranks <= k).astype(float)
        metrics[f'ndcg@{k}'] = np.where(ranks <= k, 1/np.log2(ranks+1), 0.)
    return dict(query_ids=qids, ranks=ranks.tolist(), per_query={k: v.tolist() for k, v in metrics.items()},
                mean={k: float(v.mean()) for k, v in metrics.items()})


def analyze(root, batch, manifest, score_receipts, check):
    validate_training(batch)
    expected = tasks()
    if len(score_receipts) != 6 or {r['task_id'] for r in score_receipts} != {t['id'] for t in expected}:
        raise ValueError('Complete scoring phase required before label access')
    scored = {r['task_id']: pinned(r['completion']) for r in score_receipts}
    # Validate every score and each exact query/gallery source before ANY qrels access.
    public_queries = {d: [v['uid'] for v in pinned(manifest['files'][f'{d}:query'])] for d in DOMAINS}
    public_gallery = {d: [v['uid'] for v in pinned(manifest['files'][f'{d}:gallery'])] for d in DOMAINS}
    for task in expected:
        run = scored[task['id']]
        if run['task'] != task or run['manifest_identity'] != batch['manifest']:
            raise ValueError('Scored task or dataset binding differs')
        validate_scores(run)
        for t, d, pool in CELLS:
            value = pinned(run['cells'][cell_key(t, d, pool)])
            if (value['query_ids'] != public_queries[DOMAINS[d-1]] or value['gallery_ids'] !=
                    sorted(sum([public_gallery[x] for x in DOMAINS[:pool]], []))):
                raise ValueError('Scores rebound to different query/gallery IDs')
        check()
    sources = [manifest['files'][f'{d}:qrels'] for d in DOMAINS]
    access(root, 'complete_P_labels', sources)
    qrels = {d: pinned(source) for d, source in zip(DOMAINS, sources)}
    rows = []
    for task in expected:
        run = scored[task['id']]
        cells = {cell_key(t, d, pool): query_metrics(pinned(run['cells'][cell_key(t, d, pool)]), qrels[DOMAINS[d-1]])
                 for t, d, pool in CELLS}
        write(root/(task['id']+'_metrics.json'), dict(task=task, cells=cells))
        rows.append(dict(task=task, cells={k: dict(mean=v['mean']) for k, v in cells.items()}))
        check()
    result = dict(rows=rows, decision=decide(rows), logical_scores=1824768,
                  matrices=90, training_trajectories=6, automatic_next_phase=False)
    write(root/'result.json', result)
    return result
