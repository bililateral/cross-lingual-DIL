"""Q2/FIT12 scoring, current-encoder old V heads, and whole-batch release."""
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch

from step29_vendorlink_data import Ad, View, Group, seed_for
from step29_vendorlink_methods import vendor_embeddings
from step35_vendorlink_account_io import pinned, record, write
from step35_vendorlink_account_policy import CELLS, DOMAINS, SEEDS, summarize, task_domains
from step35_vendorlink_account_run import restore


def load_role(binding, domain, role, expected):
    rows = json.loads(pinned(binding).read_text())
    ids = [r['account_uid'] for r in rows]
    size = {'fit': 12, 'gallery': 4, 'dev': 2, 'confirm': 2}[role]
    if ids != sorted(set(ids)) or len(ids) != expected or any(len(r['records']) != size for r in rows):
        raise ValueError('Frozen role size/order changed')
    views = [View(f"{domain}:{role}:{r['account_uid']}",
                  tuple(Ad(**a) for a in r['records'])) for r in rows]
    return ids, views


@torch.no_grad()
def encode(model, views, check=lambda: None):
    model.eval()
    lengths = {len(v.records) for v in views}
    if len(lengths) != 1:
        raise ValueError('Uniform record count per role required')
    if model.settings.method != 'VendorLink_SEQ':
        z = model.records(Group('role', 'evaluation', 'dev', tuple(views)), check)
    else:
        texts = [r.title+' [SEP] '+r.description for v in views for r in v.records]
        parts = []
        for start in range(0, len(texts), model.settings.microbatch):
            check()
            output, mask = model.encode(texts[start:start+model.settings.microbatch], model.settings.vendor_tokens, True)
            parts.append(vendor_embeddings(output.hidden_states, mask))
        z = torch.cat(parts)
        if (z.norm(dim=1) == 0).any():
            raise ValueError('Zero record vector')
        z = torch.nn.functional.normalize(z.float(), dim=1)
    return z.reshape(len(views), next(iter(lengths)), -1)


@torch.no_grad()
def rectangular(model, q, g, check=lambda: None):
    if q.ndim != 3 or g.ndim != 3:
        raise ValueError('Account/record/vector tensors required')
    if model.settings.method == 'VendorLink_SEQ':
        u, v = q.mean(1), g.mean(1)
        denominator = u.square().sum(1)[:, None]+v.square().sum(1)[None, :]
        if (denominator <= 0).any():
            raise ValueError('Nonpositive original set denominator')
        scores = 2*(u@v.T)/denominator
    else:
        blocks = []
        for start in range(0, len(q), 8):
            lefts, rights = q[start:start+8].flatten(0, 1), g.flatten(0, 1)
            count, width, chunks = len(lefts)*len(rights), len(rights), []
            for offset in range(0, count, 512):
                check()
                index = torch.arange(offset, min(offset+512, count), device=q.device)
                a, b = lefts[index//width], rights[index%width]
                chunks.append(model.head(torch.cat(((a-b).abs(), a*b), 1)).flatten())
            logits = torch.cat(chunks).reshape(-1, q.shape[1], len(g), g.shape[1]).permute(0, 2, 1, 3).flatten(2).double()
            blocks.append((.5*logits.mean(2)+.5*(torch.logsumexp(logits, 2)-math.log(q.shape[1]*g.shape[1]))).float())
        scores = torch.cat(blocks)
    if scores.shape != (len(q), len(g)) or not torch.isfinite(scores).all():
        raise ValueError('Incomplete/nonfinite rectangular matrix')
    return scores.cpu().numpy().astype(np.float64)


@torch.no_grad()
def classification(model, query, classes, old_head, seed, domain, check=lambda: None):
    # Only the reading head changes; the CURRENT endpoint encoder is retained.
    model.start_vendor_stage(classes, seed_for(seed, domain, 'head'))
    if old_head is not None:
        if old_head['classes'] != classes:
            raise ValueError('Old class-index mapping mismatch')
        model.classifier.load_state_dict(old_head['state'], strict=True)
    model.eval()
    texts = [r.title+' [SEP] '+r.description for v in query for r in v.records]
    probabilities = []
    for start in range(0, len(texts), model.settings.microbatch):
        check()
        output, _ = model.encode(texts[start:start+model.settings.microbatch], model.settings.vendor_tokens)
        probabilities.append(model.classifier(model.dropout(output.pooler_output)).softmax(-1).cpu())
    ads = torch.cat(probabilities).numpy().astype(np.float64)
    return ads.reshape(len(query), len(query[0].records), len(classes)).mean(1), ads


def metrics(scores, labels, class_count):
    scores, truth = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=int)
    if scores.shape != (len(truth), class_count) or not np.isfinite(scores).all():
        raise ValueError('Nonfinite/incomplete full candidate scores')
    # Candidate IDs are ascending; stable sorting is the fixed tie rule.
    order = np.argsort(-scores, axis=1, kind='stable')
    predictions = order[:, 0]
    actual = np.bincount(truth, minlength=class_count)
    predicted = np.bincount(predictions, minlength=class_count)
    correct = np.bincount(truth[predictions == truth], minlength=class_count)
    denominator = actual+predicted
    f1 = np.divide(2*correct, denominator, out=np.zeros(class_count, dtype=float), where=denominator != 0)
    ranks = np.argmax(order == truth[:, None], axis=1)+1
    result = dict(macro_F1=float(f1.mean()), Top1=float((predictions == truth).mean()),
                  micro_F1=float((predictions == truth).mean()), MRR=float((1/ranks).mean()), MAP=float((1/ranks).mean()))
    for k in (1, 3, 5, 10):
        result[f'Recall@{k}'] = float((ranks <= k).mean())
        result[f'NDCG@{k}'] = float(np.where(ranks <= k, 1/np.log2(ranks+1), 0).mean())
    return result


@torch.no_grad()
def diagnostic_readings(q, g, payload, classes):
    if (payload['classes'] != classes or q.shape[-1] != payload['dimension']
            or g.shape[-1] != payload['dimension']
            or payload['scale'] != math.sqrt(payload['dimension'])):
        raise ValueError('Auxiliary class order/dimension/scale mismatch')
    w, b = (payload['state'][k].to(q) for k in ('weight', 'bias'))
    if w.shape != (len(classes), q.shape[-1]) or b.shape != (len(classes),):
        raise ValueError('Incomplete diagnostic classifier')
    logits = torch.nn.functional.linear(q*payload['scale'], w, b)
    auxiliary = logits.softmax(-1).mean(1)
    u, v = q.mean(1), g.mean(1)
    denominator = u.square().sum(1)[:, None]+v.square().sum(1)[None, :]
    if (denominator <= 0).any():
        raise ValueError('Nonpositive diagnostic set denominator')
    collection = 2*(u@v.T)/denominator
    outputs = {'H_AUX': auxiliary, 'H_SET': collection}
    if any(x.shape != (len(q), len(classes)) or not torch.isfinite(x).all() for x in outputs.values()):
        raise ValueError('Incomplete/nonfinite diagnostic reading')
    return {name: x.cpu().numpy().astype(np.float64) for name, x in outputs.items()}


def validate_rms_scores(completion):
    if (completion['status'] != 'BLIND_SCORES_COMPLETE' or completion['role'] != 'dev'
            or completion['cells'] != ['R11']
            or set(completion['files']) != {'H_SEQ:R11', 'H_AUX:R11', 'H_SET:R11'}
            or set(completion.get('diagnostics', {})) != {'1'}):
        raise ValueError('All three same-endpoint D readings are required before opening')
    for binding in list(completion['files'].values())+list(completion['diagnostics'].values()):
        pinned(binding)


def validate_scores(completion):
    task, role = completion['task'], completion['role']
    readers = ('V_SET', 'V_CLS', 'V_CLS_ads') if task['method'] == 'VendorLink_SEQ' else (task['method'],)
    cells = [f'R{t}{j}' for t, j in CELLS]
    if (completion['status'] != 'BLIND_SCORES_COMPLETE' or role not in ('dev', 'confirm')
            or completion['cells'] != cells or completion.get('reference_role') != 'fit'
            or completion.get('reference_records') != 12
            or set(completion['files']) != {m+':'+c for m in readers for c in cells}
            or set(completion['sources']) != {d+':'+r for d in task_domains(task) for r in (role, 'fit')}):
        raise ValueError('Complete current-endpoint FIT12 matrices required before release')
    for binding in completion['files'].values():
        pinned(binding)


def score(task, completion, manifest, binding, root, role, check):
    if root.exists() or completion['status'] != 'TRAINING_COMPLETE_EVALUATION_CLOSED':
        raise ValueError('Complete training and fresh blind scoring required')
    if role not in ('dev', 'confirm'):
        raise ValueError('Invalid query release')
    root.mkdir(parents=True)
    cells = ((1, 1),) if task['phase'] == 'D' else CELLS
    domains = task_domains(task)
    sources = {d+':'+r: manifest['files'][d+':'+r] for d in domains for r in (role, 'fit')}
    write(root/'access.json', dict(role=role, sources=sources, training_identity=completion['identity']))
    inputs, exports = {}, {}
    for domain in domains:
        n = manifest['counts'][domain]['eligible_accounts']
        qids, query = load_role(sources[domain+':'+role], domain, role, n)
        gids, gallery = load_role(sources[domain+':fit'], domain, 'fit', n)
        if qids != gids:
            raise ValueError('Query/gallery registered classes differ')
        if ({r.uid for v in query for r in v.records} & {r.uid for v in gallery for r in v.records}
                or {v.uid for v in query} & {v.uid for v in gallery}):
            raise ValueError('Query/gallery text or view overlap')
        inputs[domain] = (query, gallery, gids)
    for stage in sorted({t for t, j in cells}):
        endpoint = completion['endpoints'][str(stage)]
        model, heads = restore(binding, task, pinned(endpoint), completion['identity'], stage)
        for t, j in [c for c in cells if c[0] == stage]:
            check()
            domain = domains[j-1]
            query, gallery, classes = inputs[domain]
            q, g = encode(model, query, check), encode(model, gallery, check)
            values = rectangular(model, q, g, check)
            name = 'V_SET' if task['method'] == 'VendorLink_SEQ' else task['method']
            cell = f'R{t}{j}'
            path = root/f'{name}_{cell}.npy'
            np.save(path, values)
            exports[name+':'+cell] = record(path)
            if task['settings'].get('diagnostics'):
                diagnostic_binding = completion['diagnostics'][str(stage)]
                payload = torch.load(pinned(diagnostic_binding), map_location='cpu', weights_only=False)
                if (payload['identity'] != completion['identity'] or payload['task'] != task
                        or payload['model_binding'] != binding or payload['stage'] != stage
                        or payload['endpoint'] != endpoint):
                    raise ValueError('Diagnostic head must belong to this exact trained endpoint')
                for label, matrix in diagnostic_readings(q, g, payload, classes).items():
                    p = root/f'{label}_{cell}.npy'
                    np.save(p, matrix)
                    exports[label+':'+cell] = record(p)
                del payload
            if name == 'V_SET':
                if t >= j and domain not in heads:
                    raise ValueError('Missing learned old head')
                cls, ads = classification(model, query, classes, heads.get(domain), task['seed'], domain, check)
                for label, matrix in (('V_CLS', cls), ('V_CLS_ads', ads)):
                    p = root/f'{label}_{cell}.npy'
                    np.save(p, matrix)
                    exports[label+':'+cell] = record(p)
            del q, g, values
        del model, heads
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = dict(task=task, role=role, training_identity=completion['identity'], files=exports,
                  status='BLIND_SCORES_COMPLETE', cells=[f'R{t}{j}' for t, j in cells], sources=sources,
                  reference_role='fit', reference_records=12,
                  diagnostics=completion.get('diagnostics', {}))
    if task['settings'].get('diagnostics'):
        validate_rms_scores(result)
    write(root/'completion.json', result)
    return result


def analyze(completions, expected_tasks, root):
    if root.exists() or [c['task'] for c in completions] != expected_tasks:
        raise ValueError('Whole fixed batch must finish before opening any result')
    for c in completions:
        validate_scores(c)
    root.mkdir(parents=True)
    write(root/'access.json', dict(tasks=expected_tasks, opening='whole_batch_once'))
    all_metrics, results = {}, {}
    for c in completions:
        if c['status'] != 'BLIND_SCORES_COMPLETE':
            raise ValueError('Incomplete scores')
        rows = {}
        for name, binding in c['files'].items():
            method, cell = name.split(':')
            scores = np.load(pinned(binding), allow_pickle=False)
            n = scores.shape[1]
            truth = np.repeat(np.arange(n), 2) if method == 'V_CLS_ads' else np.arange(n)
            m = metrics(scores, truth, n)
            if len(m) != 13 or any(not math.isfinite(v) for v in m.values()):
                raise ValueError('Thirteen complete finite metrics required')
            rows.setdefault(method, {})[cell] = m['macro_F1']
            all_metrics[c['task']['id']+':'+name] = m
        per_task = {}
        for method, matrix in rows.items():
            row = matrix if c['task']['phase'] == 'D' else summarize(matrix)
            per_task[method] = [dict(seed=c['task']['seed'], **row)]
            results.setdefault(method, []).append(dict(seed=c['task']['seed'], **row))
        if c['task']['phase'] == 'D':
            results[c['task']['id']] = per_task
    for name, rows in results.items():
        if isinstance(rows, list):
            rows.sort(key=lambda r: r['seed'])
    write(root/'metrics.json', all_metrics)
    write(root/'results.json', results)
    return results


def application(gate, selected, root, check):
    from collections import Counter
    from step35_vendorlink_account_policy import tasks
    if root.exists():
        raise ValueError('Fresh application output required')
    root.mkdir(parents=True)
    campaign, binding = Path(gate['campaign']), gate['model_binding']
    manifest = json.loads(pinned(gate['manifest']).read_text())
    sources = {d: manifest['files'][d+':fit'] for d in DOMAINS}
    write(root/'access.json', dict(sources=sources, purpose='unlabelled_candidate_discovery', uses_deleted_alias_data=False))
    all_items = []
    for domain in DOMAINS:
        ids, views = load_role(sources[domain], domain, 'fit', manifest['counts'][domain]['eligible_accounts'])
        all_items.extend((v.uid, i, domain, v) for i, v in zip(ids, views))
    all_items.sort(key=lambda row: row[0])
    views = [r[3] for r in all_items]
    markets = np.array([r[2] for r in all_items])
    chosen = [t for t in tasks('M', selected)+tasks('P', selected) if t['method'] in ('H_full', 'VendorLink_SEQ')]
    rankings, frequencies, file_bindings = {}, Counter(), []
    for task in chosen:
        completion_path = campaign/task['phase']/'training'/task['id']/'run/completion.json'
        completion = json.loads(completion_path.read_text())
        endpoint = completion['endpoints']['3']
        model, heads = restore(binding, task, pinned(endpoint), completion['identity'], 3)
        z = encode(model, views, check)
        scores = rectangular(model, z, z, check)
        method = 'V_SET' if task['method'] == 'VendorLink_SEQ' else 'H_full'
        entries = []
        for i, (_, account, domain, view) in enumerate(all_items):
            check()
            for mode, mask in (('within_market', markets == domain), ('cross_market', markets != domain)):
                indices = np.flatnonzero(mask)
                indices = indices[indices != i]
                order = indices[np.argsort(-scores[i, indices], kind='stable')[:10]]
                rankings[(method, task['seed'], account, mode)] = set(all_items[j][1] for j in order)
                for rank, j in enumerate(order, 1):
                    candidate = all_items[j][1]
                    frequencies[(method, account, mode, candidate)] += 1
                    entries.append(dict(account=account, market=domain, mode=mode, candidate=candidate,
                        candidate_market=all_items[j][2], rank=rank, score=float(scores[i, j]),
                        seed=task['seed'], method=method))
        path = root/f'{method}_s{task["seed"]}.json'
        write(path, dict(endpoint=endpoint, entries=entries))
        file_bindings.append(record(path))
        del model, heads, z, scores, entries
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    unions, categories = [], Counter()
    for seed in SEEDS:
        for _, account, domain, view in all_items:
            for mode in ('within_market', 'cross_market'):
                h, v = (rankings[(m, seed, account, mode)] for m in ('H_full', 'V_SET'))
                for candidate in sorted(h | v):
                    label = 'both' if candidate in h & v else ('only_H' if candidate in h else 'only_V')
                    categories[label] += 1
                    unions.append(dict(seed=seed, account=account, mode=mode, candidate=candidate, category=label,
                        H_seed_occurrences=frequencies[('H_full', account, mode, candidate)],
                        V_seed_occurrences=frequencies[('V_SET', account, mode, candidate)]))
    path = root/'candidate_union.json'
    write(path, unions)
    file_bindings.append(record(path))
    write(root/'completion.json', dict(status='UNLABELLED_APPLICATION_COMPLETE', files=file_bindings,
          categories=dict(categories), true_alias_precision=None, known_owner_labels=False))
