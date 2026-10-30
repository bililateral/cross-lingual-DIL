"""Fixed blind triangle and descriptive comparison after one joint qrels opening."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

import step29_vendorlink_formal as formal
from step29_vendorlink_data import eligible_ads, split_for, group_chunks, seed_for, Group

queue, run, DOMAINS = formal.queue, formal.run, formal.DOMAINS
CELLS = ((1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3))
FIELDS = dict(Q=((1, 1, .5), (2, 2, .5)), O=((3, 1, .5), (3, 2, .5)),
              F=((1, 1, .5), (2, 2, .5), (3, 1, -.5), (3, 2, -.5)), N=((3, 3, 1.),))
METRICS = ('map', 'mrr', 'recall@1', 'recall@3', 'recall@5', 'recall@10',
           'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10')


def read(row):
    content = Path(row['path']).read_bytes()
    if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
        raise ValueError('Pinned JSON identity differs')
    return json.loads(content)


def access(campaign, role):
    path = campaign/'evaluation_access.json'
    value = json.loads(path.read_text()) if path.exists() else dict(events=[])
    if role in [r['role'] for r in value['events']]:
        raise ValueError('This access was already attempted; no automatic retry')
    value['events'].append(dict(role=role))
    queue.write(path, value)


def group_dependencies(rows, partition):
    """Custodian only. Emit group components, never handles, text or qrels."""
    eligible, _, _, _ = eligible_ads(rows, partition['records_per_view'])
    buckets = defaultdict(list)
    for market, vendor in sorted(eligible):
        if split_for(vendor, partition['split_seed'], partition['train_fraction'], partition['dev_fraction']) == 'heldout':
            identity = hashlib.sha256(json.dumps((market, vendor), ensure_ascii=False).encode()).hexdigest()
            buckets[market].append((identity, vendor.casefold()))
    handles, sizes = defaultdict(list), {}
    for market, identities in sorted(buckets.items()):
        identities.sort(key=lambda x: (seed_for(partition['split_seed'], 'groups', market, x[0]), x[0]))
        for index, chunk in enumerate(group_chunks(identities, partition['identities_per_group'])):
            uid = f'{market}:heldout:{index}'
            sizes[uid] = 2 * len(chunk)
            for _, handle in chunk:
                handles[handle].append(uid)
    parent = {uid: uid for uid in sizes}
    def find(uid):
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid
    for connected in handles.values():
        for uid in connected[1:]:
            a, b = find(connected[0]), find(uid)
            parent[max(a, b)] = min(a, b)
    components = defaultdict(list)
    for uid in sorted(sizes):
        components[find(uid)].append(uid)
    return dict(components=sorted(components.values()), query_counts=sizes,
        scope='Shared candidates within a group; identical casefold handles across/within markets. Not verified real owners.')


def prepare_dependencies(job, allowed):
    # Permission is specifically for this custodian scan of the already pinned
    # canonical cache, not for reopening qrels or exporting identities.
    if allowed.get('dependency_metadata_access') is not True:
        raise ValueError('Canonical-to-group dependence metadata access is not approved')
    access(job.parent, 'canonical_dependency_scan')
    source = queue.verify(allowed['canonical'])
    with source.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    value = group_dependencies(rows, allowed['partition'])
    expected = {f'{domain}:heldout:{i}' for domain, count in
                (('AlphaBay', 22), ('Dream', 8), ('Agora', 27)) for i in range(count)}
    if set(value['query_counts']) != expected or sum(value['query_counts'].values()) != 912:
        raise ValueError('Dependence metadata does not match the fixed export')
    queue.write(job/'dependencies.json', {**value, 'canonical': allowed['canonical'],
        'partition': allowed['partition'], 'sources': allowed['sources']})


def complete_training(campaign, allowed):
    batch = json.loads((campaign/'training_complete.json').read_text())
    if (batch['status'] != 'ALL_TRAINING_COMPLETE_EVALUATION_CLOSED'
            or {r['id'] for r in batch['completed']} != {s['id'] for s in formal.specifications()}
            or len(batch['completed']) != 36 or not all(r['success'] for r in batch['completed'])):
        raise ValueError('All 36 actual trajectories are required before scoring')
    return {s['id']: formal.validate_completion(s, campaign/s['id']/'run', allowed)
            for s in formal.specifications()}


def validate_blind(result, allowed):
    if set(result['runs']) != {s['id'] for s in formal.specifications()} or result['forwards'] != 4104:
        raise ValueError('Incomplete 36-run / 216-cell / 4104-forward blind result')
    public = {domain: [Group.from_dict(g) for g in read(rec)] for domain, rec in allowed['heldout_sources'].items()}
    for spec in formal.specifications():
        cells = result['runs'][spec['id']]
        if set(cells) != {f'{i}_{j}' for i, j in CELLS}:
            raise ValueError('Missing or extra blind cell')
        for stage, position in CELLS:
            domain = DOMAINS[spec['order'][position-1]]
            cell = cells[f'{stage}_{position}']
            values = read(cell['scores'])
            if cell['domain'] != domain or set(values) != {g.uid for g in public[domain]}:
                raise ValueError('Actual market/group IDs differ')
            for group in public[domain]:
                row = values[group.uid]
                scores = np.asarray(row['scores'], dtype=np.float64)
                if (row['view_ids'] != [v.uid for v in group.views]
                        or scores.shape != (len(group.views) * (len(group.views)-1)//2,)
                        or not np.isfinite(scores).all()):
                    raise ValueError('Blind score shape, IDs or finite values differ')
    return public


def score(campaign, job, allowed, budget):
    import torch
    from transformers import BertModel, BertTokenizer
    records = complete_training(campaign, allowed)
    access(campaign, 'heldout_blind_text_scoring')
    public = {domain: [Group.from_dict(g) for g in read(rec)] for domain, rec in allowed['heldout_sources'].items()}
    if any(g.split != 'heldout' or any(v.identity is not None for v in g.views)
           for groups in public.values() for g in groups):
        raise ValueError('Scoring receives only fixed blinded held-out inputs')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = BertTokenizer.from_pretrained(allowed['model_path'], local_files_only=True, do_lower_case=True)
    result = dict(runs={}, forwards=0, sources=allowed['sources'])
    for spec in formal.specifications():
        budget.check()
        model = formal.Model(BertModel.from_pretrained(allowed['model_path'], local_files_only=True),
                             tokenizer, formal.Settings(method=spec['method'])).to('cuda:0')
        result['runs'][spec['id']] = {}
        for stage in (1, 2, 3):
            state = torch.load(queue.verify(records[spec['id']]['endpoint_files'][str(stage)]),
                               map_location='cpu', weights_only=False)
            if state['identity'] != records[spec['id']]['identity'] or state['stage'] != stage:
                raise ValueError('Restored inference endpoint identity differs')
            if spec['method'] == 'VendorLink_SEQ':
                model.start_vendor_stage(state['classes'], 0)
            model.load_state_dict(state['model'], strict=True)
            del state
            for _, position in (cell for cell in CELLS if cell[0] == stage):
                domain, key = DOMAINS[spec['order'][position-1]], f'{stage}_{position}'
                values = run.evaluate(model, public[domain], budget.check)
                path = job/f'{spec["id"]}_{key}.json'
                queue.write(path, values)
                result['runs'][spec['id']][key] = dict(domain=domain, scores=queue.record(path))
                result['forwards'] += len(values)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        queue.write(job/'progress.json', result)
    validate_blind(result, allowed)
    queue.write(job/'all_blind_complete.json', result)


def query_metrics(scores, labels, ids):
    ap = np.asarray(run.query_ap(scores, labels, ids))
    n = len(ids)
    table, truth = np.zeros((n, n)), np.zeros((n, n), dtype=bool)
    upper = np.triu_indices(n, 1)
    table[upper], truth[upper] = scores, labels
    table += table.T
    truth |= truth.T
    if not np.all(truth.sum(1) == 1):
        raise ValueError('This fixed proxy requires exactly one positive per query')
    result = []
    for q in range(n):
        ordered = sorted((i for i in range(n) if i != q), key=lambda i: (-table[q, i], ids[i]))
        rank = next(k for k, i in enumerate(ordered, 1) if truth[q, i])
        result.append([ap[q], 1/rank, *[float(rank <= k) for k in (1, 3, 5, 10)],
                       *[1/np.log2(rank+1) if rank <= k else 0. for k in (1, 3, 5, 10)]])
    result = np.asarray(result, np.float64)
    if not np.array_equal(result[:, 0], result[:, 1]):
        raise ValueError('Single-positive AP/MRR discrepancy')
    return result


def aggregate(totals, by_domain, dependency):
    """Each query has equal weight within its market; no independence is assumed."""
    absolute = {}
    for name, fields in totals.items():
        absolute[name] = {}
        for field, domains in fields.items():
            absolute[name][field] = sum(v.sum(0) / sum(dependency['query_counts'][uid] for uid in by_domain[d])
                                         for d, v in domains.items())
    return absolute


def comparison_summary(absolute):
    """The approved fixed-data scope cannot produce statistical acceptance."""
    comparisons = {}
    for other in ('VendorLink_SEQ', 'H_ER', 'H_SEQ'):
        comparisons['H_full-' + other] = {}
        for field in FIELDS:
            point = absolute['H_full'][field] - absolute[other][field]
            comparisons['H_full-' + other][field] = {metric: dict(mean=float(point[i]),
                conditional_95pct_interval=None) for i, metric in enumerate(METRICS)}
    return dict(comparisons=comparisons, primary_status='DESCRIPTIVE_FIXED_DATA_ONLY', passed=None)


def analyze(campaign, job, allowed, budget):
    complete_training(campaign, allowed)
    blind = json.loads((campaign/'scoring/all_blind_complete.json').read_text())
    if blind['sources'] != allowed['sources']:
        raise ValueError('Scoring source differs')
    public = validate_blind(blind, allowed)  # Entire batch before any qrels access.
    dependency = read(allowed['dependency_metadata'])
    if dependency['canonical'] != allowed['canonical'] or dependency['partition'] != allowed['partition']:
        raise ValueError('Dependence metadata source differs')
    group_ids = sorted(g.uid for groups in public.values() for g in groups)
    if (sorted(uid for comp in dependency['components'] for uid in comp) != group_ids
            or any(dependency['query_counts'][g.uid] != len(g.views) for groups in public.values() for g in groups)):
        raise ValueError('Dependence graph/group sizes differ')
    access(campaign, 'heldout_qrels_joint_opening')
    truth = {domain: read(rec) for domain, rec in allowed['qrels_sources'].items()}
    by_domain = {d: sorted(g.uid for g in public[d]) for d in public}
    totals = {name: {field: {d: np.zeros((len(ids), len(METRICS))) for d, ids in by_domain.items()}
                     for field in FIELDS} for name in formal.METHODS}
    per_run = {}
    for spec in formal.specifications():
        cells = {}
        for key, cell in blind['runs'][spec['id']].items():
            budget.check()
            domain, values = cell['domain'], read(cell['scores'])
            queries = {}
            for uid in by_domain[domain]:
                row, label = values[uid], truth[domain][uid]
                if row['view_ids'] != label['view_ids']:
                    raise ValueError('Private qrels and blind query order differ')
                queries[uid] = query_metrics(row['scores'], label['labels'], row['view_ids'])
            cell_rows = np.asarray([queries[uid].sum(0) for uid in by_domain[domain]])
            cells[key] = cell_rows
            queue.write(job/f'{spec["id"]}_{key}_queries.json',
                        dict(columns=METRICS, queries={uid: v.tolist() for uid, v in queries.items()}))
        per_run[spec['id']] = {}
        for field, terms in FIELDS.items():
            point = np.zeros(len(METRICS))
            for stage, position, coefficient in terms:
                domain = DOMAINS[spec['order'][position-1]]
                values = cells[f'{stage}_{position}']
                totals[spec['method']][field][domain] += coefficient * values / 9
                point += coefficient * values.sum(0) / sum(dependency['query_counts'][uid] for uid in by_domain[domain])
            per_run[spec['id']][field] = point.tolist()
    absolute = aggregate(totals, by_domain, dependency)
    for fields in absolute.values():
        np.testing.assert_allclose(fields['O'], fields['Q'] - fields['F'], rtol=1e-10, atol=1e-10)
    queue.write(job/'result.json', dict(status='VENDORLINK_FIXED_ANALYSIS_COMPLETE', columns=METRICS,
        **comparison_summary(absolute), per_run=per_run,
        absolute={name: {field: values.tolist() for field, values in fields.items()} for name, fields in absolute.items()},
        dependence=dict(metadata=allowed['dependency_metadata'], components=len(dependency['components']),
            component_group_sizes=sorted([len(c) for c in dependency['components']], reverse=True)),
        limits='Descriptive comparison on fixed observed-account views, candidate groups and training paths only. No overall confidence intervals or statistical confirmation of forgetting reduction. Casefold links are dependence links, not verified owners.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('score', 'analyze'))
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    allowed = queue.gate(args.gate, 'VendorLink', args.action)
    campaign = Path(allowed['campaign'])
    formal.validate_scope(allowed)
    job = campaign/{'score': 'scoring', 'analyze': 'analysis'}[args.action]
    if job.exists() and any(p.name not in ('console.txt', 'process.json') for p in job.iterdir()):
        raise ValueError('A previous action cannot silently restart')
    job.mkdir(exist_ok=True)
    budget = formal.Budget(allowed, args.deadline)
    try:
        if args.action == 'score':
            score(campaign, job, allowed, budget)
        else:
            analyze(campaign, job, allowed, budget)
    except BaseException as error:
        queue.write(job/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
