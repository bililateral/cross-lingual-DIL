"""Approved CPU-only retrospective evaluation of four fixed systems on old dev.

Run in the isolated Linux workspace. No model, item text, training labels or
owners are loaded. A single charged dev-pair read follows full blind preflight.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np

import step28_continual_population_evaluate as metrics
from step28_balanced_pooling_run import query_metrics, undefined_metrics
import step28_alias_calibration as calibration

METHODS = ('S_default', 'H_default', 'S_tuned', 'H_pos_tuned')
ORDERS, SEEDS = ('ABC', 'BCA', 'CAB'), (4, 5, 6)
ROLES = ('raw', 'stage-cal', 'first-cal')
PAIRS = (('H_default', 'S_default'), ('S_tuned', 'S_default'),
         ('H_pos_tuned', 'H_default'), ('H_pos_tuned', 'S_tuned'))
TERMS = {
    'O': ((3, 1, .5), (3, 2, .5)),
    'N': ((2, 2, .5), (3, 3, .5)), 'Z': ((3, 3, 1.),),
    'A_acq': ((1, 1, .5), (2, 2, .5)),
    'F': ((1, 1, .5), (3, 1, -.5), (2, 2, .5), (3, 2, -.5)),
    'F_first': ((1, 1, 1.), (3, 1, -1.)),
    'D2': ((1, 1, 1.), (2, 1, -1.)),
    'D3': ((2, 1, .5), (3, 1, -.5), (2, 2, .5), (3, 2, -.5)),
    'final_mean': ((3, 1, 1/3), (3, 2, 1/3), (3, 3, 1/3)),
}
PARAMETERS = {'S_default': (1., 1.), 'H_default': (1., 1.),
              'S_tuned': (1.3572393683805832, 1.295036258870551),
              'H_pos_tuned': (1.3051794547562026, 1.2832655111666686)}


def clean(v):
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    return v


def write(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, allow_nan=False,
                               separators=(',', ':'))+'\n', encoding='utf-8')


def checked(rec):
    raw = Path(rec['path']).read_bytes()
    if len(raw) != rec['bytes'] or hashlib.sha256(raw).hexdigest() != rec['sha256']:
        raise ValueError('Pinned input identity differs: '+rec['path'])
    return raw


def relative_record(root, rec):
    return {**rec, 'path': str(root/rec['path'])}


def dataset_binding(manifest, public_inputs, dataset):
    # These are manifest records only, never item/pair payload reads.
    if dataset['study'] != 'seller_alias_continual_expression_shift' or dataset['root_seed'] != 20260910:
        raise ValueError('Original development generation differs')
    if any(dataset['files'].get(k) != v for k, v in public_inputs.items()):
        raise ValueError('Score input identities and dataset manifest differ')
    pair = {**dataset['files']['development/supervision/pairs.csv'],
            'path': str(Path(manifest['dataset_manifest']['path']).parent/'development/supervision/pairs.csv')}
    if pair != manifest['development_pairs']:
        raise ValueError('Pair path/hash is not the matching development manifest member')


def preflight(manifest):
    expected = {f'{o}_s{s}_{m}' for m in METHODS for o in ORDERS for s in SEEDS}
    if set(manifest['runs']) != expected or len(manifest['runs']) != 36:
        raise ValueError('Exactly 36 fixed runs required')
    for rec in manifest['evaluation_sources']:
        checked(rec)
    records, arrays, common = {}, {}, None
    for rid, rec in manifest['runs'].items():
        row = json.loads(checked(rec)); root = Path(rec['path']).parent
        spec = row['spec']; name = spec['name']
        if (rid != f"{spec['order']}_s{spec['seed']}_{name}" or name not in METHODS
                or (spec['lambda_rank'], spec['lr_multiplier']) != PARAMETERS[name]
                or row['updates'] != 864 or row['status'] != 'COMPLETE_864_UPDATES_3_ENDPOINTS_VALID_BLIND'
                or row['access'] != dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)
                or row['source_files'] != manifest['training_sources'][name]
                or set(row['points']) != {'1', '2', '3'}):
            raise ValueError('Frozen completed trajectory differs: '+rid)
        identity = (row['partition'], row['public_inputs'])
        if common is None: common = identity
        if identity != common: raise ValueError('Four systems do not share the same dev data')
        dev = row['partition']['development']; ids = [x['group_uid'] for x in dev]
        if len(set(ids)) != 60 or [x['domain'] for x in dev] != [d for d in 'ABC' for _ in range(20)]:
            raise ValueError('Expected 20 ordered groups per actual domain')
        arrays[rid] = {}
        for stage, point in row['points'].items():
            if (point['development_ids'] != ids or point['adam_step'] != 288*int(stage)
                    or point['full_restore_verified'] is not True or point['inference_restore_verified'] is not True):
                raise ValueError('Stage, dev order or existing restoration evidence differs')
            fitted = json.loads(checked(relative_record(root, point['mapping'])))
            if ({k: fitted[k] for k in ('a', 'b')} != point['maps']['stage']
                    or point['maps']['first'] != row['points']['1']['maps']['stage']
                    or point['continual_spec'] != spec):
                raise ValueError('Stored calibration mapping or endpoint identity differs')
            vals = {}
            for role in ROLES:
                raw = checked(relative_record(root, point['scores'][role]))
                vals[role] = np.load(io.BytesIO(raw), allow_pickle=False)
                if vals[role].shape != (60, 378) or not np.isfinite(vals[role]).all():
                    raise ValueError('Incomplete blind score matrix')
            for role, key in (('stage-cal', 'stage'), ('first-cal', 'first')):
                computed = calibration.transform(vals['raw'], point['maps'][key])
                if not np.array_equal(computed, vals[role]): raise ValueError('Saved calibration transform differs')
                calibration.preserve_order(vals['raw'], vals[role])
            arrays[rid][int(stage)] = vals
        records[rid] = row
    if manifest['dataset_manifest']['sha256'] != '3f14f1ed47bffabaec90640b9e212f5eab2d702efcf9304a310f22d2a0165f6b':
        raise ValueError('Dataset manifest differs from the frozen training policy')
    dataset_binding(manifest, common[1], json.loads(checked(manifest['dataset_manifest'])))
    return records, arrays, common[0]['development']


def labels_from_bytes(raw, ids):
    """Dataset hash fixes the pair source; training used sorted account IDs too.

    Read no item text: enumerate the 28 accounts from the complete pair table,
    check all 378 canonical edges, then use the frozen lexicographic order.
    """
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8')))
    if reader.fieldnames != ['group_uid', 'seller_uid_left', 'seller_uid_right', 'label']:
        raise ValueError('Dev pair schema differs')
    groups = {}
    for r in reader:
        uid, a, b, value = (r[k] for k in reader.fieldnames)
        edges = groups.setdefault(uid, {})
        if a >= b or value not in ('0', '1') or (a, b) in edges:
            raise ValueError('Duplicate, unordered or nonbinary dev pair')
        edges[a, b] = int(value)
    if set(groups) != set(ids): raise ValueError('Dev label group identities differ')
    result = []
    for uid in ids:
        edges = groups[uid]; accounts = sorted({a for edge in edges for a in edge})
        pairs = list(itertools.combinations(accounts, 2))
        if len(accounts) != 28 or set(pairs) != set(edges): raise ValueError('Incomplete account graph')
        y = [edges[p] for p in pairs]
        if sum(y) != 20: raise ValueError('Expected 20 relevant pairs')
        result.append(y)
    return np.asarray(result, dtype=np.uint8)


def evaluate_matrix(truth, scores):
    values, counts = metrics.group_metrics(truth, scores)
    queries = query_metrics(truth, scores)
    if queries.shape != (len(truth), 28, 10) or not np.allclose(values[:, 12:], queries.mean(1), atol=1e-12, rtol=1e-12):
        raise ValueError('Query and group metrics disagree')
    if any(r['tp']+r['fn'] != 20 or r['tn']+r['fp'] != 358 for r in counts):
        raise ValueError('Confusion counts do not cover fixed positive/negative pairs')
    values[undefined_metrics(counts)] = np.nan
    return values, counts, queries


def pooled(counts):
    totals = {k: sum(c[k] for c in counts) for k in ('tp', 'fp', 'fn', 'tn')}
    v = metrics.confusion_metrics(**totals)
    if totals['tp']+totals['fp'] == 0: v['precision'] = None
    if math.prod((totals['tp']+totals['fp'], totals['tp']+totals['fn'],
                  totals['tn']+totals['fp'], totals['tn']+totals['fn'])) == 0: v['mcc'] = None
    return dict(counts=totals, metrics=v)


def coverage(values):
    a = np.asarray(values).reshape(-1, 22)
    valid = np.isfinite(a).sum(0)
    return {k: dict(total=len(a), valid=int(valid[j]), undefined=int(len(a)-valid[j]),
                   undefined_fraction=float(1-valid[j]/len(a))) for j, k in enumerate(metrics.COLUMNS)}


def contribution(order, cells, field):
    out = np.zeros((3, 20, 22), dtype=np.float64)
    for stage, position, weight in TERMS[field]:
        d = order[position-1]; out['ABC'.index(d)] += weight*cells[stage, d]
    return out


def interval(values, draws):
    samples = sum(values[d][draws[:, d]].mean(1) for d in range(3))
    point = values.mean(1).sum(0)
    return {k: dict(mean=clean(point[j]), conditional_95pct_interval=clean(np.quantile(samples[:, j], [.025, .975])))
            for j, k in enumerate(metrics.COLUMNS)}


def analyze(records, evaluated, check):
    draws = np.random.Generator(np.random.PCG64(2026110202)).integers(20, size=(5000, 3, 20))
    by_run, details, supplements = {}, {}, {}
    # All nine saved cells are reported; only the specified learned-domain
    # terms enter endpoint averages and paired comparisons.
    for rid, row in records.items():
        check(); order = row['spec']['order']; by_run[rid] = {}; details[rid] = {}
        for role in ROLES:
            cells = {(stage, d): evaluated[rid][stage][role][0][di*20:(di+1)*20]
                     for stage in (1, 2, 3) for di, d in enumerate('ABC')}
            by_run[rid][role] = {field: contribution(order, cells, field) for field in TERMS}
            details[rid][role] = {f'{stage}_{d}': dict(values=clean(vals.mean(0)), coverage=coverage(vals),
                pooled=pooled(evaluated[rid][stage][role][1]['ABC'.index(d)*20:('ABC'.index(d)+1)*20]))
                for (stage, d), vals in cells.items()}
    averaged = {}; absolute = {}; per_run = {}
    for rid, roles in by_run.items():
        per_run[rid] = {role: {field: clean(v.mean(1).sum(0)) for field, v in fs.items()} for role, fs in roles.items()}
    for name in METHODS:
        rids = [rid for rid, r in records.items() if r['spec']['name'] == name]
        averaged[name] = {}; absolute[name] = {}; supplements[name] = {}
        for role in ROLES:
            averaged[name][role] = {field: np.mean([by_run[rid][role][field] for rid in rids], axis=0) for field in TERMS}
            absolute[name][role] = {field: interval(v, draws) for field, v in averaged[name][role].items()}
            supplements[name][role] = {}
            for field, terms in TERMS.items():
                all_values = []; positive = []; negative = []
                # Every defined endpoint uses equal positive term weights and
                # (for differences) equal negative weights. Pool each side.
                for rid in rids:
                    order = records[rid]['spec']['order']
                    for stage, pos, weight in terms:
                        di = 'ABC'.index(order[pos-1]); sl = slice(di*20, (di+1)*20)
                        values, counts, _ = evaluated[rid][stage][role]
                        all_values.append(values[sl]); (positive if weight > 0 else negative).extend(counts[sl])
                pos = pooled(positive); neg = pooled(negative) if negative else None
                pv = pos['metrics'] if neg is None else {k: None if pos['metrics'][k] is None or neg['metrics'][k] is None
                      else pos['metrics'][k]-neg['metrics'][k] for k in pos['metrics']}
                supplements[name][role][field] = dict(coverage=coverage(np.concatenate(all_values)),
                    pooled_metrics=pv, positive_terms=pos, negative_terms=neg,
                    interpretation='Pooled counts per side, then difference; repeated predictions, not independent groups')
    comparisons = {a+'-'+b: {role: {field: interval(averaged[a][role][field]-averaged[b][role][field], draws)
                   for field in TERMS} for role in ROLES} for a, b in PAIRS}
    for roles in averaged.values():
        for fields in roles.values():
            np.testing.assert_allclose(fields['F'], fields['A_acq']-fields['O'], atol=1e-12, rtol=1e-12, equal_nan=True)
    return dict(status='COMPLETE_RETROSPECTIVE_OLD_DEV_NO_SELECTION', columns=list(metrics.COLUMNS),
                absolute=absolute, comparisons=comparisons, supplementary_classification=supplements,
                per_run=per_run, cells=details,
                bootstrap=dict(seed=2026110202, replicates=5000, actual_domains='ABC', groups_per_domain=20,
                               fixed_runs_per_method=9, shared_draws=True),
                limitation='Old development set used in earlier tuning; conditional retrospective intervals only, no independent confirmation or new success gate')


def execute(args):
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.getpriority(os.PRIO_PROCESS, 0) < 15
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
            or not 0 < args.seconds <= 7200):
        raise ValueError('Use the approved single-thread CPU launch and bounded allowance')
    started = time.monotonic(); out = args.output
    out.mkdir(exist_ok=False, parents=True)
    def check():
        if time.monotonic()-started > args.seconds: raise TimeoutError('Approved CPU evaluation allowance exhausted')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 4*1024*1024: raise MemoryError('4 GiB RSS cap')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 2*1024**3: raise OSError('2 GiB evidence cap')
    try:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        ready = json.loads(args.review_ready.read_text(encoding='utf-8'))
        if ready.get('status') != 'CLOSED_READY_FOR_APPROVED_A' or ready['manifest_sha256'] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
            raise ValueError('This exact implementation/input plan has not closed review')
        records, arrays, dev = preflight(manifest); check()
        write(out/'preflight.json', dict(runs=36, endpoints=108, matrices=324, groups=60,
            source_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(), status='ALL_BLIND_INPUTS_VERIFIED'))
        write(out/'evaluation_access.json', dict(attempts=[dict(role='old_dev_joint', keys=['development/supervision/pairs.csv'], epoch=time.time())]))
        truth = labels_from_bytes(checked(manifest['development_pairs']), [r['group_uid'] for r in dev])
        evaluated = {}; (out/'metrics').mkdir(); (out/'scores').mkdir()
        for rid, stages in arrays.items():
            evaluated[rid] = {}
            for stage, roles in stages.items():
                evaluated[rid][stage] = {}
                for role, scores in roles.items():
                    check(); values, counts, queries = evaluate_matrix(truth, scores)
                    evaluated[rid][stage][role] = (values, counts, queries)
                    stem = f'{rid}_{stage}_{role}'
                    write(out/'metrics'/(stem+'.json'), dict(columns=list(metrics.COLUMNS), values=values, counts=counts,
                        na_reason='Undefined precision/MCC denominator; never zero-fill'))
                    np.save(out/'metrics'/(stem+'_queries.npy'), queries, allow_pickle=False)
                    np.save(out/'scores'/(stem+'.npy'), scores, allow_pickle=False)
            print('EVALUATED',rid,flush=True)
        # No truth or pair source is exported. Subsequent verification uses saved
        # values, counts, scores and query-level metrics only.
        del truth
        write(out/'metric_completion.json',dict(runs=36, stages=108, matrices=324, group_evaluations=19440,
            group_ids=[r['group_uid'] for r in dev], domains=[r['domain'] for r in dev]))
        result = analyze(records, evaluated, check); check(); write(out/'result.json', result)
        write(out/'completion.json',dict(status=result['status'], epoch=time.time(), elapsed_seconds=time.monotonic()-started,
              charged_seconds=math.ceil(time.monotonic()-started), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              labels_read_attempts=1, gpu=False, model_loads=0, records=36, matrices=324))
    except Exception as error:
        write(out/'failure.json',dict(error=repr(error), epoch=time.time(), elapsed_seconds=time.monotonic()-started,
              charged_seconds=math.ceil(time.monotonic()-started), traceback=__import__('traceback').format_exc(),
              status='STOP_NO_LABEL_RETRY'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--review-ready', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, required=True)
    execute(parser.parse_args())
