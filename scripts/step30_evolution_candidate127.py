"""One fixed full-dev-pool diagnostic; no fitting, test access, or retries.

The original Evolution workspace is imported only after its frozen source
bindings are verified. All existing training and scoring sources stay unchanged.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PYTHON = '/home/yongpeng/miniconda3/envs/py310/bin/python'
ROOT = Path('/home/yongpeng/cross-lingual/reports/documentation/20261107/evolution_pilot')
DELTA_CAPS = {'gpu': 1800, 'auxiliary': 1800}
PREPARATION_SECONDS = 120  # Conservative connection/sync/package/return work; not human time.
TAIL_SECONDS = 30
ATOL, RTOL = 1e-5, 1e-5  # FP32 consistency with the saved within-group scores.


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def binding(path):
    p = Path(path).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


def checked(row):
    if binding(row['path']) != row:
        raise ValueError('Source binding mismatch: '+row['path'])
    return Path(row['path'])


def permitted(path):
    gate = read(path)
    if (sys.platform != 'linux' or Path(sys.executable).resolve() != Path(PYTHON).resolve()
            or gate.get('operation') != 'Evolution_candidate127_dev_once'
            or gate.get('approved') is not True or gate.get('review_closed') is not True
            or gate.get('delta_caps') != DELTA_CAPS or gate.get('workers') != 1
            or gate.get('automatic_retry') is not False or gate.get('automatic_wait') is not False
            or gate.get('preparation_seconds') != PREPARATION_SECONDS
            or gate.get('tail_seconds') != TAIL_SECONDS
            or gate.get('standalone_cpu_gpu_tests') is not False
            or Path(gate['output']).resolve() != ROOT/'candidate127/formal'):
        raise ValueError('Reviewed, single-use candidate127 permission required')
    for key in ('scope', 'authorization', 'review_disposition', 'source', 'original_gate',
                'original_completion', 'original_budget', 'batch', 'old_blind', 'old_metrics', 'old_decision'):
        checked(gate[key])
    if Path(gate['source']['path']).resolve() != Path(__file__).resolve():
        raise ValueError('Wrong actual candidate127 entry')
    old = read(gate['original_gate']['path'])
    if (Path(old['experiment_root']).resolve() != ROOT
            or Path(old['campaign']).resolve() != ROOT/'execution/campaign'
            or read(gate['original_completion']['path'])['status'] != 'STOP_B_NO_JOINT_FORGETTING'):
        raise ValueError('Only the completed original B campaign is reusable')
    expected = {'original_completion': 'completion.json', 'original_budget': 'budget.json',
                'batch': 'B/batch.json', 'old_blind': 'B/blind_complete.json',
                'old_metrics': 'B/metrics.json', 'old_decision': 'B/decision.json'}
    for key, suffix in expected.items():
        if Path(gate[key]['path']).resolve() != Path(old['campaign'])/suffix:
            raise ValueError('Wrong original B source: '+key)
    for row in old['sources']:
        checked(row)
    scripts = Path(old['workspace'])/'scripts'
    sys.path.insert(0, str(scripts))
    from step30_evolution_io import dataset
    from step30_evolution_policy import CAPS, LIMITS
    if gate['limits'] != LIMITS or old['budget_seconds'] != CAPS:
        raise ValueError('Existing 14-GiB and cumulative limits must remain unchanged')
    # Reading the manifest/language receipt does not open any text or qrels.
    manifest = dataset(old)
    return gate, old, manifest


def original_batch(gate, old):
    from step30_evolution_evaluate import training_complete, validate_blind
    from step30_evolution_io import pinned
    from step30_evolution_policy import tasks
    batch, blind = read(gate['batch']['path']), read(gate['old_blind']['path'])
    specs = tasks('B', batch['selected'])
    if (batch['phase'] != 'B' or batch['specs'] != specs or len(batch['completed']) != 6
            or batch['gate'] != gate['original_gate'] or blind['phase'] != 'B'
            or blind['model_binding'] != old['model_binding']):
        raise ValueError('Exact six original B trajectories required')
    validate_blind(blind, specs)
    completed = {r['task_id']: r for r in batch['completed']}
    if len(completed) != 6 or set(completed) != {s['id'] for s in specs}:
        raise ValueError('Missing or duplicate original B trajectory')
    for task in specs:
        row = completed[task['id']]
        expected = dict(gate=batch['gate'], spec=task, dataset=old['dataset_manifest'])
        if row['identity'] != expected:
            raise ValueError('Original trajectory/dataset identity mismatch')
        training_complete(task, pinned(row['completion']), expected)
    return batch, blind, completed


def vector_matrix(vector, n):
    import numpy as np
    a = np.asarray(vector, dtype=float)
    if a.shape != (n*(n-1)//2,) or not np.isfinite(a).all():
        raise ValueError('Incomplete or nonfinite pair vector')
    matrix = np.zeros((n, n), dtype=float)
    index = np.triu_indices(n, 1)
    matrix[index] = a
    matrix += matrix.T
    return matrix


def validate_pool_result(result, batch):
    from step30_evolution_io import pinned
    from step30_evolution_policy import DOMAINS, CELLS
    if (result['specs'] != batch['specs'] or len(result['runs']) != 6
            or result['candidate_count'] != 127 or set(result['inventory']) != set(DOMAINS)):
        raise ValueError('Full six-trajectory fixed-pool scores required')
    all_ids = []
    for groups in result['inventory'].values():
        ids = [v for group in groups.values() for v in group]
        if len(groups) != 8 or any(len(v) != 16 for v in groups.values()) or len(set(ids)) != 128:
            raise ValueError('Exactly 64 accounts / 128 distinct views per domain required')
        all_ids.extend(ids)
    if len(set(all_ids)) != 384:
        raise ValueError('Views overlap across domains')
    by_id = {r['spec']['id']: r for r in result['runs']}
    if len(by_id) != 6 or set(by_id) != {s['id'] for s in batch['specs']}:
        raise ValueError('Missing or duplicate scoring trajectory')
    for spec in batch['specs']:
        run = by_id[spec['id']]
        if run['spec'] != spec or set(run['cells']) != {f'{t}_{d}' for t, d in CELLS}:
            raise ValueError('All nine cells per trajectory required')
        for t, d in CELLS:
            row = pinned(run['cells'][f'{t}_{d}'])
            ids = [v for vs in result['inventory'][DOMAINS[d-1]].values() for v in vs]
            if row['view_ids'] != ids:
                raise ValueError('Candidate universe changed across endpoints')
            vector_matrix(row['scores'], 128)


def score(gate, old, manifest, check):
    import numpy as np
    import torch
    from step29_vendorlink_data import Group, assignment
    from step29_vendorlink_methods import h_pool, vendor_embeddings, vendor_scores
    from step30_evolution_control import configure_gpu
    from step30_evolution_data import validate_group
    from step30_evolution_io import access, pinned, record, verify, write
    from step30_evolution_policy import DOMAINS, CELLS
    from step30_evolution_run import make_model, restore
    root = Path(gate['output'])
    batch, old_blind, completed = original_batch(gate, old)
    sources = [manifest['files'][d+':dev'] for d in DOMAINS]
    if sources != old_blind['blind_sources']:
        raise ValueError('Only the already opened B dev texts are permitted')
    access(root, 'candidate127_dev_text', sources)
    public, inventory, record_ids = {}, {}, []
    for domain, source in zip(DOMAINS, sources):
        groups = sorted((Group.from_dict(x) for x in pinned(source)), key=lambda g: g.uid)
        if len(groups) != 8 or len({g.uid for g in groups}) != 8:
            raise ValueError('Eight original groups required')
        for group in groups:
            validate_group(group, False)
            if group.domain != domain or group.split != 'dev':
                raise ValueError('No fit/test or other domain permitted')
            record_ids.extend(r.uid for v in group.views for r in v.records)
        inventory[domain] = {g.uid: [v.uid for v in g.views] for g in groups}
        if inventory[domain] != old_blind['inventory'][domain]:
            raise ValueError('Original view identities differ')
        public[domain] = groups
    if len(record_ids) != 768 or len(set(record_ids)) != 768:
        raise ValueError('Dev record overlap/quota mismatch')
    model_binding = old['model_binding']
    for source in model_binding['files']:
        verify(source)
    configure_gpu()
    factory = lambda task: make_model(Path(model_binding['directory']), task)
    result = dict(specs=batch['specs'], inventory=inventory, candidate_count=127,
                  blind_sources=sources, original_blind=gate['old_blind'], runs=[],
                  within_group_atol=ATOL, within_group_rtol=RTOL)
    old_runs = {r['spec']['id']: r for r in old_blind['runs']}
    with torch.no_grad():
        for task in batch['specs']:
            saved = pinned(completed[task['id']]['completion'])
            run = dict(spec=task, cells={}, within_group_max_abs=0.)
            for stage in range(4):
                check()
                model = restore(factory, task, verify(saved['endpoints'][str(stage)]),
                                completed[task['id']]['identity'], model_binding, stage)
                for t, position in CELLS:
                    if t != stage:
                        continue
                    check()
                    domain = DOMAINS[position-1]
                    groups = public[domain]
                    joined = Group('candidate127-'+domain, domain, 'dev',
                                   tuple(v for g in groups for v in g.views))
                    parts = []
                    # Preserve each original group's microbatch/tokenization shape.
                    for group in groups:
                        if task['system'] == 'H_SEQ':
                            parts.append(model.records(group, check))
                        else:
                            texts = [r.title+' [SEP] '+r.description for v in group.views for r in v.records]
                            for start in range(0, len(texts), model.settings.microbatch):
                                check()
                                encoded, mask = model.encode(texts[start:start+model.settings.microbatch],
                                                             model.settings.vendor_tokens, True)
                                parts.append(vendor_embeddings(encoded.hidden_states, mask))
                    vectors = torch.cat(parts)
                    slots = assignment(joined)
                    # Exact unchanged record-pair head and mean/LogMeanExp account aggregation.
                    values = (h_pool(model.table(vectors), slots) if task['system'] == 'H_SEQ'
                              else vendor_scores(vectors, slots)).cpu().numpy().astype(float)
                    matrix = vector_matrix(values, 128)
                    previous = pinned(old_runs[task['id']]['cells'][f'{t}_{position}'])
                    for k, group in enumerate(groups):
                        old_row = previous[group.uid]
                        block = matrix[k*16:(k+1)*16, k*16:(k+1)*16][np.triu_indices(16, 1)]
                        before = np.asarray(old_row['scores'])
                        if old_row['view_ids'] != inventory[domain][group.uid] or not np.allclose(block, before, atol=ATOL, rtol=RTOL):
                            raise ValueError('New scoring changed the original within-group scores')
                        run['within_group_max_abs'] = max(run['within_group_max_abs'], float(np.max(np.abs(block-before))))
                    target = root/'scores'/f'{task["id"]}_{t}_{position}.json'
                    write(target, dict(view_ids=[v.uid for v in joined.views], scores=values.tolist()))
                    run['cells'][f'{t}_{position}'] = record(target)
                    del parts, vectors, values, matrix
                del model
                gc.collect()
                torch.cuda.empty_cache()
            result['runs'].append(run)
            write(root/'score_progress.json', dict(completed=len(result['runs']), total=6))
    validate_pool_result(result, batch)
    check()
    write(root/'blind_complete.json', result)


def query_metrics(matrix, positive, ids):
    import numpy as np
    n = len(ids)
    if (n != 128 or len(set(ids)) != n or matrix.shape != (n, n) or not np.isfinite(matrix).all()
            or len(positive) != n or any(p == i or positive[p] != i for i, p in enumerate(positive))):
        raise ValueError('Finite symmetric paired 128-view ranking required')
    ranks = np.asarray([next(k+1 for k, j in enumerate(sorted(
        (j for j in range(n) if j != i), key=lambda j: (-matrix[i, j], ids[j].encode('utf-8'))))
        if j == positive[i]) for i in range(n)])
    result = dict(map=(1/ranks).tolist(), mrr=(1/ranks).tolist())
    for k in (1, 3, 5, 10):
        result[f'recall@{k}'] = (ranks <= k).astype(float).tolist()
        result[f'ndcg@{k}'] = np.where(ranks <= k, 1/np.log2(ranks+1), 0.).tolist()
    return result


def analyze(gate, old, manifest, check):
    import numpy as np
    from step30_evolution_io import access, pinned, write
    from step30_evolution_policy import DOMAINS, METRICS, summaries, scenario
    root = Path(gate['output'])
    batch, _, _ = original_batch(gate, old)
    blind = read(root/'blind_complete.json')
    validate_pool_result(blind, batch)  # All 54 cells before any qrels access.
    if blind['blind_sources'] != [manifest['files'][d+':dev'] for d in DOMAINS] or blind['original_blind'] != gate['old_blind']:
        raise ValueError('Diagnostic score/data source mismatch')
    sources = [manifest['files'][d+':dev:qrels'] for d in DOMAINS]
    access(root, 'candidate127_dev_qrels', sources)
    positives = {}
    for domain, source in zip(DOMAINS, sources):
        truth = pinned(source)
        if set(truth) != set(blind['inventory'][domain]):
            raise ValueError('Original dev qrels groups differ')
        pairs = []
        for group, ids in blind['inventory'][domain].items():
            row = truth[group]
            if row['view_ids'] != ids or not set(row['labels']) <= {0, 1}:
                raise ValueError('Misaligned original dev qrels')
            table = vector_matrix(row['labels'], 16)
            if not (table.sum(1) == 1).all():
                raise ValueError('Exactly one positive per original view required')
            offset = len(pairs)
            pairs.extend(offset+int(j) for j in table.argmax(1))
        positives[domain] = pairs
    rows, old_rows = [], {r['spec']['id']: r for r in read(gate['old_metrics']['path'])}
    comparisons = []
    for run in blind['runs']:
        item = dict(spec=run['spec'], cells={})
        for key, source in run['cells'].items():
            check()
            _, position = map(int, key.split('_'))
            domain = DOMAINS[position-1]
            value = pinned(source)
            queries = query_metrics(vector_matrix(value['scores'], 128), positives[domain], value['view_ids'])
            means = {m: float(np.mean(queries[m])) for m in METRICS}
            item['cells'][key] = dict(mean=means, view_ids=value['view_ids'], queries=queries)
            comparisons.append(dict(task=run['spec']['id'], cell=key,
                                    original15=old_rows[run['spec']['id']]['cells'][key]['mean'], full127=means))
        rows.append(item)
    data = summaries(rows, batch['selected'], 'B')
    diagnostic = scenario(data)
    serial = {method: {k: v.tolist() for k, v in values.items()} for method, values in data.items()}
    report = dict(status='DIAGNOSTIC_COMPLETE', development_only=True, candidate_count=127,
                  trajectories=6, cells=54, new_training=0, test_access=False,
                  original_stop_unchanged=True, automatic_followup=False,
                  original_numeric_B_checks=diagnostic,
                  worth_discussing_new_protocol=diagnostic['passed'], summaries=serial,
                  comparisons=comparisons, confidence_intervals=None,
                  limitation='Development diagnostic only; no C/D authorization or confirmation claim')
    write(root/'metrics.json', rows)
    write(root/'result.json', report)


def supervise(gate_path, gate, old):
    import psutil
    from step28_formal_queue import GpuMemory
    from step30_evolution_control import release, resource_observation
    from step30_evolution_io import Budget, write
    from step30_evolution_policy import CAPS, LIMITS
    root = Path(gate['output'])
    root.mkdir(parents=True, exist_ok=False)  # A new formal invocation cannot replace this attempt.
    original = read(gate['original_budget']['path'])
    prior = original['used_seconds']
    delta = {'gpu': 0, 'auxiliary': PREPARATION_SECONDS+TAIL_SECONDS}
    events = [dict(operation='preparation_and_tail', bucket='auxiliary', charged_seconds=delta['auxiliary'],
                   conservative=True, covers='Connection/source/sync/package/return; not human work or pure waits')]
    def save_budget():
        total = {k: prior[k]+delta.get(k, 0) for k in CAPS}
        write(root/'budget.json', dict(prior=gate['original_budget'], used_seconds=total, delta_seconds=delta,
              caps=CAPS, delta_caps=DELTA_CAPS, cpu_test_seconds=original['cpu_test_seconds'], events=events))
        if any(total[k] > CAPS[k] for k in CAPS) or any(delta[k] > DELTA_CAPS[k] for k in delta):
            raise TimeoutError('Cumulative or diagnostic budget exhausted')
    save_budget()
    write(root/'claim.json', dict(gate=binding(gate_path), pid=os.getpid(), created=psutil.Process().create_time(),
                                owner=psutil.Process().username(), command=psutil.Process().cmdline(), started=time.time()))
    parent_active, ok = 0., False
    try:
        for action in ('score', 'analyze'):
            tick = time.monotonic()
            bucket, gpu = ('gpu', True) if action == 'score' else ('auxiliary', False)
            maximum = min(DELTA_CAPS[bucket]-delta[bucket], CAPS[bucket]-prior[bucket]-delta[bucket])
            if not gpu:
                maximum -= math.ceil(parent_active)+30
            if maximum <= 30 or delta['auxiliary']+math.ceil(parent_active)+30 >= DELTA_CAPS['auxiliary']:
                raise TimeoutError('Insufficient full worker or monitor budget')
            allowed = dict(old, other_gpu_commitments=gate['other_gpu_commitments'])
            observation = resource_observation(allowed, root, gpu)
            write(root/(action+'_admission.json'), observation)
            if not observation['ready']:
                raise RuntimeError('Resources unavailable; no automatic wait or retry')
            cores = psutil.Process().cpu_affinity()[-2:]
            if len(cores) != 2:
                raise RuntimeError('Two CPU cores required')
            argv = ['taskset', '-c', ','.join(map(str, cores)), PYTHON, '-B', '-u', str(Path(__file__).resolve()),
                    '--gate', str(gate_path), '--action', action, '--deadline', str(time.monotonic()+maximum)]
            environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='0' if gpu else '')
            p, nv, error, peak_gpu, peak_rss = None, None, None, 0, 0
            started = time.monotonic()
            try:
                with (root/(action+'_console.txt')).open('x') as output:
                    p = subprocess.Popen(argv, cwd=str(Path(__file__).resolve().parent), env=environment,
                                         stdout=output, stderr=subprocess.STDOUT, start_new_session=True, umask=0o077)
                proc = psutil.Process(p.pid)
                write(root/(action+'_process.json'), dict(pid=p.pid, owner=proc.username(), created=proc.create_time(),
                                                        command=argv, cwd=proc.cwd(), deadline=started+maximum))
                nv = GpuMemory() if gpu else None
                guard = Budget(ROOT, started+maximum)
                parent_active += time.monotonic()-tick
                while p.poll() is None:
                    tick = time.monotonic()
                    guard.check()
                    if delta['auxiliary']+math.ceil(parent_active)+30 >= DELTA_CAPS['auxiliary']:
                        raise TimeoutError('Auxiliary monitoring cap')
                    try:
                        rss = psutil.Process(p.pid).memory_info().rss
                        occupied = nv.snapshot()['processes'].get(p.pid, 0) if nv else 0
                        peak_gpu, peak_rss = max(peak_gpu, occupied), max(peak_rss, rss)
                        if occupied > LIMITS['process_gpu_bytes'] or rss > LIMITS['rss_bytes']:
                            raise RuntimeError('Worker 14-GiB GPU or 32-GiB RSS cap')
                    except psutil.NoSuchProcess:
                        pass
                    parent_active += time.monotonic()-tick
                    time.sleep(.2)
                if p.returncode:
                    raise RuntimeError(f'{action} exited {p.returncode}')
            except BaseException as exc:
                error = repr(exc)
                raise
            finally:
                tick = time.monotonic()
                cleanup_error = None
                try:
                    if p is not None:
                        release(p)
                except BaseException as exc:
                    error = error or repr(exc)
                    cleanup_error = repr(exc)
                try:
                    if nv:
                        nv.close()
                except BaseException as exc:
                    error = error or repr(exc)
                    cleanup_error = cleanup_error or repr(exc)
                elapsed = time.monotonic()-started
                if elapsed > maximum:
                    error = error or 'Full worker lifecycle cap exceeded'
                    cleanup_error = cleanup_error or error
                delta[bucket] += math.ceil(elapsed)
                events.append(dict(operation=action, bucket=bucket, actual_seconds=elapsed,
                                   charged_seconds=math.ceil(elapsed), success=error is None))
                write(root/(action+'_receipt.json'), dict(success=error is None, error=error, seconds=elapsed,
                      released=p is None or p.poll() is not None, peak_gpu_bytes=peak_gpu, peak_rss_bytes=peak_rss))
                parent_active += time.monotonic()-tick
                save_budget()
                if cleanup_error:
                    raise RuntimeError(cleanup_error)
            if action == 'score' and not (root/'blind_complete.json').is_file():
                raise ValueError('No complete scores; qrels remain closed')
            if action == 'analyze' and read(root/'result.json')['status'] != 'DIAGNOSTIC_COMPLETE':
                raise ValueError('Analysis did not complete')
        ok = True
    except BaseException as exc:
        write(root/'failure.json', dict(error=repr(exc), automatic_retry=False, original_stop_unchanged=True))
        raise
    finally:
        delta['auxiliary'] += math.ceil(parent_active)
        events.append(dict(operation='parent_active', bucket='auxiliary', actual_seconds=parent_active,
                           charged_seconds=math.ceil(parent_active), success=ok))
        try:
            save_budget()
        except BaseException as exc:
            ok = False
            write(root/'accounting_failure.json', dict(error=repr(exc)))
            raise
        finally:
            write(root/'completion.json', dict(status='DIAGNOSTIC_COMPLETE' if ok else 'INCOMPLETE',
                  success=ok, finished=time.time(), new_training=0, automatic_followup=False,
                  result=binding(root/'result.json') if ok else None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--action', choices=('run', 'score', 'analyze'), required=True)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    gate, old, manifest = permitted(args.gate)
    if args.action == 'run':
        supervise(args.gate, gate, old)
    else:
        from step30_evolution_io import Budget
        if args.deadline is None or not (Path(gate['output'])/'claim.json').is_file():
            raise ValueError('Worker requires the single-use parent claim/deadline')
        check = Budget(ROOT, args.deadline).check
        (score if args.action == 'score' else analyze)(gate, old, manifest, check)
        check(force=True)


if __name__ == '__main__':
    main()
