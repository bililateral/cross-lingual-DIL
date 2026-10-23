"""Independent external review of saved summaries; never imports project code.

Only saved metric matrices/query-level summaries/count metadata are read.
No supervision originals, models, Memory payloads, network or training.
Usage: python audit_statistics.py INPUT_ROOT OUTPUT_DIRECTORY
"""
from pathlib import Path
import argparse, csv, hashlib, json, platform, sys, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_root', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    root, out = args.input_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    reads = {}

    def record(path, method):
        path = path.resolve()
        reads[str(path.relative_to(root))] = dict(
            path=str(path.relative_to(root)), method=method,
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def readj(path):
        record(path, 'machine_parse_saved_json')
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def readn(path):
        record(path, 'machine_read_saved_query_summary_or_degree')
        return np.load(path, allow_pickle=False)

    joint = readj(root/'result/job/joint/evaluation.json')
    arms = ('S', 'S_pos', 'H_pos')
    roles = ('raw', 'stage-cal', 'first-cal')
    arrays, query_map, degree = {}, {}, {}
    columns = population = None
    for arm in arms:
        for seed in (1, 2, 3):
            run = root/'result/job'/f'replication_{seed-1:02d}_{arm}_s{seed}'/'evaluation'
            col = readj(run/'collected.json')
            current = (col['columns'], col['group_ids'], col['domains'])
            if population is None:
                population = current
                columns = col['columns']
            assert current == population, 'Group/metric alignment mismatch'
            degree[arm, seed] = readn(run/col['query_positive_counts']['path'])
            assert degree[arm, seed].shape == (40, 28)
            assert np.array_equal(degree[arm, seed], degree['S', 1])
            assert np.all((degree[arm, seed] == 1).sum(axis=1) == 16)
            assert np.all((degree[arm, seed] == 2).sum(axis=1) == 12)
            for point, role_records in col['points'].items():
                phase = point.split('_')[0]
                for role in roles:
                    rec = role_records[role]
                    data = readj(run/rec['matrix']['path'])
                    assert data['columns'] == columns
                    arrays[arm, seed, phase, role] = np.asarray(data['values'], dtype=np.float64)
                    assert arrays[arm, seed, phase, role].shape == (40, 22)
                    assert not np.isinf(arrays[arm, seed, phase, role]).any()
                query_map[arm, seed, phase] = readn(run/role_records['raw']['queries']['path'])[:, :, 0]

    domains = np.asarray(population[2])
    group_rows = {d: np.flatnonzero(domains == d) for d in 'AB'}
    assert all(len(v) == 20 for v in group_rows.values())
    # Frozen resampling specification, not a new choice of statistical method.
    draws = np.random.Generator(np.random.PCG64(20260930)).integers(0, 20, size=(5000, 2, 20))
    signs = np.asarray([-1 if c in ('brier', 'log_loss') else 1 for c in columns])
    checks, effects = [], {}
    mismatch = []

    def check(values, saved, domain, label, metric_names=None):
        v = np.asarray(values, dtype=np.float64)
        if v.ndim == 1:
            v = v[:, None]
        names = metric_names or columns
        d = 'AB'.index(domain)
        a = v[group_rows[domain], :]
        means = a.mean(axis=0)
        # Direct paired resampling (independent of project's summary functions).
        boot = a[draws[:, d, :], :].mean(axis=1)
        limits = np.quantile(boot, [0.025, 0.975], axis=0, method='linear')
        for j, metric in enumerate(names):
            t = saved[metric]
            undefined = bool(np.isnan(a[:, j]).any())
            if undefined:
                ok = t['mean'] is None and t['conditional_95pct_interval'] is None
                expected = [None, None, None]
                error = 0.0 if ok else None
            else:
                expected = [float(means[j]), float(limits[0, j]), float(limits[1, j])]
                actual = [t['mean'], *(t['conditional_95pct_interval'] or [None, None])]
                error = max(abs(x-y) for x,y in zip(expected,actual)) if all(x is not None for x in actual) else None
                ok = error is not None and error <= 1e-12
            checks.append(dict(label=label, domain=domain, metric=metric,
                               mean=expected[0], low=expected[1], high=expected[2],
                               undefined=undefined, maximum_absolute_error=error, passed=ok))
            if not ok:
                mismatch.append(checks[-1])
        return {metric:dict(mean=checks[-len(names)+j]['mean'],
                           low=checks[-len(names)+j]['low'], high=checks[-len(names)+j]['high'])
                for j,metric in enumerate(names)}

    for arm in arms:
        for phase in 'AB':
            for role in roles:
                values = np.stack([arrays[arm,s,phase,role] for s in (1,2,3)]).mean(axis=0)
                for domain in 'AB':
                    check(values, joint['three_seed_absolute'][arm][phase][role][domain], domain,
                          f'absolute/{arm}/mean3/{phase}/{role}')

    for arm, opponent in (('S_pos', 'S'), ('H_pos', 'S'), ('H_pos', 'S_pos')):
        name = f'{arm}-{opponent}'
        effects[name] = {}
        for seedset in ((1,), (2,), (3,), (1,2,3)):
            sname = f's{seedset[0]}' if len(seedset) == 1 else 'mean3'
            saved = (joint['comparisons'][name]['by_seed'][str(seedset[0])] if len(seedset) == 1
                     else joint['comparisons'][name]['three_seed_mean'])
            for phase in 'AB':
                for role in roles:
                    delta = np.mean([arrays[arm,s,phase,role]-arrays[opponent,s,phase,role] for s in seedset],axis=0)
                    for domain in 'AB':
                        results = check(delta, saved[phase][role][domain], domain, f'{name}/{sname}/{phase}/{role}')
                        if len(seedset) == 3:
                            effects[name][f'{phase}/{role}/{domain}'] = results
            for role in roles:
                delta = np.mean([(arrays[arm,s,'A',role]-arrays[arm,s,'B',role]-
                                  arrays[opponent,s,'A',role]+arrays[opponent,s,'B',role])*signs for s in seedset],axis=0)
                results = check(delta, saved['forgetting'][role], 'A', f'{name}/{sname}/forgetting/{role}')
                if len(seedset) == 3:
                    effects[name][f'forgetting/{role}/A'] = results

        for positives in (1,2):
            by_seed = []
            for seed in (1,2,3):
                mask = degree[arm,seed] == positives
                candidate = np.sum(np.where(mask, query_map[arm,seed,'B'],0.0),axis=1)/np.sum(mask,axis=1)
                comparator = np.sum(np.where(mask, query_map[opponent,seed,'B'],0.0),axis=1)/np.sum(mask,axis=1)
                by_seed.append(candidate-comparator)
                for domain in 'AB':
                    target = joint['query_subgroup_MAP_comparisons'][name][str(positives)]['by_seed'][str(seed)][domain]
                    check(by_seed[-1], {'map':target}, domain, f'{name}/s{seed}/positive{positives}', ['map'])
            for domain in 'AB':
                target = joint['query_subgroup_MAP_comparisons'][name][str(positives)]['three_seed_mean'][domain]
                result = check(np.mean(by_seed,axis=0), {'map':target}, domain, f'{name}/mean3/positive{positives}', ['map'])
                effects[name][f'positive{positives}/{domain}'] = result

    k = columns.index('map')
    principal = []
    gates = {}
    decomposition = {}
    for arm in arms:
        for seed in (1,2,3):
            values = dict(arm=arm, seed=seed)
            for phase, domain in (('A','A'), ('B','A'), ('B','B')):
                values[f'{phase}_end_{domain}_map'] = float(arrays[arm,seed,phase,'raw'][group_rows[domain],k].mean())
            values['forgetting_A'] = values['A_end_A_map']-values['B_end_A_map']
            principal.append(values)
    for arm in ('S_pos','H_pos'):
        vals = []
        for seed in (1,2,3):
            for domain in 'AB':
                delta = float((arrays[arm,seed,'B','raw']-arrays['S',seed,'B','raw'])[group_rows[domain],k].mean())
                vals.append(dict(seed=seed,domain=domain,difference=delta,strictly_positive=delta>0))
        gates[arm] = dict(cells=vals, positive_cells=sum(x['strictly_positive'] for x in vals),passed=all(x['strictly_positive'] for x in vals))
        assert gates[arm]['passed'] == joint['development_direction_consistency'][arm]
        end = effects[f'{arm}-S']['B/raw/A']['map']['mean']
        initial = effects[f'{arm}-S']['A/raw/A']['map']['mean']
        forgetting = effects[f'{arm}-S']['forgetting/raw/A']['map']['mean']
        decomposition[arm] = dict(final_old_A_advantage=end,advantage_already_at_A_end=initial,
                                  less_forgetting_point_estimate=-forgetting,
                                  identity_residual=end-(initial-forgetting))
        assert abs(decomposition[arm]['identity_residual']) <= 1e-12

    # Cross-check the machine-readable supplementary export against the saved joint summary.
    supplement = readj(root/'result/analysis/summary.json')
    copied = [('three_seed_absolute','three_seed_absolute'),('comparisons','comparisons'),
              ('positive_count','query_subgroup_MAP_comparisons'),('diagnostics','diagnostics'),
              ('direction_gate','development_direction_consistency')]
    export_matches = {a:supplement[a] == joint[b] for a,b in copied}
    assert all(export_matches.values())
    numerical = [r['maximum_absolute_error'] for r in checks if r['maximum_absolute_error'] is not None]
    summary = dict(status='PASS' if not mismatch else 'FAIL', environment='external isolated review workspace',
                   python=platform.python_version(), numpy=np.__version__,
                   input_archive_sha256='be13255ad50a0fe814f58a150574f8403181cf3a9e7aa0af37293834188baac6',
                   checked_summary_rows=len(checks),undefined_rows=sum(x['undefined'] for x in checks),
                   compared_finite_scalars=3*sum(not x['undefined'] for x in checks),
                   maximum_absolute_error=max(numerical), mismatches=mismatch,
                   bootstrap=dict(algorithm='PCG64',seed=20260930,replicates=5000,domains=2,groups_each=20,
                                  draw_shape=list(draws.shape),draw_dtype=str(draws.dtype),
                                  draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),
                                  method='whole groups paired across arms and seeds; average three fixed seeds before resampling'),
                   export_matches=export_matches, gates=gates, decomposition=decomposition,
                   formal_label_reads=0,model_loads=0,training_calls=0,project_server_connections=0,
                   elapsed_seconds=time.perf_counter()-start,
                   limits='Saved-metric arithmetic and conditional bootstrap only; not an independent label-level reconstruction or seed-population/holdout inference.')

    def savej(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def savecsv(name,rows):
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    savej('statistics_audit.json',summary)
    savej('comparison_effects.json',effects)
    savecsv('statistics_checked.csv',checks)
    savecsv('replication_map_independent.csv',principal)
    savecsv('statistics_read_scope.csv',list(reads.values()))
    print(json.dumps({k:summary[k] for k in ('status','checked_summary_rows','undefined_rows','compared_finite_scalars','maximum_absolute_error','gates','decomposition','elapsed_seconds')},ensure_ascii=False,indent=2))
    if mismatch:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
