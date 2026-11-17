"""Fixed Evolution pilot choices and decisions. No data/model IO on import."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import numpy as np

DOMAINS = ('2014-Q3', '2014-Q4', '2015-Q1')
SPLITS = {'fit': 160, 'dev': 64, 'test': 96}
CELLS = ((0, 1), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3))
METRICS = ('map', 'mrr', 'recall@1', 'recall@3', 'recall@5', 'recall@10',
           'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10')
SEEDS = {'A': (11, 12), 'B': (13, 14, 15), 'C': (13, 14, 15), 'D': (21, 22, 23)}
CAPS = {'training': 144000, 'gpu': 14400, 'auxiliary': 14400}
LIMITS = {'workers': 2, 'epochs': 40, 'rss_bytes': 32*2**30,
          'free_ram_bytes': 40*2**30, 'free_gpu_bytes': 14*2**30,
          'process_gpu_bytes': 14*2**30, 'allocator_bytes': 13*2**30,
          'disk_bytes': 64*2**30, 'free_disk_start': 128*2**30,
          'free_disk_reserve': 64*2**30, 'cpu_test_seconds': 1200}
SYSTEMS = ('H0_SEQ', 'H_SEQ', 'H_ER', 'H_full', 'V_SEQ', 'V_ER')
REFERENCES = ('H_SEQ', 'H_ER', 'V_SEQ', 'V_ER')
H_RECIPES = {
    'H0': dict(pooling='cls', encoder_lr=1.2832655111666686e-5,
               head_lr=.0012832655111666686, head_schedule='constant'),
    'H1': dict(pooling='mean_last4_sum', encoder_lr=1.2832655111666686e-5,
               head_lr=.0012832655111666686, head_schedule='constant'),
    'H2': dict(pooling='cls', encoder_lr=2.5e-5, head_lr=3e-4, head_schedule='encoder'),
    'H3': dict(pooling='mean_last4_sum', encoder_lr=2.5e-5, head_lr=3e-4, head_schedule='encoder')}
V_RECIPES = {'V0': dict(vendor_lr=4e-5, vendor_weight_decay=0.),
             'V1': dict(vendor_lr=2e-5, vendor_weight_decay=0.),
             'V2': dict(vendor_lr=4e-5, vendor_weight_decay=.01),
             'V3': dict(vendor_lr=2e-5, vendor_weight_decay=.01)}


def settings(system: str, recipe: str) -> dict:
    # Old Settings uses VendorLink_SEQ as the encoder/classifier FAMILY. The
    # Evolution system (including V_ER) remains explicit in spec and dispatch.
    from step29_vendorlink_methods import Settings
    family = 'VendorLink_SEQ' if system.startswith('V_') else ('H_SEQ' if system == 'H0_SEQ' else system)
    recipes = V_RECIPES if system.startswith('V_') else H_RECIPES
    return asdict(Settings(method=family, lr_multiplier=1., **recipes[recipe]))


def spec(phase: str, system: str, recipe: str, seed: int) -> dict:
    if phase not in SEEDS or seed not in SEEDS[phase] or system not in SYSTEMS:
        raise ValueError('Unapproved phase, system or seed')
    return dict(id=f'{phase}_{system}_{recipe}_s{seed}', phase=phase, system=system,
                recipe=recipe, seed=seed, stages=1 if phase == 'A' else 3,
                epochs=40, settings=settings(system, recipe))


def tasks(phase: str, selected: dict | None = None) -> list[dict]:
    if phase == 'A':
        return [spec('A', system, recipe, seed) for system, recipes in
                (('H_SEQ', H_RECIPES), ('V_SEQ', V_RECIPES))
                for recipe in recipes for seed in SEEDS['A']]
    if not selected or selected.get('H') not in H_RECIPES or selected.get('V') not in V_RECIPES:
        raise ValueError('Qualified stage A choices required')
    systems = ('H_SEQ', 'V_SEQ') if phase == 'B' else (
        ('H0_SEQ', 'H_ER', 'H_full', 'V_ER') if phase == 'C' else SYSTEMS)
    return [spec(phase, m, 'H0' if m == 'H0_SEQ' else selected[m[0]], seed)
            for m in systems if not (m == 'H0_SEQ' and selected['H'] == 'H0')
            for seed in SEEDS[phase]]


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def cells(stages: int) -> tuple:
    if stages not in (1, 3):
        raise ValueError('Only single-domain A or complete three-domain trajectories')
    return ((0, 1), (1, 1)) if stages == 1 else CELLS


def select_a(rows: list[dict]) -> dict:
    expected = tasks('A')
    if {r['spec']['id'] for r in rows} != {s['id'] for s in expected} or len(rows) != 16:
        raise ValueError('All sixteen A tasks required before selection')
    by_id = {r['spec']['id']: r for r in rows}
    for s in expected:
        if by_id[s['id']]['spec'] != s:
            raise ValueError('A source/config/seed mismatch')
    chosen, candidates = {}, {}
    for family, recipes in (('H', H_RECIPES), ('V', V_RECIPES)):
        candidates[family] = {}
        for recipe in recipes:
            r = [by_id[spec('A', family+'_SEQ', recipe, s)['id']]['cells']['1_1']['mean'] for s in SEEDS['A']]
            maps = np.array([x['map'] for x in r])
            recall = np.array([x['recall@1'] for x in r])
            if not np.isfinite(np.r_[maps, recall]).all():
                raise ValueError('Nonfinite A metrics')
            candidates[family][recipe] = dict(map=float(maps.mean()), recall1=float(recall.mean()),
                seed_map=maps.tolist(), eligible=bool(maps.mean() >= .50 and recall.mean() >= .35 and (maps >= .45).all()))
        eligible = {k: v for k, v in candidates[family].items() if v['eligible']}
        if eligible:
            best = max(v['map'] for v in eligible.values())
            chosen[family] = next(k for k in recipes if k in eligible and eligible[k]['map'] >= best-.005)
    benefit = False
    if 'H' in chosen:
        a = candidates['H'][chosen['H']]
        b = candidates['H']['H0']
        benefit = bool(a['map']-b['map'] >= .01 and (np.array(a['seed_map']) > b['seed_map']).all())
    return dict(passed=len(chosen) == 2, selected=chosen, candidates=candidates,
                adaptation_benefit_on_development=benefit)


def summaries(rows: list[dict], selected: dict, phase: str) -> dict:
    expected = tasks('D', selected) if phase == 'D' else tasks('B', selected)+tasks('C', selected)
    if phase == 'B':
        expected = tasks('B', selected)
    by_id = {r['spec']['id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {s['id'] for s in expected}:
        raise ValueError('Missing, extra or duplicate trajectory; never drop a seed')
    for s in expected:
        if by_id[s['id']]['spec'] != s or set(by_id[s['id']]['cells']) != {f'{t}_{d}' for t, d in CELLS}:
            raise ValueError('Configuration or nine-cell identity differs')
    result = {}
    for system in sorted({s['system'] for s in expected}):
        subset = sorted((r for r in rows if r['spec']['system'] == system), key=lambda r: r['spec']['seed'])
        def values(t, d, metric='map'):
            a = np.array([r['cells'][f'{t}_{d}']['mean'][metric] for r in subset], dtype=float)
            if a.shape != (3,) or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
                raise ValueError('Three finite bounded seed metrics required')
            return a
        initial = np.stack([values(d, d) for d in (1, 2, 3)], axis=1)
        pre = np.stack([values(d-1, d) for d in (1, 2, 3)], axis=1)
        final = np.stack([values(3, d) for d in (1, 2, 3)], axis=1)
        q, o = initial[:, :2].mean(1), final[:, :2].mean(1)
        result[system] = dict(P=pre, A=initial, G=initial-pre, final=final, Q=q, O=o, F=q-o,
                              F_domain=initial[:, :2]-final[:, :2], N=final[:, 2],
                              recall=np.stack([values(d, d, 'recall@1') for d in (1, 2, 3)], axis=1))
    return result


def learning_paths(full: dict, ref: dict) -> dict:
    original = full['G'].mean(0)-ref['G'].mean(0) >= -.02
    high = (full['P'].mean(0) > ref['P'].mean(0)) & (full['A'].mean(0) >= ref['A'].mean(0))
    own = (full['G'].mean(0) >= .01) & (full['G'] >= 0).all(0)
    return dict(passed=bool((own & (original | high)).all()),
                original=original.tolist(), high_start=high.tolist(), own_learning=own.tolist(),
                full={k: full[k].tolist() for k in ('P', 'A', 'G')},
                reference={k: ref[k].tolist() for k in ('P', 'A', 'G')})


def ability(x: dict) -> bool:
    return bool((x['A'].mean(0) >= .5).all() and (x['A'] >= .45).all() and
                (x['recall'].mean(0) >= .35).all())


def scenario(data: dict) -> dict:
    details = {}
    for name in ('H_SEQ', 'V_SEQ'):
        x = data[name]
        checks = dict(ability=ability(x), net_learning=bool((x['G'].mean(0) >= .03).all()),
                      mean_forgetting=bool(x['F'].mean() >= .05),
                      each_old_domain=bool((x['F_domain'].mean(0) >= .03).all()),
                      three_positive=bool((x['F'] > 0).all()), two_practical=bool((x['F'] >= .04).sum() >= 2))
        details[name] = dict(passed=all(checks.values()), checks=checks)
    return dict(passed=all(v['passed'] for v in details.values()), baselines=details)


def mechanism(data: dict) -> dict:
    f, comparisons = data['H_full'], {}
    absolute = ability(f) and bool((f['G'].mean(0) >= .01).all() and (f['G'] >= 0).all())
    for name in REFERENCES:
        b = data[name]
        df, do = b['F']-f['F'], f['O']-b['O']
        learning = learning_paths(f, b)
        checks = dict(forgetting=bool(df.mean() >= (max(.03, .4*b['F'].mean()) if name == 'H_SEQ' else .01)),
                      old_final=bool(do.mean() >= (.02 if name == 'H_SEQ' else .01)),
                      all_seed_directions=bool((df > 0).all() and (do > 0).all()),
                      initial_mean=bool((f['A'].mean(0)-b['A'].mean(0) >= -.01).all()),
                      initial_seed=bool((f['A']-b['A'] >= -.02).all()),
                      final_new_mean=bool(f['N'].mean()-b['N'].mean() >= -.01),
                      final_new_seed=bool((f['N']-b['N'] >= -.02).all()),
                      final_old_seed=bool((f['final'][:, :2]-b['final'][:, :2] >= -.02).all()),
                      learning=learning['passed'])
        if name == 'H_SEQ':
            checks['each_old_mean_gain'] = bool((f['final'][:, :2].mean(0)-b['final'][:, :2].mean(0) >= .01).all())
        comparisons[name] = dict(passed=absolute and all(checks.values()), checks=checks, learning=learning,
                                 delta_F=df.tolist(), delta_O=do.tolist())
    return dict(passed=absolute and all(x['passed'] for x in comparisons.values()),
                full_absolute=absolute, comparisons=comparisons)


def bootstrap(rows: list[dict], selected: dict, check=lambda: None) -> dict:
    """Paired domain-group resampling, common across fixed seeds and states."""
    summaries(rows, selected, 'D')  # Exact methods/configs/seeds/cells first.
    arrays, group_ids = {}, {}
    for row in rows:
        m, s = row['spec']['system'], row['spec']['seed']
        for t, d in CELLS:
            groups = row['cells'][f'{t}_{d}']['groups']
            ids = [g['uid'] for g in groups]
            if len(ids) != 12 or len(set(ids)) != 12 or (d in group_ids and group_ids[d] != ids):
                raise ValueError('Twelve identically ordered candidate groups per test domain required')
            group_ids[d] = ids
            values = np.array([g['queries']['map'] for g in groups])
            if values.shape != (12, 16) or not np.isfinite(values).all():
                raise ValueError('Saved complete query metrics required')
            arrays[m, s, t, d] = values.mean(1)
    rng = np.random.Generator(np.random.PCG64(2026110702))
    draws = {m: {'F': [], 'O': []} for m in REFERENCES}
    for iteration in range(5000):
        if iteration % 100 == 0:
            check()
        sampled = {d: rng.integers(0, 12, size=12) for d in (1, 2, 3)}
        def endpoint(m, t, d):
            return np.mean([arrays[m, s, t, d][sampled[d]].mean() for s in SEEDS['D']])
        def qo(m):
            return ((endpoint(m, 1, 1)+endpoint(m, 2, 2))/2,
                    (endpoint(m, 3, 1)+endpoint(m, 3, 2))/2)
        qf, of = qo('H_full')
        for m in REFERENCES:
            qb, ob = qo(m)
            draws[m]['F'].append((qb-ob)-(qf-of))
            draws[m]['O'].append(of-ob)
    return {m: {k: np.quantile(v, [.025, .975], method='linear').tolist() for k, v in ds.items()}
            for m, ds in draws.items()}


def decision(rows: list[dict], selected: dict, phase: str, check=lambda: None) -> dict:
    data = summaries(rows, selected, phase)
    b = scenario(data)
    if phase == 'B':
        return b
    c = mechanism(data)
    contributions = {}
    for label, before, after in (('english_recipe', 'H0_SEQ', 'H_SEQ'),
                                  ('replay', 'H_SEQ', 'H_ER'), ('birth_preservation', 'H_ER', 'H_full')):
        old = data.get(before, data['H_SEQ'])  # Exact H0 deduplication, not a new result.
        new = data[after]
        contributions[label] = dict(before=before, after=after,
            delta={k: (new[k]-old[k]).tolist() for k in ('P', 'A', 'G', 'Q', 'O', 'N')},
            forgetting_reduction=(old['F']-new['F']).tolist(),
            same_configuration_reused=label == 'english_recipe' and selected['H'] == 'H0')
    out = dict(passed=b['passed'] and c['passed'], scenario=b, mechanism=c,
               contributions=contributions,
               endpoints={m: {k: v.tolist() for k, v in x.items()} for m, x in data.items()})
    if phase == 'D':
        ci = bootstrap(rows, selected, check)
        support = {m: bool(b['passed'] and c['comparisons'][m]['passed'] and
                           all(ci[m][key][0] > 0 for key in ('F', 'O'))) for m in REFERENCES}
        out.update(intervals=ci, support=support,
                   internal_support=support['H_SEQ'] and support['H_ER'],
                   passed=all(support.values()))
    return out
