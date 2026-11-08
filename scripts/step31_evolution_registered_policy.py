"""One registered-account premise experiment; no automatic mechanism phase."""
from __future__ import annotations

import numpy as np
from step30_evolution_policy import settings, LIMITS as PILOT_LIMITS, METRICS, identity

PROTOCOL = 'evolution-registered-premise-v1'
DOMAINS = ('E1', 'E2', 'E3')
SEEDS = (31, 32, 33)
CAPS = dict(training=10800, gpu=3600, auxiliary=3600)
LIMITS = {**PILOT_LIMITS, 'cpu_test_seconds': 180}
CACHE_SHA = '253f171e4ff387bc970a7793e7a2c4a0d14515580c89f028c276e891e755817c'
BASE_CELLS = ((0, 1, 1), (1, 1, 1), (1, 2, 2), (2, 1, 1), (2, 2, 2),
              (2, 3, 3), (3, 1, 1), (3, 2, 2), (3, 3, 3))
GROWTH_CELLS = ((2, 1, 2), (3, 1, 3), (3, 2, 3))
FROZEN_CELLS = ((1, 1, 3), (1, 2, 3), (1, 3, 3))
CELLS = BASE_CELLS + GROWTH_CELLS + FROZEN_CELLS


def tasks():
    return [dict(id=f'P_{system}_{recipe}_s{seed}', phase='P', system=system, recipe=recipe,
                 seed=seed, stages=3, epochs=40, settings=settings(system, recipe))
            for seed in SEEDS for system, recipe in (('H_SEQ', 'H1'), ('V_SEQ', 'V1'))]


def cell_key(t, d, pool):
    return f'{t}_{d}_{pool}'


def decide(rows):
    expected = tasks()
    if len(rows) != 6 or {r['task']['id'] for r in rows} != {t['id'] for t in expected}:
        raise ValueError('All six fixed trajectories required')
    by_id = {r['task']['id']: r for r in rows}
    if any(by_id[t['id']]['task'] != t for t in expected):
        raise ValueError('Task identity differs')
    systems = {}
    for system in ('H_SEQ', 'V_SEQ'):
        ordered = [by_id[t['id']] for t in expected if t['system'] == system]
        def metric(row, t, d, pool, name='map'):
            return row['cells'][cell_key(t, d, pool)]['mean'][name]
        q = np.array([[metric(r, d, d, d) for d in (1, 2, 3)] for r in ordered])
        r1 = np.array([[metric(r, d, d, d, 'recall@1') for d in (1, 2, 3)] for r in ordered])
        g = q - np.array([[metric(r, d-1, d, d) for d in (1, 2, 3)] for r in ordered])
        o = np.array([[metric(r, 3, d, d) for d in (1, 2)] for r in ordered])
        f = q[:, :2] - o
        per_seed_f = f.mean(axis=1)
        gates = dict(initial_mean_map=bool((q.mean(0) >= .60).all()),
                     initial_mean_r1=bool((r1.mean(0) >= .45).all()),
                     initial_each_map=bool((q >= .55).all()),
                     learning_mean=bool((g.mean(0) >= .01).all()),
                     learning_each=bool((g >= 0).all()),
                     forgetting_mean=bool(f.mean() >= .05),
                     forgetting_each_batch=bool((f.mean(0) >= .03).all()),
                     forgetting_each_seed=bool((per_seed_f > 0).all()),
                     forgetting_two_seeds=bool((per_seed_f >= .04).sum() >= 2))
        frozen_birth = np.array([[metric(r, d, d, d)-metric(r, 1, d, d)
                                  for d in (2, 3)] for r in ordered])
        frozen_final = np.array([[metric(r, 3, d, 3)-metric(r, 1, d, 3)
                                  for d in (1, 2, 3)] for r in ordered])
        systems[system] = dict(seeds=list(SEEDS), Q=q.tolist(), O=o.tolist(), F=f.tolist(), G=g.tolist(),
            gates=gates, learning_passed=all(v for k, v in gates.items() if not k.startswith('forgetting')),
            forgetting_passed=all(v for k, v in gates.items() if k.startswith('forgetting')),
            vs_frozen_at_birth=frozen_birth.tolist(), vs_frozen_at_birth_mean=frozen_birth.mean(0).tolist(),
            vs_frozen_final_growth=frozen_final.tolist(), vs_frozen_final_growth_mean=frozen_final.mean(0).tolist())
    learned = all(v['learning_passed'] for v in systems.values())
    forgotten = all(v['forgetting_passed'] for v in systems.values())
    return dict(status=('STOP_LEARNING' if not learned else 'STOP_NO_JOINT_FORGETTING'
                        if not forgotten else 'PREMISE_MET_AWAIT_USER_MECHANISM_DECISION'),
        systems=systems, premise_passed=learned and forgotten, automatic_next_phase=False,
        frozen_comparison='Descriptive paired seed means and full cells; no invented value PASS or automatic M',
        interpretation='Fixed development protocol only; no independent confirmation or Full advantage tested')
