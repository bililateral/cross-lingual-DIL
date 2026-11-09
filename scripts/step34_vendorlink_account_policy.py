"""Frozen candidates, phases, and unrounded account-protocol decisions."""
from dataclasses import asdict, dataclass
import math
import statistics

from step29_vendorlink_methods import Settings

PROTOCOL = 'vendorlink-account-20261109'
DOMAINS = ('AlphaBay', 'Dream', 'Agora')
SEEDS = (43, 44, 45)
CAPS = dict(training=180000, gpu=21600, auxiliary=14400)
LIMITS = dict(workers=2, free_gpu_bytes=14*2**30, process_gpu_bytes=14*2**30,
              allocator_bytes=13*2**30, rss_bytes=32*2**30, free_ram_bytes=40*2**30,
              disk_bytes=32*2**30, free_disk_reserve=32*2**30, save_reserve_bytes=2*2**30)
CELLS = ((0, 1), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3))


@dataclass(frozen=True)
class AccountSettings(Settings):
    alpha: float = .07049361507739538
    beta: float = .767438161798796

    def __post_init__(self):
        super().__post_init__()
        if not all(math.isfinite(v) and v >= 0 for v in (self.alpha, self.beta)):
            raise ValueError('Finite nonnegative historical weights required')


def settings(candidate, method):
    changes = {}
    if candidate == 'H1':
        changes = dict(pooling='mean_last4_sum', encoder_lr=2.5e-5, head_lr=3e-4,
                       lr_multiplier=1., head_schedule='encoder')
    elif candidate == 'V1':
        changes = dict(vendor_lr=2e-5)
    elif candidate not in ('H0', 'V0'):
        raise ValueError('Unknown fixed candidate')
    if candidate.startswith('V') != (method == 'VendorLink_SEQ'):
        raise ValueError('Candidate/method family mismatch')
    return asdict(AccountSettings(method, **changes))


def tasks(phase, selection=None):
    if phase == 'D':
        return [dict(id='D_'+c, phase='D', candidate=c, method=m, seed=41, epochs=20, stages=1,
                     settings=settings(c, m)) for c, m in (
            ('H0', 'H_SEQ'), ('H1', 'H_SEQ'), ('V0', 'VendorLink_SEQ'), ('V1', 'VendorLink_SEQ'))]
    if phase not in ('P', 'M') or set(selection) != {'H', 'V'}:
        raise ValueError('Frozen D selection required')
    methods = ('H_SEQ', 'VendorLink_SEQ') if phase == 'P' else ('H_ER', 'H_full')
    return [dict(id=f'{phase}_{m}_s{s}', phase=phase, candidate=selection['V' if m.startswith('Vendor') else 'H'],
                 method=m, seed=s, epochs=40, stages=3,
                 settings=settings(selection['V' if m.startswith('Vendor') else 'H'], m))
            for s in SEEDS for m in methods]


def select(development):
    h = [development['D_H'+str(i)]['H_SEQ'][0]['R11'] for i in (0, 1)]
    v = [max(development['D_V'+str(i)][k][0]['R11'] for k in ('V_CLS', 'V_SET')) for i in (0, 1)]
    return dict(H='H'+str(int(h[1]-h[0] > .001)), V='V'+str(int(v[1]-v[0] > .001)))


def summarize(cells):
    r = dict(cells)
    r.update(Q=(r['R11']+r['R22'])/2, O=(r['R31']+r['R32'])/2,
             N=(r['R22']+r['R33'])/2, Z=r['R33'], GB=r['R22']-r['R12'], GC=r['R33']-r['R23'])
    r['F'] = r['Q']-r['O']
    return r


def decide(results, mechanism=False):
    required = ['H_SEQ', 'V_CLS', 'V_SET']+(['H_ER', 'H_full'] if mechanism else [])
    for m in required:
        if [r['seed'] for r in results[m]] != list(SEEDS):
            raise ValueError('Three fixed complete paired seeds required')
        for row in results[m]:
            if any(not math.isfinite(v) for k, v in row.items() if k != 'seed'):
                raise ValueError('Nonfinite result')
    def values(m, k): return [r[k] for r in results[m]]
    def mean(m, k): return statistics.mean(values(m, k))
    def diffs(a, b, k): return [x-y for x, y in zip(values(a, k), values(b, k))]
    def lower(a, b, keys, avg, individual):
        return all(statistics.mean(diffs(a, b, k)) >= avg and min(diffs(a, b, k)) >= individual for k in keys)
    def gain(m): return all(mean(m, k) >= .05 and min(values(m, k)) >= 0 for k in ('GB', 'GC'))
    learning = (all(mean('H_SEQ', k) >= .70 and min(values('H_SEQ', k)) >= .60
                   and mean('H_SEQ', k) >= max(mean('V_CLS', k), mean('V_SET', k))-.05
                   and max(mean('V_CLS', k), mean('V_SET', k)) >= .70
                   for k in ('R11', 'R22', 'R33')) and gain('H_SEQ'))
    forgetting = (mean('H_SEQ', 'F') >= .05 and sum(v >= .03 for v in values('H_SEQ', 'F')) >= 2
                  and min(values('H_SEQ', 'F')) >= 0)
    checks = dict(learning=learning, meaningful_forgetting=forgetting)
    if not mechanism:
        return dict(status='PASS_P' if learning and forgetting else
                    ('STOP_LEARNING' if not learning else 'STOP_NO_MEANINGFUL_FORGETTING'), checks=checks)
    def improvement(a, b, threshold, third_floor):
        old = diffs(a, b, 'O')
        forget = [-v for v in diffs(a, b, 'F')]
        return all(statistics.mean(v) >= threshold and sum(x > 0 for x in v) >= 2
                   and (third_floor is None or min(v) >= third_floor) for v in (old, forget))
    initial = lower('H_full', 'H_SEQ', ('R11', 'R22'), -.01, -.03)
    new = (all(mean('H_full', k) >= .70 for k in ('R22', 'R33')) and gain('H_full')
           and all(lower('H_full', m, ('R22', 'R33'), -.02, -.05) for m in ('H_SEQ', 'V_CLS', 'V_SET')))
    external = all(statistics.mean(diffs('H_full', m, 'O')) >= .02
                   and sum(v > 0 for v in diffs('H_full', m, 'O')) >= 2
                   and min(diffs('H_full', m, 'O')) >= -.01 for m in ('V_CLS', 'V_SET'))
    retention = improvement('H_full', 'H_SEQ', .03, -.01)
    checks.update(retention=retention, initial_protection=initial, new_protection=new, external=external)
    system = all(checks.values())
    extra = (system and improvement('H_full', 'H_ER', .02, None)
             and lower('H_full', 'H_ER', ('R11', 'R22'), -.01, -.03)
             and lower('H_full', 'H_ER', ('R22', 'R33'), -.02, -.05))
    return dict(status='PASS_SYSTEM' if system else 'STOP_NO_H_ADVANTAGE', checks=checks,
                old_score_increment=extra)
