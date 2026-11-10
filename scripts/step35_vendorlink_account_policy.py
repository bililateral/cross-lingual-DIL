"""Frozen candidates, phases, and unrounded account-protocol decisions."""
from dataclasses import asdict, dataclass
import math
import statistics

from step29_vendorlink_methods import Settings

PROTOCOL = 'vendorlink-account-ce-rms-20261110'
DOMAINS = ('AlphaBay', 'Dream', 'Agora')
SEEDS = (51, 52, 53)
CAPS = dict(training=6*3600, gpu=1800, auxiliary=1800)
LIMITS = dict(workers=2, free_gpu_bytes=5*2**30, process_gpu_bytes=None,
              allocator_bytes=None, vendor_reserve_bytes=17*2**30,
              rss_bytes=16*2**30, free_ram_bytes=20*2**30,
              disk_bytes=8*2**30, free_disk_reserve=32*2**30, save_reserve_bytes=2*2**30,
              dispatch_free_disk_bytes=64*2**30)
CELLS = ((0, 1), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3))


@dataclass(frozen=True)
class AccountSettings(Settings):
    alpha: float = .07049361507739538
    beta: float = .767438161798796
    gamma: float = 1.
    auxiliary_lr: float = 3e-4
    ce_scale_mode: str = 'unit'
    diagnostics: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not all(math.isfinite(v) and v >= 0 for v in (self.alpha, self.beta, self.auxiliary_lr)):
            raise ValueError('Finite nonnegative historical weights required')
        if self.gamma not in (0., 1.):
            raise ValueError('Only the frozen zero/one CE intervention is allowed')
        if self.ce_scale_mode not in ('unit', 'rms'):
            raise ValueError('Only unit or dimension-derived RMS input is defined')


def settings(candidate, method):
    changes = {}
    if candidate in ('H_base', 'H_CE', 'H_CE_RMS'):
        changes = dict(pooling='mean_last4_sum', encoder_lr=2.5e-5, head_lr=3e-4,
                       lr_multiplier=1., head_schedule='encoder', gamma=float(candidate != 'H_base'),
                       ce_scale_mode='rms' if candidate == 'H_CE_RMS' else 'unit',
                       diagnostics=candidate == 'H_CE_RMS')
    elif candidate == 'V1':
        changes = dict(vendor_lr=2e-5)
    elif candidate != 'V0':
        raise ValueError('Unknown fixed candidate')
    if candidate.startswith('V') != (method == 'VendorLink_SEQ'):
        raise ValueError('Candidate/method family mismatch')
    return asdict(AccountSettings(method, **changes))


def tasks(phase, selection=None):
    if phase == 'D':
        return [dict(id='D_'+d+'_H_CE_RMS', phase='D', candidate='H_CE_RMS', method='H_SEQ',
                     seed=41, epochs=40, stages=1, domains=[d], settings=settings('H_CE_RMS', 'H_SEQ'))
                for d in DOMAINS[:2]]
    raise ValueError('This finite RMS campaign permits only the two D tasks')


def select(development, baseline):
    def score(source, domain, candidate, reading):
        row, = source['D_'+domain+'_'+candidate][reading]
        if row['seed'] != 41 or not math.isfinite(row['R11']):
            raise ValueError('Complete finite fixed D endpoints required')
        return row['R11']
    for d in DOMAINS[:2]:
        for reading in ('H_SEQ', 'H_AUX', 'H_SET'):
            score(development, d, 'H_CE_RMS', reading)
    selected = dict(H='H_CE_RMS', V='V1')
    checks = {d: dict(H=score(development, d, 'H_CE_RMS', 'H_SEQ'),
                      V=score(baseline, d, 'V1', 'V_CLS'),
                      RMS_minus_CE=score(development, d, 'H_CE_RMS', 'H_SEQ')-score(baseline, d, 'H_CE', 'H_SEQ'),
                      RMS_minus_base=score(development, d, 'H_CE_RMS', 'H_SEQ')-score(baseline, d, 'H_base', 'H_SEQ'))
              for d in DOMAINS[:2]}
    passed = all(r['H'] >= .70 and r['H']-r['V'] >= -.02 for r in checks.values())
    return dict(status='PASS_D' if passed else 'STOP_ADAPTATION', selected=selected, checks=checks)


def task_domains(task):
    domains = tuple(task['domains'])
    if (len(domains) != task['stages'] or
            (task['phase'] == 'D' and (len(domains) != 1 or domains[0] not in DOMAINS[:2])) or
            (task['phase'] in ('P', 'M') and domains != DOMAINS)):
        raise ValueError('Actual task domains do not match the frozen stage')
    return domains


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
                   and lower('H_SEQ', 'V_CLS', (k,), -.02, -.05)
                   and mean('V_CLS', k) >= .70
                   for k in ('R11', 'R22', 'R33')) and gain('H_SEQ'))
    forgetting = (mean('H_SEQ', 'F') >= .035 and sum(v >= .03 for v in values('H_SEQ', 'F')) >= 2
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
    initial = (lower('H_full', 'H_SEQ', ('R11', 'R22'), -.01, -.03)
               and all(mean('H_full', k) >= .70 and min(values('H_full', k)) >= .60
                       for k in ('R11', 'R22', 'R33'))
               and lower('H_full', 'V_CLS', ('R11', 'R22', 'R33'), -.02, -.05))
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
