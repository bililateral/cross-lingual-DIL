"""Frozen candidates, phases, and unrounded account-protocol decisions."""
from dataclasses import asdict, dataclass
import math
import statistics

from step29_vendorlink_methods import Settings

PROTOCOL = 'vendorlink-account-fit12-mechanism-20261111'
DETERMINISTIC_PROTOCOL = 'vendorlink-fit12-deterministic-history-20261112'
BETA_PROTOCOL = 'vendorlink-fit12-beta-quarter-20261113'
BETA = .191859540449699
DOMAINS = ('AlphaBay', 'Dream', 'Agora')
SEEDS = (51, 52, 53)
CAPS = dict(training=80*3600, gpu=4*3600, auxiliary=4*3600)
LIMITS = dict(workers=2, free_gpu_bytes=5*2**30, process_gpu_bytes=None,
              allocator_bytes=None, vendor_reserve_bytes=17*2**30,
              rss_bytes=16*2**30, free_ram_bytes=20*2**30,
              disk_bytes=32*2**30, free_disk_reserve=32*2**30, save_reserve_bytes=2*2**30,
              dispatch_free_disk_bytes=64*2**30)
CELLS = ((0, 1), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3))
DETERMINISTIC_CAPS = dict(training=40*3600, gpu=3600, auxiliary=3600)
DETERMINISTIC_LIMITS = dict(LIMITS, disk_bytes=12*2**30)
BETA_LIMITS = dict(DETERMINISTIC_LIMITS, workers=3)


@dataclass(frozen=True)
class AccountSettings(Settings):
    alpha: float = .07049361507739538
    beta: float = .767438161798796
    gamma: float = 1.
    auxiliary_lr: float = 3e-4
    ce_scale_mode: str = 'unit'
    diagnostics: bool = False
    deterministic_history: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not all(math.isfinite(v) and v >= 0 for v in (self.alpha, self.beta, self.auxiliary_lr)):
            raise ValueError('Finite nonnegative historical weights required')
        if self.gamma not in (0., 1.):
            raise ValueError('Only the frozen zero/one CE intervention is allowed')
        if self.ce_scale_mode not in ('unit', 'rms'):
            raise ValueError('Only unit or dimension-derived RMS input is defined')
        if type(self.deterministic_history) is not bool or (self.deterministic_history and self.method != 'H_full'):
            raise ValueError('Deterministic birth preservation is only defined for Full')


def settings(candidate, method):
    changes = {}
    if candidate in ('H_base', 'H_CE', 'H_CE_RMS'):
        changes = dict(pooling='mean_last4_sum', encoder_lr=2.5e-5, head_lr=3e-4,
                       lr_multiplier=1., head_schedule='encoder', gamma=float(candidate != 'H_base'),
                       ce_scale_mode='rms' if candidate == 'H_CE_RMS' else 'unit',
                       diagnostics=False)
    elif candidate == 'V1':
        changes = dict(vendor_lr=2e-5)
    elif candidate != 'V0':
        raise ValueError('Unknown fixed candidate')
    if candidate.startswith('V') != (method == 'VendorLink_SEQ'):
        raise ValueError('Candidate/method family mismatch')
    value = asdict(AccountSettings(method, **changes))
    value.pop('deterministic_history')  # Preserve the exact prior task identities.
    return value


def tasks(phase, selection=None):
    if phase not in ('P', 'M') or selection != dict(H='H_CE_RMS', V='V1'):
        raise ValueError('Only the fixed FIT12 RMS/V1 P/M trajectories are approved')
    methods = ('H_SEQ', 'VendorLink_SEQ') if phase == 'P' else ('H_ER', 'H_full')
    return [dict(id=f'{phase}_{m}_s{s}', phase=phase,
                 candidate=selection['V' if m.startswith('Vendor') else 'H'],
                 method=m, seed=s, epochs=40, stages=3, domains=list(DOMAINS),
                 settings=settings(selection['V' if m.startswith('Vendor') else 'H'], m))
            for s in SEEDS for m in methods]


def deterministic_tasks(selection):
    result = [t for t in tasks('M', selection) if t['method'] == 'H_full']
    for task in result:
        task['id'] = task['id'].replace('H_full_', 'H_full_det_')
        task['settings']['deterministic_history'] = True
    return result


def deterministic_comparison(baseline, candidate):
    """One fixed intervention; retain original gates and descriptive old-Full pairs."""
    if set(candidate) != {'H_full'}:
        raise ValueError('Only the three new Full trajectories may be analyzed')
    original = decide(baseline, True)
    combined = {**baseline, **candidate}
    current = decide(combined, True)
    pairs = {}
    for comparator in ('H_full', 'H_ER', 'H_SEQ', 'V_CLS', 'V_SET'):
        rows = []
        for new, old in zip(candidate['H_full'], baseline[comparator], strict=True):
            if new['seed'] != old['seed']:
                raise ValueError('Same-seed comparisons required')
            rows.append(dict(seed=new['seed'], O=new['O']-old['O'], F_reduction=old['F']-new['F'],
                **{k: new[k]-old[k] for k in ('Q', 'R11', 'R22', 'R33', 'GB', 'GC')}))
        pairs[comparator] = dict(rows=rows,
            means={k: statistics.mean(r[k] for r in rows) for k in rows[0] if k != 'seed'},
            positive_seeds={k: sum(r[k] > 0 for r in rows) for k in ('O', 'F_reduction', 'R33')})
    return dict(status='DETERMINISTIC_HISTORY_EXPERIMENT_COMPLETE', original_M=original, candidate_M=current,
                comparisons=pairs, confirmation=False, application=False, automatic_followup=False,
                interpretation='Old-Full differences are descriptive; original effect/quality/forgetting gates remain separate')


def beta_tasks(selection):
    result = deterministic_tasks(selection)
    for task in result:
        task['id'] = task['id'].replace('H_full_det_', 'H_full_beta_quarter_')
        task['settings']['beta'] = BETA
    return result


def beta_comparison(baseline, reference, candidate):
    """Predeclared joint response, separate from the unchanged original gates."""
    if set(reference) != {'H_full'} or set(candidate) != {'H_full'}:
        raise ValueError('Exactly the fixed Full reference and candidate required')
    original = deterministic_comparison(baseline, candidate)
    # Validate the reference through the same finite, complete seed contract.
    decide({**baseline, **reference}, True)
    rows, ref = candidate['H_full'], reference['H_full']
    def mean(values): return statistics.mean(values)
    def values(source, key): return [r[key] for r in source]
    def delta(other, key): return [a[key]-b[key] for a, b in zip(rows, other, strict=True)]
    def lower(other, key, average, minimum):
        d = delta(other, key)
        return mean(d) >= average and min(d) >= minimum
    def gain(key):
        v = values(rows, key)
        return mean(v) >= .04 and min(v) >= 0
    dream = (gain('GB') and all(lower(baseline[m], 'R22', -.01, -.03) for m in ('H_SEQ', 'H_ER'))
             and sum(v > 0 for v in delta(ref, 'R22')) >= 2)
    old = (mean(delta(ref, 'O')) > 0 and sum(v > 0 for v in delta(ref, 'O')) >= 2
           and min(delta(ref, 'O')) >= -.01
           and all(mean(delta(ref, k)) >= 0 for k in ('R31', 'R32')))
    agora = (mean(delta(ref, 'R33')) >= 0 and sum(v >= 0 for v in delta(ref, 'R33')) >= 2
             and gain('GC') and all(lower(baseline[m], 'R33', -.02, -.05)
                                   for m in ('H_SEQ', 'H_ER', 'V_CLS', 'V_SET')))
    checks = original['candidate_M']['checks']
    response = dict(dream_restore=dream, old_final_improve=old, agora_protect=agora,
                    original_initial_protection=checks['initial_protection'],
                    original_new_protection=checks['new_protection'],
                    original_external_protection=checks['external'])
    # R11 and R12 precede the intervention. Report any exact numeric discrepancy;
    # it requires source/path investigation, never post-hoc attribution to beta.
    unchanged = {k: delta(ref, k) for k in ('R11', 'R12')}
    integrity = all(v == 0 for d in unchanged.values() for v in d)
    pairs = [dict(seed=r['seed'], **{k: r[k]-b[k] for k in
             ('Q', 'O', 'F', 'R11', 'R12', 'R22', 'R31', 'R32', 'R33', 'GB', 'GC')})
             for r, b in zip(rows, ref, strict=True)]
    return dict(status='BETA_QUARTER_EXPERIMENT_COMPLETE', response_checks=response,
                joint_response=all(response.values()), first_domain_differences=unchanged,
                interpretation_requires_identity_review=not integrity,
                reference_pairs=pairs, original_gates=original,
                confirmation=False, application=False, automatic_followup=False,
                interpretation='Development strategy total effect; no requirement to beat reference F; no automatic confirmation or beta search')


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
    def gain(m): return all(mean(m, k) >= .04 and min(values(m, k)) >= 0 for k in ('GB', 'GC'))
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
    extra_conditions = (improvement('H_full', 'H_ER', .02, None)
             and lower('H_full', 'H_ER', ('R11', 'R22'), -.01, -.03)
             and lower('H_full', 'H_ER', ('R22', 'R33'), -.02, -.05))
    return dict(status='PASS_SYSTEM' if system else 'STOP_NO_H_ADVANTAGE', checks=checks,
                old_score_increment=system and extra_conditions,
                old_score_increment_conditions=extra_conditions)
