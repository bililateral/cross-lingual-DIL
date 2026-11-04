"""Finite ABC development policy; no data access, dispatch or launch on import.

The project is currently preparing/reviewing this policy. Formal batch launch
remains closed; this module does not fabricate that authorization.
"""
from dataclasses import asdict, replace
import math

from step29_vendorlink_methods import Settings

CANDIDATES = ('P0', 'P1', 'P2', 'L1', 'L2', 'L3', 'T1', 'T2')
QUALITY_MARGIN = .005
SEED_MARGIN = .01


def baseline(method='H_full'):
    old = Settings(method)
    return replace(old, encoder_lr=old.encoder_lr*old.lr_multiplier,
                   head_lr=old.head_lr*old.lr_multiplier, lr_multiplier=1.)


def candidate(name, incumbent=None):
    base = baseline() if name.startswith('P') else incumbent
    if name not in CANDIDATES or base is None or base.method != 'H_full':
        raise ValueError('Finite candidate and selected H_full parent required')
    changes = dict(P0={}, P1={'pooling': 'mean_last'}, P2={'pooling': 'mean_last4_sum'},
                   L1={'encoder_lr': 2.5e-5}, L2={'head_lr': 3e-4},
                   L3={'head_schedule': 'encoder'}, T1={'bce_weight': .25},
                   T2={'hard_weight': 0.})
    return replace(base, **changes[name])


def metrics(row):
    values = {key: float(row[key]) for key in ('B2', 'B3', 'C3', 'A3')}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
        raise ValueError('Complete finite MAP cells in [0,1] required')
    return {**values, 'D': values['B2']-values['B3']}


def mean(rows):
    if not rows:
        raise ValueError('No partial/empty substitute for planned seeds')
    checked = [metrics(row) for row in rows]
    return {k: sum(row[k] for row in checked)/len(checked) for k in checked[0]}


def basic_quality(row, original):
    row, original = metrics(row), metrics(original)
    return all(row[k] >= original[k]-QUALITY_MARGIN for k in ('B2', 'C3', 'A3'))


def selection_key(name, settings, row, original, vendor):
    row, original, vendor = metrics(row), metrics(original), metrics(vendor)
    if name not in CANDIDATES or not basic_quality(row, original):
        raise ValueError('Ineligible candidate cannot be selected')
    good = all(row[k] >= max(original[k], vendor[k])-QUALITY_MARGIN for k in ('B3', 'C3'))
    old = asdict(baseline())
    changed = sum(value != old[key] for key, value in asdict(settings).items())
    return (not good, row['D'], -row['B3'], -row['C3'], changed, CANDIDATES.index(name))


def success(new, original, vendor, final=False):
    count = 3 if final else 2
    if not len(new) == len(original) == len(vendor) == count:
        raise ValueError('Exactly the planned paired seeds are required')
    n, h, v = mean(new), mean(original), mean(vendor)
    checks = dict(dream_forgetting_original=n['D'] <= h['D']-QUALITY_MARGIN,
                  dream_forgetting_vendor=n['D'] <= v['D']-QUALITY_MARGIN,
                  dream_initial=n['B2'] >= h['B2']-QUALITY_MARGIN,
                  dream_final=n['B3'] >= max(h['B3'], v['B3'])-QUALITY_MARGIN,
                  agora_new=n['C3'] >= max(h['C3'], v['C3'])-QUALITY_MARGIN,
                  alphabay_old=n['A3'] >= h['A3']-QUALITY_MARGIN,
                  vendor_forgetting_present=v['D'] > 0,
                  agora_seed_protection=all(metrics(a)['C3'] >= max(metrics(b)['C3'], metrics(c)['C3'])-SEED_MARGIN
                                             for a, b, c in zip(new, original, vendor)))
    if final:
        checks.update(direction_original=sum(metrics(a)['D'] < metrics(b)['D'] for a, b in zip(new, original)) >= 2,
                      direction_vendor=sum(metrics(a)['D'] < metrics(b)['D'] for a, b in zip(new, vendor)) >= 2)
    return dict(passed=all(checks.values()), checks=checks, interpretation='FIXED_DATA_DEVELOPMENT_ONLY')


def precheck(vendor_rows):
    if len(vendor_rows) != 3:
        raise ValueError('Old seeds 4/5/6 required before search')
    losses = [metrics(row)['D'] for row in vendor_rows]
    return sum(losses)/3 > 0 and sum(value > 0 for value in losses) >= 2
