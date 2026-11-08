"""Use the qualified cache once; export only development, reserve confirmation IDs."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from step29_vendorlink_data import Ad, View, Group
from step30_evolution_data import validate_group
from step30_evolution_io import access, read, record, verify, write
from step31_evolution_registered_policy import PROTOCOL, DOMAINS, CACHE_SHA


def key(purpose, *parts):
    return hashlib.sha256(json.dumps(['registered-v1', 41, purpose, *parts],
                                    ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def partition(rows, *, quota=96, confirmation_count=288):
    if len({r['vid'] for r in rows}) != len(rows) or quota % 8:
        raise ValueError('Distinct accounts and complete groups required')
    ordered = lambda xs, purpose: sorted(xs, key=lambda r: key(purpose, r['vid']))
    old_test = ordered([r for r in rows if r['prior_usage'] == 'test'], 'confirmation')
    unused = ordered([r for r in rows if r['prior_usage'] == 'unused'], 'confirmation')
    if len(old_test) > confirmation_count:
        raise ValueError('Unexpected prior-test capacity; no silent changed selection')
    reserved = old_test + unused[:confirmation_count-len(old_test)]
    if len(reserved) != confirmation_count:
        raise ValueError('STOP_CAPACITY: independent confirmation reservation')
    reserved_ids = {r['vid'] for r in reserved}
    pool = [r for r in rows if r['vid'] not in reserved_ids and r['prior_usage'] in ('fit', 'dev', 'unused')]
    chosen = ordered(pool, 'development')[:3*quota]
    if len(chosen) != 3*quota:
        raise ValueError('STOP_CAPACITY: development')
    chosen.sort(key=lambda r: (r['roles']['fit'][-1]['observed'], key('registration', r['vid'])))
    if len({r['roles']['fit'][-1]['observed'] for r in chosen}) < 3:
        raise ValueError('STOP_DATES: fewer than three registration days')
    exports, custody, dates, language = {}, {}, {}, {}
    all_lids, all_keys = set(), set()
    for d, start in zip(DOMAINS, range(0, len(chosen), quota)):
        accounts = chosen[start:start+quota]
        groups, galleries, queries, truth, fit_rows = [], [], [], {}, []
        for offset in range(0, quota, 8):
            views = []
            for row in accounts[offset:offset+8]:
                parts = row['roles']
                if {k: len(v) for k, v in parts.items()} != dict(fit=4, gallery=2, query=2):
                    raise ValueError('Exact 4/2/2 required')
                if not (max(r['observed'] for r in parts['fit']) < min(r['observed'] for r in parts['gallery'])
                        and max(r['observed'] for r in parts['gallery']) < min(r['observed'] for r in parts['query'])):
                    raise ValueError('Strict later observation day required')
                for r in sum(parts.values(), []):
                    if r['lid'] in all_lids or r['body_key'] in all_keys:
                        raise ValueError('Record or normalized body repeated across exported roles/accounts')
                    all_lids.add(r['lid']); all_keys.add(r['body_key'])
                def view(records, identity=None):
                    ads = tuple(Ad(key('record', r['lid']), r['title'], r['body']) for r in records)
                    return View(key('view', *sorted(r.uid for r in ads)), ads, identity)
                fit_identity = key('fit-identity', row['vid'])
                views += [view(parts['fit'][:2], fit_identity), view(parts['fit'][2:], fit_identity)]
                gallery, query = view(parts['gallery']), view(parts['query'])
                galleries.append(gallery); queries.append(query)
                truth[query.uid] = gallery.uid
                fit_rows.extend(parts['fit'])
            group = Group(key('group', *sorted(v.uid for v in views)), d, 'fit', tuple(views)).evaluation_order()
            validate_group(group, True)
            groups.append(group)
        exports[d] = dict(fit=[asdict(g) for g in sorted(groups, key=lambda g: g.uid)],
                          gallery=[asdict(v) for v in sorted(galleries, key=lambda v: v.uid)],
                          query=[asdict(v) for v in sorted(queries, key=lambda v: v.uid)])
        custody[d] = truth
        regs = [r['roles']['fit'][-1]['observed'] for r in accounts]
        dates[d] = dict(first=regs[0], last=regs[-1], distinct_days=len(set(regs)),
                        prior_usage=dict(Counter(r['prior_usage'] for r in accounts)))
        sampled = sorted(fit_rows, key=lambda r: key('language', d, r['lid']))[:32]
        language[d] = [dict(id=key('language', d, r['lid']), body=r['body']) for r in sampled]
    return exports, custody, language, dict(
        reserved_ids=sorted(reserved_ids), development_ids=sorted(r['vid'] for r in chosen),
        development_dates=dates, boundary_same_day=[dates[a]['last'] == dates[b]['first']
                                                   for a, b in zip(DOMAINS, DOMAINS[1:])],
        reserved_prior_usage=dict(Counter(r['prior_usage'] for r in reserved)),
        development_accounts=len(chosen), development_records=len(all_lids))


def build(root: Path, cache_binding: dict, check):
    if root.exists() or cache_binding['sha256'] != CACHE_SHA or cache_binding['bytes'] != 6250335:
        raise ValueError('Fresh development export and exact previously qualified cache required')
    root.mkdir(parents=True)
    private = root/'private_custody'; private.mkdir(mode=0o700)
    access(root, 'qualified_cache_build', [cache_binding])
    rows = [json.loads(line) for line in verify(cache_binding).read_text(encoding='utf-8').splitlines()]
    if Counter(r['prior_usage'] for r in rows) != dict(fit=262, dev=104, test=160, unused=130):
        raise ValueError('Qualified cache counts differ')
    check()
    exports, qrels, language, selection = partition(rows)
    files = {}
    for d in DOMAINS:
        for role, values in exports[d].items():
            p = root/f'{d}_{role}.json'; write(p, values); files[f'{d}:{role}'] = record(p)
        p = private/f'{d}_qrels.json'; write(p, qrels[d]); files[f'{d}:qrels'] = record(p)
    write(private/'selection.json', selection)
    write(private/'language_samples.json', language)
    summary = {k: v for k, v in selection.items() if not k.endswith('_ids')}
    manifest = dict(protocol=PROTOCOL, domains=list(DOMAINS), accounts_per_batch=96,
                    files=files, cache=cache_binding, summary=summary,
                    language_samples=record(private/'language_samples.json'),
                    selection=record(private/'selection.json'), confirmation_exported=False)
    write(root/'manifest.json', manifest)
    check()
    return manifest


def language_adjudicate(root: Path, decisions: dict):
    manifest = read(root/'manifest.json')
    samples = read(verify(manifest['language_samples']))
    if set(decisions) != set(DOMAINS):
        raise ValueError('All three fixed language samples required')
    counts = {}
    for d in DOMAINS:
        if len(samples[d]) != 32 or set(decisions[d]) != {r['id'] for r in samples[d]}:
            raise ValueError('Exactly the original 32 sample IDs required')
        if any(v not in ('english', 'other', 'unclear') for v in decisions[d].values()):
            raise ValueError('Unknown language judgment')
        counts[d] = sum(v == 'english' for v in decisions[d].values())
    p = root/'language_receipt.json'
    if p.exists():
        raise ValueError('No replacement language audit')
    result = dict(passed=all(n >= 29 for n in counts.values()), clearly_english=counts,
                  manifest=record(root/'manifest.json'), language_samples=manifest['language_samples'])
    write(p, result)
    return result
