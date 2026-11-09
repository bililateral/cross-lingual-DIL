"""One source-bound qualification for the approved account-supervised protocol.

Raw text/identity and role exports remain on Linux. No model import or scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

import psutil

from step29_vendorlink_data import canonical, eligible_ads, seed_for, split_for

PROTOCOL = 'vendorlink-account-20261109'
DOMAINS = ('AlphaBay', 'Dream', 'Agora')
PYTHON = '/home/yongpeng/miniconda3/envs/py310/bin/python'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def record(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def pinned(binding):
    if record(binding['path']) != {k: binding[k] for k in ('path', 'bytes', 'sha256')}:
        raise ValueError('Bound input differs: '+binding['path'])
    return Path(binding['path'])


def identity(key):
    # Exact historical identity, before the new lower() normalization.
    return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()


def normalized_text(ad):
    return canonical(ad.title).casefold()+'\0'+canonical(ad.description).casefold()


def components(ads, check=lambda: None):
    """Exact 0.90 Jaccard graph, using lossless prefix filtering, no LSH.

    A common global token order and n-ceil(.9*n)+1 prefixes guarantee
    intersecting prefixes for every pair whose Jaccard reaches the threshold.
    Full sets still decide every candidate edge. Connected paths are retained.
    """
    ads = sorted(ads, key=lambda x: x.uid)
    parent = list(range(len(ads)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parent[max(i, j)] = min(i, j)
    sets, short = {}, {}
    for i, ad in enumerate(ads):
        if i % 64 == 0:
            check()
        text = normalized_text(ad)
        tokens = re.findall(r'\w+', text)
        if len(tokens) < 5:
            if text in short:
                union(i, short[text])
            else:
                short[text] = i
        else:
            sets[i] = set(zip(*(tokens[k:] for k in range(5))))
    freq = Counter(t for values in sets.values() for t in values)
    postings = defaultdict(list)
    comparisons = 0
    for position, i in enumerate(sorted(sets, key=lambda j: (len(sets[j]), ads[j].uid))):
        check()
        values = sets[i]
        prefix = sorted(values, key=lambda t: (freq[t], t))[:len(values)-(9*len(values)+9)//10+1]
        candidates = {j for t in prefix for j in postings[t]
                      if 10*len(sets[j]) >= 9*len(values)}
        for j in candidates:
            comparisons += 1
            if comparisons % 2048 == 0:
                check()
            intersection = len(values & sets[j])
            if 10*intersection >= 9*(len(values)+len(sets[j])-intersection):
                union(i, j)
        for t in prefix:
            postings[t].append(i)
    groups = defaultdict(list)
    for i, ad in enumerate(ads):
        groups[find(i)].append(ad.uid)
    return sorted((sorted(v) for v in groups.values()), key=lambda v: v[0]), comparisons


def allocate(parts, used, global_used, market, vendor):
    """Role selection on the whole graph; no selection by model result."""
    unseen = [p for p in parts if not set(p) & (used | global_used)]
    if len(parts) < 20 or len(unseen) < 8:
        return None
    key = lambda p: (seed_for(1729, market, vendor, p[0]), p[0])
    unseen.sort(key=key)
    chosen = {p[0] for p in unseen[:8]}
    fit = sorted((p for p in parts if p[0] not in chosen), key=key)[:12]
    return dict(fit=[p[0] for p in fit], gallery=[p[0] for p in unseen[:4]],
                dev=[p[0] for p in unseen[4:6]], confirm=[p[0] for p in unseen[6:8]])


class Check:
    def __init__(self, root, seconds):
        self.root, self.started = Path(root), time.monotonic()
        self.deadline = self.started+seconds
        self.peak_rss = 0
        self.last_disk = 0
        process = psutil.Process()
        process.cpu_affinity(process.cpu_affinity()[:2])

    def __call__(self):
        now = time.monotonic()
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        if now >= self.deadline-5 or self.peak_rss > 32*2**30:
            raise RuntimeError('Qualification deadline/RSS cap')
        if now-self.last_disk > 5:
            self.last_disk = now
            size = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
            if size > 32*2**30 or shutil.disk_usage(self.root).free < 32*2**30:
                raise RuntimeError('Experiment disk cap/reserve')


def prepare(config, target, check):
    if target.exists():
        raise ValueError('Qualification is one-shot; target already exists')
    target.mkdir(parents=True)
    write(target/'access.json', dict(protocol=PROTOCOL, started=time.time(),
          purposes=['offline_exact_cleaning', 'old_fit_binding', 'usage_union', 'near_graph', 'roles'],
          canonical=config['canonical'], export_manifest=config['export_manifest'],
          forbidden_read=False, confirmation_evaluation=False))
    manifest = json.loads(pinned(config['export_manifest']).read_text())
    uses, old_payloads, usage_bindings = {}, {}, []
    for domain in DOMAINS:
        binding = next(r for r in manifest['files'] if r['path'].endswith('/'+domain+'_fit.json'))
        data = json.loads(pinned(binding).read_text())
        usage_bindings.append(binding)
        old_payloads[domain] = data
        for group in data:
            if group['domain'] != domain or group['split'] != 'fit':
                raise ValueError('Wrong historical fit source')
            for view in group['views']:
                uses.setdefault(view['identity'], {}).update({r['uid']: r for r in view['records']})
        small = manifest['qualification_samples'][domain]
        b = {k: small[k] for k in ('path', 'bytes', 'sha256')}
        q = json.loads(pinned(b).read_text())
        usage_bindings.append(b)
        for group in q['groups']:
            for view in group['views']:
                for r in view['records']:
                    if uses.get(view['identity'], {}).get(r['uid']) != r:
                        raise ValueError('Historical qualification exceeds U')
    if len(uses) != 2052 or any(len(v) != 4 for v in uses.values()):
        raise ValueError('Historical fit account/record count differs')
    write(target/'usage_sources.json', dict(bindings=usage_bindings,
          purposes=config['usage_evidence'], account_count=len(uses),
          record_count=sum(map(len, uses.values()))))
    check()
    rows = []
    with pinned(config['canonical']).open(encoding='utf-8') as f:
        for index, line in enumerate(f):
            rows.append(json.loads(line))
            if index % 4096 == 0:
                check()
    # Reuse the exact original global exact-cleaning implementation.
    clean, ads, rejected, observed = eligible_ads(rows, 2)
    del rows
    by_old = {identity(key): key for key in clean if identity(key) in uses}
    if set(by_old) != set(uses):
        raise ValueError('Cannot reconstruct all old fit source keys')
    normal = defaultdict(list)
    for old, key in by_old.items():
        market, vendor = key
        if split_for(vendor, 20261029, .7, .15) != 'fit':
            raise ValueError('Non-fit account cannot enter')
        # Verify original four-record selection, not just a plausible split.
        selected = sorted(clean[key], key=lambda f: (seed_for(20261029, 'ads', old, f), f))[:4]
        if set(selected) != set(uses[old]) or any(asdict(ads[(key, u)]) != uses[old][u] for u in selected):
            raise ValueError('Original used records/text mismatch')
        normal[(market, canonical(vendor).lower())].append(old)
    conflicts = {k for k, v in normal.items() if len(v) != 1}
    global_used = {u for records in uses.values() for u in records}
    counts, selections, custody = {}, {d: [] for d in DOMAINS}, []
    for domain in DOMAINS:
        counts[domain] = dict(old_fit_accounts=sum(by_old[x][0] == domain for x in by_old),
            normalized_conflict_accounts=0, below_20_components=0, below_8_unseen=0,
            eligible_accounts=0, clean_records=0, components=0, exact_near_comparisons=0)
    for position, (new_key, originals) in enumerate(sorted(normal.items())):
        check()
        market, vendor = new_key
        c = counts[market]
        if new_key in conflicts:
            c['normalized_conflict_accounts'] += len(originals)
            continue
        old = originals[0]
        key = by_old[old]
        records = [ads[(key, u)] for u in clean[key]]
        # Every U record must be included before connected components are built.
        if not set(uses[old]) <= {a.uid for a in records}:
            raise ValueError('Missing U text from full account near graph')
        parts, comparisons = components(records, check)
        c['clean_records'] += len(records)
        c['components'] += len(parts)
        c['exact_near_comparisons'] += comparisons
        unseen = sum(not set(p) & global_used for p in parts)
        if len(parts) < 20:
            c['below_20_components'] += 1
        if unseen < 8:
            c['below_8_unseen'] += 1
        roles = allocate(parts, set(uses[old]), global_used, market, vendor)
        if roles is not None:
            c['eligible_accounts'] += 1
            account = identity(new_key)
            selections[market].append(dict(account_uid=account,
                roles={role: [asdict(ads[(key, u)]) for u in ids] for role, ids in roles.items()}))
            custody.append(dict(account_uid=account, original_key=key, normalized_key=new_key,
                historical_uid=old, used_uids=sorted(uses[old]), components=parts, roles=roles))
        if position % 100 == 0:
            write(target/'progress.json', dict(accounts_processed=position+1, total=len(normal)))
    passed = all(c['eligible_accounts'] >= 128 for c in counts.values())
    result = dict(protocol=PROTOCOL, status='QUALIFIED' if passed else 'STOP_CAPACITY',
                  counts=counts, minimum_accounts_per_market=128, source_rows=236457,
                  old_used_accounts=len(uses), old_used_records=len(global_used),
                  prior_use='historical_training_and_qualification_fit_union',
                  global_exact_cleaning=rejected, observed_accounts=observed,
                  models_loaded=0, training_started=False, confirmation_opened=False,
                  roles_exported=passed, files={})
    if passed:
        for domain in DOMAINS:
            accounts = sorted(selections[domain], key=lambda a: a['account_uid'])
            for role in ('fit', 'gallery', 'dev', 'confirm'):
                path = target/'data'/f'{domain}_{role}.json'
                values = [dict(account_uid=a['account_uid'], records=a['roles'][role]) for a in accounts]
                write(path, values)
                path.chmod(0o600)
                result['files'][domain+':'+role] = record(path)
        path = target/'data'/'provenance.json'
        write(path, custody)
        path.chmod(0o600)
        result['provenance'] = record(path)
    write(target/'manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=3600)
    args = parser.parse_args()
    if sys.platform != 'linux' or Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise ValueError('Linux existing py310 required')
    config = json.loads(args.config.read_text())
    if config['protocol'] != PROTOCOL or not config['approved'] or not 1 <= args.seconds <= 3600:
        raise ValueError('Approved source-bound qualification required')
    pinned(config['protocol_source'])
    pinned(config['scope_source'])
    for r in config['implementation_sources']+config['evidence_sources']:
        pinned(r)
    args.root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.root).free < 64*2**30:
        raise RuntimeError('Initial free disk admission')
    check = Check(args.root, args.seconds)
    status, error = 'FAILED', None
    try:
        check()
        result = prepare(config, args.root/'qualification', check)
        status = result['status']
        print(json.dumps({k: result[k] for k in ('status', 'counts', 'roles_exported')}, ensure_ascii=True))
    except BaseException as e:
        error = type(e).__name__+': '+str(e)
        raise
    finally:
        elapsed = time.monotonic()-check.started
        write(args.root/'qualification_cost.json', dict(status=status, error=error,
              seconds=elapsed, charged_seconds=math.ceil(elapsed), bucket='auxiliary',
              peak_rss_bytes=check.peak_rss, cpu_affinity=psutil.Process().cpu_affinity(),
              gpu_seconds=0, training_seconds=0))


if __name__ == '__main__':
    main()
