"""Custodian-only Evolution builder; never invoked by a learning worker."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import zipfile

from step29_vendorlink_data import Ad, Group, View
from step30_evolution_policy import DOMAINS, SPLITS


def digest(purpose: str, *keys: str) -> str:
    data = ['evolution-pilot-v1', '2026110701', purpose, *map(str, keys)]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def ordered(values, purpose, *keys):
    return sorted(values, key=lambda v: (digest(purpose, *keys, v), str(v).encode('utf-8')))


@dataclass(frozen=True)
class Record:
    vid: str
    lid: str
    scrape: str
    observed: str
    source_row: int
    source_sha256: str
    title: str
    body: str
    body_key: str

    @property
    def domain(self):
        return f'{self.observed[:4]}-Q{(int(self.observed[5:7])-1)//3+1}'


def qualification_modules(directory: Path, check):
    # Exact frozen sources are bound in the operation permit. Reuse scan and
    # near-copy primitives, without invoking their Windows qualification main.
    modules = []
    for name in ('audit', 'followup'):
        spec = importlib.util.spec_from_file_location(name, directory/f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    modules[0].guard = check
    return modules


def remember_snapshot(earliest: dict, row: dict, ordinal: int, dates: dict, base, patterns: dict):
    """Mirror scan's exact earliest key, retaining the first physical row on ties."""
    v, lid, sid = (row[k].strip() for k in ('vid', 'lid', 'mscrape_id'))
    title, body = row['title'], row['description']
    if base.missing(title) or base.missing(body):
        return None
    ct, cb = base.masked(title, patterns.get(v)), base.masked(body, patterns.get(v))
    if not ct or not cb:
        return None
    raw = base.digest(base.canonical(title).casefold(), base.canonical(body).casefold())
    if v and lid and sid in dates:
        key = (dates[sid], int(sid), raw)
        if (v, lid) not in earliest or key < earliest[v, lid][0]:
            def case(text):
                text = base.CONTACT.sub(' ', base.PGP.sub(' ', text))
                pattern = patterns.get(v)
                return base.canonical(pattern.sub(' ', text) if pattern else text)
            kept_title, kept_body = case(title), case(body)
            if (kept_title.casefold(), kept_body.casefold()) != (ct, cb):
                raise ValueError('Cased/casefold masking differs')
            source_sha = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            earliest[v, lid] = (key, sid, ordinal, source_sha, kept_title, kept_body)
    return v, raw, ct, cb


def clean_archive(archive: Path, qualification: Path, expected: dict, check) -> tuple[list[Record], dict]:
    # The frozen audit.main sets this before opening the same four TSV tables.
    # This entry reuses read_table directly, so it must preserve that setting.
    csv.field_size_limit(16 * 1024**2)
    base, follow = qualification_modules(qualification, check)
    texts, earliest = {}, {}
    with zipfile.ZipFile(archive) as z:
        receipts = {}
        cats = list(base.read_table(z, 'market/categories.tsv', receipts))
        categories = {r['cid']: r['category'] for r in cats}
        parents = {r['cid']: r['parent_cid'] for r in cats}
        dates = {r['mscrape_id']: date(int(r['scrape_year']), int(r['scrape_month']), int(r['scrape_day'])).isoformat()
                 for r in base.read_table(z, 'market/scrapes.tsv', receipts)}
        names = defaultdict(set)
        for r in base.read_table(z, 'market/vendors.tsv', receipts):
            name = base.canonical(r['username'])
            if not base.missing(name):
                names[r['vid']].add(name)
        patterns = {v: re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(n) for n in sorted(ns, key=lambda n: (-len(n), n)))+r')(?!\w)', re.I)
                    for v, ns in names.items()}
        def rows():
            for n, row in enumerate(base.read_table(z, 'market/listings.tsv', receipts), 1):
                captured = remember_snapshot(earliest, row, n, dates, base, patterns)
                if captured:
                    v, raw, ct, cb = captured
                    texts[v, raw] = (ct, cb)
                yield row
        scanned = base.scan(rows(), categories, parents, dates, patterns, return_records=True)
    retained = scanned.pop('_records')
    if len(retained) != 62447 or scanned['quality']['rows'] != 1426171:
        raise ValueError('Qualification base counts changed; stop without resizing')
    body_vids, tokens_by_text = defaultdict(set), {}
    for i, ((v, h), (_, body)) in enumerate(texts.items()):
        if i % 1000 == 0:
            check()
        tokens = follow.body_signature(body)
        bh = base.digest(' '.join(tokens))
        body_vids[bh].add(v)
        tokens_by_text[v, h] = (bh, tokens)
    distinct, seen = [], set()
    for r in sorted(retained, key=lambda r: (r[3], int(r[2]), r[1])):
        bh, _ = tokens_by_text[r[0], r[5]]
        if len(body_vids[bh]) > 1 or (r[0], bh) in seen:
            continue
        seen.add((r[0], bh))
        distinct.append(r)
    rich = []
    for r in distinct:
        _, tokens = tokens_by_text[r[0], r[5]]
        alpha = [w for w in tokens if any(c.isalpha() for c in w)]
        if (len(alpha) >= 20 and sum(all(ord(c) < 128 for c in w) for w in alpha) >= .9*len(alpha)
                and len(set(alpha) & follow.ENGLISH_FUNCTION_WORDS) >= 3):
            rich.append(r)
    indexes, grams_by_index, within_comparisons = defaultdict(set), {}, 0
    for i, r in enumerate(rich):
        check()
        grams = follow.shingles(tokens_by_text[r[0], r[5]][1])
        fp = follow.fingerprint(grams)
        candidates = set().union(*(indexes[r[0], j, x] for j, x in enumerate(fp)))
        duplicate = False
        for c in sorted(candidates):
            within_comparisons += 1
            if follow.jaccard(grams, grams_by_index[c]) >= .8:
                duplicate = True
                break
        if duplicate:
            continue
        grams_by_index[i] = grams
        for j, x in enumerate(fp):
            indexes[r[0], j, x].add(i)
    indexed = {i: r for i, r in enumerate(rich) if i in grams_by_index}
    bad, comparisons = follow.cross_account_exclusions(indexed, grams_by_index)
    final = [r for i, r in indexed.items() if i not in bad]
    counts = dict(base_capacity=base.capacity(retained), body_unique_and_not_shared_capacity=base.capacity(distinct),
                  body_quality_screen_capacity=base.capacity(rich), near_copy_screen_capacity=base.capacity(list(indexed.values())),
                  cross_account_screen_capacity=base.capacity(final), cross_account_near_copy_removed_records=len(bad),
                  cross_account_exact_comparisons=comparisons, near_copy_comparisons=within_comparisons,
                  within_account_near_copies_removed=len(rich)-len(indexed))
    # Only saved counts, not frozen qualification's unrelated PGP diagnostics.
    for key, value in counts.items():
        old = expected[key]
        if isinstance(value, dict):
            if any(value[k] != old[k] for k in ('records', 'accounts', 'accounts_at_least_records')):
                raise ValueError('Qualification cleaning counts changed: '+key)
        elif value != old:
            raise ValueError('Qualification cleaning counts changed: '+key)
    output = []
    for r in final:
        key, sid, row, source_sha, title, body = earliest[r[0], r[1]]
        if key != (r[3], int(r[2]), r[5]) or sid != r[2] or (title.casefold(), body.casefold()) != texts[r[0], r[5]]:
            raise ValueError('Selected exact source snapshot is not recoverable')
        output.append(Record(r[0], r[1], r[2], r[3], row, source_sha, title, body,
                             tokens_by_text[r[0], r[5]][0].hex()))
    return output, dict(cleaning=counts, table_receipts=receipts)


def match_accounts(eligible: dict[str, set[str]], quota: int = 320, check=None) -> dict[str, list[str]]:
    check = check or (lambda: None)  # Optional only for the handwritten fixture.
    slots = [(d, str(i)) for d in DOMAINS for i in range(quota)]
    slots.sort(key=lambda x: (digest('slot', *x), tuple(k.encode() for k in x)))
    edges = {}
    for slot in slots:
        check()
        edges[slot] = ordered(eligible[slot[0]], 'edge', *slot)
    held = {}
    # Iterative DFS is equivalent to the specified recursive augment, without
    # depending on Python's recursion limit for 960 slots.
    for start in slots:
        check()
        visited, stack = set(), [(start, iter(edges[start]), None)]
        steps = 0
        found = False
        while stack:
            steps += 1
            if steps % 1024 == 0:
                check()
            slot, it, via = stack[-1]
            v = next(it, None)
            if v is None:
                stack.pop()
                continue
            if v in visited:
                continue
            visited.add(v)
            if v in held:
                stack.append((held[v], iter(edges[held[v]]), v))
                continue
            held[v] = slot
            for index in range(len(stack)-1, 0, -1):
                held[stack[index][2]] = stack[index-1][0]
            found = True
            break
        if not found:
            raise ValueError('CAPACITY_STOP: fixed quota cannot be matched')
    return {d: [v for v, slot in held.items() if slot[0] == d] for d in DOMAINS}


def validate_group(group: Group, training: bool) -> None:
    group.validate(training)
    if len(group.views) != 16 or any(len(v.records) != 2 for v in group.views):
        raise ValueError('Exactly eight accounts, sixteen views, thirty-two records required')
    if training and (len(Counter(v.identity for v in group.views)) != 8 or
                     set(Counter(v.identity for v in group.views).values()) != {2}):
        raise ValueError('Each fit account must have exactly two disjoint views')
    if not training and any(v.identity is not None for v in group.views):
        raise ValueError('Only blinded evaluation input permitted')


def partition(records: list[Record], split_counts: dict | None = None, check=None) -> tuple[dict, dict]:
    check = check or (lambda: None)
    counts = SPLITS if split_counts is None else split_counts  # Small handwritten CPU fixture only.
    if set(counts) != set(SPLITS) or any(n < 8 or n % 8 for n in counts.values()):
        raise ValueError('Exact whole eight-account groups required')
    by_account = defaultdict(list)
    for ordinal, r in enumerate(records):
        if ordinal % 1000 == 0:
            check()
        if r.domain in DOMAINS:
            by_account[r.domain, r.vid].append(r)
    eligible = {d: {v for (domain, v), rs in by_account.items() if domain == d and len(rs) >= 4} for d in DOMAINS}
    assigned = match_accounts(eligible, sum(counts.values()), check)
    exports, custody = {}, dict(mapping=[], qrels={}, language_samples={})
    selected_vids, selected_lids, selected_keys = set(), set(), set()
    for d in DOMAINS:
        vids = ordered(assigned[d], 'split', d)
        offset = 0
        for split in ('fit', 'dev', 'test'):
            selected = vids[offset:offset+counts[split]]
            offset += counts[split]
            selected = ordered(selected, 'group', d, split)
            groups, qrels, fit_records = [], {}, []
            for index in range(0, len(selected), 8):
                check()
                uid = digest('group', d, split, *selected[index:index+8])
                views = []
                for v in selected[index:index+8]:
                    if v in selected_vids:
                        raise ValueError('Observed account crosses domain/split')
                    selected_vids.add(v)
                    rs = sorted(by_account[d, v], key=lambda r: (digest('record', d, v, r.lid), r.lid.encode()))[:4]
                    for r in rs:
                        if r.lid in selected_lids or r.body_key in selected_keys:
                            raise ValueError('Record or normalized body overlaps exported views')
                        selected_lids.add(r.lid)
                        selected_keys.add(r.body_key)
                    identity = digest('fit-identity', v)
                    for j in (0, 2):
                        ads = tuple(Ad(digest('record-id', r.lid), r.title, r.body) for r in rs[j:j+2])
                        views.append(View(digest('view', *(r.uid for r in ads)), ads, identity))
                    custody['mapping'].extend(dict(domain=d, split=split, group=uid, **asdict(r)) for r in rs)
                    fit_records.extend(rs)
                group = Group(uid, d, split, tuple(views)).evaluation_order()
                if split != 'fit':
                    qrels[uid] = dict(view_ids=[v.uid for v in group.views], labels=group.labels())
                    group = group.blind()
                validate_group(group, split == 'fit')
                groups.append(group)
            groups.sort(key=lambda g: g.uid.encode())
            exports[d, split] = groups
            if split != 'fit':
                custody['qrels'][d, split] = qrels
            else:
                sampled = sorted(fit_records, key=lambda r: (digest('language-audit', d, r.lid), r.lid.encode()))[:32]
                custody['language_samples'][d] = [dict(id=digest('language-audit', d, r.lid), body=r.body) for r in sampled]
    return exports, custody


def export_dataset(root: Path, records: list[Record], cleaning: dict, write, record, check=None) -> dict:
    check = check or (lambda: None)
    check()
    if root.exists():
        raise ValueError('Fresh custodian directory required; no re-split or overwrite')
    root.mkdir(parents=True)
    exports, custody = partition(records, check=check)
    private = root/'private_custody'
    private.mkdir(mode=0o700)
    files = {}
    for (domain, split), groups in exports.items():
        check()
        path = root/f'{domain}_{split}.json'
        write(path, [asdict(g) for g in groups])
        files[domain+':'+split] = record(path)
    for (domain, split), labels in custody['qrels'].items():
        check()
        path = private/f'{domain}_{split}_qrels.json'
        write(path, labels)
        files[domain+':'+split+':qrels'] = record(path)
    write(private/'source_mapping.json', custody['mapping'])
    check()
    write(private/'language_samples.json', custody['language_samples'])
    manifest = dict(protocol='evolution-pilot-v1', files=files, counts=SPLITS, domains=list(DOMAINS),
                    language_audit_passed=False, cleaning=cleaning,
                    language_samples=record(private/'language_samples.json'))
    check()
    write(root/'manifest.json', manifest)
    check()
    return manifest


def language_result(samples: dict, decisions: dict) -> dict:
    if set(samples) != set(DOMAINS) or set(decisions) != set(DOMAINS):
        raise ValueError('Three domains of fixed manual fit judgments required')
    counts = {}
    for d in DOMAINS:
        ids = {r['id'] for r in samples[d]}
        if len(ids) != 32 or set(decisions[d]) != ids or any(v not in ('english', 'other', 'unclear') for v in decisions[d].values()):
            raise ValueError('Exactly the thirty-two preselected fit texts must be judged')
        counts[d] = sum(v == 'english' for v in decisions[d].values())
    return dict(passed=all(v >= 29 for v in counts.values()), clearly_english=counts,
                scope='Manual fit-only sample; not certification of all dev/test language')
