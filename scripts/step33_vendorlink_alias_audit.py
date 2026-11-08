"""Audit existing three-market identity evidence; never create alias truth by rule.

Linux-only, stdlib plus existing gpg. Raw identity/text evidence remains on Linux.
The public summary contains counts, source identities and explicit evidence limits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import tarfile
import time
from urllib.parse import urlparse
import zipfile

ROOT = Path('/home/yongpeng/cross-lingual')
MARKETS = {'alphabay': 'AlphaBay', 'alpha bay': 'AlphaBay',
           'dream': 'Dream', 'dream market': 'Dream', 'agora': 'Agora'}
TOKEN = re.compile(r"'(?:[^'\\]|\\.|'')*'|0x[0-9a-fA-F]+|NULL|-?\d+(?:\.\d+)?")
ARMOR = re.compile(r'-----BEGIN PGP PUBLIC KEY BLOCK-----.*?-----END PGP PUBLIC KEY BLOCK-----', re.S | re.I)
CLAIM = re.compile(r'\b(?:formerly|previously|also known as|same (?:vendor|seller)|our (?:other )?accounts|find (?:us|me)|we (?:are|were) on|i (?:am|was) on|moved (?:from|to))\b', re.I)
MARKET_MENTION = re.compile(r'\b(?:alpha\s?bay|dream(?:\s+market)?|agora)\b', re.I)
EMAIL = re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,24}')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def uid(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def plain(text):
    return html.unescape(re.sub('<[^>]+>', ' ', text))


def sql_values(line):
    """Parse one data tuple, with MySQL escapes; never execute SQL."""
    s = line.strip()
    if not s.startswith('('):
        return None
    values, pos = [], 1
    escapes = {'0': '\0', 'b': '\b', 'n': '\n', 'r': '\r', 't': '\t', 'Z': '\x1a'}
    while True:
        while pos < len(s) and s[pos].isspace():
            pos += 1
        m = TOKEN.match(s, pos)
        if not m:
            raise ValueError('Unparsed SQL value')
        token = m.group(); pos = m.end()
        if token.startswith("'"):
            raw, text, k = token[1:-1], [], 0
            while k < len(raw):
                if raw[k] == '\\':
                    k += 1; text.append(escapes.get(raw[k], raw[k]))
                elif raw[k:k+2] == "''":
                    text.append("'"); k += 1
                else:
                    text.append(raw[k])
                k += 1
            values.append(''.join(text))
        elif token == 'NULL':
            values.append(None)
        else:
            values.append(token)
        while pos < len(s) and s[pos].isspace():
            pos += 1
        if s[pos:pos+1] == ',':
            pos += 1
        elif re.fullmatch(r'\)\s*[,;]?\s*', s[pos:]):
            return values
        else:
            raise ValueError('Unparsed SQL tuple suffix')


def rows_sql(path, expected_columns, stats):
    columns = None
    with path.open(encoding='utf-8', errors='strict') as f:
        for number, line in enumerate(f, 1):
            if line.startswith('INSERT INTO'):
                m = re.match(r'INSERT INTO\s+`[^`]+`\s*\((.*?)\)\s*VALUES\s*$', line.strip())
                if not m:
                    raise ValueError('Unexpected INSERT layout')
                columns = re.findall(r'`([^`]+)`', m.group(1))
                if columns != expected_columns:
                    raise ValueError('Source columns differ from inspected schema')
            elif line.lstrip().startswith('('):
                stats['tuple_lines'] += 1
                vals = sql_values(line)
                if columns is None or len(vals) != len(columns):
                    raise ValueError('SQL arity mismatch')
                yield number, dict(zip(columns, vals))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_jsonl(path, values):
    with path.open('x', encoding='utf-8') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False)+'\n')


def self_check():
    assert sql_values("(1, 'a,b', 'it''s', 'x\\ny', 0x4142, NULL);") == ['1', 'a,b', "it's", 'x\ny', '0x4142', None]
    assert sql_values("(1, 'a\\\'b', 0),") == ['1', "a'b", '0']
    try:
        sql_values("(1, 'x') garbage")
    except ValueError:
        pass
    else:
        raise AssertionError('Malformed row accepted')
    assert CLAIM.search('I was on Agora under a different name')
    assert not CLAIM.search('The item is available')
    assert len(ARMOR.findall('-----BEGIN PGP PUBLIC KEY BLOCK-----\na\n-----END PGP PUBLIC KEY BLOCK-----')) == 1
    print(json.dumps({'self_check': 'PASS', 'tests': 6, 'does_not_certify_alias_truth': True}))


def run(out):
    started = time.monotonic()
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    signal.alarm(3500)
    out.mkdir(parents=True, exist_ok=False)
    private = out/'evidence'; private.mkdir()
    sources = {
        'canonical': ROOT/'data/vendorlink/three_market_20261029/canonical.jsonl',
        'grams': ROOT/'data/vendorlink/gwern_20261029/grams-20150714-20160417.tar.xz',
        'agora': ROOT/'data/vendorlink/agora_20261029/kaggle.zip',
        'registry': ROOT/'tijkc3xx.sql', 'pgp': ROOT/'3z669jwe.sql',
    }
    identities = {k: {'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for k,p in sources.items()}
    if identities['grams']['sha256'] != '3163bc149144b8460216ee81e7ded5594b653ad0b89bd49a88f9eff544913c97' or identities['agora']['sha256'] != 'e982c503097893f5aa07a3122ef2b0a8f96c81458966d09600d9bbf0128b26d8':
        raise ValueError('Pinned original archives changed')
    if identities['canonical']['sha256'] != 'fc80eaf41e712691bd15761a8854a3f094f93ce1153799dabb84a7b9753f4855':
        raise ValueError('Previously qualified canonical cache changed')
    accounts = {}
    by_name = defaultdict(list)
    with sources['canonical'].open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line); key = (r['market'], r['vendor'])
            a = accounts.setdefault(key, {'uid': uid(*key), 'market': key[0], 'vendor': key[1], 'records': 0, 'texts': set(), 'dates': set()})
            a['records'] += 1; a['texts'].add(uid(r['title'], r['description']))
            if r.get('source_date'): a['dates'].add(r['source_date'])
    for key, a in accounts.items():
        by_name[(key[0], key[1].casefold())].append(a)
    print(json.dumps({'phase': 'canonical', 'accounts': len(accounts), 'seconds': time.monotonic()-started}), flush=True)

    # Inspect only profile-header associations, not feedback/listing identities.
    pages, hosts, html_stats = {}, defaultdict(Counter), Counter()
    html_manifest = []
    for p in sorted((ROOT/'html-rips').glob('0x*')):
        raw = p.read_bytes(); text = raw.decode('utf-8', errors='strict')
        start = text.find('Profile Pages'); end = text.find('PGP KEY', start)
        html_stats['pages'] += 1
        if start < 0 or end < start:
            html_stats['missing_profile_region'] += 1; continue
        region = text[start:end]; links = []
        for href, body in re.findall(r'<a\b[^>]*href=[\"\']([^\"\']+)[\"\'][^>]*>(.*?)</a>', region, re.S | re.I):
            label = ' '.join(plain(body).split())
            m = re.fullmatch(r'(.*?)\s*\[([^]]+)\]', label)
            if not m: continue
            market = MARKETS.get(m.group(2).strip().casefold())
            if not market: continue
            name = m.group(1).strip(); href = html.unescape(href)
            links.append({'market': market, 'vendor': name, 'url': href})
            hosts[urlparse(href).netloc.lower()][market] += 1
        if links:
            header = text[:end]
            pages[p.name.upper()] = {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'links': links,
                'markers': {s: len(re.findall(s, header, re.I)) for s in ['verified', 'unverified', 'imposter', 'signature', 'challenge']}}
            html_manifest.append({'path': str(p), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    host_market = {h: next(iter(c)) for h,c in hosts.items() if len(c) == 1}
    if any(len(c) != 1 for c in hosts.values()):
        raise ValueError('Conflicting HTML market labels for same host')

    registry, registry_stats, matches = {}, Counter(), Counter()
    cols = ['vendor_id','user_name','market_id','user_id','link','profile','vendor_link','added','updated','scraped','imposter']
    for number, r in rows_sql(sources['registry'], cols, registry_stats):
        market = host_market.get(urlparse(r['vendor_link']).netloc.lower())
        if not market:
            registry_stats['outside_matched_three_market_hosts'] += 1; continue
        registry_stats[market] += 1
        exact = accounts.get((market, r['user_name']))
        candidates = [exact] if exact else by_name.get((market, r['user_name'].casefold()), [])
        match_kind = 'exact_market_name' if exact else ('casefold_unique' if len(candidates) == 1 else ('ambiguous_casefold' if candidates else 'no_text_match'))
        matches[match_kind] += 1
        value = {k:r[k] for k in ['vendor_id','user_name','market_id','vendor_link','added','updated','scraped','imposter']}
        value.update(market=market, source_line=number, match_kind=match_kind, account_uid=candidates[0]['uid'] if len(candidates)==1 else None)
        if r['vendor_id'] in registry: raise ValueError('Duplicate registry vendor_id')
        registry[r['vendor_id']] = value

    pgp_stats, candidate_groups, selected_keys = Counter(), [], {}
    cols = ['id','alias','fingerprint_f','fingerprint','public_key','vendor_ids','user_hash','review_count','star_rate']
    for number, r in rows_sql(sources['pgp'], cols, pgp_stats):
        refs = [x.strip() for x in (r['vendor_ids'] or '').split(',') if x.strip()]
        if any(not x.isdigit() for x in refs): raise ValueError('Unexpected vendor_ids encoding')
        members = [registry[x] for x in refs if x in registry]
        if not members: continue
        pgp_stats['rows_with_three_market_registry'] += 1
        joined = {v['account_uid']: v for v in members if v['account_uid']}
        if len(joined) < 2: continue
        pgp_stats['rows_linking_multiple_text_accounts'] += 1
        fingerprint = re.sub(r'[^A-Fa-f0-9]', '', r['fingerprint'] or '').upper()
        short = (r['fingerprint_f'] or '').upper()
        page = pages.get(short if short.startswith('0X') else '0X'+short)
        group = {'source_line': number, 'key_row_id': r['id'], 'fingerprint': fingerprint, 'fingerprint_short': short,
                 'members': list(joined.values()), 'same_name_only': len({v['user_name'].casefold() for v in joined.values()}) == 1,
                 'markets': sorted({v['market'] for v in joined.values()}),
                 'html_page': page, 'label': 'UNADJUDICATED_SHARED_KEY_CANDIDATE',
                 'imposter_marked_members': sum(v['imposter'] == '1' for v in joined.values())}
        candidate_groups.append(group)
        public_key = r['public_key'] or ''
        if not re.fullmatch(r'0x[0-9a-fA-F]+', public_key):
            pgp_stats['candidate_key_encoding_failure'] += 1
        else:
            selected_keys[r['id']] = bytes.fromhex(public_key[2:])
    gpg_dir = private/'gpg'; gpg_dir.mkdir(mode=0o700)
    for group in candidate_groups:
        data = selected_keys.get(group['key_row_id'])
        if data is None: continue
        result = subprocess.run(['gpg','--batch','--no-options','--homedir',str(gpg_dir),'--with-colons','--fingerprint','--show-keys'], input=data, capture_output=True, timeout=10)
        fingerprints = [s.split(':')[9] for s in result.stdout.decode(errors='replace').splitlines() if s.startswith('fpr:')]
        group['key_parse'] = {'returncode': result.returncode, 'fingerprints': fingerprints, 'declared_matches_primary': bool(fingerprints and fingerprints[0] == group['fingerprint'])}
        pgp_stats['candidate_public_keys_parseable' if result.returncode == 0 and fingerprints else 'candidate_public_keys_parse_failure'] += 1
        if fingerprints and fingerprints[0] == group['fingerprint']: pgp_stats['candidate_declared_fingerprint_matches_primary'] += 1
    write_jsonl(private/'registry_matches.jsonl', registry.values())
    write_jsonl(private/'shared_key_candidates.jsonl', candidate_groups)
    write_jsonl(private/'html_sources.jsonl', html_manifest)
    serialized_accounts = []
    for a in accounts.values():
        b = dict(a); b['distinct_texts'] = len(b.pop('texts')); dates = sorted(b.pop('dates'))
        b['first_observed'] = dates[0] if dates else None; b['last_observed'] = dates[-1] if dates else None
        serialized_accounts.append(b)
    write_jsonl(private/'accounts.jsonl', serialized_accounts)
    summary = {
        'status': 'EVIDENCE_CANDIDATES_AWAIT_ADJUDICATION', 'observed_server': datetime.now().astimezone().isoformat(),
        'sources': identities, 'script_sha256': digest(Path(__file__)),
        'canonical': {'records': sum(a['records'] for a in accounts.values()), 'accounts': len(accounts),
            'accounts_by_market': dict(Counter(a['market'] for a in accounts.values()))},
        'html': dict(html_stats), 'target_html_pages': len(pages),
        'market_hosts_from_labeled_profile_links': host_market, 'registry': dict(registry_stats), 'registry_text_matches': dict(matches),
        'pgp': dict(pgp_stats), 'candidate_groups': len(candidate_groups),
        'candidate_different_name_groups': sum(not g['same_name_only'] for g in candidate_groups),
        'candidate_cross_market_groups': sum(len(g['markets']) > 1 for g in candidate_groups),
        'candidate_groups_by_markets': dict(Counter('+'.join(g['markets']) for g in candidate_groups)),
        'candidate_groups_with_imposter_markers': sum(g['imposter_marked_members'] > 0 for g in candidate_groups),
        'candidate_unique_text_accounts': len({m['account_uid'] for g in candidate_groups for m in g['members']}),
        'verified_alias_pairs': None, 'verified_negative_pairs': None,
        'dataset_ready': False, 'training_started': False,
        'limits': ['Shared public keys are copyable and are not proof of account control.',
            'Market-name joins are candidate joins, not proven persistent account identity.',
            'None means not yet adjudicated, not zero real aliases.',
            'No old labels, predictions, owners, Memory or private_custody accessed.',
            'Historical source data and prior exposure are not independent new observations.'],
        'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time(),
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write_json(out/'summary.json', summary)
    total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if total > 2*1024**3: raise ValueError('Output quota exceeded')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux': raise RuntimeError('Linux existing py310 required')
    if args.self_check:
        self_check()
    elif args.output:
        run(args.output)
    else:
        parser.error('--output or --self-check required')
