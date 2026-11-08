"""Limited follow-up: Grams link groups and original-ad evidence, no labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import html
import io
import json
import os
from pathlib import Path
import re
import resource
import signal
import time
import unicodedata
import tarfile
import zipfile

from step33_vendorlink_alias_audit import (ROOT, MARKETS, ARMOR, CLAIM,
    MARKET_MENTION, digest, rows_sql, uid, write_json, write_jsonl)

SIGNED = re.compile(r'-----BEGIN PGP SIGNED MESSAGE-----.*?-----END PGP SIGNATURE-----', re.S | re.I)
DANGER = re.compile(r'\b(?:impost[eo]r|impersonat\w*|copycat|not us|not me|fake account)\b', re.I)


def canon(s):
    return ' '.join(unicodedata.normalize('NFKC', s).split())


def originals(stats):
    csv.field_size_limit(16*1024**2)
    with tarfile.open(ROOT/'data/vendorlink/gwern_20261029/grams-20150714-20160417.tar.xz', 'r:xz') as archive:
        for member in archive:
            market = {'Alpha.csv':'AlphaBay', 'Dream.csv':'Dream'}.get(Path(member.name).name)
            if not market or not member.isfile(): continue
            stats[market]['files'] += 1
            date = member.name.split('/')[1]
            with archive.extractfile(member) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', errors='replace', newline=''))
                required = ['market_name','vendor_name','name','description','item_link']
                if not set(required).issubset(reader.fieldnames or []): raise ValueError('Grams schema changed')
                for line, row in enumerate(reader, 2):
                    stats[market]['rows'] += 1
                    if None in row or any(not isinstance(row.get(k),str) for k in required):
                        stats[market]['malformed_rows'] += 1; continue
                    if MARKETS.get(canon(row['market_name']).casefold()) != market:
                        stats[market]['wrong_market_rows'] += 1; continue
                    vendor, title, description, item = [canon(row[k]) for k in ['vendor_name','name','description','item_link']]
                    if not all([vendor,title,description,item]) or '\ufffd' in vendor+title+description:
                        stats[market]['missing_or_encoding_rejected'] += 1; continue
                    yield market,vendor,row['name'],row['description'],{'archive':'grams','member':member.name,'line':line,'date':date,'item_link':row['item_link']}
    with zipfile.ZipFile(ROOT/'data/vendorlink/agora_20261029/kaggle.zip') as archive:
        if archive.namelist() != ['Agora.csv']: raise ValueError('Unexpected Agora archive')
        stats['Agora']['files'] += 1
        with archive.open('Agora.csv') as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding='ISO-8859-1', newline=''))
            for line, row in enumerate(reader, 2):
                stats['Agora']['rows'] += 1
                if None in row or any(not isinstance(row.get(k),str) for k in ['Vendor',' Item',' Item Description']):
                    stats['Agora']['malformed_rows'] += 1; continue
                vendor,title,description = [canon(row[k]) for k in ['Vendor',' Item',' Item Description']]
                if not all([vendor,title,description]):
                    stats['Agora']['missing_or_encoding_rejected'] += 1; continue
                yield 'Agora',vendor,row[' Item'],row[' Item Description'],{'archive':'agora','member':'Agora.csv','line':line,'date':None,'item_link':None}


def run(prior, out):
    start = time.monotonic(); os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(3000,3000)); signal.alarm(3000)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    out.mkdir(parents=True,exist_ok=False); evidence = out/'evidence'; evidence.mkdir()
    previous = json.loads((prior/'summary.json').read_text())
    for source in previous['sources'].values():
        if digest(source['path']) != source['sha256']: raise ValueError('Previously read source changed')
    accounts = {r['uid']:r for r in map(json.loads,(prior/'evidence/accounts.jsonl').open())}
    registry = {r['vendor_id']:r for r in map(json.loads,(prior/'evidence/registry_matches.jsonl').open())}
    groups = list(map(json.loads,(prior/'evidence/shared_key_candidates.jsonl').open()))
    columns = ['vendor_id','user_name','market_id','user_id','link','profile','vendor_link','added','updated','scraped','imposter']
    registry_stats = Counter(); links = defaultdict(list); profiles = []; times = defaultdict(list)
    for number,r in rows_sql(ROOT/'tijkc3xx.sql',columns,registry_stats):
        if r['vendor_id'] not in registry: continue
        v = registry[r['vendor_id']]; v['link_group'] = r['link']
        if v['account_uid'] and r['link'] not in ('0','',None): links[r['link']].append(v)
        if r['profile']:
            registry_stats['target_nonempty_profile_fields'] += 1
            if CLAIM.search(r['profile']) or DANGER.search(r['profile']) or 'PGP SIGNED MESSAGE' in r['profile']:
                profiles.append({'vendor_id':r['vendor_id'],'account_uid':v['account_uid'],'profile':r['profile'],'source_line':number})
        for field in ['added','updated','scraped']:
            if r[field] not in ('0','',None): times[field].append(int(r[field]))
    link_groups = []
    for key,values in links.items():
        members = {v['account_uid']:v for v in values}
        if len(members)<2: continue
        link_groups.append({'link_group':key,'members':list(members.values()),'label':'UNADJUDICATED_GRAMS_GROUP',
            'different_names':len({v['user_name'].casefold() for v in members.values()})>1,
            'markets':sorted({v['market'] for v in members.values()}),
            'imposter_members':sum(v['imposter']=='1' for v in members.values())})
    # Connected components here measure candidate dependence only, never alias truth.
    parent = {}
    def find(x):
        parent.setdefault(x,x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for g in groups+link_groups:
        members = [v['account_uid'] for v in g['members']]
        for a in members[1:]: parent[find(a)] = find(members[0])
    components = defaultdict(list)
    for a in list(parent): components[find(a)].append(a)
    capacity = {}
    for minimum in [2,4,8]:
        usable = []
        for members in components.values():
            kept = [a for a in members if accounts[a]['distinct_texts'] >= minimum]
            if len(kept)>=2: usable.append(kept)
        capacity[str(minimum)] = {'candidate_components':len(usable), 'accounts':sum(map(len,usable)),
            'different_name_components':sum(len({accounts[a]['vendor'].casefold() for a in c})>1 for c in usable),
            'by_market':dict(Counter(accounts[a]['market'] for c in usable for a in c))}
    write_jsonl(evidence/'grams_link_candidates.jsonl',link_groups)
    write_jsonl(evidence/'registry_profile_candidates.jsonl',profiles)
    write_jsonl(evidence/'candidate_components.jsonl',({'component_uid':uid(*sorted(v)),'accounts':v,'label':'DEPENDENCE_ONLY_NOT_TRUTH'} for v in components.values()))
    print(json.dumps({'phase':'registry_links','groups':len(link_groups),'components':len(components),'seconds':time.monotonic()-start}),flush=True)

    raw_stats = {m:Counter() for m in ['AlphaBay','Dream','Agora']}
    seen = set(); body_evidence = {}; date_range = {}; raw_accounts = set()
    for market,vendor,title,description,source in originals(raw_stats):
        a = uid(market,vendor); raw_accounts.add(a)
        text = title+'\n'+description
        fingerprint = uid(a,text)
        if source['date']:
            value = date_range.setdefault(a,[source['date'],source['date']])
            value[0] = min(value[0],source['date']); value[1] = max(value[1],source['date'])
        if fingerprint in seen:
            raw_stats[market]['repeated_exact_account_text'] += 1; continue
        seen.add(fingerprint); raw_stats[market]['unique_account_text_versions'] += 1
        # Preserve raw text; normalize markup/escaped line breaks only for armor detection.
        normalized = html.unescape(re.sub(r'(?i)<br\s*/?>','\n',text)).replace('\\n','\n').replace('\\r','\r')
        keys = ARMOR.findall(normalized); messages = SIGNED.findall(normalized)
        raw_stats[market]['texts_with_key_begin_marker'] += bool(re.search('BEGIN PGP PUBLIC KEY BLOCK',normalized,re.I))
        raw_stats[market]['texts_with_complete_key_armor'] += bool(keys)
        raw_stats[market]['texts_with_signed_message_marker'] += bool(re.search('BEGIN PGP SIGNED MESSAGE',normalized,re.I))
        raw_stats[market]['texts_with_complete_signed_message'] += bool(messages)
        raw_stats[market]['texts_with_signature_marker'] += bool(re.search('BEGIN PGP SIGNATURE',normalized,re.I))
        claims=[]
        for m in CLAIM.finditer(normalized):
            context=normalized[max(0,m.start()-240):m.end()+350]
            if MARKET_MENTION.search(context): claims.append(context)
        dangers=[normalized[max(0,m.start()-160):m.end()+250] for m in DANGER.finditer(normalized)]
        raw_stats[market]['texts_with_market_identity_claim'] += bool(claims)
        raw_stats[market]['texts_with_imposter_warning'] += bool(dangers)
        if keys or messages or claims or dangers or re.search('BEGIN PGP',normalized,re.I):
            body_evidence[fingerprint]={'account_uid':a,'market':market,'vendor':vendor,'source':source,'raw_title':title,'raw_description':description,
                'public_keys':keys,'signed_messages':messages,'claim_contexts':claims,'warning_contexts':dangers,
                'in_canonical':a in accounts,'label':'UNADJUDICATED_SOURCE_CLAIM'}
    write_jsonl(evidence/'original_text_candidates.jsonl',body_evidence.values())
    write_jsonl(evidence/'observed_account_dates.jsonl',({'account_uid':a,'first_observed':d[0],'last_observed':d[1]} for a,d in date_range.items()))
    counts=Counter(v['account_uid'] for v in registry.values() if v['account_uid'])
    summary={'status':'SOURCE_EVIDENCE_AWAIT_ADJUDICATION','observed_server':datetime.now().astimezone().isoformat(),
        'source_summary_sha256':digest(prior/'summary.json'),'script_sha256':digest(Path(__file__)),
        'grams_link_groups':len(link_groups),'grams_link_different_name_groups':sum(g['different_names'] for g in link_groups),
        'grams_link_groups_with_imposter':sum(g['imposter_members']>0 for g in link_groups),
        'grams_link_groups_by_markets':dict(Counter('+'.join(g['markets']) for g in link_groups)),
        'grams_link_unique_text_accounts':len({v['account_uid'] for g in link_groups for v in g['members']}),
        'canonical_accounts_with_multiple_registry_matches':sum(n>1 for n in counts.values()),
        'registry_profiles':dict(registry_stats),'profile_candidate_count':len(profiles),
        'registry_dates_utc':{k:[datetime.fromtimestamp(min(v),timezone.utc).isoformat(),datetime.fromtimestamp(max(v),timezone.utc).isoformat()] for k,v in times.items()},
        'candidate_connected_components_not_truth':len(components),'conditional_text_capacity_not_truth':capacity,
        'original_scan':{k:dict(v) for k,v in raw_stats.items()},'original_observed_accounts':len(raw_accounts),
        'body_evidence_records':len(body_evidence),'body_evidence_accounts':len({v['account_uid'] for v in body_evidence.values()}),
        'signed_message_accounts':len({v['account_uid'] for v in body_evidence.values() if v['signed_messages']}),
        'dataset_ready':False,'verified_alias_pairs':None,'verified_negative_pairs':None,
        'limitations':['Grams link groups are third-party assertions with unknown original curation, not automatically verified labels.',
            'Original-source marker/claim recall is explicit and approximate; no marker does not prove no alias.',
            'Times are registry metadata or crawl observations, not proof of when a key controlled an account.',
            'Components and capacities describe candidate dependence; they are not independent proven sellers.',
            'No formal partition or training labels generated; unknown relations remain unknown.'],
        'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time(),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(out/'summary.json',summary)
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 2*1024**3: raise ValueError('Output limit exceeded')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prior',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(args.prior,args.output)
