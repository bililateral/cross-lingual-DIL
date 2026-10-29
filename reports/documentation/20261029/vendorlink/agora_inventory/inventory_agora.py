"""Linux-only, aggregate Agora source inventory; no split, views or labels."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import resource
import time
import unicodedata

import openpyxl


ROOT = Path('/home/yongpeng/cross-lingual')
SOURCE = ROOT / '2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx'
OUT = ROOT / 'reports/documentation/20261029/vendorlink/agora_inventory/inventory.json'
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
started = time.monotonic()
counts = Counter()
accounts, raw_owners, masked_owners = {}, defaultdict(set), defaultdict(set)
entries, patterns, casefold_handles = {}, {}, defaultdict(set)


def canonical(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def fingerprint(title: str, description: str) -> bytes:
    return hashlib.sha256((title + '\0' + description).casefold().encode()).digest()


def extent(values: list[int]) -> dict:
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'min': values[0], 'max': values[-1],
            'p50': values[(len(values) - 1) // 2],
            'p90': values[int((len(values) - 1) * .9)]}


book = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True)
assert len(book.worksheets) == 1, 'Unexpected workbook structure'
sheet = book.worksheets[0]
rows = sheet.iter_rows(values_only=True)
headers = list(next(rows))
required = ['Vendor', ' Item', ' Item Description']
assert all(headers.count(key) == 1 for key in required), 'Unexpected required columns'
columns = [headers.index(key) for key in required]
for row in rows:
    counts['rows'] += 1
    if not any(value is not None for value in row):
        counts['empty_rows'] += 1
        continue
    values = [row[index] if index < len(row) else None for index in columns]
    if any(not isinstance(value, str) for value in values):
        counts['invalid_type_or_missing'] += 1
        for name, value in zip(required, values):
            if not isinstance(value, str):
                counts['invalid_' + name.strip()] += 1
        continue
    for name, value in zip(required, values):
        counts['at_excel_character_limit_' + name.strip()] += int(len(value) == 32767)
    vendor, title, description = map(canonical, values)
    if not all((vendor, title, description)):
        counts['empty_required'] += 1
        continue
    if '\ufffd' in vendor + title + description:
        counts['replacement_character'] += 1
        continue
    counts['valid_rows'] += 1
    counts['description_contains_html_tag'] += bool(re.search(r'<(?:p|br|div|span|a|b|strong)\b', description, re.I))
    if vendor not in accounts:
        accounts[vendor] = len(accounts)
        patterns[vendor] = re.compile(r'(?<!\w)' + re.escape(vendor) + r'(?!\w)', re.I)
    account = accounts[vendor]
    casefold_handles[vendor.casefold()].add(account)
    masked_title = patterns[vendor].sub('[VENDOR]', title)
    masked_description = patterns[vendor].sub('[VENDOR]', description)
    counts['rows_with_own_handle_masked'] += (masked_title, masked_description) != (title, description)
    raw, masked = fingerprint(title, description), fingerprint(masked_title, masked_description)
    raw_owners[raw].add(account)
    masked_owners[masked].add(account)
    key = (account, masked)
    if key in entries:
        counts['within_account_masked_duplicate_rows'] += 1
        entries[key][0].add(raw)
    else:
        entries[key] = ({raw}, len(title), len(description))
book.close()

clean_counts = Counter()
title_lengths, description_lengths = [], []
for (account, masked), (raws, title_len, description_len) in entries.items():
    if len(masked_owners[masked]) > 1 or any(len(raw_owners[raw]) > 1 for raw in raws):
        counts['excluded_shared_exact_account_texts'] += 1
        continue
    clean_counts[account] += 1
    title_lengths.append(title_len)
    description_lengths.append(description_len)
counts['distinct_account_masked_texts_before_shared_filter'] = len(entries)
counts['clean_distinct_ads'] = sum(clean_counts.values())
assert counts['valid_rows'] == len(entries) + counts['within_account_masked_duplicate_rows']
assert len(entries) == counts['clean_distinct_ads'] + counts['excluded_shared_exact_account_texts']
assert counts['rows'] == sum(counts[key] for key in (
    'valid_rows', 'empty_rows', 'invalid_type_or_missing', 'empty_required', 'replacement_character'))
result = {
    'scope': 'standalone_agora_source_eligibility_no_formal_split',
    'observed_server': datetime.now().astimezone().isoformat(),
    'source': {'path': str(SOURCE), 'bytes': SOURCE.stat().st_size,
               'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'headers': headers, 'field_mapping': dict(zip(('vendor', 'title', 'description'), required)),
    'counts': dict(counts), 'observed_accounts': len(accounts),
    'accounts_with_clean_ads': len(clean_counts),
    'casefold_handle_collision_groups': sum(len(v) > 1 for v in casefold_handles.values()),
    'casefold_handle_collision_accounts': sum(len(v) for v in casefold_handles.values() if len(v) > 1),
    'eligible_accounts_by_min_ads': {str(n): sum(v >= n for v in clean_counts.values()) for n in (4, 10, 20, 50)},
    'ads_in_min20_accounts': sum(v for v in clean_counts.values() if v >= 20),
    'clean_ads_per_observed_account': extent([clean_counts[i] for i in range(len(accounts))]),
    'clean_title_characters': extent(title_lengths),
    'clean_description_characters': extent(description_lengths),
    'limitations': ['No listing URL or product ID in workbook; distinct text is not verified distinct physical product.',
                   'Standalone exact deduplication only; joint cross-market filtering and near duplicates untested.',
                   'Case variants remain separate observed accounts; no verified real-owner aliases.',
                   'XLSX conversion equivalence to original CSV bytes not established.'],
    'no_split_views_qrels_models_gpu': True,
    'wall_seconds': time.monotonic() - started,
    'cpu_seconds': time.process_time(),
    'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
}
with OUT.open('x', encoding='utf-8') as target:
    json.dump(result, target, ensure_ascii=False, indent=2)
    target.write('\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
