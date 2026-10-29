"""Inventory publisher CSV and compare required fields to the existing workbook."""
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import io
from itertools import zip_longest
import json
from pathlib import Path
import re
import resource
import time
import unicodedata
import zipfile

import openpyxl

ROOT = Path('/home/yongpeng/cross-lingual')
ARCHIVE = ROOT / 'data/vendorlink/agora_20261029/kaggle.zip'
WORKBOOK = ROOT / '2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx'
OUT = ROOT / 'reports/documentation/20261029/vendorlink/agora_inventory/csv_inventory.json'
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
csv.field_size_limit(4 * 1024**2)
started = time.monotonic()
counts, comparison, lengths = Counter(), Counter(), Counter()
accounts, patterns, entries = {}, {}, {}
raw_owners, masked_owners = defaultdict(set), defaultdict(set)


def canonical(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def fingerprint(title: str, description: str) -> bytes:
    return hashlib.sha256((title + '\0' + description).casefold().encode()).digest()


book = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
xrows = book.active.iter_rows(values_only=True)
xheaders = list(next(xrows))
with zipfile.ZipFile(ARCHIVE) as archive:
    assert archive.namelist() == ['Agora.csv']
    csv_bytes = archive.getinfo('Agora.csv').file_size
    with archive.open('Agora.csv') as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding='ISO-8859-1', newline=''))
        headers = next(reader)
        assert headers == xheaders
        columns = [headers.index(k) for k in ('Vendor', ' Item', ' Item Description')]
        for row, xrow in zip_longest(reader, xrows):
            if row is None:
                comparison['workbook_extra_rows'] += 1
                continue
            counts['rows'] += 1
            if len(row) != len(headers):
                counts['invalid_width_rows'] += 1
                continue
            values = [row[i] for i in columns]
            if xrow is None:
                comparison['csv_extra_rows'] += 1
            else:
                xvalues = [xrow[i] for i in columns]
                if values == xvalues:
                    comparison['required_fields_exact_equal_rows'] += 1
                else:
                    comparison['required_fields_differ_rows'] += 1
                    for name, cvalue, xvalue in zip(('vendor', 'title', 'description'), values, xvalues):
                        if cvalue != xvalue:
                            comparison[name + '_different_' + type(xvalue).__name__] += 1
            vendor, title, description = map(canonical, values)
            if not all((vendor, title, description)):
                counts['missing_required_rows'] += 1
                continue
            if '\ufffd' in vendor + title + description:
                counts['replacement_character_rows'] += 1
                continue
            counts['valid_rows'] += 1
            lengths[len(description)] += 1
            counts['description_ends_three_dots'] += description.endswith('...')
            counts['title_length_gt_512'] += len(title) > 512
            if vendor not in accounts:
                accounts[vendor] = len(accounts)
                patterns[vendor] = re.compile(r'(?<!\w)' + re.escape(vendor) + r'(?!\w)', re.I)
            account = accounts[vendor]
            raw_key = fingerprint(title, description)
            masked = fingerprint(patterns[vendor].sub('[VENDOR]', title), patterns[vendor].sub('[VENDOR]', description))
            raw_owners[raw_key].add(account)
            masked_owners[masked].add(account)
            key = (account, masked)
            if key in entries:
                counts['within_account_masked_duplicate_rows'] += 1
            entries.setdefault(key, set()).add(raw_key)
book.close()
clean = Counter()
for (account, masked), raws in entries.items():
    if len(masked_owners[masked]) > 1 or any(len(raw_owners[k]) > 1 for k in raws):
        counts['excluded_shared_exact_account_texts'] += 1
    else:
        clean[account] += 1
counts['distinct_account_masked_texts_before_shared_filter'] = len(entries)
counts['clean_distinct_ads'] = sum(clean.values())
assert counts['valid_rows'] == len(entries) + counts['within_account_masked_duplicate_rows']
assert len(entries) == counts['clean_distinct_ads'] + counts['excluded_shared_exact_account_texts']
assert counts['rows'] == sum(counts[k] for k in ('valid_rows', 'missing_required_rows', 'invalid_width_rows', 'replacement_character_rows'))
result = {'scope': 'standalone_publisher_agora_csv_no_split',
    'observed_server': datetime.now().astimezone().isoformat(),
    'source_url': 'https://www.kaggle.com/api/v1/datasets/download/philipjames11/dark-net-marketplace-drug-data-agora-20142015',
    'archive': {'path': str(ARCHIVE), 'bytes': ARCHIVE.stat().st_size,
                'sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()},
    'member': {'name': 'Agora.csv', 'bytes': csv_bytes, 'encoding': 'ISO-8859-1'},
    'headers': headers, 'counts': dict(counts), 'workbook_comparison': dict(comparison),
    'observed_accounts': len(accounts), 'accounts_with_clean_ads': len(clean),
    'eligible_accounts_by_min_ads': {str(n): sum(v >= n for v in clean.values()) for n in (4, 10, 20, 50)},
    'ads_in_min20_accounts': sum(v for v in clean.values() if v >= 20),
    'description_length_top10': lengths.most_common(10),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'no_split_views_qrels_models_gpu': True,
    'limitations': ['Standalone exact dedup only; cross-market and near duplicates untested.',
                   'No product IDs or verified real-owner labels.',
                   'Downloaded Kaggle release is pinned by hash, not proved identical to paper experiment bytes.'],
    'wall_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time(),
    'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
with OUT.open('x', encoding='utf-8') as target:
    json.dump(result, target, ensure_ascii=False, indent=2)
    target.write('\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
