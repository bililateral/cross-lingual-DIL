"""Targeted aggregate check of observed long-title/200-character-description issue."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import resource
import time
import unicodedata

import openpyxl

ROOT = Path('/home/yongpeng/cross-lingual')
SOURCE = ROOT / '2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx'
OUT = ROOT / 'reports/documentation/20261029/vendorlink/agora_inventory/field_extent.json'
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
start = time.monotonic()
book = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True)
rows = book.active.iter_rows(values_only=True)
headers = list(next(rows))
cols = [headers.index(x) for x in ('Vendor', ' Item', ' Item Description')]
counts, lengths, invalid_types = Counter(), Counter(), Counter()
for row in rows:
    values = [row[i] for i in cols]
    for name, value in zip(('vendor', 'title', 'description'), values):
        if not isinstance(value, str):
            invalid_types[name + ':' + type(value).__name__] += 1
    if any(not isinstance(value, str) for value in values):
        continue
    vendor, title, description = [' '.join(unicodedata.normalize('NFKC', x).split()) for x in values]
    if not all((vendor, title, description)) or '\ufffd' in vendor + title + description:
        continue
    counts['valid_rows'] += 1
    lengths[len(description)] += 1
    for name, value in (('title', title), ('description', description)):
        for threshold in (10, 200, 256, 512, 1000):
            counts[name + '_length_gt_' + str(threshold)] += len(value) > threshold
    counts['title_longer_than_description'] += len(title) > len(description)
    counts['description_ends_three_dots'] += description.endswith('...')
    counts['description_ends_ellipsis'] += description.endswith('…')
    counts['description_has_literal_backslash_n'] += '\\n' in description
book.close()
result = {'observed_server': datetime.now().astimezone().isoformat(),
          'scope': 'aggregate_field_lengths_only_no_new_cleaning_or_selection',
          'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'counts': dict(counts), 'invalid_field_types': dict(invalid_types),
          'description_length_top10': lengths.most_common(10),
          'wall_seconds': time.monotonic() - start, 'cpu_seconds': time.process_time(),
          'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
with OUT.open('x', encoding='utf-8') as target:
    json.dump(result, target, ensure_ascii=False, indent=2)
    target.write('\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
