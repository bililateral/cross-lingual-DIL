"""Authorized archive/schema inventory on Linux py310; no training or splits."""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import resource
import tarfile
import time

from lxml import html

ROOT = Path('/home/yongpeng/cross-lingual')
DATA = ROOT / 'data/vendorlink/gwern_20261029'
OUT = ROOT / 'reports/documentation/20261029/vendorlink/gwern_inventory'
resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
started = time.monotonic()
result = {'scope': 'archive_structure_only_no_views_qrels_or_models', 'archives': {}}
for name in ('grams-20150714-20160417.tar.xz', 'silkroad1-20130703-anonymous.tar.xz'):
    path = DATA / name
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    item = {'bytes': path.stat().st_size, 'sha256': h.hexdigest(), 'files': 0,
            'declared_uncompressed_bytes': 0, 'extensions': collections.Counter(),
            'prefixes': collections.Counter(), 'examples': [], 'html_shapes': [], 'csv_headers': []}
    with tarfile.open(path, 'r:xz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            item['files'] += 1
            item['declared_uncompressed_bytes'] += member.size
            suffix = Path(member.name).suffix.lower()
            item['extensions'][suffix] += 1
            item['prefixes']['/'.join(member.name.split('/')[:2])] += 1
            if len(item['examples']) < 16:
                item['examples'].append({'name': member.name, 'bytes': member.size})
            if suffix == '.csv' and len(item['csv_headers']) < 4:
                with archive.extractfile(member) as f:
                    line = f.readline(65536).decode('utf-8-sig', errors='replace')
                row = next(csv.reader([line]))
                # Headers only; prevent a headerless row from exposing actual text.
                known = {'hash', 'market_name', 'item_link', 'vendor_name', 'price',
                         'name', 'description', 'image_link', 'add_time', 'ship_from'}
                item['csv_headers'].append({'member': member.name, 'columns': row if set(row) & known else ['UNRECOGNIZED_HEADER'], 'count': len(row)})
            if suffix in ('.html', '.htm', '') and len(item['html_shapes']) < 5 and 100 < member.size < 2*1024**2:
                with archive.extractfile(member) as f:
                    data = f.read()
                if b'<html' in data.lower() or b'<!doctype' in data.lower():
                    tree = html.fromstring(data)
                    nodes = collections.Counter((n.tag, n.get('class', ''), n.get('id', ''))
                                                for n in tree.iter() if isinstance(n.tag, str))
                    item['html_shapes'].append({'member': member.name,
                        'tags_classes_ids': [[*key, count] for key, count in nodes.most_common(45)],
                        'title_chars': len(' '.join(tree.xpath('//title/text()'))),
                        'text_chars': len(tree.text_content())})
    result['archives'][name] = item
result.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time(),
              peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(OUT / 'structure.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
