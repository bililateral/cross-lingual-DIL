"""Authorized source preparation and aggregate split preview, never model IO.

The canonical source cache stays on Linux. This entry deliberately does not
construct views, qrels or a training permit; final split export is a later step.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import tarfile
import time
import zipfile

from step29_vendorlink_data import canonical, preview_counts


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_sources(grams: Path, agora: Path) -> tuple[list[dict], dict]:
    """Earliest complete Grams listing; CSV text fields remain strings."""
    csv.field_size_limit(16 * 1024**2)
    stats = {market: Counter() for market in ('AlphaBay', 'Dream', 'Agora')}
    records, ambiguous, dates = {}, set(), {'AlphaBay': set(), 'Dream': set()}
    with tarfile.open(grams, 'r:xz') as archive:
        for member in archive:
            market = {'Alpha.csv': 'AlphaBay', 'Dream.csv': 'Dream'}.get(Path(member.name).name)
            if not member.isfile() or market is None:
                continue
            s = stats[market]
            date = member.name.split('/')[1]
            dates[market].add(date)
            s['snapshot_files'] += 1
            with archive.extractfile(member) as source:
                reader = csv.DictReader(io.TextIOWrapper(source, encoding='utf-8-sig', errors='replace', newline=''))
                required = ('hash', 'market_name', 'item_link', 'vendor_name', 'name', 'description')
                if not set(required).issubset(reader.fieldnames):
                    raise ValueError('Unexpected Grams required schema')
                for number, row in enumerate(reader, 2):
                    s['parsed_rows'] += 1
                    if None in row or any(not isinstance(row.get(k), str) for k in required):
                        s['malformed_rows'] += 1
                        continue
                    allowed = {'AlphaBay': ('alphabay', 'alpha bay'), 'Dream': ('dream', 'dream market')}[market]
                    if canonical(row['market_name']).casefold() not in allowed:
                        s['wrong_market_rows'] += 1
                        continue
                    product, vendor, title, description = [canonical(row[k]) for k in ('item_link', 'vendor_name', 'name', 'description')]
                    if not all((product, vendor, title, description)):
                        s['missing_required_rows'] += 1
                        continue
                    if '\ufffd' in vendor + title + description:
                        s['replacement_character_rows'] += 1
                        continue
                    s['valid_rows'] += 1
                    key, rank = (market, product), (date, member.name, number)
                    value = {'market': market, 'vendor': vendor, 'title': title, 'description': description,
                             'source_key': hashlib.sha256(json.dumps(key).encode()).hexdigest(), 'source_date': date}
                    if key in records:
                        s['repeated_listing_rows'] += 1
                        if records[key][1]['vendor'] != vendor:
                            ambiguous.add(key)
                        if rank < records[key][0]:
                            records[key] = rank, value
                    else:
                        records[key] = rank, value
    rows = []
    for key, (_, value) in sorted(records.items()):
        stats[key[0]]['distinct_listing_keys'] += 1
        if key in ambiguous:
            stats[key[0]]['ambiguous_account_listings_removed'] += 1
        else:
            rows.append(value)
    with zipfile.ZipFile(agora) as archive:
        if archive.namelist() != ['Agora.csv']:
            raise ValueError('Expected the pinned single-file Kaggle Agora ZIP')
        with archive.open('Agora.csv') as source:
            reader = csv.DictReader(io.TextIOWrapper(source, encoding='ISO-8859-1', newline=''))
            required = ('Vendor', ' Item', ' Item Description')
            if not set(required).issubset(reader.fieldnames):
                raise ValueError('Unexpected Agora required schema')
            s = stats['Agora']
            for number, row in enumerate(reader, 2):
                s['parsed_rows'] += 1
                if None in row or any(not isinstance(row.get(k), str) for k in required):
                    s['malformed_rows'] += 1
                    continue
                vendor, title, description = [canonical(row[k]) for k in required]
                if not all((vendor, title, description)):
                    s['missing_required_rows'] += 1
                    continue
                if '\ufffd' in vendor + title + description:
                    s['replacement_character_rows'] += 1
                    continue
                s['valid_rows'] += 1
                rows.append({'market': 'Agora', 'vendor': vendor, 'title': title, 'description': description,
                             'source_key': f'Agora.csv:{number}'})
    for market, s in stats.items():
        s['canonical_rows'] = sum(row['market'] == market for row in rows)
    return rows, {'markets': stats, 'dates': {m: {'first': min(v), 'last': max(v), 'snapshots': len(v)}
                                            for m, v in dates.items() if v}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or args.output.exists():
        raise ValueError('Linux and a fresh preparation directory required')
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['operation'] != 'approved_market_source_preparation_and_proposed_split_counts':
        raise ValueError('This entry cannot authorize training or label generation')
    if config['markets'] != ['AlphaBay', 'Dream', 'Agora']:
        raise ValueError('Only the three user-selected markets are in scope')
    for source in config['sources'].values():
        path = Path(source['path'])
        if path.stat().st_size != source['bytes'] or file_hash(path) != source['sha256']:
            raise ValueError('Source archive identity differs')
    args.output.mkdir(parents=True)
    rows, source_report = read_sources(Path(config['sources']['grams']['path']), Path(config['sources']['agora']['path']))
    result = preview_counts(rows, **config['proposed_partition'])
    source_cache = args.output/'canonical.jsonl'
    with source_cache.open('x', encoding='utf-8') as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False)+'\n')
    result.update({'observed_server': datetime.now().astimezone().isoformat(), 'source_report': source_report,
        'config_sha256': file_hash(args.config), 'sources': config['sources'],
        'canonical_cache': {'path': str(source_cache), 'bytes': source_cache.stat().st_size,
                            'sha256': file_hash(source_cache), 'linux_only': True},
        'script_sha256': file_hash(Path(__file__)),
        'data_module_sha256': file_hash(Path(__file__).with_name('step29_vendorlink_data.py')),
        'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time(),
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'no_model_gpu_views_qrels': True})
    (args.output/'inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'SOURCE_PREPARATION_AND_PREVIEW_COMPLETE', 'markets': result['markets'],
                      'wall_seconds': result['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
