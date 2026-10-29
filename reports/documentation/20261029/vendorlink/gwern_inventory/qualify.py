"""Linux-only inventory of authorized public ads. Never builds views or qrels."""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import resource
import time
import unicodedata
from urllib.parse import urlsplit

from lxml import html

ROOT = Path('/home/yongpeng/cross-lingual')
DATA = ROOT / 'data/vendorlink/gwern_20261029'
OUT = ROOT / 'reports/documentation/20261029/vendorlink/gwern_inventory'
resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (900, 900))
csv.field_size_limit(16*1024**2)
started = time.monotonic()
stats = {m: collections.Counter() for m in ('AlphaBay', 'Dream', 'Silk Road')}
account_ids, patterns, records, ambiguous_products = {}, {}, {}, set()
first_source_ids, source_id_conflicts, seller_id_names = {}, set(), collections.defaultdict(set)
dates, headers, market_values = collections.defaultdict(set), {}, collections.Counter()
files, failures, shapes = collections.Counter(), collections.Counter(), []


def canonical(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def fingerprint(title, description):
    return hashlib.sha256((title+'\0'+description).casefold().encode()).digest()


def add(market, vendor, title, description, product, rank, source_id='', seller_id=''):
    s = stats[market]
    s['raw_rows_or_product_pages'] += 1
    values = (vendor, title, description, product)
    if any(not isinstance(v, str) for v in values):
        s['invalid_type'] += 1
        return
    vendor, title, description, product = map(canonical, values)
    if not all((vendor, title, description, product)):
        s['missing_required'] += 1
        for name, value in zip(('vendor', 'title', 'description', 'product'), (vendor, title, description, product)):
            if not value:
                s['missing_'+name] += 1
        return
    if '\ufffd' in vendor+title+description:
        s['replacement_character'] += 1
        return
    s['valid_rows_or_pages'] += 1
    if re.search(r'<(?:p|br|div|span|a|b|strong)\b', description, re.I):
        s['description_contains_html_tag'] += 1
    key = (market, vendor)
    if key not in account_ids:
        account_ids[key] = len(account_ids)
        patterns[key] = re.compile(r'(?<!\w)'+re.escape(vendor)+r'(?!\w)', re.IGNORECASE)
    account = account_ids[key]
    raw = fingerprint(title, description)
    masked = fingerprint(patterns[key].sub('[VENDOR]', title), patterns[key].sub('[VENDOR]', description))
    product_key = (market, product)
    if source_id:
        source_key = (market, source_id)
        old = first_source_ids.setdefault(source_key, product)
        if old != product:
            source_id_conflicts.add(source_key)
    if seller_id:
        seller_id_names[(market, seller_id)].add(vendor)
    value = (rank, account, raw, masked, len(title), len(description))
    if product_key in records:
        prior = records[product_key]
        s['repeated_product_occurrences'] += 1
        if prior[1] != account:
            ambiguous_products.add(product_key)
        if prior[2] != raw:
            s['occurrences_different_from_retained_text'] += 1
        if rank < prior[0]:
            records[product_key] = value
    else:
        records[product_key] = value


import tarfile
with tarfile.open(DATA/'grams-20150714-20160417.tar.xz', 'r:xz') as archive:
    for member in archive:
        name = Path(member.name).name
        market = {'Alpha.csv': 'AlphaBay', 'Dream.csv': 'Dream'}.get(name)
        if not member.isfile() or market is None:
            continue
        date = member.name.split('/')[1]
        dates[market].add(date)
        files[market] += 1
        with archive.extractfile(member) as source:
            reader = csv.DictReader(io.TextIOWrapper(source, encoding='utf-8-sig', errors='replace', newline=''))
            headers[name] = reader.fieldnames
            required = {'hash', 'market_name', 'item_link', 'vendor_name', 'name', 'description'}
            if not required.issubset(reader.fieldnames):
                raise ValueError('Unexpected target CSV schema')
            for number, row in enumerate(reader, 2):
                if None in row or any(row.get(k) is None for k in required):
                    stats[market]['malformed_csv_rows'] += 1
                    continue
                actual = canonical(row['market_name'])
                market_values[(market, actual)] += 1
                allowed = {'AlphaBay': {'alphabay', 'alpha bay'}, 'Dream': {'dream', 'dream market'}}[market]
                if actual.casefold() not in allowed:
                    stats[market]['unexpected_market_rows'] += 1
                    continue
                add(market, row['vendor_name'], row['name'], row['description'], row['item_link'],
                    (date, 0, member.name, number), source_id=row['hash'])
        print(json.dumps({'market': market, 'snapshot': date, 'files': files[market],
                          'valid_rows': stats[market]['valid_rows_or_pages'],
                          'unique_products_all_markets': len(records)}), flush=True)

item_pattern = re.compile(r'/silkroad/item/([0-9a-f]+)(?:/[^/]*)?$', re.I)
with tarfile.open(DATA/'silkroad1-20130703-anonymous.tar.xz', 'r:xz') as archive:
    for member in archive:
        match = item_pattern.search(member.name)
        if not member.isfile() or match is None or Path(member.name).name == 'Descr.WD3':
            continue
        files['Silk Road_item_path_files'] += 1
        if member.size == 0:
            failures['empty_item_file'] += 1
            continue
        with archive.extractfile(member) as source:
            data = source.read()
        if len(data) < 1000:
            failures['short_or_redirect_item_file'] += 1
            continue
        tree = html.fromstring(data)
        content = tree.xpath('//*[@id="content"]')
        if len(content) != 1:
            failures['missing_or_ambiguous_content'] += 1
            continue
        content = content[0]
        titles = content.xpath('.//*[contains(concat(" ", normalize-space(@class), " "), " h1 ")]')
        sellers = content.xpath('.//span[contains(concat(" ", normalize-space(@class), " "), " container_small ")]//a[contains(@href,"/user/")]')
        descriptions = content.xpath('.//div[contains(concat(" ", normalize-space(@class), " "), " container_large ")][.//*[contains(concat(" ", normalize-space(@class), " "), " h3 ")]]')
        descriptions = [n for n in descriptions if any(canonical(h.text_content()).casefold() == 'description' for h in n.xpath('.//*[contains(concat(" ",normalize-space(@class)," ")," h3 ")]'))]
        if len(titles) != 1 or len(sellers) != 1 or len(descriptions) != 1:
            failures['ambiguous_title_seller_description'] += 1
            continue
        title = titles[0].text_content()
        vendor = sellers[0].text_content()
        seller_match = re.search(r'/user/([^/?#]+)', sellers[0].get('href', ''))
        if seller_match is None:
            failures['missing_seller_id'] += 1
            continue
        description_node = descriptions[0]
        for node in description_node.xpath('.//*[contains(concat(" ",normalize-space(@class)," ")," h3 ")] | .//script | .//style'):
            node.drop_tree()
        description = ' '.join(description_node.itertext())
        if len(shapes) < 4:
            shapes.append({'title_chars': len(canonical(title)), 'description_chars': len(canonical(description)),
                           'seller_anchor_in_product_container': True, 'seller_distinct_from_welcome':
                           canonical(vendor) not in [canonical(n.text_content()) for n in tree.xpath('//*[@id="welcome_name"]')]})
        dates['Silk Road'].add('2013-07-03')
        add('Silk Road', vendor, title, description, match.group(1),
            ('2013-07-03', member.name.count('/'), member.name, 0), seller_id=seller_match.group(1))

# Each listing contributes its earliest valid complete snapshot, never a separate
# ad per date/feedback page. Inventory convention only, not a frozen training set.
raw_owners, masked_owners = {}, {}
for product, (_, account, raw, masked, _, _) in records.items():
    if product in ambiguous_products:
        continue
    for mapping, key in ((raw_owners, raw), (masked_owners, masked)):
        if key in mapping and mapping[key] != account:
            mapping[key] = -1
        else:
            mapping.setdefault(key, account)
account_ads = collections.defaultdict(set)
title_lengths, description_lengths = collections.defaultdict(list), collections.defaultdict(list)
for product, (_, account, raw, masked, title_len, desc_len) in records.items():
    market = product[0]
    stats[market]['distinct_listing_keys'] += 1
    if product in ambiguous_products:
        stats[market]['listing_keys_with_multiple_accounts'] += 1
        continue
    if raw_owners[raw] == -1 or masked_owners[masked] == -1:
        stats[market]['cross_account_shared_listing_records_removed'] += 1
        continue
    account_ads[account].add(masked)
    title_lengths[market].append(title_len)
    description_lengths[market].append(desc_len)


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    return {'min': values[0], 'median': values[(len(values)-1)//2],
            'p90': values[int(.9*(len(values)-1))], 'max': values[-1]}


for market, s in stats.items():
    values = [len(account_ads[i]) for (m, _), i in account_ids.items() if m == market]
    s.update(observed_accounts_with_valid_rows=len(values), clean_distinct_text_ads=sum(values),
             eligible_accounts_ge20=sum(v >= 20 for v in values),
             clean_ads_in_eligible_accounts=sum(v for v in values if v >= 20),
             accounts_ge4=sum(v >= 4 for v in values), accounts_ge10=sum(v >= 10 for v in values),
             source_ids_with_multiple_links=sum(m == market for m, _ in source_id_conflicts))
output = {'status': 'INVENTORY_COMPLETE_NOT_FORMAL_DATA_APPROVAL',
          'observed_server': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
          'input_identity': json.loads((OUT/'transfer.json').read_text())['files'],
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'stats': stats, 'files': files, 'headers': headers,
          'actual_market_values': [{'file_market': m, 'value': v, 'rows': n} for (m, v), n in market_values.items()],
          'dates': {m: {'min': min(v), 'max': max(v), 'snapshots': len(v)} for m, v in dates.items()},
          'silkroad_extraction_failures': failures, 'silkroad_structural_examples_without_text': shapes,
          'silkroad_seller_ids': len(seller_id_names),
          'silkroad_seller_ids_with_multiple_names': sum(len(v) > 1 for v in seller_id_names.values()),
          'clean_ads_per_account': {m: distribution([len(account_ads[i]) for (market, _), i in account_ids.items() if market == m]) for m in stats},
          'title_chars': {m: distribution(v) for m, v in title_lengths.items()},
          'description_chars': {m: distribution(v) for m, v in description_lengths.items()},
          'convention': 'earliest valid complete snapshot per exact item_link (Grams) or item path ID (SR1); remove ambiguous seller IDs, account-shared raw/masked exact text, and within-account duplicates',
          'limitations': ['not original CMU/IMPACT version', 'no verified cross-account owners',
                         'no near-duplicate or model-token analysis', 'no formal split/views/qrels/metrics/model/GPU'],
          'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time(),
          'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(OUT/'qualification.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
