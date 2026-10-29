"""Export a summary without displaced ad text; retain original Linux evidence."""
import hashlib
import json
from pathlib import Path

out = Path('/home/yongpeng/cross-lingual/reports/documentation/20261029/vendorlink/gwern_inventory')
source = out / 'qualification.json'
original = source.read_bytes()
data = json.loads(original)
for row in data['actual_market_values']:
    if row['value'] not in ('Alphabay', 'Dream'):
        row['value'] = '<unexpected_market_value_redacted>'
data['output_redaction'] = 'Unexpected market values redacted; all counts unchanged; original JSON/log retained on Linux only.'
target = out / 'qualification_summary.json'
with target.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(data, ensure_ascii=False, indent=2))
receipt = {'original_json': {'bytes': len(original), 'sha256': hashlib.sha256(original).hexdigest()},
           'original_log': {'bytes': (out/'qualification.log').stat().st_size,
                            'sha256': hashlib.sha256((out/'qualification.log').read_bytes()).hexdigest()},
           'summary_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
           'originals_retained_linux': True, 'scientific_counts_unchanged': True}
with (out/'summary_export.json').open('x') as stream:
    stream.write(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
