"""Validate only supplied package bytes, contents, and immutable local extraction."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile

root = Path('/workspace/scratch/0f307e99d302')
archive = root / 'upload/positive_ranking_result_review.zip'
extracted = root / 'review_input'
payload = archive.read_bytes()
manifest = json.loads((extracted / 'manifest.json').read_text())
records = []
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    assert len(names) == 389 and len(names) == len(set(names))
    assert not any(n.startswith('/') or '..' in Path(n).parts for n in names)
    assert z.testzip() is None
    assert z.read('manifest.json') == (extracted / 'manifest.json').read_bytes()
    for item in manifest:
        b = (extracted / item['path']).read_bytes()
        assert z.read(item['path']) == b
        actual = {'path': item['path'], 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
        assert actual == item
        records.append(actual)
    assert set(names) == {r['path'] for r in records} | {'manifest.json'}
assert len(payload) == 5481875
assert hashlib.sha256(payload).hexdigest() == '4eeeb5df0e93b6c32bf352f42b656b945a8d4e4f71078b73fa4df0e015ae63f7'
result = {'status': 'IDENTITY_AND_ALL_PAYLOADS_MATCH', 'archive_path': str(archive),
          'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
          'members': len(names), 'payloads': len(records), 'zip_crc_check': 'PASS',
          'python': sys.version, 'platform': platform.platform(),
          'dependencies': {n: importlib.metadata.version(n) for n in ('numpy', 'matplotlib')}}
out = root / 'audit_output/input_identity.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
(out.parent / 'input_inventory_verified.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
