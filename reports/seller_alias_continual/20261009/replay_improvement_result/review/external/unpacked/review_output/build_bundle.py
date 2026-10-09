"""Package the completed review without copying the original input archive.

The manifest cannot hash itself. SHA256SUMS covers payloads plus the manifest;
the external delivery manifest covers the ZIP and directly delivered files.
ZIP timestamps are normalized metadata, not claimed execution timestamps.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
DEST = BASE / 'deliverables'
DEST.mkdir(exist_ok=True)
EXCLUDE = {'BUNDLE_MANIFEST.json', 'SHA256SUMS.txt'}

def identity(path, raw=None):
    raw = path.read_bytes() if raw is None else raw
    return {'path': str(path.relative_to(BASE)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}

payload = {}
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
        continue
    if path.name in EXCLUDE:
        continue
    payload[path] = path.read_bytes()

manifest = {
    'description': '2026-10-09 independent five-arm results review; current instance only',
    'created_at_UTC': datetime.now(timezone.utc).isoformat(),
    'computation_environment': 'ChatGPT web review workspace; not the project Linux server',
    'input_archive': json.loads((ROOT/'input_identity.json').read_text()),
    'manifest_rule': 'Files below exclude BUNDLE_MANIFEST.json and SHA256SUMS.txt. SHA256SUMS.txt covers these payloads plus BUNDLE_MANIFEST.json; no self-hash is asserted.',
    'original_input_copied': False,
    'payload_count': len(payload),
    'payload_bytes': sum(map(len, payload.values())),
    'files': [identity(p, b) for p, b in payload.items()],
}
mpath = ROOT / 'BUNDLE_MANIFEST.json'
mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
payload[mpath] = mpath.read_bytes()
spath = ROOT / 'SHA256SUMS.txt'
spath.write_text(''.join(f"{hashlib.sha256(raw).hexdigest()}  {path.relative_to(BASE)}\n"
                         for path, raw in sorted(payload.items())), encoding='utf-8')
payload[spath] = spath.read_bytes()

zpath = DEST / 'FIVE_ARM_REVIEW_20261009.zip'
with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path, raw in sorted(payload.items()):
        info = zipfile.ZipInfo(str(path.relative_to(BASE)), date_time=(2026,10,9,0,0,0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

with zipfile.ZipFile(zpath) as archive:
    assert archive.testzip() is None
    assert len(archive.infolist()) == len(payload)
    assert set(archive.namelist()) == {str(p.relative_to(BASE)) for p in payload}
    for path, raw in payload.items():
        assert archive.read(str(path.relative_to(BASE))) == raw
    packed_manifest = json.loads(archive.read('review_output/BUNDLE_MANIFEST.json'))
    for rec in packed_manifest['files']:
        raw = archive.read(rec['path'])
        assert len(raw) == rec['bytes']
        assert hashlib.sha256(raw).hexdigest() == rec['sha256']
    for line in archive.read('review_output/SHA256SUMS.txt').decode().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest

direct = [zpath, ROOT/'REVIEW_REPORT.zh.txt', ROOT/'READING_SCOPE.zh.txt',
          ROOT/'BUNDLE_MANIFEST.json', ROOT/'SHA256SUMS.txt']
delivery = {
    'description': 'Final delivery identity and completed ZIP verification',
    'created_at_UTC': datetime.now(timezone.utc).isoformat(),
    'self_hash_rule': 'DELIVERY_MANIFEST.json excludes its own hash; the final chat may state it separately.',
    'zip_verification': {
        'status': 'PASS', 'crc_errors': 0, 'member_count': len(payload),
        'all_member_bytes_equal_to_final_payload': True,
        'all_payload_sha256_and_sizes_verified': True,
        'all_SHA256SUMS_entries_verified': True,
        'normalized_zip_member_timestamp': '2026-10-09 00:00:00; metadata only',
    },
    'files': [identity(p) for p in direct],
}
dpath = DEST / 'DELIVERY_MANIFEST.json'
dpath.write_text(json.dumps(delivery, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'status':'PASS', 'zip_members':len(payload),
                  'files':delivery['files'], 'external_manifest':identity(dpath)},
                 ensure_ascii=False, indent=2))
