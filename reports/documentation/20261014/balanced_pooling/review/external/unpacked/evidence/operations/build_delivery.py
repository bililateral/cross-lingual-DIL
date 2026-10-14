"""Package the static review and mechanically verify its bytes; no project imports."""
from pathlib import Path
import datetime
import hashlib
import json
import platform
import sys
import zipfile

root = Path(__file__).resolve().parents[2]
out = root.parent
archive = out / 'balanced_pooling_static_review.zip'
receipt_path = out / 'balanced_pooling_review_delivery_receipt.json'
original = out / 'review_input(20261014-043655).zip'

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def file_record(path: Path) -> dict:
    data = path.read_bytes()
    return {'path': path.relative_to(root).as_posix(), 'bytes': len(data), 'sha256': sha(data)}

def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

started = utc()
identity = json.loads((root / 'evidence/input/identity.json').read_text())
execution = json.loads((root / 'evidence/matrix_check/attempt01/execution.json').read_text())
checks = json.loads((root / 'evidence/matrix_check/attempt01/output.json').read_text())
assert original.stat().st_size == identity['bytes']
assert sha(original.read_bytes()) == identity['sha256']
assert sha((root / 'evidence/matrix_check/check_math.py').read_bytes()) == execution['source_sha256']
assert sha((root / 'evidence/matrix_check/inputs.json').read_bytes()) == execution['input_sha256']
assert (root / 'review.zh.md').read_bytes() == (root / 'review.zh.txt').read_bytes()
assert execution['exit_code'] == 0
assert checks['summary']['checks'] == 45 and checks['summary']['passed'] == 45
context = {'started_utc': started, 'python_version': sys.version, 'platform': platform.platform(),
           'activity': 'artifact byte packaging only, not a scientific test or project qualification',
           'original_zip_matches_captured_identity': True,
           'matrix_source_and_input_match_actual_execution': True,
           'markdown_and_text_report_identical': True,
           'manifest_self_excluded': ['MANIFEST.tsv'],
           'external_receipt_inside_zip': False}
(root / 'evidence/operations/packaging_context.json').write_text(
    json.dumps(context, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
files = sorted(p for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix() != 'MANIFEST.tsv')
records = [file_record(p) for p in files]
manifest = root / 'MANIFEST.tsv'
lines = ['# SHA256 file manifest. Self-excluded: MANIFEST.tsv. Paths are relative to ZIP root.',
         'path\tbytes\tsha256']
lines += [f"{r['path']}\t{r['bytes']}\t{r['sha256']}" for r in records]
manifest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
all_files = sorted(files + [manifest])
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in all_files:
        z.write(p, p.relative_to(root).as_posix())
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    assert len(names) == len(set(names)) == len(all_files)
    assert z.testzip() is None
    all_records = [file_record(p) for p in all_files]
    for r in all_records:
        data = z.read(r['path'])
        assert len(data) == r['bytes'] and sha(data) == r['sha256']
    assert set(names) == {r['path'] for r in records} | {'MANIFEST.tsv'}
receipt = {
    'receipt_type': 'local byte delivery receipt for independent static review',
    'created_utc': utc(),
    'self_reported_model': 'GPT-6 Astra Pro',
    'requested_model': 'GPT 6 Pro',
    'selector_independently_verified': False,
    'backend_route_independently_verified': False,
    'third_party_signature_or_remote_receipt': False,
    'decision': 'Worth preparing implementation contract; ordinary OT adaptation/control; no execution approval',
    'original_input': {'filename': original.name, 'bytes': original.stat().st_size,
                       'sha256': sha(original.read_bytes()), 'zip_members': identity['zip_members'],
                       'included_in_return_zip': False},
    'return_zip': {'filename': archive.name, 'bytes': archive.stat().st_size,
                   'sha256': sha(archive.read_bytes()), 'member_count': len(all_records)},
    'internal_manifest': {**file_record(manifest), 'listed_payload_count': len(records),
                          'self_excluded': True, 'exclusions': ['MANIFEST.tsv']},
    'verification': {'zip_crc_ok': True, 'unique_member_paths': True,
                     'manifest_exact_coverage_excluding_self': True,
                     'all_member_bytes_and_sha256_match': True,
                     'original_input_unchanged': True,
                     'matrix_source_and_input_unchanged_since_actual_run': True,
                     'no_project_code_executed': True},
    'files': all_records,
    'receipt_self_excluded': 'This external JSON is outside the ZIP and does not claim its own SHA256.'}
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'return_zip': receipt['return_zip'], 'manifest': receipt['internal_manifest'],
                  'receipt_file': receipt_path.name, 'receipt_bytes': receipt_path.stat().st_size,
                  'receipt_sha256': sha(receipt_path.read_bytes()),
                  'verification': receipt['verification']}, ensure_ascii=False, indent=2))
