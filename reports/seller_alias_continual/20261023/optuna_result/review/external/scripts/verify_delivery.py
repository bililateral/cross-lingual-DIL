"""Verify the delivered ZIP or its extracted directory against file_manifest.json."""
from pathlib import Path, PurePosixPath
import hashlib, json, sys, zipfile

target = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
z = None
if target.is_dir():
    names = {str(p.relative_to(target)).replace('\\', '/') for p in target.rglob('*') if p.is_file()}
    read = lambda name: (target / name).read_bytes()
else:
    z = zipfile.ZipFile(target)
    members = z.namelist()
    if len(members) != len(set(members)):
        raise SystemExit('Duplicate ZIP member')
    if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in members):
        raise SystemExit('Unsafe ZIP member')
    bad = z.testzip()
    if bad:
        raise SystemExit('ZIP CRC failure: ' + bad)
    names = set(members)
    read = z.read
manifest = json.loads(read('file_manifest.json'))
expected = {r['path'] for r in manifest['files']} | {'file_manifest.json'}
failures = []
if names != expected:
    failures.append({'missing': sorted(expected-names), 'unexpected': sorted(names-expected)})
for r in manifest['files']:
    if r['path'] not in names:
        continue
    data = read(r['path'])
    if len(data) != r['bytes'] or hashlib.sha256(data).hexdigest() != r['sha256']:
        failures.append({'mismatch': r['path']})
if z:
    z.close()
print(json.dumps({'status': 'PASS' if not failures else 'FAIL', 'member_count': len(names),
                  'hashed_payloads': len(manifest['files']), 'failures': failures}, ensure_ascii=False, indent=2))
raise SystemExit(1 if failures else 0)
