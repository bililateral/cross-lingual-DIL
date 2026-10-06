"""Exact-file retirement of the user-cancelled revision; no model/data loading."""
import hashlib
import json
import os
from pathlib import Path
import pwd
import sys
from datetime import datetime, timezone, timedelta

ROOT = Path('/home/yongpeng/cross-lingual')
OUT = ROOT / 'reports/maintenance/20261006/relation_revision_closure'
RESULT = ROOT / 'reports/seller_alias_continual/20261006/relation_revision_result'
RUN = ROOT / 'reports/seller_alias_continual/20261006/relation_revision_execution/20261006_135937'
WORK = RUN / 'workspace'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def metadata(p):
    s = p.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_uid, s.st_mode]

def identity(p):
    assert p.is_file() and not p.is_symlink() and p.resolve() == p
    assert p.is_relative_to(ROOT) and p.stat().st_uid == os.getuid()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p),
                owner=pwd.getpwuid(p.stat().st_uid).pw_name, metadata=metadata(p))

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

plan = read(OUT / 'plan.json')
inv = read(RESULT / 'inventory.json')
roots = [ROOT, ROOT / 'reports/documentation/20261006/relation_revision/cpu',
         ROOT / 'reports/documentation/20261006/relation_revision/k1',
         ROOT / 'reports/documentation/20261006/relation_revision_pilot/workspace', WORK]

def targets():
    selected = set()
    for base in roots:
        for item in plan['remove_active']:
            p = base / item['path']
            if p.exists():
                selected.add(p)
        for sub in ('scripts', 'tests'):
            cache = base / sub / '__pycache__'
            if cache.is_dir():
                selected.update(cache.glob('*relation_revision*.pyc'))
    for name in ('result_verify.py', 'result_analyze.py'):
        p = RUN / name
        if p.exists():
            selected.add(p)
    return sorted(selected)

def assert_no_active():
    matches = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if 'relation_revision' in cmd and 'relation_revision_closure/cleanup.py' not in cmd:
            matches.append(dict(pid=proc.name, command=cmd))
    assert not matches, matches

assert_no_active()
if sys.argv[1] == 'inspect':
    assert not (OUT / 'receipt.json').exists()
    weight_root = WORK / 'reports/job/run/models'
    expected = inv['retained_model_files']
    assert len(expected) == 9 and sum(x['bytes'] for x in expected) == 11758415337
    assert set(weight_root.iterdir()) == {Path(x['path']) for x in expected}
    weights = []
    for item in expected:
        p = Path(item['path'])
        assert p.parent == weight_root and p.resolve().parent == weight_root.resolve()
        row = identity(p)
        assert (row['bytes'], row['sha256']) == (item['bytes'], item['sha256'])
        weights.append(row)
    protected = {}
    for item in inv['files']:
        p = RESULT / item['path']
        assert p.stat().st_size == item['bytes'] and digest(p) == item['sha256'], str(p)
        protected[str(p)] = item['sha256']
        if item['path'].startswith('job/'):
            q = WORK / 'reports' / item['path']
            assert digest(q) == item['sha256']
            protected[str(q)] = item['sha256']
    for item in inv['retained_memory_files']:
        p = Path(item['path'])
        assert p.stat().st_size == item['bytes'] and digest(p) == item['sha256']
        protected[str(p)] = item['sha256']
    for item in plan['originals']:
        if item['path'] != plan['restore_path']:
            assert digest(ROOT / item['path']) == item['sha256']
            protected[str(ROOT / item['path'])] = item['sha256']
    old = read(ROOT / 'reports/maintenance/20261006/relation_weights/receipt.json')
    exceptions = {p: metadata(ROOT / p) for p in old['retained_exception_metadata']}
    restore_before = identity(ROOT / plan['restore_path'])
    assert restore_before['sha256'] == digest(RESULT / 'result_verify.py')
    pre = dict(status='VERIFIED_BEFORE_DELETE', at=now(), authorization=plan['authorization'],
               dependency='User explicitly retires revision and cancels future load/recovery; no active revision process. Pending result review uses saved evidence, not weights; not claimed closed.',
               weights=weights, code=[identity(p) for p in targets()],
               protected=protected, exceptions=exceptions, restore_before=restore_before,
               restore_after_sha256=digest(OUT / 'original_result_verify.py'))
    save('preflight.json', pre)
    print(json.dumps(dict(status=pre['status'], weights=len(weights), weight_bytes=sum(x['bytes'] for x in weights), code=len(pre['code']), protected=len(protected), exceptions=len(exceptions))))
elif sys.argv[1] == 'apply':
    pre = read(OUT / 'preflight.json')
    assert not (OUT / 'receipt.json').exists()
    assert {str(p) for p in targets()} == {x['path'] for x in pre['code']}
    for row in pre['weights'] + pre['code'] + [pre['restore_before']]:
        assert metadata(Path(row['path'])) == row['metadata']
    for p, sha in pre['protected'].items():
        assert digest(Path(p)) == sha
    assert all(metadata(ROOT / p) == meta for p, meta in pre['exceptions'].items())
    receipt = dict(status='DELETING', started_at=now(), authorization=plan['authorization'], deleted=[])
    save('receipt.json', receipt)
    for kind, rows in [('weight', pre['weights']), ('code', pre['code'])]:
        for row in rows:
            p = Path(row['path'])
            assert p.resolve() == p and metadata(p) == row['metadata']
            p.unlink()
            receipt['deleted'].append(dict(**row, kind=kind, deleted_at=now(), absent=not p.exists()))
            save('receipt.json', receipt)
    (ROOT / plan['restore_path']).write_bytes((OUT / 'original_result_verify.py').read_bytes())
    assert digest(ROOT / plan['restore_path']) == pre['restore_after_sha256']
    assert all(digest(Path(p)) == sha for p, sha in pre['protected'].items())
    assert all(metadata(ROOT / p) == meta for p, meta in pre['exceptions'].items())
    assert not targets() and all(not Path(r['path']).exists() for r in receipt['deleted'])
    receipt.update(status='COMPLETE', finished_at=now(), weight_count=9,
                   weight_logical_bytes=sum(x['bytes'] for x in pre['weights']),
                   code_count=len(pre['code']), code_bytes=sum(x['bytes'] for x in pre['code']),
                   protected_hashes_unchanged=True, protected_count=len(pre['protected']),
                   retained_exception_metadata_unchanged=True, retained_exceptions=len(pre['exceptions']),
                   original_files_verified=plan['originals'], all_targets_absent=True,
                   models_loaded=False, formal_labels_parsed=False,
                   historical_source_records_preserved=True)
    save('receipt.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('deleted','original_files_verified','authorization')}))
else:
    raise ValueError('inspect or apply required')
