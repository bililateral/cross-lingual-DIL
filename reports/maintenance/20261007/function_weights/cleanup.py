"""Delete only the nine explicitly authorized completed function-memory weights."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import pwd
import stat

ROOT = Path('/home/yongpeng/cross-lingual')
RUN = ROOT / 'reports/seller_alias_continual/20261006/function_memory_execution/20261006_231800'
RESULT = ROOT / 'reports/seller_alias_continual/20261007/function_memory_result'
OUT = ROOT / 'reports/maintenance/20261007/function_weights'


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metadata(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_uid, s.st_mode]


def main():
    assert os.name == 'posix' and pwd.getpwuid(os.getuid()).pw_name == 'yongpeng'
    assert not (OUT / 'receipt.json').exists()
    assert read(RUN / 'job/completion.json')['status'] == 'COMPLETE_FUNCTION_MEMORY_FIXED_POINT'
    assert 'exit_code=0' in (RUN / 'job.wrapper.txt').read_text()
    inventory = read(RESULT / 'inventory.json')
    assert inventory['remote_run'] == str(RUN)
    expected = {f'job/run/models/{order}_function_memory_stage{stage}.pt'
                for order in ('ABC', 'BCA', 'CAB') for stage in (1, 2, 3)}
    records = inventory['retained_model_files']
    assert len(records) == 9 and {r['path'] for r in records} == expected
    assert sum(r['bytes'] for r in records) == 11758264524
    def check_evidence():
        for record in inventory['files']:
            path = RUN / record['path']
            assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], str(path)
    check_evidence()
    old = read(ROOT / 'reports/maintenance/20261006/relation_revision_closure/preflight.json')
    exceptions = {p: metadata(ROOT / p) for p in old['exceptions']}
    assert len(exceptions) == 23
    weights = []
    for record in records:
        path = RUN / record['path']
        assert path.resolve(strict=True) == path == Path(record['absolute_path'])
        assert path.parent == RUN / 'job/run/models' and not path.is_symlink()
        info = metadata(path)
        assert stat.S_ISREG(info[-1]) and info[4] == os.getuid() == record['uid']
        assert info[2] == record['bytes'] and sha(path) == record['sha256']
        weights.append({**record, 'metadata': info, 'owner': 'yongpeng'})
    receipt = {'status': 'PREPARED', 'prepared_at': now(), 'authorization': '训练权重一起删掉',
               'scope': '20261006_231800 nine stage inference weights only',
               'dependencies': 'Training ended; no model load/recovery planned; result review uses saved matrices/evidence',
               'inventory_sha256': sha(RESULT / 'inventory.json'), 'verified_weights': weights,
               'retained_exception_metadata': exceptions, 'nonweight_files': len(inventory['files']),
               'deleted': [], 'logical_bytes_released': 0}
    write(OUT / 'preflight.json', receipt)
    for record in weights:
        path = Path(record['absolute_path'])
        assert metadata(path) == record['metadata']
        path.unlink()
        assert not path.exists() and not path.is_symlink()
        receipt['deleted'].append({**record, 'deleted_at': now(), 'absent': True})
        receipt['logical_bytes_released'] += record['bytes']
        receipt['status'] = 'DELETING'
        write(OUT / 'receipt.json', receipt)
    check_evidence()
    assert all(metadata(ROOT / p) == value for p, value in exceptions.items())
    assert all(not (RUN / p).exists() for p in expected)
    receipt.update(status='COMPLETE', finished_at=now(), all_targets_absent=True,
                   nonweight_sha256_unchanged=True, retained_exception_metadata_unchanged=True,
                   model_loads=0, label_reads=0, scientific_conclusion_changed=False)
    write(OUT / 'receipt.json', receipt)
    print(json.dumps({k:receipt[k] for k in ('status','finished_at','logical_bytes_released','nonweight_files')}))


if __name__ == '__main__':
    main()
