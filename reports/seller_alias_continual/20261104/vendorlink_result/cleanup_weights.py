"""Remove only retired BCA/CAB VendorLink endpoints; preserve ABC and evidence."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from datetime import datetime

import psutil

ROOT = Path('/home/yongpeng/cross-lingual/reports/seller_alias_continual/20261104/vendorlink_result')
TRAIN = Path('/home/yongpeng/cross-lingual/reports/documentation/20261101/formal_chain/vendorlink/training')
HISTORY = Path('/home/yongpeng/cross-lingual/reports/documentation/20261103/history_wait')
PATTERN = re.compile(r'(ABC|BCA|CAB)_s[456]_(VendorLink_SEQ|H_SEQ|H_ER|H_full)')
# These five existing OS/login processes were inspected before cleanup. Their
# protected /proc cwd/fd entries are not made readable by running as their uid.
IDENTITY_ONLY = {
    10679: (1792572023.31, ['(sd-pam)']),
    10782: (1792572023.63, ['fusermount3', '-o', 'rw,nosuid,nodev,fsname=portal,auto_unmount,subtype=portal', '--', '/run/user/1014/doc']),
    415544: (1793798428.83, ['sshd:', 'yongpeng@pts/1']),
    417250: (1793798850.19, ['sshd:', 'yongpeng@notty']),
    417251: (1793798850.27, ['/usr/lib/openssh/sftp-server']),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def identity(path):
    actual = path.lstat()
    assert path.resolve() == path and stat.S_ISREG(actual.st_mode), str(path)
    assert actual.st_uid == os.getuid() and actual.st_nlink == 1, str(path)
    return dict(path=str(path), bytes=actual.st_size, inode=actual.st_ino,
                device=actual.st_dev, mtime_ns=actual.st_mtime_ns,
                uid=actual.st_uid, nlink=actual.st_nlink,
                allocated_bytes=actual.st_blocks * 512)


def processes(targets):
    active_history = []
    identity_only = []
    for process in psutil.process_iter(['pid', 'ppid', 'username', 'create_time', 'cmdline', 'status']):
        try:
            info = process.info
            if info['username'] != 'yongpeng' or info['status'] == psutil.STATUS_ZOMBIE:
                continue
            args = info['cmdline'] or []
            cwd = process.cwd()
            assert not cwd.startswith(str(TRAIN)), 'Active process in retired campaign'
            assert not any(str(TRAIN) in arg or arg.endswith(('step29_vendorlink_formal.py',
                'step29_vendorlink_evaluate.py')) for arg in args), 'VendorLink process remains active'
            assert not any(f.path in targets for f in process.open_files()), 'Target is open'
            if any(str(HISTORY) in arg for arg in args):
                active_history.append({**info, 'cwd': cwd})
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            assert (info['create_time'], args) == IDENTITY_ONLY.get(info['pid']), info
            identity_only.append(info)
    return dict(active_history=active_history, identity_only=identity_only)


def preserved(manifest, closeout, protected):
    for row in manifest['files']:
        path = Path(row['path'])
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], str(path)
    for row in closeout['files']:
        path = ROOT / row['relative']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], str(path)
    for row in protected:
        actual = Path(row['path']).stat()
        assert (actual.st_size, actual.st_ino, actual.st_dev, actual.st_mtime_ns) == (
            row['bytes'], row['inode'], row['device'], row['mtime_ns']), row['path']
    return dict(original_files_size_sha=len(manifest['files']),
                review_files_size_sha=len(closeout['files']), chinese_H_metadata=len(protected))


def main():
    started = time.monotonic()
    receipt_path = ROOT / 'cleanup_receipt.json'
    assert os.name == 'posix' and not receipt_path.exists(), 'No repeat cleanup attempt'
    done = read(TRAIN / 'completion.json')
    assert done['status'] == 'ALL_FORMAL_ACTIONS_COMPLETE' and not done['active']
    assert len(done['completed']) == 38 and all(row['success'] for row in done['completed'])
    assert read(ROOT / 'review/recovery.json')['status'] == 'FULL_EXTERNAL_RECOVERY_AND_TARGETED_SAVED_FIELD_CHECK_COMPLETE'
    manifest = read(ROOT / 'receive_manifest.json')
    closeout = read(ROOT / 'review/closeout_sync.json')
    protected = read(ROOT.parent.parent / '20261103/forever_result/cleanup_inventory.json')['retained_reference']
    assert len(manifest['files']) == 811 and len(closeout['files']) == 17 and len(protected) == 27
    rows = []
    for row in manifest['models']:
        path = Path(row['path'])
        relative = path.relative_to(TRAIN)
        assert len(relative.parts) == 3 and relative.parts[1] == 'run'
        assert PATTERN.fullmatch(relative.parts[0]) and relative.parts[2] in ('stage1.pt', 'stage2.pt', 'stage3.pt')
        current = identity(path)
        assert current['bytes'] == row['bytes']
        rows.append({**current, 'sha256': row['sha256'], 'run': relative.parts[0],
                     'action': 'retain' if relative.parts[0].startswith('ABC_') else 'delete'})
    targets = [row for row in rows if row['action'] == 'delete']
    kept = [row for row in rows if row['action'] == 'retain']
    assert len(targets) == 72 and len(kept) == 36
    assert sum(r['bytes'] for r in targets) == 31323821100 and sum(r['bytes'] for r in kept) == 15661910550
    assert len({(r['device'], r['inode']) for r in rows}) == 108
    assert set(TRAIN.rglob('*.pt')) == {Path(r['path']) for r in rows}
    active_before = processes({r['path'] for r in targets})
    retained_before = preserved(manifest, closeout, protected)
    for index, row in enumerate(rows, 1):
        assert sha(Path(row['path'])) == row['sha256'], row['path']
        if index % 18 == 0:
            print(json.dumps(dict(weights_sha_checked=index)), flush=True)
    inventory = dict(root=str(TRAIN), server_time=datetime.now().astimezone().isoformat(),
                     targets=targets, retained_ABC=kept, retained_chinese_H=protected,
                     process_checks=active_before, nonweight_checks=retained_before)
    write(ROOT / 'cleanup_inventory.json', inventory)
    result = dict(status='DELETING', started_at=datetime.now().astimezone().isoformat(),
                  inventory_sha256=sha(ROOT / 'cleanup_inventory.json'),
                  deleted=[], preserved_before=retained_before, processes_before=active_before,
                  free_bytes_before=psutil.disk_usage(str(TRAIN)).free)
    write(receipt_path, result)
    try:
        processes({r['path'] for r in targets})
        for row in targets:
            path = Path(row['path'])
            actual = identity(path)
            assert actual == {k: row[k] for k in actual}, str(path)
            path.unlink()
            result['deleted'].append(row['path'])
            write(receipt_path, result)
        assert set(TRAIN.rglob('*.pt')) == {Path(r['path']) for r in kept}
        for row in kept:
            actual = identity(Path(row['path']))
            assert actual == {k: row[k] for k in actual}, row['path']
        result.update(status='DELETED_AND_RETAINED_ASSETS_VERIFIED', deleted_files=72,
                      deleted_logical_bytes=sum(r['bytes'] for r in targets),
                      deleted_allocated_bytes=sum(r['allocated_bytes'] for r in targets),
                      retained_ABC_files=36, retained_ABC_bytes=sum(r['bytes'] for r in kept),
                      preserved_after=preserved(manifest, closeout, protected),
                      processes_after=processes({r['path'] for r in targets}),
                      free_bytes_after=psutil.disk_usage(str(TRAIN)).free,
                      finished_at=datetime.now().astimezone().isoformat())
    except BaseException as error:
        result.update(status='FAILED_NO_AUTOMATIC_RETRY', error=repr(error))
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        write(receipt_path, result)
    print(json.dumps({k: result[k] for k in ('status', 'deleted_files', 'deleted_logical_bytes',
        'retained_ABC_files', 'elapsed_seconds')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
