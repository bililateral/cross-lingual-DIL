"""Delete only the 27 inventoried, completed FOREVER weights after result review closes."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

import psutil

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/home/yongpeng/cross-lingual/reports/documentation/20261101/formal_chain/forever/training')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def processes():
    found = []
    for process in psutil.process_iter(['pid', 'username', 'create_time', 'cmdline', 'status']):
        try:
            info = process.info
            args = info['cmdline'] or []
            if info['username'] == 'yongpeng' and any('/formal_chain/' in a for a in args):
                found.append({**info, 'cwd': process.cwd()})
                if info['status'] != psutil.STATUS_ZOMBIE and any(a.endswith(('step28_forever_formal.py', 'step28_forever_evaluate.py')) for a in args):
                    raise RuntimeError('FOREVER process remains active')
        except psutil.NoSuchProcess:
            pass
    return found


def check_retained(manifest, inventory):
    checked = 0
    for record in manifest['files']:
        if record['local'].startswith('chain/'):
            continue  # This is the still-running VendorLink controller, not a frozen FOREVER result.
        path = Path(record['remote'])
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], str(path)
        checked += 1
    model_paths = {r['path'] for r in inventory['files']}
    metadata = 0
    for record in manifest['excluded_metadata']:
        if record['path'] in model_paths:
            continue
        assert Path(record['path']).stat().st_size == record['bytes'], record['path']
        metadata += 1  # Memory and sealed generated material are stat-only, never opened.
    for record in inventory['retained_reference']:
        actual = Path(record['path']).stat()
        assert (actual.st_size, actual.st_ino, actual.st_dev, actual.st_mtime_ns) == (
            record['bytes'], record['inode'], record['device'], record['mtime_ns'])
    return dict(size_sha_files=checked, metadata_only_files=metadata, retained_H_models=27)


def main():
    started = time.monotonic()
    assert os.name == 'posix' and ROOT == Path('/home/yongpeng/cross-lingual/reports/seller_alias_continual/20261103/forever_result')
    assert read(ROOT/'review/submission.json')['status'] == 'EXTERNAL_AND_PRIMARY_RESULT_REVIEW_CLOSED'
    assert (ROOT/'review/primary_disposition.zh.txt').is_file()
    assert not (ROOT/'cleanup_receipt.json').exists(), 'No automatic repeat deletion attempt'
    inventory, manifest = read(ROOT/'cleanup_inventory.json'), read(ROOT/'transfer_manifest.json')
    assert Path(inventory['root']).resolve() == CAMPAIGN.resolve()
    rows = inventory['files']
    assert len(rows) == 27 and sum(r['bytes'] for r in rows) == 35218653363
    done = read(CAMPAIGN/'completion.json')
    assert done['status'] == 'ALL_FORMAL_ACTIONS_COMPLETE' and len(done['completed']) == 12
    assert not done['active'] and all(r['success'] for r in done['completed'])
    before_processes = processes()
    retained = check_retained(manifest, inventory)
    for record in rows:
        path = Path(record['path'])
        actual = path.lstat()
        assert path.resolve() == Path(record['resolved']) and path.resolve().is_relative_to(CAMPAIGN.resolve())
        assert path.parent.name == 'models' and path.parent.parent.name == 'run'
        assert stat.S_ISREG(actual.st_mode) and not path.is_symlink() and actual.st_uid == os.getuid() and actual.st_nlink == 1
        assert (actual.st_size, actual.st_ino, actual.st_dev, actual.st_mtime_ns) == (
            record['bytes'], record['inode'], record['device'], record['mtime_ns'])
        assert sha(path) == record['sha256']
    output = ROOT/'cleanup_receipt.json'
    result = dict(status='DELETING', started_epoch=time.time(), inventory_sha256=sha(ROOT/'cleanup_inventory.json'),
                  targets=rows, deleted=[], retained_before=retained, processes_before=before_processes)
    def save():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    save()
    try:
        for record in rows:
            path = Path(record['path'])
            current = path.lstat()
            assert current.st_ino == record['inode'] and current.st_size == record['bytes'] and current.st_nlink == 1
            path.unlink()
            result['deleted'].append(record['path'])
            save()
        assert all(not Path(r['path']).exists() for r in rows)
        assert not list(CAMPAIGN.glob('*/run/models/*.pt'))
        result.update(status='DELETED_AND_RETAINED_ASSETS_VERIFIED', deleted_files=27,
                      deleted_unique_logical_bytes=sum(r['bytes'] for r in rows), retained_after=check_retained(manifest, inventory),
                      processes_after=processes(), finished_epoch=time.time(), elapsed_seconds=time.monotonic()-started)
        save()
    except BaseException as error:
        result.update(status='FAILED_NO_AUTOMATIC_RETRY', error=repr(error), elapsed_seconds=time.monotonic()-started)
        save()
        raise
    print(json.dumps({k:result[k] for k in ('status','deleted_files','deleted_unique_logical_bytes','elapsed_seconds')}))


if __name__ == '__main__':
    main()
