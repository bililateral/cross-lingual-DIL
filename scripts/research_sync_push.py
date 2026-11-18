"""Remote side of research_files.ps1 Push. Only SFTP carries file contents."""
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat


def file_identity(path, sizes):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'Regular file required: {path}')
    if info.st_size not in sizes:
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return info.st_size, digest.hexdigest()


def target_action(path, row):
    new = row['bytes'], row['sha256'].lower()
    old = row.get('previous')
    previous = (old['bytes'], old['sha256'].lower()) if old else None
    try:
        current = file_identity(path, {new[0], previous[0] if previous else new[0]})
    except FileNotFoundError:
        if previous:
            raise ValueError(f'Expected previous target is missing: {path}') from None
        return 'create'
    if current == new:
        return 'skip'
    if previous and current == previous:
        return 'replace'
    raise ValueError(f'Remote identity conflict: {path}')


def push(request):
    root = Path(request['root']).resolve(strict=True)
    token = request['token']
    if not root.is_dir() or not re.fullmatch('[0-9a-f]{32}', token):
        raise ValueError('Invalid remote root or transfer token')
    rows = request['files']
    targets = []
    for row in rows:
        relative = Path(row['path'])
        target = root / relative
        if (relative.is_absolute() or '..' in relative.parts or relative.as_posix() != row['path']
                or not relative.name or target.parent.resolve(strict=True) != target.parent):
            raise ValueError(f'Invalid target: {relative}')
        targets.append(target)
    if len(set(targets)) != len(targets):
        raise ValueError('Duplicate targets')
    stages = [t.parent / ('.research-sync-' + token) / str(i) for i, t in enumerate(targets)]
    phase = request['phase']
    if phase == 'prepare':
        pending = [i for i, (t, row) in enumerate(zip(targets, rows)) if target_action(t, row) != 'skip']
        made = []
        try:
            for folder in sorted({stages[i].parent for i in pending}):
                folder.mkdir(mode=0o700)
                made.append(folder)
        except OSError:
            for folder in reversed(made):
                folder.rmdir()
            raise
        return {'pending': pending}

    pending = set(request['pending'])
    folders = sorted({stages[i].parent for i in pending})
    if phase not in ('publish', 'discard') or any(d.is_symlink() for d in folders):
        raise ValueError('Invalid staging directory or phase')
    try:
        if phase == 'publish':
            # Other Push calls serialize publication in shared destination directories.
            with ExitStack() as locks:
                for parent in sorted({t.parent for t in targets}):
                    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                    locks.callback(os.close, fd)
                    fcntl.flock(fd, fcntl.LOCK_EX)
                actions = [target_action(t, row) for t, row in zip(targets, rows)]
                for i, (row, action) in enumerate(zip(rows, actions)):
                    if action != 'skip':
                        wanted = row['bytes'], row['sha256'].lower()
                        if i not in pending or file_identity(stages[i], {wanted[0]}) != wanted:
                            raise ValueError(f'Staged identity differs: {targets[i]}')
                for i, action in enumerate(actions):
                    if action == 'create':
                        os.link(stages[i], targets[i])  # Never overwrite a newly appeared target.
                    elif action == 'replace':
                        stages[i].chmod(stat.S_IMODE(targets[i].stat().st_mode))
                        os.replace(stages[i], targets[i])
            return {'actions': actions}
        return {}
    finally:
        for i in pending:
            stages[i].unlink(missing_ok=True)
        for folder in folders:
            if folder.exists():
                folder.rmdir()
