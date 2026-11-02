"""One authorized, bounded metadata/byte check; never import project/model code."""
import time
STARTED = time.monotonic()
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT = Path('/home/yongpeng/cross-lingual')
BASE = ROOT/'reports/documentation/20261102/history_optuna'
WORK = BASE/'workspace'
OUT = BASE/'preparation/run01'
CAP, TAIL = 300, 5
FILES = ('groups.csv', 'train/items.jsonl', 'train/supervision/pairs.csv',
         'development/items.jsonl', 'development/supervision/pairs.csv')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def deadline(*_):
    raise TimeoutError('Preparation allowance exhausted; retain evidence, no retry')


def record(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError('File outside the existing project')
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            deadline() if time.monotonic()-STARTED >= CAP-15 else None
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Input changed during byte verification')
    return dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest())


def checked(path, expected):
    row = record(path)
    if any(row[k] != expected[k] for k in ('bytes', 'sha256')):
        raise ValueError('Pinned file differs: '+str(path))
    return row


def main():
    if sys.platform != 'linux' or Path(__file__).resolve() != BASE/'preparation/bind_inputs.py':
        raise ValueError('Only the existing Linux preparation directory is allowed')
    OUT.mkdir()  # No rerun/overwrite of an attempt.
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(CAP-15)
    result = dict(status='RUNNING', authorization='preparation/authorization.json',
                  verified=[], links=[], parsed_formal_text=0, parsed_relationship_labels=0,
                  model_loads=0, gpu_calls=0, formal_execution=False)
    write(OUT/'started.json', dict(server_epoch=time.time(), allowance_seconds=CAP))
    try:
        import psutil
        own = []
        for p in psutil.process_iter(['pid', 'username', 'create_time', 'cmdline', 'cwd']):
            try:
                args = p.info['cmdline'] or []
                if p.info['username'] == 'yongpeng' and any(a.endswith(('active_formal_wait.py', 'step28_forever_formal.py', 'step28_vendorlink_formal.py')) for a in args):
                    own.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        result['observation'] = dict(server_epoch=time.time(), free_ram_bytes=psutil.virtual_memory().available,
                                     processes=own, cpu_affinity=psutil.Process().cpu_affinity())
        if result['observation']['free_ram_bytes'] < 2*2**30:
            raise RuntimeError('Insufficient RAM for bounded byte checks')
        gate = json.loads((BASE/'repair/formal_gate.json').read_text())
        if gate['approved'] or gate['actions'] or (WORK/'formal').exists():
            raise ValueError('Formal execution must stay closed')
        for row in gate['sources']:
            checked(WORK/row['path'], row)
        result['source_files_matched'] = len(gate['sources'])
        policy = json.loads((WORK/'schema/step28_chinese_base_policy.json').read_text())
        original = ROOT/policy['data_root']
        bindings = []
        for name, key in (('manifest.json', 'data_manifest_sha256'), ('validation.json', 'data_validation_sha256')):
            row = record(original/name)
            if row['sha256'] != policy[key]:
                raise ValueError('Dataset metadata differs')
            result['verified'].append(dict(kind='dataset_metadata', **row))
            bindings.append((WORK/policy['data_root']/name, original/name))
        manifest = json.loads((original/'manifest.json').read_text())
        validation = json.loads((original/'validation.json').read_text())
        if manifest['study'] != 'seller_alias_continual_expression_shift' or manifest['root_seed'] != 20260910 or validation['status'] != 'PASS_GENERATION_CONTRACT_NOT_MODEL_QUALIFICATION':
            raise ValueError('Original generation identity differs')
        for name in FILES:
            row = checked(original/name, manifest['files'][name])
            result['verified'].append(dict(kind='input_bytes_only', **row))
            bindings.append((WORK/policy['data_root']/name, original/name))
        model = policy['models']['split_rank']
        archive = ROOT/model['path']
        model_rows = []
        for path in sorted(archive.rglob('*'), key=lambda p: p.relative_to(archive).as_posix()):
            if path.is_file() and '.cache' not in path.parts and '__pycache__' not in path.parts and path.suffix != '.pyc':
                row = record(path)
                model_rows.append(dict(path=path.relative_to(archive).as_posix(), size_bytes=row['bytes'], sha256=row['sha256']))
                result['verified'].append(dict(kind='pretrained_bytes_only', **row))
                bindings.append((WORK/model['path']/path.relative_to(archive), path))
        digest = hashlib.sha256(json.dumps(model_rows, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if (len(model_rows), sum(r['size_bytes'] for r in model_rows), digest) != (model['file_count'], model['total_size_bytes'], model['content_sha256']):
            raise ValueError('Frozen pretrained archive differs')
        old_gate_rec = checked(Path(gate['h_reference_gate']['path']), gate['h_reference_gate'])
        result['verified'].append(dict(kind='h_reference_gate', **old_gate_rec))
        old_gate = json.loads(Path(old_gate_rec['path']).read_text())
        seen, weight_bytes = set(), 0
        for item in gate['h_reference_runs']:
            completion = Path(item['completion']['path'])
            row = checked(completion, item['completion'])
            result['verified'].append(dict(kind='h_completion', **row))
            saved = json.loads(completion.read_text())
            spec = saved['spec']; key = (spec['order'], spec['seed'])
            if spec['name'] != 'H_pos_tuned' or key in seen or saved['updates'] != 864 or saved['status'] != 'COMPLETE_864_UPDATES_3_ENDPOINTS_VALID_BLIND' or saved['source_files'] != old_gate['sources'] or set(saved['points']) != {'1','2','3'}:
                raise ValueError('Original H trajectory identity differs')
            seen.add(key)
            for point in saved['points'].values():
                if not point['full_restore_verified'] or not point['inference_restore_verified']:
                    raise ValueError('Original saved restoration evidence incomplete')
                for key2, kind in (('model', 'h_weight_bytes_only'), ('mapping', 'h_calibration_bytes_only')):
                    row = checked(completion.parent/point[key2]['path'], point[key2])
                    result['verified'].append(dict(kind=kind, **row))
                    if key2 == 'model':
                        weight_bytes += row['bytes']
        if seen != {(o,s) for o in ('ABC','BCA','CAB') for s in (4,5,6)}:
            raise ValueError('All nine original H trajectories required')
        # All identities pass before adding any input paths. No directory links:
        # heldout/owners/private custody are never brought into this workspace.
        for dest, source in bindings:
            if not dest.is_relative_to(WORK) or any(p.is_symlink() for p in dest.parents if p.is_relative_to(WORK)):
                raise ValueError('Unsafe input destination')
            if dest.exists() or dest.is_symlink():
                raise FileExistsError('Fresh input binding expected: '+str(dest))
        for dest, source in bindings:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(source)
            if dest.resolve(strict=True) != source.resolve(strict=True):
                raise ValueError('Input link target differs')
            result['links'].append(dict(path=str(dest), target=str(source.resolve())))
        result.update(status='PASS_BOUND_INPUT_IDENTITIES_EXECUTION_CLOSED', h_trajectories=9,
                      h_weights=27, h_weight_bytes=weight_bytes, pretrained_files=len(model_rows),
                      input_payload_files=len(FILES), formal_campaign_exists=(WORK/'formal').exists())
    except BaseException as exc:
        result.update(status='FAIL_NO_RETRY', error=repr(exc))
        raise
    finally:
        signal.alarm(0)
        elapsed = time.monotonic()-STARTED
        charged = math.ceil(elapsed)+TAIL
        result.update(actual_seconds_before_receipt=elapsed, conservative_tail_seconds=TAIL,
                      charged_seconds=charged, preparation_cap_seconds=CAP, prior_auxiliary_seconds=101,
                      total_auxiliary_seconds=101+charged, server_epoch=time.time())
        if charged > CAP:
            result['status'] = 'FAIL_ALLOWANCE_EXCEEDED'
        write(OUT/'receipt.json', result)
        print(result['status'], charged, flush=True)


if __name__ == '__main__':
    main()
