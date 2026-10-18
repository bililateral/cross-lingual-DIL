#!/usr/bin/env python3
"""Read/hash/diff only. Never imports or executes the supplied project modules."""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import resource
import sys
import time
import traceback
import zipfile

BASE = Path('/workspace/scratch/2456d4553fd5')
ZIP = BASE / 'upload/positive_ranking_repair_review.zip'
ROOT = BASE / 'repair_audit_work/input'
OUT = BASE / 'repair_audit_output'
EXPECTED = {'bytes':394296, 'members':94, 'sha256':'2910ad3d7cea62a334e2fd3d3912dc9b5a9ae163b133111f2e48f783b47b92af'}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation', choices=['verify','read','index'])
    parser.add_argument('paths', nargs='*')
    args=parser.parse_args()
    started=time.perf_counter()
    evidence=OUT/'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    log=evidence/'mechanical_operations.jsonl'
    seq=len(log.read_text().splitlines())+1 if log.exists() else 1
    event={'sequence':seq, 'timestamp_utc':datetime.now(timezone.utc).isoformat(),
           'command_argv':[sys.executable, *sys.argv], 'cwd':os.getcwd(),
           'script_sha256':sha(Path(__file__).read_bytes()), 'python':sys.version,
           'platform':platform.platform(), 'operation':args.operation,
           'scientific_execution':False, 'project_code_imported':False, 'records':[]}
    save(evidence/'pre_execution'/f'{seq:03d}.json', event)
    stdout=io.StringIO(); stderr=io.StringIO(); status=0
    try:
        with contextlib.redirect_stdout(stdout):
            if args.operation=='verify':
                blob=ZIP.read_bytes()
                with zipfile.ZipFile(ZIP) as z:
                    names=z.namelist()
                    actual={'bytes':len(blob), 'members':len(names), 'sha256':sha(blob)}
                    manifest_blob=z.read('package_manifest.json')
                    manifest=json.loads(manifest_blob)
                    listed=manifest['files']
                    listing=[r['path'] for r in listed]
                    assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
                    records=[]
                    for rec in listed:
                        data=z.read(rec['path'])
                        records.append({'path':rec['path'], 'bytes':len(data), 'sha256':sha(data),
                                        'matches':len(data)==rec['bytes'] and sha(data)==rec['sha256']})
                    missing=sorted(set(listing)-set(names))
                    extra=sorted(set(names)-set(listing)-{'package_manifest.json'})
                    passed=(actual==EXPECTED and all(r['matches'] for r in records) and not missing and not extra
                            and len(names)==len(set(names)) and len(listing)==len(set(listing)))
                    result={'input_path':str(ZIP), 'expected':EXPECTED, 'actual':actual,
                            'manifest_payload_count':len(listed),
                            'manifest_itself':{'path':'package_manifest.json','bytes':len(manifest_blob),'sha256':sha(manifest_blob)},
                            'missing_members':missing, 'extra_members':extra,
                            'duplicate_member_count':len(names)-len(set(names)),
                            'duplicate_manifest_path_count':len(listing)-len(set(listing)),
                            'all_passed':passed, 'files':records,
                            'scope':'Byte identity only; not a producing-host, backend-model, GPU-release or performance authenticity claim.'}
                    save(OUT/'input_verification.json', result)
                    assert passed, 'Input identity/member manifest mismatch'
                    for n in names:
                        p=ROOT/n
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(z.read(n))
                    print(json.dumps({k:v for k,v in result.items() if k!='files'}, ensure_ascii=False, indent=2))
                    print('\nINPUT MEMBER INDEX (metadata only):')
                    for rec in records:
                        print(f"{rec['path']}\t{rec['bytes']}")
                    event['records']=[{'path':str(ZIP), **actual, 'kind':'verify_all_members_and_extract'}]
            elif args.operation=='index':
                for p in sorted(ROOT.rglob('*')):
                    if p.is_file():
                        b=p.read_bytes()
                        print(f'{p.relative_to(ROOT)}\t{len(b)}\t{len(b.splitlines())}')
                event['records']=[{'kind':'member_metadata_index_only'}]
            else:
                for spec in args.paths:
                    path, sep, selection=spec.partition('::')
                    p=ROOT/path
                    blob=p.read_bytes(); lines=blob.decode('utf-8').splitlines()
                    ranges=[]
                    if sep:
                        for item in selection.split(','):
                            lo, dash, hi=item.partition('-')
                            ranges.append([int(lo), int(hi) if dash else int(lo)])
                    else:
                        ranges=[[1,len(lines)]]
                    print('\nFILE '+path)
                    for lo,hi in ranges:
                        for i in range(max(lo,1)-1,min(hi,len(lines))):
                            print(f'{i+1:5d} {lines[i]}')
                    event['records'].append({'path':path,'bytes':len(blob),'sha256':sha(blob),
                                             'total_lines':len(lines),'read_ranges':ranges})
    except BaseException:
        status=1
        stderr.write(traceback.format_exc())
    event.update({'exit_code':status, 'wall_seconds':time.perf_counter()-started,
                  'process_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  'stdout_path':f'evidence/outputs/{seq:03d}.stdout.txt',
                  'stderr_path':f'evidence/outputs/{seq:03d}.stderr.txt'})
    logs=evidence/'outputs'; logs.mkdir(parents=True, exist_ok=True)
    (logs/f'{seq:03d}.stdout.txt').write_text(stdout.getvalue(), encoding='utf-8')
    (logs/f'{seq:03d}.stderr.txt').write_text(stderr.getvalue(), encoding='utf-8')
    with log.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False)+'\n')
    sys.stdout.write(stdout.getvalue()); sys.stderr.write(stderr.getvalue())
    return status

if __name__=='__main__':
    raise SystemExit(main())
