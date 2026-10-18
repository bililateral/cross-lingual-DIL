#!/usr/bin/env python3
"""Static byte/text inspection only. Never imports or executes project modules."""
from datetime import datetime, timezone
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import resource
import sys
import time
import traceback
import zipfile

BASE=Path('/workspace/scratch/2456d4553fd5')
ZIP=BASE/'upload/cublas_workspace_review.zip'
ROOT=BASE/'cublas_audit_work/input'
OUT=BASE/'cublas_audit_output'
EXPECTED={'bytes':459440,'members':111,'sha256':'5a35e220d570779f88418385682bf22e64a9bf5b016a1cea78f2586335b768e7'}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    started=time.perf_counter(); operation=sys.argv[1]
    evidence=OUT/'evidence'; evidence.mkdir(parents=True,exist_ok=True)
    log=evidence/'operations.jsonl'
    sequence=len(log.read_text().splitlines())+1 if log.exists() else 1
    event={'sequence':sequence,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
           'command_argv':[sys.executable,*sys.argv],'cwd':os.getcwd(),
           'script_sha256':sha(Path(__file__).read_bytes()),'python':sys.version,'platform':platform.platform(),
           'operation':operation,'project_code_executed':False,'scientific_computation':False,'records':[]}
    save(evidence/'pre_execution'/f'{sequence:03d}.json',event)
    output=io.StringIO(); errors=io.StringIO(); status=0
    try:
        with contextlib.redirect_stdout(output):
            if operation=='verify':
                blob=ZIP.read_bytes()
                with zipfile.ZipFile(ZIP) as z:
                    names=z.namelist(); actual={'bytes':len(blob),'members':len(names),'sha256':sha(blob)}
                    assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
                    manifest_data=z.read('package_manifest.json'); manifest=json.loads(manifest_data)
                    listing=manifest['files']; listed_paths=[r['path'] for r in listing]
                    rows=[]
                    for row in listing:
                        b=z.read(row['path'])
                        rows.append({'path':row['path'],'bytes':len(b),'sha256':sha(b),
                                     'matches':len(b)==row['bytes'] and sha(b)==row['sha256']})
                    result={'input_path':str(ZIP),'expected':EXPECTED,'actual':actual,
                            'manifest_payload_count':len(rows),'manifest_itself':{'path':'package_manifest.json','bytes':len(manifest_data),'sha256':sha(manifest_data)},
                            'missing':sorted(set(listed_paths)-set(names)),
                            'extra':sorted(set(names)-set(listed_paths)-{'package_manifest.json'}),
                            'duplicate_zip_paths':len(names)-len(set(names)),
                            'duplicate_manifest_paths':len(listed_paths)-len(set(listed_paths)),
                            'zip_crc_ok':z.testzip() is None,'files':rows}
                    result['all_passed']=(actual==EXPECTED and not result['missing'] and not result['extra'] and
                                          not result['duplicate_zip_paths'] and not result['duplicate_manifest_paths'] and
                                          result['zip_crc_ok'] and all(r['matches'] for r in rows))
                    result['scope']='Identity and internal correspondence only; not producing-host, backend-model, GPU or performance authentication.'
                    save(OUT/'input_verification.json',result)
                    assert result['all_passed']
                    for name in names:
                        p=ROOT/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(z.read(name))
                    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
                    print('\nINPUT MEMBER INDEX:')
                    for row in rows: print(row['path']+'\t'+str(row['bytes']))
                    event['records']=[{'path':str(ZIP),**actual,'scope':'all_member_identity_and_extract'}]
            elif operation=='read':
                for spec in sys.argv[2:]:
                    relative,separator,selection=spec.partition('::')
                    p=ROOT/relative; b=p.read_bytes(); lines=b.decode('utf-8').splitlines()
                    ranges=[]
                    if separator:
                        for item in selection.split(','):
                            lo,dash,hi=item.partition('-'); ranges.append([int(lo),int(hi) if dash else int(lo)])
                    else: ranges=[[1,len(lines)]]
                    print('\nFILE '+relative)
                    for lo,hi in ranges:
                        for i in range(max(1,lo)-1,min(hi,len(lines))): print(f'{i+1:5d} {lines[i]}')
                    event['records'].append({'path':relative,'bytes':len(b),'sha256':sha(b),'total_lines':len(lines),'read_ranges':ranges})
            elif operation=='search':
                pattern=sys.argv[2]; regex=re.compile(pattern)
                for relative in sys.argv[3:]:
                    p=ROOT/relative; b=p.read_bytes(); hits=[]
                    for number,line in enumerate(b.decode('utf-8').splitlines(),1):
                        if regex.search(line): hits.append(number); print(f'{relative}:{number}:{line}')
                    event['records'].append({'path':relative,'bytes':len(b),'sha256':sha(b),'pattern':pattern,'hit_lines':hits})
            else:
                raise ValueError('Only verify/read/search supported')
    except BaseException:
        status=1; errors.write(traceback.format_exc())
    event.update({'exit_code':status,'wall_seconds':time.perf_counter()-started,
                  'process_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  'stdout_path':f'evidence/outputs/{sequence:03d}.stdout.txt','stderr_path':f'evidence/outputs/{sequence:03d}.stderr.txt'})
    (evidence/'outputs').mkdir(parents=True,exist_ok=True)
    (OUT/event['stdout_path']).write_text(output.getvalue(),encoding='utf-8')
    (OUT/event['stderr_path']).write_text(errors.getvalue(),encoding='utf-8')
    with log.open('a',encoding='utf-8') as f: f.write(json.dumps(event,ensure_ascii=False)+'\n')
    sys.stdout.write(output.getvalue()); sys.stderr.write(errors.getvalue())
    return status

if __name__=='__main__': raise SystemExit(main())
