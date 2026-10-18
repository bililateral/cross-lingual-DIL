"""Reproducible attachment / snapshot / saved-field checks. Standard library only.

No import, execution, metric reduction, bootstrap, or loading of project models.
"""
from pathlib import Path, PurePosixPath
import difflib
import hashlib
import json
import tarfile
import traceback
import zipfile

ROOT=Path('/workspace/scratch/3e46faac4c35/optuna_review_input')
OUT=Path('/workspace/scratch/3e46faac4c35/optuna_static_review')
ZIP=Path('/workspace/scratch/3e46faac4c35/upload/optuna_implementation_review.zip')
Q='reports/documentation/20261018/optuna_implementation/qualification/'

def sha(value): return hashlib.sha256(value).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def write(name,value):
    path=OUT/'mechanical'/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    errors=[];rows=[]
    with zipfile.ZipFile(ZIP) as archive:
        names=archive.namelist();manifest=json.loads(archive.read('manifest.json'))
        if len(names)!=len(set(names)):errors.append('duplicate ZIP members')
        expected={r['path']:r for r in manifest['files']}
        if set(names)!=set(expected)|{'manifest.json'}:errors.append('manifest member set mismatch')
        for name in names:
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name:errors.append('unsafe member: '+name)
        for name,record in expected.items():
            value=archive.read(name)
            rows.append(dict(path=name,bytes=len(value),sha256=sha(value),
                matches_manifest=len(value)==record['bytes'] and sha(value)==record['sha256']))
            if (ROOT/name).read_bytes()!=value:errors.append('extracted bytes differ: '+name)
    actual=dict(path=str(ZIP),bytes=ZIP.stat().st_size,sha256=sha(ZIP.read_bytes()),members=len(names),payloads=len(rows),
        expected_identity_matches=(ZIP.stat().st_size==2011821 and sha(ZIP.read_bytes())=='52bad7b321d0fa77de36ae7126213da815ba968afeb7d2aa74699f282f373672' and len(names)==141),
        errors=errors,payload_checks=rows)
    # Keep the original identity record as well; independently reproduced result has its own path.
    write('input_identity_reproduced.json',actual)
    if errors or not actual['expected_identity_matches'] or not all(r['matches_manifest'] for r in rows):
        raise ValueError('Input identity or extraction mismatch')

    selected=['attempt05/actual_objective_vjp.json','attempt01/seeds_memory_restore.json',
      'attempt04/six_call_fixture.json','attempt06/saved_evaluation.json','attempt01/budgets.json','attempt01/entry.json']
    write('saved_cpu_fields.json',{p:read(ROOT/(Q+p)) for p in selected})

    for name in ('attempt01','attempt02','attempt04','attempt05'):
        with tarfile.open(ROOT/(Q+name+'_sources.tar.gz'),'r:gz') as archive:
            old={m.name.removeprefix('./'):archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
        diffs=[]
        for path,original in sorted(old.items()):
            current=(ROOT/path).read_bytes()
            if current==original:continue
            diffs.extend(difflib.unified_diff(original.decode('utf-8-sig').splitlines(True),
                current.decode('utf-8-sig').splitlines(True),fromfile=name+'/'+path,tofile='current/'+path))
        dest=OUT/'mechanical/snapshot_diffs'/f'{name}_to_current.diff'
        dest.parent.mkdir(exist_ok=True);dest.write_text(''.join(diffs),encoding='utf-8')

    checks=read(OUT/'mechanical/received_checksum_correspondence.json')
    payloads=[r['actual_path'] for batch in checks[:2] for r in batch['records']]
    counts=dict(payload_count=len(payloads),unique_payload_count=len(set(payloads)),
      payload_bytes=sum((ROOT/p).stat().st_size for p in set(payloads)),total_files=len(set(payloads))+2,
      total_bytes=sum((ROOT/p).stat().st_size for p in set(payloads))+
        sum((ROOT/(Q+p)).stat().st_size for p in ('receive.sha256','receive_delta.sha256')))
    write('received_payload_totals.json',counts)
    if counts!=dict(payload_count=47,unique_payload_count=47,payload_bytes=1418670,total_files=49,total_bytes=1422971):
        raise ValueError('Received file totals differ')
    print(json.dumps(dict(input_identity_matches=True,payload_hashes=len(rows),received=counts,
        project_execution=False),ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        path=OUT/'execution/failures.json'
        failures=read(path) if path.exists() else []
        failures.append(dict(reader='root',operation='supplemental_mechanical_checks.py',kind='review_mechanical_failure',
            traceback=traceback.format_exc(),scientific_code_executed=False))
        path.write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        raise
