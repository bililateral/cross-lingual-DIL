"""Independent archive, manifest, source identity and local environment audit.
No project training/data/API calls; only reads supplied archives and installed metadata.
"""
import sys, json, hashlib, zipfile, importlib.metadata as md, platform
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT.parent/'relation_revision_pilot_review.zip'
def sha(b): return hashlib.sha256(b).hexdigest()
def verify_zip(p, extracted, expected=None):
    raw=p.read_bytes()
    with zipfile.ZipFile(p) as z:
        names=z.namelist(); infos=z.infolist()
        assert len(names)==len(set(names)), 'duplicate member'
        for i in infos:
            path=PurePosixPath(i.filename)
            assert not path.is_absolute() and '..' not in path.parts
            assert ((i.external_attr >> 16) & 0o170000) != 0o120000
        assert z.testzip() is None
        records=json.loads(z.read('manifest.json'))
        if isinstance(records,dict):
            records=records.get('files',records.get('payloads',records.get('items')))
        indexed={r['path'] for r in records}
        actual={i.filename for i in infos if not i.is_dir()}
        assert indexed==actual-{'manifest.json'}
        assert len(indexed)==len(records)
        for r in records:
            b=z.read(r['path']); assert len(b)==r.get('bytes',r.get('size_bytes',r.get('size')))
            assert sha(b)==r['sha256']
            assert (extracted/r['path']).read_bytes()==b
        if expected: assert sha(raw)==expected
        return dict(bytes=len(raw),sha256=sha(raw),entries=len(infos),files=len(actual),directories=sum(i.is_dir() for i in infos),manifest_payloads=len(records),crc_ok=True,payloads_ok=True,extracted_equal=True)
current=ROOT/'sources/current'; history=ROOT/'sources/history'
report={'current':verify_zip(INPUT,current),'history':verify_zip(current/'history/relation_revision_review.zip',history,'480371343ea0be50e4f8fcc614ec1aee5b35b20593151b7fbb3932e75669a9b5')}
report['environment']={'python':sys.version,'platform':platform.platform(),'packages':{}}
for package in ('torch','numpy','psutil','transformers','sentence-transformers'):
    try: report['environment']['packages'][package]=md.version(package)
    except md.PackageNotFoundError: report['environment']['packages'][package]=None
changes=[]
for p in (current/'scripts').glob('*'):
    old=history/p.relative_to(current)
    if old.is_file():
        changes.append({'path':p.relative_to(current).as_posix(),'unchanged_vs_history':p.read_bytes()==old.read_bytes(),'sha256':sha(p.read_bytes())})
report['script_comparison']=changes
# Submitted per-mode evidence binds exactly these current source bytes.
evidence=current/'reports/documentation/20261006/relation_revision_pilot/evidence'
for mode in ('cpu','native'):
    r=json.loads((evidence/'reports'/mode/'result.json').read_text())
    identities=r['source_files'] if mode=='cpu' else r['scientific_sources']
    failures=[]
    if isinstance(identities,dict):
        for name,digest in identities.items():
            if sha((current/name).read_bytes())!=digest: failures.append(name)
    else:
        for record in identities:
            b=(current/record['path']).read_bytes()
            if sha(b)!=record['sha256'] or len(b)!=record['bytes']: failures.append(record['path'])
    assert not failures, failures
    report[mode+'_submitted']={'source_count':len(identities),'all_match_current':True,'status':r['status'],'mode':r['mode']}
report['old_core_unchanged']=all(next(x for x in changes if x['path']==p)['unchanged_vs_history'] for p in ('scripts/step28_relation_revision.py','scripts/step28_relation_memory.py','scripts/step28_relation_memory_run.py'))
(ROOT/'outputs/01_identity_environment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
