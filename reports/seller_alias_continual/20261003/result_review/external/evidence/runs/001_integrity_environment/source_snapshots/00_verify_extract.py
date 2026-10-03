#!/usr/bin/env python3
"""Verify the supplied archive byte-for-byte, inventory, and safely extract."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import sys

BASE=Path('/workspace/scratch/ca13b63db3f0/review_work')
PACKAGE=BASE.parent/'upload'/'result_review(1).zip'
DEST=BASE/'input'
EVIDENCE=BASE/'evidence'/'input'
def sha(b): return hashlib.sha256(b).hexdigest()
with ZipFile(PACKAGE) as z:
    infos=z.infolist()
    report={'filename_received':PACKAGE.name,'authorized_name':'result_review.zip','bytes':PACKAGE.stat().st_size,'sha256':sha(PACKAGE.read_bytes()),'members':len(infos),'uncompressed_bytes':sum(x.file_size for x in infos),'verified_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    assert report['bytes']==8729802
    assert report['sha256']=='87ff5ae41173ea9f68172736a11bd90bf54846fc4a2f0843b12edaa2965b9f7c'
    assert report['members']==613
    assert len({x.filename for x in infos})==len(infos),'duplicate members'
    inventory=[]
    for i in infos:
        p=PurePosixPath(i.filename)
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename
        assert not stat.S_ISLNK(i.external_attr>>16),'symlink'
        data=z.read(i)
        inventory.append({'path':i.filename,'bytes':len(data),'sha256':sha(data),'crc32':f'{i.CRC:08x}','compressed_bytes':i.compress_size})
        target=DEST/Path(*p.parts)
        if i.is_dir(): target.mkdir(parents=True,exist_ok=True)
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
    report.update({'all_zip_crc_verified':True,'duplicate_members':0,'unsafe_paths':0,'symlinks':0,'all_extracted_bytes_verified':all((DEST/i['path']).is_dir() or sha((DEST/i['path']).read_bytes())==i['sha256'] for i in inventory)})
EVIDENCE.mkdir(parents=True,exist_ok=True)
(EVIDENCE/'received_members.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
(EVIDENCE/'archive_integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
versions={}
for name in ['numpy','scipy','torch','pytest','pandas','scikit-learn','transformers','sentence-transformers']:
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name]=None
environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'machine':platform.machine(),'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,'packages':versions,'model_context_name':'GPT-6 Astra Pro','independent_backend_identifier':None,'no_new_packages_installed':True}
(EVIDENCE/'web_environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'integrity':report,'web_environment':environment},ensure_ascii=False,indent=2))
print('ROOT FILES')
for p in sorted(DEST.iterdir()): print(p.name)
print('CODE AND MANIFEST PATHS')
for i in inventory:
    if i['path'].endswith('.py') or any(s in i['path'] for s in ['source_inventory','freeze.json','REQUEST','SOURCE','manifest','readme','README']):
        print(i['path'],i['bytes'])
