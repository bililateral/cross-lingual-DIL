"""Deterministic return ZIP and complete external member manifest; no external IO."""
from pathlib import Path
import csv, hashlib, io, json, sys, zipfile
root, archive = map(Path, sys.argv[1:3])
root, archive = root.resolve(), archive.resolve()
def sha(b): return hashlib.sha256(b).hexdigest()
def rec(p):
    b=p.read_bytes(); return {'path':p.relative_to(root).as_posix(),'bytes':len(b),'sha256':sha(b)}
assert (root/'authorized_input/review_input.zip').stat().st_size == 280892
assert sha((root/'authorized_input/review_input.zip').read_bytes()) == '821415de49e848672d6ae5a025bbe88e63dfb4c4990c86aaf1cc0061b2fb6b6a'
findings=json.loads((root/'review/findings.json').read_text())
assert findings['counts']=={'SCIENTIFIC_BLOCKER':1,'REPRODUCIBILITY_DEFECT':2}
assert (root/'evidence/independent_checks.exit_code.txt').read_text().strip()=='0'
assert (root/'evidence/independent_checks.stderr.txt').stat().st_size==0
assert (root/'evidence/audit_input.exit_code.txt').read_text().strip()=='0'
assert len(list((root/'evidence/na_pipeline').iterdir()))==56
assert len(json.loads((root/'review/reading_scope.json').read_text()))==69
old=[p for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix()!='manifest.json']
manifest={'schema':'review_return_member_manifest_v1','excludes_self':True,'self_path':'manifest.json',
          'note':'External complete member manifest includes this file itself; archive identity is external to avoid self-reference.',
          'files':[rec(p) for p in sorted(old)]}
(root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
all_files=sorted(p for p in root.rglob('*') if p.is_file())
full=[rec(p) for p in all_files]
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in all_files:
        info=zipfile.ZipInfo(p.relative_to(root).as_posix(),date_time=(2026,10,14,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o100644<<16
        z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(full)
    assert set(z.namelist())=={x['path'] for x in full}
    for x in full:
        b=z.read(x['path']); assert len(b)==x['bytes'] and sha(b)==x['sha256']
    im=json.loads(z.read('manifest.json')); assert im['files']==[x for x in full if x['path']!='manifest.json']
raw=archive.read_bytes()
receipt={'archive_name':archive.name,'bytes':len(raw),'sha256':sha(raw),'members':len(full),
         'payloads_excluding_internal_manifest':len(full)-1,'uncompressed_bytes':sum(x['bytes'] for x in full),
         'internal_manifest':'manifest.json','internal_manifest_excludes_self':True,
         'all_archive_members_verified':True,'input_original_preserved':True,
         'classification_counts':findings['counts'],'BGE_qualification_passed':False,'formal_run_performed':False}
external=archive.with_name(archive.stem+'_members.json')
external.write_text(json.dumps({'archive':receipt,'members':full},ensure_ascii=False,indent=2)+'\n')
tsv=archive.with_name(archive.stem+'_members.tsv')
with tsv.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256'],delimiter='\t');w.writeheader();w.writerows(full)
archive.with_name(archive.stem+'_summary.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
archive.with_name(archive.stem+'_SHA256.txt').write_text(sha(raw)+'  '+archive.name+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
