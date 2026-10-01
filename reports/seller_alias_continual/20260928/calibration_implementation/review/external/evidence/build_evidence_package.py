"""Package reviewer-generated evidence and report; never include source research arrays."""
from pathlib import Path
import hashlib,json,zipfile,shutil
E=Path(__file__).resolve().parent; W=E.parent; D=W.parent
sha=lambda b:hashlib.sha256(b).hexdigest()
report=W/'calibration_review_report.zh.md'
paths=[('calibration_review_report.zh.md',report)]
paths += [('evidence/'+p.relative_to(E).as_posix(),p) for p in sorted(E.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
rows=[]
for name,p in paths:
    data=p.read_bytes(); rows.append({'path':name,'bytes':len(data),'sha256':sha(data)})
manifest={'scope':'Reviewer-generated handmade evidence only; original approved source ZIP not duplicated. All NPY outputs in evidence are handmade, not formal predictions/labels.','files':rows,'file_count_excluding_manifest':len(rows),'bytes_excluding_manifest':sum(r['bytes'] for r in rows)}
content=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
zip_path=D/'calibration_review_evidence.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for name,p in paths:z.write(p,arcname=name)
    z.writestr('SHA256SUMS.json',content)
with zipfile.ZipFile(zip_path) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(rows)+1
    for r in rows:
        b=z.read(r['path']); assert len(b)==r['bytes'] and sha(b)==r['sha256'],r['path']
shutil.copy2(E/'independent_outputs/independent_results.json',D/'calibration_review_independent_results.json')
receipt={'package':zip_path.name,'bytes':zip_path.stat().st_size,'sha256':sha(zip_path.read_bytes()),'members':len(rows)+1,'files_with_size_and_sha':len(rows),'uncompressed_bytes_excluding_manifest':manifest['bytes_excluding_manifest'],'archive_crc_verified':True,'every_manifest_entry_verified':True,'report_sha256':sha(report.read_bytes()),'original_submission_not_modified':True,'formal_stage_run':False}
(D/'calibration_review_evidence_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
