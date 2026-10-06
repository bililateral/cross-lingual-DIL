#!/usr/bin/env python3
"""Package the audit's real files and unchanged authorized inputs; no research execution."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

WORK=Path(__file__).resolve().parents[1]
DEST=WORK/'relation_result_external_audit_20261006'
ZIP=WORK/'SELLER_ALIAS_RELATION_MEMORY.result_audit_evidence.zip'
assert not DEST.exists() and not ZIP.exists(), 'Fresh artifact destinations required'
DEST.mkdir()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(src, relative):
    target=DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,target)
    assert sha(src)==sha(target)


report=WORK/'SELLER_ALIAS_RELATION_MEMORY.result_external_review.zh.md'
put(report,report.name)
put(WORK/'result_report_artifacts/README.zh.md','README.zh.md')
context=WORK/'upload/relation_result_context_review.zip'
prior=WORK/'upload/review(10).zip'
assert context.stat().st_size==5077487 and sha(context)=='3c11a9640c13438d9eb7ace1005183d83ccafbdca7036fa39e0689b06731625a'
assert prior.stat().st_size==319319 and sha(prior)=='ddebe7ca566cff689975ca083e3acd597710c815992aed224497b23c306eb43e'
put(context,'inputs/relation_result_context_review.zip')
put(prior,'inputs/prior_pilot_review10.zip')
directories={
    'result_identity_audit':'identity',
    'result_metrics_audit':'metrics',
    'result_lifecycle_audit':'lifecycle',
    'result_resources_audit':'resources',
    'result_repairs_audit':'repairs',
    'result_interpretation_audit':'interpretation',
}
for local,short in directories.items():
    root=WORK/local
    for source in sorted(root.rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts and source.suffix!='.pyc':
            put(source,Path('audit')/short/source.relative_to(root))
put(WORK/'result_report_artifacts/independent_all_endpoints.csv','tables/independent_all_endpoints.csv')
for name in ('render_tables.py','report_template.zh.md','build_evidence_bundle.py'):
    put(WORK/'result_report_artifacts'/name,Path('audit/report_generation')/name)

records=[]
for path in sorted(DEST.rglob('*')):
    if path.is_file():
        records.append({'path':path.relative_to(DEST).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
manifest={'scope':'Complete Chinese independent result review plus actually executed evidence and unchanged authorized input archives',
          'date':'2026-10-06','excluded':['manifest.json (self-reference avoided)'],
          'files':records}
(DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for path in sorted(DEST.rglob('*')):
        if path.is_file():archive.write(path,path.relative_to(DEST).as_posix())
with zipfile.ZipFile(ZIP) as archive:
    assert archive.testzip() is None
    names=archive.namelist()
    assert len(names)==len(records)+1==len(set(names))
    assert set(names)=={r['path'] for r in records}|{'manifest.json'}
    for rec in records:
        value=archive.read(rec['path'])
        assert len(value)==rec['bytes'] and hashlib.sha256(value).hexdigest()==rec['sha256']
receipt={'status':'PASS_FINAL_AUDIT_PACKAGE_IDENTITY','report':{'path':str(report),'bytes':report.stat().st_size,'sha256':sha(report)},
         'zip':{'path':str(ZIP),'bytes':ZIP.stat().st_size,'sha256':sha(ZIP),'members':len(names)},
         'manifest_payloads':len(records),'all_final_payloads_verified':True,'zip_crc_pass':True}
(WORK/'result_report_artifacts/delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
