"""Package and mechanically validate this static review; no project execution."""
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import zipfile

ROOT=Path('/workspace/scratch/3e46faac4c35/optuna_static_review')
DEST=ROOT.parent/'optuna_implementation_static_review_20261018.zip'

def sha(value):return hashlib.sha256(value).hexdigest()
def row(path):
    value=path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=len(value),sha256=sha(value))
def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

findings=json.loads((ROOT/'findings.json').read_text())
assert [x['id'] for x in findings['previous_findings']]==[f'F{i:02d}' for i in range(1,9)]
assert [x['id'] for x in findings['new_findings']]==[f'N{i:02d}' for i in range(1,9)]
for finding in findings['new_findings']:
    assert all(finding[k] for k in ('title_zh','trigger_zh','impact_zh','minimum_change_zh','disposition_zh','locations'))
    for loc in finding['locations']:
        path=ROOT.parent/'optuna_review_input'/loc['path']
        assert sha(path.read_bytes())==loc['sha256']
        assert 1<=loc['lines'][0]<=loc['lines'][1]<=len(path.read_text(encoding='utf-8-sig').splitlines())
scope=json.loads((ROOT/'reading_scope.json').read_text())
assert len(scope['files'])==140 and all(x['identity_verified'] for x in scope['files'])
assert all((ROOT.parent/'optuna_review_input'/p).is_file() for p in scope['main_full_semantic_paths'])
failures=json.loads((ROOT/'execution/failures.json').read_text())
assert len(failures)==6
assert findings['cpu_evidence']['total_actual_cpu_optimizer_steps']==45
assert findings['cpu_evidence']['charged_seconds']==82
assert findings['time_admission']['status']=='NOT_ESTABLISHED'
report=(ROOT/'report.zh.txt').read_text()
assert 'N08 / P2' in report and '七项需要处置的问题' not in report
assert 'raw/stage-cal/first-cal' in report
refs={ref for group in re.findall(r'\[([^\]\n]+)\]',report) for ref in re.findall(r'E\d\d(?!\d)',group)}
with (ROOT/'evidence_index.tsv').open() as stream:
    available={x['ref'] for x in csv.DictReader(stream,delimiter='\t')}
assert refs<=available
for path in ROOT.rglob('*.json'):
    if path.name=='manifest.json':continue
    json.loads(path.read_text(encoding='utf-8-sig'))
write_json(ROOT/'mechanical/deliverable_validation.json',dict(
    findings_schema_fields_present=True,previous_findings=8,new_findings=8,all_finding_source_locations_match=True,
    reading_scope_payloads=140,all_report_evidence_refs_resolved=True,
    original_mechanical_failures_retained=6,cpu_production_reference_total=[39,6,45],
    new_time_admission='NOT_ESTABLISHED',scientific_execution=False,
    validation_scope='JSON/paths/source-line bounds/report-field consistency/package identity only'))

payloads=[row(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ('manifest.json','members.tsv')]
write_json(ROOT/'manifest.json',dict(schema_version='1.0',kind='STATIC_IMPLEMENTATION_REVIEW_RETURN',
    original_input_zip_sha256='52bad7b321d0fa77de36ae7126213da815ba968afeb7d2aa74699f282f373672',
    payload_count=len(payloads),payload_bytes=sum(x['bytes'] for x in payloads),
    exclusions=['manifest.json','members.tsv'],files=payloads))
with (ROOT/'members.tsv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.writer(stream,delimiter='\t');writer.writerow(['path','bytes','sha256'])
    for r in payloads+[row(ROOT/'manifest.json')]:writer.writerow([r[k] for k in ('path','bytes','sha256')])
files=sorted(p for p in ROOT.rglob('*') if p.is_file())
with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for path in files:archive.write(path,path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(DEST) as archive:
    assert archive.testzip() is None
    names=archive.namelist();assert len(names)==len(set(names))==len(files)
    zmanifest=json.loads(archive.read('manifest.json'))
    for r in zmanifest['files']:
        value=archive.read(r['path']);assert len(value)==r['bytes'] and sha(value)==r['sha256']
    all_members=[dict(path=name,bytes=len(archive.read(name)),sha256=sha(archive.read(name))) for name in names]
external_members=DEST.with_suffix('.members.tsv')
with external_members.open('w',encoding='utf-8',newline='') as stream:
    writer=csv.writer(stream,delimiter='\t');writer.writerow(['path','bytes','sha256'])
    for r in all_members:writer.writerow([r[k] for k in ('path','bytes','sha256')])
identity=dict(file=DEST.name,bytes=DEST.stat().st_size,sha256=sha(DEST.read_bytes()),members=len(all_members),
    internal_manifest_payloads=len(payloads),all_member_bytes=sum(x['bytes'] for x in all_members),
    zip_crc_valid=True,payload_manifest_all_match=True,external_members_file=external_members.name,
    original_input_bytes=2011821,original_input_members=141,
    original_input_sha256='52bad7b321d0fa77de36ae7126213da815ba968afeb7d2aa74699f282f373672')
write_json(DEST.with_suffix('.identity.json'),identity)
print(json.dumps(identity,ensure_ascii=False,indent=2))
