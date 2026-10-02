"""Finalize payload hashes and the outside delivery archive. No input mutations."""
from __future__ import annotations
import datetime,hashlib,json,os,pathlib,platform,sys,time,zipfile
start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
E=pathlib.Path('/mnt/data/er_review_evidence');R=pathlib.Path('/mnt/data/er_review_input');ORIGINAL=pathlib.Path('/mnt/data/er_weight_result_review.zip');OUT=pathlib.Path('/mnt/data/ER_weight_result_external_review_20261002.zip')
def rec(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert rec(ORIGINAL)==dict(bytes=7194455,sha256='5de5711fbe6d20107e8607e8c4aec85a2ac47293e845af9c8c6ece1358631db0')
assert ORIGINAL.read_bytes()==(E/'original/er_weight_result_review.zip').read_bytes()
with zipfile.ZipFile(ORIGINAL) as z:
 assert z.testzip() is None and len(z.infolist())==365
 for f in z.infolist():assert (R/f.filename).read_bytes()==z.read(f.filename)
report=E/'REVIEW.zh.md';standalone=pathlib.Path('/mnt/data/REVIEW.zh.md');assert report.read_bytes()==standalone.read_bytes()
expected=None;tables=0
for line in report.read_text().splitlines():
 if line.startswith('|'):
  if expected is None:expected=line.count('|');tables+=1
  assert expected==line.count('|')
 else:expected=None
required=['logs/independent_v1/execution.json','logs/supplied_audit_cpu0/execution.json','logs/hand_reference_v1/stderr.log','logs/hand_reference_v3/execution.json','logs/supplemental_v1/stderr.log','logs/supplemental_v2/execution.json','logs/portable_replay_smoke/execution.json','source/reproduce_saved_review.py','inspection/FAILURES_AND_REVISIONS.zh.md']
assert all((E/f).is_file() for f in required)
final=dict(status='PASS_FINAL_INPUT_AND_REPORT_INTEGRITY',verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_zip=rec(ORIGINAL),original_copy_identical=True,all_original_members_unchanged=365,source_inventory_not_modified=True,report=rec(report),report_lines=len(report.read_text().splitlines()),markdown_tables=tables,server_connected=False,formal_label_files_read=False,native_models_loaded=False)
(E/'results/final_input_integrity.json').write_text(json.dumps(final,indent=2)+'\n')
excluded={'FILE_MANIFEST.json','SHA256SUMS.txt'};files=sorted(p for p in E.rglob('*') if p.is_file() and p.relative_to(E).as_posix() not in excluded)
manifest=dict(generated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='All evidence payloads excluding FILE_MANIFEST.json and SHA256SUMS.txt themselves; outside delivery ZIP is separately hashed.',files=[dict(path=str(p.relative_to(E)),**rec(p)) for p in files])
(E/'FILE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(E/'SHA256SUMS.txt').write_text(''.join(f"{row['sha256']}  {row['path']}\n" for row in manifest['files']))
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(E.rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(E)))
with zipfile.ZipFile(OUT) as z:
 assert z.testzip() is None;assert z.read('original/er_weight_result_review.zip')==ORIGINAL.read_bytes()
 for row in manifest['files']:
  b=z.read(row['path']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 member_count=len(z.infolist())
delivery=dict(status='PASS_PACKAGED_EVIDENCE',started_at_utc=start,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-clock,exit_code=0,command=[sys.executable,*sys.argv],cwd=os.getcwd(),python=sys.version,platform=platform.platform(),origin='CURRENT_WEB_CONTAINER_NOT_PROJECT_LINUX',original_input=dict(path=str(ORIGINAL),**rec(ORIGINAL)),review=dict(path=str(standalone),**rec(standalone)),evidence_zip=dict(path=str(OUT),members=member_count,**rec(OUT)),inside_manifest=dict(path='FILE_MANIFEST.json',**rec(E/'FILE_MANIFEST.json')),inside_sha256sums=dict(path='SHA256SUMS.txt',**rec(E/'SHA256SUMS.txt')),all_evidence_members_read_back_verified=True)
pathlib.Path('/mnt/data/ER_weight_review_delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n')
pathlib.Path('/mnt/data/ER_weight_review_SHA256SUMS.txt').write_text(''.join(f"{rec(p)['sha256']}  {p.name}\n" for p in (standalone,OUT,ORIGINAL)))
print(json.dumps(delivery,ensure_ascii=False,indent=2))
