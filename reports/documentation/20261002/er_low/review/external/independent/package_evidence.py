"""Package immutable evidence, excluding only the manifest's own self-hash."""
from pathlib import Path
import datetime,hashlib,json,zipfile
B=Path('/mnt/data/er_low_audit');Z=Path('/mnt/data/er_low_review_evidence.zip')
if Z.exists():raise FileExistsError('Do not overwrite a delivered evidence ZIP')
paths=sorted(p for p in B.rglob('*') if p.is_file() and p.name!='EVIDENCE_MANIFEST.json')
rows=[dict(path=p.relative_to(B).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
manifest=dict(schema='er_low_contract_cpu_audit_evidence_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),payload_file_count=len(rows),payload_total_bytes=sum(r['bytes'] for r in rows),excluded_self_reference=['EVIDENCE_MANIFEST.json'],files=rows)
mp=B/'EVIDENCE_MANIFEST.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(Z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in [*paths,mp]:z.write(p,p.relative_to(B).as_posix())
with zipfile.ZipFile(Z) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(rows)+1 and len(z.namelist())==len(set(z.namelist()))
    for row in rows:
        content=z.read(row['path']);assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256']
    assert z.read('original/er_low_review.zip')==Path('/mnt/data/er_low_review.zip').read_bytes()
    assert z.read('REVIEW.zh.md')==(B/'REVIEW.zh.md').read_bytes()
record=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),filename=Z.name,bytes=Z.stat().st_size,sha256=hashlib.sha256(Z.read_bytes()).hexdigest(),zip_members=len(rows)+1,manifest_payload_files=len(rows),manifest_payload_bytes=manifest['payload_total_bytes'],crc_errors=0,all_payload_sizes_and_sha256_match=True,original_zip_preserved=True,original_sha256='bb6d99cd2cce7b72230a8f84df50e96c74454a3a4a498898bc993aa9e8549aef',review_sha256=hashlib.sha256((B/'REVIEW.zh.md').read_bytes()).hexdigest(),notes='This record is outside the ZIP to avoid a circular self-hash.')
Path('/mnt/data/er_low_delivery.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
for src,dest in [('REVIEW.zh.md','REVIEW.zh.md'),('COMMANDS.zh.md','ER_LOW_COMMANDS.zh.md'),('TEST_CASES.zh.md','ER_LOW_TEST_CASES.zh.md')]:Path('/mnt/data',dest).write_bytes((B/src).read_bytes())
print(json.dumps(record,indent=2))
