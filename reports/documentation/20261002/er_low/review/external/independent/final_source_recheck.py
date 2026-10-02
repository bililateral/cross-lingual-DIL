"""Recheck every originally extracted member and original ZIP after all tests."""
import hashlib,json,datetime
from pathlib import Path
E=Path('/mnt/data/er_low_audit/evidence');R=Path('/mnt/data/er_low_source')
rows=json.loads((E/'original_zip_member_inventory.json').read_text());mismatches=[]
for row in rows:
    b=(R/row['path']).read_bytes()
    if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:mismatches.append(row['path'])
orig=Path('/mnt/data/er_low_review.zip').read_bytes();copy=Path('/mnt/data/er_low_audit/original/er_low_review.zip').read_bytes()
report=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_member_files_rechecked=len(rows),source_mismatches=mismatches,extra_extracted_files=sorted(str(f.relative_to(R)) for f in R.rglob('*') if f.is_file() and str(f.relative_to(R)) not in {r['path'] for r in rows}),original_bytes=len(orig),original_sha256=hashlib.sha256(orig).hexdigest(),preserved_zip_identical=orig==copy)
(E/'final_source_recheck.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
assert not mismatches and not report['extra_extracted_files'] and orig==copy and len(orig)==2024711 and report['original_sha256']=='bb6d99cd2cce7b72230a8f84df50e96c74454a3a4a498898bc993aa9e8549aef'
