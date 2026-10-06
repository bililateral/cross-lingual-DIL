from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parents[2];a=r/'input';b=r/'history/previous_input/active_source'
rows=[]
for p in sorted((a/'scripts').glob('*.py')):
 old=b/'scripts'/p.name
 if old.is_file():
  match=hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(old.read_bytes()).hexdigest()
  rows.append({'path':str(p.relative_to(a)),'same_as_previous_active':match})
assert len(rows)==12 and all(v['same_as_previous_active'] for v in rows)
print(json.dumps(rows,indent=2));(r/'audit/outputs/reused_source_identity.json').write_text(json.dumps(rows,indent=2))
