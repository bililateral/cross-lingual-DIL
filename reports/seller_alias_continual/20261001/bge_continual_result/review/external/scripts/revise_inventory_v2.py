from pathlib import Path
import hashlib,json
b=Path(__file__).parent;s=b/'00_inventory.py';t=b/'00_inventory_v2.py'
a=s.read_text();v=a.replace("\n print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));print('Extracted to',dest)","\nprint(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));print('Extracted to',dest)")
assert a!=v;t.write_text(v)
print(json.dumps({'reason':'Remove accidental one-space indent at top-level print; first version retained. No input data changed.','before_sha256':hashlib.sha256(a.encode()).hexdigest(),'after_sha256':hashlib.sha256(v.encode()).hexdigest()},indent=2))
