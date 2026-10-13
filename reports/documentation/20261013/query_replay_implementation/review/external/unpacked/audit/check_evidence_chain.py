"""Mechanical byte checks only; never import or execute project code."""
from pathlib import Path
import argparse,json,hashlib,tarfile
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
b=a.input/'reports/documentation/20261013/query_replay_implementation';q=b/'qualification/cpu/attempt01'
rows=json.loads((q/'source_manifest.json').read_text());r=json.loads((q/'result/result.json').read_text());s=json.loads((b/'qualification/state.json').read_text())
state_checks=[]
for e in s['cpu_evidence_retrieval']['files']:
 x=b/e['path']; state_checks.append({'path':e['path'],'size_match':x.stat().st_size==e['bytes'],'sha_match':hashlib.sha256(x.read_bytes()).hexdigest()==e['sha256']})
tar_checks=[]
with tarfile.open(q/'source.tar.gz') as t:
 for m in t.getmembers():
  if not m.isfile():raise ValueError('Unexpected non-file member')
  x=a.input/m.name;raw=t.extractfile(m).read();tar_checks.append({'path':m.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'matches_current':raw==x.read_bytes()})
expected={e['path'] for e in rows};actual={e['path'] for e in tar_checks};extra=sorted(actual-expected)
assert len(rows)==29 and rows==r['source_files'] and len(tar_checks)==30
assert all(e['matches_current'] for e in tar_checks) and expected<=actual
assert extra==['reports/documentation/20261013/query_replay_implementation/authorization.json']
assert all(e['size_match']and e['sha_match']for e in state_checks)
out={'source_count':len(rows),'source_manifest_equals_result_source_files':True,'tar_file_members':len(tar_checks),'tar_additional_files':extra,'tar_checks':tar_checks,'state_retrieval_files':state_checks,'all_match':True,'source_manifest_location':'separate sibling file, not a tar member'}
a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['tar_checks','state_retrieval_files']},ensure_ascii=False,indent=2))
