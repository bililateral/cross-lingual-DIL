"""Verify pre-deletion RECORD identities against supplied bytes/manifest, not server state.
Does not access or materialize models or Memory payloads.
"""
from pathlib import Path
import argparse, hashlib, json, zipfile, traceback,sys

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 b=a.input.read_bytes();assert hashlib.sha256(b).hexdigest()=='9013c992a925bac6e69ba603438d5ca62692fcc38d5306a89f8a942f05973c96'
 with zipfile.ZipFile(a.input) as z:
  d=json.loads(z.read('review/support/reports/maintenance/20261014/query_weights/predelete.json'));m=json.loads(z.read('job/run/manifest.json'));by_model={Path(v['model']['path']).name:v['model'] for v in m['points'].values()};by_memory={'run/'+v['memory']['path']:v['memory'] for v in m['points'].values()}
  for v in d['files']:
   r=by_model[Path(v['path']).name];assert (v['bytes'],v['sha256'])==(r['bytes'],r['sha256'])
  checked=[];unavailable=[]
  for v in d['nonweight_files']:
   n='job/'+v['path']
   if n in z.namelist():
    b=z.read(n);assert (len(b),hashlib.sha256(b).hexdigest())==(v['bytes'],v['sha256']);checked.append(n)
   else:
    r=by_memory[v['path']];assert (v['bytes'],v['sha256'])==(r['bytes'],r['sha256']);unavailable.append(v)
  assert len(checked)==242 and len(unavailable)==5 and len(d['files'])==5
  assert d['total_bytes']==sum(v['bytes'] for v in d['files'])==6521871122
  assert sum(v['bytes'] for v in unavailable)==1734069
  out={'status':'PASS_CUSTODY_RECORD_IDENTITIES','weight_records_matched_to_endpoint_manifest':5,'weight_bytes_recorded':6521871122,'nonweight_records':247,'nonweight_payloads_verified_in_zip':242,'memory_records_matched_only':5,'memory_bytes_recorded':1734069,'absent_by_design_memory_records':unavailable,'verified_payload_paths':checked,'server_deletion_observed':False,'post_deletion_server_hashes_rechecked':False,'limits':'Frozen predelete records are not a deletion receipt. Subsequent deletion and 247 unchanged hashes are user-supplied status, not independently witnessed here. No models or Memory payloads read.'}
  (a.out/'custody_record_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k not in ['verified_payload_paths','absent_by_design_memory_records']},ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except BaseException:traceback.print_exc();sys.exit(1)
