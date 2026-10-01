"""Package preserved audit records; verify original upload again without parsing data."""
from pathlib import Path
import hashlib,json,shutil,zipfile
B=Path('/mnt/data/test_pretest_audit_20260928');E=B/'evidence';R=B/'source'
h=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile('/mnt/data/test_review.zip') as z:
 assert z.testzip() is None
 for n in z.namelist():assert (R/n).read_bytes()==z.read(n),n
shutil.copytree(R,E/'submitted_source',dirs_exist_ok=False)
shutil.copy2('/mnt/data/test_implementation_review.zh.md',E/'test_implementation_review.zh.md')
shutil.copy2(E/'outputs/independent_v2/independent_results.json','/mnt/data/test_implementation_independent_results.json')
files=[]
for p in sorted(E.rglob('*')):
 if p.is_file() and p.name not in ('evidence_inventory.json','SHA256SUMS.txt'):
  b=p.read_bytes();files.append({'path':p.relative_to(E).as_posix(),'bytes':len(b),'sha256':h(b)})
inv={'status':'PRESERVED_ACTUAL_AUDIT_RECORDS','files':files,'file_count':len(files),'total_bytes':sum(f['bytes'] for f in files),'excluded_from_self_manifest':['evidence_inventory.json','SHA256SUMS.txt'],'contains_formal_inputs':False,'contains_handmade_tensors_scores_text_and_labels':True,'first_independent_failure_preserved':True,'original_submission_sha256':h(Path('/mnt/data/test_review.zip').read_bytes())}
(E/'evidence_inventory.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2))
sha_lines=[r['sha256']+'  '+r['path'] for r in files]
sha_lines.append(h((E/'evidence_inventory.json').read_bytes())+'  evidence_inventory.json')
(E/'SHA256SUMS.txt').write_text('\n'.join(sha_lines)+'\n')
zpath=Path('/mnt/data/test_implementation_review_evidence.zip')
with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(E.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(E).as_posix())
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(files)+2
 for r in files:
  b=z.read(r['path']);assert len(b)==r['bytes'] and h(b)==r['sha256'],r['path']
 receipt={'status':'VERIFIED','zip_path':str(zpath),'zip_bytes':zpath.stat().st_size,'zip_sha256':h(zpath.read_bytes()),'members':len(z.namelist()),'manifest_files':len(files),'manifest_bytes':inv['total_bytes'],'source_original_members_unchanged':117,'no_project_source_edits':True,'reviewer_reference_v1_exit':1,'reviewer_reference_v2_exit':0,'supplied_tests_passed':15,'independent_final_passed':10,'formal_test_executed':False}
Path('/mnt/data/test_implementation_evidence_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False,indent=2))
