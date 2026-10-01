"""End-of-audit original-byte and receipt checks. No formal inputs are decoded."""
import hashlib,json,zipfile,platform,sys,importlib.metadata
from pathlib import Path
B=Path('/mnt/data/test_pretest_audit_20260928');R=B/'source';E=B/'evidence'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=R/'reports/seller_alias_continual/20260928/test_implementation'
receipt=json.loads((root/'return_sync.json').read_text())
rows=[]
for v in receipt['files']:
 p=R/v['path'];assert p.stat().st_size==v['bytes'] and h(p)==v['sha256'];rows.append(v)
with zipfile.ZipFile('/mnt/data/test_review.zip') as z:
 assert z.testzip() is None
 names=z.namelist();assert len(names)==len(set(names))
 comparisons=[]
 for n in names:
  b=z.read(n);p=R/n;assert p.read_bytes()==b,n
  comparisons.append({'path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 extras=sorted(p.relative_to(R).as_posix() for p in R.rglob('*') if p.is_file() and p.relative_to(R).as_posix() not in names)
 assert not extras,extras
versions={}
for name in ('numpy','scipy','torch','sentence-transformers','transformers'):
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]=None
obj={'status':'ORIGINAL_SUBMISSION_UNCHANGED','archive_bytes':Path('/mnt/data/test_review.zip').stat().st_size,'archive_sha256':h(Path('/mnt/data/test_review.zip')),'original_members':len(names),'all_original_bytes_identical':True,'extra_files_in_source':extras,'checked_members':comparisons,'final_project_cpu_receipt_files':len(rows),'final_project_cpu_receipt_bytes':sum(r['bytes'] for r in rows),'checked_cpu_receipt':rows,'python':sys.version,'platform':platform.platform(),'installed_distribution_versions':versions,'versions_only_no_model_library_install':True,'formal_labels_text_scores_models_accessed':False}
(E/'outputs/final_source_checks.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in obj.items() if k not in ('checked_members','checked_cpu_receipt')},ensure_ascii=False,indent=2))
