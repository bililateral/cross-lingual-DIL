"""Read-only final input check and exact-byte copies for the review deliverable."""
import datetime,hashlib,json,pathlib,shutil,time
start=time.perf_counter()
B=pathlib.Path('/mnt/data/bge_review_evidence'); R=pathlib.Path('/mnt/data/bge_input')
report=json.loads((B/'outputs/inventory_verification.json').read_text())
rows=[]
for item in report['rows']:
 p=R/item['path']; data=p.read_bytes()
 assert len(data)==item['bytes']
 assert hashlib.sha256(data).hexdigest()==item['sha256'],str(p)
 rows.append({'path':item['path'],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
assert set(p.relative_to(R).as_posix() for p in R.rglob('*') if p.is_file())=={x['path'] for x in rows}|{'source_inventory.json'}
archive=pathlib.Path(report['archive']); raw=archive.read_bytes()
assert hashlib.sha256(raw).hexdigest()==report['archive_sha256']
(B/'input').mkdir(exist_ok=True); shutil.copyfile(archive,B/'input'/archive.name)
J=R/'reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job'
m=json.loads((J/'run/manifest.json').read_text())
sources=m['source_files']
paths=list(sources) if isinstance(sources,dict) else [x['path'] for x in sources]
paths+=['scripts/step28_bge_continual_audit.py','docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md','docs/SELLER_ALIAS_BGE_RESULT.zh.md']
P='reports/documentation/20260930/continual_plan/implementation/review/'
paths += [P+x for x in ('response_visible.zh.md','external/REVIEW.zh.md','disposition.json','primary_numeric_correction.json')]
index=[]
for rel in sorted(set(paths)):
 p=R/rel; d=p.read_bytes(); q=B/'reviewed_sources'/rel; q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(d)
 lines=d.decode('utf-8').splitlines()
 index.append({'original_relative_path':rel,'copy_relative_path':q.relative_to(B).as_posix(),'line_count':len(lines),'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()})
(B/'outputs/reviewed_source_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
res={'status':'ALL_495_ORIGINAL_PAYLOADS_UNCHANGED','input_archive_bytes':len(raw),'input_archive_sha256':hashlib.sha256(raw).hexdigest(),'payloads_reverified':len(rows),'extra_extracted_files':0,'exact_text_source_copies':len(index),'formal_labels_read':0,'formal_models_loaded':0,'elapsed_seconds':time.perf_counter()-start,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(B/'outputs/final_input_reverification.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(res,ensure_ascii=False,indent=2))
