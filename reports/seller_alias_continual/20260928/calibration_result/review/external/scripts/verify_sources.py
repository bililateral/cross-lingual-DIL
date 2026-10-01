"""Byte/provenance checks only. Does not parse labels or load any array/model."""
from pathlib import Path
import json,hashlib,zipfile,difflib,sys,platform,datetime
root=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
base=root/'reports/seller_alias_continual/20260928'; job=base/'calibration_execution/20260928_140752/job'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_rec(p,r):
 assert p.stat().st_size==r['bytes'],str(p)
 assert sha(p)==r['sha256'],str(p)
 return {'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)}
inv=read(base/'calibration_result/review/source_inventory.json'); rows=[check_rec(root/r['path'],r) for r in inv['files']];assert len(rows)==inv['count']==290
outer=root.parent.parent/'calibration_result_review.zip'
with zipfile.ZipFile(outer) as z:
 assert z.testzip() is None and len(z.namelist())==291
 assert set(z.namelist())=={r['path'] for r in rows}|{'reports/seller_alias_continual/20260928/calibration_result/review/source_inventory.json'}
 for n in z.namelist():assert z.read(n)==(root/n).read_bytes()
source=read(job/'completion.json')['source_files'];assert len(source)==26
for r in source:check_rec(root/r['path'],r)
for n in ['preparation.json','prevalid.json','evaluation/collected.json','evaluation/evaluation.json']:
 assert read(job/n)['source_files']==source,n
prior=base/'calibration_implementation/review/calibration_review.zip'
assert sha(prior)=='77ab3765d0928a231fadae7cbb9d92049e0343c4a7518ea5478a0b2909bcf51d'
with zipfile.ZipFile(prior) as z:
 assert len(z.namelist())==103 and z.testzip() is None
 old_inv=read(base/'calibration_implementation/review/source_inventory.json')
 assert len(old_inv['files'])==102
 for r in old_inv['files']:
  b=z.read(r['path']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 for r in source:assert z.read(r['path'])==(root/r['path']).read_bytes()
ret=read(base/'calibration_result/return_inventory.json');print('return record sample',ret['files'][0]); ret_rows=[check_rec(root/r['path'],r) for r in ret['files']]
assert len(ret_rows)==ret['count']==82 and sum(r['bytes'] for r in ret_rows)==ret['total_bytes']==2397392
prep=read(job/'preparation.json'); auth_path=root/prep['authorization']['path'];check_rec(auth_path,prep['authorization']);auth=read(auth_path)
oldauth=read(base/'calibration_execution/authorization.json'); changes={k:{'before':oldauth.get(k),'after':auth.get(k)} for k in oldauth.keys()|auth.keys() if oldauth.get(k)!=auth.get(k)}
assert set(changes)<= {'time','job','readiness_report','launch_correction'}
assert auth['source_files']==oldauth['source_files']==source
assert auth['label_parses']=={'train':1,'development':1,'heldout':0,'owners':0}
for k in ['cpu_evidence','review_disposition']:
 rec=auth[k]; check_rec(root/rec['path'],rec);assert read(root/rec['path'])['source_files']==source
check_rec(job/'fitted.json',read(job/'prevalid.json')['fitted'])
first=base/'calibration_execution/20260928_140128'; trace=(first/'stderr.log').read_text(); assert (first/'exit_status.txt').read_text().strip()=='1'
assert 'line 403, in execute' in trace and 'data.record(authorization_path' in trace
assert not list(first.rglob('*access.json')) and not list(first.rglob('fit.json'))
assert (job.parent/'exit_status.txt').read_text().strip()=='0'; assert not list(job.rglob('failure.json'))
initial=base/'calibration_result/audit_initial/step28_alias_calibration_result.py'; current=root/'scripts/step28_alias_calibration_result.py'
s0=initial.read_text();s1=current.read_text();assert s0.count('[seed]["metrics"][metric]')==1 and s0.replace('[seed]["metrics"][metric]','[seed][metric]')==s1
patch=''.join(difflib.unified_diff(s0.splitlines(True),s1.splitlines(True),fromfile='submitted_initial',tofile='submitted_corrected'));(out/'submitted_analysis_correction.diff').write_text(patch)
assert (base/'calibration_result/exit_status.txt').read_text().strip()=='1';assert (base/'calibration_result/audit_execution/exit_status.txt').read_text().strip()=='0'
train=read(job/'train_access.json');valid=read(job/'development_access.json')
assert all(r['parse_attempts']==1 and r['heldout']==r['owners']==0 for r in (train,valid))
assert datetime.datetime.fromisoformat(train['time_utc'])<datetime.datetime.fromisoformat(valid['time_utc'])
part=prep['partition'];alignments={}
for role,fn in [('calibration','train_alignment.json'),('development','valid_alignment.json')]:
 a=read(job/fn);ids=[r['group_uid'] for r in part[role]]; assert ids==a['selected_groups'];assert set(a['selected_sellers'])==set(ids)
 sellers=[s for g in ids for s in a['selected_sellers'][g]];assert len(sellers)==len(ids)*28==len(set(sellers))
 assert all(len(ss)==28 and ss==sorted(ss) for ss in a['selected_sellers'].values()); alignments[role]={'groups':len(ids),'seller_count':len(sellers),'sellers':set(sellers)}
assert not alignments['calibration']['sellers'] & alignments['development']['sellers']
for a in alignments.values():a.pop('sellers')
result={'status':'PASS','zip_bytes':outer.stat().st_size,'zip_sha256':sha(outer),'zip_members':291,'source_count':290,'source_bytes':sum(r['bytes'] for r in rows),'frozen_sources':source,'nested_implementation_zip_sha256':sha(prior),'nested_sources_verified':102,'returned_records_count':82,'returned_records_bytes':2397392,'authorization_changes':changes,'alignments':alignments,'access_records':{'train':train,'development':valid},'official_environment':prep['environment'],'completion':{k:v for k,v in read(job/'completion.json').items() if k!='source_files'},'first_launch_failure':'relative authorization path before parse_once and fitting; no access/fit files in submitted first directory','analysis_correction':'exact one-line extra metrics nesting removal','source_inventory':rows,'scope':'Byte identities and saved records only. No external server, formal labels, arrays or models loaded by this script.'}
(out/'source_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:result[k] for k in ['status','zip_bytes','zip_sha256','zip_members','source_count','source_bytes','nested_sources_verified','returned_records_count','returned_records_bytes','alignments']},ensure_ascii=False,indent=2))
