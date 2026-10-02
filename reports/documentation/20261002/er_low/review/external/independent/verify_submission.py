"""Verify all archive payloads and freeze entries without experiment execution."""
import collections,hashlib,json,pathlib,platform,sys,zipfile,os,datetime
R=pathlib.Path('/mnt/data/er_low_source');E=pathlib.Path('/mnt/data/er_low_audit/evidence')
def digest(b):return hashlib.sha256(b).hexdigest()
def check(rows):
 out=[]
 for row in rows:
  b=(R/row['path']).read_bytes();out.append(dict(path=row['path'],expected_bytes=row['bytes'],actual_bytes=len(b),expected_sha256=row['sha256'],actual_sha256=digest(b),match=(len(b)==row['bytes'] and digest(b)==row['sha256'])))
 return out
z=zipfile.ZipFile('/mnt/data/er_low_review.zip');raw=pathlib.Path(z.filename).read_bytes();inventory=json.loads(z.read('source_inventory.json'));freeze=json.loads(z.read('reports/documentation/20261002/er_low/freeze.json'))
rows=check(inventory['files']);frows=check(freeze['source_files']);old=json.loads(z.read('reports/documentation/20261001/er_weight/freeze.json'))
changed={r['path']:r for r in freeze['changed_from_previous_er_sources']};oldrows=check(old['source_files']);preserved=[r for r in oldrows if r['path'] not in changed]
manifest_names={r['path'] for r in inventory['files']};payload_names=set(z.namelist())-{'source_inventory.json'}
result=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original=dict(path=z.filename,bytes=len(raw),sha256=digest(raw),expected_bytes=2024711,expected_sha256='bb6d99cd2cce7b72230a8f84df50e96c74454a3a4a498898bc993aa9e8549aef',members=len(z.infolist()),crc_bad_member=z.testzip(),duplicate_names=[n for n,c in collections.Counter(z.namelist()).items() if c>1]),payload_file_count=len(rows),payload_total_bytes=sum(r['actual_bytes'] for r in rows),declared_payload_bytes=inventory['total_payload_bytes'],unlisted=sorted(payload_names-manifest_names),missing=sorted(manifest_names-payload_names),payload_mismatches=[r for r in rows if not r['match']],freeze_count=len(frows),freeze_mismatches=[r for r in frows if not r['match']],changed_prior_count=len(changed),prior_unchanged_count=len(preserved),prior_unchanged_mismatches=[r for r in preserved if not r['match']],policy_digest=digest((R/'schema/step28_er_low_policy.json').read_bytes()),freeze_policy_digest=freeze['policy_sha256'])
for n,obj in [('submission_verification.json',result),('submission_inventory_verified.json',rows),('freeze_verified.json',frows),('prior_freeze_comparison.json',oldrows),('original_zip_member_inventory.json',[dict(path=n,bytes=len(z.read(n)),sha256=digest(z.read(n))) for n in z.namelist()])]:
 (E/n).write_text(json.dumps(obj,indent=2)+'\n')
import numpy,torch
info=dict(sys_version=sys.version,executable=sys.executable,platform=platform.platform(),machine=platform.machine(),numpy=numpy.__version__,torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),torch_threads=torch.get_num_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)))
(E/'environment.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(result,indent=2));print(json.dumps(info,indent=2))
assert len(rows)==293 and len(frows)==31 and sum(r['actual_bytes'] for r in rows)==5997573
assert not result['payload_mismatches'] and not result['freeze_mismatches'] and not result['prior_unchanged_mismatches']
assert not result['unlisted'] and not result['missing'] and result['policy_digest']==result['freeze_policy_digest']
