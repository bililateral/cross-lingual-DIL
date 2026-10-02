"""Independent byte identity checks, using no project module imports."""
import hashlib,json,pathlib,zipfile,sys,platform,importlib.metadata,importlib.util
R=pathlib.Path('/mnt/data/logit_audit');S=R/'submission'
def ident(p):
 b=p.read_bytes(); return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def compare(records):
 out=[]
 for r in records:
  p=S/r['path']; actual=ident(p) if p.is_file() else None
  out.append({'path':r['path'],'expected':{k:r[k] for k in ['bytes','sha256']},'actual':actual,'match':actual=={k:r[k] for k in ['bytes','sha256']}})
 return out
invpath='reports/documentation/20261002/logit_weight/review/source_inventory.json'
inv=json.loads((S/invpath).read_text()); freeze=json.loads((S/'reports/documentation/20261002/logit_weight/freeze.json').read_text())
result={'original_zip':ident(R/'original/logit_review.zip'),'inventory':compare(inv['files']),'current_freeze':compare(freeze['source_files'])}
with zipfile.ZipFile(R/'original/logit_review.zip') as z:
 names=z.namelist(); result.update(member_count=len(names),duplicate_names=len(names)-len(set(names)),crc_error=z.testzip(),payload_bytes=sum(x.file_size for x in z.infolist()),inventory_missing=sorted(set(names)-{x['path'] for x in inv['files']}-{invpath}),inventory_extra=sorted({x['path'] for x in inv['files']}-set(names)))
for name in ['er_low','er_weight']:
 p=S/f'reports/documentation/{"20261002" if name=="er_low" else "20261001"}/{name}/freeze.json'
 old=json.loads(p.read_text()); result[name+'_historical_freeze']=compare(old['source_files'])
policy=json.loads((S/'schema/step28_logit_weight_policy.json').read_text())
result['policy_identity']=ident(S/'schema/step28_logit_weight_policy.json')
result['reference_record_checks']={}
for name in ['baseline','weight_reference']:
 ref=policy[name]; records=[]
 for key,r in ref['records'].items(): records.append(dict(r,path=str(pathlib.PurePosixPath(ref['local_small_job'])/r['path'])))
 result['reference_record_checks'][name]=compare(records)
(R/'evidence/identity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('original:',result['original_zip'],'members:',result['member_count'],'payload:',result['payload_bytes'])
for k in ['inventory','current_freeze','er_low_historical_freeze','er_weight_historical_freeze']:
 rows=result[k];print(k,len(rows),'matches',sum(r['match'] for r in rows));
 for r in rows:
  if not r['match']: print('  differs from historical frozen version:',r['path'])
print('reference_records', {k:all(x['match'] for x in v) for k,v in result['reference_record_checks'].items()})
print('unlisted',result['inventory_missing'],'extra',result['inventory_extra'],'CRC',result['crc_error'])
assert result['original_zip']=={'bytes':2084038,'sha256':'ff678c2f9687ced6fa9888eca55064ca617c98d35bf371e082e7cf321f9ed08a'}
assert result['member_count']==304 and len(result['inventory'])==303 and len(result['current_freeze'])==34
assert all(r['match'] for r in result['inventory']+result['current_freeze'])
assert not result['inventory_missing'] and not result['inventory_extra']
print('environment',sys.executable,sys.version,platform.platform())
for pkg in ['numpy','torch','scipy','scikit-learn','sentence-transformers','transformers']:
 try: print(pkg,importlib.metadata.version(pkg))
 except importlib.metadata.PackageNotFoundError: print(pkg,'NOT_INSTALLED')
print('identity PASS')
