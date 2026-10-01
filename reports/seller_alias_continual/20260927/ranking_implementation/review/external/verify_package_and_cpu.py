"""Independent byte/record arithmetic verification. NO model or formal-data access."""
import os,sys,json,hashlib,zipfile,math,platform
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/data/ranking_review_source');OUT=Path('/mnt/data/ranking_reviewer_evidence')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def check_records(base,rows):
 for r in rows:
  p=base/r['path']; assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
 return len(rows),sum(r['bytes'] for r in rows)
path=Path('/mnt/data/ranking_review.zip');assert path.stat().st_size==591625 and sha(path)=='bad88d2c83feef2d03f7c545519ab134930c9b50dd81c99d6e3b849008cf5203'
with zipfile.ZipFile(path) as z:assert len(z.infolist())==61 and z.testzip() is None
inv=read(ROOT/'reports/seller_alias_continual/20260927/ranking_implementation/review/source_inventory.json')
count,nbytes=check_records(ROOT,inv['files']);assert (count,nbytes)==(60,1480799)
folder=ROOT/'reports/seller_alias_continual/20260927/ranking_implementation';cpu=folder/'cpu/20260927_105450'
sync=read(folder/'cpu_sync.json');ret=read(cpu/'return_inventory.json');assert sync['files']==ret['files']; assert check_records(cpu,sync['files'])==(13,130994)
audit=read(cpu/'audit.json');assert check_records(ROOT,audit['source_files'])[0]==18
assert check_records(cpu,list(audit['native_reports'].values()))[0]==4
reports={k:read(cpu/(k+'.json')) for k in ('original_d','wrapped_d','schedule','hard')}
for f in ('initial_model_state_sha256','post_model_state_sha256','post_optimizer_state_sha256','initial_scores','post_scores','captured_training_logits','parameter_count'):assert reports['original_d'][f]==reports['wrapped_d'][f],f
for kind in ('schedule','hard'):
 for f in ('initial_model_state_sha256','initial_scores','captured_training_logits','parameter_count'):assert reports[kind][f]==reports['original_d'][f],f
 assert reports[kind]['actual_optimizer_lrs']==[1e-5/87,.001]
# Create ONLY the explicitly documented handmade clique pattern, not any formal truth.
comp=[k for k,n in enumerate([3]*4+[2]*8) for _ in range(n)]
edges=[(i,j) for i in range(28) for j in range(i+1,28)]; y=np.array([comp[i]==comp[j] for i,j in edges],dtype=int)
edge_index={e:k for k,e in enumerate(edges)}
def reference(s,w):
 bce=float(np.mean(np.logaddexp(0,s)-y*s));rank=0.;hard=0.
 for q in range(28):
  js=[j for j in range(28) if j!=q]; ix=[edge_index[tuple(sorted((q,j)))] for j in js]
  pos=[k for k in ix if y[k]]; neg=sorted([j for j in js if not y[edge_index[tuple(sorted((q,j)))]]],key=lambda j:(-float(s[edge_index[tuple(sorted((q,j)))]]),j))[:5]
  ns=[edge_index[tuple(sorted((q,j)))] for j in neg]
  rank+=(float(np.logaddexp.reduce(s[ix]))-float(np.mean(s[pos])))/28
  hard+=math.fsum(float(np.logaddexp(0,s[n]-s[p])) for p in pos for n in ns)/(28*5*len(pos))
 return {'bce':bce,'rank':rank,'hard':hard,'total':bce+rank+w*hard}
errors={};probes=0
for kind,r in reports.items():
 assert r['native_updates']==1 and r['parameter_count']==326571265 and r['backbone_layers']==24
 s=np.array(r['captured_training_logits']);assert s.shape==(378,)
 ref=reference(s,.5 if kind=='hard' else 0);errors[kind]={}
 for key in ('bce','rank','total')+(('hard',) if kind=='hard' else ()):
  errors[kind][key]=abs(ref[key]-r['update'][key]);assert errors[kind][key]<=3e-6
 for p in r['parameter_probes']:
  assert p['changed'] and p['adam_step']==1 and math.isfinite(p['gradient_norm_after_clip']) and p['gradient_norm_after_clip']>0
  probes+=1
 assert all('.pooler.' in n for n in r['unused_parameter_names'])
assert probes==116
for k,p in reports['hard']['loss_component_probe_norms'].items():assert k in ('bce','rank','hard') and all(math.isfinite(v) and v>0 for v in p.values())
assert set(reports['hard']['loss_component_probe_norms'])=={'bce','rank','hard'}
assert (cpu/'exit_status.txt').read_text().strip()=='0'
# Read only common historical source BYTES; do not open historical metric/data payloads.
with zipfile.ZipFile('/mnt/data/pooling_result_review.zip') as z:
 same=[]
 for f in audit['source_files']:
  p=f['path']
  if p in z.namelist() and (p.startswith('scripts/') or p.startswith('schema/') or p.startswith('tests/')):
   assert z.read(p)==(ROOT/p).read_bytes();same.append(p)
result={'zip_sha256':sha(path),'members':61,'sources_verified':count,'source_bytes':nbytes,'cpu_small_files':13,'cpu_small_bytes':130994,'runtime_source_files_verified':18,'native_record_files_verified':4,'parameter_probes_verified':probes,'scalar_losses_absolute_errors':errors,'maximum_scalar_error':max(e for v in errors.values() for e in v.values()),'original_wrapper_exact_fields':7,'historical_inherited_sources_byte_identical':same,'native_component_norms':reports['hard']['loss_component_probe_norms'],'formal_data_or_labels_read':0,'native_model_loaded':0,'native_updates_executed_here':0,'native_report_environment':audit['environment'],'reviewer_numpy':np.__version__}
(OUT/'package_cpu_crosscheck.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
