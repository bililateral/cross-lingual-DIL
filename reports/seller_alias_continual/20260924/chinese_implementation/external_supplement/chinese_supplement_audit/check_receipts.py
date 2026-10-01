"""Targeted attachment/receipt/source checks. Does not load models or research data."""
from pathlib import Path
import ast, hashlib, json, math, re, sys
ROOT=Path(sys.argv[1] if len(sys.argv)>1 else '/mnt/data/chinese_supplement_review')
OUT=Path(__file__).resolve().parent
B=ROOT/'reports/seller_alias_continual/20260924/chinese_implementation'
CPU=B/'cpu_initial'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def record_check(p,r):
 return {'path':r['path'],'actual_bytes':p.stat().st_size,'actual_sha256':sha(p),
         'match':p.stat().st_size==r.get('bytes',r.get('size_bytes')) and sha(p)==r['sha256']}
inv=read(B/'supplement_inventory.json')
items=[record_check(ROOT/x['path'],x) for x in inv['files']]
assert len(items)==53 and all(x['match'] for x in items)
evidence=read(CPU/'evidence_inventory.json')
eitems=[record_check(CPU/x['path'],x) for x in evidence['files']]
assert len(eitems)==30 and all(x['match'] for x in eitems)
assert sum(x['actual_bytes'] for x in eitems)==69793==evidence['total_bytes']
finalsrc=read(CPU/'final_sources.json')['verified_files']
sitems=[record_check(ROOT/x['path'],x) for x in finalsrc]
assert len(sitems)==11 and all(x['match'] for x in sitems)
assert finalsrc==read(B/'final_inputs.json')['files']
old=B/'native_source/step28_chinese_base.py'; new=ROOT/'scripts/step28_chinese_base.py'
trees=[ast.parse(p.read_text()) for p in (old,new)]
funcs=[{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))} for t in trees]
a,b=funcs
changed=sorted(n for n in a if a[n]!=b.get(n)); added=sorted(set(b)-set(a));removed=sorted(set(a)-set(b))
assert changed==['evaluate','train_arm'] and added==['fixed_classification'] and not removed
other=[]
for t in trees:
 other.append(ast.dump(ast.Module(body=[n for n in t.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))],type_ignores=[]),include_attributes=False))
assert other[0]==other[1]
semantics=read(CPU/'final_semantics.json')
assert semantics['old_sha256']==sha(old) and semantics['final_sha256']==sha(new)
assert semantics['changed_functions']==changed and semantics['added_functions']==added
linux=read(ROOT/'reports/seller_alias_continual/20260924/chinese_model/linux_verification.json')
policy=read(ROOT/'schema/step28_chinese_base_policy.json')
print('LINUX MODEL RECEIPT', {k:v for k,v in linux.items() if k!='files'})
rows=[]
for arm in ['mean_bce','mean_rank','split_bce','split_rank']:
 r=read(CPU/f'{arm}.json')
 assert r['arm']==arm and r['status']=='NATIVE_CPU_HANDMADE_UPDATE_VERIFIED'
 assert r['device']=='cpu' and r['torch']=='2.9.1+cu130'
 assert r['native_class']=='BertModel' and r['embedding_dimension']==1024 and r['backbone_layers']==24
 assert r['native_pooling_max_error']==0 and r['gradient_checkpointing_enabled']
 assert r['handmade_updates']==1 and r['formal_label_reads']==r['formal_training_updates']==r['saved_model_payloads']==0
 assert r['pair_count']==378 and r['positive_pairs']==20 and r['items']==56
 assert r['parameter_count']==r['trainable_parameter_count']
 assert r['optimized_parameter_tensors']==395 and r['actual_adam_state_tensors']==393
 assert len(r['unused_pooler_parameters'])==2 and all('.pooler.' in x for x in r['unused_pooler_parameters'])
 probes=r['layer_embedding_head_probes']
 layers=sorted(int(re.search(r'layer\.(\d+)\.',x['name'])[1]) for x in probes if '.attention.self.query.weight' in x['name'])
 assert layers==list(range(24)) and len(probes)==29
 assert sum('.embeddings.word_embeddings.weight' in x['name'] for x in probes)==1
 assert sum(x['name'].startswith('head.') for x in probes)==4
 assert all(x['changed'] and x['adam_step']==1 and math.isfinite(x['gradient_norm_after_clip']) and x['gradient_norm_after_clip']>0 for x in probes)
 assert all(math.isfinite(v) and v>0 for arr in r['objective_probe_gradient_norms'].values() for v in arr)
 for vals in r['update']['modules'].values(): assert all(vals.values())
 sources=[]
 for s in r['source_files']:
  p=ROOT/s['path']
  if s['path'] in ('scripts/step28_chinese_base.py','tests/test_step28_chinese_base_contracts.py'):
   p=B/'native_source'/Path(s['path']).name
  sources.append(record_check(p,s))
 assert len(sources)==10 and all(x['match'] for x in sources)
 archive=r['model_archive']
 canonical=json.dumps(archive['files'],sort_keys=True,separators=(',',':')).encode()
 assert len(archive['files'])==archive['file_count']==12
 assert sum(x['size_bytes'] for x in archive['files'])==archive['total_size_bytes']==1302803319
 assert hashlib.sha256(canonical).hexdigest()==archive['content_sha256']
 for k in ['files','file_count','total_size_bytes','content_sha256']: assert archive[k]==linux[k]
 for k in ['file_count','total_size_bytes','content_sha256']: assert archive[k]==policy['models'][arm][k]
 rankerr=abs(r['update']['rank']-r['independent_rank'])
 bceerr=abs(r['update']['bce']-r['independent_bce'])
 totalerr=abs(r['update']['total']-r['independent_bce']-policy['interventions'][arm]['rank_weight']*r['independent_rank'])
 assert rankerr<1e-6 and bceerr<1e-6 and totalerr<2e-6
 rows.append({'arm':arm,'parameter_count':r['parameter_count'],'probes':len(probes),'query_layers':len(layers),
 'seconds':r['seconds'],'bce':r['update']['bce'],'rank':r['update']['rank'],'total':r['update']['total'],
 'bce_abs_error':bceerr,'rank_abs_error':rankerr,'total_abs_error':totalerr,
 'cls_abs_error':r['native_pooling_max_error'],'logit_change_max':r['logit_change_max'],
 'max_rss_kib':r['max_rss_kib'],'source_checks':sources})
assert (CPU/'native_exit.txt').read_text().strip()=='0'
assert (CPU/'final_exit.txt').read_text().strip()=='0'
reported='2b298459235d05d0934a0c822a14f8b91014e5dd3797038b080eec5d9230225cc'
result={'status':'CHECKED','supplement_files':items,'evidence_files':eitems,'final_sources':sitems,
'native_to_final':{'native_sha256':sha(old),'final_sha256':sha(new),'changed_functions':changed,'added_functions':added,
'unchanged_function_count':len(a)-len(changed),'unchanged_functions':sorted(n for n in a if a[n]==b.get(n)),
'nonfunction_module_ast_identical':True,'removed_functions':removed},
'reported_sha_typo':{'reported':reported,'length':len(reported),'actual':sha(new),'actual_length':64,
'locations':[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and reported.encode() in p.read_bytes()]},
'native_reports':rows,'max_rank_abs_error':max(x['rank_abs_error'] for x in rows),
'linux_model_receipt':{k:v for k,v in linux.items() if k!='files'},
'limits':'All model/archive/native assertions cross-check submitted source-bound receipts, not an independent rerun of BGE or a rehash of remote weights.'}
(OUT/'receipt_source_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('supplement_files','evidence_files','final_sources','native_reports')},ensure_ascii=False,indent=2))
for row in rows: print(json.dumps({k:v for k,v in row.items() if k!='source_checks'},ensure_ascii=False))
