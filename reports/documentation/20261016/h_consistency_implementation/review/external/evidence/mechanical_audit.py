"""Read-only byte/provenance/AST audit, not project research qualification."""
from __future__ import annotations
import ast, hashlib, json, math, pathlib, re, zipfile
H=pathlib.Path(__file__).resolve().parent
R=H.parent/'input'
O=H/'mechanical_outputs';O.mkdir(exist_ok=False)
archive=H.parent.parent/'review_input(20261016-053330).zip'
if not archive.exists():archive=H.parent/'original_input.zip'
def read(p):return json.loads((R/p).read_text())
def digest(b):return hashlib.sha256(b).hexdigest()
def identity(p):
 b=p.read_bytes();return {'bytes':len(b),'sha256':digest(b)}
def matched(p,row):
 d=identity(p);assert all(d[k]==row[k] for k in ('bytes','sha256')),(str(p),d,row)
def save(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def bypath(rows):return {r['path']:{k:r[k] for k in ('bytes','sha256')} for r in rows}
summary={}
with zipfile.ZipFile(archive) as z:
 names=z.namelist();assert len(names)==len(set(names));assert z.testzip() is None
 for n in names:
  p=pathlib.PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
 manifest=read('manifest.json')
 assert set(names)==set(r['path'] for r in manifest['files'])|{'manifest.json'}
 for row in manifest['files']:
  b=z.read(row['path']);assert len(b)==row['bytes'] and digest(b)==row['sha256'];matched(R/row['path'],row)
 summary['archive']={'name':archive.name,**identity(archive),'members':len(names),'manifest_payloads':len(manifest['files']),'CRC_bad_members':0,'duplicate_members':0,'all_payloads_match':True}
receipt=read('cpu_receipt.json');mapping=read('source_mappings.json')
for row in receipt['source_files']:matched(R/'sources'/row['path'],row)
current=bypath(receipt['source_files'])
assert len(current)==42
summary['current_sources']=42
attempt_rows=[];prior=0
for name,mapped in mapping.items():
 original=read('cpu/'+name+'/sources.json');result=read('cpu/'+name+'/result/result.json');w=read('cpu/'+name+'/wrapper.json')
 assert bypath(mapped)==bypath(original)==bypath(result['source_files'])
 for row in mapped:matched(R/row['package_member'],row)
 assert all(result[k]==0 for k in ('errors','failures','skipped'))
 assert w['exit_code']==0 and w['prior_seconds']==prior and result['prior_compute_seconds']==prior
 charge=math.ceil(max(w['outer_elapsed_seconds'],result['elapsed_seconds']))
 assert charge==w['charged_seconds'];prior+=charge
 assert prior==w['cumulative_charged_seconds']
 text=(R/'cpu'/name/'result'/'unittest.txt').read_text()
 m=re.search(r'Ran (\d+) tests? in ([\d.]+)s',text);assert m
 assert int(m[1])==result['tests_run']==len(result['selected_tests'])
 attempt_rows.append({'attempt':name,'source_count':len(mapped),'tests':result['selected_tests'],
   'unittest_seconds':float(m[2]),'outer_seconds':w['outer_elapsed_seconds'],'internal_seconds':result['elapsed_seconds'],
   'charge':charge,'cumulative_charge':prior,'exit':w['exit_code'],'errors':result['errors'],'failures':result['failures'],'skipped':result['skipped'],
   'paths_differing_from_current':[p for p,v in bypath(mapped).items() if current[p]!=v]})
assert prior==receipt['cumulative_charged_seconds']==41
save('CPU_provenance_and_charges.json',attempt_rows)
summary['CPU']=attempt_rows
# Test provenance at method granularity; file hashes differ while unaffected methods do not.
test_paths=[R/'cpu'/a/'sources/tests/test_step28_h_consistency.py' for a in ('attempt01','attempt02')]+[R/'sources/tests/test_step28_h_consistency.py']
def methods(p):
 t=ast.parse(p.read_text());result={}
 for n in ast.walk(t):
  if isinstance(n,ast.ClassDef):
   for f in n.body:
    if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)):result[n.name+'.'+f.name]=ast.dump(f,include_attributes=False)
 return result
mm=[methods(p) for p in test_paths]
changes=[]
for i in (0,1):
 changes.append({'from':str(test_paths[i].relative_to(R)),'to':str(test_paths[i+1].relative_to(R)),
  'changed_methods':[k for k in sorted(set(mm[i])|set(mm[i+1])) if mm[i].get(k)!=mm[i+1].get(k)]})
assert changes[0]['changed_methods']==['HConsistencyTests.test_shared_table_pullback_and_finite_difference']
assert changes[1]['changed_methods']==['HConsistencyTests.check_saved_comparisons','HConsistencyTests.test_loss_gate_and_saved_comparison_interfaces']
for name,attempt in receipt['qualification_by_test'].items():
 # Receipt values are tested below according to actual representation.
 pass
summary['test_AST_changes']=changes
save('test_AST_changes.json',changes)
# Approved contract/auth identity (authorization is evidence, not an executable gate).
au=read('sources/reports/documentation/20261016/h_consistency_implementation/authorization.json')
contract=au['contract'];matched(R/'sources'/contract['path'],contract)
assert au['status']=='APPROVED_FULL_CONTRACT_GATED_SEQUENCE'
summary['authorization']={'status':au['status'],'contract':contract,'observed_at':au.get('observed_at')}
# Verify metadata-to-metadata alignment only; no remote original is read here.
ba=read('sources/reports/documentation/20261016/h_consistency/baseline_sources.json');de=read('deployment_receipt.json')
assert len(ba['files'])==len(de['baseline_files'])==52
assert bypath(ba['files'])==bypath(de['baseline_files'])
assert sum(x['bytes'] for x in ba['files'])==ba['total_bytes']==2940294
assert bypath(de['source_files'])==bypath(mapping['attempt01'])
for row in ba['unchanged_direct_sources']:matched(R/'sources'/row['path'],row)
assert len(ba['alignment']['group_ids'])==len(set(ba['alignment']['group_ids']))==40
assert len(ba['alignment']['columns'])==22
summary['baseline']={'files':52,'bytes':2940294,'approved_deployment_identity_metadata_equal':True,
 'remote_file_bytes_independently_read_by_reviewer':False,'unchanged_direct_sources_matched':8,
 'development_ids':40,'columns':22,'deployment_sources_equal_attempt01':True,
 'initial_identity_equal':ba['baselines']['S']['initial_identity']==ba['baselines']['H']['initial_identity']}
# Verify every new side of every unified diff hunk against delivered current bytes.
diff=(R/'changes.diff').read_text().splitlines(keepends=True)
sections=[]
for line in diff:
 if line.startswith('diff --git '):sections.append([])
 assert sections
 sections[-1].append(line)
patch_rows=[]
reconstructed=O/'reconstructed_pre_change';reconstructed.mkdir()
for sec in sections:
 plus=next(l for l in sec if l.startswith('+++ ')).strip()[6:]
 minus=next(l for l in sec if l.startswith('--- ')).strip()[4:]
 current_lines=(R/'sources'/plus).read_text().splitlines(keepends=True)
 hunks=[];j=0
 while j<len(sec):
  m=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',sec[j])
  if not m:j+=1;continue
  start_new=int(m[3])-1;count_new=int(m[4] or '1');j+=1;old=[];new=[]
  while j<len(sec) and not sec[j].startswith('@@ '):
   l=sec[j]
   if l.startswith('\\ No newline'):j+=1;continue
   if not l or l[0] not in ' +-':raise AssertionError(('bad hunk',plus,l))
   if l[0] in ' +':new.append(l[1:])
   if l[0] in ' -':old.append(l[1:])
   j+=1
  assert len(new)==count_new,(plus,len(new),count_new)
  assert current_lines[start_new:start_new+count_new]==new,('current side mismatch',plus,start_new)
  hunks.append((start_new,count_new,old,new))
 restored=[];cursor=0
 for start,n,old,new in hunks:restored.extend(current_lines[cursor:start]);restored.extend(old);cursor=start+n
 restored.extend(current_lines[cursor:]);restored=''.join(restored)
 if minus!='/dev/null':
  target=reconstructed/plus;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(restored)
 row={'path':plus,'hunks':len(hunks),'current_side_exact':True,'new_file':minus=='/dev/null','old_side_sha256_from_diff':digest(restored.encode()) if minus!='/dev/null' else None}
 if minus!='/dev/null' and plus.endswith('.py'):
  oldtree=ast.parse(restored);newtree=ast.parse(''.join(current_lines))
  def tops(t):return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  olddefs,newdefs=tops(oldtree),tops(newtree)
  row['changed_top_level_definitions']=[k for k in sorted(set(olddefs)|set(newdefs)) if olddefs.get(k)!=newdefs.get(k)]
 patch_rows.append(row)
summary['diff']={'sections':patch_rows,'claimed_base_commit':'3cd20ad3','base_commit_independently_available':False,'claim':'Current side verified; old side reconstructable from supplied diff only'}
save('diff_verification.json',summary['diff'])
# Native Adam comparison evidence audit: inspect the actual function, not expected claims.
p=R/'sources/scripts/step28_g_half_verify.py';tree=ast.parse(p.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='component_effects')
keys=sorted({n.slice.value for n in ast.walk(fn) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='on' and isinstance(n.slice,ast.Constant)})
assert keys==['grads','model']
summary['native_adam_evidence']={'component_effects_saved_state_keys_read':keys,'adam_compared_in_2_vs_3_or_4_vs_5':False,
 'adam_compared_in_5_vs_6':True,'contract_reference':'sources/docs/SELLER_ALIAS_H_CONSISTENCY.zh.md:166',
 'code_reference':'sources/scripts/step28_g_half_verify.py:44-72,158-170,178-192,244-250',
 'meaning':'Missing required per-component native Adam-change record, not proof of incorrect Adam update.'}
save('native_adam_evidence_audit.json',summary['native_adam_evidence'])
save('mechanical_summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
