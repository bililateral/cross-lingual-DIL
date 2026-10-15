from pathlib import Path
import json,hashlib,zipfile,subprocess,shutil,ast,time,platform,sys,argparse
parser=argparse.ArgumentParser(description='Read-only source/receipt audit; reverse patch only in output copies.')
parser.add_argument('input_root',type=Path);parser.add_argument('output_directory',type=Path);args=parser.parse_args()
start=time.perf_counter();r=args.input_root.resolve();b=r/'reports/documentation/20261015/balanced_pooling_repair';out=args.output_directory.resolve();out.mkdir(parents=True,exist_ok=True)
j=lambda p:json.loads(p.read_text());h=lambda p:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
manifest=j(b/'review/manifest.json');allfiles={str(p.relative_to(r)) for p in r.rglob('*') if p.is_file()}
assert all(h(r/e['path'])=={k:e[k] for k in ('bytes','sha256')} for e in manifest['entries'])
assert allfiles=={e['path'] for e in manifest['entries']}|{str((b/'review/manifest.json').relative_to(r))}
sm=j(b/'source_manifest.json');cpu1=j(b/'qualification/cpu/attempt01/result/result.json');cpu2=j(b/'qualification/cpu/attempt02/result/result.json')
assert sm==cpu2['source_files'] and len(sm)==30
assert all(h(r/e['path'])=={k:e[k] for k in ('bytes','sha256')} for e in sm)
# Reconstruct sources() closure without loading any scientific model or data.
paths={'scripts/step28_balanced_pooling_run.py','scripts/step28_balanced_pooling_verify.py','tests/test_step28_balanced_pooling.py','tests/test_step28_record_replay.py'};todo=list(paths)
while todo:
 for node in ast.walk(ast.parse((r/todo.pop()).read_text())):
  names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
  for name in names:
   if name and name.startswith('step28_'):
    child='scripts/'+name+'.py'
    if child not in paths:paths.add(child);todo.append(child)
paths.update('schema/'+n for n in ('step28_balanced_pooling_policy.json','step28_record_replay_policy.json','step28_bge_continual_policy.json','step28_alias_ranking_policy.json','step28_chinese_base_policy.json'))
paths.update(('docs/SELLER_ALIAS_BALANCED_POOLING.zh.md','docs/SELLER_ALIAS_BALANCED_POOLING_AMENDMENT.zh.md','scripts/run_step28_balanced_pooling_linux_20261015.sh'))
assert sorted(paths)==[e['path'] for e in sm]
z=zipfile.ZipFile(b/'qualification/cpu/attempt01/source.zip')
assert all({'bytes':len(z.read(e['path'])),'sha256':hashlib.sha256(z.read(e['path'])).hexdigest()}=={k:e[k] for k in ('bytes','sha256')} for e in cpu1['source_files'])
changed12=[e['path'] for e in sm if z.read(e['path'])!=(r/e['path']).read_bytes()]
received=j(b/'qualification/received_files.json')
assert len(received)==14
assert all(h(b/'qualification/cpu'/e['path'])=={k:e[k] for k in ('bytes','sha256')} for e in received)
p=j(r/'schema/step28_balanced_pooling_policy.json');refs=p['prior_evidence']
assert all(h(r/e['path'])=={k:e[k] for k in ('bytes','sha256')} for e in refs.values())
gate=j(r/refs['formal_gate']['path']);native=j(r/refs['native_result']['path']);oldsrc=gate['source_files']
assert oldsrc==native['source_files']
old={e['path']:e for e in oldsrc};new={e['path']:e for e in sm}
unchanged=[k for k in old.keys()&new.keys() if old[k]==new[k]];changed=[k for k in old.keys()&new.keys() if old[k]!=new[k]]
patch=b/'review/implementation.patch';recon=out/'reconstructed_old';recon.mkdir(exist_ok=True)
for n in changed:
 dest=recon/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r/n,dest)
cp=subprocess.run(['patch','--batch','--fuzz=0','-R','-p1','-i',str(patch)],cwd=recon,text=True,capture_output=True)
assert cp.returncode==0,(cp.stdout,cp.stderr)
restored={n:h(recon/n) for n in changed}
assert all(restored[n]=={k:old[n][k] for k in ('bytes','sha256')} for n in changed)
wrappers=[j(b/f'qualification/cpu/attempt{i:02}/wrapper.json') for i in (1,2)]
assert wrappers[0]['charged_seconds']==26 and wrappers[1]['charged_seconds']==45
assert wrappers[0]['prior_charged_seconds']==97 and wrappers[1]['prior_charged_seconds']==123
assert wrappers[1]['cumulative_charged_seconds']==168
assert int(86400-18428.570150192827)==67971
shells=[r/'scripts/run_step28_balanced_pooling_linux_20261015.sh',b/'qualification/cpu/attempt01/run.sh',b/'qualification/cpu/attempt02/run.sh']
syntax={str(x.relative_to(r)):subprocess.run(['bash','-n',str(x)],capture_output=True,text=True).returncode for x in shells}
assert all(v==0 for v in syntax.values())
report={'all_65_payload_hashes_match':True,'zip_members':len(allfiles),'current_sources':len(sm),'source_manifest_cpu2_and_AST_closure_match':True,'cpu1_source_zip_hashes_match':True,'cpu1_to_cpu2_changed_paths':changed12,'received_member_records_verified':14,'transport_archives_present':False,'all_policy_prior_evidence_identity_matches':True,'old_source_list_exactly_native_result':True,'unchanged_old_to_current_count':len(unchanged),'unchanged_paths':sorted(unchanged),'changed_common_paths':sorted(changed),'old_only_paths':sorted(old.keys()-new.keys()),'new_only_paths':sorted(new.keys()-old.keys()),'patch_reverse_stdout':cp.stdout,'patch_reverse_exact_old_hashes':restored,'cpu_wrapper_ledger':wrappers,'cpu_current_charge':168,'cpu_remaining':132,'native_old_updates':native['native']['actual_updates'],'native_old_charge':j(r/refs['qualification_state']['path'])['native']['charged_seconds'],'formal_prior':18428.570150192827,'formal_effective_limit':67971,'shell_syntax':syntax,'elapsed_seconds':time.perf_counter()-start,'environment':{'python':sys.version,'platform':platform.platform()},'scope':'Static identities, AST closure, reverse diff, shell syntax, receipt arithmetic only. No formal data or project Linux.'}
(out/'mechanical_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('unchanged_paths','patch_reverse_exact_old_hashes','cpu_wrapper_ledger')},indent=2,ensure_ascii=False))
# Compact extracts retaining JSON keys to support human examination.
for i,cpures in enumerate((cpu1,cpu2),1):
 (out/f'cpu_attempt{i:02}_result_without_sources.json').write_text(json.dumps({k:v for k,v in cpures.items() if k!='source_files'},indent=2,ensure_ascii=False)+'\n')
old_summary={k:v for k,v in native.items() if k not in ('source_files','native','completed_updates','measurements')}
old_summary['native_keys']=list(native['native'])
old_summary['native']={k:v for k,v in native['native'].items() if k in ('actual_updates','both_phases_encoder_head_effect','next_update_bitwise_equal','three_pooling_restores_verified','measurements','projection','solver_summary','equality_counts','maximum_records','channel_tokens','memory_bytes','pretrained_archive')}
(out/'old_native_selected.json').write_text(json.dumps(old_summary,indent=2,ensure_ascii=False)+'\n')
