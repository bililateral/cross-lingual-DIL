"""Package all generated review payloads; external transport manifest covers every ZIP member.
No models, Memory, formal data, project tests, or input ZIP are packaged or executed.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, zipfile
O=Path('/mnt/data/g_half_review');D=Path('/mnt/data')
Z=D/'g_half_review_20261015.zip'
outer_tsv=D/'g_half_review_members.tsv'
receipt_path=D/'g_half_review_delivery.json'
manifest_names={'MANIFEST.payload.json','MANIFEST.payload.tsv'}
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def write_json(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def entry(path):
 b=path.read_bytes();return {'path':path.relative_to(O).as_posix(),'bytes':len(b),'sha256':sha_bytes(b)}
# Append the actually completed artifact consistency operation without rewriting its original builder source/logs.
ledger_path=O/'execution_ledger.json';ledger=json.loads(ledger_path.read_text())
if not any(e['id']=='R08' for e in ledger['executions']):
 ledger['executions'].append({'id':'R08','activity':'交付前元数据/阅读行范围/输入身份与手写NPZ CRC核对','source':['tools/final_static_consistency.py'],'outputs':['logs/final_consistency.stdout.txt','logs/final_consistency.stderr.txt','logs/final_consistency.exit_code.txt'],'outcome':'EXIT_0','note':'不导入项目代码，不新增科学数值试验；随后封装结果由ZIP外部交付回执完整覆盖。'})
ledger['tool_response_warning']={'operation':'build_review_metadata','visible_diagnostic':'TERM environment variable not set.','transcript':'logs/build_metadata.tool_tail.txt','script_stderr_empty':True,'script_exit_code':0,'origin':'NOT_INDEPENDENTLY_IDENTIFIED','not_a_project_or_numerical_failure':True}
write_json(ledger_path,ledger)
files=sorted(p for p in O.rglob('*') if p.is_file() and p.name not in manifest_names)
assert all(p.name!=Z.name and not p.name.startswith('review_input') for p in files)
for p in files:
 assert not p.is_symlink()
 if p.suffix=='.json':json.loads(p.read_text())
rows=[entry(p) for p in files]
write_json(O/'MANIFEST.payload.json',{'scope':'Every review payload before adding the two manifests themselves. External g_half_review_members.tsv covers all ZIP members, including these manifests.','excluded_self_manifest_paths':sorted(manifest_names),'payload_count':len(rows),'files':rows})
(O/'MANIFEST.payload.tsv').write_text('path\tbytes\tsha256\n'+''.join(f"{r['path']}\t{r['bytes']}\t{r['sha256']}\n" for r in rows),encoding='utf-8')
all_files=sorted(p for p in O.rglob('*') if p.is_file())
with zipfile.ZipFile(Z,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in all_files:z.write(p,p.relative_to(O).as_posix())
with zipfile.ZipFile(Z) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(set(z.namelist()))==len(all_files)
 all_rows=[]
 for p in all_files:
  r=entry(p);b=z.read(r['path'])
  assert len(b)==r['bytes'] and sha_bytes(b)==r['sha256']
  all_rows.append(r)
outer_tsv.write_text('path\tbytes\tsha256\n'+''.join(f"{r['path']}\t{r['bytes']}\t{r['sha256']}\n" for r in all_rows),encoding='utf-8')
result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'archive':{'path':Z.name,'bytes':Z.stat().st_size,'sha256':sha_bytes(Z.read_bytes()),'members':len(all_rows),'crc_ok':True,'all_members_match_working_files':True},'complete_members_tsv':{'path':outer_tsv.name,'bytes':outer_tsv.stat().st_size,'sha256':sha_bytes(outer_tsv.read_bytes())},'input_archive_returned':False,'runtime_approval_gate_created':False,'payload_manifest_excludes_only_itself_and_tsv':True,'complete_zip_members':all_rows,'transport_note':'This external receipt and member TSV describe the final ZIP. They are transport metadata outside the ZIP to avoid a self-hash cycle; all review payloads, scientific check sources/inputs/raw outputs and retained failures are inside.'}
write_json(receipt_path,result)
print(json.dumps({k:v for k,v in result.items() if k!='complete_zip_members'},ensure_ascii=False,indent=2))
