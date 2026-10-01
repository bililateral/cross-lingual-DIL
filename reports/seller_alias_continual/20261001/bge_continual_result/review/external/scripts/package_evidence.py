"""Package immutable input and completed audit evidence; pack receipt is external."""
from __future__ import annotations
import datetime,hashlib,json,pathlib,shlex,time,zipfile
B=pathlib.Path('/mnt/data/bge_review_evidence'); Z=pathlib.Path('/mnt/data/bge_continual_independent_review_evidence.zip'); start=time.perf_counter()
assert not Z.exists(),'Do not overwrite a previously delivered evidence archive.'
cmds=['# 实际执行命令与回执\n','这些是本次真实执行历史，包含已失败命令；不是要求用户本地执行。所有核验仅使用附件保存的小结果，正式标签读取为0。\n']
records=[]
for p in sorted((B/'logs').glob('*.json')):
 if p.stem=='14_package':continue
 d=json.loads(p.read_text());records.append(d)
 for suffix in ('stdout','stderr'):
  raw=p.with_suffix('.'+suffix).read_bytes()
  assert len(raw)==d[suffix]['bytes'] and hashlib.sha256(raw).hexdigest()==d[suffix]['sha256']
 cmds += [f"## {p.stem}\n",f"cwd: `{d['cwd']}`；退出码：{d['exit_code']}；壁钟：{d['elapsed_seconds']:.9f} 秒。\n",f"UTC {d['started_at_utc']} → {d['ended_at_utc']}\n",'```sh\n'+shlex.join(d['command'])+'\n```\n',f"原始回执：`logs/{p.stem}.json`；原始流：`.stdout` / `.stderr`；退出：`.exit`。\n"]
cmds+=['## 最终打包\n','`python /mnt/data/bge_review_evidence/scripts/run_recorded.py 14_package -- python /mnt/data/bge_review_evidence/scripts/package_evidence.py`\n','最终ZIP不能自包含自身最终哈希及打包结束日志；该机械打包步骤的精确命令、原始stdout/stderr、退出码、耗时与ZIP哈希均收在外部交付receipt。所有此前完成的核验/修订/交付检查命令（00—13）已完整包含在本ZIP内。\n']
(B/'COMMANDS.md').write_text('\n'.join(cmds),encoding='utf-8')
def selected(p):
 rel=p.relative_to(B).as_posix()
 return p.is_file() and rel not in ('MANIFEST.json','SHA256SUMS.txt') and not rel.startswith('logs/14_package.')
paths=sorted([p for p in B.rglob('*') if selected(p)],key=lambda p:p.relative_to(B).as_posix())
rows=[]
for p in paths:
 d=p.read_bytes();rows.append({'path':p.relative_to(B).as_posix(),'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()})
manifest={'format':'review-evidence-manifest-v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'payload_files':len(rows),'completed_command_records_included':len(records),'files':rows,'exclusions':['MANIFEST.json self-hash','SHA256SUMS.txt self-hash','logs/14_package.* (final packaging receipt provided separately)'],'input_archive_sha256':'1a9449d1139a0ab951712c4bb7b3bda47b9cc528f11bf6dda9ca27c01db15781','formal_labels_read':0,'new_formal_training':False}
(B/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
sha=''.join(f"{r['sha256']}  {r['path']}\n" for r in rows)
sha+=hashlib.sha256((B/'MANIFEST.json').read_bytes()).hexdigest()+'  MANIFEST.json\n'
(B/'SHA256SUMS.txt').write_text(sha)
allpaths=paths+[B/'MANIFEST.json',B/'SHA256SUMS.txt']
with zipfile.ZipFile(Z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in allpaths:
  rel=p.relative_to(B).as_posix()
  z.write(p,rel,compress_type=zipfile.ZIP_STORED if p.suffix=='.zip' else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(Z) as z:
 assert z.testzip() is None
 assert set(z.namelist())=={p.relative_to(B).as_posix() for p in allpaths}
 for row in rows:
  d=z.read(row['path']);assert len(d)==row['bytes'];assert hashlib.sha256(d).hexdigest()==row['sha256']
 for line in z.read('SHA256SUMS.txt').decode().splitlines():
  digest,rel=line.split('  ',1);assert hashlib.sha256(z.read(rel)).hexdigest()==digest
res={'status':'PASS_FINAL_ZIP_CRC_MEMBER_SET_AND_ALL_SHA','archive':str(Z),'bytes':Z.stat().st_size,'sha256':hashlib.sha256(Z.read_bytes()).hexdigest(),'zip_members':len(allpaths),'payload_files':len(rows),'completed_command_records_included':len(records),'report_bytes':(B/'REVIEW.zh.md').stat().st_size,'report_sha256':hashlib.sha256((B/'REVIEW.zh.md').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((B/'MANIFEST.json').read_bytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-start,'final_packaging_receipt_external':True}
print(json.dumps(res,ensure_ascii=False,indent=2))
