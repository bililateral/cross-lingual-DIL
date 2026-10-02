"""Rebuild report from recorded evidence, verify all identities, and deliver one ZIP.
No project scientific entrypoints, installed models, formal files, or network are used.
The final ZIP cannot contain its own hash; its receipt is written outside the ZIP.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'logit_review_evidence.zip'
RECEIPT = ROOT.parent / 'logit_review_evidence_delivery.json'
EXPECTED = 'ff678c2f9687ced6fa9888eca55064ca617c98d35bf371e082e7cf321f9ed08a'
EXCLUDED = {'FILE_MANIFEST.json', 'SHA256SUMS.txt'}
def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def meta(path: Path) -> dict:
    b = path.read_bytes()
    return {'bytes': len(b), 'sha256': sha(b)}
def now() -> str: return dt.datetime.now(dt.timezone.utc).isoformat()
def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

started = now(); tick = time.monotonic()
assembly = ROOT / 'evidence' / 'artifact_assembly'
assembly.mkdir(parents=True, exist_ok=False)
argv = [sys.executable, '-B', str(ROOT/'independent/build_review.py')]
env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'
bm = {'purpose':'Artifact assembly only; the 15 scientific/verification commands have separate original metadata',
      'command_argv':argv,'cwd':str(ROOT),'start_utc':now(),
      'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                     'PYTHONDONTWRITEBYTECODE':'1'}}
t = time.monotonic()
with (assembly/'stdout.txt').open('wb') as out, (assembly/'stderr.txt').open('wb') as err:
    result = subprocess.run(argv, cwd=ROOT, env=env, stdout=out, stderr=err, timeout=30)
bm.update(end_utc=now(),wall_seconds=time.monotonic()-t,exit_code=result.returncode,
          stdout=meta(assembly/'stdout.txt'),stderr=meta(assembly/'stderr.txt'))
dump(assembly/'command.json',bm)
if result.returncode != 0: raise RuntimeError('Report rebuild failed; preserved raw streams')

original = (ROOT/'original/logit_review.zip').read_bytes()
assert len(original)==2084038 and sha(original)==EXPECTED
assert (ROOT.parent/'logit_review.zip').read_bytes()==original
with zipfile.ZipFile(ROOT/'original/logit_review.zip') as z:
    assert len(z.namelist())==304 and len(set(z.namelist()))==304 and z.testzip() is None
    original_members = z.namelist()
    for name in original_members: assert (ROOT/'submission'/name).read_bytes()==z.read(name)
assert sorted(str(p.relative_to(ROOT/'submission')) for p in (ROOT/'submission').rglob('*') if p.is_file())==sorted(original_members)
commands=[]
for p in sorted((ROOT/'evidence').glob('[0-9][0-9]_*/command.json')):
    d=json.loads(p.read_text());commands.append(d)
    for stream in ('stdout.txt','stderr.txt'): assert meta(p.parent/stream)==d[stream]
assert len(commands)==15 and sum(d['exit_code']==0 for d in commands)==13
ledger=json.loads((ROOT/'COMMAND_LEDGER.json').read_text()); assert len(ledger)==15
# Confirm every relative Markdown link in the three entry documents points at a real file.
import re
links=[]
for name in ['README.zh.md','REVIEW.zh.md','COMMANDS.zh.md']:
    for dest in re.findall(r'\]\(([^)]+)\)',(ROOT/name).read_text()):
        if not dest.startswith(('https:','http:','#')):
            assert (ROOT/dest).is_file(), (name,dest)
            links.append(dest)
preflight={'status':'PASS','checked_at_utc':now(),'original_zip':meta(ROOT/'original/logit_review.zip'),
           'original_members_byte_identical':304,'submission_extra_files':0,'recorded_verification_commands':15,
           'recorded_command_raw_stream_hash_checks':30,'recorded_nonzero_commands':2,
           'artifact_assembly_exit_code':0,'relative_markdown_links_checked':len(links),
           'original_project_mutations':0,'scientific_work_rerun_by_packager':False}
dump(assembly/'preflight.json',preflight)

# Do not mutate any payload after this point.
records=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and str(p.relative_to(ROOT)) not in EXCLUDED:
        records.append({'path':str(p.relative_to(ROOT)),**meta(p)})
manifest={'purpose':'Every delivered payload except the two self-referential index files',
          'excluded_index_files':sorted(EXCLUDED),'payload_count':len(records),
          'payload_bytes':sum(r['bytes'] for r in records),'files':records}
dump(ROOT/'FILE_MANIFEST.json',manifest)
(ROOT/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in records),encoding='utf-8')
with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(ROOT.rglob('*')):
        if p.is_file(): z.write(p,str(p.relative_to(ROOT)))
with zipfile.ZipFile(OUT) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(records)+len(EXCLUDED)
    assert len(z.namelist())==len(set(z.namelist()))
    assert set(z.namelist())=={r['path'] for r in records}|EXCLUDED
    for r in records:
        b=z.read(r['path']); assert len(b)==r['bytes'] and sha(b)==r['sha256']
    assert z.read('original/logit_review.zip')==original
    assert z.read('FILE_MANIFEST.json')==(ROOT/'FILE_MANIFEST.json').read_bytes()
    assert z.read('SHA256SUMS.txt')==(ROOT/'SHA256SUMS.txt').read_bytes()
receipt={'status':'PASS','purpose':'External delivery receipt; this file is not inside the ZIP whose digest it gives',
         'start_utc':started,'end_utc':now(),'wall_seconds':time.monotonic()-tick,
         'command_argv':[sys.executable,'-B',str(Path(__file__).resolve())],
         'cwd':str(Path.cwd()),'exit_code':0,
         'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform()},
         'evidence_zip':{'path':str(OUT),**meta(OUT),'members':len(records)+len(EXCLUDED)},
         'original_zip':{'path':'original/logit_review.zip',**meta(ROOT/'original/logit_review.zip'),'members':304},
         'report':{'path':'REVIEW.zh.md',**meta(ROOT/'REVIEW.zh.md')},
         'manifest':{'path':'FILE_MANIFEST.json',**meta(ROOT/'FILE_MANIFEST.json')},
         'sha_index':{'path':'SHA256SUMS.txt',**meta(ROOT/'SHA256SUMS.txt')},
         'checks':{'zip_crc_pass':True,'duplicate_member_names':0,'manifest_payloads_verified':len(records),
                   'original_zip_preserved_byte_identical':True,'original_304_members_preserved':True,
                   'all_15_recorded_command_raw_streams_verified':True,'all_relative_entrypoint_links_exist':True},
         'scope':'Handwritten CPU evidence; no new formal LOGIT results; no native BGE or project Linux connection'}
dump(RECEIPT,receipt)
print(json.dumps(receipt,ensure_ascii=False,indent=2))
