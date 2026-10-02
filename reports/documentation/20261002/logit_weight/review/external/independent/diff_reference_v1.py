"""Reverse the seven changed-file unified diffs in an isolated tree; verify prior frozen bytes.
Commit object itself is not supplied and is not authenticated by this check.
"""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];src=ROOT/'submission';out=ROOT/'evidence/diff_reference';out.mkdir(exist_ok=False)
diff=(src/'reports/documentation/20261002/logit_weight/review/implementation.diff').read_text()
prior=json.loads((src/'reports/documentation/20261002/er_low/freeze.json').read_text())['source_files'];prior={r['path']:r for r in prior}
sections=re.split(r'(?=^diff --git )',diff,flags=re.M);rows=[]
for section in sections:
 if not section.strip():continue
 lines=section.splitlines(); match=re.match(r'diff --git a/(.+) b/(.+)',lines[0]);assert match and match[1]==match[2]
 path=match[1]; raw=(src/path).read_bytes();new=raw.decode().splitlines();old=[];pos=0; i=0;hunks=0
 while i<len(lines):
  if not lines[i].startswith('@@ '):i+=1;continue
  mt=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',lines[i]);assert mt
  at=int(mt[3])-1;assert at>=pos;old+=new[pos:at];pos=at;i+=1;hunks+=1
  while i<len(lines) and not lines[i].startswith('@@ '):
   s=lines[i];i+=1
   if s.startswith('\\ No newline'):continue
   assert s and s[0] in (' ','+','-'),repr(s)
   if s[0] in (' ','+'):
    assert pos<len(new) and new[pos]==s[1:],(path,pos+1,s,new[pos] if pos<len(new) else None)
    pos+=1
   if s[0] in (' ','-'):old.append(s[1:])
 old+=new[pos:]
 candidates=[(newline, (newline.join(old)+newline).encode()) for newline in ('\n','\r\n')]
 matching=[(nl,b) for nl,b in candidates if len(b)==prior[path]['bytes'] and hashlib.sha256(b).hexdigest()==prior[path]['sha256']]
 assert len(matching)==1,(path,'cannot reproduce historical frozen bytes')
 newline,blob=matching[0];target=out/'reconstructed_previous'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
 rows.append({'path':path,'hunks':hunks,'new_bytes':len(raw),'new_sha256':hashlib.sha256(raw).hexdigest(),'old_bytes':len(blob),'old_sha256':hashlib.sha256(blob).hexdigest(),'newline':repr(newline),'matches_low_freeze':True})
assert len(rows)==7
report={'status':'PASS','changed_files':rows,'changed_count':len(rows),'new_side_all_hunks_exact':True,'previous_side_all_seven_historical_frozen_bytes_exact':True,'declared_base_commit':'646ed9e226ab2e5993f762ac1014201841873cfc','commit_object_authentication':'NOT_AVAILABLE: no Git object supplied; no Git required or performed','submission_mutated':False}
(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
