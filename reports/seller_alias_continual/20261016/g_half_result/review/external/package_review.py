#!/usr/bin/env python3
"""Package immutable review payloads; publish an external all-member inventory.
The internal manifest excludes itself. The external inventory includes it.
No attempt is made to place a ZIP's own digest inside that ZIP.
"""
from pathlib import Path
import csv,hashlib,json,zipfile
R=Path(__file__).resolve().parent;D=R.parent
ZIP=D/'h_external_review.zip';M=R/'payload_manifest.json'
def digest(b):return hashlib.sha256(b).hexdigest()
items=[]
for p in sorted(R.rglob('*')):
 if p.is_file() and p!=M:
  b=p.read_bytes();items.append({'path':p.relative_to(R).as_posix(),'bytes':len(b),'sha256':digest(b)})
M.write_text(json.dumps({'schema':'external-review-payload-manifest-v1','excluded_self':'payload_manifest.json','note':'All input hashes and source bounds are in the payload; original user ZIP not duplicated. External members inventory covers this manifest too.','payloads':items},ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in sorted(R.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(R).as_posix())
all_items=[]
with zipfile.ZipFile(ZIP) as z:
 assert z.testzip() is None
 for i in z.infolist():
  b=z.read(i.filename);assert b==(R/i.filename).read_bytes()
  all_items.append({'path':i.filename,'bytes':len(b),'sha256':digest(b)})
assert len(all_items)==len(items)+1
with (D/'h_external_review.members.tsv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['path','bytes','sha256'],delimiter='\t');w.writeheader();w.writerows(all_items)
(D/'h_external_review.members.json').write_text(json.dumps(all_items,ensure_ascii=False,indent=2)+'\n')
b=ZIP.read_bytes();delivery={'zip':ZIP.name,'bytes':len(b),'sha256':digest(b),'zip_members':len(all_items),'payload_members_excluding_internal_manifest':len(items),'uncompressed_member_bytes':sum(x['bytes'] for x in all_items),'zip_crc_ok':True,'all_zip_bytes_match_review_files':True,'member_inventories':[],'input_zip':{'filename':'review_input(20261016-014443).zip','bytes':2798254,'sha256':'c1831a58ba41c7ff46c77b782aad9789730d234e9bf50faa836094cd23cc3880','members':219}}
for name in ['h_external_review.members.tsv','h_external_review.members.json']:
 b=(D/name).read_bytes();delivery['member_inventories'].append({'path':name,'bytes':len(b),'sha256':digest(b)})
(D/'h_external_review.delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(delivery,ensure_ascii=False,indent=2))
