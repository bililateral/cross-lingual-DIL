"""Verify received archive and its immutable extracted payloads; no extraction or edits."""
import argparse,collections,hashlib,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--extracted',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
b=a.zip.read_bytes();s=hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(a.zip) as z:
 names=z.namelist();assert len(names)==len(set(names))
 assert z.testzip() is None
 m=json.loads(z.read('input_manifest.json'));records=m['files'];assert len({r['path'] for r in records})==len(records)
 assert set(names)=={r['path'] for r in records}|{'input_manifest.json'}
 for r in records:
  zb=z.read(r['path']);fb=(a.extracted/r['path']).read_bytes()
  assert zb==fb and len(zb)==r['bytes'] and hashlib.sha256(zb).hexdigest()==r['sha256'],r['path']
 assert (a.extracted/'input_manifest.json').read_bytes()==z.read('input_manifest.json')
 out=dict(status='PASS_ARCHIVE_MANIFEST_AND_UNCHANGED_EXTRACTED_INPUT',zip=a.zip.name,bytes=len(b),sha256=s,zip_members=len(names),zip_uncompressed_bytes=sum(i.file_size for i in z.infolist()),manifest_payload_count=len(records),payload_bytes=sum(r['bytes'] for r in records),manifest_sha256=hashlib.sha256(z.read('input_manifest.json')).hexdigest(),all_payloads_match=True,extracted_input_unchanged=True,duplicate_members=0,extra_members_except_manifest=[],missing_members=[],identity_boundary='Measured received bytes and internal manifest consistency; user did not supply a separate expected whole-ZIP digest in the current message.')
 assert s=='588e79e3a265319b9025fe94d16b4cc583f2d35cf4a51aa7fbc9b1fe54e4fb15'
a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
