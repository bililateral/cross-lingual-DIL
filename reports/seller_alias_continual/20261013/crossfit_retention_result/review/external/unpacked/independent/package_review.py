#!/usr/bin/env python3
"""Create a complete ZIP and per-file SHA manifests; no scientific computation."""
from pathlib import Path
import csv, datetime, hashlib, json, zipfile

BASE=Path(__file__).resolve().parents[1]
DEST=BASE.parent/'C_cf_result_external_review_20261013.zip'

def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    excluded={'FILES_MANIFEST.json','FILES_MANIFEST.csv'}
    paths=sorted(p for p in BASE.rglob('*') if p.is_file() and str(p.relative_to(BASE)) not in excluded)
    entries=[{'path':str(p.relative_to(BASE)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in paths]
    manifest={'scope':'All delivery payloads except the two self-referential manifests. Original input archive and all 303 extracted members retained.',
      'payload_count':len(entries),'payload_bytes':sum(r['bytes'] for r in entries),'files':entries}
    (BASE/'FILES_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with (BASE/'FILES_MANIFEST.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(entries)
    allpaths=sorted(p for p in BASE.rglob('*') if p.is_file())
    with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in allpaths:z.write(p,str(p.relative_to(BASE)))
    with zipfile.ZipFile(DEST) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(allpaths)==len(set(z.namelist()))
        for rec in entries:
            data=z.read(rec['path'])
            assert len(data)==rec['bytes'] and sha(data)==rec['sha256']
        assert z.read('input_archive.zip')==(BASE/'input_archive.zip').read_bytes()
    receipt={'status':'COMPLETE_ARCHIVE_CRC_AND_PAYLOAD_SHA_VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'file':str(DEST),'bytes':DEST.stat().st_size,'sha256':sha(DEST.read_bytes()),'zip_members':len(allpaths),
      'manifest_payloads':len(entries),'manifests':[{'path':n,'bytes':(BASE/n).stat().st_size,'sha256':sha((BASE/n).read_bytes())} for n in sorted(excluded)],
      'source_sha256':sha(Path(__file__).read_bytes()),'self_reference_note':'This final archive receipt is external to the ZIP to avoid a digest cycle.'}
    (BASE.parent/'C_cf_result_external_review_20261013_delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
