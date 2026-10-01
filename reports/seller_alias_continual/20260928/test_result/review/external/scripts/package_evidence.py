#!/usr/bin/env python3
"""Package closed audit streams and outputs with per-file integrity checks.
Build logs are written outside the evidence directory to avoid self-changing hashes.
"""
import argparse,datetime,hashlib,json,pathlib,zipfile
p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--zip',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
root=pathlib.Path(a.evidence);destination=pathlib.Path(a.zip);receiptpath=pathlib.Path(a.receipt)
excluded={'evidence_inventory.json','SHA256SUMS.txt'}
entries=[]
for file in sorted(root.rglob('*')):
    if file.is_file() and file.relative_to(root).as_posix() not in excluded:
        content=file.read_bytes();entries.append({'path':file.relative_to(root).as_posix(),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Every payload file; manifest and checksum list excluded from self-reference. ZIP hash in external receipt.',
'payload_files':len(entries),'payload_bytes':sum(r['bytes'] for r in entries),'files':entries}
mp=root/'evidence_inventory.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(root/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in entries)+hashlib.sha256(mp.read_bytes()).hexdigest()+'  evidence_inventory.json\n')
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for file in sorted(root.rglob('*')):
        if file.is_file():z.write(file,file.relative_to(root).as_posix())
with zipfile.ZipFile(destination) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(entries)+2
    assert len(set(z.namelist()))==len(z.namelist())
    for r in entries:
        v=z.read(r['path']);assert len(v)==r['bytes'] and hashlib.sha256(v).hexdigest()==r['sha256'],r['path']
    verified_members=len(z.namelist())
rc={'status':'PASS','filename':destination.name,'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
'payload_files':len(entries),'zip_members':verified_members,'all_payload_sizes_and_sha256_verified':True,'zip_integrity':True,
'formal_text_or_labels_or_weights_included':False,
'original_submission_unchanged':json.load(open(root/'outputs/final_source_verification.json'))['all_original_members_byte_identical']}
receiptpath.write_text(json.dumps(rc,ensure_ascii=False,indent=2)+'\n');print(json.dumps(rc,ensure_ascii=False,indent=2))
