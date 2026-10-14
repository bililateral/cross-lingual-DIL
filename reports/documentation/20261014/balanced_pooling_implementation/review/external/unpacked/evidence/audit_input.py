"""Read-only archive/receipt verification. No model, training, formal data or network."""
from pathlib import Path, PurePosixPath
import hashlib, json, sys, zipfile
archive, root, out = map(Path, sys.argv[1:4])
expected = '821415de49e848672d6ae5a025bbe88e63dfb4c4990c86aaf1cc0061b2fb6b6a'
h = lambda b: hashlib.sha256(b).hexdigest()
raw = archive.read_bytes()
assert len(raw) == 280892 and h(raw) == expected
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    assert len(names) == 69 and len(set(names)) == 69
    assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
    manifest = json.loads(z.read('input_manifest.json'))
    assert manifest['excludes_self'] and len(manifest['files']) == 68
    assert set(names) == {x['path'] for x in manifest['files']} | {'input_manifest.json'}
    inventory = []
    for name in names:
        b = z.read(name)
        assert (root / name).read_bytes() == b
        inventory.append({'path':name,'bytes':len(b),'sha256':h(b)})
    by_name = {v['path']:v for v in inventory}
    for x in manifest['files']:
        assert by_name[x['path']] == x
base = root/'reports/documentation/20261014/balanced_pooling_implementation'
receipts = []
for receipt in ('receive_manifest.json','receive_attempt02.json'):
    for x in json.loads((base/receipt).read_text()):
        b = (base/x['path']).read_bytes()
        assert len(b) == x['bytes'] and h(b) == x['sha256']
        receipts.append({'receipt':receipt,**x,'matches':True})
result = {'zip_bytes':len(raw),'zip_sha256':h(raw),'member_count':69,
          'payload_count':68,'all_payloads_match':True,'extracted_originals_unchanged':True,
          'receipt_files':len(receipts),'receipt_bytes':sum(v['bytes'] for v in receipts),
          'all_receipts_match':True}
(out/'input_members.json').write_text(json.dumps(sorted(inventory,key=lambda x:x['path']),ensure_ascii=False,indent=2))
(out/'receipt_verification.json').write_text(json.dumps({'entries':receipts,'count':len(receipts),'total_bytes':sum(v['bytes'] for v in receipts),'all_match':True},ensure_ascii=False,indent=2))
(out/'audit_input.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
