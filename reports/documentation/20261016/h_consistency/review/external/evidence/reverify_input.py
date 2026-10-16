#!/usr/bin/env python3
"""Read-only input integrity audit; no project execution or extraction."""
from pathlib import Path
import sys,json,hashlib,zipfile
root=Path(__file__).resolve().parent.parent
archive=root/'input'/'review_input.zip'
extracted=Path(sys.argv[1]) if len(sys.argv)>1 else None
hash256=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    manifest=json.loads(z.read('manifest.json'))
    entries=manifest['payloads']
    rows=[]
    for e in entries:
        p=e['path']; raw=z.read(p)
        row={'path':p,'bytes':len(raw),'sha256':hash256(raw),
             'manifest_matches':len(raw)==e['bytes'] and hash256(raw)==e['sha256']}
        if extracted is not None:
            row['extracted_unchanged']=(extracted/p).read_bytes()==raw
        rows.append(row)
    result={'archive_bytes':archive.stat().st_size,'archive_sha256':hash256(archive.read_bytes()),
            'zip_members':len(z.namelist()),'crc_bad_member':z.testzip(),
            'unlisted':sorted(set(z.namelist())-{'manifest.json'}-{e['path'] for e in entries}),
            'rows':rows}
result['all_match']=all(r['manifest_matches'] and r.get('extracted_unchanged',True) for r in rows)
print(json.dumps(result,ensure_ascii=False,indent=2))
if not result['all_match']:sys.exit(1)
