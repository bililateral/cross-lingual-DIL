#!/usr/bin/env python3
"""Verify uploaded archive bytes and payload hashes; never import project code."""
import hashlib,json,zipfile,pathlib,sys
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); b=src.read_bytes()
with zipfile.ZipFile(src) as z:
    m=json.loads(z.read('input_manifest.json')); names=z.namelist(); rows=[]
    for e in m['entries']:
        p=z.read(e['member']); rows.append(dict(e,actual_bytes=len(p),actual_sha256=hashlib.sha256(p).hexdigest(),match=len(p)==e['bytes'] and hashlib.sha256(p).hexdigest()==e['sha256']))
    r=dict(filename=src.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),members=len(names),manifest_payload_count=m['payload_count'],actual_payload_count=len(rows),duplicate_names=len(names)!=len(set(names)),unlisted=sorted(set(names)-{e['member'] for e in m['entries']}-{'input_manifest.json'}),missing=sorted({e['member'] for e in m['entries']}-set(names)),all_payloads_match=all(e['match'] for e in rows),entries=rows)
out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in r.items() if k!='entries'},ensure_ascii=False,indent=2))
assert r['all_payloads_match'] and not r['duplicate_names'] and not r['unlisted'] and not r['missing']
