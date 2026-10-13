#!/usr/bin/env python3
"""Package this review, verify every member, and write an external non-circular receipt."""
from __future__ import annotations
import hashlib
import json
import pathlib
import shutil
import zipfile

ROOT = pathlib.Path('/mnt/data/S_contribution_review')
DEST = pathlib.Path('/mnt/data')
STEM = 'S_contribution_static_review_20261013'
MANIFESTS = {'MANIFEST.json', 'MANIFEST.sha256.tsv'}

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def details(path: pathlib.Path) -> dict:
    data = path.read_bytes()
    return {'path':path.relative_to(ROOT).as_posix(), 'bytes':len(data), 'sha256':sha(data)}

def json_bytes(data) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2)+'\n').encode('utf-8')

def tsv(rows) -> str:
    return 'path\tbytes\tsha256\n'+''.join(f"{r['path']}\t{r['bytes']}\t{r['sha256']}\n" for r in rows)

files = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.relative_to(ROOT).as_posix() not in MANIFESTS)
rows = [details(p) for p in files]
(ROOT/'MANIFEST.json').write_bytes(json_bytes({'scope':'all payload files, excluding MANIFEST.json and MANIFEST.sha256.tsv themselves', 'hash':'SHA-256','payload_count':len(rows),'files':rows}))
(ROOT/'MANIFEST.sha256.tsv').write_text(tsv(rows), encoding='utf-8')
all_paths = sorted(p for p in ROOT.rglob('*') if p.is_file())
archive = DEST/(STEM+'.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in all_paths:
        z.write(p, p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(set(z.namelist())) == len(all_paths)
    for r in rows:
        b=z.read(r['path'])
        assert len(b)==r['bytes'] and sha(b)==r['sha256']
    assert z.read('review.zh.md') == z.read('review.zh.txt')
    original = z.read('input/review_input(20261013-031814).zip')
    assert len(original)==198789 and sha(original)=='3db153cbfefd597d2a35b6e53bdff5500fecd447a568d8a2ea1345078b92597c'
    inner=json.loads(z.read('MANIFEST.json'))
    assert inner['files']==rows
all_rows=[details(p) for p in all_paths]
listing=DEST/(STEM+'.files.tsv')
listing.write_text(tsv(all_rows),encoding='utf-8')
receipt={'archive':{'filename':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive.read_bytes()),'members':len(all_paths)},'validation':{'zip_crc':'PASS','member_hashes':'PASS','original_input_unchanged':True,'markdown_text_equal':True},'internal_manifests':[r for r in all_rows if r['path'] in MANIFESTS],'external_all_member_listing':{'filename':listing.name,'bytes':listing.stat().st_size,'sha256':sha(listing.read_bytes())},'no_self_reference':'Internal manifests exclude themselves; the external TSV includes all ZIP members; this receipt and packaging run logs are outside the ZIP.'}
(DEST/(STEM+'.receipt.json')).write_bytes(json_bytes(receipt))
text = '# 交付回执\n\n'
text += f"ZIP：{archive.name}\n字节：{archive.stat().st_size}\nSHA-256：{receipt['archive']['sha256']}\n成员：{len(all_paths)}\n\n"
text += 'CRC与逐成员大小/SHA检查：PASS。原输入ZIP字节未变；MD/TXT正文一致。\n\n逐成员清单含两个内部manifest；内部manifest不自引用。交付回执与打包日志放在ZIP外。\n\n'
text += tsv(all_rows)
(DEST/(STEM+'.receipt.txt')).write_text(text,encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
