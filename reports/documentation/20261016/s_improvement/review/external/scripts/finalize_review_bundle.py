"""Read-only input recheck and artifact packaging, not a scientific evaluator."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent
sha = lambda b: hashlib.sha256(b).hexdigest()
initial = json.loads((ROOT/'checks/input_identity.json').read_text())
checks = []
for a in initial['attachments']:
    path=ROOT/'originals'/a['name']; b=path.read_bytes()
    assert len(b)==a['bytes'] and sha(b)==a['sha256']
    mounted=OUT/a['name']
    matches_mount = None
    if mounted.exists():
        matches_mount = b==mounted.read_bytes()
        assert matches_mount
    checks.append({**a,'original_copy_unchanged':True,'matches_original_mount_if_available':matches_mount})
with zipfile.ZipFile(ROOT/'originals/s_improvement_review_input.zip') as z:
    assert z.testzip() is None
    names=z.namelist()
    assert len(names)==22 and len(set(names))==22
    m=json.loads(z.read('input_manifest.json'))
    assert set(names)=={i['member'] for i in m['files']}|{'input_manifest.json'}
    for name in names:
        assert (ROOT/'input'/name).read_bytes()==z.read(name),name
    assert {p.relative_to(ROOT/'input').as_posix() for p in (ROOT/'input').rglob('*') if p.is_file()}==set(names)
pre=json.loads((ROOT/'checks/handwritten_pre_execution_identity.json').read_text())
for name, info in pre.items():
    b=(ROOT/name).read_bytes()
    assert len(b)==info['bytes'] and sha(b)==info['sha256']
math=json.loads((ROOT/'checks/handwritten_results.json').read_text())
assert math['script_sha256']==pre['scripts/handwritten_math.py']['sha256']
assert math['input_sha256']==pre['checks/handwritten_inputs.json']['sha256']
assert math['status']=='PASS_HANDWRITTEN_ALGEBRA_ONLY'
assert (ROOT/'logs/handwritten_math.stderr.txt').read_bytes()==b''
report=(ROOT/'review.zh.md').read_text()
assert all(f'| F{i:02d} |' in report for i in range(1,15))
assert '5.560268912e−11' in report and '1.542453448e−11' in report
assert abs(math['queries'][1]['by_dtype']['float64']['old_gradient'][0]-.4256821258485005)<1e-15
final={'status':'PASS_UNCHANGED_INPUTS_AND_REVIEWER_MATH_IDENTITIES','attachments':checks,
       'zip_members_and_extracted_bytes_equal':22,'payloads_match_original_manifest':21,
       'handwritten_source_and_input_unchanged_since_before_execution':True,
       'scope':'Packaging integrity only; no scientific/project re-execution'}
(ROOT/'checks/input_identity_final.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
manifest_path=ROOT/'payload_manifest.json'
entries=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or p==manifest_path:continue
    b=p.read_bytes();entries.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':sha(b)})
manifest={'scope':'All ZIP members except this manifest; external TSV also covers this manifest itself.',
          'payload_count':len(entries),'files':entries}
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
archive=OUT/'s_pos_h_pos_static_review.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(ROOT.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(entries)+1
    for e in entries:
        b=z.read(e['path']); assert len(b)==e['bytes'] and sha(b)==e['sha256']
    all_members=[{'path':i.filename,'bytes':i.file_size,'sha256':sha(z.read(i.filename))} for i in z.infolist()]
with (OUT/'s_pos_h_pos_members.tsv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256'],delimiter='\t');w.writeheader();w.writerows(all_members)
receipt={'file':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive.read_bytes()),
         'members':len(all_members),'uncompressed_bytes':sum(i['bytes'] for i in all_members),
         'internal_manifest_payloads':len(entries),'status':'CRC and all internal payload identities checked',
         'report_path':'s_pos_static_review/review.zh.md','all_member_manifest':'s_pos_h_pos_members.tsv'}
(OUT/'s_pos_h_pos_archive_identity.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
