"""Read-only provenance check for this discussion. No project code or data loading."""
from __future__ import annotations
import hashlib, json, platform, sys, zipfile
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
LAYERS=[
('current',ROOT/'original_input.zip',ROOT/'input','7ab2a3a079f533976a763e628e20d646a342cb4f7dd60866d452994305eb7159',12494965),
('result_external',ROOT/'input/history/relation_revision_result_external.zip',ROOT/'history/result_external','cf997222d0f32696d5722d8611c4c93c2edcf7a26e9c00120a5571f64c5cc1be',12178142),
('result_input',ROOT/'history/result_external/review_input.zip',ROOT/'history/result_input','3df2d543301bfc4ad80e1d9bf33a7a3f04123d71b0068188308a4b7a34bfd9d3',11461459),
('pilot_external',ROOT/'history/result_input/history/relation_revision_pilot_external_evidence.zip',ROOT/'history/pilot_external','c42e6e0232927ef7c4bc8decc5e9dbebc4889a036cb008ceb88e86a0c1346a6c',5958634),
('pilot_input',ROOT/'history/pilot_external/input/relation_revision_pilot_review.zip',ROOT/'history/pilot_input','992990d9c1b5f9c5053b6fc675509506ff4d66c5da22247d8c156e818ac2de84',5427862),
('revision_input',ROOT/'history/pilot_input/history/relation_revision_review.zip',ROOT/'history/revision_input','480371343ea0be50e4f8fcc614ec1aee5b35b20593151b7fbb3932e75669a9b5',5257725)]
out=[]
for name, archive, destination, expected, size in LAYERS:
    raw=archive.read_bytes(); assert len(raw)==size and sha(raw)==expected
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=[x.filename for x in z.infolist() if not x.is_dir()]
        assert len(names)==len(set(names))
        for item in z.infolist():
            p=PurePosixPath(item.filename)
            assert not p.is_absolute() and '..' not in p.parts
            assert (item.external_attr>>16)&0o170000 !=0o120000
        # The delivered older review_input deliberately has a list manifest.
        mf=json.loads(z.read('manifest.json'))
        rows=mf if isinstance(mf,list) else mf['files']
        checks=[]
        for r in rows:
            p=r.get('path') or r.get('relative_path'); b=z.read(p)
            n=r.get('bytes',r.get('size')); h=r.get('sha256')
            assert n==len(b) and h==sha(b), (name,p)
            checks.append({'path':p,'bytes':len(b),'sha256':h})
        assert {x['path'] for x in checks}==set(names)-{'manifest.json'}
        destination.mkdir(parents=True,exist_ok=True)
        z.extractall(destination)
        for n in names: assert (destination/n).read_bytes()==z.read(n)
        out.append({'layer':name,'archive_bytes':len(raw),'sha256':expected,'files':len(names),'manifest_payloads':len(rows),'all_payloads_match':True,'payloads':checks})
result={'kind':'web_sandbox_read_only_identity','python':sys.version,'platform':platform.platform(),'layers':out,
        'project_execution':False,'unavailable_assets_not_read':['formal texts','pair labels','checkpoints','Memory binaries','credentials','owners','private_custody']}
p=ROOT/'audit/outputs/identity.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{k:v for k,v in x.items() if k!='payloads'} for x in out],ensure_ascii=False,indent=2))
