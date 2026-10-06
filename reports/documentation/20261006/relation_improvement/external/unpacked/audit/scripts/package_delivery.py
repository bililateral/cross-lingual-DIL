"""Package this discussion's actual artifacts. Does not modify submitted sources."""
from __future__ import annotations
import argparse, hashlib, json, shutil, tempfile, zipfile
from pathlib import Path

def sha(b: bytes) -> str:return hashlib.sha256(b).hexdigest()
def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='function_memory_delivery_') as tmp:
        stage=Path(tmp)
        for name in ('original_input.zip','RELATION_FUNCTION_MEMORY.discussion.zh.md','README.zh.md'):
            shutil.copyfile(root/name,stage/name)
        shutil.copytree(root/'audit',stage/'audit',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        paths=sorted(p for p in stage.rglob('*') if p.is_file())
        rows=[{'path':str(p.relative_to(stage)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in paths]
        manifest={'schema':'discussion-evidence-v1','scope':'web-only scientific discussion; not project code or execution authorization','payload_count':len(rows),'files':rows,'self_hash_excluded':True}
        (stage/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(stage.rglob('*')):
                if p.is_file():z.write(p,str(p.relative_to(stage)))
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None
        m=json.loads(z.read('manifest.json'))
        assert len(z.namelist())==len(m['files'])+1
        assert set(z.namelist())=={x['path'] for x in m['files']}|{'manifest.json'}
        for item in m['files']:
            b=z.read(item['path']);assert len(b)==item['bytes'] and sha(b)==item['sha256']
    print(json.dumps({'output':str(args.output),'bytes':args.output.stat().st_size,'sha256':sha(args.output.read_bytes()),'files':len(m['files'])+1,'manifest_payloads':len(m['files']),'all_payloads_verified':True},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
