"""Read-only identity audit. No project data, models, Memory, or network access."""
import hashlib, json, pathlib, zipfile
ROOT=pathlib.Path('/mnt/data/g_half_input')
OUT=pathlib.Path('/mnt/data/g_half_review')
P=pathlib.Path('/mnt/data/review_input(20261015-150112).zip')
def sha(b): return hashlib.sha256(b).hexdigest()
z=zipfile.ZipFile(P); ns=z.namelist(); manifest=json.loads(z.read('source_manifest.json'))
checks=[]
for x in manifest['files']:
 b=z.read(x['path']);checks.append({**x,'actual_bytes':len(b),'actual_sha256':sha(b),'ok':len(b)==x['bytes'] and sha(b)==x['sha256']})
r={'archive':{'path':str(P),'bytes':P.stat().st_size,'sha256':sha(P.read_bytes()),'members':len(ns),'unique_members':len(set(ns)),'crc_failure':z.testzip()},'payload_count':len(checks),'members_exact':set(ns)=={x['path'] for x in checks}|{'source_manifest.json'},'all_payloads_ok':all(x['ok'] for x in checks),'payloads':checks}
base='reports/documentation/20261015/g_half_implementation/qualification/'
for fn in ['cpu/attempt01/source_files.json','cpu/attempt02/source_files.json','cpu_receipt.json','baseline_identity.json','receive_verification.json','duplicate_cleanup.json']:
 v=json.loads((ROOT/base/fn).read_text());print(fn,'type',type(v).__name__,'keys',list(v)[:12] if isinstance(v,dict) else len(v))
(OUT/'evidence/input_identity.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='payloads'},indent=2))
