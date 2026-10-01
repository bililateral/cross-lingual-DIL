"""Package the design review and verify that the submitted sources stayed unchanged.
No project imports or research execution. Run after the report is complete.
"""
from pathlib import Path
import hashlib, json, platform, sys, zipfile, shutil
ROOT=Path('/mnt/data/continual_plan_audit')
PROJECT=ROOT/'project'
Z=Path('/mnt/data/continual_plan_review.zip')

def sha(b): return hashlib.sha256(b).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

with zipfile.ZipFile(Z) as f:
    rows=[]
    for n in f.namelist():
        b=f.read(n); p=PROJECT/n
        rows.append({'path':n,'bytes':len(b),'sha256':sha(b),'unchanged':p.is_file() and p.read_bytes()==b})
    extras=sorted(str(p.relative_to(PROJECT)) for p in PROJECT.rglob('*') if p.is_file() and str(p.relative_to(PROJECT)) not in f.namelist())
    result={'zip_bytes':Z.stat().st_size,'zip_sha256':sha(Z.read_bytes()),'member_count':len(rows),'crc_error':f.testzip(),'unchanged':all(r['unchanged'] for r in rows),'extra_project_files':extras,'members':rows}
    assert result['unchanged'] and not extras and result['crc_error'] is None
    write(ROOT/'outputs/final_source_verification.json',result)
    indexed=[]
    for n in f.namelist():
        if n=='source_inventory.json': continue
        text=f.read(n).decode('utf-8-sig')
        numbered=''.join(f'{i:04d}: {line}\n' for i,line in enumerate(text.splitlines(),1))
        dest=ROOT/'numbered_sources'/f'{n}.txt'; dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(numbered,encoding='utf-8')
        indexed.append({'original_path':n,'original_sha256':sha(f.read(n)),'line_count':len(text.splitlines()),'numbered_copy':str(dest.relative_to(ROOT))})
    write(ROOT/'outputs/source_line_index.json',indexed)
    (ROOT/'evidence/submitted_source_inventory.json').write_bytes(f.read('source_inventory.json'))
shutil.copyfile('/mnt/data/continual_plan_external_review.zh.md',ROOT/'review.zh.md')
write(ROOT/'evidence/run_environment.json',{
    'python':platform.python_version(),'platform':platform.platform(),'executable':sys.executable,
    'packages_installed_or_changed':False,'project_modules_imported':False,
    'arithmetic_only':True,'model_loaded':False,'gpu_used':False,'project_linux_connected':False,
    'formal_inputs_read':False,'new_training_or_fitting':False,
    'actual_identity':'GPT-6 Astra Pro (self identification); UI model_slug not independently inspected',
    'commands':[
      {'command':'python -B /mnt/data/continual_plan_audit/evidence/audit_derivations.py','cwd':'not semantically required; paths derive from __file__','stdout':'evidence/derivations.stdout.txt','stderr':'evidence/derivations.stderr.txt','exit_code':'evidence/derivations.exit_code.txt','wall_time':'not measured; no timing claim','reference_revision':False},
      {'command':'python -B /mnt/data/continual_plan_audit/evidence/finalize_evidence.py','stdout':'evidence/finalize.stdout.txt','stderr':'evidence/finalize.stderr.txt','exit_code':'evidence/finalize.exit_code.txt','purpose':'source final verification, numbered copies and environment recording'}
    ],
    'failures_and_revisions':[
      'Files ZIP content retrieval returned HTTP 400; original mounted ZIP was then used.',
      'Some primary-paper PDF requests failed, recorded in source_register.json; no unread text was claimed as read.',
      'Arithmetic reference first execution exited 0; no failed arithmetic run or numeric-reference revision occurred.'
    ]})
print(json.dumps({'unchanged':result['unchanged'],'members':len(rows),'extra_project_files':extras,'numbered_sources':len(indexed)},ensure_ascii=False,indent=2))
