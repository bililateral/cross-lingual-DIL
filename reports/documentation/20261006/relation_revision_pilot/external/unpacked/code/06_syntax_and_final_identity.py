"""Read-only final syntax and source-byte recheck, not a formal launch."""
import ast,subprocess,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];C=R/'sources/current'
checked=[]
for path in [*sorted((C/'scripts').glob('*.py')),*sorted((C/'tests').glob('*.py'))]:
 ast.parse(path.read_text(encoding='utf-8'),filename=str(path));checked.append(path.relative_to(C).as_posix())
shell=[]
for path in sorted((C/'scripts').glob('*.sh')):
 result=subprocess.run(['bash','-n',str(path)],capture_output=True,text=True)
 shell.append({'path':path.relative_to(C).as_posix(),'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
 assert result.returncode==0
manifest=json.loads((C/'manifest.json').read_text())
for row in manifest:
 b=(C/row['path']).read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
result={'all_current_payloads_unchanged_after_tests':True,'payload_count':len(manifest),'python_ast_files':checked,'shell_syntax':shell,'formal_shell_executed':False}
(R/'outputs/06_syntax_and_final_identity.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
