"""Final artifact QA; no science inputs changed and no new experiment."""
import ast,hashlib,json,pathlib,shutil,zipfile
import numpy as np
E=pathlib.Path('/mnt/data/er_review_evidence');R=pathlib.Path('/mnt/data/er_review_input');P=pathlib.Path('/mnt/data/er_review_portable_smoke')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for p in (E/'source').glob('*.py'):compile(p.read_text(),str(p),'exec')
for name in ('independent_endpoints.json','independent_comparisons.json','all_92_checks.csv','independent_draws.npy'):
 a=E/'results/independent_v1'/name;b=P/'results/independent_v1'/name;assert a.read_bytes()==b.read_bytes(),name
x=read(E/'results/hand_reference_v3/hand_reference.json');y=read(P/'results/hand_reference_v3/hand_reference.json');assert x==y
assert read(E/'results/supplemental_v2/summary.json')==read(P/'results/supplemental_v2/summary.json')
for name in ('independent','supplied_audit','hand_reference','supplemental'):
 meta=read(P/'logs'/name/'execution.json');assert meta['exit_code']==0;shutil.copytree(P/'logs'/name,E/'logs'/('portable_child_'+name))
zip_path=pathlib.Path('/mnt/data/er_weight_result_review.zip');copy_path=E/'original/er_weight_result_review.zip';assert zip_path.read_bytes()==copy_path.read_bytes()
with zipfile.ZipFile(zip_path) as z:
 assert len(z.namelist())==len(set(z.namelist()))==365 and z.testzip() is None
 for info in z.infolist():assert (R/info.filename).read_bytes()==z.read(info.filename)
report=(E/'REVIEW.zh.md').read_text();assert len(report)>25000 and report==pathlib.Path('/mnt/data/REVIEW.zh.md').read_text()
# Every Markdown table row has the same number of cells as its table header.
expected=None;tables=0
for line in report.splitlines():
 if line.startswith('|'):
  cells=line.count('|')
  if expected is None:expected=cells;tables+=1
  assert cells==expected,(cells,expected,line)
 else:expected=None
checks=dict(status='PASS_FINAL_QA',all_python_sources_compile=True,portable_replay_four_children_exit_zero=True,portable_endpoints_comparisons_decisions_draws_byte_identical=True,portable_hand_and_supplemental_equal=True,original_zip_sha256=sha(zip_path),original_zip_copy_exact=True,original_members_unchanged=365,report_markdown_tables=tables,report_utf8_bytes=len(report.encode()))
(E/'results/final_quality_check.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
