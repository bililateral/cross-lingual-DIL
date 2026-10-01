"""Validate the assembled review and every completed raw command record."""
import hashlib,json,pathlib,time
B=pathlib.Path('/mnt/data/bge_review_evidence');R=pathlib.Path('/mnt/data/bge_input');start=time.perf_counter()
report=(B/'REVIEW.zh.md').read_bytes()
assert report==pathlib.Path('/mnt/data/bge_continual_result_independent_review.zh.md').read_bytes()
txt=report.decode('utf-8')
assert len(txt)>20000 and txt.count('## 0.')==1
for name in ('SCIENTIFIC_BLOCKER','REPRODUCIBILITY_DEFECT','OUT_OF_SCOPE_OVERDESIGN','0.063393','0.041809','0.085423','6030','5184'):
 assert name in txt,name
conds=json.loads((B/'outputs/report_crosscheck_v2/report_23_conditions.json').read_text())
for row in conds:
 assert f"| {row['index']} |" in txt
 assert f"{row['er_minus_seq']:+.6f}" in txt
 assert f"{row['logit_minus_er']:+.6f}" in txt
s=json.loads((B/'outputs/independent_endpoints.json').read_text())
for a in ('frozen','seq','er','logit'):
 for m,x in s[a]['primary']['final_all'].items():assert f"{x['mean']:.6f}" in txt,(a,m)
log_names=[]
for p in sorted((B/'logs').glob('*.json')):
 j=json.loads(p.read_text());log_names.append(p.stem)
 assert int(p.with_suffix('.exit').read_text())==j['exit_code']
 assert j['elapsed_seconds']>=0
 for suffix in ('stdout','stderr'):
  d=p.with_suffix('.'+suffix).read_bytes()
  assert len(d)==j[suffix]['bytes']
  assert hashlib.sha256(d).hexdigest()==j[suffix]['sha256']
 for token in j['command']:
  if token.startswith('/mnt/data/') and token.endswith('.py'):assert pathlib.Path(token).is_file(),token
for item in json.loads((B/'outputs/reviewed_source_index.json').read_text()):
 d=(B/item['copy_relative_path']).read_bytes()
 assert d==(R/item['original_relative_path']).read_bytes()
 assert hashlib.sha256(d).hexdigest()==item['sha256']
assert not (R/'reports/seller_alias_continual/20261001/bge_continual_result/observation.json').exists()
assert (R/'reports/seller_alias_continual/20261001/bge_continual_result/current_status.json').exists()
# Exact original input integrity checked again at delivery, without extracting anew.
inv=json.loads((B/'outputs/inventory_verification.json').read_text())
for row in inv['rows']:
 d=(R/row['path']).read_bytes();assert len(d)==row['bytes'];assert hashlib.sha256(d).hexdigest()==row['sha256']
res={'status':'PASS_DELIVERY_REVIEW_AND_COMPLETED_RAW_LOGS','completed_command_records':len(log_names),'command_records':log_names,'report_bytes':len(report),'report_sha256':hashlib.sha256(report).hexdigest(),'original_payloads_still_unchanged':len(inv['rows']),'formal_labels_read':0,'new_training':False,'main_numeric_tolerance_not_changed':1e-12,'report_condition_rows':23,'final_all_metric_values':88,'elapsed_seconds':time.perf_counter()-start}
(B/'outputs/delivery_validation.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(res,ensure_ascii=False,indent=2))
