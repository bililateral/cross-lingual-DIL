"""Map the explicitly recorded Linux job root to the frozen archive root; compare receipts.
This corrects the v1 literal-path comparison, without reading any missing baseline payload.
"""
import json,hashlib
from pathlib import Path
R=Path('/mnt/data/g_half_input');O=Path('/mnt/data/g_half_review');Q=R/'reports/documentation/20261015/g_half_implementation/qualification'
def load(p):return json.loads(p.read_text())
def rec(p):
 b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
base=load(R/'reports/documentation/20261015/g_refinement/baseline_sources.json');identity=load(Q/'baseline_identity.json')
linux='/home/yongpeng/cross-lingual/reports/seller_alias_continual/20261015/balanced_pooling_execution/20261015_131600/workspace/reports/job/'
archive=base['root'].rstrip('/')+'/'
expected={x['path'][len(archive):]:x for x in base['files'] if x['path'].startswith(archive)}
rows=[]
for x in identity['files']:
 assert x['path'].startswith(linux)
 suffix=x['path'][len(linux):];e=expected[suffix]
 rows.append({'relative_to_job':suffix,'linux_path':x['path'],'archive_path':e['path'],'same_size_sha256':all(x[k]==e[k] for k in ('bytes','sha256'))})
received=load(Q/'receive_verification.json');rc=[{'path':x['path'],'ok':rec(R/x['path'])=={k:x[k] for k in ('bytes','sha256')}} for x in received['files']]
cleanup=load(Q/'duplicate_cleanup.json');dc=[]
for x in cleanup['removed']:
 after=x['path'].split('/sources/',1)[1]
 dc.append({'removed':x['path'],'current_equivalent':after,'same_size_sha256':rec(R/after)=={k:x[k] for k in ('bytes','sha256')}})
prior=load(O/'evidence/source_inheritance.json')
r={'baseline_mapping':{'linux_prefix':linux,'archive_prefix':archive,'is_payload_access':False,'all_43_same':len(rows)==43 and all(x['same_size_sha256'] for x in rows),'rows':rows},'received_27':rc,'cleanup_64':dc,'old_sources_summary':[{'evidence':x['evidence'],'present_compared':len(x['comparisons']),'same':sum(v['same'] for v in x['comparisons']),'different':[v['path'] for v in x['comparisons'] if not v['same']]} for x in prior['old_qualification_source_comparisons']]}
(O/'evidence/records_v2.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print('BASELINE metadata:',json.dumps({k:v for k,v in base.items() if k!='files'},ensure_ascii=False,indent=2))
print('RESULT:',json.dumps({'baseline_all43':r['baseline_mapping']['all_43_same'],'received27_match':all(x['ok'] for x in rc),'cleanup64_equivalents_match':all(x['same_size_sha256'] for x in dc),'old_sources_summary':r['old_sources_summary']},ensure_ascii=False,indent=2))
