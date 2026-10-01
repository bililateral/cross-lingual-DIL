#!/usr/bin/env python3
"""Cross-check existing metadata, original failure/correction, source bytes and report.
Only small submitted records are used. No missing inputs or weights are accessed.
"""
from __future__ import annotations
import argparse,ast,datetime,difflib,hashlib,io,json,math,re,shutil,zipfile
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
a=p.parse_args();r=a.root.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
b=r/'reports/seller_alias_continual/20260928';run=b/'test_execution/20260928_182700';job=run/'execution/job';old=b/'test_execution/20260928_165355';checks=[]
sha=lambda x:hashlib.sha256(x).hexdigest()
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
def require(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def verify(rec,base=r):
 pp=base/rec['path'];bb=pp.read_bytes()
 require(len(bb)==rec['bytes'] and sha(bb)==rec['sha256'],'record '+str(pp.relative_to(r)))
 return pp

done=load(job/'completion.json');prep=load(job/'preparation.json');policy=load(r/'schema/step28_alias_test_policy.json');snap=done['source_files']
embedded=b/'test_implementation/review/test_review.zip'
require(embedded.stat().st_size==601670 and sha(embedded.read_bytes())=='0e315b5d4d7fca4bdff71c3a4f58bd0f4236da5c8aedad5fde846e615fdb5ed7','original implementation zip identity')
with zipfile.ZipFile(embedded) as z:
 require(z.testzip() is None,'embedded zip integrity')
 inv=load(b/'test_implementation/review/source_inventory.json')
 # This historical inventory describes only the historical ZIP, not the current submission.
 for rec in inv['files']:
  bb=z.read(rec['path']);require(len(bb)==rec['bytes'] and sha(bb)==rec['sha256'],'historical source '+rec['path'])
 for rec in snap:
  require((r/rec['path']).read_bytes()==z.read(rec['path']),'all34 old/current exact '+rec['path'])
require(len(snap)==34 and len(policy['sources']['inherited_scientific_sources'])==26,'34 scientific and26 inherited')
for rec in policy['sources']['inherited_scientific_sources']:verify(rec)

curr_auth=load(run/'authorization.json');prev_auth=load(old/'authorization.json')
changes={key:{'old':prev_auth.get(key),'new':curr_auth.get(key)} for key in set(curr_auth)|set(prev_auth) if curr_auth.get(key)!=prev_auth.get(key)}
require(set(changes)=={'created_at','job','resumption'},'resumption only time/path/new explicit receipt')
for auth in (curr_auth,prev_auth):
 require(auth['source_files']==snap and auth['label_parses']=={'heldout':1,'train':0,'development':0,'owners':0},'authorization frozen source and allowed parse extent')
 for key in ('cpu_evidence','review_disposition','user_authorization'):verify(auth[key])
require(curr_auth['job']==str(job.relative_to(r)),'current authorized run')
verify(prep['authorization']);resume=load(verify(curr_auth['resumption']))
require(resume['automatic_retry'] is False and resume['sources_changed']==0 and resume['heldout_attempts_before_resumption']==0,'explicit no automatic recovery')
require(resume['user_answer']=='允许恢复原定test（推荐）','actual user recovery statement')
repair=load(b/'test_execution/monitoring/repair_verification.json');require(repair['exit_code']==0 and not repair['stderr'],'original repair command exit')
repair_stdout=json.loads(repair['stdout']);require(repair_stdout['source_count']==34 and repair_stdout['sources']==snap,'fresh repair scientific snapshot')
require(repair_stdout['heldout_access_exists'] is False and repair_stdout['failed_output_files']==['failure.json'],'failed directory contains no label or score output')
inputs=[policy['data']['manifest'],policy['data']['validation'],*policy['data']['inputs'].values()]
require(repair_stdout['inputs']==[{k:x[k] for k in ('path','bytes','sha256')} for x in inputs],
        'repair file identities match frozen policy (descriptive identity_source is not a file receipt)')
sync=load(b/'test_execution/monitoring/input_repair_sync.json')
require(sync['count']==2 and sync['bytes']==9422271,'exact two input restoration receipt')
expected=[policy['data']['inputs'][k] for k in ('heldout/items.jsonl','heldout/supervision/pairs.csv')]
require(sync['files']==[{k:x[k] for k in ('path','bytes','sha256')} for x in expected],
        'two repaired byte identities; no actual raw input read here')
manifest=load(verify(policy['data']['manifest']))
for key,rec in policy['data']['inputs'].items():require({k:rec[k] for k in ('bytes','sha256')}==manifest['files'][key],'original generation identity '+key)

failure=load(old/'execution/job/failure.json');stderr=(old/'execution/stderr.log').read_text()
require(failure['type']=='FileNotFoundError' and failure['heldout_parse_attempts']==0,'original failure before label attempt')
require('path.stat().st_size' in stderr and 'heldout/items.jsonl' in stderr and 'groups, metadata = public_inputs(p)' in stderr,'original stat-before-open stack')
require((old/'execution/exit_status.txt').read_text().strip()=='1','failed original exit1')
require(sorted(p.name for p in (old/'execution/job').iterdir())==['failure.json'],'no first-run scores or heldout access in submitted job')
resource_old=(old/'execution/resource_usage.log').read_text();require('0:01.48' in resource_old,'initial1.48seconds')

# Cross-check completed archive receipt against its canonical digest and config,
# not against absent pretrained/model files.
arc=prep['pretrained_archive'];files=arc['files']
require(arc['file_count']==len(files)==12 and sum(x['size_bytes'] for x in files)==arc['total_size_bytes'],'12 archive metadata entries and aggregate size')
canonical=json.dumps(files,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
require(sha(canonical)==arc['content_sha256'],'archive canonical receipt digest')
for key in ('file_count','total_size_bytes','content_sha256'):require(arc[key]==policy['inference']['pretrained_archive'][key],'archive vs policy '+key)
configs={}
for role,model in policy['models'].items():
 mm=load(verify(model['source_manifest']));conf=mm['points']['6']['metadata']['reference_config'];configs[role]=conf
 require(conf['data_manifest_sha256']==policy['data']['manifest']['sha256'],'checkpoint generation binding '+role)
 require(conf['models']['split_rank']==policy['inference']['pretrained_archive'],'checkpoint archive binding '+role)
require(configs['A']==configs['C'],'common original representation/tokenizer/reference configuration')
rm=load(verify(policy['sources']['ranking_manifest']));rmroot=(r/policy['sources']['ranking_manifest']['path']).parent
partition=load(verify(rm['partition'],rmroot));oldids={x['group_uid'] for role in ('fit','calibration','development') for x in partition[role]}
ids={x['group_uid'] for x in prep['group_metadata']};require(not oldids.intersection(ids),'saved heldout IDs disjoint from old fit/cal/valid')
require(len(prep['group_metadata'])==120 and prep['group_metadata']==sorted(prep['group_metadata'],key=lambda x:(x['domain'],int(x['group_index']))),'exact public group ordering')
for d in 'ABC':
 rows=[x for x in prep['group_metadata'] if x['domain']==d]
 require([int(x['group_index']) for x in rows]==list(range(40)) and all(x['split']=='heldout' and int(x['accounts'])==28 for x in rows),'40 complete heldout groups '+d)

# Each returned-file inventory is checked against current small artifacts only.
synccheck={}
for rel in ('test_execution/monitoring/return_sync.json','test_result/return_sync.json','test_result/analysis_return_sync.json'):
 rr=load(b/rel);require(len(rr['files'])==rr['count'] and sum(x['bytes'] for x in rr['files'])==rr['bytes'],'sync aggregate '+rel)
 present, absent = [], []
 for rec in rr['files']:
  if (r / rec['path']).exists():
   verify(rec); present.append(rec)
  else:
   # Historical transfer inventories are not the current submission manifest.
   # This optional initial-observation JSON is not asserted provided by source_inventory.
   require(rel == 'test_execution/monitoring/return_sync.json' and
           rec['path'].endswith('/20260928_165355/initial_observation.json') and rec['bytes'] == 734,
           'only explicitly disclosed historical observation is unavailable')
   current_names={x['path'] for x in load(r/'source_inventory.json')['files']}
   require(rec['path'] not in current_names, 'historical missing file not falsely declared in current inventory')
   absent.append(rec)
 synccheck[rel]={'recorded_count':rr['count'],'recorded_bytes':rr['bytes'],
                'verified_present_count':len(present),'verified_present_bytes':sum(x['bytes'] for x in present),
                'not_provided_not_verified':absent}

started=datetime.datetime.fromisoformat((run/'execution/started.txt').read_text().strip());finished=datetime.datetime.fromisoformat((run/'execution/finished.txt').read_text().strip())
require((finished-started).total_seconds()==125,'formal start/end125seconds')
require(datetime.datetime.fromisoformat(resume['recorded_at'])<started,'resumption before actual start')
require(datetime.datetime.fromisoformat(curr_auth['created_at'])<started,'source authorization before start')
access=load(job/'heldout_access.json');at=datetime.datetime.fromisoformat(access['time_utc']);require(started<at<finished,'single attempt timestamp within run')
stdout=(run/'execution/stdout.log').read_text();events=[json.loads(x) for x in stdout.splitlines() if x.strip()]
require([x.get('model') for x in events[:2]]==['A','C'] and all(x.get('groups')==120 for x in events[:2]),'both full scoring progress events')
require(events[-1]['acceptance']==done['acceptance'],'stdout final verdict')
require((job/'blind_progress.json').read_bytes()==(job/'blind.json').read_bytes(),'final progress equals complete blind record')
resource=(run/'execution/resource_usage.log').read_text();require('2:05.20' in resource and 'Exit status: 0' in resource,'formal GNU125.20s exit0')
require(str(done['rss_max_kib']) in resource,'RSS matches completion')
require(done['seconds']<3600 and done['peak_observed_output_bytes']<4*1024**3,'actual recorded budget satisfied')
job_size=sum(p.stat().st_size for p in job.rglob('*') if p.is_file());require(job_size<4*1024**3,'final whole job under4GiB')

# AST shows the timing includes hashing following the forward; it excludes initial
# load/restore and later GC. This is a reporting clarification, not a rerun request.
code=(r/'scripts/step28_alias_test_run.py').read_text();tree=ast.parse(code)
fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='inference_one')
ass=next(x for x in fn.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='actual' for t in x.targets))
require(isinstance(ass.value,ast.Dict),'inference receipt dict')
keys=[x.value for x in ass.value.keys]
require(keys.index('model_file')<keys.index('score_seconds') and keys.index('parameters_sha256')<keys.index('score_seconds'),'time is sampled after postscore receipts')
timing_finding={'classification':'REPRODUCIBILITY_DEFECT','scope':'minor reporting label only, nonblocking for scientific validity',
  'evidence':'step28_alias_test_run.py:344-347 evaluates data.record (full weight hash) and state_digest before score_seconds; population_data.py:37-47 hashes full file.',
  'actual_interval':'From immediately before base.score through the subsequent full weight-file receipt and parameter-state digest; excludes initial load/restore and subsequent GC.',
  'minimal_revision':'Relabel the two recorded53s intervals as scoring plus post-score identity checks, not pure forward latency. Preserve frozen code, raw log and values; do not rerun.',
  'source_snippet':ast.get_source_segment(code,fn)}

# Parse report tables against independently computed numerical output. Rounded
# display tolerance is half one last shown decimal, not acceptance tolerance.
ref=load(a.reference);text=(r/'docs/SELLER_ALIAS_TEST_RESULT.zh.md').read_text();tables=[];block=[]
for line in text.splitlines()+['']:
 if line.startswith('|'):block.append([s.strip() for s in line.strip().strip('|').split('|')])
 elif block:
  if len(block)>2:tables.append(block)
  block=[]
report_numeric=0;roundchecks=[]
def rendered(s,value,label):
 global report_numeric
 ss=s.replace('−','-').replace('＋','+').replace(',','').strip();x=float(ss)
 decimals=len(ss.split('.')[-1]) if '.' in ss else 0
 tol=.5*10**(-decimals)+2e-14
 require(abs(x-float(value))<=tol,label+' display round');report_numeric+=1
 roundchecks.append({'label':label,'displayed':s,'expected':float(value),'display_tolerance':tol})
mt=next(t for t in tables if t[0][0]=='指标')
for row in mt[2:]:
 metric=row[0];rr=ref['comparisons']['C_cal_minus_A_cal'][metric]
 for s,v,key in zip(row[1:4],[ref['means']['A_cal'][metric],ref['means']['C_cal'][metric],rr['mean']],['Acal','Ccal','diff']):rendered(s,v,metric+'/'+key)
 lohi=re.findall(r'[-+]?\d+(?:\.\d+)?',row[4])
 for s,v in zip(lohi,rr['conditional_95pct_interval']):rendered(s,v,metric+'/CI')
ct=next(t for t in tables if t[0][0]=='条件')
for row in ct[2:]:
 key=row[0].split()[0];rendered(row[1],ref['decision']['observed'][key]['value'],key)
 require(row[2]==('通过' if ref['decision']['checks'][key] else '未通过'),key+' displayed decision')
dt=next(t for t in tables if t[0][0]=='域')
for row in dt[2:]:
 domain=row[0];want=[ref['by_domain']['A_cal'][domain]['map'],ref['by_domain']['C_cal'][domain]['map']]
 want +=[ref['comparisons']['C_cal_minus_A_cal'][m]['by_domain'][domain] for m in ('map','recall_at_5','brier','log_loss')]
 for k,(s,v) in enumerate(zip(row[1:],want)):rendered(s,v,domain+'/table/'+str(k))
ft=next(t for t in tables if t[0][0]=='系统')
for row,role in zip(ft[2:],('A_cal','C_cal')):
 stats=ref['fixed'][role]['pooled'];keys=('tp','fp','fn','tn','precision','recall','f1','fpr')
 for s,k in zip(row[1:],keys):rendered(s,stats[k],role+'/'+k)
require('下界只是略高于0' in text and '区间包含负值' in text and '事后' in text and 'passed=false' in text,'critical statistical and postdecision disclosures present')
# Byte-for-byte replay CSVs; JSON includes differing environment/time fields.
replay=r/'reports/reviewer_test_result_replay';orig=b/'test_result/analysis'
csv_exact={}
for name in ('mean_metrics.csv','paired_metrics.csv'):
 csv_exact[name]=(replay/name).read_bytes()==(orig/name).read_bytes();require(csv_exact[name],'replay CSV exact '+name)

user=load(b/'test_result/user_acceptance.json');require(datetime.datetime.fromisoformat(user['time'])>finished,'user accepts after disclosed result')
require(user['exact_user_statement']=='可以，这个test结果可以了，有提升就行' and user['original_contract_outcome']['passed'] is False,'explicit separate user decision retained')
result={'status':'PASS_METADATA_AND_REPORT_WITH_TIMING_LABEL_CLARIFICATION','checks':len(checks),'verified_frozen_sources':34,'inherited_sources':26,
 'embedded_original_zip_members':117,'history_scope':'Original117-member implementation input verified; selected old390-member external evidence is not asserted complete.',
 'authorization_changed_fields':changes,'repaired_raw_input_bytes_receipts_only':sync,'returned_artifacts':synccheck,
 'formal_environment':prep['environment'],'formal_start':started.isoformat(),'formal_finish':finished.isoformat(),'gnu_seconds':125.20,
 'completion_budget_seconds':done['seconds'],'final_job_bytes':job_size,'sampled_peak_output_bytes':done['peak_observed_output_bytes'],
 'cuda_peak_allocated_bytes':done['cuda_peak_allocated_bytes'],'cuda_peak_reserved_bytes':done['cuda_peak_reserved_bytes'],'max_rss_kib':done['rss_max_kib'],
 'timing_label_finding':timing_finding,'human_tables_checked':4,'human_numeric_cells_checked':report_numeric,
 'report_rounding_checks':roundchecks,'replay_csv_byte_equivalence':csv_exact,
 'raw_labels_text_weights_accessed':False,'external_remote_access':False,'all_checks':checks}
(out/'metadata_and_report_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
# Retain submitted failure and correction receipts as exact copies, clearly not
# results of current execution. No copying of repaired raw inputs or model files.
history=out.parent.parent/'history/project_execution';history.mkdir(parents=True,exist_ok=True)
paths=list((old/'execution').rglob('*'))+[old/'authorization.json',run/'authorization.json',run/'resumption.json',
 b/'test_execution/monitoring/failure_analysis.json',b/'test_execution/monitoring/input_repair_sync.json',b/'test_execution/monitoring/repair_verification.json',
 b/'test_execution/monitoring/recovery_plan.json',b/'test_result/user_acceptance.json']
paths +=[run/'execution'/x for x in ('started.txt','finished.txt','stdout.log','stderr.log','resource_usage.log','exit_status.txt')]
for pp in paths:
 if pp.is_file():
  dest=history/pp.relative_to(b);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(pp.read_bytes())
print(json.dumps({k:result[k] for k in ('status','checks','verified_frozen_sources','inherited_sources','formal_start','formal_finish','gnu_seconds','final_job_bytes','human_numeric_cells_checked','replay_csv_byte_equivalence')},ensure_ascii=False,indent=2))
print('Timing clarification:',timing_finding['actual_interval'])
