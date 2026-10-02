"""Reverse only supplied unified hunks in memory; verify historical source hashes."""
import hashlib,json,pathlib,re
import step28_er_weight as m
R=m.data.ROOT;E=pathlib.Path('/mnt/data/er_low_audit/evidence')
freeze=json.loads((R/'reports/documentation/20261002/er_low/freeze.json').read_text())
diff=(R/'reports/documentation/20261002/er_low/review/implementation.diff').read_bytes().splitlines(keepends=True)
blocks={};name=None
for line in diff:
 if line.startswith(b'diff --git '):
  name=line.split(b' b/',1)[1].rstrip().decode();blocks[name]=[]
 elif name is not None:blocks[name].append(line)
result=[]
for row in freeze['changed_from_previous_er_sources']:
 name=row['path'];lines=(R/name).read_bytes().splitlines(keepends=True);bl=blocks[name];hunks=[];i=0
 while i<len(bl):
  hit=re.match(rb'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',bl[i])
  if not hit:i+=1;continue
  oldstart=int(hit[1]);oldcount=int(hit[2] or b'1');newstart=int(hit[3]);newcount=int(hit[4] or b'1');before=[];after=[];i+=1
  while i<len(bl) and not bl[i].startswith(b'@@ '):
   line=bl[i]
   if line[:1] in (b' ',b'-'):before.append(line[1:])
   if line[:1] in (b' ',b'+'):after.append(line[1:])
   i+=1
  assert len(before)==oldcount and len(after)==newcount,(name,len(before),len(after),oldcount,newcount)
  assert lines[newstart-1:newstart-1+newcount]==after,name
  hunks.append((newstart,newcount,before))
 for start,count,before in reversed(hunks):lines[start-1:start-1+count]=before
 old=b''.join(lines);h=hashlib.sha256(old).hexdigest();assert h==row['previous_sha256'],(name,h,row['previous_sha256'])
 result.append(dict(path=name,hunks=len(hunks),reconstructed_old_bytes=len(old),reconstructed_old_sha256=h,previous_sha_matches=True,current_sha_matches=hashlib.sha256((R/name).read_bytes()).hexdigest()==row['current_sha256']))
p=m.contract('low');ref=m.baseline(p,R/p['baseline']['local_small_job'],R/p['weight_reference']['local_small_job'])
assert m.sources(p)==freeze['source_files']
assert m.prior.sources()==ref['manifest']['source_files']==ref['collected']['source_files']
wr=ref['weight_reference'];starts=[]
for order in m.ORDERS:
 rec=ref['manifest']['points'][order+'_shared'];shared=m.data.read_json(m.data.verify(ref['job']/'run'/rec['path'],rec))
 for arm in ['half','quarter']:
  start=wr['manifest']['restored_starts'][order+'_'+arm]
  starts.append(dict(path=order+'_'+arm,checkpoint_sha256=start['full_checkpoint']['sha256'],checkpoint_state_sha256=start['full_checkpoint']['state_sha256'],model_state_sha256=start['model_state_sha256'],adam_step=start['adam_step'],memory_source_sha256=start['memory_source']['sha256'],first_map=start['first_map'],first_scores_replayed_exactly_recorded=start['first_scores_replayed_exactly'],checkpoint_record_matches=start['full_checkpoint']==shared['full_checkpoint']))
report=dict(diff_reconstruction=result,project_files_modified=False,git_network_access=False,prior_git_commit_only_recorded_not_remotely_verified=freeze['previous_source_recovery_commit'],source_count=len(m.sources(p)),unchanged_original_pilot_source_count=len(m.prior.sources()),group_count=len(ref['collected']['group_ids']),same_group_domains_columns=all(ref['collected'][k]==wr['collected'][k] for k in ['group_ids','domains','metric_columns']),bound_starts=starts,low_expected_points=m.expected_points(p),old_metric_sets=45+36)
(E/'diff_and_reference_binding.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
