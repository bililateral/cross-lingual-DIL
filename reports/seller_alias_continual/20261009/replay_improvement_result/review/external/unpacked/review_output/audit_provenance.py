"""Check the supplied sync/gate/analysis receipts against the supplied bytes only."""
import json
import time
import audit_saved as a

start=time.monotonic()
sync=a.js(a.INPUT/'result/sync_manifest.json')
for rec in sync['files']:a.verified(a.INPUT/'result',rec)
man=a.js(a.INPUT/'result/job/run/manifest.json')
gate=a.js(a.INPUT/'result/qualification/formal_gate.json')
comp=a.js(a.INPUT/'result/job/completion.json')
a.expect(sync['source_files']==len(man['source_files'])==28 and man['source_files']==a.js(a.INPUT/'background/sources.json'),'sync_source_binding')
a.expect(sync['access']==comp['access'] and sync['completion_budget']==comp['budget'],'sync_completion_binding')
native={}
for mode in ('cpu','gpu'):
    rec={**gate[mode],'path':f'{mode}/result.json'}
    p=a.verified(a.INPUT/'background',rec);v=a.js(p)
    a.expect(v['source_files']==man['source_files'] and v['status']=='PASS_HANDWRITTEN_ONLY' and v['mode']==mode,'qualification_binding:'+mode)
    a.expect(v['formal_inputs'] is False and v['formal_labels'] is False,'qualification_handwritten_scope:'+mode)
    a.expect(v['cumulative_seconds']<=(300 if mode=='cpu' else 600),'qualification_budget:'+mode)
    if mode=='gpu':
        for name,b in v['native']['architectures'].items():
            a.expect(b['native_state_unchanged'] and b['next_model_adam_bitwise_equal'],'native_recorded_neutrality:'+name)
        a.expect(v['native']['projected_total_seconds']<=259200,'native_time_projection')
    native[mode]={k:v[k] for k in ('cumulative_seconds','elapsed_seconds','formal_inputs','formal_labels')}
analysis=a.js(a.INPUT/'result/analysis/output/execution.json')
arec={**analysis['script'],'path':'summarize_saved.py'}
a.verified(a.INPUT/'result/analysis',arec)
for rec in analysis['outputs']:
    rel=rec['path'].split('/analysis/output/',1)[1]
    a.verified(a.INPUT/'result/analysis/output',{**rec,'path':rel})
teacher=a.js(a.INPUT/'result/analysis/output/teacher_errors.json')
a.verified(a.INPUT/'result/analysis',{**teacher['script'],'path':'summarize_teacher.py'})
for rec in teacher['inputs']:
    a.verified(a.INPUT/'result/job/run/diagnostics',{**rec,'path':rec['path'].rsplit('/',1)[1]})
result={'status':'PASS','environment':'ChatGPT web review workspace; not project Linux',
        'sync_file_count':len(sync['files']),'sync_payload_bytes':sum(x['bytes'] for x in sync['files']),
        'excluded_payload_count':len(sync['excluded_payloads']),'excluded_payload_bytes':sum(x['bytes'] for x in sync['excluded_payloads']),
        'recorded_qualification':native,'analysis_original_output_count':len(analysis['outputs']),
        'teacher_input_count':len(teacher['inputs']),'current_job_failure_json_present':(a.INPUT/'result/job/failure.json').exists(),
        'elapsed_seconds':time.monotonic()-start,'check_counts':dict(a.CHECKS),'inputs':list(a.SEEN.values())}
a.save('provenance_audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='inputs'},ensure_ascii=False,indent=2))
