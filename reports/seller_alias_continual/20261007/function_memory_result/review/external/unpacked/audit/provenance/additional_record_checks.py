#!/usr/bin/env python3
"""Additional read-only chronology/record checks; no project entry points."""
import argparse
import hashlib
import json
from pathlib import Path
import re

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
r=a.input/'reports/seller_alias_continual/20261007/function_memory_result'
d=a.input/'reports/documentation/20261006/function_memory'
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
sources=load(r/'source_inventory.json')
startup=load(r/'startup_inventory.json')
changed=[]
for record in startup:
    b=(r/record['path']).read_bytes()
    if len(b)!=record['bytes'] or hashlib.sha256(b).hexdigest()!=record['sha256']:
        changed.append(record['path'])
        # The initial wrapper contains only the started_at line, and is a
        # documented prefix of the final wrapper containing exit/end/wall.
        assert record['path']=='job.wrapper.txt'
        prefix=b[:record['bytes']]
        assert hashlib.sha256(prefix).hexdigest()==record['sha256']
assert changed==['job.wrapper.txt']

progress=load(r/'job/run/progress.json');manifest=load(r/'job/run/manifest.json')
assert progress['status']=='RUNNING'
assert {k:v for k,v in progress.items() if k!='status'}=={k:v for k,v in manifest.items() if k not in ['status','logical_updates','gradient_group_presentations']}
rows=[]
for line in (r/'job.console.txt').read_text().splitlines():
    if line.startswith('{'):
        rows.append(json.loads(line))
expected=[(order,stage,step) for order in ('ABC','BCA','CAB') for stage in (2,3) for step in range(24,289,24)]
assert [(x['order'],x['stage'],x['step']) for x in rows]==expected
assert all(x['elapsed_seconds']<y['elapsed_seconds'] for x,y in zip(rows,rows[1:]))
assert len(rows)==72

cpu_console=(d/'repair/cpu_20261006_231000/console.txt').read_text()
native_console=(d/'native_20261006_231500/console.txt').read_text()
for t,wall,rss in [(cpu_console,'0:07.36','851428'),(native_console,'0:27.58','2856256')]:
    assert 'Exit status: 0' in t and wall in t and ('Maximum resident set size (kbytes): '+rss) in t
failure=load(d/'repair/failure_record.json');retry=load(d/'repair/retry_authorization.json')
assert failure['cpu_failed_wall_seconds']==4.41
assert retry['previous_actual_seconds']==failure['cpu_cumulative_seconds']==10.72
assert round(10.72+7.36,2)==load(r/'authorization.json')['cpu_cumulative_seconds']==18.08
assert retry['automatic_retry'] is False and retry['seconds_limit']==60
syntax=(r/'verification/syntax_error.txt').read_text()
assert 'SyntaxError' in syntax and 'elapsed=0.02' in syntax
verification=load(r/'verification/result.json')
script=a.input/'scripts/step28_function_memory_result_verify.py'
assert verification['script_sha256']==hashlib.sha256(script.read_bytes()).hexdigest()
assert 'exit=0' in (r/'verification/console.txt').read_text()
out={'status':'PASS','startup_records':len(startup),'startup_only_growth':'job.wrapper.txt; original started_at prefix hashes exactly',
     'last_progress_note':'Last RUNNING snapshot matches final manifest excluding finalized status/count additions',
     'console_progress_entries':len(rows),'console_exact_schedule':True,'console_monotone_elapsed':True,
     'cpu_outer_wall_seconds':7.36,'cpu_outer_rss_kib':851428,'cpu_cumulative_seconds':18.08,
     'native_outer_wall_seconds':27.58,'native_outer_rss_kib':2856256,
     'analysis_script_syntax_error_seconds':.02,'analysis_script_final_sha256':verification['script_sha256'],
     'analysis_script_scope':'Saved-results analysis, separate from frozen 23 training sources and formal job',
     'historical_failure_note':'Original failed CPU logs removed by explicit user instruction; summary/retry/deletion retained'}
(a.out/'additional_records_result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
