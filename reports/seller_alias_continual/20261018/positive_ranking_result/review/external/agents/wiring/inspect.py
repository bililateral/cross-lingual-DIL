#!/usr/bin/env python3
"""Read-only evidence viewer with hashes, read ranges and execution records."""
import sys,json,hashlib,time,datetime,resource,shlex
from pathlib import Path

ROOT=Path('/workspace/scratch/0f307e99d302/review_input')
OUT=Path('/workspace/scratch/0f307e99d302/audit_output/agents/wiring')
started=time.monotonic(); cpu_started=time.process_time()
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
scope_file=OUT/'read_scope.json'
scope=json.loads(scope_file.read_text()) if scope_file.exists() else []
chunks=[]; code=0
for arg in sys.argv[1:]:
    try:
        parts=arg.rsplit('@',1)
        path=ROOT/parts[0]
        data=path.read_bytes(); lines=data.decode('utf-8').splitlines()
        a,b=(1,len(lines)) if len(parts)==1 else tuple(map(int,parts[1].split(':')))
        if b==0: b=len(lines)
        assert 1<=a<=b<=len(lines),(a,b,len(lines))
        chunks.append('\n=== '+parts[0]+f' [{a}..{b}/{len(lines)}] ===\n'+ '\n'.join(f'{i}: {lines[i-1]}' for i in range(a,b+1)))
        scope.append({'path':parts[0],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'first_line':a,'last_line':b,'total_lines':len(lines),'read_utc':stamp,'method':'inspect.py UTF-8 numbered read'})
    except Exception as exc:
        chunks.append('READ_ERROR '+arg+' '+repr(exc)); code=1
output='\n'.join(chunks)+'\n'
print(output,end='')
scope_file.write_text(json.dumps(scope,ensure_ascii=False,indent=2)+'\n')
log_file=OUT/'commands.json'
log=json.loads(log_file.read_text()) if log_file.exists() else []
idx=len(log)+1
out_path=OUT/f'command_{idx:03d}.stdout.txt'
out_path.write_text(output)
log.append({'id':idx,'command':shlex.join(['python',str(OUT/'inspect.py'),*sys.argv[1:]]),'cwd':str(Path.cwd()),'start_utc':stamp,'exit_code':code,'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu_started,'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'stdout':out_path.name,'network_used':False,'project_server_connected':False,'model_or_training_imported':False})
log_file.write_text(json.dumps(log,ensure_ascii=False,indent=2)+'\n')
sys.exit(code)
