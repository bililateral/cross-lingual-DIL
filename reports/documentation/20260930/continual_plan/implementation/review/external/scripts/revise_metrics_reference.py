from pathlib import Path
import json,hashlib,difflib
R=Path('/mnt/data/bge_continual_external_audit');p=R/'scripts/independent_metrics.py';s=p.read_text()
s=s.replace("expected,hits,counts=ref(z);got=m.metrics.group_metrics(y,z);diff=", "expected,hits,counts=ref(z);matrix,returned_counts=m.metrics.group_metrics(y[None,:],z[None,:]);got=matrix[0];assert returned_counts[0]==counts;diff=")
p2=R/'scripts/independent_metrics_v2.py';p2.write_text(s)
(R/'scripts/metrics_reference_v1_to_v2.diff').write_text(''.join(difflib.unified_diff(p.read_text().splitlines(True),s.splitlines(True),fromfile=p.name,tofile=p2.name)))
record={'change':'Fix reviewer API misuse: group_metrics takes 2D group rows and returns (matrix,counts); no reference definition, data or tolerance changes.','original_run':'logs/independent_metrics.json','original_stderr':'logs/independent_metrics.stderr','project_sources_modified':False,'SHA256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (p,p2)}}
(R/'outputs/metrics_reference_revision.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
