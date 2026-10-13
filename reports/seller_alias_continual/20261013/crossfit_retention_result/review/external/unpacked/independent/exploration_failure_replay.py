"""Exact transcription of the early read-only pretty-printer that failed.

The first interactive output was truncated. This copy is explicitly replayed
only to retain a complete stdout/stderr record, not labelled as the first run.
"""
import json,difflib
from pathlib import Path
b=Path('input')
x=(b/'analysis_attempt01/source.py').read_text().splitlines();y=(b/'step28_crossfit_result_audit.py').read_text().splitlines()
print('\n'.join(difflib.unified_diff(x,y,fromfile='analysis_attempt01/source.py',tofile='step28_crossfit_result_audit.py',lineterm='')))
for n in ['qualification/evidence/gpu/attempt01/result/result.json','job/run/startup.json','job/run/progress.json','receive_verification.json','analysis_manifest.json']:
 o=json.loads((b/n).read_text());print('\n###',n);print(json.dumps({k:v for k,v in o.items() if k not in ['source_files','sources']},ensure_ascii=False,indent=2))
