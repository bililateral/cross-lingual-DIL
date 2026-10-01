from pathlib import Path
import sys,os,json
import numpy as np
E=Path(__file__).resolve().parent; SRC=E.parent/'submission'
sys.path.insert(0,str(SRC/'scripts'))
original_load=np.load; loads=[]
def guarded_load(path,*args,**kwargs):
 try: resolved=Path(path).resolve()
 except TypeError: resolved=None
 if resolved is not None and resolved.is_relative_to(SRC):
  raise AssertionError('Reviewer prohibition: do not parse submitted formal score or metric arrays')
 loads.append(str(path)); return original_load(path,*args,**kwargs)
np.load=guarded_load
import step28_alias_calibration_audit as audit
print('Reviewer Python/CPU',sys.version,sorted(os.sched_getaffinity(0)))
result=audit.run(E/'supplied_audit_artifacts')
result['reviewer_array_guard']={'submitted_numpy_arrays_loaded':0,'handmade_array_loads':len(loads)}
(E/'supplied_audit_guard_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'status':result['status'],'contracts':result['contracts'],'reviewer_array_guard':result['reviewer_array_guard']},ensure_ascii=False))
