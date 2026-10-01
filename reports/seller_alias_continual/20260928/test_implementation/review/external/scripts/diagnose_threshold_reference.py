from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,'/mnt/data/test_pretest_audit_20260928/source/scripts')
import step28_alias_test_run as r
raw=np.array([1.],np.float32);t=float(np.nextafter(np.float64(1.),np.inf));a=.75;b=10.
mapped=a*raw.astype(np.float64)+b;mt=a*t+b
result={'raw_dtype':str(raw.dtype),'threshold':t,'python_float_reference':float(raw[0])>=t,'numpy_float32_array_comparison':bool((raw>=t)[0]),'numpy_float32_scalar_comparison':bool(raw[0]>=t),'explicit_float64_comparison':bool((raw.astype(np.float64)>=t)[0]),'mapped_value':float(mapped[0]),'mapped_threshold':mt,'mapped_comparison':bool((mapped>=mt)[0]),'numpy_result_type':str(np.result_type(raw,t)), 'production_counts':r.base.binary_counts(np.array([[1]],np.uint8),raw[None,:],t).tolist()}
print(json.dumps(result,indent=2));Path('/mnt/data/test_pretest_audit_20260928/evidence/outputs/threshold_reference_diagnosis.json').write_text(json.dumps(result,indent=2))
