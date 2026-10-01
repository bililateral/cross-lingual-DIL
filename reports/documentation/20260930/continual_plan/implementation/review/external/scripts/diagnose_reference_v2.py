import sys,importlib.util,copy,json
from pathlib import Path
import torch,numpy as np
from unittest import mock
R=Path('/mnt/data/bge_continual_external_audit');sys.argv=['independent_checks_v2.py','--out',str(R/'outputs/diagnosis_v2_import_fixture')]
spec=importlib.util.spec_from_file_location('v2',R/'scripts/independent_checks_v2.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
c=v.m.config(v.m.contract());current,old=v.handmade('new_B'),v.handmade('old_A');base=v.SmallScore().double();op=v.m.core.make_optimizer(base,c)
with mock.patch.object(v.m.base,'logits',side_effect=lambda md,g,*a:md(g)):v.m.update(base,op,old,None,None,c,'seq',1,1,510,511)
rows=[]
for seed,g in ((520,current),(521,old)):
 base.train();torch.manual_seed(seed);z=base(g);yf=torch.tensor(v.Y,dtype=torch.float32);yd=yf.double();raw=torch.nn.functional.binary_cross_entropy_with_logits(z,yf,reduction='none')
 exact=torch.logaddexp(torch.zeros_like(z),z)-yd*z
 sequential=((1-yd)*z).float().double()+torch.logaddexp(torch.zeros_like(z),-z)
 candidate=sequential.float()
 rows.append({'group':g.uid,'exact_cast_mean':float(exact.float().mean().detach()),'sequential_store_mean':float(candidate.mean().detach()),'actual_mean':float(raw.mean().detach()),'per_element_sequential_equal':bool(torch.equal(candidate,raw)),'max_sequential_error':float((candidate-raw).abs().max().detach()),'per_element_once_rounded_disagreements':int((exact.float()!=raw).sum()),'max_once_rounded_diff':float((exact.float()-raw).abs().max().detach()),'reference_v2_scalar_correction':v.mixed_target_rounding_correction(z)})
out={'status':'PASS' if all(r['per_element_sequential_equal'] for r in rows) else 'DIFFERENT','rows':rows,'formula':'cast32(cast32((1-y)*z) + softplus(-z)) then float32 mean; compares recorded value only, independently differentiated mathematical gradient retained','project_source_modified':False}
(R/'outputs/reference_v2_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));sys.exit(0 if out['status']=='PASS' else 1)
