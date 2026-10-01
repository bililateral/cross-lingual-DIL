"""Reviewer-run supplied tests, handmade only; no formal or pretrained model loading."""
import os,sys,json,time,platform,unittest
from pathlib import Path
from unittest import mock
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
root=Path('/mnt/data/ranking_review_source')
sys.dont_write_bytecode=True
sys.path[:0]=[str(root/'scripts'),str(root/'tests')]
import numpy as np
import torch
import step28_alias_ranking as method
import step28_alias_ranking_run as runner
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
print(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'formal_access':0,'model_loads':0}))
all_results={}
with mock.patch.object(method.base.public,'public_inputs',side_effect=AssertionError('Reviewer forbids formal inputs')), mock.patch.object(method.base.public,'attach_labels',side_effect=AssertionError('Reviewer forbids formal labels')), mock.patch.object(method.data,'Archive',side_effect=AssertionError('Reviewer forbids formal Archive')), mock.patch.object(method.base,'load_model',side_effect=AssertionError('Reviewer forbids native model loading')):
 for name in ('test_step28_alias_ranking_contracts','test_step28_chinese_base_contracts'):
  t=time.monotonic(); suite=unittest.defaultTestLoader.loadTestsFromName(name); r=unittest.TextTestRunner(verbosity=2).run(suite)
  all_results[name]={'run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'seconds':time.monotonic()-t}
print(json.dumps(all_results,indent=2));Path('/mnt/data/ranking_reviewer_evidence/contract_results.json').write_text(json.dumps(all_results,indent=2))
sys.exit(int(any(v['failures'] or v['errors'] or v['skipped'] for v in all_results.values())))
