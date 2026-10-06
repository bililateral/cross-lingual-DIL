"""Web-sandbox CPU replay of supplied handwritten tests. No project/Linux/BGE/data access."""
from pathlib import Path
import os,sys,json,time,unittest,platform
R=Path(__file__).resolve().parents[2]; I=R/'input'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
scratch=R/'audit/outputs/micro_temp';scratch.mkdir(exist_ok=True)
os.environ['FUNCTION_MEMORY_TEST_ROOT']=str(scratch)
if hasattr(os,'sched_getaffinity'):
 cores=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cores[0]})
sys.path[:0]=[str(I/'scripts'),str(I/'tests')]
start=time.monotonic()
import torch,numpy as np
import step28_function_memory_run as run
import test_step28_function_memory as core_tests
import test_step28_function_memory_run as runner_tests
torch.set_num_threads(1);torch.set_num_interop_threads(1)
src=run.sources(); expected=json.loads((I/'source_inventory.json').read_text());assert sorted(src,key=lambda r:r["path"])==sorted(expected,key=lambda r:r["path"]); assert src==json.loads((I/"cpu_20261006_220500/result/result.json").read_text())["source_files"]
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(core_tests.FunctionMemoryTests),unittest.defaultTestLoader.loadTestsFromTestCase(runner_tests.IntegrationTests)])
with (R/'audit/outputs/web_cpu_unittest.txt').open('w') as f:
 result=unittest.TextTestRunner(stream=f,verbosity=2,failfast=False).run(suite)
d={'kind':'WEB_CPU_REPLAY_SUPPLIED_HANDWRITTEN_TESTS','project_linux':False,'bge_loaded':False,'formal_data_access':False,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'affinity':sorted(os.sched_getaffinity(0)),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'elapsed_including_imports_seconds':time.monotonic()-start,'source_records':src}
(R/'audit/outputs/web_cpu_result.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in d.items() if k!='source_records'},ensure_ascii=False,indent=2));print((R/'audit/outputs/web_cpu_unittest.txt').read_text())
if not result.wasSuccessful():sys.exit(1)
