"""Supplied handwritten tests in this WEB container, not project Linux.
No formal execute/native calls; supplied fixtures replace BGE and formal inputs.
One K1 and two new integration test executions, using installed packages only.
"""
import os,sys,time,json,traceback,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]; source=R/'sources/current'
start=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',RELATION_STARTED_EPOCH=str(time.time()))
affinity=os.sched_getaffinity(0); os.sched_setaffinity(0,{min(affinity)})
sys.dont_write_bytecode=True
sys.path[:0]=[str(source/'tests'),str(source/'scripts')]
report={'origin':'independent_web_container','started_epoch':time.time(),'python':sys.version,'bge_loaded':False,'project_linux_invoked':False,'formal_data_read':False,'gpu_requested':False,'affinity':sorted(os.sched_getaffinity(0))}
try:
 import torch,numpy,psutil
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 report.update(torch=torch.__version__,numpy=numpy.__version__,psutil=psutil.__version__,cuda_available=torch.cuda.is_available())
 import test_step28_relation_revision as k1
 import test_step28_relation_revision_run as integration
 suite=unittest.TestSuite([
  k1.RevisionTests('test_serial_kernel_matches_joint_and_both_history_paths_are_live'),
  integration.IntegrationTests('test_actual_revision_kernel_full_dispatch_and_restoration'),
  integration.IntegrationTests('test_revision_gate_rejects_old_or_mismatched_evidence')])
 with (R/'outputs/02_web_cpu.unittest.txt').open('w') as f:
  result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
 report.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),success=result.wasSuccessful())
except BaseException:
 report.update(success=False,exception=traceback.format_exc());raise
finally:
 report['elapsed_seconds']=time.monotonic()-start
 (R/'outputs/02_web_cpu.result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False,indent=2))
if not report.get('success'): sys.exit(1)
