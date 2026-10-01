import sys,os,json,platform,importlib.util,importlib.metadata as md
r={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'packages':{},'project_environment':False,'installed_or_changed_packages':False}
for n in ('numpy','torch','scipy','transformers','sentence-transformers'):
 try:r['packages'][n]=md.version(n)
 except md.PackageNotFoundError:r['packages'][n]=None
import numpy,torch,scipy
torch.set_num_threads(1);torch.set_num_interop_threads(1)
r['torch_cuda_available']=torch.cuda.is_available();r['torch_cuda_version']=torch.version.cuda;r['torch_threads']=torch.get_num_threads();r['torch_interop_threads']=torch.get_num_interop_threads();r['numpy_actual']=numpy.__version__;r['scipy_actual']=scipy.__version__;r['threads_at_measurement']=[s for s in open('/proc/self/status').read().splitlines() if s.startswith('Threads:')][0]
print(json.dumps(r,ensure_ascii=False,indent=2));open('/mnt/data/bge_continual_external_audit/outputs/environment.json','w').write(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
