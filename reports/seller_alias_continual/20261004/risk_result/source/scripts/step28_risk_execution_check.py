"""Linux py310 handwritten incremental checks plus observed native update probe."""
import argparse
import os
from pathlib import Path
import platform
import sys
import time
import unittest
from unittest import mock

import step28_risk_study as study
import step28_risk_replay_check as native_check


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if platform.system()!='Linux' or torch.cuda.is_available():
        raise RuntimeError('Use existing Linux py310 with CUDA disabled')
    args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(study.data.ROOT/'tests'))
    import test_step28_risk_replay as unit
    import test_step28_risk_execution as execution
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started=time.monotonic()
    sources=study.sources()
    def check():
        if time.monotonic()-started>2700: raise RuntimeError('CPU budget exceeded')
    with mock.patch.object(unit.method.base.public,'public_inputs',side_effect=AssertionError('No formal inputs')), \
         mock.patch.object(unit.method.base.public,'attach_labels',side_effect=AssertionError('No formal labels')):
        suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(unit),
                                 unittest.defaultTestLoader.loadTestsFromModule(execution)])
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        counts={'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                'failed':len(result.failures)+len(result.errors),'skipped':len(result.skipped)}
        study.data.write_json(args.out/'contracts.json',counts)
        if not result.wasSuccessful(): raise RuntimeError('Incremental tests failed')
        original=unit.method.update
        def observed(*args,**kw):
            kw['observe']=True
            return original(*args,**kw)
        with mock.patch.object(unit.method,'update',side_effect=observed):
            native=native_check.native(check)
        study.data.write_json(args.out/'native.json',native)
    if sources!=study.sources(): raise RuntimeError('Sources changed during checks')
    audit={'status':'PASS_RISK_REPLAY_HANDMADE_CPU','source_files':sources,'contracts':counts,
           'formal_inputs':False,'formal_labels':False,'native_risk_gradient_increment_verified':True,
           'native':{'risk':study.data.record(args.out/'native.json',args.out)},
           'seconds':time.monotonic()-started,'python':sys.version,'torch':torch.__version__,
           'cpu_affinity':sorted(os.sched_getaffinity(0))}
    study.data.write_json(args.out/'audit.json',audit)


if __name__=='__main__': main()
