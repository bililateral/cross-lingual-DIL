"""One bounded incremental check; reused BGE handwritten inputs, no formal loader."""
import argparse
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import unittest
from unittest import mock

import torch

import step28_group_meta_gpu_check as previous
import step28_group_meta_staged as staged

direct = staged.direct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('tiny', 'staged'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or args.output.exists():
        raise ValueError('Fresh Linux GPU0 evidence required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(28 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    sys.path.insert(0, str(direct.data.ROOT / 'tests'))
    os.environ['STAGED_TEST_DEVICE'] = 'cuda:0'
    start = time.monotonic()
    def check():
        if time.monotonic() - start > 3499:
            raise TimeoutError('Aggregate remaining budget is enforced by external wrapper')
    def progress(phase):
        torch.cuda.synchronize()
        print(direct.data.json_bytes({'phase': phase, 'seconds': time.monotonic()-start,
              'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}).decode(), flush=True)
    def update(*args):
        return staged.update(*args, progress=progress)
    result = {'mode': args.mode, 'formal_inputs': False, 'formal_labels': False, 'old_experiment_weights': False,
              'torch': torch.__version__, 'python': sys.version, 'cuda': torch.version.cuda,
              'device': torch.cuda.get_device_name(0), 'cpu_affinity': sorted(os.sched_getaffinity(0))}
    try:
        with mock.patch.object(direct.base.public, 'public_inputs', side_effect=AssertionError('No formal texts')), \
                mock.patch.object(direct.base.public, 'attach_labels', side_effect=AssertionError('No formal labels')), \
                mock.patch.object(direct.data, 'Archive', side_effect=AssertionError('No formal archive')):
            if args.mode == 'tiny':
                suite = unittest.defaultTestLoader.loadTestsFromNames(['test_step28_group_meta', 'test_step28_group_meta_staged'])
                run = unittest.TextTestRunner(verbosity=2).run(suite)
                result['tests'] = {'run': run.testsRun, 'failures': len(run.failures), 'errors': len(run.errors), 'skipped': len(run.skipped)}
                if not run.wasSuccessful():
                    raise AssertionError('Incremental gradient/update regression failed')
            else:
                result['evidence'] = previous.native(False, check, update_function=update)
        result['status'] = 'PASS'
    except Exception as error:
        result.update(status='FAIL', error_type=type(error).__name__, traceback=traceback.format_exc())
    result.update(seconds=time.monotonic()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  last_interval_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  last_interval_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    direct.data.write_json(args.output, result)
    if result['status'] != 'PASS':
        print(result['traceback'], file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
