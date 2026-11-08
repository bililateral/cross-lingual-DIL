"""CPU-only witnesses for the two reviewed entry defects; no real text/models."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import step31_evolution_registered_control as control
from step30_evolution_io import access, pinned, read, record, write


def main(root):
    if root.exists():
        raise ValueError('Fresh fixture directory required')
    root.mkdir(parents=True)
    assert not torch.cuda.is_initialized()
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(total_memory=32*2**30)), \
            patch.object(torch.cuda, 'set_per_process_memory_fraction') as allocator:
        control.gpu_admission()  # Real shared configure_gpu; only CUDA hardware calls are stubbed.
        allocator.assert_called_once_with(13/32, 0)
    gpu_flags = dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                     matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_tf32=torch.backends.cudnn.allow_tf32,
                     threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
    assert gpu_flags == dict(deterministic=True, matmul_tf32=False, cudnn_tf32=False,
                             threads=2, interop_threads=2)
    with patch.object(torch.cuda, 'is_available', return_value=False):
        try:
            control.gpu_admission()
        except ValueError:
            pass
        else:
            raise AssertionError('Missing CUDA must still be rejected')
    assert not torch.cuda.is_initialized()

    failures = []
    for mode in ('before_labels', 'after_labels'):
        experiment = root/mode
        experiment.mkdir()
        data = experiment/'dataset'; data.mkdir()
        write(data/'manifest.json', {'handwritten_fixture': True})
        qrels = experiment/'handwritten_qrels.json'
        write(qrels, {'toy-query': 'toy-gallery'})
        write(experiment/'preparation_budget.json', dict(
            used_seconds=dict(training=0, gpu=0, auxiliary=549), cpu_test_seconds=17))
        gate = dict(experiment_root=str(experiment), campaign=str(experiment/'campaign'), dataset=str(data))
        gate_path = experiment/'fixture_gate.json'; write(gate_path, gate)

        class FaultRunner:
            def __init__(self, *args, **kwargs):
                pass

            def run_many(self, *args, **kwargs):
                if mode == 'before_labels':
                    raise RuntimeError('injected before labels')

            def run(self, action, target, *args):
                assert action == 'analyze' and mode == 'after_labels'
                access(target, 'complete_P_labels', [record(qrels)])
                assert pinned(record(qrels)) == {'toy-query': 'toy-gallery'}
                raise TimeoutError('injected after handwritten qrels read')

        with patch.object(control, 'Runner', FaultRunner), \
                patch.object(control, 'dataset', return_value={'handwritten_fixture': True}):
            try:
                control.campaign(gate, gate_path)
            except (RuntimeError, TimeoutError) as exc:
                expected_error = repr(exc)
            else:
                raise AssertionError('Injected failure must propagate')
        failure = read(experiment/'campaign/failure.json')
        assert failure['error'] == expected_error
        assert 'effects_open' not in failure
        assert failure['automatic_retry'] is False and failure['automatic_next_phase'] is False
        attempted = (experiment/'campaign/analysis/access/complete_P_labels.json').exists()
        assert attempted == (mode == 'after_labels')
        failures.append(dict(case=mode, failure=failure, handwritten_label_access_recorded=attempted))
    write(root/'receipt.json', dict(status='PASS', gpu_flags=gpu_flags, allocator_fraction=13/32,
          cuda_initialized=torch.cuda.is_initialized(), failure_cases=failures,
          sources=[record(Path(control.__file__)), record(Path(__file__)),
                   record(Path(control.__file__).with_name('step30_evolution_control.py'))],
          scope='Real configuration and campaign exception path; CUDA calls and child jobs stubbed; no GPU, real labels or training'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
