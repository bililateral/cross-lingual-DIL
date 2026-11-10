"""One bounded real-BERT increment check, exclusively handwritten inputs."""
import time
STARTED = time.monotonic()
import argparse
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import sys

import torch

from step28_formal_queue import GpuMemory
from step29_vendorlink_methods import Memory, make_optimizer
from step35_vendorlink_account_control import configure_compute
from step35_vendorlink_account_io import Check, write, record
from step35_vendorlink_account_methods import update_account, set_account_rates, save_account_state, load_account_state
from step35_vendorlink_account_policy import AccountSettings, settings
from step35_vendorlink_account_run import make_model, groups_for


def handwritten(domain):
    return [dict(account_uid=f'{domain}:{k:04}', records=[dict(uid=f'{domain}:{k}:{r}',
        title=('blue item quality example ' * 80)+f' {k} {r}',
        description=('sample description delivery text ' * 80)+f' {k} {r}') for r in range(12)])
        for k in range(11)]


def main(output, binding):
    if sys.platform != 'linux' or Path(sys.executable).resolve() != Path('/home/yongpeng/miniconda3/envs/py310/bin/python').resolve():
        raise RuntimeError('Existing Linux py310 required')
    output.mkdir(parents=True, exist_ok=False)
    limits = Check(output, 1800)
    nvml = GpuMemory()
    peaks = dict(process_gpu_bytes=0)
    result = dict(status='FAIL', real_data_read=False, device='cuda:0', synthetic=True)
    def check():
        limits()
        peaks['process_gpu_bytes'] = max(peaks['process_gpu_bytes'], nvml.snapshot()['processes'].get(os.getpid(), 0))
    try:
        before = nvml.snapshot()
        if before['free'] < 5*2**30:
            raise RuntimeError('Insufficient currently free GPU memory for one H probe')
        configure_compute()
        torch.cuda.reset_peak_memory_stats()
        current = groups_for(handwritten('Agora'), 51, 'Agora', 0)[0]
        history = groups_for(handwritten('AlphaBay'), 51, 'AlphaBay', 0)[0]
        classes = [f'Agora:{k:04}' for k in range(875)]
        cfg = AccountSettings(**settings('H_CE', 'H_full'))
        task = dict(seed=51, method='H_full', settings=asdict(cfg))
        tracked = 'encoder.layer.0.attention.self.query.weight'
        model = make_model(binding, dict(task, settings=asdict(replace(cfg, gamma=0.))))
        model.start_account_stage(classes, 99)
        assert model.encoder.is_gradient_checkpointing
        original = model.encoder.state_dict()[tracked].detach().cpu().clone()
        tokens = model.tokenizer([r.title for v in current.views for r in v.records],
                                padding=True, truncation=True, max_length=256, return_tensors='pt')
        assert tuple(tokens['input_ids'].shape) == (44, 256)
        memory = Memory(12, cfg)
        memory.retain([history], 1, model, check)
        hist, teacher = memory.draw(7)
        optimizer = make_optimizer(model)
        set_account_rates(model, optimizer, 1, 20)
        no_ce = update_account(model, optimizer, current, hist, teacher, 500, check)
        baseline = model.encoder.state_dict()[tracked].detach().cpu().clone()
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        model = make_model(binding, task)
        model.start_account_stage(classes, 99)
        optimizer = make_optimizer(model)
        set_account_rates(model, optimizer, 1, 20)
        ce = update_account(model, optimizer, current, hist, teacher, 500, check)
        changed = model.encoder.state_dict()[tracked].detach().cpu()
        ce_effect = float((changed-baseline).abs().max())
        encoder_update = float((changed-original).abs().max())
        assert ce_effect > 0 and encoder_update > 0
        assert ce['current']['ce_records'] == 44 and ce['current']['record_gradient_CE_norm'] > 0
        assert 'ce' not in ce['history'] and ce['history']['distillation'] > 0
        assert ce['optimizer_updates'] == ce['auxiliary_optimizer_updates'] == 1
        checkpoint = output/'probe_latest.pt'
        identity = dict(probe='step35-handwritten-gpu01')
        save_account_state(checkpoint, model, optimizer, memory, {'stage': 3, 'next_update': 2}, identity)
        saved = record(checkpoint)
        set_account_rates(model, optimizer, 2, 20)
        second = update_account(model, optimizer, current, hist, teacher, 501, check)
        expected = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        model = make_model(binding, task)
        restored_memory = Memory(12, cfg)
        optimizer, cursor = load_account_state(checkpoint, model, restored_memory, identity)
        assert cursor == {'stage': 3, 'next_update': 2}
        assert model.auxiliary_classes == tuple(classes)
        assert all(int(v['step']) == 1 for v in optimizer.state.values())
        assert all(int(v['step']) == 1 for v in model.auxiliary_optimizer.state.values())
        restored_hist, restored_teacher = restored_memory.draw(7)
        assert restored_hist == hist and torch.equal(restored_teacher, teacher)
        set_account_rates(model, optimizer, 2, 20)
        resumed = update_account(model, optimizer, current, restored_hist, restored_teacher, 501, check)
        for n, actual in model.state_dict().items():
            torch.testing.assert_close(actual.detach().cpu(), expected[n], rtol=0, atol=0)
        assert second == resumed
        torch.cuda.synchronize()
        check()
        result.update(status='PASS', records=44, title_tokens=256, description_tokens=256,
            record_pairs=946, auxiliary_classes=875, representation_dimension=1536,
            gamma_zero=no_ce, gamma_one=ce, encoder_update_max_abs=encoder_update,
            ce_effect_max_abs=ce_effect, resumed_next_update_bit_exact=True,
            checkpoint=saved, checkpoint_retained=False, actual_update_count=4,
            birth_teacher_unchanged=True, original_and_auxiliary_optimizer_restored=True,
            gpu_before=before, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        checkpoint.unlink()  # This probe's own disposable weights only; saved identity remains.
    except BaseException as e:
        result['error'] = repr(e)
        raise
    finally:
        result.update(**peaks, peak_rss_bytes=limits.peak_rss, seconds=time.monotonic()-STARTED,
                      bucket='gpu', no_allocator_cap=True, no_process_gpu_cap=True)
        result['charged_seconds'] = math.ceil(result['seconds'])
        write(output/'receipt.json', result)
        nvml.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args()
    main(args.output, json.loads(args.binding.read_text())['model_binding'])
