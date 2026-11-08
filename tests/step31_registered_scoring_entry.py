"""Exercise the actual six-run blind-score/label-release entry with synthetic endpoints.

Completion metadata is a deliberately synthetic fixture, NOT evidence of 8640
training updates. Actual changed training execution is covered separately.
"""
import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts'), str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from step31_registered_increment import Encoder, tokenizer
from step29_vendorlink_data import Ad, View
from step29_vendorlink_methods import Model, Settings
from step30_evolution_io import read, write, record, pinned
from step31_evolution_registered_policy import DOMAINS, tasks
from step31_evolution_registered_evaluate import score_one, analyze


def main(root):
    if sys.platform != 'linux':
        raise RuntimeError('Linux CPU only')
    root.mkdir(parents=True)
    started = time.monotonic()
    h = lambda text: hashlib.sha256(text.encode()).hexdigest()
    manifest = dict(files={})
    for d in DOMAINS:
        mappings, data = {}, {}
        for role in ('query', 'gallery'):
            entries = []
            for i in range(96):
                uid = h(f'{d}/{role}/{i}')
                ads = tuple(Ad(h(f'{d}/{role}/{i}/{j}'), f'{i} {j} title', f'{d} {i} {j} synthetic {role}') for j in range(2))
                entries.append(View(uid, ads))
                if role == 'query':
                    mappings[uid] = h(f'{d}/gallery/{i}')
            data[role] = sorted(entries, key=lambda v: v.uid)
            path = root/f'{d}_{role}.json'; write(path, [asdict(v) for v in data[role]])
            manifest['files'][f'{d}:{role}'] = record(path)
        path = root/f'{d}_qrels.json'; write(path, mappings); manifest['files'][f'{d}:qrels'] = record(path)
    write(root/'manifest.json', manifest)
    batch = dict(specs=tasks(), manifest=record(root/'manifest.json'), completed=[])
    binding = dict(fixture_only=True, pretrained=False)
    def factory(task):
        torch.manual_seed(task['seed'])
        return Model(Encoder(), tokenizer, Settings(**task['settings']))
    for task in tasks():
        job = root/'synthetic_endpoints'/task['id']; job.mkdir(parents=True)
        ident = dict(fixture_only=True, synthetic_completion_metadata=True, task=task['id'])
        endpoints = {}
        for stage in range(4):
            model = factory(task)
            with torch.no_grad():
                model.encoder.embedding.weight.add_(stage*.02)
                if task['system'] == 'H_SEQ':
                    model.head[-1].bias.add_(stage*.1)
            value = dict(identity=ident, task=task, model_binding=binding, stage=stage,
                         head=model.head.state_dict(), encoder=model.encoder.state_dict())
            path = job/f'stage{stage}.pt'; torch.save(value, path); endpoints[str(stage)] = record(path)
        completion = dict(status='TRAINING_COMPLETE_EVALUATION_CLOSED', task=task, identity=ident,
            endpoints=endpoints, updates=1440, synthetic_fixture_not_training=True,
            workload=dict(current_updates=1440, current_records=46080, history_groups=0, history_records=0,
                          teacher_groups=0, teacher_records=0, stage_model_saves=3))
        write(job/'completion.json', completion)
        batch['completed'].append(dict(task_id=task['id'], identity=ident, completion=record(job/'completion.json')))
    receipts = []
    for task in tasks():
        destination = root/'scoring'/task['id']
        result = score_one(destination, task, batch, manifest, factory, binding, lambda: None)
        frozen_small = pinned(result['cells']['1_2_2'])
        frozen_large = pinned(result['cells']['1_2_3'])
        columns = [frozen_large['gallery_ids'].index(g) for g in frozen_small['gallery_ids']]
        assert np.allclose(frozen_small['scores'], np.array(frozen_large['scores'])[:, columns], atol=1e-6)
        receipts.append(dict(task_id=task['id'], completion=record(destination/'blind_complete.json')))
    analysis = root/'analysis'; analysis.mkdir()
    try:
        analyze(analysis, batch, manifest, receipts[:-1], lambda: None)
        raise AssertionError('Five of six blind outputs must not release labels')
    except ValueError as exc:
        assert 'Complete scoring phase' in str(exc) and not (analysis/'access').exists()
    result = analyze(analysis, batch, manifest, receipts, lambda: None)
    assert result['matrices'] == 90 and result['logical_scores'] == 1824768
    assert (analysis/'access/complete_P_labels.json').exists() and not result['automatic_next_phase']
    assert len(result['rows']) == 6
    write(root/'receipt.json', dict(status='PASS', fixture_only=True, synthetic_training_completion_metadata=True,
        real_training_updates_claimed=0, gpu=False, pretrained_loaded=False, elapsed_seconds=time.monotonic()-started,
        checks=['actual_score_one_six_tasks_all_90_cells', 'same_frozen_encoder_prefix_scores',
                'incomplete_scoring_blocks_label_access', 'actual_analyze_single_complete_label_release_no_M']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output)
