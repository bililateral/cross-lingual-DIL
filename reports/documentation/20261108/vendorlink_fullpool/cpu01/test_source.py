"""One Linux CPU increment: handwritten data, tiny random BERT, real scoring/queue."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer

import step32_vendorlink_fullpool as f
from step29_vendorlink_data import Ad, View, Group
from step30_evolution_control import Runner
from step30_evolution_io import Ledger, read, record, write


def tiny_factory(inputs, settings):
    tokenizer = BertTokenizer(vocab_file=str(Path(inputs['model_path'])/'vocab.txt'), do_lower_case=True)
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=4,
                        num_attention_heads=2, intermediate_size=12, max_position_embeddings=32)
    torch.manual_seed(71)
    return f.Model(BertModel(config), tokenizer, settings)


def make_inputs(root):
    (root/'vocab.txt').write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
                                         'item', 'description']+[str(i) for i in range(20)])+'\n')
    inputs = dict(model_path=str(root), dev_sources={}, dev_qrels={}, old_runs=[])
    for domain in f.DOMAINS:
        groups, qrels = [], {}
        for group in range(2):
            items = []
            for identity in range(4):
                for v in range(2):
                    uid = f'{domain}-{group}-{identity}-{v}'
                    records = tuple(Ad(f'{uid}-{r}', f'item {identity}', f'description {v} {r}') for r in range(2))
                    items.append(View(uid, records, f'{domain}-{group}-{identity}'))
            g = Group(f'{domain}-{group}', domain, 'dev', tuple(items)).evaluation_order()
            groups.append(asdict(g.blind()))
            qrels[g.uid] = dict(view_ids=[v.uid for v in g.views], labels=g.labels())
        write(root/f'{domain}_dev.json', groups)
        write(root/f'{domain}_qrels.json', qrels)
        inputs['dev_sources'][domain] = record(root/f'{domain}_dev.json')
        inputs['dev_qrels'][domain] = record(root/f'{domain}_qrels.json')
    for task in f.tasks():
        run = root/'old'/task['id']; run.mkdir(parents=True)
        settings = f.Settings(method=task['method'])
        model = tiny_factory(inputs, settings)
        classes = ('fixture-a', 'fixture-b') if task['method'] == 'VendorLink_SEQ' else ()
        if classes:
            model.start_vendor_stage(classes, 0)
        identity = dict(spec=task, sources=[], epochs=40)
        endpoints = {}
        for stage in (1, 2, 3):
            # Synthetic checkpoint metadata, not an assertion of real training.
            path = run/f'stage{stage}.pt'
            torch.save(dict(identity=identity, settings=asdict(settings), stage=stage,
                            classes=classes, model=model.state_dict()), path)
            endpoints[str(stage)] = record(path)
        complete = dict(status='THREE_STAGES_TRAINED_NOT_EVALUATED', identity=identity, updates=10280,
            inference_endpoints=True, stage_endpoints=['stage1.pt', 'stage2.pt', 'stage3.pt'],
            endpoint_files=endpoints, access=dict(fit_domains=list(f.DOMAINS), dev=0, heldout=0, qrels=0, owners=0))
        write(run/'completion.json', complete)
        inputs['old_runs'].append(dict(id=task['id'], spec=task, sources=[], completion=record(run/'completion.json')))
    return inputs


class Increment(unittest.TestCase):
    def test_actual_six_scores_and_joint_analysis(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(f.COUNTS, {d: 16 for d in f.DOMAINS}), \
                patch.dict(f.GROUPS, {d: 2 for d in f.DOMAINS}):
            root = Path(tmp)
            inputs = make_inputs(root)
            receipts = []
            for task in f.tasks():
                target = root/'scoring'/task['id']; target.mkdir(parents=True)
                f.score_one(target, task, inputs, lambda: None, factory=tiny_factory)
                receipts.append(dict(task_id=task['id'], completion=record(target/'blind_complete.json')))
            blocked = root/'blocked'
            with self.assertRaises(ValueError):
                f.analyze(blocked, inputs, receipts[:-1])
            self.assertFalse((blocked/'access/dev_qrels_joint_opening.json').exists())
            broken = read(receipts[0]['completion']['path'])
            broken['cells'].pop('3_3')
            with self.assertRaises(ValueError):
                f.validate_scores(broken, f.tasks()[0])
            result = f.analyze(root/'analysis', inputs, receipts)
            self.assertEqual(result['cells'], 48)
            self.assertEqual(result['training_updates'], 0)
            self.assertFalse(result['automatic_next_phase'])
            self.assertTrue((root/'analysis/access/dev_qrels_joint_opening.json').exists())
            for row in result['runs']:
                np.testing.assert_allclose(row['roles']['F'], 0., atol=1e-8)
                np.testing.assert_allclose(row['roles']['G_B'], 0., atol=1e-8)
                self.assertEqual(set(row['cells']), {f'{t}_{d}' for t, d in f.CELLS})

    def test_self_ties_and_wrong_labels(self):
        ids = ['a', 'b', 'c', 'd']
        positive = dict(a='c', c='a', b='d', d='b')
        scores = np.ones((4, 4)); np.fill_diagonal(scores, 999.)
        actual = f.query_metrics(scores, ids, positive)
        expected_ranks = [2, 3, 1, 2]  # Independently enumerate sorted nonself candidates.
        np.testing.assert_allclose(actual[:, 0], [1/r for r in expected_ranks])
        np.testing.assert_array_equal(actual[:, 2], [0, 0, 1, 0])
        with self.assertRaises(ValueError):
            f.query_metrics(scores, ids, dict(a='a', b='d', c='a', d='b'))
        with tempfile.TemporaryDirectory() as tmp:
            inputs = make_inputs(Path(tmp))
            pools, groups = f.load_public(inputs, Path(tmp)/'load')
            qrels = read(inputs['dev_qrels']['Dream']['path'])
            first = next(iter(qrels.values()))
            first['view_ids'] = first['view_ids'][::-1]
            with self.assertRaises(ValueError):
                f.positive_map(groups['Dream'], qrels)

    def test_real_two_slot_failure_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = root/'child.py'
            entry.write_text('import argparse,json,time\nfrom pathlib import Path\np=argparse.ArgumentParser()\n'
                'p.add_argument("--gate");p.add_argument("--action");p.add_argument("--target");p.add_argument("--deadline")\n'
                'a=p.parse_args();t=Path(a.target)\n(t/"started.json").write_text(json.dumps({"time":time.time()}))\n'
                'time.sleep(0.6 if t.name=="one" else 1.0)\n'
                '(t/"finished.json").write_text(json.dumps({"time":time.time()}))\n'
                'raise SystemExit(9 if t.name=="one" else 0)\n')
            allowed = dict(experiment_root=str(root), workspace=str(root), other_gpu_commitments=[])
            ledger = Ledger(root, dict(gpu=0, auxiliary=0), 0, caps=f.CAPS)
            runner = Runner(allowed, root/'unused_gate.json', root, ledger, worker_entry=entry,
                            caps=f.CAPS, limits=f.LIMITS, wait_for_resources=False)
            jobs = []
            for name in ('one', 'two', 'three', 'four'):
                target = root/name; target.mkdir()
                jobs.append(dict(action='fixture', target=target, bucket='auxiliary', seconds=30, extra=[]))
            complete = []
            with self.assertRaisesRegex(RuntimeError, 'Incomplete phase'):
                runner.run_many(jobs, 2, lambda job: complete.append(job['target'].name))
            self.assertEqual(complete, ['two', 'three', 'four'])
            self.assertEqual(read(root/'dispatch.json')['retired'], [0])
            self.assertLess(read(root/'two/started.json')['time'], read(root/'one/finished.json')['time'])
            self.assertEqual(read(root/'three/process.json')['slot'], 1)
            self.assertEqual(read(root/'two/process.json')['reserved_bytes'], f.LIMITS['free_gpu_bytes'])
            admission = read(root/'two/admission.json')
            self.assertGreater(admission['ram_pending'], 0)
            self.assertLessEqual(admission['ram_pending'], 8*2**30)
            self.assertEqual(admission['save_reserve'], 2*16*2**20)
            self.assertGreater(ledger.used['auxiliary'], 30)


if __name__ == '__main__':
    if sys.platform != 'linux':
        raise SystemExit('Scientific tests run only on the existing Linux py310')
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
