"""New pilot schedule, complete-result boundary, and paired qualification tests.

Fixtures are hand-labelled synthetic groups, never project supervision/models.
They verify orchestration and metrics, not actual CUDA training.
"""
from __future__ import annotations

from collections import Counter
import copy
import csv
import hashlib
import itertools
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import step28_continual_expression_run as run
from step28_continual_expression_run import data, metrics


def fixtures() -> tuple[dict, list]:
    groups, meta = {}, []
    for split, count in (('train', 60), ('development', 20)):
        groups[split] = []
        for d in 'ABC':
            for i in range(count):
                uid = f'{split}_{d}_{i:03d}'
                sellers = tuple(f'{uid}_s{k:02d}' for k in range(28))
                items = tuple(tuple((f'{s}_{j}', '商品说明', '固定的合成文本') for j in range(2)) for s in sellers)
                groups[split].append(data.Group(uid, sellers, items))
                meta.append({'domain': d, 'split': split, 'group_uid': uid,
                             'group_index': str(i), 'accounts': '28', 'items': '56'})
    return groups, meta


def hand_labels() -> list[int]:
    owners = list(itertools.chain.from_iterable([i] * (2 if i < 8 else 3) for i in range(12)))
    return [int(owners[a] == owners[b]) for a, b in itertools.combinations(range(28), 2)]


def synthetic_result(root: Path) -> tuple[Path, dict, dict, list, dict]:
    """Construct score/receipt fixtures to exercise the REAL complete gate."""
    groups, meta = fixtures()
    c = copy.deepcopy(run.contract())
    c['data_root'] = str(root / 'data')
    dataset = Path(c['data_root'])
    (dataset / 'development/supervision').mkdir(parents=True)
    label_path = dataset / 'development/supervision/pairs.csv'
    with label_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, ['group_uid', 'seller_uid_left', 'seller_uid_right', 'label'])
        writer.writeheader()
        for g in groups['development']:
            for (a, b), y in zip(itertools.combinations(g.sellers, 2), hand_labels()):
                writer.writerow(dict(group_uid=g.uid, seller_uid_left=a, seller_uid_right=b, label=y))
    dm = {'files': {'development/supervision/pairs.csv': data.record(label_path, dataset),
                    'train/supervision/pairs.csv': {'bytes': 0, 'sha256': 'hand_fixture_not_read'}}}
    data.write_json(dataset / 'manifest.json', dm)
    out = root / 'run'
    (out / 'scores').mkdir(parents=True)
    r = {'status': run.COMPLETE, 'config': c, 'source_files': run.sources(),
         'label_parses': {'train': 1, 'development': 0, 'heldout': 0, 'owners': 0},
         'physical_updates': 2160, 'points': {}, 'training': {}, 'models': {},
         'budget': {'elapsed_seconds': 1, 'peak_observed_bytes': 1}, 'pretrained': c['model'],
         'inputs': {'files': {'train/supervision/pairs.csv': dm['files']['train/supervision/pairs.csv']},
                    'metadata': meta, 'valid_group_ids': [g.uid for g in groups['development']]}}
    for i, name in enumerate(run.POINTS):
        s = np.random.default_rng(i).normal(size=(60, 378)).astype(np.float32)
        path = out / 'scores' / f'{name}_development.npy'
        np.save(path, s, allow_pickle=False)
        r['points'][name] = {'full_model_and_adam_reloaded': True,
                            'checkpoint': {'state_sha256': 'fixture_' + name},
                            'scores': {'development': data.record(path, out)}}
    r['starts'] = dict.fromkeys((*run.ORDERS, 'JOINT'), 'fixture_initial')
    r['starts'].update({o + '_continuation': 'fixture_' + o + '_shared' for o in run.ORDERS})
    for o, stage in [(o, s) for o in run.ORDERS for s in (1, 2, 3)] + [('JOINT', 1)]:
        name, _, prior = run.segment(o, stage)
        rows, seed = run.segment_schedule(groups['train'], meta, c, o, stage)
        ids = [g.uid for g in rows]
        r['training'][name] = {'updates': len(rows), 'group_ids': ids,
                              'order_sha256': hashlib.sha256(data.json_bytes(ids)).hexdigest(),
                              'dropout_stream': seed, 'pair_presentations': len(rows)*378,
                              'first_adam_step': prior+1, 'last_adam_step': prior+len(rows),
                              'first_update_modules': {k: {'finite_nonzero_gradient': True, 'parameters_changed': True} for k in ('encoder', 'head')},
                              'bce_by_epoch': [.2, .19, .18]}
    for name in c['retained_points']:
        r['models'][name] = {'path': f'models/{name}.pt', 'bytes': 1, 'sha256': 'fixture_hash',
                             'actual_loaded_model_equals_replayed_state': True}
    # No .pt file exists: this is a Windows receipt-boundary fixture, not GPU evidence.
    data.write_json(out / 'manifest.json', r)
    r = data.read_json(out / 'manifest.json')
    receipt = {'manifest_sha256': data.sha256(out / 'manifest.json'),
               'files': [{k: m[k] for k in ('path', 'bytes', 'sha256')} for m in r['models'].values()]}
    data.write_json(out / 'model_verification.json', receipt)
    return out, c, groups, meta, {}


class PilotContracts(unittest.TestCase):
    def test_confirmed_metric_constants_cannot_drift(self):
        original = run.contract()
        for key, wrong in (('threshold_logit', .5), ('bootstrap_seed', 20260911),
                           ('confidence_level', .90)):
            altered = copy.deepcopy(original)
            altered['evaluation'][key] = wrong
            with self.subTest(key=key), patch.object(data, 'read_json', return_value=altered):
                with self.assertRaisesRegex(ValueError, 'confirmed design'):
                    run.contract()
        self.assertEqual(run.contract(), original)

    def test_each_epoch_and_update_budget_independently(self):
        groups, meta = fixtures()
        c = run.contract()
        total = 0
        for o, stage in [(o, s) for o in run.ORDERS for s in (1, 2, 3)] + [('JOINT', 1)]:
            rows, seed = run.segment_schedule(groups['train'], meta, c, o, stage)
            domains = 'ABC' if o == 'JOINT' else o[stage-1]
            allowed = {r['group_uid'] for r in meta if r['split']=='train' and r['domain'] in domains}
            n = len(allowed)
            self.assertEqual(len(rows), n*3)
            expected, rng = [], random.Random(seed)
            for _ in range(3):
                epoch = sorted(allowed)
                rng.shuffle(epoch)
                expected.extend(epoch)
            self.assertEqual([g.uid for g in rows], expected)
            for start in range(0, len(rows), n):
                self.assertEqual(Counter(g.uid for g in rows[start:start+n]), Counter(allowed))
            total += len(rows)
        self.assertEqual(total, 2160)
        joint, _ = run.segment_schedule(groups['train'], meta, c, 'JOINT', 1)
        self.assertGreater(len({g.uid.split('_')[1] for g in joint[:20]}), 1)

    def test_same_order_gate_rejects_compensated_collapse(self):
        arrays = {name: np.full((60, 22), .2) for name in run.POINTS}
        arrays['initial'][:] = .1
        arrays['joint'][:] = .3
        for i, o in enumerate(run.ORDERS):
            arrays[o+'_shared'][:] = .3
            arrays[o+'_stage2'][:] = .3
            arrays[o+'_stage3'][:] = .3
            arrays[o+'_stage3'][i*20:(i+1)*20, 0] -= (.06, .06, -.001)[i]
            for stage in (2, 3):
                d = 'ABC'.index(o[stage-1])
                before = o+('_shared' if stage==2 else '_stage2')
                arrays[o+f'_stage{stage}'][d*20:(d+1)*20, 0] = arrays[before][d*20:(d+1)*20, 0]+(-.02, -.02, .2)[i]
        comp = {'first_domain_forgetting': {'conditional_95pct_interval': [.03, .05]}}
        r = run.qualification(arrays, comp, run.contract()['evaluation'])
        self.assertTrue(r['gates']['mean_forgetting'])
        self.assertTrue(r['gates']['mean_new_gain'])
        self.assertFalse(r['gates']['same_order_joint_condition'])
        self.assertFalse(r['all_numeric_gates_pass'])

    def test_same_order_gate_accepts_joint_learning_and_forgetting(self):
        arrays = {name: np.full((60, 22), .3) for name in run.POINTS}
        arrays['initial'][:] = .1
        for i, o in enumerate(run.ORDERS):
            arrays[o+'_stage3'][i*20:(i+1)*20, 0] = .25
            for stage in (2, 3):
                d = 'ABC'.index(o[stage-1])
                arrays[o+f'_stage{stage}'][d*20:(d+1)*20, 0] = .35
        comp = {'first_domain_forgetting': {'conditional_95pct_interval': [.04, .06]}}
        r = run.qualification(arrays, comp, run.contract()['evaluation'])
        self.assertTrue(r['all_numeric_gates_pass'])
        self.assertEqual(r['matched_orders'], list(run.ORDERS))

    def test_paired_sampling_matches_explicit_group_loops(self):
        delta = np.arange(60).reshape(3,20)/100
        draws = np.random.default_rng(17).integers(0,20,size=(31,3,20))
        expected = [sum(delta[d,k] for d in range(3) for k in draw[d])/60 for draw in draws]
        r = run.paired_summary(delta, draws)
        np.testing.assert_allclose(r['conditional_95pct_interval'], np.quantile(expected,[.025,.975]), atol=1e-15)

    def test_new_learning_keeps_same_group_pairing_across_orders(self):
        arrays = {name: np.zeros((60,22)) for name in run.POINTS}
        arrays['ABC_stage2'][20:40,0] = np.arange(20)
        arrays['CAB_stage3'][20:40,0] = -np.arange(20)
        # B appears as new domain twice; equal/opposite group effects cancel first.
        c = run.contrasts(arrays,0)
        np.testing.assert_array_equal(c['new_domain_gain'][1],np.zeros(20))

    def test_real_public_inputs_read_no_supervision_or_heldout(self):
        original = Path.open
        opened=[]
        def track(path,*args,**kwargs):
            name=path.as_posix()
            if '/supervision/' in name or name.endswith('/heldout/items.jsonl'):
                raise AssertionError('Forbidden label/test payload read')
            opened.append(name)
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',track):
            groups, meta, checked = run.public_inputs(run.contract())
        self.assertEqual([len(groups[s]) for s in ('train','development')],[180,60])
        self.assertTrue(all(g.labels is None for rows in groups.values() for g in rows))
        self.assertEqual(set(checked),{'groups.csv','train/items.jsonl','development/items.jsonl'})

    def test_complete_gate_rejects_missing_or_wrong_stage_evidence(self):
        with tempfile.TemporaryDirectory(prefix='expression_pilot_contract_') as tmp:
            out,c,groups,meta,checked=synthetic_result(Path(tmp))
            scores,good=run.validated_scores(out,c,groups,meta,checked)
            self.assertEqual(set(scores),set(run.POINTS))
            variants=[]
            r=copy.deepcopy(good);del r['training']['joint'];variants.append(r)
            r=copy.deepcopy(good);r['starts']['ABC_continuation']='reset';variants.append(r)
            r=copy.deepcopy(good);r['training']['BCA_stage2']['first_adam_step']=1;variants.append(r)
            r=copy.deepcopy(good);r['points']['joint']['scores']['development']['path']='scores/initial_development.npy';variants.append(r)
            r=copy.deepcopy(good);r['training']['joint']['group_ids'].reverse();variants.append(r)
            for r in variants:
                data.write_json(out/'manifest.json',r)
                with self.assertRaises((ValueError,KeyError)):
                    run.validated_scores(out,c,groups,meta,checked)

    def test_bad_gate_precedes_valid_parser(self):
        with tempfile.TemporaryDirectory(prefix='expression_pilot_contract_') as tmp:
            out,c,groups,meta,checked=synthetic_result(Path(tmp))
            r=data.read_json(out/'manifest.json');del r['points']['joint'];data.write_json(out/'manifest.json',r)
            with patch.object(run,'contract',return_value=c), patch.object(run,'public_inputs',return_value=(groups,meta,checked)), patch.object(run.platform,'system',return_value='Windows'), patch.object(run,'attach_labels',side_effect=AssertionError('Must not parse')) as parse:
                with self.assertRaises(ValueError):run.evaluate(out,Path(tmp)/'evaluation')
                parse.assert_not_called()
                self.assertFalse((Path(tmp)/'evaluation').exists())

    def test_actual_toy_evaluator_ap_matches_sklearn_once(self):
        from sklearn.metrics import average_precision_score
        with tempfile.TemporaryDirectory(prefix='expression_pilot_contract_') as tmp:
            out,c,groups,meta,checked=synthetic_result(Path(tmp))
            destination=Path(tmp)/'evaluation'
            real_attach=run.attach_labels
            with patch.object(run,'contract',return_value=c), patch.object(run,'public_inputs',return_value=(groups,meta,checked)), patch.object(run.platform,'system',return_value='Windows'), patch.object(run,'attach_labels',wraps=real_attach) as parse:
                r=run.evaluate(out,destination)
                self.assertEqual(parse.call_count,1)
                self.assertEqual(parse.call_args.args[2],'development')
            self.assertEqual(len(r['columns']),22)
            for name in run.POINTS:
                matrix=np.load(destination/(name+'_metrics.npy'),allow_pickle=False)
                scores=np.load(out/'scores'/(name+'_development.npy'),allow_pickle=False)
                expected=[average_precision_score(hand_labels(),s) for s in scores]
                np.testing.assert_allclose(matrix[:,0],expected,atol=1e-15,rtol=0)


if __name__ == '__main__':
    unittest.main()
