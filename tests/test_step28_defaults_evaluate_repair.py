"""Only the two review-identified metadata/launch gaps, synthetic CPU inputs."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

import step28_defaults_evaluate as e
import step28_defaults_evaluate_launch as launch


class Repair(unittest.TestCase):
    def test_dataset_member_mismatch_rejected_without_payloads(self):
        public={'development/items.jsonl':dict(bytes=7,sha256='items')}
        ds=dict(study='seller_alias_continual_expression_shift',root_seed=20260910,
                files={**public,'development/supervision/pairs.csv':dict(bytes=8,sha256='pairs')})
        m=dict(dataset_manifest=dict(path='/fake/manifest.json'),
               development_pairs=dict(path='/fake/development/supervision/pairs.csv',bytes=8,sha256='pairs'))
        e.dataset_binding(m,public,ds)
        for key,value in [('path','/fake/heldout/supervision/pairs.csv'),('sha256','different'),('bytes',9)]:
            wrong={**m,'development_pairs':{**m['development_pairs'],key:value}}
            with self.assertRaises(ValueError):e.dataset_binding(wrong,public,ds)
        with self.assertRaises(ValueError):e.dataset_binding(m,{'development/items.jsonl':dict(bytes=7,sha256='other')},ds)

    def test_real_child_environment_accounting_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'workspace').mkdir()
            (root/'budget.json').write_text(json.dumps(dict(limit_seconds=7200,used_seconds=198,entries=[dict(id='prior',seconds=198)])))
            dummy=root/'child.py'
            dummy.write_text("import json,os; print(json.dumps(dict(nice=os.getpriority(os.PRIO_PROCESS,0),threads=[os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']],cuda=os.environ['CUDA_VISIBLE_DEVICES'])))")
            launch.launch(root,dummy)
            actual=json.loads((root/'formal.stdout.txt').read_text());self.assertGreaterEqual(actual['nice'],15)
            self.assertEqual(actual['threads'],['1','1','1']);self.assertEqual(actual['cuda'],'')
            ledger=json.loads((root/'budget.json').read_text());self.assertGreaterEqual(ledger['used_seconds'],199)
            self.assertTrue(ledger['entries'][-1]['success']);self.assertIsNone(ledger['active'])
            with self.assertRaises(ValueError):launch.launch(root,dummy)

    def test_failure_charge_insufficient_budget_and_actual_review_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'workspace').mkdir()
            ledger=dict(limit_seconds=7200,used_seconds=6600,entries=[dict(id='prior',seconds=6600)])
            (root/'budget.json').write_text(json.dumps(ledger))
            with self.assertRaises(ValueError):launch.launch(root,root/'unreachable.py')
            self.assertFalse((root/'formal.stdout.txt').exists())
            ledger.update(used_seconds=203,entries=[dict(id='prior',seconds=203)])
            (root/'budget.json').write_text(json.dumps(ledger))
            child=root/'fail.py';child.write_text('raise SystemExit(7)')
            with self.assertRaises(RuntimeError):launch.launch(root,child)
            charged=json.loads((root/'budget.json').read_text())
            self.assertEqual(charged['entries'][-1]['exit_code'],7)
            self.assertFalse(charged['entries'][-1]['success'])
            self.assertGreater(charged['used_seconds'],203)
            with self.assertRaises(ValueError):launch.launch(root,child)
            (root/'manifest.json').write_text('{}');(root/'ready.json').write_text('{}')
            args=SimpleNamespace(output=root/'rejected',manifest=root/'manifest.json',review_ready=root/'ready.json',seconds=20)
            with self.assertRaisesRegex(ValueError,'has not closed review'):e.execute(args)
            self.assertTrue((args.output/'failure.json').exists())
            self.assertFalse((args.output/'evaluation_access.json').exists())


if __name__=='__main__':unittest.main()
