"""Linux-only metadata tests; no scientific modules, models or datasets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import psutil

spec = importlib.util.spec_from_file_location('observer', Path(__file__).resolve().parents[1] / 'scripts/research_observe.py')
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def process_identity(pid):
    p = psutil.Process(pid)
    return dict(pid=p.pid, owner=p.username(), created=p.create_time(), command=p.cmdline(), cwd=p.cwd())


class ObservationTests(unittest.TestCase):
    def test_wrong_creation_rejects_same_pid(self):
        row = process_identity(psutil.Process().pid)
        self.assertTrue(observer.identity(row)['matches'])
        row['created'] -= 10
        self.assertFalse(observer.identity(row)['matches'])

    def test_json_loader_rejects_path_binding_and_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'x.json'
            p.write_text('[]')
            with self.assertRaises(ValueError):
                observer.read_object(p)
            p.write_text('{"path":"not a parsed manifest"}')
            self.assertEqual(observer.read_object(p), {'path': 'not a parsed manifest'})

    def test_tail_redacts_fields_and_tolerates_partial_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'updates.jsonl'
            p.write_text('{"updates":7,"loss":999,"current_ids":["DO_NOT_EXPORT"]}\n{"updates":')
            self.assertEqual(observer.last_update(p), {'updates': 7})

    def test_live_identity_costs_and_effects_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / 'campaign'
            folder = campaign / 'P/training/P_H_SEQ_s1'
            (folder / 'run').mkdir(parents=True)
            (folder / 'run/completion.json').write_text('{}')
            (folder / 'dispatch_receipt.json').write_text('{"success":false,"released":true}')
            effects = campaign / 'P/analysis/result/results.json'
            effects.parent.mkdir(parents=True)
            effects.write_text('INVALID JSON: MUST NEVER BE READ')
            gate = root / 'gate.json'
            gate.write_text(json.dumps({'campaign': str(campaign)}))
            launch = process_identity(psutil.Process().pid)
            launch['gate'] = dict(bytes=gate.stat().st_size, sha256=hashlib.sha256(gate.read_bytes()).hexdigest())
            (root / 'launch.json').write_text(json.dumps(launch))
            (campaign / 'budget.json').write_text(json.dumps({'used_seconds': {'training': 0}}))
            child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(20)',
                                      '--action', 'train', '--deadline', str(time.monotonic() + 3600)])
            try:
                (campaign / 'dispatch.json').write_text(json.dumps({'active': {'0': process_identity(child.pid)}}))
                before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
                result = observer.observe(root)
                self.assertTrue(result['controller']['matches'])
                self.assertEqual(result['budget']['settled_seconds']['training'], 0)
                self.assertGreater(result['budget']['active_full_commitment_estimate_seconds']['training'], 3590)
                self.assertEqual(result['scientific_decisions']['P']['status'], 'NOT_READ')
                self.assertTrue(result['training'][0]['training_completion_record_exists'])
                self.assertFalse(result['training'][0]['wrapper_validation_success'])
                self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})
            finally:
                child.terminate()
                child.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
