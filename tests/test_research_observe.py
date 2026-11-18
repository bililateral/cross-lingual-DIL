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

    def test_migrated_gate_and_kernel_identity_via_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'beta'
            root.mkdir()
            (root / 'launch.json').write_text('obsolete; must not be opened')
            (root / 'gate.json').write_text('obsolete; must not be opened')
            gate = root / 'gate_after_postprocess_resume.json'
            gate.write_text(json.dumps({'campaign': str(root / 'campaign')}))
            row = process_identity(psutil.Process().pid)
            del row['created']
            row.update(start_ticks=int(Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19]),
                       boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            binding = dict(path=str(gate), bytes=gate.stat().st_size, sha256=hashlib.sha256(gate.read_bytes()).hexdigest())
            receipt = Path(tmp) / 'beta_handover.json'
            receipt.write_text(json.dumps(dict(identity=row, gate=binding)))
            cmd = [sys.executable, '-B', str(Path(observer.__file__)), '--root', str(root),
                   '--gate', str(gate), '--identity-receipt', str(receipt)]
            result = json.loads(subprocess.check_output(cmd))
            self.assertTrue(result['controller']['matches'])
            self.assertEqual(result['controller']['actual']['pid'], row['pid'])
            self.assertEqual(result['gate_path'], str(gate))
            self.assertFalse(result['dispatch_record_exists'])
            self.assertIsNone(result['dispatch_failure_count'])
            self.assertIsNone(result['active'])
            self.assertIsNone(result['controller_error_present'])
            for change in ({'start_ticks': row['start_ticks'] + 1}, {'boot_id': 'old boot'}):
                self.assertFalse(observer.identity(row | change)['matches'])
            binding['path'] = str(root / 'gate.json')
            receipt.write_text(json.dumps(dict(identity=row, gate=binding)))
            with self.assertRaisesRegex(ValueError, 'different gate'):
                observer.observe(root, gate, receipt)
            binding['path'] = str(gate)
            receipt.write_text(json.dumps(dict(identity=row, gate=binding)))
            gate.write_text('invalid and unbound; must not be parsed')
            with self.assertRaisesRegex(ValueError, 'size/SHA'):
                observer.observe(root, gate, receipt)

    def test_live_identity_costs_and_effects_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / 'campaign'
            folder = campaign / 'P/training/P_H_SEQ_s1'
            (folder / 'run').mkdir(parents=True)
            # A read of the detailed update log would fail, even if its output were filtered.
            (folder / 'run/updates.jsonl').mkdir()
            (folder / 'run/workload.json').write_text('{"current_updates":7,"current_ads":100}')
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
                (campaign / 'dispatch.json').write_text(json.dumps({'active': {'0': process_identity(child.pid)},
                                                                    'failures': [], 'controller_error': None}))
                before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
                result = observer.observe(root)
                self.assertTrue(result['controller']['matches'])
                self.assertEqual(result['budget']['settled_seconds']['training'], 0)
                self.assertGreater(result['budget']['active_full_commitment_estimate_seconds']['training'], 3590)
                self.assertEqual(result['scientific_decisions']['P']['status'], 'NOT_READ')
                self.assertEqual(result['training'][0]['counters'], {'updates': 7, 'total_updates': 7})
                self.assertEqual(result['dispatch_failure_count'], 0)
                self.assertFalse(result['controller_error_present'])
                self.assertTrue(result['training'][0]['training_completion_record_exists'])
                self.assertFalse(result['training'][0]['wrapper_validation_success'])
                self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})
                (folder / 'run/workload.json').unlink()
                self.assertIsNone(observer.observe(root)['training'][0]['counters'])
            finally:
                child.terminate()
                child.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
