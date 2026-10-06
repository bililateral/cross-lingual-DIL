"""Incremental new-kernel assembly checks; inherit audited workflow fixtures."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import step28_relation_revision_run as assembly
import step28_relation_revision as revision
import step28_relation_revision_admission as admission
import test_step28_relation_memory_run as inherited


class IntegrationTests(unittest.TestCase):
    def test_actual_revision_kernel_full_dispatch_and_restoration(self):
        actual, counts = revision.optimization_step, dict(current=0, historical=0)
        def inspect(*args, **kwargs):
            record = actual(*args, **kwargs)
            counts["current"] += 1
            self.assertAlmostEqual(record["current_total"], record["current_bce"] +
                record["current_rank"] + .5*record["current_hard"], places=5)
            if record["history_uid"] is not None:
                counts["historical"] += 1
                self.assertGreater(record["history_rank"], 0)
                self.assertAlmostEqual(record["history_total"], record["history_compressed"]+
                                       .1*record["history_rank"], places=6)
            return record
        self.assertIs(assembly.runner.method.update, revision.update)
        self.assertIsNot(assembly.runner, assembly.prior)
        with mock.patch.object(inherited, "run", assembly.runner), \
                mock.patch.object(revision, "optimization_step", side_effect=inspect):
            test = inherited.IntegrationTests()
            test.test_full_handwritten_dispatch_checkpoint_blind_gate_and_collection()
        self.assertEqual(counts, dict(current=2592, historical=1728))

    def test_revision_gate_rejects_old_or_mismatched_evidence(self):
        run, data = assembly.runner, revision.original.data
        with tempfile.TemporaryDirectory(prefix="revision_gate_", dir=data.ROOT/"reports") as temp:
            root = Path(temp)
            job, gate_path = root/"new_job", root/"gate.json"
            native_path, cpu_path = root/"native.json", root/"cpu.json"
            native = dict(status="PASS_HANDWRITTEN_ONLY", mode="native", scientific_sources=admission.native_sources(), fixture_only=True)
            cpu = dict(status="PASS_HANDWRITTEN_ONLY", mode="cpu", source_files=assembly.sources(), fixture_only=True)
            data.write_json(native_path, native); data.write_json(cpu_path, cpu)
            p = assembly.policy()
            gate = dict(status="APPROVED_RELATION_REVISION_PILOT", source_files=assembly.sources(),
                job=job.relative_to(data.ROOT).as_posix(), runtime=p["runtime"], supervision=p["supervision"],
                review_disposition="NO_OPEN_BLOCKERS", native=data.record(native_path, data.ROOT),
                integration_cpu=data.record(cpu_path, data.ROOT), fixture_only=True)
            data.write_json(gate_path, gate)
            self.assertEqual(assembly.validate_gate(job, gate_path), p)
            stale = copy.deepcopy(gate); stale["status"] = "APPROVED_RELATION_PILOT"
            data.write_json(gate_path, stale)
            with self.assertRaisesRegex(ValueError, "revision gate"):
                assembly.validate_gate(job, gate_path)
            native["scientific_sources"]["scripts/step28_relation_revision.py"] = "old_or_changed"
            data.write_json(native_path, native)
            gate["native"] = data.record(native_path, data.ROOT)
            data.write_json(gate_path, gate)
            with self.assertRaisesRegex(ValueError, "Native revision sources"):
                assembly.validate_gate(job, gate_path)
