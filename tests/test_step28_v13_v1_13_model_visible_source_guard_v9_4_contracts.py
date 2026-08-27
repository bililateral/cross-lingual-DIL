from __future__ import annotations

import ast
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_model_visible_source_guard_v9_4 as guard_v94


class ModelVisibleSourceGuardV94Contracts(unittest.TestCase):
    def test_current_registered_source_closure_passes(self) -> None:
        receipt = guard_v94.audit_registered_sources()
        self.assertEqual(receipt["registered_file_count"], 12)
        self.assertEqual(len(receipt["file_sha256"]), 12)
        self.assertEqual(len({value[1] for value in receipt["file_sha256"]}), 12)
        with self.assertRaises(TypeError):
            receipt["registered_file_count"] = 0

    def test_extra_project_import_is_rejected(self) -> None:
        original = guard_v94.REGISTERED_IMPORTS
        forged = dict(original)
        filename = "step28_v13_v1_13_model_visible_prebuild_source_v9_4.py"
        forged[filename] = ()
        with patch.object(guard_v94, "REGISTERED_IMPORTS", forged):
            with self.assertRaisesRegex(
                guard_v94.ModelVisibleSourceGuardV94Error,
                "project-import closure drift",
            ):
                guard_v94.audit_registered_sources()

    def test_dynamic_import_and_forbidden_identifier_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            guard_v94.ModelVisibleSourceGuardV94Error,
            "dynamic import",
        ):
            guard_v94._project_imports(ast.parse("__import__('step28_private')"))
        tree = ast.parse("value = controller_membership")
        self.assertEqual(
            guard_v94._forbidden_identifier_uses(tree),
            {"controller_membership"},
        )

    def test_mapping_and_reflective_private_field_reads_are_rejected(self) -> None:
        cases = {
            'value = row["controller_membership"]': "controller_membership",
            'value = row.get("override_audit")': "override_audit",
            'value = getattr(row, "registered_treatment")': (
                "registered_treatment"
            ),
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(
                    guard_v94._forbidden_identifier_uses(ast.parse(source)),
                    {expected},
                )


if __name__ == "__main__":
    unittest.main()
