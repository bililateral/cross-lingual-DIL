from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/step28_v13_v1_13_v9_4_1_public_projection_portable_v2.py"
SHELL = ROOT / "scripts/run_step28_v13_v1_13_v9_4_1_public_projection_portable_v2_linux_20260831.sh"


class PortableProjectionV2Contracts(unittest.TestCase):
    def test_python_and_shell_parse(self) -> None:
        ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        self.assertTrue(SHELL.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash"))

    def test_linux_has_no_git_or_authority_dependency(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in (
            "git_head",
            "git_tree",
            "git_status",
            "private_custody",
            "projection_key",
            "validate_authorization",
            "v6_regression_result",
            "v7_public_projection_authority",
        ):
            self.assertNotIn(forbidden, source)

    def test_only_existing_transfer_and_return_manifests_control_linux(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("gpu_encoder.validate_transfer", source)
        self.assertIn("gpu_encoder.encode_transfer_to_temporary", source)
        self.assertIn(
            "labels_controllers_membership_qrels_or_audit_truth_present", source
        )
        self.assertNotIn("portable_bundle_manifest", source)
        self.assertNotIn("portable_linux_result", source)

    def test_finalization_uses_existing_scientific_projection_logic(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("base_finalizer.finalize_to_temporary", source)
        self.assertIn("protocol.freeze_combined_manifest", source)
        self.assertIn("shutil.copytree", source)

    def test_shell_is_one_direct_linux_command(self) -> None:
        source = SHELL.read_text(encoding="utf-8")
        self.assertIn("public_projection_portable_v2.py encode-linux", source)
        self.assertNotIn("git ", source)
        self.assertNotIn("authority", source)


if __name__ == "__main__":
    unittest.main()
