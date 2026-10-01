#!/usr/bin/env python3
"""Audit the small project-import closure of V9.4 model-visible producers."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any


VERSION = "2026-08-27-step28-v13-v1-13-model-visible-source-guard-v9-4"
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REGISTERED_IMPORTS = {
    "step28_balanced_world_schedule.py": (),
    "step28_joint_noise_signatures.py": (),
    "step28_model_visible_matrix.py": (),
    "step28_model_visible_projection.py": (
        "step28_model_visible_matrix",
    ),
    "step28_model_visible_prebuild_source.py": (
        "step28_model_visible_projection",
    ),
    "step28_model_visible_public_replay.py": (
        "step28_model_visible_matrix",
        "step28_model_visible_projection",
    ),
    "step28_model_visible_source_guard.py": (),
    "step28_quality_probe_core.py": (
        "step28_dataset_common",
    ),
    "step28_model_visible_probe_preparer.py": (
        "step28_model_visible_matrix",
        "step28_model_visible_prebuild_source",
        "step28_quality_probe_core",
    ),
    "step28_quality_probe_labels.py": (
        "step28_balanced_world_schedule",
        "step28_quality_probe_core",
        "step28_model_visible_probe_preparer",
    ),
    "step28_quality_probe_policy.py": (
        "step28_balanced_world_schedule",
        "step28_joint_noise_signatures",
        "step28_model_visible_matrix",
        "step28_quality_probe_core",
        "step28_quality_probe_labels",
        "step28_model_visible_probe_preparer",
        "step28_model_visible_source_guard",
    ),
    "step28_dataset_common.py": (),
}
LABEL_FREE_FILES = (
    "step28_joint_noise_signatures.py",
    "step28_model_visible_matrix.py",
    "step28_model_visible_projection.py",
    "step28_model_visible_prebuild_source.py",
    "step28_model_visible_public_replay.py",
    "step28_model_visible_probe_preparer.py",
)
FORBIDDEN_IDENTIFIERS = {
    "controller_membership",
    "controller_groups",
    "override_audit",
    "registered_negative_plan",
    "registered_treatment",
    "mechanism_assignments",
    "identity_assets",
    "pair_labels",
    "train_labels",
    "development_labels",
    "label_rows",
    "y_true",
}


class ModelVisibleSourceGuardV94Error(ValueError):
    """Raised when the registered V9.4 source closure drifts."""


def _project_imports(tree: ast.AST) -> tuple[str, ...]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name for alias in node.names if alias.name.startswith("step28")
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("step28")
        ):
            imports.append(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
            "__import__",
            "eval",
            "exec",
        }:
            raise ModelVisibleSourceGuardV94Error(
                "Registered source uses an untracked dynamic import/evaluation primitive"
            )
    return tuple(imports)


def _forbidden_identifier_uses(tree: ast.AST) -> set[str]:
    """Find direct names, attributes, mapping keys, and reflective field names."""

    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and type(node.value) is str
    }
    return identifiers & FORBIDDEN_IDENTIFIERS


def audit_registered_sources() -> MappingProxyType[str, Any]:
    file_hashes: list[tuple[str, str]] = []
    for filename, expected_imports in REGISTERED_IMPORTS.items():
        path = (SCRIPTS / filename).resolve()
        if path.parent != SCRIPTS.resolve() or not path.is_file():
            raise ModelVisibleSourceGuardV94Error("Registered source path drift")
        raw = path.read_bytes()
        try:
            tree = ast.parse(raw.decode("utf-8"), filename=filename)
        except (UnicodeDecodeError, SyntaxError) as error:
            raise ModelVisibleSourceGuardV94Error(
                "Registered source parse drift"
            ) from error
        observed_imports = _project_imports(tree)
        if observed_imports != expected_imports:
            raise ModelVisibleSourceGuardV94Error(
                f"Registered project-import closure drift: {filename}"
            )
        if filename in LABEL_FREE_FILES:
            overlap = _forbidden_identifier_uses(tree)
            if overlap:
                raise ModelVisibleSourceGuardV94Error(
                    f"Label-free source uses forbidden identifiers: {sorted(overlap)}"
                )
        file_hashes.append((filename, hashlib.sha256(raw).hexdigest()))
    return MappingProxyType({
        "version": VERSION,
        "registered_file_count": len(file_hashes),
        "file_sha256": tuple(file_hashes),
        "label_free_files": LABEL_FREE_FILES,
        "forbidden_identifiers": tuple(sorted(FORBIDDEN_IDENTIFIERS)),
    })
