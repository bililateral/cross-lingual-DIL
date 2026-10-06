"""Assemble the revised objective with the audited fixed-pilot workflow.

Load the unchanged workflow in its own module namespace; no mutation of the old
runner or historical files. All reused functions resolve this explicit binding.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import step28_relation_memory_run as prior
import step28_relation_revision as revision
import step28_relation_revision_admission as admission

spec = importlib.util.spec_from_file_location("relation_revision_workflow", prior.__file__)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
original = revision.original
runner.method = SimpleNamespace(__file__=revision.__file__, update=revision.update,
    **{name: getattr(original, name) for name in
       ("config", "load_model", "make_optimizer", "Memory", "reference", "MAXIMUM_BYTES")})
runner.POLICY = original.data.ROOT / "schema/step28_relation_revision_policy.json"
runner.point_name = lambda order, stage: f"{order}_relation_revision_stage{stage}"
inherited_sources, inherited_policy = runner.sources, runner.policy


def sources() -> list[dict]:
    paths = {original.data.ROOT / row["path"] for row in inherited_sources()}
    paths.update(original.data.ROOT / path for path in (
        "scripts/step28_relation_revision_run.py", "scripts/step28_relation_revision_admission.py",
        "scripts/run_step28_relation_revision_linux_20261006.sh",
        "tests/test_step28_relation_revision_run.py", "tests/test_step28_relation_revision.py",
        "schema/step28_relation_revision_policy.json", "docs/SELLER_ALIAS_RELATION_REVISION_PILOT.zh.md"))
    return [original.data.record(path, original.data.ROOT) for path in sorted(paths)]


def policy() -> dict:
    p = inherited_policy()
    expected = dict(current="BCE+query+0.5hard", history="Q+0.1R", dimension=32,
                    epsilon=.001, history_rank_weight=.1)
    if p["study"] != "seller_alias_relation_revision" or p["objective"] != expected:
        raise ValueError("Revision objective differs from the fixed contract")
    return p


def validate_gate(job: Path, gate_path: Path) -> dict:
    data = original.data
    p, gate = policy(), data.read_json(gate_path)
    if (gate.get("status") != "APPROVED_RELATION_REVISION_PILOT"
            or gate.get("source_files") != sources()
            or gate.get("job") != job.relative_to(data.ROOT).as_posix()
            or gate.get("runtime") != p["runtime"] or gate.get("supervision") != p["supervision"]
            or gate.get("review_disposition") != "NO_OPEN_BLOCKERS"):
        raise ValueError("Matching reviewed revision gate required")
    for kind in ("native", "integration_cpu"):
        r = gate[kind]
        evidence = data.read_json(data.verify(data.ROOT/r["path"], r))
        if evidence["status"] != "PASS_HANDWRITTEN_ONLY":
            raise ValueError("Missing passed revision evidence")
        if evidence.get("mode") != ("native" if kind == "native" else "cpu"):
            raise ValueError("Revision evidence mode differs")
        if kind == "native" and evidence["scientific_sources"] != admission.native_sources():
            raise ValueError("Native revision sources differ")
        if kind == "integration_cpu" and evidence["source_files"] != sources():
            raise ValueError("Integration revision sources differ")
    return p


runner.sources, runner.policy, runner.validate_gate = sources, policy, validate_gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("execute", "finalize"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Existing Linux py310 only")
    call = runner.execute if args.action == "execute" else runner.recover_statistics
    call(args.out.resolve(), args.gate.resolve())
