"""Independent single-trajectory entry; no queue or evaluation authorization implied."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import time
import traceback

import step28_continual_defaults as method
import step28_continual_five_run as inherited

data = method.data
CONTEXT = "reports/documentation/20261026/continual_defaults"


def sources(extra_paths: tuple[str, ...] = ()) -> list[dict]:
    paths = {data.ROOT/p for p in (
        "scripts/step28_continual_defaults_run.py", "tests/test_step28_continual_defaults.py")}
    paths.update(data.ROOT/p for p in extra_paths)
    todo = list(paths)
    while todo:
        for node in ast.walk(ast.parse(todo.pop().read_text(encoding="utf-8"))):
            names = ([v.name for v in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                if name and (name.startswith("step28_") or name.startswith("test_step28_")):
                    path = data.ROOT/("tests" if name.startswith("test_") else "scripts")/(name+".py")
                    if path not in paths:
                        paths.add(path)
                        todo.append(path)
    paths.update(data.ROOT/p for p in (CONTEXT+"/implementation.zh.txt", CONTEXT+"/authorization.json"))
    for name in ("step28_chinese_base", "step28_alias_ranking", "step28_record_replay", "step28_bge_continual"):
        paths.add(data.ROOT/"schema"/(name+"_policy.json"))
    return [data.record(p, data.ROOT) for p in sorted(paths)]


def execute(job: Path, spec: method.Spec, deadline: float, gate_path: Path) -> dict:
    gate = data.read_json(gate_path)
    if (gate.get("scope") != "original_s_h_three_stage" or gate.get("train_data") is not True
            or gate.get("campaign") != str(job.parent.resolve()) or job.exists()):
        raise ValueError("Fresh independently authorized original-method trajectory required")
    try:
        return inherited.execute(job, spec, deadline, gate_path, implementation=method, source_files=sources())
    except BaseException as error:
        # Only the accepted fresh run may create this directory. Early rejection
        # keeps its original exception and never writes into a rejected old job.
        if job.is_dir():
            data.write_json(job/"failure.json", dict(error=repr(error), traceback=traceback.format_exc(),
                                                   status="STOP_NO_RETRY", time=time.time()))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    execute(args.job, method.Spec.from_dict(data.read_json(args.spec)), args.deadline, args.gate)


if __name__ == "__main__":
    main()
