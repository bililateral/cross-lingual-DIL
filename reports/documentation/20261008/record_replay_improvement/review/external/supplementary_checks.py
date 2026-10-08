"""Read-only attachment checks supporting the review; no model/data execution."""
from pathlib import Path
import argparse
import ast
import hashlib
import json


def read_function(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False), [node.lineno, node.end_lineno]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current = args.root / "source/scripts/step28_er_weight.py"
    prior = args.background / "implementation/background/frozen/logit_low_source/scripts/step28_er_weight.py"
    current_ast, current_lines = read_function(current, "update")
    prior_ast, prior_lines = read_function(prior, "update")
    assert current_ast == prior_ast
    original = args.background / "original"
    completion = json.loads((original / "result/job/completion.json").read_text())
    rows = []
    for path in sorted((original / "result/job/run/updates").glob("*.json")):
        value = json.loads(path.read_text())["training_seconds"]
        rows.append({"path": str(path.relative_to(original)), "training_seconds": value})
    assert len(rows) == 15
    training = sum(row["training_seconds"] for row in rows)
    total = completion["budget"]["elapsed_seconds"]
    report = {
        "scope": "Read-only frozen source function and already-open historical timing fields. No current native measurement.",
        "LOGIT_update": {
            "unchanged_AST": True,
            "current_lines": current_lines,
            "frozen_lines": prior_lines,
            "whole_files_identical": current.read_bytes() == prior.read_bytes(),
            "current_sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
            "frozen_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
        },
        "historical_timing": {
            "completion_seconds": total,
            "logged_training_seconds": training,
            "difference_seconds": total - training,
            "naive_36_over_15_scaling_seconds": (total - training) * 36 / 15,
            "rows": rows,
            "limits": "The difference includes old non-training forwards, already separately charged in the new projection. This arithmetic does not isolate checkpoint cost, measure new diagnostics, or establish a bound for the 3600-second allowance.",
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_READ_ONLY_ATTACHMENT_CHECKS", "unchanged_LOGIT_update_AST": True,
                      "old_timing_difference_seconds": total - training}, ensure_ascii=False))


if __name__ == "__main__":
    main()
