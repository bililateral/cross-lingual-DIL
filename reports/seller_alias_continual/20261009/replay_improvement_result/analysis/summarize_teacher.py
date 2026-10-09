"""Display saved teacher-error scalars; no inference, labels, or resampling."""
import argparse
import hashlib
import json
from pathlib import Path


def identity(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


parser = argparse.ArgumentParser()
parser.add_argument("--job", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
keys = ("D0", "D1", "residual_cross_mean", "residual_difference_mse")
rows, inputs = [], []
for path in sorted((args.job / "run/diagnostics").glob("*.json")):
    data = json.loads(path.read_text())
    errors = data["teacher_errors"]
    if not errors:
        continue
    inputs.append(identity(path))
    means = {key: (sum(item[key] for item in errors) / len(errors)
                   if errors[0][key] is not None else None) for key in keys}
    rows.append({"point": data["point"], "arm": data["arm"],
                 "order": data["order"], "stage": data["stage"],
                 "epoch": data["epoch"], "cache_groups": len(errors), **means})
scope = ("Arithmetic means of already saved error scalars over six surviving cache "
         "groups per stage/epoch. The cache changes between stages. No new scoring, "
         "metric estimation, bootstrap, model loading, or labels. Residual cross "
         "means are not gradient inner products; no causal claim or significance test.")
result = {"scope": scope, "script": identity(Path(__file__)), "inputs": inputs, "rows": rows}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"rows": len(rows), "output": identity(args.out)}))
