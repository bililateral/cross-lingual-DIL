"""Tiny review-only data/metric probe. No model, training, or test-suite run.

Usage: python data_audit_microcheck.py /absolute/path/to/extracted/input
All records below are handwritten strings. Source files are read-only.
"""
from pathlib import Path
import ast
from fractions import Fraction
import hashlib
import importlib.util
import json
import platform
import sys
sys.dont_write_bytecode = True

import numpy as np


root = Path(sys.argv[1]).resolve()
data_path = root / "scripts/step29_vendorlink_data.py"
runner_path = root / "scripts/step29_vendorlink_run.py"
spec = importlib.util.spec_from_file_location("handwritten_data_audit", data_path)
data = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = data
spec.loader.exec_module(data)

# Load only the inspected pure-numeric query_ap function, not the runner module
# (which would import Torch and the model implementation).
tree = ast.parse(runner_path.read_text(encoding="utf-8"))
definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "query_ap")
namespace = {"np": np}
exec(compile(ast.Module(body=[definition], type_ignores=[]), str(runner_path), "exec"), namespace)

vendors = []
for i in range(1000):
    name = f"account{i}"
    split_value = data.seed_for(9, "identity_split", name.casefold()) / 2**31
    if .5 <= split_value < .75:
        vendors.append(name)
    if len(vendors) == 4:
        break
assert len(vendors) == 4
rows = [
    {"market": "X", "vendor": vendor,
     "title": f"unique product {vendor} {j}",
     "description": f"text {j} collection {i} from {vendor}"}
    for i, vendor in enumerate(vendors) for j in range(20)
]
payloads, qrels, _ = data.build_views(
    rows, split_seed=9, train_fraction=.5, dev_fraction=.25,
    records_per_view=2, identities_per_group=4,
)
group = data.Group.from_dict(payloads["X:dev"][0])
labels = qrels[group.uid]
n = len(group.views)
prefixes = [v.uid.rsplit(":view", 1)[0] for v in group.views]
from_uid = [int(prefixes[i] == prefixes[j]) for i in range(n) for j in range(i + 1, n)]
from_index = [int(i // 2 == j // 2) for i in range(n) for j in range(i + 1, n)]
actual_query_ap = namespace["query_ap"]([0.] * len(labels), labels, n)

# Independent exact-arithmetic reference for the current label-adjacent ordering.
reference_ap = []
for query in range(n):
    candidates = [i for i in range(n) if i != query]
    positive_ranks = [rank for rank, other in enumerate(candidates, 1) if query // 2 == other // 2]
    reference_ap.append(sum(Fraction(k, rank) for k, rank in enumerate(positive_ranks, 1)) / len(positive_ranks))
reference_map = sum(reference_ap) / n
uniform_tie_expected_map = sum(Fraction(1, k) for k in range(1, n)) / (n - 1)

assert all(v.identity is None for v in group.views)
assert from_uid == labels == from_index
assert all(abs(a - float(b)) < 1e-15 for a, b in zip(actual_query_ap, reference_ap))
assert "torch" not in sys.modules and "transformers" not in sys.modules

result = {
    "scope": "isolated reviewer workspace; handwritten strings and pure-numeric scoring only",
    "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
    "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (data_path, runner_path)},
    "source_rows": len(rows), "observed_accounts": len(vendors), "views": n,
    "blind_identity_fields_all_none": True,
    "qrels_recovered_from_uid_prefix_exact": from_uid == labels,
    "qrels_recovered_from_adjacent_indices_exact": from_index == labels,
    "pair_count": len(labels), "positive_pairs": sum(labels),
    "actual_source_query_ap": actual_query_ap,
    "actual_source_map": sum(actual_query_ap) / n,
    "independent_index_tie_map_fraction": str(reference_map),
    "uniform_label_independent_tie_expected_map_fraction": str(uniform_tie_expected_map),
    "uniform_label_independent_tie_expected_map": float(uniform_tie_expected_map),
    "torch_or_transformers_imported": False,
    "models_loaded": 0, "training_runs": 0, "test_suite_runs": 0,
    "formal_data_or_labels_read": 0,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
