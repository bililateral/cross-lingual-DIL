"""F01/F02 targeted, review-only pure-data/NumPy check.

Usage: python data_repair_microcheck.py BASELINE_DIRECTORY REPAIR_DIRECTORY
No project model import, training, or existing test-suite execution.
"""
from pathlib import Path
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
import platform
import sys
sys.dont_write_bytecode = True

import numpy as np


old_root, new_root = (Path(v).resolve() for v in sys.argv[1:])

def load_data(root, name):
    path = root / "scripts/step29_vendorlink_data.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = load_data(old_root, "old_data_for_review")
new = load_data(new_root, "new_data_for_review")
runner = new_root / "scripts/step29_vendorlink_run.py"
tree = ast.parse(runner.read_text(encoding="utf-8"))
pure_functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"query_ap", "evaluate"}]
namespace = {"np": np}
exec(compile(ast.Module(body=pure_functions, type_ignores=[]), str(runner), "exec"), namespace)

# Four distinct handwritten accounts in each of the three deterministic splits.
names = {"fit": [], "dev": [], "heldout": []}
for i in range(10000):
    name = f"account{i}"
    v = new.seed_for(9, "identity_split", name.casefold()) / 2**31
    split = "fit" if v < .5 else "dev" if v < .75 else "heldout"
    if len(names[split]) < 4:
        names[split].append(name)
    if all(len(v) == 4 for v in names.values()):
        break
all_names = sum(names.values(), [])
assert len(all_names) == 12
rows = [
    {"market": "X", "vendor": name,
     "title": f"product {i} item {j}",
     "description": f"description {i} number {j} from {name}"}
    for i, name in enumerate(all_names) for j in range(20)
]
kwargs = dict(split_seed=9, train_fraction=.5, dev_fraction=.25,
              records_per_view=2, identities_per_group=4)
old_payload, old_qrels, _ = old.build_views(rows, **kwargs)
new_payload, new_qrels, _ = new.build_views(rows, **kwargs)
assert old_payload["X:fit"] == new_payload["X:fit"]
assert all(old_qrels[g["uid"]] == new_qrels[g["uid"]] for g in new_payload["X:fit"])

def owner(view):
    # The handwritten product number is private fixture truth, not an output UID.
    return view.records[0].title.split()[1]

def labels_for(views):
    return [int(owner(a) == owner(b)) for i, a in enumerate(views) for b in views[i+1:]]

class PureNumericScorer:
    """Protocol probe only, not an ML model or evidence of model behavior."""
    def __init__(self, all_tied):
        self.all_tied = all_tied

    def score(self, group, check):
        values = []
        for i, a in enumerate(group.views):
            for b in group.views[i+1:]:
                key = "|".join(sorted((a.uid, b.uid)))
                values.append(0. if self.all_tied else float(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 3))
        return np.array(values)


cases = []
for split in ("dev", "heldout"):
    for payload in new_payload[f"X:{split}"]:
        group = new.Group.from_dict(payload)
        expected_labels = labels_for(group.views)
        assert new_qrels[group.uid] == expected_labels
        assert all(v.identity is None and len(v.uid) == 64 and ":" not in v.uid for v in group.views)
        assert group.blind() == group
        assert group.evaluation_order().evaluation_order() == group.evaluation_order()
        assert new.Group(group.uid, group.domain, group.split, group.views[::-1]).blind() == group
        relabeled = new.Group(group.uid, group.domain, group.split, tuple(
            new.View(f"arbitrary-old-uid-{i}", v.records, f"arbitrary-private-label-{i}")
            for i, v in enumerate(group.views)))
        assert relabeled.blind() == group
        for all_tied in (True, False):
            scorer = PureNumericScorer(all_tied)
            result = namespace["evaluate"](scorer, [group])[group.uid]
            ids = result["view_ids"]
            assert ids == [v.uid for v in group.views]
            ap = namespace["query_ap"](result["scores"], expected_labels, ids)
            # Independent single-positive AP reference using pair-key lookup.
            scores_by_pair = {frozenset((ids[i], ids[j])): s
                for s, (i, j) in zip(result["scores"], ((i, j) for i in range(8) for j in range(i+1, 8)))}
            reference = []
            for query in group.views:
                ranked = sorted((v for v in group.views if v.uid != query.uid),
                    key=lambda v: (-scores_by_pair[frozenset((v.uid, query.uid))], v.uid))
                rank = next(i for i, v in enumerate(ranked, 1) if owner(v) == owner(query))
                reference.append(1 / rank)
            assert ap == reference
            permutation = [5, 0, 7, 2, 1, 6, 3, 4]
            permuted = new.Group(group.uid, group.domain, group.split, tuple(group.views[i] for i in permutation))
            other = namespace["evaluate"](scorer, [permuted])[group.uid]
            other_ap = namespace["query_ap"](other["scores"], labels_for(permuted.views), other["view_ids"])
            assert dict(zip(ids, ap)) == dict(zip(other["view_ids"], other_ap))
            cases.append({"split": split, "all_tied": all_tied, "query_ap": ap,
                          "per_query_storage_permutation_invariant": True})

assert "torch" not in sys.modules and "transformers" not in sys.modules
paths = [new_root / "scripts/step29_vendorlink_data.py", runner]
result = {
    "scope": "isolated reviewer workspace; 240 handwritten strings, pure data and NumPy protocol checks",
    "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
    "source_hashes": {str(p.relative_to(new_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "source_rows": len(rows), "accounts": len(all_names),
    "fit_payload_and_qrels_equal_to_baseline": True,
    "dev_heldout_qrels_match_private_fixture_membership": True,
    "ids_and_blind_payload_independent_of_identity_and_old_view_uid": True,
    "blind_and_evaluation_order_idempotent": True,
    "view_input_order_does_not_change_canonical_blind_payload": True,
    "evaluate_output_ids_align_with_scores": True,
    "actual_query_ap_matches_independent_single_positive_reference": True,
    "uniform_random_tie_expectation_not_required_for_fixed_realization": True,
    "cases": cases,
    "models_loaded": 0, "training_runs": 0, "test_suite_runs": 0,
    "formal_data_or_labels_read": 0, "torch_or_transformers_imported": False,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
