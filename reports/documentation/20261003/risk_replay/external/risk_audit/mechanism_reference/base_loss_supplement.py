"""Independent complete inherited loss on the saved handmade counterexample."""
import hashlib
import json
import math
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / "input" / "scripts"))
import torch
import step28_alias_ranking as ranking

torch.set_num_threads(1)
raw = (OUT / "handmade_inputs.json").read_bytes()
inp = json.loads(raw)
pairs, labels = inp["pairs"], inp["labels"]


def sp(z):
    return max(z, 0.) + math.log1p(math.exp(-abs(z)))


def reference(scores):
    # Full inherited BCE, query-rank, and mean of every positive-versus-top5
    # negative softplus margin. No production loss helper is used here.
    bce = math.fsum(sp(-s) if y else sp(s) for s, y in zip(scores, labels)) / 378
    rank_values, hard_values = [], []
    for q in range(28):
        positive, negative = [], []
        for k, (i, j) in enumerate(pairs):
            if q in (i, j):
                (positive if labels[k] else negative).append(float(scores[k]))
        all_values = positive + negative
        maximum = max(all_values)
        rank_values.append(maximum + math.log(math.fsum(math.exp(z-maximum) for z in all_values))
                           - math.fsum(positive) / len(positive))
        top5 = sorted(negative, reverse=True)[:5]
        hard_values.append(math.fsum(sp(n-p) for p in positive for n in top5) / (len(positive) * 5))
    rank = math.fsum(rank_values) / 28
    hard = math.fsum(hard_values) / 28
    return {"bce": bce, "rank": rank, "hard": hard, "total": bce + rank + .5 * hard}


print(json.dumps({"torch": torch.__version__, "python": sys.version,
                  "input_sha256": hashlib.sha256(raw).hexdigest(),
                  "scope": "saved handmade symmetric 378-score counterexample; no training"}))
results = {}
for name in ("before", "after"):
    scores = inp["map_fpr_counterexample"][name]
    independent = reference(scores)
    actual = ranking.objectives(torch.tensor(scores, dtype=torch.float64),
                                torch.tensor(labels, dtype=torch.float64), .5)
    actual = {key: float(value) for key, value in actual.items()}
    errors = {key: actual[key] - independent[key] for key in independent}
    assert all(math.isclose(actual[key], independent[key], rel_tol=1e-12, abs_tol=1e-12)
               for key in independent)
    results[name] = {"independent": independent, "production": actual, "error": errors}
    print(name.upper() + " " + json.dumps(results[name]))
for key in ("bce", "rank", "hard", "total"):
    assert results["after"]["independent"][key] < results["before"]["independent"][key]
print("DELTA_AFTER_MINUS_BEFORE " + json.dumps({key: results["after"]["independent"][key]
      - results["before"]["independent"][key] for key in ("bce", "rank", "hard", "total")}))
print("COMPLETE_HISTORICAL_SURROGATE_STRICTLY_IMPROVES_WHILE_MAP_AND_FPR_REGRESS")
print("This is an admissible score-space counterexample, not a claimed realized training trajectory.")
