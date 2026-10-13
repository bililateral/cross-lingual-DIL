"""Targeted primary check of returned statistics and the N03/N07 decision gap.

Linux py310 only. Reuse the completed original numerical audit; no resampling,
labels, model loads, Memory, or project training/evaluation imports.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from datetime import datetime

assert platform.system() == "Linux"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
os.nice(10)
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
root = args.root
review = root / "review"
web = review / "external/unpacked/independent/attempts/run03/output"
started = time.perf_counter()

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    key = lambda r: tuple(r[k] for k in ("kind", "point_or_contrast", "role", "domain", "metric"))
    result = {key(r): r for r in rows}
    assert len(result) == len(rows) == 990
    return result

errors = []
def numeric(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            numeric(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            numeric(x, y)
    else:
        x, y = float(a), float(b)
        assert math.isfinite(x) and math.isfinite(y)
        error = abs(x - y)
        assert error <= 1e-12
        errors.append(error)

local_rows = csv_rows(root / "analysis/all_metrics.csv")
web_rows = csv_rows(web / "all_metrics_recomputed.csv")
assert local_rows.keys() == web_rows.keys()
for key, row in local_rows.items():
    for field in ("mean", "lower_95", "upper_95"):
        numeric(row[field], web_rows[key][field])

saved = read(root / "job/diagnostics_evaluation/overfitting.json")
returned = read(web / "diagnostics_recomputed.json")
decisions, cf_map = [], {}
for point, roles in returned.items():
    for role, domains in roles.items():
        for domain, value in domains.items():
            original = saved["points"][point][role][domain]
            for field in ("train_raw_curve", "valid_raw_curve", "benefit_gap_curve",
                          "benefit_gap_conditional_intervals", "epoch1_to_6_train_benefit",
                          "epoch1_to_6_valid_benefit", "epoch1_to_6_gap_widening"):
                numeric(value[field], original[field])
            for k, metric in enumerate(("map", "recall_at_5", "log_loss")):
                sign = -1 if metric == "log_loss" else 1
                t = (original["train_raw_curve"][-1][k] - original["train_raw_curve"][0][k]) * sign
                v = (original["valid_raw_curve"][-1][k] - original["valid_raw_curve"][0][k]) * sign
                g = t - v
                ti = original["epoch1_to_6_train_benefit"]["conditional_95pct_interval"][k]
                vi = original["epoch1_to_6_valid_benefit"]["conditional_95pct_interval"][k]
                gi = original["epoch1_to_6_gap_widening"]["conditional_95pct_interval"][k]
                numeric([t, v, g], [original["epoch1_to_6_train_benefit"]["mean"][k],
                                   original["epoch1_to_6_valid_benefit"]["mean"][k],
                                   original["epoch1_to_6_gap_widening"]["mean"][k]])
                point_signal = t > 0 and v < 0 and g > 0
                supported = ti[0] > 0 and vi[1] < 0 and gi[0] > 0
                decision = "有相应迹象" if supported else "证据不足" if point_signal else "未见明确迹象"
                assert decision == original["interpretation"][metric] == value["interpretation"][metric]
                decisions.append(dict(point=point, role=role, domain=domain, metric=metric,
                                      point_signal=point_signal, supported=supported, decision=decision))
                if point == "B_C_cf" and metric == "map":
                    cf_map[role] = dict(domain=domain, train_benefit=t, valid_benefit=v,
                                        gap_widening=g, train_interval=ti, valid_interval=vi,
                                        gap_interval=gi, decision=decision)
assert len(decisions) == 21
assert all(x["decision"] == "未见明确迹象" for x in decisions)
assert saved["points"]["A_shared"]["cache"]["status"] == "NOT_APPLICABLE_FIRST_STAGE"
prior = read(root / "analysis/audit.json")
assert prior["status"] == "PASS_SAVED_RESULT_AUDIT"
result = dict(status="PASS_TARGETED_PRIMARY_REVIEW", observed_at=datetime.now().astimezone().isoformat(),
              python=platform.python_version(), elapsed_seconds=time.perf_counter()-started,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              complete_statistic_rows=990, numeric_values_compared=len(errors), maximum_absolute_error=max(errors),
              diagnostic_decisions=decisions, cf_map_epoch1_to_6=cf_map,
              reused_original_audit="analysis/audit.json", labels_parsed=0, model_loads=0,
              memory_deserializations=0, new_draws=0, updates=0,
              boundary="Compares returned saved statistics to the prior Linux audit and original saved diagnostics. Independently reconstructs 21 decisions from saved curves and intervals; does not regenerate raw metrics, bootstrap intervals or model updates.")
with (review / "primary_verification.json").open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({k:v for k,v in result.items() if k != "diagnostic_decisions"}, ensure_ascii=False))
