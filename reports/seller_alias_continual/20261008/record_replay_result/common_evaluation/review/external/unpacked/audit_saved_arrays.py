"""Independent review of OPENED saved matrices; no project imports or training.

This review runs in the web review workspace, not on the project server.
It uses explicit 60-row endpoint formulas and direct sampled-row gathering.
The submitted supplement uses (order, domain, group, metric) fields and
occurrence weights/einsum. Both must implement the frozen paired estimand.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import time
import zipfile

import numpy as np


INPUT_SHA = "dbed856f06482e29b444ce45b5b44973083bbd28964516ae2cb5ccf5c8be3c38"
CHECK_SHA = "55e9bf419801ec57686898cf78d8237962fc5e5549849a2e0eac3e66318353f1"
ORDERS = ("ABC", "BCA", "CAB")
ARMS = ("C", "S", "LOGIT0.1")
PAIRS = (("C", "LOGIT0.1"), ("S", "LOGIT0.1"), ("C", "S"))
ROLES = ("raw", "stage-cal", "first-cal", "primary")
EPS = ("O", "N", "Z", "F_first", "F", "G", "final_all", "A2")
COLS = ("average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct",
        "brier", "log_loss", "precision", "recall", "f1", "specificity",
        "balanced_accuracy", "mcc", "map", "mrr", "recall_at_1", "recall_at_3",
        "recall_at_5", "recall_at_10", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10")
LOSS_COLS = [COLS.index(x) for x in ("brier", "log_loss")]
RANK_COLS = [COLS.index(x) for x in (
    "average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct",
    "map", "mrr", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
    "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10")]


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def vector(stat):
    return np.array([stat["mean"], *stat["conditional_95pct_interval"],
                     *[stat["per_order"][o] for o in ORDERS]], dtype="<f8")


def main(input_zip, supplement, output):
    start = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    assert not output.exists(), "Review output must be a new directory"
    assert sha(input_zip.read_bytes()) == INPUT_SHA
    z = zipfile.ZipFile(input_zip)
    reads = {}

    def read(name, identity=None):
        blob = z.read(name)
        actual = {"bytes": len(blob), "sha256": sha(blob)}
        if identity is not None:
            assert all(actual[k] == identity[k] for k in actual), name
        reads[name] = actual
        return blob

    def obj(name, identity=None):
        return json.loads(read(name, identity))

    completion = obj("result/job/completion.json")
    original = obj("result/job/evaluation/evaluation.json", completion["evaluation"])
    current = obj("result/job/evaluation/collected.json", original["collected"])
    policy = obj("result/source/schema/step28_record_replay_policy.json")
    refs = {name: obj("reference/" + name, identity)
            for name, identity in policy["reference"]["collections"].items()}
    assert current["metric_columns"] == list(COLS)
    assert len(current["group_ids"]) == len(set(current["group_ids"])) == 60
    assert all(all(ref[k] == current[k] for k in ("metric_columns", "domains", "group_ids"))
               for ref in refs.values())
    rows = [np.flatnonzero(np.array(current["domains"]) == d) for d in "ABC"]
    assert [len(x) for x in rows] == [20, 20, 20]
    source = read("result/source/scripts/step28_bge_continual_evaluate.py")
    assert sha(source) == CHECK_SHA
    history_name = "history/implementation_review_input.zip"
    with zipfile.ZipFile(io.BytesIO(read(history_name))) as history:
        hpath = "background/frozen/logit_low_source/scripts/step28_bge_continual_evaluate.py"
        frozen = history.read(hpath)
        hmanifest = {x["path"]: x for x in json.loads(history.read("manifest.json"))}
        assert sha(frozen) == hmanifest[hpath]["sha256"] == CHECK_SHA
        assert len(frozen) == hmanifest[hpath]["bytes"]
        assert source == frozen

    submitted = json.loads((supplement / "statistics.json").read_text(encoding="utf-8"))
    assert submitted["input_zip_sha256"] == INPUT_SHA
    assert submitted["legacy_check_source_sha256"] == CHECK_SHA
    assert submitted["script_sha256"] == sha((supplement / "step28_record_replay_compare.py").read_bytes())
    log = json.loads((supplement / "calculation.log").read_text(encoding="utf-8"))
    for key, value in log.items():
        assert submitted[key] == value, ("submitted stdout differs", key)

    arrays, physical = {}, {}
    for arm in ARMS:
        for order in ORDERS:
            for stage in (1, 2, 3):
                if arm == "LOGIT0.1":
                    directory = "reference/reference" if stage == 1 else "reference"
                    registry = refs["reference/collected.json" if stage == 1 else "collected.json"]
                    point = order + "_shared" if stage == 1 else f"{order}_logit_tenth_stage{stage}"
                else:
                    directory, registry = "result/job/evaluation", current
                    point = order + "_shared" if stage == 1 else f"{order}_{arm}_stage{stage}"
                values = {}
                for role in ROLES[:3]:
                    rec = registry["points"][point][role]["matrix"]
                    path = directory + "/" + rec["path"]
                    if path not in physical:
                        v = np.load(io.BytesIO(read(path, rec)), allow_pickle=False)
                        assert v.shape == (60, 22) and v.dtype == np.dtype("<f8")
                        assert np.isfinite(v).all()
                        physical[path] = v
                    values[role] = physical[path]
                for role in ("stage-cal", "first-cal"):
                    assert np.array_equal(values[role][:, RANK_COLS], values["raw"][:, RANK_COLS])
                values["primary"] = values["stage-cal"].copy()
                values["primary"][:, RANK_COLS] = values["raw"][:, RANK_COLS]
                arrays[arm, order, stage] = values
    assert len(physical) == 72

    draws = np.random.Generator(np.random.PCG64(20260930)).integers(0, 20, size=(5000, 3, 20))
    endpoints = {}
    for arm in ARMS:
        for role in ROLES:
            fields = {ep: np.zeros((3, 60, 22)) for ep in EPS}
            for oi, order in enumerate(ORDERS):
                def at(stage, arrival):
                    v = np.zeros((60, 22))
                    keep = rows["ABC".index(order[arrival - 1])]
                    v[keep] = arrays[arm, order, stage][role][keep]
                    return v
                fields["O"][oi] = (at(3, 1) + at(3, 2)) / 2
                fields["N"][oi] = (at(2, 2) + at(3, 3)) / 2
                fields["Z"][oi] = at(3, 3)
                fields["F_first"][oi] = at(1, 1) - at(3, 1)
                fields["F"][oi] = (at(1, 1) - at(3, 1) + at(2, 2) - at(3, 2)) / 2
                fields["G"][oi] = (at(2, 2) - at(1, 2) + at(3, 3) - at(2, 3)) / 2
                fields["final_all"][oi] = (at(3, 1) + at(3, 2) + at(3, 3)) / 3
                fields["A2"][oi] = at(2, 2)
            for ep in ("F_first", "F", "G"):
                fields[ep][..., LOSS_COLS] *= -1
            for ep, v in fields.items():
                endpoints[arm, role, ep] = v

    def summarize(field):
        per_order = sum(field[:, r].mean(axis=1) for r in rows)
        means = per_order.mean(axis=0)
        averaged = field.mean(axis=0)
        boot = np.zeros((5000, 22))
        # Same actual-domain draws for ALL orders, stages, methods and roles.
        # Comparison fields have candidate minus reference BEFORE sampling.
        for d, r in enumerate(rows):
            boot += averaged[r[draws[:, d]]].mean(axis=1)
        limits = np.quantile(boot, [0.025, 0.975], axis=0, method="linear")
        return {name: {"mean": float(means[k]), "conditional_95pct_interval": limits[:, k].tolist(),
                       "per_order": dict(zip(ORDERS, per_order[:, k].tolist()))}
                for k, name in enumerate(COLS)}

    results, stat_rows = {}, []
    errors = {"existing_21120": [], "new_S_LOGIT_4224": []}
    exact_preserved = 0
    for label in (*ARMS, *(a + "-" + b for a, b in PAIRS)):
        results[label] = {}
        for role in ROLES:
            results[label][role] = {}
            for ep in EPS:
                if label in ARMS:
                    f = endpoints[label, role, ep]
                    published = original["endpoints"][label][role][ep]
                else:
                    a, b = next((a, b) for a, b in PAIRS if a + "-" + b == label)
                    f = endpoints[a, role, ep] - endpoints[b, role, ep]
                    published = original.get("comparisons", {}).get(label, {}).get("delta", {}).get(role, {}).get(ep)
                actual = summarize(f)
                results[label][role][ep] = actual
                for metric, stat in actual.items():
                    target = submitted["results"][label][role][ep][metric]
                    diff = abs(vector(stat) - vector(target))
                    assert diff.max() < 1e-12, (label, role, ep, metric, diff.max())
                    group = "existing_21120" if published is not None else "new_S_LOGIT_4224"
                    errors[group].extend(diff.tolist())
                    if published is not None:
                        assert vector(target).tobytes() == vector(published[metric]).tobytes()
                        exact_preserved += 6
                    stat_rows.append([label, role, ep, metric, *vector(stat).tolist(), float(diff.max())])
    assert len(errors["existing_21120"]) == exact_preserved == 21120
    assert len(errors["new_S_LOGIT_4224"]) == 4224

    # Transcribe the original contract as 23 named rules; do not import/evaluate
    # comparison_checks or any other project function in this independent path.
    definitions = [
        ("old_map_improves", "O", "map", "mean", ">"),
        ("old_map_interval_above_zero", "O", "map", "ci_lower", ">"),
        ("old_recall5_improves", "O", "recall_at_5", "mean", ">"),
        ("new_map_non_decrease", "N", "map", "mean", ">="),
        ("new_recall5_non_decrease", "N", "recall_at_5", "mean", ">=")]
    for ep in ("O", "N"):
        for metric, op in (("average_precision", ">="), ("roc_auc", ">="), ("brier", "<="), ("log_loss", "<=")):
            definitions.append((f"{ep}_{metric}_non_degradation", ep, metric, "mean", op))
        for metric in ("brier", "log_loss"):
            definitions.append((f"{ep}_{metric}_against_raw_reference", ep, metric, "versus_raw_mean", "<="))
    for metric, op in (("map", ">="), ("recall_at_5", ">="), ("average_precision", ">="),
                       ("roc_auc", ">="), ("brier", "<="), ("log_loss", "<=")):
        definitions.append((f"Z_{metric}_non_degradation", "Z", metric, "mean", op))
    assert len(definitions) == len({x[0] for x in definitions}) == 23
    given_details = {(v["comparison"], v["check"]): v for v in submitted["check_details"]}
    assert len(given_details) == len(submitted["check_details"]) == 69
    checks, details = {}, []
    for a, b in PAIRS:
        label = a + "-" + b
        found = {}
        for key, ep, metric, kind, op in definitions:
            stat = results[label]["primary"][ep][metric]
            if kind == "versus_raw_mean":
                f = endpoints[a, "stage-cal", ep] - endpoints[b, "raw", ep]
                value = float(sum(f[:, r].mean(1) for r in rows).mean(0)[COLS.index(metric)])
            elif kind == "ci_lower":
                value = stat["conditional_95pct_interval"][0]
            else:
                value = stat["mean"]
            passed = value > 0 if op == ">" else value >= 0 if op == ">=" else value <= 0
            given = given_details[label, key]
            assert (given["endpoint"], given["metric"], given["statistic"], given["operator"], given["threshold"]) == (ep, metric, kind, op, 0)
            assert abs(given["value"] - value) < 1e-12 and given["passed"] == passed
            assert submitted["common_23_checks"][label]["checks"][key] == passed
            found[key] = passed
            details.append({"comparison": label, "check": key, "endpoint": ep, "metric": metric,
                            "statistic": kind, "value": value, "operator": op, "threshold": 0,
                            "passed": passed, "ci_lower": stat["conditional_95pct_interval"][0],
                            "ci_upper": stat["conditional_95pct_interval"][1],
                            "ci_role_note": "primary-primary; not a CI for cross-role raw guards"})
        d = submitted["common_23_checks"][label]
        assert d["pilot_observed_checks_pass"] == all(found.values())
        assert d["failed"] == [k for k, ok in found.items() if not ok]
        assert d["positive_old_map_orders"] == sum(v > 0 for v in results[label]["primary"]["O"]["map"]["per_order"].values())
        assert d["future_three_seed_qualification"] == "NOT_EVALUATED_SINGLE_SEED_PILOT"
        checks[label] = {"passed": sum(found.values()), "total": len(found), "failed": d["failed"]}

    original_five = {k: {"checks": v["checks"], "pass": v["pass"]} for k, v in original["comparisons"].items()}
    assert original_five == submitted["original_five_checks"]
    assert submitted["original_development_criteria_pass"] is original["development_criteria_pass"] is False
    for label, expected in original_five.items():
        v = results[label]["primary"]
        recomputed = {"O_MAP_lower_positive": v["O"]["map"]["conditional_95pct_interval"][0] > 0,
                      "final_all_MAP_mean_positive": v["final_all"]["map"]["mean"] > 0,
                      "A2_MAP_lower_ge_minus_point01": v["A2"]["map"]["conditional_95pct_interval"][0] >= -.01,
                      "Z_MAP_lower_ge_minus_point01": v["Z"]["map"]["conditional_95pct_interval"][0] >= -.01,
                      "O_AP_lower_ge_minus_point01": v["O"]["average_precision"]["conditional_95pct_interval"][0] >= -.01}
        assert recomputed == expected["checks"] and all(recomputed.values()) == expected["pass"]

    # Check all 704 display rows and 69 rule display rows, not just examples.
    lines = (supplement / "tables.zh.txt").read_text(encoding="utf-8").splitlines()
    table_rows = 0
    for role in ROLES:
        for ep in EPS:
            offset = lines.index(f"{role} / {ep}") + 2
            for j, metric in enumerate(COLS):
                vals = [f"{submitted['results'][arm][role][ep][metric]['mean']:.9f}" for arm in ARMS]
                for a, b in PAIRS:
                    st = submitted["results"][a + "-" + b][role][ep][metric]
                    lo, hi = st["conditional_95pct_interval"]
                    vals.append(f"{st['mean']:+.9f} [{lo:+.9f}, {hi:+.9f}]")
                assert lines[offset + j] == " | ".join([metric, *vals])
                table_rows += 1
    for v in submitted["check_details"]:
        expected = f"{v['comparison']} | {v['check']} | {v['value']:+.12g} | {v['operator']} 0 | {v['passed']}"
        assert lines.count(expected) == 1
    assert table_rows == 704

    # Inspect opened training-log metadata only to assess which diagnostics
    # actually exist. No scores, labels, model or serialized Memory is loaded.
    exposure = []
    for name in sorted(n for n in z.namelist() if n.startswith("result/job/run/updates/") and n.endswith(".json")):
        v = obj(name)
        current_counts = Counter(v["current_ids"])
        history_counts = Counter(v["history_ids"])
        point_name = Path(name).name
        pt = obj("result/job/run/points/" + point_name)
        assert not (set(v["current_ids"]) & set(current["group_ids"]))
        assert not (set(pt["calibration_ids"]) & set(current["group_ids"]))
        assert not (set(v["current_ids"]) & set(pt["calibration_ids"]))
        assert len(v["rows"]) == len(v["current_ids"]) == 288
        assert len(current_counts) == 48 and set(current_counts.values()) == {6}
        if v["stage"] > 1:
            assert len(history_counts) == 6 and sum(history_counts.values()) == 288
        else:
            assert not history_counts
        schemas = sorted({tuple(sorted(row)) for row in v["rows"]})
        components = sorted({tuple(sorted(row["current"])) for row in v["rows"]})
        exposure.append({"path": name, "stage": v["stage"], "order": v["order"], "arm": v["arm"],
                         "updates": len(v["rows"]), "current_unique_groups": len(current_counts),
                         "current_presentations_each": 6, "history_presentations": sum(history_counts.values()),
                         "history_unique_groups": len(history_counts),
                         "history_min_max": [min(history_counts.values()), max(history_counts.values())] if history_counts else None,
                         "history_mean": sum(history_counts.values()) / len(history_counts) if history_counts else None,
                         "row_key_schemas": schemas, "current_component_key_schemas": components,
                         "saved_score_roles": sorted(pt["scores"]), "calibration_groups": len(pt["calibration_ids"]),
                         "current_calibration_valid_uid_disjoint": True})
    assert len(exposure) == 15

    output.mkdir(parents=True)
    with (output / "independent_statistics.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["comparison_or_arm", "role", "endpoint", "metric", "mean", "ci_lower", "ci_upper", *ORDERS, "maximum_absolute_error"])
        w.writerows(stat_rows)
    with (output / "all_69_checks.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(details[0])); w.writeheader(); w.writerows(details)
    (output / "independent_statistics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "overfitting_saved_evidence.json").write_text(json.dumps(exposure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"status": "PASS_SAVED_ARRAY_INCREMENTAL_REVIEW", "started_at_utc": start_utc,
               "elapsed_seconds": time.perf_counter() - start,
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "platform": platform.platform(), "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
               "scope": "Web review workspace only; not project Linux native replay; no training or raw labels/text/weights/Memory read",
               "input_zip_sha256": INPUT_SHA, "audit_script_sha256": sha(Path(__file__).read_bytes()),
               "frozen_evaluator_identical_to_LOGIT_archive": True,
               "physical_metric_matrices": len(physical), "matrix_numeric_values": sum(v.size for v in physical.values()),
               "rank_columns_equal_all_roles": True, "paired_group_metadata_equal": True,
               "statistics": {k: {"numbers": len(v), "maximum_absolute_error": max(v)} for k, v in errors.items()},
               "original_statistics_preserved_bitwise": exact_preserved,
               "table_metric_rows_verified": table_rows, "table_decision_rows_verified": 69,
               "common_23_checks": checks, "original_five_checks": original_five,
               "original_development_criteria_pass": False,
               "overfitting_diagnosis": "NOT_ASSESSABLE_FROM_SAVED_ARTIFACTS",
               "opened_log_metadata_files": len(exposure), "read_identities": reads}
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "read_identities"}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-zip", required=True, type=Path)
    p.add_argument("--supplement-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    main(a.input_zip, a.supplement_dir, a.output)
