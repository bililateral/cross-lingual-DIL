#!/usr/bin/env python3
"""Independent saved-small-result reconstruction. No project imports or labels.

The computation builds R[order, stage, arrival, bootstrap, metric] from direct
domain means. Bootstrap uses multiplicity matrix products and explicit trajectory
equations, rather than the project's endpoint-field summation. Quantiles use an
explicit sorted order-statistic interpolation (no np.quantile).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ORDERS = ("ABC", "BCA", "CAB")
ROLES = ("raw", "stage-cal", "first-cal")
ENDPOINTS = ("O", "N", "Z", "F_first", "F", "G", "final_all")
METRICS = ("average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct",
           "brier", "log_loss", "precision", "recall", "f1", "specificity",
           "balanced_accuracy", "mcc", "map", "mrr", "recall_at_1", "recall_at_3",
           "recall_at_5", "recall_at_10", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10")
RANKING = tuple(range(4)) + tuple(range(12, 22))
COUNT_NAMES = ("tp", "fp", "fn", "tn")
CLASS_NAMES = ("precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc")
ARMS = ("seq", "er", "half", "quarter", "tenth", "logit_quarter")
PAIRS = (("tenth", "quarter"), ("tenth", "seq"), ("logit_quarter", "quarter"),
         ("logit_quarter", "seq"), ("quarter", "seq"))
EXPECTED_PASSES = None  # Intentionally no claimed pass counts drive computation.


class Evidence:
    def __init__(self, project: Path, out: Path):
        self.project, self.out = project, out
        self.reads = {}
        self.errors = []
        self.compared = {}
        self.checks = []

    def read(self, path: Path, purpose: str) -> bytes:
        content = path.read_bytes()
        relative = str(path.relative_to(self.project))
        if relative not in self.reads:
            self.reads[relative] = {"path": relative, "bytes": len(content),
                                  "sha256": hashlib.sha256(content).hexdigest(), "purposes": []}
        if purpose not in self.reads[relative]["purposes"]:
            self.reads[relative]["purposes"].append(purpose)
        return content

    def json(self, path: Path, purpose: str):
        return json.loads(self.read(path, purpose))

    def npy(self, path: Path, purpose: str):
        return np.load(io.BytesIO(self.read(path, purpose)), allow_pickle=False)

    def assertion(self, name: str, condition, details=None):
        passed = bool(condition)
        self.checks.append({"check": name, "passed": passed, "details": details})
        if not passed:
            self.errors.append(name)

    def compare(self, category: str, path: str, actual, expected):
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        if a.shape != b.shape:
            self.errors.append(f"Shape mismatch {category}: {path}")
            return
        discrepancy = float(np.max(np.abs(a - b))) if a.size else 0.0
        row = self.compared.setdefault(category, {"scalars": 0, "max_abs_error": 0.0, "worst_path": None})
        row["scalars"] += int(a.size)
        if discrepancy > row["max_abs_error"]:
            row.update(max_abs_error=discrepancy, worst_path=path)
        if not np.isfinite(a).all() or discrepancy > 2e-12:
            self.errors.append(f"Numerical mismatch {category}: {path}: {discrepancy}")

    def write_json(self, name, value):
        (self.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def point(order, arm, stage):
    return f"{order}_shared" if stage == 1 else f"{order}_{arm}_stage{stage}"


def count_metrics(counts):
    """Counts may represent one group or an already pooled set; no mean of ratios."""
    tp, fp, fn, tn = (int(x) for x in counts)
    div = lambda n, d: float(n / d) if d else 0.0
    precision, recall, specificity = div(tp, tp + fp), div(tp, tp + fn), div(tn, tn + fp)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": div(2 * tp, 2 * tp + fp + fn),
            "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2,
            "mcc": div(tp * tn - fp * fn, denom), "fpr": div(fp, fp + tn)}


def linear_tails(samples):
    ordered = np.sort(samples, axis=0)
    tails = []
    for p in (0.025, 0.975):
        position = (len(samples) - 1) * p
        low, high = math.floor(position), math.ceil(position)
        fraction = position - low
        tails.append(ordered[low] + fraction * (ordered[high] - ordered[low]))
    return np.stack(tails)


def summarize(samples, per_order):
    ci = linear_tails(samples[1:])
    return {m: {"mean": float(samples[0, k]),
                "per_order": {o: float(per_order[j, k]) for j, o in enumerate(ORDERS)},
                "conditional_95pct_interval": ci[:, k].tolist()}
            for k, m in enumerate(METRICS)}


def explicit_trajectory(R):
    # R[order, stage-1, arrival-1, observed-or-bootstrap, metric].
    # First arrival and second arrival are explicitly distinguished.
    first_when_learned, first_at_end = R[:, 0, 0], R[:, 2, 0]
    second_when_learned, second_at_end = R[:, 1, 1], R[:, 2, 1]
    third_when_learned = R[:, 2, 2]
    e = {
        "O": (first_at_end + second_at_end) / 2,
        "N": (second_when_learned + third_when_learned) / 2,
        "Z": third_when_learned.copy(),
        "F_first": first_when_learned - first_at_end,
        "F": ((first_when_learned - first_at_end) + (second_when_learned - second_at_end)) / 2,
        "G": ((second_when_learned - R[:, 0, 1]) + (third_when_learned - R[:, 1, 2])) / 2,
        "final_all": (first_at_end + second_at_end + third_when_learned) / 3,
    }
    # Probability loss increases have the opposite direction to ranking gains.
    for endpoint in ("F_first", "F", "G"):
        e[endpoint][..., [METRICS.index("brier"), METRICS.index("log_loss")]] *= -1
    return e


def write_csv(out, name, rows):
    with (out / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def original_23(primary, against_raw):
    rows = []
    def add(name, endpoint, metric, quantity, relation, rhs=0.0, source="primary"):
        record = (primary if source == "primary" else against_raw)[endpoint][metric]
        value = record["mean"] if quantity == "mean" else record["conditional_95pct_interval"][0]
        passed = value > rhs if relation == ">" else value >= rhs if relation == ">=" else value <= rhs
        rows.append({"check": name, "endpoint": endpoint, "metric": metric, "source": source,
                     "quantity": quantity, "observed_mean": record["mean"],
                     "ci_low": record["conditional_95pct_interval"][0],
                     "ci_high": record["conditional_95pct_interval"][1],
                     "decision_value": value, "relation": relation, "rhs": rhs, "pass": bool(passed)})
    add("old_map_improves", "O", "map", "mean", ">")
    add("old_map_interval_above_zero", "O", "map", "ci_low", ">")
    add("old_recall5_improves", "O", "recall_at_5", "mean", ">")
    add("new_map_non_decrease", "N", "map", "mean", ">=")
    add("new_recall5_non_decrease", "N", "recall_at_5", "mean", ">=")
    for endpoint in ("O", "N"):
        for metric in ("average_precision", "roc_auc", "brier", "log_loss"):
            add(f"{endpoint}_{metric}_non_degradation", endpoint, metric, "mean",
                "<=" if metric in ("brier", "log_loss") else ">=")
        for metric in ("brier", "log_loss"):
            add(f"{endpoint}_{metric}_against_raw_reference", endpoint, metric, "mean", "<=", source="against_raw_reference")
    for metric in ("map", "recall_at_5", "average_precision", "roc_auc", "brier", "log_loss"):
        add(f"Z_{metric}_non_degradation", "Z", metric, "mean", "<=" if metric in ("brier", "log_loss") else ">=")
    assert len(rows) == 23
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ev = Evidence(args.project.resolve(), args.output.resolve())
    project = ev.project
    low = project / "reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job/evaluation"
    logit = project / "reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace/reports/job/evaluation"
    collections = [("low_new", low), ("low_reference", low / "reference"),
                   ("logit_new", logit), ("logit_reference", logit / "reference")]
    matrices, counts, provenance, collection_counts = {}, {}, {}, {}
    domains = group_ids = None
    for label, root in collections:
        collected = ev.json(root / "collected.json", "matrix/count identity and row-domain schema")
        ev.assertion(f"{label}:22 named metric columns", collected["metric_columns"] == list(METRICS))
        if domains is None:
            domains, group_ids = collected["domains"], collected["group_ids"]
        else:
            ev.assertion(f"{label}:exact ordered groups/domains", collected["domains"] == domains and collected["group_ids"] == group_ids)
        nsets = 0
        for name, role_entries in collected["points"].items():
            ev.assertion(f"{label}:{name}:roles", set(role_entries) == set(ROLES))
            for role, records in role_entries.items():
                nsets += 1
                key = (name, role)
                loaded = {}
                for kind in ("matrix", "counts"):
                    rec = records[kind]
                    path = root / rec["path"]
                    blob = ev.read(path, "saved per-group metrics" if kind == "matrix" else "saved per-group confusion counts")
                    ev.assertion(f"{label}:{name}:{role}:{kind}:size/hash", len(blob) == rec["bytes"] and hashlib.sha256(blob).hexdigest() == rec["sha256"])
                    loaded[kind] = np.load(io.BytesIO(blob), allow_pickle=False) if kind == "matrix" else json.loads(blob)
                matrix = loaded["matrix"]
                ev.assertion(f"{label}:{name}:{role}:matrix_schema", matrix.shape == (60, 22) and matrix.dtype == np.float64 and np.isfinite(matrix).all())
                c = loaded["counts"]
                ev.assertion(f"{label}:{name}:{role}:count_schema", len(c) == 60 and all(set(row) == set(COUNT_NAMES) and all(type(row[k]) is int and row[k] >= 0 for k in COUNT_NAMES) for row in c))
                c = np.array([[row[k] for k in COUNT_NAMES] for row in c], dtype=np.int64)
                ev.assertion(f"{label}:{name}:{role}:20pos358neg", np.all(c[:, 0] + c[:, 2] == 20) and np.all(c[:, 1] + c[:, 3] == 358))
                if key in matrices:
                    ev.assertion(f"{label}:{name}:{role}:reused_values_identical", np.array_equal(matrix, matrices[key]) and np.array_equal(c, counts[key]))
                else:
                    matrices[key], counts[key] = matrix, c
                    provenance[key] = str(root.relative_to(project))
        collection_counts[label] = nsets
    ev.assertion("collection cardinalities 18+81 and 18+45", collection_counts == {"low_new": 18, "low_reference": 81, "logit_new": 18, "logit_reference": 45}, collection_counts)
    ev.assertion("60 unique row identities", len(group_ids) == 60 and len(set(group_ids)) == 60)
    rows = {d: np.array([i for i, value in enumerate(domains) if value == d], dtype=np.int64) for d in "ABC"}
    ev.assertion("actual A/B/C each 20 groups", all(len(x) == 20 for x in rows.values()))
    for name in sorted({k[0] for k in matrices}):
        for role in ("stage-cal", "first-cal"):
            ev.assertion(f"{name}:{role}:saved_ranking_curves_equal_raw", np.array_equal(matrices[name, role][:, RANKING], matrices[name, "raw"][:, RANKING]))

    # Independently regenerate canonical PCG64 draws. Stored draws serve only as
    # reproducibility evidence; the computation below uses regenerated draws.
    generated = np.random.Generator(np.random.PCG64(20260930)).integers(0, 20, size=(5000, 3, 20))
    for label, root in (("low", low), ("logit", logit)):
        saved = ev.npy(root / "bootstrap_draws.npy", "saved bootstrap integer draws compared to independent regeneration")
        ev.assertion(f"{label}:canonical5000_actual_domain_draws", np.array_equal(saved, generated) and saved.dtype == generated.dtype)
    # A draw becomes 20 membership multiplicities, not resampled order indices.
    multiplicities = np.stack([np.apply_along_axis(lambda v: np.bincount(v, minlength=20), 1, generated[:, d]) for d in range(3)])
    ev.assertion("each bootstrap domain multiplicity sums to20", np.all(multiplicities.sum(axis=2) == 20))
    np.save(ev.out / "independent_actual_domain_draws.npy", generated, allow_pickle=False)
    index_map = [{"actual_domain": d, "domain_axis": j, "within_domain_index": k,
                  "saved_row_index": int(row), "group_id": group_ids[row]}
                 for j, d in enumerate("ABC") for k, row in enumerate(rows[d])]
    ev.write_json("bootstrap_index_map.json", {"seed": 20260930, "bit_generator": "PCG64", "replicates": 5000,
                   "shape": [5000, 3, 20], "resampling_unit": "whole group within actual domain",
                   "shared_across": ["method", "stage", "order", "role", "comparison"],
                   "quantile_positions_zero_based": [124.97500000000001, 4874.025], "rows": index_map})

    sample_cache, order_cache, endpoints = {}, {}, {}
    for arm in ARMS:
        endpoints[arm] = {}
        for role in ROLES:
            R = np.empty((3, 3, 3, 5001, 22), dtype=np.float64)
            for oi, order in enumerate(ORDERS):
                for si, stage in enumerate((1, 2, 3)):
                    values = matrices[point(order, arm, stage), role]
                    for ai, domain in enumerate(order):
                        v = values[rows[domain]]
                        R[oi, si, ai, 0] = v.mean(axis=0)
                        R[oi, si, ai, 1:] = (multiplicities["ABC".index(domain)] @ v) / 20.0
            e = explicit_trajectory(R)
            endpoints[arm][role] = {}
            for endpoint, per_order_samples in e.items():
                combined, each_order = per_order_samples.mean(axis=0), per_order_samples[:, 0]
                sample_cache[arm, role, endpoint], order_cache[arm, role, endpoint] = combined, each_order
                endpoints[arm][role][endpoint] = summarize(combined, each_order)
            del R, e
        endpoints[arm]["primary"] = {}
        for endpoint in ENDPOINTS:
            combined = sample_cache[arm, "stage-cal", endpoint].copy()
            each_order = order_cache[arm, "stage-cal", endpoint].copy()
            combined[:, RANKING] = sample_cache[arm, "raw", endpoint][:, RANKING]
            each_order[:, RANKING] = order_cache[arm, "raw", endpoint][:, RANKING]
            sample_cache[arm, "primary", endpoint], order_cache[arm, "primary", endpoint] = combined, each_order
            endpoints[arm]["primary"][endpoint] = summarize(combined, each_order)
        print(f"independent endpoint/bootstrap reconstruction complete: {arm}", flush=True)

    originals = {"low": ev.json(low / "evaluation.json", "comparison target only, read after independent endpoint reconstruction"),
                 "logit": ev.json(logit / "evaluation.json", "comparison target only, read after independent endpoint reconstruction")}
    for label, original in originals.items():
        for arm, roles_expected in original["endpoints"].items():
            for role, es in roles_expected.items():
                for endpoint, metric_expected in es.items():
                    for m, expected in metric_expected.items():
                        actual = endpoints[arm][role][endpoint][m]
                        for field in ("mean", "conditional_95pct_interval"):
                            ev.compare("all_absolute_endpoints_and_CIs", f"{label}/{arm}/{role}/{endpoint}/{m}/{field}", actual[field], expected[field])
                        ev.compare("all_absolute_per_order", f"{label}/{arm}/{role}/{endpoint}/{m}", list(actual["per_order"].values()), [expected["per_order"][o] for o in ORDERS])
    endpoint_rows = []
    for arm, roles_summaries in endpoints.items():
        for role, es in roles_summaries.items():
            for endpoint, summary in es.items():
                for metric, rec in summary.items():
                    endpoint_rows.append({"method": arm, "role": role, "endpoint": endpoint, "metric": metric,
                                          "observed": rec["mean"], "ci_low": rec["conditional_95pct_interval"][0],
                                          "ci_high": rec["conditional_95pct_interval"][1], **rec["per_order"]})
    write_csv(ev.out, "independent_all_endpoints_22metrics.csv", endpoint_rows)
    ev.write_json("independent_endpoints.json", endpoints)

    comparisons, comparison_rows, gate_rows = {}, [], []
    for candidate, reference in PAIRS:
        name = candidate + "_minus_" + reference
        result = {"primary": {}, "against_raw_reference": {}}
        for output, ref_role, names in (("primary", "primary", ENDPOINTS), ("against_raw_reference", "raw", ("O", "N"))):
            for endpoint in names:
                delta = sample_cache[candidate, "primary", endpoint] - sample_cache[reference, ref_role, endpoint]
                per_order_delta = order_cache[candidate, "primary", endpoint] - order_cache[reference, ref_role, endpoint]
                result[output][endpoint] = summarize(delta, per_order_delta)
                for metric, rec in result[output][endpoint].items():
                    comparison_rows.append({"comparison": name, "output": output, "endpoint": endpoint,
                                            "metric": metric, "observed_delta": rec["mean"],
                                            "ci_low": rec["conditional_95pct_interval"][0],
                                            "ci_high": rec["conditional_95pct_interval"][1], **rec["per_order"]})
        gates = original_23(result["primary"], result["against_raw_reference"])
        result["gate_count"] = len(gates)
        result["pass_count"] = sum(g["pass"] for g in gates)
        result["all_pass"] = all(g["pass"] for g in gates)
        result["failed"] = [g["check"] for g in gates if not g["pass"]]
        result["gates"] = gates
        gate_rows.extend({"comparison": name, **g} for g in gates)
        comparisons[name] = result
        found = False
        for label, original in originals.items():
            if name not in original["comparisons"]:
                continue
            found = True
            expected = original["comparisons"][name]
            ev.assertion(f"{label}:{name}:all23independentdecisions", {g["check"]: g["pass"] for g in gates} == expected["interpretation"]["checks"])
            for output in ("primary", "against_raw_reference"):
                for endpoint, metric_expected in expected[output].items():
                    for metric, item in metric_expected.items():
                        rec = result[output][endpoint][metric]
                        for field in ("mean", "conditional_95pct_interval"):
                            ev.compare("all_five_comparisons_mean_CI", f"{label}/{name}/{output}/{endpoint}/{metric}/{field}", rec[field], item[field])
                        ev.compare("all_five_comparisons_per_order", f"{label}/{name}/{output}/{endpoint}/{metric}", list(rec["per_order"].values()), [item["per_order"][o] for o in ORDERS])
        ev.assertion(f"{name}:saved comparator record exists", found)
        print(json.dumps({"comparison": name, "pass_count": result["pass_count"], "total": 23,
                          "O_MAP": result["primary"]["O"]["map"], "failed": result["failed"]}, ensure_ascii=False), flush=True)
    write_csv(ev.out, "independent_five_comparisons_22metrics.csv", comparison_rows)
    write_csv(ev.out, "independent_all_115_checks.csv", gate_rows)
    ev.write_json("independent_comparisons.json", comparisons)

    # Micro classification is recomputed from integer sums. Macro classification
    # is also checked group by group (all 60 rows), not just overall averages.
    absolutes, classification_rows, stage_rows = {}, [], []
    for name, role in sorted(matrices):
        values, c = matrices[name, role], counts[name, role]
        derived = np.array([[count_metrics(row)[m] for m in CLASS_NAMES] for row in c])
        ev.compare("per_group_confusion_to_six_metrics", f"{name}/{role}", derived, values[:, [METRICS.index(m) for m in CLASS_NAMES]])
        groups = {"pooled": np.arange(60), **rows}
        pooled = {d: count_metrics(c[ix].sum(axis=0)) for d, ix in groups.items()}
        macro = {d: dict(zip(METRICS, values[ix].mean(axis=0).tolist())) for d, ix in groups.items()}
        rec = {"group_macro": macro, "pooled_counts_then_rates": pooled}
        absolutes.setdefault(name, {})[role] = rec
        for scope, r in pooled.items():
            classification_rows.append({"point": name, "role": role, "scope": scope, **r})
        for label, original in originals.items():
            if name not in original["absolute_stage_results"]:
                continue
            expected = original["absolute_stage_results"][name][role]
            ev.compare("absolute_stage_macro_all", f"{label}/{name}/{role}", list(macro["pooled"].values()), [expected["macro_all"][m] for m in METRICS])
            for domain in "ABC":
                ev.compare("absolute_stage_macro_domain", f"{label}/{name}/{role}/{domain}", list(macro[domain].values()), [expected["macro_by_domain"][domain][m] for m in METRICS])
            expected_pooled = expected["pooled_fixed_half_classification"]
            for scope, r in {"pooled": expected_pooled["pooled"], **expected_pooled["by_domain"]}.items():
                for m, value in r.items():
                    ev.compare("pooled_integer_counts_and_rates", f"{label}/{name}/{role}/{scope}/{m}", pooled[scope][m], value)
        for domain in "ABC":
            for metric in METRICS:
                stage_rows.append({"point": name, "role": role, "actual_domain": domain, "metric": metric, "observed": macro[domain][metric]})
    write_csv(ev.out, "independent_pooled_classification.csv", classification_rows)
    write_csv(ev.out, "independent_absolute_stage_domain.csv", stage_rows)
    ev.write_json("independent_absolute_stage.json", absolutes)

    # Display every domain/stage change under the primary output convention.
    local_deltas = []
    for candidate, reference in PAIRS:
        for order in ORDERS:
            for stage in (2, 3):
                left = matrices[point(order, candidate, stage), "stage-cal"].copy()
                right = matrices[point(order, reference, stage), "stage-cal"].copy()
                left[:, RANKING] = matrices[point(order, candidate, stage), "raw"][:, RANKING]
                right[:, RANKING] = matrices[point(order, reference, stage), "raw"][:, RANKING]
                for arrival, domain in enumerate(order, 1):
                    delta = (left[rows[domain]] - right[rows[domain]]).mean(axis=0)
                    for k, metric in enumerate(METRICS):
                        local_deltas.append({"comparison": candidate + "_minus_" + reference, "order": order,
                                             "stage": stage, "actual_domain": domain, "arrival": arrival,
                                             "metric": metric, "delta": float(delta[k]),
                                             "observed_worse": bool(delta[k] > 0 if metric in ("brier", "log_loss") else delta[k] < 0)})
    write_csv(ev.out, "independent_primary_all_domain_stage_deltas.csv", local_deltas)

    low_policy = ev.json(project / "schema/step28_er_low_policy.json", "frozen development selection rules")
    eligible = [a for a in low_policy["arms"] if comparisons[a + "_minus_quarter"]["all_pass"]]
    chosen = max(eligible, key=lambda a: (endpoints[a]["primary"]["O"]["map"]["mean"],
                                         endpoints[a]["primary"]["O"]["recall_at_5"]["mean"], low_policy["arms"][a])) if eligible else "quarter"
    selection = {"selected": chosen, "history_weight": low_policy["arms"].get(chosen, 0.25),
                 "eligible": eligible, "fallback_used": not eligible,
                 "scope": "开发valid上的ER配置选择；不是独立最终验证，也不选择LOGIT替换ER"}
    for key in ("selected", "history_weight", "eligible", "fallback_used"):
        ev.assertion(f"independent ER selection:{key}", selection[key] == originals["low"]["selection"][key])
    logit_decision = {"increment_against_matched_er_passes": comparisons["logit_quarter_minus_quarter"]["all_pass"],
                      "all_guards_against_seq_pass": comparisons["logit_quarter_minus_seq"]["all_pass"]}
    for key, value in logit_decision.items():
        ev.assertion(f"independent LOGIT interpretation:{key}", value == originals["logit"]["method_checks"][key])
    ev.write_json("independent_development_decisions.json", {"ER": selection, "LOGIT": logit_decision})

    for arm in ARMS:
        for role in (*ROLES, "primary"):
            ev.compare("all_final_all_algebra_identity", f"{arm}/{role}", sample_cache[arm, role, "final_all"],
                       (2 * sample_cache[arm, role, "O"] + sample_cache[arm, role, "Z"]) / 3)
    env = {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "platform": platform.platform(),
           "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
           "imports": ["Python standard library", "numpy"], "project_code_imported": False, "formal_label_or_model_read": False}
    ev.write_json("environment.json", env)
    ev.write_json("read_coverage.json", {"programmatic_reads": list(ev.reads.values()),
                  "total_unique_input_files": len(ev.reads), "unique_point_role_sets": len(matrices),
                  "collection_sets": collection_counts, "saved_domain_count": len(rows),
                  "group_count": 60, "metric_count": 22, "independent_group_metric_reconstruction": list(CLASS_NAMES),
                  "truth_dependent_saved_metrics": [m for m in METRICS if m not in CLASS_NAMES],
                  "note": "读取已保存数值/计数以及最终JSON对照；无正式文本、标签、缓存正文、权重、test/owners读取，无模型或项目指标函数导入。AP/MAP等以保存每群量为基础，没有独立从真值重新收集。"})
    summary = {"status": "PASS_INDEPENDENT_SAVED_RESULT_RECONSTRUCTION" if not ev.errors else "FAIL",
               "collection_counts": collection_counts, "unique_point_role_sets": len(matrices),
               "endpoint_rows": len(endpoint_rows), "comparison_rows": len(comparison_rows),
               "gate_count": len(gate_rows), "comparison_passes": {k: v["pass_count"] for k, v in comparisons.items()},
               "numerical_comparisons": ev.compared, "checks_total": len(ev.checks),
               "checks_failed": [c for c in ev.checks if not c["passed"]], "errors": ev.errors,
               "ER_selection": selection, "LOGIT_decisions": logit_decision,
               "coverage_boundary": "Conditional reconstruction from 60 saved group metrics/counts; AP/MAP truth-dependent collection and native model/training are not replayed."}
    ev.write_json("checks.json", ev.checks)
    ev.write_json("summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if ev.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
