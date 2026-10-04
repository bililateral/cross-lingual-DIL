#!/usr/bin/env python3
"""External result review: explicit stage/domain algebra on saved group matrices.

Only standard-library modules and NumPy are imported. The submitted audit and
training modules are not executed or imported. No labels, texts, weights, cache
payloads or network are accessed. Output is exclusive-create, so failures remain.

The primary implementation resamples rows first, averages within actual domains,
then evaluates seven explicit endpoint formulas. A separate occurrence-count
matrix multiplication cross-checks every per-domain bootstrap mean. This differs
from the submitted audit's weighted endpoint field implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

ORDERS = ("ABC", "BCA", "CAB")
DOMAINS = ("A", "B", "C")
SAVED_ROLES = ("raw", "stage-cal", "first-cal")
ROLES = (*SAVED_ROLES, "primary")
ENDPOINTS = ("O", "N", "Z", "F_first", "F", "G", "final_all")
METRICS = (
    "average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct",
    "brier", "log_loss", "precision", "recall", "f1", "specificity",
    "balanced_accuracy", "mcc", "map", "mrr", "recall_at_1", "recall_at_3",
    "recall_at_5", "recall_at_10", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10",
)
RANK = np.array([0, 1, 2, 3, *range(12, 22)])
PROBABILITY_LOSSES = np.array([4, 5])
STUDIES = {
    "logit_low": {"arm": "logit_tenth", "references": ("tenth", "logit_quarter", "seq"),
                  "methods": ("seq", "tenth", "logit_quarter", "logit_tenth"), "reused_sets": 63,
                  "policy": "step28_logit_low_policy.json"},
    "risk": {"arm": "risk", "references": ("logit_tenth", "tenth", "seq", "logit_quarter"),
             "methods": ("seq", "tenth", "logit_quarter", "logit_tenth", "risk"), "reused_sets": 81,
             "policy": "step28_risk_policy.json"},
}


def explicit_endpoints(first, second, final, order):
    """Each input ends in (actual_domain, metric), with optional replicate axes."""
    a, b, c = (DOMAINS.index(d) for d in order)
    out = {
        "O": (final[..., a, :] + final[..., b, :]) / 2,
        "N": (second[..., b, :] + final[..., c, :]) / 2,
        "Z": final[..., c, :].copy(),
        "F_first": first[..., a, :] - final[..., a, :],
        "F": ((first[..., a, :] - final[..., a, :])
              + (second[..., b, :] - final[..., b, :])) / 2,
        "G": ((second[..., b, :] - first[..., b, :])
              + (final[..., c, :] - second[..., c, :])) / 2,
        "final_all": final.mean(axis=-2),
    }
    for endpoint in ("F_first", "F", "G"):
        out[endpoint][..., PROBABILITY_LOSSES] *= -1
    return out


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


class ReferenceReview:
    def __init__(self, root, output):
        self.root, self.output = root.resolve(), output.resolve()
        self.inputs = {}
        self.check_counts = Counter()
        self.max_abs = {"saved_results": 0.0, "direct_vs_occurrence_counts": 0.0,
                        "quantile_linear_vs_manual": 0.0, "final_all_identity": 0.0}
        self.max_locations = {}
        self.numeric_comparisons = Counter()
        self.matrix_cache = {}
        self.path_cache = {}
        self.matrix_identities = []
        self.core_bootstrap = {}
        self.studies = {}
        self.draws = np.random.Generator(np.random.PCG64(20260930)).integers(0, 20, (5000, 3, 20))
        self.occurrences = np.empty((5000, 3, 20), dtype=np.float64)
        for d in range(3):
            for g in range(20):
                self.occurrences[:, d, g] = (self.draws[:, d] == g).sum(axis=1) / 20

    def record(self, path, purpose):
        path = Path(path).resolve()
        path.relative_to(self.root)
        raw = path.read_bytes()
        rel = str(path.relative_to(self.root))
        item = {"path": rel, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if rel not in self.inputs:
            self.inputs[rel] = {**item, "purposes": [purpose]}
        elif purpose not in self.inputs[rel]["purposes"]:
            self.inputs[rel]["purposes"].append(purpose)
        return raw, item

    def json(self, path, purpose):
        return json.loads(self.record(path, purpose)[0])

    def bound_array(self, folder, descriptor, purpose):
        path = (folder / descriptor["path"]).resolve()
        path.relative_to(folder.resolve())
        raw, actual = self.record(path, purpose)
        assert all(actual[k] == descriptor[k] for k in ("bytes", "sha256")), (path, "descriptor mismatch")
        self.check_counts["descriptor_bound_arrays"] += 1
        return np.load(io.BytesIO(raw), allow_pickle=False), actual

    def compare_array(self, a, b, category, location, tolerance=3e-12):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        assert a.shape == b.shape, (location, "shape", a.shape, b.shape)
        error = np.abs(a - b)
        maximum = float(error.max(initial=0))
        self.numeric_comparisons[category] += int(error.size)
        if maximum > self.max_abs[category]:
            self.max_abs[category] = maximum
            self.max_locations[category] = location
        assert np.isfinite(error).all() and maximum <= tolerance, (location, maximum, tolerance)

    def compare_tree(self, actual, saved, path):
        if isinstance(actual, dict):
            assert set(actual) == set(saved), (path, "key mismatch", set(actual) ^ set(saved))
            for k in actual:
                self.compare_tree(actual[k], saved[k], path + "/" + k)
        elif isinstance(actual, (list, tuple)):
            assert len(actual) == len(saved), path
            for i, (a, b) in enumerate(zip(actual, saved)):
                self.compare_tree(a, b, path + "/" + str(i))
        elif isinstance(actual, (bool, str)):
            assert actual == saved, (path, actual, saved)
        else:
            self.compare_array([actual], [saved], "saved_results", path)

    def statistics_for_matrix(self, matrix, identity, rows):
        key = (identity["sha256"], tuple(tuple(r) for r in rows))
        if key in self.matrix_cache:
            return self.matrix_cache[key]
        blocks = np.stack([matrix[r] for r in rows])
        original = blocks.mean(axis=1)
        # Direct row indexing followed by within-domain group mean is primary.
        resampled = np.stack([
            blocks[d][self.draws[:, d], :].mean(axis=1) for d in range(3)
        ], axis=1)
        # Algebraically separate implementation, using occurrences of each group.
        frequency_reference = np.stack([
            self.occurrences[:, d, :] @ blocks[d] for d in range(3)
        ], axis=1)
        self.compare_array(resampled, frequency_reference, "direct_vs_occurrence_counts", identity["path"])
        self.matrix_cache[key] = (original, resampled)
        self.check_counts["unique_matrix_resampling"] += 1
        return original, resampled

    def summarize(self, point_by_order, resampled_by_order, path):
        estimate = point_by_order.mean(axis=0)
        distribution = resampled_by_order.mean(axis=0)
        intervals = np.quantile(distribution, (0.025, 0.975), axis=0, method="linear")
        # Independent linear interpolation of order statistics, retained for every summary.
        ordered = np.sort(distribution, axis=0)
        manual = []
        for probability in (0.025, 0.975):
            index = 4999 * probability
            left, right = int(np.floor(index)), int(np.ceil(index))
            alpha = index - left
            manual.append((1 - alpha) * ordered[left] + alpha * ordered[right])
        self.compare_array(intervals, manual, "quantile_linear_vs_manual", path)
        return {metric: {"mean": float(estimate[k]),
                         "per_order": {order: float(point_by_order[i, k]) for i, order in enumerate(ORDERS)},
                         "conditional_95pct_interval": [float(intervals[0, k]), float(intervals[1, k])]}
                for k, metric in enumerate(METRICS)}

    def path(self, stats, arm, role):
        key = (id(stats), arm, role)
        if key in self.path_cache:
            return self.path_cache[key]
        point_outputs, bootstrap_outputs = [], []
        for order in ORDERS:
            points = [order + "_shared", f"{order}_{arm}_stage2", f"{order}_{arm}_stage3"]
            point_outputs.append(explicit_endpoints(*(stats[p][role][0] for p in points), order))
            bootstrap_outputs.append(explicit_endpoints(*(stats[p][role][1] for p in points), order))
        result = {ep: (np.stack([x[ep] for x in point_outputs]),
                       np.stack([x[ep] for x in bootstrap_outputs])) for ep in ENDPOINTS}
        for position in (0, 1):
            self.compare_array(result["final_all"][position],
                               (2 * result["O"][position] + result["Z"][position]) / 3,
                               "final_all_identity", f"{arm}/{role}/{position}")
        self.path_cache[key] = result
        return result

    @staticmethod
    def checks(primary, against_raw):
        mean = lambda ep, metric: primary[ep][metric]["mean"]
        results = {
            "old_map_improves": mean("O", "map") > 0,
            "old_map_interval_above_zero": primary["O"]["map"]["conditional_95pct_interval"][0] > 0,
            "old_recall5_improves": mean("O", "recall_at_5") > 0,
            "new_map_non_decrease": mean("N", "map") >= 0,
            "new_recall5_non_decrease": mean("N", "recall_at_5") >= 0,
        }
        for endpoint in ("O", "N", "Z"):
            if endpoint == "Z":
                for metric in ("map", "recall_at_5"):
                    results[f"Z_{metric}_non_degradation"] = mean("Z", metric) >= 0
            for metric in ("average_precision", "roc_auc"):
                results[f"{endpoint}_{metric}_non_degradation"] = mean(endpoint, metric) >= 0
            for metric in ("brier", "log_loss"):
                results[f"{endpoint}_{metric}_non_degradation"] = mean(endpoint, metric) <= 0
                if endpoint in ("O", "N"):
                    results[f"{endpoint}_{metric}_against_raw_reference"] = against_raw[endpoint][metric]["mean"] <= 0
        assert len(results) == 23
        return results

    def study(self, name, spec):
        folder = self.root / f"reports/seller_alias_continual/20261004/{name}_result"
        ev = folder / "job/evaluation"
        policy = self.json(folder / "source/schema" / spec["policy"], "frozen study policy")
        parent = self.json(folder / "source/schema/step28_bge_continual_policy.json", "frozen inherited statistical policy")
        assert policy["evaluation"]["bootstrap_replicates"] == 5000
        assert policy["evaluation"]["bootstrap_seed"] == 20260930
        assert parent["evaluation"]["quantile_method"] == "linear"
        assert parent["evaluation"]["actual_domain_order"] == list(DOMAINS)
        assert policy["evaluation"]["observed_guard_tolerance"] == 0
        expected_pairs = [[spec["arm"], r] for r in spec["references"]]
        assert policy["evaluation"]["comparison_pairs"] == expected_pairs
        collection = self.json(ev / "collected.json", "new matrices collection")
        reused = self.json(ev / "reference/collected.json", "reused matrices collection")
        saved = self.json(ev / "evaluation.json", "comparison target only")
        saved_draws, _ = self.bound_array(ev, saved["draws"], "saved bootstrap index identity")
        assert np.array_equal(saved_draws, self.draws)
        self.check_counts["saved_draws_identical"] += 1
        for field in ("group_ids", "domains", "metric_columns"):
            assert collection[field] == reused[field], (name, field)
        assert collection["metric_columns"] == list(METRICS)
        assert len(collection["group_ids"]) == len(set(collection["group_ids"])) == 60
        rows = [np.flatnonzero(np.array(collection["domains"]) == domain) for domain in DOMAINS]
        assert [len(x) for x in rows] == [20, 20, 20]
        if hasattr(self, "group_identity"):
            assert self.group_identity == {k: collection[k] for k in ("group_ids", "domains", "metric_columns")}
        else:
            self.group_identity = {k: collection[k] for k in ("group_ids", "domains", "metric_columns")}
        expected_new = {f"{o}_{spec['arm']}_stage{s}" for o in ORDERS for s in (2, 3)}
        assert set(collection["points"]) == expected_new
        expected_all = {o + "_shared" for o in ORDERS} | {
            f"{o}_{arm}_stage{s}" for o in ORDERS for arm in spec["methods"] for s in (2, 3)
        }
        assert not (set(reused["points"]) & set(collection["points"]))
        assert set(reused["points"]) | set(collection["points"]) == expected_all
        assert len(reused["points"]) * 3 == spec["reused_sets"]
        assert saved["new_metric_count_sets"] == 18 and saved["reused_metric_count_sets"] == spec["reused_sets"]
        stats, matrices = {}, {}
        for data, base, label in ((reused, ev / "reference", "reused"), (collection, ev, "new")):
            for point, variants in data["points"].items():
                assert set(variants) == set(SAVED_ROLES), (point, "roles")
                stats[point], matrices[point] = {}, {}
                for role in SAVED_ROLES:
                    matrix, identity = self.bound_array(base, variants[role]["matrix"], f"{name}/{label}/saved group metric matrix")
                    assert matrix.shape == (60, 22) and matrix.dtype == np.float64
                    assert np.isfinite(matrix).all()
                    matrices[point][role] = matrix
                    stats[point][role] = self.statistics_for_matrix(matrix, identity, rows)
                    self.matrix_identities.append({"study": name, "point": point, "role": role,
                                                   "collection": label, **identity})
                    self.check_counts[f"{name}_{label}_matrices"] += 1
                for role in ("stage-cal", "first-cal"):
                    assert np.array_equal(matrices[point][role][:, RANK], matrices[point]["raw"][:, RANK])
                    self.check_counts["all_curve_and_retrieval_columns_invariant"] += 1
                primary_matrix = matrices[point]["stage-cal"].copy()
                primary_matrix[:, RANK] = matrices[point]["raw"][:, RANK]
                assert np.array_equal(primary_matrix, matrices[point]["stage-cal"])
                stats[point]["primary"] = stats[point]["stage-cal"]
        assert set(saved["endpoints"]) == set(spec["methods"])
        result = {"absolute_endpoints": {}, "comparisons": {}}
        for arm in spec["methods"]:
            result["absolute_endpoints"][arm] = {}
            for role in ROLES:
                arrays = self.path(stats, arm, role)
                summarized = {ep: self.summarize(*arrays[ep], f"{name}/{arm}/{role}/{ep}") for ep in ENDPOINTS}
                result["absolute_endpoints"][arm][role] = summarized
                self.compare_tree(summarized, saved["endpoints"][arm][role], f"{name}/absolute/{arm}/{role}")
        assert set(saved["comparisons"]) == {a + "_minus_" + r for a, r in expected_pairs}
        for arm, reference in expected_pairs:
            comparison = arm + "_minus_" + reference
            variants = {}
            for role in ROLES:
                candidate_arrays, reference_arrays = self.path(stats, arm, role), self.path(stats, reference, role)
                variants[role] = {}
                for ep in ENDPOINTS:
                    point = candidate_arrays[ep][0] - reference_arrays[ep][0]
                    boot = candidate_arrays[ep][1] - reference_arrays[ep][1]
                    variants[role][ep] = self.summarize(point, boot, f"{name}/{comparison}/{role}/{ep}")
                    if role == "primary":
                        self.core_bootstrap[f"{comparison}__{ep}__map"] = boot[:, :, METRICS.index("map")].mean(0)
            candidate_arrays = self.path(stats, arm, "primary")
            reference_arrays = self.path(stats, reference, "raw")
            versus_raw = {ep: self.summarize(candidate_arrays[ep][0] - reference_arrays[ep][0],
                                            candidate_arrays[ep][1] - reference_arrays[ep][1],
                                            f"{name}/{comparison}/against_raw/{ep}") for ep in ("O", "N")}
            checks = self.checks(variants["primary"], versus_raw)
            saved_comparison = saved["comparisons"][comparison]
            self.compare_tree(variants["primary"], saved_comparison["primary"], f"{name}/{comparison}/primary")
            self.compare_tree(versus_raw, saved_comparison["against_raw_reference"], f"{name}/{comparison}/against_raw")
            assert checks == saved_comparison["interpretation"]["checks"], comparison
            failed = sorted(k for k, passed in checks.items() if not passed)
            assert failed == sorted(saved_comparison["interpretation"]["failed"])
            assert all(checks.values()) == saved_comparison["interpretation"]["pilot_observed_checks_pass"]
            positive_orders = sum(v > 0 for v in variants["primary"]["O"]["map"]["per_order"].values())
            assert positive_orders == saved_comparison["interpretation"]["positive_old_map_orders"]
            result["comparisons"][comparison] = {
                "all_role_differences": variants, "against_raw_reference": versus_raw,
                "checks": checks, "passed": sum(checks.values()), "total": len(checks),
                "failed": failed, "positive_old_map_orders": positive_orders,
            }
            print(name, comparison, f"{sum(checks.values())}/23", "O MAP", variants["primary"]["O"]["map"], flush=True)
        # Check shared/reused endpoint estimates agree exactly at saved precision
        # between jobs; their underlying matrix descriptors are listed separately.
        if self.studies:
            prior = self.studies["logit_low"]["absolute_endpoints"]
            for arm in prior:
                self.compare_tree(result["absolute_endpoints"][arm], prior[arm], f"cross_job/{arm}")
        self.studies[name] = result
        self.path_cache.clear()

    def hand_reference(self):
        # Unequal domain and stage values detect arrival/domain swaps and missing
        # domain coefficients. Values are set for every metric; probability loss
        # difference directions intentionally have the opposite expected sign.
        first = np.broadcast_to(np.array([10, 20, 30])[:, None], (3, 22)).astype(float).copy()
        second = np.broadcast_to(np.array([12, 25, 33])[:, None], (3, 22)).astype(float).copy()
        final = np.broadcast_to(np.array([8, 24, 40])[:, None], (3, 22)).astype(float).copy()
        expected = {
            "ABC": {"O": 16, "N": 32.5, "Z": 40, "F_first": 2, "F": 1.5, "G": 6, "final_all": 24},
            "BCA": {"O": 32, "N": 20.5, "Z": 8, "F_first": -4, "F": -5.5, "G": -0.5, "final_all": 24},
            "CAB": {"O": 24, "N": 18, "Z": 24, "F_first": -10, "F": -3, "G": 0.5, "final_all": 24},
        }
        results = {}
        for order in ORDERS:
            got = explicit_endpoints(first, second, final, order)
            results[order] = {}
            for endpoint in ENDPOINTS:
                truth = expected[order][endpoint]
                assert got[endpoint][12] == truth, (order, endpoint, got[endpoint][12], truth)
                expected_loss = -truth if endpoint in ("F_first", "F", "G") else truth
                assert got[endpoint][4] == expected_loss and got[endpoint][5] == expected_loss
                results[order][endpoint] = {"map": float(got[endpoint][12]), "brier": float(got[endpoint][4])}
        self.check_counts["handmade_endpoint_formula_checks"] = 3 * 7 * 3
        write_json(self.output / "handmade_formula_check.json", results)

    def save_results(self):
        write_json(self.output / "independent_statistics.json", self.studies)
        write_json(self.output / "input_inventory.json", {"root": str(self.root), "files": list(self.inputs.values())})
        write_json(self.output / "matrix_identities.json", self.matrix_identities)
        write_json(self.output / "group_metric_identity.json", self.group_identity)
        with (self.output / "regenerated_draws.npy").open("xb") as f:
            np.save(f, self.draws, allow_pickle=False)
        with (self.output / "primary_map_bootstrap_distributions.npz").open("xb") as f:
            np.savez_compressed(f, **self.core_bootstrap)
        with (self.output / "full_comparisons.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("study", "comparison", "role", "endpoint", "metric", "mean", "ci_low", "ci_high", *ORDERS))
            for study, data in self.studies.items():
                for comparison, report in data["comparisons"].items():
                    for role, endpoints in report["all_role_differences"].items():
                        for ep, metrics in endpoints.items():
                            for metric, value in metrics.items():
                                writer.writerow((study, comparison, role, ep, metric, value["mean"], *value["conditional_95pct_interval"], *(value["per_order"][o] for o in ORDERS)))
        with (self.output / "all_23_checks.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("study", "comparison", "check", "passed"))
            for study, data in self.studies.items():
                for comparison, report in data["comparisons"].items():
                    for check, passed in report["checks"].items():
                        writer.writerow((study, comparison, check, passed))
        with (self.output / "full_absolute_endpoints.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("study", "arm", "role", "endpoint", "metric", "mean", "ci_low", "ci_high", *ORDERS))
            for study, data in self.studies.items():
                for arm, variants in data["absolute_endpoints"].items():
                    for role, endpoints in variants.items():
                        for ep, metrics in endpoints.items():
                            for metric, value in metrics.items():
                                writer.writerow((study, arm, role, ep, metric, value["mean"], *value["conditional_95pct_interval"], *(value["per_order"][o] for o in ORDERS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    affinity = sorted(os.sched_getaffinity(0))
    assert args.cpu in affinity, (args.cpu, affinity)
    os.sched_setaffinity(0, {args.cpu})
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    script_bytes = Path(__file__).read_bytes()
    environment = {
        "started_utc": started, "argv": sys.argv, "executable": sys.executable,
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "affinity_available": affinity, "affinity_used": sorted(os.sched_getaffinity(0)),
        "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "script_bytes": len(script_bytes),
        "environment_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "submitted_modules_imported": [], "torch_imported": "torch" in sys.modules,
        "network_access": False,
    }
    write_json(args.output / "execution_start.json", environment)
    review = ReferenceReview(args.root, args.output)
    try:
        review.hand_reference()
        for filename in ("SELLER_ALIAS_LOGIT_RISK_RESULT.zh.md", "SELLER_ALIAS_LOGIT_LOW.zh.md",
                         "SELLER_ALIAS_RISK_PILOT.zh.md", "SELLER_ALIAS_BGE_CONTINUAL.zh.md"):
            review.record(args.root / "docs" / filename, "contract/report read to determine scope and formulas")
        for study in STUDIES:
            folder = args.root / f"reports/seller_alias_continual/20261004/{study}_result/source/scripts"
            for filename in ("step28_bge_continual_evaluate.py", "step28_er_weight_evaluate.py"):
                if (folder / filename).exists():
                    review.record(folder / filename, "statistical implementation read only, not imported")
        for filename in ("step28_er_weight_audit.py", "step28_bge_continual_audit.py"):
            review.record(args.root / "scripts" / filename, "submitted audit read only, not imported")
        for study, spec in STUDIES.items():
            review.study(study, spec)
        review.save_results()
        status = {
            "status": "PASS_INDEPENDENT_SAVED_MATRIX_STATISTICS", "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - start,
            "check_counts": dict(review.check_counts), "numeric_comparisons": dict(review.numeric_comparisons),
            "max_absolute_differences": review.max_abs, "max_difference_locations": review.max_locations,
            "comparisons": {k: {"passed": v["passed"], "total": v["total"], "failed": v["failed"]}
                            for data in review.studies.values() for k, v in data["comparisons"].items()},
            "scientific_blockers": [], "reproducibility_defects": [],
            "limitations": [
                "AP/MAP and other label-dependent group metrics are inputs; no label-level recomputation.",
                "Intervals are conditional on fixed models, developed valid groups, calibration, and one training seed.",
                "ABC/BCA/CAB are orders, not independent training seeds; repeated actual groups are paired.",
                "No model, optimizer, true cache payload, server, private labels, test or owners access.",
                "Passing all 23 mixed observational checks is not simultaneous significance or population noninferiority.",
            ],
        }
        write_json(args.output / "machine_verdict.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        write_json(args.output / "failure.json", {
            "status": "FAILED_REFERENCE_RUN", "exception": repr(exc), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - start,
            "input_files_read_before_failure": list(review.inputs.values()),
        })
        raise


if __name__ == "__main__":
    main()
