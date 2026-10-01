"""Independently check saved test evidence on Linux, without labels or models."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

import numpy as np

from step28_alias_pooling_result import (
    Audit, COLUMNS, ROOT, confusion, counts_array, digest, interval, read,
)

ROLES = ("A_raw", "A_cal", "C_raw", "C_cal")
COMPARISONS = (("C_cal", "A_cal"), ("C_cal", "A_raw"),
               ("C_cal", "C_raw"), ("A_cal", "A_raw"))
CLASS_KEYS = ("precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc")
RATE_KEYS = ("tp", "fp", "fn", "tn", "fpr", "recall", "precision", "f1")
RANK_COLUMNS = (0, 1, 2, 3, *range(12, 22))
COMPLETE = "FIXED_PAIR_TEST_EVALUATED_NO_AUTOMATIC_RETRY"


def counts_check(audit: Audit, raw: list, scores: np.ndarray,
                 threshold: float, name: str) -> np.ndarray:
    counts = counts_array(raw)
    if counts.shape != (120, 4):
        raise ValueError("All120 group counts required")
    # Explicit float64 prevents a Python threshold from being rounded to float32.
    predicted = (scores.astype(np.float64) >= np.float64(threshold)).sum(1)
    audit.numbers_equal(predicted, counts[:, 0] + counts[:, 1], name)
    audit.blind_count_rows += 120
    return counts


def run(job: Path, out: Path) -> dict:
    if (out.exists() or not out.is_relative_to(ROOT / "reports")
            or not job.is_relative_to(ROOT / "reports")):
        raise ValueError("Existing project job and fresh report directory required")
    audit = Audit()
    complete = read(job / "completion.json")
    prepared = read(job / "preparation.json")
    collected = read(job / "evaluation/collected.json")
    evaluation = read(audit.file(job, complete["evaluation"]))
    blind = read(audit.file(job, collected["blind"]))
    access = read(job / "heldout_access.json")
    alignment = read(job / "alignment.json")
    if (complete["status"] != COMPLETE or evaluation["status"] != COMPLETE
            or collected["status"] != "ALL_FOUR_TEST_MATRICES_SAVED_BEFORE_STATISTICS"
            or complete["acceptance"] != evaluation["acceptance"]
            or complete["label_parses"] != {"train": 0, "development": 0, "heldout": 1, "owners": 0}
            or complete["training_updates"] != 0 or complete["calibration_fits"] != 0
            or (job.parent / "exit_status.txt").read_text().strip() != "0"
            or list(job.rglob("failure.json")) or tuple(evaluation["columns"]) != COLUMNS
            or set(evaluation["points"]) != set(ROLES)):
        raise ValueError("Incomplete or changed formal test scope")
    if (access["parse_attempts"] != 1 or access["heldout"] != 1
            or any(access[k] != 0 for k in ("train", "development", "owners"))
            or access["blind"] != collected["blind"]
            or alignment["pairs"] != 45360 or alignment["positive_pairs"] != 2400
            or alignment["queries"] != 3360 or alignment["labels_saved"]):
        raise ValueError("Access/alignment evidence differs")
    for record in complete["source_files"]:
        audit.file(ROOT, record)
    if any(obj["source_files"] != complete["source_files"] for obj in (prepared, collected, evaluation)):
        raise ValueError("Scientific snapshots differ")
    policy = read(ROOT / "schema/step28_alias_test_policy.json")
    if (prepared["policy"] != policy or evaluation["statistics"] != policy["statistics"]
            or any(obj["policy_sha256"] != digest(ROOT / "schema/step28_alias_test_policy.json")
                   for obj in (blind, collected, evaluation))):
        raise ValueError("Frozen policy identity differs")
    domains = np.asarray(evaluation["domains"])
    indices = {d: np.flatnonzero(domains == d) for d in "ABC"}
    groups = evaluation["group_ids"]
    if (len(groups) != 120 or len(set(groups)) != 120
            or any(len(x) != 40 for x in indices.values())
            or any(obj["group_ids"] != groups for obj in (blind, collected, alignment))
            or [r["group_uid"] for r in prepared["group_metadata"]] != groups
            or [r["domain"] for r in prepared["group_metadata"]] != domains.tolist()
            or set(blind["models"]) != {"A", "C"}):
        raise ValueError("Full group/domain identities differ")
    saved_draws = np.load(audit.file(job / "evaluation", evaluation["bootstrap"]), allow_pickle=False)
    draws = np.random.Generator(np.random.PCG64(20260928)).integers(0, 40, size=(5000, 3, 40))
    if not np.array_equal(saved_draws, draws):
        raise ValueError("Bootstrap index bytes/values differ")
    weights = np.zeros((5000, 120), dtype=np.int64)
    for position, domain in enumerate("ABC"):
        np.add.at(weights, (np.arange(5000)[:, None], indices[domain][draws[:, position]]), 1)
    matrices, scores, means, automatic = {}, {}, {}, {}
    for name in ("A", "C"):
        model = policy["models"][name]
        mapping = read(audit.file(ROOT, model["calibration"]))
        origin = blind["models"][name]["origin"]
        actual = blind["actual_inference"][name]
        if (origin != {"model": model["payload"], "map": model["calibration"]}
                or mapping["a"] != model["a"] or mapping["b"] != model["b"] or mapping["a"] <= 0
                or actual["model_file"] != {k: model["payload"][k] for k in ("path", "bytes", "sha256")}
                or actual["parameters_sha256"] != model["payload"]["model_parameters_sha256"]
                or actual["groups"] != 120 or actual["training_updates"] != 0):
            raise ValueError("Actual inference/model/map bindings differ")
        for suffix, dtype in (("raw", np.float32), ("cal", np.float64)):
            role = name + "_" + suffix
            scores[role] = audit.array(job, blind["models"][name]["files"][role], 120, 378, dtype)
            rec = evaluation["points"][role]
            matrix = audit.array(job / "evaluation", rec["file"], 120, 22, np.float64)
            if any(rec[k] != collected["points"][role][k] for k in ("file", "counts")):
                raise ValueError("Published metrics/counts differ from pre-statistics capture")
            matrices[role] = matrix
            counts = counts_check(audit, rec["counts"], scores[role], 0., role)
            reference = [[confusion(row[None, :])[k] for k in CLASS_KEYS] for row in counts]
            audit.numbers_equal(matrix[:, [COLUMNS.index(k) for k in CLASS_KEYS]], reference, role)
            means[role] = dict(zip(COLUMNS, matrix.mean(0).tolist(), strict=True))
            audit.mapping(rec["mean"], means[role], role + "/mean")
            for domain, rows in indices.items():
                audit.mapping(rec["by_domain"][domain],
                              dict(zip(COLUMNS, matrix[rows].mean(0).tolist(), strict=True)), role + domain)
            if rec["fixed_classification"]["threshold"] != 0.:
                raise ValueError("Fixed classification threshold changed")
            for domain, rows in {"pooled": np.arange(120), **indices}.items():
                expected = {k: v for k, v in confusion(counts[rows]).items() if k in RATE_KEYS}
                recorded = rec["fixed_classification"]["pooled"] if domain == "pooled" else rec["fixed_classification"]["by_domain"][domain]
                audit.mapping(recorded, expected, role + "/counts/" + domain)
        raw, transformed = scores[name + "_raw"].astype(np.float64), scores[name + "_cal"]
        if not np.array_equal(mapping["a"] * raw + mapping["b"], transformed):
            raise ValueError("Saved affine mapping differs")
        for x, y in zip(raw, transformed, strict=True):
            order = np.argsort(x, kind="stable")
            if (not np.array_equal(order, np.argsort(y, kind="stable"))
                    or not np.array_equal(np.diff(x[order]) == 0, np.diff(y[order]) == 0)):
                raise ValueError("Logit ordering/exact ties changed")
        if not np.array_equal(matrices[name + "_raw"][:, RANK_COLUMNS], matrices[name + "_cal"][:, RANK_COLUMNS]):
            raise ValueError("Curve/retrieval metric changed under positive calibration")
        diagnostic = evaluation["automatic"][name]
        threshold_source = read(audit.file(ROOT, model["old_threshold_source"]))
        if (diagnostic["original_threshold"] != threshold_source["threshold"]
                or diagnostic["mapped_threshold"] != mapping["a"] * threshold_source["threshold"] + mapping["b"]
                or diagnostic["decision"] != "compare_original_logits_with_original_threshold_no_refit"
                or diagnostic["counts"] != collected["automatic"][name]["counts"]):
            raise ValueError("Historical fixed decision rule changed")
        counts = counts_check(audit, diagnostic["counts"], raw, diagnostic["original_threshold"], name + "/strict")
        automatic[name] = {}
        domain_pass = []
        for domain, rows in {"pooled": np.arange(120), **indices}.items():
            summary = {k: v for k, v in confusion(counts[rows]).items() if k in RATE_KEYS}
            sampled = weights[:, rows] @ counts[rows]
            ci = {}
            for metric, positive, other in (("fpr", 1, 3), ("recall", 0, 2), ("precision", 0, 1)):
                denominator = sampled[:, positive] + sampled[:, other]
                values = np.divide(sampled[:, positive], denominator, out=np.zeros(5000), where=denominator > 0)
                ci[metric] = interval(values)
            summary["conditional_95pct_intervals"] = ci
            if domain != "pooled":
                passed = summary["fp"] * 1000 <= summary["fp"] + summary["tn"] and summary["tp"] * 2 >= summary["tp"] + summary["fn"]
                summary["passes_point_gates"] = passed
                domain_pass.append(passed)
            recorded = diagnostic["report"]["pooled"] if domain == "pooled" else diagnostic["report"]["by_domain"][domain]
            audit.mapping(recorded, summary, name + "/strict/" + domain)
            automatic[name][domain] = summary
        if diagnostic["report"]["all_domains_pass"] != all(domain_pass):
            raise ValueError("Strict diagnostic decision differs")
    comparisons, rows_csv = {}, []
    for candidate, reference in COMPARISONS:
        name = candidate + "_minus_" + reference
        delta = matrices[candidate] - matrices[reference]
        # Frequency weighted sums are independent of production's indexed means.
        intervals = np.asarray(interval(weights @ delta / 120))
        comparisons[name] = {}
        for column, metric in enumerate(COLUMNS):
            expected = {"mean": float(delta[:, column].mean()),
                        "conditional_95pct_interval": intervals[:, column].tolist(),
                        "by_domain": {d: float(delta[index, column].mean()) for d, index in indices.items()}}
            audit.mapping(evaluation["comparisons"][name][metric], expected, name + metric)
            comparisons[name][metric] = expected
            rows_csv.append({"comparison": name, "metric": metric, "difference": expected["mean"],
                             "low": intervals[0, column], "high": intervals[1, column], **expected["by_domain"]})
    primary = comparisons["C_cal_minus_A_cal"]
    original = comparisons["C_cal_minus_A_raw"]
    checks = {"T1": primary["map"]["mean"] >= .01,
              "T2": primary["map"]["conditional_95pct_interval"][0] > 0.,
              "T3": primary["recall_at_5"]["mean"] > 0.,
              "T4": primary["average_precision"]["mean"] >= 0.,
              "T5": primary["roc_auc"]["mean"] >= 0.,
              "T6": primary["brier"]["mean"] <= 0.,
              "T7": primary["log_loss"]["mean"] <= 0.,
              "T8": original["brier"]["mean"] <= 0.,
              "T9": original["log_loss"]["mean"] <= 0.}
    if (checks != evaluation["acceptance"]["checks"]
            or all(checks.values()) != evaluation["acceptance"]["passed"]
            or [k for k, v in checks.items() if not v] != evaluation["acceptance"]["failed"]):
        raise ValueError("Independent nine-condition decision differs")
    observed = (
        (primary["map"]["mean"], ">=", .01),
        (primary["map"]["conditional_95pct_interval"][0], ">", 0.),
        (primary["recall_at_5"]["mean"], ">", 0.),
        (primary["average_precision"]["mean"], ">=", 0.),
        (primary["roc_auc"]["mean"], ">=", 0.),
        (primary["brier"]["mean"], "<=", 0.),
        (primary["log_loss"]["mean"], "<=", 0.),
        (original["brier"]["mean"], "<=", 0.),
        (original["log_loss"]["mean"], "<=", 0.),
    )
    for index, (value, operator, bound) in enumerate(observed, 1):
        record = evaluation["acceptance"]["observed"][f"T{index}"]
        if record["operator"] != operator or record["bound"] != bound:
            raise ValueError("Acceptance operator or bound changed")
        audit.numbers_equal(record["value"], value, f"T{index}/observed")
    result = {"status": "PASS_SAVED_TEST_RESULT_AUDIT", "time_utc": datetime.now(timezone.utc).isoformat(),
              "numbers_checked": audit.numbers, "maximum_difference": audit.max_difference,
              "blind_count_rows": audit.blind_count_rows, "order_preserved_groups": 240,
              "acceptance": checks, "means": means, "comparisons": comparisons, "automatic": automatic,
              "source_files": [{"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": digest(Path(__file__))},
                               {"path": "scripts/step28_alias_pooling_result.py", "sha256": digest(ROOT / "scripts/step28_alias_pooling_result.py")}],
              "files": audit.files, "formal_label_parses": 0, "formal_text_parses": 0, "model_loads": 0,
              "training_updates": 0, "new_fits": 0,
              "scope": "Saved identities, mappings, ordering, counts, classification, summaries, group bootstrap and nine guards; not a fresh truth-based recomputation of all22 per-group metrics."}
    out.mkdir(parents=True)
    (out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename, records in (("paired_metrics.csv", rows_csv),
                              ("mean_metrics.csv", [{"metric": metric, **{role: means[role][metric] for role in ROLES}} for metric in COLUMNS])):
        with (out / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return {key: result[key] for key in ("status", "numbers_checked", "maximum_difference", "blind_count_rows", "acceptance")}


if __name__ == "__main__":
    if platform.system() != "Linux" or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError("Use existing Linux py310 with single CPU affinity")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.job.resolve(), arguments.out.resolve())))
