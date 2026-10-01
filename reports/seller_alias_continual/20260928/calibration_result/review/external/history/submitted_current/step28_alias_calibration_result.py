"""Audit saved calibration outputs on Linux without labels, fitting or models."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import platform
from pathlib import Path

import numpy as np

from step28_alias_pooling_result import (
    Audit, COLUMNS, ROOT, confusion, counts_array, digest, interval, read,
)

SEEDS = ("s0", "s1", "s2")
ARMS = ("d", "hard")
VARIANTS = ("d_raw", "d_calibrated", "hard_raw", "hard_calibrated")
COMPARISONS = (("hard_calibrated", "d_calibrated"), ("hard_calibrated", "d_raw"),
               ("hard_calibrated", "hard_raw"), ("d_calibrated", "d_raw"))
CLASS_KEYS = ("precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc")
RATE_KEYS = ("tp", "fp", "fn", "tn", "fpr", "recall", "precision", "f1")
RANK_COLUMNS = (0, 1, 2, 3, *range(12, 22))


def verify_counts(audit: Audit, raw: list, scores: np.ndarray,
                  matrix: np.ndarray | None, threshold: float, name: str) -> np.ndarray:
    counts = counts_array(raw)
    audit.numbers_equal((scores >= threshold).sum(1), counts[:, 0] + counts[:, 1], name)
    audit.blind_count_rows += len(counts)
    if matrix is not None:
        reference = [[confusion(row[None, :])[k] for k in CLASS_KEYS] for row in counts]
        audit.numbers_equal(matrix[:, [COLUMNS.index(k) for k in CLASS_KEYS]], reference, name)
    return counts


def run(job: Path, out: Path) -> dict:
    if out.exists() or not out.is_relative_to(ROOT / "reports"):
        raise ValueError("A fresh project report directory is required")
    audit = Audit()
    evaluation = read(job / "evaluation/evaluation.json")
    collected = read(job / "evaluation/collected.json")
    completed = read(job / "completion.json")
    prepared = read(job / "preparation.json")
    fitted = read(job / "fitted.json")
    if (evaluation["status"] != "CALIBRATION_VALID_EVALUATED_TEST_UNAUTHORIZED"
            or completed["acceptance"] != evaluation["acceptance"]
            or (job.parent / "exit_status.txt").read_text().strip() != "0"
            or completed["label_parses"] != {"train": 1, "development": 1, "heldout": 0, "owners": 0}
            or list(job.rglob("failure.json")) or tuple(evaluation["columns"]) != COLUMNS):
        raise ValueError("Completion, scope or access differs")
    for role in ("train", "development"):
        access = read(job / f"{role}_access.json")
        if access["parse_attempts"] != 1 or access["heldout"] or access["owners"]:
            raise ValueError("Unexpected label access record")
    for record in completed["source_files"]:
        audit.file(ROOT, record)
    if any(obj["source_files"] != completed["source_files"] for obj in (prepared, collected, evaluation)):
        raise ValueError("Scientific source snapshots differ")
    if not (prepared["policy"] == collected["policy"] == evaluation["policy"]):
        raise ValueError("Policy snapshots differ")
    policy = evaluation["policy"]
    old_job = ROOT / policy["historical_job"]
    for name, record in policy["historical_records"].items():
        audit.file(old_job, {**record, "path": name})
    old_collected = read(old_job / "evaluation/collected.json")
    domains = np.asarray(evaluation["domains"])
    rows = {d: np.flatnonzero(domains == d) for d in "ABC"}
    if (any(len(v) != 20 for v in rows.values()) or len(set(evaluation["group_ids"])) != 60
            or evaluation["group_ids"] != old_collected["group_ids"]
            or evaluation["domains"] != old_collected["domains"]):
        raise ValueError("Group/domain alignment differs")
    matrices, scores, fits = {}, {}, {}
    preserved_groups = 0
    for seed in SEEDS:
        for arm in ARMS:
            rid = f"{seed}_{arm}"
            records = fitted["runs"][rid]
            fit = read(audit.file(job, records["fit"]))
            if (fit["status"] != "PASS_CALIBRATION_FIT" or not fit["optimizer_success"]
                    or not .001 <= fit["a"] <= 100 or not -100 <= fit["b"] <= 100
                    or fit["projected_gradient_max"] > 1e-6
                    or fit["final_nll"] > fit["initial_nll"] + 1e-12
                    or fit["optimizer_iterations"] > 200 or fit["objective_calls"] > 2000
                    or fit["pair_count"] != 13608 or fit["positive_count"] != 720):
                raise ValueError("Actual fit does not meet the frozen numerical contract")
            fits[rid] = {key: fit[key] for key in ("a", "b", "initial_nll", "final_nll",
                "optimizer_message", "optimizer_iterations", "objective_calls", "projected_gradient_max")}
            fits[rid]["raw_threshold_for_probability_half"] = -fit["b"] / fit["a"]
            arm_root = old_job / "run" / rid
            arm_manifest = read(audit.file(old_job / "run", fit["origin"]["manifest"]))
            for role, n in (("calibration", 36), ("development", 60)):
                raw = audit.array(arm_root, fit["origin"]["scores"][role], n, 378, np.float32).astype(np.float64)
                cal = audit.array(job, records["scores"][role], n, 378, np.float64)
                if not np.array_equal(fit["a"] * raw + fit["b"], cal):
                    raise ValueError("Saved transformed scores differ from the saved map")
                for x, z in zip(raw, cal, strict=True):
                    order = np.argsort(x, kind="stable")
                    if (not np.array_equal(order, np.argsort(z, kind="stable"))
                            or not np.array_equal(np.diff(x[order]) == 0, np.diff(z[order]) == 0)):
                        raise ValueError("Group logit order/ties changed")
                preserved_groups += n
                pair_matrices = {}
                for variant, values in (("raw", raw), ("calibrated", cal)):
                    point = f"{rid}_{variant}"
                    if role == "calibration":
                        rec = records[f"calibration_{variant}"]
                        matrix = audit.array(job, rec["file"], n, 22, np.float64)
                        verify_counts(audit, rec["counts"], values, matrix, 0., point + role)
                    else:
                        rec = evaluation["points"][point]
                        matrix = audit.array(job / "evaluation", rec["file"], n, 22, np.float64)
                        count = verify_counts(audit, rec["counts_at_logit_zero"], values, matrix, 0., point)
                        matrices[point], scores[point] = matrix, values
                        for domain, indices in {"mean": np.arange(60), **rows}.items():
                            expected = dict(zip(COLUMNS, matrix[indices].mean(0).tolist(), strict=True))
                            audit.mapping(rec["mean"] if domain == "mean" else rec["by_domain"][domain], expected, point)
                        for domain, indices in {"pooled": np.arange(60), **rows}.items():
                            expected = {k: v for k, v in confusion(count[indices]).items() if k in RATE_KEYS}
                            actual = rec["fixed_classification"]["pooled"] if domain == "pooled" else rec["fixed_classification"]["by_domain"][domain]
                            audit.mapping(actual, expected, point + domain)
                    pair_matrices[variant] = matrix
                if not np.array_equal(pair_matrices["raw"][:, RANK_COLUMNS], pair_matrices["calibrated"][:, RANK_COLUMNS]):
                    raise ValueError("A ranking/curve metric changed")
                if role == "calibration":
                    original = audit.array(arm_root, arm_manifest["points"]["6"]["train_metrics"][role]["file"], n, 22, np.float64)
                else:
                    original = audit.array(old_job / "evaluation", old_collected["runs"][rid]["points"]["6"]["file"], n, 22, np.float64)
                audit.numbers_equal(pair_matrices["raw"], original, rid + "/original/" + role)
            diagnostic = evaluation["automatic_diagnostics"][rid]
            verify_counts(audit, diagnostic["counts"], scores[rid + "_raw"], None,
                          diagnostic["original_threshold"], rid + "/strict-diagnostic")
    # Reconstruct the same draws as frequency weights, separately from production indexed means.
    draws = np.random.default_rng(20260927).integers(0, 20, size=(5000, 3, 20))
    weights = np.zeros((5000, 60), dtype=np.int64)
    for i, d in enumerate("ABC"):
        np.add.at(weights, (np.arange(5000)[:, None], rows[d][draws[:, i]]), 1)
    summaries, csv_rows = {}, []
    for candidate, reference in COMPARISONS:
        name = candidate + "_minus_" + reference
        delta = np.stack([matrices[f"{s}_{candidate}"] - matrices[f"{s}_{reference}"] for s in SEEDS])
        average = delta.mean(0)
        ci = np.asarray(interval(weights @ average / 60))
        summaries[name] = {}
        for j, metric in enumerate(COLUMNS):
            value = {"mean": float(average[:, j].mean()), "per_seed": delta[:, :, j].mean(1).tolist(),
                     "by_domain": {d: float(average[index, j].mean()) for d, index in rows.items()},
                     "conditional_95pct_interval": ci[:, j].tolist()}
            audit.mapping(evaluation["comparisons"][name]["metrics"][metric], value, name + metric)
            summaries[name][metric] = value
            csv_rows.append({"comparison": name, "metric": metric, "difference": value["mean"],
                             "low": ci[0, j], "high": ci[1, j],
                             **dict(zip(SEEDS, value["per_seed"], strict=True))})
            for i, seed in enumerate(SEEDS):
                audit.mapping(evaluation["per_seed_comparisons"][name][seed][metric],
                              {"mean": float(delta[i, :, j].mean()),
                               "conditional_95pct_interval": interval(weights @ delta[i, :, j] / 60)}, name + seed)
    p = summaries["hard_calibrated_minus_d_calibrated"]
    r = summaries["hard_calibrated_minus_d_raw"]
    checks = {"map_minimum_observed_gain": p["map"]["mean"] >= .01,
              "map_interval_above_zero": p["map"]["conditional_95pct_interval"][0] > 0,
              "map_improves_each_seed": all(x > 0 for x in p["map"]["per_seed"]),
              "recall_at_5_mean_strictly_improves": p["recall_at_5"]["mean"] > 0,
              "recall_at_5_s0_strictly_improves": p["recall_at_5"]["per_seed"][0] > 0}
    for name, sign in (("average_precision", 1), ("roc_auc", 1), ("brier", -1), ("log_loss", -1)):
        checks[name + "_mean_non_degradation"] = sign * p[name]["mean"] >= 0
        checks[name + "_s0_non_degradation"] = sign * p[name]["per_seed"][0] >= 0
    for name in ("brier", "log_loss"):
        checks[name + "_mean_against_raw_a"] = r[name]["mean"] <= 0
        checks[name + "_s0_against_raw_a"] = r[name]["per_seed"][0] <= 0
    if checks != evaluation["acceptance"]["checks"] or all(checks.values()) != evaluation["acceptance"]["passed"]:
        raise ValueError("Independent seventeen-condition acceptance differs")
    means = {v: np.stack([matrices[f"{s}_{v}"] for s in SEEDS]).mean((0, 1)) for v in VARIANTS}
    result = {"status": "PASS_SAVED_CALIBRATION_RESULT_AUDIT", "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "numbers_checked": audit.numbers, "maximum_difference": audit.max_difference,
              "blind_count_rows": audit.blind_count_rows, "order_preserved_groups": preserved_groups,
              "sources": [{"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(Path(__file__))},
                          {"path": "scripts/step28_alias_pooling_result.py", "sha256": digest(ROOT / "scripts/step28_alias_pooling_result.py")}],
              "acceptance": checks, "fits": fits, "comparisons": summaries,
              "means": {v: dict(zip(COLUMNS, m.tolist(), strict=True)) for v, m in means.items()},
              "formal_label_reads": 0, "model_loads": 0, "new_fits": 0,
              "scope": "Saved aggregate statistics, blind counts, map/score reload, historical matrix identity and acceptance; no fresh truth-based group metrics.",
              "files": audit.files}
    out.mkdir(parents=True)
    import json
    (out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, records in (("paired_metrics.csv", csv_rows),
                          ("mean_metrics.csv", [{"metric": n, **{v: means[v][i] for v in VARIANTS}} for i, n in enumerate(COLUMNS)])):
        with (out / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return {k: result[k] for k in ("status", "numbers_checked", "maximum_difference", "blind_count_rows", "order_preserved_groups")}


if __name__ == "__main__":
    import json
    if platform.system() != "Linux":
        raise RuntimeError("Research result checks use existing Linux py310")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.job.resolve(), args.out.resolve())))
