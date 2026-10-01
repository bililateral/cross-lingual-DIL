"""Audit completed ranking outputs on Linux, without data, labels or models."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
from pathlib import Path

import numpy as np

from step28_alias_pooling_result import (
    Audit, COLUMNS, ROOT, confusion, counts_array, digest, interval, ratios, read,
)

SEEDS = ("s0", "s1", "s2")
ARMS = ("d", "schedule", "hard")
RUNS = tuple(f"{s}_{a}" for s in SEEDS for a in ARMS)
COMPARISONS = (("hard", "d"), ("schedule", "d"), ("hard", "schedule"))
RATE_KEYS = ("tp", "fp", "fn", "tn", "fpr", "recall", "precision", "f1")
CLASS_KEYS = ("precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc")


def audit_counts(audit: Audit, counts: np.ndarray, scores: np.ndarray,
                 matrix: np.ndarray | None, threshold: float, name: str) -> None:
    audit.numbers_equal((scores.astype(np.float64) >= threshold).sum(1),
                        counts[:, 0] + counts[:, 1], name + "/blind positives")
    audit.blind_count_rows += len(counts)
    if matrix is not None:
        expected = [[confusion(row[None, :])[key] for key in CLASS_KEYS] for row in counts]
        audit.numbers_equal(matrix[:, [COLUMNS.index(k) for k in CLASS_KEYS]], expected,
                            name + "/group classification")


def run(job: Path, out: Path) -> dict:
    if out.exists() or not out.is_relative_to(ROOT / "reports"):
        raise ValueError("New reports output directory required")
    audit = Audit()
    inventory = read(out.parent / "return_inventory.json")
    for record in inventory["files"] + inventory["frozen_sources"]:
        audit.file(ROOT, record)
    evaluation = read(job / "evaluation/evaluation.json")
    collected = read(job / "evaluation/collected.json")
    manifest = read(job / "run/manifest.json")
    complete = read(job / "completion.json")
    policy = read(ROOT / "schema/step28_alias_ranking_policy.json")
    if (evaluation["columns"] != list(COLUMNS) or evaluation["policy"] != policy
            or manifest["policy"] != policy or complete["formal_updates"] != 7776
            or complete["label_parses"] != {"train": 1, "development": 1, "heldout": 0, "owners": 0}
            or (job / "exit_status.txt").read_text().strip() != "0"
            or set(manifest["runs"]) != set(RUNS) or set(evaluation["runs"]) != set(RUNS)
            or evaluation["acceptance"] != complete["acceptance"]):
        raise ValueError("Frozen scope or completion differs")
    if read(job / "run/completion.json")["manifest_sha256"] != digest(job / "run/manifest.json"):
        raise ValueError("Training completion binding differs")
    if (read(job / "run/access.json") != {"train_parse_attempts": 1, "development": 0, "heldout": 0, "owners": 0}
            or read(job / "evaluation/access.json") != {"development_parse_attempts": 1, "train": 0, "heldout": 0, "owners": 0}
            or list(job.rglob("failure.json"))):
        raise ValueError("Access or failure records differ")
    preparse = read(job / "evaluation/preparse_verification.json")
    if (len(preparse["files"]) != 18 or preparse["sources"] != manifest["source_files"]
            or preparse["manifest_sha256"] != digest(job / "run/manifest.json")):
        raise ValueError("Pre-valid eighteen-model verification missing")
    audit.file(ROOT, evaluation["provenance"]["training_manifest"])
    audit.file(job / "evaluation", evaluation["provenance"]["preparse_verification"])
    partition = read(audit.file(job / "run", manifest["partition"]))
    domains = np.asarray(evaluation["domains"])
    rows = {d: np.flatnonzero(domains == d) for d in "ABC"}
    if (any(len(v) != 20 for v in rows.values()) or len(set(evaluation["group_ids"])) != 60
            or evaluation["group_ids"] != [r["group_uid"] for r in partition["development"]]
            or evaluation["domains"] != [r["domain"] for r in partition["development"]]
            or collected["group_ids"] != evaluation["group_ids"] or collected["domains"] != evaluation["domains"]):
        raise ValueError("Development group alignment differs")
    # Frequency-weighted sums independently reproduce the production indexed means.
    draws = np.random.default_rng(20260927).integers(0, 20, size=(5000, 3, 20))
    weights = np.zeros((5000, 60), dtype=np.int64)
    for index, domain in enumerate("ABC"):
        np.add.at(weights, (np.arange(5000)[:, None], rows[domain][draws[:, index]]), 1)
    matrices, auto_counts, training, trajectories, metric_rows = {}, {}, {}, [], []
    maximum_loss_roundoff = 0.0
    for run_id in RUNS:
        arm_root = job / "run" / run_id
        arm = read(audit.file(job / "run", manifest["runs"][run_id]["manifest"]))
        seed, kind = run_id.split("_")
        if (arm["run_id"] != run_id or arm["updates"] != 864 or arm["label_parses"] != 0
                or arm["trainable_parameter_count"] != arm["parameter_count"]
                or arm["fit_group_ids"] != [r["group_uid"] for r in partition["fit"]]
                or arm["calibration_group_ids"] != [r["group_uid"] for r in partition["calibration"]]):
            raise ValueError("Training identity, budget or roles differ")
        updates = audit.array(arm_root, arm["update_log"], 864, 6, np.float64)
        expected_lr = np.array([2e-5 if kind == "d" else
                               1e-5 * (t / 87 if t <= 87 else (864 - t) / 777)
                               for t in range(1, 865)])
        if (not np.array_equal(updates[:, 4], expected_lr) or not np.all(updates[:, 5] == .001)
                or arm["encoder_positive_lr_updates"] != int((expected_lr > 0).sum())
                or np.any(updates[:, :4] < 0) or (kind != "hard" and np.any(updates[:, 2] != 0))):
            raise ValueError("Actual learning-rate or objective trace differs")
        total = updates[:, 0] + updates[:, 1] + (.5 if kind == "hard" else 0) * updates[:, 2]
        if not np.allclose(total, updates[:, 3], rtol=1e-6, atol=1e-7):
            raise ValueError("Actual total loss differs from frozen objective")
        maximum_loss_roundoff = max(maximum_loss_roundoff, float(np.max(np.abs(total - updates[:, 3]))))
        epoch_losses = updates[:, :4].reshape(6, 144, 4).mean(1)
        for i, segment in enumerate(arm["training"]):
            if (segment["start"], segment["stop"], segment["updates"]) != (432 * i, 432 * (i + 1), 432):
                raise ValueError("Training segment differs")
            expected_steps = {str(t) for t in (1, 2, 87, 88, 433) if 432 * i < t <= 432 * (i + 1)}
            if set(segment["observations"]) != expected_steps:
                raise ValueError("Missing observed update")
            for observation in segment["observations"].values():
                if set(observation) != {"encoder", "head"} or any(
                        not x["finite_nonzero_gradient"] or not x["parameters_changed"] for x in observation.values()):
                    raise ValueError("Actual module update evidence missing")
            for j, name in enumerate(("bce", "rank", "hard", "total")):
                audit.numbers_equal(segment["mean_losses_by_epoch"][name], epoch_losses[i * 3:i * 3 + 3, j], run_id)
        training[run_id] = {"updates": 864, "encoder_positive_lr_updates": arm["encoder_positive_lr_updates"],
                            "parameter_count": arm["parameter_count"], "resources": arm["resources"],
                            "formal_training_seconds": arm["formal_training_seconds"],
                            "losses_by_epoch": dict(zip(("bce", "rank", "hard", "total"), epoch_losses.T.tolist())),
                            "initial_state": arm["preflight"]["initial_state_sha256"],
                            "schedule": arm["group_schedule_sha256"], "dropout_stream": arm["dropout_stream"]}
        for epoch in ("3", "6"):
            point = evaluation["runs"][run_id]["points"][epoch]
            if point != collected["runs"][run_id]["points"][epoch]:
                raise ValueError("Final report changed collected metrics/counts")
            matrix = audit.array(job / "evaluation", point["file"], 60, 22, np.float64)
            matrices[run_id, epoch] = matrix
            for scope, index in {"mean": np.arange(60), **rows}.items():
                expected = dict(zip(COLUMNS, matrix[index].mean(0).tolist()))
                audit.mapping(point["mean"] if scope == "mean" else point["by_domain"][scope], expected, run_id)
                metric_rows.extend({"run_id": run_id, "epoch": epoch, "scope": scope, "metric": k, "value": v}
                                   for k, v in expected.items())
            counts = counts_array(point["counts_at_logit_zero"])
            apoint = arm["points"][epoch]
            if (not apoint["full_model_and_adam_reloaded"] or not apoint["model"]["actual_reload_verified"]
                    or apoint["metadata"]["completed_updates"] != 144 * int(epoch)):
                raise ValueError("Actual checkpoint restoration record missing")
            score = audit.array(arm_root, apoint["scores"]["development"], 60, 378, np.float32)
            audit_counts(audit, counts, score, matrix, 0., run_id + "/valid")
            for scope, index in {"pooled": np.arange(60), **rows}.items():
                expected = {k: v for k, v in confusion(counts[index]).items() if k in RATE_KEYS}
                audit.mapping(point["fixed_classification"]["pooled"] if scope == "pooled"
                              else point["fixed_classification"]["by_domain"][scope], expected, run_id)
            for role, n in (("fit", 144), ("calibration", 36)):
                saved = apoint["train_metrics"][role]
                train_matrix = audit.array(arm_root, saved["file"], n, 22, np.float64)
                train_scores = audit.array(arm_root, apoint["scores"][role], n, 378, np.float32)
                audit_counts(audit, counts_array(saved["counts"]), train_scores, train_matrix, 0., run_id + "/" + role)
                trajectories.append({"run_id": run_id, "epoch": epoch, "split": role,
                                     **dict(zip(COLUMNS, train_matrix.mean(0).tolist()))})
            trajectories.append({"run_id": run_id, "epoch": epoch, "split": "valid", **point["mean"]})
        automatic = evaluation["runs"][run_id]["automatic_classification"]
        counts = counts_array(automatic["counts_by_group"])
        auto_counts[run_id] = counts
        calibration = read(audit.file(arm_root, arm["calibration"]))
        if (automatic["threshold"] != calibration["threshold"]
                or calibration["model_state_sha256"] != arm["points"]["6"]["model_state_sha256"]
                or calibration["score_sha256"] != arm["points"]["6"]["scores"]["calibration"]["sha256"]):
            raise ValueError("Calibration model/score binding differs")
        score = audit.array(arm_root, arm["points"]["6"]["scores"]["development"], 60, 378, np.float32)
        audit_counts(audit, counts, score, None, automatic["threshold"], run_id + "/calibrated valid")
        calibration_counts = counts_array(calibration["counts_by_group"])
        score = audit.array(arm_root, arm["points"]["6"]["scores"]["calibration"], 36, 378, np.float32)
        audit_counts(audit, calibration_counts, score, None, calibration["threshold"], run_id + "/calibration")
        for scope, index in {"pooled": np.arange(60), **rows}.items():
            actual = automatic["pooled"] if scope == "pooled" else automatic["by_domain"][scope]
            expected = {k: v for k, v in confusion(counts[index]).items() if k in RATE_KEYS}
            expected["conditional_95pct_intervals"] = {
                k: interval(v) for k, v in ratios(weights[:, index] @ counts[index]).items()}
            audit.mapping(actual, expected, run_id + "/automatic")
        pass_domains = [automatic["by_domain"][d]["fp"] * 1000 <= 7160
                        and automatic["by_domain"][d]["tp"] * 2 >= 400 for d in "ABC"]
        if (pass_domains != [automatic["by_domain"][d]["passes_point_gates"] for d in "ABC"]
                or all(pass_domains) != automatic["all_domains_pass"]):
            raise ValueError("Automatic diagnostic gate differs")
    for seed in SEEDS:
        for key in ("initial_state", "schedule", "dropout_stream", "parameter_count"):
            if len({training[f"{seed}_{a}"][key] for a in ARMS}) != 1:
                raise ValueError("Paired initialization, randomness or capacity differs")
    comparisons, primary_boot = {}, None
    for candidate, reference in COMPARISONS:
        name = candidate + "_minus_" + reference
        deltas = np.stack([matrices[f"{s}_{candidate}", "6"] - matrices[f"{s}_{reference}", "6"] for s in SEEDS])
        average = deltas.mean(0)
        bootstrap = weights @ average / 60
        if name == "hard_minus_d":
            primary_boot = bootstrap
        low, high = np.asarray(interval(bootstrap))
        comparisons[name] = {}
        for j, metric in enumerate(COLUMNS):
            expected = {"mean": float(average[:, j].mean()), "per_seed": deltas[:, :, j].mean(1).tolist(),
                        "by_domain": {d: float(average[index, j].mean()) for d, index in rows.items()},
                        "conditional_95pct_interval": [float(low[j]), float(high[j])]}
            audit.mapping(evaluation["comparisons"][name]["metrics"][metric], expected, name)
            comparisons[name][metric] = {"reference_mean": float(np.stack([matrices[f"{s}_{reference}", "6"] for s in SEEDS])[:, :, j].mean()),
                                         "candidate_mean": float(np.stack([matrices[f"{s}_{candidate}", "6"] for s in SEEDS])[:, :, j].mean()), **expected}
            for i, seed in enumerate(SEEDS):
                audit.mapping(evaluation["per_seed_comparisons"][name][seed]["metrics"][metric],
                              {"mean": float(deltas[i, :, j].mean()),
                               "conditional_95pct_interval": interval(weights @ deltas[i, :, j] / 60)}, name + seed)
        for seed in SEEDS:
            for scope, index in {"pooled": np.arange(60), **rows}.items():
                cand, ref = auto_counts[f"{seed}_{candidate}"][index], auto_counts[f"{seed}_{reference}"][index]
                bc, br = ratios(weights[:, index] @ cand), ratios(weights[:, index] @ ref)
                pc, pr = confusion(cand), confusion(ref)
                for metric in ("fpr", "recall", "precision"):
                    audit.mapping(evaluation["per_seed_comparisons"][name][seed]["automatic"][metric][scope],
                                  {"difference": pc[metric] - pr[metric],
                                   "conditional_95pct_interval": interval(bc[metric] - br[metric])}, name + seed)
    primary = comparisons["hard_minus_d"]
    m, r = primary["map"], primary["recall_at_5"]
    checks = {"map_minimum_observed_gain": m["mean"] >= .01,
              "map_interval_above_zero": m["conditional_95pct_interval"][0] > 0,
              "map_improves_each_seed": all(x > 0 for x in m["per_seed"]),
              "recall_at_5_mean_strictly_improves": r["mean"] > 0,
              "recall_at_5_s0_strictly_improves": r["per_seed"][0] > 0}
    for metric, sign in (("average_precision", 1), ("roc_auc", 1), ("brier", -1), ("log_loss", -1)):
        checks[metric + "_mean_non_degradation"] = sign * primary[metric]["mean"] >= 0
        checks[metric + "_s0_non_degradation"] = sign * primary[metric]["per_seed"][0] >= 0
    if (checks != evaluation["acceptance"]["checks"]
            or all(checks.values()) != evaluation["acceptance"]["passed"]
            or set(k for k, v in checks.items() if not v) != set(evaluation["acceptance"]["failed"])):
        raise ValueError("Prospective acceptance differs")
    started = dt.datetime.fromisoformat((job / "started.txt").read_text().strip())
    finished = dt.datetime.fromisoformat((job / "finished.txt").read_text().strip())
    result = {"status": "PASS_SAVED_RESULT_AUDIT", "verified_at": dt.datetime.now().astimezone().isoformat(),
              "started": started.isoformat(), "finished": finished.isoformat(), "wall_seconds": (finished - started).total_seconds(),
              "remaining_seconds": 0, "original_estimate_hours": [14, 18], "budget": complete["budget"],
              "checks": {"numerical_values": audit.numbers, "maximum_absolute_difference": audit.max_difference,
                         "loss_values_checked": 7776, "maximum_loss_roundoff": maximum_loss_roundoff,
                         "blind_count_rows": audit.blind_count_rows, "saved_metric_matrices": 54,
                         "verified_small_files": len(audit.files), "formal_label_reads": 0, "formal_text_reads": 0,
                         "model_loads": 0, "retraining_updates": 0, "bootstrap_frequency_weighted_reproduction": True},
              "verified_files": audit.files, "comparisons": comparisons, "acceptance": evaluation["acceptance"],
              "training": training, "trajectories": trajectories,
              "scope": "Independent aggregation, confusion arithmetic and bootstrap from saved outputs. No recovery of hidden labels; AP/MAP/Brier/log_loss per-group truth calculations inherit the once-only evaluation and its reviewed metric implementation. No new model replay. Conditional fixed-model/group intervals; no arm/epoch selection or causal mechanism proof."}
    out.mkdir()
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.save(out / "primary_bootstrap.npy", primary_boot, allow_pickle=False)
    comparison_rows = [{"comparison": name, "metric": metric, "reference": values["reference_mean"],
                        "candidate": values["candidate_mean"], "delta": values["mean"],
                        "lower": values["conditional_95pct_interval"][0], "upper": values["conditional_95pct_interval"][1]}
                       for name, table in comparisons.items() for metric, values in table.items()]
    for filename, fields, records in (
        ("metrics.csv", ["run_id", "epoch", "scope", "metric", "value"], metric_rows),
        ("trajectory.csv", ["run_id", "epoch", "split", *COLUMNS], trajectories),
        ("comparisons.csv", ["comparison", "metric", "reference", "candidate", "delta", "lower", "upper"], comparison_rows),
    ):
        with (out / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run only in the existing Linux py310 environment")
    result = run(args.job.resolve(), args.out.resolve())
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "primary": result["comparisons"]["hard_minus_d"], "acceptance": result["acceptance"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
