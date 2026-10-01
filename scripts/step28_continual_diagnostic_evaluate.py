"""Validate all nine diagnostic points before the authorized single valid parse."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import step28_continual_diagnostic as diag
import step28_continual_population_data as data
import step28_continual_population_evaluate as metrics


def validate(run: Path, config: dict, archive: diag.Archive) -> tuple[dict, dict]:
    if (run / "failure.json").exists():
        raise ValueError("Failed diagnostic cannot be evaluated")
    manifest = data.read_json(run / "manifest.json")
    if (manifest["status"] != diag.COMPLETE or manifest["config"] != config
            or manifest["source_files"] != diag.sources()
            or manifest["physical_updates"] != 1620
            or manifest["label_reads"] != {"train": 1, "development": 0, "heldout": 0, "owners": 0, "old_assets": 0}
            or manifest["torch_contracts"] != {"passed": 2, "skipped": 0, "failed": 0}
            or manifest["intermediate_states_removed_after_reload"] is not True):
        raise ValueError("Incomplete, stale or unauthorized diagnostic")
    for key in ("file_count", "total_size_bytes", "content_sha256"):
        if manifest["pretrained_model"][key] != config["model"][key]:
            raise ValueError("Original model provenance differs")
    if (manifest["budget"]["elapsed_seconds"] > config["runtime"]["maximum_gpu_stage_seconds"]
            or manifest["budget"]["peak_observed_bytes"] > config["runtime"]["maximum_output_bytes"]):
        raise ValueError("Diagnostic resource budget exceeded")
    expected_inputs = {**archive.checked,
        "train/supervision/pairs.csv": archive.manifest["files"]["train/supervision/pairs.csv"]}
    if manifest["verified_input_files"] != expected_inputs:
        raise ValueError("Unexpected/missing diagnostic input provenance")
    if manifest["group_ids"] != {s: [g.uid for g in archive.groups(s)] for s in diag.SPLITS}:
        raise ValueError("Fixed evaluation group mapping differs")
    if [[r["anchor"], r["target"]] for r in manifest["rotations"]] != config["rotations"]:
        raise ValueError("Rotation set/order differs")
    if len({r["initial_model_sha256"] for r in manifest["rotations"]}) != 1:
        raise ValueError("Rotations do not share original initialization")
    scores = {}
    train_rows = diag.domain_rows(archive, "train")
    for rotation in manifest["rotations"]:
        anchor, target = rotation["anchor"], rotation["target"]
        if set(rotation["points"]) != set(diag.POINTS) or set(rotation["training"]) != set(diag.POINTS):
            raise ValueError("Missing/extra trained point")
        starts = rotation["branch_starts"]
        if (set(starts) != set(diag.ARMS) or starts["same"] != starts["cross"]
                or starts["same"]["model"] != rotation["points"]["shared"]["model_state_sha256"]):
            raise ValueError("Paired model/Adam branch starts differ")
        scores[anchor] = {}
        for arm in diag.POINTS:
            domain = target if arm == "cross" else anchor
            phase = "shared" if arm == "shared" else "branch"
            seed = diag.segment_seed(config, anchor, phase)
            order = diag.positions(60, 3, seed)
            current = archive.groups("train", domain)
            log = rotation["training"][arm]
            if (log["updates"] != 180 or log["position_order"] != order
                    or log["group_ids"] != [current[i].uid for i in order]
                    or log["dropout_stream"] != seed or log["adam_step"] != (180 if arm == "shared" else 360)
                    or len(log["online_bce_per_epoch"]) != 3
                    or not np.isfinite(log["online_bce_per_epoch"]).all()
                    or not np.isfinite(log["training_seconds"]) or log["training_seconds"] < 0):
                raise ValueError("Training domain, update schedule or Adam count differs")
            for name in ("encoder", "head"):
                if (log["module_state_before"][name] == log["module_state_after"][name]
                        or not 0 < log["first_update_clipped_gradient_norms"][name] < float("inf")):
                    raise ValueError("Missing actual encoder/head learning evidence")
            point = rotation["points"][arm]
            if point["full_model_and_adam_reloaded"] is not True or set(point["scores"]) != set(diag.SPLITS):
                raise ValueError("Missing complete train/valid state replay")
            for split in diag.SPLITS:
                record = point["scores"][split]
                if record["path"] != f"scores/{anchor}_{arm}_{split}.npy":
                    raise ValueError("Score-to-point mapping differs")
                values = np.load(data.verify(run / record["path"], record), allow_pickle=False)
                if (values.shape != (len(archive.groups(split)), 378) or values.dtype != np.dtype("float32")
                        or not np.isfinite(values).all()):
                    raise ValueError("Incomplete/nonfinite score array")
                if split == "development":
                    scores[anchor][arm] = values
            train = point["train_metrics"]
            rec = train["per_group_metrics"]
            if rec["path"] != f"metrics/{anchor}_{arm}_train.npy" or train["columns"] != list(metrics.COLUMNS):
                raise ValueError("Train metric mapping/schema differs")
            matrix = np.load(data.verify(run / rec["path"], rec), allow_pickle=False)
            if matrix.shape != (180, len(metrics.COLUMNS)) or not np.isfinite(matrix).all():
                raise ValueError("Invalid saved train diagnostics")
            if {k: train[k] for k in ("by_domain", "domain_equal")} != diag.summarize(matrix, train_rows):
                raise ValueError("Saved train mean diagnostics differ")
            counts = train["per_group_confusion"]
            if len(counts) != 180:
                raise ValueError("Incomplete train confusion records")
            for row, count in zip(matrix, counts, strict=True):
                if (set(count) != {"tp", "fp", "fn", "tn"}
                        or any(type(v) is not int or v < 0 for v in count.values())
                        or count["tp"] + count["fn"] != 20 or count["fp"] + count["tn"] != 358):
                    raise ValueError("Malformed train confusion counts")
                for name, value in metrics.confusion_metrics(**count).items():
                    if abs(row[metrics.COLUMNS.index(name)] - value) > 1e-12:
                        raise ValueError("Train metric/count disagreement")
    return manifest, scores


def evaluate(run: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Use one new evaluation directory; do not repeat truth access")
    config = diag.policy()
    archive = diag.Archive(config, train_labels=False)
    manifest, scores = validate(run, config, archive)
    output.mkdir(parents=True)
    data.write_json(output / "access.json", {"status": "NINE_POINT_GATES_PASSED_BEFORE_VALID_PARSE",
        "manifest_sha256": data.sha256(run / "manifest.json"), "train_label_parses": 0, "test_access": 0})
    path = data.verify(archive.root / "development/supervision/pairs.csv",
                       archive.manifest["files"]["development/supervision/pairs.csv"])
    labels = diag.read_labels(path, archive.groups("development"))
    rows = diag.domain_rows(archive, "development")
    values, points = {}, {}
    for anchor, _ in config["rotations"]:
        values[anchor], points[anchor] = {}, {}
        for arm in diag.POINTS:
            matrix, confusion = metrics.group_metrics(labels, scores[anchor][arm])
            values[anchor][arm] = matrix
            path = output / f"{anchor}_{arm}_valid.npy"
            np.save(path, matrix, allow_pickle=False)
            points[anchor][arm] = {**diag.summarize(matrix, rows),
                "per_group_metrics": data.record(path, output), "per_group_confusion": confusion}
    comparison = diag.comparisons(values, rows, config)
    result = {"status": "DIAGNOSTIC_VALID_COMPLETE_TEST_UNTOUCHED", "config": config,
        "run_manifest_sha256": data.sha256(run / "manifest.json"), "columns": list(metrics.COLUMNS),
        "valid_group_ids": [g.uid for g in archive.groups("development")], "points": points,
        "comparisons": comparison, "decision": diag.decision(comparison, config["evaluation"]["minimum_matching_directions"]),
        "label_reads": {"development": 1, "train": 0, "heldout": 0, "owners": 0, "old_assets": 0},
        "change_orientation": "after-minus-before; brier/log_loss improve when negative; no metric-count voting",
        "interval_scope": "Descriptive paired complete-group bootstrap on previously observed valid; no claim of confirmatory testing or all training randomness",
        "train_metrics_source": "Linux in-memory authorized train labels; saved metrics/counts validated without train reparse"}
    data.write_json(output / "evaluation.json", result)
    return {"status": result["status"], "decision": result["decision"], "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(data.json_bytes(evaluate(args.run.resolve(), args.out.resolve())).decode())


if __name__ == "__main__":
    main()
