"""Linux evaluation of the frozen two-arm sensitivity, without changing training.

The approved metrics, references, shared bootstrap and label parser are retained.
Only the execution platform and pre-parse file checks differ from the old entry.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import platform

import numpy as np

import step28_continual_sensitivity as frozen
from step28_continual_sensitivity import data, distill, metrics, pilot

FROZEN_SOURCE_SHA256 = "9470896eb6d17d1886a1bd91df7ed00c9c1f5fe735246ebb4037cb11a7316053"


def verify_supervision(config: dict) -> dict:
    """Verify opaque bytes before recording a label-parse attempt; do not parse CSV."""
    root = data.ROOT / config["data_root"]
    manifest = data.read_json(root / "manifest.json")
    name = "development/supervision/pairs.csv"
    return data.record(data.verify(root / name, manifest["files"][name]), data.ROOT)


def verify_outer_job(out: Path, config: dict, record: dict) -> dict:
    job = out.parent
    if (job / "exit_status.txt").read_text().strip() != "0":
        raise ValueError("Sensitivity outer job did not exit successfully")
    start = datetime.fromisoformat((job / "started.txt").read_text().strip())
    stop = datetime.fromisoformat((job / "finished.txt").read_text().strip())
    if (start.utcoffset() is None or stop.utcoffset() is None
            or not 0 <= (stop - start).total_seconds() <= config["runtime"]["maximum_gpu_stage_seconds"] + 60):
        raise ValueError("Sensitivity outer job timing differs")
    if not (job / "resource_usage.log").is_file():
        raise ValueError("Missing sensitivity outer resource evidence")
    models = []
    for arm in frozen.ARMS:
        arm_record = data.read_json(out / record["arms"][arm]["manifest"]["path"])
        for name, model in sorted(arm_record["models"].items()):
            models.append(data.record(data.verify(out / arm / model["path"], model), out))
    if len(models) != 6:
        raise ValueError("Six retained model files are required")
    return {"elapsed_seconds": (stop - start).total_seconds(), "models": models}


def prepare(out: Path) -> tuple:
    if platform.system() != "Linux":
        raise ValueError("Use the existing Linux py310 environment")
    if data.sha256(Path(frozen.__file__)) != FROZEN_SOURCE_SHA256:
        raise ValueError("Frozen sensitivity entry changed")
    config, old_config = frozen.contract(), pilot.contract()
    groups, metadata, checked = pilot.public_inputs(old_config)
    scores, original, record = frozen.validated_stage(out, config, groups, metadata, checked)
    references = {
        "sequential": original,
        "random_er": distill.reference_baseline(config, groups, metadata, checked),
        "original_distillation": frozen.reference_distillation(config, groups, metadata, checked),
    }
    checks = {"outer_job": verify_outer_job(out, config, record),
              "supervision_file": verify_supervision(old_config),
              "evaluation_source": data.record(Path(__file__), data.ROOT),
              "frozen_source_sha256": FROZEN_SOURCE_SHA256}
    return config, old_config, groups, scores, references, record, checks


def evaluate(out: Path, destination: Path, *, collect_only: bool = False) -> dict:
    if destination.exists():
        raise ValueError("Use a new evaluation directory; no automatic retry")
    config, old_config, groups, scores, references, record, checks = prepare(out)
    original = references["sequential"]
    destination.mkdir(parents=True)
    data.write_json(destination / "precheck.json", checks)
    data.write_json(destination / "access.json", {
        "status": "BOTH_ARMS_COMPLETE_GATE_PASSED_VALID_PARSE_STARTING",
        "development_parse_attempts": 1, "train": 0, "heldout": 0, "owners": 0})
    try:
        labelled = pilot.attach_labels(groups["development"], old_config, "development")
        truth = np.asarray([group.labels for group in labelled], dtype=np.uint8)
        evaluation = config["evaluation"]
        draws = np.random.default_rng(evaluation["bootstrap_seed"]).integers(
            0, 20, size=(evaluation["bootstrap_replicates"], 3, 20))
        result = {
            "status": "BOTH_SENSITIVITIES_VALID_EVALUATED_INTERPRETATION_REQUIRED",
            "config": config, "score_manifest_sha256": data.sha256(out / "manifest.json"),
            "columns": list(metrics.COLUMNS), "arms": {},
            "label_parses": {"train": 0, "development": 1, "heldout": 0, "owners": 0},
            "formal_training_seconds": record["formal_training_seconds"],
            "bootstrap": {"replicates": 5000, "seed": 20260910, "confidence_level": .95,
                "unit": "20 groups per underlying domain; same draws across both arms/references/times/orders"},
            "interpretation": "Finite two-point sensitivity on reused valid; conditional unadjusted intervals. No optimum, unique cause, automatic winner or further search."}
        shared_differences = {}
        for arm in frozen.ARMS:
            arrays, points = {}, {}
            (destination / arm).mkdir()
            for name in frozen.NAMES:
                rows = [metrics.classification(y, x) for y, x in zip(truth, scores[arm][name])]
                matrix = np.column_stack((np.asarray([[row[k] for k in metrics.CLASS_KEYS] for row in rows]),
                    metrics.retrieval(truth, scores[arm][name], 28)))
                arrays[name] = matrix
                path = destination / arm / f"{name}_metrics.npy"
                np.save(path, matrix, allow_pickle=False)
                points[name] = {
                    "file": data.record(path, destination), "confusion": [row["confusion"] for row in rows],
                    "by_domain": {d: dict(zip(metrics.COLUMNS, matrix[i*20:(i+1)*20].mean(0).tolist()))
                                  for i, d in enumerate("ABC")}}
                if name.endswith("_shared"):
                    delta = np.abs(matrix - original[name])
                    shared_differences[f"{arm}/{name}"] = {
                        "exact": bool(np.array_equal(matrix, original[name])),
                        "maximum_absolute_difference": float(delta.max()),
                        "different_values": int(np.count_nonzero(delta)),
                        "by_column": {column: {"maximum_absolute_difference": float(delta[:, i].max()),
                                               "different_values": int(np.count_nonzero(delta[:, i]))}
                                      for i, column in enumerate(metrics.COLUMNS)}}
            result["arms"][arm] = {
                "weights": frozen.WEIGHTS[arm], "points": points,
                "comparisons": {} if collect_only else {
                    role: frozen.compare(arrays, reference, draws, config, role)
                    for role, reference in references.items()}}
        data.write_json(destination / "shared_differences.json", shared_differences)
        if collect_only:
            result["status"] = "METRICS_COLLECTED_SHARED_RECONCILIATION_REQUIRED"
            result["interpretation"] = "Intermediate metrics only. Shared-point differences need independent review before comparisons or conclusions. No label retry."
            data.write_json(destination / "metrics.json", result)
            return result
        if not all(value["exact"] for value in shared_differences.values()):
            raise ValueError("Shared metrics differ; saved matrices retained; no retry")
        data.write_json(destination / "evaluation.json", result)
        return result
    except Exception as error:
        data.write_json(destination / "failure.json", {
            "status": "EVALUATION_FAILED_NO_RETRY", "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "evaluate", "collect"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    if args.action == "check":
        *_, checks = prepare(args.out.resolve())
        result = {"status": "COMPLETE_GATE_AND_SUPERVISION_FILE_VERIFIED_NO_LABEL_PARSE", **checks}
    else:
        if args.evaluation is None:
            parser.error("evaluate requires --evaluation")
        result = evaluate(args.out.resolve(), args.evaluation.resolve(), collect_only=args.action == "collect")
    print(data.json_bytes({"status": result["status"]}).decode())


if __name__ == "__main__":
    main()
