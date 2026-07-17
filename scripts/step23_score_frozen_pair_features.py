#!/usr/bin/env python3
"""Score prebuilt Step23 pair features with a frozen all-train artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

import step9_run_few_shot_adaptation as step9
import step15_v7_common as common


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step23_item_multi_instance_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_immutable(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Step23 frozen scoring produced no rows")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    rendered = ("\ufeff" + buffer.getvalue()).encode("utf-8")
    if path.exists() and path.read_bytes() != rendered:
        raise ValueError(f"Refusing to overwrite different frozen Step23 scores: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered)


def write_json_immutable(path: Path, payload: dict) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite different frozen Step23 score metadata: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--allow-ineligible-diagnostic", action="store_true")
    args = parser.parse_args()

    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    output_root = resolve(policy["outputs_root"])
    artifact_path = output_root / policy["outputs"]["model_artifacts"]
    artifacts = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifacts.get("policy_version") != policy["version"]:
        raise ValueError("Frozen Step23 artifact policy version mismatch")
    if artifacts.get("valid_or_test_scores_used") is not False:
        raise ValueError("Frozen Step23 artifact was not produced under the no-test contract")
    if (
        policy["evaluation"].get("frozen_scorer_requires_promotion", True)
        and artifacts.get("promotion_eligible") is not True
        and not args.allow_ineligible_diagnostic
    ):
        raise ValueError(
            "Frozen Step23 scorer is blocked because the internal promotion gate failed"
        )
    model_name = args.model or policy["evaluation"]["primary_model"]
    final_artifacts = artifacts["final_all_train_artifacts"]
    if model_name not in final_artifacts:
        raise ValueError(f"Frozen Step23 model is unavailable: {model_name}")
    model_artifact = final_artifacts[model_name]
    feature_names = list(model_artifact["feature_names"])
    declared_names = list(policy["evaluation"]["model_feature_sets"][model_name])
    if feature_names != declared_names:
        raise ValueError("Frozen Step23 artifact feature order drift")

    feature_path = resolve(args.features)
    rows = load_csv(feature_path)
    if not rows:
        raise ValueError("Frozen Step23 feature input is empty")
    pair_uids = [row.get("pair_uid", "") for row in rows]
    if any(not value for value in pair_uids) or len(pair_uids) != len(set(pair_uids)):
        raise ValueError("Frozen Step23 feature input has empty or duplicate pair_uid")
    missing = [name for name in feature_names if name not in rows[0]]
    if missing:
        raise ValueError(f"Frozen Step23 feature input is missing: {missing}")
    raw_matrix = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64
    )
    matrix = common.apply_imputation(raw_matrix, model_artifact["imputation"])
    scores = step9.apply_logistic_artifact_to_matrix(
        matrix, model_artifact["logistic_artifact"]
    )
    if np.any(~np.isfinite(scores)):
        raise ValueError("Frozen Step23 scorer emitted non-finite probabilities")

    output_rows = [
        {
            "pair_uid": pair_uid,
            "model_name": model_name,
            "prob_positive": f"{float(score):.12f}",
        }
        for pair_uid, score in zip(pair_uids, scores, strict=True)
    ]
    output_path = resolve(args.output_csv)
    manifest_path = resolve(args.output_manifest)
    write_csv_immutable(output_path, output_rows)
    manifest = {
        "step": "step23_frozen_pair_feature_scoring",
        "policy_version": policy["version"],
        "model_name": model_name,
        "row_count": len(output_rows),
        "labels_or_metrics_used": False,
        "ineligible_diagnostic_override": bool(args.allow_ineligible_diagnostic),
        "input_hashes": {
            "policy": sha256_file(policy_path),
            "model_artifacts": sha256_file(artifact_path),
            "features": sha256_file(feature_path),
            "producer": sha256_file(Path(__file__)),
        },
        "output_csv": output_path.relative_to(ROOT).as_posix(),
        "output_csv_sha256": sha256_file(output_path),
    }
    write_json_immutable(manifest_path, manifest)
    print(json.dumps({"status": "scored", "model": model_name, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
