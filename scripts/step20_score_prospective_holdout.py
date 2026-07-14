#!/usr/bin/env python3
"""Score a frozen prospective holdout without reading its review labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np

import step9_run_few_shot_adaptation as step9
import step15_v7_common as common


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step20_prospective_holdout_policy.json"
V7_POLICY = ROOT / "schema" / "step15_v7_two_stage_policy.json"


def render_csv(rows: list[dict], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen prospective scores: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--v7-policy", default=str(V7_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path = common.resolve(args.policy)
    v7_policy_path = common.resolve(args.v7_policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    v7_policy = json.loads(v7_policy_path.read_text(encoding="utf-8"))
    outputs = {key: common.resolve(value) for key, value in policy["outputs"].items()}
    model_freeze_path = common.resolve(policy["model_freeze_manifest"])
    if args.validate_config_only:
        expected = set(policy["evaluation"]["models"])
        required = {
            "raw_clean_e5_cosine",
            "step9_v7_source_only_clean_fusion",
            "step9_v7_no_augmentation",
            "step9_v7_equal_effective_weight_duplication",
            "step9_v7_latent_pair_embedding_mixup",
            "step15_v7_clean_selected",
            "step15_v7_two_stage_veto",
        }
        if expected != required:
            raise ValueError("Prospective evaluation model allow-list changed")
        print(json.dumps({"status": "pass", "models": sorted(expected)}, indent=2))
        return
    required_inputs = (
        outputs["frozen_pair_universe"],
        outputs["freeze_manifest"],
        outputs["prospective_pair_features"],
        outputs["prospective_feature_manifest"],
        model_freeze_path,
    )
    for path in required_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen prospective scoring input: {path}")
    if outputs["evaluation_lock"].exists():
        raise FileExistsError("Prospective evaluation has already started; scoring is sealed")
    if outputs["frozen_model_scores"].exists() or outputs["frozen_score_manifest"].exists():
        raise FileExistsError("Prospective model scores are already frozen")

    freeze_manifest = json.loads(outputs["freeze_manifest"].read_text(encoding="utf-8"))
    model_freeze = json.loads(model_freeze_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(
        outputs["prospective_feature_manifest"].read_text(encoding="utf-8")
    )
    if freeze_manifest["model_freeze_manifest_sha256"] != common.sha256(model_freeze_path):
        raise ValueError("Prospective labels were not frozen against this model manifest")
    if feature_manifest["model_freeze_manifest_sha256"] != common.sha256(model_freeze_path):
        raise ValueError("Prospective features were not transformed against this model manifest")
    if feature_manifest["frozen_pair_universe_sha256"] != common.sha256(
        outputs["frozen_pair_universe"]
    ):
        raise ValueError("Prospective features and frozen pair universe disagree")
    if freeze_manifest["frozen_pair_universe_file_sha256"] != common.sha256(
        outputs["frozen_pair_universe"]
    ):
        raise ValueError("Prospective pair universe changed after review freeze")
    if model_freeze.get("current_internal_test_used_for_model_selection") is not False:
        raise ValueError("Prospective scorer refuses a test-informed model freeze")
    current_v7_policy_sha256 = common.sha256(v7_policy_path)
    if model_freeze.get("inputs", {}).get("v7_policy") != current_v7_policy_sha256:
        raise ValueError("Prospective scorer v7 policy differs from the frozen model policy")
    if feature_manifest.get("v7_policy_sha256") != current_v7_policy_sha256:
        raise ValueError("Prospective feature bundle used a different v7 policy")

    # This file physically contains only pair identity/endpoints and collection provenance.
    # Review labels and evidence types remain sealed for the one-time evaluation.
    label_rows = common.load_csv(outputs["frozen_pair_universe"])
    pair_identity = {
        row["pair_uid"]: {
            "pair_uid": row["pair_uid"],
            "seller_uid_left": row["seller_uid_left"],
            "seller_uid_right": row["seller_uid_right"],
        }
        for row in label_rows
    }
    if len(pair_identity) != len(label_rows):
        raise ValueError("Prospective frozen pair universe contains duplicate pair UIDs")
    feature_rows = common.load_csv(outputs["prospective_pair_features"])
    feature_index = {row["pair_uid"]: row for row in feature_rows}
    if len(feature_index) != len(feature_rows) or set(feature_index) != set(pair_identity):
        raise ValueError("Prospective score feature universe differs from the frozen label universe")
    pair_order = sorted(pair_identity)
    rows = []
    for pair_uid in pair_order:
        feature = feature_index[pair_uid]
        identity = pair_identity[pair_uid]
        if (
            feature.get("seller_uid_left") != identity["seller_uid_left"]
            or feature.get("seller_uid_right") != identity["seller_uid_right"]
        ):
            raise ValueError(f"Prospective endpoint mismatch: {pair_uid}")
        rows.append(feature)

    stable_features = list(v7_policy["inductive_features"]["stable_strict_clean_features"])
    clean_raw = common.strict_clean_matrix(rows, stable_features)
    prospective_pool_cfg = {
        "clean_e5_cache_metadata": policy["prospective_upstream"][
            "clean_e5_cache_metadata"
        ],
        "clean_e5_cache_matrix": policy["prospective_upstream"]["clean_e5_cache_matrix"],
        "item_identity_signal_sources": [
            policy["item_identity_signals"],
            policy["prospective_upstream"]["item_identity_signals"],
        ],
        "identifier_frequency_reference_sellers": model_freeze[
            "identifier_frequency_reference_sellers"
        ],
    }
    latent = common.projected_pair_latents(
        rows, prospective_pool_cfg, v7_policy["latent_pair_representation"]
    )
    thresholds = model_freeze["thresholds_from_representative_validation"]
    seed_ids = [int(value) for value in model_freeze["seed_ids"]]
    experiment_scores = {}
    experiment_seed_scores = {}
    for experiment in v7_policy["step9_latent_mixup"]["experiments"]:
        artifact_records = model_freeze["step9_artifacts"][experiment]
        by_seed = {}
        for record in artifact_records:
            artifact_path = common.resolve(record["path"])
            if common.sha256(artifact_path) != record["sha256"]:
                raise ValueError(f"Frozen Step9-v7 artifact changed: {artifact_path}")
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if artifact["experiment"] != experiment or float(artifact["support_ratio"]) != 1.0:
                raise ValueError(f"Unexpected frozen Step9-v7 artifact identity: {artifact_path}")
            expected_feature_names = stable_features + [
                f"e5_pair_latent_{index:03d}" for index in range(latent.shape[1])
            ]
            if artifact.get("feature_names") != expected_feature_names:
                raise ValueError(f"Frozen Step9-v7 artifact feature schema changed: {artifact_path}")
            seed = int(artifact["seed"])
            clean = common.apply_imputation(clean_raw, artifact["clean_feature_imputation"])
            matrix = np.hstack([clean, latent])
            by_seed[seed] = step9.apply_logistic_artifact_to_matrix(
                matrix, artifact["logistic"]
            )
        if sorted(by_seed) != sorted(seed_ids):
            raise ValueError(f"Frozen Step9-v7 artifact seeds are incomplete for {experiment}")
        matrix = np.vstack([by_seed[seed] for seed in seed_ids])
        experiment_seed_scores[experiment] = matrix
        experiment_scores[experiment] = np.mean(matrix, axis=0)

    source_only_by_seed = {}
    for record in model_freeze["step9_source_only_artifacts"]:
        artifact_path = common.resolve(record["path"])
        if common.sha256(artifact_path) != record["sha256"]:
            raise ValueError(f"Frozen Step9-v7 source-only artifact changed: {artifact_path}")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact["experiment"] != "no_augmentation" or abs(
            float(artifact["support_ratio"])
        ) > 1e-12:
            raise ValueError(f"Unexpected source-only artifact identity: {artifact_path}")
        expected_feature_names = stable_features + [
            f"e5_pair_latent_{index:03d}" for index in range(latent.shape[1])
        ]
        if artifact.get("feature_names") != expected_feature_names:
            raise ValueError(f"Frozen source-only feature schema changed: {artifact_path}")
        seed = int(artifact["seed"])
        clean = common.apply_imputation(clean_raw, artifact["clean_feature_imputation"])
        matrix = np.hstack([clean, latent])
        source_only_by_seed[seed] = step9.apply_logistic_artifact_to_matrix(
            matrix, artifact["logistic"]
        )
    if sorted(source_only_by_seed) != sorted(seed_ids):
        raise ValueError("Frozen Step9-v7 source-only artifact seeds are incomplete")
    source_only_matrix = np.vstack([source_only_by_seed[seed] for seed in seed_ids])
    if not np.allclose(source_only_matrix, source_only_matrix[0][None, :], rtol=0.0, atol=1e-12):
        raise ValueError("Frozen deterministic source-only artifacts disagree on prospective data")
    source_only_scores = np.mean(source_only_matrix, axis=0)

    selected_experiment = model_freeze["selected_clean_experiment"]
    selected_clean = experiment_scores[selected_experiment]
    vetoed, reliability_decisions, reliability_diagnostics = common.apply_reliability_veto(
        rows,
        selected_clean,
        prospective_pool_cfg,
        model_freeze["step15_stage_b_policy"],
    )
    scores = {
        "raw_clean_e5_cosine": np.asarray(
            [
                float(row["embedding_cosine_multilingual_e5_large_identifier_redacted"])
                for row in rows
            ]
        ),
        "step9_v7_source_only_clean_fusion": source_only_scores,
        "step9_v7_no_augmentation": experiment_scores["no_augmentation"],
        "step9_v7_equal_effective_weight_duplication": experiment_scores[
            "equal_effective_weight_duplication"
        ],
        "step9_v7_latent_pair_embedding_mixup": experiment_scores[
            "latent_pair_embedding_mixup"
        ],
        "step15_v7_clean_selected": selected_clean,
        "step15_v7_two_stage_veto": vetoed,
    }
    threshold_key = {
        "raw_clean_e5_cosine": "raw_clean_e5_cosine",
        "step9_v7_source_only_clean_fusion": "step9_v7_source_only_clean_fusion",
        "step9_v7_no_augmentation": "no_augmentation",
        "step9_v7_equal_effective_weight_duplication": "equal_effective_weight_duplication",
        "step9_v7_latent_pair_embedding_mixup": "latent_pair_embedding_mixup",
        "step15_v7_clean_selected": "step15_v7_clean_selected",
        "step15_v7_two_stage_veto": "step15_v7_two_stage_veto",
    }
    model_allowlist = list(policy["evaluation"]["models"])
    output_rows = []
    for index, pair_uid in enumerate(pair_order):
        decision = reliability_decisions[index]
        row = {
            "pair_uid": pair_uid,
            "reliability_decision": decision["decision"],
            "reliability_score_multiplier": f"{float(decision['score_multiplier']):.6f}",
            "strong_direct_token_count": decision["strong_direct_token_count"],
            "public_noise_token_count": decision["public_noise_token_count"],
            "ambiguous_token_count": decision["ambiguous_token_count"],
        }
        for model_id in model_allowlist:
            row[f"{model_id}__score"] = f"{float(scores[model_id][index]):.12f}"
            row[f"{model_id}__threshold"] = f"{float(thresholds[threshold_key[model_id]]):.12f}"
        output_rows.append(row)
    fields = list(output_rows[0])
    score_payload = render_csv(output_rows, fields)
    score_manifest = {
        "step": "step20_score_prospective_holdout",
        "version": policy["version"],
        "row_count": len(output_rows),
        "model_allowlist": model_allowlist,
        "labels_or_evidence_types_read_for_scoring": False,
        "pair_identity_and_endpoints_read_for_scoring": True,
        "threshold_source": "frozen_representative_validation_only",
        "selected_clean_experiment": selected_experiment,
        "reliability_decision_counts": reliability_diagnostics["decision_counts"],
        "frozen_scores_sha256": common.canonical_hash(output_rows),
        "frozen_scores_file_sha256": hashlib.sha256(score_payload).hexdigest(),
        "frozen_pair_universe_sha256": common.sha256(outputs["frozen_pair_universe"]),
        "prospective_feature_manifest_sha256": common.sha256(
            outputs["prospective_feature_manifest"]
        ),
        "model_freeze_manifest_sha256": common.sha256(model_freeze_path),
        "policy_sha256": common.sha256(policy_path),
        "producer_sha256": common.sha256(Path(__file__).resolve()),
    }
    score_manifest["manifest_sha256"] = common.canonical_hash(score_manifest)
    score_root = outputs["frozen_score_manifest"].parent
    managed_outputs = [outputs["frozen_model_scores"], outputs["frozen_score_manifest"]]
    if any(path.parent != score_root for path in managed_outputs):
        raise ValueError("Prospective score outputs must share one publication directory")
    staging_root = score_root.with_name(f".{score_root.name}.incomplete")
    if score_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Prospective score final or incomplete directory exists: {score_root} / {staging_root}"
        )

    def staged(final_path: Path) -> Path:
        return staging_root / final_path.relative_to(score_root)

    write_new(staged(outputs["frozen_model_scores"]), score_payload)
    write_new(
        staged(outputs["frozen_score_manifest"]),
        (json.dumps(score_manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    staging_root.replace(score_root)
    print(
        json.dumps(
            {
                "status": "scores_frozen_labels_still_unread",
                "row_count": len(output_rows),
                "selected_clean_experiment": selected_experiment,
                "reliability_decision_counts": dict(
                    sorted(Counter(item["decision"] for item in reliability_decisions).items())
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
