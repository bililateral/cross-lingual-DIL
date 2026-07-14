#!/usr/bin/env python3
"""Transform a frozen prospective holdout with the already-fitted v7 references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import step15_build_v6_inductive_pair_features as v6
import step15_build_v7_inductive_pair_features as v7
import step15_v7_common as common


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step20_prospective_holdout_policy.json"
V7_POLICY = ROOT / "schema" / "step15_v7_two_stage_policy.json"


def write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite prospective scoring feature artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 0.0 else float(np.dot(left, right) / denominator)


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
    upstream = {key: common.resolve(value) if isinstance(value, str) else value for key, value in policy["prospective_upstream"].items()}
    outputs = {key: common.resolve(value) for key, value in policy["outputs"].items()}
    required_keys = {
        "seller_profiles",
        "item_identity_signals",
        "step4_candidates",
        "canonical_pair_features",
        "clean_e5_cache_metadata",
        "clean_e5_cache_matrix",
        "clean_e5_manifest",
    }
    if not required_keys.issubset(upstream):
        raise ValueError("Prospective upstream scoring bundle is incomplete")
    if args.validate_config_only:
        print(json.dumps({"status": "pass", "required_upstream": sorted(required_keys)}, indent=2))
        return
    for key in sorted(required_keys):
        if not Path(upstream[key]).is_file():
            raise FileNotFoundError(f"Missing prospective upstream artifact {key}: {upstream[key]}")
    if not outputs["frozen_pair_universe"].is_file() or not outputs["freeze_manifest"].is_file():
        raise FileNotFoundError("Prospective pair universe must be independently reviewed and frozen first")
    model_freeze_path = common.resolve(policy["model_freeze_manifest"])
    if not model_freeze_path.is_file():
        raise FileNotFoundError("V7 model freeze manifest is required before prospective scoring")
    model_freeze = json.loads(model_freeze_path.read_text(encoding="utf-8"))
    current_v7_policy_sha256 = common.sha256(v7_policy_path)
    if model_freeze.get("inputs", {}).get("v7_policy") != current_v7_policy_sha256:
        raise ValueError("Current v7 policy differs from the policy bound into the model freeze")

    core_clean_manifest_path = common.resolve(
        v7_policy["clean_semantic_encoder"]["manifest_output"]
    )
    prospective_clean_manifest_path = Path(upstream["clean_e5_manifest"])
    if not core_clean_manifest_path.is_file():
        raise FileNotFoundError("Frozen core identifier-redacted E5 manifest is missing")
    core_clean_manifest = json.loads(core_clean_manifest_path.read_text(encoding="utf-8"))
    prospective_clean_manifest = json.loads(
        prospective_clean_manifest_path.read_text(encoding="utf-8")
    )
    for key in ("model_key", "model_directory_fingerprint", "producer_sha256"):
        if core_clean_manifest.get(key) != prospective_clean_manifest.get(key):
            raise ValueError(f"Prospective clean E5 cache differs from the frozen core cache: {key}")
    if prospective_clean_manifest.get("v7_policy_sha256") != current_v7_policy_sha256:
        raise ValueError("Prospective clean E5 cache used a different v7 policy")

    universe_rows = common.load_csv(outputs["frozen_pair_universe"])
    universe_index = {row["pair_uid"]: row for row in universe_rows}
    if len(universe_index) != len(universe_rows):
        raise ValueError("Prospective frozen pair universe contains duplicate pair UIDs")
    freeze_manifest = json.loads(outputs["freeze_manifest"].read_text(encoding="utf-8"))
    if freeze_manifest.get("frozen_pair_universe_file_sha256") != common.sha256(
        outputs["frozen_pair_universe"]
    ):
        raise ValueError("Prospective frozen pair universe changed after review freeze")
    profiles_list = v6.load_jsonl(Path(upstream["seller_profiles"]))
    profiles = {str(row["seller_uid"]): row for row in profiles_list}
    candidates, _ = v6.load_csv(Path(upstream["step4_candidates"]))
    candidate_index = {str(row["pair_uid"]): row for row in candidates}
    canonical_rows, canonical_fields = v6.load_csv(Path(upstream["canonical_pair_features"]))
    canonical_index = {str(row["pair_uid"]): row for row in canonical_rows}
    if not set(universe_index).issubset(canonical_index) or not set(universe_index).issubset(candidate_index):
        missing = sorted(set(universe_index) - (set(canonical_index) & set(candidate_index)))
        raise ValueError(f"Prospective labeled pair lacks Step4/Step7 upstream features: {missing[:1]}")

    reference_path = common.resolve(v7_policy["inductive_features"]["reference_bundle_output"])
    reference_bundle = json.loads(reference_path.read_text(encoding="utf-8"))
    reference = reference_bundle["domains"]["zh_target_strict"]
    feature_schema_path = common.resolve(v7_policy["inductive_features"]["step7_feature_schema"])
    feature_schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
    numeric_paths = dict(feature_schema["market_relative_numeric_fields"])
    stable_features = list(v7_policy["inductive_features"]["stable_strict_clean_features"])
    oov_cfg = v7_policy["inductive_features"]["oov_policy"]

    cache_cfg = {
        "clean_e5_cache_metadata": str(upstream["clean_e5_cache_metadata"]),
        "clean_e5_cache_matrix": str(upstream["clean_e5_cache_matrix"]),
    }
    seller_index, embedding_matrix, _ = common.load_embedding_index(cache_cfg)
    output_rows = []
    clean_semantic_field = v7_policy["clean_semantic_encoder"]["output_feature"]
    for pair_uid in sorted(universe_index):
        pair_identity = universe_index[pair_uid]
        source = dict(canonical_index[pair_uid])
        left_uid = str(pair_identity["seller_uid_left"])
        right_uid = str(pair_identity["seller_uid_right"])
        if source.get("seller_uid_left") != left_uid or source.get("seller_uid_right") != right_uid:
            raise ValueError(f"Prospective pair endpoints changed between labels and features: {pair_uid}")
        if left_uid not in profiles or right_uid not in profiles:
            raise ValueError(f"Prospective seller profile missing for pair: {pair_uid}")
        if left_uid not in seller_index or right_uid not in seller_index:
            raise ValueError(f"Prospective E5 cache misses a seller for pair: {pair_uid}")
        source.update(
            v7.derive_reference_fields(
                profiles[left_uid], profiles[right_uid], reference, numeric_paths, oov_cfg
            )
        )
        candidate = candidate_index[pair_uid]
        non_identifier = str(candidate.get("candidate_rule_count_non_identifier", "")).strip()
        if not non_identifier:
            excluded = {
                "shared_contact_exact",
                "shared_pgp_fingerprint",
                "shared_pgp_fingerprint_via_aux_alias",
            }
            non_identifier = str(
                sum(
                    1
                    for rule in str(candidate.get("candidate_rule_hits", "")).split("|")
                    if rule and rule not in excluded
                )
            )
        source["candidate_rule_count_non_identifier"] = non_identifier
        source[clean_semantic_field] = round(
            cosine(
                np.asarray(embedding_matrix[seller_index[left_uid]], dtype=float),
                np.asarray(embedding_matrix[seller_index[right_uid]], dtype=float),
            ),
            12,
        )
        for name in stable_features:
            if name not in source:
                raise ValueError(f"Prospective strict-clean feature is absent: {name}")
            value = str(source.get(name, "")).strip()
            if value and not np.isfinite(float(value)):
                raise ValueError(f"Prospective strict-clean feature is non-finite: {name}")
        output_rows.append(source)

    output_fields = list(canonical_fields)
    if "candidate_rule_count_non_identifier" not in output_fields:
        output_fields.append("candidate_rule_count_non_identifier")
    if clean_semantic_field not in output_fields:
        output_fields.append(clean_semantic_field)
    for field in v7.DIAGNOSTIC_FIELDS:
        if field not in output_fields:
            output_fields.append(field)
    payload = v6.render_csv(output_rows, output_fields)
    manifest = {
        "step": "step20_build_prospective_features",
        "version": policy["version"],
        "pair_count": len(output_rows),
        "strict_clean_feature_count": len(stable_features),
        "reference_fitted_on_prospective_data": False,
        "reference_bundle": str(reference_path.relative_to(ROOT)).replace("\\", "/"),
        "reference_bundle_sha256": common.sha256(reference_path),
        "clean_semantic_feature": clean_semantic_field,
        "legacy_profile_text_semantic_features_are_diagnostic_only": True,
        "frozen_pair_universe_sha256": common.sha256(outputs["frozen_pair_universe"]),
        "freeze_manifest_sha256": common.sha256(outputs["freeze_manifest"]),
        "review_labels_or_evidence_types_read": False,
        "model_freeze_manifest_sha256": common.sha256(model_freeze_path),
        "core_clean_embedding_manifest_sha256": common.sha256(core_clean_manifest_path),
        "prospective_clean_embedding_manifest_sha256": common.sha256(
            prospective_clean_manifest_path
        ),
        "inputs": {
            key: {
                "path": str(Path(upstream[key]).relative_to(ROOT)).replace("\\", "/"),
                "sha256": common.sha256(Path(upstream[key])),
            }
            for key in sorted(required_keys)
        },
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "policy_sha256": common.sha256(policy_path),
        "v7_policy_sha256": current_v7_policy_sha256,
    }
    manifest["manifest_sha256"] = common.canonical_hash(manifest)
    manifest_payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    feature_root = outputs["prospective_feature_manifest"].parent
    managed_outputs = [outputs["prospective_pair_features"], outputs["prospective_feature_manifest"]]
    if any(path.parent != feature_root for path in managed_outputs):
        raise ValueError("Prospective feature outputs must share one publication directory")
    staging_root = feature_root.with_name(f".{feature_root.name}.incomplete")
    if feature_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Prospective feature final or incomplete directory exists: {feature_root} / {staging_root}"
        )

    def staged(final_path: Path) -> Path:
        return staging_root / final_path.relative_to(feature_root)

    write_new(staged(outputs["prospective_pair_features"]), payload)
    write_new(staged(outputs["prospective_feature_manifest"]), manifest_payload)
    staging_root.replace(feature_root)
    print(json.dumps({"status": "pass", "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
