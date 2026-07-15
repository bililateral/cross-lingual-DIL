#!/usr/bin/env python3
"""Shared contracts for the Step15-v8 bridge and contextual evidence method."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7
import step9_run_few_shot_adaptation as step9
import step15_build_v6_inductive_pair_features as v6_features
import step15_build_v7_inductive_pair_features as v7_features
import step15_v7_common as v7


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step15_v8_contextual_evidence_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_self_hashed_json(path: str | Path, hash_field: str = "manifest_sha256") -> dict:
    resolved = resolve(path)
    payload = load_json(resolved)
    expected = str(payload.get(hash_field, "")).strip()
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    observed = canonical_hash(unsigned)
    if not expected or observed != expected:
        raise ValueError(
            f"Self-hash mismatch for {resolved}: expected={expected} observed={observed}"
        )
    return payload


def verify_readiness_runtime_chain(policy: dict, v7_policy: dict) -> dict:
    refreeze = policy.get("validation_context_refreeze", {})
    freeze_path = resolve(refreeze["freeze_manifest"])
    freeze = verify_self_hashed_json(freeze_path)
    freeze_producer = resolve(freeze["producer"])
    if sha256(freeze_producer) != freeze["producer_sha256"]:
        raise ValueError(f"Readiness freeze producer changed: {freeze_producer}")
    verified_freeze_inputs = {}
    for input_path, expected in freeze.get("inputs", {}).items():
        path = resolve(input_path)
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"Readiness freeze input changed: {path}")
        verified_freeze_inputs[str(path.relative_to(ROOT)).replace("\\", "/")] = observed
    verified_freeze_outputs = {}
    for record in freeze.get("outputs", {}).values():
        path = resolve(record["path"])
        observed = sha256(path)
        if observed != record["sha256"]:
            raise ValueError(f"Readiness freeze output changed: {path}")
        verified_freeze_outputs[str(path.relative_to(ROOT)).replace("\\", "/")] = observed

    clean_manifest_path = resolve(v7_policy["clean_semantic_encoder"]["manifest_output"])
    clean_manifest = verify_self_hashed_json(clean_manifest_path)
    clean_producer = ROOT / "scripts" / "step15_build_v7_clean_embedding_cache.py"
    if sha256(clean_producer) != clean_manifest["producer_sha256"]:
        raise ValueError(f"Identifier-redacted E5 producer changed: {clean_producer}")
    generated_v7_policy_path = resolve(policy["frozen_dependencies"]["v7_policy"])
    if sha256(generated_v7_policy_path) != clean_manifest["v7_policy_sha256"]:
        raise ValueError("Identifier-redacted E5 manifest policy hash mismatch")
    verified_clean_records = {}
    for pool_name, record in clean_manifest.get("records", {}).items():
        metadata_path = resolve(record["metadata_path"])
        matrix_path = resolve(record["matrix_path"])
        if sha256(metadata_path) != record["metadata_sha256"]:
            raise ValueError(f"Identifier-redacted E5 metadata changed: {pool_name}")
        if sha256(matrix_path) != record["matrix_sha256"]:
            raise ValueError(f"Identifier-redacted E5 matrix changed: {pool_name}")
        verified_clean_records[pool_name] = {
            "metadata_path": str(metadata_path.relative_to(ROOT)).replace("\\", "/"),
            "metadata_sha256": record["metadata_sha256"],
            "matrix_path": str(matrix_path.relative_to(ROOT)).replace("\\", "/"),
            "matrix_sha256": record["matrix_sha256"],
        }

    feature_manifest_path = resolve(v7_policy["inductive_features"]["manifest_output"])
    feature_manifest = verify_self_hashed_json(feature_manifest_path)
    feature_producer = ROOT / "scripts" / "step15_build_v7_inductive_pair_features.py"
    if sha256(feature_producer) != feature_manifest["producer_sha256"]:
        raise ValueError(f"V7 feature producer changed: {feature_producer}")
    if sha256(resolve(feature_manifest["policy"])) != feature_manifest["policy_sha256"]:
        raise ValueError("V7 feature manifest policy hash mismatch")
    reference_path = resolve(feature_manifest["reference_bundle"])
    if sha256(reference_path) != feature_manifest["reference_bundle_sha256"]:
        raise ValueError("V7 feature reference bundle changed")
    verified_outputs = {}
    verified_feature_inputs = {}
    for pool_name, record in feature_manifest["domains"].items():
        output_path = resolve(record["output_path"])
        if sha256(output_path) != record["output_sha256"]:
            raise ValueError(f"V7 feature output changed: {pool_name}:{output_path}")
        for input_path, expected in record.get("inputs", {}).items():
            resolved_input = resolve(input_path)
            if sha256(resolved_input) != expected:
                raise ValueError(
                    f"V7 feature input changed: {pool_name}:{resolved_input}"
                )
            verified_feature_inputs[
                str(resolved_input.relative_to(ROOT)).replace("\\", "/")
            ] = expected
        verified_outputs[pool_name] = {
            "path": str(output_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": record["output_sha256"],
        }
    return {
        "readiness_freeze_manifest": str(freeze_path.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "readiness_freeze_manifest_sha256": sha256(freeze_path),
        "readiness_freeze_inputs": dict(sorted(verified_freeze_inputs.items())),
        "readiness_freeze_outputs": dict(sorted(verified_freeze_outputs.items())),
        "clean_embedding_manifest": str(clean_manifest_path.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "clean_embedding_manifest_sha256": sha256(clean_manifest_path),
        "clean_embedding_records": verified_clean_records,
        "v7_feature_manifest": str(feature_manifest_path.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "v7_feature_manifest_sha256": sha256(feature_manifest_path),
        "v7_reference_bundle": {
            "path": str(reference_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": feature_manifest["reference_bundle_sha256"],
        },
        "v7_feature_inputs": dict(sorted(verified_feature_inputs.items())),
        "v7_feature_outputs": verified_outputs,
    }


def render_csv(rows: list[dict], fields: list[str]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def run_root(policy: dict, run_id: str) -> Path:
    if not run_id or not run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Step15-v8 run-id may contain only letters, digits, underscore, and hyphen")
    return resolve(policy["outputs_root_template"].format(run_id=run_id))


def materialize_effective_v7_policy(policy: dict, base_v7_policy: dict) -> dict:
    """Overlay v8-bound data freezes without mutating the historical v7 policy."""
    effective = copy.deepcopy(base_v7_policy)
    pool_key_map = {
        "frozen_labels": "frozen_labels",
        "evidence_labels": "evidence_labels",
        "seller_profiles": "seller_profiles",
        "item_identity_signals": "item_identity_signals",
        "step4_candidates": "step4_candidates",
        "v7_pair_features": "v7_pair_features",
        "v7_clean_e5_metadata": "clean_e5_cache_metadata",
        "v7_clean_e5_matrix": "clean_e5_cache_matrix",
    }
    for pool_name, pool in policy["pools"].items():
        if pool_name not in effective["pools"]:
            raise ValueError(f"Step15-v8 pool is absent from the frozen v7 policy: {pool_name}")
        for v8_key, v7_key in pool_key_map.items():
            if v8_key not in pool:
                raise ValueError(f"Step15-v8 pool lacks required binding: {pool_name}:{v8_key}")
            effective["pools"][pool_name][v7_key] = pool[v8_key]
    effective["representative_validation"]["split_assignment_output"] = policy[
        "frozen_dependencies"
    ]["representative_validation_assignments"]
    effective["representative_validation"]["manifest_output"] = policy[
        "frozen_dependencies"
    ]["representative_validation_manifest"]
    effective["_step15_v8_effective_overlay"] = {
        "base_v7_policy": policy["frozen_dependencies"]["v7_policy"],
        "representative_validation_assignments": policy["frozen_dependencies"][
            "representative_validation_assignments"
        ],
        "representative_validation_manifest": policy["frozen_dependencies"][
            "representative_validation_manifest"
        ],
    }
    return effective


def load_policy(path: str | Path) -> tuple[Path, dict, dict]:
    policy_path = resolve(path)
    policy = load_json(policy_path)
    base_v7_policy = load_json(resolve(policy["frozen_dependencies"]["v7_policy"]))
    return policy_path, policy, materialize_effective_v7_policy(policy, base_v7_policy)


def validate_policy_contract(policy: dict, v7_policy: dict) -> dict:
    bridge = policy["bridge_audit"]
    expected = [
        "B0_v7_20d_plus_e5_latent64",
        "B1_v7_20d_e5_cosine_only",
        "B2_redacted_multiencoder_consensus",
        "B3_nonidentifier_retrieval_bridge",
    ]
    if list(bridge["feature_sets"]) != expected:
        raise ValueError("Step15-v8 B0-B3 feature-set order changed after preregistration")
    stable = list(v7_policy["inductive_features"]["stable_strict_clean_features"])
    e5 = "embedding_cosine_multilingual_e5_large_identifier_redacted"
    if len(stable) != 20 or e5 not in stable:
        raise ValueError("Step15-v8 requires the frozen v7 identifier-redacted 20d view")
    forbidden = set(bridge["forbidden_features"])
    if "candidate_rule_count_raw" not in forbidden:
        raise ValueError("Raw candidate-rule count must remain forbidden")
    if set(stable) & forbidden:
        raise ValueError(
            f"Frozen v7 base reintroduced forbidden features: {sorted(set(stable) & forbidden)}"
        )
    if set(bridge["nonidentifier_candidate_rule_allowlist"]) != {
        "profile_lexical_neighbor",
        "shared_title_clone",
        "shared_description_clone",
        "structural_support",
    }:
        raise ValueError("Step15-v8 nonidentifier retrieval allowlist changed")
    if policy["clean_semantics"].get("reranker_pair_symmetrization") != "mean_forward_reverse":
        raise ValueError("Step15-v8 reranker must be symmetrized over both seller orders")
    weighting = v7_policy["factorized_evidence_weighting"]
    if policy["factorized_evidence_weighting"].get("inherit_exactly_from_v7_policy") is not True:
        raise ValueError("Step15-v8 must inherit the frozen factorized evidence weighting")
    if weighting.get("forbid_global_eight_x_multiplier") is not True:
        raise ValueError("Step15-v8 forbids the historical global eight-times evidence weight")
    if "evidence_type_factor" not in weighting or "confidence_factor" not in weighting:
        raise ValueError("Step15-v8 weighting must factor domain/evidence type/confidence")
    all_selected = set()
    for cfg in bridge["feature_sets"].values():
        all_selected.update(cfg["add_features"])
    if all_selected & forbidden:
        raise ValueError(f"Forbidden features re-entered v8: {sorted(all_selected & forbidden)}")
    if policy["evaluation"]["selection_reads_internal_test"] is not False:
        raise ValueError("Internal-test selection must remain disabled")
    if bridge["internal_test_used_for_selection"] is not False:
        raise ValueError("Bridge selection must not read the internal test")
    return {
        "status": "pass",
        "feature_sets": expected,
        "v7_strict_clean_feature_count": len(stable),
        "seed_count": len(bridge["seeds"]),
        "group_folds": int(bridge["group_folds"]),
    }


def semantic_pair_path(root: Path, pool_name: str) -> Path:
    return root / "clean_semantics" / f"identifier_redacted_pair_semantics.{pool_name}.csv"


def semantic_cache_paths(root: Path, pool_name: str, model_key: str) -> tuple[Path, Path]:
    base = root / "clean_semantics"
    return (
        base / f"{model_key}_identifier_redacted.{pool_name}.json",
        base / f"{model_key}_identifier_redacted.{pool_name}.npy",
    )


def load_joined_rows(policy: dict, v7_policy: dict, root: Path) -> dict[str, list[dict]]:
    joined = v7.load_joined_rows(v7_policy)
    for pool_name, rows in joined.items():
        path = semantic_pair_path(root, pool_name)
        semantics = {row["pair_uid"]: row for row in load_csv(path)}
        if len(semantics) != len(load_csv(resolve(policy["pools"][pool_name]["v7_pair_features"]))):
            raise ValueError(f"V8 semantic pair universe is incomplete for {pool_name}")
        for row in rows:
            item = semantics.get(row["pair_uid"])
            if item is None:
                raise ValueError(f"Missing v8 redacted semantics: {pool_name}:{row['pair_uid']}")
            row.update(item)
    return joined


def primary_benchmark_evaluation_eligible(row: dict) -> bool:
    """Return whether a row may contribute to primary validation/test metrics."""
    return (
        row.get("review_label") in {"positive", "negative"}
        and row.get("usable_for_supervision") == "1"
        and row.get("usable_for_core_transfer") == "1"
        and str(row.get("primary_identity_model_eligible", "1")).strip() != "0"
        and str(row.get("benchmark_eligible", "")).strip() == "1"
        and str(row.get("silver_train_only", "0")).strip() != "1"
    )


def split_rows(rows_by_pool: dict[str, list[dict]]) -> dict[str, list[dict]]:
    en = rows_by_pool["en_content_train_pool"]
    zh = rows_by_pool["zh_target_strict"]
    primary_en = [
        row
        for row in en
        if str(row.get("primary_identity_model_eligible", "1")).strip() != "0"
    ]
    primary_zh = [
        row
        for row in zh
        if str(row.get("primary_identity_model_eligible", "1")).strip() != "0"
    ]
    invalid_primary_eval = [
        row
        for row in primary_zh
        if row.get("v7_split_name") in {"valid", "internal_development_test"}
        and not primary_benchmark_evaluation_eligible(row)
    ]
    if invalid_primary_eval:
        first = invalid_primary_eval[0]
        raise ValueError(
            "Step15-v8 primary evaluation contains a non-benchmark or train-only "
            "row; regenerate the representative assignment before training: "
            f"split={first.get('v7_split_name')} pair_uid={first.get('pair_uid')} "
            f"benchmark_eligible={first.get('benchmark_eligible')} "
            f"silver_train_only={first.get('silver_train_only')}"
        )
    expert_controls = [
        row
        for row in zh
        if str(row.get("primary_identity_model_eligible", "1")).strip() == "0"
        and str(row.get("evidence_expert_eligible", "0")).strip() == "1"
    ]
    invalid_controls = [
        row
        for row in zh
        if str(row.get("primary_identity_model_eligible", "1")).strip() == "0"
        and str(row.get("evidence_expert_eligible", "0")).strip() != "1"
    ]
    if invalid_controls:
        raise ValueError(
            "Rows excluded from the primary identity model must be explicitly scoped to "
            f"the evidence expert: {invalid_controls[0]['pair_uid']}"
        )
    test_controls = [
        row
        for row in expert_controls
        if row["v7_split_name"] == "internal_development_test"
    ]
    if test_controls:
        raise ValueError(
            "Evidence-expert controls may not enter the frozen internal development test: "
            f"{test_controls[0]['pair_uid']}"
        )
    result = {
        "train": [row for row in primary_en if row["v7_split_name"] == "train"]
        + [row for row in primary_zh if row["v7_split_name"] == "train"],
        "valid": [row for row in primary_zh if row["v7_split_name"] == "valid"],
        "internal_development_test": [
            row
            for row in primary_zh
            if row["v7_split_name"] == "internal_development_test"
        ],
        "evidence_expert_train_controls": [
            row for row in expert_controls if row["v7_split_name"] == "train"
        ],
        "evidence_expert_valid_controls": [
            row for row in expert_controls if row["v7_split_name"] == "valid"
        ],
    }
    for split_name in ("train", "valid", "internal_development_test"):
        rows = result[split_name]
        labels = {row["review_label"] for row in rows}
        if labels != {"positive", "negative"}:
            raise ValueError(f"Step15-v8 split lacks both labels: {split_name}:{labels}")
    primary_uids = {
        row["pair_uid"]
        for split_name in ("train", "valid", "internal_development_test")
        for row in result[split_name]
    }
    control_uids = {
        row["pair_uid"]
        for split_name in (
            "evidence_expert_train_controls",
            "evidence_expert_valid_controls",
        )
        for row in result[split_name]
    }
    overlap = primary_uids & control_uids
    if overlap:
        raise ValueError(
            "Primary identity rows and evidence-expert controls overlap: "
            f"{sorted(overlap)[0]}"
        )
    return result


def load_corpus_reference_context(policy: dict, v7_policy: dict) -> dict:
    schema = load_json(resolve(v7_policy["inductive_features"]["step7_feature_schema"]))
    profiles = {}
    for pool_name, pool in policy["pools"].items():
        rows = v6_features.load_jsonl(resolve(pool["seller_profiles"]))
        profiles[pool_name] = {str(row["seller_uid"]): row for row in rows}
        if len(profiles[pool_name]) != len(rows):
            raise ValueError(f"Duplicate seller profile UID in v8 corpus context: {pool_name}")
    return {
        "profiles": profiles,
        "numeric_paths": dict(schema["market_relative_numeric_fields"]),
        "boilerplate_config": dict(schema["boilerplate_feature_config"]),
        "oov_policy": dict(v7_policy["inductive_features"]["oov_policy"]),
    }


def fit_corpus_reference(rows: list[dict], context: dict) -> dict:
    references = {}
    for pool_name in sorted({row["step15_pool"] for row in rows}):
        pool_rows = [row for row in rows if row["step15_pool"] == pool_name]
        train_sellers = {
            str(row[key])
            for row in pool_rows
            for key in ("seller_uid_left", "seller_uid_right")
        }
        profiles = context["profiles"][pool_name]
        missing = sorted(train_sellers - set(profiles))
        if missing:
            raise ValueError(f"Fold-train profile missing for {pool_name}:{missing[0]}")
        reference = v6_features.fit_reference(
            profiles,
            train_sellers,
            context["numeric_paths"],
            context["boilerplate_config"],
        )
        reference["minimum_market_group_size"] = v6_features.preview.MIN_MARKET_GROUP_SIZE
        references[pool_name] = reference
    expected_pools = {row["step15_pool"] for row in rows}
    if set(references) != expected_pools:
        raise ValueError("Step15-v8 fold corpus references are incomplete")
    return references


def apply_corpus_reference(rows: list[dict], references: dict, context: dict) -> list[dict]:
    transformed = []
    for row in rows:
        pool_name = row["step15_pool"]
        if pool_name not in references:
            raise ValueError(f"No fold-train corpus reference for evaluation pool={pool_name}")
        profiles = context["profiles"][pool_name]
        left_uid = str(row["seller_uid_left"])
        right_uid = str(row["seller_uid_right"])
        if left_uid not in profiles or right_uid not in profiles:
            raise ValueError(f"Evaluation seller profile missing: {row['pair_uid']}")
        output = dict(row)
        output.update(
            v7_features.derive_reference_fields(
                profiles[left_uid],
                profiles[right_uid],
                references[pool_name],
                context["numeric_paths"],
                context["oov_policy"],
            )
        )
        transformed.append(output)
    return transformed


def feature_names(feature_set_id: str, policy: dict, v7_policy: dict) -> list[str]:
    cfg = policy["bridge_audit"]["feature_sets"][feature_set_id]
    stable = list(v7_policy["inductive_features"]["stable_strict_clean_features"])
    e5 = "embedding_cosine_multilingual_e5_large_identifier_redacted"
    if cfg["base"] == "v7_strict_clean_20d":
        names = stable
    elif cfg["base"] == "v7_nonsemantic_19d":
        names = [name for name in stable if name != e5]
    else:
        raise ValueError(f"Unknown Step15-v8 feature base: {cfg['base']}")
    for name in cfg["add_features"]:
        if name not in names:
            names.append(name)
    if cfg["add_e5_pair_latent64"]:
        names.extend(f"e5_pair_latent_{index:03d}" for index in range(64))
    forbidden = set(policy["bridge_audit"]["forbidden_features"])
    if set(names) & forbidden:
        raise ValueError(
            f"Step15-v8 feature set {feature_set_id} contains forbidden inputs: "
            f"{sorted(set(names) & forbidden)}"
        )
    return names


def _float_or_nan(value: object) -> float:
    token = str(value if value is not None else "").strip()
    if not token:
        return math.nan
    try:
        parsed = float(token)
    except ValueError:
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _fit_domain_stats(rows: list[dict], raw_name: str) -> dict:
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float_or_nan(row.get(raw_name))
        if math.isfinite(value):
            by_domain[row["domain"]].append(value)
    result = {}
    for domain in sorted({row["domain"] for row in rows}):
        values = np.asarray(by_domain.get(domain, []), dtype=float)
        if len(values) == 0:
            raise ValueError(f"No fold-train values for domain-normalized feature {domain}:{raw_name}")
        mean = float(np.mean(values))
        scale = float(np.std(values))
        result[domain] = {"mean": mean, "scale": scale if scale > 1e-12 else 1.0}
    return result


def fit_feature_transform(
    rows: list[dict],
    feature_set_id: str,
    policy: dict,
    v7_policy: dict,
    latent: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    cfg = policy["bridge_audit"]["feature_sets"][feature_set_id]
    names = feature_names(feature_set_id, policy, v7_policy)
    normalized = policy["bridge_audit"]["domain_normalized_raw_features"]
    domain_stats = {
        output_name: _fit_domain_stats(rows, raw_name)
        for output_name, raw_name in normalized.items()
        if output_name in names
    }
    nonlatent = [name for name in names if not name.startswith("e5_pair_latent_")]
    matrix = np.empty((len(rows), len(nonlatent)), dtype=float)
    for row_index, row in enumerate(rows):
        for column, name in enumerate(nonlatent):
            if name in normalized:
                raw = _float_or_nan(row.get(normalized[name]))
                stats = domain_stats[name][row["domain"]]
                matrix[row_index, column] = (
                    math.nan if not math.isfinite(raw) else (raw - stats["mean"]) / stats["scale"]
                )
            else:
                matrix[row_index, column] = _float_or_nan(row.get(name))
    imputation = v7.fit_train_median_imputation(matrix)
    matrix = v7.apply_imputation(matrix, imputation)
    if cfg["add_e5_pair_latent64"]:
        if latent is None or latent.shape != (len(rows), 64):
            raise ValueError(f"B0 requires an aligned 64d E5 latent matrix: {latent.shape if latent is not None else None}")
        matrix = np.hstack([matrix, np.asarray(latent, dtype=float)])
    if matrix.shape[1] != len(names):
        raise ValueError(f"Step15-v8 feature dimension mismatch for {feature_set_id}")
    artifact = {
        "feature_set_id": feature_set_id,
        "feature_names": names,
        "domain_normalization": domain_stats,
        "median_imputation": imputation,
        "add_e5_pair_latent64": bool(cfg["add_e5_pair_latent64"]),
        "fit_pair_uid_sha256": canonical_hash(sorted(row["pair_uid"] for row in rows)),
    }
    return matrix, artifact


def apply_feature_transform(
    rows: list[dict],
    policy: dict,
    v7_policy: dict,
    artifact: dict,
    latent: np.ndarray | None,
) -> np.ndarray:
    feature_set_id = artifact["feature_set_id"]
    cfg = policy["bridge_audit"]["feature_sets"][feature_set_id]
    names = list(artifact["feature_names"])
    normalized = policy["bridge_audit"]["domain_normalized_raw_features"]
    nonlatent = [name for name in names if not name.startswith("e5_pair_latent_")]
    matrix = np.empty((len(rows), len(nonlatent)), dtype=float)
    for row_index, row in enumerate(rows):
        for column, name in enumerate(nonlatent):
            if name in normalized:
                raw = _float_or_nan(row.get(normalized[name]))
                stats = artifact["domain_normalization"][name].get(row["domain"])
                if stats is None:
                    raise ValueError(f"Unseen domain in v8 fold transform: {row['domain']}")
                matrix[row_index, column] = (
                    math.nan if not math.isfinite(raw) else (raw - stats["mean"]) / stats["scale"]
                )
            else:
                matrix[row_index, column] = _float_or_nan(row.get(name))
    matrix = v7.apply_imputation(matrix, artifact["median_imputation"])
    if cfg["add_e5_pair_latent64"]:
        if latent is None or latent.shape != (len(rows), 64):
            raise ValueError("Applying B0 requires aligned 64d E5 pair latents")
        matrix = np.hstack([matrix, np.asarray(latent, dtype=float)])
    if matrix.shape[1] != len(names):
        raise ValueError("Applied Step15-v8 feature dimension differs from artifact")
    return matrix


def component_group_key(row: dict) -> str:
    domain = str(row.get("domain", "")).strip()
    component = str(row.get("v7_component_id", "")).strip()
    if not domain or not component:
        raise ValueError(f"Missing domain/component for grouped OOF: {row.get('pair_uid')}")
    return f"{domain}::{component}"


def seeded_component_group_folds(
    rows: list[dict], n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("GroupKFold requires at least two folds")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[component_group_key(row)].append(index)
    if len(grouped) < n_splits:
        raise ValueError("Fewer seller components than requested GroupKFold folds")
    labels = v7.labels_array(rows)
    target_pos = float(np.sum(labels)) / n_splits
    target_neg = float(len(labels) - np.sum(labels)) / n_splits
    target_rows = float(len(labels)) / n_splits
    records = []
    for group, indices in grouped.items():
        positive = float(np.sum(labels[indices]))
        negative = float(len(indices) - positive)
        tie = hashlib.sha256(f"{seed}|{group}".encode("utf-8")).hexdigest()
        records.append((group, indices, positive, negative, tie))
    records.sort(key=lambda item: (-len(item[1]), -max(item[2], item[3]), item[4]))
    fold_groups = [set() for _ in range(n_splits)]
    fold_pos = np.zeros(n_splits, dtype=float)
    fold_neg = np.zeros(n_splits, dtype=float)
    fold_rows = np.zeros(n_splits, dtype=float)
    fold_tie_order = sorted(
        range(n_splits),
        key=lambda fold: hashlib.sha256(f"{seed}|fold|{fold}".encode("utf-8")).hexdigest(),
    )
    tie_rank = {fold: rank for rank, fold in enumerate(fold_tie_order)}
    for group, indices, positive, negative, _ in records:
        candidates = []
        for fold in range(n_splits):
            next_pos = fold_pos.copy()
            next_neg = fold_neg.copy()
            next_rows = fold_rows.copy()
            next_pos[fold] += positive
            next_neg[fold] += negative
            next_rows[fold] += len(indices)
            cost = float(
                np.sum((next_pos - target_pos) ** 2)
                + np.sum((next_neg - target_neg) ** 2)
                + 0.1 * np.sum((next_rows - target_rows) ** 2)
            )
            candidates.append((cost, fold_rows[fold], tie_rank[fold], fold))
        _, _, _, selected = min(candidates)
        fold_groups[selected].add(group)
        fold_pos[selected] += positive
        fold_neg[selected] += negative
        fold_rows[selected] += len(indices)
    all_indices = np.arange(len(rows), dtype=int)
    result = []
    for fold, groups in enumerate(fold_groups):
        valid = np.asarray(
            [index for index, row in enumerate(rows) if component_group_key(row) in groups],
            dtype=int,
        )
        train = np.setdiff1d(all_indices, valid, assume_unique=True)
        if len(valid) == 0 or set(labels[valid]) != {0.0, 1.0}:
            raise ValueError(f"Seeded GroupKFold fold {fold} lacks both OOF labels")
        if set(labels[train]) != {0.0, 1.0}:
            raise ValueError(f"Seeded GroupKFold fold {fold} lacks both train labels")
        train_groups = {component_group_key(rows[index]) for index in train}
        valid_groups = {component_group_key(rows[index]) for index in valid}
        if train_groups & valid_groups:
            raise ValueError("Seller component leaked across a Step15-v8 OOF fold")
        result.append((train, valid))
    covered = sorted(index for _, valid in result for index in valid.tolist())
    if covered != list(range(len(rows))):
        raise ValueError("Step15-v8 OOF folds do not partition train rows exactly once")
    return result


def macro_domain_average_precision(rows: list[dict], scores: np.ndarray) -> tuple[float, dict]:
    by_domain = {}
    for domain in sorted({row["domain"] for row in rows}):
        mask = np.asarray([row["domain"] == domain for row in rows], dtype=bool)
        labels = v7.labels_array([row for row in rows if row["domain"] == domain])
        if set(labels) != {0.0, 1.0}:
            raise ValueError(f"OOF domain lacks both labels: {domain}")
        by_domain[domain] = float(step7.average_precision_score(labels, scores[mask]))
    return float(np.mean(list(by_domain.values()))), by_domain


def fit_lr(
    x_train: np.ndarray,
    rows: list[dict],
    policy: dict,
    v7_policy: dict,
) -> dict:
    weights, diagnostics = v7.factorized_evidence_weights(
        rows, v7_policy["factorized_evidence_weighting"]
    )
    artifact, _ = step9.fit_regularized_logistic(
        x_train,
        v7.labels_array(rows),
        policy["bridge_audit"]["logistic"],
        sample_weight_multipliers=weights,
        sample_weight_target_total=float(len(rows)),
    )
    return {"model_family": "lr_l2", "logistic": artifact, "weight_diagnostics": diagnostics}


def apply_lr(x: np.ndarray, artifact: dict) -> np.ndarray:
    return step9.apply_logistic_artifact_to_matrix(x, artifact["logistic"])


def _pairwise_examples(
    x_scaled: np.ndarray,
    rows: list[dict],
    row_weights: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    labels = v7.labels_array(rows)
    differences = []
    targets = []
    weights = []
    pair_count_by_domain = Counter()
    maximum = int(cfg["maximum_negatives_per_positive"])
    for domain in sorted({row["domain"] for row in rows}):
        positives = [i for i, row in enumerate(rows) if row["domain"] == domain and labels[i] == 1]
        negatives = [i for i, row in enumerate(rows) if row["domain"] == domain and labels[i] == 0]
        if not positives or not negatives:
            raise ValueError(f"Pairwise ranker lacks both labels in domain={domain}")
        for positive in positives:
            ordered = sorted(
                negatives,
                key=lambda negative: hashlib.sha256(
                    f"{seed}|{rows[positive]['pair_uid']}|{rows[negative]['pair_uid']}".encode("utf-8")
                ).hexdigest(),
            )[:maximum]
            for negative in ordered:
                weight = float(min(row_weights[positive], row_weights[negative]))
                differences.append(x_scaled[positive] - x_scaled[negative])
                targets.append(1.0)
                weights.append(weight)
                if cfg["include_reverse_pair"]:
                    differences.append(x_scaled[negative] - x_scaled[positive])
                    targets.append(0.0)
                    weights.append(weight)
                pair_count_by_domain[domain] += 1
    return (
        np.asarray(differences, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(weights, dtype=float),
        {"unordered_pair_count_by_domain": dict(sorted(pair_count_by_domain.items()))},
    )


def fit_pairwise_ranker(
    x_train: np.ndarray,
    rows: list[dict],
    policy: dict,
    v7_policy: dict,
    seed: int,
) -> dict:
    cfg = policy["bridge_audit"]["linear_pairwise_ranker"]
    x_scaled, row_standardization = step9.fit_standardization(x_train, True)
    row_weights, weight_diagnostics = v7.factorized_evidence_weights(
        rows, v7_policy["factorized_evidence_weighting"]
    )
    pair_x, pair_y, pair_weights, pair_diagnostics = _pairwise_examples(
        x_scaled, rows, row_weights, cfg, seed
    )
    rank_cfg = {
        "l2_penalty": cfg["l2_penalty"],
        "max_iter": cfg["max_iter"],
        "tolerance": cfg["tolerance"],
        "class_weight": cfg["class_weight"],
        "standardize_features": False,
    }
    logistic, _ = step9.fit_regularized_logistic(
        pair_x,
        pair_y,
        rank_cfg,
        sample_weight_multipliers=pair_weights,
        sample_weight_target_total=float(len(pair_y)),
    )
    return {
        "model_family": "linear_pairwise_ranknet",
        "row_standardization": row_standardization,
        "pairwise_logistic": logistic,
        "pair_diagnostics": pair_diagnostics,
        "weight_diagnostics": weight_diagnostics,
    }


def apply_pairwise_ranker(x: np.ndarray, artifact: dict) -> np.ndarray:
    x_scaled = step9.apply_standardization(x, artifact["row_standardization"])
    logistic = artifact["pairwise_logistic"]
    coefficients = np.asarray(logistic["parameter_coefficients"], dtype=float)
    margin = float(logistic["parameter_intercept"]) + x_scaled @ coefficients
    return step9.safe_sigmoid(margin)


def apply_model(x: np.ndarray, artifact: dict) -> np.ndarray:
    if artifact["model_family"] == "lr_l2":
        return apply_lr(x, artifact)
    if artifact["model_family"] == "linear_pairwise_ranknet":
        return apply_pairwise_ranker(x, artifact)
    raise ValueError(f"Unknown Step15-v8 model family: {artifact['model_family']}")


def item_signal_index(path: Path, train_sellers: set[str]) -> tuple[dict, Counter]:
    by_seller: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sellers_by_token: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in load_csv(path):
        contact_type = str(row.get("contact_type", "")).strip().lower()
        value = str(row.get("normalized_value", "")).strip().lower()
        seller = str(row.get("seller_uid", "")).strip()
        if not contact_type or not value or not seller:
            continue
        token = (contact_type, value)
        by_seller[seller][token].append(row)
        if seller in train_sellers:
            sellers_by_token[token].add(seller)
    return by_seller, Counter({token: len(sellers) for token, sellers in sellers_by_token.items()})


def _direct_occurrence(row: dict) -> bool:
    return (
        row.get("direct_identity_eligible") == "1"
        and row.get("seller_facing_context") == "1"
        and row.get("product_data_risk_context") != "1"
        and row.get("support_only") != "1"
    )


def _risky_occurrence(row: dict) -> bool:
    return row.get("product_data_risk_context") == "1"


def _support_occurrence(row: dict) -> bool:
    return row.get("support_only") == "1"


def occurrence_evidence(
    row: dict,
    by_seller: dict,
    token_df: Counter,
    frequency_threshold: int,
) -> dict:
    left = by_seller.get(row["seller_uid_left"], {})
    right = by_seller.get(row["seller_uid_right"], {})
    shared = sorted(set(left) & set(right))
    counts = Counter()
    distinct_items = set()
    distinct_markets = set()
    token_types = set()
    token_hashes = []
    for token in shared:
        left_occ = left[token]
        right_occ = right[token]
        all_occ = left_occ + right_occ
        left_direct = any(_direct_occurrence(item) for item in left_occ)
        right_direct = any(_direct_occurrence(item) for item in right_occ)
        risky = any(_risky_occurrence(item) for item in all_occ)
        support = any(_support_occurrence(item) for item in all_occ)
        high_frequency = token_df[token] > frequency_threshold
        token_types.add(token[0])
        token_hashes.append(canonical_hash(token)[:16])
        for item in all_occ:
            source_row = str(item.get("source_row_number", "")).strip()
            source_dataset = str(item.get("source_dataset", "")).strip()
            if source_row or source_dataset:
                distinct_items.add(f"{source_dataset}:{source_row}")
            market = str(item.get("source_market_raw", "")).strip()
            if market:
                distinct_markets.add(market)
        if left_direct and right_direct and (risky or support):
            counts["mixed_context"] += 1
        elif left_direct and right_direct and not high_frequency:
            counts["verified_direct"] += 1
        elif risky:
            counts["risky_only"] += 1
        elif support:
            counts["support_only"] += 1
        elif high_frequency:
            counts["high_frequency"] += 1
        else:
            counts["ambiguous"] += 1
    if counts["verified_direct"]:
        state = "verified_direct_both_sides"
    elif counts["mixed_context"]:
        state = "direct_with_mixed_context"
    elif counts["risky_only"]:
        state = "risky_only_shared"
    elif counts["support_only"]:
        state = "support_only_shared"
    elif counts["high_frequency"]:
        state = "high_frequency_public"
    elif shared:
        state = "ambiguous"
    else:
        state = "no_shared_identifier"
    financial_phone_types = {
        "phone",
        "crypto_wallet",
        "wallet",
        "pgp_fingerprint",
        "pgp_public_key",
        "qq",
        "wechat",
        "jabber",
    }
    public_url = any(token_type in {"url", "domain", "external_url"} for token_type in token_types)
    return {
        "evidence_state": state,
        "verified_direct_token_count": int(counts["verified_direct"]),
        "risky_only_token_count": int(counts["risky_only"]),
        "support_only_token_count": int(counts["support_only"]),
        "mixed_context_token_count": int(counts["mixed_context"]),
        "high_frequency_token_count": int(counts["high_frequency"]),
        "ambiguous_token_count": int(counts["ambiguous"]),
        "shared_token_count": len(shared),
        "maximum_train_seller_token_frequency": max((token_df[token] for token in shared), default=0),
        "distinct_item_count": len(distinct_items),
        "distinct_market_count": len(distinct_markets),
        "public_url_or_domain_flag": int(public_url),
        "identifier_type_telegram_flag": int("telegram" in token_types),
        "identifier_type_email_flag": int("email" in token_types),
        "identifier_type_financial_phone_flag": int(bool(token_types & financial_phone_types)),
        "shared_token_hashes": token_hashes,
        "shared_identifier_types": sorted(token_types),
    }


def evidence_feature_matrix(rows: list[dict], evidence: list[dict], policy: dict) -> np.ndarray:
    names = policy["occurrence_evidence_expert"]["feature_names"]
    matrix = np.zeros((len(rows), len(names)), dtype=float)
    for index, (row, item) in enumerate(zip(rows, evidence, strict=True)):
        values = {
            "verified_direct_token_count_log1p": math.log1p(item["verified_direct_token_count"]),
            "risky_only_token_count_log1p": math.log1p(item["risky_only_token_count"]),
            "support_only_token_count_log1p": math.log1p(item["support_only_token_count"]),
            "mixed_context_token_count_log1p": math.log1p(item["mixed_context_token_count"]),
            "high_frequency_token_count_log1p": math.log1p(item["high_frequency_token_count"]),
            "shared_token_count_log1p": math.log1p(item["shared_token_count"]),
            "distinct_item_count_log1p": math.log1p(item["distinct_item_count"]),
            "distinct_market_count_log1p": math.log1p(item["distinct_market_count"]),
            "public_url_or_domain_flag": item["public_url_or_domain_flag"],
            "identifier_type_telegram_flag": item["identifier_type_telegram_flag"],
            "identifier_type_email_flag": item["identifier_type_email_flag"],
            "identifier_type_financial_phone_flag": item["identifier_type_financial_phone_flag"],
            "zh_x_verified_direct": int(row["domain"] == "zh") * int(item["verified_direct_token_count"] > 0),
            "zh_x_risky_only": int(row["domain"] == "zh") * int(item["risky_only_token_count"] > 0),
            "zh_x_mixed_context": int(row["domain"] == "zh") * int(item["mixed_context_token_count"] > 0),
        }
        matrix[index] = [float(values[name]) for name in names]
    return matrix


def fit_offset_logistic_expert(
    x_train: np.ndarray,
    y_train: np.ndarray,
    clean_probabilities: np.ndarray,
    sample_weights: np.ndarray,
    policy: dict,
) -> dict:
    cfg = policy["occurrence_evidence_expert"]
    x_scaled, standardization = step9.fit_standardization(x_train, True)
    y = np.asarray(y_train, dtype=float)
    offset = step9.safe_logit(np.asarray(clean_probabilities, dtype=float), 1e-6)
    weights = np.asarray(sample_weights, dtype=float)
    weights *= len(weights) / float(np.sum(weights))
    names = list(cfg["feature_names"])
    penalties = np.ones(len(names), dtype=float)
    for name in cfg["domain_interaction_features"]:
        penalties[names.index(name)] = float(cfg["chinese_interaction_l2_multiplier"])
    l2 = float(cfg["base_l2_penalty"])
    params = np.zeros(x_scaled.shape[1] + 1, dtype=float)
    converged = False
    final_delta = math.inf
    for iteration in range(1, int(cfg["max_iter"]) + 1):
        logits = offset + params[0] + x_scaled @ params[1:]
        probabilities = step9.safe_sigmoid(logits)
        residual = (probabilities - y) * weights
        gradient = np.empty(len(params), dtype=float)
        gradient[0] = float(np.sum(residual))
        gradient[1:] = x_scaled.T @ residual + l2 * penalties * params[1:]
        curvature = probabilities * (1.0 - probabilities) * weights
        weighted_x = x_scaled * curvature[:, None]
        hessian = np.empty((len(params), len(params)), dtype=float)
        hessian[0, 0] = float(np.sum(curvature))
        hessian[0, 1:] = np.sum(weighted_x, axis=0)
        hessian[1:, 0] = hessian[0, 1:]
        hessian[1:, 1:] = x_scaled.T @ weighted_x
        hessian[1:, 1:] += np.diag(l2 * penalties)
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(hessian) @ gradient
        delta = np.clip(delta, -5.0, 5.0)
        params -= delta
        final_delta = float(np.linalg.norm(delta))
        if final_delta <= float(cfg["tolerance"]):
            converged = True
            break
    return {
        "model_family": "compact_offset_logistic_l2",
        "feature_names": names,
        "standardization": standardization,
        "parameter_intercept": float(params[0]),
        "parameter_coefficients": params[1:].tolist(),
        "penalty_multipliers": penalties.tolist(),
        "base_l2_penalty": l2,
        "solver_iterations": iteration,
        "solver_converged": converged,
        "solver_final_delta_norm": final_delta,
    }


def expert_logit_correction(x: np.ndarray, artifact: dict) -> np.ndarray:
    scaled = step9.apply_standardization(x, artifact["standardization"])
    return float(artifact["parameter_intercept"]) + scaled @ np.asarray(
        artifact["parameter_coefficients"], dtype=float
    )


def apply_constrained_expert(
    clean_probabilities: np.ndarray,
    evidence: list[dict],
    corrections: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    clean_logits = step9.safe_logit(np.asarray(clean_probabilities, dtype=float), 1e-6)
    applied = np.zeros(len(corrections), dtype=float)
    decisions = []
    for index, (item, raw_delta) in enumerate(zip(evidence, corrections, strict=True)):
        state = item["evidence_state"]
        if state == "verified_direct_both_sides":
            delta = max(0.0, float(raw_delta))
            action = "nonnegative_uplift"
        elif state in {"risky_only_shared", "support_only_shared", "high_frequency_public"}:
            delta = min(0.0, float(raw_delta))
            action = "nonpositive_downgrade"
        else:
            delta = 0.0
            action = "no_score_change"
        applied[index] = delta
        decisions.append(
            {
                "evidence_state": state,
                "expert_action": action,
                "raw_logit_correction": float(raw_delta),
                "applied_logit_correction": delta,
            }
        )
    fused = step9.safe_sigmoid(clean_logits + applied)
    return fused, decisions


def evidence_slice(rows: list[dict], evidence_type: str) -> np.ndarray:
    return np.asarray([row["evidence_type"] == evidence_type for row in rows], dtype=bool)


def validation_slice_masks(rows: list[dict], evidence_states: list[str]) -> dict[str, np.ndarray]:
    """Define v8 gate slices from labels plus inference-visible occurrence states."""
    if len(rows) != len(evidence_states):
        raise ValueError("Validation rows and occurrence states have different lengths")
    public_states = {"risky_only_shared", "support_only_shared", "high_frequency_public"}
    benchmark_ok = []
    for row in rows:
        primary_benchmark = primary_benchmark_evaluation_eligible(row)
        evidence_control = (
            str(row.get("evidence_expert_validation_eligible", "0")).strip()
            == "1"
            and str(row.get("primary_identity_model_eligible", "1")).strip()
            == "0"
            and str(row.get("evidence_expert_eligible", "0")).strip() == "1"
        )
        benchmark_ok.append(primary_benchmark or evidence_control)
    public_noise = np.asarray(
        [
            eligible and row["review_label"] == "negative" and state in public_states
            for row, state, eligible in zip(
                rows, evidence_states, benchmark_ok, strict=True
            )
        ],
        dtype=bool,
    )
    verified_direct = np.asarray(
        [
            eligible
            and row["review_label"] == "positive"
            and state == "verified_direct_both_sides"
            for row, state, eligible in zip(
                rows, evidence_states, benchmark_ok, strict=True
            )
        ],
        dtype=bool,
    )
    component_anchor = np.asarray(
        [
            eligible
            and row["review_label"] == "positive"
            and row["evidence_type"] == "same_controller_component_anchor"
            and state != "verified_direct_both_sides"
            for row, state, eligible in zip(
                rows, evidence_states, benchmark_ok, strict=True
            )
        ],
        dtype=bool,
    )
    template_clone = np.asarray(
        [
            eligible
            and row["review_label"] == "negative"
            and row["evidence_type"] == "template_clone_not_controller"
            for row, eligible in zip(rows, benchmark_ok, strict=True)
        ],
        dtype=bool,
    )
    return {
        "state_backed_public_noise_negative": public_noise,
        "state_backed_verified_direct_positive": verified_direct,
        "same_controller_component_anchor_positive": component_anchor,
        "direct_or_component_positive": verified_direct | component_anchor,
        "template_clone_negative": template_clone,
    }


def false_positive_rate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float | None:
    negative = y_true == 0.0
    if not np.any(negative):
        return None
    return float(np.mean(scores[negative] >= threshold))


def recall_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float | None:
    positive = y_true == 1.0
    if not np.any(positive):
        return None
    return float(np.mean(scores[positive] >= threshold))
