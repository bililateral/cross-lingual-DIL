#!/usr/bin/env python3
"""Fail-closed common contracts for the V9.4.1 model experiment.

This module is deliberately supervision-free.  Importing or validating it must
not open train/development labels, qrels, controller files, either audit truth,
or the rejected July draft.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_v9_4_1_model_experiment_policy_v1.json"
)
EXPECTED_VERSION = "2026-08-30-step28-v13-v1.13-v9.4.1-model-experiment-policy-v1"
EXPECTED_STATUS = "IMPLEMENTATION_ONLY_NO_MODEL_RUN_NO_AUDIT_TRUTH_UNSEAL"
EXPECTED_POLICY_SIZE_BYTES = 25685
EXPECTED_POLICY_SHA256 = "11b57da9c98385a254d9d6f435d7e9fbba2c0730a765742a53008c00e95fb5af"
EXPECTED_POLICY_CANONICAL_SELF_HASH = (
    "9b555d59a934dd7263e4811f5b907d530f253400d27de2b9d411e0cecdcd94d8"
)
M1_DOMAIN = "step28-v13-v1.13-m1"
FROZEN_L2_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


class ModelExperimentContractError(ValueError):
    """Raised when a frozen model-experiment boundary does not replay."""


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ModelExperimentContractError(f"Expected JSON object: {path}")
    return value


def parse_exact_policy_bytes(raw: bytes) -> dict[str, Any]:
    """Reject every policy byte string except the independently pinned one."""

    if len(raw) != EXPECTED_POLICY_SIZE_BYTES:
        raise ModelExperimentContractError(
            "Model-experiment policy exact byte-size drift"
        )
    observed = hashlib.sha256(raw).hexdigest()
    if observed != EXPECTED_POLICY_SHA256:
        raise ModelExperimentContractError(
            "Model-experiment policy exact SHA-256 drift"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelExperimentContractError("Model-experiment policy is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ModelExperimentContractError("Model-experiment policy is not an object")
    if value.get("canonical_self_hash") != EXPECTED_POLICY_CANONICAL_SELF_HASH:
        raise ModelExperimentContractError(
            "Model-experiment policy exact canonical self-hash drift"
        )
    return value


def verify_self_hash(payload: Mapping[str, Any], *, label: str) -> None:
    claimed = payload.get("canonical_self_hash")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ModelExperimentContractError(f"{label} lacks a canonical self-hash")
    unsigned = dict(payload)
    unsigned.pop("canonical_self_hash", None)
    observed = canonical_sha256(unsigned)
    if observed != claimed:
        raise ModelExperimentContractError(
            f"{label} self-hash drift: expected={claimed} observed={observed}"
        )


def verify_file_pin(spec: Mapping[str, Any], *, label: str) -> Path:
    path = resolve(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    size = path.stat().st_size
    if size != int(spec["size_bytes"]):
        raise ModelExperimentContractError(
            f"{label} size drift: expected={spec['size_bytes']} observed={size}"
        )
    observed = sha256_file(path)
    if observed != spec["sha256"]:
        raise ModelExperimentContractError(
            f"{label} SHA-256 drift: expected={spec['sha256']} observed={observed}"
        )
    return path


def _validate_authority(policy: Mapping[str, Any]) -> None:
    registry = policy["authority_registry"]
    if list(registry) != [
        "full378_amendment",
        "scientific_contract",
        "quality_reconciliation",
        "implementation_contract",
        "historical_draft",
    ]:
        raise ModelExperimentContractError("Authority registry order/universe drift")
    for name in (
        "full378_amendment",
        "scientific_contract",
        "quality_reconciliation",
        "implementation_contract",
    ):
        verify_file_pin(registry[name], label=f"active authority {name}")

    # Do not resolve, stat, hash, parse, or import the historical draft here.
    rejected = registry["historical_draft"]
    if (
        rejected.get("role") != "HISTORICAL_DRAFT_NO_MODEL_AUTHORITY"
        or rejected.get("formal_model_process_open_count") != 0
        or rejected.get("sha256")
        != "ce18015199c864df0f76a240df782c331020e5e76d483c5440cea6a673c74729"
    ):
        raise ModelExperimentContractError("Historical-draft rejection contract drift")


def _validate_dataset_qualification(policy: Mapping[str, Any]) -> None:
    qualification = policy["dataset_qualification"]
    root_spec = qualification["root_manifest"]
    quality_spec = qualification["quality_result"]
    root_path = verify_file_pin(root_spec, label="V9.4.1 root manifest")
    quality_path = verify_file_pin(quality_spec, label="V9.4.1 quality result")
    root_manifest = load_json(root_path)
    quality_result = load_json(quality_path)
    verify_self_hash(root_manifest, label="V9.4.1 root manifest")
    verify_self_hash(quality_result, label="V9.4.1 quality result")

    for name, actual, expected in (
        ("root status", root_manifest.get("status"), root_spec["status"]),
        (
            "root training qualification",
            root_manifest.get("training_qualified"),
            root_spec["training_qualified"],
        ),
        (
            "root model authorization",
            root_manifest.get("m0_m1_m2_m3_training_authorized"),
            False,
        ),
        ("quality status", quality_result.get("status"), quality_spec["status"]),
        (
            "quality training qualification",
            quality_result.get("training_qualified"),
            quality_spec["training_qualified"],
        ),
        (
            "quality model authorization",
            quality_result.get("m0_m1_m2_m3_training_authorized"),
            False,
        ),
        (
            "quality/root lineage",
            quality_result.get("root_manifest_canonical_self_hash"),
            root_spec["canonical_self_hash"],
        ),
    ):
        if actual != expected:
            raise ModelExperimentContractError(
                f"{name} drift: expected={expected!r} observed={actual!r}"
            )

    truth = quality_result.get("truth_access", {})
    if (
        truth.get("audit_a_truth_semantic_reads") != 0
        or truth.get("audit_b_truth_semantic_reads") != 0
        or truth.get("train", {}).get("semantic_read_count") != 1
        or truth.get("development", {}).get("semantic_read_count") != 1
    ):
        raise ModelExperimentContractError("Quality/model truth-read boundary drift")


def _validate_feature_contract(policy: Mapping[str, Any]) -> None:
    features = policy["feature_contract"]
    legacy18 = list(features["legacy18"])
    labse6 = list(features["labse6"])
    identity33 = list(features["identity33"])
    if len(legacy18) != 18 or len(set(legacy18)) != 18:
        raise ModelExperimentContractError("legacy18 feature universe drift")
    if len(labse6) != 6 or len(set(labse6)) != 6:
        raise ModelExperimentContractError("LaBSE6 feature universe drift")
    if len(identity33) != 33 or len(set(identity33)) != 33:
        raise ModelExperimentContractError("identity33 feature universe drift")
    hashes = features["column_name_hashes"]
    observed = {
        "legacy18": canonical_sha256(legacy18),
        "labse6": canonical_sha256(labse6),
        "base24": canonical_sha256([*legacy18, *labse6]),
        "identity33": canonical_sha256(identity33),
        "joint57": canonical_sha256([*legacy18, *labse6, *identity33]),
    }
    for name, value in observed.items():
        if value != hashes[name]:
            raise ModelExperimentContractError(f"{name} column-name hash drift")


def _validate_static_scientific_contract(policy: Mapping[str, Any]) -> None:
    if policy.get("version") != EXPECTED_VERSION:
        raise ModelExperimentContractError("Model-experiment policy version drift")
    if policy.get("status") != EXPECTED_STATUS:
        raise ModelExperimentContractError("Model-experiment policy status drift")
    if (
        policy.get("m0_m1_m2_m3_training_authorized") is not False
        or policy.get("audit_truth_authorized") is not False
    ):
        raise ModelExperimentContractError("Pretraining policy grants forbidden authority")
    if policy["dataset_qualification"]["split_order"] != [
        "train",
        "development",
        "audit_a",
        "audit_b",
    ]:
        raise ModelExperimentContractError("Formal split order drift")
    weights = policy["folds_and_weights"]
    if weights != {
        "fold_count": 5,
        "world_sort": "sha256_world_uid_utf8_then_world_uid_utf8",
        "assignment": "sorted_ordinal_mod_5",
        "worlds_per_fold": 100,
        "m1_m2_train_row_weight": "1/(400*378)",
        "m1_m2_validation_row_weight": "1/(100*378)",
        "m1_m2_full_train_row_weight": "1/(500*378)",
        "m3_row_weight": 1.0,
        "arbitrary_constant_rescaling_allowed": False,
    }:
        raise ModelExperimentContractError("Fold/absolute-weight contract drift")
    m1 = policy["m1"]
    if (
        m1["repeat_ids"] != ["r01", "r02", "r03", "r04", "r05"]
        or m1["fixed_points"] != 0
        or m1["source_destination_endpoint_overlap"] != 0
        or m1["whole_identity33_row_moves"] is not True
        or m1["active_mask_moves_with_or_is_recomputed_from_mapped_row"] is not True
    ):
        raise ModelExperimentContractError("M1 derangement contract drift")
    if policy["m2"]["l2_selection"] != "max_lambda_with_loss_le_min_plus_1e-12":
        raise ModelExperimentContractError("M1/M2 L2 tie rule drift")
    m2 = policy["m2"]
    if (
        tuple(float(value) for value in m2["l2_grid"]) != FROZEN_L2_GRID
        or m2.get("p0_required_open_interval") != [0.0, 1.0]
        or m2.get("p0_interval_endpoints_included") is not False
    ):
        raise ModelExperimentContractError("M1/M2 grid or open-p0 contract drift")
    if policy["m3"]["grid_parameter_names"] != [
        "num_leaves",
        "min_child_samples",
        "learning_rate",
        "n_estimators",
    ]:
        raise ModelExperimentContractError("M3 grid parameter mapping drift")
    metrics = policy["metric_registry"]
    higher = set(metrics["higher_is_better_metrics"])
    lower = set(metrics["lower_is_better_metrics"])
    registered = set(
        metrics["pooled_score_metrics"]
        + metrics["threshold_metrics"]
        + metrics["retrieval_metrics"]
    )
    if higher & lower or higher | lower != registered:
        raise ModelExperimentContractError("Metric direction registry drift")
    if metrics.get("brier_formula") != "sum_i(w_i*(p_i-y_i)^2)/sum_i(w_i)":
        raise ModelExperimentContractError("Brier formula drift")
    if metrics.get("log_loss_formula") != (
        "-sum_i(w_i*(y_i*ln(p_i_clipped)+(1-y_i)*ln(1-p_i_clipped)))/sum_i(w_i)"
    ):
        raise ModelExperimentContractError("Log-loss formula drift")
    if metrics.get("comparison_direction") != {
        "higher_is_better": "metric(target_model)-metric(control_model)",
        "lower_is_better": "metric(control_model)-metric(target_model)",
        "positive_always_means_target_model_is_better": True,
        "confusion_matrices_have_scalar_direction": False,
    }:
        raise ModelExperimentContractError("Metric comparison direction drift")
    comparison_ids = [
        value.get("id") for value in metrics.get("ordered_model_comparisons", [])
    ]
    if comparison_ids != [
        "primary_m2_vs_mean_five_m1",
        "m2_vs_m0",
        "m3_joint_vs_m3_base",
        "m2_vs_m3_joint",
        "m1_equivalence_vs_m0",
    ]:
        raise ModelExperimentContractError("Ordered model comparison registry drift")
    encoding = policy["labse_encoding"]
    if list(encoding["chunk_tokenizer_payloads"]) != [
        "pcm_multilingual_authorship",
        "mstyledistance",
        "multilingual_e5_large",
        "labse",
    ]:
        raise ModelExperimentContractError("Four-tokenizer payload registry drift")
    if (
        encoding["chunk_tokenizer_payloads"]["labse"]["content_sha256"]
        != encoding["model_content_sha256"]
    ):
        raise ModelExperimentContractError("LaBSE content pin disagreement")
    fixture = encoding["compatibility_fixture"]
    if (
        fixture.get("selection_uses_supervised_labels_or_identity_evidence")
        is not False
        or fixture.get(
            "selection_uses_label_free_chunk_structure_and_frozen_labse_scores"
        )
        is not True
    ):
        raise ModelExperimentContractError("Compatibility-fixture selection boundary drift")


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    if path.resolve() != DEFAULT_POLICY.resolve():
        raise ModelExperimentContractError("Only the frozen default policy path is valid")
    policy = parse_exact_policy_bytes(path.read_bytes())
    verify_self_hash(policy, label="V9.4.1 model-experiment policy")
    _validate_static_scientific_contract(policy)
    _validate_authority(policy)
    _validate_dataset_qualification(policy)
    _validate_feature_contract(policy)
    return policy


def validate_identity33_column_names(
    policy: Mapping[str, Any], names: Sequence[str]
) -> None:
    expected = list(policy["feature_contract"]["identity33"])
    if list(names) != expected or canonical_sha256(list(names)) != policy[
        "feature_contract"
    ]["column_name_hashes"]["identity33"]:
        raise ModelExperimentContractError("identity33 exact column order drift")


def model_content_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing model/tokenizer payload directory: {path}")
    records = []
    for item in sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    ):
        if ".cache" in item.parts or "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        records.append(
            {
                "path": item.relative_to(path).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    if not records:
        raise ModelExperimentContractError(f"Empty model/tokenizer payload: {path}")
    return {
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "content_sha256": canonical_sha256(records),
    }


def validate_encoding_runtime(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Mechanical gate called immediately before any formal chunking/encoding."""

    expected_version = policy["runtime"]["encoding"]["sentence_transformers"]
    try:
        observed_version = importlib.metadata.version("sentence-transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ModelExperimentContractError("sentence-transformers is not installed") from exc
    if observed_version != expected_version:
        raise ModelExperimentContractError("sentence-transformers version drift")
    step7_path = verify_file_pin(
        policy["labse_encoding"]["step7_policy"], label="frozen Step7 policy"
    )
    step7 = load_json(step7_path)
    expected_payloads = policy["labse_encoding"]["chunk_tokenizer_payloads"]
    for key, pin in expected_payloads.items():
        source = step7["embedding_models"][key]
        if (
            source["local_path"] != pin["path"]
            or source["expected_file_count"] != pin["file_count"]
            or source["expected_total_size_bytes"] != pin["total_size_bytes"]
            or source["expected_content_sha256"] != pin["content_sha256"]
        ):
            raise ModelExperimentContractError(f"Step7 payload pin drift for {key}")
        observed = model_content_fingerprint(resolve(pin["path"]))
        expected = {
            "file_count": pin["file_count"],
            "total_size_bytes": pin["total_size_bytes"],
            "content_sha256": pin["content_sha256"],
        }
        if observed != expected:
            raise ModelExperimentContractError(f"Model/tokenizer payload drift for {key}")
    return {
        "sentence_transformers": observed_version,
        "step7_policy_sha256": policy["labse_encoding"]["step7_policy"]["sha256"],
        "payload_count": len(expected_payloads),
    }


def validate_supervised_cpu_runtime(policy: Mapping[str, Any]) -> dict[str, str]:
    """Mechanical gate called immediately before M1/M2/M3 fitting or metrics."""

    expected = policy["runtime"]["supervised_cpu"]
    observed = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "lightgbm": importlib.metadata.version("lightgbm"),
        "joblib": importlib.metadata.version("joblib"),
    }
    if observed != expected:
        raise ModelExperimentContractError(
            f"Supervised CPU runtime drift: expected={expected} observed={observed}"
        )
    return observed


def validate_frozen_model_payloads(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate joblib bytes and internal feature/imputation contracts."""

    import joblib

    features = policy["feature_contract"]
    expected_names = {
        "m0": [*features["legacy18"], *features["labse6"]],
        "c0": list(features["legacy18"]),
    }
    output: dict[str, Any] = {}
    for key in ("m0", "c0"):
        spec = policy["frozen_models"][key]
        path = verify_file_pin(spec, label=f"frozen {key.upper()} model")
        payload = joblib.load(path)
        if not isinstance(payload, dict):
            raise ModelExperimentContractError(f"Frozen {key} payload is not a dict")
        candidate = payload.get("candidate", {})
        artifact = payload.get("classifier_artifact", {})
        names = candidate.get("feature_names")
        medians = np.asarray(candidate.get("imputation_medians"), dtype=np.float64)
        if names != expected_names[key] or medians.shape != (len(expected_names[key]),):
            raise ModelExperimentContractError(f"Frozen {key} feature/imputation drift")
        if not np.isfinite(medians).all():
            raise ModelExperimentContractError(f"Frozen {key} medians are non-finite")
        if float(candidate.get("selected_threshold")) != float(spec["threshold"]):
            raise ModelExperimentContractError(f"Frozen {key} threshold drift")
        if artifact.get("classifier_id") != "lightgbm" or "model" not in artifact:
            raise ModelExperimentContractError(f"Frozen {key} classifier drift")
        if (
            payload.get("valid_label_values_read_for_fit_or_scoring") is not False
            or payload.get("historical_test_label_values_read") is not False
        ):
            raise ModelExperimentContractError(f"Frozen {key} supervision boundary drift")
        if key == "m0":
            audit = candidate.get("feature_reference_audit", {})
            reference = policy["frozen_english_reference"]
            if (
                audit.get("fit_pair_count") != reference["fit_pair_count"]
                or audit.get("fit_seller_count") != reference["fit_seller_count"]
                or audit.get("fit_seller_uid_sha256")
                != reference["seller_uid_sha256"]
                or audit.get("feature_reference_sha256")
                != reference["feature_reference_sha256"]
            ):
                raise ModelExperimentContractError("Frozen M0 English reference drift")
        output[key] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": spec["sha256"],
            "feature_count": len(names),
            "threshold": float(spec["threshold"]),
            "imputation_medians": medians,
            "model": artifact["model"],
        }
    return output


def bootstrap_indices(policy: Mapping[str, Any], split: str) -> np.ndarray:
    cfg = policy["bootstrap"]
    split_cfg = cfg["splits"][split]
    indices = np.random.Generator(np.random.PCG64(int(split_cfg["seed"]))).integers(
        0,
        int(cfg["world_count"]),
        size=(int(cfg["replicates"]), int(cfg["draw_size"])),
        endpoint=bool(cfg["endpoint"]),
        dtype=np.int64,
    )
    if indices.dtype.str != "<i8" or not indices.flags.c_contiguous:
        raise ModelExperimentContractError("Bootstrap index dtype/order drift")
    observed = hashlib.sha256(indices.tobytes(order="C")).hexdigest()
    if observed != split_cfg["index_bytes_sha256"]:
        raise ModelExperimentContractError(
            f"Bootstrap index drift for {split}: {observed}"
        )
    return indices


def assign_world_folds(world_uids: Sequence[str]) -> dict[str, int]:
    if len(world_uids) != 500 or len(set(world_uids)) != 500:
        raise ModelExperimentContractError("Training fold assignment requires 500 worlds")
    ordered = sorted(
        world_uids,
        key=lambda value: (
            hashlib.sha256(value.encode("utf-8")).digest(),
            value.encode("utf-8"),
        ),
    )
    output = {world_uid: index % 5 for index, world_uid in enumerate(ordered)}
    if Counter(output.values()) != Counter({index: 100 for index in range(5)}):
        raise ModelExperimentContractError("Training world folds are not 100 each")
    return output


def canonical_pair_endpoints(left: str, right: str) -> tuple[str, str]:
    if not left or not right or left == right:
        raise ModelExperimentContractError("Pair endpoints are empty or self-linked")
    return tuple(sorted((left, right), key=lambda value: value.encode("utf-8")))  # type: ignore[return-value]


def pair_uid_hex(left: str, right: str) -> str:
    first, second = canonical_pair_endpoints(left, right)
    return (first.encode("utf-8") + b"\x00" + second.encode("utf-8")).hex()


def _m1_factor_edges(sellers: Sequence[str], factor: int) -> list[tuple[str, str]]:
    if len(sellers) != 28 or len(set(sellers)) != 28 or not 0 <= factor < 27:
        raise ModelExperimentContractError("K28 one-factor input drift")
    v = sorted(sellers, key=lambda value: value.encode("utf-8"))
    edges = [canonical_pair_endpoints(v[27], v[factor])]
    edges.extend(
        canonical_pair_endpoints(v[(factor + k) % 27], v[(factor - k) % 27])
        for k in range(1, 14)
    )
    endpoints = [endpoint for edge in edges for endpoint in edge]
    if len(edges) != 14 or len(set(edges)) != 14 or len(set(endpoints)) != 28:
        raise ModelExperimentContractError("K28 one-factor construction drift")
    return edges


def build_m1_mapping(
    world_uid: str,
    sellers: Sequence[str],
    repeat_id: str,
) -> dict[tuple[str, str], tuple[str, str]]:
    if repeat_id not in {"r01", "r02", "r03", "r04", "r05"}:
        raise ModelExperimentContractError("Unknown M1 repeat ID")
    mapping: dict[tuple[str, str], tuple[str, str]] = {}
    for factor in range(27):
        edges = _m1_factor_edges(sellers, factor)

        def sort_key(edge: tuple[str, str]) -> tuple[bytes, bytes]:
            uid_hex = pair_uid_hex(*edge)
            domain = [M1_DOMAIN, repeat_id, world_uid, factor, uid_hex]
            return hashlib.sha256(
                json.dumps(
                    domain,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).digest(), uid_hex.encode("utf-8")

        ordered = sorted(edges, key=sort_key)
        for index, destination in enumerate(ordered):
            source = ordered[(index + 1) % len(ordered)]
            if destination == source or set(destination) & set(source):
                raise ModelExperimentContractError("M1 mapping violates null geometry")
            mapping[destination] = source
    all_edges = {
        canonical_pair_endpoints(left, right)
        for index, left in enumerate(sellers)
        for right in sellers[index + 1 :]
    }
    if set(mapping) != all_edges or set(mapping.values()) != all_edges:
        raise ModelExperimentContractError("M1 mapping is not a K28 edge bijection")
    return mapping


def active_mask(identity33: np.ndarray) -> np.ndarray:
    matrix = np.asarray(identity33, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 33:
        raise ModelExperimentContractError("identity33 matrix shape drift")
    if not np.isfinite(matrix).all():
        raise ModelExperimentContractError("identity33 contains a non-finite value")
    return np.any(matrix != 0.0, axis=1)


def fit_identity_transform(
    identity33: np.ndarray,
    world_uids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(identity33, dtype=np.float64, order="C")
    if len(matrix) != len(world_uids):
        raise ModelExperimentContractError("identity33/world row-count mismatch")
    active = active_mask(matrix)
    participating = sorted(
        {world_uids[index] for index in np.flatnonzero(active)},
        key=lambda value: value.encode("utf-8"),
    )
    if not participating:
        raise ModelExperimentContractError("Training fold has no active identity rows")
    second_moments = []
    for world_uid in participating:
        selector = np.asarray(
            [uid == world_uid and flag for uid, flag in zip(world_uids, active, strict=True)]
        )
        second_moments.append(np.mean(np.square(matrix[selector]), axis=0))
    scale = np.sqrt(np.mean(np.vstack(second_moments), axis=0))
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    world_means = []
    for world_uid in participating:
        selector = np.asarray(
            [uid == world_uid and flag for uid, flag in zip(world_uids, active, strict=True)]
        )
        world_means.append(np.mean(matrix[selector] / scale, axis=0))
    mu = np.mean(np.vstack(world_means), axis=0)
    if not np.isfinite(scale).all() or not np.isfinite(mu).all():
        raise ModelExperimentContractError("identity33 transform is non-finite")
    return scale.astype("<f8"), mu.astype("<f8")


def apply_identity_transform(
    identity33: np.ndarray,
    scale: np.ndarray,
    mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(identity33, dtype=np.float64, order="C")
    active = active_mask(matrix)
    scale_array = np.asarray(scale, dtype=np.float64)
    mu_array = np.asarray(mu, dtype=np.float64)
    if scale_array.shape != (33,) or mu_array.shape != (33,):
        raise ModelExperimentContractError("identity33 transform vector shape drift")
    phi = np.zeros_like(matrix, dtype=np.float64, order="C")
    phi[active] = matrix[active] / scale_array - mu_array
    if np.any(phi[~active] != 0.0) or not np.isfinite(phi).all():
        raise ModelExperimentContractError("Zero-history transform fallback drift")
    return phi.astype("<f8", copy=False), active


def finite_median_impute_fit(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64, order="C")
    if values.ndim != 2 or np.isinf(values).any():
        raise ModelExperimentContractError("M3 matrix shape/infinity drift")
    medians = []
    for column in range(values.shape[1]):
        finite = np.sort(values[np.isfinite(values[:, column]), column])
        if len(finite) == 0:
            raise ModelExperimentContractError(f"M3 column {column} is entirely missing")
        middle = len(finite) // 2
        value = (
            finite[middle]
            if len(finite) % 2
            else (finite[middle - 1] + finite[middle]) / 2.0
        )
        medians.append(float(value))
    result = np.asarray(medians, dtype="<f8")
    if not np.isfinite(result).all():
        raise ModelExperimentContractError("M3 imputation medians are non-finite")
    return result


def impute_with_medians(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64, order="C").copy()
    fill = np.asarray(medians, dtype=np.float64)
    if values.ndim != 2 or fill.shape != (values.shape[1],) or np.isinf(values).any():
        raise ModelExperimentContractError("M3 imputation input drift")
    rows, columns = np.where(np.isnan(values))
    values[rows, columns] = fill[columns]
    if not np.isfinite(values).all():
        raise ModelExperimentContractError("M3 imputation produced non-finite values")
    return values.astype("<f8", copy=False)


def select_shared_l2(loss_by_lambda: Mapping[float, float]) -> float:
    if not loss_by_lambda:
        raise ModelExperimentContractError("L2 selection has no candidates")
    values: dict[float, float] = {}
    for raw_key, raw_value in loss_by_lambda.items():
        key = float(raw_key)
        if not math.isfinite(key) or key in values:
            raise ModelExperimentContractError("L2 selection has duplicate/invalid keys")
        values[key] = float(raw_value)
    if set(values) != set(FROZEN_L2_GRID) or len(values) != len(FROZEN_L2_GRID):
        raise ModelExperimentContractError("L2 selection grid is not the frozen nine values")
    if not all(math.isfinite(value) for value in values.values()):
        raise ModelExperimentContractError("L2 selection loss is non-finite")
    minimum = min(values.values())
    return max(key for key, value in values.items() if value <= minimum + 1e-12)


def matrix_value_sha256(matrix: np.ndarray) -> str:
    values = np.asarray(matrix, dtype="<f8", order="C")
    if not values.flags.c_contiguous or values.dtype.str != "<f8":
        raise AssertionError("Matrix canonicalization failed")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def require_unique_ordered_rows(
    rows: Iterable[Mapping[str, Any]],
    key: str,
) -> list[Mapping[str, Any]]:
    output = list(rows)
    values = [str(row[key]) for row in output]
    if len(values) != len(set(values)):
        raise ModelExperimentContractError(f"Duplicate row key: {key}")
    return output
