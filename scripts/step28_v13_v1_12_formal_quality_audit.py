#!/usr/bin/env python3
"""Audit persisted v1.12 train/development or sealed Audit A/B stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score

import step28_v13_v1_12_exact_shortcut_preflight as shortcut
import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_formal_executor as executor
import step28_v13_v1_12_generate_split as generator
import step28_v13_v1_12_preceremony as preceremony


ROOT = Path(__file__).resolve().parents[1]
M0_TEXT_FIELDS = (
    "category_concat_top",
    "signature_title_concat",
    "title_concat_top",
    "signature_description_concat",
    "description_concat_top",
)
INTERNAL_UID_RE = re.compile(
    r"(?:^|[^0-9a-z])(?:w|sel|itm|ctl|ias|id|qry)_[0-9a-f]{64}(?:$|[^0-9a-z])",
    re.IGNORECASE,
)
FORBIDDEN_NATURAL_TOKENS = (
    "same_controller",
    "different_controller",
    "controller_uid",
    "mechanism_slot_uid",
    "identity_asset_uid",
)


class FormalQualityError(ValueError):
    """Raised when formal train/development cannot be published."""


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _publish_or_verify_json(
    path: Path, value: Mapping[str, Any], *, label: str
) -> None:
    """Finish a deterministic state write or verify an interrupted write."""

    payload = _json_bytes(value)
    if preceremony.exists_long_path(path):
        if preceremony.read_bytes_long_path(path) != payload:
            raise FormalQualityError(f"Existing {label} differs from replay")
        return
    preceremony.write_bytes_no_replace_long_path(path, payload)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(
        preceremony._filesystem_path(path),
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(
        preceremony._filesystem_path(path), "r", encoding="utf-8"
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormalQualityError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise FormalQualityError("Formal JSONL row is not an object")
            rows.append(value)
    if not rows:
        raise FormalQualityError(f"Formal JSONL is empty: {path}")
    return rows


def _canonical_value_hash(value: Any) -> str:
    return preceremony.canonical_sha256(value)


def _visible_document(profile: Mapping[str, Any]) -> str:
    if not set(M0_TEXT_FIELDS).issubset(profile):
        raise FormalQualityError("Seller profile lacks the frozen five M0 fields")
    return "\n".join(
        str(profile[field]).strip()
        for field in M0_TEXT_FIELDS
        if str(profile[field]).strip()
    )


def _split_rows_and_inventory(stage: Path, *, split: str) -> dict[str, Any]:
    public = stage / "public"
    private = stage / "private"
    generator.validate_stage(
        output_root=stage,
        split=split,
        world_count=500,
        design_only=False,
    )
    worlds = _read_csv(public / "observed/worlds.csv")
    sellers = _read_csv(public / "observed/sellers.csv")
    endpoints = _read_csv(
        public / "observed/complete_model_pair_endpoints.csv"
    )
    queries = _read_csv(public / "retrieval/queries.csv")
    redacted_items = _read_jsonl(public / "observed/redacted_items.jsonl")
    profiles = _read_jsonl(public / "observed/seller_profiles.jsonl")
    memberships = _read_csv(private / "oracle/controller_membership.csv")
    item_index = _read_csv(private / "audit/history_item_index.csv")
    public_labels_path = public / "supervision/classification_labels.csv"
    private_labels_path = private / "oracle/classification_labels.csv"
    if (
        preceremony.sha256_file(public_labels_path)
        != preceremony.sha256_file(private_labels_path)
    ):
        raise FormalQualityError("Public/private train-development labels differ")
    labels = _read_csv(public_labels_path)
    if (
        len(worlds) != 500
        or len(sellers) != 14000
        or len(endpoints) != 189000
        or len(queries) != 14000
        or len(profiles) != 14000
        or len(memberships) != 14000
        or len(labels) != 189000
        or sum(int(row["label"]) for row in labels) != 10000
    ):
        raise FormalQualityError(f"Formal {split} persisted count drift")

    time_by_item = {
        str(row["item_uid"]): int(row["time_bucket"]) for row in item_index
    }
    if len(time_by_item) != len(item_index) or set(time_by_item) != {
        str(row["item_uid"]) for row in redacted_items
    }:
        raise FormalQualityError("Formal item/time-index keyset drift")
    items_by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sellers_by_world: dict[str, list[str]] = defaultdict(list)
    membership_by_world: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in redacted_items:
        item_uid = str(row["item_uid"])
        items_by_world[str(row["world_uid"])].append(
            {**row, "time_bucket": time_by_item[item_uid]}
        )
    for row in sellers:
        sellers_by_world[row["world_uid"]].append(row["seller_uid"])
    for row in memberships:
        membership_by_world[row["world_uid"]].append(row)

    attack_rows: list[dict[str, Any]] = []
    for world_uid in sorted(sellers_by_world, key=lambda value: value.encode("utf-8")):
        attack_rows.extend(
            shortcut._pair_feature_rows(
                world_uid=world_uid,
                seller_uids=sellers_by_world[world_uid],
                items=items_by_world[world_uid],
                controller_membership=membership_by_world[world_uid],
            )
        )
    attack_by_pair = {
        str(row["canonical_pair_uid"]): int(row["label"])
        for row in attack_rows
    }
    label_by_pair = {
        str(row["canonical_pair_uid"]): int(row["label"]) for row in labels
    }
    endpoint_pairs = {str(row["canonical_pair_uid"]) for row in endpoints}
    if (
        len(attack_rows) != 189000
        or len(attack_by_pair) != 189000
        or attack_by_pair != label_by_pair
        or set(attack_by_pair) != endpoint_pairs
    ):
        raise FormalQualityError("Persisted shortcut/label/endpoint replay drift")

    identity_hash_document = preceremony.load_json_strict(
        private / "audit/identity_value_hashes.json"
    )
    preceremony.validate_canonical_self_hash(
        identity_hash_document,
        label=f"formal {split} identity-value hash document",
    )
    identity_hash_values = identity_hash_document.get("hashes", [])
    if not isinstance(identity_hash_values, list):
        raise FormalQualityError("Formal identity-value hash list is malformed")
    identity_hashes = set(identity_hash_values)
    expected_identity_hash_count = (
        500 * formal.IDENTITY_ASSETS_PER_WORLD[split]
    )
    if (
        int(identity_hash_document.get("hash_count", -1))
        != expected_identity_hash_count
        or len(identity_hash_values) != expected_identity_hash_count
        or len(identity_hashes) != expected_identity_hash_count
        or identity_hash_values != sorted(identity_hash_values)
        or any(
            preceremony.HEX_SHA256_RE.fullmatch(str(value)) is None
            for value in identity_hash_values
        )
    ):
        raise FormalQualityError("Formal identity-value hash closure drift")

    natural_texts: list[str] = []
    item_document_hashes: set[str] = set()
    for row in redacted_items:
        title = str(row["title"])
        description = str(row["description"])
        natural_texts.extend((title, description))
        item_document_hashes.add(
            _canonical_value_hash(
                {"title": title, "description": description}
            )
        )
    seller_documents: dict[str, str] = {}
    seller_document_hashes: set[str] = set()
    for profile in profiles:
        seller_uid = str(profile["seller_uid"])
        document = _visible_document(profile)
        seller_documents[seller_uid] = document
        seller_document_hashes.add(hashlib.sha256(document.encode("utf-8")).hexdigest())
        natural_texts.append(document)
    leakage_count = sum(
        1
        for value in natural_texts
        if INTERNAL_UID_RE.search(value)
        or any(token in value.casefold() for token in FORBIDDEN_NATURAL_TOKENS)
    )
    if leakage_count != 0:
        raise FormalQualityError("Internal UID/label token leaked into M0 text")

    return {
        "attack_rows": attack_rows,
        "labels": labels,
        "endpoints": endpoints,
        "seller_documents": seller_documents,
        "sets": {
            "world_uid": {row["world_uid"] for row in worlds},
            "seller_uid": {row["seller_uid"] for row in sellers},
            "item_uid": {str(row["item_uid"]) for row in redacted_items},
            "canonical_pair_uid": endpoint_pairs,
            "query_uid": {row["query_uid"] for row in queries},
            "controller_uid": {row["controller_uid"] for row in memberships},
            "identity_value_hash": identity_hashes,
            "item_document_hash": item_document_hashes,
            "seller_document_hash": seller_document_hashes,
        },
        "counts": {
            "worlds": len(worlds),
            "sellers": len(sellers),
            "items": len(redacted_items),
            "pairs": len(endpoints),
            "positives": sum(int(row["label"]) for row in labels),
            "queries": len(queries),
            "identity_value_hashes": len(identity_hashes),
            "m0_text_leakage_count": leakage_count,
        },
    }


def _old_release_inventory(draft: Mapping[str, Any]) -> dict[str, set[str]]:
    spec = draft["historical_success_release"]
    manifest_path = preceremony.verify_file_pin(
        spec, label="pinned historical v1.2 release manifest"
    )
    manifest = preceremony.load_json_strict(manifest_path)
    preceremony.validate_canonical_self_hash(
        manifest, label="historical v1.2 release manifest"
    )
    if (
        manifest.get("status") != "PASS_DATASET_ONLY_READY_FOR_M0_M1_M2"
        or manifest.get("run_id")
        != "v13_training_ready_v1_2_order_repair_20260731"
        or manifest.get("canonical_self_hash") != spec["canonical_self_hash"]
    ):
        raise FormalQualityError("Historical v1.2 release manifest drift")
    root = manifest_path.parent
    output: dict[str, set[str]] = {
        "world_uid": set(),
        "seller_uid": set(),
        "item_uid": set(),
        "canonical_pair_uid": set(),
        "item_document_hash": set(),
        "seller_document_hash": set(),
    }
    for split in formal.SPLITS:
        public = root / split
        for row in _read_csv(public / "observed/worlds.csv"):
            output["world_uid"].add(row["world_uid"])
        for row in _read_csv(public / "observed/sellers.csv"):
            output["seller_uid"].add(row["seller_uid"])
        for row in _read_csv(
            public / "observed/complete_model_pair_endpoints.csv"
        ):
            output["canonical_pair_uid"].add(row["canonical_pair_uid"])
        for row in _read_jsonl(public / "observed/redacted_items.jsonl"):
            output["item_uid"].add(str(row["item_uid"]))
            output["item_document_hash"].add(
                _canonical_value_hash(
                    {
                        "title": str(row["title"]),
                        "description": str(row["description"]),
                    }
                )
            )
        for profile in _read_jsonl(public / "observed/seller_profiles.jsonl"):
            document = _visible_document(profile)
            output["seller_document_hash"].add(
                hashlib.sha256(document.encode("utf-8")).hexdigest()
            )
    return output


def _char_trigram_similarity_diagnostic(
    inventory: Mapping[str, Any], *, split: str
) -> dict[str, Any]:
    documents = inventory["seller_documents"]
    seller_uids = sorted(documents, key=lambda value: value.encode("utf-8"))
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 3),
        n_features=1 << 16,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
    )
    matrix = vectorizer.transform([documents[uid] for uid in seller_uids])
    index = {uid: position for position, uid in enumerate(seller_uids)}
    scores = np.asarray(
        [
            float(
                matrix[index[row["seller_uid_left"]]].multiply(
                    matrix[index[row["seller_uid_right"]]]
                ).sum()
            )
            for row in inventory["endpoints"]
        ],
        dtype=np.float64,
    )
    label_by_pair = {
        str(row["canonical_pair_uid"]): int(row["label"])
        for row in inventory["labels"]
    }
    labels = np.asarray(
        [label_by_pair[row["canonical_pair_uid"]] for row in inventory["endpoints"]],
        dtype=np.int8,
    )
    auc = float(roc_auc_score(labels, scores))
    return {
        "split": split,
        "probe": "fixed_hashing_char_trigram_cosine_diagnostic_only",
        "feature_dimension": 1 << 16,
        "roc_auc": auc,
        "symmetric_roc_auc": max(auc, 1.0 - auc),
        "average_precision": float(average_precision_score(labels, scores)),
        "random_ap_baseline": 20.0 / 378.0,
        "formal_gate": False,
        "reason_not_a_gate": (
            "legitimate visible authorship/style is an intended frozen-M0 input"
        ),
    }


def _validate_m1_receipts(
    *, stage: Path, lock: Mapping[str, Any], lock_path: Path
) -> dict[str, Any]:
    receipt_hashes: list[str] = []
    for replicate in range(1, 6):
        role = f"m1_r{replicate:02d}"
        start_path = executor._m1_start_path(lock_path, role)
        executor._validate_exact_marker(
            start_path,
            executor._expected_m1_start(
                role=role, lock_path=lock_path, lock=lock
            ),
            label=f"formal {role} materialization start receipt",
        )
        path = (
            stage
            / "private"
            / "m1"
            / f"r{replicate:02d}"
            / "structural_receipt.json"
        )
        receipt = preceremony.load_json_strict(path)
        preceremony.validate_canonical_self_hash(
            receipt, label=f"formal M1 receipt {role}"
        )
        if (
            receipt.get("status") != "PASS_FORMAL_M1_STRUCTURAL_REPLAY"
            or receipt.get("replicate") != f"r{replicate:02d}"
            or int(receipt.get("world_count", -1)) != 500
            or int(receipt.get("pair_count", -1)) != 189000
            or int(receipt.get("mapping_count", -1)) != 189000
            or int(receipt.get("fixed_point_count", -1)) != 0
            or int(receipt.get("endpoint_overlap_count", -1)) != 0
            or receipt.get("whole_identity33_multiset_preserved") is not True
            or receipt.get("rewire_key_commitment")
            != lock["m1_capability_commitments"][role]
            or receipt.get("raw_rewire_key_persisted") is not False
            or receipt.get("authority_reference", {}).get("m1_start_receipt")
            != executor._repo_pin(start_path, include_self_hash=True)
        ):
            raise FormalQualityError(f"Formal M1 receipt drift: {role}")
        receipt_hashes.append(preceremony.sha256_file(path))
    return {
        "replicate_count": 5,
        "all_world_counts": 500,
        "all_pair_counts": 189000,
        "all_fixed_point_counts": 0,
        "all_endpoint_overlap_counts": 0,
        "all_whole_row_multisets_preserved": True,
        "receipt_hashes_sha256": preceremony.canonical_sha256(receipt_hashes),
    }


def run_quality_audit(
    *,
    execution_lock_path: Path = formal.DEFAULT_EXECUTION_LOCK_PATH,
    output: Path,
) -> dict[str, Any]:
    validated = formal.load_and_validate_execution_lock(execution_lock_path)
    draft = validated["draft"]
    canonical_output = preceremony._repo_path(
        str(
            validated["prelock"]["custody"][
                "train_development_quality_receipt_path"
            ]
        )
    )
    formal.require_canonical_path(
        output,
        canonical_output,
        label="v1.12 train/development quality receipt",
    )
    inventories: dict[str, dict[str, Any]] = {}
    stages: dict[str, Path] = {}
    for split in ("train", "development"):
        paths = executor._paths(draft, split)
        executor._validate_marker(
            paths["finalized_marker"],
            status="PASS_FORMAL_STAGE_FINALIZED",
            split=split,
        )
        stages[split] = paths["stage"]
        inventories[split] = _split_rows_and_inventory(
            paths["stage"], split=split
        )

    cross_split_intersections = {
        name: len(
            inventories["train"]["sets"][name]
            & inventories["development"]["sets"][name]
        )
        for name in inventories["train"]["sets"]
    }
    if any(cross_split_intersections.values()):
        raise FormalQualityError("Formal train/development isolation failed")
    historical = _old_release_inventory(draft)
    fresh_union = {
        name: inventories["train"]["sets"][name]
        | inventories["development"]["sets"][name]
        for name in historical
    }
    historical_intersections = {
        name: len(fresh_union[name] & historical[name]) for name in historical
    }
    failed_hash_intersection = len(
        (
            inventories["train"]["sets"]["identity_value_hash"]
            | inventories["development"]["sets"]["identity_value_hash"]
        )
        & set(validated["baseline"]["failed_identity_hashes"])
    )
    if any(historical_intersections.values()) or failed_hash_intersection:
        raise FormalQualityError("Fresh/historical exact-intersection gate failed")

    evaluation = shortcut.evaluate_exact_shortcut_rows(
        train_rows=inventories["train"]["attack_rows"],
        development_rows=inventories["development"]["attack_rows"],
        config=draft["shortcut_preflight"],
    )
    if not all(evaluation["gates"].values()):
        raise FormalQualityError("Formal persisted shortcut gate failed")
    m1 = _validate_m1_receipts(
        stage=stages["train"],
        lock=validated["execution_lock"],
        lock_path=execution_lock_path,
    )
    char_diagnostics = {
        split: _char_trigram_similarity_diagnostic(
            inventories[split], split=split
        )
        for split in ("train", "development")
    }
    report = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-formal-quality-v1",
            "status": "PASS_FORMAL_TRAIN_DEVELOPMENT_QUALITY_GATE",
            "run_id": draft["run_id"],
            "execution_lock": executor._repo_pin(
                execution_lock_path, include_self_hash=True
            ),
            "producer_path": (
                "scripts/step28_v13_v1_12_formal_quality_audit.py"
            ),
            "producer_sha256": preceremony.sha256_file(Path(__file__)),
            "split_counts": {
                split: inventories[split]["counts"]
                for split in ("train", "development")
            },
            "cross_split_intersection_counts": cross_split_intersections,
            "historical_v1_2_intersection_counts": historical_intersections,
            "failed_identity_hash_intersection_count": failed_hash_intersection,
            "shortcut_evaluation": evaluation,
            "m1_structural_replays": m1,
            "visible_char_trigram_diagnostics": char_diagnostics,
            "c40_generated_or_read": False,
            "audit_a_or_b_truth_read": False,
            "model_training_or_scientific_evaluation_started": False,
            "deterministic_replay_without_runtime_field": True,
        }
    )
    _publish_or_verify_json(output, report, label="formal quality receipt")
    output_pin = executor._repo_pin(output, include_self_hash=True)
    for split in ("train", "development"):
        paths = executor._paths(draft, split)
        marker = preceremony.with_canonical_self_hash(
            {
                "version": "2026-08-03-step28-v13-v1-12-quality-marker-v1",
                "status": "PASS_FORMAL_STAGE_QUALITY",
                "split": split,
                "quality_receipt": output_pin,
                "stage_published": False,
            }
        )
        _publish_or_verify_json(
            paths["quality_marker"],
            marker,
            label=f"formal {split} quality marker",
        )
    return report


def _sealed_public_inventory(
    *, public: Path, private: Path, split: str
) -> dict[str, Any]:
    worlds = _read_csv(public / "observed/worlds.csv")
    sellers = _read_csv(public / "observed/sellers.csv")
    endpoints = _read_csv(
        public / "observed/complete_model_pair_endpoints.csv"
    )
    queries = _read_csv(public / "retrieval/queries.csv")
    items = _read_jsonl(public / "observed/redacted_items.jsonl")
    profiles = _read_jsonl(public / "observed/seller_profiles.jsonl")
    identity_document = preceremony.load_json_strict(
        private / "audit/identity_value_hashes.json"
    )
    preceremony.validate_canonical_self_hash(
        identity_document, label=f"sealed {split} identity hash document"
    )
    identity_hash_values = identity_document.get("hashes", [])
    if not isinstance(identity_hash_values, list):
        raise FormalQualityError(f"Sealed {split} identity hash list is malformed")
    identity_hashes = set(identity_hash_values)
    expected_identity_hash_count = (
        500 * formal.IDENTITY_ASSETS_PER_WORLD[split]
    )
    if (
        len(worlds) != 500
        or len(sellers) != 14000
        or len(endpoints) != 189000
        or len(queries) != 14000
        or len(profiles) != 14000
        or int(identity_document.get("hash_count", -1))
        != expected_identity_hash_count
        or len(identity_hash_values) != expected_identity_hash_count
        or len(identity_hashes) != expected_identity_hash_count
        or identity_hash_values != sorted(identity_hash_values)
        or any(
            preceremony.HEX_SHA256_RE.fullmatch(str(value)) is None
            for value in identity_hash_values
        )
    ):
        raise FormalQualityError(f"Sealed {split} public count drift")
    item_hashes: set[str] = set()
    seller_hashes: set[str] = set()
    leakage_count = 0
    for row in items:
        title = str(row["title"])
        description = str(row["description"])
        item_hashes.add(
            _canonical_value_hash({"title": title, "description": description})
        )
        for value in (title, description):
            leakage_count += int(
                INTERNAL_UID_RE.search(value) is not None
                or any(
                    token in value.casefold()
                    for token in FORBIDDEN_NATURAL_TOKENS
                )
            )
    for profile in profiles:
        document = _visible_document(profile)
        seller_hashes.add(hashlib.sha256(document.encode("utf-8")).hexdigest())
        leakage_count += int(
            INTERNAL_UID_RE.search(document) is not None
            or any(
                token in document.casefold() for token in FORBIDDEN_NATURAL_TOKENS
            )
        )
    if leakage_count:
        raise FormalQualityError(f"Sealed {split} M0 text leakage")
    return {
        "world_uid": {row["world_uid"] for row in worlds},
        "seller_uid": {row["seller_uid"] for row in sellers},
        "item_uid": {str(row["item_uid"]) for row in items},
        "canonical_pair_uid": {
            row["canonical_pair_uid"] for row in endpoints
        },
        "query_uid": {row["query_uid"] for row in queries},
        "identity_value_hash": identity_hashes,
        "item_document_hash": item_hashes,
        "seller_document_hash": seller_hashes,
        "counts": {
            "worlds": len(worlds),
            "sellers": len(sellers),
            "items": len(items),
            "pairs": len(endpoints),
            "queries": len(queries),
            "identity_value_hashes": len(identity_hashes),
            "m0_text_leakage_count": leakage_count,
        },
    }


def run_audit_split_quality(
    *, audit_lock_path: Path, split: str, output: Path
) -> dict[str, Any]:
    if split not in {"audit_a", "audit_b"}:
        raise FormalQualityError("Sealed quality audit requires Audit A/B")
    validated = formal.load_and_validate_audit_lock(audit_lock_path)
    draft = validated["draft"]
    if (
        validated["audit_lock"]["authorizations"][
            f"formal_{split}_generation"
        ]
        is not True
    ):
        raise FormalQualityError(f"Sealed {split} quality is not authorized")
    canonical_output = audit_lock_path.parent / f"{split}_quality_gate.json"
    formal.require_canonical_path(
        output,
        canonical_output,
        label=f"v1.12 {split} quality receipt",
    )
    paths = executor._paths(draft, split)
    executor._validate_marker(
        paths["finalized_marker"],
        status="PASS_FORMAL_STAGE_FINALIZED",
        split=split,
    )
    generator.validate_stage(
        output_root=paths["stage"],
        split=split,
        world_count=500,
        design_only=False,
    )
    current = _sealed_public_inventory(
        public=paths["stage"] / "public",
        private=paths["stage"] / "private",
        split=split,
    )
    comparison_splits = ["train", "development"]
    if split == "audit_b":
        comparison_splits.append("audit_a")
    comparisons: dict[str, dict[str, Any]] = {}
    for other in comparison_splits:
        other_paths = executor._paths(draft, other)
        _published = executor._validate_published_split(
            public_root=other_paths["public_final"],
            private_root=other_paths["private_final"],
            split=other,
            draft=draft,
        )
        inventory = _sealed_public_inventory(
            public=other_paths["public_final"],
            private=other_paths["private_final"],
            split=other,
        )
        counts = {
            name: len(current[name] & inventory[name])
            for name in (
                "world_uid",
                "seller_uid",
                "item_uid",
                "canonical_pair_uid",
                "query_uid",
                "identity_value_hash",
                "item_document_hash",
                "seller_document_hash",
            )
        }
        if any(counts.values()):
            raise FormalQualityError(
                f"Sealed {split}/{other} exact isolation failed"
            )
        comparisons[other] = counts
    historical = _old_release_inventory(draft)
    historical_intersections = {
        name: len(current[name] & historical[name]) for name in historical
    }
    failed_identity_intersection = len(
        current["identity_value_hash"]
        & set(validated["baseline"]["failed_identity_hashes"])
    )
    if any(historical_intersections.values()) or failed_identity_intersection:
        raise FormalQualityError(f"Sealed {split} historical isolation failed")
    report = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-sealed-audit-quality-v1",
            "status": "PASS_FORMAL_SEALED_AUDIT_SPLIT_QUALITY",
            "run_id": draft["run_id"],
            "split": split,
            "audit_lock": executor._repo_pin(
                audit_lock_path, include_self_hash=True
            ),
            "producer_path": (
                "scripts/step28_v13_v1_12_formal_quality_audit.py"
            ),
            "producer_sha256": preceremony.sha256_file(Path(__file__)),
            "split_counts": current["counts"],
            "cross_split_intersection_counts": comparisons,
            "historical_v1_2_intersection_counts": historical_intersections,
            "failed_identity_hash_intersection_count": failed_identity_intersection,
            "private_manifest_bytes_hashed_but_truth_rows_parsed": False,
            "classification_labels_parsed": False,
            "retrieval_qrels_parsed": False,
            "controller_membership_parsed": False,
            "c40_generated_or_read": False,
            "model_training_or_prediction_started": False,
            "deterministic_replay_without_runtime_field": True,
        }
    )
    _publish_or_verify_json(
        output, report, label=f"formal sealed {split} quality receipt"
    )
    marker = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-quality-marker-v1",
            "status": "PASS_FORMAL_STAGE_QUALITY",
            "split": split,
            "quality_receipt": executor._repo_pin(
                output, include_self_hash=True
            ),
            "stage_published": False,
        }
    )
    _publish_or_verify_json(
        paths["quality_marker"],
        marker,
        label=f"formal {split} quality marker",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-lock",
        type=Path,
        default=None,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-split", choices=("audit_a", "audit_b"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_split is None:
        lock_path = (
            args.execution_lock.resolve()
            if args.execution_lock is not None
            else formal.DEFAULT_EXECUTION_LOCK_PATH
        )
        report = run_quality_audit(
            execution_lock_path=lock_path,
            output=args.output.resolve(),
        )
        print(
            report["status"],
            report["shortcut_evaluation"]["train_row_count"],
            report["shortcut_evaluation"]["development_row_count"],
            preceremony.sha256_file(args.output.resolve()),
        )
    else:
        lock_path = (
            args.execution_lock.resolve()
            if args.execution_lock is not None
            else (
                formal.DEFAULT_AUDIT_A_LOCK_PATH
                if args.audit_split == "audit_a"
                else formal.DEFAULT_AUDIT_B_LOCK_PATH
            )
        )
        report = run_audit_split_quality(
            audit_lock_path=lock_path,
            split=args.audit_split,
            output=args.output.resolve(),
        )
        print(
            report["status"],
            report["split"],
            report["split_counts"]["pairs"],
            preceremony.sha256_file(args.output.resolve()),
        )


if __name__ == "__main__":
    main()
