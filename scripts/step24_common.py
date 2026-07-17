#!/usr/bin/env python3
"""Shared contracts for Step24 content-independent authorship experiments."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import step15_build_v7_clean_embedding_cache as redaction
import step7_build_semantic_pair_features as semantic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step24_content_independent_authorship_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def bool_value(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def render_csv(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("Step24 refuses to render an empty CSV")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("Step24 CSV rows have inconsistent field order")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def write_bytes_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite a different Step24 artifact: {path}")
        return
    path.write_bytes(payload)


def write_csv_immutable(path: Path, rows: list[dict]) -> None:
    write_bytes_immutable(path, render_csv(rows))


def write_json_immutable(path: Path, payload: dict) -> None:
    rendered = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    write_bytes_immutable(path, rendered)


def write_npy_immutable(path: Path, matrix: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(matrix))
    write_bytes_immutable(path, buffer.getvalue())


def directory_fingerprint(path: Path) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"Step24 model directory is missing: {path}")
    records = []
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative_path = file_path.relative_to(path)
        if ".cache" in relative_path.parts:
            continue
        records.append(
            {
                "path": str(relative_path).replace("\\", "/"),
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    if not records:
        raise ValueError(f"Step24 model directory is empty: {path}")
    return {
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files_sha256": canonical_hash(records),
    }


def validate_policy(policy: dict) -> None:
    text_cfg = policy["clean_text_contract"]
    pair_cfg = policy["pair_features"]
    eval_cfg = policy["evaluation"]
    if text_cfg["encode_split"] != "train" or text_cfg["encode_valid_or_test"]:
        raise ValueError("Step24-v1 must encode canonical train sellers only")
    if not text_cfg["require_exact_v7_corpus_hash_replay"]:
        raise ValueError("Step24 requires exact replay of the v7 redacted corpus")
    forbidden_flags = (
        "identifiers_forbidden",
        "candidate_rule_features_forbidden",
        "random_projection_forbidden",
        "item_distribution_features_forbidden",
        "valid_or_test_fitted_statistics_forbidden",
    )
    if any(not pair_cfg.get(flag) for flag in forbidden_flags):
        raise ValueError("Step24 clean-feature isolation was relaxed")
    if not eval_cfg["candidate_selection_forbidden"] or not eval_cfg[
        "valid_or_test_selection_forbidden"
    ]:
        raise ValueError("Step24 model selection contract was relaxed")
    expected_feature_sets = {
        "e5_lr_l2_control": ["identifier_redacted_e5_cosine"],
        "style_only_lr_l2_control": [
            "pcm_multilingual_authorship_cosine",
            "mstyledistance_cosine",
        ],
        "semantic_style_lr_l2_primary": [
            "identifier_redacted_e5_cosine",
            "pcm_multilingual_authorship_cosine",
            "mstyledistance_cosine",
        ],
    }
    if eval_cfg["model_feature_sets"] != expected_feature_sets:
        raise ValueError("Step24 model/control matrix differs from preregistration")
    primary = eval_cfg["primary_model"]
    if primary != "semantic_style_lr_l2_primary":
        raise ValueError("Step24 primary feature set differs from preregistration")
    if eval_cfg["matched_baseline_model"] != "e5_lr_l2_control":
        raise ValueError("Step24 matched baseline must remain the E5-only LR/L2 control")
    if eval_cfg["raw_score_controls"] != [
        "identifier_redacted_e5_cosine",
        "pcm_multilingual_authorship_cosine",
        "mstyledistance_cosine",
    ]:
        raise ValueError("Step24 raw controls differ from preregistration")
    logistic = eval_cfg["logistic"]
    if (
        float(logistic["l2_penalty"]) != 10.0
        or logistic["class_weight"] != "none"
        or not logistic["standardize_features"]
    ):
        raise ValueError("Step24 fixed LR/L2 contract was changed")
    if int(eval_cfg["fold_count"]) != 5 or eval_cfg["canonical_split"] != "train":
        raise ValueError("Step24 train-only five-fold contract was changed")
    expected_gates = {
        "minimum_target_oof_ap_gain_over_e5_control": 0.03,
        "minimum_target_oof_grouped_bootstrap_lower_bound": 0.0,
        "minimum_source_only_ap_gain_over_e5_control": 0.02,
        "minimum_source_only_grouped_bootstrap_lower_bound": 0.0,
        "minimum_source_only_non_silver_ap_delta": -0.01,
        "minimum_source_only_direct_component_plus_all_negatives_ap_delta": 0.0,
        "minimum_non_silver_ap_delta": -0.01,
        "minimum_direct_component_plus_all_negatives_ap_delta": 0.0,
        "minimum_direct_component_positive_mean_score_delta": -0.03,
        "maximum_template_or_topic_mean_score_increase": 0.02,
        "maximum_template_or_topic_q95_score_increase": 0.02,
        "maximum_template_or_topic_top_decile_mean_score_increase": 0.02,
    }
    gates = policy["promotion_gates"]
    if any(float(gates.get(key, math.nan)) != value for key, value in expected_gates.items()):
        raise ValueError("Step24 promotion gates differ from preregistration")
    for encoder in policy["frozen_style_encoders"].values():
        revision = str(encoder.get("revision", ""))
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError("Step24 encoder revision must be a pinned 40-character commit")
        if not encoder["local_finetuning_forbidden"]:
            raise ValueError("Step24-v1 does not permit local encoder fine-tuning")
        if not encoder["inference_time_content_masking_forbidden"]:
            raise ValueError("Step24-v1 must use the released encoder inference contract")


def validate_model_provenance(model_path: Path, encoder_cfg: dict) -> dict:
    provenance_path = model_path / "step24_model_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Step24 model provenance is missing; use the pinned Windows downloader: {provenance_path}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "repo_id": encoder_cfg["repo_id"],
        "requested_revision": encoder_cfg["revision"],
        "resolved_revision": encoder_cfg["revision"],
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"Step24 model provenance mismatch for {encoder_cfg['repo_id']}: {key}"
            )
    return provenance


def load_component_assignments(policy: dict) -> dict[tuple[str, str], dict]:
    rows = load_csv(resolve(policy["inputs"]["component_assignments"]))
    result = {}
    for row in rows:
        key = (row["dataset"], row["pair_uid"])
        if key in result:
            raise ValueError(f"Duplicate Step24 component assignment: {key}")
        if row["cross_split_component_leakage"] == "1" or row["cross_split_seller_leakage"] == "1":
            raise ValueError(f"Step24 refuses a leaking component assignment: {key}")
        result[key] = row
    return result


def load_canonical_train_rows(policy: dict) -> dict[str, list[dict]]:
    eval_cfg = policy["evaluation"]
    assignments = load_component_assignments(policy)
    output = {}
    for pool_name, pool_cfg in policy["pools"].items():
        labels = load_csv(resolve(pool_cfg["frozen_labels"]))
        evidence_rows = load_csv(resolve(pool_cfg["evidence_labels"]))
        evidence = {row["pair_uid"]: row for row in evidence_rows}
        if len(evidence) != len(evidence_rows):
            raise ValueError(f"Step24 found duplicate evidence labels: {pool_name}")
        rows = []
        selected_pair_uids = set()
        for label in labels:
            if label.get("split_name") != eval_cfg["canonical_split"]:
                continue
            if label.get("review_label") not in {"positive", "negative"}:
                continue
            if eval_cfg["require_usable_for_supervision"] and not bool_value(
                label.get("usable_for_supervision")
            ):
                continue
            if eval_cfg["require_usable_for_core_transfer"] and not bool_value(
                label.get("usable_for_core_transfer")
            ):
                continue
            pair_uid = label["pair_uid"]
            if pair_uid in selected_pair_uids:
                raise ValueError(f"Step24 found duplicate canonical label: {pool_name}:{pair_uid}")
            selected_pair_uids.add(pair_uid)
            evidence_row = evidence.get(pair_uid)
            if evidence_row is None:
                raise ValueError(f"Step24 evidence label is missing: {pool_name}:{pair_uid}")
            if eval_cfg["require_identity_training_eligible"] and not bool_value(
                evidence_row.get("identity_training_eligible")
            ):
                continue
            assignment = assignments.get((pool_name, pair_uid))
            if assignment is None:
                raise ValueError(f"Step24 component assignment is missing: {pool_name}:{pair_uid}")
            if assignment["split_name"] != "train":
                raise ValueError(f"Step24 canonical/component split mismatch: {pool_name}:{pair_uid}")
            rows.append(
                {
                    **label,
                    "domain": pool_cfg["domain"],
                    "step24_pool": pool_name,
                    "evidence_type": evidence_row["evidence_type"],
                    "evidence_type_confident": evidence_row["evidence_type_confident"],
                    "v7_component_id": assignment["recomputed_component_id"],
                    "step24_component_id": assignment["recomputed_component_id"],
                }
            )
        if not rows or {row["review_label"] for row in rows} != {"positive", "negative"}:
            raise ValueError(f"Step24 canonical train pool is empty or single-class: {pool_name}")
        output[pool_name] = sorted(rows, key=lambda row: row["pair_uid"])
    return output


def train_sellers(rows: list[dict]) -> list[str]:
    return sorted(
        {
            row[field]
            for row in rows
            for field in ("seller_uid_left", "seller_uid_right")
            if row.get(field)
        }
    )


def replay_v7_clean_texts(pool_cfg: dict, text_cfg: dict, seller_uids: list[str]) -> tuple[list[str], dict]:
    profiles_path = resolve(pool_cfg["seller_profiles"])
    signals_path = resolve(pool_cfg["item_identity_signals"])
    profiles_list = semantic.load_jsonl(profiles_path)
    profiles = {str(row["seller_uid"]): row for row in profiles_list}
    if len(profiles) != len(profiles_list):
        raise ValueError(f"Step24 found duplicate seller profiles: {profiles_path}")
    missing = sorted(set(seller_uids) - set(profiles))
    if missing:
        raise ValueError(f"Step24 seller profile is missing: {missing[0]}")
    literals, signal_diagnostics = redaction.signal_literals_by_seller(signals_path)
    clean_texts = []
    counts: Counter[str] = Counter()
    for seller_uid in seller_uids:
        source_text = redaction.build_content_text(
            profiles[seller_uid], {"text_fields": text_cfg["text_fields"]}
        )
        seller_literals = list(literals.get(seller_uid, []))
        for alias_field in ("source_seller_raw", "alias_normalized"):
            alias_literal = redaction.safe_signal_literal(
                "seller_alias", profiles[seller_uid].get(alias_field, "")
            )
            if alias_literal:
                seller_literals.append(alias_literal)
        seller_literals = sorted(
            set(seller_literals), key=lambda value: (-len(value), value.casefold())
        )
        clean_text, diagnostics = redaction.redact_identifiers(source_text, seller_literals)
        redaction.assert_no_known_identifier_residue(clean_text, seller_literals, seller_uid)
        counts.update(diagnostics)
        if not clean_text:
            clean_text = text_cfg["empty_text_fallback"]
            counts["empty_after_redaction_count"] += 1
        clean_texts.append(clean_text)
    return clean_texts, {
        **signal_diagnostics,
        "seller_count": len(seller_uids),
        "generic_identifier_match_count": counts["generic_identifier_match_count"],
        "signal_literal_match_count": counts["signal_literal_match_count"],
        "empty_after_redaction_count": counts["empty_after_redaction_count"],
        "clean_text_corpus_sha256": canonical_hash(
            list(zip(seller_uids, clean_texts, strict=True))
        ),
    }


def load_normalized_cache(metadata_path: Path, matrix_path: Path) -> tuple[dict[str, int], np.ndarray, dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matrix = np.load(matrix_path, mmap_mode="r")
    seller_uids = list(metadata.get("seller_uids", []))
    if list(matrix.shape) != list(metadata.get("shape", [])):
        raise ValueError(f"Step24 cache shape mismatch: {matrix_path}")
    if len(seller_uids) != matrix.shape[0] or len(set(seller_uids)) != len(seller_uids):
        raise ValueError(f"Step24 cache seller index mismatch: {metadata_path}")
    norms = np.linalg.norm(np.asarray(matrix, dtype=np.float32), axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-3:
        raise ValueError(f"Step24 cache is not unit-normalized: {matrix_path}")
    return {uid: index for index, uid in enumerate(seller_uids)}, matrix, metadata


def balanced_component_folds(rows: list[dict], fold_count: int, seed: int) -> dict[str, int]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["step24_component_id"]].append(row)
    if len(grouped) < fold_count:
        raise ValueError("Step24 has fewer target components than folds")
    totals = (
        len(rows) / fold_count,
        sum(row["review_label"] == "positive" for row in rows) / fold_count,
        sum(row["review_label"] == "negative" for row in rows) / fold_count,
    )
    records = []
    for component_id, component_rows in grouped.items():
        positives = sum(row["review_label"] == "positive" for row in component_rows)
        negatives = len(component_rows) - positives
        mass = (
            len(component_rows) / max(totals[0], 1.0)
            + positives / max(totals[1], 1.0)
            + negatives / max(totals[2], 1.0)
        )
        tie = hashlib.sha256(f"{seed}|{component_id}".encode("utf-8")).hexdigest()
        records.append((component_id, len(component_rows), positives, negatives, mass, tie))
    records.sort(key=lambda item: (-item[4], item[5], item[0]))
    positive_component_count = sum(record[2] > 0 for record in records)
    negative_component_count = sum(record[3] > 0 for record in records)
    if positive_component_count < fold_count or negative_component_count < fold_count:
        raise ValueError(
            "Step24 cannot create double-class held-out folds from the available "
            f"components: positive_components={positive_component_count} "
            f"negative_components={negative_component_count} folds={fold_count}"
        )

    fold_totals = [[0, 0, 0] for _ in range(fold_count)]
    assignment = {}

    def assign(record: tuple, fold: int) -> None:
        component_id, count, positives, negatives, _mass, _tie = record
        assignment[component_id] = fold
        fold_totals[fold][0] += count
        fold_totals[fold][1] += positives
        fold_totals[fold][2] += negatives

    # Seed every fold with both classes before optimizing balance. Without this
    # constraint, minimizing one fold at a time can fill early folds completely
    # and leave the last fold empty even when a valid grouped split exists.
    remaining = list(records)
    dual_class = [record for record in remaining if record[2] > 0 and record[3] > 0]
    for fold in range(min(fold_count, len(dual_class))):
        record = dual_class[fold]
        assign(record, fold)
        remaining.remove(record)

    for fold in range(fold_count):
        if fold_totals[fold][1] == 0:
            candidate = next((record for record in remaining if record[2] > 0), None)
            if candidate is None:
                raise ValueError(f"Step24 cannot seed a positive component in fold {fold}")
            assign(candidate, fold)
            remaining.remove(candidate)
        if fold_totals[fold][2] == 0:
            candidate = next((record for record in remaining if record[3] > 0), None)
            if candidate is None:
                raise ValueError(f"Step24 cannot seed a negative component in fold {fold}")
            assign(candidate, fold)
            remaining.remove(candidate)

    for component_id, count, positives, negatives, mass, tie in remaining:
        candidate_scores = []
        for fold in range(fold_count):
            projected_totals = [list(values) for values in fold_totals]
            projected_totals[fold][0] += count
            projected_totals[fold][1] += positives
            projected_totals[fold][2] += negatives
            global_error = sum(
                ((projected_totals[candidate_fold][index] - totals[index])
                 / max(totals[index], 1.0))
                ** 2
                for candidate_fold in range(fold_count)
                for index in range(3)
            )
            candidate_scores.append((global_error, fold_totals[fold][0], fold))
        selected = min(candidate_scores)[2]
        assign((component_id, count, positives, negatives, mass, tie), selected)
    for fold, (count, positives, negatives) in enumerate(fold_totals):
        if not count or not positives or not negatives:
            raise ValueError(
                f"Step24 grouped fold {fold} is empty or single-class: "
                f"rows={count} positives={positives} negatives={negatives}"
            )
    return assignment


def negative_tail_metrics(scores: np.ndarray) -> dict:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("Step24 negative-tail metrics require scores")
    top_count = max(1, int(math.ceil(values.size * 0.10)))
    ordered = np.sort(values)
    return {
        "row_count": int(values.size),
        "mean": float(np.mean(values)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "top_decile_mean": float(np.mean(ordered[-top_count:])),
        "maximum": float(np.max(values)),
    }
