#!/usr/bin/env python3
"""Shared contracts for the Step26 frozen authorship bridge."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import step24_common as step24


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step26_frozen_authorship_bridge_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: str | Path) -> dict:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def load_csv(path: str | Path) -> list[dict]:
    with resolve(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_csv(rows: list[dict], fields: list[str] | None = None) -> bytes:
    if not rows:
        raise ValueError("Step26 refuses to render an empty CSV")
    selected_fields = fields or list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=selected_fields,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def write_bytes_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite a different Step26 artifact: {path}")
        return
    path.write_bytes(payload)


def write_json_immutable(path: Path, payload: dict) -> None:
    write_bytes_immutable(
        path,
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def write_csv_immutable(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    write_bytes_immutable(path, render_csv(rows, fields))


def load_policy(path: str | Path = DEFAULT_POLICY) -> tuple[Path, dict, dict]:
    policy_path = resolve(path)
    policy = load_json(policy_path)
    step24_policy = load_json(policy["frozen_sources"]["step24_policy"])
    validate_policy(policy, step24_policy)
    return policy_path, policy, step24_policy


def validate_policy(policy: dict, step24_policy: dict) -> None:
    frozen = policy["frozen_models"]
    if frozen["encoder_keys"] != ["pcm_multilingual_authorship", "mstyledistance"]:
        raise ValueError("Step26 frozen encoder order changed")
    if frozen["source_artifact_keys"] != [
        "e5_lr_l2_control",
        "style_only_lr_l2_control",
        "semantic_style_lr_l2_primary",
    ]:
        raise ValueError("Step26 source artifact order changed")
    if frozen["primary_bridge_model"] != "source_only_semantic_style_lr_l2_primary":
        raise ValueError("Step26 primary bridge model changed after preregistration")
    if frozen["encoder_parameters_updated"] is not False:
        raise ValueError("Step26 encoders must remain frozen")
    if frozen["source_artifact_refit_forbidden"] is not True:
        raise ValueError("Step26 source artifacts may not be refit")
    if frozen["valid_test_threshold_selection_forbidden"] is not True:
        raise ValueError("Step26 may not select a threshold on Chinese valid/test")
    if step24_policy["clean_text_contract"]["encode_valid_or_test"] is not False:
        raise ValueError("Frozen Step24 policy no longer has the original train-only contract")
    if step24_policy["evaluation"]["valid_or_test_selection_forbidden"] is not True:
        raise ValueError("Frozen Step24 policy permits valid/test selection")
    split_cfg = policy["evaluation_data"]["split_allowlists"]
    if list(split_cfg) != [
        "representative_valid",
        "internal_development_test_diagnostic_only",
    ]:
        raise ValueError("Step26 evaluation split order changed")
    expected = {
        "representative_valid": (120, 30, 90),
        "internal_development_test_diagnostic_only": (200, 50, 150),
    }
    for split_name, counts in expected.items():
        cfg = split_cfg[split_name]
        if (cfg["expected_rows"], cfg["expected_positive"], cfg["expected_negative"]) != counts:
            raise ValueError(f"Step26 expected split counts changed: {split_name}")
    if policy["evaluation"]["internal_test_satisfies_no_promotion_gate"] is not True:
        raise ValueError("Step26 internal test may satisfy no promotion gate")
    if policy["evaluation"]["publication_claim_requires_step20"] is not True:
        raise ValueError("Step26 must require Step20 for publication")


def _prediction_index(path: str | Path, score_field: str) -> dict[str, dict]:
    rows = load_csv(path)
    result = {}
    for row in rows:
        uid = row.get("pair_uid", "")
        if not uid or uid in result:
            raise ValueError(f"Step26 comparator contains missing/duplicate pair UID: {path}")
        value = float(row[score_field])
        if not np.isfinite(value):
            raise ValueError(f"Step26 comparator score is non-finite: {path}:{uid}")
        result[uid] = row
    return result


def pair_uid_sellers(pair_uid: str) -> tuple[str, str]:
    parts = str(pair_uid).split("||")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Step26 cannot parse the canonical pair UID: {pair_uid}")
    return parts[0], parts[1]


def load_blind_pair_allowlists(policy: dict) -> dict[str, list[str]]:
    """Load pair identities without reading labels or evidence fields."""
    output = {}
    for split_name, cfg in policy["evaluation_data"]["split_allowlists"].items():
        indexes = [
            _prediction_index(cfg["v8_b0_predictions"], "prob_positive"),
            _prediction_index(cfg["v8_clean_predictions"], "prob_positive"),
            _prediction_index(
                cfg["v8_contextual_predictions"], "contextual_evidence_prob_positive"
            ),
        ]
        if set(indexes[0]) != set(indexes[1]) or set(indexes[0]) != set(indexes[2]):
            raise ValueError(f"Step26 V8 comparator pair universes differ: {split_name}")
        pair_uids = sorted(indexes[0])
        if len(pair_uids) != int(cfg["expected_rows"]):
            raise ValueError(f"Step26 blind allowlist count changed: {split_name}")
        for pair_uid in pair_uids:
            pair_uid_sellers(pair_uid)
        output[split_name] = pair_uids
    valid_sellers = {
        seller
        for uid in output["representative_valid"]
        for seller in pair_uid_sellers(uid)
    }
    test_sellers = {
        seller
        for uid in output["internal_development_test_diagnostic_only"]
        for seller in pair_uid_sellers(uid)
    }
    overlap = valid_sellers & test_sellers
    if overlap:
        raise ValueError(f"Step26 blind valid/test seller leakage: {sorted(overlap)[0]}")
    return output


def load_evaluation_rows(policy: dict) -> dict[str, list[dict]]:
    data_cfg = policy["evaluation_data"]
    label_rows = load_csv(data_cfg["frozen_labels"])
    evidence_rows = load_csv(data_cfg["evidence_labels"])
    labels = {row["pair_uid"]: row for row in label_rows}
    evidence = {row["pair_uid"]: row for row in evidence_rows}
    if len(labels) != len(label_rows) or len(evidence) != len(evidence_rows):
        raise ValueError("Step26 found duplicate label/evidence pair UIDs")

    output = {}
    for split_name, cfg in data_cfg["split_allowlists"].items():
        b0 = _prediction_index(cfg["v8_b0_predictions"], "prob_positive")
        clean = _prediction_index(cfg["v8_clean_predictions"], "prob_positive")
        contextual = _prediction_index(
            cfg["v8_contextual_predictions"], "contextual_evidence_prob_positive"
        )
        if set(b0) != set(clean) or set(b0) != set(contextual):
            raise ValueError(f"Step26 V8 comparator pair universes differ: {split_name}")
        rows = []
        for uid in sorted(b0):
            label = labels.get(uid)
            evidence_row = evidence.get(uid)
            if label is None or evidence_row is None:
                raise ValueError(f"Step26 pair lacks frozen label/evidence: {uid}")
            if label.get("split_name") != cfg["label_split_name"]:
                raise ValueError(
                    f"Step26 comparator/label split mismatch: {split_name}:{uid}"
                )
            if label.get("review_label") not in {"positive", "negative"}:
                raise ValueError(f"Step26 evaluation pair is not binary-labeled: {uid}")
            expected_label = label["review_label"]
            expected_evidence = evidence_row["evidence_type"]
            for comparator_name, comparator_row in (
                ("v8_b0", b0[uid]),
                ("v8_clean", clean[uid]),
                ("v8_contextual", contextual[uid]),
            ):
                if comparator_row.get("review_label") != expected_label:
                    raise ValueError(
                        f"Step26 current/frozen label drift: {comparator_name}:{uid}"
                    )
                if comparator_row.get("evidence_type") != expected_evidence:
                    raise ValueError(
                        f"Step26 current/frozen evidence drift: {comparator_name}:{uid}"
                    )
            if label.get("usable_for_supervision") != "1" or label.get(
                "usable_for_core_transfer"
            ) != "1":
                raise ValueError(f"Step26 evaluation pair is not benchmark eligible: {uid}")
            if str(label.get("silver_train_only", "0")).strip() == "1":
                raise ValueError(f"Step26 evaluation contains train-only silver: {uid}")
            component = b0[uid].get("v7_component_id") or clean[uid].get(
                "v7_component_id"
            )
            if not component:
                raise ValueError(f"Step26 comparator lacks seller component: {uid}")
            if clean[uid].get("v7_component_id") != component or contextual[uid].get(
                "v7_component_id"
            ) != component:
                raise ValueError(f"Step26 frozen comparator components differ: {uid}")
            parsed_left, parsed_right = pair_uid_sellers(uid)
            if (label.get("seller_uid_left"), label.get("seller_uid_right")) != (
                parsed_left,
                parsed_right,
            ):
                raise ValueError(f"Step26 pair UID and frozen seller fields differ: {uid}")
            rows.append(
                {
                    **label,
                    "step26_split": split_name,
                    "evidence_type": expected_evidence,
                    "v7_component_id": component,
                    "v8_b0_prob_positive": b0[uid]["prob_positive"],
                    "v8_clean_prob_positive": clean[uid]["prob_positive"],
                    "v8_contextual_prob_positive": contextual[uid][
                        "contextual_evidence_prob_positive"
                    ],
                }
            )
        positive = sum(row["review_label"] == "positive" for row in rows)
        negative = len(rows) - positive
        observed = (len(rows), positive, negative)
        expected = (
            int(cfg["expected_rows"]),
            int(cfg["expected_positive"]),
            int(cfg["expected_negative"]),
        )
        if observed != expected:
            raise ValueError(
                f"Step26 split count mismatch: {split_name} expected={expected} observed={observed}"
            )
        output[split_name] = rows

    train_rows = [row for row in label_rows if row.get("split_name") == "train"]
    train_sellers = {
        row[key]
        for row in train_rows
        for key in ("seller_uid_left", "seller_uid_right")
        if row.get(key)
    }
    split_sellers = {
        name: {
            row[key]
            for row in rows
            for key in ("seller_uid_left", "seller_uid_right")
            if row.get(key)
        }
        for name, rows in output.items()
    }
    for name, sellers in split_sellers.items():
        overlap = train_sellers & sellers
        if overlap:
            raise ValueError(f"Step26 train/evaluation seller leakage: {name}:{sorted(overlap)[0]}")
    names = list(split_sellers)
    overlap = split_sellers[names[0]] & split_sellers[names[1]]
    if overlap:
        raise ValueError(f"Step26 valid/test seller leakage: {sorted(overlap)[0]}")
    return output


def evaluation_sellers(rows_by_split: dict[str, list[dict]]) -> list[str]:
    return sorted(
        {
            row[key]
            for rows in rows_by_split.values()
            for row in rows
            for key in ("seller_uid_left", "seller_uid_right")
            if row.get(key)
        }
    )


def labels_array(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [1.0 if row["review_label"] == "positive" else 0.0 for row in rows],
        dtype=float,
    )


def grouped_bootstrap_delta(
    rows: list[dict],
    baseline: np.ndarray,
    candidate: np.ndarray,
    metric_fn,
    resamples: int,
    seed: int,
) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["v7_component_id"]].append(index)
    components = sorted(groups)
    labels = labels_array(rows)
    rng = np.random.default_rng(seed)
    values = []
    skipped = 0
    for _ in range(resamples):
        sampled = []
        for _component in components:
            selected = components[int(rng.integers(0, len(components)))]
            sampled.extend(groups[selected])
        sampled = np.asarray(sampled, dtype=int)
        if len(np.unique(labels[sampled])) < 2:
            skipped += 1
            continue
        values.append(
            metric_fn(labels[sampled], candidate[sampled])
            - metric_fn(labels[sampled], baseline[sampled])
        )
    if len(values) < max(100, int(resamples * 0.9)):
        raise ValueError("Step26 grouped bootstrap produced too few valid resamples")
    values_array = np.asarray(values, dtype=float)
    return {
        "resamples_requested": resamples,
        "resamples_completed": len(values),
        "single_class_resamples_skipped": skipped,
        "component_count": len(components),
        "point_delta": float(metric_fn(labels, candidate) - metric_fn(labels, baseline)),
        "mean_delta": float(np.mean(values_array)),
        "ci95_lower": float(np.quantile(values_array, 0.025)),
        "ci95_upper": float(np.quantile(values_array, 0.975)),
        "probability_delta_gt_zero": float(np.mean(values_array > 0.0)),
    }


def validate_frozen_sources(policy: dict) -> dict:
    frozen = policy["frozen_sources"]
    step24_clean_manifest = load_json(frozen["step24_clean_text_manifest"])
    step24_summary = load_json(frozen["step24_evaluation_summary"])
    step24_artifacts = load_json(frozen["step24_model_artifacts"])
    v8_summary = load_json(frozen["step15_v8_step12_summary"])
    v8_freeze = load_json(frozen["step15_v8_freeze_manifest"])
    if step24_summary.get("valid_test_pair_features_scored") != 0:
        raise ValueError("Step26 parent Step24 scored valid/test during model selection")
    if step24_summary.get("method_selected_from_valid_or_test") is not False:
        raise ValueError("Step26 parent Step24 selected a model from valid/test")
    if step24_summary.get("encoder_parameters_updated") is not False:
        raise ValueError("Step26 parent Step24 updated encoder parameters")
    if step24_clean_manifest.get("valid_test_scores_or_labels_read") is not False:
        raise ValueError("Step26 parent Step24 clean replay read valid/test")
    zh_clean_record = step24_clean_manifest.get("records", {}).get("zh_target_strict", {})
    data_cfg = policy["evaluation_data"]
    expected_step24_input_hashes = {
        "v7_e5_metadata_sha256": sha256(data_cfg["identifier_redacted_e5_metadata"]),
        "v7_e5_matrix_sha256": sha256(data_cfg["identifier_redacted_e5_matrix"]),
        "profile_sha256": sha256(data_cfg["seller_profiles"]),
        "identity_signal_sha256": sha256(data_cfg["item_identity_signals"]),
    }
    for key, observed in expected_step24_input_hashes.items():
        if zh_clean_record.get(key) != observed:
            raise ValueError(f"Step26 current clean input differs from frozen Step24: {key}")
    if v8_freeze.get("internal_test_used_for_selection") is not False:
        raise ValueError("Step26 parent Step15-v8 used the internal test for selection")
    if v8_summary.get("selection", {}).get("internal_test_metrics_used_for_selection") is not False:
        raise ValueError("Step26 parent Step15-v8 summary used internal-test selection")
    return {
        "step24_summary": step24_summary,
        "step24_clean_manifest": step24_clean_manifest,
        "step24_artifacts": step24_artifacts,
        "v8_summary": v8_summary,
        "v8_freeze": v8_freeze,
        "hashes": {key: sha256(value) for key, value in frozen.items()},
    }
