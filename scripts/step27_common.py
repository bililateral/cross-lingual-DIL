#!/usr/bin/env python3
"""Shared, fail-closed contracts for Step27 train-only synthetic adaptation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

import step15_build_v7_clean_embedding_cache as redaction
import step24_common


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step27_english_pretrained_synthetic_adaptation_policy.json"
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "step27_english_pretrained_synthetic_adaptation"
DEFAULT_TEXT_FIELDS = [
    "category_concat_top",
    "signature_title_concat",
    "title_concat_top",
    "signature_description_concat",
    "description_concat_top",
]


def shared_dependency_hashes() -> dict[str, str]:
    return {
        relative(Path(redaction.__file__).resolve()): sha256_file(Path(redaction.__file__).resolve()),
        relative(Path(step24_common.__file__).resolve()): sha256_file(
            Path(step24_common.__file__).resolve()
        ),
    }
LABEL_PRESERVING_TRANSFORMS = (
    "section_rotation",
    "segment_rotation",
    "layout_punctuation_normalization",
)
FEATURE_NAMES = (
    "identifier_redacted_e5_cosine",
    "clean_token_jaccard",
    "clean_char3_jaccard",
    "clean_title_token_jaccard",
    "clean_description_token_jaccard",
    "clean_category_token_jaccard",
    "clean_text_length_gap_ratio",
    "clean_segment_count_gap_ratio",
    "clean_field_presence_match_fraction",
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def bool_value(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Step27 expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def render_csv(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("Step27 refuses to render an empty CSV")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("Step27 CSV rows have inconsistent field order")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def render_jsonl(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("Step27 refuses to render an empty JSONL file")
    text = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    return text.encode("utf-8")


def write_bytes_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_dir() or path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite a different Step27 artifact: {path}")
        return
    path.write_bytes(payload)


def write_csv_immutable(path: Path, rows: list[dict]) -> None:
    write_bytes_immutable(path, render_csv(rows))


def write_json_immutable(path: Path, value: dict) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    write_bytes_immutable(path, payload)


def write_jsonl_immutable(path: Path, rows: list[dict]) -> None:
    write_bytes_immutable(path, render_jsonl(rows))


def write_npy_immutable(path: Path, matrix: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(matrix))
    write_bytes_immutable(path, buffer.getvalue())


def file_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def records_for(paths: Iterable[Path]) -> list[dict]:
    return [file_record(path) for path in sorted(set(paths), key=relative)]


def assert_existing_manifest_identity(path: Path, identity: dict) -> dict | None:
    """Refuse reuse when an existing output root belongs to another run."""
    if not path.exists():
        return None
    manifest = load_json(path)
    if manifest.get("identity_sha256") != canonical_hash(identity):
        raise ValueError(
            "Existing Step27 output has a different code/data/policy identity; "
            f"use a new versioned output root: {path}"
        )
    for record in manifest.get("outputs", []):
        output = resolve(record["path"])
        if not output.is_file() or sha256_file(output) != record["sha256"]:
            raise ValueError(f"Existing Step27 manifest output is missing or changed: {output}")
    return manifest


def write_manifest_immutable(
    path: Path,
    *,
    stage: str,
    identity: dict,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    extra: dict | None = None,
) -> dict:
    manifest = {
        "step": stage,
        "identity": identity,
        "identity_sha256": canonical_hash(identity),
        "inputs": records_for(inputs),
        "outputs": records_for(outputs),
    }
    if extra:
        manifest.update(extra)
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    write_json_immutable(path, manifest)
    return manifest


def load_policy(path: str | Path) -> tuple[Path, dict]:
    policy_path = resolve(path)
    policy = load_json(policy_path)
    return policy_path, policy


def policy_input(policy: dict, name: str, *aliases: str) -> Path:
    inputs = policy.get("inputs", {})
    implicit_aliases = {
        "frozen_labels": ("zh_frozen_labels",),
        "evidence_labels": ("zh_evidence_labels",),
        "seller_profiles": ("zh_seller_profiles",),
        "item_identity_signals": ("zh_item_identity_signals",),
        "component_assignments": ("seller_component_assignments",),
    }
    for candidate in (name, *aliases, *implicit_aliases.get(name, ())):
        value = inputs.get(candidate)
        if value:
            return resolve(value)
    raise KeyError(f"Step27 policy is missing inputs.{name}")


def output_root(policy: dict) -> Path:
    return resolve(policy.get("outputs_root", DEFAULT_OUTPUT_ROOT))


def text_fields(policy: dict) -> list[str]:
    generation = policy.get("generation", {})
    fields = generation.get("source_text_fields") or policy.get("clean_text_contract", {}).get(
        "text_fields"
    )
    fields = list(fields or DEFAULT_TEXT_FIELDS)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("Step27 clean text fields are empty or duplicated")
    forbidden = {"contact_concat_top", "structured_snapshot_concat_top"}
    if forbidden.intersection(fields):
        raise ValueError("Step27 clean text fields include identifier/provenance content")
    return fields


def generation_seeds(policy: dict) -> list[int]:
    raw = (
        policy.get("replication", {}).get("seeds")
        or policy.get("generation", {}).get("seeds")
        or policy.get("seeds")
    )
    if raw is None:
        raw = list(range(20260320, 20260330))
    seeds = [int(value) for value in raw]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Step27 seeds are empty or duplicated")
    return seeds


def fold_config(policy: dict) -> tuple[int, int]:
    cv = policy.get("cross_validation", policy.get("cv", {}))
    count = int(cv.get("fold_count", 4))
    seed = int(cv.get("fold_assignment_seed", cv.get("fold_seed", 20260718)))
    if count != 4:
        raise ValueError("Step27 requires exactly four seller-component folds")
    return count, seed


def generation_limits(policy: dict) -> dict:
    cfg = policy.get("generation", {})
    primary = policy.get("parent_cohorts", {}).get("primary_non_silver", {})
    silver = policy.get("parent_cohorts", {}).get("silver_sensitivity", {})
    limits = {
        "primary_positive_parents": int(primary.get("expected_positive_parent_pairs", cfg.get("primary_positive_parent_cap", 16))),
        "primary_negative_parents": int(primary.get("matched_negative_parent_pairs", cfg.get("primary_negative_parent_cap", 16))),
        "primary_variants": int(primary.get("positive_variants_per_parent", cfg.get("primary_variants_per_parent", 2))),
        "primary_child_cap": int(primary.get("maximum_synthetic_rows_per_seed", cfg.get("primary_child_cap_per_seed", 64))),
        "silver_positive_parents": int(silver.get("maximum_positive_parent_pairs", cfg.get("silver_positive_parent_cap", 56))),
        "silver_negative_parents": int(silver.get("maximum_matched_negative_parent_pairs", cfg.get("silver_negative_parent_cap", 56))),
        "silver_variants": int(silver.get("positive_variants_per_parent", cfg.get("silver_variants_per_parent", 1))),
        "silver_child_cap": int(silver.get("maximum_synthetic_rows_per_seed", cfg.get("silver_child_cap_per_seed", 112))),
    }
    if limits["primary_child_cap"] > 64:
        raise ValueError("Step27 primary synthetic cap may not exceed 64 rows per seed")
    expected = (
        limits["primary_positive_parents"] + limits["primary_negative_parents"]
    ) * limits["primary_variants"]
    if expected > limits["primary_child_cap"]:
        raise ValueError("Step27 primary parent/variant budget exceeds the 64-row cap")
    silver_expected = (
        limits["silver_positive_parents"] + limits["silver_negative_parents"]
    ) * limits["silver_variants"]
    if silver_expected > limits["silver_child_cap"]:
        raise ValueError("Step27 silver sensitivity parent/variant budget exceeds its cap")
    return limits


def transform_schedule(policy: dict, variants: int) -> list[str]:
    configured = policy.get("generation", {}).get(
        "matched_recipe_schedule_for_positive_and_negative"
    ) or policy.get("generation", {}).get("transform_schedule")
    schedule = list(configured or LABEL_PRESERVING_TRANSFORMS[: max(variants, 1)])
    allowed = set(LABEL_PRESERVING_TRANSFORMS) | {
        "section_order_rotation",
        "segment_order_permutation_with_layout_normalization",
    }
    forbidden = sorted(set(schedule) - allowed)
    if forbidden:
        raise ValueError(f"Step27 refuses non-label-preserving transforms: {forbidden}")
    if len(schedule) < variants:
        raise ValueError("Step27 transform schedule is shorter than variants_per_parent")
    return schedule


def deterministic_rng(seed: int, *parts: str) -> random.Random:
    key = "\x1f".join([str(seed), *map(str, parts)])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def split_segments(value: object) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"\s*(?:\|\||\r?\n)+\s*", str(value or ""))
        if segment.strip()
    ]


def profile_literals(profile: dict, signal_literals: dict[str, list[str]]) -> list[str]:
    seller_uid = str(profile["seller_uid"])
    literals = list(signal_literals.get(seller_uid, []))
    for alias_field in ("source_seller_raw", "alias_normalized"):
        literal = redaction.safe_signal_literal("seller_alias", profile.get(alias_field, ""))
        if literal:
            literals.append(literal)
    return sorted(set(literals), key=lambda item: (-len(item), item.casefold()))


def clean_profile_fields(
    profile: dict,
    fields: list[str],
    signal_literals: dict[str, list[str]],
) -> tuple[dict[str, str], dict]:
    literals = profile_literals(profile, signal_literals)
    cleaned: dict[str, str] = {}
    diagnostics: Counter[str] = Counter()
    for field in fields:
        value, detail = redaction.redact_identifiers(str(profile.get(field, "") or ""), literals)
        redaction.assert_no_known_identifier_residue(value, literals, str(profile["seller_uid"]))
        cleaned[field] = value
        diagnostics.update(detail)
    return cleaned, dict(diagnostics)


def redact_transformed_fields(
    transformed: dict[str, str], literals: list[str], synthetic_uid: str
) -> tuple[dict[str, str], dict]:
    """Run redaction again after transformation rather than trusting parent cleaning."""
    output: dict[str, str] = {}
    diagnostics: Counter[str] = Counter()
    for field, raw in transformed.items():
        value, detail = redaction.redact_identifiers(str(raw or ""), literals)
        redaction.assert_no_known_identifier_residue(value, literals, synthetic_uid)
        output[field] = value
        diagnostics.update(detail)
    return output, dict(diagnostics)


def normalize_layout(text: str) -> str:
    value = str(text or "")
    value = value.replace(",", "，").replace(";", "；")
    value = value.replace("!", "！").replace("?", "？")
    value = re.sub(r"[，,]{2,}", "，", value)
    value = re.sub(r"[。.!！]{2,}", "。", value)
    value = re.sub(r"[；;]{2,}", "；", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    return value.strip()


def transform_fields(
    clean_fields: dict[str, str], transform_name: str, rng: random.Random
) -> dict[str, str]:
    """Apply content-preserving order/layout changes; never add or splice content."""
    output = dict(clean_fields)
    if transform_name in {"section_rotation", "section_order_rotation"}:
        populated = [field for field, value in output.items() if value]
        if len(populated) > 1:
            shift = 1 + rng.randrange(len(populated) - 1)
            rotated = populated[shift:] + populated[:shift]
            output = {field: clean_fields[field] for field in rotated}
            output.update({field: clean_fields[field] for field in clean_fields if field not in populated})
    elif transform_name in {"segment_rotation", "segment_order_permutation_with_layout_normalization"}:
        for field, value in list(output.items()):
            segments = split_segments(value)
            if len(segments) > 1:
                shift = 1 + rng.randrange(len(segments) - 1)
                output[field] = " || ".join(segments[shift:] + segments[:shift])
        if transform_name == "segment_order_permutation_with_layout_normalization":
            output = {field: normalize_layout(value) for field, value in output.items()}
    elif transform_name == "layout_punctuation_normalization":
        output = {field: normalize_layout(value) for field, value in output.items()}
    else:
        raise ValueError(f"Unknown Step27 transform: {transform_name}")
    return output


def render_profile_text(fields: dict[str, str]) -> str:
    labels = {
        "category_concat_top": "CATEGORIES",
        "signature_title_concat": "SIGNATURE_TITLES",
        "title_concat_top": "TITLES",
        "signature_description_concat": "SIGNATURE_DESCRIPTIONS",
        "description_concat_top": "DESCRIPTIONS",
    }
    sections = [
        f"[{labels.get(field, field.upper())}] {value}"
        for field, value in fields.items()
        if str(value).strip()
    ]
    return "\n".join(sections)


def make_synthetic_profile(
    *,
    synthetic_uid: str,
    parent_uid: str,
    parent_pair_uid: str,
    component_id: str,
    fold: int,
    label: str,
    track: str,
    seed: int,
    variant_index: int,
    transform_name: str,
    fields: dict[str, str],
) -> dict:
    """Create a minimal derived view without fabricated market or identity fields."""
    segments = {field: split_segments(value) for field, value in fields.items()}
    profile_text = render_profile_text(fields)
    if not profile_text:
        raise ValueError(f"Step27 synthetic profile is empty: {synthetic_uid}")
    return {
        "seller_uid": synthetic_uid,
        "data_bucket": "zh_synthetic_train_only",
        "source_dataset": "",
        "source_market_raw": "",
        "source_seller_raw": "",
        "source_seller_id_raw": "",
        "alias_normalized": "",
        "item_count": max((len(items) for items in segments.values()), default=0),
        "unique_title_count": len(set(segments.get("title_concat_top", []))),
        "unique_description_snippet_count": len(
            set(segments.get("description_concat_top", []))
        ),
        "unique_category_count": len(set(segments.get("category_concat_top", []))),
        "contact_type_count": 0,
        "contact_token_count_total": 0,
        "contact_signals": {
            "email": [],
            "telegram": [],
            "wickr": [],
            "wechat": [],
            "qq": [],
            "phone": [],
        },
        "contact_concat_top": "",
        "structured_snapshot_examples": [],
        "structured_snapshot_concat_top": "",
        "synthetic_field_order": list(fields),
        **fields,
        "profile_text": profile_text,
        "synthetic_train_only": True,
        "benchmark_eligible": False,
        "synthetic_lineage": {
            "parent_seller_uid": parent_uid,
            "parent_pair_uid": parent_pair_uid,
            "parent_component_id": component_id,
            "inherited_fold": fold,
            "inherited_label": label,
            "track": track,
            "seed": seed,
            "variant_index": variant_index,
            "transform": transform_name,
            "fabricated_identifier_count": 0,
            "fabricated_market_or_provenance": False,
        },
    }


def load_profiles_index(path: Path) -> dict[str, dict]:
    rows = load_jsonl(path)
    result = {str(row["seller_uid"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Step27 found duplicate seller profiles: {path}")
    return result


def component_assignments(policy: dict) -> dict[str, dict]:
    path = policy_input(policy, "component_assignments")
    rows = load_csv(path)
    result: dict[str, dict] = {}
    for row in rows:
        if row.get("dataset") != "zh_target_strict":
            continue
        uid = row["pair_uid"]
        if uid in result:
            raise ValueError(f"Step27 duplicate component assignment: {uid}")
        if bool_value(row.get("cross_split_component_leakage")) or bool_value(
            row.get("cross_split_seller_leakage")
        ):
            raise ValueError(f"Step27 refuses leaking component assignment: {uid}")
        result[uid] = row
    return result


def canonical_rows(policy: dict, splits: set[str] | None = None) -> list[dict]:
    labels_path = policy_input(policy, "frozen_labels", "zh_frozen_labels")
    evidence_path = policy_input(policy, "evidence_labels", "zh_evidence_labels")
    labels = load_csv(labels_path)
    evidence_rows = load_csv(evidence_path)
    evidence = {row["pair_uid"]: row for row in evidence_rows}
    if len(evidence) != len(evidence_rows):
        raise ValueError("Step27 found duplicate evidence labels")
    assignments = component_assignments(policy)
    selected: list[dict] = []
    seen: set[str] = set()
    for label in labels:
        split = label.get("split_name", "")
        if splits is not None and split not in splits:
            continue
        if label.get("review_label") not in {"positive", "negative"}:
            continue
        if not bool_value(label.get("usable_for_supervision")):
            continue
        if not bool_value(label.get("usable_for_core_transfer")):
            continue
        pair_uid = label["pair_uid"]
        if pair_uid in seen:
            raise ValueError(f"Step27 duplicate canonical label: {pair_uid}")
        seen.add(pair_uid)
        evidence_row = evidence.get(pair_uid)
        if evidence_row is None or not bool_value(evidence_row.get("identity_training_eligible")):
            continue
        assignment = assignments.get(pair_uid)
        if assignment is None:
            raise ValueError(f"Step27 component assignment is missing: {pair_uid}")
        if assignment.get("split_name") != split:
            raise ValueError(f"Step27 canonical/component split mismatch: {pair_uid}")
        selected.append(
            {
                **label,
                "evidence_type": evidence_row.get("evidence_type", ""),
                "evidence_type_confident": evidence_row.get("evidence_type_confident", ""),
                "has_direct_identifier_signal": evidence_row.get(
                    "has_direct_identifier_signal", ""
                ),
                "component_id": assignment["recomputed_component_id"],
            }
        )
    if not selected:
        raise ValueError("Step27 canonical dataset is empty")
    return sorted(selected, key=lambda row: row["pair_uid"])


def build_fixed_component_folds(rows: list[dict], fold_count: int, seed: int) -> dict[str, int]:
    train_rows = [row for row in rows if row["split_name"] == "train"]
    adapted = [{**row, "step24_component_id": row["component_id"]} for row in train_rows]
    return step24_common.balanced_component_folds(adapted, fold_count, seed)


def stable_select(rows: list[dict], count: int, seed: int, namespace: str) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}|{namespace}|{row['pair_uid']}".encode("utf-8")).hexdigest(),
            row["pair_uid"],
        ),
    )
    return ordered[:count]


def balanced_parent_select(
    rows: list[dict], count: int, seed: int, namespace: str
) -> list[dict]:
    """Round-robin evidence strata before deterministic hash fill."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("evidence_type", "unknown")].append(row)
    for key in groups:
        groups[key] = stable_select(groups[key], len(groups[key]), seed, f"{namespace}:{key}")
    selected: list[dict] = []
    keys = sorted(groups)
    while len(selected) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
    return selected


def ensure_track_isolation(primary: list[dict], silver: list[dict]) -> None:
    primary_uids = {row["pair_uid"] for row in primary}
    silver_uids = {row["pair_uid"] for row in silver}
    overlap = sorted(primary_uids & silver_uids)
    if overlap:
        raise ValueError(f"Step27 primary/silver tracks overlap: {overlap[0]}")


def token_set(text: object) -> set[str]:
    value = str(text or "").casefold()
    latin = re.findall(r"[a-z0-9]+", value)
    cjk = re.findall(r"[\u3400-\u9fff]", value)
    return set(latin + cjk)


def char_ngrams(text: object, n: int = 3) -> set[str]:
    value = re.sub(r"\s+", "", str(text or "").casefold())
    if not value:
        return set()
    if len(value) < n:
        return {value}
    return {value[index : index + n] for index in range(len(value) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def ratio_similarity(left: int | float, right: int | float) -> float:
    maximum = max(float(left), float(right), 1.0)
    return max(0.0, 1.0 - abs(float(left) - float(right)) / maximum)


def recompute_pair_features(
    left: dict,
    right: dict,
    left_embedding: np.ndarray,
    right_embedding: np.ndarray,
    fields: list[str],
) -> dict[str, float]:
    left_text = render_profile_text({field: str(left.get(field, "") or "") for field in fields})
    right_text = render_profile_text({field: str(right.get(field, "") or "") for field in fields})
    title_fields = [field for field in fields if "title" in field]
    description_fields = [field for field in fields if "description" in field]
    category_fields = [field for field in fields if "category" in field]

    def joined(profile: dict, selected: list[str]) -> str:
        return "\n".join(str(profile.get(field, "") or "") for field in selected)

    left_segments = sum(len(split_segments(left.get(field, ""))) for field in fields)
    right_segments = sum(len(split_segments(right.get(field, ""))) for field in fields)
    left_present = {field for field in fields if str(left.get(field, "")).strip()}
    right_present = {field for field in fields if str(right.get(field, "")).strip()}
    cosine = float(np.dot(np.asarray(left_embedding), np.asarray(right_embedding)))
    features = {
        "identifier_redacted_e5_cosine": cosine,
        "clean_token_jaccard": jaccard(token_set(left_text), token_set(right_text)),
        "clean_char3_jaccard": jaccard(char_ngrams(left_text), char_ngrams(right_text)),
        "clean_title_token_jaccard": jaccard(
            token_set(joined(left, title_fields)), token_set(joined(right, title_fields))
        ),
        "clean_description_token_jaccard": jaccard(
            token_set(joined(left, description_fields)),
            token_set(joined(right, description_fields)),
        ),
        "clean_category_token_jaccard": jaccard(
            token_set(joined(left, category_fields)), token_set(joined(right, category_fields))
        ),
        "clean_text_length_gap_ratio": 1.0 - ratio_similarity(len(left_text), len(right_text)),
        "clean_segment_count_gap_ratio": 1.0 - ratio_similarity(left_segments, right_segments),
        "clean_field_presence_match_fraction": (
            sum((field in left_present) == (field in right_present) for field in fields)
            / max(len(fields), 1)
        ),
    }
    if set(features) != set(FEATURE_NAMES):
        raise AssertionError("Step27 feature contract drift")
    if any(not math.isfinite(value) for value in features.values()):
        raise ValueError("Step27 generated a non-finite pair feature")
    if any(value < -1.0001 or value > 1.0001 for value in features.values()):
        raise ValueError("Step27 generated an out-of-range normalized pair feature")
    return features


def load_normalized_cache(metadata_path: Path, matrix_path: Path) -> tuple[dict[str, int], np.ndarray, dict]:
    metadata = load_json(metadata_path)
    matrix = np.load(matrix_path, mmap_mode="r")
    seller_uids = list(metadata.get("seller_uids", []))
    if list(matrix.shape) != list(metadata.get("shape", [])):
        raise ValueError(f"Step27 embedding cache shape mismatch: {matrix_path}")
    if len(seller_uids) != matrix.shape[0] or len(set(seller_uids)) != len(seller_uids):
        raise ValueError(f"Step27 embedding cache UID mismatch: {metadata_path}")
    norms = np.linalg.norm(np.asarray(matrix, dtype=np.float32), axis=1)
    if norms.size and np.max(np.abs(norms - 1.0)) > 1e-3:
        raise ValueError(f"Step27 embedding cache is not normalized: {matrix_path}")
    return {uid: index for index, uid in enumerate(seller_uids)}, matrix, metadata


def parent_root(policy: dict) -> Path:
    return output_root(policy) / "parent_manifest"


def seed_root(policy: dict, seed: int) -> Path:
    return output_root(policy) / f"seed_{int(seed)}"


def track_root(policy: dict, seed: int, track: str) -> Path:
    if track not in {"primary", "silver_sensitivity"}:
        raise ValueError(f"Unknown Step27 track: {track}")
    return seed_root(policy, seed) / track


def profile_cache_paths(policy: dict, seed: int | None, track: str) -> tuple[Path, Path]:
    if track == "real":
        root = output_root(policy) / "embeddings" / "real"
    else:
        if seed is None:
            raise ValueError("Synthetic Step27 cache requires a seed")
        root = track_root(policy, seed, track) / "embeddings"
    return root / "identifier_redacted_e5.npy", root / "identifier_redacted_e5.json"
