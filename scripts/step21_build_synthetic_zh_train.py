#!/usr/bin/env python3
"""Build auditable Chinese train-only synthetic positive-pair augmentations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import step15_build_v7_clean_embedding_cache as redaction


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step21_synthetic_train_only_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bool_value(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def split_segments(value: object) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"\s*(?:\|\||\r?\n)+\s*", str(value or ""))
        if segment.strip()
    ]


def deterministic_rng(seed: int, *parts: str) -> random.Random:
    key = "\x1f".join([str(seed), *parts])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def normalize_layout(text: str) -> str:
    output = str(text or "")
    output = output.replace(",", "，").replace(";", "；")
    output = output.replace("!", "！").replace("?", "？")
    output = re.sub(r"[，,]{2,}", "，", output)
    output = re.sub(r"[。.!！]{2,}", "。", output)
    output = re.sub(r"[；;]{2,}", "；", output)
    output = re.sub(r"[ \t]+", " ", output)
    output = re.sub(r"\s*\n\s*", "\n", output)
    return output.strip()


def transform_sections(
    clean_fields: dict[str, str], transform_name: str, rng: random.Random
) -> dict[str, str]:
    output = dict(clean_fields)
    field_names = list(clean_fields)
    if transform_name == "section_rotation":
        populated = [name for name in field_names if clean_fields[name]]
        if len(populated) > 1:
            shift = 1 + rng.randrange(len(populated) - 1)
            rotated = populated[shift:] + populated[:shift]
            output = {name: clean_fields[name] for name in rotated}
            output.update(
                {name: clean_fields[name] for name in field_names if name not in populated}
            )
    elif transform_name == "segment_subsample":
        for name in field_names:
            segments = split_segments(clean_fields[name])
            if len(segments) > 1:
                keep = max(1, (len(segments) + 1) // 2)
                selected = sorted(rng.sample(range(len(segments)), keep))
                output[name] = " || ".join(segments[index] for index in selected)
    elif transform_name == "layout_punctuation_normalization":
        for name in field_names:
            output[name] = normalize_layout(clean_fields[name])
    else:
        raise ValueError(f"Unknown Step21 transform: {transform_name}")
    return output


def profile_literals(
    profile: dict, signal_literals: dict[str, list[str]]
) -> list[str]:
    seller_uid = str(profile["seller_uid"])
    literals = list(signal_literals.get(seller_uid, []))
    alias_literal = redaction.safe_signal_literal(
        "seller_alias", profile.get("alias_normalized", "")
    )
    if alias_literal:
        literals.append(alias_literal)
    return sorted(set(literals), key=lambda value: (-len(value), value.casefold()))


def clean_profile_fields(
    profile: dict,
    field_names: list[str],
    signal_literals: dict[str, list[str]],
) -> tuple[dict[str, str], dict]:
    literals = profile_literals(profile, signal_literals)
    cleaned = {}
    diagnostics = Counter()
    for field_name in field_names:
        clean, field_diagnostics = redaction.redact_identifiers(
            str(profile.get(field_name, "") or ""), literals
        )
        redaction.assert_no_known_identifier_residue(clean, literals, str(profile["seller_uid"]))
        cleaned[field_name] = clean
        diagnostics.update(field_diagnostics)
    return cleaned, dict(diagnostics)


def render_profile_text(fields: dict[str, str]) -> str:
    section_labels = {
        "category_concat_top": "CATEGORIES",
        "signature_title_concat": "SIGNATURE_TITLES",
        "title_concat_top": "TITLES",
        "signature_description_concat": "SIGNATURE_DESCRIPTIONS",
        "description_concat_top": "DESCRIPTIONS",
    }
    sections = []
    for field_name, value in fields.items():
        if value:
            sections.append(f"[{section_labels.get(field_name, field_name.upper())}] {value}")
    return "\n".join(sections)


def synthetic_profile(
    parent: dict,
    synthetic_uid: str,
    transformed_fields: dict[str, str],
    marker: str,
) -> dict:
    output = deepcopy(parent)
    output["seller_uid"] = synthetic_uid
    output["data_bucket"] = "zh_synthetic_train_only"
    output["source_dataset"] = marker
    output["source_market_raw"] = marker
    output["source_seller_raw"] = synthetic_uid.rsplit("/", 1)[-1]
    output["source_seller_id_raw"] = ""
    output["alias_normalized"] = ""
    for field_name, value in transformed_fields.items():
        output[field_name] = value
    title_segments = split_segments(transformed_fields.get("title_concat_top", ""))
    signature_title_segments = split_segments(
        transformed_fields.get("signature_title_concat", "")
    )
    description_segments = split_segments(
        transformed_fields.get("description_concat_top", "")
    )
    signature_description_segments = split_segments(
        transformed_fields.get("signature_description_concat", "")
    )
    output["top_titles"] = [
        {"value": value, "count": 1} for value in title_segments
    ]
    output["signature_titles"] = [
        {"value": value, "count": 1, "specificity_score": None, "seller_df": None}
        for value in signature_title_segments
    ]
    output["top_description_snippets"] = [
        {"value": value, "count": 1} for value in description_segments
    ]
    output["signature_description_segments"] = [
        {"value": value, "count": 1, "specificity_score": None, "seller_df": None}
        for value in signature_description_segments
    ]
    output["contact_type_count"] = 0
    output["contact_token_count_total"] = 0
    output["contact_signals"] = {
        "email": [],
        "telegram": [],
        "wickr": [],
        "wechat": [],
        "qq": [],
        "phone": [],
    }
    output["contact_concat_top"] = ""
    output["structured_snapshot_examples"] = []
    output["structured_snapshot_concat_top"] = ""
    output["profile_text"] = render_profile_text(transformed_fields)
    output["synthetic_train_only"] = True
    return output


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def label_row(
    parent: dict,
    label_fields: list[str],
    synthetic_pair_uid: str,
    left_uid: str,
    right_uid: str,
    marker: str,
    child_weight: float,
    transform_name: str,
) -> dict:
    output = {field: str(parent.get(field, "") or "") for field in label_fields}
    output.update(
        {
            "pair_uid": synthetic_pair_uid,
            "data_bucket": "zh_synthetic_train_only",
            "candidate_scope": "synthetic_train_only",
            "review_status": "synthetic_train_only",
            "review_label": "positive",
            "reviewer_id": "step21_deterministic_generator",
            "review_notes": (
                f"TRAIN-ONLY synthetic child of {parent['pair_uid']}; "
                f"transform={transform_name}; not an independent identity label"
            ),
            "usable_for_supervision": "1",
            "usable_for_core_transfer": "0",
            "split_name": "train",
            "seller_uid_left": left_uid,
            "seller_uid_right": right_uid,
            "source_market_raw_left": marker,
            "source_market_raw_right": marker,
            "source_seller_raw_left": left_uid.rsplit("/", 1)[-1],
            "source_seller_raw_right": right_uid.rsplit("/", 1)[-1],
            "same_market_raw": "1",
            "benchmark_eligible": "0",
            "silver_train_only": "1",
            "training_sample_weight": f"{child_weight:.12f}",
            "shared_contact_count": "0",
            "shared_contact_values": "",
            "shared_pgp_fingerprint_count": "0",
            "shared_pgp_fingerprint_values": "",
        }
    )
    return output


def select_parent_rows(
    labels: list[dict],
    evidence_index: dict[str, dict],
    component_assignments: dict[str, dict],
    track_cfg: dict,
    eligibility: dict,
) -> list[tuple[dict, dict]]:
    forbidden_splits = set(eligibility["forbidden_splits"])
    allowed_evidence_types = set(track_cfg["allowed_evidence_types"])
    parents = []
    for row in labels:
        evidence = evidence_index.get(row["pair_uid"])
        if not evidence:
            continue
        if row.get("split_name") in forbidden_splits:
            continue
        if row.get("split_name") != eligibility["required_split_name"]:
            continue
        if row.get("review_label") != eligibility["required_review_label"]:
            continue
        if bool_value(row.get("usable_for_supervision")) != bool(
            eligibility["required_usable_for_supervision"]
        ):
            continue
        if bool_value(row.get("usable_for_core_transfer")) != bool(
            eligibility["required_usable_for_core_transfer"]
        ):
            continue
        if bool_value(row.get("silver_train_only")) != bool(
            track_cfg["silver_train_only"]
        ):
            continue
        if evidence.get("evidence_type") not in allowed_evidence_types:
            continue
        assignment = component_assignments.get(row["pair_uid"])
        if assignment is None:
            raise ValueError(f"Missing Step16I component assignment for {row['pair_uid']}")
        if assignment.get("dataset") != "zh_target_strict":
            raise ValueError(f"Wrong Step16I assignment dataset for {row['pair_uid']}")
        if assignment.get("split_name") != "train":
            raise ValueError(f"Step16I assignment split drift for {row['pair_uid']}")
        if bool_value(assignment.get("cross_split_component_leakage")):
            raise ValueError(f"Step16I component leakage for {row['pair_uid']}")
        effective_parent = dict(row)
        effective_parent["split_component_id"] = assignment["recomputed_component_id"]
        parents.append((effective_parent, evidence))
    return parents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--track", action="append", dest="tracks")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = resolve(args.policy)
    policy = load_json(policy_path)
    inputs = {name: resolve(path) for name, path in policy["inputs"].items()}
    labels, label_fields = load_csv(inputs["frozen_labels"])
    evidence_rows, _ = load_csv(inputs["evidence_labels"])
    evidence_index = {row["pair_uid"]: row for row in evidence_rows}
    profiles = load_jsonl(inputs["seller_profiles"])
    profile_index = {row["seller_uid"]: row for row in profiles}
    signal_literals, signal_summary = redaction.signal_literals_by_seller(
        inputs["item_identity_signals"]
    )
    component_assignments = {
        row["pair_uid"]: row for row in load_csv(inputs["component_assignments"])[0]
    }
    output_root = resolve(policy["outputs_root"])
    selected_tracks = args.tracks or list(policy["tracks"])
    unknown = sorted(set(selected_tracks) - set(policy["tracks"]))
    if unknown:
        raise ValueError(f"Unknown Step21 tracks: {unknown}")
    input_manifest = {
        name: {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in inputs.items()
    }
    summary_path = output_root / policy["outputs"]["summary"]
    manifest_path = output_root / policy["outputs"]["manifest"]
    producer_path = Path(__file__).resolve()
    if output_root.exists() and not args.force:
        if not summary_path.is_file() or not manifest_path.is_file():
            raise FileExistsError(
                f"Incomplete Step21 generation root exists: {output_root}; archive or remove only this Step21 root"
            )
        existing_manifest = load_json(manifest_path)
        expected_identity = {
            "policy_sha256": sha256(policy_path),
            "producer_sha256": sha256(producer_path),
            "selected_tracks": selected_tracks,
            "inputs": input_manifest,
        }
        observed_identity = {
            key: existing_manifest.get(key) for key in expected_identity
        }
        if observed_identity != expected_identity:
            raise FileExistsError(
                "Existing Step21 generation belongs to different code, policy, tracks, or data; "
                "use a new versioned output root"
            )
        for record in existing_manifest.get("outputs", []):
            path = resolve(record["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(record["size_bytes"])
                or sha256(path) != record["sha256"]
            ):
                raise ValueError(f"Existing Step21 generation artifact drift: {path}")
        print(summary_path.read_text(encoding="utf-8"))
        return

    eligibility = policy["eligibility"]
    source_fields = list(policy["generation"]["source_text_fields"])
    marker = policy["generation"]["synthetic_market_marker"]
    uid_prefix = policy["generation"]["synthetic_uid_prefix"].rstrip("/")
    transform_schedule = list(policy["generation"]["transform_schedule"])
    global_seed = int(policy["global_seed"])
    budget_multiplier = float(
        policy["weighting"]["synthetic_budget_per_parent_multiplier"]
    )
    if budget_multiplier > float(
        policy["weighting"]["maximum_total_synthetic_weight_per_parent_relative_to_parent"]
    ):
        raise ValueError("Step21 synthetic parent weight budget exceeds the policy cap")

    summaries = {}
    all_output_paths = []
    for track_name in selected_tracks:
        track_cfg = policy["tracks"][track_name]
        variants = int(track_cfg["variants_per_parent_pair"])
        if variants <= 0:
            raise ValueError(f"Step21 track {track_name} has no variants")
        parents = select_parent_rows(
            labels, evidence_index, component_assignments, track_cfg, eligibility
        )
        if len(parents) < int(track_cfg["minimum_parent_pairs"]):
            raise ValueError(
                f"Step21 track {track_name} has {len(parents)} eligible parents, below "
                f"minimum={track_cfg['minimum_parent_pairs']}"
            )

        track_root = output_root / policy["outputs"]["tracks_directory"] / track_name
        synthetic_profiles = []
        synthetic_labels = []
        duplicate_labels = []
        lineage_rows = []
        item_rows = []
        redaction_totals = Counter()
        parent_weight_total = 0.0
        synthetic_weight_total = 0.0
        for parent_index, (parent, evidence) in enumerate(
            sorted(parents, key=lambda item: item[0]["pair_uid"])
        ):
            left_parent = profile_index.get(parent["seller_uid_left"])
            right_parent = profile_index.get(parent["seller_uid_right"])
            if left_parent is None or right_parent is None:
                raise ValueError(f"Step21 parent profiles are missing for {parent['pair_uid']}")
            left_clean, left_diag = clean_profile_fields(
                left_parent, source_fields, signal_literals
            )
            right_clean, right_diag = clean_profile_fields(
                right_parent, source_fields, signal_literals
            )
            if not any(left_clean.values()) or not any(right_clean.values()):
                raise ValueError(
                    f"Step21 identifier redaction removed all content for {parent['pair_uid']}"
                )
            redaction_totals.update(left_diag)
            redaction_totals.update(right_diag)
            parent_weight = float(parent.get("training_sample_weight") or 1.0)
            child_weight = parent_weight * budget_multiplier / variants
            parent_weight_total += parent_weight
            for variant_index in range(variants):
                transform_name = transform_schedule[variant_index % len(transform_schedule)]
                variant_key = f"p{parent_index:04d}v{variant_index:02d}"
                rng_left = deterministic_rng(
                    global_seed, track_name, parent["pair_uid"], variant_key, "left"
                )
                rng_right = deterministic_rng(
                    global_seed, track_name, parent["pair_uid"], variant_key, "right"
                )
                left_fields = transform_sections(left_clean, transform_name, rng_left)
                right_fields = transform_sections(right_clean, transform_name, rng_right)
                left_text_changed = render_profile_text(left_fields) != render_profile_text(
                    left_clean
                )
                right_text_changed = render_profile_text(right_fields) != render_profile_text(
                    right_clean
                )
                left_uid = f"{uid_prefix}/{track_name}/{variant_key}/left"
                right_uid = f"{uid_prefix}/{track_name}/{variant_key}/right"
                synthetic_pair_uid = f"{left_uid}||{right_uid}"
                duplicate_pair_uid = (
                    f"synthetic://step21/duplication/{track_name}/{variant_key}/left||"
                    f"synthetic://step21/duplication/{track_name}/{variant_key}/right"
                )
                synthetic_profiles.extend(
                    [
                        synthetic_profile(left_parent, left_uid, left_fields, marker),
                        synthetic_profile(right_parent, right_uid, right_fields, marker),
                    ]
                )
                synthetic_labels.append(
                    label_row(
                        parent,
                        label_fields,
                        synthetic_pair_uid,
                        left_uid,
                        right_uid,
                        marker,
                        child_weight,
                        transform_name,
                    )
                )
                duplicate = label_row(
                    parent,
                    label_fields,
                    duplicate_pair_uid,
                    parent["seller_uid_left"],
                    parent["seller_uid_right"],
                    marker,
                    child_weight,
                    "equal_effective_weight_duplication",
                )
                duplicate["seller_uid_left"] = parent["seller_uid_left"]
                duplicate["seller_uid_right"] = parent["seller_uid_right"]
                duplicate_labels.append(duplicate)
                synthetic_weight_total += child_weight
                lineage_rows.append(
                    {
                        "synthetic_pair_uid": synthetic_pair_uid,
                        "duplication_control_pair_uid": duplicate_pair_uid,
                        "parent_pair_uid": parent["pair_uid"],
                        "parent_split_name": parent["split_name"],
                        "parent_split_component_id": parent["split_component_id"],
                        "parent_evidence_type": evidence["evidence_type"],
                        "parent_label_tier": parent.get("label_tier", ""),
                        "parent_silver_train_only": parent.get("silver_train_only", ""),
                        "parent_training_sample_weight": f"{parent_weight:.12f}",
                        "synthetic_training_sample_weight": f"{child_weight:.12f}",
                        "track": track_name,
                        "transform": transform_name,
                        "left_text_changed": "1" if left_text_changed else "0",
                        "right_text_changed": "1" if right_text_changed else "0",
                        "variant_index": variant_index,
                        "synthetic_train_only": "1",
                        "benchmark_eligible": "0",
                        "independent_real_sample": "0",
                    }
                )
                for side, fields, uid in (
                    ("left", left_fields, left_uid),
                    ("right", right_fields, right_uid),
                ):
                    item_rows.append(
                        {
                            "vendor": uid.rsplit("/", 1)[-1],
                            "ship_from": "",
                            "title": fields.get("title_concat_top", ""),
                            "description": fields.get("description_concat_top", ""),
                            "price": "",
                            "category": fields.get("category_concat_top", ""),
                            "market": marker,
                            "synthetic_pair_uid": synthetic_pair_uid,
                            "synthetic_side": side,
                            "parent_pair_uid": parent["pair_uid"],
                            "parent_split_component_id": parent["split_component_id"],
                            "track": track_name,
                            "transform": transform_name,
                            "synthetic_train_only": "1",
                        }
                    )

        if abs(synthetic_weight_total - parent_weight_total * budget_multiplier) > 1e-8:
            raise ValueError(f"Step21 track {track_name} violated its parent weight budget")
        if {row["split_name"] for row in synthetic_labels} != {"train"}:
            raise ValueError(f"Step21 track {track_name} emitted a non-train synthetic row")
        if any(bool_value(row.get("benchmark_eligible")) for row in synthetic_labels):
            raise ValueError(f"Step21 track {track_name} emitted benchmark-eligible synthetic data")

        paths = {
            "profiles": track_root / "synthetic_seller_profiles.jsonl",
            "items": track_root / "synthetic_market_items.csv",
            "labels": track_root / "synthetic_pair_labels.step5_compatible.csv",
            "duplication": track_root / "equal_weight_duplication_control.step5_compatible.csv",
            "lineage": track_root / "synthetic_pair_lineage.csv",
        }
        write_jsonl(paths["profiles"], synthetic_profiles)
        write_csv(
            paths["items"],
            item_rows,
            [
                "vendor",
                "ship_from",
                "title",
                "description",
                "price",
                "category",
                "market",
                "synthetic_pair_uid",
                "synthetic_side",
                "parent_pair_uid",
                "parent_split_component_id",
                "track",
                "transform",
                "synthetic_train_only",
            ],
        )
        write_csv(paths["labels"], synthetic_labels, label_fields)
        write_csv(paths["duplication"], duplicate_labels, label_fields)
        lineage_fields = list(lineage_rows[0]) if lineage_rows else []
        write_csv(paths["lineage"], lineage_rows, lineage_fields)
        track_output = {
            name: str(path.relative_to(ROOT)).replace("\\", "/")
            for name, path in paths.items()
        }
        all_output_paths.extend(paths.values())
        summaries[track_name] = {
            "description": track_cfg["description"],
            "real_parent_pair_count": len(parents),
            "real_parent_component_count": len(
                {parent[0]["split_component_id"] for parent in parents}
            ),
            "synthetic_pair_count": len(synthetic_labels),
            "synthetic_seller_profile_count": len(synthetic_profiles),
            "variants_per_parent_pair": variants,
            "parent_evidence_type_counts": dict(
                sorted(Counter(item[1]["evidence_type"] for item in parents).items())
            ),
            "parent_label_tier_counts": dict(
                sorted(Counter(item[0].get("label_tier", "") for item in parents).items())
            ),
            "parent_training_weight_total": parent_weight_total,
            "synthetic_training_weight_total": synthetic_weight_total,
            "effective_independent_sample_count": len(
                {parent[0]["split_component_id"] for parent in parents}
            ),
            "redaction_diagnostics": dict(sorted(redaction_totals.items())),
            "synthetic_pair_text_change_counts": dict(
                sorted(
                    Counter(
                        "both_changed"
                        if row["left_text_changed"] == "1"
                        and row["right_text_changed"] == "1"
                        else "one_side_changed"
                        if row["left_text_changed"] == "1"
                        or row["right_text_changed"] == "1"
                        else "no_text_change"
                        for row in lineage_rows
                    ).items()
                )
            ),
            "output_paths": track_output,
        }

    summary = {
        "step": "step21_synthetic_train_only_generation",
        "policy_version": policy["version"],
        "status": "generated_train_only_not_benchmark",
        "tracks": summaries,
        "signal_source_summary": signal_summary,
        "input_manifest_sha256": canonical_hash(input_manifest),
        "scientific_interpretation": {
            "new_real_positive_count": 0,
            "new_independent_identity_count": 0,
            "may_be_used_for_training_augmentation": True,
            "may_be_used_for_validation_or_test": False,
            "must_compare_against_equal_effective_weight_duplication": True,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, summary)
    manifest = {
        "policy": str(policy_path.relative_to(ROOT)).replace("\\", "/"),
        "policy_sha256": sha256(policy_path),
        "producer": str(producer_path.relative_to(ROOT)).replace("\\", "/"),
        "producer_sha256": sha256(producer_path),
        "selected_tracks": selected_tracks,
        "inputs": input_manifest,
        "outputs": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted([*all_output_paths, summary_path])
        ],
        "outputs_sha256": canonical_hash(
            [str(path.relative_to(ROOT)).replace("\\", "/") for path in all_output_paths]
        ),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
