#!/usr/bin/env python3
"""Generate matched, identifier-free Step27 train-only profile views."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import step15_build_v7_clean_embedding_cache as redaction
import step27_common as common


def synthetic_pair_row(
    parent: dict,
    *,
    pair_uid: str,
    left_uid: str,
    right_uid: str,
    seed: int,
    variant_index: int,
    transform_name: str,
    child_weight: float,
) -> dict:
    return {
        "pair_uid": pair_uid,
        "track": parent["track"],
        "matched_set_id": parent["matched_set_id"],
        "parent_pair_uid": parent["parent_pair_uid"],
        "review_label": parent["review_label"],
        "seller_uid_left": left_uid,
        "seller_uid_right": right_uid,
        "split_name": "train",
        "component_id": parent["component_id"],
        "fold": parent["fold"],
        "evidence_type": parent["evidence_type"],
        "silver_train_only": parent["silver_train_only"],
        "training_sample_weight": f"{child_weight:.12f}",
        "seed": str(seed),
        "variant_index": str(variant_index),
        "transform": transform_name,
        "synthetic_train_only": "1",
        "benchmark_eligible": "0",
        "independent_real_sample": "0",
        "fabricated_identifier_count": "0",
        "fabricated_market_or_provenance": "0",
    }


def duplication_row(parent: dict, *, pair_uid: str, child_weight: float, seed: int, variant: int) -> dict:
    return {
        "pair_uid": pair_uid,
        "track": parent["track"],
        "matched_set_id": parent["matched_set_id"],
        "parent_pair_uid": parent["parent_pair_uid"],
        "review_label": parent["review_label"],
        "seller_uid_left": parent["seller_uid_left"],
        "seller_uid_right": parent["seller_uid_right"],
        "split_name": "train",
        "component_id": parent["component_id"],
        "fold": parent["fold"],
        "evidence_type": parent["evidence_type"],
        "silver_train_only": parent["silver_train_only"],
        "training_sample_weight": f"{child_weight:.12f}",
        "seed": str(seed),
        "variant_index": str(variant),
        "transform": "equal_effective_weight_duplication",
        "synthetic_train_only": "1",
        "benchmark_eligible": "0",
        "independent_real_sample": "0",
        "fabricated_identifier_count": "0",
        "fabricated_market_or_provenance": "0",
    }


def transformed_parent_pair(
    parent: dict,
    profiles: dict[str, dict],
    clean_cache: dict[str, tuple[dict[str, str], list[str], dict]],
    transform_name: str,
    seed: int,
    variant_index: int,
    synthetic_uid_prefix: str,
) -> tuple[dict[str, str], dict[str, str], dict, dict] | None:
    pair_uid = parent["parent_pair_uid"]
    sides = []
    for side, field in (("left", "seller_uid_left"), ("right", "seller_uid_right")):
        parent_uid = parent[field]
        if parent_uid not in profiles or parent_uid not in clean_cache:
            raise ValueError(f"Step27 parent profile is missing: {pair_uid}:{parent_uid}")
        clean_fields, literals, clean_diagnostics = clean_cache[parent_uid]
        rng = common.deterministic_rng(seed, parent["matched_set_id"], pair_uid, str(variant_index), side)
        transformed = common.transform_fields(clean_fields, transform_name, rng)
        synthetic_uid = (
            f"{synthetic_uid_prefix}/{parent['track']}/seed_{seed}/"
            f"{parent['matched_set_id']}/{parent['parent_role']}/v{variant_index:02d}/{side}"
        )
        transformed, post_diagnostics = common.redact_transformed_fields(
            transformed, literals, synthetic_uid
        )
        changed = common.render_profile_text(transformed) != common.render_profile_text(clean_fields)
        sides.append((transformed, changed, clean_diagnostics, post_diagnostics))
    if not any(item[1] for item in sides):
        return None
    return sides[0][0], sides[1][0], sides[0][2] | {f"post_{k}": v for k, v in sides[0][3].items()}, sides[1][2] | {f"post_{k}": v for k, v in sides[1][3].items()}


def choose_matched_transform(
    parents: list[dict],
    profiles: dict[str, dict],
    clean_cache: dict[str, tuple[dict[str, str], list[str], dict]],
    schedule: list[str],
    seed: int,
    variant_index: int,
    synthetic_uid_prefix: str,
) -> tuple[str, dict[str, tuple[dict[str, str], dict[str, str], dict, dict]]] | None:
    """Use one recipe for both labels or skip the entire matched variant."""
    for offset in range(len(schedule)):
        transform = schedule[(variant_index + offset) % len(schedule)]
        results = {}
        for parent in parents:
            result = transformed_parent_pair(
                parent,
                profiles,
                clean_cache,
                transform,
                seed,
                variant_index,
                synthetic_uid_prefix,
            )
            if result is None:
                break
            results[parent["parent_pair_uid"]] = result
        if len(results) == len(parents):
            return transform, results
    return None


def profile_view_rows(profile: dict) -> list[dict]:
    lineage = profile["synthetic_lineage"]
    rows = []
    for field in common.DEFAULT_TEXT_FIELDS:
        for index, segment in enumerate(common.split_segments(profile.get(field, ""))):
            rows.append(
                {
                    "synthetic_seller_uid": profile["seller_uid"],
                    "parent_seller_uid": lineage["parent_seller_uid"],
                    "parent_pair_uid": lineage["parent_pair_uid"],
                    "component_id": lineage["parent_component_id"],
                    "fold": str(lineage["inherited_fold"]),
                    "review_label": lineage["inherited_label"],
                    "track": lineage["track"],
                    "seed": str(lineage["seed"]),
                    "variant_index": str(lineage["variant_index"]),
                    "transform": lineage["transform"],
                    "source_field": field,
                    "segment_index": str(index),
                    "text": segment,
                    "source_market_raw": "",
                    "identifier_value": "",
                    "synthetic_train_only": "1",
                }
            )
    return rows


def generate_track(
    *,
    policy: dict,
    seed: int,
    track: str,
    parent_rows: list[dict],
    profiles: dict[str, dict],
    signal_literals: dict[str, list[str]],
    fields: list[str],
    clean_cfg: dict,
    variants: int,
    child_cap: int,
) -> tuple[list[Path], dict]:
    schedule = common.transform_schedule(policy, variants)
    synthetic_uid_prefix = str(
        policy.get("generation", {}).get("synthetic_uid_prefix", "")
    ).rstrip("/")
    if not synthetic_uid_prefix.startswith("synthetic://step27/"):
        raise ValueError("Step27 synthetic_uid_prefix must be a versioned synthetic://step27 URI")
    clean_cache: dict[str, tuple[dict[str, str], list[str], dict]] = {}
    for parent in parent_rows:
        for field in ("seller_uid_left", "seller_uid_right"):
            seller_uid = parent[field]
            if seller_uid in clean_cache:
                continue
            profile = profiles.get(seller_uid)
            if profile is None:
                raise ValueError(f"Step27 parent seller profile is missing: {seller_uid}")
            cleaned, diagnostics = common.clean_profile_fields(profile, fields, signal_literals)
            if not common.render_profile_text(cleaned):
                raise ValueError(f"Step27 redaction removed all parent content: {seller_uid}")
            literals = common.profile_literals(profile, signal_literals)
            source_text = redaction.build_content_text(profile, clean_cfg)
            exact_text, _ = redaction.redact_identifiers(source_text, literals)
            redaction.assert_no_known_identifier_residue(exact_text, literals, seller_uid)
            if not exact_text:
                exact_text = "content unavailable"
            common.assert_exact_real_text_replay(
                {seller_uid: exact_text},
                {seller_uid: common.render_profile_text(cleaned)},
            )
            clean_cache[seller_uid] = (cleaned, literals, diagnostics)

    by_match: dict[str, list[dict]] = defaultdict(list)
    for row in parent_rows:
        by_match[row["matched_set_id"]].append(row)
    if any({row["review_label"] for row in rows} != {"positive", "negative"} for rows in by_match.values()):
        raise ValueError(f"Step27 {track} parent matches are not label-balanced")

    synthetic_profiles: list[dict] = []
    synthetic_pairs: list[dict] = []
    duplication_pairs: list[dict] = []
    lineage_rows: list[dict] = []
    view_rows: list[dict] = []
    skipped: list[dict] = []
    redaction_totals: Counter[str] = Counter()
    parent_child_weights: Counter[str] = Counter()
    for matched_set_id, matched_parents in sorted(by_match.items()):
        matched_parents = sorted(matched_parents, key=lambda row: row["parent_role"])
        if len(matched_parents) != 2:
            raise ValueError(f"Step27 matched set must contain two parent pairs: {matched_set_id}")
        for variant_index in range(variants):
            chosen = choose_matched_transform(
                matched_parents,
                profiles,
                clean_cache,
                schedule,
                seed,
                variant_index,
                synthetic_uid_prefix,
            )
            if chosen is None:
                if bool(
                    policy.get("generation", {})
                    .get("recipe_contract", {})
                    .get("fail_closed_on_no_op", False)
                ):
                    raise ValueError(
                        "Step27 fail-closed no-op contract could not generate every "
                        f"matched variant: {track}/{matched_set_id}/v{variant_index:02d}"
                    )
                skipped.append(
                    {
                        "track": track,
                        "seed": str(seed),
                        "matched_set_id": matched_set_id,
                        "variant_index": str(variant_index),
                        "reason": "all_label_preserving_recipes_were_no_op_for_at_least_one_label",
                    }
                )
                continue
            transform_name, transformed = chosen
            for parent in matched_parents:
                left_fields, right_fields, left_diag, right_diag = transformed[
                    parent["parent_pair_uid"]
                ]
                redaction_totals.update(left_diag)
                redaction_totals.update(right_diag)
                base = (
                    f"{synthetic_uid_prefix}/{track}/seed_{seed}/{matched_set_id}/"
                    f"{parent['parent_role']}/v{variant_index:02d}"
                )
                left_uid = f"{base}/left"
                right_uid = f"{base}/right"
                pair_uid = f"{left_uid}||{right_uid}"
                parent_weight = float(parent["parent_training_sample_weight"])
                child_weight = parent_weight * 0.5 / variants
                parent_child_weights[parent["parent_pair_uid"]] += child_weight
                left_profile = common.make_synthetic_profile(
                    synthetic_uid=left_uid,
                    parent_uid=parent["seller_uid_left"],
                    parent_pair_uid=parent["parent_pair_uid"],
                    component_id=parent["component_id"],
                    fold=int(parent["fold"]),
                    label=parent["review_label"],
                    track=track,
                    seed=seed,
                    variant_index=variant_index,
                    transform_name=transform_name,
                    fields=left_fields,
                )
                right_profile = common.make_synthetic_profile(
                    synthetic_uid=right_uid,
                    parent_uid=parent["seller_uid_right"],
                    parent_pair_uid=parent["parent_pair_uid"],
                    component_id=parent["component_id"],
                    fold=int(parent["fold"]),
                    label=parent["review_label"],
                    track=track,
                    seed=seed,
                    variant_index=variant_index,
                    transform_name=transform_name,
                    fields=right_fields,
                )
                synthetic_profiles.extend([left_profile, right_profile])
                view_rows.extend(profile_view_rows(left_profile))
                view_rows.extend(profile_view_rows(right_profile))
                synthetic_pairs.append(
                    synthetic_pair_row(
                        parent,
                        pair_uid=pair_uid,
                        left_uid=left_uid,
                        right_uid=right_uid,
                        seed=seed,
                        variant_index=variant_index,
                        transform_name=transform_name,
                        child_weight=child_weight,
                    )
                )
                duplication_pairs.append(
                    duplication_row(
                        parent,
                        pair_uid=f"{synthetic_uid_prefix}/duplication/{track}/seed_{seed}/"
                        f"{matched_set_id}/{parent['parent_role']}/v{variant_index:02d}",
                        child_weight=child_weight,
                        seed=seed,
                        variant=variant_index,
                    )
                )
                lineage_rows.append(
                    {
                        "synthetic_pair_uid": pair_uid,
                        "duplication_control_pair_uid": duplication_pairs[-1]["pair_uid"],
                        "parent_pair_uid": parent["parent_pair_uid"],
                        "parent_seller_uid_left": parent["seller_uid_left"],
                        "parent_seller_uid_right": parent["seller_uid_right"],
                        "component_id": parent["component_id"],
                        "fold": parent["fold"],
                        "review_label": parent["review_label"],
                        "evidence_type": parent["evidence_type"],
                        "track": track,
                        "seed": str(seed),
                        "variant_index": str(variant_index),
                        "transform": transform_name,
                        "left_profile_sha256": common.canonical_hash(left_profile),
                        "right_profile_sha256": common.canonical_hash(right_profile),
                        "parent_training_sample_weight": f"{parent_weight:.12f}",
                        "child_training_sample_weight": f"{child_weight:.12f}",
                        "synthetic_train_only": "1",
                        "benchmark_eligible": "0",
                    }
                )

    if len(synthetic_pairs) > child_cap:
        raise ValueError(f"Step27 {track} exceeded its per-seed child cap")
    if (
        policy.get("generation", {})
        .get("recipe_contract", {})
        .get("fail_closed_on_no_op", False)
        and len(synthetic_pairs) != child_cap
    ):
        raise ValueError(
            f"Step27 {track} did not materialize its complete fail-closed child budget: "
            f"observed={len(synthetic_pairs)} expected={child_cap}"
        )
    if len(synthetic_pairs) != len(duplication_pairs) or len(synthetic_profiles) != 2 * len(
        synthetic_pairs
    ):
        raise ValueError(f"Step27 {track} synthetic/duplication/profile counts disagree")
    if len({row["pair_uid"] for row in synthetic_pairs}) != len(synthetic_pairs):
        raise ValueError(f"Step27 {track} generated duplicate pair UIDs")
    for parent in parent_rows:
        total = parent_child_weights[parent["parent_pair_uid"]]
        maximum = float(parent["parent_training_sample_weight"]) * 0.5 + 1e-12
        if total > maximum:
            raise ValueError(f"Step27 child weight exceeded 0.5 parent budget: {parent['parent_pair_uid']}")
    if any(row["fold"] != next(p["fold"] for p in parent_rows if p["parent_pair_uid"] == row["parent_pair_uid"]) for row in synthetic_pairs):
        raise ValueError(f"Step27 {track} child fold inheritance failed")
    if any(row["component_id"] != next(p["component_id"] for p in parent_rows if p["parent_pair_uid"] == row["parent_pair_uid"]) for row in synthetic_pairs):
        raise ValueError(f"Step27 {track} child component inheritance failed")
    if any(row["review_label"] != next(p["review_label"] for p in parent_rows if p["parent_pair_uid"] == row["parent_pair_uid"]) for row in synthetic_pairs):
        raise ValueError(f"Step27 {track} child label inheritance failed")

    root = common.track_root(policy, seed, track)
    profiles_path = root / "synthetic_profiles.jsonl"
    pairs_path = root / "synthetic_pairs.csv"
    duplication_path = root / "equal_weight_duplication_pairs.csv"
    lineage_path = root / "lineage.csv"
    views_path = root / "synthetic_segment_views.csv"
    skipped_path = root / "skipped_matched_variants.csv"
    common.write_jsonl_immutable(profiles_path, synthetic_profiles)
    common.write_csv_immutable(pairs_path, synthetic_pairs)
    common.write_csv_immutable(duplication_path, duplication_pairs)
    common.write_csv_immutable(lineage_path, lineage_rows)
    common.write_csv_immutable(views_path, view_rows)
    outputs = [profiles_path, pairs_path, duplication_path, lineage_path, views_path]
    if skipped:
        common.write_csv_immutable(skipped_path, skipped)
        outputs.append(skipped_path)
    summary = {
        "track": track,
        "seed": seed,
        "parent_pair_count": len(parent_rows),
        "matched_set_count": len(by_match),
        "synthetic_pair_count": len(synthetic_pairs),
        "synthetic_positive_count": sum(row["review_label"] == "positive" for row in synthetic_pairs),
        "synthetic_negative_count": sum(row["review_label"] == "negative" for row in synthetic_pairs),
        "synthetic_profile_count": len(synthetic_profiles),
        "skipped_matched_variant_count": len(skipped),
        "child_cap": child_cap,
        "maximum_child_weight_relative_to_parent": 0.5,
        "transform_counts": dict(sorted(Counter(row["transform"] for row in synthetic_pairs).items())),
        "redaction_diagnostics": dict(sorted(redaction_totals.items())),
        "fabricated_identifier_count": 0,
        "fabricated_market_or_provenance_count": 0,
        "valid_or_test_child_count": 0,
        "effective_independent_sample_count": len({row["component_id"] for row in parent_rows}),
        "synthetic_uid_prefix": synthetic_uid_prefix,
        "parent_clean_text_exactly_replays_step15_v7": True,
    }
    return outputs, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--track", action="append", choices=["primary", "silver_sensitivity"], dest="tracks")
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_path, policy = common.load_policy(args.policy)
    limits = common.generation_limits(policy)
    seeds = args.seeds or common.generation_seeds(policy)
    tracks = args.tracks or ["primary", "silver_sensitivity"]
    if args.validate_config_only:
        print(json.dumps({"status": "pass", "seeds": seeds, "tracks": tracks, "limits": limits, "numerical_execution_performed": False}, indent=2))
        return

    parent_root = common.parent_root(policy)
    parent_manifest_path = parent_root / "manifest.json"
    if not parent_manifest_path.is_file():
        raise FileNotFoundError("Run step27_build_parent_manifest.py first")
    parent_manifest = common.load_json(parent_manifest_path)
    for record in parent_manifest.get("outputs", []):
        path = common.resolve(record["path"])
        if not path.is_file() or common.sha256_file(path) != record["sha256"]:
            raise ValueError(f"Step27 parent manifest output changed: {path}")
    profiles_path = common.policy_input(policy, "seller_profiles", "zh_seller_profiles")
    signals_path = common.policy_input(policy, "item_identity_signals", "zh_item_identity_signals")
    profiles = common.load_profiles_index(profiles_path)
    signal_literals, signal_summary = redaction.signal_literals_by_seller(signals_path)
    fields = common.text_fields(policy)
    v7_policy_path = common.policy_input(policy, "identifier_redaction_policy")
    v7_policy = common.load_json(v7_policy_path)
    clean_cfg = dict(v7_policy["clean_semantic_encoder"])
    if list(clean_cfg.get("text_fields", [])) != fields:
        raise ValueError("Step27 generation fields do not exactly replay Step15-v7")
    parent_paths = {
        "primary": parent_root / "primary_matched_parents.csv",
        "silver_sensitivity": parent_root / "silver_sensitivity_matched_parents.csv",
    }
    producer = Path(__file__).resolve()
    for seed in seeds:
        run_root = common.seed_root(policy, seed)
        manifest_path = run_root / "generation_manifest.json"
        identity = {
            "stage": "step27_generate_train_only_views",
            "seed": seed,
            "tracks": tracks,
            "policy_sha256": common.sha256_file(policy_path),
            "producer_sha256": common.sha256_file(producer),
            "common_sha256": common.sha256_file(Path(common.__file__).resolve()),
            "shared_dependency_sha256": common.shared_dependency_hashes(),
            "parent_manifest_sha256": common.sha256_file(parent_manifest_path),
            "profiles_sha256": common.sha256_file(profiles_path),
            "signals_sha256": common.sha256_file(signals_path),
            "identifier_redaction_policy_sha256": common.sha256_file(v7_policy_path),
        }
        existing = common.assert_existing_manifest_identity(manifest_path, identity)
        if existing is not None:
            print(json.dumps({"status": "identical_replay", "seed": seed, "manifest": common.relative(manifest_path)}))
            continue
        if run_root.exists() and any(run_root.iterdir()):
            raise ValueError(f"Incomplete or foreign Step27 seed root exists: {run_root}")
        outputs: list[Path] = []
        summaries = {}
        for track in tracks:
            rows = common.load_csv(parent_paths[track])
            track_outputs, summary = generate_track(
                policy=policy,
                seed=seed,
                track=track,
                parent_rows=rows,
                profiles=profiles,
                signal_literals=signal_literals,
                fields=fields,
                clean_cfg=clean_cfg,
                variants=limits["primary_variants"] if track == "primary" else limits["silver_variants"],
                child_cap=limits["primary_child_cap"] if track == "primary" else limits["silver_child_cap"],
            )
            outputs.extend(track_outputs)
            summaries[track] = summary
        summary_path = run_root / "generation_summary.json"
        summary = {
            "step": "step27_generate_train_only_views",
            "status": "generated_train_only_views",
            "seed": seed,
            "tracks": summaries,
            "primary_and_silver_physically_separate": True,
            "matched_transform_schedule": True,
            "all_children_inherit_parent_fold_component_label": True,
            "parent_pair_features_copied": False,
            "identifier_or_market_provenance_fabricated": False,
            "signal_source_summary": signal_summary,
        }
        summary["summary_content_sha256"] = common.canonical_hash(summary)
        common.write_json_immutable(summary_path, summary)
        outputs.append(summary_path)
        common.write_manifest_immutable(
            manifest_path,
            stage="step27_generate_train_only_views",
            identity=identity,
            inputs=[policy_path, parent_manifest_path, profiles_path, signals_path, v7_policy_path, *[parent_paths[t] for t in tracks]],
            outputs=outputs,
            extra={"summary_sha256": common.sha256_file(summary_path)},
        )
        print(json.dumps({"status": "pass", "seed": seed, "tracks": summaries, "manifest": common.relative(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
