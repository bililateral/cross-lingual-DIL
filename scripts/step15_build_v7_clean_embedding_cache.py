#!/usr/bin/env python3
"""Build identifier-redacted Multilingual-E5 seller caches for the v7 clean ranker."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import step3_build_seller_profiles as step3
import step7_build_semantic_pair_features as semantic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step15_v7_two_stage_policy.json"
DEFAULT_PROSPECTIVE_POLICY = ROOT / "schema" / "step20_prospective_holdout_policy.json"


GENERIC_IDENTIFIER_RULES = (
    ("pgp_block", step3.PGP_BLOCK_RE),
    ("pgp_fingerprint", step3.PGP_FINGERPRINT_RE),
    ("email", step3.EMAIL_RE),
    ("jabber", step3.JABBER_RE),
    ("url", step3.URL_RE),
    ("bare_domain", step3.BARE_DOMAIN_RE),
    ("crypto_wallet", step3.CRYPTO_WALLET_RE),
    ("phone_context", step3.PHONE_CONTEXT_RE),
    ("telegram_profile", step3.TELEGRAM_RE),
    ("wickr", step3.WICKR_RE),
    ("wechat_profile", step3.WECHAT_RE),
    ("qq_profile", step3.QQ_RE),
    ("wechat_item", step3.WECHAT_ITEM_RE),
    ("qq_item", step3.QQ_ITEM_RE),
    ("bat", step3.BAT_RE),
) + tuple(
    (f"telegram_item_{rule_name}", pattern)
    for rule_name, pattern in step3.TELEGRAM_ITEM_PATTERNS
)
MAX_REDACTION_PASSES = 8


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def directory_fingerprint(path: Path) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"Clean semantic model directory is missing: {path}")
    records = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        records.append(
            {
                "path": str(file_path.relative_to(path)).replace("\\", "/"),
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256(file_path),
            }
        )
    if not records:
        raise ValueError(f"Clean semantic model directory is empty: {path}")
    return {
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files_sha256": canonical_hash(records),
    }


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_signal_literal(contact_type: str, value: str) -> str | None:
    token = str(value or "").strip()
    if not token:
        return None
    compact = re.sub(r"\s+", "", token)
    if contact_type == "seller_alias":
        if compact.isdigit():
            return token if len(compact) >= 5 else None
        if re.search(r"[\u3400-\u9fff]", compact):
            return token if len(compact) >= 2 else None
        return token if len(compact) >= 4 else None
    if contact_type in {"qq", "phone"}:
        return token if len(re.sub(r"\D", "", compact)) >= 5 else None
    if contact_type in {"pgp_public_key", "pgp_fingerprint", "crypto_wallet"}:
        return token if len(compact) >= 12 else None
    return token if len(compact) >= 4 else None


def signal_literals_by_seller(path: Path) -> tuple[dict[str, list[str]], dict]:
    literals: dict[str, set[str]] = defaultdict(set)
    type_counts: Counter[str] = Counter()
    rows = load_csv(path)
    for row in rows:
        seller_uid = str(row.get("seller_uid", "")).strip()
        contact_type = str(row.get("contact_type", "")).strip().lower()
        if not seller_uid or not contact_type:
            continue
        type_counts[contact_type] += 1
        for field in ("raw_value", "normalized_value"):
            literal = safe_signal_literal(contact_type, row.get(field, ""))
            if literal:
                literals[seller_uid].add(literal)
    return (
        {
            seller_uid: sorted(values, key=lambda value: (-len(value), value.casefold()))
            for seller_uid, values in literals.items()
        },
        {
            "signal_row_count": len(rows),
            "signal_type_counts": dict(sorted(type_counts.items())),
            "seller_with_signal_literal_count": len(literals),
        },
    )


def build_content_text(profile: dict, cfg: dict) -> str:
    sections = []
    for field in cfg["text_fields"]:
        value = str(profile.get(field, "") or "").strip()
        if value:
            sections.append(value)
    return "\n".join(sections)


def normalize_redacted_text(text: str) -> str:
    output = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\s*\n\s*", "\n", output).strip()


def redact_identifiers(text: str, literals: list[str]) -> tuple[str, dict]:
    output = str(text or "")
    generic_matches = 0
    literal_matches = 0
    redaction_pass_count = 0
    for redaction_pass_count in range(1, MAX_REDACTION_PASSES + 1):
        before = output
        for _, pattern in GENERIC_IDENTIFIER_RULES:
            output, count = pattern.subn(" ", output)
            generic_matches += int(count)
        for literal in literals:
            pattern = re.compile(re.escape(literal), re.IGNORECASE)
            output, count = pattern.subn(" ", output)
            literal_matches += int(count)
        output = normalize_redacted_text(output)
        if output == before:
            break
    else:
        raise ValueError(
            f"Identifier redaction did not reach a fixed point after {MAX_REDACTION_PASSES} passes"
        )
    return output, {
        "generic_identifier_match_count": generic_matches,
        "signal_literal_match_count": literal_matches,
        "redaction_pass_count": redaction_pass_count,
    }


def assert_no_known_identifier_residue(text: str, literals: list[str], seller_uid: str) -> None:
    folded = text.casefold()
    residual = [literal for literal in literals if literal.casefold() in folded]
    if residual:
        raise ValueError(
            f"Identifier redaction left a known Step3 signal in clean text for {seller_uid}: "
            f"{residual[0][:40]}"
        )
    for rule_name, pattern in GENERIC_IDENTIFIER_RULES:
        match = pattern.search(text)
        if match:
            residue = match.group(0)
            residue_sha256 = hashlib.sha256(residue.encode("utf-8")).hexdigest()[:16]
            raise ValueError(
                f"Identifier redaction left high-precision rule={rule_name} for {seller_uid}; "
                f"residue_length={len(residue)} residue_sha256={residue_sha256}"
            )


def pool_specs(v7_policy: dict, prospective_policy: dict | None) -> dict[str, dict]:
    specs = {}
    for pool_name, cfg in v7_policy["pools"].items():
        specs[pool_name] = {
            "seller_profiles": cfg["seller_profiles"],
            "item_identity_signals": cfg["item_identity_signals"],
            "canonical_pair_features": cfg["canonical_pair_features"],
            "clean_e5_cache_metadata": cfg["clean_e5_cache_metadata"],
            "clean_e5_cache_matrix": cfg["clean_e5_cache_matrix"],
            "manifest_output": v7_policy["clean_semantic_encoder"]["manifest_output"],
        }
    if prospective_policy is not None:
        upstream = prospective_policy["prospective_upstream"]
        specs["zh_prospective"] = {
            "seller_profiles": upstream["seller_profiles"],
            "item_identity_signals": upstream["item_identity_signals"],
            "canonical_pair_features": upstream["canonical_pair_features"],
            "clean_e5_cache_metadata": upstream["clean_e5_cache_metadata"],
            "clean_e5_cache_matrix": upstream["clean_e5_cache_matrix"],
            "manifest_output": upstream["clean_e5_manifest"],
        }
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--prospective-policy", default=None)
    parser.add_argument("--pool", action="append")
    parser.add_argument("--device", default=None)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    clean_cfg = policy["clean_semantic_encoder"]
    prospective_policy = None
    prospective_policy_path = None
    if args.prospective_policy:
        prospective_policy_path = resolve(args.prospective_policy)
        prospective_policy = json.loads(prospective_policy_path.read_text(encoding="utf-8"))
    specs = pool_specs(policy, prospective_policy)
    selected_pools = args.pool or list(policy["pools"])
    unknown = sorted(set(selected_pools) - set(specs))
    if unknown:
        raise ValueError(f"Unknown clean embedding pools: {unknown}")
    if clean_cfg["replacement"] != "single_space_no_identifier_presence_marker":
        raise ValueError("V7 clean embedding redaction must not expose an identifier-presence marker")
    text_fields = set(clean_cfg["text_fields"])
    excluded_fields = set(clean_cfg["excluded_profile_fields"])
    required_excluded = {
        "source_seller_raw",
        "alias_normalized",
        "source_market_raw",
        "contact_concat_top",
        "structured_snapshot_concat_top",
        "profile_text",
    }
    if text_fields & excluded_fields or not required_excluded.issubset(excluded_fields):
        raise ValueError("V7 clean semantic text/exclusion fields violate identity isolation")
    output_feature = clean_cfg["output_feature"]
    stable_features = policy["inductive_features"]["stable_strict_clean_features"]
    if output_feature not in stable_features:
        raise ValueError("Identifier-redacted E5 cosine is absent from the v7 clean feature view")
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "selected_pools": selected_pools,
                    "model_key": clean_cfg["model_key"],
                    "output_feature": clean_cfg["output_feature"],
                },
                indent=2,
            )
        )
        return

    semantic_policy_path = resolve(clean_cfg["semantic_model_policy"])
    semantic_policy = json.loads(semantic_policy_path.read_text(encoding="utf-8"))
    model_key = clean_cfg["model_key"]
    model_cfg = semantic_policy["embedding_models"][model_key]
    model_directory = semantic.resolve_local_model_dir(model_key, model_cfg)
    model_fingerprint = directory_fingerprint(model_directory)
    prepared = {}
    combined_texts = []
    combined_keys = []
    for pool_name in selected_pools:
        spec = specs[pool_name]
        profiles_path = resolve(spec["seller_profiles"])
        signals_path = resolve(spec["item_identity_signals"])
        pairs_path = resolve(spec["canonical_pair_features"])
        for path in (profiles_path, signals_path, pairs_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing clean embedding input for {pool_name}: {path}")
        profiles_list = semantic.load_jsonl(profiles_path)
        profiles = {str(row["seller_uid"]): row for row in profiles_list}
        if len(profiles) != len(profiles_list):
            raise ValueError(f"Duplicate seller UID in clean embedding profiles: {pool_name}")
        pair_rows = load_csv(pairs_path)
        seller_uids = sorted(
            {
                str(row[key])
                for row in pair_rows
                for key in ("seller_uid_left", "seller_uid_right")
                if str(row.get(key, "")).strip()
            }
        )
        missing_profiles = [seller_uid for seller_uid in seller_uids if seller_uid not in profiles]
        if missing_profiles:
            raise ValueError(
                f"Clean embedding seller profile missing for {pool_name}: {missing_profiles[0]}"
            )
        literals, signal_diagnostics = signal_literals_by_seller(signals_path)
        clean_texts = []
        redaction_counts: Counter[str] = Counter()
        empty_after_redaction = 0
        for seller_uid in seller_uids:
            source_text = build_content_text(profiles[seller_uid], clean_cfg)
            seller_literals = list(literals.get(seller_uid, []))
            for alias_field in ("source_seller_raw", "alias_normalized"):
                alias_literal = safe_signal_literal("seller_alias", profiles[seller_uid].get(alias_field, ""))
                if alias_literal:
                    seller_literals.append(alias_literal)
            seller_literals = sorted(
                set(seller_literals), key=lambda value: (-len(value), value.casefold())
            )
            clean_text, diagnostics = redact_identifiers(source_text, seller_literals)
            assert_no_known_identifier_residue(clean_text, seller_literals, seller_uid)
            redaction_counts.update(diagnostics)
            if diagnostics["redaction_pass_count"] > 2:
                redaction_counts["fixed_point_extra_pass_seller_count"] += 1
            redaction_counts["max_redaction_pass_count"] = max(
                redaction_counts["max_redaction_pass_count"],
                diagnostics["redaction_pass_count"],
            )
            if not clean_text:
                clean_text = "content unavailable"
                empty_after_redaction += 1
            clean_texts.append(clean_text)
            combined_keys.append(f"{pool_name}:{seller_uid}")
            combined_texts.append(clean_text)
        prepared[pool_name] = {
            "spec": spec,
            "seller_uids": seller_uids,
            "clean_texts": clean_texts,
            "input_paths": (profiles_path, signals_path, pairs_path),
            "diagnostics": {
                **signal_diagnostics,
                "seller_count": len(seller_uids),
                "pair_count": len(pair_rows),
                "generic_identifier_match_count": redaction_counts[
                    "generic_identifier_match_count"
                ],
                "signal_literal_match_count": redaction_counts["signal_literal_match_count"],
                "redaction_pass_count_total": redaction_counts["redaction_pass_count"],
                "fixed_point_extra_pass_seller_count": redaction_counts[
                    "fixed_point_extra_pass_seller_count"
                ],
                "max_redaction_pass_count": redaction_counts["max_redaction_pass_count"],
                "empty_after_redaction_count": empty_after_redaction,
                "clean_text_corpus_sha256": canonical_hash(
                    list(zip(seller_uids, clean_texts, strict=True))
                ),
            },
        }

    torch, tokenizer_cls, model_cls, _ = semantic.require_torch_and_transformers()
    device = semantic.choose_device(torch, semantic_policy["device_preference"], args.device)
    embeddings = semantic.encode_texts(
        model_key,
        model_cfg,
        combined_texts,
        device,
        torch,
        tokenizer_cls,
        model_cls,
    )
    if len(embeddings) != len(combined_keys) or not np.all(np.isfinite(embeddings)):
        raise ValueError("Identifier-redacted E5 encoding returned an invalid matrix")

    output_paths = []
    for pool_name in selected_pools:
        spec = specs[pool_name]
        output_paths.extend(
            [resolve(spec["clean_e5_cache_metadata"]), resolve(spec["clean_e5_cache_matrix"])]
        )
    manifest_paths = {resolve(specs[pool_name]["manifest_output"]) for pool_name in selected_pools}
    if len(manifest_paths) != 1:
        raise ValueError("Selected clean embedding pools must share one aggregate manifest")
    manifest_path = next(iter(manifest_paths))
    publication_root = manifest_path.parent
    if any(path.parent != publication_root for path in output_paths):
        raise ValueError("Clean embedding outputs must share one atomic publication directory")
    staging_root = publication_root.with_name(f".{publication_root.name}.incomplete")
    if publication_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Clean embedding final or incomplete directory exists: {publication_root} / {staging_root}"
        )
    staging_root.mkdir(parents=True, exist_ok=False)

    records = {}
    start = 0
    for pool_name in selected_pools:
        item = prepared[pool_name]
        count = len(item["seller_uids"])
        matrix = np.asarray(embeddings[start : start + count], dtype=np.float32)
        start += count
        spec = item["spec"]
        final_matrix = resolve(spec["clean_e5_cache_matrix"])
        final_metadata = resolve(spec["clean_e5_cache_metadata"])
        staged_matrix = staging_root / final_matrix.name
        staged_metadata = staging_root / final_metadata.name
        np.save(staged_matrix, matrix)
        metadata = {
            "model_key": f"{model_key}_identifier_redacted_v7",
            "model_repo_id": model_cfg["repo_id"],
            "model_local_path": model_cfg["local_path"],
            "model_directory_fingerprint": model_fingerprint,
            "identifier_redacted": True,
            "clean_text_fields": clean_cfg["text_fields"],
            "excluded_profile_fields": clean_cfg["excluded_profile_fields"],
            "seller_uids": item["seller_uids"],
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "matrix_sha256": sha256(staged_matrix),
            "redaction_diagnostics": item["diagnostics"],
            "inputs": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                for path in item["input_paths"]
            },
            "producer_sha256": sha256(Path(__file__).resolve()),
        }
        metadata["metadata_sha256"] = canonical_hash(metadata)
        staged_metadata.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        records[pool_name] = {
            "metadata_path": str(final_metadata.relative_to(ROOT)).replace("\\", "/"),
            "matrix_path": str(final_matrix.relative_to(ROOT)).replace("\\", "/"),
            "metadata_sha256": sha256(staged_metadata),
            "matrix_sha256": sha256(staged_matrix),
            "seller_count": count,
            "redaction_diagnostics": item["diagnostics"],
        }
    manifest = {
        "step": "step15_build_v7_clean_embedding_cache",
        "version": policy["version"],
        "model_key": model_key,
        "model_directory_fingerprint": model_fingerprint,
        "identifier_redacted": True,
        "selected_pools": selected_pools,
        "combined_seller_count": len(combined_keys),
        "records": records,
        "v7_policy_sha256": sha256(policy_path),
        "semantic_policy_sha256": sha256(semantic_policy_path),
        "prospective_policy_sha256": None
        if prospective_policy_path is None
        else sha256(prospective_policy_path),
        "producer_sha256": sha256(Path(__file__).resolve()),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    staged_manifest = staging_root / manifest_path.name
    staged_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    staging_root.replace(publication_root)
    print(
        json.dumps(
            {
                "status": "pass",
                "device": str(device),
                "selected_pools": selected_pools,
                "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
