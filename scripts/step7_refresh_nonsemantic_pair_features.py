#!/usr/bin/env python3
"""Refresh Step7 nonsemantic columns while proving semantic scores are unchanged."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "step7_transfer_safe_pair_feature_schema.json"
POOLS = {
    "en_content_train_pool": (
        ROOT / "reports" / "step7_pair_feature_preview.en_content_train_pool.csv",
        ROOT / "reports" / "step7_pair_features.en_content_train_pool.csv",
    ),
    "zh_target_strict": (
        ROOT / "reports" / "step7_pair_feature_preview.zh_target_strict.csv",
        ROOT / "reports" / "step7_pair_features.zh_target_strict.csv",
    ),
    "zh_target_aux": (
        ROOT / "reports" / "step7_pair_feature_preview.zh_target_aux.csv",
        ROOT / "reports" / "step7_pair_features.zh_target_aux.csv",
    ),
}


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def semantic_hash(rows: list[dict], fields: list[str]) -> str:
    canonical = [
        [row["pair_uid"], *[str(row.get(field, "")) for field in fields]]
        for row in sorted(rows, key=lambda value: value["pair_uid"])
    ]
    encoded = json.dumps(canonical, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-manifest",
        default="reports/step15_v6/manifests/step7_nonsemantic_refresh_manifest.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    output_fields = list(schema["semantic_enriched_output_fields"])
    semantic_fields = list(schema["feature_views"]["future_multilingual_semantics"])
    records = {}
    pending_writes: list[tuple[Path, list[dict]]] = []
    for pool, (preview_path, semantic_path) in POOLS.items():
        preview_rows = load_csv(preview_path)
        semantic_rows = load_csv(semantic_path)
        preview_index = {row["pair_uid"]: row for row in preview_rows}
        semantic_index = {row["pair_uid"]: row for row in semantic_rows}
        if set(preview_index) != set(semantic_index):
            missing = sorted(set(preview_index) - set(semantic_index))
            extra = sorted(set(semantic_index) - set(preview_index))
            raise ValueError(
                f"Step7 nonsemantic refresh pair universe mismatch for {pool}: "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
        before_hash = semantic_hash(semantic_rows, semantic_fields)
        merged_rows = []
        for pair_uid in [row["pair_uid"] for row in preview_rows]:
            preview = preview_index[pair_uid]
            existing = semantic_index[pair_uid]
            merged_rows.append(
                {
                    field: existing.get(field, "") if field in semantic_fields else preview.get(field, "")
                    for field in output_fields
                }
            )
        after_hash = semantic_hash(merged_rows, semantic_fields)
        if before_hash != after_hash:
            raise ValueError(f"Step7 semantic columns changed during nonsemantic refresh for {pool}")
        if any(
            not str(row.get("candidate_rule_count_non_identifier", "")).strip()
            for row in merged_rows
        ):
            raise ValueError(f"Step7 refresh failed to propagate candidate_rule_count_non_identifier for {pool}")
        records[pool] = {
            "pair_count": len(merged_rows),
            "semantic_column_count": len(semantic_fields),
            "semantic_hash_before": before_hash,
            "semantic_hash_after": after_hash,
            "semantic_scores_unchanged": True,
            "output": str(semantic_path.relative_to(ROOT)),
        }
        pending_writes.append((semantic_path, merged_rows))
    if not args.validate_only:
        for path, rows in pending_writes:
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=output_fields)
                writer.writeheader()
                writer.writerows(rows)
    manifest = {
        "step": "step7_refresh_nonsemantic_pair_features",
        "mode": "validate_only" if args.validate_only else "write_verified_nonsemantic_refresh",
        "schema": str(SCHEMA_PATH.relative_to(ROOT)),
        "candidate_universe_changed": False,
        "semantic_scores_recomputed": False,
        "pools": records,
    }
    output_path = Path(args.output_manifest)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output_path.relative_to(ROOT)), "pools": records}, indent=2))


if __name__ == "__main__":
    main()
