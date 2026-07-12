#!/usr/bin/env python3
"""Capture or verify that a Step4 rerun preserves the candidate pair universe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES = {
    "en_content_train_pool": "reports/step4_en_silver_candidate_pairs.csv",
    "zh_target_strict": "reports/step4_zh_target_strict_silver_candidate_pairs.csv",
    "zh_target_aux": "reports/step4_zh_target_aux_silver_candidate_pairs.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["capture", "verify"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--verification-output", default=None)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def universe_record(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        pair_uids = sorted(row["pair_uid"] for row in csv.DictReader(handle))
    if len(pair_uids) != len(set(pair_uids)):
        raise ValueError(f"Duplicate Step4 pair_uid values in {path}")
    encoded = "\n".join(pair_uids).encode("utf-8")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "pair_count": len(pair_uids),
        "pair_uid_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def current_records() -> dict[str, dict]:
    records = {}
    for pool, path_value in DEFAULT_FILES.items():
        path = resolve(path_value)
        if not path.exists():
            raise FileNotFoundError(path)
        records[pool] = universe_record(path)
    return records


def main() -> None:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    current = current_records()
    if args.mode == "capture":
        payload = {
            "step": "step4_candidate_universe_capture",
            "purpose": "prove_feature_lineage_changes_do_not_change_pair_sample_universe",
            "pools": current,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "captured", "manifest": str(manifest_path.relative_to(ROOT))}, indent=2))
        return
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
    def comparable(record: dict | None) -> dict | None:
        if record is None:
            return None
        return {
            "pair_count": int(record["pair_count"]),
            "pair_uid_sha256": str(record["pair_uid_sha256"]),
        }

    mismatches = {
        pool: {"before": baseline["pools"].get(pool), "after": current.get(pool)}
        for pool in sorted(set(baseline["pools"]) | set(current))
        if comparable(baseline["pools"].get(pool)) != comparable(current.get(pool))
    }
    verification = {
        "step": "step4_candidate_universe_verification",
        "status": "pass" if not mismatches else "fail",
        "baseline_manifest": str(manifest_path.relative_to(ROOT)),
        "pools": current,
        "mismatches": mismatches,
    }
    output_path = resolve(
        args.verification_output
        or "reports/step15_v6/manifests/step4_candidate_universe_verification.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if mismatches:
        raise SystemExit(f"Step4 candidate universe changed: {mismatches}")
    print(json.dumps({"status": "pass", "verification": str(output_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
