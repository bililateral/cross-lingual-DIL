#!/usr/bin/env python3
"""Build a label-free pair-UID exclusion registry for prospective Step28 scoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import step28_common as base


DEFAULT_SOURCE = base.ROOT / "reports" / "step5_zh_target_strict_frozen_silver_labels.csv"
DEFAULT_OUTPUT = (
    base.ROOT / "reports" / "step28_review_exclusions"
    / "known_reviewed_pair_uids.zh_target_strict_20260720.csv"
)
DEFAULT_SUMMARY = (
    base.ROOT / "reports" / "step28_review_exclusions"
    / "known_reviewed_pair_uids.zh_target_strict_20260720.summary.json"
)


def extract_pair_uids(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header.count("pair_uid") != 1:
            raise ValueError("Step28 exclusion source must contain exactly one pair_uid column")
        pair_index = header.index("pair_uid")
        pair_uids = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError("Step28 exclusion source contains a malformed row")
            pair_uid = row[pair_index].strip()
            if not pair_uid:
                raise ValueError("Step28 exclusion source contains a blank pair_uid")
            pair_uids.append(pair_uid)
    if len(pair_uids) != len(set(pair_uids)):
        raise ValueError("Step28 exclusion source contains duplicate pair_uids")
    return sorted(pair_uids), header


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()
    source = base.resolve(args.source)
    output = base.resolve(args.output)
    summary = base.resolve(args.summary)
    pair_uids, source_header = extract_pair_uids(source)
    base.write_csv_immutable(output, [{"pair_uid": uid} for uid in pair_uids], ["pair_uid"])
    payload = {
        "purpose": "exclude every historically reviewed pair before prospective Step28 scoring",
        "pair_uid_count": len(pair_uids),
        "source_path": str(source.relative_to(base.ROOT)).replace("\\", "/"),
        "source_sha256": base.sha256_file(source),
        "output_path": str(output.relative_to(base.ROOT)).replace("\\", "/"),
        "output_sha256": base.sha256_file(output),
        "output_columns": ["pair_uid"],
        "source_non_uid_columns_excluded": [name for name in source_header if name != "pair_uid"],
        "label_values_serialized_to_output": False,
        "old_labels_used_for_model_fitting_selection_or_scoring": False,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    base.write_json_immutable(summary, payload)
    print(json.dumps({"status": "ok", "pair_uid_count": len(pair_uids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
