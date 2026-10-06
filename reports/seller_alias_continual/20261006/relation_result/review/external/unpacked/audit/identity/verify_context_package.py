#!/usr/bin/env python3
"""Read-only ZIP and extracted-payload identity audit, standard library only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import stat
import zipfile

EXPECTED_BYTES = 5077487
EXPECTED_SHA256 = "3c11a9640c13438d9eb7ace1005183d83ccafbdca7036fa39e0689b06731625a"
INVENTORY = "reports/seller_alias_continual/20261006/relation_result/review/context_inventory.json"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = args.zip.read_bytes()
    assert len(payload) == EXPECTED_BYTES
    assert digest(payload) == EXPECTED_SHA256
    checked = []
    with zipfile.ZipFile(args.zip) as archive:
        members = archive.infolist()
        assert len(members) == 272
        assert len({info.filename for info in members}) == len(members)
        assert archive.testzip() is None
        for info in members:
            name = PurePosixPath(info.filename)
            assert not name.is_absolute() and ".." not in name.parts
            assert not stat.S_ISLNK(info.external_attr >> 16)
            assert not info.is_dir()
        inventory = json.loads(archive.read(INVENTORY))
        records = inventory["files"]
        assert len(records) == 271
        assert len({r["path"] for r in records}) == 271
        assert {info.filename for info in members} == {r["path"] for r in records} | {INVENTORY}
        for record in records:
            raw = archive.read(record["path"])
            local = args.extracted_root / record["path"]
            assert local.resolve().is_relative_to(args.extracted_root.resolve())
            assert len(raw) == record["bytes"]
            assert digest(raw) == record["sha256"]
            assert local.read_bytes() == raw
            checked.append(record)
        assert (args.extracted_root / INVENTORY).read_bytes() == archive.read(INVENTORY)
    result = {
        "status": "PASS_CURRENT_RESULT_CONTEXT_IDENTITY",
        "scope": "Actual submitted ZIP and all extracted inventory payload bytes; no model/data access or training.",
        "python": platform.python_version(),
        "attachment": {"name": args.zip.name, "bytes": len(payload), "sha256": digest(payload)},
        "zip_members": len(members), "regular_files": len(members),
        "inventory_payloads": len(checked), "all_payload_bytes_and_sha256_match": True,
        "all_extracted_files_equal_zip": True, "zip_crc_pass": True,
        "duplicate_or_unsafe_names": False, "mismatches": [],
        "inventory": {"path": INVENTORY, "sha256": digest((args.extracted_root / INVENTORY).read_bytes())},
        "checked_files": checked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "checked_files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
