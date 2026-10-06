"""Read-only verification of the authorized revision ZIP and its extracted payloads.

Uses only the Python standard library; does not import or execute project code.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

EXPECTED_BYTES = 5_257_725
EXPECTED_SHA256 = "480371343ea0be50e4f8fcc614ec1aee5b35b20593151b7fbb3932e75669a9b5"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    zip_bytes = args.zip.read_bytes()
    errors: list[str] = []
    checks: dict[str, bool] = {}
    payload_records: list[dict] = []
    checks["archive_bytes"] = len(zip_bytes) == EXPECTED_BYTES
    checks["archive_sha256"] = digest(zip_bytes) == EXPECTED_SHA256
    with ZipFile(args.zip) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        files = [entry.filename for entry in entries if not entry.is_dir()]
        dirs = [entry.filename for entry in entries if entry.is_dir()]
        checks["entry_counts"] = (len(entries), len(dirs), len(files)) == (319, 14, 305)
        checks["unique_names"] = len(names) == len(set(names))
        checks["safe_paths"] = all(
            not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts
            and "\\" not in name
            and ":" not in name
            for name in names
        )
        checks["no_symlinks"] = all(
            not stat.S_ISLNK(entry.external_attr >> 16) for entry in entries
        )
        checks["all_crc"] = archive.testzip() is None
        manifest_bytes = archive.read("manifest.json")
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        declared = manifest["files"]
        declared_names = [entry["path"] for entry in declared]
        checks["manifest_304_unique_payloads"] = len(declared_names) == len(set(declared_names)) == 304
        checks["manifest_exact_payload_set"] = set(declared_names) == set(files) - {"manifest.json"}
        checks["extracted_manifest_identical"] = (args.root / "manifest.json").read_bytes() == manifest_bytes
        for entry in declared:
            name = entry["path"]
            packed = archive.read(name)
            unpacked = (args.root / name).read_bytes()
            record = {
                "path": name,
                "bytes": len(packed),
                "sha256": digest(packed),
                "declared_size_matches": len(packed) == entry["bytes"],
                "declared_sha256_matches": digest(packed) == entry["sha256"],
                "extracted_bytes_identical": unpacked == packed,
            }
            payload_records.append(record)
            if not all(record[key] for key in (
                "declared_size_matches", "declared_sha256_matches", "extracted_bytes_identical"
            )):
                errors.append(name)
        checks["all_payload_sizes_and_hashes_and_extraction"] = not errors
    failed = [key for key, value in checks.items() if not value]
    result = {
        "scope": "Web review environment, actual standard-library file verification only",
        "status": "PASS" if not failed else "FAIL",
        "input_zip": str(args.zip.resolve()),
        "extracted_root": str(args.root.resolve()),
        "actual_archive_bytes": len(zip_bytes),
        "actual_archive_sha256": digest(zip_bytes),
        "archive_entries": len(entries),
        "archive_directories": len(dirs),
        "archive_files": len(files),
        "manifest_payload_count": len(payload_records),
        "manifest_bytes": len(manifest_bytes),
        "manifest_sha256": digest(manifest_bytes),
        "python": platform.python_version(),
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "project_code_executed": False,
        "formal_data_or_models_accessed": False,
        "checks": checks,
        "failed_checks": failed,
        "payload_failures": errors,
        "payloads": payload_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "payloads"}, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
