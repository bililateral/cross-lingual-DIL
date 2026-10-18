"""Package this static review and verify its byte-level manifest only.

Uses Python standard library. No project import, Optuna, models, new data,
research computation, network connection, installation, or resampling.
The original input ZIP is copied unchanged. This file is included in output.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "upload" / "optuna_plan_review.zip"
OUT = ROOT.parent / "optuna_review_output"
TREE = OUT / "return_package"
EXPECTED = {
    "size_bytes": 186342,
    "member_count": 26,
    "sha256": "60a248b5599f96e3a944a530fa6818e31d8bfc90133e74a0d7634fd0b9fd81cd",
}
REVIEW_FILES = [
    "report.zh.txt", "findings.json", "reading_scope.json", "evidence_index.tsv",
    "sources.web.json", "submission_supplement.zh.txt",
    "execution/commands.json", "execution/failures.json",
    "mechanical/input_identity.json", "mechanical/manifest_check.json",
    "mechanical/budget_arithmetic.json", "mechanical/source_locations.json",
    "mechanical/saved_field_extract.json", "mechanical/targeted_fields.json",
    "scripts/inspect_input.py", "scripts/mechanical_checks.py",
    "scripts/targeted_fields.py", "scripts/package_review.py",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tsv(rows):
    return "member\tsize_bytes\tsha256\n" + "".join(
        f"{r['path']}\t{r['size_bytes']}\t{r['sha256']}\n" for r in rows
    )


def row(path, data):
    return {"path": path, "size_bytes": len(data), "sha256": sha(data)}


def main():
    original = SOURCE.read_bytes()
    parsed_original_json = []
    with zipfile.ZipFile(io.BytesIO(original)) as z:
        original_names = z.namelist()
        actual_input = {"size_bytes": len(original), "member_count": len(original_names), "sha256": sha(original)}
        if actual_input != EXPECTED or len(original_names) != len(set(original_names)):
            raise ValueError("Original attachment identity mismatch")
        for name in original_names:
            if name.endswith(".json"):
                json.loads(z.read(name).decode("utf-8"))
                parsed_original_json.append(name)

    findings = json.loads((ROOT / "findings.json").read_text(encoding="utf-8"))
    scope = json.loads((ROOT / "reading_scope.json").read_text(encoding="utf-8"))
    if [f["id"] for f in findings["findings"]] != [f"F{i:02d}" for i in range(1, 9)]:
        raise ValueError("Structured finding identifiers differ from report contract")
    if {r["path"] for r in scope["members"]} != set(original_names):
        raise ValueError("Reading scope does not cover the actual input member set")
    if any(not r.get("semantic_scope") for r in scope["members"]):
        raise ValueError("Missing actual semantic reading scope")
    lines = (ROOT / "evidence_index.tsv").read_text(encoding="utf-8").splitlines()
    if any(len(line.split("\t")) != 4 for line in lines):
        raise ValueError("Evidence index is not a four-column TSV")

    TREE.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for rel in REVIEW_FILES:
        data = (ROOT / rel).read_bytes()
        if rel.endswith(".json"):
            json.loads(data.decode("utf-8"))
        payloads["review/" + rel] = data
    payloads["README.zh.txt"] = (ROOT / "README.zh.txt").read_bytes()
    payloads["upload/optuna_plan_review.zip"] = original
    payload_rows = [row(name, payloads[name]) for name in sorted(payloads)]
    manifest = {
        "schema_version": "1.0",
        "purpose": "静态外审交付文件身份，不是科学有效性或执行资格证明",
        "input_identity": EXPECTED,
        "manifest_scope": "全部非清单载荷；排除manifest.json和members.tsv两份清单自身，避免自引用",
        "payload_count": len(payload_rows),
        "files": payload_rows,
    }
    members = dict(payloads)
    members["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    members["members.tsv"] = tsv(payload_rows).encode("utf-8")
    existing = {p.relative_to(TREE).as_posix() for p in TREE.rglob("*") if p.is_file()}
    if existing - set(members):
        raise ValueError("Output tree contains unexpected files; no deletion performed")
    for name, data in members.items():
        target = TREE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    zip_path = OUT / "optuna_plan_external_review.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 18, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, members[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    with zipfile.ZipFile(zip_path) as z:
        if z.namelist() != sorted(members):
            raise ValueError("Return ZIP member list mismatch")
        actual_rows = []
        for name in z.namelist():
            data = z.read(name)
            if data != members[name]:
                raise ValueError("Return ZIP byte mismatch: " + name)
            actual_rows.append(row(name, data))
        if z.read("upload/optuna_plan_review.zip") != original:
            raise ValueError("Frozen input was altered")
        stored = json.loads(z.read("manifest.json").decode("utf-8"))
        if stored["files"] != payload_rows or z.read("members.tsv").decode("utf-8") != tsv(payload_rows):
            raise ValueError("Internal manifest mismatch")

    listing_path = OUT / "optuna_plan_external_review.members.tsv"
    listing_path.write_text(tsv(actual_rows), encoding="utf-8")
    report_copy = OUT / "optuna_plan_external_review.zh.txt"
    finding_copy = OUT / "optuna_plan_external_review.findings.json"
    shutil.copyfile(ROOT / "report.zh.txt", report_copy)
    shutil.copyfile(ROOT / "findings.json", finding_copy)
    zipped = zip_path.read_bytes()
    identity = {
        "schema_version": "1.0",
        "return_zip": {"filename": zip_path.name, "size_bytes": len(zipped), "member_count": len(actual_rows), "sha256": sha(zipped)},
        "external_member_list": {"filename": listing_path.name, "sha256": sha(listing_path.read_bytes()), "scope": "返回ZIP全部实际成员，包括两份包内清单自身；大小为未压缩成员字节数"},
        "input_identity": EXPECTED,
        "frozen_input_copy_byte_identical": True,
        "manifest_payload_count": len(payload_rows),
        "verification": {"all_zip_members_byte_equal_to_packaged_payloads": True, "internal_manifest_matches": True, "input_json_parse_completed": parsed_original_json, "structured_findings_and_reading_scope_checked": True},
        "standalone_copies": [row(report_copy.name, report_copy.read_bytes()), row(finding_copy.name, finding_copy.read_bytes())],
        "packaging_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "packager_process_result": "由调用方在实际命令返回后补记；不事先虚构退出码",
        "meaning": "仅字节身份、JSON语法、记录结构和打包对应；不证明研究科学有效性或准入。",
    }
    identity_path = OUT / "optuna_plan_external_review.identity.json"
    write_json(identity_path, identity)
    print(json.dumps({"return_zip": identity["return_zip"], "manifest_payload_count": len(payload_rows), "member_list": str(listing_path), "identity_path": str(identity_path), "report_path": str(report_copy), "findings_path": str(finding_copy)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
