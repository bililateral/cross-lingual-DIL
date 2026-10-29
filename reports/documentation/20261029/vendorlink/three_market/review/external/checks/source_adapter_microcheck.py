"""Isolated reviewer checks: handwritten archives only; no model/data downloads.

Usage: python3 -B source_adapter_microcheck.py INPUT_ROOT OUTPUT_JSON
Imports only the supplied stdlib data/preparation modules; never calls main().
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
from pathlib import Path
import platform
import sys
import tarfile
import tempfile
import zipfile

sys.dont_write_bytecode = True


def csv_bytes(header, rows, encoding):
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows([header, *rows])
    return stream.getvalue().encode(encoding)


def main():
    input_root, output = map(Path, sys.argv[1:])
    sys.path.insert(0, str(input_root / "scripts"))
    prepare = importlib.import_module("step29_vendorlink_prepare")
    header = ["hash", "market_name", "item_link", "vendor_name", "price", "name", "description", "image_link", "add_time", "ship_from", ""]

    def row(link, owner, title, description, market="AlphaBay"):
        return ["h", market, link, owner, "0", title, description, "", "", "", ""]

    snapshots = [
        ("root/2016-01-03/Alpha.csv", [
            row("p1", "alice", "later title", "later description"),
            row("p2", "charlie", "changed owner", "changed description"),
            row("p3", "dora", "complete third", "complete description"),
            row("p4", "ed", "bad \ufffd title", "bad text"),
        ]),
        ("root/2016-01-02/Alpha.csv", [
            row("p1", "alice", "  first  café title ", "first description"),
            row("p2", "bob", "first owner", "first owner description"),
            row("p3", "dora", "", "missing title"),
            row("wrong", "other", "wrong market", "wrong text", market="Dream"),
            row("extra", "other", "extra title", "extra text") + ["extra column"],
            ["h", "AlphaBay", "short", "other", "0"],
        ]),
        ("root/2016-01-01/Alpha.csv", [row("p1", "alice", "too early incomplete", "")]),
        ("root/2016-01-01/Dream.csv", [row("dream1", "dream_vendor", "dream title", "dream text", market="Dream")]),
    ]
    agora_header = ["Vendor", " Item", " Item Description"]
    agora_rows = [["00123", "café title", "déjà vu description"],
                  ["17.0", "  other  title ", "other description"],
                  ["empty", "empty title", ""],
                  ["bad", "bad title", "bad description", "extra"]]
    with tempfile.TemporaryDirectory(prefix="vendorlink_source_review_") as tmp:
        base = Path(tmp)
        agora = base / "handwritten_agora.zip"
        with zipfile.ZipFile(agora, "w") as archive:
            archive.writestr("Agora.csv", csv_bytes(agora_header, agora_rows, "ISO-8859-1"))
        results = []
        for i, ordering in enumerate((snapshots, snapshots[::-1])):
            grams = base / f"handwritten_grams_{i}.tar.xz"
            with tarfile.open(grams, "w:xz") as archive:
                for name, rows in ordering:
                    payload = csv_bytes(header, rows, "utf-8-sig")
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
            results.append(prepare.read_sources(grams, agora))
        assert results[0] == results[1]
        rows, report = results[0]
        assert len(rows) == 5
        by_owner = {r["vendor"]: r for r in rows}
        assert by_owner["alice"]["title"] == "first café title"
        assert by_owner["alice"]["source_date"] == "2016-01-02"
        assert by_owner["dora"]["source_date"] == "2016-01-03"
        assert "bob" not in by_owner and "charlie" not in by_owner
        assert by_owner["00123"]["description"] == "déjà vu description"
        assert by_owner["17.0"]["vendor"] == "17.0"
        assert by_owner["17.0"]["title"] == "other title"
        expected_alpha = {"snapshot_files": 3, "parsed_rows": 11, "valid_rows": 5,
                          "repeated_listing_rows": 2, "replacement_character_rows": 1,
                          "missing_required_rows": 2, "wrong_market_rows": 1,
                          "malformed_rows": 2, "distinct_listing_keys": 3,
                          "ambiguous_account_listings_removed": 1, "canonical_rows": 2}
        assert dict(report["markets"]["AlphaBay"]) == expected_alpha
        assert report["markets"]["Agora"] == {"parsed_rows": 4, "valid_rows": 2,
                                                "missing_required_rows": 1, "malformed_rows": 1,
                                                "canonical_rows": 2}
        for counts in report["markets"].values():
            failures = sum(counts.get(k, 0) for k in ("malformed_rows", "wrong_market_rows", "missing_required_rows", "replacement_character_rows"))
            assert counts["parsed_rows"] == counts["valid_rows"] + failures
        # Exact-schema failures are fail-closed on the supplied function, not caller mocks.
        wrong_zip = base / "wrong_member.zip"
        with zipfile.ZipFile(wrong_zip, "w") as archive:
            archive.writestr("different.csv", b"a,b\n1,2\n")
        try:
            prepare.read_sources(grams, wrong_zip)
        except ValueError as exc:
            assert "pinned single-file" in str(exc)
        else:
            raise AssertionError("Unexpected Agora ZIP member must be rejected")
    inventory = json.loads((input_root / "reports/documentation/20261029/vendorlink/three_market/inventory.json").read_text())
    config_path = input_root / "reports/documentation/20261029/vendorlink/three_market/source_config.json"
    config = json.loads(config_path.read_text())
    assert inventory["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert inventory["sources"] == config["sources"]
    markets = inventory["source_report"]["markets"]
    assert sum(x["canonical_rows"] for x in markets.values()) == inventory["source_rows"] == 236457
    for name, counts in markets.items():
        rejected = sum(counts.get(k, 0) for k in ("malformed_rows", "wrong_market_rows", "missing_required_rows", "replacement_character_rows"))
        assert counts["parsed_rows"] == counts["valid_rows"] + rejected
        if name != "Agora":
            assert counts["valid_rows"] == counts["repeated_listing_rows"] + counts["distinct_listing_keys"]
            assert counts["canonical_rows"] == counts["distinct_listing_keys"] - counts.get("ambiguous_account_listings_removed", 0)
        else:
            assert counts["canonical_rows"] == counts["valid_rows"]
    evidence = {
        "scope": "Reviewer isolated environment; handwritten CSV/tar.xz/ZIP only. No original archives, cache, Torch, model, training or server access.",
        "environment": {"python": sys.version, "platform": platform.platform()},
        "source_sha256": hashlib.sha256((input_root / "scripts/step29_vendorlink_prepare.py").read_bytes()).hexdigest(),
        "checks": {"archive_order_invariance": True, "earliest_complete_not_earliest_incomplete": True,
                   "ambiguous_listing_excluded": True, "actual_full_grams_header_with_trailing_blank": True,
                   "utf8_bom_and_latin1_accents": True, "numeric_handles_preserve_string": True,
                   "wrong_market_extra_and_missing_columns_rejected": True,
                   "replacement_char_text_rejected": True, "unexpected_agora_member_rejected": True,
                   "provided_inventory_source_arithmetic": True, "provided_inventory_config_identity": True},
        "handwritten_report": report,
        "original_archive_identity_independently_rehashed": False,
        "qualifications": ["Canonical text has intentional NFKC/whitespace normalization; preservation means string type/meaningful leading zeros, not original bytes.",
                           "This checker tests pinned-schema behavior, not an exhaustive CSV parser or full production corpus."],
    }
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "checks": len(evidence["checks"]), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
