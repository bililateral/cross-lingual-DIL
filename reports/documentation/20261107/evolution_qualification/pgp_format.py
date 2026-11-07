"""Inspect PGP field serialization and aggregate sharing, without claiming controller truth."""
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile
import csv
import audit as base


def decode_armor(s):
    s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    m = re.search(r"-----BEGIN PGP PUBLIC KEY BLOCK-----(.*?)-----END PGP PUBLIC KEY BLOCK-----", s, re.S)
    if not m:
        return None
    # The published TSV may flatten all line breaks to spaces. Locate the first
    # base64 payload token; exclude short header words and the optional CRC token.
    tokens = m[1].split()
    starts = [i for i, t in enumerate(tokens) if re.fullmatch(r"[A-Za-z0-9+/]{40,}={0,2}", t)]
    if not starts:
        return None
    payload = [t for t in tokens[starts[0]:] if not re.fullmatch(r"=[A-Za-z0-9+/]{4}", t)]
    try:
        decoded = base64.b64decode("".join(payload), validate=True)
        return decoded if len(decoded) >= 100 else None
    except ValueError:
        return None


start = time.monotonic()
base.DEADLINE = start + 60
csv.field_size_limit(16 * 1024**2)
result = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING"}
try:
    token = base64.b64encode(bytes(range(128))).decode()
    for separator in ("\n", " ", "\\n", "<br>"):
        fake = separator.join(("-----BEGIN PGP PUBLIC KEY BLOCK-----", "Version: Example", token, "-----END PGP PUBLIC KEY BLOCK-----"))
        assert decode_armor(fake) == bytes(range(128))
    assert decode_armor("no key") is None
    result["self_check"] = {"passed": True, "formats": ["real newlines", "flattened spaces", "literal escapes", "HTML breaks", "absent marker"]}
    fields, profiles, decoded_by_field = {}, Counter(), {}
    owners, sightings, names = defaultdict(set), defaultdict(lambda: defaultdict(set)), defaultdict(set)
    receipts = {}
    with zipfile.ZipFile(r"C:\Users\35734\Downloads\Compressed\data-and-readme.zip") as z:
        for r in base.read_table(z, "market/vendors.tsv", receipts):
            v, field = r["vid"], r["pgp_key"]
            names[v].add(base.canonical(r["username"]).casefold())
            if base.missing(field):
                continue
            h = base.digest("".join(field.split()))
            owners[h].add(v)
            sightings[h][r["mscrape_id"]].add(v)
            if h in fields:
                continue
            fields[h] = True
            profiles["has_public_key_marker"] += "-----BEGIN PGP PUBLIC KEY BLOCK-----" in field
            profiles["contains_real_newline"] += "\n" in field
            profiles["contains_literal_newline_escape"] += "\\n" in field
            profiles["contains_html_break"] += bool(re.search(r"<br\s*/?>", field, re.I))
            decoded = decode_armor(field)
            if decoded is not None:
                decoded_by_field[h] = hashlib.sha256(decoded).digest()
    shared = {h: vs for h, vs in owners.items() if len(vs) > 1}
    valid_shared = {h: vs for h, vs in shared.items() if h in decoded_by_field}
    result["counts"] = {"vendor_rows": receipts["market/vendors.tsv"]["rows"], "distinct_fields": len(fields),
        "field_format_counts": dict(profiles), "fields_with_decodable_armor_payload": len(decoded_by_field),
        "shared_fields": len(shared), "shared_fields_with_decodable_armor_payload": len(valid_shared),
        "accounts_sharing_decodable_fields": len(set().union(*valid_shared.values())) if valid_shared else 0,
        "shared_decodable_fields_seen_on_multiple_vids_in_same_scrape": sum(any(len(vs)>1 for vs in sightings[h].values()) for h in valid_shared),
        "shared_decodable_fields_covering_different_casefold_names": sum(len(set().union(*(names[v] for v in vs)))>1 for vs in valid_shared.values())}
    result["correction"] = "Earlier followup.py armor check discarded flattened Version/header and payload together. Its zero decodable-field count is withdrawn; text/capacity results are independent."
    result["limits"] = ["Base64 decoding does not verify OpenPGP packets, signatures or control.", "No positive/negative controller labels created."]
    result["status"] = "COMPLETE"
except BaseException as e:
    result["status"], result["error_type"] = "FAILED", type(e).__name__
    raise SystemExit(1) from None
finally:
    result["wall_seconds"] = time.monotonic()-start
    result["peak_memory_bytes"] = base.memory_bytes()
    result["source_sha256"] = {n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ("pgp_format.py", "audit.py")}
    Path("pgp_format_windows.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"]}), flush=True)
