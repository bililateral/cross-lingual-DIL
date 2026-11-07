"""Bounded, aggregate-only qualification of four Evolution market tables.

No models, split construction, relation labels or forum payload reads.
Windows qualification was explicitly authorized by the user on 2026-11-07.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import time
import unicodedata
import zipfile

try:
    import resource
except ImportError:
    resource = None

DEADLINE = float("inf")


def memory_bytes():
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ValueError("MEMORY_QUERY_FAILED")
    return counters.PeakWorkingSetSize


def guard():
    if time.monotonic() >= DEADLINE:
        raise TimeoutError("WALL_BUDGET_EXCEEDED")
    if memory_bytes() > 8 * 1024**3:
        raise ValueError("MEMORY_BUDGET_EXCEEDED")

ALLOWED = ("README.md", "market/categories.tsv", "market/scrapes.tsv",
           "market/vendors.tsv", "market/listings.tsv")
MD5 = "e144a1a0bdb9a8bc6bf831a84e0b501e"
THRESHOLDS = (4, 8, 20)
PGP = re.compile(r"-----BEGIN PGP [^-]+-----.*?-----END PGP [^-]+-----", re.S)
CONTACT = re.compile(r"https?://\S+|\b[^\s@]+@[^\s@]+\.[^\s@]+", re.I)


def canonical(s):
    return " ".join(unicodedata.normalize("NFKC", s).split()).strip()


def missing(s):
    return canonical(s).casefold() in ("", "nan", "none", "null")


def digest(*parts):
    return hashlib.sha256("\0".join(parts).encode("utf-8")).digest()


def masked(s, pattern):
    s = PGP.sub(" ", s)
    s = CONTACT.sub(" ", s)
    if pattern:
        s = pattern.sub(" ", s)
    return canonical(s).casefold()


def capacity(records):
    counts = Counter(r[0] for r in records)
    distribution = Counter(counts.values())
    return {"records": len(records), "accounts": len(counts),
            "accounts_at_least_records": {str(n): sum(v >= n for v in counts.values()) for n in THRESHOLDS},
            "account_record_histogram": dict(sorted(distribution.items()))}


def layers(records):
    by_cat, by_leaf, by_month, by_quarter = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        by_cat[r[4]].append(r)
        by_leaf[r[11]].append(r)
        month = r[3][:7]
        by_month[month].append(r)
        by_quarter[f"{r[3][:4]}-Q{(int(r[3][5:7])-1)//3+1}"].append(r)
    cats = {k: capacity(v) for k, v in sorted(by_cat.items())}
    months = {k: capacity(v) for k, v in sorted(by_month.items())}
    quarters = {k: capacity(v) for k, v in sorted(by_quarter.items())}
    leaves = {k: capacity(v) for k, v in sorted(by_leaf.items())}
    sets = {k: {r[0] for r in v} for k, v in by_cat.items()}
    cat_pairs = []
    keys = sorted(sets)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            cat_pairs.append({"a": a, "b": b, "shared_accounts": len(sets[a] & sets[b])})
    exclusively_one = Counter()
    memberships = defaultdict(set)
    for c, ids in sets.items():
        for identity in ids:
            memberships[identity].add(c)
    for identity, cs in memberships.items():
        if len(cs) == 1:
            exclusively_one[next(iter(cs))] += 1
    exclusive_capacity = {}
    for c, rows in by_cat.items():
        exclusive_capacity[c] = capacity([r for r in rows if len(memberships[r[0]]) == 1])
    quarter_sets = {k: {r[0] for r in v} for k, v in by_quarter.items()}
    quarter_pairs = []
    qkeys = sorted(quarter_sets)
    for i, a in enumerate(qkeys):
        for b in qkeys[i+1:]:
            quarter_pairs.append({"a": a, "b": b, "shared_accounts": len(quarter_sets[a] & quarter_sets[b])})
    return {"categories": cats, "leaf_categories": leaves, "first_complete_month": months,
            "calendar_quarter_of_first_complete_record": quarters,
            "category_account_intersections": cat_pairs,
            "quarter_account_intersections": quarter_pairs,
            "accounts_occurring_in_one_category_only": dict(exclusively_one),
            "exclusive_root_category_capacity": exclusive_capacity}


def scan(rows, categories, parents, dates, patterns, return_records=False):
    quality = Counter()
    lid_owners = defaultdict(set)
    raw_owners, clean_owners = defaultdict(set), defaultdict(set)
    slots = {}
    ambiguous_keys = set()
    earliest = {}
    complete_texts = {}
    observed_accounts = set()
    category_seen = defaultdict(set)
    per_month = defaultdict(Counter)

    def category_root(cid):
        seen = set()
        while cid in parents and parents[cid] and parents[cid] in categories:
            if cid in seen:
                raise ValueError("CATEGORY_CYCLE")
            seen.add(cid)
            cid = parents[cid]
        return categories.get(cid, "UNKNOWN_CATEGORY")

    for n, row in enumerate(rows, 1):
        if n % 2000 == 1:
            guard()
        quality["rows"] += 1
        vid, lid, sid, cid = (row.get(x, "").strip() for x in ("vid", "lid", "mscrape_id", "cid"))
        title, description = row.get("title", ""), row.get("description", "")
        if not vid or not lid or sid not in dates:
            quality["invalid_identifier_or_scrape"] += 1
            continue
        observed_accounts.add(vid)
        lid_owners[lid].add(vid)
        when = dates[sid]
        month = when[:7]
        per_month[month]["rows"] += 1
        raw_title = canonical(title).casefold()
        raw_description = canonical(description).casefold()
        slot = (sid, vid, lid)
        title_id = digest(raw_title)
        if slot in slots and slots[slot] != title_id:
            ambiguous_keys.add((vid, lid))
        else:
            slots[slot] = title_id
        empty_title, empty_desc = missing(title), missing(description)
        quality["missing_title_rows"] += empty_title
        quality["missing_description_rows"] += empty_desc
        per_month[month]["missing_description_rows"] += empty_desc
        if empty_title or empty_desc:
            continue
        quality["complete_rows"] += 1
        per_month[month]["complete_rows"] += 1
        raw_id = digest(raw_title, raw_description)
        clean_title = masked(title, patterns.get(vid))
        clean_desc = masked(description, patterns.get(vid))
        if not clean_title or not clean_desc:
            quality["empty_after_identifier_masking_rows"] += 1
            continue
        clean_id = digest(clean_title, clean_desc)
        raw_owners[raw_id].add(vid)
        clean_owners[clean_id].add(vid)
        category = category_root(cid)
        category_seen[(vid, lid)].add(category)
        rec = (vid, lid, sid, when, category, raw_id, clean_id,
               len(clean_title), len(clean_desc), len(clean_desc.split()),
               sum(ord(c) > 127 for c in clean_title + clean_desc), categories.get(cid, "UNKNOWN_CATEGORY"))
        old = complete_texts.get((vid, raw_id))
        if old is None or (when, int(sid), lid) < (old[3], int(old[2]), old[1]):
            complete_texts[(vid, raw_id)] = rec
        old = earliest.get((vid, lid))
        if old is None or (when, int(sid), raw_id) < (old[3], int(old[2]), old[5]):
            earliest[(vid, lid)] = rec
        if n % 200000 == 0:
            print(json.dumps({"listing_rows_scanned": n}), flush=True)

    bad_lids = {k for k, v in lid_owners.items() if len(v) > 1}
    unique_raw = list(complete_texts.values())
    strict_before_shared = [r for k, r in earliest.items()
                            if k not in ambiguous_keys and r[1] not in bad_lids]
    strict = {}
    for r in strict_before_shared:
        if len(raw_owners[r[5]]) != 1 or len(clean_owners[r[6]]) != 1:
            continue
        k = (r[0], r[6])
        old = strict.get(k)
        if old is None or (r[3], int(r[2]), r[1]) < (old[3], int(old[2]), old[1]):
            strict[k] = r
    retained = list(strict.values())

    def quantiles(values):
        a = sorted(values)
        return {str(p): a[round((len(a)-1)*p)] if a else None for p in (0, .25, .5, .75, .95, 1)}

    quality.update({"observed_accounts": len(observed_accounts),
                    "distinct_lids": len(lid_owners),
                    "lids_with_multiple_vids": len(bad_lids),
                    "scrape_account_lid_slots": len(slots),
                    "account_lid_keys_with_multiple_titles_within_a_scrape": len(ambiguous_keys),
                    "account_lid_keys_with_complete_text": len(earliest),
                    "account_lid_keys_seen_in_multiple_root_categories": sum(len(x) > 1 for x in category_seen.values()),
                    "raw_text_pairs_shared_across_accounts": sum(len(x) > 1 for x in raw_owners.values()),
                    "masked_text_pairs_shared_across_accounts": sum(len(x) > 1 for x in clean_owners.values()),
                    "strict_records_before_shared_text_exclusion": len(strict_before_shared)})
    result = {"quality": dict(quality), "raw_rows_by_scrape_month": dict(per_month),
            "account_distinct_complete_raw_text_capacity": capacity(unique_raw),
            "conservative_capacity": capacity(retained),
            "conservative_layers": layers(retained),
            "conservative_text_lengths": {"title_chars": quantiles([r[7] for r in retained]),
                "description_chars": quantiles([r[8] for r in retained]),
                "description_whitespace_words": quantiles([r[9] for r in retained]),
                "records_with_non_ascii_chars": sum(r[10] > 0 for r in retained)},
            "limits": ["No language identification: ASCII content is not proof of English.",
                       "Exact text deduplication only; near-duplicate/template independence not established.",
                       "Conservative record accounting, not a certified reconstruction of true products.",
                       "Date means earliest observed complete retained text, not product creation time.",
                       "Account overlap uses observed vid, not verified controllers.",
                       "No actual train/dev/test split or pair labels created.",
                       "No forgetting, performance, or model advantage measured."]}
    if return_records:
        result["_records"] = retained  # In-memory follow-up only; no account records are exported.
    return result


def read_table(z, member, receipts):
    with z.open(member) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), delimiter="\t")
        headers = reader.fieldnames
        receipts[member] = {"headers": headers, "rows": 0, "bytes": z.getinfo(member).file_size}
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("TSV_FIELD_COUNT_MISMATCH:" + member)
            receipts[member]["rows"] += 1
            if receipts[member]["rows"] % 2000 == 1:
                guard()
            yield row


def self_check():
    cats, parents = {"1": "Alpha", "2": "Beta"}, {"1": "", "2": ""}
    dates = {"1": "2014-01-01", "2": "2014-02-01"}
    def row(v, l, s, t, d, c="1"):
        return dict(vid=v, lid=l, mscrape_id=s, title=t, description=d, cid=c)
    rows = [row("v1", "1", "1", "A", "unique A"), row("v1", "1", "2", "A", "unique A"),
            row("v1", "2", "1", "B", "unique B"), row("v1", "2", "1", "Changed B", "unique B"),
            row("v1", "3", "1", "Shared", "same text"), row("v2", "4", "1", "Shared", "same text"),
            row("v2", "5", "1", "Missing", ""), row("v2", "6", "1", "C", "unique C", "2"),
            row("v3", "7", "1", "Alice Pack", "Alice common offer"),
            row("v4", "8", "1", "Bob Pack", "Bob common offer")]
    result = scan(rows, cats, parents, dates,
                  {"v3": re.compile(r"\balice\b", re.I), "v4": re.compile(r"\bbob\b", re.I)})
    assert result["quality"]["rows"] == 10
    assert result["quality"]["complete_rows"] == 9
    assert result["quality"]["masked_text_pairs_shared_across_accounts"] == 2
    assert result["quality"]["account_lid_keys_with_multiple_titles_within_a_scrape"] == 1
    assert result["conservative_capacity"]["records"] == 2
    assert result["conservative_capacity"]["accounts"] == 2
    assert result["conservative_layers"]["first_complete_month"]["2014-01"]["records"] == 2
    assert masked("alice offer alice@example.test https://example.test", re.compile(r"\balice\b", re.I)) == "offer"
    quoted = list(csv.DictReader(io.StringIO('title\tdescription\n"a\tb"\t"first\nsecond"\n'), delimiter="\t"))
    assert quoted == [{"title": "a\tb", "description": "first\nsecond"}]
    return {"passed": True, "fixture_rows": 10,
            "checks": ["repeated scrape not a new product", "within-scrape title conflict excluded",
                       "shared cross-account text excluded", "missing description excluded",
                       "earliest retained month", "identity/contact masking in actual scan", "quoted TSV"]}


def main():
    global DEADLINE
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("self-check", "inspect", "audit"))
    p.add_argument("--archive", default="data-and-readme.zip")
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=int, default=1800)
    a = p.parse_args()
    start = time.monotonic()
    DEADLINE = start + a.seconds
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("WALL_BUDGET_EXCEEDED")))
        signal.alarm(a.seconds)
    csv.field_size_limit(16 * 1024**2)
    result = {"mode": a.mode, "platform": os.name, "started_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING"}
    try:
        if a.mode == "self-check":
            result["self_check"] = self_check()
        else:
            archive = Path(a.archive)
            with archive.open("rb") as f:
                md = hashlib.md5()
                while block := f.read(4 * 1024**2):
                    guard()
                    md.update(block)
            if md.hexdigest() != MD5:
                raise ValueError("ARCHIVE_MD5_MISMATCH")
            result["archive"] = {"bytes": archive.stat().st_size, "md5": md.hexdigest()}
            with zipfile.ZipFile(archive) as z:
                missing_members = [n for n in ALLOWED if n not in z.namelist()]
                if missing_members:
                    raise ValueError("EXPECTED_MEMBERS_ABSENT")
                if a.mode == "inspect":
                    result["allowed_members"] = {n: z.getinfo(n).file_size for n in ALLOWED}
                    result["headers"] = {}
                    for member in ALLOWED[1:]:
                        with z.open(member) as f:
                            result["headers"][member] = io.TextIOWrapper(f, encoding="utf-8-sig").readline().strip()
                    Path("dataset_README.md").write_bytes(z.read("README.md"))
                else:
                    receipts = {}
                    cats = list(read_table(z, "market/categories.tsv", receipts))
                    categories = {r["cid"]: r["category"] for r in cats}
                    parents = {r["cid"]: r["parent_cid"] for r in cats}
                    dates = {r["mscrape_id"]: date(int(r["scrape_year"]), int(r["scrape_month"]), int(r["scrape_day"])).isoformat()
                             for r in read_table(z, "market/scrapes.tsv", receipts)}
                    names, name_accounts, pgp_accounts = defaultdict(set), defaultdict(set), defaultdict(set)
                    pgp_vids, seen_vids = set(), set()
                    pgp_nonempty = 0
                    for r in read_table(z, "market/vendors.tsv", receipts):
                        vid = r["vid"]
                        seen_vids.add(vid)
                        name = canonical(r["username"])
                        if not missing(name):
                            names[vid].add(name)
                            name_accounts[name.casefold()].add(vid)
                        key = r["pgp_key"]
                        if not missing(key):
                            pgp_nonempty += 1
                            pgp_vids.add(vid)
                            pgp_accounts[digest("".join(key.split()))].add(vid)
                    patterns = {v: re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in sorted(ns, key=len, reverse=True)) + r")(?!\w)", re.I)
                                for v, ns in names.items()}
                    result["observed_identity_signals"] = {
                        "vendor_rows": receipts["market/vendors.tsv"]["rows"], "distinct_vids": len(seen_vids),
                        "distinct_casefold_names": len(name_accounts),
                        "names_with_multiple_vids": sum(len(v) > 1 for v in name_accounts.values()),
                        "vids_with_multiple_names": sum(len(v) > 1 for v in names.values()),
                        "nonempty_pgp_field_rows": pgp_nonempty, "vids_with_nonempty_pgp_field": len(pgp_vids),
                        "distinct_whitespace_normalized_pgp_fields": len(pgp_accounts),
                        "pgp_fields_shared_by_multiple_vids": sum(len(v) > 1 for v in pgp_accounts.values()),
                        "pgp_shared_account_count": len(set().union(*(v for v in pgp_accounts.values() if len(v) > 1))),
                        "note": "Text-field equality only: no cryptographic validation or controller labels."}
                    result["inventory"] = scan(read_table(z, "market/listings.tsv", receipts), categories, parents, dates, patterns)
                    result["tables"] = receipts
                    result["category_dictionary"] = {r["cid"]: {"category": r["category"], "parent_cid": r["parent_cid"]} for r in cats}
                    result["scrapes"] = {"count": len(dates), "first": min(dates.values()), "last": max(dates.values())}
        result["status"] = "COMPLETE"
    except BaseException as e:
        # Never put row values, account names, keys or trace locals in output.
        result["status"] = "FAILED"
        result["error_type"] = type(e).__name__
        approved_errors = ("CATEGORY_CYCLE", "TSV_FIELD_COUNT_MISMATCH:",
                           "ARCHIVE_MD5_MISMATCH", "EXPECTED_MEMBERS_ABSENT", "WALL_BUDGET_EXCEEDED",
                           "MEMORY_QUERY_FAILED", "MEMORY_BUDGET_EXCEEDED")
        result["error_code"] = str(e) if str(e).startswith(approved_errors) else "NON_VALUE_EXCEPTION"
        raise SystemExit(1) from None
    finally:
        result["wall_seconds"] = time.monotonic() - start
        result["peak_memory_bytes"] = memory_bytes()
        result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "mode": a.mode, "wall_seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
