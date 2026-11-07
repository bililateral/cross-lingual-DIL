"""Aggregate sensitivity checks; no datasets, split assignments or account IDs exported."""
import argparse
import base64
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
import zipfile

import audit as base

WORDS = re.compile(r"[^\W_]+", re.UNICODE)
ENGLISH_FUNCTION_WORDS = set("the and with for to from of this that is are in on your you our we will can be as it at an or by have has not all please".split())
MASK64 = (1 << 64) - 1
RNG = random.Random(1107)
PERMUTATIONS = [(RNG.getrandbits(64) | 1, RNG.getrandbits(64)) for _ in range(8)]
QUARTERS = ("2014-Q3", "2014-Q4", "2015-Q1")
ROOTS = ("Drugs", "Fraud Related", "Guides &amp; Tutorials")


def quarter(record):
    return f"{record[3][:4]}-Q{(int(record[3][5:7])-1)//3+1}"


def body_signature(body):
    return tuple(re.sub(r"\d+", "#", w) for w in WORDS.findall(body))


def shingles(tokens):
    windows = (tokens[i:i+5] for i in range(max(1, len(tokens)-4)))
    return {int.from_bytes(hashlib.blake2b(" ".join(w).encode(), digest_size=8).digest(), "big") for w in windows}


def fingerprint(grams):
    return tuple(min((a*x+b) & MASK64 for x in grams) for a, b in PERMUTATIONS)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def disjoint_capacity(rows, domain_fn, domain_names, threshold):
    """Hall condition for equal per-domain quotas; emits counts, not an assignment."""
    counts = {d: Counter(r[0] for r in rows if domain_fn(r) == d) for d in domain_names}
    eligible = {d: {v for v, n in c.items() if n >= threshold} for d, c in counts.items()}
    constraints = []
    for mask in range(1, 1 << len(domain_names)):
        ds = [d for i, d in enumerate(domain_names) if mask & (1 << i)]
        union = set().union(*(eligible[d] for d in ds))
        constraints.append({"domains": ds, "union_accounts": len(union), "equal_quota_upper_bound": len(union)//len(ds)})
    return {"minimum_records_per_account_in_domain": threshold,
            "eligible_accounts": {d: len(v) for d, v in eligible.items()},
            "maximum_equal_disjoint_vid_quota": min(c["equal_quota_upper_bound"] for c in constraints),
            "hall_constraints": constraints}


def domain_summary(rows):
    qmix = {q: Counter(r[4] for r in rows if quarter(r) == q) for q in QUARTERS}
    shift = []
    for a, b in zip(QUARTERS, QUARTERS[1:]):
        ca, cb = qmix[a], qmix[b]
        na, nb = sum(ca.values()), sum(cb.values())
        tv = .5*sum(abs(ca[c]/na-cb[c]/nb) for c in set(ca)|set(cb)) if na and nb else None
        shift.append({"a": a, "b": b, "root_category_total_variation": tv})
    return {"quarters": {q: base.capacity([r for r in rows if quarter(r) == q]) for q in QUARTERS},
            "root_categories": {c: base.capacity([r for r in rows if r[4] == c]) for c in ROOTS},
            "quarter_disjoint_capacity": [disjoint_capacity(rows, quarter, QUARTERS, k) for k in (4, 8, 20)],
            "root_disjoint_capacity": [disjoint_capacity(rows, lambda r: r[4], ROOTS, k) for k in (4, 8, 20)],
            "quarter_root_category_counts": {k: dict(v) for k, v in qmix.items()}, "adjacent_quarter_shift": shift}


def public_key_armor_shape(field):
    m = re.search(r"-----BEGIN PGP PUBLIC KEY BLOCK-----(.*?)-----END PGP PUBLIC KEY BLOCK-----", field, re.S)
    if not m:
        return False
    lines = [s.strip() for s in m[1].splitlines() if s.strip() and not s.strip().startswith("=") and ":" not in s]
    try:
        return len(base64.b64decode("".join(lines), validate=True)) >= 100
    except (ValueError, base64.binascii.Error):
        return False


def fixture():
    assert body_signature("item 100mg, the offer!") == body_signature("item 200mg the offer")
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1
    assert jaccard({1, 2}, {3, 4}) == 0
    rows = [("x", 0, 0, "2014-07-01", "A"), ("x", 0, 0, "2014-10-01", "B"),
            ("y", 0, 0, "2015-01-01", "C")]
    assert disjoint_capacity(rows, quarter, QUARTERS, 1)["maximum_equal_disjoint_vid_quota"] == 0
    rows += [("z", 0, 0, "2014-07-01", "A")]
    assert disjoint_capacity(rows, quarter, QUARTERS, 1)["maximum_equal_disjoint_vid_quota"] == 1
    assert not public_key_armor_shape("no key")
    return {"passed": True, "checks": ["body normalization", "exact Jaccard", "identity overlap constrains joint quota", "invalid key field"]}


def run(archive):
    texts = {}
    with zipfile.ZipFile(archive) as z:
        receipts = {}
        cats = list(base.read_table(z, "market/categories.tsv", receipts))
        categories = {r["cid"]: r["category"] for r in cats}
        parents = {r["cid"]: r["parent_cid"] for r in cats}
        dates = {r["mscrape_id"]: date(int(r["scrape_year"]), int(r["scrape_month"]), int(r["scrape_day"])).isoformat()
                 for r in base.read_table(z, "market/scrapes.tsv", receipts)}
        names, exact_names, pgps = defaultdict(set), defaultdict(set), defaultdict(set)
        shaped = set()
        for r in base.read_table(z, "market/vendors.tsv", receipts):
            v, n, key = r["vid"], base.canonical(r["username"]), r["pgp_key"]
            if not base.missing(n):
                names[v].add(n)
                exact_names[n].add(v)
            if not base.missing(key):
                h = base.digest("".join(key.split()))
                pgps[h].add(v)
                if h not in shaped and public_key_armor_shape(key):
                    shaped.add(h)
        patterns = {v: re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in sorted(ns, key=len, reverse=True)) + r")(?!\w)", re.I) for v, ns in names.items()}

        def rows_and_texts():
            for row in base.read_table(z, "market/listings.tsv", receipts):
                v, title, desc = row["vid"].strip(), row["title"], row["description"]
                if not base.missing(title) and not base.missing(desc):
                    ct, cd = base.masked(title, patterns.get(v)), base.masked(desc, patterns.get(v))
                    if ct and cd:
                        raw = base.digest(base.canonical(title).casefold(), base.canonical(desc).casefold())
                        texts[(v, raw)] = (ct, cd)
                yield row

        scanned = base.scan(rows_and_texts(), categories, parents, dates, patterns, return_records=True)
    retained = scanned.pop("_records")
    assert len(retained) == 62447 and scanned["quality"]["rows"] == 1426171, "BASE_RECONCILIATION_FAILED"
    print(json.dumps({"phase": "base_reconciled", "retained": len(retained)}), flush=True)
    owners = defaultdict(set)
    tokens_by_text = {}
    for i, ((v, h), (_, body)) in enumerate(texts.items()):
        if i % 1000 == 0:
            base.guard()
        tokens = body_signature(body)
        bh = base.digest(" ".join(tokens))
        owners[bh].add(v)
        tokens_by_text[(v, h)] = (bh, tokens)
    body_distinct, seen = [], set()
    for r in sorted(retained, key=lambda r: (r[3], int(r[2]), r[1])):
        bh, _ = tokens_by_text[(r[0], r[5])]
        if len(owners[bh]) > 1 or (r[0], bh) in seen:
            continue
        seen.add((r[0], bh))
        body_distinct.append(r)
    rich = []
    for r in body_distinct:
        _, tokens = tokens_by_text[(r[0], r[5])]
        alpha = [w for w in tokens if any(c.isalpha() for c in w)]
        ascii_alpha = sum(all(ord(c) < 128 for c in w) for w in alpha)
        if len(alpha) >= 20 and ascii_alpha >= .9*len(alpha) and len(set(alpha) & ENGLISH_FUNCTION_WORDS) >= 3:
            rich.append(r)
    # Greedy within-account near-copy removal, earliest record retained.
    # Eight MinHash candidate indexes, followed by exact 5-word-shingle Jaccard >= .80.
    indexes = defaultdict(set)
    representatives = {}
    kept, comparisons, removed = [], 0, 0
    for i, r in enumerate(rich):
        if i % 100 == 0:
            base.guard()
        _, tokens = tokens_by_text[(r[0], r[5])]
        grams = shingles(tokens)
        fp = fingerprint(grams)
        candidates = set().union(*(indexes[(r[0], j, x)] for j, x in enumerate(fp)))
        duplicate = False
        for c in sorted(candidates):
            comparisons += 1
            if jaccard(grams, representatives[c]) >= .8:
                duplicate = True
                break
        if duplicate:
            removed += 1
            continue
        representatives[i] = grams
        kept.append(r)
        for j, x in enumerate(fp):
            indexes[(r[0], j, x)].add(i)
    counts = Counter(r[0] for r in kept)
    shared = {k: vs for k, vs in pgps.items() if len(vs) > 1}
    supported = {k: {v for v in vs if counts[v] >= 8} for k, vs in shared.items() if k in shaped}
    supported = {k: vs for k, vs in supported.items() if len(vs) >= 2}
    return {"base_reconciled": True, "base_capacity": scanned["conservative_capacity"],
            "body_unique_and_not_shared_capacity": base.capacity(body_distinct),
            "body_quality_screen_capacity": base.capacity(rich), "near_copy_screen_capacity": base.capacity(kept),
            "near_copy_comparisons": comparisons, "within_account_near_copies_removed": removed,
            "domains_after_body_and_near_copy_screen": domain_summary(kept),
            "domains_before_screens": domain_summary(retained),
            "identity_clues": {"literal_case_sensitive_names_with_multiple_vids": sum(len(v)>1 for v in exact_names.values()),
                "shared_pgp_text_fields": len(shared), "shared_fields_with_decodable_public_key_armor": sum(k in shaped for k in shared),
                "shared_shaped_fields_with_two_accounts_each_at_least_eight_screened_records": len(supported),
                "accounts_in_these_fields": len(set().union(*supported.values())) if supported else 0,
                "verified_real_controller_labels": 0},
            "limits": ["Body screen is a transparent sensitivity rule, not a validated language detector or approved dataset protocol.",
                "Near-copy candidate retrieval is approximate; exact Jaccard applies only to retrieved within-account pairs.",
                "Cross-account exact normalized bodies excluded; cross-account near-copy independence not established.",
                "MinHash retains earliest representatives globally; future-aware exclusions are offline inventory only.",
                "Hall quotas use observed vid, not true controllers, and construct no train/dev/test or stage assignment.",
                "PGP armor shape/base64 is not signature or control verification.",
                "Category variation is descriptive, not a measure of model forgetting."]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=int, default=1500)
    a = p.parse_args()
    start = time.monotonic()
    base.DEADLINE = start + a.seconds
    csv.field_size_limit(16 * 1024**2)
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING"}
    try:
        result["self_check"] = fixture()
        result["archive_sha256"] = hashlib.sha256(Path(a.archive).read_bytes()).hexdigest()
        assert result["archive_sha256"] == "5c8a321d51ca2f2dd10afbdc80a44c0c44498954f7c1ca604932b6667a8673bb"
        result["analysis"] = run(a.archive)
        result["status"] = "COMPLETE"
    except BaseException as e:
        result["status"], result["error_type"] = "FAILED", type(e).__name__
        raise SystemExit(1) from None
    finally:
        result["wall_seconds"] = time.monotonic()-start
        result["peak_memory_bytes"] = base.memory_bytes()
        result["source_sha256"] = {n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ("followup.py", "audit.py")}
        Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
