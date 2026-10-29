"""Observed-account view benchmark; never interprets a handle as a real owner.

Preparation is a separately authorized operation. No files are read on import.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
import re
import unicodedata


def seed_for(*parts) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).digest()[:8], "big") % (2**31)


@dataclass(frozen=True)
class Ad:
    uid: str
    title: str
    description: str


@dataclass(frozen=True)
class View:
    uid: str
    records: tuple[Ad, ...]
    identity: str | None = None


@dataclass(frozen=True)
class Group:
    uid: str
    domain: str
    split: str
    views: tuple[View, ...]

    def validate(self, training: bool = False) -> None:
        if len(self.views) < 2 or len({v.uid for v in self.views}) != len(self.views):
            raise ValueError("Distinct account views required")
        records = [r.uid for v in self.views for r in v.records]
        if not all(len(v.records) >= 2 for v in self.views) or len(records) != len(set(records)):
            raise ValueError("Views require disjoint sets of at least two advertisements")
        if training and (self.split != "fit" or any(v.identity is None for v in self.views)):
            raise ValueError("Only authorized fit identities can train/regroup")
        if training:
            for view in self.views:
                positives = sum(v.identity == view.identity for v in self.views) - 1
                if positives < 1 or len(self.views) - positives - 1 < 5:
                    raise ValueError("Each training query needs a positive and five negatives")

    def labels(self) -> list[int]:
        if any(v.identity is None for v in self.views):
            raise ValueError("Blind views do not contain relation labels")
        return [int(a.identity == b.identity) for i, a in enumerate(self.views) for b in self.views[i+1:]]

    def blind(self) -> Group:
        return Group(self.uid, self.domain, self.split, tuple(View(v.uid, v.records) for v in self.views))

    @classmethod
    def from_dict(cls, value: dict) -> Group:
        result = cls(value["uid"], value["domain"], value["split"], tuple(
            View(v["uid"], tuple(Ad(**r) for r in v["records"]), v.get("identity")) for v in value["views"]))
        result.validate(training=result.split == "fit")
        return result


def canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def build_views(rows: list[dict], *, split_seed: int, train_fraction: float,
                dev_fraction: float, records_per_view: int, identities_per_group: int) -> tuple[dict, dict, dict]:
    """Canonical rows: market/vendor/title/description. No model-based selection.

    Equal lowercased handles share a split across markets, NOT a relation label.
    Exact text shared across observed accounts is excluded, not labeled positive.
    Near duplicates and true owner separation are not claimed.
    """
    if not (0 < train_fraction < 1 and 0 < dev_fraction < 1-train_fraction):
        raise ValueError("Three nonempty split fractions required")
    if records_per_view < 2 or identities_per_group < 4:
        raise ValueError("Two records/view and at least four identities/group required")
    accounts, owners, ads, raw_owners, raw_fingerprints = {}, {}, {}, {}, {}
    for row in rows:
        if any(not isinstance(row.get(k), str) for k in ("market", "vendor", "title", "description")):
            raise ValueError("Explicit canonical string fields required; no silent NaN conversion")
        market, vendor = canonical(row["market"]), canonical(row["vendor"])
        title, description = canonical(row["title"]), canonical(row["description"])
        if not market or not vendor or not title or not description:
            raise ValueError("Empty market/vendor/title/description")
        key = (market, vendor)  # Case variants are NOT asserted to be the same observed account.
        raw_fingerprint = hashlib.sha256((title+"\0"+description).casefold().encode()).hexdigest()
        raw_owners.setdefault(raw_fingerprint, set()).add(key)
        # Remove this account's direct identifier consistently for all models.
        pattern = re.compile(r"(?<!\w)"+re.escape(vendor)+r"(?!\w)", re.IGNORECASE)
        title, description = pattern.sub("[VENDOR]", title), pattern.sub("[VENDOR]", description)
        fingerprint = hashlib.sha256((title+"\0"+description).casefold().encode()).hexdigest()
        owners.setdefault(fingerprint, set()).add(key)
        accounts.setdefault(key, set()).add(fingerprint)
        ads[(key, fingerprint)] = Ad(fingerprint, title, description)
        raw_fingerprints.setdefault((key, fingerprint), set()).add(raw_fingerprint)
    buckets, rejected = {}, {"shared_exact_text": 0, "too_few_ads": 0, "insufficient_group": 0}
    for key in sorted(accounts):
        market, vendor = key
        clean = [f for f in accounts[key] if len(owners[f]) == 1 and all(
            len(raw_owners[raw]) == 1 for raw in raw_fingerprints[(key, f)])]
        rejected["shared_exact_text"] += len(accounts[key])-len(clean)
        # Paper's 20-ad eligibility, plus the requested disjoint view capacity.
        if len(clean) < max(20, 2*records_per_view):
            rejected["too_few_ads"] += 1
            continue
        value = seed_for(split_seed, "identity_split", vendor.casefold()) / 2**31
        split = "fit" if value < train_fraction else "dev" if value < train_fraction+dev_fraction else "heldout"
        identity = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()
        selected = sorted(clean, key=lambda f: (seed_for(split_seed, "ads", identity, f), f))[:2*records_per_view]
        views = tuple(View(identity+f":view{k}", tuple(ads[(key, f)] for f in
                      selected[k*records_per_view:(k+1)*records_per_view]), identity) for k in range(2))
        buckets.setdefault((market, split), []).append((identity, views))
    payloads, labels = {}, {}
    for (market, split), identities in sorted(buckets.items()):
        identities.sort(key=lambda x: (seed_for(split_seed, "groups", market, x[0]), x[0]))
        chunks = [identities[i:i+identities_per_group] for i in range(0, len(identities), identities_per_group)]
        if len(chunks[-1]) < 4:
            if len(chunks) == 1:
                rejected["insufficient_group"] += len(chunks[0])
                continue
            chunks[-2].extend(chunks.pop())
        for index, chunk in enumerate(chunks):
            group = Group(f"{market}:{split}:{index}", market, split,
                          tuple(v for _, views in chunk for v in views))
            group.validate(training=split == "fit")
            labels[group.uid] = group.labels()
            # Evaluation inputs carry no identity labels; these remain in the separate qrels output.
            payloads.setdefault(f"{market}:{split}", []).append(asdict(group if split == "fit" else group.blind()))
    return payloads, labels, {"scope": "observed_account_disjoint_ad_views_not_verified_aliases",
        "source_rows": len(rows), "observed_accounts": len(accounts), "rejected": rejected,
        "group_counts": {k: len(v) for k, v in payloads.items()}, "split_key": "casefold_handle_across_markets",
        "near_duplicate_exclusion": False, "latent_owner_disjointness_verified": False}


def assignment(group: Group, regroup_seed: int | None = None) -> list[int]:
    group.validate(training=regroup_seed is not None)
    natural = [i for i, v in enumerate(group.views) for _ in v.records]
    if regroup_seed is None:
        return natural
    result, rng = natural.copy(), random.Random(regroup_seed)
    components = {}
    for i, view in enumerate(group.views):
        components.setdefault(view.identity, []).append(i)
    for component in components.values():
        positions = [i for i, slot in enumerate(natural) if slot in component]
        rng.shuffle(positions)
        start = 0
        for slot in component:
            count = len(group.views[slot].records)
            for position in positions[start:start+count]:
                result[position] = slot
            start += count
    return result
