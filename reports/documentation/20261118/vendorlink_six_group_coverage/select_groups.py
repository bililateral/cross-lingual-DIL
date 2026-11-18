"""Fit-only stage-local selection. This module has no evaluator or score input."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


WORK = Path(__file__).resolve().parent
PROJECT = WORK.parents[3]
MECHANISM = PROJECT / "reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism"
FIT = PROJECT / "reports/documentation/20261109/vendorlink_account_protocol/execution/qualification/data"
SEEDS = (51, 52, 53)
DOMAINS = ("AlphaBay", "Dream", "Agora")
SIZES = (546, 157)
FIT_SHA = ("34b2b568c562cd963a6607ca3b4f059a33a5d2cf52b4bffd463fa0a2e874f817",
           "b4c0e7c552a1abb83e42537b7083e9b270dd63476a5d120ab392f06870b7cd6e")
GATE_SHA = "6ecf760bfa3d7e25be7a5e5e69cf4962211be3225e933389fcb149352592c9cc"
CAP = 1048576
DRAWS = 1000
PRIVATE = WORK / "selection_private"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pinned(path, expected):
    require(path.resolve() == path, "Unexpected linked input")
    data = path.read_bytes()
    require(sha(data) == expected, "Input identity mismatch: " + path.name)
    return data


def put_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def seal(path, value):
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    compressed = gzip.compress(data, compresslevel=5, mtime=0)
    with path.open("xb") as stream:
        stream.write(compressed)
    path.chmod(0o400)
    return dict(name=path.name, bytes=len(compressed), sha256=sha(compressed))


def runtime():
    require(sys.platform == "linux" and Path(sys.executable).resolve() ==
            Path("/home/yongpeng/miniconda3/envs/py310/bin/python").resolve()
            and len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and torch.get_num_threads() == 1, "Use project Linux py310, one CPU core and no GPU")
    os.umask(0o077)


def originals():
    gate = json.loads(pinned(MECHANISM / "gate.json", GATE_SHA))
    directory = MECHANISM / "workspace/scripts"
    for item in gate["sources"]:
        source = Path(item["path"])
        require(source.parent == directory, "Unexpected original source directory")
        pinned(source, item["sha256"])
    sys.path.insert(0, str(directory))
    from step35_vendorlink_account_run import groups_for
    from step29_vendorlink_data import seed_for
    from step29_vendorlink_methods import Memory
    return groups_for, seed_for, Memory


@dataclass(eq=False)
class Slot:
    uid: str
    stage: int
    packet: int
    members: tuple
    row: dict


def make_pool(accounts, seed, stage, groups_for):
    market = DOMAINS[stage - 1]
    ids = [a["account_uid"] for a in accounts]
    require(ids == sorted(set(ids)), "Fit account ordering")
    id_to_index = {v: i for i, v in enumerate(ids)}
    groups = groups_for(accounts, seed, market, 0)
    require(len(groups) % 3 == 0, "Incomplete record packets")
    per_packet = len(groups) // 3
    views = [[None] * 6 for _ in ids]
    pool, catalog = [], {}
    for i, g in enumerate(groups):
        packet = i // per_packet
        members, view_records = [], []
        for k in range(0, len(g.views), 2):
            v0, v1 = g.views[k:k+2]
            require(v0.identity == v1.identity and len(v0.records) == len(v1.records) == 2,
                    "Original paired fit views required")
            account = id_to_index[v0.identity]
            members.append(account)
            for offset, v in enumerate((v0, v1)):
                require(views[account][2*packet+offset] is None, "Repeated fit view")
                views[account][2*packet+offset] = v.records
                view_records.append([v.uid, [r.uid for r in v.records]])
        require(len(set(members)) == len(members) and g.uid not in catalog, "Repeated group/account")
        n_records = 4 * len(members)
        teacher = torch.zeros(n_records*(n_records-1)//2, dtype=torch.float32)
        slot = Slot(g.uid, stage, packet, tuple(members),
                    dict(group=asdict(g), origin=stage, teacher=teacher))
        pool.append(slot)
        catalog[g.uid] = dict(stage=stage, packet=packet, members=members, views=view_records)
    for account in views:
        require(all(v is not None for v in account) and len({r.uid for v in account for r in v}) == 12,
                "Six disjoint two-record fit views required")
    return pool, views, catalog


def cosine_field(references, queries):
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), norm="l2", smooth_idf=True,
                          sublinear_tf=False, dtype=np.float64)
    if not any(vec.build_analyzer()(s) for s in references):
        return np.zeros((len(queries), len(references)))
    ref = vec.fit_transform(references)
    return (vec.transform(queries) @ ref.T).toarray()


def nominate(scores):
    require(scores.ndim == 2 and scores.shape[0] == scores.shape[1] and np.isfinite(scores).all(),
            "Invalid fit similarity")
    edges = []
    for i, row in enumerate(scores):
        order = np.argsort(-row, kind="stable")
        chosen = [int(j) for j in order if j != i and row[j] > 0][:5]
        edges.extend((i, j) for j in chosen)
    return edges


def fit_weights(views):
    counts = Counter()
    for fold in range(6):
        total = np.zeros((len(views), len(views)))
        for field in ("title", "description"):
            refs = [" ".join(getattr(r, field) for k, pair in enumerate(account) if k != fold
                             for r in pair) for account in views]
            queries = [" ".join(getattr(r, field) for r in account[fold]) for account in views]
            total += .5 * cosine_field(refs, queries)
        counts.update(nominate(total))
    return counts


def group_edges(pool, weights):
    return {g.uid: {(a, b): weights[a, b] for a in g.members for b in g.members
                    if (a, b) in weights} for g in pool}


def random_slots(old, ordered, rng, seen):
    result = list(old)
    for group in ordered:
        seen += 1
        slot = len(result) if len(result) < 6 else rng.randrange(seen)
        if slot < 6:
            if slot == len(result):
                result.append(group)
            else:
                result[slot] = group
    require(len(result) == 6, "Reservoir not full")
    return result, seen


def mixed_slots(baseline, carried, stage, choices, edges):
    chosen = [group if group.stage == stage else carried[i] for i, group in enumerate(baseline)]
    for i in (1, 3, 5):
        original = chosen[i]
        if original.stage != stage:
            continue
        masks, occupied, covered = defaultdict(int), set(), set()
        for k, group in enumerate(chosen):
            if k == i or group.stage != stage:
                continue
            occupied.add(group.uid)
            covered.update(edges[group.uid])
            for account in group.members:
                masks[account] |= 1 << k
        # The multiset of five-slot masks on replacement accounts keeps every
        # six-slot incidence count unchanged, including triple overlaps.
        def signature(g):
            return sorted(masks.get(a, 0) for a in g.members)

        def gain(g):
            return sum(w for pair, w in edges[g.uid].items() if pair not in covered)

        required = signature(original)
        best, value = original, gain(original)
        for candidate in choices[original.packet, len(original.members)]:
            if candidate.uid in occupied or signature(candidate) != required:
                continue
            score = gain(candidate)
            if score > value:
                best, value = candidate, score
        chosen[i] = best
    return chosen


def memory_size(slots, seen, rng_state, Memory):
    memory = Memory(0, SimpleNamespace(memory_groups=6, memory_bytes=CAP))
    memory.seen, memory.entries = seen, [g.row for g in slots]
    memory.rng.setstate(rng_state)
    return len(memory.bytes())


def index_receipt(seed, stage, groups, pool, seen, size):
    path = MECHANISM / f"campaign/M/training/M_H_ER_s{seed}/run/memory_stage{stage}.json"
    require(path.resolve() == path, "Memory index path mismatch")
    data = path.read_bytes()
    index = json.loads(data)
    require(index["groups"] == [dict(uid=g.uid, origin=g.stage) for g in groups]
            and index["seen"] == seen and index["candidates"] == len(pool),
            "Original random slots/order/origins/candidate count mismatch")
    require(index["bytes"] == size, "Original serialized byte count mismatch")
    return dict(seed=seed, stage=stage, sha256=sha(data), origins=[g.stage for g in groups],
                seen=seen, candidates=len(pool), bytes=size)


def run_stage(accounts, seed, stage, states, groups_for, seed_for, Memory):
    pool, views, catalog = make_pool(accounts, seed, stage, groups_for)
    edges = group_edges(pool, fit_weights(views))
    ordered = sorted(pool, key=lambda g: g.uid)
    choices = defaultdict(list)
    for g in ordered:
        choices[g.packet, len(g.members)].append(g)
    next_states, snapshots = [], []
    original_index = None
    for rep in range(DRAWS + 1):
        original_seed = seed_for(seed, DOMAINS, "reservoir")
        rng = random.Random(original_seed if rep == 0 else f"six-group-fit/v1/{seed}/{rep}")
        old_random, old_mixed, seen = [], [], 0
        if stage == 2:
            old_random, old_mixed, seen, rng_state = states[rep]
            rng.setstate(rng_state)
        baseline, seen = random_slots(old_random, ordered, rng, seen)
        mixed = mixed_slots(baseline, old_mixed, stage, choices, edges)
        rng_state = rng.getstate()
        sizes = [memory_size(slots, seen, rng_state, Memory) for slots in (baseline, mixed)]
        if rep == 0:
            original_index = index_receipt(seed, stage, baseline, pool, seen, sizes[0])
        snapshots.append(dict(random=[g.uid for g in baseline], mixed=[g.uid for g in mixed],
                              bytes=sizes))
        if stage == 1:
            # Each independent replicate receives only its six retained original
            # groups per arm and sampler state. No corpus or edge table is carried.
            next_states.append((baseline, mixed, seen, rng_state))
    return next_states, snapshots, catalog, original_index


def select():
    start = time.monotonic()
    check = json.loads((WORK / "check.json").read_text())
    source_files = ("select_groups.py", "evaluate.py", "check_coverage.py", "scope.zh.md")
    sources = {f: sha((WORK / f).read_bytes()) for f in source_files}
    require(check["status"] == "PASS" and check["sources"] == sources, "Current synthetic check required")
    groups_for, seed_for, Memory = originals()
    PRIVATE.mkdir(mode=0o700, exist_ok=False)
    states, catalogs, stages, indices = {}, {}, {}, []
    role_hashes = {}
    for stage, market in enumerate(DOMAINS[:2], 1):
        path = FIT / f"{market}_fit.json"
        with (PRIVATE / "fit_access.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(dict(market=market, sha256=FIT_SHA[stage-1], time=datetime.now().astimezone().isoformat()))+"\n")
        accounts = json.loads(pinned(path, FIT_SHA[stage-1]))
        require(len(accounts) == SIZES[stage-1], "Fit role size")
        role_hashes[market] = sha(json.dumps([a["account_uid"] for a in accounts],
                                             separators=(",", ":")).encode())
        for seed in SEEDS:
            carry, snapshots, catalog, index = run_stage(accounts, seed, stage, states.pop(seed, []),
                                                        groups_for, seed_for, Memory)
            if stage == 1:
                states[seed] = carry
            catalogs.setdefault(seed, {}).update(catalog)
            stages.setdefault(seed, {})[market] = snapshots
            indices.append(index)
            print(market, seed, "1000 paired layouts and historical reference frozen", flush=True)
        del accounts
    require(not states and len(indices) == 6, "Unfinished markets")
    files = [seal(PRIVATE / "catalog.json.gz", catalogs), seal(PRIVATE / "layouts.json.gz", stages)]
    byte_stats = {}
    for seed, markets in stages.items():
        for market, snapshots in markets.items():
            sizes = np.array([s["bytes"] for s in snapshots])
            byte_stats[f"{market}/{seed}"] = dict(max=sizes.max(axis=0).tolist(),
                over_cap=(sizes > CAP).sum(axis=0).tolist())
    receipt = dict(status="FROZEN", time=datetime.now().astimezone().isoformat(), seeds=SEEDS,
                   draws_per_seed=DRAWS, historical_layouts=3, cap=CAP, sources=sources,
                   gate_sha256=GATE_SHA, fit_sha256=dict(zip(DOMAINS[:2], FIT_SHA)),
                   row_order_sha256=role_hashes, original_indices=indices, manifests=files,
                   byte_stats=byte_stats, seconds=time.monotonic()-start)
    put_json(WORK / "freeze.json", receipt)
    (WORK / "freeze.json").chmod(0o400)


if __name__ == "__main__":
    runtime()
    select()
