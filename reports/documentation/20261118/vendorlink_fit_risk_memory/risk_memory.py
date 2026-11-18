"""Freeze directed fit predictions and both six-slot policies before ER access."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

WORK = Path(__file__).resolve().parent
PRIOR = WORK.parent / "vendorlink_fit_regroup"
PRIOR_FREEZE = "27d8c69e346097ce782b869322fbfc963c0cbc325a1a5df26e7f1ec4471a87ba"
REGROUP_SHA = "3c968dd1e8c07d6a1ccfbc4846ed18f28f1b1d91deb801563584b1c7cbd6197b"
if hashlib.sha256((PRIOR / "regroup.py").read_bytes()).hexdigest() != REGROUP_SHA:
    raise ValueError("Unexpected group materializer")
sys.path.insert(0, str(PRIOR))
from regroup import materialize, now, view_table
from select_groups import (CAP, DOMAINS, DRAWS, FIT, FIT_SHA, SEEDS, SIZES, cosine_field,
                           make_pool, memory_size, originals, pinned, put_json, random_slots,
                           require, runtime, seal, sha)

PRIVATE = WORK / "selection_private"
FILES = ("scope.zh.md", "risk_memory.py", "evaluate_risk.py", "check_risk.py")


def sources():
    return {name: sha((WORK / name).read_bytes()) for name in FILES}


def read_manifest(directory, receipt, name):
    item, = [r for r in receipt["manifests"] if r["name"] == name]
    compressed = pinned(directory / name, item["sha256"])
    require(len(compressed) == item["bytes"], "Frozen layout size")
    return json.loads(gzip.decompress(compressed))


def fold_winners(scores):
    n = len(scores)
    require(scores.shape == (n, n) and n and np.isfinite(scores).all(), "Invalid lexical scores")
    winners = scores.argmax(axis=1)
    return np.where(scores[np.arange(n), winners] > 0, winners, -1)


def risk_edges(winners):
    """Return q, competitor, correct folds, competitor folds, directed weight."""
    require(winners.ndim == 2 and len(winners) == 6, "Six aligned fit views required")
    records = []
    for q, column in enumerate(winners.T):
        counts = Counter(map(int, column))
        if counts[q]:
            for j in sorted(counts):
                if j >= 0 and j != q:
                    records.append([q, j, counts[q], counts[j], min(counts[q], counts[j])])
    return records


def predict(views):
    results = []
    for fold in range(6):
        scores = np.zeros((len(views), len(views)), dtype=np.float64)
        for field in ("title", "description"):
            queries = [" ".join(getattr(r, field) for r in account[fold]) for account in views]
            refs = [" ".join(getattr(r, field) for k, pair in enumerate(account) if k != fold
                             for r in pair) for account in views]
            scores += 0.5 * cosine_field(refs, queries)
        results.append(fold_winners(scores))
    winners = np.asarray(results)
    return risk_edges(winners), int((winners == -1).sum())


def packet_index(templates, n):
    group_at = np.full((3, n), -1, dtype=np.int32)
    for i, g in enumerate(templates):
        group_at[g["packet"], g["members"]] = i
    require((group_at >= 0).all(), "Incomplete frozen packets")
    return group_at


def candidate_edges(group_at, count, perm, q, j):
    """Translate account-space predictions to the positions of this frozen permutation."""
    inverse = np.argsort(perm)
    edges = [set() for _ in range(count)]
    for packet in group_at:
        left, right = packet[inverse[q]], packet[inverse[j]]
        for idx in np.flatnonzero(left == right):
            edges[left[idx]].add(int(idx))
    return edges


def select_slots(baseline, templates, choices, edges, weights):
    """Indices here are local to the current market. None marks an old-domain slot."""
    chosen = list(baseline)
    changed = 0
    for i, old in enumerate(baseline):
        if old is None:
            continue
        covered, masks, occupied = set(), defaultdict(int), set()
        for k, other in enumerate(chosen):
            if k != i and other is not None:
                occupied.add(other)
                covered.update(edges[other])
                for pos in templates[other]["members"]:
                    masks[pos] |= 1 << k

        def signature(g):
            return sorted(masks.get(p, 0) for p in templates[g]["members"])

        def gain(g):
            return sum(weights[e] for e in edges[g] if e not in covered)

        shape = signature(old)
        best, value = old, gain(old)
        t = templates[old]
        for g in choices[t["packet"], len(t["members"])]:
            if g in occupied:
                continue
            score = gain(g)
            if score > value and signature(g) == shape:
                best, value = g, score
        chosen[i] = best
        changed += int(best != old)
    return chosen, changed


def objective(slots, edges, weights):
    covered = set().union(*(edges[g] for g in slots if g is not None))
    return sum(weights[e] for e in covered)


def select_market(accounts, seed, stage, prior, previous, groups_for, Memory):
    market = DOMAINS[stage-1]
    name = f"{market}_{seed}.json.gz"
    data = read_manifest(PRIOR / "selection_private", prior, name)
    offset, templates = data["offset"], data["templates"]
    pool, views, catalog = make_pool(accounts, seed, stage, groups_for)
    n = len(accounts)
    require((data["seed"], data["stage"], data["accounts"]) == (seed, stage, n)
            and len(data["layouts"]) == DRAWS + 1
            and templates == [dict(uid=g.uid, **catalog[g.uid]) for g in pool],
            "Current fit group content differs from frozen catalog")
    predictions, abstentions = predict(views)
    q, j = (np.asarray([e[k] for e in predictions], dtype=int) for k in (0, 1))
    weights = [e[4] for e in predictions]
    seats = packet_index(templates, n)
    choices = defaultdict(list)
    for i, g in enumerate(templates):
        choices[g["packet"], len(g["members"])].append(i)
    table = view_table(pool, n)
    order = sorted(range(offset, offset+len(pool)), key=lambda g: pool[g-offset].uid)
    rows, carry = [], []
    changes = improved = total_gain = peak = 0
    for rep in range(1, DRAWS + 1):
        layout = data["layouts"][rep]
        perm = layout["perm"][1]
        if previous is None:
            old_slots, old_ids, old_groups = [], [], []
            rng = random.Random(f"six-group-fit/v1/{seed}/{rep}")
        else:
            old_slots, old_ids, old_groups, rng = previous[rep-1]
        baseline, seen = random_slots(old_slots, order, rng, offset)
        require(baseline == layout["slots"], "Original Algorithm R schedule differs")
        local = [v-offset if v >= offset else None for v in baseline]
        edges = candidate_edges(seats, len(pool), perm, q, j)
        chosen, changed = select_slots(local, templates, choices, edges, weights)
        before, after = (objective(s, edges, weights) for s in (local, chosen))
        require(after >= before and (after > before) == (changed > 0), "Invalid marginal coverage")
        ids, groups = [], []
        for k, g in enumerate(chosen):
            if g is None:
                require(baseline[k] == old_slots[k], "Reintroduced old slot")
                ids.append(old_ids[k])
                groups.append(old_groups[k])
            else:
                ids.append(g+offset)
                group = materialize(pool, table, g, perm, seed)
                if g == local[k]:
                    require(group.uid == layout["uids"][2][k], "Uniform content UID")
                groups.append(group)
        size = memory_size(groups, seen, rng.getstate(), Memory)
        require(len(set(ids)) == len({g.uid for g in groups}) == 6
                and 0 < size <= CAP and 0 < layout["bytes"][2] <= CAP, "Six-slot capacity")
        rows.append(dict(seats=ids, uids=[g.uid for g in groups], bytes=size))
        if stage == 1:
            carry.append((baseline, ids, groups, rng))
        changes += changed
        improved += after > before
        total_gain += after-before
        peak = max(peak, size)
    manifest = seal(PRIVATE / name, dict(seed=seed, stage=stage, ordered_risks=predictions,
                                         layouts=rows))
    stats = dict(market=market, seed=seed, directed_predictions=len(predictions),
                 predicted_queries=len(set(q.tolist())), abstained_query_folds=abstentions,
                 improved_draws=improved, changed_slots=changes, total_fit_objective_gain=total_gain,
                 max_priority_bytes=peak, max_uniform_bytes=max(r["bytes"][2] for r in data["layouts"][1:]))
    return manifest, stats, carry


def freeze():
    start = time.monotonic()
    bindings = sources()
    check = json.loads((WORK / "check.json").read_bytes())
    require(check["status"] == "PASS" and check["sources"] == bindings, "Current synthetic check missing")
    prior = json.loads(pinned(PRIOR / "freeze.json", PRIOR_FREEZE))
    require(prior["status"] == "FROZEN" and prior["replicates"] == DRAWS and prior["cap"] == CAP
            and prior["arms"][2] == "directed", "Wrong prior layout arm")
    groups_for, _, Memory = originals()
    PRIVATE.mkdir(mode=0o700, exist_ok=False)
    manifests, stats, carried = [], [], {}
    with (WORK / "fit_access.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        for stage, market in enumerate(DOMAINS[:2], 1):
            log.write(json.dumps(dict(time=now(), market=market, sha256=FIT_SHA[stage-1]))+"\n")
            accounts = json.loads(pinned(FIT / f"{market}_fit.json", FIT_SHA[stage-1]))
            ids = [a["account_uid"] for a in accounts]
            require(len(accounts) == SIZES[stage-1] and
                    sha(json.dumps(ids, separators=(",", ":")).encode()) == prior["row_order_sha256"][market],
                    "Original fit account alignment")
            for seed in SEEDS:
                manifest, stat, carry = select_market(accounts, seed, stage, prior, carried.pop(seed, None),
                                                       groups_for, Memory)
                manifests.append(manifest)
                stats.append(stat)
                if stage == 1:
                    carried[seed] = carry
                del carry
                print(market, seed, "sealed", DRAWS, flush=True)
            del accounts, ids
    require(len(manifests) == 6 and not carried, "Incomplete freeze")
    put_json(WORK / "freeze.json", dict(status="FROZEN", time=now(), sources=bindings,
             prior_freeze_sha256=PRIOR_FREEZE, draws_per_seed=DRAWS, seeds=SEEDS,
             uniform_layout_arm="directed", shared_consumers=["Full", "ER"], cap=CAP,
             fit_sha256=prior["fit_sha256"], row_order_sha256=prior["row_order_sha256"],
             manifests=manifests, stats=stats, seconds=time.monotonic()-start))
    (WORK / "freeze.json").chmod(0o400)


if __name__ == "__main__":
    runtime()
    freeze()
