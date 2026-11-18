"""Freeze market-local fit regroupings and their six Algorithm-R survivors."""
from datetime import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import sys
import time

import torch


WORK = Path(__file__).resolve().parent
PREVIOUS = WORK.parent / "vendorlink_six_group_coverage"
SELECT_SHA = "ab38a96a324e4ea6d79d65404a59a03239d1e2c7b5b1d52f5346692b0ae9b69b"
OLD_FREEZE = "099892d3d4b617dd9492e0a551347608c8bc8bfeafdc472b057201fb3491730b"
if hashlib.sha256((PREVIOUS / "select_groups.py").read_bytes()).hexdigest() != SELECT_SHA:
    raise ValueError("Unexpected fit or reservoir implementation")
sys.path.insert(0, str(PREVIOUS))
from select_groups import (CAP, DOMAINS, DRAWS, FIT, FIT_SHA, SEEDS, SIZES, Slot, fit_weights,
                           index_receipt, make_pool, memory_size, originals, pinned, put_json,
                           random_slots, require, runtime, seal, sha)

PRIVATE = WORK / "selection_private"
FILES = ("scope.zh.md", "regroup.py", "evaluate_regroup.py", "check_regroup.py")
ARMS = ("original", "reshuffled", "directed")


def now():
    return datetime.now().astimezone().isoformat()


def sources():
    return {name: sha((WORK / name).read_bytes()) for name in FILES}


def stream(seed, stage, rep, purpose):
    return random.Random(f"fit-regroup/v1/{seed}/{stage}/{rep}/{purpose}")


def disjoint(edges):
    used, pairs = set(), []
    for a, b in edges:
        if a not in used and b not in used:
            pairs.append((a, b))
            used.update((a, b))
    return pairs


def fit_pairs(weights, rng):
    scores = {}
    for (a, b), count in weights.items():
        key = (min(a, b), max(a, b))
        scores[key] = scores.get(key, 0) + count
    ranked = sorted(scores)
    rng.shuffle(ranked)
    ranked.sort(key=scores.get, reverse=True)
    return disjoint(ranked)


def directed_layout(n, rivals, seat_edges, rng):
    seats = list(seat_edges)
    rng.shuffle(seats)
    seats = disjoint(seats)
    rng.shuffle(seats)
    assigned = [-1] * n
    used = set()
    for (p, q), (a, b) in zip(seats, rivals):
        if rng.randrange(2):
            a, b = b, a
        assigned[p], assigned[q] = a, b
        used.update((a, b))
    left = [i for i in range(n) if i not in used]
    rng.shuffle(left)
    fill = iter(left)
    return [next(fill) if a == -1 else a for a in assigned], len(used) // 2


def view_table(pool, n):
    table = [[None] * n for _ in range(3)]
    for slot in pool:
        views = slot.row["group"]["views"]
        for ix, member in enumerate(slot.members):
            require(table[slot.packet][member] is None, "Repeated template position")
            table[slot.packet][member] = views[2*ix:2*ix+2]
    require(all(all(packet) for packet in table), "Missing template position")
    return table


def group_uid(seed, domain, packet, index, records):
    key = [seed, domain, 0, packet, index, records]
    return sha(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode())


def materialize(pool, table, index, layout, seed):
    template = pool[index]
    group = template.row["group"]
    members = tuple(layout[p] for p in template.members)
    views = tuple(v for a in members for v in table[template.packet][a])
    records = [[r["uid"] for r in v["records"]] for v in views]
    uid = group_uid(seed, group["domain"], template.packet, index % (len(pool)//3), records)
    # Preserve the original Memory serializer's container types and distinct teacher storage.
    row = dict(group=dict(uid=uid, domain=group["domain"], split=group["split"], views=views),
               origin=template.stage, teacher=torch.zeros_like(template.row["teacher"]))
    return Slot(uid, template.stage, template.packet, members, row)


def stage_layouts(accounts, seed, stage, states, offset, engine, old_indices):
    groups_for, seed_for, Memory = engine
    n = len(accounts)
    market = DOMAINS[stage-1]
    pool, split_views, catalog = make_pool(accounts, seed, stage, groups_for)
    table = view_table(pool, n)
    rivals = fit_pairs(fit_weights(split_views), stream(seed, stage, 0, "fit-ties"))
    seats = sorted({tuple(sorted(pair)) for g in pool for pair in combinations(g.members, 2)})
    order = sorted(range(offset, offset + len(pool)), key=lambda k: pool[k-offset].uid)
    templates = [dict(uid=g.uid, **catalog[g.uid]) for g in pool]
    identity = list(range(n))
    layouts, carry = [], []
    historical = None
    for rep in range(DRAWS + 1):
        random_map = identity.copy()
        stream(seed, stage, rep, "uniform-members").shuffle(random_map)
        directed, placed = directed_layout(n, rivals, seats, stream(seed, stage, rep, "directed-members"))
        require(sorted(random_map) == sorted(directed) == identity, "Invalid account-to-position bijection")
        if stage == 1:
            old_positions, old_arms = [], [[], [], []]
            rng = random.Random(seed_for(seed, DOMAINS, "reservoir") if rep == 0
                                else f"six-group-fit/v1/{seed}/{rep}")
        else:
            old_positions, old_arms, rng = states[rep]
        positions, seen = random_slots(old_positions, order, rng, offset)
        current = [[], [], []]
        for slot, position in enumerate(positions):
            if position < offset:
                require(old_positions[slot] == position, "Revived old position")
                for arm in range(3):
                    current[arm].append(old_arms[arm][slot])
            else:
                k = position-offset
                current[0].append(pool[k])
                current[1].append(materialize(pool, table, k, random_map, seed))
                current[2].append(materialize(pool, table, k, directed, seed))
        sizes = [memory_size(groups, seen, rng.getstate(), Memory) for groups in current]
        if rep == 0:
            historical = index_receipt(seed, stage, current[0], pool, seen, sizes[0])
            require(historical == old_indices[seed, stage], "Historical original index changed")
        layouts.append(dict(perm=[random_map, directed], slots=positions, placed=placed,
                            uids=[[g.uid for g in groups] for groups in current], bytes=sizes))
        if stage == 1:
            carry.append((positions, current, rng))
    name = f"{market}_{seed}.json.gz"
    manifest = seal(PRIVATE / name, dict(seed=seed, stage=stage, offset=offset, accounts=n,
                         templates=templates, layouts=layouts))
    stats = dict(seed=seed, market=market, templates=len(pool), seen=offset+len(pool),
                 disjoint_fit_pairs=len(rivals), placed_minmax=[min(r["placed"] for r in layouts),
                 max(r["placed"] for r in layouts)], max_bytes_by_arm=[max(r["bytes"][k] for r in layouts)
                 for k in range(3)], over_cap=sum(max(r["bytes"]) > CAP for r in layouts))
    return manifest, stats, historical, carry


def freeze():
    start = time.monotonic()
    bindings = sources()
    check = json.loads((WORK / "check.json").read_bytes())
    require(check["status"] == "PASS" and check["sources"] == bindings, "Current synthetic check required")
    old = json.loads(pinned(PREVIOUS / "freeze.json", OLD_FREEZE))
    old_indices = {(r["seed"], r["stage"]): r for r in old["original_indices"]}
    engine = originals()
    PRIVATE.mkdir(mode=0o700, exist_ok=False)
    manifests, stats, indices, carried, orders = [], [], [], {}, {}
    for stage, market in enumerate(DOMAINS[:2], 1):
        with (PRIVATE / "fit_access.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(dict(market=market, time=now(), sha256=FIT_SHA[stage-1]))+"\n")
        accounts = json.loads(pinned(FIT / f"{market}_fit.json", FIT_SHA[stage-1]))
        require(len(accounts) == SIZES[stage-1], "Fit count")
        ids = [a["account_uid"] for a in accounts]
        orders[market] = sha(json.dumps(ids, separators=(",", ":")).encode())
        require(orders[market] == old["row_order_sha256"][market], "Original ordered account identity")
        for seed in SEEDS:
            states, offset = carried.pop(seed, ([], 0))
            manifest, stat, index, carry = stage_layouts(accounts, seed, stage, states, offset,
                                                         engine, old_indices)
            manifests.append(manifest)
            stats.append(stat)
            indices.append(index)
            if stage == 1:
                carried[seed] = (carry, stat["seen"])
            del states, carry
            print(market, seed, "paired layouts frozen:", DRAWS, "over cap:", stat["over_cap"], flush=True)
        del accounts, ids
    require(not carried and len(manifests) == 6, "Incomplete freeze")
    receipt = dict(status="FROZEN" if all(r["over_cap"] == 0 for r in stats) else "CAPACITY_EXCEEDED",
                   time=now(), sources=bindings, selector_sha256=SELECT_SHA, old_freeze_sha256=OLD_FREEZE,
                   original_gate_sha256=old["gate_sha256"], fit_sha256=dict(zip(DOMAINS[:2], FIT_SHA)),
                   row_order_sha256=orders, arms=ARMS, seeds=SEEDS, replicates=DRAWS,
                   historical_draws_per_seed=1, max_slots=6, cap=CAP, stats=stats,
                   indices=indices, manifests=manifests, seconds=time.monotonic()-start)
    put_json(WORK / "freeze.json", receipt)
    (WORK / "freeze.json").chmod(0o400)
    require(receipt["status"] == "FROZEN", "Capacity exceeded; all draws recorded")


if __name__ == "__main__":
    runtime()
    freeze()
