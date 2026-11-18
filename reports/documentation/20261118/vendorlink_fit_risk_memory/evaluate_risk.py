"""Evaluate the twelve registered ER endpoints only after the entire new freeze."""
from collections import Counter
import json
import sys
import time

import numpy as np

from risk_memory import (CAP, DOMAINS, DRAWS, PRIOR, PRIOR_FREEZE, PRIVATE, SEEDS, SIZES, WORK,
                         now, packet_index, pinned, put_json, read_manifest, require, runtime,
                         sha, sources)
from regroup import group_uid

PRIOR_SUM = "acf1eec5efff754d073a9b533e609fee52b98e5d8cf52d6bb4ca62bb2ab0a3ec"
pinned(PRIOR / "evaluate_regroup.py", "d36227fb2eb8ad9912c95df6455f66d2a520c87b7302da2070f65a88a745bdd7")
from evaluate_regroup import TOP5, TOP5_SOURCE, events


def incidence(members):
    masks = {}
    for i, ids in enumerate(members):
        for account in ids:
            masks[account] = masks.get(account, 0) | (1 << i)
    return Counter(masks.values())


def validate_pair(old, new, previous):
    stage, seed, offset = old["stage"], old["seed"], old["offset"]
    n = old["accounts"]
    require((new["stage"], new["seed"], n) == (stage, seed, SIZES[stage-1])
            and len(new["layouts"]) == DRAWS and len(old["layouts"]) == DRAWS + 1, "Layout identity")
    groups = old["templates"]
    require(offset == (len(previous[0]["templates"]) if stage == 2 else 0), "Stage boundary")
    records = [[None]*n for _ in range(3)]
    for g in groups:
        for k, a in enumerate(g["members"]):
            records[g["packet"]][a] = [v[1] for v in g["views"][2*k:2*k+2]]
    edges = new["ordered_risks"]
    pairs = {(q, j) for q, j, _, _, _ in edges}
    require(len(pairs) == len(edges) and all(0 <= q < n and 0 <= j < n and q != j and
            1 <= w == min(c, v) and c+v <= 6 for q, j, c, v, w in edges), "Directed risk list")
    for rep, row in enumerate(new["layouts"], 1):
        base = old["layouts"][rep]
        require(len(row["seats"]) == len(set(row["seats"])) == len(set(row["uids"])) ==
                len(row["uids"]) == 6 and 0 < row["bytes"] <= CAP and 0 < base["bytes"][2] <= CAP,
                "Invalid frozen six slots")
        perm = base["perm"][1]
        assigned = [[[] for _ in range(6)] for _ in range(2)]
        for i, (a, b) in enumerate(zip(base["slots"], row["seats"])):
            if a < offset:
                old_old, old_new = previous
                require(stage == 2 and a == old_old["layouts"][rep]["slots"][i]
                        and b == old_new["layouts"][rep-1]["seats"][i]
                        and row["uids"][i] == old_new["layouts"][rep-1]["uids"][i],
                        "Prioritized AlphaBay group did not survive in its own slot")
                # These unchanged old slots were already validated at AlphaBay exit.
                continue
            require(offset <= b < offset+len(groups), "Current slot changed origin")
            g, h = groups[a-offset], groups[b-offset]
            packet = h["packet"]
            require(g["packet"] == packet and len(g["members"]) == len(h["members"]), "Slot shape")
            for dest, t in zip(assigned, (g, h)):
                dest[i] = [perm[p] for p in t["members"]]
            content = [v for p in h["members"] for v in records[packet][perm[p]]]
            require(group_uid(seed, DOMAINS[stage-1], packet, (b-offset) % (len(groups)//3), content)
                    == row["uids"][i], "Prioritized slot content")
        # Current-domain accounts and old-domain accounts have distinct identity namespaces.
        require(incidence(assigned[0]) == incidence(assigned[1]),
                "Different account intersection multiplicities")


def unpack():
    raw = (WORK / "freeze.json").read_bytes()
    f = json.loads(raw)
    require(f["status"] == "FROZEN" and f["sources"] == sources() and f["prior_freeze_sha256"] == PRIOR_FREEZE
            and f["draws_per_seed"] == DRAWS and f["cap"] == CAP and tuple(f["seeds"]) == SEEDS
            and len(f["manifests"]) == 6 and f["shared_consumers"] == ["Full", "ER"], "Incomplete freeze")
    prior = json.loads(pinned(PRIOR / "freeze.json", PRIOR_FREEZE))
    all_rows = {}
    for s in SEEDS:
        for stage, market in enumerate(DOMAINS[:2], 1):
            name = f"{market}_{s}.json.gz"
            old = read_manifest(PRIOR / "selection_private", prior, name)
            new = read_manifest(PRIVATE, f, name)
            require((old["seed"], old["stage"]) == (s, stage), "Unexpected market or seed")
            validate_pair(old, new, all_rows.get((s, 1)))
            all_rows[s, stage] = old, new
    return sha(raw), all_rows


def cover(pair, subsequent, lost):
    old, new = pair
    group_at = packet_index(old["templates"], old["accounts"]) + old["offset"]
    q, w = (np.asarray([p[k] for p in lost], dtype=int) for k in (0, 1))
    originally_reachable = (group_at[:, q] == group_at[:, w]).any(axis=0)
    edge_set = {(a, b) for a, b, _, _, _ in new["ordered_risks"]}
    predicted = np.asarray([p in edge_set for p in lost], dtype=bool)
    layouts = old["layouts"][1:]
    inv = np.argsort([r["perm"][1] for r in layouts], axis=1)

    def slot_arrays(snapshot):
        baseline, targeted = snapshot
        return np.asarray([r["slots"] for r in baseline["layouts"][1:]]), np.asarray(
            [r["seats"] for r in targeted["layouts"]])

    layers = {"exit": slot_arrays(pair)}
    if subsequent is not None:
        layers["pre_agora"] = slot_arrays(subsequent)
    shape = (len(layouts), len(lost))
    pool = np.zeros(shape, dtype=bool)
    masks = {name: [np.zeros(shape, dtype=bool) for _ in range(2)] for name in layers}
    for packet in group_at:
        location = packet[inv[:, q]]
        joint = location == packet[inv[:, w]]
        pool |= joint
        for name, arms in layers.items():
            for covered, slots in zip(masks[name], arms):
                covered |= joint & (location[:, :, None] == slots[:, None, :]).any(axis=2)
    for arm in range(2):
        at_exit = masks["exit"][arm]
        require((~at_exit | pool).all() and ("pre_agora" not in masks or
                (~masks["pre_agora"][arm] | at_exit).all()), "Non-nested survival")
    return originally_reachable, predicted, pool, masks


def paired_tally(arms, subset, predicted):
    uniform, priority = [x[:, subset] for x in arms]
    scored = predicted[subset]
    counts = {"both": (uniform & priority).sum(axis=1), "only_uniform": (uniform & ~priority).sum(axis=1),
              "only_priority": (~uniform & priority).sum(axis=1), "neither": (~uniform & ~priority).sum(axis=1),
              "uniform": uniform.sum(axis=1), "priority": priority.sum(axis=1),
              "guessed_uniform": (uniform & scored).sum(axis=1), "guessed_priority": (priority & scored).sum(axis=1)}
    require(np.all(sum(counts[k] for k in ("both", "only_uniform", "only_priority", "neither"))
                   == int(subset.sum())), "Lost pair partition")
    counts["net"] = counts["priority"] - counts["uniform"]
    totals = {k: int(v.sum()) for k, v in counts.items()}
    return dict(totals=totals, means={k: v/len(uniform) for k, v in totals.items()},
                net_histogram=dict(sorted(Counter(map(int, counts["net"])).items())))


def decide(rows):
    a = [r for r in rows if r["market"] == "AlphaBay"]
    def net(r, name, layer):
        return r["subsets"][name]["coverage"][layer]["totals"]["net"]
    exit_nets = [net(r, "original_miss", "exit") for r in a]
    pre = sum(net(r, "original_miss", "pre_agora") for r in a)
    both = [sum(net(r, "all", k) for r in a) for k in ("exit", "pre_agora")]
    positive = sum(x > 0 for x in exit_nets)
    return dict(missed_exit_net=sum(exit_nets)/DRAWS, positive_exit_seeds=positive,
                missed_pre_agora_net=pre/DRAWS, all_exit_net=both[0]/DRAWS, all_pre_agora_net=both[1]/DRAWS,
                discussion_gate=sum(exit_nets) >= DRAWS and positive >= 2 and pre > 0 and min(both) >= 0)


def evaluate():
    start = time.monotonic()
    digest, frozen = unpack()
    prior = json.loads(pinned(PRIOR / "result/summary.json", PRIOR_SUM))
    anchors = {(r["seed"], r["market"]): r for r in prior["rows"]}
    pinned(TOP5 / "challengers.py", TOP5_SOURCE)
    sys.path.insert(0, str(TOP5))
    from challengers import inspect_er, read_endpoint
    out = WORK / "result"
    out.mkdir(mode=0o700, exist_ok=False)
    inputs = inspect_er()
    rows = []
    with (out / "access.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        for seed in SEEDS:
            for stage, cells in enumerate((("R11", "R31"), ("R22", "R32")), 1):
                market = DOMAINS[stage-1]
                anchor = anchors[seed, market]
                endpoints = []
                for cell in cells:
                    item = inputs[seed, cell]
                    log.write(json.dumps(dict(time=now(), seed=seed, cell=cell, sha256=item[1]["sha256"],
                                              freeze_sha256=digest))+"\n")
                    endpoints.append(read_endpoint(item, SIZES[stage-1]))
                lost = events(*endpoints)
                pair = frozen[seed, stage]
                reachable, pred, pool, masks = cover(pair, frozen[seed, 2] if stage == 1 else None, lost)
                require(len(lost) == anchor["losses"] and int((endpoints[0][1] == 1).sum()) == anchor["initially_first"]
                        and int((~reachable).sum()) == anchor["original_missed"], "Historical event population")
                subsets = {}
                for name, subset in (("original_miss", ~reachable), ("original_reachable", reachable),
                                     ("all", np.ones(len(lost), dtype=bool))):
                    old = anchor["coverage"][name]
                    pool_n = int(pool[:, subset].sum())
                    require(pool_n == old["pool"]["totals"]["directed"], "Previously frozen candidate pool")
                    coverage = {k: paired_tally(v, subset, pred) for k, v in masks.items()}
                    for k, tally in coverage.items():
                        require(tally["totals"]["uniform"] == old[k]["totals"]["directed"], "Same-pool uniform anchor")
                    subsets[name] = dict(events=int(subset.sum()), guessed=int((pred & subset).sum()),
                                         pool_mean=pool_n/DRAWS, guessed_in_pool_mean=float(pool[:, pred & subset].sum()/DRAWS),
                                         coverage=coverage)
                rows.append(dict(seed=seed, market=market, predictions=len(pair[1]["ordered_risks"]),
                                 subsets=subsets))
    require(len(rows) == 6 and [r["subsets"]["original_miss"]["events"] for r in rows] == [15, 1, 13, 2, 19, 4],
            "Incomplete market/seed evaluation")
    put_json(out / "summary.json", dict(status="COMPLETE", time=now(), freeze_sha256=digest, draws_per_seed=DRAWS,
             endpoints=12, metric_anchors=36, rows=rows, decision=decide(rows), seconds=time.monotonic()-start))


if __name__ == "__main__":
    runtime()
    evaluate()
