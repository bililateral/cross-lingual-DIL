"""Evaluate the same twelve saved ER matrices after verifying the complete freeze."""
from collections import Counter
import gzip
import json
import sys
import time

import numpy as np

from regroup import (ARMS, CAP, DOMAINS, DRAWS, OLD_FREEZE, PREVIOUS, PRIVATE, SEEDS, SIZES,
                     WORK, group_uid, now, pinned, put_json, require, runtime, sha, sources)

PREV_EVAL = "d717b8f75438b434042abb8399b7447a1531efb353eaee25be7c98b0b7e1ed58"
PREV_SUM = "256050bee943107dcfc9384ed8e9c7baf89fc5a8cbc21006806c79a9e9b65138"
pinned(PREVIOUS / "evaluate.py", PREV_EVAL)
from evaluate import TOP5, TOP5_SOURCE, TOP5_SUMMARY, events


def validate_market(data, prior, stat):
    seed, stage, n, offset = (data[k] for k in ("seed", "stage", "accounts", "offset"))
    groups, layouts = data["templates"], data["layouts"]
    require(n == SIZES[stage-1] and offset == (len(prior["templates"]) if prior else 0)
            and len(groups) == stat["templates"] and len(groups) % 3 == 0
            and len(layouts) == DRAWS + 1, "Manifest shape")
    by_packet = [[None] * n for _ in range(3)]
    ids = set()
    for i, g in enumerate(groups):
        p = i // (len(groups)//3)
        require(g["stage"] == stage and g["packet"] == p and len(g["views"]) == 2*len(g["members"])
                and g["uid"] not in ids, "Invalid template")
        ids.add(g["uid"])
        for j, member in enumerate(g["members"]):
            require(by_packet[p][member] is None, "Repeated packet account")
            by_packet[p][member] = [v[1] for v in g["views"][2*j:2*j+2]]
        require(group_uid(seed, DOMAINS[stage-1], p, i % (len(groups)//3), [v[1] for v in g["views"]])
                == g["uid"], "Original content UID")
    require(all(all(packet) for packet in by_packet), "Incomplete packet")
    identity = list(range(n))
    for rep, row in enumerate(layouts):
        require(len(row["perm"]) == 2 and all(sorted(m) == identity for m in row["perm"])
                and len(row["slots"]) == len(set(row["slots"])) == 6 and len(row["bytes"]) == 3
                and 0 < min(row["bytes"]) <= max(row["bytes"]) <= CAP and len(row["uids"]) == 3
                and all(len(s) == len(set(s)) == 6 for s in row["uids"]), "Invalid paired memory")
        for k, seat in enumerate(row["slots"]):
            require(0 <= seat < offset+len(groups), "Invalid slot index")
            if seat < offset:
                old = prior["layouts"][rep]
                require(old["slots"][k] == seat and all(row["uids"][a][k] == old["uids"][a][k]
                        for a in range(3)), "Changed or reintroduced AlphaBay group")
                continue
            g = groups[seat-offset]
            require(row["uids"][0][k] == g["uid"], "Historical membership")
            for arm, permutation in enumerate(row["perm"], 1):
                records = [v for pos in g["members"] for v in by_packet[g["packet"]][permutation[pos]]]
                expected = group_uid(seed, DOMAINS[stage-1], g["packet"], (seat-offset) % (len(groups)//3), records)
                require(row["uids"][arm][k] == expected, "Survivor content disagrees with frozen permutation")


def unpack():
    raw = (WORK / "freeze.json").read_bytes()
    freeze = json.loads(raw)
    require(freeze["status"] == "FROZEN" and freeze["sources"] == sources() and tuple(freeze["arms"]) == ARMS
            and tuple(freeze["seeds"]) == SEEDS and freeze["replicates"] == DRAWS
            and freeze["cap"] == CAP and len(freeze["indices"]) == 6, "Incomplete freeze")
    old = json.loads(pinned(PREVIOUS / "freeze.json", OLD_FREEZE))
    require(freeze["indices"] == old["original_indices"] and freeze["fit_sha256"] == old["fit_sha256"]
            and freeze["row_order_sha256"] == old["row_order_sha256"], "Historical identity")
    files = {row["name"]: row for row in freeze["manifests"]}
    stats = {(r["market"], r["seed"]): r for r in freeze["stats"]}
    require(len(files) == len(stats) == 6, "Missing paired manifests")
    out = {}
    for s in SEEDS:
        for stage, market in enumerate(DOMAINS[:2], 1):
            name = f"{market}_{s}.json.gz"
            compressed = pinned(PRIVATE / name, files[name]["sha256"])
            require(len(compressed) == files[name]["bytes"], "Compressed layout size")
            data = json.loads(gzip.decompress(compressed))
            require(data["seed"] == s and data["stage"] == stage, "Wrong layout identity")
            validate_market(data, out.get((s, 1)), stats[market, s])
            out[s, stage] = data
    return sha(raw), out


def hits(data, next_market, pairs):
    n, groups, rows = data["accounts"], data["templates"], data["layouts"]
    group_at = np.empty((3, n), dtype=np.int32)
    for i, group in enumerate(groups, data["offset"]):
        group_at[group["packet"], group["members"]] = i
    slots = np.asarray([r["slots"] for r in rows])
    after = np.asarray([r["slots"] for r in next_market["layouts"]]) if next_market else None
    q = np.asarray([pair[0] for pair in pairs], dtype=int)
    w = np.asarray([pair[1] for pair in pairs], dtype=int)
    by_layer = {k: [] for k in (("pool", "exit", "pre_agora") if next_market else ("pool", "exit"))}
    for arm in range(3):
        inv = (np.argsort([r["perm"][arm-1] for r in rows], axis=1) if arm else
               np.broadcast_to(np.arange(n), (len(rows), n)))
        shape = (len(rows), len(pairs))
        covered, at_exit, retained = (np.zeros(shape, dtype=bool) for _ in range(3))
        for packet in group_at:
            location = packet[inv[:, q]]
            joint = location == packet[inv[:, w]]
            covered |= joint
            at_exit |= joint & (location[:, :, None] == slots[:, None, :]).any(axis=2)
            if next_market:
                retained |= joint & (location[:, :, None] == after[:, None, :]).any(axis=2)
        require(np.all(~at_exit | covered) and np.all(~retained | at_exit), "Non-nested coverage")
        by_layer["pool"].append(covered)
        by_layer["exit"].append(at_exit)
        if next_market:
            by_layer["pre_agora"].append(retained)
    return by_layer


def summarize(masks, subset):
    orig, random, directed = [x[:, subset] for x in masks]
    draws = len(orig) - 1
    both = random & directed
    keys = {"original": orig, "reshuffled": random, "directed": directed, "both": both,
            "only_reshuffled": random & ~directed, "only_directed": ~random & directed,
            "neither": ~(random | directed)}
    counts = {k: values.sum(axis=1) for k, values in keys.items()}
    delta = counts["directed"]-counts["reshuffled"]
    require(np.all(counts["both"]+counts["only_directed"]+counts["only_reshuffled"]+counts["neither"]
                   == int(subset.sum())), "Pair partition")
    totals = {k: int(values[1:].sum()) for k, values in counts.items()}
    totals["net"] = int(delta[1:].sum())
    return dict(events=int(subset.sum()), historical_schedule={k: int(v[0]) for k, v in counts.items()},
                draws=draws, totals=totals, means={k: v/draws for k, v in totals.items()},
                net_histogram=dict(sorted(Counter(int(v) for v in delta[1:]).items())))


def decision(rows):
    alpha = [r for r in rows if r["market"] == "AlphaBay"]
    def net(row, subset, layer):
        return row["coverage"][subset][layer]["totals"]["net"]
    values = [net(r, "original_miss", "exit") for r in alpha]
    exit_sum = sum(values)
    remaining = sum(net(r, "original_miss", "pre_agora") for r in alpha)
    protected = [sum(net(r, "all", name) for r in alpha) for name in ("exit", "pre_agora")]
    positive = sum(v > 0 for v in values)
    return dict(missed_exit_net=exit_sum/DRAWS, missed_positive_seeds=positive,
                missed_pre_agora_net=remaining/DRAWS, all_exit_net=protected[0]/DRAWS,
                all_pre_agora_net=protected[1]/DRAWS,
                discussion_gate=exit_sum >= DRAWS and positive >= 2 and remaining > 0 and min(protected) >= 0)


def evaluate():
    start_time = time.monotonic()
    freeze_sha, frozen = unpack()
    old = json.loads(pinned(PREVIOUS / "result/summary.json", PREV_SUM))
    ref = {(r["seed"], r["market"]): r for r in old["rows"]}
    retained_ref = {r["seed"]: r for r in old["alpha_before_agora"]}
    top5 = json.loads(pinned(TOP5 / "result/summary.json", TOP5_SUMMARY))
    denominators = {(r["seed"], r["market"]): r for r in top5["rows"]}
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
                endpoints = []
                for cell in cells:
                    item = inputs[seed, cell]
                    log.write(json.dumps(dict(seed=seed, cell=cell, time=now(), sha256=item[1]["sha256"],
                                              freeze_sha256=freeze_sha))+"\n")
                    endpoints.append(read_endpoint(item, SIZES[stage-1]))
                lost = events(*endpoints)
                den = denominators[seed, market]
                require(len(lost) == den["lost_first"] and int((endpoints[0][1] == 1).sum()) == den["initially_first"],
                        "Wrong original ER event denominator")
                layers = hits(frozen[seed, stage], frozen[seed, 2] if stage == 1 else None, lost)
                old_pool = layers["pool"][0]
                reachable = old_pool[0]
                ref_row = ref[seed, market]
                require(np.all(old_pool == reachable) and int(reachable.sum()) == ref_row["pool_ceiling"],
                        "Original reachable subset differs")
                subsets = {"all": np.ones(len(lost), dtype=bool), "original_miss": ~reachable,
                           "original_reachable": reachable}
                summary = {name: {layer: summarize(masks, mask) for layer, masks in layers.items()}
                           for name, mask in subsets.items()}
                for layer, anchor in [("exit", ref_row)] + ([("pre_agora", retained_ref[seed])] if stage == 1 else []):
                    row = summary["all"][layer]
                    require(row["means"]["original"] == anchor["paired_mean"]["random_hits"] and
                            row["historical_schedule"]["original"] == anchor["historical"]["random_hits"],
                            "Historical reservoir trace not reproduced")
                rows.append(dict(seed=seed, market=market, initially_first=den["initially_first"],
                                 losses=len(lost), original_missed=int((~reachable).sum()),
                                 original_reachable=int(reachable.sum()), coverage=summary))
    require(len(rows) == 6 and [r["original_missed"] for r in rows] == [15, 1, 13, 2, 19, 4],
            "Incomplete original missed population")
    put_json(out / "summary.json", dict(status="COMPLETE", time=now(), freeze_sha256=freeze_sha,
             draws_per_seed=DRAWS, endpoints=12, metric_anchors=36, rows=rows,
             decision=decision(rows), seconds=time.monotonic()-start_time))


if __name__ == "__main__":
    runtime()
    evaluate()
