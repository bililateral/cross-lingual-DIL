"""Evaluate frozen market-exit buffers against twelve original ER endpoints."""
from collections import Counter
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np

from select_groups import CAP, DOMAINS, DRAWS, FIT_SHA, PRIVATE, SEEDS, SIZES, WORK, pinned, put_json, require, runtime, sha


TOP5 = WORK.parent / "vendorlink_er_top5"
TOP5_SOURCE = "e69ae966d9d038fc3b0a69d683b58d1da5ff2c07b3db07d9ea6bb57435d6a23f"
TOP5_SUMMARY = "06c90f09c1592ac23d6f6277e1c7ae86a559e941252092551b8906de54b60807"
STRATEGIES = ("random", "mixed")


def overlap(groups):
    masks = {}
    for k, g in enumerate(groups):
        for a in g["members"]:
            key = (g["stage"], a)
            masks[key] = masks.get(key, 0) | (1 << k)
    return Counter(masks.values())


def validate(catalogs, layouts):
    require(set(catalogs) == set(layouts) == {str(s) for s in SEEDS}, "All three layout seeds required")
    for s in layouts:
        catalog, stages = catalogs[s], layouts[s]
        require(set(stages) == set(DOMAINS[:2]), "Both market exits required")
        for stage, market in enumerate(DOMAINS[:2], 1):
            require(len(stages[market]) == DRAWS + 1, "Missing paired draw")
            for rep, snapshot in enumerate(stages[market]):
                rg, mg = ([catalog[u] for u in snapshot[key]] for key in STRATEGIES)
                require(len(rg) == len(mg) == 6 and len(snapshot["bytes"]) == 2
                        and min(snapshot["bytes"]) > 0, "Invalid slot/byte count")
                for key in STRATEGIES:
                    require(len(set(snapshot[key])) == 6, "Duplicate group")
                require(overlap(rg) == overlap(mg), "Different account incidence mask counts")
                for i, (r, m) in enumerate(zip(rg, mg)):
                    require(r["stage"] == m["stage"] <= stage and r["packet"] == m["packet"]
                            and len(r["members"]) == len(m["members"]), "Unmatched slot")
                    if i % 2 == 0:
                        require(snapshot["random"][i] == snapshot["mixed"][i], "Changed random slot")
                    if stage == 2 and r["stage"] == 1:
                        for key in STRATEGIES:
                            require(snapshot[key][i] == stages["AlphaBay"][rep][key][i],
                                    "Reintroduced or revised AlphaBay survivor")


def unpack():
    raw = (WORK / "freeze.json").read_bytes()
    f = json.loads(raw)
    require(f["status"] == "FROZEN" and f["draws_per_seed"] == DRAWS and tuple(f["seeds"]) == SEEDS
            and f["cap"] == CAP and len(f["original_indices"]) == 6
            and f["fit_sha256"] == dict(zip(DOMAINS[:2], FIT_SHA)), "Incomplete freeze")
    for name, digest in f["sources"].items():
        pinned(WORK / name, digest)
    files = {record["name"]: record for record in f["manifests"]}
    require(set(files) == {"catalog.json.gz", "layouts.json.gz"}, "Incomplete frozen manifests")
    def load(name):
        data = pinned(PRIVATE / name, files[name]["sha256"])
        require(len(data) == files[name]["bytes"], "Incomplete frozen file")
        return json.loads(gzip.decompress(data))
    catalogs, layouts = load("catalog.json.gz"), load("layouts.json.gz")
    validate(catalogs, layouts)
    return sha(raw), catalogs, layouts


def events(start, end):
    _, first_ranks = start
    scores, end_ranks = end
    winners = scores.argmax(axis=1)
    require(np.array_equal(winners == np.arange(len(winners)), end_ranks == 1), "Final tie order")
    lost = np.flatnonzero((first_ranks == 1) & (end_ranks != 1))
    return [(int(q), int(winners[q])) for q in lost]


def categories(pairs, uids, catalog, stage):
    masks = {}
    for k, uid in enumerate(uids):
        g = catalog[uid]
        if g["stage"] == stage:
            for a in g["members"]:
                masks[a] = masks.get(a, 0) | (1 << k)
    result = []
    for q, w in pairs:
        qmask, wmask = masks.get(q, 0), masks.get(w, 0)
        if qmask & wmask:
            result.append("same_group")
        elif qmask and wmask:
            result.append("different_groups")
        elif wmask:
            result.append("winner_only")
        elif qmask:
            result.append("query_only")
        else:
            result.append("neither_account")
    return result


def tally(pairs, snapshot, catalog, stage):
    labels = [categories(pairs, snapshot[key], catalog, stage) for key in STRATEGIES]
    counts = Counter()
    for a, b in zip(*labels):
        hits = (a == "same_group", b == "same_group")
        counts[{(True, True): "both", (True, False): "only_random",
                (False, True): "only_mixed", (False, False): "neither"}[hits]] += 1
    paired = {k: counts[k] for k in ("both", "only_random", "only_mixed", "neither")}
    paired["random_hits"] = counts["both"] + counts["only_random"]
    paired["mixed_hits"] = counts["both"] + counts["only_mixed"]
    paired["net"] = counts["only_mixed"] - counts["only_random"]
    for key, row in zip(STRATEGIES, labels):
        for label in ("winner_only", "different_groups", "query_only", "neither_account"):
            paired[f"{key}_{label}"] = row.count(label)
    require(sum(paired[k] for k in ("both", "only_random", "only_mixed", "neither")) == len(pairs),
            "Lost-event accounting")
    return paired


def pool_ceiling(pairs, catalog, stage):
    groups = [set(g["members"]) for g in catalog.values() if g["stage"] == stage]
    return sum(any(q in g and w in g for g in groups) for q, w in pairs)


def summarize(pairs, s, market, stage, catalog, frozen, old_stage):
    historical = None
    random_draws, invalid = [], Counter()
    upper = pool_ceiling(pairs, catalog, stage)
    for rep, snapshot in enumerate(frozen):
        prior = old_stage[rep] if old_stage is not None else snapshot
        over = [max(snapshot["bytes"][k], prior["bytes"][k]) > CAP for k in range(2)]
        if any(over):
            invalid["historical" if rep == 0 else "paired"] += 1
            for k, key in enumerate(STRATEGIES):
                invalid[key] += int(over[k])
            continue
        row = tally(pairs, snapshot, catalog, stage)
        require(max(row["mixed_hits"], row["random_hits"]) <= upper, "Hit above candidate ceiling")
        if old_stage is not None and stage == 1:
            for key in STRATEGIES:
                prior_labels = categories(pairs, prior[key], catalog, stage)
                now_labels = categories(pairs, snapshot[key], catalog, stage)
                require(all(now != "same_group" or before == "same_group"
                            for now, before in zip(now_labels, prior_labels)), "Spurious retained AlphaBay hit")
        if rep == 0:
            historical = row
        else:
            random_draws.append(row)
    require(len(random_draws) + invalid["paired"] == DRAWS, "Draws lost")
    means = {k: sum(r[k] for r in random_draws)/len(random_draws) for k in random_draws[0]} if random_draws else {}
    hist = Counter(r["net"] for r in random_draws)
    diff = [row["net"] for row in random_draws]
    return dict(seed=s, market=market, losses=len(pairs), pool_ceiling=upper, historical=historical,
                paired_draws=DRAWS, valid=len(random_draws), over_cap=dict(invalid), paired_mean=means,
                net_histogram=dict(sorted(hist.items())),
                net_sd=float(np.std(diff)) if diff else None,
                net_p05_p95=[float(v) for v in np.percentile(diff, [5, 95])] if diff else None)


def judge(rows, retained):
    a = [r for r in rows if r["market"] == "AlphaBay"]
    if not all(r["valid"] == DRAWS for r in rows + retained):
        return dict(status="INCOMPLETE_PAIRED_CAPACITY", discussion_gate=None)
    exit_net = sum(r["paired_mean"]["net"] for r in a)
    survive_net = sum(r["paired_mean"]["net"] for r in retained)
    positive = sum(r["paired_mean"]["net"] > 0 for r in a)
    return dict(status="COMPLETE", alpha_exit_net=exit_net, alpha_positive_seeds=positive,
                alpha_retained_net=survive_net, discussion_gate=exit_net >= 1 and positive >= 2 and survive_net > 0)


def evaluate():
    started = time.monotonic()
    freeze_sha, catalogs, layouts = unpack()
    dst = WORK / "result"
    dst.mkdir(mode=0o700, exist_ok=False)
    pinned(TOP5 / "challengers.py", TOP5_SOURCE)
    sys.path.insert(0, str(TOP5))
    from challengers import inspect_er, read_endpoint
    old = json.loads(pinned(TOP5 / "result/summary.json", TOP5_SUMMARY))
    released = {(row["market"], row["seed"]): row for row in old["rows"]}
    selected = inspect_er()
    rows, retained = [], []
    with (dst / "access.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        for s in SEEDS:
            catalog, replay = catalogs[str(s)], layouts[str(s)]
            for stage, cells in enumerate((("R11", "R31"), ("R22", "R32")), 1):
                market, size = DOMAINS[stage-1], SIZES[stage-1]
                endpoints = []
                for cell in cells:
                    item = selected[s, cell]
                    log.write(json.dumps(dict(seed=s, market=market, cell=cell, sha256=item[1]["sha256"],
                                              freeze_sha256=freeze_sha))+"\n")
                    endpoints.append(read_endpoint(item, size))
                lost = events(*endpoints)
                anchor = released[market, s]
                require(len(lost) == anchor["lost_first"]
                        and int((endpoints[0][1] == 1).sum()) == anchor["initially_first"],
                        "Historical lost or initially-first denominator changed")
                rows.append(summarize(lost, s, market, stage, catalog, replay[market],
                                      None if stage == 1 else replay["AlphaBay"]))
                if stage == 1:
                    retained.append(summarize(lost, s, market, stage, catalog, replay["Dream"], replay["AlphaBay"]))
    require(len(rows) == 6 and len(retained) == 3, "Incomplete paired table")
    put_json(dst / "summary.json", dict(status="COMPLETE", time=datetime.now().astimezone().isoformat(),
             freeze_sha256=freeze_sha, endpoints_read=12, endpoint_metric_anchors=36, rows=rows,
             alpha_before_agora=retained, decision=judge(rows, retained), seconds=time.monotonic()-started))


if __name__ == "__main__":
    runtime()
    evaluate()
