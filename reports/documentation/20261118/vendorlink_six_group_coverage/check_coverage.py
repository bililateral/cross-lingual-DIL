"""Hand-tallied fit selection, reservoir, byte-budget and coverage checks."""
from datetime import datetime
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from select_groups import (CAP, DOMAINS, DRAWS, WORK, Slot, cosine_field, fit_weights, group_edges,
                           make_pool, memory_size, mixed_slots, nominate, originals, put_json,
                           random_slots, require, runtime, sha)
from evaluate import categories, events, overlap, pool_ceiling, summarize, tally


def hand_selection():
    def slot(name, members):
        return Slot(name, 1, 0, tuple(members), {})

    baseline = [slot(str(k), (2*k, 2*k+1)) for k in range(6)]
    extra = [slot("gain", (12, 13)), slot("overlap", (9, 12))]
    pool = baseline + extra
    weights = {(12, 13): 4, (9, 12): 6}
    choices = {(0, 2): sorted(pool, key=lambda g: g.uid)}
    out = mixed_slots(baseline, [], 1, choices, group_edges(pool, weights))
    expected = ["0", "gain", "2", "3", "4", "5"]
    require([g.uid for g in out] == expected, "Text edge/overlap selection failed")
    require(mixed_slots(baseline, [], 1, choices, group_edges(pool, {})) == baseline, "No gain fallback")
    catalog = {g.uid: dict(members=g.members, stage=1) for g in pool}
    require(overlap([catalog[g.uid] for g in baseline]) == overlap([catalog[g.uid] for g in out]),
            "Changed account incidence")
    return expected


def hand_text():
    sample = np.ones((8, 8))
    require([b for a, b in nominate(sample) if a == 0] == [1, 2, 3, 4, 5], "Tied five")
    require(nominate(np.zeros((8, 8))) == [], "Positive-only nomination")
    sims = cosine_field(["apple", "banana"], ["apple apple", "unseen"])
    require(np.allclose(sims, [[1, 0], [0, 0]]) and not cosine_field(["", ""], ["apple"]).any(),
            "Reference-only vocabulary or normalization")
    def folds(word):
        return [[SimpleNamespace(title=word, description="")] * 2 for _ in range(6)]
    require(fit_weights([folds("apple"), folds("apple"), folds("banana")]) ==
            {(0, 1): 6, (1, 0): 6}, "Six folded nominations")


def synthetic_reservoir():
    groups_for, seed_for, Memory = originals()
    history, count = [], 0
    seed = seed_for(51, DOMAINS, "reservoir")
    rng = random.Random(seed)
    actual = Memory(seed, SimpleNamespace(memory_groups=6, memory_bytes=CAP))
    class Dummy:
        def teacher(self, group, check):
            records = sum(len(v.records) for v in group.views)
            return torch.ones(records*(records-1)//2, dtype=torch.float32)
    dims = []
    for stage, n in ((1, 64), (2, 32)):
        market = DOMAINS[stage-1]
        accounts = [dict(account_uid=f"{market}-{i:03}", records=[
            dict(uid=f"{market}-{i:03}-{j:02}", title=f"item {i}", description=f"body {j}") for j in range(12)])
                    for i in range(n)]
        pool, views, registry = make_pool(accounts, 51, stage, groups_for)
        require(len(views) == n and all(len(a) == 6 for a in views) and len(registry) == len(pool),
                "Original multi-record views")
        original = groups_for(accounts, 51, market, 0)
        actual.retain(original, stage, Dummy())
        history, count = random_slots(history, sorted(pool, key=lambda g: g.uid), rng, count)
        size = memory_size(history, count, rng.getstate(), Memory)
        require([g.uid for g in history] == [r["group"]["uid"] for r in actual.entries]
                and count == actual.seen and rng.getstate() == actual.rng.getstate()
                and size == len(actual.bytes()), "Two-stage or original serializer disagreement")
        dims.append(dict(stage=stage, seen=count, bytes=size, original_receipt_equivalent=True))
    return dims


def hand_evaluate():
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    catalog = {str(i): dict(stage=1, members=members)
               for i, members in enumerate(((0, 1), (2, 3), (4,), (5,), (6,), (8, 9), (4, 5)))}
    baseline = ["0", "1", "2", "3", "4", "5"]
    mixed = ["0", "6", "2", "3", "4", "5"]
    snap = dict(random=baseline, mixed=mixed, bytes=[250, 258])
    cens = tally(pairs, snap, catalog, 1)
    require([cens[k] for k in ("both", "only_random", "only_mixed", "neither", "net")] ==
            [1, 1, 1, 1, 0] and cens["random_different_groups"] == 1, "Paired tally")
    require(categories([(9, 7), (7, 9)], baseline, catalog, 1) == ["query_only", "winner_only"],
            "Winner-only vs query-only")
    require(pool_ceiling(pairs, catalog, 1) == 3, "Candidate ceiling")
    catalog["dream"] = dict(stage=2, members=(4, 5))
    after = dict(random=["dream", *baseline[1:]], mixed=["dream", *mixed[1:]], bytes=[250, 258])
    prior = [snap] * (DRAWS + 1)
    final = [after] * (DRAWS + 1)
    final[-1] = {**after, "bytes": [CAP + 1, 258]}
    res = summarize(pairs, 51, "AlphaBay", 1, catalog, final, prior)
    require(res["valid"] == DRAWS-1 and res["over_cap"]["random"] == 1 and
            res["paired_mean"]["random_hits"] == res["paired_mean"]["mixed_hits"] == 1,
            "Eviction or capacity conditioning")
    begin = (None, np.array([1, 1, 2]))
    end = (np.array([[1, 2, 2], [0, 3, 1], [4, 0, 1]]), np.array([3, 1, 2]))
    require(events(begin, end) == [(0, 1)], "Original final tie and lost-event denominator")
    return {k: cens[k] for k in ("both", "only_random", "only_mixed", "neither")}


if __name__ == "__main__":
    runtime()
    start = time.monotonic()
    hand_text()
    chosen = hand_selection()
    dims = synthetic_reservoir()
    table = hand_evaluate()
    names = ("select_groups.py", "evaluate.py", "check_coverage.py", "scope.zh.md")
    put_json(WORK / "check.json", dict(status="PASS", sources={n: sha((WORK/n).read_bytes()) for n in names},
             text="six folds, positive-only, stable ties, held-out vocabulary", selected=chosen,
             original_reservoir=dims, hand_tally=table, time=datetime.now().astimezone().isoformat(),
             seconds=time.monotonic()-start, real_role_reads=0, model_loads=0))
