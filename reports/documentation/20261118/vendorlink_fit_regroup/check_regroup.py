"""Independent hand examples for the new matching, membership, sizing and coverage."""
import random
import time

import numpy as np

from regroup import (DOMAINS, WORK, directed_layout, fit_pairs, make_pool, materialize,
                     memory_size, now, originals, put_json, require, runtime, sources, view_table)
from evaluate_regroup import hits, summarize


def pairing():
    edges = {(0, 1): 6, (1, 0): 6, (1, 2): 5, (2, 3): 4}
    ranked = fit_pairs(edges, random.Random(13))
    require(ranked == [(0, 1), (2, 3)], "Bidirectional priority and disjoint matching")
    seats = [(0, 4), (1, 3)]
    for seed in (9, 10):
        p, placed = directed_layout(5, ranked, seats, random.Random(seed))
        together = {tuple(sorted((p[i], p[j]))) for i, j in seats}
        require(placed == 2 and sorted(p) == list(range(5)) and together == set(ranked), "Rivals in seats")
    no_rivals, empty = directed_layout(5, [], seats, random.Random(10))
    require(not empty and sorted(no_rivals) == list(range(5)), "Unused accounts retained")
    return dict(matched_pairs=2, interfering_edge_skipped=True, uncovered_account_filled=True)


def actual_records():
    groups_for, _, Memory = originals()
    from step29_vendorlink_data import Group
    n = 21
    accounts = [dict(account_uid=f"acct-{a:02d}", records=[
        dict(uid=f"ad-{a}-{r}", title=f"account {a}", description=f"record {r}") for r in range(12)])
        for a in range(n)]
    pool, _, _ = make_pool(accounts, 51, 1, groups_for)
    table = view_table(pool, n)
    identity = list(range(n))
    rebuilt = [materialize(pool, table, i, identity, 51) for i in range(len(pool))]
    changed = [(x+1) % n for x in identity]
    displaced = [materialize(pool, table, i, changed, 51) for i in range(len(pool))]
    lookup = {(g.packet, a): g.row["group"]["views"][2*i:2*i+2]
              for g in pool for i, a in enumerate(g.members)}
    for original, same, moved in zip(pool, rebuilt, displaced):
        require(original.uid == same.uid and original.row["group"] == same.row["group"]
                and original.uid != moved.uid, "Content ID or identity permuted incorrectly")
        group = Group.from_dict(moved.row["group"])
        for j, original_position in enumerate(original.members):
            a = changed[original_position]
            expected = lookup[original.packet, a]
            actual_views = group.views[2*j:2*j+2]
            require(all(v.identity == accounts[a]["account_uid"] for v in actual_views)
                    and [[(r.uid, r.title, r.description) for r in v.records] for v in actual_views]
                    == [[(r["uid"], r["title"], r["description"]) for r in v["records"]] for v in expected],
                    "Moved label without its four real records")
    rng = random.Random(51).getstate()
    require(memory_size(pool[:6], len(pool), rng, Memory) == memory_size(rebuilt[:6], len(pool), rng, Memory),
            "Identity permutation changed original serialized byte size")
    return dict(accounts=n, groups=len(pool), distinct_records=n*12, identity_bytes_match=True,
                displaced_views_and_text_checked=True)


def coverage():
    # Three original packets each split four positions as {0,1}/{2,3}.
    groups = [dict(packet=i//2, members=[0, 1] if i%2 == 0 else [2, 3]) for i in range(6)]
    perm = [0, 2, 1, 3]
    rows = [dict(perm=[list(range(4)), perm], slots=[0, 1]), dict(perm=[perm, list(range(4))], slots=[2, 3])]
    data = dict(accounts=4, offset=0, templates=groups, layouts=rows)
    after = dict(layouts=[dict(slots=[0, 7]), dict(slots=[6, 3])])
    rivals = [(0, 2), (1, 3), (0, 1)]
    masks = hits(data, after, rivals)
    for rep, row in enumerate(rows):
        for arm, layout in enumerate([list(range(4))] + row["perm"]):
            for layer, positions in [("pool", range(6)), ("exit", row["slots"]),
                                     ("pre_agora", after["layouts"][rep]["slots"])]:
                actual = [{layout[x] for x in groups[k]["members"]} for k in positions if k < 6]
                manual = [any(q in g and w in g for g in actual) for q, w in rivals]
                require(manual == masks[layer][arm][rep].tolist(), "Coverage or carry disagrees with enumeration")
    subset = ~masks["pool"][0][0]
    out = summarize(masks["exit"], subset)
    require(subset.tolist() == [True, True, False] and out["totals"]["only_reshuffled"] == 2
            and out["totals"]["net"] == -2 and out["historical_schedule"]["only_directed"] == 2
            and masks["pre_agora"][2][0].tolist() == [True, False, False], "Paired missed/lost partition")
    require(summarize(masks["pool"], np.zeros(3, dtype=bool))["totals"]["net"] == 0, "Empty stratum")
    return dict(original_misses=2, paired_net=-2, hand_carry_hits=1, enumerated_cells=18)


if __name__ == "__main__":
    runtime()
    started = time.monotonic()
    put_json(WORK / "check.json", dict(status="PASS", time=now(), pairing=pairing(),
             records=actual_records(), coverage=coverage(), sources=sources(), seconds=time.monotonic()-started))
