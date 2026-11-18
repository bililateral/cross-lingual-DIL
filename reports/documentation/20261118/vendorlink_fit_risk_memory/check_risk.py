"""Hand checks for the new risk signal, constrained writing, and survival tally."""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from risk_memory import (WORK, candidate_edges, fold_winners, packet_index, predict, put_json, require,
                         risk_edges, runtime, select_slots, sources)
from evaluate_risk import cover, decide, paired_tally


def check():
    # Four correct folds and two wins by the same rival, with separate abstentions.
    winners = np.array([[0, 0, 2], [0, 0, 2], [0, 0, 1], [0, 0, 1], [1, -1, 0], [1, -1, 0]])
    expected = [[0, 1, 4, 2, 2], [2, 0, 2, 2, 2], [2, 1, 2, 2, 2]]
    require(risk_edges(winners) == expected, "Hand-counted directional weights")
    require(fold_winners(np.array([[0., 0., 0.], [.7, .7, .1], [0., .2, .5]])).tolist() == [-1, 0, 2],
            "Highest positive score, first tie, or abstention")
    views = []
    for terms in (["alpha"]*4 + ["beta"]*2, ["beta"]*6, ["gamma"]*6):
        views.append([[SimpleNamespace(title=t, description=t) for _ in range(2)] for t in terms])
    require(predict(views) == ([[0, 1, 4, 2, 2]], 0), "Current 2-versus-10 fit text must predict the rival")

    # A rival with a higher gain but extra overlap is illegal. An already-covered edge earns no reward.
    templates = [dict(packet=p, members=m) for p, m in
                 [(0,[0,1]), (1,[4,5]), (0,[2,3]), (0,[4,6]), (0,[7,8]), (0,[9,10])]]
    choices = defaultdict(list)
    for i, g in enumerate(templates):
        choices[g["packet"], len(g["members"])].append(i)
    edges = [set(), {0}, {1}, {2}, {0}, {1}]
    baseline = [0, 1, None, None, None, None]
    chosen, count = select_slots(baseline, templates, choices, edges, [3, 2, 6])
    require(chosen == [2, 1, None, None, None, None] and count == 1, "Constrained marginal gain and tie order")
    edges[0] = {1}
    require(select_slots(baseline, templates, choices, edges, [3, 2, 6]) == (baseline, 0),
            "Equal gain should retain the original group")

    # Hand layout has three repeated packets. The non-identity permutation puts 0 and 2 together in all three.
    t = [dict(packet=p, members=[a, a+1]) for p in range(3) for a in (0, 2, 4, 6)]
    perm = [0, 2, 1, 3, 4, 5, 6, 7]
    edges = candidate_edges(packet_index(t, 8), 12, perm, np.array([0, 1]), np.array([2, 4]))
    require([i for i, e in enumerate(edges) if e] == [0, 4, 8] and all(e <= {0} for e in edges),
            "Directed prediction-to-group membership mapping")
    random = dict(slots=[1, 2, 3, 5, 6, 7], perm=[None, perm])
    chosen = dict(seats=[0, 2, 3, 5, 6, 7])
    old = dict(templates=t, accounts=8, offset=0, layouts=[None, random, random])
    new = dict(ordered_risks=[[0, 2, 3, 2, 2]], layouts=[chosen, chosen])
    subsequent = (dict(layouts=[None, random, random]), dict(layouts=[chosen, dict(seats=[12, 2, 3, 5, 6, 7])]))
    reachable, pred, pool, masks = cover((old, new), subsequent, [(0, 2), (2, 0)])
    require(not reachable.any() and pred.tolist() == [True, False] and pool.all(), "Distinguish prediction and exposure")
    result = paired_tally(masks["pre_agora"], ~reachable, pred)
    require(masks["exit"][1].all() and not masks["exit"][0].any() and result["totals"]["net"] == 2
            and result["totals"]["guessed_priority"] == 1 and result["net_histogram"] == {0: 1, 2: 1},
            "Carry counts own surviving groups, with query-level deduplication")
    def row(net):
        sample = {k: dict(totals=dict(net=v)) for k, v in (("exit", net), ("pre_agora", 1))}
        return dict(market="AlphaBay", subsets={k: dict(coverage=sample) for k in ("original_miss", "all")})
    require(decide([row(500), row(500), row(0)])["discussion_gate"] and
            not decide([row(499), row(500), row(0)])["discussion_gate"] and
            not decide([row(1000), row(0), row(0)])["discussion_gate"], "Registered joint decision")
    put_json(WORK / "check.json", dict(status="PASS", inputs="synthetic only", sources=sources(),
             text_example=[0, 1, 4, 2, 2], hand_edges=expected, selected_first_slot=2,
             carry_joint_events=2, carry_predicted_events=1, carry_net_histogram=result["net_histogram"]))


if __name__ == "__main__":
    runtime()
    check()
