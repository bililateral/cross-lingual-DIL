"""Independent mathematical counterexamples: handmade scores, no formal data.

The 28-account examples have exactly eight positive pairs and four positive
triangles. All 378 scores are symmetric. Scalar risks and AP/FPR below are
computed independently before comparing the supplied production functions.
The tie/noise examples intentionally operate on the risk tensor API and are
labelled algebraic examples, not outputs asserted realizable by a BGE model.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PAIR = list(itertools.combinations(range(28), 2))
OWNER = [i // 2 if i < 16 else 8 + (i - 16) // 3 for i in range(28)]
Y = [int(OWNER[i] == OWNER[j]) for i, j in PAIR]
INDEX = {pair: k for k, pair in enumerate(PAIR)}


def scalar_risks(scores):
    risks, counts = [], []
    for q in range(28):
        pos, neg = [], []
        for k, (i, j) in enumerate(PAIR):
            if q not in (i, j):
                continue
            (pos if Y[k] else neg).append(float(scores[k]))
        all_scores = pos + neg
        vmax = max(all_scores)
        lse = vmax + math.log(math.fsum(math.exp(z - vmax) for z in all_scores))
        sp = lambda z: max(z, 0.0) + math.log1p(math.exp(-abs(z)))
        risks.append([
            lse - math.fsum(pos) / len(pos),
            math.fsum(sp(-z) for z in pos) / len(pos),
            math.fsum(sp(z) for z in neg) / len(neg),
        ])
        counts.append(len(pos))
    return risks, counts


def independent_penalty(now, before, counts):
    result = [0., 0., 0.]
    for c in range(3):
        for m in (1, 2):
            aa = sorted(now[q][c] for q in range(28) if counts[q] == m)
            bb = sorted(before[q][c] for q in range(28) if counts[q] == m)
            result[c] += math.fsum(max(0., a - b) ** 2 for a, b in zip(aa, bb)) / 28
    return result


def independent_metrics(scores):
    ap = []
    for q in range(28):
        candidates = []
        for k, (i, j) in enumerate(PAIR):
            if q in (i, j):
                candidates.append((float(scores[k]), j if i == q else i, Y[k]))
        candidates.sort(key=lambda row: (-row[0], row[1]))
        found, total = 0, 0.
        for rank, (_, _, label) in enumerate(candidates, start=1):
            if label:
                found += 1
                total += found / rank
        ap.append(total / found)
    fp = sum(int(s > 0 and not y) for s, y in zip(scores, Y))
    tp = sum(int(s > 0 and y) for s, y in zip(scores, Y))
    negative = Y.count(0)
    return {"MAP": math.fsum(ap) / 28, "AP_by_query": ap, "FP": fp,
            "negative_pairs": negative, "FPR_at_zero": fp / negative,
            "TP": tp, "positive_pairs": Y.count(1)}


def production_comparison(torch, method, before_scores, after_scores):
    # The actual supplied reference path requires float32 and is used as-is.
    import numpy as np
    reference = method.make_reference(np.asarray(before_scores, dtype=np.float32), tuple(Y))
    now = torch.tensor(after_scores, dtype=torch.float32, requires_grad=True)
    risk, count = method.query_risks(now, torch.tensor(Y))
    d = method.distribution_penalty(risk, count, reference)
    d.mean().backward()
    scalar, _ = scalar_risks(after_scores)
    error = max(abs(a-b) for row_a, row_b in zip(risk.detach().tolist(), scalar)
                for a, b in zip(row_a, row_b))
    return {"D": d.detach().tolist(), "max_abs_risk_error_float32": error,
            "retention_logit_gradient_max_abs": float(now.grad.abs().max())}


def main():
    print(json.dumps({"python": sys.version, "platform": platform.platform(),
                      "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "scope": "handmade scores and algebraic risk tensors only"}, ensure_ascii=False))
    sys.path.insert(0, str(ROOT / "input" / "scripts"))
    import torch
    import step28_risk_replay as method
    torch.set_num_threads(1)
    print(json.dumps({"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}))
    assert sorted(OWNER.count(i) for i in set(OWNER)) == [2] * 8 + [3] * 4
    assert Y.count(1) == 20 and Y.count(0) == 358

    # Strict, symmetric, label-valid example: fewer severe mistakes versus more
    # mild mistakes. The surrogate becomes better for EVERY query, but AP falls.
    before = [1. if y else -10. for y in Y]
    after = before.copy()
    matching = [(i, i + 14) for i in range(14)]
    cycle_order = [q for i in range(14) for q in (i, i + 14)]
    cycle = [tuple(sorted((cycle_order[i], cycle_order[(i + 1) % 28]))) for i in range(28)]
    for edge in matching:
        assert Y[INDEX[edge]] == 0
        before[INDEX[edge]] = 10.
    for edge in cycle:
        assert Y[INDEX[edge]] == 0
        after[INDEX[edge]] = 1.1
    assert len(set(cycle)) == 28
    rb, counts = scalar_risks(before)
    ra, ca = scalar_risks(after)
    assert counts == ca and counts.count(1) == 16 and counts.count(2) == 12
    assert all(a[0] < b[0] and a[1] == b[1] and a[2] < b[2] for a, b in zip(ra, rb))
    independent_d = independent_penalty(ra, rb, counts)
    mb, ma = independent_metrics(before), independent_metrics(after)
    assert independent_d == [0., 0., 0.]
    assert math.isclose(mb["MAP"], 15 / 28, abs_tol=1e-15)
    assert math.isclose(ma["MAP"], 31 / 84, abs_tol=1e-15)
    assert math.isclose(ma["MAP"] - mb["MAP"], -1 / 6, abs_tol=1e-15)
    assert mb["FP"] == 14 and ma["FP"] == 28
    pc = production_comparison(torch, method, before, after)
    assert pc["D"] == [0., 0., 0.]
    assert pc["retention_logit_gradient_max_abs"] == 0.
    print("COUNTEREXAMPLE_D_ZERO_MAP_FPR_REGRESSION " + json.dumps({
        "before": mb, "after": ma, "independent_D": independent_d,
        "before_risks_m1_m2": [rb[0], rb[16]], "after_risks_m1_m2": [ra[0], ra[16]],
        "production": pc, "pointwise_risk_dominance": True}, ensure_ascii=False))

    # Exact identity exchange under a label-graph automorphism. Every marginal
    # target is unchanged, although selected individual AP values exchange.
    individual_before = [2. if y else 0. for y in Y]
    individual_before[INDEX[(0, 1)]] = -4.
    individual_before[INDEX[(2, 3)]] = 4.
    perm = [2, 3, 0, 1] + list(range(4, 28))
    individual_after = [individual_before[INDEX[tuple(sorted((perm[i], perm[j])))]] for i, j in PAIR]
    assert all(Y[k] == Y[INDEX[tuple(sorted((perm[i], perm[j])))]] for k, (i, j) in enumerate(PAIR))
    ib, cc = scalar_risks(individual_before)
    ia, _ = scalar_risks(individual_after)
    assert independent_penalty(ia, ib, cc) == [0., 0., 0.]
    imb, ima = independent_metrics(individual_before), independent_metrics(individual_after)
    ic = production_comparison(torch, method, individual_before, individual_after)
    assert ic["D"] == [0., 0., 0.]
    assert ima["AP_by_query"][2] < imb["AP_by_query"][2]
    print("COUNTEREXAMPLE_INDIVIDUAL_EXCHANGE " + json.dumps({
        "before_AP_first4": imb["AP_by_query"][:4], "after_AP_first4": ima["AP_by_query"][:4],
        "MAP_before": imb["MAP"], "MAP_after": ima["MAP"], "production": ic}))

    # Algebraic exact-tie example: the sorted objective is not differentiable
    # here. Stable-sort backward selects one limiting gradient. A central
    # finite difference at the kink is NOT the correct unique-gradient oracle.
    ct = torch.tensor([1] * 16 + [2] * 12)
    ref = {"1": [[float(i), 0., 0.] for i in range(16)],
           "2": [[float(i), 0., 0.] for i in range(12)]}
    xx = torch.tensor(ref["1"] + ref["2"], dtype=torch.float64)
    xx[0, 0] = xx[1, 0] = 1.
    xx.requires_grad_(True)
    dt = method.distribution_penalty(xx, ct, ref)[0]
    dt.backward()
    direction = torch.zeros_like(xx); direction[0, 0] = 1.; direction[1, 0] = -1.
    eps = 1e-6
    with torch.no_grad():
        fplus = float(method.distribution_penalty(xx + eps * direction, ct, ref)[0])
        fminus = float(method.distribution_penalty(xx - eps * direction, ct, ref)[0])
    fzero = float(dt.detach())
    assert math.isclose(float(xx.grad[0, 0]), 2 / 28, abs_tol=1e-15)
    assert float(xx.grad[1, 0]) == 0.
    print("ALGEBRAIC_TIE_NONDIFFERENTIABILITY " + json.dumps({
        "gradient_first_two": xx.grad[:2, 0].tolist(),
        "autograd_dot_direction": float((xx.grad * direction).sum()),
        "right_derivative_estimate": (fplus - fzero) / eps,
        "left_derivative_estimate": (fzero - fminus) / eps,
        "central_difference": (fplus - fminus) / (2 * eps),
        "claim": "stable backward is one limiting gradient, not a unique derivative"}))

    # Algebraic symmetric stochastic risk noise at an unchanged deterministic
    # mean: constant references eliminate sorting, making the bias exact.
    noise = .25
    ref_noise = {"1": [[1., 1., 1.]] * 16, "2": [[1., 1., 1.]] * 12}
    clean = torch.ones((28, 3), dtype=torch.float64)
    hi = clean.clone(); hi[:, 0] += noise
    lo = clean.clone(); lo[:, 0] -= noise
    expected_noise_penalty = .5 * float(method.distribution_penalty(hi, ct, ref_noise)[0]) + .5 * float(method.distribution_penalty(lo, ct, ref_noise)[0])
    assert expected_noise_penalty == noise ** 2 / 2
    assert float(method.distribution_penalty(clean, ct, ref_noise).sum()) == 0.
    print("ALGEBRAIC_TRAIN_NOISE_BIAS " + json.dumps({
        "noise": [-noise, noise], "probabilities": [.5, .5],
        "mean_risk_equals_reference": True, "deterministic_D": 0.,
        "expected_noisy_D_rank": expected_noise_penalty,
        "expected_noisy_three_channel_mean": expected_noise_penalty / 3,
        "claim": "not a BGE/dropout magnitude measurement"}))

    # Joint dependence is absent from the API target. This risk-space example
    # is an algebraic illustration, NOT an asserted BGE-realizable score pair.
    joint_before = [[1., 3., 1.], [3., 1., 1.]] * 14
    joint_after = [[1., 1., 1.], [3., 3., 1.]] * 14
    d_joint = independent_penalty(joint_after, joint_before, [1] * 16 + [2] * 12)
    assert d_joint == [0., 0., 0.]
    print("ALGEBRAIC_MARGINALS_DO_NOT_FIX_JOINT " + json.dumps({
        "D": d_joint, "max_joint_sum_before": max(map(sum, joint_before)),
        "max_joint_sum_after": max(map(sum, joint_after)),
        "simultaneously_rank_and_positive_above_2_before": 0.,
        "simultaneously_rank_and_positive_above_2_after": .5,
        "claim": "risk-space illustration only"}))

    inp = {"note": "handmade synthetic labels; owner_index is a constructed clique identifier, not private data",
           "pairs": PAIR, "labels": Y, "clique_index": OWNER,
           "map_fpr_counterexample": {"before": before, "after": after,
                                      "reference_matching": matching, "current_cycle": cycle},
           "individual_exchange": {"before": individual_before, "after": individual_after,
                                   "permutation": perm}}
    (OUT / "handmade_inputs.json").write_text(json.dumps(inp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ALL_INDEPENDENT_MECHANISM_ASSERTIONS_PASSED")


if __name__ == "__main__":
    main()
