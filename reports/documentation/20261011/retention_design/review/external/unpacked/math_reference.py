"""Independent finite hand-example arithmetic for the static review.

No project imports, data, model, optimizer, general QP solver, or training.
Fractions certify the supplied small projections by checking all permutation
vertices (n <= 4); floats are used only for logsumexp and a closed-form n=2
finite-difference reference. This is not project-native qualification evidence.
"""
from fractions import Fraction as F
from itertools import combinations, permutations
from pathlib import Path
import json
import math
import platform
import sys


def mat(rows):
    return [[F(str(x)) for x in row] for row in rows]


def transpose(a):
    return list(map(list, zip(*a)))


def mm(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in transpose(b)] for row in a]


def sub(a, b):
    return [[x-y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def norm2(a):
    return sum(x*x for row in a for x in row)


def dot(a, b):
    return sum(x*y for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def center(a):
    means = [sum(col)/len(a) for col in transpose(a)]
    return [[x-m for x, m in zip(row, means)] for row in a]


def feasible(w):
    return (all(x >= 0 for row in w for x in row)
            and all(sum(row) == 1 for row in w)
            and all(sum(col) == 1 for col in transpose(w)))


def certify(x, y, w):
    assert feasible(w)
    z = mm(w, y)
    residual = sub(x, z)
    inequalities = [dot(residual, sub([y[i] for i in p], z))
                    for p in permutations(range(len(y)))]
    assert max(inequalities) <= 0
    return {"normalized_distance": norm2(residual)/(len(x)*len(x[0])),
            "projection": z, "maximum_projection_VI": max(inequalities),
            "permutation_vertices_checked": len(inequalities)}


def lse(v):
    values = list(map(float, v))
    m = max(values)
    return m + math.log(math.fsum(math.exp(x-m) for x in values))


def subset_means(a, b):
    return [[sum(a[i][j] for i in inds)/b for j in range(len(a[0]))]
            for inds in combinations(range(len(a)), b)]


def average(v):
    return sum(v)/len(v)


def run():
    out = {"scope": "ChatGPT workspace independent mathematical reference only",
           "project_execution": False, "project_data_access": False,
           "python": sys.version, "runtime_platform": platform.platform()}
    out["example_6_1"] = certify(mat([[-1], [1]]), mat([[-2], [2]]),
                                  mat([[.75, .25], [.25, .75]]))
    x, y = mat([[1, -1], [-1, 1]]), mat([[1, 1], [-1, -1]])
    out["example_6_2"] = certify(x, y, mat([[.5, .5], [.5, .5]]))
    source_loss = (lse([0, 1, 1])+lse([0, -1, -1]))/2
    current_loss = lse([0, 1, -1])
    assert current_loss > source_loss
    out["example_6_2"].update(source_singleton_loss=source_loss,
                              current_singleton_loss=current_loss,
                              task_legal_b2_loss_both=math.log(3))
    # A legal n=4, b=2, d=27 extension, with all candidates kept complete.
    y27 = mat([[0, a, a]+[-10]*24 for a in [1, 1, -1, -1]])
    x27 = mat([[0, a, -a]+[-10]*24 for a in [1, 1, -1, -1]])
    projection = certify(center(x27), center(y27), [[F(1, 4)]*4 for _ in range(4)])
    source_b2 = average([lse(v) for v in subset_means(y27, 2)])
    current_b2 = average([lse(v) for v in subset_means(x27, 2)])
    assert current_b2 > source_b2
    projection.update(source_b2_loss=source_b2, current_b2_loss=current_b2,
                      candidate_count=27, query_records=4)
    out["example_6_2_legal_extension"] = projection
    x, y = mat([[-1.2], [-1.2], [2.4]]), mat([[1.2], [1.2], [-2.4]])
    w = mat([[.5, 0, .5], [0, .5, .5], [.5, .5, 0]])
    out["example_6_3"] = certify(x, y, w)
    out["example_6_3"]["gradient"] = [[2*v/3 for v in row] for row in sub(x, mm(w, y))]
    # Full symmetric 28-account hand table. Eight K2 and four K3 components.
    groups = [[i, i+1] for i in range(0, 16, 2)] + [list(range(i, i+3)) for i in range(16, 28, 3)]
    owner = {q: g for g, qs in enumerate(groups) for q in qs}
    counts = [4]+[2]*27
    slots, ids = [], []
    for q, n in enumerate(counts):
        ids.append(list(range(len(slots), len(slots)+n)))
        slots.extend([q]*n)
    old_margin = list(map(F, [-3, 1, 1, 5]))
    new_margin = [F(-1, 3)]*3+[F(5)]

    def table(margin):
        result = [[F(0) for _ in slots] for _ in slots]
        for i in range(len(slots)):
            for j in range(i+1, len(slots)):
                qi, qj = slots[i], slots[j]
                score = F(1 if owner[qi] == owner[qj] else -10)
                if qi == 0 and qj == 1:
                    score = margin[ids[0].index(i)]
                elif qi == 0 and qj == 2:
                    score = F(0)
                result[i][j] = result[j][i] = score
        return result

    kt, ks = table(old_margin), table(new_margin)

    def responses(k, q):
        cand = [j for j in range(28) if j != q]
        pos = [a for a, j in enumerate(cand) if owner[j] == owner[q]]
        a = [[average([k[r][s] for s in ids[j]]) for j in cand] for r in ids[q]]
        v = [[x-average([row[p] for p in pos]) for x in row] for row in a]
        return a, v

    w4 = [[F(1, 3) if i < 3 and j < 3 else F(int(i == j == 3))
           for j in range(4)] for i in range(4)]
    assert feasible(w4)
    for q in range(28):
        at, vt = responses(kt, q)
        ast, vs = responses(ks, q)
        assert [average(c) for c in transpose(at)] == [average(c) for c in transpose(ast)]
        if q == 0:
            assert mm(w4, center(vt)) == center(vs)
        else:
            assert norm2(center(vt)) == norm2(center(vs)) == 0
    old_pairs = [v[0] for v in subset_means([[x] for x in old_margin], 2)]
    new_pairs = [v[0] for v in subset_means([[x] for x in new_margin], 2)]
    assert all(x != 0 for x in old_pairs+new_pairs)
    old_ap = average([F(1) if x > 0 else F(1, 2) for x in old_pairs])
    new_ap = average([F(1) if x > 0 else F(1, 2) for x in new_pairs])
    out["example_6_4_full_28"] = {
        "accounts": 28, "records": len(slots), "positive_edges": 20,
        "shared_W_all_27_columns": True, "all_28_R_zero": True, "D0_zero": True,
        "all_full_account_scores_equal": True,
        "query_b2_source_margins": old_pairs, "query_b2_current_margins": new_pairs,
        "source_wrong_subsets": sum(x < 0 for x in old_pairs),
        "current_wrong_subsets": sum(x < 0 for x in new_pairs),
        "query_b2_source_AP": old_ap, "query_b2_current_AP": new_ap,
        "query_only_subsampling_group_MAP_difference": (new_ap-old_ap)/28,
        "full_account_MAP_difference": F(0)}
    at, vt = responses(kt, 0)
    ast, vs = responses(ks, 0)
    risks = []
    for b in range(1, 5):
        old = average([lse(v) for v in subset_means(vt, b)])
        new = average([lse(v) for v in subset_means(vs, b)])
        assert new <= old+1e-14
        risks.append({"b": b, "source": old, "current": new})
    out["example_6_4_all_b_convex_risks"] = risks
    # Verify the finite-population second moment in the stated residual bound.
    e = center(x27)
    out["finite_population_identity"] = []
    for b in range(1, 5):
        lhs = average([sum(x*x for x in row) for row in subset_means(e, b)])
        rhs = F(4-b, 3*b)*norm2(e)/4
        assert lhs == rhs
        out["finite_population_identity"].append({"b": b, "lhs": lhs, "rhs": rhs})
    # A shared linear parameter defeats preservation of the table-block mean.
    theta, eta = F(2), F(1, 10)
    g = [-1, 1]
    parameter_gradient = g[0]+2*g[1]
    out["parameter_mean_counterexample"] = {
        "table_gradient_sum": sum(g), "parameter_gradient": parameter_gradient,
        "mean_before": F(3, 2)*theta,
        "mean_after_SGD": F(3, 2)*(theta-eta*parameter_gradient)}
    # Independent n=2 closed-form projection through the positive-relative map.
    def response_float(a, pos):
        v = [[z-sum(row[p] for p in pos)/len(pos) for z in row] for row in a]
        return [[v[i][j]-(v[0][j]+v[1][j])/2 for j in range(27)] for i in range(2)]

    def loss_and_grad(a, teacher, pos):
        x, y = response_float(a, pos), response_float(teacher, pos)
        alpha = max(-1., min(1., sum(u*v for u, v in zip(x[0], y[0]))/sum(v*v for v in y[0])))
        residual = [[u-alpha*v for u, v in zip(rx, ry)] for rx, ry in zip(x, y)]
        gx = [[2*z/54 for z in row] for row in residual]
        ga = [[z-(sum(row)/len(pos) if j in pos else 0) for j, z in enumerate(row)] for row in gx]
        return sum(z*z for row in residual for z in row)/54, ga

    pos = [0, 1]
    teacher = [[((i+2)*(j+3)%11)/5 for j in range(27)] for i in range(2)]
    a = [[3*teacher[i][j]+((i+1)*(j+1)%7)/8 for j in range(27)] for i in range(2)]
    value, grad = loss_and_grad(a, teacher, pos)
    errors = []
    for i in range(2):
        for j in range(27):
            h = 1e-6
            ap = [row[:] for row in a]
            am = [row[:] for row in a]
            ap[i][j] += h
            am[i][j] -= h
            fd = (loss_and_grad(ap, teacher, pos)[0]-loss_and_grad(am, teacher, pos)[0])/(2*h)
            errors.append(abs(fd-grad[i][j]))
    assert max(errors) < 1e-8
    assert max(abs(grad[0][j]+grad[1][j]) for j in range(27)) < 1e-14
    out["n2_positive_relative_gradient"] = {"entries": 54, "positive_candidates": 2,
        "distance": value, "maximum_finite_difference_error": max(errors),
        "scope": "closed-form hand example only; no BGE or full VJP execution"}
    out["paper_cost_arithmetic"] = {
        "max_records": 224, "unordered_pairs": 224*223//2,
        "fp32_bytes_per_table": 224*223//2*4,
        "six_tables_bytes": 6*224*223//2*4,
        "remaining_bytes_for_all_other_history_at_table_maximum": 1048576-6*224*223//2*4,
        "physical_updates": 3*288+4*3*2*288,
        "gradient_group_presentations": 3*288+2*4*3*2*288,
        "New_history_presentations": 3*2*288,
        "New_maximum_qp_calls": 3*2*288*28,
        "New_maximum_1024_iteration_assignment_calls": 3*2*288*28*1024}
    out["status"] = "MATHEMATICAL_REFERENCE_CHECKS_PASSED_NOT_NATIVE_QUALIFICATION"
    return out


if __name__ == "__main__":
    result = run()
    target = Path(__file__).with_name("math_checks.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "status": result["status"],
                      "n2_gradient_max_error": result["n2_positive_relative_gradient"]["maximum_finite_difference_error"]}))
