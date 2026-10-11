"""Independent finite mathematical reference for the 2026-10-11 static review.

Standard library only. No project imports, model, data, labels, Memory or network.
This is NOT the proposed learner, a VJP implementation, or native qualification.
All arrays below are explicitly handwritten mathematical examples.
"""
from __future__ import annotations

from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import platform
import struct
import sys
import time


def matchings(nodes):
    if not nodes:
        return [()]
    first = nodes[0]
    return [((first, other),) + rest
            for other in nodes[1:]
            for rest in matchings(tuple(i for i in nodes[1:] if i != other))]


EDGES = list(combinations(range(6), 2))
MATCHINGS = matchings(tuple(range(6)))
H = [[int(edge in matching) for edge in EDGES] for matching in MATCHINGS]
A = [[int(node in edge) for node in range(6)] for edge in EDGES]
TRUE = ((0, 1), (2, 3), (4, 5))
TRUE_INDEX = MATCHINGS.index(TRUE)


def transpose(m):
    return list(map(list, zip(*m)))


def multiply(m, n):
    nt = transpose(n)
    return [[sum(a*b for a, b in zip(row, col)) for col in nt] for row in m]


def rank_exact(m):
    rows = [[F(v) for v in row] for row in m]
    r = 0
    for col in range(len(rows[0])):
        pivot = next((i for i in range(r, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        scale = rows[r][col]
        rows[r] = [v/scale for v in rows[r]]
        for i in range(len(rows)):
            if i != r:
                scale = rows[i][col]
                rows[i] = [v-scale*w for v, w in zip(rows[i], rows[r])]
        r += 1
        if r == len(rows):
            break
    return r


INV = [[F(int(i == j), 4)-F(1, 40) for j in range(6)] for i in range(6)]
P = multiply(multiply(A, INV), transpose(A))
Q = [[F(int(i == j))-P[i][j] for j in range(15)] for i in range(15)]


def mv(m, v):
    return [sum(x*y for x, y in zip(row, v)) for row in m]


def state(q, tau=1.0):
    energy = [math.fsum(float(q[k]) for k, flag in enumerate(row) if flag)/3
              for row in H]
    logit = [v/tau for v in energy]
    top = max(logit)
    normalizer = math.log(math.fsum(math.exp(v-top) for v in logit))
    logp = [(v-top)-normalizer for v in logit]
    p = [math.exp(v) for v in logp]
    marginal = [math.fsum(p[m]*H[m][k] for m in range(15)) for k in range(15)]
    qualified = energy[TRUE_INDEX] > max(v for i, v in enumerate(energy) if i != TRUE_INDEX)
    return energy, logp, p, marginal, int(qualified)


def loss(teacher, student, tau=1.0):
    _, lt, pt, _, g = state(teacher, tau)
    _, ls, _, _, _ = state(student, tau)
    return g*tau*tau*math.fsum(p*(x-y) for p, x, y in zip(pt, lt, ls))


def analytic_gradient(teacher, student, tau=1.0):
    _, _, _, mt, g = state(teacher, tau)
    _, _, _, ms, _ = state(student, tau)
    return [g*tau*(s-t)/3 for s, t in zip(ms, mt)]


def covariance(p):
    mu = [math.fsum(p[m]*H[m][k] for m in range(15)) for k in range(15)]
    return [[math.fsum(p[m]*(H[m][i]-mu[i])*(H[m][j]-mu[j])
                       for m in range(15)) for j in range(15)] for i in range(15)]


def norm2(v):
    return sum(x*x for x in v)


def macro_ap(scores, components):
    """Handwritten examples have no positive-negative ties above positive ranks."""
    n = len(scores)
    ids = [next(k for k, c in enumerate(components) if i in c) for i in range(n)]
    values = []
    for i in range(n):
        order = sorted((j for j in range(n) if i != j), key=lambda j: (-scores[i][j], j))
        hits = 0
        precision = F(0)
        for rank, j in enumerate(order, 1):
            if ids[i] == ids[j]:
                hits += 1
                precision += F(hits, rank)
        values.append(precision/hits)
    return sum(values)/n


def main():
    started = time.perf_counter()
    checks = []
    details = {}

    def check(name, condition, result):
        checks.append({"name": name, "passed": bool(condition), "result": result})
        if not condition:
            raise AssertionError(name)

    check("perfect_matchings", len(MATCHINGS) == len(set(MATCHINGS)) == 15
          and all(sorted(i for edge in m for i in edge) == list(range(6)) for m in MATCHINGS)
          and all(sum(row[k] for row in H) == 3 for k in range(15)),
          {"count": len(MATCHINGS), "matchings_one_based": [[[i+1,j+1] for i,j in m] for m in MATCHINGS],
           "each_edge_occurrences": 3})
    ata = multiply(transpose(A), A)
    ranks = {"A": rank_exact(A), "H": rank_exact(H),
             "matching_differences": rank_exact([[v-w for v,w in zip(row,H[0])] for row in H[1:]]),
             "quotient_projection": rank_exact(Q)}
    check("exact_quotient_linear_algebra", ranks == {"A":6,"H":10,"matching_differences":9,"quotient_projection":9}
          and ata == [[4*int(i==j)+1 for j in range(6)] for i in range(6)]
          and multiply(Q, Q) == Q and multiply(transpose(A), Q) == [[F(0)]*15 for _ in range(6)], ranks)

    teacher = [5.0 if edge in TRUE else (((edge[0]*7+edge[1]*11)%13)-6)/20 for edge in EDGES]
    student = [t+(k-7)*0.031 for k,t in enumerate(teacher)]
    tau = 1.3
    grad = analytic_gradient(teacher, student, tau)
    eps = 1e-5
    finite = []
    for k in range(15):
        plus, minus = student.copy(), student.copy()
        plus[k] += eps
        minus[k] -= eps
        finite.append((loss(teacher, plus, tau)-loss(teacher, minus, tau))/(2*eps))
    error = max(abs(x-y) for x,y in zip(grad, finite))
    node_sums = mv(transpose(A), grad)
    check("gradient_finite_difference", state(teacher,tau)[-1] == 1 and error < 2e-10,
          {"tau": tau,"central_difference_step":eps,"max_absolute_error":error,
           "analytic":grad,"finite_difference":finite})
    biases = [0.75,-1.5,2.0,0.125,-0.5,1.25]
    shifted = [v+biases[i]+biases[j] for v,(i,j) in zip(student,EDGES)]
    pt, ps = state(student,tau)[2], state(shifted,tau)[2]
    inv_error = max(abs(x-y) for x,y in zip(pt, ps))
    check("node_invariance_and_gradient_node_sums", inv_error < 1e-14 and max(map(abs,node_sums)) < 1e-14,
          {"max_probability_difference":inv_error,"gradient_node_sums":node_sums,
           "loss_before":loss(teacher,student,tau),"loss_after":loss(teacher,shifted,tau)})

    residual = [F((k*7)%19-9,5) for k in range(15)]
    sums = mv(transpose(A), residual)
    ucenter = [v-(sums[i]+sums[j])/4+sum(sums)/20 for v,(i,j) in zip(residual,EDGES)]
    check("projection_is_u_centering", ucenter == mv(Q,residual),
          {"exact_equality":True,"formula":"e_ij-(s_i+s_j)/4+sum_i(s_i)/20"})

    uniform_cov = [[F(sum(H[m][i]*H[m][j] for m in range(15)),15)-F(1,25)
                    for j in range(15)] for i in range(15)]
    check("uniform_covariance_exact", uniform_cov == [[F(4,15)*v for v in row] for row in Q],
          {"covariance":"(4/15)Q", "scaled_KL_quadratic":"(2/135)||Qe||^2",
           "ratio_to_quotient_MSE":"2/9", "qualification_note":"Exactly uniform has g=0; compare g=1 limit as true-pair margin tends to 0+."})
    direction = [float(v) for v in mv(Q,residual)]
    cov = covariance(state(teacher,tau)[2])
    quadratic = sum(v*w for v,w in zip(direction,mv(cov,direction)))/18
    local = []
    for step in (0.1,0.01,0.001):
        value = loss(teacher,[t+step*v for t,v in zip(teacher,direction)],tau)
        local.append({"step":step,"loss":value,"quadratic_prediction":quadratic*step*step,
                      "ratio":value/(quadratic*step*step)})
    check("local_second_order", abs(local[-1]["ratio"]-1) < 0.002,
          {"qualified":1,"tau":tau,"evaluations":local})

    simple = [1.0 if edge in TRUE else 0.0 for edge in EDGES]
    altered = simple.copy()
    altered[EDGES.index((0,1))] = 0.0
    components = [list(range(i,i+2)) for i in range(0,16,2)] + [list(range(i,i+3)) for i in range(16,28,3)]
    ids = [next(k for k,c in enumerate(components) if i in c) for i in range(28)]
    def entry(account_i, record_i, account_j, record_j, changed=False):
        v = F(int(ids[account_i] == ids[account_j]))
        if changed and {account_i,account_j} == {0,1}:
            v += -1 if record_i < 2 and record_j < 2 else F(1,3)
        return v
    means_before, means_after = [], []
    for i,j in combinations(range(28),2):
        means_before.append(sum(entry(i,r,j,s) for r in range(4) for s in range(4))/16)
        means_after.append(sum(entry(i,r,j,s,True) for r in range(4) for s in range(4))/16)
    d0 = sum((a-b)**2 for a,b in zip(means_before,means_after))/378
    r_view = loss(simple,altered)
    check("legal_subbag_D0_zero_R_positive", d0 == 0 and r_view > 0 and state(simple)[-1] == 1,
          {"accounts":28,"controller_sizes":[len(c) for c in components],"records_per_account":4,
           "complete_records":112,"source_positive_block":1,"selected_2x2_change":-1,
           "other_12_change":"1/3","D0_exact":str(d0),"view_R":r_view,
           "whole_R_lower_bound_if_this_view_sampled":r_view/16,
           "weighted_extra_lower_bound":r_view/64,
           "scope":"One legal view in a lawful 16-view realization; not the project's sampled views."})

    bias = [F(0)]*28
    bias[2] = F(2)
    before = [[F(int(i!=j and ids[i]==ids[j])) for j in range(28)] for i in range(28)]
    after = [[before[i][j]+bias[i]+bias[j] if i!=j else F(0) for j in range(28)] for i in range(28)]
    after_six = [float(after[i][j]) for i,j in EDGES]
    probability_error = max(abs(x-y) for x,y in zip(state(simple)[2],state(after_six)[2]))
    ap0, ap1 = macro_ap(before,components), macro_ap(after,components)
    ranking_d0 = sum((before[i][j]-after[i][j])**2 for i,j in combinations(range(28),2))/378
    check("zero_pairing_loss_can_lower_ordinary_MAP", probability_error < 1e-14 and ap0 == 1 and ap1 == F(4,7),
          {"MAP_before":str(ap0),"MAP_after":str(ap1),"MAP_after_decimal":float(ap1),
           "D0_after":"2/7","D0_computed_exact":str(ranking_d0),"six_view_probability_max_error":probability_error,
           "scope":"R is invariant; D0 and full S are NOT zero. This is not a counterexample to minimizing the full objective."})

    bad_teacher = [0.0]*15
    for edge,value in zip(TRUE,(2.0,2.0,-1.0)):
        bad_teacher[EDGES.index(edge)] = value
    energy = state(bad_teacher)[0]
    check("global_qualification_does_not_imply_query_correctness", state(bad_teacher)[-1] == 1,
          {"positive_scores":[2,2,-1],"all_negative_scores":0,"g":1,
           "true_energy":energy[TRUE_INDEX],"best_wrong_energy":max(v for i,v in enumerate(energy) if i!=TRUE_INDEX),
           "queries_5_6_true_partner_rank_among_five":5})
    sharp = [2*v for v in simple]
    check("soft_KL_penalizes_sharpening", loss(simple,sharp) > 0,
          {"soft_R":loss(simple,sharp),"hard_CE_teacher":-state(simple)[1][TRUE_INDEX],
           "hard_CE_sharper":-state(sharp)[1][TRUE_INDEX]})
    style_residual = mv(Q,[F(int(edge in TRUE)) for edge in EDGES])
    check("nonadditive_style_agreement_survives_projection", norm2(style_residual) == F(12,5),
          {"styles":[0,0,1,1,2,2],"score":"1{style_i=style_j}","quotient_squared_norm":"12/5",
           "qualified_when_styles_align_with_training_controllers":1,
           "scope":"Allowed illustrative coincidence, not a frequency or BGE measurement."})
    off = [0.0]*15
    check("zero_gate_and_equal_table", state(off)[-1] == 0 and loss(off,student) == 0 and loss(teacher,teacher,tau) == 0,
          {"all_tied_source_g":0,"g0_view_loss":loss(off,student),"equal_table_view_loss":loss(teacher,teacher,tau),
           "scope":"Table function only; no claim of native bitwise S equivalence."})
    details["conditional_soft_hard_limit"] = []
    fixed_student = [v*0.7 for v in simple]
    hard_gradient = [(a-b)/3 for a,b in zip(state(fixed_student)[3],H[TRUE_INDEX])]
    for strength in (1,10,30):
        t = [strength*v for v in simple]
        soft_gradient = analytic_gradient(t,fixed_student)
        details["conditional_soft_hard_limit"].append({"teacher_positive_score":strength,
            "p_true":state(t)[2][TRUE_INDEX],
            "max_gradient_difference_from_hard":max(abs(a-b) for a,b in zip(soft_gradient,hard_gradient))})
    details["counts"] = {"shared_A_updates":288,"B_updates_per_arm":288,"B_arms":4,
        "physical_updates":288+4*288,"gradient_group_presentations":288+4*288*2,
        "views_per_history_group":16,"controller_occurrences_per_presentation":4,
        "expected_K2_true_edge_occurrences":"4","expected_K3_true_edge_occurrences":"4/3",
        "expected_K3_account_occurrences":"8/3","two_record_terms_per_view":60,
        "record_terms_per_side_per_history_presentation":960,
        "matching_energies_per_side_per_history_presentation":240,
        "maximum_source_FP32_table_bytes_for_six_groups":6*(224*223//2)*4}
    # Clarifies FP64 is downstream of existing FP32 quantization, not recovery of lost bits.
    x = 1.0+2**-25
    stored = struct.unpack('<f',struct.pack('<f',x))[0]
    details["fp32_then_fp64"] = {"original_real_reference":x,"stored_fp32":stored,
                                  "promoted_fp64":float(stored),"precision_recovered":False}
    result = {"scope":"WEB_STATIC_REVIEW_MATHEMATICS_ONLY_NOT_PROJECT_NATIVE_QUALIFICATION",
        "input_kind":"handwritten finite mathematical examples only",
        "project_modules_imported":[],"model_data_Memory_access":False,
        "python":sys.version,"platform":platform.platform(),"dependencies":"Python standard library only",
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds":time.perf_counter()-started,"checks":checks,"details":details,
        "passed":all(row['passed'] for row in checks)}
    destination = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('math_results.json')
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({"passed":result['passed'],"checks":len(checks),"output":str(destination),
                      "elapsed_seconds":result['elapsed_seconds']},ensure_ascii=False))


if __name__ == '__main__':
    main()
