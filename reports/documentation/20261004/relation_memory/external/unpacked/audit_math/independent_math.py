"""Independent design audit; NumPy only; no project data, model or server.

The examples exercise the proposed equations, not a supplied implementation.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "upload" / "SELLER_ALIAS_RELATION_MEMORY.zh.md"
rng = np.random.default_rng(20261004)


def group_spec():
    owners = np.array([k for k in range(4) for _ in range(3)] +
                      [k for k in range(4, 12) for _ in range(2)])
    edges = [(i, j) for i in range(28) for j in range(i + 1, 28)]
    edge_index = {e: k for k, e in enumerate(edges)}
    labels = np.array([owners[i] == owners[j] for i, j in edges], dtype=int)
    comparisons = []
    queries = []
    for q in range(28):
        positives = [i for i in range(28) if i != q and owners[i] == owners[q]]
        negatives = [i for i in range(28) if owners[i] != owners[q]]
        pairs = [(edge_index[tuple(sorted((q, p)))],
                  edge_index[tuple(sorted((q, n)))]) for p in positives for n in negatives]
        comparisons += pairs
        queries.append(pairs)
    return owners, edges, edge_index, labels, queries


owners, edges, edge_index, labels, queries = group_spec()
targets = 2 * labels - 1


def explicit_group(Z, w):
    scores = Z.T @ w
    identification = np.mean((scores - targets) ** 2)
    query_losses = [np.mean([(scores[p] - scores[n] - 2) ** 2 for p, n in qp])
                    for qp in queries]
    return float(identification + np.mean(query_losses))


def explicit_w_gradient(Z, w):
    scores = Z.T @ w
    grad = 2 * (Z @ (scores - targets)) / 378
    for qp in queries:
        for p, n in qp:
            grad += 2 * (scores[p] - scores[n] - 2) * (Z[:, p] - Z[:, n]) / (28 * len(qp))
    return grad


def statistics(Z):
    H = Z @ Z.T / 378
    b = Z @ targets / 378
    for qp in queries:
        D = np.stack([Z[:, p] - Z[:, n] for p, n in qp], axis=1)
        H += D @ D.T / (28 * len(qp))
        b += 2 * np.mean(D, axis=1) / 28
    return H, b, 5.0


def q_value(v, H, b, c, N=1):
    return float((v @ H @ v - 2 * b @ v + c) / N)


def map_solve(X, Y, eps=0.001):
    m = X.shape[1]
    d = X.shape[0]
    K = X @ X.T / m + eps * np.eye(d)
    R = Y @ X.T / m + eps * np.eye(d)
    T = np.linalg.solve(K.T, R.T).T
    return T, K


def score_solve(X, Y, w, eps=0.001):
    m = X.shape[1]
    K = X @ X.T / m + eps * np.eye(X.shape[0])
    return np.linalg.solve(K, X @ (Y.T @ w) / m + eps * w)


def central_gradient(fn, x, h=1e-5, selected=None):
    out = np.zeros_like(x)
    if selected is None:
        selected = list(np.ndindex(x.shape))
    for index in selected:
        hi = x.copy()
        lo = x.copy()
        hi[index] += h
        lo[index] -= h
        out[index] = (fn(hi) - fn(lo)) / (2 * h)
    return out


report = {"environment": {"python": sys.version, "numpy": np.__version__,
                          "platform": platform.platform(), "torch_used": False,
                          "official_data_accessed": False, "project_server_used": False}}
raw = SOURCE.read_bytes()
report["source_identity"] = {"bytes": len(raw), "lines": len(raw.decode("utf-8").splitlines()),
                             "sha256": hashlib.sha256(raw).hexdigest()}

# Actual 28-account complete group, with 4 triples and 8 pairs.
Z = np.vstack([np.tanh(rng.normal(size=(31, 378))), np.ones(378)])
w = rng.normal(size=32)
H, b, c = statistics(Z)
L = explicit_group(Z, w)
Q = q_value(w, H, b, c)
gw = 2 * (H @ w - b)
gw_explicit = explicit_w_gradient(Z, w)
gw_fd = central_gradient(lambda ww: explicit_group(Z, ww), w)
report["full_group_reduction"] = {
    "accounts": 28, "edges": 378, "positive_edges": int(labels.sum()),
    "query_comparisons": sum(len(qp) for qp in queries),
    "query_comparison_counts": [len(qp) for qp in queries],
    "explicit_loss": L, "quadratic_loss": Q, "loss_absolute_error": abs(L-Q),
    "gradient_max_error_explicit_vs_statistics": float(np.max(np.abs(gw-gw_explicit))),
    "gradient_max_error_finite_difference": float(np.max(np.abs(gw-gw_fd))),
    "H_last_last": float(H[-1, -1]), "b_last": float(b[-1]),
    "expected_b_last": -169/189, "c": c,
    "min_eigenvalue_H": float(np.linalg.eigvalsh(H).min()),
}
assert abs(L-Q) < 1e-10
assert np.max(np.abs(gw-gw_explicit)) < 1e-10
assert np.max(np.abs(gw-gw_fd)) < 1e-7

# Complete-square and augmented PSD properties of a genuine full-group statistic.
mu = np.linalg.pinv(H) @ b
residual = c - b @ mu
Q_centered = (w-mu) @ H @ (w-mu) + residual
G = np.block([[H, -b[:, None]], [-b[None, :], np.array([[c]])]])
report["supervised_quadratic_center"] = {
    "range_H_residual_norm": float(np.linalg.norm(H @ mu - b)),
    "minimum_quadratic_value": float(residual),
    "complete_square_error": float(abs(Q_centered - Q)),
    "min_eigenvalue_augmented_gram": float(np.linalg.eigvalsh(G).min()),
}

# Ridge drift, including the constant row, and exact equation (9) equivalence.
X = np.vstack([np.tanh(rng.normal(size=(31, 378))), np.ones(378)])
input_features = rng.normal(size=(5, 378))
theta = rng.normal(scale=0.2, size=(31, 5))
Y = np.vstack([np.tanh(theta @ input_features), np.ones(378)])
w = rng.normal(scale=0.2, size=32)
eps = 0.001
T, K = map_solve(X, Y, eps)
v_matrix = T.T @ w
v_score = score_solve(X, Y, w, eps)
a = 2 * (H @ v_score - b)
analytic_w = T @ a
analytic_Y = np.outer(w, X.T @ np.linalg.solve(K, a)) / 378
loss_Y = lambda YY: q_value(score_solve(X, YY, w), H, b, c)
loss_w = lambda ww: q_value(score_solve(X, Y, ww), H, b, c)
Y_indices = [(i, j) for i, j in zip(rng.integers(0, 32, 60), rng.integers(0, 378, 60))]
fd_Y = central_gradient(loss_Y, Y, selected=Y_indices)
fd_w = central_gradient(loss_w, w)
analytic_theta = (analytic_Y[:-1] * (1 - Y[:-1]**2)) @ input_features.T
loss_theta = lambda th: loss_Y(np.vstack([np.tanh(th @ input_features), np.ones(378)]))
fd_theta = central_gradient(loss_theta, theta)
T_identity, _ = map_solve(X, X, eps)
direct_term_gradient = eps * np.linalg.solve(K, a)
report["equations_6_9_10"] = {
    "v_max_error_explicit_T_vs_scores": float(np.max(np.abs(v_matrix-v_score))),
    "identity_map_max_error": float(np.max(np.abs(T_identity-np.eye(32)))),
    "constant_row_max_error": float(np.max(np.abs(T[-1]-np.eye(32)[-1]))),
    "w_gradient_max_finite_difference_error": float(np.max(np.abs(analytic_w-fd_w))),
    "Y_gradient_max_selected_finite_difference_error": float(max(abs(analytic_Y[ix]-fd_Y[ix]) for ix in Y_indices)),
    "theta_gradient_max_finite_difference_error": float(np.max(np.abs(analytic_theta-fd_theta))),
    "theta_gradient_norm": float(np.linalg.norm(analytic_theta)),
    "missing_direct_epsilon_w_gradient_norm": float(np.linalg.norm(direct_term_gradient)),
    "X_rank": int(np.linalg.matrix_rank(X)),
    "score_nullspace_dimension": 378-int(np.linalg.matrix_rank(X)),
}
assert np.max(np.abs(v_matrix-v_score)) < 1e-10
assert np.max(np.abs(analytic_w-fd_w)) < 1e-7
assert max(abs(analytic_Y[ix]-fd_Y[ix]) for ix in Y_indices) < 1e-7
assert np.max(np.abs(analytic_theta-fd_theta)) < 1e-7

# Exact ridge-bias formula for bounded linear drift.
A = np.eye(32)
A[:-1, :-1] *= 0.8
A[:-1, -1] = 0.05
YA = A @ X
TA, KA = map_solve(X, YA, eps)
T_predicted = A + (np.eye(32)-A) @ (eps*np.linalg.inv(KA))
report["ridge_bias"] = {
    "formula_7_max_error": float(np.max(np.abs(TA-T_predicted))),
    "T_minus_A_frobenius": float(np.linalg.norm(TA-A)),
    "max_absolute_nonconstant_Y": float(np.abs(YA[:-1]).max()),
}

# Strong cache-internal blind direction: actual 378-edge group, d=2.
# e0 is negative to q=0; e1 is a positive to q=0; e2 is an unrelated positive.
e0 = edge_index[(0, 3)]
e1 = edge_index[(0, 1)]
e2 = edge_index[(4, 5)]
X2 = np.vstack([0.5*targets.astype(float), np.ones(378)])
X2[0, e1] = 0.0
delta_p = np.zeros(378)
delta_p[[e0, e1, e2]] = [0.2, -0.4, 0.2]
Y2 = X2.copy()
Y2[0] += delta_p
w2 = np.array([1.0, 0.0])
H2, b2, c2 = statistics(X2)
T2, K2 = map_solve(X2, Y2, eps)
before = q_value(w2, H2, b2, c2)
after_surrogate = q_value(T2.T @ w2, H2, b2, c2)
actual_before = explicit_group(X2, w2)
actual_after = explicit_group(Y2, w2)
report["cache_internal_blind_direction_full_group"] = {
    "dimension": 2, "edges": 378,
    "changed_edges": [[0, 3], [0, 1], [4, 5]],
    "changed_edge_labels": labels[[e0, e1, e2]].tolist(),
    "changed_scores_before": (X2.T@w2)[[e0, e1, e2]].tolist(),
    "changed_scores_after": (Y2.T@w2)[[e0, e1, e2]].tolist(),
    "X_delta_p": (X2@delta_p).tolist(),
    "mapping_change_frobenius": float(np.linalg.norm(T2-np.eye(2))),
    "old_surrogate_before": before, "old_surrogate_after": after_surrogate,
    "actual_same_cache_loss_before": actual_before,
    "actual_same_cache_loss_after": actual_after,
    "query_0_selected_positive_minus_negative_before": float(X2[0, e1]-X2[0, e0]),
    "query_0_selected_positive_minus_negative_after": float(Y2[0, e1]-Y2[0, e0]),
    "max_absolute_nonconstant_Y": float(np.abs(Y2[0]).max()),
    "interpretation": "The proposed history term is invariant even though a cached query pair reverses and the true full-group objective increases. This is a feature/output-level counterexample; no claim of parameter reachability for an unseen project implementation.",
}
assert np.max(np.abs(X2@delta_p)) < 1e-15
assert abs(before-after_surrogate) < 1e-12
assert actual_after > actual_before
assert Y2[0, e1]-Y2[0, e0] < 0 < X2[0, e1]-X2[0, e0]

# Forgotten-point blind direction entirely within the tanh range.
def nonlinear_drift(x):
    return (32/3)*x**3 - (5/3)*x

anchors = np.array([-0.5, 0.5])
forgotten = np.array([-0.25, 0.25])
grid = np.linspace(-0.5, 0.5, 10001)
report["forgotten_point_counterexample"] = {
    "map": "g(x)=(32/3)*x^3-(5/3)*x; z=[x,1]; w=[1,0]",
    "cache_before": anchors.tolist(), "cache_after": nonlinear_drift(anchors).tolist(),
    "forgotten_before": forgotten.tolist(), "forgotten_after": nonlinear_drift(forgotten).tolist(),
    "max_absolute_g_on_interval": float(np.abs(nonlinear_drift(grid)).max()),
    "interpretation": "Identical cache X/Y makes ridge T=I for any epsilon, while forgotten positive/negative scores can swap. All coordinates remain strictly inside (-1,1).",
}

# Stage boundary: same covariance -> joint T equals mean T, but the trained
# expected quadratic includes a nonnegative variance penalty that is discarded.
XS = np.vstack([0.5*targets.astype(float), np.ones(378)])
YS0 = XS.copy()
YS1 = XS.copy()
YS1[0] *= -1
HS, bs, cs = statistics(XS)
TS0, _ = map_solve(XS, YS0, eps)
TS1, _ = map_solve(XS, YS1, eps)
T_end, _ = map_solve(np.concatenate([XS, XS], axis=1),
                     np.concatenate([YS0, YS1], axis=1), eps)
ws = np.array([2.0, 0.0])
vs = [TS0.T@ws, TS1.T@ws]
mean_v = np.mean(vs, axis=0)
pre = np.mean([q_value(vv, HS, bs, cs) for vv in vs])
post = q_value(T_end.T@ws, HS, bs, cs)
variance = np.mean([(vv-mean_v)@HS@(vv-mean_v) for vv in vs])
transport_H = T_end@HS@T_end.T
transport_b = T_end@bs
post_migrated = q_value(ws, transport_H, transport_b, cs)
report["stage_boundary_same_covariance"] = {
    "T_0": TS0.tolist(), "T_1": TS1.tolist(), "T_end": T_end.tolist(),
    "mean_T_minus_joint_max_error": float(np.max(np.abs((TS0+TS1)/2-T_end))),
    "expected_old_term_before_commit": float(pre),
    "joint_old_term_after_commit": float(post),
    "old_term_from_migrated_statistics": float(post_migrated),
    "mapping_variance_term": float(variance),
    "identity_decomposition_error": float(abs(pre-post-variance)),
    "state_reset_without_weight_update_changes_old_term": float(post-pre),
    "post_migration_H_last_last": float(transport_H[-1,-1]),
    "post_migration_b_last": float(transport_b[-1]),
}
assert abs(pre-post-variance) < 1e-10

# Covariance mismatch makes mean T != joint T, including disagreement in the
# opposite direction. Search a tiny fixed grid only for analytic counterexample,
# not an algorithmic hyperparameter search or experimental setting selection.
unequal = []
for scales in [(0.1,0.8), (0.2,0.6), (0.3,0.8)]:
    for ys in [(-0.8,0.1), (0.1,-0.8), (0.8,0.8), (-0.1,0.8)]:
        Xs = [np.array([[-a,a],[1.0,1.0]]) for a in scales]
        Ys = [np.array([[-bb,bb],[1.0,1.0]]) for bb in ys]
        Tjs = [map_solve(xx,yy,eps)[0] for xx,yy in zip(Xs,Ys)]
        Tendj = map_solve(np.concatenate(Xs,axis=1),np.concatenate(Ys,axis=1),eps)[0]
        # This is a valid least-squares quadratic with constant feature:
        # one pos +a, one neg -a and their query difference.
        HQ = np.diag([1.25,1.0]); bQ = np.array([2.5,0.0]); cQ = 5.0
        wj = np.array([2.0,0.0])
        pre_j = np.mean([q_value(tj.T@wj,HQ,bQ,cQ) for tj in Tjs])
        post_j = q_value(Tendj.T@wj,HQ,bQ,cQ)
        unequal.append({"reference_amplitudes":scales,"current_amplitudes":ys,
                        "T_scalars":[float(tj[0,0]) for tj in Tjs],
                        "T_joint_scalar":float(Tendj[0,0]),
                        "T_mean_scalar":float(np.mean([tj[0,0] for tj in Tjs])),
                        "before":float(pre_j),"after":float(post_j),"after_minus_before":float(post_j-pre_j)})
report["stage_boundary_unequal_covariance_examples"] = {
    "increase": max(unequal,key=lambda x:x["after_minus_before"]),
    "decrease": min(unequal,key=lambda x:x["after_minus_before"]),
}

# Two fully consistent 28-account historical group types. Repeating each type
# three times yields six cache groups without changing the means or joint map.
full_boundary_cases = []
for new_amplitudes in [(0.1, -0.8), (-0.8, 0.1)]:
    ref_amplitudes = (0.1, 0.8)
    ref_groups = [np.vstack([amp*targets, np.ones(378)]) for amp in ref_amplitudes]
    new_groups = [np.vstack([amp*targets, np.ones(378)]) for amp in new_amplitudes]
    group_statistics = [statistics(xx) for xx in ref_groups]
    Hb = sum(s[0] for s in group_statistics)
    bb = sum(s[1] for s in group_statistics)
    Tb = [map_solve(xx, yy, eps)[0] for xx, yy in zip(ref_groups, new_groups)]
    Tjoint = map_solve(np.concatenate(ref_groups, axis=1),
                       np.concatenate(new_groups, axis=1), eps)[0]
    wb = np.array([2.0, 0.0])
    pre_b = np.mean([q_value(tt.T@wb, Hb, bb, 10.0, N=2) for tt in Tb])
    post_b = q_value(Tjoint.T@wb, Hb, bb, 10.0, N=2)
    full_boundary_cases.append({
        "reference_amplitudes": ref_amplitudes,
        "current_amplitudes": new_amplitudes,
        "groups": 2, "accounts_per_group":28, "edges_per_group":378,
        "T_individual": [tt.tolist() for tt in Tb],
        "T_joint": Tjoint.tolist(),
        "T_mean_minus_joint_frobenius": float(np.linalg.norm(np.mean(Tb,axis=0)-Tjoint)),
        "expected_old_term_before_commit":float(pre_b),
        "joint_old_term_after_commit":float(post_b),
        "after_minus_before":float(post_b-pre_b),
    })
report["stage_boundary_unequal_covariance_full_groups"] = full_boundary_cases

# Invertibility of K does not imply invertibility of T. Here the current feature
# drift cancels the identity ridge in the cross-moment exactly.
XC = np.vstack([0.5*targets.astype(float), np.ones(378)])
mean_x = XC[0].mean()
var_x = XC[0].var()
YC = XC.copy()
YC[0] = -eps/var_x * (XC[0]-mean_x)
TC, KC = map_solve(XC,YC,eps)
HC,bc,cc = statistics(XC)
HC_after = TC@HC@TC.T
report["ridge_does_not_prevent_singular_transport"] = {
    "max_absolute_current_nonconstant_feature":float(np.abs(YC[0]).max()),
    "K_min_eigenvalue":float(np.linalg.eigvalsh(KC).min()),
    "T":TC.tolist(),
    "T_singular_values":np.linalg.svd(TC,compute_uv=False).tolist(),
    "H_before_eigenvalues":np.linalg.eigvalsh(HC).tolist(),
    "H_after_eigenvalues":np.linalg.eigvalsh(HC_after).tolist(),
    "interpretation":"Legal bounded features can erase a statistic direction under stage-end congruence while K remains positive definite; this is a mathematical limitation, not evidence of occurrence in project training.",
}

# Persistent numeric state raw-byte count only; full container is not supplied.
report["declared_numeric_state_raw_bytes"] = {
    "H":32*32*8,"b":32*8,"c_and_N":16,
    "references":6*378*32*4,
    "sum":32*32*8+32*8+16+6*378*32*4,
    "remaining_of_1_MiB":1048576-(32*32*8+32*8+16+6*378*32*4),
}

out_path = ROOT / "independent_math_results.json"
out_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
