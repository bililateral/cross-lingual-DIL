"""Independent attachment-only review checks; no server, BGE, labels, or training.

Only handmade numerical examples and already-open saved metric matrices are used.
PyTorch/psutil are not installed here. Production Torch updates are NOT rerun.
Selected pure functions are compiled from their unchanged source AST where a
runner's unrelated top-level psutil import would otherwise block a saved-array
check. This is explicitly not execution of the formal runner or native verifier.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
import types
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import numpy as np


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def extract_functions(path, names, namespace):
    """Execute unchanged function AST only; save exact source-line attribution."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return {n.name: [n.lineno, n.end_lineno] for n in selected}


def package_identity(root, original):
    manifest = json.loads((root / "package_manifest.json").read_text())
    mismatches = []
    for row in manifest:
        payload = (root / row["path"]).read_bytes()
        if len(payload) != row["bytes"] or digest(payload) != row["sha256"]:
            mismatches.append(row["path"])
    assert not mismatches
    frozen = original / "result/source"
    shared = []
    for path in sorted((root / "source").rglob("*")):
        if path.is_file() and (frozen / path.relative_to(root / "source")).is_file():
            prior = frozen / path.relative_to(root / "source")
            shared.append({"path": str(path.relative_to(root)), "identical": path.read_bytes() == prior.read_bytes()})
    attempts = []
    for i in range(1, 6):
        folder = root / f"evidence/cpu/attempt{i:02}"
        report = json.loads((folder / "result.json").read_text())
        log = (root / f"evidence/cpu/attempt{i:02}.console.txt").read_text()
        wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", log).group(1)
        seconds = sum(float(v) * 60**j for j, v in enumerate(reversed(wall.split(":"))))
        rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", log).group(1))
        attempts.append({"attempt": i, "status": report["status"], "tests_run": report["tests_run"],
                         "failures": report["failures"], "errors": report["errors"], "skipped": report["skipped"],
                         "wall_seconds_external": seconds, "rss_kib_external": rss,
                         "cumulative_seconds_reported": report["cumulative_seconds"]})
    final = json.loads((root / "evidence/cpu/attempt05/result.json").read_text())
    final_mismatches = []
    for row in final["source_files"]:
        payload = (root / "source" / row["path"]).read_bytes()
        if len(payload) != row["bytes"] or digest(payload) != row["sha256"]:
            final_mismatches.append(row["path"])
    assert not final_mismatches
    return {"outer_registered_payloads": len(manifest), "outer_mismatches": mismatches,
            "current_vs_frozen": shared, "cpu_attempts": attempts,
            "cpu_external_wall_sum": sum(r["wall_seconds_external"] for r in attempts),
            "cpu_external_max_rss_kib": max(r["rss_kib_external"] for r in attempts),
            "attempt05_registered_sources": len(final["source_files"]), "attempt05_source_mismatches": final_mismatches}


def independent_curve(train, valid, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    td = rng.integers(train.shape[1], size=(5000, train.shape[1]))
    vd = rng.integers(valid.shape[1], size=(5000, valid.shape[1]))
    signs = np.array([1., 1., -1.])
    samples = []
    for epoch in range(6):
        # Explicit epoch loop, separate independent populations.
        samples.append(np.mean(train[epoch][td], axis=1) - np.mean(valid[epoch][vd], axis=1))
    samples = np.stack(samples) * signs
    return {"gap": (train.mean(1) - valid.mean(1)) * signs,
            "ci": np.quantile(samples, [.025, .975], axis=1).transpose(1, 2, 0),
            "widening_ci": np.quantile(samples[-1] - samples[0], [.025, .975], axis=0).T,
            "td": td, "vd": vd}


def diagnostic_statistics(diag, record):
    rng = np.random.Generator(np.random.PCG64(314159))
    train = rng.uniform(.1, .7, (6, 48, 3))
    valid = rng.uniform(.1, .7, (6, 20, 3))
    seed = 1729
    actual = diag.curve_comparison(train, valid, seed)
    expected = independent_curve(train, valid, seed)
    errors = {
        "benefit_gap": float(np.max(np.abs(np.asarray(actual["benefit_gap_curve"]) - expected["gap"]))),
        "gap_intervals": float(np.max(np.abs(np.asarray(actual["benefit_gap_conditional_intervals"]) - expected["ci"]))),
        "widening_intervals": float(np.max(np.abs(np.asarray(actual["epoch1_to_6_gap_widening"]["conditional_95pct_interval"]) - expected["widening_ci"])))
    }
    assert max(errors.values()) < 1e-12
    same = diag.curve_comparison(train.copy(), valid.copy(), seed)
    assert same == actual
    known_t = np.tile([.4, .4, .8], (6, 48, 1)) + np.arange(6)[:, None, None] * np.array([.04, .04, -.04])
    known_v = np.tile([.4, .4, .8], (6, 20, 1)) + np.arange(6)[:, None, None] * np.array([-.02, -.02, .02])
    known = diag.curve_comparison(known_t, known_v, 19)
    assert list(known["interpretation"].values()) == ["有相应迹象"] * 3
    np.testing.assert_allclose(known["epoch1_to_6_gap_widening"]["mean"], [.3] * 3, atol=1e-12)
    owners = np.repeat(np.arange(12), [3] * 4 + [2] * 8)
    y = np.asarray([owners[i] == owners[j] for i, j in itertools.combinations(range(28), 2)], dtype=np.uint8)
    scores = 2 * y.astype(float) - 1
    g = record.data.Group("handmade", tuple(f"s{i:02}" for i in range(28)),
                          tuple(((f"r{i:02}a", "甲", "乙"), (f"r{i:02}b", "丙", "丁")) for i in range(28)), tuple(map(int, y)))
    values = diag.metrics([g], scores[None])
    np.testing.assert_allclose(values[0], [1., 1., np.log1p(np.exp(-1))], rtol=1e-12, atol=1e-12)
    extreme = record.parent.metrics.classification(y, np.full(378, 100., dtype=float))["log_loss"]
    unclipped = float(np.mean(np.logaddexp(0., 100.) - y * 100.))
    return {"nonconstant_independent_bootstrap_max_errors": errors, "same_inputs_same_method_seed_exact": same == actual,
            "known_curve": known, "known_metrics": values.tolist(),
            "empty_cache": diag.curve_comparison(train[:, :0], valid, seed),
            "log_loss_definition_boundary": {"all_logits_100_saved_metric": extreme, "unclipped_BCE": unclipped,
                                              "note": "Inherited clipped log_loss, not an implementation change or an observed formal result."}}


def projection_matrix(slots):
    pairs = list(itertools.combinations(range(len(slots)), 2))
    edges = list(itertools.combinations(range(28), 2))
    index = {edge: k for k, edge in enumerate(edges)}
    counts = np.bincount(slots, minlength=28)
    q = np.zeros((378, len(pairs)))
    for k, (i, j) in enumerate(pairs):
        a, b = sorted((int(slots[i]), int(slots[j])))
        if a != b:
            q[index[a, b], k] = 1. / (counts[a] * counts[b])
    return q


def handmade_chain_rule(record):
    """Independent NumPy encoder/head teacher-loss derivatives; NOT Torch replay."""
    owners = np.repeat(np.arange(12), [3] * 4 + [2] * 8)
    labels = tuple(int(owners[i] == owners[j]) for i, j in itertools.combinations(range(28), 2))
    g = record.data.Group("hand_geometry", tuple(f"s{i:02}" for i in range(28)),
                          tuple(((f"r{i:02}a", "甲", "乙"), (f"r{i:02}b", "丙", "丁")) for i in range(28)), labels)
    a0, a1 = record.assignment(g), record.assignment(g, 17)
    q0, q1 = projection_matrix(a0), projection_matrix(a1)
    null_index = next(i for i, (u, v) in enumerate(itertools.combinations(range(56), 2)) if a0[u] == a0[v] and a1[u] != a1[v])
    target_change = np.zeros(1540)
    target_change[null_index] = 3.
    coefficient = -.5 * q1.T @ (q1 @ target_change) / 378
    assert np.max(np.abs(q0 @ target_change)) == 0
    assert abs(coefficient[null_index] + 3. / (32 * 378)) < 1e-15
    rng = np.random.Generator(np.random.PCG64(271828))
    x = rng.normal(size=(112, 3))
    theta = np.r_[rng.normal(size=6), rng.uniform(-.1, .1, 24), [1., 1.1, 1.2], [.2, -.3, .4], [.1]]
    left, right = np.triu_indices(56, 1)
    target = rng.normal(0, .2, len(left))

    def forward(t):
        e, w, b, v, c = t[:6].reshape(3, 2), t[6:30].reshape(8, 3), t[30:33], t[33:36], t[36]
        channels = x @ e
        norms = np.linalg.norm(channels, axis=1, keepdims=True)
        normalized = channels / norms
        z = np.concatenate((normalized[:56], normalized[56:]), axis=1) / np.sqrt(2.)
        u, vv = z[left], z[right]
        features = np.concatenate((np.abs(u - vv), u * vv), axis=1)
        hidden0 = features @ w + b
        hidden = np.maximum(0., hidden0)
        table = hidden @ v + c
        return table, (e, w, b, v, channels, norms, normalized, z, u, vv, features, hidden0, hidden)

    def value_gradient(t, weights):
        table, cache = forward(t)
        residuals = [q @ (table - target) for q in (q0, q1)]
        value = sum(weight * np.mean(err**2) for weight, err in zip(weights, residuals))
        dt = sum(2 * weight * q.T @ err / 378 for weight, q, err in zip(weights, (q0, q1), residuals))
        e, w, b, v, channels, norms, normalized, z, u, vv, features, hidden0, hidden = cache
        dh = dt[:, None] * v * (hidden0 > 0)
        dw, db, dv, dc = features.T @ dh, dh.sum(0), hidden.T @ dt, dt.sum()
        df = dh @ w.T
        dz = np.zeros_like(z)
        np.add.at(dz, left, df[:, :4] * np.sign(u - vv) + df[:, 4:] * vv)
        np.add.at(dz, right, -df[:, :4] * np.sign(u - vv) + df[:, 4:] * u)
        dn = np.concatenate((dz[:, :2], dz[:, 2:]), axis=0) / np.sqrt(2.)
        dc_channels = (dn - normalized * np.sum(dn * normalized, axis=1, keepdims=True)) / norms
        de = x.T @ dc_channels
        return value, np.r_[de.ravel(), dw.ravel(), db, dv, dc]

    results, gradients = {}, {}
    for arm, weights in {"C": (.25, .25), "S": (.5, 0.), "C_plus": (.5, .25), "S_strong": (.75, 0.)}.items():
        value, analytic = value_gradient(theta, weights)
        finite = np.zeros_like(theta)
        eps = 1e-6
        for i in range(len(theta)):
            plus, minus = theta.copy(), theta.copy()
            plus[i] += eps
            minus[i] -= eps
            finite[i] = (value_gradient(plus, weights)[0] - value_gradient(minus, weights)[0]) / (2 * eps)
        gradients[arm] = analytic
        err = float(np.max(np.abs(analytic - finite)))
        assert err < 1e-8
        # One independently computed AdamW step, with explicitly handmade moments.
        moment = np.sin(np.arange(len(theta))) * .01
        variance = np.full(len(theta), .02)
        rates = np.r_[np.full(6, 1e-5 / 29), np.full(31, .001)]
        decay = np.r_[np.full(6, .01), np.zeros(31)]
        def adam(g):
            g = g * min(1., 1. / (np.linalg.norm(g) + 1e-6))
            m = .9 * moment + .1 * g
            vv = .999 * variance + .001 * g*g
            return theta * (1. - rates * decay) - rates * (m/(1-.9**289)) / (np.sqrt(vv/(1-.999**289)) + 1e-8)
        adam_error = float(np.max(np.abs(adam(analytic) - adam(finite))))
        results[arm] = {"teacher_loss": float(value), "encoder_gradient_norm": float(np.linalg.norm(analytic[:6])),
                        "head_gradient_norm": float(np.linalg.norm(analytic[6:])), "finite_difference_max_error": err,
                        "handmade_AdamW_step_max_error": adam_error}
    d0 = value_gradient(theta, (1., 0.))[1]
    d1 = value_gradient(theta, (0., 1.))[1]
    identities = {"C_plus_minus_S_equals_quarter_D1": float(np.max(np.abs(gradients["C_plus"] - gradients["S"] - .25*d1))),
                  "C_plus_minus_C_equals_quarter_D0": float(np.max(np.abs(gradients["C_plus"] - gradients["C"] - .25*d0))),
                  "C_plus_minus_S_strong_equals_quarter_D1_minus_D0": float(np.max(np.abs(gradients["C_plus"] - gradients["S_strong"] - .25*(d1-d0))))}
    assert max(identities.values()) < 1e-12
    return {"scope": "Independent differentiable handwritten NumPy network and teacher terms; no production Torch update or formal model.",
            "nullspace": {"table_index": null_index, "A0_effect": 0., "quarter_D1_gradient_at_index": float(coefficient[null_index]),
                          "expected": -3. / (32*378)}, "arms": results, "parameter_gradient_identities": identities}


def resource_counts(namespace, verifier):
    specs = namespace["specifications"]()
    result = {}
    for family in ("record", "LOGIT0.1"):
        entries = [v for v in specs.values() if (v[1] == "LOGIT0.1") == (family == "LOGIT0.1")]
        stages = len(entries)
        later = sum(s > 1 for _, _, s in entries)
        epochs = stages * 6
        fit_cache_metrics = stages*6*48 + later*6*6
        epoch_score_groups = fit_cache_metrics + epochs*60
        extra_tables = later*6*6 if family == "record" else 0
        stage_restore_scores = stages*2*(12+60)
        source_upper = sum(s < 3 for _, _, s in entries)*6
        result[family] = {"physical_stages": stages, "updates": stages*288,
                          "gradient_groups": sum(288 if s == 1 else 576 for _, _, s in entries),
                          "epochs": epochs, "epoch_score_groups": epoch_score_groups, "extra_cache_tables": extra_tables,
                          "checkpoint_and_restored_score_groups": stage_restore_scores, "new_source_upper_bound": source_upper,
                          "all_forward_only_groups": epoch_score_groups+extra_tables+stage_restore_scores+source_upper,
                          "during_training_CPU_metric_groups": fit_cache_metrics}
    assert result["record"]["all_forward_only_groups"] == 23202
    assert result["LOGIT0.1"]["all_forward_only_groups"] == 7380
    tree = ast.parse(verifier.read_text())
    native = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "native")
    calls = sorted({ast.unparse(n.func) for n in ast.walk(native) if isinstance(n, ast.Call)})
    return {"families": result, "physical_points": len(specs), "logical_points": 5*3*3,
            "logical_epoch_points": 5*3*3*6, "total_physical_epochs": 216,
            "metric_group_calls_all_phases": 11448 + 216*60 + 36*3*60,
            "calibration_fits": 36, "full_state_saves": 36, "inference_state_saves": 36,
            "full_state_restores_including_branches": 36+12, "inference_restores": 36,
            "native_function_calls": calls,
            "native_does_call_complete_capture": "diag.capture" in calls,
            "native_does_measure_metrics": "diag.metrics" in calls,
            "native_does_measure_full_save_restore": any(v in calls for v in ("core.save_state", "core.restore_state")),
            "unmeasured_shared_allowance_seconds": 3600,
            "interpretation": "Counts are exact/upper bounds as labeled. No native runtime measured here. A fixed allowance is not a measured cost or a guaranteed upper bound."}


def saved_history(root, original, namespace, evaluation, old_namespace, parent):
    col = json.loads((original / "result/job/evaluation/collected.json").read_text())
    ref = json.loads((original / "reference/collected.json").read_text())
    first = json.loads((original / "reference/reference/collected.json").read_text())
    arrays = {arm: {} for arm in ("C", "S", "LOGIT0.1")}
    records_read = {}
    for arm in arrays:
        for order in parent.ORDERS:
            for stage in (1, 2, 3):
                if arm == "LOGIT0.1":
                    base, registry, name = ((original / "reference/reference", first, order+"_shared") if stage == 1 else
                                            (original / "reference", ref, f"{order}_logit_tenth_stage{stage}"))
                else:
                    base, registry, name = original / "result/job/evaluation", col, order+"_shared" if stage == 1 else f"{order}_{arm}_stage{stage}"
                roles = {}
                for role in parent.ROLES:
                    rec = registry["points"][name][role]["matrix"]
                    path = base / rec["path"]
                    payload = path.read_bytes()
                    assert len(payload) == rec["bytes"] and digest(payload) == rec["sha256"]
                    roles[role] = np.load(io.BytesIO(payload), allow_pickle=False)
                    records_read[str(path.relative_to(original))] = rec
                arrays[arm][evaluation.point_name(order, "er", stage)] = roles
    p = copy.deepcopy(namespace["policy"]())
    p["historical_results"]["path"] = str(root.parent / "review_background/common/background/original_result_input.zip")
    old_policy = namespace["policy"]
    namespace["policy"] = lambda: p  # File-location adaptation only, exact bytes/SHA retained.
    try:
        result = namespace["historical_replay"](col, arrays)
    finally:
        namespace["policy"] = old_policy
    entries = [v for roles in result["points"].values() for v in roles.values()]
    assert len(entries) == 81 and all(v["bitwise_equal"] for v in entries)
    assert max(v["maximum_absolute_group_difference"] for v in entries) == 0
    # Same saved arrays under new names test historical lookup, not a new run.
    draws = evaluation.bootstrap_draws()
    field_c = old_namespace["fields"](arrays["C"], col["domains"], "primary", "O")
    field_s = old_namespace["fields"](arrays["S"], col["domains"], "primary", "O")
    actual = evaluation.summarize_field(field_c - field_s, draws)["map"]
    # Independently accumulate the two final old-domain group differences for each order.
    domain_arrays = [np.zeros(20) for _ in range(3)]
    k = parent.metrics.COLUMNS.index("map")
    per_order = {}
    for order in parent.ORDERS:
        key = evaluation.point_name(order, "er", 3)
        delta = arrays["C"][key]["raw"][:, k] - arrays["S"][key]["raw"][:, k]
        observed = 0.
        for d in order[:2]:
            values = delta[np.asarray(col["domains"]) == d]
            domain_arrays["ABC".index(d)] += values / 6
            observed += values.mean()/2
        per_order[order] = observed
    boot = sum(v[draws[:, d]].mean(1) for d, v in enumerate(domain_arrays))
    expected = {"mean": sum(v.mean() for v in domain_arrays),
                "ci": np.quantile(boot, [.025, .975]).tolist(), "per_order": per_order}
    error = max(abs(actual["mean"]-expected["mean"]),
                float(np.max(np.abs(np.asarray(actual["conditional_95pct_interval"])-expected["ci"]))))
    assert error < 1e-12
    return {"unique_saved_metric_matrices_read": len(records_read), "historical_lookup_role_points": len(entries),
            "all_same_input_comparisons_exact_zero": True, "read_matrices": records_read,
            "C_minus_S_old_MAP_actual_paired_summary": actual, "independent_expected": expected,
            "independent_paired_summary_max_error": error,
            "scope": "Only opened group metric matrices. No old raw labels, weights, or Memory. Not new five-arm performance."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"status": "RUNNING", "python": sys.version, "numpy": np.__version__,
              "torch_available": importlib.util.find_spec("torch") is not None,
              "psutil_available": importlib.util.find_spec("psutil") is not None,
              "formal_data_or_labels_or_models": False, "project_server_access": False, "gpu": False,
              "production_Torch_updates_executed": 0}
    try:
        sys.path.insert(0, str(args.root / "source/scripts"))
        import step28_record_replay as record
        import step28_replay_diagnostics as diag
        import step28_bge_continual_evaluate as evaluation
        import step28_bge_continual_run as previous
        parent, data = record.parent, record.data
        old_namespace = dict(np=np, parent=parent, evaluation=evaluation)
        old_lines = extract_functions(args.root / "source/scripts/step28_record_replay_run.py", ["fields"], old_namespace)
        namespace = dict(np=np, parent=parent, data=data, json=json, hashlib=hashlib, io=io, zipfile=zipfile,
                         POLICY=args.root/"source/schema/step28_replay_improvement_policy.json",
                         ARMS=("LOGIT0.1", "C", "S", "C_plus", "S_strong"), evaluation=evaluation)
        runner_lines = extract_functions(args.root / "source/scripts/step28_replay_improvement_run.py",
                                         ["policy", "point_name", "specifications", "historical_replay"], namespace)
        status["AST_selected_unmodified_functions"] = {"old_runner": old_lines, "new_runner": runner_lines}
        tests = [
            ("identity", lambda: package_identity(args.root, args.background/"original")),
            ("diagnostic_statistics", lambda: diagnostic_statistics(diag, record)),
            ("handmade_chain_rule", lambda: handmade_chain_rule(record)),
            ("resource_counts", lambda: resource_counts(namespace, args.root/"source/scripts/step28_replay_improvement_verify.py")),
            ("saved_history", lambda: saved_history(args.root, args.background/"original", namespace, evaluation, old_namespace, parent)),
        ]
        for name, function in tests:
            tick = time.perf_counter()
            result = function()
            dump(args.out / (name+".json"), result)
            print(json.dumps({"check": name, "status": "PASS", "seconds": time.perf_counter()-tick}, ensure_ascii=False), flush=True)
        zero = {ep: {name: {"mean": 0., "conditional_95pct_interval": [0., 0.],
                            "per_order": {o: 0. for o in parent.ORDERS}} for name in parent.metrics.COLUMNS}
                for ep in evaluation.ENDPOINTS}
        equal = evaluation.comparison_checks(zero, zero)
        assert len(equal["checks"]) == 23 and sum(equal["checks"].values()) == 20
        assert equal["pilot_observed_checks_pass"] is False
        dump(args.out / "original_23_equal_example.json", equal)
        status["status"] = "PASS_ATTACHMENT_ONLY_CHECKS"
    except BaseException:
        status["status"] = "FAIL"
        status["traceback"] = traceback.format_exc()
        raise
    finally:
        status["seconds"] = time.perf_counter()-started
        dump(args.out / "execution.json", status)


if __name__ == "__main__":
    main()
